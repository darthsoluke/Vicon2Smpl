"""Exercise the actual web worker script without hardware or numerical optimisation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_web


class BodyCalibrationWorkerTests(unittest.TestCase):
    def test_worker_uses_frozen_body_regions_and_fits_the_new_subject(self):
        body = [f"BODY_{i:02d}" for i in range(52)]
        fingers = [f"FINGER_{i:02d}" for i in range(20)]
        names = np.array(body + fingers)[np.random.default_rng(3).permutation(72)].tolist()
        rec = dict(names=names,
                   pos=np.arange(4 * 72 * 3, dtype=np.float64).reshape(4, 72, 3) / 1000,
                   valid=(np.arange(4)[:, None] + np.arange(72)[None]) % 3 != 0,
                   fps=300.0, frames=np.array([100, 103, 107, 110]))
        original_pos, original_valid = rec["pos"].copy(), rec["valid"].copy()
        original_frames = rec["frames"].copy()
        frozen_regions = {name: [i, i + 100] for i, name in enumerate(body)}
        expected_names = [name for name in names if name in frozen_regions]
        expected_indices = [names.index(name) for name in expected_names]
        expected_excluded = [name for name in names if name not in frozen_regions]
        model = SimpleNamespace(v_template=np.zeros((256, 3)))

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            sampled = directory / "calibration_markers.npz"
            np.savez(sampled, names=names, pos=rec["pos"], valid=rec["valid"], fps=rec["fps"],
                     frames=rec["frames"])
            source_bytes = sampled.read_bytes()
            region_path = directory / "marker_regions.json"
            region_path.write_text(json.dumps(frozen_regions, indent=2))
            region_sha = hashlib.sha256(region_path.read_bytes()).hexdigest()
            request_path = directory / "calibration_request.json"
            request_path.write_text(json.dumps(dict(tpose=[0, 3], static_pose="a", source_subject="zhr",
                                                    body_regions_file=region_path.name)))
            # This live configuration differs from the frozen per-run snapshot on purpose.
            live_regions_path = directory / "changed_live_regions.json"
            live_regions_path.write_text(json.dumps({"FINGER_00": [255]}))
            old_path = directory / "old_st_calibration.npz"
            np.savez(old_path, source_subject="st", betas=np.full(10, 99.),
                     offsets=np.full((52, 3), 99.), vertex_ids=np.full(52, 255))
            old_bytes = old_path.read_bytes()

            app = run_web.FieldApp.__new__(run_web.FieldApp)
            app.args = SimpleNamespace(model_path="fresh-native-model.npz", model_type="smplx",
                                       gender="neutral", body_regions=str(live_regions_path), calib=str(old_path))
            app.run = directory
            app.lock = threading.RLock()
            app.selected = "zhr"
            app.sample_static_pose = "a"
            app.mode = "calibrating"
            app.message = ""
            app.calibration_path = old_path
            app.rig = SimpleNamespace(betas=np.full(10, 99.), offsets=np.full((52, 3), 99.))
            for name in ("fit_button", "capture", "subject", "pose_kind", "track_button"):
                setattr(app, name, SimpleNamespace(disabled=True))
            app._load_calibration = Mock()
            app._receipt = Mock()

            def fit_fresh(actual_model, actual_rec, **kwargs):
                self.assertIs(actual_model, model)
                self.assertEqual(actual_rec["names"], expected_names)
                self.assertEqual(actual_rec["pos"].shape, (4, 52, 3))
                np.testing.assert_array_equal(actual_rec["pos"], original_pos[:, expected_indices])
                np.testing.assert_array_equal(actual_rec["valid"], original_valid[:, expected_indices])
                np.testing.assert_array_equal(actual_rec["frames"], original_frames)
                self.assertEqual(actual_rec["fps"], 300.0)
                self.assertEqual(kwargs["marker_regions"],
                                 {name: frozen_regions[name] for name in expected_names})
                self.assertEqual(kwargs["tpose"], (0, 3))
                self.assertEqual(kwargs["static_pose"], "a")
                # Anatomical constraints may be reused; the old person's fitted parameters may not.
                for old_parameter in ("betas", "offsets", "vertex_ids", "marker_map", "initial_params"):
                    self.assertNotIn(old_parameter, kwargs)
                return dict(marker_names=np.array(actual_rec["names"]), vertex_ids=np.arange(52),
                            betas=np.full(10, .25), offsets=np.zeros((52, 3)),
                            per_marker_rms=np.full(52, .004), static_pose="a")

            original_run = subprocess.run
            worker_calls = []

            def run_inline(command, *positional, **kwargs):
                if not isinstance(command, list) or command[1:3] != ["-u", "-c"]:
                    return original_run(command, *positional, **kwargs)
                worker_calls.append(command)
                self.assertTrue(kwargs["check"])
                # Execute the script that production passes to its subprocess, including imports and save.
                with patch.object(sys, "argv", ["-c", *command[4:]]):
                    exec(compile(command[3], "<web-calibration-worker>", "exec"), {"__name__": "__main__"})
                return subprocess.CompletedProcess(command, 0)

            with patch("vsmpl.body_model.BodyModel.load", return_value=model) as load_model, \
                    patch("vsmpl.sources.load_recording", return_value=rec) as load_rec, \
                    patch("vsmpl.calibration.calibrate", side_effect=fit_fresh) as fit, \
                    patch("subprocess.run", side_effect=run_inline):
                app._calibrate()

            self.assertEqual(app.mode, "calibrated", app.message)
            fit.assert_called_once()
            self.assertEqual(len(worker_calls), 1)
            load_model.assert_called_once_with("fresh-native-model.npz", "smplx", "neutral")
            load_rec.assert_called_once_with(str(sampled))
            output = directory / "subject_calib.npz"
            app._load_calibration.assert_called_once_with(output)
            app._receipt.assert_called_once_with("calibrated", calibration_file=output.name)
            self.assertFalse(app.track_button.disabled)
            with np.load(output, allow_pickle=False) as saved:
                self.assertEqual(str(saved["source_subject"]), "zhr")
                self.assertEqual(str(saved["marker_scope"]), "body_wrist_only")
                self.assertEqual(str(saved["marker_regions_sha256"]), region_sha)
                self.assertEqual(saved["excluded_marker_names"].tolist(), expected_excluded)
                self.assertEqual(saved["marker_names"].tolist(), expected_names)
                np.testing.assert_array_equal(saved["betas"], np.full(10, .25))
                np.testing.assert_array_equal(saved["offsets"], np.zeros((52, 3)))
            self.assertEqual(sampled.read_bytes(), source_bytes)
            self.assertEqual(old_path.read_bytes(), old_bytes)
            self.assertEqual(rec["names"], names)
            np.testing.assert_array_equal(rec["pos"], original_pos)
            np.testing.assert_array_equal(rec["valid"], original_valid)
            np.testing.assert_array_equal(rec["frames"], original_frames)


if __name__ == "__main__":
    unittest.main()
