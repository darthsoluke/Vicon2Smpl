"""Exercise full web-worker orchestration with synthetic files and mocked fits."""
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


class FullWebWorkerTests(unittest.TestCase):
    def test_two_stage_worker_keeps_raw72_and_saves_native55_hand_metadata(self):
        rng = np.random.default_rng(304)
        body = [f"BODY_{i:02d}" for i in range(52)]
        fingers = [f"FINGER_{i:02d}" for i in range(20)]
        names = np.array(body + fingers)[rng.permutation(72)].tolist()
        rec = dict(names=names, pos=rng.normal(size=(4, 72, 3)), valid=np.ones((4, 72), bool), fps=300.)
        before_pos = rec["pos"].copy()
        region_map = {name: [i, 100 + i] for i, name in enumerate(body)}
        body_names = [name for name in names if name in region_map]
        model = SimpleNamespace(v_template=np.zeros((256, 3)), num_joints=55)
        prior = SimpleNamespace(n_components=12)
        source_frames = np.array([500, 503, 509, 511])
        source_times = np.array([0., .01, .03, .036])
        result_body = dict(marker_names=np.array(body_names), vertex_ids=np.arange(52),
                           betas=np.full(10, .3), offsets=np.zeros((52, 3)), marker_radius=.0095,
                           per_marker_rms=np.full(52, .004), static_pose="a")
        native_rotations = np.broadcast_to(np.eye(3), (2, 55, 3, 3)).copy()
        hand_result = dict(marker_names=np.array(names), vertex_ids=np.arange(72),
                           betas=result_body["betas"], offsets=np.zeros((72, 3)), marker_radius=.0095,
                           per_marker_rms=np.full(72, .005), static_pose="a", marker_scope="body_hands",
                           hand_pca_components=12, hand_pca_initial_latent=np.full((2, 12), .2),
                           hand_prior_weight=1e-5, finger_marker_names=np.array(fingers),
                           selected_hand_R=native_rotations, selected_hand_trans=np.zeros((2, 3)),
                           selected_hand_frames=np.array([1, 3]), selected_hand_latent=np.zeros((2, 2, 12)))
        stage_order = []
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            sampled = directory / "calibration_markers.npz"
            np.savez(sampled, names=names, pos=rec["pos"], valid=rec["valid"], fps=300.)
            source_bytes = sampled.read_bytes()
            region_file = directory / "marker_regions.json"
            region_file.write_text(json.dumps(region_map))
            prior_path = directory / "synthetic_prior.npz"
            prior_path.write_bytes(b"temporary synthetic hand prior stand-in")
            prior_sha = hashlib.sha256(prior_path.read_bytes()).hexdigest()
            np.savez(directory / "sampling_clock.npz", source_frame_numbers=source_frames,
                     relative_host_arrival_seconds=source_times)
            request = dict(tpose=[0, 3], static_pose="a", source_subject="zhr",
                           body_regions_file=region_file.name, fit_hands=True,
                           hand_prior_source=str(prior_path), hand_prior_sha256=prior_sha,
                           hand_pca_components=12)
            (directory / "calibration_request.json").write_text(json.dumps(request))
            app = run_web.FieldApp.__new__(run_web.FieldApp)
            app.args = SimpleNamespace(model_path="new-subject-native-model.npz", model_type="smplx", gender="neutral")
            app.run, app.lock, app.selected = directory, threading.RLock(), "zhr"
            app.sample_static_pose, app.mode, app.message = "a", "calibrating", ""
            for name in ("fit_button", "capture", "subject", "pose_kind", "track_button"):
                setattr(app, name, SimpleNamespace(disabled=True))
            app._load_calibration, app._receipt = Mock(), Mock()

            def fit_body(actual_model, body_rec, **kwargs):
                stage_order.append("body")
                self.assertIs(actual_model, model)
                self.assertEqual(body_rec["names"], body_names)
                self.assertEqual(body_rec["pos"].shape, (4, 52, 3))
                self.assertEqual(set(kwargs["marker_regions"]), set(body))
                self.assertNotIn("betas", kwargs)
                return result_body

            def fit_hands(actual_model, full_rec, body_calib, actual_prior, actual_regions, **kwargs):
                stage_order.append("hands")
                self.assertIs(actual_model, model)
                self.assertIs(full_rec, rec)
                self.assertIs(body_calib, result_body)
                self.assertIs(actual_prior, prior)
                self.assertEqual(full_rec["names"], names)
                self.assertEqual(full_rec["pos"].shape, (4, 72, 3))
                self.assertEqual(set(actual_regions), set(body))
                np.testing.assert_array_equal(kwargs["frame_numbers"], source_frames)
                np.testing.assert_array_equal(kwargs["times"], source_times)
                self.assertTrue((directory / "body_calib.npz").exists())
                return hand_result, dict(acceptance="PENDING_VISUAL_REVIEW", selected_frames=[1, 3],
                                         marker_rms_mm=np.float64(5.))

            original_run = subprocess.run
            worker_calls = []
            def run_inline(command, *positional, **kwargs):
                if not isinstance(command, list) or command[1:3] != ["-u", "-c"]:
                    return original_run(command, *positional, **kwargs)
                worker_calls.append(command)
                with patch.object(sys, "argv", ["-c", *command[4:]]):
                    exec(compile(command[3], "<full-web-worker>", "exec"), {"__name__": "__main__"})
                return subprocess.CompletedProcess(command, 0)

            with patch("vsmpl.body_model.BodyModel.load", return_value=model), \
                    patch("vsmpl.sources.load_recording", return_value=rec), \
                    patch("vsmpl.calibration.calibrate", side_effect=fit_body) as fit, \
                    patch("vsmpl.hand_pca.HandPCA.load", return_value=prior) as load_prior, \
                    patch("vsmpl.full_calibration.calibrate_hands", side_effect=fit_hands) as hands, \
                    patch("subprocess.run", side_effect=run_inline):
                app._calibrate()
            self.assertEqual(app.mode, "calibrated", app.message)
            self.assertEqual(stage_order, ["body", "hands"])
            self.assertEqual(len(worker_calls), 1)
            fit.assert_called_once()
            hands.assert_called_once()
            load_prior.assert_called_once_with(prior_path.resolve(), n_components=12)
            with np.load(directory / "subject_calib.npz", allow_pickle=False) as saved:
                self.assertEqual(str(saved["source_subject"]), "zhr")
                self.assertEqual(str(saved["marker_scope"]), "body_hands")
                self.assertEqual(str(saved["hand_prior_source"]), str(prior_path.resolve()))
                self.assertEqual(str(saved["hand_prior_sha256"]), prior_sha)
                self.assertEqual(saved["excluded_marker_names"].size, 0)
                self.assertEqual(saved["marker_names"].size, 72)
                self.assertEqual(saved["selected_hand_R"].shape, (2, 55, 3, 3))
                self.assertEqual(saved["hand_pca_initial_latent"].shape, (2, 12))
                self.assertEqual(set(saved["finger_marker_names"].tolist()), set(fingers))
            with np.load(directory / "body_calib.npz", allow_pickle=False) as saved_body:
                self.assertEqual(str(saved_body["source_subject"]), "zhr")
                self.assertEqual(saved_body["marker_names"].size, 52)
                self.assertEqual(str(saved_body["marker_scope"]), "body_wrist_only")
            report = json.loads((directory / "hand_calibration_report.json").read_text())
            self.assertEqual(report["selected_frames"], [1, 3])
            self.assertEqual(sampled.read_bytes(), source_bytes)
            np.testing.assert_array_equal(rec["pos"], before_pos)

    def test_full_checkpoint_loads_pca_solver_only_with_matching_prior_sha(self):
        body = [f"BODY_{i:02d}" for i in range(52)]
        fingers = [f"FINGER_{i:02d}" for i in range(20)]
        names = body + fingers
        latent = np.zeros((2, 12))
        rig = SimpleNamespace(num_markers=72, names=names)
        prior = SimpleNamespace(n_components=12)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            prior_file = directory / "prior.npz"
            prior_file.write_bytes(b"synthetic prior bytes")
            path = directory / "full_calib.npz"
            np.savez(path, marker_scope="body_hands", per_marker_rms=np.full(72, .005),
                     hand_prior_source=str(prior_file), hand_prior_sha256=hashlib.sha256(prior_file.read_bytes()).hexdigest(),
                     hand_pca_components=12, hand_pca_initial_latent=latent, hand_prior_weight=1e-5,
                     finger_marker_names=np.array(fingers))
            app = run_web.FieldApp.__new__(run_web.FieldApp)
            app.model, app.args = object(), SimpleNamespace(hand_prior=None)
            app.detail = SimpleNamespace(content="")
            with patch.object(run_web.MarkerRig, "from_calibration", return_value=rig), \
                    patch.object(run_web.HandPCA, "load", return_value=prior) as load_prior, \
                    patch.object(run_web, "PCASolver") as pca_solver, \
                    patch.object(run_web, "RealtimeSolver") as body_solver:
                app._load_calibration(path)
                load_prior.assert_called_once_with(prior_file.resolve(), n_components=12)
                pca_solver.assert_called_once()
                call = pca_solver.call_args
                self.assertEqual(call.args, (rig, prior))
                np.testing.assert_array_equal(call.kwargs["initial_latent"], latent)
                self.assertEqual(call.kwargs["hand_prior_weight"], 1e-5)
                body_solver.assert_not_called()
                self.assertEqual(app.finger_marker_mask.sum(), 20)
                self.assertIn("身体＋双手 72 点", app.detail.content)
            prior_file.write_bytes(b"changed prior bytes")
            with patch.object(run_web.MarkerRig, "from_calibration", return_value=rig), \
                    patch.object(run_web, "PCASolver") as pca_solver:
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    app._load_calibration(path)
                pca_solver.assert_not_called()

    def test_visible_group_rms_does_not_treat_missing_fingers_as_zero_error(self):
        app = run_web.FieldApp.__new__(run_web.FieldApp)
        app.finger_marker_mask = np.array([False, False, True, True])
        app.last_result = SimpleNamespace(R=np.empty((55, 3, 3)), trans=np.zeros(3))
        app.rig = SimpleNamespace(forward=lambda R, t: np.zeros((4, 3)))
        app.last_observation = (np.array([[.003, .004, 0], [0, 0, .005], [0, 0, .01], [0, 0, .02]]),
                                np.array([True, True, False, False]))
        groups = app._group_marker_rms()
        self.assertAlmostEqual(groups["body_mm"], 5.)
        self.assertIsNone(groups["finger_mm"])


if __name__ == "__main__":
    unittest.main()
