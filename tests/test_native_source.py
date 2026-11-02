"""Contract checks for the Linux read-only bridge adapter."""
import sys
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vsmpl.native_source import subject_frame


class NativeFrameTests(unittest.TestCase):
    def test_native_npz_geometry_matches_existing_model(self):
        from tests.synthetic import make_body
        from vsmpl.body_model import BodyModel
        from vsmpl.rotations import exp_so3
        model, _ = make_body()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.npz"
            np.savez(path, v_template=model.v_template, shapedirs=model.shapedirs,
                     posedirs=model.posedirs.transpose(1, 2, 0), J_regressor=model.J_regressor,
                     weights=model.weights, kintree_table=np.stack([model.parents, np.arange(24)]),
                     f=model.faces)
            loaded = BodyModel.load(path, "smpl")
        R = exp_so3(np.random.default_rng(1).normal(size=(24, 3)) * .2)
        betas = np.zeros(10)
        np.testing.assert_allclose(loaded.vertices(R, np.zeros(3), *loaded.shaped(betas)),
                                   model.vertices(R, np.zeros(3), *model.shaped(betas)))

    def test_subjects_units_axes_and_occlusion(self):
        packet = {"frame": 42, "arrival_time": 2.5, "subjects": [
            {"name": "person", "markers": [
                {"name": "person:head", "xyz": [1000, 2000, 3000], "valid": True},
                {"name": "hidden", "xyz": None, "valid": False}]},
            {"name": "prop", "markers": [
                {"name": "head", "xyz": [9000, 9000, 9000], "valid": True}]}]}
        fr = subject_frame(packet, "person")
        self.assertEqual(fr["names"], ["head", "hidden"])
        np.testing.assert_allclose(fr["pos"][0], [1, 3, -2])
        self.assertTrue(np.isnan(fr["pos"][1]).all())
        self.assertEqual(fr["valid"].tolist(), [True, False])
        self.assertEqual(fr["frame"], 42)
        self.assertEqual(fr["time"], 2.5)
        self.assertIsNone(subject_frame(packet, "absent"))

    def test_duplicate_labels_rejected(self):
        marker = {"name": "head", "xyz": [0, 0, 1000], "valid": True}
        packet = {"frame": 1, "arrival_time": 0, "subjects": [
            {"name": "person", "markers": [marker, {**marker, "name": "person:head"}]}]}
        with self.assertRaises(ValueError):
            subject_frame(packet, "person")

    def test_operator_sampling_and_derived_output(self):
        import run_web
        from tests.synthetic import make_body, pick_markers, make_motion, simulate
        model, _ = make_body()
        rotations, trans = make_motion(120)
        obs, valid, rig = simulate(model, pick_markers(model), np.zeros(10), rotations, trans,
                                    dropout=0, swap=False)
        import time
        packet = dict(frame=0, arrival_time=time.perf_counter(), subjects=[dict(name="test", markers=[
            dict(name=n, valid=True, xyz=[100, 200, 1000]) for n in rig.names])])

        class FakeSource:
            frame_rate = 10
            def __init__(self, *args): pass
            def snapshot(self): return 0, packet
            def close(self): pass

        with tempfile.TemporaryDirectory() as folder, patch.object(run_web.BodyModel, "load", return_value=model), \
                patch.object(run_web, "NativeSource", FakeSource):
            args = SimpleNamespace(output=folder, model_path="unused", model_type="smpl", gender="neutral",
                                   native_reader="unused", source="test-only", bind="127.0.0.1", port=0)
            app = run_web.FieldApp(args)
            try:
                self.assertEqual(app.mode, "preview")
                self.assertIsNone(app.recorder)
                app.selected = "test"
                app.start_sampling(None)
                self.assertEqual(app.mode, "countdown")
                app.samples = [dict(frame=200 + 3*i, time=100+i/10, names=rig.names,
                                    pos=obs[i], valid=valid[i]) for i in range(len(obs))]
                app.mode = "sampling"
                app.finish_sampling(None)
                saved = np.load(app.run / "calibration_markers.npz")
                np.testing.assert_array_equal(saved["frames"], 200 + 3*np.arange(120))
                self.assertFalse(app.fit_button.disabled)
                path = app.run / "oracle_calib.npz"
                np.savez(path, marker_names=rig.names, vertex_ids=rig.vertex_ids, offsets=rig.offsets,
                         betas=rig.betas, per_marker_rms=np.zeros(rig.num_markers))
                app._load_calibration(path)
                app.mode = "calibrated"
                app.start_tracking(None)
                app.recorder.add(500, np.zeros((24, 3)), np.zeros(3), .001)
                app.stop_tracking(None)
                receipt = __import__("json").loads((app.run / "session.json").read_text())
                self.assertEqual(receipt["solved_frames"], 1)
                self.assertFalse(receipt["raw_capture_modified"])
                output = np.load(app.run / receipt["last_solved_file"])
                self.assertEqual(output["frame_numbers"].tolist(), [500])
            finally:
                app.server.stop()


if __name__ == "__main__":
    unittest.main()
