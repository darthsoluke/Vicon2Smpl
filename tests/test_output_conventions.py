"""Recorder contract regressions without SOMA assets or motion-capture hardware."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vsmpl.outputs import SequenceRecorder


LEGACY_FIELDS = {"poses", "root_orient", "trans", "frame_numbers", "rmse", "model_type", "gender",
                 "betas", "mocap_framerate"}


class StubSoma:
    model_type = "soma"
    hips = np.array([.1, .9, -.2])

    def public_pose(self, poses, trans):
        # A visible convention change, not a substitute for SOMA FK parity tests.
        return np.asarray(poses).reshape(77, 3) + .125, np.asarray(trans) + self.hips

    def recording_metadata(self, betas):
        identity = np.zeros(128)
        identity[:len(betas)] = betas
        return dict(identity_coeffs=identity, identity_model_type="soma", soma_lod="low",
                    soma_joint_names=np.array([f"joint_{i}" for i in range(77)]),
                    pose_convention="77 public T-relative rotations; trans is Hips position in metres")


class OutputConventionTests(unittest.TestCase):
    def save_and_load(self, recorder):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sequence.npz"
            count = recorder.save(path)
            with np.load(path, allow_pickle=False) as archive:
                saved = {key: archive[key].copy() for key in archive.files}
        return count, saved

    def assert_legacy(self, recorder, poses, trans):
        self.assertIsNone(recorder.model)
        self.assertFalse(hasattr(recorder, "solver_poses"))
        self.assertFalse(hasattr(recorder, "solver_trans"))
        recorder.add(42, poses, trans, .002)
        count, saved = self.save_and_load(recorder)
        self.assertEqual(count, 1)
        self.assertEqual(set(saved), LEGACY_FIELDS)
        np.testing.assert_array_equal(saved["poses"], np.asarray(poses, np.float32).reshape(1, -1))
        np.testing.assert_array_equal(saved["root_orient"], saved["poses"][:, :3])
        np.testing.assert_array_equal(saved["trans"], np.asarray(trans, np.float32)[None])
        self.assertEqual(saved["frame_numbers"].tolist(), [42])

    def test_no_model_keeps_original_recording_contract(self):
        poses = np.arange(165).reshape(55, 3) * .001
        self.assert_legacy(SequenceRecorder("smplx", "neutral", np.zeros(10), 300), poses, np.array([1.,2.,3.]))

    def test_smpl_family_does_not_call_noop_hooks_or_duplicate_frames(self):
        for backend, joints in (("smpl", 24), ("smplh", 52), ("smplx", 55)):
            with self.subTest(backend=backend):
                model = SimpleNamespace(model_type=backend,
                    public_pose=Mock(side_effect=AssertionError("ordinary poses need no conversion")),
                    recording_metadata=Mock(side_effect=AssertionError("legacy metadata must stay unchanged")))
                recorder = SequenceRecorder(backend, "neutral", np.zeros(10), 300, model=model)
                self.assert_legacy(recorder, np.zeros((joints,3)), np.array([.1,.2,.3]))
                model.public_pose.assert_not_called()
                model.recording_metadata.assert_not_called()

    def test_soma_keeps_77_joint_public_and_solver_conventions_with_identity(self):
        model = StubSoma()
        betas = np.arange(10) * .01
        recorder = SequenceRecorder("soma", "neutral", betas, 300, model=model)
        solver_pose = np.arange(231).reshape(77,3) * .001
        solver_trans = np.array([.3,.4,.5])
        recorder.add(501, solver_pose, solver_trans, .003)
        count, saved = self.save_and_load(recorder)
        self.assertEqual(count, 1)
        self.assertEqual(saved["poses"].shape, (1, 77*3))
        self.assertEqual(saved["solver_poses"].shape, (1, 77*3))
        np.testing.assert_array_equal(saved["poses"], (solver_pose+.125).astype(np.float32).reshape(1,-1))
        np.testing.assert_array_equal(saved["solver_poses"], solver_pose.astype(np.float32).reshape(1,-1))
        np.testing.assert_array_equal(saved["trans"], (solver_trans+model.hips).astype(np.float32)[None])
        np.testing.assert_array_equal(saved["solver_trans"], solver_trans.astype(np.float32)[None])
        np.testing.assert_array_equal(saved["root_orient"], saved["poses"][:, :3])
        self.assertEqual(saved["identity_coeffs"].shape, (128,))
        np.testing.assert_array_equal(saved["identity_coeffs"][:10], betas)
        self.assertEqual(str(saved["identity_model_type"]), "soma")
        self.assertEqual(str(saved["soma_lod"]), "low")
        self.assertEqual(saved["soma_joint_names"].shape, (77,))
        self.assertIn("Hips", str(saved["pose_convention"]))

    def test_soma_label_and_model_disagreement_is_rejected(self):
        for label, model in (("smplx", StubSoma()), ("soma", SimpleNamespace(model_type="smplx"))):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "must match"):
                SequenceRecorder(label, "neutral", np.zeros(10), 300, model=model)

    def test_soma_virtual_root_is_not_accepted_as_an_extra_public_joint(self):
        model = StubSoma()
        model.public_pose = lambda pose, trans: (np.zeros((78,3)), np.zeros(3))
        recorder = SequenceRecorder("soma", "neutral", np.zeros(10), 300, model=model)
        with self.assertRaisesRegex(ValueError, "77 joint"):
            recorder.add(1, np.zeros((77,3)), np.zeros(3), .001)
        for name in ("poses", "trans", "frames", "rmse", "solver_poses", "solver_trans"):
            self.assertEqual(getattr(recorder, name), [])


if __name__ == "__main__":
    unittest.main()
