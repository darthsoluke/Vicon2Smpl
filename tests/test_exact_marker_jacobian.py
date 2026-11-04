"""Finite-difference checks for the optional full pose-blendshape data Jacobian."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.synthetic import make_body, pick_markers
from vsmpl import body_model
from vsmpl.body_model import MarkerRig
from vsmpl.rotations import exp_so3


HAVE_NUMBA = body_model.HAVE_NUMBA


def make_rig(seed=17):
    model, _ = make_body(n_rings=4, n_sec=6)
    rng = np.random.default_rng(seed)
    ids = pick_markers(model, seed=3)
    # Both pose corrections and nonzero rest-space marker offsets are retained.
    beta = rng.normal(0., .3, model.num_betas)
    offsets = rng.normal(0., .008, (len(ids), 3))
    return MarkerRig(model, [f"M{i}" for i in range(len(ids))], ids, offsets, beta)


def finite_difference(rig, R, trans, joints, eps=1e-6):
    expected = np.empty((rig.num_markers, 3, 3*len(joints)+3))
    for slot, joint in enumerate(joints):
        for axis in range(3):
            delta = np.eye(3)[axis] * eps
            plus, minus = R.copy(), R.copy()
            plus[joint] = R[joint] @ exp_so3(delta)
            minus[joint] = R[joint] @ exp_so3(-delta)
            expected[..., 3*slot+axis] = (rig.forward(plus, trans)-rig.forward(minus, trans))/(2*eps)
    for axis in range(3):
        delta = np.eye(3)[axis] * eps
        expected[..., -3+axis] = (rig.forward(R, trans+delta)-rig.forward(R, trans-delta))/(2*eps)
    return expected


class ExactMarkerJacobianTests(unittest.TestCase):
    def _check_exact(self, use_numba):
        with patch.object(body_model, "HAVE_NUMBA", use_numba):
            rig = make_rig()
            self.assertGreater(np.abs(rig.PD).max(), 0)
            self.assertGreater(np.abs(rig.offsets).max(), 0)
            rig.include_pose_derivatives = True
            rng = np.random.default_rng(4)
            trans = np.array([.3, .8, -.2])
            poses = [exp_so3(rng.normal(0., .4, (24, 3))), np.tile(np.eye(3), (24, 1, 1))]
            for R in poses:
                for joints in (np.arange(24), np.array([23, 0, 5, 18, 1])):
                    with self.subTest(numba=use_numba, joints=joints.tolist(), rest=np.array_equal(R[0], np.eye(3))):
                        _, cache = rig.forward(R, trans, with_cache=True)
                        analytic = rig.jacobian(cache, joints)
                        numeric = finite_difference(rig, R, trans, joints)
                        np.testing.assert_allclose(analytic, numeric, atol=2e-7, rtol=2e-6)

    def test_numpy_full_pose_jacobian_matches_forward(self):
        self._check_exact(False)

    @unittest.skipUnless(HAVE_NUMBA, "numba is not installed")
    def test_numba_full_pose_jacobian_matches_forward(self):
        self._check_exact(True)

    def test_default_approximation_and_root_translation_are_preserved(self):
        for use_numba in ([False, True] if HAVE_NUMBA else [False]):
            with self.subTest(numba=use_numba), patch.object(body_model, "HAVE_NUMBA", use_numba):
                rig = make_rig()
                self.assertFalse(rig.include_pose_derivatives)
                R = exp_so3(np.random.default_rng(7).normal(0., .3, (24, 3)))
                t = np.array([1., .4, -.7])
                joints = np.arange(24)
                _, cache = rig.forward(R, t, with_cache=True)
                original = rig.jacobian(cache, joints)
                rig.include_pose_derivatives = True
                exact = rig.jacobian(cache, joints)
                numeric = finite_difference(rig, R, t, joints)
                self.assertGreater(np.max(np.abs(original-numeric)), 2e-5)
                np.testing.assert_allclose(exact, numeric, atol=2e-7, rtol=2e-6)
                # Neither direct pose features nor their derivatives contain the root or translation.
                np.testing.assert_array_equal(exact[..., :3], original[..., :3])
                np.testing.assert_array_equal(exact[..., -3:], original[..., -3:])
                rig.include_pose_derivatives = False
                np.testing.assert_array_equal(rig.jacobian(cache, joints), original)

    def test_no_pose_corrections_and_root_only_are_identical(self):
        for use_numba in ([False, True] if HAVE_NUMBA else [False]):
            with self.subTest(numba=use_numba), patch.object(body_model, "HAVE_NUMBA", use_numba):
                rig = make_rig()
                R = exp_so3(np.random.default_rng(11).normal(0., .3, (24, 3)))
                _, cache = rig.forward(R, np.zeros(3), with_cache=True)
                default_root = rig.jacobian(cache, [0])
                rig.include_pose_derivatives = True
                np.testing.assert_array_equal(rig.jacobian(cache, [0]), default_root)
                rig.model.posedirs[:] = 0
                zero_rig = MarkerRig(rig.model, rig.names, rig.vertex_ids, rig.offsets, rig.betas)
                _, zero_cache = zero_rig.forward(R, np.zeros(3), with_cache=True)
                default = zero_rig.jacobian(zero_cache, np.arange(24))
                zero_rig.include_pose_derivatives = True
                np.testing.assert_array_equal(zero_rig.jacobian(zero_cache, np.arange(24)), default)


if __name__ == "__main__":
    unittest.main()
