"""Synthetic prior checks; no licensed models, hardware or remote connections."""
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vsmpl.hand_pca import BODY_JOINTS, HAND_JOINTS, HandPCA
from vsmpl.pca_solver import PCASolver
from vsmpl.rotations import exp_so3, log_so3, skew
from vsmpl.solver import RealtimeSolver, SolverConfig


def rotations():
    return np.broadcast_to(np.eye(3), (55, 3, 3)).copy()


def make_prior(components=6):
    rng = np.random.default_rng(41)
    basis = rng.normal(size=(2, 45, components)) * .07
    # Deliberately non-orthogonal columns with different scales.
    basis[:, :, 1] += .6 * basis[:, :, 0]
    basis *= np.linspace(.5, 1.8, components)[None, None]
    return HandPCA(rng.normal(size=(2, 45)) * .04, basis)


class FakeRig:
    def __init__(self):
        self.model = SimpleNamespace(num_joints=55)
        self.W = np.zeros((3, 55))
        self.W[:, 0] = 1.
        self.J_rest = np.zeros((55, 3))

    def joint_observability(self):
        return np.ones(55)


class AnalyticRig(FakeRig):
    """Independent smooth rotation/pose-correction rig for cost-chain checks."""
    def __init__(self):
        super().__init__()
        rng = np.random.default_rng(127)
        self.num_markers = 8
        self.W = np.zeros((self.num_markers, 55))
        self.W[:, 0] = 1.
        self.vectors = rng.normal(size=(self.num_markers, 55, 3)) * .005
        self.pose_correction = rng.normal(size=(self.num_markers, 55, 3, 3, 3)) * .004
        self.include_pose_derivatives = False

    def forward(self, R, trans, with_cache=False):
        points = np.einsum("jab,mjb->ma", R, self.vectors)
        points += np.einsum("mjcuv,juv->mc", self.pose_correction, R - np.eye(3))
        points += trans
        return (points, R.copy()) if with_cache else points

    def jacobian(self, R, joints):
        out = np.zeros((self.num_markers, 3, len(joints) * 3 + 3))
        axes = skew(np.eye(3))
        for index, joint in enumerate(joints):
            dR = R[joint][None] @ axes
            block = np.einsum("kab,mb->mak", dR, self.vectors[:, joint])
            if self.include_pose_derivatives:
                block += np.einsum("mcab,kab->mck", self.pose_correction[:, joint], dR)
            out[:, :, 3*index:3*index+3] = block
        out[:, :, -3:] = np.eye(3)
        return out


class HandPCATests(unittest.TestCase):
    def test_file_mean_scale_order_and_nonorthogonal_roundtrip(self):
        rng = np.random.default_rng(4)
        components_left = rng.normal(size=(45, 45)) * .1
        components_right = rng.normal(size=(45, 45)) * .15
        means = rng.normal(size=(2, 45)) * .04
        coeff_left = rng.normal(size=(150, 45)) * np.linspace(.4, 1.3, 45)
        coeff_right = rng.normal(size=(150, 45)) * np.linspace(1.4, .6, 45)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic_prior.npz"
            np.savez(path, componentsl=components_left, componentsr=components_right,
                     hands_meanl=means[0], hands_meanr=means[1],
                     hands_coeffsl=coeff_left, hands_coeffsr=coeff_right)
            prior = HandPCA.load(path, n_components=12)
        self.assertEqual(prior.basis.shape, (2, 45, 12))
        np.testing.assert_allclose(prior.mean, means)
        for side, (component, coefficient) in enumerate([(components_left, coeff_left),
                                                         (components_right, coeff_right)]):
            scale = coefficient[:, :12].std(axis=0, ddof=0)
            np.testing.assert_allclose(prior.scale[side], scale)
            np.testing.assert_allclose(prior.basis[side], component[:12].T * scale)
        z = rng.normal(size=(2, 12)) * .25
        original = rotations()
        original[:22] = exp_so3(rng.normal(size=(22, 3)) * .1)
        original[22:25] = exp_so3(rng.normal(size=(3, 3)) * .1)
        before = original.copy()
        result = prior.apply(original, z)
        np.testing.assert_allclose(prior.encode(result), z, atol=1e-10)
        np.testing.assert_array_equal(original, before)
        np.testing.assert_array_equal(result[:25], original[:25])
        zero = prior.apply(original, np.zeros((2, 12)))
        np.testing.assert_allclose(zero[25:55], exp_so3(means.reshape(30, 3)), atol=1e-12)

    def test_right_increment_tangent_matches_finite_difference(self):
        prior = make_prior()
        rng = np.random.default_rng(8)
        z = rng.normal(size=(2, prior.n_components)) * .8
        current = rotations()
        current[:22] = exp_so3(rng.normal(size=(22, 3)) * .3)
        current = prior.apply(current, z)
        active = np.concatenate([BODY_JOINTS, HAND_JOINTS])
        tangent = prior.tangent(current, active)
        self.assertEqual(tangent.shape, (len(active) * 3 + 3, prior.reduced_size))
        direction = rng.normal(size=prior.reduced_size)
        direction /= np.linalg.norm(direction)
        eps = 1e-6
        updated = current.copy()
        updated[:22] = updated[:22] @ exp_so3((eps * direction[:66]).reshape(22, 3))
        updated = prior.apply(updated, z + (eps * direction[66:-3]).reshape(2, prior.n_components))
        numeric = np.r_[log_so3(np.swapaxes(current[active], -1, -2) @ updated[active]).reshape(-1),
                        eps * direction[-3:]] / eps
        np.testing.assert_allclose(tangent @ direction, numeric, atol=2e-7, rtol=2e-6)

    def test_solver_projects_prediction_and_freezes_face(self):
        prior = make_prior(3)
        initial = np.full((2, 3), .15)
        solver = PCASolver(FakeRig(), prior, initial_latent=initial)
        self.assertTrue(solver.rig.include_pose_derivatives)
        np.testing.assert_allclose(solver.hand_latent, initial, atol=1e-12)
        np.testing.assert_array_equal(solver.active, np.r_[0:22, 25:55])
        self.assertTrue(np.all(solver.prior_w[solver.active >= 25] == 0))
        z0 = np.array([[1.2, -.8, .5], [-1., .7, .8]])
        z1 = np.array([[-.6, 1.5, -.8], [.9, -.8, 1.3]])
        r0 = prior.apply(rotations(), z0)
        r1 = prior.apply(rotations(), z1)
        r0[22:25] = exp_so3(np.full((3, 3), .1))
        r1[22:25] = exp_so3(np.full((3, 3), .2))
        solver.R = r1
        solver.trans = np.ones(3)
        solver.hist = [(r0, np.zeros(3), 1), (r1, np.ones(3), 2)]
        raw, raw_translation = RealtimeSolver._predict(solver, 3)
        predicted, translation = solver._predict(3)
        projected = prior.apply(predicted, prior.encode(predicted))
        np.testing.assert_allclose(predicted[25:55], projected[25:55], atol=1e-12)
        np.testing.assert_array_equal(predicted[22:25], np.broadcast_to(np.eye(3), (3, 3, 3)))
        np.testing.assert_allclose(translation, raw_translation)
        self.assertGreater(np.linalg.norm(raw[25:55] - predicted[25:55]), 1e-6)

    def test_solver_reduces_native_normal_equations_and_adds_latent_prior(self):
        prior = make_prior(4)
        weight = 3e-4
        solver = PCASolver(FakeRig(), prior, hand_prior_weight=weight)
        z = np.full((2, 4), .2)
        current = prior.apply(rotations(), z)
        full_size = len(solver.active) * 3 + 3
        H = np.diag(np.linspace(1, 2, full_size))
        gradient = np.linspace(-.1, .2, full_size)
        valid = np.ones(3, bool)
        distance = np.zeros(3)
        robust = np.ones(3)
        with patch.object(RealtimeSolver, "_cost_terms", return_value=(2., valid, distance, H, gradient, robust)):
            result = solver._cost_terms(None, valid, current, np.zeros(3), current, np.zeros(3),
                                        1e-3, 1e-2, .1, with_jac=True)
        B = prior.tangent(current, solver.active)
        expected_H = B.T @ H @ B
        expected_gradient = B.T @ gradient
        hand = slice(66, 74)
        expected_H[hand, hand] += weight * np.eye(8)
        expected_gradient[hand] += weight * z.reshape(-1)
        self.assertAlmostEqual(result[0], 2 + weight * np.square(z).sum())
        np.testing.assert_allclose(result[3], expected_H)
        np.testing.assert_allclose(result[4], expected_gradient)
        delta = np.linspace(-.01, .01, prior.reduced_size)
        updated, translation = solver._apply(current, np.zeros(3), delta)
        np.testing.assert_allclose(prior.encode(updated), z + delta[66:-3].reshape(2, 4), atol=1e-12)
        np.testing.assert_allclose(translation, delta[-3:])

    def test_reduced_cost_gradient_includes_pose_correction_and_all_priors(self):
        prior = make_prior(5)
        rig = AnalyticRig()
        cfg = SolverConfig(lam_prior=.005, lam_temp=.003, lam_trans=.02, huber_delta=1.)
        solver = PCASolver(rig, prior, cfg, hand_prior_weight=.007)
        self.assertTrue(rig.include_pose_derivatives)
        rng = np.random.default_rng(128)
        current = rotations()
        current[:22] = exp_so3(rng.normal(size=(22, 3)) * .2)
        current = prior.apply(current, rng.normal(size=(2, 5)) * .3)
        predicted = rotations()
        predicted[:22] = exp_so3(rng.normal(size=(22, 3)) * .15)
        predicted = prior.apply(predicted, rng.normal(size=(2, 5)) * .2)
        translation, predicted_translation = np.array([.1, .2, -.1]), np.array([.09, .21, -.12])
        obs = rig.forward(current, translation) + rng.normal(size=(rig.num_markers, 3)) * .005
        valid = np.ones(rig.num_markers, bool)
        args = (obs, valid, current, translation, predicted, predicted_translation,
                cfg.lam_temp, cfg.lam_trans, np.inf)
        result = solver._cost_terms(*args, with_jac=True)
        direction = rng.normal(size=prior.reduced_size)
        direction /= np.linalg.norm(direction)
        eps = 1e-6
        r_plus, t_plus = solver._apply(current, translation, eps * direction)
        r_minus, t_minus = solver._apply(current, translation, -eps * direction)
        def cost(R, t):
            return solver._cost_terms(obs, valid, R, t, predicted, predicted_translation,
                                      cfg.lam_temp, cfg.lam_trans, np.inf, w_rob=result[5])[0]
        numeric = (cost(r_plus, t_plus) - cost(r_minus, t_minus)) / (2 * eps)
        # The inherited normal equations use the gradient of one-half the squared cost.
        self.assertAlmostEqual(numeric, 2 * (result[4] @ direction), delta=2e-8)
        rig.include_pose_derivatives = False
        approximate = solver._cost_terms(*args, with_jac=True)
        self.assertGreater(abs(numeric - 2 * (approximate[4] @ direction)), 1e-8)

    def test_rejects_degenerate_prior_and_non_native_model(self):
        with self.assertRaises(ValueError):
            HandPCA(np.zeros((2, 45)), np.zeros((2, 45, 3)))
        bad_rig = FakeRig()
        bad_rig.model.num_joints = 24
        with self.assertRaises(ValueError):
            PCASolver(bad_rig, make_prior())
        with self.assertRaises(ValueError):
            PCASolver(FakeRig(), make_prior(), SolverConfig(fixed_joints=[20]))


if __name__ == "__main__":
    unittest.main()
