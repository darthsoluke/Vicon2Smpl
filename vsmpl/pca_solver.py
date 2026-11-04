"""Native full-body SMPL-X fitting with hand rotations constrained by a hand prior.

PCA is a parameterisation/regulariser, not evidence of independent observability
of every finger rotation from two markers per finger.
"""
from dataclasses import replace

import numpy as np

from .hand_pca import BODY_JOINTS, HAND_JOINTS, HandPCA
from .rotations import exp_batch, kabsch
from .solver import RealtimeSolver, SolverConfig


class PCASolver(RealtimeSolver):
    def __init__(self, rig, prior: HandPCA, cfg=None, initial_latent=None, hand_prior_weight=1e-5):
        if rig.model.num_joints != 55:
            raise ValueError("PCASolver requires the native 55-joint SMPL-X model")
        if not np.isfinite(hand_prior_weight) or hand_prior_weight < 0:
            raise ValueError("hand_prior_weight must be finite and nonnegative")
        self.hand_pca = prior
        self.hand_prior_weight = float(hand_prior_weight)
        self._initial_latent = prior._latent(np.zeros((2, prior.n_components)) if initial_latent is None
                                             else initial_latent).copy()
        requested = cfg or SolverConfig()
        if set(requested.fixed_joints) - {22, 23, 24}:
            raise ValueError("PCASolver body/hand controls cannot be independently frozen via fixed_joints")
        active = np.concatenate([BODY_JOINTS, HAND_JOINTS]).tolist()
        if requested.active_joints is not None and set(requested.active_joints) != set(active):
            raise ValueError("PCASolver uses body 0:22 and both hands 25:55")
        configured = replace(requested, active_joints=active, fixed_joints=[22, 23, 24])
        # Finger lever arms are short: omitting pose blendshape derivatives is
        # material for the reduced hand data Jacobian. Body-only callers retain
        # the MarkerRig default unless they explicitly use this solver.
        rig.include_pose_derivatives = True
        super().__init__(rig, configured)
        # Hand rest-AA L2 would compete with the nonzero native mean hand pose.
        # Hand regularisation is applied only to standardised prior coordinates.
        self.prior_w[self.active >= 25] = 0.0

    @property
    def hand_latent(self):
        return self.hand_pca.encode(self.R)

    def _project_hands(self, rotations):
        projected = self.hand_pca.apply(rotations, self.hand_pca.encode(rotations))
        projected[22:25] = np.eye(3)
        return projected

    def reset(self):
        super().reset()
        self.R = self.hand_pca.apply(self.R, self._initial_latent)
        self.R[22:25] = np.eye(3)

    def _predict(self, frame_no):
        rotations, trans = super()._predict(frame_no)
        return self._project_hands(rotations), trans

    def _kabsch_init(self, obs, valid):
        rig = self.rig
        rotations = np.broadcast_to(np.eye(3), (55, 3, 3)).copy()
        rotations = self.hand_pca.apply(rotations, self._initial_latent)
        initial = rig.forward(rotations, np.zeros(3))
        selected = np.asarray(valid, bool) & self.torso_markers
        if selected.sum() < 3:
            selected = np.asarray(valid, bool)
        global_rotation, translation = kabsch(initial[selected], obs[selected])
        rotations[0] = global_rotation
        root = rig.J_rest[0]
        self.R = rotations
        self.trans = translation + global_rotation @ root - root

    def _cost_terms(self, obs, valid, R, t, R_pred, t_pred, lam_t, lam_tr, reject,
                    w_rob=None, with_jac=False):
        result = super()._cost_terms(obs, valid, R, t, R_pred, t_pred, lam_t, lam_tr, reject,
                                     w_rob=w_rob, with_jac=with_jac)
        latent = self.hand_pca.encode(R).reshape(-1)
        cost = result[0] + self.hand_prior_weight * (latent @ latent)
        if not with_jac:
            return cost, result[1], result[2]
        mapping = self.hand_pca.tangent(R, self.active)
        H = mapping.T @ result[3] @ mapping
        gradient = mapping.T @ result[4]
        columns = slice(66, 66 + 2 * self.hand_pca.n_components)
        H[columns, columns] += self.hand_prior_weight * np.eye(len(latent))
        gradient[columns] += self.hand_prior_weight * latent
        return cost, result[1], result[2], H, gradient, result[5]

    def _apply(self, R, t, delta):
        delta = np.asarray(delta, dtype=np.float64)
        if delta.shape != (self.hand_pca.reduced_size,):
            raise ValueError("Reduced solver increment has the wrong size")
        rotations = np.asarray(R).copy()
        rotations[:22] = rotations[:22] @ exp_batch(delta[:66])
        latent = self.hand_pca.encode(R) + delta[66:-3].reshape(2, self.hand_pca.n_components)
        rotations = self.hand_pca.apply(rotations, latent)
        rotations[22:25] = np.eye(3)
        return rotations, t + delta[-3:]
