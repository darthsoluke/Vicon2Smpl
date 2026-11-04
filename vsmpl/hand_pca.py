"""A licensed hand prior as a reduced parameterisation of native SMPL-X rotations.

The model geometry, kinematic tree and skinning remain unchanged. ``basis`` maps
standardised latent coefficients into 45 native axis-angle values per hand.
"""
from dataclasses import dataclass

import numpy as np

from .rotations import exp_so3, log_so3, rjinv_batch


BODY_JOINTS = np.arange(22, dtype=np.int64)
LEFT_HAND_JOINTS = np.arange(25, 40, dtype=np.int64)
RIGHT_HAND_JOINTS = np.arange(40, 55, dtype=np.int64)
HAND_JOINTS = np.concatenate([LEFT_HAND_JOINTS, RIGHT_HAND_JOINTS])


@dataclass
class HandPCA:
    mean: np.ndarray  # (2,45), native SMPL-X order
    basis: np.ndarray  # (2,45,K), includes coefficient standard deviations
    components: np.ndarray = None  # (2,K,45), file values without rescaling
    scale: np.ndarray = None  # (2,K), coefficient population std (ddof=0)

    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=np.float64).copy()
        self.basis = np.asarray(self.basis, dtype=np.float64).copy()
        if self.mean.shape != (2, 45) or self.basis.ndim != 3 or self.basis.shape[:2] != (2, 45):
            raise ValueError("Hand prior requires mean (2,45) and basis (2,45,K)")
        if not 1 <= self.basis.shape[2] <= 45:
            raise ValueError("Hand prior component count must be between 1 and 45")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.basis).all():
            raise ValueError("Hand prior contains non-finite values")
        if any(np.linalg.matrix_rank(b, tol=1e-10) != self.n_components for b in self.basis):
            raise ValueError("Hand prior basis must have independent columns")
        # File components need not be orthogonal or unit length.
        self._encoder = np.stack([np.linalg.pinv(b) for b in self.basis])
        if self.components is not None:
            self.components = np.asarray(self.components, dtype=np.float64).copy()
        if self.scale is not None:
            self.scale = np.asarray(self.scale, dtype=np.float64).copy()

    @property
    def n_components(self):
        return self.basis.shape[2]

    @property
    def reduced_size(self):
        return 22 * 3 + 2 * self.n_components + 3

    @property
    def left_slice(self):
        return slice(66, 66 + self.n_components)

    @property
    def right_slice(self):
        return slice(66 + self.n_components, 66 + 2 * self.n_components)

    @classmethod
    def load(cls, path, n_components=12):
        if not isinstance(n_components, (int, np.integer)) or not 1 <= n_components <= 45:
            raise ValueError("n_components must be an integer between 1 and 45")
        means, components, scales = [], [], []
        with np.load(path, allow_pickle=False) as data:
            for side in ("l", "r"):
                component = np.asarray(data["components" + side], dtype=np.float64)
                mean = np.asarray(data["hands_mean" + side], dtype=np.float64)
                coeff = np.asarray(data["hands_coeffs" + side], dtype=np.float64)
                if component.shape != (45, 45) or mean.shape != (45,):
                    raise ValueError(f"Unexpected {side} hand component or mean shape")
                if coeff.ndim != 2 or coeff.shape[1] != 45 or len(coeff) < 2:
                    raise ValueError(f"Unexpected {side} hand coefficient samples")
                if not np.isfinite(coeff).all():
                    raise ValueError(f"Non-finite {side} hand coefficient samples")
                scale = coeff[:, :n_components].std(axis=0, ddof=0)
                if not np.isfinite(scale).all() or np.any(scale <= 1e-10):
                    raise ValueError(f"Degenerate {side} hand coefficient variance")
                means.append(mean)
                components.append(component[:n_components])
                scales.append(scale)
        component = np.stack(components)
        scale = np.stack(scales)
        basis = np.transpose(component * scale[..., None], (0, 2, 1))
        return cls(np.stack(means), basis, component, scale)

    @staticmethod
    def _rotations(R55):
        rotations = np.asarray(R55, dtype=np.float64)
        if rotations.shape != (55, 3, 3) or not np.isfinite(rotations).all():
            raise ValueError("Expected finite native SMPL-X rotations (55,3,3)")
        return rotations

    def _latent(self, z):
        latent = np.asarray(z, dtype=np.float64)
        if latent.shape != (2, self.n_components) or not np.isfinite(latent).all():
            raise ValueError(f"Expected finite hand latent (2,{self.n_components})")
        return latent

    def encode(self, R55):
        """Least-squares projection of native principal-branch axis angles onto the prior."""
        rotations = self._rotations(R55)
        aa = log_so3(rotations[HAND_JOINTS]).reshape(2, 45)
        return np.einsum("skt,st->sk", self._encoder, aa - self.mean)

    def apply(self, R55, z):
        """Copy full native rotations and replace only the two hands with prior rotations."""
        rotations = self._rotations(R55).copy()
        latent = self._latent(z)
        aa = self.mean + np.einsum("stk,sk->st", self.basis, latent)
        rotations[HAND_JOINTS] = exp_so3(aa.reshape(30, 3))
        return rotations

    def tangent(self, R55, active):
        """Map reduced deltas to full local right-increments and translation.

        Column order: 22 body rotation increments, left/right standardised hand
        coefficients, translation. Hand rotations must already lie in the prior
        subspace. The rotation logarithm uses its principal branch.
        """
        rotations = self._rotations(R55)
        active = np.asarray(active, dtype=np.int64)
        if active.ndim != 1 or len(np.unique(active)) != len(active) or np.any((active < 0) | (active >= 55)):
            raise ValueError("active must contain unique native SMPL-X joint indices")
        mapping = np.zeros((len(active) * 3 + 3, self.reduced_size))
        aa = log_so3(rotations[HAND_JOINTS])
        # For R=exp(r), exp(r+dr) = R exp(J_right(r) dr) + O(dr^2).
        # Existing rjinv_batch is J_right^-1, not J_right.
        inverse_right = rjinv_batch(aa)
        loading = self.basis.reshape(2, 15, 3, self.n_components).reshape(30, 3, self.n_components)
        right_loading = np.linalg.solve(inverse_right, loading)
        for index, joint in enumerate(active):
            row = slice(3 * index, 3 * index + 3)
            if joint < 22:
                mapping[row, 3 * joint:3 * joint + 3] = np.eye(3)
            elif joint >= 25:
                hand_index = joint - 25
                columns = self.left_slice if hand_index < 15 else self.right_slice
                mapping[row, columns] = right_loading[hand_index]
        mapping[-3:, -3:] = np.eye(3)
        return mapping
