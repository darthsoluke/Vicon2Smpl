"""SO(3) helpers in numpy (float64). All functions accept batched inputs [..., 3] / [..., 3, 3]."""
import numpy as np

_EPS = 1e-8


def skew(v):
    v = np.asarray(v)
    o = np.zeros(v.shape[:-1])
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(*v.shape[:-1], 3, 3)


def exp_so3(r):
    """Axis-angle [...,3] -> rotation matrix [...,3,3] (Rodrigues, stable near 0)."""
    r = np.asarray(r, dtype=np.float64)
    t2 = np.sum(r * r, -1)[..., None, None]
    small = t2 < _EPS
    t2s = np.where(small, 1.0, t2)
    t = np.sqrt(t2s)
    a = np.where(small, 1.0 - t2 / 6.0, np.sin(t) / t)
    b = np.where(small, 0.5 - t2 / 24.0, (1.0 - np.cos(t)) / t2s)
    K = skew(r)
    return np.eye(3) + a * K + b * (K @ K)


def log_so3(R):
    """Rotation matrix [...,3,3] -> axis-angle [...,3]. Handles angles near 0 and pi."""
    R = np.asarray(R, dtype=np.float64)
    if R.ndim == 2:
        return log_so3(R[None])[0]
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    w = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(theta)
    small = theta < 1e-4
    near_pi = theta > np.pi - 1e-3
    fac = np.where(small, 0.5 + theta ** 2 / 12.0, theta / (2.0 * np.where(small | near_pi, 1.0, s)))
    out = fac[..., None] * w
    if np.any(near_pi):
        # near pi: (R + R^T)/2 - cos(theta) I = (1 - cos(theta)) a a^T  (exact, unlike (R + I)/2)
        Rp, th = R[near_pi], theta[near_pi]
        c = np.cos(th)[:, None, None]
        S = ((Rp + np.swapaxes(Rp, -1, -2)) / 2.0 - c * np.eye(3)) / (1.0 - c)
        k = np.argmax(np.diagonal(S, axis1=-2, axis2=-1), -1)
        ax = S[np.arange(len(k)), :, k]
        ax = ax / np.linalg.norm(ax, axis=-1, keepdims=True)
        sgn = np.sign(np.sum(ax * w[near_pi], -1))  # the antisymmetric part fixes the sign
        sgn[sgn == 0] = 1.0
        out[near_pi] = ax * (th * sgn)[..., None]
    return out


def right_jacobian_inv(phi):
    """Inverse right Jacobian of SO(3): d log(R exp(w)) / dw at w=0, with R = exp(phi)."""
    phi = np.asarray(phi, dtype=np.float64)
    t2 = np.sum(phi * phi, -1)[..., None, None]
    t = np.sqrt(t2)
    small = t2 < 1e-6
    ts = np.where(small, 1.0, t)
    c = np.where(small, 1.0 / 12.0, 1.0 / ts ** 2 - (1.0 + np.cos(ts)) / (2.0 * ts * np.sin(np.where(small, 1.0, ts)) + 1e-12))
    K = skew(phi)
    return np.eye(3) + 0.5 * K + c * (K @ K)


def kabsch(P, Q, w=None):
    """Find R, t minimising sum w ||R P + t - Q||^2. P, Q: [N,3]."""
    if w is None:
        w = np.ones(len(P))
    w = w / w.sum()
    mp, mq = (w[:, None] * P).sum(0), (w[:, None] * Q).sum(0)
    H = ((P - mp) * w[:, None]).T @ (Q - mq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, mq - R @ mp


def rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# ---------------------------------------------------------------- fast batched versions (numba if available)
from ._kernels import HAVE_NUMBA  # noqa: E402

if HAVE_NUMBA:
    from ._kernels import exp_nb, log_nb, rjinv_nb


def exp_batch(r):
    """(N,3) -> (N,3,3)."""
    r = np.ascontiguousarray(r, dtype=np.float64).reshape(-1, 3)
    return exp_nb(r) if HAVE_NUMBA else exp_so3(r)


def log_batch(R):
    """(N,3,3) -> (N,3)."""
    R = np.ascontiguousarray(R, dtype=np.float64).reshape(-1, 3, 3)
    return log_nb(R) if HAVE_NUMBA else log_so3(R)


def rjinv_batch(phi):
    phi = np.ascontiguousarray(phi, dtype=np.float64).reshape(-1, 3)
    return rjinv_nb(phi) if HAVE_NUMBA else right_jacobian_inv(phi)
