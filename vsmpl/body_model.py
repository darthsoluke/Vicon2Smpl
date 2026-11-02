"""Numpy body model (SMPL / SMPL-H / SMPL-X body) + the 'marker rig' used by the real-time solver.

Conventions (same as smplx):
  - local rotations R[j] relative to parent, R[0] = global root orientation
  - pose blendshape feature = (R[1:] - I).reshape(-1)
  - output = LBS(v_rest + pose_offsets) + trans
All arrays are float64 numpy.
"""
from dataclasses import dataclass, field
import numpy as np

from ._kernels import HAVE_NUMBA

if HAVE_NUMBA:
    from ._kernels import fk_nb, skin_nb, jacobian_nb

# SMPL / SMPL-X share the first 22 body joints.
TORSO_JOINTS = (0, 3, 6, 9)  # pelvis, spine1, spine2, spine3


class BodyModel:
    def __init__(self, v_template, shapedirs, posedirs, J_regressor, weights, parents, faces, model_type="smpl"):
        self.v_template = np.asarray(v_template, np.float64)  # (V,3)
        self.shapedirs = np.asarray(shapedirs, np.float64)  # (V,3,B)
        self.posedirs = np.asarray(posedirs, np.float64)  # (P,V,3)
        self.J_regressor = np.asarray(J_regressor, np.float64)  # (J,V)
        self.weights = np.asarray(weights, np.float64)  # (V,J)
        self.parents = np.asarray(parents, np.int64).copy()
        self.parents[0] = -1
        self.faces = np.asarray(faces, np.int64)
        self.model_type = model_type
        self.num_joints = self.J_regressor.shape[0]
        self.num_betas = self.shapedirs.shape[-1]
        # subtree[j, k] = 1 if k is j or a descendant of j
        J = self.num_joints
        self.subtree = np.zeros((J, J))
        for k in range(J):
            j = k
            while j >= 0:
                self.subtree[j, k] = 1.0
                j = self.parents[j]
        self._order_ok = all(self.parents[j] < j for j in range(1, J))
        assert self._order_ok, "kinematic tree must be topologically ordered"
        # joints grouped by depth, so forward kinematics needs only one batched step per tree level
        depth = np.zeros(J, np.int64)
        for j in range(1, J):
            depth[j] = depth[self.parents[j]] + 1
        self.levels = [np.nonzero(depth == d)[0] for d in range(1, depth.max() + 1)]

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, model_path, model_type="smpl", gender="neutral", num_betas=10):
        """Load through the `smplx` package (supports .pkl / .npz model files)."""
        import smplx

        kw = dict(model_type=model_type, gender=gender, num_betas=num_betas)
        if model_type in ("smplh", "smplx"):
            kw.update(use_pca=False, flat_hand_mean=True)
        if model_type == "smplx":
            kw.update(num_expression_coeffs=10)
        m = smplx.create(model_path, **kw)

        def t(x):
            return x.detach().cpu().double().numpy()

        V = m.v_template.shape[0]
        posedirs = t(m.posedirs).reshape(-1, V, 3)  # smplx stores (P, V*3)
        shapedirs = t(m.shapedirs)[..., :num_betas]
        return cls(t(m.v_template), shapedirs, posedirs, t(m.J_regressor), t(m.lbs_weights),
                   t(m.parents).astype(np.int64), np.asarray(m.faces, np.int64), model_type)

    # ------------------------------------------------------------------ shape
    def shaped(self, betas):
        b = np.zeros(self.num_betas)
        betas = np.asarray(betas, np.float64).ravel()[: self.num_betas]
        b[: len(betas)] = betas
        v = self.v_template + self.shapedirs @ b
        return v, self.J_regressor @ v

    def vertex_normals(self, v):
        f = self.faces
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        n = np.zeros_like(v)
        for i in range(3):
            np.add.at(n, f[:, i], fn)
        return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)

    # ------------------------------------------------------------------ kinematics
    def fk(self, R, J_rest):
        """R: (J,3,3) local rotations. Returns world rotations Rw (J,3,3) and world joint positions c (J,3)
        (without global translation)."""
        if HAVE_NUMBA:
            return fk_nb(np.ascontiguousarray(R, dtype=np.float64), np.ascontiguousarray(J_rest), self.parents)
        J = self.num_joints
        Rw = np.empty((J, 3, 3))
        c = np.empty((J, 3))
        Rw[0] = R[0]
        c[0] = J_rest[0]
        par = self.parents
        for js in self.levels:
            ps = par[js]
            Rp = Rw[ps]
            Rw[js] = Rp @ R[js]
            c[js] = (Rp @ (J_rest[js] - J_rest[ps])[:, :, None])[..., 0] + c[ps]
        return Rw, c

    def pose_feature(self, R):
        return (R[1:] - np.eye(3)).reshape(-1)

    def vertices(self, R, trans, v_rest, J_rest):
        """Full mesh (for visualisation)."""
        Rw, c = self.fk(R, J_rest)
        v = v_rest + np.einsum("p,pvc->vc", self.pose_feature(R), self.posedirs)
        At = c - np.einsum("jab,jb->ja", Rw, J_rest)  # (J,3)
        T_R = np.einsum("vj,jab->vab", self.weights, Rw)
        T_t = self.weights @ At
        return np.einsum("vab,vb->va", T_R, v) + T_t + trans

    def joints(self, R, trans, J_rest):
        return self.fk(R, J_rest)[1] + trans


@dataclass
class MarkerRig:
    """Markers attached to the body: each marker = vertex(β) + a rest-space offset, skinned with
    that vertex's weights and pose blendshapes ("virtual vertex")."""

    model: BodyModel
    names: list
    vertex_ids: np.ndarray  # (M,)
    offsets: np.ndarray  # (M,3) rest-space offset from the surface vertex
    betas: np.ndarray
    p_rest: np.ndarray = field(init=False)
    W: np.ndarray = field(init=False)
    PD: np.ndarray = field(init=False)
    J_rest: np.ndarray = field(init=False)
    v_rest: np.ndarray = field(init=False)

    def __post_init__(self):
        self.vertex_ids = np.asarray(self.vertex_ids, np.int64)
        self.offsets = np.asarray(self.offsets, np.float64)
        self.betas = np.asarray(self.betas, np.float64)
        self.v_rest, self.J_rest = self.model.shaped(self.betas)
        self.p_rest = np.ascontiguousarray(self.v_rest[self.vertex_ids] + self.offsets)
        self.W = np.ascontiguousarray(self.model.weights[self.vertex_ids])  # (M,J)
        self.PD = self.model.posedirs[:, self.vertex_ids]  # (P,M,3)
        self._PD2 = np.ascontiguousarray(self.PD.reshape(len(self.PD), -1))  # (P, M*3)
    @classmethod
    def from_calibration(cls, model, calib):
        return cls(model, [str(n) for n in calib["marker_names"]], calib["vertex_ids"], calib["offsets"], calib["betas"])

    @property
    def num_markers(self):
        return len(self.names)

    def forward(self, R, trans, with_cache=False):
        """Marker positions (M,3). If with_cache also returns quantities needed for the Jacobian."""
        m = self.model
        Rw, c = m.fk(R, self.J_rest)
        p = self.p_rest + (m.pose_feature(R) @ self._PD2).reshape(-1, 3)
        if HAVE_NUMBA:
            out, q = skin_nb(Rw, c, self.J_rest, p, self.W, np.asarray(trans, np.float64), with_cache)
            return (out, (Rw, c, q)) if with_cache else out
        At = c - (Rw @ self.J_rest[:, :, None])[..., 0]
        # blended transform per marker: T_m = sum_j w_mj [Rw_j | At_j]
        TR = (self.W @ Rw.reshape(len(Rw), 9)).reshape(-1, 3, 3)
        out = (TR @ p[:, :, None])[..., 0] + self.W @ At + trans
        if with_cache:
            q = np.transpose(Rw @ p.T, (2, 0, 1)) + At[None]  # (M,J,3) point transformed by each joint
            return out, (Rw, c, q)
        return out

    def jacobian(self, cache, joints):
        """d marker / d (local right-increment of R[j]) for j in `joints`, and d/d trans.
        Returns (M,3,len(joints)*3 + 3). Pose blendshape derivatives are ignored (Gauss-Newton approx)."""
        Rw, c, q = cache
        if HAVE_NUMBA:
            return jacobian_nb(Rw, c, q, self.W, self.model.subtree, np.asarray(joints, np.int64))
        D = self.model.subtree[joints]  # (K,J)
        M = len(q)
        wq = self.W[:, :, None] * q  # (M,J,3)
        s = D @ wq  # (M,K,3)
        S = self.W @ D.T  # (M,K)
        u = s - S[..., None] * c[joints][None]  # (M,K,3)
        # d p / d w_local = -[u]x Rw  ->  rows x of the block
        RwK = Rw[joints]  # (K,3,3)
        ux, uy, uz = u[..., 0:1], u[..., 1:2], u[..., 2:3]  # (M,K,1)
        r0, r1, r2 = RwK[None, :, 0, :], RwK[None, :, 1, :], RwK[None, :, 2, :]  # rows of Rw, (1,K,3)
        # -[u]x = [[0, uz, -uy], [-uz, 0, ux], [uy, -ux, 0]]
        row0 = uz * r1 - uy * r2
        row1 = -uz * r0 + ux * r2
        row2 = uy * r0 - ux * r1
        Jp = np.stack([row0, row1, row2], 1).reshape(M, 3, -1)  # (M,3,K*3)
        out = np.empty((M, 3, Jp.shape[-1] + 3))
        out[..., :-3] = Jp
        out[..., -3:] = np.eye(3)
        return out

    def joint_observability(self):
        """Max over markers of |d marker / d rotation| (m/rad) for each joint at rest pose."""
        R = np.broadcast_to(np.eye(3), (self.model.num_joints, 3, 3)).copy()
        _, cache = self.forward(R, np.zeros(3), with_cache=True)
        Jm = self.jacobian(cache, np.arange(self.model.num_joints))[..., :-3]
        Jm = Jm.reshape(len(Jm), 3, -1, 3)
        # smallest singular direction matters less; use per-joint Frobenius over markers
        return np.sqrt((Jm ** 2).sum(axis=(1, 3))).max(0)
