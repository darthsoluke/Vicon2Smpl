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

# d exp(delta) / d delta_a at zero, for local right rotation increments.
_RIGHT_ROTATION_GENERATORS = np.array([
    [[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]],
    [[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]],
    [[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]],
])


class BodyModel:
    """SMPL-form body: rest frames are world-aligned, so the rest pose is all-identity rotations.

    Two optional generalisations (used by the SOMA backend, see soma_model.py) keep SMPL behaviour unchanged:
      * posedirs=None: no pose-corrective blendshapes (pure LBS);
      * J_template/J_shapedirs: rest joints given directly as J_template + J_shapedirs @ betas instead of
        J_regressor @ vertices.
    """

    # Skeleton roles used by initialisation and priors (SMPL / SMPL-H / SMPL-X joint numbering).
    torso_joints = TORSO_JOINTS
    head_joint = 15
    rest_is_tpose = True  # rotations = identity reproduces the T-pose
    joint_names = None

    def __init__(self, v_template, shapedirs, posedirs, J_regressor, weights, parents, faces, model_type="smpl",
                 J_template=None, J_shapedirs=None):
        self.v_template = np.asarray(v_template, np.float64)  # (V,3)
        self.shapedirs = np.asarray(shapedirs, np.float64)  # (V,3,B)
        self.posedirs = None if posedirs is None else np.asarray(posedirs, np.float64)  # (P,V,3)
        self.J_regressor = None if J_regressor is None else np.asarray(J_regressor, np.float64)  # (J,V)
        self.J_template = None if J_template is None else np.asarray(J_template, np.float64)  # (J,3)
        self.J_shapedirs = None if J_shapedirs is None else np.asarray(J_shapedirs, np.float64)  # (J,3,B)
        if self.J_regressor is None and (self.J_template is None or self.J_shapedirs is None):
            raise ValueError("Rest joints need either J_regressor or J_template + J_shapedirs")
        self.weights = np.asarray(weights, np.float64)  # (V,J)
        self.parents = np.asarray(parents, np.int64).copy()
        self.parents[0] = -1
        self.faces = np.asarray(faces, np.int64)
        self.model_type = model_type
        self.num_joints = (self.J_template if self.J_regressor is None else self.J_regressor).shape[0]
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
    def load(cls, model_path, model_type="smpl", gender="neutral", num_betas=10, soma_lod="mid",
             soma_identity="soma"):
        """Load through the `smplx` package (supports .pkl / .npz model files), or NVIDIA SOMA for
        model_type="soma" (model_path = SOMA asset folder, or "hf" for the HuggingFace cache)."""
        if model_type == "soma":
            from .soma_model import SomaBodyModel
            kwargs = None
            if soma_identity in ("smpl", "smplh", "smplx"):
                kwargs = dict(model_path=model_path, gender=gender)  # licensed SMPL files for that identity backend
            return SomaBodyModel.load(None if soma_identity != "soma" else model_path, num_betas=num_betas,
                                      lod=soma_lod, identity_model_type=soma_identity, identity_model_kwargs=kwargs)
        from pathlib import Path
        if Path(model_path).is_file() and str(model_path).lower().endswith(".npz"):
            # Geometry-only MoSh support models lack hand PCA metadata required by smplx.create,
            # even when PCA is disabled. The solver needs only these native FK/LBS arrays.
            with np.load(model_path, allow_pickle=False) as d:
                v = d["v_template"]
                pd = d["posedirs"]
                if pd.ndim == 3 and pd.shape[:2] == v.shape:
                    pd = pd.transpose(2, 0, 1)
                elif pd.ndim == 2:
                    pd = (pd.T if pd.shape[0] == v.size else pd).reshape(-1, len(v), 3)
                return cls(v, d["shapedirs"][..., :num_betas], pd, d["J_regressor"],
                           d["weights"], d["kintree_table"][0].astype(np.int64), d["f"], model_type)
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
        if self.J_regressor is None:
            return v, self.J_template + self.J_shapedirs @ b
        return v, self.J_regressor @ v

    def posedirs_at(self, vertex_ids):
        """Pose blendshapes (P,len(ids),3) of selected vertices; zeros for a model without correctives."""
        vertex_ids = np.asarray(vertex_ids, np.int64)
        if self.posedirs is None:
            return np.zeros(((self.num_joints - 1) * 9, len(vertex_ids), 3))
        return self.posedirs[:, vertex_ids]

    # ------------------------------------------------------------------ backend hooks (no-ops for SMPL)
    def prepare_for(self, betas):
        """Called before building a rig for `betas` (SOMA re-linearises its identity model there)."""

    def public_pose(self, pose_aa, trans):
        """Solver parameters -> the model's published convention. SMPL: unchanged (pose (J,3), trans)."""
        return np.asarray(pose_aa), np.asarray(trans)

    def calibration_metadata(self):
        """Extra fields stored in a calibration file so the same model can be rebuilt later."""
        return {}

    def recording_metadata(self, betas):
        """Extra fields stored with a solved sequence (published convention, identity)."""
        return {}

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
        v = v_rest if self.posedirs is None else v_rest + np.einsum("p,pvc->vc", self.pose_feature(R), self.posedirs)
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
    include_pose_derivatives: bool = False
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
        self.has_posedirs = self.model.posedirs is not None
        self.PD = self.model.posedirs_at(self.vertex_ids)  # (P,M,3)
        self._PD2 = np.ascontiguousarray(self.PD.reshape(len(self.PD), -1))  # (P, M*3)
        self._PD_blocks = self._PD2.reshape(self.model.num_joints - 1, 9, len(self.names), 3)
    @classmethod
    def from_calibration(cls, model, calib):
        model.prepare_for(calib["betas"])
        return cls(model, [str(n) for n in calib["marker_names"]], calib["vertex_ids"], calib["offsets"], calib["betas"])

    @property
    def num_markers(self):
        return len(self.names)

    def forward(self, R, trans, with_cache=False):
        """Marker positions (M,3). If with_cache also returns quantities needed for the Jacobian."""
        m = self.model
        Rw, c = m.fk(R, self.J_rest)
        p = self.p_rest + (m.pose_feature(R) @ self._PD2).reshape(-1, 3) if self.has_posedirs else self.p_rest
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
        Returns (M,3,len(joints)*3 + 3). The default preserves the approximation that ignores
        pose blendshape derivatives; set include_pose_derivatives=True to include their direct term."""
        Rw, c, q = cache
        if HAVE_NUMBA:
            out = jacobian_nb(Rw, c, q, self.W, self.model.subtree, np.asarray(joints, np.int64))
            if self.include_pose_derivatives:
                self._add_pose_derivatives(out, Rw, joints)
            return out
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
        if self.include_pose_derivatives:
            self._add_pose_derivatives(out, Rw, joints)
        return out

    def _add_pose_derivatives(self, out, Rw, joints):
        """Add TR_m * d pose_offset_m / d delta_j, without numerical differencing."""
        if not self.has_posedirs:
            return
        active = np.asarray(joints, dtype=np.int64)
        slots = np.flatnonzero(active != 0)
        if not len(slots):
            return
        selected = active[slots]
        # Pose features use each non-root local R_j, not the accumulated world rotation.
        parents = self.model.parents[selected]
        local = np.transpose(Rw[parents], (0, 2, 1)) @ Rw[selected]
        d_feature = (local[:, None] @ _RIGHT_ROTATION_GENERATORS[None]).reshape(len(selected), 3, 9)
        # (marker, rest-coordinate, active-joint, increment-axis)
        d_rest = np.einsum("kap,kpmc->mcka", d_feature, self._PD_blocks[selected - 1], optimize=True)
        blended = (self.W @ Rw.reshape(len(Rw), 9)).reshape(self.num_markers, 3, 3)
        d_world = np.einsum("muc,mcka->muka", blended, d_rest, optimize=True)
        blocks = out[..., :-3].reshape(self.num_markers, 3, len(active), 3)
        blocks[:, :, slots, :] += d_world

    def joint_observability(self):
        """Max over markers of |d marker / d rotation| (m/rad) for each joint at rest pose."""
        R = np.broadcast_to(np.eye(3), (self.model.num_joints, 3, 3)).copy()
        _, cache = self.forward(R, np.zeros(3), with_cache=True)
        Jm = self.jacobian(cache, np.arange(self.model.num_joints))[..., :-3]
        Jm = Jm.reshape(len(Jm), 3, -1, 3)
        # smallest singular direction matters less; use per-joint Frobenius over markers
        return np.sqrt((Jm ** 2).sum(axis=(1, 3))).max(0)
