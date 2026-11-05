"""NVIDIA SOMA (SOMA-X, ``pip install py-soma-x``) as an SMPL-form body for the marker solver.

SOMA poses its bind-pose mesh with plain linear blend skinning over a 78-joint public skeleton (joint 0 is a
virtual, always-identity ``Root``). Its public pose parameters are local rotations *relative to the T-pose joint
orient*:

    R_abs[j] = O[p]^T R_rel[j] O[j]            O = T-pose world orientations, p = parent(j)
    W[j]     = W[p] R_abs[j]                    world joint rotation
    v'       = sum_j w_vj (W[j] B[j]^T (v - b[j]) + x[j])     B, b = identity-fitted bind rotations/positions

Re-parameterising relative to the *bind* orient, R'[j] = B[p] R_abs[j] B[j]^T, gives S[j] = W[j] B[j]^T with
S[j] = S[p] R'[j] and x[j] - x[p] = S[p] (b[j] - b[p]) - exactly SMPL forward kinematics and LBS, with rest mesh
= bind mesh (an A-pose), rest joints = b and no joint orient. So the existing analytic-Jacobian LM solver, the
numba kernels and the calibration run unchanged; only the published parameters are converted back:

    R_rel[j] = C[p] R'[j] C[j]^T,   C[j] = O[j] B[j]^T        transl (Hips position) = trans + b[Hips]

The 77 user joints are kept (Hips is the solver root). Vertices match ``SOMALayer.pose(apply_correctives=False)``
of a layer built with ``enable_procedural_transforms=False`` (SOMA's 78-joint LBS rig) to float32 precision.
SOMA's default procedural twist joints and learned pose correctives are not part of the fitting model.

Identity: SOMA's rest shape is (almost) linear in its identity coefficients (<3 mm deviation at +-2 sigma for
the native SOMA PCA). The model is linearised around a centre by central finite differences through SOMA itself;
``prepare_for(betas)`` re-centres, so a rig built for a calibration's coefficients is exact.
"""
import numpy as np
from scipy import sparse

from .body_model import BodyModel
from .rotations import exp_so3, log_so3

SOMA_LODS = ("mid", "low", "xlo")
TORSO_NAMES = ("Hips", "Spine1", "Spine2", "Chest")
# Joints whose static-pose angle varies most between people (lower confidence in static-pose priors).
_LOW_CONFIDENCE_PARTS = ("Shoulder", "Arm", "Hand", "Jaw", "Eye", "End")


def nearest_rotation(R):
    """Project (...,3,3) matrices onto SO(3); SOMA stores float32 transforms."""
    U, _, Vt = np.linalg.svd(R)
    D = np.ones(R.shape[:-1])
    D[..., -1] = np.sign(np.linalg.det(U @ Vt))
    return (U * D[..., None, :]) @ Vt


class SomaBodyModel(BodyModel):
    rest_is_tpose = False  # identity rotations reproduce SOMA's bind A-pose, not its T-pose

    def __init__(self, layer, num_betas=10, identity_model_type="soma", lod="mid", data_root=None, fd_step=0.5):
        import torch

        self.layer = layer
        self.identity_model_type = identity_model_type
        self.lod = lod
        self.data_root = None if data_root is None else str(data_root)
        self.fd_step = float(fd_step)
        self.num_identity_total = int(layer.num_shape_components)
        if not 1 <= num_betas <= self.num_identity_total:
            raise ValueError(f"num_betas must be between 1 and {self.num_identity_total} for this SOMA identity model")
        names = [str(n) for n in layer.public_joint_names]
        parents_public = layer.public_joint_parent_ids.detach().cpu().numpy().astype(np.int64)
        if names[0] != "Root" or parents_public[1] != 0 or names[1] != "Hips":
            raise ValueError("Unexpected SOMA public skeleton (expected virtual Root followed by Hips)")
        with torch.no_grad():
            layer.prepare_identity(torch.zeros(1, self.num_identity_total))
            view = layer.public_rig_view()
            weights = view.skinning_weights.detach().double().cpu().numpy()
            self._orient = nearest_rotation(view.t_pose_world.detach().double().cpu().numpy()[1:, :3, :3])  # O
        if np.abs(weights[:, 0]).max() > 1e-8:
            raise ValueError("SOMA virtual Root carries skinning weight; it cannot be dropped")
        parents = np.concatenate([[-1], parents_public[2:] - 1])
        faces = layer.faces.detach().cpu().numpy().astype(np.int64)
        J = len(names) - 1
        V = len(weights)
        super().__init__(np.zeros((V, 3)), np.zeros((V, 3, num_betas)), None, None, weights[:, 1:], parents, faces,
                         model_type="soma", J_template=np.zeros((J, 3)), J_shapedirs=np.zeros((J, 3, num_betas)))
        self.joint_names = names[1:]
        index = {n: i for i, n in enumerate(self.joint_names)}
        self.torso_joints = tuple(index[n] for n in TORSO_NAMES)
        self.head_joint = index["Head"]
        self._W_sparse = sparse.csr_matrix(self.weights)
        self.center = None
        self.recenter(np.zeros(num_betas))

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, data_root=None, num_betas=10, lod="mid", identity_model_type="soma", identity_model_kwargs=None):
        """`data_root`: SOMA asset folder, or None / "hf" / "auto" to use the HuggingFace cache (downloads once)."""
        try:
            from soma import SOMALayer
        except ImportError as error:
            raise ImportError("The SOMA backend needs `pip install py-soma-x` (Apache-2.0, NVIDIA SOMA-X).") from error
        if lod not in SOMA_LODS:
            raise ValueError(f"SOMA lod must be one of {SOMA_LODS}")
        if data_root in (None, "", "hf", "auto"):
            data_root = None
        layer = SOMALayer(data_root=data_root, device="cpu", mode="torch", lod=lod,
                          identity_model_type=identity_model_type, identity_model_kwargs=identity_model_kwargs,
                          enable_procedural_transforms=False, correctives_model_path=None)
        return cls(layer, num_betas=num_betas, identity_model_type=identity_model_type, lod=lod, data_root=data_root)

    # ------------------------------------------------------------------ identity
    def _full_coeffs(self, betas):
        full = np.zeros(self.num_identity_total)
        b = np.asarray(betas, np.float64).ravel()[: self.num_betas]
        full[: len(b)] = b
        return full

    def exact_identity(self, betas):
        """SOMA's own identity pipeline: bind mesh (V,3), bind joint positions (J,3) and orientations (J,3,3)."""
        import torch

        with torch.no_grad():
            self.layer.prepare_identity(torch.tensor(self._full_coeffs(betas)[None], dtype=torch.float32))
            bind = self.layer.public_rig_view().bind_transforms_world[0, 1:].detach().double().cpu().numpy()
            verts = self.layer._cached_rest_shape[0].detach().double().cpu().numpy()
        return verts, bind[:, :3, 3].copy(), nearest_rotation(bind[:, :3, :3])

    def recenter(self, betas, jacobian=True):
        """Linearise the identity model at `betas` (exact there); finite-difference shape directions."""
        betas = np.asarray(betas, np.float64).ravel()[: self.num_betas]
        v_c, b_c, B_c = self.exact_identity(betas)
        if jacobian:
            h = self.fd_step
            for k in range(self.num_betas):
                e = np.zeros(self.num_betas)
                e[k] = h
                vp, bp, _ = self.exact_identity(betas + e)
                vm, bm, _ = self.exact_identity(betas - e)
                self.shapedirs[:, :, k] = (vp - vm) / (2 * h)
                self.J_shapedirs[:, :, k] = (bp - bm) / (2 * h)
        self.v_template = v_c - self.shapedirs @ betas
        self.J_template = b_c - self.J_shapedirs @ betas
        self.bind_rotations = B_c
        self.joint_orient_change = self._orient @ np.transpose(B_c, (0, 2, 1))  # C = O B^T
        self.center = betas.copy()

    def prepare_for(self, betas):
        betas = np.asarray(betas, np.float64).ravel()[: self.num_betas]
        if self.center is None or len(self.center) != len(betas) or not np.allclose(betas, self.center, atol=1e-10):
            self.recenter(betas)

    # ------------------------------------------------------------------ public SOMA convention
    def _parent_change(self):
        C = self.joint_orient_change
        Cp = np.empty_like(C)
        Cp[0] = np.eye(3)
        Cp[1:] = C[self.parents[1:]]
        return C, Cp

    def to_public_rotations(self, R):
        """Bind-relative solver rotations (77,3,3) -> SOMA T-pose-relative public rotations."""
        C, Cp = self._parent_change()
        return Cp @ np.asarray(R) @ np.transpose(C, (0, 2, 1))

    def from_public_rotations(self, R_public):
        C, Cp = self._parent_change()
        return np.transpose(Cp, (0, 2, 1)) @ np.asarray(R_public) @ C

    def hips_rest_position(self):
        return self.J_template[0] + self.J_shapedirs[0] @ self.center

    def public_pose(self, pose_aa, trans):
        """-> (poses (77,3) axis-angle for SOMALayer.pose, transl (3,) Hips position in metres)."""
        R = exp_so3(np.asarray(pose_aa, np.float64).reshape(-1, 3))
        return log_so3(self.to_public_rotations(R)), np.asarray(trans, np.float64) + self.hips_rest_position()

    def from_public_pose(self, poses, transl):
        R = self.from_public_rotations(exp_so3(np.asarray(poses, np.float64).reshape(-1, 3)))
        return R, np.asarray(transl, np.float64) - self.hips_rest_position()

    # ------------------------------------------------------------------ skeleton roles for calibration priors
    def static_reference_pose(self, static_pose="t", arm_angle_deg=45.0):
        """T- or A-pose in solver (bind-relative) axis-angle. Public T-relative frames are world-aligned in the
        T-pose, exactly like SMPL, so the A-pose lowers LeftArm/RightArm about +-z as in the SMPL path."""
        if static_pose not in ("t", "a"):
            raise ValueError("static_pose must be 't' or 'a'")
        if not 0 <= arm_angle_deg <= 90:
            raise ValueError("arm_angle_deg must be between 0 and 90")
        public = np.zeros((self.num_joints, 3))
        if static_pose == "a":
            public[self.joint_names.index("LeftArm"), 2] = -np.radians(arm_angle_deg)
            public[self.joint_names.index("RightArm"), 2] = np.radians(arm_angle_deg)
        return log_so3(self.from_public_rotations(exp_so3(public)))

    def static_prior_weights(self, arm_w=0.1):
        return np.array([arm_w if any(part in name for part in _LOW_CONFIDENCE_PARTS) else 1.0
                         for name in self.joint_names[1:]])

    def calibration_metadata(self):
        import soma

        return dict(soma_identity_model=self.identity_model_type, soma_lod=self.lod,
                    soma_num_identity_total=self.num_identity_total, soma_joint_names=np.array(self.joint_names),
                    soma_version=str(getattr(soma, "__version__", "")),
                    pose_convention="solver poses are bind-relative; public SOMA poses via SomaBodyModel.public_pose")

    def recording_metadata(self, betas):
        return dict(identity_coeffs=self._full_coeffs(betas), identity_model_type=self.identity_model_type,
                    soma_lod=self.lod, soma_joint_names=np.array(self.joint_names),
                    pose_convention="SOMALayer(enable_procedural_transforms=False).prepare_identity(identity_coeffs[None]); "
                                    "pose(poses.reshape(-1,77,3), transl=trans, apply_correctives=False); "
                                    "poses are relative to the T-pose joint orient, trans is the Hips position (m)")

    # ------------------------------------------------------------------ faster dense mesh for display
    def vertices(self, R, trans, v_rest, J_rest):
        Rw, c = self.fk(R, J_rest)
        At = c - np.einsum("jab,jb->ja", Rw, J_rest)
        TR = (self._W_sparse @ Rw.reshape(len(Rw), 9)).reshape(-1, 3, 3)
        return np.einsum("vab,vb->va", TR, v_rest) + self._W_sparse @ At + trans


def check_calibration_model(model, calib):
    """Refuse to pair a SOMA calibration with a differently configured model (or vice versa)."""
    is_soma = isinstance(model, SomaBodyModel)
    calib_soma = "soma_lod" in calib
    if is_soma != calib_soma:
        raise ValueError("Calibration and body model disagree on the SOMA backend; recalibrate or change --model-type")
    if is_soma:
        if str(calib["soma_lod"]) != model.lod or str(calib["soma_identity_model"]) != model.identity_model_type:
            raise ValueError(f"Calibration was made with SOMA lod={calib['soma_lod']} identity={calib['soma_identity_model']}; "
                             f"the loaded model uses lod={model.lod} identity={model.identity_model_type}")
        if len(np.asarray(calib["betas"])) != model.num_betas:
            raise ValueError("Calibration identity coefficient count differs from the loaded SOMA model")
