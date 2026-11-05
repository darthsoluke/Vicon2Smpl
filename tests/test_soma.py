"""SOMA backend checks against NVIDIA's own SOMALayer (needs `pip install py-soma-x`; assets download once).

1. Forward parity: our SMPL-form re-expression vs SOMALayer.pose for random identities and poses.
2. Public parameter round trip and the T-/A-pose reference.
3. Identity linearisation error away from the centre, and exactness after prepare_for().
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vsmpl.body_model import BodyModel, MarkerRig
from vsmpl.rotations import exp_so3, log_so3


def load(lod="mid", num_betas=10):
    t = time.time()
    model = BodyModel.load("hf", "soma", num_betas=num_betas, soma_lod=lod)
    print(f"[soma] {lod}: {len(model.v_template)} vertices, {model.num_joints} joints, "
          f"{model.num_betas}/{model.num_identity_total} identity coefficients, loaded in {time.time() - t:.1f}s")
    return model


def soma_reference(model, betas, poses, transl):
    import torch
    with torch.no_grad():
        model.layer.prepare_identity(torch.tensor(model._full_coeffs(betas)[None], dtype=torch.float32))
        out = model.layer.pose(torch.tensor(poses[None], dtype=torch.float32),
                               transl=torch.tensor(transl[None], dtype=torch.float32), apply_correctives=False)
    return out["vertices"][0].double().numpy(), out["joints"][0].double().numpy()


def test_parity(model):
    rng = np.random.default_rng(0)
    worst_v = worst_j = 0.0
    for trial in range(4):
        betas = rng.standard_normal(model.num_betas) * 0.8
        model.prepare_for(betas)
        poses = rng.standard_normal((model.num_joints, 3)) * 0.35
        transl = rng.standard_normal(3)
        R, trans = model.from_public_pose(poses, transl)
        v_rest, J_rest = model.shaped(betas)
        ours_v = model.vertices(R, trans, v_rest, J_rest)
        ours_j = model.joints(R, trans, J_rest)
        ref_v, ref_j = soma_reference(model, betas, poses, transl)
        worst_v = max(worst_v, np.abs(ours_v - ref_v).max())
        worst_j = max(worst_j, np.abs(ours_j - ref_j).max())
        # round trip back to SOMA's public parameters
        back_pose, back_transl = model.public_pose(log_so3(R), trans)
        assert np.abs(exp_so3(back_pose) - exp_so3(poses)).max() < 1e-9 and np.abs(back_transl - transl).max() < 1e-12
        # marker rig (arbitrary vertices + offsets) is consistent with the mesh
        vids = rng.choice(len(v_rest), 40, replace=False)
        rig = MarkerRig(model, [str(i) for i in vids], vids, np.zeros((40, 3)), betas)
        assert np.abs(rig.forward(R, trans) - ours_v[vids]).max() < 1e-10
    print(f"[soma] forward parity vs SOMALayer.pose: max vertex diff {worst_v * 1000:.4f} mm, joint {worst_j * 1000:.4f} mm")
    assert worst_v < 1e-4 and worst_j < 1e-4


def test_reference_poses(model):
    model.prepare_for(np.zeros(model.num_betas))
    names = model.joint_names
    v_rest, J_rest = model.shaped(model.center)
    for kind, expect in (("t", 0.0), ("a", 45.0)):
        R = exp_so3(model.static_reference_pose(kind, 45.0))
        J = model.joints(R, np.zeros(3), J_rest)
        arm = J[names.index("LeftHand")] - J[names.index("LeftArm")]
        drop = np.degrees(np.arctan2(-arm[1], arm[0]))
        arm_r = J[names.index("RightHand")] - J[names.index("RightArm")]
        drop_r = np.degrees(np.arctan2(-arm_r[1], -arm_r[0]))
        print(f"[soma] {kind.upper()}-pose reference: left arm {drop:.1f} deg below horizontal (+x), right {drop_r:.1f} deg (-x)")
        assert abs(drop - expect) < 3 and abs(drop_r - expect) < 3 and arm[0] > 0 > arm_r[0]
    bind = model.joints(np.broadcast_to(np.eye(3), (model.num_joints, 3, 3)), np.zeros(3), J_rest)
    arm = bind[names.index("LeftHand")] - bind[names.index("LeftArm")]
    print(f"[soma] bind (solver rest) pose: left arm {np.degrees(np.arctan2(-arm[1], arm[0])):.1f} deg below horizontal")


def test_linearisation(model):
    rng = np.random.default_rng(3)
    model.prepare_for(np.zeros(model.num_betas))
    errs = []
    for _ in range(4):
        betas = rng.standard_normal(model.num_betas)
        v_lin, j_lin = model.shaped(betas)
        v_ex, j_ex, _ = model.exact_identity(betas)
        errs.append((np.abs(v_lin - v_ex).max(), np.abs(j_lin - j_ex).max()))
    errs = np.array(errs) * 1000
    print(f"[soma] linearised at 0, 1-sigma identities: max vertex err {errs[:, 0].max():.2f} mm, joint {errs[:, 1].max():.2f} mm")
    model.prepare_for(betas)
    v_lin, _ = model.shaped(betas)
    assert np.abs(v_lin - model.exact_identity(betas)[0]).max() < 1e-9


if __name__ == "__main__":
    m = load(sys.argv[1] if len(sys.argv) > 1 else "mid")
    test_parity(m)
    test_reference_poses(m)
    test_linearisation(m)
    print("[soma] OK")
