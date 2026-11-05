"""End-to-end on the SOMA body: automatic calibration (identity coefficients + marker attachments, no
correspondences given) from a static pose + ROM take, then real-time tracking of an unseen take.
Errors are measured on SOMA's own output: SOMALayer.pose(public poses, transl) vs ground truth.

    python tests/test_soma_pipeline.py [lod] [t|a] [seed]
"""
import io, contextlib, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.soma_synthetic import BODY_JOINT_NAMES, make_public_motion, pick_markers, simulate
from tests.test_soma import load, soma_reference
from vsmpl.body_model import MarkerRig
from vsmpl.calibration import calibrate
from vsmpl.solver import RealtimeSolver


def run(lod="low", static_pose="t", seed=0, verbose=False):
    model = load(lod)
    rng = np.random.default_rng(seed)
    betas_true = np.clip(rng.standard_normal(model.num_betas), -1.5, 1.5)
    vids = pick_markers(model, seed=seed + 1)
    fps = 100
    P_c, tr_c = make_public_motion(model, 3150, fps, seed=seed + 10, static_frames=150, static_pose=static_pose)
    obs_c, val_c, rig_true, _, _ = simulate(model, vids, betas_true, P_c, tr_c, seed=seed + 20)
    P_t, tr_t = make_public_motion(model, 3000, fps, seed=seed + 30)
    obs_t, val_t, _, _, _ = simulate(model, vids, betas_true, P_t, tr_t, seed=seed + 40)
    ref_joints = []
    for i in range(0, len(P_t), 10):
        ref_joints.append(soma_reference(model, betas_true, P_t[i], tr_t[i])[1])
    ref_joints = np.array(ref_joints)

    rec = {"names": rig_true.names, "pos": obs_c, "valid": val_c, "fps": fps}
    t0 = time.time()
    out = io.StringIO()
    with contextlib.redirect_stdout(out if not verbose else sys.stdout):
        calib = calibrate(model, rec, tpose=(10, 140), static_pose=static_pose, arm_angle_deg=45.0)
    t_cal = time.time() - t0
    if not verbose:
        print("\n".join(l for l in out.getvalue().splitlines() if "WARNING" in l or "hypothesis" in l or "refine" in l))
    rig = MarkerRig.from_calibration(model, calib)
    solver = RealtimeSolver(rig)
    sel = np.array([model.joint_names.index(n) for n in BODY_JOINT_NAMES])
    errs, ms, publics = [], [], []
    for i in range(len(P_t)):
        r = solver.step(obs_t[i], val_t[i], frame_no=i)
        ms.append(r.ms)
        if i % 10 == 0:
            publics.append(model.public_pose(r.pose_aa, r.trans))
    # evaluate through SOMA itself with the calibrated identity
    for k, (pose, transl) in enumerate(publics):
        if k < 3:
            continue
        joints = soma_reference(model, calib["betas"], pose, transl)[1]
        errs.append(np.linalg.norm(joints[sel] - ref_joints[k][sel], axis=-1).mean())
    errs = np.array(errs) * 1000
    v_true = model.exact_identity(betas_true)[0]
    v_est = model.exact_identity(calib["betas"])[0]
    shape_err = np.linalg.norm(v_true - v_est, axis=-1).mean() * 1000
    print(f"[soma-pipeline lod={lod} static={static_pose} seed={seed}] calib {t_cal:.0f}s | markers {len(vids)} | "
          f"active joints {len(solver.active)} | bind-shape vertex err {shape_err:.1f} mm | "
          f"test MPJPE {errs.mean():.1f} mm (p95 {np.percentile(errs, 95):.1f}) via SOMALayer | "
          f"solve {np.median(ms):.2f} ms median, {np.percentile(ms, 99):.2f} ms p99 | "
          f"calib marker rms median {np.median(calib['per_marker_rms']) * 1000:.1f} mm")
    return errs.mean(), shape_err


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0] if a else "low", a[1] if len(a) > 1 else "t", int(a[2]) if len(a) > 2 else 0, verbose="-v" in a)
