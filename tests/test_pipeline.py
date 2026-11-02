"""End-to-end on synthetic data: calibrate from a T-pose + ROM take (labels known, correspondences unknown),
then track a different, unseen take in real time and compare to ground-truth joints."""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.synthetic import make_body, pick_markers, make_motion, simulate
from vsmpl.body_model import MarkerRig
from vsmpl.calibration import calibrate
from vsmpl.solver import RealtimeSolver, SolverConfig


def run(seed=0, facing="auto", verbose=True):
    model, _ = make_body()
    vids = pick_markers(model, seed=seed + 1)
    rng = np.random.default_rng(seed)
    betas_true = np.concatenate([rng.uniform(-1.2, 1.2, 5), np.zeros(5)])
    fps = 100
    # calibration take: 1.5 s T-pose + 30 s ROM
    Rc, tc = make_motion(3150, fps, seed=seed + 10, tpose_frames=150)
    obs_c, val_c, rig_true = simulate(model, vids, betas_true, Rc, tc, seed=seed + 20)
    rec = {"names": rig_true.names, "pos": obs_c, "valid": val_c, "fps": fps}
    t0 = time.time()
    calib = calibrate(model, rec, tpose=(10, 140), facing=facing, n_frames=40, rounds=2, verbose=verbose)
    t_cal = time.time() - t0
    rig = MarkerRig(model, list(calib["marker_names"]), calib["vertex_ids"], calib["offsets"], calib["betas"])

    # unseen test take
    Rt, tt = make_motion(3000, fps, seed=seed + 30)
    obs_t, val_t, _ = simulate(model, vids, betas_true, Rt, tt, seed=seed + 40)
    solver = RealtimeSolver(rig, SolverConfig())
    errs, ms = [], []
    for i in range(len(Rt)):
        r = solver.step(obs_t[i], val_t[i], frame_no=i)
        jt = model.joints(Rt[i], tt[i], rig_true.J_rest)[:22]
        je = model.joints(r.R, r.trans, rig.J_rest)[:22]
        errs.append(np.linalg.norm(jt - je, axis=-1).mean())
        ms.append(r.ms)
    errs = np.array(errs[30:]) * 1000
    # shape error: rest-pose vertex distance between true and estimated body
    v_true, _ = model.shaped(betas_true)
    v_est, _ = model.shaped(calib["betas"])
    shape_err = np.linalg.norm(v_true - v_est, axis=-1).mean() * 1000
    print(f"[pipeline seed={seed}] calib {t_cal:.0f}s | rest-shape vertex err {shape_err:.1f} mm | "
          f"test MPJPE {errs.mean():.1f} mm (p95 {np.percentile(errs, 95):.1f}) | "
          f"solve {np.median(ms):.2f} ms median, {np.percentile(ms, 99):.2f} ms p99")
    return errs.mean(), shape_err


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [0]
    for s in seeds:
        run(s)
