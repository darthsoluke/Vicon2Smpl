"""Real-time SMPL from Vicon markers.

Live:
    python -m aitviewer.server                       # terminal 1 (viewer)
    python run_realtime.py --calib subject_calib.npz --model-path ./body_models \
        --host 192.168.1.10:801 --subject Actor1     # terminal 2
Replay a recording (no Vicon needed):
    python run_realtime.py --calib subject_calib.npz --model-path ./body_models --replay take.npz
"""
import argparse
import time
import numpy as np

from vsmpl.body_model import BodyModel, MarkerRig
from vsmpl.solver import RealtimeSolver, SolverConfig
from vsmpl.sources import NameMapper, ReplaySource, ViconSource


def build(calib_path, model_path, solver_kw=None, model=None):
    calib = np.load(calib_path, allow_pickle=False)
    if model is None:
        model = BodyModel.load(model_path, str(calib["model_type"]), str(calib["gender"]), int(calib["num_betas"]))
    rig = MarkerRig.from_calibration(model, calib)
    cfg = SolverConfig(**(solver_kw or {}))
    solver = RealtimeSolver(rig, cfg)
    # warm-up (numba JIT compile) on a synthetic rest-pose frame, then reset
    R0 = np.broadcast_to(np.eye(3), (model.num_joints, 3, 3)).copy()
    p0 = rig.forward(R0, np.zeros(3))
    for _ in range(3):
        solver.step(p0, np.ones(rig.num_markers, bool))
    solver.reset()
    return calib, model, rig, solver


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--model-path", required=True)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--host", default="localhost:801", help="Vicon DataStream host:port (801, or 804 for Shogun low-latency)")
    src.add_argument("--replay", default=None, help="play an .npz/.c3d instead of connecting to Vicon")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--up-axis", default="z", choices=["z", "y"])
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed")
    ap.add_argument("--viz", default="aitviewer", choices=["aitviewer", "none"])
    ap.add_argument("--viewer-host", default="localhost:8417")
    ap.add_argument("--spawn-viewer", action="store_true", help="start `python -m aitviewer.server` automatically")
    ap.add_argument("--viz-fps", type=float, default=30)
    ap.add_argument("--udp", default=None, help="also stream poses as UDP packets to host:port (see vsmpl/outputs.py)")
    ap.add_argument("--save", default=None, help="save the solved sequence (AMASS-style .npz) on exit")
    ap.add_argument("--iters", type=int, default=4, help="LM iterations per frame")
    ap.add_argument("--lam-temp", type=float, default=1e-3, help="temporal smoothing weight")
    ap.add_argument("--fixed-joints", default="", help="comma-separated joint indices kept at rest (e.g. 22,23)")
    args = ap.parse_args()

    solver_kw = dict(iters=args.iters, lam_temp=args.lam_temp,
                     fixed_joints=[int(j) for j in args.fixed_joints.split(",") if j.strip()])
    calib, model, rig, solver = build(args.calib, args.model_path, solver_kw)
    print(f"[rt] {rig.num_markers} markers, {len(solver.active)} active joints: {solver.active.tolist()}")
    frozen = sorted(set(range(model.num_joints)) - set(solver.active.tolist()))
    if frozen:
        print(f"[rt] joints without marker support (kept at rest pose): {frozen}")

    source = ReplaySource(args.replay, args.speed, axis=args.up_axis) if args.replay else \
        ViconSource(args.host, args.subject, args.up_axis)
    mapper = NameMapper(rig.names)

    viz = None
    if args.viz == "aitviewer":
        from vsmpl.viz import AitviewerPublisher
        h, p = args.viewer_host.rsplit(":", 1)
        viz = AitviewerPublisher(model, rig, h, int(p), args.viz_fps, spawn=args.spawn_viewer)
    udp = None
    if args.udp:
        from vsmpl.outputs import UdpPublisher
        udp = UdpPublisher(args.udp)
    rec = None
    if args.save:
        from vsmpl.outputs import SequenceRecorder
        rec = SequenceRecorder(str(calib["model_type"]), str(calib["gender"]), calib["betas"], 0)

    seq, last_frame, dropped, n = -1, None, 0, 0
    stats_t, ms_hist, rmse_hist = time.time(), [], []
    print("[rt] running... Ctrl+C to stop")
    try:
        while True:
            seq, fr = source.get_latest(seq, timeout=2.0)
            if fr is None:
                if getattr(source, "done", False):
                    break
                print("[rt] waiting for data...")
                continue
            if last_frame is not None and fr["frame"] - last_frame > 1:
                dropped += fr["frame"] - last_frame - 1
            last_frame = fr["frame"]
            pos, valid = mapper(fr)
            res = solver.step(pos, valid, frame_no=fr["frame"])
            n += 1
            ms_hist.append(res.ms)
            if np.isfinite(res.rmse):
                rmse_hist.append(res.rmse)
            pose = res.pose_aa
            if viz:
                viz.push(res.R, res.trans, pos, valid)
            if udp:
                udp.send(fr["frame"], fr["time"], pose, res.trans, res.rmse)
            if rec:
                rec.add(fr["frame"], pose, res.trans, res.rmse)
            if time.time() - stats_t > 2.0:
                dt = time.time() - stats_t
                print(f"[rt] {n / dt:6.1f} Hz solved | solve {np.median(ms_hist):.2f} ms (p99 {np.percentile(ms_hist, 99):.2f}) | "
                      f"rmse {np.median(rmse_hist) * 1000 if rmse_hist else float('nan'):.1f} mm | "
                      f"markers {res.n_inliers}/{rig.num_markers} | skipped frames {dropped} | {res.status}")
                stats_t, n, ms_hist, rmse_hist, dropped = time.time(), 0, [], [], 0
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        if viz:
            viz.close()
        if rec:
            rec.meta["mocap_framerate"] = float(getattr(source, "frame_rate", 0) or 0)
            k = rec.save(args.save)
            print(f"[rt] saved {k} frames -> {args.save}")


if __name__ == "__main__":
    main()
