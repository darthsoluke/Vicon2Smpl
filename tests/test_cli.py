"""CLI integration test on synthetic data, without SMPL files or Vicon hardware:
record (synthetic .npz and .c3d) -> calibrate.py -> run_realtime.py --replay, with a fake aitviewer server,
a UDP receiver and --save. Checks the outputs against ground truth."""
import asyncio, os, pickle, socket, sys, tempfile, threading, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.synthetic import make_body, pick_markers, make_motion, simulate
from vsmpl import body_model
from vsmpl.sources import save_recording, VICON_TO_SMPL

MODEL, _ = make_body()
body_model.BodyModel.load = classmethod(lambda cls, *a, **k: MODEL)  # stand-in for the SMPL files


def fake_viewer(port, log):
    import websockets

    async def handler(ws):
        async for msg in ws:
            m = pickle.loads(msg)
            log.append((m["type"].name, {k: getattr(v, "shape", v) for k, v in m["kwargs"].items()}))

    async def main():
        async with websockets.serve(handler, "localhost", port, max_size=None):
            await asyncio.Future()

    threading.Thread(target=lambda: asyncio.run(main()), daemon=True).start()


def write_c3d(path, names, pos_smpl_m, fps):
    """Write a Z-up, millimetre C3D like Nexus/Shogun would."""
    import ezc3d
    c = ezc3d.c3d()
    c["parameters"]["POINT"]["RATE"]["value"] = [fps]
    c["parameters"]["POINT"]["LABELS"]["value"] = list(names)
    c["parameters"]["POINT"]["UNITS"]["value"] = ["mm"]
    xyz = pos_smpl_m @ VICON_TO_SMPL * 1000.0  # inverse rotation: SMPL -> Vicon (R^T applied as p @ R)
    T, M, _ = xyz.shape
    pts = np.ones((4, M, T))
    pts[:3] = np.transpose(np.nan_to_num(xyz, nan=0.0), (2, 1, 0))
    pts[:3, np.isnan(xyz[..., 0]).T] = np.nan
    c["data"]["points"] = pts
    c.write(path)


def main():
    tmp = tempfile.mkdtemp()
    fps = 100
    vids = pick_markers(MODEL, seed=5)
    betas = np.array([0.5, -0.7, 0.3, 0.6, -0.2, 0, 0, 0, 0, 0])
    Rc, tc = make_motion(3150, fps, seed=21, tpose_frames=150)
    obs_c, val_c, rig_true = simulate(MODEL, vids, betas, Rc, tc, seed=22)
    Rt, tt = make_motion(1500, fps, seed=23)
    obs_t, val_t, _ = simulate(MODEL, vids, betas, Rt, tt, seed=24)
    names = [f"LBL{i:02d}" for i in range(len(vids))]
    # calibration take as C3D (exercises the loader + Z-up/mm conversion), test take as npz
    calib_c3d = os.path.join(tmp, "calib.c3d")
    write_c3d(calib_c3d, names, obs_c, fps)
    test_npz = os.path.join(tmp, "test.npz")
    # shuffle marker order in the test file: the runtime must map by label
    perm = np.random.default_rng(0).permutation(len(names))
    save_recording(test_npz, [names[i] for i in perm], np.arange(len(obs_t)), obs_t[:, perm], val_t[:, perm], fps)

    import calibrate, run_realtime
    calib_out = os.path.join(tmp, "subject_calib.npz")
    sys.argv = ["calibrate.py", "--rec", calib_c3d, "--model-path", "unused", "--tpose", "0.1:1.4",
                "--out", calib_out]
    t0 = time.time()
    calibrate.main()
    print(f"[cli] calibrate.py took {time.time() - t0:.0f}s")

    log = []
    fake_viewer(18417, log)
    time.sleep(0.5)
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind(("127.0.0.1", 0))
    udp_sock.settimeout(0.5)
    udp_port = udp_sock.getsockname()[1]
    packets = []

    def rx():
        from vsmpl.outputs import UdpPublisher
        while True:
            try:
                packets.append(UdpPublisher.decode(udp_sock.recv(65536)))
            except socket.timeout:
                if done.is_set():
                    return
    done = threading.Event()
    threading.Thread(target=rx, daemon=True).start()

    out_npz = os.path.join(tmp, "solved.npz")
    sys.argv = ["run_realtime.py", "--calib", calib_out, "--model-path", "unused", "--replay", test_npz,
                "--viewer-host", "localhost:18417", "--udp", f"127.0.0.1:{udp_port}", "--save", out_npz, "--speed", "1.0"]
    run_realtime.main()
    time.sleep(0.6)
    done.set()

    # ---- checks
    sol = np.load(out_npz)
    calib = np.load(calib_out)
    from vsmpl.rotations import exp_so3
    rig_est = body_model.MarkerRig.from_calibration(MODEL, calib)
    fr = sol["frame_numbers"]
    err = []
    for k, f in enumerate(fr):
        if f < 30:
            continue
        R = exp_so3(sol["poses"][k].reshape(-1, 3))
        je = MODEL.joints(R, sol["trans"][k], rig_est.J_rest)[:22]
        jt = MODEL.joints(Rt[f], tt[f], rig_true.J_rest)[:22]
        err.append(np.linalg.norm(je - jt, axis=-1).mean())
    err = np.array(err) * 1000
    kinds = sorted({k for k, _ in log})
    print(f"[cli] solved {len(fr)} / {len(obs_t)} frames in real-time replay; MPJPE {err.mean():.1f} mm")
    print(f"[cli] UDP packets received: {len(packets)}, first pose shape {packets[0]['pose'].shape if packets else None}")
    print(f"[cli] viewer messages: {len(log)} ({kinds}); first: {log[0] if log else None}")
    assert err.mean() < 20 and len(packets) > 0.9 * len(fr) and "MESHES" in kinds and "UPDATE_FRAMES" in kinds
    print("[cli] OK")


if __name__ == "__main__":
    main()
