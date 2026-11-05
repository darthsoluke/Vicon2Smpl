"""CLI integration on the SOMA body: synthetic take -> calibrate.py --model-type soma -> run_realtime.py --replay
(--save, --udp). The saved file and the UDP packets must be in SOMA's published convention: feeding them to
SOMALayer.pose reproduces the ground-truth joints."""
import os, socket, sys, tempfile, threading, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.soma_synthetic import BODY_JOINT_NAMES, make_public_motion, pick_markers, simulate
from tests.test_soma import load, soma_reference
from vsmpl.sources import save_recording

LOD = sys.argv[1] if len(sys.argv) > 1 else "low"


def main():
    tmp = tempfile.mkdtemp()
    model = load(LOD)
    fps = 100
    betas = np.array([0.6, -0.8, 0.4, 0.5, -0.3, 0.2, 0, 0, 0, 0])
    vids = pick_markers(model, seed=5)
    Pc, tc = make_public_motion(model, 3150, fps, seed=21, static_frames=150, static_pose="a")
    obs_c, val_c, rig_true, _, _ = simulate(model, vids, betas, Pc, tc, seed=22)
    Pt, tt = make_public_motion(model, 1500, fps, seed=23)
    obs_t, val_t, _, _, _ = simulate(model, vids, betas, Pt, tt, seed=24)
    names = [f"LBL{i:02d}" for i in range(len(vids))]
    calib_npz, test_npz = os.path.join(tmp, "calib.npz"), os.path.join(tmp, "test.npz")
    save_recording(calib_npz, names, np.arange(len(obs_c)), obs_c, val_c, fps)
    perm = np.random.default_rng(0).permutation(len(names))
    save_recording(test_npz, [names[i] for i in perm], np.arange(len(obs_t)), obs_t[:, perm], val_t[:, perm], fps)

    import calibrate, run_realtime
    calib_out = os.path.join(tmp, "subject_calib.npz")
    sys.argv = ["calibrate.py", "--rec", calib_npz, "--model-type", "soma", "--model-path", "hf", "--soma-lod", LOD,
                "--static-pose", "a", "--tpose", "0.1:1.4", "--out", calib_out]
    t0 = time.time()
    calibrate.main()
    print(f"[soma-cli] calibrate.py took {time.time() - t0:.0f}s")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(0.5)
    packets, done = [], threading.Event()

    def rx():
        from vsmpl.outputs import UdpPublisher
        while not done.is_set():
            try:
                packets.append(UdpPublisher.decode(sock.recv(65536)))
            except socket.timeout:
                pass
    threading.Thread(target=rx, daemon=True).start()
    out_npz = os.path.join(tmp, "solved.npz")
    sys.argv = ["run_realtime.py", "--calib", calib_out, "--model-path", "hf", "--replay", test_npz, "--viz", "none",
                "--udp", f"127.0.0.1:{sock.getsockname()[1]}", "--save", out_npz]
    run_realtime.main()
    time.sleep(0.6)
    done.set()

    sol = np.load(out_npz)
    assert str(sol["model_type"]) == "soma" and sol["poses"].shape[1] == 77 * 3
    coeffs = sol["identity_coeffs"]
    assert coeffs.shape == (model.num_identity_total,)
    sel = np.array([model.joint_names.index(n) for n in BODY_JOINT_NAMES])
    errs, udp_diff = [], []
    by_frame = {p["frame"]: p for p in packets}
    for k in range(0, len(sol["frame_numbers"]), 15):
        f = int(sol["frame_numbers"][k])
        if f < 30:
            continue
        pose = sol["poses"][k].reshape(77, 3)
        joints = soma_reference(model, coeffs[: model.num_betas], pose, sol["trans"][k])[1]
        truth = soma_reference(model, betas, Pt[f], tt[f])[1]
        errs.append(np.linalg.norm(joints[sel] - truth[sel], axis=-1).mean())
        if f in by_frame:
            udp_diff.append(max(np.abs(by_frame[f]["pose"] - pose).max(), np.abs(by_frame[f]["trans"] - sol["trans"][k]).max()))
    errs = np.array(errs) * 1000
    print(f"[soma-cli] solved {len(sol['frame_numbers'])}/{len(obs_t)} frames; MPJPE through SOMALayer {errs.mean():.1f} mm; "
          f"UDP packets {len(packets)}, max |UDP - saved| {max(udp_diff):.2e}")
    assert errs.mean() < 20 and len(packets) > 0.9 * len(sol["frame_numbers"]) and max(udp_diff) < 1e-5
    print("[soma-cli] OK")


if __name__ == "__main__":
    main()
