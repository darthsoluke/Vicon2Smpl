"""Record labelled markers from Vicon into an .npz (metres, Y-up) for calibration / replay.

    python record.py --host 192.168.1.10:801 --subject Actor1 --out calib_session.npz
Press Ctrl+C to stop. Suggested calibration take: ~3 s static T-pose, then 30-60 s range-of-motion
(arm circles, squats, torso twists/bends, leg swings, neck rotations).
"""
import argparse
import time
import numpy as np

from vsmpl.sources import ViconSource, save_recording


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost:801")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--up-axis", default="z", choices=["z", "y"], help="capture up axis (Vicon default: z)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seconds", type=float, default=0, help="stop automatically (0 = until Ctrl+C)")
    args = ap.parse_args()

    src = ViconSource(args.host, args.subject, args.up_axis)
    names, frames, pos, valid = None, [], [], []
    seq, t0, last_print = -1, time.time(), 0
    print("[record] recording... Ctrl+C to stop")
    try:
        while True:
            seq, fr = src.get_latest(seq, timeout=2.0)
            if fr is None:
                print("[record] no data")
                continue
            if names is None:
                names = fr["names"]
            if fr["names"] != names:  # labels changed (subject re-calibrated?) -> map by name
                lut = {n: i for i, n in enumerate(fr["names"])}
                idx = [lut.get(n, -1) for n in names]
                p = np.full((len(names), 3), np.nan); v = np.zeros(len(names), bool)
                for k, i in enumerate(idx):
                    if i >= 0:
                        p[k], v[k] = fr["pos"][i], fr["valid"][i]
            else:
                p, v = fr["pos"], fr["valid"]
            frames.append(fr["frame"]); pos.append(p); valid.append(v)
            if time.time() - last_print > 1:
                last_print = time.time()
                print(f"\r[record] {len(frames)} frames, {int(np.sum(v))}/{len(v)} markers visible", end="")
            if args.max_seconds and time.time() - t0 > args.max_seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
    if frames:
        save_recording(args.out, names, frames, pos, valid, src.frame_rate)
        print(f"\n[record] saved {len(frames)} frames, {len(names)} markers -> {args.out}")


if __name__ == "__main__":
    main()
