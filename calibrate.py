"""Subject calibration: estimate SMPL betas and marker placement from a recorded take.

    python calibrate.py --rec calib_session.npz --model-path ./body_models --model-type smpl \
        --gender neutral --tpose 0:2.0 --out subject_calib.npz

--rec accepts .npz from record.py or a .c3d exported from Nexus/Shogun.
--tpose is a time window in seconds (start:end) during which the subject stands still in a T-pose.
"""
import argparse
import numpy as np

from vsmpl.body_model import BodyModel
from vsmpl.calibration import calibrate
from vsmpl.sources import load_recording


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", required=True, help=".npz (record.py) or .c3d")
    ap.add_argument("--up-axis", default="z", choices=["z", "y"], help="up axis of a .c3d file")
    ap.add_argument("--model-path", required=True, help="folder containing smpl/ smplx/ ... model files")
    ap.add_argument("--model-type", default="smpl", choices=["smpl", "smplh", "smplx"])
    ap.add_argument("--gender", default="neutral")
    ap.add_argument("--num-betas", type=int, default=10)
    ap.add_argument("--tpose", default="0:1.0", help="static T-pose window in seconds, e.g. 0.5:2.5")
    ap.add_argument("--static-pose", default="t", choices=["t", "a"], help="pose held in the static window")
    ap.add_argument("--arm-angle", type=float, default=45.0, help="A-pose shoulder lowering initialisation, degrees")
    ap.add_argument("--facing", default="auto", choices=["auto", "+z", "-z", "+x", "-x"],
                    help="direction the subject faces during the T-pose, in SMPL world axes "
                         "(Vicon +Y -> SMPL -Z, Vicon +X -> SMPL +X)")
    ap.add_argument("--marker-map", default=None, help="optional JSON {label: vertex_id}; skips auto-assignment")
    ap.add_argument("--marker-regions", help="optional JSON {label: [allowed vertex IDs]}; explicit anatomical constraints")
    ap.add_argument("--marker-radius", type=float, default=0.0095, help="marker centre to skin distance (m)")
    ap.add_argument("--n-frames", type=int, default=40, help="poses used in the joint refinement")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rec = load_recording(args.rec, axis=args.up_axis)
    fps = rec["fps"] or 100.0
    a, b = (float(x) for x in args.tpose.split(":"))
    tpose = (int(a * fps), max(int(b * fps), int(a * fps) + 1))
    print(f"[calib] {len(rec['pos'])} frames @ {fps:.0f} Hz, {len(rec['names'])} markers")

    model = BodyModel.load(args.model_path, args.model_type, args.gender, args.num_betas)
    res = calibrate(model, rec, tpose=tpose, facing=args.facing, marker_map=args.marker_map,
                    marker_radius=args.marker_radius, n_frames=args.n_frames, rounds=args.rounds,
                    static_pose=args.static_pose, arm_angle_deg=args.arm_angle, marker_regions=args.marker_regions)
    np.savez(args.out, model_type=args.model_type, gender=args.gender, num_betas=args.num_betas, **res)
    print(f"[calib] betas = {np.round(res['betas'], 3).tolist()}")
    print(f"[calib] per-marker rms (mm): median {np.median(res['per_marker_rms']) * 1000:.1f}, "
          f"max {res['per_marker_rms'].max() * 1000:.1f}")
    print(f"[calib] saved -> {args.out}")


if __name__ == "__main__":
    main()
