"""Review saved full-body/hand candidates with native FK and full native LBS.

Only the rendered triangles are restricted to the hand. Skinning weights,
ancestors, saved translations and source marker coordinates are never cropped,
refitted or independently aligned. The same native wrist transform is applied
to mesh, observations, fitted markers and joint chains in each panel.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vsmpl.body_model import BodyModel, MarkerRig
from vsmpl.sources import load_recording


HAND_SUFFIXES = ("IWR", "OWR", "IHAND", "OHAND", "THM3", "THM6", "IDX3", "IDX6",
                 "MID0", "MID6", "RNG3", "RNG6", "PNK3", "PNK6")
LEFT_CHAINS = ((20, 25, 26, 27), (20, 28, 29, 30), (20, 31, 32, 33),
               (20, 34, 35, 36), (20, 37, 38, 39))
RIGHT_CHAINS = tuple(tuple(21 if joint == 20 else joint + 15 for joint in chain) for chain in LEFT_CHAINS)


def _choose_frames(frames, times, rec):
    if len(frames) < 3:
        raise ValueError("At least three fitted hand frames are needed for the review")
    names = list(rec["names"])
    score = np.full(len(frames), np.nan)
    for row, index in enumerate(frames):
        distances = []
        for side in ("L", "R"):
            wrist_ids = [names.index(side + suffix) for suffix in ("IWR", "OWR")]
            finger_ids = [names.index(side + suffix) for suffix in HAND_SUFFIXES[4:]]
            wrist_valid = rec["valid"][index, wrist_ids]
            finger_valid = rec["valid"][index, finger_ids]
            if not wrist_valid.any() or not finger_valid.any():
                continue
            wrist = rec["pos"][index, np.array(wrist_ids)[wrist_valid]].mean(0)
            finger = rec["pos"][index, np.array(finger_ids)[finger_valid]]
            distances.extend(np.linalg.norm(finger - wrist, axis=1).tolist())
        if distances:
            score[row] = np.mean(distances)
    chosen = [int(np.argmin(np.abs(times[frames] - 1.5)))]
    valid_rows = np.flatnonzero(np.isfinite(score))
    for ranking in (valid_rows[np.argsort(score[valid_rows])], valid_rows[np.argsort(-score[valid_rows])]):
        next_row = next((int(row) for row in ranking if int(row) not in chosen), None)
        if next_row is not None:
            chosen.append(next_row)
    for row in (0, len(frames)//2, len(frames)-1, *range(len(frames))):
        if len(chosen) == 3:
            break
        if row not in chosen:
            chosen.append(row)
    return chosen


def _hand_face_mask(model, rig, side):
    wrist = 20 if side == "L" else 21
    fingers = np.arange(25, 40) if side == "L" else np.arange(40, 55)
    owner = model.weights.argmax(1)
    visible = np.isin(owner, np.r_[wrist, fingers])
    # A small forearm display band retains the visible wrist attachment. This
    # is a triangle display mask applied after complete native deformation.
    forearm = 18 if side == "L" else 19
    visible |= ((owner == forearm)
                & (np.linalg.norm(rig.v_rest - rig.J_rest[wrist], axis=1) < .055))
    selected = model.faces[np.all(visible[model.faces], axis=1)]
    if not len(selected):
        raise ValueError("No native hand triangles for " + side)
    return selected


def _project(points, side, view):
    # Fixed native wrist-coordinate camera, equally applied to every object.
    sign = 1. if side == "L" else -1.
    if view == "a":
        return np.c_[points[:, 2], sign * points[:, 0]], points[:, 1]
    return np.c_[points[:, 1], sign * points[:, 0]], points[:, 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    args.output.mkdir(parents=True, exist_ok=True)
    targets = [args.output / name for name in ("hands_view_a.png", "hands_view_b.png", "review_selection.json")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Review outputs already exist; choose a new output directory")
    rec = load_recording(args.source_run / "calibration_markers.npz")
    with np.load(args.source_run / "sampling_clock.npz", allow_pickle=False) as clock:
        times = clock["relative_host_arrival_seconds"].copy()
        source_frames = clock["source_frame_numbers"].copy() if "source_frame_numbers" in clock else None
    with np.load(args.calib, allow_pickle=False) as archive:
        calib = {name: archive[name].copy() for name in archive.files}
    model = BodyModel.load(args.model_path, str(calib.get("model_type", "smplx")),
                           str(calib.get("gender", "neutral")), int(calib.get("num_betas", 10)))
    if model.num_joints != 55:
        raise ValueError("This review requires the native 55-joint SMPL-X model")
    rig = MarkerRig.from_calibration(model, calib)
    frames = np.asarray(calib["selected_hand_frames"])
    R = np.asarray(calib["selected_hand_R"])
    trans = np.asarray(calib["selected_hand_trans"])
    if (frames.ndim != 1 or not np.issubdtype(frames.dtype, np.integer)
            or len(set(frames.tolist())) != len(frames) or np.any(frames < 0) or np.any(frames >= len(rec["pos"]))
            or R.shape != (len(frames), 55, 3, 3) or trans.shape != (len(frames), 3)
            or len(times) != len(rec["pos"]) or not np.isfinite(R).all() or not np.isfinite(trans).all()):
        raise ValueError("Saved hand frames, native poses, translations and source clock do not agree")
    raw_names, fit_names = list(rec["names"]), list(rig.names)
    for side in ("L", "R"):
        required = [side + suffix for suffix in HAND_SUFFIXES]
        missing = [name for name in required if name not in raw_names or name not in fit_names]
        if missing:
            raise ValueError("Missing hand markers in source or calibration: " + ", ".join(missing))
    chosen = _choose_frames(frames, times, rec)
    faces = {side: _hand_face_mask(model, rig, side) for side in ("L", "R")}
    panels = []
    diagnostics = []
    for row in chosen:
        index = int(frames[row])
        # Evaluate the COMPLETE model before restricting triangle visibility.
        world_vertices = model.vertices(R[row], trans[row], rig.v_rest, rig.J_rest)
        world_rotations, joint_positions = model.fk(R[row], rig.J_rest)
        world_joints = joint_positions + trans[row]
        world_fit = rig.forward(R[row], trans[row])
        for side in ("L", "R"):
            labels = [side + suffix for suffix in HAND_SUFFIXES]
            raw_ids = np.array([raw_names.index(name) for name in labels])
            fit_ids = np.array([fit_names.index(name) for name in labels])
            valid = rec["valid"][index, raw_ids] & np.isfinite(rec["pos"][index, raw_ids]).all(1)
            observed = rec["pos"][index, raw_ids]
            fitted = world_fit[fit_ids]
            wrist = 20 if side == "L" else 21
            origin, rotation = world_joints[wrist], world_rotations[wrist]
            local = lambda points: (points - origin) @ rotation * 1000.
            errors = np.linalg.norm((observed-fitted)[valid], axis=1) * 1000.
            record = dict(side=side, row=row, sample_index=index, time_seconds=float(times[index]),
                          visible=int(valid.sum()), marker_rmse_mm=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                          max_error_mm=float(errors.max()) if len(errors) else None,
                          per_marker_error_mm={label: float(np.linalg.norm(observed[k]-fitted[k])*1000)
                                               if valid[k] else None for k, label in enumerate(labels)})
            diagnostics.append(record)
            panels.append(dict(record=record, mesh=local(world_vertices), observed=local(observed),
                               fitted=local(fitted), joints=local(world_joints), valid=valid, labels=labels))

    for view, target in zip(("a", "b"), targets[:2]):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
        projected = []
        for panel in panels:
            side = panel["record"]["side"]
            mesh, depth = _project(panel["mesh"], side, view)
            obs, _ = _project(panel["observed"], side, view)
            fit, _ = _project(panel["fitted"], side, view)
            joints, _ = _project(panel["joints"], side, view)
            visible_vertices = np.unique(faces[side])
            projected.append((panel, mesh, depth, obs, fit, joints))
        bounds = np.concatenate([np.r_[mesh[np.unique(faces[p["record"]["side"]])], obs[p["valid"]], fit]
                                 for p, mesh, _, obs, fit, _ in projected])
        low, high = bounds.min(0)-12., bounds.max(0)+12.
        for panel, mesh, depth, obs, fit, joints in projected:
            info = panel["record"];side = info["side"]
            col = chosen.index(info["row"]);ax = axes[0 if side == "L" else 1, col]
            hand_faces = faces[side];order = np.argsort(depth[hand_faces].mean(1))
            ax.add_collection(PolyCollection(mesh[hand_faces][order], facecolors=(.55,.69,.84,.65),
                                              edgecolors="none", rasterized=True))
            valid = panel["valid"]
            for a, b in zip(obs[valid], fit[valid]):
                ax.plot([a[0],b[0]], [a[1],b[1]], color="#bd8a7e", lw=.65, zorder=3)
            ax.scatter(obs[valid,0],obs[valid,1],c="#d34b43",s=21,zorder=6,label="Observed markers")
            ax.scatter(fit[:,0],fit[:,1],facecolors="none",edgecolors="#12856a",s=31,lw=1.,zorder=7,label="Fitted markers")
            for label, point in zip(np.array(panel["labels"])[valid], obs[valid]):
                ax.annotate(label[1:],point,xytext=(3,3),textcoords="offset points",fontsize=6,color="#9a302b",zorder=8)
            for chain in LEFT_CHAINS if side == "L" else RIGHT_CHAINS:
                points = joints[list(chain)]
                ax.plot(points[:,0],points[:,1],".-",color="#7b4ca4",ms=3,lw=1.,zorder=5)
            ax.plot([],[],".-",color="#7b4ca4",label="Native wrist/finger joints")
            rms = info["marker_rmse_mm"]
            statistic = f"RMS {rms:.1f} / max {info['max_error_mm']:.1f} mm" if rms is not None else "No visible observations"
            ax.set_title(f"{'Left' if side=='L' else 'Right'} | t={info['time_seconds']:.3f}s | {info['visible']}/14\n{statistic}",fontsize=10)
            ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1]);ax.set_aspect("equal");ax.grid(alpha=.2)
            ax.set_xlabel("Native wrist-local horizontal / mm");ax.set_ylabel("Native wrist-local finger direction / mm")
        axes[0,0].legend(loc="best",fontsize=7)
        fig.suptitle(f"SMPL-X native full LBS + FK | Prior-assisted candidate | Fixed wrist-frame view {view.upper()}\n"
                     "One shared transform for mesh/observations/joints; no extra alignment. Finger joints are not ground truth.",fontsize=12)
        fig.savefig(target,dpi=145);plt.close(fig)
    targets[2].write_text(json.dumps(dict(calibration=str(Path(args.calib).resolve()),source_run=str(args.source_run.resolve()),
        selection="near 1.5 seconds, minimum and maximum observed finger-to-wrist distance among fitted frames",
        native_full_model_evaluated_before_display_mask=True,extra_alignment=False,acceptance="PENDING_VISUAL_REVIEW",
        source_frame_numbers=[int(source_frames[int(frames[row])]) for row in chosen] if source_frames is not None else None,
        panels=diagnostics),indent=2))
    print(json.dumps({"outputs":[str(path) for path in targets],"selected_sample_indices":[int(frames[row]) for row in chosen]}))


if __name__ == "__main__":
    main()
