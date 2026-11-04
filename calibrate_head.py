"""Calibrate four SMPL-X head attachments from a matched neutral reference.

The reference NPZ must contain names, pos (F,72,3), valid (F,72), R
(F,55,3,3), trans (F,3), and source_subject. Coordinates are already in the
native solver's metres, +Y-up world; this command performs no axis conversion,
capture, or frame matching. Optional reference_times/times are seconds.

An operator confirms an upright head facing the body's forward direction.
The imposed calibration orientation is never used as replay acceptance evidence.
Old/new bindings are replayed with ordinary, unlocked solvers, from both cold
start and the same original matched pose. This reuses calibration observations;
it is not independent ground truth or a held-out motion test.

Example:
  python calibrate_head.py --model-path /models/smplx.npz \
    --parent-calibration /calibrations/actor_full.npz \
    --reference /references/actor_matched_reference.npz \
    --output-dir /derived/actor_head_new_run --subject actor \
    --operator-confirmed-neutral

Exit 0 means the declared replay limits passed; visual/motion acceptance remains
pending. Exit 2 means a retained candidate failed a numeric replay limit. Exit 1
means an input/runtime error. A candidate produced before an error is retained.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from vsmpl.body_model import BodyModel, MarkerRig
from vsmpl.hand_pca import HandPCA
from vsmpl.head_calibration import (
    HEAD_LABELS, SHOULDER_LABELS, _desired_frames, _orientation_report,
    calibrate_neutral_head,
)
from vsmpl.pca_solver import PCASolver
from vsmpl.solver import RealtimeSolver, SolverConfig


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_receipt(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    return dict(path=str(path), sha256=_sha256(path), size_bytes=path.stat().st_size)


def _scalar(data, key, default=None):
    if key not in data:
        if default is not None:
            return default
        raise ValueError(f"Missing required field: {key}")
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"{key} must be a scalar")
    return value.item()


def _subject(data):
    value = _scalar(data, "source_subject")
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("source_subject must be a nonempty, unpadded string")
    return value


def _load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _validate_reference(parent, reference, subject):
    if not subject or subject.strip() != subject:
        raise ValueError("--subject must be a nonempty, unpadded subject name")
    if _subject(parent) != subject or _subject(reference) != subject:
        raise ValueError("Parent calibration, reference and --subject must identify the same subject")
    required = {"names", "pos", "valid", "R", "trans", "source_subject"}
    if required - reference.keys():
        raise ValueError("Reference is missing fields: " + ", ".join(sorted(required-reference.keys())))
    names = np.asarray(reference["names"])
    parent_names = np.asarray(parent.get("marker_names", []))
    if (names.shape != (72,) or names.dtype.kind != "U" or len(set(names)) != 72
            or parent_names.shape != (72,) or parent_names.dtype.kind != "U"
            or set(names) != set(parent_names)):
        raise ValueError("Reference must have 72 unique Unicode labels matching the parent calibration")
    if set(HEAD_LABELS + SHOULDER_LABELS) - set(names):
        raise ValueError("Reference is missing required head/shoulder labels")
    pos = np.asarray(reference["pos"], dtype=np.float64)
    valid_source = np.asarray(reference["valid"])
    valid = np.asarray(valid_source, dtype=bool)
    R = np.asarray(reference["R"], dtype=np.float64)
    trans = np.asarray(reference["trans"], dtype=np.float64)
    F = len(pos) if pos.ndim else 0
    if (F < 3 or pos.shape != (F,72,3) or valid.shape != (F,72)
            or R.shape != (F,55,3,3) or trans.shape != (F,3)):
        raise ValueError("Reference markers, R and trans must have matching native frame dimensions")
    if valid_source.dtype.kind not in "buif" or not np.isin(valid_source, [0,1]).all():
        raise ValueError("Reference valid must contain only boolean/0/1 values")
    if not np.isfinite(pos[valid]).all() or not np.isfinite(R).all() or not np.isfinite(trans).all():
        raise ValueError("Reference contains non-finite visible markers or poses")
    if np.max(np.abs(R.swapaxes(-1,-2) @ R - np.eye(3))) > 5e-4 or np.linalg.det(R).min() <= 0:
        raise ValueError("Reference R must contain proper native rotation matrices")
    times = None
    for key in ("reference_times", "times"):
        if key in reference:
            value = np.asarray(reference[key], dtype=np.float64)
            if value.shape != (F,) or not np.isfinite(value).all() or np.any(np.diff(value) <= 0):
                raise ValueError(f"{key} must be finite, strictly increasing seconds, one per frame")
            if times is not None and not np.array_equal(times, value):
                raise ValueError("Conflicting reference_times and times")
            times = value
    return dict(names=names.tolist(), pos=pos, valid=valid), R, trans, times


def _prior(parent, parent_path, override):
    if _scalar(parent, "marker_scope", "all_markers") != "body_hands":
        if override:
            raise ValueError("--hand-prior is only valid for a body_hands parent")
        return None, None
    saved_source = _scalar(parent, "hand_prior_source")
    expected = str(_scalar(parent, "hand_prior_sha256")).lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("Parent hand_prior_sha256 is not a SHA256 digest")
    path = Path(override or saved_source).expanduser()
    if not path.is_absolute():
        path = parent_path.parent / path
    receipt = _file_receipt(path)
    if receipt["sha256"] != expected:
        raise ValueError("Hand prior SHA256 does not match the parent calibration")
    count = _scalar(parent, "hand_pca_components")
    if not isinstance(count, (int, np.integer)):
        raise ValueError("hand_pca_components must be an integer")
    prior = HandPCA.load(receipt["path"], n_components=int(count))
    prior._latent(parent["hand_pca_initial_latent"])
    weight = float(_scalar(parent, "hand_prior_weight"))
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("Invalid parent hand_prior_weight")
    fingers = np.asarray(parent.get("finger_marker_names", []))
    if len(fingers) != 20 or len(set(fingers)) != 20 or not set(fingers).issubset(parent["marker_names"]):
        raise ValueError("A body_hands parent must identify its 20 finger labels")
    receipt.update(parent_recorded_source=str(saved_source), source_override=bool(override),
                   components=prior.n_components, prior_weight=weight)
    return prior, receipt


def _native_model_receipt(model, path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    for name in ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "parents", "faces"):
        array = np.ascontiguousarray(getattr(model, name))
        digest.update(name.encode() + str(array.dtype).encode() + str(array.shape).encode())
        digest.update(memoryview(array).cast("B"))
    receipt = dict(path=str(path), loaded_native_arrays_sha256=digest.hexdigest())
    if path.is_file():
        receipt.update(source_kind="file", **{k:v for k,v in _file_receipt(path).items() if k != "path"})
    else:
        receipt.update(source_kind="directory", source_file_selection="BodyModel.load; loaded arrays hashed above")
    return receipt


def _validate_candidate(parent, candidate):
    """Check preservation independently of the attachment optimiser's assertions."""
    if _subject(candidate) != _subject(parent):
        raise ValueError("Candidate changed the parent subject association")
    names = np.asarray(parent["marker_names"])
    if not np.array_equal(candidate["marker_names"], names):
        raise ValueError("Candidate changed marker label ordering")
    nonhead = np.flatnonzero(~np.isin(names, HEAD_LABELS))
    for key in ("vertex_ids", "offsets", "betas"):
        original, updated = np.asarray(parent[key]), np.asarray(candidate[key])
        if original.shape != updated.shape or original.dtype != updated.dtype:
            raise ValueError(f"Candidate changed native {key} shape/type")
        before, after = (original,updated) if key == "betas" else (original[nonhead],updated[nonhead])
        if before.tobytes() != after.tobytes():
            raise ValueError(f"Candidate changed protected {key}: only four head attachments may change")
    if _scalar(candidate,"head_online_pose_locked") is not False:
        raise ValueError("Candidate must not lock online head rotations")


def _rms(prediction, observed, valid, indices, names):
    error = np.linalg.norm(prediction[:,indices]-observed[:,indices], axis=-1)
    mask = valid[:,indices]
    if not mask.any():
        raise ValueError("No visible markers for a replay metric")
    each = []
    for j in range(len(indices)):
        each.append(float(np.sqrt(np.mean(error[mask[:,j],j]**2))*1000) if mask[:,j].any() else None)
    return dict(rms_mm=float(np.sqrt(np.mean(error[mask]**2))*1000),
                per_marker_rms_mm={names[i]: value for i,value in zip(indices,each)})


def _replay(model, calibration, prior, rec, R_reference, trans_reference, indices, cfg, start, output):
    rig = MarkerRig.from_calibration(model, calibration)
    if prior is None:
        solver = RealtimeSolver(rig, cfg)
    else:
        solver = PCASolver(rig, prior, cfg, initial_latent=calibration["hand_pca_initial_latent"],
                           hand_prior_weight=float(calibration["hand_prior_weight"]))
    if any(j not in solver.active for j in (12,15)) or any(j in solver.cfg.fixed_joints for j in (12,15)):
        raise ValueError("Replay must leave native neck12 and head15 active and unlocked")
    order = [rec["names"].index(name) for name in rig.names]
    obs, valid = rec["pos"][indices][:,order], rec["valid"][indices][:,order]
    desired = _desired_frames(rec["pos"][indices].copy(), rec["names"])
    if start == "matched_pose":
        # Same original fit for old/new; never initialise from imposed neutral poses.
        solver.R = R_reference[indices[0]].copy()
        if prior is not None:
            solver.R = prior.apply(solver.R, prior.encode(solver.R))
            solver.R[22:25] = np.eye(3)
        solver.trans = trans_reference[indices[0]].copy()
        solver.initialised, solver.hist = True, []
    rotations, translations, predictions, worlds, statuses, inliers = [], [], [], [], [], []
    for offset, frame in enumerate(indices):
        result = solver.step(obs[offset], valid[offset], frame_no=int(frame))
        rotations.append(result.R)
        translations.append(result.trans)
        predictions.append(rig.forward(result.R, result.trans))
        worlds.append(model.fk(result.R, rig.J_rest)[0][15])
        statuses.append(result.status)
        inliers.append(result.n_inliers)
    predictions, worlds = np.asarray(predictions), np.asarray(worlds)
    orientation = _orientation_report(worlds.copy(), desired)
    head = [rig.names.index(name) for name in HEAD_LABELS]
    other = [i for i,name in enumerate(rig.names) if name not in HEAD_LABELS]
    np.savez_compressed(output, R=np.asarray(rotations), trans=np.asarray(translations),
        predictions=predictions, names=np.asarray(rig.names), sample_indices=indices,
        head_world_R=worlds, shoulder_desired_world_R=desired, status=np.asarray(statuses),
        n_inliers=np.asarray(inliers), active_joints=solver.active,
        head_online_pose_locked=False, initialisation=start,
        **{key:np.asarray(value) for key,value in orientation.items()})
    up, yaw = np.asarray(orientation["up_deviation_degrees"]), np.abs(orientation["yaw_error_degrees"])
    return dict(solver=type(solver).__name__, initialisation=start, samples=len(indices),
        sample_indices=indices.tolist(), active_joints=solver.active.tolist(),
        active_neck12=True, active_head15=True, fixed_joints=list(solver.cfg.fixed_joints),
        online_head_locked=False, solver_config=asdict(solver.cfg),
        head=_rms(predictions,obs,valid,head,rig.names),
        nonhead=_rms(predictions,obs,valid,other,rig.names), orientation=orientation,
        up_deviation_mean_degrees=float(up.mean()), up_deviation_p95_degrees=float(np.percentile(up,95)),
        absolute_heading_error_mean_degrees=float(yaw.mean()),
        absolute_heading_error_p95_degrees=float(np.percentile(yaw,95)),
        held_frames=statuses.count("hold"), n_inliers=inliers, output=_file_receipt(output))


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k):_json_value(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [_json_value(v) for v in value]
    return value


def _write_report(path, report):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(_json_value(report), indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def _quality(replays, fit_report, args):
    failures = []
    for start in ("cold", "matched_pose"):
        old, new = replays["parent_"+start], replays["candidate_"+start]
        checks = ((new["up_deviation_p95_degrees"], args.max_head_up_degrees, "head up p95"),
                  (new["absolute_heading_error_p95_degrees"], args.max_heading_error_degrees, "heading error p95"),
                  (new["head"]["rms_mm"], args.max_head_rms_mm, "all-visible head RMS"),
                  (new["nonhead"]["rms_mm"]-old["nonhead"]["rms_mm"], args.max_nonhead_rms_increase_mm, "non-head RMS increase"))
        for value, limit, label in checks:
            if not np.isfinite(value) or value > limit:
                failures.append(f"{start}: {label} {value:.6g} exceeds {limit:.6g}")
        if new["held_frames"]:
            failures.append(f"{start}: {new['held_frames']} reference frames were held")
    return dict(status="FAIL" if failures else "PASS", failures=failures,
        warnings=list(fit_report.get("quality_warnings", [])),
        limits=dict(head_up_p95_degrees=args.max_head_up_degrees,
            absolute_heading_error_p95_degrees=args.max_heading_error_degrees,
            all_visible_head_rms_mm=args.max_head_rms_mm,
            nonhead_rms_increase_mm=args.max_nonhead_rms_increase_mm),
        limit_origin="caller quality limits; no independent ground truth")


def _parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--parent-calibration", required=True)
    ap.add_argument("--reference", required=True, help="already matched native reference NPZ; no capture or axis conversion")
    ap.add_argument("--output-dir", required=True, help="must not already exist")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--operator-confirmed-neutral", action="store_true", required=True,
                    help="operator confirms upright head facing body forward, standing still")
    ap.add_argument("--hand-prior", help="relocated copy of the parent's prior; its saved SHA256 must match")
    ap.add_argument("--replay-iters", type=int, default=4)
    ap.add_argument("--replay-init-iters", type=int, default=40)
    ap.add_argument("--max-head-up-degrees", type=float, default=5.)
    ap.add_argument("--max-heading-error-degrees", type=float, default=10.)
    ap.add_argument("--max-head-rms-mm", type=float, default=10.)
    ap.add_argument("--max-nonhead-rms-increase-mm", type=float, default=2.)
    return ap


def run(args):
    """Run one new, provenance-preserving candidate; return 0/2 for numeric PASS/FAIL."""
    if not args.operator_confirmed_neutral:
        raise ValueError("Operator-confirmed upright, forward-facing neutral reference is required")
    for value in (args.max_head_up_degrees, args.max_heading_error_degrees,
                  args.max_head_rms_mm, args.max_nonhead_rms_increase_mm):
        if not np.isfinite(value) or value < 0:
            raise ValueError("Quality limits must be finite and nonnegative")
    if args.replay_iters < 1 or args.replay_init_iters < 1:
        raise ValueError("Replay iteration counts must be positive")
    parent_path = Path(args.parent_calibration).expanduser().resolve(strict=True)
    reference_path = Path(args.reference).expanduser().resolve(strict=True)
    output = Path(args.output_dir).expanduser().absolute()
    if output.exists():
        raise ValueError("--output-dir must be a new directory; existing results are never overwritten")
    parent, reference = _load_npz(parent_path), _load_npz(reference_path)
    rec, rotations, translations, times = _validate_reference(parent, reference, args.subject)
    prior, prior_receipt = _prior(parent, parent_path, args.hand_prior)
    model_type, gender = str(_scalar(parent,"model_type","smplx")), str(_scalar(parent,"gender","neutral"))
    if model_type != "smplx":
        raise ValueError("Head calibration CLI requires native SMPL-X55")
    model_path = Path(args.model_path).expanduser().resolve(strict=True)
    model = BodyModel.load(str(model_path), model_type, gender, int(_scalar(parent,"num_betas",len(parent["betas"]))))
    provenance = dict(parent_calibration=_file_receipt(parent_path), reference=_file_receipt(reference_path),
        model=_native_model_receipt(model,model_path), hand_prior=prior_receipt,
        source_subject=args.subject, reference_coordinate_contract="native solver metres, configured +Y up",
        raw_source_modified=False, reference_frame_matching_performed=False,
        matching_claim="R/trans supplied by caller as fits of these exact marker frames; no independent verification")
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "head_calibration_report.json"
    report = dict(status="RUNNING", provenance=provenance, operator_confirmed_neutral=True,
        operator_condition="upright head facing body forward, standing still", independent_ground_truth=False,
        validation_scope="same neutral reference reused for calibration and unlocked replay; no held-out motion",
        acceptance="PENDING_VISUAL_REVIEW_AND_INDEPENDENT_MOTION_HOLDOUT", online_head_locked=False)
    _write_report(report_path, report)
    try:
        candidate, fit_report = calibrate_neutral_head(model,parent,rec,rotations,translations,
            operator_confirmed_neutral=True, reference_times=times)
        # Relocation changes provenance only; the prior bytes and hand parameters are identical.
        if prior_receipt:
            candidate["hand_prior_source"] = np.asarray(prior_receipt["path"])
        candidate.update(head_parent_calibration_source=np.asarray(str(parent_path)),
            head_parent_calibration_sha256=np.asarray(provenance["parent_calibration"]["sha256"]),
            head_reference_source=np.asarray(str(reference_path)),
            head_reference_sha256=np.asarray(provenance["reference"]["sha256"]),
            head_reference_source_subject=np.asarray(args.subject),
            head_model_native_arrays_sha256=np.asarray(provenance["model"]["loaded_native_arrays_sha256"]))
        checkpoint = output / "subject_head_calibration.npz"
        np.savez_compressed(checkpoint, **candidate)
        report.update(calibration=fit_report, candidate_checkpoint=_file_receipt(checkpoint), status="REPLAY_PENDING")
        _write_report(report_path,report)
        _validate_candidate(parent,candidate)
        indices = np.asarray(candidate["head_reference_sample_indices"], dtype=np.int64)
        if indices.ndim != 1 or len(indices) < 3 or np.any(np.diff(indices) <= 0) or indices[0] < 0 or indices[-1] >= len(rotations):
            raise ValueError("Calibration returned invalid reference sample indices")
        cfg = SolverConfig(iters=args.replay_iters, init_iters=args.replay_init_iters)
        if prior is None:
            cfg.active_joints = list(range(22))
            cfg.fixed_joints = list(range(22,55))
        report["unlocked_replay"] = {}
        for start in ("cold","matched_pose"):
            for label, calibration in (("parent",parent),("candidate",candidate)):
                key = label+"_"+start
                report["unlocked_replay"][key] = _replay(model,calibration,prior,rec,rotations,translations,
                    indices,cfg,start,output/("replay_"+key+".npz"))
                _write_report(report_path,report)
        quality = _quality(report["unlocked_replay"],fit_report,args)
        report["numeric_quality"] = quality
        report["status"] = "NUMERIC_PASS_REVIEW_PENDING" if quality["status"] == "PASS" else "CANDIDATE_FAIL"
        unchanged = all(_sha256(provenance[key]["path"]) == provenance[key]["sha256"]
                        for key in ("parent_calibration","reference"))
        report["provenance"]["parent_and_reference_hashes_unchanged"] = unchanged
        if not unchanged:
            quality["status"] = "FAIL"
            quality["failures"].append("Parent/reference source bytes changed during the run")
            report["status"] = "CANDIDATE_FAIL"
        _write_report(report_path, report)
        print(f"{report['status']}: {report_path}")
        return 0 if quality["status"] == "PASS" else 2
    except Exception as exc:
        report.update(status="CANDIDATE_ERROR", error=dict(type=type(exc).__name__, message=str(exc)))
        _write_report(report_path,report)
        raise


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"Head calibration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
