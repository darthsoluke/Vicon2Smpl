"""Operator-referenced head attachment calibration on native SMPL-X FK/LBS.

Only the four head attachments are changed. The neutral orientation constraint
exists during this calibration, not in the online pose solver. World up is the
configured Vicon/SMPL world axis, not an independent gravity measurement.
"""
from itertools import combinations
import hashlib

import numpy as np

from .rotations import exp_so3, log_so3


HEAD_LABELS = ("LFHD", "LBHD", "RFHD", "RBHD")
SHOULDER_LABELS = ("LFSH", "LBSH", "RFSH", "RBSH")
NECK, HEAD, NECK_PARENT = 12, 15, 9
WORLD_UP = np.array([0., 1., 0.])


def _world_rotations(rotations, parents):
    out = np.empty_like(rotations)
    out[:, 0] = rotations[:, 0]
    for joint in range(1, len(parents)):
        out[:, joint] = out[:, parents[joint]] @ rotations[:, joint]
    return out


def neutral_head_rotations(R_reference, desired_world_R, neck_delta, parents):
    """Copy native local rotations, changing only 12/15 to impose the reference head R.

    This helper deliberately has no translation or independent head transform.
    The resulting head position must be realised by the native neck/head chain.
    """
    source = np.asarray(R_reference, dtype=np.float64)
    single = source.ndim == 3
    if single:
        source = source[None]
    target = np.asarray(desired_world_R, dtype=np.float64)
    if target.shape == (3, 3):
        target = np.broadcast_to(target, (len(source), 3, 3))
    delta = np.asarray(neck_delta, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int64)
    if (source.ndim != 4 or source.shape[1:] != (55, 3, 3)
            or target.shape != (len(source), 3, 3) or delta.shape != (3,)
            or parents.shape != (55,) or parents[12] != 9 or parents[15] != 12):
        raise ValueError("Expected the native SMPL-X55 neck/head hierarchy and matching rotations")
    if parents[0] != -1 or any(not 0 <= parents[j] < j for j in range(1,55)):
        raise ValueError("Native rotations require a topologically ordered kinematic tree")
    if not np.isfinite(source).all() or not np.isfinite(target).all() or not np.isfinite(delta).all():
        raise ValueError("Head reference rotations must be finite")
    world = _world_rotations(source, parents)
    neck_world = target @ exp_so3(delta)
    result = source.copy()
    result[:, 12] = np.swapaxes(world[:, 9], -1, -2) @ neck_world
    result[:, 15] = np.swapaxes(neck_world, -1, -2) @ target
    return result[0] if single else result


def head_vertex_regions(model, betas):
    """Upper-cranium coordinate quadrants with nearly rigid head15 ownership.

    Normal signs are intentionally not used to interpret front/back. These
    anatomical bounds are caller priors and are reported with the result.
    """
    if getattr(model, "num_joints", None) != 55:
        raise ValueError("Head calibration requires native SMPL-X55")
    vertices, joints = model.shaped(np.asarray(betas, dtype=np.float64))
    weights = np.asarray(model.weights)
    if weights.shape != (len(vertices), 55) or joints.shape != (55, 3):
        raise ValueError("Invalid native head geometry")
    pure = weights[:, 15] >= .98
    if pure.sum() < 8:
        raise ValueError("Too few nearly rigid head15 surface vertices")
    low, high = vertices[pure].min(0), vertices[pure].max(0)
    # A headband's posterior markers can be much lower than its front pair.
    # Keep the nearly rigid head surface, including the occipital/temple band.
    floor = max(float(joints[15, 1] + .02), float(low[1] + .2 * (high[1] - low[1])))
    upper = pure & (vertices[:, 1] >= floor)
    if upper.sum() < 4:
        raise ValueError("No usable upper-cranium surface")
    centre = (vertices[upper].min(0) + vertices[upper].max(0)) / 2
    regions = {}
    for label in HEAD_LABELS:
        side = vertices[:, 0] > centre[0] + .003 if label[0] == "L" else vertices[:, 0] < centre[0] - .003
        # Coordinate overlap admits temple/headband points without using normals
        # or a strict frontal plane to force an incorrect neck compensation.
        front = vertices[:, 2] >= centre[2] - .025 if label[1] == "F" else vertices[:, 2] <= centre[2] + .025
        ids = np.flatnonzero(upper & side & front)
        if not len(ids):
            raise ValueError(f"No upper-cranium candidates for {label}")
        regions[label] = ids
    return regions, dict(head_weight_min=.98, upper_height_floor_m=floor,
                        upper_fraction_of_head_extent=.2, left_right_margin_m=.003,
                        front_back_overlap_half_width_m=.025,
                        head_surface_centre_m=centre.tolist(),
                        candidate_counts={name: len(ids) for name, ids in regions.items()})


def _desired_frames(points, names):
    at = lambda label: points[:, names.index(label)]
    lateral = (at("LFSH") + at("LBSH") - at("RFSH") - at("RBSH")) / 2
    lateral[:, 1] = 0
    width = np.linalg.norm(lateral, axis=1)
    if np.any(width < .08):
        raise ValueError("Shoulder labels do not provide a stable horizontal left axis")
    x = lateral / width[:, None]
    y = np.broadcast_to(WORLD_UP, x.shape)
    z = np.cross(x, y)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    x = np.cross(y, z)
    desired = np.stack((x, y, z), axis=-1)
    # Reject obvious source-label inversions instead of compensating with bindings.
    for a, b, axis in (("LFHD", "RFHD", x), ("LBHD", "RBHD", x),
                       ("LFHD", "LBHD", z), ("RFHD", "RBHD", z)):
        signed = ((at(a) - at(b)) * axis).sum(-1)
        if np.median(signed) <= .003:
            raise ValueError(f"Head label ordering conflicts with confirmed forward-neutral reference: {a}/{b}")
    return desired


def _orientation_report(world_head, desired):
    up = world_head[:, :, 1]
    up /= np.linalg.norm(up, axis=-1, keepdims=True)
    up_angle = np.degrees(np.arccos(np.clip(up @ WORLD_UP, -1, 1)))
    heading = np.arctan2(world_head[:, 0, 2], world_head[:, 2, 2])
    target = np.arctan2(desired[:, 0, 2], desired[:, 2, 2])
    difference = np.arctan2(np.sin(heading-target), np.cos(heading-target))
    return dict(up_deviation_degrees=up_angle.tolist(),
                yaw_degrees=np.degrees(heading).tolist(),
                desired_yaw_degrees=np.degrees(target).tolist(),
                yaw_error_degrees=np.degrees(difference).tolist())


def calibrate_neutral_head(model, parent_calib, rec, R_reference, trans_reference, *,
                           operator_confirmed_neutral, reference_times=None):
    """Re-estimate four head attachments from an explicitly confirmed neutral reference.

    The operator condition means upright head, facing the body's forward
    direction, while standing still. It must not be inferred from four cap dots.
    All returned reference poses obey that calibration constraint; acceptance
    still requires an unconstrained native online/replay check.
    """
    if operator_confirmed_neutral is not True:
        raise ValueError("An operator-confirmed upright, forward-facing neutral head reference is required")
    if getattr(model, "num_joints", None) != 55:
        raise ValueError("Head calibration requires native SMPL-X55")
    parents = np.asarray(model.parents, dtype=np.int64)
    if parents.shape != (55,) or parents[12] != 9 or parents[15] != 12:
        raise ValueError("Unexpected native SMPL-X neck/head hierarchy")
    names = list(rec["names"])
    if len(names) != 72 or len(set(names)) != 72:
        raise ValueError("Reference must retain 72 unique source marker labels")
    required = set(HEAD_LABELS + SHOULDER_LABELS)
    if not required.issubset(names):
        raise ValueError("Missing required head/shoulder marker labels: " + ", ".join(sorted(required-set(names))))
    parent_names = [str(n) for n in parent_calib["marker_names"]]
    if len(parent_names) != 72 or set(names) != set(parent_names):
        raise ValueError("Unknown or missing source labels relative to the parent calibration")
    position, valid = np.asarray(rec["pos"], dtype=np.float64), np.asarray(rec["valid"], dtype=bool)
    rotation, translation = np.asarray(R_reference, dtype=np.float64), np.asarray(trans_reference, dtype=np.float64)
    if position.ndim != 3:
        raise ValueError("Reference markers, native poses and translations must have matching frame dimensions")
    F = len(position)
    if (position.shape != (F, 72, 3) or valid.shape != (F, 72)
            or rotation.shape != (F, 55, 3, 3) or translation.shape != (F, 3) or F < 3):
        raise ValueError("Reference markers, native poses and translations must have matching frame dimensions")
    if (not np.isfinite(position[valid]).all() or not np.isfinite(rotation).all()
            or not np.isfinite(translation).all()):
        raise ValueError("Reference contains invalid positions or poses")
    orthogonal_error = np.max(np.abs(np.swapaxes(rotation, -1, -2) @ rotation - np.eye(3)))
    if orthogonal_error > 5e-4 or np.linalg.det(rotation).min() <= 0:
        raise ValueError("Reference pose is not a proper native rotation")
    times = None if reference_times is None else np.asarray(reference_times, dtype=np.float64)
    if times is not None and (times.shape != (F,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("Reference times must be finite and strictly increasing")
    required_ids = [names.index(n) for n in required]
    clean = np.flatnonzero(valid[:, required_ids].all(1))
    if len(clean) < 3:
        raise ValueError("Too few frames with all head and shoulder markers visible")
    selected = clean[np.unique(np.round(np.linspace(0, len(clean)-1, min(32, len(clean)))).astype(int))]
    desired = _desired_frames(position[selected].copy(), names)

    betas = np.asarray(parent_calib["betas"], dtype=np.float64)
    original_ids = np.asarray(parent_calib["vertex_ids"])
    original_offsets = np.asarray(parent_calib["offsets"])
    if not np.issubdtype(original_ids.dtype, np.integer) or not np.issubdtype(original_offsets.dtype,np.floating):
        raise ValueError("Parent attachment IDs/offsets must retain integer/floating native types")
    old_ids = np.asarray(original_ids, dtype=np.int64)
    old_offsets = np.asarray(original_offsets, dtype=np.float64)
    if (betas.ndim != 1 or not np.isfinite(betas).all() or old_ids.shape != (72,)
            or old_offsets.shape != (72, 3) or not np.isfinite(old_offsets).all()
            or old_ids.min() < 0 or old_ids.max() >= len(model.v_template)
            or len(betas) > getattr(model,"num_betas",len(betas))):
        raise ValueError("Parent calibration has invalid native shape or attachments")
    # Work in source-label order, then restore the parent's exact order on output.
    order = np.array([parent_names.index(n) for n in names])
    ids_source, offsets_source = old_ids[order], old_offsets[order]
    head_indices = np.array([names.index(n) for n in HEAD_LABELS])
    nonhead = np.array([i for i, n in enumerate(names) if n not in HEAD_LABELS])
    regions, region_report = head_vertex_regions(model, betas)
    v_rest, j_rest = model.shaped(betas)
    normals = model.vertex_normals(v_rest)
    R0, tr0 = rotation[selected], translation[selected]
    obs, mask = position[selected], valid[selected]

    import torch
    from .torch_model import TorchBody, batch_rodrigues
    tb = TorchBody(model)
    dtype = torch.float64
    base_R = torch.tensor(R0, dtype=dtype)
    trans_t = torch.tensor(tr0, dtype=dtype)
    desired_t = torch.tensor(desired, dtype=dtype)
    world_reference = _world_rotations(R0, parents)
    parent_world = torch.tensor(world_reference[:, 9], dtype=dtype)
    observed = torch.tensor(np.nan_to_num(obs), dtype=dtype)
    visible = torch.tensor(mask, dtype=dtype)
    rest_t, joints_t = torch.tensor(v_rest, dtype=dtype), torch.tensor(j_rest, dtype=dtype)
    fixed_offsets = torch.tensor(offsets_source, dtype=dtype)
    head_t = torch.tensor(head_indices)

    def rotations(neck_delta):
        neck_world = desired_t @ batch_rodrigues(neck_delta)
        local12 = parent_world.transpose(-1, -2) @ neck_world
        local15 = neck_world.transpose(-1, -2) @ desired_t
        result = base_R.clone()
        result[:, 12] = local12
        result[:, 15] = local15
        return result

    def choose_vertices(R, heights, tangent=None):
        result = ids_source.copy()
        tangent = np.zeros((4, 3)) if tangent is None else tangent
        with torch.no_grad():
            for h, name in enumerate(HEAD_LABELS):
                candidates = regions[name]
                n = normals[candidates]
                off = heights[h]*n + tangent[h] - (tangent[h]*n).sum(-1, keepdims=True)*n
                candidate_t = torch.tensor(candidates)
                points = rest_t[candidate_t] + torch.tensor(off, dtype=dtype)
                pred = tb.skin(R, trans_t, points, tb.weights[candidate_t], tb.posedirs[:, candidate_t], joints_t)
                squared = ((pred-observed[:, head_indices[h], None])**2).sum(-1).mean(0)
                result[head_indices[h]] = candidates[int(squared.argmin())]
        return result

    relative_old_neck = np.swapaxes(desired, -1, -2) @ world_reference[:, 12]
    old_delta = np.median(log_so3(relative_old_neck), axis=0)
    seeds = [np.zeros(3), old_delta, np.array([.26, 0, 0]), np.array([-.26, 0, 0]),
             np.array([0, 0, .26]), np.array([0, 0, -.26])]
    candidates, trials = [], []
    for seed_number, seed in enumerate(seeds):
        delta = torch.tensor(seed, dtype=dtype, requires_grad=True)
        height_raw = torch.full((4,), np.log(.0095/(.03-.0095)), dtype=dtype, requires_grad=True)
        tangent_raw = torch.zeros((4, 3), dtype=dtype, requires_grad=True)
        ids = choose_vertices(rotations(delta), np.full(4, .0095))
        for em in range(2):
            id_t = torch.tensor(ids)
            normal = torch.tensor(normals[ids[head_indices]], dtype=dtype)

            def offsets():
                height = .03 * torch.sigmoid(height_raw)
                tangent = .005 * torch.tanh(tangent_raw)
                tangent = tangent - (tangent*normal).sum(-1, keepdim=True)*normal
                tangent = tangent * torch.clamp(.005 / tangent.abs().amax(-1, keepdim=True).clamp_min(1e-12), max=1.)
                off = fixed_offsets.index_copy(0, head_t, height[:, None]*normal+tangent)
                return off, height, tangent

            def loss():
                off, height, tangent = offsets()
                R = rotations(delta)
                pred = tb.skin(R, trans_t, rest_t[id_t]+off, tb.weights[id_t], tb.posedirs[:, id_t], joints_t)
                distance = (pred-observed).norm(dim=-1)
                robust = torch.where(distance < .015, .5*distance.square(), .015*(distance-.0075))
                data = (robust*visible).sum()/len(selected)*1e4
                angle = torch.linalg.vector_norm(delta)
                prior = .1*delta.square().sum() + 5*torch.relu(angle-np.radians(35)).square()
                prior = prior + 500*(height-.0095).square().sum() + 5e3*tangent.square().sum()
                return data+prior, pred, off, height, tangent

            params = [delta, height_raw, tangent_raw]
            optimizer = torch.optim.Adam([
                dict(params=[delta], lr=.02), dict(params=[height_raw], lr=.06),
                dict(params=[tangent_raw], lr=.04)])
            for _ in range(120):
                optimizer.zero_grad()
                value, *_ = loss()
                if not torch.isfinite(value):
                    raise ValueError("Neutral head optimisation produced a non-finite loss")
                value.backward()
                optimizer.step()
            optimizer = torch.optim.LBFGS(params, max_iter=80, history_size=12, line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad()
                value, *_ = loss()
                value.backward()
                return value

            optimizer.step(closure)
            value, pred, off, heights, tangent = loss()
            if em == 0:
                ids = choose_vertices(rotations(delta), heights.detach().numpy(), tangent.detach().numpy())
        solution = dict(loss=float(value.detach()), ids=ids.copy(), offsets=off.detach().numpy(),
                        R=rotations(delta).detach().numpy(), delta=delta.detach().numpy(),
                        pred=pred.detach().numpy(), heights=heights.detach().numpy(), tangent=tangent.detach().numpy())
        candidates.append(solution)
        trials.append(dict(seed=seed_number, initial_neck_delta=seed.tolist(), loss=solution["loss"],
                           neck_angle_degrees=float(np.degrees(np.linalg.norm(solution["delta"])))))
    best = min(candidates, key=lambda item: item["loss"])

    from .body_model import MarkerRig
    old_rig = MarkerRig(model, names, ids_source, offsets_source, betas)
    before = np.stack([old_rig.forward(R, t) for R, t in zip(R0, tr0)])
    after = best["pred"]

    def residuals(pred, indices):
        err = np.linalg.norm(pred[:, indices]-obs[:, indices], axis=-1)
        ok = mask[:, indices]
        rms = float(np.sqrt(np.mean(err[ok]**2)))
        each = np.sqrt((np.where(ok, err, 0)**2).sum(0)/np.maximum(ok.sum(0), 1))
        return dict(rms_mm=rms*1000, per_marker_rms_mm={names[i]: float(e*1000) for i, e in zip(indices, each)})

    old_head, new_head = residuals(before, head_indices), residuals(after, head_indices)
    old_other, new_other = residuals(before, nonhead), residuals(after, nonhead)
    result = {name: np.asarray(parent_calib[name]).copy() for name in parent_calib.keys()}
    result_ids, result_offsets = original_ids.copy(), original_offsets.copy()
    for h, name in enumerate(HEAD_LABELS):
        target, origin = parent_names.index(name), names.index(name)
        result_ids[target], result_offsets[target] = best["ids"][origin], best["offsets"][origin]
    untouched = [i for i, name in enumerate(parent_names) if name not in HEAD_LABELS]
    assert np.array_equal(result_ids[untouched], old_ids[untouched])
    assert np.array_equal(result_offsets[untouched], old_offsets[untouched])
    result.update(vertex_ids=result_ids, offsets=result_offsets,
                  head_neutral_reference_confirmed=True, head_online_pose_locked=False,
                  head_reference_sample_indices=selected, head_neutral_reference_R=best["R"],
                  head_neutral_reference_trans=tr0, head_neutral_desired_world_R=desired,
                  head_neutral_neck_delta=best["delta"])
    if times is not None:
        result["head_reference_times"] = times[selected].copy()
    if "per_marker_rms" in result:
        updated = np.asarray(result["per_marker_rms"], dtype=np.float64).copy()
        for name, value in new_head["per_marker_rms_mm"].items():
            updated[parent_names.index(name)] = value/1000
        result["per_marker_rms"] = updated
    assert np.array_equal(np.asarray(result["betas"]), np.asarray(parent_calib["betas"]))
    final_world = _world_rotations(best["R"], parents)
    pairs = list(combinations(range(4), 2))
    observed_edges = np.stack([np.linalg.norm(obs[:, head_indices[a]]-obs[:, head_indices[b]], axis=-1) for a,b in pairs], 1)
    predicted_edges = np.stack([np.linalg.norm(after[:, head_indices[a]]-after[:, head_indices[b]], axis=-1) for a,b in pairs], 1)
    warnings = []
    angle = float(np.degrees(np.linalg.norm(best["delta"])))
    if angle > 35:
        warnings.append("Native neck compensation exceeds the 35-degree soft prior")
    if new_head["rms_mm"] > 10:
        warnings.append("Neutral head marker RMS exceeds 10mm")
    if new_other["rms_mm"] > old_other["rms_mm"]+2:
        warnings.append("Non-head RMS worsened by more than 2mm")
    if best["heights"].max() > .0299:
        warnings.append("A head offset saturated the 30mm height prior")
    fingerprint = hashlib.sha256(old_ids.tobytes()+old_offsets.tobytes()+betas.tobytes()).hexdigest()
    report = dict(operator_confirmed_neutral=True, neutral_condition="upright head facing body forward",
        world_up_reference="Vicon configured up, expressed as SMPL +Y; not independently measured gravity",
        reference_sample_indices=selected.tolist(), reference_times=times[selected].tolist() if times is not None else None,
        parent_binding_shape_fingerprint=fingerprint, native_fk_lbs=True, raw_source_modified=False,
        unchanged_nonhead_bindings=68, body_shape_preserved=True, online_head_locked=False,
        region_prior=region_report, attachment_prior=dict(normal_bounds_mm=[0,30], initial_normal_mm=9.5,
            tangential_per_coordinate_bound_mm=5, neck_soft_angle_degrees=35),
        baseline_head=old_head, corrected_head=new_head, baseline_nonhead=old_other, corrected_nonhead=new_other,
        baseline_orientation=_orientation_report(world_reference[:, 15].copy(), desired),
        imposed_neutral_orientation=_orientation_report(final_world[:, 15].copy(), desired),
        neutral_head_constraint_max_matrix_error=float(np.max(np.abs(final_world[:, 15]-desired))),
        neck_compensation_angle_degrees=angle,
        native_local_neck_angle_degrees=np.degrees(np.linalg.norm(log_so3(best["R"][:,12]),axis=-1)).tolist(),
        native_local_head_angle_degrees=np.degrees(np.linalg.norm(log_so3(best["R"][:,15]),axis=-1)).tolist(),
        head_marker_heights_mm=(best["heights"]*1000).tolist(),
        head_marker_tangential_offsets_mm=(best["tangent"]*1000).tolist(),
        head_skinning_weights={name: model.weights[best["ids"][names.index(name)]].tolist() for name in HEAD_LABELS},
        head_pair_edges=[HEAD_LABELS[a]+"/"+HEAD_LABELS[b] for a,b in pairs],
        observed_head_pair_mean_mm=(observed_edges.mean(0)*1000).tolist(),
        observed_head_pair_std_mm=(observed_edges.std(0)*1000).tolist(),
        predicted_head_pair_mean_mm=(predicted_edges.mean(0)*1000).tolist(),
        predicted_head_pair_std_mm=(predicted_edges.std(0)*1000).tolist(),
        seed_trials=trials, quality_warnings=warnings,
        inherited_selected_pose_cache_refitted=False,
        acceptance="PENDING_UNCONSTRAINED_NATIVE_SOLVER_REPLAY_AND_VISUAL_REVIEW")
    return result, report
