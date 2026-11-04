"""Calibrate articulated SMPL-X hands on the same native full-body FK/LBS rig.

The body shape/attachments from this actor remain fixed; body poses start from
the existing fit and can refine jointly with hands. Hand articulation uses
the supplied learned PCA prior; sparse markers do not independently measure all
finger rotation degrees of freedom. Attachment bounds are optimisation priors.
"""
import time

import numpy as np
import torch

from .body_model import MarkerRig
from .hand_regions import finger_vertex_regions
from .sources import NameMapper
from .solver import RealtimeSolver
from .torch_model import TorchBody, batch_rodrigues


def body_tracks(model, rec, body_calib, frame_numbers):
    rig = MarkerRig.from_calibration(model, body_calib)
    mapper, solver = NameMapper(rig.names), RealtimeSolver(rig)
    rotations, translations, residual = [], [], []
    for i, (pos, valid) in enumerate(zip(rec["pos"], rec["valid"])):
        p, v = mapper(dict(names=rec["names"], pos=pos, valid=valid))
        result = solver.step(p, v, frame_no=int(frame_numbers[i]))
        rotations.append(result.R)
        translations.append(result.trans)
        residual.append(result.rmse)
    return np.asarray(rotations), np.asarray(translations), np.asarray(residual)


def calibrate_hands(model, rec, body_calib, prior, body_regions, *, frame_numbers=None,
                    times=None, tracks=None, n_frames=48, rounds=2, steps=250):
    if model.num_joints != 55:
        raise ValueError("Full hand fitting requires native SMPL-X55")
    started = time.perf_counter()
    names = list(rec["names"])
    finger_names = [name for name in names if name not in body_regions]
    if len(finger_names) != 20:
        raise ValueError("Expected the project's 20 explicitly labelled finger markers")
    betas = np.asarray(body_calib["betas"], dtype=np.float64)
    regions = dict(body_regions)
    regions.update(finger_vertex_regions(model, betas, finger_names))
    body_names = [str(name) for name in body_calib["marker_names"]]
    if set(names) != set(body_names) | set(finger_names):
        raise ValueError("Some source body markers have no calibrated attachment")
    body_lookup = {name: i for i, name in enumerate(body_names)}
    finger_indices = np.array([names.index(name) for name in finger_names])
    body_indices = np.array([names.index(name) for name in body_names])
    T = len(rec["pos"])
    frame_numbers = np.arange(T) if frame_numbers is None else np.asarray(frame_numbers)
    times = np.arange(T) / rec["fps"] if times is None else np.asarray(times)
    if tracks is None:
        R_body, trans, body_rms = body_tracks(model, rec, body_calib, frame_numbers)
    else:
        R_body, trans, body_rms = (np.asarray(tracks[key]) for key in ("R", "trans", "rmse"))
    if R_body.shape != (T, 55, 3, 3) or trans.shape != (T, 3):
        raise ValueError("Body tracking cache does not match source frames")
    v_rest, j_rest = model.shaped(betas)
    normals = model.vertex_normals(v_rest)

    # Choose diverse hand configurations in each native wrist coordinate frame.
    features = np.empty((T, 60))
    for i in range(T):
        rw, c = model.fk(R_body[i], j_rest)
        parts = []
        for side, wrist in (("L", 20), ("R", 21)):
            indices = [names.index(n) for n in finger_names if n.startswith(side)]
            parts.append((rec["pos"][i, indices] - c[wrist] - trans[i]) @ rw[wrist])
        features[i] = np.concatenate(parts).reshape(-1)
    good = np.flatnonzero(np.isfinite(features).all(1) & np.isfinite(body_rms) & (body_rms < .025))
    if len(good) < 12:
        raise ValueError("Too few clean frames with both hands visible")
    chosen = [int(np.argmin(np.abs(times[good] - 1.5)))]
    distances = np.linalg.norm(features[good] - features[good[chosen[0]]], axis=1)
    for _ in range(min(n_frames, len(good)) - 1):
        index = int(np.argmax(distances))
        chosen.append(index)
        distances = np.minimum(distances, np.linalg.norm(features[good] - features[good[index]], axis=1))
        distances[chosen] = -1
    selected = np.sort(good[chosen])
    F, M = len(selected), len(names)
    reference = int(selected[np.argmin(np.abs(times[selected] - 1.5))])
    z_initial = prior.encode(np.broadcast_to(np.eye(3), (55, 3, 3)))
    initial_R = prior.apply(R_body[reference], z_initial)
    initial_vertices = model.vertices(initial_R, trans[reference], v_rest, j_rest)
    initial_normals = model.vertex_normals(initial_vertices)
    static_obs = np.nanmedian(rec["pos"][(times >= .4) & (times < 2.6)], axis=0)
    if not np.isfinite(static_obs[finger_indices]).all():
        raise ValueError("Finger markers are missing from the static calibration window")
    vertex_ids = np.empty(M, np.int64)
    frozen_offsets = np.zeros((M, 3))
    for i, name in enumerate(names):
        if name in body_lookup:
            j = body_lookup[name]
            vertex_ids[i] = body_calib["vertex_ids"][j]
            frozen_offsets[i] = body_calib["offsets"][j]
        else:
            candidates = np.asarray(regions[name], np.int64)
            surface = initial_vertices[candidates] + .005 * initial_normals[candidates]
            vertex_ids[i] = candidates[np.argmin(np.linalg.norm(surface - static_obs[i], axis=1))]

    dtype = torch.float64
    tb = TorchBody(model)
    body_R = torch.tensor(R_body[selected], dtype=dtype)
    base_translation = torch.tensor(trans[selected], dtype=dtype)
    body_delta = torch.zeros((F, 22, 3), dtype=dtype, requires_grad=True)
    trans_delta = torch.zeros((F, 3), dtype=dtype, requires_grad=True)
    observed = torch.tensor(np.nan_to_num(rec["pos"][selected]), dtype=dtype)
    valid = torch.tensor(rec["valid"][selected], dtype=dtype)
    mean = torch.tensor(prior.mean, dtype=dtype)
    basis = torch.tensor(prior.basis, dtype=dtype)
    z = torch.tensor(np.tile(z_initial[None], (F, 1, 1)), dtype=dtype, requires_grad=True)
    h_raw = torch.full((20,), -np.log(2.), dtype=dtype, requires_grad=True)
    tangent_raw = torch.zeros((20, 3), dtype=dtype, requires_grad=True)
    finger_t = torch.as_tensor(finger_indices)
    v_t, j_t = torch.tensor(v_rest, dtype=dtype), torch.tensor(j_rest, dtype=dtype)
    fixed_off = torch.tensor(frozen_offsets, dtype=dtype)
    order = np.argsort(times[selected])
    dt = torch.tensor(np.maximum(np.diff(times[selected][order]), .03), dtype=dtype)

    def rotations():
        aa = mean[None] + torch.einsum("fsn,san->fsa", z, basis)
        hand_R = batch_rodrigues(aa.reshape(F, 30, 3))
        refined_body = body_R[:, :22] @ batch_rodrigues(body_delta)
        return torch.cat((refined_body, body_R[:, 22:25], hand_R), 1)

    def translation():
        return base_translation + trans_delta

    per_marker = None
    for rd in range(rounds):
        ids = torch.as_tensor(vertex_ids)
        weight, pd = tb.weights[ids], tb.posedirs[:, ids]
        normal = torch.tensor(normals[vertex_ids[finger_indices]], dtype=dtype)

        def offsets():
            height = .015 * torch.sigmoid(h_raw)
            tangent = .004 * torch.tanh(tangent_raw)
            tangent = tangent - (tangent * normal).sum(-1, keepdim=True) * normal
            hand_off = height[:, None] * normal + tangent
            return fixed_off.index_copy(0, finger_t, hand_off), height, tangent

        def loss():
            off, height, tangent = offsets()
            pred = tb.skin(rotations(), translation(), v_t[ids] + off, weight, pd, j_t)
            distance = (pred - observed).norm(dim=-1)
            huber = torch.where(distance < .015, .5 * distance.square(), .015 * (distance - .0075))
            data = (huber * valid).sum() / F * 1e4
            reg = .02 * z.square().sum((1, 2)).mean()
            reg = reg + .5 * body_delta.square().sum((1, 2)).mean() + 1e3 * trans_delta.square().sum(1).mean()
            reg = reg + 5e3 * (height - .005).square().sum() + 1e4 * tangent.square().sum()
            reg = reg + .1 * torch.relu(z.abs() - 4).square().sum((1, 2)).mean()
            if F > 1:
                reg = reg + .002 * (((z[order[1:]] - z[order[:-1]]).square().sum((1, 2))) / dt).mean()
                reg = reg + .02 * (((body_delta[order[1:]] - body_delta[order[:-1]]).square().sum((1, 2))) / dt).mean()
            return data + reg, distance, pred

        parameters = [z, h_raw, tangent_raw, body_delta, trans_delta]
        optimizer = torch.optim.Adam([
            {"params": [z, h_raw, tangent_raw], "lr": .04},
            {"params": [body_delta], "lr": .01},
            {"params": [trans_delta], "lr": .001},
        ])
        for step in range(steps):
            optimizer.zero_grad()
            value, _, _ = loss()
            value.backward()
            optimizer.step()
            if step % 100 == 0:
                print(f"[hands] round {rd+1}/{rounds}, Adam {step}/{steps}, loss={value.item():.4f}", flush=True)
        optimizer = torch.optim.LBFGS(parameters, max_iter=100, history_size=20,
                                      line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            value, _, _ = loss()
            value.backward()
            return value

        optimizer.step(closure)
        value, distance, _ = loss()
        final_R = rotations().detach().numpy()
        final_trans = translation().detach().numpy()
        final_offsets, heights, tangent = offsets()
        final_offsets = final_offsets.detach().numpy()
        error = distance.detach().numpy()
        mask = rec["valid"][selected]
        per_marker = np.sqrt((np.where(mask, error, 0) ** 2).sum(0) / np.maximum(mask.sum(0), 1))
        print(f"[hands] round {rd+1}: finger RMS={np.sqrt(np.mean(error[:,finger_indices]**2))*1000:.2f} mm; "
              f"body RMS={np.sqrt(np.mean(error[:,body_indices]**2))*1000:.2f} mm", flush=True)
        if rd + 1 == rounds:
            break
        # EM update stays in the labelled finger/segment, with bounded attachment offsets.
        rest_observations = []
        for f in range(F):
            rw, c = model.fk(final_R[f], j_rest)
            tr = np.einsum("mj,jab->mab", model.weights[vertex_ids], rw)
            at = c - np.einsum("jab,jb->ja", rw, j_rest)
            tt = model.weights[vertex_ids] @ at
            q = np.linalg.solve(tr, (rec["pos"][selected[f]] - final_trans[f] - tt)[..., None])[..., 0]
            q -= np.einsum("p,pmc->mc", model.pose_feature(final_R[f]), model.posedirs[:, vertex_ids])
            rest_observations.append(q)
        median = np.nanmedian(rest_observations, axis=0)
        height_np = heights.detach().numpy()
        changed = 0
        for h, m in enumerate(finger_indices):
            candidates = np.asarray(regions[names[m]], np.int64)
            surface = v_rest[candidates] + height_np[h] * normals[candidates]
            new = int(candidates[np.argmin(np.linalg.norm(surface - median[m], axis=1))])
            changed += new != vertex_ids[m]
            vertex_ids[m] = new
        print(f"[hands] reassigned {changed}/20 finger attachments within their regions", flush=True)
        new_normal = normals[vertex_ids[finger_indices]]
        delta = median[finger_indices] - v_rest[vertex_ids[finger_indices]]
        height = np.clip((delta * new_normal).sum(-1), .0001, .0149)
        tangent_init = delta - height[:, None] * new_normal
        with torch.no_grad():
            h_raw.copy_(torch.tensor(np.log(height / (.015-height)), dtype=dtype))
            tangent_raw.copy_(torch.tensor(np.arctanh(np.clip(tangent_init/.004, -.95, .95)), dtype=dtype))

    assert np.array_equal(vertex_ids[body_indices], np.asarray(body_calib["vertex_ids"]))
    assert np.array_equal(final_offsets[body_indices], np.asarray(body_calib["offsets"]))
    initial_latent = z.detach().numpy()[np.argmin(np.abs(times[selected] - 1.5))]
    result = dict(marker_names=np.array(names), vertex_ids=vertex_ids, offsets=final_offsets, betas=betas,
        marker_radius=float(body_calib["marker_radius"]), per_marker_rms=per_marker,
        static_pose=str(body_calib["static_pose"]), marker_scope="body_hands",
        finger_marker_names=np.array(finger_names),
        hand_pca_components=prior.n_components, hand_pca_initial_latent=initial_latent,
        hand_prior_weight=1e-5, finger_normal_bounds_m=np.array([0., .015]), finger_tangent_bound_m=.004,
        selected_hand_frames=selected, selected_hand_R=final_R,
        selected_hand_trans=final_trans, selected_hand_latent=z.detach().numpy())
    diagnostics = dict(elapsed_seconds=time.perf_counter()-started, selected_frames=selected.tolist(),
        selected_times_seconds=times[selected].tolist(), body_parameters_preserved=True,
        body_pose_refined=True,
        body_pose_delta_max_degrees=float(torch.linalg.vector_norm(body_delta.detach(),dim=-1).max()*180/np.pi),
        translation_delta_max_mm=float(torch.linalg.vector_norm(trans_delta.detach(),dim=-1).max()*1000),
        finger_marker_rms_mm={n:float(per_marker[names.index(n)]*1000) for n in finger_names},
        body_marker_median_mm=float(np.median(per_marker[body_indices])*1000),
        finger_marker_median_mm=float(np.median(per_marker[finger_indices])*1000),
        finger_marker_max_mm=float(np.max(per_marker[finger_indices])*1000),
        hand_latent_max_abs=float(z.detach().abs().max()),
        hand_parameterisation=f"{prior.n_components} learned PCA coefficients per hand, native SMPL-X55 FK/LBS",
        attachment_prior="normal 0..15mm, initial 5mm; tangential bounded to 4mm per coordinate",
        acceptance="PENDING_NATIVE_HAND_VISUAL_REVIEW")
    return result, diagnostics
