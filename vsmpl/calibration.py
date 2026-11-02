"""Offline subject calibration ("MoSh-lite").

Estimates, from a short take (static T/A-pose + range of motion or free movement):
  * body shape betas
  * for every marker: the SMPL vertex it sits on + a rest-space 3D offset (marker radius, misplacement)

Pipeline
  A. T-pose fit: average the static window, fit global orientation / translation / betas / small body pose to the
     mesh surface (marker -> nearest vertex, no labels needed), trying several facing directions.
  B. Correspondences: each marker -> nearest offset vertex (or a user-supplied {label: vertex_id} map).
  C. Track the whole take with the real-time solver, pick K maximally different poses (farthest-point sampling).
  D. Joint optimisation over the K frames: betas, marker offsets and per-frame poses (torch, L-BFGS).
  E. Re-assign each marker to the vertex nearest its refined rest position; repeat C-D.
"""
import json
import numpy as np
import torch

from .body_model import BodyModel, MarkerRig
from .rotations import exp_so3, log_so3, rot_y
from .solver import RealtimeSolver, SolverConfig
from .torch_model import TorchBody, batch_rodrigues

FACING = {"+z": 0.0, "-z": np.pi, "+x": np.pi / 2, "-x": -np.pi / 2}


def static_reference_pose(num_joints, static_pose="t", arm_angle_deg=45.0):
    """An initialisation prior, not a claim that the recorded actor has exact joint angles."""
    if static_pose not in ("t", "a"):
        raise ValueError("static_pose must be 't' or 'a'")
    if not 0 <= arm_angle_deg <= 90:
        raise ValueError("arm_angle_deg must be between 0 and 90")
    pose = np.zeros((num_joints, 3))
    if static_pose == "a":
        if num_joints <= 17:
            raise ValueError("A-pose requires the SMPL body shoulder joints")
        pose[16, 2] = -np.radians(arm_angle_deg)
        pose[17, 2] = np.radians(arm_angle_deg)
    return pose


def _huber(d, delta):
    return torch.where(d < delta, 0.5 * d ** 2, delta * (d - 0.5 * delta))


def tpose_average(pos, valid, start, end, min_frac=0.5):
    p, v = pos[start:end], valid[start:end]
    cnt = v.sum(0)
    ok = cnt >= max(1, min_frac * len(p))
    avg = np.where(v[..., None], p, 0).sum(0) / np.maximum(cnt, 1)[:, None]
    avg[~ok] = np.nan
    return avg, ok


def tpose_joint_weights(J, arm_w=0.1):
    """Confidence that a joint is at its rest angle in a T-pose (joints 1..J-1). Arms vary more between people."""
    w = np.ones(J - 1)
    for j in range(13, J):  # collars, shoulders, elbows, wrists, and hands/fingers/jaw/eyes (>= 22)
        if j != 15:  # 15 = head
            w[j - 1] = arm_w
    return w


# ----------------------------------------------------------------------------- stage A
def fit_tpose(model: BodyModel, obs, ok, yaw, marker_radius=0.0095, iters=400, vids=None, verbose=False,
              reference_pose=None, regions=None):
    """Fit root/trans/betas/body pose to a static marker set. If `vids` is given use those correspondences,
    otherwise nearest-vertex (correspondence-free). Returns dict with params and final loss."""
    tb = TorchBody(model)
    J = model.num_joints
    obs_t = torch.tensor(obs[ok])
    betas = torch.zeros(model.num_betas, dtype=torch.float64, requires_grad=True)
    root = torch.tensor([0.0, yaw, 0.0], dtype=torch.float64, requires_grad=True)
    body_ref = np.zeros((J - 1, 3)) if reference_pose is None else np.asarray(reference_pose)[1:]
    body_target = torch.tensor(body_ref, dtype=torch.float64)
    body = body_target.clone().requires_grad_(True)
    v_rest0, j_rest0 = model.shaped(np.zeros(model.num_betas))
    R0 = exp_so3(np.concatenate([np.zeros((1, 3)), body_ref]))
    v0 = model.vertices(R0, np.zeros(3), v_rest0, j_rest0)
    trans0 = obs[ok].mean(0) - rot_y(yaw) @ v0.mean(0)
    trans = torch.tensor(trans0, requires_grad=True)
    vid_t = None if vids is None else torch.as_tensor(np.asarray(vids)[ok])
    allowed = None
    if regions is not None:
        mask = np.zeros((len(obs), len(v0)), bool)
        for i, indices in enumerate(regions):
            mask[i, indices] = True
        allowed = torch.as_tensor(mask[ok])
    # per-joint prior: arms (collar..hand, joints 13+ except head) may deviate from a perfect T-pose
    body_w = torch.tensor(tpose_joint_weights(J, 0.05))[:, None]

    def loss_fn(stage):
        R = batch_rodrigues(torch.cat([root[None], body], 0))[None]
        verts, v_rest, _ = tb.vertices(R, trans[None], betas)
        n = tb.vertex_normals(verts[0])
        surf = verts[0] + marker_radius * n
        if vid_t is None:
            distances = torch.cdist(obs_t, surf)
            if allowed is not None:
                distances = distances.masked_fill(~allowed, float("inf"))
            d = distances.min(1).values
        else:
            d = (surf[vid_t] - obs_t).norm(dim=-1)
        L = _huber(d, 0.03).mean() * 1e4
        L = L + 1e-1 * (betas ** 2).sum() + (body_w * (body - body_target) ** 2).sum()
        return L, d

    for stage, (params, n_it) in enumerate([([root, trans, betas], iters // 3), ([root, trans, betas, body], iters - iters // 3)]):
        opt = torch.optim.Adam(params, lr=0.02)
        for it in range(n_it):
            opt.zero_grad()
            L, d = loss_fn(stage)
            L.backward()
            opt.step()
    L, d = loss_fn(1)
    if verbose:
        print(f"    yaw {np.degrees(yaw):6.1f} deg: loss {L.item():8.3f}, marker-surface rms {d.pow(2).mean().sqrt().item() * 1000:.1f} mm")
    return {"loss": float(L.item()), "betas": betas.detach().numpy().copy(), "root": root.detach().numpy().copy(),
            "body": body.detach().numpy().copy(), "trans": trans.detach().numpy().copy(),
            "rms": float(d.pow(2).mean().sqrt().item())}


def assign_vertices(model, fit, obs, ok, marker_radius, regions=None):
    """Nearest offset vertex for each (valid) marker in the fitted T-pose."""
    R = exp_so3(np.concatenate([fit["root"][None], fit["body"]], 0))
    v_rest, J_rest = model.shaped(fit["betas"])
    verts = model.vertices(R, fit["trans"], v_rest, J_rest)
    surf = verts + marker_radius * model.vertex_normals(verts)
    vids = np.full(len(obs), -1)
    dist = np.full(len(obs), np.nan)
    for m in np.nonzero(ok)[0]:
        candidates = np.arange(len(surf)) if regions is None else regions[m]
        d = np.linalg.norm(surf[candidates] - obs[m], axis=-1)
        index = int(np.argmin(d))
        vids[m] = int(candidates[index])
        dist[m] = d[index]
    return vids, dist


# ----------------------------------------------------------------------------- stage C
def track_take(rig, pos, valid, cfg=None, start=0):
    solver = RealtimeSolver(rig, cfg or SolverConfig())
    T = len(pos)
    Rs = np.zeros((T, rig.model.num_joints, 3, 3))
    ts = np.zeros((T, 3))
    rmse = np.full(T, np.inf)
    clean = np.zeros(T, bool)  # every visible marker was used (no outlier / label swap)
    for i in range(start, T):
        r = solver.step(pos[i], valid[i], frame_no=i)
        Rs[i], ts[i], rmse[i] = r.R, r.trans, r.rmse
        clean[i] = r.status == "track" and r.n_inliers == r.n_valid
    return Rs, ts, rmse, clean


def farthest_point_frames(Rs, cand, k, first):
    """Pick k frames from `cand` maximising pose diversity (body joints, axis-angle space)."""
    feats = log_so3(Rs[cand][:, 1:]).reshape(len(cand), -1)
    chosen = [int(np.argmin(np.abs(cand - first)))]
    d = np.linalg.norm(feats - feats[chosen[0]], axis=1)
    for _ in range(min(k, len(cand)) - 1):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(feats - feats[i], axis=1))
    return cand[np.array(chosen)]


# ----------------------------------------------------------------------------- stage D
def refine(model, vids, offsets, betas, frames_obs, frames_valid, Rs, ts, marker_radius, iters=200,
           lam_beta=1e-2, lam_off_n=3e4, lam_off_t=300.0, lam_pose=1e-2, anchor=None, lam_anchor=30.0,
           verbose=True, anchor_pose=None):
    """Anchor the static frame to T-rest, or to the fitted A-pose supplied in `anchor_pose`.
    This constrains the ambiguity between segment orientation and marker attachment location."""
    tb = TorchBody(model)
    vid_t = torch.as_tensor(vids)
    W = tb.weights[vid_t]
    PD = tb.posedirs[:, vid_t]
    obs = torch.tensor(np.nan_to_num(frames_obs))
    val = torch.tensor(frames_valid, dtype=torch.float64)
    b = torch.tensor(betas, requires_grad=True)
    off = torch.tensor(offsets, requires_grad=True)
    aa = torch.tensor(log_so3(Rs), requires_grad=True)  # (F,J,3)
    tr = torch.tensor(ts, requires_grad=True)
    anchor_w = torch.tensor(tpose_joint_weights(model.num_joints))[:, None]
    anchor_R = None if anchor_pose is None else batch_rodrigues(torch.tensor(
        np.asarray(anchor_pose)[1:], dtype=torch.float64))

    def closure_loss():
        v_rest, J_rest = tb.shaped(b)
        n = tb.vertex_normals(v_rest)
        p_rest = v_rest[vid_t] + off
        pred = tb.skin(batch_rodrigues(aa), tr, p_rest, W, PD, J_rest)
        d = (pred - obs).norm(dim=-1)
        # data: mean over frames, summed over markers -> per-marker scale comparable to the offset prior
        data = (_huber(d, 0.02) * val).sum() / len(aa) * 1e4
        nv = n[vid_t]
        o_n = (off * nv).sum(-1)  # normal component: stays near the marker radius
        o_t = off - o_n[:, None] * nv  # tangential component: marker may slide along the skin
        reg = lam_beta * (b ** 2).sum() + lam_off_n * ((o_n - marker_radius) ** 2).sum() \
            + lam_off_t * (o_t ** 2).sum() + lam_pose * (aa[:, 1:] ** 2).sum() / len(aa)
        if anchor is not None:
            if anchor_R is None:
                reg = reg + lam_anchor * (anchor_w * aa[anchor, 1:] ** 2).sum()
            else:
                # Keep the fitted static A-pose; do not force the anchor frame back into a T-pose.
                relative = anchor_R.transpose(-1, -2) @ batch_rodrigues(aa[anchor, 1:])
                distance = 0.5 * ((relative - torch.eye(3, dtype=torch.float64)) ** 2).sum((-1, -2))
                reg = reg + lam_anchor * (anchor_w[:, 0] * distance).sum()
        return data + reg, d

    opt = torch.optim.LBFGS([b, off, aa, tr], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        L, _ = closure_loss()
        L.backward()
        return L

    opt.step(closure)
    L, d = closure_loss()
    d = d.detach().numpy()
    per_marker = np.sqrt((np.where(frames_valid, d, 0) ** 2).sum(0) / np.maximum(frames_valid.sum(0), 1))
    if verbose:
        print(f"    refine: loss {L.item():.3f}, marker rms {np.sqrt((d[frames_valid] ** 2).mean()) * 1000:.1f} mm")
    return b.detach().numpy(), off.detach().numpy(), exp_so3(aa.detach().numpy()), tr.detach().numpy(), per_marker


def inverse_skin_reassign(model, rig, pos, valid, Rs, ts, frames, marker_radius, inner=2, regions=None):
    """EM-style correspondence update: map every observation back to rest space with the tracked pose,
    take the per-marker median, then snap to the nearest offset surface vertex (re-done `inner` times since
    the skinning weights change with the vertex)."""
    v_rest, J_rest = model.shaped(rig.betas)
    surf = v_rest + marker_radius * model.vertex_normals(v_rest)
    vids = rig.vertex_ids.copy()
    M = len(vids)
    Rws, Ats, pfs = [], [], []
    for f in frames:
        Rw, c = model.fk(Rs[f], J_rest)
        Rws.append(Rw)
        Ats.append(c - np.einsum("jab,jb->ja", Rw, J_rest))
        pfs.append(model.pose_feature(Rs[f]))
    Rws, Ats, pfs = np.array(Rws), np.array(Ats), np.array(pfs)
    obs = pos[frames] - ts[frames][:, None]  # (F,M,3)
    val = valid[frames]
    p_rest = np.zeros((M, 3))
    for _ in range(inner):
        W = model.weights[vids]  # (M,J)
        TR = np.einsum("mj,fjab->fmab", W, Rws)
        Tt = np.einsum("mj,fja->fma", W, Ats)
        q = np.linalg.solve(TR, (np.nan_to_num(obs) - Tt)[..., None])[..., 0]  # (F,M,3)
        q -= np.einsum("fp,pmc->fmc", pfs, model.posedirs[:, vids])
        for m in range(M):
            qm = q[val[:, m], m]
            if len(qm):
                p_rest[m] = np.median(qm, 0)
        vids = np.array([int(np.argmin(np.linalg.norm(surf - p, axis=-1))) if regions is None else
                         int(regions[m][np.argmin(np.linalg.norm(surf[regions[m]] - p, axis=-1))])
                         for m, p in enumerate(p_rest)])
    return vids, p_rest - v_rest[vids]


def reassign(model, vids, offsets, betas, marker_radius, regions=None):
    """Move each marker to the vertex whose offset surface point is closest to its refined rest position."""
    v_rest, _ = model.shaped(betas)
    surf = v_rest + marker_radius * model.vertex_normals(v_rest)
    p_rest = v_rest[vids] + offsets
    new = np.array([int(np.argmin(np.linalg.norm(surf - p, axis=-1))) if regions is None else
                    int(regions[m][np.argmin(np.linalg.norm(surf[regions[m]] - p, axis=-1))])
                    for m, p in enumerate(p_rest)])
    return new, p_rest - v_rest[new]


# ----------------------------------------------------------------------------- driver
def calibrate(model, rec, tpose=(0, 100), facing="auto", marker_map=None, marker_radius=0.0095,
              n_frames=40, rounds=2, min_valid_frac=0.3, refine_kw=None, verbose=True,
              static_pose="t", arm_angle_deg=45.0, marker_regions=None):
    names, pos, valid = rec["names"], rec["pos"], rec["valid"]
    keep = valid.mean(0) >= min_valid_frac
    if (~keep).any():
        print(f"[calib] dropping rarely-visible markers: {[n for n, k in zip(names, keep) if not k]}")
    names = [n for n, k in zip(names, keep) if k]
    pos, valid = pos[:, keep], valid[:, keep]
    M = len(names)
    regions = None
    if marker_regions is not None:
        if marker_map is not None:
            raise ValueError("Choose marker_regions or a fixed marker_map, not both")
        raw_regions = json.load(open(marker_regions)) if isinstance(marker_regions, str) else marker_regions
        regions = []
        for name in names:
            indices = np.unique(np.asarray(raw_regions.get(name, []), dtype=np.int64))
            if not len(indices) or indices.min() < 0 or indices.max() >= len(model.v_template):
                raise ValueError(f"Invalid or missing vertex region for {name}")
            regions.append(indices)
        print(f"[calib] caller-provided anatomical vertex regions for {M} markers")

    # ---- A
    obs0, ok0 = tpose_average(pos, valid, *tpose)
    reference = static_reference_pose(model.num_joints, static_pose, arm_angle_deg)
    print(f"[calib] {static_pose.upper()}-pose window frames {tpose[0]}-{tpose[1]}: {ok0.sum()}/{M} markers usable")
    map_vids = None
    if marker_map:
        mm = json.load(open(marker_map)) if isinstance(marker_map, str) else marker_map
        map_vids = np.array([int(mm.get(n, -1)) for n in names])
        if (map_vids < 0).any():
            raise ValueError(f"marker map lacks labels: {[n for n, v in zip(names, map_vids) if v < 0]}")
    yaws = [FACING[facing]] if facing in FACING else [i * np.pi / 4 for i in range(8)]
    print(f"[calib] stage A: {static_pose.upper()}-pose fit ({len(yaws)} facing hypotheses)")
    fits = [fit_tpose(model, obs0, ok0, y, marker_radius, vids=map_vids, verbose=verbose,
                      reference_pose=reference, regions=regions) for y in yaws]
    order = np.argsort([f["loss"] for f in fits])
    if map_vids is not None:
        if (~ok0).any():
            print(f"[calib] note: {int((~ok0).sum())} markers not visible in the T-pose window")
    elif (~ok0).any():
        raise ValueError(f"markers not visible in T-pose window, cannot auto-assign: "
                         f"{[n for n, u in zip(names, ~ok0) if u]} (extend --tpose or give --marker-map)")

    def init_from_fit(f):
        if map_vids is None:
            vids, dist = assign_vertices(model, f, obs0, ok0, marker_radius, regions=regions)
        else:
            vids, dist = map_vids, None
        v_rest, _ = model.shaped(f["betas"])
        return vids, marker_radius * model.vertex_normals(v_rest)[vids], f["betas"], dist

    # Facing disambiguation: a static T-pose is nearly front/back symmetric, motion is not (knees, elbows,
    # feet). Track the start of the take with the top distinct hypotheses and keep the one that explains it best.
    cands = []
    for i in order:
        Rr = exp_so3(fits[i]["root"])
        if all(np.degrees(np.linalg.norm(log_so3(Rr.T @ exp_so3(fits[j]["root"])))) > 30 for j in cands):
            cands.append(i)
        if len(cands) == 3:
            break
    if len(cands) > 1 and map_vids is None:
        end_f = min(len(pos), tpose[1] + int(10 * (rec.get("fps") or 100)))  # ~10 s of motion
        scores = []
        for i in cands:
            vids_i, off_i, b_i, _ = init_from_fit(fits[i])
            rig_i = MarkerRig(model, names, vids_i, off_i, b_i)
            _, _, rmse_i, _ = track_take(rig_i, pos[:end_f], valid[:end_f], start=tpose[0])
            fin = rmse_i[np.isfinite(rmse_i)]
            scores.append(np.median(fin) if len(fin) else np.inf)
            if verbose:
                print(f"    hypothesis yaw {np.degrees(yaws[i]):5.0f} deg -> tracking rms {scores[-1] * 1000:.1f} mm")
        fit = fits[cands[int(np.argmin(scores))]]
        srt = np.sort(scores)
        if srt[1] < 1.3 * srt[0]:
            print("[calib] WARNING: facing still ambiguous after tracking test; consider --facing")
    else:
        fit = fits[order[0]]

    # ---- B
    vids, offsets, betas, dist = init_from_fit(fit)
    anchor_pose = np.concatenate([np.zeros((1, 3)), fit["body"]]) if static_pose == "a" else None
    if dist is not None and (dist > 0.04).any():
        print(f"[calib] WARNING: markers far from the fitted surface: "
              f"{[(n, round(float(x) * 1000)) for n, x in zip(names, dist) if x > 0.04]} mm")

    # ---- C/D/E
    tp_mid = (tpose[0] + tpose[1]) // 2
    per_marker = None
    for rd in range(rounds):
        rig = MarkerRig(model, names, vids, offsets, betas)
        print(f"[calib] round {rd + 1}/{rounds}: tracking {len(pos)} frames ...")
        Rs, ts, rmse, clean = track_take(rig, pos, valid, start=0)
        good = np.nonzero((rmse < 0.03) & clean & (valid.mean(1) >= 0.8))[0]
        if len(good) < 5:
            raise RuntimeError("tracking failed on most frames; check facing / marker labels")
        sel = farthest_point_frames(Rs, good, n_frames, tp_mid)
        print(f"    tracked rms {np.median(rmse[good]) * 1000:.1f} mm (median), using {len(sel)} diverse frames")
        if map_vids is None:
            sub = good[:: max(1, len(good) // 400)]
            new_vids, offsets = inverse_skin_reassign(model, rig, pos, valid, Rs, ts, sub, marker_radius, regions=regions)
            print(f"    inverse-skinning: {int((new_vids != vids).sum())} markers moved to new vertices")
            vids = new_vids
        anchor = 0 if tpose[0] <= sel[0] < tpose[1] else None
        if verbose:
            print(f"    anchor frame: {sel[0] if anchor is not None else None}")
        betas, offsets, _, _, per_marker = refine(model, vids, offsets, betas, pos[sel], valid[sel], Rs[sel], ts[sel],
                                                  marker_radius, anchor=anchor, verbose=verbose, anchor_pose=anchor_pose,
                                                  **(refine_kw or {}))
        if map_vids is None:
            vids, offsets = reassign(model, vids, offsets, betas, marker_radius, regions=regions)

    bad = per_marker > 0.015
    if bad.any():
        print(f"[calib] WARNING: high residual markers (check labels/placement): "
              f"{[(n, round(float(r) * 1000, 1)) for n, r, b in zip(names, per_marker, bad) if b]} mm")
    return {"marker_names": np.array(names), "vertex_ids": vids, "offsets": offsets, "betas": betas,
            "marker_radius": marker_radius, "per_marker_rms": per_marker, "static_pose": static_pose,
            "static_reference_pose": reference, "initial_static_fit_pose": np.concatenate([fit["root"][None], fit["body"]])}
