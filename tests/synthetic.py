"""A synthetic SMPL-like body (same 24-joint kinematic tree, capsule geometry, linear shape and pose
blendshapes) plus a motion/marker simulator. Lets the pipeline be tested without the licensed SMPL files."""
import numpy as np

from vsmpl.body_model import BodyModel
from vsmpl.rotations import exp_so3

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
JOINTS = np.array([
    [0, 0, 0], [.09, -.08, 0], [-.09, -.08, 0], [0, .11, 0], [.10, -.47, 0], [-.10, -.47, 0],
    [0, .24, 0], [.10, -.87, -.02], [-.10, -.87, -.02], [0, .30, 0], [.11, -.93, .11], [-.11, -.93, .11],
    [0, .52, 0], [.07, .43, 0], [-.07, .43, 0], [0, .62, .03], [.18, .45, 0], [-.18, .45, 0],
    [.44, .45, 0], [-.44, .45, 0], [.69, .45, 0], [-.69, .45, 0], [.78, .45, 0], [-.78, .45, 0]], float)

# (start joint, end point or joint index, owner joint, radius_x, radius_z)
def _segments():
    J = JOINTS
    S = []
    S += [(0, J[3], 0, .15, .11), (3, J[6], 3, .15, .11), (6, J[9], 6, .16, .11), (9, J[12], 9, .17, .11)]
    for s in (1, -1):
        L = 0 if s > 0 else 1
        hip, knee, ank, foot = 1 + L, 4 + L, 7 + L, 10 + L
        S += [(hip, J[knee], hip, .075, .075), (knee, J[ank], knee, .055, .055), (ank, J[foot], ank, .045, .05),
              (foot, J[foot] + [0, 0, .08], foot, .04, .03)]
        col, sho, elb, wri, hand = 13 + L, 16 + L, 18 + L, 20 + L, 22 + L
        S += [(col, J[sho], col, .05, .05), (sho, J[elb], sho, .05, .05), (elb, J[wri], elb, .04, .04),
              (wri, J[hand], wri, .03, .035), (hand, J[hand] + [s * .1, 0, 0], hand, .025, .04)]
    S += [(12, J[15], 12, .055, .055), (15, J[15] + [0, .18, 0], 15, .09, .10)]
    return S


def make_body(n_rings=10, n_sec=16, seed=0, num_betas=10):
    rng = np.random.default_rng(seed)
    V, W, F, radial, ring_ids = [], [], [], [], {}
    for si, (a, b, owner, rx, rz) in enumerate(_segments()):
        p0 = JOINTS[a]
        p1 = b
        ax = p1 - p0
        L = np.linalg.norm(ax)
        d = ax / L
        up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
        e1 = np.cross(d, up); e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        # child joint index (if the end is a joint)
        child = [j for j in range(24) if PARENTS[j] == owner and np.allclose(JOINTS[j], p1)]
        base = len(V)
        for r in range(n_rings):
            s = r / (n_rings - 1)
            c = p0 + s * ax
            for k in range(n_sec):
                ang = 2 * np.pi * k / n_sec
                rad = np.cos(ang) * e1 * (rx if abs(e1[0]) > .5 else rz) + np.sin(ang) * e2 * (rz if abs(e1[0]) > .5 else rx)
                V.append(c + rad)
                radial.append(rad / np.linalg.norm(rad))
                w = np.zeros(24)
                wc = np.clip((s - 0.7) / 0.6, 0, 0.5) if child else 0.0
                w[owner] = 1 - wc
                if child:
                    w[child[0]] += wc
                W.append(w)
            if r == 0:
                ring_ids.setdefault(("start", a), list(range(len(V) - n_sec, len(V))))
            if r == n_rings - 1 and child:
                ring_ids.setdefault(("end", child[0]), list(range(len(V) - n_sec, len(V))))
        for r in range(n_rings - 1):
            for k in range(n_sec):
                i0 = base + r * n_sec + k
                i1 = base + r * n_sec + (k + 1) % n_sec
                F += [[i0, i1, i0 + n_sec], [i1, i1 + n_sec, i0 + n_sec]]
    V, W, radial = np.array(V), np.array(W), np.array(radial)
    nV = len(V)
    Jreg = np.zeros((24, nV))
    for j in range(24):
        ids = ring_ids.get(("start", j)) or ring_ids.get(("end", j))
        Jreg[j, ids] = 1.0 / len(ids)
    # shape blendshapes
    S = np.zeros((nV, 3, num_betas))
    S[:, :, 0] = 0.06 * V  # overall scale
    S[:, :, 1] = 0.012 * radial  # girth
    leg = V[:, 1] < -0.08
    S[leg, 1, 2] = 0.06 * (V[leg, 1] + 0.08)  # leg length
    arm = np.abs(V[:, 0]) > 0.18
    S[arm, 0, 3] = 0.06 * (V[arm, 0] - np.sign(V[arm, 0]) * 0.18)  # arm length
    tor = (V[:, 1] > 0) & ~arm
    S[tor, 1, 4] = 0.05 * V[tor, 1]  # torso length
    for b in range(5, num_betas):
        S[:, :, b] = 0.003 * rng.standard_normal(3) * radial  # weak extra modes
    P = 23 * 9
    PD = 4e-4 * rng.standard_normal((P, nV, 3))
    Jt = Jreg @ V
    return BodyModel(V, S, PD, Jreg, W, PARENTS, np.array(F), "synthetic"), Jt


# --------------------------------------------------------------- markers & motion
# markers per segment (owner joint), roughly like a full-body production marker set (~53 markers):
# pelvis/chest/head 3-4, thighs/shanks/upper arms/forearms 2, feet 3 (ankle, heel, mid-foot) + toe 1,
# hands 2 + finger 1, collars 1. Every rigid segment chain gets >= 3 non-collinear constraints.
N_PER_SEG = {0: 4, 3: 2, 6: 2, 9: 3, 12: 1, 15: 4, 13: 1, 14: 1, 7: 3, 8: 3, 10: 1, 11: 1,
             20: 2, 21: 2, 22: 1, 23: 1}


def pick_markers(model, n_per_seg=2, seed=1):
    """Pick marker vertices: a few per body segment, at varied heights/angles."""
    rng = np.random.default_rng(seed)
    owners = np.argmax(model.weights, 1)
    vids = []
    for j in range(24):
        cand = np.nonzero((owners == j) & (model.weights.max(1) > 0.99))[0]
        if len(cand) == 0:
            continue
        k = N_PER_SEG.get(j, n_per_seg)
        vids += list(rng.choice(cand, size=min(k, len(cand)), replace=False))
    return np.array(sorted(vids))


AMP = np.zeros(24)
AMP[[1, 2]] = 0.9; AMP[[4, 5]] = 1.0; AMP[[7, 8]] = 0.4; AMP[[3, 6, 9]] = 0.25; AMP[[12, 15]] = 0.4
AMP[[13, 14]] = 0.2; AMP[[16, 17]] = 1.1; AMP[[18, 19]] = 1.1; AMP[[20, 21]] = 0.6; AMP[[10, 11, 22, 23]] = 0.2


def make_motion(T, fps=100, seed=0, tpose_frames=0, amp_scale=1.0):
    """Smooth random joint rotations (sum of sinusoids), turning + walking root. Returns R (T,24,3,3), trans (T,3)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fps
    aa = np.zeros((T, 24, 3))
    for j in range(1, 24):
        for a in range(3):
            for _ in range(3):
                f = rng.uniform(0.1, 0.8)
                aa[:, j, a] += AMP[j] * amp_scale / 3 * rng.uniform(.3, 1) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.3))
    yaw = 0.6 * t + rng.uniform(0, 6.3)
    aa[:, 0, 1] = yaw
    aa[:, 0, 0] = 0.15 * np.sin(0.7 * t)
    trans = np.stack([1.2 * np.sin(0.3 * t), 1.0 + 0.03 * np.sin(3 * t), 1.2 * np.cos(0.3 * t)], -1)
    if tpose_frames:
        aa[:tpose_frames] = 0
        aa[:tpose_frames, 0, 1] = aa[tpose_frames, 0, 1]
        aa[:tpose_frames] += 0.01 * rng.standard_normal((tpose_frames, 24, 3))
        trans[:tpose_frames] = trans[tpose_frames]
    return exp_so3(aa), trans


def simulate(model, vids, betas, R, trans, marker_offset=0.0095, noise=0.0015, dropout=0.05, swap=True, seed=0):
    """Returns observed markers (T,M,3), valid (T,M), and the true rig."""
    from vsmpl.body_model import MarkerRig
    rng = np.random.default_rng(seed)
    v, _ = model.shaped(betas)
    n = model.vertex_normals(v)
    rig = MarkerRig(model, [f"M{i:02d}" for i in range(len(vids))], vids, marker_offset * n[vids], betas)
    T, M = len(R), len(vids)
    obs = np.stack([rig.forward(R[i], trans[i]) for i in range(T)])
    obs += noise * rng.standard_normal(obs.shape)
    valid = np.ones((T, M), bool)
    # bursty occlusions
    for m in range(M):
        n_ev = rng.poisson(dropout * T / 20)
        for _ in range(n_ev):
            s = rng.integers(0, T)
            valid[s:s + rng.integers(5, 35), m] = False
    if swap and T > 400:  # a label-swap event
        a, b, s = 3, 7, T // 2
        obs[s:s + 20, [a, b]] = obs[s:s + 20, [b, a]]
    obs[~valid] = np.nan
    return obs, valid, rig
