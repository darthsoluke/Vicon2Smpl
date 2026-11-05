"""Synthetic markers and motion on the real SOMA body (end-to-end tests of the SOMA backend)."""
import numpy as np

from vsmpl.body_model import MarkerRig
from vsmpl.rotations import exp_so3

# markers per segment (owner joint by largest skinning weight), roughly a production full-body set (~56)
N_PER_SEG = {"Hips": 4, "Spine1": 2, "Spine2": 2, "Chest": 3, "Neck1": 1, "Head": 4,
             "LeftShoulder": 1, "RightShoulder": 1, "LeftArm": 2, "RightArm": 2, "LeftForeArm": 2, "RightForeArm": 2,
             "LeftHand": 3, "RightHand": 3, "LeftLeg": 2, "RightLeg": 2, "LeftShin": 2, "RightShin": 2,
             "LeftFoot": 3, "RightFoot": 3, "LeftToeBase": 1, "RightToeBase": 1}

# motion amplitude (rad) by joint name, in SOMA's public T-relative convention (world-aligned at the T-pose)
AMP = {"Spine1": .2, "Spine2": .2, "Chest": .2, "Neck1": .2, "Neck2": .2, "Head": .35, "LeftShoulder": .15,
       "RightShoulder": .15, "LeftArm": 1.0, "RightArm": 1.0, "LeftForeArm": 1.0, "RightForeArm": 1.0,
       "LeftHand": .5, "RightHand": .5, "LeftLeg": .8, "RightLeg": .8, "LeftShin": .9, "RightShin": .9,
       "LeftFoot": .35, "RightFoot": .35, "LeftToeBase": .15, "RightToeBase": .15}

BODY_JOINT_NAMES = [n for n in AMP] + ["Hips"]


def pick_markers(model, seed=1):
    rng = np.random.default_rng(seed)
    owner = np.argmax(model.weights, 1)
    vids = []
    for name, k in N_PER_SEG.items():
        cand = np.nonzero((owner == model.joint_names.index(name)) & (model.weights.max(1) > 0.6))[0]
        vids += list(rng.choice(cand, size=min(k, len(cand)), replace=False))
    return np.array(sorted(vids))


def make_public_motion(model, T, fps=100, seed=0, static_frames=0, static_pose="t", arm_angle=45.0):
    """Public T-relative axis-angle (T,77,3) and Hips translation (T,3)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fps
    aa = np.zeros((T, model.num_joints, 3))
    for name, amp in AMP.items():
        j = model.joint_names.index(name)
        for a in range(3):
            for _ in range(3):
                f = rng.uniform(0.1, 0.8)
                aa[:, j, a] += amp / 3 * rng.uniform(.3, 1) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.3))
    aa[:, 0, 1] = 0.6 * t + rng.uniform(0, 6.3)
    aa[:, 0, 0] = 0.15 * np.sin(0.7 * t)
    transl = np.stack([1.2 * np.sin(0.3 * t), 0.95 + 0.03 * np.sin(3 * t), 1.2 * np.cos(0.3 * t)], -1)
    if static_frames:
        ref = np.zeros((model.num_joints, 3))
        if static_pose == "a":
            ref[model.joint_names.index("LeftArm"), 2] = -np.radians(arm_angle)
            ref[model.joint_names.index("RightArm"), 2] = np.radians(arm_angle)
        yaw = aa[static_frames, 0, 1]
        aa[:static_frames] = ref + 0.01 * rng.standard_normal((static_frames, model.num_joints, 3))
        aa[:static_frames, 0] = [0, yaw, 0]
        transl[:static_frames] = transl[static_frames]
    return aa, transl


def simulate(model, vids, betas, poses_public, transl, marker_offset=0.0095, noise=0.0015, dropout=0.05, seed=0):
    """Markers from SOMA public parameters. Returns obs (T,M,3), valid (T,M), the true rig and solver poses."""
    rng = np.random.default_rng(seed)
    model.prepare_for(betas)
    v, _ = model.shaped(betas)
    n = model.vertex_normals(v)
    rig = MarkerRig(model, [f"M{i:02d}" for i in range(len(vids))], vids, marker_offset * n[vids], betas)
    T = len(poses_public)
    R = np.empty((T, model.num_joints, 3, 3))
    trans = np.empty((T, 3))
    for i in range(T):
        R[i], trans[i] = model.from_public_pose(poses_public[i], transl[i])
    obs = np.stack([rig.forward(R[i], trans[i]) for i in range(T)])
    obs += noise * rng.standard_normal(obs.shape)
    valid = np.ones((T, len(vids)), bool)
    for m in range(len(vids)):
        for _ in range(rng.poisson(dropout * T / 20)):
            s = rng.integers(0, T)
            valid[s:s + rng.integers(5, 35), m] = False
    if T > 400:  # a label swap
        a, b, s = 3, 7, T // 2
        obs[s:s + 20, [a, b]] = obs[s:s + 20, [b, a]]
    obs[~valid] = np.nan
    return obs, valid, rig, R, trans
