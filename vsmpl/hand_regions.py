"""Native SMPL-X surface regions for this project's near-palm / near-tip labels.

These labels deliberately do not use MoSh's standard ``IDX3`` interpretation:
project suffix 3 (and MID0) is near the palm, while suffix 6 is near the tip.
The caller still estimates the actual attachment inside each region. No marker
radius, surface normal direction or finger flexion axis is imposed here.
"""
import numpy as np


_LEFT_FINGER_JOINTS = {
    "IDX": (25, 26, 27),
    "MID": (28, 29, 30),
    "PNK": (31, 32, 33),
    "RNG": (34, 35, 36),
    "THM": (37, 38, 39),
}


def _label_joint_regions():
    regions = {}
    for side, shift in (("L", 0), ("R", 15)):
        for finger, left_joints in _LEFT_FINGER_JOINTS.items():
            joints = tuple(joint + shift for joint in left_joints)
            near_suffix = "0" if finger == "MID" else "3"
            regions[side + finger + near_suffix] = joints[:2]
            regions[side + finger + "6"] = joints[2:]
    return regions


def finger_vertex_regions(model, betas, names):
    """Return allowed vertex IDs for an explicit subset of the 20 project labels.

    Only a native 55-joint SMPL-X model is accepted. A vertex must be owned by
    one of the designated finger joints (largest skinning weight), so neither
    another finger nor a wrist-owned vertex can enter the result. Near-palm
    and near-tip regions use disjoint joint ownership and cannot select the
    same vertex. ``names`` must contain only finger labels, not body labels.

    Geometry is evaluated through the caller's native ``model.shaped`` path to
    validate the supplied shape, without changing the model or the parameters.
    """
    if getattr(model, "num_joints", None) != 55:
        raise ValueError("Finger regions require a native 55-joint SMPL-X model")
    if isinstance(names, (str, bytes)):
        raise ValueError("Supply an explicit list of project finger labels")
    try:
        labels = list(names)
    except TypeError as error:
        raise ValueError("Supply an explicit list of project finger labels") from error
    allowed = _label_joint_regions()
    if not labels or any(not isinstance(name, str) or name not in allowed for name in labels):
        raise ValueError("Unknown or missing project finger labels; use THM/IDX/RNG/PNK 3/6 or MID 0/6")
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate project finger labels")

    shape = np.asarray(betas, dtype=np.float64)
    if shape.ndim != 1 or not shape.size or not np.isfinite(shape).all():
        raise ValueError("Finger regions require a finite one-dimensional shape vector")
    vertices, joints = model.shaped(shape.copy())
    vertices, joints = np.asarray(vertices), np.asarray(joints)
    weights = np.asarray(model.weights)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or joints.shape != (55, 3)
            or weights.shape != (len(vertices), 55)):
        raise ValueError("SMPL-X geometry and skinning weights have incompatible dimensions")
    if (not np.isfinite(vertices).all() or not np.isfinite(joints).all()
            or not np.isfinite(weights).all() or (weights < 0).any()
            or (weights.sum(axis=1) <= 0).any()):
        raise ValueError("SMPL-X geometry and skinning weights must be finite and valid")

    owner = weights.argmax(axis=1)
    result = {}
    for label in labels:
        ids = np.flatnonzero(np.isin(owner, allowed[label]))
        if not len(ids):
            raise ValueError(f"No native finger surface vertices for {label}")
        result[label] = ids.tolist()
    return result
