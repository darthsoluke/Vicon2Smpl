"""Select explicitly configured calibration markers without changing the source recording."""
import numpy as np


def prepare_calibration_input(rec, regions, num_vertices=None):
    if regions is None:
        return rec, None
    if not isinstance(regions, dict) or not regions:
        raise ValueError("Body marker regions must be a non-empty label dictionary")
    names = list(rec["names"])
    if len(set(names)) != len(names):
        raise ValueError("Duplicate source marker labels")
    missing = sorted(set(regions) - set(names))
    if missing:
        raise ValueError("Missing required body marker labels: " + ", ".join(missing))
    checked = {}
    for name, values in regions.items():
        ids = np.asarray(values)
        if ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(f"Invalid vertex region for {name}")
        if ids.min() < 0 or (num_vertices is not None and ids.max() >= num_vertices):
            raise ValueError(f"Vertex region out of bounds for {name}")
        checked[name] = np.unique(ids).tolist()
    pos, valid = np.asarray(rec["pos"]), np.asarray(rec["valid"])
    if pos.ndim != 3 or pos.shape[1:] != (len(names), 3) or valid.shape != pos.shape[:2]:
        raise ValueError("Marker arrays do not match the label list")
    indices = [i for i, name in enumerate(names) if name in checked]
    selected_names = [names[i] for i in indices]
    selected = dict(rec, names=selected_names, pos=pos[:, indices].copy(), valid=valid[:, indices].copy())
    return selected, {name: checked[name] for name in selected_names}
