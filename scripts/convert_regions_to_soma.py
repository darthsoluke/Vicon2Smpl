"""Convert a reviewed SMPL-X (or SMPL) marker-region JSON {label: [vertex ids]} to SOMA vertex ids.

    python scripts/convert_regions_to_soma.py --regions body_regions_smplx.json --out body_regions_soma.json \
        [--source smplx] [--soma-lod mid] [--soma-assets hf]

SOMA ships, for each supported backend, its own topology fitted onto that backend's template
(`<BACKEND>/SOMA_wrap.obj`, SOMA vertex order) next to the template itself (`<BACKEND>/base_body.obj`, backend
vertex order). Both describe the same surface, so every SOMA vertex inherits the label of its nearest source
vertex. A source region that captures no SOMA vertex (very small regions) falls back to the SOMA vertices nearest
to its own vertices. The conversion only moves the caller's anatomical regions onto the SOMA mesh; the regions
must still be the ones the operator reviewed for the actual marker layout.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

EXPECTED_SOURCE_VERTICES = {"smplx": 10475, "smpl": 6890}


def load_meshes(source="smplx", lod="mid", assets=None):
    import trimesh

    if assets in (None, "", "hf", "auto"):
        from soma.assets import get_assets_dir
        assets = get_assets_dir()
    folder = Path(assets) / source.upper()
    base = trimesh.load(folder / "base_body.obj", process=False, maintain_order=True)
    wrap = trimesh.load(folder / "SOMA_wrap.obj", process=False, maintain_order=True)
    source_v, soma_v = np.asarray(base.vertices, np.float64), np.asarray(wrap.vertices, np.float64)
    if len(source_v) != EXPECTED_SOURCE_VERTICES[source]:
        raise ValueError(f"{folder / 'base_body.obj'} has {len(source_v)} vertices, expected {EXPECTED_SOURCE_VERTICES[source]}")
    if lod != "mid":
        # SOMA's low/xlo meshes are vertex subsets / separate meshes; resolve through SOMA's own LOD tables.
        from soma import SOMALayer
        layer = SOMALayer(data_root=None if assets is None else assets, device="cpu", mode="torch", lod=lod,
                          identity_model_type="soma", enable_procedural_transforms=False, correctives_model_path=None)
        subset = getattr(layer.identity_model, "_nv_lod_mid_to_low", None)
        if subset is None:
            raise ValueError(f"SOMA lod={lod} has no mid->{lod} vertex subset; convert at lod=mid")
        soma_v = soma_v[np.asarray(subset)]
    return source_v, soma_v


def convert(regions, source_v, soma_v, max_distance=0.015):
    """`max_distance` (m): SOMA vertices farther than this from every source vertex (interior geometry with no
    counterpart on the source surface) are never labelled."""
    distance, nearest_source = cKDTree(source_v).query(soma_v)  # SOMA vertex -> source vertex
    nearest_source = np.where(distance <= max_distance, nearest_source, -1)
    soma_tree = cKDTree(soma_v)
    out, report = {}, {}
    for label, ids in regions.items():
        ids = np.unique(np.asarray(ids, np.int64))
        if not len(ids) or ids.min() < 0 or ids.max() >= len(source_v):
            raise ValueError(f"Invalid source vertex ids for {label}")
        members = np.flatnonzero(np.isin(nearest_source, ids))
        fallback = not len(members)
        if fallback:
            members = np.unique(soma_tree.query(source_v[ids])[1])
        out[label] = members.tolist()
        report[label] = dict(source=len(ids), soma=len(members), fallback=fallback)
    return out, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", required=True, help="source JSON {label: [vertex ids]}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="smplx", choices=sorted(EXPECTED_SOURCE_VERTICES))
    ap.add_argument("--soma-lod", default="mid", choices=["mid", "low"])
    ap.add_argument("--soma-assets", default="hf", help="SOMA asset folder, or 'hf' for the HuggingFace cache")
    ap.add_argument("--max-distance", type=float, default=0.015,
                    help="ignore SOMA vertices farther than this (m) from the source surface")
    args = ap.parse_args()
    raw = Path(args.regions).read_bytes()
    regions = json.loads(raw)
    source_v, soma_v = load_meshes(args.source, args.soma_lod, args.soma_assets)
    out, report = convert(regions, source_v, soma_v, args.max_distance)
    Path(args.out).write_text(json.dumps(out))
    meta = dict(source_file=str(Path(args.regions).resolve()), source_sha256=hashlib.sha256(raw).hexdigest(),
                source_model=args.source, soma_lod=args.soma_lod, soma_vertices=len(soma_v), labels=report)
    Path(args.out).with_suffix(".conversion.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    small = [k for k, r in report.items() if r["fallback"]]
    print(f"[regions] {len(out)} labels -> {args.out} (SOMA lod={args.soma_lod}, {len(soma_v)} vertices)")
    print(f"[regions] SOMA vertices per label: median {int(np.median([r['soma'] for r in report.values()]))}, "
          f"min {min(r['soma'] for r in report.values())}")
    if small:
        print(f"[regions] nearest-vertex fallback (region smaller than SOMA spacing): {small}")


if __name__ == "__main__":
    main()
