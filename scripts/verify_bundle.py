"""Verify the immutable files and internal symlinks of a deployment bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def verify(bundle):
    bundle = Path(bundle).resolve(strict=True)
    manifest = json.loads((bundle / "bundle-manifest.json").read_text())
    if manifest.get("format") != "vicon2smpl.internal-bundle.v1" or manifest.get("complete") is not True:
        raise ValueError("Incomplete or unsupported internal bundle")
    files = manifest.get("files", {})
    required = {"SOURCE_COMMIT", "restore.sh", "config/site.json", "environment/manifest.json",
                "assets/model.npz", "assets/pose_hand_prior.npz", "assets/body_marker_regions.json", "native/live_markers"}
    if not required.issubset(files):
        raise ValueError("Bundle manifest is missing required files")
    for name, expected in files.items():
        relative = Path(name)
        path = bundle / relative
        if relative.is_absolute() or ".." in relative.parts or path.is_symlink() or not path.resolve().is_relative_to(bundle):
            raise ValueError("Unsafe bundle file path: " + name)
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("Bundle SHA256 mismatch: " + name)
    for name, target in manifest.get("symlinks", {}).items():
        relative = Path(name)
        path = bundle / relative
        if relative.is_absolute() or ".." in relative.parts or not path.is_symlink() or os.readlink(path) != target:
            raise ValueError("Bundle symlink mismatch: " + name)
        if not path.resolve(strict=True).is_relative_to(bundle):
            raise ValueError("External bundle symlink: " + name)
    if (bundle / "SOURCE_COMMIT").read_text().strip() != manifest.get("source_commit"):
        raise ValueError("Bundle source commit mismatch")
    return dict(status="PASS", checked_files=len(files), checked_symlinks=len(manifest.get("symlinks", {})),
                source_commit=manifest["source_commit"], clean_install_verified=False,
                note="Integrity only; this does not verify installation, hardware, or fitting accuracy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.bundle), indent=2))


if __name__ == "__main__":
    main()
