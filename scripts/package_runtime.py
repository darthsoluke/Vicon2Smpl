"""Build an internal Linux deployment bundle; never upload its licensed assets to Git."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(argv, cwd=None):
    return subprocess.check_output(argv, cwd=cwd, text=True).strip()


def build(args):
    source = Path(args.source).resolve(strict=True)
    environment = Path(args.environment).resolve(strict=True)
    python_home = Path(args.python_home).resolve(strict=True)
    sdk = Path(args.sdk_root).resolve(strict=True)
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Bundle output must be a new directory; existing artifacts are preserved")
    if any(output.is_relative_to(directory) for directory in (environment, python_home, sdk)):
        raise ValueError("Bundle output must not be inside a copied dependency directory")
    if command(["git", "status", "--porcelain", "--untracked-files=normal"], source):
        raise ValueError("Commit the source changes before packaging; the bundle contains exact Git HEAD")
    commit = command(["git", "rev-parse", "HEAD"], source)
    metadata = json.loads((environment / "manifest.json").read_text())
    if metadata.get("complete") is not True:
        raise ValueError("Environment export is incomplete; finish wheel download first")
    for relative, expected in metadata["files"].items():
        path = (environment / relative).resolve(strict=True)
        if not path.is_relative_to(environment) or sha256(path) != expected:
            raise ValueError("Environment artifact hash mismatch: " + relative)
    interpreter = python_home / "bin/python3"
    if not interpreter.is_file():
        raise ValueError("python-home must contain a self-contained Python installation with bin/python3")
    for path in python_home.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(python_home):
            raise ValueError("Python installation has an external symlink: " + str(path))
    library_dirs = []
    for value in args.sdk_library_dir:
        library = Path(value).resolve(strict=True)
        if not library.is_dir() or not library.is_relative_to(sdk):
            raise ValueError("SDK library directories must belong to sdk-root")
        library_dirs.append("../vendor/vicon/" + str(library.relative_to(sdk)))
    inputs = {"model.npz": Path(args.model).resolve(strict=True),
              "pose_hand_prior.npz": Path(args.hand_prior).resolve(strict=True),
              "body_marker_regions.json": Path(args.body_regions).resolve(strict=True)}
    native = Path(args.native_reader).resolve(strict=True)
    if not native.is_file() or not os.access(native, os.X_OK):
        raise ValueError("native-reader must be executable")
    # All validation above is read-only. A failed build below remains available for diagnosis.
    output.mkdir(parents=True)
    try:
        code = output / "code"
        code.mkdir()
        with tempfile.TemporaryDirectory(prefix="vicon-source-") as temp:
            archive = Path(temp) / "source.tar"
            subprocess.run(["git", "archive", "--format=tar", "-o", str(archive), commit], cwd=source, check=True)
            with tarfile.open(archive) as tar:
                tar.extractall(code, filter="data")
        shutil.copytree(environment, output / "environment")
        shutil.copytree(python_home, output / "python", symlinks=True)
        # Make absolute links within the original Python prefix relocatable too.
        for original in python_home.rglob("*"):
            if original.is_symlink() and Path(os.readlink(original)).is_absolute():
                copied = output / "python" / original.relative_to(python_home)
                target = output / "python" / original.resolve().relative_to(python_home)
                copied.unlink()
                copied.symlink_to(os.path.relpath(target, copied.parent))
        shutil.copytree(sdk, output / "vendor/vicon")
        (output / "native").mkdir()
        shutil.copy2(native, output / "native/live_markers")
        (output / "assets").mkdir()
        for name, path in inputs.items():
            shutil.copy2(path, output / "assets" / name)
        for directory in ("runtime", "numba_cache", "config"):
            (output / directory).mkdir(mode=0o700 if directory == "runtime" else 0o755)
        config = dict(python="../venv/bin/python", code_root="../code", output="../runtime",
                      source=args.vicon_source, native_reader="../native/live_markers",
                      model_path="../assets/model.npz", model_type="smplx", bind=args.bind,
                      port=args.port, body_regions="../assets/body_marker_regions.json",
                      fit_hands=True, hand_prior="../assets/pose_hand_prior.npz",
                      hand_components=20, library_path=library_dirs, numba_cache="../numba_cache")
        (output / "config/site.json").write_text(json.dumps(config, indent=2) + "\n")
        restore = '''#!/bin/sh
set -eu
BUNDLE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
"$BUNDLE_ROOT/python/bin/python3" "$BUNDLE_ROOT/code/scripts/verify_bundle.py" --bundle "$BUNDLE_ROOT"
exec "$BUNDLE_ROOT/python/bin/python3" "$BUNDLE_ROOT/code/scripts/install_offline.py" \\
  --bundle "$BUNDLE_ROOT/environment" --python "$BUNDLE_ROOT/python/bin/python3" \\
  --venv "$BUNDLE_ROOT/venv"
'''
        (output / "restore.sh").write_text(restore)
        (output / "restore.sh").chmod(0o755)
        (output / "SOURCE_COMMIT").write_text(commit + "\n")
        manifest = dict(format="vicon2smpl.internal-bundle.v1", complete=True,
                        source_commit=commit, created_utc=datetime.now(timezone.utc).isoformat(),
                        builder_platform=platform.platform(), machine=platform.machine(),
                        licensed_assets_internal_only=True, contains_personal_calibration=False,
                        clean_install_verified=False, hardware_tested=False,
                        system_libraries="Host Linux/glibc and system ELF libraries remain OS dependencies; verify ldd on target",
                        files={str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*"))
                               if path.is_file() and not path.is_symlink()},
                        symlinks={str(path.relative_to(output)): os.readlink(path) for path in sorted(output.rglob("*"))
                                  if path.is_symlink()})
        (output / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    except Exception as error:
        (output / "BUILD_FAILED.txt").write_text(str(error) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "environment", "python-home", "sdk-root", "native-reader", "model", "hand-prior", "body-regions", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--sdk-library-dir", action="append", required=True)
    parser.add_argument("--vicon-source", required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8792)
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps({key: manifest[key] for key in ("complete", "source_commit", "clean_install_verified", "hardware_tested")}, indent=2))


if __name__ == "__main__":
    main()
