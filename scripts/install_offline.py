#!/usr/bin/env python3
"""Verify a complete environment export, then install it into a new offline venv.

The supplied --python is a separately provided compatible Python interpreter.
This installs exact wheels; it does not copy an existing environment or claim
that its base Python/OS libraries are portable.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import uuid


SCHEMA = "vicon2smpl.offline_environment.v1"
URL = re.compile(r"(?i)\b(?:git\+|hg\+|svn\+|bzr\+)?(?:https?|ftp|ssh|file)://\S+")
PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9_.+!-]*)")
PLATFORM_CODE = """import json,platform,struct,sys,sysconfig
print(json.dumps(dict(version=list(sys.version_info[:3]),implementation=platform.python_implementation(),
system=platform.system(),machine=platform.machine(),platform=sysconfig.get_platform(),
libc=list(platform.libc_ver()),executable=sys.executable,soabi=sysconfig.get_config_var('SOABI'),
cache_tag=sys.implementation.cache_tag,pointer_bits=struct.calcsize('P')*8)))
"""
IMPORT_CODE = """import importlib,json
names=['numpy','torch','smplx','numba','viser']
modules={name:importlib.import_module(name) for name in names}
print(json.dumps(dict(imports={name:str(getattr(module,'__version__','imported')) for name,module in modules.items()},
torch_cuda_build=modules['torch'].version.cuda)))
"""


def redact(value):
    return URL.sub("[source URL omitted]", str(value))


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def run(command):
    env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INDEX": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "CUDA_VISIBLE_DEVICES": ""}
    # Disallow inherited settings from overriding an intentionally offline install.
    for name in ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS", "PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)


def checked(command, logs, label):
    result = run(command)
    logs[label] = dict(returncode=result.returncode, stdout=redact(result.stdout), stderr=redact(result.stderr))
    if result.returncode:
        raise RuntimeError(label + " failed: " + redact(result.stderr or result.stdout)[-1800:])
    return result.stdout


def safe_artifact(bundle, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts or "\\" in relative:
        raise ValueError("Invalid relative artifact path")
    path = bundle.joinpath(*rel.parts)
    for length in range(1, len(rel.parts) + 1):
        if bundle.joinpath(*rel.parts[:length]).is_symlink():
            raise ValueError("Artifact paths must not contain symlinks")
    if not path.is_file():
        raise ValueError("An expected artifact is missing: " + relative)
    return path


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(bundle):
    manifest_path = safe_artifact(bundle, "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("state") != "complete" or manifest.get("complete") is not True:
        raise ValueError("The environment export is not marked complete; metadata/failed exports cannot be installed")
    if manifest.get("requirements") != "requirements.lock" or manifest.get("wheelhouse") != "wheelhouse":
        raise ValueError("Unsupported environment export layout")
    paths = set()
    for relative, digest in manifest["files"].items():
        if relative in paths:
            raise ValueError("Duplicate manifest artifact")
        path = safe_artifact(bundle, relative)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid SHA256 in manifest")
        size = manifest.get("file_bytes", {}).get(relative)
        if (size is not None and path.stat().st_size != size) or digest_file(path) != digest:
            raise ValueError("SHA256/size mismatch: " + relative)
        paths.add(relative)
    if "requirements.lock" not in paths or "pip-list.json" not in paths:
        raise ValueError("The manifest must hash both exact package inventories")
    wheelhouse = bundle / "wheelhouse"
    if wheelhouse.is_symlink() or not wheelhouse.is_dir():
        raise ValueError("Missing regular wheelhouse directory")
    wheels = 0
    for path in wheelhouse.rglob("*"):
        if path.is_symlink():
            raise ValueError("Wheelhouse must not contain symlinks")
        if path.is_file():
            relative = path.relative_to(bundle).as_posix()
            if relative not in paths or path.suffix != ".whl" or path.parent != wheelhouse:
                raise ValueError("Wheelhouse contains an unhashed or unsupported artifact: " + relative)
            wheels += 1
    if not wheels:
        raise ValueError("Wheelhouse contains no wheels")
    expected = {}
    for line in (bundle / "requirements.lock").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = PIN.fullmatch(line.strip())
        if not match:
            raise ValueError("Requirements must contain only exact name==version pins")
        key = canonical(match[1])
        if key in expected:
            raise ValueError("Duplicate locked distribution")
        expected[key] = match[2]
    listed = json.loads((bundle / "pip-list.json").read_text(encoding="utf-8"))
    inventory = {canonical(p["name"]): p["version"] for p in listed}
    if not expected or inventory != expected or len(listed) != len(expected) or len(expected) != manifest["package_count"]:
        raise ValueError("The exact package inventories disagree")
    return manifest, expected, digest_file(manifest_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable, help="separately supplied compatible base Python")
    parser.add_argument("--venv", required=True, type=Path, help="new destination; existing directories are never overwritten")
    parser.add_argument("--verify-only", action="store_true", help="verify hashes/platform without creating an environment")
    parser.add_argument("--receipt", type=Path, help="optional installation receipt path outside the bundle")
    args = parser.parse_args(argv)
    bundle = args.bundle.expanduser().resolve()
    venv = args.venv.expanduser().absolute()
    unique = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    receipt_path = args.receipt.expanduser().absolute() if args.receipt else venv.parent / (venv.name + ".installation_" + unique + ".json")
    receipt = dict(schema="vicon2smpl.offline_installation.v1", state="failed",
                   created_utc=datetime.now(timezone.utc).isoformat(), bundle=str(bundle), venv=str(venv),
                   network_allowed=False, checks={})
    error = None
    try:
        if receipt_path.is_relative_to(bundle):
            raise ValueError("Receipt must be outside the verified bundle")
        if receipt_path.exists() or receipt_path.is_symlink():
            raise ValueError("Receipt already exists; choose a new --receipt path")
        manifest, expected, digest = verify(bundle)
        receipt.update(environment_manifest_sha256=digest, artifacts_verified=len(manifest["files"]),
                       package_count=len(expected))
        actual_python = json.loads(checked([args.python, "-I", "-c", PLATFORM_CODE], receipt["checks"], "base_python"))
        source_python = manifest["python"]
        for key in ("implementation", "system", "machine", "soabi", "cache_tag", "pointer_bits"):
            if actual_python[key] != source_python[key]:
                raise ValueError("Base Python platform does not match export: " + key)
        if actual_python["version"][:2] != source_python["version"][:2]:
            raise ValueError("Base Python major/minor version does not match exported wheels")
        source_libc, actual_libc = source_python["libc"], actual_python["libc"]
        if source_libc[0] and source_libc[0] != actual_libc[0]:
            raise ValueError("Base Python C library does not match export")
        if source_libc[0] and re.fullmatch(r"\d+(?:\.\d+)*", source_libc[1]) and re.fullmatch(r"\d+(?:\.\d+)*", actual_libc[1]):
            source_version = tuple(map(int, source_libc[1].split(".")))
            actual_version = tuple(map(int, actual_libc[1].split(".")))
            if actual_version < source_version:
                raise ValueError("Host C library is older than the verified export host")
        receipt["base_python"] = actual_python
        if args.verify_only:
            receipt["state"] = "verified_only"
        else:
            if venv.exists() or venv.is_symlink():
                raise ValueError("Destination already exists; choose a new --venv path")
            venv.parent.mkdir(parents=True, exist_ok=True)
            checked([args.python, "-I", "-m", "venv", "--copies", str(venv)], receipt["checks"], "create_venv")
            python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            checked([str(python), "-I", "-m", "pip", "install", "--no-index", "--no-cache-dir",
                     "--only-binary=:all:", "--find-links", str(bundle / "wheelhouse"),
                     "-r", str(bundle / "requirements.lock")], receipt["checks"], "offline_install")
            checked([str(python), "-I", "-m", "pip", "check"], receipt["checks"], "pip_check")
            installed = json.loads(checked([str(python), "-I", "-m", "pip", "list", "--format=json"],
                                           receipt["checks"], "installed_inventory"))
            actual = {canonical(p["name"]): p["version"] for p in installed}
            if actual != expected:
                raise ValueError("Installed distribution versions differ from the exact export")
            receipt["core_imports"] = json.loads(checked([str(python), "-I", "-c", IMPORT_CODE],
                                                         receipt["checks"], "core_imports"))
            expected_torch = manifest.get("torch_runtime")
            if "torch" in expected:
                if not isinstance(expected_torch, dict):
                    raise ValueError("Export omitted the Torch CPU/CUDA build identity")
                actual_torch = dict(version=receipt["core_imports"]["imports"]["torch"],
                                    cuda=receipt["core_imports"]["torch_cuda_build"])
                if actual_torch != expected_torch:
                    raise ValueError("Installed Torch CPU/CUDA build differs from the validated export")
            receipt["state"] = "installed"
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        error = redact(exc)
        receipt.update(state="failed", error=error)
    # A failed/partial venv remains for diagnosis; never remove user data.
    if not receipt_path.is_relative_to(bundle) and not receipt_path.is_symlink() and not receipt_path.exists():
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with receipt_path.open("x", encoding="utf-8") as destination:
                destination.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        except FileExistsError:
            receipt.update(state="failed", error="Receipt was created concurrently and was not overwritten")
            error = receipt["error"]
    print(json.dumps(dict(state=receipt["state"], receipt=str(receipt_path), error=error)))
    return 1 if receipt["state"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
