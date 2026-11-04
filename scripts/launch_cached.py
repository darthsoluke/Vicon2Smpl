#!/usr/bin/env python3
"""Restore a fixed-prefix Linux RAM runtime, then execute its service manager.

This is a same-server/same-UID cache, not a relocatable environment. The cold
archive remains persistent elsewhere. Preparation has its own short-lived flock;
service.py retains responsibility for the output/service singleton lock.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import stat
import sys
import tarfile
import uuid


SCHEMA = "vicon2smpl.fixed-prefix-cache.v1"
READY = "READY.json"
THREAD_ENV = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
THREAD_ENV["CUDA_VISIBLE_DEVICES"] = ""


class CacheError(RuntimeError):
    pass


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _relative(value, name):
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise CacheError(name + " must be a nonempty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise CacheError(name + " must stay inside the cache")
    return path.as_posix()


def load_config(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"archive", "archive_sha256", "cache_root"}
    allowed = required | {"service_config", "runtime_manifest", "runtime_manifest_sha256", "cold_code_root"}
    if not isinstance(raw, dict) or required - raw.keys() or raw.keys() - allowed:
        raise CacheError("Invalid cached-launch config keys")
    cfg = dict(raw)
    for key in ("archive", "cache_root", "cold_code_root"):
        if key not in cfg:
            continue
        value = cfg[key]
        if not isinstance(value, str) or not Path(value).is_absolute() or "\x00" in value or ".." in Path(value).parts:
            raise CacheError(key + " must be an absolute path without parent traversal")
        cfg[key] = Path(value)
    if cfg["cache_root"] == Path("/"):
        raise CacheError("cache_root must not be the filesystem root")
    if not _sha(cfg["archive_sha256"]):
        raise CacheError("archive_sha256 must be a lowercase SHA256")
    cfg["service_config"] = _relative(cfg.get("service_config", "config/site.json"), "service_config")
    if "runtime_manifest_sha256" in cfg:
        if not _sha(cfg["runtime_manifest_sha256"]):
            raise CacheError("runtime_manifest_sha256 must be a lowercase SHA256")
        cfg["runtime_manifest"] = _relative(cfg.get("runtime_manifest", "runtime-manifest.json"), "runtime_manifest")
    elif "runtime_manifest" in cfg:
        raise CacheError("runtime_manifest requires runtime_manifest_sha256")
    return cfg


def digest_stream(source):
    digest = hashlib.sha256()
    for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _fingerprint(info):
    return dict(device=info.st_dev, inode=info.st_ino, bytes=info.st_size, mtime_ns=info.st_mtime_ns)


def _open_archive(cfg):
    source = cfg["archive"].open("rb")
    if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
        source.close()
        raise CacheError("archive must be a regular file")
    return source


def _verified_archive(cfg, source):
    before = _fingerprint(os.fstat(source.fileno()))
    if digest_stream(source) != cfg["archive_sha256"]:
        raise CacheError("Archive SHA256 mismatch; existing cache was preserved")
    after = _fingerprint(os.fstat(source.fileno()))
    if before != after:
        raise CacheError("Archive changed during verification")
    source.seek(0)
    return after


def _member(member, cache_root):
    name = member.name
    if "\x00" in name or "\\" in name:
        raise CacheError("Unsupported archive member path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise CacheError("Archive member escapes cache: " + name)
    if not path.parts:
        if member.isdir():
            return None
        raise CacheError("Invalid archive root member")
    if path.as_posix() == READY:
        raise CacheError("READY.json is reserved for the verified cache preparer")
    if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
        raise CacheError("Archive contains a special device/FIFO member")
    member = member.replace(name=path.as_posix(), deep=False)
    if member.issym() or member.islnk():
        target = member.linkname
        if not target or "\x00" in target or "\\" in target:
            raise CacheError("Invalid archive link target")
        if PurePosixPath(target).is_absolute():
            absolute = PurePosixPath(posixpath.normpath(target))
            try:
                destination = absolute.relative_to(PurePosixPath(str(cache_root))).as_posix()
            except ValueError as exc:
                raise CacheError("Archive has an external absolute link") from exc
            # Preserve an internal absolute link's meaning while letting the
            # standard data filter enforce extraction boundaries.
            target = (posixpath.relpath(destination, str(path.parent)) if member.issym()
                      else destination)
        resolved = posixpath.normpath(posixpath.join(str(path.parent), target) if member.issym() else target)
        if resolved == ".." or resolved.startswith("../") or PurePosixPath(resolved).is_absolute():
            raise CacheError("Archive has an external relative link")
        member = member.replace(linkname=target, deep=False)
    return member


def _extract(source, cache_root):
    with tarfile.open(fileobj=source, mode="r:gz") as archive:
        members = [_member(member, cache_root) for member in archive.getmembers()]
        archive.extractall(cache_root, members=[m for m in members if m is not None], filter="data")


def _inside(cache_root, relative, executable=False):
    path = cache_root / relative
    try:
        resolved = path.resolve(strict=True)
        root = cache_root.resolve(strict=True)
    except OSError as exc:
        raise CacheError("Missing cached runtime path: " + relative) from exc
    if not resolved.is_relative_to(root) or not path.is_file():
        raise CacheError("Runtime file is missing or resolves outside cache: " + relative)
    if executable and not os.access(path, os.X_OK):
        raise CacheError("Cached Python is not executable: " + relative)
    return path


def _cold_code(cfg):
    destination = cfg["cache_root"] / "code"
    if destination.exists() or "cold_code_root" not in cfg:
        return
    source = cfg["cold_code_root"]
    if not source.is_dir() or source.is_symlink():
        raise CacheError("cold_code_root must be a regular directory")
    ignored = {".git", ".venv", "runtime", "__pycache__"}
    for directory, folders, files in os.walk(source, followlinks=False):
        folders[:] = [name for name in folders if name not in ignored]
        for name in folders + files:
            if name.endswith(".pyc"):
                continue
            if (Path(directory) / name).is_symlink():
                raise CacheError("cold_code_root must not contain source symlinks")
    shutil.copytree(source, destination, symlinks=True,
                    ignore=shutil.ignore_patterns(*ignored, "*.pyc"))


def validate_runtime(cfg):
    root = cfg["cache_root"]
    _inside(root, "python/bin/python3", executable=True)
    _inside(root, "venv/bin/python", executable=True)
    _inside(root, "code/scripts/service.py")
    service_path = _inside(root, cfg["service_config"])
    service = json.loads(service_path.read_text(encoding="utf-8"))
    if not isinstance(service, dict):
        raise CacheError("Cached service config must be a JSON object")
    for key, expected in (("python", root / "venv/bin/python"), ("code_root", root / "code")):
        if key not in service or not isinstance(service[key], str):
            raise CacheError("Cached service config is missing " + key)
        actual = Path(os.path.abspath(service_path.parent / service[key]))
        if actual != Path(os.path.abspath(expected)):
            raise CacheError("Cached service " + key + " does not use the fixed cache prefix")
    if "runtime_manifest_sha256" in cfg:
        manifest = _inside(root, cfg["runtime_manifest"])
        with manifest.open("rb") as source:
            if digest_stream(source) != cfg["runtime_manifest_sha256"]:
                raise CacheError("Cached runtime manifest SHA256 mismatch")


def _atomic_ready(root, value):
    target = root / READY
    if target.is_symlink():
        raise CacheError("READY.json must not be a symlink")
    temporary = root / ("." + READY + "." + str(os.getpid()) + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def prepare(cfg):
    root = cfg["cache_root"]
    if not root.parent.is_dir() or root.parent.stat().st_uid != os.getuid():
        raise CacheError("Cache parent must exist and belong to the launching UID")
    lock_path = root.parent / ("." + root.name + ".prepare.lock")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(lock_path, flags, 0o600)
    try:
        if os.fstat(fd).st_uid != os.getuid() or not stat.S_ISREG(os.fstat(fd).st_mode):
            raise CacheError("Preparation lock must be a regular file owned by the launching UID")
        fcntl.flock(fd, fcntl.LOCK_EX)
        if root.exists() or root.is_symlink():
            ready_path = root / READY
            if root.is_symlink() or not root.is_dir() or ready_path.is_symlink() or not ready_path.is_file():
                raise CacheError("Existing cache is incomplete; preserved without overwrite")
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
            if not isinstance(ready, dict):
                raise CacheError("Existing READY is invalid; cache was preserved")
            expected = dict(schema=SCHEMA, archive_sha256=cfg["archive_sha256"],
                            cache_root=str(root), uid=os.getuid(), service_config=cfg["service_config"],
                            runtime_manifest_sha256=cfg.get("runtime_manifest_sha256"),
                            runtime_manifest=cfg.get("runtime_manifest"))
            if any(ready.get(key) != value for key, value in expected.items()):
                raise CacheError("Existing READY belongs to a different runtime; preserved without overwrite")
            current = _fingerprint(cfg["archive"].stat())
            if ready.get("archive_metadata") != current or ready.get("archive_path") != str(cfg["archive"]):
                with _open_archive(cfg) as source:
                    current = _verified_archive(cfg, source)
                ready.update(archive_metadata=current, archive_path=str(cfg["archive"]))
                validate_runtime(cfg)
                _atomic_ready(root, ready)
            else:
                validate_runtime(cfg)
            return dict(state="reused", cache_root=str(root), archive_sha256=cfg["archive_sha256"])
        with _open_archive(cfg) as source:
            metadata = _verified_archive(cfg, source)
            root.mkdir(mode=0o700)
            # A failure from here leaves the incomplete directory for diagnosis.
            _extract(source, root)
            if _fingerprint(os.fstat(source.fileno())) != metadata:
                raise CacheError("Archive changed during extraction; incomplete cache was preserved")
        _cold_code(cfg)
        validate_runtime(cfg)
        ready = dict(schema=SCHEMA, archive_sha256=cfg["archive_sha256"], archive_path=str(cfg["archive"]),
                     archive_metadata=metadata, cache_root=str(root), uid=os.getuid(),
                     service_config=cfg["service_config"], runtime_manifest=cfg.get("runtime_manifest"),
                     runtime_manifest_sha256=cfg.get("runtime_manifest_sha256"),
                     prepared_utc=datetime.now(timezone.utc).isoformat())
        _atomic_ready(root, ready)
        return dict(state="prepared", cache_root=str(root), archive_sha256=cfg["archive_sha256"])
    finally:
        os.close(fd)  # Release preparation lock before the service manager runs.


def execute(cfg):
    root = cfg["cache_root"]
    python = str(root / "venv/bin/python")
    argv = [python, str(root / "code/scripts/service.py"), "--config", str(root / cfg["service_config"]), "run"]
    env = os.environ.copy()
    env.update(THREAD_ENV)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["VIRTUAL_ENV"] = str(root / "venv")
    os.execve(python, argv, env)
    raise CacheError("Service exec unexpectedly returned")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="prepare/verify cache without connecting to hardware")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        result = prepare(cfg)
        if args.prepare_only:
            print(json.dumps(result))
            return 0
        execute(cfg)
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, CacheError) as exc:
        print(json.dumps(dict(state="failed", error=str(exc))), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
