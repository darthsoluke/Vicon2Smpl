#!/usr/bin/env python3
"""Export this interpreter's exact packages and offline wheels, without source URLs.

Run with the already validated environment's Python, or provide --python.
The manifest is complete only after every locked distribution has a compatible
wheel and the exporting environment passes ``pip check``. No CUDA/CPU variant is
substituted: local versions such as torch==2.x+cpu remain exact requirements.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone
from email.parser import Parser
import zipfile


SCHEMA = "vicon2smpl.offline_environment.v1"
MANIFEST = "manifest.json"
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+!-]*$")
URL = re.compile(r"(?i)\b(?:git\+|hg\+|svn\+|bzr\+)?(?:https?|ftp|ssh|file)://\S+")
PLATFORM_CODE = """import json,platform,struct,sys,sysconfig
print(json.dumps(dict(version=list(sys.version_info[:3]),implementation=platform.python_implementation(),
system=platform.system(),machine=platform.machine(),platform=sysconfig.get_platform(),
libc=list(platform.libc_ver()),executable=sys.executable,soabi=sysconfig.get_config_var('SOABI'),
cache_tag=sys.implementation.cache_tag,pointer_bits=struct.calcsize('P')*8)))
"""
ORIGIN_CODE = """import importlib.metadata,json
result=[]
for distribution in importlib.metadata.distributions():
    raw=distribution.read_text('direct_url.json')
    if raw is not None:
        try: info=json.loads(raw)
        except (TypeError,ValueError): info={}
        result.append(dict(name=distribution.metadata.get('Name',''),direct_reference=True,
            editable=bool(info.get('dir_info',{}).get('editable',False))))
print(json.dumps(result))
"""


def redact(text):
    # Freeze direct-reference origins and pip's configured index URLs must not
    # enter an internal bundle manifest or a user-visible failure message.
    return URL.sub("[source URL omitted]", str(text))


def run(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
    return result.returncode, result.stdout, result.stderr


def checked_json(command):
    code, out, err = run(command)
    if code:
        raise RuntimeError(redact(err or out))
    return json.loads(out)


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def freeze_without_origins(raw, packages):
    """Preserve freeze's pins; replace editable/direct references with exact pins."""
    by_name = {canonical(p["name"]): p for p in packages}
    seen, lines, omitted = set(), [], 0
    for line in raw.splitlines():
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9_.+!-]*)", line.strip())
        if match and canonical(match[1]) in by_name:
            name = canonical(match[1])
            package = by_name[name]
            lines.append(package["name"] + "==" + package["version"])
            seen.add(name)
        else:
            omitted += bool(line.strip())
    for package in packages:
        if canonical(package["name"]) not in seen:
            lines.append(package["name"] + "==" + package["version"])
    header = ("# pip freeze --all; source/editable origins omitted.\n"
              "# Direct references use their installed distribution name/version instead.\n")
    return header + "\n".join(lines) + "\n", omitted


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_wheel_identities(wheelhouse, packages):
    """pip's == pins ignore a candidate local suffix when the pin has none.

    Require identical installed metadata versions as well, preventing an index
    from silently selecting a different CPU/CUDA/local build of the same release.
    """
    expected = {canonical(p["name"]): p["version"] for p in packages}
    found = set()
    for wheel in wheelhouse.glob("*.whl"):
        try:
            with zipfile.ZipFile(wheel) as archive:
                # A wheel may vendor dependencies and their nested dist-info.
                # Only its single top-level distribution owns this wheel.
                members = [name for name in archive.namelist()
                           if name.count("/") == 1 and name.endswith(".dist-info/METADATA")]
                if len(members) != 1:
                    raise ValueError("Expected one distribution METADATA in wheel")
                metadata = Parser().parsestr(archive.read(members[0]).decode("utf-8"))
                name, version = metadata.get("Name", ""), metadata.get("Version", "")
        except (OSError, zipfile.BadZipFile, UnicodeError, ValueError) as exc:
            raise RuntimeError("Invalid bundled wheel: " + wheel.name) from exc
        if not NAME.fullmatch(name) or not VERSION.fullmatch(version):
            raise RuntimeError("Unsupported distribution identity in bundled wheel: " + wheel.name)
        key = canonical(name)
        if key in expected:
            if version != expected[key]:
                raise RuntimeError("Bundled wheel variant differs from installed distribution: " + name +
                                   "==" + version + "; expected ==" + expected[key])
            found.add(key)
    missing = set(expected) - found
    if missing:
        raise RuntimeError("Exact wheel identities are missing for: " + ", ".join(sorted(missing)))


def download_summary(stage, code, out, err):
    # Network exceptions can disclose signed query strings even after their
    # parent URL was redacted. Never persist raw index/download output.
    text = out + err
    missing = re.findall(r"No matching distribution found for ([A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+)(?:\s|$)", text)
    lines = [stage + ": returncode=" + str(code)]
    lines.extend("No compatible exact wheel: " + pin for pin in missing)
    lines.append("Index/download output omitted to keep configured source credentials private.")
    return "\n".join(lines) + "\n"


def artifacts(bundle, owned):
    paths = list(owned)
    wheelhouse = bundle / "wheelhouse"
    if wheelhouse.exists():
        paths.extend(p for p in wheelhouse.rglob("*") if p.is_file() or p.is_symlink())
    result = []
    for path in sorted(set(paths)):
        if not path.exists():
            continue
        if path.is_symlink():
            raise RuntimeError("Bundle artifacts must be regular files, not symlinks")
        if path.is_relative_to(wheelhouse) and (path.parent != wheelhouse or path.suffix != ".whl"):
            raise RuntimeError("wheelhouse must contain only flat .whl files")
        for parent in path.relative_to(bundle).parents:
            if (bundle / parent).is_symlink():
                raise RuntimeError("Bundle artifact directories must not be symlinks")
        result.append(dict(path=path.relative_to(bundle).as_posix(), bytes=path.stat().st_size,
                           sha256=digest_file(path)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--bundle", dest="bundle", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable, help="validated environment's Python")
    parser.add_argument("--metadata-only", action="store_true", help="write an explicitly incomplete metadata snapshot")
    parser.add_argument("--resume", action="store_true", help="explicitly continue a previous failed/metadata-only export")
    parser.add_argument("--offline-only", action="store_true", help="fail rather than contact an index")
    parser.add_argument("--wheel-cache", action="append", default=[], type=Path,
                        help="existing flat wheel directory; repeatable, checked without network first")
    parser.add_argument("--index-url", help="explicit package index, e.g. the matching CPU torch index")
    parser.add_argument("--extra-index-url", action="append", default=[], help="explicit additional index; never persisted")
    parser.add_argument("--timeout", type=int, default=60, help="pip network timeout in seconds")
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args(argv)
    bundle = args.bundle.expanduser().resolve()
    bundle.mkdir(parents=True, exist_ok=True)
    previous_manifest = bundle / MANIFEST
    if previous_manifest.is_symlink():
        parser.error("manifest.json must not be a symlink")
    if previous_manifest.exists():
        try:
            previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            parser.error("Existing manifest is unreadable; choose a new output directory")
        if previous.get("complete") is True:
            parser.error("A complete export already exists; choose a new output directory")
        if not args.resume or previous.get("state") not in ("failed", "metadata_only"):
            parser.error("Existing incomplete export requires explicit --resume")
    if (bundle / "wheelhouse").is_symlink():
        parser.error("wheelhouse must not be a symlink")
    wheelhouse = bundle / "wheelhouse"
    wheelhouse.mkdir(exist_ok=True)
    manifest = dict(schema=SCHEMA, state="failed", complete=False,
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    requirements="requirements.lock", wheelhouse="wheelhouse",
                    source_urls_saved=False, files={}, file_bytes={})
    owned = [bundle / name for name in ("requirements.lock", "pip-list.json", "pip-freeze.txt",
                                        "environment.json", "pip-check.txt", "wheel-download.log")]
    error = None
    try:
        for path in owned + [bundle / MANIFEST]:
            if path.is_symlink():
                raise RuntimeError("Export destinations must not be symlinks")
        python = args.python
        manifest["python"] = checked_json([python, "-I", "-c", PLATFORM_CODE])
        listed = checked_json([python, "-m", "pip", "list", "--format=json", "--disable-pip-version-check"])
        packages = []
        for item in listed:
            name, version = item["name"], item["version"]
            if not NAME.fullmatch(name) or not VERSION.fullmatch(version):
                raise RuntimeError("An installed distribution has an unsupported name/version; exact wheel export cannot proceed")
            packages.append(dict(name=name, version=version))
        packages.sort(key=lambda p: canonical(p["name"]))
        if not packages or len({canonical(p["name"]) for p in packages}) != len(packages):
            raise RuntimeError("Installed distribution inventory is empty or contains duplicate names")
        manifest["package_count"] = len(packages)
        origin_info = checked_json([python, "-I", "-c", ORIGIN_CODE])
        known = {canonical(p["name"]): p["name"] for p in packages}
        source_distributions = [dict(name=known[canonical(p["name"])], direct_reference=True,
                                     editable=bool(p.get("editable")))
                                for p in origin_info if canonical(p["name"]) in known]
        editable = {canonical(p["name"]) for p in listed if p.get("editable_project_location")}
        for name in editable - {canonical(p["name"]) for p in source_distributions}:
            source_distributions.append(dict(name=known[name], direct_reference=True, editable=True))
        manifest["non_registry_distributions"] = source_distributions
        manifest["source_origins_omitted"] = bool(source_distributions)
        torch_runtime = None
        if any(canonical(p["name"]) == "torch" for p in packages):
            torch_runtime = checked_json([python, "-I", "-c",
                "import json,torch; print(json.dumps(dict(version=str(torch.__version__),cuda=torch.version.cuda)))"])
            manifest["torch_runtime"] = torch_runtime
        (bundle / "requirements.lock").write_text("".join(p["name"] + "==" + p["version"] + "\n" for p in packages), encoding="utf-8")
        write_json(bundle / "pip-list.json", packages)
        code, freeze, err = run([python, "-m", "pip", "freeze", "--all"])
        if code:
            raise RuntimeError(redact(err or freeze))
        safe_freeze, omitted = freeze_without_origins(freeze, packages)
        manifest["source_specific_freeze_entries_omitted"] = omitted
        manifest["source_origins_omitted"] = manifest["source_origins_omitted"] or omitted > 0
        (bundle / "pip-freeze.txt").write_text(safe_freeze, encoding="utf-8")
        code, pip_version, err = run([python, "-m", "pip", "--version"])
        if code:
            raise RuntimeError(redact(err or pip_version))
        write_json(bundle / "environment.json", dict(python=manifest["python"],
                    pip=redact(pip_version.strip()), origin_lines_omitted=omitted, torch_runtime=torch_runtime,
                    note="Installed names/versions are exact; source URLs and editable paths are deliberately omitted."))
        code, out, err = run([python, "-m", "pip", "check"])
        (bundle / "pip-check.txt").write_text(redact(out + err), encoding="utf-8")
        manifest["export_environment_pip_check"] = "pass" if code == 0 else "fail"
        if (source_distributions or omitted) and not args.metadata_only:
            raise RuntimeError("Editable/direct-reference distributions were found. Their origins are omitted; registry wheels with the same version cannot substitute for that source. Use a separately validated original-wheel export.")
        if code and not args.metadata_only:
            raise RuntimeError("Exporting environment failed pip check; see pip-check.txt")
        if args.metadata_only:
            manifest.update(state="metadata_only", download=dict(status="not_requested", complete=False))
        else:
            common = [python, "-m", "pip", "download", "--disable-pip-version-check", "--only-binary=:all:",
                      "--no-deps", "--dest", str(wheelhouse), "-r", str(bundle / "requirements.lock")]
            links = [wheelhouse] + [p.expanduser().resolve() for p in args.wheel_cache]
            for path in links:
                if not path.is_dir():
                    raise RuntimeError("A --wheel-cache directory does not exist")
            find_links = [value for path in links for value in ("--find-links", str(path))]
            code, out, err = run(common + ["--no-index"] + find_links)
            log = download_summary("OFFLINE WHEEL CHECK", code, out, err)
            used_network = False
            if code and not args.offline_only:
                command = common + find_links + ["--timeout", str(args.timeout), "--retries", str(args.retries)]
                if args.index_url:
                    command += ["--index-url", args.index_url]
                for url in args.extra_index_url:
                    command += ["--extra-index-url", url]
                code, out, err = run(command)
                log += download_summary("EXPLICIT/CONFIGURED INDEX DOWNLOAD", code, out, err)
                used_network = True
            (bundle / "wheel-download.log").write_text(log, encoding="utf-8")
            manifest["download"] = dict(status="pass" if code == 0 else "fail", complete=code == 0,
                                         network_attempted=used_network)
            if code:
                raise RuntimeError("Exact compatible wheels are incomplete; see wheel-download.log. No package variant was substituted.")
            # A fresh offline resolution proves the resulting wheelhouse itself
            # is sufficient, rather than merely an external cache directory.
            code, out, err = run(common + ["--no-index", "--find-links", str(wheelhouse)])
            with (bundle / "wheel-download.log").open("a", encoding="utf-8") as log_file:
                log_file.write(download_summary("BUNDLED WHEELS ONLY CHECK", code, out, err))
            if code:
                raise RuntimeError("Bundled wheelhouse alone did not resolve all exact requirements")
            validate_wheel_identities(wheelhouse, packages)
            manifest.update(state="complete", complete=True)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        error = redact(exc)
        manifest.update(state="failed", complete=False, error=error)
    try:
        entries = artifacts(bundle, owned)
        manifest["files"] = {entry["path"]: entry["sha256"] for entry in entries}
        manifest["file_bytes"] = {entry["path"]: entry["bytes"] for entry in entries}
    except (OSError, RuntimeError) as exc:
        manifest.update(state="failed", complete=False, error=redact(exc), files={}, file_bytes={})
        error = manifest["error"]
    if (bundle / MANIFEST).is_symlink():
        print(json.dumps(dict(state="failed", complete=False, error="manifest.json must not be a symlink")))
        return 1
    write_json(bundle / MANIFEST, manifest)
    print(json.dumps(dict(state=manifest["state"], complete=manifest["complete"],
                          package_count=manifest.get("package_count"), manifest=str(bundle / MANIFEST), error=error)))
    return 1 if manifest["state"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
