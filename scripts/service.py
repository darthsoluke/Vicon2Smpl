#!/usr/bin/env python3
"""Single-instance launcher for the shared Viser console (POSIX, standard library).

Usage: service.py --config deployment.json preflight|start|status|stop|run
Paths in JSON are resolved relative to that JSON file. ``preflight --dry-run``
prints the exact child argv/environment and never starts the marker reader.
"""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit


LOCK_NAME = ".vicon2smpl-service.lock"
PID_NAME = "service.pid.json"
LOG_NAME = "service.log"
BUSY_MODES = {"tracking", "sampling", "calibrating", "countdown"}
IDLE_MODES = {"calibrated", "preview"}
THREAD_ENV = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
THREAD_ENV["CUDA_VISIBLE_DEVICES"] = ""


class ServiceError(RuntimeError):
    pass


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_json(path, data):
    temp = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    try:
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()


def _profile_api(cfg):
    """Load only the standard-library profile parser from this code checkout."""
    module_path = Path(cfg["code_root"]) / "vsmpl" / "model_profiles.py"
    spec = importlib.util.spec_from_file_location("_service_model_profiles", module_path)
    if spec is None or spec.loader is None:
        raise ServiceError("Cannot load model profile parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_profiles(cfg, validate):
    module = _profile_api(cfg)
    default, profiles = module.load_profiles(cfg["model_profiles"], argparse.Namespace(**cfg), validate_paths=validate)
    return module, default, profiles


def load_config(path):
    path = Path(path).expanduser().resolve(strict=True)
    raw = _read_json(path)
    required = {"python", "code_root", "output", "source", "native_reader", "model_path", "bind", "port"}
    allowed = required | {"model_type", "body_regions", "fit_hands", "hand_prior",
                          "hand_components", "calib", "subject", "library_path", "numba_cache", "data_output",
                          "gender", "num_betas", "soma_lod", "soma_identity", "model_profiles", "warp_cache"}
    if not isinstance(raw, dict) or required - raw.keys() or raw.keys() - allowed:
        missing = sorted(required - raw.keys()) if isinstance(raw, dict) else sorted(required)
        extra = sorted(raw.keys() - allowed) if isinstance(raw, dict) else []
        raise ServiceError(f"Invalid config: missing={missing}, unknown={extra}")
    cfg = dict(raw)
    if isinstance(raw.get("model_path"), str) and (raw["model_path"].strip().lower() in {"hf", "auto"}
                                                  or urlsplit(raw["model_path"]).scheme):
        raise ServiceError("Service model_path must use local assets, not hf/auto/URLs")
    for name in ("python", "code_root", "output", "native_reader", "model_path",
                 "body_regions", "hand_prior", "calib", "numba_cache", "data_output", "model_profiles", "warp_cache"):
        if name in cfg:
            if not isinstance(cfg[name], str) or not cfg[name]:
                raise ServiceError(f"{name} must be a nonempty path string")
            value = Path(cfg[name]).expanduser()
            # A venv interpreter is commonly a symlink to the base Python.
            # Invoking its resolved target would silently discard that venv.
            cfg[name] = (os.path.abspath(path.parent / value) if name == "python"
                         else str((path.parent / value).resolve()))
    for name in ("source", "bind"):
        if not isinstance(cfg[name], str) or not cfg[name] or "\x00" in cfg[name]:
            raise ServiceError(f"{name} must be a nonempty string")
    try:
        host, port = cfg["source"].rsplit(":", 1)
        if not host or not 1 <= int(port) <= 65535:
            raise ValueError()
        ipaddress.ip_address(cfg["bind"])
    except ValueError as exc:
        raise ServiceError("source must be host:port; bind must be a local numeric IP address") from exc
    if type(cfg["port"]) is not int or not 1 <= cfg["port"] <= 65535:
        raise ServiceError("port must be an integer from 1 to 65535")
    cfg.setdefault("model_type", "smplx")
    cfg.setdefault("fit_hands", False)
    cfg.setdefault("hand_components", 20)
    cfg.setdefault("gender", "neutral")
    cfg.setdefault("num_betas", 10)
    cfg.setdefault("soma_lod", "mid")
    cfg.setdefault("soma_identity", "soma")
    if "model_profiles" in cfg:
        module, default, profiles = _load_profiles(cfg, False)
        for key in module.ARG_FIELDS:
            if profiles[default][key] is None:
                cfg.pop(key, None)
            else:
                cfg[key] = profiles[default][key]
        cfg["model_profile_default"] = default
        cfg["model_profile_signatures"] = {key: module.profile_signature(profile) for key, profile in profiles.items()}
        cfg["model_profiles_sha256"] = hashlib.sha256(Path(cfg["model_profiles"]).read_bytes()).hexdigest()
    if cfg["model_type"] not in {"smpl", "smplh", "smplx", "soma"}:
        raise ServiceError("Invalid model_type")
    if cfg["gender"] not in {"neutral", "male", "female"}:
        raise ServiceError("Invalid gender")
    if type(cfg["num_betas"]) is not int or cfg["num_betas"] < 1:
        raise ServiceError("num_betas must be a positive integer")
    if cfg["soma_lod"] not in {"mid", "low", "xlo"} or cfg["soma_identity"] not in {"soma", "smpl", "smplx"}:
        raise ServiceError("Invalid SOMA LOD or identity model")
    if cfg["model_type"] == "soma" and cfg["soma_identity"] != "soma":
        raise ServiceError("licensed identity 仅实验 CLI；部署 profile 需 native soma 本地 assets")
    if type(cfg["fit_hands"]) is not bool:
        raise ServiceError("fit_hands must be a JSON boolean")
    if type(cfg["hand_components"]) is not int or not 1 <= cfg["hand_components"] <= 45:
        raise ServiceError("hand_components must be an integer from 1 to 45")
    if cfg["fit_hands"] and (cfg["model_type"] != "smplx" or not cfg.get("hand_prior") or not cfg.get("body_regions")):
        raise ServiceError("fit_hands requires smplx, hand_prior and body_regions")
    if bool(cfg.get("calib")) != bool(cfg.get("subject")):
        raise ServiceError("calib and subject must be provided together")
    if "subject" in cfg and (not isinstance(cfg["subject"], str) or not cfg["subject"]):
        raise ServiceError("subject must be a nonempty string")
    libraries = cfg.get("library_path", [])
    if not isinstance(libraries, list) or any(not isinstance(v, str) or not v for v in libraries):
        raise ServiceError("library_path must be a list of directory paths")
    cfg["library_path"] = [str((path.parent / Path(v).expanduser()).resolve()) for v in libraries]
    cfg["config_path"] = str(path)
    cfg["service_script"] = str((Path(cfg["code_root"]) / "scripts" / "service.py").resolve())
    return cfg


def _require_path(path, kind="file", writable=False, executable=False):
    value = Path(path)
    if not (value.is_dir() if kind == "dir" else value.is_file()):
        raise ServiceError(f"Expected {kind}: {value}")
    mode = os.R_OK | (os.X_OK if kind == "dir" or executable else 0) | (os.W_OK if writable else 0)
    if not os.access(value, mode):
        raise ServiceError(f"Insufficient path permissions: {value}")


def validate_paths(cfg):
    _require_path(cfg["python"], executable=True)
    _require_path(cfg["native_reader"], executable=True)
    _require_path(cfg["code_root"], "dir")
    _require_path(str(Path(cfg["code_root"]) / "run_web.py"))
    _require_path(cfg["service_script"])
    if Path(cfg["service_script"]).resolve() != Path(__file__).resolve():
        raise ServiceError("Run the service.py located inside this config's code_root")
    _require_path(cfg["output"], "dir", writable=True)
    if "data_output" in cfg:
        _require_path(cfg["data_output"], "dir", writable=True)
    model = Path(cfg["model_path"])
    _require_path(model, "dir" if cfg["model_type"] == "soma" or model.is_dir() else "file")
    for key in ("body_regions", "hand_prior", "calib"):
        if key in cfg:
            _require_path(cfg[key])
    for directory in cfg["library_path"]:
        _require_path(directory, "dir")
    if "numba_cache" in cfg:
        _require_path(cfg["numba_cache"], "dir", writable=True)
    if "warp_cache" in cfg:
        _require_path(cfg["warp_cache"], "dir", writable=True)
    if "model_profiles" in cfg:
        _require_path(cfg["model_profiles"])
        module, default, profiles = _load_profiles(cfg, True)
        signatures = {key: module.profile_signature(profile) for key, profile in profiles.items()}
        if (default != cfg["model_profile_default"] or signatures != cfg["model_profile_signatures"]
                or hashlib.sha256(Path(cfg["model_profiles"]).read_bytes()).hexdigest() != cfg["model_profiles_sha256"]):
            raise ServiceError("Model profile configuration changed after it was loaded")
        return dict(default=default, profiles=[dict(id=key, label=profile["label"], model_type=profile["model_type"],
                    soma_lod=profile["soma_lod"], fingerprint=signatures[key]) for key, profile in profiles.items()])
    return None


def child_argv(cfg):
    args = [cfg["python"], "-u", str(Path(cfg["code_root"]) / "run_web.py")]
    for key in ("source", "native_reader", "model_path", "model_type", "gender", "num_betas", "soma_lod", "soma_identity",
                "model_profiles", "bind", "port", "output", "data_output", "body_regions"):
        if key in cfg:
            args.extend(["--" + key.replace("_", "-"), str(cfg[key])])
    if cfg["fit_hands"]:
        args.extend(["--fit-hands", "--hand-prior", cfg["hand_prior"],
                     "--hand-components", str(cfg["hand_components"])])
    if cfg.get("calib"):
        args.extend(["--calib", cfg["calib"], "--subject", cfg["subject"]])
    return args


def child_environment(cfg):
    env = os.environ.copy()
    env.update(THREAD_ENV)
    if cfg["library_path"]:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(cfg["library_path"] + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    if "numba_cache" in cfg:
        env["NUMBA_CACHE_DIR"] = cfg["numba_cache"]
    if "warp_cache" in cfg:
        env["WARP_CACHE_PATH"] = cfg["warp_cache"]
    return env


def check_port(cfg):
    family = socket.AF_INET6 if ":" in cfg["bind"] else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.bind((cfg["bind"], cfg["port"]))
    except OSError as exc:
        raise ServiceError(f"Cannot bind {cfg['bind']}:{cfg['port']}: {exc}") from exc


def fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def _process_info(pid):
    """Read process birth identity and argv without psutil or shell commands."""
    try:
        if type(pid) is not int or pid <= 1:
            return None
        proc = Path("/proc") / str(pid)
        if proc.is_dir():
            stat = (proc / "stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()
            if fields[0] == "Z":
                return None
            argv = [v.decode(errors="replace") for v in (proc / "cmdline").read_bytes().split(b"\0") if v]
            return {"birth": fields[19], "argv": argv, "command": None}
        out = subprocess.run(["ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
                             capture_output=True, text=True, check=False).stdout.strip()
        if not out:
            return None
        return {"birth": out[:24], "argv": None, "command": out[24:].strip()}
    except (OSError, IndexError):
        return None


def _owned(info, birth, required):
    if info is None or info["birth"] != birth:
        return False
    if info["argv"] is not None:
        return all(token in info["argv"] for token in required)
    return all(token in info["command"] for token in required)


def _open_lock(cfg):
    return os.open(Path(cfg["output"]) / LOCK_NAME, os.O_CREAT | os.O_RDWR, 0o600)


def _try_lock(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def service_status(cfg):
    fd = _open_lock(cfg)
    try:
        held = not _try_lock(fd)
    finally:
        os.close(fd)
    try:
        record = _read_json(Path(cfg["output"]) / PID_NAME)
    except (OSError, ValueError):
        record = {}
    identity = record.get("config_sha256") == fingerprint(cfg)
    manager = _process_info(record.get("manager_pid"))
    owner = identity and _owned(manager, record.get("manager_birth"),
                                 [cfg["service_script"], cfg["config_path"], "run"])
    child = _process_info(record.get("child_pid"))
    child_owner = identity and _owned(child, record.get("child_birth"),
                                       [str(Path(cfg["code_root"]) / "run_web.py"), cfg["output"]])
    result = dict(state="running" if held and owner else "starting_or_foreign" if held else "stopped",
                  lock_held=held, owned=bool(owner), child_owned=bool(child_owner),
                  manager_pid=record.get("manager_pid"), child_pid=record.get("child_pid"),
                  url=f"http://{cfg['bind']}:{cfg['port']}/", log=str(Path(cfg["output"]) / LOG_NAME))
    status_path = Path(cfg["output"]) / "status.json"
    try:
        status = _read_json(status_path)
        result.update(mode=status.get("mode"), status_age_seconds=max(0., time.time() - status_path.stat().st_mtime),
                      status_source=status.get("source"))
    except (OSError, ValueError):
        result.update(mode=None, status_age_seconds=None, status_source=None)
    return result


def preflight(cfg):
    profiles = validate_paths(cfg)
    check_port(cfg)
    env = child_environment(cfg)
    keys = list(THREAD_ENV) + ["LD_LIBRARY_PATH", "NUMBA_CACHE_DIR", "WARP_CACHE_PATH"]
    return dict(state="preflight_pass", argv=child_argv(cfg), cwd=cfg["code_root"],
                environment={key: env[key] for key in keys if key in env},
                hardware_connected=False, model_profiles=profiles)


def start(cfg):
    validate_paths(cfg)
    fd = _open_lock(cfg)
    try:
        if not _try_lock(fd):
            result = service_status(cfg)
            result["action"] = "already_running"
            return result
        check_port(cfg)
        args = [cfg["python"], cfg["service_script"], "--config", cfg["config_path"], "run", "--lock-fd", str(fd)]
        with open(Path(cfg["output"]) / LOG_NAME, "ab", buffering=0) as log:
            manager = subprocess.Popen(args, cwd=cfg["code_root"], stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=(fd,))
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            result = service_status(cfg)
            if result["manager_pid"] == manager.pid and result["owned"] and result["child_owned"]:
                result["action"] = "started"
                return result
            if manager.poll() is not None:
                raise ServiceError(f"Service exited during startup ({manager.returncode}); inspect {Path(cfg['output']) / LOG_NAME}")
            time.sleep(.05)
        raise ServiceError("Service is still starting; inspect status and service.log before trying again")
    finally:
        # Do not LOCK_UN: the child inherited this same open file description.
        os.close(fd)


def run(cfg, inherited_fd=None):
    validate_paths(cfg)
    fd = inherited_fd if inherited_fd is not None else _open_lock(cfg)
    child = None
    record = None
    interrupted = False

    def interrupt(_signum, _frame):
        nonlocal interrupted
        if interrupted:
            return  # A second stop must not interrupt the child's final save.
        interrupted = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)

    previous = {}
    try:
        expected = os.stat(Path(cfg["output"]) / LOCK_NAME)
        actual = os.fstat(fd)
        if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino) or not _try_lock(fd):
            raise ServiceError("Another service owns this output directory")
        check_port(cfg)
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, interrupt)
        identity = _process_info(os.getpid())
        record = dict(version=1, config_sha256=fingerprint(cfg), manager_pid=os.getpid(),
                      manager_birth=identity["birth"], started_at=time.time(), phase="starting",
                      code_root=cfg["code_root"], config=cfg["config_path"], child_argv=child_argv(cfg))
        _atomic_json(Path(cfg["output"]) / PID_NAME, record)
        child = subprocess.Popen(child_argv(cfg), cwd=cfg["code_root"], env=child_environment(cfg),
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        child_identity = _process_info(child.pid)
        record.update(child_pid=child.pid, child_birth=child_identity["birth"] if child_identity else None,
                      phase="running")
        _atomic_json(Path(cfg["output"]) / PID_NAME, record)
        if interrupted:
            child.send_signal(signal.SIGINT)
        code = child.wait()
        record.update(phase="stopped", exit_code=code, stopped_at=time.time())
        _atomic_json(Path(cfg["output"]) / PID_NAME, record)
        return code
    finally:
        if child is not None and child.poll() is None:
            # A receipt/write error must not leave an unlocked, unmanaged reader.
            if not interrupted:
                child.send_signal(signal.SIGINT)
            child.wait()
        for number, handler in previous.items():
            signal.signal(number, handler)
        os.close(fd)


def stop(cfg):
    validate_paths(cfg)
    state = service_status(cfg)
    if not state["lock_held"]:
        return dict(state=state["state"], action="already_stopped")
    if not state["owned"] or not state["child_owned"]:
        raise ServiceError("Refusing to signal a PID without matching service/child ownership")
    if state["mode"] in BUSY_MODES:
        raise ServiceError(f"Refusing to stop while {state['mode']}; use the web page to stop and save first")
    if (state["mode"] not in IDLE_MODES or state["status_age_seconds"] is None
            or state["status_age_seconds"] > 5 or state["status_source"] != cfg["source"]):
        raise ServiceError("Refusing to stop without fresh idle status from this source; check the web page first")
    # Re-read ownership immediately before sending a graceful interrupt.
    fresh = service_status(cfg)
    if not fresh["owned"] or fresh["manager_pid"] != state["manager_pid"]:
        raise ServiceError("Service ownership changed; no signal sent")
    if (fresh["mode"] not in IDLE_MODES or fresh["status_age_seconds"] is None
            or fresh["status_age_seconds"] > 5 or fresh["status_source"] != cfg["source"]):
        raise ServiceError("Service is no longer freshly idle; no signal sent")
    os.kill(state["manager_pid"], signal.SIGINT)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = service_status(cfg)
        if not result["lock_held"]:
            return dict(state="stopped", action="stopped")
        time.sleep(.1)
    return dict(state="stopping", action="sigint_sent", message="Waiting for graceful exit; no force kill was sent")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "start", "status", "stop", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, dest="command_config")
        if name in {"preflight", "start", "run"}:
            command.add_argument("--dry-run", action="store_true", help="validate and print argv without starting hardware")
        if name == "run":
            command.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    path = args.command_config or args.config
    if path is None:
        parser.error("--config is required")
    try:
        cfg = load_config(path)
        if args.command == "preflight" or getattr(args, "dry_run", False):
            result = preflight(cfg)
        elif args.command == "run":
            return run(cfg, args.lock_fd)
        else:
            result = {"start": start, "status": service_status, "stop": stop}[args.command](cfg)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ServiceError, OSError, ValueError) as exc:
        print(json.dumps(dict(state="error", error=str(exc)), ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
