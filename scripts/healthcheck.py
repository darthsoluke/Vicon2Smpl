"""Check an installed deployment without opening another hardware connection."""
import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


def check(config_path, *, skip_model=False):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    checks = []

    def record(name, operation):
        try:
            detail = operation()
            checks.append(dict(name=name, status="PASS", detail=detail))
        except Exception as error:
            checks.append(dict(name=name, status="FAIL", detail=str(error)))

    def path(key):
        value = Path(config[key]).expanduser()
        return (config_path.parent / value).resolve() if not value.is_absolute() else value.resolve()

    def command(argv):
        result = subprocess.run(argv, text=True, capture_output=True, timeout=60)
        if result.returncode:
            raise RuntimeError((result.stdout + result.stderr)[-4000:])
        return result.stdout.strip()[-4000:]

    root = path("code_root")
    sys.path.insert(0, str(root))
    record("python", lambda: dict(executable=sys.executable, version=sys.version,
                                  platform=platform.platform(), machine=platform.machine()))
    record("pip_check", lambda: command([sys.executable, "-m", "pip", "check"]))
    for module in ("numpy", "torch", "smplx", "numba", "viser", "websockets", "ezc3d"):
        record("import_" + module, lambda module=module: str(getattr(importlib.import_module(module), "__version__", "importable")))

    def output_probe(key):
        output = path(key)
        if not output.is_dir():
            raise ValueError(key + " must be an existing directory")
        with tempfile.NamedTemporaryFile(prefix=".healthcheck-", dir=output) as file:
            file.write(b"vicon2smpl deployment probe\n")
            file.flush()
            os.fsync(file.fileno())
        return dict(path=str(output), free_bytes=shutil.disk_usage(output).free)

    record("output", lambda: output_probe("output"))
    if config.get("data_output"):
        record("data_output", lambda: output_probe("data_output"))

    def native_check():
        binary = path("native_reader")
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError("native reader is missing or not executable")
        if platform.system() != "Linux":
            raise ValueError("This packaged native reader requires Linux")
        env = os.environ.copy()
        directories = []
        for value in config.get("library_path", []):
            directory = Path(value)
            directories.append(str((config_path.parent / directory).resolve() if not directory.is_absolute() else directory))
        env["LD_LIBRARY_PATH"] = os.pathsep.join(directories + [env.get("LD_LIBRARY_PATH", "")])
        result = subprocess.run(["ldd", str(binary)], text=True, capture_output=True, env=env, timeout=10)
        detail = result.stdout + result.stderr
        if result.returncode or "not found" in detail:
            raise RuntimeError(detail)
        return detail.strip()

    record("native_dynamic_libraries", native_check)

    if skip_model:
        checks.append(dict(name="licensed_model_and_marker_regions", status="SKIP", detail="Explicit --skip-model; incomplete deployment check"))
    else:
        def model_check():
            import hashlib
            import numpy as np
            from vsmpl.body_model import BodyModel
            from vsmpl.calibration_input import prepare_calibration_input
            from vsmpl.hand_pca import HandPCA
            model = BodyModel.load(str(path("model_path")), config.get("model_type", "smplx"))
            if config.get("fit_hands") and (model.num_joints != 55 or len(model.v_template) != 10475):
                raise ValueError("Full hands require native SMPL-X55 with 10475 vertices")
            detail = dict(joints=model.num_joints, vertices=len(model.v_template), betas=model.num_betas)
            if config.get("body_regions"):
                regions = json.loads(path("body_regions").read_text())
                dummy = dict(names=list(regions), pos=np.empty((0, len(regions), 3)), valid=np.empty((0, len(regions)), bool))
                _, validated = prepare_calibration_input(dummy, regions, len(model.v_template))
                if config.get("fit_hands") and len(validated) != 52:
                    raise ValueError("Full hands require the verified 52-label body region map")
                detail["body_labels"] = len(validated)
            if config.get("fit_hands"):
                prior_path = path("hand_prior")
                prior = HandPCA.load(prior_path, n_components=int(config.get("hand_components", 20)))
                detail["hand_components_per_side"] = prior.n_components
                detail["hand_prior_sha256"] = hashlib.sha256(prior_path.read_bytes()).hexdigest()
            if config.get("calib"):
                with np.load(path("calib"), allow_pickle=False) as calibration:
                    if "source_subject" not in calibration or str(calibration["source_subject"]) != config.get("subject"):
                        raise ValueError("Saved calibration subject mismatch")
                    detail["calibration_subject"] = str(calibration["source_subject"])
                    detail["calibration_markers"] = len(calibration["marker_names"])
                    if config.get("fit_hands"):
                        if str(calibration["marker_scope"]) != "body_hands" or len(calibration["marker_names"]) != 72:
                            raise ValueError("Saved calibration is not a full 72-marker calibration")
                        if str(calibration["hand_prior_sha256"]) != detail["hand_prior_sha256"]:
                            raise ValueError("Saved hand prior hash mismatch")
            return detail
        record("licensed_model_and_marker_regions", model_check)
    failed = any(item["status"] == "FAIL" for item in checks)
    skipped = any(item["status"] == "SKIP" for item in checks)
    return dict(status="FAIL" if failed else "PASS_WITH_SKIPS" if skipped else "PASS",
                hardware_connection_opened=False, raw_capture_modified=False, checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", help="Optional new JSON report path")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()
    result = check(args.config, skip_model=args.skip_model)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.report:
        with Path(args.report).open("x") as file:
            file.write(text + "\n")
    return 1 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
