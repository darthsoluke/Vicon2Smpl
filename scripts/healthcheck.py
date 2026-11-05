"""Check an installed deployment without opening another hardware connection."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit


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
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if config.get("warp_cache"):
        os.environ["WARP_CACHE_PATH"] = str(path("warp_cache"))
    profiles = {}
    default_profile = None

    def profile_inventory():
        nonlocal profiles, default_profile
        from vsmpl.model_profiles import load_profiles, profile_signature
        legacy = dict(config)
        legacy.setdefault("model_type", "smplx")
        for key in ("model_path", "body_regions", "hand_prior", "calib"):
            if config.get(key):
                legacy[key] = str(path(key))
        model_source = str(config.get("model_path", ""))
        if model_source.strip().lower() in {"hf", "auto"} or urlsplit(model_source).scheme:
            raise ValueError("Deployment healthcheck requires local model assets")
        default_profile, profiles = load_profiles(path("model_profiles") if config.get("model_profiles") else None,
                                                  argparse.Namespace(**legacy), validate_paths=True)
        return dict(default=default_profile, profiles=[dict(id=key, label=value["label"],
                    model_type=value["model_type"], lod=value["soma_lod"], fingerprint=profile_signature(value))
                    for key, value in profiles.items()])

    record("model_profiles", profile_inventory)
    record("python", lambda: dict(executable=sys.executable, version=sys.version,
                                  platform=platform.platform(), machine=platform.machine()))
    record("pip_check", lambda: command([sys.executable, "-m", "pip", "check"]))
    for module in ("numpy", "torch", "smplx", "numba", "viser", "websockets", "ezc3d"):
        record("import_" + module, lambda module=module: str(getattr(importlib.import_module(module), "__version__", "importable")))
    if any(profile["model_type"] == "soma" for profile in profiles.values()):
        for module in ("soma", "scipy", "warp", "pxr", "trimesh", "rtree", "cholespy", "huggingface_hub"):
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
    for key in ("numba_cache", "warp_cache"):
        if config.get(key):
            record(key, lambda key=key: output_probe(key))

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
        def model_check(profile):
            import numpy as np
            from vsmpl.body_model import BodyModel
            from vsmpl.calibration_input import prepare_calibration_input
            from vsmpl.hand_pca import HandPCA
            from vsmpl.model_profiles import profile_signature
            soma = profile["model_type"] == "soma"
            kwargs = dict(soma_lod=profile["soma_lod"], soma_identity=profile["soma_identity"]) if soma else {}
            model = BodyModel.load(profile["model_path"], profile["model_type"], profile["gender"],
                                   num_betas=profile["num_betas"], **kwargs)
            if profile["fit_hands"] and (model.num_joints != 55 or len(model.v_template) != 10475):
                raise ValueError("Full hands require native SMPL-X55 with 10475 vertices")
            detail = dict(profile_id=profile["id"], fingerprint=profile_signature(profile), model_type=profile["model_type"],
                          joints=model.num_joints, vertices=len(model.v_template), betas=model.num_betas)
            if soma:
                expected_vertices = {"mid": 18056, "low": 4505, "xlo": 612}[profile["soma_lod"]]
                if model.num_joints != 77 or len(model.v_template) != expected_vertices:
                    raise ValueError("SOMA public rig/LOD does not match the pinned 77-pose-joint asset contract")
                detail.update(lod=profile["soma_lod"], identity=profile["soma_identity"], public_skeleton_joints=78,
                              procedural_transforms=False, pose_correctives=False)
            if profile["body_regions"]:
                regions = json.loads(Path(profile["body_regions"]).read_text())
                dummy = dict(names=list(regions), pos=np.empty((0, len(regions), 3)), valid=np.empty((0, len(regions)), bool))
                _, validated = prepare_calibration_input(dummy, regions, len(model.v_template))
                if profile["fit_hands"] and len(validated) != 52:
                    raise ValueError("Full hands require the verified 52-label body region map")
                detail["body_labels"] = len(validated)
                detail["body_regions_sha256"] = profile["body_regions_sha256"]
            if profile["fit_hands"]:
                prior_path = Path(profile["hand_prior"])
                prior = HandPCA.load(prior_path, n_components=profile["hand_components"])
                detail["hand_components_per_side"] = prior.n_components
                detail["hand_prior_sha256"] = hashlib.sha256(prior_path.read_bytes()).hexdigest()
            if profile["calib"]:
                with np.load(profile["calib"], allow_pickle=False) as calibration:
                    if soma:
                        from vsmpl.soma_model import check_calibration_model
                        check_calibration_model(model, calibration)
                    if "source_subject" not in calibration or str(calibration["source_subject"]) != profile["subject"]:
                        raise ValueError("Saved calibration subject mismatch")
                    detail["calibration_subject"] = str(calibration["source_subject"])
                    detail["calibration_markers"] = len(calibration["marker_names"])
                    if profile["fit_hands"]:
                        if str(calibration["marker_scope"]) != "body_hands" or len(calibration["marker_names"]) != 72:
                            raise ValueError("Saved calibration is not a full 72-marker calibration")
                        if str(calibration["hand_prior_sha256"]) != detail["hand_prior_sha256"]:
                            raise ValueError("Saved hand prior hash mismatch")
            return detail
        if not profiles:
            record("licensed_model_and_marker_regions", lambda: (_ for _ in ()).throw(ValueError("No valid model profile configuration")))
        for key, profile in profiles.items():
            name = "licensed_model_and_marker_regions" if key == default_profile else "model_profile_" + key
            record(name, lambda profile=profile: model_check(profile))
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
