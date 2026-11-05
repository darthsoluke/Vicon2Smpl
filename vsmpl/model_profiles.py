"""Validated local model choices and immutable-by-copy calibration snapshots.

JSON profiles are deployment inputs: they never trigger asset downloads. Legacy
argparse callers retain their existing path/mock behaviour when no JSON is used.
Hashes are captured once while normalising a profile; a later sampling snapshot
does not silently adopt changed files.
"""
import copy
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit


MODEL_TYPES = frozenset(("smpl", "smplh", "smplx", "soma"))
_WINDOWS_HOST = os.name == "nt"
ARG_FIELDS = ("model_type", "model_path", "gender", "num_betas", "body_regions", "fit_hands",
              "hand_prior", "hand_components", "soma_lod", "soma_identity", "calib", "subject")
PROFILE_FIELDS = frozenset(("id", "label", *ARG_FIELDS))
DEFAULTS = dict(gender="neutral", num_betas=10, body_regions=None, fit_hands=False,
                hand_prior=None, hand_components=20, soma_lod="mid", soma_identity="soma",
                calib=None, subject=None)
SIGNATURE_FIELDS = tuple(key for key in ARG_FIELDS if key not in ("calib", "subject")) + (
    "body_regions_sha256", "hand_prior_sha256", "model_files_sha256")


class ProfileError(ValueError):
    pass


def _text(value, field):
    if not isinstance(value, str) or not value.strip() or any(c in value for c in ("\x00", "\n", "\r")):
        raise ProfileError(field + " must be a nonempty single-line string")
    return value.strip()


def _choice(value, field, choices):
    if not isinstance(value, str) or value not in choices:
        raise ProfileError(field + " must be one of " + ", ".join(sorted(choices)))
    return value


def _integer(value, field, maximum=None):
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        raise ProfileError(field + " must be a positive integer" + (" <= " + str(maximum) if maximum else ""))
    return value


def _windows_absolute(value):
    return isinstance(value, str) and PureWindowsPath(value).is_absolute()


def _resolve_local(value, base):
    return (base / Path(value).expanduser()).resolve()


def _path(value, field, base, validate_paths, optional=False, directory=False, legacy=False):
    if value is None and optional:
        return None
    if legacy and isinstance(value, os.PathLike):
        value = os.fspath(value)
    value = _text(value, field)
    # hf/auto are supported by the old SOMA CLI, but never accepted in a JSON
    # deployment profile. Retain those aliases only for the legacy caller.
    alias = value.lower() in {"hf", "auto"}
    if alias and legacy and field == "model_path":
        return value
    windows_absolute = _windows_absolute(value)
    if alias or (urlsplit(value).scheme and not windows_absolute):
        raise ProfileError(field + " must be a local filesystem path; online asset aliases/URLs are forbidden")
    if windows_absolute and not _WINDOWS_HOST:
        if validate_paths:
            raise ProfileError(field + " is a Windows absolute path that cannot be validated on this POSIX host")
        # Preserve foreign legacy/config provenance, without pretending a drive
        # path is a relative filename inside the current POSIX checkout.
        return str(PureWindowsPath(value))
    path = _resolve_local(value, base)
    if validate_paths:
        if field == "model_path":
            valid = path.is_dir() if directory else (path.is_file() or path.is_dir())
        else:
            valid = path.is_file()
        if not valid:
            raise ProfileError(field + " does not name an existing " + ("directory" if directory else "file/path") + ": " + str(path))
    return str(path)


def _file_sha(path):
    if _windows_absolute(path) and not _WINDOWS_HOST:
        return None
    if path is None or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_files(profile):
    if _windows_absolute(profile["model_path"]) and not _WINDOWS_HOST:
        return {}
    root = Path(profile["model_path"])
    if root.is_file():
        return {str(root): _file_sha(root)}
    if profile["model_type"] == "soma":
        candidates = [root / name for name in (
            "SOMA_neutral.npz", "SOMA_template_rig.usda", "SOMA_procedural_transforms.json")]
    else:
        kind, gender = profile["model_type"], profile["gender"].upper()
        candidates = [folder / (kind.upper() + "_" + gender + suffix)
                      for folder in (root, root / kind) for suffix in (".npz", ".pkl")]
    return {str(path): _file_sha(path) for path in candidates if path.is_file()}


def verify_profile_files(request):
    """Reject a changed frozen local input; generated region snapshots are checked separately."""
    for path, expected in request.get("model_files_sha256", {}).items():
        if expected is not None and _file_sha(path) != expected:
            raise ProfileError("Model asset changed since the model profile was frozen: " + path)
    for path_key, hash_key in (("body_regions", "body_regions_sha256"),
                               ("hand_prior", "hand_prior_sha256")):
        expected = request.get(hash_key)
        if expected is not None and _file_sha(request.get(path_key)) != expected:
            raise ProfileError(path_key + " changed since the model profile was frozen")


def _normalise(raw, base, validate_paths, legacy=False):
    if not isinstance(raw, dict) or raw.keys() - PROFILE_FIELDS:
        raise ProfileError("Profile must be an object with only recognised keys")
    missing = {"id", "label", "model_type", "model_path"} - raw.keys()
    if missing:
        raise ProfileError("Profile is missing: " + ", ".join(sorted(missing)))
    profile = copy.deepcopy(DEFAULTS)
    profile.update(copy.deepcopy(raw))
    profile["id"] = _text(profile["id"], "id")
    if profile["id"] in (".", "..") or any(c in profile["id"] for c in ("/", "\\")):
        raise ProfileError("id must not contain path separators or parent traversal")
    profile["label"] = _text(profile["label"], "label")
    model_type = _choice(profile["model_type"], "model_type", MODEL_TYPES)
    _choice(profile["gender"], "gender", ("neutral", "male", "female"))
    _integer(profile["num_betas"], "num_betas")
    _integer(profile["hand_components"], "hand_components", 45)
    _choice(profile["soma_lod"], "soma_lod", ("mid", "low", "xlo"))
    _choice(profile["soma_identity"], "soma_identity", ("soma", "smpl", "smplx"))
    if type(profile["fit_hands"]) is not bool:
        raise ProfileError("fit_hands must be a JSON boolean")
    if profile["fit_hands"] and model_type != "smplx":
        raise ProfileError("fit_hands currently requires SMPL-X; native SOMA hands are not connected")
    profile["model_path"] = _path(profile["model_path"], "model_path", base, validate_paths,
                                         directory=model_type == "soma", legacy=legacy)
    for field in ("body_regions", "hand_prior", "calib"):
        profile[field] = _path(profile[field], field, base, validate_paths, optional=True, legacy=legacy)
    if profile["subject"] is not None:
        profile["subject"] = _text(profile["subject"], "subject")
    if not legacy and bool(profile["calib"]) != bool(profile["subject"]):
        raise ProfileError("calib and subject must be provided together")
    if profile["fit_hands"] and (not profile["body_regions"] or not profile["hand_prior"]):
        raise ProfileError("fit_hands requires body_regions and hand_prior")
    profile["body_regions_sha256"] = _file_sha(profile["body_regions"])
    profile["hand_prior_sha256"] = _file_sha(profile["hand_prior"])
    profile["model_files_sha256"] = _model_files(profile)
    return profile


def load_profiles(path, args, validate_paths=True):
    """Return ``(default_id, profiles_by_id)`` without changing the input args.

    File profiles validate real deployment paths by default. ``path=None``
    creates the backwards-compatible single CLI profile and deliberately does
    not require mock/legacy paths to exist, irrespective of validate_paths.
    """
    if type(validate_paths) is not bool:
        raise ProfileError("validate_paths must be a boolean")
    if path is None:
        if args is None:
            raise ProfileError("Legacy profile requires argparse-style arguments")
        model_type = getattr(args, "model_type", "smplx")
        labels = {"smpl": "SMPL", "smplh": "SMPL-H", "smplx": "SMPL-X", "soma": "SOMA"}
        raw = dict(id="default", label=labels.get(model_type, str(model_type)),
                   model_type=model_type, model_path=getattr(args, "model_path", None))
        for key, default in DEFAULTS.items():
            legacy_default = 12 if key == "hand_components" else default
            value = getattr(args, key, legacy_default)
            raw[key] = copy.deepcopy(legacy_default if value is None and legacy_default is not None else value)
        profile = _normalise(raw, Path.cwd(), False, legacy=True)
        return profile["id"], {profile["id"]: profile}
    filename = Path(path).expanduser().resolve(strict=True)
    raw = json.loads(filename.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"default", "profiles"}:
        raise ProfileError("Model profiles JSON must contain exactly default and profiles")
    default_id = _text(raw["default"], "default")
    if not isinstance(raw["profiles"], list) or not raw["profiles"]:
        raise ProfileError("profiles must be a nonempty JSON list")
    profiles, labels = {}, set()
    for item in raw["profiles"]:
        profile = _normalise(item, filename.parent, validate_paths)
        if profile["id"] in profiles:
            raise ProfileError("Duplicate model profile id: " + profile["id"])
        if profile["label"] in labels:
            raise ProfileError("Duplicate model profile label: " + profile["label"])
        profiles[profile["id"]] = profile
        labels.add(profile["label"])
    if default_id not in profiles:
        raise ProfileError("default must refer to an existing model profile id")
    return default_id, profiles


def apply_profile_args(args, profile):
    """Deep-copy a Namespace and apply only model-related arguments."""
    result = copy.deepcopy(args)
    for key in ARG_FIELDS:
        setattr(result, key, copy.deepcopy(profile[key]))
    return result


def profile_signature(profile):
    """Stable calibration identity; UI names/checkpoint/actor state are excluded."""
    payload = {key: copy.deepcopy(profile.get(key)) for key in SIGNATURE_FIELDS}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def profile_request(profile):
    """Return an independent JSON-safe sampling/worker provenance snapshot.

    Canonical profile names are retained, with aliases matching the existing
    calibration request vocabulary. Callers may replace body_regions_file with
    their copied per-sample snapshot, while retaining the frozen source SHA.
    """
    request = copy.deepcopy(profile)
    request.update(profile_id=profile["id"], profile_label=profile["label"],
                   profile_signature=profile_signature(profile),
                   body_regions_file=profile["body_regions"], hand_prior_source=profile["hand_prior"],
                   hand_pca_components=profile["hand_components"])
    return request
