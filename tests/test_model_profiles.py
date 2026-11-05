"""Model selection/profile isolation contracts; standard library only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vsmpl.model_profiles import (ProfileError, apply_profile_args, load_profiles,
                                  profile_request, profile_signature, verify_profile_files)
import vsmpl.model_profiles as model_profiles


class ModelProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vsmpl-profiles-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "model.npz").write_bytes(b"model fixture")
        (self.root / "soma-assets").mkdir()
        (self.root / "regions.json").write_text('{"C7":[1,2]}')
        (self.root / "hand.npz").write_bytes(b"hand prior fixture")
        (self.root / "calib.npz").write_bytes(b"calibration fixture")
        self.args = argparse.Namespace(model_type="smplx", model_path="legacy-missing-model",
                                       output="preserve-output", source="host:8802", extras={"list": [1]})

    def profile(self, **changes):
        profile = dict(id="smplx", label="SMPL-X 身体与双手", model_type="smplx", model_path="model.npz",
                       body_regions="regions.json", fit_hands=True, hand_prior="hand.npz", hand_components=20)
        profile.update(changes)
        return profile

    def load(self, profiles=None, default="smplx", validate_paths=True, **top):
        raw = dict(default=default, profiles=profiles if profiles is not None else [self.profile()])
        raw.update(top)
        filename = self.root / "profiles.json"
        filename.write_text(json.dumps(raw, ensure_ascii=False))
        return load_profiles(filename, self.args, validate_paths=validate_paths)

    def test_two_local_profiles_resolve_paths_and_defaults(self):
        soma = self.profile(id="soma", label="SOMA 身体", model_type="soma", model_path="soma-assets",
                            fit_hands=False, hand_prior=None, soma_lod="low", soma_identity="soma", num_betas=12)
        default, profiles = self.load([self.profile(), soma], default="soma")
        self.assertEqual(default, "soma")
        self.assertEqual(profiles["soma"]["model_path"], str(self.root / "soma-assets"))
        self.assertEqual(profiles["soma"]["gender"], "neutral")
        self.assertEqual(profiles["soma"]["num_betas"], 12)
        self.assertEqual(profiles["smplx"]["body_regions_sha256"], hashlib.sha256((self.root / "regions.json").read_bytes()).hexdigest())
        self.assertEqual(profiles["smplx"]["hand_prior_sha256"], hashlib.sha256(b"hand prior fixture").hexdigest())

    def test_strict_top_level_and_profile_keys(self):
        with self.assertRaises(ProfileError):
            self.load(unexpected=True)
        with self.assertRaises(ProfileError):
            self.load([self.profile(unexpected=True)])
        with self.assertRaises(ProfileError):
            self.load([self.profile(model_type="unknown")])

    def test_unique_ids_labels_and_valid_default(self):
        for second in (self.profile(label="Another"), self.profile(id="another")):
            with self.subTest(second=second), self.assertRaises(ProfileError):
                self.load([self.profile(), second])
        with self.assertRaises(ProfileError):
            self.load(default="missing")

    def test_online_assets_are_forbidden_even_without_existence_checks(self):
        for value in ("hf", "auto", "https://example.invalid/model.npz", "hf://nvidia/soma-x"):
            with self.subTest(value=value), self.assertRaisesRegex(ProfileError, "local filesystem"):
                self.load([self.profile(model_path=value)], validate_paths=False)

    def test_missing_paths_and_asset_directory_are_validated(self):
        for changes in (dict(model_path="missing"), dict(body_regions="missing"), dict(hand_prior="missing"),
                        dict(calib="missing", subject="actor"),
                        dict(model_type="soma", fit_hands=False, model_path="model.npz")):
            with self.subTest(changes=changes), self.assertRaises(ProfileError):
                self.load([self.profile(**changes)])
        _, profiles = self.load([self.profile(model_path="missing")], validate_paths=False)
        self.assertEqual(profiles["smplx"]["model_path"], str(self.root / "missing"))

    def test_hand_and_soma_parameters_fail_closed(self):
        invalid = (dict(model_type="soma", model_path="soma-assets"), dict(hand_prior=None),
                   dict(body_regions=None), dict(hand_components=0), dict(hand_components=46),
                   dict(hand_components=True), dict(num_betas=True), dict(num_betas=0),
                   dict(fit_hands=1), dict(soma_lod="high"), dict(soma_identity="mhr"), dict(gender="any"))
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ProfileError):
                self.load([self.profile(**changes)])

    def test_calibration_and_actor_are_paired(self):
        for changes in (dict(calib="calib.npz"), dict(subject="actor")):
            with self.subTest(changes=changes), self.assertRaises(ProfileError):
                self.load([self.profile(**changes)])
        _, profiles = self.load([self.profile(calib="calib.npz", subject="actor")])
        self.assertEqual(profiles["smplx"]["subject"], "actor")

    def test_deployed_soma_requires_native_identity_but_legacy_cli_keeps_experiments(self):
        for identity in ("smpl", "smplx"):
            raw = self.profile(model_type="soma", model_path="soma-assets", fit_hands=False,
                               hand_prior=None, soma_identity=identity)
            for validate in (True, False):
                with self.subTest(identity=identity, validate=validate), self.assertRaisesRegex(ProfileError, "licensed identity"):
                    self.load([raw], validate_paths=validate)
            self.args.model_type, self.args.model_path = "soma", str(self.root / "model.npz")
            self.args.soma_identity = identity
            _, profiles = load_profiles(None, self.args)
            self.assertEqual(profiles["default"]["soma_identity"], identity)

    def test_apply_profile_never_changes_source_namespace_or_other_profile(self):
        _, profiles = self.load()
        before = copy.deepcopy(vars(self.args))
        applied = apply_profile_args(self.args, profiles["smplx"])
        self.assertEqual(vars(self.args), before)
        self.assertEqual(applied.output, "preserve-output")
        self.assertEqual(applied.source, "host:8802")
        applied.extras["list"].append(2)
        applied.model_path = "changed"
        self.assertEqual(vars(self.args), before)
        self.assertNotEqual(profiles["smplx"]["model_path"], "changed")

    def test_signature_ignores_ui_checkpoint_actor_but_isolates_models(self):
        _, profiles = self.load()
        first = profiles["smplx"]
        second = copy.deepcopy(first)
        second.update(id="another", label="Other label", calib="/another/calib", subject="another actor")
        self.assertEqual(profile_signature(first), profile_signature(second))
        for key, value in (("model_path", "/different/model"), ("num_betas", 11), ("body_regions_sha256", "a" * 64),
                           ("hand_prior_sha256", "b" * 64), ("soma_lod", "low"), ("model_type", "soma")):
            second = copy.deepcopy(first)
            second[key] = value
            self.assertNotEqual(profile_signature(first), profile_signature(second))

    def test_sampling_request_is_deep_copy_with_frozen_source_hashes(self):
        _, profiles = self.load()
        profile = profiles["smplx"]
        before = copy.deepcopy(profile)
        request = profile_request(profile)
        self.assertEqual(request["hand_pca_components"], 20)
        self.assertEqual(request["profile_signature"], profile_signature(profile))
        self.assertEqual(request["body_regions_file"], profile["body_regions"])
        request["model_path"] = "wrong model"
        request["hand_prior_sha256"] = "wrong hash"
        self.assertEqual(profile, before)
        (self.root / "regions.json").write_text('{"C7":[3]}')
        self.assertEqual(profile_request(profile)["body_regions_sha256"], before["body_regions_sha256"])

    def test_legacy_namespace_preserves_mock_paths_and_cli_aliases(self):
        before = copy.deepcopy(vars(self.args))
        default, profiles = load_profiles(None, self.args)
        self.assertEqual(default, "default")
        self.assertEqual(profiles[default]["model_type"], "smplx")
        self.assertEqual(profiles[default]["num_betas"], 10)
        self.assertEqual(vars(self.args), before)
        self.args.model_type, self.args.model_path = "soma", "hf"
        _, profiles = load_profiles(None, self.args)
        self.assertEqual(profiles["default"]["model_path"], "hf")

    def test_legacy_hand_default_is_12_but_json_default_is_20(self):
        _, profiles = load_profiles(None, self.args)
        self.assertEqual(profiles["default"]["hand_components"], 12)
        self.args.hand_components = 18
        _, profiles = load_profiles(None, self.args)
        self.assertEqual(profiles["default"]["hand_components"], 18)
        raw = self.profile()
        raw.pop("hand_components")
        _, profiles = self.load([raw])
        self.assertEqual(profiles["smplx"]["hand_components"], 20)

    def test_windows_legacy_paths_are_preserved_without_posix_rebasing(self):
        with patch.object(model_profiles, "_WINDOWS_HOST", False):
            for location in (r"C:\capture models\SMPLX_NEUTRAL.npz", r"\\server\share\models\SMPLX_NEUTRAL.npz"):
                self.args.model_path = location
                _, profiles = load_profiles(None, self.args)
                self.assertEqual(profiles["default"]["model_path"], location)
                self.assertEqual(profiles["default"]["model_files_sha256"], {})

    def test_posix_deployment_does_not_claim_windows_assets_exist(self):
        location = r"C:\models\SMPLX_NEUTRAL.npz"
        with patch.object(model_profiles, "_WINDOWS_HOST", False):
            with self.assertRaisesRegex(ProfileError, "cannot be validated"):
                self.load([self.profile(model_path=location)])
            _, profiles = self.load([self.profile(model_path=location)], validate_paths=False)
        self.assertEqual(profiles["smplx"]["model_path"], location)

    def test_windows_host_uses_native_path_validation_not_url_rejection(self):
        location = r"C:\models\SMPLX_NEUTRAL.npz"
        native = MagicMock()
        native.is_file.return_value = True
        native.is_dir.return_value = False
        native.__str__.return_value = location
        with patch.object(model_profiles, "_WINDOWS_HOST", True):
            with patch.object(model_profiles, "_resolve_local", return_value=native) as resolve:
                self.assertEqual(model_profiles._path(location, "model_path", self.root, True), location)
                resolve.assert_called_once_with(location, self.root)
                native.is_file.assert_called_once()

    def test_changed_model_contents_change_signature_and_frozen_verification_fails(self):
        _, profiles = self.load()
        original = profiles["smplx"]
        request = profile_request(original)
        verify_profile_files(request)
        (self.root / "model.npz").write_bytes(b"different model at the same path")
        with self.assertRaisesRegex(ProfileError, "Model asset changed"):
            verify_profile_files(request)
        _, changed = self.load()
        self.assertNotEqual(profile_signature(original), profile_signature(changed["smplx"]))
        self.assertEqual(profile_request(original)["model_files_sha256"], request["model_files_sha256"])

    def test_soma_core_content_hashes_are_preserved_and_verified(self):
        asset = self.root / "soma-assets"
        for filename in ("SOMA_neutral.npz", "SOMA_template_rig.usda", "SOMA_procedural_transforms.json"):
            (asset / filename).write_bytes(filename.encode())
        raw = self.profile(model_type="soma", model_path="soma-assets", fit_hands=False, hand_prior=None)
        _, profiles = self.load([raw])
        request = profile_request(profiles["smplx"])
        self.assertEqual(len(request["model_files_sha256"]), 3)
        verify_profile_files(request)
        (asset / "SOMA_template_rig.usda").write_bytes(b"changed rig")
        with self.assertRaisesRegex(ProfileError, "Model asset changed"):
            verify_profile_files(request)


if __name__ == "__main__":
    unittest.main()
