"""Offline bundle contracts; standard library only, no models, Torch or network."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPORT = script("export_environment")
INSTALL = script("install_offline")


class OfflineEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vsmpl-offline-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.platform = json.loads(subprocess.check_output(
            [sys.executable, "-I", "-c", INSTALL.PLATFORM_CODE], text=True))

    def invoke(self, module, arguments):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return module.main(list(map(str, arguments)))

    def bundle(self, name="bundle"):
        bundle = self.root / name
        (bundle / "wheelhouse").mkdir(parents=True)
        (bundle / "requirements.lock").write_text("demo==1.0\n", encoding="utf-8")
        (bundle / "pip-list.json").write_text('[{"name":"demo","version":"1.0"}]', encoding="utf-8")
        wheel = bundle / "wheelhouse/demo-1.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("demo-1.0.dist-info/METADATA",
                             "Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n")
        files = {p.relative_to(bundle).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in bundle.rglob("*") if p.is_file()}
        manifest = dict(schema=INSTALL.SCHEMA, state="complete", complete=True,
                        python=self.platform, package_count=1, requirements="requirements.lock",
                        wheelhouse="wheelhouse", files=files)
        (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return bundle

    def fake_run(self, command):
        if "-c" in command:
            code = command[command.index("-c") + 1]
            if code == EXPORT.PLATFORM_CODE:
                return 0, json.dumps(self.platform), ""
            if code == EXPORT.ORIGIN_CODE:
                return 0, '[{"name":"demo","direct_reference":true,"editable":true}]', ""
            self.fail("Unexpected interpreter probe")
        if "list" in command:
            return 0, '[{"name":"demo","version":"1.0","editable_project_location":"/private/source"}]', ""
        if "freeze" in command:
            return 0, "demo @ https://alice:secret-password@private.example/pkg.whl?token=secret-query\n", ""
        if "check" in command:
            return 0, "No broken requirements found.\n", ""
        if "--version" in command:
            return 0, "pip 24.0\n", ""
        self.fail("A metadata/direct-reference test must never download or install")

    def test_complete_export_is_never_overwritten_even_with_resume(self):
        bundle = self.bundle()
        before = {p.relative_to(bundle): p.read_bytes() for p in bundle.rglob("*") if p.is_file()}
        with self.assertRaises(SystemExit) as raised:
            self.invoke(EXPORT, ["--output", bundle, "--metadata-only", "--resume"])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(before, {p.relative_to(bundle): p.read_bytes() for p in bundle.rglob("*") if p.is_file()})

    def test_direct_reference_fails_without_persisting_private_source(self):
        bundle = self.root / "direct"
        with patch.object(EXPORT, "run", side_effect=self.fake_run):
            self.assertEqual(self.invoke(EXPORT, ["--output", bundle]), 1)
        manifest = json.loads((bundle / "manifest.json").read_text())
        self.assertFalse(manifest["complete"])
        self.assertTrue(manifest["source_origins_omitted"])
        self.assertIn("cannot substitute", manifest["error"])
        for path in bundle.rglob("*"):
            if path.is_file():
                content = path.read_text(encoding="utf-8")
                for secret in ("private.example", "secret-password", "secret-query", "/private/source"):
                    self.assertNotIn(secret, content, str(path))

    def test_metadata_only_is_incomplete_and_requires_explicit_resume(self):
        bundle = self.root / "metadata"
        with patch.object(EXPORT, "run", side_effect=self.fake_run):
            self.assertEqual(self.invoke(EXPORT, ["--output", bundle, "--metadata-only"]), 0)
            before = (bundle / "manifest.json").read_bytes()
            with self.assertRaises(SystemExit):
                self.invoke(EXPORT, ["--output", bundle, "--metadata-only"])
            self.assertEqual(before, (bundle / "manifest.json").read_bytes())
            self.assertEqual(self.invoke(EXPORT, ["--output", bundle, "--metadata-only", "--resume"]), 0)
        manifest = json.loads((bundle / "manifest.json").read_text())
        self.assertEqual(manifest["state"], "metadata_only")
        self.assertFalse(manifest["complete"])
        self.assertIsInstance(manifest["files"], dict)
        self.assertEqual(self.invoke(INSTALL, ["--bundle", bundle, "--venv", self.root / "unused", "--verify-only"]), 1)
        self.assertFalse((self.root / "unused").exists())

    def test_index_exception_fragments_are_not_logged(self):
        log = EXPORT.download_summary("DOWNLOAD", 1, "",
            "HTTPSConnectionPool(host='private.example',port=443): secret-query\n"
            "ERROR: No matching distribution found for torch==2.5.0+cpu\n")
        self.assertNotIn("private.example", log)
        self.assertNotIn("secret-query", log)
        self.assertIn("torch==2.5.0+cpu", log)

    def test_wheel_local_variant_must_match_exact_distribution(self):
        bundle = self.bundle()
        EXPORT.validate_wheel_identities(bundle / "wheelhouse", [dict(name="demo", version="1.0")])
        with self.assertRaisesRegex(RuntimeError, "variant differs"):
            EXPORT.validate_wheel_identities(bundle / "wheelhouse", [dict(name="demo", version="1.0+cpu")])

    def test_manifest_hash_tamper_fails_before_creating_venv(self):
        bundle = self.bundle()
        INSTALL.verify(bundle)
        (bundle / "requirements.lock").write_text("demo==2.0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            INSTALL.verify(bundle)
        destination = self.root / "never-created"
        self.assertEqual(self.invoke(INSTALL, ["--bundle", bundle, "--venv", destination]), 1)
        self.assertFalse(destination.exists())

    def test_existing_receipt_is_preserved_and_defaults_are_unique(self):
        bundle = self.bundle()
        receipt = self.root / "protected.json"
        original = b'{"state":"installed","evidence":"completed"}'
        receipt.write_bytes(original)
        arguments = ["--bundle", bundle, "--venv", self.root / "unused", "--verify-only"]
        self.assertEqual(self.invoke(INSTALL, arguments + ["--receipt", receipt]), 1)
        self.assertEqual(receipt.read_bytes(), original)
        self.assertEqual(self.invoke(INSTALL, arguments), 0)
        self.assertEqual(self.invoke(INSTALL, arguments), 0)
        defaults = list(self.root.glob("unused.installation_*.json"))
        self.assertEqual(len(defaults), 2)
        self.assertTrue(all(json.loads(p.read_text())["state"] == "verified_only" for p in defaults))
        self.assertFalse((self.root / "unused").exists())


if __name__ == "__main__":
    unittest.main()
