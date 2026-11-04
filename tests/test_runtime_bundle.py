"""Internal bundle provenance, relocation and failure boundaries (no real assets)."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


package_runtime = module("package_runtime")
verify_bundle = module("verify_bundle")


class RuntimeBundleTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        source = root / "source"; source.mkdir()
        subprocess.run(["git", "init", "-q", "--initial-branch=codex/bundle-test", str(source)], check=True)
        (source / "run_web.py").write_text("# bundle test source\n")
        subprocess.run(["git", "add", "."], cwd=source, check=True)
        subprocess.run(["git", "-c", "user.name=Bundle Test", "-c", "user.email=bundle@example.invalid",
                        "commit", "-qm", "test source"], cwd=source, check=True)
        env = root / "environment"; env.mkdir()
        (env / "wheelhouse").mkdir()
        wheel = env / "wheelhouse/example.whl"; wheel.write_bytes(b"test placeholder, not an installable wheel")
        (env / "manifest.json").write_text(json.dumps(dict(complete=True, files={"wheelhouse/example.whl": hashlib.sha256(wheel.read_bytes()).hexdigest()})))
        python = root / "python"; (python / "bin").mkdir(parents=True)
        (python / "bin/python3.real").write_bytes(b"test interpreter placeholder")
        (python / "bin/python3").symlink_to(python / "bin/python3.real")
        sdk = root / "sdk"; (sdk / "lib").mkdir(parents=True)
        (sdk / "lib/example.so").write_bytes(b"test SDK placeholder")
        native = root / "live_markers"; native.write_text("#!/bin/sh\nexit 0\n"); native.chmod(0o755)
        for name in ("model.npz", "prior.npz", "regions.json"):
            (root / name).write_bytes(b"test asset placeholder")
        return SimpleNamespace(source=str(source), environment=str(env), python_home=str(python), sdk_root=str(sdk),
                               sdk_library_dir=[str(sdk / "lib")], native_reader=str(native), model=str(root / "model.npz"),
                               hand_prior=str(root / "prior.npz"), body_regions=str(root / "regions.json"),
                               output=str(root / "bundle"), vicon_source="127.0.0.1:8802", bind="127.0.0.1", port=8792)

    def test_bundle_copies_exact_git_and_relocates_absolute_python_link(self):
        with TemporaryDirectory() as directory:
            args = self.fixture(directory)
            manifest = package_runtime.build(args)
            bundle = Path(args.output)
            self.assertTrue(manifest["complete"])
            self.assertFalse(manifest["clean_install_verified"])
            self.assertFalse((bundle / "code/.git").exists())
            self.assertEqual((bundle / "code/run_web.py").read_text(), (Path(args.source) / "run_web.py").read_text())
            self.assertFalse(Path(os.readlink(bundle / "python/bin/python3")).is_absolute())
            moved = bundle.with_name("moved-bundle")
            bundle.rename(moved)
            self.assertEqual(verify_bundle.verify(moved)["status"], "PASS")
            (moved / "code/run_web.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_bundle.verify(moved)

    def test_missing_wheel_integrity_fails_before_creating_bundle(self):
        with TemporaryDirectory() as directory:
            args = self.fixture(directory)
            (Path(args.environment) / "wheelhouse/example.whl").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                package_runtime.build(args)
            self.assertFalse(Path(args.output).exists())

    def test_incomplete_export_and_existing_output_are_preserved(self):
        with TemporaryDirectory() as directory:
            args = self.fixture(directory)
            (Path(args.environment) / "manifest.json").write_text('{"complete":false}')
            with self.assertRaisesRegex(ValueError, "incomplete"):
                package_runtime.build(args)
            Path(args.output).mkdir()
            marker = Path(args.output) / "keep.txt"; marker.write_text("preserve")
            with self.assertRaisesRegex(ValueError, "new directory"):
                package_runtime.build(args)
            self.assertEqual(marker.read_text(), "preserve")

    def test_external_python_symlink_rejected(self):
        with TemporaryDirectory() as directory:
            args = self.fixture(directory)
            external = Path(directory) / "outside"; external.write_text("outside")
            (Path(args.python_home) / "bin/outside").symlink_to(external)
            with self.assertRaisesRegex(ValueError, "external symlink"):
                package_runtime.build(args)
            self.assertFalse(Path(args.output).exists())

    def test_recursive_output_inside_dependency_is_rejected(self):
        with TemporaryDirectory() as directory:
            args = self.fixture(directory)
            args.output = str(Path(args.python_home) / "nested-bundle")
            with self.assertRaisesRegex(ValueError, "inside a copied"):
                package_runtime.build(args)
            self.assertFalse(Path(args.output).exists())


if __name__ == "__main__":
    unittest.main()
