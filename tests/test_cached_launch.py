"""Fixed-prefix cache startup tests: standard library, no hardware or packages."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/launch_cached.py"
SPEC = importlib.util.spec_from_file_location("launch_cached", SCRIPT)
LAUNCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCH)


class CachedLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vsmpl-cache-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "runtime"
        self.archive = self.root / "runtime.tar.gz"

    def fixture(self, extra=(), internal_python_link=False, include_code=True):
        files = [("python/bin/python3", b"#!/bin/sh\nexit 0\n", 0o755),
                 ("config/site.json", json.dumps(dict(python="../venv/bin/python", code_root="../code")).encode(), 0o600)]
        if not internal_python_link:
            files.append(("venv/bin/python", b"#!/bin/sh\nexit 0\n", 0o755))
        if include_code:
            files.append(("code/scripts/service.py", b"# toy service; must not execute in prepare-only\n", 0o644))
        with tarfile.open(self.archive, "w:gz") as archive:
            for name, data, mode in files:
                member = tarfile.TarInfo(name)
                member.size, member.mode = len(data), mode
                archive.addfile(member, io.BytesIO(data))
            if internal_python_link:
                member = tarfile.TarInfo("venv/bin/python")
                member.type, member.linkname = tarfile.SYMTYPE, str(self.cache / "python/bin/python3")
                archive.addfile(member)
            for member, data in extra:
                archive.addfile(member, io.BytesIO(data) if data is not None else None)
        config = dict(archive=str(self.archive), archive_sha256=hashlib.sha256(self.archive.read_bytes()).hexdigest(),
                      cache_root=str(self.cache), service_config="config/site.json")
        path = self.root / "launch.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path, LAUNCH.load_config(path)

    def test_prepare_then_reuse_without_rehashing_unchanged_archive(self):
        _, cfg = self.fixture()
        self.assertEqual(LAUNCH.prepare(cfg)["state"], "prepared")
        ready = json.loads((self.cache / "READY.json").read_text())
        self.assertEqual(ready["archive_sha256"], cfg["archive_sha256"])
        with patch.object(LAUNCH, "digest_stream", side_effect=AssertionError("hot startup must not hash archive")):
            self.assertEqual(LAUNCH.prepare(cfg)["state"], "reused")

    def test_archive_stat_change_is_rehashed_and_corruption_preserves_ready_cache(self):
        _, cfg = self.fixture()
        LAUNCH.prepare(cfg)
        info = self.archive.stat()
        os.utime(self.archive, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
        with patch.object(LAUNCH, "digest_stream", wraps=LAUNCH.digest_stream) as digest:
            self.assertEqual(LAUNCH.prepare(cfg)["state"], "reused")
            self.assertEqual(digest.call_count, 1)
        before = (self.cache / "READY.json").read_bytes()
        with self.archive.open("ab") as archive:
            archive.write(b"changed")
        with self.assertRaisesRegex(LAUNCH.CacheError, "SHA256 mismatch"):
            LAUNCH.prepare(cfg)
        self.assertEqual((self.cache / "READY.json").read_bytes(), before)
        self.assertTrue((self.cache / "venv/bin/python").exists())

    def test_incomplete_existing_cache_is_preserved_without_overwrite(self):
        _, cfg = self.fixture()
        self.cache.mkdir()
        marker = self.cache / "preserve.txt"
        marker.write_text("partial evidence")
        with self.assertRaisesRegex(LAUNCH.CacheError, "incomplete"):
            LAUNCH.prepare(cfg)
        self.assertEqual(marker.read_text(), "partial evidence")
        self.assertFalse((self.cache / "READY.json").exists())

    def test_different_ready_archive_does_not_replace_current_environment(self):
        _, cfg = self.fixture()
        LAUNCH.prepare(cfg)
        before = (self.cache / "READY.json").read_bytes()
        cfg["archive_sha256"] = "a" * 64
        with self.assertRaisesRegex(LAUNCH.CacheError, "different runtime"):
            LAUNCH.prepare(cfg)
        self.assertEqual((self.cache / "READY.json").read_bytes(), before)

    def test_bad_archive_hash_fails_before_making_cache(self):
        _, cfg = self.fixture()
        cfg["archive_sha256"] = "b" * 64
        with self.assertRaisesRegex(LAUNCH.CacheError, "SHA256 mismatch"):
            LAUNCH.prepare(cfg)
        self.assertFalse(self.cache.exists())

    def test_tar_parent_traversal_is_rejected_and_failure_is_preserved(self):
        member = tarfile.TarInfo("../outside.txt")
        member.size = 3
        _, cfg = self.fixture(extra=[(member, b"bad")])
        with self.assertRaisesRegex(LAUNCH.CacheError, "escapes cache"):
            LAUNCH.prepare(cfg)
        self.assertFalse((self.root / "outside.txt").exists())
        self.assertTrue(self.cache.is_dir())
        self.assertFalse((self.cache / "READY.json").exists())

    def test_external_link_is_rejected(self):
        member = tarfile.TarInfo("code/unsafe")
        member.type, member.linkname = tarfile.SYMTYPE, "/etc/passwd"
        _, cfg = self.fixture(extra=[(member, None)])
        with self.assertRaisesRegex(LAUNCH.CacheError, "external absolute link"):
            LAUNCH.prepare(cfg)
        self.assertFalse((self.cache / "READY.json").exists())

    def test_internal_absolute_venv_link_is_normalized_and_preserved(self):
        _, cfg = self.fixture(internal_python_link=True)
        LAUNCH.prepare(cfg)
        python = self.cache / "venv/bin/python"
        self.assertTrue(python.is_symlink())
        self.assertFalse(Path(os.readlink(python)).is_absolute())
        self.assertEqual(python.resolve(), (self.cache / "python/bin/python3").resolve())

    def old_distro_filter(self):
        standard = tarfile.data_filter
        calls = []
        def old(member, destination):
            calls.append((member.name, member.type, member.linkname))
            # Model the Ubuntu backport bug: both link kinds are checked
            # relative to the archive root, ignoring a symlink's parent.
            if member.issym() or member.islnk():
                root = os.path.realpath(destination)
                target = os.path.realpath(os.path.join(root, member.linkname))
                if os.path.commonpath([root, target]) != root:
                    raise tarfile.LinkOutsideDestinationError(member, target)
            return standard(member, destination)
        return old, calls

    def terminfo_members(self, target="python/share/terminfo/a/adm1178"):
        data = b"valid terminal data"
        regular = tarfile.TarInfo(target)
        regular.size = len(data)
        alias = tarfile.TarInfo("python/share/terminfo/1/1178")
        alias.type, alias.linkname = tarfile.SYMTYPE, "../a/adm1178"
        return [(regular, data), (alias, None)]

    def test_terminfo_relative_symlink_passes_old_distro_filter(self):
        members = self.terminfo_members()
        _, cfg = self.fixture(extra=members)
        old, _ = self.old_distro_filter()
        with self.assertRaises(tarfile.LinkOutsideDestinationError):
            old(members[-1][0], self.cache)
        with patch.object(LAUNCH.tarfile, "data_filter", side_effect=old):
            LAUNCH.prepare(cfg)
        alias = self.cache / "python/share/terminfo/1/1178"
        self.assertEqual(os.readlink(alias), "../a/adm1178")
        self.assertEqual(alias.read_bytes(), b"valid terminal data")

    def test_internal_ancestor_chain_keeps_symlink_semantics(self):
        members = self.terminfo_members("python/share/terminfo/data/a/adm1178")
        directory = tarfile.TarInfo("python/share/terminfo/data/1")
        directory.type = tarfile.DIRTYPE
        first = tarfile.TarInfo("python/share/terminfo/1")
        first.type, first.linkname = tarfile.SYMTYPE, "aliases/one"
        second = tarfile.TarInfo("python/share/terminfo/aliases/one")
        second.type, second.linkname = tarfile.SYMTYPE, "../data/1"
        _, cfg = self.fixture(extra=[members[0], (directory, None), (second, None), (first, None), members[1]])
        old, _ = self.old_distro_filter()
        with patch.object(LAUNCH.tarfile, "data_filter", side_effect=old):
            LAUNCH.prepare(cfg)
        alias = self.cache / "python/share/terminfo/1/1178"
        self.assertEqual(alias.read_bytes(), b"valid terminal data")
        self.assertEqual(alias.resolve(), (self.cache / "python/share/terminfo/data/a/adm1178").resolve())

    def test_external_ancestor_symlink_is_still_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        ancestor = self.cache / "python/share/terminfo/1"
        ancestor.parent.mkdir(parents=True)
        ancestor.symlink_to(outside, target_is_directory=True)
        alias = self.terminfo_members()[-1][0]
        old, _ = self.old_distro_filter()
        with patch.object(LAUNCH.tarfile, "data_filter", side_effect=old):
            with self.assertRaises(tarfile.LinkOutsideDestinationError):
                LAUNCH.runtime_data_filter(alias, self.cache)
        self.assertEqual(list(outside.iterdir()), [])

    def test_hardlink_is_passed_unchanged_to_standard_filter(self):
        members = self.terminfo_members()
        hard = tarfile.TarInfo("python/share/terminfo/hard-copy")
        hard.type, hard.linkname = tarfile.LNKTYPE, members[0][0].name
        _, cfg = self.fixture(extra=[members[0], (hard, None)])
        old, calls = self.old_distro_filter()
        with patch.object(LAUNCH.tarfile, "data_filter", side_effect=old):
            LAUNCH.prepare(cfg)
        self.assertIn((hard.name, tarfile.LNKTYPE, hard.linkname), calls)
        original = self.cache / members[0][0].name
        copied = self.cache / hard.name
        self.assertEqual(original.stat().st_ino, copied.stat().st_ino)

    def test_filter_reapplies_member_validation_and_skips_archive_root(self):
        directory = tarfile.TarInfo(".")
        directory.type = tarfile.DIRTYPE
        self.assertIsNone(LAUNCH.runtime_data_filter(directory, self.cache))
        escaping = tarfile.TarInfo("../outside")
        with self.assertRaisesRegex(LAUNCH.CacheError, "escapes cache"):
            LAUNCH.runtime_data_filter(escaping, self.cache)

    def test_symlink_failure_never_uses_copy_fallback_or_marks_ready(self):
        alias = tarfile.TarInfo("interpreter-alias")
        alias.type, alias.linkname = tarfile.SYMTYPE, "python/bin/python3"
        _, cfg = self.fixture(extra=[(alias, None)])
        with patch.object(LAUNCH.os, "symlink", side_effect=OSError("forced link failure")):
            with patch.object(LAUNCH.RuntimeTarFile, "_find_link_target", side_effect=AssertionError("copy fallback forbidden")):
                with self.assertRaisesRegex(LAUNCH.CacheError, "link creation failed"):
                    LAUNCH.prepare(cfg)
        self.assertTrue(self.cache.is_dir())
        self.assertTrue((self.cache / "python/bin/python3").is_file())
        self.assertFalse((self.cache / alias.name).exists())
        self.assertFalse((self.cache / "READY.json").exists())

    def test_hardlink_failure_never_uses_copy_fallback_or_marks_ready(self):
        hard = tarfile.TarInfo("interpreter-hardlink")
        hard.type, hard.linkname = tarfile.LNKTYPE, "python/bin/python3"
        _, cfg = self.fixture(extra=[(hard, None)])
        with patch.object(LAUNCH.os, "link", side_effect=OSError("forced link failure")):
            with patch.object(LAUNCH.RuntimeTarFile, "_find_link_target", side_effect=AssertionError("copy fallback forbidden")):
                with self.assertRaisesRegex(LAUNCH.CacheError, "link creation failed"):
                    LAUNCH.prepare(cfg)
        self.assertTrue(self.cache.is_dir())
        self.assertTrue((self.cache / "python/bin/python3").is_file())
        self.assertFalse((self.cache / hard.name).exists())
        self.assertFalse((self.cache / "READY.json").exists())

    def test_forward_hardlink_is_rejected_without_copying_later_member(self):
        hard = tarfile.TarInfo("code/forward-copy")
        hard.type, hard.linkname = tarfile.LNKTYPE, "code/later.bin"
        later = tarfile.TarInfo("code/later.bin")
        later.size = 4
        _, cfg = self.fixture(extra=[(hard, None), (later, b"data")])
        with patch.object(LAUNCH.RuntimeTarFile, "_find_link_target", side_effect=AssertionError("forward copy forbidden")):
            with self.assertRaisesRegex(LAUNCH.CacheError, "has not been extracted"):
                LAUNCH.prepare(cfg)
        self.assertTrue(self.cache.is_dir())
        self.assertFalse((self.cache / hard.name).exists())
        self.assertFalse((self.cache / "READY.json").exists())

    def test_optional_runtime_manifest_is_checked_on_hot_reuse(self):
        data = b'{"build":"validated"}'
        member = tarfile.TarInfo("runtime-manifest.json")
        member.size = len(data)
        _, cfg = self.fixture(extra=[(member, data)])
        cfg.update(runtime_manifest="runtime-manifest.json", runtime_manifest_sha256=hashlib.sha256(data).hexdigest())
        LAUNCH.prepare(cfg)
        (self.cache / "runtime-manifest.json").write_text("changed")
        with self.assertRaisesRegex(LAUNCH.CacheError, "manifest SHA256 mismatch"):
            LAUNCH.prepare(cfg)

    def test_prepare_only_does_not_execute_service(self):
        path, _ = self.fixture()
        with patch.object(LAUNCH, "execute", side_effect=AssertionError("must not start hardware")):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(LAUNCH.main(["--config", str(path), "--prepare-only"]), 0)

    def test_flock_serializes_two_simultaneous_preparers(self):
        path, _ = self.fixture()
        command = [sys.executable, str(SCRIPT), "--config", str(path), "--prepare-only"]
        first = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        second = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        results = [process.communicate(timeout=10) for process in (first, second)]
        self.assertEqual((first.returncode, second.returncode), (0, 0), results)
        self.assertEqual(sorted(json.loads(out)["state"] for out, _ in results), ["prepared", "reused"])

    def test_execute_uses_cache_prefix_and_sanitized_thread_environment(self):
        _, cfg = self.fixture()
        LAUNCH.prepare(cfg)
        class ExecObserved(BaseException):
            pass
        with patch.object(LAUNCH.os, "execve", side_effect=ExecObserved) as execute:
            with patch.dict(os.environ, {"PYTHONHOME": "/foreign", "PYTHONPATH": "/foreign", "CUDA_VISIBLE_DEVICES": "0"}):
                with self.assertRaises(ExecObserved):
                    LAUNCH.execute(cfg)
        python, argv, env = execute.call_args.args
        self.assertEqual(python, str(self.cache / "venv/bin/python"))
        self.assertEqual(argv[-1], "run")
        self.assertEqual(argv[1], str(self.cache / "code/scripts/service.py"))
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertNotIn("PYTHONHOME", env)
        self.assertNotIn("PYTHONPATH", env)


if __name__ == "__main__":
    unittest.main()
