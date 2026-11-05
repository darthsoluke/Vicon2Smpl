"""Launcher integration checks use only fake local services, never Vicon hardware."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import ModuleType, SimpleNamespace


SERVICE = Path(__file__).resolve().parents[1] / "scripts" / "service.py"
spec = importlib.util.spec_from_file_location("vicon_service", SERVICE)
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)

FAKE_WEB = r'''
import argparse,json,os,signal,socket,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--output');p.add_argument('--source');p.add_argument('--bind');p.add_argument('--port',type=int)
a,_=p.parse_known_args();out=Path(a.output)
alive=True
def stop(sig,frame):
    global alive
    alive=False
signal.signal(signal.SIGINT,stop)
s=socket.socket();s.bind((a.bind,a.port));s.listen()
with (out/'starts').open('a') as f:f.write(str(os.getpid())+'\n')
(out/'environment.json').write_text(json.dumps({k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_VISIBLE_DEVICES','NUMBA_CACHE_DIR','LD_LIBRARY_PATH')}))
while alive:
    mode=(out/'mode.txt').read_text().strip() if (out/'mode.txt').exists() else 'preview'
    if mode!='no_status':
        temp=out/'fake-status.tmp'
        temp.write_text(json.dumps(dict(mode=mode,source=a.source)))
        temp.replace(out/'status.json')
    time.sleep(.05)
(out/'graceful_exit').write_text('SIGINT')
s.close()
'''


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vicon service test ")
        self.root = Path(self.temp.name).resolve()
        (self.root / "scripts").mkdir()
        (self.root / "output").mkdir()
        (self.root / "library").mkdir()
        (self.root / "cache").mkdir()
        shutil.copyfile(SERVICE, self.root / "scripts" / "service.py")
        (self.root / "run_web.py").write_text(FAKE_WEB)
        (self.root / "reader").write_text("#!/bin/sh\nexit 91\n")
        (self.root / "reader").chmod(0o755)
        (self.root / "model.npz").write_bytes(b"fake model; never opened by fake service")
        with socket.socket() as port:
            port.bind(("127.0.0.1", 0))
            self.port = port.getsockname()[1]
        self.config = self.root / "config.json"
        self.cfg = dict(python=sys.executable, code_root=".", output="output", source="127.0.0.1:8802",
                        native_reader="reader", model_path="model.npz", model_type="smplx",
                        bind="127.0.0.1", port=self.port, fit_hands=False,
                        library_path=["library"], numba_cache="cache")
        self.write_config()

    def write_config(self):
        self.config.write_text(json.dumps(self.cfg))

    def command(self, name, *extra):
        return [sys.executable, str(self.root / "scripts" / "service.py"), "--config", str(self.config), name, *extra]

    def cli(self, name, *extra, success=True):
        result = subprocess.run(self.command(name, *extra), capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        return json.loads(result.stdout if success else result.stderr)

    def wait_file(self, name):
        path = self.root / "output" / name
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if path.exists():
                return path
            time.sleep(.03)
        self.fail(f"Timed out waiting for {name}")

    def tearDown(self):
        # Tests never force-kill. Keep even failure cleanup graceful.
        record_path = self.root / "output" / service.PID_NAME
        if record_path.exists():
            record = json.loads(record_path.read_text())
            info = service._process_info(record.get("manager_pid"))
            if service._owned(info, record.get("manager_birth"), [str(self.root / "scripts" / "service.py")]):
                with self.subTest(cleanup=True):
                    os.kill(record["manager_pid"], signal.SIGINT)
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline and service._process_info(record["manager_pid"]):
                        time.sleep(.05)
        self.temp.cleanup()

    def test_dry_run_resolves_relative_paths_prints_env_and_never_starts_reader(self):
        result = self.cli("preflight", "--dry-run")
        self.assertFalse(result["hardware_connected"])
        self.assertEqual(result["cwd"], str(self.root))
        self.assertIn(str(self.root / "reader"), result["argv"])
        self.assertEqual(result["environment"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(result["environment"]["NUMBA_CACHE_DIR"], str(self.root / "cache"))
        self.assertTrue(result["environment"]["LD_LIBRARY_PATH"].startswith(str(self.root / "library")))
        self.assertFalse((self.root / "output" / "starts").exists())
        self.assertFalse((self.root / "output" / service.PID_NAME).exists())

    def write_profiles(self, profiles, default="soma"):
        package = self.root / "vsmpl"
        package.mkdir(exist_ok=True)
        (package / "__init__.py").write_text("")
        shutil.copyfile(SERVICE.parents[1] / "vsmpl/model_profiles.py", package / "model_profiles.py")
        (self.root / "models.json").write_text(json.dumps(dict(default=default, profiles=profiles)))
        self.cfg["model_profiles"] = "models.json"
        self.write_config()

    def test_soma_local_assets_arguments_and_independent_warp_cache(self):
        (self.root / "soma-assets").mkdir()
        self.cfg.update(model_type="soma", model_path="soma-assets", num_betas=12,
                        soma_lod="low", soma_identity="soma", warp_cache="warp-cache")
        self.write_config()
        self.assertIn("Expected dir", self.cli("preflight", success=False)["error"])
        (self.root / "warp-cache").mkdir()
        result = self.cli("preflight", "--dry-run")
        argv = result["argv"]
        for option, value in (("--model-type", "soma"), ("--num-betas", "12"),
                              ("--soma-lod", "low"), ("--soma-identity", "soma")):
            self.assertEqual(argv[argv.index(option) + 1], value)
        self.assertEqual(result["environment"]["WARP_CACHE_PATH"], str(self.root / "warp-cache"))
        self.assertEqual(result["environment"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertFalse((self.root / "output/starts").exists())

    def test_soma_rejects_full_hands_and_service_online_assets(self):
        (self.root / "soma-assets").mkdir()
        self.cfg.update(model_type="soma", model_path="soma-assets", fit_hands=True)
        self.write_config()
        self.assertIn("fit_hands requires smplx", self.cli("preflight", success=False)["error"])
        self.cfg["fit_hands"] = False
        for value in ("hf", "auto", "https://example.invalid/assets"):
            self.cfg["model_path"] = value
            self.write_config()
            with self.subTest(value=value):
                self.assertIn("local assets", self.cli("preflight", success=False)["error"])

    def test_service_soma_rejects_experimental_licensed_identity(self):
        (self.root / "soma-assets").mkdir()
        for identity in ("smpl", "smplx"):
            self.cfg.update(model_type="soma", model_path="soma-assets", soma_identity=identity)
            self.write_config()
            with self.subTest(identity=identity):
                self.assertIn("licensed identity", self.cli("preflight", success=False)["error"])
        self.assertFalse((self.root / "output/starts").exists())

    def test_profile_default_binds_actual_child_arguments_without_numpy(self):
        (self.root / "soma-assets").mkdir()
        profiles = [dict(id="smplx", label="SMPL-X", model_type="smplx", model_path="model.npz"),
                    dict(id="soma", label="SOMA", model_type="soma", model_path="soma-assets", soma_lod="low", num_betas=12)]
        self.write_profiles(profiles)
        before = set(sys.modules)
        loaded = service.load_config(self.config)
        self.assertNotIn("numpy", set(sys.modules) - before)
        self.assertEqual(loaded["model_type"], "soma")
        self.assertEqual(loaded["model_path"], str(self.root / "soma-assets"))
        result = self.cli("preflight", "--dry-run")
        argv = result["argv"]
        self.assertEqual(argv[argv.index("--model-type") + 1], "soma")
        self.assertEqual(argv[argv.index("--model-profiles") + 1], str(self.root / "models.json"))
        self.assertEqual(result["model_profiles"]["default"], "soma")
        self.assertEqual(len(result["model_profiles"]["profiles"]), 2)
        self.assertTrue(all(len(item["fingerprint"]) == 64 for item in result["model_profiles"]["profiles"]))
        self.assertFalse((self.root / "output/starts").exists())

    def test_every_profile_path_is_checked_before_starting_default(self):
        (self.root / "soma-assets").mkdir()
        (self.root / "regions.json").write_text('{"C7":[1]}')
        self.write_profiles([
            dict(id="soma", label="SOMA", model_type="soma", model_path="soma-assets"),
            dict(id="smplx", label="SMPL-X hands", model_type="smplx", model_path="model.npz",
                 fit_hands=True, body_regions="regions.json", hand_prior="not-downloaded.npz")])
        result = self.cli("start", success=False)
        self.assertIn("hand_prior", result["error"])
        self.assertFalse((self.root / "output/starts").exists())

    def test_changed_profile_snapshot_fails_before_launch(self):
        self.write_profiles([dict(id="smplx", label="SMPL-X", model_type="smplx", model_path="model.npz")], default="smplx")
        loaded = service.load_config(self.config)
        content = json.loads((self.root / "models.json").read_text())
        content["profiles"][0]["label"] = "changed after load"
        (self.root / "models.json").write_text(json.dumps(content))
        with patch.object(service, "__file__", str(self.root / "scripts/service.py")):
            with self.assertRaisesRegex(service.ServiceError, "changed after"):
                service.validate_paths(loaded)

    def test_healthcheck_validates_all_profile_model_arguments_without_hardware(self):
        health_spec = importlib.util.spec_from_file_location("profile_healthcheck", SERVICE.parent / "healthcheck.py")
        health = importlib.util.module_from_spec(health_spec)
        health_spec.loader.exec_module(health)
        (self.root / "soma-assets").mkdir()
        self.write_profiles([
            dict(id="soma", label="SOMA", model_type="soma", model_path="soma-assets", soma_lod="low", num_betas=12),
            dict(id="smplx", label="SMPL-X", model_type="smplx", model_path="model.npz")])
        calls = []
        def load(*args, **kwargs):
            calls.append((args, kwargs))
            soma = args[1] == "soma"
            return SimpleNamespace(num_joints=77 if soma else 55,
                                   v_template=range(4505 if soma else 10475), num_betas=kwargs["num_betas"])
        fake = {}
        for name, attributes in {
            "numpy": dict(empty=lambda *a, **kw: None),
            "vsmpl.body_model": dict(BodyModel=SimpleNamespace(load=load)),
            "vsmpl.calibration_input": dict(prepare_calibration_input=lambda rec, regions, count: (rec, regions)),
            "vsmpl.hand_pca": dict(HandPCA=SimpleNamespace()),
        }.items():
            module = ModuleType(name)
            module.__dict__.update(attributes)
            fake[name] = module
        with patch.dict(sys.modules, fake):
            with patch.object(health.importlib, "import_module", return_value=SimpleNamespace(__version__="fixture")):
                with patch.object(health.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="fixture", stderr="")):
                    with patch.object(health.platform, "system", return_value="Linux"):
                        with patch.object(health.platform, "platform", return_value="fixture Linux"):
                            report = health.check(self.config)
                            self.cfg.pop("model_profiles")
                            self.cfg.update(model_type="soma", model_path="soma-assets", soma_identity="smplx")
                            self.write_config()
                            rejected = health.check(self.config)
        self.assertEqual(report["status"], "PASS", report)
        self.assertFalse(report["hardware_connection_opened"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(rejected["status"], "FAIL")
        self.assertIn("licensed identity", next(item for item in rejected["checks"] if item["name"] == "model_profiles")["detail"])
        self.assertEqual(calls[0][1], dict(num_betas=12, soma_lod="low", soma_identity="soma"))
        self.assertEqual(calls[1][1], dict(num_betas=10))
        default = next(item for item in report["checks"] if item["name"] == "licensed_model_and_marker_regions")
        self.assertEqual(default["detail"]["joints"], 77)
        self.assertEqual(default["detail"]["lod"], "low")

    def test_venv_python_symlink_is_preserved_in_child_and_manager_launches(self):
        venv_root = self.root / "venv"
        # Include pyvenv.cfg so standalone Python distributions find their stdlib.
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--symlinks", str(venv_root)],
                       check=True, capture_output=True, text=True, timeout=30)
        venv_bin = venv_root / "bin"
        python = venv_bin / "python"
        self.cfg["python"] = "venv/bin/python"
        self.write_config()
        loaded = service.load_config(self.config)
        self.assertEqual(loaded["python"], str(python))
        self.assertNotEqual(loaded["python"], str(python.resolve()))
        self.assertEqual(service.child_argv(loaded)[0], str(python))
        self.assertEqual(self.cli("preflight", "--dry-run")["argv"][0], str(python))
        self.cli("start")
        self.wait_file("status.json")
        record = json.loads((self.root / "output" / service.PID_NAME).read_text())
        self.assertEqual(record["child_argv"][0], str(python))
        state = self.cli("status")
        self.assertTrue(state["owned"] and state["child_owned"])
        self.assertEqual(self.cli("stop")["state"], "stopped")

    def test_data_output_is_validated_and_forwarded_without_moving_hot_state(self):
        bulk = self.root / "bulk"
        self.cfg["data_output"] = "bulk"
        self.write_config()
        self.assertIn("Expected dir", self.cli("preflight", success=False)["error"])
        bulk.mkdir()
        args = self.cli("preflight")["argv"]
        self.assertEqual(args[args.index("--data-output") + 1], str(bulk))
        self.assertEqual(args[args.index("--output") + 1], str(self.root / "output"))
        self.cli("start")
        self.wait_file("status.json")
        self.assertTrue((self.root / "output" / service.LOCK_NAME).is_file())
        self.assertFalse((bulk / service.LOCK_NAME).exists())
        self.assertFalse((bulk / "status.json").exists())
        self.assertEqual(self.cli("stop")["state"], "stopped")

    def test_concurrent_starts_create_one_child_and_repeat_start_reuses_it(self):
        starters = [subprocess.Popen(self.command("start"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    for _ in range(4)]
        replies = []
        for process in starters:
            out, err = process.communicate(timeout=15)
            self.assertEqual(process.returncode, 0, out + err)
            replies.append(json.loads(out))
        self.assertEqual(sum(r["action"] == "started" for r in replies), 1)
        self.wait_file("status.json")
        state = self.cli("status")
        self.assertTrue(state["owned"] and state["child_owned"])
        self.assertEqual(self.cli("start")["child_pid"], state["child_pid"])
        self.assertEqual(len(self.wait_file("starts").read_text().splitlines()), 1)
        env = json.loads(self.wait_file("environment.json").read_text())
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(self.cli("stop")["state"], "stopped")
        self.assertEqual(self.wait_file("graceful_exit").read_text(), "SIGINT")

    def test_busy_modes_refuse_stop_and_keep_same_process_alive(self):
        self.cli("start")
        self.wait_file("status.json")
        child = self.cli("status")["child_pid"]
        for mode in sorted(service.BUSY_MODES):
            (self.root / "output" / "mode.txt").write_text(mode)
            deadline = time.monotonic() + 3
            while self.cli("status")["mode"] != mode and time.monotonic() < deadline:
                time.sleep(.05)
            error = self.cli("stop", success=False)
            self.assertIn("Refusing to stop while", error["error"])
            self.assertEqual(self.cli("status")["child_pid"], child)
            self.assertIsNotNone(service._process_info(child))
        (self.root / "output" / "mode.txt").write_text("calibrated")
        time.sleep(.15)
        self.assertEqual(self.cli("stop")["state"], "stopped")

    def test_occupied_port_prevents_start_without_creating_child(self):
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", self.port))
            occupied.listen()
            error = self.cli("start", success=False)
        self.assertIn("Cannot bind", error["error"])
        self.assertFalse((self.root / "output" / "starts").exists())

    def test_forged_pid_receipt_is_never_signalled(self):
        self.cli("start")
        self.wait_file("status.json")
        path = self.root / "output" / service.PID_NAME
        real = json.loads(path.read_text())
        forged = dict(real, manager_pid=os.getpid(), manager_birth=service._process_info(os.getpid())["birth"])
        path.write_text(json.dumps(forged))
        try:
            error = self.cli("stop", success=False)
            self.assertIn("ownership", error["error"])
            self.assertIsNotNone(service._process_info(real["child_pid"]))
        finally:
            path.write_text(json.dumps(real))

    def test_invalid_config_and_missing_path_fail_before_launch(self):
        self.cfg["fit_hands"] = "false"
        self.write_config()
        self.assertIn("JSON boolean", self.cli("start", success=False)["error"])
        self.cfg["fit_hands"] = False
        self.cfg["native_reader"] = "missing-reader"
        self.write_config()
        self.assertIn("Expected file", self.cli("start", success=False)["error"])
        self.assertFalse((self.root / "output" / "starts").exists())

    def test_no_status_refuses_stop(self):
        (self.root / "output" / "mode.txt").write_text("no_status")
        self.cli("start")
        error = self.cli("stop", success=False)
        self.assertIn("fresh idle status", error["error"])


if __name__ == "__main__":
    unittest.main()
