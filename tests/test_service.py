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

    def test_venv_python_symlink_is_preserved_in_child_and_manager_launches(self):
        venv_bin = self.root / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        python = venv_bin / "python"
        python.symlink_to(Path(sys.executable).resolve())
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
