"""Official C++ SDK subprocess adapter, using a single read-only DataStream client."""
import json
import subprocess
import threading
import time

import numpy as np

from .sources import strip_subject, to_smpl_frame


def subject_frame(packet, subject):
    """Select an explicit subject; never merge markers belonging to different people."""
    subjects = {s["name"]: s for s in packet["subjects"]}
    if subject not in subjects:
        return None
    markers = subjects[subject]["markers"]
    names = [strip_subject(m["name"]) for m in markers]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate marker labels in subject")
    pos = np.array([m["xyz"] if m["valid"] and m["xyz"] is not None else [np.nan] * 3
                    for m in markers], dtype=np.float64).reshape(-1, 3)
    valid = np.array([m["valid"] for m in markers], bool) & np.isfinite(pos).all(-1)
    pos = to_smpl_frame(pos)
    pos[~valid] = np.nan
    return dict(frame=int(packet["frame"]), time=packet["arrival_time"], names=names,
                pos=pos, valid=valid, subject=subject)


class NativeSource:
    def __init__(self, binary, host):
        self.binary, self.host = str(binary), host
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.process = None
        self.latest = None
        self.seq = 0
        self.error = "等待 DataStream"
        self.frame_rate = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                self.process = subprocess.Popen([self.binary, self.host], stdout=subprocess.PIPE,
                                                text=True, bufsize=1)
                for line in self.process.stdout:
                    if self.stop.is_set():
                        break
                    packet = json.loads(line)
                    packet["arrival_time"] = time.perf_counter()
                    with self.lock:
                        self.latest = packet
                        self.frame_rate = float(packet["fps"])
                        self.seq += 1
                        self.error = ""
                code = self.process.wait(timeout=4)
                self.error = f"DataStream 连接结束（{code}），正在重连"
            except Exception as e:
                self.error = str(e)
            finally:
                if self.process and self.process.poll() is None:
                    self.process.kill()
                    self.process.wait()
            self.stop.wait(2)

    def snapshot(self):
        with self.lock:
            return self.seq, self.latest

    def close(self):
        self.stop.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.thread.join(timeout=5)
