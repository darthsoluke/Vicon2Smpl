"""Marker data sources: live Vicon DataStream, recorded .npz replay, and C3D files.

Every source yields frames in a common format, already converted to the SMPL world convention:
    metres, Y-up (Vicon is millimetres, Z-up).
A frame is a dict: {"frame": int, "time": float (s, host clock), "names": [str], "pos": (M,3), "valid": (M,)}.
"""
import threading
import time
import numpy as np

# Vicon (x, y, z) Z-up  ->  SMPL (x, z, -y) Y-up. Proper rotation (det = +1) about the X axis.
VICON_TO_SMPL = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])


def to_smpl_frame(pos_mm, axis="z"):
    """Convert (…,3) millimetre positions from a Z-up (default) or Y-up capture frame to metres, Y-up."""
    p = np.asarray(pos_mm, np.float64) / 1000.0
    if axis == "z":
        return p @ VICON_TO_SMPL.T
    return p


def strip_subject(name):
    """Vicon sometimes reports 'Subject:LASI'; keep only the marker label."""
    return name.split(":")[-1]


class ViconSource:
    """Background thread that keeps only the newest frame from the Vicon DataStream SDK.

    Requires the official Python SDK (`vicon_dssdk`), installed from the DataStream SDK folder:
        pip install "<DataStream SDK>/Win64/Python/vicon_dssdk"   (or the Linux/macOS equivalent)
    """

    def __init__(self, host="localhost:801", subject=None, up_axis="z", server_push=True):
        try:
            from vicon_dssdk import ViconDataStream
        except ImportError as e:
            raise ImportError("vicon_dssdk not found. Install the Python package that ships with the Vicon "
                              "DataStream SDK (pip install <SDK>/Python/vicon_dssdk).") from e
        self.VDS = ViconDataStream
        self.host, self.subject, self.up_axis = host, subject, up_axis
        self.client = ViconDataStream.Client()
        print(f"[vicon] connecting to {host} ...")
        self.client.Connect(host)
        if not self.client.IsConnected():
            raise RuntimeError(f"could not connect to Vicon DataStream at {host}")
        self.client.EnableMarkerData()
        mode = ViconDataStream.Client.StreamMode.EServerPush if server_push else ViconDataStream.Client.StreamMode.EClientPull
        self.client.SetStreamMode(mode)
        try:  # newer SDKs: keep the queue short to minimise latency
            self.client.SetBufferSize(1)
        except Exception:
            pass
        self._lock = threading.Lock()
        self._latest = None
        self._seq = 0
        self._stop = threading.Event()
        self.frame_rate = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _pick_subject(self):
        subs = list(self.client.GetSubjectNames())
        if not subs:
            return None
        if self.subject is None:
            if len(subs) > 1:
                print(f"[vicon] several subjects {subs}; using '{subs[0]}' (pass --subject to choose)")
            self.subject = subs[0]
        elif self.subject not in subs:
            raise RuntimeError(f"subject '{self.subject}' not in stream; available: {subs}")
        return self.subject

    def _run(self):
        c = self.client
        while not self._stop.is_set():
            try:
                c.GetFrame()  # blocks (server push) until a new frame arrives
            except self.VDS.DataStreamException:
                time.sleep(0.001)
                continue
            t = time.perf_counter()
            subj = self._pick_subject()
            if subj is None:
                continue
            if self.frame_rate is None:
                try:
                    self.frame_rate = c.GetFrameRate()
                except Exception:
                    pass
            names, pos, valid = [], [], []
            for item in c.GetMarkerNames(subj):
                name = item[0] if isinstance(item, (tuple, list)) else item
                (xyz, occluded) = c.GetMarkerGlobalTranslation(subj, name)
                names.append(strip_subject(name))
                pos.append(xyz)
                valid.append(not occluded)
            pos = to_smpl_frame(np.array(pos, np.float64).reshape(-1, 3), self.up_axis)
            valid = np.array(valid, bool)
            pos[~valid] = np.nan
            try:
                latency = c.GetLatencyTotal()
            except Exception:
                latency = float("nan")
            frame = {"frame": int(c.GetFrameNumber()), "time": t, "names": names, "pos": pos, "valid": valid,
                     "latency": latency}
            with self._lock:
                self._latest = frame
                self._seq += 1

    def get_latest(self, last_seq=-1, timeout=1.0):
        """Return (seq, frame) for the newest frame newer than `last_seq`; blocks up to `timeout` s."""
        t0 = time.perf_counter()
        while True:
            with self._lock:
                if self._seq != last_seq and self._latest is not None:
                    return self._seq, self._latest
            if time.perf_counter() - t0 > timeout:
                return last_seq, None
            time.sleep(0.0002)

    def close(self):
        self._stop.set()
        try:
            self.client.Disconnect()
        except Exception:
            pass


# ------------------------------------------------------------------------- files
def save_recording(path, names, frames, pos, valid, fps):
    np.savez_compressed(path, names=np.array(names), frames=np.asarray(frames), pos=np.asarray(pos, np.float32),
                        valid=np.asarray(valid, bool), fps=float(fps or 0))


def load_recording(path, axis="z"):
    """Load an .npz written by record.py (already metres / Y-up) or a .c3d file.
    Returns dict(names, pos (T,M,3) with NaN where invalid, valid (T,M), fps)."""
    path = str(path)
    if path.lower().endswith(".c3d"):
        return load_c3d(path, axis=axis)
    d = np.load(path, allow_pickle=False)
    pos = d["pos"].astype(np.float64)
    valid = d["valid"].astype(bool) & np.all(np.isfinite(pos), -1)
    pos[~valid] = np.nan
    return {"names": [str(n) for n in d["names"]], "pos": pos, "valid": valid, "fps": float(d["fps"])}


def load_c3d(path, axis="z"):
    """C3D via `ezc3d` (pip install ezc3d). Assumes millimetres; `axis` is the capture up-axis."""
    import ezc3d

    c = ezc3d.c3d(path)
    pts = c["data"]["points"]  # (4, M, T)
    names = [strip_subject(n) for n in c["parameters"]["POINT"]["LABELS"]["value"]]
    unit = c["parameters"]["POINT"].get("UNITS", {}).get("value", ["mm"])[0].strip().lower()
    scale = {"mm": 1.0, "m": 1000.0, "cm": 10.0}.get(unit, 1.0)
    xyz = np.transpose(pts[:3], (2, 1, 0)) * scale  # (T,M,3) in mm
    pos = to_smpl_frame(xyz, axis)
    valid = np.all(np.isfinite(pos), -1) & ~np.all(np.abs(xyz) < 1e-9, -1)
    if pts.shape[0] > 3:  # residual < 0 marks an invalid point
        valid &= np.transpose(pts[3], (1, 0)) >= 0
    pos[~valid] = np.nan
    fps = float(c["header"]["points"]["frame_rate"])
    return {"names": names, "pos": pos, "valid": valid, "fps": fps}


class ReplaySource:
    """Plays a recording back in (approximately) real time — same interface as ViconSource."""

    def __init__(self, path, speed=1.0, loop=False, axis="z"):
        self.rec = load_recording(path, axis)
        self.fps = self.rec["fps"] or 100.0
        self.speed, self.loop = speed, loop
        self.t0 = time.perf_counter()
        self.frame_rate = self.fps
        self.done = False

    def get_latest(self, last_seq=-1, timeout=1.0):
        T = len(self.rec["pos"])
        while True:
            i = int((time.perf_counter() - self.t0) * self.fps * self.speed)
            if i >= T:
                if not self.loop:
                    self.done = True
                    return last_seq, None
                self.t0 = time.perf_counter()
                i = 0
            if i != last_seq:
                return i, {"frame": i, "time": time.perf_counter(), "names": self.rec["names"],
                           "pos": self.rec["pos"][i], "valid": self.rec["valid"][i], "latency": 0.0}
            time.sleep(0.0005)

    def close(self):
        pass


class NameMapper:
    """Reorders a frame's markers into the rig's marker order (by label). Unknown/missing labels -> invalid."""

    def __init__(self, rig_names):
        self.rig_names = list(rig_names)
        self._cache_key, self._idx = None, None

    def __call__(self, frame):
        key = tuple(frame["names"])
        if key != self._cache_key:
            lut = {n: i for i, n in enumerate(frame["names"])}
            self._idx = np.array([lut.get(n, -1) for n in self.rig_names])
            missing = [n for n, i in zip(self.rig_names, self._idx) if i < 0]
            if missing:
                print(f"[mapper] {len(missing)} calibrated markers not in stream: {missing}")
            self._cache_key = key
        M = len(self.rig_names)
        pos = np.full((M, 3), np.nan)
        valid = np.zeros(M, bool)
        ok = self._idx >= 0
        pos[ok] = frame["pos"][self._idx[ok]]
        valid[ok] = frame["valid"][self._idx[ok]]
        return pos, valid
