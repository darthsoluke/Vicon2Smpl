"""Live visualisation through aitviewer's remote mode.

The solver process only needs `aitviewer.remote` (pure python + websockets), so the viewer can run in a
separate environment / machine:
    python -m aitviewer.server            # in an env where aitviewer is fully installed
The publisher computes the full mesh itself (no SMPL files needed on the viewer side) and sends it at a fixed
display rate from a background thread, so rendering never blocks the solver.
"""
import threading
import time
import numpy as np


class AitviewerPublisher:
    def __init__(self, model, rig, host="localhost", port=8417, fps=30, spawn=False, show_markers=True):
        from aitviewer.remote.viewer import RemoteViewer
        from aitviewer.remote.renderables.meshes import RemoteMeshes
        from aitviewer.remote.renderables.spheres import RemoteSpheres

        self.model, self.rig, self.dt = model, rig, 1.0 / fps
        self.viewer = RemoteViewer.create_new_process() if spawn else RemoteViewer(host, port)
        if not self.viewer.connected:
            raise RuntimeError(f"no aitviewer server at {host}:{port} (start it with: python -m aitviewer.server)")
        J = model.num_joints
        R0 = np.broadcast_to(np.eye(3), (J, 3, 3))
        v0 = model.vertices(R0, np.zeros(3), rig.v_rest, rig.J_rest).astype(np.float32)
        self.mesh = RemoteMeshes(self.viewer, v0, model.faces.astype(np.int32), name="SMPL",
                                 color=(0.62, 0.70, 0.86, 1.0))
        self.spheres = None
        if show_markers:
            self.spheres = RemoteSpheres(self.viewer, np.zeros((rig.num_markers, 3), np.float32), radius=0.012,
                                         color=(0.95, 0.25, 0.2, 1.0), name="Markers")
        self._slot = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push(self, R, trans, obs=None, valid=None):
        with self._lock:
            self._slot = (R.copy(), trans.copy(), None if obs is None else obs.copy(),
                          None if valid is None else valid.copy())

    def _run(self):
        last = None
        while not self._stop.is_set():
            t0 = time.perf_counter()
            with self._lock:
                item, self._slot = self._slot, None
            if item is not None and self.viewer.connected:
                R, trans, obs, valid = item
                verts = self.model.vertices(R, trans, self.rig.v_rest, self.rig.J_rest).astype(np.float32)
                self.mesh.update_frames(verts, frames=0)
                if self.spheres is not None and obs is not None:
                    p = np.where(valid[:, None], obs, -100.0).astype(np.float32)  # hide occluded markers
                    self.spheres.update_frames(p, frames=0)
            time.sleep(max(0.0, self.dt - (time.perf_counter() - t0)))

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)
        # RemoteViewer.close_connection() waits for the close handshake without a timeout; bound it.
        try:
            import asyncio
            asyncio.run_coroutine_threadsafe(self.viewer._async_close(), self.viewer.loop).result(timeout=1.0)
        except Exception:
            pass
