"""Browser controls for a separate Vicon2Smpl field test; never controls Shogun capture."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import viser

from vsmpl.body_model import BodyModel, MarkerRig
from vsmpl.native_source import NativeSource, subject_frame
from vsmpl.outputs import SequenceRecorder
from vsmpl.solver import RealtimeSolver
from vsmpl.sources import NameMapper, save_recording


class FieldApp:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.output)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.mode, self.message = "preview", "选择被试后开始标定；尚未生成任何拟合结果。"
        self.run = None
        self.samples = []
        self.recording = None
        self.calibration_path = None
        self.solver = self.rig = self.recorder = None
        self.selected = None
        self.last_result = None
        self.fit_count = 0
        self.model = BodyModel.load(args.model_path, args.model_type, args.gender)
        self.source = NativeSource(args.native_reader, args.source)
        self.server = viser.ViserServer(host=args.bind, port=args.port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="fixed", control_width="medium",
                                        show_share_button=False, show_logo=False)
        self.server.scene.set_up_direction("+y")
        self.server.scene.add_grid("/floor", plane="xz", width=10, height=10)
        self.cloud = self.server.scene.add_point_cloud("/observed", points=np.zeros((1, 3), np.float32),
                                                       colors=(239, 91, 86), point_size=0.016, visible=False)
        self.fitted = self.server.scene.add_point_cloud("/fitted", points=np.zeros((1, 3), np.float32),
                                                        colors=(70, 200, 160), point_size=0.012, visible=False)
        self.mesh = None
        self.server.gui.add_markdown("# Vicon → SMPL\n全身实时拟合 · 独立现场测试")
        self.status = self.server.gui.add_markdown("正在连接动捕数据…")
        self.subject = self.server.gui.add_dropdown("被试", options=("等待被试",))
        self.subject.on_update(self.change_subject)
        with self.server.gui.add_folder("1 · 被试标定", expand_by_default=True):
            self.duration = self.server.gui.add_slider("采样时长 / 秒", min=15, max=90, step=5, initial_value=45)
            self.server.gui.add_markdown("点击后有 **5 秒准备**。先保持 **T-pose 3 秒**，再做手臂画圈、屈肘、深蹲、转身与抬腿。")
            self.capture = self.server.gui.add_button("开始标定采样")
            self.capture.on_click(self.start_sampling)
            self.finish = self.server.gui.add_button("停止采样并保留", disabled=True)
            self.finish.on_click(self.finish_sampling)
            self.fit_button = self.server.gui.add_button("计算本次标定", disabled=True)
            self.fit_button.on_click(self.start_calibration)
        with self.server.gui.add_folder("2 · 实时对照", expand_by_default=True):
            self.track_button = self.server.gui.add_button("开始实时拟合", disabled=True)
            self.track_button.on_click(self.start_tracking)
            self.stop_button = self.server.gui.add_button("停止拟合并保存", disabled=True)
            self.stop_button.on_click(self.stop_tracking)
            self.center = self.server.gui.add_button("居中查看")
            self.center.on_click(self.center_view)
            self.show_mesh = self.server.gui.add_checkbox("显示人体网格", initial_value=True)
            self.show_mesh.on_update(lambda _: self._mesh_visibility())
        self.detail = self.server.gui.add_markdown("红点：观测 marker；绿点：拟合 marker。手指质量取决于手部贴点布局。")
        self.server.gui.add_markdown("本页仅采样标定 marker 与保存派生拟合；正式录制仍使用原录制页面。")
        self.server.on_client_connect(lambda event: self.center_view(event))

    def _mesh_visibility(self):
        if self.mesh:
            self.mesh.visible = self.show_mesh.value and self.mode == "tracking"

    def center_view(self, _):
        _, packet = self.source.snapshot()
        fr = subject_frame(packet, self.selected) if packet and self.selected else None
        points = fr["pos"][fr["valid"]] if fr else np.empty((0, 3))
        target = np.median(points, axis=0) if len(points) else np.array([0., 0.8, 0.])
        for client in self.server.get_clients().values():
            client.camera.look_at = target
            client.camera.position = target + [2.8, 1.2, 3.2]
            client.camera.up_direction = (0., 1., 0.)

    def change_subject(self, _):
        with self.lock:
            if self.subject.value == "等待被试" or self.subject.value == self.selected:
                return
            if self.mode not in ("preview", "calibrated"):
                return
            self.selected = self.subject.value
            self.solver = self.rig = None
            self.recording = None
            self.calibration_path = None
            self.last_result = None
            self.mode = "preview"
            self.track_button.disabled = self.fit_button.disabled = True
            self.fitted.visible = False
            if self.mesh:
                self.mesh.visible = False
            self.message = f"已选择 {self.selected}；请为此人采样标定。"
            self.center_view(None)

    def start_sampling(self, _):
        with self.lock:
            _, packet = self.source.snapshot()
            if self.mode not in ("preview", "calibrated") or not packet or not self.selected:
                return
            fr = subject_frame(packet, self.selected)
            if time.perf_counter() - packet["arrival_time"] > 1 or not fr or fr["valid"].sum() < 15:
                self.message = "可见 marker 不足或数据过期；请检查 Shogun 的全身标签。"
                return
            self.mode = "countdown"
            self.countdown_end = time.perf_counter() + 5
            self.samples = []
            self.run = self.root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
            self.run.mkdir()
            self.recording = self.calibration_path = None
            self.solver = self.rig = None
            self.track_button.disabled = self.fit_button.disabled = self.capture.disabled = True
            self.subject.disabled = self.duration.disabled = True
            self.finish.disabled = False
            self.fitted.visible = False
            if self.mesh:
                self.mesh.visible = False

    def finish_sampling(self, _):
        with self.lock:
            if self.mode not in ("sampling", "countdown"):
                return
            self.mode = "preview"
            self.capture.disabled = self.subject.disabled = self.duration.disabled = False
            self.finish.disabled = True
            if not self.samples:
                self.message = "未采到 marker；可重新开始。"
                return
            first = self.samples[0]
            mapper = NameMapper(first["names"])
            ordered = [mapper(fr) for fr in self.samples]
            pos, valid = (np.stack([v[i] for v in ordered]) for i in range(2))
            times = np.array([fr["time"] - first["time"] for fr in self.samples])
            frames = np.array([fr["frame"] for fr in self.samples])
            fps = (len(times) - 1) / max(times[-1], 1e-6)
            self.recording = dict(names=first["names"], pos=pos, valid=valid, fps=fps)
            self.tpose_window = (int(np.searchsorted(times, 0.4)), int(np.searchsorted(times, 2.6)))
            save_recording(self.run / "calibration_markers.npz", first["names"], frames, pos, valid, fps)
            np.savez_compressed(self.run / "sampling_clock.npz", relative_host_arrival_seconds=times,
                                source_frame_numbers=frames)
            self._receipt("sampled", source_fps=self.source.frame_rate, sampled_fps=fps,
                          samples=len(times), tpose_window=list(self.tpose_window), duration_seconds=float(times[-1]))
            self.fit_button.disabled = times[-1] < 10 or self.tpose_window[1] - self.tpose_window[0] < 5
            self.message = f"已保留 {len(times)} 帧 / {times[-1]:.1f} 秒。" + ("可以计算标定。" if not self.fit_button.disabled else "采样太短，请重新采样。")

    def _receipt(self, state, **extra):
        p = self.run / "session.json"
        old = json.loads(p.read_text()) if p.exists() else {}
        old.update(state=state, subject=self.selected, model_type=self.args.model_type,
                   source=self.args.source, coordinate_system="metres, Y-up; Vicon (x,z,-y)",
                   raw_capture_modified=False, **extra)
        p.write_text(json.dumps(old, ensure_ascii=False, indent=2))

    def start_calibration(self, _):
        with self.lock:
            if self.mode != "preview" or self.fit_button.disabled:
                return
            self.mode = "calibrating"
            self.fit_button.disabled = self.capture.disabled = self.subject.disabled = True
            self.message = "正在计算被试体型与 marker 布局；进度保存在 calibration.log。"
            # Run the existing CLI in a worker process: its CPU work cannot block the browser.
            metadata = self.run / "calibration_request.json"
            metadata.write_text(json.dumps({"tpose": list(self.tpose_window)}))
            threading.Thread(target=self._calibrate, daemon=True).start()

    def _calibrate(self):
        import subprocess
        import sys
        output = self.run / "subject_calib.npz"
        code = """import json,numpy as np,sys
from vsmpl.body_model import BodyModel
from vsmpl.calibration import calibrate
from vsmpl.sources import load_recording
model=BodyModel.load(sys.argv[1],sys.argv[2],sys.argv[3])
rec=load_recording(sys.argv[4])
window=json.load(open(sys.argv[5]))['tpose']
result=calibrate(model,rec,tpose=tuple(window))
np.savez(sys.argv[6],model_type=sys.argv[2],gender=sys.argv[3],num_betas=10,**result)
"""
        try:
            with (self.run / "calibration.log").open("w") as log:
                subprocess.run([sys.executable, "-u", "-c", code, self.args.model_path, self.args.model_type,
                                self.args.gender, str(self.run / "calibration_markers.npz"),
                                str(self.run / "calibration_request.json"), str(output)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            with self.lock:
                self.calibration_path = output
                self._load_calibration(output)
                self.mode = "calibrated"
                self.message = "标定完成。检查逐点残差后，点击开始实时拟合。"
                self._receipt("calibrated", calibration_file=output.name)
                self.track_button.disabled = False
        except Exception as e:
            with self.lock:
                self.mode = "preview"
                self.message = f"标定失败：{e}。采样与日志已保留。"
                self._receipt("calibration_failed", error=str(e))
                self.fit_button.disabled = False
        finally:
            self.capture.disabled = self.subject.disabled = False

    def _load_calibration(self, path):
        with np.load(path, allow_pickle=False) as calib:
            self.rig = MarkerRig.from_calibration(self.model, calib)
            residual = calib["per_marker_rms"] * 1000
        self.solver = RealtimeSolver(self.rig)
        self.mapper = NameMapper(self.rig.names)
        self.detail.content = f"标定 marker {len(residual)} 个；残差中位数 **{np.median(residual):.1f} mm**，最大 **{residual.max():.1f} mm**。请检查明显偏大的点及朝向。"

    def start_tracking(self, _):
        with self.lock:
            if self.mode != "calibrated" or not self.solver:
                return
            self.solver.reset()
            self.recorder = SequenceRecorder(self.args.model_type, self.args.gender, self.rig.betas,
                                             self.source.frame_rate)
            self.fit_count = 0
            self.last_result = None
            self.mode = "tracking"
            self.track_button.disabled = self.capture.disabled = self.subject.disabled = True
            self.stop_button.disabled = False
            self.message = "实时拟合中。红点为观测，绿点为拟合。"
            self._receipt("tracking")
            self.center_view(None)

    def stop_tracking(self, _):
        with self.lock:
            if self.mode != "tracking":
                return
            self.mode = "calibrated"
            name = "solved_" + uuid.uuid4().hex[:8] + ".npz"
            n = self.recorder.save(self.run / name) if self.recorder.poses else 0
            self.recorder = None
            self.message = f"已停止并保存 {n} 个输出帧。"
            self._receipt("saved", last_solved_file=name if n else None, solved_frames=n)
            self.track_button.disabled = self.capture.disabled = self.subject.disabled = False
            self.stop_button.disabled = True
            self.fitted.visible = False
            if self.mesh:
                self.mesh.visible = False

    def loop(self):
        seq = -1
        ui_time = draw_time = 0.
        try:
            while True:
                now = time.perf_counter()
                new_seq, packet = self.source.snapshot()
                age = now - packet["arrival_time"] if packet else float("inf")
                with self.lock:
                    if self.mode == "countdown" and now >= self.countdown_end:
                        self.sample_start = now
                        self.mode = "sampling"
                    fr = subject_frame(packet, self.selected) if packet and self.selected else None
                    if new_seq != seq:
                        seq = new_seq
                        if fr and age < 1:
                            if self.mode == "sampling":
                                self.samples.append(fr)
                            elif self.mode == "tracking":
                                pos, valid = self.mapper(fr)
                                self.last_result = self.solver.step(pos, valid, frame_no=fr["frame"])
                                res = self.last_result
                                self.recorder.add(fr["frame"], res.pose_aa, res.trans, res.rmse)
                                self.fit_count += 1
                    if self.mode == "sampling" and now - self.sample_start >= self.duration.value:
                        self.finish_sampling(None)
                    if now - draw_time > 1 / 20:
                        draw_time = now
                        self.cloud.visible = bool(fr and age < 1 and fr["valid"].any())
                        if self.cloud.visible:
                            self.cloud.points = fr["pos"][fr["valid"]].astype(np.float32)
                        res = self.last_result
                        visible = self.mode == "tracking" and age < 1 and res is not None
                        self.fitted.visible = visible
                        if visible:
                            self.fitted.points = self.rig.forward(res.R, res.trans).astype(np.float32)
                            verts = self.model.vertices(res.R, res.trans, self.rig.v_rest, self.rig.J_rest).astype(np.float32)
                            if self.mesh is None:
                                self.mesh = self.server.scene.add_mesh_simple("/body", vertices=verts,
                                    faces=self.model.faces.astype(np.uint32), color=(135, 162, 198), opacity=0.65)
                            else:
                                self.mesh.vertices = verts
                        if self.mesh:
                            self.mesh.visible = visible and self.show_mesh.value
                    if now - ui_time > 0.25:
                        ui_time = now
                        subjects = tuple(s["name"] for s in packet["subjects"] if len(s["markers"]) >= 15) if packet else ()
                        options = subjects or ("等待被试",)
                        # Retain an explicitly selected person even when they temporarily leave the stream.
                        if self.selected and self.selected not in options:
                            options = (self.selected,) + tuple(x for x in options if x != "等待被试")
                        if tuple(self.subject.options) != options:
                            self.subject.options = options
                        if not self.selected and subjects:
                            self.selected = self.subject.value
                        phase = self.message
                        if self.mode == "countdown":
                            phase = f"准备 T-pose · {max(0, self.countdown_end-now):.0f} 秒"
                        elif self.mode == "sampling":
                            elapsed = now - self.sample_start
                            phase = f"{'保持 T-pose' if elapsed < 3 else '开始活动范围动作'} · {elapsed:.1f}/{self.duration.value} 秒"
                        visible_count = int(fr["valid"].sum()) if fr and age < 1 else 0
                        connection = f"{self.source.frame_rate:.0f} Hz · 数据年龄 {age*1000:.0f} ms" if age < 1 else "等待新数据 · " + self.source.error
                        stats = ""
                        if self.mode == "tracking" and self.last_result:
                            r = self.last_result
                            stats = f"\n\n求解 **{r.ms:.2f} ms** · marker RMSE **{r.rmse*1000:.1f} mm** · {r.status} · 已输出 {self.fit_count} 帧"
                        self.status.content = f"**{connection}**\n\n可见 marker **{visible_count}** · {self.selected or '未选择被试'}\n\n{phase}{stats}"
                        receipt = dict(mode=self.mode, subject=self.selected, source=self.args.source,
                                       source_fps=self.source.frame_rate, frame=packet["frame"] if packet else None,
                                       age_ms=round(age*1000, 1) if np.isfinite(age) else None,
                                       subjects=list(subjects), visible_markers=visible_count, message=phase,
                                       solved_frames=self.fit_count, raw_capture_modified=False)
                        target = self.root / "status.json"
                        temp = target.with_suffix(".tmp")
                        temp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
                        temp.replace(target)
                time.sleep(0.001)
        finally:
            self.stop_tracking(None)
            self.finish_sampling(None)
            self.source.close()
            self.server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Shogun full subject DataStream, normally host:801 or host:8802")
    parser.add_argument("--native-reader", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-type", default="smplx", choices=("smpl", "smplh", "smplx"))
    parser.add_argument("--gender", default="neutral")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8792)
    parser.add_argument("--output", default="runtime")
    FieldApp(parser.parse_args()).loop()


if __name__ == "__main__":
    main()
