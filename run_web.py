"""Browser controls for a separate Vicon2Smpl field test; never controls Shogun capture."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from vsmpl.body_model import BodyModel, MarkerRig
from vsmpl.native_source import NativeSource, subject_frame
from vsmpl.outputs import SequenceRecorder
from vsmpl.solver import RealtimeSolver
from vsmpl.sources import NameMapper, save_recording
from vsmpl.calibration_input import prepare_calibration_input
from vsmpl.hand_pca import HandPCA
from vsmpl.pca_solver import PCASolver


class FieldApp:
    def __init__(self, args):
        import viser
        self.args = args
        self.root = Path(args.output)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.mode, self.message = "preview", "选择被试后开始标定；尚未生成任何拟合结果。"
        self.run = None
        self.samples = []
        self.recording = None
        self.calibration_path = None
        self.calibration_scope = None
        self.solver = self.rig = self.recorder = None
        self.selected = None
        self.last_result = None
        self.last_observation = None
        self.finger_marker_mask = None
        self.view_mode = "body"
        self.fit_count = 0
        self.model = BodyModel.load(args.model_path, args.model_type, args.gender)
        self.body_regions = None
        if getattr(args, "body_regions", None):
            regions = json.loads(Path(args.body_regions).read_text())
            count = len(regions)
            dummy = dict(names=list(regions), pos=np.empty((0, count, 3)), valid=np.empty((0, count), bool))
            _, self.body_regions = prepare_calibration_input(dummy, regions, len(self.model.v_template))
        self.fit_hands = bool(getattr(args, "fit_hands", False))
        self.hand_prior_source = self.hand_prior_sha256 = None
        self.hand_pca_components = getattr(args, "hand_components", 12)
        if self.fit_hands:
            if args.model_type != "smplx" or self.model.num_joints != 55:
                raise ValueError("Full body/hand fitting requires native SMPL-X55")
            if self.body_regions is None or len(self.body_regions) != 52 or not getattr(args, "hand_prior", None):
                raise ValueError("--fit-hands requires the validated 52-label --body-regions and --hand-prior")
            prior_path = Path(args.hand_prior).resolve()
            HandPCA.load(prior_path, n_components=self.hand_pca_components)
            self.hand_prior_source = str(prior_path)
            self.hand_prior_sha256 = hashlib.sha256(prior_path.read_bytes()).hexdigest()
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
            self.pose_kind = self.server.gui.add_dropdown("开头静止姿态", options=("A-pose（手臂斜下）", "T-pose（手臂平举）"))
            self.duration = self.server.gui.add_slider("采样时长 / 秒", min=15, max=90, step=5, initial_value=45)
            self.pose_hint = self.server.gui.add_markdown("")
            self.pose_kind.on_update(lambda _: self.update_pose_hint())
            self.update_pose_hint()
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
            if self.fit_hands:
                self.server.gui.add_button("左手近景").on_click(lambda _: self.show_hand("L"))
                self.server.gui.add_button("右手近景").on_click(lambda _: self.show_hand("R"))
            self.show_mesh = self.server.gui.add_checkbox("显示人体网格", initial_value=True)
            self.show_mesh.on_update(lambda _: self._mesh_visibility())
        self.base_detail = ("本次拟合 **SMPL-X 身体＋双手 72 点**：52 个身体/手腕点和 20 个手指点。手指使用手形先验约束，需检查张手、握拳与捏合。" if self.fit_hands else
                            f"本次仅拟合 **{len(self.body_regions)} 个身体/手腕点**，手指点不参与。" if self.body_regions else
                            "红点：观测 marker；绿点：拟合 marker。手指质量取决于手部贴点布局。")
        self.detail = self.server.gui.add_markdown(self.base_detail)
        self.server.gui.add_markdown("本页仅采样标定 marker 与保存派生拟合；正式录制仍使用原录制页面。")
        self.server.on_client_connect(lambda event: self.center_view(event))
        if getattr(args, "calib", None):
            if not args.subject:
                raise ValueError("--calib requires --subject to keep the actor association explicit")
            path = Path(args.calib).resolve()
            with np.load(path, allow_pickle=False) as d:
                if "source_subject" not in d or str(d["source_subject"]) != args.subject:
                    raise ValueError("Saved calibration subject does not match --subject")
                pose = str(d["static_pose"]) if "static_pose" in d else "t"
            self.selected, self.run, self.calibration_path = args.subject, path.parent, path
            self.subject.options = (self.selected,)
            self.subject.value = self.selected
            self.pose_kind.value = "A-pose（手臂斜下）" if pose == "a" else "T-pose（手臂平举）"
            self._load_calibration(path)
            self.mode = "calibrated"
            self.track_button.disabled = False
            self.message = f"已加载 {pose.upper()}-pose 标定候选，等待真人叠加检查。"

    def update_pose_hint(self):
        name = "A-pose" if self.pose_kind.value.startswith("A") else "T-pose"
        self.pose_hint.content = f"点击后有 **5 秒准备**。先保持 **{name} 3 秒**，再做活动范围动作或自由活动。"
        if self.fit_hands:
            self.pose_hint.content += "请同时包含张手、握拳、逐指弯曲和捏合。"

    def _mesh_visibility(self):
        if self.mesh:
            self.mesh.visible = self.show_mesh.value and self.mode == "tracking"

    def center_view(self, _):
        self.view_mode = "body"
        _, packet = self.source.snapshot()
        fr = subject_frame(packet, self.selected) if packet and self.selected else None
        points = fr["pos"][fr["valid"]] if fr else np.empty((0, 3))
        target = np.median(points, axis=0) if len(points) else np.array([0., 0.8, 0.])
        for client in self.server.get_clients().values():
            client.camera.look_at = target
            client.camera.position = target + [2.8, 1.2, 3.2]
            client.camera.up_direction = (0., 1., 0.)

    def show_hand(self, side):
        if self.mode != "tracking" or self.last_result is None or self.model.num_joints != 55:
            return
        self.view_mode = side
        self.update_hand_camera()

    def update_hand_camera(self):
        if self.view_mode not in ("L", "R") or self.last_result is None:
            return
        result = self.last_result
        rw, joints = self.model.fk(result.R, self.rig.J_rest)
        wrist = 20 if self.view_mode == "L" else 21
        start = 25 if self.view_mode == "L" else 40
        target = (joints[start:start+15].mean(0) + joints[wrist]) / 2 + result.trans
        rotation = rw[wrist]
        up = rotation @ np.array([1. if self.view_mode == "L" else -1., 0., 0.])
        position = target + rotation @ np.array([0., .32, .13])
        for client in self.server.get_clients().values():
            client.camera.look_at = target
            client.camera.position = position
            client.camera.up_direction = up

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
            self.calibration_scope = None
            self.last_result = None
            self.last_observation = None
            self.finger_marker_mask = None
            self.detail.content = self.base_detail
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
            self.sample_static_pose = "a" if self.pose_kind.value.startswith("A") else "t"
            self.sample_body_regions = None if self.body_regions is None else json.loads(json.dumps(self.body_regions))
            self.sample_fit_hands = self.fit_hands
            self.sample_hand_prior_source = self.hand_prior_source
            self.sample_hand_prior_sha256 = self.hand_prior_sha256
            self.countdown_end = time.perf_counter() + 5
            self.samples = []
            self.run = self.root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
            self.run.mkdir()
            self.recording = self.calibration_path = None
            self.calibration_scope = None
            self.solver = self.rig = None
            self.last_result = self.last_observation = None
            self.finger_marker_mask = None
            self.track_button.disabled = self.fit_button.disabled = self.capture.disabled = True
            self.subject.disabled = self.duration.disabled = self.pose_kind.disabled = True
            self.finish.disabled = False
            self.fitted.visible = False
            if self.mesh:
                self.mesh.visible = False

    def finish_sampling(self, _):
        with self.lock:
            if self.mode not in ("sampling", "countdown"):
                return
            self.mode = "preview"
            self.capture.disabled = self.subject.disabled = self.duration.disabled = self.pose_kind.disabled = False
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
            if self.sample_body_regions is not None:
                # Freeze only anatomical regions. The new subject's shape and attachments are fitted afresh.
                (self.run / "marker_regions.json").write_text(json.dumps(self.sample_body_regions))
            self._receipt("sampled", source_fps=self.source.frame_rate, sampled_fps=fps,
                          samples=len(times), tpose_window=list(self.tpose_window), duration_seconds=float(times[-1]),
                          static_pose=self.sample_static_pose,
                          marker_scope="body_hands" if self.sample_fit_hands else
                                       "body_wrist_only" if self.sample_body_regions is not None else "all_markers",
                          hand_prior_source=self.sample_hand_prior_source if self.sample_fit_hands else None,
                          hand_prior_sha256=self.sample_hand_prior_sha256 if self.sample_fit_hands else None)
            self.fit_button.disabled = times[-1] < 10 or self.tpose_window[1] - self.tpose_window[0] < 5
            self.message = f"已保留 {len(times)} 帧 / {times[-1]:.1f} 秒。" + ("可以计算标定。" if not self.fit_button.disabled else "采样太短，请重新采样。")

    def _receipt(self, state, **extra):
        p = self.run / "session.json"
        old = json.loads(p.read_text()) if p.exists() else {}
        if getattr(self, "calibration_scope", None) is not None:
            extra.setdefault("marker_scope", self.calibration_scope)
        old.update(state=state, subject=self.selected, model_type=self.args.model_type,
                   source=self.args.source, coordinate_system="metres, Y-up; Vicon (x,z,-y)",
                   raw_capture_modified=False, **extra)
        p.write_text(json.dumps(old, ensure_ascii=False, indent=2))

    def start_calibration(self, _):
        with self.lock:
            if self.mode != "preview" or self.fit_button.disabled:
                return
            self.mode = "calibrating"
            self.fit_button.disabled = self.capture.disabled = self.subject.disabled = self.pose_kind.disabled = True
            self.message = "正在计算被试体型与 marker 布局；进度保存在 calibration.log。"
            # Run the existing CLI in a worker process: its CPU work cannot block the browser.
            metadata = self.run / "calibration_request.json"
            metadata.write_text(json.dumps({"tpose": list(self.tpose_window), "static_pose": self.sample_static_pose,
                "source_subject": self.selected,
                "body_regions_file": "marker_regions.json" if self.sample_body_regions is not None else None,
                "fit_hands": self.sample_fit_hands,
                "hand_prior_source": self.sample_hand_prior_source if self.sample_fit_hands else None,
                "hand_prior_sha256": self.sample_hand_prior_sha256 if self.sample_fit_hands else None,
                "hand_pca_components": self.hand_pca_components if self.sample_fit_hands else None}))
            threading.Thread(target=self._calibrate, daemon=True).start()

    def _calibrate(self):
        import subprocess
        import sys
        output = self.run / "subject_calib.npz"
        code = """import json,numpy as np,sys,hashlib
from pathlib import Path
from vsmpl.body_model import BodyModel
from vsmpl.calibration import calibrate
from vsmpl.calibration_input import prepare_calibration_input
from vsmpl.sources import load_recording
model=BodyModel.load(sys.argv[1],sys.argv[2],sys.argv[3])
rec=load_recording(sys.argv[4])
request=json.loads(Path(sys.argv[5]).read_text())
assert request['source_subject']==sys.argv[8], 'Calibration subject association changed'
region_file=Path(sys.argv[5]).parent/request['body_regions_file'] if request['body_regions_file'] else None
regions=json.loads(region_file.read_text()) if region_file else None
raw_rec=rec
all_names=list(raw_rec['names'])
rec,regions=prepare_calibration_input(raw_rec,regions,len(model.v_template))
fit_hands=bool(request.get('fit_hands',False))
prior=None
if fit_hands:
    if sys.argv[2]!='smplx' or model.num_joints!=55 or regions is None or len(regions)!=52:
        raise ValueError('Full fitting requires SMPL-X55 and 52 body marker regions')
    from vsmpl.hand_pca import HandPCA
    prior_path=Path(request['hand_prior_source']).resolve()
    prior_sha=hashlib.sha256(prior_path.read_bytes()).hexdigest()
    if prior_sha!=request['hand_prior_sha256']:
        raise ValueError('Hand prior changed since the sampling request was frozen')
    prior=HandPCA.load(prior_path,n_components=int(request.get('hand_pca_components',12)))
print('[field] fitting the current subject body',flush=True)
body_result=calibrate(model,rec,tpose=tuple(request['tpose']),static_pose=request['static_pose'],marker_regions=regions)
result=dict(body_result)
scope='body_wrist_only' if regions is not None else 'all_markers'
if fit_hands:
    from vsmpl.full_calibration import calibrate_hands
    body_saved=dict(body_result)
    body_saved.update(model_type=sys.argv[2],gender=sys.argv[3],num_betas=10,
                      source_subject=request['source_subject'],marker_scope='body_wrist_only')
    np.savez(Path(sys.argv[6]).with_name('body_calib.npz'),**body_saved)
    clock_path=Path(sys.argv[5]).parent/'sampling_clock.npz'
    with np.load(clock_path,allow_pickle=False) as clock:
        frames=clock['source_frame_numbers'].copy()
        times=clock['relative_host_arrival_seconds'].copy()
    if len(frames)!=len(raw_rec['pos']) or len(times)!=len(raw_rec['pos']) or not np.isfinite(times).all():
        raise ValueError('Sampling clock does not match the full source recording')
    print('[field] fitting native SMPL-X hands on the complete source markers',flush=True)
    result,report=calibrate_hands(model,raw_rec,body_result,prior,regions,frame_numbers=frames,times=times)
    result=dict(result)
    if str(result.get('marker_scope',''))!='body_hands':
        raise ValueError('Full hand fitting returned an unexpected scope')
    def json_value(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    (Path(sys.argv[6]).parent/'hand_calibration_report.json').write_text(
        json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False,default=json_value))
    result.update(hand_prior_source=str(prior_path),hand_prior_sha256=prior_sha)
    scope='body_hands'
result.update(model_type=sys.argv[2],gender=sys.argv[3],num_betas=10,source_subject=request['source_subject'],
marker_scope=scope,marker_regions_sha256=hashlib.sha256(region_file.read_bytes()).hexdigest() if region_file else '',
excluded_marker_names=np.array([n for n in all_names if n not in set(map(str,result['marker_names']))],dtype=str))
np.savez(sys.argv[6],**result)
"""
        try:
            with (self.run / "calibration.log").open("w") as log:
                subprocess.run([sys.executable, "-u", "-c", code, self.args.model_path, self.args.model_type,
                                self.args.gender, str(self.run / "calibration_markers.npz"),
                                str(self.run / "calibration_request.json"), str(output), self.sample_static_pose, self.selected],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            with self.lock:
                self.calibration_path = output
                self._load_calibration(output)
                self.mode = "calibrated"
                self.message = "标定计算完成，尚待真人叠加检查。请先检查逐点残差和朝向，再开始测试拟合。"
                self._receipt("calibrated", calibration_file=output.name)
                self.track_button.disabled = False
        except Exception as e:
            with self.lock:
                self.mode = "preview"
                self.message = f"标定失败：{e}。采样与日志已保留。"
                self._receipt("calibration_failed", error=str(e))
                self.fit_button.disabled = False
        finally:
            self.capture.disabled = self.subject.disabled = self.pose_kind.disabled = False

    def _load_calibration(self, path):
        with np.load(path, allow_pickle=False) as calib:
            rig = MarkerRig.from_calibration(self.model, calib)
            residual = calib["per_marker_rms"] * 1000
            scope = str(calib["marker_scope"]) if "marker_scope" in calib else "all_markers"
            if scope == "body_hands":
                configured = getattr(self.args, "hand_prior", None)
                prior_source = configured or (str(calib["hand_prior_source"]) if "hand_prior_source" in calib else None)
                if not prior_source or "hand_prior_sha256" not in calib:
                    raise ValueError("Full calibration is missing its hand prior provenance")
                prior_path = Path(prior_source).resolve()
                if hashlib.sha256(prior_path.read_bytes()).hexdigest() != str(calib["hand_prior_sha256"]):
                    raise ValueError("Hand prior SHA256 does not match the saved calibration")
                prior = HandPCA.load(prior_path, n_components=int(calib["hand_pca_components"]))
                solver = PCASolver(rig, prior, initial_latent=calib["hand_pca_initial_latent"],
                                   hand_prior_weight=float(calib["hand_prior_weight"]))
                fingers = [str(name) for name in calib["finger_marker_names"]]
                if rig.num_markers != 72 or len(set(fingers)) != 20 or not set(fingers).issubset(rig.names):
                    raise ValueError("Full calibration does not contain 72 markers and 20 finger labels")
                finger_mask = np.isin(rig.names, fingers)
            else:
                solver = RealtimeSolver(rig)
                finger_mask = None
        self.rig, self.solver = rig, solver
        self.calibration_scope = scope
        self.finger_marker_mask = finger_mask
        self.last_observation = None
        self.mapper = NameMapper(self.rig.names)
        self.detail.content = f"标定 marker {len(residual)} 个；残差中位数 **{np.median(residual):.1f} mm**，最大 **{residual.max():.1f} mm**。请检查明显偏大的点及朝向。"
        high = [name for name, error in zip(self.rig.names, residual) if error > 30]
        if high:
            self.detail.content += "\n\n**候选存在较大残差，尚未验收：** " + "、".join(high)
        if scope == "body_wrist_only":
            self.detail.content += "\n\n**此候选仅标定身体与手腕，手指 marker 未参与；手指姿态不代表测量结果。**"
        elif scope == "body_hands":
            self.detail.content += (f"\n\n**SMPL-X 身体＋双手 72 点**。身体点残差中位数 "
                                    f"**{np.median(residual[~finger_mask]):.1f} mm**；手指点 "
                                    f"**{np.median(residual[finger_mask]):.1f} mm**。手指使用学习手形先验约束，尚待真人近景检查。")

    def _group_marker_rms(self):
        if self.finger_marker_mask is None or self.last_result is None or self.last_observation is None:
            return None
        obs, valid = self.last_observation
        pred = self.rig.forward(self.last_result.R, self.last_result.trans)
        finite = valid & np.isfinite(obs).all(-1) & np.isfinite(pred).all(-1)
        error = np.linalg.norm(pred - obs, axis=1)
        def rms(mask):
            values = error[mask & finite]
            return float(np.sqrt(np.mean(values ** 2)) * 1000) if len(values) else None
        return dict(body_mm=rms(~self.finger_marker_mask), finger_mm=rms(self.finger_marker_mask))

    def start_tracking(self, _):
        with self.lock:
            if self.mode != "calibrated" or not self.solver:
                return
            _, packet = self.source.snapshot()
            fr = subject_frame(packet, self.selected) if packet else None
            mapped_valid = self.mapper(fr)[1] if fr else np.zeros(0, bool)
            if not fr or time.perf_counter() - packet["arrival_time"] > 1 or mapped_valid.sum() < self.solver.cfg.min_markers:
                self.message = "当前被试的可见点不足；请就位并恢复 marker 后再开始实时拟合。"
                return
            self.solver.reset()
            self.recorder = SequenceRecorder(self.args.model_type, self.args.gender, self.rig.betas,
                                             self.source.frame_rate)
            self.fit_count = 0
            self.last_result = None
            self.last_observation = None
            self.mode = "tracking"
            self.track_button.disabled = self.capture.disabled = self.subject.disabled = self.pose_kind.disabled = True
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
            self.track_button.disabled = self.capture.disabled = self.subject.disabled = self.pose_kind.disabled = False
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
                                self.last_observation = (pos.copy(), valid.copy())
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
                        if visible and self.view_mode != "body":
                            self.update_hand_camera()
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
                            phase = f"准备 {self.sample_static_pose.upper()}-pose · {max(0, self.countdown_end-now):.0f} 秒"
                        elif self.mode == "sampling":
                            elapsed = now - self.sample_start
                            phase = f"{'保持 '+self.sample_static_pose.upper()+'-pose' if elapsed < 3 else '开始活动范围动作'} · {elapsed:.1f}/{self.duration.value} 秒"
                        visible_count = int(fr["valid"].sum()) if fr and age < 1 else 0
                        connection = f"{self.source.frame_rate:.0f} Hz · 数据年龄 {age*1000:.0f} ms" if age < 1 else "等待新数据 · " + self.source.error
                        stats = ""
                        groups = None
                        if self.mode == "tracking" and self.last_result:
                            r = self.last_result
                            stats = f"\n\n求解 **{r.ms:.2f} ms** · marker RMSE **{r.rmse*1000:.1f} mm** · {r.status} · 已输出 {self.fit_count} 帧"
                            groups = self._group_marker_rms()
                            if groups is not None:
                                def display(value):
                                    return "—" if value is None else f"{value:.1f} mm"
                                stats += f"\n\n可见点 RMS · 身体 **{display(groups['body_mm'])}** · 手指 **{display(groups['finger_mm'])}**"
                        self.status.content = f"**{connection}**\n\n可见 marker **{visible_count}** · {self.selected or '未选择被试'}\n\n{phase}{stats}"
                        receipt = dict(mode=self.mode, subject=self.selected, source=self.args.source,
                                       source_fps=self.source.frame_rate, frame=packet["frame"] if packet else None,
                                       age_ms=round(age*1000, 1) if np.isfinite(age) else None,
                                       subjects=list(subjects), visible_markers=visible_count, message=phase,
                                       solved_frames=self.fit_count, raw_capture_modified=False,
                                       marker_scope=self.calibration_scope,
                                       body_marker_rms_mm=groups["body_mm"] if groups else None,
                                       finger_marker_rms_mm=groups["finger_mm"] if groups else None,
                                       solver_status=self.last_result.status if self.mode == "tracking" and self.last_result else None)
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
    parser.add_argument("--calib", help="load a separately reviewed calibration candidate without resampling")
    parser.add_argument("--subject", help="explicit subject associated with --calib")
    parser.add_argument("--body-regions", help="validated body/wrist region JSON; freeze per sample and exclude other markers from fitting")
    parser.add_argument("--fit-hands", action="store_true", help="fit body and both articulated hands using the complete source markers")
    parser.add_argument("--hand-prior", help="licensed native SMPL-X hand PCA prior for --fit-hands/full saved calibration")
    parser.add_argument("--hand-components", type=int, default=12, help="hand PCA coefficients per side for new full calibrations")
    FieldApp(parser.parse_args()).loop()


if __name__ == "__main__":
    main()
