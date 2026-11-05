"""Browser controls for a separate Vicon2Smpl field test; never controls Shogun capture."""
import argparse
import copy
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
from vsmpl.solver import RealtimeSolver, SolverConfig
from vsmpl.sources import NameMapper, save_recording
from vsmpl.calibration_input import prepare_calibration_input
from vsmpl.hand_pca import HandPCA
from vsmpl.pca_solver import PCASolver
from vsmpl.calibration import model_static_reference
from vsmpl.rotations import exp_so3
from vsmpl.model_profiles import load_profiles, apply_profile_args, profile_request, profile_signature, verify_profile_files


IDLE_MODES = frozenset(("preview", "calibrated"))


def soma_finger_joints(model):
    """SOMA user-joint IDs below the named wrists; never reuse SMPL-X indices."""
    names = getattr(model, "joint_names", None)
    if names is None or len(names) != model.num_joints:
        raise ValueError("SOMA body/wrist fitting requires its named public skeleton")
    fingers = [j for j, name in enumerate(names)
               if any(str(name).startswith(wrist) and str(name) != wrist
                      for wrist in ("LeftHand", "RightHand"))]
    if not fingers:
        raise ValueError("SOMA body/wrist fitting could not identify the finger joints")
    return fingers


class FieldApp:
    def __init__(self, args):
        import viser
        self.profiles_enabled = bool(getattr(args, "model_profiles", None))
        self.active_profile, self.profiles = load_profiles(getattr(args, "model_profiles", None), args)
        self.base_args = copy.deepcopy(args)
        self.args = apply_profile_args(args, self.profiles[self.active_profile]) if self.profiles_enabled else args
        args = self.args
        self._profile_calibrations = {}
        self._loaded_models = {}
        self._allow_profile_startup_calibrations = True
        self.sample_model_request = self._calibration_job = None
        self.calibration_compatibility = None
        self.root = Path(args.output)
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_root = Path(getattr(args, "data_output", None) or args.output)
        self.data_root.mkdir(parents=True, exist_ok=True)
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
        self.__dict__.update(self._model_state(args, self.profiles[self.active_profile]))
        self._loaded_models[self.active_profile] = self.model
        self.source = NativeSource(args.native_reader, args.source)
        self.server = viser.ViserServer(host=args.bind, port=args.port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="fixed", control_width="medium",
                                        show_share_button=False, show_logo=False)
        self.server.scene.set_up_direction("+y")
        self.server.scene.add_grid("/floor", plane="xz", width=10, height=10)
        self.cloud = self.server.scene.add_point_cloud("/observed", points=np.zeros((1, 3), np.float32),
                                                       colors=(239, 91, 86), point_size=0.008, point_shape="circle", visible=False)
        self.fitted = self.server.scene.add_point_cloud("/fitted", points=np.zeros((1, 3), np.float32),
                                                        colors=(70, 200, 160), point_size=0.006, point_shape="circle", visible=False)
        self.mesh = self.preview_mesh = None
        self.server.gui.add_markdown(("# Vicon → 人体模型" if self.profiles_enabled else "# Vicon → SMPL")
                                     + "\n全身实时拟合 · 独立现场测试")
        self.status = self.server.gui.add_markdown("正在连接动捕数据…")
        self.model_selector = None
        if self.profiles_enabled:
            self.model_selector = self.server.gui.add_dropdown("人体模型", options=tuple(p["label"] for p in self.profiles.values()),
                initial_value=self.profiles[self.active_profile]["label"])
            self.model_selector.on_update(self.change_model)
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
            self.hand_buttons = {}
            if self.fit_hands or self.profiles_enabled:
                for side, label in (("L", "左手近景"), ("R", "右手近景")):
                    self.hand_buttons[side] = self.server.gui.add_button(label)
                    self.hand_buttons[side].on_click(lambda _, side=side: self.show_hand(side))
            self.head_button = None
            if self.head_view or self.profiles_enabled:
                self.head_button = self.server.gui.add_button("头部近景")
                self.head_button.on_click(lambda _: self.show_head())
            self.show_mesh = self.server.gui.add_checkbox("显示人体网格", initial_value=True)
            self.show_mesh.on_update(lambda _: self._mesh_visibility())
        self.base_detail = self._model_description()
        self.detail = self.server.gui.add_markdown(self.base_detail)
        self.preview_detail = self.server.gui.add_markdown("") if self.profiles_enabled else None
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
            self._remember_calibration()
        self._sync_model_controls()
        self._show_model_preview()

    def _model_state(self, args, profile):
        verify_profile_files(profile)
        model = self._loaded_models.get(profile["id"])
        if model is None:
            soma_kw = dict(soma_lod=getattr(args, "soma_lod", "mid"),
                           soma_identity=getattr(args, "soma_identity", "soma")) if args.model_type == "soma" else {}
            if self.profiles_enabled or hasattr(args, "num_betas"):
                soma_kw["num_betas"] = getattr(args, "num_betas", 10)
            model = BodyModel.load(args.model_path, args.model_type, args.gender, **soma_kw)
        regions = None
        if getattr(args, "body_regions", None):
            data = Path(args.body_regions).read_bytes()
            if profile.get("body_regions_sha256") and hashlib.sha256(data).hexdigest() != profile["body_regions_sha256"]:
                raise ValueError("Marker regions changed since the model profile was loaded")
            raw = json.loads(data)
            dummy = dict(names=list(raw), pos=np.empty((0, len(raw), 3)), valid=np.empty((0, len(raw)), bool))
            _, regions = prepare_calibration_input(dummy, raw, len(model.v_template))
        if self.profiles_enabled and args.model_type == "soma" and regions is None:
            raise ValueError("SOMA body/wrist profiles require an explicit body marker region whitelist")
        hands = bool(getattr(args, "fit_hands", False))
        prior_source = prior_sha = None
        components = getattr(args, "hand_components", 12)
        if hands:
            if args.model_type != "smplx" or model.num_joints != 55:
                raise ValueError("Full body/hand fitting requires native SMPL-X55")
            if regions is None or len(regions) != 52 or not getattr(args, "hand_prior", None):
                raise ValueError("--fit-hands requires the validated 52-label --body-regions and --hand-prior")
            prior_path = Path(args.hand_prior).resolve()
            prior_sha = hashlib.sha256(prior_path.read_bytes()).hexdigest()
            if profile.get("hand_prior_sha256") and prior_sha != profile["hand_prior_sha256"]:
                raise ValueError("Hand prior changed since the model profile was loaded")
            HandPCA.load(prior_path, n_components=components)
            prior_source = str(prior_path)
        return dict(args=args, model=model, body_regions=regions, fit_hands=hands,
                    hand_prior_source=prior_source, hand_prior_sha256=prior_sha, hand_pca_components=components,
                    head_view=model.num_joints == 55 or getattr(model, "model_type", None) == "soma")

    def _model_description(self):
        if getattr(self.model, "model_type", self.args.model_type) == "soma":
            return "**SOMA 身体＋手腕**：原生纯 LBS；不拟合手指，不使用 SMPL-X 手指 PCA 先验。"
        if self.fit_hands:
            return "本次拟合 **SMPL-X 身体＋双手 72 点**：52 个身体/手腕点和 20 个手指点。手指使用手形先验约束，需检查张手、握拳与捏合。"
        return (f"本次仅拟合 **{len(self.body_regions)} 个身体/手腕点**，手指点不参与。" if self.body_regions else
                "红点：观测 marker；绿点：拟合 marker。手指质量取决于手部贴点布局。")

    def _sync_model_controls(self):
        selector = getattr(self, "model_selector", None)
        if selector is not None:
            selector.disabled = self.mode not in IDLE_MODES
        for button in getattr(self, "hand_buttons", {}).values():
            button.disabled = not self.fit_hands
        if getattr(self, "head_button", None) is not None:
            self.head_button.disabled = not self.head_view

    def _model_receipt(self):
        profile = getattr(self, "profiles", {}).get(getattr(self, "active_profile", None))
        return dict(active_model=getattr(self.model, "model_type", self.args.model_type),
                    model_profile_id=profile["id"] if profile else None,
                    model_profile_label=profile["label"] if profile else None,
                    profile_signature=profile_signature(profile) if profile else None,
                    model_capability="body_hands_pca" if self.fit_hands else
                                     "body_wrist_lbs" if getattr(self.model, "model_type", self.args.model_type) == "soma" else "body_wrist")

    def _remember_calibration(self):
        if (getattr(self, "profiles_enabled", False) and self.mode == "calibrated"
                and self.rig is not None and self.solver is not None and self.selected and self.calibration_path
                and Path(self.calibration_path).is_file()):
            self._profile_calibrations[(self.active_profile, self.selected)] = Path(self.calibration_path)

    def _remove_meshes(self):
        for name in ("mesh", "preview_mesh"):
            handle = getattr(self, name, None)
            if handle is not None:
                handle.remove()
            setattr(self, name, None)

    def _preview_geometry(self):
        if self.rig is not None:
            rest, joints = self.rig.v_rest, self.rig.J_rest
        else:
            betas = np.zeros(self.model.num_betas)
            self.model.prepare_for(betas)
            rest, joints = self.model.shaped(betas)
        rotations = exp_so3(model_static_reference(self.model, "a"))
        vertices = self.model.vertices(rotations, np.zeros(3), rest, joints).copy()
        if not np.isfinite(vertices).all() or not len(vertices):
            raise ValueError("Model template contains invalid vertices")
        vertices[:, 1] -= vertices[:, 1].min()
        return vertices.astype(np.float32)

    def _show_model_preview(self, vertices=None):
        if not getattr(self, "profiles_enabled", False):
            return
        if self.mode not in IDLE_MODES:
            if self.preview_mesh is not None:
                self.preview_mesh.visible = False
            self.preview_detail.content = ""
            return
        vertices = self._preview_geometry() if vertices is None else vertices
        if self.preview_mesh is None:
            self.preview_mesh = self.server.scene.add_mesh_simple("/model_preview", vertices=vertices,
                faces=self.model.faces.astype(np.uint32), color=(160, 173, 190), opacity=1.)
        else:
            self.preview_mesh.vertices = vertices
        self.preview_mesh.visible = self.show_mesh.value
        label = "未标定模型预览" if self.rig is None else "已标定体型的静态模型预览"
        self.preview_detail.content = f"**{label}**：静态 A-pose，脚底放在地面；**非真人实时拟合**。"

    def _clear_fit_state(self):
        self.run = self.recording = self.calibration_path = self.calibration_scope = None
        self.rig = self.solver = self.recorder = None
        self.last_result = self.last_observation = self.finger_marker_mask = None
        self.samples, self.fit_count = [], 0
        self.sample_model_request = self._calibration_job = self.calibration_compatibility = None
        self.mode, self.view_mode = "preview", "body"
        self.base_detail = self._model_description()
        self.detail.content = self.base_detail
        self.track_button.disabled = self.fit_button.disabled = self.stop_button.disabled = self.finish.disabled = True
        self.capture.disabled = self.subject.disabled = self.pose_kind.disabled = self.duration.disabled = False
        self.fitted.visible = False

    def change_model(self, _):
        if not self.profiles_enabled:
            return
        with self.lock:
            previous = self.active_profile
            current_label = self.profiles[previous]["label"]
            wanted = next((key for key, profile in self.profiles.items() if profile["label"] == self.model_selector.value), None)
            if wanted == previous:
                return
            if wanted is None or self.mode not in IDLE_MODES:
                self.model_selector.value = current_label
                self.message = "当前阶段不能切换模型；请先停止并保存，再切换。"
                return
            self._remember_calibration()
            old = self.__dict__.copy()
            old_detail, old_pose = self.detail.content, self.pose_kind.value
            old_hint = self.pose_hint.content
            old_controls = {name: getattr(self, name).disabled for name in
                ("capture", "finish", "fit_button", "track_button", "stop_button", "subject", "pose_kind", "duration")}
            self.mode = "loading"
            self.message = "正在加载所选人体模型…"
            self._sync_model_controls()
        try:
            profile = self.profiles[wanted]
            state = self._model_state(apply_profile_args(self.base_args, profile), profile)
            with self.lock:
                self.__dict__.update(state)
                self.active_profile = wanted
                self._clear_fit_state()
                checkpoint = self._profile_calibrations.get((wanted, self.selected))
                if (checkpoint is None and self._allow_profile_startup_calibrations
                        and profile.get("subject") == self.selected):
                    checkpoint = profile.get("calib")
                if checkpoint:
                    self._load_calibration(checkpoint)
                    self.calibration_path = Path(checkpoint).resolve()
                    self.run = self.calibration_path.parent
                    self.mode = "calibrated"
                    self.track_button.disabled = False
                vertices = self._preview_geometry()
                self._remove_meshes()
                self._loaded_models[wanted] = self.model
                self._show_model_preview(vertices)
                self.model_selector.value = profile["label"]
                self.message = ("已切换模型并重新加载本被试的独立标定。" if checkpoint else
                    "已切换模型；请为本模型重新采样标定。先前采样文件保留，计算入口已清空。")
                self.update_pose_hint()
                self._sync_model_controls()
                self.center_view(None)
        except Exception as exc:
            with self.lock:
                changed_mesh = self.preview_mesh is not old.get("preview_mesh")
                if changed_mesh:
                    self._remove_meshes()
                self.__dict__.update(old)
                if changed_mesh:
                    self.mesh = self.preview_mesh = None
                self.detail.content = old_detail
                self.pose_kind.value = old_pose
                self.pose_hint.content = old_hint
                self.model_selector.value = current_label
                self.message = f"模型加载失败，已保留原模型与标定：{exc}"
                for name, disabled in old_controls.items():
                    getattr(self, name).disabled = disabled
                if changed_mesh:
                    self._show_model_preview()
                self._sync_model_controls()

    def update_pose_hint(self):
        name = "A-pose" if self.pose_kind.value.startswith("A") else "T-pose"
        self.pose_hint.content = f"点击后有 **5 秒准备**。先保持 **{name} 3 秒**，再做活动范围动作或自由活动。"
        if self.fit_hands:
            self.pose_hint.content += "请同时包含张手、握拳、逐指弯曲和捏合。"

    def _mesh_visibility(self):
        if self.mesh:
            self.mesh.visible = self.show_mesh.value and self.mode == "tracking"
        if getattr(self, "preview_mesh", None) is not None:
            self.preview_mesh.visible = self.show_mesh.value and self.mode in IDLE_MODES

    def center_view(self, _):
        self.view_mode = "body"
        self.cloud.point_size, self.fitted.point_size = .008, .006
        _, packet = self.source.snapshot()
        fr = subject_frame(packet, self.selected) if packet and self.selected else None
        points = fr["pos"][fr["valid"]] if fr else np.empty((0, 3))
        target = np.median(points, axis=0) if len(points) else np.array([0., 0.8, 0.])
        for client in self.server.get_clients().values():
            client.camera.look_at = target
            client.camera.position = target + [2.8, 1.2, 3.2]
            client.camera.up_direction = (0., 1., 0.)

    def show_hand(self, side):
        if self.mode != "tracking" or self.last_result is None or self.model.num_joints != 55 or not self.fit_hands:
            return
        self.view_mode = side
        self.cloud.point_size, self.fitted.point_size = .003, .0025
        self.update_hand_camera()

    def update_hand_camera(self):
        if self.view_mode == "head":
            self.update_head_camera()
            return
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

    def show_head(self):
        if self.mode != "tracking" or self.last_result is None or not getattr(self, "head_view", False):
            return
        self.view_mode = "head"
        self.cloud.point_size, self.fitted.point_size = .003, .0025
        self.update_head_camera()

    def update_head_camera(self):
        result = self.last_result
        rw, joints = self.model.fk(result.R, self.rig.J_rest)
        hj, chest = getattr(self.model, "head_joint", 15), getattr(self.model, "torso_joints", (9,))[-1]
        head = self.rig.v_rest[self.model.weights.argmax(1) == hj].mean(0)
        target = joints[hj] + result.trans + rw[hj] @ (head - self.rig.J_rest[hj])
        # SMPL rest frames are world-aligned; SOMA's solver rest is its bind pose, whose chest frame is too.
        forward = rw[chest, :, 2].copy()
        forward[1] = 0
        forward /= max(np.linalg.norm(forward), 1e-9)
        for client in self.server.get_clients().values():
            client.camera.look_at = target
            client.camera.position = target + .5 * forward + np.array([0., .02, 0.])
            # A head-fixed camera would visually hide the very tilt being inspected.
            client.camera.up_direction = (0., 1., 0.)

    def change_subject(self, _):
        with self.lock:
            if self.subject.value == "等待被试" or self.subject.value == self.selected:
                return
            if self.mode not in IDLE_MODES:
                if self.selected:
                    self.subject.value = self.selected
                return
            if self.profiles_enabled:
                self._profile_calibrations.clear()
                self._loaded_models = {self.active_profile: self.model}
                self._allow_profile_startup_calibrations = False
            self.selected = self.subject.value
            self.solver = self.rig = None
            self.recording = None
            self.calibration_path = None
            self.calibration_scope = None
            self.last_result = None
            self.last_observation = None
            self.finger_marker_mask = None
            self.sample_model_request = self._calibration_job = self.calibration_compatibility = None
            self.detail.content = self.base_detail
            self.mode = "preview"
            self.track_button.disabled = self.fit_button.disabled = True
            self.fitted.visible = False
            if self.mesh:
                self.mesh.visible = False
            self.message = f"已选择 {self.selected}；请为此人采样标定。"
            self._remove_meshes()
            self._show_model_preview()
            self._sync_model_controls()
            self.center_view(None)

    def _freeze_model_request(self):
        profile = copy.deepcopy(self.profiles[self.active_profile])
        profile["num_betas"] = getattr(self.model, "num_betas", profile["num_betas"])
        profile["hand_components"] = self.hand_pca_components
        for field in ("body_regions", "hand_prior"):
            filename = profile.get(field)
            if filename:
                actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
                expected = profile.get(field + "_sha256")
                if expected and expected != actual:
                    raise ValueError(field + " changed since the model profile was loaded")
                profile[field + "_sha256"] = actual
        frozen = profile_request(profile)
        frozen.update(source_subject=self.selected,
                      body_regions_file="marker_regions.json" if self.body_regions is not None else None,
                      body_regions_source_sha256=profile.get("body_regions_sha256"),
                      hand_prior_source=self.hand_prior_source,
                      hand_prior_sha256=self.hand_prior_sha256,
                      hand_pca_components=self.hand_pca_components)
        if self.profiles_enabled and profile["model_type"] == "soma":
            frozen.update(fixed_joints=soma_finger_joints(self.model), fitting_scope="body_wrist_only")
        verify_profile_files(frozen)
        return frozen

    def start_sampling(self, _):
        with self.lock:
            _, packet = self.source.snapshot()
            if self.mode not in IDLE_MODES or not packet or not self.selected:
                return
            fr = subject_frame(packet, self.selected)
            if time.perf_counter() - packet["arrival_time"] > 1 or not fr or fr["valid"].sum() < 15:
                self.message = "可见 marker 不足或数据过期；请检查 Shogun 的全身标签。"
                return
            try:
                frozen = self._freeze_model_request()
            except Exception as exc:
                self.message = f"模型配置检查失败，未开始采样：{exc}"
                return
            self.mode = "countdown"
            self.sample_model_request = frozen
            self.sample_static_pose = "a" if self.pose_kind.value.startswith("A") else "t"
            self.sample_body_regions = None if self.body_regions is None else json.loads(json.dumps(self.body_regions))
            self.sample_fit_hands = self.fit_hands
            self.sample_hand_prior_source = self.hand_prior_source
            self.sample_hand_prior_sha256 = self.hand_prior_sha256
            self.countdown_end = time.perf_counter() + 5
            self.samples = []
            self.run = self.data_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
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
            self._show_model_preview()
            self._sync_model_controls()

    def finish_sampling(self, _):
        with self.lock:
            if self.mode not in ("sampling", "countdown"):
                return
            self.mode = "preview"
            self.capture.disabled = self.subject.disabled = self.duration.disabled = self.pose_kind.disabled = False
            self.finish.disabled = True
            if not self.samples:
                self.message = "未采到 marker；可重新开始。"
                self._show_model_preview()
                self._sync_model_controls()
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
            if self.sample_model_request is not None:
                self.sample_model_request.update(tpose=list(self.tpose_window), static_pose=self.sample_static_pose,
                    marker_regions_snapshot_sha256=hashlib.sha256((self.run / "marker_regions.json").read_bytes()).hexdigest()
                        if self.sample_body_regions is not None else None)
            self._receipt("sampled", source_fps=self.source.frame_rate, sampled_fps=fps,
                          samples=len(times), tpose_window=list(self.tpose_window), duration_seconds=float(times[-1]),
                          static_pose=self.sample_static_pose,
                          marker_scope="body_hands" if self.sample_fit_hands else
                                       "body_wrist_only" if self.sample_body_regions is not None else "all_markers",
                          hand_prior_source=self.sample_hand_prior_source if self.sample_fit_hands else None,
                          hand_prior_sha256=self.sample_hand_prior_sha256 if self.sample_fit_hands else None)
            self.fit_button.disabled = times[-1] < 10 or self.tpose_window[1] - self.tpose_window[0] < 5
            self.message = f"已保留 {len(times)} 帧 / {times[-1]:.1f} 秒。" + ("可以计算标定。" if not self.fit_button.disabled else "采样太短，请重新采样。")
            self._show_model_preview()
            self._sync_model_controls()

    def _receipt(self, state, **extra):
        p = self.run / "session.json"
        old = json.loads(p.read_text()) if p.exists() else {}
        if getattr(self, "calibration_scope", None) is not None:
            extra.setdefault("marker_scope", self.calibration_scope)
        old.update(self._model_receipt())
        old.update(state=state, subject=self.selected, model_type=getattr(self.model, "model_type", self.args.model_type),
                   source=self.args.source, coordinate_system="metres, Y-up; Vicon (x,z,-y)",
                   raw_capture_modified=False, **extra)
        p.write_text(json.dumps(old, ensure_ascii=False, indent=2))

    def start_calibration(self, _):
        with self.lock:
            if self.mode != "preview" or self.fit_button.disabled:
                return
            if self.sample_model_request is None:
                self.message = "采样配置已失效；请为当前模型重新采样。历史文件已保留。"
                self.fit_button.disabled = True
                return
            request = copy.deepcopy(self.sample_model_request)
            if request["source_subject"] != self.selected:
                self.message = "被试已改变；请重新采样标定。"
                return
            # Run the existing CLI in a worker process: its CPU work cannot block the browser.
            metadata = self.run / "calibration_request.json"
            try:
                metadata.write_text(json.dumps(request, ensure_ascii=False, indent=2))
            except OSError as exc:
                self.message = f"无法保存标定请求，未启动计算；采样文件已保留：{exc}"
                return
            self.mode = "calibrating"
            self.fit_button.disabled = self.capture.disabled = self.subject.disabled = self.pose_kind.disabled = True
            self.message = "正在计算被试体型与 marker 布局；进度保存在 calibration.log。"
            job = dict(request=request, run=str(self.run))
            self._calibration_job = copy.deepcopy(job)
            self._show_model_preview()
            self._sync_model_controls()
            threading.Thread(target=self._calibrate, args=(copy.deepcopy(job),), daemon=True).start()

    def _calibrate(self, job=None):
        import subprocess
        import sys
        job = copy.deepcopy(job if job is not None else getattr(self, "_calibration_job", None))
        if job is None:
            # Compatibility for existing direct callers with an old request file.
            # Capture args once here; never read them after dispatching the worker.
            directory = Path(self.run)
            request = json.loads((directory / "calibration_request.json").read_text())
            for key, default in (("model_path", None), ("model_type", "smplx"), ("gender", "neutral"),
                                 ("num_betas", 10), ("soma_lod", "mid"), ("soma_identity", "soma")):
                request.setdefault(key, getattr(self.args, key, default))
            job = dict(run=str(directory), request=request)
        directory, request = Path(job["run"]), job["request"]
        output = directory / "subject_calib.npz"
        code = """import json,numpy as np,sys,hashlib
from pathlib import Path
from vsmpl.body_model import BodyModel
from vsmpl.calibration import calibrate
from vsmpl.calibration_input import prepare_calibration_input
from vsmpl.sources import load_recording
from vsmpl.model_profiles import verify_profile_files
from run_web import soma_finger_joints
request=json.loads(Path(sys.argv[5]).read_text())
if 'hand_prior' not in request and request.get('hand_prior_source'):
    request['hand_prior']=request['hand_prior_source']
verify_profile_files(request)
soma_kw=dict(soma_lod=request.get('soma_lod','mid'),soma_identity=request.get('soma_identity','soma')) if sys.argv[2]=='soma' else {}
if 'num_betas' in request:
    soma_kw['num_betas']=int(request['num_betas'])
model=BodyModel.load(sys.argv[1],sys.argv[2],sys.argv[3],**soma_kw)
rec=load_recording(sys.argv[4])
assert request['source_subject']==sys.argv[8], 'Calibration subject association changed'
region_file=Path(sys.argv[5]).parent/request['body_regions_file'] if request['body_regions_file'] else None
if region_file and request.get('marker_regions_snapshot_sha256'):
    if hashlib.sha256(region_file.read_bytes()).hexdigest()!=request['marker_regions_snapshot_sha256']:
        raise ValueError('Frozen marker regions snapshot changed')
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
fitting_args={}
if sys.argv[2]=='soma' and request.get('fixed_joints') is not None:
    expected_fixed=soma_finger_joints(model)
    if request.get('fixed_joints')!=expected_fixed or request.get('fitting_scope')!='body_wrist_only' or regions is None:
        raise ValueError('SOMA frozen body/wrist fitting configuration does not match its skeleton/regions')
    fitting_args['fixed_joints']=expected_fixed
body_result=calibrate(model,rec,tpose=tuple(request['tpose']),static_pose=request['static_pose'],marker_regions=regions,**fitting_args)
result=dict(body_result)
if fitting_args:
    result.update(fitting_fixed_joints=np.asarray(fitting_args['fixed_joints'],dtype=np.int64),
                  fitting_scope='body_wrist_only')
scope='body_wrist_only' if regions is not None else 'all_markers'
if fit_hands:
    from vsmpl.full_calibration import calibrate_hands
    body_saved=dict(body_result)
    body_saved.update(model_type=sys.argv[2],gender=sys.argv[3],num_betas=getattr(model,'num_betas',10),
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
result.update(model_type=sys.argv[2],gender=sys.argv[3],num_betas=getattr(model,'num_betas',10),source_subject=request['source_subject'],
marker_scope=scope,marker_regions_sha256=hashlib.sha256(region_file.read_bytes()).hexdigest() if region_file else '',
excluded_marker_names=np.array([n for n in all_names if n not in set(map(str,result['marker_names']))],dtype=str))
for key in ('profile_signature','profile_id','profile_label','body_regions_source_sha256'):
    if request.get(key) is not None:
        result[key]=request[key]
np.savez(sys.argv[6],**result)
"""
        try:
            metadata = directory / "calibration_request.json"
            if getattr(self, "profiles_enabled", False) or getattr(self, "sample_model_request", None) is not None:
                metadata.write_text(json.dumps(request, ensure_ascii=False, indent=2))
            with (directory / "calibration.log").open("w") as log:
                subprocess.run([sys.executable, "-u", "-c", code, request["model_path"], request["model_type"],
                                request["gender"], str(directory / "calibration_markers.npz"),
                                str(metadata), str(output), request.get("static_pose", "a"), request["source_subject"]],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            with self.lock:
                if self.selected != request["source_subject"]:
                    raise ValueError("Worker completed for a different subject; output retained without activation")
                if (getattr(self, "profiles_enabled", False)
                        and request.get("profile_signature") != profile_signature(self.profiles[self.active_profile])):
                    raise ValueError("Worker completed for a different model profile; output retained without activation")
                self.calibration_path = output
                self._load_calibration(output)
                self.mode = "calibrated"
                self.message = "标定计算完成，尚待真人叠加检查。请先检查逐点残差和朝向，再开始测试拟合。"
                self._receipt("calibrated", calibration_file=output.name)
                self.track_button.disabled = False
                self._remember_calibration()
        except Exception as e:
            with self.lock:
                self.mode = "preview"
                self.message = f"标定失败：{e}。采样与日志已保留。"
                self._receipt("calibration_failed", error=str(e))
                self.fit_button.disabled = False
        finally:
            self.capture.disabled = self.subject.disabled = self.pose_kind.disabled = False
            self._show_model_preview()
            self._sync_model_controls()

    def _load_calibration(self, path):
        with np.load(path, allow_pickle=False) as calib:
            strict = getattr(self, "profiles_enabled", False)
            expected_type = getattr(self.model, "model_type", getattr(self.args, "model_type", None))
            expected_subject = getattr(self, "selected", None)
            if strict and any(key not in calib for key in ("source_subject", "model_type", "betas")):
                raise ValueError("Calibration must identify its subject, body model and identity coefficients")
            if "source_subject" in calib and expected_subject is not None and str(calib["source_subject"]) != expected_subject:
                raise ValueError("Saved calibration subject does not match the selected actor")
            if "model_type" in calib and expected_type is not None and str(calib["model_type"]) != expected_type:
                raise ValueError("Saved calibration body model does not match the active model")
            expected_gender = getattr(self.args, "gender", None)
            if "gender" in calib and expected_gender is not None and str(calib["gender"]) != expected_gender:
                raise ValueError("Saved calibration gender does not match the active model gender")
            if "betas" in calib and hasattr(self.model, "num_betas"):
                if calib["betas"].shape != (self.model.num_betas,) or not np.isfinite(calib["betas"]).all():
                    raise ValueError("Calibration identity coefficient dimensions do not match the active model")
            if strict and "profile_signature" in calib:
                if str(calib["profile_signature"]) != profile_signature(self.profiles[self.active_profile]):
                    raise ValueError("Calibration belongs to a different model profile signature")
            compatibility = None
            if getattr(self.model, "model_type", None) == "soma" or ("model_type" in calib and str(calib["model_type"]) == "soma"):
                from vsmpl.soma_model import check_calibration_model
                compatibility = check_calibration_model(self.model, calib)
            scope = str(calib["marker_scope"]) if "marker_scope" in calib else "all_markers"
            soma_fixed = None
            if strict and expected_type == "soma":
                if scope != "body_wrist_only":
                    raise ValueError("SOMA body/wrist profiles only accept body_wrist_only calibrations")
                allowed = set(self.body_regions or ())
                names = list(map(str,calib["marker_names"]))
                if not names or len(set(names)) != len(names) or not set(names).issubset(allowed):
                    raise ValueError("SOMA calibration markers must be a subset of the body's region whitelist")
                soma_fixed = soma_finger_joints(self.model)
                if "fitting_fixed_joints" in calib:
                    saved = np.asarray(calib["fitting_fixed_joints"])
                    if (saved.ndim != 1 or not np.issubdtype(saved.dtype,np.integer)
                            or len(np.unique(saved)) != len(saved) or set(saved.tolist()) != set(soma_fixed)):
                        raise ValueError("SOMA calibration fitting_fixed_joints do not match the active finger joints")
                else:
                    compatibility = copy.deepcopy(compatibility or {})
                    compatibility["status"] = "LEGACY_PARTIAL"
                    compatibility.setdefault("unverified_fields",[]).append("fitting_fixed_joints")
            rig = MarkerRig.from_calibration(self.model, calib)
            residual = calib["per_marker_rms"] * 1000
            if strict and self.fit_hands and scope != "body_hands":
                raise ValueError("The body/hand profile requires its own full body_hands calibration")
            if scope == "body_hands":
                if strict and (expected_type != "smplx" or not self.fit_hands or self.model.num_joints != 55):
                    raise ValueError("Full hand calibration requires the matching SMPL-X body/hand profile")
                configured = getattr(self, "hand_prior_source", None) or getattr(self.args, "hand_prior", None)
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
                if strict and expected_type == "soma":
                    solver = RealtimeSolver(rig, SolverConfig(fixed_joints=soma_fixed))
                else:
                    solver = RealtimeSolver(rig)
                finger_mask = None
        self.rig, self.solver = rig, solver
        self.calibration_scope = scope
        self.finger_marker_mask = finger_mask
        self.last_observation = None
        self.calibration_compatibility = compatibility
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
        with np.load(path, allow_pickle=False) as metadata:
            if "head_neutral_reference_confirmed" in metadata and bool(metadata["head_neutral_reference_confirmed"]):
                self.detail.content += "\n\n头部绑定已按本人正头参考校准；点头、转头和侧倾继续由 marker 驱动。"
        if getattr(self.model, "model_type", None) == "soma":
            self.detail.content += "\n\n" + self._model_description()
        if compatibility and compatibility.get("status") == "LEGACY_PARTIAL":
            self.detail.content += "\n\n**LEGACY_PARTIAL**：旧标定缺少部分模型来源字段，兼容性仅作部分核对。"
            if "fitting_fixed_joints" in compatibility.get("unverified_fields",[]):
                self.detail.content += "旧 SOMA 标定未记录离线手指冻结约束；本次仅保证在线固定手指，离线静态拟合与精修尚未核验，建议重新标定。"

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
            self.recorder = SequenceRecorder(getattr(self.model, "model_type", self.args.model_type), self.args.gender, self.rig.betas,
                                             self.source.frame_rate, model=self.model)
            self.fit_count = 0
            self.last_result = None
            self.last_observation = None
            self.mode = "tracking"
            self._show_model_preview()
            self._sync_model_controls()
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
            self._remember_calibration()
            self._show_model_preview()
            self._sync_model_controls()

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
                                    faces=self.model.faces.astype(np.uint32), color=(135, 162, 198), opacity=1.0)
                            else:
                                self.mesh.vertices = verts
                        if self.mesh:
                            self.mesh.visible = visible and self.show_mesh.value
                        if visible and self.view_mode != "body":
                            self.update_hand_camera()
                    if now - ui_time > 0.25:
                        ui_time = now
                        self._sync_model_controls()
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
                        receipt.update(self._model_receipt())
                        if self.mode == "tracking" and self.last_result is not None:
                            rw, _ = self.model.fk(self.last_result.R, self.rig.J_rest)
                            hj = getattr(self.model, "head_joint", 15)
                            receipt["head_world_up_tilt_degrees"] = float(np.degrees(np.arccos(np.clip(rw[hj, 1, 1], -1., 1.))))
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
    parser.add_argument("--model-type", default="smplx", choices=("smpl", "smplh", "smplx", "soma"),
                        help="soma = NVIDIA SOMA (py-soma-x); --model-path is then the SOMA asset folder or 'hf'")
    parser.add_argument("--soma-lod", default="mid", choices=("mid", "low", "xlo"), help="SOMA mesh level of detail")
    parser.add_argument("--soma-identity", default="soma", choices=("soma", "smpl", "smplx"),
                        help="SOMA identity backend (native SOMA PCA by default)")
    parser.add_argument("--gender", default="neutral")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8792)
    parser.add_argument("--output", default="runtime")
    parser.add_argument("--data-output", help="directory for new calibration samples/results; defaults to --output; loaded calibrations retain their existing run directory")
    parser.add_argument("--calib", help="load a separately reviewed calibration candidate without resampling")
    parser.add_argument("--subject", help="explicit subject associated with --calib")
    parser.add_argument("--body-regions", help="validated body/wrist region JSON; freeze per sample and exclude other markers from fitting")
    parser.add_argument("--fit-hands", action="store_true", help="fit body and both articulated hands using the complete source markers")
    parser.add_argument("--hand-prior", help="licensed native SMPL-X hand PCA prior for --fit-hands/full saved calibration")
    parser.add_argument("--hand-components", type=int, default=12, help="hand PCA coefficients per side for new full calibrations")
    parser.add_argument("--num-betas", type=int, default=10, help="identity coefficient count for the selected backend")
    parser.add_argument("--model-profiles", help="local model profiles JSON; enables idle-only model switching and static previews")
    FieldApp(parser.parse_args()).loop()


if __name__ == "__main__":
    main()
