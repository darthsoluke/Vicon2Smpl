"""Model-selection lifecycle and frozen worker contracts, with no SOMA assets/hardware."""
import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_web
from vsmpl.model_profiles import profile_signature


BODY = [f"BODY{i:02}" for i in range(52)]
FINGERS = [f"FINGER{i:02}" for i in range(20)]


class Control:
    def __init__(self, **kwargs):
        self.disabled, self.visible, self.content = False, True, ""
        self.__dict__.update(kwargs)
    def on_click(self, callback): self.callback = callback
    def on_update(self, callback): self.callback = callback
    def __enter__(self): return self
    def __exit__(self, *args): pass


class FakeGui:
    def configure_theme(self, **kwargs): pass
    def add_markdown(self, text): return Control(content=text)
    def add_folder(self, *args, **kwargs): return Control()
    def add_button(self, *args, **kwargs): return Control(**kwargs)
    def add_slider(self, *args, **kwargs): return Control(value=kwargs["initial_value"])
    def add_checkbox(self, *args, **kwargs): return Control(value=kwargs["initial_value"])
    def add_dropdown(self, label, options, **kwargs):
        return Control(options=options, value=kwargs.get("initial_value", options[0]))


class Mesh(Control):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.removed = False
    def remove(self): self.removed, self.visible = True, False


class FakeScene:
    def set_up_direction(self, *args): pass
    def add_grid(self, *args, **kwargs): pass
    def add_point_cloud(self, *args, **kwargs): return Control(**kwargs)
    def add_mesh_simple(self, *args, **kwargs): return Mesh(**kwargs)


class FakeServer:
    def __init__(self, **kwargs): self.gui, self.scene = FakeGui(), FakeScene()
    def on_client_connect(self, callback): pass
    def get_clients(self): return {}
    def stop(self): pass


class FakeSource:
    frame_rate, error = 300, ""
    def __init__(self, *args): pass
    def snapshot(self):
        return 1, dict(frame=1, arrival_time=time.perf_counter(), subjects=[dict(name="zhr", markers=[
            dict(name=name, valid=True, xyz=[100,200,1000]) for name in BODY+FINGERS])])
    def close(self): pass


class FakeModel:
    def __init__(self, kind, num_betas=2):
        self.model_type, self.num_betas = kind, num_betas
        self.num_joints = 77 if kind == "soma" else 55
        count = 8 if kind == "soma" else 6
        self.v_template = np.c_[np.linspace(-.2,.2,count), np.linspace(-.9,1.,count), np.zeros(count)]
        self.faces = np.array([[0,1,2],[2,3,4]])
        self.weights = np.zeros((count,self.num_joints)); self.weights[:,0] = 1
        self.joint_names = [f"Joint{i}" for i in range(self.num_joints)]
        if kind == "soma":
            for index, name in ((20,"LeftHand"),(21,"RightHand"),(25,"LeftHandIndex1"),
                                (26,"LeftHandIndex2"),(40,"RightHandThumb1"),(41,"RightHandThumbEnd")):
                self.joint_names[index] = name
        self.prepared = []
    def prepare_for(self, betas): self.prepared.append(np.asarray(betas).copy())
    def shaped(self, betas): return self.v_template + float(np.asarray(betas)[0])*.01, np.zeros((self.num_joints,3))
    def static_reference_pose(self, *args): return np.zeros((self.num_joints,3))
    def vertices(self, R, trans, rest, joints): return rest + trans


class FakeSolver:
    def __init__(self, *args, **kwargs):
        self.cfg = (args[1] if len(args) > 1 and isinstance(args[1],run_web.SolverConfig)
                    else SimpleNamespace(min_markers=4))
    def reset(self): pass


class WebModelSwitchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vsmpl-profile-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        smpl = self.root / "smplx.npz"; smpl.write_bytes(b"fake SMPL-X geometry")
        soma = self.root / "soma"; soma.mkdir()
        for name in ("SOMA_neutral.npz", "SOMA_template_rig.usda", "SOMA_procedural_transforms.json"):
            (soma / name).write_bytes(b"fake SOMA identity asset")
        regions = self.root / "regions.json"
        regions.write_text(json.dumps({name:[0,1] for name in BODY}))
        prior = self.root / "prior.npz"; prior.write_bytes(b"fake learned prior")
        self.profiles_file = self.root / "profiles.json"
        self.profiles_file.write_text(json.dumps(dict(default="smplx", profiles=[
            dict(id="smplx",label="SMPL-X 身体双手",model_type="smplx",model_path=str(smpl),num_betas=2,
                 body_regions=str(regions),fit_hands=True,hand_prior=str(prior),hand_components=20),
            dict(id="soma",label="SOMA 身体手腕",model_type="soma",model_path=str(soma),num_betas=2,
                 body_regions=str(regions),soma_lod="low",soma_identity="soma")
        ])))
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(sys.modules, {"viser":SimpleNamespace(ViserServer=FakeServer)}))
        self.loads = self.stack.enter_context(patch.object(run_web.BodyModel,"load",side_effect=self.load_model))
        self.source_factory = self.stack.enter_context(patch.object(run_web,"NativeSource",FakeSource))
        self.stack.enter_context(patch.object(run_web.HandPCA,"load",return_value=SimpleNamespace(n_components=20)))
        self.rig_factory = self.stack.enter_context(patch.object(run_web.MarkerRig,"from_calibration",side_effect=self.make_rig))
        self.stack.enter_context(patch.object(run_web,"PCASolver",FakeSolver))
        self.stack.enter_context(patch.object(run_web,"RealtimeSolver",FakeSolver))
        self.stack.enter_context(patch("vsmpl.soma_model.check_calibration_model",return_value=dict(status="LEGACY_PARTIAL")))
        args = SimpleNamespace(model_path="unused",model_type="smplx",gender="neutral",output=str(self.root/"hot"),
            data_output=str(self.root/"data"),source="test-only",native_reader="unused",bind="127.0.0.1",port=0,
            model_profiles=str(self.profiles_file))
        self.app = run_web.FieldApp(args)
        self.app.selected, self.app.subject.value = "zhr", "zhr"

    @staticmethod
    def load_model(path, kind, gender, **kwargs): return FakeModel(kind,kwargs.get("num_betas",2))

    @staticmethod
    def make_rig(model, calib):
        beta = calib["betas"].copy()
        model.prepare_for(beta)
        rest, joints = model.shaped(beta)
        names = list(map(str,calib["marker_names"]))
        return SimpleNamespace(names=names,num_markers=len(names),betas=beta,v_rest=rest,J_rest=joints,
                               forward=lambda R,t:np.zeros((len(names),3)))

    def switch(self, name):
        self.app.model_selector.value = self.app.profiles[name]["label"]
        self.app.change_model(None)

    def checkpoint(self, profile="smplx", subject="zhr", betas=(.2,.4), signature=True, **extra):
        cfg = self.app.profiles[profile]
        fingers = profile == "smplx"
        names = BODY+FINGERS if fingers else BODY
        fields = dict(model_type=profile,source_subject=subject,betas=np.array(betas),marker_names=np.array(names),
            vertex_ids=np.zeros(len(names),int),offsets=np.zeros((len(names),3)),per_marker_rms=np.full(len(names),.005),
            marker_scope="body_hands" if fingers else "body_wrist_only",static_pose="a")
        if fingers:
            fields.update(finger_marker_names=np.array(FINGERS),hand_prior_source=cfg["hand_prior"],
                hand_prior_sha256=cfg["hand_prior_sha256"],hand_pca_components=20,
                hand_pca_initial_latent=np.zeros((2,20)),hand_prior_weight=1e-5,
                head_neutral_reference_confirmed=True)
        if signature: fields["profile_signature"] = profile_signature(cfg)
        fields.update(extra)
        path = self.root / (profile+"_"+subject+"_"+str(len(list(self.root.glob('*.npz'))))+".npz")
        np.savez(path,**fields)
        return path

    def activate_checkpoint(self, path):
        app = self.app
        app._load_calibration(path)
        app.calibration_path, app.run, app.mode = path, path.parent, "calibrated"
        app.track_button.disabled = False
        app._remember_calibration()
        app._show_model_preview()

    def sample(self):
        app = self.app
        app.start_sampling(None)
        self.assertEqual(app.mode,"countdown",app.message)
        _, packet = app.source.snapshot()
        frame = run_web.subject_frame(packet,"zhr")
        app.samples = [dict(frame,frame=i,time=100+i/3) for i in range(40)]
        app.mode = "sampling"
        app.finish_sampling(None)
        self.assertFalse(app.fit_button.disabled)

    def test_lazy_load_changes_topology_and_disables_soma_hand_fitting(self):
        app = self.app
        self.assertEqual(self.loads.call_count,1)
        self.assertEqual(self.loads.call_args.args[1],"smplx")
        old_preview = app.preview_mesh
        old_body = app.mesh = Mesh(vertices=np.zeros((6,3)))
        source = app.source
        self.switch("soma")
        self.assertEqual(app.active_profile,"soma",app.message)
        self.assertEqual(self.loads.call_count,2)
        self.assertTrue(old_preview.removed and old_body.removed)
        self.assertEqual(len(app.preview_mesh.vertices),8)
        self.assertEqual(float(app.preview_mesh.vertices[:,1].min()),0.)
        self.assertIn("非真人",app.preview_detail.content)
        self.assertIn("不拟合手指",app.detail.content)
        self.assertTrue(all(b.disabled for b in app.hand_buttons.values()))
        self.assertFalse(app.fit_hands)
        self.assertIs(app.source,source)

    def test_busy_phases_reject_model_and_subject_changes(self):
        for mode in ("countdown","sampling","calibrating","tracking","loading"):
            with self.subTest(mode=mode):
                self.app.mode = mode
                self.app._sync_model_controls()
                self.assertTrue(self.app.model_selector.disabled)
                self.switch("soma")
                self.assertEqual(self.app.active_profile,"smplx")
                self.assertEqual(self.app.model_selector.value,self.app.profiles["smplx"]["label"])
                self.app.subject.value = "other"
                self.app.change_subject(None)
                self.assertEqual(self.app.selected,"zhr")
                self.assertEqual(self.app.subject.value,"zhr")
        self.assertEqual(self.loads.call_count,1)

    def test_load_failure_restores_model_calibration_and_controls(self):
        app = self.app
        path = self.checkpoint()
        self.activate_checkpoint(path)
        model, rig, template, detail = app.model,app.rig,app.preview_mesh,app.detail.content
        self.loads.side_effect = FileNotFoundError("missing local SOMA assets")
        self.switch("soma")
        self.assertIs(app.model,model)
        self.assertIs(app.rig,rig)
        self.assertIs(app.preview_mesh,template)
        self.assertFalse(template.removed)
        self.assertEqual(app.detail.content,detail)
        self.assertEqual(app.calibration_path,path)
        self.assertEqual(app.mode,"calibrated")
        self.assertFalse(app.track_button.disabled or app.model_selector.disabled)
        self.assertIn("保留原模型",app.message)

    def test_incompatible_saved_profile_calibration_rolls_back_after_loading(self):
        original = self.checkpoint()
        self.activate_checkpoint(original)
        model, rig, template = self.app.model,self.app.rig,self.app.preview_mesh
        self.app._profile_calibrations[("soma","zhr")] = self.checkpoint("soma",subject="other")
        self.switch("soma")
        self.assertEqual(self.app.active_profile,"smplx")
        self.assertIs(self.app.model,model)
        self.assertIs(self.app.rig,rig)
        self.assertIs(self.app.preview_mesh,template)
        self.assertEqual(self.app.calibration_path,original)
        self.assertFalse(self.app.track_button.disabled)
        self.assertTrue(self.app.fit_button.disabled)

    def test_switch_invalidates_sample_ui_but_retains_sample_files(self):
        self.sample()
        directory = self.app.run
        before = (directory/"calibration_markers.npz").read_bytes()
        self.switch("soma")
        self.assertIsNone(self.app.recording)
        self.assertIsNone(self.app.sample_model_request)
        self.assertTrue(self.app.fit_button.disabled)
        self.assertEqual((directory/"calibration_markers.npz").read_bytes(),before)
        self.assertIn("文件保留",self.app.message)

    def test_profiles_rebuild_saved_rigs_without_transferring_identity(self):
        smpl = self.checkpoint(signature=False)
        original = smpl.read_bytes()
        self.activate_checkpoint(smpl)
        smpl_rig = self.app.rig
        self.switch("soma")
        self.assertIsNone(self.app.rig)
        soma = self.checkpoint("soma",betas=(1.,-.3))
        self.activate_checkpoint(soma)
        soma_rig = self.app.rig
        self.assertIn("LEGACY_PARTIAL",self.app.detail.content)
        self.switch("smplx")
        self.assertEqual(self.app.mode,"calibrated",self.app.message)
        self.assertIsNot(self.app.rig,smpl_rig)
        np.testing.assert_array_equal(self.app.rig.betas,[.2,.4])
        self.assertIn("本人正头",self.app.detail.content)
        self.switch("soma")
        self.assertIsNot(self.app.rig,soma_rig)
        np.testing.assert_array_equal(self.app.rig.betas,[1.,-.3])
        self.assertEqual(smpl.read_bytes(),original)

    def test_soma_body_wrist_solver_explicitly_freezes_fingers_but_keeps_wrists(self):
        self.switch("soma")
        self.activate_checkpoint(self.checkpoint("soma"))
        self.assertEqual(self.app.solver.cfg.fixed_joints,[25,26,40,41])
        self.assertNotIn(20,self.app.solver.cfg.fixed_joints)
        self.assertNotIn(21,self.app.solver.cfg.fixed_joints)

    def test_soma_rejects_non_body_scope_and_markers_outside_whitelist_but_allows_subset(self):
        self.switch("soma")
        for scope in ("all_markers","body_hands","unknown"):
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError,"body_wrist_only"):
                self.app._load_calibration(self.checkpoint("soma",marker_scope=scope))
        with self.assertRaisesRegex(ValueError,"whitelist"):
            self.app._load_calibration(self.checkpoint("soma",marker_names=np.array(BODY+[FINGERS[0]])))
        subset = self.checkpoint("soma",marker_names=np.array(BODY[:3]),vertex_ids=np.zeros(3,int),
            offsets=np.zeros((3,3)),per_marker_rms=np.zeros(3),fitting_fixed_joints=np.array([25,26,40,41]))
        self.app._load_calibration(subset)
        self.assertEqual(self.app.rig.names,BODY[:3])

    def test_soma_rejects_changed_offline_finger_constraints_and_flags_missing_legacy_proof(self):
        self.switch("soma")
        for fixed in (np.array([25,40]),np.array([25,26,40,41,41]),np.array([25.,26.,40.,41.])):
            with self.subTest(fixed=fixed), self.assertRaisesRegex(ValueError,"fitting_fixed_joints"):
                self.app._load_calibration(self.checkpoint("soma",fitting_fixed_joints=fixed))
        with patch("vsmpl.soma_model.check_calibration_model",return_value=dict(status="PASS",unverified_fields=[])):
            self.app._load_calibration(self.checkpoint("soma"))
        self.assertEqual(self.app.calibration_compatibility["status"],"LEGACY_PARTIAL")
        self.assertIn("离线静态拟合与精修尚未核验",self.app.detail.content)
        self.assertEqual(self.app.solver.cfg.fixed_joints,[25,26,40,41])

    def test_new_subject_invalidates_every_profile_cache_without_deleting_files(self):
        smpl = self.checkpoint(); self.activate_checkpoint(smpl)
        self.switch("soma")
        soma = self.checkpoint("soma"); self.activate_checkpoint(soma)
        self.app.subject.value = "other"
        self.app.change_subject(None)
        self.assertEqual(self.app._profile_calibrations,{})
        self.assertIsNone(self.app.rig)
        self.switch("smplx")
        self.assertEqual(self.app.mode,"preview")
        self.assertIsNone(self.app.calibration_path)
        self.assertTrue(smpl.is_file() and soma.is_file())

    def test_failed_candidate_path_never_replaces_a_saved_calibration_cache(self):
        original = self.checkpoint()
        self.activate_checkpoint(original)
        failed = self.checkpoint(subject="other")
        self.app.mode,self.app.calibration_path,self.app.rig,self.app.solver = "preview",failed,None,None
        self.app._remember_calibration()
        self.assertEqual(self.app._profile_calibrations[("smplx","zhr")],original)

    def test_calibration_header_rejects_subject_backend_dimensions_and_signature_mixups(self):
        cases = [dict(subject="other"),dict(model_type="soma"),dict(betas=(1.,2.,3.)),
                 dict(profile_signature="wrong-profile")]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                path = self.checkpoint(**kwargs)
                with self.assertRaises(ValueError): self.app._load_calibration(path)

    def test_worker_uses_sampling_profile_after_live_args_are_mutated(self):
        self.switch("soma")
        self.sample()
        app = self.app
        frozen = copy.deepcopy(app.sample_model_request)
        directory = app.run
        app.args.model_path,app.args.soma_lod,app.args.soma_identity,app.args.num_betas = "changed", "mid", "smplx", 99
        app.args.model_type = "smplx"
        threads = []
        class DeferredThread:
            def __init__(self,target,args,**kwargs): threads.append((target,args))
            def start(self): pass
        with patch.object(run_web.threading,"Thread",DeferredThread): app.start_calibration(None)
        self.assertEqual(app.mode,"calibrating")
        job = threads[0][1][0]
        self.assertEqual(job["request"]["model_type"],"soma")
        self.assertEqual(job["request"]["soma_lod"],"low")
        calls = []
        def inline(command,**kwargs):
            calls.append(command)
            with patch.object(sys,"argv",["-c",*command[4:]]):
                exec(compile(command[3],"<frozen-model-worker>","exec"),{"__name__":"__main__"})
            return subprocess.CompletedProcess(command,0)
        def fit(model,rec,**kwargs):
            self.assertEqual(model.model_type,"soma")
            self.assertEqual(model.num_betas,2)
            self.assertEqual(kwargs["fixed_joints"],[25,26,40,41])
            return dict(marker_names=np.array(rec["names"]),vertex_ids=np.zeros(52,int),
                offsets=np.zeros((52,3)),betas=np.array([.5,.2]),per_marker_rms=np.zeros(52),static_pose="a")
        with patch("subprocess.run",side_effect=inline), patch("vsmpl.calibration.calibrate",side_effect=fit):
            threads[0][0](*threads[0][1])
        self.assertEqual(app.mode,"calibrated",app.message)
        self.assertEqual(calls[0][4:7],[frozen["model_path"],"soma","neutral"])
        self.assertEqual(self.loads.call_args.kwargs,dict(soma_lod="low",soma_identity="soma",num_betas=2))
        with np.load(directory/"subject_calib.npz",allow_pickle=False) as result:
            self.assertEqual(str(result["model_type"]),"soma")
            self.assertEqual(str(result["profile_signature"]),frozen["profile_signature"])
            self.assertEqual(result["fitting_fixed_joints"].tolist(),[25,26,40,41])
            self.assertEqual(str(result["fitting_scope"]),"body_wrist_only")
        request = json.loads((directory/"calibration_request.json").read_text())
        self.assertEqual(request["num_betas"],2)
        self.assertEqual(request["fixed_joints"],frozen["fixed_joints"])
        self.assertEqual(request["marker_regions_snapshot_sha256"],hashlib.sha256((directory/"marker_regions.json").read_bytes()).hexdigest())

    def test_changed_assets_prevent_sampling_without_creating_a_new_run(self):
        Path(self.app.profiles["smplx"]["model_path"]).write_bytes(b"changed geometry bytes")
        self.app.start_sampling(None)
        self.assertEqual(self.app.mode,"preview")
        self.assertIsNone(self.app.run)
        self.assertIsNone(self.app.sample_model_request)
        self.assertIn("未开始采样",self.app.message)

    def test_worker_rejects_modified_frozen_region_snapshot_and_preserves_source(self):
        self.switch("soma")
        self.sample()
        app = self.app
        directory = app.run
        raw = (directory/"calibration_markers.npz").read_bytes()
        request = copy.deepcopy(app.sample_model_request)
        (directory/"marker_regions.json").write_text(json.dumps({name:[0,1,2] for name in BODY}))
        app.mode = "calibrating"
        def inline(command,**kwargs):
            with patch.object(sys,"argv",["-c",*command[4:]]):
                exec(compile(command[3],"<tampered-region-worker>","exec"),{"__name__":"__main__"})
            return subprocess.CompletedProcess(command,0)
        with patch("subprocess.run",side_effect=inline), patch("vsmpl.calibration.calibrate") as fit:
            app._calibrate(dict(run=str(directory),request=request))
            fit.assert_not_called()
        self.assertEqual(app.mode,"preview")
        self.assertIn("snapshot changed",app.message)
        self.assertTrue((directory/"calibration.log").exists())
        self.assertEqual((directory/"calibration_markers.npz").read_bytes(),raw)
        self.assertFalse((directory/"subject_calib.npz").exists())

    def test_legacy_without_profiles_keeps_no_selector_or_static_preview(self):
        args = SimpleNamespace(model_path=self.app.profiles["smplx"]["model_path"],model_type="smplx",gender="neutral",
            output=str(self.root/"legacy"),source="test-only",native_reader="unused",bind="127.0.0.1",port=0)
        legacy = run_web.FieldApp(args)
        self.assertIsNone(legacy.model_selector)
        self.assertIsNone(legacy.preview_mesh)
        self.assertFalse(legacy.profiles_enabled)

    def test_legacy_checkpoint_gender_is_checked_without_changing_missing_field_compatibility(self):
        self.app.profiles_enabled = False
        with self.assertRaisesRegex(ValueError,"gender"):
            self.app._load_calibration(self.checkpoint(signature=False,gender="female"))
        self.app._load_calibration(self.checkpoint(signature=False))
        self.assertIn("本人正头",self.app.detail.content)

    def test_smplx_sampling_does_not_add_soma_finger_constraints(self):
        self.sample()
        self.assertNotIn("fixed_joints",self.app.sample_model_request)
        self.assertNotIn("fitting_scope",self.app.sample_model_request)

    def test_smplx_tracking_keeps_original_output_fields_and_template_lifecycle(self):
        path = self.checkpoint(signature=False)
        self.activate_checkpoint(path)
        self.app.start_tracking(None)
        self.assertEqual(self.app.mode,"tracking")
        self.assertFalse(self.app.preview_mesh.visible)
        self.assertTrue(self.app.model_selector.disabled)
        self.app.recorder.add(5,np.zeros((55,3)),np.zeros(3),.001)
        self.app.stop_tracking(None)
        receipt = json.loads((self.app.run/"session.json").read_text())
        with np.load(self.app.run/receipt["last_solved_file"],allow_pickle=False) as saved:
            self.assertEqual(saved["poses"].shape,(1,165))
            self.assertNotIn("solver_poses",saved)
            self.assertEqual(str(saved["model_type"]),"smplx")
        self.assertTrue(self.app.preview_mesh.visible)
        self.assertFalse(self.app.model_selector.disabled)
        self.assertEqual(receipt["active_model"],"smplx")
        self.assertEqual(receipt["model_capability"],"body_hands_pca")


if __name__ == "__main__":
    unittest.main()
