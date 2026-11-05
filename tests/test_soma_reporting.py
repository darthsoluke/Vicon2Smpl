"""Final native residuals and static SOMA rig compatibility, without downloaded assets."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vsmpl.body_model import BodyModel
from vsmpl.calibration import calibrate
from vsmpl.rotations import exp_so3
from vsmpl.soma_model import SomaBodyModel, check_calibration_model, _rig_fingerprint


class NonlinearIdentity(BodyModel):
    """The local model at zero omits a 100mm quadratic displacement at beta=1."""
    def __init__(self, model_type):
        self.base = np.array([[0.,0.,0.],[0.,1.,0.],[1.,0.,0.],[0.,0.,1.],
                              [1.,0.,1.],[1.,1.,0.],[0.,0.,2.]])
        regressor = np.zeros((2,7)); regressor[0,0] = regressor[1,1] = 1.
        weights = np.zeros((7,2)); weights[:,0] = 1.; weights[5] = [0.,1.]
        super().__init__(self.base.copy(), np.zeros((7,3,1)), None, regressor, weights,
                         np.array([-1,0]), np.array([[0,1,2],[0,2,3],[2,3,4],[1,2,5],[3,4,6]]), model_type)
        self.prepare_for(np.zeros(1))

    def prepare_for(self, betas):
        b = float(betas[0])
        direction = np.array([0.,0.,.1])
        derivative = np.broadcast_to(2*b*direction, self.base.shape)
        self.shapedirs = derivative[...,None].copy()
        self.v_template = self.base + b*b*direction - derivative*b


class FakeLayer:
    """Exercises the real SomaBodyModel constructor/recenter and metadata paths."""
    num_shape_components = 2
    public_joint_names = ("Root","Hips","Spine1","Spine2","Chest","Head","LeftArm","RightArm")
    public_joint_parent_ids = torch.tensor([-1,0,1,2,3,4,4,4])
    procedural_transforms_enabled = False
    correctives_model = None

    def __init__(self, weight_change=0.):
        self.faces = torch.tensor([[0,1,2],[2,3,4],[4,5,6]])
        self.weights = torch.full((7,8),1/7,dtype=torch.float32)
        self.weights[:,0] = 0.
        self.weights[0,1] += weight_change; self.weights[0,2] -= weight_change
        self.t_pose_world = torch.eye(4).repeat(8,1,1)
        self.prepare_identity(torch.zeros(1,2))

    def prepare_identity(self, coefficients):
        b = float(coefficients[0,0])
        self._cached_rest_shape = torch.arange(21,dtype=torch.float32).reshape(1,7,3)*.01 + b*b*.01
        self.bind = torch.eye(4).repeat(1,8,1,1)
        self.bind[0,:,1,3] = torch.arange(8)*.1 + b*b*.01

    def public_rig_view(self):
        return SimpleNamespace(skinning_weights=self.weights,t_pose_world=self.t_pose_world,
                               bind_transforms_world=self.bind)


class SomaReportingTests(unittest.TestCase):
    def setUp(self):
        version = patch("vsmpl.soma_model._soma_package_version",return_value="review-test-1")
        version.start(); self.addCleanup(version.stop)

    def run_calibration(self, model_type="soma"):
        model = NonlinearIdentity(model_type)
        F = 6; selected = np.array([0,2,4]); J = model.num_joints
        tracked_R = np.tile(np.eye(3),(F,J,1,1))
        refined_R = tracked_R[selected].copy()
        refined_R[:,1] = exp_so3([0.,0.,np.pi/2])
        refined_t = np.tile([.2,0.,0.],(len(selected),1))
        # Correct native predictions after beta=1, vertex reassignment and the
        # returned refined R/trans. Vertex 5 is child-owned, unlike original 0.
        expected = np.array([[.2,2.,.1],[.2,0.,2.1],[1.2,0.,.1],[.2,0.,1.1],[1.2,0.,1.1]])
        observed = np.broadcast_to(expected,(F,5,3)).copy()
        observed[0,0,0] += .003; observed[2,0,0] += .004
        observed[0,1,1] += .005; observed[4,1,1] += .012
        valid = np.ones((F,5),bool); valid[4,0] = valid[2,1] = False
        observed[~valid] = np.nan
        rec = dict(names=list("ABCDE"),pos=observed,valid=valid,fps=100.)
        offsets = np.zeros((5,3))
        offsets[0] = model.base[5]-model.base[0]
        offsets[1] = model.base[6]-model.base[1]
        fit = dict(loss=0.,betas=np.zeros(1),root=np.zeros(3),body=np.zeros((J-1,3)),trans=np.zeros(3))
        with patch("vsmpl.calibration.fit_tpose",return_value=fit), \
                patch("vsmpl.calibration.assign_vertices",return_value=(np.arange(5),np.zeros(5))), \
                patch("vsmpl.calibration.track_take",return_value=(tracked_R,np.zeros((F,3)),np.zeros(F),np.ones(F,bool))), \
                patch("vsmpl.calibration.farthest_point_frames",return_value=selected), \
                patch("vsmpl.calibration.inverse_skin_reassign",return_value=(np.arange(5),np.zeros((5,3)))), \
                patch("vsmpl.calibration.refine",return_value=(np.ones(1),offsets,refined_R,refined_t,np.array([.001,.002,0.,0.,0.]))), \
                redirect_stdout(io.StringIO()):
            # The actual final reassign, nonlinear prepare_for, MarkerRig and
            # native FK/LBS are intentionally not mocked.
            return calibrate(model,rec,tpose=(0,3),facing="+z",rounds=1,n_frames=3,verbose=False)

    def test_final_rms_uses_exact_identity_final_vertex_and_refined_pose_with_valid_denominator(self):
        result = self.run_calibration()
        np.testing.assert_array_equal(result["vertex_ids"],[5,6,2,3,4])
        np.testing.assert_allclose(result["per_marker_rms"],
            [np.sqrt((.003**2+.004**2)/2),np.sqrt((.005**2+.012**2)/2),0.,0.,0.],atol=2e-15)
        np.testing.assert_array_equal(result["linearised_refine_per_marker_rms"],[.001,.002,0.,0.,0.])
        np.testing.assert_array_equal(result["per_marker_rms_valid_counts"],[2,2,3,3,3])
        np.testing.assert_array_equal(result["per_marker_rms_evaluation_frames"],[0,2,4])
        self.assertEqual(result["per_marker_rms_stage"],"exact_final_identity_and_binding_at_refined_poses")

    def test_smplx_keeps_existing_refine_rms_and_does_not_add_soma_diagnostics(self):
        result = self.run_calibration("smplx")
        np.testing.assert_array_equal(result["per_marker_rms"],[.001,.002,0.,0.,0.])
        self.assertNotIn("per_marker_rms_stage",result)
        self.assertNotIn("linearised_refine_per_marker_rms",result)

    def make_model(self, weight_change=0.):
        return SomaBodyModel(FakeLayer(weight_change),num_betas=2,lod="low")

    def calibration(self, model):
        return dict(betas=np.zeros(2),**model.calibration_metadata())

    def test_config_and_rig_snapshot_are_explicit_and_independent_of_fitted_identity(self):
        model = self.make_model()
        before = model.calibration_metadata()
        model.prepare_for(np.array([1.,-.7]))
        after = model.calibration_metadata()
        self.assertEqual(before["soma_rig_fingerprint_sha256"],after["soma_rig_fingerprint_sha256"])
        self.assertEqual(after["soma_native_config"],"legacy78_lbs")
        self.assertFalse(after["soma_enable_procedural_transforms"])
        self.assertFalse(after["soma_apply_correctives"])
        self.assertFalse(after["soma_correctives_model_loaded"])
        self.assertEqual(after["soma_version"],"review-test-1")
        self.assertIn("excludes identity-dependent",after["soma_rig_fingerprint_scope"])
        recording = model.recording_metadata(np.ones(2))
        self.assertEqual(recording["soma_rig_fingerprint_sha256"],after["soma_rig_fingerprint_sha256"])

    def test_current_calibration_checks_all_stored_fields_but_does_not_claim_asset_files_verified(self):
        model = self.make_model()
        receipt = check_calibration_model(model,self.calibration(model))
        self.assertEqual(receipt["status"],"PASS")
        self.assertEqual(receipt["unverified_fields"],[])
        self.assertIn("soma_joint_names",receipt["verified_fields"])
        self.assertIn("soma_rig_fingerprint_sha256",receipt["verified_fields"])
        self.assertFalse(receipt["external_asset_files_verified"])

    def test_legacy_calibration_is_compatible_but_missing_checks_are_not_claimed_as_verified(self):
        model = self.make_model()
        legacy = dict(betas=np.zeros(2),soma_lod="low",soma_identity_model="soma")
        receipt = check_calibration_model(model,legacy)
        self.assertEqual(receipt["status"],"LEGACY_PARTIAL")
        self.assertIn("soma_joint_names",receipt["unverified_fields"])
        self.assertIn("soma_version",receipt["unverified_fields"])
        self.assertIn("soma_native_config",receipt["unverified_fields"])
        self.assertIn("soma_rig_fingerprint_sha256",receipt["unverified_fields"])
        self.assertNotIn("soma_rig_fingerprint_sha256",receipt["verified_fields"])

    def test_stored_joint_order_version_config_or_rig_mismatch_is_rejected(self):
        model = self.make_model()
        changes = dict(soma_joint_names=np.array(model.joint_names)[::-1],soma_version="other-version",
            soma_native_config="procedural_default",soma_enable_procedural_transforms=True,
            soma_apply_correctives=True,soma_correctives_model_loaded=True,
            soma_rig_fingerprint_sha256="0"*64,soma_rig_fingerprint_schema="other-schema")
        for key,value in changes.items():
            with self.subTest(key=key),self.assertRaises(ValueError):
                calibration = self.calibration(model); calibration[key] = value
                check_calibration_model(model,calibration)
        with self.assertRaisesRegex(ValueError,"boolean"):
            calibration=self.calibration(model);calibration["soma_apply_correctives"]="False"
            check_calibration_model(model,calibration)

    def test_different_loaded_static_weights_change_fingerprint_and_fail_pairing(self):
        original = self.make_model()
        changed = self.make_model(weight_change=.01)
        with self.assertRaisesRegex(ValueError,"soma_rig_fingerprint_sha256"):
            check_calibration_model(changed,self.calibration(original))

    def test_fingerprint_has_canonical_dtype_and_byte_order(self):
        weights=np.array([[.25,.75],[.5,.5]],dtype=np.float32)
        parents=np.array([-1,0],dtype=np.int64)
        faces=np.array([[0,1,0]],dtype=np.int64)
        orient=np.tile(np.eye(3,dtype=np.float32),(2,1,1));names=["Hips","Head"]
        expected=_rig_fingerprint(weights,parents,faces,orient,names)
        actual=_rig_fingerprint(weights.astype(">f8"),parents.astype(">i8"),faces.astype(">i8"),orient.astype(">f8"),names)
        self.assertEqual(expected,actual)

    def test_constructor_refuses_a_default_procedural_or_loaded_corrective_layer(self):
        layer=FakeLayer();layer.procedural_transforms_enabled=True
        with self.assertRaisesRegex(ValueError,"enable_procedural_transforms=False"):
            SomaBodyModel(layer,num_betas=2)
        layer=FakeLayer();layer.correctives_model=object()
        with self.assertRaisesRegex(ValueError,"correctives_model_path=None"):
            SomaBodyModel(layer,num_betas=2)


if __name__ == "__main__":
    unittest.main()
