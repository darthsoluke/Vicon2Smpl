"""Neutral head constraints preserve the native chain; invalid references fail closed."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vsmpl.body_model import BodyModel
from vsmpl.head_calibration import calibrate_neutral_head, neutral_head_rotations, head_vertex_regions, HEAD_LABELS, SHOULDER_LABELS
from vsmpl.rotations import exp_so3


def parents55():
    parents = np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,15,15,15]
                       + [0]*30, dtype=np.int64)
    for wrist, start in ((20,25),(21,40)):
        for first in range(start,start+15,3):
            parents[first:first+3] = [wrist,first,first+1]
    return parents


def model55():
    vertices = np.zeros((55,3))
    vertices[:,1] = np.arange(55)*.01
    return BodyModel(vertices,np.zeros((55,3,1)),np.zeros((54*9,55,3)),
        np.eye(55),np.eye(55),parents55(),np.array([[0,1,2]]),"smplx")


class HeadCalibrationTests(unittest.TestCase):
    def test_head_regions_allow_low_posterior_and_temple_band_but_keep_native_head_ownership(self):
        vertices = np.array([
            [.05,.1,.06],[-.05,.1,.06],[.05,.1,-.07],[-.05,.1,-.07],
            [.05,.025,-.07],[-.05,.025,-.07],
            [.06,.05,-.02],[-.06,.05,-.02],
            [.06,0.,0.],[-.06,0.,0.],
            [.06,.08,.01],
        ])
        weights = np.zeros((len(vertices),55));weights[:,15] = 1.
        weights[-1,15],weights[-1,12] = .97,.03
        model = SimpleNamespace(num_joints=55,weights=weights,
            shaped=lambda beta:(vertices.copy(),np.zeros((55,3))))
        beta=np.zeros(1);before_vertices=vertices.copy();before_weights=weights.copy()
        regions,report=head_vertex_regions(model,beta)
        self.assertIn(4,regions["LBHD"])
        self.assertIn(5,regions["RBHD"])
        self.assertIn(6,regions["LFHD"])
        self.assertIn(7,regions["RFHD"])
        self.assertNotIn(4,regions["LFHD"])
        self.assertNotIn(5,regions["RFHD"])
        for ids in regions.values():
            self.assertNotIn(10,ids)
            self.assertNotIn(8,ids);self.assertNotIn(9,ids)
        self.assertEqual(report["front_back_overlap_half_width_m"],.025)
        np.testing.assert_array_equal(vertices,before_vertices)
        np.testing.assert_array_equal(weights,before_weights)
        np.testing.assert_array_equal(beta,np.zeros(1))

    def test_neutral_head_world_rotation_with_arbitrary_parent_and_delta(self):
        model = model55()
        rng = np.random.default_rng(8)
        source = exp_so3(rng.normal(size=(4,55,3))*.4)
        desired = exp_so3(rng.normal(size=(4,3))*.7)
        original = source.copy()
        result = neutral_head_rotations(source,desired,np.array([.22,-.17,.31]),model.parents)
        for frame in range(4):
            world,_ = model.fk(result[frame],np.zeros((55,3)))
            np.testing.assert_allclose(world[15],desired[frame],atol=2e-14)
        keep = [j for j in range(55) if j not in (12,15)]
        np.testing.assert_array_equal(result[:,keep],source[:,keep])
        np.testing.assert_array_equal(source,original)

    def test_single_frame_upright_target_removes_a_tilt_without_changing_root(self):
        model = model55()
        source = np.tile(np.eye(3),(55,1,1))
        source[0] = exp_so3([0.,-1.3,0.])
        source[15] = exp_so3([.4,0.,.6])
        desired = exp_so3([0.,-.9,0.])
        result = neutral_head_rotations(source,desired,[.1,0.,-.2],model.parents)
        world,_ = model.fk(result,np.zeros((55,3)))
        np.testing.assert_allclose(world[15]@np.array([0.,1.,0.]),[0.,1.,0.],atol=2e-14)
        np.testing.assert_array_equal(result[0],source[0])

    def test_operator_confirmation_is_required_before_any_fitting(self):
        for value in (False,None,"yes",1):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,"operator-confirmed"):
                calibrate_neutral_head(None,{},None,None,None,operator_confirmed_neutral=value)

    def fixture(self):
        names = list(HEAD_LABELS+SHOULDER_LABELS)+[f"OTHER_{i}" for i in range(64)]
        rec = dict(names=names,pos=np.zeros((3,72,3)),valid=np.ones((3,72),bool))
        parent = dict(marker_names=np.array(names),vertex_ids=np.arange(72),offsets=np.zeros((72,3)),betas=np.zeros(1))
        model = SimpleNamespace(num_joints=55,parents=parents55(),v_template=np.zeros((100,3)))
        R = np.tile(np.eye(3),(3,55,1,1))
        return model,parent,rec,R,np.zeros((3,3))

    def test_missing_required_head_label_is_rejected(self):
        model,parent,rec,R,t = self.fixture()
        rec["names"][0] = "UNKNOWN_HEAD"
        with self.assertRaisesRegex(ValueError,"Missing required"):
            calibrate_neutral_head(model,parent,rec,R,t,operator_confirmed_neutral=True)

    def test_unknown_source_label_is_rejected(self):
        model,parent,rec,R,t = self.fixture()
        rec["names"][-1] = "UNKNOWN_BODY"
        with self.assertRaisesRegex(ValueError,"Unknown or missing"):
            calibrate_neutral_head(model,parent,rec,R,t,operator_confirmed_neutral=True)

    def test_mismatched_reference_pose_frame_count_is_rejected(self):
        model,parent,rec,R,t = self.fixture()
        with self.assertRaisesRegex(ValueError,"matching frame dimensions"):
            calibrate_neutral_head(model,parent,rec,R[:2],t,operator_confirmed_neutral=True)

    def test_incorrect_native_neck_hierarchy_is_rejected(self):
        model,parent,rec,R,t = self.fixture()
        model.parents[15] = 9
        with self.assertRaisesRegex(ValueError,"hierarchy"):
            calibrate_neutral_head(model,parent,rec,R,t,operator_confirmed_neutral=True)


if __name__ == "__main__":
    unittest.main()
