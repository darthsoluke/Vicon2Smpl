"""Provenance/subject gates and real unlocked replay, without licensed model files.

Only the expensive attachment optimisation is stubbed. Native BodyModel,
MarkerRig, HandPCA loading and PCASolver replay are exercised together.
"""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import calibrate_head
from tests.test_head_calibration import parents55
from vsmpl.body_model import BodyModel, MarkerRig
from vsmpl.head_calibration import HEAD_LABELS, SHOULDER_LABELS
from vsmpl.rotations import exp_so3


class HeadCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        rng = np.random.default_rng(3)
        vertices = rng.uniform([-.3,.6,-.15],[.3,1.4,.15],size=(72,3))
        vertices[:8] = [[.08,1.72,.06],[.08,1.68,-.08],[-.08,1.72,.06],[-.08,1.68,-.08],
                        [.22,1.4,.04],[.22,1.4,-.04],[-.22,1.4,.04],[-.22,1.4,-.04]]
        vertices[8:11] = [[0,1.2,0],[0,1.58,0],[0,1.45,0]]
        regressor = np.zeros((55,72)); regressor[:,8] = 1
        regressor[12] = 0; regressor[12,10] = 1
        regressor[15] = 0; regressor[15,9] = 1
        weights = np.zeros((72,55)); weights[:,9] = 1
        weights[:4] = 0; weights[:4,15] = 1
        self.model = BodyModel(vertices,np.zeros((72,3,1)),np.zeros((54*9,72,3)),
            regressor,weights,parents55(),np.array([[0,1,2]]),"smplx")
        self.model_path = self.directory/"native_model.npz"
        np.savez(self.model_path,v_template=vertices,shapedirs=self.model.shapedirs,
            posedirs=self.model.posedirs,J_regressor=regressor,weights=weights,
            kintree_table=np.stack((self.model.parents,np.arange(55))),f=self.model.faces)
        self.names = np.array(HEAD_LABELS+SHOULDER_LABELS+tuple(f"BODY{i}" for i in range(64)))
        self.prior_path = self.directory/"prior.npz"
        samples = np.tile(np.arange(4.)[:,None],(1,45))
        np.savez(self.prior_path,componentsl=np.eye(45),componentsr=np.eye(45),
            hands_meanl=np.zeros(45),hands_meanr=np.zeros(45),hands_coeffsl=samples,hands_coeffsr=samples)
        self.parent = dict(marker_names=self.names,vertex_ids=np.arange(72),offsets=np.zeros((72,3)),
            betas=np.zeros(1),model_type="smplx",gender="neutral",num_betas=1,source_subject="actor",
            marker_scope="body_hands",finger_marker_names=self.names[-20:],hand_prior_source=str(self.prior_path),
            hand_prior_sha256=hashlib.sha256(self.prior_path.read_bytes()).hexdigest(),hand_pca_components=1,
            hand_pca_initial_latent=np.zeros((2,1)),hand_prior_weight=1e-5)
        self.parent_path = self.directory/"parent.npz"
        np.savez(self.parent_path,**self.parent)
        self.reference_path = self.directory/"matched_reference.npz"
        self.write_reference()
        self.output = self.directory/"new_head_candidate"

    def write_reference(self, *, subject="actor", tilt=0.):
        R = np.tile(np.eye(3),(3,55,1,1))
        R[:,15] = exp_so3([0.,0.,np.radians(tilt)])
        rig = MarkerRig.from_calibration(self.model,self.parent)
        positions = np.stack([rig.forward(rotation,np.zeros(3)) for rotation in R])
        np.savez(self.reference_path,names=self.names,pos=positions,valid=np.ones((3,72),bool),
            R=R,trans=np.zeros((3,3)),source_subject=subject,reference_times=np.arange(3.)*.01)

    def arguments(self):
        return calibrate_head._parser().parse_args([
            "--model-path",str(self.model_path),"--parent-calibration",str(self.parent_path),
            "--reference",str(self.reference_path),"--output-dir",str(self.output),
            "--subject","actor","--operator-confirmed-neutral"])

    @staticmethod
    def fake_calibration(model,parent,rec,R,trans,**kwargs):
        candidate = {key:np.asarray(value).copy() for key,value in parent.items()}
        candidate.update(head_reference_sample_indices=np.arange(len(R)),
            head_neutral_reference_R=np.tile(np.eye(3),(len(R),55,1,1)),
            head_neutral_reference_trans=trans.copy(),head_neutral_reference_confirmed=True,
            head_online_pose_locked=False)
        return candidate,dict(quality_warnings=[],imposed_neutral_orientation={"up_deviation_degrees":[0.,0.,0.]})

    def run_fixture(self):
        with patch.object(calibrate_head,"calibrate_neutral_head",side_effect=self.fake_calibration),redirect_stdout(io.StringIO()):
            return calibrate_head.run(self.arguments())

    def test_mismatched_subject_is_rejected_before_fitting_or_output(self):
        self.write_reference(subject="different_actor")
        with patch.object(calibrate_head,"calibrate_neutral_head") as fit, self.assertRaisesRegex(ValueError,"same subject"):
            calibrate_head.run(self.arguments())
        fit.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_prior_sha_mismatch_is_rejected_before_fitting(self):
        self.prior_path.write_bytes(self.prior_path.read_bytes()+b"changed")
        with patch.object(calibrate_head,"calibrate_neutral_head") as fit,self.assertRaisesRegex(ValueError,"SHA256"):
            calibrate_head.run(self.arguments())
        fit.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir(); sentinel=self.output/"previous.txt"; sentinel.write_text("keep")
        with self.assertRaisesRegex(ValueError,"new directory"):
            calibrate_head.run(self.arguments())
        self.assertEqual(sentinel.read_text(),"keep")

    def test_missing_native_poses_are_rejected(self):
        reference = calibrate_head._load_npz(self.reference_path)
        del reference["R"]
        np.savez(self.reference_path,**reference)
        with self.assertRaisesRegex(ValueError,"missing fields.*R"):
            calibrate_head.run(self.arguments())

    def test_operator_confirmation_cannot_be_omitted_programmatically(self):
        args=self.arguments();args.operator_confirmed_neutral=False
        with self.assertRaisesRegex(ValueError,"Operator-confirmed"):
            calibrate_head.run(args)
        self.assertFalse(self.output.exists())

    def test_native_unlocked_replay_and_source_receipts_remain_review_pending(self):
        before_parent=self.parent_path.read_bytes();before_reference=self.reference_path.read_bytes()
        self.assertEqual(self.run_fixture(),0)
        report=json.loads((self.output/"head_calibration_report.json").read_text())
        self.assertEqual(report["status"],"NUMERIC_PASS_REVIEW_PENDING")
        self.assertEqual(report["numeric_quality"]["status"],"PASS")
        self.assertIn("PENDING_VISUAL_REVIEW",report["acceptance"])
        self.assertFalse(report["independent_ground_truth"])
        self.assertTrue(report["provenance"]["parent_and_reference_hashes_unchanged"])
        for key,replay in report["unlocked_replay"].items():
            self.assertEqual(replay["solver"],"PCASolver")
            self.assertTrue({12,15}.issubset(replay["active_joints"]))
            self.assertFalse({12,15}.intersection(replay["fixed_joints"]))
            self.assertFalse(replay["online_head_locked"])
            self.assertLess(replay["head"]["rms_mm"],1e-6)
            self.assertTrue((self.output/("replay_"+key+".npz")).exists())
        checkpoint=self.output/"subject_head_calibration.npz"
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),report["candidate_checkpoint"]["sha256"])
        with np.load(checkpoint,allow_pickle=False) as candidate:
            np.testing.assert_array_equal(candidate["betas"],self.parent["betas"])
            np.testing.assert_array_equal(candidate["vertex_ids"][4:],self.parent["vertex_ids"][4:])
            np.testing.assert_array_equal(candidate["offsets"][4:],self.parent["offsets"][4:])
            self.assertEqual(str(candidate["head_reference_source_subject"]),"actor")
        self.assertEqual(self.parent_path.read_bytes(),before_parent)
        self.assertEqual(self.reference_path.read_bytes(),before_reference)

    def test_zero_imposed_angle_does_not_pass_a_tilted_unlocked_replay(self):
        self.write_reference(tilt=25.)
        self.assertEqual(self.run_fixture(),2)
        report=json.loads((self.output/"head_calibration_report.json").read_text())
        self.assertEqual(report["status"],"CANDIDATE_FAIL")
        self.assertEqual(report["calibration"]["imposed_neutral_orientation"]["up_deviation_degrees"],[0.,0.,0.])
        self.assertGreater(report["unlocked_replay"]["candidate_matched_pose"]["up_deviation_p95_degrees"],5.)
        self.assertTrue((self.output/"subject_head_calibration.npz").exists())
        self.assertTrue(report["numeric_quality"]["failures"])

    def test_replay_error_retains_checkpoint_and_error_report(self):
        with patch.object(calibrate_head,"calibrate_neutral_head",side_effect=self.fake_calibration), \
                patch.object(calibrate_head,"_replay",side_effect=ValueError("replay diagnostic failure")), \
                self.assertRaisesRegex(ValueError,"replay diagnostic failure"):
            calibrate_head.run(self.arguments())
        report=json.loads((self.output/"head_calibration_report.json").read_text())
        self.assertEqual(report["status"],"CANDIDATE_ERROR")
        self.assertEqual(report["error"]["message"],"replay diagnostic failure")
        self.assertTrue((self.output/"subject_head_calibration.npz").exists())

    def test_protected_binding_changes_fail_closed_and_are_retained_for_diagnosis(self):
        def corrupt_candidate(*args,**kwargs):
            candidate,report=self.fake_calibration(*args,**kwargs)
            candidate["offsets"][10,0] += .001
            return candidate,report
        with patch.object(calibrate_head,"calibrate_neutral_head",side_effect=corrupt_candidate), \
                self.assertRaisesRegex(ValueError,"protected offsets"):
            calibrate_head.run(self.arguments())
        report=json.loads((self.output/"head_calibration_report.json").read_text())
        self.assertEqual(report["status"],"CANDIDATE_ERROR")
        self.assertTrue((self.output/"subject_head_calibration.npz").exists())


if __name__ == "__main__":
    unittest.main()
