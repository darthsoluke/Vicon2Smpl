"""Static-pose semantics: A-pose is a pose, not a body-shape change or a T-pose anchor."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.synthetic import make_body, pick_markers
from vsmpl.body_model import MarkerRig
from vsmpl.calibration import static_reference_pose, fit_tpose, refine, assign_vertices, reassign
from vsmpl.rotations import exp_so3, log_so3


class StaticPoseTests(unittest.TestCase):
    def test_attachment_region_prevents_a_leg_marker_moving_to_a_hand(self):
        model, _ = make_body(n_rings=5, n_sec=8)
        owners = model.weights.argmax(1)
        leg = np.flatnonzero(owners == 4)
        hand = int(np.flatnonzero(owners == 22)[0])
        fit = dict(root=np.zeros(3), body=np.zeros((23, 3)), trans=np.zeros(3), betas=np.zeros(10))
        obs = model.v_template[[hand]]
        vids, _ = assign_vertices(model, fit, obs, np.ones(1, bool), 0, regions=[leg])
        self.assertEqual(owners[vids[0]], 4)
        new, _ = reassign(model, vids, obs-model.v_template[vids], np.zeros(10), 0, regions=[leg])
        self.assertEqual(owners[new[0]], 4)

    def test_a_pose_lowers_both_arms_without_changing_shape(self):
        model, _ = make_body()
        v, j = model.shaped(np.zeros(10))
        pose = static_reference_pose(24, "a", 60)
        posed = model.joints(exp_so3(pose), np.zeros(3), j)
        self.assertLess(posed[20, 1], posed[16, 1] - .3)
        self.assertLess(posed[21, 1], posed[17, 1] - .3)
        np.testing.assert_array_equal(static_reference_pose(24, "t"), np.zeros((24, 3)))
        np.testing.assert_array_equal(v, model.shaped(np.zeros(10))[0])

    def test_known_layout_static_fit_accepts_relaxed_a_pose(self):
        model, _ = make_body(n_rings=5, n_sec=8)
        vids = pick_markers(model)
        pose = static_reference_pose(24, "a", 60)
        v, _ = model.shaped(np.zeros(10))
        rig = MarkerRig(model, [str(i) for i in vids], vids,
                        .0095 * model.vertex_normals(v)[vids], np.zeros(10))
        obs = rig.forward(exp_so3(pose), np.array([0., 1., 0.]))
        fit = fit_tpose(model, obs, np.ones(len(vids), bool), 0, vids=vids, iters=350,
                        reference_pose=static_reference_pose(24, "a", 45))
        self.assertLess(fit["rms"], .01)
        self.assertLess(fit["body"][15, 2], -.7)
        self.assertGreater(fit["body"][16, 2], .7)

    def test_refinement_anchor_keeps_recorded_a_pose(self):
        model, _ = make_body(n_rings=5, n_sec=8)
        vids = pick_markers(model)
        pose = static_reference_pose(24, "a", 60)
        v, _ = model.shaped(np.zeros(10))
        off = .0095 * model.vertex_normals(v)[vids]
        rig = MarkerRig(model, [str(i) for i in vids], vids, off, np.zeros(10))
        R = np.tile(exp_so3(pose)[None], (3, 1, 1, 1))
        trans = np.tile([0., 1., 0.], (3, 1))
        obs = np.stack([rig.forward(r, t) for r, t in zip(R, trans)])
        _, _, refined, _, _ = refine(model, vids, off, np.zeros(10), obs,
            np.ones(obs.shape[:2], bool), R, trans, .0095, iters=30,
            anchor=0, anchor_pose=pose, verbose=False)
        recovered = log_so3(refined[0])
        np.testing.assert_allclose(recovered[[16, 17], 2], pose[[16, 17], 2], atol=.04)


if __name__ == "__main__":
    unittest.main()
