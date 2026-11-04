"""Check project finger semantics and native joint ownership without model assets."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vsmpl.hand_regions import finger_vertex_regions


class FakeSmplx:
    def __init__(self):
        self.num_joints = 55
        self.num_betas = 10
        self.joints = np.zeros((55, 3))
        self.vertices = []
        self.weights = []
        # Different fingers intentionally share some positions: anatomy comes
        # from native skinning ownership, not an arbitrary world-space axis.
        for joint in range(25, 55):
            self.joints[joint] = [(.02 if joint < 40 else -.02) * ((joint - 25) % 3 + 1), 0., 0.]
            for offset in (0., .001, .002):
                self.vertices.append(self.joints[joint] + [0., offset, 0.])
                row = np.zeros(55)
                row[joint] = 1.
                self.weights.append(row)
        # A wrist-dominated boundary must stay out, even with a large finger weight.
        for wrist, finger in ((20, 25), (21, 40)):
            self.vertices.append([0., 0., 0.])
            row = np.zeros(55)
            row[wrist], row[finger] = .51, .49
            self.weights.append(row)
        self.vertices = np.array(self.vertices)
        self.weights = np.array(self.weights)
        self.last_shape = None

    def shaped(self, betas):
        self.last_shape = betas
        return self.vertices + betas[0] * .001, self.joints + betas[0] * .001


class HandRegionTests(unittest.TestCase):
    def test_all_twenty_labels_have_correct_side_finger_and_segment(self):
        model = FakeSmplx()
        expected = {
            "LIDX3": (25, 26), "LIDX6": (27,), "LMID0": (28, 29), "LMID6": (30,),
            "LPNK3": (31, 32), "LPNK6": (33,), "LRNG3": (34, 35), "LRNG6": (36,),
            "LTHM3": (37, 38), "LTHM6": (39,),
            "RIDX3": (40, 41), "RIDX6": (42,), "RMID0": (43, 44), "RMID6": (45,),
            "RPNK3": (46, 47), "RPNK6": (48,), "RRNG3": (49, 50), "RRNG6": (51,),
            "RTHM3": (52, 53), "RTHM6": (54,),
        }
        regions = finger_vertex_regions(model, np.zeros(10), list(expected))
        self.assertEqual(list(regions), list(expected))
        owners = model.weights.argmax(1)
        for label, joints in expected.items():
            self.assertEqual(set(owners[regions[label]].tolist()), set(joints))
        labels = list(regions)
        for i, label in enumerate(labels):
            for other in labels[i+1:]:
                self.assertFalse(set(regions[label]) & set(regions[other]), (label, other))
        self.assertFalse(any(len(model.vertices)-1 in region or len(model.vertices)-2 in region
                             for region in regions.values()))

    def test_shape_and_model_are_unchanged(self):
        model = FakeSmplx()
        beta = np.linspace(-.2, .2, 10)
        original_beta = beta.copy()
        original_vertices = model.vertices.copy()
        original_joints = model.joints.copy()
        original_weights = model.weights.copy()
        result = finger_vertex_regions(model, beta, ["RTHM6", "LIDX3"])
        self.assertEqual(list(result), ["RTHM6", "LIDX3"])
        np.testing.assert_array_equal(beta, original_beta)
        np.testing.assert_array_equal(model.vertices, original_vertices)
        np.testing.assert_array_equal(model.joints, original_joints)
        np.testing.assert_array_equal(model.weights, original_weights)
        self.assertFalse(np.shares_memory(beta, model.last_shape))

    def test_unknown_duplicate_or_empty_label_selection_fails(self):
        model = FakeSmplx()
        for labels in (["LIDX1"], ["LMID3"], ["LIDX3", "LELB"], ["LIDX3", "LIDX3"], [], None, "LIDX3"):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                finger_vertex_regions(model, np.zeros(10), labels)

    def test_non_smplx_or_empty_native_region_fails(self):
        model = FakeSmplx()
        model.num_joints = 24
        with self.assertRaises(ValueError):
            finger_vertex_regions(model, np.zeros(10), ["LIDX3"])
        model.num_joints = 55
        mask = model.weights.argmax(1) == 27
        model.weights[mask] = 0
        model.weights[mask, 20] = 1
        with self.assertRaisesRegex(ValueError, "LIDX6"):
            finger_vertex_regions(model, np.zeros(10), ["LIDX6"])

    def test_invalid_shape_or_skinning_is_rejected(self):
        for beta in (np.zeros((1, 10)), np.array([np.nan]), np.array([])):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                finger_vertex_regions(FakeSmplx(), beta, ["LTHM3"])
        model = FakeSmplx()
        model.weights[0, 25] = np.nan
        with self.assertRaises(ValueError):
            finger_vertex_regions(model, np.zeros(10), ["LTHM3"])


if __name__ == "__main__":
    unittest.main()
