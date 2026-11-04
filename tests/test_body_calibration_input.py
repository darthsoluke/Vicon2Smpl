"""Body marker selection must retain the source take and its frame/time provenance."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vsmpl.calibration_input import prepare_calibration_input


class BodyCalibrationInputTests(unittest.TestCase):
    def make_recording(self):
        body = [f"BODY_{i:02d}" for i in range(52)]
        fingers = [f"FINGER_{i:02d}" for i in range(20)]
        names = np.array(body + fingers)[np.random.default_rng(12).permutation(72)].tolist()
        # Distinct coordinates and validity columns expose an accidental reorder of only one array.
        pos = np.arange(6 * 72 * 3, dtype=np.float64).reshape(6, 72, 3) / 1000
        valid = (np.arange(6)[:, None] + np.arange(72)[None]) % 3 != 0
        frames = np.array([101, 104, 108, 109, 114, 120], dtype=np.int64)
        recording = dict(names=names, pos=pos, valid=valid, fps=300.0, frames=frames,
                         relative_host_arrival_seconds=np.array([0., .01, .023, .027, .043, .063]),
                         subject="zhr", metadata={"take": "source-only"})
        regions = {name: [i, i + 100] for i, name in enumerate(body)}
        return recording, regions

    def assert_source_unchanged(self, original, before):
        self.assertEqual(original["names"], before["names"])
        for key in ("pos", "valid", "frames", "relative_host_arrival_seconds"):
            np.testing.assert_array_equal(original[key], before[key])
        self.assertEqual(original["fps"], before["fps"])
        self.assertEqual(original["metadata"], before["metadata"])
        self.assertEqual(original["subject"], before["subject"])

    def test_shuffled_72_markers_selects_only_52_body_columns(self):
        recording, regions = self.make_recording()
        before = copy.deepcopy(recording)
        regions_before = copy.deepcopy(regions)
        selected, selected_regions = prepare_calibration_input(recording, regions, num_vertices=200)

        indices = [i for i, name in enumerate(recording["names"]) if name in regions]
        self.assertEqual(len(selected["names"]), 52)
        self.assertEqual(selected["names"], [recording["names"][i] for i in indices])
        self.assertFalse(any(name.startswith("FINGER_") for name in selected["names"]))
        np.testing.assert_array_equal(selected["pos"], recording["pos"][:, indices])
        np.testing.assert_array_equal(selected["valid"], recording["valid"][:, indices])
        self.assertEqual(set(selected_regions), set(selected["names"]))
        for name in selected["names"]:
            np.testing.assert_array_equal(selected_regions[name], regions[name])
        self.assertEqual(selected["fps"], recording["fps"])
        self.assertEqual(selected["subject"], "zhr")
        self.assertEqual(selected["metadata"], recording["metadata"])
        np.testing.assert_array_equal(selected["frames"], recording["frames"])
        np.testing.assert_array_equal(selected["relative_host_arrival_seconds"],
                                      recording["relative_host_arrival_seconds"])
        self.assert_source_unchanged(recording, before)
        self.assertEqual(regions, regions_before)

    def test_none_keeps_full_recording_and_provenance(self):
        recording, _ = self.make_recording()
        before = copy.deepcopy(recording)
        selected, selected_regions = prepare_calibration_input(recording, None)
        self.assertIsNone(selected_regions)
        self.assertEqual(selected["names"], recording["names"])
        np.testing.assert_array_equal(selected["pos"], recording["pos"])
        np.testing.assert_array_equal(selected["valid"], recording["valid"])
        np.testing.assert_array_equal(selected["frames"], recording["frames"])
        np.testing.assert_array_equal(selected["relative_host_arrival_seconds"],
                                      recording["relative_host_arrival_seconds"])
        self.assertEqual(selected["fps"], recording["fps"])
        self.assert_source_unchanged(recording, before)

    def test_missing_required_body_label_fails_without_falling_back_to_all_markers(self):
        recording, regions = self.make_recording()
        recording["names"][recording["names"].index("BODY_07")] = "UNEXPECTED_LABEL"
        before = copy.deepcopy(recording)
        with self.assertRaises(ValueError):
            prepare_calibration_input(recording, regions, num_vertices=200)
        self.assert_source_unchanged(recording, before)

    def test_invalid_region_definitions_fail(self):
        cases = [
            ("empty mapping", {}, 200),
            ("empty vertex list", {"BODY_00": []}, 200),
            ("negative index", {"BODY_00": [-1, 2]}, 200),
            ("index equal to vertex count", {"BODY_00": [200]}, 200),
            ("index beyond vertex count", {"BODY_00": [201]}, 200),
        ]
        for label, regions, num_vertices in cases:
            with self.subTest(label=label):
                recording, _ = self.make_recording()
                before = copy.deepcopy(recording)
                with self.assertRaises(ValueError):
                    prepare_calibration_input(recording, regions, num_vertices=num_vertices)
                self.assert_source_unchanged(recording, before)

    def test_optional_vertex_count_does_not_invent_a_topology_limit(self):
        recording, _ = self.make_recording()
        selected, selected_regions = prepare_calibration_input(recording, {"BODY_00": [100000]})
        self.assertEqual(selected["names"], ["BODY_00"])
        np.testing.assert_array_equal(selected_regions["BODY_00"], [100000])

    def test_duplicate_live_labels_are_rejected(self):
        recording, regions = self.make_recording()
        recording["names"][recording["names"].index("FINGER_00")] = "BODY_00"
        before = copy.deepcopy(recording)
        with self.assertRaises(ValueError):
            prepare_calibration_input(recording, regions, num_vertices=200)
        self.assert_source_unchanged(recording, before)


if __name__ == "__main__":
    unittest.main()
