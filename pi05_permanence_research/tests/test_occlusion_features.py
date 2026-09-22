import unittest

import numpy as np

import occlusion_features as oc


def scores_from(occlusion, slab, color, absence) -> dict[str, np.ndarray]:
    return {
        "occlusion": np.asarray(occlusion, dtype=np.float64),
        "slab": np.asarray(slab, dtype=np.float64),
        "color": np.asarray(color, dtype=np.float64),
        "absence": np.asarray(absence, dtype=np.float64),
    }


class SpecificFeatures(unittest.TestCase):
    def test_keeps_a_feature_that_rises_only_for_occlusion(self):
        # Ten quiet features, then one occlusion feature above the 90th percentile.
        occlusion = np.array([0.01] * 10 + [5.0])
        quiet = np.array([0.01] * 11)
        found = oc.specific_ids(
            scores_from(occlusion, quiet, quiet, quiet),
            "occlusion",
            oc.QUIET_FOR["occlusion"],
            0.5,
        )
        self.assertEqual(found.tolist(), [10])

    def test_drops_a_feature_that_also_rises_for_paint_off_the_bowl(self):
        occlusion = np.array([0.01] * 10 + [5.0])
        slab = np.array([0.01] * 10 + [5.0])
        quiet = np.array([0.01] * 11)
        found = oc.specific_ids(
            scores_from(occlusion, slab, quiet, quiet),
            "occlusion",
            oc.QUIET_FOR["occlusion"],
            0.5,
        )
        self.assertEqual(found.tolist(), [])

    def test_drops_a_feature_that_falls_instead_of_rising(self):
        occlusion = np.array([0.0] * 10 + [-5.0])
        quiet = np.zeros(11)
        found = oc.specific_ids(
            scores_from(occlusion, quiet, quiet, quiet),
            "occlusion",
            oc.QUIET_FOR["occlusion"],
            0.5,
        )
        self.assertEqual(found.tolist(), [])

    def test_empty_result_is_not_replaced_by_the_largest_scores(self):
        occlusion = np.linspace(0.0, 1.0, 20)
        found = oc.specific_ids(
            scores_from(occlusion, occlusion, occlusion, occlusion),
            "occlusion",
            oc.QUIET_FOR["occlusion"],
            0.5,
        )
        self.assertEqual(found.size, 0)
        self.assertEqual(oc.prefix_counts(0, [8, 32, 128]), [])

    def test_prefixes_stop_at_the_number_that_passed(self):
        self.assertEqual(oc.prefix_counts(5, [8, 32, 128]), [5])
        self.assertEqual(oc.prefix_counts(40, [8, 32, 128]), [8, 32, 40])


class OcclusionVerdict(unittest.TestCase):
    def test_names_the_smallest_set_that_beats_both_controls(self):
        rows = [
            {"count": 8, "patch": "occlusion_features", "mean_gap_closed": 0.12},
            {"count": 8, "patch": "random_remap", "mean_gap_closed": 0.08},
            {"count": 8, "patch": "color_features", "mean_gap_closed": 0.04},
            {"count": 32, "patch": "occlusion_features", "mean_gap_closed": 0.40},
            {"count": 32, "patch": "random_remap", "mean_gap_closed": 0.10},
            {"count": 32, "patch": "color_features", "mean_gap_closed": 0.05},
        ]
        text = oc.occlusion_verdict(rows, [8, 32], 32)
        self.assertIn("Smallest occlusion set", text)
        self.assertIn("32", text)
        self.assertNotIn(": 8.", text)

    def test_says_when_nothing_passed_the_filter(self):
        text = oc.occlusion_verdict([], [], 0)
        self.assertIn("No occlusion feature passed", text)

    def test_says_when_a_set_loses_to_a_control(self):
        rows = [
            {"count": 8, "patch": "occlusion_features", "mean_gap_closed": 0.20},
            {"count": 8, "patch": "random_remap", "mean_gap_closed": 0.18},
            {"count": 8, "patch": "color_features", "mean_gap_closed": 0.02},
        ]
        text = oc.occlusion_verdict(rows, [8], 8)
        self.assertIn("No occlusion set beat both controls", text)


if __name__ == "__main__":
    unittest.main()
