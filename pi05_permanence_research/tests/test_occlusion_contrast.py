"""Pure checks for the action-expert occlusion contrast."""

import unittest

import numpy as np

import occlusion_contrast as oc


class OcclusionContrastTests(unittest.TestCase):
    def test_feature_id_uses_the_requested_form(self):
        self.assertEqual(oc.user_feature_id(12, 1.0, 7584), "L12/tau1.F7584")
        self.assertEqual(oc.tracer_feature_key(12, 1.0, 7584), "L12:tau1:F7584")

    def test_a_frame_is_dropped_when_contact_or_the_angle_is_missing(self):
        self.assertIsNone(oc.label_from_measurement(False, False, False, False))
        self.assertEqual(oc.label_from_measurement(True, False, False, False), "not_held")
        self.assertIsNone(oc.label_from_measurement(True, True, False, False))
        self.assertEqual(oc.label_from_measurement(True, True, True, True), "held_hidden")
        self.assertEqual(oc.label_from_measurement(True, True, True, False), "held_visible")

    def test_moving_the_object_does_not_count_as_an_arm_change(self):
        before = np.arange(12, dtype=np.float64)
        after = before.copy()
        after[4:11] = 9
        self.assertTrue(oc.arm_unchanged(oc.arm_joints(before, (4, 11)), oc.arm_joints(after, (4, 11))))
        after[2] = 50
        self.assertFalse(oc.arm_unchanged(oc.arm_joints(before, (4, 11)), oc.arm_joints(after, (4, 11))))

    def test_the_grasp_feature_is_the_one_whose_top_frames_are_held(self):
        chosen = oc.choose_grasp(
            [
                {"rank": 1, "held_top": 1, "top_count": 5, "feature": 1},
                {"rank": 4, "held_top": 4, "top_count": 5, "feature": 9},
                {"rank": 2, "held_top": 0, "top_count": 5, "feature": 3},
            ]
        )
        self.assertEqual(chosen["feature"], 9)
        self.assertIsNone(oc.choose_grasp([{"rank": 1, "held_top": 0, "top_count": 3}]))

    def test_demo_index_is_the_rank_inside_one_task(self):
        rows = [
            {"episode_index": 0, "task": "pick the bowl"},
            {"episode_index": 1, "task": "open the drawer"},
            {"episode_index": 2, "task": "pick the bowl"},
        ]
        self.assertEqual(oc.demo_index(rows, 2), 1)

    def test_the_feature_must_fire_for_both_held_classes_and_drop_when_gone(self):
        self.assertTrue(oc.occlusion_passes(1.0, 0.8, 0.1, False))
        self.assertFalse(oc.occlusion_passes(1.0, 0.8, 0.1, True))
        self.assertFalse(oc.occlusion_passes(1.0, 0.0, 0.0, False))
        self.assertFalse(oc.occlusion_passes(1.0, 1.0, 0.5, False))
        self.assertFalse(oc.occlusion_passes(1.0, None, 0.0, False))

    def test_a_gripper_line_fails_and_an_object_indicator_does_not(self):
        grippers = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
        held = np.array([False, False, False, True, True, True])
        self.assertTrue(oc.gripper_correlation_failed(grippers, grippers, held))
        scores = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        mixed = np.array([0.1, 0.2, 0.9, 0.15, 0.8, 0.85])
        mixed_held = np.array([True, False, True, False, True, False])
        self.assertFalse(oc.gripper_correlation_failed(scores, mixed, mixed_held))

    def test_the_control_feature_is_close_in_firing_rate_and_not_the_target(self):
        frequencies = np.array([0.01, 0.5, 0.49, 0.02])
        chosen = oc.similar_feature(frequencies, 1, seed=0)
        self.assertEqual(chosen, 2)

    def test_the_real_feature_counts_only_when_its_action_moves_more(self):
        self.assertTrue(oc.real_feature_counts(0.2, 0.05))
        self.assertFalse(oc.real_feature_counts(0.05, 0.2))

    def test_the_circuit_path_follows_the_strongest_edge_to_the_target(self):
        graph = {
            "config": {"target": "L12:tau1:F7"},
            "nodes": [
                {"node_key": "L02:tau1:F1", "layer": 2, "timestep": 1.0, "feature": 1, "depth": 2},
                {"node_key": "L08:tau1:F4", "layer": 8, "timestep": 1.0, "feature": 4, "depth": 1},
                {"node_key": "L12:tau1:F7", "layer": 12, "timestep": 1.0, "feature": 7, "depth": 0},
            ],
            "edges": [
                {"source_key": "L02:tau1:F1", "target_key": "L08:tau1:F4", "edge_score": 0.2},
                {"source_key": "L08:tau1:F4", "target_key": "L12:tau1:F7", "edge_score": 0.9},
                {"source_key": "L02:tau1:F1", "target_key": "L12:tau1:F7", "edge_score": 0.1},
            ],
        }
        self.assertEqual(oc.circuit_path(graph), "L2/tau1.F1 -> L8/tau1.F4 -> L12/tau1.F7")

    def test_a_failed_contrast_uses_the_required_sentence_and_skips_the_circuit(self):
        document = oc.verdict_document(
            passed=False,
            feature_id="L3/tau0.5.F9",
            firing_rates={"held_visible": 0.2, "held_hidden": None, "gone": 0.2},
            gripper_correlation_failed_flag=False,
            circuit_path_text="should not be kept",
            real_delta=1.0,
            random_delta=0.0,
        )
        self.assertEqual(document["verdict"], oc.FAIL_SENTENCE)
        self.assertIsNone(document["circuit_path"])
        self.assertIsNone(document["real_versus_random_action_delta"])
        self.assertEqual(document["feature_id"], "L3/tau0.5.F9")

    def test_candidate_json_is_unwrapped(self):
        rows = oc._candidate_rows({"candidates": [{"feature": 4}]})
        self.assertEqual(rows, [{"feature": 4}])


if __name__ == "__main__":
    unittest.main()
