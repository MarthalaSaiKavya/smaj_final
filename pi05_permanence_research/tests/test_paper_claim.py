import unittest

import numpy as np

import controlled_contrasts as cc
import paper_claim as pc


class MaskAndResidue(unittest.TestCase):
    def test_uncapped_mask_keeps_a_change_larger_than_twenty_percent(self):
        before = np.zeros((40, 40, 3), dtype=np.uint8)
        after = before.copy()
        after[:20, :] = 255
        self.assertIsNone(cc.masks_from_change(before, after))
        kept = pc.uncapped_change_mask(before, after)
        self.assertIsNotNone(kept)
        self.assertEqual(int(kept.sum()), 800)

    def test_painting_the_residue_makes_the_images_match(self):
        present = np.zeros((8, 8, 3), dtype=np.uint8)
        absent = present.copy()
        absent[1, 1] = (90, 0, 0)
        mask = cc.changed_pixels(present, absent)
        left = pc.paint_camera(present, mask)
        right = pc.paint_camera(absent, mask)
        count, peak = pc.diff_stats(left, right)
        self.assertEqual(count, 0)
        self.assertEqual(peak, 0)

    def test_a_bright_residue_is_named_as_visible(self):
        self.assertEqual(pc.residue_reading(20, 98, 0.38), "visible residue remains")

    def test_matched_images_with_a_moving_action_stay_separate_from_a_match(self):
        self.assertEqual(pc.residue_reading(0, 0, 0.48), "images match and the action still moves")
        self.assertEqual(pc.residue_reading(0, 0, 0.0), "images match and the action matches")


class OccluderGeometry(unittest.TestCase):
    def test_point_on_ray_sits_between_the_camera_and_the_bowl(self):
        camera = np.array([0.0, 0.0, 0.0])
        bowl = np.array([0.0, 0.0, 1.0])
        point = pc.point_on_ray(camera, bowl, 0.62)
        self.assertAlmostEqual(float(point[2]), 0.62)

    def test_a_center_on_the_ray_counts_as_in_front(self):
        camera = np.zeros(3)
        bowl = np.array([0.0, 0.0, 1.0])
        front = np.array([0.0, 0.0, 0.6])
        depth, ratio = pc.covering_ratio(camera, front, bowl, 0.05)
        self.assertGreater(depth, 0.0)
        self.assertLess(ratio, 0.05)

    def test_ramekin_is_preferred_over_another_bowl(self):
        chosen = pc.prefer_occluder([(1, "akita black bowl"), (2, "ramekin"), (3, "plate")], 1)
        self.assertEqual(chosen, 2)

    def test_a_placement_that_covers_the_whole_scene_loses_to_a_tighter_one(self):
        wide = {"agent_cover": 1.0, "scene_fraction": 0.9}
        tight = {"agent_cover": 0.6, "scene_fraction": 0.2}
        self.assertIs(pc.choose_placement([wide, tight]), tight)

    def test_task_count_accepts_a_property_or_a_method(self):
        class PropertySuite:
            n_tasks = 10

        class MethodSuite:
            def n_tasks(self):
                return 4

        self.assertEqual(pc.task_count(PropertySuite()), 10)
        self.assertEqual(pc.task_count(MethodSuite()), 4)

    def test_overlap_reports_only_the_ids_that_passed_again(self):
        self.assertEqual(pc.overlap([498, 12], pc.FROZEN_PAPER), [498])
        self.assertEqual(pc.overlap([1, 2], pc.FROZEN_STRICT), [])


if __name__ == "__main__":
    unittest.main()
