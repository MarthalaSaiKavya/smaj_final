import unittest

import numpy as np

import controlled_contrasts as cc
import remaining_gaps as rg


class BothCameras(unittest.TestCase):
    def test_a_placement_that_covers_both_cameras_beats_an_agent_only_cover(self):
        agent_only = {"agent_cover": 1.0, "wrist_cover": 0.0, "scene_fraction": 0.04}
        both = {"agent_cover": 0.91, "wrist_cover": 0.88, "scene_fraction": 0.22}
        wide = {"agent_cover": 1.0, "wrist_cover": 1.0, "scene_fraction": 0.9}
        self.assertIs(rg.choose_both([agent_only, wide, both]), both)

    def test_the_second_free_body_prefers_the_plate(self):
        chosen = rg.second_body([(1, "akita black bowl"), (2, "ramekin"), (4, "plate")], {1, 2})
        self.assertEqual(chosen, 4)

    def test_painting_both_masks_covers_both_cameras(self):
        base_agent = np.zeros((8, 8, 3), dtype=np.uint8)
        base_wrist = np.zeros((8, 8, 3), dtype=np.uint8)
        agent_mask = np.zeros((8, 8), dtype=bool)
        wrist_mask = np.zeros((8, 8), dtype=bool)
        agent_mask[0, :2] = True
        wrist_mask[:, 0] = True
        agent = cc.paint_mask(base_agent, agent_mask, cc.GRAY)
        wrist = cc.paint_mask(base_wrist, wrist_mask, cc.GRAY)
        scores = rg.view_scores(base_agent, base_wrist, agent, wrist, agent_mask, wrist_mask)
        self.assertAlmostEqual(scores["agent_cover"], 1.0)
        self.assertAlmostEqual(scores["wrist_cover"], 1.0)
        self.assertGreater(scores["scene_fraction"], 0.0)

    def test_a_composite_that_covers_both_views_is_named_as_two_renders(self):
        text = rg.both_reading(1.0, 1.0, 0.1, True)
        self.assertIn("two renders", text)


class Postprocessor(unittest.TestCase):
    def test_apply_post_uses_a_dict_that_scales_the_chunk(self):
        class Double:
            def __call__(self, batch):
                return {"action": batch["action"] * 2}

        chunk = np.arange(12, dtype=np.float32).reshape(4, 3)
        out = rg.apply_post(Double(), chunk)
        np.testing.assert_allclose(out, chunk * 2)

    def test_apply_post_falls_back_to_single_steps_when_a_chunk_is_rejected(self):
        class StepOnly:
            def __call__(self, value):
                action = value["action"] if isinstance(value, dict) else value
                if int(action.ndim) != 2:
                    raise RuntimeError("chunk rejected")
                return {"action": action * 3}

        chunk = np.ones((4, 3), dtype=np.float32)
        out = rg.apply_post(StepOnly(), chunk)
        np.testing.assert_allclose(out, chunk * 3)

    def test_apply_post_raises_when_every_attempt_fails(self):
        class Boom:
            def __call__(self, _value):
                raise RuntimeError("nope")

        with self.assertRaises(RuntimeError) as caught:
            rg.apply_post(Boom(), np.ones((2, 2), dtype=np.float32))
        self.assertIn("nope", str(caught.exception))

    def test_an_unchanged_scale_that_matches_zero_motion_is_named(self):
        text = rg.harness_reading(0.2, 0.2, 0.01, 0.01)
        self.assertIn("unchanged", text)
        self.assertIn("zero action", text)


if __name__ == "__main__":
    unittest.main()
