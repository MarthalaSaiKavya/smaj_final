import unittest

import numpy as np

import paper_gaps as pg


class PaperGapMath(unittest.TestCase):
    def test_bootstrap_interval_covers_the_sample_mean(self):
        rng = np.random.default_rng(0)
        stats = pg.bootstrap_interval([0.1, 0.2, 0.3, 0.4], rng, draws=400)
        self.assertEqual(stats["n"], 4)
        self.assertAlmostEqual(stats["mean"], 0.25)
        self.assertLessEqual(stats["low"], stats["mean"])
        self.assertGreaterEqual(stats["high"], stats["mean"])

    def test_zoom_box_pads_the_changed_pixels(self):
        mask = np.zeros((40, 50), dtype=bool)
        mask[10:12, 20:23] = True
        y0, y1, x0, x1 = pg.zoom_box(mask, 4, 40, 50)
        self.assertLessEqual(y0, 10)
        self.assertGreaterEqual(y1, 12)
        self.assertLessEqual(x0, 20)
        self.assertGreaterEqual(x1, 23)

    def test_spread_frames_takes_one_from_each_episode_first(self):
        rows = [
            {"episode": 0, "step": 1},
            {"episode": 0, "step": 2},
            {"episode": 3, "step": 4},
        ]
        picked = pg.spread_frames(rows, 2)
        self.assertEqual([(row["episode"], row["step"]) for row in picked], [(0, 1), (3, 4)])

    def test_a_falling_feature_is_rejected_even_when_it_is_large(self):
        reasons = pg.rejection_reasons(
            {"occlusion": -0.75, "slab": 0.03, "color": -0.13, "absence": 0.29, "specificity": 0.64},
            0.5,
        )
        self.assertIn("falls instead of rising", reasons)

    def test_claim_puts_the_ceiling_above_the_features(self):
        text = pg.claim_line(0.976, 0.93, 0.053)
        self.assertIn("0.976", text)
        self.assertIn("93.0%", text)
        self.assertIn("5.3%", text)

    def test_claim_states_when_no_feature_passed(self):
        text = pg.claim_line(0.976, 0.93, None)
        self.assertIn("no feature passed", text)


if __name__ == "__main__":
    unittest.main()
