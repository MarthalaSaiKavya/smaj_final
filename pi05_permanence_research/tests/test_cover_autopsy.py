import unittest

import numpy as np

import cover_autopsy as ca


class CoverReading(unittest.TestCase):
    def test_a_large_gap_on_agreeing_images_is_not_a_small_residual(self):
        self.assertEqual(ca.cover_reading(0.00008, 0.55877), "images agree and the action still moves")

    def test_a_hundredth_of_an_rmse_on_agreeing_images_is_small(self):
        self.assertEqual(ca.cover_reading(0.0004, 0.011), "images agree and the action gap is small")

    def test_matching_actions_agree(self):
        self.assertEqual(ca.cover_reading(0.0001, 1e-4), "images agree and the action agrees")

    def test_window_rmse_uses_only_the_requested_prefix(self):
        source = np.zeros((6, 2))
        edited = np.zeros((6, 2))
        edited[4:] = 3.0
        self.assertEqual(ca.window_rmse(source, edited, 4), 0.0)
        self.assertGreater(ca.window_rmse(source, edited, 6), 0.0)

    def test_window_rmse_on_an_empty_tail_is_nan(self):
        source = np.zeros((4, 2))
        edited = np.ones((4, 2))
        self.assertTrue(np.isnan(ca.window_rmse(source[4:], edited[4:], 1)))

    def test_diff_stats_count_changed_pixels(self):
        before = np.zeros((4, 4, 3), dtype=np.uint8)
        after = before.copy()
        after[1, 2] = (5, 0, 0)
        count, peak = ca.diff_stats(before, after)
        self.assertEqual(count, 1)
        self.assertEqual(peak, 5)


if __name__ == "__main__":
    unittest.main()
