import unittest

import numpy as np

import token_position_patch as tp


class TokenSelection(unittest.TestCase):
    def test_fraction_parser_and_minimum_count(self):
        self.assertEqual(tp.parse_fractions("0.01,0.05,0.15,0.50"), [0.01, 0.05, 0.15, 0.50])
        self.assertEqual(tp.token_count(256, 0.01), 3)
        self.assertEqual(tp.token_count(10, 0.01), 1)
        self.assertEqual(tp.token_count(100, 0.50), 50)

    def test_largest_and_least_are_the_two_ends(self):
        source = np.zeros((5, 2), dtype=np.float32)
        donor = np.zeros((5, 2), dtype=np.float32)
        donor[3] = (3.0, 0.0)
        donor[1] = (0.0, 1.0)
        order, distance = tp.movement(source, donor)
        self.assertEqual(order[0], 3)
        self.assertEqual(int(np.argmax(distance)), 3)
        rng = np.random.default_rng(0)
        largest = tp.select_positions(order, 5, 1, "largest", rng)
        least = tp.select_positions(order, 5, 1, "least", rng)
        self.assertEqual(largest.tolist(), [3])
        self.assertEqual(least.tolist(), [4])
        self.assertEqual(len(set(largest.tolist()) & set(least.tolist())), 0)

    def test_delta_is_zero_outside_the_chosen_tokens(self):
        source = np.zeros((4, 3), dtype=np.float32)
        donor = np.arange(12, dtype=np.float32).reshape(4, 3)
        indexes = np.array([1, 3])
        delta = tp.position_delta(source, donor, indexes)
        self.assertTrue(np.array_equal(delta[1], donor[1]))
        self.assertTrue(np.array_equal(delta[3], donor[3]))
        self.assertEqual(float(np.abs(delta[0]).sum()), 0.0)
        self.assertEqual(float(np.abs(delta[2]).sum()), 0.0)
        covered = tp.position_delta(source, donor, np.arange(4))
        self.assertTrue(np.allclose(covered, donor - source))
        donor64 = donor.astype(np.float64)
        expected = float(np.linalg.norm(donor64[indexes]) / np.linalg.norm(donor64))
        self.assertAlmostEqual(tp.token_l2_share(source, donor, indexes), expected)


class PositionVerdict(unittest.TestCase):
    def test_a_small_head_of_tokens_is_localized(self):
        curve = [
            (0.01, 0.20, 0.02, 0.01),
            (0.05, 0.70, 0.05, 0.02),
            (0.15, 0.74, 0.10, 0.03),
            (0.50, 0.76, 0.20, 0.05),
        ]
        text = tp.position_verdict("color", curve, 0.76)
        self.assertIn("tokens that move most", text)
        self.assertIn("5%", text)

    def test_a_rising_curve_is_spread_across_tokens(self):
        curve = [
            (0.01, 0.04, 0.01, 0.00),
            (0.05, 0.12, 0.04, 0.01),
            (0.15, 0.35, 0.08, 0.02),
            (0.50, 0.70, 0.15, 0.04),
        ]
        text = tp.position_verdict("absence", curve, 0.80)
        self.assertIn("spread across the prefix", text)
        self.assertIn("12.0%", text)
        self.assertIn("70.0%", text)
