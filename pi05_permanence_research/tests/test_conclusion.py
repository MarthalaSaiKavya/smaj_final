"""Pure checks for leftover bowl pixels after a tight hide."""

import unittest

import numpy as np

import conclusion as cl


class LeftoverSplit(unittest.TestCase):
    def test_leftover_is_bowl_pixels_the_edit_did_not_change(self):
        base = np.zeros((2, 3, 3), dtype=np.uint8)
        edited = np.zeros_like(base)
        edited[0, 0] = 10
        bowl = np.array([[True, True, False], [False, False, False]])
        row = cl.leftover_split(base, edited, bowl)
        self.assertEqual(row["leftover_pixels"], 1)
        self.assertTrue(row["leftover"][0, 1])
        self.assertFalse(row["leftover"][0, 0])
        self.assertEqual(row["extra_pixels"], 0)
        self.assertTrue(row["known"])

    def test_plate_extra_is_changed_pixels_outside_the_bowl(self):
        base = np.zeros((2, 2, 3), dtype=np.uint8)
        edited = np.zeros_like(base)
        edited[0, 1] = 20
        edited[1, 0] = 20
        bowl = np.array([[True, False], [False, False]])
        row = cl.leftover_split(base, edited, bowl)
        self.assertEqual(row["extra_pixels"], 2)
        self.assertEqual(row["leftover_pixels"], 1)

    def test_a_missing_bowl_mask_is_named_unknown(self):
        base = np.zeros((1, 1, 3), dtype=np.uint8)
        edited = np.array([[[9, 9, 9]]], dtype=np.uint8)
        row = cl.leftover_split(base, edited, None)
        self.assertFalse(row["known"])
        self.assertEqual(row["leftover_pixels"], 0)
        self.assertEqual(row["extra_pixels"], 1)


class HideReading(unittest.TestCase):
    def test_zero_leftover_names_the_plate_not_a_hidden_bowl(self):
        text = cl.hide_reading(0, 0, True, 0.0, 0.8, 0.01)
        self.assertIn("plate", text)
        self.assertIn("hidden", text)

    def test_a_rim_that_paints_to_the_cover_is_named(self):
        text = cl.hide_reading(0, 20, True, 0.0, 0.8, 0.2)
        self.assertIn("rim", text)
        self.assertIn("matches the full cover", text)

    def test_a_hide_that_matches_removal_is_named(self):
        text = cl.hide_reading(0, 0, True, 0.0, 0.001, 0.001)
        self.assertEqual(text, "The hide matches removal.")

    def test_the_claim_keeps_the_closed_loop_rates(self):
        table = {"reading": "The bowl is hidden as fully as gray paint."}
        held = {"reading": "A visible bowl rim remains. Painting it matches the full cover."}
        rates = [
            {"edit": "base", "successes": 10, "episodes": 10},
            {"edit": "physical", "successes": 0, "episodes": 10},
        ]
        text = cl.final_claim(table, held, rates)
        self.assertIn("base 10/10", text)
        self.assertIn("physical 0/10", text)
        self.assertIn("1.8%", text)


class Placements(unittest.TestCase):
    def test_summary_placements_override_the_logged_defaults(self):
        found = cl.placements_from_summary(
            {"frames": [{"episode": 1, "step": 0, "tight": {"fraction": 0.55, "scale": 1.0}}]}
        )
        self.assertEqual(found[(1, 0)]["fraction"], 0.55)
        self.assertEqual(found[(1, 44)]["scale"], 0.75)


if __name__ == "__main__":
    unittest.main()
