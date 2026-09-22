import unittest

import tight_cover as tc


class TightSelection(unittest.TestCase):
    def test_the_smaller_full_cover_wins_when_both_covers_tie(self):
        large = {"agent_cover": 1.0, "wrist_cover": 1.0, "scene_fraction": 0.424}
        tight = {"agent_cover": 1.0, "wrist_cover": 1.0, "scene_fraction": 0.173}
        agent_only = {"agent_cover": 1.0, "wrist_cover": 0.0, "scene_fraction": 0.04}
        self.assertIs(tc.choose_tight([large, agent_only, tight]), tight)

    def test_a_tiny_scene_that_misses_the_wrist_loses_to_a_full_cover(self):
        missed = {"agent_cover": 1.0, "wrist_cover": 0.0, "scene_fraction": 0.04}
        full = {"agent_cover": 1.0, "wrist_cover": 1.0, "scene_fraction": 0.42}
        self.assertIs(tc.choose_tight([missed, full]), full)

    def test_a_full_frame_plate_names_the_camera_distance(self):
        text = tc.tight_reading(1.0, 1.0, 1.0, 0.127)
        self.assertIn("0.127 m", text)
        self.assertIn("100%", text)

    def test_a_small_full_cover_is_named_as_tight(self):
        text = tc.tight_reading(1.0, 1.0, 0.17, 0.396)
        self.assertIn("tight", text)


class ResetAndRates(unittest.TestCase):
    def test_clear_episode_zeros_the_done_flag_on_the_inner_env(self):
        class Inner:
            done = True
            timestep = 180

        class Outer:
            def __init__(self):
                self.env = Inner()
                self.done = True

        env = Outer()
        tc.clear_episode(env)
        self.assertFalse(env.done)
        self.assertFalse(env.env.done)
        self.assertEqual(env.env.timestep, 0)

    def test_step_done_reads_a_four_tuple_and_a_five_tuple(self):
        self.assertTrue(tc.step_done(("obs", 0.0, True, {})))
        self.assertFalse(tc.step_done(("obs", 0.0, False, False, {})))
        self.assertTrue(tc.step_done(("obs", 0.0, False, True, {})))

    def test_rates_count_a_success_and_an_episode_that_never_stepped(self):
        rows = [
            {"edit": "base", "success_after": True, "steps": 80},
            {"edit": "base", "success_after": False, "steps": 220},
            {"edit": "absent", "success_after": False, "steps": 0},
        ]
        rates = {row["edit"]: row for row in tc.rate_rows(rows)}
        self.assertEqual(rates["base"]["successes"], 1)
        self.assertEqual(rates["base"]["episodes"], 2)
        self.assertEqual(rates["absent"]["no_step"], 1)


if __name__ == "__main__":
    unittest.main()
