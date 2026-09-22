import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import controlled_contrasts as cc
import feature_count_sweep as fc


class SweepPlan(unittest.TestCase):
    def test_counts_clip_and_keep_order(self):
        self.assertEqual(fc.parse_counts("8,32,128,512", 512), [8, 32, 128, 512])
        self.assertEqual(fc.parse_counts("8, 32, 8, 1000", 64), [8, 32, 64])

    def test_larger_counts_extend_the_same_prefix(self):
        score = np.array([0.1, -0.5, 0.2, 0.9, -0.05], dtype=np.float32)
        small = fc.top_features(score, 2)
        large = fc.top_features(score, 4)
        self.assertEqual(small.tolist(), [3, 1])
        self.assertEqual(large[:2].tolist(), small.tolist())

    def test_full_dictionary_control_is_a_permutation(self):
        rng = np.random.default_rng(0)
        selected = fc.top_features(np.arange(16), 16)
        destination, kind = fc.control_destination(selected, 16, rng)
        self.assertEqual(kind, "permute_all")
        self.assertEqual(sorted(destination.tolist()), list(range(16)))
        code = np.arange(32, dtype=np.float32).reshape(2, 16)
        remapped = cc.remap_features(cc.keep_features(code, selected), selected, destination)
        self.assertEqual(sorted(np.round(remapped[0], 5).tolist()), sorted(np.round(code[0], 5).tolist()))

    def test_partial_control_is_disjoint(self):
        rng = np.random.default_rng(1)
        selected = np.array([1, 4, 7])
        destination, kind = fc.control_destination(selected, 16, rng)
        self.assertEqual(kind, "disjoint")
        self.assertEqual(len(destination), 3)
        self.assertEqual(len(set(destination.tolist()) & set(selected.tolist())), 0)

    def test_checkpoint_roundtrip(self):
        model = cc.TokenTranscoder(4, 8, 2, False)
        mean5 = np.zeros(4, dtype=np.float32)
        std5 = np.ones(4, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transcoder.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "dim": 4,
                    "n_features": 8,
                    "k": 2,
                    "predict_next": False,
                    "mean5": mean5,
                    "std5": std5,
                },
                path,
            )
            loaded, loaded_mean, loaded_std, n_features = fc.load_transcoder(path)
        self.assertEqual(n_features, 8)
        self.assertEqual(loaded_mean.shape, (4,))
        self.assertTrue(np.allclose(loaded_std, std5))
        self.assertEqual(list(loaded.state_dict()), list(model.state_dict()))


class SweepVerdict(unittest.TestCase):
    def test_climbing_curve_names_the_smallest_count(self):
        curve = [(8, 0.15, 0.10), (32, 0.40, 0.12), (512, 0.70, 0.20)]
        text = fc.sweep_verdict("color", curve, 0.88)
        self.assertIn("climbs toward token structure", text)
        self.assertIn("32", text)

    def test_flat_curve_asks_for_a_token_patch(self):
        curve = [(8, 0.15, 0.10), (512, 0.18, 0.16)]
        text = fc.sweep_verdict("absence", curve, 0.77)
        self.assertIn("stays near the 8-feature result", text)
        self.assertIn("token-position patch", text)

    def test_partial_curve_reports_both_ends(self):
        curve = [(8, 0.15, 0.10), (512, 0.40, 0.12)]
        text = fc.sweep_verdict("occlusion", curve, 0.86)
        self.assertIn("15.0%", text)
        self.assertIn("40.0%", text)
        self.assertIn("86.0%", text)
