import unittest

import numpy as np

import token_subspace as ts


class _Weight:
    def __init__(self, rows: int) -> None:
        self.shape = (rows, 8)
        self.weight = self


class _Policy:
    def __init__(self) -> None:
        self._modules = [
            ("model.language_model.embeddings.position_embedding", _Weight(32)),
            ("model.vision_tower.vision_model.embeddings.position_embedding", _Weight(256)),
        ]
        self.config = type("Config", (), {})()
        self.config.input_features = {
            "observation.images.image": object(),
            "observation.images.image2": object(),
            "observation.state": object(),
        }

    def named_modules(self):
        return self._modules


class LowRankPayload(unittest.TestCase):
    def test_rank_of_a_rank2_block_recovers_the_delta(self):
        rng = np.random.default_rng(0)
        source = np.zeros((12, 6), dtype=np.float32)
        donor = source.copy()
        indexes = np.array([1, 3, 5, 8, 10])
        basis = rng.normal(size=(2, 6))
        donor[indexes] = (rng.normal(size=(len(indexes), 2)) @ basis).astype(np.float32)
        recovered = ts.low_rank_payload(source, donor, indexes, 2)
        self.assertTrue(np.allclose(recovered[indexes], donor[indexes] - source[indexes], atol=1e-5))
        self.assertEqual(float(np.abs(recovered[0]).sum()), 0.0)
        short = ts.low_rank_payload(source, donor, indexes, 1)
        self.assertGreater(float(np.linalg.norm(recovered - short)), 1e-3)

    def test_mean_payload_is_constant_on_the_chosen_tokens(self):
        source = np.zeros((6, 3), dtype=np.float32)
        donor = np.arange(18, dtype=np.float32).reshape(6, 3)
        indexes = np.array([0, 2, 4])
        payload = ts.low_rank_payload(source, donor, indexes, 0)
        self.assertTrue(np.allclose(payload[indexes], payload[indexes][:1]))
        self.assertEqual(float(np.abs(payload[1]).sum()), 0.0)

    def test_square_vision_embedding_sets_the_image_span(self):
        policy = _Policy()
        self.assertEqual(ts.tokens_per_image(policy), 256)
        self.assertEqual(ts.image_count(policy), 2)
        self.assertEqual(ts.token_region(10, 256, 2, 530), "image0")
        self.assertEqual(ts.token_region(300, 256, 2, 530), "image1")
        self.assertEqual(ts.token_region(520, 256, 2, 530), "language")
        self.assertEqual(ts.token_region(10, 256, 2, 400), "unknown")


class SubspaceVerdict(unittest.TestCase):
    def test_rank4_matching_the_position_patch_is_a_shared_subspace(self):
        text = ts.subspace_verdict("absence", 0.05, 0.60, {"mean": 0.05, "rank1": 0.20, "rank4": 0.50, "rank16": 0.58})
        self.assertIn("low-rank subspace", text)
        self.assertIn("5%", text)

    def test_rank16_far_below_the_position_patch_is_token_specific(self):
        text = ts.subspace_verdict("color", 0.05, 0.47, {"mean": 0.02, "rank1": 0.05, "rank4": 0.10, "rank16": 0.18})
        self.assertIn("token-specific", text)
        self.assertIn("18.0%", text)
