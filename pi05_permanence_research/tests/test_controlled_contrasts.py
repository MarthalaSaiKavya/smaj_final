import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import controlled_contrasts as cc


class ImageControls(unittest.TestCase):
    def test_paint_changes_only_the_mask(self):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[20:30, 20:34] = (180, 20, 20)
        mask = cc.disk_mask(48, 64, 27, 25, 8)
        painted = cc.paint_mask(image, mask, cc.GRAY)
        changed = cc.changed_pixels(image, painted)
        self.assertFalse(np.logical_and(changed, ~mask).any())
        self.assertTrue(cc.mask_edit_is_exact(image, painted, mask))
        self.assertGreater(cc.changed_fraction(image, painted), 0)

    def test_translated_copy_matches_the_occlusion_shape(self):
        full = cc.disk_mask(80, 96, 50, 40, 12)
        partial = cc.half_mask(full)
        control = cc.translated_copy(partial, full)
        self.assertIsNotNone(control)
        self.assertEqual(int(partial.sum()), int(control.sum()))
        self.assertFalse(np.logical_and(control, cc.dilate_mask(full, 4)).any())
        self.assertLess(int(partial.sum()), int(full.sum()))
        self.assertFalse(np.logical_and(partial, ~full).any())

    def test_projection_of_a_point_in_front_of_the_camera(self):
        xpos = np.zeros(3)
        xmat = np.eye(3)
        center = cc.project_point(xpos, xmat, 90.0, np.array([0.0, 0.0, -1.0]), 100, 120, flip180=False)
        self.assertIsNotNone(center)
        u, v, depth = center
        self.assertAlmostEqual(u, 60.0)
        self.assertAlmostEqual(v, 50.0)
        self.assertAlmostEqual(depth, 1.0)
        self.assertIsNone(
            cc.project_point(xpos, xmat, 90.0, np.array([0.0, 0.0, 1.0]), 100, 120, flip180=False)
        )
        flipped = cc.project_point(xpos, xmat, 90.0, np.array([0.0, 0.0, -1.0]), 100, 120, flip180=True)
        self.assertAlmostEqual(flipped[0], 120 - 1 - 60.0)
        self.assertAlmostEqual(flipped[1], 100 - 1 - 50.0)

    def test_pixel_radius_at_ninety_degrees(self):
        self.assertAlmostEqual(cc.pixel_radius(90.0, 100, 1.0, 1.0), 50.0)

    def test_recolor_swaps_red_and_green(self):
        red = np.array([[0.8, 0.1, 0.1, 1.0]], dtype=np.float32)
        green = cc.swapped_rgba(red)
        self.assertGreater(green[0, 1], green[0, 0])
        self.assertEqual(float(green[0, 3]), 1.0)
        back = cc.swapped_rgba(green)
        self.assertGreater(back[0, 0], back[0, 1])

    def test_color_swap_defines_the_occlusion_mask(self):
        base = np.zeros((64, 80, 3), dtype=np.uint8)
        base[20:40, 30:50] = (180, 20, 20)
        edited = base.copy()
        edited[20:40, 30:50] = (20, 170, 40)
        found = cc.masks_from_change(base, edited)
        self.assertIsNotNone(found)
        full, partial, control = found
        self.assertEqual(int(full.sum()), 20 * 20)
        self.assertEqual(int(partial.sum()), int(control.sum()))
        self.assertFalse(np.logical_and(partial, control).any())
        occluded = cc.paint_mask(base, partial, cc.GRAY)
        self.assertTrue(cc.mask_edit_is_exact(base, occluded, partial))
        self.assertEqual(cc.changed_fraction(base, occluded), int(partial.sum()) / base.shape[0] / base.shape[1])

    def test_full_cover_of_a_contained_object_matches(self):
        present = np.zeros((32, 32, 3), dtype=np.uint8)
        present[10:20, 10:20] = (200, 0, 0)
        absent = present.copy()
        absent[10:20, 10:20] = 0
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:22, 8:22] = True
        covered_present = cc.paint_mask(present, mask, cc.GRAY)
        covered_absent = cc.paint_mask(absent, mask, cc.GRAY)
        self.assertEqual(cc.outside_fraction(covered_present, covered_absent, mask), 0.0)
        self.assertTrue(np.array_equal(covered_present, covered_absent))
        agreed = cc.full_cover_interpretation(0.0, 0.0)
        self.assertIn("action chunks agree", agreed)
        residual = cc.full_cover_interpretation(0.00040, 0.01040)
        self.assertIn("covered images agree", residual)
        self.assertIn("0.01040", residual)
        self.assertIn("residual floor", residual)
        self.assertNotIn("not a deterministic", residual)
        outside = cc.full_cover_interpretation(0.05, 0.2)
        self.assertIn("outside the cover", outside)


class TranscoderScores(unittest.TestCase):
    def test_structure_delta_removes_the_token_mean(self):
        source = np.zeros((6, 4), dtype=np.float32)
        donor = np.arange(24, dtype=np.float32).reshape(6, 4)
        delta = cc.structure_delta(donor, source)
        self.assertTrue(np.allclose(delta.mean(axis=0), 0.0, atol=1e-6))

    def test_action_gap_closes_when_the_edit_reaches_the_donor(self):
        source = np.zeros((4, 2), dtype=np.float32)
        donor = np.ones((4, 2), dtype=np.float32)
        closed = cc.action_metrics(donor, source, donor)
        unchanged = cc.action_metrics(source, source, donor)
        self.assertAlmostEqual(closed["fraction_gap_closed"], 1.0)
        self.assertAlmostEqual(unchanged["fraction_gap_closed"], 0.0)
        self.assertIsNone(cc.action_metrics(source, source, source)["fraction_gap_closed"])

    def test_feature_pick_keeps_the_specific_direction(self):
        scores = {
            "color": np.array([5.0, 0.1, 0.0, 0.2]),
            "absence": np.array([0.1, 4.0, 0.0, 0.2]),
            "occlusion": np.array([0.05, 0.05, 3.0, 0.1]),
            "slab": np.array([0.0, 0.0, 0.1, 2.5]),
        }
        color_ids, specific = cc.pick_features(
            scores["color"], [scores["absence"], scores["occlusion"], scores["slab"]], k=1
        )
        self.assertTrue(specific)
        self.assertEqual(color_ids.tolist(), [0])
        occlusion_ids, occlusion_specific = cc.pick_features(
            scores["occlusion"], [scores["color"], scores["absence"], scores["slab"]], k=1
        )
        self.assertTrue(occlusion_specific)
        self.assertEqual(occlusion_ids.tolist(), [2])

    def test_label_frame_may_include_geometry(self):
        label, body = cc.frame_label(("held_visible", 4, {"depth_gap": 0.1, "angle": 0.2}))
        self.assertEqual((label, body), ("held_visible", 4))
        label, body = cc.frame_label(("not_held", None))
        self.assertEqual(label, "not_held")
        self.assertIsNone(body)
        label, body = cc.frame_label((None, None, None))
        self.assertIsNone(label)
        self.assertIsNone(body)

    def test_off_contrast_pairs(self):
        self.assertEqual(cc.off_contrast("color"), "absence")
        self.assertEqual(cc.off_contrast("absence"), "color")
        self.assertEqual(cc.off_contrast("occlusion"), "color")

    def test_remap_and_norm_match(self):
        delta = np.zeros((5, 4), dtype=np.float32)
        delta[:, 0] = 2.0
        remapped = cc.remap_features(delta, np.array([0]), np.array([3]))
        self.assertTrue(np.allclose(remapped[:, 3], 2.0))
        self.assertTrue(np.allclose(remapped[:, 0], 0.0))
        reference = np.ones((5, 4), dtype=np.float32)
        matched = cc.match_l2(remapped, reference)
        self.assertAlmostEqual(float(np.linalg.norm(matched)), float(np.linalg.norm(reference)), places=5)
        self.assertTrue(np.allclose(cc.match_l2(np.zeros_like(delta), reference), 0.0))

    def test_moving_tokens_are_the_ones_that_changed(self):
        base = np.zeros((20, 3), dtype=np.float32)
        edited = base.copy()
        edited[:3] = 10.0
        mask = cc.moving_token_mask(base, edited, quantile=0.85)
        self.assertGreaterEqual(int(mask.sum()), 3)
        self.assertTrue(mask[:3].all())

    def test_transcoder_loss_drops_and_reconstructs(self):
        torch.manual_seed(0)
        dim, n_features, k = 16, 8, 2
        decoder = torch.randn(dim, n_features)
        tokens = []
        for _ in range(4):
            code = torch.zeros(32, n_features)
            index = torch.randint(0, n_features, (32, k))
            code.scatter_(1, index, torch.rand(32, k))
            tokens.append((code @ decoder.T).numpy().astype(np.float32))
        model, mean, std, _mean6, _std6 = cc.train_transcoder(
            tokens, None, n_features=n_features, k=k, steps=80, batch=64, seed=0
        )
        flat = np.concatenate(tokens, axis=0)
        recon = cc.reconstruction(model, flat, mean, std)
        self.assertGreater(cc.r2_score(recon, flat), 0.3)
        code = cc.encode_tokens(model, flat, mean, std)
        active = (code > 0).sum(axis=-1)
        self.assertTrue(np.all(active <= k))

    def test_sheet_pads_short_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sheet.png"
            cc.write_sheet(
                [
                    ("base", [np.zeros((20, 24, 3), dtype=np.uint8) for _ in range(3)]),
                    ("cover", [np.full((20, 24, 3), 128, dtype=np.uint8)]),
                ],
                path,
            )
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
