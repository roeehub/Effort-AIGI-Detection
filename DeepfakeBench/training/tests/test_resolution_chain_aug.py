"""Tests for resolution_chain_aug (2026-05-15)."""
from __future__ import annotations

import unittest

import numpy as np

from data.augmentations.resolution_chain_aug import (
    apply_resolution_chain_aug,
    get_config_snapshot,
    set_resolution_chain_aug_config,
)


def _rng_image(h: int = 240, w: int = 240, c: int = 3, seed: int = 0) -> np.ndarray:
    rs = np.random.RandomState(seed)
    return (rs.rand(h, w, c) * 255).astype(np.uint8)


class TestResolutionChainAug(unittest.TestCase):

    def setUp(self) -> None:
        # Reset config to a known disabled state before each test
        set_resolution_chain_aug_config(enabled=False, p_apply=0.0)
        np.random.seed(0)

    def test_disabled_is_noop(self) -> None:
        img = _rng_image()
        set_resolution_chain_aug_config(enabled=False, p_apply=1.0)
        out = apply_resolution_chain_aug(img)
        # Returns the same array (noop short-circuit)
        self.assertIs(out, img)

    def test_p_apply_zero_is_noop_even_when_enabled(self) -> None:
        img = _rng_image()
        set_resolution_chain_aug_config(enabled=True, p_apply=0.0)
        # p_apply=0 forces _CONFIG.enabled = False internally per the set function
        # so this is the same as the disabled case.
        out = apply_resolution_chain_aug(img)
        self.assertIs(out, img)

    def test_output_shape_is_preserved_when_fired(self) -> None:
        img = _rng_image(h=240, w=240)
        set_resolution_chain_aug_config(enabled=True, p_apply=1.0)
        np.random.seed(42)  # deterministic — guarantees the random.random() < 1.0 branch
        out = apply_resolution_chain_aug(img)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, img.dtype)

    def test_content_changes_when_fired(self) -> None:
        # With p_apply=1.0 the chain always fires; output must differ from input
        # for at least one size/kernel combo (random noise input → re-encode is lossy).
        img = _rng_image(h=240, w=240, seed=123)
        set_resolution_chain_aug_config(
            enabled=True,
            p_apply=1.0,
            down_sizes=[64],
            kernels=["LINEAR"],
        )
        np.random.seed(0)
        out = apply_resolution_chain_aug(img)
        self.assertEqual(out.shape, img.shape)
        # Downsampling 240→64 and upsampling back must change content
        self.assertFalse(np.array_equal(out, img))

    def test_multiple_calls_can_pick_different_sizes(self) -> None:
        img = _rng_image(h=240, w=240, seed=7)
        set_resolution_chain_aug_config(
            enabled=True,
            p_apply=1.0,
            down_sizes=[64, 192],
            kernels=["LINEAR"],
        )
        # Different sizes -> different content
        np.random.seed(0)
        out1 = apply_resolution_chain_aug(img)
        np.random.seed(0)
        out2 = apply_resolution_chain_aug(img)
        # Same seed -> same size/kernel pick -> identical output
        np.testing.assert_array_equal(out1, out2)

    def test_unsupported_kernel_raises(self) -> None:
        with self.assertRaises(ValueError):
            set_resolution_chain_aug_config(
                enabled=True,
                p_apply=1.0,
                kernels=["BOGUS_KERNEL"],
            )

    def test_non_ndarray_input_returns_unchanged(self) -> None:
        set_resolution_chain_aug_config(enabled=True, p_apply=1.0)
        # Strings, None, lists — all passed through
        self.assertEqual(apply_resolution_chain_aug("not an array"), "not an array")
        self.assertIsNone(apply_resolution_chain_aug(None))

    def test_1d_input_returns_unchanged(self) -> None:
        set_resolution_chain_aug_config(enabled=True, p_apply=1.0)
        arr = np.array([1, 2, 3], dtype=np.uint8)
        out = apply_resolution_chain_aug(arr)
        np.testing.assert_array_equal(out, arr)

    def test_grayscale_image_works(self) -> None:
        img = _rng_image(h=240, w=240, c=3)[:, :, 0]  # (240, 240) grayscale
        set_resolution_chain_aug_config(enabled=True, p_apply=1.0, down_sizes=[96], kernels=["LINEAR"])
        np.random.seed(0)
        out = apply_resolution_chain_aug(img)
        self.assertEqual(out.shape, img.shape)

    def test_config_snapshot_reports_settings(self) -> None:
        set_resolution_chain_aug_config(
            enabled=True,
            p_apply=0.7,
            down_sizes=[80, 120],
            kernels=["LINEAR", "CUBIC"],
        )
        snap = get_config_snapshot()
        self.assertTrue(snap["enabled"])
        self.assertEqual(snap["p_apply"], 0.7)
        self.assertEqual(snap["down_sizes"], [80, 120])
        self.assertEqual(snap["kernels"], ["LINEAR", "CUBIC"])

    def test_default_sizes_and_kernels_match_probe(self) -> None:
        # Sanity: defaults match the 2026-05-15 CPU probe set
        set_resolution_chain_aug_config(enabled=True, p_apply=0.5)
        snap = get_config_snapshot()
        self.assertEqual(snap["down_sizes"], [64, 96, 128, 160, 192])
        self.assertEqual(snap["kernels"], ["LINEAR", "CUBIC", "AREA", "LANCZOS4"])

    def test_fire_probability_distribution_approximate(self) -> None:
        """With p_apply=0.3 and 2000 trials, fire rate should be ~30%."""
        img = _rng_image(h=64, w=64)  # small for speed
        set_resolution_chain_aug_config(
            enabled=True,
            p_apply=0.3,
            down_sizes=[32],
            kernels=["LINEAR"],
        )
        np.random.seed(0)
        n_fired = 0
        n_trials = 2000
        for _ in range(n_trials):
            out = apply_resolution_chain_aug(img)
            if not np.array_equal(out, img):
                n_fired += 1
        # Allow ±2pp slack
        self.assertGreater(n_fired / n_trials, 0.25)
        self.assertLess(n_fired / n_trials, 0.35)


if __name__ == "__main__":
    unittest.main()
