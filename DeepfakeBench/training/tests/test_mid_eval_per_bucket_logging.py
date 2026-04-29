"""Tests for mid-training W&B Block B observability — per-dataset-bucket metrics.

Why this test exists: trainer.py builds `wandb_log_dict` during the existing
mid-training validation cycle (driven by `evaluate_every_steps`). Block B
adds per-dataset-bucket recall/FPR panels under the `mid_eval/<dataset>/...`
namespace so we can see, at the configured cadence, which buckets carry
training signal vs. which lag.

Scope (intentionally limited):
  - Read-only with respect to existing tensors / data flow — we only consume
    the per-bucket prediction/label dicts that the trainer already maintains
    (`method_preds`, `method_labels` in `trainer/trainer.py`).
  - No dataloader / yaml / train_sweep changes.
  - Pure-numpy at logging time, so there is zero autograd risk (the eval loop
    is already `inference=True` + `setEval()`).

Out of scope (deferred): clip_capture_mode webcam vs studio split,
face-pixel-area correlation panel, custom W&B workspace pinning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.trainer import _compute_per_bucket_recall_fpr  # noqa: E402


def _basic_inputs():
    """Build a small per-bucket prediction/label fixture covering 2 fake + 1 real bucket."""
    method_preds = {
        # Fake bucket A — strong: 4/5 above 0.5 -> recall_fake = 0.8
        "deeplive": [0.95, 0.91, 0.88, 0.72, 0.30],
        # Fake bucket B — weak: 1/4 above 0.5 -> recall_fake = 0.25
        "viso": [0.80, 0.20, 0.10, 0.40],
        # Real bucket — 1/5 above 0.5 -> fpr = 0.2, recall_real = 0.8
        "real_dor": [0.10, 0.20, 0.05, 0.55, 0.40],
    }
    method_labels = {
        "deeplive": [1, 1, 1, 1, 1],
        "viso": [1, 1, 1, 1],
        "real_dor": [0, 0, 0, 0, 0],
    }
    real_source_names = ["real_dor"]
    return method_preds, method_labels, real_source_names


def test_per_bucket_metrics_dict_contains_expected_keys_for_each_dataset():
    """For each method/dataset bucket, the helper must emit a key under
    `mid_eval/<dataset>/...`. Fake buckets get `recall_fake`; real buckets
    get `recall_real` and `fpr`. This is the contract Block B relies on
    when pinning panels in the W&B workspace."""
    method_preds, method_labels, real_source_names = _basic_inputs()

    out = _compute_per_bucket_recall_fpr(
        method_preds=method_preds,
        method_labels=method_labels,
        real_source_names=real_source_names,
        threshold=0.5,
        log_prefix="mid_eval",
    )

    # Fake buckets: must have recall_fake under mid_eval/<dataset>/recall_fake
    assert "mid_eval/deeplive/recall_fake" in out
    assert "mid_eval/viso/recall_fake" in out

    # Real bucket: must have both recall_real and fpr under mid_eval/<dataset>/...
    assert "mid_eval/real_dor/recall_real" in out
    assert "mid_eval/real_dor/fpr" in out


def test_per_bucket_metric_values_are_numerically_correct():
    """Sanity-check the math: recall is fraction-of-positives flagged at τ;
    FPR is fraction-of-negatives flagged at τ."""
    method_preds, method_labels, real_source_names = _basic_inputs()

    out = _compute_per_bucket_recall_fpr(
        method_preds=method_preds,
        method_labels=method_labels,
        real_source_names=real_source_names,
        threshold=0.5,
        log_prefix="mid_eval",
    )

    # deeplive fake: 4/5 at >=0.5 -> 0.8
    assert np.isclose(out["mid_eval/deeplive/recall_fake"], 0.8)
    # viso fake: 1/4 at >=0.5 -> 0.25
    assert np.isclose(out["mid_eval/viso/recall_fake"], 0.25)
    # real_dor: 1/5 at >=0.5 -> fpr = 0.2 ; recall_real = 1 - fpr = 0.8
    assert np.isclose(out["mid_eval/real_dor/fpr"], 0.2)
    assert np.isclose(out["mid_eval/real_dor/recall_real"], 0.8)


def test_per_bucket_helper_skips_empty_buckets_without_crashing():
    """A bucket with zero samples should be silently skipped — we never want
    a no-data bucket to NaN-poison the W&B run or raise."""
    method_preds = {
        "deeplive": [0.9, 0.8],
        "empty_bucket": [],
        "real_dor": [],
    }
    method_labels = {
        "deeplive": [1, 1],
        "empty_bucket": [],
        "real_dor": [],
    }
    real_source_names = ["real_dor"]

    out = _compute_per_bucket_recall_fpr(
        method_preds=method_preds,
        method_labels=method_labels,
        real_source_names=real_source_names,
        threshold=0.5,
        log_prefix="mid_eval",
    )

    # The non-empty fake bucket shows up.
    assert "mid_eval/deeplive/recall_fake" in out
    # Empty buckets are silently skipped (no key emitted).
    assert not any(k.startswith("mid_eval/empty_bucket/") for k in out)
    assert not any(k.startswith("mid_eval/real_dor/") for k in out)


def test_per_bucket_helper_respects_custom_threshold():
    """A higher τ shrinks recall — we want the helper to honor whatever
    threshold the trainer hands it (0.5 today, but possibly the
    operating-point τ from the contract scorer in the future)."""
    method_preds = {"deeplive": [0.95, 0.55, 0.45, 0.30]}
    method_labels = {"deeplive": [1, 1, 1, 1]}
    real_source_names = []

    out_05 = _compute_per_bucket_recall_fpr(
        method_preds=method_preds,
        method_labels=method_labels,
        real_source_names=real_source_names,
        threshold=0.5,
        log_prefix="mid_eval",
    )
    out_06 = _compute_per_bucket_recall_fpr(
        method_preds=method_preds,
        method_labels=method_labels,
        real_source_names=real_source_names,
        threshold=0.6,
        log_prefix="mid_eval",
    )

    # τ=0.5 -> 2/4 above; τ=0.6 -> 1/4 above.
    assert np.isclose(out_05["mid_eval/deeplive/recall_fake"], 0.5)
    assert np.isclose(out_06["mid_eval/deeplive/recall_fake"], 0.25)


def test_per_bucket_helper_uses_log_prefix_argument():
    """The trainer reuses the same helper across `mid_eval`, `val`,
    `val_holdout` namespaces — the prefix must be honored exactly."""
    method_preds = {"deeplive": [0.9, 0.8], "real_dor": [0.1, 0.2]}
    method_labels = {"deeplive": [1, 1], "real_dor": [0, 0]}
    real_source_names = ["real_dor"]

    out = _compute_per_bucket_recall_fpr(
        method_preds=method_preds,
        method_labels=method_labels,
        real_source_names=real_source_names,
        threshold=0.5,
        log_prefix="val_holdout/per_bucket",
    )

    assert "val_holdout/per_bucket/deeplive/recall_fake" in out
    assert "val_holdout/per_bucket/real_dor/recall_real" in out
    assert "val_holdout/per_bucket/real_dor/fpr" in out


def test_per_bucket_helper_returns_pure_python_floats_not_tensors():
    """W&B serializes scalars; we want plain floats, not numpy scalars or
    tensors. Belt-and-suspenders against autograd / serialization weirdness."""
    method_preds = {"deeplive": [0.9, 0.4], "real_dor": [0.1, 0.6]}
    method_labels = {"deeplive": [1, 1], "real_dor": [0, 0]}
    real_source_names = ["real_dor"]

    out = _compute_per_bucket_recall_fpr(
        method_preds=method_preds,
        method_labels=method_labels,
        real_source_names=real_source_names,
        threshold=0.5,
        log_prefix="mid_eval",
    )

    for key, value in out.items():
        assert isinstance(value, float), f"{key} should be a plain float, got {type(value)}"
