"""Tests for mid-training per-capture-mode metrics (Approach B, trainer-side).

Why this test exists: lockbox FPR is dominated by webcam-style captures (memory
`project_lockbox_fpr_dominated_by_webcam_mode.md`). Block B already logs
per-bucket recall/FPR, but webcam vs screen vs normal_photo cuts inside a
single bucket are invisible. This helper joins per-video predictions against
a `clip_capture_mode` lookup (built from the lockbox tags parquet) and emits
per-mode panels under `{log_prefix}/per_capture_mode/<mode>/...`.

Scope (intentionally limited):
  - Pure-numpy metric computation, like Block B's helper.
  - The trainer feeds this a per-video representative-path list, parallel to
    `method_preds` / `method_labels`. No dataloader changes.
  - The parquet load is a separate function returning a plain dict, so tests
    can inject fixtures without touching pandas at all.

Out of scope (deferred): face_pixel_area correlation panel, custom W&B
workspace pinning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.trainer import (  # noqa: E402
    _compute_per_capture_mode_recall_fpr,
    _load_capture_mode_lookup,
)


def _basic_inputs():
    """Build a fixture: 1 fake bucket + 1 real bucket, with capture-mode tags
    that split each bucket non-trivially across modes."""
    method_preds = {
        # Fake bucket — 4 videos, two webcam two normal_photo
        "deeplive": [0.95, 0.30, 0.91, 0.20],
        # Real bucket — 5 videos: 2 webcam (high FPR), 2 normal_photo (low FPR), 1 screen
        "real_dor": [0.80, 0.85, 0.10, 0.05, 0.95],
    }
    method_labels = {
        "deeplive": [1, 1, 1, 1],
        "real_dor": [0, 0, 0, 0, 0],
    }
    method_paths = {
        "deeplive": [
            "gs://x/deeplive_a.jpg",
            "gs://x/deeplive_b.jpg",
            "gs://x/deeplive_c.jpg",
            "gs://x/deeplive_d.jpg",
        ],
        "real_dor": [
            "gs://x/real_dor_w1.jpg",
            "gs://x/real_dor_w2.jpg",
            "gs://x/real_dor_n1.jpg",
            "gs://x/real_dor_n2.jpg",
            "gs://x/real_dor_s1.jpg",
        ],
    }
    capture_mode_lookup = {
        "gs://x/deeplive_a.jpg": "webcam",
        "gs://x/deeplive_b.jpg": "webcam",
        "gs://x/deeplive_c.jpg": "normal_photo",
        "gs://x/deeplive_d.jpg": "normal_photo",
        "gs://x/real_dor_w1.jpg": "webcam",
        "gs://x/real_dor_w2.jpg": "webcam",
        "gs://x/real_dor_n1.jpg": "normal_photo",
        "gs://x/real_dor_n2.jpg": "normal_photo",
        "gs://x/real_dor_s1.jpg": "screen",
    }
    real_source_names = ["real_dor"]
    return method_preds, method_labels, method_paths, capture_mode_lookup, real_source_names


def test_per_capture_mode_metrics_emit_expected_keys():
    """Real buckets contribute to per-mode `recall_real` + `fpr`; fake buckets
    contribute to per-mode `recall_fake`. The namespace is flat under
    `{log_prefix}/<mode>/...` — mode is the cut, not the dataset."""
    method_preds, method_labels, method_paths, lookup, real_source_names = _basic_inputs()

    out = _compute_per_capture_mode_recall_fpr(
        method_preds=method_preds,
        method_labels=method_labels,
        method_paths=method_paths,
        capture_mode_lookup=lookup,
        real_source_names=real_source_names,
        threshold=0.5,
        log_prefix="mid_eval/per_capture_mode",
    )

    # Real-side metrics should exist for every mode that has real samples
    assert "mid_eval/per_capture_mode/webcam/fpr" in out
    assert "mid_eval/per_capture_mode/webcam/recall_real" in out
    assert "mid_eval/per_capture_mode/normal_photo/fpr" in out
    assert "mid_eval/per_capture_mode/normal_photo/recall_real" in out
    assert "mid_eval/per_capture_mode/screen/fpr" in out
    assert "mid_eval/per_capture_mode/screen/recall_real" in out

    # Fake-side metrics should exist for every mode that has fake samples
    assert "mid_eval/per_capture_mode/webcam/recall_fake" in out
    assert "mid_eval/per_capture_mode/normal_photo/recall_fake" in out


def test_per_capture_mode_metric_values_are_numerically_correct():
    """Sanity-check the math against the fixture.

    real_dor webcam: 2 samples, both flagged (0.80, 0.85 ≥ 0.5) → fpr = 1.0
    real_dor normal_photo: 2 samples, neither flagged → fpr = 0.0
    real_dor screen: 1 sample, flagged → fpr = 1.0

    deeplive webcam: 2 samples, 1 flagged (0.95 ≥ 0.5, 0.30 < 0.5) → recall = 0.5
    deeplive normal_photo: 2 samples, 1 flagged (0.91, 0.20) → recall = 0.5
    """
    method_preds, method_labels, method_paths, lookup, real_source_names = _basic_inputs()

    out = _compute_per_capture_mode_recall_fpr(
        method_preds=method_preds,
        method_labels=method_labels,
        method_paths=method_paths,
        capture_mode_lookup=lookup,
        real_source_names=real_source_names,
        threshold=0.5,
        log_prefix="mid_eval/per_capture_mode",
    )

    assert out["mid_eval/per_capture_mode/webcam/fpr"] == 1.0
    assert out["mid_eval/per_capture_mode/webcam/recall_real"] == 0.0
    assert out["mid_eval/per_capture_mode/normal_photo/fpr"] == 0.0
    assert out["mid_eval/per_capture_mode/normal_photo/recall_real"] == 1.0
    assert out["mid_eval/per_capture_mode/screen/fpr"] == 1.0
    assert out["mid_eval/per_capture_mode/screen/recall_real"] == 0.0
    assert out["mid_eval/per_capture_mode/webcam/recall_fake"] == 0.5
    assert out["mid_eval/per_capture_mode/normal_photo/recall_fake"] == 0.5


def test_per_capture_mode_helper_logs_unknown_bucket_when_path_not_in_lookup():
    """Coverage matters. If a frame's path isn't in the lookup, count it under
    `unknown` so a quiet coverage drop is visible in W&B rather than silent."""
    method_preds = {"real_dor": [0.10, 0.95]}
    method_labels = {"real_dor": [0, 0]}
    method_paths = {
        "real_dor": ["gs://x/known.jpg", "gs://x/missing.jpg"],
    }
    lookup = {"gs://x/known.jpg": "webcam"}
    real_source_names = ["real_dor"]

    out = _compute_per_capture_mode_recall_fpr(
        method_preds=method_preds,
        method_labels=method_labels,
        method_paths=method_paths,
        capture_mode_lookup=lookup,
        real_source_names=real_source_names,
        threshold=0.5,
        log_prefix="mid_eval/per_capture_mode",
    )

    assert "mid_eval/per_capture_mode/webcam/fpr" in out
    assert "mid_eval/per_capture_mode/unknown/fpr" in out
    # known: 1 sample, not flagged -> fpr 0
    assert out["mid_eval/per_capture_mode/webcam/fpr"] == 0.0
    # unknown: 1 sample, flagged -> fpr 1
    assert out["mid_eval/per_capture_mode/unknown/fpr"] == 1.0


def test_per_capture_mode_helper_returns_empty_dict_when_lookup_is_empty():
    """An empty lookup means no panels — the call site should still see a dict
    it can `wandb_log_dict.update(...)` against without crashing."""
    method_preds = {"real_dor": [0.10, 0.95]}
    method_labels = {"real_dor": [0, 0]}
    method_paths = {"real_dor": ["gs://x/a.jpg", "gs://x/b.jpg"]}

    out = _compute_per_capture_mode_recall_fpr(
        method_preds=method_preds,
        method_labels=method_labels,
        method_paths=method_paths,
        capture_mode_lookup={},
        real_source_names=["real_dor"],
        threshold=0.5,
        log_prefix="mid_eval/per_capture_mode",
    )

    # All paths fall into 'unknown' when lookup is empty; no other modes appear
    assert "mid_eval/per_capture_mode/unknown/fpr" in out
    assert all(
        k.startswith("mid_eval/per_capture_mode/unknown/")
        for k in out.keys()
    )


def test_per_capture_mode_helper_skips_buckets_with_no_paths():
    """If `method_paths` is missing a bucket entirely, that bucket is skipped
    (no crash). Defensive — the trainer's per-bucket parallel append should
    keep these in sync, but the helper must not assume it."""
    method_preds = {"real_dor": [0.10, 0.95], "ghost_bucket": [0.5]}
    method_labels = {"real_dor": [0, 0], "ghost_bucket": [0]}
    method_paths = {"real_dor": ["gs://x/a.jpg", "gs://x/b.jpg"]}  # no ghost_bucket
    lookup = {"gs://x/a.jpg": "webcam", "gs://x/b.jpg": "webcam"}

    out = _compute_per_capture_mode_recall_fpr(
        method_preds=method_preds,
        method_labels=method_labels,
        method_paths=method_paths,
        capture_mode_lookup=lookup,
        real_source_names=["real_dor"],
        threshold=0.5,
        log_prefix="mid_eval/per_capture_mode",
    )

    # No keys for ghost_bucket modes; only the real_dor → webcam aggregation
    assert "mid_eval/per_capture_mode/webcam/fpr" in out
    assert all("ghost" not in k for k in out.keys())


def test_per_capture_mode_helper_returns_pure_python_floats_not_numpy():
    """W&B compat: the helper's outputs must be plain Python floats so a naive
    JSON serializer doesn't choke. Same contract as Block B's helper."""
    method_preds, method_labels, method_paths, lookup, real_source_names = _basic_inputs()
    out = _compute_per_capture_mode_recall_fpr(
        method_preds=method_preds,
        method_labels=method_labels,
        method_paths=method_paths,
        capture_mode_lookup=lookup,
        real_source_names=real_source_names,
        threshold=0.5,
        log_prefix="mid_eval/per_capture_mode",
    )
    for key, value in out.items():
        assert isinstance(value, float), f"{key} is {type(value)}, expected float"
        assert not isinstance(value, np.floating), f"{key} is np.floating"


def test_load_capture_mode_lookup_from_parquet_builds_uri_to_mode_dict(tmp_path):
    """The parquet loader: read tag parquet, return `{gcs_uri: clip_capture_mode}`.
    Tested via a tiny synthetic parquet — keeps pandas in the dependency graph
    explicit."""
    import pandas as pd

    df = pd.DataFrame({
        "gcs_uri": [
            "gs://test/a.jpg",
            "gs://test/b.jpg",
            "gs://test/c.jpg",
        ],
        "clip_capture_mode": ["webcam", "normal_photo", "screen"],
        "label": [0, 0, 1],
    })
    p = tmp_path / "fake_tags.parquet"
    df.to_parquet(p)

    lookup = _load_capture_mode_lookup(str(p))
    assert lookup == {
        "gs://test/a.jpg": "webcam",
        "gs://test/b.jpg": "normal_photo",
        "gs://test/c.jpg": "screen",
    }


def test_load_capture_mode_lookup_returns_empty_dict_when_path_missing():
    """Non-existent path is the disabled case — return {} so the call site can
    no-op without a try/except."""
    lookup = _load_capture_mode_lookup("/nonexistent/path/does_not_exist.parquet")
    assert lookup == {}


def test_load_capture_mode_lookup_returns_empty_dict_when_path_is_none():
    """None path is also the disabled case — a config that didn't set
    `mid_eval_capture_mode_parquet` at all."""
    lookup = _load_capture_mode_lookup(None)
    assert lookup == {}
