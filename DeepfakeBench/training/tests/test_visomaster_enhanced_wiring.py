"""Tests for the visomaster_enhanced data-source wiring.

The ``visomaster_enhanced`` family is the train-side substrate that matches
the eval suite ``visomaster_enhanced_macro_dev`` (eval bucket
``gs://teams-faces-data-test-2914-fake-4420-real-feb-28``). This test
locks in the contract that ``create_unified_samples_from_visomaster_enhanced``
emits properly-tagged UnifiedPairedSample wrappers so that family-weighted
sampling can find them.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_combined_paired():
    pytest.importorskip("torch")
    from data.sources.combined_paired import (
        QUALITY_DOMAIN_MAP,
        UnifiedPairedSample,
        create_unified_samples_from_visomaster_enhanced,
    )
    from data.sources.visomaster import VisoMasterEnhancedSample

    return (
        QUALITY_DOMAIN_MAP,
        UnifiedPairedSample,
        create_unified_samples_from_visomaster_enhanced,
        VisoMasterEnhancedSample,
    )


def _make_enhanced_sample(VisoMasterEnhancedSample, identity_token: str, enhancer: str, tier: str):
    return VisoMasterEnhancedSample(
        sample_id=f"visomaster_CSCS_00007_enhanced_{enhancer}",
        original_sample_id="visomaster_CSCS_00007",
        swap_model="CSCS",
        enhancer=enhancer,
        frame_count=16,
        tier=tier,
        identity_delta=0.30,
        artifact_delta=0.10,
        enhanced_bucket="visomaster-enhanced-face-cropped",
        original_bucket="live-deepfake-methods-real-and-fake-frames-cropped",
        manifest={"original_video_name": f"cropped_{identity_token}.mp4"},
    )


def test_quality_domain_map_recognizes_visomaster_enhanced():
    """The GRL pipeline's domain head needs a stable domain id for the new family."""
    QUALITY_DOMAIN_MAP, *_ = _load_combined_paired()
    assert "visomaster_enhanced" in QUALITY_DOMAIN_MAP
    assert QUALITY_DOMAIN_MAP["visomaster_enhanced"] == 2, (
        "visomaster_enhanced must share studio_capture domain (2) with "
        "deeplive/visomaster — same real source, different post-hoc enhancer"
    )


def test_create_unified_samples_from_visomaster_enhanced_tags_correctly():
    """Each VisoMasterEnhancedSample becomes one UnifiedPairedSample whose
    source is 'visomaster_enhanced', method is 'visomaster_enhanced_{enhancer}',
    identity is 'realpool_{identity}'. The 'realpool_' prefix is what prevents
    identity-leak between this family and visomaster/deeplive in the
    identity-stratified splitter.
    """
    (
        _,
        UnifiedPairedSample,
        create_unified_samples_from_visomaster_enhanced,
        VisoMasterEnhancedSample,
    ) = _load_combined_paired()

    enhanced = [
        _make_enhanced_sample(VisoMasterEnhancedSample, "personA", "gfpgan", "STRONG"),
        _make_enhanced_sample(VisoMasterEnhancedSample, "personA", "codeformer", "STRONG"),
        _make_enhanced_sample(VisoMasterEnhancedSample, "personB", "gpen-512", "MODERATE"),
    ]

    logger = logging.getLogger("test_visomaster_enhanced")
    unified = create_unified_samples_from_visomaster_enhanced(enhanced, logger)

    assert len(unified) == 3
    assert all(isinstance(s, UnifiedPairedSample) for s in unified)

    sources = {s.source for s in unified}
    assert sources == {"visomaster_enhanced"}, (
        "All converted samples must report source='visomaster_enhanced' so the "
        "family-weighted sampler can route them via 'visomaster_enhanced_fake'."
    )

    methods = sorted({s.method for s in unified})
    assert methods == [
        "visomaster_enhanced_codeformer",
        "visomaster_enhanced_gfpgan",
        "visomaster_enhanced_gpen-512",
    ], "method must encode the enhancer for per-enhancer eval-suite metrics"

    identities = sorted({s.identity for s in unified})
    assert identities == ["realpool_personA", "realpool_personB"], (
        "identity must use the realpool_ prefix shared with visomaster/deeplive — "
        "this is what prevents the same person appearing in train+val splits"
    )

    # Each unified sample must keep a reference back to the source object
    # so the dataloader can resolve frame_paths via real_frame_paths/fake_frame_paths.
    for s in unified:
        assert s.original_sample is not None
        assert hasattr(s.original_sample, "real_frame_paths")
        assert hasattr(s.original_sample, "fake_frame_paths")


def test_visomaster_enhanced_frame_paths_target_correct_buckets():
    """Real frames pull from the original (training-already-used) bucket; fake
    frames pull from the enhanced bucket. This is the structural difference
    from the unenhanced ``visomaster`` family that closes the train/eval gap
    on ``visomaster_enhanced_macro_dev``.
    """
    *_, VisoMasterEnhancedSample = _load_combined_paired()

    sample = _make_enhanced_sample(
        VisoMasterEnhancedSample, identity_token="x", enhancer="gfpgan", tier="STRONG"
    )

    real_paths = sample.real_frame_paths([0, 2, 4])
    fake_paths = sample.fake_frame_paths([0, 2, 4])
    assert all(p.startswith("gs://live-deepfake-methods-real-and-fake-frames-cropped/") for p in real_paths)
    assert all(p.startswith("gs://visomaster-enhanced-face-cropped/") for p in fake_paths)
    assert len(real_paths) == 3 and len(fake_paths) == 3
