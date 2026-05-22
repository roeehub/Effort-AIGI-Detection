"""
Combined Paired Data Source - Merges DF40 and DeepLive datasets.

This module provides a unified data pipeline that combines:
- DF40 Paired dataset (~5,379 pairs, ~954 identities, NO landmarks)
- DeepLive dataset (~1k samples, ~920 unique identities, WITH landmarks)

The combined dataset:
1. Extracts identity from both sources (prevents cross-source identity collision)
2. Performs identity-stratified splitting (no identity in multiple splits)
3. Supports identity-balanced sampling (one method per identity per epoch)
4. Handles landmarks for DeepLive samples, None for DF40 samples
5. Optional method-level holdout mode (holdout methods unseen in training)
6. Optional external OOD monitoring set construction (e.g., WMA + external real)

Usage:
    from data.sources import create_data_pipeline
    
    data_config['data_source'] = 'combined_paired'
    data_config['combined_paired'] = {
        'df40': {
            'enabled': True,
            'pair_json': 'dataset/df40_pairs/df40-pair-matching.json',
            'gcs_bucket': 'df40-frames-recropped-rfa85',
        },
        'deeplive': {
            'enabled': True,
            'gcs_bucket': 'live-deepfake-methods-real-and-fake-frames-cropped',
        },
        'identity_balanced_sampling': True,
        'train_split': 0.8,
        'val_split': 0.1,
    }
    
    result = create_data_pipeline(config, data_config, logger)
"""

import hashlib
import inspect
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from fsspec.core import url_to_fs
from google.cloud import storage  # GCS client for Teams passthrough data
from torch.utils.data import DataLoader, IterableDataset

from . import register_data_source, DataPipelineResult
from utils.grouping import DEFAULT_ENHANCED_STRATEGIES, infer_family_key, normalize_method_name
from data.sources.method_domain_map import lookup_method_domain_with_label

logger = logging.getLogger(__name__)
DEFAULT_PARALLEL_GCS_DOWNLOAD_WORKERS = 4


# =============================================================================
# Quality Domain Mapping (for gradient-reversal quality-invariance head)
# =============================================================================
# LEGACY 4-class — kept for backward compat with existing callers + the
# detector-side QualityDomainHead.DOMAIN_MAP consistency test
# (test_unpaired_reals_and_grl::TestQualityDomainHead::test_domain_map_consistency).
# When the yaml's quality_domain_count == 4, samples will be assigned IDs 0–3
# via _quality_domain_for_source. When count > 4 (e.g., the 12-class Phase 3
# method-conditional GRL), samples should be assigned via
# _method_domain_for_sample below.
QUALITY_DOMAIN_MAP = {
    "df40": 0,           # clean_academic — soft, smooth, low-noise
    "external": 1,       # webcam_codec — sharp, noisy, codec artefacts (VCD / webcam)
    "deeplive": 2,       # studio_capture — studio lighting, variable quality
    "visomaster": 2,     # studio_capture — same real source as deeplive
    "visomaster_hints": 2,  # weak-signal baseline hints still come from the clean bucket
    "visomaster_hints_teams": 1,  # Teams-played weak-signal hints keep Teams codec traits
    "visomaster_enhanced": 2,  # studio_capture — enhanced fakes, same real source
    "visomaster_res_variant": 2,  # studio_capture — resolution variants, same real source
    "visomaster_teams_enhanced": 1,  # mixed source; iterator overrides per-item domains
    "deeplive_teams": 1, # Teams-passthrough — sharpened, codec noise, like webcam
    "proper_visomaster_clean": 2,
    "proper_visomaster_enhanced_clean": 2,
    "proper_visomaster_teams": 1,
    "proper_visomaster_enhanced_teams": 1,
    "youtube": 3,        # social_media — heavier compression, variable resolution
}


def _quality_domain_for_source(source: str) -> int:
    """LEGACY 4-class lookup — kept for backward compat with callers that
    don't have method+label in scope. New iterator code should use
    _domain_for_sample below to support the 12-bucket method-conditional GRL
    map (Phase 3, P18 onwards)."""
    return QUALITY_DOMAIN_MAP.get(source, 0)


# Module-level mode flag. Default OFF (legacy 4-class labels). The trainer
# init flips this to True when the yaml has the 12-class method-conditional
# GRL config (specifically: quality_domain_count == 12 or higher AND
# use_quality_domain_head == True). Iterators check this flag at every
# `_domain_for_sample` call.
_METHOD_DOMAIN_MODE: bool = False


def set_method_domain_mode(enabled: bool, log: Optional[logging.Logger] = None) -> None:
    """Toggle between legacy 4-class GRL labels and Phase 3 12-class
    method-conditional GRL labels. Trainer should call this exactly once at
    init based on the yaml config. Idempotent."""
    global _METHOD_DOMAIN_MODE
    prev = _METHOD_DOMAIN_MODE
    _METHOD_DOMAIN_MODE = bool(enabled)
    log = log or logger
    if prev != _METHOD_DOMAIN_MODE:
        if _METHOD_DOMAIN_MODE:
            log.info(
                "Method-domain mode ENABLED — iterators will emit 12-class "
                "method-conditional GRL labels per "
                "data.sources.method_domain_map.lookup_method_domain_with_label."
            )
        else:
            log.info(
                "Method-domain mode DISABLED — iterators will emit legacy "
                "4-class quality-domain GRL labels per QUALITY_DOMAIN_MAP."
            )


def get_method_domain_mode() -> bool:
    """Read-only snapshot of the current dispatch mode."""
    return _METHOD_DOMAIN_MODE


def _domain_for_sample(method: str, source: str, label: int) -> int:
    """Dispatching GRL-label lookup. Returns 4-class legacy ID (when
    _METHOD_DOMAIN_MODE is False) or 12-class method-conditional ID (when
    True). All iterator call sites should use this function rather than
    _quality_domain_for_source directly so the dispatch is centralized."""
    if _METHOD_DOMAIN_MODE:
        return lookup_method_domain_with_label(
            method=method, source=source, label=label
        )
    return QUALITY_DOMAIN_MAP.get(source, 0)


# =============================================================================
# Unified Sample Wrapper
# =============================================================================

@dataclass
class UnifiedPairedSample:
    """
    Unified wrapper for paired samples from different data sources.
    
    This provides a common interface for DF40 and DeepLive samples,
    enabling unified identity-stratified splitting and sampling.
    """
    # Unique identity across all sources (prefixed to avoid collision)
    identity: str
    
    # Source identifier
    source: str  # 'df40' or 'deeplive'
    
    # Original sample object (DF40PairedSample or DeepLiveSample)
    original_sample: Any
    
    # Method/strategy name (for logging and method distribution)
    method: str
    
    # Whether this sample has landmarks
    has_landmarks: bool = False
    
    # Additional metadata
    sample_id: str = ""
    method_variants: Tuple[str, ...] = field(default_factory=tuple)
    sampling_family_key: Optional[str] = None


@dataclass
class UnifiedUnpairedRealSample:
    """
    Wrapper for unpaired real-only samples from external sources.

    Unlike UnifiedPairedSample, these have no fake counterpart.
    Each sample represents a single identity with one or more real frames
    from an external GCS source (e.g., VCD webcam captures).
    """
    identity: str           # Unique identity (e.g., "external_vcd_<md5>")
    source: str             # 'external' (triggers external_real routing)
    method: str             # e.g., "external_vcd_real" (must contain "external")
    gcs_bucket: str         # e.g., "effort-collected-data"
    frame_paths: List[str]  # GCS paths to individual frame PNGs
    sample_id: str = ""     # Unique sample ID
    has_landmarks: bool = False
    is_unpaired_real: bool = True  # Distinguishes from paired samples
    original_sample: Any = None    # Compat: not used, but present for duck typing


def extract_deeplive_identity(sample: Any) -> str:
    """
    Extract identity from DeepLive sample's original_video_name.
    
    The original_video_name format is: cropped_XXX.mp4
    We extract XXX as the identity (YouTube video ID or similar).
    
    Args:
        sample: DeepLiveSample with original_video_name in manifest
        
    Returns:
        Identity string (without cropped_ prefix and .mp4 suffix)
    """
    # The original_video_name is stored in the manifest, need to load it
    # For now, we'll use the sample_id which contains the strategy prefix
    # and use the sample's underlying video reference
    
    # If we have access to original_video_name from manifest:
    original_name = getattr(sample, 'original_video_name', None)
    
    if original_name:
        # Strip cropped_ prefix and .mp4 suffix
        identity = original_name
        if identity.startswith('cropped_'):
            identity = identity[8:]  # len('cropped_') = 8
        if identity.endswith('.mp4'):
            identity = identity[:-4]
        return identity
    
    # Fallback: use sample_id (less ideal but works)
    # sample_id format: strategy_XXXX (e.g., "edge_cases_0000")
    return sample.sample_id


def create_unified_samples_from_df40(
    df40_samples: List[Any],
    logger: logging.Logger
) -> List[UnifiedPairedSample]:
    """
    Convert DF40PairedSample objects to UnifiedPairedSample wrappers.
    
    Args:
        df40_samples: List of DF40PairedSample objects
        logger: Logger
        
    Returns:
        List of UnifiedPairedSample wrappers
    """
    unified = []
    
    for sample in df40_samples:
        unified_sample = UnifiedPairedSample(
            identity=f"df40_{sample.target_identity}",  # Prefix to avoid collision
            source='df40',
            original_sample=sample,
            method=sample.method,
            has_landmarks=False,  # DF40 has no landmarks
            sample_id=sample.pair_id,
        )
        unified.append(unified_sample)
    
    logger.info(f"Created {len(unified)} unified samples from DF40")
    return unified


def create_unified_samples_from_deeplive(
    deeplive_samples: List[Any],
    deeplive_dataset: Any,
    logger: logging.Logger
) -> List[UnifiedPairedSample]:
    """
    Convert DeepLiveSample objects to UnifiedPairedSample wrappers.
    
    This function also loads manifest data to extract the original_video_name
    for proper identity extraction.
    
    Args:
        deeplive_samples: List of DeepLiveSample objects
        deeplive_dataset: DeepLiveDataset instance (for loading manifests)
        logger: Logger
        
    Returns:
        List of UnifiedPairedSample wrappers
    """
    unified = []
    identity_counts = defaultdict(int)
    
    for sample in deeplive_samples:
        # Extract identity from sample
        identity = extract_deeplive_identity(sample)
        # Use 'realpool_' prefix (NOT 'deeplive_') so that DeepLive and VisoMaster
        # samples from the same source video are grouped together during identity-
        # stratified splitting. This prevents identity leakage where the same real
        # person could appear in train (via DeepLive) and val (via VisoMaster).
        identity_with_prefix = f"realpool_{identity}"
        identity_counts[identity_with_prefix] += 1
        
        unified_sample = UnifiedPairedSample(
            identity=identity_with_prefix,
            source='deeplive',
            original_sample=sample,
            method=f"deeplive_{getattr(sample, 'effective_strategy', sample.strategy)}",
            has_landmarks=sample.has_landmarks,
            sample_id=sample.sample_id,
        )
        unified.append(unified_sample)
    
    # Log identity distribution
    unique_identities = len(identity_counts)
    max_samples_per_id = max(identity_counts.values()) if identity_counts else 0
    avg_samples_per_id = len(unified) / unique_identities if unique_identities > 0 else 0
    
    logger.info(f"Created {len(unified)} unified samples from DeepLive")
    logger.info(f"  - Unique identities: {unique_identities}")
    logger.info(f"  - Avg samples per identity: {avg_samples_per_id:.2f}")
    logger.info(f"  - Max samples per identity: {max_samples_per_id}")
    
    return unified


def create_unified_samples_from_visomaster(
    visomaster_samples: List[Any],
    logger: logging.Logger
) -> List[UnifiedPairedSample]:
    """
    Convert VisoMasterSample objects to UnifiedPairedSample wrappers.
    
    Args:
        visomaster_samples: List of VisoMasterSample objects
        logger: Logger
        
    Returns:
        List of UnifiedPairedSample wrappers
    """
    unified = []
    identity_counts = defaultdict(int)
    model_counts = defaultdict(int)
    tier_counts = defaultdict(int)
    
    for sample in visomaster_samples:
        # Use 'realpool_' prefix (NOT 'visomaster_') so that VisoMaster and DeepLive
        # samples from the same source video are grouped together during identity-
        # stratified splitting. This prevents identity leakage.
        identity = f"realpool_{sample.identity}"
        identity_counts[identity] += 1
        model_counts[sample.swap_model] += 1
        tier_counts[sample.tier] += 1
        
        unified_sample = UnifiedPairedSample(
            identity=identity,
            source='visomaster',
            original_sample=sample,
            method=f"visomaster_{sample.swap_model}",
            has_landmarks=False,  # VisoMaster has no landmarks
            sample_id=sample.sample_id,
        )
        unified.append(unified_sample)
    
    unique_identities = len(identity_counts)
    max_samples_per_id = max(identity_counts.values()) if identity_counts else 0
    avg_samples_per_id = len(unified) / unique_identities if unique_identities > 0 else 0
    
    logger.info(f"Created {len(unified)} unified samples from VisoMaster")
    logger.info(f"  - Unique identities: {unique_identities}")
    logger.info(f"  - Avg samples per identity: {avg_samples_per_id:.2f}")
    logger.info(f"  - Max samples per identity: {max_samples_per_id}")
    logger.info(f"  - Per-model: {dict(model_counts)}")
    logger.info(f"  - Per-tier: {dict(tier_counts)}")
    
    return unified


def create_unified_samples_from_proper_data(
    proper_data_samples: List[Any],
    logger: logging.Logger,
) -> List[UnifiedPairedSample]:
    """Wrap explicit proper-data inventory samples for the combined pipeline."""
    unified: List[UnifiedPairedSample] = []
    identity_counts = defaultdict(int)
    source_counts = defaultdict(int)
    quality_band_counts = defaultdict(int)
    face_scale_band_counts = defaultdict(int)
    split_group_counts = defaultdict(int)

    for sample in proper_data_samples:
        # WT-F split hygiene is defined at the split_group level, not raw identity.
        identity = f"realpool_splitgroup_{sample.split_group_id}"
        identity_counts[identity] += 1
        split_group_counts[sample.split_group_id] += 1
        source_counts[sample.source] += 1
        quality_band_counts[sample.quality_band] += 1
        face_scale_band_counts[sample.face_scale_band] += 1

        unified.append(
            UnifiedPairedSample(
                identity=identity,
                source=sample.source,
                original_sample=sample,
                method=sample.method,
                has_landmarks=False,
                sample_id=sample.sample_id,
            )
        )

    unique_identities = len(identity_counts)
    avg_per_id = len(unified) / unique_identities if unique_identities > 0 else 0

    logger.info(f"Created {len(unified)} unified samples from proper-data inventory")
    logger.info(f"  - Unique identities: {unique_identities}")
    logger.info(f"  - Unique split groups: {len(split_group_counts)}")
    logger.info(f"  - Avg samples per identity: {avg_per_id:.2f}")
    logger.info(f"  - Per-lane: {dict(source_counts)}")
    logger.info(f"  - Quality bands: {dict(quality_band_counts)}")
    logger.info(f"  - Face-scale bands: {dict(face_scale_band_counts)}")

    return unified


def _load_visomaster_policy_runtime():
    """Load the shared VisoMaster bad-data policy helpers when needed."""
    try:
        from viewer import visomaster_policy as policy_module

        return policy_module
    except Exception:
        import importlib.util
        import sys

        module_name = "viewer.visomaster_policy"
        cached = sys.modules.get(module_name)
        if cached is not None:
            return cached

        training_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        module_path = os.path.join(training_dir, "viewer", "visomaster_policy.py")
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not import viewer.visomaster_policy from {module_path}"
            )

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


def _load_visomaster_bad_data_policy_bundle(
    config: Dict[str, Any],
    combined_config: Dict[str, Any],
    logger: logging.Logger,
    *,
    required: bool = False,
):
    """Resolve the local April 17 policy packet used for explicit hint lanes."""
    policy_module = _load_visomaster_policy_runtime()
    policy_lookup_config = (
        config
        if isinstance(config, dict) and "combined_paired" in config
        else {"combined_paired": combined_config}
    )
    policy = policy_module.load_visomaster_bad_data_policy(policy_lookup_config)
    if policy is None:
        if required:
            raise ValueError(
                "WT-B explicit hint lanes require the VisoMaster bad-data policy packet. "
                "Set combined_paired.bad_data_policy_manifest / bad_data_policy_summary "
                "or place the April 17 files under training/debug/."
            )
        return None

    logger.info(
        "Loaded VisoMaster bad-data policy bundle: date=%s rows=%d manifest=%s",
        getattr(policy, "date_tag", "unknown"),
        len(getattr(policy, "rows_by_sample_id", {}) or {}),
        getattr(policy, "manifest_path", ""),
    )
    summary_path = getattr(policy, "summary_path", None)
    if summary_path:
        logger.info(f"  - Policy summary: {summary_path}")
    return policy


def _clone_visomaster_sample_with_policy(sample: Any, policy_row: Any) -> Any:
    """Return a VisoMaster sample with policy-corrected metadata attached."""
    manifest = dict(getattr(sample, "manifest", {}) or {})
    if getattr(policy_row, "original_video_name", None):
        manifest["original_video_name"] = policy_row.original_video_name
    if getattr(policy_row, "policy_label", None):
        manifest["policy_label"] = policy_row.policy_label
    if getattr(policy_row, "policy_lane", None):
        manifest["policy_lane"] = policy_row.policy_lane

    identity_delta = sample.identity_delta
    if getattr(policy_row, "identity_delta", None) is not None:
        identity_delta = float(policy_row.identity_delta)

    return replace(
        sample,
        swap_model=getattr(policy_row, "swap_model", "") or sample.swap_model,
        tier=getattr(policy_row, "tier", "") or sample.tier,
        identity_delta=identity_delta,
        manifest=manifest,
    )


def _clone_teams_sample_with_policy(sample: Any, policy_row: Any) -> Any:
    """Return a Teams sample with policy-corrected identity metadata attached."""
    return replace(
        sample,
        original_video_name=(
            getattr(policy_row, "original_video_name", None) or sample.original_video_name
        ),
    )


def _select_visomaster_hints_samples(
    raw_visomaster_samples: List[Any],
    policy_bundle: Any,
    logger: logging.Logger,
) -> List[Any]:
    """Select retained baseline hints from the raw VisoMaster sample pool."""
    policy_module = _load_visomaster_policy_runtime()

    selected: List[Any] = []
    skipped_non_policy = 0
    skipped_wrong_lane = 0

    for sample in raw_visomaster_samples:
        policy_row = policy_bundle.row_for(sample.sample_id) if policy_bundle else None
        if policy_row is None:
            skipped_non_policy += 1
            continue
        if getattr(policy_row, "in_teams_pair_complete", False):
            skipped_wrong_lane += 1
            continue
        if (
            getattr(policy_row, "policy_action", "") != policy_module.KEEP_ACTION
            or getattr(policy_row, "policy_label", "") != policy_module.BASELINE_LABEL
        ):
            skipped_wrong_lane += 1
            continue
        selected.append(_clone_visomaster_sample_with_policy(sample, policy_row))

    logger.info(
        "VisoMaster baseline hint selection: kept=%d skipped_non_policy=%d skipped_wrong_lane=%d raw_total=%d",
        len(selected),
        skipped_non_policy,
        skipped_wrong_lane,
        len(raw_visomaster_samples),
    )
    return selected


def _select_clean_teams_samples(
    raw_teams_samples: List[Any],
    policy_bundle: Any,
    logger: logging.Logger,
) -> List[Any]:
    """Keep only direct Teams rows that are not reclassified into weak-signal hints."""
    clean_samples: List[Any] = []
    skipped_policy_rows = 0

    for sample in raw_teams_samples:
        policy_row = policy_bundle.row_for(sample.sample_id) if policy_bundle else None
        if policy_row is not None and getattr(policy_row, "in_teams_pair_complete", False):
            skipped_policy_rows += 1
            continue
        clean_samples.append(sample)

    logger.info(
        "Teams clean-lane policy filter: kept=%d skipped_policy_rows=%d raw_total=%d",
        len(clean_samples),
        skipped_policy_rows,
        len(raw_teams_samples),
    )
    return clean_samples


def _select_hint_teams_samples(
    raw_teams_samples: List[Any],
    policy_bundle: Any,
    logger: logging.Logger,
) -> List[Any]:
    """Select retained Teams-played hint rows from the raw Teams pool."""
    policy_module = _load_visomaster_policy_runtime()

    selected: List[Any] = []
    skipped_non_policy = 0
    skipped_wrong_lane = 0

    for sample in raw_teams_samples:
        policy_row = policy_bundle.row_for(sample.sample_id) if policy_bundle else None
        if policy_row is None:
            skipped_non_policy += 1
            continue
        if not getattr(policy_row, "in_teams_pair_complete", False):
            skipped_wrong_lane += 1
            continue
        if (
            getattr(policy_row, "policy_action", "") != policy_module.KEEP_ACTION
            or getattr(policy_row, "policy_label", "") != policy_module.TEAMS_LABEL
        ):
            skipped_wrong_lane += 1
            continue
        selected.append(_clone_teams_sample_with_policy(sample, policy_row))

    logger.info(
        "Teams weak-signal hint selection: kept=%d skipped_non_policy=%d skipped_wrong_lane=%d raw_total=%d",
        len(selected),
        skipped_non_policy,
        skipped_wrong_lane,
        len(raw_teams_samples),
    )
    return selected


def create_unified_samples_from_visomaster_hints(
    visomaster_samples: List[Any],
    logger: logging.Logger,
) -> List[UnifiedPairedSample]:
    """Convert retained baseline hint rows into explicit training samples."""
    unified: List[UnifiedPairedSample] = []
    identity_counts = defaultdict(int)
    model_counts = defaultdict(int)
    tier_counts = defaultdict(int)

    for sample in visomaster_samples:
        identity = f"realpool_{sample.identity}"
        identity_counts[identity] += 1
        model_counts[sample.swap_model] += 1
        tier_counts[sample.tier] += 1

        unified.append(
            UnifiedPairedSample(
                identity=identity,
                source="visomaster_hints",
                original_sample=sample,
                method="visomaster_hints",
                has_landmarks=False,
                sample_id=sample.sample_id,
            )
        )

    unique_identities = len(identity_counts)
    max_samples_per_id = max(identity_counts.values()) if identity_counts else 0
    avg_samples_per_id = len(unified) / unique_identities if unique_identities > 0 else 0

    logger.info(f"Created {len(unified)} unified samples from VisoMaster hints")
    logger.info(f"  - Unique identities: {unique_identities}")
    logger.info(f"  - Avg samples per identity: {avg_samples_per_id:.2f}")
    logger.info(f"  - Max samples per identity: {max_samples_per_id}")
    logger.info(f"  - Per-model: {dict(model_counts)}")
    logger.info(f"  - Per-tier: {dict(tier_counts)}")

    return unified


def create_unified_samples_from_visomaster_hints_teams(
    teams_samples: List[Any],
    logger: logging.Logger,
) -> List[UnifiedPairedSample]:
    """Convert retained Teams-played weak-signal rows into explicit training samples."""
    unified: List[UnifiedPairedSample] = []
    identity_counts = defaultdict(int)
    strategy_counts = defaultdict(int)

    for sample in teams_samples:
        identity_raw = sample.original_video_name
        if identity_raw.startswith("cropped_"):
            identity_raw = identity_raw[len("cropped_"):]
        if identity_raw.endswith(".mp4"):
            identity_raw = identity_raw[:-4]
        identity = f"realpool_{identity_raw}"
        identity_counts[identity] += 1
        strategy_counts[sample.strategy] += 1

        unified.append(
            UnifiedPairedSample(
                identity=identity,
                source="visomaster_hints_teams",
                original_sample=sample,
                method="visomaster_hints_teams",
                has_landmarks=False,
                sample_id=sample.sample_id,
            )
        )

    unique_identities = len(identity_counts)
    max_per_id = max(identity_counts.values()) if identity_counts else 0
    avg_per_id = len(unified) / unique_identities if unique_identities else 0

    logger.info(f"Created {len(unified)} unified samples from VisoMaster Teams hints")
    logger.info(f"  - Unique identities: {unique_identities}")
    logger.info(f"  - Avg samples per identity: {avg_per_id:.2f}")
    logger.info(f"  - Max samples per identity: {max_per_id}")
    logger.info(f"  - Underlying Teams strategies: {dict(strategy_counts)}")

    return unified


def create_unified_samples_from_visomaster_enhanced(
    enhanced_samples: List[Any],
    logger: logging.Logger
) -> List[UnifiedPairedSample]:
    """
    Convert VisoMasterEnhancedSample objects to UnifiedPairedSample wrappers.

    Identity uses the **same** ``realpool_`` prefix as DeepLive/VisoMaster so
    that the same person never appears in multiple splits (preventing leakage).
    Enhanced samples share identity with their originals.

    Method is set to ``visomaster_enhanced_{enhancer}`` to enable per-enhancer
    validation metrics while all samples map to the single
    ``visomaster_enhanced_fake`` family for weighted sampling.

    Args:
        enhanced_samples: List of VisoMasterEnhancedSample objects
        logger: Logger

    Returns:
        List of UnifiedPairedSample wrappers
    """
    unified = []
    identity_counts = defaultdict(int)
    enhancer_counts = defaultdict(int)
    tier_counts = defaultdict(int)

    for sample in enhanced_samples:
        # Identity matches the original VisoMaster sample — prevents split leakage
        identity = f"realpool_{sample.identity}"
        identity_counts[identity] += 1
        enhancer_counts[sample.enhancer] += 1
        tier_counts[sample.tier] += 1

        unified_sample = UnifiedPairedSample(
            identity=identity,
            source='visomaster_enhanced',
            original_sample=sample,
            method=f"visomaster_enhanced_{sample.enhancer}",
            has_landmarks=False,
            sample_id=sample.sample_id,
        )
        unified.append(unified_sample)

    unique_identities = len(identity_counts)
    max_samples_per_id = max(identity_counts.values()) if identity_counts else 0
    avg_samples_per_id = len(unified) / unique_identities if unique_identities > 0 else 0

    logger.info(f"Created {len(unified)} unified samples from VisoMaster Enhanced")
    logger.info(f"  - Unique identities: {unique_identities}")
    logger.info(f"  - Avg samples per identity: {avg_samples_per_id:.2f}")
    logger.info(f"  - Max samples per identity: {max_samples_per_id}")
    logger.info(f"  - Per-enhancer: {dict(enhancer_counts)}")
    logger.info(f"  - Per-tier: {dict(tier_counts)}")

    return unified


def create_unified_samples_from_visomaster_teams_enhanced(
    merged_samples: List[Any],
    logger: logging.Logger,
    sampling_family_key: str = "visomaster_enhanced_fake",
) -> List[UnifiedPairedSample]:
    """
    Convert merged resolver samples to one UnifiedPairedSample per base sample.

    Unlike the older clean enhanced path, this does not explode one base sample
    into one sample object per enhancer. Branch selection happens at iteration
    time instead.
    """
    unified: List[UnifiedPairedSample] = []
    identity_counts: Dict[str, int] = defaultdict(int)
    domain_counts: Dict[str, int] = defaultdict(int)
    enhancer_counts: Dict[str, int] = defaultdict(int)

    for sample in merged_samples:
        identity = f"realpool_{sample.identity}"
        identity_counts[identity] += 1
        domain_counts[sample.companion_domain] += 1
        for enhancer in sample.available_enhancers:
            enhancer_counts[enhancer] += 1

        unified.append(
            UnifiedPairedSample(
                identity=identity,
                source="visomaster_teams_enhanced",
                original_sample=sample,
                method=sample.original_method,
                has_landmarks=False,
                sample_id=sample.sample_id,
                method_variants=sample.method_variants,
                sampling_family_key=sampling_family_key,
            )
        )

    unique_identities = len(identity_counts)
    max_samples_per_id = max(identity_counts.values()) if identity_counts else 0
    avg_samples_per_id = len(unified) / unique_identities if unique_identities > 0 else 0

    logger.info(f"Created {len(unified)} unified samples from VisoMaster Teams-Enhanced")
    logger.info(f"  - Unique identities: {unique_identities}")
    logger.info(f"  - Avg samples per identity: {avg_samples_per_id:.2f}")
    logger.info(f"  - Max samples per identity: {max_samples_per_id}")
    logger.info(f"  - Companion domains: {dict(domain_counts)}")
    logger.info(f"  - Available enhancer coverage: {dict(enhancer_counts)}")

    return unified


def create_unified_samples_from_visomaster_res_variant(
    res_variant_samples: List[Any],
    logger: logging.Logger,
) -> List[UnifiedPairedSample]:
    """
    Convert VisoMaster resolution-variant samples to UnifiedPairedSample wrappers.

    Each resolution variant is emitted as source ``visomaster_res_variant`` with
    method ``visomaster_inswapper128_res{resolution}`` so per-resolution
    validation metrics are available.

    Identity uses the ``realpool_`` prefix (shared pool) to avoid leakage.
    """
    unified = []
    identity_counts: Dict[str, int] = defaultdict(int)
    res_counts: Dict[str, int] = defaultdict(int)
    tier_counts: Dict[str, int] = defaultdict(int)

    for sample in res_variant_samples:
        identity = f"realpool_{sample.identity}"
        identity_counts[identity] += 1
        res_counts[sample.enhancer] += 1  # pseudo-enhancer encodes resolution
        tier_counts[sample.tier] += 1

        unified_sample = UnifiedPairedSample(
            identity=identity,
            source='visomaster_res_variant',
            original_sample=sample,
            method=f"visomaster_{sample.enhancer}",  # e.g. visomaster_inswapper128_res256
            has_landmarks=False,
            sample_id=sample.sample_id,
        )
        unified.append(unified_sample)

    unique_identities = len(identity_counts)
    avg_per_id = len(unified) / unique_identities if unique_identities > 0 else 0

    logger.info(f"Created {len(unified)} unified samples from VisoMaster Res-Variant")
    logger.info(f"  - Unique identities: {unique_identities}")
    logger.info(f"  - Avg samples per identity: {avg_per_id:.2f}")
    logger.info(f"  - Per-resolution: {dict(res_counts)}")
    logger.info(f"  - Per-tier: {dict(tier_counts)}")

    return unified


# =============================================================================
# Teams Passthrough Sample Dataclass & Helpers
# =============================================================================

@dataclass
class TeamsSample:
    """Raw metadata for a single Teams-passthrough sample."""
    sample_id: str
    strategy: str
    original_video_name: str
    frame_count: int
    gcs_bucket: str
    real_prefix: str   # e.g. "samples/<id>/frames/real/"
    fake_prefix: str   # e.g. "samples/<id>/frames/fake/"


def _load_teams_frame_from_blob(bucket: Any, blob_path: str) -> Any:
    import cv2
    import numpy as np

    buf = bucket.blob(blob_path).download_as_bytes()
    img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode Teams frame: {blob_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_teams_frame_map(
    bucket: Any,
    prefix: str,
    frame_indices: Sequence[int],
    *,
    executor: Optional[Any] = None,
    parallel_download_workers: int = DEFAULT_PARALLEL_GCS_DOWNLOAD_WORKERS,
) -> Dict[int, Any]:
    """Load Teams JPG frames keyed by frame index."""
    frame_map: Dict[int, Any] = {}
    valid_indices = list(frame_indices)

    if not valid_indices:
        return frame_map

    def _load_one(idx: int) -> Tuple[int, Any]:
        blob_path = f"{prefix}frame_{idx:04d}.jpg"
        return idx, _load_teams_frame_from_blob(bucket, blob_path)

    def _record(idx: int, img: Any) -> None:
        frame_map[idx] = img

    def _log_failure(idx: int, exc: Exception) -> None:
        blob_path = f"{prefix}frame_{idx:04d}.jpg"
        logger.debug("Failed to load Teams frame %s: %s", blob_path, exc)

    def _load_sequential() -> None:
        for idx in valid_indices:
            try:
                frame_idx, img = _load_one(idx)
            except Exception as exc:
                _log_failure(idx, exc)
                continue
            _record(frame_idx, img)

    if executor is None:
        max_workers = max(1, int(parallel_download_workers or 1))
        if max_workers <= 1 or len(valid_indices) <= 1:
            _load_sequential()
            return frame_map

        with ThreadPoolExecutor(max_workers=min(max_workers, len(valid_indices))) as pool:
            future_to_idx = {
                pool.submit(_load_one, idx): idx
                for idx in valid_indices
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    frame_idx, img = future.result()
                except Exception as exc:
                    _log_failure(idx, exc)
                    continue
                _record(frame_idx, img)
        return frame_map

    future_to_idx = {
        executor.submit(_load_one, idx): idx
        for idx in valid_indices
    }
    for future in as_completed(future_to_idx):
        idx = future_to_idx[future]
        try:
            frame_idx, img = future.result()
        except Exception as exc:
            _log_failure(idx, exc)
            continue
        _record(frame_idx, img)

    return frame_map


def _read_json_uri(uri: str) -> Optional[Dict[str, Any]]:
    try:
        fs, path = url_to_fs(uri)
        if not fs.exists(path):
            return None
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json_uri(uri: str, payload: Dict[str, Any], log: logging.Logger) -> None:
    try:
        fs, path = url_to_fs(uri)
        parent = os.path.dirname(path)
        if parent and not fs.exists(parent):
            fs.makedirs(parent, exist_ok=True)
        with fs.open(path, "w") as f:
            json.dump(payload, f)
        log.info("Wrote Teams discovery cache: %s", uri)
    except Exception as exc:
        log.warning("Failed to write Teams discovery cache %s: %s", uri, exc)


def _list_teams_manifest_blobs(bucket: Any) -> List[Any]:
    return list(bucket.list_blobs(prefix="samples/", match_glob="**/manifest.json"))


def _build_teams_discovery_signature(manifest_blobs: Sequence[Any]) -> Dict[str, Any]:
    digest = hashlib.sha256()

    for blob in manifest_blobs:
        updated = getattr(blob, "updated", None)
        updated_iso = updated.isoformat() if updated else ""
        size = int(getattr(blob, "size", 0) or 0)
        digest.update(f"{blob.name}|{updated_iso}|{size}\n".encode("utf-8"))

    return {
        "manifest_count": len(manifest_blobs),
        "manifest_hash": digest.hexdigest(),
    }


def _cache_to_teams_samples(cache_payload: Dict[str, Any]) -> List[TeamsSample]:
    samples: List[TeamsSample] = []

    for row in cache_payload.get("samples", []) or []:
        samples.append(
            TeamsSample(
                sample_id=row["sample_id"],
                strategy=row.get("strategy", "unknown"),
                original_video_name=row.get("original_video_name", row["sample_id"]),
                frame_count=int(row.get("frame_count") or 0),
                gcs_bucket=row.get("gcs_bucket", ""),
                real_prefix=row["real_prefix"],
                fake_prefix=row["fake_prefix"],
            )
        )

    return samples


def _teams_samples_to_cache_rows(samples: Sequence[TeamsSample]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for sample in samples:
        rows.append(
            {
                "sample_id": sample.sample_id,
                "strategy": sample.strategy,
                "original_video_name": sample.original_video_name,
                "frame_count": int(sample.frame_count),
                "gcs_bucket": sample.gcs_bucket,
                "real_prefix": sample.real_prefix,
                "fake_prefix": sample.fake_prefix,
            }
        )

    return rows


def discover_teams_passthrough_samples(
    gcs_bucket: str,
    gcs_project: Optional[str] = None,
    require_pair_complete: bool = True,
    cache_manifest_uri: Optional[str] = None,
    cache_max_age_hours: float = 12.0,
    cache_validate_listing: bool = True,
    cache_revision: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> List[TeamsSample]:
    """
    Discover Teams-passthrough samples from GCS.

    Bucket layout::

        samples/{sample_id}/manifest.json
        samples/{sample_id}/frames/real/frame_NNNN.jpg
        samples/{sample_id}/frames/fake/frame_NNNN.jpg

    ``manifest.json`` is expected to contain at least::

        { "sample_id": ..., "strategy": ..., "original_video_name": ...,
          "frame_count": ..., "pair_complete": true|false }

    Args:
        gcs_bucket: GCS bucket name.
        gcs_project: GCP project (default: env ``GOOGLE_CLOUD_PROJECT``).
        require_pair_complete: If True, skip samples where ``pair_complete`` is
            not ``true``.
        cache_manifest_uri: Optional URI (local or gs://) for discovery cache JSON.
        cache_max_age_hours: Max cache age before refresh.
        cache_validate_listing: If True, compare cached manifest signature with GCS.
        cache_revision: Optional manual revision token to invalidate stale cache.
        logger: Optional logger.

    Returns:
        List of :class:`TeamsSample` instances.
    """
    log = logger or logging.getLogger(__name__)
    project = gcs_project or os.environ.get("GOOGLE_CLOUD_PROJECT", "train-cvit2")
    cache_config = {
        "gcs_bucket": gcs_bucket,
        "require_pair_complete": bool(require_pair_complete),
        "cache_revision": cache_revision or "",
    }

    client = None
    bucket = None
    manifest_blobs: Optional[List[Any]] = None

    if cache_manifest_uri:
        cached = _read_json_uri(cache_manifest_uri)
        if cached:
            cache_age = time.time() - float(cached.get("timestamp") or 0.0)
            config_matches = (cached.get("config") or {}) == cache_config
            cache_too_old = cache_age > float(cache_max_age_hours) * 3600.0

            if config_matches and not cache_too_old:
                if cache_validate_listing:
                    client = storage.Client(project=project)
                    bucket = client.bucket(gcs_bucket)
                    manifest_blobs = _list_teams_manifest_blobs(bucket)
                    current_signature = _build_teams_discovery_signature(manifest_blobs)
                    if current_signature == (cached.get("listing_signature") or {}):
                        samples = _cache_to_teams_samples(cached)
                        log.info(
                            "Loaded %d Teams samples from discovery cache: %s",
                            len(samples),
                            cache_manifest_uri,
                        )
                        return samples
                    log.info(
                        "Teams discovery cache invalidated because listing changed: %s",
                        cache_manifest_uri,
                    )
                else:
                    samples = _cache_to_teams_samples(cached)
                    log.info(
                        "Loaded %d Teams samples from discovery cache (listing validation disabled): %s",
                        len(samples),
                        cache_manifest_uri,
                    )
                    return samples
            else:
                reason = "config mismatch" if not config_matches else "cache expired"
                log.info("Teams discovery cache miss (%s): %s", reason, cache_manifest_uri)

    if client is None:
        client = storage.Client(project=project)
    if bucket is None:
        bucket = client.bucket(gcs_bucket)

    # Discover sample directories by listing manifest.json files
    if manifest_blobs is None:
        manifest_blobs = _list_teams_manifest_blobs(bucket)

    samples: List[TeamsSample] = []
    skipped_incomplete = 0
    skipped_error = 0

    for blob in manifest_blobs:
        try:
            manifest_data = json.loads(blob.download_as_text())
        except Exception as exc:
            log.warning("Failed to read Teams manifest %s: %s", blob.name, exc)
            skipped_error += 1
            continue

        if require_pair_complete and not manifest_data.get("pair_complete", False):
            skipped_incomplete += 1
            continue

        sample_id = manifest_data.get("sample_id", "")
        if not sample_id:
            # Derive from path: samples/<sample_id>/manifest.json
            parts = blob.name.split("/")
            if len(parts) >= 2:
                sample_id = parts[1]

        strategy = manifest_data.get("strategy", "unknown")
        original_video_name = manifest_data.get("original_video_name", sample_id)
        frame_count = int(manifest_data.get("frame_count") or 0)
        # Build prefixes
        base = f"samples/{sample_id}/frames"
        samples.append(TeamsSample(
            sample_id=sample_id,
            strategy=strategy,
            original_video_name=original_video_name,
            frame_count=frame_count,
            gcs_bucket=gcs_bucket,
            real_prefix=f"{base}/real/",
            fake_prefix=f"{base}/fake/",
        ))

    log.info(
        "Teams discovery: found %d complete samples (skipped %d incomplete, %d errors) in gs://%s",
        len(samples), skipped_incomplete, skipped_error, gcs_bucket,
    )

    if cache_manifest_uri:
        cache_payload = {
            "version": 1,
            "timestamp": time.time(),
            "config": cache_config,
            "listing_signature": _build_teams_discovery_signature(manifest_blobs),
            "samples": _teams_samples_to_cache_rows(samples),
        }
        _write_json_uri(cache_manifest_uri, cache_payload, log=log)

    return samples


def create_unified_samples_from_teams(
    teams_samples: List[TeamsSample],
    logger: logging.Logger,
) -> List[UnifiedPairedSample]:
    """
    Convert :class:`TeamsSample` objects to :class:`UnifiedPairedSample` wrappers.

    Identity uses the **same** ``realpool_`` prefix as DeepLive/VisoMaster so
    that the same person is never split across train/val (preventing leakage).
    """
    unified: List[UnifiedPairedSample] = []
    identity_counts: Dict[str, int] = defaultdict(int)
    strategy_counts: Dict[str, int] = defaultdict(int)

    for sample in teams_samples:
        # Extract identity — same logic as DeepLive
        identity_raw = sample.original_video_name
        if identity_raw.startswith("cropped_"):
            identity_raw = identity_raw[len("cropped_"):]
        if identity_raw.endswith(".mp4"):
            identity_raw = identity_raw[:-4]
        identity = f"realpool_{identity_raw}"
        identity_counts[identity] += 1
        strategy_counts[sample.strategy] += 1

        method = f"deeplive_teams_{sample.strategy}"

        unified.append(UnifiedPairedSample(
            identity=identity,
            source="deeplive_teams",
            original_sample=sample,
            method=method,
            has_landmarks=False,  # Teams capturing does not preserve landmarks
            sample_id=sample.sample_id,
        ))

    unique_identities = len(identity_counts)
    max_per_id = max(identity_counts.values()) if identity_counts else 0
    avg_per_id = len(unified) / unique_identities if unique_identities else 0

    logger.info(f"Created {len(unified)} unified samples from Teams passthrough")
    logger.info(f"  - Unique identities: {unique_identities}")
    logger.info(f"  - Avg samples per identity: {avg_per_id:.2f}")
    logger.info(f"  - Max samples per identity: {max_per_id}")
    logger.info(f"  - Per-strategy: {dict(strategy_counts)}")

    return unified


# =============================================================================
# Identity-Stratified Splitting
# =============================================================================

def split_samples_by_identity(
    samples: List[UnifiedPairedSample],
    train_split: float,
    val_split: float,
    seed: int,
    logger: logging.Logger,
    split_mode: str = "shuffle",
) -> Tuple[List[UnifiedPairedSample], List[UnifiedPairedSample], List[UnifiedPairedSample]]:
    """
    Split samples into train/val/test sets BY IDENTITY.
    
    CRITICAL: This splits by identity to prevent data leakage.
    No identity will appear in multiple splits.
    
    Args:
        samples: List of UnifiedPairedSample
        train_split: Proportion for training
        val_split: Proportion for validation
        seed: Random seed
        logger: Logger
        split_mode: `shuffle` reproduces the legacy global shuffle behavior.
            `hash_stable` assigns identities independently by hash threshold so
            existing identities do not move when new identities are added.
        
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    split_mode = normalize_method_name(split_mode or "shuffle")
    if split_mode not in {"shuffle", "hash_stable"}:
        raise ValueError(
            f"Unsupported identity split mode '{split_mode}'. "
            "Expected one of: shuffle, hash_stable."
        )

    rng = random.Random(seed)
    
    # Group samples by identity
    by_identity = defaultdict(list)
    for sample in samples:
        by_identity[sample.identity].append(sample)
    
    # Get list of unique identities and shuffle
    identities = list(by_identity.keys())
    rng.shuffle(identities)
    
    # Count identities by source for logging
    df40_identities = [i for i in identities if i.startswith('df40_')]
    proper_splitgroup_identities = [
        i for i in identities if i.startswith('realpool_splitgroup_')
    ]
    # DeepLive / VisoMaster share the legacy realpool_ prefix. Proper-data stays
    # in the same broad pool but is keyed by WT-F split group.
    realpool_identities = [
        i
        for i in identities
        if i.startswith('realpool_') and not i.startswith('realpool_splitgroup_')
    ]
    
    logger.info(f"Total unique identities: {len(identities)}")
    logger.info(f"  - DF40 identities: {len(df40_identities)}")
    logger.info(
        f"  - Real pool identities (DeepLive + VisoMaster merged): {len(realpool_identities)}"
    )
    if proper_splitgroup_identities:
        logger.info(
            "  - Proper-data split-group identities: %d",
            len(proper_splitgroup_identities),
        )
    
    n_identities = len(identities)
    if n_identities == 0:
        logger.warning("split_samples_by_identity received 0 identities.")
        return [], [], []

    if split_mode == "hash_stable":
        train_identities = set()
        val_identities = set()
        test_identities = set()
        val_cutoff = train_split + val_split

        for identity in identities:
            frac = _identity_hash_fraction(identity, seed)
            if frac < train_split:
                train_identities.add(identity)
            elif frac < val_cutoff:
                val_identities.add(identity)
            else:
                test_identities.add(identity)
    else:
        # Legacy global-shuffle split. This preserves historical behavior, but
        # identities can move when later experiments add more samples.
        n_train_ids = int(n_identities * train_split)
        n_val_ids = int(n_identities * val_split)

        train_identities = set(identities[:n_train_ids])
        val_identities = set(identities[n_train_ids:n_train_ids + n_val_ids])
        test_identities = set(identities[n_train_ids + n_val_ids:])
    
    # Assign ALL samples for each identity to that identity's split
    train_samples = []
    val_samples = []
    test_samples = []
    
    for sample in samples:
        if sample.identity in train_identities:
            train_samples.append(sample)
        elif sample.identity in val_identities:
            val_samples.append(sample)
        else:
            test_samples.append(sample)
    
    # Final shuffle
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)
    
    # Log split statistics
    n_total = len(samples)
    logger.info(f"Data split BY IDENTITY (seed={seed}, mode={split_mode}):")
    logger.info(f"  - Train identities: {len(train_identities)} ({len(train_identities)/n_identities*100:.1f}%)")
    logger.info(f"  - Val identities: {len(val_identities)} ({len(val_identities)/n_identities*100:.1f}%)")
    logger.info(f"  - Test identities: {len(test_identities)} ({len(test_identities)/n_identities*100:.1f}%)")
    logger.info(f"  - Train samples: {len(train_samples)} ({len(train_samples)/n_total*100:.1f}%)")
    logger.info(f"  - Val samples: {len(val_samples)} ({len(val_samples)/n_total*100:.1f}%)")
    logger.info(f"  - Test samples: {len(test_samples)} ({len(test_samples)/n_total*100:.1f}%)")
    
    # Log source distribution per split
    def count_by_source(sample_list):
        counts = defaultdict(int)
        for s in sample_list:
            counts[s.source] += 1
        return dict(counts)
    
    logger.info(f"  - Train by source: {count_by_source(train_samples)}")
    logger.info(f"  - Val by source: {count_by_source(val_samples)}")
    logger.info(f"  - Test by source: {count_by_source(test_samples)}")
    
    # Sanity check: verify no identity overlap
    train_ids_check = set(s.identity for s in train_samples)
    val_ids_check = set(s.identity for s in val_samples)
    test_ids_check = set(s.identity for s in test_samples)
    
    overlaps = (
        train_ids_check & val_ids_check,
        train_ids_check & test_ids_check,
        val_ids_check & test_ids_check
    )
    
    if any(overlaps):
        logger.error("IDENTITY LEAKAGE DETECTED!")
        raise ValueError("Identity leakage between splits!")
    else:
        logger.info("  ✓ No identity overlap between splits (leakage check passed)")
    
    return train_samples, val_samples, test_samples


def _resolve_df40_pair_json_path(
    pair_json_path: str,
) -> str:
    """Resolve DF40 pair JSON path relative to training root when needed."""
    if os.path.isabs(pair_json_path):
        return pair_json_path
    training_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(training_dir, pair_json_path)


def _load_df40_methods_for_orientation(
    pair_json_path: str,
    orientation: str,
    logger: logging.Logger,
) -> List[str]:
    """
    Load DF40 fake methods for a requested orientation from pair-matching JSON.

    This enables holdout policies such as:
      - holdout all target_source methods
      - train on source_target methods
    """
    orientation_norm = normalize_method_name(orientation)
    if orientation_norm not in {"target_source", "source_target"}:
        raise ValueError(
            f"Unsupported DF40 orientation for holdout: {orientation}. "
            "Expected one of {target_source, source_target}."
        )

    resolved_path = _resolve_df40_pair_json_path(pair_json_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"DF40 pair JSON not found: {resolved_path}")

    with open(resolved_path, "r") as f:
        pair_data = json.load(f)
    method_orientation = pair_data.get("method_orientation", {}) or {}
    if not method_orientation:
        raise ValueError(
            "DF40 pair JSON missing `method_orientation`; cannot resolve orientation-based holdout methods."
        )

    methods = sorted(
        {
            normalize_method_name(method)
            for method, method_orientation_value in method_orientation.items()
            if normalize_method_name(method_orientation_value) == orientation_norm
        }
    )
    logger.info(
        "Resolved %d DF40 methods for orientation `%s`: %s",
        len(methods),
        orientation_norm,
        methods,
    )
    return methods


def _resolve_holdout_methods(
    holdout_cfg: Dict[str, Any],
    df40_cfg: Dict[str, Any],
    logger: logging.Logger,
) -> List[str]:
    """
    Resolve method-holdout list from explicit methods + optional DF40 orientation.
    """
    resolved: List[str] = []

    explicit_methods = (
        holdout_cfg.get("methods")
        or holdout_cfg.get("holdout_methods")
        or holdout_cfg.get("fake_methods")
        or []
    )
    for method in explicit_methods:
        method_norm = normalize_method_name(str(method))
        if method_norm:
            resolved.append(method_norm)

    df40_orientation = normalize_method_name(holdout_cfg.get("df40_orientation", ""))
    if df40_orientation in {"target_source", "source_target"}:
        pair_json_path = holdout_cfg.get("df40_pair_json") or df40_cfg.get(
            "pair_json", "dataset/df40_pairs/df40-pair-matching.json"
        )
        df40_methods = _load_df40_methods_for_orientation(
            pair_json_path=str(pair_json_path),
            orientation=df40_orientation,
            logger=logger,
        )
        resolved.extend(df40_methods)
        logger.info(
            "Holdout config added DF40 `%s` orientation methods (count=%d).",
            df40_orientation,
            len(df40_methods),
        )

    deduped: List[str] = []
    seen = set()
    for method in resolved:
        if method not in seen:
            deduped.append(method)
            seen.add(method)
    return deduped


def split_samples_by_method_holdout(
    samples: List[UnifiedPairedSample],
    holdout_methods: Sequence[str],
    train_split: float,
    val_split: float,
    seed: int,
    logger: logging.Logger,
    split_mode: str = "shuffle",
    max_samples_per_method: Optional[int] = None,
    per_method_caps: Optional[Dict[str, int]] = None,
) -> Tuple[List[UnifiedPairedSample], List[UnifiedPairedSample], List[UnifiedPairedSample]]:
    """
    Split samples with strict method holdout semantics.

    - `test_samples` (holdout) contains only methods listed in `holdout_methods`.
    - `train_samples` and `val_samples` are sampled from remaining methods.
    - Method overlap between train and holdout is disallowed.

    Note:
        This mode enforces method-level OOD for holdout. Identity overlap between
        seen and held-out methods is logged as a warning (not hard-failed), because
        strict no-overlap can remove too much data for some method mixes.
    """
    holdout_methods_norm = {
        normalize_method_name(m) for m in holdout_methods if m and str(m).strip()
    }
    if not holdout_methods_norm:
        raise ValueError(
            "method holdout mode requires non-empty combined_paired.holdout.methods"
        )

    holdout_samples: List[UnifiedPairedSample] = []
    seen_samples: List[UnifiedPairedSample] = []
    for sample in samples:
        method_norm = normalize_method_name(sample.method)
        if method_norm in holdout_methods_norm:
            holdout_samples.append(sample)
        else:
            seen_samples.append(sample)

    if not holdout_samples:
        raise ValueError(
            "Method holdout split produced 0 holdout samples. "
            "Check combined_paired.holdout.methods against discovered methods."
        )
    if not seen_samples:
        raise ValueError(
            "Method holdout split produced 0 train/val samples. "
            "Holdout methods consumed all data."
        )

    total_seen_split = train_split + val_split
    if total_seen_split <= 0:
        raise ValueError("train_split + val_split must be > 0 for method holdout mode.")
    train_ratio_seen = train_split / total_seen_split
    val_ratio_seen = 1.0 - train_ratio_seen

    # Reuse identity-based split for seen methods to preserve no-leakage behavior
    # within the seen pool. Merge val/test seen slices into a single in-dist val set.
    seen_train, seen_val_a, seen_val_b = split_samples_by_identity(
        seen_samples,
        train_split=train_ratio_seen,
        val_split=val_ratio_seen,
        seed=seed,
        logger=logger,
        split_mode=split_mode,
    )
    seen_val = seen_val_a + seen_val_b

    if max_samples_per_method is not None:
        max_samples_per_method = int(max_samples_per_method)
        if max_samples_per_method <= 0:
            max_samples_per_method = None
    method_caps = {
        normalize_method_name(k): int(v)
        for k, v in (per_method_caps or {}).items()
        if k and v and int(v) > 0
    }

    rng = random.Random(seed)
    rng.shuffle(seen_train)
    rng.shuffle(seen_val)
    rng.shuffle(holdout_samples)

    if max_samples_per_method or method_caps:
        by_method: Dict[str, List[UnifiedPairedSample]] = defaultdict(list)
        for sample in holdout_samples:
            by_method[normalize_method_name(sample.method)].append(sample)

        truncated_counts: Dict[str, int] = {}
        capped_holdout_samples: List[UnifiedPairedSample] = []
        for method_name in sorted(by_method.keys()):
            method_samples = by_method[method_name]
            rng.shuffle(method_samples)
            cap = method_caps.get(method_name, max_samples_per_method)
            if cap is None or len(method_samples) <= cap:
                capped_holdout_samples.extend(method_samples)
                continue
            capped_holdout_samples.extend(method_samples[:cap])
            truncated_counts[method_name] = len(method_samples) - cap

        holdout_samples = capped_holdout_samples
        rng.shuffle(holdout_samples)
        if truncated_counts:
            logger.info(
                "Applied holdout per-method caps: %s",
                dict(sorted(truncated_counts.items())),
            )
        if not holdout_samples:
            raise ValueError(
                "Holdout caps removed all holdout samples. "
                "Increase max_samples_per_method/per_method_caps."
            )

    train_methods = {normalize_method_name(s.method) for s in seen_train}
    holdout_methods_effective = {normalize_method_name(s.method) for s in holdout_samples}
    overlap_methods = train_methods & holdout_methods_effective
    if overlap_methods:
        raise RuntimeError(
            "Method holdout violated: overlapping methods in train and holdout: "
            f"{sorted(overlap_methods)}"
        )

    train_ids = {s.identity for s in seen_train}
    val_ids = {s.identity for s in seen_val}
    holdout_ids = {s.identity for s in holdout_samples}
    overlap_train_holdout = train_ids & holdout_ids
    overlap_val_holdout = val_ids & holdout_ids
    if overlap_train_holdout or overlap_val_holdout:
        logger.warning(
            "Method holdout identity overlap detected (train_holdout=%d, val_holdout=%d). "
            "This is allowed in method-holdout mode, but should be interpreted accordingly.",
            len(overlap_train_holdout),
            len(overlap_val_holdout),
        )

    logger.info("Data split BY METHOD HOLDOUT (seed=%d):", seed)
    logger.info("  - Holdout methods: %s", sorted(holdout_methods_norm))
    logger.info("  - Train samples (seen methods): %d", len(seen_train))
    logger.info("  - Val samples (seen methods): %d", len(seen_val))
    logger.info("  - Holdout samples (held-out methods): %d", len(holdout_samples))

    return seen_train, seen_val, holdout_samples


def _parse_bucket_and_prefix(bucket_arg: str, prefix_arg: str) -> Tuple[str, str]:
    """
    Parse bucket argument supporting:
      - bucket: "effort-collected-data"
      - bucket/prefix: "effort-collected-data/wma_validation/enhanced_fake"
      - gs://bucket/prefix
    """
    bucket_value = (bucket_arg or "").strip()
    if bucket_value.startswith("gs://"):
        bucket_value = bucket_value.replace("gs://", "", 1)
    if "/" in bucket_value:
        bucket_name, prefix = bucket_value.split("/", 1)
        return bucket_name, prefix
    return bucket_value, (prefix_arg or "").strip()


def _discover_external_training_reals(
    combined_config: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[List["UnifiedUnpairedRealSample"], set]:
    """
    Discover and group external real images for training.

    Reads ``combined_paired.external_training_reals`` config, lists GCS objects,
    groups by identity (regex-extracted from filename), selects the training
    split of identities, and returns ``UnifiedUnpairedRealSample`` objects.

    Returns:
        Tuple of (list of training samples, set of training identity strings).
        The identity set is used downstream to exclude these identities from OOD
        monitoring so we don't evaluate on training data.
    """
    import re
    from pathlib import Path

    ext_cfg_list = combined_config.get("external_training_reals") or []
    if not ext_cfg_list:
        return [], set()

    all_samples: List[UnifiedUnpairedRealSample] = []
    all_training_identities: set = set()

    for ext_cfg in ext_cfg_list:
        bucket = str(ext_cfg.get("bucket", "")).strip()
        if not bucket:
            logger.warning("external_training_reals entry missing 'bucket' — skipping")
            continue

        bucket_name, prefix = _parse_bucket_and_prefix(
            bucket, str(ext_cfg.get("prefix", "")).strip()
        )
        method = str(ext_cfg.get("method", "external_vcd_real"))
        grouping = str(ext_cfg.get("grouping", "per_image"))
        identity_pattern = ext_cfg.get(
            "identity_pattern",
            r"real__VCD__(?P<md5>[a-f0-9]{32})_",
        )
        cache_manifest_path = ext_cfg.get("cache_manifest_path")
        id_train_frac = float(ext_cfg.get("identity_train_fraction") or 0.20)
        id_split_seed = int(ext_cfg.get("identity_split_seed") or 737)
        max_frames_per_id = ext_cfg.get("max_frames_per_identity")
        max_total = ext_cfg.get("max_total_samples")

        # ----- discover frames from GCS -----
        base_path = f"gs://{bucket_name}/{prefix}".rstrip("/")
        frame_paths: List[str] = []
        loaded_from_cache = False
        if cache_manifest_path:
            try:
                from fsspec.core import url_to_fs

                cache_fs, cache_path = url_to_fs(cache_manifest_path)
                if cache_fs.exists(cache_path):
                    with cache_fs.open(cache_path, "r") as f:
                        frame_paths = json.load(f)
                    loaded_from_cache = True
                    logger.info(
                        "External training reals: loaded %d cached frame paths from %s",
                        len(frame_paths),
                        cache_manifest_path,
                    )
            except Exception as exc:
                logger.warning(
                    "External training reals: failed reading cache %s (%s), falling back to GCS listing.",
                    cache_manifest_path,
                    exc,
                )

        if not loaded_from_cache:
            try:
                from fsspec.core import url_to_fs

                fs = url_to_fs(base_path)[0]
                raw_paths = fs.glob(f"{base_path}/**")
            except Exception as exc:
                logger.error("Failed to list GCS for external reals (%s): %s", base_path, exc)
                continue

            allowed_exts = {".png", ".jpg", ".jpeg"}
            frame_paths = [
                f"gs://{p}" for p in raw_paths if Path(p).suffix.lower() in allowed_exts
            ]
            logger.info(
                "External training reals: discovered %d frames from %s",
                len(frame_paths),
                base_path,
            )
            if cache_manifest_path:
                try:
                    cache_fs, cache_path = url_to_fs(cache_manifest_path)
                    cache_parent = os.path.dirname(cache_path)
                    if cache_parent and not cache_fs.exists(cache_parent):
                        cache_fs.makedirs(cache_parent, exist_ok=True)
                    with cache_fs.open(cache_path, "w") as f:
                        json.dump(frame_paths, f)
                    logger.info(
                        "External training reals: wrote cache manifest %s",
                        cache_manifest_path,
                    )
                except Exception as exc:
                    logger.warning(
                        "External training reals: failed writing cache %s: %s",
                        cache_manifest_path,
                        exc,
                    )
        if not frame_paths:
            continue

        # ----- group frames by identity -----
        id_regex = re.compile(identity_pattern)
        frames_by_identity: Dict[str, List[str]] = defaultdict(list)

        for fp in sorted(frame_paths):
            m = id_regex.search(fp)
            if m:
                # Use the first named group as identity
                identity = m.group(1)
            else:
                # Fall back to parent folder name
                identity = Path(fp).parent.name
            frames_by_identity[identity].append(fp)

        all_identities = sorted(frames_by_identity.keys())
        logger.info(
            "External training reals: %d unique identities found",
            len(all_identities),
        )

        # ----- deterministic identity split -----
        rng = random.Random(id_split_seed)
        shuffled = list(all_identities)
        rng.shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * id_train_frac))
        train_identities = set(shuffled[:n_train])
        logger.info(
            "External training reals identity split: %d/%d identities for training (frac=%.2f, seed=%d)",
            len(train_identities),
            len(all_identities),
            id_train_frac,
            id_split_seed,
        )

        # ----- create samples from training identities -----
        source_samples: List[UnifiedUnpairedRealSample] = []
        total_frames_used = 0

        for identity in sorted(train_identities):
            frames = frames_by_identity[identity]
            if max_frames_per_id and len(frames) > max_frames_per_id:
                frames = sorted(frames)[:max_frames_per_id]

            sample = UnifiedUnpairedRealSample(
                identity=f"external_vcd_{identity}",
                source="external",
                method=method,
                gcs_bucket=bucket_name,
                frame_paths=frames,
                sample_id=f"ext_real_{identity}",
            )
            source_samples.append(sample)
            total_frames_used += len(frames)

        # ----- apply total cap -----
        if max_total and total_frames_used > max_total:
            # Trim by removing identities with the most frames first
            source_samples.sort(key=lambda s: len(s.frame_paths))
            trimmed = []
            running = 0
            for s in source_samples:
                if running + len(s.frame_paths) > max_total:
                    # Include partial
                    remaining = max_total - running
                    if remaining > 0:
                        s.frame_paths = s.frame_paths[:remaining]
                        trimmed.append(s)
                        running += remaining
                    break
                trimmed.append(s)
                running += len(s.frame_paths)
            source_samples = trimmed
            total_frames_used = sum(len(s.frame_paths) for s in source_samples)

        all_samples.extend(source_samples)
        all_training_identities.update(train_identities)

        logger.info(
            "External training reals source '%s': %d samples, %d total frames (training identities: %d)",
            method,
            len(source_samples),
            total_frames_used,
            len(train_identities),
        )

    return all_samples, all_training_identities


def _partition_ood_videos_heldout(
    videos: List[Any],
    heldout_fraction: float,
    logger: logging.Logger,
    hash_rule: str = "blake2b_lo10",
) -> Tuple[List[Any], List[Any]]:
    """Deterministic video-id-level partition (A10 in R13 Packet 3 plan).

    Uses blake2b(video_id, digest_size=8). Videos whose 8-byte big-endian hash
    value falls in the lowest `heldout_fraction` of the 2**64 range are moved to
    the held-out slice; the remainder is monitored during training.

    The partition is reproducible across runs that share the same video_ids and
    the same hash_rule, so retroactive application (A2b) gives byte-for-byte
    matched splits.

    Args:
        videos: flat list of OOD video objects, each with a ``video_id``
            attribute (fall back to ``sample_id`` or str(id(v))).
        heldout_fraction: target fraction in [0, 1] for the held-out slice.
            0 → no partition (all monitored), 1 → everything held out.
        logger: python logger.
        hash_rule: tag recorded in logs / data_stats for audit / reproduction.

    Returns:
        ``(monitored_videos, heldout_videos)``
    """
    if heldout_fraction <= 0.0 or not videos:
        return list(videos), []
    if heldout_fraction >= 1.0:
        return [], list(videos)

    cutoff = int(round(heldout_fraction * float(2 ** 64)))
    if cutoff <= 0:
        return list(videos), []
    if cutoff >= 2 ** 64:
        return [], list(videos)

    monitored: List[Any] = []
    heldout: List[Any] = []
    per_method_heldout: Counter = Counter()
    per_method_total: Counter = Counter()

    for v in videos:
        vid = getattr(v, "video_id", None) or getattr(v, "sample_id", None) or str(id(v))
        key = str(vid).encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        val = int.from_bytes(digest, byteorder="big", signed=False)
        method = getattr(v, "method", "unknown")
        per_method_total[method] += 1
        if val < cutoff:
            heldout.append(v)
            per_method_heldout[method] += 1
        else:
            monitored.append(v)

    logger.info(
        "OOD held-out partition (rule=%s, fraction=%.3f): monitored=%d heldout=%d",
        hash_rule,
        heldout_fraction,
        len(monitored),
        len(heldout),
    )
    for method, total in sorted(per_method_total.items()):
        logger.info(
            "  method=%s total=%d heldout=%d monitored=%d",
            method,
            total,
            per_method_heldout.get(method, 0),
            total - per_method_heldout.get(method, 0),
        )

    return monitored, heldout


def _build_external_ood_videos(
    combined_config: Dict[str, Any],
    frames_per_video: int,
    logger: logging.Logger,
    exclude_identities: Optional[set] = None,
) -> List[Any]:
    """
    Build optional OOD videos from external real/fake sources.

    This reuses validation source loaders so WMA/external sets can be monitored
    during training without affecting gradients.
    """
    ood_cfg = (combined_config.get("ood_monitoring") or {})
    if not ood_cfg.get("enabled", False):
        return []

    from data.validation_sources import (
        load_external_fake_videos,
        load_external_real_videos,
    )

    def _collect_real_sources() -> List[Dict[str, Any]]:
        explicit = ood_cfg.get("external_real_sources") or []
        if explicit:
            return list(explicit)
        legacy_bucket = ood_cfg.get("external_real_bucket")
        if not legacy_bucket:
            return []
        return [{
            "bucket": legacy_bucket,
            "prefix": ood_cfg.get("external_real_prefix", "real/external_youtube_avspeech"),
            "method": ood_cfg.get("external_real_method", "external_youtube_avspeech"),
            "cache_manifest_path": ood_cfg.get("external_real_cache"),
            "max_videos": ood_cfg.get("max_external_real"),
            "seed": ood_cfg.get("external_real_seed", 737),
            "grouping": ood_cfg.get("external_real_grouping", "by_folder"),
            "deterministic": ood_cfg.get("external_real_deterministic", False),
        }]

    def _collect_fake_sources() -> List[Dict[str, Any]]:
        explicit = ood_cfg.get("external_fake_sources") or []
        if explicit:
            return list(explicit)
        legacy_bucket = ood_cfg.get("external_fake_bucket")
        if not legacy_bucket:
            return []
        return [{
            "bucket": legacy_bucket,
            "prefix": ood_cfg.get("external_fake_prefix", "wma_validation/enhanced_fake"),
            "method": ood_cfg.get("external_fake_method", "wma_failure_fake"),
            "cache_manifest_path": ood_cfg.get("external_fake_cache"),
            "max_videos": ood_cfg.get("max_external_fake"),
            "seed": ood_cfg.get("external_fake_seed", 737),
            "grouping": ood_cfg.get("external_fake_grouping", "by_folder"),
            "deterministic": ood_cfg.get("external_fake_deterministic", False),
        }]

    def _effective_deterministic_frames(grouping: str, deterministic: bool) -> Optional[int]:
        if deterministic:
            return frames_per_video
        if grouping == "per_image" and frames_per_video > 1:
            logger.info(
                "OOD source auto-deterministic enabled: grouping=per_image, frames_per_video=%d",
                frames_per_video,
            )
            return frames_per_video
        return None

    videos: List[Any] = []

    for source_cfg in _collect_real_sources():
        bucket_arg = str(source_cfg.get("bucket", "")).strip()
        if not bucket_arg:
            raise ValueError("OOD real source is missing required `bucket`.")
        bucket_name, prefix = _parse_bucket_and_prefix(
            bucket_arg,
            str(source_cfg.get("prefix", "real/external_youtube_avspeech")),
        )
        grouping = str(source_cfg.get("grouping", "by_folder"))
        deterministic_frames = _effective_deterministic_frames(
            grouping=grouping,
            deterministic=bool(source_cfg.get("deterministic", False)),
        )
        source_videos = load_external_real_videos(
            bucket_name=bucket_name,
            prefix=prefix,
            method_name=str(source_cfg.get("method", "external_youtube_avspeech")),
            cache_manifest_path=source_cfg.get("cache_manifest_path"),
            max_videos=source_cfg.get("max_videos"),
            seed=int(source_cfg.get("seed") or 737),
            grouping=grouping,
            deterministic_frame_count=deterministic_frames,
            path_contains=source_cfg.get("path_contains"),
            path_exclude_contains=source_cfg.get("path_exclude_contains"),
            video_id_depth=int(source_cfg.get("video_id_depth") if source_cfg.get("video_id_depth") is not None else -2),
        )
        # Filter out identities already used for training (if applicable)
        if exclude_identities:
            import re
            before_count = len(source_videos)
            id_regex = re.compile(r"real__VCD__(?P<md5>[a-f0-9]{32})_")
            filtered = []
            for v in source_videos:
                # Try to extract identity from frame paths
                identity = None
                for fp in (v.frame_paths or []):
                    m = id_regex.search(fp)
                    if m:
                        identity = m.group(1)
                        break
                if identity is None:
                    # Fall back to video_id
                    identity = getattr(v, "video_id", "")
                if identity not in exclude_identities:
                    filtered.append(v)
            source_videos = filtered
            if before_count != len(source_videos):
                logger.info(
                    "OOD identity exclusion: %d -> %d videos (excluded %d training identities)",
                    before_count,
                    len(source_videos),
                    before_count - len(source_videos),
                )
        videos.extend(source_videos)

    for source_cfg in _collect_fake_sources():
        bucket_arg = str(source_cfg.get("bucket", "")).strip()
        if not bucket_arg:
            raise ValueError("OOD fake source is missing required `bucket`.")
        bucket_name, prefix = _parse_bucket_and_prefix(
            bucket_arg,
            str(source_cfg.get("prefix", "wma_validation/enhanced_fake")),
        )
        grouping = str(source_cfg.get("grouping", "by_folder"))
        deterministic_frames = _effective_deterministic_frames(
            grouping=grouping,
            deterministic=bool(source_cfg.get("deterministic", False)),
        )
        source_videos = load_external_fake_videos(
            bucket_name=bucket_name,
            prefix=prefix,
            method_name=str(source_cfg.get("method", "wma_failure_fake")),
            cache_manifest_path=source_cfg.get("cache_manifest_path"),
            max_videos=source_cfg.get("max_videos"),
            seed=int(source_cfg.get("seed") or 737),
            grouping=grouping,
            deterministic_frame_count=deterministic_frames,
            path_contains=source_cfg.get("path_contains"),
            path_exclude_contains=source_cfg.get("path_exclude_contains"),
            video_id_depth=int(source_cfg.get("video_id_depth") if source_cfg.get("video_id_depth") is not None else -2),
        )
        videos.extend(source_videos)

    # A3 / A3b: deterministic eval-time stress OOD lanes. Each list entry has
    # the same schema as external_real_sources / external_fake_sources plus an
    # ``eval_augmentation`` field naming a preset in
    # ``data.augmentations.pipelines.EVAL_STRESS_PRESETS``. The preset is
    # stamped onto ``VideoInfo.eval_aug_preset`` so ``load_and_process_video``
    # applies it at load time. A ``label`` field ('real' | 'fake') selects the
    # underlying loader; default is 'real'.
    try:
        from data.augmentations.pipelines import EVAL_STRESS_PRESETS
    except Exception:
        EVAL_STRESS_PRESETS = set()

    stress_block_names = (
        ("lighting_stress_sources", "ood_lighting_stress"),
        ("spatial_stress_sources", "ood_spatial_stress"),
    )
    for block_key, default_prefix in stress_block_names:
        stress_sources = ood_cfg.get(block_key) or []
        for source_cfg in stress_sources:
            eval_aug = str(source_cfg.get("eval_augmentation") or "").strip()
            if not eval_aug:
                raise ValueError(
                    f"OOD stress source in `{block_key}` is missing required "
                    "`eval_augmentation` (preset name)."
                )
            if EVAL_STRESS_PRESETS and eval_aug not in EVAL_STRESS_PRESETS:
                logger.warning(
                    "OOD stress source in `%s` uses unknown preset %r "
                    "(known presets: %s). The load will fail at eval time.",
                    block_key, eval_aug, sorted(EVAL_STRESS_PRESETS),
                )
            label_str = str(source_cfg.get("label", "real")).strip().lower()
            if label_str not in ("real", "fake"):
                raise ValueError(
                    f"OOD stress source in `{block_key}` has invalid "
                    f"`label`={label_str!r}; must be 'real' or 'fake'."
                )

            bucket_arg = str(source_cfg.get("bucket", "")).strip()
            if not bucket_arg:
                raise ValueError(
                    f"OOD stress source in `{block_key}` is missing required `bucket`."
                )
            default_prefix_str = (
                "real/external_youtube_avspeech" if label_str == "real"
                else "wma_validation/enhanced_fake"
            )
            bucket_name, prefix = _parse_bucket_and_prefix(
                bucket_arg,
                str(source_cfg.get("prefix", default_prefix_str)),
            )
            grouping = str(source_cfg.get("grouping", "by_folder"))
            deterministic_frames = _effective_deterministic_frames(
                grouping=grouping,
                deterministic=bool(source_cfg.get("deterministic", False)),
            )
            method_name = str(
                source_cfg.get("method")
                or f"{default_prefix}_{eval_aug}_{label_str}"
            )
            loader_fn = load_external_real_videos if label_str == "real" else load_external_fake_videos
            stress_videos = loader_fn(
                bucket_name=bucket_name,
                prefix=prefix,
                method_name=method_name,
                cache_manifest_path=source_cfg.get("cache_manifest_path"),
                max_videos=source_cfg.get("max_videos"),
                seed=int(source_cfg.get("seed") or 737),
                grouping=grouping,
                deterministic_frame_count=deterministic_frames,
                path_contains=source_cfg.get("path_contains"),
                path_exclude_contains=source_cfg.get("path_exclude_contains"),
                video_id_depth=int(
                    source_cfg.get("video_id_depth")
                    if source_cfg.get("video_id_depth") is not None
                    else -2
                ),
            )
            # Stamp the eval preset onto every loaded VideoInfo.
            for v in stress_videos:
                try:
                    v.eval_aug_preset = eval_aug
                except AttributeError:
                    pass

            # Reuse the training-identity exclusion for real lanes (mirrors the
            # non-stress real path above).
            if label_str == "real" and exclude_identities:
                import re
                id_regex = re.compile(r"real__VCD__(?P<md5>[a-f0-9]{32})_")
                filtered = []
                for v in stress_videos:
                    identity = None
                    for fp in (v.frame_paths or []):
                        m = id_regex.search(fp)
                        if m:
                            identity = m.group(1)
                            break
                    if identity is None:
                        identity = getattr(v, "video_id", "")
                    if identity not in exclude_identities:
                        filtered.append(v)
                stress_videos = filtered
            logger.info(
                "Stress OOD source (%s, preset=%s, label=%s, method=%s): %d videos",
                block_key, eval_aug, label_str, method_name, len(stress_videos),
            )
            videos.extend(stress_videos)

    method_counts = Counter(v.method for v in videos)
    logger.info(
        "External OOD monitoring set assembled: videos=%d methods=%d method_counts=%s",
        len(videos),
        len(method_counts),
        dict(sorted(method_counts.items())),
    )

    expected_counts = ood_cfg.get("expected_counts", {}) or {}
    strict_expected = bool(ood_cfg.get("strict_expected_counts", False))
    if expected_counts:
        normalized_actual = Counter(normalize_method_name(k) for k in method_counts.elements())
        for expected_method, expected_count in expected_counts.items():
            method_norm = normalize_method_name(str(expected_method))
            expected_count_int = int(expected_count)
            actual_count = int(normalized_actual.get(method_norm, 0))
            if strict_expected and actual_count != expected_count_int:
                raise ValueError(
                    f"OOD expected count mismatch for `{expected_method}`: "
                    f"expected exactly {expected_count_int}, got {actual_count}"
                )
            if not strict_expected and actual_count < expected_count_int:
                logger.warning(
                    "OOD expected minimum not met for `%s`: expected >= %d, got %d",
                    expected_method,
                    expected_count_int,
                    actual_count,
                )

    return videos


# =============================================================================
# Helpers
# =============================================================================

def _transform_accepts_meta(transform: Optional[Callable]) -> bool:
    """Return True when a transform supports (image, landmarks, meta)."""
    if transform is None:
        return False
    try:
        signature = inspect.signature(transform)
    except (TypeError, ValueError):
        return False

    params = list(signature.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return True

    positional = [
        p for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 3


def _strategy_name_from_method(method: str) -> Optional[str]:
    method_norm = normalize_method_name(method)
    if method_norm.startswith("deeplive_"):
        return method_norm[len("deeplive_") :]
    return None


def _sample_family_for_sampling(
    sample,
    enhanced_strategy_names: Sequence[str],
) -> str:
    """
    Map a sample to the family used for weighted identity sampling.

    For paired samples, we prioritize fake-family routing (label=1) because each
    paired sample yields both real/fake frames.
    For unpaired real samples, we route directly to the real family.
    """
    override = normalize_method_name(getattr(sample, "sampling_family_key", None))
    if override:
        return override

    # Unpaired real samples have no fake counterpart — route directly to real family
    if getattr(sample, 'is_unpaired_real', False):
        return infer_family_key(
            0,
            method=sample.method,
            source=sample.source,
            enhanced_strategy_names=enhanced_strategy_names,
        )

    fake_family = infer_family_key(
        1,
        method=sample.method,
        source=sample.source,
        enhanced_strategy_names=enhanced_strategy_names,
    )
    if fake_family not in {"unknown", "unknown_fake"}:
        return fake_family

    return infer_family_key(
        0,
        method=sample.method,
        source=sample.source,
        enhanced_strategy_names=enhanced_strategy_names,
    )


def _sample_possible_methods(sample: UnifiedPairedSample) -> List[str]:
    methods: List[str] = []
    for candidate in [sample.method, *(getattr(sample, "method_variants", ()) or ())]:
        if candidate and candidate not in methods:
            methods.append(candidate)
    return methods


def _build_overview_method_lists(
    samples: List[UnifiedPairedSample],
) -> Dict[str, List[str]]:
    """
    Build real/fake method lists for run-overview reporting.

    Paired samples expose only the fake-side method name, so their real side is
    represented by the synthetic `paired_real` label. Unpaired real samples
    keep their explicit method names.
    """
    real_methods = set()
    fake_methods = set()
    has_paired_samples = False

    for sample in samples:
        if getattr(sample, "is_unpaired_real", False):
            method_norm = normalize_method_name(sample.method)
            if method_norm:
                real_methods.add(method_norm)
            continue

        has_paired_samples = True
        method_norm = normalize_method_name(sample.method)
        if method_norm:
            fake_methods.add(method_norm)

    if has_paired_samples:
        real_methods.add("paired_real")

    return {
        "real_methods": sorted(real_methods),
        "fake_methods": sorted(fake_methods),
    }


def _identity_hash_fraction(identity: str, seed: int) -> float:
    """
    Map `(identity, seed)` to a deterministic fraction in [0, 1).

    Unlike the legacy global shuffle split, this keeps existing identities in
    the same partition even when later experiments add new identities.
    """
    digest = hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).digest()
    numerator = int.from_bytes(digest[:8], "big")
    return numerator / float(1 << 64)


def _count_deeplive_strategies(
    samples: List[Any],
) -> Dict[str, Dict[str, int]]:
    """Return raw/effective DeepLive strategy counts from discovered samples."""
    raw_counts = Counter()
    effective_counts = Counter()

    for sample in samples:
        raw_strategy = normalize_method_name(getattr(sample, "raw_strategy", getattr(sample, "strategy", "unknown")))
        effective_strategy = normalize_method_name(
            getattr(sample, "effective_strategy", getattr(sample, "strategy", "unknown"))
        )
        raw_counts[raw_strategy] += 1
        effective_counts[effective_strategy] += 1

    return {
        "raw": dict(sorted(raw_counts.items())),
        "effective": dict(sorted(effective_counts.items())),
    }


def _resolve_enhanced_expectation_mode(
    include_strategies: Sequence[str],
    exclude_strategies: Sequence[str],
    enhanced_strategy_names: Sequence[str],
    explicit_mode: Optional[str],
) -> Optional[str]:
    mode = normalize_method_name(explicit_mode) if explicit_mode else ""
    if mode in {"withenhanced", "noenhanced"}:
        return mode

    include_set = {normalize_method_name(s) for s in include_strategies if s}
    exclude_set = {normalize_method_name(s) for s in exclude_strategies if s}
    enhanced_set = {normalize_method_name(s) for s in enhanced_strategy_names if s}

    if include_set:
        if include_set & enhanced_set:
            return "withenhanced"
        return "noenhanced"

    if enhanced_set and enhanced_set.issubset(exclude_set):
        return "noenhanced"
    return None


def _run_deeplive_preflight_checks(
    effective_counts: Dict[str, int],
    expectation_mode: Optional[str],
    min_counts: Optional[Dict[str, int]],
) -> Dict[str, Any]:
    """
    Validate enhanced strategy discovery against config intent.

    Modes:
      - withenhanced: enforce minimum enhanced strategy counts
      - noenhanced: enforce strict zero for enhanced strategies
      - None: skip strict assertions
    """
    min_counts_norm = {
        normalize_method_name(k): int(v)
        for k, v in (min_counts or {}).items()
    }
    if not min_counts_norm:
        min_counts_norm = {
            "edge_cases_enhanced": 400,
            "minimal_processing_enhanced": 390,
        }

    errors: List[str] = []
    mode = normalize_method_name(expectation_mode) if expectation_mode else None

    if mode == "withenhanced":
        for strategy, min_required in min_counts_norm.items():
            actual = int(effective_counts.get(strategy, 0))
            if actual < min_required:
                errors.append(
                    f"{strategy}: expected >= {min_required}, discovered {actual}"
                )

    elif mode == "noenhanced":
        for strategy in min_counts_norm.keys():
            actual = int(effective_counts.get(strategy, 0))
            if actual != 0:
                errors.append(
                    f"{strategy}: expected 0, discovered {actual}"
                )

    return {
        "mode": mode,
        "min_counts": min_counts_norm,
        "effective_counts": dict(sorted(effective_counts.items())),
        "passed": len(errors) == 0,
        "errors": errors,
    }


def _count_strategy_and_family(
    samples: List[UnifiedPairedSample],
    enhanced_strategy_names: Tuple[str, ...],
) -> Dict[str, Dict[str, int]]:
    """Build strategy-level and family-level counts for logging/reporting."""
    strategy_counts = Counter()
    family_counts = Counter()

    for sample in samples:
        strategy = _strategy_name_from_method(sample.method)
        if strategy:
            strategy_counts[strategy] += 1

        # Unpaired external reals contribute only a real example. Paired samples
        # contribute one real and one fake example.
        family_counts[infer_family_key(0, method=sample.method, source=sample.source,
                                       enhanced_strategy_names=enhanced_strategy_names)] += 1
        if getattr(sample, "is_unpaired_real", False):
            continue
        fake_family = normalize_method_name(getattr(sample, "sampling_family_key", None)) or infer_family_key(
            1,
            method=sample.method,
            source=sample.source,
            enhanced_strategy_names=enhanced_strategy_names,
        )
        family_counts[fake_family] += 1

    return {
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
    }


# =============================================================================
# Combined Iterable Dataset
# =============================================================================

@dataclass
class CombinedBatchingConfig:
    """Configuration for combined paired batching."""
    batch_size: int = 32
    num_workers: int = 4
    prefetch_factor: int = 2
    visomaster_parallel_download_workers: int = 4
    teams_parallel_download_workers: int = 4
    identity_balanced_sampling: bool = True
    identity_sampling_strategy: str = "identity_resample_uniform"
    identity_family_weights: Dict[str, float] = field(default_factory=dict)
    enhanced_strategy_names: Tuple[str, ...] = field(default_factory=tuple)
    
    # Frame sampling (shared across sources)
    frames_per_sample: int = 8
    
    # DF40-specific
    df40_sparse_indices: List[int] = field(default_factory=lambda: [0, 4, 8, 12, 16, 20, 24, 28])
    
    # DeepLive-specific  
    deeplive_sparse_indices: List[int] = field(default_factory=lambda: [0, 2, 4, 6, 8, 10, 12, 14])
    
    # VisoMaster-specific
    visomaster_sparse_indices: List[int] = field(default_factory=lambda: [0, 2, 4, 6, 8, 10, 12, 14])

    # VisoMaster Enhanced-specific (same defaults as VisoMaster)
    visomaster_enhanced_sparse_indices: List[int] = field(default_factory=lambda: [0, 2, 4, 6, 8, 10, 12, 14])

    # VisoMaster Teams-Enhanced merged source
    visomaster_teams_enhanced_sparse_indices: List[int] = field(default_factory=lambda: [0, 2, 4, 6, 8, 10, 12, 14])
    visomaster_teams_enhanced_p_original: float = 0.5

    # VisoMaster Resolution-Variant-specific (same defaults)
    visomaster_res_variant_sparse_indices: List[int] = field(default_factory=lambda: [0, 2, 4, 6, 8, 10, 12, 14])

    # Proper-data inventory-backed lanes
    proper_data_sparse_indices: List[int] = field(default_factory=lambda: [0, 2, 4, 6, 8, 10, 12, 14])
    proper_data_parallel_download_workers: int = DEFAULT_PARALLEL_GCS_DOWNLOAD_WORKERS

    # Teams passthrough-specific
    teams_sparse_indices: List[int] = field(default_factory=lambda: [0, 2, 4, 6, 8, 10, 12, 14])
    # Optional per-frame keep-list for Teams REAL frames (T3 SLOT1/2/3 IQ-shortcut packets,
    # 2026-05-09). When provided as a frozenset of full gs:// URIs, only real frames whose
    # URI is present in the set are yielded by _iterate_teams_sample. Fake frames are not
    # affected. When None (the default), Teams ingestion is identical to legacy behavior.
    teams_real_frame_keep_list: Optional[frozenset] = None
    # Optional per-method keep-list for Teams REAL frames (T3 SLOT3, 2026-05-09).
    # Maps `method` (e.g. "deeplive_teams_edge_cases") to a frozenset of allowed
    # gs:// frame URIs for THAT method's pairing. When set, the real frame must
    # appear in the corresponding method's set; otherwise it is skipped. None
    # = legacy behavior.
    teams_real_frame_keep_list_per_method: Optional[Dict[str, frozenset]] = None


class CombinedPairedIterableDataset(IterableDataset):
    """
    Iterable dataset that yields paired real/fake frames from combined sources.
    
    Supports:
    - Identity-balanced sampling (one method per identity per epoch)
    - Both DF40 (no landmarks) and DeepLive (with landmarks) samples
    - Unified transform interface
    """
    
    def __init__(
        self,
        samples: List[UnifiedPairedSample],
        df40_dataset: Any,  # DF40PairedDataset
        deeplive_dataset: Any,  # DeepLiveDataset
        config: CombinedBatchingConfig,
        transform: Optional[Callable] = None,
        shuffle: bool = True,
        seed: int = 42,
        visomaster_anchor_indices: Optional[List[int]] = None,
        method_mapping: Optional[Dict[str, int]] = None,
        face_area_parquet_path: Optional[str] = None,
    ):
        self.samples = samples
        self.df40_dataset = df40_dataset
        self.deeplive_dataset = deeplive_dataset
        self.config = config
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self._transform_accepts_meta = _transform_accepts_meta(transform)
        self.visomaster_anchor_indices = visomaster_anchor_indices or config.visomaster_sparse_indices
        self.method_mapping = method_mapping or {}
        self._enhanced_strategy_names = tuple(
            config.enhanced_strategy_names or DEFAULT_ENHANCED_STRATEGIES
        )

        # Face-area-fraction lookup for the correlation-penalty loss
        # (added 2026-05-05). The parquet is built by a CPU pre-tag job
        # over the deeplive bucket; here we aggregate per-sample mean.
        # If absent, _face_area_for returns NaN — the loss-side code skips
        # NaN-only batches gracefully.
        self._face_area_per_sample: Dict[str, float] = {}
        if face_area_parquet_path:
            try:
                import pandas as pd
                df = pd.read_parquet(face_area_parquet_path)
                if 'sample_id' not in df.columns:
                    # Derive sample_id from frame_path: strip the
                    # `.../samples/<sample_id>/frames/...` middle component.
                    import re
                    pat = re.compile(r"/samples/([^/]+)/frames/")
                    def _extract(path: str) -> Optional[str]:
                        m = pat.search(path)
                        return m.group(1) if m else None
                    df['sample_id'] = df['frame_path'].map(_extract)
                # Per-sample mean (skip NaNs); samples with all-NaN frames stay missing.
                grouped = df.dropna(subset=['face_area_fraction']).groupby('sample_id')['face_area_fraction'].mean()
                self._face_area_per_sample = grouped.to_dict()
                logger.info(
                    f"Loaded face_area_fraction lookup: {len(self._face_area_per_sample)} samples "
                    f"from {face_area_parquet_path}"
                )
            except Exception as exc:
                logger.warning(
                    f"Failed to load face_area_fraction parquet at "
                    f"{face_area_parquet_path}: {exc}. Face-area axis will be NaN at training."
                )
        
        # Group samples by identity for identity-balanced sampling
        self._samples_by_identity: Dict[str, List[UnifiedPairedSample]] = defaultdict(list)
        self._sample_family_key: Dict[int, str] = {}
        for sample in samples:
            self._samples_by_identity[sample.identity].append(sample)
            self._sample_family_key[id(sample)] = _sample_family_for_sampling(
                sample,
                enhanced_strategy_names=self._enhanced_strategy_names,
            )
        
        self._identities = list(self._samples_by_identity.keys())
        
        # Log statistics
        samples_per_identity = [len(s) for s in self._samples_by_identity.values()]
        source_counts = defaultdict(int)
        for s in samples:
            source_counts[s.source] += 1
        
        logger.info(f"CombinedPairedIterableDataset initialized:")
        logger.info(f"  - Total samples: {len(samples)} (by source: {dict(source_counts)})")
        logger.info(f"  - Unique identities: {len(self._identities)}")
        if samples_per_identity:
            logger.info(f"  - Samples per identity: min={min(samples_per_identity)}, max={max(samples_per_identity)}")
        else:
            logger.info("  - Samples per identity: dataset is empty")
        logger.info(f"  - Identity-balanced sampling: {config.identity_balanced_sampling}")
        logger.info(f"  - Identity sampling strategy: {config.identity_sampling_strategy}")
        if config.identity_family_weights:
            logger.info(f"  - Identity family weights: {dict(config.identity_family_weights)}")
        if self.transform:
            logger.info(f"  - Transform meta routing: {'enabled' if self._transform_accepts_meta else 'disabled'}")
    
    def set_epoch(self, epoch: int):
        """Set epoch for varying random method selection."""
        self._epoch = epoch
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples, yielding frame dicts."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            rng = random.Random(self.seed + self._epoch * 1000 + worker_id)
        else:
            worker_id = 0
            num_workers = 1
            rng = random.Random(self.seed + self._epoch * 1000)
        
        # Get samples to iterate
        if self.config.identity_balanced_sampling:
            samples_to_iterate = self._get_identity_balanced_samples(rng, worker_id, num_workers)
        else:
            samples_to_iterate = self.samples[worker_id::num_workers]
            if self.shuffle:
                samples_to_iterate = samples_to_iterate.copy()
                rng.shuffle(samples_to_iterate)
        
        # Log method distribution (worker 0 only)
        if worker_id == 0:
            method_counts = defaultdict(int)
            source_counts = defaultdict(int)
            for s in samples_to_iterate:
                method_counts[s.method] += 1
                source_counts[s.source] += 1
            logger.info(f"Epoch {self._epoch} source distribution: {dict(source_counts)}")
        
        # Iterate through samples
        for unified_sample in samples_to_iterate:
            try:
                if getattr(unified_sample, 'is_unpaired_real', False):
                    yield from self._iterate_unpaired_real_sample(unified_sample, rng)
                elif unified_sample.source == 'df40':
                    yield from self._iterate_df40_sample(unified_sample, rng)
                elif unified_sample.source == 'visomaster_enhanced':
                    yield from self._iterate_visomaster_enhanced_sample(unified_sample, rng)
                elif unified_sample.source == 'visomaster_teams_enhanced':
                    yield from self._iterate_visomaster_teams_enhanced_sample(unified_sample, rng)
                elif unified_sample.source == 'visomaster_res_variant':
                    yield from self._iterate_visomaster_res_variant_sample(unified_sample, rng)
                elif unified_sample.source.startswith('proper_visomaster_'):
                    yield from self._iterate_proper_data_sample(unified_sample, rng)
                elif unified_sample.source in {'visomaster', 'visomaster_hints'}:
                    yield from self._iterate_visomaster_sample(unified_sample, rng)
                elif unified_sample.source in {'deeplive_teams', 'visomaster_hints_teams'}:
                    yield from self._iterate_teams_sample(unified_sample, rng)
                else:  # deeplive
                    yield from self._iterate_deeplive_sample(unified_sample, rng)
            except Exception as e:
                logger.warning(f"Failed to load sample {unified_sample.sample_id}: {e}")
                continue

    def _get_visomaster_gcs_client(self):
        if not hasattr(self, '_visomaster_gcs_client'):
            self._visomaster_gcs_client = storage.Client()
        return self._visomaster_gcs_client

    def _get_teams_gcs_client(self):
        if not hasattr(self, '_teams_gcs_client'):
            self._teams_gcs_client = storage.Client()
        return self._teams_gcs_client

    def _get_proper_data_gcs_client(self):
        if not hasattr(self, '_proper_data_gcs_client'):
            self._proper_data_gcs_client = storage.Client()
        return self._proper_data_gcs_client

    def _get_visomaster_download_executor(self):
        max_workers = max(
            1,
            int(getattr(self.config, "visomaster_parallel_download_workers", 1) or 1),
        )
        if max_workers <= 1:
            return None

        executor = getattr(self, "_visomaster_download_executor", None)
        executor_workers = getattr(self, "_visomaster_download_executor_workers", None)
        if executor is None or executor_workers != max_workers:
            from concurrent.futures import ThreadPoolExecutor

            self._visomaster_download_executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="visomaster-gcs",
            )
            self._visomaster_download_executor_workers = max_workers
        return self._visomaster_download_executor

    def _get_teams_download_executor(self):
        max_workers = max(
            1,
            int(getattr(self.config, "teams_parallel_download_workers", 1) or 1),
        )
        if max_workers <= 1:
            return None

        executor = getattr(self, "_teams_download_executor", None)
        executor_workers = getattr(self, "_teams_download_executor_workers", None)
        if executor is None or executor_workers != max_workers:
            self._teams_download_executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="teams-gcs",
            )
            self._teams_download_executor_workers = max_workers
        return self._teams_download_executor

    def _get_proper_data_download_executor(self):
        max_workers = max(
            1,
            int(getattr(self.config, "proper_data_parallel_download_workers", 1) or 1),
        )
        if max_workers <= 1:
            return None

        executor = getattr(self, "_proper_data_download_executor", None)
        executor_workers = getattr(self, "_proper_data_download_executor_workers", None)
        if executor is None or executor_workers != max_workers:
            self._proper_data_download_executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="proper-data-gcs",
            )
            self._proper_data_download_executor_workers = max_workers
        return self._proper_data_download_executor
    
    def _get_identity_balanced_samples(
        self,
        rng: random.Random,
        worker_id: int,
        num_workers: int
    ) -> List[UnifiedPairedSample]:
        """Get one sample per identity with randomly selected method."""
        selected_samples = []
        
        for identity in self._identities:
            identity_samples = self._samples_by_identity[identity]
            if self.config.identity_sampling_strategy == "identity_resample_weighted":
                weights = []
                for sample in identity_samples:
                    family_key = self._sample_family_key.get(id(sample), "unknown")
                    weight = float(self.config.identity_family_weights.get(family_key) or 1.0)
                    # Keep behavior robust against accidental negative/zero configs.
                    weights.append(max(weight, 0.0))
                if sum(weights) > 0:
                    selected = rng.choices(identity_samples, weights=weights, k=1)[0]
                else:
                    selected = rng.choice(identity_samples)
            else:
                selected = rng.choice(identity_samples)
            selected_samples.append(selected)
        
        if self.shuffle:
            rng.shuffle(selected_samples)

        return selected_samples[worker_id::num_workers]

    def _apply_transform(
        self,
        image: Any,
        landmarks: Any,
        meta: Dict[str, Any],
    ) -> Any:
        """Apply transform with backward compatibility for 2-arg callables."""
        if not self.transform:
            return image
        if self._transform_accepts_meta:
            return self.transform(image, landmarks, meta)
        return self.transform(image, landmarks)
    
    def _iterate_df40_sample(
        self,
        unified_sample: UnifiedPairedSample,
        rng: random.Random
    ) -> Iterator[Dict[str, Any]]:
        """Load and yield frames from a DF40 sample."""
        sample = unified_sample.original_sample
        frame_indices = self.config.df40_sparse_indices
        
        # Load frames
        real_frames, fake_frames = self.df40_dataset.load_sample_frames(
            sample,
            frame_indices=frame_indices,
            as_array=True
        )
        
        # Yield paired frames
        for i, frame_idx in enumerate(frame_indices):
            if i >= len(real_frames) or i >= len(fake_frames):
                continue
            
            # Real frame (no landmarks for DF40)
            real_img = real_frames[i]
            if self.transform:
                real_img = self._apply_transform(
                    real_img,
                    None,
                    {'label': 0, 'source': 'df40', 'method': unified_sample.method},
                )
            
            yield {
                'image': real_img,
                'label': 0,
                'identity': unified_sample.identity,
                'source': 'df40',
                'method': unified_sample.method,
                'method_id': self.method_mapping.get(unified_sample.method, -1),
                'sample_id': unified_sample.sample_id,
                'frame_idx': frame_idx,
                'quality_domain': _domain_for_sample(unified_sample.method, 'df40', 0),
            }

            # Fake frame (no landmarks for DF40)
            fake_img = fake_frames[i]
            if self.transform:
                fake_img = self._apply_transform(
                    fake_img,
                    None,
                    {'label': 1, 'source': 'df40', 'method': unified_sample.method},
                )

            yield {
                'image': fake_img,
                'label': 1,
                'identity': unified_sample.identity,
                'source': 'df40',
                'method': unified_sample.method,
                'method_id': self.method_mapping.get(unified_sample.method, -1),
                'sample_id': unified_sample.sample_id,
                'frame_idx': frame_idx,
                'quality_domain': _domain_for_sample(unified_sample.method, 'df40', 1),
            }
    
    def _iterate_deeplive_sample(
        self,
        unified_sample: UnifiedPairedSample,
        rng: random.Random
    ) -> Iterator[Dict[str, Any]]:
        """Load and yield frames from a DeepLive sample."""
        sample = unified_sample.original_sample
        frame_indices = self.config.deeplive_sparse_indices
        
        # Load frames
        real_frames, fake_frames = self.deeplive_dataset.load_sample_frames(
            sample,
            frame_indices=frame_indices,
            as_array=True
        )
        
        # Load landmarks if available and dataset supports it
        real_landmarks = None
        fake_landmarks = None
        if unified_sample.has_landmarks and self.deeplive_dataset.use_landmarks:
            try:
                real_landmarks, fake_landmarks = self.deeplive_dataset.load_landmarks(sample)
                if real_landmarks is None and fake_landmarks is None:
                    # Avoid repeated landmark fetch attempts for samples known to have no usable files.
                    unified_sample.has_landmarks = False
            except Exception as e:
                logger.debug(f"Failed to load landmarks for {sample.sample_id}: {e}")
        
        # Yield paired frames
        for i, frame_idx in enumerate(frame_indices):
            if i >= len(real_frames) or i >= len(fake_frames):
                continue
            
            # Get landmarks for this frame if available
            real_lm = real_landmarks[i] if real_landmarks and i < len(real_landmarks) else None
            fake_lm = fake_landmarks[i] if fake_landmarks and i < len(fake_landmarks) else None
            
            # Real frame (with landmarks if available)
            real_img = real_frames[i]
            if self.transform:
                real_img = self._apply_transform(
                    real_img,
                    real_lm,
                    {'label': 0, 'source': 'deeplive', 'method': unified_sample.method},
                )
            
            face_area_value = float(
                self._face_area_per_sample.get(unified_sample.sample_id, float('nan'))
            )
            yield {
                'image': real_img,
                'label': 0,
                'identity': unified_sample.identity,
                'source': 'deeplive',
                'method': unified_sample.method,
                'method_id': self.method_mapping.get(unified_sample.method, -1),
                'sample_id': unified_sample.sample_id,
                'frame_idx': frame_idx,
                'quality_domain': _domain_for_sample(unified_sample.method, 'deeplive', 0),
                'face_area_fraction': face_area_value,
            }

            # Fake frame (with landmarks if available)
            fake_img = fake_frames[i]
            if self.transform:
                fake_img = self._apply_transform(
                    fake_img,
                    fake_lm,
                    {'label': 1, 'source': 'deeplive', 'method': unified_sample.method},
                )

            yield {
                'image': fake_img,
                'label': 1,
                'identity': unified_sample.identity,
                'source': 'deeplive',
                'method': unified_sample.method,
                'method_id': self.method_mapping.get(unified_sample.method, -1),
                'sample_id': unified_sample.sample_id,
                'frame_idx': frame_idx,
                'quality_domain': _domain_for_sample(unified_sample.method, 'deeplive', 1),
                'face_area_fraction': face_area_value,
            }

    def _iterate_visomaster_sample(
        self,
        unified_sample: UnifiedPairedSample,
        rng: random.Random
    ) -> Iterator[Dict[str, Any]]:
        """Load and yield frames from a VisoMaster sample."""
        from .visomaster import load_visomaster_frames
        
        sample = unified_sample.original_sample
        src = unified_sample.source
        frame_indices = self.visomaster_anchor_indices
        client = self._get_visomaster_gcs_client()
        executor = self._get_visomaster_download_executor()
        
        # Load frames from GCS
        real_frames, fake_frames = load_visomaster_frames(
            sample,
            frame_indices,
            as_array=True,
            client=client,
            executor=executor,
            parallel_download_workers=self.config.visomaster_parallel_download_workers,
        )
        
        # Yield paired frames (no landmarks for VisoMaster)
        for i, frame_idx in enumerate(frame_indices):
            if i >= len(real_frames) or i >= len(fake_frames):
                continue
            if real_frames[i] is None or fake_frames[i] is None:
                continue
            
            # Real frame
            real_img = real_frames[i]
            if self.transform:
                real_img = self._apply_transform(
                    real_img,
                    None,
                    {'label': 0, 'source': src, 'method': unified_sample.method},
                )
            
            yield {
                'image': real_img,
                'label': 0,
                'identity': unified_sample.identity,
                'source': src,
                'method': unified_sample.method,
                'method_id': self.method_mapping.get(unified_sample.method, -1),
                'sample_id': unified_sample.sample_id,
                'frame_idx': frame_idx,
                'quality_domain': _domain_for_sample(unified_sample.method, src, 0),
            }

            # Fake frame
            fake_img = fake_frames[i]
            if self.transform:
                fake_img = self._apply_transform(
                    fake_img,
                    None,
                    {'label': 1, 'source': src, 'method': unified_sample.method},
                )

            yield {
                'image': fake_img,
                'label': 1,
                'identity': unified_sample.identity,
                'source': src,
                'method': unified_sample.method,
                'method_id': self.method_mapping.get(unified_sample.method, -1),
                'sample_id': unified_sample.sample_id,
                'frame_idx': frame_idx,
                'quality_domain': _domain_for_sample(unified_sample.method, src, 1),
            }

    def _iterate_visomaster_enhanced_sample(
        self,
        unified_sample: UnifiedPairedSample,
        rng: random.Random,
    ) -> Iterator[Dict[str, Any]]:
        """Load and yield frames from a VisoMaster Enhanced sample (cross-bucket)."""
        from .visomaster import load_visomaster_enhanced_frames

        sample = unified_sample.original_sample
        frame_indices = self.config.visomaster_enhanced_sparse_indices
        client = self._get_visomaster_gcs_client()
        executor = self._get_visomaster_download_executor()

        # Cross-bucket loading: real from original bucket, fake from enhanced bucket
        real_frames, fake_frames = load_visomaster_enhanced_frames(
            sample,
            frame_indices,
            as_array=True,
            client=client,
            executor=executor,
            parallel_download_workers=self.config.visomaster_parallel_download_workers,
        )

        # Yield paired frames (no landmarks for VisoMaster Enhanced)
        for i, frame_idx in enumerate(frame_indices):
            if i >= len(real_frames) or i >= len(fake_frames):
                continue
            if real_frames[i] is None or fake_frames[i] is None:
                continue

            # Real frame
            real_img = real_frames[i]
            if self.transform:
                real_img = self._apply_transform(
                    real_img,
                    None,
                    {'label': 0, 'source': 'visomaster_enhanced', 'method': unified_sample.method},
                )

            yield {
                'image': real_img,
                'label': 0,
                'identity': unified_sample.identity,
                'source': 'visomaster_enhanced',
                'method': unified_sample.method,
                'method_id': self.method_mapping.get(unified_sample.method, -1),
                'sample_id': unified_sample.sample_id,
                'frame_idx': frame_idx,
                'quality_domain': _domain_for_sample(unified_sample.method, 'visomaster_enhanced', 0),
            }

            # Enhanced fake frame
            fake_img = fake_frames[i]
            if self.transform:
                fake_img = self._apply_transform(
                    fake_img,
                    None,
                    {'label': 1, 'source': 'visomaster_enhanced', 'method': unified_sample.method},
                )

            yield {
                'image': fake_img,
                'label': 1,
                'identity': unified_sample.identity,
                'source': 'visomaster_enhanced',
                'method': unified_sample.method,
                'method_id': self.method_mapping.get(unified_sample.method, -1),
                'sample_id': unified_sample.sample_id,
                'frame_idx': frame_idx,
                'quality_domain': _domain_for_sample(unified_sample.method, 'visomaster_enhanced', 1),
            }

    def _iterate_visomaster_teams_enhanced_sample(
        self,
        unified_sample: UnifiedPairedSample,
        rng: random.Random,
    ) -> Iterator[Dict[str, Any]]:
        """Load one merged base sample and choose a single fake branch at iteration time."""
        from .visomaster import (
            load_visomaster_teams_enhanced_frames,
            visomaster_enhancer_to_method_name,
        )

        sample = unified_sample.original_sample
        frame_indices = self.config.visomaster_teams_enhanced_sparse_indices
        available_enhancers = list(sample.available_enhancers)
        p_original = min(max(float(self.config.visomaster_teams_enhanced_p_original), 0.0), 1.0)
        client = self._get_visomaster_gcs_client()
        executor = self._get_visomaster_download_executor()

        branch = "original"
        if available_enhancers and rng.random() >= p_original:
            branch = rng.choice(available_enhancers)

        method_name = (
            sample.original_method
            if branch == "original"
            else visomaster_enhancer_to_method_name(branch)
        )

        real_frames, fake_frames = load_visomaster_teams_enhanced_frames(
            sample,
            frame_indices,
            fake_branch=branch,
            as_array=True,
            client=client,
            executor=executor,
            parallel_download_workers=self.config.visomaster_parallel_download_workers,
        )

        real_source = (
            'deeplive_teams' if sample.companion_domain == 'teams_v2' else 'visomaster'
        )
        real_quality_domain = _domain_for_sample(method_name, real_source, 0)
        fake_source = (
            real_source if branch == "original" else 'visomaster_enhanced'
        )
        fake_quality_domain = _domain_for_sample(method_name, fake_source, 1)

        src = 'visomaster_teams_enhanced'
        method_id = self.method_mapping.get(method_name, -1)

        for i, frame_idx in enumerate(frame_indices):
            if i >= len(real_frames) or i >= len(fake_frames):
                continue
            if real_frames[i] is None or fake_frames[i] is None:
                continue

            real_img = real_frames[i]
            if self.transform:
                real_img = self._apply_transform(
                    real_img,
                    None,
                    {'label': 0, 'source': src, 'method': method_name},
                )

            yield {
                'image': real_img,
                'label': 0,
                'identity': unified_sample.identity,
                'source': src,
                'method': method_name,
                'method_id': method_id,
                'sample_id': unified_sample.sample_id,
                'frame_idx': frame_idx,
                'quality_domain': real_quality_domain,
                'companion_domain': sample.companion_domain,
                'fake_branch': branch,
            }

            fake_img = fake_frames[i]
            if self.transform:
                fake_img = self._apply_transform(
                    fake_img,
                    None,
                    {'label': 1, 'source': src, 'method': method_name},
                )

            yield {
                'image': fake_img,
                'label': 1,
                'identity': unified_sample.identity,
                'source': src,
                'method': method_name,
                'method_id': method_id,
                'sample_id': unified_sample.sample_id,
                'frame_idx': frame_idx,
                'quality_domain': fake_quality_domain,
                'companion_domain': sample.companion_domain,
                'fake_branch': branch,
            }

    def _iterate_visomaster_res_variant_sample(
        self,
        unified_sample: UnifiedPairedSample,
        rng: random.Random,
    ) -> Iterator[Dict[str, Any]]:
        """Load and yield frames from a VisoMaster resolution-variant sample.

        Reuses the same cross-bucket loading as enhanced samples (real from
        original bucket, fake from enhanced bucket) but tags output with
        ``source='visomaster_res_variant'`` and uses the res-variant sparse
        indices config.
        """
        from .visomaster import load_visomaster_enhanced_frames

        sample = unified_sample.original_sample
        frame_indices = self.config.visomaster_res_variant_sparse_indices
        client = self._get_visomaster_gcs_client()
        executor = self._get_visomaster_download_executor()

        real_frames, fake_frames = load_visomaster_enhanced_frames(
            sample,
            frame_indices,
            as_array=True,
            client=client,
            executor=executor,
            parallel_download_workers=self.config.visomaster_parallel_download_workers,
        )

        src = 'visomaster_res_variant'
        for i, frame_idx in enumerate(frame_indices):
            if i >= len(real_frames) or i >= len(fake_frames):
                continue
            if real_frames[i] is None or fake_frames[i] is None:
                continue

            # Real frame
            real_img = real_frames[i]
            if self.transform:
                real_img = self._apply_transform(
                    real_img,
                    None,
                    {'label': 0, 'source': src, 'method': unified_sample.method},
                )

            yield {
                'image': real_img,
                'label': 0,
                'identity': unified_sample.identity,
                'source': src,
                'method': unified_sample.method,
                'method_id': self.method_mapping.get(unified_sample.method, -1),
                'sample_id': unified_sample.sample_id,
                'frame_idx': frame_idx,
                'quality_domain': _domain_for_sample(unified_sample.method, src, 0),
            }

            # Fake frame
            fake_img = fake_frames[i]
            if self.transform:
                fake_img = self._apply_transform(
                    fake_img,
                    None,
                    {'label': 1, 'source': src, 'method': unified_sample.method},
                )

            yield {
                'image': fake_img,
                'label': 1,
                'identity': unified_sample.identity,
                'source': src,
                'method': unified_sample.method,
                'method_id': self.method_mapping.get(unified_sample.method, -1),
                'sample_id': unified_sample.sample_id,
                'frame_idx': frame_idx,
                'quality_domain': _domain_for_sample(unified_sample.method, src, 1),
            }

    def _iterate_proper_data_sample(
        self,
        unified_sample: UnifiedPairedSample,
        rng: random.Random,
    ) -> Iterator[Dict[str, Any]]:
        """Load and yield frames from an explicit proper-data inventory sample."""
        from .proper_data import load_proper_data_frames

        sample = unified_sample.original_sample
        src = unified_sample.source
        frame_indices = self.config.proper_data_sparse_indices
        uses_gcs = any(
            str(path).startswith('gs://')
            for path in [*sample.real_frame_paths, *sample.fake_frame_paths]
        )
        client = self._get_proper_data_gcs_client() if uses_gcs else None
        executor = self._get_proper_data_download_executor() if uses_gcs else None

        real_frames, fake_frames = load_proper_data_frames(
            sample,
            frame_indices,
            as_array=True,
            client=client,
            executor=executor,
            parallel_download_workers=self.config.proper_data_parallel_download_workers,
        )

        for i, frame_idx in enumerate(frame_indices):
            if i >= len(real_frames) or i >= len(fake_frames):
                continue
            if real_frames[i] is None or fake_frames[i] is None:
                continue

            real_img = real_frames[i]
            if self.transform:
                real_img = self._apply_transform(
                    real_img,
                    None,
                    {'label': 0, 'source': src, 'method': unified_sample.method},
                )

            shared_meta = {
                'identity': unified_sample.identity,
                'source': src,
                'method': unified_sample.method,
                'method_id': self.method_mapping.get(unified_sample.method, -1),
                'sample_id': unified_sample.sample_id,
                'base_capture_id': sample.base_capture_id,
                'capture_session_id': sample.capture_session_id,
                'split_group_id': sample.split_group_id,
                'quality_band': sample.quality_band,
                'face_scale_band': sample.face_scale_band,
                'transport': sample.transport,
                'enhancement': sample.enhancement,
                'generator_family': sample.generator_family,
                'generator_method': sample.generator_method,
                'frame_idx': frame_idx,
            }

            yield {
                'image': real_img,
                'label': 0,
                'quality_domain': _domain_for_sample(unified_sample.method, src, 0),
                **shared_meta,
            }

            fake_img = fake_frames[i]
            if self.transform:
                fake_img = self._apply_transform(
                    fake_img,
                    None,
                    {'label': 1, 'source': src, 'method': unified_sample.method},
                )

            yield {
                'image': fake_img,
                'label': 1,
                'quality_domain': _domain_for_sample(unified_sample.method, src, 1),
                **shared_meta,
            }

    def _iterate_teams_sample(
        self,
        unified_sample: UnifiedPairedSample,
        rng: random.Random,
    ) -> Iterator[Dict[str, Any]]:
        """Load and yield frames from a Teams-passthrough sample (JPG from GCS)."""
        sample: TeamsSample = unified_sample.original_sample
        src = unified_sample.source
        frame_indices = self.config.teams_sparse_indices

        # Re-use one client + thread pool per worker so Teams-heavy runs do not
        # rebuild GCS sessions or thread pools on every sample.
        client = self._get_teams_gcs_client()
        executor = self._get_teams_download_executor()
        bucket = client.bucket(sample.gcs_bucket)

        real_by_idx = _load_teams_frame_map(
            bucket,
            sample.real_prefix,
            frame_indices,
            executor=executor,
            parallel_download_workers=self.config.teams_parallel_download_workers,
        )
        fake_by_idx = _load_teams_frame_map(
            bucket,
            sample.fake_prefix,
            frame_indices,
            executor=executor,
            parallel_download_workers=self.config.teams_parallel_download_workers,
        )

        # Optional T3 IQ-shortcut keep-list (2026-05-09): real frames whose
        # URI is not in the frozenset are skipped. Fakes are not affected.
        teams_real_keep = getattr(self.config, "teams_real_frame_keep_list", None)
        teams_real_keep_per_method = getattr(
            self.config, "teams_real_frame_keep_list_per_method", None
        )
        per_method_set = None
        if teams_real_keep_per_method is not None:
            per_method_set = teams_real_keep_per_method.get(
                unified_sample.method, frozenset()
            )

        for frame_idx in frame_indices:
            if frame_idx not in real_by_idx or frame_idx not in fake_by_idx:
                continue

            real_uri = (
                f"gs://{sample.gcs_bucket}/{sample.real_prefix}"
                f"frame_{frame_idx:04d}.jpg"
            )
            if teams_real_keep is not None and real_uri not in teams_real_keep:
                continue
            if per_method_set is not None and real_uri not in per_method_set:
                continue

            # Real frame (no landmarks for Teams)
            real_img = real_by_idx[frame_idx]
            if self.transform:
                real_img = self._apply_transform(
                    real_img,
                    None,
                    {'label': 0, 'source': src, 'method': unified_sample.method},
                )

            yield {
                'image': real_img,
                'label': 0,
                'identity': unified_sample.identity,
                'source': src,
                'method': unified_sample.method,
                'method_id': self.method_mapping.get(unified_sample.method, -1),
                'sample_id': unified_sample.sample_id,
                'frame_idx': frame_idx,
                'quality_domain': _domain_for_sample(unified_sample.method, src, 0),
            }

            # Fake frame (no landmarks for Teams)
            fake_img = fake_by_idx[frame_idx]
            if self.transform:
                fake_img = self._apply_transform(
                    fake_img,
                    None,
                    {'label': 1, 'source': src, 'method': unified_sample.method},
                )

            yield {
                'image': fake_img,
                'label': 1,
                'identity': unified_sample.identity,
                'source': src,
                'method': unified_sample.method,
                'method_id': self.method_mapping.get(unified_sample.method, -1),
                'sample_id': unified_sample.sample_id,
                'frame_idx': frame_idx,
                'quality_domain': _domain_for_sample(unified_sample.method, src, 1),
            }

    def _iterate_unpaired_real_sample(
        self,
        unified_sample: "UnifiedUnpairedRealSample",
        rng: random.Random,
    ) -> Iterator[Dict[str, Any]]:
        """Load and yield frames from an unpaired real-only external sample."""
        import cv2
        import numpy as np
        from io import BytesIO

        frame_paths = unified_sample.frame_paths
        # Sample up to frames_per_sample frames
        n = min(len(frame_paths), self.config.frames_per_sample)
        selected = rng.sample(frame_paths, n) if len(frame_paths) > n else frame_paths

        for i, gcs_path in enumerate(selected):
            try:
                from fsspec import url_to_fs

                fs = url_to_fs(gcs_path)[0]
                with fs.open(gcs_path, "rb") as f:
                    buf = f.read()
                img = cv2.imdecode(
                    np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if img is None:
                    logger.warning("Failed to decode image: %s", gcs_path)
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as exc:
                logger.warning("Failed to load external real frame %s: %s", gcs_path, exc)
                continue

            if self.transform:
                img = self._apply_transform(
                    img,
                    None,
                    {
                        "label": 0,
                        "source": unified_sample.source,
                        "method": unified_sample.method,
                    },
                )

            yield {
                "image": img,
                "label": 0,
                "identity": unified_sample.identity,
                "source": unified_sample.source,
                "method": unified_sample.method,
                "method_id": self.method_mapping.get(unified_sample.method, -1),
                "sample_id": unified_sample.sample_id,
                "frame_idx": i,
                "quality_domain": _domain_for_sample(
                    unified_sample.method, unified_sample.source, 0
                ),
            }


# =============================================================================
# PE_PAIR_RANK_DRO group_id derivation (added 2026-05-07)
# =============================================================================
# Asymmetric R-D / F-B grouping per
# `analysis/group_id_design_audit_2026-05-06/outputs/group_id_python_snippet.py`.
# Embedded here (rather than imported) so the loader is self-contained — the
# snippet file remains the source of truth and tests pin the embedded copy
# against the snippet's behaviour.
#
# The two derive helpers below produce sensible group_id strings from the
# loader's existing fields. They are deliberately lossy: `quality` is set to
# 'unknown' (we don't carry hi-q/lo-q labels per-row), and method_family /
# enhancer_family / transport are derived by substring matching. Any mismatch
# vs the snippet's audit-time mapping degrades gracefully — the GroupDRO
# mixin's tolerant unknown-group_id path buckets unmatched groups to 0 with
# a one-shot warning.

CHRONIC_IDENTITIES = (
    "bla_bla_chow",
    "bla_bla_chow__s2",
    "PC_Generator__s22",
    "PC_Generator__s45",
    "roy_d",
    "Q__s6",
)


def _is_chronic_identity(base_identity: str) -> bool:
    if not isinstance(base_identity, str) or not base_identity:
        return False
    if base_identity in CHRONIC_IDENTITIES:
        return True
    s = base_identity.lower()
    return any(cid.lower() in s for cid in CHRONIC_IDENTITIES)


# Identities used by the multi-axis-GRL `is_dor` shortcut classifier.
# Per project_dor_drift_named_axes_2026-05-06, dor identities are the binding
# real-side cluster that all FT-from-P8A variants regress on. This list
# matches the dor cohort used by the per-frame Wilcoxon analysis in
# HANDOFF_2026-05-02_P18_DIAGNOSTICS_COMPLETE.
DOR_IDENTITIES = (
    "dor_shkedi",
    "healthy_dor",
    "dor",
)


def _is_dor_identity(base_identity: str) -> bool:
    if not isinstance(base_identity, str) or not base_identity:
        return False
    s = base_identity.lower()
    return any(d in s for d in ("dor_shkedi", "healthy_dor")) or s == "dor"


_METHOD_FAMILY_KEYWORDS = (
    # (substring lowercased, family label)
    ("inswapper", "inswapper"),
    ("inswap", "inswapper"),
    ("simswap", "simswap"),
    ("ghostface", "ghostface"),
    ("instyleswapper", "instyleswapper"),
    ("cscs", "cscs"),
    ("facedancer", "facedancer"),
    ("blendface", "blendface"),
    ("e4s", "e4s"),
    ("mobileswap", "mobileswap"),
    ("uniface", "uniface"),
)


def _method_family_from(method: str, label: int, source: str) -> str:
    """Coarse method-family bucketing for F-B GroupDRO key.

    Real-side rows always map to 'real_or_unknown'. Fake-side rows pattern-
    match the method/source against the known keyword list.
    """
    if int(label) == 0:
        return "real_or_unknown"
    m = (method or "").lower()
    s = (source or "").lower()
    # Source-anchored buckets first (visomaster + deeplive families).
    if s.startswith("visomaster"):
        return "visomaster"
    if s == "deeplive" or s == "deeplive_teams" or s == "deeplive_clean":
        return "deeplive"
    if s == "df40":
        # df40 carries the per-method label in `method`; fall through to keyword match.
        pass
    for keyword, family in _METHOD_FAMILY_KEYWORDS:
        if keyword in m:
            return family
    return m or "unknown_method"


_ENHANCER_FAMILY_KEYWORDS = (
    ("gpen", "gpen"),
    ("gfpgan", "gfpgan"),
    ("codeformer", "codeformer"),
    ("restoreformer", "restoreformer"),
)


def _enhancer_family_from(method: str, source: str) -> str:
    """Coarse enhancer-family bucketing.

    Substring match on method/source for the known restoration models. Falls
    back to 'enhanced_unknown' when the source signals enhancement but no
    specific model is identifiable, else 'none'.
    """
    haystack = f"{(method or '').lower()}|{(source or '').lower()}"
    for keyword, family in _ENHANCER_FAMILY_KEYWORDS:
        if keyword in haystack:
            return family
    if "enhanced" in haystack:
        return "enhanced_unknown"
    return "none"


def _transport_from_source(source: str, method: str) -> str:
    """Map per-row source/method to the transport axis used by GroupDRO.

    'teams_capture' for anything that traversed the Teams pipeline; the
    visomaster_teams_enhanced lane goes here even though its rendering is
    visomaster — the audit's `transport` semantic is "how the frame was
    captured / delivered", not "what fakery was applied".
    """
    s = (source or "").lower()
    m = (method or "").lower()
    if "teams" in s or "teams" in m:
        return "teams_capture"
    if s.startswith("visomaster"):
        return "visomaster"
    if s == "external":
        return "external"
    if s in ("df40", "deeplive", "deeplive_clean"):
        return "raw_capture"
    return "raw_capture"


def _make_group_id_string(
    label: int,
    method_family: str,
    enhancer_family: str,
    transport: str,
    quality: str,
    source: str,
    base_identity: str,
) -> str:
    """Build the asymmetric group_id string. Mirrors make_group_id in the
    audit snippet exactly; tested against it in
    tests/test_pair_rank_and_group_dro.py."""
    qb = quality.lower() if isinstance(quality, str) and quality.lower() in {"hi-q", "lo-q"} else "unknown"
    if int(label) == 1:
        return (
            f"fake|{method_family}|{enhancer_family}|{transport}|{qb}"
        )
    chronic_tag = "chronic" if _is_chronic_identity(base_identity) else "regular"
    return (
        f"real|{source or 'unknown'}|{transport}|{qb}|{chronic_tag}"
    )


def _derive_group_id_for_yield_row(row: Dict[str, Any]) -> Optional[str]:
    """Derive the asymmetric group_id string from a per-frame yield dict.

    Used by the collate when the loader didn't pre-stamp `group_id`.
    Returns None for rows that lack the minimal fields (label / source /
    method); the GroupDRO mixin treats None as "missing" and routes to the
    unknown-id fallback bucket.
    """
    if "label" not in row:
        return None
    label = int(row.get("label", 0))
    method = row.get("method") or ""
    source = row.get("source") or ""
    identity = row.get("identity") or row.get("base_identity") or ""
    method_family = _method_family_from(method, label, source)
    enhancer_family = _enhancer_family_from(method, source)
    transport = _transport_from_source(source, method)
    quality = row.get("quality", "unknown") or "unknown"
    return _make_group_id_string(
        label=label,
        method_family=method_family,
        enhancer_family=enhancer_family,
        transport=transport,
        quality=quality,
        source=source,
        base_identity=identity,
    )


def _derive_group_id_for_sample(sample: Any, label: int) -> Optional[str]:
    """Same derivation but operating on a UnifiedPairedSample-like dataclass.

    Used by the build pre-pass below to enumerate the universe of group_ids
    over `all_samples × {label=0, label=1}` at config-load time. Each paired
    sample contributes both a real-side and a fake-side group_id; an
    unpaired-real sample only contributes its real-side group_id (the caller
    is responsible for skipping label=1 on those).
    """
    if sample is None:
        return None
    method = getattr(sample, "method", "") or ""
    source = getattr(sample, "source", "") or ""
    identity = getattr(sample, "identity", "") or ""
    method_family = _method_family_from(method, label, source)
    enhancer_family = _enhancer_family_from(method, source)
    transport = _transport_from_source(source, method)
    return _make_group_id_string(
        label=label,
        method_family=method_family,
        enhancer_family=enhancer_family,
        transport=transport,
        quality="unknown",
        source=source,
        base_identity=identity,
    )


def build_group_id_mapping_for_samples(samples: List[Any]) -> Dict[str, int]:
    """Walk a list of `UnifiedPairedSample` / `UnifiedUnpairedRealSample`
    instances and build the group_id → int mapping for the GroupDRO mixin.

    Paired samples contribute both label=0 and label=1 derivations; unpaired
    real samples (`is_unpaired_real=True`) contribute only label=0.
    """
    seen = set()
    for sample in samples:
        if getattr(sample, "is_unpaired_real", False):
            gid = _derive_group_id_for_sample(sample, 0)
            if gid is not None:
                seen.add(gid)
            continue
        for lbl in (0, 1):
            gid = _derive_group_id_for_sample(sample, lbl)
            if gid is not None:
                seen.add(gid)
    return {g: i for i, g in enumerate(sorted(seen))}


# =============================================================================
# Collate Function
# =============================================================================

def combined_paired_collate_fn(
    batch: List[Dict[str, Any]],
    target_size: Tuple[int, int] = (224, 224)
) -> Dict[str, Any]:
    """
    Collate function for combined paired batches.
    
    Groups frames by (sample_id, label) to create video-style batches.
    """
    import cv2
    import numpy as np
    
    # CLIP normalization values
    CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    
    # Group frames by (sample_id, label)
    groups = defaultdict(list)
    for item in batch:
        key = f"{item['sample_id']}_{item['label']}"
        groups[key].append(item)
    
    # Sort frames within each group
    for key in groups:
        groups[key].sort(key=lambda x: x['frame_idx'])
    
    # Process each group
    video_images = []
    video_labels = []
    video_ids = []
    video_method_ids = []
    video_quality_domains = []
    video_pair_ids = []  # per-video pair_id (= sample_id) for PE_PAIR_RANK_DRO loss
    video_group_ids = []  # per-video group_id string for GroupDRO (R-D / F-B keying)

    video_face_area_fraction = []  # per-video face_area_fraction (NaN if absent)
    # Multi-axis GRL labels (added 2026-05-10 — used by detectors.effort_detector
    # MultiAxisGRLBlock when `multi_axis_grl.enabled: true` in yaml). Identity-
    # derived per-video binary flags; safe to emit unconditionally — the model
    # only reads these when the feature is enabled.
    video_chronic_flag = []
    video_is_dor = []
    # Substrate-pair fields (BACKBONE 2026-05-22): per-video pair_id + transport
    # stamped from the inventory CSV by data.sample.substrate_paired.SubstratePairStamper.
    # When the stamper is disabled (default), every entry is -1 (no-op).
    video_substrate_pair_id = []
    video_substrate_transport = []

    for video_key, frames in groups.items():
        if len(frames) == 0:
            continue

        frame_tensors = []
        for item in frames:
            img = item['image']

            if isinstance(img, torch.Tensor):
                img = img.numpy()

            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            # Face scale-jitter (anti-shortcut, label-symmetric) — runs BEFORE
            # the canonical 224×224 resize so the final face area is randomized.
            from data.augmentations.face_scale_jitter import apply_face_scale_jitter
            img = apply_face_scale_jitter(img)

            # Resolution-chain aug — random downsample->upsample chain BEFORE
            # the canonical resize. Targets the 2026-05-15 CPU-probe finding
            # that source-resolution dominates the per-frame score swing on
            # reals (size axis 38-42% of variance vs kernel 21-23%).
            from data.augmentations.resolution_chain_aug import apply_resolution_chain_aug
            img = apply_resolution_chain_aug(img)

            if img.shape[:2] != target_size:
                img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

            # Band-limited Fourier amplitude randomization — operates on the
            # canonical 224×224 frame, post-resize, pre-normalize. Bands 8-13
            # randomized; bands 5-6 preserved (manipulation-signal-carrying).
            from data.augmentations.fourier_band_aug import apply_fourier_band_aug
            img = apply_fourier_band_aug(img)

            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_tensor = (img_tensor - CLIP_MEAN) / CLIP_STD
            frame_tensors.append(img_tensor)

        video_tensor = torch.stack(frame_tensors)
        video_images.append(video_tensor)
        video_labels.append(frames[0]['label'])
        video_ids.append(video_key)
        # Preserve method_id for Group DRO (all frames in a group share the same method)
        video_method_ids.append(frames[0].get('method_id', -1))
        video_quality_domains.append(frames[0].get('quality_domain', 0))
        # Optional per-video face_area_fraction (deeplive only); NaN if absent
        video_face_area_fraction.append(float(frames[0].get('face_area_fraction', float('nan'))))
        # PE_PAIR_RANK_DRO: pair_id links real-fake same-source videos.
        # sample_id is the source-grain id; the existing groupby splits by label
        # so a paired (real, fake) pair shares pair_id but lands in two videos.
        # Empty string (unpaired reals like external_vcd_real) → skipped by the loss.
        video_pair_ids.append(str(frames[0].get('sample_id', '')))
        # PE_PAIR_RANK_DRO: per-video group_id (asymmetric R-D / F-B key) for
        # the multi-axis GroupDRO term. If the loader didn't pre-stamp
        # `group_id` on the per-frame row, derive it from the existing
        # method/source/identity fields. Returning None is safe — the
        # GroupDRO mixin's tolerant unknown-group_id path covers the gap.
        gid = frames[0].get('group_id')
        if gid is None:
            gid = _derive_group_id_for_yield_row(frames[0])
        video_group_ids.append(gid)
        # Multi-axis GRL identity-derived flags (added 2026-05-10).
        identity_str = (
            frames[0].get('identity')
            or frames[0].get('base_identity')
            or ""
        )
        video_chronic_flag.append(int(_is_chronic_identity(identity_str)))
        video_is_dor.append(int(_is_dor_identity(identity_str)))

        # Substrate-pair fields (BACKBONE 2026-05-22). When the stamper is
        # active and the identity matches an inventory row, this stamps
        # substrate_pair_id (int) and substrate_transport (0=clean, 1=teams).
        # Default no-op: returns (-1, -1) when stamper is disabled.
        try:
            from data.sample.substrate_paired import get_active_stamper
            stamper = get_active_stamper()
            if stamper is not None and stamper.enabled:
                pair_id, transport = stamper.lookup(
                    identity=identity_str,
                    source=frames[0].get('source', '') or '',
                    companion_domain=frames[0].get('companion_domain'),
                    label=int(frames[0].get('label', 0)),
                )
            else:
                pair_id, transport = -1, -1
        except Exception:
            pair_id, transport = -1, -1
        video_substrate_pair_id.append(int(pair_id))
        video_substrate_transport.append(int(transport))

    if len(video_images) == 0:
        return {
            'image': torch.zeros(0, 1, 3, target_size[0], target_size[1]),
            'label': torch.zeros(0, dtype=torch.long),
            'video_id': [],
            'method_id': torch.zeros(0, dtype=torch.long),
            'quality_domain': torch.zeros(0, dtype=torch.long),
            'pair_id': [],
            'group_id': [],
            'chronic_flag': torch.zeros(0, dtype=torch.long),
            'is_dor': torch.zeros(0, dtype=torch.long),
            'substrate_pair_id': torch.zeros(0, dtype=torch.long),
            'substrate_transport': torch.zeros(0, dtype=torch.long),
        }

    # Pad to same length
    max_frames = max(v.shape[0] for v in video_images)
    padded_videos = []
    for video in video_images:
        if video.shape[0] < max_frames:
            padding = torch.zeros(max_frames - video.shape[0], *video.shape[1:])
            video = torch.cat([video, padding], dim=0)
        padded_videos.append(video)

    images = torch.stack(padded_videos)
    labels = torch.tensor(video_labels, dtype=torch.long)
    method_ids = torch.tensor(video_method_ids, dtype=torch.long)
    quality_domains = torch.tensor(video_quality_domains, dtype=torch.long)

    return {
        'image': images,
        'label': labels,
        'video_id': video_ids,
        'method_id': method_ids,
        'quality_domain': quality_domains,
        'face_area_fraction': torch.tensor(video_face_area_fraction, dtype=torch.float32),
        'pair_id': video_pair_ids,
        'group_id': video_group_ids,
        'chronic_flag': torch.tensor(video_chronic_flag, dtype=torch.long),
        'is_dor': torch.tensor(video_is_dor, dtype=torch.long),
        'substrate_pair_id': torch.tensor(video_substrate_pair_id, dtype=torch.long),
        'substrate_transport': torch.tensor(video_substrate_transport, dtype=torch.long),
    }


# =============================================================================
# Validation Adapter
# =============================================================================

class CombinedValidationAdapter:
    """Adapter to provide trainer-compatible interface for validation."""
    
    def __init__(
        self,
        dataloader: DataLoader,
        samples: List[UnifiedPairedSample],
        name: str = 'combined_val',
        method_mapping: Optional[Dict[str, int]] = None,
    ):
        self._dataloader = dataloader
        self._samples = samples
        self._name = name
        self.videos_by_method = {name: samples}
        self.method_mapping = method_mapping or {}
        self.method_id_to_name = {v: k for k, v in self.method_mapping.items()}
    
    def keys(self):
        return self.videos_by_method.keys()
    
    def __getitem__(self, method: str) -> DataLoader:
        return self._dataloader
    
    def __bool__(self) -> bool:
        return len(self._samples) > 0
    
    def __iter__(self):
        return iter(self._dataloader)
    
    @property
    def dataset(self):
        return self._dataloader.dataset


# =============================================================================
# Data Source Registration
# =============================================================================

@register_data_source('combined_paired')
def create_combined_paired_pipeline(
    config: Dict[str, Any],
    data_config: Dict[str, Any],
    logger: logging.Logger,
    **kwargs
) -> DataPipelineResult:
    """
    Create data pipeline combining DF40, DeepLive, and optionally VisoMaster datasets.
    
    Args:
        config: Main training configuration
        data_config: Data-specific configuration
        logger: Logger
        
    Returns:
        DataPipelineResult with train_loader, validation loaders, and metadata
    """
    combined_config = data_config.get('combined_paired', {})
    run_seed = config.get('manualSeed')
    if run_seed is None:
        run_seed = config.get('seed')
    if run_seed is None:
        run_seed = combined_config.get('seed', 737)
    split_seed = combined_config.get('split_seed', run_seed)
    identity_split_mode = normalize_method_name(
        combined_config.get('identity_split_mode', 'shuffle')
    ) or 'shuffle'
    combined_config['split_seed'] = split_seed
    combined_config['identity_split_mode'] = identity_split_mode

    logger.info("=" * 70)
    logger.info("Combined Paired Data Source: Initializing")
    logger.info("=" * 70)
    logger.info(
        "Seed configuration: run_seed=%s, split_seed=%s, identity_split_mode=%s",
        run_seed,
        split_seed,
        identity_split_mode,
    )

    aug_config = config.get('augmentation') or (data_config or {}).get('augmentation') or {}
    enhanced_strategy_names = tuple(
        (aug_config.get('routing', {}) or {}).get('enhanced_strategy_names')
        or DEFAULT_ENHANCED_STRATEGIES
    )

    visomaster_config = combined_config.get('visomaster', {})
    visomaster_hints_config = combined_config.get('visomaster_hints', {})
    teams_config = combined_config.get('teams', {})
    visomaster_hints_teams_config = combined_config.get('visomaster_hints_teams', {})
    proper_data_config = combined_config.get('proper_data', {})

    visomaster_enabled = bool(visomaster_config.get('enabled', False))
    visomaster_hints_enabled = bool(visomaster_hints_config.get('enabled', False))
    teams_enabled = bool(teams_config.get('enabled', False))
    visomaster_hints_teams_enabled = bool(
        visomaster_hints_teams_config.get('enabled', False)
    )
    proper_data_enabled = bool(proper_data_config.get('enabled', False))
    teams_policy_filter_enabled = bool(teams_config.get('apply_bad_data_policy', False))

    if visomaster_enabled and visomaster_hints_enabled:
        raise ValueError(
            "combined_paired.visomaster and combined_paired.visomaster_hints "
            "cannot both be enabled. Use the explicit hint lane instead of the old "
            "VisoMaster lane."
        )
    if teams_enabled and visomaster_hints_teams_enabled and not teams_policy_filter_enabled:
        raise ValueError(
            "combined_paired.teams.apply_bad_data_policy must be true when "
            "visomaster_hints_teams is enabled, otherwise retained weak-signal "
            "rows would be duplicated inside the direct Teams lane."
        )

    visomaster_policy = None
    if visomaster_hints_enabled or visomaster_hints_teams_enabled or teams_policy_filter_enabled:
        visomaster_policy = _load_visomaster_bad_data_policy_bundle(
            config=config,
            combined_config=combined_config,
            logger=logger,
            required=True,
        )
    
    # ==========================================================================
    # Load DF40 Dataset
    # ==========================================================================
    df40_config = combined_config.get('df40', {})
    df40_enabled = df40_config.get('enabled', True)
    df40_samples = []
    df40_dataset = None
    
    if df40_enabled:
        from dataset.df40_paired_dataset import DF40PairedDataset
        
        pair_json_path = df40_config.get('pair_json', 'dataset/df40_pairs/df40-pair-matching.json')
        if not os.path.isabs(pair_json_path):
            training_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            pair_json_path = os.path.join(training_dir, pair_json_path)

        gcs_bucket = df40_config.get('gcs_bucket', 'df40-frames-recropped-rfa85')
        gcs_project = df40_config.get('gcs_project', os.environ.get('GOOGLE_CLOUD_PROJECT', 'train-cvit2'))
        methods = df40_config.get('methods')
        if methods is not None and len(methods) == 0:
            methods = None

        logger.info(f"Loading DF40 dataset:")
        logger.info(f"  - Pair JSON: {pair_json_path}")
        logger.info(f"  - GCS bucket: {gcs_bucket}")
        logger.info(f"  - Methods: {methods if methods else 'ALL'}")

        df40_dataset = DF40PairedDataset(
            pair_json_path=pair_json_path,
            bucket_name=gcs_bucket,
            gcs_project=gcs_project,
            methods=methods,
        )
        
        raw_df40_samples = df40_dataset.discover_samples()
        df40_samples = create_unified_samples_from_df40(raw_df40_samples, logger)
    else:
        logger.info("DF40 dataset: DISABLED")
    
    # ==========================================================================
    # Load DeepLive Dataset
    # ==========================================================================
    deeplive_config = combined_config.get('deeplive', {})
    deeplive_enabled = deeplive_config.get('enabled', True)
    deeplive_samples = []
    deeplive_dataset = None
    deeplive_strategy_counts = {"raw": {}, "effective": {}}
    deeplive_preflight = {
        "mode": None,
        "passed": True,
        "errors": [],
        "effective_counts": {},
        "min_counts": {},
    }
    
    if deeplive_enabled:
        from dataset.deeplive_dataset import DeepLiveDataset
        
        gcs_bucket = deeplive_config.get('gcs_bucket', 'live-deepfake-methods-real-and-fake-frames-cropped')
        gcs_project = deeplive_config.get('gcs_project', os.environ.get('GOOGLE_CLOUD_PROJECT', 'train-cvit2'))
        use_landmarks = deeplive_config.get('use_landmarks', True)
        include_strategies = deeplive_config.get('include_strategies')
        if include_strategies is None:
            legacy_strategies = deeplive_config.get('strategies')
            if legacy_strategies and legacy_strategies != 'all':
                include_strategies = legacy_strategies
        exclude_strategies = deeplive_config.get('exclude_strategies') or []
        include_strategies_norm = (
            [normalize_method_name(s) for s in include_strategies] if include_strategies else []
        )
        exclude_strategies_norm = [normalize_method_name(s) for s in exclude_strategies]
        strategies_arg = include_strategies if include_strategies else 'all'
        deeplive_cache_manifest_uri = (
            deeplive_config.get("cache_manifest_uri")
            or deeplive_config.get("cache_manifest_path")
        )
        deeplive_cache_max_age_hours = float(deeplive_config.get("cache_max_age_hours") or 12.0)
        deeplive_cache_revision = deeplive_config.get("cache_revision")
        deeplive_max_samples_total = deeplive_config.get("max_samples_total")
        deeplive_max_samples_per_strategy = deeplive_config.get("max_samples_per_strategy")
        strategy_expect_cfg = deeplive_config.get("strategy_expectations", {}) or {}
        expectation_mode = _resolve_enhanced_expectation_mode(
            include_strategies=include_strategies_norm,
            exclude_strategies=exclude_strategies_norm,
            enhanced_strategy_names=enhanced_strategy_names,
            explicit_mode=strategy_expect_cfg.get("mode"),
        )
        expectation_min_counts = strategy_expect_cfg.get("min_counts")

        logger.info(f"Loading DeepLive dataset:")
        logger.info(f"  - GCS bucket: {gcs_bucket}")
        logger.info(f"  - Use landmarks: {use_landmarks}")
        logger.info(f"  - Include strategies: {include_strategies if include_strategies else 'ALL'}")
        logger.info(f"  - Exclude strategies: {exclude_strategies if exclude_strategies else 'NONE'}")
        logger.info(
            "  - Discovery cache: %s (max_age_hours=%s revision=%s)",
            deeplive_cache_manifest_uri or "DISABLED",
            deeplive_cache_max_age_hours,
            deeplive_cache_revision or "",
        )
        logger.info(
            "  - Discovery caps: total=%s per_strategy=%s",
            deeplive_max_samples_total if deeplive_max_samples_total else "NONE",
            deeplive_max_samples_per_strategy if deeplive_max_samples_per_strategy else "NONE",
        )

        deeplive_dataset = DeepLiveDataset(
            bucket_name=gcs_bucket,
            gcs_project=gcs_project,
            use_landmarks=use_landmarks,
            strategies=strategies_arg,
            cache_manifest_uri=deeplive_cache_manifest_uri,
            cache_max_age_hours=deeplive_cache_max_age_hours,
            cache_revision=deeplive_cache_revision,
            max_samples_total=deeplive_max_samples_total,
            max_samples_per_strategy=deeplive_max_samples_per_strategy,
        )

        raw_deeplive_samples = deeplive_dataset.discover_samples()
        if exclude_strategies_norm:
            before_count = len(raw_deeplive_samples)
            raw_deeplive_samples = [
                sample for sample in raw_deeplive_samples
                if normalize_method_name(getattr(sample, 'effective_strategy', getattr(sample, 'strategy', '')))
                not in exclude_strategies_norm
            ]
            logger.info(
                f"Applied DeepLive exclude_strategies: {before_count} -> {len(raw_deeplive_samples)} samples"
            )
        if include_strategies_norm:
            # Defensive second-pass filter for any strategy casing mismatch.
            before_count = len(raw_deeplive_samples)
            raw_deeplive_samples = [
                sample for sample in raw_deeplive_samples
                if normalize_method_name(getattr(sample, 'effective_strategy', getattr(sample, 'strategy', '')))
                in include_strategies_norm
            ]
            logger.info(
                f"Applied DeepLive include_strategies (defensive): {before_count} -> {len(raw_deeplive_samples)} samples"
            )

        deeplive_strategy_counts = _count_deeplive_strategies(raw_deeplive_samples)
        logger.info(f"DeepLive raw strategy counts (post-filter): {deeplive_strategy_counts['raw']}")
        logger.info(f"DeepLive effective strategy counts (post-filter): {deeplive_strategy_counts['effective']}")

        deeplive_preflight = _run_deeplive_preflight_checks(
            effective_counts=deeplive_strategy_counts["effective"],
            expectation_mode=expectation_mode,
            min_counts=expectation_min_counts,
        )
        if deeplive_preflight["mode"]:
            logger.info(
                "DeepLive strategy preflight (%s): passed=%s",
                deeplive_preflight["mode"],
                deeplive_preflight["passed"],
            )
            if not deeplive_preflight["passed"]:
                for err in deeplive_preflight["errors"]:
                    logger.error(f"  - {err}")
                raise ValueError(
                    "DeepLive strategy preflight failed: "
                    + "; ".join(deeplive_preflight["errors"])
                )

        # Need to load manifests to get original_video_name for identity extraction
        # For now, we'll extend the sample objects with manifest data
        deeplive_samples = create_unified_samples_from_deeplive(
            raw_deeplive_samples, deeplive_dataset, logger
        )
    else:
        logger.info("DeepLive dataset: DISABLED")
    
    # ==========================================================================
    # Load VisoMaster Dataset (Phase 2 / WT-B explicit hint lanes)
    # ==========================================================================
    visomaster_samples: List[UnifiedPairedSample] = []
    visomaster_hint_samples: List[UnifiedPairedSample] = []
    raw_visomaster_samples: List[Any] = []

    if visomaster_enabled or visomaster_hints_enabled:
        from .visomaster import discover_visomaster_samples

        viso_source_cfg = visomaster_config if visomaster_enabled else visomaster_hints_config
        viso_bucket = viso_source_cfg.get(
            'gcs_bucket', 'live-deepfake-methods-real-and-fake-frames-cropped'
        )
        viso_frames_bucket = viso_source_cfg.get(
            'frames_bucket', 'live-deepfake-methods-real-and-fake-frames'
        )
        viso_gcs_project = viso_source_cfg.get(
            'gcs_project', os.environ.get('GOOGLE_CLOUD_PROJECT', 'train-cvit2')
        )
        viso_swap_models = viso_source_cfg.get('swap_models')  # None = all
        viso_tiers = viso_source_cfg.get('tiers')  # None = all
        viso_cache_manifest_uri = (
            viso_source_cfg.get('cache_manifest_uri')
            or viso_source_cfg.get('cache_manifest_path')
        )
        viso_cache_max_age_hours = float(viso_source_cfg.get('cache_max_age_hours') or 12.0)
        viso_cache_validate_listing = bool(viso_source_cfg.get('cache_validate_listing', True))
        viso_cache_revision = viso_source_cfg.get('cache_revision')
        viso_label = "VisoMaster dataset" if visomaster_enabled else "VisoMaster weak-signal source pool"

        logger.info(f"Loading {viso_label}:")
        logger.info(f"  - GCS bucket: {viso_bucket}")
        logger.info(f"  - Swap models: {viso_swap_models or 'ALL'}")
        logger.info(f"  - Tiers: {viso_tiers or 'ALL'}")
        logger.info(
            "  - Discovery cache: %s (max_age_hours=%s validate_listing=%s revision=%s)",
            viso_cache_manifest_uri or "DISABLED",
            viso_cache_max_age_hours,
            viso_cache_validate_listing,
            viso_cache_revision or "",
        )
        
        raw_visomaster_samples = discover_visomaster_samples(
            bucket_name=viso_bucket,
            frames_bucket_name=viso_frames_bucket,
            gcs_project=viso_gcs_project,
            swap_models=viso_swap_models,
            tiers=viso_tiers,
            cache_manifest_uri=viso_cache_manifest_uri,
            cache_max_age_hours=viso_cache_max_age_hours,
            cache_validate_listing=viso_cache_validate_listing,
            cache_revision=viso_cache_revision,
            log=logger,
        )

        if visomaster_enabled:
            visomaster_samples = create_unified_samples_from_visomaster(
                raw_visomaster_samples, logger
            )
    else:
        logger.info("VisoMaster dataset: DISABLED")

    if visomaster_hints_enabled:
        raw_visomaster_hint_samples = _select_visomaster_hints_samples(
            raw_visomaster_samples,
            visomaster_policy,
            logger,
        )
        if len(raw_visomaster_hint_samples) == 0:
            logger.warning(
                "VisoMaster hints are ENABLED but 0 retained hint samples were selected "
                "from the April 17 policy packet."
            )
        visomaster_hint_samples = create_unified_samples_from_visomaster_hints(
            raw_visomaster_hint_samples,
            logger,
        )
    else:
        logger.info("VisoMaster hints dataset: DISABLED")
    
    # ==========================================================================
    # Load Teams Passthrough Dataset (Phase 2 R9)
    # ==========================================================================
    teams_samples: List[UnifiedPairedSample] = []
    visomaster_hint_teams_samples: List[UnifiedPairedSample] = []
    raw_teams_samples: List[Any] = []

    if teams_enabled or visomaster_hints_teams_enabled:
        teams_bucket = teams_config.get(
            'gcs_bucket',
            'live-deepfake-methods-real-and-fake-frames-cropped-teams',
        )
        hint_teams_bucket = visomaster_hints_teams_config.get('gcs_bucket', teams_bucket)
        teams_gcs_project = teams_config.get(
            'gcs_project',
            os.environ.get('GOOGLE_CLOUD_PROJECT', 'train-cvit2'),
        )
        hint_teams_gcs_project = visomaster_hints_teams_config.get(
            'gcs_project',
            teams_gcs_project,
        )
        teams_require_complete = teams_config.get('require_pair_complete', True)
        hint_teams_require_complete = visomaster_hints_teams_config.get(
            'require_pair_complete',
            teams_require_complete,
        )
        teams_cache_manifest_uri = teams_config.get('cache_manifest_uri')
        if teams_cache_manifest_uri is None:
            teams_cache_manifest_uri = visomaster_hints_teams_config.get('cache_manifest_uri')
        teams_cache_max_age_hours = teams_config.get('cache_max_age_hours')
        if teams_cache_max_age_hours is None:
            teams_cache_max_age_hours = visomaster_hints_teams_config.get('cache_max_age_hours')
        teams_cache_max_age_hours = float(teams_cache_max_age_hours or 12.0)
        if 'cache_validate_listing' in teams_config:
            teams_cache_validate_listing = bool(teams_config.get('cache_validate_listing'))
        elif 'cache_validate_listing' in visomaster_hints_teams_config:
            teams_cache_validate_listing = bool(
                visomaster_hints_teams_config.get('cache_validate_listing')
            )
        else:
            teams_cache_validate_listing = True
        teams_cache_revision = teams_config.get('cache_revision')
        if teams_cache_revision is None:
            teams_cache_revision = visomaster_hints_teams_config.get('cache_revision')

        if teams_enabled and visomaster_hints_teams_enabled and hint_teams_bucket != teams_bucket:
            raise ValueError(
                "combined_paired.teams and combined_paired.visomaster_hints_teams "
                "must point at the same Teams bucket when both are enabled."
            )
        if teams_enabled and visomaster_hints_teams_enabled and hint_teams_gcs_project != teams_gcs_project:
            raise ValueError(
                "combined_paired.teams and combined_paired.visomaster_hints_teams "
                "must use the same GCS project when both are enabled."
            )

        discovery_bucket = teams_bucket if teams_enabled else hint_teams_bucket
        discovery_project = teams_gcs_project if teams_enabled else hint_teams_gcs_project
        discovery_require_complete = (
            teams_require_complete if teams_enabled else hint_teams_require_complete
        )

        logger.info("Loading Teams passthrough source pool:")
        logger.info(f"  - GCS bucket: {discovery_bucket}")
        logger.info(f"  - Require pair_complete: {discovery_require_complete}")
        logger.info(f"  - Apply bad-data policy to direct Teams lane: {teams_policy_filter_enabled}")
        logger.info(
            "  - Discovery cache: %s (max_age_hours=%s validate_listing=%s revision=%s)",
            teams_cache_manifest_uri or "DISABLED",
            teams_cache_max_age_hours,
            teams_cache_validate_listing,
            teams_cache_revision or "",
        )

        raw_teams_samples = discover_teams_passthrough_samples(
            gcs_bucket=discovery_bucket,
            gcs_project=discovery_project,
            require_pair_complete=discovery_require_complete,
            cache_manifest_uri=teams_cache_manifest_uri,
            cache_max_age_hours=teams_cache_max_age_hours,
            cache_validate_listing=teams_cache_validate_listing,
            cache_revision=teams_cache_revision,
            logger=logger,
        )
        if teams_enabled and teams_policy_filter_enabled:
            raw_clean_teams_samples = _select_clean_teams_samples(
                raw_teams_samples,
                visomaster_policy,
                logger,
            )
        else:
            raw_clean_teams_samples = raw_teams_samples

        if teams_enabled and len(raw_clean_teams_samples) == 0:
            logger.warning(
                "Teams passthrough is ENABLED but 0 complete samples were discovered "
                "in gs://%s. The deeplive_teams_* family weights will have no effect. "
                "Check that the bucket exists, contains samples/, and manifests have "
                "pair_complete=true.",
                discovery_bucket,
            )
        if teams_enabled:
            teams_samples = create_unified_samples_from_teams(raw_clean_teams_samples, logger)
    else:
        logger.info("Teams passthrough dataset: DISABLED")

    if visomaster_hints_teams_enabled:
        raw_teams_hint_samples = _select_hint_teams_samples(
            raw_teams_samples,
            visomaster_policy,
            logger,
        )
        if len(raw_teams_hint_samples) == 0:
            logger.warning(
                "VisoMaster Teams hints are ENABLED but 0 retained Teams hint samples "
                "were selected from the April 17 policy packet."
            )
        visomaster_hint_teams_samples = create_unified_samples_from_visomaster_hints_teams(
            raw_teams_hint_samples,
            logger,
        )
    else:
        logger.info("VisoMaster Teams hints dataset: DISABLED")

    # ==========================================================================
    # Load Proper-Data Inventory Lanes (WT-F provisional contract)
    # ==========================================================================
    proper_data_samples: List[UnifiedPairedSample] = []
    proper_data_discovery_summary: Dict[str, Any] = {}

    if proper_data_enabled:
        from .proper_data import discover_proper_data_samples

        proper_inventory_uri = (
            proper_data_config.get('inventory_uri')
            or proper_data_config.get('inventory_path')
        )
        proper_manifest_uri = (
            proper_data_config.get('manifest_uri')
            or proper_data_config.get('manifest_path')
        )
        proper_include_lanes = proper_data_config.get('include_lanes')
        proper_max_samples_per_lane = proper_data_config.get('max_samples_per_lane')
        proper_max_samples_total = proper_data_config.get('max_samples_total')

        if not proper_inventory_uri:
            raise ValueError(
                "combined_paired.proper_data.enabled=true requires "
                "combined_paired.proper_data.inventory_path or inventory_uri."
            )

        logger.info("Loading proper-data inventory lanes:")
        logger.info(f"  - Inventory: {proper_inventory_uri}")
        logger.info(f"  - Manifest: {proper_manifest_uri or 'NONE'}")
        logger.info(f"  - Include lanes: {proper_include_lanes or 'ALL'}")
        logger.info(
            "  - Caps: per_lane=%s total=%s",
            proper_max_samples_per_lane if proper_max_samples_per_lane is not None else "NONE",
            proper_max_samples_total if proper_max_samples_total is not None else "NONE",
        )

        raw_proper_data_samples, proper_data_discovery_summary = discover_proper_data_samples(
            inventory_uri=str(proper_inventory_uri),
            manifest_uri=str(proper_manifest_uri) if proper_manifest_uri else None,
            include_lanes=proper_include_lanes,
            max_samples_per_lane=proper_max_samples_per_lane,
            max_samples_total=proper_max_samples_total,
            log=logger,
        )
        if len(raw_proper_data_samples) == 0:
            logger.warning(
                "Proper-data inventory is ENABLED but 0 paired samples were discovered "
                "from %s. Check the inventory path and lane filters.",
                proper_inventory_uri,
            )
        proper_data_samples = create_unified_samples_from_proper_data(
            raw_proper_data_samples,
            logger,
        )
    else:
        logger.info("Proper-data inventory lanes: DISABLED")

    # ==========================================================================
    # Load VisoMaster Teams-Enhanced merged dataset (resolver-driven)
    # ==========================================================================
    viso_teams_enhanced_config = combined_config.get('visomaster_teams_enhanced', {})
    viso_teams_enhanced_enabled = viso_teams_enhanced_config.get('enabled', False)
    visomaster_teams_enhanced_samples: List[UnifiedPairedSample] = []

    if viso_teams_enhanced_enabled:
        from .visomaster import discover_visomaster_teams_enhanced_samples

        resolver_manifest_uri = (
            viso_teams_enhanced_config.get('resolver_manifest_uri')
            or viso_teams_enhanced_config.get('resolver_manifest_path')
        )
        vte_bucket = viso_teams_enhanced_config.get(
            'enhanced_bucket', 'enhanced-visomaster-cropped'
        )
        vte_domains = viso_teams_enhanced_config.get('companion_domains')
        vte_enhancers = viso_teams_enhanced_config.get('enhancers')
        vte_statuses = viso_teams_enhanced_config.get('include_statuses')
        vte_require_all = bool(viso_teams_enhanced_config.get('require_all_expected_enhancers', False))
        vte_max_samples_total = viso_teams_enhanced_config.get('max_samples_total')
        vte_sampling_family = normalize_method_name(
            viso_teams_enhanced_config.get('sampling_family_key', 'visomaster_enhanced_fake')
        ) or 'visomaster_enhanced_fake'

        logger.info("Loading VisoMaster Teams-enhanced merged dataset:")
        logger.info(f"  - Resolver manifest: {resolver_manifest_uri}")
        logger.info(f"  - Enhanced bucket: {vte_bucket}")
        logger.info(f"  - Companion domains: {vte_domains or 'ALL'}")
        logger.info(f"  - Enhancers: {vte_enhancers or 'ALL'}")
        logger.info(f"  - Include statuses: {vte_statuses or ['teams_v2_companion', 'clean_companion_only']}")
        logger.info(f"  - Require all expected enhancers: {vte_require_all}")
        logger.info(f"  - Sampling family key: {vte_sampling_family}")

        raw_teams_enhanced_samples = discover_visomaster_teams_enhanced_samples(
            resolver_manifest_uri=resolver_manifest_uri,
            enhanced_bucket=vte_bucket,
            companion_domains=vte_domains,
            enhancers=vte_enhancers,
            include_statuses=vte_statuses,
            require_all_expected_enhancers=vte_require_all,
            max_samples_total=vte_max_samples_total,
            log=logger,
        )
        if len(raw_teams_enhanced_samples) == 0:
            logger.warning(
                "VisoMaster Teams-enhanced is ENABLED but 0 merged samples were loaded "
                "from %s. Check the resolver manifest path and filters.",
                resolver_manifest_uri,
            )
        visomaster_teams_enhanced_samples = create_unified_samples_from_visomaster_teams_enhanced(
            raw_teams_enhanced_samples,
            logger,
            sampling_family_key=vte_sampling_family,
        )
    else:
        logger.info("VisoMaster Teams-enhanced merged dataset: DISABLED")

    # ==========================================================================
    # Load VisoMaster Enhanced Dataset (post-hoc face enhancement)
    # ==========================================================================
    viso_enhanced_config = combined_config.get('visomaster_enhanced', {})
    viso_enhanced_enabled = viso_enhanced_config.get('enabled', False)
    visomaster_enhanced_samples: List[UnifiedPairedSample] = []

    if viso_enhanced_enabled:
        from .visomaster import discover_visomaster_enhanced_samples

        ve_bucket = viso_enhanced_config.get(
            'gcs_bucket', 'visomaster-enhanced-face-cropped'
        )
        ve_original_bucket = viso_enhanced_config.get(
            'original_bucket', 'live-deepfake-methods-real-and-fake-frames-cropped'
        )
        ve_gcs_project = viso_enhanced_config.get(
            'gcs_project', os.environ.get('GOOGLE_CLOUD_PROJECT', 'train-cvit2')
        )
        ve_exclude_tiers = viso_enhanced_config.get('exclude_tiers', ['ARTIFACT'])
        ve_enhancers = viso_enhanced_config.get('enhancers')  # None = all 8
        ve_cache_manifest_uri = (
            viso_enhanced_config.get('cache_manifest_uri')
            or viso_enhanced_config.get('cache_manifest_path')
        )
        ve_cache_max_age_hours = float(viso_enhanced_config.get('cache_max_age_hours') or 12.0)

        logger.info("Loading VisoMaster Enhanced dataset:")
        logger.info(f"  - Enhanced GCS bucket: {ve_bucket}")
        logger.info(f"  - Original GCS bucket: {ve_original_bucket}")
        logger.info(f"  - Exclude tiers: {ve_exclude_tiers or 'NONE'}")
        logger.info(f"  - Enhancers: {ve_enhancers or 'ALL'}")
        logger.info(
            "  - Discovery cache: %s (max_age_hours=%s)",
            ve_cache_manifest_uri or "DISABLED",
            ve_cache_max_age_hours,
        )

        raw_enhanced_samples = discover_visomaster_enhanced_samples(
            enhanced_bucket=ve_bucket,
            original_bucket=ve_original_bucket,
            gcs_project=ve_gcs_project,
            exclude_tiers=ve_exclude_tiers,
            enhancers=ve_enhancers,
            cache_manifest_uri=ve_cache_manifest_uri,
            cache_max_age_hours=ve_cache_max_age_hours,
            log=logger,
        )

        if len(raw_enhanced_samples) == 0:
            logger.warning(
                "VisoMaster Enhanced is ENABLED but 0 samples were discovered "
                "in gs://%s. The visomaster_enhanced_fake family weight will "
                "have no effect.",
                ve_bucket,
            )
        visomaster_enhanced_samples = create_unified_samples_from_visomaster_enhanced(
            raw_enhanced_samples, logger
        )
    else:
        logger.info("VisoMaster Enhanced dataset: DISABLED")

    # ==========================================================================
    # Load VisoMaster Resolution-Variant Dataset (Inswapper128 res sweep)
    # ==========================================================================
    viso_rv_config = combined_config.get('visomaster_res_variant', {})
    viso_rv_enabled = viso_rv_config.get('enabled', False)
    visomaster_res_variant_samples: List[UnifiedPairedSample] = []

    if viso_rv_enabled:
        from .visomaster import discover_visomaster_res_variant_samples

        rv_bucket = viso_rv_config.get(
            'gcs_bucket', 'visomaster-enhanced-face-cropped'
        )
        rv_original_bucket = viso_rv_config.get(
            'original_bucket', 'live-deepfake-methods-real-and-fake-frames-cropped'
        )
        rv_gcs_project = viso_rv_config.get(
            'gcs_project', os.environ.get('GOOGLE_CLOUD_PROJECT', 'train-cvit2')
        )
        rv_exclude_tiers = viso_rv_config.get('exclude_tiers', ['ARTIFACT'])
        rv_resolutions = viso_rv_config.get('resolutions')  # None = all resolutions
        rv_cache_manifest_uri = (
            viso_rv_config.get('cache_manifest_uri')
            or viso_rv_config.get('cache_manifest_path')
        )
        rv_cache_max_age_hours = float(viso_rv_config.get('cache_max_age_hours') or 12.0)

        logger.info("Loading VisoMaster Resolution-Variant dataset:")
        logger.info(f"  - GCS bucket: {rv_bucket}")
        logger.info(f"  - Original GCS bucket: {rv_original_bucket}")
        logger.info(f"  - Exclude tiers: {rv_exclude_tiers or 'NONE'}")
        logger.info(f"  - Resolutions: {rv_resolutions or 'ALL'}")
        logger.info(
            "  - Discovery cache: %s (max_age_hours=%s)",
            rv_cache_manifest_uri or "DISABLED",
            rv_cache_max_age_hours,
        )

        raw_rv_samples = discover_visomaster_res_variant_samples(
            bucket_name=rv_bucket,
            original_bucket=rv_original_bucket,
            gcs_project=rv_gcs_project,
            exclude_tiers=rv_exclude_tiers,
            resolutions=rv_resolutions,
            cache_manifest_uri=rv_cache_manifest_uri,
            cache_max_age_hours=rv_cache_max_age_hours,
            log=logger,
        )

        if len(raw_rv_samples) == 0:
            logger.warning(
                "VisoMaster Resolution-Variant is ENABLED but 0 samples were "
                "discovered in gs://%s. The visomaster_res_variant_fake family "
                "weight will have no effect.",
                rv_bucket,
            )
        visomaster_res_variant_samples = create_unified_samples_from_visomaster_res_variant(
            raw_rv_samples, logger
        )
    else:
        logger.info("VisoMaster Resolution-Variant dataset: DISABLED")

    # ==========================================================================
    # Discover External Training Reals (VCD, webcam, etc.)
    # ==========================================================================
    external_real_samples, external_training_identities = _discover_external_training_reals(
        combined_config, logger
    )
    if external_real_samples:
        logger.info(
            f"Adding {len(external_real_samples)} unpaired real samples "
            f"({sum(len(s.frame_paths) for s in external_real_samples)} frames) to training pool"
        )
    
    # ==========================================================================
    # Combine Samples
    # ==========================================================================
    all_samples = (
        df40_samples + deeplive_samples + visomaster_samples + visomaster_hint_samples
        + proper_data_samples
        + visomaster_teams_enhanced_samples + visomaster_enhanced_samples
        + visomaster_res_variant_samples
        + teams_samples + visomaster_hint_teams_samples + external_real_samples
    )
    
    if len(all_samples) == 0:
        raise ValueError("No samples found! Enable at least one data source.")
    
    logger.info(f"Combined dataset: {len(all_samples)} total samples")
    logger.info(f"  - DF40: {len(df40_samples)}")
    logger.info(f"  - DeepLive: {len(deeplive_samples)}")
    logger.info(f"  - VisoMaster: {len(visomaster_samples)}")
    logger.info(f"  - VisoMaster hints: {len(visomaster_hint_samples)}")
    logger.info(f"  - Proper-data: {len(proper_data_samples)}")
    logger.info(f"  - VisoMaster Teams-Enhanced: {len(visomaster_teams_enhanced_samples)}")
    logger.info(f"  - VisoMaster Enhanced: {len(visomaster_enhanced_samples)}")
    logger.info(f"  - VisoMaster Res-Variant: {len(visomaster_res_variant_samples)}")
    logger.info(f"  - Teams passthrough: {len(teams_samples)}")
    logger.info(f"  - VisoMaster Teams hints: {len(visomaster_hint_teams_samples)}")
    logger.info(f"  - External reals: {len(external_real_samples)}")
    
    # Build method mapping early (needed by dataset for Group DRO method_id)
    unique_methods = set()
    for sample in all_samples:
        unique_methods.update(_sample_possible_methods(sample))
    sorted_methods = sorted(unique_methods)
    method_mapping = {m: i for i, m in enumerate(sorted_methods)}
    logger.info(f"Method mapping ({len(method_mapping)} methods): {method_mapping}")

    # PE_PAIR_RANK_DRO group_id mapping (added 2026-05-07) — asymmetric
    # R-D / F-B key per analysis/group_id_design_audit_2026-05-06. Walks
    # all_samples × {label=0, label=1} (paired) or {label=0} (unpaired_real).
    # Recommended scheme yields ~27 groups; cardinality varies with the data
    # plane (df40 methods + visomaster swap_models + deeplive strategies).
    group_id_mapping = build_group_id_mapping_for_samples(all_samples)
    logger.info(
        f"Group ID mapping ({len(group_id_mapping)} groups for PE_PAIR_RANK_DRO; "
        f"R-D real-side + F-B fake-side keys)"
    )


    # ==========================================================================
    # Identity-Stratified Split
    # ==========================================================================
    train_split = combined_config.get('train_split', 0.8)
    val_split = combined_config.get('val_split', 0.1)
    holdout_cfg = combined_config.get("holdout", {}) or {}
    holdout_mode = normalize_method_name(holdout_cfg.get("mode", "identity"))

    if holdout_mode in {"method", "method_holdout", "by_method"}:
        holdout_methods = _resolve_holdout_methods(
            holdout_cfg=holdout_cfg,
            df40_cfg=df40_config,
            logger=logger,
        )
        holdout_max_per_method = holdout_cfg.get("max_samples_per_method")
        holdout_per_method_caps = holdout_cfg.get("per_method_caps") or {}
        train_samples, val_samples, test_samples = split_samples_by_method_holdout(
            all_samples,
            holdout_methods=holdout_methods,
            train_split=train_split,
            val_split=val_split,
            seed=split_seed,
            logger=logger,
            split_mode=identity_split_mode,
            max_samples_per_method=holdout_max_per_method,
            per_method_caps=holdout_per_method_caps,
        )
        resolved_holdout_methods = sorted(
            {normalize_method_name(m) for m in holdout_methods if m}
        )
        logger.info(
            "Holdout mode: method_holdout (methods=%s)",
            resolved_holdout_methods,
        )
        df40_train_methods = sorted(
            {normalize_method_name(s.method) for s in train_samples if s.source == "df40"}
        )
        df40_holdout_methods = sorted(
            {normalize_method_name(s.method) for s in test_samples if s.source == "df40"}
        )
        if df40_train_methods or df40_holdout_methods:
            logger.info(
                "DF40 method placement after holdout split: train=%s holdout=%s",
                df40_train_methods,
                df40_holdout_methods,
            )
    else:
        train_samples, val_samples, test_samples = split_samples_by_identity(
            all_samples, train_split, val_split, split_seed, logger,
            split_mode=identity_split_mode,
        )
        resolved_holdout_methods = []
        holdout_mode = "identity"
    
    # ==========================================================================
    # Create Augmentation Transform
    # ==========================================================================
    transform = kwargs.get('transform')
    if transform is None:
        transform = _create_combined_transform(config, combined_config, logger, data_config)
    
    # ==========================================================================
    # Create Batching Config
    # ==========================================================================
    no_multiprocessing = os.environ.get('NO_MULTIPROCESSING', '').lower() in ('1', 'true', 'yes')
    device_is_cpu = not torch.cuda.is_available()
    
    num_workers = 0 if (no_multiprocessing or device_is_cpu) else config.get(
        'num_workers',
        combined_config.get('num_workers', 4),
    )
    
    sampling_cfg = combined_config.get("sampling", {}) or {}
    sampling_strategy = normalize_method_name(
        sampling_cfg.get(
            "strategy",
            "identity_resample_uniform" if combined_config.get('identity_balanced_sampling', True) else "none",
        )
    )
    family_weights_cfg = sampling_cfg.get("family_weights", {}) or {}
    family_weights = {
        normalize_method_name(k): float(v)
        for k, v in family_weights_cfg.items()
    }

    # T3 SLOT 1/2/3 IQ-shortcut packets (2026-05-09): optional Teams REAL frame keep-list.
    # When `combined_paired.teams.frame_keep_list_path` is set, we load the CSV and pass
    # a frozenset of `frame_uri` values to the iterable dataset. Real frames whose URI is
    # NOT in the set are skipped during _iterate_teams_sample. Default behavior (no
    # config field) is identical to the legacy code path.
    teams_real_frame_keep_list_set: Optional[frozenset] = None
    teams_keep_list_path = teams_config.get('frame_keep_list_path') if teams_enabled else None
    if teams_keep_list_path:
        if not os.path.isabs(teams_keep_list_path):
            training_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            teams_keep_list_path = os.path.join(training_dir, teams_keep_list_path)
        if not os.path.exists(teams_keep_list_path):
            raise FileNotFoundError(
                f"combined_paired.teams.frame_keep_list_path={teams_keep_list_path} "
                "is set but the file does not exist."
            )
        import csv as _keep_csv
        keep_uris: List[str] = []
        with open(teams_keep_list_path, "r", newline="") as _kfh:
            _reader = _keep_csv.DictReader(_kfh)
            if _reader.fieldnames is None or "frame_uri" not in _reader.fieldnames:
                raise ValueError(
                    f"combined_paired.teams.frame_keep_list_path={teams_keep_list_path} "
                    "is missing a `frame_uri` header column."
                )
            for _row in _reader:
                u = (_row.get("frame_uri") or "").strip()
                if u:
                    keep_uris.append(u)
        teams_real_frame_keep_list_set = frozenset(keep_uris)
        logger.info(
            "Teams REAL frame keep-list loaded: %d URIs from %s",
            len(teams_real_frame_keep_list_set),
            teams_keep_list_path,
        )

    # T3 SLOT 3 (2026-05-09): optional per-method keep-list. CSV columns
    # required: method, frame_uri (lap_var optional/ignored). The map is
    # method -> frozenset(uri).
    teams_real_frame_keep_list_per_method_map: Optional[Dict[str, frozenset]] = None
    teams_keep_list_per_method_path = (
        teams_config.get('frame_keep_list_per_method') if teams_enabled else None
    )
    if teams_keep_list_per_method_path:
        if not os.path.isabs(teams_keep_list_per_method_path):
            training_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            teams_keep_list_per_method_path = os.path.join(training_dir, teams_keep_list_per_method_path)
        if not os.path.exists(teams_keep_list_per_method_path):
            raise FileNotFoundError(
                f"combined_paired.teams.frame_keep_list_per_method="
                f"{teams_keep_list_per_method_path} is set but the file does not exist."
            )
        import csv as _keep_csv
        per_method_uris: Dict[str, List[str]] = {}
        with open(teams_keep_list_per_method_path, "r", newline="") as _kfh:
            _reader = _keep_csv.DictReader(_kfh)
            if (
                _reader.fieldnames is None
                or "method" not in _reader.fieldnames
                or "frame_uri" not in _reader.fieldnames
            ):
                raise ValueError(
                    f"combined_paired.teams.frame_keep_list_per_method="
                    f"{teams_keep_list_per_method_path} is missing the required "
                    "`method` and `frame_uri` header columns."
                )
            for _row in _reader:
                m = (_row.get("method") or "").strip()
                u = (_row.get("frame_uri") or "").strip()
                if not m or not u:
                    continue
                per_method_uris.setdefault(m, []).append(u)
        teams_real_frame_keep_list_per_method_map = {
            m: frozenset(uris) for m, uris in per_method_uris.items()
        }
        total = sum(len(v) for v in teams_real_frame_keep_list_per_method_map.values())
        logger.info(
            "Teams REAL per-method keep-list loaded: %d methods, %d URIs total from %s",
            len(teams_real_frame_keep_list_per_method_map),
            total,
            teams_keep_list_per_method_path,
        )

    batching_config = CombinedBatchingConfig(
        batch_size=config.get('frames_per_batch', combined_config.get('frames_per_batch', 32)),
        num_workers=num_workers,
        prefetch_factor=config.get('prefetch_factor', 2) if num_workers > 0 else 2,
        visomaster_parallel_download_workers=combined_config.get(
            'visomaster_parallel_download_workers',
            4,
        ),
        teams_parallel_download_workers=combined_config.get(
            'teams_parallel_download_workers',
            DEFAULT_PARALLEL_GCS_DOWNLOAD_WORKERS,
        ),
        identity_balanced_sampling=combined_config.get('identity_balanced_sampling', True),
        identity_sampling_strategy=sampling_strategy,
        identity_family_weights=family_weights,
        enhanced_strategy_names=enhanced_strategy_names,
        frames_per_sample=config.get('frames_per_video', 8),
        df40_sparse_indices=combined_config.get('df40', {}).get('anchor_indices', [0, 4, 8, 12, 16, 20, 24, 28]),
        deeplive_sparse_indices=combined_config.get('deeplive', {}).get('anchor_indices', [0, 2, 4, 6, 8, 10, 12, 14]),
        visomaster_sparse_indices=combined_config.get('visomaster', {}).get('anchor_indices', [0, 2, 4, 6, 8, 10, 12, 14]),
        visomaster_enhanced_sparse_indices=combined_config.get('visomaster_enhanced', {}).get('anchor_indices', [0, 2, 4, 6, 8, 10, 12, 14]),
        visomaster_teams_enhanced_sparse_indices=combined_config.get('visomaster_teams_enhanced', {}).get('anchor_indices', [0, 2, 4, 6, 8, 10, 12, 14]),
        visomaster_teams_enhanced_p_original=float(combined_config.get('visomaster_teams_enhanced', {}).get('p_original', 0.5)),
        proper_data_sparse_indices=combined_config.get('proper_data', {}).get('anchor_indices', [0, 2, 4, 6, 8, 10, 12, 14]),
        proper_data_parallel_download_workers=combined_config.get(
            'proper_data_parallel_download_workers',
            DEFAULT_PARALLEL_GCS_DOWNLOAD_WORKERS,
        ),
        teams_sparse_indices=combined_config.get('teams', {}).get('anchor_indices', [0, 2, 4, 6, 8, 10, 12, 14]),
        teams_real_frame_keep_list=teams_real_frame_keep_list_set,
        teams_real_frame_keep_list_per_method=teams_real_frame_keep_list_per_method_map,
    )

    # VisoMaster anchor indices for the iterable dataset
    visomaster_anchor_indices = combined_config.get('visomaster', {}).get(
        'anchor_indices', [0, 2, 4, 6, 8, 10, 12, 14]
    )
    
    # ==========================================================================
    # Create Dataloaders
    # ==========================================================================
    train_iterable = CombinedPairedIterableDataset(
        samples=train_samples,
        df40_dataset=df40_dataset,
        deeplive_dataset=deeplive_dataset,
        config=batching_config,
        transform=transform,
        shuffle=True,
        seed=run_seed,
        visomaster_anchor_indices=visomaster_anchor_indices if visomaster_enabled else None,
        method_mapping=method_mapping,
        face_area_parquet_path=combined_config.get('face_area_parquet_path'),
    )
    persistent_workers = batching_config.num_workers > 0
    
    train_loader = DataLoader(
        train_iterable,
        batch_size=batching_config.batch_size,
        num_workers=batching_config.num_workers,
        prefetch_factor=batching_config.prefetch_factor if batching_config.num_workers > 0 else None,
        collate_fn=combined_paired_collate_fn,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    
    # Validation loaders (no augmentation, no shuffle)
    val_iterable = CombinedPairedIterableDataset(
        samples=val_samples,
        df40_dataset=df40_dataset,
        deeplive_dataset=deeplive_dataset,
        config=batching_config,
        transform=None,
        shuffle=False,
        seed=run_seed,
        visomaster_anchor_indices=visomaster_anchor_indices if visomaster_enabled else None,
        method_mapping=method_mapping,
        face_area_parquet_path=combined_config.get('face_area_parquet_path'),
    )
    
    val_loader_raw = DataLoader(
        val_iterable,
        batch_size=batching_config.batch_size,
        num_workers=batching_config.num_workers,
        prefetch_factor=batching_config.prefetch_factor if batching_config.num_workers > 0 else None,
        collate_fn=combined_paired_collate_fn,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    
    test_iterable = CombinedPairedIterableDataset(
        samples=test_samples,
        df40_dataset=df40_dataset,
        deeplive_dataset=deeplive_dataset,
        config=batching_config,
        transform=None,
        shuffle=False,
        seed=run_seed,
        visomaster_anchor_indices=visomaster_anchor_indices if visomaster_enabled else None,
        method_mapping=method_mapping,
        face_area_parquet_path=combined_config.get('face_area_parquet_path'),
    )
    
    test_loader_raw = DataLoader(
        test_iterable,
        batch_size=batching_config.batch_size,
        num_workers=batching_config.num_workers,
        prefetch_factor=batching_config.prefetch_factor if batching_config.num_workers > 0 else None,
        collate_fn=combined_paired_collate_fn,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    
    # Wrap with adapters
    val_loader = CombinedValidationAdapter(
        val_loader_raw,
        val_samples,
        'combined_val',
        method_mapping=method_mapping,
    )
    test_loader = CombinedValidationAdapter(
        test_loader_raw,
        test_samples,
        'combined_holdout',
        method_mapping=method_mapping,
    )

    # Optional external OOD monitoring loader (e.g., WMA + external real).
    # Exclude VCD identities already used for training to avoid data leakage.
    ood_cfg = combined_config.get("ood_monitoring", {}) or {}
    exclude_ood_identities = (
        external_training_identities
        if ood_cfg.get("exclude_training_identities", True)
        else None
    )
    build_ood_loader_at_startup = bool(ood_cfg.get("build_loader_at_startup", True))
    # A10: held-out 10% partition (blake2b hash on video_id). Off by default so
    # pre-A10 configs keep their full monitored pool.
    ood_heldout_fraction = float(ood_cfg.get("heldout_fraction", 0.0) or 0.0)
    ood_hash_rule = str(ood_cfg.get("heldout_hash_rule", "blake2b_lo10"))
    ood_loader = None
    ood_heldout_loader = None
    ood_videos: List[Any] = []
    ood_monitored_videos: List[Any] = []
    ood_heldout_videos: List[Any] = []
    if build_ood_loader_at_startup:
        ood_videos = _build_external_ood_videos(
            combined_config=combined_config,
            frames_per_video=batching_config.frames_per_sample,
            logger=logger,
            exclude_identities=exclude_ood_identities,
        )
        if ood_videos:
            from dataset.dataloaders import create_ood_loader

            if ood_heldout_fraction > 0.0:
                ood_monitored_videos, ood_heldout_videos = _partition_ood_videos_heldout(
                    videos=ood_videos,
                    heldout_fraction=ood_heldout_fraction,
                    logger=logger,
                    hash_rule=ood_hash_rule,
                )
            else:
                ood_monitored_videos = list(ood_videos)
                ood_heldout_videos = []

            ood_loader = create_ood_loader(ood_monitored_videos, config, data_config)
            logger.info(
                "Created OOD monitoring loader from external sources: "
                "videos=%d methods=%d (monitored partition)",
                len(ood_monitored_videos),
                len({v.method for v in ood_monitored_videos}) if ood_monitored_videos else 0,
            )
            if ood_heldout_videos:
                ood_heldout_loader = create_ood_loader(
                    ood_heldout_videos, config, data_config
                )
                logger.info(
                    "Created OOD held-out loader (A10): videos=%d methods=%d "
                    "(never seen during training-time OOD eval; consumed only by final_eval)",
                    len(ood_heldout_videos),
                    len({v.method for v in ood_heldout_videos}),
                )
    else:
        logger.info(
            "Skipping OOD loader build at startup (ood_monitoring.build_loader_at_startup=false)."
        )
    
    # ==========================================================================
    # Compute Statistics
    # ==========================================================================
    unique_identities = set(s.identity for s in all_samples)
    train_identities = set(s.identity for s in train_samples)
    overall_counts = _count_strategy_and_family(all_samples, enhanced_strategy_names)
    train_counts = _count_strategy_and_family(train_samples, enhanced_strategy_names)
    overview_methods = _build_overview_method_lists(all_samples)
    source_counts = Counter(s.source for s in all_samples)
    train_source_counts = Counter(s.source for s in train_samples)
    proper_data_lane_counts = {
        key: int(value)
        for key, value in sorted(source_counts.items())
        if str(key).startswith('proper_')
    }
    train_proper_data_lane_counts = {
        key: int(value)
        for key, value in sorted(train_source_counts.items())
        if str(key).startswith('proper_')
    }

    # Estimate frames per sample (DF40 has 8 frames, DeepLive has sparse indices)
    frames_per_sample = batching_config.frames_per_sample * 2  # real + fake

    # method_mapping was already computed above (before dataset creation)
    
    data_stats = {
        'total_samples': len(all_samples),
        'df40_samples': len(df40_samples),
        'deeplive_samples': len(deeplive_samples),
        'visomaster_samples': len(visomaster_samples),
        'visomaster_hints_samples': len(visomaster_hint_samples),
        'proper_data_samples': len(proper_data_samples),
        'proper_visomaster_clean_samples': int(source_counts.get('proper_visomaster_clean', 0)),
        'proper_visomaster_enhanced_clean_samples': int(source_counts.get('proper_visomaster_enhanced_clean', 0)),
        'proper_visomaster_teams_samples': int(source_counts.get('proper_visomaster_teams', 0)),
        'proper_visomaster_enhanced_teams_samples': int(source_counts.get('proper_visomaster_enhanced_teams', 0)),
        'visomaster_teams_enhanced_samples': len(visomaster_teams_enhanced_samples),
        'visomaster_enhanced_samples': len(visomaster_enhanced_samples),
        'teams_samples': len(teams_samples),
        'visomaster_hints_teams_samples': len(visomaster_hint_teams_samples),
        'external_real_samples': len(external_real_samples),
        'external_training_identities': len(external_training_identities),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'total_identities': len(unique_identities),
        'train_identities': len(train_identities),
        'identity_balanced_sampling': batching_config.identity_balanced_sampling,
        'has_landmarks': deeplive_enabled,
        'run_seed': run_seed,
        'split_seed': split_seed,
        'identity_split_mode': identity_split_mode,
        'deeplive_raw_strategy_counts': deeplive_strategy_counts['raw'],
        'deeplive_effective_strategy_counts': deeplive_strategy_counts['effective'],
        'deeplive_strategy_preflight': deeplive_preflight,
        # Required by train_sweep.py for run overview logging
        'discovered_videos': len(all_samples),  # Each sample is a video pair
        'discovered_methods': len(unique_methods),  # Number of unique methods
        'unbalanced_train_count': len(train_samples),
        'unbalanced_val_count': len(val_samples) + len(test_samples),
        'train_video_count': len(train_samples),
        'train_frame_count': len(train_samples) * frames_per_sample,
        'val_video_count': len(val_samples),
        'val_frame_count': len(val_samples) * frames_per_sample,
        'train_split': train_split,
        'methods': list(unique_methods),
        'strategy_counts': overall_counts['strategy_counts'],
        'family_counts': overall_counts['family_counts'],
        'source_counts': dict(sorted(source_counts.items())),
        'train_strategy_counts': train_counts['strategy_counts'],
        'train_family_counts': train_counts['family_counts'],
        'train_source_counts': dict(sorted(train_source_counts.items())),
        'enhanced_strategy_names': list(enhanced_strategy_names),
        'sampling_strategy': sampling_strategy,
        'sampling_family_weights': family_weights,
        'bad_data_policy_active': bool(visomaster_policy),
        'bad_data_policy_manifest': str(getattr(visomaster_policy, 'manifest_path', '')) if visomaster_policy else '',
        'bad_data_policy_summary': str(getattr(visomaster_policy, 'summary_path', '')) if visomaster_policy and getattr(visomaster_policy, 'summary_path', None) else '',
        'teams_apply_bad_data_policy': teams_policy_filter_enabled,
        'holdout_mode': holdout_mode,
        'holdout_methods': resolved_holdout_methods,
        'holdout_method_counts': dict(
            sorted(Counter(normalize_method_name(s.method) for s in test_samples).items())
        ),
        'overview_real_methods': overview_methods['real_methods'],
        'overview_fake_methods': overview_methods['fake_methods'],
        'train_df40_methods': sorted(
            {normalize_method_name(s.method) for s in train_samples if s.source == 'df40'}
        ),
        'holdout_df40_methods': sorted(
            {normalize_method_name(s.method) for s in test_samples if s.source == 'df40'}
        ),
        'ood_video_count': len(ood_videos),
        'ood_method_count': len({v.method for v in ood_videos}) if ood_videos else 0,
        'ood_monitored_video_count': len(ood_monitored_videos),
        'ood_heldout_video_count': len(ood_heldout_videos),
        'ood_heldout_fraction_target': ood_heldout_fraction,
        'ood_heldout_hash_rule': ood_hash_rule,
        'proper_data_discovery': proper_data_discovery_summary,
        'proper_data_build_id': str(proper_data_discovery_summary.get('wave_id', '') or ''),
        'proper_data_lane_counts': proper_data_lane_counts,
        'train_proper_data_lane_counts': train_proper_data_lane_counts,
        # Group DRO method mapping (method_name → int ID)
        'method_mapping': method_mapping,
        # PE_PAIR_RANK_DRO multi-axis GroupDRO mapping (asymmetric R-D / F-B
        # key string → int ID). Consumed by trainer/mixins/group_dro.py via
        # train_sweep.py wiring (added 2026-05-07).
        'group_id_mapping': group_id_mapping,
    }
    
    logger.info("=" * 70)
    logger.info("Combined Paired pipeline created successfully")
    logger.info(f"  - Train: {len(train_samples)} samples, {len(train_identities)} identities")
    logger.info(f"  - Val: {len(val_samples)} samples")
    logger.info(f"  - Test: {len(test_samples)} samples")
    logger.info("=" * 70)
    
    return DataPipelineResult(
        train_loader=train_loader,
        val_in_dist_loader=val_loader,
        val_holdout_loader=test_loader,
        train_samples=train_samples,
        data_stats=data_stats,
        ood_loader=ood_loader,
        test_loader=test_loader,
        ood_heldout_loader=ood_heldout_loader,
    )


def _create_combined_transform(
    config: Dict[str, Any],
    combined_config: Dict[str, Any],
    logger: logging.Logger,
    data_config: Optional[Dict[str, Any]] = None
) -> Optional[Callable]:
    """
    Create augmentation transform for combined dataset.
    
    This transform handles both:
    - DF40 samples (landmarks=None)
    - DeepLive samples (landmarks available)
    
    Supported augmentation versions:
    - 'base_only' (default): Mild color + quality augmentations
    - 'quality_robust': Breaks quality shortcuts (degrade + enhance + downscale)
      Accepts 'strength' sub-key: 'light', 'moderate', 'strong'
    - 'quality_targeted_family': Family-aware router with metadata-based routing
      Accepts:
        - 'strength': 'light' | 'moderate' (or 'strong' for compatibility)
        - 'routing.mode': 'family_aware'
        - 'routing.enhanced_strategy_names': list of enhanced DeepLive strategies
    - 'quality_robust_light' / 'quality_robust_moderate' / 'quality_robust_strong':
      Direct strength aliases (equivalent to quality_robust + strength key)
    - Integer or other string versions: Dispatched to the augmentation registry
    """
    aug_config = config.get('augmentation') or (data_config or {}).get('augmentation') or {}
    
    if not aug_config:
        logger.info("No augmentation config - using base augmentations only")
    
    aug_version = aug_config.get('version', 'base_only')
    
    try:
        import albumentations as A
    except ImportError:
        logger.warning("albumentations not installed - skipping augmentations")
        return None
    
    # ------------------------------------------------------------------
    # Dispatch to quality_robust pipeline
    # ------------------------------------------------------------------
    if isinstance(aug_version, str) and aug_version.startswith('quality_robust'):
        from data.augmentations.pipelines import create_quality_robust_pipeline
        
        # Determine strength: explicit suffix, or 'strength' sub-key, or default
        if aug_version == 'quality_robust_light':
            strength = 'light'
        elif aug_version == 'quality_robust_strong':
            strength = 'strong'
        elif aug_version == 'quality_robust_moderate':
            strength = 'moderate'
        else:
            # 'quality_robust' — look for strength in config
            strength = aug_config.get('strength', 'moderate')
        
        pipeline = create_quality_robust_pipeline(strength=strength)
        logger.info(f"Created quality_robust augmentation (strength={strength})")
        
        def transform_fn(image, landmarks=None, meta=None):
            import numpy as np
            if isinstance(image, np.ndarray):
                return pipeline(image=image)['image']
            return image

        return transform_fn

    # ------------------------------------------------------------------
    # Dispatch to quality_targeted_family router
    # ------------------------------------------------------------------
    if aug_version == 'quality_targeted_family':
        from data.augmentations.pipelines import create_quality_targeted_family_router

        routing_cfg = aug_config.get('routing', {}) or {}
        strength = aug_config.get('strength', 'moderate')
        routing_mode = routing_cfg.get('mode', 'family_aware')
        enhanced_strategy_names = tuple(
            routing_cfg.get('enhanced_strategy_names') or DEFAULT_ENHANCED_STRATEGIES
        )

        # Collect preset-level overrides from the YAML augmentation block.
        # Any key in aug_config that is also a valid preset key (e.g.
        # context_variation_enabled, context_variation_brightness, …)
        # will override the hardcoded preset value for that strength.
        _NON_PRESET_KEYS = {'version', 'strength', 'routing'}
        from data.augmentations.pipelines import _QUALITY_TARGETED_PRESETS
        _valid_preset_keys = set(next(iter(_QUALITY_TARGETED_PRESETS.values())).keys())
        preset_overrides = {
            k: v for k, v in aug_config.items()
            if k not in _NON_PRESET_KEYS and k in _valid_preset_keys
        }

        # ── Teams codec simulation config (optional) ─────────────────
        teams_codec_cfg = aug_config.get('teams_codec_simulation', None)

        # ── Pipeline-randomization config (anti-shortcut, label-symmetric) ──
        pipeline_random_cfg = aug_config.get('pipeline_randomization', None)

        router = create_quality_targeted_family_router(
            strength=strength,
            routing_mode=routing_mode,
            enhanced_strategy_names=enhanced_strategy_names,
            preset_overrides=preset_overrides or None,
            teams_codec_simulation=teams_codec_cfg,
            pipeline_randomization=pipeline_random_cfg,
        )
        logger.info(
            "Created quality_targeted_family augmentation "
            f"(strength={strength}, routing_mode={routing_mode}, "
            f"enhanced_strategies={list(enhanced_strategy_names)})"
        )
        if preset_overrides:
            logger.info(f"  Preset overrides from YAML: {preset_overrides}")
        if teams_codec_cfg and teams_codec_cfg.get('enabled'):
            logger.info(
                f"  TeamsCodecSimulation: enabled, p={teams_codec_cfg.get('probability', 0.15)}, "
                f"exclude={teams_codec_cfg.get('exclude_families', [])}"
            )

        def transform_fn(image, landmarks=None, meta=None):
            import numpy as np
            if isinstance(image, np.ndarray):
                return router(image=image, landmarks=landmarks, meta=meta)
            return image

        return transform_fn

    # ------------------------------------------------------------------
    # Dispatch to augmentation registry (V3-V7, surgical, etc.)
    # ------------------------------------------------------------------
    if aug_version != 'base_only':
        try:
            from data.augmentations import get_pipeline
            pipeline = get_pipeline(version=aug_version)
            logger.info(f"Created augmentation from registry: version={aug_version}")
            
            # Registry pipelines are either A.Compose or callables
            if callable(pipeline) and not isinstance(pipeline, A.Compose):
                # V6/V7 style — callable(img_np, ...) -> img_np
                def transform_fn(image, landmarks=None, meta=None):
                    import numpy as np
                    if isinstance(image, np.ndarray):
                        return pipeline(image)
                    return image
            else:
                # A.Compose style
                def transform_fn(image, landmarks=None, meta=None):
                    import numpy as np
                    if isinstance(image, np.ndarray):
                        return pipeline(image=image)['image']
                    return image
            
            return transform_fn
        except (ValueError, KeyError) as e:
            logger.warning(
                f"Augmentation version '{aug_version}' not in registry ({e}). "
                f"Falling back to base_only."
            )
    
    # ------------------------------------------------------------------
    # Default: base_only pipeline (original behavior)
    # ------------------------------------------------------------------
    base_config = aug_config.get('base', {})
    transforms = []
    
    if base_config.get('horizontal_flip', True):
        transforms.append(A.HorizontalFlip(p=0.5))
    
    if base_config.get('color_augmentations', True):
        transforms.append(A.OneOf([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        ], p=0.5))
    
    if base_config.get('quality_augmentations', True):
        transforms.append(A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.ImageCompression(quality_lower=70, quality_upper=95, p=1.0),
            A.GaussNoise(var_limit=(5, 25), p=1.0),
        ], p=0.3))
    
    if len(transforms) == 0:
        return None
    
    pipeline = A.Compose(transforms)
    
    def transform_fn(image, landmarks=None, meta=None):
        """Apply augmentations. Handles both DF40 (no landmarks) and DeepLive."""
        import numpy as np
        if isinstance(image, np.ndarray):
            # TODO: Add landmark-based occlusion for DeepLive samples when landmarks is not None
            augmented = pipeline(image=image)
            return augmented['image']
        return image
    
    logger.info(f"Created base_only augmentation transform with {len(transforms)} augmentations")
    return transform_fn
