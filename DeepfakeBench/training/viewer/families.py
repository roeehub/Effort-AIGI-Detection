"""
Family and group key inference — standalone port of utils/grouping.py.

This is a self-contained reimplementation that avoids importing from the
training pipeline (which pulls in torch/CLIP dependencies).
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union

from .visomaster_policy import (
    BASELINE_FAKE_FAMILY,
    BASELINE_METHOD,
    BASELINE_SOURCE,
    TEAMS_FAKE_FAMILY,
    TEAMS_METHOD,
    TEAMS_SOURCE,
)

DEFAULT_ENHANCED_STRATEGIES = (
    "quality_enhancement",
    "edge_cases_enhanced",
    "minimal_processing_enhanced",
)

DF40_REAL_METHOD_HINTS = {
    "faceforensics++", "faceforensicspp", "ff++", "ffpp", "df40_real",
}

DF40_FAKE_METHOD_HINTS = {
    "blendface", "facedancer", "facevid2vid", "faceswap", "fomm",
    "fsgan", "inswap", "lia", "mraa", "mcnet", "mobileswap",
    "one_shot_free", "pirender", "sadtalker", "simswap", "tpsm",
    "uniface", "wav2lip", "danet", "e4s", "ff_df", "ff_f2f",
    "ff_fs", "ff_nt",
}


def normalize_method_name(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_label_value(label) -> Optional[int]:
    if label is None:
        return None
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, float)):
        return 1 if int(label) == 1 else 0
    label_norm = str(label).strip().lower()
    if label_norm in {"1", "fake", "f"}:
        return 1
    if label_norm in {"0", "real", "r"}:
        return 0
    return None


def _normalize_strategy_set(names):
    if not names:
        return set()
    return {normalize_method_name(n) for n in names if n is not None}


def _extract_deeplive_strategy(method_norm: str, source_norm: str):
    if method_norm.startswith("deeplive_"):
        return method_norm[len("deeplive_"):]
    if source_norm == "deeplive" and method_norm and not method_norm.startswith("visomaster_"):
        return method_norm
    return None


def infer_group_key(
    label, method=None, source=None, enhanced_strategy_names=None,
) -> str:
    label_id = normalize_label_value(label)
    method_norm = normalize_method_name(method)
    source_norm = normalize_method_name(source)
    enhanced = _normalize_strategy_set(enhanced_strategy_names)
    if not enhanced:
        enhanced = _normalize_strategy_set(DEFAULT_ENHANCED_STRATEGIES)

    # VisoMaster bad-data weak-signal lanes
    if source_norm == TEAMS_SOURCE or method_norm.startswith(TEAMS_METHOD):
        if label_id == 1:
            return TEAMS_FAKE_FAMILY
        if label_id == 0:
            return "visomaster_hints_teams_real"
    if source_norm == BASELINE_SOURCE or method_norm.startswith(BASELINE_METHOD):
        if label_id == 1:
            return BASELINE_FAKE_FAMILY
        if label_id == 0:
            return "visomaster_hints_real"

    # Teams passthrough
    if "deeplive_teams" in source_norm or "deeplive_teams" in method_norm:
        if label_id == 1:
            return "deeplive_teams_fake"
        if label_id == 0:
            return "deeplive_teams_real"

    # WMA failure
    if label_id == 1 and ("wma_failure" in method_norm or method_norm in {"failure_fake", "wma_failure_fake"}):
        return "wma_failure_fake"

    # External real
    if label_id == 0 and (
        source_norm == "external" or method_norm.startswith("external_") or "external" in method_norm
    ):
        return "external_real"

    # VisoMaster enhanced
    if source_norm == "visomaster_enhanced" or method_norm.startswith("visomaster_enhanced_"):
        if label_id == 0:
            return "visomaster_enhanced_real"
        if label_id == 1:
            return "visomaster_enhanced_fake"

    # VisoMaster res-variant
    if source_norm == "visomaster_res_variant" or method_norm.startswith("visomaster_inswapper128_res"):
        if label_id == 0:
            return "visomaster_res_variant_real"
        if label_id == 1:
            return "visomaster_res_variant_fake"

    # VisoMaster
    is_visomaster = (
        method_norm == "visomaster_real"
        or method_norm.startswith("visomaster_")
        or method_norm.startswith("ood_visomaster_")
    )
    if label_id == 0 and (source_norm == "visomaster" or is_visomaster):
        return "visomaster_real"
    if label_id == 1 and (source_norm == "visomaster" or is_visomaster):
        return "visomaster_fake"

    # DeepLive
    strategy = _extract_deeplive_strategy(method_norm, source_norm)
    if strategy is not None:
        if label_id == 0:
            if strategy in ("edge_cases", "edge_cases_enhanced"):
                return "deeplive_edge_cases_real"
            if strategy in ("minimal_processing", "minimal_processing_enhanced"):
                return "deeplive_minimal_processing_real"
            if strategy == "quality_enhancement":
                return "deeplive_quality_enhancement_real"
            return f"deeplive_{strategy}_real"
        if label_id == 1:
            if strategy == "edge_cases":
                return "deeplive_edge_cases_fake"
            if strategy == "minimal_processing":
                return "deeplive_minimal_processing_fake"
            if strategy == "quality_enhancement":
                return "deeplive_quality_enhancement_fake"
            if strategy == "edge_cases_enhanced":
                return "deeplive_edge_cases_enhanced_fake"
            if strategy == "minimal_processing_enhanced":
                return "deeplive_minimal_processing_enhanced_fake"
            return f"deeplive_{strategy}_fake"

    # DF40
    if source_norm == "df40":
        return "df40_fake" if label_id == 1 else "df40_real"
    if label_id == 0 and method_norm in DF40_REAL_METHOD_HINTS:
        return "df40_real"
    if label_id == 1 and method_norm in DF40_FAKE_METHOD_HINTS:
        return "df40_fake"

    if label_id == 0:
        return "unknown_real"
    if label_id == 1:
        return "unknown_fake"
    return "unknown"


def infer_family_key(label, method=None, source=None, enhanced_strategy_names=None) -> str:
    group_key = infer_group_key(label, method, source, enhanced_strategy_names)

    if group_key == BASELINE_FAKE_FAMILY:
        return BASELINE_FAKE_FAMILY
    if group_key == TEAMS_FAKE_FAMILY:
        return TEAMS_FAKE_FAMILY
    if group_key == "visomaster_hints_real":
        return "realpool_real"
    if group_key == "visomaster_hints_teams_real":
        return "deeplive_teams_real"
    if group_key == "df40_real":
        return "df40_real"
    if group_key == "df40_fake":
        return "df40_fake"
    if group_key == "deeplive_teams_fake":
        return "deeplive_teams_fake"
    if group_key == "deeplive_teams_real":
        return "deeplive_teams_real"
    if group_key in {"deeplive_edge_cases_fake", "deeplive_minimal_processing_fake"}:
        return "deeplive_non_enhanced_fake"
    if group_key in {
        "deeplive_quality_enhancement_fake",
        "deeplive_edge_cases_enhanced_fake",
        "deeplive_minimal_processing_enhanced_fake",
    }:
        return "deeplive_enhanced_fake"
    if group_key.startswith("deeplive_") and group_key.endswith("_fake"):
        if "enhanced" in group_key or "quality_enhancement" in group_key:
            return "deeplive_enhanced_fake"
        return "deeplive_non_enhanced_fake"
    if group_key == "visomaster_enhanced_fake":
        return "visomaster_enhanced_fake"
    if group_key == "visomaster_res_variant_fake":
        return "visomaster_res_variant_fake"
    if group_key == "visomaster_fake":
        return "visomaster_fake"
    if group_key in {
        "visomaster_enhanced_real", "visomaster_res_variant_real",
        "visomaster_real",
        "deeplive_edge_cases_real", "deeplive_minimal_processing_real",
        "deeplive_quality_enhancement_real",
    }:
        return "realpool_real"
    if group_key.startswith("deeplive_") and group_key.endswith("_real"):
        return "realpool_real"
    if group_key == "external_real":
        return "external_real"
    if group_key == "wma_failure_fake":
        return "wma_failure_fake"
    if group_key.endswith("_real"):
        return "realpool_real"
    if group_key.endswith("_fake"):
        return "unknown_fake"
    return "unknown"


def infer_group_and_family(label, method=None, source=None, enhanced_strategy_names=None):
    group_key = infer_group_key(label, method, source, enhanced_strategy_names)
    family_key = infer_family_key(label, method, source, enhanced_strategy_names)
    return group_key, family_key
