"""
Group and family taxonomy helpers.

These helpers provide a single mapping from validation/training metadata
(label/source/method) to canonical group keys and family keys.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union


DEFAULT_ENHANCED_STRATEGIES = (
    "quality_enhancement",
    "edge_cases_enhanced",
    "minimal_processing_enhanced",
)


DF40_REAL_METHOD_HINTS = {
    "faceforensics++",
    "faceforensicspp",
    "ff++",
    "ffpp",
    "df40_real",
}


DF40_FAKE_METHOD_HINTS = {
    "blendface",
    "facedancer",
    "facevid2vid",
    "faceswap",
    "fomm",
    "fsgan",
    "inswap",
    "lia",
    "mraa",
    "mcnet",
    "mobileswap",
    "one_shot_free",
    "pirender",
    "sadtalker",
    "simswap",
    "tpsm",
    "uniface",
    "wav2lip",
    "danet",
    "e4s",
    "ff_df",
    "ff_f2f",
    "ff_fs",
    "ff_nt",
}


def normalize_method_name(value: Optional[str]) -> str:
    """Normalize a method/source name to lowercase underscore format."""
    if value is None:
        return ""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_label_value(label: Union[str, int, float, bool, None]) -> Optional[int]:
    """Normalize labels to {0,1}. Returns None when parsing fails."""
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


def _normalize_strategy_set(names: Optional[Iterable[str]]) -> set[str]:
    if not names:
        return set()
    return {normalize_method_name(name) for name in names if name is not None}


def _extract_deeplive_strategy(method_norm: str, source_norm: str) -> Optional[str]:
    if method_norm.startswith("deeplive_"):
        return method_norm[len("deeplive_") :]
    if source_norm == "deeplive" and method_norm and not method_norm.startswith("visomaster_"):
        return method_norm
    return None


def _extract_proper_lane(method_norm: str, source_norm: str) -> Optional[str]:
    for lane in (
        "proper_real_clean",
        "proper_real_teams",
        "proper_visomaster_clean",
        "proper_visomaster_enhanced_clean",
        "proper_visomaster_teams",
        "proper_visomaster_enhanced_teams",
    ):
        if source_norm == lane:
            return lane
        if method_norm == lane or method_norm.startswith(f"{lane}__"):
            return lane
    return None


def infer_group_key(
    label: Union[str, int, float, bool, None],
    method: Optional[str] = None,
    source: Optional[str] = None,
    enhanced_strategy_names: Optional[Sequence[str]] = None,
) -> str:
    """
    Resolve canonical group_key for a sample using label/method/source.
    """
    label_id = normalize_label_value(label)
    method_norm = normalize_method_name(method)
    source_norm = normalize_method_name(source)
    enhanced = _normalize_strategy_set(enhanced_strategy_names)
    if not enhanced:
        enhanced = _normalize_strategy_set(DEFAULT_ENHANCED_STRATEGIES)

    proper_lane = _extract_proper_lane(method_norm, source_norm)
    if proper_lane is not None:
        if label_id == 0:
            return "proper_real_teams" if proper_lane.endswith("_teams") else "proper_real_clean"
        if label_id == 1 and proper_lane.startswith("proper_visomaster_"):
            return f"{proper_lane}_fake"

    # Teams passthrough groups — must be checked BEFORE generic DeepLive
    # routing because method strings start with "deeplive_teams_".
    if "deeplive_teams" in source_norm or "deeplive_teams" in method_norm:
        if label_id == 1:
            return "deeplive_teams_fake"
        if label_id == 0:
            return "deeplive_teams_real"

    # External WMA failure set (fake-only gate dataset).
    if label_id == 1 and ("wma_failure" in method_norm or method_norm in {"failure_fake", "wma_failure_fake"}):
        return "wma_failure_fake"

    # External real pools.
    if label_id == 0 and (
        source_norm == "external"
        or method_norm.startswith("external_")
        or "external" in method_norm
    ):
        return "external_real"

    # VisoMaster enhanced groups — must be checked BEFORE generic VisoMaster
    # routing because method strings start with "visomaster_enhanced_".
    is_visomaster_enhanced = (
        source_norm == "visomaster_enhanced"
        or method_norm.startswith("visomaster_enhanced_")
    )
    if is_visomaster_enhanced:
        if label_id == 0:
            return "visomaster_enhanced_real"
        if label_id == 1:
            return "visomaster_enhanced_fake"

    # VisoMaster resolution-variant groups — also before generic VisoMaster.
    is_visomaster_res_variant = (
        source_norm == "visomaster_res_variant"
        or method_norm.startswith("visomaster_inswapper128_res")
    )
    if is_visomaster_res_variant:
        if label_id == 0:
            return "visomaster_res_variant_real"
        if label_id == 1:
            return "visomaster_res_variant_fake"

    # Explicit WT-B hint lanes — must be checked BEFORE generic VisoMaster
    # routing because the method strings still share the "visomaster_" prefix.
    if source_norm == "visomaster_hints" or method_norm == "visomaster_hints":
        if label_id == 0:
            return "visomaster_hints_real"
        if label_id == 1:
            return "visomaster_hints_fake"

    if source_norm == "visomaster_hints_teams" or method_norm == "visomaster_hints_teams":
        if label_id == 0:
            return "visomaster_hints_teams_real"
        if label_id == 1:
            return "visomaster_hints_teams_fake"

    # VisoMaster groups.
    is_visomaster_method = (
        method_norm == "visomaster_real"
        or method_norm.startswith("visomaster_")
        or method_norm.startswith("ood_visomaster_")
    )
    if label_id == 0 and (source_norm == "visomaster" or is_visomaster_method):
        return "visomaster_real"
    if label_id == 1 and (source_norm == "visomaster" or is_visomaster_method):
        return "visomaster_fake"

    # DeepLive groups.
    strategy = _extract_deeplive_strategy(method_norm, source_norm)
    if strategy is not None:
        if label_id == 0:
            if strategy == "edge_cases":
                return "deeplive_edge_cases_real"
            if strategy == "minimal_processing":
                return "deeplive_minimal_processing_real"
            if strategy == "quality_enhancement":
                return "deeplive_quality_enhancement_real"
            if strategy == "edge_cases_enhanced":
                return "deeplive_edge_cases_real"
            if strategy == "minimal_processing_enhanced":
                return "deeplive_minimal_processing_real"
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
            if strategy in enhanced or strategy.endswith("_enhanced"):
                return f"deeplive_{strategy}_fake"
            return f"deeplive_{strategy}_fake"

    # DF40 groups.
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


def infer_family_key(
    label: Union[str, int, float, bool, None],
    method: Optional[str] = None,
    source: Optional[str] = None,
    enhanced_strategy_names: Optional[Sequence[str]] = None,
) -> str:
    """
    Resolve family_key from canonical group mapping.
    """
    group_key = infer_group_key(
        label=label,
        method=method,
        source=source,
        enhanced_strategy_names=enhanced_strategy_names,
    )

    if group_key == "df40_real":
        return "df40_real"
    if group_key == "df40_fake":
        return "df40_fake"

    # Teams passthrough families
    if group_key == "deeplive_teams_fake":
        return "deeplive_teams_fake"
    if group_key == "deeplive_teams_real":
        return "deeplive_teams_real"

    if group_key in {
        "deeplive_edge_cases_fake",
        "deeplive_minimal_processing_fake",
    }:
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

    if group_key == "visomaster_hints_fake":
        return "visomaster_hints_fake"

    if group_key == "visomaster_hints_teams_fake":
        return "visomaster_hints_teams_fake"

    if group_key == "visomaster_fake":
        return "visomaster_fake"

    if group_key in {
        "proper_visomaster_clean_fake",
        "proper_visomaster_enhanced_clean_fake",
        "proper_visomaster_teams_fake",
        "proper_visomaster_enhanced_teams_fake",
        "proper_real_clean",
        "proper_real_teams",
    }:
        return group_key

    if group_key == "visomaster_hints_real":
        return "visomaster_hints_real"

    if group_key == "visomaster_hints_teams_real":
        return "visomaster_hints_teams_real"

    if group_key in {
        "visomaster_enhanced_real",
        "visomaster_res_variant_real",
        "visomaster_real",
        "deeplive_edge_cases_real",
        "deeplive_minimal_processing_real",
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


def infer_group_and_family(
    label: Union[str, int, float, bool, None],
    method: Optional[str] = None,
    source: Optional[str] = None,
    enhanced_strategy_names: Optional[Sequence[str]] = None,
) -> Tuple[str, str]:
    """Convenience helper to resolve both group_key and family_key."""
    group_key = infer_group_key(
        label=label,
        method=method,
        source=source,
        enhanced_strategy_names=enhanced_strategy_names,
    )
    family_key = infer_family_key(
        label=label,
        method=method,
        source=source,
        enhanced_strategy_names=enhanced_strategy_names,
    )
    return group_key, family_key


__all__ = [
    "DEFAULT_ENHANCED_STRATEGIES",
    "infer_family_key",
    "infer_group_and_family",
    "infer_group_key",
    "normalize_label_value",
    "normalize_method_name",
]
