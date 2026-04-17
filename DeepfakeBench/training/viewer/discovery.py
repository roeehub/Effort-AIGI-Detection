"""
GCS bucket discovery — enumerate samples from all training data sources.

Each discover_*() function returns a list of SampleInfo dicts.
Results are cached as JSON in .viewer_cache/discovery/ to avoid
slow re-listing on subsequent runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from google.cloud import storage

from .splitting import SampleInfo, build_identity, extract_identity_from_video_name
from .visomaster_policy import (
    BASELINE_FAKE_FAMILY,
    BASELINE_LABEL,
    BASELINE_METHOD,
    BASELINE_SOURCE,
    BASELINE_WEIGHT_ALIAS,
    PolicyBundle,
    TEAMS_FAKE_FAMILY,
    TEAMS_LABEL,
    TEAMS_METHOD,
    TEAMS_SOURCE,
    TEAMS_WEIGHT_ALIAS,
    load_visomaster_bad_data_policy,
)

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".viewer_cache/discovery")
CACHE_MAX_AGE_SECONDS = 86400  # 24 hours

# ─── helpers ────────────────────────────────────────────────────────────────


def _strip_gcs_prefix(path: str, bucket_name: str) -> str:
    """Strip gs://bucket/ prefix to get blob-relative path."""
    prefix = f"gs://{bucket_name}/"
    if path.startswith(prefix):
        return path[len(prefix):]
    if path.startswith("gs://"):
        # Different bucket — strip the gs://other-bucket/ anyway
        parts = path[5:].split("/", 1)
        return parts[1] if len(parts) > 1 else ""
    return path


def _get_client(project: Optional[str] = None) -> storage.Client:
    proj = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "train-cvit2")
    return storage.Client(project=proj)


def _cache_path(source_key: str) -> Path:
    return CACHE_DIR / f"{source_key}.json"


def _load_cache(source_key: str, max_age: float = CACHE_MAX_AGE_SECONDS) -> Optional[List[dict]]:
    p = _cache_path(source_key)
    if not p.exists():
        return None
    age = time.time() - p.stat().st_mtime
    if age > max_age:
        return None
    try:
        data = json.loads(p.read_text())
        return data
    except Exception:
        return None


def _save_cache(source_key: str, samples: List[SampleInfo]):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _cache_path(source_key)
    p.write_text(json.dumps([asdict(s) for s in samples], default=str))


def samples_from_dicts(rows: List[dict]) -> List[SampleInfo]:
    out = []
    for r in rows:
        si = SampleInfo(
            sample_id=r["sample_id"],
            source=r["source"],
            method=r["method"],
            identity=r["identity"],
            label=r["label"],
            bucket=r["bucket"],
            frame_count=r.get("frame_count", 0),
            strategy=r.get("strategy", ""),
            swap_model=r.get("swap_model", ""),
            enhancer=r.get("enhancer", ""),
            tier=r.get("tier", ""),
            original_video_name=r.get("original_video_name", ""),
            has_pair=r.get("has_pair", False),
            has_teams_counterpart=r.get("has_teams_counterpart", False),
            teams_bucket=r.get("teams_bucket", ""),
            sampling_family_key=r.get("sampling_family_key"),
            extra=r.get("extra", {}),
            split=r.get("split", ""),
        )
        out.append(si)
    return out


def _cache_matches_policy_overlay(
    rows: List[dict],
    policy: Optional[PolicyBundle],
    expected_source: str,
) -> bool:
    if not policy:
        return True
    for row in rows:
        extra = row.get("extra") or {}
        if row.get("source") == expected_source and extra.get("policy_date_tag") == policy.date_tag:
            return True
    return False


def _policy_nominal_method(row) -> str:
    return f"visomaster_{row.swap_model}" if row.swap_model else "visomaster"


def _policy_extra(
    policy: PolicyBundle,
    row,
    *,
    family_weight_alias: str,
) -> Dict[str, Any]:
    return {
        "policy_action": row.policy_action,
        "policy_label": row.policy_label,
        "policy_lane": row.policy_lane,
        "policy_decision_reason": row.decision_reason,
        "policy_selection_group": row.selection_group,
        "policy_selection_seed": row.selection_seed,
        "policy_date_tag": policy.date_tag,
        "policy_nominal_method": _policy_nominal_method(row),
        "family_weight_alias": family_weight_alias,
    }


# ─── DF40 ───────────────────────────────────────────────────────────────────


def discover_df40(
    pair_json_path: str,
    gcs_bucket: str,
    methods: Optional[List[str]] = None,
    progress_cb: Optional[Callable] = None,
) -> List[SampleInfo]:
    """Discover DF40 paired samples from the local pair-matching JSON."""
    cache = _load_cache("df40")
    if cache is not None:
        logger.info("DF40: loaded %d samples from cache", len(cache))
        return samples_from_dicts(cache)

    if not os.path.isabs(pair_json_path):
        training_dir = Path(__file__).resolve().parent.parent
        pair_json_path = str(training_dir / pair_json_path)

    with open(pair_json_path) as f:
        data = json.load(f)

    allowed_methods = set(m.lower() for m in methods) if methods else None
    samples: List[SampleInfo] = []

    pairs = data.get("pairs", [])
    for i, pair in enumerate(pairs):
        method = pair["method"]
        if allowed_methods and method.lower() not in allowed_methods:
            continue

        target_id = pair.get("target_identity", "")
        source_id = pair.get("source_identity", "")
        pair_id = pair.get("pair_id", f"{method}__{target_id}_{source_id}")
        fake_info = pair.get("fake", {})
        real_info = pair.get("real", {})
        fake_frame_list = fake_info.get("frames", [])
        real_frame_list = real_info.get("frames", [])
        fake_frames = len(fake_frame_list)
        real_frames = len(real_frame_list)

        # Strip gs://bucket/ prefix from paths to get blob-relative paths
        raw_fake = fake_info.get("path", "")
        raw_real = real_info.get("path", "")
        fake_path = _strip_gcs_prefix(raw_fake, gcs_bucket)
        real_path = _strip_gcs_prefix(raw_real, gcs_bucket)

        shared_extra = {
            "pair_id": pair_id,
            "target_identity": target_id,
            "source_identity": source_id,
            "fake_path": fake_path,
            "real_path": real_path,
            "first_fake_frame": fake_frame_list[0] if fake_frame_list else "000.png",
            "first_real_frame": real_frame_list[0] if real_frame_list else "000.png",
            "fake_frames": fake_frame_list,
            "real_frames": real_frame_list,
        }

        # Fake sample
        samples.append(SampleInfo(
            sample_id=f"{pair_id}__fake",
            source="df40",
            method=method,
            identity=build_identity("df40", target_id),
            label=1,
            bucket=gcs_bucket,
            frame_count=fake_frames,
            has_pair=True,
            extra=dict(shared_extra),
        ))

        # Real sample
        samples.append(SampleInfo(
            sample_id=f"{pair_id}__real",
            source="df40",
            method=method,
            identity=build_identity("df40", target_id),
            label=0,
            bucket=gcs_bucket,
            frame_count=real_frames,
            has_pair=True,
            extra=dict(shared_extra),
        ))

        if progress_cb and i % 500 == 0:
            progress_cb("df40", i, len(pairs))

    _save_cache("df40", samples)
    logger.info("DF40: discovered %d samples (%d pairs)", len(samples), len(samples) // 2)
    return samples


# ─── DeepLive ───────────────────────────────────────────────────────────────


def discover_deeplive(
    gcs_bucket: str,
    include_strategies: Optional[List[str]] = None,
    exclude_strategies: Optional[List[str]] = None,
    progress_cb: Optional[Callable] = None,
) -> List[SampleInfo]:
    """Discover DeepLive samples by listing manifest.json files in GCS."""
    cache = _load_cache("deeplive")
    if cache is not None:
        logger.info("DeepLive: loaded %d samples from cache", len(cache))
        return samples_from_dicts(cache)

    client = _get_client()
    bucket = client.bucket(gcs_bucket)

    # Build strategy prefixes to scan
    default_strategies = [
        "edge_cases", "minimal_processing", "quality_enhancement",
        "edge_cases_enhanced", "minimal_processing_enhanced",
    ]
    strategies = include_strategies or default_strategies
    exclude_set = set(exclude_strategies or [])
    strategies = [s for s in strategies if s not in exclude_set]

    samples: List[SampleInfo] = []
    total_blobs = 0

    for strategy in strategies:
        prefix = f"samples/{strategy}_"
        # Use match_glob for server-side filtering — only return manifest.json blobs
        manifest_count = 0
        for blob in bucket.list_blobs(
            prefix=prefix,
            match_glob=f"samples/{strategy}_*/manifest.json",
        ):

            try:
                manifest = json.loads(blob.download_as_text())
            except Exception as exc:
                logger.warning("DeepLive: failed to read %s: %s", blob.name, exc)
                continue

            sample_id = manifest.get("sample_id", "")
            if not sample_id:
                parts = blob.name.split("/")
                sample_id = parts[1] if len(parts) >= 2 else ""

            # Skip visomaster samples (they share the bucket)
            if sample_id.startswith("visomaster_"):
                continue

            effective_strategy = manifest.get("strategy", strategy)
            ovn = manifest.get("original_video_name", sample_id)
            frame_count = int(manifest.get("frame_count") or 0)
            identity_raw = extract_identity_from_video_name(ovn)

            for label in [0, 1]:
                samples.append(SampleInfo(
                    sample_id=f"{sample_id}__{'fake' if label else 'real'}",
                    source="deeplive",
                    method=f"deeplive_{effective_strategy}",
                    identity=build_identity("deeplive", identity_raw),
                    label=label,
                    bucket=gcs_bucket,
                    frame_count=frame_count,
                    strategy=effective_strategy,
                    original_video_name=ovn,
                    has_pair=True,
                    extra={"has_landmarks": manifest.get("has_landmarks", False)},
                ))

            total_blobs += 1

        if progress_cb:
            progress_cb("deeplive", strategies.index(strategy) + 1, len(strategies))

    _save_cache("deeplive", samples)
    logger.info("DeepLive: discovered %d samples (%d paired dirs)", len(samples), total_blobs)
    return samples


# ─── VisoMaster ─────────────────────────────────────────────────────────────


def _extract_swap_model_from_sample_id(sample_id: str) -> str:
    """Extract swap model token from sample_id (e.g. visomaster_CSCS_00000 -> CSCS)."""
    if not sample_id:
        return ""
    remainder = sample_id
    if remainder.startswith("visomaster_"):
        remainder = remainder[len("visomaster_"):]
    first_us = remainder.find("_")
    if first_us <= 0:
        return ""
    return remainder[:first_us]


def discover_visomaster(
    gcs_bucket: str,
    swap_models: Optional[List[str]] = None,
    policy: Optional[PolicyBundle] = None,
    progress_cb: Optional[Callable] = None,
) -> List[SampleInfo]:
    """Discover VisoMaster samples from GCS."""
    cache = _load_cache("visomaster")
    if cache is not None and _cache_matches_policy_overlay(cache, policy, BASELINE_SOURCE):
        logger.info("VisoMaster: loaded %d samples from cache", len(cache))
        return samples_from_dicts(cache)
    if cache is not None and policy:
        logger.info("VisoMaster: rebuilding cache to apply bad-data policy overlay")

    client = _get_client()
    bucket = client.bucket(gcs_bucket)
    allowed_models = set(swap_models) if swap_models else None

    # Use match_glob to list only manifest.json blobs (skips ~33x frame blobs)
    samples: List[SampleInfo] = []
    manifest_count = 0
    for blob in bucket.list_blobs(
        prefix="samples/visomaster_",
        match_glob="samples/visomaster_*/manifest.json",
    ):

        # Pre-filter by swap model from blob path before downloading
        parts = blob.name.split("/")
        sample_id_hint = parts[1] if len(parts) >= 3 else ""
        if allowed_models:
            swap_hint = _extract_swap_model_from_sample_id(sample_id_hint)
            if swap_hint and swap_hint not in allowed_models:
                continue

        try:
            manifest = json.loads(blob.download_as_text())
        except Exception:
            continue

        sample_id = manifest.get("sample_id", sample_id_hint)

        policy_row = policy.row_for(sample_id) if policy else None
        if policy_row:
            if policy_row.in_teams_pair_complete:
                # The Teams-played bad lane is modeled separately.
                continue
            if policy_row.policy_action != "keep" or policy_row.policy_label != BASELINE_LABEL:
                continue

        # Extract swap model from sample_id (manifest may not have swap_model field)
        swap_model = manifest.get("swap_model", "") or _extract_swap_model_from_sample_id(sample_id)
        if allowed_models and swap_model not in allowed_models:
            continue

        # Skip enhanced samples (they have enhancer field)
        if manifest.get("enhancer"):
            continue

        ovn = manifest.get("original_video_name", sample_id)
        frame_count = int(manifest.get("frame_count") or 0)
        identity_raw = extract_identity_from_video_name(ovn)

        # Extract tier from manifest tier_data (may be in cropped bucket manifest)
        tier_data = manifest.get("tier_data", {})
        tier = ""
        if isinstance(tier_data, dict):
            tier = tier_data.get("identity_delta_tier", tier_data.get("tier", ""))
        if not tier:
            tier = manifest.get("tier", "")

        source = "visomaster"
        method = f"visomaster_{swap_model}"
        extra: Dict[str, Any] = {
            "identity_delta": tier_data.get("identity_delta", -1.0) if isinstance(tier_data, dict) else -1.0,
        }
        if policy_row:
            source = BASELINE_SOURCE
            method = BASELINE_METHOD
            if policy_row.swap_model:
                swap_model = policy_row.swap_model
            if policy_row.tier:
                tier = policy_row.tier
            if policy_row.original_video_name:
                ovn = policy_row.original_video_name
                identity_raw = extract_identity_from_video_name(ovn)
            if policy_row.identity_delta is not None:
                extra["identity_delta"] = policy_row.identity_delta
            extra.update(_policy_extra(policy, policy_row, family_weight_alias=BASELINE_WEIGHT_ALIAS))

        for label in [0, 1]:
            samples.append(SampleInfo(
                sample_id=f"{sample_id}__{'fake' if label else 'real'}",
                source=source,
                method=method,
                identity=build_identity("visomaster", identity_raw),
                label=label,
                bucket=gcs_bucket,
                frame_count=frame_count,
                swap_model=swap_model,
                tier=tier,
                original_video_name=ovn,
                has_pair=True,
                extra=dict(extra),
            ))

        manifest_count += 1
        if progress_cb and manifest_count % 50 == 0:
            progress_cb("visomaster", manifest_count, -1)

    _save_cache("visomaster", samples)
    logger.info("VisoMaster: discovered %d samples (%d paired dirs)", len(samples), len(samples) // 2)
    return samples


# ─── VisoMaster Enhanced ───────────────────────────────────────────────────


def discover_visomaster_enhanced(
    gcs_bucket: str,
    original_bucket: str,
    progress_cb: Optional[Callable] = None,
) -> List[SampleInfo]:
    """Discover VisoMaster Enhanced samples (enhancer post-processing)."""
    cache = _load_cache("visomaster_enhanced")
    if cache is not None:
        logger.info("VisoMaster Enhanced: loaded %d samples from cache", len(cache))
        return samples_from_dicts(cache)

    client = _get_client()
    bucket_obj = client.bucket(gcs_bucket)

    # Use match_glob to list only manifest.json blobs
    samples: List[SampleInfo] = []
    manifest_count = 0
    for blob in bucket_obj.list_blobs(
        prefix="samples/",
        match_glob="samples/*/manifest.json",
    ):

        try:
            manifest = json.loads(blob.download_as_text())
        except Exception:
            continue

        sample_id = manifest.get("sample_id", "")
        enhancer = manifest.get("enhancer", "")
        original_sample_id = manifest.get("original_sample_id", "")
        swap_model = manifest.get("swap_model", "") or _extract_swap_model_from_sample_id(sample_id)
        ovn = manifest.get("original_video_name", original_sample_id)
        frame_count = int(manifest.get("frame_count") or 0)
        tier_data = manifest.get("tier_data", {})
        tier = tier_data.get("tier", "") if isinstance(tier_data, dict) else ""
        identity_raw = extract_identity_from_video_name(ovn)

        # Enhanced samples: fake from this bucket, real from original bucket
        for label in [0, 1]:
            samples.append(SampleInfo(
                sample_id=f"{sample_id}__{'fake' if label else 'real'}",
                source="visomaster_enhanced",
                method=f"visomaster_enhanced_{enhancer}",
                identity=build_identity("visomaster_enhanced", identity_raw),
                label=label,
                bucket=gcs_bucket if label == 1 else original_bucket,
                frame_count=frame_count,
                swap_model=swap_model,
                enhancer=enhancer,
                tier=tier,
                original_video_name=ovn,
                has_pair=True,
                extra={
                    "original_sample_id": original_sample_id,
                    "tier_data": tier_data,
                },
            ))

        if progress_cb and manifest_count % 100 == 0:
            progress_cb("visomaster_enhanced", manifest_count, -1)

        manifest_count += 1

    _save_cache("visomaster_enhanced", samples)
    logger.info("VisoMaster Enhanced: discovered %d samples (%d paired dirs)",
                len(samples), len(samples) // 2)
    return samples


# ─── VisoMaster Teams-Enhanced (via resolver manifest) ──────────────────────


def discover_visomaster_teams_enhanced(
    resolver_manifest_uri: str,
    enhanced_bucket: str,
    include_statuses: Optional[List[str]] = None,
    sampling_family_key: str = "visomaster_enhanced_fake",
    progress_cb: Optional[Callable] = None,
) -> List[SampleInfo]:
    """Discover VisoMaster Teams-Enhanced samples from resolver manifest."""
    cache = _load_cache("visomaster_teams_enhanced")
    if cache is not None:
        logger.info("VisoMaster Teams-Enhanced: loaded %d samples from cache", len(cache))
        return samples_from_dicts(cache)

    # Download resolver manifest from GCS
    # Parse gs://bucket/path format
    if resolver_manifest_uri.startswith("gs://"):
        uri_parts = resolver_manifest_uri[5:].split("/", 1)
        manifest_bucket_name = uri_parts[0]
        manifest_blob_path = uri_parts[1] if len(uri_parts) > 1 else ""
        client = _get_client()
        manifest_bucket = client.bucket(manifest_bucket_name)
        blob = manifest_bucket.blob(manifest_blob_path)
        resolver_data = json.loads(blob.download_as_text())
    else:
        # Local file fallback
        with open(resolver_manifest_uri, "r") as f:
            resolver_data = json.load(f)

    rows = resolver_data.get("rows", [])
    allowed_statuses = set(include_statuses) if include_statuses else None

    samples: List[SampleInfo] = []
    for i, row in enumerate(rows):
        status = row.get("resolution_status", "")
        if allowed_statuses and status not in allowed_statuses:
            continue

        sample_id = row.get("sample_id", "")
        companion_bucket = row.get("resolved_companion_bucket", "")
        real_count = row.get("resolved_real_frame_count", 0)
        fake_count = row.get("resolved_fake_frame_count", 0)
        available_enhancers = row.get("available_enhancers", [])
        enhancer_frame_counts = row.get("enhancer_frame_counts", {})

        # Parse swap model and identity from sample_id
        # Format: visomaster_{SwapModel}_{NNNNN}
        parts = sample_id.split("_")
        swap_model = ""
        if len(parts) >= 3 and parts[0] == "visomaster":
            # Handle multi-word swap model names like GhostFace-v1
            swap_model = "_".join(parts[1:-1]) if len(parts) > 3 else parts[1]

        # Identity comes from companion bucket path structure
        # We need original_video_name — check row for it
        ovn = row.get("original_video_name", "")
        strategy = row.get("strategy", "")
        if not ovn:
            # Try to derive from sample_id via the original manifest
            ovn = sample_id
        identity_raw = extract_identity_from_video_name(ovn)

        is_teams = status == "teams_v2_companion"

        for label in [0, 1]:
            samples.append(SampleInfo(
                sample_id=f"{sample_id}__{'fake' if label else 'real'}",
                source="visomaster_teams_enhanced",
                method=f"visomaster_{swap_model}" if swap_model else strategy,
                identity=build_identity("visomaster_teams_enhanced", identity_raw),
                label=label,
                bucket=companion_bucket if label == 0 else enhanced_bucket,
                frame_count=real_count if label == 0 else fake_count,
                swap_model=swap_model,
                original_video_name=ovn,
                has_pair=True,
                has_teams_counterpart=is_teams,
                teams_bucket=companion_bucket if is_teams else "",
                sampling_family_key=sampling_family_key,
                extra={
                    "resolution_status": status,
                    "available_enhancers": available_enhancers,
                    "enhancer_frame_counts": enhancer_frame_counts,
                    "companion_bucket": companion_bucket,
                },
            ))

        if progress_cb and i % 50 == 0:
            progress_cb("visomaster_teams_enhanced", i, len(rows))

    _save_cache("visomaster_teams_enhanced", samples)
    logger.info("VisoMaster Teams-Enhanced: discovered %d samples (%d paired dirs)",
                len(samples), len(samples) // 2)
    return samples


# ─── Teams V2 ───────────────────────────────────────────────────────────────


def discover_teams(
    gcs_bucket: str,
    require_pair_complete: bool = True,
    policy: Optional[PolicyBundle] = None,
    progress_cb: Optional[Callable] = None,
) -> List[SampleInfo]:
    """Discover Teams passthrough samples from GCS."""
    cache = _load_cache("teams")
    if cache is not None and _cache_matches_policy_overlay(cache, policy, TEAMS_SOURCE):
        logger.info("Teams: loaded %d samples from cache", len(cache))
        return samples_from_dicts(cache)
    if cache is not None and policy:
        logger.info("Teams: rebuilding cache to apply bad-data policy overlay")

    client = _get_client()
    bucket = client.bucket(gcs_bucket)

    manifest_blobs = list(bucket.list_blobs(
        prefix="samples/", match_glob="**/manifest.json"
    ))

    samples: List[SampleInfo] = []
    skipped = 0
    for i, blob in enumerate(manifest_blobs):
        try:
            manifest = json.loads(blob.download_as_text())
        except Exception:
            continue

        if require_pair_complete and not manifest.get("pair_complete", False):
            skipped += 1
            continue

        sample_id = manifest.get("sample_id", "")
        if not sample_id:
            parts = blob.name.split("/")
            sample_id = parts[1] if len(parts) >= 2 else ""

        strategy = manifest.get("strategy", "unknown")
        ovn = manifest.get("original_video_name", sample_id)
        frame_count = int(manifest.get("frame_count") or 0)
        identity_raw = extract_identity_from_video_name(ovn)

        policy_row = policy.row_for(sample_id) if policy else None
        source = "deeplive_teams"
        method = f"deeplive_teams_{strategy}"
        swap_model = ""
        extra: Dict[str, Any] = {}
        if policy_row and policy_row.in_teams_pair_complete:
            if policy_row.policy_action != "keep" or policy_row.policy_label != TEAMS_LABEL:
                skipped += 1
                continue
            source = TEAMS_SOURCE
            method = TEAMS_METHOD
            swap_model = policy_row.swap_model
            if policy_row.original_video_name:
                ovn = policy_row.original_video_name
                identity_raw = extract_identity_from_video_name(ovn)
            extra.update(_policy_extra(policy, policy_row, family_weight_alias=TEAMS_WEIGHT_ALIAS))

        for label in [0, 1]:
            samples.append(SampleInfo(
                sample_id=f"{sample_id}__{'fake' if label else 'real'}",
                source=source,
                method=method,
                identity=build_identity("deeplive_teams", identity_raw),
                label=label,
                bucket=gcs_bucket,
                frame_count=frame_count,
                strategy=strategy,
                swap_model=swap_model,
                original_video_name=ovn,
                has_pair=True,
                has_teams_counterpart=True,
                teams_bucket=gcs_bucket,
                extra=dict(extra),
            ))

        if progress_cb and i % 50 == 0:
            progress_cb("teams", i, len(manifest_blobs))

    _save_cache("teams", samples)
    logger.info("Teams: discovered %d samples (%d paired dirs, skipped %d incomplete)",
                len(samples), len(samples) // 2, skipped)
    return samples


# ─── External Reals ─────────────────────────────────────────────────────────


def discover_external_reals(
    gcs_bucket: str,
    prefix: str = "real/VCD/",
    identity_pattern: str = r"real__VCD__(?P<md5>[a-f0-9]{32})_",
    max_total_samples: int = 1200,
    max_frames_per_identity: int = 15,
    progress_cb: Optional[Callable] = None,
) -> List[SampleInfo]:
    """Discover external real images from GCS (VCD webcam captures)."""
    cache = _load_cache("external_reals")
    if cache is not None:
        logger.info("External reals: loaded %d samples from cache", len(cache))
        return samples_from_dicts(cache)

    client = _get_client()
    bucket = client.bucket(gcs_bucket)
    rx = re.compile(identity_pattern)

    blobs = list(bucket.list_blobs(prefix=prefix))
    image_blobs = [b for b in blobs if any(b.name.endswith(ext)
                   for ext in (".png", ".jpg", ".jpeg", ".webp"))]

    # Group by identity
    by_identity: dict[str, list] = defaultdict(list)
    for blob in image_blobs:
        m = rx.search(blob.name)
        if m:
            md5 = m.group("md5")
            by_identity[md5].append(blob.name)

    samples: List[SampleInfo] = []
    total = 0
    for md5, paths in by_identity.items():
        paths = paths[:max_frames_per_identity]
        identity = build_identity("external", md5)
        for p in paths:
            if total >= max_total_samples:
                break
            samples.append(SampleInfo(
                sample_id=os.path.basename(p),
                source="external",
                method="external_vcd_real",
                identity=identity,
                label=0,
                bucket=gcs_bucket,
                frame_count=1,
                extra={"gcs_path": p},
            ))
            total += 1
        if total >= max_total_samples:
            break

    if progress_cb:
        progress_cb("external_reals", 1, 1)

    _save_cache("external_reals", samples)
    logger.info("External reals: discovered %d samples (%d identities)",
                len(samples), len(by_identity))
    return samples


# ─── Orchestrator ───────────────────────────────────────────────────────────


def discover_all(
    config: Dict[str, Any],
    progress_cb: Optional[Callable] = None,
    force_refresh: bool = False,
) -> List[SampleInfo]:
    """
    Discover all data sources defined in the experiment config.

    Args:
        config: Parsed experiment YAML (the full dict).
        progress_cb: Called with (source_key, current, total) for progress.
        force_refresh: If True, ignore cached discovery results.

    Returns:
        Combined list of SampleInfo from all enabled sources.
    """
    if force_refresh:
        # Clear all caches
        if CACHE_DIR.exists():
            for f in CACHE_DIR.glob("*.json"):
                f.unlink()

    cp = config.get("combined_paired", {})
    all_samples: List[SampleInfo] = []
    errors: List[str] = []
    visomaster_policy = load_visomaster_bad_data_policy(config)

    def _safe_discover(name: str, fn, **kwargs):
        """Run a discovery function with error resilience."""
        try:
            return fn(**kwargs)
        except Exception as exc:
            logger.exception("Discovery failed for %s", name)
            errors.append(f"{name}: {exc}")
            return []

    # 1. DF40
    df40_cfg = cp.get("df40", {})
    if df40_cfg and df40_cfg.get("enabled", True):
        logger.info("Discovering DF40...")
        s = _safe_discover("df40", discover_df40,
            pair_json_path=df40_cfg.get("pair_json", "dataset/df40_pairs/df40-pair-matching.json"),
            gcs_bucket=df40_cfg.get("gcs_bucket", "df40-frames-recropped-rfa85"),
            methods=df40_cfg.get("methods"),
            progress_cb=progress_cb,
        )
        all_samples.extend(s)
        if progress_cb:
            progress_cb("df40", 1, 1)

    # 2. DeepLive
    dl_cfg = cp.get("deeplive", {})
    if dl_cfg and dl_cfg.get("enabled", True):
        logger.info("Discovering DeepLive...")
        s = _safe_discover("deeplive", discover_deeplive,
            gcs_bucket=dl_cfg.get("gcs_bucket", "live-deepfake-methods-real-and-fake-frames-cropped"),
            include_strategies=dl_cfg.get("include_strategies"),
            exclude_strategies=dl_cfg.get("exclude_strategies"),
            progress_cb=progress_cb,
        )
        all_samples.extend(s)

    # 3. VisoMaster
    vm_cfg = cp.get("visomaster", {})
    if vm_cfg and vm_cfg.get("enabled", True):
        logger.info("Discovering VisoMaster...")
        s = _safe_discover("visomaster", discover_visomaster,
            gcs_bucket=vm_cfg.get("gcs_bucket", "live-deepfake-methods-real-and-fake-frames-cropped"),
            swap_models=vm_cfg.get("swap_models"),
            policy=visomaster_policy,
            progress_cb=progress_cb,
        )
        all_samples.extend(s)

    # 4. VisoMaster Enhanced
    ve_cfg = cp.get("visomaster_enhanced", {})
    if ve_cfg and ve_cfg.get("enabled", True):
        logger.info("Discovering VisoMaster Enhanced...")
        s = _safe_discover("visomaster_enhanced", discover_visomaster_enhanced,
            gcs_bucket=ve_cfg.get("gcs_bucket", "visomaster-enhanced-face-cropped"),
            original_bucket=ve_cfg.get("original_bucket",
                                       "live-deepfake-methods-real-and-fake-frames-cropped"),
            progress_cb=progress_cb,
        )
        all_samples.extend(s)

    # 5. VisoMaster Teams-Enhanced
    vte_cfg = cp.get("visomaster_teams_enhanced", {})
    if vte_cfg and vte_cfg.get("enabled", True):
        logger.info("Discovering VisoMaster Teams-Enhanced...")
        s = _safe_discover("visomaster_teams_enhanced", discover_visomaster_teams_enhanced,
            resolver_manifest_uri=vte_cfg["resolver_manifest_uri"],
            enhanced_bucket=vte_cfg.get("enhanced_bucket", "enhanced-visomaster-cropped"),
            include_statuses=vte_cfg.get("include_statuses"),
            sampling_family_key=vte_cfg.get("sampling_family_key", "visomaster_enhanced_fake"),
            progress_cb=progress_cb,
        )
        all_samples.extend(s)

    # 6. Teams V2
    teams_cfg = cp.get("teams", {})
    if teams_cfg and teams_cfg.get("enabled", True):
        logger.info("Discovering Teams V2...")
        s = _safe_discover("teams", discover_teams,
            gcs_bucket=teams_cfg.get("gcs_bucket",
                                     "live-deepfake-methods-real-and-fake-frames-cropped-teams-v2"),
            require_pair_complete=teams_cfg.get("require_pair_complete", True),
            policy=visomaster_policy,
            progress_cb=progress_cb,
        )
        all_samples.extend(s)

    # 7. External training reals
    ext_cfg_list = cp.get("external_training_reals", [])
    if ext_cfg_list:
        for ext_cfg in ext_cfg_list:
            logger.info("Discovering external reals from %s/%s...",
                        ext_cfg.get("bucket"), ext_cfg.get("prefix"))
            s = _safe_discover("external_reals", discover_external_reals,
                gcs_bucket=ext_cfg.get("bucket", "effort-collected-data"),
                prefix=ext_cfg.get("prefix", "real/VCD/"),
                identity_pattern=ext_cfg.get("identity_pattern",
                                             r"real__VCD__(?P<md5>[a-f0-9]{32})_"),
                max_total_samples=ext_cfg.get("max_total_samples", 1200),
                max_frames_per_identity=ext_cfg.get("max_frames_per_identity", 15),
                progress_cb=progress_cb,
            )
            all_samples.extend(s)

    if errors:
        logger.warning("Discovery completed with %d errors: %s", len(errors), "; ".join(errors))
    logger.info("Total discovered: %d samples from all sources", len(all_samples))
    return all_samples
