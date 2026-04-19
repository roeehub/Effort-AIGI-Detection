"""
VisoMaster Training Data Source

Loads VisoMaster paired real/fake frames from GCS for training.
Supports filtering by swap_model and tier, identity-stratified splitting,
and integration with the unified combined_paired pipeline.

VisoMaster samples are stored in the same GCS bucket as DeepLive:
  gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_*

Each sample has:
  - sample_id: visomaster_{SwapModel}_{number}
  - Paired real/fake frames: samples/{sample_id}/frames/{real,fake}/frame_XXXX.png
  - manifest.json with: strategy="visomaster", swap_model, frame_count
  - Tier data (from full-frames bucket): identity_delta_tier (MINIMAL/MODERATE/STRONG)
  - NO landmarks

Usage (standalone):
    data_config['data_source'] = 'visomaster'
    data_config['visomaster'] = {
        'gcs_bucket': 'live-deepfake-methods-real-and-fake-frames-cropped',
        'swap_models': ['CSCS', 'GhostFace-v1', ...],  # None = all
        'tiers': None,  # None = all tiers
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
    }

Usage (via combined_paired — preferred):
    See combined_paired.py for integration.

Created: Phase 2 implementation (February 2026)
"""

import json
import hashlib
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from fsspec.core import url_to_fs
from torch.utils.data import DataLoader, IterableDataset

from . import register_data_source, DataPipelineResult

logger = logging.getLogger(__name__)
DEFAULT_PARALLEL_FRAME_DOWNLOAD_WORKERS = 4


# =============================================================================
# VisoMaster Sample Data Class
# =============================================================================

@dataclass
class VisoMasterSample:
    """Represents a single VisoMaster paired real/fake sample."""
    sample_id: str
    swap_model: str
    frame_count: int
    tier: str  # MINIMAL, MODERATE, STRONG, or UNKNOWN
    identity_delta: float  # -1.0 if unknown
    bucket_name: str
    manifest: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def identity(self) -> str:
        """
        Extract identity from sample_id.

        VisoMaster sample_ids are: visomaster_{SwapModel}_{videoID}_{number}
        We use the videoID portion as the identity to ensure the same source
        video across different swap models is grouped together.
        
        Fallback: use the full sample_id hash if parsing fails.
        """
        # Try to extract from manifest first (most reliable)
        original_video = self.manifest.get("original_video_name", "")
        if original_video:
            # Strip extension and path
            name = original_video.rsplit("/", 1)[-1]
            if "." in name:
                name = name.rsplit(".", 1)[0]
            if name.startswith("cropped_"):
                name = name[8:]
            return name

        # Fallback: parse from sample_id
        # Format: visomaster_{SwapModel}_{rest}
        # SwapModel can have hyphens (e.g., GhostFace-v1)
        # Strategy: strip "visomaster_" prefix and trailing number
        remainder = self.sample_id
        if remainder.startswith("visomaster_"):
            remainder = remainder[len("visomaster_"):]
        # The last segment after the last underscore is typically the numeric ID
        last_us = remainder.rfind("_")
        if last_us > 0:
            return remainder[last_us + 1:]
        return self.sample_id

    def real_frame_paths(self, anchor_indices: List[int]) -> List[str]:
        """Build GCS paths for real frames at given anchor indices."""
        return [
            f"gs://{self.bucket_name}/samples/{self.sample_id}/frames/real/frame_{i:04d}.png"
            for i in anchor_indices if i < self.frame_count
        ]

    def fake_frame_paths(self, anchor_indices: List[int]) -> List[str]:
        """Build GCS paths for fake frames at given anchor indices."""
        return [
            f"gs://{self.bucket_name}/samples/{self.sample_id}/frames/fake/frame_{i:04d}.png"
            for i in anchor_indices if i < self.frame_count
        ]


# =============================================================================
# GCS Discovery
# =============================================================================

def _extract_swap_model_from_sample_id(sample_id: str) -> str:
    """Extract swap model token from sample_id when possible."""
    if not sample_id:
        return ""
    remainder = sample_id
    if remainder.startswith("visomaster_"):
        remainder = remainder[len("visomaster_"):]
    first_us = remainder.find("_")
    if first_us <= 0:
        return ""
    return remainder[:first_us]


def _normalize_token_list(values: Optional[List[str]]) -> List[str]:
    if not values:
        return []
    return sorted({str(v) for v in values if str(v).strip()})


def _build_discovery_signature(
    bucket,
    swap_models: Optional[List[str]],
) -> Dict[str, Any]:
    """
    Build a lightweight signature over VisoMaster manifest objects.

    Uses blob name + updated timestamp + size to detect dataset changes without
    downloading each manifest.
    """
    digest = hashlib.sha256()
    manifest_count = 0

    for blob in bucket.list_blobs(prefix="samples/visomaster_"):
        if not blob.name.endswith("manifest.json"):
            continue
        parts = blob.name.split("/")
        sample_id_hint = parts[1] if len(parts) >= 3 else ""
        if swap_models:
            swap_hint = _extract_swap_model_from_sample_id(sample_id_hint)
            if swap_hint and swap_hint not in swap_models:
                continue

        updated = getattr(blob, "updated", None)
        updated_iso = updated.isoformat() if updated else ""
        size = int(getattr(blob, "size", 0) or 0)
        digest.update(f"{blob.name}|{updated_iso}|{size}\n".encode("utf-8"))
        manifest_count += 1

    return {
        "manifest_count": manifest_count,
        "manifest_hash": digest.hexdigest(),
    }


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
        log.info("Wrote VisoMaster discovery cache: %s", uri)
    except Exception as exc:
        log.warning("Failed to write VisoMaster discovery cache %s: %s", uri, exc)


def _cache_to_samples(cache_payload: Dict[str, Any], bucket_name: str) -> List["VisoMasterSample"]:
    samples: List[VisoMasterSample] = []
    for row in cache_payload.get("samples", []) or []:
        original_video_name = row.get("original_video_name", "")
        samples.append(
            VisoMasterSample(
                sample_id=row["sample_id"],
                swap_model=row.get("swap_model", ""),
                frame_count=int(row.get("frame_count") or 16),
                tier=row.get("tier", "UNKNOWN"),
                identity_delta=float(row.get("identity_delta") or -1.0),
                bucket_name=bucket_name,
                manifest={"original_video_name": original_video_name} if original_video_name else {},
            )
        )
    return samples


def _samples_to_cache_rows(samples: List["VisoMasterSample"]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        rows.append(
            {
                "sample_id": sample.sample_id,
                "swap_model": sample.swap_model,
                "frame_count": int(sample.frame_count),
                "tier": sample.tier,
                "identity_delta": float(sample.identity_delta),
                "original_video_name": sample.manifest.get("original_video_name", ""),
            }
        )
    return rows


def discover_visomaster_samples(
    bucket_name: str = "live-deepfake-methods-real-and-fake-frames-cropped",
    frames_bucket_name: str = "live-deepfake-methods-real-and-fake-frames",
    gcs_project: str = "train-cvit2",
    swap_models: Optional[List[str]] = None,
    tiers: Optional[List[str]] = None,
    cache_manifest_uri: Optional[str] = None,
    cache_max_age_hours: float = 12.0,
    cache_validate_listing: bool = True,
    cache_revision: Optional[str] = None,
    log: Optional[logging.Logger] = None,
) -> List[VisoMasterSample]:
    """
    Discover VisoMaster samples from GCS and enrich with tier data.

    Args:
        bucket_name: Cropped frames bucket.
        frames_bucket_name: Full-frames bucket (for tier metadata).
        gcs_project: GCP project ID.
        swap_models: Optional list of swap models to include. None = all.
        tiers: Optional list of tiers to include. None = all.
        cache_manifest_uri: Optional URI (local or gs://) for discovery cache JSON.
        cache_max_age_hours: Max cache age before refresh.
        cache_validate_listing: If True, compare cached listing signature with GCS.
        cache_revision: Optional manual revision token to invalidate stale cache.
        log: Logger instance.

    Returns:
        List of VisoMasterSample objects.
    """
    if log is None:
        log = logger

    from google.cloud import storage

    client = storage.Client(project=gcs_project)
    swap_models_norm = _normalize_token_list(swap_models)
    tiers_norm = _normalize_token_list(tiers)

    # ------------------------------------------------------------------
    # 1. Discover visomaster manifests from cropped bucket
    # ------------------------------------------------------------------
    log.info("Discovering VisoMaster samples from gs://%s ...", bucket_name)
    cropped_bucket = client.bucket(bucket_name)
    cache_config = {
        "bucket_name": bucket_name,
        "frames_bucket_name": frames_bucket_name,
        "swap_models": swap_models_norm,
        "tiers": tiers_norm,
        "cache_revision": cache_revision or "",
    }

    if cache_manifest_uri and tiers_norm:
        log.info(
            "VisoMaster discovery cache bypassed because tier filtering is enabled "
            "(tier metadata freshness check is conservative)."
        )

    if cache_manifest_uri and not tiers_norm:
        cached_payload = _read_json_uri(cache_manifest_uri)
        if cached_payload and cached_payload.get("version") == 1:
            cache_matches = cached_payload.get("config") == cache_config
            cache_age_seconds = time.time() - float(cached_payload.get("created_unix", 0) or 0)
            cache_too_old = cache_age_seconds > float(cache_max_age_hours) * 3600.0
            if cache_matches and not cache_too_old:
                signature_matches = True
                if cache_validate_listing:
                    current_signature = _build_discovery_signature(
                        bucket=cropped_bucket,
                        swap_models=swap_models_norm or None,
                    )
                    signature_matches = (
                        current_signature == (cached_payload.get("manifest_signature") or {})
                    )
                    if not signature_matches:
                        log.info("VisoMaster discovery cache invalidated by listing signature change.")
                if signature_matches:
                    cached_samples = _cache_to_samples(cached_payload, bucket_name=bucket_name)
                    log.info(
                        "Loaded %d VisoMaster samples from cache: %s",
                        len(cached_samples),
                        cache_manifest_uri,
                    )
                    return cached_samples
            else:
                reason = "config_mismatch" if not cache_matches else "expired"
                log.info(
                    "VisoMaster discovery cache bypassed (%s): %s",
                    reason,
                    cache_manifest_uri,
                )

    manifests: Dict[str, Dict[str, Any]] = {}
    listing_signature_hasher = hashlib.sha256()
    listing_signature_count = 0

    for blob in cropped_bucket.list_blobs(prefix="samples/visomaster_"):
        if not blob.name.endswith("manifest.json"):
            continue
        # blob.name: samples/{sample_id}/manifest.json
        parts = blob.name.split("/")
        sample_id_hint = parts[1] if len(parts) >= 3 else ""
        if swap_models_norm:
            swap_hint = _extract_swap_model_from_sample_id(sample_id_hint)
            if swap_hint and swap_hint not in swap_models_norm:
                continue
        updated = getattr(blob, "updated", None)
        updated_iso = updated.isoformat() if updated else ""
        size = int(getattr(blob, "size", 0) or 0)
        listing_signature_hasher.update(f"{blob.name}|{updated_iso}|{size}\n".encode("utf-8"))
        listing_signature_count += 1
        try:
            manifest = json.loads(blob.download_as_text())
        except Exception as exc:
            log.warning("Failed to parse manifest %s: %s", blob.name, exc)
            continue
        if manifest.get("strategy") != "visomaster":
            continue
        manifests[manifest["sample_id"]] = manifest

    log.info("Discovered %d VisoMaster samples in cropped bucket", len(manifests))

    # ------------------------------------------------------------------
    # 2. Enrich with tier data from full-frames bucket
    # ------------------------------------------------------------------
    tier_cache: Dict[str, Dict[str, Any]] = {}
    if tiers_norm:
        log.info("Fetching tier data from gs://%s for %d sample(s) ...", frames_bucket_name, len(manifests))
        frames_bucket = client.bucket(frames_bucket_name)
        for sid in manifests.keys():
            blob = frames_bucket.blob(f"samples/{sid}/manifest.json")
            try:
                fm = json.loads(blob.download_as_text())
            except Exception:
                continue
            tier_data = fm.get("tier_data")
            if tier_data:
                tier_cache[sid] = tier_data
        log.info("Fetched tier data for %d sample(s)", len(tier_cache))
    else:
        log.info("Skipping tier manifest fetch (no tier filter configured).")

    # ------------------------------------------------------------------
    # 3. Build VisoMasterSample objects with filtering
    # ------------------------------------------------------------------
    samples: List[VisoMasterSample] = []
    model_counts: Dict[str, int] = Counter()
    tier_counts: Dict[str, int] = Counter()

    for sample_id, manifest in sorted(manifests.items()):
        # Parse swap_model
        swap_model = manifest.get("swap_model", "")
        if not swap_model:
            remainder = sample_id[len("visomaster_"):]
            last_us = remainder.rfind("_")
            if last_us > 0:
                swap_model = remainder[:last_us]
            else:
                swap_model = remainder

        # Filter by swap_model
        if swap_models_norm and swap_model not in swap_models_norm:
            continue

        # Resolve tier
        tier_data = tier_cache.get(sample_id, {})
        tier = tier_data.get("identity_delta_tier", "UNKNOWN")
        identity_delta = tier_data.get("identity_delta") if tier_data.get("identity_delta") is not None else -1.0

        # Filter by tier
        if tiers_norm and tier not in tiers_norm:
            continue

        sample = VisoMasterSample(
            sample_id=sample_id,
            swap_model=swap_model,
            frame_count=int(manifest.get("frame_count") or 16),
            tier=tier,
            identity_delta=float(identity_delta if identity_delta is not None else -1.0),
            bucket_name=bucket_name,
            manifest=manifest,
        )
        samples.append(sample)
        model_counts[swap_model] += 1
        tier_counts[tier] += 1

    log.info(
        "VisoMaster samples after filtering: %d (models=%s, tiers=%s)",
        len(samples), dict(model_counts), dict(tier_counts),
    )

    if cache_manifest_uri:
        cache_payload = {
            "version": 1,
            "created_unix": int(time.time()),
            "config": cache_config,
            "manifest_signature": {
                "manifest_count": listing_signature_count,
                "manifest_hash": listing_signature_hasher.hexdigest(),
            },
            "samples": _samples_to_cache_rows(samples),
        }
        _write_json_uri(cache_manifest_uri, cache_payload, log=log)

    return samples


# =============================================================================
# Frame Loading
# =============================================================================

def _decode_image_bytes(data: bytes, as_array: bool) -> Any:
    import io
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    if as_array:
        return np.array(img)
    return img


def _load_blob_image(
    bucket: Any,
    blob_path: str,
    as_array: bool,
) -> Any:
    data = bucket.blob(blob_path).download_as_bytes()
    return _decode_image_bytes(data, as_array=as_array)


def _load_paired_frames_in_order(
    anchor_indices: List[int],
    frame_count: int,
    load_pair: Callable[[int], Tuple[Any, Any]],
    *,
    sample_id: str,
    source_name: str,
    executor: Optional[Any] = None,
    parallel_download_workers: int = DEFAULT_PARALLEL_FRAME_DOWNLOAD_WORKERS,
) -> Tuple[List[Optional[Any]], List[Optional[Any]]]:
    """
    Load paired frames while preserving anchor position ordering.

    Missing or failed positions are returned as ``None`` placeholders so callers
    can skip them without misaligning later frame indices.
    """
    real_frames: List[Optional[Any]] = [None] * len(anchor_indices)
    fake_frames: List[Optional[Any]] = [None] * len(anchor_indices)
    valid_positions = [
        (position, idx)
        for position, idx in enumerate(anchor_indices)
        if idx < frame_count
    ]

    if not valid_positions:
        return real_frames, fake_frames

    def _store(position: int, real_img: Any, fake_img: Any) -> None:
        real_frames[position] = real_img
        fake_frames[position] = fake_img

    def _log_failure(position: int, exc: Exception) -> None:
        idx = anchor_indices[position]
        logger.debug(
            "Failed to load paired %s frame idx=%s for %s: %s",
            source_name,
            idx,
            sample_id,
            exc,
        )

    def _load_sequential() -> None:
        for position, idx in valid_positions:
            try:
                real_img, fake_img = load_pair(idx)
            except Exception as exc:
                _log_failure(position, exc)
                continue
            _store(position, real_img, fake_img)

    if executor is None:
        max_workers = max(1, int(parallel_download_workers or 1))
        if max_workers <= 1 or len(valid_positions) <= 1:
            _load_sequential()
            return real_frames, fake_frames

        with ThreadPoolExecutor(max_workers=min(max_workers, len(valid_positions))) as pool:
            future_to_position = {
                pool.submit(load_pair, idx): position
                for position, idx in valid_positions
            }
            for future in as_completed(future_to_position):
                position = future_to_position[future]
                try:
                    real_img, fake_img = future.result()
                except Exception as exc:
                    _log_failure(position, exc)
                    continue
                _store(position, real_img, fake_img)
        return real_frames, fake_frames

    future_to_position = {
        executor.submit(load_pair, idx): position
        for position, idx in valid_positions
    }
    for future in as_completed(future_to_position):
        position = future_to_position[future]
        try:
            real_img, fake_img = future.result()
        except Exception as exc:
            _log_failure(position, exc)
            continue
        _store(position, real_img, fake_img)

    return real_frames, fake_frames

def load_visomaster_frames(
    sample: VisoMasterSample,
    anchor_indices: List[int],
    as_array: bool = True,
    client: Optional[Any] = None,
    executor: Optional[Any] = None,
    parallel_download_workers: int = DEFAULT_PARALLEL_FRAME_DOWNLOAD_WORKERS,
) -> Tuple[List[Optional[Any]], List[Optional[Any]]]:
    """
    Load paired real/fake frames for a VisoMaster sample from GCS.

    Args:
        sample: VisoMasterSample to load.
        anchor_indices: Frame indices to load.
        as_array: If True, return numpy arrays; otherwise PIL images.
        client: Optional reusable GCS client.
        executor: Optional reusable thread pool for parallel downloads.
        parallel_download_workers: Fallback worker count when no executor is supplied.

    Returns:
        Tuple of (real_frames, fake_frames) aligned to ``anchor_indices``.
        Missing/failed positions are returned as ``None``.
    """
    from google.cloud import storage

    if client is None:
        client = storage.Client()
    bucket = client.bucket(sample.bucket_name)

    def _load_pair(idx: int) -> Tuple[Any, Any]:
        real_blob_path = f"samples/{sample.sample_id}/frames/real/frame_{idx:04d}.png"
        fake_blob_path = f"samples/{sample.sample_id}/frames/fake/frame_{idx:04d}.png"
        return (
            _load_blob_image(bucket, real_blob_path, as_array=as_array),
            _load_blob_image(bucket, fake_blob_path, as_array=as_array),
        )

    return _load_paired_frames_in_order(
        anchor_indices,
        sample.frame_count,
        _load_pair,
        sample_id=sample.sample_id,
        source_name="visomaster",
        executor=executor,
        parallel_download_workers=parallel_download_workers,
    )


# =============================================================================
# VisoMaster Enhanced Sample Data Class
# =============================================================================

@dataclass
class VisoMasterEnhancedSample:
    """Represents a post-hoc face-enhanced VisoMaster paired sample.

    Enhanced samples live in a *separate* bucket from the originals.
    The real frames come from the **original** bucket (cross-bucket reference)
    while the enhanced fake frames come from the enhanced bucket.

    Bucket layout::

        gs://<enhanced_bucket>/samples/<enhanced_sample_id>/frames/fake/frame_NNNN.png
        gs://<enhanced_bucket>/samples/<enhanced_sample_id>/manifest.json

    The manifest contains ``gcs_paths`` that point back to the original bucket
    for the real frames.
    """
    sample_id: str               # e.g. "visomaster_CSCS_00007_enhanced_gfpgan"
    original_sample_id: str      # e.g. "visomaster_CSCS_00007"
    swap_model: str              # e.g. "CSCS"
    enhancer: str                # e.g. "gfpgan"
    frame_count: int             # typically 16
    tier: str                    # ARTIFACT / STRONG / MODERATE / MINIMAL
    identity_delta: float        # from tier_data
    artifact_delta: float        # from tier_data
    enhanced_bucket: str         # bucket for enhanced fake frames
    original_bucket: str         # bucket for real frames (cross-reference)
    manifest: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def identity(self) -> str:
        """Extract identity from original_sample_id (same as parent VisoMaster)."""
        original_video = self.manifest.get("original_video_name", "")
        if original_video:
            name = original_video.rsplit("/", 1)[-1]
            if "." in name:
                name = name.rsplit(".", 1)[0]
            if name.startswith("cropped_"):
                name = name[8:]
            return name

        # Fallback: parse from original_sample_id
        remainder = self.original_sample_id
        if remainder.startswith("visomaster_"):
            remainder = remainder[len("visomaster_"):]
        last_us = remainder.rfind("_")
        if last_us > 0:
            return remainder[last_us + 1:]
        return self.original_sample_id

    def real_frame_paths(self, anchor_indices: List[int]) -> List[str]:
        """Build GCS paths for real frames from the *original* bucket."""
        return [
            f"gs://{self.original_bucket}/samples/{self.original_sample_id}/frames/real/frame_{i:04d}.png"
            for i in anchor_indices if i < self.frame_count
        ]

    def fake_frame_paths(self, anchor_indices: List[int]) -> List[str]:
        """Build GCS paths for enhanced fake frames from the *enhanced* bucket."""
        return [
            f"gs://{self.enhanced_bucket}/samples/{self.sample_id}/frames/fake/frame_{i:04d}.png"
            for i in anchor_indices if i < self.frame_count
        ]


VISOMASTER_ENHANCER_METHOD_MAP: Dict[str, str] = {
    "codeformer": "visomaster_enhanced_codeformer",
    "gfpgan": "visomaster_enhanced_gfpgan",
    "gpen-256": "visomaster_enhanced_gpen_bfr_256",
    "gpen-512": "visomaster_enhanced_gpen_bfr_512",
    "gpen-1024": "visomaster_enhanced_gpen_bfr_1024",
    "gpen-2048": "visomaster_enhanced_gpen_bfr_2048",
    "restoreformer++": "visomaster_enhanced_restoreformer_pp",
    "vqfr-v2": "visomaster_enhanced_vqfr_v2",
}
_FRAME_EXTENSION_CANDIDATES: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")


def visomaster_enhancer_to_method_name(enhancer: str) -> str:
    """Map resolver enhancer keys to the repo's label-space method names."""
    enhancer_norm = str(enhancer or "").strip().lower()
    mapped = VISOMASTER_ENHANCER_METHOD_MAP.get(enhancer_norm)
    if mapped:
        return mapped

    safe = enhancer_norm.replace("++", "_pp").replace("-", "_").replace("+", "_plus")
    return f"visomaster_enhanced_{safe}"


def _primary_frame_extension(extension_counts: Dict[str, int], default: str = ".png") -> str:
    """Pick the most common discovered frame extension for a side."""
    if not extension_counts:
        return default

    best_ext = default
    best_count = -1
    for ext, count in extension_counts.items():
        if int(count or 0) > best_count:
            best_ext = ext
            best_count = int(count or 0)
    return best_ext


def _ordered_frame_extensions(preferred_ext: Optional[str]) -> List[str]:
    preferred = str(preferred_ext or "").strip().lower()
    if preferred and not preferred.startswith("."):
        preferred = f".{preferred}"

    ordered: List[str] = []
    for ext in [preferred, *_FRAME_EXTENSION_CANDIDATES]:
        if ext and ext not in ordered:
            ordered.append(ext)
    return ordered


def _load_frame_from_bucket(
    bucket: Any,
    frame_prefix: str,
    preferred_ext: Optional[str],
    as_array: bool,
) -> Any:
    last_exc: Optional[Exception] = None
    for ext in _ordered_frame_extensions(preferred_ext):
        blob_path = f"{frame_prefix}{ext}"
        try:
            return _load_blob_image(bucket, blob_path, as_array=as_array)
        except Exception as exc:
            last_exc = exc
            continue

    if last_exc is None:
        last_exc = FileNotFoundError(frame_prefix)
    raise last_exc


@dataclass
class VisoMasterTeamsEnhancedSample:
    """Resolver-driven merged sample for the Teams-enhanced VisoMaster track."""

    sample_id: str
    strategy: str
    swap_model: str
    frame_count: int
    companion_bucket: str
    companion_domain: str  # teams_v2 or clean_fallback
    companion_real_ext: str
    companion_fake_ext: str
    companion_real_frame_count: int
    companion_fake_frame_count: int
    enhanced_bucket: str
    available_enhancers: Tuple[str, ...] = field(default_factory=tuple)
    enhancer_frame_counts: Dict[str, int] = field(default_factory=dict)
    manifest: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def identity(self) -> str:
        """Extract the shared real identity from the base sample ID."""
        original_video = self.manifest.get("original_video_name", "")
        if original_video:
            name = original_video.rsplit("/", 1)[-1]
            if "." in name:
                name = name.rsplit(".", 1)[0]
            if name.startswith("cropped_"):
                name = name[8:]
            return name

        remainder = self.sample_id
        if remainder.startswith("visomaster_"):
            remainder = remainder[len("visomaster_"):]
        last_us = remainder.rfind("_")
        if last_us > 0:
            return remainder[last_us + 1:]
        return self.sample_id

    @property
    def original_method(self) -> str:
        return f"visomaster_{self.swap_model}"

    @property
    def enhancer_methods(self) -> Tuple[str, ...]:
        return tuple(
            visomaster_enhancer_to_method_name(enhancer)
            for enhancer in self.available_enhancers
        )

    @property
    def method_variants(self) -> Tuple[str, ...]:
        ordered = [self.original_method, *self.enhancer_methods]
        deduped: List[str] = []
        for method in ordered:
            if method and method not in deduped:
                deduped.append(method)
        return tuple(deduped)


# =============================================================================
# VisoMaster Enhanced Discovery
# =============================================================================

def discover_visomaster_enhanced_samples(
    enhanced_bucket: str = "visomaster-enhanced-face-cropped",
    original_bucket: str = "live-deepfake-methods-real-and-fake-frames-cropped",
    gcs_project: str = "train-cvit2",
    exclude_tiers: Optional[List[str]] = None,
    enhancers: Optional[List[str]] = None,
    cache_manifest_uri: Optional[str] = None,
    cache_max_age_hours: float = 12.0,
    log: Optional[logging.Logger] = None,
) -> List[VisoMasterEnhancedSample]:
    """
    Discover post-hoc face-enhanced VisoMaster samples from GCS.

    Args:
        enhanced_bucket: GCS bucket containing enhanced fake crops.
        original_bucket: GCS bucket containing original real + non-enhanced fake crops.
        gcs_project: GCP project ID.
        exclude_tiers: Tier labels to exclude (e.g., ["ARTIFACT"]).
        enhancers: Optional list of enhancer names to include. None = all 8.
        cache_manifest_uri: Optional URI for discovery cache JSON.
        cache_max_age_hours: Max cache age before refresh.
        log: Logger instance.

    Returns:
        List of VisoMasterEnhancedSample objects.
    """
    if log is None:
        log = logger

    from google.cloud import storage as gcs_storage

    client = gcs_storage.Client(project=gcs_project)
    bucket = client.bucket(enhanced_bucket)
    exclude_tiers_norm = {t.upper() for t in (exclude_tiers or [])}
    enhancers_norm = {e.lower() for e in enhancers} if enhancers else None

    # ------------------------------------------------------------------
    # 1. Try loading from cache
    # ------------------------------------------------------------------
    if cache_manifest_uri:
        cached = _read_json_uri(cache_manifest_uri)
        if cached and cached.get("version") == 2:
            cache_config = cached.get("config", {})
            cache_age = time.time() - float(cached.get("created_unix", 0) or 0)
            config_matches = (
                cache_config.get("enhanced_bucket") == enhanced_bucket
                and cache_config.get("original_bucket") == original_bucket
            )
            if config_matches and cache_age <= cache_max_age_hours * 3600:
                samples = _enhanced_cache_to_samples(
                    cached, enhanced_bucket, original_bucket,
                    exclude_tiers_norm, enhancers_norm,
                )
                log.info(
                    "Loaded %d VisoMaster enhanced samples from cache: %s",
                    len(samples), cache_manifest_uri,
                )
                return samples
            else:
                reason = "config_mismatch" if not config_matches else "expired"
                log.info(
                    "VisoMaster enhanced cache bypassed (%s): %s",
                    reason, cache_manifest_uri,
                )

    # ------------------------------------------------------------------
    # 2. Discover manifests from enhanced bucket
    # ------------------------------------------------------------------
    log.info("Discovering VisoMaster enhanced samples from gs://%s ...", enhanced_bucket)

    manifests: Dict[str, Dict[str, Any]] = {}
    for blob in bucket.list_blobs(prefix="samples/"):
        if not blob.name.endswith("manifest.json"):
            continue
        parts = blob.name.split("/")
        if len(parts) < 3:
            continue
        try:
            manifest = json.loads(blob.download_as_text())
        except Exception as exc:
            log.warning("Failed to parse enhanced manifest %s: %s", blob.name, exc)
            continue
        sid = manifest.get("sample_id") or parts[1]
        manifests[sid] = manifest

    log.info("Discovered %d enhanced manifests in gs://%s", len(manifests), enhanced_bucket)

    # ------------------------------------------------------------------
    # 3. Build sample objects with filtering
    # ------------------------------------------------------------------
    samples: List[VisoMasterEnhancedSample] = []
    enhancer_counts: Dict[str, int] = Counter()
    tier_counts: Dict[str, int] = Counter()
    skipped_tier = 0
    skipped_enhancer = 0

    for sid, manifest in sorted(manifests.items()):
        enhancer = manifest.get("enhancer", "")
        if not enhancer:
            # Try to extract from sample_id: ..._enhanced_{enhancer}
            if "_enhanced_" in sid:
                enhancer = sid.rsplit("_enhanced_", 1)[-1]

        enhancer_lower = enhancer.lower()

        # Skip samples that have no enhancer at all (e.g. resolution variants
        # stored in the same bucket).  They are handled by the dedicated
        # discover_visomaster_res_variant_samples() function.
        if not enhancer_lower:
            skipped_enhancer += 1
            continue

        if enhancers_norm and enhancer_lower not in enhancers_norm:
            skipped_enhancer += 1
            continue

        # Tier data
        tier_data = manifest.get("tier_data", {}) or {}
        tier = tier_data.get("tier", manifest.get("tier", "UNKNOWN")).upper()
        if tier in exclude_tiers_norm:
            skipped_tier += 1
            continue

        _id_raw = tier_data.get("identity_delta") if tier_data.get("identity_delta") is not None else manifest.get("identity_delta")
        identity_delta = float(_id_raw if _id_raw is not None else -1.0)
        _ad_raw = tier_data.get("artifact_delta") if tier_data.get("artifact_delta") is not None else manifest.get("artifact_delta")
        artifact_delta = float(_ad_raw if _ad_raw is not None else -1.0)

        original_sample_id = manifest.get("original_sample_id", "")
        swap_model = manifest.get("swap_model", "")
        if not swap_model and original_sample_id:
            # Parse from original_sample_id: visomaster_{SwapModel}_{number}
            remainder = original_sample_id
            if remainder.startswith("visomaster_"):
                remainder = remainder[len("visomaster_"):]
            first_us = remainder.find("_")
            if first_us > 0:
                swap_model = remainder[:first_us]

        # Resolve original_video_name for identity extraction
        gcs_paths = manifest.get("gcs_paths", {}) or {}
        original_video_name = manifest.get("original_video_name", "")
        if not original_video_name:
            # Try to get from cross-referenced original manifest
            original_video_name = gcs_paths.get("original_video_name", "")

        sample = VisoMasterEnhancedSample(
            sample_id=sid,
            original_sample_id=original_sample_id,
            swap_model=swap_model,
            enhancer=enhancer_lower,
            frame_count=int(manifest.get("frame_count") or 16),
            tier=tier,
            identity_delta=identity_delta,
            artifact_delta=artifact_delta,
            enhanced_bucket=enhanced_bucket,
            original_bucket=original_bucket,
            manifest={
                "original_video_name": original_video_name,
                **{k: v for k, v in manifest.items() if k != "gcs_paths"},
            },
        )
        samples.append(sample)
        enhancer_counts[enhancer_lower] += 1
        tier_counts[tier] += 1

    log.info(
        "VisoMaster enhanced samples after filtering: %d "
        "(enhancers=%s, tiers=%s, skipped_tier=%d, skipped_enhancer=%d)",
        len(samples), dict(enhancer_counts), dict(tier_counts),
        skipped_tier, skipped_enhancer,
    )

    # ------------------------------------------------------------------
    # 4. Write cache
    # ------------------------------------------------------------------
    if cache_manifest_uri:
        cache_payload = {
            "version": 2,
            "created_unix": int(time.time()),
            "config": {
                "enhanced_bucket": enhanced_bucket,
                "original_bucket": original_bucket,
            },
            "samples": _enhanced_samples_to_cache_rows(samples),
        }
        _write_json_uri(cache_manifest_uri, cache_payload, log=log)

    return samples


def _enhanced_samples_to_cache_rows(samples: List[VisoMasterEnhancedSample]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for s in samples:
        rows.append({
            "sample_id": s.sample_id,
            "original_sample_id": s.original_sample_id,
            "swap_model": s.swap_model,
            "enhancer": s.enhancer,
            "frame_count": s.frame_count,
            "tier": s.tier,
            "identity_delta": s.identity_delta,
            "artifact_delta": s.artifact_delta,
            "original_video_name": s.manifest.get("original_video_name", ""),
        })
    return rows


def _enhanced_cache_to_samples(
    cache_payload: Dict[str, Any],
    enhanced_bucket: str,
    original_bucket: str,
    exclude_tiers: set,
    enhancers_norm: Optional[set],
) -> List[VisoMasterEnhancedSample]:
    samples: List[VisoMasterEnhancedSample] = []
    for row in cache_payload.get("samples", []) or []:
        tier = row.get("tier", "UNKNOWN").upper()
        if tier in exclude_tiers:
            continue
        enhancer = row.get("enhancer", "").lower()
        if enhancers_norm and enhancer not in enhancers_norm:
            continue
        original_video_name = row.get("original_video_name", "")
        samples.append(VisoMasterEnhancedSample(
            sample_id=row["sample_id"],
            original_sample_id=row.get("original_sample_id", ""),
            swap_model=row.get("swap_model", ""),
            enhancer=enhancer,
            frame_count=int(row.get("frame_count") or 16),
            tier=tier,
            identity_delta=float(row.get("identity_delta") or -1.0),
            artifact_delta=float(row.get("artifact_delta") or -1.0),
            enhanced_bucket=enhanced_bucket,
            original_bucket=original_bucket,
            manifest={"original_video_name": original_video_name} if original_video_name else {},
        ))
    return samples


def discover_visomaster_teams_enhanced_samples(
    resolver_manifest_uri: str,
    enhanced_bucket: str = "enhanced-visomaster-cropped",
    companion_domains: Optional[List[str]] = None,
    enhancers: Optional[List[str]] = None,
    include_statuses: Optional[List[str]] = None,
    require_all_expected_enhancers: bool = False,
    max_samples_total: Optional[int] = None,
    log: Optional[logging.Logger] = None,
) -> List[VisoMasterTeamsEnhancedSample]:
    """
    Load merged Teams-enhanced VisoMaster samples from a resolver manifest.

    The resolver manifest is produced by
    ``tools/audit_enhanced_visomaster_resolver.py`` and is the training-safe
    source of truth for companion bucket resolution.
    """
    if log is None:
        log = logger

    if not resolver_manifest_uri:
        raise ValueError(
            "visomaster_teams_enhanced requires resolver_manifest_uri"
        )

    payload = _read_json_uri(resolver_manifest_uri)
    if payload is None:
        raise FileNotFoundError(
            f"Failed to read VisoMaster Teams-enhanced resolver manifest: {resolver_manifest_uri}"
        )

    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        raise ValueError(
            f"Invalid VisoMaster Teams-enhanced resolver manifest (missing rows list): {resolver_manifest_uri}"
        )

    domains_norm = {str(v).strip().lower() for v in (companion_domains or []) if str(v).strip()}
    enhancers_norm = {str(v).strip().lower() for v in (enhancers or []) if str(v).strip()}
    statuses_norm = {
        str(v).strip().lower() for v in (
            include_statuses or ["teams_v2_companion", "clean_companion_only"]
        )
        if str(v).strip()
    }

    samples: List[VisoMasterTeamsEnhancedSample] = []
    domain_counts: Dict[str, int] = Counter()
    status_counts: Dict[str, int] = Counter()
    enhancer_counts: Dict[str, int] = Counter()
    skipped_missing_companion = 0
    skipped_domain = 0
    skipped_enhancers = 0
    skipped_require_all = 0

    for row in sorted(rows, key=lambda item: str(item.get("sample_id") or "")):
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            continue

        status = str(row.get("resolution_status") or "").strip().lower()
        resolved_bucket = str(row.get("resolved_companion_bucket") or "").strip()
        if status not in statuses_norm or not resolved_bucket:
            skipped_missing_companion += 1
            continue

        companion_domain = "teams_v2" if status == "teams_v2_companion" else "clean_fallback"
        if domains_norm and companion_domain not in domains_norm:
            skipped_domain += 1
            continue

        if require_all_expected_enhancers and not bool(row.get("all_expected_enhancers_present")):
            skipped_require_all += 1
            continue

        raw_enhancer_counts = row.get("enhancer_frame_counts") or {}
        enhancer_frame_counts = {
            str(name).strip().lower(): int(count or 0)
            for name, count in raw_enhancer_counts.items()
            if str(name).strip()
        }
        available_enhancers = [
            str(name).strip().lower()
            for name in (row.get("available_enhancers") or list(enhancer_frame_counts.keys()))
            if str(name).strip()
        ]
        if enhancers_norm:
            available_enhancers = [name for name in available_enhancers if name in enhancers_norm]
            enhancer_frame_counts = {
                name: count
                for name, count in enhancer_frame_counts.items()
                if name in enhancers_norm
            }

        available_enhancers = sorted(
            {
                name for name in available_enhancers
                if int(enhancer_frame_counts.get(name, 0) or 0) > 0
            }
        )
        if not available_enhancers:
            skipped_enhancers += 1
            continue

        resolved_extensions = row.get("resolved_extensions") or {}
        real_ext = _primary_frame_extension(
            resolved_extensions.get("real") or {},
            default=".png",
        )
        fake_ext = _primary_frame_extension(
            resolved_extensions.get("fake") or {},
            default=real_ext,
        )

        strategy = str(row.get("strategy") or "").strip()
        swap_model = ""
        if strategy.startswith("visomaster_"):
            swap_model = strategy[len("visomaster_") :]
        if not swap_model:
            swap_model = _extract_swap_model_from_sample_id(sample_id)

        frame_count_candidates = [
            int(row.get("resolved_real_frame_count") or 0),
            int(row.get("resolved_fake_frame_count") or 0),
            *(int(enhancer_frame_counts.get(name, 0) or 0) for name in available_enhancers),
        ]

        sample = VisoMasterTeamsEnhancedSample(
            sample_id=sample_id,
            strategy=strategy,
            swap_model=swap_model,
            frame_count=max(frame_count_candidates) if frame_count_candidates else 0,
            companion_bucket=resolved_bucket,
            companion_domain=companion_domain,
            companion_real_ext=real_ext,
            companion_fake_ext=fake_ext,
            companion_real_frame_count=int(row.get("resolved_real_frame_count") or 0),
            companion_fake_frame_count=int(row.get("resolved_fake_frame_count") or 0),
            enhanced_bucket=enhanced_bucket,
            available_enhancers=tuple(available_enhancers),
            enhancer_frame_counts={
                name: int(enhancer_frame_counts.get(name, 0) or 0)
                for name in available_enhancers
            },
            manifest={
                "resolver_manifest_uri": resolver_manifest_uri,
                **dict(row),
            },
        )
        samples.append(sample)
        domain_counts[companion_domain] += 1
        status_counts[status] += 1
        for enhancer in sample.available_enhancers:
            enhancer_counts[enhancer] += 1

        if max_samples_total is not None and len(samples) >= max_samples_total:
            break

    log.info(
        "Loaded %d VisoMaster Teams-enhanced merged samples from resolver manifest: %s",
        len(samples),
        resolver_manifest_uri,
    )
    log.info(
        "  - Domains=%s statuses=%s enhancers=%s skipped_missing_companion=%d skipped_domain=%d skipped_enhancers=%d skipped_require_all=%d",
        dict(sorted(domain_counts.items())),
        dict(sorted(status_counts.items())),
        dict(sorted(enhancer_counts.items())),
        skipped_missing_companion,
        skipped_domain,
        skipped_enhancers,
        skipped_require_all,
    )
    return samples


# =============================================================================
# VisoMaster Resolution-Variant Discovery
# =============================================================================

def discover_visomaster_res_variant_samples(
    bucket_name: str = "visomaster-enhanced-face-cropped",
    original_bucket: str = "live-deepfake-methods-real-and-fake-frames-cropped",
    gcs_project: str = "train-cvit2",
    exclude_tiers: Optional[List[str]] = None,
    resolutions: Optional[List[int]] = None,
    cache_manifest_uri: Optional[str] = None,
    cache_max_age_hours: float = 12.0,
    log: Optional[logging.Logger] = None,
) -> List[VisoMasterEnhancedSample]:
    """
    Discover Inswapper128 resolution-variant samples from GCS.

    These samples live in the same bucket as enhanced fakes but are identified
    by the ``swapper_resolution`` manifest field.  Each resolution variant is
    represented as a :class:`VisoMasterEnhancedSample` with ``enhancer`` set
    to ``inswapper128_res{resolution}`` for compatibility with the loading
    pipeline.

    Args:
        bucket_name: GCS bucket containing the resolution-variant fake crops.
        original_bucket: GCS bucket for the paired real frames.
        gcs_project: GCP project ID.
        exclude_tiers: Tier labels to skip (e.g., ``["ARTIFACT"]``).
        resolutions: Optional list of resolutions to include (e.g., [128, 256]).
            ``None`` means include all discovered resolutions.
        cache_manifest_uri: Optional GCS URI for a discovery cache JSON (version 3).
        cache_max_age_hours: Max cache age before refresh.
        log: Logger instance.

    Returns:
        List of :class:`VisoMasterEnhancedSample` objects.
    """
    if log is None:
        log = logger

    from google.cloud import storage as gcs_storage

    client = gcs_storage.Client(project=gcs_project)
    bucket = client.bucket(bucket_name)
    exclude_tiers_norm = {t.upper() for t in (exclude_tiers or [])}
    resolutions_set = set(resolutions) if resolutions else None

    # ------------------------------------------------------------------
    # 1. Try loading from cache (version 3)
    # ------------------------------------------------------------------
    if cache_manifest_uri:
        cached = _read_json_uri(cache_manifest_uri)
        if cached and cached.get("version") == 3:
            cache_config = cached.get("config", {})
            cache_age = time.time() - float(cached.get("created_unix", 0) or 0)
            config_matches = (
                cache_config.get("bucket") == bucket_name
                and cache_config.get("original_bucket") == original_bucket
            )
            if config_matches and cache_age <= cache_max_age_hours * 3600:
                samples = _res_variant_cache_to_samples(
                    cached, bucket_name, original_bucket,
                    exclude_tiers_norm, resolutions_set,
                )
                log.info(
                    "Loaded %d VisoMaster res-variant samples from cache: %s",
                    len(samples), cache_manifest_uri,
                )
                return samples
            else:
                reason = "config_mismatch" if not config_matches else "expired"
                log.info(
                    "VisoMaster res-variant cache bypassed (%s): %s",
                    reason, cache_manifest_uri,
                )

    # ------------------------------------------------------------------
    # 2. Discover manifests — filter for swapper_resolution field
    # ------------------------------------------------------------------
    log.info("Discovering VisoMaster res-variant samples from gs://%s ...", bucket_name)

    manifests: Dict[str, Dict[str, Any]] = {}
    for blob in bucket.list_blobs(prefix="samples/"):
        if not blob.name.endswith("manifest.json"):
            continue
        parts = blob.name.split("/")
        if len(parts) < 3:
            continue
        try:
            manifest = json.loads(blob.download_as_text())
        except Exception as exc:
            log.warning("Failed to parse res-variant manifest %s: %s", blob.name, exc)
            continue

        # Only include samples that have a swapper_resolution field —
        # this distinguishes them from post-hoc enhanced samples.
        if "swapper_resolution" not in manifest:
            continue

        sid = manifest.get("sample_id") or parts[1]
        manifests[sid] = manifest

    log.info(
        "Discovered %d res-variant manifests in gs://%s", len(manifests), bucket_name,
    )

    # ------------------------------------------------------------------
    # 3. Build sample objects with filtering
    # ------------------------------------------------------------------
    samples: List[VisoMasterEnhancedSample] = []
    res_counts: Dict[int, int] = Counter()
    tier_counts: Dict[str, int] = Counter()
    skipped_tier = 0
    skipped_res = 0

    for sid, manifest in sorted(manifests.items()):
        swapper_resolution = int(manifest.get("swapper_resolution", 0) or 0)
        if resolutions_set and swapper_resolution not in resolutions_set:
            skipped_res += 1
            continue

        # Tier data — may be nested under tier_data or flat at top-level
        tier_data = manifest.get("tier_data", {}) or {}
        tier = tier_data.get("tier", manifest.get("tier", "UNKNOWN")).upper()
        if tier in exclude_tiers_norm:
            skipped_tier += 1
            continue

        _id_raw = tier_data.get("identity_delta") if tier_data.get("identity_delta") is not None else manifest.get("identity_delta")
        identity_delta = float(_id_raw if _id_raw is not None else -1.0)
        _ad_raw = tier_data.get("artifact_delta") if tier_data.get("artifact_delta") is not None else manifest.get("artifact_delta")
        artifact_delta = float(_ad_raw if _ad_raw is not None else -1.0)

        swap_model = manifest.get("swap_model", "Inswapper128")
        original_sample_id = manifest.get("original_sample_id", "")

        # Resolve original_video_name for identity extraction
        gcs_paths = manifest.get("gcs_paths", {}) or {}
        original_video_name = manifest.get("original_video_name", "")
        if not original_video_name:
            original_video_name = gcs_paths.get("original_video_name", "")

        # Pseudo-enhancer encodes resolution for the loading pipeline
        pseudo_enhancer = f"inswapper128_res{swapper_resolution}"

        sample = VisoMasterEnhancedSample(
            sample_id=sid,
            original_sample_id=original_sample_id,
            swap_model=swap_model,
            enhancer=pseudo_enhancer,
            frame_count=int(manifest.get("frame_count") or 16),
            tier=tier,
            identity_delta=identity_delta,
            artifact_delta=artifact_delta,
            enhanced_bucket=bucket_name,
            original_bucket=original_bucket,
            manifest={
                "original_video_name": original_video_name,
                "swapper_resolution": swapper_resolution,
                **{k: v for k, v in manifest.items() if k not in ("gcs_paths",)},
            },
        )
        samples.append(sample)
        res_counts[swapper_resolution] += 1
        tier_counts[tier] += 1

    log.info(
        "VisoMaster res-variant samples after filtering: %d "
        "(resolutions=%s, tiers=%s, skipped_tier=%d, skipped_res=%d)",
        len(samples), dict(res_counts), dict(tier_counts),
        skipped_tier, skipped_res,
    )

    # ------------------------------------------------------------------
    # 4. Write cache (version 3)
    # ------------------------------------------------------------------
    if cache_manifest_uri:
        cache_rows = []
        for s in samples:
            cache_rows.append({
                "sample_id": s.sample_id,
                "original_sample_id": s.original_sample_id,
                "swap_model": s.swap_model,
                "enhancer": s.enhancer,
                "frame_count": s.frame_count,
                "tier": s.tier,
                "identity_delta": s.identity_delta,
                "artifact_delta": s.artifact_delta,
                "original_video_name": s.manifest.get("original_video_name", ""),
                "swapper_resolution": s.manifest.get("swapper_resolution", 0),
            })
        cache_payload = {
            "version": 3,
            "created_unix": int(time.time()),
            "config": {
                "bucket": bucket_name,
                "original_bucket": original_bucket,
            },
            "samples": cache_rows,
        }
        _write_json_uri(cache_manifest_uri, cache_payload, log=log)

    return samples


def _res_variant_cache_to_samples(
    cache_payload: Dict[str, Any],
    bucket_name: str,
    original_bucket: str,
    exclude_tiers: set,
    resolutions_set: Optional[set],
) -> List[VisoMasterEnhancedSample]:
    """Deserialise resolution-variant samples from a version-3 cache."""
    samples: List[VisoMasterEnhancedSample] = []
    for row in cache_payload.get("samples", []) or []:
        tier = row.get("tier", "UNKNOWN").upper()
        if tier in exclude_tiers:
            continue
        swapper_resolution = int(row.get("swapper_resolution", 0) or 0)
        if resolutions_set and swapper_resolution not in resolutions_set:
            continue
        enhancer = row.get("enhancer", "")
        original_video_name = row.get("original_video_name", "")
        samples.append(VisoMasterEnhancedSample(
            sample_id=row["sample_id"],
            original_sample_id=row.get("original_sample_id", ""),
            swap_model=row.get("swap_model", "Inswapper128"),
            enhancer=enhancer,
            frame_count=int(row.get("frame_count") or 16),
            tier=tier,
            identity_delta=float(row.get("identity_delta") or -1.0),
            artifact_delta=float(row.get("artifact_delta") or -1.0),
            enhanced_bucket=bucket_name,
            original_bucket=original_bucket,
            manifest={
                "original_video_name": original_video_name,
                "swapper_resolution": swapper_resolution,
            },
        ))
    return samples


def load_visomaster_enhanced_frames(
    sample: VisoMasterEnhancedSample,
    anchor_indices: List[int],
    as_array: bool = True,
    client: Optional[Any] = None,
    executor: Optional[Any] = None,
    parallel_download_workers: int = DEFAULT_PARALLEL_FRAME_DOWNLOAD_WORKERS,
) -> Tuple[List[Optional[Any]], List[Optional[Any]]]:
    """
    Load paired real/enhanced-fake frames for a VisoMaster enhanced sample.

    **Cross-bucket loading**: real frames from the original bucket,
    enhanced fake frames from the enhanced bucket.

    Args:
        sample: VisoMasterEnhancedSample to load.
        anchor_indices: Frame indices to load.
        as_array: If True, return numpy arrays; otherwise PIL images.
        client: Optional reusable GCS client.
        executor: Optional reusable thread pool for parallel downloads.
        parallel_download_workers: Fallback worker count when no executor is supplied.

    Returns:
        Tuple of aligned real/enhanced-fake frame lists.
    """
    from google.cloud import storage as gcs_storage

    if client is None:
        client = gcs_storage.Client()
    original_bucket = client.bucket(sample.original_bucket)
    enhanced_bucket = client.bucket(sample.enhanced_bucket)

    def _load_pair(idx: int) -> Tuple[Any, Any]:
        real_blob_path = f"samples/{sample.original_sample_id}/frames/real/frame_{idx:04d}.png"
        fake_blob_path = f"samples/{sample.sample_id}/frames/fake/frame_{idx:04d}.png"
        return (
            _load_blob_image(original_bucket, real_blob_path, as_array=as_array),
            _load_blob_image(enhanced_bucket, fake_blob_path, as_array=as_array),
        )

    return _load_paired_frames_in_order(
        anchor_indices,
        sample.frame_count,
        _load_pair,
        sample_id=sample.sample_id,
        source_name="visomaster_enhanced",
        executor=executor,
        parallel_download_workers=parallel_download_workers,
    )


def load_visomaster_teams_enhanced_frames(
    sample: VisoMasterTeamsEnhancedSample,
    anchor_indices: List[int],
    fake_branch: str = "original",
    as_array: bool = True,
    client: Optional[Any] = None,
    executor: Optional[Any] = None,
    parallel_download_workers: int = DEFAULT_PARALLEL_FRAME_DOWNLOAD_WORKERS,
) -> Tuple[List[Optional[Any]], List[Optional[Any]]]:
    """
    Load paired frames for the merged Teams-enhanced VisoMaster source.

    Real frames always come from the resolved companion bucket. Fake frames come
    either from the resolved companion fake branch (``fake_branch='original'``)
    or from one selected enhancer folder in ``enhanced_bucket``.
    """
    from google.cloud import storage as gcs_storage

    branch = str(fake_branch or "original").strip().lower()
    if branch != "original" and branch not in set(sample.available_enhancers):
        raise ValueError(
            f"Unknown VisoMaster Teams-enhanced branch '{fake_branch}' for sample {sample.sample_id}"
        )

    if client is None:
        client = gcs_storage.Client()
    companion_bucket = client.bucket(sample.companion_bucket)
    enhanced_bucket = client.bucket(sample.enhanced_bucket)

    def _load_pair(idx: int) -> Tuple[Any, Any]:
        real_prefix = f"samples/{sample.sample_id}/frames/real/frame_{idx:04d}"
        real_img = _load_frame_from_bucket(
            companion_bucket,
            real_prefix,
            preferred_ext=sample.companion_real_ext,
            as_array=as_array,
        )

        if branch == "original":
            fake_prefix = f"samples/{sample.sample_id}/frames/fake/frame_{idx:04d}"
            fake_img = _load_frame_from_bucket(
                companion_bucket,
                fake_prefix,
                preferred_ext=sample.companion_fake_ext,
                as_array=as_array,
            )
        else:
            fake_prefix = f"samples/{sample.sample_id}/frames/{branch}/frame_{idx:04d}"
            fake_img = _load_frame_from_bucket(
                enhanced_bucket,
                fake_prefix,
                preferred_ext=".png",
                as_array=as_array,
            )

        return real_img, fake_img

    return _load_paired_frames_in_order(
        anchor_indices,
        sample.frame_count,
        _load_pair,
        sample_id=sample.sample_id,
        source_name=f"visomaster_teams_enhanced[{branch}]",
        executor=executor,
        parallel_download_workers=parallel_download_workers,
    )


# =============================================================================
# VisoMaster Iterable Dataset (standalone usage)
# =============================================================================

class VisoMasterIterableDataset(IterableDataset):
    """
    Iterable dataset for VisoMaster paired real/fake frames.

    Used for standalone 'visomaster' data source. For combined_paired usage,
    VisoMaster samples are integrated into CombinedPairedIterableDataset instead.
    """

    def __init__(
        self,
        samples: List[VisoMasterSample],
        anchor_indices: List[int],
        batch_size: int = 32,
        transform: Optional[Callable] = None,
        shuffle: bool = True,
        seed: int = 42,
        identity_balanced: bool = True,
        parallel_download_workers: int = DEFAULT_PARALLEL_FRAME_DOWNLOAD_WORKERS,
    ):
        self.samples = samples
        self.anchor_indices = anchor_indices
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self.identity_balanced = identity_balanced
        self.batch_size = batch_size
        self.parallel_download_workers = max(1, int(parallel_download_workers or 1))

        # Group by identity for balanced sampling
        self._by_identity: Dict[str, List[VisoMasterSample]] = defaultdict(list)
        for s in samples:
            self._by_identity[f"visomaster_{s.identity}"].append(s)
        self._identities = list(self._by_identity.keys())

        model_counts = Counter(s.swap_model for s in samples)
        logger.info(
            "VisoMasterIterableDataset: %d samples, %d identities, models=%s",
            len(samples), len(self._identities), dict(model_counts),
        )

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _get_gcs_client(self):
        if not hasattr(self, "_gcs_client"):
            from google.cloud import storage as gcs_storage
            self._gcs_client = gcs_storage.Client()
        return self._gcs_client

    def _get_download_executor(self):
        if self.parallel_download_workers <= 1:
            return None
        executor = getattr(self, "_download_executor", None)
        current_workers = getattr(self, "_download_executor_workers", None)
        if executor is None or current_workers != self.parallel_download_workers:
            self._download_executor = ThreadPoolExecutor(
                max_workers=self.parallel_download_workers,
                thread_name_prefix="visomaster-gcs",
            )
            self._download_executor_workers = self.parallel_download_workers
        return self._download_executor

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            rng = random.Random(self.seed + self._epoch * 1000 + worker_id)
        else:
            worker_id = 0
            num_workers = 1
            rng = random.Random(self.seed + self._epoch * 1000)

        if self.identity_balanced:
            samples_to_iterate = []
            for identity in self._identities:
                chosen = rng.choice(self._by_identity[identity])
                samples_to_iterate.append(chosen)
            if self.shuffle:
                rng.shuffle(samples_to_iterate)
            samples_to_iterate = samples_to_iterate[worker_id::num_workers]
        else:
            samples_to_iterate = self.samples[worker_id::num_workers]
            if self.shuffle:
                samples_to_iterate = list(samples_to_iterate)
                rng.shuffle(samples_to_iterate)

        client = self._get_gcs_client()
        executor = self._get_download_executor()

        for sample in samples_to_iterate:
            try:
                real_frames, fake_frames = load_visomaster_frames(
                    sample,
                    self.anchor_indices,
                    as_array=True,
                    client=client,
                    executor=executor,
                    parallel_download_workers=self.parallel_download_workers,
                )
            except Exception as e:
                logger.warning("Failed to load VisoMaster sample %s: %s", sample.sample_id, e)
                continue

            identity = f"visomaster_{sample.identity}"
            method = f"visomaster_{sample.swap_model}"

            for i, idx in enumerate(self.anchor_indices):
                if i >= len(real_frames) or i >= len(fake_frames):
                    continue
                if real_frames[i] is None or fake_frames[i] is None:
                    continue

                real_img = real_frames[i]
                if self.transform:
                    real_img = self.transform(real_img, None)

                yield {
                    "image": real_img,
                    "label": 0,
                    "identity": identity,
                    "source": "visomaster",
                    "method": "visomaster_real",
                    "sample_id": sample.sample_id,
                    "frame_idx": idx,
                }

                fake_img = fake_frames[i]
                if self.transform:
                    fake_img = self.transform(fake_img, None)

                yield {
                    "image": fake_img,
                    "label": 1,
                    "identity": identity,
                    "source": "visomaster",
                    "method": method,
                    "sample_id": sample.sample_id,
                    "frame_idx": idx,
                }


# =============================================================================
# Standalone Data Source Registration
# =============================================================================

@register_data_source('visomaster')
def create_visomaster_pipeline(
    config: Dict[str, Any],
    data_config: Dict[str, Any],
    logger_inst: logging.Logger,
    **kwargs,
) -> DataPipelineResult:
    """
    Create a standalone VisoMaster training pipeline.

    Supports filtering by swap_model and tier, identity-stratified splitting,
    and identity-balanced sampling.
    """
    viso_config = data_config.get("visomaster", {})
    seed = config.get("manualSeed", config.get("seed", 737))

    bucket_name = viso_config.get(
        "gcs_bucket", "live-deepfake-methods-real-and-fake-frames-cropped"
    )
    frames_bucket = viso_config.get(
        "frames_bucket", "live-deepfake-methods-real-and-fake-frames"
    )
    gcs_project = viso_config.get(
        "gcs_project", os.environ.get("GOOGLE_CLOUD_PROJECT", "train-cvit2")
    )
    swap_models = viso_config.get("swap_models")  # None = all
    tiers = viso_config.get("tiers")  # None = all

    logger_inst.info("=" * 70)
    logger_inst.info("VisoMaster Data Source: Initializing")
    logger_inst.info("  - Bucket: %s", bucket_name)
    logger_inst.info("  - Swap models filter: %s", swap_models or "ALL")
    logger_inst.info("  - Tiers filter: %s", tiers or "ALL")
    logger_inst.info("=" * 70)

    # Discover samples
    samples = discover_visomaster_samples(
        bucket_name=bucket_name,
        frames_bucket_name=frames_bucket,
        gcs_project=gcs_project,
        swap_models=swap_models,
        tiers=tiers,
        log=logger_inst,
    )

    if not samples:
        raise ValueError("No VisoMaster samples found with the given filters!")

    # Identity-stratified splitting
    train_split = viso_config.get("train_split", 0.7)
    val_split = viso_config.get("val_split", 0.15)

    rng = random.Random(seed)
    by_identity: Dict[str, List[VisoMasterSample]] = defaultdict(list)
    for s in samples:
        by_identity[f"visomaster_{s.identity}"].append(s)

    identities = list(by_identity.keys())
    rng.shuffle(identities)

    n_ids = len(identities)
    n_train = int(n_ids * train_split)
    n_val = int(n_ids * val_split)

    train_ids = set(identities[:n_train])
    val_ids = set(identities[n_train : n_train + n_val])
    test_ids = set(identities[n_train + n_val :])

    train_samples = [s for s in samples if f"visomaster_{s.identity}" in train_ids]
    val_samples = [s for s in samples if f"visomaster_{s.identity}" in val_ids]
    test_samples = [s for s in samples if f"visomaster_{s.identity}" in test_ids]

    logger_inst.info("VisoMaster split (by identity, seed=%d):", seed)
    logger_inst.info("  - Identities: %d train / %d val / %d test", len(train_ids), len(val_ids), len(test_ids))
    logger_inst.info("  - Samples:    %d train / %d val / %d test", len(train_samples), len(val_samples), len(test_samples))

    # Create datasets
    anchor_indices = viso_config.get("anchor_indices", [0, 2, 4, 6, 8, 10, 12, 14])
    batch_size = config.get("frames_per_batch", 32)
    identity_balanced = viso_config.get("identity_balanced_sampling", True)

    transform = kwargs.get("transform")

    train_ds = VisoMasterIterableDataset(
        samples=train_samples,
        anchor_indices=anchor_indices,
        batch_size=batch_size,
        transform=transform,
        shuffle=True,
        seed=seed,
        identity_balanced=identity_balanced,
        parallel_download_workers=viso_config.get(
            "parallel_download_workers",
            DEFAULT_PARALLEL_FRAME_DOWNLOAD_WORKERS,
        ),
    )

    no_mp = os.environ.get("NO_MULTIPROCESSING", "").lower() in ("1", "true", "yes")
    num_workers = 0 if (no_mp or not torch.cuda.is_available()) else viso_config.get("num_workers", 4)
    prefetch = config.get("prefetch_factor", 2) if num_workers > 0 else 2

    # Import collate from combined_paired (same format)
    from .combined_paired import combined_paired_collate_fn

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch if num_workers > 0 else None,
        collate_fn=combined_paired_collate_fn,
        pin_memory=True,
    )

    val_ds = VisoMasterIterableDataset(
        samples=val_samples,
        anchor_indices=anchor_indices,
        batch_size=batch_size,
        transform=None,
        shuffle=False,
        seed=seed,
        identity_balanced=False,
        parallel_download_workers=viso_config.get(
            "parallel_download_workers",
            DEFAULT_PARALLEL_FRAME_DOWNLOAD_WORKERS,
        ),
    )
    val_loader_raw = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch if num_workers > 0 else None,
        collate_fn=combined_paired_collate_fn,
        pin_memory=True,
    )

    # Use CombinedValidationAdapter for compatibility
    from .combined_paired import CombinedValidationAdapter

    val_loader = CombinedValidationAdapter(val_loader_raw, val_samples, "visomaster_val")

    # Build method mapping for Group DRO
    unique_methods = sorted(set(f"visomaster_{s.swap_model}" for s in samples))
    method_mapping = {m: i for i, m in enumerate(unique_methods)}

    unique_identities = set(f"visomaster_{s.identity}" for s in train_samples)
    frames_per_sample = len(anchor_indices) * 2  # real + fake

    data_stats = {
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "visomaster_samples": len(samples),
        "total_identities": n_ids,
        "train_identities": len(train_ids),
        "identity_balanced_sampling": identity_balanced,
        "has_landmarks": False,
        "discovered_videos": len(samples),
        "discovered_methods": len(unique_methods),
        "unbalanced_train_count": len(train_samples),
        "unbalanced_val_count": len(val_samples) + len(test_samples),
        "train_video_count": len(train_samples),
        "train_frame_count": len(train_samples) * frames_per_sample,
        "val_video_count": len(val_samples),
        "val_frame_count": len(val_samples) * frames_per_sample,
        "train_split": train_split,
        "methods": unique_methods,
        "method_mapping": method_mapping,
        "swap_model_counts": dict(Counter(s.swap_model for s in train_samples)),
        "tier_counts": dict(Counter(s.tier for s in train_samples)),
    }

    logger_inst.info("=" * 70)
    logger_inst.info("VisoMaster pipeline created successfully")
    logger_inst.info("  - Train: %d samples, %d identities", len(train_samples), len(unique_identities))
    logger_inst.info("  - Val: %d samples", len(val_samples))
    logger_inst.info("  - Test: %d samples", len(test_samples))
    logger_inst.info("  - Methods: %s", unique_methods)
    logger_inst.info("=" * 70)

    return DataPipelineResult(
        train_loader=train_loader,
        val_in_dist_loader=val_loader,
        val_holdout_loader=None,
        train_samples=train_samples,
        data_stats=data_stats,
        ood_loader=None,
    )
