"""Substrate-paired inventory data lane (2026-05-22).

Adds two new "real-only" data sources whose frames sit in clean and teams
substrates for the same identity:

  - ``hdtf_visomaster_teams``     — 1,094 paired identity rows (HDTF base)
  - ``quickclips_visomaster_teams`` — 732 paired identity rows (QCLIP base)

Together with the 54 ``visomaster_teams_enhanced`` rows already wired by
``data/sources/visomaster.py::discover_visomaster_teams_enhanced_samples``,
these complete the A0.1 inventory of 1,880 substrate-paired identity rows
recorded in ``analysis/substrate_pair_geometry_2026-05-22/inventory_manifest.csv``.

For each inventory row we emit TWO :class:`UnifiedPairedSample` objects:

  * one ``substrate_transport='clean'`` row keyed
    ``source='hdtf_visomaster' / 'quickclips_visomaster'`` (no ``_teams``
    suffix so the SubstratePairStamper resolves it to ``transport=0``).
  * one ``substrate_transport='teams'`` row keyed
    ``source='hdtf_visomaster_teams' / 'quickclips_visomaster_teams'``
    (``_teams`` suffix triggers ``transport=1`` in the stamper).

Both rows share the same ``identity = realpool_<inventory.identity_id>`` so
the per-video collate ends up with matching ``substrate_pair_id`` for the
clean and teams sides.

Frames are loaded from the GCS buckets recorded in the inventory CSV:

  * hdtf clean  : gs://hdtf_visomaster_cropped_frames
  * hdtf teams  : gs://hdtf_visomaster_cropped_frames_teams
  * quickclips clean : gs://quickclips_visomaster_cropped_frames
  * quickclips teams : gs://quickclips_visomaster_cropped_frames_teams

Each ``samples/<base_capture_id>/frames/real/frame_NNNN.png`` PNG is fetched
on demand by the iterator using the same ``_load_blob_image`` helper used by
``data/sources/visomaster.py``.

Wired into ``data/sources/combined_paired.py`` via:

  - ``CombinedBatchingConfig.substrate_paired_sparse_indices``
  - ``CombinedPairedIterableDataset._iterate_substrate_paired_sample``
  - ``CombinedPairedIterableDataset.__iter__`` source dispatch
  - ``create_combined_paired_pipeline`` discovery + sample creation
"""
from __future__ import annotations

import csv
import logging
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_PARALLEL_FRAME_DOWNLOAD_WORKERS = 4
DEFAULT_INVENTORY_PATH = (
    "analysis/substrate_pair_geometry_2026-05-22/inventory_manifest.csv"
)

# Inventory source labels (column `source` in the CSV).
INVENTORY_SOURCE_HDTF = "hdtf_visomaster_teams"
INVENTORY_SOURCE_QCLIP = "quickclips_visomaster_teams"
INVENTORY_SOURCE_VISO_TEAMS_ENH = "visomaster_teams_enhanced"

# Yield-time source labels emitted to the per-frame dict. The teams labels
# are intentionally identical to the inventory source labels so the
# SubstratePairStamper's ``_is_teams_source`` substring check resolves to
# ``transport=TEAMS``; the clean labels drop the ``_teams`` suffix so the
# stamper resolves to ``transport=CLEAN``.
CLEAN_SOURCE_HDTF = "hdtf_visomaster"
CLEAN_SOURCE_QCLIP = "quickclips_visomaster"
TEAMS_SOURCE_HDTF = INVENTORY_SOURCE_HDTF
TEAMS_SOURCE_QCLIP = INVENTORY_SOURCE_QCLIP

# Method label emitted per yield row. Empty/unique enough that the existing
# method_mapping in CombinedPairedIterableDataset falls through to method_id=-1.
METHOD_CLEAN_REAL = "substrate_paired_clean_real"
METHOD_TEAMS_REAL = "substrate_paired_teams_real"

# Sentinel sample_id suffixes used to keep clean and teams sides as separate
# `video_id` keys inside ``combined_paired_collate_fn``.
SAMPLE_ID_SUFFIX_CLEAN = "__clean"
SAMPLE_ID_SUFFIX_TEAMS = "__teams"


@dataclass
class SubstratePairedInventorySample:
    """Single inventory-row pair.

    Each instance encodes both substrate sides; the consumer materializes
    two UnifiedPairedSample wrappers (one per transport) so the
    CombinedPairedIterableDataset treats them as two videos sharing identity.
    """

    identity_id: str            # raw inventory identity (e.g., 'RD_Radio14').
    base_capture_id: str        # e.g. 'HDTF20260416_00000'.
    source: str                 # inventory `source` column (e.g., 'hdtf_visomaster_teams').
    clean_bucket: str           # GCS bucket name (no gs:// prefix).
    clean_prefix: str           # bucket key prefix, trailing slash.
    clean_frame_count: int
    teams_bucket: str
    teams_prefix: str
    teams_frame_count: int
    notes: str = ""
    # Computed per-row source labels emitted by the iterator.
    clean_source_label: str = ""
    teams_source_label: str = ""

    @property
    def identity_with_prefix(self) -> str:
        """Match the realpool_ convention used by DeepLive/VisoMaster."""
        return f"realpool_{self.identity_id}"

    @property
    def clean_sample_id(self) -> str:
        return f"{self.base_capture_id}{SAMPLE_ID_SUFFIX_CLEAN}"

    @property
    def teams_sample_id(self) -> str:
        return f"{self.base_capture_id}{SAMPLE_ID_SUFFIX_TEAMS}"

    @property
    def clean_method(self) -> str:
        # Per-source method so dev metrics surface per-source.
        return f"{self.clean_source_label}_real"

    @property
    def teams_method(self) -> str:
        return f"{self.teams_source_label}_real"

    @property
    def frame_count(self) -> int:
        """Conservative frame_count used by the sparse-index iterator."""
        return min(int(self.clean_frame_count), int(self.teams_frame_count))


# =============================================================================
# Inventory discovery
# =============================================================================


def _resolve_inventory_path(inventory_path: Optional[str]) -> str:
    path = inventory_path or DEFAULT_INVENTORY_PATH
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    return path


def discover_substrate_paired_samples(
    inventory_path: Optional[str] = None,
    sources: Optional[List[str]] = None,
    log: Optional[logging.Logger] = None,
) -> List[SubstratePairedInventorySample]:
    """Read inventory CSV and return one row per (identity, base_capture).

    Args:
        inventory_path: Repo-relative or absolute path to the CSV. Defaults
            to ``DEFAULT_INVENTORY_PATH``.
        sources: Optional list of inventory `source` labels to keep. When
            ``None`` we keep ``hdtf_visomaster_teams`` and
            ``quickclips_visomaster_teams`` — the two sources this module
            wires. ``visomaster_teams_enhanced`` is intentionally filtered
            out because its clean+teams pair is already produced by
            ``_iterate_visomaster_teams_enhanced_sample`` in
            ``data/sources/combined_paired.py``.
        log: Optional logger. Defaults to module logger.

    Returns:
        List of :class:`SubstratePairedInventorySample`.
    """
    log = log or logger
    path = _resolve_inventory_path(inventory_path)
    if not os.path.exists(path):
        log.warning(
            "Substrate-paired inventory CSV not found at %s — emitting 0 samples.",
            path,
        )
        return []

    # Default to the two non-enhanced sources. The enhanced rows already flow
    # through visomaster_teams_enhanced.
    if sources is None:
        sources_set = {INVENTORY_SOURCE_HDTF, INVENTORY_SOURCE_QCLIP}
    else:
        sources_set = set(sources)

    samples: List[SubstratePairedInventorySample] = []
    skipped_unknown_source = 0
    skipped_missing_buckets = 0
    skipped_zero_frames = 0
    per_source_count: Dict[str, int] = Counter()

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            inv_source = str(row.get("source", "")).strip()
            if not inv_source:
                continue
            if inv_source not in sources_set:
                skipped_unknown_source += 1
                continue

            identity_id = str(row.get("identity_id", "")).strip()
            base_capture_id = str(row.get("base_capture_id", "")).strip()
            clean_bucket = str(row.get("clean_bucket", "")).strip()
            clean_prefix = str(row.get("clean_prefix", "")).strip()
            teams_bucket = str(row.get("teams_bucket", "")).strip()
            teams_prefix = str(row.get("teams_prefix", "")).strip()
            if not (clean_bucket and clean_prefix and teams_bucket and teams_prefix):
                skipped_missing_buckets += 1
                continue
            if not identity_id or not base_capture_id:
                skipped_missing_buckets += 1
                continue

            try:
                clean_frame_count = int(row.get("clean_real_frame_count") or 0)
            except (TypeError, ValueError):
                clean_frame_count = 0
            try:
                teams_frame_count = int(row.get("teams_real_frame_count") or 0)
            except (TypeError, ValueError):
                teams_frame_count = 0
            if clean_frame_count <= 0 or teams_frame_count <= 0:
                skipped_zero_frames += 1
                continue

            if inv_source == INVENTORY_SOURCE_HDTF:
                clean_label = CLEAN_SOURCE_HDTF
                teams_label = TEAMS_SOURCE_HDTF
            elif inv_source == INVENTORY_SOURCE_QCLIP:
                clean_label = CLEAN_SOURCE_QCLIP
                teams_label = TEAMS_SOURCE_QCLIP
            else:
                # Future-proof: any inventory source we accept but don't
                # have an explicit clean-label mapping for falls back to a
                # generic ``<inv_source>_clean`` label.
                clean_label = inv_source.replace("_teams", "")
                if clean_label == inv_source:
                    clean_label = f"{inv_source}_clean"
                teams_label = inv_source

            sample = SubstratePairedInventorySample(
                identity_id=identity_id,
                base_capture_id=base_capture_id,
                source=inv_source,
                clean_bucket=clean_bucket,
                clean_prefix=clean_prefix,
                clean_frame_count=clean_frame_count,
                teams_bucket=teams_bucket,
                teams_prefix=teams_prefix,
                teams_frame_count=teams_frame_count,
                notes=str(row.get("notes") or ""),
                clean_source_label=clean_label,
                teams_source_label=teams_label,
            )
            samples.append(sample)
            per_source_count[inv_source] += 1

    log.info(
        "Discovered %d substrate-paired inventory rows from %s "
        "(per_source=%s; skipped_unknown_source=%d, skipped_missing_buckets=%d, "
        "skipped_zero_frames=%d).",
        len(samples),
        path,
        dict(per_source_count),
        skipped_unknown_source,
        skipped_missing_buckets,
        skipped_zero_frames,
    )
    return samples


# =============================================================================
# Frame I/O
# =============================================================================


def _load_one_frame_bytes(bucket: Any, blob_path: str) -> Any:
    """Return RGB ndarray for a GCS blob (PNG/JPG-tolerant)."""
    import io

    import numpy as np
    from PIL import Image

    data = bucket.blob(blob_path).download_as_bytes()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)


def load_substrate_paired_real_frames(
    sample: SubstratePairedInventorySample,
    side: str,
    frame_indices: List[int],
    *,
    client: Any,
    executor: Optional[Any] = None,
    parallel_download_workers: int = DEFAULT_PARALLEL_FRAME_DOWNLOAD_WORKERS,
) -> Dict[int, Any]:
    """Load PNG frames from one side (clean or teams) of an inventory row.

    Returns ``{frame_idx: np.ndarray}`` for successfully loaded frames; failed
    loads are skipped (logged at debug level).
    """
    if side == "clean":
        bucket_name = sample.clean_bucket
        prefix = sample.clean_prefix
        frame_count = sample.clean_frame_count
    elif side == "teams":
        bucket_name = sample.teams_bucket
        prefix = sample.teams_prefix
        frame_count = sample.teams_frame_count
    else:
        raise ValueError(f"Unknown side: {side!r} (expected 'clean' or 'teams')")

    bucket = client.bucket(bucket_name)

    valid_indices = [idx for idx in frame_indices if idx < frame_count]
    if not valid_indices:
        return {}

    def _load_one(idx: int) -> Tuple[int, Any]:
        blob_path = f"{prefix}frame_{idx:04d}.png"
        return idx, _load_one_frame_bytes(bucket, blob_path)

    def _log_failure(idx: int, exc: Exception) -> None:
        blob_path = f"{prefix}frame_{idx:04d}.png"
        logger.debug("Failed to load substrate-paired frame %s: %s", blob_path, exc)

    out: Dict[int, Any] = {}

    if executor is None:
        max_workers = max(1, int(parallel_download_workers or 1))
        if max_workers <= 1 or len(valid_indices) <= 1:
            for idx in valid_indices:
                try:
                    frame_idx, img = _load_one(idx)
                except Exception as exc:
                    _log_failure(idx, exc)
                    continue
                out[frame_idx] = img
            return out
        with ThreadPoolExecutor(max_workers=min(max_workers, len(valid_indices))) as pool:
            futures = {pool.submit(_load_one, idx): idx for idx in valid_indices}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    frame_idx, img = future.result()
                except Exception as exc:
                    _log_failure(idx, exc)
                    continue
                out[frame_idx] = img
        return out

    futures = {executor.submit(_load_one, idx): idx for idx in valid_indices}
    for future in as_completed(futures):
        idx = futures[future]
        try:
            frame_idx, img = future.result()
        except Exception as exc:
            _log_failure(idx, exc)
            continue
        out[frame_idx] = img
    return out


__all__ = [
    "DEFAULT_INVENTORY_PATH",
    "INVENTORY_SOURCE_HDTF",
    "INVENTORY_SOURCE_QCLIP",
    "INVENTORY_SOURCE_VISO_TEAMS_ENH",
    "CLEAN_SOURCE_HDTF",
    "CLEAN_SOURCE_QCLIP",
    "TEAMS_SOURCE_HDTF",
    "TEAMS_SOURCE_QCLIP",
    "SubstratePairedInventorySample",
    "discover_substrate_paired_samples",
    "load_substrate_paired_real_frames",
]
