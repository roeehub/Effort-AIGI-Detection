"""
Validation-only data loaders for custom sources (no parquet dependency).

This module provides helper functions to build VideoInfo lists for validation
from:
  - DF40 paired JSON (with orientation + mode filters)
  - External GCS buckets for real/fake validation pools (folder of frame sequences)
  - DeepLive GCS bucket (paired real/fake with strategy-based methods)
  - VisoMaster GCS bucket (paired real/fake with swap_model + tier metadata)
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from fsspec.core import url_to_fs

from prepare_splits import VideoInfo
from dataset.deeplive_dataset import resolve_effective_strategy

log = logging.getLogger(__name__)


def _normalize_method_name(name: str) -> str:
    """Normalize method/source names to lowercase with underscores."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _safe_identity(value: Optional[str], fallback_key: tuple) -> int:
    """Convert identity to int if possible; otherwise use a stable hash."""
    if value is not None:
        try:
            return int(value)
        except (ValueError, TypeError):
            pass
    return hash(fallback_key) & 0x7FFFFFFF


def _stable_int_hash(value: str) -> int:
    """Deterministic non-negative integer hash suitable for manifest identities."""
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def _stable_identity(value: Optional[object], fallback_key: tuple) -> int:
    """Convert identity to int if possible; otherwise use deterministic hashing."""
    if value is not None:
        try:
            return int(value)
        except (ValueError, TypeError):
            return _stable_int_hash(str(value))
    fallback = "||".join(str(part) for part in fallback_key)
    return _stable_int_hash(fallback)


def _video_id_from_path(path: str) -> str:
    """Extract the last folder name from a gs:// path."""
    return path.rstrip("/").split("/")[-1]


def _build_frame_paths(base_path: str, frames: Iterable[str]) -> List[str]:
    """Build full gs:// frame paths from a base folder and frame names."""
    base = base_path.rstrip("/")
    return [f"{base}/{frame}" for frame in frames]


def _shape_frames_deterministically(
    frames: Iterable[str],
    deterministic_frame_count: Optional[int] = None,
) -> List[str]:
    ordered = sorted(frames)
    if not ordered:
        return ordered
    if deterministic_frame_count is None:
        return ordered
    if len(ordered) >= deterministic_frame_count:
        return ordered[:deterministic_frame_count]
    return ordered + [ordered[-1]] * (deterministic_frame_count - len(ordered))


def _read_text_from_path(path: str) -> str:
    if path.startswith("gs://"):
        from google.cloud import storage

        uri = path.replace("gs://", "", 1)
        bucket_name, blob_path = uri.split("/", 1)
        client = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
        blob = client.bucket(bucket_name).blob(blob_path)
        return blob.download_as_text()

    with open(path, "r") as f:
        return f.read()


def _load_validation_manifest_rows(path: str) -> List[dict]:
    text = _read_text_from_path(path)
    stripped = text.strip()
    if not stripped:
        return []

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    if data is None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        jsonl_rows: List[dict] = []
        try:
            for line in lines:
                jsonl_rows.append(json.loads(line))
            data = jsonl_rows
        except json.JSONDecodeError:
            data = None

    if data is None:
        import yaml

        data = yaml.safe_load(text)

    if isinstance(data, dict):
        for key in ("videos", "entries", "samples"):
            rows = data.get(key)
            if isinstance(rows, list):
                return rows
        raise ValueError(
            f"Manifest {path} must contain one of: videos, entries, samples"
        )

    if isinstance(data, list):
        return data

    raise ValueError(f"Unsupported manifest format for {path}")


def _coerce_list(value: Optional[object]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def load_external_manifest_videos(
    manifest_path: str,
    label: Optional[str] = None,
    split: Optional[str] = None,
    slices: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    default_method: Optional[str] = None,
    default_label: Optional[str] = None,
    max_videos: Optional[int] = None,
    seed: int = 737,
    deterministic_frame_count: Optional[int] = None,
) -> List[VideoInfo]:
    """
    Load validation videos from a frozen manifest.

    Supported manifest layouts:
      - JSON list of row objects
      - JSON/YAML mapping with ``videos`` / ``entries`` / ``samples`` list
      - JSONL (one row object per line)

    Each row should provide:
      - ``frame_paths`` (preferred) or ``frame_path`` / ``path``
      - ``video_id`` (preferred) or ``sample_id`` / derived from path
      - ``method`` (or use ``default_method``)
      - ``label`` (or use ``default_label``)

    Optional row metadata used for filtering:
      - ``split``
      - ``slices`` / ``slice_tags`` / ``tags``
    """
    if label is not None and label not in {"real", "fake"}:
        raise ValueError(f"label must be 'real', 'fake', or None, got: {label}")
    if deterministic_frame_count is not None and deterministic_frame_count <= 0:
        raise ValueError(
            f"deterministic_frame_count must be > 0 when provided, got: {deterministic_frame_count}"
        )

    rows = _load_validation_manifest_rows(manifest_path)
    target_slices = {_normalize_method_name(item) for item in (slices or [])}
    target_methods = {_normalize_method_name(item) for item in (methods or [])}

    videos: List[VideoInfo] = []
    skipped = Counter()

    for idx, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Manifest row #{idx} in {manifest_path} must be an object.")

        row_label = str(row.get("label") or default_label or "").strip().lower()
        if not row_label:
            raise ValueError(f"Manifest row #{idx} missing label and no default_label provided.")
        if label and row_label != label:
            skipped["label"] += 1
            continue

        row_split = str(row.get("split") or "").strip()
        if split and row_split != split:
            skipped["split"] += 1
            continue

        row_slices = {
            _normalize_method_name(item)
            for item in (
                _coerce_list(row.get("slices"))
                + _coerce_list(row.get("slice_tags"))
                + _coerce_list(row.get("tags"))
            )
        }
        if target_slices and not row_slices.intersection(target_slices):
            skipped["slices"] += 1
            continue

        row_method = str(row.get("method") or default_method or "").strip()
        if not row_method:
            raise ValueError(f"Manifest row #{idx} missing method and no default_method provided.")
        if target_methods and _normalize_method_name(row_method) not in target_methods:
            skipped["methods"] += 1
            continue

        frame_paths = _coerce_list(row.get("frame_paths"))
        if not frame_paths:
            frame_paths = _coerce_list(row.get("frames"))
        if not frame_paths:
            single_path = row.get("frame_path") or row.get("path")
            frame_paths = _coerce_list(single_path)
        if not frame_paths:
            skipped["empty_frames"] += 1
            continue

        video_id = str(
            row.get("video_id")
            or row.get("sample_id")
            or row.get("id")
            or Path(frame_paths[0]).stem
        ).strip()
        shaped_frames = _shape_frames_deterministically(
            frame_paths, deterministic_frame_count=deterministic_frame_count
        )
        identity = _stable_identity(
            row.get("identity") or row.get("identity_key"),
            (row_method, video_id),
        )

        videos.append(
            VideoInfo(
                label=row_label,
                method=row_method,
                video_id=video_id,
                frame_paths=shaped_frames,
                identity=identity,
            )
        )

    total_before_sampling = len(videos)
    if max_videos and total_before_sampling > max_videos:
        import random

        rng = random.Random(seed)
        rng.shuffle(videos)
        videos = videos[:max_videos]
        log.info(
            "Sampled %d/%d manifest videos from %s (seed=%d)",
            max_videos,
            total_before_sampling,
            manifest_path,
            seed,
        )

    log.info(
        "Manifest validation videos loaded: path=%s total=%d skipped=%s label=%s split=%s slices=%s",
        manifest_path,
        len(videos),
        dict(sorted(skipped.items())),
        label,
        split,
        sorted(target_slices),
    )
    return videos


def load_df40_pairs_validation(
    pair_json_path: str,
    orientation: str = "all",
    mode: str = "paired",
    methods: Optional[List[str]] = None,
    real_method_override: Optional[str] = None,
) -> List[VideoInfo]:
    """
    Build a validation list from DF40 paired JSON with orientation and mode filters.

    Args:
        pair_json_path: Path to df40-pair-matching.json.
        orientation: 'all' | 'target_source' | 'source_target'.
        mode: 'paired' | 'fake_only' | 'real_only'.
        methods: Optional list of fake methods to include.
        real_method_override: Optional method name for real videos. If not set,
                              uses the JSON real.source normalized to match config.

    Returns:
        List[VideoInfo] for validation.
    """
    if orientation not in {"all", "target_source", "source_target"}:
        raise ValueError(f"Invalid orientation: {orientation}")
    if mode not in {"paired", "fake_only", "real_only"}:
        raise ValueError(f"Invalid mode: {mode}")

    with open(pair_json_path, "r") as f:
        data = json.load(f)

    method_orientation = data.get("method_orientation", {})
    pairs = data.get("pairs", [])

    videos: List[VideoInfo] = []
    counts = Counter()

    for pair in pairs:
        method = pair.get("method", "unknown")
        if methods and method not in methods:
            continue

        if orientation != "all":
            if method_orientation.get(method) != orientation:
                continue

        if mode in {"paired", "fake_only"}:
            fake = pair.get("fake", {})
            fake_path = fake.get("path", "")
            fake_frames = fake.get("frames", [])
            if fake_path and fake_frames:
                fake_video_id = _video_id_from_path(fake_path)
                fake_paths = _build_frame_paths(fake_path, fake_frames)
                identity = _safe_identity(pair.get("target_identity"), (method, fake_video_id))
                videos.append(
                    VideoInfo(
                        label="fake",
                        method=method,
                        video_id=fake_video_id,
                        frame_paths=fake_paths,
                        identity=identity,
                    )
                )
                counts["fake"] += 1

        if mode in {"paired", "real_only"}:
            real = pair.get("real", {})
            real_path = real.get("path", "")
            real_frames = real.get("frames", [])
            if real_path and real_frames:
                if real_method_override:
                    real_method = real_method_override
                else:
                    real_method = _normalize_method_name(real.get("source", "df40_real"))
                real_video_id = _video_id_from_path(real_path)
                if mode == "paired":
                    real_video_id = f"{real_video_id}__{pair.get('pair_id', real_video_id)}"
                real_paths = _build_frame_paths(real_path, real_frames)
                identity = _safe_identity(pair.get("target_identity"), (real_method, real_video_id))
                videos.append(
                    VideoInfo(
                        label="real",
                        method=real_method,
                        video_id=real_video_id,
                        frame_paths=real_paths,
                        identity=identity,
                    )
                )
                counts["real"] += 1

    log.info(
        "DF40 validation videos loaded: mode=%s orientation=%s total=%d (fake=%d real=%d)",
        mode,
        orientation,
        len(videos),
        counts.get("fake", 0),
        counts.get("real", 0),
    )
    return videos


def load_external_real_videos(
    bucket_name: str,
    prefix: str,
    method_name: str,
    cache_manifest_path: Optional[str] = None,
    allowed_exts: Optional[Iterable[str]] = None,
    max_videos: Optional[int] = None,
    seed: int = 737,
    label: str = "real",
    grouping: str = "by_folder",
    deterministic_frame_count: Optional[int] = None,
    path_contains: Optional[str] = None,
    path_exclude_contains: Optional[Iterable[str]] = None,
    video_id_depth: int = -2,
) -> List[VideoInfo]:
    """
    Load external videos from a GCS bucket path with frame folders.

    Expected structure:
        gs://<bucket>/<prefix>/<video_id>/<frame>.png

    For nested structures like Teams v2::

        gs://<bucket>/samples/<sample_id>/frames/real/<frame>.jpg

    Use ``path_contains="/frames/real/"`` and ``video_id_depth=-4``.

    Args:
        bucket_name: GCS bucket name (no gs:// prefix).
        prefix: Path prefix under bucket (e.g., real/external_youtube_avspeech).
        method_name: Method name to assign to all videos.
        cache_manifest_path: Optional local JSON cache of frame paths.
        allowed_exts: Optional iterable of allowed extensions.
        max_videos: Optional max number of videos to sample. None = all.
        seed: Random seed for reproducible sampling (default 737).
        label: Video label ('real' or 'fake').
        grouping:
            - "by_folder" (default): all frames under <prefix>/<video_id>/ belong to one video
            - "per_image": each image path is treated as its own video sample
        deterministic_frame_count:
            If set, each emitted VideoInfo is forced to exactly this number of frame
            paths by deterministic trimming/padding. Useful to avoid random frame
            subsampling in downstream validation.
        path_contains: Optional substring filter applied to discovered frame paths.
            Only paths containing this substring are retained.  Useful for nested
            bucket layouts (e.g., ``path_contains="/frames/real/"`` to select only
            real frames from a bucket that stores both real and fake under each sample).
        video_id_depth: Which path component (from the end) to use as the video ID
            when ``grouping="by_folder"``.  Default ``-2`` (the parent directory of
            each frame).  For deeper nesting, use e.g. ``-4`` so that
            ``samples/<sample_id>/frames/real/frame.jpg`` extracts ``<sample_id>``.

    Returns:
        List[VideoInfo] objects.
    """
    if label not in {"real", "fake"}:
        raise ValueError(f"label must be 'real' or 'fake', got: {label}")
    if grouping not in {"by_folder", "per_image"}:
        raise ValueError(f"grouping must be 'by_folder' or 'per_image', got: {grouping}")
    if deterministic_frame_count is not None and deterministic_frame_count <= 0:
        raise ValueError(
            f"deterministic_frame_count must be > 0 when provided, got: {deterministic_frame_count}"
        )

    if allowed_exts is None:
        allowed_exts = (".png", ".jpg", ".jpeg")

    base_path = f"gs://{bucket_name}/{prefix}".rstrip("/")

    frame_paths: List[str] = []
    if cache_manifest_path and os.path.exists(cache_manifest_path):
        with open(cache_manifest_path, "r") as f:
            frame_paths = json.load(f)
        log.info("Loaded %d frame paths from cache: %s", len(frame_paths), cache_manifest_path)
    else:
        fs = url_to_fs(base_path)[0]
        raw_paths = fs.glob(f"{base_path}/**")
        frame_paths = [
            f"gs://{p}"
            for p in raw_paths
            if Path(p).suffix.lower() in allowed_exts
        ]
        log.info("Discovered %d frame paths from %s", len(frame_paths), base_path)
        if cache_manifest_path:
            cache_dir = os.path.dirname(cache_manifest_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(cache_manifest_path, "w") as f:
                json.dump(frame_paths, f)
            log.info("Wrote cache manifest: %s", cache_manifest_path)

    # Apply path_contains filter (for nested bucket layouts like Teams v2)
    if path_contains:
        before_filter = len(frame_paths)
        frame_paths = [p for p in frame_paths if path_contains in p]
        log.info(
            "path_contains filter '%s': %d -> %d frame paths",
            path_contains, before_filter, len(frame_paths),
        )

    # Apply path_exclude_contains filter (drop paths matching any pattern).
    # Used e.g. to strip VisoMaster "hint" folders (failed deepfakes) from the
    # teams_ood_fake pool: those frames look near-real by design and polluting
    # the fake-side TPR metric with them depresses `other_fakes_tpr` in the
    # value_composite.
    if path_exclude_contains:
        if isinstance(path_exclude_contains, str):
            exclude_patterns = [path_exclude_contains]
        else:
            exclude_patterns = list(path_exclude_contains)
        if exclude_patterns:
            before_filter = len(frame_paths)
            frame_paths = [
                p for p in frame_paths
                if not any(pat in p for pat in exclude_patterns)
            ]
            log.info(
                "path_exclude_contains filter %s: %d -> %d frame paths",
                exclude_patterns, before_filter, len(frame_paths),
            )

    videos: List[VideoInfo] = []

    if grouping == "by_folder":
        videos_by_id: Dict[str, List[str]] = defaultdict(list)
        min_depth = abs(video_id_depth)
        for path in frame_paths:
            parts = path.rstrip("/").split("/")
            if len(parts) < min_depth:
                continue
            video_id = parts[video_id_depth]
            videos_by_id[video_id].append(path)

        for video_id in sorted(videos_by_id.keys()):
            frames = videos_by_id[video_id]
            shaped_frames = _shape_frames_deterministically(
                frames,
                deterministic_frame_count=deterministic_frame_count,
            )
            identity = _safe_identity(None, (method_name, video_id))
            videos.append(
                VideoInfo(
                    label=label,
                    method=method_name,
                    video_id=video_id,
                    frame_paths=shaped_frames,
                    identity=identity,
                )
            )
    else:
        # per_image mode: each image path is evaluated as an independent sample
        # with a stable synthetic video_id.
        for path in sorted(frame_paths):
            parent = Path(path).parent.name
            stem = Path(path).stem
            video_id = f"{parent}__{stem}"
            shaped_frames = _shape_frames_deterministically(
                [path],
                deterministic_frame_count=deterministic_frame_count,
            )
            identity = _safe_identity(None, (method_name, video_id))
            videos.append(
                VideoInfo(
                    label=label,
                    method=method_name,
                    video_id=video_id,
                    frame_paths=shaped_frames,
                    identity=identity,
                )
            )

    total_before_sampling = len(videos)
    if max_videos and total_before_sampling > max_videos:
        import random
        rng = random.Random(seed)
        rng.shuffle(videos)
        videos = videos[:max_videos]
        log.info(
            "Sampled %d/%d external %s videos (seed=%d)",
            max_videos,
            total_before_sampling,
            label,
            seed,
        )

    log.info(
        "External %s videos loaded: method=%s grouping=%s deterministic_frame_count=%s "
        "total_videos=%d total_frames=%d",
        label,
        method_name,
        grouping,
        deterministic_frame_count,
        len(videos),
        sum(len(v.frame_paths) for v in videos),
    )
    return videos


def load_external_fake_videos(
    bucket_name: str,
    prefix: str,
    method_name: str,
    cache_manifest_path: Optional[str] = None,
    allowed_exts: Optional[Iterable[str]] = None,
    max_videos: Optional[int] = None,
    seed: int = 737,
    grouping: str = "by_folder",
    deterministic_frame_count: Optional[int] = None,
    path_contains: Optional[str] = None,
    path_exclude_contains: Optional[Iterable[str]] = None,
    video_id_depth: int = -2,
) -> List[VideoInfo]:
    """Load fake-only external videos from GCS (e.g., WMA failure set)."""
    return load_external_real_videos(
        bucket_name=bucket_name,
        prefix=prefix,
        method_name=method_name,
        cache_manifest_path=cache_manifest_path,
        allowed_exts=allowed_exts,
        max_videos=max_videos,
        seed=seed,
        label="fake",
        grouping=grouping,
        deterministic_frame_count=deterministic_frame_count,
        path_contains=path_contains,
        path_exclude_contains=path_exclude_contains,
        video_id_depth=video_id_depth,
    )


def load_deeplive_validation(
    bucket_name: str = "live-deepfake-methods-real-and-fake-frames-cropped",
    gcs_project: str = "train-cvit2",
    split: str = "val",
    train_split: float = 0.9,
    val_split: float = 0.1,
    seed: int = 737,
    anchor_indices: Optional[List[int]] = None,
    strategies: Optional[List[str]] = None,
) -> List[VideoInfo]:
    """
    Load DeepLive validation videos from GCS.
    
    DeepLive has paired real/fake frames per sample. Each sample generates
    two VideoInfo entries: one real and one fake.
    
    Args:
        bucket_name: GCS bucket name (no gs:// prefix).
        gcs_project: GCP project ID.
        split: Which split to load ('train', 'val', or 'all').
        train_split: Proportion for training (default 0.9).
        val_split: Proportion for validation (default 0.1).
        seed: Random seed for reproducibility (default 737 - matches training).
        anchor_indices: Frame indices to use (default: [0, 2, 4, 6, 8, 10, 12, 14]).
        strategies: List of strategies to include. Default (None) uses
            {"edge_cases", "minimal_processing"} for backward compatibility.
            Pass explicit list to override (e.g., ["quality_enhancement"]).
    
    Returns:
        List of VideoInfo objects (2 per sample: real + fake).
    """
    import random
    from google.cloud import storage
    
    if anchor_indices is None:
        anchor_indices = [0, 2, 4, 6, 8, 10, 12, 14]
    
    log.info("Loading DeepLive samples from gs://%s", bucket_name)
    
    # Connect to GCS
    client = storage.Client(project=gcs_project)
    bucket = client.bucket(bucket_name)
    
    # Discover samples via manifest.json files.
    # Filter by effective strategy so *_enhanced sample_id is routed correctly.
    deeplive_strategies = {
        _normalize_method_name(s) for s in (strategies or ["edge_cases", "minimal_processing"])
    }
    samples = []
    manifest_blobs = bucket.list_blobs(prefix="samples/")
    raw_strategy_counts = Counter()
    effective_strategy_counts = Counter()
    
    for blob in manifest_blobs:
        if not blob.name.endswith("manifest.json"):
            continue
        try:
            manifest_data = json.loads(blob.download_as_text())
            sample_id = manifest_data.get("sample_id", "")
            raw_strategy = manifest_data.get("strategy", "")
            effective_strategy, enhancement = resolve_effective_strategy(
                sample_id=sample_id,
                raw_strategy=raw_strategy,
                enhancement=manifest_data.get("enhancement"),
            )
            raw_strategy_norm = _normalize_method_name(raw_strategy)
            effective_strategy_norm = _normalize_method_name(effective_strategy)
            raw_strategy_counts[raw_strategy_norm] += 1
            effective_strategy_counts[effective_strategy_norm] += 1
            if effective_strategy_norm not in deeplive_strategies:
                continue
            manifest_data["raw_strategy"] = raw_strategy_norm
            manifest_data["effective_strategy"] = effective_strategy_norm
            manifest_data["enhancement"] = enhancement
            samples.append(manifest_data)
        except Exception as e:
            log.warning("Failed to parse manifest %s: %s", blob.name, e)
            continue
    
    log.info(
        "Discovered %d DeepLive samples (filtered to effective strategies: %s)",
        len(samples),
        sorted(deeplive_strategies),
    )
    log.info("DeepLive raw strategy counts (all manifests): %s", dict(sorted(raw_strategy_counts.items())))
    log.info("DeepLive effective strategy counts (all manifests): %s", dict(sorted(effective_strategy_counts.items())))
    
    # Split samples using the same logic as training
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)
    
    n_total = len(shuffled)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    if split == "train":
        selected = shuffled[:n_train]
        log.info("Selected %d train samples", len(selected))
    elif split == "val":
        selected = shuffled[n_train:n_train + n_val]
        log.info("Selected %d val samples", len(selected))
    else:  # "all"
        selected = shuffled
        log.info("Selected all %d samples", len(selected))
    
    # Build VideoInfo entries
    videos: List[VideoInfo] = []
    
    for sample in selected:
        sample_id = sample.get("sample_id")
        effective_strategy = sample.get("effective_strategy") or _normalize_method_name(sample.get("strategy", "deeplive"))
        frame_count = sample.get("frame_count", 16)
        
        # Build frame paths for selected anchor indices
        real_frames = [
            f"gs://{bucket_name}/samples/{sample_id}/frames/real/frame_{i:04d}.png"
            for i in anchor_indices if i < frame_count
        ]
        fake_frames = [
            f"gs://{bucket_name}/samples/{sample_id}/frames/fake/frame_{i:04d}.png"
            for i in anchor_indices if i < frame_count
        ]

        # Skip samples with no usable frames (frame_count < smallest anchor index)
        if not real_frames or not fake_frames:
            log.warning(
                "Skipping sample %s: empty frame list (frame_count=%d, min anchor=%d)",
                sample_id, frame_count, min(anchor_indices),
            )
            continue
        
        # Use strategy as the method name
        method = f"deeplive_{effective_strategy}"
        
        # Generate identity from sample_id hash (consistent across runs)
        identity = hash(sample_id) & 0x7FFFFFFF
        
        # Real video
        videos.append(
            VideoInfo(
                label="real",
                method=method,
                video_id=f"{sample_id}_real",
                frame_paths=real_frames,
                identity=identity,
            )
        )
        
        # Fake video
        videos.append(
            VideoInfo(
                label="fake",
                method=method,
                video_id=f"{sample_id}_fake",
                frame_paths=fake_frames,
                identity=identity,
            )
        )
    
    real_count = len([v for v in videos if v.label == "real"])
    fake_count = len([v for v in videos if v.label == "fake"])
    
    log.info(
        "DeepLive validation loaded: split=%s videos=%d (real=%d fake=%d)",
        split,
        len(videos),
        real_count,
        fake_count,
    )
    
    return videos


def load_visomaster_validation(
    cropped_bucket_name: str = "live-deepfake-methods-real-and-fake-frames-cropped",
    frames_bucket_name: str = "live-deepfake-methods-real-and-fake-frames",
    gcs_project: str = "train-cvit2",
    swap_models: Optional[List[str]] = None,
    tiers: Optional[List[str]] = None,
    anchor_indices: Optional[List[int]] = None,
    include_tier_methods: bool = True,
) -> List[VideoInfo]:
    """
    Load VisoMaster validation videos from GCS with tier metadata.

    Each VisoMaster sample produces VideoInfo entries keyed by swap_model
    (e.g., ``visomaster_CSCS``) **and** optionally by tier (e.g.,
    ``visomaster_tier_STRONG``).  Both the real and fake frames of every
    sample are returned.

    To get per-tier accuracy, the same fake sample is emitted under **two**
    method names:

    * ``visomaster_{swap_model}``  – for per-model accuracy
    * ``visomaster_tier_{tier}``   – for per-tier accuracy

    The corresponding real side is emitted under ``visomaster_real`` (shared
    across all models/tiers).

    Tier data is fetched from the authoritative *full-frames* bucket
    (``live-deepfake-methods-real-and-fake-frames``).  Samples that exist
    only in the cropped bucket (SimSwap512, InStyleSwapper256-C) will have
    ``tier = "UNKNOWN"`` and are still included unless filtered out.

    Args:
        cropped_bucket_name: Bucket with cropped frames for inference.
        frames_bucket_name: Bucket with full manifests including tier_data.
        gcs_project: GCP project ID.
        swap_models: Optional list of swap model names to include
            (e.g., ``["CSCS", "GhostFace-v1"]``).  ``None`` = all.
        tiers: Optional list of tiers to include
            (e.g., ``["STRONG", "MODERATE"]``).  ``None`` = all.
        anchor_indices: Frame indices to load (default: 8 anchors).
        include_tier_methods: If True, emit additional ``visomaster_tier_*``
            fake entries for per-tier reporting.

    Returns:
        List of VideoInfo objects.
    """
    import json as _json
    from google.cloud import storage

    if anchor_indices is None:
        anchor_indices = [0, 2, 4, 6, 8, 10, 12, 14]

    log.info("Loading VisoMaster samples from gs://%s", cropped_bucket_name)

    client = storage.Client(project=gcs_project)

    # ------------------------------------------------------------------
    # 1. Discover all visomaster samples from the cropped bucket
    # ------------------------------------------------------------------
    cropped_bucket = client.bucket(cropped_bucket_name)
    cropped_manifests = {}  # sample_id -> manifest dict
    for blob in cropped_bucket.list_blobs(prefix="samples/"):
        if not blob.name.endswith("manifest.json"):
            continue
        try:
            manifest = _json.loads(blob.download_as_text())
        except Exception as exc:
            log.warning("Failed to parse cropped manifest %s: %s", blob.name, exc)
            continue
        strategy = manifest.get("strategy", "")
        if strategy != "visomaster":
            continue
        cropped_manifests[manifest["sample_id"]] = manifest

    log.info("Discovered %d VisoMaster samples in cropped bucket", len(cropped_manifests))

    # ------------------------------------------------------------------
    # 2. Enrich with tier data from the full-frames bucket
    # ------------------------------------------------------------------
    frames_bucket = client.bucket(frames_bucket_name)
    tier_cache: Dict[str, dict] = {}  # sample_id -> tier_data dict

    for blob in frames_bucket.list_blobs(prefix="samples/visomaster_"):
        if not blob.name.endswith("manifest.json"):
            continue
        try:
            manifest = _json.loads(blob.download_as_text())
        except Exception:
            continue
        sid = manifest.get("sample_id", "")
        tier_data = manifest.get("tier_data")
        if sid and tier_data:
            tier_cache[sid] = tier_data

    log.info("Fetched tier data for %d samples from frames bucket", len(tier_cache))

    # Also fetch tier data for DeepLiveCam samples in case they ended up
    # in the same bucket – but we only care about visomaster here.

    # ------------------------------------------------------------------
    # 3. Build VideoInfo entries
    # ------------------------------------------------------------------
    videos: List[VideoInfo] = []
    stats: Dict[str, int] = Counter()
    tier_stats: Dict[str, int] = Counter()
    model_stats: Dict[str, int] = Counter()

    for sample_id, manifest in sorted(cropped_manifests.items()):
        frame_count = manifest.get("frame_count", 16)

        # Parse swap_model from sample_id: visomaster_{model}_{number}
        # The cropped manifest may not have swap_model, so parse from id
        parts = sample_id.split("_")
        # sample_id format: visomaster_{SwapModel}_{number}
        # SwapModel can contain hyphens (e.g., GhostFace-v1)
        # Strategy: everything between "visomaster_" and the last "_NNNNN"
        swap_model = manifest.get("swap_model", "")
        if not swap_model:
            # Parse from sample_id: strip "visomaster_" prefix and trailing _NNNNN
            remainder = sample_id[len("visomaster_"):]
            last_underscore = remainder.rfind("_")
            if last_underscore > 0:
                swap_model = remainder[:last_underscore]
            else:
                swap_model = remainder

        # Filter by swap_model
        if swap_models and swap_model not in swap_models:
            continue

        # Resolve tier
        tier_data = tier_cache.get(sample_id, {})
        tier = tier_data.get("identity_delta_tier", "UNKNOWN")
        identity_delta = tier_data.get("identity_delta", -1.0)

        # Filter by tier
        if tiers and tier not in tiers:
            continue

        # Build frame paths (cropped bucket)
        real_frames = [
            f"gs://{cropped_bucket_name}/samples/{sample_id}/frames/real/frame_{i:04d}.png"
            for i in anchor_indices if i < frame_count
        ]
        fake_frames = [
            f"gs://{cropped_bucket_name}/samples/{sample_id}/frames/fake/frame_{i:04d}.png"
            for i in anchor_indices if i < frame_count
        ]

        if not real_frames or not fake_frames:
            log.warning("Skipping %s: no frames within anchor indices", sample_id)
            continue

        identity = hash(sample_id) & 0x7FFFFFFF
        model_method = f"visomaster_{swap_model}"

        # --- Real entry (shared real source) ---
        videos.append(
            VideoInfo(
                label="real",
                method="visomaster_real",
                video_id=f"{sample_id}_real",
                frame_paths=real_frames,
                identity=identity,
            )
        )
        stats["real"] += 1

        # --- Fake entry keyed by swap_model ---
        videos.append(
            VideoInfo(
                label="fake",
                method=model_method,
                video_id=f"{sample_id}_fake",
                frame_paths=fake_frames,
                identity=identity,
            )
        )
        stats["fake_by_model"] += 1
        model_stats[swap_model] += 1

        # --- Fake entry keyed by tier (optional duplicate for tier reporting) ---
        if include_tier_methods and tier != "UNKNOWN":
            tier_method = f"visomaster_tier_{tier}"
            videos.append(
                VideoInfo(
                    label="fake",
                    method=tier_method,
                    video_id=f"{sample_id}_fake_tier",
                    frame_paths=fake_frames,
                    identity=identity,
                )
            )
            stats["fake_by_tier"] += 1
            tier_stats[tier] += 1

    # ------------------------------------------------------------------
    # 4. Log summary
    # ------------------------------------------------------------------
    log.info(
        "VisoMaster validation loaded: total_videos=%d (real=%d fake_by_model=%d fake_by_tier=%d)",
        len(videos),
        stats.get("real", 0),
        stats.get("fake_by_model", 0),
        stats.get("fake_by_tier", 0),
    )
    log.info("  Per-model counts: %s", dict(model_stats))
    log.info("  Per-tier counts:  %s", dict(tier_stats))

    return videos
