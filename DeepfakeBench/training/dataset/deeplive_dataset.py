"""
DeepLive Dataset - Paired real/fake frame dataset from GCS.

This dataset loads frames from the `live-deepfake-methods-real-and-fake-frames` bucket
which contains paired real and fake frames with landmarks.

Data structure in GCS:
    gs://live-deepfake-methods-real-and-fake-frames/
    └── samples/{sample_id}/
        ├── manifest.json           # Sample metadata
        ├── frames/real/*.png       # 16 real frames
        ├── frames/fake/*.png       # 16 fake frames
        └── landmarks/*.json        # MediaPipe landmarks

Frame indices:
    - Anchor frames: 0, 2, 4, 6, 8, 10, 12, 14 (uniformly distributed)
    - Consecutive frames: 1, 3, 5, 7, 9, 11, 13, 15 (immediately after anchors)

Usage:
    from dataset.deeplive_dataset import DeepLiveDataset
    
    dataset = DeepLiveDataset(
        bucket_name="live-deepfake-methods-real-and-fake-frames",
        frame_sampling="sparse",  # or "full", "pairs"
        strategies=["edge_cases", "minimal_processing"],  # or "all"
        use_landmarks=True,
    )
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import io
import re

import numpy as np
from PIL import Image
from fsspec.core import url_to_fs

logger = logging.getLogger(__name__)

# Frame sampling indices
ANCHOR_INDICES = [0, 2, 4, 6, 8, 10, 12, 14]  # 8 anchor frames
CONSECUTIVE_INDICES = [1, 3, 5, 7, 9, 11, 13, 15]  # 8 consecutive frames
ALL_INDICES = list(range(16))  # All 16 frames


@dataclass
class DeepLiveLandmark:
    """Landmark data for a single frame."""
    frame_index: int
    face_detected: bool
    landmarks: List[Dict[str, float]]  # Raw 478-point MediaPipe landmarks [{x, y, z}, ...]
    regions: Dict[str, Dict]  # {region_name: {landmarks: [...], bbox: {...}}} - may be empty
    blendshapes: Optional[Dict[str, float]] = None
    head_pose: Optional[Dict[str, float]] = None


@dataclass
class DeepLiveSample:
    """
    Represents a single sample from the DeepLive dataset.
    
    Each sample contains paired real and fake frames from the same video source,
    along with optional landmark data for augmentation.
    """
    sample_id: str
    strategy: str
    raw_strategy: str = ""
    effective_strategy: str = ""
    enhancement: str = "none"
    
    # Frame paths (relative to sample directory)
    real_frame_paths: List[str] = field(default_factory=list)
    fake_frame_paths: List[str] = field(default_factory=list)
    
    # Metadata
    frame_count: int = 16
    has_landmarks: bool = False
    real_faces_detected: int = 0
    fake_faces_detected: int = 0
    
    # Original video name (used for identity extraction)
    # Format: cropped_XXX.mp4 where XXX is the YouTube video ID
    original_video_name: str = ""
    
    # Loaded data (populated on demand)
    real_landmarks: Optional[List[DeepLiveLandmark]] = None
    fake_landmarks: Optional[List[DeepLiveLandmark]] = None


_ENHANCED_SAMPLE_ID_PATTERN = re.compile(r"^(?P<base>.+)_enhanced_[0-9]+$")


def _normalize_strategy_name(value: Optional[str]) -> str:
    """Normalize strategy names for stable filtering/comparison."""
    if value is None:
        return "unknown"
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def resolve_effective_strategy(
    sample_id: Optional[str],
    raw_strategy: Optional[str],
    enhancement: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Resolve DeepLive effective strategy from manifest metadata.

    Enhanced DeepLive manifests may keep raw strategy as base (e.g. edge_cases)
    while sample_id carries *_enhanced_* (e.g. edge_cases_enhanced_0001).
    This resolver normalizes that into effective_strategy=edge_cases_enhanced.
    """
    sample_id_norm = _normalize_strategy_name(sample_id)
    raw_strategy_norm = _normalize_strategy_name(raw_strategy)

    matched = _ENHANCED_SAMPLE_ID_PATTERN.match(sample_id_norm)
    if matched:
        base = matched.group("base")
        return f"{base}_enhanced", "enhanced"

    if raw_strategy_norm.endswith("_enhanced"):
        return raw_strategy_norm, "enhanced"

    enhancement_norm = _normalize_strategy_name(enhancement)
    if enhancement_norm not in {"", "none", "unknown"}:
        return raw_strategy_norm, enhancement_norm

    return raw_strategy_norm, "none"


class DeepLiveDataset:
    """
    Dataset for loading paired real/fake frames from GCS.
    
    This is a base dataset class that handles:
    - Sample discovery via manifest.json files
    - Frame loading (sparse/full/pairs modes)
    - Landmark loading for augmentation
    - Strategy-based filtering
    
    The actual batching and augmentation is handled by DeepLiveBatchingStrategy.
    """
    
    # Sampling mode configurations
    SAMPLING_MODES = {
        'sparse': ANCHOR_INDICES,          # 8 anchor frames
        'full': ALL_INDICES,               # All 16 frames
        'pairs': list(zip(ANCHOR_INDICES, CONSECUTIVE_INDICES)),  # Anchor-consecutive pairs
    }
    
    def __init__(
        self,
        bucket_name: str = "live-deepfake-methods-real-and-fake-frames",
        frame_sampling: Literal['sparse', 'full', 'pairs'] = 'sparse',
        strategies: Union[str, List[str]] = 'all',
        use_landmarks: bool = True,
        local_cache_dir: Optional[str] = None,
        gcs_project: str = "train-cvit2",
        cache_manifest_uri: Optional[str] = None,
        cache_max_age_hours: float = 12.0,
        cache_revision: Optional[str] = None,
        max_samples_total: Optional[int] = None,
        max_samples_per_strategy: Optional[int] = None,
    ):
        """
        Initialize the DeepLive dataset.
        
        Args:
            bucket_name: GCS bucket containing the dataset
            frame_sampling: Frame sampling mode
                - 'sparse': 8 anchor frames (0, 2, 4, 6, 8, 10, 12, 14)
                - 'full': All 16 frames
                - 'pairs': Anchor-consecutive pairs for temporal analysis
            strategies: Which data strategies to include
                - 'all': Include all strategies
                - List of strategy names: ['edge_cases', 'minimal_processing', ...]
            use_landmarks: Whether to load landmark data
            local_cache_dir: Optional local directory for caching frames
            gcs_project: GCP project ID
            cache_manifest_uri: Optional local/gs:// JSON cache for discovery results
            cache_max_age_hours: Discovery cache freshness window
            cache_revision: Optional revision token for manual cache invalidation
            max_samples_total: Optional cap on total discovered samples
            max_samples_per_strategy: Optional cap per strategy prefix (only when
                strategies are explicitly provided)
        """
        self.bucket_name = bucket_name
        self.frame_sampling = frame_sampling
        if strategies == 'all':
            self.strategies = None
        elif isinstance(strategies, str):
            self.strategies = [strategies]
        else:
            self.strategies = list(strategies)
        self.use_landmarks = use_landmarks
        self.local_cache_dir = local_cache_dir
        self.gcs_project = gcs_project
        self.cache_manifest_uri = cache_manifest_uri
        self.cache_max_age_hours = float(cache_max_age_hours)
        self.cache_revision = str(cache_revision or "")
        self.max_samples_total = int(max_samples_total) if max_samples_total else None
        self.max_samples_per_strategy = (
            int(max_samples_per_strategy) if max_samples_per_strategy else None
        )
        if self.max_samples_total is not None and self.max_samples_total <= 0:
            self.max_samples_total = None
        if self.max_samples_per_strategy is not None and self.max_samples_per_strategy <= 0:
            self.max_samples_per_strategy = None
        
        # Get frame indices based on sampling mode
        self.frame_indices = self.SAMPLING_MODES[frame_sampling]
        
        # Storage client (lazy initialized)
        self._storage_client = None
        self._bucket = None
        
        # Sample cache
        self._samples: List[DeepLiveSample] = []
        self._samples_discovered = False
        # Landmark miss caches to avoid repeated GCS 404 calls/log spam.
        self._missing_landmark_paths = set()
        self._samples_without_landmarks = set()
        self._warned_missing_landmark_paths = set()
        
        logger.info(f"DeepLiveDataset initialized:")
        logger.info(f"  - Bucket: {bucket_name}")
        logger.info(f"  - Frame sampling: {frame_sampling} ({len(self.frame_indices)} frames)")
        logger.info(f"  - Strategies: {strategies}")
        logger.info(f"  - Use landmarks: {use_landmarks}")
        logger.info(
            "  - Discovery cache: %s (max_age_hours=%.1f, revision=%s)",
            cache_manifest_uri or "DISABLED",
            self.cache_max_age_hours,
            self.cache_revision or "",
        )
        logger.info(
            "  - Discovery caps: total=%s per_strategy=%s",
            self.max_samples_total if self.max_samples_total is not None else "NONE",
            self.max_samples_per_strategy if self.max_samples_per_strategy is not None else "NONE",
        )
    
    @property
    def storage_client(self):
        """Lazy-load GCS storage client."""
        if self._storage_client is None:
            try:
                from google.cloud import storage
                self._storage_client = storage.Client(project=self.gcs_project)
                self._bucket = self._storage_client.bucket(self.bucket_name)
                logger.info(f"Connected to GCS bucket: {self.bucket_name}")
            except ImportError:
                raise ImportError(
                    "google-cloud-storage required for GCS access. "
                    "Install with: pip install google-cloud-storage"
                )
            except Exception as e:
                logger.error(f"Failed to connect to GCS: {e}")
                raise
        return self._storage_client
    
    @property
    def bucket(self):
        """Get the GCS bucket object."""
        if self._bucket is None:
            _ = self.storage_client  # Initialize client and bucket
        return self._bucket

    def _discovery_cache_config(self) -> Dict[str, Any]:
        strategies_norm = (
            sorted(_normalize_strategy_name(s) for s in self.strategies)
            if self.strategies is not None
            else ["all"]
        )
        return {
            "bucket_name": self.bucket_name,
            "strategies": strategies_norm,
            "max_samples_total": self.max_samples_total or 0,
            "max_samples_per_strategy": self.max_samples_per_strategy or 0,
            "cache_revision": self.cache_revision,
        }

    def _cache_rows_to_samples(self, rows: List[Dict[str, Any]]) -> List[DeepLiveSample]:
        samples: List[DeepLiveSample] = []
        for row in rows:
            frame_count = int(row.get("frame_count", 16))
            sample = DeepLiveSample(
                sample_id=row.get("sample_id", ""),
                strategy=_normalize_strategy_name(row.get("strategy", "unknown")),
                raw_strategy=_normalize_strategy_name(row.get("raw_strategy", "unknown")),
                effective_strategy=_normalize_strategy_name(row.get("effective_strategy", "unknown")),
                enhancement=_normalize_strategy_name(row.get("enhancement", "none")),
                frame_count=frame_count,
                has_landmarks=bool(row.get("has_landmarks", False)),
                real_faces_detected=int(row.get("real_faces_detected", 0)),
                fake_faces_detected=int(row.get("fake_faces_detected", 0)),
                original_video_name=row.get("original_video_name", ""),
            )
            sample.real_frame_paths = [
                f"samples/{sample.sample_id}/frames/real/frame_{i:04d}.png"
                for i in range(frame_count)
            ]
            sample.fake_frame_paths = [
                f"samples/{sample.sample_id}/frames/fake/frame_{i:04d}.png"
                for i in range(frame_count)
            ]
            samples.append(sample)
        return samples

    def _samples_to_cache_rows(self, samples: List[DeepLiveSample]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for sample in samples:
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "strategy": sample.strategy,
                    "raw_strategy": sample.raw_strategy,
                    "effective_strategy": sample.effective_strategy,
                    "enhancement": sample.enhancement,
                    "frame_count": int(sample.frame_count),
                    "has_landmarks": bool(sample.has_landmarks),
                    "real_faces_detected": int(sample.real_faces_detected),
                    "fake_faces_detected": int(sample.fake_faces_detected),
                    "original_video_name": sample.original_video_name,
                }
            )
        return rows

    def _load_discovery_cache(self) -> Optional[List[DeepLiveSample]]:
        if not self.cache_manifest_uri:
            return None

        try:
            fs, path = url_to_fs(self.cache_manifest_uri)
            if not fs.exists(path):
                return None
            with fs.open(path, "r") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning("Failed reading DeepLive discovery cache %s: %s", self.cache_manifest_uri, exc)
            return None

        if payload.get("version") != 1:
            return None
        if payload.get("config") != self._discovery_cache_config():
            return None

        created_unix = float(payload.get("created_unix", 0) or 0)
        cache_age_seconds = time.time() - created_unix
        if cache_age_seconds > self.cache_max_age_hours * 3600.0:
            logger.info(
                "DeepLive discovery cache expired (age=%.1fh > max_age=%.1fh): %s",
                cache_age_seconds / 3600.0,
                self.cache_max_age_hours,
                self.cache_manifest_uri,
            )
            return None

        rows = payload.get("samples", []) or []
        samples = self._cache_rows_to_samples(rows)
        logger.info(
            "Loaded %d DeepLive samples from discovery cache: %s",
            len(samples),
            self.cache_manifest_uri,
        )
        return samples

    def _write_discovery_cache(self, samples: List[DeepLiveSample]) -> None:
        if not self.cache_manifest_uri:
            return

        payload = {
            "version": 1,
            "created_unix": time.time(),
            "config": self._discovery_cache_config(),
            "samples": self._samples_to_cache_rows(samples),
        }
        try:
            fs, path = url_to_fs(self.cache_manifest_uri)
            parent = os.path.dirname(path)
            if parent and not fs.exists(parent):
                fs.makedirs(parent, exist_ok=True)
            with fs.open(path, "w") as f:
                json.dump(payload, f)
            logger.info("Wrote DeepLive discovery cache: %s", self.cache_manifest_uri)
        except Exception as exc:
            logger.warning("Failed writing DeepLive discovery cache %s: %s", self.cache_manifest_uri, exc)
    
    def discover_samples(self, force_refresh: bool = False) -> List[DeepLiveSample]:
        """
        Discover all available samples from GCS.
        
        Samples are identified by their manifest.json files. A sample is only
        considered complete if its manifest exists (uploaded last in the pipeline).
        
        Args:
            force_refresh: Force re-discovery even if samples were already loaded
            
        Returns:
            List of DeepLiveSample objects
        """
        if self._samples_discovered and not force_refresh:
            return self._samples

        if not force_refresh:
            cached_samples = self._load_discovery_cache()
            if cached_samples is not None:
                self._samples = cached_samples
                self._samples_discovered = True
                return self._samples

        logger.info("Discovering samples from GCS...")
        samples = []
        
        # List manifest.json files. If strategies are explicitly provided, scan
        # strategy-specific prefixes to avoid crawling unrelated samples (e.g. visomaster_*).
        prefixes = ["samples/"]
        if self.strategies is not None:
            strategy_prefixes = []
            for strategy in self.strategies:
                strategy_norm = _normalize_strategy_name(strategy)
                if strategy_norm in {"", "all", "unknown"}:
                    strategy_prefixes = []
                    break
                strategy_prefixes.append(f"samples/{strategy_norm}_")
            if strategy_prefixes:
                prefixes = sorted(set(strategy_prefixes))

        logger.info("Manifest discovery prefixes: %s", prefixes)
        allowed_strategies = (
            {_normalize_strategy_name(s) for s in self.strategies}
            if self.strategies is not None
            else None
        )
        seen_manifest_names = set()
        manifest_files_scanned = 0
        max_total_reached = False
        use_prefix_cap = self.max_samples_per_strategy is not None and self.strategies is not None

        for prefix in prefixes:
            prefix_sample_count = 0
            manifest_blobs = self.bucket.list_blobs(prefix=prefix)
            for blob in manifest_blobs:
                if self.max_samples_total is not None and len(samples) >= self.max_samples_total:
                    max_total_reached = True
                    break
                if use_prefix_cap and prefix_sample_count >= self.max_samples_per_strategy:
                    break
                if not blob.name.endswith("manifest.json"):
                    continue
                if blob.name in seen_manifest_names:
                    continue
                seen_manifest_names.add(blob.name)
                manifest_files_scanned += 1
                
                try:
                    # Parse manifest
                    manifest_data = json.loads(blob.download_as_text())
                    
                    sample_id = manifest_data.get('sample_id', '')
                    raw_strategy = manifest_data.get('strategy', 'unknown')
                    effective_strategy, enhancement = resolve_effective_strategy(
                        sample_id=sample_id,
                        raw_strategy=raw_strategy,
                        enhancement=manifest_data.get('enhancement'),
                    )

                    # Filter by effective strategy if specified
                    if allowed_strategies is not None:
                        if effective_strategy not in allowed_strategies:
                            continue

                    # Create sample object
                    sample = DeepLiveSample(
                        sample_id=sample_id,
                        strategy=effective_strategy,  # Backward-compatible alias
                        raw_strategy=_normalize_strategy_name(raw_strategy),
                        effective_strategy=effective_strategy,
                        enhancement=enhancement,
                        frame_count=manifest_data.get('frame_count', 16),
                        has_landmarks=manifest_data.get('has_landmarks', False),
                        real_faces_detected=manifest_data.get('real_faces_detected', 0),
                        fake_faces_detected=manifest_data.get('fake_faces_detected', 0),
                        original_video_name=manifest_data.get('original_video_name', ''),
                    )
                    
                    # Build frame paths
                    sample.real_frame_paths = [
                        f"samples/{sample.sample_id}/frames/real/frame_{i:04d}.png"
                        for i in range(sample.frame_count)
                    ]
                    sample.fake_frame_paths = [
                        f"samples/{sample.sample_id}/frames/fake/frame_{i:04d}.png"
                        for i in range(sample.frame_count)
                    ]
                    
                    samples.append(sample)
                    prefix_sample_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to parse manifest {blob.name}: {e}")
                    continue
            if max_total_reached:
                break

        logger.info(
            "Scanned %d manifest files across %d prefix(es)",
            manifest_files_scanned,
            len(prefixes),
        )
        if max_total_reached:
            logger.info(
                "DeepLive discovery hit max_samples_total=%d (startup cap)",
                self.max_samples_total,
            )
        if use_prefix_cap:
            logger.info(
                "DeepLive discovery used per-strategy cap=%d",
                self.max_samples_per_strategy,
            )
        
        self._samples = samples
        self._samples_discovered = True
        self._write_discovery_cache(samples)
        
        logger.info(f"Discovered {len(samples)} samples")
        if samples:
            raw_counts: Dict[str, int] = {}
            effective_counts: Dict[str, int] = {}
            for s in samples:
                raw_counts[s.raw_strategy] = raw_counts.get(s.raw_strategy, 0) + 1
                effective_counts[s.effective_strategy] = effective_counts.get(s.effective_strategy, 0) + 1
            logger.info("  - Raw strategy counts: %s", dict(sorted(raw_counts.items())))
            logger.info("  - Effective strategy counts: %s", dict(sorted(effective_counts.items())))
        
        return samples
    
    def get_frame_indices(self) -> List[int]:
        """Get the frame indices to load based on sampling mode."""
        if self.frame_sampling == 'pairs':
            # For pairs mode, return flat list of indices
            return [i for pair in self.frame_indices for i in pair]
        return self.frame_indices
    
    def load_frame(
        self, 
        gcs_path: str, 
        as_array: bool = True
    ) -> Union[np.ndarray, Image.Image]:
        """
        Load a single frame from GCS.
        
        Args:
            gcs_path: Path to frame in GCS bucket
            as_array: Return as numpy array (True) or PIL Image (False)
            
        Returns:
            Frame as numpy array [H, W, C] or PIL Image
        """
        # Check local cache first
        if self.local_cache_dir:
            local_path = os.path.join(self.local_cache_dir, gcs_path)
            if os.path.exists(local_path):
                img = Image.open(local_path).convert('RGB')
                return np.array(img) if as_array else img
        
        # Download from GCS
        blob = self.bucket.blob(gcs_path)
        img_bytes = blob.download_as_bytes()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Optionally cache locally
        if self.local_cache_dir:
            local_path = os.path.join(self.local_cache_dir, gcs_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            img.save(local_path)
        
        return np.array(img) if as_array else img
    
    def load_sample_frames(
        self,
        sample: DeepLiveSample,
        frame_indices: Optional[List[int]] = None,
        as_array: bool = True,
    ) -> Tuple[List, List]:
        """
        Load frames for a sample.
        
        Args:
            sample: The sample to load frames for
            frame_indices: Specific frame indices to load (defaults to self.frame_indices)
            as_array: Return as numpy arrays
            
        Returns:
            Tuple of (real_frames, fake_frames), each a list of arrays/images
        """
        if frame_indices is None:
            frame_indices = self.get_frame_indices()
        
        real_frames = []
        fake_frames = []
        
        for idx in frame_indices:
            if idx >= len(sample.real_frame_paths):
                logger.warning(f"Frame index {idx} out of range for sample {sample.sample_id}")
                continue
            
            real_frame = self.load_frame(sample.real_frame_paths[idx], as_array)
            fake_frame = self.load_frame(sample.fake_frame_paths[idx], as_array)
            
            real_frames.append(real_frame)
            fake_frames.append(fake_frame)
        
        return real_frames, fake_frames
    
    def load_landmarks(
        self,
        sample: DeepLiveSample,
    ) -> Tuple[Optional[List[DeepLiveLandmark]], Optional[List[DeepLiveLandmark]]]:
        """
        Load landmark data for a sample.
        
        Args:
            sample: The sample to load landmarks for
            
        Returns:
            Tuple of (real_landmarks, fake_landmarks)
        """
        if not sample.has_landmarks:
            return None, None
        if sample.sample_id in self._samples_without_landmarks:
            return None, None
        
        def load_landmark_file(path: str) -> List[DeepLiveLandmark]:
            if path in self._missing_landmark_paths:
                return []
            blob = self.bucket.blob(path)
            try:
                data = json.loads(blob.download_as_text())
                landmarks = []
                for frame_data in data.get('frames', []):
                    landmark = DeepLiveLandmark(
                        frame_index=frame_data['frame_index'],
                        face_detected=frame_data.get('face_detected', False),
                        landmarks=frame_data.get('landmarks', []),  # Raw 478-point MediaPipe landmarks
                        regions=frame_data.get('regions', {}),
                        blendshapes=frame_data.get('blendshapes'),
                        head_pose=frame_data.get('head_pose'),
                    )
                    landmarks.append(landmark)
                return landmarks
            except Exception as e:
                error_text = str(e)
                is_missing = ("404" in error_text) or ("No such object" in error_text)
                if is_missing:
                    self._missing_landmark_paths.add(path)
                    if path not in self._warned_missing_landmark_paths:
                        logger.warning(f"Failed to load landmarks from {path}: {e}")
                        self._warned_missing_landmark_paths.add(path)
                else:
                    logger.warning(f"Failed to load landmarks from {path}: {e}")
                return []
        
        real_path = f"samples/{sample.sample_id}/landmarks/real_landmarks.json"
        fake_path = f"samples/{sample.sample_id}/landmarks/fake_landmarks.json"
        
        real_landmarks = load_landmark_file(real_path)
        fake_landmarks = load_landmark_file(fake_path)

        if not real_landmarks and not fake_landmarks:
            # Stop attempting landmark loads for this sample for the rest of the run.
            sample.has_landmarks = False
            self._samples_without_landmarks.add(sample.sample_id)
            return None, None

        return real_landmarks, fake_landmarks
    
    def get_landmark_bbox(
        self,
        landmark: DeepLiveLandmark,
        region: str,
        image_shape: Tuple[int, int],
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box for a facial region from landmarks.
        
        Args:
            landmark: Landmark data for a frame
            region: Region name (e.g., 'left_eye', 'mouth', 'nose')
            image_shape: (height, width) of the image
            
        Returns:
            Bounding box as (x_min, y_min, x_max, y_max) in pixels, or None
        """
        if not landmark.face_detected:
            return None
        
        if region not in landmark.regions:
            return None
        
        bbox = landmark.regions[region].get('bbox')
        if bbox is None:
            return None
        
        h, w = image_shape
        return (
            int(bbox['x_min'] * w),
            int(bbox['y_min'] * h),
            int(bbox['x_max'] * w),
            int(bbox['y_max'] * h),
        )
    
    def __len__(self) -> int:
        """Number of samples in the dataset."""
        if not self._samples_discovered:
            self.discover_samples()
        return len(self._samples)
    
    def __getitem__(self, idx: int) -> DeepLiveSample:
        """Get a sample by index."""
        if not self._samples_discovered:
            self.discover_samples()
        return self._samples[idx]
    
    def get_samples(self) -> List[DeepLiveSample]:
        """Get all discovered samples."""
        if not self._samples_discovered:
            self.discover_samples()
        return self._samples
    
    def split_samples(
        self,
        train_ratio: float = 0.8,
        seed: int = 42,
        stratify_by: str = 'strategy',
    ) -> Tuple[List[DeepLiveSample], List[DeepLiveSample]]:
        """
        Split samples into train and validation sets.
        
        Args:
            train_ratio: Proportion of samples for training
            seed: Random seed for reproducibility
            stratify_by: Stratification method ('strategy', 'none')
            
        Returns:
            Tuple of (train_samples, val_samples)
        """
        if not self._samples_discovered:
            self.discover_samples()
        
        import random
        rng = random.Random(seed)
        
        if stratify_by == 'none':
            # Simple random split
            samples = self._samples.copy()
            rng.shuffle(samples)
            split_idx = int(len(samples) * train_ratio)
            return samples[:split_idx], samples[split_idx:]
        
        elif stratify_by == 'strategy':
            # Stratified split by strategy
            by_strategy: Dict[str, List[DeepLiveSample]] = {}
            for sample in self._samples:
                strategy_key = sample.effective_strategy or sample.strategy
                if strategy_key not in by_strategy:
                    by_strategy[strategy_key] = []
                by_strategy[strategy_key].append(sample)
            
            train_samples = []
            val_samples = []
            
            for strategy, samples in by_strategy.items():
                rng.shuffle(samples)
                split_idx = int(len(samples) * train_ratio)
                train_samples.extend(samples[:split_idx])
                val_samples.extend(samples[split_idx:])
            
            rng.shuffle(train_samples)
            rng.shuffle(val_samples)
            
            return train_samples, val_samples
        
        else:
            raise ValueError(f"Unknown stratify_by method: {stratify_by}")


# Convenience function for creating dataset from config
def create_deeplive_dataset(config: dict) -> DeepLiveDataset:
    """
    Create a DeepLiveDataset from a config dictionary.
    
    Args:
        config: Configuration dictionary with deeplive_data section
        
    Returns:
        Configured DeepLiveDataset instance
    """
    deeplive_config = config.get('deeplive_data', {})
    
    return DeepLiveDataset(
        bucket_name=deeplive_config.get('bucket_name', 'live-deepfake-methods-real-and-fake-frames'),
        frame_sampling=deeplive_config.get('frame_sampling', 'sparse'),
        strategies=deeplive_config.get('strategies', 'all'),
        use_landmarks=deeplive_config.get('use_landmarks', True),
        local_cache_dir=deeplive_config.get('local_cache_dir'),
        gcs_project=deeplive_config.get('gcs_project', 'train-cvit2'),
    )
