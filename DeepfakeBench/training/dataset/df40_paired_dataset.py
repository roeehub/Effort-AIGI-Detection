"""
DF40 Paired Dataset - Paired real/fake frame dataset from GCS.

This dataset loads paired frames from the df40-frames-recropped-rfa85 bucket
using the pre-computed pair matching JSON file.

Data structure in GCS:
    gs://df40-frames-recropped-rfa85/
    ├── real/{source}/{identity}/
    │   └── *.png              # 32 real frames
    └── fake/{method}/{target_identity}_{source_identity}/
        └── *.png              # 32 fake frames

Pair matching JSON structure:
    {
        "pairs": [
            {
                "pair_id": "blendface__001_870",
                "method": "blendface",
                "target_identity": "001",
                "source_identity": "870",
                "fake": {"path": "gs://...", "frames": ["000.png", ...]},
                "real": {"path": "gs://...", "frames": ["000.png", ...]}
            },
            ...
        ]
    }

Frame indices:
    - Uses same sparse sampling as DeepLive: 8 frames from 32 available
    - Default indices: [0, 4, 8, 12, 16, 20, 24, 28] (or first 8 from the 32)

Usage:
    from dataset.df40_paired_dataset import DF40PairedDataset
    
    dataset = DF40PairedDataset(
        pair_json_path="dataset/df40_pairs/df40-pair-matching.json",
        methods=["simswap", "facedancer", "blendface"],  # or None for all
    )
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union
import io

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Frame sampling indices for DF40 (32 frames total)
# Use evenly spaced 8 frames from 32
DF40_SPARSE_INDICES = [0, 4, 8, 12, 16, 20, 24, 28]  # 8 anchor frames
DF40_ALL_INDICES = list(range(32))  # All 32 frames


@dataclass
class DF40PairedSample:
    """
    Represents a single paired sample from the DF40 dataset.
    
    Each sample contains paired real and fake frames, where:
    - fake = manipulated video of target_identity using source_identity
    - real = original video of target_identity
    """
    pair_id: str
    method: str
    target_identity: str
    source_identity: str
    
    # GCS paths (full gs:// paths)
    real_path: str
    fake_path: str
    
    # Frame filenames (just the filenames, not full paths)
    real_frames: List[str] = field(default_factory=list)
    fake_frames: List[str] = field(default_factory=list)
    
    # Metadata
    real_source: str = ""  # e.g., "FaceForensics++", "YouTube-real", "Celeb-real"
    frame_count: int = 32
    
    # NO landmarks for DF40 (key difference from DeepLive)
    has_landmarks: bool = False


class DF40PairedDataset:
    """
    Dataset for loading paired real/fake frames from DF40 GCS bucket.
    
    This dataset:
    - Loads pairs from a pre-computed JSON file
    - Supports method-based filtering
    - Uses sparse frame sampling (8 of 32 frames)
    - Does NOT support landmarks (no augmentation based on face regions)
    
    The actual batching and augmentation is handled by DF40PairedBatchingStrategy.
    """
    
    # Sampling mode configurations
    SAMPLING_MODES = {
        'sparse': DF40_SPARSE_INDICES,  # 8 frames evenly spaced
        'full': DF40_ALL_INDICES,       # All 32 frames
    }
    
    def __init__(
        self,
        pair_json_path: str,
        bucket_name: str = "df40-frames-recropped-rfa85",
        frame_sampling: Literal['sparse', 'full'] = 'sparse',
        methods: Optional[List[str]] = None,
        local_cache_dir: Optional[str] = None,
        gcs_project: str = "train-cvit2",
    ):
        """
        Initialize the DF40 Paired dataset.
        
        Args:
            pair_json_path: Path to the df40-pair-matching.json file
            bucket_name: GCS bucket containing the frames
            frame_sampling: Frame sampling mode
                - 'sparse': 8 evenly spaced frames
                - 'full': All 32 frames
            methods: List of methods to include, or None for all
            local_cache_dir: Optional local directory for caching frames
            gcs_project: GCP project ID
        """
        self.pair_json_path = pair_json_path
        self.bucket_name = bucket_name
        self.frame_sampling = frame_sampling
        self.methods = methods
        self.local_cache_dir = local_cache_dir
        self.gcs_project = gcs_project
        
        # Get frame indices based on sampling mode
        self.frame_indices = self.SAMPLING_MODES[frame_sampling]
        
        # DF40 does NOT have landmarks
        self.use_landmarks = False
        
        # Storage client (lazy initialized)
        self._storage_client = None
        self._bucket = None
        
        # Sample cache
        self._samples: List[DF40PairedSample] = []
        self._samples_discovered = False
        
        # Pair data (loaded from JSON)
        self._pair_data: Optional[Dict] = None
        
        logger.info(f"DF40PairedDataset initialized:")
        logger.info(f"  - Pair JSON: {pair_json_path}")
        logger.info(f"  - Bucket: {bucket_name}")
        logger.info(f"  - Frame sampling: {frame_sampling} ({len(self.frame_indices)} frames)")
        logger.info(f"  - Methods filter: {methods if methods else 'all'}")
        logger.info(f"  - ⚠️  NO LANDMARKS AVAILABLE - landmark-based augmentation disabled")
    
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
    
    def _load_pair_json(self) -> Dict:
        """Load and cache the pair matching JSON."""
        if self._pair_data is not None:
            return self._pair_data
        
        logger.info(f"Loading pair JSON from: {self.pair_json_path}")
        
        with open(self.pair_json_path, 'r') as f:
            self._pair_data = json.load(f)
        
        summary = self._pair_data.get('summary', {})
        logger.info(f"  - Total pairs in JSON: {summary.get('total_pairs', 'unknown')}")
        logger.info(f"  - Methods available: {self._pair_data.get('methods', [])}")
        
        return self._pair_data
    
    def discover_samples(self, force_refresh: bool = False) -> List[DF40PairedSample]:
        """
        Discover all available samples from the pair JSON.
        
        Args:
            force_refresh: Force re-discovery even if samples were already loaded
            
        Returns:
            List of DF40PairedSample objects
        """
        if self._samples_discovered and not force_refresh:
            return self._samples
        
        logger.info("Discovering samples from pair JSON...")
        
        pair_data = self._load_pair_json()
        pairs = pair_data.get('pairs', [])
        
        samples = []
        method_counts = {}
        
        for pair in pairs:
            method = pair.get('method', 'unknown')
            
            # Filter by method if specified
            if self.methods is not None and method not in self.methods:
                continue
            
            # Create sample object
            sample = DF40PairedSample(
                pair_id=pair['pair_id'],
                method=method,
                target_identity=pair.get('target_identity', ''),
                source_identity=pair.get('source_identity', ''),
                real_path=pair['real']['path'],
                fake_path=pair['fake']['path'],
                real_frames=pair['real'].get('frames', []),
                fake_frames=pair['fake'].get('frames', []),
                real_source=pair['real'].get('source', 'unknown'),
                frame_count=pair['real'].get('frame_count', 32),
                has_landmarks=False,  # DF40 has no landmarks
            )
            
            samples.append(sample)
            method_counts[method] = method_counts.get(method, 0) + 1
        
        self._samples = samples
        self._samples_discovered = True
        
        logger.info(f"Discovered {len(samples)} samples")
        for method, count in sorted(method_counts.items()):
            logger.info(f"  - {method}: {count} pairs")
        
        return samples
    
    def get_frame_indices(self) -> List[int]:
        """Get the frame indices to load based on sampling mode."""
        return self.frame_indices
    
    def _gcs_path_to_blob_path(self, gcs_path: str) -> str:
        """Convert gs://bucket/path to just path (strip bucket prefix)."""
        # gs://df40-frames-recropped-rfa85/fake/blendface/... -> fake/blendface/...
        if gcs_path.startswith('gs://'):
            parts = gcs_path.replace('gs://', '').split('/', 1)
            if len(parts) > 1:
                return parts[1]
        return gcs_path
    
    def load_frame(
        self, 
        gcs_path: str, 
        as_array: bool = True
    ) -> Union[np.ndarray, Image.Image]:
        """
        Load a single frame from GCS.
        
        Args:
            gcs_path: Full GCS path or relative path to frame
            as_array: Return as numpy array (True) or PIL Image (False)
            
        Returns:
            Frame as numpy array [H, W, C] or PIL Image
        """
        blob_path = self._gcs_path_to_blob_path(gcs_path)
        
        # Check local cache first
        if self.local_cache_dir:
            local_path = os.path.join(self.local_cache_dir, blob_path)
            if os.path.exists(local_path):
                img = Image.open(local_path).convert('RGB')
                return np.array(img) if as_array else img
        
        # Download from GCS
        blob = self.bucket.blob(blob_path)
        img_bytes = blob.download_as_bytes()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Optionally cache locally
        if self.local_cache_dir:
            local_path = os.path.join(self.local_cache_dir, blob_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            img.save(local_path)
        
        return np.array(img) if as_array else img
    
    def load_sample_frames(
        self,
        sample: DF40PairedSample,
        frame_indices: Optional[List[int]] = None,
        as_array: bool = True,
    ) -> Tuple[List[Union[np.ndarray, Image.Image]], List[Union[np.ndarray, Image.Image]]]:
        """
        Load frames for a sample.
        
        Args:
            sample: DF40PairedSample to load frames for
            frame_indices: Which frame indices to load (default: use self.frame_indices)
            as_array: Return as numpy arrays (True) or PIL Images (False)
            
        Returns:
            Tuple of (real_frames, fake_frames) lists
        """
        if frame_indices is None:
            frame_indices = self.frame_indices
        
        real_frames = []
        fake_frames = []
        
        # Map frame indices to actual frame filenames
        # DF40 has 32 frames but filenames may not be 0-31
        # Use the frames list from the JSON
        real_frame_list = sample.real_frames
        fake_frame_list = sample.fake_frames
        
        for idx in frame_indices:
            # Ensure index is within bounds
            if idx < len(real_frame_list) and idx < len(fake_frame_list):
                real_filename = real_frame_list[idx]
                fake_filename = fake_frame_list[idx]
                
                # Build full paths
                real_path = sample.real_path.rstrip('/') + '/' + real_filename
                fake_path = sample.fake_path.rstrip('/') + '/' + fake_filename
                
                # Load frames
                real_frame = self.load_frame(real_path, as_array=as_array)
                fake_frame = self.load_frame(fake_path, as_array=as_array)
                
                real_frames.append(real_frame)
                fake_frames.append(fake_frame)
        
        return real_frames, fake_frames
    
    def load_landmarks(self, sample: DF40PairedSample) -> Tuple[None, None]:
        """
        Load landmarks for a sample - DF40 does NOT have landmarks.
        
        This method exists for API compatibility with DeepLive but always returns (None, None).
        
        Args:
            sample: DF40PairedSample
            
        Returns:
            (None, None) - DF40 has no landmarks
        """
        # DF40 does not have landmarks
        return None, None
    
    def get_available_methods(self) -> List[str]:
        """Get list of all methods available in the pair JSON."""
        pair_data = self._load_pair_json()
        return pair_data.get('methods', [])
    
    def get_method_counts(self) -> Dict[str, int]:
        """Get count of pairs per method."""
        pair_data = self._load_pair_json()
        return pair_data.get('summary', {}).get('pairs_per_method', {})
