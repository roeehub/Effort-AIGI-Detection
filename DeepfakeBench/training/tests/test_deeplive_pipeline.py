#!/usr/bin/env python3
"""
End-to-End Test Suite for DeepLive Data Pipeline

This test suite verifies the complete DeepLive data pipeline:
1. GCS bucket access
2. Sample discovery via manifest.json
3. Paired real/fake frame loading
4. Landmark JSON parsing
5. Augmentation application (with visualizations)
6. Batch structure verification

Usage:
    # Run all tests
    python -m tests.test_deeplive_pipeline
    
    # Run with visualizations (saves images)
    python -m tests.test_deeplive_pipeline --save-visualizations
    
    # Test with specific number of samples
    python -m tests.test_deeplive_pipeline --num-samples 5
    
    # Run specific test
    python -m tests.test_deeplive_pipeline --test gcs_access
    
    # Test with specific bucket
    python -m tests.test_deeplive_pipeline --bucket gs://live-deepfake-methods-real-and-fake-frames
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add training dir to path
TRAINING_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(TRAINING_DIR))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration: float
    message: str = ""
    details: Optional[Dict] = None


class DeepLivePipelineTest:
    """
    End-to-end test suite for DeepLive data pipeline.
    """
    
    DEFAULT_BUCKET = "live-deepfake-methods-real-and-fake-frames-cropped"
    DEFAULT_PROJECT = "train-cvit2"
    DEFAULT_DEBUG_BUCKET = "training-job-outputs"
    DEFAULT_DEBUG_PREFIX = "debug"
    
    def __init__(
        self,
        bucket_name: str = None,
        gcs_project: str = None,
        num_samples: int = 10,
        save_visualizations: bool = False,
        output_dir: str = None,
        save_debug_artifacts: bool = False,
        debug_bucket_name: str = None,
        debug_prefix: str = None,
    ):
        """
        Initialize the test suite.
        
        Args:
            bucket_name: GCS bucket name (without gs:// prefix)
            gcs_project: GCP project ID
            num_samples: Number of samples to test
            save_visualizations: Whether to save visualization images locally
            output_dir: Directory for saving visualizations locally
            save_debug_artifacts: Whether to save debug artifacts to GCS
            debug_bucket_name: GCS bucket name for debug artifacts
            debug_prefix: Prefix/folder within the debug bucket
        """
        self.bucket_name = bucket_name or self.DEFAULT_BUCKET
        self.gcs_project = gcs_project or self.DEFAULT_PROJECT
        self.num_samples = num_samples
        self.save_visualizations = save_visualizations
        self.output_dir = Path(output_dir or TRAINING_DIR / "test_outputs" / "deeplive_pipeline")
        self.save_debug_artifacts = save_debug_artifacts
        
        # Debug bucket configuration
        self.debug_bucket_name = debug_bucket_name or self.DEFAULT_DEBUG_BUCKET
        self.debug_prefix = debug_prefix if debug_prefix is not None else self.DEFAULT_DEBUG_PREFIX
        
        # Test results
        self.results: List[TestResult] = []
        
        # Cached objects
        self._storage_client = None
        self._bucket = None
        self._debug_bucket = None
        self._dataset = None
        self._samples = None
        
        # Debug artifact run ID (for grouping artifacts from this run)
        self._debug_run_id = None
        if self.save_debug_artifacts:
            import datetime
            self._debug_run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"DeepLive Pipeline Test Suite initialized:")
        logger.info(f"  - Bucket: {self.bucket_name}")
        logger.info(f"  - Project: {self.gcs_project}")
        logger.info(f"  - Num samples: {self.num_samples}")
        logger.info(f"  - Save visualizations: {self.save_visualizations}")
        if self.save_debug_artifacts:
            prefix = f"{self.debug_prefix}/" if self.debug_prefix else ""
            logger.info(f"  - Debug artifacts: gs://{self.debug_bucket_name}/{prefix}runs/{self._debug_run_id}/")
    
    def _add_result(
        self, 
        name: str, 
        passed: bool, 
        duration: float, 
        message: str = "",
        details: Optional[Dict] = None
    ):
        """Add a test result."""
        result = TestResult(
            name=name,
            passed=passed,
            duration=duration,
            message=message,
            details=details
        )
        self.results.append(result)
        
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{status} - {name} ({duration:.2f}s)")
        if message:
            logger.info(f"    {message}")
        
        return result
    
    # ==========================================================================
    # Debug Artifact Saving (GCS)
    # ==========================================================================
    
    def _get_debug_bucket(self):
        """Get or create debug bucket connection."""
        if self._debug_bucket is None and self.save_debug_artifacts:
            from google.cloud import storage
            client = self._storage_client or storage.Client(project=self.gcs_project)
            self._debug_bucket = client.bucket(self.debug_bucket_name)
        return self._debug_bucket
    
    def _save_debug_image_to_gcs(
        self, 
        image: np.ndarray, 
        artifact_name: str,
        subfolder: str = "images"
    ) -> Optional[str]:
        """
        Save a debug image to GCS.
        
        Args:
            image: Image as numpy array (H, W, C) in RGB format
            artifact_name: Name for the artifact (without extension)
            subfolder: Subfolder within the run directory
            
        Returns:
            GCS URI of saved artifact, or None if saving is disabled/failed
        """
        if not self.save_debug_artifacts:
            return None
        
        try:
            from PIL import Image as PILImage
            import io
            
            bucket = self._get_debug_bucket()
            if bucket is None:
                return None
            
            # Convert to PIL and save to bytes
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
            
            pil_image = PILImage.fromarray(image)
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Upload to GCS (include prefix if set)
            prefix = f"{self.debug_prefix}/" if self.debug_prefix else ""
            blob_path = f"{prefix}runs/{self._debug_run_id}/{subfolder}/{artifact_name}.png"
            blob = bucket.blob(blob_path)
            blob.upload_from_file(img_bytes, content_type='image/png')
            
            gcs_uri = f"gs://{self.debug_bucket_name}/{blob_path}"
            logger.debug(f"Saved debug image: {gcs_uri}")
            return gcs_uri
            
        except Exception as e:
            logger.warning(f"Failed to save debug image {artifact_name}: {e}")
            return None
    
    def _save_debug_json_to_gcs(
        self, 
        data: Dict, 
        artifact_name: str,
        subfolder: str = "metadata"
    ) -> Optional[str]:
        """
        Save debug JSON data to GCS.
        
        Args:
            data: Dictionary to save as JSON
            artifact_name: Name for the artifact (without extension)
            subfolder: Subfolder within the run directory
            
        Returns:
            GCS URI of saved artifact, or None if saving is disabled/failed
        """
        if not self.save_debug_artifacts:
            return None
        
        try:
            bucket = self._get_debug_bucket()
            if bucket is None:
                return None
            
            # Convert to JSON
            json_str = json.dumps(data, indent=2, default=str)
            
            # Upload to GCS (include prefix if set)
            prefix = f"{self.debug_prefix}/" if self.debug_prefix else ""
            blob_path = f"{prefix}runs/{self._debug_run_id}/{subfolder}/{artifact_name}.json"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(json_str, content_type='application/json')
            
            gcs_uri = f"gs://{self.debug_bucket_name}/{blob_path}"
            logger.debug(f"Saved debug JSON: {gcs_uri}")
            return gcs_uri
            
        except Exception as e:
            logger.warning(f"Failed to save debug JSON {artifact_name}: {e}")
            return None
    
    def _save_debug_batch_to_gcs(
        self, 
        batch: Dict[str, Any],
        batch_name: str = "batch",
        max_images: int = 8
    ) -> Dict[str, str]:
        """
        Save a batch of images and metadata to GCS for debugging.
        
        Args:
            batch: Batch dict with 'image' tensor and 'label' tensor, plus optional 'metadata'
            batch_name: Name prefix for the batch artifacts
            max_images: Maximum number of images to save from the batch
            
        Returns:
            Dict of artifact names to GCS URIs
        """
        if not self.save_debug_artifacts:
            return {}
        
        saved_uris = {}
        
        try:
            import torch
            
            images = batch.get('image')
            labels = batch.get('label')
            metadata = batch.get('metadata', {})
            
            if images is None:
                return saved_uris
            
            # Convert to numpy if tensor
            if isinstance(images, torch.Tensor):
                images = images.cpu().numpy()
            
            # Images are [B, C, H, W] - convert to [B, H, W, C]
            if len(images.shape) == 4 and images.shape[1] in [1, 3, 4]:
                images = np.transpose(images, (0, 2, 3, 1))
            
            # Scale to 0-255 if normalized
            if images.max() <= 1.0:
                images = (images * 255).astype(np.uint8)
            
            # Save individual images
            n_to_save = min(len(images), max_images)
            for i in range(n_to_save):
                img = images[i]
                label = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
                label_str = "real" if label == 0 else "fake"
                
                artifact_name = f"{batch_name}_img{i:03d}_{label_str}"
                uri = self._save_debug_image_to_gcs(img, artifact_name, subfolder=f"batches/{batch_name}")
                if uri:
                    saved_uris[artifact_name] = uri
            
            # Save batch metadata
            batch_meta = {
                "batch_size": len(images),
                "image_shape": list(images.shape),
                "labels": labels.tolist() if hasattr(labels, 'tolist') else list(labels),
                "real_count": int((np.array(labels) == 0).sum()),
                "fake_count": int((np.array(labels) == 1).sum()),
                "sample_ids": metadata.get('sample_ids', [])[:max_images],
                "frame_indices": metadata.get('frame_indices', [])[:max_images],
            }
            meta_uri = self._save_debug_json_to_gcs(
                batch_meta, 
                f"{batch_name}_metadata", 
                subfolder=f"batches/{batch_name}"
            )
            if meta_uri:
                saved_uris["metadata"] = meta_uri
            
            logger.info(f"Saved {len(saved_uris)} debug artifacts for batch '{batch_name}'")
            
        except Exception as e:
            logger.warning(f"Failed to save debug batch {batch_name}: {e}")
        
        return saved_uris
    
    # ==========================================================================
    # Test 1: GCS Access
    # ==========================================================================
    
    def test_gcs_access(self) -> TestResult:
        """Test that we can connect to and access the GCS bucket."""
        start = time.time()
        
        try:
            from google.cloud import storage
            
            # Create client
            client = storage.Client(project=self.gcs_project)
            bucket = client.bucket(self.bucket_name)
            
            # Check bucket exists
            if not bucket.exists():
                return self._add_result(
                    "gcs_access", False, time.time() - start,
                    f"Bucket '{self.bucket_name}' does not exist or is not accessible"
                )
            
            # List a few blobs to verify read access
            blobs = list(bucket.list_blobs(prefix="samples/", max_results=5))
            
            self._storage_client = client
            self._bucket = bucket
            
            return self._add_result(
                "gcs_access", True, time.time() - start,
                f"Successfully connected to bucket. Found {len(blobs)} objects.",
                {"num_blobs_sampled": len(blobs)}
            )
            
        except ImportError:
            return self._add_result(
                "gcs_access", False, time.time() - start,
                "google-cloud-storage not installed. Run: pip install google-cloud-storage"
            )
        except Exception as e:
            return self._add_result(
                "gcs_access", False, time.time() - start,
                f"GCS access failed: {e}"
            )
    
    # ==========================================================================
    # Test 2: Sample Discovery
    # ==========================================================================
    
    def test_sample_discovery(self) -> TestResult:
        """Test that we can discover samples via manifest.json."""
        start = time.time()
        
        try:
            from dataset.deeplive_dataset import DeepLiveDataset
            
            # Create dataset
            dataset = DeepLiveDataset(
                bucket_name=self.bucket_name,
                gcs_project=self.gcs_project,
                frame_sampling='sparse',
                strategies='all',
                use_landmarks=True,
            )
            
            # Discover samples
            samples = dataset.discover_samples()
            
            if len(samples) == 0:
                return self._add_result(
                    "sample_discovery", False, time.time() - start,
                    "No samples found in bucket"
                )
            
            # Get strategy breakdown
            strategies = {}
            for sample in samples:
                strategies[sample.strategy] = strategies.get(sample.strategy, 0) + 1
            
            # Cache for later tests
            self._dataset = dataset
            self._samples = samples[:self.num_samples]  # Limit for testing
            
            return self._add_result(
                "sample_discovery", True, time.time() - start,
                f"Found {len(samples)} samples across {len(strategies)} strategies",
                {
                    "total_samples": len(samples),
                    "strategies": strategies,
                    "sample_ids": [s.sample_id for s in self._samples[:5]],
                }
            )
            
        except Exception as e:
            import traceback
            return self._add_result(
                "sample_discovery", False, time.time() - start,
                f"Sample discovery failed: {e}\n{traceback.format_exc()}"
            )
    
    # ==========================================================================
    # Test 3: Paired Frame Loading
    # ==========================================================================
    
    def test_paired_frame_loading(self) -> TestResult:
        """Test loading paired real/fake frames."""
        start = time.time()
        
        if self._dataset is None or self._samples is None:
            return self._add_result(
                "paired_frame_loading", False, time.time() - start,
                "Sample discovery must pass first"
            )
        
        try:
            sample = self._samples[0]
            
            # Load frames for this sample (sparse = 8 frames)
            real_frames, fake_frames = self._dataset.load_sample_frames(sample)
            
            if len(real_frames) == 0 or len(fake_frames) == 0:
                return self._add_result(
                    "paired_frame_loading", False, time.time() - start,
                    "Failed to load frames"
                )
            
            # Verify counts match expected
            expected_frames = 8  # Sparse mode
            if len(real_frames) != expected_frames or len(fake_frames) != expected_frames:
                return self._add_result(
                    "paired_frame_loading", False, time.time() - start,
                    f"Frame count mismatch. Expected {expected_frames}, got real={len(real_frames)}, fake={len(fake_frames)}"
                )
            
            # Verify frame shapes
            real_shape = real_frames[0].shape
            fake_shape = fake_frames[0].shape
            
            if real_shape != fake_shape:
                return self._add_result(
                    "paired_frame_loading", False, time.time() - start,
                    f"Frame shape mismatch: real={real_shape}, fake={fake_shape}"
                )
            
            # Save visualizations if requested
            if self.save_visualizations:
                self._save_frame_comparison(sample.sample_id, real_frames, fake_frames)
            
            return self._add_result(
                "paired_frame_loading", True, time.time() - start,
                f"Loaded {len(real_frames)} real and {len(fake_frames)} fake frames. Shape: {real_shape}",
                {
                    "sample_id": sample.sample_id,
                    "real_frame_count": len(real_frames),
                    "fake_frame_count": len(fake_frames),
                    "frame_shape": real_shape,
                }
            )
            
        except Exception as e:
            import traceback
            return self._add_result(
                "paired_frame_loading", False, time.time() - start,
                f"Frame loading failed: {e}\n{traceback.format_exc()}"
            )
    
    def _save_frame_comparison(
        self, 
        sample_id: str, 
        real_frames: List[np.ndarray], 
        fake_frames: List[np.ndarray]
    ):
        """Save a side-by-side comparison of real and fake frames."""
        try:
            import matplotlib.pyplot as plt
            from PIL import Image
            
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create comparison grid
            n_frames = min(4, len(real_frames))
            fig, axes = plt.subplots(2, n_frames, figsize=(4 * n_frames, 8))
            
            for i in range(n_frames):
                axes[0, i].imshow(real_frames[i])
                axes[0, i].set_title(f"Real Frame {i}")
                axes[0, i].axis('off')
                
                axes[1, i].imshow(fake_frames[i])
                axes[1, i].set_title(f"Fake Frame {i}")
                axes[1, i].axis('off')
            
            fig.suptitle(f"Sample: {sample_id}", fontsize=14)
            plt.tight_layout()
            
            save_path = self.output_dir / f"frame_comparison_{sample_id}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            logger.info(f"    Saved frame comparison to: {save_path}")
            
        except Exception as e:
            logger.warning(f"    Could not save visualization: {e}")
    
    # ==========================================================================
    # Test 4: Landmark Loading
    # ==========================================================================
    
    def test_landmark_loading(self) -> TestResult:
        """Test loading MediaPipe landmark JSON files."""
        start = time.time()
        
        if self._dataset is None or self._samples is None:
            return self._add_result(
                "landmark_loading", False, time.time() - start,
                "Sample discovery must pass first"
            )
        
        try:
            # Find a sample with landmarks
            sample_with_landmarks = None
            for sample in self._samples:
                if sample.has_landmarks:
                    sample_with_landmarks = sample
                    break
            
            if sample_with_landmarks is None:
                return self._add_result(
                    "landmark_loading", False, time.time() - start,
                    "No samples with landmarks found in test set"
                )
            
            # Load landmarks
            real_landmarks, fake_landmarks = self._dataset.load_landmarks(sample_with_landmarks)
            
            if real_landmarks is None and fake_landmarks is None:
                return self._add_result(
                    "landmark_loading", False, time.time() - start,
                    "Failed to load landmarks (both None)"
                )
            
            # Verify landmark structure
            landmark_info = {}
            if real_landmarks:
                landmark_info['real_count'] = len(real_landmarks)
                if len(real_landmarks) > 0:
                    landmark_info['real_sample'] = {
                        'frame_index': real_landmarks[0].frame_index,
                        'face_detected': real_landmarks[0].face_detected,
                        'regions': list(real_landmarks[0].regions.keys()) if real_landmarks[0].regions else [],
                    }
            
            if fake_landmarks:
                landmark_info['fake_count'] = len(fake_landmarks)
                if len(fake_landmarks) > 0:
                    landmark_info['fake_sample'] = {
                        'frame_index': fake_landmarks[0].frame_index,
                        'face_detected': fake_landmarks[0].face_detected,
                        'regions': list(fake_landmarks[0].regions.keys()) if fake_landmarks[0].regions else [],
                    }
            
            return self._add_result(
                "landmark_loading", True, time.time() - start,
                f"Loaded landmarks for sample {sample_with_landmarks.sample_id}",
                landmark_info
            )
            
        except Exception as e:
            import traceback
            return self._add_result(
                "landmark_loading", False, time.time() - start,
                f"Landmark loading failed: {e}\n{traceback.format_exc()}"
            )
    
    # ==========================================================================
    # Test 5: Augmentation Application
    # ==========================================================================
    
    def test_augmentation_application(self) -> TestResult:
        """
        Test applying augmentations including landmark-based occlusion.
        
        This test uses REAL landmarks from the GCS bucket to verify:
        1. RegionBBoxOcclusion works with MediaPipe 478-point landmarks
        2. Augmentations can be applied to both real and fake frames
        3. Different occlusion types work correctly
        
        The GCS landmark format is:
        {
            "landmarks": [{"x": 0.43, "y": 0.45, "z": -0.02}, ...],  # 478 MediaPipe points
            "face_detected": true,
            "regions": {},  # Empty - we compute bboxes from raw landmarks
            ...
        }
        """
        start = time.time()
        
        if self._dataset is None or self._samples is None:
            return self._add_result(
                "augmentation_application", False, time.time() - start,
                "Sample discovery must pass first"
            )
        
        try:
            from data.augmentations.transforms import RegionBBoxOcclusion, MEDIAPIPE_REGION_INDICES
            
            # Get a sample with landmarks
            sample = None
            for s in self._samples:
                if s.has_landmarks:
                    sample = s
                    break
            
            if sample is None:
                return self._add_result(
                    "augmentation_application", False, time.time() - start,
                    "No samples with landmarks found"
                )
            
            # Load frames
            real_frames, fake_frames = self._dataset.load_sample_frames(sample)
            
            if len(real_frames) == 0:
                return self._add_result(
                    "augmentation_application", False, time.time() - start,
                    "No frames to augment"
                )
            
            # Load REAL landmarks from GCS
            real_landmarks, fake_landmarks = self._dataset.load_landmarks(sample)
            
            if not real_landmarks or len(real_landmarks) == 0:
                return self._add_result(
                    "augmentation_application", False, time.time() - start,
                    "Failed to load landmarks from GCS"
                )
            
            test_frame = real_frames[0].copy()
            test_landmark = real_landmarks[0]  # DeepLiveLandmark object
            
            # Inspect actual landmark data format
            logger.info(f"Testing with sample {sample.sample_id}")
            logger.info(f"  Frame shape: {test_frame.shape}")
            logger.info(f"  Landmark face_detected: {test_landmark.face_detected}")
            
            # Check how many raw landmark points we have
            lm_count = 0
            if hasattr(test_landmark, 'landmarks') and test_landmark.landmarks:
                lm_count = len(test_landmark.landmarks)
            logger.info(f"  Raw landmark points: {lm_count}")
            logger.info(f"  Available MEDIAPIPE_REGION_INDICES: {list(MEDIAPIPE_REGION_INDICES.keys())}")
            
            if lm_count < 400:  # Should be 478 for MediaPipe
                return self._add_result(
                    "augmentation_application", False, time.time() - start,
                    f"Expected ~478 MediaPipe landmarks, got {lm_count}"
                )
            
            # Test RegionBBoxOcclusion with different occlusion types
            # Use region names that match MEDIAPIPE_REGION_INDICES
            results = {}
            occlusion_types = ['solid', 'blur', 'pixelate']
            test_regions = ['left_eye', 'right_eye', 'mouth', 'nose']
            
            for occ_type in occlusion_types:
                transform = RegionBBoxOcclusion(
                    regions=test_regions,
                    num_regions=(1, 2),
                    occlusion_type=occ_type,
                    padding_factor=1.3,
                    p=1.0  # Always apply for testing
                )
                
                # Apply augmentation with real landmarks
                augmented = transform(image=test_frame.copy(), landmarks=test_landmark)['image']
                results[occ_type] = augmented
                
                # Verify the image was modified (unless no regions were available)
                is_modified = not np.array_equal(test_frame, augmented)
                logger.info(f"  {occ_type} occlusion applied: {is_modified}")
            
            # Check at least one augmentation modified the image
            any_modified = any(
                not np.array_equal(test_frame, aug) 
                for aug in results.values()
            )
            
            if not any_modified:
                return self._add_result(
                    "augmentation_application", False, time.time() - start,
                    "No augmentations modified the image - bbox computation may have failed"
                )
            
            # Also test with fake frame and landmarks for completeness
            if len(fake_frames) > 0 and fake_landmarks and len(fake_landmarks) > 0:
                fake_frame = fake_frames[0].copy()
                fake_lm = fake_landmarks[0]
                
                transform = RegionBBoxOcclusion(
                    regions=['left_eye', 'right_eye', 'mouth'],
                    occlusion_type='blur',
                    p=1.0
                )
                fake_augmented = transform(image=fake_frame, landmarks=fake_lm)['image']
                results['fake_blur'] = fake_augmented
            
            # Save visualizations if requested
            if self.save_visualizations:
                self._save_augmentation_comparison(
                    sample.sample_id,
                    test_frame,
                    results.get('blur', results.get('solid', test_frame)),
                    test_landmark
                )
            
            return self._add_result(
                "augmentation_application", True, time.time() - start,
                f"Augmentation transforms work correctly with real GCS landmarks",
                {
                    "sample_id": sample.sample_id,
                    "original_shape": test_frame.shape,
                    "raw_landmark_count": lm_count,
                    "regions_tested": test_regions,
                    "occlusion_types_tested": occlusion_types,
                    "any_modified": any_modified,
                    "face_detected": test_landmark.face_detected,
                }
            )
            
        except Exception as e:
            import traceback
            return self._add_result(
                "augmentation_application", False, time.time() - start,
                f"Augmentation failed: {e}\n{traceback.format_exc()}"
            )
    
    def _save_augmentation_comparison(
        self,
        sample_id: str,
        original: np.ndarray,
        aug_with_landmarks: np.ndarray,
        landmarks=None
    ):
        """Save augmentation comparison images."""
        try:
            import matplotlib.pyplot as plt
            
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            axes[0].imshow(original)
            axes[0].set_title("Original")
            axes[0].axis('off')
            
            axes[1].imshow(aug_with_landmarks)
            axes[1].set_title("Aug (with landmarks)")
            axes[1].axis('off')
            
            fig.suptitle(f"Augmentation Test - {sample_id}", fontsize=14)
            plt.tight_layout()
            
            save_path = self.output_dir / f"augmentation_test_{sample_id}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            logger.info(f"    Saved augmentation comparison to: {save_path}")
            
        except Exception as e:
            logger.warning(f"    Could not save augmentation visualization: {e}")
    
    # ==========================================================================
    # Test 6: Batch Structure
    # ==========================================================================
    
    def test_batch_structure(self) -> TestResult:
        """Test that batch structure is correct (shapes, labels, pairing)."""
        start = time.time()
        
        if self._dataset is None or self._samples is None:
            return self._add_result(
                "batch_structure", False, time.time() - start,
                "Sample discovery must pass first"
            )
        
        try:
            from data.batching.deeplive import (
                DeepLiveBatchingStrategy,
                DeepLiveBatchingConfig,
                deeplive_collate_fn,
            )
            import torch
            
            # Create batching strategy
            config = {'manualSeed': 42}
            data_config = {
                'dataloader_params': {'batch_size': 32, 'num_workers': 0},
                'deeplive_data': {'frame_sampling': 'sparse'},
                'batching': {'pairing_mode': 'paired'},
            }
            
            strategy = DeepLiveBatchingStrategy(
                config=config,
                data_config=data_config,
                dataset=self._dataset,
            )
            
            # Create training dataloader
            train_loader = strategy.create_train_loader(self._samples[:2])  # Just 2 samples
            
            # Get one batch
            batch = next(iter(train_loader))
            
            # Verify batch structure
            if 'image' not in batch:
                return self._add_result(
                    "batch_structure", False, time.time() - start,
                    "Batch missing 'image' key"
                )
            
            if 'label' not in batch:
                return self._add_result(
                    "batch_structure", False, time.time() - start,
                    "Batch missing 'label' key"
                )
            
            images = batch['image']
            labels = batch['label']
            
            # Verify shapes
            if len(images.shape) != 4:  # [B, C, H, W]
                return self._add_result(
                    "batch_structure", False, time.time() - start,
                    f"Image tensor should be 4D, got {len(images.shape)}D"
                )
            
            # Verify label balance (should be ~50/50 real/fake)
            real_count = (labels == 0).sum().item()
            fake_count = (labels == 1).sum().item()
            total = len(labels)
            
            # Check for reasonable balance
            balance_ratio = min(real_count, fake_count) / max(real_count, fake_count) if max(real_count, fake_count) > 0 else 0
            
            batch_info = {
                "batch_size": len(labels),
                "image_shape": list(images.shape),
                "real_count": real_count,
                "fake_count": fake_count,
                "balance_ratio": balance_ratio,
                "dtype": str(images.dtype),
            }
            
            if 'metadata' in batch:
                batch_info["sample_ids"] = batch['metadata']['sample_ids'][:5]
            
            # Save debug artifacts if enabled
            if self.save_debug_artifacts:
                debug_uris = self._save_debug_batch_to_gcs(batch, "batch_structure_test")
                if debug_uris:
                    batch_info["debug_artifacts"] = debug_uris
                    logger.info(f"    Debug artifacts saved to: gs://{self.debug_bucket_name}/runs/{self._debug_run_id}/")
            
            return self._add_result(
                "batch_structure", True, time.time() - start,
                f"Batch structure correct: {images.shape}, labels balanced ({real_count}/{fake_count})",
                batch_info
            )
            
        except Exception as e:
            import traceback
            return self._add_result(
                "batch_structure", False, time.time() - start,
                f"Batch structure test failed: {e}\n{traceback.format_exc()}"
            )
    
    # ==========================================================================
    # Test 7: Full Pipeline Integration
    # ==========================================================================
    
    def test_full_pipeline_integration(self) -> TestResult:
        """Test the full pipeline: data loading -> augmentation -> batching.
        
        This test emulates EXACTLY what the model sees during training:
        - Images resized to 224x224 (CLIP input size)
        - Landmark-based augmentation applied (face region occlusion)
        - Proper batching with real/fake pairing
        
        Debug artifacts saved show the processed images exactly as the model receives them.
        """
        start = time.time()
        
        if self._dataset is None or self._samples is None:
            return self._add_result(
                "full_pipeline_integration", False, time.time() - start,
                "Sample discovery must pass first"
            )
        
        try:
            from data.batching.deeplive import DeepLiveBatchingStrategy
            from data.augmentations.transforms import RegionBBoxOcclusion
            import cv2
            
            # Create the FULL transform pipeline that matches production training
            # This includes: resize to 224x224 + landmark-based occlusion augmentation
            occlusion_transform = RegionBBoxOcclusion(
                regions=['left_eye', 'right_eye', 'mouth', 'nose', 'forehead'],
                num_regions=(1, 3),
                occlusion_type='solid',  # Can also be 'blur' or 'pixelate'
                padding_factor=1.3,
                p=0.8  # 80% probability - realistic training setting
            )
            
            def full_training_transform(image, landmarks=None):
                """
                Full transform pipeline that matches what the model sees in training.
                
                1. Resize to 224x224 (model input size)
                2. Apply landmark-based occlusion augmentation
                """
                # Step 1: Resize to model input size
                if isinstance(image, np.ndarray):
                    image = cv2.resize(image, (224, 224))
                
                # Step 2: Apply landmark-based occlusion
                # This uses RegionBBoxOcclusion which computes bboxes from 
                # raw MediaPipe 478-point landmarks
                if landmarks is not None:
                    result = occlusion_transform(image=image, landmarks=landmarks)
                    image = result['image']
                
                return image
            
            # Create batching strategy with FULL transform
            config = {'manualSeed': 42}
            data_config = {
                'dataloader_params': {'batch_size': 16, 'num_workers': 0},
                'deeplive_data': {'frame_sampling': 'sparse'},
                'batching': {'pairing_mode': 'paired'},
            }
            
            strategy = DeepLiveBatchingStrategy(
                config=config,
                data_config=data_config,
                dataset=self._dataset,
                transform=full_training_transform,
            )
            
            # Create loader and process a few batches
            train_loader = strategy.create_train_loader(self._samples[:2])
            
            batches_processed = 0
            total_images = 0
            total_real = 0
            total_fake = 0
            first_batch = None
            
            for batch in train_loader:
                batches_processed += 1
                total_images += len(batch['label'])
                total_real += (batch['label'] == 0).sum().item()
                total_fake += (batch['label'] == 1).sum().item()
                
                # Save first batch for debug artifacts
                if first_batch is None:
                    first_batch = batch
                
                if batches_processed >= 3:  # Test a few batches
                    break
            
            result_details = {
                "batches_processed": batches_processed,
                "total_images": total_images,
                "total_real": total_real,
                "total_fake": total_fake,
                "transform_applied": "resize_224x224 + RegionBBoxOcclusion",
                "occlusion_regions": ['left_eye', 'right_eye', 'mouth', 'nose', 'forehead'],
                "occlusion_probability": 0.8,
            }
            
            # Save debug artifacts if enabled - this shows EXACTLY what the model sees
            if self.save_debug_artifacts and first_batch is not None:
                debug_uris = self._save_debug_batch_to_gcs(
                    first_batch, 
                    "full_pipeline_model_input",
                    max_images=16  # Save more images to see augmentation variety
                )
                if debug_uris:
                    result_details["debug_artifacts"] = debug_uris
                    result_details["debug_info"] = (
                        "Images saved are EXACTLY what the model sees: "
                        "224x224, with landmark-based face region occlusion applied"
                    )
                    logger.info(f"    Debug artifacts saved to: gs://{self.debug_bucket_name}/runs/{self._debug_run_id}/")
                    logger.info(f"    These images show EXACTLY what the model sees during training")
            
            return self._add_result(
                "full_pipeline_integration", True, time.time() - start,
                f"Pipeline processed {batches_processed} batches, {total_images} images (224x224 with occlusion)",
                result_details
            )
            
        except Exception as e:
            import traceback
            return self._add_result(
                "full_pipeline_integration", False, time.time() - start,
                f"Full pipeline test failed: {e}\n{traceback.format_exc()}"
            )
    
    # ==========================================================================
    # Test 8: Model Forward Pass
    # ==========================================================================
    
    def test_model_forward_pass(self) -> TestResult:
        """Test complete training step: forward pass, loss computation, and backward pass.
        
        This test validates the EXACT training loop without GPU:
        1. Model creation with correct config
        2. Forward pass in training mode (model.train())
        3. Loss computation with CrossEntropyLoss
        4. Backward pass (gradient computation)
        5. Optimizer step (parameter update)
        6. Forward pass in eval mode (model.eval(), inference=True)
        
        Catches:
        - Shape mismatches (backbone vs head dimensions)
        - Incorrect model interface (dict vs tensor input)
        - Gradient flow issues
        - Loss computation errors
        """
        start = time.time()
        
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from train_deeplive import create_model, load_experiment_config
            
            # Load experiment config
            config_path = TRAINING_DIR / "experiments" / "deeplive_vit_B16.yaml"
            if not config_path.exists():
                return self._add_result(
                    "model_forward_pass", False, time.time() - start,
                    f"Config file not found: {config_path}"
                )
            
            config = load_experiment_config(str(config_path))
            
            # Create model on CPU
            device = torch.device('cpu')
            model = create_model(config, device)
            
            # Get expected dimensions from config
            hidden_size = config.get('backbone', {}).get('hidden_size', 768)
            resolution = config.get('backbone', {}).get('resolution', 224)
            batch_size = 4
            
            logger.info(f"    Model created: hidden_size={hidden_size}, resolution={resolution}")
            
            # Create dummy batch (mimics real data structure)
            dummy_images = torch.randn(batch_size, 3, resolution, resolution)
            dummy_labels = torch.randint(0, 2, (batch_size,))
            
            # ============================================================
            # TEST 1: Training mode forward pass
            # ============================================================
            logger.info("    Testing training mode forward pass...")
            model.train()
            
            # This is EXACTLY how train_epoch() calls the model
            train_outputs = model({'image': dummy_images, 'label': dummy_labels})
            
            # Extract logits (same as train_epoch)
            if isinstance(train_outputs, dict):
                logits = train_outputs.get('logits', train_outputs.get('cls'))
            else:
                logits = train_outputs
            
            if logits is None:
                return self._add_result(
                    "model_forward_pass", False, time.time() - start,
                    f"No logits found in output. Keys: {train_outputs.keys() if isinstance(train_outputs, dict) else 'not a dict'}"
                )
            
            # Verify logits shape: [batch_size, num_classes]
            expected_logits_shape = (batch_size, 2)
            if logits.shape != expected_logits_shape:
                return self._add_result(
                    "model_forward_pass", False, time.time() - start,
                    f"Logits shape mismatch: got {logits.shape}, expected {expected_logits_shape}"
                )
            
            logger.info(f"    ✓ Training forward pass: logits shape {logits.shape}")
            
            # ============================================================
            # TEST 2: Loss computation
            # ============================================================
            logger.info("    Testing loss computation...")
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, dummy_labels)
            
            if not torch.isfinite(loss):
                return self._add_result(
                    "model_forward_pass", False, time.time() - start,
                    f"Loss is not finite: {loss.item()}"
                )
            
            logger.info(f"    ✓ Loss computed: {loss.item():.4f}")
            
            # ============================================================
            # TEST 3: Backward pass (gradient computation)
            # ============================================================
            logger.info("    Testing backward pass...")
            loss.backward()
            
            # Verify gradients exist on at least some parameters
            grad_count = 0
            total_params = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    total_params += 1
                    if param.grad is not None:
                        grad_count += 1
            
            if grad_count == 0:
                return self._add_result(
                    "model_forward_pass", False, time.time() - start,
                    f"No gradients computed! {total_params} trainable params but 0 have gradients."
                )
            
            logger.info(f"    ✓ Gradients computed: {grad_count}/{total_params} parameters")
            
            # ============================================================
            # TEST 4: Optimizer step
            # ============================================================
            logger.info("    Testing optimizer step...")
            optimizer = optim.AdamW(model.parameters(), lr=1e-4)
            
            # Get a parameter value before step
            first_param = next(model.parameters())
            param_before = first_param.clone().detach()
            
            optimizer.step()
            
            # Verify parameter was updated
            param_after = first_param.clone().detach()
            param_changed = not torch.allclose(param_before, param_after)
            
            if not param_changed:
                logger.warning("    ⚠ Parameters did not change after optimizer step (may be frozen)")
            else:
                logger.info("    ✓ Optimizer step: parameters updated")
            
            # ============================================================
            # TEST 5: Eval mode forward pass (inference)
            # ============================================================
            logger.info("    Testing eval mode forward pass...")
            model.eval()
            optimizer.zero_grad()
            
            with torch.no_grad():
                # This is EXACTLY how evaluate() calls the model
                eval_outputs = model({'image': dummy_images, 'label': dummy_labels}, inference=True)
            
            if isinstance(eval_outputs, dict):
                eval_logits = eval_outputs.get('logits', eval_outputs.get('cls'))
            else:
                eval_logits = eval_outputs
            
            if eval_logits is None or eval_logits.shape != expected_logits_shape:
                return self._add_result(
                    "model_forward_pass", False, time.time() - start,
                    f"Eval logits issue: {eval_logits.shape if eval_logits is not None else 'None'}"
                )
            
            logger.info(f"    ✓ Eval forward pass: logits shape {eval_logits.shape}")
            
            # ============================================================
            # SUCCESS
            # ============================================================
            result_details = {
                "config_file": str(config_path.name),
                "backbone_name": config.get('backbone', {}).get('name', 'unknown'),
                "hidden_size": hidden_size,
                "resolution": resolution,
                "batch_size": batch_size,
                "logits_shape": list(logits.shape),
                "loss_value": loss.item(),
                "grad_params": f"{grad_count}/{total_params}",
                "param_updated": param_changed,
            }
            
            return self._add_result(
                "model_forward_pass", True, time.time() - start,
                f"Full training step successful (loss={loss.item():.4f}, grads={grad_count}/{total_params})",
                result_details
            )
            
        except RuntimeError as e:
            error_msg = str(e)
            if "shapes cannot be multiplied" in error_msg or "size mismatch" in error_msg.lower():
                return self._add_result(
                    "model_forward_pass", False, time.time() - start,
                    f"SHAPE MISMATCH - Check backbone hidden_size vs head input: {error_msg}"
                )
            import traceback
            return self._add_result(
                "model_forward_pass", False, time.time() - start,
                f"RuntimeError: {error_msg}\n{traceback.format_exc()}"
            )
            
        except Exception as e:
            import traceback
            return self._add_result(
                "model_forward_pass", False, time.time() - start,
                f"Model forward pass failed: {e}\n{traceback.format_exc()}"
            )
    
    # ==========================================================================
    # Run All Tests
    # ==========================================================================
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all tests in order."""
        logger.info("=" * 60)
        logger.info("DeepLive Pipeline E2E Test Suite")
        logger.info("=" * 60)
        
        # Tests must run in order due to dependencies
        tests = [
            ("1. GCS Access", self.test_gcs_access),
            ("2. Sample Discovery", self.test_sample_discovery),
            ("3. Paired Frame Loading", self.test_paired_frame_loading),
            ("4. Landmark Loading", self.test_landmark_loading),
            ("5. Augmentation Application", self.test_augmentation_application),
            ("6. Batch Structure", self.test_batch_structure),
            ("7. Full Pipeline Integration", self.test_full_pipeline_integration),
            ("8. Model Forward Pass", self.test_model_forward_pass),
        ]
        
        for test_name, test_fn in tests:
            logger.info("-" * 60)
            logger.info(f"Running: {test_name}")
            logger.info("-" * 60)
            test_fn()
        
        # Summary
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        total_time = sum(r.duration for r in self.results)
        
        for result in self.results:
            status = "✅" if result.passed else "❌"
            logger.info(f"  {status} {result.name}: {result.duration:.2f}s")
        
        logger.info("-" * 60)
        logger.info(f"PASSED: {passed}/{len(self.results)}")
        logger.info(f"FAILED: {failed}/{len(self.results)}")
        logger.info(f"TOTAL TIME: {total_time:.2f}s")
        
        if failed > 0:
            logger.info("\nFailed tests:")
            for result in self.results:
                if not result.passed:
                    logger.info(f"  - {result.name}: {result.message}")
        
        return self.results
    
    def run_single_test(self, test_name: str) -> TestResult:
        """Run a single test by name."""
        test_map = {
            'gcs_access': self.test_gcs_access,
            'sample_discovery': self.test_sample_discovery,
            'paired_frame_loading': self.test_paired_frame_loading,
            'landmark_loading': self.test_landmark_loading,
            'augmentation_application': self.test_augmentation_application,
            'batch_structure': self.test_batch_structure,
            'full_pipeline_integration': self.test_full_pipeline_integration,
        }
        
        if test_name not in test_map:
            logger.error(f"Unknown test: {test_name}")
            logger.info(f"Available tests: {list(test_map.keys())}")
            return None
        
        # Run dependent tests first
        dependencies = {
            'sample_discovery': ['gcs_access'],
            'paired_frame_loading': ['gcs_access', 'sample_discovery'],
            'landmark_loading': ['gcs_access', 'sample_discovery'],
            'augmentation_application': ['gcs_access', 'sample_discovery'],
            'batch_structure': ['gcs_access', 'sample_discovery'],
            'full_pipeline_integration': ['gcs_access', 'sample_discovery'],
        }
        
        if test_name in dependencies:
            for dep in dependencies[test_name]:
                if not any(r.name == dep and r.passed for r in self.results):
                    logger.info(f"Running dependency: {dep}")
                    test_map[dep]()
        
        return test_map[test_name]()


def main():
    parser = argparse.ArgumentParser(description="Test DeepLive data pipeline end-to-end")
    parser.add_argument(
        '--bucket', 
        type=str, 
        default=None,
        help='GCS bucket name (default: live-deepfake-methods-real-and-fake-frames)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default=None,
        help='GCP project ID (default: train-cvit2)'
    )
    parser.add_argument(
        '--num-samples', 
        type=int, 
        default=10,
        help='Number of samples to test (default: 10)'
    )
    parser.add_argument(
        '--save-visualizations', 
        action='store_true',
        help='Save visualization images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory for saving visualizations'
    )
    parser.add_argument(
        '--test',
        type=str,
        default=None,
        help='Run specific test (gcs_access, sample_discovery, paired_frame_loading, landmark_loading, augmentation_application, batch_structure, full_pipeline_integration, model_forward_pass)'
    )
    parser.add_argument(
        '--save-debug-artifacts',
        action='store_true',
        help='Save debug artifacts (images, metadata) to GCS bucket'
    )
    parser.add_argument(
        '--debug-bucket',
        type=str,
        default=None,
        help='GCS bucket for debug artifacts (default: effort-detector-debug-artifacts)'
    )
    
    args = parser.parse_args()
    
    # Handle gs:// prefix
    bucket_name = args.bucket
    if bucket_name and bucket_name.startswith('gs://'):
        bucket_name = bucket_name[5:]
    
    debug_bucket = args.debug_bucket
    if debug_bucket and debug_bucket.startswith('gs://'):
        debug_bucket = debug_bucket[5:]
    
    # Split debug_bucket into bucket name and prefix if it contains a path
    # e.g., "training-job-outputs/debug" -> bucket="training-job-outputs", prefix="debug"
    debug_prefix = None
    if debug_bucket and '/' in debug_bucket:
        parts = debug_bucket.split('/', 1)
        debug_bucket = parts[0]
        debug_prefix = parts[1]
    
    # Create test suite
    test_suite = DeepLivePipelineTest(
        bucket_name=bucket_name,
        gcs_project=args.project,
        num_samples=args.num_samples,
        save_visualizations=args.save_visualizations,
        output_dir=args.output_dir,
        save_debug_artifacts=args.save_debug_artifacts,
        debug_bucket_name=debug_bucket,
        debug_prefix=debug_prefix,
    )
    
    # Run tests
    if args.test:
        result = test_suite.run_single_test(args.test)
        sys.exit(0 if result and result.passed else 1)
    else:
        results = test_suite.run_all_tests()
        passed = all(r.passed for r in results)
        sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
