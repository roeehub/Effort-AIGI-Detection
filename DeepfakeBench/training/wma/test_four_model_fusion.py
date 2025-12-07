#!/usr/bin/env python3
"""
Standalone test for four-model ensemble fusion system.

This script mimics how the WMA server will work with four models:
1. Send frames (1-32) to 4 different model servers
2. Aggregate predictions per model (topk4 or softmax_b5)
3. Apply isotonic calibration
4. Fuse with Noisy-OR
5. Classify with three-way thresholds

Usage:
    # Test with mock predictions (no actual API calls)
    python test_four_model_fusion.py --mock
    
    # Test with real API calls (when servers are ready)
    python test_four_model_fusion.py --participant-name "John Doe" --num-frames 27
"""

import argparse
import asyncio
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

import numpy as np
import aiohttp
import cv2

# Add parent directory to path to import video_preprocessor
sys.path.insert(0, str(Path(__file__).parent.parent))
from video_preprocessor import _find_and_prepare_faces

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION - UPDATE THESE WITH YOUR ACTUAL MODEL SERVER IPs
# ═══════════════════════════════════════════════════════════════════════════

MODEL_SERVERS = {
    '9rfa62j1': {  # Model 1 - Cluster: EFS+Platform - deepfakebench4
        'host': '34.14.89.109',
        'port': '8998',
        'aggregator': 'topk4',
        'endpoint': '/check_frame_batch'
    },
    '1mjgo9w1': {  # Model 2 - Cluster: FaceSwap - deepfakebench3
        'host': '34.76.180.225',
        'port': '8998',
        'aggregator': 'softmax_b5',
        'endpoint': '/check_frame_batch'
    },
    'dfsesrgu': {  # Model 3 - Cluster: Reenact - deepfakebench5
        'host': '34.86.88.240',
        'port': '8998',
        'aggregator': 'topk4',
        'endpoint': '/check_frame_batch'
    },
    '4vtny88m': {  # Model 4 - Cluster: noisy_or - deepfakebench2
        'host': '34.16.217.28',
        'port': '8998',
        'aggregator': 'topk4',
        'endpoint': '/check_frame_batch'
    }
}

# Thresholds from your training (detection strategy results)
THRESHOLDS = {
    'T_low': 0.996700,   # Below this = REAL
    'T_high': 0.998248,  # Above this = FAKE
    # Between = UNCERTAIN
}

# API configuration
API_TIMEOUT = 30  # seconds
YOLO_CONF_THRESHOLD = 0.80


# ═══════════════════════════════════════════════════════════════════════════
# AGGREGATION FUNCTIONS (from run_video_level_fusion_v2.py)
# ═══════════════════════════════════════════════════════════════════════════

def topk_mean(x: np.ndarray, k: int = 4) -> float:
    """
    Average of top k highest values.
    This selects the most "suspicious" frames.
    
    For k=4 with ~27 frames, this is top ~15% of frames.
    """
    if x.size == 0:
        return 0.0
    k = max(1, min(k, x.size))
    idx = np.argpartition(x, -k)[-k:]
    result = float(np.mean(x[idx]))
    logger.debug(f"topk_mean(k={k}): input_size={x.size}, top_k_mean={result:.4f}")
    return result


def softmax_pool(x: np.ndarray, beta: float = 5.0) -> float:
    """
    Softmax pooling with temperature parameter beta.
    Higher beta = more weight to maximum values.
    
    This is a smooth approximation to max pooling.
    """
    if x.size == 0:
        return 0.0
    # Stabilized log-sum-exp
    m = np.max(beta * x)
    result = float((np.log(np.mean(np.exp(beta * x - m))) + m) / beta)
    logger.debug(f"softmax_pool(β={beta}): input_size={x.size}, result={result:.4f}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# FUSION FUNCTION (from run_video_level_fusion_v2.py)
# ═══════════════════════════════════════════════════════════════════════════

def noisy_or_fusion(calibrated_probs: Dict[str, float]) -> float:
    """
    Noisy-OR fusion: P(fake) = 1 - ∏(1 - p_i)
    
    Assumes models make independent errors.
    If all models agree, confidence approaches 1.0.
    
    Example:
        If all 4 models say 0.90: 1 - (0.1)^4 = 0.9999 ✓
        If all 4 models say 0.95: 1 - (0.05)^4 = 0.99999937 ✓
    
    This is why your thresholds are so high (0.9967 and 0.9982).
    """
    probs_array = np.array(list(calibrated_probs.values()))
    fusion_score = float(1.0 - np.prod(1.0 - probs_array))
    
    logger.debug(f"Noisy-OR fusion:")
    for model_id, prob in calibrated_probs.items():
        logger.debug(f"  {model_id}: {prob:.6f}")
    logger.debug(f"  → Fused: {fusion_score:.6f}")
    
    return fusion_score


# ═══════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def classify_score(fusion_score: float) -> Tuple[str, str]:
    """
    Three-way classification based on fusion score.
    
    Returns:
        (verdict, confidence_level)
        verdict: 'REAL', 'UNCERTAIN', or 'FAKE'
        confidence_level: 'High', 'Medium', 'Low', or 'Uncertain'
    """
    if fusion_score < THRESHOLDS['T_low']:
        verdict = 'REAL'
        # For REAL: lower score = higher confidence
        if fusion_score < THRESHOLDS['T_low'] * 0.5:
            confidence = 'High'
        elif fusion_score < THRESHOLDS['T_low'] * 0.75:
            confidence = 'Medium'
        else:
            confidence = 'Low'
    elif fusion_score > THRESHOLDS['T_high']:
        verdict = 'FAKE'
        # For FAKE: higher score = higher confidence
        range_span = 1.0 - THRESHOLDS['T_high']
        relative_score = fusion_score - THRESHOLDS['T_high']
        if relative_score < range_span * 0.33:
            confidence = 'Low'
        elif relative_score < range_span * 0.67:
            confidence = 'Medium'
        else:
            confidence = 'High'
    else:
        verdict = 'UNCERTAIN'
        confidence = 'Uncertain'
    
    return verdict, confidence


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATOR MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

class CalibratorManager:
    """
    Manages loading and applying isotonic calibrators for each model.
    """
    
    def __init__(self, calibrators_path: Path = None):
        """
        Args:
            calibrators_path: Path to saved calibrators (pickle file).
                             If None, uses mock calibrators (identity function).
        """
        self.calibrators = {}
        
        if calibrators_path and calibrators_path.exists():
            logger.info(f"Loading calibrators from {calibrators_path}")
            with open(calibrators_path, 'rb') as f:
                self.calibrators = pickle.load(f)
            logger.info(f"Loaded {len(self.calibrators)} calibrators")
        else:
            logger.warning("No calibrators provided - using identity function (no calibration)")
            # Mock calibrators that just return input
            for model_id in MODEL_SERVERS.keys():
                self.calibrators[model_id] = MockCalibrator()
    
    def calibrate(self, model_id: str, score: float) -> float:
        """Apply calibration to a single aggregated score."""
        if model_id not in self.calibrators:
            logger.error(f"No calibrator found for model {model_id}")
            return score
        
        calibrated = self.calibrators[model_id].transform([score])[0]
        logger.debug(f"Calibration [{model_id}]: {score:.6f} → {calibrated:.6f}")
        return float(calibrated)


class MockCalibrator:
    """Mock calibrator for testing without real calibration."""
    def transform(self, X):
        return np.array(X)  # Identity function


# ═══════════════════════════════════════════════════════════════════════════
# FOUR MODEL API CLIENT
# ═══════════════════════════════════════════════════════════════════════════

class FourModelAPIClient:
    """
    Handles communication with 4 model servers and fusion logic.
    """
    
    def __init__(self, calibrator_manager: CalibratorManager):
        """
        Args:
            calibrator_manager: Manager for isotonic calibrators
        """
        self.calibrator_manager = calibrator_manager
        self.aggregator_functions = {
            'topk4': lambda x: topk_mean(x, k=4),
            'softmax_b5': lambda x: softmax_pool(x, beta=5.0),
        }
        
        # Build API URLs
        self.api_urls = {}
        for model_id, config in MODEL_SERVERS.items():
            self.api_urls[model_id] = f"http://{config['host']}:{config['port']}{config['endpoint']}"
        
        logger.info("=" * 80)
        logger.info("FourModelAPIClient initialized")
        logger.info("=" * 80)
        for model_id, config in MODEL_SERVERS.items():
            logger.info(f"  {model_id}:")
            logger.info(f"    URL: {self.api_urls[model_id]}")
            logger.info(f"    Aggregator: {config['aggregator']}")
        logger.info(f"  Thresholds: T_low={THRESHOLDS['T_low']:.6f}, T_high={THRESHOLDS['T_high']:.6f}")
        logger.info("=" * 80)
    
    async def call_single_model_api(self, 
                                     model_id: str, 
                                     image_bytes_list: List[bytes]) -> List[float]:
        """
        Call a single model's API endpoint to get per-frame probabilities.
        
        Args:
            model_id: Model identifier (e.g., '9rfa62j1')
            image_bytes_list: List of JPEG-encoded frame bytes
        
        Returns:
            List of probabilities (one per frame), empty list on error
        """
        if not image_bytes_list:
            return []
        
        api_url = self.api_urls[model_id]
        
        try:
            logger.info(f"[{model_id}] Sending {len(image_bytes_list)} frames to {api_url}")
            
            # Prepare multipart form data
            data = aiohttp.FormData()
            for i, img_bytes in enumerate(image_bytes_list):
                data.add_field('files', img_bytes, 
                             filename=f'frame_{i}.jpg', 
                             content_type='image/jpeg')
            
            # API parameters (matching current WMA server)
            params = {
                'model_type': 'custom',
                'threshold': '0.75',  # Not used in our pipeline, but API expects it
                'debug': 'false',
                'yolo_conf_threshold': str(YOLO_CONF_THRESHOLD)
            }
            
            timeout = aiohttp.ClientTimeout(total=API_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(api_url, data=data, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        probs = result.get('probs', [])
                        logger.info(f"[{model_id}] Received {len(probs)} probabilities")
                        return probs
                    else:
                        error_text = await response.text()
                        logger.error(f"[{model_id}] API error {response.status}: {error_text}")
                        return []
        
        except asyncio.TimeoutError:
            logger.error(f"[{model_id}] API timeout after {API_TIMEOUT}s")
            return []
        except Exception as e:
            logger.error(f"[{model_id}] API call failed: {e}")
            return []
    
    async def get_all_model_predictions(self, 
                                       image_bytes_list: List[bytes]) -> Dict[str, List[float]]:
        """
        Call all 4 model APIs in parallel.
        
        Args:
            image_bytes_list: List of JPEG-encoded frame bytes
        
        Returns:
            Dict mapping model_id to list of per-frame probabilities
        """
        logger.info(f"Calling all 4 models in parallel with {len(image_bytes_list)} frames...")
        
        tasks = [
            self.call_single_model_api(model_id, image_bytes_list)
            for model_id in MODEL_SERVERS.keys()
        ]
        
        results = await asyncio.gather(*tasks)
        
        predictions = {
            model_id: probs 
            for model_id, probs in zip(MODEL_SERVERS.keys(), results)
        }
        
        # Log summary
        for model_id, probs in predictions.items():
            if probs:
                logger.info(f"[{model_id}] Got {len(probs)} predictions, "
                          f"mean={np.mean(probs):.4f}, max={np.max(probs):.4f}")
            else:
                logger.warning(f"[{model_id}] No predictions returned (likely no faces detected)")
        
        return predictions
    
    def aggregate_and_calibrate(self, 
                                predictions: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Aggregate per-frame predictions and apply calibration for each model.
        
        Args:
            predictions: Dict mapping model_id to list of per-frame probabilities
        
        Returns:
            Dict mapping model_id to calibrated aggregated score
        """
        logger.info("Aggregating predictions per model...")
        
        calibrated_scores = {}
        
        for model_id in MODEL_SERVERS.keys():
            probs = predictions.get(model_id, [])
            
            if not probs:
                logger.warning(f"[{model_id}] No predictions to aggregate")
                calibrated_scores[model_id] = 0.0
                continue
            
            # Get aggregator function for this model
            aggregator_name = MODEL_SERVERS[model_id]['aggregator']
            aggregator_func = self.aggregator_functions[aggregator_name]
            
            # Aggregate
            probs_array = np.array(probs)
            aggregated = aggregator_func(probs_array)
            
            # Calibrate
            calibrated = self.calibrator_manager.calibrate(model_id, aggregated)
            
            calibrated_scores[model_id] = calibrated
            
            logger.info(f"[{model_id}] {aggregator_name}: "
                       f"{len(probs)} frames → aggregated={aggregated:.6f} → "
                       f"calibrated={calibrated:.6f}")
        
        return calibrated_scores
    
    async def process_frames(self, 
                            image_bytes_list: List[bytes],
                            participant_name: str = "test_participant") -> Dict:
        """
        Full pipeline: API calls → aggregation → calibration → fusion → decision.
        
        This is the main function you'll integrate into participant_manager.py.
        
        Args:
            image_bytes_list: List of JPEG-encoded frame bytes (1-32 frames)
            participant_name: Name for logging
        
        Returns:
            Dict with complete results
        """
        logger.info("=" * 80)
        logger.info(f"PROCESSING {len(image_bytes_list)} frames for '{participant_name}'")
        logger.info("=" * 80)
        
        # Step 1: Get predictions from all 4 models (parallel)
        predictions = await self.get_all_model_predictions(image_bytes_list)
        
        # Check if any model returned predictions
        if not any(predictions.values()):
            logger.warning("No faces detected by any model - returning REAL verdict")
            return {
                'verdict': 'REAL',
                'confidence': 'High',
                'fusion_score': 0.0,
                'calibrated_scores': {},
                'predictions': predictions,
                'participant_name': participant_name,
                'num_frames': len(image_bytes_list)
            }
        
        # Step 2: Aggregate per model and calibrate
        calibrated_scores = self.aggregate_and_calibrate(predictions)
        
        # Step 3: Fuse with Noisy-OR
        fusion_score = noisy_or_fusion(calibrated_scores)
        
        # Step 4: Classify
        verdict, confidence = classify_score(fusion_score)
        
        # Log final result
        logger.info("=" * 80)
        logger.info("FINAL RESULT")
        logger.info("=" * 80)
        logger.info(f"  Participant: {participant_name}")
        logger.info(f"  Frames: {len(image_bytes_list)}")
        logger.info(f"  Fusion Score: {fusion_score:.6f}")
        logger.info(f"  Verdict: {verdict}")
        logger.info(f"  Confidence: {confidence}")
        logger.info("=" * 80)
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'fusion_score': fusion_score,
            'calibrated_scores': calibrated_scores,
            'predictions': predictions,
            'participant_name': participant_name,
            'num_frames': len(image_bytes_list),
            'thresholds': THRESHOLDS
        }


# ═══════════════════════════════════════════════════════════════════════════
# VIDEO FRAME EXTRACTION (using video_preprocessor.py)
# ═══════════════════════════════════════════════════════════════════════════

def extract_frames_from_video(video_path: str, 
                              max_frames: int = 32,
                              pre_method: str = 'yolo') -> List[bytes]:
    """
    Extract uniformly sampled frames from a video using video_preprocessor.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract (8-32)
        pre_method: Face detection method ('yolo', 'yolo_haar')
    
    Returns:
        List of JPEG-encoded frame bytes
    """
    logger.info(f"Extracting frames from video: {video_path}")
    logger.info(f"  Method: {pre_method}, Max frames: {max_frames}")
    
    # Clamp max_frames to 8-32 range
    max_frames = max(8, min(32, max_frames))
    
    # Use the video_preprocessor to extract face-cropped frames
    # This returns a list of numpy arrays (BGR format)
    collected_faces = _find_and_prepare_faces(
        video_path=video_path,
        pre_method=pre_method,
        debug_save_path=None,
        debug_frames_count=None
    )
    
    if not collected_faces:
        logger.error(f"Failed to extract faces from video: {video_path}")
        return []
    
    # Limit to max_frames
    collected_faces = collected_faces[:max_frames]
    
    logger.info(f"Extracted {len(collected_faces)} face frames from video")
    
    # Convert numpy arrays (BGR) to JPEG bytes
    image_bytes_list = []
    for i, face_bgr in enumerate(collected_faces):
        # Encode as JPEG
        success, buffer = cv2.imencode('.jpg', face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if success:
            image_bytes_list.append(buffer.tobytes())
        else:
            logger.warning(f"Failed to encode frame {i} as JPEG")
    
    logger.info(f"Successfully encoded {len(image_bytes_list)} frames as JPEG")
    
    return image_bytes_list


# ═══════════════════════════════════════════════════════════════════════════
# MOCK DATA GENERATOR (for testing without real API calls)
# ═══════════════════════════════════════════════════════════════════════════

def generate_mock_predictions(num_frames: int, is_fake: bool = False) -> Dict[str, List[float]]:
    """
    Generate mock predictions for testing.
    
    Args:
        num_frames: Number of frames
        is_fake: If True, generate high scores (fake), else low scores (real)
    
    Returns:
        Dict mapping model_id to list of probabilities
    """
    logger.info(f"Generating mock predictions: {num_frames} frames, is_fake={is_fake}")
    
    predictions = {}
    
    for model_id in MODEL_SERVERS.keys():
        if is_fake:
            # Fake: high scores with some variation
            base = np.random.uniform(0.85, 0.95, num_frames)
            noise = np.random.normal(0, 0.02, num_frames)
            probs = np.clip(base + noise, 0, 1)
        else:
            # Real: low scores with some variation
            base = np.random.uniform(0.05, 0.25, num_frames)
            noise = np.random.normal(0, 0.02, num_frames)
            probs = np.clip(base + noise, 0, 1)
        
        predictions[model_id] = probs.tolist()
        logger.debug(f"[{model_id}] Mock probs: mean={np.mean(probs):.4f}, "
                    f"min={np.min(probs):.4f}, max={np.max(probs):.4f}")
    
    return predictions


async def test_with_mock_data():
    """Test the pipeline with mock data (no API calls)."""
    logger.info("=" * 80)
    logger.info("TESTING WITH MOCK DATA")
    logger.info("=" * 80)
    
    # Initialize (no calibrators = identity function)
    calibrator_mgr = CalibratorManager()
    client = FourModelAPIClient(calibrator_mgr)
    
    # Test case 1: Real video (27 frames)
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Real video (low scores)")
    logger.info("=" * 80)
    mock_preds_real = generate_mock_predictions(27, is_fake=False)
    
    # Since we're not calling APIs, directly test aggregation
    calibrated_real = client.aggregate_and_calibrate(mock_preds_real)
    fusion_real = noisy_or_fusion(calibrated_real)
    verdict_real, conf_real = classify_score(fusion_real)
    
    logger.info(f"Result: {verdict_real} (confidence: {conf_real}, score: {fusion_real:.6f})")
    
    # Test case 2: Fake video (27 frames)
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Fake video (high scores)")
    logger.info("=" * 80)
    mock_preds_fake = generate_mock_predictions(27, is_fake=True)
    
    calibrated_fake = client.aggregate_and_calibrate(mock_preds_fake)
    fusion_fake = noisy_or_fusion(calibrated_fake)
    verdict_fake, conf_fake = classify_score(fusion_fake)
    
    logger.info(f"Result: {verdict_fake} (confidence: {conf_fake}, score: {fusion_fake:.6f})")
    
    # Test case 3: Edge case - only 5 frames
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Short sequence (5 frames, fake)")
    logger.info("=" * 80)
    mock_preds_short = generate_mock_predictions(5, is_fake=True)
    
    calibrated_short = client.aggregate_and_calibrate(mock_preds_short)
    fusion_short = noisy_or_fusion(calibrated_short)
    verdict_short, conf_short = classify_score(fusion_short)
    
    logger.info(f"Result: {verdict_short} (confidence: {conf_short}, score: {fusion_short:.6f})")


async def test_with_real_api(participant_name: str, 
                            num_frames: int, 
                            image_paths: List[str] = None,
                            image_bytes_list: List[bytes] = None):
    """
    Test with real API calls (requires model servers to be running).
    
    Args:
        participant_name: Name for logging
        num_frames: Number of frames to send
        image_paths: Optional list of image file paths. If None, image_bytes_list must be provided.
        image_bytes_list: Optional pre-loaded image bytes (e.g., from video extraction)
    """
    logger.info("=" * 80)
    logger.info("TESTING WITH REAL API CALLS")
    logger.info("=" * 80)
    
    # Initialize
    # TODO: Update this path when you export calibrators from training
    calibrators_path = Path("../combined_model/out_full_v3/calibrators.pkl")
    calibrator_mgr = CalibratorManager(calibrators_path)
    client = FourModelAPIClient(calibrator_mgr)
    
    # Load image bytes
    if image_bytes_list:
        # Already have bytes (from video extraction)
        logger.info(f"Using {len(image_bytes_list)} pre-extracted frames from video")
    elif image_paths:
        logger.info(f"Loading {len(image_paths)} images from disk...")
        image_bytes_list = []
        for img_path in image_paths[:num_frames]:
            with open(img_path, 'rb') as f:
                image_bytes_list.append(f.read())
    else:
        logger.error("No image paths or pre-loaded bytes provided for real API test")
        logger.info("You need to provide either --image-dir or --video")
        return
    
    # Run full pipeline
    result = await client.process_frames(image_bytes_list, participant_name)
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("API TEST COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Verdict: {result['verdict']}")
    logger.info(f"Confidence: {result['confidence']}")
    logger.info(f"Fusion Score: {result['fusion_score']:.6f}")
    logger.info(f"Thresholds: T_low={result['thresholds']['T_low']:.6f}, "
               f"T_high={result['thresholds']['T_high']:.6f}")
    logger.info("Per-model calibrated scores:")
    for model_id, score in result['calibrated_scores'].items():
        logger.info(f"  {model_id}: {score:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Test four-model ensemble fusion system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with mock data (no API calls)
  python test_four_model_fusion.py --mock
  
  # Test with real API calls using image directory
  python test_four_model_fusion.py --participant-name "John Doe" --num-frames 27 \\
      --image-dir /path/to/frames
  
  # Test with a video file (extracts and processes frames automatically)
  python test_four_model_fusion.py --participant-name "Noyn" --video /path/to/video.mp4
  
  # Test with video and custom max frames
  python test_four_model_fusion.py --participant-name "Test" --video video.mp4 --num-frames 16
  
  # Enable debug logging
  python test_four_model_fusion.py --mock --debug
        """
    )
    
    parser.add_argument('--mock', action='store_true',
                       help='Use mock data instead of real API calls')
    parser.add_argument('--participant-name', type=str, default='test_participant',
                       help='Participant name for logging')
    parser.add_argument('--num-frames', type=int, default=27,
                       help='Number of frames to process (or max frames from video, clamped to 8-32)')
    parser.add_argument('--image-dir', type=str,
                       help='Directory containing frame images (for real API test)')
    parser.add_argument('--video', type=str,
                       help='Video file to extract frames from (alternative to --image-dir)')
    parser.add_argument('--video-method', type=str, default='yolo',
                       choices=['yolo', 'yolo_haar'],
                       help='Face detection method for video preprocessing (default: yolo)')
    parser.add_argument('--calibrators', type=str,
                       help='Path to calibrators pickle file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.mock:
        asyncio.run(test_with_mock_data())
    else:
        # Validate that either video or image-dir is provided
        if not args.video and not args.image_dir:
            logger.error("Error: Either --video or --image-dir must be provided for real API testing")
            parser.print_help()
            return
        
        # Load image bytes
        image_paths = None
        image_bytes_list = None
        
        if args.video:
            # Extract frames from video
            video_path = args.video
            if not Path(video_path).exists():
                logger.error(f"Video file not found: {video_path}")
                return
            
            logger.info(f"Processing video: {video_path}")
            image_bytes_list = extract_frames_from_video(
                video_path=video_path,
                max_frames=args.num_frames,
                pre_method=args.video_method
            )
            
            if not image_bytes_list:
                logger.error("Failed to extract frames from video")
                return
                
        elif args.image_dir:
            # Load images from directory
            image_dir = Path(args.image_dir)
            if image_dir.exists():
                image_paths = sorted([str(p) for p in image_dir.glob('*.jpg')])
                logger.info(f"Found {len(image_paths)} images in {image_dir}")
            else:
                logger.error(f"Image directory not found: {image_dir}")
                return
        
        asyncio.run(test_with_real_api(
            args.participant_name, 
            args.num_frames, 
            image_paths=image_paths,
            image_bytes_list=image_bytes_list
        ))


if __name__ == '__main__':
    main()
