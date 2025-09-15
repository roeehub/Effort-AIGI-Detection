import os
import argparse
import logging
import time
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from google.cloud import storage
from google.api_core import exceptions

# Import necessary components from your project files
import video_preprocessor
from detectors import DETECTOR, EffortDetector  # Make sure this import works based on your project structure

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- GCS Asset Downloading Utility (Adapted from app3.py) ---
def download_gcs_asset(bucket: storage.Bucket, gcs_path: str, local_path: str) -> bool:
    """Downloads a single file from GCS."""
    logger.info(f"Downloading gs://{bucket.name}/{gcs_path} to {local_path}...")
    blob = bucket.blob(gcs_path)
    if not blob.exists():
        logger.error(f"File not found at gs://{bucket.name}/{gcs_path}")
        return False

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        blob.download_to_filename(local_path)
        return True
    except Exception as e:
        logger.error(f"Failed to download {blob.name}: {e}")
        return False


# --- Model Loading (The Correct Way) ---
def load_model(config: dict, weights_path: str) -> nn.Module:
    """
    Instantiates the EffortDetector with the correct configuration and loads weights.

    Args:
        config (dict): The full configuration dictionary, including the correct
                       setting for 'use_arcface_head'.
        weights_path (str): Path to the .pth model weights file.

    Returns:
        nn.Module: The loaded and evaluation-ready model.
    """
    logger.info("Initializing model with the specified configuration...")
    # The 'config' dictionary now correctly defines the model architecture
    model = DETECTOR[config["model_name"]](config).to(device)

    logger.info(f"Loading weights from: {weights_path}")
    ckpt = torch.load(weights_path, map_location=device)

    # Standardize checkpoint loading
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    # Using strict=True is better for debugging. It will fail loudly if
    # the architecture and checkpoint keys do not match perfectly.
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        logger.error(f"Failed to load state_dict with strict=True. Key mismatch error: {e}")
        logger.info("Attempting to load with strict=False as a fallback...")
        model.load_state_dict(state, strict=False)

    model.eval()
    logger.info("Model loaded successfully and set to evaluation mode.")
    return model


def main(args):
    """Main inference script."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, running on CPU. This will be slow.")

    # 1. Load Base Configurations
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        with open('./config/train_config.yaml', 'r') as f:
            config.update(yaml.safe_load(f))
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}. Please check paths.")
        return

    # 2. **THE FIX**: Override config with command-line arguments
    # This ensures the model is instantiated with the same architecture it was trained with.
    config['use_arcface_head'] = args.use_arcface_head
    if args.use_arcface_head:
        logger.info("✅ Configuration overridden: Model will be initialized with an ArcFace head.")
    else:
        logger.info("✅ Configuration confirmed: Model will be initialized with a standard Linear head.")

    weights_path = args.weights

    # 3. Handle GCS Checkpoint if provided
    if args.checkpoint_gcs_path:
        logger.info(f"GCS checkpoint path provided: {args.checkpoint_gcs_path}")
        local_weights_dir = Path("./weights/downloaded/")
        local_weights_dir.mkdir(parents=True, exist_ok=True)

        gcs_path = args.checkpoint_gcs_path.replace("gs://", "")
        bucket_name, blob_name = gcs_path.split('/', 1)

        weights_path = local_weights_dir / Path(blob_name).name

        if weights_path.exists():
            logger.info(f"Checkpoint already exists locally at {weights_path}. Skipping download.")
        else:
            try:
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                if not download_gcs_asset(bucket, blob_name, str(weights_path)):
                    logger.error("Failed to download checkpoint from GCS.")
                    return
            except Exception as e:
                logger.error(f"Error connecting to GCS: {e}")
                logger.error("Please ensure you have authenticated with 'gcloud auth application-default login'")
                return

    if not Path(weights_path).exists():
        logger.error(f"Weights file not found at: {weights_path}")
        return

    # 4. Initialize Preprocessor Models
    try:
        video_preprocessor.initialize_yolo_model()
    except Exception as e:
        logger.error(f"Failed to initialize YOLO model for face detection: {e}")
        return

    # 5. Load the Model correctly
    model = load_model(config, str(weights_path))

    # 6. Preprocess Video
    logger.info(f"Processing video file: {args.video}")
    video_tensor = video_preprocessor.preprocess_video_for_effort_model(
        video_path=args.video,
        pre_method="yolo"  # This was used for training
    )

    if video_tensor is None or video_tensor.shape[1] == 0:
        logger.error("Failed to process video. No faces might have been detected.")
        return

    num_frames = video_tensor.shape[1]
    logger.info(f"Successfully extracted and processed {num_frames} frames.")

    # 7. Run Inference
    logger.info("Running inference on the processed frames...")
    with torch.inference_mode():
        predictions = model({'image': video_tensor.to(device)}, inference=True)
        frame_probabilities = predictions["prob"].cpu().numpy().tolist()

    # 8. Display Results
    print("\n--- Inference Results ---")
    if len(frame_probabilities) > 0:
        for i, prob in enumerate(frame_probabilities):
            decision = "FAKE" if prob >= 0.5 else "REAL"
            print(f"Frame {i + 1:02d}: Fake Probability = {prob:.8f} -> {decision}")

        avg_prob = sum(frame_probabilities) / len(frame_probabilities)
        avg_decision = "FAKE" if avg_prob >= 0.5 else "REAL"
        print(f"\nAverage Fake Probability across {len(frame_probabilities)} frames: {avg_prob:.8f} -> {avg_decision}")
    else:
        print("No probabilities were generated.")
    print("-----------------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run EffortDetector inference on a single video.")

    parser.add_argument('--video', type=str, required=True, help='Path to the local video file.')

    # Provide either local weights or a GCS path
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--weights', type=str, help='Path to the local .pth model weights file.')
    group.add_argument('--checkpoint-gcs-path', type=str,
                       help='GCS path to the model weights (e.g., gs://bucket/model.pth).')

    parser.add_argument('--config', type=str, default='./config/detector/effort.yaml',
                        help='Path to the base detector config YAML file.')

    # This flag is the critical fix
    parser.add_argument(
        '--use-arcface-head',
        action='store_true',
        help='Specify this flag if the model was trained with the ArcFace head.'
    )

    parsed_args = parser.parse_args()
    main(parsed_args)
