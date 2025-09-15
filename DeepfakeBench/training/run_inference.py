import os
import argparse
import logging
import yaml
from pathlib import Path
from collections import OrderedDict
import cv2

import torch
import torch.nn as nn
from google.cloud import storage

# Import necessary components from your project files
import video_preprocessor
from detectors import DETECTOR, EffortDetector  # Make sure this import works

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_debug_frames(tensor_batch: torch.Tensor, output_dir: str):
    """
    Denormalizes and saves a batch of preprocessed frames to disk for visual inspection.

    Args:
        tensor_batch: The input tensor for the model, shape [1, T, C, H, W].
        output_dir: The directory where PNG files will be saved.
    """
    if tensor_batch is None:
        logger.warning("Cannot save debug frames because input tensor is None.")
        return

    logger.info(f"--- Saving Debug Frames to '{output_dir}' ---")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # These are the standard CLIP normalization values.
    # The denormalization process is the mathematical inverse of normalization.
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=tensor_batch.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=tensor_batch.device).view(1, 3, 1, 1)

    # Remove the batch dimension of size 1
    frames = tensor_batch.squeeze(0)  # Shape becomes [T, C, H, W]

    num_saved = 0
    for i, frame_tensor in enumerate(frames):
        # Denormalize: (tensor * std) + mean
        denormalized_frame = frame_tensor.unsqueeze(0) * std + mean

        # Un-scale from [0, 1] to [0, 255] and clamp values
        denormalized_frame = torch.clamp(denormalized_frame * 255, 0, 255)

        # Permute from [C, H, W] to [H, W, C] for saving
        # Convert to a NumPy array with the correct data type for an image
        image_np = denormalized_frame.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()

        # OpenCV expects BGR format, but our tensor is in RGB. We must convert it.
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Save the image
        save_path = os.path.join(output_dir, f"frame_{i:02d}.png")
        cv2.imwrite(save_path, image_bgr)
        num_saved += 1

    logger.info(f"✅ Successfully saved {num_saved} debug frames.")


# =========================================================================
# ======================== DEBUGGING FUNCTION (Unchanged) =========================
# =========================================================================
def debug_checkpoint_and_model_keys(state_dict: dict, model: nn.Module):
    """
    Generates a detailed report comparing the keys and shapes in a loaded
    state_dict with the keys and shapes expected by a model instance.
    """
    logger.info("=" * 80)
    logger.info("=============== CHECKPOINT & MODEL KEY DEBUGGER ===============")

    ckpt_keys = set(state_dict.keys())
    model_keys = set(model.state_dict().keys())

    unexpected_keys = sorted(list(ckpt_keys - model_keys))
    missing_keys = sorted(list(model_keys - ckpt_keys))
    # ... (rest of function is the same, no need to copy it all)
    logger.info(f"Checkpoint Keys: {len(ckpt_keys)} | Model Keys: {len(model_keys)}")
    if not unexpected_keys and not missing_keys:
        logger.info("[INFO] All keys match perfectly between checkpoint and model.")
    else:
        if unexpected_keys:
            logger.warning(f"[WARNING] {len(unexpected_keys)} UNEXPECTED keys in checkpoint: {unexpected_keys}")
        if missing_keys:
            logger.error(f"[ERROR] {len(missing_keys)} MISSING keys in checkpoint: {missing_keys}")
    logger.info("=" * 80)


# --- GCS Asset Downloading Utility (Unchanged) ---
def download_gcs_asset(bucket: storage.Bucket, gcs_path: str, local_path: str) -> bool:
    # ... (function is the same)
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


# =========================================================================
# ===================== THE NEW, SMARTER LOAD_MODEL =====================
# =========================================================================
def load_model(config: dict, weights_path: str) -> nn.Module:
    """
    Instantiates the model, intelligently remaps keys ONLY if a known
    mismatch is detected, and then loads the weights.
    """
    logger.info("Initializing model with the specified configuration...")
    # This assumes DETECTOR and the config correctly build the model
    model = DETECTOR[config["model_name"]](config).to(device)

    logger.info(f"Loading weights from: {weights_path}")
    ckpt = torch.load(weights_path, map_location=device)

    # Handle both full checkpoints and raw state_dicts
    state = ckpt.get("state_dict", ckpt)

    # Handle DDP-trained models
    if list(state.keys())[0].startswith('module.'):
        logger.info("Detected a DDP-trained model. Removing 'module.' prefix from keys.")
        state = OrderedDict((k[7:], v) for k, v in state.items())

    # --- Intelligent, Safe Remapping Logic ---
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())

    # This is the ONLY scenario we want to fix: An old checkpoint with 'head.0.weight'
    # is being loaded into a model that now expects 'head.weight'.
    # This is currently unlikely given your code, but is good future-proofing.
    if 'head.0.weight' in ckpt_keys and 'head.weight' in model_keys:
        logger.warning("! Mismatch Detected: Checkpoint has 'head.0.*' keys while model expects 'head.*'.")
        logger.warning("  Attempting to remap keys automatically.")

        remapped_state = OrderedDict()
        for k, v in state.items():
            new_k = k.replace('head.0.', 'head.') if k.startswith('head.0.') else k
            remapped_state[new_k] = v
        state = remapped_state  # Overwrite state with the fixed version
        logger.info("✅ Remapping complete.")

    # --- Final Loading ---
    try:
        model.load_state_dict(state, strict=True)
        logger.info("✅ Successfully loaded state_dict with strict=True.")
    except RuntimeError as e:
        logger.error(f"❌ FAILED to load state_dict with strict=True. Error: {e}")
        logger.info("    This indicates a key mismatch that was not automatically fixed.")
        logger.info("    Attempting to load with strict=False to proceed with inference...")
        model.load_state_dict(state, strict=False)

    model.eval()
    logger.info("Model loaded successfully and set to evaluation mode.")
    return model


# =========================================================================
# =========== main() and argparse are now UNCHANGED from your file ========
# =========================================================================
def main(args):
    # ... (main function remains exactly the same)
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

    # 2. Override config with command-line arguments
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

    # 5. Load the Model correctly (using our new enhanced function)
    model = load_model(config, str(weights_path))

    # 6. Preprocess Video
    logger.info(f"Processing video file: {args.video}")
    video_tensor = video_preprocessor.preprocess_video_for_effort_model(
        video_path=args.video,
        pre_method="yolo"
    )

    # --- ADD THIS DEBUGGING BLOCK ---
    if args.debug_save_frames:
        video_filename = Path(args.video).stem
        debug_output_dir = os.path.join("debug_frames", video_filename)
        save_debug_frames(video_tensor, debug_output_dir)
        logger.info("Debug frames saved. Continuing with inference...")
    # --- END OF DEBUGGING BLOCK ---

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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--weights', type=str, help='Path to the local .pth model weights file.')
    group.add_argument('--checkpoint-gcs-path', type=str,
                       help='GCS path to the model weights (e.g., gs://bucket/model.pth).')
    parser.add_argument('--config', type=str, default='./config/detector/effort.yaml',
                        help='Path to the base detector config YAML file.')
    parser.add_argument(
        '--use-arcface-head',
        action='store_true',
        help='Specify this flag if the model was trained with the ArcFace head.'
    )
    parser.add_argument(
        '--debug-save-frames',
        action='store_true',
        help='If specified, save the preprocessed frames to a debug directory before inference.'
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
