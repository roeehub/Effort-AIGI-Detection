import os
import argparse
import logging
import yaml
from pathlib import Path
from collections import OrderedDict

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


# =========================================================================
# ======================== NEW DEBUGGING FUNCTION =========================
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
    matching_keys = sorted(list(ckpt_keys & model_keys))

    logger.info(f"Checkpoint Keys: {len(ckpt_keys)} | Model Keys: {len(model_keys)}")
    logger.info("-" * 80)

    # --- Report Unexpected Keys ---
    if unexpected_keys:
        logger.warning(f"[WARNING] {len(unexpected_keys)} UNEXPECTED keys found in checkpoint (will be ignored):")
        for key in unexpected_keys:
            logger.warning(f"  - {key} (Shape: {state_dict[key].shape})")
    else:
        logger.info("[INFO] No unexpected keys found in checkpoint.")

    logger.info("-" * 80)

    # --- Report Missing Keys ---
    if missing_keys:
        logger.error(f"[ERROR] {len(missing_keys)} MISSING keys in checkpoint (model layers will not be loaded):")
        for key in missing_keys:
            logger.error(f"  - {key} (Expected Shape: {model.state_dict()[key].shape})")
    else:
        logger.info("[INFO] All model keys are present in the checkpoint.")

    logger.info("-" * 80)

    # --- Report Shape Mismatches ---
    mismatched_shape_keys = []
    for key in matching_keys:
        ckpt_shape = state_dict[key].shape
        model_shape = model.state_dict()[key].shape
        if ckpt_shape != model_shape:
            mismatched_shape_keys.append(
                f"  - {key}: Ckpt Shape={ckpt_shape} vs Model Shape={model_shape}"
            )

    if mismatched_shape_keys:
        logger.error(f"[ERROR] {len(mismatched_shape_keys)} keys have MISMATCHED SHAPES:")
        for mismatch in mismatched_shape_keys:
            logger.error(mismatch)
    else:
        logger.info("[INFO] All matching keys have correct shapes.")

    logger.info("=" * 80)


# =========================================================================
# ====================== NEW KEY REMAPPING FUNCTION =======================
# =========================================================================
def remap_checkpoint_keys(state_dict: dict, use_arcface_head: bool) -> OrderedDict:
    """
    Dynamically remaps checkpoint keys to handle architectural differences between
    how old Linear heads ('head.0.weight') were saved and how ArcFace heads
    ('head.weight') are structured.
    """
    new_state_dict = OrderedDict()
    remapped = False

    for key, value in state_dict.items():
        new_key = key
        # SCENARIO: Loading an old Linear checkpoint (key 'head.0.weight')
        # into a NEW ARCFACE model (which expects 'head.weight').
        if use_arcface_head and key.startswith('head.0.'):
            new_key = key.replace('head.0.', 'head.')
            remapped = True

        # SCENARIO: Loading an ArcFace checkpoint (key 'head.weight')
        # into a NEW LINEAR model (which expects 'head.0.weight').
        elif not use_arcface_head and key.startswith('head.weight'):
            # This handles the reverse case for future compatibility
            new_key = 'head.0.' + key[len('head.'):]
            remapped = True

        new_state_dict[new_key] = value

    if remapped:
        logger.info("✅ Checkpoint keys were successfully remapped to match the current model architecture.")

    return new_state_dict


# --- GCS Asset Downloading Utility ---
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


# --- MODIFIED Model Loading Function ---
def load_model(config: dict, weights_path: str) -> nn.Module:
    """
    Instantiates the EffortDetector, debugs keys, remaps if necessary, and loads weights.
    """
    logger.info("Initializing model with the specified configuration...")
    model = DETECTOR[config["model_name"]](config).to(device)

    logger.info(f"Loading weights from: {weights_path}")
    ckpt = torch.load(weights_path, map_location=device)

    # Standardize checkpoint loading (handles raw state_dict and full checkpoints)
    state = ckpt.get("state_dict", ckpt)

    # Clean up 'module.' prefix if it exists from DDP training
    if list(state.keys())[0].startswith('module.'):
        logger.info("Detected a DDP-trained model. Removing 'module.' prefix from keys.")
        state = OrderedDict((k[7:], v) for k, v in state.items())

    # --- DEBUGGING AND REMAPPING LOGIC ---
    logger.info("--- Analyzing RAW checkpoint keys BEFORE remapping ---")
    debug_checkpoint_and_model_keys(state, model)

    remapped_state = remap_checkpoint_keys(state, config['use_arcface_head'])

    logger.info("--- Analyzing REMAPPED checkpoint keys AFTER remapping ---")
    debug_checkpoint_and_model_keys(remapped_state, model)

    # Load the remapped state dict. Use strict=True for a final sanity check.
    try:
        model.load_state_dict(remapped_state, strict=True)
        logger.info("✅ Successfully loaded remapped state_dict with strict=True.")
    except RuntimeError as e:
        logger.error(f"❌ FAILED to load remapped state_dict with strict=True. Error: {e}")
        logger.info("Attempting to load with strict=False to proceed with inference...")
        model.load_state_dict(remapped_state, strict=False)

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
    parsed_args = parser.parse_args()
    main(parsed_args)
