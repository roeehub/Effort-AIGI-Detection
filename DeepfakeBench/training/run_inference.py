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
from transformers import CLIPModel

# Import necessary components from your project files
import video_preprocessor
from detectors.effort_detector import EffortDetector, ArcMarginProduct  # Need both

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================
# ======================== UTILITY FUNCTIONS (Unchanged) ==================
# =========================================================================

def save_debug_frames(tensor_batch: torch.Tensor, output_dir: str):
    # This function remains unchanged.
    if tensor_batch is None: return
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=tensor_batch.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=tensor_batch.device).view(1, 3, 1, 1)
    frames = tensor_batch.squeeze(0)
    for i, frame_tensor in enumerate(frames):
        denormalized_frame = torch.clamp((frame_tensor.unsqueeze(0) * std + mean) * 255, 0, 255)
        image_np = denormalized_frame.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"frame_{i:02d}.png"), image_bgr)
    logger.info(f"✅ Successfully saved {frames.shape[0]} debug frames to '{output_dir}'.")


def download_gcs_asset(bucket: storage.Bucket, gcs_path: str, local_path: str) -> bool:
    # This function remains unchanged.
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
# ======================== MODELS & FUSION LOGIC ==========================
# =========================================================================

class InferenceModel(nn.Module):
    """A simple, fast wrapper for inference that uses a standard backbone."""

    # <<< MODIFIED SIGNATURE to accept a path
    def __init__(self, use_arcface_head=False, clip_model_path: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        # <<< MODIFIED LINE to use the provided path
        logger.info(f"Initializing standard CLIP Vision Transformer from: {clip_model_path}")
        clip_model = CLIPModel.from_pretrained(clip_model_path)
        self.backbone = clip_model.vision_model

        self.use_arcface_head = use_arcface_head
        if self.use_arcface_head:
            self.head = ArcMarginProduct(in_features=1024, out_features=2, s=30.0, m=0.35)
        else:
            self.head = nn.Linear(1024, 2)

    def forward(self, data_dict: dict, inference=True) -> dict:  # Match signature
        # This part is unchanged
        image = data_dict['image']
        is_video = image.dim() == 5
        if is_video:
            B, T, C, H, W = image.shape
            image = image.view(B * T, C, H, W)
        features = self.backbone(image)['pooler_output']
        raw_logits = self.head(features, label=None)
        prob = torch.softmax(raw_logits, dim=1)[:, 1]
        return {'prob': prob, 'raw_logits': raw_logits, 'feat': features}


def fuse_model_weights(original_model: EffortDetector) -> OrderedDict:
    """Takes a loaded EffortDetector and returns a state_dict with fused weights."""
    fused_state_dict = OrderedDict()
    original_state_dict = original_model.state_dict()
    logger.info("Starting weight fusion process...")

    # Step 1: Copy all parameters that DON'T need fusion (e.g., LayerNorms, embeddings, the head)
    # This is a safe way to get all the non-SVD parameters into the new dict.
    for key, value in original_state_dict.items():
        if 'S_residual' in key or 'U_residual' in key or 'V_residual' in key or \
                'weight_main' in key or '_fnorm' in key or '_r' in key:
            continue  # Skip all SVD components for now
        fused_state_dict[key] = value.clone().detach()

    # Step 2: Iterate through the model's modules to find SVDResidualLinear instances
    # and perform the fusion, creating the keys the InferenceModel expects.
    for name, module in original_model.named_modules():
        if isinstance(module, SVDResidualLinear):
            logger.info(f"  -> Fusing weights for: {name}")

            # Calculate the final, fused weight
            fused_weight = module.compute_current_weight()

            # Create the key that the standard nn.Linear layer expects
            weight_key = name + '.weight'
            fused_state_dict[weight_key] = fused_weight.clone().detach()

            # The original module might have had a bias, which we copied in Step 1.
            # We need to ensure it's present if the SVD layer has one too.
            if module.bias is not None:
                bias_key = name + '.bias'
                if bias_key not in fused_state_dict:
                    fused_state_dict[bias_key] = module.bias.clone().detach()

    logger.info("✅ Fusion calculation complete.")
    return fused_state_dict


def load_model(config: dict, weights_path: str, use_fused: bool, clip_model_path: str) -> nn.Module:
    """
    Main model loading function. Handles both original and fused model paths.
    """
    if not use_fused:
        # This path is unchanged
        logger.info("Loading original model with custom SVD layers (use_fused=False).")
        model = EffortDetector(config).to(device)
        state_dict = torch.load(weights_path, map_location=device)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
        model.load_state_dict(state_dict, strict=True)
    else:
        # This path is mostly the same, just passes the clip_model_path
        original_path = Path(weights_path)
        fused_path = original_path.with_name(f"fused_{original_path.name}")

        if not fused_path.exists():
            logger.warning(f"Fused checkpoint not found at '{fused_path}'. Creating one now...")
            original_model = EffortDetector(config)
            state_dict = torch.load(original_path, map_location=device)
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
            original_model.load_state_dict(state_dict, strict=True)
            fused_state_dict = fuse_model_weights(original_model)
            torch.save(fused_state_dict, fused_path)
            logger.info(f"✅ Saved new fused checkpoint to '{fused_path}'.")
            del original_model

        logger.info(f"Loading fused weights from '{fused_path}' into standard model.")
        # <<< MODIFIED LINE to pass the clip_model_path
        model = InferenceModel(config['use_arcface_head'], clip_model_path=clip_model_path).to(device)
        fused_state_dict = torch.load(fused_path, map_location=device)
        model.load_state_dict(fused_state_dict, strict=True)

    model.eval()
    logger.info("Model loaded successfully and set to evaluation mode.")
    return model


# =========================================================================
# ============================ MAIN EXECUTION =============================
# =========================================================================
def main(args):
    """
    Main execution function for the inference script.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, running on CPU. This will be slow.")

    # 1. Load Base Configurations from YAML files
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        with open('./config/train_config.yaml', 'r') as f:
            config.update(yaml.safe_load(f))
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}. Please check paths.")
        return

    # 2. Override config with command-line arguments for model architecture
    config['use_arcface_head'] = args.use_arcface_head
    logger.info(f"Model architecture configured with use_arcface_head = {args.use_arcface_head}")

    # 3. Handle downloading the BASE CLIP model from GCS (if necessary)
    local_clip_path = args.clip_model_path
    if args.clip_model_path.startswith("gs://"):
        logger.info("Base CLIP model path is a GCS URI. Checking for a local copy...")
        local_clip_dir = Path("./base_models/clip-vit-large-patch14/")

        if (local_clip_dir / "config.json").exists() and not args.force_download:
            logger.info(f"Found existing local copy of base model at '{local_clip_dir}'.")
            local_clip_path = str(local_clip_dir)
        else:
            logger.info(f"Downloading base CLIP model from GCS to '{local_clip_dir}'...")
            try:
                client = storage.Client()
                gcs_path = args.clip_model_path.replace("gs://", "")
                bucket_name, prefix = gcs_path.split('/', 1)
                if not prefix.endswith('/'): prefix += '/'

                bucket = client.bucket(bucket_name)
                blobs = list(client.list_blobs(bucket, prefix=prefix))

                if not blobs:
                    logger.error(f"No files found at GCS path: {args.clip_model_path}")
                    return

                local_clip_dir.mkdir(parents=True, exist_ok=True)
                for blob in blobs:
                    if blob.name.endswith('/'): continue

                    file_name = blob.name[len(prefix):]
                    destination_file_name = local_clip_dir / file_name

                    logger.info(f"  - Downloading {blob.name} to {destination_file_name}")
                    destination_file_name.parent.mkdir(parents=True, exist_ok=True)
                    blob.download_to_filename(str(destination_file_name))

                local_clip_path = str(local_clip_dir)
                logger.info("Base CLIP model download complete.")
            except Exception as e:
                logger.error(f"Failed to download base CLIP model from GCS: {e}")
                return

    # =====================================================================
    # ========================= THE CRITICAL FIX (v2) =====================
    # Update the config dictionary with the correct, verified local path.
    # This path matches the structure read by the EffortDetector class.
    config['gcs_assets']['clip_backbone']['local_path'] = local_clip_path
    logger.info(f"Updated config to use local CLIP model path: '{local_clip_path}'")
    # =====================================================================

    # 4. Handle downloading the trained model WEIGHTS from GCS (if necessary)
    weights_path = args.weights
    if args.checkpoint_gcs_path:
        local_weights_dir = Path("./weights/downloaded/")
        gcs_path = args.checkpoint_gcs_path.replace("gs://", "")
        bucket_name, blob_name = gcs_path.split('/', 1)
        weights_path = local_weights_dir / Path(blob_name).name

        if weights_path.exists() and not args.force_download:
            logger.info(f"Trained checkpoint already exists locally at {weights_path}.")
        else:
            try:
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                if not download_gcs_asset(bucket, blob_name, str(weights_path)):
                    logger.error("Failed to download trained checkpoint from GCS.")
                    return
            except Exception as e:
                logger.error(f"Error connecting to GCS: {e}")
                return

    if not Path(weights_path).exists():
        logger.error(f"Weights file not found at: {weights_path}")
        return

    # 5. Initialize Preprocessor Models
    video_preprocessor.initialize_yolo_model()

    # 6. Load the Model (it will now receive the *updated* config)
    model = load_model(config, str(weights_path),
                       use_fused=args.use_fused_weights,
                       clip_model_path=local_clip_path)

    # 7. Preprocess Video
    logger.info(f"Processing video file: {args.video}")
    video_tensor = video_preprocessor.preprocess_video_for_effort_model(
        video_path=args.video, pre_method="yolo"
    )

    if args.debug_save_frames:
        save_debug_frames(video_tensor, f"debug_frames/{Path(args.video).stem}")

    if video_tensor is None or video_tensor.shape[1] == 0:
        logger.error("Failed to process video. No faces might have been detected.")
        return

    # 8. Run Inference
    logger.info(f"Running inference on {video_tensor.shape[1]} frames...")
    with torch.inference_mode():
        predictions = model({'image': video_tensor.to(device)}, inference=True)
        frame_probabilities = predictions["prob"].cpu().numpy().tolist()

    # 9. Display Results
    print("\n--- Inference Results ---")
    if frame_probabilities:
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
    group.add_argument('--checkpoint-gcs-path', type=str, help='GCS path to model weights.')

    parser.add_argument('--clip-model-path', type=str,
                        default='gs://base-checkpoints/effort-aigi/models--openai--clip-vit-large-patch14/',
                        help='Path to the base CLIP model. Can be a GCS URI or a local path.')

    parser.add_argument('--config', type=str, default='./config/detector/effort.yaml',
                        help='Path to the detector config.')

    parser.add_argument('--use-arcface-head', action='store_true', help='Crucial: Specify if model uses ArcFace head.')
    parser.add_argument('--use-fused-weights', action='store_true',
                        help='Use (or create) a fused ckpt for faster, correct inference.')
    parser.add_argument('--debug-save-frames', action='store_true', help='Save preprocessed frames for inspection.')
    parser.add_argument('--force-download', action='store_true',
                        help='Force re-download of assets from GCS even if they exist locally.')

    parsed_args = parser.parse_args()
    main(parsed_args)
