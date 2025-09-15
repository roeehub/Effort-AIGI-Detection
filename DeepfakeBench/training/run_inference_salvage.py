import argparse
import logging
import os
import yaml
from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import CLIPModel

# Assuming your project structure
from detectors.effort_detector import ArcMarginProduct
from data_process.process_video import Preprocess_videos

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class SalvageInferenceModel(nn.Module):
    """
    A clean, standard model architecture for inference.
    NO SVD modifications.
    """

    def __init__(self, use_arcface_head: bool, clip_model_path: str):
        super().__init__()
        logger.info(f"Initializing standard CLIP Vision Transformer from: {clip_model_path}")
        try:
            # Use local_files_only to ensure it doesn't try to download
            clip_model = CLIPModel.from_pretrained(clip_model_path, local_files_only=True)
        except Exception as e:
            logger.error(f"Failed to load CLIP model from '{clip_model_path}'. Ensure the path is correct.")
            raise e

        self.backbone = clip_model.vision_model

        self.use_arcface_head = use_arcface_head
        if self.use_arcface_head:
            # Note: s and m values don't matter for inference logic, only for the weights.
            self.head = ArcMarginProduct(in_features=1024, out_features=2, s=30.0, m=0.35)
        else:
            self.head = nn.Linear(1024, 2)

    def forward(self, image):
        # The forward pass is simpler now.
        # Handle video tensor shape
        is_video = image.dim() == 5
        if is_video:
            B, T, C, H, W = image.shape
            image = image.view(B * T, C, H, W)

        features = self.backbone(image)['pooler_output']

        # During inference, ArcFace head is called with no label
        logits = self.head(features)

        prob = torch.softmax(logits, dim=1)[:, 1]
        return prob


def load_salvaged_model(weights_path: str, use_arcface: bool, clip_model_path: str) -> nn.Module:
    """
    Loads the trained head and bias weights from a checkpoint into a clean model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Create the clean, standard-architecture model.
    # Its weights are from the base CLIP model + random head.
    model = SalvageInferenceModel(use_arcface, clip_model_path).to(device)
    clean_state_dict = model.state_dict()

    # 2. Load your trained (but problematic) checkpoint.
    logger.info(f"Loading salvageable weights from checkpoint: {weights_path}")
    salvage_state_dict = torch.load(weights_path, map_location='cpu')
    if list(salvage_state_dict.keys())[0].startswith('module.'):
        salvage_state_dict = OrderedDict((k[7:], v) for k, v in salvage_state_dict.items())

    # 3. Transplant the weights that matter.
    logger.info("Performing weight transplant...")
    transplanted_count = 0
    for name, param in salvage_state_dict.items():
        # We only care about the head and the biases.
        # The SVD fine-tuning was effectively zero, so we use the base CLIP weights
        # for everything else, which are already in our clean model.
        if name.startswith('head.') or name.endswith('.bias'):
            if name in clean_state_dict and clean_state_dict[name].shape == param.shape:
                clean_state_dict[name].copy_(param)
                transplanted_count += 1
                # logger.info(f"  - Transplanted: {name}")

    logger.info(f"Successfully transplanted {transplanted_count} parameter tensors.")

    # 4. Load the modified state dict into our clean model.
    model.load_state_dict(clean_state_dict)
    model.eval()
    return model


def main(args):
    # --- Load Model ---
    # We don't need the full config, just the critical paths and flags.
    model = load_salvaged_model(
        weights_path=args.weights,
        use_arcface=args.use_arcface_head,
        clip_model_path=args.clip_model_path
    )
    device = next(model.parameters()).device

    # --- Preprocess Video ---
    # This part remains the same.
    preprocess = Preprocess_videos(detector='yolo', face_aligner='fan', image_size=224, device=device)
    frames_t = preprocess(args.video)  # Returns a tensor [1, T, C, H, W]
    if frames_t is None:
        logger.error("Video processing failed. No faces detected or an error occurred.")
        return
    frames_t = frames_t.to(device)

    # --- Run Inference ---
    logger.info(f"Running inference on {frames_t.shape[1]} frames...")
    with torch.no_grad():
        # The model's forward pass now returns probabilities directly
        # for each frame in the batch.
        probs = model(frames_t).cpu().numpy()

    # --- Display Results ---
    print("\n--- Inference Results ---")
    for i, prob in enumerate(probs):
        prediction = "FAKE" if prob > 0.5 else "REAL"
        print(f"Frame {i + 1:02d}: Fake Probability = {prob:.8f} -> {prediction}")

    video_prob = probs.mean()
    video_prediction = "FAKE" if video_prob > 0.5 else "REAL"
    print("\n--- Overall Video ---")
    print(f"Average Fake Probability = {video_prob:.8f} -> {video_prediction}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Salvage and run inference on an EffortDetector checkpoint.")
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--weights', type=str, required=True, help='Path to the .pth checkpoint file to salvage.')
    parser.add_argument('--clip-model-path', type=str, required=True,
                        help='Path to the LOCAL base CLIP model directory.')
    parser.add_argument('--use-arcface-head', action='store_true',
                        help='CRITICAL: Specify this flag if the checkpoint was trained with an ArcFace head.')

    parsed_args = parser.parse_args()
    main(parsed_args)
