import argparse
import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import CLIPModel

from video_preprocessor import preprocess_video_for_effort_model
from detectors.effort_detector import ArcMarginProduct

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class TransplantInferenceModel(nn.Module):
    """
    A clean, standard model architecture. It will be populated by transplanting
    weights from two different checkpoints.
    """

    def __init__(self, use_arcface_head: bool, clip_model_path: str):
        super().__init__()
        logger.info(f"Initializing standard CLIP Vision Transformer from: {clip_model_path}")
        clip_model = CLIPModel.from_pretrained(clip_model_path, local_files_only=True)
        self.backbone = clip_model.vision_model

        self.use_arcface_head = use_arcface_head
        if self.use_arcface_head:
            self.head = ArcMarginProduct(in_features=1024, out_features=2, s=30.0, m=0.35)
        else:
            # Placeholder for the golden checkpoint's head
            self.head = nn.Linear(1024, 2)

    def forward(self, image):
        is_video = image.dim() == 5
        if is_video:
            B, T, C, H, W = image.shape
            image = image.view(B * T, C, H, W)

        features = self.backbone(image)['pooler_output']
        logits = self.head(features)
        prob = torch.softmax(logits, dim=1)[:, 1]
        return prob


def load_transplanted_model(
        your_arcface_ckpt_path: str,
        golden_linear_ckpt_path: str,
        clip_model_path: str
) -> nn.Module:
    """
    Performs a model transplant:
    1. Loads the backbone from the golden checkpoint.
    2. Loads the trained ArcFace head from your checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Step 1: Create the final model shell ---
    # We configure it with an ArcFace head because that's what our final model will have.
    model = TransplantInferenceModel(use_arcface_head=True, clip_model_path=clip_model_path).to(device)

    # --- Step 2: Load the golden checkpoint to get the working backbone ---
    logger.info("--- STAGE 1: Loading backbone from GOLDEN checkpoint ---")
    logger.info(f"Path: {golden_linear_ckpt_path}")
    golden_state_dict = torch.load(golden_linear_ckpt_path, map_location='cpu')

    # The golden checkpoint head is Linear, ours is ArcFace. So there will be a size mismatch.
    # `strict=False` is essential here. It will load all matching keys (the entire backbone)
    # and ignore the non-matching head.
    model.load_state_dict(golden_state_dict, strict=False)
    logger.info("✅ Backbone transplant successful.")

    # --- Step 3: Load your checkpoint to get the trained ArcFace head ---
    logger.info("--- STAGE 2: Transplanting trained ArcFace head from YOUR checkpoint ---")
    logger.info(f"Path: {your_arcface_ckpt_path}")
    your_state_dict = torch.load(your_arcface_ckpt_path, map_location='cpu')
    if list(your_state_dict.keys())[0].startswith('module.'):
        your_state_dict = OrderedDict((k[7:], v) for k, v in your_state_dict.items())

    # Manually copy only the head weights from your checkpoint into the model
    with torch.no_grad():
        if 'head.weight' in your_state_dict:
            model.head.weight.copy_(your_state_dict['head.weight'])
            logger.info("  - Transplanted 'head.weight'")
        else:
            logger.warning("Could not find 'head.weight' in your checkpoint!")

    logger.info("✅ Head transplant successful. Model assembly complete.")

    model.eval()
    return model


def main(args):
    model = load_transplanted_model(
        your_arcface_ckpt_path=args.arcface_weights,
        golden_linear_ckpt_path=args.golden_weights,
        clip_model_path=args.clip_model_path
    )
    device = next(model.parameters()).device

    frames_t = preprocess_video_for_effort_model(
        video_path=args.video,
        pre_method=args.pre_method
    )

    if frames_t is None:
        return
    frames_t = frames_t.to(device)

    logger.info(f"Running inference on {frames_t.shape[1]} frames...")
    with torch.no_grad():
        probs = model(frames_t).cpu().numpy()

    print("\n--- Inference Results ---")
    for i, prob in enumerate(probs):
        prediction = "FAKE" if prob > 0.5 else "REAL"
        print(f"Frame {i + 1:02d}: Fake Probability = {prob:.8f} -> {prediction}")

    video_prob = probs.mean()
    video_prediction = "FAKE" if video_prob > 0.5 else "REAL"
    print("\n--- Overall Video ---")
    print(f"Average Fake Probability = {video_prob:.8f} -> {video_prediction}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transplant weights and run inference.")
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--arcface-weights', type=str, required=True,
                        help="Path to YOUR trained .pth with the ArcFace head.")
    parser.add_argument('--golden-weights', type=str, required=True,
                        help="Path to the author's golden .pth with the working backbone.")
    parser.add_argument('--clip-model-path', type=str, required=True)
    parser.add_argument('--pre-method', type=str, default='yolo', choices=['yolo', 'yolo_haar'])

    parsed_args = parser.parse_args()
    main(parsed_args)
