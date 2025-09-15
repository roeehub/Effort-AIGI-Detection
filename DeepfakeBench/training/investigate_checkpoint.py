import os
import argparse
import logging
import yaml
from collections import OrderedDict
import torch

# Import the necessary classes from your project.
# Adjust the path if your file structure is different.
from detectors.effort_detector import EffortDetector, ArcMarginProduct, SVDResidualLinear

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def inspect_model(model: EffortDetector, model_name: str):
    """Prints critical information about the model's state."""
    print("\n" + "=" * 80)
    print(f"🔬 INSPECTING MODEL: '{model_name}'")
    print("=" * 80)

    # 1. Print the head architecture
    print("\n[-- ARCHITECTURE --]")
    print(f"Model Head Type: {type(model.head)}")
    # print(model.head) # Uncomment for full head details

    # 2. Print stats for the head's weight parameter
    print("\n[-- HEAD WEIGHTS --]")
    if hasattr(model.head, 'weight') and model.head.weight is not None:
        head_weights = model.head.weight.data
        print(f"  Shape: {head_weights.shape}")
        print(f"  Mean:  {head_weights.mean():.8f}")
        print(f"  Std:   {head_weights.std():.8f}")
        print(f"  Max:   {head_weights.abs().max():.8f}")
    else:
        print("  Head has no 'weight' parameter.")

    # 3. Print stats for a representative SVD residual parameter
    print("\n[-- SVD RESIDUAL WEIGHTS --]")
    try:
        # Let's inspect the query projection in the first attention block
        svd_layer_s = model.backbone.encoder.layers[0].self_attn.q_proj.S_residual
        if svd_layer_s is not None:
            svd_weights = svd_layer_s.data
            print("  Inspecting '...q_proj.S_residual':")
            print(f"  Shape: {svd_weights.shape}")
            print(f"  Mean:  {svd_weights.mean():.8f}")
            print(f"  Std:   {svd_weights.std():.8f}")
            print(f"  Max:   {svd_weights.abs().max():.8f}")
        else:
            print("  S_residual is None for the inspected layer.")
    except AttributeError as e:
        print(f"  Could not access SVD residual layer for inspection: {e}")

    print("=" * 80 + "\n")


def main(args):
    # --- Load Base Config ---
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        with open('./config/train_config.yaml', 'r') as f:
            config.update(yaml.safe_load(f))
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}.")
        return

    # Update the config to use the provided local path for the base model
    # This is critical for EffortDetector's initialization
    config['gcs_assets']['clip_backbone']['local_path'] = args.clip_model_path
    logger.info(f"Using local CLIP path: {args.clip_model_path}")

    # ===============================================================
    # INVESTIGATION 1: The new, FAILING ArcFace checkpoint
    # ===============================================================
    logger.info("--- Starting Investigation 1: FAILING ArcFace Checkpoint ---")

    # A. Create a model CONFIGURED FOR ARC-FACE
    config['use_arcface_head'] = True
    model_arc = EffortDetector(config)
    inspect_model(model_arc, "Randomly Initialized (ArcFace Config)")

    # B. Load the FAILING checkpoint
    logger.info(f"Loading failing checkpoint from: {args.new_arcface_ckpt}")
    state_dict_arc = torch.load(args.new_arcface_ckpt, map_location='cpu')
    if list(state_dict_arc.keys())[0].startswith('module.'):
        state_dict_arc = OrderedDict((k[7:], v) for k, v in state_dict_arc.items())

    # Load the weights and inspect the result
    model_arc.load_state_dict(state_dict_arc, strict=False)
    inspect_model(model_arc, "After Loading FAILING ArcFace Checkpoint")

    # ===============================================================
    # INVESTIGATION 2: The old, WORKING Linear checkpoint
    # ===============================================================
    logger.info("--- Starting Investigation 2: WORKING Linear Checkpoint ---")

    # A. Create a model CONFIGURED FOR A LINEAR HEAD
    config['use_arcface_head'] = False
    model_linear = EffortDetector(config)
    inspect_model(model_linear, "Randomly Initialized (Linear Config)")

    # B. Load the WORKING checkpoint
    logger.info(f"Loading working checkpoint from: {args.old_linear_ckpt}")
    state_dict_linear = torch.load(args.old_linear_ckpt, map_location='cpu')
    if list(state_dict_linear.keys())[0].startswith('module.'):
        state_dict_linear = OrderedDict((k[7:], v) for k, v in state_dict_linear.items())

    # Load the weights and inspect the result
    model_linear.load_state_dict(state_dict_linear, strict=False)
    inspect_model(model_linear, "After Loading WORKING Linear Checkpoint")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Investigate model checkpoints.")
    parser.add_argument('--new-arcface-ckpt', type=str, required=True,
                        help='Path to the NEW, FAILING .pth file with ArcFace head.')
    parser.add_argument('--old-linear-ckpt', type=str, required=True,
                        help='Path to the OLD, WORKING .pth file with Linear head.')
    parser.add_argument('--clip-model-path', type=str, required=True,
                        help='Path to the LOCAL base CLIP model directory.')
    parser.add_argument('--config', type=str, default='./config/detector/effort.yaml',
                        help='Path to the detector config.')
    parsed_args = parser.parse_args()
    main(parsed_args)
