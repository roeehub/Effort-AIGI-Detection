import argparse
import logging
import yaml
from collections import OrderedDict
import torch
import torch.nn as nn

# Assuming effort_detector.py is in a 'detectors' subdirectory
from detectors.effort_detector import EffortDetector, SVDResidualLinear

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def report_stats(tensor: torch.Tensor, name: str):
    """Prints key statistics for a given tensor."""
    if tensor is None:
        print(f"  - {name}: None")
        return
    # Use scientific notation for small numbers, fixed-point for larger ones
    format_str = ".4e" if tensor.abs().max().item() < 1e-3 else ".6f"
    print(f"  - {name}:")
    print(f"    - Shape: {list(tensor.shape)}")
    print(f"    - Mean:  {tensor.mean().item():{format_str}}")
    print(f"    - Std:   {tensor.std().item():{format_str}}")
    print(f"    - Max Abs: {tensor.abs().max().item():{format_str}}")


def load_model_for_inspection(config: dict, weights_path: str, use_arcface: bool):
    """Loads a checkpoint into a correctly configured EffortDetector."""
    config['use_arcface_head'] = use_arcface
    model = EffortDetector(config)

    logger.info(f"Loading checkpoint from: {weights_path}")
    state_dict = torch.load(weights_path, map_location='cpu')

    # Handle DDP 'module.' prefix
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def main(args):
    # --- Load Base Config ---
    try:
        with open(args.config, 'r') as f:
            base_config = yaml.safe_load(f)
        with open('./config/train_config.yaml', 'r') as f:
            base_config.update(yaml.safe_load(f))
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}.")
        return

    base_config['gcs_assets']['clip_backbone']['local_path'] = args.clip_model_path

    # --- Load Models ---
    logger.info("\n" + "=" * 80 + "\nLOADING GOLDEN (LINEAR HEAD) MODEL\n" + "=" * 80)
    golden_model = load_model_for_inspection(base_config.copy(), args.golden_linear_ckpt, use_arcface=False)

    logger.info("\n" + "=" * 80 + "\nLOADING SUSPECT (ARCFACE HEAD) MODEL\n" + "=" * 80)
    suspect_model = load_model_for_inspection(base_config.copy(), args.new_arcface_ckpt, use_arcface=True)

    # --- 1. Compare Heads ---
    print("\n\n" + "#" * 80)
    print("### 1. HEAD WEIGHTS COMPARISON")
    print("#" * 80 + "\n")

    print("--- [GOLDEN MODEL] Head (Linear) ---")
    report_stats(golden_model.head.weight.data, "head.weight")

    print("\n--- [SUSPECT MODEL] Head (ArcMarginProduct) ---")
    report_stats(suspect_model.head.weight.data, "head.weight")

    # --- 2. Compare Backbone Layers ---
    print("\n\n" + "#" * 80)
    print("### 2. BACKBONE SVD LAYER COMPARISON")
    print("#" * 80 + "\n")

    golden_modules = dict(golden_model.backbone.named_modules())
    suspect_modules = dict(suspect_model.backbone.named_modules())

    # Find all SVDResidualLinear layers to compare
    svd_layer_names = [name for name, module in golden_modules.items() if isinstance(module, SVDResidualLinear)]

    for i, name in enumerate(svd_layer_names[:3]):  # Limit to first 3 SVD layers for brevity
        print(f"\n--- Comparing Layer {i + 1}: '{name}' ---\n")

        g_module = golden_modules[name]
        s_module = suspect_modules[name]

        # --- Sub-Comparison A: The Frozen Base Weights ---
        print("[A] FROZEN 'weight_main' COMPARISON (Should be identical)")
        are_identical = torch.allclose(g_module.weight_main.data, s_module.weight_main.data)
        print(f"  - Are 'weight_main' tensors identical? -> {are_identical}")
        if not are_identical:
            logger.error("CRITICAL ERROR: 'weight_main' differs. Base models may be different.")
        print("-" * 20)

        # --- Sub-Comparison B: The Trained Residual Singular Values ---
        print("[B] TRAINED 'S_residual' COMPARISON (CRITICAL)")
        print("  --- GOLDEN MODEL ---")
        report_stats(g_module.S_residual.data, "S_residual")
        print("  --- SUSPECT MODEL ---")
        report_stats(s_module.S_residual.data, "S_residual")
        print("-" * 20)

        # --- Sub-Comparison C: The Computed Delta Magnitude ---
        print("[C] COMPUTED DELTA NORM COMPARISON (Total Learned Change)")
        with torch.no_grad():
            # Golden Delta
            g_delta = g_module.U_residual @ torch.diag(g_module.S_residual) @ g_module.V_residual
            g_norm = torch.norm(g_delta, p='fro').item()

            # Suspect Delta
            s_delta = s_module.U_residual @ torch.diag(s_module.S_residual) @ s_module.V_residual
            s_norm = torch.norm(s_delta, p='fro').item()

            print(f"  - Golden  Model Delta Norm: {g_norm:.6f}")
            print(f"  - Suspect Model Delta Norm: {s_norm:.6f}")

    logger.info("\nComparison finished. Review the 'S_residual' and 'Delta Norm' values.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deeply compare two EffortDetector checkpoints.")
    parser.add_argument('--new-arcface-ckpt', type=str, required=True,
                        help='Path to the NEW, FAILING .pth file with ArcFace head.')
    parser.add_argument('--golden-linear-ckpt', type=str, required=True,
                        help='Path to the GOLDEN, WORKING .pth file with Linear head.')
    parser.add_argument('--clip-model-path', type=str, required=True,
                        help='Path to the LOCAL base CLIP model directory.')
    parser.add_argument('--config', type=str, default='./config/detector/effort.yaml',
                        help='Path to the detector config.')
    parsed_args = parser.parse_args()
    main(parsed_args)
