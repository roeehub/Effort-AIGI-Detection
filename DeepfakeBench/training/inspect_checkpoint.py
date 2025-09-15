# --- START OF FILE inspect_checkpoint.py ---

import argparse
import yaml
import torch
from collections import OrderedDict

# We need the model definition to build an instance of it
from detectors.effort_detector import EffortDetector


def inspect(checkpoint_path: str, config: dict, clip_model_path: str, use_arcface: bool):
    """
    Inspects a checkpoint file and a corresponding model architecture, then compares them.
    """
    print(f"\n=======================================================================")
    print(f"  INSPECTING: {checkpoint_path}")
    print(f"  MODEL CONFIG: use_arcface_head = {use_arcface}")
    print(f"=======================================================================")

    # --- Part 1: Analyze the Checkpoint File ---
    print("\n--- [1] Analyzing keys in the checkpoint file... ---")
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        # Handle DataParallel prefix
        if list(state_dict.keys())[0].startswith('module.'):
            print("  -> Found and removed 'module.' prefix.")
            state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())

        checkpoint_keys = set(state_dict.keys())
        print(f"  -> Found {len(checkpoint_keys)} keys in the file.")
        if 'head.s' in checkpoint_keys:
            print("  ✅ CRITICAL FINDING: 'head.s' IS PRESENT in this checkpoint file.")
        else:
            print("  ❌ CRITICAL FINDING: 'head.s' IS MISSING from this checkpoint file.")

        print("\n  All keys in checkpoint file:")
        for key in sorted(list(checkpoint_keys)):
            print(f"    - {key}")

    except Exception as e:
        print(f"  -> ERROR: Could not load or parse the checkpoint file: {e}")
        return

    # --- Part 2: Analyze the Model Architecture ---
    print("\n--- [2] Analyzing keys expected by the current model code... ---")
    try:
        # Override config with the flag we passed
        config['use_arcface_head'] = use_arcface
        config['gcs_assets']['clip_backbone']['local_path'] = clip_model_path

        print(f"  -> Instantiating EffortDetector with use_arcface_head={use_arcface}...")
        model = EffortDetector(config)
        model_keys = set(model.state_dict().keys())
        print(f"  -> Model expects {len(model_keys)} keys.")

        if 'head.s' in model_keys:
            print("  ✅ CRITICAL FINDING: The current code's model architecture EXPECTS the 'head.s' key.")
        else:
            print("  ❌ CRITICAL FINDING: The current code's model architecture DOES NOT expect the 'head.s' key.")

        print("\n  All keys expected by model:")
        for key in sorted(list(model_keys)):
            print(f"    - {key}")

    except Exception as e:
        print(f"  -> ERROR: Could not instantiate the model. Check config/paths: {e}")
        return

    # --- Part 3: Compare and Conclude ---
    print("\n--- [3] Comparison and Final Diagnosis ---")
    missing_from_checkpoint = model_keys - checkpoint_keys
    extra_in_checkpoint = checkpoint_keys - model_keys

    if not missing_from_checkpoint and not extra_in_checkpoint:
        print("\n  ✅ DIAGNOSIS: PERFECT MATCH!")
        print("  The keys in the checkpoint file exactly match the keys expected by the model architecture.")
        print("  If inference is failing, the problem is not a key mismatch.")
    else:
        if missing_from_checkpoint:
            print(f"\n  ❌ DIAGNOSIS: Mismatch found! The model expects keys that are MISSING from the checkpoint file:")
            for key in sorted(list(missing_from_checkpoint)):
                print(f"    - {key}")
            if 'head.s' in missing_from_checkpoint:
                print("\n    >>> VERDICT: This confirms the checkpoint was saved with OLD code (without buffer 's'),")
                print("    >>> and you are trying to load it with NEW code (with buffer 's').")
                print("    >>> To load this, you MUST use `load_state_dict(..., strict=False)`.")
                print(
                    "    >>> If `strict=True` is NOT crashing, your Python environment is running STALE, CACHED code.")

        if extra_in_checkpoint:
            print(f"\n  ❌ DIAGNOSIS: Mismatch found! The checkpoint file has EXTRA keys the model does not recognize:")
            for key in sorted(list(extra_in_checkpoint)):
                print(f"    - {key}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diagnose PyTorch checkpoints and model architectures.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the .pth model checkpoint file.')
    parser.add_argument('--config', type=str, default='./config/detector/effort.yaml',
                        help='Path to the detector config.')
    parser.add_autorun('store_true', help='Specify if the model architecture to be tested uses an ArcFace head.')
    # This path needs to be valid for the script to instantiate the model
    parser.add_argument('--clip-model-path', type=str, default='./base_models/clip-vit-large-patch14',
                        help='Path to the base CLIP model.')

    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Config file not found at {args.config}")
        exit()

    inspect(args.checkpoint, config, args.clip_model_path, args.use_arcface_head)
