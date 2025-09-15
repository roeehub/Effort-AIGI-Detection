# fuse_weights.py
import os
import argparse
import yaml
from pathlib import Path
from collections import OrderedDict
import logging

import torch
from google.cloud import storage

# Make sure your project structure allows this import
from detectors.effort_detector import EffortDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def download_from_gcs(gcs_path: str, local_dir: str) -> str:
    """Downloads a file from GCS and returns the local path."""
    local_path = Path(local_dir) / Path(gcs_path).name
    if local_path.exists():
        logger.info(f"File already exists locally: {local_path}")
        return str(local_path)

    logger.info(f"Downloading {gcs_path}...")
    client = storage.Client()
    bucket_name, blob_name = gcs_path.replace("gs://", "").split('/', 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    logger.info("Download complete.")
    return str(local_path)


def main(args):
    # 1. Load config to build the model correctly
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Assuming a base train_config might also be needed
        with open('./config/train_config.yaml', 'r') as f:
            config.update(yaml.safe_load(f))
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        return

    # Ensure the config reflects the trained model's architecture
    config['use_arcface_head'] = args.use_arcface_head
    logger.info(f"Building model with use_arcface_head = {args.use_arcface_head}")

    # 2. Download checkpoint if necessary
    local_weights_path = download_from_gcs(args.checkpoint_gcs_path, "./weights/downloaded/")

    # 3. Load the original model with custom SVD layers
    logger.info("Loading original model with custom SVD layers...")
    model = EffortDetector(config)

    # Load the state dict from the training checkpoint
    state_dict = torch.load(local_weights_path, map_location='cpu')
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())

    model.load_state_dict(state_dict, strict=True)
    logger.info("Successfully loaded training checkpoint into custom model.")

    # 4. Create the new, clean state dictionary
    fused_state_dict = OrderedDict()

    logger.info("Starting weight fusion process...")
    for key, value in model.state_dict().items():
        # The SVD-related parameters are what we want to ELIMINATE.
        # We only care about the final effective 'weight' and 'bias'.
        if 'S_residual' in key or 'U_residual' in key or 'V_residual' in key or \
                'weight_main' in key or '_fnorm' in key or '_r' in key:
            continue  # Skip these component parts

        # If we encounter a bias, just copy it over.
        # The magic happens when we find a weight for a layer that was previously an SVDResidualLinear.
        if key.endswith('.weight'):
            # Reconstruct the original module path to access the custom module instance
            module_path = key.rsplit('.', 1)[0]
            try:
                sub_module = model.get_submodule(module_path)

                # Check if this was one of our custom layers
                if hasattr(sub_module, 'compute_current_weight'):
                    logger.info(f"Fusing weights for: {module_path}")
                    # This is the key step: calculate the final effective weight
                    fused_weight = sub_module.compute_current_weight()
                    fused_state_dict[key] = fused_weight
                else:
                    # It's a normal layer (like the head), so just copy its weight
                    fused_state_dict[key] = value
            except AttributeError:
                # This handles layers not found via get_submodule, like the head itself
                fused_state_dict[key] = value
        else:
            # Copy all other parameters (biases, embeddings, etc.) directly
            fused_state_dict[key] = value

    # 5. Save the new fused checkpoint
    output_path = Path(local_weights_path).parent / f"fused_{Path(local_weights_path).name}"
    torch.save(fused_state_dict, str(output_path))
    logger.info(f"✅ Fusion complete! Inference-ready checkpoint saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fuse SVD-decomposed weights into a standard checkpoint for inference.")
    parser.add_argument('--checkpoint-gcs-path', type=str, required=True, help='GCS path to the model weights.')
    parser.add_argument('--config', type=str, default='./config/detector/effort.yaml',
                        help='Path to the base detector config YAML.')
    parser.add_argument('--use-arcface-head', action='store_true',
                        help='Specify if the model was trained with the ArcFace head.')
    parsed_args = parser.parse_args()
    main(parsed_args)
