# fuse_weights.py
import os
import argparse
import yaml
from pathlib import Path
from collections import OrderedDict
import logging

import torch
import torch.nn as nn
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


# =============================================================================
# OpenCLIP SVD Weight Fusion (Task B)
# =============================================================================

def fuse_openclip_svd_weights(model, return_state_dict=False):
    """
    Fuse SVD-decomposed weights from OpenCLIP backbone into standard weights.
    
    This function handles the SVDMultiheadAttentionWrapper modules used in
    OpenCLIP backbones. It computes the fused weight from:
    
        W_fused = W_main + U_residual @ diag(S_residual) @ V_residual
    
    And then restores the original nn.MultiheadAttention structure with the
    fused weight in out_proj.
    
    Args:
        model: EffortDetector model with OpenCLIP backbone
        return_state_dict: If True, return the fused state_dict instead of modifying in place
        
    Returns:
        If return_state_dict=True: OrderedDict with fused weights
        Otherwise: The modified model with fused weights
        
    Usage:
        # Option 1: Modify model in place
        fuse_openclip_svd_weights(model)
        
        # Option 2: Get fused state dict for saving
        fused_state_dict = fuse_openclip_svd_weights(model, return_state_dict=True)
        torch.save(fused_state_dict, 'fused_checkpoint.pth')
        
        # Option 3: Load into a fresh model
        clean_model = create_clean_openclip_model(config)
        clean_model.load_state_dict(fused_state_dict)
    """
    from detectors.effort_detector import SVDMultiheadAttentionWrapper, SVDResidualLinear
    
    fused_count = 0
    
    if return_state_dict:
        fused_state_dict = OrderedDict()
    
    def fuse_svd_mha_wrapper(wrapper):
        """Compute fused out_proj weight from SVDMultiheadAttentionWrapper."""
        svd_layer = wrapper.svd_out_proj
        
        # Compute fused weight: W_main + U @ diag(S) @ V
        weight_main = svd_layer.weight_main.data
        
        if svd_layer.S_residual is not None and svd_layer.U_residual is not None and svd_layer.V_residual is not None:
            residual = svd_layer.U_residual @ torch.diag(svd_layer.S_residual) @ svd_layer.V_residual
            fused_weight = weight_main + residual
        else:
            fused_weight = weight_main
            
        return fused_weight
    
    def fuse_svd_linear(svd_linear):
        """Compute fused weight from SVDResidualLinear."""
        if hasattr(svd_linear, 'compute_current_weight'):
            return svd_linear.compute_current_weight()
        
        # Manual computation if method doesn't exist
        weight_main = svd_linear.weight_main.data
        
        if svd_linear.S_residual is not None and svd_linear.U_residual is not None and svd_linear.V_residual is not None:
            residual = svd_linear.U_residual @ torch.diag(svd_linear.S_residual) @ svd_linear.V_residual
            return weight_main + residual
        else:
            return weight_main
    
    # Iterate through all modules and fuse SVD weights
    for name, module in model.named_modules():
        if isinstance(module, SVDMultiheadAttentionWrapper):
            logger.info(f"Fusing OpenCLIP SVD wrapper: {name}")
            fused_weight = fuse_svd_mha_wrapper(module)
            
            if return_state_dict:
                # Store with the original out_proj path
                # The wrapper stores original MHA in .mha, so the weight path would be:
                # {name}.mha.out_proj.weight
                key_weight = f"{name}.mha.out_proj.weight"
                key_bias = f"{name}.mha.out_proj.bias" if module.out_proj_bias is not None else None
                fused_state_dict[key_weight] = fused_weight
                if key_bias:
                    fused_state_dict[key_bias] = module.out_proj_bias.data
            else:
                # In-place: restore the out_proj weight in the wrapped MHA
                module.mha.out_proj.weight.data.copy_(fused_weight)
                module.mha.out_proj.weight.requires_grad = False
                
            fused_count += 1
            
        elif isinstance(module, SVDResidualLinear):
            logger.info(f"Fusing SVDResidualLinear: {name}")
            fused_weight = fuse_svd_linear(module)
            
            if return_state_dict:
                # Convert path from svd to standard linear format
                key_weight = f"{name}.weight"
                key_bias = f"{name}.bias"
                fused_state_dict[key_weight] = fused_weight
                if module.bias is not None:
                    fused_state_dict[key_bias] = module.bias.data
            
            fused_count += 1
    
    logger.info(f"✅ Fused {fused_count} SVD modules")
    
    if return_state_dict:
        # Also copy non-SVD parameters
        for name, param in model.named_parameters():
            # Skip SVD-specific parameters that we've already handled
            if any(x in name for x in ['S_residual', 'U_residual', 'V_residual', 
                                        'weight_main', 'svd_out_proj', '_fnorm', 
                                        '_r', 'U_r', 'V_r', 'S_r']):
                continue
            
            # Skip parameters we've already added
            if name in fused_state_dict:
                continue
                
            fused_state_dict[name] = param.data
            
        return fused_state_dict
    
    return model


def create_openclip_inference_checkpoint(
    trained_checkpoint_path: str,
    config: dict,
    output_path: str = None
) -> str:
    """
    Create an inference-ready checkpoint from a trained OpenCLIP model.
    
    This function:
    1. Loads the trained checkpoint with SVD wrappers
    2. Fuses SVD weights back into standard format
    3. Saves a clean checkpoint that can be loaded by standard OpenCLIP
    
    Args:
        trained_checkpoint_path: Path to trained checkpoint (.pth)
        config: Model configuration dict
        output_path: Where to save fused checkpoint (default: adds 'fused_' prefix)
        
    Returns:
        Path to the fused checkpoint
        
    Example:
        fused_path = create_openclip_inference_checkpoint(
            'checkpoints/epoch_10.pth',
            config,
            'checkpoints/fused_epoch_10.pth'
        )
    """
    logger.info(f"Loading trained checkpoint: {trained_checkpoint_path}")
    
    # Load model with SVD architecture
    model = EffortDetector(config)
    state_dict = torch.load(trained_checkpoint_path, map_location='cpu')
    
    # Handle DataParallel prefix
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
    
    model.load_state_dict(state_dict, strict=True)
    logger.info("Loaded checkpoint into model")
    
    # Fuse SVD weights
    fused_state_dict = fuse_openclip_svd_weights(model, return_state_dict=True)
    
    # Determine output path
    if output_path is None:
        checkpoint_path = Path(trained_checkpoint_path)
        output_path = checkpoint_path.parent / f"fused_{checkpoint_path.name}"
    
    # Save fused checkpoint
    torch.save(fused_state_dict, output_path)
    logger.info(f"✅ Saved inference-ready checkpoint: {output_path}")
    
    return str(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fuse SVD-decomposed weights into a standard checkpoint for inference.")
    parser.add_argument('--checkpoint-gcs-path', type=str, required=True, help='GCS path to the model weights.')
    parser.add_argument('--config', type=str, default='./config/detector/effort.yaml',
                        help='Path to the base detector config YAML.')
    parser.add_argument('--use-arcface-head', action='store_true',
                        help='Specify if the model was trained with the ArcFace head.')
    parser.add_argument('--openclip', action='store_true',
                        help='Use OpenCLIP-specific fusion (for LAION backbones).')
    parsed_args = parser.parse_args()
    main(parsed_args)
