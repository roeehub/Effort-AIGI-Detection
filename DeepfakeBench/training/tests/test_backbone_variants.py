#!/usr/bin/env python3
"""
Test script to verify the 3 NEW backbone variants work correctly with the Effort detector.

This script tests:
1. ViT-B-16 (OpenAI, HuggingFace)
2. ViT-B-32 (OpenAI, HuggingFace)
3. ViT-B-16-DataComp-XL (LAION, OpenCLIP)

NOTE: ViT-L-14 is NOT tested here as it's the baseline already in production.

Usage:
    python test_backbone_variants.py --variant all
    python test_backbone_variants.py --variant vit-b-16
    python test_backbone_variants.py --variant vit-b-32
    python test_backbone_variants.py --variant datacomp

Requirements:
    - transformers (for HuggingFace models)
    - open_clip_torch (for LAION models)
    - torch
"""

import argparse
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Local paths for testing (will be downloaded from GCS in production)
LOCAL_WEIGHT_PATHS = {
    'vit-b-16': '/Users/roeedar/Downloads/patch_16',
    'vit-b-32': '/Users/roeedar/Downloads/patch_32',
    'datacomp': '/Users/roeedar/Downloads/openclip',
}

# Backbone configurations for the 3 NEW variants (ViT-L-14 is baseline, not tested)
BACKBONE_CONFIGS = {
    'vit-b-16': {
        'name': 'ViT-B-16 (OpenAI)',
        'config': {
            'backbone': {
                'type': 'clip',
                'variant': 'ViT-B-16',
                'source': 'openai',
                'hidden_size': 768,
                'resolution': 224,
            },
            'rank': 767,
            'gcs_assets': {
                'clip_backbone': {
                    'local_path': LOCAL_WEIGHT_PATHS['vit-b-16']
                }
            }
        },
        'library': 'transformers',
    },
    'vit-b-32': {
        'name': 'ViT-B-32 (OpenAI)',
        'config': {
            'backbone': {
                'type': 'clip',
                'variant': 'ViT-B-32',
                'source': 'openai',
                'hidden_size': 768,
                'resolution': 224,
            },
            'rank': 767,
            'gcs_assets': {
                'clip_backbone': {
                    'local_path': LOCAL_WEIGHT_PATHS['vit-b-32']
                }
            }
        },
        'library': 'transformers',
    },
    'datacomp': {
        'name': 'ViT-B-16-DataComp-XL (LAION)',
        'config': {
            'backbone': {
                'type': 'clip',
                'variant': 'ViT-B-16',
                'source': 'laion',
                'openclip_model': 'ViT-B-16',
                'openclip_pretrained': 'datacomp_xl_s13b_b90k',
                'hidden_size': 512,  # OpenCLIP output_dim (projected from 768 embed_dim)
                'resolution': 224,
            },
            'rank': 511,  # output_dim - 1
            'gcs_assets': {
                'clip_backbone': {
                    'local_path': LOCAL_WEIGHT_PATHS['datacomp']
                }
            }
        },
        'library': 'open_clip',
    },
}


def check_dependencies(variant_config):
    """Check if required libraries are installed."""
    library = variant_config['library']
    
    if library == 'transformers':
        try:
            import transformers
            logger.info(f"✓ transformers version: {transformers.__version__}")
            return True
        except ImportError:
            logger.error("✗ transformers not installed. Run: pip install transformers")
            return False
    
    elif library == 'open_clip':
        try:
            import open_clip
            logger.info(f"✓ open_clip version: {open_clip.__version__}")
            return True
        except ImportError:
            logger.error("✗ open_clip not installed. Run: pip install open_clip_torch")
            return False
    
    return True


def download_if_needed(variant_key, variant_config):
    """Download model weights if not already present."""
    config = variant_config['config']
    local_path = config['gcs_assets']['clip_backbone']['local_path']
    
    if os.path.exists(local_path):
        logger.info(f"✓ Weights already present at: {local_path}")
        return True
    
    logger.info(f"Weights not found at: {local_path}")
    
    if variant_config['library'] == 'transformers':
        huggingface_id = f"openai/clip-vit-{variant_key.replace('vit-', '').replace('-', '-patch')}"
        if variant_key == 'vit-l-14':
            huggingface_id = "openai/clip-vit-large-patch14"
        elif variant_key == 'vit-b-16':
            huggingface_id = "openai/clip-vit-base-patch16"
        elif variant_key == 'vit-b-32':
            huggingface_id = "openai/clip-vit-base-patch32"
        
        logger.info(f"Downloading from HuggingFace: {huggingface_id}")
        try:
            from transformers import CLIPModel
            model = CLIPModel.from_pretrained(huggingface_id)
            logger.info(f"✓ Downloaded {huggingface_id}")
            
            # Save to local path
            os.makedirs(local_path, exist_ok=True)
            model.save_pretrained(local_path)
            logger.info(f"✓ Saved to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download: {e}")
            return False
    
    elif variant_config['library'] == 'open_clip':
        backbone_config = config['backbone']
        openclip_model = backbone_config['openclip_model']
        openclip_pretrained = backbone_config['openclip_pretrained']
        
        logger.info(f"Downloading OpenCLIP: {openclip_model} ({openclip_pretrained})")
        try:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(
                openclip_model, 
                pretrained=openclip_pretrained
            )
            logger.info(f"✓ Downloaded {openclip_model} ({openclip_pretrained})")
            # Note: OpenCLIP caches models in ~/.cache/huggingface/hub/
            # We don't need to save separately for now
            return True
        except Exception as e:
            logger.error(f"Failed to download: {e}")
            return False
    
    return False


def test_backbone(variant_key, variant_config, skip_download=False):
    """Test a single backbone variant."""
    import torch
    
    name = variant_config['name']
    config = variant_config['config']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {name}")
    logger.info(f"{'='*60}")
    
    # Check dependencies
    if not check_dependencies(variant_config):
        return False
    
    # Download if needed
    if not skip_download:
        if not download_if_needed(variant_key, variant_config):
            logger.warning("Skipping test due to missing weights (use --skip-download to test with auto-download disabled)")
            return False
    
    # Create model
    try:
        from detectors.effort_detector import EffortDetector
        
        logger.info("Creating EffortDetector...")
        model = EffortDetector(config)
        logger.info(f"✓ Model created successfully")
        logger.info(f"  - Hidden size: {model.hidden_size}")
        logger.info(f"  - Rank: {model.rank}")
        logger.info(f"  - Head type: {type(model.head).__name__}")
        
    except Exception as e:
        logger.error(f"✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test forward pass
    try:
        logger.info("Testing forward pass...")
        
        # Create dummy input
        batch_size = 2
        resolution = config['backbone'].get('resolution', 224)
        dummy_input = {
            'image': torch.randn(batch_size, 3, resolution, resolution),
            'label': torch.tensor([0, 1])
        }
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            features = model.features(dummy_input)
            logits = model.classifier(features)
        
        logger.info(f"✓ Forward pass successful")
        logger.info(f"  - Features shape: {features.shape}")
        logger.info(f"  - Logits shape: {logits.shape}")
        
        # Verify shapes
        expected_hidden = config['backbone']['hidden_size']
        assert features.shape == (batch_size, expected_hidden), \
            f"Expected features shape ({batch_size}, {expected_hidden}), got {features.shape}"
        assert logits.shape == (batch_size, 2), \
            f"Expected logits shape ({batch_size}, 2), got {logits.shape}"
        
        logger.info(f"✓ Shape verification passed")
        
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test backward pass (gradient flow)
    try:
        logger.info("Testing backward pass (gradient flow)...")
        
        model.train()
        dummy_input = {
            'image': torch.randn(batch_size, 3, resolution, resolution),
            'label': torch.tensor([0, 1])
        }
        
        features = model.features(dummy_input)
        logits = model.classifier(features)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(logits, dummy_input['label'])
        loss.backward()
        
        # Check that SVD residual params have gradients
        grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
        
        logger.info(f"✓ Backward pass successful")
        logger.info(f"  - Parameters with gradients: {grad_count}")
        
    except Exception as e:
        logger.error(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Count trainable parameters
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"\nParameter counts:")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Frozen parameters: {total_params - trainable_params:,}")
        logger.info(f"  - Trainable ratio: {trainable_params / total_params * 100:.2f}%")
        
    except Exception as e:
        logger.warning(f"Could not count parameters: {e}")
    
    logger.info(f"\n✓ All tests passed for {name}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Test backbone variants for Effort detector')
    parser.add_argument('--variant', type=str, default='all',
                       choices=['all', 'vit-b-16', 'vit-b-32', 'datacomp'],
                       help='Which variant to test (vit-l-14 excluded - already in production)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip automatic download of missing weights')
    args = parser.parse_args()
    
    # Change to training directory
    training_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(training_dir)
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Determine which variants to test
    if args.variant == 'all':
        variants_to_test = list(BACKBONE_CONFIGS.keys())
    else:
        variants_to_test = [args.variant]
    
    # Run tests
    results = {}
    for variant_key in variants_to_test:
        variant_config = BACKBONE_CONFIGS[variant_key]
        success = test_backbone(variant_key, variant_config, args.skip_download)
        results[variant_key] = success
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    
    all_passed = True
    for variant_key, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"  {BACKBONE_CONFIGS[variant_key]['name']}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        logger.info("\n✓ All backbone variants working correctly!")
        return 0
    else:
        logger.error("\n✗ Some backbone variants failed. See above for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
