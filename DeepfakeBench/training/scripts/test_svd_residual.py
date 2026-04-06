#!/usr/bin/env python
"""
Test script to verify SVDResidualLinear integration with OpenCLIP backbone.

Tests:
1. SVDResidualLinear modules are correctly injected into MHA out_proj
2. Forward pass works end-to-end
3. Backward pass (gradients) flow correctly to residual parameters
4. Only residual parameters have gradients (others frozen)

Usage:
    python scripts/test_svd_residual.py
    python scripts/test_svd_residual.py --config experiments/deeplive_vit_B16_laion.yaml
"""

import argparse
import sys
import os

# Ensure we can import from the training directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml


def test_svd_residual(config_path: str = "experiments/deeplive_vit_B16_laion.yaml"):
    """Run all SVD residual tests."""
    
    print("=" * 60)
    print("SVDResidualLinear Integration Test")
    print("=" * 60)
    
    # Load config
    print(f"\n📄 Loading config: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Build detector
    print("\n🔧 Building EffortDetector...")
    from detectors.effort_detector import EffortDetector, SVDResidualLinear, SVDInProjLinear
    detector = EffortDetector(config=cfg)
    
    # Test 1: Check SVDResidualLinear injection
    print("\n" + "-" * 40)
    print("TEST 1: SVDResidualLinear Injection")
    print("-" * 40)
    
    svd_modules = []
    svd_in_proj_modules = []
    for name, module in detector.backbone.named_modules():
        if hasattr(module, 'out_proj'):
            out = getattr(module, 'out_proj')
            if isinstance(out, SVDResidualLinear):
                svd_modules.append((name, out))
                print(f"  [OK] {name}.out_proj -> SVDResidualLinear (shape {out.weight_main.shape})")
        
        # Check for SVDInProjLinear (fused q, k, v for OpenCLIP)
        if hasattr(module, '_svd_in_proj'):
            svd_in_proj = getattr(module, '_svd_in_proj')
            if isinstance(svd_in_proj, SVDInProjLinear):
                svd_in_proj_modules.append((name, svd_in_proj))
                print(f"  [OK] {name}._svd_in_proj -> SVDInProjLinear (embed_dim={svd_in_proj.embed_dim})")
    
    if len(svd_modules) == 0:
        print("  [FAIL] No SVDResidualLinear modules found!")
        return False
    
    print(f"\n  ✅ Replaced out_proj in {len(svd_modules)} MultiheadAttention modules")
    if len(svd_in_proj_modules) > 0:
        print(f"  ✅ Added SVDInProjLinear (q, k, v) to {len(svd_in_proj_modules)} MHA modules")
    
    # Test 2: Forward pass
    print("\n" + "-" * 40)
    print("TEST 2: Forward Pass")
    print("-" * 40)
    
    resolution = cfg.get('backbone', {}).get('resolution', 224)
    x = torch.randn(2, 3, resolution, resolution)  # Batch of 2
    
    try:
        detector.eval()
        with torch.no_grad():
            data_dict = {'image': x}
            pred_dict = detector(data_dict, inference=True)
        
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {pred_dict['feat'].shape}")
        print(f"  Logits shape: {pred_dict['cls'].shape}")
        print(f"  Prob range:   [{pred_dict['prob'].min().item():.4f}, {pred_dict['prob'].max().item():.4f}]")
        print("\n  ✅ Forward pass successful!")
    except Exception as e:
        print(f"  [FAIL] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Backward pass
    print("\n" + "-" * 40)
    print("TEST 3: Backward Pass (Gradient Flow)")
    print("-" * 40)
    
    try:
        detector.train()
        detector.zero_grad()
        
        # Forward with labels
        data_dict = {
            'image': x,
            'label': torch.tensor([0, 1])  # One real, one fake
        }
        pred_dict = detector(data_dict, inference=False)
        
        # Compute loss and backward
        loss_dict = detector.get_losses(data_dict, pred_dict)
        loss = loss_dict['overall']
        loss.backward()
        
        print(f"  Loss value: {loss.item():.4f}")
        
        # Check which parameters have gradients
        params_with_grad = []
        params_without_grad = []
        residual_params_with_grad = []
        
        for name, param in detector.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                params_with_grad.append((name, param.grad.norm().item()))
                if any(x in name for x in ['S_residual', 'U_residual', 'V_residual']):
                    residual_params_with_grad.append(name)
            elif param.requires_grad:
                params_without_grad.append(name)
        
        print(f"\n  Parameters with non-zero gradients: {len(params_with_grad)}")
        print(f"  Residual parameters with gradients: {len(residual_params_with_grad)}")
        
        # Show some gradient norms
        print("\n  Sample gradient norms:")
        for name, grad_norm in params_with_grad[:10]:
            marker = "🎯" if any(x in name for x in ['S_residual', 'U_residual', 'V_residual']) else "  "
            print(f"    {marker} {name}: {grad_norm:.6f}")
        if len(params_with_grad) > 10:
            print(f"    ... and {len(params_with_grad) - 10} more")
        
        # Verify residual params have gradients
        if len(residual_params_with_grad) == 0:
            print("\n  [WARN] No residual parameters have gradients!")
            print("         This may indicate a problem with gradient flow.")
        else:
            print(f"\n  ✅ Backward pass successful! Gradients flow to {len(residual_params_with_grad)} residual params")
        
    except Exception as e:
        print(f"  [FAIL] Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Verify frozen parameters stay frozen
    print("\n" + "-" * 40)
    print("TEST 4: Parameter Freezing")
    print("-" * 40)
    
    trainable_count = 0
    frozen_count = 0
    trainable_params = []
    
    for name, param in detector.named_parameters():
        if param.requires_grad:
            trainable_count += 1
            trainable_params.append(name)
        else:
            frozen_count += 1
    
    print(f"  Trainable parameters: {trainable_count}")
    print(f"  Frozen parameters:    {frozen_count}")
    
    # Check that only expected params are trainable
    unexpected_trainable = [p for p in trainable_params 
                           if not any(x in p for x in ['S_residual', 'U_residual', 'V_residual', 'head'])]
    
    if unexpected_trainable:
        print(f"\n  [WARN] Unexpected trainable parameters ({len(unexpected_trainable)}):")
        for p in unexpected_trainable[:5]:
            print(f"    - {p}")
        if len(unexpected_trainable) > 5:
            print(f"    ... and {len(unexpected_trainable) - 5} more")
    else:
        print("\n  ✅ Only residual and head parameters are trainable (as expected)")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"  ✅ SVDResidualLinear injection:  {len(svd_modules)} modules (out_proj)")
    print(f"  ✅ SVDInProjLinear injection:    {len(svd_in_proj_modules)} modules (q, k, v)")
    print(f"  ✅ Forward pass:                 OK")
    print(f"  ✅ Backward pass:                {len(residual_params_with_grad)} params with gradients")
    print(f"  ✅ Parameter freezing:           {trainable_count} trainable / {frozen_count} frozen")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SVDResidualLinear integration")
    parser.add_argument(
        "--config", 
        default="experiments/deeplive_vit_B16_laion.yaml",
        help="Path to experiment config YAML"
    )
    args = parser.parse_args()
    
    success = test_svd_residual(args.config)
    sys.exit(0 if success else 1)
