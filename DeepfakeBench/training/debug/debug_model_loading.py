#!/usr/bin/env python3
"""
Debug script to investigate model loading and inference issues.
Compares training vs inference behavior for EffortDetector with ArcFace.
"""

import os
import sys

# Add project paths
current_file_path = os.path.abspath(__file__)
debug_folder = os.path.dirname(current_file_path)
training_folder = os.path.dirname(debug_folder)  # Go up one level to training/
project_root = os.path.dirname(os.path.dirname(training_folder))  # Go up to repo root

# Add training folder (where detectors module is) to Python path
sys.path.insert(0, training_folder)
sys.path.insert(0, project_root)

print(f"Debug folder: {debug_folder}")
print(f"Training folder: {training_folder}")
print(f"Project root: {project_root}")

from detectors import DETECTOR
import torch.nn.functional as F
import yaml
from pathlib import Path
import numpy as np

# Import GCS libraries with graceful fallback
try:
    from google.cloud import storage
    from google.api_core import exceptions
    from google.cloud.storage import Bucket

    GCS_AVAILABLE = True
except ImportError:
    print("Warning: Google Cloud Storage libraries not available. GCS downloads will not work.")
    GCS_AVAILABLE = False


    # Mock classes for syntax checking
    class storage:
        @staticmethod
        def Client():
            return None


    class exceptions:
        Forbidden = Exception
        NotFound = Exception


    class Bucket:
        pass

import torch
import torch.nn.functional as F


@torch.no_grad()
def forward_logits_only(m, x):
    m.eval()
    # EffortDetector expects a dict with 'image' key
    if isinstance(x, torch.Tensor):
        data_dict = {'image': x}
    else:
        data_dict = x

    out = m(data_dict, inference=True)
    # EffortDetector returns dict with 'raw_logits', 'prob', 'feat'
    if isinstance(out, dict) and "raw_logits" in out:
        logits = out["raw_logits"]
    elif isinstance(out, dict) and "cls" in out:
        logits = out["cls"]
    elif isinstance(out, dict) and "logits" in out:
        logits = out["logits"]
    elif isinstance(out, (list, tuple)):
        logits = out[0]
    else:
        logits = out
    return logits.float()


def download_gcs_asset(bucket: Bucket, gcs_path: str, local_path: str) -> bool:
    """
    Downloads a single blob or a directory of blobs from GCS.
    Adapted from train_sweep.py
    """
    if gcs_path.endswith('/'):  # It's a directory
        prefix = gcs_path.split(bucket.name + '/', 1)[1]
        blobs = bucket.list_blobs(prefix=prefix)
        downloaded = False
        for blob in blobs:
            if blob.name.endswith('/'):  # Skip directory markers
                continue
            local_blob_path = os.path.join(local_path, os.path.relpath(blob.name, prefix))
            os.makedirs(os.path.dirname(local_blob_path), exist_ok=True)
            blob.download_to_filename(local_blob_path)
            print(f"Downloaded: {blob.name} -> {local_blob_path}")
            downloaded = True
        if not downloaded:
            print(f"No files found in directory: {gcs_path}")
        return True
    else:  # It's a single file
        blob_name = gcs_path.split(bucket.name + '/', 1)[1]
        blob = bucket.blob(blob_name)
        if not blob.exists():
            print(f"Blob not found: {gcs_path}")
            return False
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"Downloaded: {gcs_path} -> {local_path}")
        return True


def download_assets_from_gcs(assets_config):
    """
    Downloads specified assets from GCS if they don't already exist.
    Returns True if all assets are available locally.
    """
    if not GCS_AVAILABLE:
        print("ERROR: Google Cloud Storage libraries not available. Cannot download from GCS.")
        print("Please install google-cloud-storage: pip install google-cloud-storage")
        return False

    if not assets_config:
        print("No GCS assets configured for download.")
        return False

    # First, check if all assets already exist locally
    all_exist = True
    for key, asset_info in assets_config.items():
        local_path = asset_info.get('local_path')
        if not local_path or not os.path.exists(local_path):
            all_exist = False
            break

    if all_exist:
        print("All assets already exist locally. Skipping downloads.")
        return True

    print("\n=== GCS Asset Download ===")
    try:
        storage_client = storage.Client()

        for key, asset_info in assets_config.items():
            gcs_path = asset_info.get('gcs_path')
            local_path = asset_info.get('local_path')

            if not gcs_path or not local_path:
                print(f"Skipping {key}: missing gcs_path or local_path")
                continue

            # Check if already exists
            if os.path.exists(local_path):
                print(f"Asset {key} already exists at {local_path}, skipping")
                continue

            print(f"Downloading {key}: {gcs_path} -> {local_path}")

            # Parse GCS path
            parts = gcs_path.replace('gs://', '').split('/')
            bucket_name = parts[0]

            bucket = storage_client.bucket(bucket_name)
            success = download_gcs_asset(bucket, gcs_path, local_path)

            if not success:
                print(f"Failed to download {key}")
                return False

        print("✅ All GCS assets downloaded successfully.")
        return True

    except exceptions.Forbidden as e:
        print(f"GCP Permissions error: {e}")
        return False
    except exceptions.NotFound as e:
        print(f"GCS asset not found: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during download: {e}")
        return False


def load_config():
    """Load a basic config for EffortDetector with ArcFace."""
    config = {
        'use_arcface_head': True,
        'arcface_s': 30.0,
        'arcface_m': 0.35,
        's_start': 30.0,
        's_end': 30.0,
        'anneal_steps': 0,
        'lambda_reg': 1.0,
        'rank': 1023,
        'gcs_assets': {
            'base_checkpoint': {
                'gcs_path': "gs://training-job-outputs/best_checkpoints/o08s5u94/top_n_effort_20250927_ep1_auc0.9631_eer0.1007.pth",
                'local_path': "./debug_model/checkpoint.pth"
            },
            'clip_backbone': {
                'gcs_path': "gs://base-checkpoints/effort-aigi/models--openai--clip-vit-large-patch14/",
                'local_path': "./debug_model/models--openai--clip-vit-large-patch14/"
            }
        }
    }
    return config


def inspect_checkpoint(ckpt_path):
    """Inspect checkpoint contents."""
    print(f"\n=== INSPECTING CHECKPOINT: {ckpt_path} ===")

    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return None

    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        print(f"Checkpoint keys: {list(checkpoint.keys())}")

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"Model state dict has {len(state_dict)} parameters")

            # Look for ArcFace parameters
            arcface_params = {k: v for k, v in state_dict.items() if 'head.' in k}
            print(f"ArcFace head parameters:")
            for k, v in arcface_params.items():
                if v.numel() == 1:  # Scalar values like 's'
                    print(f"  {k}: {v.item()}")
                else:
                    print(f"  {k}: shape {v.shape}")

        # Check other metadata
        for key in ['epoch', 'auc', 'eer', 'acc']:
            if key in checkpoint:
                print(f"{key}: {checkpoint[key]}")

        return checkpoint

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def create_test_batch():
    """Create a dummy test batch."""
    batch_size = 4
    data_dict = {
        'image': torch.randn(batch_size, 3, 224, 224),  # Random images
        'label': torch.randint(0, 2, (batch_size,))  # Random binary labels
    }
    return data_dict


def test_model_behavior(model, data_dict, test_name=""):
    """Test model behavior and return detailed outputs."""
    print(f"\n=== TESTING: {test_name} ===")

    model.eval()
    with torch.no_grad():
        # Test inference mode
        pred_dict_inference = model(data_dict, inference=True)

        # Test training mode (simulate)
        pred_dict_training = model(data_dict, inference=False)

    # Compare outputs
    print("Inference mode outputs:")
    for key, value in pred_dict_inference.items():
        if torch.is_tensor(value):
            print(f"  {key}: shape {value.shape}, range [{value.min():.4f}, {value.max():.4f}]")
            if key == 'prob':
                print(f"    probabilities: {value.cpu().numpy()}")

    print("Training mode outputs:")
    for key, value in pred_dict_training.items():
        if torch.is_tensor(value):
            print(f"  {key}: shape {value.shape}, range [{value.min():.4f}, {value.max():.4f}]")
            if key == 'prob':
                print(f"    probabilities: {value.cpu().numpy()}")

    # Check if predictions are different
    pred_diff = torch.abs(pred_dict_inference['prob'] - pred_dict_training['prob'])
    print(f"Max probability difference between inference/training: {pred_diff.max():.6f}")

    return pred_dict_inference, pred_dict_training


def compare_arcface_parameters(model1, model2, name1="Model1", name2="Model2"):
    """Compare ArcFace parameters between two models."""
    print(f"\n=== COMPARING ARCFACE PARAMETERS: {name1} vs {name2} ===")

    if hasattr(model1, 'head') and hasattr(model2, 'head'):
        if hasattr(model1.head, 's') and hasattr(model2.head, 's'):
            s1 = model1.head.s.item()
            s2 = model2.head.s.item()
            print(f"Scale (s): {name1}={s1}, {name2}={s2}, diff={abs(s1 - s2)}")

        if hasattr(model1.head, 'm') and hasattr(model2.head, 'm'):
            m1 = model1.head.m
            m2 = model2.head.m
            print(f"Margin (m): {name1}={m1}, {name2}={m2}, diff={abs(m1 - m2)}")

        # Compare weights
        if hasattr(model1.head, 'weight') and hasattr(model2.head, 'weight'):
            weight_diff = torch.norm(model1.head.weight - model2.head.weight).item()
            print(f"Weight difference (Frobenius norm): {weight_diff}")


def main():
    print("=== EFFORT DETECTOR DEBUG SCRIPT ===")

    # Load config with GCS assets
    config = load_config()
    print(f"\n=== CONFIG ===")
    for k, v in config.items():
        if k != 'gcs_assets':  # Skip nested dict for cleaner output
            print(f"{k}: {v}")

    # Download all assets from GCS
    print("\n=== DOWNLOADING ASSETS ===")
    success = download_assets_from_gcs(config['gcs_assets'])
    if not success:
        print("Failed to download required assets. Please check your GCS configuration.")
        return

    # Get local paths
    local_model_path = config['gcs_assets']['base_checkpoint']['local_path']
    clip_path = config['gcs_assets']['clip_backbone']['local_path']

    print(f"\nUsing checkpoint: {local_model_path}")
    print(f"Using CLIP backbone: {clip_path}")

    # Inspect checkpoint
    checkpoint = inspect_checkpoint(local_model_path)
    if checkpoint is None:
        return

    # Verify CLIP model exists
    if not os.path.exists(clip_path):
        print(f"\nERROR: CLIP model not found at {clip_path}")
        print("This should have been downloaded. Please check the GCS path.")
        return

    try:
        # Create model with config
        print("\n=== CREATING MODEL ===")
        cfg = dict(type='effort', config=config)

        # Create model using the registry
        try:
            EffortCls = DETECTOR[cfg["type"]]  # Get the registered class
            model = EffortCls(config=cfg["config"])
        except KeyError:
            raise RuntimeError(
                f"'{cfg['type']}' not registered in DETECTOR. Available: {list(DETECTOR.data.keys())}")

        print("Model created.")
        print("Registry type:", type(DETECTOR))
        print("Known detector keys:", list(DETECTOR.data.keys()))
        print(f"Model created successfully")
        print(f"Using ArcFace: {model.use_arcface_head}")

        if model.use_arcface_head:
            print(f"Initial s: {model.head.s.item()}")
            print(f"Margin m: {model.head.m}")

        # Load checkpoint
        print(f"\n=== LOADING CHECKPOINT ===")
        if 'model_state_dict' in checkpoint:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")

        sd = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        print("\n=== CHECKPOINT INSPECT ===")
        print("Has head.s?:", any(k.endswith("head.s") for k in sd.keys()))

        # If your model exposes them:
        if hasattr(model, "head"):
            if hasattr(model.head, "s"):
                try:
                    print("Loaded head.s:", float(model.head.s))
                except Exception:
                    print("Loaded head.s (raw):", model.head.s)
            if hasattr(model.head, "m"):
                try:
                    print("Loaded head.m:", float(model.head.m))
                except Exception:
                    print("Loaded head.m (raw):", model.head.m)

        if model.use_arcface_head:
            print(f"Loaded s: {model.head.s.item()}")
            print(f"Loaded m: {model.head.m}")

        # 3) what is model.head.s AFTER load?
        if hasattr(model, "head") and hasattr(model.head, "s"):
            try:
                print("model.head.s AFTER load:", float(model.head.s))
            except Exception:
                print("model.head.s AFTER load (raw):", model.head.s)

        print("\n=== LOADING STATE DICT (inspection with strict=True) ===")
        state_dict = checkpoint.get("state_dict", checkpoint)

        # try a dry strict load to see drift
        missing_keys = []
        unexpected_keys = []
        error_msg = None
        try:
            # Do a strict test load into a fresh, temporary model copy if possible
            import copy
            tmp_model = copy.deepcopy(model)
            result = tmp_model.load_state_dict(state_dict, strict=True)
            # if strict=True returns without exception, result should be empty lists
            print("Strict load succeeded: no missing/unexpected keys.")
        except Exception as e:
            error_msg = str(e)
            print("Strict load RAISED an exception (expected if keys drift):")
            print(error_msg)

        # Now do the real load with strict=False so we can proceed
        print("\n=== LOADING STATE DICT (proceed with strict=False) ===")
        res = model.load_state_dict(state_dict, strict=False)
        # res is a _IncompatibleKeys with .missing_keys and .unexpected_keys
        try:
            missing_keys = list(getattr(res, "missing_keys", []))
            unexpected_keys = list(getattr(res, "unexpected_keys", []))
        except Exception:
            pass

        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

        print("\n=== ARCFACE s ORIGIN CHECK ===")
        # 1) what is exactly stored in the checkpoint?
        if "head.s" in state_dict:
            try:
                print("state_dict['head.s']:", float(state_dict["head.s"].detach().cpu()))
            except Exception:
                print("state_dict['head.s'] (raw):", state_dict["head.s"])

        # 2) what is model.head.s BEFORE load?
        if hasattr(model, "head") and hasattr(model.head, "s"):
            try:
                print("model.head.s BEFORE load:", float(model.head.s))
            except Exception:
                print("model.head.s BEFORE load (raw):", model.head.s)

        # quick norms to sanity-check the head actually loaded
        if hasattr(model, "head"):
            if hasattr(model.head, "weight"):
                try:
                    print("||head.weight||:", float(model.head.weight.norm()))
                except Exception:
                    print("head.weight norm: <unavailable>")
            if hasattr(model.head, "s"):
                try:
                    print("head.s (after load):", float(model.head.s))
                except Exception:
                    print("head.s (raw):", model.head.s)
            if hasattr(model.head, "m"):
                try:
                    print("head.m (after load):", float(model.head.m))
                except Exception:
                    print("head.m (raw):", model.head.m)

        # Create test data
        print(f"\n=== CREATING TEST DATA ===")
        data_dict = create_test_batch()
        print(f"Test batch: {data_dict['image'].shape}, labels: {data_dict['label']}")

        print("\n=== SENSITIVITY TEST: rescale logits by s ===")
        import torch
        device = next(model.parameters()).device
        x_test = torch.randn(4, 3, 224, 224, device=device)  # or reuse your test batch
        logits_loaded = forward_logits_only(model, x_test)

        # current s in the loaded model:
        s_loaded = float(model.head.s) if hasattr(model.head, "s") else 1.0

        # A) naive "force s=30" run (temporarily override)
        restore_s = None
        if hasattr(model.head, "s"):
            restore_s = model.head.s.data.clone()
            model.head.s.data.fill_(30.0)
        logits_s30 = forward_logits_only(model, x_test)
        if restore_s is not None:
            model.head.s.data.copy_(restore_s)

        # B) purely rescale logits by factor (30/s_loaded) without changing model
        scale = 30.0 / max(s_loaded, 1e-6)
        logits_rescaled = logits_loaded * scale

        def summarize(tag, logits):
            probs = F.softmax(logits, dim=1)[:, 1]
            print(f"{tag}: logits range [{logits.min().item():.4f}, {logits.max().item():.4f}]  "
                  f"p(fake) range [{probs.min().item():.4f}, {probs.max().item():.4f}]  "
                  f"mean p(fake) {probs.mean().item():.4f}")

        summarize("loaded (s≈current)", logits_loaded)
        summarize("forced s=30 (rerun)", logits_s30)
        summarize("rescaled logits * (30/s_loaded)", logits_rescaled)
        print(f"observed s_loaded={s_loaded:.6f}, scale factor={scale:.6f}")

        print("\n=== DTYPE / DEVICE DRIFT AUDIT ===")
        import torch

        # 1) lock defaults to a clean baseline
        torch.set_default_dtype(torch.float32)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval().to(device)

        # 2) list unique devices/dtypes across params & buffers
        param_devices = sorted({str(p.device) for p in model.parameters()})
        param_dtypes = sorted({str(p.dtype) for p in model.parameters()})
        buffer_devices = sorted({str(b.device) for b in model.buffers()}) if any(True for _ in model.buffers()) else []
        buffer_dtypes = sorted({str(b.dtype) for b in model.buffers()}) if any(True for _ in model.buffers()) else []

        print("Param devices:", param_devices)
        print("Param dtypes :", param_dtypes)
        print("Buffer devices:", buffer_devices)
        print("Buffer dtypes :", buffer_dtypes)
        print("Selected device:", device)

        # 3) make sure everything is actually fp32 (for now)
        mismatch = [(n, p.dtype, p.device) for (n, p) in model.named_parameters() if p.dtype != torch.float32]
        print("Non-fp32 params:", len(mismatch))
        if mismatch:
            for n, dt, dv in mismatch[:10]:
                print("  -", n, dt, dv)

        # 4) run a tiny forward in fp32 vs autocast and compare numerically
        @torch.no_grad()
        def get_logits(m, x):
            # EffortDetector expects a dictionary with 'image' key
            if hasattr(m, '__class__') and 'EffortDetector' in m.__class__.__name__:
                data_dict = {'image': x}
                out = m(data_dict, inference=True)  # Always use inference mode for get_logits
            else:
                out = m(x)
            if isinstance(out, dict) and "cls" in out:
                return out["cls"].float()
            if isinstance(out, (list, tuple)):
                return out[0].float()
            return out.float()

        x = torch.randn(4, 3, 224, 224, device=device, dtype=torch.float32)

        # fp32
        logits32 = get_logits(model, x)
        probs32 = F.softmax(logits32, dim=1)[:, 1]

        # autocast half/bfloat16 (only if cuda; cpu autocast is bfloat16)
        use_amp = (device == "cuda")
        if use_amp:
            amp_dtype = torch.float16  # we’ll test fp16 first
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits_amp = get_logits(model, x)
            probs_amp = F.softmax(logits_amp.float(), dim=1)[:, 1]

            # deltas
            import math
            diff_logits = (logits32 - logits_amp.float()).abs().max().item()
            diff_probs = (probs32 - probs_amp).abs().max().item()

            print(f"AMP test (cuda, dtype={amp_dtype}): max|Δlogits|={diff_logits:.6f}, max|Δp(fake)|={diff_probs:.6f}")
        else:
            print("AMP test skipped (no cuda).")

        print(
            f"fp32 p(fake) range: [{probs32.min().item():.6f}, {probs32.max().item():.6f}]  mean={probs32.mean().item():.6f}")

        # Test model behavior
        pred_inference, pred_training = test_model_behavior(model, data_dict, "Loaded Model")

        # Test with fresh model (no loaded weights)
        print(f"\n=== COMPARING WITH FRESH MODEL ===")
        fresh_model = DETECTOR['effort'](config=config)
        pred_fresh_inf, pred_fresh_train = test_model_behavior(fresh_model, data_dict, "Fresh Model")

        # Compare models
        compare_arcface_parameters(model, fresh_model, "Loaded", "Fresh")

        print(f"\n=== SUMMARY ===")
        print("If you see major differences between loaded and fresh models,")
        print("but small differences between inference/training modes,")
        print("the issue is likely in checkpoint loading or config mismatch.")
        print("\nIf you see large differences between inference/training modes,")
        print("the issue is in the forward pass logic.")
        print(f"\n=== FILES USED ===")
        print(f"Checkpoint: {local_model_path}")
        print(f"CLIP Backbone: {clip_path}")
        print(f"\nTo re-download assets, delete the debug_model folder and run again.")

    except Exception as e:
        print(f"Error during model testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
