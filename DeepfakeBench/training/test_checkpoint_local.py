#!/usr/bin/env python3
"""
test_checkpoint_local.py – Local CPU test for Effort checkpoints.

Downloads a checkpoint from GCS and verifies it loads correctly on CPU,
then optionally runs inference on a test image.

Usage:
    # 1. Verify checkpoint loads (no image needed)
    python test_checkpoint_local.py

    # 2. Run inference on a test image
    python test_checkpoint_local.py --image /path/to/face.jpg

    # 3. Custom checkpoint
    python test_checkpoint_local.py \
        --checkpoint gs://training-job-outputs/best_checkpoints/corrected/top_n_effort_20260119_step14000_auc0.9947_eer0.0218_B16_LAION.patched.pth
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import yaml

# ── Ensure project root is importable ──
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detectors import DETECTOR  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("test-checkpoint-local")

# ── Constants ──
DEFAULT_CHECKPOINT = (
    "gs://training-job-outputs/best_checkpoints/corrected/"
    "top_n_effort_20260119_step14000_auc0.9947_eer0.0218_B16_LAION.patched.pth"
)
LOCAL_WEIGHTS_DIR = Path("./weights/local_test")
DEVICE = torch.device("cpu")

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


# ─────────────────────────────────────────
# Checkpoint download
# ─────────────────────────────────────────
def download_from_gcs(gcs_path: str, local_dir: Path) -> Path:
    """Download a file from GCS using gsutil. Returns the local path."""
    local_dir.mkdir(parents=True, exist_ok=True)
    filename = gcs_path.rsplit("/", 1)[-1]
    local_path = local_dir / filename

    if local_path.exists():
        logger.info(f"Checkpoint already exists locally: {local_path}")
        return local_path

    logger.info(f"Downloading: {gcs_path} → {local_path}")
    t0 = time.time()
    result = subprocess.run(
        ["gsutil", "cp", gcs_path, str(local_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error(f"gsutil failed:\n{result.stderr}")
        raise RuntimeError(f"Failed to download checkpoint from {gcs_path}")

    elapsed = time.time() - t0
    size_mb = local_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Downloaded in {elapsed:.1f}s ({size_mb:.1f} MB)")
    return local_path


# ─────────────────────────────────────────
# Model loading (adapted from app3.py, CPU-safe)
# ─────────────────────────────────────────
def load_detector_cpu(weights_path: str) -> torch.nn.Module:
    """Load an EffortDetector checkpoint on CPU.

    The .patched.pth format embeds model_config, so no external
    config files are needed for model architecture reconstruction.
    """
    logger.info(f"Loading checkpoint: {weights_path}")
    ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=False)

    # ── Parse checkpoint ──
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        model_config = ckpt.get("model_config", {})
        logger.info(f"📋 Checkpoint format: new (with model_config)")
        logger.info(f"   Epoch:  {ckpt.get('epoch')}")
        logger.info(f"   AUC:    {ckpt.get('auc', 'N/A')}")
        logger.info(f"   EER:    {ckpt.get('eer', 'N/A')}")
        logger.info(f"   Step:   {ckpt.get('global_step', 'N/A')}")
    else:
        state_dict = ckpt
        model_config = {}
        logger.warning("⚠️  Old checkpoint format (no model_config). Using defaults.")

    # ── Build config dict from base config + checkpoint overrides ──
    # Load the base detector yaml
    base_cfg_path = Path(__file__).resolve().parent / "config" / "detector" / "effort.yaml"
    if base_cfg_path.exists():
        with open(base_cfg_path) as f:
            cfg = yaml.safe_load(f)
        logger.info(f"Loaded base config: {base_cfg_path}")
    else:
        cfg = {"model_name": "effort"}
        logger.warning(f"Base config not found at {base_cfg_path}, using minimal config")

    # Apply saved model_config overrides
    if model_config:
        logger.info("📋 Applying model_config from checkpoint:")
        for key, value in model_config.items():
            if key == "current_arcface_s":
                continue  # dynamic param, handled separately
            old = cfg.get(key)
            cfg[key] = value
            if old != value:
                logger.info(f"   {key}: {old} → {value}")
    else:
        logger.warning("No model_config in checkpoint — architecture may not match!")

    # ── Print key architecture info ──
    backbone = cfg.get("backbone", {})
    logger.info(f"── Architecture ──")
    logger.info(f"   Backbone source:  {backbone.get('source', 'openai')}")
    logger.info(f"   Backbone variant: {backbone.get('variant', 'N/A')}")
    logger.info(f"   Hidden size:      {backbone.get('hidden_size', cfg.get('hidden_size', 1024))}")
    logger.info(f"   Rank:             {cfg.get('rank', 'N/A')}")
    logger.info(f"   ArcFace head:     {cfg.get('use_arcface_head', False)}")

    # ── Instantiate model ──
    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(DEVICE)

    # Restore ArcFace dynamic param if needed
    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])
            logger.info(f"   Restored ArcFace s: {model_config['current_arcface_s']}")

    # ── Load state dict ──
    state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model.eval()
    logger.info("✅ Model loaded and set to eval mode on CPU")

    # ── Parameter summary ──
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"   Total params:     {total:,}")
    logger.info(f"   Trainable params: {trainable:,}")

    return model


# ─────────────────────────────────────────
# Inference
# ─────────────────────────────────────────
def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
    """Resize + normalize an image for the Effort model (no face detection)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    return _transform(rgb).unsqueeze(0)  # (1, 3, 224, 224)


@torch.inference_mode()
def run_inference(model: torch.nn.Module, image_tensor: torch.Tensor) -> dict:
    """Run a forward pass and return cls logits + probability."""
    preds = model({"image": image_tensor.to(DEVICE)}, inference=True)
    prob = preds["prob"].squeeze().cpu().item()
    cls = preds["cls"].squeeze().cpu().numpy()
    return {"prob": prob, "cls": cls}


# ─────────────────────────────────────────
# Smoke test: random tensor
# ─────────────────────────────────────────
@torch.inference_mode()
def smoke_test(model: torch.nn.Module) -> None:
    """Feed a random 224×224 tensor through the model to verify it runs."""
    logger.info("── Smoke test (random tensor) ──")
    dummy = torch.randn(1, 3, 224, 224, device=DEVICE)
    t0 = time.time()
    preds = model({"image": dummy}, inference=True)
    elapsed = (time.time() - t0) * 1000
    prob = preds["prob"].squeeze().cpu().item()
    cls = preds["cls"].squeeze().cpu().numpy()
    logger.info(f"   cls logits: {cls}")
    logger.info(f"   fake prob:  {prob:.6f}")
    logger.info(f"   latency:    {elapsed:.1f} ms (CPU)")
    logger.info("✅ Smoke test passed")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Local CPU test for Effort checkpoint")
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help="GCS path or local path to .pth checkpoint",
    )
    parser.add_argument(
        "--image", default=None,
        help="Optional: path to a face image for real inference test",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="FAKE/REAL classification threshold (default: 0.5)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip GCS download (use if checkpoint is already local)",
    )
    args = parser.parse_args()

    # ── Resolve checkpoint path ──
    if args.checkpoint.startswith("gs://"):
        if args.skip_download:
            filename = args.checkpoint.rsplit("/", 1)[-1]
            weights_path = str(LOCAL_WEIGHTS_DIR / filename)
        else:
            weights_path = str(download_from_gcs(args.checkpoint, LOCAL_WEIGHTS_DIR))
    else:
        weights_path = args.checkpoint

    if not Path(weights_path).exists():
        logger.error(f"Checkpoint not found: {weights_path}")
        sys.exit(1)

    # ── Load model ──
    model = load_detector_cpu(weights_path)

    # ── Smoke test ──
    smoke_test(model)

    # ── Image inference (optional) ──
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            logger.error(f"Image not found: {img_path}")
            sys.exit(1)

        logger.info(f"── Image inference: {img_path} ──")
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.error("Failed to decode image")
            sys.exit(1)

        tensor = preprocess_image(img_bgr)
        result = run_inference(model, tensor)

        label = "FAKE" if result["prob"] >= args.threshold else "REAL"
        logger.info(f"   cls logits: {result['cls']}")
        logger.info(f"   fake prob:  {result['prob']:.6f}")
        logger.info(f"   label:      {label} (threshold={args.threshold})")
        logger.info(f"   ⚠️  Note: No face detection applied — pass a pre-cropped face for accurate results")

    logger.info("🎉 All tests passed!")


if __name__ == "__main__":
    main()
