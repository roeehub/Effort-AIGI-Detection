#!/usr/bin/env python3
"""
V1 — Sharpness Causality Test (Intervention Experiment)
========================================================
The strongest possible test: if we can flip predictions by changing sharpness
alone, the shortcut is proven as causal (not just correlated).

Experiment:
  A) WMA images → SHARPEN (unsharp mask) → re-infer with R25_F1
  B) WMA images → ORIGINAL (baseline, should match ~21% accuracy)
  C) WMA images → BLUR (Gaussian) → re-infer (should be even worse)

If sharpening pushes accuracy from 21% toward 80%+, the model is causally
relying on sharpness.

Requires:
  - WMA images at /Users/roeedar/Downloads/wma_export/all_images/
  - R25_F1 checkpoint (auto-downloaded from GCS)

Usage:
    cd DeepfakeBench/training
    python ../../analysis_results/verification/V1_sharpness_causality.py

Output:
    analysis_results/verification/V1_causality_results.csv
    analysis_results/verification/V1_causality_summary.txt
"""

from __future__ import annotations

import csv
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image as pil_image

# ── Project imports (need to be on path) ──
TRAINING_DIR = Path(__file__).resolve().parent.parent.parent / "DeepfakeBench" / "training"
sys.path.insert(0, str(TRAINING_DIR))
from detectors import DETECTOR  # noqa: E402

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
IMAGE_DIR = Path("/Users/roeedar/Downloads/wma_export/all_images")
OUTPUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = TRAINING_DIR / "weights" / "enhancer_eval"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

R25_F1_GCS = ("gs://training-job-outputs/phase2r2_experiments/5w453our/"
              "top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth")

# Sharpening/blurring parameters to sweep
SHARPEN_AMOUNTS = [1.0, 2.0, 3.0, 5.0]  # unsharp mask gain
BLUR_SIGMAS = [1.0, 2.0, 3.0]            # Gaussian blur sigma

BATCH_SIZE = 32

# ─────────────────────────────────────────
# Logging
# ─────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("V1-causality")

# ─────────────────────────────────────────
# Device
# ─────────────────────────────────────────
def _resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = _resolve_device()

# ─────────────────────────────────────────
# CLIP preprocessing
# ─────────────────────────────────────────
_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711]),
])


def compute_sharpness(img_gray: np.ndarray) -> float:
    """Laplacian variance — our key metric."""
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()


def apply_sharpen(img_rgb: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp mask sharpening: sharpen = original + amount * (original - blurred)."""
    blurred = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(img_rgb, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_blur(img_rgb: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur."""
    return cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=sigma)


def load_and_process(img_path: Path, mode: str, param: float = 0.0
                     ) -> Tuple[torch.Tensor, float]:
    """Load image, optionally sharpen/blur, resize to 224×224, return tensor + sharpness.
    
    mode: 'original', 'sharpen', 'blur'
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Apply manipulation BEFORE resize (at native resolution)
    if mode == "sharpen":
        img_rgb = apply_sharpen(img_rgb, param)
    elif mode == "blur":
        img_rgb = apply_blur(img_rgb, param)

    # Compute sharpness AFTER manipulation, BEFORE resize
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sharpness = compute_sharpness(img_gray)

    # Resize to 224×224 for model
    img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Also compute sharpness after resize (what the model "sees")
    img_gray_resized = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sharpness_resized = compute_sharpness(img_gray_resized)

    tensor = _transform(pil_image.fromarray(img_rgb))
    return tensor, sharpness, sharpness_resized


# ─────────────────────────────────────────
# Model loading (same as eval_enhancer_local.py)
# ─────────────────────────────────────────
def download_checkpoint(gcs_path: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = gcs_path.rsplit("/", 1)[-1]
    local_path = CACHE_DIR / filename
    if local_path.exists():
        logger.info(f"Cached: {local_path.name}")
        return local_path
    logger.info(f"Downloading: {gcs_path}")
    result = subprocess.run(["gsutil", "cp", gcs_path, str(local_path)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gsutil failed: {result.stderr}")
    return local_path


def load_model(weights_path: Path) -> torch.nn.Module:
    ckpt = torch.load(str(weights_path), map_location=DEVICE, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model_config = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}

    cfg_path = TRAINING_DIR / "config" / "detector" / "effort.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    for k, v in model_config.items():
        if k != "current_arcface_s":
            cfg[k] = v

    model = DETECTOR[cfg["model_name"]](cfg).to(DEVICE)
    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])

    state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.inference_mode()
def run_batch(model, tensors):
    data = {"image": tensors.to(DEVICE),
            "label": torch.zeros(tensors.size(0), dtype=torch.long, device=DEVICE)}
    preds = model(data, inference=True)
    return preds["prob"].squeeze(-1).cpu().numpy()


# ─────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────
def run_condition(model, image_paths, mode, param, max_images=None):
    """Run one experimental condition. Returns list of (filename, prob, sharpness_native, sharpness_224)."""
    paths = image_paths[:max_images] if max_images else image_paths
    results = []
    tensors = []
    metas = []

    for p in paths:
        try:
            t, sharp_native, sharp_224 = load_and_process(p, mode, param)
            tensors.append(t)
            metas.append((p.name, sharp_native, sharp_224))
        except Exception as e:
            logger.warning(f"Skipping {p.name}: {e}")

    # Batch inference
    all_probs = []
    for i in range(0, len(tensors), BATCH_SIZE):
        batch = torch.stack(tensors[i:i + BATCH_SIZE])
        probs = run_batch(model, batch)
        all_probs.extend(probs.tolist())

    for (fname, sharp_n, sharp_224), prob in zip(metas, all_probs):
        results.append((fname, prob, sharp_n, sharp_224))

    return results


def main():
    # ── Collect images ──────────────────────────────────────────────
    image_paths = sorted([p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])
    logger.info(f"Found {len(image_paths)} WMA images")

    if not image_paths:
        logger.error("No images found!")
        return

    # ── Load model ──────────────────────────────────────────────────
    ckpt_path = download_checkpoint(R25_F1_GCS)
    model = load_model(ckpt_path)
    logger.info("R25_F1 model loaded\n")

    # ── Define conditions ───────────────────────────────────────────
    conditions = []

    # Baseline
    conditions.append(("original", "original", 0.0))

    # Sharpening
    for amt in SHARPEN_AMOUNTS:
        conditions.append((f"sharpen_{amt:.1f}", "sharpen", amt))

    # Blurring
    for sig in BLUR_SIGMAS:
        conditions.append((f"blur_{sig:.1f}", "blur", sig))

    # ── Run all conditions ──────────────────────────────────────────
    all_rows = []   # per-image detail
    summary = []    # per-condition summary

    for cond_name, mode, param in conditions:
        logger.info(f"Running condition: {cond_name}")
        t0 = time.time()
        results = run_condition(model, image_paths, mode, param)
        elapsed = time.time() - t0
        logger.info(f"  Done in {elapsed:.1f}s ({len(results)/elapsed:.1f} img/s)")

        probs = np.array([r[1] for r in results])
        sharpness_native = np.array([r[2] for r in results])
        sharpness_224 = np.array([r[3] for r in results])

        acc_50 = np.mean(probs > 0.5)
        acc_70 = np.mean(probs > 0.7)
        mean_prob = np.mean(probs)
        median_prob = np.median(probs)
        mean_sharp_native = np.mean(sharpness_native)
        mean_sharp_224 = np.mean(sharpness_224)

        logger.info(f"  Accuracy @0.5: {100*acc_50:.1f}%  @0.7: {100*acc_70:.1f}%")
        logger.info(f"  Mean prob: {mean_prob:.4f}  Median: {median_prob:.4f}")
        logger.info(f"  Sharpness (native): {mean_sharp_native:.1f}  (at 224): {mean_sharp_224:.1f}")

        summary.append({
            "condition": cond_name,
            "mode": mode,
            "param": param,
            "n_images": len(results),
            "accuracy_0.5": f"{100*acc_50:.1f}%",
            "accuracy_0.7": f"{100*acc_70:.1f}%",
            "mean_prob": f"{mean_prob:.4f}",
            "median_prob": f"{median_prob:.4f}",
            "mean_sharpness_native": f"{mean_sharp_native:.1f}",
            "mean_sharpness_224": f"{mean_sharp_224:.1f}",
            "pct_above_0.9": f"{100*np.mean(probs > 0.9):.1f}%",
            "pct_below_0.3": f"{100*np.mean(probs < 0.3):.1f}%",
        })

        for fname, prob, s_n, s_224 in results:
            all_rows.append({
                "condition": cond_name,
                "filename": fname,
                "fake_prob": prob,
                "sharpness_native": s_n,
                "sharpness_224": s_224,
            })

    # ── Save per-image CSV ──────────────────────────────────────────
    detail_path = OUTPUT_DIR / "V1_causality_results.csv"
    pd_available = True
    try:
        import pandas as pd
        pd.DataFrame(all_rows).to_csv(detail_path, index=False)
    except ImportError:
        pd_available = False
        with open(detail_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)

    logger.info(f"\n✅ Per-image results: {detail_path}")

    # ── Print and save summary ──────────────────────────────────────
    summary_lines = []
    def log(msg):
        print(msg)
        summary_lines.append(msg)

    log("\n" + "=" * 90)
    log("V1 SHARPNESS CAUSALITY TEST — RESULTS")
    log("=" * 90)
    log(f"{'Condition':<18s} {'Acc@0.5':>8s} {'Acc@0.7':>8s} {'MeanProb':>9s} "
        f"{'MedianP':>8s} {'Sharp(nat)':>11s} {'Sharp(224)':>11s} {'<0.3':>6s} {'>0.9':>6s}")
    log("-" * 90)

    for s in summary:
        log(f"{s['condition']:<18s} {s['accuracy_0.5']:>8s} {s['accuracy_0.7']:>8s} "
            f"{s['mean_prob']:>9s} {s['median_prob']:>8s} "
            f"{s['mean_sharpness_native']:>11s} {s['mean_sharpness_224']:>11s} "
            f"{s['pct_below_0.3']:>6s} {s['pct_above_0.9']:>6s}")

    # ── Verdict ─────────────────────────────────────────────────────
    log("\n" + "=" * 90)
    log("VERDICT")
    log("=" * 90)

    baseline = next(s for s in summary if s["condition"] == "original")
    best_sharpen = max([s for s in summary if s["mode"] == "sharpen"],
                       key=lambda s: float(s["mean_prob"]))
    worst_blur = min([s for s in summary if s["mode"] == "blur"],
                     key=lambda s: float(s["mean_prob"]))

    baseline_acc = float(baseline["accuracy_0.5"].rstrip("%"))
    sharpen_acc = float(best_sharpen["accuracy_0.5"].rstrip("%"))
    blur_acc = float(worst_blur["accuracy_0.5"].rstrip("%"))

    delta_sharpen = sharpen_acc - baseline_acc
    delta_blur = blur_acc - baseline_acc

    log(f"  Baseline (original):     {baseline_acc:.1f}% accuracy")
    log(f"  Best sharpen ({best_sharpen['condition']}): {sharpen_acc:.1f}% accuracy (Δ = {delta_sharpen:+.1f}pp)")
    log(f"  Worst blur ({worst_blur['condition']}):   {blur_acc:.1f}% accuracy (Δ = {delta_blur:+.1f}pp)")

    if delta_sharpen > 20:
        log(f"\n✅ CAUSAL CONFIRMATION: Sharpening increased accuracy by {delta_sharpen:+.1f}pp")
        log("   The model's predictions are causally driven by image sharpness.")
        log("   This is NOT just a correlation — changing sharpness changes the decision.")
    elif delta_sharpen > 5:
        log(f"\n⚠️  PARTIAL: Sharpening helped ({delta_sharpen:+.1f}pp) but not dramatically.")
        log("   Sharpness is one factor but not the sole driver.")
    else:
        log(f"\n❌ NOT CONFIRMED: Sharpening had minimal effect ({delta_sharpen:+.1f}pp).")
        log("   The model may be using other features, not primarily sharpness.")

    # Save summary
    summary_path = OUTPUT_DIR / "V1_causality_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    log(f"\n✅ Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
