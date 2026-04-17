#!/usr/bin/env python3
"""
eval_df40_by_method.py — Targeted DF40 evaluation with per-method breakdown.

The meta-analysis sampled 150 DF40 fakes randomly from ALL 29 methods in
the GCS bucket, but only 17 are in the training pair JSON (and only 8 are
face-swap / target_source methods relevant to our use case).

This script:
  1. Downloads N images per DF40 method (preserving method identity)
  2. Downloads paired reals from the training pair JSON
  3. Runs R25_F1 inference on each group
  4. Reports accuracy per method, per orientation, and overall
  5. Computes image properties (sharpness, edge density) per method

This will tell us whether the model truly fails on DF40 face-swaps it was
trained on, or if the meta-analysis numbers were contaminated.

Usage:
    cd DeepfakeBench/training
    conda run -n sweep-env python eval_df40_by_method.py

Output:
    analysis_results/verification/df40_by_method_results.csv
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import yaml
from PIL import Image

# ── Project imports (run from DeepfakeBench/training/) ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "DeepfakeBench" / "training"))
from detectors import DETECTOR  # noqa: E402

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DF40_BUCKET = "df40-frames-recropped-rfa85"
PAIR_JSON = Path(__file__).resolve().parent.parent.parent / \
    "DeepfakeBench/training/dataset/df40_pairs/df40-pair-matching.json"
OUTPUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = OUTPUT_DIR / "df40_method_cache"
WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent / \
    "DeepfakeBench/training/weights/enhancer_eval"

# R25_F1 checkpoint — the deployment model
R25_F1_GCS = ("gs://training-job-outputs/phase2r2_experiments/5w453our/"
              "top_n_effort_20260212_step18500_auc0.9893_eer0.0356.pth")

# How many images to sample per method
SAMPLES_PER_METHOD = 50

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# DF40 method orientations from pair JSON
TARGET_SOURCE_METHODS = [  # Face swaps — our use case
    'blendface', 'e4s', 'facedancer', 'faceswap',
    'inswap', 'mobileswap', 'simswap', 'uniface',
]
SOURCE_TARGET_METHODS = [  # Reenactment — less relevant
    'MRAA', 'danet', 'facevid2vid', 'fomm', 'fsgan',
    'lia', 'mcnet', 'one_shot_free', 'pirender',
]
# Extra methods in bucket but NOT in pair JSON (never trained on)
EXTRA_METHODS_NOT_IN_TRAINING = [
    'DiT', 'RDDM', 'SiT', 'StyleGAN2', 'StyleGAN3', 'StyleGANXL',
    'VQGAN', 'ddim', 'hyperreenact', 'sadtalker', 'tpsm', 'wav2lip',
]

# ─────────────────────────────────────────
# Logging
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
)
logger = logging.getLogger("df40-eval")

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


def preprocess_numpy(img_rgb: np.ndarray) -> torch.Tensor:
    """Resize to 224x224, normalize with CLIP stats."""
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    return _transform(Image.fromarray(img_resized))


# ─────────────────────────────────────────
# Image properties
# ─────────────────────────────────────────
def compute_properties(img_rgb: np.ndarray) -> dict:
    """Compute key image properties for a single image."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Sharpness (Laplacian variance)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness_var = float(lap.var())

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    # Noise estimate (high-pass filter std)
    blur = cv2.GaussianBlur(gray.astype(np.float64), (5, 5), 0)
    noise = float(np.std(gray.astype(np.float64) - blur))

    return {
        'width': w,
        'height': h,
        'sharpness_laplacian_var': sharpness_var,
        'edge_density': edge_density,
        'noise_estimate': noise,
    }


# ─────────────────────────────────────────
# GCS helpers
# ─────────────────────────────────────────
def _gcs_client():
    from google.cloud import storage
    return storage.Client()


def download_method_images(method: str, label: str, n: int) -> List[Tuple[str, np.ndarray]]:
    """Download n images from a specific DF40 method, preserving method info."""
    cache_dir = CACHE_DIR / f"{label}_{method}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    cached = sorted([f for f in cache_dir.iterdir() if f.suffix in IMG_EXTS])
    if len(cached) >= n:
        logger.info(f"  Using {n} cached images for {label}/{method}")
        results = []
        for p in cached[:n]:
            img = np.array(Image.open(p).convert("RGB"))
            results.append((f"{method}/{p.stem}", img))
        return results

    logger.info(f"  Downloading {n} images from gs://{DF40_BUCKET}/{label}/{method}/")
    client = _gcs_client()
    bucket = client.bucket(DF40_BUCKET)

    prefix = f"{label}/{method}/"
    blobs = list(bucket.list_blobs(prefix=prefix, max_results=5000))
    image_blobs = [b for b in blobs
                   if b.name.lower().endswith(('.png', '.jpg', '.jpeg'))
                   and not b.name.endswith("/")]

    if not image_blobs:
        logger.warning(f"  No images found for {label}/{method}")
        return []

    rng = np.random.RandomState(42)
    indices = rng.choice(len(image_blobs), size=min(n, len(image_blobs)), replace=False)

    results = []
    for idx in indices:
        fb = image_blobs[idx]
        try:
            img_bytes = fb.download_as_bytes()
            img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

            # Cache with method info preserved
            safe_name = fb.name.replace("/", "__")
            Image.fromarray(img).save(cache_dir / f"{safe_name}.png")

            results.append((f"{method}/{fb.name.split('/')[-1]}", img))
        except Exception as e:
            logger.warning(f"  Error: {fb.name}: {e}")

    logger.info(f"  Got {len(results)} images for {label}/{method}")
    return results


def download_paired_reals(n_per_source: int) -> List[Tuple[str, np.ndarray]]:
    """Download reals from the pair JSON — these are the actual training reals."""
    cache_dir = CACHE_DIR / "paired_reals"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    cached = sorted([f for f in cache_dir.iterdir() if f.suffix in IMG_EXTS])
    if len(cached) >= n_per_source:
        logger.info(f"  Using {n_per_source} cached paired reals")
        results = []
        for p in cached[:n_per_source]:
            img = np.array(Image.open(p).convert("RGB"))
            results.append((f"paired_real/{p.stem}", img))
        return results

    # Load pair JSON to get real paths
    with open(PAIR_JSON) as f:
        pair_data = json.load(f)

    # Collect unique real identities and their GCS paths
    real_entries = {}
    for p in pair_data['pairs']:
        identity = p['real']['identity']
        if identity not in real_entries:
            real_entries[identity] = {
                'path': p['real']['path'],
                'frames': p['real']['frames'],
                'source': p['real']['source'],
            }

    logger.info(f"  Found {len(real_entries)} unique real identities in pair JSON")

    # Sample identities, then pick one frame per identity
    rng = np.random.RandomState(42)
    identity_keys = sorted(real_entries.keys())
    sample_ids = rng.choice(identity_keys,
                            size=min(n_per_source, len(identity_keys)),
                            replace=False)

    client = _gcs_client()
    bucket = client.bucket(DF40_BUCKET)

    results = []
    for identity in sample_ids:
        entry = real_entries[identity]
        frame = rng.choice(entry['frames'])
        # Path format: gs://df40-frames-recropped-rfa85/real/FaceForensics++/001/
        gcs_prefix = entry['path'].replace(f"gs://{DF40_BUCKET}/", "")
        blob_name = f"{gcs_prefix}{frame}"

        try:
            blob = bucket.blob(blob_name)
            img_bytes = blob.download_as_bytes()
            img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

            safe_name = blob_name.replace("/", "__")
            Image.fromarray(img).save(cache_dir / f"{safe_name}.png")

            results.append((f"paired_real/{identity}_{frame}", img))
        except Exception as e:
            logger.warning(f"  Error downloading real {blob_name}: {e}")

    logger.info(f"  Got {len(results)} paired reals")
    return results


# ─────────────────────────────────────────
# Model loading & inference
# ─────────────────────────────────────────
def download_checkpoint(gcs_path: str) -> Path:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = gcs_path.rsplit("/", 1)[-1]
    local_path = WEIGHTS_DIR / filename
    if local_path.exists():
        return local_path
    logger.info(f"  Downloading checkpoint...")
    result = subprocess.run(["gsutil", "cp", gcs_path, str(local_path)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gsutil failed: {result.stderr}")
    return local_path


def load_model(weights_path: Path) -> torch.nn.Module:
    ckpt = torch.load(str(weights_path), map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        model_config = ckpt.get("model_config", {})
    else:
        state_dict = ckpt
        model_config = {}

    base_cfg_path = (Path(__file__).resolve().parent.parent.parent /
                     "DeepfakeBench/training/config/detector/effort.yaml")
    with open(base_cfg_path) as f:
        cfg = yaml.safe_load(f)

    if model_config:
        for key, value in model_config.items():
            if key != "current_arcface_s":
                cfg[key] = value

    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(DEVICE)

    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])

    state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.inference_mode()
def run_batch(model: torch.nn.Module, tensors: torch.Tensor) -> np.ndarray:
    data = {
        "image": tensors.to(DEVICE),
        "label": torch.zeros(tensors.size(0), dtype=torch.long, device=DEVICE),
    }
    preds = model(data, inference=True)
    return preds["prob"].squeeze(-1).cpu().numpy()


def infer_images(model: torch.nn.Module,
                 images: List[Tuple[str, np.ndarray]],
                 batch_size: int = 32) -> List[dict]:
    """Run inference + compute properties for each image."""
    results = []
    n = len(images)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = images[start:end]

        tensors = []
        batch_props = []
        for name, img in batch:
            tensors.append(preprocess_numpy(img))
            props = compute_properties(img)
            props['name'] = name
            batch_props.append(props)

        batch_tensor = torch.stack(tensors)
        probs = run_batch(model, batch_tensor)

        for props, prob in zip(batch_props, probs):
            props['model_fake_prob'] = float(prob)
            results.append(props)

    return results


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    t0 = time.time()
    logger.info(f"Device: {DEVICE}")

    # 1. Load model
    logger.info("\n=== Loading R25_F1 model ===")
    ckpt_path = download_checkpoint(R25_F1_GCS)
    model = load_model(ckpt_path)
    logger.info("  Model loaded successfully")

    all_results = []

    # 2. Evaluate face-swap methods (target_source — our use case)
    logger.info("\n=== Evaluating TARGET_SOURCE (face swap) methods ===")
    for method in TARGET_SOURCE_METHODS:
        logger.info(f"\n--- {method} ---")
        images = download_method_images(method, "fake", SAMPLES_PER_METHOD)
        if images:
            results = infer_images(model, images)
            for r in results:
                r['method'] = method
                r['orientation'] = 'target_source'
                r['label'] = 'fake'
                r['in_training'] = True
            all_results.extend(results)

    # 3. Evaluate reenactment methods (source_target — less relevant)
    logger.info("\n=== Evaluating SOURCE_TARGET (reenactment) methods ===")
    for method in SOURCE_TARGET_METHODS:
        logger.info(f"\n--- {method} ---")
        images = download_method_images(method, "fake", SAMPLES_PER_METHOD)
        if images:
            results = infer_images(model, images)
            for r in results:
                r['method'] = method
                r['orientation'] = 'source_target'
                r['label'] = 'fake'
                r['in_training'] = True
            all_results.extend(results)

    # 4. Evaluate a few EXTRA methods (not in training — as control)
    logger.info("\n=== Evaluating EXTRA methods (NOT in training, control) ===")
    control_methods = ['StyleGAN2', 'ddim', 'wav2lip']  # 3 diverse unseen methods
    for method in control_methods:
        logger.info(f"\n--- {method} (CONTROL - not in training) ---")
        images = download_method_images(method, "fake", SAMPLES_PER_METHOD)
        if images:
            results = infer_images(model, images)
            for r in results:
                r['method'] = method
                r['orientation'] = 'extra'
                r['label'] = 'fake'
                r['in_training'] = False
            all_results.extend(results)

    # 5. Evaluate paired reals
    logger.info("\n=== Evaluating PAIRED REALS (from training pair JSON) ===")
    real_images = download_paired_reals(150)
    if real_images:
        results = infer_images(model, real_images)
        for r in results:
            r['method'] = 'FaceForensics++_real'
            r['orientation'] = 'real'
            r['label'] = 'real'
            r['in_training'] = True
        all_results.extend(results)

    # 6. Save raw results
    df = pd.DataFrame(all_results)
    output_path = OUTPUT_DIR / "df40_by_method_results.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"\n✅ Saved {len(df)} rows to {output_path}")

    # 7. Print summary
    print("\n" + "=" * 80)
    print("DF40 PER-METHOD EVALUATION — R25_F1")
    print("=" * 80)

    print("\n--- FAKE methods (in training) ---")
    print(f"{'Method':<22} {'Orient':<16} {'N':>4}  {'Acc@0.5':>7}  "
          f"{'MeanProb':>8}  {'Sharpness':>10}  {'EdgeDens':>8}  {'Noise':>6}")
    print("-" * 100)

    for orient in ['target_source', 'source_target', 'extra']:
        subset = df[df['orientation'] == orient]
        if subset.empty:
            continue
        for method in sorted(subset['method'].unique()):
            mdf = subset[subset['method'] == method]
            n = len(mdf)
            acc = (mdf['model_fake_prob'] > 0.5).mean() * 100
            mean_prob = mdf['model_fake_prob'].mean()
            sharp = mdf['sharpness_laplacian_var'].mean()
            edge = mdf['edge_density'].mean()
            noise = mdf['noise_estimate'].mean()
            marker = "" if orient != 'extra' else " ⚠️ NOT IN TRAINING"
            print(f"{method:<22} {orient:<16} {n:>4}  {acc:>6.1f}%  "
                  f"{mean_prob:>8.3f}  {sharp:>10.1f}  {edge:>8.4f}  {noise:>6.3f}{marker}")

        # Group subtotal
        n = len(subset)
        acc = (subset['model_fake_prob'] > 0.5).mean() * 100
        mean_prob = subset['model_fake_prob'].mean()
        sharp = subset['sharpness_laplacian_var'].mean()
        orient_label = {'target_source': 'FACE SWAP TOTAL',
                        'source_target': 'REENACT TOTAL',
                        'extra': 'EXTRA TOTAL'}[orient]
        print(f"{'>>> ' + orient_label:<22} {'':>16} {n:>4}  {acc:>6.1f}%  "
              f"{mean_prob:>8.3f}  {sharp:>10.1f}")
        print()

    # Reals
    reals = df[df['label'] == 'real']
    if not reals.empty:
        print("\n--- REAL (paired, from training pair JSON) ---")
        n = len(reals)
        correct = (reals['model_fake_prob'] < 0.5).mean() * 100
        fpr = (reals['model_fake_prob'] > 0.5).mean() * 100
        mean_prob = reals['model_fake_prob'].mean()
        sharp = reals['sharpness_laplacian_var'].mean()
        edge = reals['edge_density'].mean()
        noise = reals['noise_estimate'].mean()
        print(f"  N={n}, Correct (TNR)={correct:.1f}%, FPR={fpr:.1f}%, "
              f"MeanProb={mean_prob:.3f}")
        print(f"  Sharpness={sharp:.1f}, EdgeDensity={edge:.4f}, Noise={noise:.3f}")

        # Real prob distribution
        bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
        print(f"\n  Real prob distribution:")
        for lo, hi in bins:
            count = ((reals['model_fake_prob'] >= lo) & (reals['model_fake_prob'] < hi)).sum()
            pct = count / n * 100
            bar = "█" * int(pct / 2)
            print(f"    [{lo:.1f}-{hi:.1f}): {count:>4} ({pct:>5.1f}%) {bar}")

    # 8. Key question answered
    print("\n" + "=" * 80)
    print("KEY QUESTION: Does the model fail on DF40 face-swaps it was trained on?")
    print("=" * 80)

    ts = df[df['orientation'] == 'target_source']
    st = df[df['orientation'] == 'source_target']

    if not ts.empty:
        ts_acc = (ts['model_fake_prob'] > 0.5).mean() * 100
        ts_sharp = ts['sharpness_laplacian_var'].mean()
        print(f"\n  Face-swap (target_source) accuracy: {ts_acc:.1f}%  (sharpness: {ts_sharp:.1f})")

    if not st.empty:
        st_acc = (st['model_fake_prob'] > 0.5).mean() * 100
        st_sharp = st['sharpness_laplacian_var'].mean()
        print(f"  Reenactment (source_target) accuracy: {st_acc:.1f}%  (sharpness: {st_sharp:.1f})")

    if not reals.empty:
        real_tnr = (reals['model_fake_prob'] < 0.5).mean() * 100
        real_sharp = reals['sharpness_laplacian_var'].mean()
        print(f"  Paired reals TNR (correct): {real_tnr:.1f}%  (sharpness: {real_sharp:.1f})")

    # Sharpness correlation within DF40
    fakes = df[df['label'] == 'fake']
    if len(fakes) > 10:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(fakes['sharpness_laplacian_var'], fakes['model_fake_prob'])
        print(f"\n  Sharpness ↔ fake_prob Spearman ρ (DF40 fakes): {rho:+.3f} (p={pval:.2e})")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
