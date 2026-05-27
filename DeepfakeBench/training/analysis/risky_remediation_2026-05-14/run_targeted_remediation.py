"""Targeted remediation sweep — test whether ANY remediation rescues T5C
errors on the production-eligible G2-pass pool.

Tests 8 candidate remediations beyond what's already been done:
1. blend@0.35 — milder unsharp (effective amount 0.175)
2. blend@0.65 — stronger unsharp (effective amount 0.325)
3. blur_5 — 5x5 Gaussian low-pass (OPPOSITE of unsharp; tests if T5C overfires
   on residual high-freq content not neutralized by GRL)
4. blur_7 — 7x7 Gaussian low-pass (more aggressive)
5. desat_50 — 50% HSV-S desaturation
6. downup_168 — downscale to 168 then back to 224 (frequency-band low-pass)
7. clahe_mild — CLAHE with clipLimit=1.0 (local contrast, milder than before)
8. tta_3way — mean(orig, blend_050, blur_5) — TTA aggregation

For each: score T5C on the full G2-pass pool, write CSV.

Runtime: ~5 min per remediation × 8 = ~40 min CPU.
"""
from __future__ import annotations

import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arena.model_arena import CLIP_MEAN, CLIP_STD  # noqa: E402
from detectors import DETECTOR  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("targeted")

T5C_CKPT = REPO_ROOT / "analysis/r13_overnight_may6_retest_2026-05-13/_ckpt_cache/periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth"
DETECTOR_CONFIG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO_ROOT / "config/defaults.yaml"
OUT_DIR = THIS_DIR / "outputs"

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
RES = 224


# ============================================================================
# Remediations
# ============================================================================


def make_unsharp(amount):
    """Module-level closure-free factory."""
    class _Sharp:
        def __init__(self, amt): self.amt = amt
        def __call__(self, img):
            blur = cv2.GaussianBlur(img, (5, 5), 1.0)
            return cv2.addWeighted(img, 1.0 + self.amt, blur, -self.amt, 0)
    return _Sharp(amount)


def make_blend(w):
    """Blend of orig + unsharp@0.5 with weight w on the sharpened."""
    class _Blend:
        def __init__(self, weight): self.w = weight
        def __call__(self, img):
            sharp = cv2.addWeighted(img, 1.5, cv2.GaussianBlur(img, (5, 5), 1.0), -0.5, 0)
            return cv2.addWeighted(sharp, self.w, img, 1.0 - self.w, 0)
    return _Blend(w)


class _Blur:
    def __init__(self, k, sigma): self.k = k; self.sigma = sigma
    def __call__(self, img):
        return cv2.GaussianBlur(img, (self.k, self.k), self.sigma)


class _Desat:
    def __init__(self, factor): self.factor = factor
    def __call__(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = hsv[..., 1] * (1.0 - self.factor)
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


class _DownUp:
    def __init__(self, low_res): self.low = low_res
    def __call__(self, img):
        small = cv2.resize(img, (self.low, self.low), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (RES, RES), interpolation=cv2.INTER_LINEAR)


class _CLAHE:
    def __init__(self, clip): self.clip = clip
    def __call__(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip, tileGridSize=(8, 8))
        L = clahe.apply(L)
        return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)


# Note: TTA cannot be expressed as a single transform; we'll compute it
# post-hoc by averaging scores of orig, blend_050, blur_5.

REMEDIATIONS = OrderedDict([
    ("blend_035",   make_blend(0.35)),
    ("blend_065",   make_blend(0.65)),
    ("blur_5",      _Blur(5, 1.0)),
    ("blur_7",      _Blur(7, 1.5)),
    ("desat_50",    _Desat(0.5)),
    ("downup_168",  _DownUp(168)),
    ("clahe_mild",  _CLAHE(1.0)),
])


# ============================================================================
# Dataset + scoring
# ============================================================================


class RemDataset(Dataset):
    def __init__(self, paths, remediation):
        self.paths = paths
        self.remediation = remediation
        self.transform = T.Compose([T.ToTensor(),
                                     T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]), cv2.IMREAD_COLOR)
        if img is None:
            return torch.zeros(3, RES, RES), idx
        img = cv2.resize(img, (RES, RES), interpolation=cv2.INTER_LINEAR)
        img = self.remediation(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), idx


def load_model(ckpt_path):
    import yaml
    with open(DETECTOR_CONFIG) as f: cfg = yaml.safe_load(f)
    with open(TRAIN_CONFIG) as f: cfg.update(yaml.safe_load(f))
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model_cfg = ckpt.get("model_config", {})
    for k, v in model_cfg.items():
        if k != "current_arcface_s": cfg[k] = v
    cfg["multi_axis_grl"] = {"enabled": False}
    model = DETECTOR[cfg["model_name"]](cfg).to(DEVICE)
    if model_cfg.get("use_arcface_head") and "current_arcface_s" in model_cfg:
        if hasattr(model.head, "s"):
            model.head.s.data.fill_(model_cfg["current_arcface_s"])
    clean = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
    missing, unexpected = model.load_state_dict(clean, strict=False)
    log.info(f"  load: {len(missing)} missing, {len(unexpected)} unexpected")
    model.eval()
    return model


def score(model, paths, remediation, name):
    ds = RemDataset(paths, remediation)
    loader = DataLoader(ds, batch_size=8, num_workers=0)
    probs = np.zeros(len(paths), dtype=np.float32)
    t0 = time.time()
    n_batches = 0
    for images, indices in loader:
        with torch.inference_mode():
            out = model({"image": images.to(DEVICE)}, inference=True)["prob"]
            out = out.detach().cpu().numpy().reshape(-1)
        for i, gi in enumerate(indices.numpy()):
            probs[int(gi)] = float(out[i])
        n_batches += 1
        if n_batches % 100 == 0:
            elapsed = time.time() - t0
            log.info(f"    [{name}] {n_batches*8}/{len(paths)} ({elapsed:.0f}s)")
    log.info(f"  [{name}] DONE in {time.time()-t0:.1f}s")
    return probs


def main():
    log.info("loading G2-pass pool...")
    df = pd.read_csv(OUT_DIR / "g2_pass_pool.csv")
    log.info(f"pool: {len(df)} frames")
    paths = df["local"].tolist()

    log.info(f"loading T5C step3500...")
    model = load_model(T5C_CKPT)

    # Run each new remediation
    for name, fn in REMEDIATIONS.items():
        log.info(f"\n=== {name} ===")
        df[f"T5C_{name}"] = score(model, paths, fn, name)

    # TTA: post-hoc average of orig + blend_050 + blur_5
    log.info("\n=== tta_3way (post-hoc) ===")
    df["T5C_tta_3way"] = (df["T5C_orig"] + df["T5C_blend_050"] + df["T5C_blur_5"]) / 3.0

    df.to_csv(OUT_DIR / "targeted_remediation_scored.csv", index=False)
    log.info(f"\nwrote -> {OUT_DIR / 'targeted_remediation_scored.csv'}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
