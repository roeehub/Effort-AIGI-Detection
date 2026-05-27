"""
P22 pre-validation — Laplacian gap closure simulation.

Loads ~200 training-bucket frames (mix of realpool + visomaster + enhanced).
Applies augmentation sweeps:
  - Gaussian blur σ ∈ {0, 1, 2, 3, 4, 5, 6, 8}
  - JPEG quality q ∈ {30, 50, 70, 90, 100}
  - Brightness shift β ∈ {-60, -40, -20, 0, +20, +40}

For each combination (or marginal), measures post-aug Laplacian variance, luminance, skin_frac.

Compares to the eval-side empirical distribution (viso, lockbox) loaded from
cross_suite_attributes.csv.

Critical falsifier: if no aug param brings post-aug Laplacian into eval range,
the P22 hypothesis is dead.
"""

import io
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

TRAIN_ROOT = Path("analysis/score_distribution_2026-05-02/outputs/train_data_samples")
EVAL_ATTRS = Path("analysis/score_distribution_2026-05-02/outputs/cross_suite_attributes.csv")
OUT = Path("analysis/cpu_decision_2026-05-02_pm_late/outputs")
FIG = Path("analysis/cpu_decision_2026-05-02_pm_late/figures")
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# Load training frames (mix categories)
imgs = []
sources = []
for cat_dir in sorted(TRAIN_ROOT.iterdir()):
    if not cat_dir.is_dir():
        continue
    files = sorted(cat_dir.glob("*.png"))[:30]  # cap per category
    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            imgs.append(img)
            sources.append(cat_dir.name)
print(f"loaded {len(imgs)} training frames across {len(set(sources))} categories")


def laplacian_var(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def luma_mean(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(g.mean())


def apply_blur(img, sigma):
    if sigma <= 0:
        return img.copy()
    k = int(2 * np.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


def apply_jpeg(img, q):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(q))
    buf.seek(0)
    arr = np.array(Image.open(buf))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def apply_brightness(img, beta):
    out = img.astype(np.int32) + int(beta)
    return np.clip(out, 0, 255).astype(np.uint8)


# Sweep — marginal effects
results = []

# Baseline (no aug)
for img, src in zip(imgs, sources):
    results.append({"aug": "none", "param": 0, "src": src, "lap": laplacian_var(img), "luma": luma_mean(img)})

# Blur sweep
for sigma in [1, 2, 3, 4, 5, 6, 8]:
    for img, src in zip(imgs, sources):
        out_img = apply_blur(img, sigma)
        results.append({"aug": "blur", "param": sigma, "src": src, "lap": laplacian_var(out_img), "luma": luma_mean(out_img)})

# JPEG sweep
for q in [30, 50, 70, 90]:
    for img, src in zip(imgs, sources):
        out_img = apply_jpeg(img, q)
        results.append({"aug": "jpeg", "param": q, "src": src, "lap": laplacian_var(out_img), "luma": luma_mean(out_img)})

# Brightness sweep (no Laplacian effect; for luma-gap diagnostic)
for beta in [-60, -40, -20, 20, 40]:
    for img, src in zip(imgs, sources):
        out_img = apply_brightness(img, beta)
        results.append({"aug": "brightness", "param": beta, "src": src, "lap": laplacian_var(out_img), "luma": luma_mean(out_img)})

# Combined: blur + jpeg (target eval distribution)
for sigma in [2, 3, 4]:
    for q in [50, 70]:
        for img, src in zip(imgs, sources):
            out_img = apply_jpeg(apply_blur(img, sigma), q)
            results.append({"aug": f"blur+jpeg", "param": f"σ={sigma},q={q}", "src": src, "lap": laplacian_var(out_img), "luma": luma_mean(out_img)})

aug_df = pd.DataFrame(results)
aug_df.to_csv(OUT / "p22_aug_sweep_per_frame.csv", index=False)

# Summary by (aug, param)
summary = aug_df.groupby(["aug", "param"]).agg(
    lap_mean=("lap", "mean"),
    lap_p25=("lap", lambda s: s.quantile(0.25)),
    lap_p50=("lap", "median"),
    lap_p75=("lap", lambda s: s.quantile(0.75)),
    luma_mean=("luma", "mean"),
    n=("lap", "count"),
).reset_index()
summary.to_csv(OUT / "p22_aug_sweep_summary.csv", index=False)
print("\nAugmentation effect on Laplacian variance:")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

# Eval reference distribution
eval_df = pd.read_csv(EVAL_ATTRS)
print("\nEval-side empirical Laplacian distribution (target zone):")
for s in eval_df.suite.unique():
    sub = eval_df[eval_df.suite == s]
    print(f"  {s:35s} n={len(sub):3d}  lap_mean={sub.laplacian_var.mean():6.1f}  p25={sub.laplacian_var.quantile(0.25):6.1f}  p50={sub.laplacian_var.median():6.1f}  p75={sub.laplacian_var.quantile(0.75):6.1f}")

# Critical match: training distribution lap mean (no aug) ≈ 130
# Eval target: viso ≈ 60, lockbox ≈ 17
# Find which aug brings the median into [40, 80] (viso target band)

target_low, target_high = 40, 80
matching = summary[(summary.lap_p50 >= target_low) & (summary.lap_p50 <= target_high)]
print(f"\nAug params that put median Laplacian into viso target band [{target_low}, {target_high}]:")
print(matching.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

target_low2, target_high2 = 5, 30
matching2 = summary[(summary.lap_p50 >= target_low2) & (summary.lap_p50 <= target_high2)]
print(f"\nAug params that put median Laplacian into lockbox-extreme target band [{target_low2}, {target_high2}]:")
print(matching2.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

# Plot: blur sigma sweep — Laplacian distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax = axes[0]
for sigma in [0, 1, 2, 3, 4, 5, 6, 8]:
    if sigma == 0:
        sub = aug_df[aug_df.aug == "none"]
    else:
        sub = aug_df[(aug_df.aug == "blur") & (aug_df.param == sigma)]
    ax.hist(sub.lap, bins=40, alpha=0.45, label=f"σ={sigma}", density=True)
ax.set_xlabel("Laplacian variance")
ax.set_ylabel("density")
ax.set_xlim(0, 400)
ax.set_title("Training frames after Gaussian blur — Laplacian shift")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)

# Overlay eval distributions
viso = eval_df[eval_df.suite == "teams_fake_all_lockbox"].laplacian_var
ax.axvline(viso.median(), color="red", linestyle="--", linewidth=2, label="lockbox median")
ax.axvline(eval_df[eval_df.suite == "teams_fake_all_dev"].laplacian_var.median(), color="orange", linestyle="--", linewidth=2)

ax2 = axes[1]
labels = []
data = []
for sigma in [0, 2, 4, 6]:
    src = aug_df[aug_df.aug == "none"] if sigma == 0 else aug_df[(aug_df.aug == "blur") & (aug_df.param == sigma)]
    data.append(src.lap.values)
    labels.append(f"blur σ={sigma}\nn={len(src)}")
for s in ["teams_fake_all_dev", "deeplive_enhanced_dev", "teams_fake_all_lockbox"]:
    sub = eval_df[eval_df.suite == s]
    data.append(sub.laplacian_var.values)
    labels.append(f"eval:\n{s.split('_')[-2]}\nn={len(sub)}")
ax2.boxplot(data, labels=labels, showfliers=False)
ax2.set_ylabel("Laplacian variance")
ax2.set_title("Train+aug vs eval — Laplacian gap closure")
ax2.grid(alpha=0.3)
plt.setp(ax2.get_xticklabels(), rotation=20, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "p22_aug_lap_distribution.png", dpi=140, bbox_inches="tight")
print(f"\nwrote {FIG / 'p22_aug_lap_distribution.png'}")
