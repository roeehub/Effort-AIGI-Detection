"""Distributional figures for the top discriminating axes."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/xinhe_cross_camera_audit_2026-05-06"
)
OUT = ROOT / "outputs"

df = pd.read_csv(OUT / "per_frame_features.csv")
m6 = df[df["population"] == "may6_falseflag"]
m5 = df[df["population"] == "may5_correct"]

# Top axes from the comparison table
axes = [
    "sat_std",
    "sobel_mean_face",
    "lap_var_face",
    "hf_ratio_face",
    "luma_std",
    "r_std",
    "g_std",
    "b_std",
    "face_area",
    "width",
    "luma_mean",
    "deploy_score",
]

fig, axarr = plt.subplots(3, 4, figsize=(18, 11))
axarr = axarr.flatten()
for i, ax_name in enumerate(axes):
    ax = axarr[i]
    a = m6[ax_name].dropna().values
    b = m5[ax_name].dropna().values if ax_name in m5.columns else np.array([])
    bins = 30
    if len(a):
        ax.hist(a, bins=bins, alpha=0.6, label=f"may6_falseflag (n={len(a)})", color="C3")
    if len(b):
        ax.hist(b, bins=bins, alpha=0.6, label=f"may5_correct (n={len(b)})", color="C2")
    ax.set_title(ax_name)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle("Xinhe cross-camera audit — distributions per IQ axis (may6 false-flag vs may5 correct)", y=0.995)
plt.tight_layout()
plt.savefig(OUT / "fig_distributions.png", dpi=120)
print(f"wrote {OUT / 'fig_distributions.png'}")

# scatter: deploy_score vs top correlated axes (within may6)
fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4.5))
for ax, name in zip(axes2, ["r_mean", "lap_var_face", "r_std", "luma_mean"]):
    x = m6[name].astype(float).values
    y = m6["deploy_score"].astype(float).values
    mask = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[mask], y[mask], alpha=0.6, s=20)
    if mask.sum() > 2:
        z = np.polyfit(x[mask], y[mask], 1)
        xs = np.linspace(x[mask].min(), x[mask].max(), 50)
        ax.plot(xs, np.polyval(z, xs), "r--", alpha=0.7)
    r = np.corrcoef(x[mask], y[mask])[0, 1] if mask.sum() > 2 else float("nan")
    ax.set_title(f"may6: deploy_score vs {name}  (r={r:+.3f})")
    ax.set_xlabel(name)
    ax.set_ylabel("deploy_score")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig_score_correlations.png", dpi=120)
print(f"wrote {OUT / 'fig_score_correlations.png'}")
