"""Joint-marginal audit: does train pool span lockbox / chronic-6 on image-property axes?

Inputs:
  artifacts/unified_tags.parquet (atlas data, 14k frames)
  artifacts/chronic_full_tags.parquet (identity-keyed chronic frames, 3981 rows)

Outputs:
  tables/*.csv  -- per-axis marginal stats, pairwise Wasserstein/KS, discriminator coefs,
                   density-ratio CDFs, per-chronic-identity location
  figs/*.png   -- overlaid univariate histograms, top-2-PC scatter,
                  discriminator-axis density, density-ratio CDF
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "analysis/joint_marginal_audit_2026-05-19"
ART = f"{BASE}/artifacts"
TAB = f"{BASE}/tables"
FIG = f"{BASE}/figs"
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

AXES = ["lap_var", "luma_mean", "color_a_dev", "color_b_dev", "saturation_mean", "min_dim", "skin_frac"]
AXIS_LABEL = {
    "lap_var": "Sharpness (Laplacian var)",
    "luma_mean": "Brightness (luma mean)",
    "color_a_dev": "Color cast a (std)",
    "color_b_dev": "Color cast b (std)",
    "saturation_mean": "Saturation (mean)",
    "min_dim": "Resolution (min(W,H))",
    "skin_frac": "Skin fraction",
}

df = pd.read_parquet(f"{ART}/unified_tags.parquet")
print(f"Loaded unified_tags: {len(df)} rows, buckets: {df['bucket'].value_counts().to_dict()}")

# ============================================================================
# Part 1 — Univariate per-axis marginals
# ============================================================================
print("\n=== Part 1: Per-axis marginal stats ===")
real_buckets = ["train_real", "dev_real", "lockbox_real"]

per_axis_rows = []
for axis in AXES:
    for b in real_buckets:
        v = df.loc[df["bucket"] == b, axis].values
        per_axis_rows.append({
            "axis": axis, "bucket": b, "n": len(v),
            "mean": float(np.mean(v)), "std": float(np.std(v)),
            "q05": float(np.quantile(v, 0.05)),
            "q25": float(np.quantile(v, 0.25)),
            "q50": float(np.quantile(v, 0.50)),
            "q75": float(np.quantile(v, 0.75)),
            "q95": float(np.quantile(v, 0.95)),
        })
per_axis = pd.DataFrame(per_axis_rows)
per_axis.to_csv(f"{TAB}/per_axis_marginal.csv", index=False)
print(per_axis.to_string(index=False))

# Pairwise Wasserstein-1 + K-S between train_real and {dev_real, lockbox_real, chronic_lockbox}
print("\n=== Part 1b: Pairwise Wasserstein-1 and K-S ===")
shift_rows = []
train_real = df[df["bucket"] == "train_real"]
dev_real = df[df["bucket"] == "dev_real"]
lockbox_real = df[df["bucket"] == "lockbox_real"]
chronic_lockbox = df[(df["bucket"] == "lockbox_real") & (df["is_chronic"])]
chronic_dev = df[(df["bucket"] == "dev_real") & (df["is_chronic"])]

pairs = [
    ("train_real", "dev_real", train_real, dev_real),
    ("train_real", "lockbox_real", train_real, lockbox_real),
    ("train_real", "chronic_lockbox_real", train_real, chronic_lockbox),
    ("train_real", "chronic_dev_real", train_real, chronic_dev),
    ("dev_real", "lockbox_real", dev_real, lockbox_real),
]
for name_a, name_b, a, b in pairs:
    for axis in AXES:
        va, vb = a[axis].values, b[axis].values
        if len(va) < 5 or len(vb) < 5:
            continue
        w1 = stats.wasserstein_distance(va, vb)
        ks_stat, ks_p = stats.ks_2samp(va, vb)
        # normalize Wasserstein by combined std for cross-axis comparability
        combined_std = np.std(np.concatenate([va, vb]))
        w1_norm = w1 / combined_std if combined_std > 0 else np.nan
        shift_rows.append({
            "pair": f"{name_a} → {name_b}", "axis": axis,
            "n_a": len(va), "n_b": len(vb),
            "w1": w1, "w1_normalized": w1_norm,
            "ks_stat": ks_stat, "ks_p": ks_p,
        })
shift_df = pd.DataFrame(shift_rows)
shift_df.to_csv(f"{TAB}/pairwise_shift.csv", index=False)
print("\nWasserstein-1 (normalized by combined std) per axis:")
piv = shift_df.pivot(index="axis", columns="pair", values="w1_normalized")
print(piv.round(3).to_string())

# ============================================================================
# Part 2 — KLIEP-style discriminator: train_real vs lockbox_real
# ============================================================================
print("\n=== Part 2: Logistic discriminator train_real vs lockbox_real ===")
A = train_real[AXES].values.astype(float)
B = lockbox_real[AXES].values.astype(float)
X = np.vstack([A, B])
y = np.array([0] * len(A) + [1] * len(B))

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# 5-fold CV balanced-accuracy for an honest readout
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
fold_baccs = []
for tr, te in skf.split(Xs, y):
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    clf.fit(Xs[tr], y[tr])
    pred = clf.predict(Xs[te])
    tn = np.sum((pred == 0) & (y[te] == 0))
    fp = np.sum((pred == 1) & (y[te] == 0))
    fn = np.sum((pred == 0) & (y[te] == 1))
    tp = np.sum((pred == 1) & (y[te] == 1))
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    fold_baccs.append((sens + spec) / 2)
print(f"  5-fold balanced accuracy: mean={np.mean(fold_baccs):.4f}  std={np.std(fold_baccs):.4f}")
print(f"  fold values: {[f'{x:.3f}' for x in fold_baccs]}")

# Full-data fit for coefficient interpretation
clf_full = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
clf_full.fit(Xs, y)
coef = clf_full.coef_[0]
coef_df = pd.DataFrame({
    "axis": AXES,
    "label": [AXIS_LABEL[a] for a in AXES],
    "coef_standardized": coef,
    "abs_coef": np.abs(coef),
}).sort_values("abs_coef", ascending=False)
coef_df.to_csv(f"{TAB}/discriminator_coefs_train_vs_lockbox.csv", index=False)
print("\n  Discriminator standardized coefficients (sign: + → lockbox-side, − → train-side):")
print(coef_df.to_string(index=False))

# ============================================================================
# Part 3 — PCA + KDE density-ratio analysis
# ============================================================================
print("\n=== Part 3: PCA + KDE density-ratio analysis ===")
# Fit PCA on combined real-side data
all_real = df[df["bucket"].isin(["train_real", "dev_real", "lockbox_real"])][AXES].values.astype(float)
all_real_scaled = StandardScaler().fit_transform(all_real)
pca = PCA(n_components=5, random_state=0)
pca.fit(all_real_scaled)
print(f"  PCA explained variance ratios (top-5): {pca.explained_variance_ratio_.round(4)}")
print(f"  cumulative: {pca.explained_variance_ratio_.cumsum().round(4)}")

# PCA component loadings — which axes drive each PC?
loadings = pd.DataFrame(pca.components_.T, index=AXES,
                         columns=[f"PC{i+1}" for i in range(pca.n_components_)])
loadings.to_csv(f"{TAB}/pca_loadings.csv")
print("\n  PCA loadings (axes × PCs):")
print(loadings.round(3).to_string())

# Project each bucket into PC space
scaler_real = StandardScaler().fit(all_real)
def project(bucket_df):
    X = bucket_df[AXES].values.astype(float)
    X = scaler_real.transform(X)
    return pca.transform(X)

P_train = project(train_real)
P_dev = project(dev_real)
P_lock = project(lockbox_real)
P_chronic_lock = project(chronic_lockbox)

# KDE density ratio on top-K PCs (K=3 captures most variance, keeps KDE tractable)
K = 3
kde_train = KernelDensity(bandwidth="scott", kernel="gaussian").fit(P_train[:, :K])
kde_lock = KernelDensity(bandwidth="scott", kernel="gaussian").fit(P_lock[:, :K])
kde_dev = KernelDensity(bandwidth="scott", kernel="gaussian").fit(P_dev[:, :K])

def eval_kde(kde, X):
    """Return density (not log-density), clipped to a small positive minimum."""
    log_p = kde.score_samples(X)
    return np.exp(log_p)

# For each lockbox-real frame, compute train density / lockbox density
d_train_at_lock = eval_kde(kde_train, P_lock[:, :K])
d_lock_at_lock = eval_kde(kde_lock, P_lock[:, :K])
ratio_lock = d_train_at_lock / np.maximum(d_lock_at_lock, 1e-30)

# Same for dev_real (control — should show much smaller fraction below threshold)
d_train_at_dev = eval_kde(kde_train, P_dev[:, :K])
d_dev_at_dev = eval_kde(kde_dev, P_dev[:, :K])
ratio_dev = d_train_at_dev / np.maximum(d_dev_at_dev, 1e-30)

# Chronic lockbox subset
d_train_at_chronic = eval_kde(kde_train, P_chronic_lock[:, :K])
d_lock_at_chronic = eval_kde(kde_lock, P_chronic_lock[:, :K])
ratio_chronic = d_train_at_chronic / np.maximum(d_lock_at_chronic, 1e-30)

# Coverage thresholds: fraction of deployment mass where train density is < α × deployment density
thresholds = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
cov_rows = []
for name, ratio in [("dev_real", ratio_dev), ("lockbox_real", ratio_lock), ("chronic_lockbox_real", ratio_chronic)]:
    for t in thresholds:
        frac = float(np.mean(ratio < t))
        cov_rows.append({"deployment_pool": name, "n": len(ratio),
                         "threshold_alpha": t,
                         "fraction_with_ratio<alpha": frac})
cov_df = pd.DataFrame(cov_rows)
cov_df.to_csv(f"{TAB}/coverage_density_ratio.csv", index=False)
print("\n  Coverage table: fraction of deployment mass with train_density / dep_density < α")
print("  (top-3 PCs, Gaussian KDE, scott bandwidth)")
piv2 = cov_df.pivot(index="threshold_alpha", columns="deployment_pool",
                     values="fraction_with_ratio<alpha")
print(piv2.round(3).to_string())

# ============================================================================
# Part 4 — Per-chronic-identity breakdown
# ============================================================================
print("\n=== Part 4: Per-chronic-identity location ===")
chronic_in_atlas = df[df["is_chronic"]].copy()
chronic_in_atlas["pc1"] = project(chronic_in_atlas)[:, 0]
chronic_in_atlas["pc2"] = project(chronic_in_atlas)[:, 1]
# Per-frame density ratio
P_chronic_all = project(chronic_in_atlas)[:, :K]
d_train_c = eval_kde(kde_train, P_chronic_all)
d_lock_c = eval_kde(kde_lock, P_chronic_all)
chronic_in_atlas["density_ratio_train_over_lock"] = d_train_c / np.maximum(d_lock_c, 1e-30)

# Aggregate per identity
chronic_summary = chronic_in_atlas.groupby("identity_key").agg(
    n=("frame_path", "count"),
    bucket=("bucket", lambda x: x.value_counts().index[0]),
    median_ratio=("density_ratio_train_over_lock", "median"),
    frac_below_0_10=("density_ratio_train_over_lock", lambda x: float(np.mean(x < 0.10))),
    frac_below_0_25=("density_ratio_train_over_lock", lambda x: float(np.mean(x < 0.25))),
    median_lap_var=("lap_var", "median"),
    median_color_a_dev=("color_a_dev", "median"),
    median_skin_frac=("skin_frac", "median"),
    median_min_dim=("min_dim", "median"),
).reset_index().sort_values("median_ratio")
chronic_summary.to_csv(f"{TAB}/per_chronic_identity_location.csv", index=False)
print("\n  Per-chronic-identity table (sorted by median train/lockbox density ratio):")
print(chronic_summary.head(30).to_string(index=False))

# ============================================================================
# Part 5 — Figures
# ============================================================================
print("\n=== Part 5: Figures ===")

# Fig 1: per-axis histogram overlay (train_real / dev_real / lockbox_real)
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes_flat = axes.flatten()
colors = {"train_real": "C0", "dev_real": "C1", "lockbox_real": "C3"}
for i, axis_name in enumerate(AXES):
    ax = axes_flat[i]
    for b in ["train_real", "dev_real", "lockbox_real"]:
        v = df.loc[df["bucket"] == b, axis_name].values
        # clip to percentile range for plotting
        lo, hi = np.percentile(v, [1, 99])
        bins = np.linspace(lo, hi, 50)
        ax.hist(v, bins=bins, density=True, alpha=0.45, label=f"{b} (n={len(v)})", color=colors[b])
    ax.set_title(AXIS_LABEL[axis_name])
    ax.set_xlabel(axis_name)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
for j in range(len(AXES), len(axes_flat)):
    axes_flat[j].axis("off")
fig.suptitle("Per-axis marginal: train_real vs dev_real vs lockbox_real", fontsize=14)
fig.tight_layout()
fig.savefig(f"{FIG}/per_axis_histograms.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# Fig 2: PC1 × PC2 scatter with density contours per bucket
fig, ax = plt.subplots(figsize=(10, 8))
for b, c, alpha in [("train_real", "C0", 0.35), ("dev_real", "C1", 0.35), ("lockbox_real", "C3", 0.55)]:
    sub = df[df["bucket"] == b]
    P = project(sub)
    ax.scatter(P[:, 0], P[:, 1], s=4, alpha=alpha, label=f"{b} (n={len(sub)})", color=c)
# Overlay chronic-6 frames as black markers
if len(chronic_in_atlas) > 0:
    ax.scatter(chronic_in_atlas["pc1"], chronic_in_atlas["pc2"], s=8, alpha=0.6,
               marker="x", color="black", label=f"chronic-6 (n={len(chronic_in_atlas)})")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
ax.set_title("Top-2 PC scatter — train vs dev vs lockbox, with chronic-6 overlay")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIG}/pc1_pc2_scatter.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# Fig 3: density-ratio CDF
fig, ax = plt.subplots(figsize=(10, 7))
for name, ratio, color in [
    ("dev_real (n=%d)" % len(ratio_dev), ratio_dev, "C1"),
    ("lockbox_real (n=%d)" % len(ratio_lock), ratio_lock, "C3"),
    ("chronic ∩ lockbox_real (n=%d)" % len(ratio_chronic), ratio_chronic, "black"),
]:
    sorted_r = np.sort(ratio)
    y = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
    ax.plot(sorted_r, y, label=name, color=color, linewidth=2)
ax.set_xscale("log")
ax.set_xlim(1e-3, 1e2)
ax.axvline(0.10, color="gray", linestyle="--", alpha=0.5, label="α=0.10 threshold")
ax.axvline(0.25, color="gray", linestyle=":", alpha=0.5, label="α=0.25 threshold")
ax.set_xlabel("density ratio  =  train_density(x) / deployment_density(x)")
ax.set_ylabel("CDF (fraction of mass below)")
ax.set_title("Density-ratio CDF (top-3 PCs)\nLeft = train under-covers; Right = train over-covers")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG}/density_ratio_cdf.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# Fig 4: discriminator-axis density
disc_axis_train = scaler.transform(train_real[AXES].values) @ coef
disc_axis_dev = scaler.transform(dev_real[AXES].values) @ coef
disc_axis_lock = scaler.transform(lockbox_real[AXES].values) @ coef
fig, ax = plt.subplots(figsize=(10, 6))
for v, name, c in [
    (disc_axis_train, f"train_real (n={len(disc_axis_train)})", "C0"),
    (disc_axis_dev, f"dev_real (n={len(disc_axis_dev)})", "C1"),
    (disc_axis_lock, f"lockbox_real (n={len(disc_axis_lock)})", "C3"),
]:
    ax.hist(v, bins=60, density=True, alpha=0.55, label=name, color=c)
ax.set_xlabel("Discriminator score (− = train-side, + = lockbox-side)")
ax.set_ylabel("density")
ax.set_title(f"Logistic discriminator on 7-axis IQ space\n5-fold balanced acc = {np.mean(fold_baccs):.3f} ± {np.std(fold_baccs):.3f}")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIG}/discriminator_axis_density.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# Fig 5: per-axis coefficient bar chart
fig, ax = plt.subplots(figsize=(10, 5))
coef_sorted = coef_df.sort_values("coef_standardized")
colors_bar = ["C3" if x > 0 else "C0" for x in coef_sorted["coef_standardized"]]
ax.barh(coef_sorted["label"], coef_sorted["coef_standardized"], color=colors_bar)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("Standardized coefficient (+ → lockbox-distinctive, − → train-distinctive)")
ax.set_title("Discriminator weights: what axes separate train from lockbox?")
fig.tight_layout()
fig.savefig(f"{FIG}/discriminator_axis_coefs.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ============================================================================
# Part 6 — Summary JSON
# ============================================================================
summary = {
    "n_frames_by_bucket": df["bucket"].value_counts().to_dict(),
    "discriminator_balanced_acc_mean": float(np.mean(fold_baccs)),
    "discriminator_balanced_acc_std": float(np.std(fold_baccs)),
    "top_3_discriminating_axes_by_abs_coef": coef_df.head(3)["axis"].tolist(),
    "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    "coverage_density_ratio": piv2.to_dict(),
    "wasserstein_train_vs_lockbox_per_axis": {
        a: float(shift_df.loc[(shift_df["pair"] == "train_real → lockbox_real") &
                              (shift_df["axis"] == a), "w1_normalized"].values[0])
        for a in AXES
    },
    "chronic_identity_count": int((df["is_chronic"]).sum()),
}
with open(f"{TAB}/SUMMARY.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nWrote SUMMARY.json")
print(json.dumps(summary, indent=2))
print("\nDONE.")
