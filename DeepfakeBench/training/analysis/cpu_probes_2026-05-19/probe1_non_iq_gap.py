"""Probe 1 — characterize the non-IQ component of the dev→lockbox gap in CLIP-feature space.

Plan:
  (a) Reproduce D8: logistic discriminator on raw CLIP features (dev_real vs lockbox_real).
  (b) Residualize CLIP features against the 7 IQ axes (regress IQ-explainable component OUT).
  (c) Re-run the discriminator on the residual CLIP features. The accuracy that remains
      is the "non-IQ" component of the gap.
  (d) Identify which residual CLIP-feature dimensions carry the remaining separability and
      cluster the heaviest dimensions to look for an interpretable axis.

Inputs:
  analysis/lockbox_tagging/full_tags_2026-04-27.parquet (clip_embed 512-d + IQ axes
  sharpness_laplacian, brightness_v_mean, saturation_s_mean, face_pixel_area,
  face_area_ratio, contrast_rms, width, height)
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_TAB = "analysis/cpu_probes_2026-05-19/tables"
OUT_FIG = "analysis/cpu_probes_2026-05-19/figs"
OUT_ART = "analysis/cpu_probes_2026-05-19/artifacts"
os.makedirs(OUT_TAB, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_ART, exist_ok=True)

# IQ axes available in full_tags. Note: full_tags uses different names than the atlas:
IQ_AXES = [
    "sharpness_laplacian",     # ≈ atlas lap_var
    "brightness_v_mean",       # ≈ atlas luma_mean
    "saturation_s_mean",       # ≈ atlas saturation_mean
    "face_pixel_area",         # face size in pixels (different than atlas min_dim/skin_frac)
    "face_area_ratio",         # ≈ atlas skin_frac proxy
    "contrast_rms",
    "yaw_deg", "pitch_deg", "roll_deg",   # add pose to broaden the "explainable nuisance" basis
]
# We'll also add categorical CLIP-aux: clip_capture_mode (one-hot), clip_quality
CATS = ["clip_capture_mode", "clip_quality", "clip_lighting", "clip_occlusion"]

print("Loading full_tags ...")
df = pd.read_parquet("analysis/lockbox_tagging/full_tags_2026-04-27.parquet")
print(f"  rows: {len(df)}")

# Filter to real-only (the binding pair is dev_real vs lockbox_real per D8)
real = df[df["label"] == "real"].copy()
print(f"  real-only: {len(real)} ({real['split'].value_counts().to_dict()})")

# Drop rows with NaN in IQ axes
n_before = len(real)
for a in IQ_AXES:
    real = real[real[a].notna()].copy()
print(f"  dropped {n_before - len(real)} rows with NaN IQ; kept {len(real)}")

# Build CLIP feature matrix
clip_X = np.stack(real["clip_embed"].values).astype(np.float32)
print(f"  CLIP feature matrix: {clip_X.shape}")

# Build IQ feature matrix (continuous + categorical one-hots)
iq_cont = real[IQ_AXES].values.astype(np.float32)
# Categorical one-hots
cat_dfs = []
for c in CATS:
    if c in real.columns:
        oh = pd.get_dummies(real[c], prefix=c, dummy_na=True).astype(np.float32)
        cat_dfs.append(oh.values)
iq_cat = np.concatenate(cat_dfs, axis=1) if cat_dfs else np.zeros((len(real), 0))
iq_full = np.concatenate([iq_cont, iq_cat], axis=1)
print(f"  IQ feature matrix: {iq_full.shape}  (cont {iq_cont.shape[1]} + cat {iq_cat.shape[1]})")

# Labels: 0=dev, 1=lockbox
y = (real["split"] == "lockbox").astype(int).values
print(f"  label distribution: dev={np.sum(y==0)}, lockbox={np.sum(y==1)}")

# ===========================================================================
# STAGE A — Discriminator on raw CLIP features (reproduce D8)
# ===========================================================================
print("\n=== STAGE A: raw CLIP-feature discriminator ===")

def run_discriminator(X, y, n_splits=5, seed=0, C=1.0):
    """Return 5-fold balanced accuracy + AUC."""
    from sklearn.metrics import roc_auc_score
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    baccs, aucs = [], []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", n_jobs=1)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        prob = clf.predict_proba(Xte)[:, 1]
        tn = np.sum((pred == 0) & (y[te] == 0))
        fp = np.sum((pred == 1) & (y[te] == 0))
        fn = np.sum((pred == 0) & (y[te] == 1))
        tp = np.sum((pred == 1) & (y[te] == 1))
        sens = tp / (tp + fn + 1e-12)
        spec = tn / (tn + fp + 1e-12)
        baccs.append((sens + spec) / 2)
        aucs.append(roc_auc_score(y[te], prob))
    return np.mean(baccs), np.std(baccs), np.mean(aucs), np.std(aucs)

a_bacc, a_bacc_std, a_auc, a_auc_std = run_discriminator(clip_X, y)
print(f"  raw CLIP features: bacc={a_bacc:.4f} ± {a_bacc_std:.4f}  AUC={a_auc:.4f} ± {a_auc_std:.4f}")

# ===========================================================================
# STAGE B — Discriminator on IQ-only features (baseline for "IQ-explainable" gap)
# ===========================================================================
print("\n=== STAGE B: IQ-only feature discriminator ===")
b_bacc, b_bacc_std, b_auc, b_auc_std = run_discriminator(iq_full, y)
print(f"  IQ-only features: bacc={b_bacc:.4f} ± {b_bacc_std:.4f}  AUC={b_auc:.4f} ± {b_auc_std:.4f}")

# ===========================================================================
# STAGE C — Residualize CLIP features against IQ, then discriminate
# ===========================================================================
print("\n=== STAGE C: CLIP features residualized against IQ ===")
# Standardize IQ features, then fit a linear model from IQ → each CLIP dim, and subtract
iq_scaler = StandardScaler()
iq_std = iq_scaler.fit_transform(iq_full)

# For each CLIP dim, fit linear regression on IQ, take residuals
clip_resid = np.zeros_like(clip_X)
r2_per_dim = np.zeros(clip_X.shape[1])
for d in range(clip_X.shape[1]):
    lr = LinearRegression()
    lr.fit(iq_std, clip_X[:, d])
    pred = lr.predict(iq_std)
    clip_resid[:, d] = clip_X[:, d] - pred
    ss_res = np.sum((clip_X[:, d] - pred) ** 2)
    ss_tot = np.sum((clip_X[:, d] - clip_X[:, d].mean()) ** 2)
    r2_per_dim[d] = 1 - ss_res / max(ss_tot, 1e-12)
print(f"  Per-dim R² (CLIP_dim ~ IQ): mean={r2_per_dim.mean():.4f}  median={np.median(r2_per_dim):.4f}  max={r2_per_dim.max():.4f}")
print(f"  Number of CLIP dims with R²>0.10: {(r2_per_dim > 0.10).sum()} / {len(r2_per_dim)}")

# Discriminator on residual CLIP features
c_bacc, c_bacc_std, c_auc, c_auc_std = run_discriminator(clip_resid, y)
print(f"  residualized CLIP: bacc={c_bacc:.4f} ± {c_bacc_std:.4f}  AUC={c_auc:.4f} ± {c_auc_std:.4f}")

# Also try a stronger projection: kill the top-K CLIP-PCs that align with IQ
print("\n  Additional sanity: drop top-20 IQ-explained CLIP dims, rerun ...")
top20 = np.argsort(r2_per_dim)[::-1][:20]
keep_mask = np.ones(clip_X.shape[1], dtype=bool)
keep_mask[top20] = False
clip_drop = clip_X[:, keep_mask]
d_bacc, d_bacc_std, d_auc, d_auc_std = run_discriminator(clip_drop, y)
print(f"  dropped top-20 IQ-aligned dims: bacc={d_bacc:.4f} AUC={d_auc:.4f}")

# ===========================================================================
# STAGE D — What does the residual axis look like?
# ===========================================================================
print("\n=== STAGE D: characterize the residual discriminator axis ===")
# Fit a single discriminator on all residualized CLIP features
scaler_resid = StandardScaler()
Xs = scaler_resid.fit_transform(clip_resid)
clf_resid = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
clf_resid.fit(Xs, y)
coef_resid = clf_resid.coef_[0]
# Project per-sample onto the discriminator axis
disc_axis_resid = Xs @ coef_resid

# What fraction of the residual axis correlates with each IQ axis?
print("\n  Correlation of residual discriminator axis with each IQ axis:")
for i, name in enumerate(IQ_AXES):
    rho = np.corrcoef(disc_axis_resid, iq_cont[:, i])[0, 1]
    print(f"    {name:<25} rho = {rho:+.4f}")

# Take the residual discriminator axis and see if it aligns with arcface_norm (a face-recognition feature)
# or with face geometry features
extra_features = ["arcface_norm", "yaw_deg", "pitch_deg", "roll_deg", "aspect_ratio",
                  "eye_aspect_ratio_left", "eye_aspect_ratio_right", "mouth_aspect_ratio",
                  "jpeg_qf_estimate", "file_bytes", "width", "height"]
print("\n  Correlation with extra geometric/file features:")
for name in extra_features:
    if name in real.columns:
        v = pd.to_numeric(real[name], errors="coerce").values
        m = np.isfinite(v)
        if m.sum() < 100:
            continue
        rho = np.corrcoef(disc_axis_resid[m], v[m])[0, 1]
        print(f"    {name:<27} rho = {rho:+.4f}  (n={m.sum()})")

# Compare arcface_embed (identity-cluster) to the residual axis
arc = real["arcface_embed"]
if arc.notna().all():
    arc_X = np.stack(arc.values).astype(np.float32)
    print(f"\n  ArcFace embedding dim: {arc_X.shape[1]}; running ArcFace-only discriminator for comparison")
    af_bacc, af_bacc_std, af_auc, af_auc_std = run_discriminator(arc_X, y)
    print(f"    ArcFace-only: bacc={af_bacc:.4f} AUC={af_auc:.4f}")
    # Project ArcFace onto same dev/lockbox axis as CLIP residual
    af_scaler = StandardScaler()
    af_std = af_scaler.fit_transform(arc_X)
    af_clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    af_clf.fit(af_std, y)
    af_axis = af_std @ af_clf.coef_[0]
    rho_clip_arc = np.corrcoef(disc_axis_resid, af_axis)[0, 1]
    print(f"    Correlation between residual CLIP axis and ArcFace axis: rho = {rho_clip_arc:+.4f}")

# ===========================================================================
# STAGE E — Try identity-cluster residualization
# ===========================================================================
print("\n=== STAGE E: identity-cluster residualization ===")
# Use ArcFace embeddings (256/512-d, face identity space) as additional nuisance regressors
if 'arc_X' in dir():
    nuisance_mat = np.concatenate([iq_std, StandardScaler().fit_transform(arc_X)], axis=1)
    clip_resid_id = np.zeros_like(clip_X)
    for d in range(clip_X.shape[1]):
        lr = LinearRegression()
        lr.fit(nuisance_mat, clip_X[:, d])
        clip_resid_id[:, d] = clip_X[:, d] - lr.predict(nuisance_mat)
    e_bacc, e_bacc_std, e_auc, e_auc_std = run_discriminator(clip_resid_id, y)
    print(f"  CLIP residualized vs (IQ ∪ ArcFace): bacc={e_bacc:.4f} ± {e_bacc_std:.4f}  AUC={e_auc:.4f}")
else:
    e_bacc = e_auc = float("nan")

# ===========================================================================
# Save summary
# ===========================================================================
summary = {
    "n_rows": int(len(real)),
    "dev_count": int((y == 0).sum()),
    "lockbox_count": int((y == 1).sum()),
    "stage_A_raw_clip": {"bacc": float(a_bacc), "bacc_std": float(a_bacc_std), "auc": float(a_auc)},
    "stage_B_iq_only":  {"bacc": float(b_bacc), "bacc_std": float(b_bacc_std), "auc": float(b_auc)},
    "stage_C_clip_resid_vs_iq": {"bacc": float(c_bacc), "bacc_std": float(c_bacc_std), "auc": float(c_auc)},
    "stage_D_drop_top20_iq_aligned": {"bacc": float(d_bacc), "auc": float(d_auc)},
    "stage_E_clip_resid_vs_iq_arcface": {"bacc": float(e_bacc), "auc": float(e_auc)},
    "n_clip_dims_with_iq_r2_gt_0_10": int((r2_per_dim > 0.10).sum()),
    "iq_r2_mean": float(r2_per_dim.mean()),
    "iq_r2_max": float(r2_per_dim.max()),
}
with open(f"{OUT_TAB}/probe1_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Save coefs / r2
pd.DataFrame({
    "clip_dim": np.arange(clip_X.shape[1]),
    "iq_r2": r2_per_dim,
    "discriminator_coef_residual": coef_resid,
}).to_csv(f"{OUT_TAB}/probe1_clip_dim_stats.csv", index=False)

# Figure: bacc + AUC bar chart across stages
stages = ["A: raw CLIP", "B: IQ only", "C: CLIP - IQ", "D: drop top-20", "E: CLIP - (IQ + ArcFace)"]
baccs = [a_bacc, b_bacc, c_bacc, d_bacc, e_bacc]
aucs  = [a_auc, b_auc, c_auc, d_auc, e_auc]
fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(stages))
w = 0.35
ax.bar(x - w/2, baccs, w, label="balanced acc", color="C0")
ax.bar(x + w/2, aucs, w, label="AUC", color="C1")
for i, (b, a) in enumerate(zip(baccs, aucs)):
    ax.text(i - w/2, b + 0.005, f"{b:.3f}", ha="center", fontsize=9)
    ax.text(i + w/2, a + 0.005, f"{a:.3f}", ha="center", fontsize=9)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(stages, rotation=15, ha="right")
ax.set_ylabel("score")
ax.set_ylim(0.4, 1.05)
ax.set_title("dev_real vs lockbox_real discriminator across nuisance-residualization stages\n(higher = stronger separation)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/probe1_stages.png", dpi=120, bbox_inches="tight")
plt.close(fig)

print("\n=== Probe 1 summary ===")
print(json.dumps(summary, indent=2))
print("\nDONE.")
