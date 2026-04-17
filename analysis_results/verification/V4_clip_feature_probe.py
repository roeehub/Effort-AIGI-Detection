#!/usr/bin/env python3
"""
V4 — CLIP Feature Probe: Does CLIP Encode Sharpness?
=====================================================
Trains linear models on the 512-dim CLIP features to predict:
  1. Image sharpness (Laplacian variance) — linear regression
  2. Model fake probability (>0.5 threshold) — logistic regression
  3. Ground-truth label (real/fake) — logistic regression

Answers: Does the sharpness shortcut live in CLIP's frozen features,
or did the SVD residual layers learn it?

Input:  analysis_results/meta_analysis_features.npz
        analysis_results/meta_analysis_properties.csv
Output: Printed results + analysis_results/verification/V4_probe_results.txt
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # ── Load CLIP features ──────────────────────────────────────────
    npz_path = os.path.join(RESULTS_DIR, "meta_analysis_features.npz")
    csv_path = os.path.join(RESULTS_DIR, "meta_analysis_properties.csv")

    if not os.path.exists(npz_path):
        print(f"ERROR: {npz_path} not found. Run meta_analysis_enhancer.py first.")
        sys.exit(1)
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        sys.exit(1)

    npz = np.load(npz_path, allow_pickle=True)
    features = npz["features"]       # (N, 512)
    probs = npz["probs"]             # (N,) — R25_F1 predictions
    source_labels = npz["source_labels"]  # (N,) string
    gt_labels = npz["gt_labels"]     # (N,) string "real"/"fake"
    filenames = npz["filenames"]     # (N,) string

    df = pd.read_csv(csv_path)

    print(f"CLIP features: {features.shape}")
    print(f"Properties CSV: {len(df)} rows")
    print(f"Sources: {np.unique(source_labels)}\n")

    # ── Align features with properties via filename ─────────────────
    # Create a lookup from filename → row index in the properties CSV
    df_filename_to_idx = {fn: i for i, fn in enumerate(df["filename"].values)}

    aligned_indices = []
    feat_indices = []
    for i, fn in enumerate(filenames):
        if fn in df_filename_to_idx:
            aligned_indices.append(df_filename_to_idx[fn])
            feat_indices.append(i)

    print(f"Aligned {len(aligned_indices)} / {len(filenames)} images between NPZ and CSV\n")

    if len(aligned_indices) < 100:
        print("WARNING: Very few aligned images. Check filename formats.")
        # Still proceed with what we have

    X = features[feat_indices]
    sharpness = df.iloc[aligned_indices]["sharpness_laplacian_var"].values
    edge_density = df.iloc[aligned_indices]["edge_density"].values
    freq_high = df.iloc[aligned_indices]["freq_high_energy_ratio"].values
    model_prob = probs[feat_indices]
    gt = (gt_labels[feat_indices] == "fake").astype(int)
    sources = source_labels[feat_indices]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results_lines = []

    def log(msg):
        print(msg)
        results_lines.append(msg)

    log("=" * 75)
    log("PROBE 1: Can CLIP features predict SHARPNESS (Laplacian var)?")
    log("=" * 75)
    log("  (High R² → CLIP encodes quality → fix must be at data level)")
    log("  (Low R²  → SVD layers learned it → architectural fix possible)\n")

    # Ridge regression (more stable than OLS for 512 features)
    ridge = Ridge(alpha=1.0)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = cross_val_score(ridge, X_scaled, sharpness, cv=kf, scoring="r2")
    log(f"  Sharpness prediction R² (5-fold CV): {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

    # Also check for edge density
    r2_edge = cross_val_score(ridge, X_scaled, edge_density, cv=kf, scoring="r2")
    log(f"  Edge density prediction R²:          {r2_edge.mean():.4f} ± {r2_edge.std():.4f}")

    # And high-freq energy
    r2_freq = cross_val_score(ridge, X_scaled, freq_high, cv=kf, scoring="r2")
    log(f"  High-freq energy prediction R²:      {r2_freq.mean():.4f} ± {r2_freq.std():.4f}")

    log("")
    log("=" * 75)
    log("PROBE 2: Can CLIP features predict MODEL DECISION (prob > 0.5)?")
    log("=" * 75)
    log("  (Sanity check — should be very high if features carry the decision signal)\n")

    model_decision = (model_prob > 0.5).astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr = LogisticRegression(max_iter=1000, C=1.0)
    acc_scores = cross_val_score(lr, X_scaled, model_decision, cv=skf, scoring="accuracy")
    auc_scores = cross_val_score(lr, X_scaled, model_decision, cv=skf, scoring="roc_auc")
    log(f"  Model decision accuracy (5-fold CV): {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
    log(f"  Model decision AUC:                  {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")

    log("")
    log("=" * 75)
    log("PROBE 3: Can CLIP features predict GROUND TRUTH (real vs fake)?")
    log("=" * 75)
    log("  (How much of CLIP's representation is about actual fakeness?)\n")

    gt_acc = cross_val_score(lr, X_scaled, gt, cv=skf, scoring="accuracy")
    gt_auc = cross_val_score(lr, X_scaled, gt, cv=skf, scoring="roc_auc")
    log(f"  Ground truth accuracy (5-fold CV):   {gt_acc.mean():.4f} ± {gt_acc.std():.4f}")
    log(f"  Ground truth AUC:                    {gt_auc.mean():.4f} ± {gt_auc.std():.4f}")

    log("")
    log("=" * 75)
    log("PROBE 4: CLIP features → sharpness, broken down by source")
    log("=" * 75)
    log("  (Do certain sources drive the overall R²?)\n")

    for src in sorted(np.unique(sources)):
        mask = sources == src
        n = mask.sum()
        if n < 20:
            log(f"  {src:<45s}  n={n:>5d}  (too few for CV)")
            continue

        X_src = X_scaled[mask]
        y_src = sharpness[mask]
        prob_src = model_prob[mask]

        # Simple fit (no CV — per-source is small)
        ridge_src = Ridge(alpha=1.0).fit(X_src, y_src)
        r2_src = ridge_src.score(X_src, y_src)

        # Raw correlation: sharpness ↔ model prob within this source
        from scipy.stats import spearmanr
        corr, _ = spearmanr(y_src, prob_src)

        log(f"  {src:<45s}  n={n:>5d}  CLIP→sharpness R²={r2_src:.3f}  sharpness↔prob ρ={corr:+.3f}")

    # ── Verdict ─────────────────────────────────────────────────────
    log("\n" + "=" * 75)
    log("VERDICT")
    log("=" * 75)

    r2_mean = r2_scores.mean()
    if r2_mean > 0.5:
        log(f"✅ CLIP strongly encodes sharpness (R² = {r2_mean:.3f})")
        log("   → The shortcut is baked into CLIP's frozen representation.")
        log("   → Fix MUST happen at the data/augmentation level.")
        log("   → Architectural changes to SVD layers alone won't help.")
    elif r2_mean > 0.2:
        log(f"⚠️  CLIP moderately encodes sharpness (R² = {r2_mean:.3f})")
        log("   → Partially in CLIP, partially learned by SVD.")
        log("   → Both data-level and architecture-level fixes may help.")
    else:
        log(f"🔍 CLIP weakly encodes sharpness (R² = {r2_mean:.3f})")
        log("   → The SVD residual layers learned this shortcut themselves.")
        log("   → Architectural interventions (e.g., orthogonalization) could work.")
        log("   → Data augmentation may also help but isn't the only option.")

    gt_auc_mean = gt_auc.mean()
    log(f"\n   CLIP→ground_truth AUC = {gt_auc_mean:.3f}")
    if gt_auc_mean < 0.75:
        log("   → CLIP features alone are poor at distinguishing real/fake.")
        log("   → The SVD layers are doing most of the work (good or bad).")
    else:
        log("   → CLIP features carry genuine forgery signal beyond sharpness.")

    # ── Save ────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "V4_probe_results.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(results_lines))
    print(f"\n✅ Results saved to: {out_path}")


if __name__ == "__main__":
    main()
