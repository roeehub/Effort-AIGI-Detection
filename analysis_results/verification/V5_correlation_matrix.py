#!/usr/bin/env python3
"""
V5 — Feature-Prediction Correlation Matrix
===========================================
Computes Pearson & Spearman correlations between every image property
and model_fake_prob across the full 2,602-image meta-analysis dataset.

Answers: Is sharpness really the #1 predictor, or one of many?

Input:  analysis_results/meta_analysis_properties.csv
Output: Printed table + analysis_results/verification/V5_correlation_results.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

RESULTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    # ── Load data ───────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "meta_analysis_properties.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run meta_analysis_enhancer.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} images from {df['source'].nunique()} sources\n")

    # ── Identify numeric property columns ───────────────────────────
    exclude_cols = {"source", "label", "filename", "model_fake_prob"}
    prop_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.int64, float, int]]

    target = "model_fake_prob"
    if target not in df.columns:
        print(f"ERROR: '{target}' column not found.")
        sys.exit(1)

    # ── Compute correlations ────────────────────────────────────────
    results = []
    y = df[target].values

    for col in prop_cols:
        x = df[col].values
        # Drop NaN pairs
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 10:
            continue
        x_clean, y_clean = x[mask], y[mask]

        pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
        spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)

        results.append({
            "property": col,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "abs_pearson": abs(pearson_r),
            "abs_spearman": abs(spearman_r),
        })

    results_df = pd.DataFrame(results).sort_values("abs_spearman", ascending=False)

    # ── Print results ───────────────────────────────────────────────
    print("=" * 85)
    print("CORRELATION: image property → model_fake_prob  (sorted by |Spearman|)")
    print("=" * 85)
    print(f"{'Property':<30s}  {'Pearson r':>10s}  {'Spearman r':>10s}  {'Direction':<12s}  {'Strength'}")
    print("-" * 85)

    for _, row in results_df.iterrows():
        direction = "↑ sharpER→faker" if row["spearman_r"] > 0 else "↓ smoothER→faker"
        if row["abs_spearman"] > 0.4:
            strength = "★★★ STRONG"
        elif row["abs_spearman"] > 0.2:
            strength = "★★  MODERATE"
        elif row["abs_spearman"] > 0.1:
            strength = "★   WEAK"
        else:
            strength = "·   negligible"
        print(f"{row['property']:<30s}  {row['pearson_r']:>+10.4f}  {row['spearman_r']:>+10.4f}  {direction:<12s}  {strength}")

    # ── Per-source breakdown for top features ───────────────────────
    top_features = results_df.head(5)["property"].tolist()
    print("\n" + "=" * 85)
    print("PER-SOURCE MEANS for top correlated features")
    print("=" * 85)

    summary = df.groupby("source").agg(
        n=("model_fake_prob", "count"),
        model_prob_mean=("model_fake_prob", "mean"),
        model_prob_median=("model_fake_prob", "median"),
        **{f"{feat}_mean": (feat, "mean") for feat in top_features},
    ).reindex([s for s in [
        "wma_enhanced",
        "deeplive_quality_enhancement_fake",
        "deeplive_minimal_processing_fake",
        "deeplive_edge_cases_fake",
        "visomaster_fake",
        "df40_fake",
        "deeplive_quality_enhancement_real",
        "deeplive_minimal_processing_real",
        "df40_real",
        "external_youtube_real",
    ] if s in df["source"].unique()])

    print(summary.to_string())

    # ── Real vs Fake breakdown (ignoring source) ────────────────────
    print("\n" + "=" * 85)
    print("REAL vs FAKE means (does model use label-correlated features?)")
    print("=" * 85)

    for label_val in ["real", "fake"]:
        sub = df[df["label"] == label_val]
        print(f"\n  {label_val.upper()} (n={len(sub)}):")
        print(f"    model_fake_prob  mean={sub['model_fake_prob'].mean():.4f}  median={sub['model_fake_prob'].median():.4f}")
        for feat in top_features:
            print(f"    {feat:<30s}  mean={sub[feat].mean():.4f}  std={sub[feat].std():.4f}")

    # ── Save CSV ────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "V5_correlation_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n✅ Full results saved to: {out_path}")

    # ── Verdict ─────────────────────────────────────────────────────
    top1 = results_df.iloc[0]
    print("\n" + "=" * 85)
    print("VERDICT")
    print("=" * 85)
    if "sharpness" in top1["property"] or "edge" in top1["property"] or "tenengrad" in top1["property"]:
        print(f"✅ CONFIRMED: Sharpness/texture is the #1 predictor of model output.")
        print(f"   Top feature: {top1['property']} (Spearman r = {top1['spearman_r']:+.4f})")
    else:
        print(f"⚠️  UNEXPECTED: Top predictor is '{top1['property']}' (Spearman r = {top1['spearman_r']:+.4f})")
        print(f"   Sharpness may not be the primary shortcut — investigate further.")


if __name__ == "__main__":
    main()
