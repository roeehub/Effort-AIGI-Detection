#!/usr/bin/env python3
"""
V6 — Cross-Model Sharpness Consistency
=======================================
Checks whether the sharpness↔prediction correlation holds across ALL 5 models
or is specific to certain training configurations.

Answers: Is the sharpness shortcut universal (data problem) or model-specific?

Input:  analysis_results/enhancer_eval_per_image.csv  (1,202 WMA images × 5 models)
        analysis_results/meta_analysis_properties.csv  (WMA images with sharpness)
Output: Printed results + analysis_results/verification/V6_cross_model_results.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

RESULTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # ── Load per-image predictions (5 models × 1,202 WMA images) ───
    eval_path = os.path.join(RESULTS_DIR, "enhancer_eval_per_image.csv")
    props_path = os.path.join(RESULTS_DIR, "meta_analysis_properties.csv")

    if not os.path.exists(eval_path):
        print(f"ERROR: {eval_path} not found.")
        sys.exit(1)
    if not os.path.exists(props_path):
        print(f"ERROR: {props_path} not found.")
        sys.exit(1)

    eval_df = pd.read_csv(eval_path)
    props_df = pd.read_csv(props_path)

    print(f"Eval data: {len(eval_df)} images × {len(eval_df.columns)-1} models")

    # ── Filter properties to WMA source only ────────────────────────
    wma_props = props_df[props_df["source"] == "wma_enhanced"].copy()
    print(f"WMA properties: {len(wma_props)} images")

    # ── Join on filename ────────────────────────────────────────────
    # eval_df has bare filenames: "6__20260214_152604__frame_026905_crop_001.jpg"
    # props_df has prefixed: "wma_enhanced/6__20260214_152604__frame_026905_crop_001.jpg"
    wma_props["bare_filename"] = wma_props["filename"].str.replace("wma_enhanced/", "", n=1)
    merged = eval_df.merge(wma_props, left_on="filename", right_on="bare_filename", how="inner")
    print(f"Joined: {len(merged)} images\n")

    if len(merged) < 50:
        print("WARNING: Very few joined images. Check filename formats.")
        print(f"  Eval filenames sample: {eval_df['filename'].head(3).tolist()}")
        print(f"  Props filenames sample: {wma_props['bare_filename'].head(3).tolist()}")

    # ── Model columns ──────────────────────────────────────────────
    model_cols = [c for c in eval_df.columns if c.endswith("_prob")]
    sharpness_cols = ["sharpness_laplacian_var", "sharpness_tenengrad", "edge_density",
                      "freq_high_energy_ratio", "freq_high_to_low_ratio"]

    # Model metadata for display
    model_info = {
        "R25_F1_prob":   {"name": "R25_F1",  "desc": "k=32, base_only (deployed)",  "acc": "21.0%"},
        "R3_FT3_prob":   {"name": "R3_FT3",  "desc": "FT, base_only, +QE data",     "acc": "39.5%"},
        "R3_FT1_prob":   {"name": "R3_FT1",  "desc": "FT, quality_robust moderate",  "acc": "8.3%"},
        "R3_FT2_prob":   {"name": "R3_FT2",  "desc": "FT, quality_robust light",     "acc": "9.3%"},
        "B16_old_prob":  {"name": "B16_old",  "desc": "Phase 1 (k=8, CE)",           "acc": "55.7%"},
    }

    # ── Compute correlations per model ──────────────────────────────
    print("=" * 90)
    print("SHARPNESS ↔ MODEL PREDICTION CORRELATION  (per model, WMA images only)")
    print("=" * 90)

    results = []

    for mcol in model_cols:
        info = model_info.get(mcol, {"name": mcol, "desc": "?", "acc": "?"})
        row = {"model": info["name"], "description": info["desc"], "wma_accuracy": info["acc"]}

        y = merged[mcol].values

        for scol in sharpness_cols:
            x = merged[scol].values
            mask = ~(np.isnan(x) | np.isnan(y))
            r_spear, p_spear = stats.spearmanr(x[mask], y[mask])
            r_pear, p_pear = stats.pearsonr(x[mask], y[mask])
            row[f"{scol}_spearman"] = r_spear
            row[f"{scol}_pearson"] = r_pear

        results.append(row)

    results_df = pd.DataFrame(results)

    # ── Pretty print ────────────────────────────────────────────────
    print(f"\n{'Model':<10s}  {'WMA Acc':>8s}  ", end="")
    for scol in sharpness_cols:
        short = scol.replace("sharpness_", "").replace("freq_", "f_")[:12]
        print(f"{'ρ_' + short:>14s}  ", end="")
    print()
    print("-" * 90)

    for _, row in results_df.iterrows():
        print(f"{row['model']:<10s}  {row['wma_accuracy']:>8s}  ", end="")
        for scol in sharpness_cols:
            r = row[f"{scol}_spearman"]
            marker = "★" if abs(r) > 0.3 else " "
            print(f"{r:>+12.4f}{marker} ", end="")
        print()

    # ── Inter-model agreement ───────────────────────────────────────
    print("\n" + "=" * 90)
    print("INTER-MODEL CORRELATION  (do models agree on WHICH images are hard?)")
    print("=" * 90)

    prob_matrix = merged[model_cols].values  # (N, 5)
    print(f"\n{'':>12s}", end="")
    for mcol in model_cols:
        short = model_info.get(mcol, {}).get("name", mcol)[:8]
        print(f"{short:>10s}", end="")
    print()

    for i, mcol_i in enumerate(model_cols):
        short_i = model_info.get(mcol_i, {}).get("name", mcol_i)[:8]
        print(f"{short_i:>12s}", end="")
        for j, mcol_j in enumerate(model_cols):
            r, _ = stats.spearmanr(prob_matrix[:, i], prob_matrix[:, j])
            print(f"{r:>10.3f}", end="")
        print()

    # ── Verdict ─────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)

    # Check if all models have strong sharpness correlation
    sharp_col = "sharpness_laplacian_var_spearman"
    all_corrs = results_df[sharp_col].values
    mean_corr = np.mean(np.abs(all_corrs))

    if mean_corr > 0.25:
        print(f"✅ ALL models correlate with sharpness (mean |ρ| = {mean_corr:.3f})")
        print("   → This is a SYSTEMIC data issue, not model-specific.")
        print("   → The training distribution teaches this shortcut regardless of config.")
    else:
        print(f"⚠️  Mixed results (mean |ρ| = {mean_corr:.3f})")

    # Check if B16_old is different
    b16_corr = results_df[results_df["model"] == "B16_old"][sharp_col].values[0]
    f1_corr = results_df[results_df["model"] == "R25_F1"][sharp_col].values[0]
    if abs(b16_corr) < abs(f1_corr) * 0.7:
        print(f"\n🔍 B16_old has weaker sharpness correlation ({b16_corr:+.3f} vs R25_F1 {f1_corr:+.3f})")
        print("   → B16_old may have partially learned real forgery features.")
        print("   → Its higher WMA accuracy (55.7%) may not be just luck.")
    else:
        print(f"\n   B16_old and R25_F1 have similar sharpness correlation")
        print(f"   ({b16_corr:+.3f} vs {f1_corr:+.3f}) — both use the same shortcut.")

    # ── Save ────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "V6_cross_model_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n✅ Results saved to: {out_path}")


if __name__ == "__main__":
    main()
