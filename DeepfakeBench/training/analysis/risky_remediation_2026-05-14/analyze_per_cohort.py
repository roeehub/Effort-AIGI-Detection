"""Per-cohort analysis: where does universal blend@0.50 help vs regress?

Reads outputs/all_cohorts_scored.csv (T5C_orig, T5C_blend_050, P8A_orig,
P8A_blend_050 columns + IQ features).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

OUT = Path(__file__).resolve().parent / "outputs"
df = pd.read_csv(OUT / "all_cohorts_scored.csv")
print(f"loaded {len(df)} frames across {df['suite'].nunique()} suites")
print()

# A "cohort" here = suite × label (since some cohorts mix labels)
print("Cohort overview (n, mean_lap_var, mean_luma):")
ov = df.groupby(["suite", "label"]).agg(
    n=("local", "count"),
    lap_var_mean=("iq_lap_var", "mean"),
    luma_mean=("iq_luma", "mean"),
    color_b_mean=("iq_lab_b_dev", "mean"),
).round(1)
print(ov.to_string())
print()


def recall_at_fpr(scores, labels, target_fpr=0.05):
    fpr, tpr, thr = roc_curve(labels, scores)
    ok = fpr <= target_fpr
    if ok.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    idx = np.where(ok)[0][-1]
    return thr[idx], fpr[idx], tpr[idx]


# ============================================================================
# Per-suite AUC + recall@FPR
# ============================================================================
print("=" * 102)
print("PER-SUITE AUC and recall@5%FPR for orig vs blend@0.50")
print("=" * 102)
rows = []
for suite in sorted(df["suite"].unique()):
    sub = df[df["suite"] == suite]
    if sub["label"].nunique() < 2:
        # Single-label cohort — can't compute AUC. Show score-shift stats instead.
        only_label = int(sub["label"].iloc[0])
        n = len(sub)
        row = {"suite": suite, "n": n, "label": "real-only" if only_label == 0 else "fake-only"}
        for ckpt in ["T5C", "P8A"]:
            row[f"{ckpt}_mean_orig"] = sub[f"{ckpt}_orig"].mean()
            row[f"{ckpt}_mean_blend"] = sub[f"{ckpt}_blend_050"].mean()
            row[f"{ckpt}_Δmean"] = sub[f"{ckpt}_blend_050"].mean() - sub[f"{ckpt}_orig"].mean()
            # frac with high score (regression signal for real, win signal for fake)
            high_o = (sub[f"{ckpt}_orig"] > 0.5).mean()
            high_b = (sub[f"{ckpt}_blend_050"] > 0.5).mean()
            row[f"{ckpt}_frac>0.5_orig"] = high_o
            row[f"{ckpt}_frac>0.5_blend"] = high_b
            row[f"{ckpt}_Δfrac>0.5"] = high_b - high_o
        rows.append(row)
        continue
    n = len(sub)
    n_real = (sub["label"] == 0).sum()
    n_fake = (sub["label"] == 1).sum()
    row = {"suite": suite, "n": n, "label": f"{n_real}r/{n_fake}f"}
    for ckpt in ["T5C", "P8A"]:
        auc_o = roc_auc_score(sub["label"], sub[f"{ckpt}_orig"])
        auc_b = roc_auc_score(sub["label"], sub[f"{ckpt}_blend_050"])
        row[f"{ckpt}_AUC_orig"] = auc_o
        row[f"{ckpt}_ΔAUC"] = auc_b - auc_o
        # recall@5%FPR
        _, _, ro = recall_at_fpr(sub[f"{ckpt}_orig"], sub["label"], 0.05)
        _, _, rb = recall_at_fpr(sub[f"{ckpt}_blend_050"], sub["label"], 0.05)
        row[f"{ckpt}_recall@5_orig"] = ro
        row[f"{ckpt}_Δrecall@5"] = rb - ro
    rows.append(row)
rd = pd.DataFrame(rows)
print()
for ckpt in ["T5C", "P8A"]:
    cols = ["suite", "n", "label"]
    extra = [c for c in rd.columns if c.startswith(f"{ckpt}_")]
    print(f"\n### {ckpt}")
    print(rd[cols + extra].to_string(index=False))
print()
rd.to_csv(OUT / "per_cohort_analysis.csv", index=False)
print(f"wrote per-cohort summary -> {OUT / 'per_cohort_analysis.csv'}")

# ============================================================================
# Cross-substrate verdict — single-label cohorts (real or fake only)
# These can't compute AUC but show whether blend SHIFTS scores in the right direction.
# ============================================================================
print()
print("=" * 102)
print("CROSS-SUBSTRATE SCORE SHIFTS (single-label cohorts, where ground truth is uniform)")
print("=" * 102)
print()
print("If real-only cohort: NEGATIVE Δmean means blend correctly drops the FP scores (GOOD)")
print("If fake-only cohort: POSITIVE Δmean means blend correctly lifts the TP scores (GOOD)")
print()
single = []
for suite in sorted(df["suite"].unique()):
    sub = df[df["suite"] == suite]
    if sub["label"].nunique() != 1:
        continue
    only_label = int(sub["label"].iloc[0])
    pool_type = "real-only (lower=better)" if only_label == 0 else "fake-only (higher=better)"
    for ckpt in ["T5C", "P8A"]:
        d = sub[f"{ckpt}_blend_050"] - sub[f"{ckpt}_orig"]
        # Direction-aware sign: for real, "good direction" = negative; for fake, positive
        good_direction = (d < 0).mean() if only_label == 0 else (d > 0).mean()
        single.append({
            "suite": suite, "ckpt": ckpt, "type": pool_type, "n": len(sub),
            "orig_mean": sub[f"{ckpt}_orig"].mean(),
            "blend_mean": sub[f"{ckpt}_blend_050"].mean(),
            "Δmean": d.mean(),
            "good_dir_frac": good_direction,
            "high_orig": (sub[f"{ckpt}_orig"] > 0.5).mean(),
            "high_blend": (sub[f"{ckpt}_blend_050"] > 0.5).mean(),
        })
sd = pd.DataFrame(single)
sd = sd.round(4)
print(sd.to_string(index=False))
sd.to_csv(OUT / "single_label_cohorts.csv", index=False)
print()
print(f"wrote single-label cohort shifts -> {OUT / 'single_label_cohorts.csv'}")
