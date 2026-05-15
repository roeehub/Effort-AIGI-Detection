"""Analyze the resolution-chain probe outputs.

Inputs:
  outputs/per_frame_per_variant.parquet
  outputs/per_frame_summary.parquet

Headlines (per ckpt):
  - distribution of score_range across 388 frames (median + p90)
  - per-cohort + per-identity breakdown of score_range
  - 2-way ANOVA: how much variance is explained by down_size vs kernel
  - per-frame ckpt-disagreement: stdev across ckpts at the same variant
  - flip rate: how often does a perturbation cross the 0.5 boundary

Outputs:
  outputs/range_by_cohort.csv
  outputs/range_by_identity.csv
  outputs/anova_per_ckpt.csv
  outputs/flip_rate_per_ckpt.csv
  outputs/ckpt_disagreement_per_frame.csv
  outputs/per_size_mean_score.csv
  RESULTS_FACTS_2026-05-15.md (manually written from these)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR.parent / "outputs"
FULL_PATH = OUT_DIR / "per_frame_per_variant.parquet"
SUM_PATH = OUT_DIR / "per_frame_summary.parquet"

full = pd.read_parquet(FULL_PATH)
summary = pd.read_parquet(SUM_PATH)

print(f"\n=== Loaded ===")
print(f"  full   : {len(full)} rows ({full['ckpt'].nunique()} ckpts x {full['variant_key'].nunique()} variants x {full['frame_path'].nunique()} frames)")
print(f"  summary: {len(summary)} rows")

# 1. Per-ckpt, per-cohort score_range distribution.
print(f"\n=== 1. Score range distribution per (ckpt, cohort) ===")
agg = summary.groupby(["ckpt", "cohort"]).agg(
    n=("score_range", "size"),
    range_p50=("score_range", lambda s: s.quantile(0.5)),
    range_p90=("score_range", lambda s: s.quantile(0.9)),
    range_max=("score_range", "max"),
    std_p50=("score_std", lambda s: s.quantile(0.5)),
    abs_res_corr_p50=("res_corr", lambda s: s.abs().quantile(0.5)),
).reset_index()
agg.to_csv(OUT_DIR / "range_by_cohort.csv", index=False)
print(agg.to_string(index=False))

# 2. Per-ckpt, per-identity score_range (for the chronic-6 cohort)
print(f"\n=== 2. Range by identity (top by p90) ===")
agg_id = summary.groupby(["ckpt", "identity"]).agg(
    n=("score_range", "size"),
    range_p50=("score_range", lambda s: s.quantile(0.5)),
    range_p90=("score_range", lambda s: s.quantile(0.9)),
    score_baseline_p50=("score_baseline", "median"),
    score_min_p50=("score_min", "median"),
    score_max_p50=("score_max", "median"),
).reset_index()
agg_id.to_csv(OUT_DIR / "range_by_identity.csv", index=False)
# Show only the identities present in each ckpt sorted by range_p90 desc, top 15 per ckpt
for c in agg_id["ckpt"].unique():
    sub = agg_id[agg_id["ckpt"] == c].sort_values("range_p90", ascending=False).head(15)
    print(f"--- {c} ---")
    print(sub.to_string(index=False))

# 3. 2-way ANOVA per ckpt: how much variance is explained by down_size vs kernel
print(f"\n=== 3. 2-way variance attribution (down_size vs kernel) per ckpt ===")
perturbed = full[full["down_size"] != -1].copy()
attribution_rows = []
for c in perturbed["ckpt"].unique():
    sub = perturbed[perturbed["ckpt"] == c].copy()
    # For each frame, compute SS due to down_size, SS due to kernel, total SS
    rows = []
    for fp, g in sub.groupby("frame_path"):
        if len(g) != 20:
            continue
        grand_mean = g["score"].mean()
        ss_total = ((g["score"] - grand_mean) ** 2).sum()
        # SS down_size: between-size variance (averaged over kernels)
        size_means = g.groupby("down_size")["score"].mean()
        ss_size = sum((size_means[s] - grand_mean) ** 2 * 4 for s in size_means.index)
        # SS kernel: between-kernel variance (averaged over sizes)
        kernel_means = g.groupby("kernel")["score"].mean()
        ss_kernel = sum((kernel_means[k] - grand_mean) ** 2 * 5 for k in kernel_means.index)
        rows.append({
            "frame_path": fp,
            "ckpt": c,
            "ss_size": ss_size,
            "ss_kernel": ss_kernel,
            "ss_total": ss_total,
            "size_frac": ss_size / ss_total if ss_total > 0 else 0,
            "kernel_frac": ss_kernel / ss_total if ss_total > 0 else 0,
        })
    if rows:
        rdf = pd.DataFrame(rows)
        attribution_rows.append({
            "ckpt": c,
            "n_frames": len(rdf),
            "size_frac_mean": rdf["size_frac"].mean(),
            "size_frac_p50": rdf["size_frac"].median(),
            "kernel_frac_mean": rdf["kernel_frac"].mean(),
            "kernel_frac_p50": rdf["kernel_frac"].median(),
            "interaction_frac_mean": 1 - rdf["size_frac"].mean() - rdf["kernel_frac"].mean(),
        })
anova = pd.DataFrame(attribution_rows)
anova.to_csv(OUT_DIR / "anova_per_ckpt.csv", index=False)
print(anova.to_string(index=False))

# 4. Flip rate: how often does ANY perturbation flip score across 0.5
print(f"\n=== 4. Flip rate (does ANY perturbation push score across 0.5?) ===")
flip_rows = []
for c in summary["ckpt"].unique():
    s = summary[summary["ckpt"] == c].copy()
    flipped = ((s["score_baseline"] < 0.5) & (s["score_max"] >= 0.5)) | \
              ((s["score_baseline"] >= 0.5) & (s["score_min"] < 0.5))
    flip_rows.append({
        "ckpt": c,
        "n_frames": len(s),
        "flip_n": int(flipped.sum()),
        "flip_rate": float(flipped.mean()),
        # Threshold 0.49 (T5C ship τ from memory project_blend_unsharp_lever_2026-05-14)
        "flip_at_0.49_n": int((((s["score_baseline"] < 0.49) & (s["score_max"] >= 0.49)) |
                               ((s["score_baseline"] >= 0.49) & (s["score_min"] < 0.49))).sum()),
        "flip_at_0.7_n": int((((s["score_baseline"] < 0.7) & (s["score_max"] >= 0.7)) |
                              ((s["score_baseline"] >= 0.7) & (s["score_min"] < 0.7))).sum()),
        "flip_at_0.9_n": int((((s["score_baseline"] < 0.9) & (s["score_max"] >= 0.9)) |
                              ((s["score_baseline"] >= 0.9) & (s["score_min"] < 0.9))).sum()),
    })
flips = pd.DataFrame(flip_rows)
flips.to_csv(OUT_DIR / "flip_rate_per_ckpt.csv", index=False)
print(flips.to_string(index=False))

# 5. Per-size mean score: does score systematically increase/decrease as we
#    downsample more aggressively?
print(f"\n=== 5. Per-size mean score (real-cohort only, fake-cohort only) ===")
for label_val, lbl in [(0, "REALS"), (1, "FAKES")]:
    sub = perturbed[perturbed["label"] == label_val]
    means = sub.groupby(["ckpt", "down_size"])["score"].agg(["mean", "median", "std"]).reset_index()
    means.to_csv(OUT_DIR / f"per_size_mean_score_{lbl.lower()}.csv", index=False)
    print(f"--- {lbl} (label={label_val}, n={(summary['label']==label_val).sum()}) ---")
    print(means.to_string(index=False))

# 6. Cross-ckpt disagreement at the same variant: per (frame, variant) stdev
print(f"\n=== 6. Per-frame cross-ckpt disagreement ===")
wide = full.pivot_table(index=["frame_path", "variant_key", "cohort", "identity", "label"],
                        columns="ckpt", values="score").reset_index()
ckpt_cols = [c for c in wide.columns if c not in ["frame_path", "variant_key", "cohort", "identity", "label"]]
wide["ckpt_std"] = wide[ckpt_cols].std(axis=1)
# Per frame across all 21 variants
disagree = wide.groupby(["frame_path", "cohort", "identity", "label"]).agg(
    ckpt_std_max=("ckpt_std", "max"),
    ckpt_std_p50=("ckpt_std", "median"),
).reset_index()
disagree.to_csv(OUT_DIR / "ckpt_disagreement_per_frame.csv", index=False)
print(disagree.groupby("cohort")[["ckpt_std_max", "ckpt_std_p50"]].describe().to_string())

print("\nDONE — wrote 6 CSVs to outputs/.")
