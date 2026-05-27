"""
Failure-mode attribution.

For each missed fake at the contract τ (per model), classify which shortcut
it 'failed via':
  - low-sharpness (laplacian < eval_p25_among_caught_fakes)
  - dark (luma_mean < threshold)
  - low-skin (skin_frac < threshold)
  - none-of-the-above (the model just disagrees for unrelated reasons)

This gives the user a quantified picture of how much of the production
blocker (1.1% viso recall) would be "attacked" by P22 augmentation
specifically, vs how much would remain after the shortcut is closed.

Output: percent of misses each shortcut accounts for, per (model × suite).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ATTR = pd.read_csv("analysis/score_distribution_2026-05-02/outputs/cross_suite_attributes.csv")
OUT = Path("analysis/cpu_decision_2026-05-02_pm_late/outputs")
FIG = Path("analysis/cpu_decision_2026-05-02_pm_late/figures")

# Define caught/missed per model at "deployment τ" (FPR=2% on teams_real_all_dev)
# We need to compute τ on full reals from combined_frames.parquet then apply to attrs sample
all_frames = pd.read_parquet("analysis/score_distribution_2026-05-02/outputs/combined_frames.parquet")


def threshold_for_fpr(reals, target_fpr):
    sorted_desc = np.sort(reals)[::-1]
    n = len(sorted_desc)
    k = int(np.floor(target_fpr * n))
    if k <= 0: return float(sorted_desc[0]) + 1e-9
    if k >= n: return 0.0
    return float(sorted_desc[k - 1] + 1e-12)


taus = {}
for m in ["P8A", "P18T", "P18C"]:
    reals = all_frames[(all_frames.model == m) & (all_frames.suite == "teams_real_all_dev")].frame_prob.to_numpy()
    taus[m] = threshold_for_fpr(reals, 0.02)
print("τ at FPR=2% (computed on full reals):", taus)

# Restrict to fake-labeled frames in attrs sample
fake_suites = ["teams_fake_all_dev", "teams_fake_all_lockbox", "deeplive_enhanced_dev"]
fake_attrs = ATTR[ATTR.suite.isin(fake_suites)].copy()
print(f"{len(fake_attrs)} fake frames in attrs sample")

# Compute the "caught" reference distribution for each shortcut (the fakes that ARE caught)
shortcut_attrs = ["laplacian_var", "luma_mean", "skin_frac"]


rows = []
for model in ["P8A", "P18T", "P18C"]:
    score_col = f"score_{model}"
    fake_attrs[f"caught_{model}"] = fake_attrs[score_col] >= taus[model]

    for suite in fake_suites:
        sub = fake_attrs[fake_attrs.suite == suite]
        n = len(sub)
        n_caught = int(sub[f"caught_{model}"].sum())
        n_missed = n - n_caught
        if n == 0 or n_missed == 0:
            continue
        # Reference distribution: caught-fake percentiles
        caught = sub[sub[f"caught_{model}"]]
        missed = sub[~sub[f"caught_{model}"]]
        if len(caught) == 0:
            # If nothing was caught, use this suite's "globally-caught" stats from any model
            caught_global = fake_attrs[fake_attrs[[f"caught_{m}" for m in ['P8A','P18T','P18C']]].any(axis=1)]
            ref = caught_global if len(caught_global) > 0 else sub
        else:
            ref = caught
        thresholds = {a: ref[a].quantile(0.25) for a in shortcut_attrs}
        # Classify each missed frame
        for _, row in missed.iterrows():
            fail_modes = []
            if row.laplacian_var < thresholds["laplacian_var"]: fail_modes.append("low_sharpness")
            if row.luma_mean < thresholds["luma_mean"]: fail_modes.append("dark")
            if row.skin_frac < thresholds["skin_frac"]: fail_modes.append("low_skin")
            tag = "+".join(fail_modes) if fail_modes else "none"
            rows.append({
                "model": model, "suite": suite,
                "frame_path": row.frame_path,
                "score": row[score_col],
                "laplacian": row.laplacian_var,
                "luma": row.luma_mean,
                "skin": row.skin_frac,
                "fail_modes": tag,
            })

attribution = pd.DataFrame(rows)
attribution.to_csv(OUT / "failure_mode_attribution.csv", index=False)
print(f"\nwrote {OUT / 'failure_mode_attribution.csv'}: {len(attribution)} missed fakes attributed")

# Summary: what fraction of misses each shortcut accounts for
summary = attribution.groupby(["model", "suite", "fail_modes"]).size().unstack(fill_value=0)
print("\nFraction of missed fakes by shortcut tag:")
print(summary.to_string())

# Aggregate per-shortcut prevalence
prevalence_rows = []
for model in ["P8A", "P18T", "P18C"]:
    for suite in fake_suites:
        sub = attribution[(attribution.model == model) & (attribution.suite == suite)]
        if len(sub) == 0: continue
        n = len(sub)
        prevalence_rows.append({
            "model": model, "suite": suite, "n_missed": n,
            "pct_low_sharpness": (sub.fail_modes.str.contains("low_sharpness")).mean(),
            "pct_dark": (sub.fail_modes.str.contains("dark")).mean(),
            "pct_low_skin": (sub.fail_modes.str.contains("low_skin")).mean(),
            "pct_any_shortcut": (sub.fail_modes != "none").mean(),
            "pct_none": (sub.fail_modes == "none").mean(),
        })
prev = pd.DataFrame(prevalence_rows)
prev.to_csv(OUT / "failure_mode_prevalence.csv", index=False)
print("\nPrevalence of each failure mode among missed fakes:")
print(prev.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, model in zip(axes, ["P8A", "P18T", "P18C"]):
    sub = prev[prev.model == model]
    if len(sub) == 0: continue
    suites = sub.suite.tolist()
    pcts = sub[["pct_low_sharpness", "pct_dark", "pct_low_skin", "pct_none"]].to_numpy() * 100
    bottom = np.zeros(len(suites))
    colors_ = ["tab:red", "tab:purple", "tab:orange", "tab:gray"]
    labels_ = ["low_sharpness", "dark", "low_skin", "none-of-the-above"]
    for i, (col, label) in enumerate(zip(colors_, labels_)):
        ax.bar(suites, pcts[:, i], bottom=bottom, color=col, label=label)
        bottom += pcts[:, i]
    ax.set_title(f"{model}: failure-mode attribution")
    ax.set_ylabel("% of missed fakes (overlapping)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

plt.suptitle("Failure-mode attribution at deployment τ (FPR=2%)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG / "failure_mode_attribution.png", dpi=140, bbox_inches="tight")
print(f"\nwrote {FIG / 'failure_mode_attribution.png'}")
