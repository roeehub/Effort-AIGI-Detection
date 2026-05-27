"""
P21 — decision-grade operating-point table.

For each model (P8A, P18T, P18C), at each real_FPR floor in {1, 2, 3, 5, 7, 10, 15, 20}%:
  - Pick τ that yields exactly that FPR on teams_real_all_dev (or smallest τ such that FPR ≤ floor).
  - Report recall on every fake suite.
  - Report FPR on stress suites + lockbox real.

Identifies the FPR elbow where each fake suite "unlocks".

Output: outputs/p21_operating_point_full.csv + figures/p21_recall_vs_fpr.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("analysis/score_distribution_2026-05-02/outputs")
OUT = Path("analysis/cpu_decision_2026-05-02_pm_late/outputs")
FIG = Path("analysis/cpu_decision_2026-05-02_pm_late/figures")
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(ROOT / "combined_frames.parquet")

REAL_PRIMARY = "teams_real_all_dev"
FAKE_SUITES = [
    "teams_fake_all_dev",
    "teams_fake_all_lockbox",
    "visomaster_enhanced_macro_dev",
    "deeplive_enhanced_dev",
]
REAL_STRESS = ["teams_real_dor_dev", "teams_real_lighting_extreme_dev", "teams_real_poor_quality_dev"]
LOCKBOX_REAL = "teams_real_all_lockbox"

FPR_FLOORS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]


def threshold_for_fpr(reals_scores: np.ndarray, target_fpr: float) -> float:
    """Largest τ such that FPR(τ) ≤ target_fpr.  FPR(τ) = mean(score >= τ)."""
    if len(reals_scores) == 0:
        return 1.0
    sorted_desc = np.sort(reals_scores)[::-1]
    n = len(sorted_desc)
    k = int(np.floor(target_fpr * n))
    if k <= 0:
        return float(sorted_desc[0]) + 1e-9
    if k >= n:
        return 0.0
    return float(sorted_desc[k - 1] + 1e-12)  # pick threshold just above the k-th largest


rows = []
for model in ["P8A", "P18T", "P18C"]:
    sub = df[df.model == model]
    reals = sub[sub.suite == REAL_PRIMARY].frame_prob.to_numpy()
    for floor in FPR_FLOORS:
        tau = threshold_for_fpr(reals, floor)
        actual_fpr = float((reals >= tau).mean())
        row = {"model": model, "fpr_floor": floor, "tau": tau, "actual_primary_fpr": actual_fpr}
        for s in FAKE_SUITES:
            ss = sub[sub.suite == s].frame_prob.to_numpy()
            row[f"{s}_recall"] = float((ss >= tau).mean()) if len(ss) else np.nan
            row[f"{s}_n"] = len(ss)
        for s in REAL_STRESS:
            ss = sub[sub.suite == s].frame_prob.to_numpy()
            row[f"{s}_FPR"] = float((ss >= tau).mean()) if len(ss) else np.nan
            row[f"{s}_n"] = len(ss)
        ll = sub[sub.suite == LOCKBOX_REAL].frame_prob.to_numpy()
        row[f"{LOCKBOX_REAL}_FPR"] = float((ll >= tau).mean()) if len(ll) else np.nan
        row[f"{LOCKBOX_REAL}_n"] = len(ll)
        rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(OUT / "p21_operating_point_full.csv", index=False)
print("wrote", OUT / "p21_operating_point_full.csv")
print()

# Pretty print key columns
key_cols = ["model", "fpr_floor", "tau",
            "visomaster_enhanced_macro_dev_recall",
            "deeplive_enhanced_dev_recall",
            "teams_fake_all_dev_recall",
            "teams_fake_all_lockbox_recall",
            "teams_real_dor_dev_FPR",
            "teams_real_all_lockbox_FPR"]
print(out[key_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Plot: recall vs FPR floor for each (model, fake suite)
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
suite_titles = {
    "visomaster_enhanced_macro_dev": "viso_enhanced_macro (production blocker)",
    "deeplive_enhanced_dev": "deeplive_enhanced (single-id, all dor)",
    "teams_fake_all_dev": "teams_fake_all (heterogeneous)",
    "teams_fake_all_lockbox": "teams_fake_lockbox (held-out)",
}
markers = {"P8A": "o", "P18T": "s", "P18C": "^"}
colors = {"P8A": "tab:blue", "P18T": "tab:orange", "P18C": "tab:red"}

for ax, (suite, title) in zip(axes.flat, suite_titles.items()):
    for model in ["P8A", "P18T", "P18C"]:
        sub = out[out.model == model]
        ax.plot(sub.fpr_floor * 100, sub[f"{suite}_recall"] * 100,
                marker=markers[model], color=colors[model], label=model, linewidth=2)
    ax.set_xlabel("primary real FPR floor (%)")
    ax.set_ylabel("recall (%)")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    # Mark contract operating point
    ax.axvline(2.0, color="grey", linestyle="--", alpha=0.5)
    ax.text(2.05, 5, "contract\nτ", fontsize=8, color="grey")

plt.suptitle("P21 — Recall vs FPR floor (decision grid)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG / "p21_recall_vs_fpr_floor.png", dpi=140, bbox_inches="tight")
print("wrote", FIG / "p21_recall_vs_fpr_floor.png")

# Elbow analysis: at each suite × model, find the FPR floor where recall first crosses 30%, 50%, 70%
elbow_rows = []
for model in ["P8A", "P18T", "P18C"]:
    sub = out[out.model == model].sort_values("fpr_floor")
    for suite in FAKE_SUITES:
        recalls = sub[f"{suite}_recall"].to_numpy()
        floors = sub.fpr_floor.to_numpy()
        for thresh in [0.30, 0.50, 0.70]:
            idx = np.where(recalls >= thresh)[0]
            crossover = float(floors[idx[0]]) if len(idx) > 0 else None
            elbow_rows.append({
                "model": model,
                "suite": suite,
                "recall_threshold": thresh,
                "min_fpr_to_reach": crossover,
            })

elbows = pd.DataFrame(elbow_rows)
elbows.to_csv(OUT / "p21_elbow_table.csv", index=False)
print("\nElbows (min FPR floor to reach recall threshold):")
print(elbows.to_string(index=False))
