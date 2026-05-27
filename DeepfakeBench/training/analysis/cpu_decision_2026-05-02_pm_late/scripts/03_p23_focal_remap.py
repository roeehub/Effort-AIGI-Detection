"""
P23 pre-validation — focal-style score remapping.

Take existing per-frame scores. Apply a γ-style transform that mimics what
focal loss training would do at inference time:
  - σ' = σ^γ / (σ^γ + (1-σ)^γ)   for γ ∈ {1, 1.5, 2, 3, 5, 8}
  - This is the focal-loss inverse-link: monotone, but pushes the τ-tail.

Recompute recall vs FPR for each γ.

Critical falsifier: if no γ moves the recall-FPR curve favorably (i.e., the
existing score *ranking* is already locally optimal and only re-thresholding it
won't help), then training with focal loss won't help either — the issue isn't
calibration, it's the underlying score ranking.

Note: monotone transforms preserve AUC by construction. What focal loss can do
during *training* is push the ranking itself; this CPU pre-val tests whether the
inference-time remapping shape is the bottleneck. If yes → P23 likely helps.
If no → P23 won't help at the contract τ; the issue is ranking, not shape.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("analysis/score_distribution_2026-05-02/outputs")
OUT = Path("analysis/cpu_decision_2026-05-02_pm_late/outputs")
FIG = Path("analysis/cpu_decision_2026-05-02_pm_late/figures")

df = pd.read_parquet(ROOT / "combined_frames.parquet")

REAL_PRIMARY = "teams_real_all_dev"
FAKE_SUITES = ["visomaster_enhanced_macro_dev", "deeplive_enhanced_dev",
               "teams_fake_all_dev", "teams_fake_all_lockbox"]


def focal_remap(s, gamma):
    s = np.clip(s, 1e-9, 1 - 1e-9)
    num = s ** gamma
    return num / (num + (1 - s) ** gamma)


def threshold_for_fpr(reals, target_fpr):
    if len(reals) == 0:
        return 1.0
    sorted_desc = np.sort(reals)[::-1]
    n = len(sorted_desc)
    k = int(np.floor(target_fpr * n))
    if k <= 0:
        return float(sorted_desc[0]) + 1e-9
    if k >= n:
        return 0.0
    return float(sorted_desc[k - 1] + 1e-12)


GAMMAS = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
FLOORS = [0.02, 0.05, 0.10]

rows = []
for model in ["P8A", "P18T", "P18C"]:
    sub = df[df.model == model]
    reals = sub[sub.suite == REAL_PRIMARY].frame_prob.to_numpy()
    for gamma in GAMMAS:
        for floor in FLOORS:
            reals_remap = focal_remap(reals, gamma)
            tau = threshold_for_fpr(reals_remap, floor)
            row = {"model": model, "gamma": gamma, "fpr_floor": floor, "tau": tau}
            for s in FAKE_SUITES:
                ss = sub[sub.suite == s].frame_prob.to_numpy()
                if len(ss) == 0:
                    row[f"{s}_recall"] = np.nan
                    continue
                ss_remap = focal_remap(ss, gamma)
                row[f"{s}_recall"] = float((ss_remap >= tau).mean())
            rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(OUT / "p23_focal_remap.csv", index=False)
print("wrote", OUT / "p23_focal_remap.csv")
print()

# Show: for each (model, suite, fpr_floor), is the recall *invariant* across γ?
# If yes — focal won't help (issue is ranking, not shape).
# If no — focal might help (shape changes recall at fixed FPR).
key_cols = ["model", "gamma", "fpr_floor",
            "visomaster_enhanced_macro_dev_recall",
            "deeplive_enhanced_dev_recall",
            "teams_fake_all_dev_recall",
            "teams_fake_all_lockbox_recall"]
print(out[key_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Sanity: is recall the same across γ at fixed FPR floor?
print("\n--- INVARIANCE TEST ---")
print("If recall is ~constant across γ at fixed (model, fpr_floor), focal-via-remap is uninformative.")
print("This is expected because monotone transforms preserve ranking AND fpr_floor is satisfied via re-fit τ.")
print("The point of this analysis: if ANY transform shape changes recall, focal training has a chance.")
print()

for model in ["P8A", "P18T", "P18C"]:
    for floor in FLOORS:
        sub = out[(out.model == model) & (out.fpr_floor == floor)]
        if len(sub) == 0:
            continue
        for s in FAKE_SUITES:
            recalls = sub[f"{s}_recall"].to_numpy()
            if len(recalls) > 0:
                spread = recalls.max() - recalls.min()
                print(f"  {model:5s} fpr={floor:.2f}  {s:35s} recall range: [{recalls.min():.4f}, {recalls.max():.4f}]  spread={spread:.4f}")

# Plot: recall vs gamma at fpr=0.02
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for model in ["P8A", "P18T", "P18C"]:
    sub = out[(out.model == model) & (out.fpr_floor == 0.02)]
    ax.plot(sub.gamma, sub["visomaster_enhanced_macro_dev_recall"] * 100, marker="o", label=model)
ax.set_xlabel("γ (focal remap exponent)")
ax.set_ylabel("viso_enhanced_macro recall (%)")
ax.set_title("P23 remap test — viso recall @ FPR=2% across γ")
ax.legend()
ax.grid(alpha=0.3)
ax.set_xscale("log")

ax = axes[1]
for model in ["P8A", "P18T", "P18C"]:
    sub = out[(out.model == model) & (out.fpr_floor == 0.02)]
    ax.plot(sub.gamma, sub["deeplive_enhanced_dev_recall"] * 100, marker="s", label=model)
ax.set_xlabel("γ")
ax.set_ylabel("deeplive_enhanced recall (%)")
ax.set_title("P23 remap test — deeplive recall @ FPR=2% across γ")
ax.legend()
ax.grid(alpha=0.3)
ax.set_xscale("log")

plt.suptitle("P23 pre-val — does focal-shape remap move recall at fixed FPR?", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG / "p23_focal_remap_recall.png", dpi=140, bbox_inches="tight")
print(f"\nwrote {FIG / 'p23_focal_remap_recall.png'}")
