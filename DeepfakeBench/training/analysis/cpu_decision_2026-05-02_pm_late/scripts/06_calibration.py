"""
Calibration / reliability diagrams.

Per (model × suite): bin scores into deciles, compute empirical fake rate
in each bin, compare to predicted (the bin's mean score). If well-calibrated,
empirical = predicted.

Also computes ECE (expected calibration error).

Critical for P23: focal loss reshapes the *score distribution* but does NOT
necessarily improve ranking. If scores are already well-calibrated at τ-tail
but recall is low, focal won't help. If scores are mis-calibrated (e.g.,
many "true positives" sit at score 0.6 instead of 0.99), focal could.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("analysis/cpu_decision_2026-05-02_pm_late/outputs")
FIG = Path("analysis/cpu_decision_2026-05-02_pm_late/figures")

df = pd.read_parquet("analysis/score_distribution_2026-05-02/outputs/combined_frames.parquet")

# Build a real-vs-fake mix per (model × {teams_real_all_dev + teams_fake_all_dev})
def reliability(scores, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (scores >= lo) & (scores < hi if i < n_bins - 1 else scores <= hi)
        if mask.sum() == 0: continue
        out.append({
            "bin_lo": lo, "bin_hi": hi,
            "mean_score": float(scores[mask].mean()),
            "empirical_rate": float(labels[mask].mean()),
            "n": int(mask.sum())
        })
    return pd.DataFrame(out)

def ece(scores, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(scores)
    err = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (scores >= lo) & (scores < hi if i < n_bins - 1 else scores <= hi)
        if mask.sum() == 0: continue
        err += (mask.sum() / n) * abs(scores[mask].mean() - labels[mask].mean())
    return err

# For each model, build (real, fake) mix and compute reliability
results = {}
for model in ["P8A", "P18T", "P18C"]:
    sub = df[df.model == model]
    real = sub[sub.suite == "teams_real_all_dev"]
    out_per_suite = {}
    for fake_suite in ["teams_fake_all_dev", "visomaster_enhanced_macro_dev",
                       "deeplive_enhanced_dev", "teams_fake_all_lockbox"]:
        fake = sub[sub.suite == fake_suite]
        if len(fake) == 0: continue
        scores = np.concatenate([real.frame_prob.to_numpy(), fake.frame_prob.to_numpy()])
        labels = np.concatenate([np.zeros(len(real)), np.ones(len(fake))])
        rel = reliability(scores, labels, n_bins=10)
        e = ece(scores, labels, n_bins=10)
        out_per_suite[fake_suite] = (rel, e)
    results[model] = out_per_suite

# Save
all_rows = []
for model, by_suite in results.items():
    for suite, (rel, e) in by_suite.items():
        rel = rel.copy()
        rel["model"] = model
        rel["suite"] = suite
        rel["ece"] = e
        all_rows.append(rel)
combined = pd.concat(all_rows, ignore_index=True)
combined.to_csv(OUT / "reliability_diagrams.csv", index=False)

# ECE summary
ece_rows = []
for model, by_suite in results.items():
    for suite, (rel, e) in by_suite.items():
        ece_rows.append({"model": model, "suite": suite, "ECE": e})
ece_df = pd.DataFrame(ece_rows)
print("Expected Calibration Error (ECE) — lower is better-calibrated:")
print(ece_df.pivot(index="suite", columns="model", values="ECE").to_string(float_format=lambda x: f"{x:.4f}"))
ece_df.to_csv(OUT / "ece_per_model_suite.csv", index=False)

# Plot reliability for the production-blocker suites
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
suites = ["teams_fake_all_dev", "visomaster_enhanced_macro_dev",
          "deeplive_enhanced_dev", "teams_fake_all_lockbox"]
for j, suite in enumerate(suites):
    ax = axes[0, j]
    for model in ["P8A", "P18T", "P18C"]:
        rel = results[model].get(suite, [None])[0]
        if rel is None: continue
        ax.plot(rel.mean_score, rel.empirical_rate, marker="o", label=f"{model} (ECE={results[model][suite][1]:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.5)
    ax.set_xlabel("predicted (bin mean)")
    ax.set_ylabel("empirical fake rate")
    ax.set_title(suite)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # row 2: bin counts (where do scores live?)
    ax2 = axes[1, j]
    for model in ["P8A", "P18T", "P18C"]:
        rel = results[model].get(suite, [None])[0]
        if rel is None: continue
        ax2.bar(np.arange(len(rel)) + ["P8A", "P18T", "P18C"].index(model) * 0.25,
                rel.n, width=0.22, label=model, alpha=0.8)
    ax2.set_xlabel("score bin (0..10)")
    ax2.set_ylabel("frame count")
    ax2.set_title(f"score distribution — {suite}")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_yscale("log")

plt.suptitle("Reliability diagrams + score distributions (real_anchor=teams_real_all_dev)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG / "reliability_diagrams.png", dpi=140, bbox_inches="tight")
print(f"\nwrote {FIG / 'reliability_diagrams.png'}")
