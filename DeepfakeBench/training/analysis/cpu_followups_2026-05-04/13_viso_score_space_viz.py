"""Standalone visualization of the viso fakes score-space (no viewer needed)."""
import csv
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANALYSIS = Path(__file__).parent
FIG_DIR = ANALYSIS / "figures"

rows = list(csv.DictReader((ANALYSIS / "outputs" / "viso_score_space_tsne.csv").open()))
print(f"Loaded {len(rows)} viso fake points")

# Categorize for color
SUBSET_COLORS = {
    "NONE": "#bfbfbf",  # gray
    "P8A": "tab:blue",
    "E2B_3200": "tab:orange",
    "E3_6600": "tab:green",
    "P8A+E3_6600": "tab:purple",
    "E2B_3200+P8A": "tab:cyan",
    "E2B_3200+E3_6600": "tab:olive",
    "E2B_3200+E3_6600+P8A": "tab:red",
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: t-SNE colored by caught_by_subset
ax = axes[0]
xs = np.array([float(r["tsne_x"]) for r in rows])
ys = np.array([float(r["tsne_y"]) for r in rows])
for subset, color in SUBSET_COLORS.items():
    mask = np.array([r["caught_by_subset"] == subset for r in rows])
    if mask.sum() > 0:
        ax.scatter(xs[mask], ys[mask], c=color, s=20, alpha=0.7, label=f"{subset} (n={mask.sum()})", edgecolors="white", linewidths=0.3)
ax.set_title("Viso fakes in score-space (t-SNE on (P8A, E2B, E3) scores)\nColored by which ckpt subset catches at FPR=10%")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.legend(fontsize=8, loc="best")
ax.grid(alpha=0.3)

# Plot 2: 2D scatter of P8A_score vs max(E2B_score, E3_score)
ax = axes[1]
p8a_s = np.array([float(r["P8A_score"]) for r in rows])
e2_s = np.array([float(r["E2B_3200_score"]) for r in rows])
e3_s = np.array([float(r["E3_6600_score"]) for r in rows])
max_scratch = np.maximum(e2_s, e3_s)
for subset, color in SUBSET_COLORS.items():
    mask = np.array([r["caught_by_subset"] == subset for r in rows])
    if mask.sum() > 0:
        ax.scatter(p8a_s[mask], max_scratch[mask], c=color, s=20, alpha=0.7,
                   label=f"{subset} (n={mask.sum()})", edgecolors="white", linewidths=0.3)
# Decision lines (FPR=10% τ values)
ax.axvline(0.7052, color="tab:blue", linestyle="--", alpha=0.5, label="P8A τ@10%")
ax.axhline(0.5075, color="tab:orange", linestyle="--", alpha=0.3, label="E2B τ@10%")
ax.axhline(0.8526, color="tab:green", linestyle="--", alpha=0.3, label="E3 τ@10%")
ax.set_title("Viso fakes: P8A score vs max(E2B, E3) score\nLines show FPR=10% thresholds")
ax.set_xlabel("P8A score")
ax.set_ylabel("max(E2B_3200, E3_6600) score")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=7, loc="upper left")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "viso_score_space.png", dpi=100)
plt.close(fig)
print(f"  → {FIG_DIR / 'viso_score_space.png'}")
