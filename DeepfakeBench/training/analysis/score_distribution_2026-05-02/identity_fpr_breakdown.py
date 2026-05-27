"""Per-identity FPR breakdown for teams_real_all_dev (n=4564).

The contract treats teams_real_all_dev as one population with mean FPR ~2.7%.
This script asks: which identities account for the bulk of those false flags?

Identities are extracted from frame_path (the leading prefix before frame_NNNN_*).
For each identity, compute FPR at the deployment tau for P8A/P18T/P18C and
the cumulative FPR concentration (Lorenz-like curve).

Cross-references the dor identity (known FPR driver per memory).

Output:
  outputs/identity_fpr_breakdown.csv (per-identity, per-model)
  outputs/figures/identity_fpr_concentration.png
  outputs/figures/identity_fpr_top20.png
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"

DEPLOYED_TAU = {"P8A": 0.990946, "P18T": 0.99359, "P18C": 0.994625}
RELAX_TAUS = [0.5, 0.7, 0.9, 0.95, 0.97]


def parse_identity(frame_path: str, video_id: Optional[str] = None) -> str:
    """Extract identity prefix from a teams_real_* frame_path.

    Frame paths look like:
      gs://teams-faces-data-test-.../real/<identity>__frame_NNNN.{png,jpg}
      gs://teams-faces-data-test-.../real/<identity>__frame_NNNN_*.{png,jpg}
    """
    name = frame_path.rsplit("/", 1)[-1]
    # Try the dunder split first
    if "__frame_" in name:
        return name.split("__frame_")[0]
    # Fall back to video_id-based prefix
    if video_id and "__" in video_id:
        return video_id.split("__")[0]
    # Otherwise return the basename without extension
    return name.rsplit(".", 1)[0]


def main():
    df = pd.read_parquet(OUT / "combined_frames.parquet")

    # All real suites for completeness
    real_suites = ["teams_real_all_dev", "teams_real_all_lockbox", "teams_real_dor_dev",
                   "teams_real_lighting_extreme_dev", "teams_real_poor_quality_dev"]
    real_df = df[df["suite"].isin(real_suites)].copy()
    real_df["identity"] = real_df.apply(lambda r: parse_identity(r["frame_path"], r.get("video_id")), axis=1)
    print(f"[load] {len(real_df):,} real frames across {len(real_suites)} suites")
    print(f"[load] {real_df['identity'].nunique()} unique identity strings overall")

    # Per (suite, identity) frame counts
    print("\n=== Top 10 identities per real suite ===")
    for suite in real_suites:
        sub = real_df[real_df["suite"] == suite]
        if sub.empty:
            continue
        top = sub["identity"].value_counts().head(10)
        print(f"\n{suite} (n={len(sub)//3}):  # frames per identity, top 10:")
        print(top.to_string())

    # ── Per-identity FPR table on teams_real_all_dev ──
    primary = real_df[real_df["suite"] == "teams_real_all_dev"]
    rows = []
    for ident, g in primary.groupby("identity"):
        n_per_model = g.groupby("model").size().min()  # should be the same per model
        row = {"identity": ident, "n_frames": int(n_per_model)}
        for model in ["P8A", "P18T", "P18C"]:
            sub = g[g["model"] == model]
            if sub.empty:
                continue
            tau = DEPLOYED_TAU[model]
            row[f"FPR_{model}_at_deployed_tau"] = float((sub["frame_prob"] >= tau).mean())
            row[f"n_FP_{model}_at_deployed_tau"] = int((sub["frame_prob"] >= tau).sum())
            for tau_relax in RELAX_TAUS:
                row[f"FPR_{model}_tau_{tau_relax}"] = float((sub["frame_prob"] >= tau_relax).mean())
            row[f"mean_prob_{model}"] = float(sub["frame_prob"].mean())
        rows.append(row)
    ident_df = pd.DataFrame(rows).sort_values("FPR_P8A_at_deployed_tau", ascending=False)
    ident_df.to_csv(OUT / "identity_fpr_breakdown.csv", index=False)

    # ── Concentration of false positives ──
    print("\n=== False-positive concentration on teams_real_all_dev ===")
    for model in ["P8A", "P18T", "P18C"]:
        col = f"n_FP_{model}_at_deployed_tau"
        if col not in ident_df.columns:
            continue
        sorted_fp = ident_df.sort_values(col, ascending=False)
        total_fp = sorted_fp[col].sum()
        if total_fp == 0:
            print(f"\n{model}: 0 false positives total at deployed tau (clean)")
            continue
        cumulative = sorted_fp[col].cumsum() / total_fp
        n_for_50 = (cumulative >= 0.5).idxmax() + 1 if any(cumulative >= 0.5) else len(sorted_fp)
        n_for_80 = (cumulative >= 0.8).idxmax() + 1 if any(cumulative >= 0.8) else len(sorted_fp)
        n_for_95 = (cumulative >= 0.95).idxmax() + 1 if any(cumulative >= 0.95) else len(sorted_fp)
        print(f"\n{model}: {total_fp} total FPs across {sorted_fp[col].astype(bool).sum()} identities ({len(sorted_fp)} total)")
        print(f"  50% of FPs concentrated in top {n_for_50} identities")
        print(f"  80% of FPs concentrated in top {n_for_80} identities")
        print(f"  95% of FPs concentrated in top {n_for_95} identities")
        print(f"  Top 10 by FP count:")
        print(sorted_fp.head(10)[["identity", "n_frames", col, f"FPR_{model}_at_deployed_tau", f"mean_prob_{model}"]].to_string(index=False))

    # ── Plot 1: cumulative FP concentration ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for model, color in [("P8A", "#1f77b4"), ("P18T", "#ff7f0e"), ("P18C", "#2ca02c")]:
        col = f"n_FP_{model}_at_deployed_tau"
        if col not in ident_df.columns:
            continue
        sorted_fp = ident_df.sort_values(col, ascending=False)
        total_fp = sorted_fp[col].sum()
        if total_fp == 0:
            continue
        x = np.arange(1, len(sorted_fp) + 1) / len(sorted_fp) * 100
        y = sorted_fp[col].cumsum() / total_fp * 100
        ax.plot(x, y, label=f"{model} (n_FP={total_fp}, n_id={(sorted_fp[col]>0).sum()})", color=color, linewidth=2)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.4, label="uniform (no concentration)")
    ax.set_xlabel("% of identities (sorted by FP count, descending)")
    ax.set_ylabel("% of total false positives accounted for")
    ax.set_title("False-positive concentration on teams_real_all_dev (at deployed τ)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "identity_fpr_concentration.png", dpi=120)
    plt.close(fig)

    # ── Plot 2: top 20 identities by FPR ──
    top_n = 20
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    for ax, model in zip(axes, ["P8A", "P18T", "P18C"]):
        col = f"FPR_{model}_at_deployed_tau"
        n_col = f"n_frames"
        if col not in ident_df.columns:
            continue
        # Sort by FPR, but only for identities with at least 5 frames
        sorted_id = ident_df[ident_df[n_col] >= 5].sort_values(col, ascending=False).head(top_n)
        labels = [f"{ident[:30]} (n={n})" for ident, n in zip(sorted_id["identity"], sorted_id[n_col])]
        ax.barh(range(len(sorted_id)), sorted_id[col], color="indianred", alpha=0.8)
        ax.set_yticks(range(len(sorted_id)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(f"FPR @ deployed τ ({DEPLOYED_TAU[model]:.4f})")
        ax.set_title(f"{model} — top {top_n} identities by FPR (≥5 frames)")
        ax.axvline(0.05, color="black", linestyle="--", alpha=0.5, label="5%")
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("Per-identity FPR — which identities drive the false-positive rate on teams_real_all_dev?")
    fig.tight_layout()
    fig.savefig(FIG / "identity_fpr_top20.png", dpi=120)
    plt.close(fig)

    print(f"\n[done] outputs in {OUT}")


if __name__ == "__main__":
    main()
