"""Summary figures for the 2026-05-19 pre-plan CPU diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402


def fig_summary():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Baseline AUC + AUC vs size (Job C)
    ax = axes[0, 0]
    auc_summary = pd.read_csv(common.OUT / "job_c_auc_summary.csv")
    base = pd.read_csv(common.OUT / "job_c_baseline_auc.csv")
    ckpt_order = ["P8A_step5000", "T5C_step3500", "E2B_step3200", "SLOT_A_RESCHAIN", "SLOT_B_6AXIS_GRL"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(ckpt_order)))
    for color, ckpt in zip(colors, ckpt_order):
        sub = auc_summary[(auc_summary["ckpt"] == ckpt) & (auc_summary["down_size"] > 0)]
        ax.plot(sub["down_size"], sub["median_auc"], "o-", label=ckpt, color=color, linewidth=2)
        b = base[base["ckpt"] == ckpt]
        if not b.empty:
            ax.axhline(b["baseline_auc"].iloc[0], color=color, linestyle=":", alpha=0.4)
    ax.set_xlabel("downsample target size (px)")
    ax.set_ylabel("fake-vs-real AUC")
    ax.set_title("(a) Job C — AUC under resolution-chain aug\n(dotted = no-aug baseline)")
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Per-ckpt × chronic identity FPR matrix (Job E)
    ax = axes[0, 1]
    fpr_wide = pd.read_csv(common.OUT / "job_e_fpr_matrix_wide.csv")
    chronic_view = fpr_wide[fpr_wide["is_chronic"] == True].head(5)
    ckpts_short = {
        "P8A_REFERENCE_STEP5000": "P8A",
        "T5C_PERIODIC_STEP3500": "T5C",
        "SLOT_A_RESCHAIN_STEP1500": "Slot α s1.5k",
        "SLOT_A_RESCHAIN_STEP3500": "Slot α s3.5k",
        "SLOT_B_6AXIS_GRL_STEP3500": "Slot β",
    }
    full_cols = list(ckpts_short.keys())
    mat = chronic_view[full_cols].values
    im = ax.imshow(mat, aspect="auto", cmap="Reds", vmin=0, vmax=0.30)
    ax.set_xticks(range(len(full_cols)))
    ax.set_xticklabels([ckpts_short[c] for c in full_cols], rotation=30, ha="right")
    ax.set_yticks(range(len(chronic_view)))
    ax.set_yticklabels(chronic_view["identity_key"].tolist())
    for i in range(len(chronic_view)):
        for j in range(len(full_cols)):
            val = mat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.15 else "black", fontsize=9)
    ax.set_title("(b) Job E — Per-ckpt chronic FPR\n(at each ckpt's selected τ)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (c) Tiebreak rank comparison (Job B)
    ax = axes[1, 0]
    ranks = pd.read_csv(common.OUT / "job_b_ranking_comparison.csv")
    rank_cols = [c for c in ranks.columns if c.startswith("rank_")]
    ranks_short = {
        "rank_v3fix_lockbox_asc": "v3-fix\n(current)",
        "rank_alt_recall_desc": "dev_recall\ntiebreak",
        "rank_alt_lockbox_fake_desc": "lockbox_fake\ntiebreak",
        "rank_alt_viso_desc": "viso\ntiebreak",
        "rank_composite_k1": "composite\nk=1",
        "rank_composite_k2": "composite\nk=2",
        "rank_composite_k5": "composite\nk=5",
        "rank_composite_k10": "composite\nk=10",
    }
    rank_cols = list(ranks_short.keys())
    ckpts = ranks["checkpoint_key"].tolist()
    mat = ranks[rank_cols].values
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=5)
    ax.set_xticks(range(len(rank_cols)))
    ax.set_xticklabels([ranks_short[c] for c in rank_cols], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(ckpts)))
    ckpt_short = [c.replace("_REFERENCE_STEP5000", "")
                   .replace("_PERIODIC_STEP3500", "")
                   .replace("_RESCHAIN_STEP", " s")
                   .replace("_6AXIS_GRL_STEP3500", "")
                   for c in ckpts]
    ax.set_yticklabels(ckpt_short)
    for i in range(len(ckpts)):
        for j in range(len(rank_cols)):
            ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", color="black", fontsize=10, fontweight="bold")
    ax.set_title("(c) Job B — Rank under 8 tiebreak policies\n(green=better; v3-fix is current)")
    plt.colorbar(im, ax=ax, fraction=0.046, label="rank")

    # (d) Slot β over-fire concentration (Job A)
    ax = axes[1, 1]
    per_ident = pd.read_csv(common.OUT / "job_a_per_identity_slot_b.csv")
    top = per_ident.sort_values("n_overfire", ascending=False).head(6)
    color_map = {"under_covered": "#d62728", "moderate": "#ff7f0e", "over_covered": "#1f77b4"}
    colors_b = [color_map.get(c, "#7f7f7f") for c in top["coverage_class"].fillna("none")]
    bars = ax.barh(top["identity_key"], top["n_overfire"], color=colors_b)
    ax.set_xlabel("number of lockbox over-fires by Slot β step3500")
    ax.set_title(f"(d) Job A — Slot β over-fire localization (n=124 total)\n93.5% concentrated on dor_shkedi (under-covered)")
    for bar, n, frac in zip(bars, top["n_overfire"], top["overfire_share_of_total"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{int(n)} ({frac:.1%})", va="center", fontsize=9)
    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(color="#d62728", label="under_covered"),
        Patch(color="#ff7f0e", label="moderate"),
        Patch(color="#1f77b4", label="over_covered"),
        Patch(color="#7f7f7f", label="absent from joint-marginal"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    fig.suptitle("Pre-plan CPU diagnostics — 2026-05-19 (10 jobs, $0 cost)\nFour key findings", fontsize=14)
    fig.tight_layout()
    out = common.FIGS / "summary_2026-05-19.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    fig_summary()
