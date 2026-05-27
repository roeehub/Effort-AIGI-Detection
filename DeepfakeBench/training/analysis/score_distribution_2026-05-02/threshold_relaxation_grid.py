"""Threshold-relaxation grid: per (suite × model) recall/FPR at canonical taus.

For each fake suite: at what τ does it cross 50%? At what τ does it cross 80%?
For each real suite: at what τ does it cross 5% FPR?

Uses the existing recall_curve_by_suite.csv computed by run_diagnostics.py,
plus combined_frames.parquet for finer τ resolution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"

DEPLOYED_TAU = {"P8A": 0.990946, "P18T": 0.99359, "P18C": 0.994625}
CANONICAL_TAUS = [0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
MODELS = ["P8A", "P18T", "P18C"]
FAKE_SUITES = [
    "teams_fake_all_dev",
    "teams_fake_all_lockbox",
    "visomaster_enhanced_macro_dev",
    "deeplive_enhanced_dev",
]
REAL_SUITES = [
    "teams_real_all_dev",
    "teams_real_all_lockbox",
    "teams_real_dor_dev",
    "teams_real_lighting_extreme_dev",
    "teams_real_poor_quality_dev",
]


def main():
    df = pd.read_parquet(OUT / "combined_frames.parquet")

    # ------------- Per (suite, model, tau) recall/FPR table
    rows = []
    for suite in FAKE_SUITES + REAL_SUITES:
        for model in MODELS:
            sub = df[(df["suite"] == suite) & (df["model"] == model)]
            if sub.empty:
                continue
            is_real = (sub["label"] == 0).all()
            metric = "FPR" if is_real else "RECALL"
            row = {"suite": suite, "model": model, "n_frames": len(sub), "metric": metric}
            for tau in CANONICAL_TAUS:
                row[f"τ={tau}"] = float((sub["frame_prob"] >= tau).mean())
            row["τ=deployed"] = float((sub["frame_prob"] >= DEPLOYED_TAU[model]).mean())
            rows.append(row)
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "threshold_grid.csv", index=False)
    print("=== Threshold-relaxation grid ===\n")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(grid.to_string(index=False, float_format="%.3f"))

    # ------------- "What τ does each suite cross floor X?" pivot
    print("\n\n=== Crossing thresholds (frame-level) ===\n")
    crossings = []
    for suite in FAKE_SUITES + REAL_SUITES:
        for model in MODELS:
            sub = df[(df["suite"] == suite) & (df["model"] == model)]
            if sub.empty:
                continue
            is_real = (sub["label"] == 0).all()
            tau_grid = np.linspace(0.5, 0.999, 500)
            metric_vals = [(sub["frame_prob"] >= t).mean() for t in tau_grid]
            row = {"suite": suite, "model": model, "metric": "FPR" if is_real else "RECALL"}
            if is_real:
                # τ at which FPR drops below floor
                for floor in [0.10, 0.05, 0.02, 0.01]:
                    crossed = next((t for t, v in zip(tau_grid, metric_vals) if v <= floor), None)
                    row[f"τ_FPR≤{floor}"] = float(crossed) if crossed else float("nan")
            else:
                # τ at which RECALL drops below floor (i.e. above this τ recall < floor)
                for floor in [0.10, 0.30, 0.50, 0.70, 0.90]:
                    crossed = next((t for t, v in zip(tau_grid, metric_vals) if v <= floor), None)
                    row[f"τ_RECALL≤{floor}"] = float(crossed) if crossed else float("nan")
            crossings.append(row)
    cross_df = pd.DataFrame(crossings)
    cross_df.to_csv(OUT / "threshold_crossings.csv", index=False)
    print(cross_df.to_string(index=False, float_format="%.4f"))

    # ------------- Operating-point feasibility table:
    # If real_FPR floor is X, what's the recall on each fake suite?
    print("\n\n=== Operating-point feasibility (per model) ===\n")
    print("If we require teams_real_all_dev FPR ≤ X, what tau and what fake recalls do we get?\n")

    feas_rows = []
    for model in MODELS:
        real = df[(df["suite"] == "teams_real_all_dev") & (df["model"] == model)]
        if real.empty:
            continue
        for fpr_floor in [0.20, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01]:
            # Find smallest τ that yields FPR ≤ fpr_floor (i.e., the loosest τ satisfying constraint).
            tau_grid = np.linspace(0.5, 0.9999, 1000)
            for tau in tau_grid:
                if (real["frame_prob"] >= tau).mean() <= fpr_floor:
                    chosen_tau = tau
                    break
            else:
                chosen_tau = float("nan")
            row = {"model": model, "real_FPR_floor": fpr_floor, "chosen_τ": chosen_tau}
            for suite in FAKE_SUITES:
                fk = df[(df["suite"] == suite) & (df["model"] == model)]
                if fk.empty:
                    continue
                row[f"{suite}_recall"] = float((fk["frame_prob"] >= chosen_tau).mean())
            for rs in ["teams_real_dor_dev", "teams_real_all_lockbox", "teams_real_lighting_extreme_dev"]:
                rl = df[(df["suite"] == rs) & (df["model"] == model)]
                if rl.empty:
                    continue
                row[f"{rs}_FPR"] = float((rl["frame_prob"] >= chosen_tau).mean())
            feas_rows.append(row)
    feas = pd.DataFrame(feas_rows)
    feas.to_csv(OUT / "operating_point_feasibility.csv", index=False)
    print(feas.to_string(index=False, float_format="%.4f"))

    # ------------- Plot: operating-point curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, model in zip(axes, MODELS):
        m_feas = feas[feas["model"] == model].sort_values("real_FPR_floor")
        for suite, color in [
            ("teams_fake_all_dev", "#1f77b4"),
            ("teams_fake_all_lockbox", "#9467bd"),
            ("visomaster_enhanced_macro_dev", "#d62728"),
            ("deeplive_enhanced_dev", "#2ca02c"),
        ]:
            col = f"{suite}_recall"
            if col not in m_feas.columns:
                continue
            ax.plot(m_feas["real_FPR_floor"], m_feas[col], "o-", color=color,
                    linewidth=1.6, label=suite, markersize=6)
        ax.set_xlabel("teams_real_all_dev FPR floor")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlim(0.21, 0.005)
        ax.set_ylim(0, 1.02)
        ax.set_title(model)
        ax.grid(alpha=0.3)
        ax.axvline(0.02, color="black", linestyle="--", alpha=0.5, label="contract floor 0.02")
    axes[0].set_ylabel("frame-level recall on each fake suite")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("If we relax the real-FPR constraint, how much fake recall do we unlock?")
    fig.tight_layout()
    fig.savefig(FIG / "operating_point_relaxation.png", dpi=120)
    plt.close(fig)

    print(f"\n[done] wrote {OUT / 'threshold_grid.csv'}, threshold_crossings.csv, operating_point_feasibility.csv")
    print(f"[plot] {FIG / 'operating_point_relaxation.png'}")


if __name__ == "__main__":
    main()
