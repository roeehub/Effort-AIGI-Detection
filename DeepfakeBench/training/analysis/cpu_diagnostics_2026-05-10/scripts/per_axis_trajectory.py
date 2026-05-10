"""Per-axis AUC trajectory across T4-λ1.0 training steps.

Plots how each shortcut axis's L11 LR-probe AUC changes from P8A baseline
through T4-λ1.0 steps {5000, 9000, 10500, 11250}.

Reveals WHICH axes the encoder actually rearranged over training, vs which
stayed pinned to P8A baseline (= mechanism didn't bite that axis).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
INPUT = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-10/outputs/forgery_signal_atlas_with_t4.csv"
OUT_PNG = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-10/outputs/T4_per_axis_trajectory.png"
OUT_CSV = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-10/outputs/T4_per_axis_trajectory.csv"

T4_CKPTS_ORDERED = [
    ("P8A", 0),
    ("T4_L1_step5000", 5000),
    ("T4_L1_step9000", 9000),
    ("T4_L1_step10500", 10500),
    ("T4_L1_step11250", 11250),
]

AXES = ["is_real_vs_fake", "is_dor", "is_chronic_6", "lap_var_high", "min_dim_high", "face_size_high"]


def main():
    df = pd.read_csv(INPUT)
    df = df[df["layer"] == 11]

    rows = []
    for ckpt, step in T4_CKPTS_ORDERED:
        sub = df[df["ckpt"] == ckpt]
        if sub.empty:
            print(f"WARN: {ckpt} not in atlas")
            continue
        for axis in AXES:
            ax_row = sub[sub["signal"] == axis]
            if not ax_row.empty:
                rows.append({"ckpt": ckpt, "step": step, "axis": axis, "auc": float(ax_row["auc"].iloc[0])})
    table = pd.DataFrame(rows)
    pivot = table.pivot(index="step", columns="axis", values="auc").reindex(columns=AXES)
    pivot.to_csv(OUT_CSV)
    print("Per-axis L11 AUC trajectory (rows = step, cols = axis):")
    print(pivot.to_string(float_format="%.4f"))
    print()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, axis in enumerate(AXES):
        if axis in pivot.columns:
            label = f"{axis} (shortcut)" if axis != "is_real_vs_fake" else "forgery_AUC (target)"
            ax.plot(pivot.index, pivot[axis], marker="o", label=label, color=colors[i],
                    linewidth=2.5 if axis == "is_real_vs_fake" else 1.5)

    # Reference line at P8A baseline for each axis
    for axis in AXES:
        if axis in pivot.columns:
            p8a_val = pivot[axis].iloc[0]
            ax.axhline(p8a_val, color="gray", linestyle=":", linewidth=0.5, alpha=0.4)

    ax.set_xlabel("T4-λ1.0 training step (P8A = baseline at step 0)")
    ax.set_ylabel("L11 LR-probe AUC")
    ax.set_title("T4-λ1.0 per-axis L11 AUC trajectory\nMechanism is biting iff shortcut AUC drops while forgery AUC holds")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(linestyle=":", alpha=0.4)
    ax.set_ylim(0.85, 1.005)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nWrote {OUT_PNG}")
    print(f"Wrote {OUT_CSV}")

    # Print summary deltas
    print()
    print("=" * 70)
    print("Δ from P8A baseline at each step (negative for shortcuts = good)")
    print("=" * 70)
    deltas = pivot.sub(pivot.loc[0], axis=1).iloc[1:]
    print(deltas.to_string(float_format="%+.4f"))


if __name__ == "__main__":
    raise SystemExit(main())
