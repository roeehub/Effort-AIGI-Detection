"""Compute per-layer inv_mean for T4 ckpts vs P8A baseline.

Answers: does the GRL at L11 attachment also move inv_mean at earlier
layers (L6, L9)? Or does the effect concentrate at the attachment point?

Reads outputs/forgery_signal_atlas_with_t4.csv (all layers × all ckpts ×
all signals). Computes inv_mean = forgery_AUC - mean(shortcut_AUCs) for
each (ckpt, layer) cell.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
INPUT = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-10/outputs/forgery_signal_atlas_with_t4.csv"
OUT = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-10/outputs/per_layer_inv_mean.csv"

SHORTCUT_SIGNALS = ["is_dor", "is_chronic_6", "lap_var_high", "min_dim_high", "face_size_high"]


def main():
    df = pd.read_csv(INPUT)
    # Pivot: (ckpt, layer) → dict of signal AUCs
    rows = []
    for (ckpt, layer), grp in df.groupby(["ckpt", "layer"]):
        forgery = grp[grp["signal"] == "is_real_vs_fake"]["auc"]
        if forgery.empty:
            continue
        forgery_auc = float(forgery.iloc[0])
        shortcut_aucs = []
        per_axis = {}
        for sig in SHORTCUT_SIGNALS:
            sig_row = grp[grp["signal"] == sig]
            if not sig_row.empty:
                v = float(sig_row["auc"].iloc[0])
                if not np.isnan(v):
                    shortcut_aucs.append(v)
                    per_axis[sig] = v
        if not shortcut_aucs:
            continue
        inv_mean = forgery_auc - np.mean(shortcut_aucs)
        row = {"ckpt": ckpt, "layer": layer, "forgery_auc": forgery_auc, "inv_mean": inv_mean}
        row.update(per_axis)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)

    # Display: pivot to wide format ckpt × layer
    wide = table.pivot(index="ckpt", columns="layer", values="inv_mean")
    print("=" * 70)
    print("inv_mean per (ckpt, layer)")
    print("=" * 70)
    print(wide.to_string(float_format="%+.4f"))
    print()

    # Highlight T4 vs P8A change per layer
    if "P8A" in wide.index:
        p8a_row = wide.loc["P8A"]
        print("=" * 70)
        print("Δinv_mean per layer (T4 ckpts vs P8A)")
        print("=" * 70)
        t4_rows = wide[wide.index.str.startswith("T4")]
        delta = t4_rows.sub(p8a_row, axis=1)
        print(delta.to_string(float_format="%+.4f"))
        print()
        print("Best T4 (max inv_mean across all layers): ", wide[wide.index.str.startswith("T4")].max().to_dict())
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
