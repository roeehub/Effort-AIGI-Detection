"""Extract best-recall numbers at FPR<=5% and FPR<=2% from the saved ROC sweep CSVs.

Produces a small auxiliary table for the FACTS doc body.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT_DIR = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/"
    "DeepfakeBench/training/analysis/p1_pe_eval_2026-05-07/"
    "calibration_probe_2026-05-07"
)

CKPTS = ["p1_bundle_periodic_step500", "p1_pairrank_periodic_step500"]
FPR_TARGETS = [0.10, 0.05, 0.02]


def best_for(df, fpr_max):
    feasible = df[df["fpr"] <= fpr_max + 1e-12]
    if len(feasible) == 0:
        # smallest FPR fallback row
        row = df.sort_values(["fpr", "recall"], ascending=[True, False]).iloc[0]
        return float(row["tau"]), float(row["fpr"]), float(row["recall"]), False
    feasible = feasible.sort_values(
        ["recall", "fpr", "tau"], ascending=[False, True, True]
    )
    row = feasible.iloc[0]
    return float(row["tau"]), float(row["fpr"]), float(row["recall"]), True


rows = []
for ckpt in CKPTS:
    roc = pd.read_csv(OUT_DIR / f"roc_sweep_{ckpt}.csv")
    for variant in ["raw", "platt", "isotonic"]:
        sub = roc[roc["variant"] == variant]
        for fpr_max in FPR_TARGETS:
            tau, fpr, recall, feas = best_for(sub, fpr_max)
            rows.append(
                {
                    "ckpt": ckpt,
                    "variant": variant,
                    "fpr_target": fpr_max,
                    "best_tau": tau,
                    "fpr_at_best": fpr,
                    "recall_at_best": recall,
                    "feasible_on_grid": feas,
                }
            )

out = pd.DataFrame(rows)
out.to_csv(OUT_DIR / "best_recall_by_fpr_target.csv", index=False)
print(out.to_string(index=False))
print(f"\nSaved: {OUT_DIR / 'best_recall_by_fpr_target.csv'}")
