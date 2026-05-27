"""Per-ckpt τ sweep + per-human gate analysis.

Inputs: analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/per_frame_full.csv
Outputs:
  outputs/tau_sweep_full.csv         — long table: ckpt × τ × per-human metrics + agg
  outputs/passing_tau_ranges.csv     — for each ckpt × (FPR floor, recall floor),
                                       the contiguous τ ranges that pass
  outputs/per_ckpt_summary.csv       — single row per ckpt with best-τ for each
                                       (FPR floor, recall floor) and full per-human
                                       breakdown at that τ
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]  # training/
SRC = ROOT / "analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/per_frame_full.csv"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

CKPTS = ["P8A", "E2B", "T5C", "SlotAv2_CLS", "SlotAv2_FACE"]
SCORE_COL = {c: f"prob_{c}" for c in CKPTS}

REAL_HUMANS = ["Noyn", "Roee_Windows", "Xiang", "Xinhe", "dor"]
FAKE_HUMANS = ["Xiang", "Xinhe", "dor"]
FAKE_ROLE = {h: f"fake_target_{h}" for h in FAKE_HUMANS}

# Gate combinations (max_real_fpr_floor, min_fake_recall_floor)
GATES = [
    ("user_bar",  0.05, 0.50),
    ("relaxed",   0.05, 0.40),
    ("very_strict", 0.02, 0.60),
    ("recall_lean", 0.10, 0.50),
]

TAUS = np.linspace(0.0, 1.0, 1001)  # 0.001 step


def load() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    df = df[df["deploy_relevant"] == True].copy()
    return df


def per_human_real_fpr(df: pd.DataFrame, score_col: str, tau: float, human: str) -> tuple[int, int, float]:
    sub = df[(df["human"] == human) & (df["role"] == "real")]
    n = len(sub)
    if n == 0:
        return 0, 0, float("nan")
    flagged = int((sub[score_col] >= tau).sum())
    return flagged, n, flagged / n


def per_human_fake_recall(df: pd.DataFrame, score_col: str, tau: float, human: str) -> tuple[int, int, float]:
    sub = df[(df["human"] == human) & (df["role"] == FAKE_ROLE[human])]
    n = len(sub)
    if n == 0:
        return 0, 0, float("nan")
    caught = int((sub[score_col] >= tau).sum())
    return caught, n, caught / n


def sweep_one(df: pd.DataFrame, ckpt: str) -> pd.DataFrame:
    score_col = SCORE_COL[ckpt]
    # Build score arrays per (human, role) for vectorization
    arrays: dict = {}
    for h in REAL_HUMANS:
        arrays[("real", h)] = df[(df["human"] == h) & (df["role"] == "real")][score_col].to_numpy()
    for h in FAKE_HUMANS:
        arrays[("fake", h)] = df[(df["human"] == h) & (df["role"] == FAKE_ROLE[h])][score_col].to_numpy()

    rows = []
    for tau in TAUS:
        row: dict = {"ckpt": ckpt, "tau": float(tau)}
        real_fprs = []
        for h in REAL_HUMANS:
            arr = arrays[("real", h)]
            n = len(arr)
            flagged = int((arr >= tau).sum()) if n else 0
            fpr = flagged / n if n else float("nan")
            row[f"real_n_{h}"] = n
            row[f"real_flag_{h}"] = flagged
            row[f"real_fpr_{h}"] = fpr
            if not np.isnan(fpr):
                real_fprs.append(fpr)
        row["max_real_fpr"] = max(real_fprs) if real_fprs else float("nan")
        row["agg_real_fpr"] = sum(row[f"real_flag_{h}"] for h in REAL_HUMANS) / sum(row[f"real_n_{h}"] for h in REAL_HUMANS)

        fake_recalls = []
        for h in FAKE_HUMANS:
            arr = arrays[("fake", h)]
            n = len(arr)
            caught = int((arr >= tau).sum()) if n else 0
            rec = caught / n if n else float("nan")
            row[f"fake_n_{h}"] = n
            row[f"fake_catch_{h}"] = caught
            row[f"fake_recall_{h}"] = rec
            if not np.isnan(rec):
                fake_recalls.append(rec)
        row["min_fake_recall"] = min(fake_recalls) if fake_recalls else float("nan")
        row["mean_fake_recall"] = float(np.mean(fake_recalls)) if fake_recalls else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def find_passing_ranges(sweep: pd.DataFrame, fpr_floor: float, recall_floor: float) -> list[tuple[float, float]]:
    pass_mask = (sweep["max_real_fpr"] <= fpr_floor) & (sweep["min_fake_recall"] >= recall_floor)
    if not pass_mask.any():
        return []
    # Contiguous ranges in tau
    taus = sweep.loc[pass_mask, "tau"].to_numpy()
    if len(taus) == 0:
        return []
    ranges: list[tuple[float, float]] = []
    start = taus[0]
    prev = taus[0]
    for t in taus[1:]:
        if t - prev > 0.0015:  # gap larger than 1 step (allow rounding wiggle)
            ranges.append((float(start), float(prev)))
            start = t
        prev = t
    ranges.append((float(start), float(prev)))
    return ranges


def main() -> None:
    df = load()
    print(f"loaded {len(df)} deploy-relevant frames")

    all_sweeps: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    passing_rows: list[dict] = []

    for ckpt in CKPTS:
        print(f"sweeping {ckpt}...")
        sweep = sweep_one(df, ckpt)
        all_sweeps.append(sweep)
        sweep.to_csv(OUT / f"sweep_{ckpt}.csv", index=False)

        summary: dict = {"ckpt": ckpt}
        for gate_name, fpr_floor, recall_floor in GATES:
            ranges = find_passing_ranges(sweep, fpr_floor, recall_floor)
            for (lo, hi) in ranges:
                # representative τ: midpoint
                mid = (lo + hi) / 2.0
                passing_rows.append({
                    "ckpt": ckpt,
                    "gate": gate_name,
                    "fpr_floor": fpr_floor,
                    "recall_floor": recall_floor,
                    "tau_lo": lo,
                    "tau_hi": hi,
                    "tau_mid": mid,
                    "range_width": hi - lo,
                })
            summary[f"{gate_name}_n_ranges"] = len(ranges)
            summary[f"{gate_name}_tau_lo"] = ranges[0][0] if ranges else float("nan")
            summary[f"{gate_name}_tau_hi"] = ranges[-1][1] if ranges else float("nan")

            # Best τ inside range = midpoint of widest range; report full per-human at that τ
            if ranges:
                widest = max(ranges, key=lambda r: r[1] - r[0])
                best_tau = (widest[0] + widest[1]) / 2.0
                row_at_tau = sweep.iloc[(sweep["tau"] - best_tau).abs().argsort().iloc[0]]
                summary[f"{gate_name}_best_tau"] = float(row_at_tau["tau"])
                for h in REAL_HUMANS:
                    summary[f"{gate_name}_realFPR_{h}"] = float(row_at_tau[f"real_fpr_{h}"])
                for h in FAKE_HUMANS:
                    summary[f"{gate_name}_fakeRec_{h}"] = float(row_at_tau[f"fake_recall_{h}"])
                summary[f"{gate_name}_min_fake_recall"] = float(row_at_tau["min_fake_recall"])
                summary[f"{gate_name}_max_real_fpr"] = float(row_at_tau["max_real_fpr"])
            else:
                # report best-effort: τ that maximizes (min_fake_recall) subject to max_real_fpr ≤ fpr_floor
                ok = sweep[sweep["max_real_fpr"] <= fpr_floor]
                if len(ok):
                    best_idx = ok["min_fake_recall"].idxmax()
                    row_at_tau = sweep.loc[best_idx]
                    summary[f"{gate_name}_best_tau"] = float(row_at_tau["tau"])
                    summary[f"{gate_name}_min_fake_recall"] = float(row_at_tau["min_fake_recall"])
                    summary[f"{gate_name}_max_real_fpr"] = float(row_at_tau["max_real_fpr"])
                    for h in REAL_HUMANS:
                        summary[f"{gate_name}_realFPR_{h}"] = float(row_at_tau[f"real_fpr_{h}"])
                    for h in FAKE_HUMANS:
                        summary[f"{gate_name}_fakeRec_{h}"] = float(row_at_tau[f"fake_recall_{h}"])
                else:
                    summary[f"{gate_name}_best_tau"] = float("nan")
                    summary[f"{gate_name}_min_fake_recall"] = float("nan")
                    summary[f"{gate_name}_max_real_fpr"] = float("nan")

        summary_rows.append(summary)

    combined = pd.concat(all_sweeps, ignore_index=True)
    combined.to_csv(OUT / "tau_sweep_full.csv", index=False)

    pd.DataFrame(passing_rows).to_csv(OUT / "passing_tau_ranges.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / "per_ckpt_summary.csv", index=False)

    # Also dump as JSON for easy reading
    with open(OUT / "per_ckpt_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)

    # Print compact textual summary
    print("\n=== Per-ckpt summary (user_bar = max_real_fpr ≤ 5%, min_fake_recall ≥ 50%) ===")
    for s in summary_rows:
        ub = "PASS" if s["user_bar_n_ranges"] > 0 else "FAIL"
        print(f"  {s['ckpt']:15s}  user_bar={ub}  n_ranges={s['user_bar_n_ranges']}  "
              f"tau_range=[{s['user_bar_tau_lo']:.3f}, {s['user_bar_tau_hi']:.3f}]  "
              f"best_tau={s.get('user_bar_best_tau', float('nan')):.3f}  "
              f"min_recall={s.get('user_bar_min_fake_recall', float('nan')):.3f}  "
              f"max_fpr={s.get('user_bar_max_real_fpr', float('nan')):.3f}")

    print("\n=== Relaxed (max_real_fpr ≤ 5%, min_fake_recall ≥ 40%) ===")
    for s in summary_rows:
        ub = "PASS" if s["relaxed_n_ranges"] > 0 else "FAIL"
        print(f"  {s['ckpt']:15s}  relaxed={ub}  n_ranges={s['relaxed_n_ranges']}  "
              f"tau_range=[{s['relaxed_tau_lo']:.3f}, {s['relaxed_tau_hi']:.3f}]")


if __name__ == "__main__":
    main()
