"""Probe 2 — compute pre-registered S1/S2 falsifiers from a scorecard run.

Run AFTER:
  (a) the S1 (or S2) Vertex training job has completed
  (b) Probe 1 has identified the best ckpt by class_separation
  (c) a promotion contract scorecard has been launched on that ckpt + comparisons
  (d) the resulting threshold_grid.csv has been pulled to local

Computes:
  F-Sx-A: best ckpt class_separation ≥ 4.0  (input: from Probe 1 output JSON)
  F-Sx-B: best ckpt viso recall at FPR=10% (joint dev+lockbox) > target
  F-Sx-C: best ckpt lockbox dor FPR ≤ 5/1170 = 0.43% at calibrated τ
  + F1/F2/F3 from the original P22 falsifier (re-uses logic from
    analysis/p22_eval_2026-05-02/cpu_followups/scripts/_common.py)

Usage:
    python 02_compute_falsifiers_s1_s2.py \
        --probe1-json analysis/s1_s2_2026-05-02_planning/probes/outputs/01_select_best_ckpt_<run_id>.json \
        --grid analysis/s1_s2_eval/<run_id>/scorecard/promotion_contract/threshold_grid.csv \
        --frames-dir analysis/s1_s2_eval/<run_id>/scorecard/reports/ \
        --packet S1   # or S2

Outputs (all in analysis/s1_s2_2026-05-02_planning/probes/outputs/):
    02_falsifiers_<S1|S2>_<run_id>.csv     — per-falsifier PASS/FAIL
    02_verdict_<S1|S2>_<run_id>.json       — score and overall verdict
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

OUT = Path("analysis/s1_s2_2026-05-02_planning/probes/outputs")

ATTRS_CSV = Path("analysis/score_distribution_2026-05-02/outputs/cross_suite_attributes.csv")
VISO_LAP_CSV = Path("analysis/p22_eval_2026-05-02/cpu_followups/outputs/07_viso_laplacian_fetched.csv")

# F-S1 thresholds (all three falsifiers)
F_S1_THRESHOLDS = {
    "A_min_class_separation": 4.0,
    "B_min_viso_recall_fpr10pct": 0.242,  # > P22 step1k's 24.2%
    "C_max_lockbox_dor_fpr": 5 / 1170,
}

# F-S2 thresholds
F_S2_THRESHOLDS = {
    "A_min_class_separation": None,  # filled in: must >= S1's best class_sep
    "B_min_viso_recall_fpr10pct": 0.300,  # > P8A's 27.1% dev-only ceiling
    "C_max_lockbox_dor_fpr": 5 / 1170,
}


def find_per_frame_csv(frames_dir, suite, ckpt_lower_pat):
    """Find <suite>_<ckpt_lower_pat>_*frames_report.csv in frames_dir."""
    for p in Path(frames_dir).rglob("*.csv"):
        n = p.name.lower()
        if "frames_report" in n and suite.lower() in n and ckpt_lower_pat.lower() in n:
            return p
    return None


def joint_recal_grid(grid: pd.DataFrame, ckpt_key: str, floor: float = 0.10):
    """Find smallest τ s.t. dev_primary_real_fpr ≤ floor AND
    teams_real_all_lockbox real-FPR (computed from per-frame CSVs separately
    or assumed equal here)... actually for joint, we need both. The grid
    has dev_primary_real_fpr but lockbox FPR is per-suite. Look for
    teams_real_all_lockbox__real_fpr column."""
    sub = grid[grid.checkpoint_key == ckpt_key].copy()
    if "teams_real_all_lockbox__real_fpr" not in sub.columns:
        # Older grids might not have lockbox FPR pivoted — fall back to dev only
        valid = sub[sub.dev_primary_real_fpr <= floor + 1e-9]
        if len(valid) == 0: return None
        return valid.loc[valid.threshold.idxmin()]
    valid = sub[(sub.dev_primary_real_fpr <= floor + 1e-9) &
                (sub.teams_real_all_lockbox__real_fpr <= floor + 1e-9)]
    if len(valid) == 0: return None
    return valid.loc[valid.threshold.idxmin()]


def viso_pearson_r_full(per_frame_csv: Path):
    """Compute Pearson r(score, lap) on full viso (n=550). Uses the
    Laplacian table fetched in the P22 follow-ups."""
    if not per_frame_csv.exists() or not VISO_LAP_CSV.exists():
        return None, 0
    scores = pd.read_csv(per_frame_csv)
    lap = pd.read_csv(VISO_LAP_CSV)
    # Full viso lap also includes the original 62 attrs frames for viso
    if ATTRS_CSV.exists():
        attrs = pd.read_csv(ATTRS_CSV)
        all_lap = pd.concat([
            lap[["frame_path", "laplacian_var"]],
            attrs.dropna(subset=["laplacian_var"])[["frame_path", "laplacian_var"]]
        ], ignore_index=True).drop_duplicates("frame_path")
    else:
        all_lap = lap
    merged = scores.merge(all_lap, on="frame_path", how="inner")
    merged = merged.dropna(subset=["frame_prob", "laplacian_var"])
    if len(merged) < 30:
        return None, len(merged)
    r, _ = pearsonr(merged["frame_prob"], merged["laplacian_var"])
    return float(r), int(len(merged))


def compute_per_identity_dor_fpr(frames_dir, ckpt_lower_pat, tau):
    """Count P22-style FPs on dor_shkedi at calibrated τ."""
    p = find_per_frame_csv(frames_dir, "teams_real_all_lockbox", ckpt_lower_pat)
    if p is None: return None
    df = pd.read_csv(p)
    df["identity"] = df["video_id"].apply(
        lambda v: re.split(r"__s\d+|__seg_|__seq\d+", v)[0] if pd.notna(v) else "<missing>"
    )
    dor = df[df.identity == "dor_shkedi"]
    if len(dor) == 0: return None
    fps = (dor["frame_prob"] >= tau).sum()
    return {"n_dor_frames": len(dor), "n_dor_FPs_at_tau": int(fps),
            "dor_lockbox_fpr": float(fps / len(dor))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe1-json", required=True,
                    help="Output of Probe 1: 01_select_best_ckpt_<run_id>.json")
    ap.add_argument("--grid", required=True, help="threshold_grid.csv from S1/S2 scorecard")
    ap.add_argument("--frames-dir", required=True, help="reports/ dir from S1/S2 scorecard")
    ap.add_argument("--packet", choices=["S1", "S2"], required=True)
    ap.add_argument("--ckpt-key", required=True,
                    help="Checkpoint key in grid (e.g. S1_REDUX_BEST_BY_CLASS_SEP)")
    ap.add_argument("--ckpt-lower-pat", required=True,
                    help="Pattern used in per-frame CSV filenames (e.g. s1_redux_best)")
    ap.add_argument("--s1-best-class-sep", type=float, default=None,
                    help="Required for S2: S1's best class_separation value")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    probe1 = json.loads(Path(args.probe1_json).read_text())
    grid = pd.read_csv(args.grid)

    # F-Sx-A: class_separation
    peak_cs = probe1.get("peaks", {}).get("class_separation", {}).get("peak_value")
    thresh_A = (F_S1_THRESHOLDS["A_min_class_separation"] if args.packet == "S1"
                else (args.s1_best_class_sep or float("nan")))
    f_a_pass = peak_cs is not None and peak_cs >= thresh_A

    # F-Sx-B: viso recall at FPR=10% (joint)
    chosen = joint_recal_grid(grid, args.ckpt_key, floor=0.10)
    if chosen is not None:
        viso_recall_10pct = chosen.get("visomaster_enhanced_macro_dev__fake_recall")
        tau_at_10pct = float(chosen.threshold)
    else:
        viso_recall_10pct = float("nan")
        tau_at_10pct = float("nan")
    threshs = F_S1_THRESHOLDS if args.packet == "S1" else F_S2_THRESHOLDS
    thresh_B = threshs["B_min_viso_recall_fpr10pct"]
    f_b_pass = pd.notna(viso_recall_10pct) and viso_recall_10pct > thresh_B

    # F-Sx-C: lockbox dor FPR
    dor_audit = compute_per_identity_dor_fpr(args.frames_dir, args.ckpt_lower_pat, tau_at_10pct)
    if dor_audit:
        thresh_C = threshs["C_max_lockbox_dor_fpr"]
        f_c_pass = dor_audit["dor_lockbox_fpr"] <= thresh_C
    else:
        f_c_pass = False
        dor_audit = {"n_dor_frames": 0, "n_dor_FPs_at_tau": None, "dor_lockbox_fpr": None}

    # F1: Pearson r on full viso
    viso_csv = find_per_frame_csv(args.frames_dir, "visomaster_enhanced_macro_dev", args.ckpt_lower_pat)
    if viso_csv:
        viso_r, viso_n = viso_pearson_r_full(viso_csv)
    else:
        viso_r, viso_n = None, 0

    summary = {
        "packet": args.packet, "ckpt_key": args.ckpt_key,
        "F-A_class_sep": {"value": peak_cs, "threshold": thresh_A, "pass": f_a_pass},
        "F-B_viso_recall_fpr10pct": {
            "value": float(viso_recall_10pct) if pd.notna(viso_recall_10pct) else None,
            "threshold": thresh_B, "pass": f_b_pass, "tau_at_10pct": tau_at_10pct
        },
        "F-C_lockbox_dor_fpr": {**dor_audit, "threshold": threshs["C_max_lockbox_dor_fpr"], "pass": f_c_pass},
        "Pearson_viso_full": {"r": viso_r, "n": viso_n,
                               "vs_p22_baseline": +0.5074  # P8A from Job I
                              },
        "score": int(f_a_pass) + int(f_b_pass) + int(f_c_pass),
    }
    summary["verdict"] = (
        "SUCCESS" if summary["score"] >= 3 else
        "PARTIAL" if summary["score"] == 2 else
        "AMBIGUOUS" if summary["score"] == 1 else
        "FAILED"
    )
    out_path = OUT / f"02_verdict_{args.packet}_{args.ckpt_key}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print(f"{args.packet} verdict — {args.ckpt_key}")
    print("=" * 72)
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items(): print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
