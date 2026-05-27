"""Unified CPU analytics across the chain — P8A + P22 + S1 + S2.

Reuses methodology from P22 cpu_followups: joint dev+lockbox recal at multi-FPR,
score-distribution forensics, per-identity FPR, viso Pearson r on full sample.

Usage:
    python 04_unified_chain_analytics.py \
        --s1-grid analysis/s1_s2_eval/<s1_run>/scorecard/promotion_contract/threshold_grid.csv \
        --s2-grid analysis/s1_s2_eval/<s2_run>/scorecard/promotion_contract/threshold_grid.csv \
        --s1-frames-dir <s1 reports/> \
        --s2-frames-dir <s2 reports/>

Output:
    analysis/s1_s2_2026-05-02_planning/probes/outputs/
        04_chain_joint_recal_grid.csv          # all ckpts × FPR floors
        04_chain_min_fpr_for_90pct_recall.csv  # capability ceiling per ckpt
        04_chain_score_quantiles.csv           # distribution quantiles
        04_chain_pearson_viso_full.csv         # Pearson r on full viso (n=550)
        04_chain_per_identity_fpr.csv          # dor/PC_Generator FPR per ckpt
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

OUT = Path("analysis/s1_s2_2026-05-02_planning/probes/outputs")
P22_GRID = Path("analysis/p22_eval_2026-05-02/scorecard_data/promotion_contract/threshold_grid.csv")
P22_FRAMES = Path("/tmp/p22_frames_local")
P18_FRAMES = Path("analysis/score_distribution_2026-05-02/raw_reports")
ATTRS = Path("analysis/score_distribution_2026-05-02/outputs/cross_suite_attributes.csv")
VISO_LAP = Path("analysis/p22_eval_2026-05-02/cpu_followups/outputs/07_viso_laplacian_fetched.csv")

SUITES_FAKE = ["teams_fake_all_dev", "teams_fake_all_lockbox",
               "visomaster_enhanced_macro_dev", "deeplive_enhanced_dev"]
SUITE_REAL_DEV = "teams_real_all_dev"
SUITE_REAL_LB = "teams_real_all_lockbox"


def merge_grids(s1_grid_path, s2_grid_path):
    """Merge S1 + S2 + P22 grids; tag rows by source."""
    parts = []
    if P22_GRID.exists():
        p22 = pd.read_csv(P22_GRID)
        p22["source"] = "p22"
        parts.append(p22)
    if s1_grid_path and Path(s1_grid_path).exists():
        s1 = pd.read_csv(s1_grid_path); s1["source"] = "s1"; parts.append(s1)
    if s2_grid_path and Path(s2_grid_path).exists():
        s2 = pd.read_csv(s2_grid_path); s2["source"] = "s2"; parts.append(s2)
    return pd.concat(parts, ignore_index=True) if parts else None


def joint_recal_at_floor(grid: pd.DataFrame, ckpt_key: str, floor: float):
    """Find smallest τ s.t. dev_primary AND lockbox real-FPR ≤ floor."""
    sub = grid[grid.checkpoint_key == ckpt_key].copy()
    if "teams_real_all_lockbox__real_fpr" not in sub.columns: return None
    valid = sub[(sub.dev_primary_real_fpr <= floor + 1e-9) &
                (sub["teams_real_all_lockbox__real_fpr"] <= floor + 1e-9)]
    if len(valid) == 0:
        return None
    return valid.loc[valid.threshold.idxmin()]


def chain_joint_recal_table(grid: pd.DataFrame, ckpts: list[str]):
    """For each ckpt × FPR floor, report compliance + per-suite recall."""
    rows = []
    for ckpt in ckpts:
        for floor in [0.02, 0.05, 0.10, 0.20]:
            chosen = joint_recal_at_floor(grid, ckpt, floor)
            if chosen is None:
                rows.append({"ckpt": ckpt, "floor": floor, "joint_compliant": False})
                continue
            row = {"ckpt": ckpt, "floor": floor, "joint_compliant": True,
                   "tau": float(chosen.threshold),
                   "dev_real_fpr": float(chosen.dev_primary_real_fpr),
                   "lockbox_real_fpr": float(chosen["teams_real_all_lockbox__real_fpr"])}
            for s in SUITES_FAKE:
                col = f"{s}__fake_recall"
                if col in chosen.index:
                    row[f"{s}_recall"] = float(chosen[col])
            rows.append(row)
    return pd.DataFrame(rows)


def min_fpr_for_recall(grid: pd.DataFrame, ckpts: list[str], target=0.9):
    """For each (ckpt, fake suite), find smallest joint-FPR floor to hit recall ≥ target."""
    floors = np.linspace(0.005, 0.50, 100)
    rows = []
    for ckpt in ckpts:
        sub = grid[grid.checkpoint_key == ckpt]
        if "teams_real_all_lockbox__real_fpr" not in sub.columns:
            continue
        for s in SUITES_FAKE:
            col = f"{s}__fake_recall"
            if col not in sub.columns:
                continue
            best = None
            for floor in floors:
                valid = sub[(sub.dev_primary_real_fpr <= floor + 1e-9) &
                            (sub["teams_real_all_lockbox__real_fpr"] <= floor + 1e-9)]
                if len(valid) == 0: continue
                if valid[col].max() >= target:
                    best = float(floor); break
            rows.append({"ckpt": ckpt, "fake_suite": s,
                         "min_fpr_for_target": best,
                         "max_recall_observed": float(sub[col].max())})
    return pd.DataFrame(rows)


def parse_identity(video_id):
    if pd.isna(video_id): return "<missing>"
    parts = re.split(r"__s\d+|__seg_|__seq\d+", video_id)
    return parts[0] if parts else video_id


def per_identity_fpr_at_floor(frames_dir: Path, ckpt_lower: str, dev_real_dir: Path, floor=0.02):
    """Pull lockbox-real per-frame CSV, join to identity, compute per-id FP at dev-cal-τ."""
    # Find lockbox CSV
    lockbox_csv = None
    real_dev_csv = None
    for p in frames_dir.rglob("*.csv"):
        n = p.name.lower()
        if "frames_report" not in n: continue
        if ckpt_lower not in n: continue
        if "teams_real_all_lockbox" in n: lockbox_csv = p
        elif "teams_real_all_dev" in n: real_dev_csv = p
    if lockbox_csv is None or real_dev_csv is None: return None

    # Calibrate τ on dev
    rd = pd.read_csv(real_dev_csv).frame_prob.to_numpy()
    rd_sorted = np.sort(rd)
    idx = int(np.ceil(len(rd_sorted) * (1 - floor)))
    tau = float(rd_sorted[min(idx, len(rd_sorted) - 1)])

    df = pd.read_csv(lockbox_csv)
    df["identity"] = df["video_id"].apply(parse_identity)
    df["is_FP"] = (df.frame_prob >= tau).astype(int)
    by_id = df.groupby("identity").agg(
        n_frames=("frame_prob", "count"),
        n_FP=("is_FP", "sum"),
    ).reset_index()
    by_id["fpr"] = by_id["n_FP"] / by_id["n_frames"]
    by_id["ckpt_lower"] = ckpt_lower
    by_id["tau_dev_cal"] = tau
    return by_id


def compute_pearson_full_viso(frames_dir: Path, ckpt_lower: str):
    """Pearson r(score, lap) on full viso using fetched lap from P22 follow-ups."""
    viso_csv = None
    for p in frames_dir.rglob("*.csv"):
        n = p.name.lower()
        if "frames_report" not in n: continue
        if ckpt_lower not in n: continue
        if "visomaster_enhanced_macro_dev" in n: viso_csv = p; break
    if viso_csv is None: return None, 0
    scores = pd.read_csv(viso_csv)
    lap = pd.read_csv(VISO_LAP) if VISO_LAP.exists() else pd.DataFrame()
    if ATTRS.exists():
        attrs = pd.read_csv(ATTRS).dropna(subset=["laplacian_var"])
        all_lap = pd.concat([lap[["frame_path", "laplacian_var"]],
                              attrs[["frame_path", "laplacian_var"]]],
                             ignore_index=True).drop_duplicates("frame_path")
    else:
        all_lap = lap
    if len(all_lap) == 0: return None, 0
    merged = scores.merge(all_lap, on="frame_path", how="inner").dropna(subset=["frame_prob", "laplacian_var"])
    if len(merged) < 30: return None, len(merged)
    r, _ = pearsonr(merged.frame_prob, merged.laplacian_var)
    return float(r), int(len(merged))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1-grid")
    ap.add_argument("--s2-grid")
    ap.add_argument("--s1-frames-dir")
    ap.add_argument("--s2-frames-dir")
    ap.add_argument("--s1-ckpt-lower-pat", help="e.g. s1_redux_short_step200")
    ap.add_argument("--s2-ckpt-lower-pat", help="e.g. s2_earlier_base_step200")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    grid = merge_grids(args.s1_grid, args.s2_grid)
    if grid is None:
        print("ERROR: no scorecard grids found")
        return

    # Find all ckpt keys in the merged grid
    ckpts = sorted(grid.checkpoint_key.dropna().unique())
    print(f"Found {len(ckpts)} ckpts in merged grid:")
    for c in ckpts: print(f"  - {c}")

    # 1) Joint recal table
    jr = chain_joint_recal_table(grid, ckpts)
    jr.to_csv(OUT / "04_chain_joint_recal_grid.csv", index=False)
    print(f"\nJoint recal grid: {OUT}/04_chain_joint_recal_grid.csv ({len(jr)} rows)")

    # 2) Min FPR for 90% recall
    mf = min_fpr_for_recall(grid, ckpts, target=0.9)
    mf.to_csv(OUT / "04_chain_min_fpr_for_90pct_recall.csv", index=False)
    print(f"Min-FPR-for-90pct: {OUT}/04_chain_min_fpr_for_90pct_recall.csv ({len(mf)} rows)")

    # 3) Pretty print the key view
    print("\n" + "=" * 110)
    print("KEY VIEW: per-suite recall at joint dev+lockbox FPR=10%")
    print("=" * 110)
    sub = jr[(jr.floor == 0.10) & jr.joint_compliant].copy()
    cols = ["ckpt", "tau", "dev_real_fpr", "lockbox_real_fpr"] + [f"{s}_recall" for s in SUITES_FAKE if f"{s}_recall" in sub.columns]
    print(sub[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))

    # 4) Pearson + per-identity (CPU sub-jobs that need frames-dir + ckpt-pat)
    if args.s1_frames_dir and args.s1_ckpt_lower_pat:
        r, n = compute_pearson_full_viso(Path(args.s1_frames_dir), args.s1_ckpt_lower_pat)
        if r is not None:
            print(f"\nS1 viso Pearson r (n={n}): {r:+.4f} (P8A baseline +0.5074, P22 step1k +0.4471)")
        ident = per_identity_fpr_at_floor(Path(args.s1_frames_dir), args.s1_ckpt_lower_pat, None)
        if ident is not None:
            ident.to_csv(OUT / "04_chain_per_identity_fpr_s1.csv", index=False)
            print(f"\nS1 lockbox per-identity FPR:")
            print(ident.sort_values("n_FP", ascending=False).to_string(index=False))

    if args.s2_frames_dir and args.s2_ckpt_lower_pat:
        r, n = compute_pearson_full_viso(Path(args.s2_frames_dir), args.s2_ckpt_lower_pat)
        if r is not None:
            print(f"\nS2 viso Pearson r (n={n}): {r:+.4f}")
        ident = per_identity_fpr_at_floor(Path(args.s2_frames_dir), args.s2_ckpt_lower_pat, None)
        if ident is not None:
            ident.to_csv(OUT / "04_chain_per_identity_fpr_s2.csv", index=False)
            print(f"\nS2 lockbox per-identity FPR:")
            print(ident.sort_values("n_FP", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
