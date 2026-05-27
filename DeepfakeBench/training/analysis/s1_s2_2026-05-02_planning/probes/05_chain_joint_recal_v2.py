"""Unified joint dev+lockbox τ-recal across the chain — using per-frame CSVs.

Threshold grids only have dev real FPR per τ; lockbox real FPR has to be
computed from per-frame scorecard data. Mimics the methodology from
analysis/p22_eval_2026-05-02/cpu_followups/scripts/01_joint_recal.py but
extended to S1 + S2 ckpts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("analysis/s1_s2_2026-05-02_planning/probes/outputs")

# Per-frame CSV directories
P22_FRAMES = Path("/tmp/p22_frames_local")
S1S2_FRAMES = Path("/tmp/scorecards_AB_frames")

# Map ckpt name -> (dir, file_pat)
CKPT_FRAMES = {
    "P8A_REFERENCE_STEP5000": ("p8a_reference_step5000", S1S2_FRAMES),
    "P22_AUG_STEP1000": ("p22_aug_step1000", S1S2_FRAMES),
    # P22 step4k/8k from the original P22 followups
    "P22_AUG_STEP4000": ("p22_aug_step4000", P22_FRAMES),
    "P22_AUG_STEP8000": ("p22_aug_step8000", P22_FRAMES),
    "S1_REDUX_SHORT_STEP100": ("s1_redux_short_step100", S1S2_FRAMES),
    "S1_REDUX_SHORT_STEP400": ("s1_redux_short_step400", S1S2_FRAMES),
    "S1_REDUX_SHORT_STEP600": ("s1_redux_short_step600", S1S2_FRAMES),
    "S2_EARLIER_BASE_STEP200": ("s2_earlier_base_step200", S1S2_FRAMES),
    "S2_EARLIER_BASE_STEP300": ("s2_earlier_base_step300", S1S2_FRAMES),
    "S2_EARLIER_BASE_STEP600": ("s2_earlier_base_step600", S1S2_FRAMES),
    "S3_VISO_WEIGHT_STEP500": ("s3_viso_weight_step500", Path("/tmp/scorecard_C_frames")),
    "S3_VISO_WEIGHT_STEP700": ("s3_viso_weight_step700", Path("/tmp/scorecard_C_frames")),
    "S3_VISO_WEIGHT_STEP1000": ("s3_viso_weight_step1000", Path("/tmp/scorecard_C_frames")),

}

FAKE_SUITES = ["teams_fake_all_dev", "teams_fake_all_lockbox",
                "visomaster_enhanced_macro_dev", "deeplive_enhanced_dev"]
FLOORS = [0.02, 0.05, 0.10, 0.20]
TAU_GRID = np.linspace(0.001, 0.999, 999)


def load_per_frame(ckpt_pat: str, dir_path: Path, suite: str):
    candidates = list(dir_path.rglob(f"*{suite}*{ckpt_pat}*frames_report*.csv"))
    if not candidates:
        # Try reverse order
        candidates = list(dir_path.rglob(f"*{ckpt_pat}*{suite}*frames_report*.csv"))
    if not candidates:
        return None
    return pd.read_csv(candidates[0])


def measure_ckpt(ckpt_key: str):
    pat, dir_path = CKPT_FRAMES[ckpt_key]
    real_dev = load_per_frame(pat, dir_path, "teams_real_all_dev")
    real_lb = load_per_frame(pat, dir_path, "teams_real_all_lockbox")
    if real_dev is None or real_lb is None:
        return None
    fakes = {}
    for s in FAKE_SUITES:
        f = load_per_frame(pat, dir_path, s)
        if f is not None:
            fakes[s] = f["frame_prob"].to_numpy()
    rd = real_dev["frame_prob"].to_numpy()
    rl = real_lb["frame_prob"].to_numpy()
    rows = []
    for tau in TAU_GRID:
        dev_fpr = float((rd >= tau).mean())
        lb_fpr = float((rl >= tau).mean())
        row = {"ckpt": ckpt_key, "tau": float(tau),
               "dev_fpr": dev_fpr, "lockbox_fpr": lb_fpr}
        for s, scores in fakes.items():
            row[f"{s}_recall"] = float((scores >= tau).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def joint_compliant_at_floor(df: pd.DataFrame, floor: float):
    valid = df[(df.dev_fpr <= floor + 1e-9) & (df.lockbox_fpr <= floor + 1e-9)]
    if len(valid) == 0: return None
    return valid.loc[valid.tau.idxmin()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary_rows = []
    for ckpt in CKPT_FRAMES:
        df = measure_ckpt(ckpt)
        if df is None:
            print(f"  [{ckpt}] missing per-frame data, skipping")
            continue
        all_rows.append(df)
        for floor in FLOORS:
            chosen = joint_compliant_at_floor(df, floor)
            row = {"ckpt": ckpt, "floor": floor, "joint_compliant": chosen is not None}
            if chosen is not None:
                row["tau"] = float(chosen.tau)
                row["dev_fpr"] = float(chosen.dev_fpr)
                row["lockbox_fpr"] = float(chosen.lockbox_fpr)
                for s in FAKE_SUITES:
                    col = f"{s}_recall"
                    if col in chosen.index:
                        row[col] = float(chosen[col])
            summary_rows.append(row)
    if all_rows:
        full = pd.concat(all_rows, ignore_index=True)
        full.to_csv(OUT / "05_chain_joint_full_grid.csv", index=False)
    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(OUT / "05_chain_joint_summary.csv", index=False)

    # Pretty print at FPR=10% (the user's regime)
    print("=" * 110)
    print("Joint dev+lockbox compliant best τ per ckpt — recall-vs-FPR view")
    print("=" * 110)
    for floor in FLOORS:
        sub = sdf[sdf.floor == floor].copy()
        cols = ["ckpt", "joint_compliant", "tau", "dev_fpr", "lockbox_fpr",
                "teams_fake_all_dev_recall", "visomaster_enhanced_macro_dev_recall",
                "deeplive_enhanced_dev_recall", "teams_fake_all_lockbox_recall"]
        cols = [c for c in cols if c in sub.columns]
        print(f"\n>>> FPR floor = {floor:.2f}")
        print(sub[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))


if __name__ == "__main__":
    main()
