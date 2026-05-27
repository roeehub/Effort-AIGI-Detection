"""Finalize Option B: extend constant-τ per-human + comparison_table outputs to include OPTB heads.

Reads per_frame_baseline_v2.csv (which already has the 4 Option B head columns) and produces:
- per_human_baseline_v2.csv  — appended rows for prob_{LR,MLP}_OPTB{,_DEV}
- per_human_combined_with_baselines_v2.csv  — FT'd ckpts + ALL 8 baselines in one file
- comparison_table_v2.csv  — appended rows for FrozenCLIP_{LR,MLP}_OPTB{,_DEV}
- comparison_table_v2_constanttau.md  — markdown form of the above

Reuses the same MODES dict and aggregation logic as scripts/train_and_score.py.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPANDED_READOUT = REPO_ROOT / "analysis/team_identity_deploy_readout_expanded_2026-05-23"
THIS_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = THIS_DIR / "outputs"

PER_HUMAN_SUMMARY_FT = EXPANDED_READOUT / "outputs/per_human_summary.csv"
PER_FRAME_V2 = OUTPUTS / "per_frame_baseline_v2.csv"

MODES = {
    "mode_A_tau_0_535": 0.535,
    "mode_B_tau_0_78": 0.78,
    "mode_C_tau_0_87": 0.87,
}

ALL_HEAD_COLS = [
    "prob_LR_DEV", "prob_MLP_DEV", "prob_LR_DEV_LB", "prob_MLP_DEV_LB",
    "prob_LR_OPTB", "prob_MLP_OPTB", "prob_LR_OPTB_DEV", "prob_MLP_OPTB_DEV",
]
BASELINE_CKPT_NAMES = {
    "prob_LR_DEV":         "FrozenCLIP_LR_DEV",
    "prob_MLP_DEV":        "FrozenCLIP_MLP_DEV",
    "prob_LR_DEV_LB":      "FrozenCLIP_LR_DEV_LB",
    "prob_MLP_DEV_LB":     "FrozenCLIP_MLP_DEV_LB",
    "prob_LR_OPTB":        "FrozenCLIP_LR_OPTB",
    "prob_MLP_OPTB":       "FrozenCLIP_MLP_OPTB",
    "prob_LR_OPTB_DEV":    "FrozenCLIP_LR_OPTB_DEV",
    "prob_MLP_OPTB_DEV":   "FrozenCLIP_MLP_OPTB_DEV",
}

logger = logging.getLogger("finalize_optb")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        handlers=[
            logging.FileHandler(OUTPUTS / "_finalize_optb.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    t0 = time.time()

    team_df = pd.read_csv(PER_FRAME_V2, low_memory=False)
    team_df["deploy_relevant"] = team_df["deploy_relevant"].astype(bool)
    logger.info("team frames: %d (deploy_relevant=%d)", len(team_df), int(team_df["deploy_relevant"].sum()))

    deploy_team = team_df[team_df.deploy_relevant].reset_index(drop=True)

    per_human_rows = []
    for head_col in ALL_HEAD_COLS:
        for human in sorted(deploy_team.human.unique()):
            sub = deploy_team[deploy_team.human == human]
            for role in sorted(sub.role.unique()):
                ss = sub[sub.role == role]
                probs = ss[head_col].to_numpy()
                pmask = ~np.isnan(probs)
                p = probs[pmask]
                n_total = len(probs)
                n_scored = int(pmask.sum())
                if n_scored == 0:
                    continue
                row = {
                    "head": head_col,
                    "human": human,
                    "role": role,
                    "n_frames": n_total,
                    "n_scored": n_scored,
                    "mean_prob": float(p.mean()),
                }
                for mode_name, tau in MODES.items():
                    row[f"metric_{mode_name}"] = float((p >= tau).mean())
                per_human_rows.append(row)

    per_human_df = pd.DataFrame(per_human_rows)
    per_human_df.to_csv(OUTPUTS / "per_human_baseline_v2.csv", index=False)
    logger.info("wrote %s (%d rows)", OUTPUTS / "per_human_baseline_v2.csv", len(per_human_df))

    # Combined with FT'd
    ft_per_human = pd.read_csv(PER_HUMAN_SUMMARY_FT)
    aug_rows = []
    for _, r in per_human_df.iterrows():
        # tau_0_5 column for parity
        sub = team_df[(team_df.human == r["human"]) & (team_df.role == r["role"])]
        probs = sub[r["head"]].dropna().to_numpy()
        tau05 = float((probs >= 0.5).mean()) if len(probs) > 0 else float("nan")
        aug_rows.append({
            "ckpt": BASELINE_CKPT_NAMES[r["head"]],
            "human": r["human"],
            "role": r["role"],
            "n_frames": r["n_frames"],
            "n_scored": r["n_scored"],
            "mean_prob": r["mean_prob"],
            "metric_tau_0_5": tau05,
            "metric_mode_A_tau_0_535": r["metric_mode_A_tau_0_535"],
            "metric_mode_B_tau_0_78": r["metric_mode_B_tau_0_78"],
            "metric_mode_C_tau_0_87": r["metric_mode_C_tau_0_87"],
        })
    combined = pd.concat([ft_per_human, pd.DataFrame(aug_rows)], ignore_index=True)
    combined.to_csv(OUTPUTS / "per_human_combined_with_baselines_v2.csv", index=False)
    logger.info("wrote %s (%d rows)", OUTPUTS / "per_human_combined_with_baselines_v2.csv", len(combined))

    # Comparison table: one row per ckpt
    summary_rows = []
    all_ckpts = sorted(combined.ckpt.unique())
    for ck in all_ckpts:
        sub = combined[combined.ckpt == ck]
        real = sub[sub.role == "real"]
        fakes = sub[sub.role.str.startswith("fake_target_")]
        weighted_B = (real.metric_mode_B_tau_0_78 * real.n_scored).sum() / max(real.n_scored.sum(), 1)
        max_fpr_B = real.metric_mode_B_tau_0_78.max() if len(real) else float("nan")
        min_fake_recall_B = fakes.metric_mode_B_tau_0_78.min() if len(fakes) else float("nan")
        weighted_A = (real.metric_mode_A_tau_0_535 * real.n_scored).sum() / max(real.n_scored.sum(), 1)
        max_fpr_A = real.metric_mode_A_tau_0_535.max() if len(real) else float("nan")
        min_fake_recall_A = fakes.metric_mode_A_tau_0_535.min() if len(fakes) else float("nan")
        weighted_C = (real.metric_mode_C_tau_0_87 * real.n_scored).sum() / max(real.n_scored.sum(), 1)
        max_fpr_C = real.metric_mode_C_tau_0_87.max() if len(real) else float("nan")
        min_fake_recall_C = fakes.metric_mode_C_tau_0_87.min() if len(fakes) else float("nan")

        row = {
            "ckpt": ck,
            "team_aggregate_fpr_A": float(weighted_A),
            "team_max_fpr_A": float(max_fpr_A),
            "team_min_fake_recall_A": float(min_fake_recall_A),
            "team_aggregate_fpr_B": float(weighted_B),
            "team_max_fpr_B": float(max_fpr_B),
            "team_min_fake_recall_B": float(min_fake_recall_B),
            "team_aggregate_fpr_C": float(weighted_C),
            "team_max_fpr_C": float(max_fpr_C),
            "team_min_fake_recall_C": float(min_fake_recall_C),
        }
        for human in sorted(real.human.unique()):
            v = real[real.human == human].metric_mode_B_tau_0_78
            row[f"fpr_B__{human}"] = float(v.iloc[0]) if len(v) else float("nan")
        for human in sorted(fakes.human.unique()):
            v = fakes[fakes.human == human].metric_mode_B_tau_0_78
            row[f"fake_recall_B__{human}"] = float(v.iloc[0]) if len(v) else float("nan")
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUTS / "comparison_table_v2.csv", index=False)
    logger.info("wrote %s (%d rows)", OUTPUTS / "comparison_table_v2.csv", len(summary_df))

    # Markdown (constant-τ at mode B)
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}" if not np.isnan(v) else "nan"
        return str(v)

    lines = [
        "# Constant-τ comparison @ mode B (τ=0.78)",
        "",
        "Eight FT'd / frozen-CLIP heads scored against the 5,941 deploy-relevant team-identity frames.",
        "",
        "| Ckpt | Team-agg FPR @B | Team-max FPR @B | Team-min fake recall @B | Roee_Win FPR | dor FPR | Xinhe FPR | Xiang FPR | Noyn FPR | dor fake recall | Xinhe fake recall | Xiang fake recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    # Sort by team_min_fake_recall_B descending
    summary_sorted = summary_df.sort_values("team_min_fake_recall_B", ascending=False).reset_index(drop=True)
    for _, r in summary_sorted.iterrows():
        lines.append(
            f"| {r.ckpt} | {fmt(r.team_aggregate_fpr_B)} | {fmt(r.team_max_fpr_B)} | "
            f"{fmt(r.team_min_fake_recall_B)} | "
            f"{fmt(r.get('fpr_B__Roee_Windows', float('nan')))} | "
            f"{fmt(r.get('fpr_B__dor', float('nan')))} | "
            f"{fmt(r.get('fpr_B__Xinhe', float('nan')))} | "
            f"{fmt(r.get('fpr_B__Xiang', float('nan')))} | "
            f"{fmt(r.get('fpr_B__Noyn', float('nan')))} | "
            f"{fmt(r.get('fake_recall_B__dor', float('nan')))} | "
            f"{fmt(r.get('fake_recall_B__Xinhe', float('nan')))} | "
            f"{fmt(r.get('fake_recall_B__Xiang', float('nan')))} |"
        )
    (OUTPUTS / "comparison_table_v2_constanttau.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", OUTPUTS / "comparison_table_v2_constanttau.md")

    logger.info("DONE in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
