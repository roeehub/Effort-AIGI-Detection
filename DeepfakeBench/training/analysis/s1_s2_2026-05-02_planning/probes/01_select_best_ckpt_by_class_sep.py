"""Probe 1 — pick best ckpt by `train/collapse/class_separation`.

Once an S1 or S2 run has finished training, we have ~10 periodic-save ckpts
(steps 100-1000). The right ckpt to evaluate is the one where
`train/collapse/class_separation` peaked — NOT the one with highest val_AUC
(which misled us on P22 step8k).

Pulls W&B history for the run, identifies peak class_separation, and
reports the GCS path of the closest periodic save.

Usage:
    python 01_select_best_ckpt_by_class_sep.py --run-id <wandb_id>
        [--project phase2r13-experiments]
        [--saves-bucket gs://training-job-outputs/phase2r13_experiments/<run_id>/]

Output:
    01_select_best_ckpt_<run_id>.json — { peak_step, peak_class_sep, peak_logit_std,
                                          recommended_ckpt_path, top3_candidates }
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

import wandb

OUT = Path("analysis/s1_s2_2026-05-02_planning/probes/outputs")


def fetch_history(api, entity, project, run_id):
    run = api.run(f"{entity}/{project}/{run_id}")
    keys = [
        "train/step", "train/collapse/class_separation",
        "train/collapse/logit_std", "train/collapse/prob_entropy",
        "val_holdout/overall/auc", "val_in_dist/overall/auc",
        "train/loss/overall", "train/arcface/s",
    ]
    return run.history(samples=20000), run


def find_peak_step(df, metric_col, step_col="train/step"):
    sub = df.dropna(subset=[metric_col, step_col])
    if len(sub) == 0:
        return None, None
    peak_idx = sub[metric_col].idxmax()
    return float(sub.loc[peak_idx, step_col]), float(sub.loc[peak_idx, metric_col])


def find_closest_periodic_save(saves_uri: str, target_step: int):
    """List GCS bucket and find the periodic save closest to target_step.
    Returns (path, actual_step) or (None, None)."""
    import subprocess
    try:
        result = subprocess.run(
            ["gsutil", "ls", saves_uri.rstrip("/") + "/*.pth"],
            capture_output=True, text=True, timeout=120
        )
    except Exception as e:
        return None, None
    if result.returncode != 0:
        return None, None
    paths = [p for p in result.stdout.splitlines() if p.startswith("gs://")]
    pat = re.compile(r"_step(\d+)_")
    candidates = []
    for p in paths:
        m = pat.search(p)
        if m:
            step = int(m.group(1))
            candidates.append((step, p))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: abs(x[0] - target_step))
    return candidates[0][1], candidates[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--entity", default="dtect-vision")
    ap.add_argument("--project", default="phase2r13-experiments")
    ap.add_argument("--saves-bucket", default=None,
                    help="GCS prefix for ckpts. Default: gs://training-job-outputs/phase2r13_experiments/<run_id>/")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    df, run = fetch_history(api, args.entity, args.project, args.run_id)
    print(f"Run {args.run_id}: {len(df)} W&B history rows")

    saves_uri = args.saves_bucket or f"gs://training-job-outputs/phase2r13_experiments/{args.run_id}/"

    metrics = {
        "class_separation": "train/collapse/class_separation",
        "logit_std": "train/collapse/logit_std",
        "val_holdout_auc": "val_holdout/overall/auc",
        "val_in_dist_auc": "val_in_dist/overall/auc",
    }

    summary = {"run_id": args.run_id, "saves_uri": saves_uri, "peaks": {}}
    for label, col in metrics.items():
        if col not in df.columns:
            print(f"  metric {col} missing in W&B — skipping")
            continue
        step, peak = find_peak_step(df, col)
        if step is None:
            continue
        ckpt_path, ckpt_step = find_closest_periodic_save(saves_uri, int(step))
        summary["peaks"][label] = {
            "peak_step": step, "peak_value": peak,
            "closest_ckpt_step": ckpt_step, "closest_ckpt_path": ckpt_path,
        }
        print(f"\n  {label}: peak={peak:.4f} at step {step:.0f}")
        if ckpt_path:
            print(f"    closest periodic save: step {ckpt_step} → {ckpt_path}")

    # Top-3 by class_separation (the canonical selector)
    if "class_separation" in summary["peaks"]:
        cs_col = metrics["class_separation"]
        sub = df.dropna(subset=[cs_col, "train/step"]).sort_values(cs_col, ascending=False).head(20)
        seen_steps = set()
        top3 = []
        for _, row in sub.iterrows():
            step_int = int(row["train/step"])
            # snap to nearest periodic save
            ckpt_path, ckpt_step = find_closest_periodic_save(saves_uri, step_int)
            if ckpt_step is None or ckpt_step in seen_steps:
                continue
            seen_steps.add(ckpt_step)
            top3.append({
                "wandb_step": step_int, "class_separation": float(row[cs_col]),
                "ckpt_step": ckpt_step, "ckpt_path": ckpt_path
            })
            if len(top3) >= 3: break
        summary["top3_by_class_separation"] = top3
        print("\nTop-3 ckpts by class_separation (deduplicated):")
        for i, t in enumerate(top3):
            print(f"  #{i+1}: ckpt step {t['ckpt_step']} (class_sep {t['class_separation']:.4f})")
            print(f"       {t['ckpt_path']}")

    out_path = OUT / f"01_select_best_ckpt_{args.run_id}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
