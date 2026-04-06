#!/usr/bin/env python3
"""Deep-dive into R11 runs: get early metrics from history."""
import wandb

api = wandb.Api()
runs = api.runs("dtect-vision/phase2r11-experiments", order="-created_at")

METRIC_KEYS = [
    "_step", "_runtime", "train_loss", "val_auc", "val_eer",
    "best_auc", "best_eer", "best_ood_auc", "best_unified",
    "teams_holdout_avg", "ood_auc", "unified_score",
    "val/teams_v2_real_acc", "val/teams_v2_fake_acc",
    "cosine_sim_mean", "S_res_max",
]

for r in runs:
    step = r.summary.get("_step", 0)
    runtime = r.summary.get("_runtime", 0)
    runtime_h = runtime / 3600 if runtime else 0
    total_steps = r.config.get("train_total_steps", "?")
    lr = r.config.get("optimizer", {}).get("lr", "?")
    
    print(f"\n{'='*70}")
    print(f"{r.name}")
    print(f"  State: {r.state}  |  Step: {step}/{total_steps}  |  Runtime: {runtime_h:.1f}h  |  LR: {lr}")
    
    # Get the latest summary metrics
    summary_metrics = {}
    for k in METRIC_KEYS:
        v = r.summary.get(k)
        if v is not None and k != "_step" and k != "_runtime":
            summary_metrics[k] = v
    if summary_metrics:
        print(f"  Summary: {summary_metrics}")
    
    # Get last few rows of history with validation metrics
    try:
        rows = list(r.scan_history(
            keys=["_step", "train_loss", "val_auc", "val_eer", "best_auc", "best_eer", 
                  "best_ood_auc", "teams_holdout_avg", "ood_auc"],
            min_step=max(0, step - 2000)
        ))
        val_rows = [row for row in rows if row.get("val_auc") is not None]
        if val_rows:
            print(f"  Validation checkpoints ({len(val_rows)} found):")
            for vr in val_rows[-5:]:  # last 5
                s = vr.get("_step", "?")
                auc = vr.get("val_auc", "")
                eer = vr.get("val_eer", "")
                bauc = vr.get("best_auc", "")
                beer = vr.get("best_eer", "")
                ood = vr.get("best_ood_auc", "")
                teams = vr.get("teams_holdout_avg", "")
                print(f"    step={s:>6}  val_auc={auc}  val_eer={eer}  best_auc={bauc}  best_eer={beer}  ood={ood}  teams={teams}")
        else:
            # Maybe no val yet — show train loss
            train_rows = [row for row in rows if row.get("train_loss") is not None]
            if train_rows:
                last = train_rows[-1]
                print(f"  No validation yet. Last train_loss at step {last.get('_step')}: {last.get('train_loss', '?'):.4f}")
            else:
                print(f"  No metrics logged yet in recent history.")
    except Exception as e:
        print(f"  Error fetching history: {e}")

print("\n\nDone.")
