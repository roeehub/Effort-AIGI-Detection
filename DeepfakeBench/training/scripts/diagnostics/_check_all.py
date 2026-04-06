#!/usr/bin/env python3
"""Check R10 crash details + R11 status."""
import wandb

api = wandb.Api()

# ── R11 ──
print("=" * 60)
print("R11 STATUS")
print("=" * 60)
try:
    runs = api.runs("dtect-vision/phase2r11-experiments", order="-created_at")
    if not runs:
        print("No R11 runs found.")
    for r in runs:
        step = r.summary.get("_step", 0)
        auc = r.summary.get("best_auc", "")
        eer = r.summary.get("best_eer", "")
        ood = r.summary.get("best_ood_auc", "")
        uni = r.summary.get("best_unified", "")
        teams = ""
        for k in ["teams_holdout_avg", "best_teams_holdout_avg"]:
            if k in r.summary:
                teams = r.summary[k]
                break
        print(f"  {r.name:50s} {r.state:12s} step={step:>6}  AUC={auc}  EER={eer}  OOD={ood}  Unified={uni}  Teams={teams}")
except Exception as e:
    print(f"  R11 project not found: {e}")

# ── R10 final status ──
print()
print("=" * 60)
print("R10 FINAL STATUS")
print("=" * 60)
runs = api.runs("dtect-vision/phase2r10-experiments", order="-created_at")

# Dedupe: keep only the latest run per name prefix (R10_X)
seen = {}
for r in runs:
    prefix = r.name.split("_0")[0]  # e.g. "R10_A_scratch_wide_aug"
    if prefix not in seen:
        seen[prefix] = r

for prefix in sorted(seen.keys()):
    r = seen[prefix]
    step = r.summary.get("_step", 0)
    auc = r.summary.get("best_auc", "")
    eer = r.summary.get("best_eer", "")
    ood = r.summary.get("best_ood_auc", "")
    uni = r.summary.get("best_unified", "")
    runtime = r.summary.get("_runtime", 0)
    runtime_h = runtime / 3600 if runtime else 0
    total_steps = r.config.get("train_total_steps", "?")
    print(f"  {r.name:55s} {r.state:10s} step={step:>6}/{total_steps}  runtime={runtime_h:.1f}h  AUC={auc}  EER={eer}  OOD={ood}")

# ── R10 crash details ──
print()
print("=" * 60)
print("R10 CRASHED RUNS — INVESTIGATION")
print("=" * 60)
for prefix in sorted(seen.keys()):
    r = seen[prefix]
    if r.state != "crashed":
        continue
    step = r.summary.get("_step", 0)
    runtime = r.summary.get("_runtime", 0)
    runtime_h = runtime / 3600 if runtime else 0
    total_steps = r.config.get("train_total_steps", "?")
    print(f"\n  {r.name}")
    print(f"    State: {r.state}, Step: {step}/{total_steps}, Runtime: {runtime_h:.1f}h")
    
    # Check last logged metrics
    try:
        rows = list(r.scan_history(keys=["_step", "train_loss", "val_auc", "best_auc", "best_eer"], 
                                    min_step=max(0, step - 10)))
        if rows:
            last = rows[-1]
            print(f"    Last logged: step={last.get('_step')}, train_loss={last.get('train_loss', '?')}, "
                  f"val_auc={last.get('val_auc', '?')}, best_auc={last.get('best_auc', '?')}")
    except Exception as e:
        print(f"    Could not fetch history: {e}")

    # Try to get GCP logs via run metadata
    metadata = r.metadata or {}
    if metadata:
        gpu = metadata.get("gpu", "?")
        gpu_count = metadata.get("gpu_count", "?")
        print(f"    GPU: {gpu} x{gpu_count}")

print("\nDone.")
