#!/usr/bin/env python3
"""Comprehensive R11 status check with correct metric keys."""
import wandb

api = wandb.Api()
runs = api.runs("dtect-vision/phase2r11-experiments", order="-created_at")

# Teams-specific method keys
TEAMS_METHODS = [
    "deeplive_teams_edge_cases",
    "deeplive_teams_minimal_processing", 
    "deeplive_teams_quality_enhancement",
    "deeplive_teams_visomaster_CSCS",
    "deeplive_teams_visomaster_GhostFace-v1",
    "deeplive_teams_visomaster_GhostFace-v2",
    "deeplive_teams_visomaster_GhostFace-v3",
    "deeplive_teams_visomaster_InStyleSwapper256-A",
    "deeplive_teams_visomaster_InStyleSwapper256-B",
    "deeplive_teams_visomaster_Inswapper128",
]

WEAK_METHODS_R10 = [
    "visomaster_GhostFace-v2",
    "visomaster_InStyleSwapper256-B",
    "facedancer",
]

print("=" * 120)
print(f"{'R11 STATUS CHECK':^120}")
print("=" * 120)

# Header
print(f"\n{'Run':<45} {'State':<10} {'Step':>6} {'Epoch':>5} {'Runtime':>7} {'AUC':>7} {'EER':>7} {'TPR@1%':>7} {'OOD':>7} {'Unified':>7} {'HO-AUC':>7}")
print("-" * 120)

for r in sorted(runs, key=lambda x: x.name):
    s = r.summary
    step = s.get("_step", 0)
    epoch = s.get("epoch", "")
    runtime_h = s.get("_runtime", 0) / 3600 if s.get("_runtime") else 0
    total_steps = r.config.get("train_total_steps", "?")
    
    auc = s.get("best/auc", "")
    eer = s.get("best/eer", "")
    tpr1 = s.get("best/tpr_at_fpr1pct", "")
    ood = s.get("ood/overall/auc", "")
    unified = s.get("unified/auc", "")
    ho_auc = s.get("val_holdout/overall/auc", "")
    
    def fmt(v, digits=4):
        if isinstance(v, (int, float)):
            return f"{v:.{digits}f}"
        return str(v)[:7] if v else "  —"
    
    name_short = r.name.replace("_0307-", "@").replace("_0308-", "@")
    print(f"{name_short:<45} {r.state:<10} {step:>6} {fmt(epoch,0):>5} {runtime_h:>6.1f}h {fmt(auc):>7} {fmt(eer):>7} {fmt(tpr1):>7} {fmt(ood):>7} {fmt(unified):>7} {fmt(ho_auc):>7}")

# Detailed per-run breakdown
print("\n")
for r in sorted(runs, key=lambda x: x.name):
    s = r.summary
    step = s.get("_step", 0)
    runtime_h = s.get("_runtime", 0) / 3600 if s.get("_runtime") else 0
    
    name_short = r.name.replace("_0307-", "@").replace("_0308-", "@")
    print(f"\n{'─'*80}")
    print(f"  {name_short}  |  {r.state}  |  step={step}  |  {runtime_h:.1f}h")
    
    auc = s.get("best/auc", "—")
    eer = s.get("best/eer", "—")
    tpr1 = s.get("best/tpr_at_fpr1pct", "—")
    ood = s.get("ood/overall/auc", "—")
    unified = s.get("unified/auc", "—")
    
    if isinstance(auc, float):
        print(f"  In-Dist:  AUC={auc:.4f}  EER={eer:.4f}  TPR@1%={tpr1:.4f}")
        print(f"  OOD:      AUC={ood:.4f}" if isinstance(ood, float) else f"  OOD:      {ood}")
        print(f"  Unified:  AUC={unified:.4f}" if isinstance(unified, float) else f"  Unified:  {unified}")
        
        # Best checkpoint
        ckpt = s.get("overall_best_ckpt_gcs", "—")
        print(f"  Checkpoint: {ckpt}")
        
        # Weakest method
        weak_m = s.get("val_holdout/weakest/fake_method", "—")
        weak_a = s.get("val_holdout/weakest/fake_acc", "—")
        print(f"  Weakest holdout method: {weak_m} → {weak_a:.2%}" if isinstance(weak_a, float) else f"  Weakest: {weak_m} → {weak_a}")
        
        # Teams methods accuracy
        teams_accs = {}
        for m in TEAMS_METHODS:
            key = f"val_holdout/method/{m}/acc"
            v = s.get(key)
            if v is not None:
                teams_accs[m.replace("deeplive_teams_", "")] = v
        if teams_accs:
            avg = sum(teams_accs.values()) / len(teams_accs)
            print(f"  Teams methods avg: {avg:.2%}")
            for m, a in sorted(teams_accs.items(), key=lambda x: x[1]):
                print(f"    {m:<40s} {a:.2%}")
        
        # R10 weak methods
        print(f"  R10-weak methods:")
        for m in WEAK_METHODS_R10:
            key = f"val_holdout/method/{m}/acc"
            v = s.get(key) 
            if v is not None:
                print(f"    {m:<40s} {v:.2%}")
            else:
                print(f"    {m:<40s} —")
    else:
        print(f"  No validation metrics yet.")

# R9 baselines for comparison
print("\n")
print("=" * 120)
print("R9 BASELINES (for comparison)")
print("=" * 120)
r9_runs = api.runs("dtect-vision/phase2r9-experiments", order="-created_at")
for r in r9_runs:
    if "R9_D" in r.name or "R9_F" in r.name or "R9_A" in r.name:
        s = r.summary
        auc = s.get("best/auc", s.get("best_auc", "—"))
        eer = s.get("best/eer", s.get("best_eer", "—"))
        ood = s.get("ood/overall/auc", s.get("best_ood_auc", "—"))
        unified = s.get("unified/auc", s.get("best_unified", "—"))
        def fmt2(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {r.name:<55s} {r.state:<10s} AUC={fmt2(auc)}  EER={fmt2(eer)}  OOD={fmt2(ood)}  Unified={fmt2(unified)}")

print("\nDone.")
