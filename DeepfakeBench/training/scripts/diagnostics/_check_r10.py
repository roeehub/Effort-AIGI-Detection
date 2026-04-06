#!/usr/bin/env python3
"""Quick R10 health check — current status."""
import wandb
api = wandb.Api()

def f(v, d=4):
    return f"{v:.{d}f}" if isinstance(v, (int, float)) else "—"

print("R10 STATUS CHECK")
print(f"{'Run':<50} {'State':10} {'Step':>6} {'BestAUC':>8} {'BestEER':>8} {'OOD':>8} {'Unified':>8} {'TeamsHO':>8}")
print("-"*120)

for r in api.runs("dtect-vision/phase2r10-experiments", order="-created_at"):
    name = r.config.get("experiment_name", r.name)
    if r.state not in ("running", "finished"):
        # Show crashed/failed too but mark them
        if r.state in ("crashed", "failed"):
            s = r.summary
            print(f"{name:<50} {r.state:10} {s.get('_step',0):>6}")
        continue
    s = r.summary
    step = s.get("_step", 0)
    ba = s.get('best/auc')
    be = s.get('best/eer')
    oa = s.get('ood/overall/auc')
    ua = s.get('unified/auc')
    
    # Teams holdout avg
    teams_ho = {}
    for k, v in s.items():
        if 'val_holdout/method/deeplive_teams_' in k and '/acc' in k:
            method = k.split('deeplive_teams_')[1].replace('/acc', '')
            teams_ho[method] = v
    teams_avg = sum(teams_ho.values()) / len(teams_ho) if teams_ho else None
    
    print(f"{name:<50} {r.state:10} {step:>6} {f(ba):>8} {f(be):>8} {f(oa):>8} {f(ua):>8} {f(teams_avg):>8}")
