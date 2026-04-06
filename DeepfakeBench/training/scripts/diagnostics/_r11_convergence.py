#!/usr/bin/env python3
"""Get convergence history for R11_G and compare epoch-by-epoch for all runs."""
import wandb

api = wandb.Api()
runs = api.runs("dtect-vision/phase2r11-experiments", order="-created_at")

# Get R11_G detailed history
for r in runs:
    if "R11_G" not in r.name:
        continue
    print(f"=== {r.name} === (step={r.summary.get('_step')}, state={r.state})")
    rows = list(r.scan_history(
        keys=["_step", "best/auc", "best/eer", "best/tpr_at_fpr1pct", 
              "ood/overall/auc", "unified/auc",
              "val_holdout/weakest/fake_method", "val_holdout/weakest/fake_acc",
              "val_holdout/derived/macro_accuracy",
              "val_holdout/method/deeplive_teams_visomaster_GhostFace-v2/acc",
              "val_holdout/method/deeplive_teams_visomaster_GhostFace-v3/acc",
              "val_holdout/method/facedancer/acc",
              "svd/S_residual_max", "train/arcface/s"],
    ))
    # Filter to validation rows
    val_rows = [row for row in rows if row.get("best/auc") is not None]
    print(f"  Total val checkpoints: {len(val_rows)}")
    for vr in val_rows:
        s = vr.get("_step", "?")
        auc = vr.get("best/auc", "")
        eer = vr.get("best/eer", "")
        tpr1 = vr.get("best/tpr_at_fpr1pct", "")
        ood = vr.get("ood/overall/auc", "")
        uni = vr.get("unified/auc", "")
        gv2 = vr.get("val_holdout/method/deeplive_teams_visomaster_GhostFace-v2/acc", "")
        gv3 = vr.get("val_holdout/method/deeplive_teams_visomaster_GhostFace-v3/acc", "")
        fd = vr.get("val_holdout/method/facedancer/acc", "")
        smax = vr.get("svd/S_residual_max", "")
        arc_s = vr.get("train/arcface/s", "")
        def f(v, d=4): return f"{v:.{d}f}" if isinstance(v, float) else "  —"
        print(f"  step={s:>6}  AUC={f(auc)}  EER={f(eer)}  TPR@1%={f(tpr1)}  OOD={f(ood)}  Uni={f(uni)}  GF-v2={f(gv2,2)}  GF-v3={f(gv3,2)}  FD={f(fd,2)}  S_max={f(smax,3)}  arc_s={f(arc_s,1)}")
    break

# Compact comparison: all runs at their latest checkpoint
print("\n\n=== ALL RUNS — DELTA FROM LAST CHECK (12h → 17h) ===")
print(f"{'Run':<32} {'Step':>5} {'AUC':>7} {'EER':>7} {'TPR@1%':>7} {'OOD':>7} {'Unified':>7}")
print("-" * 90)

# Previous values at 12h check
prev = {
    "R11_A": {"auc": 0.9827, "eer": 0.0569, "tpr1": 0.8653, "ood": 0.9496, "uni": 0.9834},
    "R11_B": {"auc": 0.9827, "eer": 0.0569, "tpr1": 0.8609, "ood": 0.9497, "uni": 0.9831},
    "R11_C": {"auc": 0.9821, "eer": 0.0613, "tpr1": 0.8521, "ood": 0.9511, "uni": 0.9802},
    "R11_D": {"auc": 0.9829, "eer": 0.0613, "tpr1": 0.8477, "ood": 0.9399, "uni": 0.9820},
    "R11_E": {"auc": 0.9687, "eer": 0.0788, "tpr1": 0.7395, "ood": 0.8941, "uni": 0.9667},
    "R11_F": {"auc": 0.9828, "eer": 0.0569, "tpr1": 0.8631, "ood": 0.9491, "uni": 0.9797},
    "R11_G": {"auc": 0.9912, "eer": 0.0263, "tpr1": 0.8764, "ood": 0.9055, "uni": 0.9907},
    "R11_H": {"auc": 0.9810, "eer": 0.0569, "tpr1": 0.8322, "ood": 0.9532, "uni": 0.9794},
}

for r in sorted(runs, key=lambda x: x.name):
    s = r.summary
    name = r.name.split("_0307")[0].split("_0308")[0]
    prefix = name.split("_")[0] + "_" + name.split("_")[1]  # R11_X
    
    auc = s.get("best/auc", 0)
    eer = s.get("best/eer", 0)
    tpr1 = s.get("best/tpr_at_fpr1pct", 0)
    ood = s.get("ood/overall/auc", 0)
    uni = s.get("unified/auc", 0)
    step = s.get("_step", 0)
    
    p = prev.get(prefix, {})
    def delta(cur, key):
        old = p.get(key, cur)
        d = cur - old
        if abs(d) < 0.00005: return ""
        return f" ({d:+.4f})" if key != "eer" else f" ({d:+.4f})"
    
    print(f"{name:<32} {step:>5}  {auc:.4f}{delta(auc,'auc'):<10}  {eer:.4f}{delta(eer,'eer'):<10}  {tpr1:.4f}{delta(tpr1,'tpr1'):<10}  {ood:.4f}{delta(ood,'ood'):<10}  {uni:.4f}{delta(uni,'uni'):<10}")
