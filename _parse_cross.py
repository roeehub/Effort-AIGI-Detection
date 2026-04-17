#!/usr/bin/env python3
"""Parse cross-threshold and margin metrics for R12."""
import json, sys

with open(sys.argv[1]) as f:
    data = json.load(f)

project = data['result']['project']

for label in ['A', 'B', 'C', 'D', 'E', 'G', 'H']:
    run = project[label]
    name = run['displayName']
    
    ood_rows = run.get('ood_at_indist', [[]])[0]
    unified_rows = run.get('unified', [[]])[0]
    margin_rows = run.get('margin', [[]])[0]
    
    print(f'=== R12_{label}: {name} ===')
    
    # OOD at in-dist threshold
    if ood_rows:
        print(f'  OOD @ In-dist Threshold ({len(ood_rows)} evals):')
        for r in ood_rows:
            t = r.get('ood/at_indist/threshold', '?')
            acc = r.get('ood/at_indist/acc', '?')
            f1 = r.get('ood/at_indist/f1', '?')
            fpr = r.get('ood/at_indist/fpr', '?')
            fnr = r.get('ood/at_indist/fnr', '?')
            parts = []
            if isinstance(t, float): parts.append(f'thresh={t:.4f}')
            if isinstance(acc, float): parts.append(f'acc={acc:.4f}')
            if isinstance(f1, float): parts.append(f'f1={f1:.4f}')
            if isinstance(fpr, float): parts.append(f'fpr={fpr:.4f}')
            if isinstance(fnr, float): parts.append(f'fnr={fnr:.4f}')
            print(f'    {" ".join(parts)}')
    else:
        print(f'  OOD @ In-dist: NO DATA')
    
    # Unified threshold
    if unified_rows:
        print(f'  Unified Threshold ({len(unified_rows)} evals):')
        for r in unified_rows:
            t = r.get('unified/eer_threshold', '?')
            eer = r.get('unified/eer', '?')
            auc = r.get('unified/auc', '?')
            f1 = r.get('unified/f1', '?')
            parts = []
            if isinstance(t, float): parts.append(f'thresh={t:.4f}')
            if isinstance(eer, float): parts.append(f'eer={eer:.4f}')
            if isinstance(auc, float): parts.append(f'auc={auc:.4f}')
            if isinstance(f1, float): parts.append(f'f1={f1:.4f}')
            print(f'    {" ".join(parts)}')
    else:
        print(f'  Unified: NO DATA')
    
    # Score margin / TPR at strict FPR
    if margin_rows:
        latest = margin_rows[-1]
        tpr1 = latest.get('val_in_dist/overall/tpr_at_fpr1pct', '?')
        th1 = latest.get('val_in_dist/overall/thresh_at_fpr1pct', '?')
        tpr5 = latest.get('val_in_dist/overall/tpr_at_fpr5pct', '?')
        th5 = latest.get('val_in_dist/overall/thresh_at_fpr5pct', '?')
        print(f'  Score Margin (latest):')
        if isinstance(tpr1, float):
            print(f'    TPR@1%FPR: {tpr1:.4f} (thresh={th1:.4f})')
        if isinstance(tpr5, float):
            print(f'    TPR@5%FPR: {tpr5:.4f} (thresh={th5:.4f})')
        # Show trajectory
        tpr1_traj = [r.get('val_in_dist/overall/tpr_at_fpr1pct', '?') for r in margin_rows]
        print(f'    TPR@1%FPR trajectory: {" -> ".join(f"{v:.3f}" if isinstance(v, float) else "?" for v in tpr1_traj)}')
    else:
        print(f'  Score Margin: NO DATA')
    
    print()
