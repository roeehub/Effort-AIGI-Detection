"""Multi-seed head probe sensitivity sweep.

Runs 5 seeds × 3 ckpts × 3 head kinds at 50 epochs each. Writes incrementally
to avoid losing work if killed.
"""
import sys
import os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, 'analysis/frozen_pair_head_probe_2026-05-06')

from run_probe_path_a import load_features_filtered, map_pair_features, train_head, evaluate_head
import numpy as np
import pandas as pd
from pathlib import Path
import json

OUT_CSV = Path('analysis/frozen_pair_head_probe_2026-05-06/multi_seed_metrics.csv')
OUT_AGG = Path('analysis/frozen_pair_head_probe_2026-05-06/multi_seed_aggregate.csv')

seeds = [737, 100, 200, 300, 400]
ckpts = ['p8a', 'e2b', 'clip_b16_raw']

all_results = []
for ckpt in ckpts:
    print(f"=== Loading {ckpt} ===", flush=True)
    feats = load_features_filtered(Path(f'analysis/path_a_launch_2026-05-07/outputs/{ckpt}_paired_features.npz'))
    pairs = map_pair_features(feats, Path('/tmp/pair_gaps.csv'))
    feat_arr = feats['features']
    real_idx_arr = pairs['real_idx'].to_numpy()
    fake_idx_arr = pairs['fake_idx'].to_numpy()
    pairs['group'] = pairs['method'].astype(str) + '__' + pairs.get('fake_transport', 'raw').astype(str)
    g_map = {g: i for i, g in enumerate(sorted(pairs['group'].unique()))}
    group_id = pairs['group'].map(g_map).to_numpy()

    for seed in seeds:
        rng = np.random.RandomState(seed)
        subjects = sorted(pairs['canonical_subject'].unique())
        rng.shuffle(subjects)
        n_test = max(1, int(len(subjects) * 0.2))
        test_subjects = set(subjects[:n_test])
        test_mask = pairs['canonical_subject'].isin(test_subjects).to_numpy()
        train_mask = ~test_mask

        for kind in ['ce', 'ce_pair', 'ce_pair_group']:
            head, _ = train_head(
                feat_arr,
                real_idx_arr[train_mask], fake_idx_arr[train_mask],
                group_id[train_mask] if kind == 'ce_pair_group' else None,
                head_kind=kind,
                margin=0.5, lambda_pair=0.2, epochs=50, seed=seed,
            )
            m = evaluate_head(head, feat_arr, real_idx_arr[test_mask], fake_idx_arr[test_mask])
            all_results.append({
                'ckpt': ckpt, 'seed': seed, 'kind': kind,
                'test_subj': sorted(list(test_subjects))[0],
                **m,
            })
            print(f'{ckpt}/{seed}/{kind}: auc={m["auc"]:.4f} p_f>r={m["p_fake_gt_real_on_pairs"]:.4f} acc={m["accuracy_at_tau_logit_0"]:.4f}', flush=True)
            # Write incrementally
            pd.DataFrame(all_results).to_csv(OUT_CSV, index=False)

# Aggregate across seeds
df = pd.DataFrame(all_results)
agg = df.groupby(['ckpt', 'kind']).agg(
    auc_mean=('auc', 'mean'), auc_std=('auc', 'std'),
    pgr_mean=('p_fake_gt_real_on_pairs', 'mean'), pgr_std=('p_fake_gt_real_on_pairs', 'std'),
    acc_mean=('accuracy_at_tau_logit_0', 'mean'), acc_std=('accuracy_at_tau_logit_0', 'std'),
).reset_index()
print()
print('=== Aggregate over seeds ===', flush=True)
print(agg.to_string(), flush=True)
agg.to_csv(OUT_AGG, index=False)

# Compute lift B-A on aggregate
print()
print('=== B-A and C-A lifts (aggregate) ===', flush=True)
for ckpt in ckpts:
    sub = df[df['ckpt'] == ckpt]
    a = sub[sub['kind'] == 'ce']
    b = sub[sub['kind'] == 'ce_pair']
    c = sub[sub['kind'] == 'ce_pair_group']
    lift_ba_pgr = (b['p_fake_gt_real_on_pairs'].mean() - a['p_fake_gt_real_on_pairs'].mean()) * 100
    lift_ba_acc = (b['accuracy_at_tau_logit_0'].mean() - a['accuracy_at_tau_logit_0'].mean()) * 100
    lift_ca_pgr = (c['p_fake_gt_real_on_pairs'].mean() - a['p_fake_gt_real_on_pairs'].mean()) * 100
    lift_ca_acc = (c['accuracy_at_tau_logit_0'].mean() - a['accuracy_at_tau_logit_0'].mean()) * 100
    print(f'{ckpt}: lift B-A pgr={lift_ba_pgr:+.3f}pp acc={lift_ba_acc:+.3f}pp; lift C-A pgr={lift_ca_pgr:+.3f}pp acc={lift_ca_acc:+.3f}pp', flush=True)

print('Wrote:', OUT_CSV, flush=True)
print('Wrote:', OUT_AGG, flush=True)
