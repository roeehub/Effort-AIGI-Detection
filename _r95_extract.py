#!/usr/bin/env python3
import json

fp = '/Users/roeedar/Library/Application Support/Code/User/workspaceStorage/56912a3dd1f72c18f566d9ca5ef267f9/GitHub.copilot-chat/chat-session-resources/8d347c4e-f4d2-4d8a-83b6-56a222dcc296/toolu_bdrk_01Q5Y9cohfeqeEpdiZi7dGyK__vscode-1772226611657/content.json'
with open(fp) as f:
    data = json.load(f)

runs = []
for edge in data['result']['project']['runs']['edges']:
    node = edge['node']
    m = json.loads(node['summaryMetrics'])
    runs.append({
        'displayName': node['displayName'],
        'name': node['name'],
        'state': node['state'],
        'metrics': m
    })

runs.sort(key=lambda r: r['displayName'])

def g(m, key, fmt='.4f'):
    v = m.get(key)
    if v is None:
        return 'N/A'
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        if fmt == 'd':
            return str(int(v))
        return format(v, fmt)
    return str(v)

labels = [r['displayName'].split('_')[1] for r in runs]

def print_header():
    print('{:<47}'.format('Metric'), end='')
    for lb in labels:
        print('{:>12}'.format(lb), end='')
    print()
    print('-' * 119)

def print_row(metric_key, fmt='.4f'):
    lbl = metric_key
    if len(lbl) > 46:
        lbl = '...' + lbl[-43:]
    print('{:<47}'.format(lbl), end='')
    for r in runs:
        val = g(r['metrics'], metric_key, fmt)
        if len(str(val)) > 11:
            val = str(val)[:11]
        print('{:>12}'.format(val), end='')
    print()

print('=' * 119)
print('R9.5 SWEEP — 6 RUNS COMPARISON (all running, best_epoch=3 for all)')
print('=' * 119)

print()
print('## IDENTITY')
print_header()
print('{:<47}'.format('displayName'), end='')
for r in runs:
    dn = r['displayName'].replace('_0302-0009', '').replace('_0302-0008', '')
    print('{:>12}'.format(dn[:12]), end='')
print()
print('{:<47}'.format('name (ID)'), end='')
for r in runs:
    print('{:>12}'.format(r['name']), end='')
print()
print('{:<47}'.format('state'), end='')
for r in runs:
    print('{:>12}'.format(r['state']), end='')
print()

groups = [
    ('PROGRESS', [
        ('_step', 'd'), ('train/step', 'd'), ('epoch', 'd'),
    ]),
    ('CORE PERFORMANCE', [
        ('best/auc', '.4f'), ('best/eer', '.4f'), ('best/eer_threshold', '.4f'), ('best/epoch', 'd'),
        ('unified/auc', '.4f'), ('unified/eer', '.4f'), ('unified/eer_threshold', '.4f'),
        ('val_in_dist/overall/auc', '.4f'), ('val_in_dist/overall/eer', '.4f'),
        ('val_holdout/overall/auc', '.4f'), ('val_holdout/overall/eer', '.4f'), ('val_holdout/overall/loss', '.4f'),
    ]),
    ('STABILITY & TRAINING LOSS', [
        ('train/loss/stability', '.6f'),
        ('train/loss/cls_loss', '.4f'), ('train/loss/overall', '.4f'),
        ('train/loss/reg_loss', '.6f'), ('train/loss/reg_cls_ratio', '.4f'),
    ]),
    ('TEAMS DOMAIN (val_in_dist)', [
        ('val_in_dist/method/deeplive_teams_edge_cases/acc', '.4f'),
        ('val_in_dist/method/deeplive_teams_minimal_processing/acc', '.4f'),
        ('val_in_dist/method/deeplive_teams_quality_enhancement/acc', '.4f'),
    ]),
    ('TEAMS DOMAIN (val_holdout)', [
        ('val_holdout/method/deeplive_teams_edge_cases/acc', '.4f'),
        ('val_holdout/method/deeplive_teams_minimal_processing/acc', '.4f'),
        ('val_holdout/method/deeplive_teams_quality_enhancement/acc', '.4f'),
    ]),
    ('OOD OVERALL', [
        ('ood/overall/auc', '.4f'), ('ood/overall/eer', '.4f'), ('ood/overall/eer_threshold', '.4f'),
        ('ood/method/zoom_vcd_real_real/acc', '.4f'),
        ('ood/method/external_youtube_avspeech_real/acc', '.4f'),
        ('ood/method/wma_failure_fake_fake/acc', '.4f'),
        ('ood/score_jitter/external_youtube_avspeech', '.4f'),
        ('ood/score_jitter/wma_failure_fake', '.4f'),
        ('ood/score_jitter/zoom_vcd_real', '.4f'),
        ('ood/at_indist/acc', '.4f'), ('ood/at_indist/fpr', '.4f'), ('ood/at_indist/fnr', '.4f'),
    ]),
    ('FACEDANCER & WEAKEST', [
        ('val_holdout/method/facedancer/acc', '.4f'),
        ('val_in_dist/method/facedancer/acc', '.4f'),
        ('val_holdout/weakest/fake_method', 's'),
        ('val_holdout/weakest/fake_acc', '.4f'),
        ('val_in_dist/weakest/fake_method', 's'),
        ('val_in_dist/weakest/fake_acc', '.4f'),
    ]),
    ('ARCFACE', [
        ('train/arcface/s', '.4f'), ('train/arcface/scale', '.4f'),
    ]),
    ('CONFIDENCE & COLLAPSE', [
        ('train/confidence/mean', '.4f'), ('train/confidence/std', '.4f'),
        ('train/confidence/fraction_confident', '.4f'),
        ('train/collapse/prob_entropy', '.4f'), ('train/collapse/prob_spread', '.4f'),
        ('train/collapse/class_separation', '.4f'),
    ]),
    ('THRESHOLDS (best)', [
        ('best/thresh_at_fpr1pct', '.4f'), ('best/thresh_at_fpr2pct', '.4f'), ('best/thresh_at_fpr5pct', '.4f'),
        ('best/tpr_at_fpr1pct', '.4f'), ('best/tpr_at_fpr2pct', '.4f'), ('best/tpr_at_fpr5pct', '.4f'),
    ]),
    ('THRESHOLDS (holdout & OOD)', [
        ('val_holdout/overall/thresh_at_fpr1pct', '.4f'),
        ('val_holdout/overall/tpr_at_fpr1pct', '.4f'),
        ('ood/overall/thresh_at_fpr1pct', '.4f'),
        ('ood/overall/tpr_at_fpr1pct', '.4f'),
    ]),
    ('VAL PRIMARY', [
        ('val_primary/best_epoch', 'd'), ('val_primary/best_metric', '.4f'),
        ('val_primary/epochs_without_improvement', 'd'),
    ]),
    ('HOLDOUT METHODS (selected)', [
        ('val_holdout/method/e4s/acc', '.4f'),
        ('val_holdout/method/mobileswap/acc', '.4f'),
        ('val_holdout/method/simswap/acc', '.4f'),
        ('val_holdout/method/blendface/acc', '.4f'),
        ('val_holdout/method/external_vcd_real/acc', '.4f'),
    ]),
]

for group_name, metrics_list in groups:
    print()
    print('## ' + group_name)
    print_header()
    for mk, fmt in metrics_list:
        print_row(mk, fmt)

# GCS checkpoints separately (long strings)
print()
print('## GCS CHECKPOINTS')
for r in runs:
    lb = r['displayName'].split('_')[1]
    ckpt = r['metrics'].get('overall_best_ckpt_gcs', 'N/A')
    print('  {}: {}'.format(lb, ckpt))

# Rankings
print()
print('=' * 119)
print('RANKINGS & ANALYSIS')
print('=' * 119)

def rank(metric, reverse=True, fmt='.4f'):
    items = [(r['displayName'].split('_')[1], r['metrics'].get(metric, 0)) for r in runs]
    items.sort(key=lambda x: x[1], reverse=reverse)
    return items

def print_rank(title, items, fmt='.4f'):
    print()
    print('### ' + title)
    for i, (name, val) in enumerate(items):
        print('  {}. {}: {}'.format(i + 1, name, format(val, fmt)))

print_rank('best/auc (higher=better)', rank('best/auc'))
print_rank('best/eer (lower=better)', rank('best/eer', reverse=False))
print_rank('ood/overall/auc (higher=better)', rank('ood/overall/auc'))
print_rank('ood/overall/eer (lower=better)', rank('ood/overall/eer', reverse=False))
print_rank('zoom_vcd_real acc (higher=better)', rank('ood/method/zoom_vcd_real_real/acc'))
print_rank('youtube_avspeech acc (higher=better)', rank('ood/method/external_youtube_avspeech_real/acc'))
print_rank('Teams edge_cases holdout', rank('val_holdout/method/deeplive_teams_edge_cases/acc'))
print_rank('Teams edge_cases in_dist', rank('val_in_dist/method/deeplive_teams_edge_cases/acc'))
print_rank('Facedancer holdout', rank('val_holdout/method/facedancer/acc'))
print_rank('Facedancer in_dist', rank('val_in_dist/method/facedancer/acc'))
print_rank('train/loss/stability (lower=better)', rank('train/loss/stability', reverse=False), fmt='.6f')
print_rank('val_holdout/overall/loss (lower=better)', rank('val_holdout/overall/loss', reverse=False))
print_rank('ood/at_indist/fpr (lower=better)', rank('ood/at_indist/fpr', reverse=False))

# Key observations
print()
print('=' * 119)
print('KEY OBSERVATIONS')
print('=' * 119)

print("""
1. ALL 6 RUNS: best_epoch=3, state=running, epochs_without_improvement varies (C=1, rest=2)
   - C is behind (epoch 3 vs epoch 4 for others) and only at step ~6900 vs ~8200

2. BEST AUC CLUSTER: F (0.9900) ≈ A (0.9900) > E (0.9895) ≈ B (0.9894) > C (0.9887) >> D (0.9850)
   - D (scratch_stability, LR=2e-4) is clearly the weakest — 50 bps below top

3. OOD — BIG SPREAD: B (0.9758) >> A/F (0.9705) > D (0.9670) > E (0.9647) > C (0.9581)
   - B (light_stability) dominates OOD by a wide margin (+50-170 bps over others)
   - B also has best zoom_vcd_real (0.785) and lowest ood/at_indist/fpr (0.218)

4. TEAMS DOMAIN — UNIFORM across runs:
   - holdout: edge_cases=0.70 (C/D=0.75), min_proc=0.917 (D=1.0), qual_enh=1.0 (all)
   - in_dist: edge_cases=0.77-0.80, min_proc=1.0, qual_enh=0.83-1.0
   - D has best Teams holdout edge_cases (0.75) but worst overall AUC
   - C has best in_dist edge_cases (0.80) and in_dist min_proc (1.0)

5. FACEDANCER — PERSISTENT WEAKNESS:
   - holdout: C (0.636) > D (0.636) >> A/B/E/F (0.591) — still the weakest method
   - in_dist: C/D (0.625) > rest (0.594)
   - F (facedancer_focus) did NOT improve facedancer despite its name

6. STABILITY:
   - D has the best stability (0.0048) — likely due to higher LR smoothing faster
   - B is second best (0.0113)
   - C/E/A/F cluster around 0.012-0.021

7. e4s NOTABLE:
   - holdout: D=0.50, C=0.75, rest=0.625 — D is catastrophically weak on e4s
   - D's weakest method is e4s (not facedancer!) — unique among runs

8. ARCFACE: E has dramatically higher scale (13.22 vs ~8.0-8.4 for others)
   - This is the r8_scale variant — scale ~60% higher but didn't help OOD

9. HOLDOUT LOSS: B (0.255) << F (0.324) ≈ A (0.325) < E (0.327) < D (0.338) < C (0.376)
   - B has dramatically better generalization loss — 25% lower than runner-up

10. OOD TPR@FPR1%: B (0.380) >> F (0.285) > A (0.282) > E (0.245) > D (0.200) > C (0.182)
    - B is the clear winner for low-FPR operating points too
""")
