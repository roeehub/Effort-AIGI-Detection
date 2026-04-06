#!/usr/bin/env python3
"""Quick pre-launch audit of R9 experiment configs."""
import yaml, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

configs = [
    'R9_A_teams_ft_stability',
    'R9_B_teams_ft_nofixes',
    'R9_C_teams_scratch',
    'R9_D_stability_only',
    'R9_E_teams_ft_v2',
    'R9_F_teams_sim_aug',
    'R9_G_teams_both',
    'R9_SMOKE',
]

issues = []

for name in configs:
    path = f'experiments/phase2_round9/{name}.yaml'
    with open(path) as f:
        cfg = yaml.safe_load(f)

    ckpt = cfg.get('gcs_base_checkpoint', 'NOT SET')
    load_ckpt = cfg.get('load_base_checkpoint', 'NOT SET')
    teams_cfg = cfg.get('combined_paired', {}).get('teams', {})
    teams_enabled = teams_cfg.get('enabled', 'NOT SET')
    stability = cfg.get('stability_lambda', 'NOT SET')
    ls = cfg.get('label_smoothing', 'NOT SET')
    s_start = cfg.get('s_start', 'NOT SET')
    s_end = cfg.get('s_end', 'NOT SET')
    steps = cfg.get('total_training_steps', 'NOT SET')
    lr = cfg.get('learning_rate', 'NOT SET')
    aug = cfg.get('augmentation', {})
    teams_sim = aug.get('teams_codec_simulation', {})
    teams_sim_enabled = teams_sim.get('enabled', False)
    
    # Family weights
    sampling = cfg.get('combined_paired', {}).get('sampling', {})
    family_weights = sampling.get('family_weights', {})
    
    print(f'=== {name} ===')
    print(f'  checkpoint: {str(ckpt)[:90]}')
    print(f'  load_base_checkpoint: {load_ckpt}')
    print(f'  teams.enabled: {teams_enabled}')
    print(f'  teams_codec_simulation.enabled: {teams_sim_enabled}')
    print(f'  stability_lambda: {stability}')
    print(f'  label_smoothing: {ls}')
    print(f'  s_start/s_end: {s_start}/{s_end}')
    print(f'  lr: {lr}, steps: {steps}')
    print(f'  family_weights: {family_weights}')
    
    # Checks
    if '<' in str(ckpt) or 'placeholder' in str(ckpt).lower():
        issues.append(f'  BLOCKING: {name} has placeholder checkpoint path: {ckpt}')
    
    if name == 'R9_C_teams_scratch' and load_ckpt != False:
        issues.append(f'  BLOCKING: {name} (scratch) should have load_base_checkpoint: false, got {load_ckpt}')
    
    if name == 'R9_D_stability_only' and teams_enabled != False:
        issues.append(f'  BLOCKING: {name} (no-teams control) should have teams.enabled: false, got {teams_enabled}')
    
    if name == 'R9_F_teams_sim_aug' and teams_enabled != False:
        issues.append(f'  BLOCKING: {name} (sim-only) should have teams.enabled: false, got {teams_enabled}')
    
    if name == 'R9_F_teams_sim_aug' and not teams_sim_enabled:
        issues.append(f'  BLOCKING: {name} needs teams_codec_simulation.enabled: true, got {teams_sim_enabled}')
    
    if name == 'R9_B_teams_ft_nofixes':
        if stability not in [0, 0.0, 'NOT SET']:
            issues.append(f'  HIGH: {name} (no-fixes ablation) has stability_lambda={stability}, should be 0')
        if ls not in [0, 0.0, 'NOT SET']:
            issues.append(f'  HIGH: {name} (no-fixes ablation) has label_smoothing={ls}, should be 0')
    
    # Check Teams family weights exist when Teams is enabled
    if teams_enabled == True:
        if 'deeplive_teams_fake' not in family_weights:
            issues.append(f'  HIGH: {name} has teams enabled but missing deeplive_teams_fake in family_weights')
        if 'deeplive_teams_real' not in family_weights:
            issues.append(f'  HIGH: {name} has teams enabled but missing deeplive_teams_real in family_weights')
    
    # Check Teams family weights absent when Teams is disabled
    if teams_enabled == False:
        if 'deeplive_teams_fake' in family_weights:
            issues.append(f'  WARN: {name} has teams disabled but deeplive_teams_fake in family_weights (will try to sample non-existent data)')
        if 'deeplive_teams_real' in family_weights:
            issues.append(f'  WARN: {name} has teams disabled but deeplive_teams_real in family_weights')
    
    # Check GCS bucket names for typos
    for source in ['deeplive', 'visomaster', 'teams', 'df40']:
        src_cfg = cfg.get('combined_paired', {}).get(source, {})
        bucket = src_cfg.get('gcs_bucket', '')
        if bucket and not bucket.startswith('gs://') and ' ' in bucket:
            issues.append(f'  WARN: {name}.{source}.gcs_bucket has spaces: {bucket}')
    
    print()

print('=' * 60)
if issues:
    print(f'ISSUES FOUND ({len(issues)}):')
    for issue in issues:
        print(issue)
else:
    print('NO ISSUES FOUND - all configs look correct')
