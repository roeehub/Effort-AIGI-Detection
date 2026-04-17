# Track C Promotion Contract Runbook

**Date:** April 17, 2026  
**Scope:** authoritative calibrated promotion path for the frozen Teams evaluation contract

## Goal

Decide promotion with one explicit contract:

1. run the frozen promotion-authoritative Teams suites
2. fit a checkpoint-specific threshold on dev only
3. freeze that threshold
4. rank checkpoints on lockbox real FPR first, then lockbox fake recall

This is the only checked-in path that should be treated as promotion-authoritative.

## Promotion-Authoritative Vs Diagnostic-Only

| Artifact or path | Status | Why |
| --- | --- | --- |
| `arena/score_teams_promotion_contract.py` | promotion-authoritative | Fits thresholds on dev and freezes before lockbox |
| `arena/target_domain_suites.teams_promotion_contract_2026-04-17.yaml` | promotion-authoritative | Contains only the eight contract suites |
| `arena/checkpoint_maps/teams_target_domain.promotion_shortlist_2026-04-17.yaml` | promotion-authoritative | Freezes the five meaningful current candidates |
| `arena/run_teams_promotion_contract.sh` | promotion-authoritative | One-command local wrapper for reports plus calibrated contract outputs |
| `arena/launch_teams_promotion_contract.sh` | promotion-authoritative | One-command Vertex launcher for the same contract |
| `arena/run_target_domain_validation_sequential.py --scorecard_*` outputs | diagnostic-only | Fixed threshold `0.5`; useful comparison, not final promotion |
| `arena/run_teams_target_domain_scorecard.sh` | diagnostic-only | Convenience wrapper around fixed-`0.5` scorecards |
| `arena/launch_target_domain_scorecard.sh` | diagnostic-only | Vertex launcher for fixed-`0.5` scorecards |
| `arena/target_domain_suites.teams_manifest.frozen_2026-04-06.yaml` | diagnostic-first | Includes promotion suites plus extra per-session fake slices |
| `arena/target_domain_suites.r13_best_megaval_2026-04-13.yaml` | diagnostic-only | Mixed-source regression suite, not the promotion contract |
| training `val_holdout/*`, `ood/*`, `ood_composite` | diagnostic-only for promotion | Useful for shortlist reduction, not final deployment judgment |

## Current Authoritative Inputs

### Suite manifest

Use:

- `DeepfakeBench/training/arena/target_domain_suites.teams_promotion_contract_2026-04-17.yaml`

This manifest contains exactly:

- `teams_real_all_dev`
- `teams_real_poor_quality_dev`
- `teams_real_lighting_extreme_dev`
- `teams_fake_all_dev`
- `visomaster_enhanced_macro_dev`
- `deeplive_enhanced_dev`
- `teams_real_all_lockbox`
- `teams_fake_all_lockbox`

Important clarification:

- `teams_fake_all_lockbox` is now present in the checked-in authoritative suite path
- older fixed-threshold scorecards that predate this inclusion should be treated as incomplete on the fake-side lockbox axis

### Checkpoint shortlist

Use:

- `DeepfakeBench/training/arena/checkpoint_maps/teams_target_domain.promotion_shortlist_2026-04-17.yaml`

The frozen shortlist is:

- `R12_G_FP32`
- `R13_A_STEP15500`
- `R13_E_BESTSOFAR`
- `R13_FT7_FP32`
- `R13_FT9_FP32`

Not promoted into the authoritative shortlist:

- `R13_FT8_FP32`
- `R13_FT10_FP32`

Reason:

- under current local runtime truth they are likely near-duplicates of `FT7` and `FT9`

## Lane-Semantics Note

This contract scores the current frozen Teams evaluation slices exactly as they
exist today. It does **not** reinterpret current lane names as future
proper-data lanes.

Current stance:

- `teams_real_*` and `teams_fake_*` remain deployment-relevant frozen slices
- they are promotion-authoritative for current repo-stage checkpoint judgment
- they are **not** proof that the repo already has future provenance-clean proper-data naming

That keeps this path aligned with `WT-A` lane-truth caution and leaves future
exact-condition naming to `WT-F`.

## Local Run

From the training root:

```bash
cd DeepfakeBench/training

bash arena/run_teams_promotion_contract.sh --dry-run
```

Real run:

```bash
cd DeepfakeBench/training

bash arena/run_teams_promotion_contract.sh --checkpoints ALL
```

Outputs land under:

- local diagnostic sidecar:
  - `arena/promotion_contracts/<timestamp>/diagnostic_scorecard/`
- local authoritative contract:
  - `arena/promotion_contracts/<timestamp>/promotion_contract/`
- detailed reports:
  - `gs://training-job-outputs/test_results/teams_promotion_contract/<run-name>/reports/`

## Vertex Run

From the training root:

```bash
cd DeepfakeBench/training

bash arena/launch_teams_promotion_contract.sh --dry-run
```

Real launch:

```bash
cd DeepfakeBench/training

bash arena/launch_teams_promotion_contract.sh --checkpoints ALL
```

Vertex artifacts land under:

- reports:
  - `gs://training-job-outputs/test_results/teams_promotion_contract/<job-name>/reports/`
- diagnostic sidecar scorecards:
  - `gs://training-job-outputs/test_results/teams_promotion_contract/<job-name>/diagnostic_scorecard/`
- authoritative contract outputs:
  - `gs://training-job-outputs/test_results/teams_promotion_contract/<job-name>/promotion_contract/`

## How To Read The Outputs

### Diagnostic sidecar

These remain useful, but only for quick comparison:

- `scorecard.csv`
- `scorecard.wide.csv`
- `scorecard.int8_delta.csv`
- `scorecard.json`

Rule:

- never claim promotion from these alone because they stay at threshold `0.5`

### Authoritative contract outputs

- `threshold_grid.csv`
  - every candidate threshold considered for each checkpoint
- `selected_threshold_scorecard.csv`
  - per-suite metrics at the frozen threshold
- `checkpoint_summary.csv`
  - checkpoint ranking under the contract
- `promotion_contract.json`
  - full machine-readable payload
- `promotion_winner.json`
  - compact winner readout

Promotion read order:

1. `lockbox_real_fpr`
2. `lockbox_fake_recall`
3. `dev_primary_real_fpr`
4. `dev_worst_real_stress_fpr`
5. dev fake-suite recalls

## Practical Worker Rules

- Do not call a fixed-threshold `0.5` table the promotion result.
- Do not let the mixed-source mega-eval stand in for the promotion contract.
- Do not invent future proper-data suite names inside the current frozen contract.
- If a run omits `teams_fake_all_lockbox`, treat it as older or incomplete.
- If a worker needs broader regression evidence, run the diagnostic suites separately and report them as diagnostic-only.
