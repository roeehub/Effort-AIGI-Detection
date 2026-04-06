# Phase 2 Round 5 (R5) — Scratch Improvement Cycle

Date: February 17, 2026

## Objectives

1. Keep FT7 as the provisional deployment-validation candidate.
2. Launch a scratch retraining cycle that preserves architecture/loss and only changes:
   - Data mix emphasis
   - Sampling strategy
   - Augmentation realism
3. Validate all candidates with canonical gates:
   - External real FPR `<= 8%`
   - WMA flat per-image fake detection `>= 50%` (practical target `>= 85%`)
   - Enhanced DeepLive fake TPR `>= 95%`
   - Maintain DF40 fake strength

## Scratch Matrix (Minimal 3 Runs)

1. `R5_S1_scratch_baseline_ft7mix.yaml`
2. `R5_S2_scratch_ft7mix_weighted.yaml`
3. `R5_S3_scratch_ft7mix_weighted_targetdomain_aug.yaml`
4. `R5_SMOKE_30MIN_scratch_ft7mix_weighted.yaml` (fast integration smoke check)

All runs are from scratch (`load_base_checkpoint: false`) and use the same backbone/loss family as R4.

## In-Training OOD Monitoring

R5 configs now enable OOD monitoring directly in training using external sets:
1. External real (folder-grouped videos): `real/external_youtube_avspeech` (cap `200`)
2. External real Zoom VCD (per-image): `real/VCD` (cap `1200`)
3. WMA fake (per-image): `wma_validation/enhanced_fake` (cap `1202`)

Cadence:
1. Start at step `1000`
2. Run every `2000` steps

This is monitoring-only and does not affect gradients.

## Holdout Semantics

R5 configs use method-level holdout for validation holdout:
1. All DF40 `target_source` methods (auto-derived from `df40-pair-matching.json`)
2. Selected VisoMaster methods (`Inswapper128`, `GhostFace-v2`)
3. Holdout cap: `300` samples per holdout method

DeepLive `edge_cases_enhanced` remains in the training pool.

These holdout methods are excluded from training and evaluated in `val_holdout`.

## Discovery Caching

VisoMaster discovery now supports remote cache manifests (e.g., `gs://...json`) with:
1. TTL (`cache_max_age_hours`)
2. Optional listing-signature validation (`cache_validate_listing`)
3. Manual invalidation token (`cache_revision`)

This avoids re-downloading all manifests on every launch while still refreshing when data changes.

## Runtime Fixes Applied

1. OOD cadence flags are now propagated from YAML into runtime config:
   - `ood_monitoring_enabled`
   - `ood_monitoring_start_step`
   - `ood_monitoring_every_steps`
2. Top-level `seed` in param-config now propagates correctly to training.
3. DeepLive landmark miss handling is hardened:
   - Missing landmark files are cached to avoid repeated GCS 404 spam.
   - Samples with no usable landmarks are marked and skipped for landmark reload.
4. R5 overnight configs explicitly set `deeplive.use_landmarks: false` for speed/stability.

## Overnight 6-Run Matrix

1. `R5_S1_scratch_baseline_ft7mix.yaml`
2. `R5_S2_scratch_ft7mix_weighted.yaml`
3. `R5_S3_scratch_ft7mix_weighted_targetdomain_aug.yaml`
4. `R5_S1_scratch_baseline_ft7mix_seed1337.yaml`
5. `R5_S2_scratch_ft7mix_weighted_seed1337.yaml`
6. `R5_S3_scratch_ft7mix_weighted_targetdomain_aug_seed1337.yaml`

Launch all 6 in `asia-southeast1`:

```bash
cd DeepfakeBench/training
./launch_r5_overnight_6.sh <WANDB_PROJECT> asia-southeast1
```

## Launch Example

```bash
cd DeepfakeBench/training
python train_sweep.py --param-config experiments/phase2_round5/R5_S1_scratch_baseline_ft7mix.yaml
python train_sweep.py --param-config experiments/phase2_round5/R5_S2_scratch_ft7mix_weighted.yaml
python train_sweep.py --param-config experiments/phase2_round5/R5_S3_scratch_ft7mix_weighted_targetdomain_aug.yaml
```

## Validation Scripts

### R4 Sidecar (explicit WMA grouping semantics)

```bash
python run_r4_validation_sequential.py \
  --checkpoints FT5,FT7 \
  --checkpoint_map experiments/phase2_round5/R4_PROVISIONAL_CHECKPOINTS.json \
  --external_fake_grouping per_image \
  --external_fake_deterministic \
  --max_external_fake 1202 \
  --output_gcs_folder gs://training-job-outputs/test_results/r4_sidecar_refresh
```

### Target-Domain Suites (Zoom reality)

```bash
python run_target_domain_validation_sequential.py \
  --checkpoints FT7 \
  --checkpoint_map experiments/phase2_round5/R4_PROVISIONAL_CHECKPOINTS.json \
  --suite_manifest experiments/phase2_round5/TARGET_DOMAIN_SUITES_TEMPLATE.yaml \
  --output_gcs_folder gs://training-job-outputs/test_results/target_domain_validation
```
