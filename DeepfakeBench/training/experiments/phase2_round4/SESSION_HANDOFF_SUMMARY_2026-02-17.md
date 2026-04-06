# Phase 2 Round 4 Session Handoff Summary

Date prepared: February 17, 2026  
Scope covered: end-to-end work done in the long planning/implementation/validation chat for R4/R4b

---

## 1) Problem Context We Started From

The core issue from deployment was:

- A model that looked strong in training/standard validation still failed in target-domain usage.
- Specifically:
  - Enhanced DeepLive-like fakes were often missed (fake predicted as real).
  - Some real videos/faces were falsely flagged as fake, with smoothness/quality suspected as a shortcut feature.

The working hypothesis was:

- The model was using quality/smoothness artifacts as shortcuts.
- The existing train/val setup did not fully expose or report this failure mode with enough granularity.

---

## 2) High-Level Objectives Agreed In This Session

1. Make augmentation quality-robust and family-aware (not only fake-label-aware).
2. Keep compatibility with `albumentations==0.4.6`.
3. Ensure train configs truly control with-enhanced vs no-enhanced DeepLive slices.
4. Add group-level reporting so bias/failure pockets are visible.
5. Enforce hard acceptance gates:
   - External real FPR <= 8%
   - WMA failure fake detection >= 50%

---

## 3) Main Plans That Were Discussed And Finalized

Two major planning phases were executed:

### 3.1 Initial R4 Plan

- Family-aware augmentation routing with `quality_targeted_family`.
- Group-of-interest reporting (`group_key`, `family_key`, `group_metrics.csv`).
- 6-run matrix (FT1-FT6) with/without enhanced and base/light/moderate augmentation.

### 3.2 Revised R4b Relaunch Plan

- Fix data-contract mismatch causing enhanced/non-enhanced strategy confusion.
- Add hard preflight checks so wrong data states fail fast.
- Keep train loop lightweight; move full evidence generation to sidecar validation jobs.
- Expand to 8-run matrix by adding FT7/FT8 weighted DeepLive-targeted sampling.
- Keep architecture/loss unchanged for this cycle.

---

## 4) Key Root-Cause Discoveries During Implementation

1. DeepLive enhanced samples had `sample_id` carrying `_enhanced_...`, but manifest `strategy` could still be base strategy.
2. Strategy filters operating on raw `strategy` could silently mix enhanced/non-enhanced data.
3. Train-time validation did not emit detailed group reports by default.
4. Method attribution in combined loaders needed explicit method-id mapping to avoid collapsed reporting.
5. Some launches failed due to image/config mismatch and env/template constraints; this was operational, not modeling logic.
6. Startup time bottlenecks came largely from broad manifest scanning (especially VisoMaster discovery), not full frame pre-download.

---

## 5) Code Changes Implemented

### 5.1 DeepLive strategy resolution and filtering

File: `DeepfakeBench/training/dataset/deeplive_dataset.py`

- Added explicit `raw_strategy`, `effective_strategy`, `enhancement` semantics on samples.
- Added `resolve_effective_strategy(...)` to infer enhanced strategy from `sample_id` when needed.
- Updated discovery/filtering to use effective strategy logic.
- Added faster strategy-prefix discovery path to reduce startup scan cost.

### 5.2 Combined paired data source + preflight + weighted sampling

File: `DeepfakeBench/training/data/sources/combined_paired.py`

- Wired `combined_paired.df40.methods`.
- Wired DeepLive `include_strategies` / `exclude_strategies`.
- Added `split_seed` handling from config with canonical fallback.
- Added strict DeepLive strategy preflight checks:
  - `withenhanced`: required minimum enhanced counts.
  - `noenhanced`: required zero enhanced counts.
- Added `identity_resample_weighted` with family weights.
- Ensured transform backward compatibility for old 2-arg signatures while supporting meta-aware routing.

### 5.3 VisoMaster discovery performance/behavior adjustments

File: `DeepfakeBench/training/data/sources/visomaster.py`

- Restricted discovery prefix to VisoMaster samples.
- Added swap-model-aware prefiltering.
- Avoided unnecessary tier fetch when tiers are not requested.

### 5.4 Training config/reporting integration

Files:

- `DeepfakeBench/training/train_sweep.py`
- `DeepfakeBench/training/trainer/trainer.py`

Implemented:

- Strategy/family/source counts and preflight payload logging to W&B summary.
- Method attribution fixes through method-id mapping in validation.
- Group-aware detailed report generation:
  - `frames_report.csv` with `group_key`, `family_key`
  - `videos_report.csv` with `group_key`, `family_key`
  - `group_metrics.csv`
  - summary text with per-group section and best-worst gap

### 5.5 Sidecar validation path

Files:

- `DeepfakeBench/training/validate_custom_sources.py`
- `DeepfakeBench/training/run_r4_validation_sequential.py`
- `DeepfakeBench/training/data/validation_sources.py` (supporting pieces)

Implemented:

- On-demand checkpoint validation across:
  - DF40 target_source
  - DeepLive split=all
  - VisoMaster
  - external real source
  - WMA external fake source
- Detailed reports enabled and uploaded to GCS.

### 5.6 R4 experiment configs

Files under `DeepfakeBench/training/experiments/phase2_round4/`

- Updated FT1-FT6 with preflight expectations and new config keys.
- Added FT7/FT8 weighted variants.
- Added smoke configs for quick pipeline checks.
- Added/update README for R4 run matrix and gate expectations.

---

## 6) Testing Executed During This Session

1. `tests/test_phase4_family_pipeline.py` was added/extended and run on Vertex.
2. Unit-level checks covered:
   - DF40 methods filtering
   - DeepLive include/exclude behavior
   - family routing correctness
   - backward compatibility for 2-arg transforms
   - group mapping coverage
3. Basic compile/syntax sanity checks were run for modified files.
4. Smoke launch workflow was iterated to catch container/config path errors.

---

## 7) Operational Issues Encountered and Resolved

1. Vertex job env template failure: empty env var slot (`container_spec.env[2].value`) caused submission errors.
2. Config-in-image mismatch:
   - Job attempted to load `/workspace/experiments/...yaml` not present in built image.
   - Resolution: rebuild image after config changes or use GCS-based config loading path.
3. Shell ergonomics issue (`set -euo pipefail`) caused confusion in interactive usage.
4. Startup latency:
   - Initial jobs appeared “stuck” during discovery.
   - Not full frame download; mainly manifest scanning.
   - Discovery optimizations were patched.

---

## 8) The 4 Overnight R4 Runs Reviewed

Runs analyzed (training):

- FT1: `R4_FT1_base_noenhanced_0217-0028`
- FT2: `R4_FT2_family_light_noenhanced_0217-0028`
- FT5: `R4_FT5_family_light_withenhanced_0217-0028`
- FT7: `R4_FT7_family_light_withenhanced_weighted_0217-0028`

What was verified:

1. Preflight routing and enhanced inclusion behaved correctly:
   - FT1/FT2: noenhanced mode, enhanced counts absent.
   - FT5/FT7: withenhanced mode, enhanced counts present as expected.
2. Method-level tables had many concrete methods; no single aggregate collapse.
3. FT7 correctly used weighted sampling strategy.
4. Trajectories were not exact duplicates.

---

## 9) Sidecar Validation Results (Most Recent Source of Truth)

Sidecar runs completed and artifacts generated under:

- `gs://training-job-outputs/test_results/r4_sidecar_sofar_20260217_113917`

Per-run sidecar group metrics summary:

### FT1 (sidecar)

- external_real FPR: 4.17% (passes external-real gate)
- deeplive enhanced fake avg TPR: ~28.97% (poor)
- df40_fake TPR: ~83.77%
- Worst group: `deeplive_edge_cases_enhanced_fake` (acc ~27.13%)

### FT2 (sidecar)

- external_real FPR: 4.82% (passes external-real gate)
- deeplive enhanced fake avg TPR: ~31.29% (poor)
- df40_fake TPR: ~81.78%
- Worst group: `deeplive_edge_cases_enhanced_fake` (acc ~30.11%)

### FT5 (sidecar)

- external_real FPR: 3.35% (passes external-real gate)
- deeplive enhanced fake avg TPR: ~98.38% (strong)
- df40_fake TPR: ~83.08%
- Worst group: `df40_fake` (acc ~83.08%)

### FT7 (sidecar)

- external_real FPR: 4.38% (passes external-real gate)
- deeplive enhanced fake avg TPR: ~99.53% (strongest)
- df40_fake TPR: ~86.85% (better than FT5)
- Worst group: `df40_fake` (acc ~86.85%)

Practical ranking from available evidence:

1. FT7 (best overall balance)
2. FT5 (close second)
3. FT1/FT2 (not acceptable for enhanced fake objective)

---

## 10) Critical Caveat: WMA Gate Was Not Actually Measured

Hard gate still unresolved:

- `wma_failure_fake` detection >= 50%

Why unresolved:

- Sidecar logs showed:
  - `Discovered 0 frame paths from gs://effort-collected-data/wma_validation/enhanced_fake`
  - External fake loaded with `total_videos=0`
- Therefore WMA source contributed no samples in sidecar.

This means:

- We can assess external-real and group behavior strongly.
- We still cannot make a final gate-compliant model selection until WMA path/data is fixed and rerun.

---

## 11) Current Decision Status

Status right now:

1. External-real gate: effectively passed by FT1/FT2/FT5/FT7.
2. Enhanced DeepLive robustness: FT5/FT7 clearly superior to FT1/FT2.
3. WMA gate: unknown (not measured due to missing source data in sidecar).
4. Final production winner: not yet finalized under agreed criteria.

If forced to choose a provisional candidate now:

- FT7 is the best provisional candidate.

---

## 12) Immediate Next Steps (Priority Order)

### Step A: Fix WMA validation source path/data

1. Verify where the 1200 failure-set frames/videos are actually stored in GCS.
2. Update sidecar `--external_fake_bucket` and `--external_fake_prefix` to that real location.
3. Re-run sidecar on FT5 and FT7 checkpoints (minimum) with detailed reports.

### Step B: Re-evaluate gates with true WMA data

Use the agreed acceptance rule:

1. Keep only models passing both:
   - external real FPR <= 8%
   - WMA failure fake detection >= 50%
2. Among eligible models:
   - maximize WMA detection first
   - minimize external-real FPR second
   - holdout AUC as tie-breaker

### Step C: Optional but useful follow-up

1. If FT5/FT7 both pass WMA gate, compare:
   - DF40 fake sensitivity tradeoff
   - external real FPR margin
   - worst-group accuracy gap
2. Launch FT8 only if further DeepLive weighting is still desired.

---

## 13) Key Checkpoint References (Current Bests Used In Sidecar)

- FT1 checkpoint: `gs://training-job-outputs/phase2r4_experiments/9eqvw1i2/top_n_effort_20260217_step1500_auc0.9928_eer0.0150.pth`
- FT2 checkpoint: `gs://training-job-outputs/phase2r4_experiments/zzzpm53d/top_n_effort_20260217_step500_auc0.9945_eer0.0150.pth`
- FT5 checkpoint: `gs://training-job-outputs/phase2r4_experiments/tu0zeofr/top_n_effort_20260217_step500_auc0.9920_eer0.0122.pth`
- FT7 checkpoint: `gs://training-job-outputs/phase2r4_experiments/udgwsu7o/top_n_effort_20260217_step500_auc0.9935_eer0.0088.pth`

---

## 14) One-Line Executive Summary

This cycle successfully fixed data-contract/reporting blind spots and produced decision-useful evidence: FT7/FT5 materially solve the enhanced-fake failure seen in deployment while keeping external-real FPR low, but final selection is still blocked until WMA failure-set validation is run with a corrected data path.

---

## 15) Status Reset (Supersedes Sections 10-14) — February 17, 2026

### 15.1 Ground Truth Snapshot

1. Latest checkpoints currently available in GCS:
   - FT1 (`9eqvw1i2`) has later `top_n` at step 1500.
   - FT2 (`zzzpm53d`) has `top_n` step 500.
   - FT5 (`tu0zeofr`) has `top_n` step 500.
   - FT7 (`udgwsu7o`) has `top_n` step 500.
2. Sidecar outputs (full suite):
   - `gs://training-job-outputs/test_results/r4_sidecar_sofar_20260217_113917/`
3. WMA-only outputs (FT5/FT7):
   - `gs://training-job-outputs/test_results/r4_wma_only_20260217-163324/`
4. Canonical WMA flat per-image pass (`n=1202`, threshold=0.5):
   - FT5: `893/1202 = 74.29%`
   - FT7: `1062/1202 = 88.35%`
   - Artifact: `DeepfakeBench/training/debug/wma_flat_eval_ft5_ft7.json`

### 15.2 Canonical Scoreboard

| Run | External real FPR (gate <=8%) | WMA gate (flat, 1202, >=50%) | Enhanced DeepLive fake TPR | DF40 fake TPR | Overall sidecar AUC | Practical read |
|---|---:|---:|---:|---:|---:|---|
| FT1 | 4.17% | Not measured in flat pass | 28.97% | 83.77% | 0.9899 | Fails enhanced-fake objective |
| FT2 | 4.82% | Not measured in flat pass | 31.29% | 81.78% | 0.9883 | Fails enhanced-fake objective |
| FT5 | 3.35% | 74.29% | 98.38% | 83.08% | 0.9943 | Strong |
| FT7 | 4.38% | 88.35% | 99.53% | 86.85% | 0.9941 | Best balance |

### 15.3 Conclusion Now

1. FT1 and FT2 are out for production intent (enhanced fake TPR too low).
2. FT5 and FT7 both pass external-real and WMA gates; FT7 is stronger on robustness and fake sensitivity.
3. FT5 keeps a slightly larger external-real margin, but FT7 remains under the <=8% gate and wins on targeted fake detection.
4. Provisional candidate for deployment validation: **FT7**.
5. Rollback candidate: **FT5**.

### 15.4 Validation Semantics Update

WMA is now treated as canonical **flat per-image** evaluation for this failure set.  
Grouped-by-folder behavior remains available for backward compatibility but is no longer the primary gate metric for WMA.

### 15.5 Immediate Next Actions

1. Continue validation with explicit external fake grouping semantics:
   - `--external_fake_grouping per_image`
   - `--external_fake_deterministic`
2. Run target-domain suites (Zoom real/regular/enhanced/stress).
3. Launch R5 scratch matrix (`S1/S2/S3`) and compare against FT7 using the same gates plus worst-suite target-domain gap.
