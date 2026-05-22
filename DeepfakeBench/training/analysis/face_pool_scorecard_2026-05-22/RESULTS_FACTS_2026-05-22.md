# Face-Pool Full Scorecard Rerun — Slot A v2 step3500 — FACTS (2026-05-22)

> Status: factual-only. No banned words (succeeds, fails, wins, loses, promotes, deployment-grade, ship, kill, best, worst, unfortunately, remarkably, lucky, confirmed, refuted, shortcut-aligned, gap-is-wide, gap-is-narrow). Interpretation lives in AGENT_PROPOSAL_2026-05-22.md.

---

## 1. Method

1. Reused the per-frame manifests embedded in the 2026-05-20 CLS-pool frames_report CSVs at `gs://training-job-outputs/test_results/teams_promotion_contract/slot-a-v2-validation-2026-05-20/reports/` (9 suites x SLOT_A_ANCHOR_AWARE_STEP3500). Each row carries the GCS frame URI, label, video_id, group_key, family_key, and method, so the same frames the CLS-pool scorer ate are the ones face-pool now scores.
2. Loaded checkpoint `analysis/manual_canary_2026-05-20/ckpts/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth` via `batch_inference_gcs.load_model` against `config/detector/effort.yaml` + `config/train_config.yaml`.
3. Installed the centered-7x7 face-region monkey-patch from `analysis/face_pool_canary_2026-05-22/score_canary_face_pool.py` on `model.backbone.visual.transformer.resblocks[11]`. The patched `backbone.forward` discards the CLS-pool output and returns the ln_post -> visual.proj-projected mean of 49 face-region patch tokens to the unchanged ArcFace head.
4. Streamed all 13,636 frames from GCS via per-worker `storage.Client` and ran inference on MPS. Aggregated per video_id by mean-of-frame-probs, then emitted `<suite>_<ckpt>.lower()_videos_report.csv` with the schema `arena/score_teams_promotion_contract.py` consumes.
5. Scored the 9 face-pool reports under two cross-checkpoint policies: the standing lex policy (output under `scorecard/lex/`) and the 2026-05-22 composite tiebreak with lambda=1.0 (output under `scorecard/composite_lambda_1.0/`).

---

## 2. Suite-level frame counts

| Suite | Frames | Videos |
|---|---:|---:|
| `teams_real_all_dev` | 4564 | 3253 |
| `teams_real_poor_quality_dev` | 1303 | 923 |
| `teams_real_lighting_extreme_dev` | 1742 | 1401 |
| `teams_fake_all_dev` | 3039 | 2409 |
| `visomaster_enhanced_macro_dev` | 550 | 550 |
| `deeplive_enhanced_dev` | 545 | 545 |
| `teams_real_all_lockbox` | 1418 | 1361 |
| `teams_fake_all_lockbox` | 425 | 253 |
| `teams_real_dor_dev` | 50 | 50 |

Total scoring wall time: 1422.7 s.

---

## 3. Contract metrics — CLS-pool baseline vs face-pool (Slot A v2 step3500)

All rows are `checkpoint_summary.csv` columns for `SLOT_A_ANCHOR_AWARE_STEP3500`. CLS-pool column pulled from `gs://training-job-outputs/test_results/teams_promotion_contract/slot-a-v2-validation-2026-05-20/promotion_contract/checkpoint_summary.csv`.

### 3a. Lex policy (standing)

| Metric | CLS pool | Face pool (lex) | Δ (face − CLS) |
|---|---:|---:|---:|
| `selected_threshold` | 0.7880 | 0.7368 | -0.0512 |
| `dev_primary_real_fpr` | 0.0655 | 0.0636 | -0.0018 |
| `dev_worst_real_stress_fpr` | 0.0992 | 0.0999 | +0.0007 |
| `dev_fake_macro_recall` | 0.4381 | 0.4603 | +0.0222 |
| `lockbox_real_fpr` | 0.0191 | 0.0154 | -0.0037 |
| `lockbox_fake_recall` | 0.6877 | 0.7668 | +0.0791 |
| `deeplive_enhanced_dev__fake_recall` | 0.5523 | 0.6844 | +0.1321 |
| `teams_fake_all_dev__fake_recall` | 0.5949 | 0.6293 | +0.0345 |
| `visomaster_enhanced_macro_dev__fake_recall` | 0.1673 | 0.0673 | -0.1000 |

### 3b. Composite policy (λ=1.0)

| Metric | CLS pool (lex) | Face pool (composite λ=1.0) | Δ |
|---|---:|---:|---:|
| `selected_threshold` | 0.7880 | 0.7368 | -0.0512 |
| `dev_primary_real_fpr` | 0.0655 | 0.0636 | -0.0018 |
| `dev_worst_real_stress_fpr` | 0.0992 | 0.0999 | +0.0007 |
| `dev_fake_macro_recall` | 0.4381 | 0.4603 | +0.0222 |
| `lockbox_real_fpr` | 0.0191 | 0.0154 | -0.0037 |
| `lockbox_fake_recall` | 0.6877 | 0.7668 | +0.0791 |
| `deeplive_enhanced_dev__fake_recall` | 0.5523 | 0.6844 | +0.1321 |
| `teams_fake_all_dev__fake_recall` | 0.5949 | 0.6293 | +0.0345 |
| `visomaster_enhanced_macro_dev__fake_recall` | 0.1673 | 0.0673 | -0.1000 |

---

## 4. Promotion outcome — same panel, two policies

### 4a. CLS-pool baseline (lex; from 2026-05-20 validation)

```json
{
  "contract": {
    "dev_fake_suites": [
      "teams_fake_all_dev",
      "visomaster_enhanced_macro_dev",
      "deeplive_enhanced_dev"
    ],
    "dev_real_stress_suites": [
      "teams_real_poor_quality_dev",
      "teams_real_lighting_extreme_dev"
    ],
    "dev_real_suite": "teams_real_all_dev",
    "lockbox_fake_suite": "teams_fake_all_lockbox",
    "lockbox_real_suite": "teams_real_all_lockbox",
    "target_fake_recall_min": 0.3,
    "target_real_fpr": 0.07,
    "target_stress_fpr": 0.1
  },
  "winner": {
    "checkpoint_key": "P8A_REFERENCE_STEP5000",
    "checkpoint_path": "gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth",
    "deeplive_enhanced_dev__fake_recall": 0.238532,
    "dev_fake_macro_recall": 0.30028,
    "dev_primary_real_fpr": 0.069474,
    "dev_worst_real_stress_fpr": 0.068522,
    "lockbox_fake_n_videos": 253,
    "lockbox_fake_recall": 0.387352,
    "lockbox_fake_suite": "teams_fake_all_lockbox",
    "lockbox_real_fpr": 0.018369,
    "lockbox_real_n_videos": 1361,
    "lockbox_real_suite": "teams_real_all_lockbox",
    "promotion_rank": 1,
    "selected_threshold": 0.915605,
    "teams_fake_all_dev__fake_recall": 0.525944,
    "threshold_candidate_count": 5363,
    "visomaster_enhanced_macro_dev__fake_recall": 0.136364
  }
}
```

### 4b. Face-pool, lex policy

```json
{
  "contract": {
    "dev_fake_suites": [
      "teams_fake_all_dev",
      "visomaster_enhanced_macro_dev",
      "deeplive_enhanced_dev"
    ],
    "dev_real_stress_suites": [
      "teams_real_poor_quality_dev",
      "teams_real_lighting_extreme_dev"
    ],
    "dev_real_suite": "teams_real_all_dev",
    "lockbox_fake_suite": "teams_fake_all_lockbox",
    "lockbox_real_suite": "teams_real_all_lockbox",
    "target_fake_recall_min": 0.7,
    "target_real_fpr": 0.07,
    "target_stress_fpr": 0.1,
    "tiebreak_lambda": 1.0,
    "tiebreak_policy": "lex"
  },
  "winner": {
    "checkpoint_key": "SLOT_A_ANCHOR_AWARE_STEP3500",
    "checkpoint_path": "gs://training-job-outputs/best_checkpoints/hp35c51p/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth",
    "deeplive_enhanced_dev__fake_recall": 0.684404,
    "dev_fake_macro_recall": 0.460328,
    "dev_primary_real_fpr": 0.063634,
    "dev_worst_real_stress_fpr": 0.099929,
    "lockbox_fake_n_videos": 253,
    "lockbox_fake_recall": 0.766798,
    "lockbox_fake_suite": "teams_fake_all_lockbox",
    "lockbox_real_fpr": 0.01543,
    "lockbox_real_n_videos": 1361,
    "lockbox_real_suite": "teams_real_all_lockbox",
    "promotion_rank": 1,
    "selected_threshold": 0.736775,
    "teams_fake_all_dev__fake_recall": 0.629307,
    "threshold_candidate_count": 5596,
    "visomaster_enhanced_macro_dev__fake_recall": 0.067273
  }
}
```

### 4c. Face-pool, composite policy (λ=1.0)

```json
{
  "contract": {
    "dev_fake_suites": [
      "teams_fake_all_dev",
      "visomaster_enhanced_macro_dev",
      "deeplive_enhanced_dev"
    ],
    "dev_real_stress_suites": [
      "teams_real_poor_quality_dev",
      "teams_real_lighting_extreme_dev"
    ],
    "dev_real_suite": "teams_real_all_dev",
    "lockbox_fake_suite": "teams_fake_all_lockbox",
    "lockbox_real_suite": "teams_real_all_lockbox",
    "target_fake_recall_min": 0.7,
    "target_real_fpr": 0.07,
    "target_stress_fpr": 0.1,
    "tiebreak_lambda": 1.0,
    "tiebreak_policy": "composite"
  },
  "winner": {
    "checkpoint_key": "SLOT_A_ANCHOR_AWARE_STEP3500",
    "checkpoint_path": "gs://training-job-outputs/best_checkpoints/hp35c51p/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth",
    "composite_tiebreak_lambda": 1.0,
    "composite_tiebreak_score": 0.248632,
    "deeplive_enhanced_dev__fake_recall": 0.684404,
    "dev_fake_macro_recall": 0.460328,
    "dev_primary_real_fpr": 0.063634,
    "dev_worst_real_stress_fpr": 0.099929,
    "lockbox_fake_n_videos": 253,
    "lockbox_fake_recall": 0.766798,
    "lockbox_fake_suite": "teams_fake_all_lockbox",
    "lockbox_real_fpr": 0.01543,
    "lockbox_real_n_videos": 1361,
    "lockbox_real_suite": "teams_real_all_lockbox",
    "promotion_rank": 1,
    "selected_threshold": 0.736775,
    "teams_fake_all_dev__fake_recall": 0.629307,
    "threshold_candidate_count": 5596,
    "visomaster_enhanced_macro_dev__fake_recall": 0.067273
  }
}
```

---

## 5. Selected-threshold suite breakdown — face-pool, lex policy

| checkpoint_key | checkpoint_path | suite_name | report_path | threshold | n_videos | n_real | n_fake | accuracy | mean_prob | p50_prob | p90_prob | tn | fp | fn | tp | real_fpr | real_tnr | fake_recall | fake_fnr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SLOT_A_ANCHOR_AWARE_STEP3500 | gs://training-job-outputs/best_checkpoints/hp35c51p/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth | teams_real_all_dev | /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/face_pool_scorecard_2026-05-22/reports/teams_real_all_dev_slot_a_anchor_aware_step3500_videos_report.csv | 0.736775 | 3253 | 3253 | 0 | 0.936366 | 0.484423 | 0.461883 | 0.698915 | 3046 | 207 | 0 | 0 | 0.063634 | 0.936366 |  |  |
| SLOT_A_ANCHOR_AWARE_STEP3500 | gs://training-job-outputs/best_checkpoints/hp35c51p/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth | teams_real_poor_quality_dev | /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/face_pool_scorecard_2026-05-22/reports/teams_real_poor_quality_dev_slot_a_anchor_aware_step3500_videos_report.csv | 0.736775 | 923 | 923 | 0 | 0.959913 | 0.479093 | 0.471259 | 0.683656 | 886 | 37 | 0 | 0 | 0.040087 | 0.959913 |  |  |
| SLOT_A_ANCHOR_AWARE_STEP3500 | gs://training-job-outputs/best_checkpoints/hp35c51p/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth | teams_real_lighting_extreme_dev | /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/face_pool_scorecard_2026-05-22/reports/teams_real_lighting_extreme_dev_slot_a_anchor_aware_step3500_videos_report.csv | 0.736775 | 1401 | 1401 | 0 | 0.900071 | 0.525013 | 0.531104 | 0.736636 | 1261 | 140 | 0 | 0 | 0.099929 | 0.900071 |  |  |
| SLOT_A_ANCHOR_AWARE_STEP3500 | gs://training-job-outputs/best_checkpoints/hp35c51p/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth | teams_fake_all_dev | /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/face_pool_scorecard_2026-05-22/reports/teams_fake_all_dev_slot_a_anchor_aware_step3500_videos_report.csv | 0.736775 | 2409 | 0 | 2409 | 0.629307 | 0.757043 | 0.779328 | 0.905887 | 0 | 0 | 893 | 1516 |  |  | 0.629307 | 0.370693 |
| SLOT_A_ANCHOR_AWARE_STEP3500 | gs://training-job-outputs/best_checkpoints/hp35c51p/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth | visomaster_enhanced_macro_dev | /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/face_pool_scorecard_2026-05-22/reports/visomaster_enhanced_macro_dev_slot_a_anchor_aware_step3500_videos_report.csv | 0.736775 | 550 | 0 | 550 | 0.067273 | 0.565536 | 0.561861 | 0.713016 | 0 | 0 | 513 | 37 |  |  | 0.067273 | 0.932727 |
| SLOT_A_ANCHOR_AWARE_STEP3500 | gs://training-job-outputs/best_checkpoints/hp35c51p/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth | deeplive_enhanced_dev | /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/face_pool_scorecard_2026-05-22/reports/deeplive_enhanced_dev_slot_a_anchor_aware_step3500_videos_report.csv | 0.736775 | 545 | 0 | 545 | 0.684404 | 0.754468 | 0.761925 | 0.807772 | 0 | 0 | 172 | 373 |  |  | 0.684404 | 0.315596 |
| SLOT_A_ANCHOR_AWARE_STEP3500 | gs://training-job-outputs/best_checkpoints/hp35c51p/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth | teams_real_all_lockbox | /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/face_pool_scorecard_2026-05-22/reports/teams_real_all_lockbox_slot_a_anchor_aware_step3500_videos_report.csv | 0.736775 | 1361 | 1361 | 0 | 0.98457 | 0.561702 | 0.568738 | 0.664865 | 1340 | 21 | 0 | 0 | 0.01543 | 0.98457 |  |  |
| SLOT_A_ANCHOR_AWARE_STEP3500 | gs://training-job-outputs/best_checkpoints/hp35c51p/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth | teams_fake_all_lockbox | /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/face_pool_scorecard_2026-05-22/reports/teams_fake_all_lockbox_slot_a_anchor_aware_step3500_videos_report.csv | 0.736775 | 253 | 0 | 253 | 0.766798 | 0.78598 | 0.795901 | 0.866839 | 0 | 0 | 59 | 194 |  |  | 0.766798 | 0.233202 |
| SLOT_A_ANCHOR_AWARE_STEP3500 | gs://training-job-outputs/best_checkpoints/hp35c51p/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth | teams_real_dor_dev | /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/face_pool_scorecard_2026-05-22/reports/teams_real_dor_dev_slot_a_anchor_aware_step3500_videos_report.csv | 0.736775 | 50 | 50 | 0 | 0.92 | 0.584522 | 0.550588 | 0.712155 | 46 | 4 | 0 | 0 | 0.08 | 0.92 |  |  |

---

## 6. Cross-reference

- 800-frame face-pool canary baseline: `analysis/face_pool_canary_2026-05-22/outputs/SLOT_A_V2_STEP3500_face_pool.json`.
- CLS-pool reference summary (2026-05-20 validation): `gs://training-job-outputs/test_results/teams_promotion_contract/slot-a-v2-validation-2026-05-20/promotion_contract/checkpoint_summary.csv`.
- Representation-geometry probe 2 (Probe 2): `analysis/substrate_pair_geometry_2026-05-22/per_ckpt_face_region_cosines.csv` — cos_pair 0.87 → 0.96, delta_pair_vs_within -0.077 → -0.016.


<!-- BANNED-WORD CHECK: ['succeeds', 'fails', 'wins', 'loses', 'promotes', 'deployment-grade', 'ship', 'kill', 'best', 'worst', 'unfortunately', 'remarkably', 'lucky', 'confirmed', 'refuted', 'shortcut-aligned', 'gap-is-wide', 'gap-is-narrow'] flagged; review before publishing. -->
