# Thread: Train-time correlation penalty as an explicit shortcut-avoidance loss

> **Status as of 2026-05-06**: training half complete (R13_DEEPLIVE_CORR_PENALTY run `8jgyw1am` SUCCEEDED 2026-05-06 00:53 UTC, R13_VISO_CORR_PENALTY run `7u3zc5zt` SUCCEEDED 2026-05-06 01:01 UTC, both image 1.3.264/1.3.265 us-east1). **Promotion-contract scorecard not yet run** (image 1.3.267 rebuild in progress at writeup time, blocked on Cloud Build upload of source tarball). 5-axis baseline reference table established for 8 ckpts (E2B, P8A, PA fleet, PC fleet) — see [`PD`](../packets/PD.md) and `analysis/deeplive_viso_corr_eval_2026-05-06/`.

## The question

Does adding an explicit train-time penalty term `lambda * sum_axes |Pearson(score, axis)|` over a configured set of nuisance axes (sharpness, luma, face area) weaken the documented R13 shortcuts (image-quality / face-size / capture-mode) by a material margin, and does the weakening translate to a deployment-grade lift on the Teams promotion contract — or does it (like every other R13 anti-shortcut intervention before it) either fail to bite or shift the shortcut to an untargeted axis?

This thread is the home for the **explicit-form anti-shortcut loss** lever class. Sibling threads cover the implicit-form levers: [`face_size_label_leak`](face_size_label_leak.md) (face_scale_jitter as augmentation), [`anti_shortcut_bundle_decomposition`](anti_shortcut_bundle_decomposition.md) (the bundle-stacking discipline that motivated single-lever testing), [`processing_signature_shortcut`](processing_signature_shortcut.md) (the underlying shortcut taxonomy this lever attacks).

## Initial belief

When this lever was scoped on 2026-05-05, the working belief was: **R13 detectors learn capture-condition shortcuts that the model uses as fake predictors** (memory `project_image_quality_shortcut.md`, score correlates negatively with Laplacian variance / luma / skin_frac across most suites; eval lockbox `cam_test_s33` 14-43× less sharp than training; threshold relaxation 2%→10% real_FPR unlocks 4-77% recall depending on suite). The 2026-05-02 score-distribution audit (memory `project_score_distribution_audit_2026_05_02.md`, 11 CPU diagnostics, 70 figures, 33 CSVs) established the shortcut empirically; the P22 augmentation curriculum (memory `project_p22_succeeded_2026-05-02.md` then revised to `project_p22_cpu_followups_reframe_2026-05-02.md`) showed an augmentation-only intervention gave a real but bounded weakening with the dominant shortcut shifting between checkpoints.

The corr-penalty hypothesis: **if the shortcut is "model uses sharpness/luma/face_area as a fake predictor", an explicit batch-level decorrelation penalty on the score against those axes should weaken it more directly than augmenting the input distribution**. The decorrelation is a regularizer in the same loose sense that label smoothing or weight decay are; it doesn't change the input distribution, only what gradients can shape the score.

## What changed our mind

- **2026-05-05 ~12:26 UTC** — Frozen-head Phase 1 prototype (`analysis/corr_penalty_prototype_2026-05-05/run.log`, `lambda_sweep.csv`): 30-epoch logistic regression on cached P8A frozen features (n=8698 train + 1037 identity-disjoint test, 4 axes: sharpness, luma, face_area, is_webcam). Results across `lambda ∈ {0, 0.1, 1, 10, 100}`:

  | λ | test AUC | recall@FPR=0.10 | \|r_sharp\| | \|r_luma\| | \|r_face\| | \|r_webcam\| |
  |---|---:|---:|---:|---:|---:|---:|
  | 0   | 0.889 | 0.355 | 0.156 | 0.233 | 0.032 | 0.081 |
  | 0.1 | 0.885 | 0.350 | 0.129 | 0.246 | 0.019 | 0.085 |
  | **1** | **0.871** | **0.491** | **0.053** | **0.071** | 0.157 | 0.205 |
  | 10  | 0.765 | 0.337 | 0.041 | 0.279 | 0.055 | 0.081 |
  | 100 | 0.499 | 0.160 | 0.127 | 0.350 | 0.123 | 0.146 |

  At λ=1: targeted axes |r_sharp| 0.156→0.053 (-66%), |r_luma| 0.233→0.071 (-69%); recall@FPR=0.10 INCREASED 0.355→0.491 (+13.6pp). λ=10 oscillates; λ=100 collapses to chance. Untargeted axes (`face_area`, `is_webcam`) **rebound** at λ=1 (0.032→0.157, 0.081→0.205) — the canonical "shortcut shifting" symptom on a frozen head. The untargeted-axis shifting is what motivated adding `face_area_fraction` as a third axis in the encoder fine-tune.

- **2026-05-05 evening** — Loss class committed (`loss/correlation_penalty.py`, commit `b1a0173`). API:

  ```python
  CorrelationPenalty(axes=['sharpness_laplacian', 'luma_mean', 'face_area_fraction'], lambda_=1.0)
  # __call__(score: [B] tensor, axis_values: dict[str, [B] tensor]) -> (loss, per_axis_r)
  # loss = lambda_ * sum_axis |Pearson_batch(score, axis_values[axis])|
  ```

  21 unit tests (`tests/test_correlation_penalty.py`): differentiable Pearson, 5-D video-batch handling (`compute_pixel_axes` accepts both 4-D `[B,3,H,W]` and 5-D `[B,T,3,H,W]`, flattens to per-frame), zero-lambda short-circuit, gradient-flow check.

- **2026-05-05 evening** — Three integration bugs surfaced and fixed during smoke runs (1.3.260 → 1.3.265):
  - **Smoke #1** (commit `29d0342`, image 1.3.263): `correlation_penalty` block was being SILENTLY DISABLED at training time because `wandb.init(config=single_cfg)` flattens nested dicts and the trainer reads `self.config.get('correlation_penalty')` which returns `None`. Fixed by adding `correlation_penalty` to `train_sweep.py`'s re-apply allowlist (lines ~176-282). This is the same wandb-flattening recurrence pattern documented in [`wandb_flattening`](wandb_flattening.md) — the 5th surface bug of that class. Regression test added (`tests/test_train_sweep_reapply_allowlist.py:32-46`).
  - **Smoke #2** (commit `0cd5df5`, image 1.3.264): `compute_pixel_axes` raised `ValueError: expected [B,3,H,W]; got (4, 8, 3, 224, 224)`. The `combined_paired` collate emits a 5-D video tensor `[B, T, 3, H, W]`. Fixed by reshaping to `[B*T, 3, H, W]` and broadcasting per-video axes (face_area_fraction is per-video) to per-frame via `repeat_interleave(t_repeat)` when `score_n % raw.shape[0] == 0`.
  - **Smoke #3 / VISO image** (commit `9dfd16f`, image 1.3.265): visomaster sister yaml added (`R13_VISO_CORR_PENALTY.yaml`).

- **2026-05-05 → 2026-05-06 overnight** — Both encoder fine-tune jobs SUCCEEDED:
  - **DEEPLIVE arm** (`R13_DEEPLIVE_CORR_PENALTY`, run `8jgyw1am` "rogue-senate-951"): started 22:19:22 UTC, ended 00:53:09 UTC (~2h34m). Image 1.3.264, us-east1, image-currency check verified `correlation_penalty: enabled=True lambda=1.0 axes=['sharpness_laplacian', 'luma_mean', 'face_area_fraction']` at training time. 11 ckpts saved at `gs://training-job-outputs/best_checkpoints/8jgyw1am/`. Best top_n step4800 AUC 0.9970 EER 0.0113. Note: ran past nominal `total_training_steps: 2500` (highest top_n at step4800), suggesting `nEpochs: 8` took precedence.
  - **VISO arm** (`R13_VISO_CORR_PENALTY`, run `7u3zc5zt` "elegant-emperor-952"): started 23:34:57 UTC, ended 01:01:20 UTC (~1h26m, faster). Image 1.3.265, us-east1. 7 ckpts saved at `gs://training-job-outputs/best_checkpoints/7u3zc5zt/`. Best top_n step600 AUC 0.9934 EER 0.0273. Stopped earlier than deeplive (max periodic step 2000 vs deeplive's step4800) — root cause not isolated.

  Both arms FT from `E2B_TOP_N_STEP3200` (`gs://training-job-outputs/best_checkpoints/rmat8lwx/top_n_effort_20260503_step3200_auc0.9863_eer0.0310.pth`); seed 5051 (deeplive) / 5052 (viso); single-lever discipline (`anchor_aware: false`, `face_scale_jitter: false`, `arcface_head: false`); 3-axis penalty at λ=1.0; visomaster sources DISABLED on deeplive (per user 2026-05-05 "we don't need any visomaster data") and ENABLED on viso (the controlled-comparison axis between the two arms).

- **2026-05-06 morning** — CPU 5-axis audit on 8 BASELINE ckpts (E2B, P8A, PA × 3, PC × 3) ran on existing `analysis/pa_pc_eval_2026-05-05/raw_reports/` (`analysis/deeplive_viso_corr_eval_2026-05-06/run_audit.py`, ~5min CPU). This is the F2/F3 reference table the new corr-penalty ckpts will be compared against once the GPU scorecard finishes writing their per-frame reports. **Mean |Pearson r| (score vs penalty axis), averaged over suites**:

  | ckpt | suite_kind | face_area | luma_mean | sharpness |
  |---|---|---:|---:|---:|
  | **e2b** (FT base for corr-penalty) | real | 0.317 | 0.094 | 0.369 |
  | e2b | fake | 0.322 | 0.175 | 0.162 |
  | p8a | real | 0.117 | 0.312 | 0.188 |
  | p8a | fake | 0.412 | 0.148 | 0.188 |
  | pa_top_n_5600 | real | 0.222 (-30% vs E2B) | 0.106 | 0.335 (-9%) |
  | pc_top_n_5400 | real | 0.351 (+11%!) | 0.270 (+187%!) | 0.181 (-51%) |
  | pc_periodic_5000 | fake | 0.283 (-12%) | 0.066 (-62%) | 0.086 (-47%) |

  **Reading**: PA shows modest, axis-uneven weakening; PC demonstrates the **shortcut-shifting failure mode** (sharpness weakens 51% on real but luma amplifies 187%, untargeted-axis +50% breaks F3 of the close criterion). Neither hit "≥30% on ≥2 axes AND no axis +50%". The corr-penalty ckpts will be measured against this baseline once their scorecard reports land.

## Current stance (2026-05-06 morning)

**Training is verified clean** — the wiring works (loss enabled, gradients flow per `tests/test_correlation_penalty.py:test_gradient_flows_through_score`, λ=1.0 confirmed at training time on both arms). **Deployment-grade verdict is unknown** until the GPU scorecard runs the new ckpts on the Teams promotion contract substrate. The CPU baseline is the comparison frame: the corr-penalty's 4/4 close criterion is

- **F1**: lockbox recall ≥ 90% at FPR ≤ 10% (excluding `is_no_face` / extreme low-IQ frames per `eval_substrate_data_hygiene.md`)
- **F2**: shortcut weakening ≥ 30% on ≥ 2 of 5 axes (sharpness, luma, face_area, is_webcam, is_screen) vs E2B baseline
- **F3**: no shortcut shifting — no untargeted axis amplifies > 50%
- **F4**: HDTF cross-substrate FPR ≤ 5% on `proper_real_teams_dev` / `proper_real_clean_dev` (per `viso_bucket_gap.md` 2026-05-05 morning update — HDTF is the substrate where P8A gives 93%+ viso recall, the cross-substrate validation gate that PA failed)

The frozen-head prototype suggested λ=1 is in the stable regime AND that untargeted axes shift; both predictions will be testable against the GPU scorecard's per-frame reports. **The frozen-head test population was too small/synthetic to predict deployment behavior** — recall@FPR=0.10 lifted from 0.355 to 0.491 on a 1037-frame disk-cached test split, which is informative for the loss-class direction but cannot stand in for the full lockbox evaluation. The encoder fine-tune is the real test.

**Risks to be aware of when reading scorecard results**:
1. The frozen-head prototype showed shortcut-shifting on `face_area` and `is_webcam`. The deeplive yaml adds `face_area_fraction` as a third penalty axis (closes one of the two shifting paths); `is_webcam` is NOT in the penalty axes (Teams does not surface capture mode at inference per `feedback_per_mode_tau_not_deployable.md`, so it can't be a deployable target). If the scorecard shows webcam-substrate FPR amplification, that's the predicted shifting failure.
2. The PA/PC pattern (PA's F4 v2 lift didn't generalize to HDTF) means a v2-only scorecard read is inadequate. **Cross-substrate validation MUST be part of the close criterion** per the lesson from `project_pa_does_not_generalize_to_hdtf_2026-05-05.md`. The HDTF F4 gate above is the operationalization.
3. Both arms FT from E2B which is itself a CLIP-scratch-from-scratch (not the FT chain through R12G that produced P8A). The PA/PC walkback established that "FT-from-CLIP-scratch family fails on HDTF" — this is a load-bearing risk for the corr-penalty arms inheriting the same trajectory.

## Packet timeline

- [P22](../packets/P22.md) — augmentation-curriculum predecessor; established that score-axis correlation can be weakened via input-distribution intervention but the weakening is bounded (memory `project_p22_cpu_followups_reframe_2026-05-02.md`).
- [PA](../packets/PA.md) / [PC](../packets/PC.md) — data-axis lever (PA worked on v2 F4, didn't generalize to HDTF; PC codec aug HURT viso 35-50pp). Established the PA/PC walkback that the close criterion must include cross-substrate validation.
- [PD](../packets/PD.md) — the corr-penalty packet. Two arms (deeplive ship + viso ship) both FT-from-E2B + λ=1.0 + 3-axis penalty; differ only on whether visomaster sources are enabled. Training complete; scorecard pending.

## Evidence locations

- `loss/correlation_penalty.py` — the loss module (88 lines including 60 lines of doc).
- `tests/test_correlation_penalty.py` — 21 unit tests including `test_accepts_5d_video_batch`.
- `tests/test_train_sweep_reapply_allowlist.py:32-46` — TRAINER_NESTED_KEYS list with `correlation_penalty` (the wandb-flattening regression test).
- `experiments/phase2_round13/R13_DEEPLIVE_CORR_PENALTY.yaml` — deeplive arm yaml (seed 5051, no visomaster, 3-axis penalty).
- `experiments/phase2_round13/R13_VISO_CORR_PENALTY.yaml` — viso arm yaml (seed 5052, full visomaster sources, 3-axis penalty).
- `experiments/phase2_round13/R13_DEEPLIVE_CORR_PENALTY_SMOKE.yaml` — 100-step smoke variant (DF40 disabled).
- `analysis/corr_penalty_prototype_2026-05-05/{lambda_sweep.csv,loss_curves.csv,run.log,scripts/}` — frozen-head Phase 1 prototype.
- `analysis/deeplive_viso_corr_eval_2026-05-06/{run_audit.py,coverage.csv,per_axis_recall_fpr.csv,correlations.csv,abs_pearson_summary.csv,tau_calibration.csv}` — 8-baseline 5-axis reference table (Phase 1 audit; Phase 2 adds new ckpts when scorecard reports land).
- `arena/checkpoint_maps/teams_target_domain.deeplive_viso_corr_2026-05-06.yaml` — 8-entry checkpoint map (P8A, E2B, 3 deeplive_corr, 3 viso_corr) for the pending GPU scorecard.
- `analysis/deeplive_face_geometry_2026-05-05/face_area.parquet` — 65,202 rows / 99.89% face-detection coverage; the per-sample `face_area_fraction` lookup the corr-penalty axis reads via the dataloader.
- Memory: `project_image_quality_shortcut.md`, `project_score_distribution_audit_2026_05_02.md`, `project_p22_cpu_followups_reframe_2026-05-02.md`.
- Commits: `b1a0173` (loss class + deeplive yaml), `ccb03f7` (face_area parquet + 3rd axis), `b7e7477` (force-add parquet over `*.parquet` gitignore), `29d0342` (wandb-flattening allowlist fix), `0cd5df5` (5-D batch handling), `a12e6e5` (viso yaml).
- Wandb: https://wandb.ai/dtect-vision/phase2r13-experiments/runs/8jgyw1am (deeplive), https://wandb.ai/dtect-vision/phase2r13-experiments/runs/7u3zc5zt (viso).

## Open loops

### Open loop: corr-penalty-deployment-grade-verdict-pending
status: in-progress
severity: high
first_seen: 2026-05-06
last_verified: 2026-05-06
close_criterion: a promotion-contract scorecard run on the 8-entry ckpt map `arena/checkpoint_maps/teams_target_domain.deeplive_viso_corr_2026-05-06.yaml` against `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml` lands and is read against the 4/4 close criterion (F1 lockbox recall ≥ 90% at FPR ≤ 10%; F2 shortcut weakening ≥ 30% on ≥ 2 of 5 axes vs E2B baseline; F3 no untargeted axis +50%; F4 HDTF cross-substrate FPR ≤ 5%). The Phase-2 5-axis audit (`analysis/deeplive_viso_corr_eval_2026-05-06/run_audit.py` Phase 2 block, currently commented out — uncomment after `gcloud storage cp -r <scorecard_output>/raw_reports analysis/deeplive_viso_corr_eval_2026-05-06/raw_reports/` lands) is the comparison frame. Either (a) verdict is positive on all 4 criteria → corr-penalty is the first R13 anti-shortcut intervention to clear deployment grade and gets a "Confirmed good" entry in the README; or (b) any criterion fails → packet is muddled / failed and the lever class joins anchor_aware, GRL, and face_scale_jitter on the "structurally working but not deployment-grade" list.

As of 2026-05-06 09:30 UTC the image 1.3.267 rebuild is in flight (Cloud Build source upload at ~5GB / 5GB), GPU scorecard launch will follow. Estimated total time-to-verdict: ~2-3h after image lands (2.5-3h scorecard run + ~30min CPU audit Phase 2).

### Open loop: corr-penalty-frozen-head-shifting-axes-not-targeted
status: open
severity: medium
first_seen: 2026-05-06
last_verified: 2026-05-06
close_criterion: either (a) the encoder fine-tune scorecard's 5-axis audit shows |Pearson r| on `is_webcam` AND `is_screen` did NOT amplify > 50% relative to E2B baseline (predicted shifting did not occur on the encoder; resolves favorably) OR (b) the audit shows amplification > 50% on at least one of those axes (the frozen-head prediction holds) AND a successor packet design includes `is_webcam` or `is_screen` in the penalty axes, with a path to deploy them at inference (currently blocked because Teams does not surface capture mode — would need an upstream classifier or a different inference-time signal that proxies for capture mode). The frozen-head prototype showed `face_area` 0.032→0.157 and `is_webcam` 0.081→0.205 as λ ramped 0→1, so the encoder is the place this gets adjudicated. Resolution flips this loop's status to `resolved` favorably or `superseded` (lever-class cap discovered).

## Cross-thread refs

- [`processing_signature_shortcut`](processing_signature_shortcut.md) — the underlying shortcut taxonomy this lever attacks; 2026-05-06 update appended noting the corr-penalty arms train.
- [`face_size_label_leak`](face_size_label_leak.md) — the face_area_fraction axis is the explicit decorrelation companion to `face_scale_jitter`'s implicit augmentation lever; both attack the same underlying leak from different sides.
- [`anti_shortcut_bundle_decomposition`](anti_shortcut_bundle_decomposition.md) — the corr-penalty arms run with `anchor_aware: false` and `face_scale_jitter: false` in single-lever discipline. If the scorecard verdict is positive, this is the first single-lever encoder-side anti-shortcut intervention to clear deployment grade on R13.
- [`viso_bucket_gap`](viso_bucket_gap.md) — the viso arm includes visomaster_enhanced + visomaster_teams_enhanced sources at fw=4.0 (mirroring PA), so it tests both data-availability AND corr-penalty in the same packet; deeplive arm tests corr-penalty alone with no visomaster data.
- [`wandb_yaml_propagation_bugs`](wandb_yaml_propagation_bugs.md) — Smoke #1's wandb-flattening recurrence is the 5th surface bug of that class; mitigated via the regression test that now lives in `tests/test_train_sweep_reapply_allowlist.py:32-46`.
- [`quality_enhancement_strategy_misrouting`](quality_enhancement_strategy_misrouting.md) — both corr-penalty arms train POST-fix (image 1.3.264 / 1.3.265 derive from 1.3.259, the post-fix image), so this is the first pair of R13 packets NOT contaminated by the routing bug. Cross-packet comparisons to pre-fix R13 packets carry the contamination caveat from the canonical thread.
- [`promotion_contract_evolution`](promotion_contract_evolution.md) — verdict is read against the same scorecard infrastructure that produced the PA/PC verdict. The recall-floor v3 fix's deployment status remains in `contract-policy-bug-fix-not-committed` open loop.
