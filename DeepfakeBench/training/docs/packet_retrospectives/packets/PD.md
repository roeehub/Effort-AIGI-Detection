# Packet PD · Train-time correlation-penalty loss as explicit shortcut-avoidance lever (deeplive ship + viso ship arms)

> **Verdict pending**: training half complete (both arms SUCCEEDED 2026-05-06 overnight, ckpts saved). Promotion-contract scorecard NOT YET RUN as of writeup; image 1.3.267 rebuild in flight (Cloud Build source-upload phase). The deployment-grade 4/4 close criterion (F1 lockbox recall, F2 shortcut weakening, F3 no shortcut shifting, F4 HDTF cross-substrate FPR) is unresolved on every pillar. Read the F2/F3 baseline reference table at `analysis/deeplive_viso_corr_eval_2026-05-06/abs_pearson_summary.csv` before reading any verdict-style framing into the in-flight numbers.

> **Critical-reading note**: both arms train on the **post-fix** codebase for `quality_enhancement` routing (commits `ce76289` + `6102b05` landed 2026-05-05; image 1.3.259 is the first post-fix image; PD's images 1.3.264/1.3.265 derive from 1.3.259). PD is therefore the **first R13 packet pair NOT contaminated** by the routing bug documented in [`quality_enhancement_strategy_misrouting`](../threads/quality_enhancement_strategy_misrouting.md). Cross-packet comparisons to pre-fix R13 ckpts (P8A, E2B, PA, PC, etc.) carry the contamination caveat from the canonical thread; relative comparisons inside PD (deeplive vs viso arm) do not.

## Status card

| Field | Value |
|---|---|
| Dates | 2026-05-05 (loss class committed; smoke iteration; encoder fine-tune launched) → 2026-05-06 (training SUCCEEDED for both arms; CPU baseline audit complete; scorecard pending) |
| Slots | 2 (`R13_DEEPLIVE_CORR_PENALTY` = deeplive ship, `R13_VISO_CORR_PENALTY` = viso ship) |
| Headline lever | Train-time penalty `lambda * sum_axes \|Pearson_batch(score, axis)\|` over {sharpness_laplacian, luma_mean, face_area_fraction} at λ=1.0; encoder fine-tune from E2B_TOP_N_STEP3200; single-lever discipline (anchor_aware + face_scale_jitter + arcface_head all DISABLED) |
| Leader slot | TBD (scorecard pending — current "leader" by holdout AUC is `8jgyw1am/top_n_step4800` AUC 0.9970, but holdout AUC is not deployment-grade per `value_composite_semantics.md`) |
| Leader metric | TBD |
| Verdict | 🟡 in-flight — training complete; scorecard pending |
| Next-packet decision | Conditional on scorecard verdict: positive on 4/4 → corr-penalty becomes a "Confirmed good" lever; negative on any pillar → join the "structurally working but not deployment-grade" list with anchor_aware / GRL / face_scale_jitter |
| Themes touched | [correlation_penalty_loss](../threads/correlation_penalty_loss.md) (primary — packet's home thread) · [processing_signature_shortcut](../threads/processing_signature_shortcut.md) (the shortcut taxonomy this lever attacks) · [face_size_label_leak](../threads/face_size_label_leak.md) (face_area_fraction as the explicit-form decorrelation companion to face_scale_jitter) · [anti_shortcut_bundle_decomposition](../threads/anti_shortcut_bundle_decomposition.md) (single-lever discipline) · [viso_bucket_gap](../threads/viso_bucket_gap.md) (the viso arm enables visomaster_enhanced + teams_enhanced sources at fw=4.0 mirroring PA) · [wandb_yaml_propagation_bugs](../threads/wandb_yaml_propagation_bugs.md) (Smoke #1 was the 5th wandb-flattening recurrence) · [quality_enhancement_strategy_misrouting](../threads/quality_enhancement_strategy_misrouting.md) (PD is the first R13 packet pair post-fix) |

## Configuration

### Common to both arms

- **FT base**: `E2B_TOP_N_STEP3200` at `gs://training-job-outputs/best_checkpoints/rmat8lwx/top_n_effort_20260503_step3200_auc0.9863_eer0.0310.pth` — same FT-from-CLIP-scratch base PA and PC inherited from. **Risk inherited**: per memory `project_pa_does_not_generalize_to_hdtf_2026-05-05.md`, FT-from-E2B family failed on HDTF cross-substrate (PA 7.87% vs P8A 93.57%). Whether this generalizes to the corr-penalty arms is one of the 4 unresolved close-criterion pillars.
- **Loss block** (`experiments/phase2_round13/R13_DEEPLIVE_CORR_PENALTY.yaml:106-117`):

  ```yaml
  correlation_penalty:
    enabled: true
    lambda: 1.0  # Phase 2 prototype proved stable; lambda>=10 oscillates.
    axes:
      - sharpness_laplacian   # Lap-variance on luma channel; documented shortcut
      - luma_mean             # mean Y-channel; secondary IQ shortcut
      - face_area_fraction    # per-sample face area (bbox / image); face-size leak axis
  ```

  `face_area_fraction` is sourced via dataloader lookup from `analysis/deeplive_face_geometry_2026-05-05/face_area.parquet` (65,202 rows, 99.89% MediaPipe face-detection coverage). Falls back to NaN-skip if absent in the parquet.

- **Single-lever discipline** (`experiments/phase2_round13/R13_DEEPLIVE_CORR_PENALTY.yaml:118-130`): `anchor_aware: false`, `face_scale_jitter: false`, `use_arcface_head: false`. The corr-penalty is the ONLY anti-shortcut intervention active; per the discipline rule in [`anti_shortcut_bundle_decomposition`](../threads/anti_shortcut_bundle_decomposition.md), this is required for the bundle-vs-single-lever comparison to be readable.

- **Schedule**: `nEpochs: 8`, `total_training_steps: 2500` (the deeplive arm ran past the 2500-step cap to ~step5000 via `nEpochs` — see Results). Periodic saves at steps `[200, 500, 1000, 1500, 2000, 2500]`. Top-N saves on holdout AUC.

### Variants

- **Deeplive arm** (`R13_DEEPLIVE_CORR_PENALTY.yaml`, seed 5051):
  - Visomaster sources DISABLED (per user 2026-05-05: "we don't need any visomaster data Instead, we focus on using the DF40 data...").
  - Sources enabled: `deeplive` family (clean + enhanced + teams variants), `df40`.
  - Question this arm answers: does the corr-penalty + deeplive ship setup hit the 90/10 deployment goal on the deeplive lockbox suites without visomaster training data?
  - Run id: `8jgyw1am` ("rogue-senate-951"). Image 1.3.264. us-east1.

- **Viso arm** (`R13_VISO_CORR_PENALTY.yaml`, seed 5052):
  - Visomaster sources ENABLED at fw=4.0 (`visomaster`, `visomaster_enhanced`, `visomaster_teams_enhanced`) — mirrors PA's data-axis lever.
  - Sources enabled: deeplive family, df40, full visomaster lanes.
  - Question this arm answers: does corr-penalty + the data-availability lever (the only single-lever data-axis intervention that has produced an F4 v2 lift, per [`viso_bucket_gap`](../threads/viso_bucket_gap.md) 2026-05-05 mid-morning update) compose into a deployment-grade win, or does the v2-substrate-bound caveat inherited from PA hold?
  - Run id: `7u3zc5zt` ("elegant-emperor-952"). Image 1.3.265. us-east1.

### Yaml-only enablement plus three integration fixes

The lever required three integration fixes between Smoke #1 and the launched runs (commits `b1a0173` → `9dfd16f`, image 1.3.260 → 1.3.265):

1. **Wandb-flattening allowlist** (`29d0342`, image 1.3.263): `correlation_penalty` was being silently DISABLED at training time because `wandb.init(config=single_cfg)` flattens nested dicts and `self.config.get('correlation_penalty')` returned `None`. Added to `train_sweep.py` re-apply allowlist (lines ~176-282). Regression-tested in `tests/test_train_sweep_reapply_allowlist.py:32-46`. This is the **5th surface bug of the wandb-flattening class** documented in [`wandb_flattening`](../threads/wandb_flattening.md) — every new top-level nested-dict yaml block hits the same trap.
2. **5-D video batch handling** (`0cd5df5`, image 1.3.264): `compute_pixel_axes` raised `ValueError: expected [B,3,H,W]; got (4, 8, 3, 224, 224)`. The `combined_paired` collate emits 5-D `[B, T, 3, H, W]`. Reshape to `[B*T, 3, H, W]` and broadcast per-video face_area_fraction to per-frame via `repeat_interleave(t_repeat)` when `score_n % raw.shape[0] == 0`.
3. **Face-area parquet force-add** (`b7e7477`): `*.parquet` was in `.gitignore`; `git add -f` was required to commit `face_area.parquet`.

## Results at the time

### Training-phase outcomes (2026-05-05 → 2026-05-06)

| arm | run id | started UTC | ended UTC | wall-clock | image | n ckpts | best top_n AUC | best top_n EER |
|---|---|---|---|---|---|---:|---:|---:|
| Deeplive | `8jgyw1am` | 2026-05-05 22:19:22 | 2026-05-06 00:53:09 | 2h34m | 1.3.264 | 11 | 0.9970 (step4800) | 0.0113 |
| Viso | `7u3zc5zt` | 2026-05-05 23:34:57 | 2026-05-06 01:01:20 | 1h26m | 1.3.265 | 7 | 0.9934 (step600) | 0.0273 |

Both jobs SUCCEEDED. Both report `Correlation penalty ENABLED: lambda=1.0, axes=['sharpness_laplacian', 'luma_mean', 'face_area_fraction']` in the early training log (per stream-logs grep on 2026-05-06).

**Asymmetric step counts**: deeplive top_n at step4800; viso top_n at step600. Both arms have `total_training_steps: 2500` and `nEpochs: 8` set; `nEpochs` evidently took precedence on deeplive but not on viso. Root cause not isolated. Best deeplive ckpt (`top_n step4800`) trained well past the planned 2500-step cap.

### Checkpoint inventory

**Deeplive** (`gs://training-job-outputs/best_checkpoints/8jgyw1am/`, 11 ckpts):
- periodic: step200, step1000, step2000
- top_n: step200 (0.9937), step1400 (0.9948), step1800 (0.9962), step3800 (0.9964), step4800 (0.9970)
- value_composite: step200, step600
- first_best: ep1

**Viso** (`gs://training-job-outputs/best_checkpoints/7u3zc5zt/`, 7 ckpts):
- periodic: step200, step1000, step2000
- top_n: step200 (0.9927), step600 (0.9934)
- value_composite: step200
- first_best: ep1

3 deeplive + 3 viso entries (highest top_n + mid-trajectory top_n + a periodic anchor) plus P8A and E2B baselines were committed to `arena/checkpoint_maps/teams_target_domain.deeplive_viso_corr_2026-05-06.yaml` for the GPU scorecard.

### CPU baseline audit (2026-05-06 morning)

`analysis/deeplive_viso_corr_eval_2026-05-06/run_audit.py` ran the 5-axis shortcut audit on **8 BASELINE ckpts** (E2B, P8A, PA top_n_5600 / top_n_3800 / periodic_5000, PC top_n_5400 / top_n_7400 / periodic_5000) using the existing `analysis/pa_pc_eval_2026-05-05/raw_reports/` data. This is the F2/F3 reference table the new corr-penalty ckpts will be compared against once their scorecard reports land. Outputs: `coverage.csv`, `per_axis_recall_fpr.csv` (1808 rows), `correlations.csv` (120 rows), `abs_pearson_summary.csv` (48 rows), `tau_calibration.csv`.

**Mean |Pearson r| (score vs penalty axis), averaged across suites**:

```
real suites
                         face_area  luma_mean  sharpness
e2b   (FT base)           0.317     0.094      0.369
p8a                       0.117     0.312      0.188
pa_top_n_5600             0.222     0.106      0.335
pa_top_n_3800             0.220     0.175      0.340
pa_periodic_5000          0.208     0.137      0.269
pc_top_n_5400             0.351     0.270      0.181
pc_top_n_7400             0.367     0.294      0.182
pc_periodic_5000          0.380     0.316      0.119

fake suites
                         face_area  luma_mean  sharpness
e2b                       0.322     0.175      0.162
p8a                       0.412     0.148      0.188
pa_top_n_5600             0.335     0.221      0.200
pc_top_n_5400             0.196     0.049      0.098
```

**Key reads from the baseline (these are NOT corr-penalty results — they are the pre-corr-penalty reference)**:

- **E2B (the FT base)** carries non-trivial correlation on all 3 axes for both real and fake suites (0.094-0.369). PD's penalty is targeting these.
- **PA** (which trained on the same E2B base + visomaster data lever, no corr-penalty) shows axis-uneven weakening: face_area -30% on real (0.317→0.222), sharpness -9% on real, luma +13%. Modest, not bundle-grade.
- **PC** (E2B base + codec aug + visomaster) demonstrates the **shortcut-shifting failure mode** without corr-penalty: real-suite sharpness -51% (0.369→0.181, the codec aug's intended target) but luma AMPLIFIED +187% (0.094→0.270). On F3's "no axis +50%" criterion, PC fails by ~4× on luma alone.
- **P8A** (the production anchor — trained from a different FT chain through R12G) shows a different shortcut profile: lower face_area on real (0.117) but higher on fake (0.412); higher luma on real (0.312); roughly half the sharpness signal of E2B (0.188 vs 0.369).

**The corr-penalty F2 close criterion** (≥30% weakening on ≥2 of {sharpness, luma, face_area, is_webcam, is_screen}) is computed against E2B's baseline above. The corr-penalty F3 close criterion (no untargeted axis +50%) requires `is_webcam` and `is_screen` audit columns; those will be added when the scorecard lands its per-frame reports and the Phase 2 audit runs.

### Promotion-contract scorecard

**Not yet run.** Image 1.3.267 rebuild in flight (Cloud Build source-upload phase, 5GB tarball, ~10min upload + ~5-10min build). Scorecard launches once 1.3.267 is published; us-east1; suite `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml`; ckpt map `arena/checkpoint_maps/teams_target_domain.deeplive_viso_corr_2026-05-06.yaml` (8 entries: P8A baseline, E2B FT base, 3 deeplive_corr ckpts, 3 viso_corr ckpts). ETA to verdict: ~2-3h after image lands.

## Conclusions drawn in-session

- **Lever class is technically working**: gradient flows through the penalty term (verified by `tests/test_correlation_penalty.py:test_gradient_flows_through_score`), 5-D video batches are correctly handled (`test_accepts_5d_video_batch` after `0cd5df5`), wandb-flattening is bypassed (`tests/test_train_sweep_reapply_allowlist.py`), the loss is enabled at training time (per stream-logs grep), and both arms produced expected ckpt counts on a non-trivial schedule. Whether the lever class translates to deployment-grade outcome is unresolved.

- **The frozen-head Phase 1 prototype** (analysis/corr_penalty_prototype_2026-05-05/) showed:
  - Mechanism converges across `λ ∈ {0, 0.1, 1, 10, 100}`. λ=10 starts oscillating; λ=100 collapses to chance.
  - At λ=1: targeted axes |r_sharp| 0.156→0.053 (-66%), |r_luma| 0.233→0.071 (-69%); recall@FPR=0.10 INCREASED 0.355→0.491 (+13.6pp) on a 1037-frame identity-disjoint test split.
  - Untargeted axes (`face_area`, `is_webcam`) **rebound** at λ=1 (face_area 0.032→0.157, is_webcam 0.081→0.205) — the canonical "shortcut shifting" symptom on a frozen head. **The PD encoder fine-tune adds `face_area_fraction` as a third penalty axis** to close one of the two prototype-shifting paths. `is_webcam` is NOT a penalty axis because Teams does not surface capture mode at inference (`feedback_per_mode_tau_not_deployable.md`); whether the encoder fine-tune amplifies `is_webcam` correlation is one of the questions the scorecard's 5-axis audit will answer.

- **The 8-baseline reference table establishes that** PA and PC's data-axis / codec-aug interventions did NOT clear the 4/4 close criterion on shortcut-weakening grounds **even before** the cross-substrate validation question (which they failed independently — see `viso_bucket_gap.md` 2026-05-05 mid-morning + late-night updates). PD's success or failure will be readable against the same table.

- **The lever class is theoretically orthogonal to the data-axis and augmentation-axis levers** that prior R13 packets exercised. P22 / S1-S3 / E-series varied the input distribution (augmentation curriculum, data composition); PA / PC varied data sources and codec aug. PD varies the loss surface — adding a regularizer that penalizes the model for using the nuisance axes as predictors. Whether this orthogonality empirically composes into a stronger result remains to be seen on the scorecard.

- **Single-lever discipline is preserved**: PD does not stack `anchor_aware`, `face_scale_jitter`, or `arcface_head` on top of the corr-penalty. Per `anti_shortcut_bundle_decomposition.md`, the bundle-vs-single-lever ablation requires this. If PD succeeds, a successor packet can ablate stack-of-2 (e.g., corr-penalty + face_scale_jitter) to test whether the levers compose; if PD fails, the lever class is muddled with no single-lever-vs-bundle confound to debug.

- **Session IDs / commits**: `b1a0173` (loss class + deeplive yaml + smoke), `ccb03f7` (face_area parquet + 3rd axis), `b7e7477` (parquet force-add), `29d0342` (wandb-flattening allowlist fix), `0cd5df5` (5-D batch handling), `a12e6e5` (viso yaml), `9dfd16f` (VERSION 1.3.265 publish for viso), `b1a0173` covers the 21 unit tests in `test_correlation_penalty.py`.

## Retrospective (as of 2026-05-06 morning)

**Packet is in-flight; deployment-grade verdict pending.** Training succeeded for both arms, the lever class is wired correctly per smoke verification, and the F2/F3 baseline reference table is in hand. The scorecard run is mechanically queued (waiting on image 1.3.267 to publish).

**What this packet WILL tell us** (post-scorecard):
1. Whether the corr-penalty as deployed weakens E2B's documented shortcuts on the 5 audit axes by ≥ 30% on ≥ 2 axes (F2).
2. Whether the prototype's predicted shortcut-shifting on `is_webcam` materializes on the encoder (F3).
3. Whether either arm clears the 90/10 lockbox criterion (F1).
4. Whether the v2-substrate ceiling P8A breaks at 27% on `visomaster_enhanced_macro_dev` is also broken by either arm — and crucially whether the PA/PC walkback pattern (F4 v2 lift not generalizing to HDTF) repeats here.

**What this packet WILL NOT directly tell us** (out of scope for PD's design):
1. Whether λ=1 is the operating point — there is no in-packet sweep over λ.
2. Whether stacking corr-penalty + face_scale_jitter composes — that's a successor packet's question, deliberately deferred to preserve single-lever discipline.
3. Whether `is_webcam` or `is_screen` should be in the penalty axes — these are not deployable at inference under current Teams constraints, so even if the audit shows they're shifting axes, adding them requires a different inference-time signal.

**Preprocessing-parity note**: PD's training inputs run through the post-fix INTER_LINEAR path (commit `855871e` landed in 2026-04-24 well before PD). Any post-scorecard cross-packet comparison to pre-fix R13 ckpts (RLP1-RLP6 except RLP6_04, etc.) carries the parity caveat from [`preprocessing_parity_bug`](../threads/preprocessing_parity_bug.md).

**Quality_enhancement parity note**: PD's training inputs also run through the **post-fix** family-routing for `quality_enhancement` (commits `ce76289` + `6102b05` landed 2026-05-05; PD's images derive from 1.3.259 the first post-fix image). PD is the first R13 packet pair NOT contaminated. Any cross-packet comparison to pre-fix R13 ckpts (every prior P-series, S-series, E-series, A/B/C-series) should annotate the contamination delta from [`quality_enhancement_strategy_misrouting`](../threads/quality_enhancement_strategy_misrouting.md). Relative comparisons inside PD (deeplive vs viso arm) are clean.

**Open workstreams as of 2026-05-06 morning**:
- Image 1.3.267 rebuild Cloud Build job, source-upload phase.
- Once published: GPU scorecard on the 8-entry ckpt map; ~2-3h to verdict.
- Phase 2 of the 5-axis audit (uncomment block in `analysis/deeplive_viso_corr_eval_2026-05-06/run_audit.py`, run after `gcloud storage cp -r <scorecard>/raw_reports analysis/deeplive_viso_corr_eval_2026-05-06/raw_reports/`).
- Cross-substrate (HDTF) eval is NOT in the scheduled scorecard's suite manifest; once F0 + F4 land for v2 substrate, the F4-criterion HDTF verification is the next CPU/GPU follow-up. Job-B-style validation against `proper_data_future.provisional_2026-04-19.yaml` is the precedent.

**Cross-references.** Story continues in [`correlation_penalty_loss`](../threads/correlation_penalty_loss.md) (the lever-class home thread) and the open loop `corr-penalty-deployment-grade-verdict-pending` therein.

## Source files

- **Yamls**: `experiments/phase2_round13/R13_DEEPLIVE_CORR_PENALTY.yaml`, `experiments/phase2_round13/R13_VISO_CORR_PENALTY.yaml`, `experiments/phase2_round13/R13_DEEPLIVE_CORR_PENALTY_SMOKE.yaml`.
- **Code**: `loss/correlation_penalty.py` (88 LOC), `tests/test_correlation_penalty.py` (21 tests), `tests/test_train_sweep_reapply_allowlist.py:32-46` (regression test). Detector hook in `detectors/effort_detector.py` `_setup_loss_function` + `get_losses` (per session-summary).
- **Checkpoint map**: `arena/checkpoint_maps/teams_target_domain.deeplive_viso_corr_2026-05-06.yaml` (8 entries).
- **Analysis**: `analysis/corr_penalty_prototype_2026-05-05/{lambda_sweep.csv,loss_curves.csv,run.log,scripts/}` (Phase 1 frozen-head prototype); `analysis/deeplive_viso_corr_eval_2026-05-06/{run_audit.py,coverage.csv,correlations.csv,abs_pearson_summary.csv,per_axis_recall_fpr.csv,tau_calibration.csv}` (Phase 1 baseline audit; Phase 2 corr-penalty audit pending scorecard).
- **Face-area parquet**: `analysis/deeplive_face_geometry_2026-05-05/face_area.parquet` (1.13 MB, 65,202 rows, 99.89% face-detection coverage).
- **Memory pointers**: `project_image_quality_shortcut.md` (the shortcut taxonomy); `project_score_distribution_audit_2026_05_02.md` (the empirical foundation); `project_p22_cpu_followups_reframe_2026-05-02.md` (the previous-generation augmentation-axis lever).
- **Wandb**: https://wandb.ai/dtect-vision/phase2r13-experiments/runs/8jgyw1am (deeplive), https://wandb.ai/dtect-vision/phase2r13-experiments/runs/7u3zc5zt (viso).
- **Vertex jobs**: `8413298392595169280` (deeplive ship), `4185544242401116160` (viso ship), both us-east1, both SUCCEEDED.
