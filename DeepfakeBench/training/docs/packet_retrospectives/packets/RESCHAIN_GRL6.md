# Packet RESCHAIN_GRL6 · two overnight single-lever forks targeting resolution-chain instability

> **Template contract**: one packet retro covering two jointly-scorecard'd
> runs (Slot α resolution_chain_aug + Slot β multi_axis_grl 4→6 axes), both
> FT-from-T5C-step3500, launched 2026-05-15 evening.

## Status card

| Field | Value |
|---|---|
| Dates | 2026-05-15 22:39 UTC (launch) → 2026-05-16 10:36 UTC (scorecard verdict) |
| Slots | 2 (Slot α RESCHAIN · Slot β 6AXIS_GRL) |
| Headline lever | data-axis vs loss-axis attacks on the resolution-chain shortcut (CPU probe earlier same day measured P8A real score_range 0.97 across 20 down→up variants) |
| Leader slot | *(none — P8A retains rank 1; Slot β has highest dev_fake_macro_recall but largest lockbox_real_fpr)* |
| Leader metric | Slot β step3500: dev_fake_macro_recall 0.545 (largest in scorecard), viso 0.235 (largest, +70% over T5C), lockbox_real_fpr 0.088 (largest, 4.8× P8A) |
| Verdict | ⚠️ muddled — Slot α mechanism partially confirmed at CPU level, falsified at contract level (step3500 fails recall floor; step1500 promotes but regresses viso); Slot β surfaces unexpected viso lift via a different mechanism than designed |
| Next-packet decision | CPU-first per-identity decomposition of Slot β lockbox_real_fpr (~$0) + lockbox_real_fpr tiebreak re-eval per D4 — pending user authorization |
| Themes touched | [iq_shortcut_deconvolution_program_2026-05-08](../threads/iq_shortcut_deconvolution_program_2026-05-08.md), [image_quality_shortcut](../threads/image_quality_shortcut.md), [face_size_label_leak](../threads/face_size_label_leak.md) |

## Configuration

What changed vs T5C step3500 base:

- **Slot α** (`R13_T5C_RESCHAIN_2026-05-15.yaml`, seed 9911). NEW augmentation
  `data/augmentations/resolution_chain_aug.py` (12 unit tests, all pass).
  Random downsample to {64,96,128,160,192} then upsample back to native (H,W)
  with the same kernel ∈ {LINEAR,CUBIC,AREA,LANCZOS4}, `p_apply=0.5`. Applied
  BEFORE the canonical 224×224 resize, after `face_scale_jitter` and before
  `fourier_band_aug` in the collate path. Wired in `trainer/trainer.py:694`,
  `data/sources/combined_paired.py:3838`, `data/batching/df40_paired.py:364`.
  Allowlist entry at `train_sweep.py:308`. All other levers identical to
  T5C base.

- **Slot β** (`R13_T5C_6AXIS_GRL_2026-05-15.yaml`, seed 9912). Pure YAML diff —
  extends `multi_axis_grl.axes` from 4 (chronic_flag, is_dor,
  sharpness_laplacian_high, color_a_approx_dev_high) to 6 (added
  `color_b_dev_high` + `luma_mean_high`). Both new axes are already in
  `MultiAxisGRLBlock.SUPPORTED_AXES` (`detectors/effort_detector.py:390`).
  The 2 new per-axis classifier heads initialize randomly at FT start (the
  T5C base ckpt has heads for the original 4 axes only); the existing 4 axes'
  heads load from T5C's state_dict.

Anchors in the scorecard map:
- `P8A_REFERENCE_STEP5000` (rank-1 across 13+ packets)
- `T5C_PERIODIC_STEP3500` (FT base for both slots; rank-3 on v3-fix per T6/T7/T5C scorecard 2026-05-12)

Variants tested in the scorecard:
- `SLOT_A_RESCHAIN_STEP1500` (mid-trajectory; chosen to catch potential late-step overfit)
- `SLOT_A_RESCHAIN_STEP3500` (terminal periodic)
- `SLOT_B_6AXIS_GRL_STEP3500` (terminal periodic; Slot β trajectory not sampled mid-step due to scorecard cell-count budget)

Links to yamls:
- [`experiments/phase2_round13/R13_T5C_RESCHAIN_2026-05-15.yaml`](../../../experiments/phase2_round13/R13_T5C_RESCHAIN_2026-05-15.yaml)
- [`experiments/phase2_round13/R13_T5C_6AXIS_GRL_2026-05-15.yaml`](../../../experiments/phase2_round13/R13_T5C_6AXIS_GRL_2026-05-15.yaml)

Scorecard ckpt map: [`arena/checkpoint_maps/teams_target_domain.overnight_2026-05-16.yaml`](../../../arena/checkpoint_maps/teams_target_domain.overnight_2026-05-16.yaml). Suite manifest: `arena/target_domain_suites.teams_promotion_contract_minimal_9suite_2026-05-14.yaml` (9 suites, ~3.2× cheaper than the 29-suite full).

## Results at the time

### Training outcomes (both SUCCEEDED)

| # | Slot | W&B run | Vertex job | Region | Start (UTC) | End (UTC) | Runtime | Terminal ckpt |
|---|---|---|---|---|---|---|---:|---|
| α | R13_T5C_RESCHAIN | `lsx4n0t7` | `679364389144363008` | us-east1 | 2026-05-15T22:41:31Z | 2026-05-16T00:28:34Z | 1h 47m | `periodic_step3500` |
| β | R13_T5C_6AXIS_GRL | `gwntcld0` | `5517273481877651456` | us-west4 | 2026-05-15T22:48:04Z | 2026-05-16T03:11:07Z | 4h 23m | `periodic_step3500` |

Slot β's longer wall time is consistent with the 2 additional axis classifier heads + bottleneck per step.

### Scorecard verdict (45 cells = 5 ckpts × 9 suites)

Vertex `4599114546072780800`, us-east1, image `1.3.290`, runtime 2h 32m, SUCCEEDED 2026-05-16T10:35:56 UTC.

| rank | ckpt | dev_fake_macro | viso_enh | deeplive_enh | teams_fake_dev | lockbox_real_fpr | lockbox_fake | τ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | P8A_REFERENCE_STEP5000 | 0.300 | 0.136 | 0.239 | 0.526 | **0.018** | 0.387 | 0.916 |
| 2 | T5C_PERIODIC_STEP3500 | 0.459 | 0.138 | 0.626 | 0.613 | 0.028 | 0.660 | 0.831 |
| 3 | SLOT_A_RESCHAIN_STEP1500 | 0.443 | 0.056 | **0.717** | 0.554 | 0.046 | 0.364 | 0.899 |
| 4 | SLOT_B_6AXIS_GRL_STEP3500 | **0.545** | **0.235** | **0.738** | **0.664** | 0.088 | 0.541 | 0.816 |
| 5 | SLOT_A_RESCHAIN_STEP3500 | **0.226** ❌ | 0.018 | 0.198 | 0.461 | 0.015 | 0.372 | 0.860 |

Slot α step3500 falls below the `dev_fake_macro_recall ≥ 0.30` floor.
Slot β step3500 has the highest dev_fake_macro_recall of any ckpt scored
AND the largest lockbox_real_fpr.

CPU probe follow-up (`analysis/cpu_diagnostics_2026-05-15_resolution_chain/RESULTS_OVERNIGHT_FACTS_2026-05-16.md`): Slot α step3500 median real `score_range` 0.448 vs T5C 0.605 (25% reduction); Slot β step3500 0.586 (3% reduction, within noise). The CPU-probe mechanism close criterion was approximately met for Slot α; the contract close criteria were not.

## Conclusions drawn in-session

### Factual evidence (read first)

- [`analysis/reschain_grl6_eval_2026-05-16/RESULTS_FACTS_2026-05-16.md`](../../../analysis/reschain_grl6_eval_2026-05-16/RESULTS_FACTS_2026-05-16.md) — full scorecard tables.
  - §3: P8A rank 1; Slot α step3500 fails recall floor; Slot β step3500 has highest dev_fake_macro across 3 fake suites.
  - §4: Slot α step1500 vs step3500 trajectory deltas — viso regresses at both, dev_macro stays only at step1500.
  - §5: Slot β step3500 +0.097 viso_enhanced_macro_dev vs T5C (+70% relative); +0.060 lockbox_real_fpr (3.2× T5C, 4.8× P8A).
- [`analysis/reschain_grl6_eval_2026-05-16/DEEP_DIVE_FACTS_2026-05-16.md`](../../../analysis/reschain_grl6_eval_2026-05-16/DEEP_DIVE_FACTS_2026-05-16.md) — CPU-probe vs scorecard cross-reference.
  - §2-3: Slot α step3500 fake-side score distribution compressed alongside the real-side; mean fake scores [0.65, 0.78] vs T5C's wider spread.
  - §7: two distinct mechanisms (encoder invariance vs score-distribution compression) both produce low `score_range`; the probe did not distinguish them.
  - §8: canary probe was disabled in both yamls (inherited omission from T5C base yaml); would have surfaced fake-side score crash by step 500-1000.

### In-session opinion (read second, with skepticism)

[`analysis/reschain_grl6_eval_2026-05-16/AGENT_PROPOSAL_2026-05-16.md`](../../../analysis/reschain_grl6_eval_2026-05-16/AGENT_PROPOSAL_2026-05-16.md) — single OPINION doc, same agent as the FACTS docs.

Mechanism claims summary (NOT verdict):

1. (MEDIUM) `resolution_chain_aug` compresses fake-side and real-side score distributions together; falsifier proposed at §3.1.
2. (MEDIUM-HIGH) The 6-axis GRL bit a visomaster-detection axis not designed-for; falsifier proposed at §3.2 (a 5-axis variant testing color_b marginal contribution).
3. (LOW-MEDIUM) Slot β's lockbox_real_fpr is plausibly identity-localized; CPU-only decomposition probe is the cheap test.

Retractions named in the opinion-doc Self-correction log:
1. (6.1) "Slot α is a clear win on resolution-chain stability" — scope-overstated; correct at the CPU-probe metric, refuted at the contract level.
2. (6.2) "Slot β is a null result" — scoped to the resolution-chain metric only; on contract metrics Slot β has the largest positive deltas of any scored ckpt.
3. (6.3) "scorecard is the followup test of the mechanism" — conflated two distinct tests; the CPU probe tests the mechanism, the scorecard tests the contract.

Pass-1 / pass-2 independence is NOT guaranteed in this session — same agent wrote FACTS + OPINIONS docs. Treat the OPINIONS doc with extra skepticism.

## Eval folder

[`analysis/reschain_grl6_eval_2026-05-16/`](../../../analysis/reschain_grl6_eval_2026-05-16/) follows [`eval_folder_template.md`](../eval_folder_template.md):

- FACTS: `RESULTS_FACTS_2026-05-16.md`, `DEEP_DIVE_FACTS_2026-05-16.md`.
- OPINION: `AGENT_PROPOSAL_2026-05-16.md` (with §6 Self-correction log).

The sister CPU-probe artifacts at `analysis/cpu_diagnostics_2026-05-15_resolution_chain/` (`RESULTS_FACTS_2026-05-15.md` for the discovery probe, `RESULTS_OVERNIGHT_FACTS_2026-05-16.md` for the new-ckpt follow-up) are the original mechanism characterization; this eval folder cross-references them.

## Retrospective

- **Open at session end**: per-identity decomposition of Slot β step3500's lockbox_real_fpr is the highest-EV CPU follow-up. Decision criterion in OPINION §7.
- **Open at session end (load-bearing for the program)**: the lockbox_real_fpr tiebreak that keeps P8A at rank 1 across 13+ packets may not be load-bearing (D4 2026-05-12 showed lockbox FPR at dev-cal τ is below dev FPR). Re-eval per OPINION §5.2 is the structural question; cost is $0 CPU on the existing scorecard outputs.
- **Preprocessing-parity note**: both slots trained on image `1.3.290` (post-`855871e` INTER_LINEAR fix, post-`2feea58` codepath, post-quality-enhancement-routing-fix). Comparable with all 2026-05+ packets including T3, T4, T5C, T6/T7, U_SLOTS.
- *(2026-05-16 — placeholder)* If a Slot β per-identity probe lands, this section should record whether the lockbox_real_fpr penalty is identity-localized.
- Cross-reference to threads: [`iq_shortcut_deconvolution_program_2026-05-08`](../threads/iq_shortcut_deconvolution_program_2026-05-08.md) — Slot α tests Stage 2a's IQ-axis-attack at the data layer for the size sub-axis; the result tightens the close criterion for Stage 2a (preserve fake-vs-real AUC, not just reduce score_range).

## Source files

- **Yamls**: `experiments/phase2_round13/R13_T5C_RESCHAIN_2026-05-15.yaml`, `R13_T5C_6AXIS_GRL_2026-05-15.yaml`
- **Aug module**: `data/augmentations/resolution_chain_aug.py` (new, 2026-05-15) + `tests/test_resolution_chain_aug.py` (12 unit tests)
- **Wiring**: `trainer/trainer.py:694`, `data/sources/combined_paired.py:3838`, `data/batching/df40_paired.py:364`, `train_sweep.py:308`
- **Checkpoint map**: `arena/checkpoint_maps/teams_target_domain.overnight_2026-05-16.yaml`
- **Suite manifest**: `arena/target_domain_suites.teams_promotion_contract_minimal_9suite_2026-05-14.yaml`
- **Scorecard outputs**: `gs://training-job-outputs/test_results/teams_promotion_contract/overnight-scorecard-2026-05-16/`
- **CPU probe (discovery)**: `analysis/cpu_diagnostics_2026-05-15_resolution_chain/RESULTS_FACTS_2026-05-15.md`
- **CPU probe (follow-up on trained ckpts)**: `analysis/cpu_diagnostics_2026-05-15_resolution_chain/RESULTS_OVERNIGHT_FACTS_2026-05-16.md`
- **Eval folder**: `analysis/reschain_grl6_eval_2026-05-16/`
- **Memory pointers**: `project_resolution_chain_instability_2026-05-15.md`, `project_overnight_resolution_chain_2026-05-16.md`, `project_t6_t7_t5c_scorecard_2026-05-12.md` (T5C step3500 baseline), `project_chronic_offenders_partition_per_ckpt_2026-05-04.md` (per-identity decomposition rationale)
- **Commits**: `14d4711` (overnight infra + yamls), `476be51` (overnight CPU-probe verdict), `fea4d5e` (scorecard ckpt map + image 1.3.290)
