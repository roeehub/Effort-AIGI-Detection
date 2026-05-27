# Packet T6 / T7 / T5C  ·  three single-lever forks (jitter on T3, jitter on T4, classifier-1024 on T4)

> **Template contract**: this is one packet retro covering three jointly-scorecard'd packets that share a launch session + scorecard run. T6 = T3 + face_scale_jitter@0.50; T7 = T4 + face_scale_jitter@0.50; T5C = T4 with online classifier hidden_dim 256 → 1024.

## Status card

| Field | Value |
|---|---|
| Dates | 2026-05-11 (launch) → 2026-05-12 (scorecard verdict) |
| Slots | 3 (T6, T7, T5C) |
| Headline lever | three single-lever forks: jitter on T3 / jitter on T4 / classifier-capacity on T4 |
| Leader slot | T5C (rank-3 contract, above T3_SLOT1 anchor) |
| Leader metric | T5C step3500: rank 3 on v3-fix; lockbox_real_fpr 0.0279 / lockbox_fake_recall 0.6601 |
| Verdict | ⚠️ muddled — T5C partially supported (one new candidate, narrow lockbox-clean window); T6+T7 ❌ failed (jitter regresses recall floor on both bases) |
| Next-packet decision | CPU first (L11 atlas recompute on T5C); only then consider T5C × hidden_dim sweep OR T5C × attachment-layer ablation |
| Themes touched | [iq_shortcut_deconvolution_program_2026-05-08](../threads/iq_shortcut_deconvolution_program_2026-05-08.md), [processing_signature_shortcut](../threads/processing_signature_shortcut.md), [promotion_contract_evolution](../threads/promotion_contract_evolution.md) |

## Configuration

What changed vs the prior packet (T4):

- **T6** forks from T3 SLOT1 (the F0-passing keep-list FT-from-P8A_step5000 from 2026-05-09). Single lever: adds `face_scale_jitter` with `scale_limit=0.50` to the augmentation stack. NO GRL. Seed 9601.
- **T7** forks from T4 (multi-axis-L11-GRL with hidden_dim=256). Single lever: adds `face_scale_jitter` with `scale_limit=0.50`. Seed 9701.
- **T5C** forks from T4. Single lever: changes online classifier `hidden_dim` from 256 → 1024. NO other recipe change. Seed 9501.

Control slot: T4 itself (multi-axis-L11-GRL, hidden_dim=256, no jitter) — already scorecarded on 2026-05-11 morning (see `packets/T4.md` reference; eval folder `analysis/t4_eval_2026-05-11/`).

Variants tested:

- T6 ckpts sampled: `periodic_step1500`, `periodic_step3500`, `top_n_step10250` (3 candidates). Run trained past nominal cap.
- T7 ckpts sampled: `periodic_step2500`, `periodic_step3500`, `top_n_step4750` (3 candidates). Run early-stopped at step 5000.
- T5C ckpts sampled: `periodic_step1500`, `periodic_step3500`, `top_n_step3750` (3 candidates). Run trained to step 5000 full.
- Anchors: P8A_REFERENCE_STEP5000 (production), E2B_TOP_N_STEP3200 (currently deployed), T3_SLOT1_PERIODIC_STEP1500 (last week's strongest new candidate).

Links to yamls:

- [`experiments/phase2_round13/R13_T6_T3_PLUS_JITTER_2026-05-11.yaml`](../../../experiments/phase2_round13/R13_T6_T3_PLUS_JITTER_2026-05-11.yaml)
- [`experiments/phase2_round13/R13_T7_T4_PLUS_JITTER_2026-05-11.yaml`](../../../experiments/phase2_round13/R13_T7_T4_PLUS_JITTER_2026-05-11.yaml)
- [`experiments/phase2_round13/R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.yaml`](../../../experiments/phase2_round13/R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.yaml)

Ckpt selection per `arena/checkpoint_maps/teams_target_domain.t67_t5c_2026-05-11.yaml` (pruned via AUC-progression heuristics — late-training AUC peaks tend to over-saturate the shortcut, mid-training ckpts often outperform late peaks at deployment).

## Results at the time

Scorecard `t67-t5c-scorecard-2026-05-11` (Vertex `6455763074874867712`, us-east1, image `1.3.283`, runtime 9h 54m): SUCCEEDED 2026-05-12 03:13 UTC.

Headline:

- **P8A retains contract rank-1.** No T6 / T7 / T5C ckpt displaces it.
- **T5C step3500 ranks 3**, the highest non-anchor placement: `lockbox_real_fpr` 0.0279 / `lockbox_fake_recall` 0.6601 / `dev_fake_macro_recall` 0.4589.
- **T3_SLOT1 step1500 (anchor) ranks 4**: lockbox 0.0309 / 0.7747 / 0.3763.
- **All 6 jitter ckpts (T6 ×3 + T7 ×3) fail the `dev_fake_macro_recall` floor of 0.30.**

Leader vs control delta (T5C step3500 vs T4 step10500, which ranked 3 of 7 on the T4 scorecard):

- `dev_fake_macro_recall`: T5C 0.4589 vs T4 0.4166 (+0.04)
- `lockbox_real_fpr`: T5C 0.0279 vs T4 0.0536 (−0.026, about half)
- `lockbox_fake_recall`: T5C 0.6601 vs T4 0.3676 (+0.29)
- `teams_real_dor_dev` FPR (n=50): T5C 0.06 (3/50) vs T4 0.18 (9/50) — directionally improved chronic-cohort behavior at small-N noise band.

Pool-level numbers that mattered:

- 6 / 12 ckpts pass all three contract gates. Among them, rank ordering tracks ascending `lockbox_real_fpr` (P8A 0.0184 → E2B 0.0235 → T5C step3500 0.0279 → T3_SLOT1 0.0309 = T5C step3750 0.0309 → T5C step1500 0.3204).
- T5C step1500 has the highest `dev_fake_macro_recall` (0.6832) of any scored ckpt but a catastrophic `lockbox_real_fpr` (0.3204, 17× P8A). Training-step trajectory matters: step3500 is clean, step1500 is not.
- T6 step1500 has the highest `lockbox_fake_recall` (0.8696) of any scored ckpt but `dev_fake_macro_recall` 0.2598 (below floor).

Links to scorecards / analysis artifacts / checkpoint maps:

- Eval folder: [`analysis/t6_t7_t5c_scorecard_eval_2026-05-12/`](../../../analysis/t6_t7_t5c_scorecard_eval_2026-05-12/)
- GCS scorecard: `gs://training-job-outputs/test_results/teams_promotion_contract/t67-t5c-scorecard-2026-05-11/`
- Checkpoint map: [`arena/checkpoint_maps/teams_target_domain.t67_t5c_2026-05-11.yaml`](../../../arena/checkpoint_maps/teams_target_domain.t67_t5c_2026-05-11.yaml)

## Conclusions drawn in-session

> **Required structure.** New agents read the FACTS subsection first to form an independent view. Read the OPINION subsection only after the user authorizes (or after independent view is committed).

### Factual evidence (read first)

- [`analysis/t6_t7_t5c_scorecard_eval_2026-05-12/RESULTS_FACTS_2026-05-12.md`](../../../analysis/t6_t7_t5c_scorecard_eval_2026-05-12/RESULTS_FACTS_2026-05-12.md) — full 12-ckpt scorecard verdict + per-suite tables.
  - §3.1: P8A is rank-1; selected_threshold 0.9156; dev_macro 0.3003; lockbox_real_fpr 0.0184; lockbox_fake_recall 0.3874.
  - §5.1: 6 of 12 ckpts pass all three contract gates; 6 fail (all T6 ×3 + all T7 ×3 fail the recall floor).
  - §5.2: among all-pass ckpts, rank ordering tracks ascending `lockbox_real_fpr`.
  - §6: T5C step3500's dev_macro lift (+0.1586 vs P8A) is concentrated on `deeplive_enhanced_dev` (+0.3853 of total).
  - §7.2: T5C step3500 vs P8A on lockbox: +0.0095 absolute FPR, +0.2727 absolute recall.
  - §8: teams_real_dor_dev (n=50) FPR — T5C step3500 0.06 vs P8A 0.08 vs T4 step10500 0.18 (from T4 eval).

### In-session opinion (read second, with skepticism)

- [`analysis/t6_t7_t5c_scorecard_eval_2026-05-12/AGENT_PROPOSAL_2026-05-12.md`](../../../analysis/t6_t7_t5c_scorecard_eval_2026-05-12/AGENT_PROPOSAL_2026-05-12.md) — single OPINION doc with self-correction log.
- Mechanism claim summary (NOT verdict): classifier-capacity is the lever responsible for T5C's rank-3 placement (MEDIUM); face_scale_jitter does not generalize as a free additive lever (HIGH); dev → lockbox substrate-transfer is the binding production constraint (MEDIUM).
- Retractions named in the opinion-doc Self-correction log: (1) memory `project_face_scale_jitter_load_bearing.md` is scoped too broadly — its "load-bearing single lever" claim was on the P14 bundle base; T6/T7 refute the composable-lever extension. (2) Yaml header comments claiming T5C "resolves" the classifier-capacity question are downgraded: the scorecard supports the proposition directionally but a verdict requires CPU atlas re-run on T5C features. (3) The Slot-3 T7 decision was suboptimal in hindsight — head_retrain β-outcome had already ruled out the alternative.
- The opinion doc is single-author and was written by the same agent who wrote the FACTS doc — pass-1 / pass-2 independence is not guaranteed in this session; treat the opinion doc with extra skepticism.

## Eval folder

[`analysis/t6_t7_t5c_scorecard_eval_2026-05-12/`](../../../analysis/t6_t7_t5c_scorecard_eval_2026-05-12/) — follows the [`eval_folder_template.md`](../eval_folder_template.md) FACTS/OPINIONS file-level split:

- FACTS: `RESULTS_FACTS_2026-05-12.md` (single doc; no F1-F5 close-criterion mechanics applied this session, since the contract verdict is the close criterion).
- OPINION: `AGENT_PROPOSAL_2026-05-12.md` (single doc with self-correction log).

## Retrospective

- **Open at session end**: the lockbox `lockbox_real_fpr` gap that keeps P8A at rank 1 across packets is now identified as the operational binding constraint (per opinion-doc §3.3 + thread `iq_shortcut_deconvolution_program_2026-05-08` open loop `dev-to-lockbox-substrate-transfer-gap`). T5C step3500 is the closest non-P8A candidate (0.0279 vs 0.0184 = +0.0095 absolute gap, ~52% larger).
- **Preprocessing-parity note**: T6/T7/T5C all trained on image `1.3.283`, post commit `855871e` (INTER_LINEAR fix); comparable with all 2026-05+ packets including T3 and T4.
- *(2026-05-12 — placeholder)* When the L11 atlas inv_mean recompute lands for T5C step3500, this section will record whether §3.1's "classifier capacity → chronic_6 invariance" mechanism is supported at the feature level.
- *(2026-05-12 — placeholder)* If a follow-up T5C × hidden_dim sweep launches, this section will record whether 1024 is a local optimum or a larger sweet spot exists.
- Cross-reference to threads: [`iq_shortcut_deconvolution_program_2026-05-08`](../threads/iq_shortcut_deconvolution_program_2026-05-08.md) — the program-level continuation; the open loops `dev-to-lockbox-substrate-transfer-gap` (high severity, open) and `chronic_6-feature-regression-on-t4` (low severity, partial-resolve) are amended in this session's update.

## Source files

- **Yamls**: `experiments/phase2_round13/R13_T6_T3_PLUS_JITTER_2026-05-11.yaml`, `R13_T7_T4_PLUS_JITTER_2026-05-11.yaml`, `R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.yaml`
- **Checkpoint map**: `arena/checkpoint_maps/teams_target_domain.t67_t5c_2026-05-11.yaml`
- **Scorecard outputs**: `gs://training-job-outputs/test_results/teams_promotion_contract/t67-t5c-scorecard-2026-05-11/`
- **Eval folder**: `analysis/t6_t7_t5c_scorecard_eval_2026-05-12/`
- **Pre-launch CPU diagnostics** (2026-05-11): `analysis/cpu_diagnostics_2026-05-11_a1_cross_substrate/`, `cpu_diagnostics_2026-05-11_a2_linear_probe/`, `cpu_diagnostics_2026-05-11_a3_atlas_composition/`, `cpu_diagnostics_2026-05-11_a2_extension/`, `cpu_diagnostics_2026-05-11_head_retrain/`. (T67_T5C_PROBE_FACTS at `cpu_diagnostics_2026-05-11_t67_t5c_probe/` abstained due to network throughput — not load-bearing for the scorecard verdict.)
- **Memory pointers**: `project_t4_substrate_overfit_inv_mean_misleading_2026-05-11.md` (parent finding for T5C); `project_face_scale_jitter_load_bearing.md` (refuted as composable lever this session); `project_signature_shortcut_finding.md` (dor_shkedi vs real_dor signature flip; concentrates the lockbox failure cohort).
