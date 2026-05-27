# Independent critic review — 2026-05-13

> **Author**: independent ML vision specialist (no prior context with this codebase or the prior agents).
> **Brief**: [`docs/packet_retrospectives/INDEPENDENT_CRITIC_REVIEW_TASK_2026-05-13.md`](../../docs/packet_retrospectives/INDEPENDENT_CRITIC_REVIEW_TASK_2026-05-13.md).
> **Reading discipline**: Pass-1 FACTS-only view of §1–§3 was drafted before opening any `*_OPINIONS_*.md` / `*_CRITIC_REVIEW_*.md` doc; §4 and §5 added after Pass 2.
>
> **Forbidden words in FACTS-citing claims** (per `AGENT_GUIDE.md` Rule 5): succeeds, fails, wins, loses, promotes, deployment-grade, unfortunately, remarkably, shortcut-aligned, lucky, confirmed, refuted. Where an interpretive verdict was unavoidable in §2–§5, the verdict is flagged as **OPINION** in line.

---

## §1 — Current standing assessment

The deployed model is `E2B_TOP_N_STEP3200`; the production anchor for promotion comparisons is `P8A_REFERENCE_STEP5000`; the strongest new candidate from the most recent packet is `T5C_step3500` at promotion-contract rank 3 (`STATE.md` §"Where we stand right now"). Three pillars from `MODEL_GOALS.md` apply: fake recall on target methods, real FPR < 5% on production-relevant cohorts, and robustness across capture-condition axes; promotion requires beating E2B on `dev_fake_macro_recall`, not regressing P8A's chronic-FP behavior, F1 (≥90% lockbox recall at FPR ≤10%), Phase C HDTF cross-substrate, and a named lever.

**Pillar 2 (FPR)** is the cleanest pillar. At dev-cal 5% τ, lockbox FPR is 0.92% (P8A), 2.40% (T5C), 2.19% (T3) — all under 5% (`D4_FACTS_2026-05-12.md` §1, table at line 28-32). The promotion contract's `lockbox_real_fpr` tiebreak is the single column on which P8A is unique-best, and that column has determined contract rank-1 across 13+ R13 packets (`STATE.md` §"Where we stand right now" + `iq_shortcut_deconvolution_program_2026-05-08.md` 2026-05-12 update §"Current stance").

**Pillar 1 (recall)** is mid-trajectory. T5C step3500 catches 1.7× more lockbox fakes than P8A (0.66 vs 0.39 lockbox_fake_recall) at +0.0095 absolute lockbox_real_fpr (`STATE.md`; `iq_shortcut_deconvolution_program_2026-05-08.md` 2026-05-12 update §"Facts"). Headline F1 numbers reach 91-96% on F4-cleaned substrate per memory `project_f1_recall_results_2026-05-04` and `project_t3_slot1_step1500_lockbox_lift_2026-05-09`, but on full v2 lockbox the strongest measured candidate (T5C step3500) sits at 0.66.

**Pillar 3 (robustness)** is where the program is structurally stuck. `D7_FACTS_2026-05-12.md` §5.2 measures a McFadden-R² block-drop on 8 features (6 IQ + substrate_distance + chronic_indicator) predicting per-frame FP at dev-cal 5% τ: IQ block ΔR² = +0.236 (P8A) / +0.314 (T5C); chronic_indicator +0.014 / +0.063; substrate_distance +0.015 / +0.0001. Univariate substrate_distance AUC is 0.575 (P8A) / 0.527 (T5C) — barely above chance. Chronic-vs-non-chronic FPR ratio at the same τ is 4.68× (P8A) / 8.66× (T5C) (`D7_FACTS_2026-05-12.md` §4 table at line 109-111). `D2_FACTS_2026-05-12.md` §3 + `D6_FACTS_2026-05-12.md` §4 quantify a chronic-6 encoder-direction angle drift of 5.06° (P8A) / 7.55° (T3) / 9.54° (T5C) / 10.45° (E2B) below the empirical CLIP-frozen baseline of 79.53°. The dev-to-lockbox structural gap is real (`D3_FACTS_2026-05-12.md` §2: 0/4 ckpts pass substrate-agnostic AUC>0.95 both directions; `D9_FACTS_2026-05-12.md` §5: 0% substrate overlap between training reals and eval reals) but `D10_FACTS_2026-05-12.md` §6 shows the KLIEP substrate-axis is 89.71° from the IQ-PC1 axis — operationally orthogonal to the FPR-driving direction at the CLIP-frozen feature level.

---

## §2 — Proposed GPU experiments

**My OPINION on the binding constraint**: Pillar 3's chronic-cohort FPR concentration is driven by IQ-aligned encoder learning during FT, not by substrate transfer. D7 (IQ block dominates), D2/D6 (chronic-6 IQ angle drift), and D10 (substrate axis ⊥ IQ axis) jointly support this reading. P8A's tiebreak rank-1 is an artifact of being the FT'd ckpt that drifted least (5.06° vs E2B 10.45°) — i.e., P8A learned the IQ shortcut LEAST, which is also why it has the lowest chronic-cohort FPR ratio (4.68× vs 8.66×). The promotion program needs an intervention that explicitly counter-trains the IQ-direction drift while preserving forgery-signal capacity.

### CPU-first prerequisite (Stage A; ~$0; ~3h MPS)

Per AGENT_GUIDE Rule 3, a CPU diagnostic must precede any GPU spend if it can change the GPU call. **T5C L11 atlas inv_mean recompute on step1500/step3500/step3750** (open loop `t5c-classifier-capacity-mechanism`, source `iq_shortcut_deconvolution_program_2026-05-08.md` line 815). If T5C step3500 chronic_6 inv_mean ≥ P8A's, classifier capacity is the lever responsible for T5C's lift and Experiment 3 below is calibrated. If chronic_6 inv_mean is below P8A's, the dor_dev FPR delta (0.06 vs T4's 0.18, n=50) is not via the L11 invariance route and Experiment 3 is uncalibrated — drop or redesign.

### Ranked GPU experiments

**Experiment 1 — Continuous-axis-GRL on 6 IQ axes, T5C base** (~$30-45 single-arm Vertex)
- **Hypothesis**: explicitly counter-training the chronic-6 IQ-direction drift (D2 single-split 75.19° → ≥82°; `D6_FACTS_2026-05-12.md` §4 bootstrap-mean 69.99° → ≥82°) lifts the encoder direction toward orthogonality and reduces chronic-cohort FPR.
- **Falsifier** (open loop `chronic-6-encoder-iq-angle-drift-during-ft` close criterion, `iq_shortcut_deconvolution_program_2026-05-08.md` line 873): post-training chronic-6 angle ≥ 82° AND chronic-cohort FPR drops by ≥ 0.05 absolute at FPR-cal τ. If angle moves but FPR doesn't, drift was incidental and this lever class is cooked for chronic-cohort FPR.
- **One-line argument**: this is the single GPU experiment with a pre-stated falsifier that maps directly onto the dominant FP-driving block measured in D7. T5C is the rank-3 base on the v3-fix scorecard (`STATE.md`); stacking GRL on a base that already holds dor_dev (n=50) FPR at 0.06 isolates the IQ-direction lever from prior confounds.

**Experiment 2 — LoRA on P8A layers 10-11** (~$40-60 single-arm Vertex)
- **Hypothesis**: parameter-efficient FT on the layers where chronic-cohort FPR concentrates (per the L10-L11 divergence noted in memory `project_per_layer_divergence_2026-05-06`) lets the encoder add chronic-6 discrimination capacity without rewriting the substrate-invariance directions that give P8A its 0.92% lockbox FPR. Stage 2 (`STATE.md` §"In flight / running right now") showed that full FT-from-P8A regresses Roy_D real-FPR to 96-100% within 500-2500 steps; LoRA's structural preservation by construction (frozen base + low-rank residual) is the cheapest test of whether the regression was FT-itself or any-FT.
- **Falsifier**: lockbox_real_fpr ≤ 0.025 (within +0.5pp of P8A) AND no chronic identity in {Roy_D, dor_shkedi, PC_Generator, bla_bla_chow} regresses by >5pp absolute FPR vs P8A.
- **One-line argument**: infrastructure is ready (yaml + 10/10 unit tests + smoke per `STATE.md` 2026-05-12 LoRA section); structurally distinct from every prior R13 packet; preserves the only ckpt currently at the contract tiebreak.

**Experiment 3 — T5C × hidden_dim {512, 1024, 2048} sweep** (~$70-100, 3 arms; **gated by CPU**)
- **Hypothesis**: classifier capacity scaling produces monotonic improvement in chronic_6 inv_mean and the dor-cohort FPR delta (T5C step3500 dor_dev FPR 0.06 vs T4 0.18 at n=50).
- **Falsifier**: monotonic improvement in chronic_6 inv_mean with hidden_dim AND lockbox_real_fpr remains ≤ 0.030 across all three arms. If T5C step3500 (1024) is non-monotone vs 512/2048, capacity is not the lever; if 2048 lifts inv_mean but lockbox_real_fpr regresses past 0.030, the lever is structurally trading Pillar 3 for chronic-6 invariance.
- **One-line argument**: ONLY if the CPU-first L11 atlas recompute confirms classifier capacity drives T5C's lift. Without that, the sweep is an uncalibrated $70-100 spend.

**Experiment 4 — B16-scratch + Fourier-band-amp randomization (bands 8-13)** (~$70-100, single-arm; OPTIONAL)
- **Hypothesis**: per memory `project_fourier_band_overlap_2026-05-06`, mid-band Fourier amp randomization is greenlit because high-freq bands 12-13 carry the IQ shortcut (AUC 0.97) but not the manipulation signal (AUC 0.46-0.52). A scratch B16 path tests whether the chronic-6 IQ-direction drift is FT-induced (would not appear in scratch) or representation-fundamental (would appear in scratch as well).
- **Falsifier**: scratch + Fourier achieves chronic-6 IQ-PC1 angle ≥ 82° (`D6_FACTS_2026-05-12.md` empirical 79.53° baseline) AND lockbox_fake_recall ≥ 0.60 AND lockbox_real_fpr ≤ 0.030.
- **One-line argument**: the only experiment that doesn't inherit FT-from-P8A bias; addresses the open question whether the IQ-alignment is fundamental to CLIP-trained representations or specific to the FT recipe. Lower confidence because B16-scratch packets E2B/E3 broke the deeplive ceiling but did not break the viso ceiling (memory `project_e2b_breaks_deeplive_ceiling`, `project_l14_does_not_break_viso_ceiling`).

**Excluded from the ranked list**: data-side direct-capture-real ingestion. D9 documents the substrate-class gap (training 0% direct_teams_capture vs eval 100%) but D7 shows substrate_distance has univariate AUC 0.527-0.575 and partial R² ≤ 0.015 at predicting per-frame FP; D10 shows the substrate axis is 89.71° from IQ-PC1. The structural gap is real, the operational impact on FPR predicted to be small.

---

## §3 — Confidence calibration

| Experiment | Confidence | Justification |
|---|---|---|
| 1. continuous-axis-GRL on T5C | **MEDIUM** | Mechanism matches D2/D6 measurements; falsifier is pre-stated; risk: GRL has not bitten in prior R13 packets (`STATE.md` references P15/P18) and the angle may move without FPR moving. |
| 2. LoRA on P8A L10-11 | **MEDIUM** | Infrastructure ready; structurally distinct from full FT that regressed dor cluster in Stage 2; risk: LoRA-on-CLIP has no prior signal in this codebase. |
| 3. T5C × hidden_dim sweep | **LOW** (HIGH if CPU-first lifts to MEDIUM) | Gated by CPU prerequisite; T5C trajectory is fragile (step1500 lockbox_real_fpr 0.3204 vs step3500 0.0279 per `STATE.md`); single-lever scaling rarely been transformative in R13. |
| 4. B16-scratch + Fourier 8-13 | **LOW** | Greenlit by memory's overlap analysis but scratch paths have specific weaknesses (E3 viso 11.6% per memory `project_l14_does_not_break_viso_ceiling`); longest-shot. |

**Combined confidence: if all 4 succeed, does the project hit its stated goals?** **LOW.** T5C step3500 is at 0.66 lockbox_fake_recall vs the F1 target ≥0.90 at FPR ≤10%; even a successful continuous-axis-GRL or LoRA arm is unlikely to add +24pp from a single intervention. F1 is reachable on F4-cleaned substrate per memory `project_f1_recall_results_2026-05-04` (T3_S1 step2500 hits 92% on F4@10%) but Pillar 3 explicitly requires robustness across the chronic-cohort identities that F4 strips. The deployment IQ-gate (per `MODEL_GOALS.md` §"Resolution / IQ gate") may make most chronic-6 frames below-gate and therefore not production-relevant, but no FACTS doc quantifies the above-gate fraction of chronic-6. **Honest reading**: even the best of these 4 GPU bets gives a candidate that meets Pillar 1 + Pillar 2 + above-IQ-gate Pillar 3 with HIGHER probability than today, but the F1-on-full-lockbox bar is structurally bound by the dev↔lockbox substrate gap (`D3` 0/4, `D9` 0% overlap) which none of the 4 addresses. If F1-on-full-lockbox is required, the project needs a structurally different intervention than any of the four above.

---

## §4 — Where I disagree with prior agents

Three load-bearing disagreements after pass-2 reading.

**4.1 Critic agent's Slot A (lockbox-style data ingestion as HIGHEST priority) — `D1_D5_CRITIC_REVIEW_2026-05-12.md` §4 Slot A.** The critic positions data-side direct-capture-real ingestion as the highest-priority slot on the grounds of D3 (encoder substrate-specific directions) + D9 (training/eval substrate gap) + memory `project_v2_substrate_is_dor_diverse_swap`. **My OPINION**: the critic's Slot A rationale rests on the substrate-distance axis being load-bearing for FPR. D7 (which post-dates the critic main review by ~half a day — landed 2026-05-12 night per `iq_shortcut_deconvolution_program_2026-05-08.md` line 938) measures `substrate_distance` partial R² ≤ 0.015 (`D7_FACTS_2026-05-12.md` §5.2 table at line 127-133) and univariate AUC 0.527 (T5C) / 0.575 (P8A) at predicting per-frame FP (`D7_FACTS_2026-05-12.md` §5.4 table at line 149-157). D10 measures KLIEP substrate-axis vs IQ-PC1 angle = 89.71° (`D10_FACTS_2026-05-12.md` §6.1 table at line 191) — operationally orthogonal at the CLIP-frozen feature level. The substrate gap is mechanistically real (D3, D9) but D7+D10 jointly say it does not predict per-frame FP. The critic's Slot A as HIGHEST is not supported by the full FACTS set landed by 2026-05-12 night; my pass-1 view excludes data-side from the ranked list for this reason. If the critic had D7+D10 in hand, my expectation is the Slot A ranking would drop to MEDIUM-LOW or be replaced.

**4.2 Planning agent's Slot 2 (B16-scratch + 4-lever bundle as HIGHEST priority) — `D1_D5_OPINIONS_2026-05-12.md` §3.1.** Agree with the critic-review's §4 "Drop" position. The slot stacks scratch B16 + Fourier-aug bands 8-13 + continuous-axis-GRL + T3 keep-list — four simultaneous interventions on a training base whose ceiling is documented in memory `project_e2b_breaks_deeplive_ceiling` and `project_l14_does_not_break_viso_ceiling`. The planning agent §1.1 marks the slot MEDIUM but the design violates AGENT_GUIDE Rule 6 single-lever discipline. Even a "win" on the slot would not isolate which of the 4 levers was responsible. My pass-1 Experiment 4 keeps the scratch + Fourier idea but as a single-lever variant at LOW confidence — which is the right shape for a high-uncertainty / high-cost / structurally-distinct probe, not a HIGHEST-priority slot.

**4.3 Planning agent's "P8A is identity-cluster memorizing" (MEDIUM-HIGH confidence) — `D1_D5_OPINIONS_2026-05-12.md` §1 + §1.1 row 4.** Agree with the critic's §3.2 reframe. The post-OPINIONS GCS identity audit (`GCS_IDENTITY_AUDIT_FACTS_2026-05-12.md` §4 table at line 96-105 + §8 obs 4) shows 0/23 identity-name patterns matched across 1646 sample listings and 250 training manifests; the manifest schema is YouTube-ID-anonymized and contains no identity-style field at all (`GCS_IDENTITY_AUDIT_FACTS_2026-05-12.md` §3 + §8 obs 3). Person-level overlap is structurally undetectable from the available data, and the train-overlap audit (`TRAIN_OVERLAP_FACTS_2026-05-12.md` §3.1, §8 obs 4) shows 0 references to the eval bucket across all R13 training yamls. The "memorization" framing is unsupported at the level the question can be answered. The neutral phrasing in `STATE.md` 2026-05-12 late-night retraction ("identity-clustered structure on eval data … mechanism not resolved") is the right tier; the planning agent's MEDIUM-HIGH confidence on the memorization framing should be re-graded LOW per the late-evening retraction.

---

## §5 — Self-correction log

Material changes from my pass-1 view to the post-OPINIONS final view: minor.

- **Experiment 1 (continuous-axis-GRL on T5C) sharpened, not changed**: planning agent's Slot 1 (`D1_D5_OPINIONS_2026-05-12.md` §3.2) and critic's Slot B (`D1_D5_CRITIC_REVIEW_2026-05-12.md` §4 Slot B) both target the same mechanism with falsifiers in the same range as mine (planning agent angle ≥ 81°; critic ≥ 82°; mine ≥ 82° per the open loop close criterion at `iq_shortcut_deconvolution_program_2026-05-08.md` line 873). Convergence across three independent reads on the same FACTS strengthens confidence that this is the right top-of-list bet. Confidence label unchanged at MEDIUM.
- **Experiment 2 (LoRA on P8A) ranking unchanged**: planning agent has it MEDIUM (Slot 4); critic MEDIUM-LOW (Slot D); mine MEDIUM. The critic's "internally inconsistent" objection (LoRA freezes the most substrate-specific encoder per D3) is fair but does not change the cost-benefit — Stage 2's "FT-from-P8A regresses dor cluster" finding leaves LoRA as the cheapest structurally-distinct test of FT-itself-as-binding-constraint.
- **Experiment 3 (T5C × hidden_dim sweep, gated by CPU) unchanged**: neither prior agent ranked this slot directly. Planning agent's Slot 3 (T3 + 1024 classifier) is a different intervention than mine (T5C × hidden_dim scan). Mine remains LOW unless the CPU prerequisite lifts it.
- **Experiment 4 (B16-scratch + single-lever Fourier-aug) unchanged at LOW**: critic's drop of the planning agent's compound Slot 2 is an argument FOR the single-lever variant I named, not against it.
- **No view change on data-side ingestion exclusion**: this is the most material disagreement and is robust to Pass 2 because the supporting evidence (D7 substrate ΔR² ≤ 0.015 + D10 substrate axis 89.71° from IQ axis) post-dated both prior agents' main analyses. The disagreement stands on the FACTS arrival timeline, not on interpretation.
- **§3 combined confidence unchanged at LOW**: T5C step3500 0.66 lockbox_fake_recall vs F1 target ≥0.90 is a +24pp gap; no single GPU lever in either prior agent's plan is sized to close that on full lockbox. The F4-cleaned substrate result already meets F1 (memory `project_f1_recall_results_2026-05-04` T3_S1 step2500 = 92%); the structural bottleneck is the chronic-cohort identities that F4 strips, and whether IQ-gate at deployment makes them not production-relevant.

---

## Appendix — file paths cited

FACTS docs (Pass 1 — read first):
- `analysis/cpu_diagnostics_2026-05-12_d1_p8a_iq_alignment/D1_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d2_encoder_separation/D2_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d3_cross_substrate_probe/D3_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d4_iq_quartile_alignment/D4_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d5_per_identity_iq/D5_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d6_empirical_orthogonality/D6_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d7_fpr_decomposition/D7_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d8_head_retrain_substrate_balanced/D8_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d9_source_substrate_inventory/D9_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d10_training_pool_position/D10_FACTS_2026-05-12.md`
- `analysis/train_overlap_audit_2026-05-12/TRAIN_OVERLAP_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_gcs_identity_audit/GCS_IDENTITY_AUDIT_FACTS_2026-05-12.md`
- `docs/packet_retrospectives/STATE.md`
- `docs/packet_retrospectives/MODEL_GOALS.md`
- `docs/packet_retrospectives/AGENT_GUIDE.md`
- `docs/packet_retrospectives/AGENTS.md`
- `docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md`
- `docs/packet_retrospectives/TIMELINE.md`

OPINIONS docs (Pass 2 — read after Pass-1 §1-§3 drafted):
- `analysis/cpu_diagnostics_2026-05-12_d1_d5_synthesis/D1_D5_OPINIONS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d1_d5_synthesis/D1_D5_CRITIC_REVIEW_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_stage_a/STAGE_A_PLUS_PACKET_PLAN_OPINIONS_2026-05-12.md`
