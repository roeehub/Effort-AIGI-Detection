# Stage A + 3-slot GPU plan — OPINIONS (2026-05-12)

> **Status: OPINION, not RECORD.** Reviewer is invited to disagree. Every load-bearing claim in this doc cites a FACT under one of the sibling FACTS docs (see §0 Reading order). Forbidden-words constraint does NOT apply to this file.
>
> **Required disclaimer per `docs/packet_retrospectives/AGENT_GUIDE.md` Rule 5**: past framings in this project have been demonstrably wrong (e.g. P14 `face_scale_jitter@0.50` was framed as a "load-bearing composable lever" in memory `project_face_scale_jitter_load_bearing.md` and refuted as composable by T6/T7 verdict 2026-05-12; the T4 "substrate-overfit" framing was partially retracted via A1/A2/A3 2026-05-11). This OPINIONS doc adds another candidate framing to that ledger; treat it skeptically.
>
> **Authoring**: drafted 2026-05-12 by the same agent who ran Stage A and CPU Jobs 1-3, then proposed the 3 GPU slots. Single-agent session; FACTS-doc pass-1 / OPINIONS pass-2 independence is NOT guaranteed in this session — treat the synthesis below with extra skepticism. An independent agent picking this up cold should read the FACTS docs (§0) FIRST and form their own view before reading §1 onward.

---

## 0. Reading order for an independent fresh agent

Form your own view from the FACTS first. Then optionally compare to §1+ here.

**FACTS docs (read in this order — pass 1)**:

1. [`../cpu_diagnostics_2026-05-11_t67_t5c_probe/CKPT_SCORING_FACTS_2026-05-12.md`](../cpu_diagnostics_2026-05-11_t67_t5c_probe/CKPT_SCORING_FACTS_2026-05-12.md) — T5C/T6 candidate ckpt scoring on may6 / dor / chronic_6 / Roy_D cohorts. The dispositive data for "is anything better than P8A".
2. [`../cpu_diagnostics_2026-05-11_t67_t5c_probe/INV_MEAN_FACTS_2026-05-12.md`](../cpu_diagnostics_2026-05-11_t67_t5c_probe/INV_MEAN_FACTS_2026-05-12.md) — L11 atlas inv_mean per substrate slice for 7 ckpts (including the T5C chronic_6 partial-recovery finding).
3. [`STAGE_A_FACTS_2026-05-12.md`](STAGE_A_FACTS_2026-05-12.md) — Stage A 3 probes (per-IQ-bin policy, ensemble policy grid, disagreement audit) on the 13,636-frame contract matrix.
4. [`JOB1_L11_DISTANCE_FACTS_2026-05-12.md`](JOB1_L11_DISTANCE_FACTS_2026-05-12.md) — per-frame L2 + cosine distance from P8A's L11 features on the 800-frame triptych.
5. [`JOB2_OVERFIRE_FACTS_2026-05-12.md`](JOB2_OVERFIRE_FACTS_2026-05-12.md) — T5C step3500 vs P8A real-side overfire identity concentration (95.4% on top 5 identities).
6. [`JOB3_INFRA_AND_RULE1_AUDIT_FACTS_2026-05-12.md`](JOB3_INFRA_AND_RULE1_AUDIT_FACTS_2026-05-12.md) — codebase infra inventory + AGENT_GUIDE Rule 1 grep audit confirming the 3 proposed levers are untested.

**Project context (read AFTER forming an independent view)**:

7. [`../../docs/packet_retrospectives/MODEL_GOALS.md`](../../docs/packet_retrospectives/MODEL_GOALS.md) — the NO ENSEMBLE rule + three pillars + B16-only constraint + E2B/P8A roles.
8. [`../../docs/packet_retrospectives/AGENT_GUIDE.md`](../../docs/packet_retrospectives/AGENT_GUIDE.md) — the 6-rule contract for packet proposals.

**OPINION docs (read LAST — this file + cited memory entries)**:

9. This file (§1 onward).
10. Memory entries cited inline (see §1.2 for the full memory citation list).

---

## 1. Synthesis interpretation (one paragraph)

Across the 13+ R13 packets this is the first time a candidate ckpt **measurably moves chronic_6 invariance in the direction of P8A** while preserving most of T5C step3500's catch lift — T5C's bigger online classifier closed +0.0094 of T4's −0.0315 chronic_6 inv_mean regression vs P8A (`INV_MEAN_FACTS_2026-05-12.md` §4). The residual gap to P8A is −0.0221 absolute. The per-frame data tells us the residual gap is concentrated on 5 identities (dor_shkedi, bla_bla_chow, Roy_D, xiang, dor — `JOB2_OVERFIRE_FACTS_2026-05-12.md` §5) and that the L11 representation difference is small on those identities (L2 16-23 vs 21-28 on non-chronic — `JOB1_L11_DISTANCE_FACTS_2026-05-12.md` §2). My read: this is the first time a structurally-distinct lever class has produced measurable encoder-level invariance lift without breaking head-side capacity — the lever class is alive. The remaining gap is closeable by either (a) parameter-efficient FT that cannot drift the base, (b) explicit L11 anchor pulling features back to P8A's on the 5 binding identities, or (c) explicit output-space anchor pulling the score distribution back to P8A's on a broader real cohort. Three slots, three structurally distinct hypotheses. Confidence: LOW that any single slot solves it cleanly; MEDIUM that one of them produces measurable lift; HIGH that all three together discriminate which lever class is the real binding constraint.

### 1.1 Confidence-tiered claims

| Claim | Confidence | Citation |
|---|---|---|
| T5C step3500 partially recovers T4's chronic_6 inv_mean regression by +0.0094 absolute | HIGH | `INV_MEAN_FACTS_2026-05-12.md` §3.3 + §4 |
| 95.4% of T5C's real-side overfires concentrate on 5 identities | HIGH | `JOB2_OVERFIRE_FACTS_2026-05-12.md` §5 |
| T5C step3500's L11 drift from P8A is uniform across real cohorts (cosdist median ≈ 0.50 on chronic and healthy alike) | HIGH | `JOB1_L11_DISTANCE_FACTS_2026-05-12.md` §3 |
| Top-10 L11-drifted frames are all FAKES (the drift is concentrated where the catch lives) | HIGH | `JOB1_L11_DISTANCE_FACTS_2026-05-12.md` §6 |
| Specialist routing is the cheapest single-model path | **REFUTED** by `MODEL_GOALS.md` "Single model — NO ENSEMBLE" rule (specialist routing requires multi-forward-pass routing infrastructure, classed as ensemble) | `MODEL_GOALS.md` §"Single model — NO ENSEMBLE" |
| The encoder pair P8A + T5C step3500 carries complementary information sufficient to break the chronic_6 ceiling at $0 inference cost | HIGH | `STAGE_A_FACTS_2026-05-12.md` §2.4 — but inadmissible per NO ENSEMBLE rule |
| L11 feature anchor (Slot 1) closes the residual chronic_6 inv_mean gap | LOW-MEDIUM | mechanism is plausible per `JOB1_L11_DISTANCE_FACTS_2026-05-12.md` finding + memory `project_stage2_all_levers_regress_p8a_2026-05-09.md` flagging this as untested option 3a |
| Output distillation (Slot 2) narrows the real-side score width gap | MEDIUM | mechanism is plausible per `JOB2_OVERFIRE_FACTS_2026-05-12.md` §8 score-width finding |
| T3_SLOT1 init + 1024 classifier (Slot 3) inherits T3's smaller encoder drift with T5C's catch | LOW | speculation; compose-of-wins has not been empirically tested |
| FT itself is the binding constraint (and all 3 slots will fail) | LOW-MEDIUM | partially supported by `project_stage2_all_levers_regress_p8a_2026-05-09.md`; partially refuted by T5C's +0.0094 chronic_6 lift |

### 1.2 Memory citations

- `project_stage2_all_levers_regress_p8a_2026-05-09` — flags Slot 1 and Slot 2 lever classes as **explicitly untested** ("hard-output-preservation losses on a reference cohort, multi-objective training with Pillar-2 explicit constraint, frozen-encoder regimes other than head-only retrain"). Directly names "L11 anchor-loss FT (option 3a)" as a candidate intervention.
- `project_t4_substrate_overfit_inv_mean_misleading_2026-05-11` (AMENDED) — partial retraction of the substrate-overfit framing; T4's failure is at trained head + chronic_6 invariance, not at the encoder representation.
- `project_t6_t7_t5c_scorecard_2026-05-12` — current scorecard verdict + T5C step3500 rank-3 placement.
- `project_t5c_classifier_capacity_mechanism` (open loop) — close criterion was the L11 atlas inv_mean recompute; **CLOSED** by this session's `INV_MEAN_FACTS_2026-05-12.md` finding of partial recovery (+0.0094 vs T4 baseline; gap to P8A −0.0221).
- `project_p8a_breakthrough` — P8A baseline mechanism (visual.proj + ln_post + MLP-SVD unfreezing).
- `project_xinhe_may6_falseflag_2026-05-06` — may6 production-drift cohort; cited by `CKPT_SCORING_FACTS_2026-05-12.md` §4.1.
- `project_canary_below_production_resolution_2026-05-08` — IQ-gate deployment-time policy; pillar 3 robustness.

---

## 2. Three GPU slots — single-model bets

These three slots are STRUCTURALLY DISTINCT levers against the same observed gap. If one wins, we know the mechanism. If all fail in the same way, FT itself is the binding constraint and the next packet pivots to LoRA-only or scratch-from-CLIP retraining.

### 2.1 Slot 1 — T5C + L11 feature anchor on 5-identity chronic cohort

**Hypothesis**: T5C's chronic-identity drift is at the L11 representation level on 5 identities (95.4% concentration per `JOB2_OVERFIRE_FACTS_2026-05-12.md` §5). Pulling the model's L11 CLS features toward P8A's L11 CLS features on those 5 identities should preserve P8A's chronic invariance while keeping T5C's head capacity for fake catch.

**Configuration delta vs T5C step3500**:
- Same FT-from-P8A (`gcs_base_checkpoint` = P8A_step5000 = 9lmvb5b4)
- Same multi-axis GRL with hidden_dim=1024, bottleneck_dim=128
- Same data sources (T3 SLOT1 keep-list, visomaster_enhanced + visomaster_teams_enhanced enabled, etc.)
- **NEW**: `l11_anchor:` block in yaml
  - anchor cohort: real frames from the top-5 identities (dor_shkedi, bla_bla_chow, Roy_D, xiang, dor) — approximately 3,000-5,000 frames
  - target: pre-computed P8A L11 CLS features per anchor frame, stored as parquet (one-time precompute, ~30 min CPU)
  - loss: per-step batch (samples_per_step=16) MSE between model's L11 CLS feature and target feature
  - λ schedule: linear warmup 0 → 0.5 over first 500 steps, flat at 0.5 thereafter
- New code: `loss/l11_anchor_loss.py` + a trainer hook similar to `AnchorAwarePenalty` (already-existing pattern per `JOB3_INFRA_AND_RULE1_AUDIT_FACTS_2026-05-12.md` §1, §4.1)

**What would falsify**: chronic_6 inv_mean ≤ +0.0224 (no lift over T5C step3500). Per `INV_MEAN_FACTS_2026-05-12.md` §3.3 that's the baseline.

**What would confirm (close criterion)**:
- chronic_6 inv_mean ≥ +0.035 (more than halfway from T5C step3500 to P8A's +0.0445)
- dor_real_lockbox FPR ≤ 1.0% at FPR-calibrated τ (vs T5C step3500's 2.4% per `STAGE_A_FACTS_2026-05-12.md` §4.2 + `CKPT_SCORING_FACTS_2026-05-12.md` §2 DOR_REAL_LOCKBOX at τ=0.9)
- teams_fake_all_lockbox recall ≥ 0.55 (preserve most of T5C's +37pp catch gain over P8A's 0.306 — see `STAGE_A_FACTS_2026-05-12.md` §2.4)
- HDTF teams_dev no regression > 5pp vs E2B (per `MODEL_GOALS.md` Pillar 3 promotion bar)

**Cost**: ~$50 Vertex (us-east1, A100×1, 5000 steps + ~$5 P8A reference cache pre-compute).

**Seed**: 9601.

### 2.2 Slot 2 — T5C + per-frame output distillation anchor on broad real cohort

**Hypothesis**: T5C's real-side score distribution is uniformly ~15× wider than P8A's at the median (across chronic AND healthy reals per `JOB2_OVERFIRE_FACTS_2026-05-12.md` §8). A per-frame output anchor that pulls T5C's `prob_fake` toward P8A's `prob_fake` on each anchor real frame should narrow the entire real distribution. This is a head-side intervention independent of L11 feature drift.

**Configuration delta vs T5C step3500**:
- Same FT, GRL, data as Slot 1
- **NEW**: extension to `AnchorAwarePenalty` (existing per `JOB3_INFRA_AND_RULE1_AUDIT_FACTS_2026-05-12.md` §1, §4.2) — add `mode: per_frame_distillation` switch
  - anchor cohort: ~5,000 real frames covering chronic-5 (~3,000) + healthy reals (~2,000)
  - target: pre-computed P8A `prob_fake` per anchor frame, stored as parquet
  - loss: per-step batch (samples_per_step=16) BCE(model_prob_fake, target_prob_fake) on anchor batch
  - λ schedule: linear warmup 0 → 0.3 over first 500 steps, flat at 0.3
- New code: ~80 lines extending `loss/anchor_aware_penalty.py` (existing class)

**What would falsify**: T5C's lockbox_real_fpr stays ≥ 0.02 at FPR-calibrated τ, OR fake-recall drops to P8A levels (anchor too aggressive, killed the catch).

**What would confirm (close criterion)**:
- lockbox_real_fpr ≤ 0.015 at FPR-calibrated τ (matches or beats P8A's 0.0092 from `STAGE_A_FACTS_2026-05-12.md` §4)
- lockbox_fake_recall ≥ 0.50 (preserves majority of T5C's +20pp gain over P8A)
- visomaster_enh_macro_dev recall ≥ 0.12 (matches T5C step3500's 0.102)
- HDTF teams_dev no regression > 5pp vs E2B

**Cost**: ~$50 Vertex (us-west4, A100×1, 5000 steps + reference cache pre-compute).

**Seed**: 9602.

### 2.3 Slot 3 — T3_SLOT1 base + 1024 classifier (compose two known wins)

**Hypothesis**: T3_SLOT1 step1500's keep-list data filter produces ~half the L11 drift of T5C (L2 9.07 vs T5C 17.40 on lockbox real per `JOB1_L11_DISTANCE_FACTS_2026-05-12.md` §2) and is the closest non-P8A ckpt to P8A in feature space. T5C's wider classifier closes 30% of T4's chronic_6 inv_mean regression. The composition (T3_SLOT1 init + T5C-style 1024 classifier) hasn't been tested per `JOB3_INFRA_AND_RULE1_AUDIT_FACTS_2026-05-12.md` §5.3.

**Configuration delta vs T5C step3500**:
- Init: `gcs_base_checkpoint` = T3_SLOT1_PERIODIC_STEP1500 (instead of P8A_step5000)
- All other recipe levers identical to T5C step3500 (multi-axis GRL hidden_dim=1024, T3 data filter, etc.)
- No new code; pure yaml fork

**What would falsify**: lockbox_fake_recall ≤ T3_SLOT1's 0.7747 AND chronic_6 inv_mean stays at T5C step3500's +0.0224 — then the composition doesn't add value beyond either lever alone.

**What would confirm (close criterion)**:
- lockbox_fake_recall ≥ 0.55 AND chronic_6 inv_mean ≥ +0.030 (above T5C step3500, below P8A)
- dor_real_lockbox FPR ≤ 1.5% at FPR-calibrated τ
- HDTF teams_dev no regression > 5pp vs E2B

**Cost**: ~$30-40 Vertex (us-central1, A100×1, 5000 steps from T3_SLOT1 init).

**Seed**: 9603.

---

## 3. Shared scorecard after training

After all 3 training jobs complete, run ONE combined promotion-contract scorecard on 6 ckpts: 3 candidates + 3 anchors (P8A_step5000, E2B_step3200, T5C_step3500).

- Suite manifest: `target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml` (29-suite, same as T5C scorecard)
- Policy: v3-fix (target_real_fpr=0.07, target_stress_fpr=0.10, target_fake_recall_min=0.30)
- Cost: ~$25-30 Vertex

**Combined budget**: 3 × $40 + $30 ≈ $150 across the full lineage.

---

## 4. Decision tree after scorecard

| Slot 1 close? | Slot 2 close? | Slot 3 close? | Reading | Next action |
|---|---|---|---|---|
| ✓ | — | — | L11 anchor is the lever; representation-level anchoring works | Iterate Slot 1: different anchor cohorts, λ values, anchor layer (L9 vs L11) |
| — | ✓ | — | Output anchor is the lever; score-distribution anchoring works | Iterate Slot 2: broader cohort, λ schedule, KL vs BCE |
| — | — | ✓ | Init-base composition works; T3_SLOT1's data filter scales head capacity | Iterate Slot 3: hidden_dim sweep on T3_SLOT1 base |
| ≥2 close | | | Lever class is robust to multiple interventions | Pick cheapest deployable; recipe finalized |
| All fail same way | | | FT itself is the binding constraint regardless of intervention class | Pivot to LoRA-only (frozen base, adapter-only) or scratch-from-CLIP retraining |
| All fail differently | | | Each mechanism captures partial truth | Combine top-2 levers in a 4th slot |

---

## 5. AGENT_GUIDE rule compliance

Per `AGENT_GUIDE.md`:

- **Rule 1 (Validate-before-suggest)**: ✓ `JOB3_INFRA_AND_RULE1_AUDIT_FACTS_2026-05-12.md` §5 documents the grep results for each of the 3 levers; none have been previously tested in R13.
- **Rule 2 (Read failure modes)**: ✓ `JOB3_INFRA_AND_RULE1_AUDIT_FACTS_2026-05-12.md` §5.4 cites the memory entry that explicitly flags L11 anchor + hard-output-preservation as untested. The 3 proposed levers do NOT re-launch any previously-failed lever class (no jitter, no new GRL axes, no pair_rank, no group_dro).
- **Rule 3 (CPU-first-then-GPU)**: ✓ Stage A + 4 CPU jobs ran before any GPU spend proposal. Each slot has explicit falsification criteria (§2.1-§2.3 above).
- **Rule 4 (Viewer integration)**: Planned — Stage A outputs land at `analysis/cpu_diagnostics_2026-05-12_stage_a/outputs/` in standard CSV format; the new candidate ckpts (after Slots 1-3 land) will be registered as run entries in `viewer/model_dashboard_runs.yaml` per the eval folder template pattern. **NOT YET DONE in this session — open work item.**
- **Rule 5 (FACT vs OPINION split)**: ✓ This file is the OPINIONS doc. FACTS are in the 5 sibling FACTS docs listed in §0. The split was completed 2026-05-12 evening; prior to this commit a single mixed doc existed.

---

## 6. Self-correction log

| Earlier framing | Why retracted | Where the correction lives now |
|---|---|---|
| "Specialist routing is the cheapest path to T5C-strength + P8A-invariance" (from this session's earlier chat turns) | Refuted by `MODEL_GOALS.md` "Single model — NO ENSEMBLE" hard rule. Specialist routing is an ensemble even if conceptually simple. | This doc §1.1 (REFUTED row); the Stage A finding is preserved as research evidence but not as a deployment path. |
| "T5C step3500 is rank 3 by contract lex-policy and partially supports the classifier-capacity hypothesis" (from `t6_t7_t5c_scorecard_eval_2026-05-12/AGENT_PROPOSAL_2026-05-12.md` §3.1) | Partially refuted: the L11 atlas recompute (this session, `INV_MEAN_FACTS_2026-05-12.md` §3.3) confirms partial chronic_6 recovery (+0.0094 vs T4) but NOT full restoration of P8A invariance (gap −0.0221 absolute). The mechanism is alive but partial. | This doc §1 + `INV_MEAN_FACTS_2026-05-12.md` §4. |
| "STAGE_A_PLUS_PACKET_PLAN_2026-05-12.md is FACTS+OPINION split" (earlier this session) | The original mixed doc did not separate FACTS from OPINION. This file (OPINIONS) was renamed from that mixed doc and rewritten 2026-05-12 evening to cite the per-job FACTS docs rather than restate the numbers. | Reading order in §0 is the canonical entry point. |

---

## 7. What I want from the user

1. **Authorize** the 3 GPU slots (Slot 1 / 2 / 3 in §2)
2. **Confirm** the ~$150 budget envelope
3. **Confirm** new code goes in `loss/l11_anchor_loss.py` + extension to `loss/anchor_aware_penalty.py` (no feature branch needed; main-line additive change)
4. **Confirm** the anchor cohort curation — top-5 identities or include healthy + chronic_6 broader set

Once authorized, the next agent steps are:
1. Write `loss/l11_anchor_loss.py` + extension to `loss/anchor_aware_penalty.py` + unit tests under `tests/`
2. Curate the anchor cohort (extending `analysis/teams_pool_rescore.py`'s pool list) and run the P8A reference pre-compute script
3. Author 3 yamls: `R13_T8_T5C_L11_ANCHOR_2026-05-12.yaml`, `R13_T9_T5C_OUTPUT_DISTILL_2026-05-12.yaml`, `R13_T10_T3SLOT1_BIG_CLASSIFIER_2026-05-12.yaml`
4. Smoke-test each (single-step, 0 GPU)
5. Image rebuild via `./dev.sh build-prod -y` (auto-bumps VERSION per memory `reference_image_rebuild.md`)
6. Parallel Vertex launch across us-east1 / us-west4 / us-central1 (per `CLAUDE.md` US-region preference)
7. Monitor until all `RUNNING`, then yield
