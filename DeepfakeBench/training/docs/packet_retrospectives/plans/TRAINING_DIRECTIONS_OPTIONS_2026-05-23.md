# Training Directions — Options for Decision

**Date**: 2026-05-23 (afternoon)
**Author**: working agent in session with user (Roee)
**Status**: draft, awaiting independent reviewer
**Decision asked**: operator picks a 1-2 week training plan (~$140-500 GPU) from a structured options menu; reviewer is asked to verify the option-set is correctly framed, complete, and ranked

This document is a successor to the morning's `STRUCTURAL_REFRAME_PROPOSAL_2026-05-23.md` and incorporates the corrections from `STRUCTURAL_REFRAME_REVIEW_2026-05-23.md`. Both should be read for context; this document is the operator-facing decision menu.

---

## §0 — Reviewer onboarding (read first if you are the next agent)

You are reviewing this document cold. The user (Roee) wants an independent pass on the option-set: is it correctly framed, complete, ranked sensibly, and free of the errors the prior review caught? Push back; don't rubber-stamp.

### Reading order

1. **This document end-to-end** (~20 min)
2. **The two prior docs** (~30 min):
   - `docs/packet_retrospectives/plans/STRUCTURAL_REFRAME_PROPOSAL_2026-05-23.md` — original proposal (contains errors the review caught)
   - `docs/packet_retrospectives/plans/STRUCTURAL_REFRAME_REVIEW_2026-05-23.md` — the adversarial review (load-bearing for understanding what corrections landed in this document)
3. **Authoritative project docs** (~30 min):
   - `docs/packet_retrospectives/MODEL_GOALS.md` — hard constraints (no ensemble, no per-substrate τ, target methods, FPR budget)
   - `docs/packet_retrospectives/AGENTS.md` — wiki protocol (FACTS/OPINIONS split, threads-win rule, pre-proposal validation checklist)
   - `docs/packet_retrospectives/STATE.md` — rolling current-state snapshot (last ~200 lines is enough)
   - `docs/packet_retrospectives/TIMELINE.md` — last 10-15 entries for recent arc
4. **The binding measurement today** (~10 min):
   - `analysis/team_identity_deploy_readout_expanded_2026-05-23/RESULTS_FACTS_2026-05-23.md` + per-human / per-cohort CSVs
   - `analysis/xinhe_may6_t5c_revisit_2026-05-23/RESULTS_FACTS_2026-05-23.md`
   - `analysis/frozen_clip_team_identity_baseline_2026-05-23/RESULTS_FACTS_2026-05-23.md` (if landed; was extracting at draft time)

### Where evidence lives

| What | Where |
|---|---|
| Per-packet retros (one per training run/batch) | `docs/packet_retrospectives/packets/*.md` — read by name (P22, T3, T5C_TRIPLE_2026-05-19, U_SLOTS_2026-05-13, FACE_POOL_2026-05-22, etc.) |
| Cross-cutting topic threads | `docs/packet_retrospectives/threads/*.md` — read by topic (iq_shortcut_deconvolution_program, viso_bucket_gap, processing_signature_shortcut, clean_teams_identity_pairing, etc.) |
| Open issue inventory (mechanically generated) | `docs/packet_retrospectives/OPEN_LOOPS.md` |
| Per-experiment CPU/MPS analyses (FACTS + OPINIONS docs) | `analysis/<topic>_<date>/RESULTS_FACTS_*.md` + `AGENT_PROPOSAL_*.md` |
| Auto-memory (load-bearing across sessions) | `/Users/roeedar/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/*.md` — index at `MEMORY.md` |
| Frozen plans | `docs/packet_retrospectives/plans/*.md` — including MASTER_PLAN_2026-04-29.md |
| Historical session handoffs (frozen archive — don't author new) | `docs/relaunch_handoffs/*.md` |

### How to read a packet retro (the FACTS/OPINIONS discipline)

Per `AGENTS.md` §"Eval-folder authoring contract":
- `RESULTS_FACTS_<date>.md` files contain **numbers + tables only**, no interpretation. Banned words: succeeds, fails, wins, loses, promotes, deployment-grade. Use mechanical pass/fail against pre-stated bars.
- `AGENT_PROPOSAL_<date>.md` files contain a single agent's interpretation, with a mandatory `Self-correction log` section.
- When forming your own view, read FACTS docs first. Read OPINIONS docs second, with extra skepticism.
- When memory and threads disagree, **threads win**. Memory is a fast index that lags; threads are deliberated synthesis.

### Key memories that explain the journey (load these into context)

These are the "if you read nothing else, read these" set:

1. `feedback_cpu_jobs_standing_greenlight.md` — your CPU verification work doesn't need user approval
2. `project_team_identities_multi_labeled_2026-05-23.md` — the 5-team-human cohort + Mac/Windows split (Roy_D=Roee-on-Mac=out-of-scope; team_sanity_may5 is the canonical real cohort)
3. `project_production_is_t5c_not_e2b_2026-05-23.md` — T5C is currently deployed (NOT E2B as previously documented)
4. `project_promotion_contract.md` — the authoritative deployment contract definition
5. `project_face_pool_scorecard_pareto_2026-05-22.md` — the $0 face-pool inference finding (now partially refuted by today's Xinhe-may6 catastrophe; flag this discrepancy in your review)
6. `project_band_shortcut_ood_hypothesis_2026-05-16.md` — anchor_aware mechanism (the one win)
7. `project_p22_cpu_followups_reframe_2026-05-02.md` — P22 step1k is the robust P22 ckpt; never used as a downstream FT base
8. `project_lora_l10_l11_chronic_fp_hard_coupled_2026-05-15.md` — LoRA viso lift hard-coupled to 24% lockbox FPR; refuted at moderate ranks
9. `project_image_quality_shortcut.md` — the IQ shortcut documented at scale
10. `project_per_ckpt_substrate_axis_2026-05-22.md` — Probe 1 finding (FT'd encoders rotate 85-88° from frozen prior)
11. `project_job12_ensemble_ceiling_2026-05-04.md` — empirical evidence that label-free ensembles can't break the viso ceiling (backs MODEL_GOALS's NO-ENSEMBLE rule)

### Verification asks for the reviewer

Before forming your own view, verify:

1. **Are the options framed correctly?** Spot-check 2-3 of my cost / wall / confidence estimates by reading the relevant memory or packet retro and seeing if my numbers reproduce.
2. **Is the option-set complete?** Are there structural levers I've missed? The prior reviewer caught 5 missing alternatives (output-preservation aux loss, P22 step1k as base, training-data subsetting, substrate-pair contrastive pretrain, smaller encoder). Did I integrate all 5? Are there more I'm still missing?
3. **Are the confidence estimates honest?** I list P(materially better) in the 15-30% range for most options. Is that calibrated or am I being optimistic/pessimistic?
4. **Are the "don't do" exclusions correct?** I list Roy_D anchor pool expansion, B32 smaller encoder, LoRA placement variants as lower-EV. Verify by reading the relevant memories.
5. **Does the proposed ranking match the evidence?** I rank B.I.1+B.I.2 (output-preservation aux loss + P22 step1k base) as week-1 top pick. Is that right?

### What the reviewer should produce

Write your review as a new file:
`docs/packet_retrospectives/plans/TRAINING_DIRECTIONS_REVIEW_2026-05-23.md`

Structure suggestion:
1. Executive verdict (do you agree with the ranked recommendation?)
2. Option-set completeness review (any missing levers?)
3. Per-option critique (cost / wall / confidence corrections)
4. Ranking critique (different week-1 pick?)
5. Risks under-weighted
6. Direct answers to the 3 decisions the operator must make (§7 of this document)
7. Self-correction log

Length: 1500-3000 words. The user wants pushback, not validation.

---

## §1 — State of the program (top-level)

After ~6 weeks and ~25-30 GPU training runs (~$1,500-2,500 cumulative spend), the deploy-relevant picture today:

**No ckpt clears the team-identity bar at mode B** (per-human FPR ≤5% AND fake-recall ≥50% on the 5-human team cohort: Roee on Windows + dor + Noyn Sharker + Xiang + Xinhe). Per `analysis/team_identity_deploy_readout_expanded_2026-05-23/`:

| Ckpt | Max real FPR @ mode B | Min fake recall @ mode B | Notes |
|---|---:|---:|---|
| P8A | 0.016 | 0.487 (Xinhe) | Mode A is the only passing mode (worst real FPR 0.040 / worst recall 0.633) |
| E2B | 0.036 | 0.473 (dor) | Marginal on both floors |
| **T5C (current production)** | **0.052 (dor)** | **0.247 (Xinhe)** | Over real-FPR floor on dor; well under fake-recall floor on Xinhe |
| Slot A v2 CLS | 0.024 | 0.293 (Xinhe) | Cleanest real-FPR among trained, but Xinhe-fake-recall weak |
| Slot A v2 face-pool | 0.003 | 0.252 (dor) | Cleanest aggregate real-FPR — **BUT catastrophic on Xinhe-may6 (91% FPR @ τ=0.5); cannot ship** |

The Xinhe-may6 reproduction probe (`analysis/xinhe_may6_t5c_revisit_2026-05-23/`):
- E2B (May production) reproduced bytewise-exactly at 57.6% FPR @ τ=0.5 / 31.5% @ mode B — the original 2026-05-06 catastrophic claim was correct
- T5C inherits ~10× less of E2B's may6 failure: 17.4% @ τ=0.5 / 3.3% @ mode B. Above the 5% floor at mode B but not catastrophic
- Slot A v2 face-pool is **catastrophically broken** on Xinhe-may6 (91.3% @ τ=0.5 / 85.9% @ mode A) — the $0 face-pool baseline that has been treated as a Pareto improvement for ~24 hours does NOT generalize to a real production cohort of a teammate

**Diagnosis status** (per the morning's proposal + adversarial review):
- The "structural ceiling" framing is **directionally correct** (Probe 1's per-ckpt-axis rotation finding is load-bearing; ~20 single-lever refutations are real evidence)
- But the framing was **overstated** — three structurally-distinct levers inside the current paradigm haven't been tried (output-preservation aux loss, P22 step1k as FT base, KLIEP per-frame training subset)
- And cross-ckpt τ comparison is **unfair to candidates with shifted score distributions** — per-ckpt τ-recalibration is a prerequisite ($0, 1hr CPU) that hasn't been done
- 100% YouTube training reals framing was wrong: D9 §5 actually shows 7.8% teams-v2 + 55.3% HDTF + 37.0% QCLIPS, all YouTube-derived but in three distinct capture pipelines. This **strengthens** the data argument and gives any IRM variant more environment-partition structure

---

## §1.5 — Frozen-CLIP probe results (LANDED for Option A; Option B in flight)

Probe complete at `analysis/frozen_clip_team_identity_baseline_2026-05-23/`. FACTS-only summary of new findings (full readout in that folder's RESULTS_FACTS doc).

### 1.5.1 Joint-calibrated comparison at team-real-FPR = 5%

Each head's τ set so fraction(team-real ≥ τ) = 0.05 on the 1,821 deploy-relevant team-real frames. Fake recall per team-human on the fake-attack cohorts. Source: `outputs/joint_calibrated_summary.csv`.

| Head | τ | dor recall | Xinhe recall | Xiang recall | **min recall** | mean recall |
|---|---:|---:|---:|---:|---:|---:|
| **P8A** (FT'd) | 0.239 | 0.870 | 0.786 | 0.936 | **0.786** | 0.864 |
| E2B (FT'd) | 0.455 | 0.700 | 0.813 | 0.945 | 0.700 | 0.819 |
| Slot A v2 face-pool (FT'd) | 0.653 | 0.562 | 0.905 | 0.988 | 0.562 | 0.818 |
| Slot A v2 CLS (FT'd) | 0.461 | 0.622 | 0.546 | 0.964 | 0.546 | 0.710 |
| **T5C (current production)** | 0.657 | 0.759 | 0.379 | 0.948 | **0.379** | 0.695 |
| MLP_DEV_LB (frozen-CLIP, lockbox-cheat) | 0.913 | 0.527 | 0.205 | 0.465 | 0.205 | 0.399 |
| MLP_DEV (frozen-CLIP, fair) | 0.975 | 0.292 | 0.160 | 0.159 | 0.159 | 0.204 |
| LR_DEV (frozen-CLIP, fair) | 0.993 | 0.289 | 0.055 | 0.197 | 0.055 | 0.180 |

### 1.5.2 Cross-check vs D8

D8's `P8A_dev_unweighted` (LR on trained-P8A encoder L11 features) had DEV→LOCKBOX transfer AUC **0.587**.
This probe's `LR_DEV` (LR on frozen-CLIP L11 features, identical sample protocol) has DEV→LOCKBOX transfer AUC **0.783**.
**Delta: +0.196 AUC on the substrate-transfer task — frozen-CLIP features transfer better than trained-P8A features on the same train/test split.**

D8's strongest weighted variant (Estimator-C-KLIEP) was 0.679. Frozen-CLIP unweighted exceeds it by +0.10.

### 1.5.3 Per-cohort decomposition (mode B τ=0.78, frozen-CLIP heads)

Source: `analysis/frozen_clip_team_identity_baseline_2026-05-23/RESULTS_FACTS_2026-05-23.md` §5.

- **Roee-Windows** (330 frames `tester tester` + 30 `team_may5__Roee`): **0% FPR across every frozen-CLIP head at mode B.** Same as the FT'd ckpts.
- **team_may5__Dor** (30 frames): 0% FPR across every head.
- **dor_evening** (n=150): LR_DEV 29.3%, LR_DEV_LB 22.7%, MLP_DEV_LB 18.7%
- **dor_morning** (n=150): LR_DEV 72.7%, LR_DEV_LB 63.3%, MLP_DEV_LB 56.7%
- **dor_shkedi__s16** (n=147 chronic): LR_DEV 89.1%, LR_DEV_LB 20.4%, MLP_DEV_LB 4.8% — adding lockbox to training collapses this chronic-FP

### 1.5.4 Gap of FT'd ckpts over best frozen-CLIP head (joint-calibrated, min per-human recall)

| Comparison | min Δ |
|---|---:|
| P8A − MLP_DEV_LB (Option A, lockbox-cheat) | +0.581 |
| P8A − MLP_DEV (Option A, fair) | +0.626 |
| **P8A − MLP_OPTB_DEV (Option B + DEV, best Option B)** | **+0.749** |
| **P8A − LR_OPTB (Option B alone)** | **+0.786** |
| T5C − MLP_DEV_LB | +0.174 |
| T5C − MLP_OPTB_DEV | +0.342 |

### 1.5.5 Option B results (LANDED 2026-05-23 16:38)

Option B = 6,000 frames sampled from the training corpus the FT'd ckpts saw (FF++ / deep-live-cam / simswap / AVSpeech / celeb_synthesis variants, stratified by method, seed=42). Source: `analysis/frozen_clip_team_identity_baseline_2026-05-23/RESULTS_FACTS_2026-05-23.md` §2 + §3.

Joint-calibrated comparison at team-real-FPR = 5% (all 13 heads now):

| Head | Corpus | dor | Xinhe | Xiang | min | mean |
|---|---|---:|---:|---:|---:|---:|
| P8A | (FT'd) | 0.870 | 0.786 | 0.936 | 0.786 | 0.864 |
| E2B | (FT'd) | 0.700 | 0.813 | 0.945 | 0.700 | 0.819 |
| Slot A v2 face-pool | (FT'd) | 0.562 | 0.905 | 0.988 | 0.562 | 0.818 |
| Slot A v2 CLS | (FT'd) | 0.622 | 0.546 | 0.964 | 0.546 | 0.710 |
| T5C | (FT'd, production) | 0.759 | 0.379 | 0.948 | 0.379 | 0.695 |
| MLP_DEV_LB | DEV+LB (4839) | 0.527 | 0.205 | 0.465 | 0.205 | 0.399 |
| LR_DEV_LB | DEV+LB (4839) | 0.422 | 0.153 | 0.381 | 0.153 | 0.318 |
| MLP_DEV | DEV (4000) | 0.292 | 0.160 | 0.159 | 0.159 | 0.204 |
| LR_DEV | DEV (4000) | 0.289 | 0.055 | 0.197 | 0.055 | 0.180 |
| **MLP_OPTB_DEV** | OPTB+DEV (10000) | **0.171** | **0.037** | **0.303** | **0.037** | **0.170** |
| **LR_OPTB_DEV** | OPTB+DEV (10000) | 0.160 | 0.018 | 0.268 | 0.018 | 0.149 |
| **MLP_OPTB** | OPTB (6000) | 0.092 | 0.014 | 0.005 | 0.005 | 0.037 |
| **LR_OPTB** | OPTB (6000) | 0.071 | 0.002 | 0.000 | 0.000 | 0.024 |

### 1.5.6 OPTB-specific findings

1. **Adding training-corpus diversity (OPTB) made the frozen-CLIP head WORSE on team-identity, not better.** Best Option B head (MLP_OPTB_DEV) min_recall = 0.037. Best Option A head (MLP_DEV_LB) min_recall = 0.205. Δ = −0.168.

2. **OPTB head AUCs**: LR_OPTB train AUC 0.978 / dev AUC 0.818 / **lockbox AUC 0.345** (below random chance). MLP_OPTB train 0.994 / dev 0.797 / lockbox 0.516. The OPTB-trained classifier points the WRONG way on lockbox (linear separator inverted).

3. **OPTB+DEV combined heads (10K training frames)** got +14pp on Xiang vs MLP_DEV but lost 12-36pp on dor and 4-17pp on Xinhe. Net Δ min_recall vs MLP_DEV = −0.122.

4. **The standalone OPTB heads collapse to near-zero recall** on Xinhe (0.002-0.014) and Xiang (0.000-0.005) at the joint-calibrated 5% team-real-FPR threshold. The OPTB corpus does not contain Teams-substrate-comparable signal at training-data scale.

5. **OPTB+DEV mean team-real prob = 0.21-0.33** vs DEV-only 0.20-0.27. The OPTB pool dominated the calibration during training; dev_auc stays 1.000 on held-out DEV (DEV is internally separable in CLIP space) but the OPTB-induced calibration drift propagates into team-identity scoring.

### 1.5.7 Status

- Option A: COMPLETE.
- Option B: COMPLETE (extraction 1,753s; head training 248s).
- Total wall: ~55 min on local MPS for all 8 frozen-CLIP heads.
- Total cost: $0.

---

## §2 — Option set A: Deployable today/this week (operator de-prioritized this)

Brief, since the operator explicitly said deployment is less important right now than training direction. Listed for completeness.

| ID | Option | Cost | Wall | What it gives |
|---|---|---:|---:|---|
| A-1 | Per-ckpt τ-recalibration on the 5 ckpts | $0 | 1h CPU | Honest cross-ckpt comparison. Prerequisite for any ship decision. |
| A-2 | Switch production T5C → P8A at mode A | $0 | hours | Mechanically passes team-identity bar. Xinhe-may6 0%. Conservative ship. |
| A-3 | Switch production T5C → Slot A v2 CLS at mode A or B | $0 | hours | Cleaner aggregate real-FPR than P8A; Xinhe-may6 4.3% / 1.1% at mode B; weaker on Xinhe fake-attack recall (0.293). Recall-leaning ship. |
| A-4 | Hold T5C, no change | $0 | — | Status quo. T5C @ Xinhe-may6 17.4% @ τ=0.5 / 3.3% @ mode B. Not catastrophic but not great. |

Recommended in this set: **A-1 + A-2** if you want to act today. A-1 is mandatory before any meaningful cross-ckpt comparison. A-2 is the defensible same-day switch given Slot A v2 face-pool is out (Xinhe-may6 catastrophe).

---

## §3 — Option set B: Training options for a materially better model

This is the primary content. Organized by lever class. None of the options have >40% confidence-to-be-materially-better on the team-identity bar; the realistic shape of "actually better" is a multi-experiment sequence over 1-3 weeks at $200-500 cumulative spend, with the chance of hitting on one that bites.

### §3.I — Untried levers INSIDE the current paradigm

These respond to the adversarial review's catch that the "structural ceiling" diagnosis was overstated. Test whether properly-implemented levers inside the current FT paradigm move the bar before concluding the paradigm itself is wrong.

| ID | Lever | Cost | Wall | P(materially better) | P(informative) | Mechanism + rationale |
|---|---|---:|---:|---|---|---|
| **B.I.1** | Output-preservation aux loss vs frozen anchor | $50 | 4h | 15-25% | 60-70% | Regularize FT encoder features toward Slot A v2 step3500 reference on held-out reference pool. **Directly motivated by Probe 1**: if FT drifts 85-88° from frozen prior, penalize the drift explicitly. Prior reviewer's top in-paradigm recommendation. Proposed-never-run per the refuted-lever catalog (see U_SLOTS / T5C_TRIPLE retros). |
| **B.I.2** | P22 step1k as FT base for any new lever | $50 | 4h | 15-25% | 70-80% | P22 step1k has 140× wider score variance than step8k per `project_p22_cpu_followups_reframe_2026-05-02`. The "robust" P22 ckpt. Has NEVER been used as a downstream FT base, only as a reference / ensemble component. Different starting point on the gradient landscape. **Composable** with anchor_aware or B.I.1 or any new lever. |
| **B.I.3** | KLIEP per-frame training subset | $60 | 5h | 20-30% | 50-60% | D8 (`analysis/cpu_diagnostics_2026-05-12_d8_head_retrain_substrate_balanced/`) showed the KLIEP discriminator has effective sample size 269/2000 — ~13% of training reals are "near" the deploy distribution. Filter the training pool to that subset (with class balancing), then FT. **Closes train/test substrate gap at the data level**, no new data acquisition. |
| **B.I.4** | Multi-lever stacking with proper HP sweep | $100-200 | 8-12h | 20-35% | 60-70% | Reviewer's "no proper stacking tried" catch. E.g., anchor_aware + resolution_chain_aug + face_scale_jitter on Slot A v2 base with grid search on weights. The one prior stacking attempt (T6 jitter on T3) was single-config, not a sweep. |

### §3.II — Structural reframe (IRM / VIB / contrastive class)

Test whether categorically-different training objectives move the bar. The prior reviewer rejected the combined B1+C1 run as the wrong shape (IRM gradient-noise-sensitive; VIB injects gradient noise by design; mathematically incompatible without care). Specs below are sequential with reviewer's modifications.

| ID | Lever | Cost | Wall | P(materially better) | P(informative) | Mechanism + rationale |
|---|---|---:|---:|---|---|---|
| **B.II.1** | IRM-only smoke (reviewer's modified spec) | $30 | 1.5h | 15-25% | 50-70% | **Per-method environment partition (12 envs)** from `project_phase1a_method_cluster_axis_2026-05-01` — NOT per-substrate (2 envs); DomainBed empirical shows 2-env IRM is in the brittle regime. **β-anneal-DOWN** (start high, decrease) not anneal-up — literature consensus is anneal-up exposes the penalty to a shortcut-loaded representation. **Base: P22 step1k** (B.I.2). **Abort criteria**: CE loss diverges >0.5 OR substrate-pair gradient variance reduction <10% by step 500. |
| **B.II.2** | VIB-only smoke (sequential, after B.II.1) | $30 | 1.5h | 15-20% | 50-60% | KL penalty on bottleneck layer. β=1e-3 to 1e-2 with warmup. Detection of "encoder routes shortcut through mean" via end-of-training MI between μ and known shortcut axes. Independent of B.II.1; ablation clean. |
| **B.II.3** | Substrate-pair contrastive pretrain + FT | $100 | 6h | 20-30% | 70%+ | Use 1,825 paired (clean, teams) captures from A0.1 inventory as labeled invariance signal. Contrastive objective pulls (clean_i, teams_i) closer, pushes apart different identities regardless of substrate. Then FT for binary task. **Directly encodes** the invariance Probe 1 says we want; doesn't depend on IRM's gradient-variance penalty. The sampler fix from `c2ed748` made matched-pair in-batch co-occurrence work. |
| **B.II.4** | B.II.1 + B.II.2 combined (conditional escalation) | $80 | 5h | 25-40% conditional | 70%+ | Run ONLY if both standalone smokes (B.II.1, B.II.2) land informative individually. The morning's combined-from-start proposal was rejected; this is the defensible escalation path. |

### §3.III — Bigger structural commitments (ceiling-breaking potential)

The only categories with academic-literature support for **materially better** in the sense of breaking the single-frame ceiling. High cost, high commitment.

| ID | Lever | Cost | Wall | P(materially better, given it runs) | Notes |
|---|---|---:|---:|---|---|
| **B.III.1** | Temporal model (3-5 frame input) | $300-800 GPU + 1-2 weeks infra | days | 30-40% (conditional) | Highest single-experiment upside. Academic lit (Li 2020, Cozzolino 2021, Haliassos 2022) supports temporal-inconsistency detection as the signal channel single-frame models can't access. **Critical gating dependency**: a $30 pre-experiment to verify the temporal signal survives Teams compression on a small sample. If compression kills it, this option dies. |
| **B.III.2** | Self-supervised pretrain on video-call data | $500-1500 + weeks | weeks | Hard-to-estimate; conditional 20-40% | Replace OpenCLIP web-image-text starting point with a webcam-specific prior via masked-patch reconstruction or temporal-consistency objective on unlabeled video-call captures we already have. Deep change; could be transformative or marginal. |
| **B.III.3** | Acquire 7000 webcam-video dataset + retrain | $30-60 operator + retrain cost | weeks | Multiplier on any training; standalone 15-25% | User flagged as potentially-acquirable but "not guaranteed to represent production webcams" and "might be old data." Pursue in parallel — multiplies impact of every other training experiment. |

### §3.IV — Lower-EV options (listed for completeness, not recommended as next-step)

| ID | Lever | Why not lead with this |
|---|---|---|
| B.IV.1 | B32 smaller-patch encoder (~$80) | Capacity isn't clearly the binding constraint; B16 has been the stable choice and L14 was already a closed capacity test |
| B.IV.2 | Another LoRA placement (L5-L7, L11-L12) (~$80) | Placement axis pulled twice (L10-L11, L8-L9) with same family of failure |
| B.IV.3 | Roy_D-specific anchor pool extension | Roy_D = Mac-Roee = out-of-scope per today's team-identity finding; no longer a real lever |
| B.IV.4 | More HEAD-only variants on frozen Slot A v2 (~$30) | Phase 2 HEAD already plateaued at composite 0.235 (1,026 trainable params); adding more head capacity hasn't moved the needle |

---

## §4 — Cumulative cost / EV honest framing

The uncomfortable truth: there is no $80 single experiment with >50% confidence-to-be-materially-better. Realistic shapes of "actually better":

- **$200-300 over 1 week**: run B.I.1+B.I.2 (combined output-preservation + P22-step1k base, $80-100), then B.II.1 IRM-only smoke ($30), then B.III.1 compression-survival pre-experiment ($30 CPU + minimal GPU). Cumulative P(at least one bites materially) ~40-60% under weak-independence assumption.
- **$500-800 over 2 weeks**: above plus follow-up on whatever bit + B.II.3 contrastive pretrain ($100) if both single-lever and structural-reframe classes are null.
- **$800-2000 over 2-4 weeks**: commit to B.III.1 (temporal, gated on compression-survival pre-experiment passing) AND/OR B.III.2 (pretrain).

The reviewer's expected-cost framing is correct: my $80 morning proposal was the floor. Realistic expected commitment for "actually better" is $200-800.

---

## §5 — My recommended ranking

If forced to commit a 2-week plan with ~$300-400 total budget, in order:

### Week 1 (cheap-and-informative, ~$140)

1. **B.I.2 + B.I.1 combined**: P22 step1k base + output-preservation aux loss against Slot A v2 step3500 reference. **$80-100, ~5h wall**. Reviewer's top in-paradigm lever AND a different starting point in one experiment. If output-preservation is the right mechanism but Slot A v2 step3500 was the wrong starting point, this catches both. Best single-experiment-inside-current-paradigm shot.
2. **B.II.1**: IRM-only smoke (reviewer's spec). **$30, ~1.5h**. Tests whether structural-reframe class bites at all; clean attribution.
3. **B.III.1 pre-experiment**: ~$30 (CPU + small GPU). Verify the temporal signal survives Teams compression on a small sample. **Gates whether B.III.1 is a real option going forward.**

### Week 2 (escalate on what bit)

- If B.I.2+B.I.1 bit on team-identity bar: ablate to attribute (P22 base alone vs aux-loss alone) and iterate the winner with anchor_aware
- If B.II.1 bit (gradient variance reduced without CE collapse): run B.II.2 (VIB) and possibly B.II.4 (combined)
- If B.III.1 pre-experiment passed: commit to temporal-model build, $300-500 + infra; **highest-EV continuation**
- If Week 1 produced nothing material: B.I.3 (KLIEP subset) + B.III.3 (7000-webcam acquisition) become primary path

### Parallel throughout

- **B.III.3** (7000 webcam dataset acquisition) — pursue independently; operator-time cost only until ingestion

### What I'd explicitly NOT do

- One $80 single-experiment moonshot (reviewer's critique of the morning's combined B1+C1 is correct)
- B.III.2 (self-supervised pretrain) before B.III.1 results — temporal addresses a more concrete ceiling
- Continued iteration of single-lever anchor_aware / GroupDRO / LoRA-placement variants

---

## §6 — Risks the proposal under-weights (reviewer should probe)

1. **The 5%/50% bar is brand-new and not user-validated.** Per `analysis/team_identity_deploy_readout_expanded_2026-05-23/AGENT_PROPOSAL_2026-05-23.md` §6, P8A at mode B misses Xinhe fake recall by 0.013pp on n=1099 — within sample noise of the 0.50 floor. Relaxing to 0.40 gives P8A mode-B both-pass. The "no ckpt passes" conclusion is fragile to the floor choice.
2. **Cross-ckpt τ comparison is unfair to ckpts with shifted score distributions.** Per-ckpt τ-recalibration (A-1) is a prerequisite that has not been done. The Slot A v2 face-pool "catastrophe" on Xinhe-may6 may be τ-calibration-dependent — though even at conservative τs the failure mode is real per the agent's analysis.
3. **The "single experiment fails informatively" framing remains sneaky.** Each of B.I and B.II options can fail for multiple reasons; attribution requires ablation runs. The cost floor of $80 has an expected total of $160-240 if results are ambiguous.
4. **Confidence intervals on the P(materially better) numbers are wide.** I've given point estimates of 15-30%. These are my honest read but could be off by ±10pp. Reviewer should probe whether they're calibrated against the 20+ refuted single-lever interventions over 6 weeks.
5. **The frozen-CLIP probe could change priorities.** If frozen-CLIP team-identity baseline lands within ~10pp of T5C, data-acquisition (B.III.3) jumps in priority and structural reframe (B.II) drops. Probe was extracting at draft time; result is decision-relevant.

---

## §7 — Decisions the operator must make

Three explicit asks:

1. **Pick week 1 from §5**: full plan ($140), subset, or different ordering. If different ordering, justify.
2. **Decide on B.III.3 (7000-webcam dataset acquisition)**: pursue / defer / decline.
3. **Decide on production switch (Option A-1 + A-2 or A-3)**: yes / no / "decide after frozen-CLIP probe finishes". Default if no decision: status quo (T5C remains production with documented Xinhe-may6 elevation 17.4% @ τ=0.5).

### Secondary decisions (lower priority)

4. Should B.III.1 (temporal) be elevated to Week 1 instead of being held for Week 2? Cost is higher but the option is the only category with academic-literature support for ceiling-breaking.
5. Should B.II.3 (substrate-pair contrastive pretrain) be elevated to Week 1? Different mechanism than IRM/VIB; uses data we already have (1,825 pairs).

---

## §8 — Reviewer ask (summary)

The user wants you to:

1. Verify the option-set is correctly framed and complete (any missing levers?)
2. Critique the cost / wall / confidence numbers (are they calibrated?)
3. Critique the ranking (different week-1 pick? different Week 2 contingencies?)
4. Flag risks I've under-weighted
5. Answer the 3 (+2) decisions in §7 with your recommendation
6. Push back where I'm wrong

Write your review to `docs/packet_retrospectives/plans/TRAINING_DIRECTIONS_REVIEW_2026-05-23.md`. Standards: cite specific evidence (file:line where useful), don't fabricate criticisms, be concrete about alternatives, bound your own uncertainty honestly. Length 1500-3000 words.

---

## §References (the things you should read at minimum to form a view)

### Authoritative project docs
- `docs/packet_retrospectives/MODEL_GOALS.md`
- `docs/packet_retrospectives/AGENTS.md`
- `docs/packet_retrospectives/STATE.md`
- `docs/packet_retrospectives/TIMELINE.md` (last 15 entries)

### Today's other docs in plans/ (prior context)
- `docs/packet_retrospectives/plans/STRUCTURAL_REFRAME_PROPOSAL_2026-05-23.md`
- `docs/packet_retrospectives/plans/STRUCTURAL_REFRAME_REVIEW_2026-05-23.md`
- `docs/packet_retrospectives/plans/MASTER_PLAN_2026-04-29.md` (frozen — gives 6-week program context)

### Today's binding measurements
- `analysis/team_identity_deploy_readout_expanded_2026-05-23/RESULTS_FACTS_2026-05-23.md`
- `analysis/team_identity_deploy_readout_expanded_2026-05-23/AGENT_PROPOSAL_2026-05-23.md`
- `analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/per_human_summary.csv`
- `analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/per_cohort_summary.csv`
- `analysis/xinhe_may6_t5c_revisit_2026-05-23/RESULTS_FACTS_2026-05-23.md`
- `analysis/xinhe_may6_t5c_revisit_2026-05-23/AGENT_PROPOSAL_2026-05-23.md`
- `analysis/frozen_clip_team_identity_baseline_2026-05-23/RESULTS_FACTS_2026-05-23.md` (if landed)

### Memories establishing today's reframe
- `~/.claude/projects/.../memory/project_team_identities_multi_labeled_2026-05-23.md`
- `~/.claude/projects/.../memory/project_production_is_t5c_not_e2b_2026-05-23.md`
- `~/.claude/projects/.../memory/feedback_cpu_jobs_standing_greenlight.md`

### Probe 1 (the structural-ceiling load-bearing measurement)
- `analysis/substrate_pair_geometry_2026-05-22/FALLBACK1_PROBE1_FACTS_2026-05-22.md`

### The shortcut taxonomy (CPU diagnostics D1-D10, May 12)
- `analysis/cpu_diagnostics_2026-05-12_d1_p8a_iq_alignment/D1_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d2_encoder_separation/D2_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d3_cross_substrate_probe/D3_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d4_iq_quartile_alignment/D4_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d5_per_identity_iq/D5_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d6_empirical_orthogonality/D6_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d7_fpr_decomposition/D7_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d8_head_retrain_substrate_balanced/D8_FACTS_2026-05-12.md` (the prior frozen-CLIP partial baseline)
- `analysis/cpu_diagnostics_2026-05-12_d9_source_substrate_inventory/D9_FACTS_2026-05-12.md` (training-real composition: 7.8% teams-v2 + 55.3% HDTF + 37.0% QCLIPS — NOT "100% YouTube" as the morning's proposal incorrectly stated)

### Recent packet retros (the refuted-lever evidence base)
- `docs/packet_retrospectives/packets/U_SLOTS_2026-05-13.md` (LoRA L10-L11, T5C+jitter, Fourier — all refuted)
- `docs/packet_retrospectives/packets/T5C_TRIPLE_2026-05-19.md` (stacking on anchor_aware refuted)
- `docs/packet_retrospectives/packets/AUTO_MODE_ANCHOR_REBALANCE_PREVIEW.md` (anchor_aware = the one win)
- `docs/packet_retrospectives/packets/FACE_POOL_2026-05-22.md` (the face-pool finding — partially refuted by today's Xinhe-may6 catastrophe)
- `analysis/phase_3_scorecard_2026-05-23/RESULTS_FACTS_2026-05-23.md` (today's Phase 2/3 verdicts)

### Relevant threads (cross-cutting topics)
- `docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md` (the strategic program thread; long, skim to recent sections)
- `docs/packet_retrospectives/threads/viso_bucket_gap.md`
- `docs/packet_retrospectives/threads/processing_signature_shortcut.md`
- `docs/packet_retrospectives/threads/clean_teams_identity_pairing.md`
- `docs/packet_retrospectives/threads/face_size_label_leak.md`

### Memory index
- `~/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/MEMORY.md` — full one-liner index of all memories

### Academic references for B.II options
- Arjovsky et al. 2019, "Invariant Risk Minimization" (IRM)
- Krueger et al. 2020, "V-REx: Out-of-Distribution Generalization via Risk Extrapolation"
- Rosenfeld et al. 2020, "The Risks of Invariant Risk Minimization" (IRM brittleness)
- Alemi et al. 2017, "Deep Variational Information Bottleneck" (VIB)
- Achille & Soatto 2018, "Information Dropout"
- Li & Lyu 2020 / Cozzolino 2021 / Haliassos 2022 — temporal deepfake detection lit (for B.III.1)

---

End of document.
