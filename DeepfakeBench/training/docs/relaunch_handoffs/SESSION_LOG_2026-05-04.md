# Session log — 2026-05-04 (post-HANDOFF_FRESH_AGENT)

This document captures everything done in the session that started by reading
`docs/relaunch_handoffs/HANDOFF_FRESH_AGENT_2026-05-04.md`. It is intentionally
non-evaluative — successes, failures, dead ends, and reversals are all reported
the same way. Times are local PDT/PST as dated by file mtimes.

> **Promotion status (added 2026-05-04 night)**: this document was the contemporaneous narrative of the session. Per the AGENTS.md protocol (`docs/packet_retrospectives/AGENTS.md`), findings have been promoted to canonical wiki surfaces in a follow-up maintenance pass. This handoff stays as a baton-passing artifact; the canonical record now lives in:
>
> - **Threads (extended with 2026-05-04 evening update sections)**: [`clean_teams_identity_pairing`](../packet_retrospectives/threads/clean_teams_identity_pairing.md) (DEBATE block on pair-loss refutation), [`viso_bucket_gap`](../packet_retrospectives/threads/viso_bucket_gap.md) (clean single-lever retest in-progress), [`eval_substrate_data_hygiene`](../packet_retrospectives/threads/eval_substrate_data_hygiene.md) (F4 reusable pipeline), [`webcam_fpr_dominance`](../packet_retrospectives/threads/webcam_fpr_dominance.md) (per-substrate τ tool + deployability caveat), [`calibration_vs_training_aug`](../packet_retrospectives/threads/calibration_vs_training_aug.md) (same), [`processing_signature_shortcut`](../packet_retrospectives/threads/processing_signature_shortcut.md) (codec aug verification), [`wandb_yaml_propagation_bugs`](../packet_retrospectives/threads/wandb_yaml_propagation_bugs.md) (4th wandb-side surface bug — entity gotcha)
> - **New packet retros (in-flight stubs)**: [`packets/PA.md`](../packet_retrospectives/packets/PA.md), [`packets/PC.md`](../packet_retrospectives/packets/PC.md)
> - **TIMELINE**: 8 entries appended for 2026-05-04 evening + night
> - **OPEN_LOOPS.md**: regenerated; 2 new structured loops (`pair-loss-asymmetric-variant-untested`, `data-axis-clean-single-lever-retest-in-progress`)
> - **Memory** (3 new entries with frontmatter `wiki_ref` to threads): `project_pair_loss_premise_refuted_2026-05-04.md`, `project_data_axis_clean_retest_packet_a_c_2026-05-04.md`, `feedback_per_mode_tau_not_deployable.md` + index appended to MEMORY.md
> - **Existing memory extended**: `feedback_promotion_contract_launch.md` (added the `launch_experiment.sh` defaults caveat for the wandb-entity gotcha)
>
> The AGENTS.md protocol violation that this session committed (initial findings landed only in handoff, not threads) is corrected. A separate "biases identified in existing wiki content" call-out from the same maintenance pass identified five biases that may need further user-resolved DEBATE blocks; those were flagged in the user-facing assistant turn that triggered this maintenance work, not edited into the threads (per the user's "do not edit existing factual content; flag biases via additions" directive). The DEBATE block in `clean_teams_identity_pairing` is the only one of the five that was authored as a thread addition this session — the remaining four (frame-level AUC framing understates deployment failure; per-mode τ in webcam_fpr_dominance / calibration_vs_training_aug; jitter@0.50 endorsement in README; superseded `p14-data-fix-not-launched` based on confounded test) were either already partially addressed in prior thread updates or were addressed by the additions in this maintenance pass.

## 0. Starting state (from HANDOFF doc)

- Anchor candidates: P8A (CLIP-init, 5x FT chain) vs E2B_3200 (B16 scratch + CE + heavy aug)
- Known facts coming in:
  - P8A holds viso recall ~27% ceiling unbroken across 13+ R13 packets
  - E2B_3200 breaks deeplive ceiling (87.5% at FPR=10%) but regresses viso (27→7%)
  - L14 (E3) does not break viso ceiling (11.6%)
  - F4 substrate cleaning (drop chronic-6 + lowres + no-face) gives P8A viso 27→67%
  - Job B verification probe was pending in us-east1 (testing pre-RLP6_04 viso ceiling hypothesis)
- Outstanding goals: 24h deeplive ship + 72h overall ship (deeplive + viso, low FPR, robust)

## 1. CPU diagnostic jobs (chronological)

All artifacts under `analysis/<dir>/2026-05-05/` with `summary.json`, `run.log`,
`run_*.py`, and per-job CSV outputs.

### 1.1 `chronic6_capture_mode_iou_2026-05-05` (17:03)
**Question:** Are the chronic-6 FP-prone identities the same set as the "screen"
capture-mode cohort?

**Result:**
- 5 of 6 chronic-6 identities fall inside the screen-cohort (frac=1.0 of those covered)
- IoU(full chronic-6, screen-cohort) = **0.294**
- IoU(covered chronic-6, screen-cohort) = **0.313**
- 16 identities total in the screen-cohort

**Read:** chronic-6 are a *subset* of the screen cohort but NOT equivalent — there
are 11 other screen-cohort identities not on the chronic-6 list, and 1 chronic-6
identity outside the screen-cohort. So a "drop screen-cohort" filter is broader
than chronic-6 dropping.

### 1.2 `p8a_signature_decomposition_2026-05-05` (17:03)
**Question:** Decompose the 550 viso fakes by which checkpoints catch them, to
isolate what P8A's 5-stage FT chain uniquely contributes.

**Result (cohort sizes, n=550 viso fakes):**

| cohort | count | % | meaning |
|---|---|---|---|
| D_missed_all | 364 | 66.2% | none of P8A/E2B/E3 catch |
| C_caught_P8A_only | 93 | 16.9% | unique to P8A |
| B_caught_P8A_E3_not_E2B | 26 | 4.7% | CLIP-init chains catch, B16-scratch misses |
| A_caught_all | 25 | 4.5% | all three catch |
| Y_caught_E3_only | 21 | 3.8% | unique to L14-scratch |
| X_caught_E2B_only | 13 | 2.4% | unique to B16-scratch |
| Z_caught_P8A_E2B_not_E3 | 4 | 0.7% | CLIP+B16 catch, L14 misses |
| W_caught_E2B_E3_not_P8A | 4 | 0.7% | scratch backbones catch, P8A misses |

- P8A caught: 26.9% (matches Job 14 baseline 27%)
- E2B caught: 8.4%
- E3 caught: 13.8%
- C cohort (P8A unique 93 frames): split 63 raw / 30 teams — both subtypes contribute

**Read:** 16.9% of viso fakes are caught ONLY by P8A. 66.2% missed by all three.
The 26-frame B cohort is the "lesson a from-scratch B16 needs" if it has a
coherent IQ pattern.

### 1.3 `chronic6_fingerprint_2026-05-05` (17:21)
**Question:** Can a per-frame quality gate distinguish chronic-6 frames from
non-chronic webcam frames? (Decides whether deployment can reject chronic-6
without a per-mode classifier.)

**Result:**
- Top separating feature: `width` (KS=0.483, p≈1e-125)
- Threshold for "deployment-viable per-frame gate" was KS≥0.5 with p<1e-5
- Chronic-6 covered: 830 frames; non-chronic webcam: 2048; normal_photo: 1128

**Read:** KS=0.483 is RIGHT BELOW the 0.5 threshold. Chronic-6 vs non-chronic
webcam are partially but not cleanly separable by frame-level features. A pure
quality gate cannot reliably reject chronic-6 — needs training-side intervention
or a coarser gate (`min(W,H)<200` is the chosen compromise in F4).

### 1.4 `lockbox_per_mode_tau_2026-05-05` (17:33)
**Question:** Does a per-capture-mode τ calibrated on dev generalize to lockbox?

**Result (per-checkpoint, per-mode τ values dev vs lockbox):**

| ckpt | normal_photo dev | normal_photo lockbox | webcam dev | webcam lockbox |
|---|---|---|---|---|
| P8A | 0.100 | 0.332 | 0.987 | 0.979 |
| E2B_3200 | 0.123 | 0.529 | 0.419 | 0.903 |
| E3_6600 | 0.007 | 0.018 | 0.503 | 0.972 |

- **Lockbox real FPR under dev-calibrated per-mode τ:**
  - P8A: 16.9%
  - E2B: 47.6%
  - E3: 62.8%
- **Lockbox fake recall under dev-calibrated per-mode τ:**
  - P8A: 78.6%, E2B: 92.9%, E3: 93.2%

**Read:** Dev-calibrated per-mode τ does NOT generalize cleanly to lockbox.
P8A is the best — only 16.9% lockbox FPR — but still well above the 10% target.
E2B and E3 over-fire badly (47-63% FPR). The per-mode τ idea is a recall lever
but the calibration shifts substantially substrate-to-substrate.

### 1.5 `viso_capture_mode_proxy_2026-05-05` (17:33)
**Question:** If we predict capture-mode for viso frames using a lightweight
classifier (since viso doesn't have ground-truth tags), does per-mode τ help?

**Result:**
- Classifier CV accuracy: 52.2% ±1.6% on 4 features (width, height, brightness, saturation_norm)
- Predicted distribution: 543 normal_photo, 7 webcam (out of 550 viso fakes)
- Recall lift if predicted-mode τ is applied:

| ckpt | global τ recall | per-mode predicted-tau recall | lift |
|---|---|---|---|
| P8A | 26.9% | 55.8% | **+28.9pp** |
| E2B | 8.4% | 23.1% | +14.7pp |
| E3 | 13.8% | 88.7% | **+74.9pp** |

**Read:** The lifts are large but the classifier accuracy (52%) is barely above
random. The numbers are upper bounds on the lift if predictions were perfect.
The E3 lift is suspicious — likely artifact of E3's per-mode τ being well-tuned
to predicted normal_photo. CRITICAL CAVEAT: per-mode τ at deployment requires
detecting capture mode in production, which the user later ruled out.

### 1.6 `deeplive_deployment_24h_2026-05-05` (18:21)
**Output:** `DEPLOYMENT_PACKET_DEEPLIVE.md` — QA-ready deployment artifact.

**Policy chosen:**
- Anchor: E2B_3200 (B16 scratch + CE)
- Quality gate: `is_no_face OR min(W,H)<150 OR laplacian_var<8`
- Single global τ (NOT per-mode)
- τ_recommended = **0.4368** (calibrated on lockbox-shaped real population at 5% FPR)
- Higher-recall option: τ = 0.2889 (10% FPR)

**Performance under recommended policy:**
| metric | value |
|---|---|
| Deeplive recall | 97.0% |
| FPR on lockbox reals | 5.1% |
| Frames rejected by gate (lockbox reals) | 3.9% |
| Frames rejected by gate (deeplive fakes) | 1.7% |

**Caveat included in the doc:** lockbox real population is 81% Dor Shkedi
(1144/1418 frames). Identity-leak artifact means production FPR on unseen
identities will likely be lower than 5.1%. The 16 deeplive false-negatives
are also all Dor.

### 1.7 `codec_aug_verification_2026-05-05` (19:17)
**Question:** Does `TeamsCodecSimulation` (the existing aug class) produce
output that matches actual Teams transport on the same source frames?

**Method:**
- Downloaded 30 paired (raw, teams) viso frames where the same source frame
  exists in both clean and teams-transported buckets
- Applied `TeamsCodecSimulation` to the raw frames
- Computed 8 IQ feature deltas (laplacian, sobel_edge, luma_mean, luma_std,
  saturation, contrast, hf_energy)

**Result:**
- Cosine similarity between (aug delta) and (actual transport delta): **0.87 - 0.99** across all 8 features
- Magnitude ratio (aug delta / actual delta): **97% - 106%**

**Read:** Codec aug is a faithful simulation of actual Teams transport on these
8 IQ axes. Pre-existing calibration (against 18 videos × 132 frame pairs in
the original aug class docstring) is empirically validated on a different
sample. Re-enabling codec aug is low-risk.

### 1.8 `pair_loss_scoping_2026-05-05` (19:10)
**Output:** `FINDINGS.md` — implementation effort estimate for pair loss.

**Key findings:**
- Closest existing analog: `StabilityRegMixin` in `trainer/mixins/stability.py`
- It already implements `KL(perturbed_logits || clean_logits.detach())`
- The missing piece is using actual companion frames (real teams transport)
  instead of synthetic perturbations
- Companion infrastructure exists: `VisoMasterTeamsEnhancedSample` carries
  `companion_bucket` reference; `_iterate_visomaster_teams_enhanced_sample`
  currently picks ONE branch per iteration (original OR teams), never both
- Implementation effort: 4-6h focused dev work
- Cost estimate (if proceeded): ~$92 + 6h dev for one packet

### 1.9 `pair_loss_effect_verification_2026-05-05` (19:32)
**Question:** Before spending 4-6h dev + $87 GPU, verify the pair-loss premise
empirically on real data.

**Method:**
- 275 paired (raw, teams) viso fakes with seq_id matching
- Q1: Per-pair feature cosine similarity (P8A frozen 512-d features, used as
  proxy because E2B features not cached on disk)
- Q2: Pearson correlation between feature distance and E2B score gap
- Q3: Cohort sizes at various τ thresholds — how many pairs would pair loss help?

**Result:**

| metric | value | interpretation threshold | verdict |
|---|---|---|---|
| Q1: paired cosine | 0.878 | <0.75 STRONG, >0.95 LOW | MODERATE |
| Q1: random within-subtype baseline | 0.686 | (paired Δ = +0.19) | |
| Q2: Pearson r(feat dist, E2B abs score gap) | -0.089 (p=0.14) | <0.2 LOW | LOW |
| Q3: cohort raw>0.5 & teams<0.5 | 3 / 275 (1.1%) | <10pp LOW | LOW |
| Q3: opposite cohort teams>0.5 & raw<0.5 | **29 / 275 (10.5%)** | (3-10× larger) | NET NEGATIVE |

**Sign-of-effect (load-bearing):**
- E2B mean raw_score = 0.086, mean teams_score = 0.172 (Wilcoxon p=0.0019)
- Teams transport HELPS E2B catch viso fakes (opposite of premise)
- 85.8% of viso pairs miss in BOTH versions — bottleneck is representation, not transport invariance

**Threshold sensitivity:**
| symmetric τ | target cohort | wrong-way cohort | net pp if symmetric pair loss perfect |
|---|---|---|---|
| 0.05 | 26 | 62 | −13.1 |
| 0.10 | 17 | 55 | −13.8 |
| 0.20 | 11 | 47 | −13.1 |
| 0.50 | 3 | 29 | −9.5 |

**Verdict:** LOW. Skip the pair-loss packet. The premise is empirically
refuted. The probe could refine Q1/Q2 (feature geometry) but cannot reverse
the sign-of-effect or cohort math (computed directly from E2B scores, not
from features).

**Caveat acknowledged:** Q1/Q2 use P8A features as a proxy. A $5 GPU probe
to extract E2B features and re-run Q1/Q2 directly would address this
specifically — but cannot change the cohort verdict.

### 1.10 `substrate_cleaning_eval_2026-05-05` (20:02)
**Output:** Reusable F4 substrate-cleaning eval pipeline ready for new ckpts.

**Built:**
- `run_clean_eval.py` (single-ckpt mode + reference-run regression mode)
- `cleaned_substrate_manifest.json` — frozen F4 contract
- `reference_run_p8a.json`, `reference_run_e2b.json` — Job 14 reproduction

**Reference run reproduction:**

| ckpt | suite | F0 (full) | F4 (cleaned) | Job 14 expected |
|---|---|---|---|---|
| P8A | viso | 26.9% | 67.1% | 27→67% ✓ |
| P8A | deeplive | 42.4% | 92.5% | 42→92% ✓ |
| P8A | teams_fake | 69.9% | 92.2% | (new) |
| P8A | real FPR | 10.0% | 0.86% | (new) |
| E2B_3200 | viso | 8.4% | 30.9% | 8→31% ✓ |
| E2B_3200 | deeplive | 93.9% | 100% | (new) |
| E3_6600 | viso | 13.8% | 77.6% | 14→78% ✓ |

All assertions within 0.5pp. **Reproduction PASS.**

**F4 contract:**
- Drop chronic-6 identities: `bla_bla_chow`, `bla_bla_chow__s2`, `pc_generator__s22`, `pc_generator__s45`, `roy_d`, `q__s6`
- Drop frames with `min(W, H) < 200`
- Drop frames where `is_no_face == True`

### 1.11 `per_substrate_tau_calibration_2026-05-05` (20:04)
**Output:** Reusable per-substrate τ-calibration tool ready for new ckpts.

**Built:**
- `run_calibration.py` (real_dev for calibration, real_lockbox for held-out diagnostic, N fake suites)
- 5 substrates: normal_photo, webcam, phone_screen, screen, screen_recording
- 100-point τ sweep + quantile anchors
- Reports per-substrate FPR + per-suite recall

**P8A reference results (single τ, substrate-aware deployment-viable picks):**

| target FPR | τ | lockbox FPR | teams_fake_lockbox recall | viso recall | deeplive recall |
|---|---|---|---|---|---|
| 5% (strict) | 0.9941 | 0.07% | 16.2% | 0.18% | 0.0% |
| 10% (moderate) | 0.9891 | 0.35% | 24.9% | 1.6% | 4.0% |
| 20% (loose) | 0.8409 | 2.68% | 46.8% | 20.9% | 31.0% |

**Job-7 21pp lift validation:**
- Oracle per-mode dev-calibrated τ on P8A: lockbox teams_fake recall 78.6% at 16.9% FPR
- Single global TAU_F0_DEV (0.705): 54.1% recall at 4.3% FPR
- **24.5pp lift at 4× the FPR** (matches/exceeds memory note of 21pp)

**CRITICAL READ:** The 24.5pp lift comes from oracle per-mode τ which requires
detecting substrate at deployment — NOT viable per user constraint. The
deployable single-τ-substrate-aware results above are bleak: at 5% FPR
ceiling, P8A viso recall is 0.18%. At 20% FPR (probably too high for
production), it's still only 21%.

## 2. Decisions and reframings (driven by user)

### 2.1 P8A as anchor — questioned by user
**User statement:** "I'm a bit skeptical that P8A is the right way to go even
if it shows good promise on paper. This is a five times Fine tuned model... if
it performs better in some way, you should be able to trace back and figure out
what is it exactly that it does better why and how and then train a better
version learning all the other lessons that we have - From scratch"

**Action:** P8A signature decomposition (1.2) was launched specifically to
identify which 16.9% (93/550) of viso catches are unique to P8A. C cohort
analysis identifies them; further work to characterize them by IQ pattern is
unstarted.

### 2.2 Per-mode τ killed as deployment policy
**User statement:** "I'm not fully confident I understand what 'Per-mode τ
calibration' means, But if this means we have to know in advance the mode of
the camera that is sending the video - It's a no go, We have no way of knowing
this practically."

**Action:** Deployment policy was rewritten to single global τ + narrow
quality gate. The deeplive deployment packet (1.6) was rebuilt to use single
global τ=0.4368. Per-mode τ is retained as offline analysis tool only.

**Acknowledged misframing:** Earlier in the session, per-mode τ was
recommended as a deployment policy. This was a mistake — corrected as soon
as user flagged it.

### 2.3 Teams-transported viso = THE deployment threat metric
**User statement:** "teams-transported viso -> Perhaps the single most important
metric for visoMaster - This is the classy case when someone uses this tool
over a teams call (Exactly what we need to catch)."

**Action:** This destroyed the value of distillation packets (L14 worse than
B16 on teams subtype) and reframed Packet C-codec design specifically to
target teams-transport robustness. Pair loss verification was triggered by
this reframing too.

### 2.4 Data availability assumption corrected
**Earlier session claim:** "We have used the available data."

**User pushback:** asked to verify.

**Investigation:** Found ~5400 unused viso frames in
`visomaster_enhanced` and `visomaster_teams_enhanced` GCS buckets. These
are disabled in current packets. Memory entry "data-axis lever pulled twice"
was based on tests with confounds (P14_DATA_FIX with bundle, P16_DATA_AXIS
without).

**Action:** Packet A built around enabling these data sources as a
single-lever test.

## 3. Packets created and launched

### 3.1 R13_PA_VISOMASTER_ENHANCED_DATA.yaml
**Created:** 19:04, 11.6 KB

**Design:**
- Base: E2B_3200 baseline (B16 scratch + CE + heavy aug)
- Single change: enable `visomaster_enhanced` and `visomaster_teams_enhanced`
  data sources
- `family_weight: visomaster_enhanced_fake = 4.0` (matches base visomaster_fake)
- Companion infrastructure: `companion_domains: [teams_v2]`, `p_original: 0.5`
- Seed: 3023

**Launch:** Vertex job `3140330896851206144`, us-east1, image `1.3.256`
**State as of session end:** `JOB_STATE_RUNNING`

### 3.2 R13_PC_CODEC_PLUS_DATA.yaml
**Created:** 19:20, 12.4 KB

**Design:**
- Same as Packet A
- PLUS: `teams_codec_simulation` enabled with `policy: adaptive_mixture`
- `enhanced_families`: viso/deeplive enhanced (heavier mode)
- `exclude_families`: all `*_teams_*` buckets (don't double-codec real teams transport)
- `probability: 0.5` (matches P8A historical value)
- Seed: 3024

**First launch attempt:** Vertex job `318825730303590400`, us-east1, image `1.3.257`
**Result: FAILED in 30 seconds.**

**Root cause:** I exported `WANDB_ENTITY=roeehub` before launch. The default
in `scripts/launch/launch_experiment.sh` is `dtect-vision`. There is no
`roeehub` entity on wandb. The container ran, hit `wandb.init`, and crashed
with `entity roeehub not found during upsertBucket` (404).

**Mistake acknowledged:** I conflated the wandb username (`roeehub`) with the
entity. Memory `feedback_promotion_contract_launch.md` updated to flag this
specifically for `launch_experiment.sh`.

**Relaunch:** Vertex job `1202657157175050240`, us-east1, image `1.3.257`
**State as of session end:** `JOB_STATE_PENDING`

### 3.3 Pair loss packet (R13_PD_PAIR_LOSS.yaml) — NOT created, NOT launched
**Reason:** Sub-agent verification (1.9) returned LOW verdict. Pair-loss
premise is empirically refuted on E2B for viso. Decision: skip.

## 4. Image rebuild

**Cloud Build job:** `924a8468-f862-4655-a104-10acbcb9fa89`
**Triggered:** because `R13_PC_CODEC_PLUS_DATA.yaml` was created 16s after the
prior image push. Image-currency check on launch script flagged it.
**Duration:** 7m 42s
**Result:** SUCCESS, image `1.3.257`

## 5. Memory updates

### 5.1 `feedback_promotion_contract_launch.md` — extended
Added a new section flagging that `scripts/launch/launch_experiment.sh` has
default values for `WANDB_API_KEY` and `WANDB_ENTITY` (line 95-96), unlike
the promotion contract launchers. Specific guidance: do NOT override
`WANDB_ENTITY=roeehub` — it does not exist on wandb. Cited the exact failed
job ID for traceability.

## 6. Outstanding state at session end

### Vertex jobs in flight (us-east1)
- **Packet A**: `JOB_STATE_RUNNING` (job `3140330896851206144`)
- **Packet C-codec**: `JOB_STATE_RUNNING` (job `1202657157175050240`, started 20:03 PDT)
- **Job B (verification probe from prior session)**: `9082408392701509632` — status not re-checked in this session

### CPU artifacts ready for use when training jobs finish
- `analysis/substrate_cleaning_eval_2026-05-05/run_clean_eval.py`
- `analysis/per_substrate_tau_calibration_2026-05-05/run_calibration.py`
- Both have USAGE.md and reference runs that match prior memory claims

### Decisions pending (from end-of-session message)
- A) $5 GPU probe to disconfirm pair-loss with actual E2B features (vs P8A proxy) — proposed, no decision
- B) AUTHORIZED — substrate cleaning + per-substrate τ work — DONE
- C) Other shape changes to plan — unanswered

## 7. Summary of empirical findings (non-evaluative)

### Things that became more strongly supported by today's diagnostics
- **F4 substrate cleaning is real** — reproduced cleanly across P8A, E2B, E3 (1.10)
- **Codec aug is faithful to actual Teams transport** — cosine 0.87-0.99 on 8 IQ axes (1.7)
- **Per-mode τ is a strong recall lever offline but not deployable** — 21-25pp lift but requires substrate detection at inference (1.4, 1.5, 1.11)
- **P8A unique catches exist (16.9% of viso fakes)** — characterization of WHY is unstarted (1.2)

### Things that became more strongly refuted by today's diagnostics
- **Pair-loss premise on E2B for viso** — sign of effect reversed, opposite-direction cohort dominates (1.9)
- **Chronic-6 = screen capture mode** — IoU 0.31, not equivalent (1.1)
- **Per-frame quality gate alone can replace chronic-6 dropping** — KS=0.48 below threshold (1.3)

### Things still unresolved
- What specifically does P8A's C cohort (93 unique viso catches) have in common? — IQ profile decomposition partially done (`per_cohort_iq_profile.csv` exists) but not summarized
- Will Packet A or C-codec actually beat E2B on viso? — answer in ~24h
- Is there a deployable single-τ policy that gets us to viso recall ≥30% AND FPR ≤5%? — current analysis suggests NO without per-mode classification
- Is the $5 GPU disconfirmation probe for pair loss worth running? — pending user decision

## 8. Mistakes / corrections during this session

For accountability:

1. **Per-mode τ initially recommended as deployment policy** (corrected after user
   flagged the deployment-detection requirement)
2. **"We have used the available data" claim** (corrected after user pushback;
   investigation found unused enhanced viso buckets)
3. **Treated teams-transported viso as "next problem"** (corrected to "primary
   threat metric" after user statement)
4. **`WANDB_ENTITY=roeehub` override killed Packet C-codec first launch** (memory
   updated to prevent recurrence)
