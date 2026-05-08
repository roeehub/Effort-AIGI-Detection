# Model Goals & Success Criteria — Teams Deepfake Detector

> **Status**: AUTHORITATIVE. This is the single-page articulation of what success looks like for the production model. New agents read this BEFORE proposing experiments. If a packet's hypothesis cannot be defended in terms of the goals on this page, the packet is wrong-target.
>
> **Audience**: any agent picking up this work cold; the user when adjudicating proposed experiments.
>
> **Authoring**: 2026-05-08, drafted by an agent and **user-reviewed** before commit. Edits to this file require the same review discipline — this doc directs future GPU spend, so it must not drift unverified.
>
> **Where it sits**: top-level wiki sibling of [`AGENTS.md`](AGENTS.md), [`AGENT_GUIDE.md`](AGENT_GUIDE.md), [`STATE.md`](STATE.md). Agents should read this AFTER `AGENTS.md` and BEFORE the user's specific task.

---

## What we are building

A **single-model**, **real-time**, **per-frame** deepfake detector deployed as a hook into Microsoft Teams calls. Input: face crops from the Teams capture pipeline. Output: a calibrated `prob_fake ∈ [0, 1]` per frame.

The deployed system makes a per-window decision (alert / no-alert) by aggregating per-frame scores; **the model itself is responsible only for the per-frame score**. Aggregation, thresholding, and alerting policy live downstream of the model in the deployment runtime.

---

## Three pillars (the success criteria)

A model that aces one and falls on another **is not deployable**. All three are load-bearing.

### Pillar 1 — Fake recall on target methods

The model must catch the fakes the deployment is exposed to.

- **Target methods**: primary risk is `deeplive` and `viso` families, with `_enhanced` (GFPGAN-applied) and `_teams` (Teams-pipeline-transported) transport variants. Method-axis taxonomy in memory `project_data_domain_taxonomy.md`.
- **Headline metric**: `dev_fake_macro_recall` at the contract-calibrated τ. Promotion contract recall floor: `target_fake_recall_min = 0.30`.
- **Lockbox readout**: `lockbox_recall_at_FPR_10pct`. F1 target = ≥90% lockbox recall at deployable τ.
- **Cross-substrate**: HDTF readout (Phase C) must not regress >5pp vs E2B at the same τ.

### Pillar 2 — Real FPR < 5% in production

The model must not cry wolf.

- **Promotion-contract anchor**: `target_real_fpr = 0.07`. Production target tighter (≤5%).
- **Stress-FPR ceiling**: `target_stress_fpr = 0.10` on chronic-FP-prone substrates.
- **Per-identity bound**: single-identity FPR > 30% breaks the product. Memory `project_chronic_offenders_partition_per_ckpt_2026-05-04.md`: each chronic identity has at least one ckpt that handles it cleanly — so per-identity catastrophic regression is preventable, and is grounds for promotion rejection.

### Pillar 3 — Robustness across capture conditions

The model is exposed to a **distribution** of capture conditions; collapse on any one is a product failure. "Low FPR on average" is not enough — the FPR must be low **conditional on each axis-bin**.

Required-robust axes:

| axis | thread / memory | known status |
|---|---|---|
| Sharpness / Laplacian | `threads/image_quality_shortcut.md`, `project_image_quality_shortcut.md` | active shortcut; mid-band Fourier amp aug (Slot D) is one mitigation in flight |
| Camera (webcam / DSLR / phone) | `project_lockbox_fpr_dominated_by_webcam_mode.md` | webcam mode dominates lockbox FPR |
| Codec (Teams-pipeline H.264, low-bitrate, packet loss) | `project_pc_codec_aug_hurts_viso_2026-05-05.md` | codec aug as single lever refuted; codec-axis robustness still required, just not via that lever |
| Color cast (warm/cool tint, color-graded skin) | `project_dor_drift_named_axes_2026-05-06.md` | live; chronic Roy_D regression on `color_b_dev` axis |
| Face-pixel-area (close-up vs medium shot) | `threads/face_size_label_leak.md`, `project_face_size_label_leak.md` | mitigated by `face_scale_jitter@0.50` (P14 winner) |
| Crop tightness | `threads/eval_production_crop_tightness_gap.md` | live structural gap eval-vs-production |
| Capture-mode (webcam vs screen-share) | `threads/processing_signature_shortcut.md`, `feedback_per_mode_tau_not_deployable.md` | per-mode τ NOT deployable (Teams doesn't surface mode); single-τ-substrate-aware required |
| Lighting (low-key, harsh shadow, washed-out) | (no dedicated thread) | implicit in IQ axes; not currently isolated as a probe |

A passing model on Pillar 3 has **decoupled** behavior on each axis: per-axis FPR doesn't track the axis. The IQ-shortcut threads document the structural hazards; new packets must articulate which axis they target and what evidence would falsify the proposal.

---

## Architecture

### Production: OpenCLIP B16-DataComp-XL backbone

All deployment-grade candidates use **B16**. Memory chain `project_phase1a_method_cluster_axis_2026-05-01.md`, `project_p18_diagnostics_complete_2026-05-02.md`, `project_p22_succeeded_2026-05-02.md`, `project_per_layer_divergence_2026-05-06.md` all reference B16. The trainable head + SVD-residual fine-tuning recipe (rank=736, k=32, `apply_svd_to_in_proj=true`, `apply_svd_to_mlp=true`) is the standard configuration.

### L14 is for capacity tests ONLY

Memory `project_l14_does_not_break_viso_ceiling.md`: L14-scratch+CE matched B16-scratch on the deeplive ceiling-break but did not break the viso ceiling. The structural reading: **L14 packets answer "is the issue model-capacity?"** When the answer comes back "no" (as it has, multiple times), L14 is closed for that question. **Do NOT propose L14 production candidates.** L14 is a probe instrument, not a deployment path.

---

## Deployment

### Current production model: E2B

**The deployed model is `E2B_TOP_N_STEP3200`** (B16 scratch+CE, FT-from-CLIP, family E2B). Per memory `project_deployment_is_e2b_2026-05-06.md`: deployment scores match local CPU E2B inference at Pearson r=+1.000.

This is a fact, not a goal. New candidates are measured against E2B as the bar to beat.

### Production anchor: P8A

**P8A_REFERENCE_STEP5000** is the substrate-invariance anchor. Memory `project_p8a_breakthrough.md`, `project_job7_head_retrain_REFUTED_2026-05-04.md`, `project_p18_diagnostics_complete_2026-05-02.md`. P8A's signature property is `0/1170` real_lockbox_dor_FPR — **new models must not regress this**. P8A is the substrate-invariance North Star; if a candidate beats E2B on recall but breaks P8A's chronic-FP behavior, it does not promote.

### Single model — NO ENSEMBLE

**This is a hard rule, not a preference.** Do not propose ensemble solutions in any form.

The forbidden shapes:
- Score-fusion (P8A + E2B mean / max / min)
- Specialist-per-method (deeplive specialist + viso specialist + router) — even when memory `project_deeplive_specialist_option.md` notes deeplive-specialist as a possible separate product, that is a **separate-product decision**, not an ensemble inside the Teams detector
- Multi-checkpoint min/max/mean rules at deployment
- Mixture-of-experts heads inside one architecture if it requires multiple forward passes
- Cascade-of-models (cheap classifier + expensive classifier on uncertain frames)

The reasons:
1. **Latency budget**: deployment admits one forward pass per frame.
2. **Ops surface**: a single binary delivered as one Vertex artifact maps cleanly to the deployment pipeline; ensembles split that contract and multiply ops surface area.
3. **Empirical evidence on the option-of-last-resort**: memory `project_job12_ensemble_ceiling_2026-05-04.md` showed **label-free ensembles do NOT break the viso ceiling** (best non-oracle ensemble = 16.2% < P8A 27%). The capability isn't in the ensemble; it's in better single-model representation. Do not propose ensembles in the hope they extract free recall — Job 12 says they don't.

The **only** acceptable composite reasoning is **scalar policy on top of one model's score**:
- Per-substrate τ (where deployable; capture-mode-aware τ is offline-only per `feedback_per_mode_tau_not_deployable.md`)
- IQ gate (abstain below a sharpness threshold; see "Deployment policies" below)
- Per-frame score → per-window decision aggregation (downstream of model)

These are deployment-runtime calibrations on a single model's output, not multi-model ensembles.

---

## Canonical eval substrates

| substrate | role | when used |
|---|---|---|
| **Phase A — 29-suite contract scorecard** | The arbiter for "does this ckpt promote". Lexicographic τ + lockbox readout per `promotion_contract` policy. | Required for any promotion decision. |
| **Phase C — 16-suite HDTF cross-substrate** | Cross-substrate validation; mandatory for any substrate-invariance claim. | Required for any promotion candidate; no >5pp HDTF regression vs E2B. |
| **F-suite (F0-F5)** | Production-honest filtered substrates. F4 strips chronic-6 + lowres + no-face per memory `project_job14_substrate_clean_2026-05-04.md`. | Used to read past substrate-pollution; F4 is the production-honest baseline. |
| **v2 lockbox** | Dor + 16-swap-model substrate per memory `project_v2_substrate_is_dor_diverse_swap.md`. | Lockbox-recall readout in promotion contract. |
| **Per-frame canary (`trainer/mixins/canary_probe.py`)** | In-training 800-frame mid-epoch readout. Detects early-stopping triggers and trajectory non-monotonicities. | Diagnostic only — **NOT a substitute for Phase A**. |

---

## FPR budget (the gate)

Promotion contract policy (current; see `utils/promotion_contract/`):

| key | value | meaning |
|---|---:|---|
| `target_real_fpr` | 0.07 | calibration anchor for τ search |
| `target_stress_fpr` | 0.10 | robustness FPR ceiling on stress substrates |
| `target_fake_recall_min` | 0.30 | recall floor; drops τ-tail-collapse readouts |
| F1 target | ≥0.90 lockbox recall | at production-deployable τ (per memory `project_f1_recall_results_2026-05-04.md`) |

These reflect the chosen tradeoff between user trust and detection capability. **A model that doesn't clear them in offline eval doesn't ship.** They may be re-tuned at deployment time, but offline-eval-clearance is the gate, not a target to "approach".

---

## Deployment-time policies (NOT model-internal)

These are policies the deployment runtime applies on top of the model output. **Not in scope for training**, but relevant for understanding what the model is responsible for.

### Resolution / IQ gate

**Below a face min-dim or sharpness threshold, the deployment runtime should ABSTAIN rather than score.**

User policy (eyeball 2026-05-08, memory `project_canary_below_production_resolution_2026-05-08.md`): the chronic-6 canary substrate is pixelated/upscaled-looking; production frames will be sharper; "we can probably afford in production to simply ignore that level of pixelation and just reject algorithm analysis until we get a better resolution."

Implication for training: **chronic-FP failures on below-gate frames are NOT necessarily production-relevant**. Don't burn GPU on packets that target only below-gate failure modes. The model is not required to handle frames below the gate; the gate handles them.

The threshold is TBD — calibrated by comparing production-frame statistics to canary-frame statistics on (face_min_dim, Laplacian variance). This is a deployment-runtime decision, not a training input.

### Substrate-aware τ

Separate τ for webcam vs. screen-share would be deployable IF Teams surfaced capture-mode at inference. **It does not.** Memory `feedback_per_mode_tau_not_deployable.md`: Teams doesn't surface capture mode; a mode classifier is 52% CV (not deployable); the 24.5pp lift from per-mode τ is **offline-only**. Single-τ for production for now.

### Frame-rate aware aggregation

Per-frame score → per-window decision is deployment runtime logic. Not in scope for training.

---

## Promotion bar (the gate to replace E2B)

A candidate must clear ALL of:

1. **Beat E2B on `dev_fake_macro_recall`** at contract τ (current E2B: 0.508 at the calibrated τ).
2. **Not regress P8A's chronic-FP behavior** — specifically PC_Generator, Roy_D, bla_bla_chow per-identity FPR must not exceed P8A's by more than 5pp.
3. **Pass the F1 target** — ≥90% lockbox recall at FPR ≤ 10%.
4. **Pass Phase C cross-substrate (HDTF)** — no >5pp regression vs E2B on the corresponding suites.
5. **Articulate the lever** — the agent proposing promotion must name the structural change responsible for the lift, with evidence from the experimental record. **No "trained for longer" or "different seed" promotions.**

Below ANY of these bars: it is not a promotion candidate, regardless of headline lift.

---

## In-scope but explicitly low-priority

- Improving HDTF cross-substrate generalization beyond E2B's level (E2B is OK, not great)
- Reducing per-frame inference latency (current B16 forward is fine for Teams)
- Specialist deeplive-only model (memory `project_deeplive_specialist_option.md`) — a possible **separate product**, NOT a path to the main Teams detector

---

## Explicit OUT-of-scope (do not optimize for)

- **Non-Teams platforms** (Zoom, Meet, etc.) — different capture pipelines, different shortcuts. Out of scope for this model.
- **Mobile-phone capture / cell-phone client** — different camera characteristics. Out of scope.
- **Pre-recorded video forensics** — different problem class (frame-history reasoning, source-attribution). Out of scope.
- **Source-attribution** ("which method produced this fake") — output is binary `prob_fake`, not method ID.
- **Ensemble approaches** of any shape (see "NO ENSEMBLE" above).
- **Training-time policies that are actually deployment policies** — IQ gating is a deployment policy, not a training loss.

---

## Maintenance discipline

- Edits to this file require user review (per the authoring discipline above). The doc directs GPU spend; uncontrolled drift would direct future agents at the wrong target.
- New IQ-robustness axes get added to the Pillar 3 table and to the threads index; do NOT delete axes as "solved" without a thread that proves the axis no longer matters.
- The deployed model name (E2B_TOP_N_STEP3200) and the production anchor (P8A_REFERENCE_STEP5000) are **versioned facts** — when they change, update this doc AND the corresponding memory entry AND `arena/launch_teams_promotion_contract.sh` (`PRODUCTION_ANCHORS`) AND `SCORECARD_GUIDE.md` in the same commit.

## Companion docs

- [`SCORECARD_GUIDE.md`](SCORECARD_GUIDE.md) — operating manual for promotion-contract scorecards (when to use iterative / trajectory / full mode). Read this before assembling a scorecard.
- [`AGENT_GUIDE.md`](AGENT_GUIDE.md) — pre-proposal validation checklist (read before proposing any packet).
- [`AGENTS.md`](AGENTS.md) — agent onboarding overview.

---

## Revision history

- **2026-05-08** — initial authorship after user request following P2 eyeball + scorecard launch. Drafted by agent, reviewed by user. Captures: three pillars (per memory `project_success_criteria.md`), B16-production / L14-capacity-test, deployment ≡ E2B, P8A ≡ substrate-invariance anchor, NO-ENSEMBLE rule (per user instruction + Job 12 evidence), FPR budget, deployment-time IQ gate (per user eyeball memory `project_canary_below_production_resolution_2026-05-08.md`), out-of-scope items.
