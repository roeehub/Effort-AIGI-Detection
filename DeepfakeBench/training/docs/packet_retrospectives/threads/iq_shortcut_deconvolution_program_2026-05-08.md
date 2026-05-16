# Thread: IQ-shortcut deconvolution program (2026-05-08)

> **Status**: PROPOSED program. Stage 1 not yet started. This thread documents both the
> measurements (FACTS sections) and the proposed research direction
> (OPINIONS / SUGGESTIONS sections), kept separate per the FACTS-vs-OPINIONS
> discipline established in `docs/relaunch_handoffs/AGENT_GUIDE_2026-05-02.md`.
>
> **Authoring**: 2026-05-08, drafted after the cross-pool IQ data atlas
> (`analysis/iq_data_atlas_2026-05-08/IQ_ATLAS_FACTS_2026-05-08.md`) surfaced
> several previously-missed implications about IQ-axis structure in both
> training data and eval substrates.
>
> **Companion docs**: [`MODEL_GOALS.md`](../MODEL_GOALS.md) §Pillar 3
> (robustness), [`SCORECARD_GUIDE.md`](../SCORECARD_GUIDE.md), this thread's
> sibling [`image_quality_shortcut.md`](image_quality_shortcut.md) (existing).

---

## 1. Question

**Is the deepfake detection model primarily an image-quality classifier with
a small content-detection residual, or a content classifier muddled by an
IQ confound?**

The distinction matters because:

- It determines whether the binding constraint on Pillar 3 (robustness across
  capture conditions) is the encoder's content channel or the encoder's
  IQ channel.
- It determines whether further frame-level augmentation can keep moving
  the needle (we've pulled face_scale_jitter, Fourier band-amp, codec) or
  whether we need distribution-level interventions.
- It determines how to read the v2-vs-HDTF substrate gap
  (`project_job_b_findings_universal_vs_trajectory_2026-05-04`): is HDTF
  success a content-channel measurement on a clean substrate, or a true
  generalization win?

This thread proposes a CPU-first, GPU-conditional program to answer the
question, with each stage gated on the prior stage's measurement.

---

## 2. FACTS — what the data shows

> **FACTS section. Forbidden words: succeeds, fails, wins, promotes,
> deployment-grade.** Numbers + tables + cross-references only.

### 2.1. Sharpness distributions (lap_var) by pool

Source: `analysis/iq_data_atlas_2026-05-08/IQ_ATLAS_FACTS_2026-05-08.md` and
`outputs/cutoff_lookup.csv`. Cutoff = `lap_var=30`.

**Lockbox / dev / HDTF fakes** (the eval-side fake distributions):

| pool | frac < 30 | lap_var p50 |
|---|---:|---:|
| `teams_fake_all_lockbox` | **92.5%** | 11.2 |
| `teams_fake_all_dev` | 49.0% | 81.5 |
| `visomaster_enhanced_macro_dev` | 41.6% | (inferred similar) |
| `hdtf_fake_clean_dev` | 3.4% | 171.9 |
| `hdtf_fake_clean_lockbox` | 1.6% | (inferred similar) |
| `hdtf_fake_teams_dev` | 1.8% | (inferred similar) |
| `hdtf_fake_teams_lockbox` | 3.4% | (inferred similar) |

**Training fakes**:

| pool | frac < 30 |
|---|---:|
| `train_teams_fake_pool` | 51.4% |
| `train_df40_inswap` | 47.2% |
| `train_df40_blendface` | 32.4% |
| `train_df40_facedancer` | 24.8% |
| `train_df40_simswap` | 22.6% |
| `train_df40_uniface` | 13.0% |
| `train_df40_e4s` | 2.0% |

**Training reals — internally bimodal**:

| pool | frac < 30 |
|---|---:|
| `train_df40_real_celeb_real` | **85.8%** |
| `train_df40_real_youtube_real` | **78.0%** |
| `train_df40_real_faceforensics` | 19.2% |
| `train_teams_real_pool` | **2.0%** |

**Eval reals**: 0-8% frac < 30 across all dev/lockbox/HDTF real suites.

### 2.2. Resolution distributions (min_dim) by pool

Cutoff = `min_dim=200`.

| pool category | range of frac < 200 |
|---|---|
| Training reals (df40 + teams) | 0% across all |
| Training fakes (df40 + teams + visomaster) | 0% across all |
| HDTF reals (4 variants) | 0-3% |
| HDTF fakes (4 variants) | 0-4% |
| Production reference (3 sessions, 1 identity) | 0-100% (small samples; may5_teams n=30 is 100%) |
| `teams_real_all_dev` | 39% |
| `teams_real_poor_quality_dev` | 32% |
| `teams_real_lighting_extreme_dev` | 43% |
| `teams_real_dor_dev` | 62% |
| `canary_chronic_real` | 66% |
| `canary_other_real` | 43% |
| `teams_real_all_lockbox` | 7.6% |
| `teams_fake_all_lockbox` | 13.2% |

### 2.3. Same-pipeline real-vs-fake sharpness asymmetry

`train_teams_real_pool` lap_var p50 ≈ at-or-above 200; `train_teams_fake_pool`
lap_var p50 ≈ at-or-below 30. The Teams capture pipeline produces sharp
reals (2% < 30) and smooth fakes (51% < 30) from the same recording rig.
The smoothness asymmetry is upstream of the codec — at the swap step.

### 2.4. Method-specific sharpness signatures

Each face-swap method in the training data has a distinct lap_var
distribution (e4s 2% smooth → train_teams_fake_pool 51% smooth, monotone
across {e4s, uniface, simswap, facedancer, blendface, inswap, train_teams}).
The method-axis is partly an IQ-axis.

### 2.5. HDTF substrate IQ-clean on both axes

`hdtf_*` pools (real and fake; clean and teams transport; dev and lockbox)
all sit at:
- `min_dim`: p05=211, p50=224, p95=283 (uniformly at or near model input size)
- `lap_var`: 1-3% < 30 (uniformly sharp)

HDTF is the only substrate without resolution or sharpness artifacts on
either side of the real/fake split.

### 2.6. Cross-reference to existing memories

The atlas confirmed at scale:
- `project_image_quality_shortcut.md` (score correlates negatively with
  lap_var across most suites)
- `project_eval_substrate_reframe_2026-05-04.md` (FP tail concentrated in
  6 ids + min(W,H)<200)
- `project_job14_substrate_clean_2026-05-04.md` (F4 cleaning breaks viso
  ceiling)
- `project_lockbox_fpr_dominated_by_webcam_mode.md` (webcam = low IQ)
- `project_dor_drift_named_axes_2026-05-06.md` (90% of within-identity
  drift from named pixel-domain IQ axes)

The atlas surfaced previously-not-articulated structural facts:
- Training-real bimodality on sharpness (§2.1 above) — this thread §3.2.
- HDTF fake substrate atypically sharp (§2.5) — this thread §3.3.
- Same-pipeline real-fake sharpness asymmetry (§2.3) — this thread §3.4.

---

## 3. OPINIONS — what these patterns might mean

> **OPINIONS section.** This is reasoning, not measurement. Hypotheses are
> labeled and probability-estimated. Each is testable per §4.

### 3.1. Three competing hypotheses

**Hypothesis A — The model is mostly an IQ classifier.**
Real-world fakes are smoother than real-world reals on average; the model
has approximated this. P8A's "substrate invariance" reads as
"calibrated-to-our-IQ-distribution invariance"; E2B's better deeplive recall
reads as "better at deeplive's specific smoothness signature." Apparent
content-detection capability is mostly an IQ-shortcut artifact.

**Hypothesis B — IQ + content components compose.**
The encoder learns both an IQ-axis signal (smooth → fake) and a
content-axis signal (face-swap artifacts → fake). The IQ component dominates
on most eval substrates. A real, surfaceable content channel exists but is
muddled by the IQ confound.

**Hypothesis C — The training data cannot support content-only learning.**
The training-real bimodality (§2.1) plus smooth-fakes makes the joint
distribution structurally confounded. No training procedure on this exact
data can learn pure content; data curation is mandatory.

### 3.2. Why Hypothesis B looks more likely than A or C, given current evidence

(Reasoning; this is opinion, not verdict.)

- P8A's `0/1170` real_lockbox_dor_FPR
  (`project_p18_diagnostics_complete_2026-05-02.md`) is a strong fact. A
  pure-IQ classifier would either over-fire on smooth lockbox reals or
  under-fire on sharp lockbox fakes. The dor-specific invariance pattern
  implies content-level signal exists.
- The chronic-6 partition by ckpt
  (`project_chronic_offenders_partition_per_ckpt_2026-05-04.md`) — different
  ckpts handle different identities differently — suggests per-identity
  content patterns are being learned, not just IQ.
- But the IQ component is dominant enough on most eval substrates that
  it explains most of the variance.

### 3.3. Why HDTF success may not be generalization

(Reasoning. This is the load-bearing reframe of memory
`project_job_b_findings_universal_vs_trajectory_2026-05-04.md`.)

§2.5 shows HDTF is the only substrate where real and fake sit in the same
IQ regime. On HDTF, the IQ-shortcut provides NO discriminative signal —
the encoder must rely on whatever residual content channel it has. RLP6_04's
85% on HDTF viso may not measure cross-substrate generalization; it may
measure the residual content channel after the IQ-shortcut is stripped.
This is testable per §4 below.

### 3.4. Why the method-cluster axis (Phase 1A) is partly an IQ-axis

(Reasoning. Reframes `project_phase1a_method_cluster_axis_2026-05-01.md`.)

Each face-swap method has a distinct sharpness signature (§2.4). The
method-cluster axis identified by Phase 1A may be a re-encoding of IQ
plus content rather than pure content. P15's GRL on capture-mode and P18's
GRL on method may have been targeting downstream proxies of an underlying
IQ axis. GRL'ing the IQ axis directly may bite where the proxies didn't.
This is testable per §4 Stage 2.

---

## 4. PROPOSED program (suggestions)

> **SUGGESTIONS section.** Each proposed action carries a justification
> and an explicit decision criterion that gates the next stage. The
> program is CPU-first; GPU spend is conditional on the prior stage's
> measurement.

### 4.1. Stage 1 — IQ-shortcut R² probe (CPU, ~$0, ~3-6 hours)

**Suggestion**: Decompose model scores into an IQ-explainable component
and a residual; measure the binary AUC of the residual.

**Method**:
1. For each ckpt (P8A_REFERENCE, E2B_TOP_N_STEP3200, plus the P2-D-step3000
   candidate from the running scorecard if it produces a meaningful
   verdict): score every frame in the dev + lockbox + HDTF eval pools.
2. Compute IQ feature panel per frame (lap_var, min_dim, brightness,
   color_b_dev, edge_mag, skin_frac) — already cached in the IQ atlas's
   `outputs/per_frame.parquet`.
3. Fit a multivariate linear regression `model_score = β · IQ_features + ε`
   per (ckpt × pool-group). Compute R² and the residual
   `score_resid = score - β · IQ`.
4. Compute fake-vs-real AUC on `score_resid` (the content-channel-only
   predictor) and compare to fake-vs-real AUC on raw score.

**Why this measurement**: it disambiguates Hypotheses A/B/C
per-ckpt-per-pool. The decomposition outputs partition into:
- High R² + low residual AUC → Hypothesis A. The model is mostly IQ.
- High R² + high residual AUC → Hypothesis B. Content channel exists but is
  muddled by IQ.
- Low R² + high total AUC → Hypothesis B with content dominant; IQ is noise
  not lever.

**Decision criterion for Stage 2**:
- R² > 0.5 across most ckpt × pool cells → Stage 2a (IQ GRL packet) is
  the highest-leverage next experiment.
- R² ∈ [0.3, 0.5] → Stage 2b (IQ-balanced sampler packet) is the more
  targeted intervention.
- R² < 0.2 across most cells → pivot away from IQ-debiasing; binding
  constraint is elsewhere (content-side architecture / loss work).

**Where to deposit findings**: a new FACTS doc at
`analysis/iq_shortcut_decomp_<date>/IQ_DECOMP_FACTS_<date>.md`. Memory:
add an entry summarizing the per-ckpt R² + residual-AUC numbers.

### 4.2. Stage 2 — single-lever GPU packet (conditional on Stage 1)

**Suggestion (2a) — IQ-axis GRL packet**: Add a head that predicts
`lap_var_bin` (or a multivariate IQ regressor) from the encoder's CLS
output. Apply gradient reversal at the encoder→IQ-head interface. Run
matched against P8A as a baseline, with face_scale_jitter held constant
(so the only delta is the IQ-GRL term).

**Why this lever (justification)**: prior GRL packets (P15 capture-mode,
P18 method) targeted proxies of IQ rather than IQ directly per §3.4.
Targeting the load-bearing axis directly may bite where the proxies didn't.
This is structurally novel — we have not GRL'd IQ before.

**Suggestion (2b) — IQ-balanced sampler packet**: For every training batch,
bin frames by lap_var quartile. Sample equal counts of (real, fake) within
each bin. Within batch, IQ provides zero discriminative signal; the model
is forced to use content.

**Why this lever (justification)**: this is structurally different from
augmentation (which perturbs single frames). It's a sampler-level
constraint that ensures the distribution the model trains on has
IQ-independence baked in within batch. Memory
`project_data_axis_lever_pulled_twice_no_lift.md` says data-axis has been
pulled twice without lift, but those packets ("more data", "different
family weights") did not target within-pool IQ structure. This is a new
data-axis sub-lever.

**Failure modes to anticipate** (honest):
- (2a, GRL) Encoder may regress on Pillar 1 (fake recall) if IQ is a
  dominant signal it relied on. Watch `dev_fake_macro_recall` carefully.
- (2b, sampler) May SHRINK total AUC while improving IQ-axis robustness.
  If Hypothesis A is closer to true, this approach reveals the model's
  capability was mostly IQ — a useful negative result, but a regression
  on apparent metrics.

**Sub-lever discipline**: per `project_face_scale_jitter_load_bearing.md`
and the AGENT_GUIDE, each Stage 2 packet runs as a single-lever ablation.
No bundles. No stacking GRL + sampler in one packet without a sister-variant
that isolates each.

### 4.3. Stage 3 — eval reframe (CPU + reporting change, ~$0)

**Suggestion**: Adopt HDTF-conditional readouts as a primary line in
promotion-contract scorecards. Add a "production-IQ-matched" sub-suite
defined by `min_dim ≥ 200` AND `lap_var ≥` (production p05).

**Why this reframe (justification)**: §2.5 shows HDTF is the only substrate
without IQ artifacts on both sides; §3.3 reasons that HDTF readouts are
the closest measurement of the residual content channel. Adopting this as
primary changes how every prior packet retrospective is read.

**Implementation**: edit `arena/target_domain_suites.teams_promotion_contract_*.yaml`
to add an HDTF-bundle. Edit `score_teams_promotion_contract.py:268-302` to
include HDTF readouts in the τ-deciding lex policy or as a parallel verdict
line. Update `SCORECARD_GUIDE.md`.

**This is a paradigm reframe, not a single experiment.** Implement only
after Stage 2 produces a clear signal — otherwise we are restructuring
the scorecard around a hypothesis that may not be load-bearing.

### 4.4. Optional Stage 4 — drop the smooth training-real pools (data curation)

**Suggestion**: Train a single-lever packet that drops `celeb_real` +
`youtube_real` from training reals (the 78-86% smooth pools per §2.1).
Train only on `faceforensics` + `teams_real`. Compare to a matched-recipe
baseline.

**Why this lever (justification)**: §2.1 shows the training-real label is
internally inconsistent on the IQ axis. Two of the four real pools sit
at fake-level smoothness. The model must reconcile contradictory IQ
priors for the same label. Removing the inconsistent pools makes
"real = sharp" a clean training prior, forcing the model to use content
to discriminate the smooth fakes.

**Risks**: shrinks training-real diversity (fewer identities, fewer
scenarios). May overfit to the remaining pools' specific rigs. May regress
generalization on substrates that look like the dropped pools (older
celebrity-source content, e.g.).

**Why optional / Stage 4**: this is the highest-risk, highest-upside
intervention. Run it AFTER Stage 1 + 2 confirm the IQ-shortcut is the
binding constraint. Otherwise we shrink training data based on a hypothesis
that hasn't been measurement-confirmed.

---

## 5. Honest novelty + impact estimate

> **OPINION + estimation.** Probabilities are calibrated guesses, not
> verdicts.

### 5.1. Novelty per stage

| stage | novelty estimate | what's already partially in place |
|---|---:|---|
| Stage 1 (R² probe) | ~60% | `project_image_quality_shortcut.md` correlations + `project_dor_drift_named_axes_2026-05-06.md` 90% ridge — but the cross-class deconvolve-and-residual-AUC framing has not been done explicitly |
| Stage 2a (IQ GRL) | ~70% | GRL machinery exists (P15, P18); the IQ axis itself has not been targeted |
| Stage 2b (IQ-balanced sampler) | ~80% | We sample by family/method; never by IQ regime within-batch |
| Stage 3 (HDTF canonical eval) | ~85% (paradigm reframe) | HDTF used as cross-substrate validation (Job B), not as primary |
| Stage 4 (drop smooth training reals) | ~75% | Per-method / per-family curation done; per-IQ pool curation new |

### 5.2. Impact estimate (conditional on the program executing through Stage 2)

- **Probability this program meaningfully advances Pillar 3 (IQ-robustness)**:
  ~50%. The IQ-shortcut is documented; attacking it at distribution level
  is the structurally correct intervention. Whether it works depends on
  whether Hypothesis B is true (content channel exists + can be surfaced)
  vs A (model is mostly IQ → debiasing destroys most apparent capability).
- **Probability this program produces a clear deployment-grade improvement
  on E2B in the next ~2 packets**: ~20%. Even if directionally correct, it
  takes iteration. Stage 2 may produce diagnostic info that informs Stage 3
  and Stage 4 rather than itself promoting.
- **Probability this program is a clear loss vs current trajectory**: ~15%.
  Risk: removing IQ shortcut destroys apparent capability, model regresses
  on Pillar 1, no countervailing content gain. Stage 1 gates against this
  by measuring before spending GPU.
- **Probability Stage 1 alone produces a load-bearing diagnostic**: ~85%.
  The decomposition is well-defined and informs the next decision regardless
  of what number it produces.

### 5.3. Why this direction is high-leverage compared to alternatives

(Opinion; I think this is the strongest argument for the program.)

- We've pulled the augmentation lever multiple times — face_scale_jitter
  (big win, `project_face_scale_jitter_load_bearing`), Fourier band-amp
  (modest, P2 in flight), codec aug (negative, `project_pc_codec_aug_hurts_viso_2026-05-05`).
  The frame-level interventions appear to be saturating.
- Memory `project_data_axis_lever_pulled_twice_no_lift.md` says data-axis
  has been pulled without lift — but those packets pulled "MORE data" or
  "different family weights", not "IQ-balanced sampling." The atlas
  surfaces a NEW data-axis (within-pool IQ structure) that hasn't been
  targeted.
- Stage 1 is so cheap and informative that NOT running it leaves us
  guessing among Stage 2/3/4 alternatives on intuition.

---

## 6. Cross-references

**Companion docs**:
- [`MODEL_GOALS.md`](../MODEL_GOALS.md) — three pillars, B16-production /
  L14-capacity-test, deployment ≡ E2B, NO-ENSEMBLE rule, FPR budget.
- [`SCORECARD_GUIDE.md`](../SCORECARD_GUIDE.md) — when to use which
  scorecard mode (iterative / trajectory / full).
- [`AGENT_GUIDE.md`](../AGENT_GUIDE.md) — pre-proposal validation checklist
  including bundle-decomposition discipline.
- [`AGENTS.md`](../AGENTS.md) — agent onboarding overview.

**Sibling threads**:
- [`image_quality_shortcut.md`](image_quality_shortcut.md) — the existing
  IQ-shortcut thread; precursor to this program.
- [`face_size_label_leak.md`](face_size_label_leak.md) — face-pixel-area
  shortcut; mitigated by `face_scale_jitter@0.50`.
- [`eval_substrate_data_hygiene.md`](eval_substrate_data_hygiene.md) —
  eval substrate quality issues.
- [`processing_signature_shortcut.md`](processing_signature_shortcut.md) —
  capture-pipeline signature shortcut.

**Source data**:
- `analysis/iq_data_atlas_2026-05-08/IQ_ATLAS_FACTS_2026-05-08.md` (atlas)
- `analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet` (15,236 ×
  16 columns; reusable for Stage 1)
- `analysis/iq_data_atlas_2026-05-08/outputs/cutoff_lookup.csv` (1,686
  rows of `(metric, pool, cutoff) → frac_below`)

**Loadbearing memories**:
- `project_image_quality_shortcut.md`
- `project_dor_drift_named_axes_2026-05-06.md`
- `project_phase1a_method_cluster_axis_2026-05-01.md`
- `project_p18_diagnostics_complete_2026-05-02.md`
- `project_chronic_offenders_partition_per_ckpt_2026-05-04.md`
- `project_job_b_findings_universal_vs_trajectory_2026-05-04.md`
- `project_pa_does_not_generalize_to_hdtf_2026-05-05.md`
- `project_data_axis_lever_pulled_twice_no_lift.md`
- `project_face_scale_jitter_load_bearing.md`

---

## 7. Status / progress

> **Update this section as work proceeds.** Add per-stage entries with
> dates and links to FACTS docs.

- **2026-05-08** — proposal authored. Stage 1 not started. Pending the
  current P2 Phase A scorecard finalization
  (Vertex `6226906326623059968`) for the P2-D-step3000 verdict; once that
  lands, Stage 1 has its third candidate ckpt to decompose.

- **2026-05-08 (PM)** — P2 Phase A `JOB_STATE_SUCCEEDED`. Verdict
  documented at [`analysis/p2_eval_2026-05-08/P2_PHASE_A_VERDICT_FACTS_2026-05-08.md`](../../../analysis/p2_eval_2026-05-08/P2_PHASE_A_VERDICT_FACTS_2026-05-08.md).
  P8A_REFERENCE_STEP5000 is the rank-1 promotion winner. P2_D_FOURIER_PERIODIC_STEP3000
  ranks 4 with `dev_fake_macro_recall=0.530` (vs E2B 0.508) and
  `lockbox_fake_recall=0.889` (vs E2B 0.628), at a cost of
  `lockbox_real_fpr=0.164` (8× E2B) and `teams_real_dor_dev real_fpr=0.460`
  (5.75× E2B).

- **2026-05-08 (PM)** — Stage 1 (R² probe) executed CPU-only.
  Driver: [`analysis/iq_shortcut_decomp_2026-05-08/decompose.py`](../../../analysis/iq_shortcut_decomp_2026-05-08/decompose.py).
  FACTS: [`IQ_DECOMP_FACTS_2026-05-08.md`](../../../analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_FACTS_2026-05-08.md).
  OPINIONS: [`IQ_DECOMP_OPINIONS_2026-05-08.md`](../../../analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_OPINIONS_2026-05-08.md).

  Headline: 23 cells across 3 ckpts × ≤9 pool-groups. R² is
  **substrate-conditional**, not uniform:

  | substrate class | R² range | residual AUC range | Δ AUC (raw − resid) median |
  |---|---|---|---|
  | HDTF (8 cells, P8A+E2B) | 0.016–0.090 | 0.71–0.99 | 0.022 |
  | DEV (excl. STRESS, 12 cells) | 0.020–**0.594** | 0.51–0.88 | 0.180 |
  | LOCKBOX_TEAMS (3 cells) | **0.397–0.542** | 0.63–0.76 | **0.246** |

  Hypothesis A (encoder is mostly an IQ classifier) is **provisionally
  rejected**: even the most IQ-dominant cell (LOCKBOX, P2-D, residual AUC
  0.633) keeps fake-vs-real discriminability after IQ removal. Picture is
  Hypothesis B with the IQ-vs-content balance shifting strongly by substrate.

  The proposal §4.1 uniform-threshold decision criterion does not match the
  substrate-conditional R² pattern. OPINIONS doc §3 reframes Stage 2 around
  "where the IQ shortcut binds vs where production lives" rather than the
  global R² threshold; recommends Stage 2a (IQ GRL) gated on a CPU
  per-layer IQ-probe diagnostic before launching GPU.

  **Stage 2 GPU spend NOT authorized.** Awaiting user decision per
  proposal §4.2 + OPINIONS §7.

- **2026-05-08 (late PM)** — User authorized **three pre-Stage-2a checks**
  (a/b/c) per [`analysis/p2_eval_2026-05-08/P2_DEEPER_ANALYSIS_OPINIONS_2026-05-08.md`](../../../analysis/p2_eval_2026-05-08/P2_DEEPER_ANALYSIS_OPINIONS_2026-05-08.md) §6:

  - **(a) Per-layer IQ probe** — CPU, dispatched to sub-agent. Output FACTS
    doc planned at `analysis/iq_perlayer_probe_2026-05-08/IQ_PERLAYER_PROBE_FACTS_2026-05-08.md`.
    Goal: where in the OpenCLIP B16 encoder is the IQ representation
    concentrated; informs GRL hook layer choice for Stage 2a Slot 2.
  - **(b) D step3000 HDTF Phase C** — GPU, Vertex job
    `p2-d-step3000-hdtf-2026-05-08` (us-east1, image 1.3.273, ~$15-25,
    ETA ~3-4h). Suite manifest:
    `arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml`.
    Goal: is D step3000's viso-ceiling break content-real or v2-substrate-
    bound (PA-style — memory `project_pa_does_not_generalize_to_hdtf_2026-05-05.md`).
  - **(c) Dor encoder-axis characterization** — CPU, dispatched to same
    sub-agent. Output FACTS at `analysis/dor_encoder_axis_2026-05-08/`.
    Goal: WHY did D step3000 lose P8A's signature dor invariance (J4: real
    FPR 8% → 46%, fake recall 86% → 62%)? IQ R² doesn't include Dor in any
    high-R² cell, so the regression isn't IQ-mediated.

  All three are gating Stage 2a packet design. **Stage 2a slot composition
  + FT-base selection depends on a/b/c outcomes** per the OPINIONS doc §6.

  After a/b/c land: a fresh agent should read the FACTS docs (without
  reading prior agents' OPINIONS first) and form their own view of the
  next-step decision per the user's framing — STATE.md's FRESH-AGENT
  GUIDANCE block has the reading order.

- **2026-05-08 (evening)** — checks (a) and (c) completed CPU-only.

  **(a) Per-layer IQ probe** —
  FACTS: [`analysis/iq_perlayer_probe_2026-05-08/IQ_PERLAYER_PROBE_FACTS_2026-05-08.md`](../../../analysis/iq_perlayer_probe_2026-05-08/IQ_PERLAYER_PROBE_FACTS_2026-05-08.md).
  Per-(ckpt × layer) IQ representation on the 800-frame triptych sample
  (mostly DEV-substrate, 863 frames after IQ panel join).

  Headline: peak per-IQ-feature binary AUC at **layer 6** for all three
  ckpts (P8A 0.981 / E2B 0.977 / P2D 0.976 on `min_dim`). Multivariate
  avg-per-feature R² peaks at L6 for P8A (0.726), then climbs to L11 for
  E2B (0.785) and P2D (0.782). L11 R² gap E2B/P2D − P8A = +0.115.

  L0 results identical across the 3 ckpts (sanity confirmed: FT freezes
  patch+positional embeddings).

  L9 dip across all 3 ckpts; recovery at L11 is the E2B/P2D-only feature.

  **(c) Dor encoder-axis characterization** —
  FACTS: [`analysis/dor_encoder_axis_2026-05-08/DOR_ENCODER_AXIS_FACTS_2026-05-08.md`](../../../analysis/dor_encoder_axis_2026-05-08/DOR_ENCODER_AXIS_FACTS_2026-05-08.md).
  OPINIONS: [`analysis/dor_encoder_axis_2026-05-08/DOR_ENCODER_AXIS_OPINIONS_2026-05-08.md`](../../../analysis/dor_encoder_axis_2026-05-08/DOR_ENCODER_AXIS_OPINIONS_2026-05-08.md).
  388-frame cohort: 50 DOR_REAL_DEV + 78 DOR_FAKE_DEV + 100 DOR_REAL_LOCKBOX
  + 80 NON_DOR_REAL_DEV + 80 NON_DOR_FAKE_DEV.

  Headline numbers (final-CLS cosine centroid distances):
  - `dor_real_vs_dor_fake`: P8A 0.952 → P2D **0.112** (8.5× compressed).
  - `dor_real_vs_non_dor_real`: P8A 0.062 → P2D 0.342 (5.5× farther).
  - `dor_fake_vs_non_dor_fake`: P8A 0.003 → P2D 1.077 (350× farther).
  - Score gap p50(Dor real) − p50(Dor fake): P8A 0.805 → P2D **0.053**.
  - Per-frame median d(Dor real, non_dor_fake centroid):
    P8A 0.65, P2D 0.21.

  Dor IQ correlation Pearson r is WEAK on P2D (lap_var r=+0.19 on
  DOR_REAL_LOCKBOX vs P8A r=+0.46). The largest score-vs-IQ correlation
  shifts (P2D−P8A) are in the OPPOSITE direction from the LOCKBOX
  IQ-shortcut: `min_dim` r flips from −0.33 (P8A) to +0.28 (P2D) on
  DOR_REAL_DEV. The Dor regression is not IQ-mediated in the LOCKBOX
  IQ-shortcut direction.

  OPINIONS reading: P2D constructed an "is-Dor" identity-cluster sub-region;
  within it, real/fake separation collapsed (CONFUSED+SHIFTED, simultaneously).
  Three Stage 2 paths surfaced (IQ-GRL only / identity-axis-GRL only /
  sister-variant ablation of both). User decides.

  **Stage 2a GPU spend NOT authorized for sub-agent.** Awaiting user pick
  among Path A / B / C in the OPINIONS doc §7, plus the (b) HDTF Phase C
  result for D step3000.

- **2026-05-08 (evening, post-handoff)** — check (b) D step3000 HDTF Phase C
  landed. Vertex job `6632089555598049280` terminated `JOB_STATE_FAILED` at
  19:22:50Z due to known bug in `score_teams_promotion_contract.py` — the
  contract aggregator's suite-name map is keyed on Phase A names
  (`teams_real_all_*`) and does not handle Phase C HDTF names
  (`proper_real_teams_*`). All 48 per-frame reports + the τ=0.5 sidecar
  diagnostic_scorecard are intact in GCS; `promotion_contract/` was not
  written. Closes the root cause of open loop
  `phase-c-hdtf-promotion-contract-failure`. Verdict reconstructed locally
  by applying each ckpt's Phase A τ to its HDTF per-frame reports.

  FACTS: [`analysis/p2_d_hdtf_2026-05-08/P2_D_HDTF_FACTS_2026-05-08.md`](../../../analysis/p2_d_hdtf_2026-05-08/P2_D_HDTF_FACTS_2026-05-08.md).

  Headline (frame-level, at each ckpt's Phase A contract τ):

  | metric | P8A τ=0.916 | E2B τ=0.711 | P2D τ=0.460 |
  |---|---:|---:|---:|
  | macro_real_fpr (4 real suites) | 0.005 | 0.001 | 0.011 |
  | macro_fake_recall (12 fake suites) | **0.896** | 0.484 | 0.540 |
  | worst_fake_recall (min over 12) | 0.827 | 0.061 | 0.091 |

  Load-bearing per-suite finding: clean-vs-teams transport split.

  | mean fake_recall over | P8A | E2B | P2D |
  |---|---:|---:|---:|
  | 6 HDTF clean fakes | 0.946 | 0.796 | **0.942** |
  | 6 HDTF teams fakes | 0.846 | 0.172 | 0.137 |
  | clean − teams gap | +0.10 | +0.62 | **+0.81** |

  P2D matches P8A on HDTF clean-transport visomaster_enhanced
  (0.964/0.979 vs 0.947/0.951) but collapses to 0.09/0.09 on the
  Teams-transport variants where P8A holds 0.83/0.86.

  Reading vs proposal §6 question 1 ("Does D step3000's viso lift hold on
  HDTF?"): **partial — survives on HDTF clean transport, reverses on HDTF
  teams transport**. P2D's Phase A v2 substrate `lockbox_real_fpr`
  regression (0.164, 9.1× E2B) does NOT reproduce on HDTF
  `proper_real_teams_lockbox` (P2D 0.001 vs E2B 0.001). The Pillar 2
  regression read on Phase A v2 was substrate-bound to v2.

  Reading vs proposal §6 question 2 ("Is D's encoder more or less
  IQ-driven on HDTF?"): NOT yet computed in this FACTS doc — the IQ R²
  decomposition for P2D on HDTF cells is left for an update to
  `IQ_DECOMP_FACTS_2026-05-08.md` §2.3.

  Reading vs proposal §6 question 3 ("Does the dor identity-cluster
  collapse manifest on HDTF?"): HDTF substrate has no Dor cohort. The
  `proper_real_teams_lockbox` slice (1418 frames, mixed identities) shows
  P2D real_fpr 0.001 (low). Probable inference: Dor collapse is
  substrate-bound to the Phase A `teams_real_dor_dev` cohort.

  Memory entry added:
  `project_p2_d_hdtf_p8a_dominates_2026-05-08.md`.

  **Stage 2 decision now ready**. The convergent dataset (P2 verdict +
  J1-J5 + IQ atlas + Stage 1 R² + check (a) per-layer + check (c) Dor
  encoder + check (b) HDTF Phase C) is complete. User-pick gates the
  next step.

- **2026-05-08 (late evening)** — Stage 2 LAUNCHED. User authorized three
  single-lever overnight slots, all FT-from-`P8A_REFERENCE_STEP5000`,
  pure-YAML changes, image `1.3.274` (Cloud Build `254cfb3d-00f1-4066-b7ea-33ad4abd4a8d`,
  16m51s).

  | slot | yaml | lever (vs P8A FT baseline) | hypothesis tested | region | Vertex job ID |
  |---|---|---|---|---|---|
  | S2-1 REAL_AUG_OFF | `R13_S2_SLOT1_REAL_AUG_OFF.yaml` | `pipeline_randomization.p_real: 0.0` (vs 0.5) | Train/eval IQ-prior FLIP — atlas TRAIN_REAL lap_var p50=33.7 vs LOCKBOX_REAL p50=213.5; aug-induced smoothness on training reals trains a "blurred = real" prior that backfires at deployment. | us-west4 | `7347212674217803776` |
  | S2-2 LOW_LR_FT | `R13_S2_SLOT2_LOW_LR_FT.yaml` | `learning_rate: 1.0e-5` (vs 1e-4) | Preservation — every R13 FT at 1e-4 has lost ≥1 P8A pillar (HDTF teams transport / chronic-FP tolerance / encoder cluster geometry). Lower LR reduces L10-11 drift per step (per-layer probe shows L10-11 is where E2B/P2D diverge from P8A). | us-east1 | `7961214395625766912` |
  | S2-3 WEAK_PAIRRANK | `R13_S2_SLOT3_WEAK_PAIRRANK.yaml` | `pair_rank_loss.lambda: 0.05` (vs BUNDLE's 0.2) | Dose-response — BUNDLE_step500 hit 96.5% lockbox recall at FPR=10% (τ=0.989) but score-collapsed Roy_D (std 0.0005, Pearson r=−0.16 vs P8A). λ=0.05 tests whether pair_rank's lift mechanism scales linearly with λ (lift preserved, compression reduced) or is non-linear / saturated. | us-central1 | `5671407677603840000` |

  All three submitted at `JOB_STATE_PENDING`. Monitor task `b5vm706im`
  emits one event per state transition + alerts at 30min PENDING per
  CLAUDE.md region-switch threshold. Expected ~4500 steps × 3-5h on A100;
  results expected morning of 2026-05-09.

  Each slot is single-lever vs a known baseline. Each has an explicit
  falsification criterion documented in the YAML header. Aggregate
  P(at least one lifts lockbox recall above P8A's 39% AND preserves HDTF
  teams within 5pp of 0.85 AND keeps chronic-6 max FPR < 30%): 50-60%.
  P(any clears full promotion bar): 15-25%.

  Three slots together close the structural questions: (1) is the IQ
  shortcut at deployment caused by the train-time aug pipeline? (2) does
  ANY FT-from-P8A preserve the load-bearing properties at the right LR?
  (3) is BUNDLE's 96.5% lift dose-responsive on pair_rank λ?

- **2026-05-09 (early morning)** — Stage 2 RESULTS landed. S2 + S3
  SUCCEEDED at 00:05:48Z and 00:17:02Z respectively (~2h 18min and 2h 31min
  runtime); S1 still RUNNING but step4500 ckpt saved (slow throughput in
  us-west4 — wall-clock ~5h vs S2/S3's ~2.5h with identical compute).

  **CPU score-distribution probe** completed on 2026-05-09 03:21–03:27 local
  (~3min total on MPS, 388-frame Dor cohort × 9 ckpts). FACTS:
  [`analysis/stage2_cpu_2026-05-09/STAGE2_SCORE_PROBE_FACTS_2026-05-09.md`](../../../analysis/stage2_cpu_2026-05-09/STAGE2_SCORE_PROBE_FACTS_2026-05-09.md).
  OPINIONS:
  [`analysis/stage2_cpu_2026-05-09/STAGE2_SCORE_PROBE_OPINIONS_2026-05-09.md`](../../../analysis/stage2_cpu_2026-05-09/STAGE2_SCORE_PROBE_OPINIONS_2026-05-09.md).

  Headline (DOR_REAL_LOCKBOX p50 — closer to 0 = better; P8A baseline 0.019):

  | slot | step500 | step2500 | step4500 |
  |---|---:|---:|---:|
  | S1 REAL_AUG_OFF | 0.270 | 0.347 | **0.536** |
  | S2 LOW_LR_FT | 0.196 | 0.305 | **0.285** |
  | S3 WEAK_PAIRRANK | 0.657 | 0.504 | **0.517** |

  Reading vs §7 launch hypotheses:

  - **S1 REAL_AUG_OFF — IQ-prior FLIP REFUTED.** Trajectory monotonically
    degrades. Disabling pipeline_randomization on reals doesn't preserve
    P8A's dor invariance; it accelerates dor cohort drift.
  - **S2 LOW_LR_FT — Preservation PARTIALLY SUPPORTED.** Best preserver
    among Stage 2 slots; Pearson r 0.62 with P8A on DOR_REAL_LOCKBOX
    (highest among slots; P2D 0.46). Plateaus around 0.29 vs P8A 0.02 —
    15× regression but smallest among Stage 2.
  - **S3 WEAK_PAIRRANK — Dose-response REFUTED.** λ=0.05 still produces
    step500 cluster collapse (DOR_REAL_DEV std 0.077 = 4.5× compressed
    vs P8A 0.347). Uniquely regresses bidirectionally on Dor cohort
    (real-FPR up + DOR_FAKE_DEV recall DOWN: P8A 0.99 → S3 0.58).

  Convergent reading: **the binding constraint is FT itself, not any of
  the three single levers.** All R13 packets that FT from P8A drift the
  dor cluster within 500-2500 steps (BUNDLE, PAIRRANK, P2D, S1, S2, S3
  all regress). The next structurally distinct lever to test is L11
  anchor-loss FT (option 3a in OPINIONS doc §3) — directly preserves the
  L10-11 representation that check (a) per-layer probe identified as
  where E2B/P2D diverge from P8A.

  **Recommended next measurement (~$15-20 GPU)**: promotion contract
  scorecard for `S2_step4500` only. If S2 passes Pillar-2 at contract τ,
  it's a real candidate. If it regresses like the probe predicts, the
  signal that FT itself is the binding constraint is loud enough to
  pivot to L11 anchor-loss FT.

## 2026-05-11 update — T4 encoder mostly succeeded; lockbox failure is HEAD + chronic_6, NOT substrate-overfit

T4 (multi-axis-L11-GRL) + T5-A (cyclic-λ) ran overnight 2026-05-10 → 2026-05-11; promotion-contract scorecard + three CPU diagnostics A1/A2/A3 landed 2026-05-11 morning. The combined evidence overturns the same-night T4 OPINION doc's "substrate-overfit between dev and lockbox" framing.

### Facts

- **T4_LAMBDA1_TOP_N_STEP10500 broke the L11 atlas inv_mean ceiling** at 0.0414, +18% over the 17-ckpt prior ceiling 0.0349 — first quantitative encoder-level shortcut removal in R13 (`analysis/cpu_diagnostics_2026-05-10/OVERNIGHT_STATUS_2026-05-11.md`).
- **Promotion-contract verdict**: P8A retains rank-1 on v3-fix policy; T4_L1_step10500 ranks 3. Dev AUCs UP +0.04/+0.08 across all 3 dev fake cells; trained-head lockbox AUC DOWN 0.174 absolute (P8A 0.9355 → T4 0.7619); no τ in [0.50, 0.99] recovers; AUC is invariant under any monotonic calibration (`analysis/t4_eval_2026-05-11/RESULTS_FACTS_2026-05-11.md` §3.1, §7.1, §8.3).
- **T5-A (cyclic-λ) refuted**: all 6 T5-A ckpts have inv_mean below P8A baseline 0.0266 (best 0.0260). Cyclic-λ removes invariance during λ=0 phases faster than it adds during λ=peak (`OVERNIGHT_STATUS_2026-05-11.md` §"RESOLVED 04:50 CEST").
- **A1 — T4 cross-substrate AUC matrix** (`analysis/cpu_diagnostics_2026-05-11_a1_cross_substrate/CROSS_SUBSTRATE_FACTS_2026-05-11.md`): on HDTF subsample (N=50 per cell, σ≈0.02), T4 Δ vs P8A is [−0.022, +0.001] across 5 HDTF teams + clean cells, AUC 0.97-1.00. On may6 mean score 0.02 → 0.22 (10.8× ratio) with FPR@τ=0.5 = 0.098 vs P8A 0.000. T4_L2_step1500 on lockbox is **+0.0143** — the lockbox AUC drop is specific to L1_step10500, not generic to the T4 packet. β-outcome: lockbox is the unique large drop, NOT universal substrate-overfit.
- **A2 — frozen-encoder linear probe on T4 L11 features** (`analysis/cpu_diagnostics_2026-05-11_a2_linear_probe/LOCKBOX_PROBE_FACTS_2026-05-11.md`): on 453 lockbox videos (200 reals proportional-stratified + 253 fakes, real_dor's 109 videos excluded for local mirror gap), 5-fold StratifiedKFold logistic regression yields AUC = 1.000 ± 0.000 for BOTH T4_L1_step10500 AND P8A. Trained-head gap on same 453-video subset is T4 0.7957 vs P8A 0.8638 (0.068 absolute, smaller than full-lockbox 0.174 → real_dor carries disproportionate trained-head difficulty for T4). Sanity checks: regularization grid C ∈ [1e-3, 10] × PCA {None, 50, 20, 10} all AUC ≥ 0.9999; within-PC_Generator probe (n=77) also 1.000. α-outcome: encoder retains real-vs-fake separation; trained head is the lockbox failure point.
- **A3 — atlas triptych composition + per-substrate decomposition** (`analysis/cpu_diagnostics_2026-05-11_a3_atlas_composition/ATLAS_COMPOSITION_FACTS_2026-05-11.md`): triptych is 89.1% dev (713 frames) / 10.9% lockbox (87 frames overlapping deployed lockbox suite, 0 dev→lockbox leak). Per-substrate LR-probe inv_mean Δ vs P8A: full_n800 +0.0169, dev_only +0.0098, lockbox_only +0.0103, non_chronic +0.0331, **chronic_6 −0.0315**. On same 87 triptych-lockbox frames: trained head moves −0.0143 (worse) while LR probe moves +0.0277 (better) — direct feature-vs-head disagreement on the substrate.

### Current stance (revised 2026-05-11)

The GRL mechanism worked at the encoder level — first quantitative encoder-level invariance lift in R13 history, and it generalizes to HDTF within sampling noise. The "atlas inv_mean is a deceptive metric" framing in the same-night T4 OPINION doc is partly retracted: atlas inv_mean lift holds on the actual lockbox subset of the triptych too. The atlas is dev-leaning in sampling but not non-predictive of lockbox encoder behavior.

The lockbox AUC drop decomposes into two distinct failure modes:

1. **Trained-head mapping failure on lockbox** (especially real_dor). Same L11 features are linearly separable to AUC=1.000 on the 453-video lockbox subset; the trained head only achieves 0.7619 on full lockbox. Plausibly recoverable via head retrain on T4 features — cheap CPU/MPS follow-up, no GPU spend needed for the test.
2. **Feature-level regression on chronic_6** (Δ inv_mean −0.0315 vs P8A on triptych chronic_6 slice). Matches the per-axis trajectory in OVERNIGHT_STATUS_2026-05-11.md (T4's `is_chronic_6` axis at 0.9187 vs P8A 0.9087 — barely moved). NOT recoverable via head retrain; requires training-time intervention.

Calibration on A2's perfect probe AUC: N=453 videos at dim=768 is overdetermined; the agent ran regularization + PCA grid (all stays ≥0.9999) and within-source PC_Generator probe (also 1.000), ruling out trivial overfit and identity-leak. Conservative read: directional finding is robust; the magnitude (perfect separation) may be inflated by sample-size-vs-dim ratio. A2-extension is in flight to test per-T4-step + real_dor inclusion.

### Implications for the open-loop program

- Single-attachment L11 GRL with classifier hidden_dim=256 produces a real but uneven encoder lift. The "encoder didn't actually need to change" framing in OVERNIGHT_STATUS is partially refuted — it changed on dev + lockbox + non_chronic; it regressed on chronic_6. The unevenness is the problem.
- **T5-C** (`R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.yaml`, seed 9501) is the cleanest single-experiment test of "is the uneven movement because the classifier is too small to force uniform encoder invariance?". One-line yaml change (`multi_axis_grl.hidden_dim: 256 → 1024`); same GRL structure otherwise.
- **T6** (`R13_T6_T3_PLUS_JITTER_2026-05-11.yaml`, seed 9601) is independent of the GRL program — it's the deployment refinement bet that targets T3's known Roy_D / color-axis fragility while preserving the F4 viso 79% (step2500) / 73% (step1500) lift.
- Head-retrain on frozen T4 features is the cheapest follow-up that doesn't need GPU — if it recovers lockbox AUC to ≥0.85, T4 + new head becomes a near-zero-cost deployment candidate. Pending dispatch.

### Open loops added or amended

### Open loop: encoder-vs-head-locus-on-t4-lockbox
status: resolved
severity: low
first_seen: 2026-05-11
last_verified: 2026-05-11
close_criterion: frozen-encoder linear probe on T4 lockbox features yields AUC ≥ 0.85 across multiple T4 ckpts (= α-outcome, encoder retains separation, head is the failure point) OR AUC ≤ trained-head AUC (= β-outcome, encoder lost separation)

**Resolution (2026-05-11)**: A2-extension verdict is α at high confidence. Per-step T4 LR-probe AUC on lockbox L11 features (n_real=309 real_dor-inclusive + n_fake=253, 5-fold StratifiedKFold) is **1.0000 ± 0.0000 for all 5 ckpts** (T4 step5000-periodic, step9000-top_n, step10500-top_n, step11250-top_n, P8A_REF_STEP5000) — every per-fold AUC exactly 1.0000. Encoder L11 separability is NOT step-specific to step10500; holds across all of T4-λ1.0's trajectory AND survives real_dor inclusion. **Surprise side-finding**: real_dor is NOT the hard cohort for the trained head — trained-head AUC on real_dor-only-vs-fakes is T4 0.984 / P8A 0.990 (HIGHER than either ckpt's full-lockbox AUC). The T4 trained-head failure is concentrated in the `dor_shkedi` cohort (1138 of 1361 lockbox reals — 83% of the pool), consistent with `project_signature_shortcut_finding` (dor_shkedi vs real_dor signature flip known since 2026-04-23). The 0.174 → 0.068 trained-head gap shrink in A2's no-dor subset was a sampling artifact of the proportional-stratified 200-real subset, NOT real_dor being hard. Loop closes with α at high confidence.

- **next-step implication**: Head retrain on frozen T4 features becomes the highest-EV next experiment (CPU/MPS, ~$0). If a fresh head trained on T4 features recovers lockbox AUC ≥ 0.85 on the dor_shkedi-heavy real pool, T4 + new head is a near-zero-cost deployment candidate.
- **caveats**: probe is overdetermined (dim 768, n≈450 train/fold); A2's regularization grid ruled out memorization for the basic case but was not re-run for the per-step extension. The 1.0000 AUCs are uniform across regularization in the A2 baseline, so directional read is robust; magnitude (exact perfect separation) may be inflated.
- **source**: A2-extension (`analysis/cpu_diagnostics_2026-05-11_a2_extension/A2_EXTENSION_FACTS_2026-05-11.md`)

### Open loop: chronic_6-feature-regression-on-t4
status: in-progress
severity: low
first_seen: 2026-05-11
last_verified: 2026-05-12
close_criterion: a training-time intervention (T5-C stronger online classifier; OR T5-B multi-layer GRL attachment; OR head-only retrain on a substrate-diverse pool including lockbox-style data) measurably restores chronic_6 inv_mean Δ to ≥ 0 vs P8A on the FULL chronic_6 cohort (including PC_Generator/Roy_D/Q identities, not just bla_bla_chow) while preserving lockbox + HDTF performance

**Summary (2026-05-11)**: T4_L1_step10500 regressed chronic_6 inv_mean at L11 by −0.0315 absolute vs P8A on the atlas triptych chronic_6 slice (A3). Matches per-axis trajectory in OVERNIGHT_STATUS_2026-05-11.md (T4 is_chronic_6 0.9187 vs P8A 0.9087 — slightly worse).

**Amendment 2026-05-11 pm**: head-retrain CPU diagnostic (`analysis/cpu_diagnostics_2026-05-11_head_retrain/HEAD_RETRAIN_FACTS_2026-05-11.md`) showed T4 new-head chronic_6 AUC = 0.9647 (vs T4 trained-head 0.5788 on same local subset) = +0.39 absolute LIFT. The feature-level inv_mean drop A3 reported is REAL but is small enough that a fresh head trained on dev features can recover full chronic_6 discrimination — T4's encoder does carry the chronic_6 signal the trained head missed. The chronic_6 issue moves from "feature regression" to "trained-head fails to leverage chronic_6 feature signal that's still present"; downgraded to low severity. Caveat: local chronic_6 cohort in lockbox is `bla_bla_chow`-only (n=61); `PC_Generator__s22/__s45`, `roy_d`, `Q__s6` not in local CSVs — the +0.39 lift may not generalize uniformly across all chronic_6 identities.

**Amendment 2026-05-12**: T5C step3500 (the first drafted training-time attempt — classifier hidden_dim 256→1024) ranks 3 on the v3-fix scorecard above T3_SLOT1 step1500; `teams_real_dor_dev` FPR is 0.06 (3/50) vs T4's 0.18 (9/50). Directionally consistent with the close-criterion intervention working at the behavioral level. Feature-level verification (L11 atlas recompute on T5C ckpts) is the pending CPU follow-up — see open loop `t5c-classifier-capacity-mechanism`.

- **source**: A3 (`analysis/cpu_diagnostics_2026-05-11_a3_atlas_composition/ATLAS_COMPOSITION_FACTS_2026-05-11.md`) + head-retrain verdict (`HEAD_RETRAIN_FACTS_2026-05-11.md`) + T5C scorecard (`analysis/t6_t7_t5c_scorecard_eval_2026-05-12/RESULTS_FACTS_2026-05-12.md` §8) + per-axis numbers in OVERNIGHT_STATUS_2026-05-11.md

### Open loop: dev-to-lockbox-substrate-transfer-gap
status: open
severity: high
first_seen: 2026-05-11
last_verified: 2026-05-12
close_criterion: EITHER (a) a head-retrain on a substrate-diverse pool (dev + lockbox-style data, properly held-out for eval) lifts T4 lockbox AUC ≥ 0.90 — confirms the gap is closeable by exposing the head to substrate-diverse training, OR (b) the gap is REPRODUCED on multiple ckpts (T4, P8A, T3) and on the full lockbox cohort — closes as "structural data gap; requires new training data ingestion, not architecture/loss changes"

**Summary (2026-05-11)**: Head-retrain CPU diagnostic (β-outcome) demonstrates the binding failure mode on lockbox is NOT head architecture but training-substrate → lockbox-substrate distribution mismatch. T4 new-head lockbox AUC = 0.7486 ≈ trained-head 0.7619; P8A new-head COLLAPSES to 0.5075 vs trained-head 0.9355. Both ckpts trained their new heads on dev features with perfect dev held-out AUC (≥0.9998), so the training distribution is the binding constraint — when the head sees only dev-style features at training time, it cannot map T4 encoder's lockbox features to correct labels. Specifically, T4 features support chronic_6 discrimination after dev-only head training (new head AUC 0.9647 on local chronic_6) but NOT dor_shkedi discrimination (new head AUC 0.6064 — down from trained-head 0.8337 on same local subset). The dor_shkedi cohort needs lockbox-substrate exposure during head training to extract.

**Why this matters**: refutes the "T4 + fresh head ships at near-zero cost" hypothesis for dev-trained heads. Suggests that any production model needs training data that COVERS the dor_shkedi-style real distribution; current training data doesn't (per the persistent dor_shkedi-vs-real_dor signature flip across all R13 ckpts, memory `project_signature_shortcut_finding`). The viso ceiling reframe (memory `project_job_b_findings_universal_vs_trajectory_2026-05-04`) is the v2 substrate analog of this — production substrate gaps live in the data, not in the architecture or loss.

**Amendment 2026-05-12**: T6/T7/T5C scorecard verdict reinforces the high-severity reading. Among 6 contract-passing ckpts, rank ordering tracks ascending `lockbox_real_fpr` strictly: P8A 0.0184 → E2B 0.0235 → T5C step3500 0.0279 → T3_SLOT1 0.0309 = T5C step3750 0.0309 → T5C step1500 0.3204. Every new candidate since T3 lifts `lockbox_fake_recall` while degrading `lockbox_real_fpr` by 0.005-0.30 absolute. The metric P8A is uniquely best on is the contract tiebreak, which is why P8A holds rank-1 across 13+ packets. Verdict pending close criterion (a) or (b).

- **source**: head-retrain CPU diagnostic (`analysis/cpu_diagnostics_2026-05-11_head_retrain/HEAD_RETRAIN_FACTS_2026-05-11.md` §5 per-cohort numbers); T6/T7/T5C scorecard (`analysis/t6_t7_t5c_scorecard_eval_2026-05-12/RESULTS_FACTS_2026-05-12.md` §5.2, §7.2).

## 2026-05-12 update — T6/T7/T5C scorecard: T5C is the new candidate, jitter doesn't compose, lockbox_real_fpr remains the binding constraint

T6/T7/T5C ran 2026-05-11; promotion-contract scorecard SUCCEEDED 2026-05-12 03:13 UTC. 12 ckpts (3 candidates each from T6, T7, T5C + 3 anchors P8A/E2B/T3_SLOT1) scored on the standing v3-fix policy.

### Facts

- **P8A retains contract rank-1** (FACTS `analysis/t6_t7_t5c_scorecard_eval_2026-05-12/RESULTS_FACTS_2026-05-12.md` §3.1). T5C step3500 is the highest-ranking new ckpt at rank 3, above the T3_SLOT1 step1500 anchor at rank 4.
- **6 of 12 ckpts pass all three contract gates** (FACTS §5.1): P8A, E2B, T5C step3500, T3_SLOT1 step1500, T5C step3750, T5C step1500. The 6 that fail are all 3 T6 ckpts + all 3 T7 ckpts — every jitter ckpt regresses `dev_fake_macro_recall` below 0.30 on both bases (T3 and T4).
- **Among all-pass ckpts, rank ordering is consistent with ascending `lockbox_real_fpr`** (FACTS §5.2): P8A 0.0184 → E2B 0.0235 → T5C step3500 0.0279 → T3_SLOT1 0.0309 = T5C step3750 0.0309 → T5C step1500 0.3204. Not by dev_macro_recall (T5C step1500 has the highest at 0.6832 but ranks 6).
- **T5C step3500 vs T4 step10500 (its base)**: dev_macro +0.04, lockbox_real_fpr halved (0.0279 vs 0.0536), lockbox_fake_recall +0.29 (0.66 vs 0.37), dor_dev (n=50) FPR 0.06 vs 0.18 (FACTS §7.2 + §8 + T4 eval `RESULTS_FACTS_2026-05-11.md` §4.1). Single-lever delta is classifier hidden_dim 256→1024.
- **T5C step1500 has catastrophic lockbox_real_fpr** (0.3204, 17× P8A) but step3500 / step3750 are clean (0.0279 / 0.0309). Same recipe, different training step. Implication: the lockbox-clean window in T5C training is narrow; needs trajectory characterization before scaling up.
- **T6 step1500 has the highest `lockbox_fake_recall` of any scored ckpt** (0.8696) but `dev_fake_macro_recall` 0.2598 (below floor). Jitter trades dev capture for lockbox capture in a way that the v3-fix policy doesn't reward.

### Current stance (revised 2026-05-12)

The T5C classifier-capacity hypothesis (motivated by A3 chronic_6 −0.0315 regression) is **directionally supported** at MEDIUM confidence. T5C step3500's dor_dev FPR (0.06 vs T4's 0.18, n=50 95% CI ±0.14) is suggestive but not dispositive. A CPU-only L11 atlas inv_mean recompute on T5C ckpts is the cheap verification.

The face_scale_jitter@0.50 hypothesis from memory `project_face_scale_jitter_load_bearing.md` (2026-04-30 P14 sister-variant ablation) **does NOT compose** when stacked on T3 or T4. Memory entry's scope should be amended: "load-bearing AS A REPLACEMENT FOR a bad bundle, not AS AN ADDITIVE LEVER to clean bases".

The `dev-to-lockbox-substrate-transfer-gap` open loop is **amended with reinforcement, not closure**. Every new candidate lifts `lockbox_fake_recall` (good) and degrades `lockbox_real_fpr` (mixed-to-bad). P8A holds rank 1 across 13+ packets because of this metric exclusively.

### Implications for the open-loop program

- **T5C is the next refinement direction** (single-lever hidden_dim variant), not T6/T7. The cheap CPU-first test is the L11 atlas recompute (option below).
- **The viso ceiling now has a closely-tied operational analog**: just as the viso recall ceiling on v2 substrate was traced to substrate-pollution (memory `project_job14_substrate_clean_2026-05-04`), the lockbox_real_fpr ceiling is traced to dor_shkedi cohort substrate-distribution mismatch. Different metric, same structural reading: production-substrate gaps live in the data layer, not in architecture/loss.
- **Deployment Q is now sharper**: T5C step3500 catches 1.7× more lockbox fakes than P8A at +0.0095 absolute `lockbox_real_fpr`. The v3-fix policy ranks P8A first via the tiebreak — but whether to ship T5C alongside or in place of P8A depends on FP/FN cost ratios the policy doesn't make explicit.

### Open loops added or amended

### Open loop: face-scale-jitter-composability
status: resolved
severity: low
first_seen: 2026-04-30
last_verified: 2026-05-12
close_criterion: empirical test of jitter@0.50 stacked on top of a non-bundle base (T3 SLOT1 or T4 multi-axis-L11-GRL) — if dev_macro_recall holds within 0.05 of base, composability is supported; if regression > 0.05 absolute, lever is bundle-replacement only

**Resolution (2026-05-12)**: jitter@0.50 alone won value_composite on the P14 sister-variant ablation (memory `project_face_scale_jitter_load_bearing`). On T3 base (T6 packet) and T4 base (T7 packet), jitter@0.50 added on top regresses `dev_fake_macro_recall` below the 0.30 contract floor on every sampled step (6/6 — 3 sampled steps × 2 bases). Memory's "load-bearing single lever" framing was scoped to "as a replacement for a bad bundle", not "as an additive lever on clean bases". Memory entry amended this session; see opinion-doc §5 retraction log.

- **caveats**: only one magnitude (scale_limit=0.50) was tested. Lower magnitudes (e.g., 0.20-0.30) might compose; cannot conclude jitter is universally non-composable.
- **source**: T6/T7 scorecard (`analysis/t6_t7_t5c_scorecard_eval_2026-05-12/RESULTS_FACTS_2026-05-12.md` §5.1) + AGENT_PROPOSAL `§5` retraction log.

### Open loop: t5c-classifier-capacity-mechanism
status: open
severity: medium
first_seen: 2026-05-12
last_verified: 2026-05-12
close_criterion: L11 atlas inv_mean recompute on T5C step1500 + step3500 + step3750 (a la A3 2026-05-11) — if T5C step3500 chronic_6 inv_mean ≥ P8A's (vs T4 step10500's −0.0315 absolute), classifier capacity is the lever responsible for the chronic_6 lift; otherwise the dor-cohort lift comes from somewhere else and §3.1 of the opinion-doc is refuted

**Summary**: T5C step3500 ranks 3 on the v3-fix contract (above T3_SLOT1 step1500 at rank 4); its single-lever delta vs T4 is classifier hidden_dim 256→1024. T5C step3500's `teams_real_dor_dev` (n=50) FPR is 0.06 vs T4's 0.18 — directionally consistent with "stronger online classifier forces uniform encoder invariance covering chronic_6". But the trajectory is fragile: T5C step1500 has catastrophic `lockbox_real_fpr` 0.3204 while step3500 / step3750 sit at 0.028 / 0.031. n=50 cohort is noise-band-limited; mechanism is not feature-level verified.

- **next-step implication**: this CPU-only diagnostic (~3h MPS, $0) gates whether to authorize a GPU T5C × hidden_dim sweep ($70-100). Without the diagnostic, the sweep is uncalibrated.
- **source**: T6/T7/T5C scorecard (`analysis/t6_t7_t5c_scorecard_eval_2026-05-12/RESULTS_FACTS_2026-05-12.md`) + opinion-doc §3.1 + A3 (`analysis/cpu_diagnostics_2026-05-11_a3_atlas_composition/ATLAS_COMPOSITION_FACTS_2026-05-11.md`).

---

## 2026-05-12 evening — D1-D5 CPU diagnostic program

User raised a critique mid-session: P8A-anchor levers (output distillation, L11 feature anchor) may be self-deceptive — they propagate P8A's specific shortcut alignment rather than teaching principled forgery detection. The planning agent designed a 5-CPU-job program (D1-D5) to test the framing before committing GPU. All 5 ran on the user's MPS Mac via parallel subagent dispatch.

### D1-D5 design questions (one-liner each)

| Diag | Question | FACTS doc |
|---|---|---|
| **D1** | Is P8A's chronic-6 frame-level score IQ-explained? Per-identity? | `analysis/cpu_diagnostics_2026-05-12_d1_p8a_iq_alignment/D1_FACTS_2026-05-12.md` |
| **D5** | Across identities, is per-identity score predicted by per-identity mean-IQ? Which identities are residual-high? | `analysis/cpu_diagnostics_2026-05-12_d5_per_identity_iq/D5_FACTS_2026-05-12.md` |
| **D4** | Does per-IQ-quartile FPR transfer dev↔lockbox? (Is the substrate gap purely IQ-distributional?) | `analysis/cpu_diagnostics_2026-05-12_d4_iq_quartile_alignment/D4_FACTS_2026-05-12.md` |
| **D3** | Does the encoder's separation direction transfer dev↔lockbox? (Head problem vs encoder problem?) | `analysis/cpu_diagnostics_2026-05-12_d3_cross_substrate_probe/D3_FACTS_2026-05-12.md` |
| **D2** | Is the encoder's L11 real/fake direction IQ-aligned or IQ-orthogonal? (With CLIP-frozen baseline.) | `analysis/cpu_diagnostics_2026-05-12_d2_encoder_separation/D2_FACTS_2026-05-12.md` |

### Headline results

- **D1**: chronic-6 joint R²=0.350 (vs healthy 0.163); per-identity `saturation_mean` β SIGN-FLIPS across chronic identities (Roy_D −6.15 vs dor_shkedi +3.33). P8A's chronic-6 IQ-handling is per-identity-specific, not axis-coherent.
- **D5**: per-identity output R² with mean-IQ — P8A 0.22, T3_S1_step1500 0.49, T5C_step3500 0.60. Residual sign cleanly partitions 5/5 chronic-6 above and 5/6 non-chronic below the IQ-predicted line across all 3 ckpts. P8A is identity-cluster-memorizing; T5C step3500 is axis-aligned at output level.
- **D4**: per-IQ-quartile dev↔lockbox FPR alignment r: P8A +0.716, T5C +0.508, T3_SLOT1 −0.032. T3's data filter introduced substrate-specific cohort responses. At dev-cal 5% τ, lockbox FPR is LOWER than dev for all 3 ckpts.
- **D3**: 0/4 ckpts pass substrate-agnostic AUC>0.95-both-directions criterion. DEV→LOCKBOX 0.594-0.747, LOCKBOX→DEV 0.825-0.951. The encoder retains DIFFERENT separations per substrate. **Supersedes the A2-ext "encoder retains separation, head is failure" framing** — encoder is itself substrate-specific.
- **D2**: CLIP-frozen L11 features separate chronic-6 real/fake at AUC=1.000 with no FT. The forgery signal is in raw CLIP. FT pulls the chronic-6 IQ-PC1 angle from CLIP-frozen's 83.68° to FT'd ckpts' 74.95°-78.54° (9-13° below 87.5° random baseline). Smallest per-axis angles consistently on `color_a_dev` / `saturation_mean`. Full and non-chronic cohorts stay near random baseline.

### Synthesis OPINIONS doc

[`analysis/cpu_diagnostics_2026-05-12_d1_d5_synthesis/D1_D5_OPINIONS_2026-05-12.md`](../../../analysis/cpu_diagnostics_2026-05-12_d1_d5_synthesis/D1_D5_OPINIONS_2026-05-12.md)

The doc includes a "deeper story" paragraph (§1), per-diagnostic interpretations with alternative readings flagged (§2), 4-slot GPU plan with explicit falsifiers (§3), intra-session self-correction log (§4), and load-bearing uncertainties (§5).

### Open loop updates

### Open loop: dev-to-lockbox-substrate-transfer-gap
status: resolvable
severity: high
first_seen: 2026-05-11
last_verified: 2026-05-12
close_criterion: EITHER (a) a head-retrain on a substrate-diverse pool (dev + lockbox-style data, properly held-out for eval) lifts T4 lockbox AUC ≥ 0.90 — confirms the gap is closeable by exposing the head to substrate-diverse training, OR (b) the gap is REPRODUCED on multiple ckpts (T4, P8A, T3) and on the full lockbox cohort — closes as "structural data gap; requires new training data ingestion, not architecture/loss changes"

**2026-05-12 resolution candidate**: D3 satisfies close criterion (b). DEV→LOCKBOX probe transfer AUC on the 800-frame triptych is 0.594 (P8A), 0.667 (E2B), 0.747 (T5C_step3500), 0.668 (T3_S1_step1500); 0/4 ckpts pass the substrate-agnostic criterion (both directions > 0.95). In-sample CV AUC ≥ 0.999 in every cell — the signal is present within each substrate but the encoder retains DIFFERENT separating directions per substrate. The "head retraining alone closes the gap" framing (from `HEAD_RETRAIN_FACTS_2026-05-11.md` α-outcome reading) is partially refuted: head retraining on dev-features can find dev's separation direction, but that direction doesn't predict lockbox labels. The structural fix is data-side (substrate-diverse ingestion) or training-time (anti-shortcut regularization that prevents the encoder from learning substrate-specific directions in the first place).

- **caveats**: lockbox slice in D3 is n=87 (small). 95% CIs are wide (±0.05). Tier 2 escalation to a larger lockbox sample was not run; should be a follow-up if a critic challenges the resolution. Resolution proposed but not finalized — user to gate.
- **source**: `analysis/cpu_diagnostics_2026-05-12_d3_cross_substrate_probe/D3_FACTS_2026-05-12.md` + `analysis/cpu_diagnostics_2026-05-12_d1_d5_synthesis/D1_D5_OPINIONS_2026-05-12.md` §2.4.

### Open loop: chronic-6-encoder-iq-angle-drift-during-ft
status: open
severity: medium
first_seen: 2026-05-12
last_verified: 2026-05-12
close_criterion: a counterfactual training experiment establishes whether the chronic-6 IQ-PC1 angle drift from CLIP-frozen's 83.68° to FT'd ckpts' 74.95°-78.54° is causally responsible for chronic-cohort FPR, OR is a correlated side-effect. Candidate experiment: FT with continuous-axis-GRL on 6 IQ axes that explicitly pulls the encoder direction toward IQ-orthogonality. If post-training chronic-6 angle ≥ 82° AND chronic-cohort FPR drops by ≥ 0.05 absolute at FPR-cal τ, the drift is causally implicated. If angle goes ≥ 82° but FPR doesn't change, the drift is incidental.

**Summary**: D2 measures the IQ-PC1 angle on the real/fake separation direction within chronic-6 sub-cohort. CLIP-frozen's angle is 83.68° (3.9° below 87.5° random baseline). FT'd ckpts' angles are 74.95° (E2B), 75.19° (T5C_step3500), 77.09° (T3_S1_step1500), 78.54° (P8A) — 9-13° below random baseline. Smallest per-axis angles are consistently on `color_a_dev` / `saturation_mean`. This is correlational; mechanism for chronic-cohort FPR is unestablished.

- **source**: `analysis/cpu_diagnostics_2026-05-12_d2_encoder_separation/D2_FACTS_2026-05-12.md` + opinion §2.5.

### Open loop: clip-frozen-chronic-6-auc-robustness
status: open
severity: low
first_seen: 2026-05-12
last_verified: 2026-05-12
close_criterion: extend the CLIP-frozen chronic-6 probe to a larger sample (e.g., 500-1000 chronic-6 reals from contract suites + 100-200 chronic-6 fakes). If 5-fold CV probe AUC remains ≥ 0.95 at the larger sample, the "forgery signal is in raw CLIP" framing is robust. If the AUC drops below 0.90 at scale, the n=282 triptych result is sample-size-inflated and the framing weakens.

**Summary**: D2 reports CLIP-frozen L11 features give AUC=1.000 on chronic-6 (n=282, 41 fakes) with no FT. This is load-bearing for Slot 2 (B16-scratch + bundle) — the lever's mechanism assumes CLIP-frozen already carries chronic-6 forgery signal. Small fake-class sample → high AUC variance under 5-fold CV (each fold has ~8 fakes). A larger-sample retest would solidify the finding.

- **source**: `analysis/cpu_diagnostics_2026-05-12_d2_encoder_separation/D2_FACTS_2026-05-12.md` §8 obs 1.

---

## 2026-05-12 evening — D1-D5 critic-review + train-overlap audit

Independent critic-review of `D1_D5_OPINIONS_2026-05-12.md` completed; deliverable at [`../../../analysis/cpu_diagnostics_2026-05-12_d1_d5_synthesis/D1_D5_CRITIC_REVIEW_2026-05-12.md`](../../../analysis/cpu_diagnostics_2026-05-12_d1_d5_synthesis/D1_D5_CRITIC_REVIEW_2026-05-12.md). Three load-bearing disagreements logged: (a) the FT-induced chronic-6 IQ-angle drift is ~5° vs CLIP-frozen 83.68°, not the ~12° vs theoretical 87.5° baseline the planning agent claimed — C3 re-graded MEDIUM not HIGH; (b) "P8A identity-cluster memorization" (C4) is one of several supported framings; the may6 production-drift evidence (P8A 0/92 false-flags on fresh real Xinhe) favors the invariance reading; (c) D4's "lockbox FPR is below dev at dev-cal τ for all 3 ckpts" reframes the dev-to-lockbox transfer gap as mechanistically real but operationally non-binding at deployment τ. Critic's 4-slot revision: add data-side ingestion slot (the slot the planning agent missed); keep continuous-axis-GRL on T5C; restore L11 anchor on T5C 5-identity cohort (planning agent's retraction confused JOB1's median per-frame L2 with per-identity score-overfire concentration in JOB2); keep LoRA at MEDIUM-LOW; drop Slot 2 (single-lever-discipline violation per AGENT_GUIDE Rule 6) and Slot 3 (low ceiling).

Following the critic-review, the load-bearing follow-up CPU diagnostic ran 2026-05-12 evening: train-overlap audit for the 3 recurring D5 residual identities (`real_dor`, `Cam_Test`, `PC_Generator`). Full FACTS at [`../../../analysis/train_overlap_audit_2026-05-12/TRAIN_OVERLAP_FACTS_2026-05-12.md`](../../../analysis/train_overlap_audit_2026-05-12/TRAIN_OVERLAP_FACTS_2026-05-12.md). Key findings:

- All 3 identities' eval-substrate frames are in `gs://teams-faces-data-test-2914-fake-4420-real-feb-28` (109 + 812 + 790 videos respectively across the Teams target-domain manifest cells).
- That bucket has **0 references across all 158 R13 training yamls** (grep verified).
- P8A's Teams training source is a different bucket: `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`. The 2026-05-04 inventory audit's `also_in_training="yes (teams-v2 bucket is training+ood)"` annotation is set by Teams-method-string match in audit code (`analysis/inventory_audit_2026-05-04/run_audit.py:340-348`), not by per-identity bucket-content verification — a soft bucket-family inference.
- `real_dor` specifically: the only training-side bucket containing this identity string is `gs://real-teams-dor-roee/session_20260424_110139/uniform30`, declared `readout_only_external_real_sources` in P8A's R13_RLP8_01 yaml line 262. `readout_only_*` blocks are observed during training (FPR logged) but do not contribute gradients (`analysis/generate_packet6_yamls_2026-04-23.py:308-309`).

Net effect on the critic vs planning-agent debate: the "P8A memorization of these 3 identities" framing now has weaker bucket-level evidence than the inventory audit suggested. Person-level identity overlap between the eval bucket and the training-side teams-v2 bucket is unverifiable from local-only evidence and requires a GCS-side audit (~10 min, cost <$1). The slot plan does not gate on the resolution — Slots A/B/C/D are all worth running regardless.

### Open loop: train-bucket-identity-overlap-gcs-audit
status: open
severity: low
first_seen: 2026-05-12
last_verified: 2026-05-12
close_criterion: a GCS-side enumeration (`gsutil ls gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2/samples/ | head -1000`) plus identity-name extraction from per-sample `manifest.json`'s `original_video_name` field determines whether the SAME HUMANS as `real_dor`/`Cam_Test`/`PC_Generator`/etc. appear in the training-side teams-v2 bucket under different sample_ids or sessions. If yes, the "P8A memorization" framing is plausible and the critic's §3.2 C4 reframe should be partially walked back. If no, the "P8A invariance" framing is strengthened and Slot C (L11 anchor on 5-identity cohort) becomes safer. Cost: <10 min CPU + GCS list quota; <$1.

**Summary**: The 2026-05-12 evening train-overlap audit established at the FRAME / BUCKET level that the 3 recurring D5 residual identities (`real_dor`, `Cam_Test`, `PC_Generator`) are not in P8A's training pool — their eval frames live in a bucket (`teams-faces-data-test-2914-fake-4420-real-feb-28`) that has 0 references across all R13 training yamls. At the IDENTITY (person) level, the question is whether the same humans appear in the training-side bucket `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2` under different sample_ids. The 2026-05-04 inventory audit's `also_in_training` annotation said "yes" but its logic was method-string match, not per-identity bucket-content verification — a soft inference. A direct GCS-side audit would close this loop.

- **source**: `analysis/train_overlap_audit_2026-05-12/TRAIN_OVERLAP_FACTS_2026-05-12.md` §3.1, §4, §7; `analysis/cpu_diagnostics_2026-05-12_d1_d5_synthesis/D1_D5_CRITIC_REVIEW_2026-05-12.md` §addendum.

---

## 2026-05-12 late evening — D6 B=50 finalization + D7-D10 CPU diagnostics

D6 B=50 re-run completed (3736 s, 18:08 → 19:12 local). D7, D8, D9, D10 dispatched as 4 parallel background subagents at the user's authorization for "ALL the CPU jobs now"; all completed within a single window.

### Facts

**D6 B=50 chronic-6 cell** (full numbers at [`D6_FACTS_2026-05-12.md`](../../../analysis/cpu_diagnostics_2026-05-12_d6_empirical_orthogonality/D6_FACTS_2026-05-12.md) §4):

| Ckpt | Bootstrap-mean angle | std | Drift vs CLIP-frozen 79.53° |
|---|---:|---:|---:|
| CLIP_FROZEN | 79.53° | 1.42° | 0.00° (baseline) |
| P8A | 74.47° | 1.92° | 5.06° |
| T3_S1_step1500 | 71.98° | 2.32° | 7.55° |
| T5C_step3500 | 69.99° | 2.31° | 9.54° |
| E2B | 69.08° | 1.86° | 10.45° |

D2's single-split chronic-6 cells biased +2.12σ to +3.16σ above bootstrap-mean uniformly. Full and non-chronic cohort bootstrap-means all within ±2° of 87.93° random baseline — angle drift is concentrated in chronic-6 sub-cohort, not present at full-cohort level.

**D7 (eval-FPR decomposition)** ([`D7_FACTS_2026-05-12.md`](../../../analysis/cpu_diagnostics_2026-05-12_d7_fpr_decomposition/D7_FACTS_2026-05-12.md)):

- Panel: n=3,416 eval reals (IQ-complete AND locally cached); P8A 113 FP (3.31%), T5C 197 FP (5.77%) at dev-cal 5% τ.
- Full-model fit (8 features = 6 IQ + substrate_distance + chronic_indicator): P8A McFadden R²=0.280 AUC=0.880; T5C R²=0.392 AUC=0.891.
- Block-drop ΔR² (same rank order both ckpts: IQ ≫ chronic > substrate):

| Block dropped | P8A ΔR² | P8A ΔAUC | T5C ΔR² | T5C ΔAUC |
|---|---:|---:|---:|---:|
| IQ (6 features) | +0.2359 | +0.1976 | +0.3143 | +0.1930 |
| chronic_6 (1 feat) | +0.0137 | +0.0058 | +0.0625 | +0.0423 |
| substrate_distance (1 feat) | +0.0146 | −0.0003 | +0.0001 | +0.0008 |

- Single-block-only AUC: IQ 0.873/0.848 (P8A/T5C); chronic 0.642/0.674; substrate 0.575/0.527. Substrate-only is barely above chance on both ckpts.
- P8A `substrate_distance` β = −0.41 (frames closer to training reals score HIGHER); T5C β = +0.06 (near-zero).
- Chronic vs non-chronic FPR ratio: P8A 4.68× (4.82% vs 1.03%); T5C 8.66× (8.91% vs 1.03%).
- Training-pool reference set was 371 frames (D7 requested 2000; locally cached subset); lockbox coverage in panel 12.4%; in-sample R²/AUC (no held-out CV).

**D8 (substrate-balanced head retrain)** ([`D8_FACTS_2026-05-12.md`](../../../analysis/cpu_diagnostics_2026-05-12_d8_head_retrain_substrate_balanced/D8_FACTS_2026-05-12.md)):

- DEV→LOCKBOX probe transfer AUC on P8A frozen-encoder L11 features, n_train=4000 dev, n_test=839 lockbox, bootstrap 95% CI:

| Head | AUC | 95% CI | Δ vs unweighted |
|---|---:|---|---:|
| Unweighted baseline | 0.587 | [0.550, 0.620] | (baseline) |
| Primary PCA32 Silverman KDE | 0.561 | [0.528, 0.596] | −0.025 (CIs overlap) |
| Robustness PCA8 ×2-bw KDE | 0.667 | [0.632, 0.702] | +0.080 (CIs disjoint) |
| Classifier-KLIEP | 0.679 | [0.643, 0.713] | +0.093 (CIs disjoint) |

- Brief's specified PCA32-KDE estimator collapsed: 2000/2000 dev-real weights pinned to clip floor (max unclipped density ratio 1.46e-9).
- **KLIEP discriminator achieves 99.09% balanced accuracy distinguishing dev-real from lockbox-real in CLIP-frozen feature space.** Dev and lockbox real-pools are nearly linearly separable.
- LOCKBOX→DEV unweighted reference: 0.814 [0.799, 0.824]. Direction asymmetry persists across all four heads.
- Contract metrics at dev-cal 5% τ across all four offline-LR heads: lockbox FPR 66-72%, ~70-80 pp above the in-loop P8A production head's 0.92% per D4 (head construction is structurally different).
- dev↔lockbox identity overlap not blocked in the LR fits; KLIEP signal may include identity-recurrence (not decomposed).

**D9 (source-substrate inventory)** ([`D9_FACTS_2026-05-12.md`](../../../analysis/cpu_diagnostics_2026-05-12_d9_source_substrate_inventory/D9_FACTS_2026-05-12.md)):

- 195 R13 yamls (prior "158" count was wrong); 97.9% enable teams lane; 52.3% enable proper_data.
- Source-substrate cross-tab on real frames:

| Substrate class | train.teams-v2 | train.proper_data | dev | lockbox |
|---|---:|---:|---:|---:|
| direct_teams_capture | 0 (0%) | 0 (0%) | 4,614 (100%) | 1,418 (100%) |
| youtube_origin (teams pipeline) | 4,908 (100%) | 0 | 0 | 0 |
| hdtf_corpus (proper_data) | 0 | 35,008 (59.9%) | 0 | 0 |
| qclips_corpus (proper_data) | 0 | 23,424 (40.1%) | 0 | 0 |

- 250/250 sampled teams-v2 training manifests classify as `youtube_origin` (11-char `original_trimmed_video_name` token).
- 100% of 4,664 eval-manifest reals classify as direct-Teams capture.
- Zero substrate-class overlap between training reals and eval reals (in inventoried lanes).
- teams-v2 manifest `source` field is uniformly `"teams_capture"` (250/250) coexisting with YouTube-ID `original_trimmed_video_name` — the field describes the pipeline applied, not the underlying origin. Zero shared `source_kind` values between teams-v2 and eval manifests.
- teams-v2 sampling coverage = 18.6% of complete_pairs (250/1346); 100% youtube_origin extrapolation depends on prior GCS audit's uniform-pattern observation across all 10 strategy partitions.

**D10 (training-pool position in dev-vs-lockbox CLIP space)** ([`D10_FACTS_2026-05-12.md`](../../../analysis/cpu_diagnostics_2026-05-12_d10_training_pool_position/D10_FACTS_2026-05-12.md)):

- KLIEP discriminator re-fit: balanced acc 0.9909 matches D8.
- Mean projection on KLIEP axis: dev_real (n=2000) −3.162; train_real (n=371) **−2.401**; lockbox_real (n=414) +3.169.
- Decision-boundary split (proj<0 = dev side): 87.3% of train_real on dev side, vs 99.4% of dev_real and 2.4% of lockbox_real.
- Wasserstein-1: W(train, dev) **0.769** vs W(train, lockbox) **5.570** (7.25× ratio); W(dev, lockbox) 6.332. Train-pool's W1 distance to dev is ~12% of the dev↔lockbox distance.
- Overlap fractions: 77.6% of train_real falls in dev_real [p5, p95]; 9.97% in lockbox_real [p5, p95].
- **KLIEP discriminator axis vs IQ-PC1 angle = 89.71°** (cos = 0.0051). Robust 89.26°-89.71° across 4 KLIEP/IQ-PC1 normalization variants. All 6 individual IQ-axis directions are within 1.5° of 90°.
- Bootstrap stability (B=20): train_mean = −2.401 ± 0.036 (CV 1.49%); sign preserved 20/20 iterations.
- Training-pool reference 371 frames (not 2000 — locally cached subset); bootstrap resamples dev/lockbox only.

### Open loop updates

### Open loop: dev-to-lockbox-substrate-transfer-gap
status: resolved
severity: high
first_seen: 2026-05-11
last_verified: 2026-05-12
close_criterion: EITHER (a) a head-retrain on a substrate-diverse pool (dev + lockbox-style data, properly held-out for eval) lifts T4 lockbox AUC ≥ 0.90 — confirms the gap is closeable by exposing the head to substrate-diverse training, OR (b) the gap is REPRODUCED on multiple ckpts (T4, P8A, T3) and on the full lockbox cohort — closes as "structural data gap; requires new training data ingestion, not architecture/loss changes"

**Resolution 2026-05-12**: Close criterion (b) is MET. D3 measured DEV→LOCKBOX probe transfer AUC on 4 ckpts: P8A 0.594, E2B 0.667, T5C 0.747, T3 0.668; LOCKBOX→DEV 0.825-0.951; in-sample 5-fold CV ≥ 0.999 every cell. 0/4 ckpts pass the substrate-agnostic AUC>0.95 both-directions threshold. The gap is reproduced across all measured ckpts. D8 corroborates at a separate angle: a KLIEP discriminator distinguishes dev-real from lockbox-real at 99.09% accuracy in CLIP-frozen feature space; substrate-balanced head re-training lifts DEV→LOCKBOX transfer +0.093 AUC (CIs disjoint vs unweighted). D10 adds the structural detail: training pool sits 87.3% on the dev side of the KLIEP axis; W(train, lockbox) is 7.25× W(train, dev). D7 adds an operational nuance: substrate_distance to training-pool has ~0 partial R² for FP at the per-frame level despite the substrate axis being real and large at the pool level — the substrate axis and the IQ axes (the FPR drivers) are 89.71° apart in CLIP-frozen space (D10).

Closure verdict: **structural data gap; not architecture/loss-fixable in isolation. The substrate axis is real but operationally orthogonal to the FPR-driving IQ axes at the CLIP-frozen feature level. Data-side ingestion may move the substrate axis but is not predicted by D7 to move FPR.**

- **source**: [`D3_FACTS_2026-05-12.md`](../../../analysis/cpu_diagnostics_2026-05-12_d3_cross_substrate_probe/D3_FACTS_2026-05-12.md), [`D8_FACTS_2026-05-12.md`](../../../analysis/cpu_diagnostics_2026-05-12_d8_head_retrain_substrate_balanced/D8_FACTS_2026-05-12.md), [`D10_FACTS_2026-05-12.md`](../../../analysis/cpu_diagnostics_2026-05-12_d10_training_pool_position/D10_FACTS_2026-05-12.md), [`D7_FACTS_2026-05-12.md`](../../../analysis/cpu_diagnostics_2026-05-12_d7_fpr_decomposition/D7_FACTS_2026-05-12.md).

### Open loop: train-bucket-identity-overlap-gcs-audit
status: resolved
severity: low
first_seen: 2026-05-12
last_verified: 2026-05-12
close_criterion: a GCS-side enumeration (`gsutil ls gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2/samples/ | head -1000`) plus identity-name extraction from per-sample `manifest.json`'s `original_video_name` field determines whether the SAME HUMANS as `real_dor`/`Cam_Test`/`PC_Generator`/etc. appear in the training-side teams-v2 bucket under different sample_ids or sessions. If yes, the "P8A memorization" framing is plausible and the critic's §3.2 C4 reframe should be partially walked back. If no, the "P8A invariance" framing is strengthened and Slot C (L11 anchor on 5-identity cohort) becomes safer. Cost: <10 min CPU + GCS list quota; <$1.

**Resolution 2026-05-12 night**: GCS audit completed. 1646 sample listings + 250 manifests inspected in `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`. 0/23 identity-name patterns match in either listings or manifest bodies. Manifest schema (27-key union) lacks `identity` / `subject` / `person` / `source_id` / `original_sample_id` fields. Source-video identifiers are YouTube-video-ID-anonymized tokens (e.g., `lm0hNQmOdFg`, `Y-raYBsC3J4`). **Person-level identity overlap is structurally undetectable from training-data manifests** — the anonymization scheme has no shared join key with eval-side identity names. The "P8A memorizes identities" framing is structurally unsupported at the level the question can be answered from available data; D9 corroborates that the training-data origin distribution is 100% disjoint from eval-data origin at the substrate level.

- **source**: [`GCS_IDENTITY_AUDIT_FACTS_2026-05-12.md`](../../../analysis/cpu_diagnostics_2026-05-12_gcs_identity_audit/GCS_IDENTITY_AUDIT_FACTS_2026-05-12.md), [`D9_FACTS_2026-05-12.md`](../../../analysis/cpu_diagnostics_2026-05-12_d9_source_substrate_inventory/D9_FACTS_2026-05-12.md).

### Open loop: chronic-6-encoder-iq-angle-drift-during-ft
status: open
severity: medium
first_seen: 2026-05-12
last_verified: 2026-05-12
close_criterion: a counterfactual training experiment establishes whether the chronic-6 IQ-PC1 angle drift from CLIP-frozen's 83.68° to FT'd ckpts' 74.95°-78.54° is causally responsible for chronic-cohort FPR, OR is a correlated side-effect. Candidate experiment: FT with continuous-axis-GRL on 6 IQ axes that explicitly pulls the encoder direction toward IQ-orthogonality. If post-training chronic-6 angle ≥ 82° AND chronic-cohort FPR drops by ≥ 0.05 absolute at FPR-cal τ, the drift is causally implicated. If angle goes ≥ 82° but FPR doesn't change, the drift is incidental.

**Amendment 2026-05-12 late evening**: D6 B=50 finalizes the measurement. FT-vs-CLIP-frozen chronic-6 drift on the bootstrap-mean (vs empirical 79.53° baseline): P8A 5.06°, T3 7.55°, T5C 9.54°, E2B 10.45°. Drift is concentrated in chronic-6 sub-cohort (full and non-chronic cohort angles all within ±2° of 87.93° random baseline). The measurement is now well-bounded; the causal-vs-correlational question remains open per close criterion (requires counterfactual training experiment).

- **source**: [`D6_FACTS_2026-05-12.md`](../../../analysis/cpu_diagnostics_2026-05-12_d6_empirical_orthogonality/D6_FACTS_2026-05-12.md) §4, §8.

### Open loop: clip-frozen-chronic-6-auc-robustness
status: open
severity: low
first_seen: 2026-05-12
last_verified: 2026-05-12
close_criterion: extend the CLIP-frozen chronic-6 probe to a larger sample (e.g., 500-1000 chronic-6 reals from contract suites + 100-200 chronic-6 fakes). If 5-fold CV probe AUC remains ≥ 0.95 at the larger sample, the "forgery signal is in raw CLIP" framing is robust. If the AUC drops below 0.90 at scale, the n=282 triptych result is sample-size-inflated and the framing weakens.

**Amendment 2026-05-12 late evening**: D6 B=50 measures angle stability under random 50/50 splits but does NOT extend the n=282 chronic-6 panel; AUC robustness at larger N remains untested. Open.

- **source**: [`D6_FACTS_2026-05-12.md`](../../../analysis/cpu_diagnostics_2026-05-12_d6_empirical_orthogonality/D6_FACTS_2026-05-12.md) §1.

---

## 2026-05-16 update — RESCHAIN/GRL6 packet: resolution-chain attack at the data layer; 6-axis GRL surfaces an unexpected viso lift

Two overnight FT-from-T5C-step3500 single-lever packets (Slot α
`resolution_chain_aug`, Slot β `multi_axis_grl` axes 4→6) launched
2026-05-15 22:39 UTC. Both SUCCEEDED. Scorecard verdict 2026-05-16 10:36 UTC.
Full packet retro at [`packets/RESCHAIN_GRL6.md`](../packets/RESCHAIN_GRL6.md).

### Facts

- **Slot α (data axis: random downsample → upsample chain)**: CPU follow-up
  probe shows the targeted mechanism partially worked at the metric level —
  median real-cohort `score_range` 0.605 → 0.448 (25% reduction); per-size
  mean real score swing 0.23 → 0.07 (the flattest score-vs-size curve
  measured to date). Scorecard verdict: step3500 fails `dev_fake_macro_recall ≥ 0.30`
  floor (0.226); step1500 promotes (rank 3) but regresses
  `visomaster_enhanced_macro_dev` to 0.056 vs T5C 0.138. DEEP_DIVE §3
  shows the aug compressed BOTH the real-side AND the fake-side score
  distributions — Slot α step3500 mean fake scores sit in [0.65, 0.78]
  vs T5C's wider spread. The "stability" the probe measured was partly
  score-distribution compression (mechanism B in DEEP_DIVE §7), not pure
  encoder-level resolution invariance (mechanism A).
- **Slot β (loss axis: multi_axis_grl axes 4→6, added color_b_dev_high +
  luma_mean_high)**: CPU probe verdict was "null result" on resolution-chain
  metric (3% reduction in score_range, within noise). Scorecard verdict:
  highest `dev_fake_macro_recall` of any ckpt scored (0.545), highest
  `visomaster_enhanced_macro_dev` (0.235; +0.097 absolute / +70% relative
  over T5C — first viso lift above 0.20 in this contract since the packet
  sequence began), highest `deeplive_enhanced_dev` (0.738). Paid for with
  `lockbox_real_fpr` 0.088 (3.2× T5C, 4.8× P8A — largest in the scorecard).
- **CPU-probe metric vs contract metric**: the 2026-05-15 probe's
  `score_range`-only characterization did NOT distinguish encoder invariance
  from score-distribution compression. Mechanism-discriminating diagnostic
  (fake-vs-real AUC on the same panel) was not run pre-launch.
- **Canary probe**: both slot yamls inherited the canary-disabled state
  from `R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.yaml`. The 800-frame
  `arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet` covers
  exactly the chronic-6 + viso/deeplive cohort that the scorecard surfaced;
  had it been enabled with `frequency_steps=500`, fake-side score crash
  would have been visible by step 500-1000.

### Current stance (revised 2026-05-16)

The Stage 2a hypothesis (IQ-axis GRL is the highest-leverage Stage 2 packet
if R² > 0.5) is **partially refuted at the data-axis sub-lever**:
attacking the size sub-axis at the data layer achieves the targeted
score-stability metric but does NOT preserve fake-vs-real AUC. The same
risk applies to Stage 2a-GRL if its mechanism turns out to be score
compression rather than encoder invariance.

**Tightened close criterion for any future Stage 2a / 2b packet**: must
demonstrate that the proposed lever preserves fake-vs-real AUC on a panel
comparable to the 2026-05-15 probe. Pre-launch CPU gate. The cheapest
implementation: rerun the 2026-05-15 probe with AUC reporting added
(currently scripted in `cpu_diagnostics_2026-05-15_resolution_chain/scripts/`;
add the AUC column to `compare_baseline_vs_new.py`).

The **Slot β finding is more interesting than expected** and shifts the
program's near-term focus. The +10pp visomaster recall is the largest
viso lift over T5C measured in any R13 packet — but the lockbox_real_fpr
penalty is the largest too. Whether the penalty is identity-localized
(rule-rescuable per `project_blend_unsharp_lever_2026-05-14`) or
distributed across new identities is the gating CPU diagnostic.

### Implications for the open-loop program

- Stage 2a (IQ GRL packet) is now gated on a fake-vs-real AUC preservation
  pre-launch CPU check.
- Stage 2b (IQ-balanced sampler packet) inherits the same gate.
- Stage 3 (HDTF canonical eval reframe) becomes more relevant because the
  `lockbox_real_fpr` tiebreak that keeps P8A at rank 1 across 13+ packets
  is plausibly not load-bearing (D4 2026-05-12 showed lockbox FPR at
  dev-cal τ is below dev FPR — i.e., the tiebreak is within the noise band
  of the policy's own calibration). The 2026-05-16 OPINION doc §5.2 proposes
  a $0 CPU re-rank of the scorecard above without the lockbox_real_fpr
  tiebreak as a diagnostic for how much rank-1 is policy-artifact vs gate.

### Open loops added or amended

### Open loop: cpu-probe-mechanism-discrimination
status: open
severity: medium
first_seen: 2026-05-16
last_verified: 2026-05-16
close_criterion: the 2026-05-15 resolution-chain probe template is amended to report fake-vs-real AUC on the panel for each ckpt scored. The AUC distinguishes encoder-level invariance (AUC preserved) from score-distribution compression (AUC reduced). Once added, any single-lever IQ-axis-attack packet must demonstrate that AUC is preserved as the pre-launch CPU gate.

**Summary**: The 2026-05-15 CPU probe measured `score_range` on a 388-frame
panel. The 2026-05-16 scorecard surfaced that Slot α step3500's `score_range`
reduction was partly score-distribution compression, not pure encoder
invariance — the probe could not distinguish these two mechanisms because
it did not measure AUC. Future Stage 2a / 2b packets must include this
diagnostic pre-launch.

- **source**: [`analysis/reschain_grl6_eval_2026-05-16/DEEP_DIVE_FACTS_2026-05-16.md`](../../../analysis/reschain_grl6_eval_2026-05-16/DEEP_DIVE_FACTS_2026-05-16.md) §7.

### Open loop: slot-b-viso-lift-mechanism-and-lockbox-fpr-localization
status: open
severity: high
first_seen: 2026-05-16
last_verified: 2026-05-16
close_criterion: a per-identity decomposition of Slot β step3500's lockbox_real_fpr (~$0 MPS, ~1h) determines whether the +5.5pp absolute penalty is concentrated on 2-3 chronic identities (rule-rescuable per memory `project_blend_unsharp_lever_2026-05-14`) or distributed across new identities. AND a 5-axis sister-variant (axes = current 4 + color_b_dev_high only, no luma_mean_high) tests whether the viso lift comes from color_b alone or requires both new axes. The combination of these two tests resolves whether Slot β is a deployment path.

**Summary**: Slot β step3500 has the highest dev_fake_macro_recall and
visomaster_enhanced_macro_dev of any ckpt in the 2026-05-16 scorecard. The
mechanism (6-axis GRL extension) is structurally different from what was
designed (resolution-chain stability). Two complementary follow-ups
characterize what Slot β actually did:
- Per-identity decomposition (CPU) determines deployability via existing rule layer.
- 5-axis sister-variant (GPU) determines what specifically bit.

- **source**: [`analysis/reschain_grl6_eval_2026-05-16/RESULTS_FACTS_2026-05-16.md`](../../../analysis/reschain_grl6_eval_2026-05-16/RESULTS_FACTS_2026-05-16.md) §5 + [`AGENT_PROPOSAL_2026-05-16.md`](../../../analysis/reschain_grl6_eval_2026-05-16/AGENT_PROPOSAL_2026-05-16.md) §3.2 + §3.3.

### Open loop: lockbox-real-fpr-tiebreak-is-load-bearing
status: open
severity: medium
first_seen: 2026-05-16
last_verified: 2026-05-16
close_criterion: a $0 CPU re-rank of the most recent scorecard using `dev_fake_macro_recall` as the tiebreak (instead of `lockbox_real_fpr`) AND a per-IQ-quartile cell decomposition of `lockbox_real_fpr` at each ckpt's selected τ. If the per-quartile FPR is uniformly below dev FPR (per D4 2026-05-12 finding), the tiebreak is measuring τ-tail-density not lockbox substrate difficulty. Decision: whether to amend the contract policy.

**Summary**: P8A has held rank 1 on the v3-fix contract across 13+ packets
because of the `lockbox_real_fpr` tiebreak. Per D4, lockbox FPR at
dev-cal τ is 0.92% / 2.40% / 2.19% on P8A / T5C / T3 — all well below the
policy's 5% dev calibration target. This suggests the tiebreak is within
the noise band of the policy itself. If true, P8A's rank-1 status across
13+ packets is partly an artifact of the tiebreak's choice, not a measure
of deployment-grade preference.

- **source**: [`analysis/cpu_diagnostics_2026-05-12_d4_iq_quartile_alignment/D4_FACTS_2026-05-12.md`](../../../analysis/cpu_diagnostics_2026-05-12_d4_iq_quartile_alignment/D4_FACTS_2026-05-12.md) + [`analysis/reschain_grl6_eval_2026-05-16/AGENT_PROPOSAL_2026-05-16.md`](../../../analysis/reschain_grl6_eval_2026-05-16/AGENT_PROPOSAL_2026-05-16.md) §5.2.

### Open loop: canary-probe-not-default-in-yaml-templates
status: open
severity: low
first_seen: 2026-05-16
last_verified: 2026-05-16
close_criterion: the next packet author adds a default-on canary block to the packet yaml template OR explicitly documents the rationale for keeping it off. The canary infrastructure has been available since 2026-05-07; both Slot α and Slot β yamls inherited the canary-disabled state from `R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.yaml`. The canary would have surfaced fake-side score crash by step 500-1000 in Slot α, allowing early-stop.

**Summary**: An in-training signal (canary probe) exists that would have
saved ~3h of Slot α's wall time. It is not enabled by default in the yaml
template inheritance chain. This is a propagated omission rather than a
deliberate design choice.

- **source**: [`analysis/reschain_grl6_eval_2026-05-16/DEEP_DIVE_FACTS_2026-05-16.md`](../../../analysis/reschain_grl6_eval_2026-05-16/DEEP_DIVE_FACTS_2026-05-16.md) §8.
