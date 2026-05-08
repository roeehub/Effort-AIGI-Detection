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

(Future entries: a/b/c results, then Stage 2a decision.)
