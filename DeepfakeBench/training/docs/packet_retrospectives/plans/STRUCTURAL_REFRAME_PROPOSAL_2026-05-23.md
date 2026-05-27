# Structural Reframe Proposal — Teams Deepfake Detector Training Program

**Date**: 2026-05-23
**Author**: working agent in session with user (Roee)
**Status**: draft, awaiting independent reviewer
**Decision asked**: approve / modify / reject a proposed structural change to the training paradigm (specifically, a single GPU experiment combining Invariant Risk Minimization with substrate-pair environments and a Variational Information Bottleneck head)

---

## How to read this document

This is a strategic proposal for an independent reviewer agent. You have no prior session context. The document is self-contained but cites specific files for evidence. Read in this order:

1. This document end-to-end (§1-§11)
2. Skim the cited memories and FACTS docs (paths in §References)
3. Read the outputs of two CPU probes that were running at draft time (paths in §9)
4. Form an independent view

The user explicitly wants pushback and alternative framings, not a rubber-stamp. The structural reframe is a hypothesis under test, not a foregone conclusion. The decision in question carries ~$80 of GPU cost — small enough to run as an exploratory probe, large enough that the experimental design should be defensible.

---

## §1 — Executive summary

**The diagnostic claim**: After 6 weeks and ~25-30 GPU training runs (cumulative ~$1,500-2,500), the deploy-relevant frontier of this training program has not moved meaningfully. The currently-deployed model (T5C step3500, switched in from E2B at unspecified date before 2026-05-23) fails today's team-identity binding gate on both the real-FPR side (dor 5.2% > 5% floor) AND the fake-recall side (Xinhe 24.7% << 50% floor). No ckpt produced in 6 weeks passes both gates at mode B (τ=0.78). The pattern of failure is consistent: shortcut suppression on one axis → encoder routes around → next shortcut becomes visible. This is structural, not "one more lever away."

**The proposed reframe**: change the mode of attack from "shortcut suppression on a fixed data + task formulation" (where we've been for 6 weeks) to one of four categories of structural change (data acquisition / task reformulation / information bottleneck / generation-pipeline). Two of those four (data, generation-pipeline) are infeasible or partial in the near term per user constraints. The remaining two (task reformulation, information bottleneck) are cheap-to-medium GPU cost and structurally distinct from prior work.

**The specific recommendation**: a single GPU experiment combining (B1) Invariant Risk Minimization with substrate-pair environments and (C1) Variational Information Bottleneck on the bottleneck layer. ~$80, ~5h wall. Designed so the result is informative regardless of whether it produces a deploy-grade ckpt — if it fails to move the team-identity bar, that is itself evidence that the data is the binding constraint (which informs the data-acquisition decision).

---

## §2 — The big picture: what we are building and the operational gate

### 2.1 The task

A single-model, real-time, per-frame deepfake detector deployed as a hook into Microsoft Teams calls. Input: face crops from the Teams capture pipeline. Output: calibrated `prob_fake ∈ [0, 1]`. Aggregation, thresholding, and alerting policy live downstream of the model. The model is responsible only for the per-frame score.

Authoritative spec: `docs/packet_retrospectives/MODEL_GOALS.md`.

### 2.2 Hard constraints (rule out classes of solutions)

- **NO ENSEMBLE** (MODEL_GOALS.md §"Single model"). Forbidden shapes: score-fusion, multi-checkpoint min/max, specialist-per-method, cascade-of-models. Backed by `project_job12_ensemble_ceiling_2026-05-04`: label-free ensembles peaked at 16.2% viso < P8A 27%.
- **NO per-substrate τ at deploy** (MODEL_GOALS.md §"Substrate-aware τ" + `feedback_per_mode_tau_not_deployable`). Teams doesn't surface capture-mode at inference; a mode classifier is 52% CV. Per-substrate / per-mode τ is offline-only.
- **Production target is Lenovo laptops** (user, 2026-05-23). Mac-captured frames are out-of-scope for the deploy gate, though they may appear in the data.

### 2.3 The binding gate (team-identity bar, established 2026-05-23)

The team is 5 humans across multiple data labels — see `project_team_identities_multi_labeled_2026-05-23.md`:

| Team human | Real-frame pools | Fake-attack pools |
|---|---|---|
| Roee | `roee-real-windows-laptop-correct` (Windows only), `team_may5__Roee`, `roee_tester_real_*`, `tester_roee_real_*` | — |
| dor | `dor_*_laptop_correct`, `dor_*_webcam`, `dor_evening`, `dor_morning`, `team_may5__Dor`, `real_dor` | `dor_fake_local`, fake-target entries in dev/lockbox |
| Noyn (Sharker) | `team_may5__Noyn`, `noyn_Sharker__s*` | — |
| Xiang | `team_may5__Xiang`, `extra_xiang__real` | `live_prod__xiang-fake-{1..6}`, `Xiang_Xiang2_Feng__*__fake`, `extra_xiang__fake` |
| Xinhe | `team_may5__Xinhe`, `xinhe_may6_falseflag` | `live_prod__xinhe-fake-{1..11}` (some -glasses variants) |

**Bar**: per-team-human FPR ≤ 5% AND per-team-human fake-recall ≥ 50% at the chosen deployment τ.

**Today's measurement** (`analysis/team_identity_deploy_readout_expanded_2026-05-23/`, ~6400 frames scored across the 5 ckpts):

| Ckpt | Max per-human real FPR @ mode B | Min per-human fake recall @ mode B | Pass? |
|---|---:|---:|---|
| P8A | 0.016 | 0.487 (Xinhe) | Mode B fails on fake-recall floor; mode A passes both |
| E2B | 0.036 | 0.473 (dor) | Mode B fails both floors marginally |
| **T5C (current production)** | **0.052 (dor)** | **0.247 (Xinhe)** | **Fails both floors on dor + Xinhe** |
| Slot A v2 CLS | 0.024 | 0.293 (Xinhe) | Fails fake-recall floor |
| Slot A v2 face-pool | 0.003 | 0.252 (dor) | Fails fake-recall floor |

**Headline**: no ckpt passes both gates at mode B. P8A alone passes both at mode A (τ=0.535) but is older and worse on Xinhe-fake-attacks than the current production T5C. The current production T5C fails both gates.

---

## §3 — The shortcut taxonomy (what the encoder demonstrably uses)

In 6 weeks of CPU diagnostics (D1-D10 series 2026-05-11/12, E1-E9 series 2026-05-19, plus inline probes through every packet), the following shortcut axes have been quantified with citations:

| # | Axis | Evidence | Magnitude |
|---|---|---|---|
| 1 | Image-quality (Laplacian/sharpness/codec artifacts) | D7 FACTS 2026-05-12 | IQ block explains 23-31% of real-FPR variance |
| 2 | Camera/ISP signature | RLP6 → P8A camera-flip finding; E5 cross-ckpt sign-flip | P8A vs E2B sign-flip on 5 IQ axes on SAME data |
| 3 | Substrate-transfer gap | D8 KLIEP discriminator in CLIP-frozen feature space | 99.09% accuracy distinguishing dev-real from lockbox-real |
| 4 | Per-method face-pixel-area | 2026-04-27 audit | Each fake method clusters at a tight face-size band; cohen's d > 0.5 across methods |
| 5 | Webcam-mode dominance | 2026-04-27 `project_lockbox_fpr_dominated_by_webcam_mode` | webcam captures = 65.7% within-mode FPR |
| 6 | Per-identity memorization (chronic-6) | D2/D5/D6 2026-05-12 | chronic-6 IQ-PC1 angles 70-78° (vs 87.93° random baseline) |
| 7 | Resolution-chain instability | 2026-05-15 CPU probe | 0.47 absolute score swing across downsample sizes on P8A real frames |
| 8 | Spatial geometry (face vs non-face) | CPU-1 2026-05-22 (saliency map) | viso fakes 72.6% non-face saliency mass; face-pool readout loses 10pp viso recall |
| 9 | Sign-flip across ckpts on IQ axes | E5 2026-05-19 | P8A r vs T5C r vary by 0.5+ on 5 IQ axes on same data — same training, different shortcuts found |
| 10 | Device-axis (Mac vs Windows webcam) | 2026-05-23 | Same person (Roee) gets 24%+ FPR on Mac captures, ~0% on Windows captures |

The encoder uses these. It is not speculation. Each row above is a measurement.

### 3.1 The Probe 1 finding that frames the structural problem

Per `analysis/substrate_pair_geometry_2026-05-22/FALLBACK1_PROBE1_FACTS_2026-05-22.md`: each FT'd encoder (P8A, Slot A v2, T5C) learns its own substrate-discrimination axis at ~98% held-out accuracy on 1825 paired clean↔teams identity captures. The cosines between the three trained encoders' axes and the frozen-CLIP-L11 KLIEP axis: 0.04, 0.10, 0.06 (≈85-88° angular separation). The trained encoders agree with each other more (cos 0.69-0.92) than with the frozen prior — but they ALL diverge from the frozen prior by similar angles.

**Interpretation**: FT moves the encoder a LOT — in a different direction than the frozen prior. Each FT recipe finds SOME discriminative axis at high accuracy. The axes that get found are not arbitrary; they all rotate away from the frozen-CLIP prior in similar ways. **The encoder is learning to discriminate substrate, not the binary task**, and different FT recipes find different substrate-shortcuts.

---

## §4 — The refuted-lever catalog (what doesn't work)

Single-lever training-time interventions tried since 2026-04 onward, with verdicts:

| Lever | Packet | Result | Citation |
|---|---|---|---|
| pipeline_randomization | P22 | step1k robust, step8k degraded; ensemble of P8A+step1k worked at $0 (forbidden per rules) | `project_p22_*` |
| face_scale_jitter@0.50 | P14 jitter | Best single-lever value_composite but didn't promote on contract | `project_face_scale_jitter_load_bearing` |
| Fourier band aug | U_SLOTS Slot 4 | B16-scratch+Fourier didn't break viso ceiling | `U_SLOTS_2026-05-13.md` |
| resolution_chain_aug | Slot α 2026-05-15 | Composite metric fired (score range cut 25%) but failed dev_fake_macro_recall floor at 0.226 | `project_overnight_resolution_chain_2026-05-16` |
| 6-axis GRL | Slot β 2026-05-15 | Highest viso recall but 4.8× P8A lockbox FPR | `project_overnight_resolution_chain_2026-05-16` |
| 6-axis GRL + anchor_aware (stack) | T5C_TRIPLE Slot 1 | Monotonic decline in lockbox recall during training | `T5C_TRIPLE_2026-05-19.md` |
| 5-axis GRL no-luma | T5C_TRIPLE Slot 3 | ≈ Slot 1 — luma axis was NOT the load-bearing one | same |
| LoRA L10-L11 r=16 | U_SLOTS Slot 1+2 (P8A + T5C bases) | +74pp viso lift hard-coupled to 24% lockbox FPR | `project_lora_l10_l11_chronic_fp_hard_coupled_2026-05-15` |
| LoRA L8-L9 r=8 | T5C_TRIPLE Slot 2 | Indeterminate due to `load_model` infra bug; in-training W&B canary suggests parity-to-degradation | `T5C_TRIPLE_2026-05-19.md` |
| T5C + jitter@0.30 | U_SLOTS Slot 3 | Regression | same |
| Codec aug (3 variants) | Codec triple 2026-05-21 | Bit transport axis 37-60% but regressed may6 on all 3 | `project_overnight_3_packets_2026_05_20` |
| anchor_aware | Slot A v2 | **THE ONE WIN** — fixed Chikara_Takahashi / PC_Generator / Q chronic FPs; introduced bla_bla_chow chronic; was rank-2 on 9-suite/29-suite | `project_band_shortcut_ood_hypothesis_2026-05-16` |
| real_rebalance | Slot B 2026-05-16 | Refuted — lockbox FPR blew up 4.8× P8A | same |
| HEAD face-pool retrain | Phase 2 HEAD 2026-05-22 | Plateaued at composite 0.235 (over 0.20 gate); viso stuck at 7% | `project_head_face_pool_verdict_iterate_2026-05-23` |
| GroupDRO substrate-balanced | Phase 2 BACKBONE-SlotAv2 2026-05-23 | ABORT — boundary shifted to τ=0.826, viso collapsed to 0.5% | `project_backbone_slotav2_groupdro_abort_2026-05-23` |
| Asymmetric pair-loss | Phase 2 BACKBONE-T5C 2026-05-23 | BLOCKED — sampler design bug; fixed today; not yet rerun | `project_backbone_t5c_blocked_pair_sampling_design_2026-05-23` |
| Roy_D-specific anchor pool | proposed never run | — | `roy-d-specific-anchor-pool-packet` open loop |
| Output-preservation aux loss | proposed never run | — | 2026-05-20 candidate list |
| Distillation from anchor-aware | proposed never run | — | same |

**Pattern**: ~20 single-lever interventions in 6 weeks. One genuinely "won" something measurable (anchor_aware on dor cluster) but doesn't generalize beyond its trained pool. The rest are refuted, indeterminate, or marginal. **Cumulative information value of the refutations is real and load-bearing for this proposal** — they collectively rule out a wide span of "tweak the loss / augmentation / data weighting" interventions on the current data + task formulation.

---

## §5 — The structural diagnosis

### 5.1 Why shortcut-suppression has a structural ceiling

The encoder is a function approximator with capacity. When trained with cross-entropy on (real, fake) labels, it learns ANY signal that separates the labels. If the data contains 10+ such signals (the taxonomy in §3) and one of them is the "true" signal (real generative artifacts), the encoder will preferentially learn whichever signal is most easily extractable.

The shortcut signals enumerated in §3 are blunt and high-bandwidth. The true generative-artifact signal is subtle and low-bandwidth. Cross-entropy will preferentially find the blunt shortcuts. Adding a loss term to suppress shortcut N causes the encoder to either:
- (a) Route around it (use shortcut N+1 instead) — see U_SLOTS Slot 4 Fourier-aug suppressing specific bands → encoder used different bands
- (b) Pay the suppression cost on the suppression axis but reorganize representations to use OTHER signals more aggressively — see Slot β GRL on 6 IQ axes: viso recall up, lockbox real-FPR also up 4.8× because encoder used the freed-up capacity to find other axes

This is whack-a-mole with structural inevitability. We are not failing because we haven't found the right loss term yet. We are failing because **the encoder will always find a shortcut as long as any shortcut is easier to learn than the true signal**, and the true signal IS harder than any of the 10 enumerated shortcuts.

### 5.2 The single most-diagnostic measurement

Probe 1 (2026-05-22) showed each FT'd encoder's substrate-discrimination axis at ~98% held-out accuracy, rotated 85-88° from the frozen-CLIP prior, and rotated differently from each other. **FT is efficient at finding shortcuts** — different FT recipes find different shortcuts at near-perfect accuracy. The true signal isn't being found because nothing in the training setup makes it easier to find than the shortcuts.

### 5.3 The structural implication

The required intervention is one of:
- **(A) Data**: make the shortcut signals genuinely uninformative by training on data where they don't correlate with the label (massive diversity)
- **(B) Task**: explicitly model the confounders so the binary task only has the residual (multi-task / IRM / temporal change-of-task)
- **(C) Architecture**: constrain the encoder's representational capacity so blunt shortcuts can't all fit (information bottleneck)
- **(D) Pipeline**: close the train/test pipeline gap on the fake side (production-pipeline fake generation)

We have done zero of these in 6 weeks. Every intervention we tried was a variant of "modify training data weighting or loss on the same task on the same encoder with no structural constraint."

---

## §6 — The four structural categories, ranked by feasibility

### (A) Data — radically more diverse real captures

**What it would take**: 100K+ video-call captures across diverse cameras, users, lighting, codecs. Replace or massively augment the current 100%-YouTube-origin real-side training distribution (per D9 2026-05-12).

**User-stated feasibility (2026-05-23)**: ~7000 webcam videos might be acquirable via a dataset purchase. Not guaranteed to represent production webcams. Possibly old data. This is helpful (real diversity > zero) but not transformative (still bounded by what's in the purchased set).

**Cost**: data-acquisition project, weeks-to-months, plus storage + ingestion engineering.

**Verdict**: pursue opportunistically if practical, don't gate other work on it.

### (B) Task reformulation

Four sub-options ranked below in §7.

**Verdict**: highest near-term EV per dollar. Categorically distinct from prior work.

### (C) Information bottleneck

Three sub-options ranked below in §7.

**Verdict**: compose-able with (B). Same verdict.

### (D) Generation pipeline — closing train/test fake-pipeline gap

**What it would take**: run real captures through a deepfake method, then re-run through the Teams capture pipeline, generating training fakes that exactly match the production fake distribution.

**User-stated feasibility (2026-05-23)**: "I can't create live deepfakes at scale over a webcam." The Teams capture pipeline includes physical-camera capture + network compression + Teams ISP processing; replicating the FAKE side of this chain at scale (where the fake is then captured by a physical camera) is operationally infeasible.

**Verdict**: not viable in the near term. Defer.

---

## §7 — The specific experiments under (B) and (C), and the recommended combination

### (B1) Invariant Risk Minimization with substrate-pair environments

**Reference**: Arjovsky et al. 2019 "Invariant Risk Minimization" (and Krueger 2020 V-REx variant).

**Premise**: if the encoder's representation is "right" (captures only causal signal, not spurious correlation), then the optimal classifier on top of it should be invariant across different "environments" where the spurious correlations vary but the causal relationship is the same. The IRM objective penalizes the variance of the per-environment classifier gradient — high variance means the classifier is using environment-specific signal.

**Our environments**: We have the substrate-pair data wiring from earlier this week (1,880 paired clean↔teams identity captures from A0.1 inventory, today's sampler fix enables matched-pair in-batch co-occurrence). Each substrate-pair instance gives a (real-clean, real-teams) pair from the same identity. Environment partition options:
- Per-substrate (clean vs teams) — small partition (2 envs), most direct
- Per-source (HDTF vs quickclips vs visomaster_teams_enhanced) — 3 envs
- Per-identity-cluster — many envs, more data per cluster but smaller absolute training set

**Implementation**: ~20-50 LOC in trainer/loss. IRM penalty: `λ_IRM · sum_env(||∇_w L_env(w · z, y)||² @ w=1)` added to the standard CE loss. With β-annealing (start IRM penalty at 0, ramp up over first 500 steps so the model finds a useful representation before being asked to make it invariant).

**Why it's different from prior GRL work**: GRL on a hand-engineered axis (Slot β's 6-axis GRL) tries to remove that specific axis from the representation. IRM is structurally different — it constrains the OPTIMAL CLASSIFIER to be invariant across environments. The encoder can carry shortcut information; it just can't USE it differently per environment. Empirically IRM and V-REx have shown OOD-generalization gains on benchmarks where naive GRL fails (Colored MNIST, CivilComments, Camelyon17 in Domainbed).

**Cost**: ~$50-100. Single Vertex run on existing infra. Wall ~4h.

**Risks**:
- IRM has known instability — the penalty can dominate the CE loss and prevent learning entirely; β-annealing helps but isn't a guarantee
- Substrate-pair environments are a small subset of total training data (~1880 pairs vs the full corpus); the IRM penalty may not bite if the environment partition is too narrow
- Recent literature (Rosenfeld 2020) has shown IRM under-performs on some benchmarks vs simple ERM; the gain is task-dependent

**Expected information value**: high. Even a null result has meaning — it would suggest that the substrate-pair environment partition is too narrow to capture the shortcut variance, which points to needing a broader environment partition (per-method, per-device) that we'd build out in follow-up work.

### (C1) Variational Information Bottleneck (VIB) on the head

**Reference**: Alemi et al. 2017 "Deep Variational Information Bottleneck"; Achille-Soatto 2018 "Information Dropout".

**Premise**: replace the deterministic encoder output with a Gaussian distributional layer (μ, σ from the encoder), sample z ~ N(μ, σ²) at training time, pass z to the classifier head, and add a KL penalty between p(z|x) and a Gaussian prior N(0, I). This explicitly minimizes I(X; Z) subject to I(Z; Y) being preserved — the information-bottleneck objective.

**Why this might help**: shortcut signals are high-bandwidth (camera signature uses many pixel statistics, IQ profile is multi-dimensional, identity is high-dimensional). A KL penalty preferentially preserves what's load-bearing for the task and squeezes out the rest. The true generative-artifact signal is subtle but task-relevant; shortcuts are blunt but not necessarily task-relevant on the OOD test distribution. In theory, VIB representations generalize OOD better than deterministic ones.

**Implementation**: ~30 LOC. Replace the head's first linear layer with two parallel linear layers (one for μ, one for log σ²). Sample via reparameterization trick at training time. Add `β · KL(N(μ, σ²) || N(0, I))` to the loss. β-anneal from 0 to a small value (e.g., 1e-3 to 1e-2) over warmup.

**Cost**: ~$30-50. Single Vertex run, possibly combined with B1.

**Risks**:
- Vibes aren't a magic bullet. The encoder might learn to encode shortcut information in the mean while saturating the variance to satisfy the KL penalty.
- β choice is sensitive — too small = no regularization, too large = collapse. Some scaffolding required.
- VIB applied only to the head (not throughout the encoder) limits the bottleneck's reach; the full information-bottleneck objective is on the whole representation.

### Why combine B1 + C1 in a single experiment

Both are structural reframes attacking the same underlying problem ("encoder finds shortcuts because nothing constrains it to find the true signal") from different angles. They are mathematically compatible — the IRM penalty operates on the classifier gradient, the VIB penalty operates on the encoder's representation distribution; both are additive loss terms.

The combined experiment tests **both structural reframes simultaneously with one Vertex run**. If it lands a deploy-grade ckpt, we know the combination works (and can ablate in follow-up to attribute the gain). If it doesn't move the team-identity bar at all, we have strong evidence that BOTH reframes are at-best-marginal under the current data + encoder, which points to needing (A) data-side work as the binding constraint.

**Failure mode if we DON'T combine**: running B1 alone, then C1 alone, costs 2x. Running them separately first gives cleaner ablation but takes more total time. The combined run is the faster path to a directional verdict; ablation can follow.

**Combined experimental spec**:
- Base: Slot A v2 step3500 (current best deploy-relevant ckpt; encoder weights to start from)
- IRM environments: per-substrate (clean vs teams) on the substrate-pair data; ERM on all other training data with IRM penalty weight 0
- VIB: head's first linear layer replaced with μ/σ heads; β=1e-3 with linear ramp from 0 over first 500 steps
- Anchor-aware: ON (preserve the one win we have)
- Periodic ckpts: 250, 500, 1000, 1500, 2500, 3500
- Cost: ~$80, ~5h wall on single A100
- Scoring: per-team-human FPR + fake-recall on the cohort built today

**Success criteria**:
- **Primary**: per-team-human FPR ≤ 5% AND fake-recall ≥ 50% on at least one ckpt at mode B (the bar no current ckpt clears)
- **Secondary**: per-team-human metrics strictly Pareto-improve Slot A v2 step3500 at mode B
- **Informative-even-if-failure**: composite λ=1.0 within ±0.05 of Slot A v2 step3500 AND substrate-pair gradient variance reduced by ≥30% vs baseline. If this holds AND the team-identity bar still doesn't move, the diagnosis is "structural reframes are working as designed but the data doesn't contain enough invariant signal" — strong support for prioritizing (A) data acquisition.

---

## §8 — Alternatives considered and explicitly rejected

### Alternatives we explicitly considered

| Alternative | Reason rejected |
|---|---|
| HEAD ALT (dual face+non-face readout → 1024-dim head) | Phase 2 HEAD already failed today (plateaued at composite 0.235 / viso 7%). The dual-readout variant is a head-architecture change; under the team-identity gate from today, the binding constraint is fake-recall on Xinhe and dor, not viso non-face localization specifically. EV moved from 25-35% (this morning's estimate) to ~10-15% under the team-identity reframe. |
| Another GroupDRO variant (smaller β, clip_max) | Phase 2 BACKBONE-SlotAv2 ABORTed today — the boundary-shift direction is structural to worst-group reweighting; smaller β attenuates magnitude not direction. EV ~5-10%. |
| Another LoRA placement (L5-L7, L11-L12) | LoRA L10-L11 hard-coupled to 24% lockbox FPR; LoRA L8-L9 indeterminate (infra bug). The placement axis has been pulled twice with the same family of failure. EV ~10%. |
| Roy_D-specific anchor pool extension | Doesn't address fake-recall failure on Xinhe (the binding constraint per today's scan). Was high-EV before today's findings; lower now. |
| (B2) Temporal model (3-5 frame input) | Highest possible upside but $300-800 + 1-2 weeks of infra work; Teams compression may wipe out temporal signal before model sees it (requires pre-experiment to verify). Defer until cheaper structural levers tried. |
| (B3) Multi-task with per-method classifier as auxiliary task | Lower EV than B1 — the per-method axis is one shortcut among many; B1 IRM addresses the broader environment-invariance problem. If B1+C1 fails, B3 is a reasonable follow-up. |
| (B4) Self-supervised pretrain on video-call data | $500-1500 GPU + weeks of work for the pretrain step itself. Pre-mature given cheaper structural options unexplored. |
| (C2) Severely-restricted-rank LoRA + MI regularizer | More complex implementation than VIB (HSIC or InfoNCE has its own hyperparameters and stability considerations). Same theoretical EV as VIB but harder to implement correctly. |
| (C3) Smaller bottleneck dim in head | Phase 2 HEAD essentially did this (1026 trainable params on frozen encoder) and plateaued. The version that's untested is small bottleneck DURING training of the encoder, but that's a more invasive architecture change than VIB. |

### Why not "ship current production T5C + improved monitoring"

User has explicitly rejected this (2026-05-23): "I have no time or desire to start tinkering specific spray identity like that it feels like another battle loop." Monitoring per-team-identity in production is treated as another whack-a-mole loop. The user wants structural change, not better failure observation.

---

## §9 — Concurrent CPU probes (running at draft time)

Two CPU probes are running in background as of this document's draft time. Their outputs will land before the reviewer's decision is needed. Both are $0 and inside the standing CPU greenlight.

### 9.1 Frozen-CLIP team-identity baseline

**Path**: `analysis/frozen_clip_team_identity_baseline_2026-05-23/` (when complete)

**Question**: how much is FT actually buying us over a properly-trained frozen-CLIP linear-head baseline, scored on the team-identity deploy cohort? Prior work (D8 2026-05-12) established frozen-CLIP linear head at 0.679 dev→lockbox AUC vs P8A's much higher headline AUC, but with apples-to-oranges training corpus and substrate-transfer (not team-identity) metric. This probe is the apples-to-apples team-identity version.

**Decision relevance**:
- If frozen-CLIP linear/MLP head within ~10pp of T5C on team-aggregate FPR + min-fake-recall: FT has been approximately net-zero on the population we deploy to. Strong evidence that the next experiment needs to be structural; B1+C1 is a candidate but data-side work moves up in priority.
- If frozen-CLIP much worse than T5C: FT is doing real work — the question becomes "what FT recipe class actually beats current T5C," for which B1+C1 is the proposed answer.

### 9.2 T5C on Xinhe-may6 92-frame cohort

**Path**: `analysis/xinhe_may6_t5c_revisit_2026-05-23/` (when complete)

**Question**: does T5C (current production) show the catastrophic Xinhe-may6 regression that E2B was claimed to in May (E2B: 57.6% FPR @ τ=0.5; P8A: 0% on same 92 frames)? If T5C has E2B's failure mode, current production is approximately broken on a real teammate's production-substrate frames — operational urgency.

**Decision relevance**:
- If T5C is at-least may6-stable: structural-reframe conversation can proceed at deliberate pace
- If T5C is catastrophic on Xinhe-may6: urgency to switch to a non-broken alternative (P8A is the obvious candidate per the original memo). Doesn't change the structural-reframe direction but adds time pressure to having a better alternative available

---

## §10 — Risks, caveats, and uncertainties

### 10.1 Risks to the B1+C1 experiment specifically

1. **IRM instability**: well-documented in literature. The penalty can dominate the CE loss and prevent learning. Mitigation: β-annealing, monitor train-loss curves, abort early if loss explodes.
2. **VIB shortcut routing**: the encoder might learn to encode shortcut information in the mean while saturating the variance to satisfy the KL penalty. Detection: at end-of-training, compute mutual information between μ output and known shortcut axes (sharpness, identity); if MI is high, VIB didn't bite.
3. **Substrate-pair environments are narrow**: 1,880 pairs vs the full corpus. IRM penalty may not bite hard enough to constrain the model meaningfully. Mitigation: ablation in follow-up with broader environment partitions (per-method, per-device).
4. **Compound experiment makes attribution harder**: if B1+C1 works, we don't know whether B1 or C1 or both did the work. Follow-up ablation runs needed before drawing structural conclusions. Cost: ~$30-50 each for the ablations.

### 10.2 Risks to the structural-reframe diagnosis itself

The diagnosis says "shortcut suppression on fixed data + task has a structural ceiling." This is supported by:
- The pattern of ~20 refuted single-lever interventions
- Probe 1's measurement that FT'd encoders find different substrate axes each at high accuracy
- The team-identity bar failure across all 5 candidates

But the diagnosis could be wrong if:
- The shortcut suppression interventions tried weren't well-implemented (most had some compromise — see Slot 2 LoRA load_model bug, Slot 1+3 canary-silence bug, etc.)
- The team-identity bar is misspecified (50% fake-recall floor may be too tight; under a 40% floor, more candidates would pass)
- The right loss term genuinely hasn't been tried yet (always possible)

Reviewer should probe: how confident is the diagnosis? Are there unexplored single-lever interventions inside the current paradigm that haven't been tried?

### 10.3 Uncertainty about the production state

The promotion-bar inconsistency from earlier today (MODEL_GOALS.md says beat E2B on dev_fake_macro_recall ≥ 0.508; no current ckpt does; yet various candidates have been "deployed" or "deployment candidates") suggests the working-promotion-rule has shifted from documented to implicit over the last several weeks. The reviewer should probe whether the team-identity bar is now the binding promotion rule, or whether the documented MODEL_GOALS bar is still in force and we've simply been operating under a relaxed version.

---

## §11 — Open questions for the reviewer

1. **Is the structural-reframe diagnosis correctly characterized?** Specifically, is the pattern of ~20 refuted single-lever interventions sufficient evidence that the regime has hit a ceiling, or is it consistent with "we haven't tried the right single lever yet"?

2. **Is B1+C1 the right specific experiment to test the reframe?** Alternatives to consider: B1 alone first (cheaper, cleaner ablation), C1 alone first, B3 (per-method aux task — different angle), B1 with broader environment partitions (per-method instead of per-substrate).

3. **Are the substrate-pair environments the right environment partition for IRM?** The substrate axis is one of 10 enumerated shortcuts. Other partitions: per-identity-cluster, per-method, per-device. The "right" partition depends on which shortcut is the most binding for OOD generalization to the team-identity cohort.

4. **Should we run a small-scale dry-run of just IRM ($30) or just VIB ($30) before the combined run?** Tradeoff: $30 + $30 + $80 = $140 total for sequential vs $80 for combined. Sequential gives cleaner ablation but slower verdict.

5. **Should the 7000-webcam dataset acquisition be pursued in parallel?** The user says it's "not guaranteed to represent production webcams" but it's the only data-side lever available. Acquiring it doesn't gate the B1+C1 experiment, but if it lands within the B1+C1 turnaround it could inform a follow-up experiment.

6. **What if B1+C1 fails the team-identity bar but improves substrate-pair gradient variance by 30%+?** Per the "informative-even-if-failure" criterion, this would support prioritizing (A) data acquisition. Is that the right interpretation, or is there a different read?

7. **Are there architectural alternatives I haven't considered?** The full enumeration considered B/C/D classes; within those, the four-option B and three-option C lists are not exhaustive. The reviewer should probe whether other categories of structural change should have been on the table.

---

## §References (file paths to cited evidence)

### Authoritative project docs

- `docs/packet_retrospectives/MODEL_GOALS.md` — the operational constraint set, NO ENSEMBLE rule, no-per-substrate-τ rule, target methods, FPR budget
- `docs/packet_retrospectives/AGENTS.md` — wiki read/update protocol
- `docs/packet_retrospectives/STATE.md` — rolling current-state snapshot
- `docs/packet_retrospectives/TIMELINE.md` — chronological master index
- `docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md` — the strategic program thread

### Today's expanded team-deploy readout (the binding bar)

- `analysis/team_identity_deploy_readout_expanded_2026-05-23/RESULTS_FACTS_2026-05-23.md`
- `analysis/team_identity_deploy_readout_expanded_2026-05-23/AGENT_PROPOSAL_2026-05-23.md`
- `analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/per_human_summary.csv`
- `analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/per_cohort_summary.csv`
- `analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/ship_verdict.csv`

### Memories establishing the team-identity reframe

- `~/.claude/projects/.../memory/project_team_identities_multi_labeled_2026-05-23.md` — 5 team humans, multi-label-per-human, Mac/Windows device split
- `~/.claude/projects/.../memory/project_production_is_t5c_not_e2b_2026-05-23.md` — current production is T5C
- `~/.claude/projects/.../memory/feedback_cpu_jobs_standing_greenlight.md` — standing CPU authorization

### Today's other key memories (Phase 2 verdicts)

- `~/.claude/projects/.../memory/project_backbone_t5c_blocked_pair_sampling_design_2026-05-23.md` — T5C asymmetric pair-loss BLOCKED; sampler fix landed today
- `~/.claude/projects/.../memory/project_backbone_slotav2_groupdro_abort_2026-05-23.md` — GroupDRO ABORT, boundary-collapse mechanism
- `~/.claude/projects/.../memory/project_head_face_pool_verdict_iterate_2026-05-23.md` — HEAD face-pool ITERATE

### The shortcut taxonomy (CPU diagnostics D1-D10)

- `analysis/cpu_diagnostics_2026-05-12_d1_p8a_iq_alignment/D1_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d2_encoder_separation/D2_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d3_cross_substrate_probe/D3_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d4_iq_quartile_alignment/D4_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d5_per_identity_iq/D5_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d6_empirical_orthogonality/D6_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d7_fpr_decomposition/D7_FACTS_2026-05-12.md`
- `analysis/cpu_diagnostics_2026-05-12_d8_head_retrain_substrate_balanced/D8_FACTS_2026-05-12.md` (prior partial frozen-CLIP baseline)
- `analysis/cpu_diagnostics_2026-05-12_d9_source_substrate_inventory/D9_FACTS_2026-05-12.md` (real-side training distribution: 100% YouTube origin)

### Probe 1 (the encoder-axis-rotation finding)

- `analysis/substrate_pair_geometry_2026-05-22/FALLBACK1_PROBE1_FACTS_2026-05-22.md`
- `analysis/substrate_pair_geometry_2026-05-22/per_ckpt_face_region_cosines.csv`

### Recent failed packets (the refuted-lever catalog source)

- `docs/packet_retrospectives/packets/U_SLOTS_2026-05-13.md`
- `docs/packet_retrospectives/packets/T5C_TRIPLE_2026-05-19.md`
- `docs/packet_retrospectives/packets/AUTO_MODE_ANCHOR_REBALANCE_PREVIEW.md`
- `docs/packet_retrospectives/packets/FACE_POOL_2026-05-22.md`
- `analysis/phase_3_scorecard_2026-05-23/RESULTS_FACTS_2026-05-23.md` (today's Phase 2/3 verdict)

### Concurrent CPU probes (paths will exist when probes complete)

- `analysis/frozen_clip_team_identity_baseline_2026-05-23/RESULTS_FACTS_2026-05-23.md`
- `analysis/xinhe_may6_t5c_revisit_2026-05-23/RESULTS_FACTS_2026-05-23.md`

### Academic references for the proposed experiment

- Arjovsky et al. 2019, "Invariant Risk Minimization" (https://arxiv.org/abs/1907.02893)
- Krueger et al. 2020, "Out-of-Distribution Generalization via Risk Extrapolation (V-REx)" (https://arxiv.org/abs/2003.00688)
- Alemi et al. 2017, "Deep Variational Information Bottleneck" (https://arxiv.org/abs/1612.00410)
- Achille & Soatto 2018, "Information Dropout: Learning Optimal Representations Through Noisy Computation" (https://arxiv.org/abs/1611.01353)
- Tishby et al. 2015, "Deep Learning and the Information Bottleneck Principle" (https://arxiv.org/abs/1503.02406)

---

## §End — explicit decision asks

The reviewer's deliverable: a written response (sibling file or appended section to this document) answering at minimum:

1. **Verdict on the structural-reframe diagnosis**: agree / partial-agree / disagree, with reasoning
2. **Verdict on the B1+C1 experiment**: approve / modify / reject, with reasoning
3. **If modify**: what specific changes (sequential vs combined? different environment partition for IRM? different VIB hyperparameters? include B3 or substitute one for C1?)
4. **If reject**: what should be done instead, with concrete cost estimate
5. **Recommendation on the 7000-webcam dataset acquisition**: pursue in parallel / pursue serial / defer
6. **Recommendation on operational urgency**: contingent on the CPU probes' results (T5C-may6 outcome especially), is there a near-term ship-the-best-current-candidate action that should run in parallel with the structural experiment?

The user (Roee) will adjudicate between the working agent's recommendation and the reviewer's response. The decision in question is committing ~$80 GPU spend on a single experiment, plus opportunity-cost of not running other things during the same window.

End of proposal.
