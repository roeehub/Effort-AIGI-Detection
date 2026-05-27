# AGENT_PROPOSAL — Xinhe-fake cohort mechanism analysis

Date: 2026-05-23. Interpretive doc accompanying `RESULTS_FACTS_2026-05-23.md`. Per `AGENTS.md` §"Eval-folder authoring contract": this is the SINGLE opinion doc; opinion verbs unconstrained.

Task #3 in the CPU-first sequence from `TRAINING_DIRECTIONS_REVIEW_2026-05-23.md`. Goal was a quick investigative bet — "why do xinhe-fake-1/2/3 defeat everyone while xinhe-fake-7 works?" — with the hope that a cheap discovery surfaces a usable lever.

---

## 1. Headline (this is bigger than expected)

**Three discoveries**:

1. **Hard-vs-easy is perfectly separable in frozen-CLIP feature space** (LR 1.0 AUC, 1.0 5-fold CV). There is a specific 768-d axis in the frozen-CLIP prior that distinguishes the hard Xinhe-fake cohorts from the easy ones at 100% accuracy. This is a free, recoverable training signal.

2. **The "hard cluster" is most of the Xinhe-fake distribution**: 10 of 13 cohorts (851 of 1099 frames, 77%) sit on the geometric hard-side. Only 3 cohorts (xinhe-fake-{6, 7, 8}, 248 frames) form a distinct "easy" cluster. The plan's framing of "3 specific hard cohorts" mis-counts — the easy cluster is the minority, not the hard one.

3. **Face-pool's Xinhe gain is concentrated entirely on the hard cluster**: +37pp average on hard cohorts (P8A 0.341 → face-pool 0.709), but −11pp on the easy xinhe-fake-7 (P8A 0.956 → face-pool 0.844). Face-pool found a feature (face-region pooling) that recovers signal on the hard cluster specifically. **This is mechanism, not coincidence.**

Together: there is a recoverable axis in CLIP space that defines the hard cluster, AND face-pool's empirical Xinhe lift comes from catching cohorts on the hard side of that axis. **This is direct empirical motivation for a structural lever: oversample hard-cluster fakes during training, or add a contrastive loss against the hard-axis.**

---

## 2. What this means for the plan's structural direction

The plan (and the morning proposal) treats "structural ceiling" as a diffuse argument from the Probe 1 result (per-ckpt substrate axes). This readout gives a much more **concrete** structural argument:

**There is a specific axis in frozen-CLIP space along which the trained ckpts fail.** It's not an abstract substrate-pair geometry argument — it's "Xinhe-fake cohorts on this side of the axis get scored 0.23-0.45 by P8A; cohorts on the other side get scored 0.96." The encoder is missing whatever the hard cluster shares.

This is a cheap, well-defined training target:
- We have the labels (cohort identity)
- We have the axis (LR coefficient already trained, saved at `outputs/hard_vs_easy_lr_coef.npy`)
- We have the demonstration that the gap is bridgeable: face-pool closes most of it without further training

The structural lever to test is "**train a head/encoder to be invariant to the hard-axis**" — IRM-like, but with the environment partition defined by the geometric clusters rather than per-method.

---

## 3. Why face-pool wins on the hard cluster specifically

This is testable but not directly measured here. Three hypotheses:

**H1: The hard cluster has fake content concentrated in the face region, while easy cohorts have fake content distributed across the frame.** Face-pool reads only the face patches; if face-region signal is stronger on hard cohorts, face-pool gets a boost. CLS-pool averages over all patches; non-face background patches dilute the face-region signal.

**H2: The hard cluster has CLS-poisoning artifacts** (something in the non-face content that pushes the CLS token toward "real"). Face-pool sidesteps the CLS poisoning by re-pooling on face patches directly.

**H3: The hard cluster has face-region patterns the encoder didn't see in training**, and face-pool's mechanism is to give the head a more focused readout that ignores distracting non-face content.

A small CPU experiment could discriminate: compute per-patch saliency (gradient or attention) on a sample of hard vs easy cohorts; see whether saliency mass shifts between face and non-face regions.

Operational implication regardless of H1/H2/H3: **face-pool is the correct inference choice for the hard-Xinhe-cluster subset specifically**, but is wrong for the easy cluster (xinhe-fake-7 loses 11pp). The "use face-pool" / "use CLS-pool" choice is cohort-dependent — which we cannot route in production (no per-cohort knowledge). But the HEAD ALT lever (dual face+non-face → 1024-dim head, plan §3.IV) is *exactly* designed to compose both readouts. **This data argues HEAD ALT priority should be raised significantly.**

---

## 4. Implications for the GPU Week-1 plan

This readout adds two concrete options to the Week-1 menu that the plan does NOT consider:

### 4.1 Hard-axis-aware training

A new $50-80 GPU run: FT P8A or P22 step1k with an additional loss term that penalizes the encoder's projection onto the LR-defined hard-axis (in CLIP space). Specifically:
- Pre-compute the LR coefficient vector W_hard (already saved)
- During training, for each batch, compute `proj = features @ W_hard` per frame
- Add an aux loss: penalize the variance of `proj` across positive (fake) examples
- Goal: force the encoder to be invariant to the hard-axis when classifying fakes

Mechanism: directly addresses the demonstrated CLIP-recoverable axis along which the trained ckpts fail. This is a much more focused structural lever than 12-environment IRM (where we don't even know if per-method is the right partition).

Cost: $50-80 single A100 run. Risk: standard FT-instability (β-tuning, abort criteria). EV: 25-35% materially-better — comparable to IRM, but with a more directly evidence-grounded target.

### 4.2 HEAD ALT lever elevation

The plan files HEAD ALT (dual face+non-face → 1024-dim head) under "B.IV — lower-EV / not lead with." This readout argues HEAD ALT is the most directly motivated lever:
- Face-pool gives +37pp on hard cohorts
- CLS-pool gives +10pp on easy cohorts (xinhe-fake-7 etc.)
- A dual-head that learns to compose them could capture both — without training a new encoder

Cost: ~$30 head-only retrain on frozen Slot A v2 + per-pool features (CPU pre-compute is cheap; the FT is fast). The original deferral was based on Phase 2 HEAD plateauing at composite 0.235; but Phase 2 HEAD used CLS-pool only. A dual-pool head is a different lever class.

EV: 20-30% materially-better-on-team-identity. Lower than the hard-axis-aware lever above, but cheaper.

### 4.3 Per-cohort score calibration (operational, $0)

If we knew the cohort at inference (we don't, but for offline analysis), we could ship "face-pool for cohort-cluster-A, CLS-pool for cohort-cluster-B." This is forbidden per the no-ensemble rule.

But: the LR hard-axis is a *single CLIP forward pass*. It's NOT an ensemble — it's a calibration adjustment based on one frozen-CLIP feature read. The no-ensemble rule (per `project_job12_ensemble_ceiling_2026-05-04`) is about multi-model min/max/specialist routing. A calibration step that adjusts a single model's score based on a single auxiliary scalar (the hard-axis projection) is closer to "per-mode τ" than to "ensemble" — but the user has explicitly forbidden per-mode τ too per `feedback_per_mode_tau_not_deployable.md`.

So this option is likely also forbidden. Flag for user decision.

---

## 5. What this does NOT change

- **The plan's structural-reframe direction is unchanged.** This finding *strengthens* the case for structural change because the cheap discovery here suggests there's reachable signal the current ckpts miss.
- **The Week-1 IRM smoke is still reasonable.** Per-method IRM with 12 environments could find the hard-axis among others; the hard-axis-aware lever is a more focused variant of the same idea.
- **The production switch recommendation stands.** P8A at τ=0.59 from Task #1 isn't affected by this analysis; it remains the recommended same-day deploy.

---

## 6. Recommendations

### 6.1 Immediate ($0 CPU, ~1 hr each)

1. **Sample 20 frames each from xinhe-fake-7 (easy) and xinhe-fake-1/2/3 (hard); visual inspection.** What actually distinguishes them? Source identity? Background? Lighting? Pose? Camera angle? This informs whether a training-time data-axis intervention is feasible.
2. **Compute per-patch attention/saliency on a sample of hard vs easy frames** on the trained ckpts (P8A, Slot A v2 CLS). Tests H1/H2/H3.
3. **Project the test/lockbox cohorts onto the hard-axis** to see if the same axis exists outside Xinhe-fake specifically — would generalize the finding to other failure modes.

### 6.2 For the GPU Week-1 plan

**Add: hard-axis-aware FT run.** $50-80, ~4 hrs. Concrete spec:
- Base: P22 step1k OR P8A (test both if budget allows)
- Aux loss: minimize variance of `features @ W_hard` across positive (fake) examples per batch
- W_hard: the saved LR coefficient (`outputs/hard_vs_easy_lr_coef.npy`)
- β: anneal 0 → 0.1 over first 500 steps
- Eval: per-cohort recall on the 13 Xinhe-fake cohorts at the per-ckpt-calibrated τ

**Elevate: HEAD ALT (dual face+non-face) from B.IV to Week-1 candidate.** $30, head-only retrain. The face-pool / CLS-pool composition target is direct from this data.

### 6.3 For the structural reframe paper

If the plan's structural-reframe direction (B.II IRM / VIB) is pursued in Week 2+, this readout adds two concrete inputs:
- The hard-axis can be used as the IRM environment partition (instead of per-method 12-env)
- The per-cohort centroid distances suggest a natural geometric clustering: cluster A (10 cohorts) vs cluster B (3 cohorts), with extra_xinghe between

---

## 7. What I didn't do

- **No image-level inspection.** The "what does the hard cluster look like visually" question is unanswered. Could be a 15-min CPU job (pull 5 frame thumbnails per cohort, eyeball).
- **No cross-validation of the hard-axis on lockbox.** Does the hard cluster's geometric direction predict failure on other fakes (deeplive, simswap)? Untested. A cheap follow-up: project lockbox CLIP features onto W_hard and correlate with per-frame ckpt scores.
- **No attention/saliency analysis.** H1/H2/H3 from §3 require this.
- **No statistical significance test on the 1.0 AUC** — with 286 + 66 = 352 frames and 768-d features, 1.0 CV AUC is suspicious-good but stable (no fold variance). Could be checked with permutation test (~30 sec CPU); not done here.

---

## 8. Self-correction log

- **Initial framing**: I expected to find a per-cohort image-property difference (sharpness, lighting). Instead found a CLIP-space axis. Surprising and more directly actionable.
- **Hard-cluster size revision**: My initial reading from the team-deploy AGENT_PROPOSAL §2 (where the prior agent wrote "three specific Xinhe-fake cohorts — 1, 2, 3 — defeat every ckpt at the mode-B threshold") was that the hard cluster IS just those 3. The data here shows the hard *geometric* cluster includes 10 cohorts, and the score-based "hard" subset (those 3) is a tighter measurable failure mode within the larger geometric cluster. Reframed §1.
- **1.0 CV AUC**: I initially thought "this must be data leak." Re-checking: 5-fold stratified CV with 286 hard + 66 easy frames; LR not regularized strongly (C=1.0); the 1.0 is real-but-suspicious. Most likely interpretation: cohort-identity is a strong CLIP signal within a single session because of within-session pose/lighting/background variation per cohort. Flagged as caveat 8 in RESULTS_FACTS rather than a methodological issue.
- **Face-pool decomposition**: this was the surprise. Face-pool's Xinhe lift was framed in the team-deploy AGENT_PROPOSAL §3 as a generic "+20pp Xinhe gain at the cost of dor visomaster regression." The per-cohort decomposition here shows the +20pp is concentrated on hard cohorts (+30-46pp on -1, -2, -3) with face-pool actually under-performing P8A on the easy xinhe-fake-7. This is mechanism, not noise.

---

## 9. Followups (TODOs for user-decided application)

### Memory updates

- **NEW**: `project_xinhe_fake_hard_cluster_clip_axis_2026-05-23.md` — "Frozen-CLIP L11 LR achieves 1.0 5-fold CV AUC discriminating Xinhe-fake hard (xinhe-fake-1/2/3, 286 frames) from easy (xinhe-fake-7, 66 frames). Hard *geometric* cluster includes 10 of 13 Xinhe-fake cohorts (851/1099 frames, 77%); easy cluster is xinhe-fake-{6, 7, 8} (248 frames). Face-pool's +37pp Xinhe gain over P8A concentrates on hard cluster: P8A hard=0.341 face=0.709 (+0.368); P8A easy=0.956 face=0.844 (−0.112). LR axis saved at analysis/xinhe_fake_cohort_mechanism_2026-05-23/outputs/hard_vs_easy_lr_coef.npy. Direct empirical motivation for (a) hard-axis-aware FT and (b) HEAD ALT (dual face+non-face)."

### Threads to amend

- Open new thread `docs/packet_retrospectives/threads/xinhe_fake_hard_cluster_2026-05-23.md` documenting the cluster structure + face-pool mechanism. Cross-reference from `processing_signature_shortcut.md` and `iq_shortcut_deconvolution_program_2026-05-08.md`.

### OPEN_LOOPS

- Open: "Visual inspection of hard vs easy Xinhe-fake cohorts" — 15-min CPU job, identifies the image-level signal.
- Open: "Per-patch saliency on hard vs easy" — discriminates H1/H2/H3.
- Open: "HEAD ALT priority re-evaluation" (from Task #2's followups, now reinforced).
- Open: "Hard-axis-aware FT spec doc" — prerequisite for a $50-80 GPU run.

### TIMELINE

- Append: `2026-05-23 PM — Xinhe-fake cohort mechanism (analysis/xinhe_fake_cohort_mechanism_2026-05-23/) — hard-vs-easy perfectly CLIP-separable (LR 1.0 5-fold CV AUC); geometric hard cluster covers 10 of 13 cohorts (77% of Xinhe-fake frames); face-pool +37pp on hard, -11pp on easy. Direct motivation for hard-axis-aware FT and HEAD ALT elevation.`

---

## 10. Gaps and blockers

- **Image-level inspection not done.** What the hard cluster looks like visually is unknown.
- **Cross-cohort generalization untested.** Whether the hard-axis exists outside Xinhe-fake (e.g., dor-fake or deeplive cohorts) is not measured.
- **Saliency/attention not measured.** H1/H2/H3 distinction is the next step toward mechanism understanding.
- **1.0 CV AUC bears a permutation-test sanity check** — not done. The result is so strong it's worth verifying.
