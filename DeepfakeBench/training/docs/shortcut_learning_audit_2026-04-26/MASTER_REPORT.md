# Shortcut Learning Audit — Master Report

**Date written:** 2026-04-26
**Branch:** `teams-relaunch-root-2026-04-17`
**Reading time:** 25–40 min for the master, 4–6 hours including sub-reports
**Audience:** A fresh investigative agent with no prior context on this codebase

---

## 0. Read This First

### 0.1 Who you are

You are an investigative agent. You did not run any of the past experiments. You did not write any of the existing code. You are reading this report to **understand a methodology problem**, not to execute a plan.

### 0.2 Your job

The team has spent ~3 months training a deepfake detector for Microsoft Teams deployment. The model achieves AUC ≥ 0.99 and EER ≤ 0.02 on its training-time validation. It fails on real deployment-relevant probes — same identity on a different camera flips the prediction; "enhanced" (cleaned-up) fakes get classified as real; specific capture sessions miss 30% of fakes when sibling sessions miss <2%.

The team has tried to FT their way out of this for ~10 days. Every attempt has failed or shifted the failure mode without removing it.

The user (Roee) suspects the methodology is wrong: that we keep fine-tuning from a checkpoint that already has the shortcut baked in, that our metrics are insufficient to detect the shortcut, and that we cannot prove a clean baseline exists or is reachable from the current state.

**Your job is NOT to propose solutions.** Your job is to:

1. Read this master report and the six sub-reports below.
2. Form your own independent view of where the actual problem is.
3. Identify what you would investigate next, given a clean slate.
4. Tell us which past conclusions you find dubious and why.
5. Tell us what you think we have implicitly looked over.

The user explicitly does not want you to design new training runs at this stage. He wants a clear-eyed independent read on the state of the problem.

### 0.3 What this report is and isn't

This report **is**:
- An entry point to a structured audit of the shortcut-learning problem.
- An index to six sub-reports that exhaustively document specific aspects.
- A skeptical re-examination of past strategic conclusions.
- A list of things that may have been implicitly looked over.

This report **is not**:
- A plan.
- A decision document.
- A celebration of progress.
- An attempt to be balanced. The framing is deliberately skeptical because the user asked for that.

### 0.4 Recommended reading order

1. Sections 1–3 of this master (10 min) — get the problem and the index.
2. Sub-report `04_shortcut_evidence_inventory.md` (45 min) — the empirical foundation.
3. Sub-report `01_experiment_lineage.md` (30 min) — the chain of FT runs.
4. Sub-report `02_data_sources_and_bad_data.md` (30 min) — the data picture.
5. Sub-report `03_augmentation_architecture.md` (20 min) — the per-class augmentation asymmetry.
6. Sub-report `05_measurement_apparatus.md` (30 min) — what we measure, what we don't.
7. Sub-report `06_strategic_doc_trail.md` (30 min) — history of strategic moves.
8. Sections 4–10 of this master (45 min) — skeptical synthesis.

If you only have 90 minutes, read sections 1–7 of this master and skim sub-report 04.

---

## 1. Problem Statement

### 1.1 What we're trying to build

A deepfake detector for Microsoft Teams deployment. Architecture: OpenCLIP ViT-B/16-DataComp-XL backbone (frozen by default) + ArcFace head (trainable) + low-rank SVD residuals on attention `in_proj` (rank 736, trainable). Optionally: MLP-SVD residuals and unfrozen final projection / LayerNorm.

The model is loaded from a long FT chain rooted in the LAION DataComp-XL pretraining of OpenCLIP. Most R13 packets (all of P7, P8, P9, P10 except a few forks) FT from `RLP6_04 step 23500` — a checkpoint that has itself been fine-tuned for ~23,500 steps from R12g, which had been fine-tuned from earlier rounds. See sub-report 01 for the complete lineage.

### 1.2 The success criteria (three pillars)

From memory `project_success_criteria.md`, all three are load-bearing:

1. **Fake recall on target methods** at chosen operating threshold τ. The deployment-relevant fake methods are realtime swap tools (deeplive variants) and offline tools post-processed to look like Teams traffic (visomaster_enhanced).
2. **Real-pool FPR < 5%** on Teams-camera-style data (5% target, 7% hard cap).
3. **Robustness** across lighting, camera, codec, and color-pipeline variation typical of consumer hardware and Teams compression.

The team has been celebrating wins on (1) and (2) while pillar (3) silently fails. The shortcut-learning problem is a pillar (3) failure dressed up as pillar (1)+(2) success.

### 1.3 Why "AUC ≥ 0.99, EER ≤ 0.02" is misleading

Two reasons:

**Reason A — the validation set is in-distribution-by-source.** Trainer-time AUC/EER is computed on a bisection of `combined_paired` data. That bisection includes `df40_real`, `external_youtube_avspeech_real`, `zoom_vcd_real`, `proper_clean_real`, `proper_teams_real`, and `teams_ood_real`. The deployment target — Teams-passthrough with consumer cameras and codecs — is NOT well represented in this bisection. AUC ≥ 0.99 on this mix is consistent with the model using source-of-origin as a feature and being right because the source-label correlation is strong in the validation data. See sub-report 05 §5.

**Reason B — the in-trainer `value_composite` and the deployment-side `promotion contract scorecard` calibrate τ on different distributions.** The trainer's value_composite uses heuristic FPR thresholds; the contract calibrates τ lexicographically on dev and reads out lockbox at that τ. They typically pick different τ values. A run can score `value_composite ≈ 0.99` and still fail the contract — exactly what happens with P8A.

This means **the headline numbers ("AUC 0.9942, EER 0.0169" on RLP6_04) are consistent with a model that has perfectly learned a shortcut that generalizes to the validation set but not to the deployment target.**

### 1.4 The specific failure modes (concrete observations)

From sub-report 04 and prior packet handoffs:

- **Anchor pool spread (camera-signature shortcut):** Same person Dor on a laptop camera scores 0.02 (clean real), on his Logitech webcam scores 0.94 (false-flag fake). Cross-subject (Roee on Mac vs Windows) replicates the pattern. Six anchor pools span this axis. RLP6_04 anchor mean: 0.932. P8A: 0.744 (best). P10 (latest): 0.94–0.96 (no improvement).
- **The `s33` outlier (slot-07 / camera-signature shortcut):** In `lockbox_fake`, `Cam_Test__s33` (a single capture session) misses 30.4% of fakes. Sibling sessions `s32, s35, s38, s46` from the same swap family all miss < 2%. Only s33 has a capture-session quality signature distinct from the others.
- **dor_shkedi vs real_dor processing-signature shortcut:** Same person, two label attributions: `dor_shkedi` (training-side) and `real_dor` (lockbox-side). Model gives substantially different outputs. Memory `project_signature_shortcut_finding.md` records this as "90/90 on lockbox is NOT threshold-reachable" — i.e., separability collapse.
- **Per-method recall collapse on enhanced fakes:** Under RLP6_04 and worse under P8A:
  - `deeplive_enhanced`: RLP6_04 79.6% → P8A 53.0% (−26.6 pp)
  - `visomaster_enhanced_macro`: RLP6_04 57.3% → P8A 35.6% (−21.6 pp)
  - `teams_flat_xiang_xiang2_feng`: RLP6_04 78.5% → P8A 45.2% (−33.3 pp)
- **The trade is one-sided.** Per sub-report 04: 329 videos where RLP6_04 caught the fake and P8A missed; **zero compensating wins**.

These are not distinct problems. They are the same problem expressed in different probes.

### 1.5 The user's mental-model question (the question that drove this audit)

> "How are we proving, how are we making sure that when we train the model we can confidently say there are no shortcut learnings and no overfitting to specific irrelevant patterns, so that we have a baseline for training the model?"

**The honest answer is: we are not proving it. We do not have an apparatus that can give us such a guarantee.** We have one probe (the anchor pool) used manually and at the end of training. We do not have linear probes for camera/identity, cross-camera holdouts, source-classifiers on frozen embeddings, or training-time monitoring of any of these.

This is the methodology problem the user is hitting. The receiving agent should take this question as the central question.

---

## 2. The Two-Sentence Summary

The team has been training a model whose own training pipeline installs the shortcut it then tries to remove via FT, on top of a checkpoint chain that has already baked in shortcut features through ~37,500 cumulative FT steps from CLIP. We have no probe battery that can certify a checkpoint as shortcut-free, so even successful experiments cannot establish that they actually solved anything.

---

## 3. The Six Sub-Reports

| # | Title | Author | Key takeaway |
|---|-------|--------|--------------|
| 01 | [Experiment Lineage & Checkpoint History](sub_reports/01_experiment_lineage.md) | Lineage agent | RLP6_04 step 23500 = 37,500 cumulative FT steps from CLIP. Most P10 runs are 47,500 cumulative steps. There has been NO clean-CLIP retrain with the modern data mix in 47+ days. |
| 02 | [Data Sources, Manifests, and Bad-Data Policies](sub_reports/02_data_sources_and_bad_data.md) | Data agent | Source-label correlation is severe (entire buckets are one-class). Identity split is correct within realpool but uncoordinated with the lockbox identity_key namespace. Family weights span 33× (deeplive_teams_fake=5.0, df40_fake=0.15). |
| 03 | [Augmentation Pipeline Architecture](sub_reports/03_augmentation_architecture.md) | Pipeline agent | The "family-aware" router applies 2.8× more degradation to fakes, JPEG floor 34 points lower, sharpen probability 10× lower than reals. The new symmetric branch (commit 2c9778b, 2026-04-26) removes the within-pipeline asymmetry but does NOT collapse the Teams-vs-non-Teams source split. |
| 04 | [Shortcut Evidence Inventory](sub_reports/04_shortcut_evidence_inventory.md) | Evidence agent | Cross-checkpoint anchor table: RLP6_04 mean 0.932 → P8A 0.744 (best) → P10 SYM_LIGHT 0.945 (regression). P8A's anchor breakthrough comes with −13.66 pp aggregate fake-recall regression and ZERO compensating wins. ~80% of P8A's misses are NOT τ-recoverable (separability loss). |
| 05 | [Measurement Apparatus, GRL, and Metric Gaps](sub_reports/05_measurement_apparatus.md) | Measurement agent | The GRL's 4-bucket quality-domain head is structurally orthogonal to the camera-signature shortcut. Lockbox has person-level overlap with dev and same-camera-different-session overlap. value_composite is in-distribution-by-source. Eight major probes are missing. |
| 06 | [Strategic Documentation Trail](sub_reports/06_strategic_doc_trail.md) | History agent | Three strategic-framing pivots in 9 days. RLP6_04 step 23500 was selected by `value_composite` — the metric subsequently demoted as not deployment-grade. avspeech/vcd dropping from the gate (RLP6) made `value_composite` jump 0.7736 → 0.9006 without retraining; metric-relaxation was conflated with model improvement. |

Each sub-report is self-contained and can be read independently. Cross-references are noted inline. The 6 sub-reports collectively run ~5,000 lines.

---

## 4. Critical Skeptical Framing

This section names the load-bearing assumptions the team has been operating under and challenges each.

### 4.1 The cumulative-FT-step problem (the user's question literalized)

From sub-report 01:

| Checkpoint | FT steps from CLIP base | Comment |
|---|---:|---|
| Plain CLIP (DataComp-XL) | 0 | Untouched |
| R12g `0xxqwhxg` step 14000 | 14,000 | Most recent clean-CLIP scratch in this codebase, ~47 days old, used pre-shortcut-diagnosis data composition |
| RLP6_04 `h2pdu6i5` step 23500 | 37,500 | Forks from R12g step 14000, FT 23,500 steps |
| P8A `9lmvb5b4` step 5000 | 42,500 | Forks from RLP6_04 step ~step-(unknown), FT 5,000 steps. Now has unfrozen final projection, ln_post, and MLP-SVD on top |
| P10_SYM step 4500 | 42,000 | Forks from RLP6_04 step 23500, FT 4,500 steps |
| P10_SYM_on_P8A (finished) | ~52,500 | Forks from P8A step 5000, ~5,000 more steps |

**Implications:**

1. **Every FT pass since R12g (47 days ago) has accumulated on top of weights that already contain whatever shortcut features R12g installed.** R12g itself was trained on a data composition that pre-dated the shortcut diagnosis, the symmetric augmentation, and the proper-data inventory. If a shortcut is in R12g's frozen-CLIP-plus-residual representation, no amount of FT-from-R12g-descendants can guarantee its removal — it can only re-weight.

2. **The frozen-backbone mechanics make this worse.** The default recipe (R12g, RLP6_04, all P7, P10 SYM/GRL/SYM_GRL/SYM_LIGHT) updates only the ArcFace head and the SVD residuals on attention `in_proj`. The CLIP backbone weights themselves are FROZEN. **A shortcut that lives in the CLIP backbone cannot be removed by FT in the frozen-backbone recipe.** It can only be partially counter-modulated by the residual.

3. **P8A unfreezes more (visual.proj, ln_post, MLP-SVD on last 2 blocks) — and is the only run that moved the anchor mean.** This is consistent with reach-limitation. It's also consistent with simply "more capacity to fit the FT data" without unique progress on the shortcut. Sub-report 04 confirms P8A traded FPR for fake recall.

4. **There is no recent clean-CLIP run that actually completed and was scored** with the modern data mix and modern measurement apparatus. P8B was attempted and hung at step 11000/30000. R12g exists but is stale.

**Skeptical question for the receiving agent:** What evidence do we actually have that the shortcut is unremoveable from R12g-descendants? The "shortcut is upstream" memory is grounded in RLP7_08 (forking from RLP6_04 step 4500 instead of 23500 hits the same anchor ceiling). But this only rules out "late RLP6_04 consolidation," not "shortcut originated in R12g or earlier." The hypothesis "the shortcut is in the CLIP backbone weights themselves" has not been tested with a clean-CLIP scratch run on modern data. P8B was supposed to test this but didn't finish.

### 4.2 The frozen-backbone problem

Per sub-report 01 §4 and the architecture context: in the default recipe, only ~9M parameters are trainable (ArcFace head + attention-SVD residual). The 86M-parameter CLIP backbone is frozen. The MLP layers in particular are entirely untouched.

**This is a "shortcut un-removability" claim disguised as a "small finetune is efficient" claim.** If the shortcut feature is encoded in the CLIP MLP weights (which is plausible — MLPs encode much of CLIP's learned content), the default recipe cannot remove it. P8A attacks part of this by adding MLP-SVD residuals, but the SVD is rank-bounded and additive — it can amplify or dampen existing directions but cannot rebuild representations from scratch.

**Skeptical question:** Why have we never run a full-fine-tune (no freeze) on a small portion of recent data to see what the model is *capable* of when allowed to relearn its features? The B16_capacity_ceiling experiments from `experiments/B16_capacity_ceiling/` (1_baseline_effort, 2_unfreeze_last_2, 3_unfreeze_last_4, 4_full_finetune) appear to have done this in earlier rounds, but on different data and probably not on the deployment-relevant probes. The receiving agent should investigate what those experiments concluded.

### 4.3 The cross-source camera asymmetry (the data shape problem)

Even if augmentation is now perfectly symmetric per-class, the underlying data sources are NOT class-symmetric:

| Source | Real samples? | Fake samples? |
|--------|---------------|---------------|
| df40 | Yes (down-weighted 0.4) | Yes (down-weighted 0.15) |
| deeplive | Yes | Yes |
| visomaster (proper_visomaster_*) | Yes | Yes |
| teams (live-deepfake-methods-...-teams-v2) | Yes | Yes |
| external_training_reals (VCD) | **Yes only** | No |
| external_youtube_avspeech | **Yes only** | No |
| real-teams-dor-roee | **Yes only** | No (anchor pool, readout-only) |

Three sources are real-only. The model can learn "image is from VCD bucket → real" with very high accuracy without using image content. This is a source-label correlation that survives any symmetric augmentation pipeline because the image-statistics signature of "VCD-style" or "AVSpeech-style" is in the pixels themselves, not in any augmentation-routed transform.

This is a different kind of shortcut from camera-signature: it's **source-signature**. The model can learn to discriminate "what corpus does this come from" and many corpuses are pure-class.

**Skeptical question:** Has anyone trained a linear probe on the frozen CLIP embedding (or RLP6_04's embedding) to predict the source-of-origin (df40 vs deeplive vs visomaster vs teams vs external-vcd vs external-avspeech)? If the probe achieves >95% accuracy, the embedding has the source as a feature, and the label-versus-source correlation can leak through any downstream classifier. **This probe has not been run.** Sub-report 05 lists it as a missing measurement.

### 4.4 The lockbox is not as held-out as we treat it

Sub-report 05 §4 documents three layers of overlap:

1. **Person-level overlap.** "Dor" appears across dev (`dor`, `dor_shkedi__s16`, `deeplive_dor`) and lockbox (`dor_shkedi`, `real_dor`). Same for `bla_bla_chow*` and `PC_Generator__s*`. Identity_key disjointness is verified at the identity_key namespace, but the underlying person can appear in both via different identity_keys (different sessions, different processing).
2. **Camera-signature overlap.** `Cam_Test__s33` is the only Cam_Test session in lockbox; sessions s32, s35, s38, s46, s53, s73, s76 are all in dev. Same camera, same room, same encoder. The lockbox is "held out" only at the session-id level, not the camera-pipeline level.
3. **Bucket overlap.** Validation pulls from `gs://teams-faces-data-test-2914-fake-4420-real-feb-28`; training pulls from `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`. Different buckets, but no automated audit confirms the underlying recordings don't overlap.

**Implication:** When we say "P8A achieves lockbox_real_fpr 0.147%," we should read this as "achieves 0.147% on data that overlaps in person and camera with training." This is not a deployment-grade hold-out.

**Skeptical question:** What would it take to construct a TRULY held-out lockbox — different people, different cameras, different rooms, different codec? Has anyone tried? The `real-teams-dor-roee` anchor pool is genuinely held out (firewalled to `readout_only_external_real_sources` with `max_videos=10`), but it has only ~30 frames, which is too small for headline metrics.

### 4.5 The contract policy bug as a measurement confound

Memory `project_contract_policy_bug.md` and sub-report 05 §3:

The promotion contract has a known bug: when no τ satisfies the budget, the fallback policy minimizes FPR with no recall floor, driving τ to ~0.995 and crushing fake recall to near-zero. The fix is conditional — if budgets `target_real_fpr=0.02 / target_stress_fpr=0.05` are active, the budget path is taken; otherwise the legacy broken path activates.

**Both P8A's and RLP6_04's reported lockbox metrics use τ ≈ 0.991-0.992.** Sub-report 06 FLAG 1 raises the possibility that P8A's "breakthrough" lockbox_real_fpr 0.147% is an artifact of this τ inflation: at τ=0.991, ANY model with a reasonable score distribution will have very few real-side false positives, but it will also have very few fake-side detections (which is exactly what we see).

**Skeptical question for the receiving agent:** Has anyone computed P8A's lockbox metrics at a range of τ values to see whether the FPR win is robust? The sub-report 04 anchor analysis shows P8A's anchor mean is genuinely lower (0.744 vs 0.932), which is τ-independent. But the lockbox-FPR win at τ=0.991 is not the same kind of evidence — it could be a τ artifact.

### 4.6 The "P8A breakthrough" reframe

Memory `project_p8a_breakthrough.md` says: "P8A unfreezes visual.proj+ln_post+MLP-SVD, cuts anchor FPR ~2× vs best P7 with no real-pool regression."

Sub-report 04 says: P8A's anchor mean is genuinely lower. AND P8A's fake recall regressed by 13.66 pp aggregate, with 329 newly-missed fakes and 0 compensating wins. ~80% of P8A's regression is separability loss, not τ-shift.

**The two are not contradictory but they are not the same story.** P8A traded fake recall for FPR. The "breakthrough" framing in memory is selective.

**Skeptical question:** Was P8A actually a breakthrough on the shortcut, or was it just a checkpoint with a different operating point on the FPR/recall trade-off? Sub-report 04 reports P8A still flips 13/30 anchor frames at threshold > 0.9. The shortcut is **weakened, not removed**. Is "weakened by half" actually an improvement, or is it just a different cut of the same fundamental representation? The receiving agent should examine the analysis at `analysis/p8a_fake_failure_analysis_2026-04-25/` (sub-report 04 §3) and form an independent view.

### 4.7 The "shortcut is upstream of RLP6_04" claim is undertested

Memory `project_shortcut_is_upstream.md`: "RLP7_08 ruled out 'late RLP6_04 consolidation'; FT-only from any RLP6_04 step hits ~0.84-0.89 anchor ceiling."

This is grounded in ONE experiment: forking from RLP6_04 step 4500 instead of step 23500. It hit the same ceiling.

**This rules out:** "the shortcut consolidates between step 4500 and step 23500 of RLP6_04."

**This does NOT rule out:**
- "The shortcut is in R12g (RLP6_04's parent)." Untested by RLP7_08 — RLP7_08 still uses an RLP6_04 step as base, not a non-RLP6_04 base.
- "The shortcut is in the CLIP backbone weights." Same — RLP7_08 doesn't use a non-CLIP-DataComp-XL base.
- "The shortcut is in the data composition rather than the model state." RLP7_08 used the same data mix as RLP6_04.

P8B was the ONLY recent experiment that tested "fresh head on plain CLIP." It hung at step 11000/30000 and was cancelled before completing. Its partial results showed it was worse on the anchor than RLP6_04 — consistent with "the data installs the shortcut on a clean backbone too" — but with a partial run that's a weak claim.

**Skeptical question:** What would it take to settle this? At minimum: (a) finish a clean-CLIP retrain on modern data (the failed P8B attempt), (b) probe the frozen CLIP embedding for camera/source classification accuracy, (c) probe RLP6_04's embedding for the same. If clean CLIP has the source/camera classification at high accuracy, the shortcut substrate is in CLIP itself. If RLP6_04 has it dramatically higher, FT has amplified it.

### 4.8 The "hints are bad data" thread

The user mentioned this in passing. The full picture from sub-report 02:

**`visomaster_hints` is currently disabled in all P10 configs.** It was a dataset of 480-682 samples flagged as "under-swapped" via a `face_parser_enabled=True` bug in the original VisoMaster pipeline. The April-17 bad-data policy excluded 4904 of 5589 legacy under-swapped samples and kept 480 as `visomaster_hints` and 202 as `visomaster_hints_teams`. Both lanes were eventually disabled in current configs.

The April-17 "proper data" inventory (sub-report 02 §7) was created to replace the visomaster legacy workstream entirely. Commits `77facfc` and `d07f06d` introduced the proper-data manifest with 1826 captures × 4 lanes = 7304 videos.

**The fact that we had to flag 4904 samples as bad and exclude them strongly suggests the overall data quality of pre-relaunch training data was much lower than we now assume.** The model trained through R12g and earlier used data that we have since classified as bad. R12g's representation was shaped in part by data we no longer trust. Every R13 FT pass inherits that representation.

**Skeptical question:** Has anyone audited what fraction of the data the model was trained on (pre-R13) is now considered bad? If 30% of the original training data is now considered bad, the model's "shortcut features" might literally be features it learned to discriminate the bad samples from the good ones — features which then misalign with the cleaner deployment data.

---

## 5. Taxonomy of Shortcuts (and what we know about each)

The team uses "shortcut" loosely. Untangling the kinds is critical for the receiving agent:

### 5.1 Quality shortcut

**Definition:** "Lower image quality / heavier processing → fake."

**Mechanism:** Asymmetric augmentation. Reals get gentle JPEG/downscale at low probability; fakes get heavy JPEG/downscale at high probability. After training, the model maps "JPEG-degraded image" to "fake-class probability."

**Status:** Confirmed by data audit (sub-report 03). The current default `family_aware` router applies fakes 2.8× more outer degradation, JPEG floor 34 points lower, sharpen 10× less likely.

**Test bed:** Visomaster_enhanced (cleaned fakes that look like training reals) — model classifies them as real and recall collapses.

**Attempted fix:** Symmetric routing in P10. Sub-report 04 shows P10_SYM, P10_GRL, P10_SYM_LIGHT all received "no_meaningful_change" verdict on anchor pools. Sub-report 04 also notes per-method recall verdict pending.

### 5.2 Camera-signature / ISP-signature shortcut

**Definition:** "Image came from a specific camera/codec pipeline → fake."

**Mechanism:** Real samples in training come predominantly from certain camera signatures (laptop, mobile, AVSpeech). Fake samples (especially deeplive_teams) come from a different cluster of camera signatures (specific webcam ISPs, Teams compression). The model learns the camera signature and the label is correlated.

**Status:** Confirmed by sub-report 04 anchor analysis. RLP6_04 anchor mean 0.932 (mostly false-flagging Dor's webcam as fake despite being real). P8A 0.744 (improved but not solved). P10 attempts 0.94–0.96 (essentially unchanged).

**Test bed:** The 6-pool anchor:
- DOR-LAPTOP-WHITE — real, laptop camera, white background, clean lighting
- DOR-LAPTOP-YELLOW — real, laptop camera, yellow background (control for "color")
- DOR-WEBCAM-FALSE-FLAG — real, Dor's Logitech webcam, clean lighting (the canonical test)
- DOR-WEBCAM-NO-VBG — real, Dor's Logitech webcam, no virtual background
- ROEE-MAC-VBG — real, Mac webcam, with virtual background
- ROEE-LAPTOP-WIN — real, Windows laptop camera

**Attempted fix:** P8A unfreeze (partial improvement). P10 (no improvement). Neither addresses the underlying issue: the GRL is wired to a quality-binned domain head, not a camera-signature head. Sub-report 05 §1 says: "the GRL as currently wired CANNOT penalize the camera-signature shortcut."

### 5.3 Codec / processing-signature shortcut

**Definition:** A specialized form of camera-signature: "Image was processed through pipeline X → fake (or → real)."

**Mechanism:** Different label-related processing pipelines leave statistical fingerprints in the pixels even after augmentation. Slot-07's `dor_shkedi` (training-side processing, fake-coded) vs `real_dor` (lockbox-side processing, real-coded) is the canonical example. Same person, two processing pipelines, different model output.

**Status:** Confirmed by memory `project_signature_shortcut_finding.md` and slot-07 retrospective documents.

**Test bed:** The s33 outlier (one capture session out of five missing 30.4% of fakes when siblings miss <2%); the dor_shkedi/real_dor split.

**Attempted fix:** Teams-passthrough augmentation (`_maybe_apply_teams_sim`) was added partly to address this. Effectiveness unclear.

### 5.4 Identity / face-recognition shortcut

**Definition:** "I recognize this specific face → it has been seen as class X in training."

**Mechanism:** ArcFace head is designed for face identity classification. Combined with identity-balanced sampling and high family weights on specific identity-rich sources, the model could memorize identities and use identity as the discriminator.

**Status:** Untested rigorously. The team ruled out "pure identity" by noting Dor scores correctly on his laptop and incorrectly on his webcam (so the identity isn't the discriminator alone). But this only rules out identity-as-only-discriminator, not identity-as-co-discriminator-with-camera.

**Test bed:** Linear probe on the embedding for identity-recall accuracy. None has been run.

**Attempted fix:** None specific to this axis.

### 5.5 Source / domain shortcut

**Definition:** "Image came from this corpus of origin → predict the corpus's predominant label."

**Mechanism:** Some corpora are 100% one class (`external_training_reals`, `external_youtube_avspeech`, `real-teams-dor-roee` are real-only). The model can learn to detect the corpus from image statistics and use that as the label.

**Status:** Untested. Probable based on data audit (sub-report 02). The bucket-of-origin signal is hard to remove because it's encoded in pixel statistics that survive the augmentation pipeline.

**Test bed:** Linear probe on embedding for source classification. None has been run.

**Attempted fix:** None.

### 5.6 File-path / metadata leak

**Definition:** "File path or EXIF/codec metadata leaks the label to the data loader."

**Mechanism:** Path conventions, codec metadata read by the loader, EXIF tags.

**Status:** Sub-report 02 §10 audits this partially. EXIF is stripped by `cv2.imdecode`, but JPEG quantization tables remain in the pixel content. Path-based label assignment is structurally separated from the image content load. Probably not a leak, but no formal audit exists.

---

## 6. Methodology Failures: What We Measured vs What Mattered

### 6.1 What we measured (and trusted)

1. Trainer-time AUC and EER on a `combined_paired` validation bisection — IN-DISTRIBUTION-BY-SOURCE.
2. Trainer-time `value_composite` — a heuristic combining mean/max FPR and per-method recall.
3. Promotion-contract scorecard on a "lockbox" that overlaps with training at person, camera, and possibly bucket level.
4. Anchor-pool spread, manually, end-of-training, on ~30 frames per pool.

### 6.2 What we should have measured

1. **Frozen-embedding probes.** Linear classifier accuracy from the (frozen) embedding to:
   - Source/corpus of origin (df40 vs deeplive vs visomaster vs teams vs external_*)
   - Camera signature (group cameras as anchor pools do)
   - Identity (face-recognition probe)
   - Codec / capture session
2. **Cross-camera train/test split.** Train holding out one camera or one capture session; test recall on it. If recall craters, the model overfits camera.
3. **Same-identity-different-camera spread monitor during training.** Run anchor pools every N steps. Alert on divergence.
4. **Calibration consistency across pools.** At one τ, equivalent real subpopulations should give similar FPR.
5. **Per-source label-balance audit.** Confirm that no source is one-class.
6. **Source-of-origin distribution audit on lockbox.** Confirm the lockbox covers the deployment source distribution rather than the training source distribution.

Sub-report 05 §8 expands on this gap.

### 6.3 What this means

We have been training a model whose own metrics rewarded shortcut learning, on data with structural source-label correlations, with measurement infrastructure that did not test the dimensions where the failures actually live. The "10 days of FT trying to fix it" period is the natural consequence: each FT attempt was scored on metrics that did not directly reflect the failure mode.

---

## 7. Things That May Have Been Implicitly Looked Over

The following are observations or hypotheses that appear under-investigated relative to their potential importance. The receiving agent should sanity-check each.

### 7.1 The "47 days since last clean-CLIP scratch" gap

Sub-report 01 finds R12g `0xxqwhxg` is the most recent clean-CLIP scratch run, ~47 days old, on pre-relaunch data. **No clean-CLIP-with-modern-data run has completed.** P8B attempted this and hung. The "shortcut is upstream of RLP6_04" claim is grounded in one experiment (RLP7_08) that doesn't actually fork from a non-RLP6_04 base. The CLIP-base-with-modern-data hypothesis is **not actually tested.**

### 7.2 The R12g↔W&B yaml discrepancy

Sub-report 01 flags: registry says `0xxqwhxg` is seed 737, but repo yaml `R12_G_scratch_seed_control.yaml` declares seed 1337. The actual yaml that launched `0xxqwhxg` is uncertain. **This is the load-bearing parent of every RLP1-RLP6 chain.** If the yaml is uncertain, the data composition R12g actually trained on is uncertain. Worth pinning down.

### 7.3 Source-of-origin classifiability

No probe has tested whether the embedding can identify the source corpus. If the answer is yes (very likely, given how distinct the corpora are), the source is a feature the model can use. The lockbox may include sources that aren't in the deployment distribution; the deployment distribution may not be well-covered by any source. **This is a research-grade question that has not been asked.**

### 7.4 The `proper_visomaster_*` smoothing artifact

Sub-report 04 flags: the proper_visomaster non-enhanced lanes look post-smoothed (Laplacian variance 51-56) — much lower than non-smoothed Teams real (>120). The model could be learning "smooth → fake" from these lanes specifically. This has not been formally tested.

### 7.5 The `visomaster_enhanced_teams_fake` family weight

Family weight is 1.0, same as `proper_visomaster_clean_fake`. Sub-report 02 flags this as suspicious — `_enhanced_teams` should be the deployment-most-relevant fake family, but it's weighted equal to the clean one. Why?

### 7.6 The relaunch was a data-policy reset, not a model reset

Sub-report 06 emphasizes: "main has not advanced since 2026-04-08. The relaunch was a data-policy and evaluation-machinery reset, not a model reset." Every R13 packet still forks from R12g (RLP1-RLP3) or RLP6_04 (RLP7+). **The "relaunch" framing implies a fresh start that didn't happen at the model level.**

### 7.7 The avspeech/vcd metric jump

Sub-report 06 FLAG 21: dropping avspeech/vcd from the value_composite gate (RLP6) jumped value_composite 0.7736 → 0.9006 without retraining. **This is a metric-relaxation-presented-as-improvement.** A receiving agent should investigate whether the strong headline numbers we now report on RLP6_04 derive from this gate change rather than actual model improvement.

### 7.8 The `selected_threshold` audit on past scorecards

Memory `project_contract_policy_bug.md` and sub-report 05 §3: when τ ≥ 0.99 in scorecard outputs, the contract-policy bug may be active. **It's unclear whether all past scorecards have been re-checked against this filter.** Some "wins" reported in handoff docs may be τ-inflated artifacts.

### 7.9 The bucket-overlap audit on lockbox

Sub-report 05 §4: validation pulls from `gs://teams-faces-data-test-2914-fake-4420-real-feb-28`; training pulls from `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`. **No automated audit confirms the underlying recordings don't overlap.** "The lockbox is held out" is a procedural claim, not a verified one.

### 7.10 The `Cam_Test__s33` placement

Sub-report 05 §4: `Cam_Test__s33` is the only Cam_Test session in lockbox; sessions s32, s35, s38, s46, s53, s73, s76 are all in dev. **The same camera is in both partitions.** The s33 outlier (30.4% miss rate) might be detectable by a "this is a Cam_Test camera" feature, with s33 specifically falling on the wrong side of the decision boundary because of one capture-session-specific quality signature. The sibling sessions don't fail because they are in dev (used to calibrate τ).

### 7.11 The pre-relaunch bad-data fraction

Section 4.8 above. R12g and earlier were trained partly on data we now flag as bad. The shortcut features may be learned from discriminating bad-data samples from good ones — features that misalign with deployment.

### 7.12 The frozen-backbone capacity ceiling

Section 4.2 above. The B16_capacity_ceiling experiments may have shown what the model is *capable* of when allowed to fully fine-tune. If those results showed full-FT solving a problem the frozen recipe couldn't, the frozen-recipe choice is a deliberate accuracy-vs-stability trade-off — and the team has been operating under "frozen is correct" without re-validating that under modern data.

### 7.13 The augmentation-symmetry-but-source-asymmetry case

Section 4.3 above. Symmetric routing makes within-pipeline transforms label-symmetric, but the sources themselves are class-asymmetric. The model can learn source via source-specific image statistics that survive augmentation (codec lineage, JPEG quantization tables, lighting profiles). **Within-pipeline symmetry is necessary but not sufficient.**

### 7.14 The `apply_svd_to_mlp` scope

Sub-report 01 marks UNCERTAIN: whether MLP-SVD applies to all 12 transformer blocks or only the last N. P8A's "topology" hypothesis (P9_03 testing `apply_svd_to_mlp: false`) only makes sense if we know what scope was active. The receiving agent should pin down the actual scope in the code.

### 7.15 The "P8A breakthrough" replication is N=1

Memory describes P8A as a breakthrough. The P9 slate included `P9_R` (P8A reseed) precisely because the P8A result is N=1. **As of this report, P9 results are pending or partial.** The breakthrough claim has not been replicated.

---

## 8. Open Questions for the Receiving Agent

These are the questions the user wants you to think hard about, in priority order.

### 8.1 Where does the shortcut actually live?

Hypotheses to disentangle:
- (a) In the CLIP DataComp-XL backbone weights themselves (untested directly).
- (b) In R12g's accumulated FT — i.e., shortcut features learned during R12g training (untested directly; only R6 → RLP7 chain has been probed).
- (c) In RLP6_04's accumulated FT — i.e., between R12g and RLP6_04 step 4500 (untested; RLP7_08 only rules out late RLP6_04 consolidation).
- (d) In the data composition itself, regardless of base — i.e., any model trained on this data mix would learn the same shortcut (P8B's partial result is consistent with this but inconclusive).
- (e) Some combination.

What experiments would settle it? What probes (linear classifiers on frozen embeddings) would rank the hypotheses without retraining?

### 8.2 What does a clean baseline look like?

If we wanted to certify a model as "shortcut-free," what battery of probes would we need to pass? At minimum:
- Linear probe accuracy on the embedding for [source, camera, identity, codec]. What thresholds count as "shortcut-free"?
- Anchor-pool spread under what range of τ?
- Cross-camera holdout recall — what camera, what fraction of the test set, what threshold?
- Same-source label-balance — every source has both classes in some minimum proportion?

If we don't know the answer, **what's the path to defining it?**

### 8.3 Is the FT-from-RLP6_04 paradigm fundamentally broken?

If shortcut features are baked into the FT chain, no FT-from-FT-from-FT approach can fix it. Should the next move be a clean-CLIP retrain on modern data with anti-shortcut mechanisms baked in from step 0? P8B attempted this and hung — what would it take to actually finish such a run, and what should it look like?

### 8.4 Are our "wins" actually wins?

Many past handoff docs declare progress (RLP6_04 leader, P8A breakthrough). Section 4 above questions some of these. The receiving agent should form an independent view: which past wins are **robustly true** under careful re-examination, and which are **measurement artifacts**?

### 8.5 What's the minimum probe battery?

The user needs to be able to certify a model as deployment-ready. We currently can't because our probes don't cover the failure surface. Section 6.2 lists candidate probes. What would a minimum viable probe battery look like — one that's cheap enough to run regularly and broad enough to detect the major shortcut classes?

### 8.6 Should the existing data even be trusted?

Section 7.11. R12g was trained on data we now consider partially bad. The "proper data" inventory is from April 19 — recent. RLP6_04 was trained from R12g, which was trained on legacy data. **The clean-data RLP6_04 retrain has not happened.** Is the path forward to retrain RLP6_04-equivalent on clean data only, then continue from there?

### 8.7 The mental-model question literalized

> "How are we proving... that when we train the model we can confidently say there are no shortcut learnings and no overfitting to specific irrelevant patterns?"

The user wants a methodology. **Not solutions to fixing this model. A methodology for how to build a CLEAN MODEL FROM SCRATCH that we can certify is shortcut-free at each milestone.** What does such a methodology look like? What is the minimum infrastructure to support it?

---

## 9. Reading Guide & Repository Map

### 9.1 Critical files to know

```
DeepfakeBench/training/
├── data/
│   ├── augmentations/pipelines.py         — sub-report 03
│   ├── sources/combined_paired.py         — sub-report 02 (data assembly)
│   ├── sources/*.py                       — sub-report 02 (per-source loaders)
│   └── validation_sources.py
├── detectors/
│   └── effort_detector.py                 — sub-report 05 §1 (GRL)
├── arena/
│   ├── score_teams_promotion_contract.py  — sub-report 05 §2
│   ├── run_target_domain_validation_sequential.py
│   ├── target_domain_suites.*.yaml        — sub-report 05 §3
│   ├── checkpoint_maps/*.yaml             — score session checkpoint groups
│   └── reports/*.json                     — scorecard outputs
├── experiments/
│   ├── phase2/                            — early R-rounds
│   ├── phase2_round2/ ... phase2_round13/ — sub-report 01
│   └── B16_*/                             — capacity / rank sweeps
├── policy/
│   └── visomaster_bad_data/*              — sub-report 02 §2
├── analysis/                              — sub-report 04 (ad-hoc probes)
├── docs/
│   ├── relaunch_handoffs/*.md             — sub-report 06
│   └── shortcut_learning_audit_2026-04-26/   <- you are here
└── memory/                                — at /Users/roeedar/.claude/projects/.../memory/
```

### 9.2 Key memory files to read

In `/Users/roeedar/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/`:

- `project_promotion_contract.md` — what the contract scorecard is
- `project_contract_policy_bug.md` — the τ inflation bug
- `project_signature_shortcut_finding.md` — slot-07 / dor_shkedi
- `project_shortcut_is_upstream.md` — the load-bearing claim about shortcut location
- `project_p8a_breakthrough.md` — the load-bearing claim about P8A
- `project_success_criteria.md` — three pillars
- `feedback_no_cancelling_vertex_jobs.md` — operational
- `MEMORY.md` — index

Read these critically. Several represent claims that section 4 of this master argues are undertested.

### 9.3 Recent doc trail (chronological)

From sub-report 06:
- 2026-04-17: Relaunch begins. Branch `teams-relaunch-root-2026-04-17` is cut.
- 2026-04-17: Multiple WT-* worktree handoffs frame the data-policy reset.
- 2026-04-19: NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF, RELAUNCH_UPGRADE_REVIEW_PACKET. Proper-data inventory introduced.
- 2026-04-21: R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN. RLP3.5, RLP4.
- 2026-04-22: RELAUNCH_PACKET3 retros, PACKET4 plan.
- 2026-04-23: VISOMASTER_TEAMS_POOL_DIAGNOSTIC_FINDINGS. PACKET6 plan.
- 2026-04-24: PACKET7_CAMERA_SIGNATURE_HANDOFF. Anchor-pool spread documented.
- 2026-04-25: STATE_OF_THE_TEAMS_DETECTOR. FULL_STORY_PRE_PACKET_9. P8A regression discovered.
- 2026-04-26 (today): PACKET_9_MID_FLIGHT_HANDOFF. P10 plan written. P10 partial results: anchor pools unchanged.

### 9.4 Key W&B run IDs (from sub-report 01)

| Name | W&B ID | Step | Role |
|---|---|---|---|
| R12g | `0xxqwhxg` | 14000 | Most recent clean-CLIP scratch ancestor |
| RLP6_04 | `h2pdu6i5` | 23500 | The FT base for everything from RLP7 onwards |
| P8A | `9lmvb5b4` | 5000 | Anchor breakthrough but recall regression |
| P10_SYM_baseline | `osji02ho` | 4500 | Just finished early-stop; partial scorecard pending |
| P10_GRL_baseline | `ntbx1hh1` | 4500 | Just finished |
| P10_SYM_LIGHT | `2wajepid` | 5500 | Just finished |

---

## 10. Glossary

- **Anchor pool / anchor frame** — A small set of frames (~30) used as a high-signal probe for camera-signature shortcuts. The canonical anchor is the Dor-on-Logitech-webcam clip the model historically over-flags as fake.
- **ArcFace** — Angular-margin classification head. Originally for face recognition; here repurposed for binary real-vs-fake.
- **CLIP DataComp-XL** — The pretraining recipe of the OpenCLIP backbone we use. `ViT-B-16-DataComp-XL` from LAION.
- **Combined paired** — `data/sources/combined_paired.py` — the multi-source data-loading strategy that aggregates df40, deeplive, visomaster, teams, and external sources with per-family weights.
- **Effort detector** — The SVD-residual + ArcFace + frozen-CLIP architecture in `detectors/effort_detector.py`.
- **family_weights** — Per-source (per-family) sampling weights in `combined_paired.sampling.family_weights`.
- **GRL** — Gradient Reversal Layer. A layer that reverses gradient sign during backprop, used to make the embedding invariant to a domain label. Currently wired to a 4-bucket quality-domain head.
- **identity_key** — The split-key used to enforce identity-disjointness across train/dev/lockbox. Defined per-source.
- **Lockbox** — A hold-out partition used to read out FPR/recall at the contract-calibrated τ. NOT used during training.
- **P8A / Packet 8A** — `R13_RLP8_01_unfreeze_clip_codec.yaml` — the experiment that unfroze visual.proj + ln_post + MLP-SVD and showed an anchor-pool breakthrough alongside a per-method recall regression.
- **proper_data** — The April-19 inventory that replaced visomaster legacy. 1826 captures × 4 lanes.
- **RLP** — "Relaunch Packet" — the post-2026-04-17 packet sequence. RLP1 through RLP6 then RLP7, RLP8, etc.
- **R12g** — The most recent clean-CLIP scratch checkpoint, ~47 days old, base for the RLP1-RLP6 chain.
- **τ (tau)** — The decision threshold. Trainer's value_composite uses one τ; promotion contract calibrates a different τ on dev.
- **value_composite** — Trainer-time aggregate metric. Memory says it is NOT deployment-grade.
- **visomaster_bad_data** — The April-17 policy that excluded 4904 samples flagged as under-swapped due to the `face_parser_enabled=True` bug.

---

## 11. Final Note from the Author

This report was written by an agent (me) that has been participating in the conversation that produced the most recent P10 packet. I have a vested interest in not declaring my own packet's framing wrong. Where this report says "the framing was wrong" or "we did not measure the right thing," I am implicating prior work I helped author.

I have tried to be honest about that. The reading agent should treat sub-reports 01–06 as factual and sourced; sub-report 06 in particular is annotated with FLAG markers where conclusions warrant skepticism. The synthesis in sections 4–8 of this master is more interpretive and should be challenged.

**The single most important thing the receiving agent can do** is not produce a plan but produce an independent answer to:

> "How do we prove the model has no shortcut learning before we trust its metrics?"

That is the methodology the project needs. Everything else — better data, better augmentation, better FT recipes — is downstream of having this methodology.

— Claude, 2026-04-26
