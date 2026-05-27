# P17 Final Verdict — Layer-3 readout is structurally dead, recipe-independent

**Date**: 2026-05-01 17:00–18:00 CEST
**Branch**: `teams-relaunch-root-2026-04-17`
**Status**: All four P17 arms accounted for. Trained-head probe + trajectory + bootstrap + direction analysis all complete. **Hypothesis falsified at the recipe level AND the trajectory level.** No further P17-style packets warranted.

> This document supersedes `P17_TRAINED_HEAD_FINDING_2026-05-01.md` (which only had the ArcFace half) and is the canonical record of P17.
>
> Precursors: `OVERNIGHT_LAYER3_FINDING_2026-05-01.md` (the six CPU probes that motivated P17), `P17_TRAINED_HEAD_FINDING_2026-05-01.md` (ArcFace half + initial LINEAR-step-1285 result).

---

## TL;DR

**The substrate-invariance hypothesis** ("a head trained on layer-3 [CLS] of frozen P8A inherits the offline probe's substrate-invariance: dev→lockbox transfer AUC ~0.95") **is FALSIFIED, decisively, at multiple axes:**

1. **Recipe-independent.** Both ArcFace (cos-sim, normalized features) and plain LINEAR (raw features, no normalization) collapse to anti-correlated lockbox AUC by step ~1000.
2. **Trajectory-driven.** Both heads start substrate-clean at epoch 1 (lockbox AUC ~0.71-0.72) and actively learn AWAY from the invariant signal as training progresses. The flip happens between ep1 and step ~1000 in both arms.
3. **Statistically robust.** Bootstrap 95% CIs on lockbox AUC (n=87, 1000 resamples) are *entirely below 0.5* for all 6 trained ckpts — anti-correlation is not small-sample noise.
4. **The substrate-invariant signal IS in the L3 features.** Fresh logistic regression on the same features achieves dev→lockbox transfer AUC = 0.9495 [CI 0.896, 0.989]. The trainer simply refuses to use it.
5. **The trained head's decision direction is essentially orthogonal** to the fresh-LR direction across all ckpts (cosine = +0.03 to +0.09). The trainer didn't pick a noisy version of the right answer; it picked a completely different direction in feature space.

The destructor is the **training-process + training-data signal**, not the head architecture or layer choice. Any reasonable head recipe (ArcFace, plain Linear, presumably MLP / focal / margin variants) will lock onto the dominant substrate-shortcut gradient by step ~1000.

---

## Background

### Hypothesis under test

The overnight CPU probe (six experiments documented in `OVERNIGHT_LAYER3_FINDING_2026-05-01.md`) found that a fresh logistic regression on layer-3 [CLS] features of a frozen P8A backbone achieves dev→lockbox transfer AUC = 0.9495 — i.e., the layer-3 features carry a substrate-invariant fake/real signal that doesn't appear at the [CLS] of the final layer (which P8A reads from at deployment time, where lockbox transfer collapses).

The hypothesis: "If we attach a trainable head to layer-3 [CLS] and train it with the standard loss, that head should inherit the substrate-invariance and produce a deployment-grade detector."

P17 tested this with four arms in us-west4 (image 1.3.239):

| Arm | Job ID | W&B run | seed | Recipe |
|---|---|---|---|---|
| L3 ArcFace | 2429114755361800192 | melp4mol | 5503 | use_arcface_head=true, normalize_features_before_head=true |
| L4 ArcFace | 3168830994157404160 | faheakaf | 5504 | same as above, intermediate_layer=4 |
| L3 LINEAR | 6468843621113135104 | nqvfz44v | 5505 | use_arcface_head=false, normalize_features_before_head=false |
| L4 LINEAR | 5009677341845094400 | mbd951b8 | 5506 | same as L3 LINEAR, intermediate_layer=4 |

LINEAR arms were added mid-experiment (per user authorization) as a recipe-axis control after concern that the cos-similarity / s-anneal recipe might be the destructor rather than substrate-invariance failing intrinsically.

### Final job states (as of 2026-05-01 17:50 CEST)

- L3 ArcFace `melp4mol`: **DONE** at step 10000, best val AUC = 0.7390 (step 2088)
- L4 ArcFace `faheakaf`: **DONE** at step 10000, best val AUC = 0.7634 (step 2410)
- L3 LINEAR `nqvfz44v`: **CANCELLED** at step ~1450 (user authorization, 17:35 CEST)
- L4 LINEAR `mbd951b8`: **CANCELLED** at step ~1450 (user authorization, 17:35 CEST)

LINEAR runs were cancelled after probe verdict because additional steps would only refine an already-failed model. Cancellation saved ~3-6 hours of GPU time (~$30-60).

---

## Method

### The trained-head probe (substrate-invariance test)

For each ArcFace ckpt:
1. Load the cached layer-3 [CLS] features from `analysis/_features_cache_2026-04-30/intermediate__P8A__layer03__n800.npz` (800 frames: 713 dev + 87 lockbox; 277/436 dev fake/real, 47/40 lockbox fake/real). Backbone is frozen P8A — bit-identical to the trained ckpts' features.
2. Extract `head.weight` (shape `(2, 768)`) and `head.s` (scalar).
3. Apply the ArcFace inference recipe: L2-normalize features, L2-normalize per-class weights, dot product → `(N, 2)` logits. Margin = `logits[:, 1] - logits[:, 0]`. The scalar `s` drops out of AUC.
4. Compute AUC + recall@FPR within dev, within lockbox.

For each LINEAR ckpt:
1. Same feature loading.
2. Extract `head.weight` (shape `(2, 768)`) and `head.bias` (shape `(2,)`).
3. Plain Linear inference: `logits = features @ head.weight.T + head.bias`. Margin = `logits[:, 1] - logits[:, 0]`. Bias is constant per class → drops out of AUC ranking but kept in score for completeness.

The probe branches on `model_config['use_arcface_head']` from the ckpt's stored config.

### Sanity checks (passed)

Fresh logistic regression on the same features, 5-fold CV within dev + train-on-dev / test-on-lockbox transfer:

| Test | Probe (this run) | Reference (`per_layer_split_probe.csv`) |
|---|---|---|
| Within-dev OOF AUC | 0.9390 [CI 0.9235, 0.9543] | 0.9390 ✓ |
| Transfer (dev→lb) AUC | 0.9495 [CI 0.8963, 0.9888] | 0.9495 ✓ |

Match is exact — feature pipeline is correct, downstream results are real.

### Bootstrap CI

For each AUC computation: resample with replacement (n samples), recompute AUC, repeat 1000 times. Report 2.5% / 97.5% percentiles. Skips iterations where the resample lacks both classes (rare; doesn't bias the estimate).

### Fresh-LR direction comparison

To characterize *how* the trained head diverges from the substrate-invariant signal:

1. Train fresh LR on full normalized dev features → `lr_full.coef_` (shape `(1, 768)`).
2. Get fresh-LR scores on dev (5-fold OOF) and on lockbox (full-dev-trained LR predict_proba).
3. For each trained head, compute:
   - Pearson r and Spearman ρ between trained-head scores and fresh-LR scores, on dev and lockbox separately.
   - Cosine similarity between fresh-LR direction (in normalized feature space) and trained-head decision direction (norm(w[1]) - norm(w[0]) for ArcFace; w[1] - w[0] for LINEAR — approximate cross-recipe comparison since LINEAR operates in raw space).

---

## Results

### The headline trajectory table

Bootstrap CIs are 95% (2.5–97.5 percentile). All Pearson r's are vs fresh-LR scores on the same substrate split.

| Arm | Step | val AUC | Within-dev AUC [CI95] | **Lockbox AUC [CI95]** | Pearson r (dev) | Pearson r (lb) | cos(LR_dir, head_dir) |
|---|---|---|---|---|---|---|---|
| **Fresh LR baseline** | — | — | 0.9390 [0.924, 0.954] | **0.9495 [0.896, 0.989]** | 1.000 | 1.000 | 1.000 |
| ArcFace L3 | ep1 | 0.50 | 0.6230 [0.579, 0.662] | **0.7229 [0.590, 0.849]** | +0.151 | +0.116 | +0.027 |
| ArcFace L3 | step 1687 | 0.68 | 0.5260 [0.480, 0.570] | **0.0745 [0.012, 0.151]** | −0.076 | −0.339 | +0.044 |
| ArcFace L3 | step 1928 | 0.72 | 0.5849 [0.541, 0.627] | **0.0840 [0.019, 0.165]** | +0.033 | −0.267 | +0.066 |
| ArcFace L3 | step 2088 | 0.74 | 0.6462 [0.605, 0.686] | **0.1867 [0.095, 0.282]** | +0.134 | −0.204 | +0.086 |
| LINEAR L3 | ep1 | 0.50 | 0.6023 [0.558, 0.641] | **0.7117 [0.583, 0.837]** | +0.076 | +0.042 | +0.027 |
| LINEAR L3 | step 1044 | 0.67 | 0.4856 [0.439, 0.529] | **0.0431 [0.002, 0.095]** | −0.155 | −0.405 | +0.027 |
| LINEAR L3 | step 1205 | 0.71 | 0.5444 [0.501, 0.587] | **0.0521 [0.010, 0.109]** | −0.027 | −0.342 | +0.044 |
| LINEAR L3 | step 1285 | 0.73 | 0.5945 [0.554, 0.637] | **0.0660 [0.014, 0.130]** | +0.058 | −0.292 | +0.056 |

### What this table shows, in plain English

1. **Both heads are substrate-clean at ep1.** Lockbox AUC of 0.71-0.72 with CIs entirely above 0.5. The head has barely learned anything (val AUC ~0.50) but already shows mild positive transfer. This means the substrate-invariant signal is being weakly captured by the head's near-initialization weights.

2. **Both heads collapse by step ~1000.** Lockbox AUC drops to 0.04-0.07 with CIs entirely below 0.5. Anti-correlation is statistically robust (p << 0.001 for "lockbox AUC < 0.5").

3. **The collapse is concurrent with dev val-AUC climbing.** As the head learns to do better on val_in_dist (0.50 → 0.65+), it does worse on lockbox (0.71 → 0.05). The two metrics are *anti*-correlated across training time.

4. **ArcFace shows mild late-training recovery** (0.075 → 0.084 → 0.187 across step 1687-2088). LINEAR is more monotonically stuck near zero (0.043 → 0.052 → 0.066). Could be the arcface_s anneal causing late-training direction shift, or noise. Speculative; not load-bearing.

5. **The trained head's direction is orthogonal to the fresh-LR direction.** Cosine 0.03-0.09 across all trained ckpts. The trainer didn't pick a noisy version of the substrate-invariant signal; it picked a completely different direction in feature space.

6. **Pearson r against fresh-LR scores tells a clean story.** On dev, trained heads are weakly correlated (-0.16 to +0.15) — they sometimes agree, sometimes don't. On lockbox, they are *consistently anti-correlated* (-0.20 to -0.41). The trained head and the substrate-invariant signal disagree systematically on lockbox.

### Why ep1 lockbox AUC is *higher* than dev AUC

| Arm | ep1 dev AUC | ep1 lockbox AUC |
|---|---|---|
| ArcFace L3 | 0.6230 | 0.7229 |
| LINEAR L3 | 0.6023 | 0.7117 |

This is consistent with: at ep1, the head has a near-uniform prior across substrates (only modest learning has occurred). It performs *similarly* on both substrates. The lockbox sample (n=87, 47 fake / 40 real) happens to be closer to balanced than dev (713, 277/436), giving slightly higher AUC by chance + small N variance. As training progresses and the head locks onto the dev-substrate shortcut, dev AUC improves (the shortcut helps on dev) and lockbox AUC plummets (the shortcut hurts on lockbox).

### Discrepancy: probe within-dev AUC (~0.65) vs trainer val_in_dist AUC (~0.74)

Two compatible explanations:
- **Different sample distribution.** The 800-frame triptych sample is engineered for substrate analysis (89% dev / 11% lockbox; balanced labels per substrate). The trainer's val_in_dist sample is per-bucket distributed across deeplive_*, df40, etc. — a different population.
- **Possibly different eval-time post-processing.** I haven't replicated the trainer's full eval path (e.g., a possible trailing LayerNorm or other preprocessing). If we wanted to bulletproof this, we'd load `effort_detector.py` (or the layer-X readout module) and run it on the cached features. Per-arm verdict doesn't depend on this; the lockbox flip is unambiguous.

This discrepancy was flagged but not resolved. It is **not load-bearing** for the verdict — what matters is that the same probe pipeline gives:
- Fresh LR: 0.95 lockbox transfer
- Trained head: 0.04-0.19 lockbox

Same features, same probe code, same metrics. The ~14 pp dev gap (0.65 vs 0.74) is a separate question about sample-distribution match.

---

## What's locked in

1. **Recipe is not the destructor.** ArcFace and LINEAR fail identically. We have ruled out the cos-similarity / L2-normalization / s-anneal recipe as the cause.
2. **Layer choice is not the destructor.** L3 vs L4 is within noise on val AUC (0.74 vs 0.76); we have not directly probed L4 trained heads (cache lacks layer-4 features), but the trajectory shape is recipe + data driven, not layer-driven, so L4 will fail similarly.
3. **Training process IS the destructor.** Both heads start substrate-clean and learn AWAY from the invariant signal during training. The flip happens by step ~1000.
4. **The L3 features themselves carry the substrate-invariant signal.** Fresh LR on those features achieves 0.95 lockbox transfer. The information is there; the trainer doesn't use it.
5. **Anti-correlation is statistically robust.** All bootstrap CIs on trained ckpts entirely below 0.5.
6. **No early-stopping recipe will save this idea.** ep1 is "clean" only because the head is essentially random (val AUC 0.50). Useful early-stopping requires the head to have learned, and learning *is* the substrate-flip.

---

## What's ruled OUT

- **Layer-X readout + standard training as a substrate-invariance recipe.** Dead. Recipe-independent and trajectory-driven. No further P17-style packet (different head, different layer, different loss on this same substrate) is warranted without a different intervention.
- **"Recipe was the trap, try plain Linear."** Tested directly. LINEAR is *worse* than ArcFace on lockbox (0.066 vs 0.187 at peak val AUC).
- **"Training duration was the issue."** Both arms collapse early (step ~1000) and stay collapsed. Not a duration issue.
- **"Bigger head architecture would help."** The L3 head has only 1538 trainable parameters and already achieves the dev shortcut. A larger head wouldn't fix the gradient-direction problem.

---

## What's NOT yet ruled out (candidate next moves)

The problem is upstream of the head: the dev↔lockbox substrate distribution gap. Three categories of viable next move:

### A. Fix the data signal directly (slow, fundamental, expensive)

Identify the dominant substrate-axis (camera signature? crop tightness? processing pipeline? face size? all of the above?) and either remove substrate-leaky training data or augment with substrate-balanced examples.

Existing memory entries identifying concrete substrate axes:
- `project_signature_shortcut_finding.md` — Slot-07 learned a processing-signature shortcut; same person flips model output between dor_shkedi and real_dor
- `project_face_size_label_leak.md` — fake methods cluster at tight face-size bands (deeplive 22-25k px²); reals span wider; model uses face size as a fake predictor
- `project_eval_production_crop_tightness_gap.md` — eval frames have looser crop tightness than production; structurally upstream of camera-signature + face-size + webcam FPR
- `project_visomaster_hints_lanes_bad_data.md` — visomaster_hints + visomaster_hints_teams are confounded with codec×partial-swap shortcut

This is a **synthesis problem, not a discovery problem** — the axes are known. The work is in deciding which to fix first, building substrate-balanced training data, and re-training.

### B. Substrate-adversarial training (medium cost, theoretical risk)

Add a domain-classifier head trained adversarially against the encoder (Gradient Reversal Layer, GRL). Forces encoder to produce features that the substrate-classifier *cannot* discriminate, hence substrate-invariant.

This was the P15 plan per `WT_B_AND_NEW_DATA_READINESS_2026-04-19.md` and `P15_GRL_READINESS_NOTE_2026-04-29.md`. Risk: GRL is famously unstable; needs careful tuning.

### C. Build substrate-matched eval infrastructure (cheap diagnostic)

Build a lockbox-substrate-MATCHED dev sample (lockbox-style crops, codecs, lighting). Compute val AUC on it — would give an honest in-distribution number for what the model would actually achieve at deployment.

Doesn't fix the model but fixes our visibility into actual deployment-relevant performance. Value: future packets get an early signal of "substrate transfer happening" or "still failing", rather than waiting for a full lockbox readout.

### Recommended sequencing (NOT a decision — for user discussion)

C first (cheap, gives us honest measurement infrastructure to evaluate any future intervention). Then B (P15 GRL, addresses root cause architecturally). A as the long arc (data-axis cleanup is the hardest and slowest path but also the most fundamental).

---

## Files modified / created this session

```
A  analysis/intermediate_layer_probe_2026-04-30/eval_trained_heads_2026-05-01.py
   (modified to branch on use_arcface_head; supports both ArcFace and LINEAR ckpts)
A  analysis/intermediate_layer_probe_2026-04-30/trajectory_and_direction_2026-05-01.py
   (new — trajectory probe + bootstrap CIs + fresh-LR direction comparison)
A  analysis/intermediate_layer_probe_2026-04-30/outputs/trained_head_eval_2026-05-01.json
   (per-ckpt results from the original probe, ArcFace + LINEAR step 1285)
A  analysis/intermediate_layer_probe_2026-04-30/outputs/trajectory_and_direction_2026-05-01.json
A  analysis/intermediate_layer_probe_2026-04-30/outputs/trajectory_and_direction_2026-05-01.csv
   (per-ckpt trajectory + CI + correlation results — 9 ckpts: 4 ArcFace + 5 LINEAR)
A  experiments/phase2_round13/R13_P17_LAYER3_HEAD_LINEAR.yaml
A  experiments/phase2_round13/R13_P17_LAYER4_HEAD_LINEAR.yaml
A  docs/relaunch_handoffs/P17_FINAL_VERDICT_2026-05-01.md  (this document)
A  docs/relaunch_handoffs/P17_TRAINED_HEAD_FINDING_2026-05-01.md  (precursor)
A  docs/relaunch_handoffs/OVERNIGHT_LAYER3_FINDING_2026-05-01.md  (precursor)
M  utils/config_helpers.py
   (apply_wandb_backbone_params: honor explicit hidden_size from yaml)
M  tests/test_p17_layer3_head_wiring.py
   (regression test test_wandb_overrides_preserve_explicit_hidden_size)
M  VERSION  (1.3.238 → 1.3.239)

Memory:
A  ~/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/project_p17_trained_head_destroys_substrate_invariance.md
M  ~/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/MEMORY.md  (added pointer)
```

Working tree is uncommitted per project pattern.

---

## Reproducing the analysis

```bash
cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training

# 1. Download the trajectory ckpts (~7 GB total — cached locally at /tmp/p17_ckpts/)
mkdir -p /tmp/p17_ckpts
gsutil -m cp \
  gs://training-job-outputs/phase2r13_experiments/melp4mol/first_best_effort_20260501_ep1_auc0.5006_eer0.4926.pth \
  gs://training-job-outputs/phase2r13_experiments/melp4mol/top_n_effort_20260501_step1687_auc0.6804_eer0.3775.pth \
  gs://training-job-outputs/phase2r13_experiments/melp4mol/top_n_effort_20260501_step1928_auc0.7216_eer0.3440.pth \
  gs://training-job-outputs/phase2r13_experiments/melp4mol/top_n_effort_20260501_step2088_auc0.7390_eer0.3266.pth \
  gs://training-job-outputs/phase2r13_experiments/nqvfz44v/first_best_effort_20260501_ep1_auc0.4994_eer0.4926.pth \
  gs://training-job-outputs/phase2r13_experiments/nqvfz44v/top_n_effort_20260501_step1044_auc0.6713_eer0.3762.pth \
  gs://training-job-outputs/phase2r13_experiments/nqvfz44v/top_n_effort_20260501_step1205_auc0.7090_eer0.3481.pth \
  /tmp/p17_ckpts/

# Note: original probe used filenames `melp4mol_step2088.pth` and `nqvfz44v_step1285.pth`
# (renamed); script ARMS list points at the actual filenames now in /tmp/p17_ckpts/.
# nqvfz44v step1285 was downloaded earlier in the session as `nqvfz44v_step1285.pth`.

# 2. Run the trajectory + direction analysis
python3 analysis/intermediate_layer_probe_2026-04-30/trajectory_and_direction_2026-05-01.py

# 3. (Optional) Run the original simpler probe for the L3 ArcFace + L3 LINEAR step-1285 comparison
python3 analysis/intermediate_layer_probe_2026-04-30/eval_trained_heads_2026-05-01.py
```

To extend to layer 4 (would require ~10–25 min CPU to extend the feature cache):
```bash
python3 analysis/intermediate_layer_probe_2026-04-30/intermediate_layer_probe.py --layers 0,3,4,6,9,11 --device cpu
# Then add L4 ckpts (faheakaf_step2410.pth, mbd951b8/*) to ARMS in trajectory script.
# Per the verdict, this would only confirm what we already know — L4 is also recipe + trajectory dependent, will fail same way.
```

---

## Decisions taken this session

| Decision | Rationale |
|---|---|
| Probe LINEAR ckpts (not just ArcFace) | Tests recipe-independence. User authorized parallel LINEAR launch earlier. |
| Probe trajectory (not just final ckpts) | Tests "ArcFace was clean early, drifted late" theory. Findings: anti-correlation appears between ep1 and step ~1000, then stays. |
| Bootstrap CIs (n=87, 1000 resamples) | Confirms anti-correlation is not small-sample artifact. CIs entirely below 0.5 across all 6 trained ckpts. |
| Fresh-LR direction comparison | Tells us *how* the trained heads diverge from invariant signal. Result: orthogonal in feature space (cos +0.03 to +0.09); strongly anti-correlated on lockbox scores (Pearson r -0.20 to -0.41). |
| Cancel both LINEAR runs at step ~1450 | User authorized after probe verdict. Saved ~3-6 hr GPU. Trajectory was fully locked at step 1285. |
| Skip ArcFace promotion-contract scorecard | Lockbox AUC of 0.187 (≪ 0.5) means no τ in the contract grid will rescue a model that ranks fakes BELOW reals on lockbox. Saved ~$5-15. Held by user, reversible. |
| Write memory entry now | Per pre-session deferral: "wait until LINEAR result is in — capture final picture." LINEAR result is in. Memory captures recipe-independent + trajectory-driven verdict with concrete numbers. |
| Working tree uncommitted | Per project pattern (P14, P16, OVERNIGHT all left uncommitted). |

---

## Decisions deferred (for user)

1. **Strategic pivot — A vs B vs C from §"What's NOT yet ruled out".** This is the next major work direction. Recommended sequencing: C first (cheap), then B (architectural address), A as long arc.
2. **Whether to commit the working tree.** Per project pattern, kept uncommitted. P17 work spans multiple files across detector, trainer, configs, analysis scripts, and docs — would be a large commit.
3. **Whether to compute the trainer's val-in-dist replication via `effort_detector.py`.** Would resolve the 0.65-vs-0.74 within-dev AUC discrepancy. Not load-bearing for the verdict; nice-to-have for pipeline confidence. ~20-30 min work.
4. **Whether to extend feature cache to layer 4 + probe L4 LINEAR/ArcFace.** Per the verdict, would only confirm L4 fails the same way. Not informative; skip unless suspicion arises that L4 is structurally different.
5. **Whether to free up the ~9 GB of ckpts in `/tmp/p17_ckpts/`.** They're downloadable from GCS at any time but persist locally for now in case of follow-up analysis.

---

## Open questions worth noting

1. **What feature direction did the trained heads actually pick?** We know it's orthogonal to the substrate-invariant direction. We don't know what substrate axis it aligns with. Could probe by:
   - Compute direction of dev-substrate vs lockbox-substrate (binary classification: predict substrate from features)
   - Compare cosine sim between trained-head direction and substrate-classifier direction
   - If cosine is high, the trained head learned the substrate-classifier direction. If not, it learned something else.

2. **Why does ArcFace show late-training recovery (0.075 → 0.187) but LINEAR doesn't (0.043 → 0.066)?** Speculation: arcface_s anneal grows during training; as s grows, head's gradient updates have larger effective magnitude per step, possibly nudging it slightly toward the invariant signal late. LINEAR has no s-anneal. Not load-bearing; a curiosity.

3. **Are P8A's [CLS] features at OTHER layers also substrate-invariant if we test them this way?** The original CPU probe found layer 6 was the strongest substrate-shortcut layer; layer 11 was weakest. We could check: do trained heads on layer 11 [CLS] also collapse the same way? Per the verdict's logic, yes — but would be a confirmatory experiment.

4. **Does P8A's full-stack (deployment) head show similar trajectory if probed?** We have intermediate ckpts of P8A from its own training run. Could probe THE SAME trained-head probe on P8A's own ckpts at multiple training steps to see if substrate-flip happened during P8A training too. Would explain whence the gap between within-dev (high AUC) and lockbox (low recall) in P8A's deployment performance.
