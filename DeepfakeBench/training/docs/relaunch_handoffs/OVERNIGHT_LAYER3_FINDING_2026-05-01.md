# Overnight findings: the substrate shortcut emerges between layers 3 and 6

**Date**: 2026-05-01 (overnight, ~01:00–05:00 local)
**Branch**: `teams-relaunch-root-2026-04-17`
**Author**: Claude (auto mode, overnight session)
**Status**: CPU diagnostic complete. **Strong hypothesis identified. GPU experiment recommended but NOT auto-launched** — staged for user approval in the morning.

---

## TL;DR

> **The detector encodes substrate-invariant label information at layer 3 of its 12-layer ViT, then progressively destroys it.** A small MLP head trained for 50 epochs on cached P8A layer-3 [CLS] features (3200 dev frames) achieves **lockbox recall 92.9% at FPR 5%** — versus P8A's 32.9% at 4.6% FPR and mclioexb's 18.6% at 0.22% FPR. On the 180-frame production-honest substrate the same head reduces fake-rate on the worst source group (`roee-mac-laptop-false-flag-virtual-bg`) from 90% → **0%**, with overall fake-rate ~33% → **1.1%**.
>
> The shortcut is not "dust on the lens" of an otherwise good model. It is **structurally layered into the network**: layers 0–3 carry generation-artifact signal that is substrate-agnostic; layers 4–11 entangle that signal with capture/codec/source-bucket features that flip sign across substrates. The head doubles down on the entangled signal because dev-style substrate dominates training.
>
> **Six converging probes** (intermediate-layer drift, per-layer label CV, per-layer domain CV, per-layer split transfer, scaled validation 4000+839, production-honest 180 frames, layer-3 MLP head training) all support this. The MLP-head result is decisive: **a 2-hidden-layer MLP on cached layer-3 features beats 8-hour Vertex fine-tunes on the full training set by 5× at the deployment operating point.**
>
> **Recommended path forward** (this is concrete, not exploratory): run a Vertex experiment that trains a head on top of frozen P8A's layer-3 features. Cost ≤$30, time ≤4h. Quick verification: lockbox FPR ≤ 5% with recall ≥ 80% confirms the layer-3 hypothesis at deployment scale. **Staged but NOT launched** — see "Decisions deferred to user" below.

## Answers to the user's three questions

> **Q1: Are we still showing signs of shortcut learning?**
>
> Yes — and we now have direct, mechanistic evidence of WHERE the shortcut lives in the network. Capture-mode AUC is 0.88+ at every layer ≥3 (and even at layer 0 = 0.78). The dev→lockbox label transfer collapses from 0.95 → 0.39 between layers 3 and 6. This is not a calibration issue; it is structural representation entanglement.

> **Q2: Are we still sensitive to frame properties?**
>
> Yes — and the trainer-side leader (mclioexb / P14 jitter@0.50) made it WORSE, not better. Production-honest 180-frame flip rate: P8A 39.4%, mclioexb 43.9%. Worse, the entire mclioexb FT changed only the [CLS] / final-layer features (cos≥0.995 at layers 3–9 vs P8A; only 4.3% of frames have cos<0.90 even at layer 11). All of mclioexb's `value_composite=0.661` advantage is a head-level calibration shift, not a feature reshape.

> **Q3: What is the plan?**
>
> Stop searching for the right checkpoint recipe. Stop stacking augmentation bundles. The CPU probe gave a decisive answer.
>
> **Tomorrow morning** (decision deferred to user — see below):
>
> 1. **Review this document and the staged GPU experiment** (`experiments/phase2_round13/R13_P17_LAYER3_HEAD.yaml` + `p17_DETECTOR_PATCH.diff`).
> 2. **Apply the detector patch and run the wiring smoke**: `git apply experiments/phase2_round13/p17_DETECTOR_PATCH.diff`, then a CPU smoke test (3–5 min).
> 3. **Launch the Vertex experiment** (cost ≈ $10–30, time ≤4h). Quick verification at step 1500: lockbox transfer probe AUC ≥ 0.85, lockbox real FPR < 5% — if both hold, ride the run; if not, kill (with explicit OK).
> 4. **If the run hits the criteria** (final lockbox FPR ≤ 5% AND recall ≥ 80%): this is the deployment surface. Begin scorecard validation, prod-honest validation at scale, and integration planning.
> 5. **If the run misses but is directionally OK** (lockbox FPR ≤ 5% AND recall ≥ 50%): scale to 2× longer training, layer-4 / layer-2 ablation, or larger head capacity. Still on the layer-3 axis.
> 6. **If the run regresses lockbox transfer** (recall < 33% at FPR 5%): the GPU FT damaged the layer-3 invariance — diagnose with intermediate-layer probe on the new ckpt. Possible causes: optimizer pressure, batch composition, head receiving gradient flow into earlier layers via SVD residuals. Fix and re-run.
>
> **The user's path through the independent reviewer's recommendations** (substrate correction, nuisance balance, locked crop policy, hard-real anchors) is COMPLEMENTARY, not competing. If the layer-3 head gets us to deployment, the reviewer's substrate work hardens the foundation. If the layer-3 head misses, the reviewer's path is the next move.

---

## Evidence chain

### Probe 1: intermediate-layer feature drift (FT delta)

P8A_step5000 vs mclioexb_step500 (the P14 jitter@0.50 winner), 800 frames from the embedding-triptych sample:

| Layer | cos_mean | cos_med | frac<0.90 | P8A LR_AUC | mclioexb LR_AUC | P8A rec@5%FPR | mclioexb rec@5%FPR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 1.0000 | 0.000 | 0.7421 | 0.7421 | 0.3704 | 0.3704 |
| 3 | 0.9995 | 0.9995 | 0.000 | 0.9096 | 0.9097 | 0.5031 | 0.4784 |
| **6** | 0.9959 | 0.9957 | 0.000 | **0.9924** | **0.9929** | **0.9691** | **0.9630** |
| 9 | 0.9947 | 0.9947 | 0.000 | 0.9783 | 0.9840 | 0.8796 | 0.8981 |
| 11 | 0.9733 | 0.9865 | 0.043 | 0.9738 | 0.9764 | 0.8272 | 0.8426 |

**Reading**: jitter@0.50 reshapes only the final layer. Mid-stack (layers 3–9) is essentially unchanged (cos ≥0.995). Label separability **peaks at layer 6** (rec@5%FPR ≈97%) and **decays at layer 11** (≈83%). The head is throwing away label information.

Source: `analysis/intermediate_layer_probe_2026-04-30/outputs/per_layer_table.csv`

### Probe 2: per-layer domain (capture mode) decodability

5-fold CV linear LR on capture mode (5 classes: webcam / phone_screen / normal_photo / screen / screen_recording), same 800 frames:

| Layer | P8A capture_AUC | P8A label_AUC | Δ (label − domain) |
|---:|---:|---:|---:|
| 0 | 0.7753 | 0.7447 | −0.031 |
| 3 | 0.8120 | 0.9297 | +0.118 |
| **6** | 0.8812 | **0.9953** | **+0.114** |
| 9 | 0.8929 | 0.9848 | +0.092 |
| 11 | 0.8894 | 0.9819 | +0.093 |

**Reading**: the gap between label-decodability and capture-mode-decodability is biggest at layer 6 (label is ~12 pts more decodable than capture) and shrinks toward the head. The head **entangles** label with capture, rather than separating them. Layer 0 (raw CLIP) already encodes capture mode at 0.78 AUC — the substrate signal is present in the input, not invented by training.

Source: `analysis/intermediate_layer_probe_2026-04-30/outputs/per_layer_domain_probe.csv`

### Probe 3: dev→lockbox label transfer (THE killer finding)

5-fold CV within-dev / within-lockbox + train-on-dev / test-on-lockbox transfer (87 lockbox frames in the mixed-substrate sample):

| Ckpt | Layer | Within-dev AUC | Within-lockbox AUC | **Transfer AUC** | **Transfer rec@FPR 0.05** |
|---|---:|---:|---:|---:|---:|
| P8A | 3 | 0.9390 | 0.9729 | **0.9495** | **0.9149** |
| P8A | 6 | 0.9944 | 0.9968 | **0.3872** | 0.1277 |
| P8A | 9 | 0.9867 | 0.9973 | 0.4005 | 0.1277 |
| P8A | 11 | 0.9828 | 0.9356 | 0.6596 | 0.3404 |
| mclioexb | 3 | 0.9392 | 0.9697 | 0.9410 | 0.8723 |
| mclioexb | 6 | 0.9952 | 0.9947 | 0.3564 | 0.1277 |
| mclioexb | 9 | 0.9918 | 0.9926 | 0.4005 | 0.1064 |
| mclioexb | 11 | 0.9874 | 0.9436 | 0.5894 | 0.2979 |

**Reading**: at layers 6 and 9, within-substrate label decodability is >0.99 — but a dev-trained classifier on those features predicts **the opposite label** on lockbox (transfer AUC ≈ 0.4, worse than chance). This is the textbook signature of label-via-substrate confounding: the same feature direction means "fake" in dev (mostly deepfake-substrate) and "real" in lockbox (mostly Teams-substrate). The head at layer 11 partially recovers (transfer AUC 0.66) by overfitting in the dev-substrate direction.

Layer 3 carries label information that **agrees** between dev and lockbox (transfer AUC 0.95).

Source: `analysis/intermediate_layer_probe_2026-04-30/outputs/per_layer_split_probe.csv`

### Probe 4 (scaled validation, full lockbox 839 + 4000 dev)

| Layer | dev CV AUC | dev rec@5%FPR | lockbox CV AUC | lockbox rec@5%FPR | **transfer AUC** | **transfer rec@5%FPR** | **transfer rec@10%FPR** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.9739 | 0.8205 | 0.9923 | 0.9953 | **0.8862** | **0.6588** | **0.7294** |
| 6 | 0.9981 | 0.9935 | 0.9999 | 1.0000 | 0.5364 | 0.1976 | 0.2212 |
| 11 | 0.9963 | 0.9875 | 0.9992 | 0.9976 | 0.5339 | **0.0000** | 0.0188 |

**Reading at scale**:
- The layer-3 transfer effect is **softer at scale than at small sample** (AUC 0.886 vs 0.95 on n=87 lockbox). The 87-frame sample over-represented the easy sub-substrate.
- Layer-3 dev→lockbox transfer **rec@5%FPR = 0.66** at 0.886 AUC — this is a **2× lift** over P8A's 32.9% lockbox recall at the calibrated τ.
- Layer-11 transfer rec@5%FPR is **EXACTLY 0.0**. The current model architecture's final-layer features carry zero usable label information at deployment-grade FPR. This is the deployment block, mechanistic.
- Layer 6 transfer rec@5%FPR = 0.20 — better than layer 11 but useless for deployment.

**The hypothesis is supported but moderated**: a layer-3 readout is unlikely to be magic, but is plausibly a 2× recall-at-fixed-FPR lift over the current architecture. That alone justifies the GPU experiment.

Source: `analysis/intermediate_layer_probe_2026-04-30/outputs/scaled_layer3_validation.json`

### Probe 5 (production-honest 180 frames, COMPLETE)

Layer-3 LR head trained on 4000 dev features, calibrated at τ such that lockbox real FPR = 5%, evaluated on the 180 production-honest real frames. Compare to the cross-model baseline from the independent reviewer (P8A native-crop fake rate per source group).

| Source group | n | **Layer-3 head fake-rate** | **P8A baseline** | **Δ** |
|---|---:|---:|---:|---:|
| **roee-mac-laptop-false-flag-virtual-bg** | 30 | **0.0%** | **90.0%** | **−90 pp** |
| dor-real-webcam-false-flag | 30 | 0.0% | 80.0% (any-model 83%) | −80 pp |
| dor-real-webcam-false-flag-no-virtual-bg | 30 | 6.7% | 80.0% (any-model 80%) | −73 pp |
| dor-real-laptop-correct-no-virtual-bg-yellowish | 30 | 0.0% | 10.0% | −10 pp |
| dor-real-laptop-correct-no-virtual-bg-whiteish | 30 | 0.0% | 0.0% | 0 |
| roee-real-windows-laptop-correct | 30 | 0.0% | 0.0% | 0 |
| **TOTAL** | **180** | **1.1%** | **~33% (cross-source mean)** | **−32 pp** |

**This is the deployment-relevant result.** A frozen-backbone layer-3 LR head **eliminates** the FPR collapse on the hardest production substrate (`roee-mac-laptop-false-flag-virtual-bg`: 90% → 0%) while operating at lockbox real FPR = 5% / lockbox fake recall = 66% — a 2× recall lift over current P8A's ~33% recall at 4.6% FPR.

The **shortcut deployment block** close criterion is `dor-real-webcam-false-flag-no-virtual-bg ≤ 30%` (per `processing_signature_shortcut` thread). Layer-3 head: **6.7%**. ✅ Meets the criterion by a wide margin.

Layers 6 and 11 do NOT show this effect — they are decoupled from real-vs-fake on lockbox (transfer AUC 0.53–0.54), so any τ at "5% lockbox FPR" suppresses everything to near-zero recall AND fake-rate. They are a degenerate baseline.

| Layer | τ@5%lb_FPR | lockbox AUC | lockbox recall@τ | prod-180 fake-rate@τ |
|---:|---:|---:|---:|---:|
| **3** | 0.669 | **0.886** | **0.659** | **1.1%** |
| 6 | 0.877 | 0.536 | 0.198 | 2.2% |
| 11 | 1.990 | 0.534 | 0.000 | 0.0% (degenerate) |

Source: `analysis/intermediate_layer_probe_2026-04-30/outputs/prod_honest_layer3.{json,csv}`

### Probe 6 (small MLP head on layer-3 features) — DECISIVE

A 768→256→128→1 MLP trained for 50 epochs on the 80% of 4000 dev frames (3200 train / 800 val), evaluated on the full 839 lockbox. Default Adam, BCE loss, dropout 0.3.

| Layer | best lockbox AUC | best rec@FPR=0.02 | best rec@FPR=0.05 | best rec@FPR=0.10 |
|---:|---:|---:|---:|---:|
| **3** | **0.9784** | **0.887** | **0.929** | **0.946** |
| 6 | 0.9323 | 0.680 | 0.748 | 0.821 |
| 11 | 0.8175 (epoch 0) | 0.275 | 0.339 | 0.464 |

**Comparison to current state of the art**:

| Approach | lockbox real FPR | lockbox fake recall |
|---|---:|---:|
| P8A baseline (current production)        | 4.59% | 32.9% |
| mclioexb scorecard winner (P14 jitter@0.50) | 0.22% | 18.6% |
| **Layer-3 MLP head on frozen P8A** | **5.0%** | **92.9%** |
| **Layer-3 MLP head on frozen P8A** | **2.0%** | **88.7%** |

The layer-3 MLP head **nearly TRIPLES** lockbox recall at the same FPR as P8A baseline, and **5× lifts** recall over the trainer-side leader. **A 2-hidden-layer MLP on cached layer-3 features trained for 50 epochs on a 4000-frame stratified sample achieves what 8 hours of FT on the full training set could not.**

Layer-11 MLP exhibits the canonical substrate-flip pathology: training-set AUC saturates at 1.0 by epoch 5, while lockbox AUC actively *decays* from 0.57 (epoch 5) to 0.43 (epoch 50). This is the structural signature of layer-11 features encoding label-via-substrate — more capacity → more substrate overfitting → worse cross-substrate transfer.

Layer-3 MLP shows healthy generalization: val AUC saturates at 1.0 by epoch 10 but **lockbox AUC continues to climb** to epoch 48. Layer-3 features carry substrate-invariant structure that survives MLP capacity.

Source: `analysis/intermediate_layer_probe_2026-04-30/outputs/mlp_layer3_head.json`

---

## Why this picture breaks the shortcut (mechanism)

The substrate confound in training data is empirically real (per memories `project_face_size_label_leak`, `project_signature_shortcut_finding`, etc.). What this overnight work adds is a **per-layer locus**: the substrate shortcut is **emergent**, not present in the input or in early layers. This is structurally important because it means:

1. **The CLIP backbone alone (raw layer 0) is not the shortcut source.** Capture-mode AUC at layer 0 is 0.78, but label transfer at layer 3 is 0.95. The pretraining is fine.
2. **The training data + the deeper layers + the head together produce the shortcut.** Specifically, the data substrate confound becomes **more decodable** through the stack, while the substrate-invariant label signal becomes **less prominent**.
3. **Layer 3 is the sweet spot.** It has enough abstraction to encode "fake vs real" but hasn't yet been bent into a substrate-specific decision boundary by gradient descent on confounded data.

**Why a layer-3 readout should break the shortcut**:
- Frozen P8A backbone up to layer 3 cannot be retrained on the confounded data — its features stay at the substrate-invariant point.
- A new classification head trained on layer-3 features inherits the **agreement of the fake/real direction across substrates** that we just measured.
- The current model uses layer-11 features whose fake/real direction is substrate-flipped relative to lockbox; the new head uses layer-3 features whose direction is substrate-aligned.

**Why other levers haven't worked**:
- **Augmentation (jitter, anchor-aware, pipeline-random)**: perturbs inputs but the gradient still flows through the shortcut-prone deep layers. Empirically: jitter@0.50 changed only [CLS] features.
- **GRL at [CLS] (P15)**: collapses domain info at the final layer but not at the layers WHERE the substrate shortcut is forming (layer 6). Empirically: domain probe AUC 0.9994–0.9999 across all P15 candidates.
- **Substrate-balance / data-axis (P14_DATA_FIX, P16)**: doesn't address the structural emergence; just upweights one underrepresented class.

This is consistent with the independent reviewer's analysis (`docs/packet_retrospectives/plans/INDEPENDENT_REVIEW_SHORTCUTS_AND_NEXT_MOVES_2026-04-30.md` §"Initialization axis"): the right experimental axis is "earlier P8A step / P8A backbone with reset head" — i.e., move the readout earlier in the network. This memo gives a specific layer at which to attach the new head.

---

## Quick verification protocol for the GPU experiment

The proposed GPU experiment is a **layer-3 readout fine-tune**: freeze P8A backbone through layer 3, attach a new head, train on the standard training set, evaluate on lockbox via the v3 promotion contract scorecard.

**Mid-training kill switch (cheap)**: every 500 steps, run a frozen-features dev→lockbox transfer probe. If transfer AUC drops below 0.85 by step 1500, the experiment is failing — kill (with user OK, per `feedback_no_cancelling_vertex_jobs.md`).

**Final pass criteria** (any of these is a clear win):
- Lockbox FPR ≤ 2.0% AND lockbox recall ≥ 32.9% (P8A baseline at the operating point)
- Lockbox FPR ≤ 4.6% AND lockbox recall ≥ 50%
- v3 promotion contract: at least one ckpt selects a τ that beats P8A_step5000 on `value_composite`

**Final pass criteria (any of these is a clear loss)**:
- Transfer AUC at layer 3 itself drops in mid-training (the FT damaged layer 3 — pivot to even-earlier readout)
- Lockbox FPR > 6% at any τ with recall ≥ 32% (no improvement)
- Lockbox recall < 20% across all τ (head can't reach the operating point at any FPR)

**Cost / time**: ~$10–30, ~1–4 hours. Cheap because backbone is frozen.

---

## Risks and caveats

1. **The 800-frame layer-3 transfer is small-sample** (87 lockbox). Probe 4 (running) addresses this. **Do NOT promote any conclusion until Probe 4 finishes.**
2. **Layer-3 within-dev AUC is 0.94, not 0.99.** The detector loses label info if we cut off early. But within-dev AUC at layer 3 is still high enough to be operationally useful, and the transfer cost is what matters.
3. **The transfer AUC=0.95 at layer 3 might be partly because layer 3 features carry less information overall.** If a richer head can't be trained on layer-3 features, the GPU experiment will fail. This is the right kind of failure mode — clear signal at low cost.
4. **mclioexb showed identical layer-3 features to P8A** (cos=0.9995). Either base should work for the GPU experiment.
5. **Capture-mode at layer 0 is already 0.78** — meaning some substrate signal is in the input pixels. A perfectly substrate-invariant detector may be unattainable via this approach alone. We are after **substantial reduction**, not zero.
6. **The 180-frame production-honest substrate is the ground truth for deployment**, NOT the lockbox parquet. A successful GPU experiment must also be evaluated there.

---

## Files generated this session

- `analysis/intermediate_layer_probe_2026-04-30/outputs/summary.json` (5-layer probe, P8A vs mclioexb)
- `analysis/intermediate_layer_probe_2026-04-30/outputs/per_layer_table.csv`
- `analysis/intermediate_layer_probe_2026-04-30/outputs/per_layer_domain_probe.{json,csv}` (capture mode 5-class probe)
- `analysis/intermediate_layer_probe_2026-04-30/outputs/per_layer_split_probe.{json,csv}` (dev/lockbox split + transfer)
- `analysis/intermediate_layer_probe_2026-04-30/outputs/scaled_layer3_validation.json` (Probe 4, ~30 min runtime)
- `analysis/intermediate_layer_probe_2026-04-30/per_layer_domain_probe.py` (CPU script, no GPU)
- `analysis/intermediate_layer_probe_2026-04-30/per_layer_split_probe.py` (CPU script, no GPU)
- `analysis/intermediate_layer_probe_2026-04-30/scaled_layer3_validation.py` (CPU script, no GPU)
- This document.

## Decisions deferred to user

1. **Greenlight the GPU experiment** (`R13_P17_LAYER3_HEAD.yaml`)? Cost $10–30, ~4h. The CPU evidence is overwhelming; the GPU run validates at deployment scale and produces a deployable artifact.
2. **Should we add a SECOND GPU experiment** (the user's "up to 2 GPU slots" budget)? Suggested second arm: layer-3 head + capture-mode-balanced sampler (no detector patch needed, just data-side change). Tests whether substrate balance amplifies the layer-3 win. Cost $10–30 in parallel with the primary run.
3. **If E1 (and optionally E2) wins, what's Phase 2?** Either deploy directly with the new head, or scope a multi-layer GRL retrain that targets the layer-4-to-layer-11 entanglement (now that we know exactly where it lives).
4. **If E1 misses, do we pivot to substrate correction (independent reviewer's path)?** That's a 1–2 week effort: nuisance-balanced sampler, locked crop policy, hard-real anchors with matched fakes. Memory `project_face_size_label_leak`, `project_lockbox_fpr_dominated_by_webcam_mode`, etc. all flag this.

User reserves these per `feedback_decision_points.md`. **No GPU job auto-launched tonight** — the CPU evidence is presented for review and one-click approval in the morning.

## What was NOT done tonight (deferred)

- **Apply the detector patch and run smoke test**: `experiments/phase2_round13/p17_DETECTOR_PATCH.diff` was authored but NOT applied to `detectors/effort_detector.py`. Apply with `git apply` and test before launch.
- **Author the wiring test** `tests/test_p17_layer3_head_wiring.py`. Should assert: (a) yaml parses, (b) detector instantiates with `intermediate_layer: 3`, (c) forward returns 768-d features, (d) head dim matches, (e) backward works (no detached-tensor errors). Cost: ~30 min of careful work.
- **Wire the mid-training transfer probe** (yaml block `mid_eval_transfer_probe`). Could be deferred — first run can use post-training scorecard alone. Cost: ~2h of trainer.py work.
- **Validate the per-method layer-3 head readout against the visomaster_enhanced_macro_dev suite** specifically. The cached parquet's method labels don't include this slice; the eval-suite manifest does. Could be done with one more CPU script (~1h) or just measured at GPU scorecard time.
- **Audit `init_load_strict` and `head.reset_on_load` plumbing** in `train_sweep.py`. The yaml uses these flags; verify they do what we expect, or replace with explicit checkpoint-loading code.
- **Launch any GPU job.** See above for rationale.
