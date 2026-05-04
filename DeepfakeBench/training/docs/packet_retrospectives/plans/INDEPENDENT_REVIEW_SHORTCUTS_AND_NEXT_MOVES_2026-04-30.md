# Independent Review: Shortcut Learning, Crop Regimes, Initialization Choice, And Next Moves

Date: 2026-04-30

Status: independent reviewer memo, not a canonical packet verdict

Intended reader: the final master reviewer deciding where to put engineering, data, and training resources next.

## Purpose

This memo summarizes and extends the conclusions discussed after the dashboard work shifted from visualization mechanics to substantive model behavior. It is intentionally written as an independent review: it should be considered alongside the canonical packet retrospectives, not treated as a replacement for them.

The central question is:

> What do the artifacts we already have suggest about the detector's shortcut behavior, and where should we spend the next units of time and money?

The review focuses on:

- Cross-model behavior on frame properties.
- Crop tightness and face-size effects.
- False positives versus fake recall.
- Historical agreement with `packet_retrospectives`.
- GRL and augmentation as possible remedies.
- Whether scratch, P8A-style warm-starting, or an intermediate initialization is the right base for future work.

## Executive Summary

My read is that the detector is not simply weak at "real vs fake." It has learned a useful fake-detection signal, but that signal is entangled with capture pipeline, crop tightness, face-size distribution, source bucket, and method family. Cross-model agreement on the same real-frame failure regimes makes this look intrinsic to the data/training substrate rather than accidental checkpoint noise.

The most important practical conclusion is:

> We should stop treating the next move as "find the right checkpoint recipe" and instead treat it as "build a substrate and evaluation protocol where shortcut variables are no longer predictive of the label."

That does not mean model work is irrelevant. It means model work should be tied to explicit shortcut-collapse criteria: domain-probe AUC, crop-sweep flip rate, per-capture-mode real FPR, per-method fake recall, and per-identity worst-case FPR. Trainer-side `value_composite` is useful as a directional signal, but the `mclioexb` scorecard shows it is not deployment-grade by itself.

The highest-priority recommendation is a small set of controlled probes before another broad training packet:

1. Treat crop recropping as a diagnostic and training-substrate question, not a deployment-time preprocessing fix for current P8A.
2. Run crop-tightness sweeps on both real and fake frames for every serious candidate, with P8A quality-gate results as the baseline warning case.
3. Run an initialization-axis comparison: raw CLIP, RLP6/RLP7-style base, earlier P8A step, P8A step5000, and P8A with a reset/retrained head.
4. Make any P16-style training packet include single-lever ablations. Do not stack GRL, jitter, anchor-aware, and pipeline-randomization without a clean ablation slot.

## Evidence Base Reviewed

This memo draws from the local artifacts and wiki surfaces already discussed in the session:

- `analysis/face_size_invariance_2026-04-30/outputs/*_invariance.csv`
- `analysis/quality_gate_2026-04-30/outputs/{phase1_summary.json,phase2_tightness_fpr_table.csv,phase2_normalize_result.json}`
- `analysis/deployment_honest_eval_2026-04-27/production_180_tags.parquet`
- `analysis/lockbox_tagging/full_tags_2026-04-27.parquet`
- `analysis/modern_lockbox_v2_2026-04-27/p8a_lockbox_subsets_fpr_recall.csv`
- `analysis/domain_confusion_probe_2026-04-30/outputs/probe_p8a_slot2_slot3/summary.json`
- `analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/triptych_coords_tsne.csv`
- `analysis/head_feature_decomposition_2026-04-30/outputs/summary.json`
- `analysis/intermediate_layer_probe_2026-04-30/outputs/summary.json`
- `analysis/scorecard_mclioexb_2026-04-30/*`
- `docs/packet_retrospectives/threads/processing_signature_shortcut.md`
- `docs/packet_retrospectives/threads/face_size_label_leak.md`
- `docs/packet_retrospectives/threads/webcam_fpr_dominance.md`
- `docs/packet_retrospectives/threads/eval_production_crop_tightness_gap.md`
- `docs/packet_retrospectives/threads/anti_shortcut_bundle_decomposition.md`
- `docs/packet_retrospectives/threads/jitter_winner_mechanism_unknown.md`
- `docs/packet_retrospectives/packets/P8A.md`
- `docs/packet_retrospectives/packets/P13.md`
- `docs/packet_retrospectives/packets/P14.md`
- `docs/packet_retrospectives/packets/P15.md`
- `docs/packet_retrospectives/STATE_2026-04-30.md`

## High-Confidence Findings

### 1. The Shared Failure Pattern Is About Frame Properties, Not One Model

The clearest dashboard-level finding is that real-frame false positives are concentrated by visual/capture regime.

On the 180 real production-honest frames, the risky groups were not subtle:

| Source group | Cross-model mean `prob_fake` at native crop | Any model says fake | All 4 say fake |
|---|---:|---:|---:|
| `roee-mac-laptop-false-flag-virtual-bg` | 0.520 | 90% | 23% |
| `dor-real-webcam-false-flag` | 0.416 | 83% | 0% |
| `dor-real-webcam-false-flag-no-virtual-bg` | 0.378 | 80% | 0% |
| `dor-real-laptop-correct-no-virtual-bg-yellowish` | 0.135 | 10% | 0% |
| `dor-real-laptop-correct-no-virtual-bg-whiteish` | 0.041 | 0% | 0% |
| `roee-real-windows-laptop-correct` | 0.012 | 0% | 0% |

This is not a pattern I would attribute to random checkpoint variance. Multiple recipes are reading the same source/capture/crop conditions as evidence for "fake." That points to the data geometry.

The historical wiki already had this diagnosis from the same-person camera swap. The dashboard now strengthens it because the effect repeats across models and across more frame properties.

### 2. Crop Policy Is Load-Bearing, But "Tighter Is Better" Is Too Simple

The face-size invariance sweep provides strong evidence that crop policy changes model behavior. On the 180 real production-honest frames, making the generated variant tighter at `t=1.5` reduced fake predictions on real frames at the 0.5 threshold:

On the 180 real production frames:

| Model | Native crop fake-rate on real frames | Tight `1.5` fake-rate |
|---|---:|---:|
| P8A | 30.0% | 6.7% |
| Slot 2 GRL (`w5tky6ss`) | 10.0% | 0.6% |
| Slot 3 Jitter (`mclioexb`) | 35.0% | 3.9% |
| DATA_FIX (`xan4dfto`) | 17.2% | 3.3% |

However, the later `quality_gate_2026-04-30` P8A recrop-rescore result changes the recommendation. The scripts use this convention:

```text
t > 1.0 = tighter center crop
t = 1.0 = native crop
t < 1.0 = looser crop via shrink + edge padding
```

Under that convention, the P8A fixed-tightness sweep at tau=0.9741 on the full lockbox gives:

| Tightness | Direction | Lockbox FPR | Lockbox recall | FPR delta | Recall delta |
|---:|---|---:|---:|---:|---:|
| 0.70 | looser | 7.5% | 15.8% | +3.0pp | -17.2pp |
| 0.85 | looser | 3.6% | 16.5% | -1.0pp | -16.5pp |
| 1.00 | native | 4.6% | 32.9% | 0.0pp | 0.0pp |
| 1.20 | tighter | 3.1% | 23.1% | -1.5pp | -9.9pp |
| 1.50 | tighter | 3.6% | 30.1% | -1.0pp | -2.8pp |

The other agent's warning is therefore directionally important but one detail must be corrected: `t=0.70` and `t=0.85` are looser, not tighter, in the local scripts. The stronger conclusion is not "tighter is worst." It is:

> Current P8A is highly crop-policy dependent, and changing crops at inference time does not give a clean operating-point win.

The normalize-to-target experiment is the clearest deployment warning. Selectively tightening loose frames to `face_area_ratio=0.45` changed 39.6% of lockbox frames and moved lockbox FPR from 4.6% to 8.0%, with recall unchanged at 32.9%. On dev, the same normalization increased recall from 66.3% to 80.3%, but also increased FPR from 6.3% to 7.6%. That is useful diagnostic evidence, not a deployment-safe preprocessing rule.

The consolidated interpretation is:

> The model is using non-face or crop-context evidence on real frames. Removing more context suppresses a large fraction of false positives.

But for current P8A, some surrounding context also appears to act as real-class evidence on parts of lockbox/dev. Stripping or perturbing that context can reduce some false positives, increase others, and materially hurt fake recall depending on split, threshold, and crop operation.

This is consistent with the historical `eval_production_crop_tightness_gap` thread in a more precise way: eval frames carry a different crop/background distribution than production, so changing crop policy changes the evidence surface. That does not imply current P8A should be recropped at deployment. It implies training and evaluation need to own a stable, deliberate crop policy.

### 3. The Open Question Is Training-Time Crop Policy, Not P8A Inference-Time Recropping

The quality-gate artifacts now provide a clean P8A fake+real intervention on lockbox and dev. That is enough to reject a naive deployment-time rule like "tighten current P8A inputs."

What remains open is whether a model trained and evaluated under a stable crop policy can do better. The best observational proxies are encouraging but not decisive:

| Slice | Real FPR | Fake recall |
|---|---:|---:|
| all dev frames | 19.1% | 97.6% |
| large face pixel area, top quartile | 1.0% | 97.2% |
| large face area ratio, top quartile | 2.9% | 97.8% |
| large clean face | 0.43% | 98.5% |
| normal-photo + large clean face | 0.0% | 97.9% |
| phone-screen + large clean face | 1.4% | 100.0% |

This suggests that the "large/tight/clean face" regime can be good for both FPR and recall. However, this is observational and confounded by source/method. It is not the same as applying a tight-crop intervention to fake frames and measuring recall deltas.

The other important quality-gate result is the face-pixel-area filter:

| Gate | Coverage | Lockbox FPR | Lockbox recall |
|---|---:|---:|---:|
| none | 100.0% | 4.6% | 32.9% |
| `face_pixel_area >= 20k` | 88.1% | 0.6% | 32.9% |

That is a strong eval-hygiene result, but it is not the same as a model fix. It drops a set of small-face real frames that current P8A often falsely flags while preserving the lockbox fake set in this artifact. This should be considered as a reporting/gating policy, with explicit coverage accounting, not as evidence that the detector has learned a cleaner representation.

The missing probe is now narrower:

```text
same train/eval crop policy
same nuisance-balanced data
same checkpoint initialization
train with that policy, do not only perturb inputs at inference
report: real FPR, fake recall, AUC, recall delta by fake method, source/capture probe AUC
```

Until that exists, I would state:

> Tight crops are not a safe deployment-time fix for current P8A. Crop policy is a substrate variable that must be trained and evaluated consistently.

### 4. Domain/Capture Information Remains Highly Separable

The domain-confusion probe is one of the most important negative results. It shows that the final feature space still carries domain/capture information almost perfectly:

| Checkpoint | Domain macro AUC |
|---|---:|
| P8A | 0.9999 |
| Slot 2 GRL | 0.9994 |
| Slot 3 Jitter | 0.9969 |

This means the GRL run did not achieve the canonical DANN objective at the final `[CLS]` representation level. The result does not make GRL useless: Slot 2 preserved cross-method TPR and incidentally reduced crop flip rate. But it does mean the mechanism is not "domain confusion succeeded" unless future probes at other layers or under different domain labels show otherwise.

For future GRL experiments, the closure criterion should not be trainer metric alone. It should include:

```text
domain-probe AUC drops materially
fake recall by method does not collapse
real FPR by capture mode improves
crop-sweep flip rate improves
```

### 5. Jitter@0.50 Won A Trainer Metric But Did Not Solve The Intended Shortcut

`mclioexb` was the trainer-side leader, but the direct measurement refuted the original mechanism story.

Face-size flip rate:

| Checkpoint | Flip rate | Median `|delta prob_fake|` |
|---|---:|---:|
| P8A | 39.4% | 0.128 |
| `mclioexb` jitter@0.50 | 43.9% | 0.303 |
| `w5tky6ss` GRL | 22.8% | 0.124 |
| `xan4dfto` DATA_FIX | 22.8% | 0.143 |

The intended face-size lever made crop sensitivity worse on the production-honest substrate. The promotion scorecard then showed that the trainer-side win did not translate to deployment-grade promotion.

The useful lesson is not "jitter is bad." The useful lesson is:

> A training lever can improve a directional trainer metric while failing its design-intent diagnostic and failing the promotion operating point.

That should become a standard rule: every new anti-shortcut lever needs a design-intent probe.

### 6. Stacked Anti-Shortcut Bundles Are Risky Without Ablations

The P14/P15 evidence strongly supports the new bundle-decomposition discipline.

The anti-shortcut bundle was:

```text
anchor_aware
pipeline_randomization
face_scale_jitter@0.25
```

The sister isolated run was:

```text
face_scale_jitter@0.50 only
anchor_aware disabled
pipeline_randomization disabled
```

The isolated run beat the bundle by 5.7x on `value_composite`. The GRL run was also layered on top of the same potentially harmful bundle, so its result is hard to interpret as a clean GRL result.

I agree with the wiki's operational rule:

> Any future multi-lever anti-shortcut packet should include a single strongest-lever ablation slot with the same init, data, LR, and schedule.

Without this, failures will keep being misattributed to "the idea does not work" instead of "the bundle composition was wrong."

## Historical Alignment With `packet_retrospectives`

The current read is broadly aligned with the historical wiki.

| Historical wiki finding | Current review position |
|---|---|
| Same-person camera swap proves processing-signature shortcut. | Confirmed and strengthened by cross-model agreement on risky source groups. |
| Face-pixel-area/crop tightness is a label leak. | Confirmed. Current P8A is crop-policy brittle; naive inference-time recropping is rejected by quality-gate evidence. Training-time crop normalization remains open. |
| Webcam-mode lockbox tail dominates headline FPR. | Confirmed. Capture-mode/reporting hygiene remains essential. |
| Eval and production crop tightness may differ. | Confirmed as strategically central. This is probably a high-value next measurement. |
| P8A breaks some anchor ceiling but regresses fake recall. | Still true. P8A is useful but not clean. |
| P13 from-scratch weakened shortcut but lost cross-domain capability. | Still the best argument against naive scratch. |
| P15 GRL is structurally different and worth trying. | Yes, but static GRL did not collapse final domain geometry. Future GRL needs stricter diagnostics. |
| Value composite is directional, not deployment-grade. | Strongly confirmed by `mclioexb` scorecard fail. |

The only material sharpening is this:

> The current evidence makes "design intent" and "metric movement" separable. Jitter and GRL both moved some metrics, but neither achieved its advertised mechanism in the most direct probes.

That should make the next experimental design more diagnostic and less narrative-driven.

## Initialization Axis: Scratch, P8A, And Intermediate Bases

The recurring question is whether the model is predisposed to shortcut learning because of earlier fine-tunes.

My answer is:

> Possibly yes, but not enough to make scratch the default. The stronger evidence is that the shortcut is data/substrate-bound. Warm-starting preserves useful fake-detection structure and harmful shortcut geometry at the same time.

### What The Historical Evidence Says

The team explicitly worried about FT inheritance early. RLP1 included a scratch hedge because a pure FT family might inherit old assumptions, while an all-scratch packet would be too expensive and slow.

P8A/P8B was the most direct test:

- P8A: warm-start from RLP6_04, unfreeze more CLIP modules, improves real FPR and anchor shortcut.
- P8B: scratch from plain CLIP-DataComp-XL, no R12/R13 inheritance, worse than the baseline on the relevant shortcut readout.

P13 then tested from-scratch with anti-shortcut interventions actually live. It moved the shortcut in the right direction but collapsed cross-domain fake detection and modern_v2 FPR. That is the strongest evidence that scratch is not currently the answer by itself.

### My Interpretation

P8A is not a neutral base. It is not shortcut-clean. It almost certainly carries a learned capture/crop/source geometry.

However, scratch training on the same confounded data is not a clean reset. It can relearn shortcuts while losing the useful prior that P8A inherited from CLIP and earlier training.

So the binary "scratch vs P8A" is underspecified. The right experimental axis is:

```text
A. raw CLIP + new head
B. RLP6_04 / RLP7-style base
C. earlier P8A step
D. P8A step5000
E. P8A backbone with reset or retrained head
F. P8A features with new calibration-aware head
```

Each base should be compared under the same data, crop policy, scorecard, and probes.

The selection criterion should be:

```text
maximize label separability
minimize domain/source/capture separability
minimize crop-sweep flip rate
preserve fake recall by method
control real FPR by capture mode and identity
```

That would tell us whether P8A is too contaminated, whether earlier P8A is better, or whether the issue is mostly the head/calibration layer.

## Recommendations

### Priority 0: Fix The Measurement Surface

These are low-to-medium cost and prevent wasted training spend.

#### 0.1. Treat Recropping As Diagnostic, Not A P8A Deployment Fix

The crop-tightness mismatch is structurally upstream of many conclusions, but `quality_gate_2026-04-30` shows that current P8A does not become better by simply recropping inputs at inference time. On lockbox, normalize-to-target `face_area_ratio=0.45` worsened FPR from 4.6% to 8.0% with no recall gain.

Future recrop experiments should be framed as:

```text
diagnose how crop policy moves the score distribution
select a train/eval crop policy
retrain or recalibrate under that policy
then measure deployment performance
```

Do not ship "tighten P8A inputs" as a shortcut fix.

#### 0.2. Repeat Fake+Real Crop Sweeps For Candidate Checkpoints

P8A now has a fake+real crop intervention baseline. Every serious successor should be evaluated the same way, because a training lever can improve one metric while making crop sensitivity worse.

Minimum version:

```text
200 real + 200 fake
stratified by method, capture mode, face_area_bucket
tightness: 0.85, 1.0, 1.2, 1.5
models: P8A, Slot2_GRL, Slot3_Jitter, maybe RLP6_04
thresholds: tau=0.5 and calibrated operating tau
```

This resolves whether an intervention made the model less crop-dependent or merely moved one operating point.

#### 0.3. Make Contract v3 And v2/Identity Reporting Non-Optional

Future scorecards should always report:

```text
baseline lockbox FPR
modern_v2 FPR
per-identity max FPR
fake recall by method
selected threshold provenance
whether recall floor was active
```

The `mclioexb` scorecard is useful, but it did not exercise the v3 recall-floor path because the launcher omitted the flag. This kind of reporting ambiguity should be eliminated.

### Priority 1: Correct The Data Substrate

#### 1.1. Normalize Crop And Source-Resolution Policy In Training/Evaluation

Apply a production-like crop policy and a source-resolution floor. The model should not score 99x110 eval artifacts as if they represented deployment.

Suggested baseline:

```text
min(width, height) >= 200
face present
not pose-extreme
stable documented crop policy selected by probe
stable padding/resizing policy
```

This is not just evaluation hygiene. It changes what visual evidence the model can exploit. Because current P8A worsens under some inference-time recropping regimes, the crop policy must be part of the training/evaluation substrate, not an after-the-fact preprocessing patch.

#### 1.2. Balance Nuisance Axes

Sampling should make it difficult to predict label from:

```text
capture_mode
source_bucket
face_area_bucket
crop_tightness
identity/session
method family
codec/compression regime
virtual-background / screen-like context
```

A practical sampler target:

```text
label x method_family x capture_mode x face_area_bucket x source_bucket
```

If a full factorial is too sparse, at least enforce marginal balance on face-area bucket and capture mode within real/fake and within major fake families.

#### 1.3. Use Hard Real Anchors With Matched Fake Positives

The hard real clusters should be in training or at least in an anchor validation set:

```text
webcam real
virtual-background real
phone/screen-like real
low-sharpness real
same identity across capture devices
```

But each hard-real regime needs matched fake positives under similar capture/crop conditions. Otherwise the model learns "webcam real" or "virtual background real" instead of artifact content.

### Priority 2: Model-Side Experiments

#### 2.1. Controlled Initialization Packet

Run a small initialization-axis packet before committing to more P8A-derived tuning.

Proposed arms:

| Arm | Purpose |
|---|---|
| raw CLIP + new head | checks whether scratch can work under corrected substrate |
| RLP6_04 or RLP7_05 base | older warm-start, potentially less P8A-specific contamination |
| earlier P8A step | tests whether P8A_step5000 over-specialized |
| P8A_step5000 | current reference |
| P8A backbone + reset head | separates feature contamination from head/calibration contamination |

This should be a diagnostic packet, not a promotion packet.

#### 2.2. GRL, But With A Better Protocol

GRL belongs in the solution space, but the first live run did not collapse final-domain separability.

Future GRL should include:

```text
lambda ramp from 0 to target
balanced domain batches
domain labels audited against source confounds
layer-specific domain probes
linear and maybe shallow-MLP domain probes
closure criterion on domain AUC, not value_composite only
```

The goal is not "add GRL." The goal is "make the representation less predictive of capture/source while preserving fake recall."

#### 2.3. Augmentations, But Isolated

Use augmentations, but treat each as a hypothesis with a diagnostic:

| Augmentation | Intended shortcut | Required diagnostic |
|---|---|---|
| crop/scale jitter | face-size/crop leak | crop-sweep flip rate drops |
| codec/JPEG/chroma | camera/ISP signature | source/domain probe drops |
| background/context masking | non-face context leak | tight-vs-loose crop delta shrinks |
| lighting/blur/noise | quality shortcuts | FPR by quality slice improves |

Do not stack multiple new augmentations unless one arm isolates the strongest single lever.

#### 2.4. Head-Side And Calibration-Aware Levers

The `mclioexb` scorecard suggests the recall-FPR frontier is too shallow at the operating point. That may be a head/calibration problem, not only a backbone problem.

Candidate directions:

```text
reset/retrain head on corrected substrate
calibration-aware loss
FPR-constrained or hard-negative-aware objective
separate artifact head from domain/context head
two-head model where domain head is adversarial/diagnostic, not label evidence
```

This should be informed by the head-vs-feature decomposition. If a new head over stable features moves the frontier, full retraining is not the first lever.

## What I Would Not Do Next

### Do Not Launch Another Broad Bundle Without Ablations

The P14/P15 evidence says bundle composition can be net-negative. A future packet that stacks jitter, GRL, anchor-aware, codec aug, and data-fix without isolating the components will likely be hard to interpret even if it works.

### Do Not Treat Scratch As A Clean Reset By Default

Scratch training can relearn shortcuts from the same data while losing useful cross-domain priors. P13 is the cautionary example.

Scratch is worth testing only under a corrected substrate and with strong controls. It is not the next default.

### Do Not Treat P8A As Clean

P8A is useful but shortcut-contaminated. It should remain a reference and possible base, but not a trusted neutral origin.

### Do Not Use Eval-Bucket Training Without Group-Split Discipline

If a data-fix uses sources close to the eval substrate, the split must be by identity, capture session, and source bucket. Frame-level random splits would make the next result uninterpretable.

### Do Not Optimize `value_composite` Alone

The `mclioexb` result is the current warning case. It won trainer-side ranking and failed deployment scorecard. Any future "leader" needs a scorecard and design-intent probes.

## Suggested Next Decision Tree

### Step 1: Measurement Closure

Run:

```text
crop-policy rescore baseline
fake+real crop sweep on candidate checkpoints
contract v3 representative scorecard
```

If crop recropping materially shifts scores, prioritize substrate/crop correction before broad model training.

If current P8A worsens under recropping, do not deploy recropping; use the result to design a training-time crop policy.

If a retrained model under a stable crop policy improves both FPR and recall, then that policy becomes a default candidate.

### Step 2: Initialization Diagnostic

Run a small same-data initialization packet.

If raw CLIP/scratch remains weak, drop scratch as a near-term path.

If earlier P8A or RLP7 base has lower shortcut metrics with tolerable recall, use that as P16 base.

If P8A features are good but the head is bad, prioritize head-side retraining/calibration-aware loss.

### Step 3: Model Intervention

Only after Steps 1 and 2:

```text
candidate = best initialization base
+ stable crop/data policy selected by probe
+ strongest isolated augmentation
+ optional ramped GRL
```

With at least one single-lever ablation.

## Proposed P16 Shape

If forced to propose one concrete P16 direction today, I would not run the old bundle. I would run a small, controlled packet:

```text
Base: best of P8A_step5000 vs earlier-P8A/RLP7 candidate if available
Data: stable crop policy selected by probe, nuisance-balanced sampler
Arm 1: base + corrected data only
Arm 2: Arm 1 + face_scale_jitter@0.50
Arm 3: Arm 1 + ramped GRL
Arm 4: Arm 1 + face_scale_jitter@0.50 + ramped GRL
```

Do not include anchor-aware or pipeline-randomization unless there is a separate arm isolating their contribution.

Required readout:

```text
promotion scorecard with v3 policy
modern_v2 FPR and baseline FPR
per-identity max FPR
fake recall by method
crop-sweep real FPR and fake recall
domain-probe AUC
source-bucket probe
cross-model/shared-failure gallery
```

## Risks And Caveats

1. The cross-model 180-frame crop sweep is real-frame heavy; the fake+real quality-gate sweep is P8A-only.
2. The `quality_gate_2026-04-30` crop intervention is P8A-specific and uses tau=0.9741; do not transfer it uncritically to other checkpoints.
3. The dev large-clean-face recall numbers are observational and may be confounded by source/method.
4. The domain-confusion probe is final `[CLS]` and mostly linear. GRL effects might exist at intermediate layers or in norms, although the gross evidence still says no final manifold collapse.
5. Some reported metrics use eval substrate with known crop/source-quality caveats.
6. `full_tags.prob_fake` should be treated carefully because some lockbox score columns do not align with P8A summary artifacts unless joined to the proper report score.
7. `value_composite` is a useful directional metric but repeatedly misaligned with deployment-grade scorecards.

## Final Reviewer Position

My final position is:

> The project should spend less effort on broad checkpoint recipe search and more effort on substrate correction, nuisance-axis balancing, and controlled causal probes.

The current model family has real signal. It is not a model-of-nothing. But that signal is interwoven with shortcuts. P8A-style warm starts preserve useful signal and shortcut geometry. Scratch reduces inheritance but loses too much useful prior under the current data. Augmentation and GRL are both plausible, but both need explicit design-intent diagnostics and clean ablations.

The next successful packet is likely to come from combining:

```text
production-faithful or deliberately chosen crop/eval substrate
+ nuisance-balanced data
+ carefully selected initialization
+ one or two isolated anti-shortcut levers
+ scorecard/reporting discipline
```

The key decision for the master reviewer is not "GRL or jitter" and not "scratch or P8A" in isolation. The key decision is whether the next spend is allowed to be diagnostic and substrate-correcting, or whether it repeats the older pattern of recipe tuning against a confounded measurement surface.

My recommendation is to make the next spend diagnostic first. The cost is lower, the information value is higher, and it prevents another round where a model moves a metric without moving the mechanism we actually care about.
