# Stability, Calibration, and Low-FP Operation for Live Teams Use

**Date:** April 15, 2026  
**Scope:** frame-based deepfake detection under live Microsoft Teams conditions, with strong emphasis on low false positives and score stability  
**Inputs used:** repo docs and experiment records, selective WandB checks, targeted literature review

## Bottom Line

The repo evidence and the broader literature point in the same direction:

1. The current problem is not just a classifier problem. It is also a **decision-system problem**.
2. For low false positives on live Teams, the highest-leverage near-term work is **target-domain calibration + short-window temporal decision logic + explicit uncertainty/disagreement handling**.
3. Generic training-time "stability regularization" is **not** the safest next bet here. This repo already found that broad stability-loss ideas were weak or inconsistent, and the literature supports using **narrower consistency constraints or temporal aggregation** instead of hoping one extra loss term fixes deployment.

## Repo-Specific Evidence To Respect

- Internal docs already identify a major calibration gap: the repo recorded an in-distribution EER threshold around `0.45` versus production/OOD thresholds around `0.77`, meaning raw scores are not portable across domains. See [R12_POST_LAUNCH_PLAN.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/R12_POST_LAUNCH_PLAN.md) and [PHASE2_SUMMARY_R8_R11.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/experiments/PHASE2_SUMMARY_R8_R11.md).
- The repo also documented strong frame-to-frame instability on near-identical frames, especially with the B/16 ArcFace setup. See [SCORE_INSTABILITY_ANALYSIS.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/experiments/phase2_round8/SCORE_INSTABILITY_ANALYSIS.md).
- The current Track A candidate improved target fake recall but still regressed real-Teams false positive rate versus `R12_G`. That means better fake sensitivity alone is not enough; the deployment objective is asymmetric and must be optimized explicitly. See [TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md).
- Selective WandB checks confirm that `R13_A_trackA` still trails `R12_G` on the repo's own OOD-composite axis (`0.9758` vs `0.9869`) even though it closes some enhanced-fake gaps. This reinforces that raw detection gains can come with stability/real-domain cost.

## What Can Be Fixed Without Changing The Backbone Class

### 1. Post-hoc calibration on the actual target domain

This is the cleanest underused lever in the repo.

- The repo already has an offline calibration path (`run_r8_calibration_fit.py`) and a documented plan to load a calibrator into inference, but that path does not appear to be the active production default. See [R12_POST_LAUNCH_PLAN.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/R12_POST_LAUNCH_PLAN.md).
- Classic temperature scaling is still the simplest post-hoc baseline, but under distribution shift it is often not enough. Surrogate/shift-aware calibration methods were proposed specifically because vanilla temperature scaling degrades when the test domain moves away from the calibration domain. See *Frustratingly Easy Uncertainty Estimation for Distribution Shift* in the reading list below.

Practical implication for this repo:

- Do not calibrate on generic holdout and assume it transfers.
- Calibrate on the **frozen Teams target-domain manifest**, especially real-Teams slices, because the business cost is concentrated there.
- Treat calibration as a first-class experiment artifact, not a final afterthought.

### 2. Short-window temporal aggregation at inference

The model is frame-based, but the deployment object is a live video stream. The literature repeatedly shows that temporal consistency is useful when the per-frame detector is noisy:

- TI2Net argues that temporal identity inconsistency helps generalization to unseen datasets and remains robust under compression and additive noise.
- Temporal methods in face anti-spoofing similarly report better cross-domain robustness by modeling motion or temporal structure rather than relying only on per-frame texture cues.

For this repo, the important point is not "replace the model with a temporal model now." It is:

- aggregate logits over a short trailing window;
- use **logit-space EMA or median**, not probability averaging only;
- add **hysteresis** so the system needs sustained evidence before flipping from real to fake.

This is much cheaper than sequence-model retraining and directly targets the user-reported failure mode: visually similar adjacent frames getting materially different scores.

### 3. Disagreement-based uncertainty without changing the backbone

The repo already knows the model is spatially fragile. That means disagreement across small input perturbations is informative.

A practical uncertainty signal can come from:

- two YOLO operating points;
- two small crop jitters;
- one clean crop plus one lightly shifted crop;
- or one raw frame plus one weak codec-like perturbation.

If the detector is stable, those should agree. If they disagree sharply, the system should be less willing to emit a confident "fake" decision. This is cheaper than deep ensembles and more targeted than generic Monte Carlo uncertainty.

### 4. Asymmetric thresholding and abstain/review bands

The user's objective is not symmetric accuracy. It is low FP on real Teams participants while retaining useful fake recall.

That means the right decision policy is likely:

- one threshold for internal model ranking and analysis;
- another threshold for deployment actioning;
- plus an **abstain / uncertain band** where the system does not escalate to "fake" on weak evidence.

Open-set deepfake work is relevant here because its central claim is that emerging or mismatched manipulations should not be collapsed back into "real." That is conceptually similar to keeping uncertain cases out of the hard real/fake decision path.

For this repo, "abstain" can be implemented without new training by combining:

- calibrated score,
- perturbation disagreement,
- short-window temporal stability,
- and possibly a minimum-run-length rule before surfacing "fake."

## What Requires Sequence Modeling Or Deployment Logic

### Sequence modeling

If you want the model itself, not just the deployment stack, to learn temporal coherence, then you are leaving the pure frame-based regime.

The literature-backed options are:

- identity-consistency modeling as in TI2Net;
- geometric/landmark temporal dynamics as in GAIN-style FAS systems;
- paired or triplet temporal training objectives.

These are plausible long-term research areas, but they are not the best immediate bet under time pressure because they require:

- new data packaging,
- different batching or sequence sampling,
- new inference plumbing,
- and a different evaluation contract.

### Explicit open-set or unknown-class handling

The open-set deepfake literature is promising for robustness and confidence handling, but it is not a quick patch. It requires a new training objective and usually a new deployment policy.

Still, it is conceptually important because it frames a useful deployment truth:

- many dangerous mistakes happen when the system is forced to output a binary answer for inputs outside its comfort zone.

That is highly relevant for live Teams, where lighting, compression, enhancement, and crop dynamics can move samples off the training manifold even when the participant is real.

## Low-FP Operating Recommendations

### Recommendation 1: Stop treating `0.5` as meaningful

For this project, the default `0.5` threshold is not semantically trustworthy. The repo has repeatedly shown that the score scale itself drifts by domain.

Use:

- calibrated score for decisioning;
- threshold selected on target-domain real slices;
- fake recall reported at the selected low-FP operating point, not only at a generic threshold.

### Recommendation 2: Optimize the decision stack for real-participant safety

If false positives matter most, the deployment stack should require more evidence to declare fake than to remain undecided.

A practical policy:

1. Calibrate per-frame logits on Teams dev data.
2. Aggregate over a short window.
3. Require both:
   - score above threshold, and
   - low perturbation disagreement, and
   - persistence across multiple consecutive frames.

This is a deployment policy change, not a backbone change.

### Recommendation 3: Separate model-selection metrics from deployment metrics

The repo already moved in this direction with OOD-composite checkpointing. That should continue. But for deployment, the top line should include:

- `Teams real FPR` at the chosen operating point,
- `Teams fake recall`,
- `visomaster_enhanced_macro` recall,
- and a stability metric such as short-window variance or flip rate.

If a checkpoint slightly improves fake recall but raises real-Teams FPR, it is not a deployment improvement.

### Recommendation 4: Use narrow consistency tests, not broad stability ideology

The repo's own history is a warning sign here:

- R9's stability setting was initially bugged.
- R9.5 revisited it.
- The broader experiment narrative in the registry is that generic stability regularization did not become a winning lane.

This does **not** prove consistency ideas are useless. It does mean the next attempt should be narrow and deployment-aligned:

- crop-jitter consistency,
- logit agreement under tiny spatial shifts,
- or target-domain-specific perturbations only.

That is a better bet than re-running a generic noise consistency loss and hoping the outcome changes.

## 3 High-Leverage Stability/Calibration Experiments

### 1. Frozen-checkpoint decision-policy sweep on Track C

Use current best checkpoints only. No retraining.

Compare:

- raw thresholding;
- target-domain temperature-scaled thresholding;
- target-domain calibrated + EMA;
- target-domain calibrated + EMA + hysteresis;
- target-domain calibrated + disagreement gate.

Primary metrics:

- `teams_real_all_dev` FPR,
- `teams_real_poor_quality_dev` FPR,
- `teams_fake_all_dev` recall,
- `visomaster_enhanced_macro_dev` recall,
- score flip rate on matched near-identical frame sequences.

Why this is high leverage:

- it is directly aligned to deployment;
- it isolates decision-policy benefit from model-training noise;
- it can produce meaningful FP reduction faster than another full training round.

### 2. Target-domain calibration study using the frozen manifest

Evaluate at least:

- vanilla temperature scaling,
- classwise threshold tuning,
- and one shift-aware post-hoc method if implementation cost is tolerable.

If you want the pragmatic baseline, do not overcomplicate this:

- start with temperature scaling plus threshold selection on Teams-real slices;
- only then test a shift-aware method if calibration still drifts badly.

Why this matters:

- the repo already knows the score space is misaligned;
- calibration is one of the few levers that can materially reduce FP without touching model weights.

### 3. A narrow training ablation: spatial consistency, not generic stability

If one new training-side stability experiment is run, it should be tightly scoped:

- take the best current Track A or R12-style config;
- keep the proposed spatial augmentation increase;
- add only a **small crop-jitter agreement loss** or dual-crop agreement head;
- do not bring back broad noise-heavy stability regularization as the main story.

Measure:

- OOD composite,
- Teams-real FPR,
- YOLO-setting sensitivity,
- frame-to-frame variance.

Why this is the right training ablation:

- it attacks a failure mode the repo directly observed;
- it matches the B/16 ViT patch-alignment sensitivity story better than generic noise consistency;
- it is less likely to wash out useful artifact cues than broad smoothing losses.

## Final Recommendation

Under time pressure, the best stability/low-FP strategy is:

1. keep the current frame backbone class;
2. add a serious target-domain calibration lane;
3. add deployment-side temporal aggregation and hysteresis;
4. use perturbation disagreement as an uncertainty gate;
5. only then spend training budget on a **narrow spatial-consistency** ablation.

If the project jumps straight back into generic stability-loss training, it is likely to spend time on a hypothesis that is both weakly supported by the repo's own history and less directly connected to the deployment objective than calibration plus decision logic.

## External Reading

- TI2Net: Temporal Identity Inconsistency Network for Deepfake Detection (WACV 2023): https://openaccess.thecvf.com/content/WACV2023/papers/Liu_TI2Net_Temporal_Identity_Inconsistency_Network_for_Deepfake_Detection_WACV_2023_paper.pdf
- A Closer Look at Geometric Temporal Dynamics for Face Anti-Spoofing (CVPRW 2023): https://openaccess.thecvf.com/content/CVPR2023W/Biometrics/papers/Chang_A_Closer_Look_at_Geometric_Temporal_Dynamics_for_Face_Anti-Spoofing_CVPRW_2023_paper.pdf
- Beyond Deepfake vs Real: Facial Deepfake Detection in the Open-Set Paradigm (arXiv 2025): https://arxiv.org/abs/2503.08055
- Frustratingly Easy Uncertainty Estimation for Distribution Shift (arXiv): https://arxiv.org/pdf/2106.03762
