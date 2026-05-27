
# Shortcut Learning / Data Gap Research Summary

## Context-Compressed Version for LLM Handoff

Based on the current handoff and research direction.

## Core Diagnosis

The current failure mode is most likely **not lack of model capacity**. The frozen CLIP representation already contains strong fake-vs-real separability, including on difficult chronic false-positive cohorts. The problem is that fine-tuning pushes the classifier toward **shortcut axes** that are correlated with labels in training but unstable in deployment.

The central shortcut axes appear to be:

- sharpness / blur
    
- color cast
    
- face pixel area
    
- crop tightness
    
- capture mode / webcam pipeline
    
- source/provenance
    
- compression / codec artifacts
    
- identity × capture-mode interactions
    

The model is learning:  
**“this image looks like the fake side of my training set”**  
instead of:  
**“this image contains manipulation evidence.”**

This is especially dangerous because the training real data is mostly YouTube-derived and synthetically passed through a Teams-like pipeline, while the deployment/lockbox real data contains actual webcam/Teams captures. That creates a large real-side distribution gap. The model can separate dev-real from lockbox-real almost perfectly in frozen CLIP space, which means the evaluation and deployment real distributions are already far apart before fine-tuning.

## Main Research Conclusion

The most valuable direction is **data and validation redesign**, not another backbone or generic training trick.

The literature strongly supports this practical order:

1. **Build deployment-matched validation and lockbox sets.**
    
2. **Measure performance by nuisance slices, not only average metrics.**
    
3. **Collect or synthesize missing label × nuisance combinations.**
    
4. **Use simple robust baselines before heavy invariant-learning methods.**
    
5. **Only then test more complex causal, adversarial, or invariant objectives.**
    

The key point: shortcut learning happens when the model can use nuisance features that correlate with the label during training and are not punished by validation. If validation shares the same shortcut, the model looks good until it hits real deployment.

## What This Means for Your Case

Your data gap is not “we need more identities” or “we need more frames.”

It is more specific:

> The train set does not sufficiently cover the deployment joint distribution over image-quality and capture axes.

So “more data” only helps if it fills the missing combinations. Adding 2,000 more identities from the same data regime may do almost nothing. Adding fewer but well-targeted real Teams captures in under-covered regions of sharpness, color, face size, crop tightness, and capture mode may be much more valuable.

The right data question is not:

> How many identities do we have?

It is:

> For every important label × capture-mode × quality-region combination, do we have enough real and fake examples so the nuisance is not predictive of the label?

## Practical Data Principle

Every suspected shortcut axis should be decorrelated from the label.

For example, if fake examples are often softer, more compressed, medium face-size, or generated through one pipeline, then the model can learn those properties as fake predictors. To prevent that, the training set needs examples like:

- real frames with the same softness/compression/face size as fakes
    
- fake frames with the same sharpness/capture properties as reals
    
- same-source or near-matched real/fake pairs
    
- same identity across multiple capture modes
    
- same capture mode across multiple identities
    
- same manipulation method across multiple image-quality regions
    
- real webcam captures that occupy the same quality regions as fake samples
    

The goal is to make the shortcut useless.

## Highest-Priority Action Plan

### 1. Build a slice-aware lockbox

The lockbox should not just be “held out.” It should be structured.

Minimum slice metadata:

- identity
    
- source / provenance
    
- capture mode
    
- device / webcam type if available
    
- lighting condition
    
- face size
    
- crop tightness
    
- sharpness
    
- color cast
    
- compression level
    
- real/fake label
    
- fake method
    
- resolution
    
- whether the frame was directly captured or synthetically pipelined
    

Then evaluate:

- average fake recall
    
- average real FPR
    
- worst-slice FPR
    
- worst-slice fake recall
    
- chronic-identity FPR
    
- capture-mode FPR
    
- quality-bin FPR
    
- average-to-worst gap
    

A model should not pass just because average FPR is acceptable. If one capture mode or chronic identity explodes, that is a release blocker.

### 2. Run a joint-marginal coverage audit

For the five key image-property axes:

- sharpness
    
- color cast
    
- face pixel area
    
- crop tightness
    
- capture mode
    

Measure where lockbox real and chronic-FP frames sit relative to train density.

The important question:

> What percentage of lockbox-real and chronic-6 mass falls in regions where train density is very low?

If a large fraction of chronic-FP frames sit in low-density regions, then targeted data ingestion is structurally live.

Useful decision rule:

- If ≥30% of lockbox-real mass and ≥50% of chronic-FP mass sit in low-train-density regions, prioritize data collection.
    
- If not, the bigger issue may be training-recipe deconvolution rather than missing data support.
    

### 3. Collect targeted real Teams data

Do not collect generic real data.

Collect real Teams/webcam frames that specifically fill missing regions in the image-property space.

Examples:

- darker webcam captures
    
- softer / blurrier webcam captures
    
- low-resolution webcam captures
    
- phone-screen captures
    
- screen-recorded Teams calls
    
- different webcam models
    
- different lighting conditions
    
- different compression settings
    
- different face sizes
    
- loose and tight crops
    
- chronic-like identities or visual conditions
    

Each new data batch should be selected because it covers a known low-density deployment region.

### 4. Use matched or paired data where possible

The best structure is paired or near-paired data:

- same identity, real and fake
    
- same source video, real and fake
    
- same capture pipeline, real and fake
    
- same face size / compression / lighting, real and fake
    
- same Teams path, real and fake
    

This forces the model to look for manipulation evidence rather than source, quality, or pipeline artifacts.

For deepfake detection, same-source real/fake pairing is especially important because otherwise the model can learn generator/source differences instead of forgery features.

### 5. Treat augmentation as targeted, not generic

Generic augmentation helps but is not enough.

Useful augmentations should directly attack the shortcut axes:

- resolution-chain augmentation
    
- compression randomization
    
- sharpness / blur randomization
    
- color cast randomization
    
- face-scale jitter
    
- crop-tightness jitter
    
- simulated webcam degradation
    
- bandwidth degradation
    
- screen-recording degradation
    
- lighting shifts
    
- capture-mode simulation
    

The augmentation should make each nuisance axis less predictive of the label.

Your recent result with `resolution_chain_aug` is important because it reduced reliance on source resolution. That is exactly the kind of structural effect to look for.

## Training-Side Recommendations

### 1. Start with simple robust baselines

Before changing backbone or adding complex objectives, test:

- frozen encoder + linear probe
    
- last-layer retraining on balanced nuisance slices
    
- classifier retraining on curated lockbox-like validation data
    
- reweighting / resampling hard or bias-conflicting examples
    
- group-aware or pseudo-group-aware sampling
    
- worst-slice model selection
    

These are valuable because if they help, the representation already contains the right signal and the problem is mostly readout/data-selection.

### 2. Do not rely only on full fine-tuning

Full fine-tuning appears to rotate the decision boundary toward image-quality axes.

That means full FT may amplify shortcuts even when the frozen representation is good.

You should compare:

- frozen features
    
- last-layer-only training
    
- shallow adapter / LoRA
    
- partial fine-tuning
    
- full fine-tuning
    
- FT with nuisance penalties
    
- FT with slice-balanced sampling
    

Track whether each recipe moves the fake/real direction closer to or farther from IQ axes.

The key diagnostic is not only AUC or recall. It is whether the discriminative direction becomes more or less aligned with nuisance axes.

### 3. Use nuisance-aware losses carefully

Potentially useful:

- continuous-axis adversarial loss against sharpness, color, face size, crop tightness
    
- capture-mode adversarial loss
    
- group DRO over known chronic slices
    
- supervised contrastive loss with identity/capture anchors
    
- pair-rank loss on matched real/fake pairs
    
- prototype or anchor-aware regularization
    
- feature decorrelation from IQ axes
    

But the warning is important: these methods can easily overcorrect or remove useful signal if the nuisance is entangled with the real forgery cue. They should be evaluated by worst-slice metrics, not average metrics.

Your `anchor_aware` result is a good example: it fixed some chronic identities but worsened Roy_D. That means the mechanism has signal, but it needs better slice control.

## Evaluation Contract

The evaluation protocol should change from:

> “Does the model achieve good average recall/FPR?”

to:

> “Does the model maintain acceptable FPR and recall across deployment-relevant nuisance regions?”

Recommended reporting table for every checkpoint:

|Metric|Why it matters|
|---|---|
|Average fake recall|Main detection utility|
|Average real FPR|Product safety|
|Worst-identity FPR|Chronic false-positive risk|
|Worst-capture-mode FPR|Deployment robustness|
|Worst-quality-bin FPR|IQ shortcut risk|
|Fake recall by method|Method-specific blind spots|
|FPR by sharpness bin|Blur shortcut|
|FPR by face-size bin|Face-area leakage|
|FPR by crop-tightness bin|Crop shortcut|
|FPR by color-cast bin|Color shortcut|
|Average-to-worst gap|Shortcut risk summary|
|Dev vs lockbox delta|Generalization gap|
|Frozen vs FT direction alignment|Whether FT amplifies shortcut|

A model that improves average recall but widens the worst-slice gap should be treated as a regression.

## Interpretation of Current Findings

### Frozen CLIP result

Frozen CLIP already sees the fake signal. This means the bottleneck is probably not raw feature capacity.

### Fine-tuning result

Fine-tuning rotates the classifier toward image-quality features. This means the model is being trained into a shortcut.

### Chronic-6 result

A small number of identities/capture conditions concentrate most FPs. This suggests the failure is not random. It is structured around specific regions of deployment space.

### Dev-lockbox separability

Dev-real and lockbox-real are almost linearly separable. This means dev is not a reliable proxy for production real data.

### Head retraining failure

Past head retraining failed because it likely used the wrong balancing axis or still did not cover the deployment nuisance distribution properly. Head retraining should be re-tested only with a deliberately slice-balanced, deployment-like subset.

### More identities failure

More identities did not help because identity count was not the missing axis. The missing axis is deployment image-property coverage.

## Best Near-Term Experiment

Run a clean 3-arm experiment:

### Arm A — Current baseline

Current best checkpoint / recipe.

### Arm B — Targeted data ingestion

Add real Teams/webcam frames specifically selected to fill low-density regions on the five IQ/capture axes. Keep training recipe mostly fixed.

### Arm C — Training deconvolution

Keep data fixed, but add continuous nuisance-axis regularization, stronger resolution/capture augmentation, and slice-balanced sampling.

### Optional Arm D — Combined

Targeted data + training deconvolution.

The key comparison:

- If B helps strongly → data support is the main bottleneck.
    
- If C helps strongly → training recipe is the main bottleneck.
    
- If D helps but B/C alone do not → data and training are coupled.
    
- If none help → revisit representation/backbone or evaluation contract.
    

## Recommended Data Selection Strategy

Create a “gap sampler.”

For every candidate new real frame, compute:

- sharpness
    
- color statistics
    
- face size
    
- crop tightness
    
- capture mode
    
- compression/resolution metadata
    
- CLIP embedding distance to train/dev/lockbox
    
- whether it falls near chronic-FP regions
    

Prioritize samples that:

1. are close to lockbox/chronic-FP regions,
    
2. are underrepresented in training,
    
3. are real deployment-like captures,
    
4. break known label correlations,
    
5. are not simply more of the same YouTube-derived data.
    

This turns data collection into active coverage repair, not volume scaling.

## Recommended Model Selection Strategy

Do not select by dev-calibrated average metric alone.

Use a composite score:

- fake recall at fixed real FPR
    
- worst-slice FPR penalty
    
- chronic-FP penalty
    
- capture-mode FPR penalty
    
- robustness across quality bins
    
- lockbox/dev gap penalty
    
- calibration stability
    

A candidate should only win if it improves the product-relevant operating point without hiding regressions in minority slices.

## When to Consider New Backbones

Backbone changes are lower priority, but still worth testing as probes.

Useful candidates:

- DINOv2
    
- SigLIP / SigLIP-2
    
- MAE-style encoders
    
- forensic/frequency-aware encoders
    
- hybrid CLIP + frequency branch
    

But the test should be specific:

> Does the frozen backbone separate fake/real while remaining more orthogonal to IQ axes than CLIP?

If a backbone improves AUC but is equally or more aligned with image quality, it does not solve the real problem.

The useful backbone is one that preserves forgery signal in a direction less entangled with sharpness, compression, face size, and capture mode.

## Frequency-Domain Direction

There is a plausible secondary path: the true forgery signal may live in spectral or phase artifacts that CLIP does not preserve cleanly.

This suggests testing:

- FFT/DCT auxiliary branch
    
- frequency residual features
    
- high-frequency artifact head
    
- phase-consistency features
    
- pixel-space + frequency-space fusion
    
- spectral augmentations
    

But this should not replace the data-gap work. Frequency features can also learn shortcuts if the real/fake pipelines differ in compression or resolution.

So frequency methods must be evaluated on same-source or capture-matched data.

## Temporal Direction

Frame-level classification may be underusing deployment signal.

Potentially useful:

- temporal smoothing
    
- clip-level pooling
    
- identity-level aggregation
    
- frame-score consistency
    
- temporal artifact detection
    
- motion-based features
    
- per-call risk aggregation
    

This may reduce false positives if chronic-FP frames are isolated, but it may not solve systematic chronic-FP identities if all frames in a capture mode are biased.

Temporal aggregation should be tested as a product-layer improvement, not as a substitute for fixing training shortcuts.

## What Not To Do

Avoid these as primary strategies:

- adding generic identities without targeting deployment gaps
    
- rebalancing buckets without measuring nuisance coverage
    
- trusting dev metrics if dev-real is synthetic-pipeline YouTube
    
- selecting models by average score only
    
- assuming a bigger backbone fixes shortcut learning
    
- applying generic augmentation without measuring shortcut reduction
    
- using IRM/adversarial methods without reliable environment definitions
    
- treating chronic identities as just “bad examples” instead of evidence of structured missing support
    

## Final Strategic Recommendation

Treat this as a **data-support and shortcut-deconvolution problem**.

The most likely winning path is:

1. Build a precise nuisance-axis audit.
    
2. Identify under-covered lockbox/chronic regions.
    
3. Add targeted real Teams/webcam data in those regions.
    
4. Add matched fake examples where possible.
    
5. Train with nuisance-breaking augmentations and slice-aware sampling.
    
6. Select models by worst-slice deployment metrics.
    
7. Use frozen/last-layer baselines to verify whether the representation already contains the right signal.
    
8. Only escalate to new backbones, temporal models, or causal/invariant methods if the structured data fix does not close the gap.
    

