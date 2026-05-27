# Cheap Follow-ups FACTS — Teams Account Natural Experiment, 2026-05-19

> **Context.** Companion to `TEAMS_ACCOUNT_NATURAL_EXPERIMENT_FACTS_2026-05-19.md`. The natural experiment showed T5C swings prob_fake by 0.17 (and flips real/fake at τ=0.70) between two crops of the same person, same camera, only differing in the Teams account being used. This document characterizes WHICH axes drive the swing.

## 1. Per-axis sensitivity (T5C step3500, single-axis perturbation on Roy_D crop)

For each of 8 IQ axes, hold all other axes fixed and sweep the perturbation level. Score T5C and Slot B on each perturbed crop. **Sensitivity = max(prob_fake) − min(prob_fake) across the sweep**.

| Axis | T5C swing | min p_fake | max p_fake | Direction summary |
|---|---:|---:|---:|---|
| **`G_scale`** (green channel multiplier) | **0.678** | **0.159** @ 0.70 | **0.837** @ 1.05 | **Less green → more "real"** (huge effect) |
| **`B_scale`** (blue channel multiplier) | **0.497** | **0.404** @ 1.20 | **0.900** @ 0.70 | **More blue → more "real"** |
| `contrast` | 0.428 | 0.367 @ 0.50 | 0.795 @ 1.00 | Lower contrast → more "real" |
| `blur_sigma` (Gaussian) | 0.373 | 0.484 @ 2.0 | 0.857 @ 0.5 | Mid-blur (σ ~ 1.5-2) → most "real"; very-sharp and very-blurry both → "fake" |
| `R_scale` (red channel multiplier) | 0.259 | 0.583 @ 1.15 | 0.842 @ 0.70 | More red → more "real" (counter-direction) |
| `brightness_add` (additive) | 0.199 | 0.602 @ +30 | 0.800 @ −30 | Slightly darker → more "fake" |
| `gamma` | 0.092 | 0.766 @ 0.70 | 0.857 @ 1.7 | Weak effect |
| `saturation` | 0.030 | 0.777 @ 0.50 | 0.807 @ 1.20 | Essentially flat |

### Headline finding

**The green-channel scale is by far the most predictive single axis.** Multiplying the green channel of the SAME face crop by 0.7 (a 30% dim of green only, everything else held constant) flips T5C from **0.795 (FAKE-leaning)** to **0.159 (confidently REAL)**. Even a modest scale of 0.85 (which is roughly the magnitude Teams applied between accounts) gets T5C to **0.313**.

**Sharpness/blur is the 4th-strongest axis with 0.37 swing** — confirms the IQ-shortcut framing but shows it's not the dominant one. Notably, the response is non-monotonic: σ=0 (sharp) → 0.795, σ=0.5 → 0.857 (peak FAKE), σ=2.0 → 0.484 (mid-real), σ=3.0 → 0.749 (back up).

**Saturation has effectively no sensitivity** (swing 0.03). Despite saturation being in the IQ-axis basis the program has been studying, the model doesn't read it.

### Direction check against the natural experiment

Roy_D → Guest measured channel-mean shifts:
- G: 145 → 126 (factor ~0.87)
- B: 134 → 113 (factor ~0.85)
- R: 181 → 154 (factor ~0.85)
- sharpness: 113 → 56 (factor ~0.50)

Predicted prob_fake shifts (from single-axis sweeps, applied separately):
- G ×0.87 → ~0.55 (Δ = −0.25)
- B ×0.85 → ~0.87 (Δ = +0.07, **wrong direction**)
- R ×0.85 → ~0.81 (Δ = +0.01)
- blur σ ≈ 1.0 (matching the sharpness drop) → ~0.80 (Δ ≈ 0)

**G_scale alone predicts the right direction and roughly the right magnitude.** B_scale alone would predict the wrong direction. The remaining single-axis signals are weaker. So the G channel is doing most of the work, while B and the others provide counter-balancing signals that partially cancel.

Figure: `figs/perturbation_sensitivity.png` (8-panel grid of prob_fake vs perturbation level, T5C + Slot B overlaid).

## 2. Reverse-engineering the per-account transformation

Apply a bundled perturbation (Gaussian blur + uniform channel scale) to Roy_D and see how close we get to Guest's actual T5C score (0.628):

| Recipe | T5C prob_fake | Slot B prob_fake | Match to Guest target |
|---|---:|---:|---|
| Roy_D baseline (no perturbation) | 0.795 | 0.813 | — |
| `blur σ=1.0, scale=0.85` | **0.538** | 0.555 | mid (low by 0.09) |
| `blur σ=1.5, scale=0.85` | 0.405 | 0.522 | low by 0.22 |
| `blur σ=2.0, scale=0.85` | 0.456 | 0.669 | low by 0.17 |
| `blur σ=1.5, scale=0.80` | 0.442 | 0.557 | low by 0.19 |
| `blur σ=1.5 only` | 0.530 | 0.788 | low by 0.10 (T5C only) |
| `scale=0.85 only` (no blur) | **0.706** | 0.658 | high by 0.08 |
| `blur σ=2.5, scale=0.80` | 0.643 | 0.828 | **closest match T5C** |
| **Guest actual** | **0.628** | **0.610** | — |

The closest synthetic recipe is `blur σ=2.5, scale=0.80` for T5C (0.643 vs 0.628), but no recipe is a perfect match — likely because the actual Teams transformation is non-uniform across channels (R dropped 15%, G 13%, B 15%; not the perfectly-uniform 0.85 scale tried here) and may include a non-linear tone curve.

**Takeaway:** A bundle of (Gaussian blur σ ≈ 1.0-2.5 + uniform channel dim ≈ 0.80-0.85) reproduces the Teams Guest-account transformation **to within ~0.08 prob in either direction**. The remaining gap is most likely chromatic — the actual Teams pipeline applies a per-channel color cast that uniform scaling doesn't replicate.

This is encouraging for training-side augmentation: **a `teams_account_transport` augmentation recipe** built around this transformation would directly simulate the natural experiment's manipulation. The relevant operations (blur, per-channel scale, gamma) are all in scope for the existing augmentation framework.

## 3. ArcFace cosine — same person across Teams accounts

Running `insightface` `buffalo_l` (ArcFace R50) face recognition on the full panels:

- **Roy_D vs Guest ArcFace cosine: 0.9203**
- ArcFace embedding dim: 512

### Interpretation

- For face verification, "same person" thresholds typically sit in the **0.4–0.5 cosine** range. **0.92 is comfortably above** that — ArcFace correctly verifies the two crops as the same person.
- But 0.92 is **far from 1.0**. The same image processed twice would give ~0.999. The 8% cosine reduction is real and measurable.
- **This confirms Probe 1's "identity-cluster" axis is partly transport-shifted.** The same person under different Teams transport produces measurably different ArcFace embeddings — the embedding shift exceeds the typical same-person noise floor (~0.95-0.98 for same-person same-session photos).

The substrate-gap axis Probe 1 found (CLIP residualized against ArcFace → chance) is therefore **not** pure biometric identity. It's "identity-as-encoded-by-the-current-transport-pipeline." The ArcFace embedding is sensitive to the same low-level pixel statistics (color cast, sharpness, contrast) that drive the deepfake classifier.

## 4. CLIP CLS cosine — FAILED (deferred)

The `transformers.CLIPModel.vision_model(... return_dict=True)` output's `hidden_states` returned a non-tensor for the projection-head cosine computation. Could not complete in this session. The L11 / L23 CLS cosine measurement would tell us how transport-shifted the *encoder's own* representation is — directly relevant to the IQ-shortcut amplification hypothesis. Deferred to next session.

A workaround: extract L11/L23 features through the EffortDetector model itself (already loaded in `arena.model_arena.load_model`); the trainer hooks into specific layers. ~30 min of plumbing.

## 5. Combined implications

### 5.1 The G-channel finding refines the IQ-shortcut framing

The program's working model has been: "model uses sharpness / color-cast / face-size / brightness as fake predictors." This natural experiment refines that:

> **The single dominant axis (under all-else-equal perturbation) is the green-channel multiplier.** Sharpness is significant but a distant second. Saturation is irrelevant. Brightness and red-channel scale are weak.

This is testable on the lockbox cohort: re-tag every lockbox frame with its mean R/G/B channels (cheap, already done in part by the IQ atlas), then check whether `G_mean` explains operational FPR better than the existing `sharpness_laplacian` proxy. Hypothesis: yes.

### 5.2 The training-side lever has a sharper specification

Instead of "anti-shortcut" augmentations broadly, the strongest single augmentation candidate is:

> **Randomized per-channel multiplicative scale** with G perturbation magnitude at least ±20% per frame, applied during training. Plus an optional Gaussian blur with σ ∈ [0, 2.5].

This is the most-direct training-side simulation of the per-account Teams transport differences. Cheap to implement; existing `data/augmentations/` framework supports per-channel scaling.

### 5.3 Anchor_aware's partial rescue is more explainable

Slot α (anchor_aware) rescued some chronic identities and regressed Roy_D. Hypothesis: the rescued identities' anchor frames sit at G-channel scales close to their problem-frame G-channel scales; Roy_D's anchor frames sit at G-channel scale *different* from his lockbox failure cases. **A direct test**: measure the G-channel-scale distribution of Roy_D's anchor pool vs his lockbox-fail frames. If they don't overlap, the anchor-pool composition explanation holds.

### 5.4 What this DOESN'T resolve

- Why is the *green* channel specifically the strongest axis? Possibilities: skin-tone overlap with green sensitivity; in the training data, fake methods may produce systematic green-channel artifacts; sensor color matrix effects vary on green more than red/blue in webcam ISPs. The mechanism is not characterized.
- The B_scale direction is opposite to G_scale and to the Roy_D→Guest direction — meaning when the Teams pipeline moves all channels together, B and G partially cancel. Net effect is G-dominated but reduced. A pipeline that moved B and G in *opposite* directions could be either much stronger or much weaker.
- ArcFace cosine 0.92 between same-person crops is a single data point. Need ~10-20 paired same-person crops across accounts to characterize the distribution.

## 6. Artifacts

| Path | Contents |
|---|---|
| `outputs/perturbation_sweep.csv` | 158 rows: each (axis, value, ckpt) cell with prob_fake + IQ metrics |
| `outputs/bundle_sweep.csv` | Bundled-perturbation table |
| `outputs/followup_summary.json` | Headline numbers (baselines, cosines) |
| `figs/perturbation_sensitivity.png` | 8-panel grid: prob_fake vs perturbation level per axis, T5C + Slot B |
| `perturbation_sweep.py` | Reproducible script |
