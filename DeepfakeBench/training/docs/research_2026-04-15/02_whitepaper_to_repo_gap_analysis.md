# Whitepaper-to-Repo Gap Analysis

## Scope

This note compares the Effort whitepaper, **"Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection" / "Effort: Efficient Orthogonal Modeling for Generalizable AI-Generated Image Detection"** (`arXiv:2411.15633`), against the current repository and the user's current adaptation direction.

The comparison is not asking "is the paper good?" It is asking the more useful question:

> How much of the paper still transfers when the real target is a **small**, **generalizable**, **low-false-positive** detector for **live Microsoft Teams deepfakes**?

That is a much narrower and harsher target than the paper's published benchmark setting.

---

## 1. What The Paper Actually Proposes

### 1.1 Core model idea

The paper's central claim is that naive deepfake/AIGI detectors overfit to the limited fake patterns seen during training, creating a low-rank discriminative space. Effort tries to prevent that by starting from a pretrained vision foundation model and explicitly separating:

- a **frozen semantic subspace** carried by the dominant singular directions of pretrained weights
- a **trainable forgery subspace** carried by the residual singular directions

The paper frames this as preserving semantic knowledge while learning fake-specific cues in an orthogonal residual space instead of distorting the whole pretrained backbone.[^paper-arxiv][^paper-html]

### 1.2 Architecture

For deepfake detection, the paper's default backbone is **CLIP ViT-L/14**. It also reports ablations on other VFMs and on CLIP-Base/16, but the main headline numbers are attached to CLIP-L/14.[^paper-html]

Important architectural takeaways:

- The paper is fundamentally a **CLIP-based image/frame detector**, not a temporal video model.
- It uses **SVD-based residual adaptation** rather than full fine-tuning.
- Its efficiency headline for CLIP-L/14 is about **0.19M tunable parameters**, which is consistent with a very narrow residual adaptation regime.[^paper-html]

### 1.3 Losses / objective

The paper optimizes:

- a **classification loss**
- an **orthogonality constraint**
- a **singular-value-preservation / weight-norm-preservation style constraint**

The accessible paper text and appendix make the existence of those three components clear, and Table 6 specifically ablates the singular-value and orthogonal constraints. What the paper does **not** communicate clearly enough in prose is the exact hyperparameterization that matters most for reproduction in a new domain: exact lambda values, scheduling, layer coverage, and whether different layers receive different effective weighting.[^paper-html]

### 1.4 Training strategy

For deepfake experiments, the paper says it:

- uses the **DeepfakeBench** preprocessing/training pipeline
- samples **8 frames per video for training**
- uses **32 frames per video for inference**
- uses **Adam** with a fixed **`2e-4`** learning rate
- uses batch size **32**
- applies common augmentations such as **Gaussian blur** and **image compression**
- reports **video-level AUC** by averaging frame probabilities within a video[^paper-html]

This is important: the paper is still basically a **frame model aggregated to video AUC**, not a detector optimized around deployment-threshold false-positive control or per-frame stability.

### 1.5 Datasets and evaluation protocols

For deepfake detection, the paper uses two standard academic protocols:[^paper-html]

- **Protocol 1: cross-dataset**
  - train on **FF++ c23**
  - test on **CDF-v2, DFD, DFDC, DFDCP, DeeperForensics, WildDeepfake, FFIW**
- **Protocol 2: cross-manipulation within similar domain**
  - train on **FF++ c23**
  - test on **DF40**

This is a respectable generalization setup, but it is still an **academic benchmark regime**: curated datasets, standard crops, no Microsoft Teams transport layer, no live-call denoising/sharpening loop, and no business requirement framed as "keep real-call false positives low."

### 1.6 Paper assumptions

The paper implicitly assumes:

- generalization failure is driven mainly by **overfitting to seen fake artifacts**
- richer pretrained semantics help because they enlarge the discriminative space
- the key challenge is **unseen fake methods**, not deployment-specific real-domain shift
- frame/video benchmark AUC is a reasonable proxy for deployment usefulness

Those assumptions are not wrong. They are just incomplete for Teams.

### 1.7 Missing details that matter in practice

The paper is strong on the big idea and weaker on operational details. From the perspective of reproducing or extending it for this repo, the main missing details are:

- exact **layer-selection policy** for SVD adaptation is not explicit enough in the paper text
- exact **constraint weights / schedules** are not surfaced clearly enough
- exact augmentation probabilities and ranges are not specified at the level needed for a Teams-targeted reproduction
- there is no discussion of **threshold calibration**, **false-positive operating point**, or **frame-to-frame logit stability**
- there is no real treatment of **live deepfake transport effects** such as codec blur, sharpening, denoising, exposure changes, or crop jitter

That means the paper is useful as a principle paper, not as a deployment recipe.

---

## 2. What This Repo Currently Is

### 2.1 High-level system

This repo is not a paper-faithful clone anymore. It is now a **Teams-targeted Effort-derived detector stack**.

The current detector implementation in `detectors/effort_detector.py` is still recognizably Effort-like:

- CLIP/OpenCLIP vision backbone
- SVD residual adaptation over attention projections
- orthogonality and singular-value-preservation regularization

But the repo has added several pragmatic extensions that are driven by the actual target problem rather than the paper:

- OpenCLIP support for LAION backbones
- optional SVD on fused `q/k/v` projections for OpenCLIP
- optional selective block coverage and optional MLP coverage
- cosine/ArcFace-style heads with scale annealing
- label smoothing
- embedding-space mixup
- per-sample loss path for Group-DRO-like training
- optional quality-domain adversarial head via gradient reversal

That is already a major philosophical shift: the repo is treating Effort as a **base mechanism** inside a broader robustness system.

### 2.2 Current training system and domain emphasis

The current best-path configs are heavily tuned around the user's deployment target, not around FF++ benchmark reproduction. In particular, the recent R12/R13 configs show:

- **identity-balanced / identity-weighted combined training**
- training data mixed across **DF40**, **DeepLiveCam**, **VisoMaster**, **Teams passthrough**, **external YouTube/AVSpeech reals**, and **VCD reals**
- explicit weighting toward **Teams fake**, **Teams real**, and **enhanced fake** families
- OOD monitoring on external reals and Teams holdouts
- augmentation families targeted by source/family rather than a single generic academic preset

This is the correct direction for the user's stated goal. It is also far away from the paper's world.

### 2.3 Backbone reality in the repo vs the user's current adaptation

Most of the checked-in repo configs still point to **`ViT-B-16-DataComp-XL`** under OpenCLIP. The user says the current adaptation has been changed to **`laion/CLIP-ViT-B-16-laion2B-s34B-b88K`** to keep the model smaller.

Even though those are different pretrained checkpoints, the geometry is still basically the same **B/16 OpenCLIP family**:

- image size **224**
- **12** vision layers
- width **768**
- patch size **16**
- projected embedding dimension **512**[^laion-b16-card][^laion-b16-config]

So the key architectural comparison to the paper is unchanged:

- paper default: **OpenAI CLIP ViT-L/14**
- current user direction: **OpenCLIP ViT-B/16**

That is a real downshift in backbone capacity and semantic prior.

---

## 3. Paper vs Repo: Side-by-Side

| Axis | Whitepaper | Repo / current adaptation | Why it matters |
|---|---|---|---|
| Backbone default | CLIP ViT-L/14 | OpenCLIP ViT-B/16 family in current work | Smaller backbone weakens the paper's "semantic prior" advantage. |
| Adaptation width | Paper's best-efficient story centers on **k=1** residual direction per layer | Current R12/R13 setups use **`rank: 736`** on 768-dim attention layers, i.e. **k=32** trainable directions per layer | Repo has already moved from "minimal orthogonal adaptation" toward "pragmatic extra capacity." |
| Head / classifier | Classification loss on top of adapted backbone | Current best configs use `use_arcface_head: true` with `arcface_m: 0.0`, `arcface_s: 14.0` and long scale annealing | That is closer to cosine-softmax than paper-default CE over a plain linear head. |
| Data | FF++-centric academic protocols, DF40 for cross-manipulation | DF40 plausible swaps + DeepLiveCam + VisoMaster + Teams passthrough + enhanced fakes + external reals | Repo is optimizing for live-domain realism, not paper-style benchmark cleanliness. |
| Evaluation target | Video-level AUC | Business target is low FP on live Teams video; repo also tracks OOD and Teams-specific suites | AUC is not enough when false positives are expensive. |
| Augmentation | Generic blur/compression and standard DeepfakeBench pipeline | Family-aware quality-targeted augmentation; still recognized spatial fragility | Repo is already solving failure modes the paper never studied. |
| Stability goal | Not a primary objective | User explicitly cares about frame-level instability and threshold behavior | Paper does not answer this. |

---

## 4. The Most Important Gap: This Repo Is Solving A Different Problem

The paper's problem statement is:

> "How do we generalize to unseen fake generation methods without overfitting to training artifacts?"

The user's problem statement is:

> "How do we keep a **small** model **generalizable**, specifically for **live Teams deepfakes**, while keeping **false positives low** on real calls?"

Those overlap, but they are not the same optimization target.

For Teams, the dominant failure modes are not only unseen fake methods. They are also:

- **real-domain shift** caused by Teams transport and enhancement
- **enhanced deepfakes** whose mask/blend cues are much cleaner
- **lighting sensitivity**
- **crop-position sensitivity / patch-boundary sensitivity**
- **frame-to-frame score instability**
- the practical need to operate at a **conservative decision threshold**

The paper does not really speak to those.

---

## 5. Which Paper Claims Still Survive The Backbone Downshift?

The backbone downshift here means: from the paper's main **CLIP ViT-L/14** framing to the user's current **ViT-B/16 LAION** framing.

### 5.1 Claims that likely still survive

#### A. "Use a pretrained vision-language model as the anchor" still survives

This is still one of the paper's most durable ideas. Even after downshifting to B/16, using a pretrained CLIP-family encoder as the anchor is still more aligned with generalization than training a small detector from scratch.

Why this likely survives:

- the paper's own ablations show that **CLIP-Base/16 + Effort still improves substantially over plain CLIP-Base/16**, even if it does not match CLIP-L/14 absolute performance[^paper-html]
- the repo's history also points in the same direction: full scratch-like minimal detector approaches are not the main winning path; CLIP-family prior remains important

#### B. "Do not casually full-fine-tune everything if generalization is the goal" still survives

The paper is directionally right that unconstrained full fine-tuning risks collapsing the useful pretrained prior into dataset-specific artifact learning. For a Teams detector with scarce true target-domain data, that warning still matters.

#### C. "Orthogonal residual adaptation is a useful bias" still survives

Even if the exact paper-default `k=1` is too narrow for this repo, the structural bias of:

- preserving a semantic anchor
- learning in a constrained residual subspace

still makes sense for low-data, high-generalization settings.

#### D. "Coverage of attention projections matters" still survives

One repo lesson strongly reinforces the paper rather than contradicting it: partial or asymmetric adaptation coverage can badly distort results. The repo's own debugging around OpenCLIP `in_proj` coverage shows that architecture-faithful attention adaptation matters a lot when trying to port the method from HuggingFace CLIP to OpenCLIP.

### 5.2 Claims that become weaker after the downshift

#### A. The paper's headline magnitude becomes weaker

The paper's strongest story is attached to **CLIP-L/14** and its richer semantic basis. The paper also reports that **CLIP-Base/16 + Effort** helps, but it is still worse than **CLIP-L/14 + Effort**.[^paper-html]

So the safe conclusion is:

- the **principle survives**
- the **headline strength does not**

#### B. "Tiny residual capacity is enough" becomes much weaker

The paper's efficient sweet spot is basically the `k=1` story. This repo has already produced evidence that for B/16 on the actual target problem, **very narrow residual capacity is often not enough**.

Important repo reality:

- paper-like `k=1` on B/16 full attention coverage is about **73,776** SVD parameters
- current R12/R13-style `k=32` on B/16 full attention coverage is roughly **2,360,832** SVD parameters

That is not a small tweak. It is a different operating point.

The repo appears to have learned a hard lesson that the paper does not need to confront: on the smaller backbone and harder deployment domain, **extra residual capacity may be required just to get a stable, useful detector**.

#### C. "Semantic prior will rescue everything" becomes weaker

This becomes weaker for two reasons:

1. **ViT-B/16 has less representational slack than ViT-L/14.**
2. The user's chosen `laion2B-s34B-b88K` checkpoint is trained on a very large but explicitly **uncurated** LAION subset, which is not the same prior as the paper's default OpenAI CLIP-L/14.[^laion-b16-card]

That does not make the backbone bad. It just means the paper's argument about semantic preservation should be treated as **less powerful**, not equally powerful.

---

## 6. Mismatches Between The Paper's Assumptions And The User's Target Domain

### 6.1 Academic generalization is not Teams generalization

The paper mostly studies:

- unseen manipulation methods
- conventional benchmark datasets
- standard video/image compression conditions

The user's target domain adds another entire axis:

- **live communication stack shift**

That includes Teams-specific compression, denoising, exposure changes, sharpening, temporal bitrate changes, and camera/lighting variability. The paper does not model this distribution shift directly.

### 6.2 The paper is not optimizing low false positives

The paper reports AUC and average discrimination quality. That is useful but insufficient here.

For the user's deployment target, a detector can look strong on AUC and still be operationally poor if:

- the real-score tail is too wide
- calibration is bad
- lighting or crop jitter moves real frames across threshold

The paper does not treat low-FP thresholding as a first-class objective.

### 6.3 The paper is not about enhanced live deepfakes

The user's current weakness on **enhanced** DeepLiveCam / VisoMaster outputs, especially after Teams transport, is a deployment-specific challenge. The paper's evaluation regime does not cover the "face enhancement + live-call transport" combination in any comparable way.

### 6.4 The paper does not address spatial crop fragility

The paper uses the standard frame-sampling / image-processing pipeline, but it does not treat small crop shifts as a core failure mode. In this repo, that issue is now explicit: tiny crop boundary changes can cause large score swings. For ViT-B/16 face crops, that matters a lot because patch assignment shifts quickly.

### 6.5 The paper does not address frame-level instability

The user reports a major symptom:

- visually similar nearby frames can produce materially different probabilities

That is not a main topic of the paper, because paper evaluation is centered on aggregate video-level AUC, not temporal smoothness or intra-segment logit consistency.

---

## 7. What The Repo Is Doing That Is Smarter Than The Paper For This Problem

The repo has already moved in several directions that are more appropriate for Teams than paper-default Effort:

- **target-domain data weighting** instead of benchmark-balanced purity
- explicit inclusion of **Teams passthrough** data
- emphasis on **external real data** to protect false positives
- explicit handling of **enhanced fake families**
- family-aware augmentation rather than one generic recipe
- willingness to widen the residual subspace beyond `k=1`
- OOD suites closer to the actual deployment risk

This is not "drift from the paper." It is mostly **necessary adaptation**.

In other words: if the user reverted fully back toward paper-default settings just because the paper is elegant, there is a good chance the detector would become **more benchmark-faithful and less useful**.

---

## 8. Where The Paper Is Still Telling The Repo Something Important

Even though the repo is right to diverge, the paper still gives several useful warnings:

### 8.1 Do not let the target-domain push erase the pretrained prior

Because Teams-targeted weighting and enhanced-fake focus are becoming stronger, there is a real risk of building a detector that is locally optimized to current Teams artifacts and weakly robust beyond them. The paper's core warning about overfitting to narrow fake patterns remains relevant.

### 8.2 Bigger residual subspace is useful, but it should not become unconstrained full drift

The repo's move from `k=1` toward `k=16/24/32` may be correct for this problem. But the paper's philosophy still says:

> more capacity should be added carefully, because the entire point is to preserve useful pretrained structure

That means capacity increases should still be judged by:

- Teams fake recall
- Teams real FP
- enhanced-fake recall
- stability under crop / lighting perturbation
- and robustness on external reals

not only by in-domain holdout AUC.

### 8.3 Semantic preservation matters most when real diversity is large

The user's target domain has very diverse real data:

- YouTube / AVSpeech style reals
- VCD / webcam reals
- Teams passthrough reals
- lighting variation

That is actually a place where the paper's asymmetry story still bites hard: fake classes are narrow and engineered; real classes are messy and broad. So the paper's diagnosis remains relevant even if the benchmark setup does not.

---

## 9. Practical Bottom Line

If I translate the paper into repo-specific language, the honest conclusion is:

1. **Respect the paper's core bias** toward preserving pretrained semantics and learning fake-specific residuals in a constrained way.
2. **Do not over-respect the exact paper recipe**, because the target problem is no longer the same.
3. **Assume the backbone downshift weakens the paper's strongest claims**, especially the "tiny residual capacity is enough" claim.
4. **Treat Teams-domain realism, enhanced-fake coverage, crop robustness, lighting robustness, calibration, and score stability as first-class concerns** even when the paper is silent.

The repo is already, in broad strokes, moving the right way by becoming less paper-pure and more target-honest.

---

## What From The Paper Still Deserves To Be Respected

- **The asymmetry diagnosis**: detectors really do overfit to a narrow set of fake artifacts faster than they learn the full real manifold. That is still true here, and probably even more true when live fake families are limited but real Teams calls are diverse.[^paper-arxiv][^paper-html]
- **CLIP-family prior as the anchor**: starting from a pretrained vision-language model remains more credible for generalization than training a lightweight detector from scratch.
- **Orthogonal residual adaptation as a bias**: preserving the semantic anchor while learning the fake residual is still a sound principle for low-data, generalization-sensitive deployment.
- **Constraint-aware adaptation**: the idea that not all extra trainable capacity is equally safe still matters. Wider `k` may be needed here, but it should still be treated as controlled expansion, not an excuse to forget why Effort worked.
- **Paper-backed warning that B/16 is weaker than L/14**: the paper's own ablations support the idea that the method still helps on CLIP-Base/16, but not as strongly as on CLIP-L/14.[^paper-html]

## Where The Paper Is Not Enough For Teams

- It does **not** solve the real deployment problem of **low false positives on live Teams reals**.
- It does **not** address **enhanced live deepfakes** passing through Teams.
- It does **not** treat **lighting shift**, **crop jitter**, or **frame-score instability** as core objectives.
- It does **not** optimize around a **thresholded operating point**; AUC alone is too weak for this use case.
- It does **not** tell you whether paper-default `k=1` is still appropriate once the backbone is downshifted to B/16 and the domain is shifted to Teams. Repo evidence already suggests that this becomes much less believable.
- It does **not** justify trusting academic cross-dataset success as a proxy for real video-call success. Teams is its own domain.

---

## References

[^paper-arxiv]: arXiv abstract page: https://arxiv.org/abs/2411.15633
[^paper-html]: ar5iv HTML rendering of the paper: https://ar5iv.labs.arxiv.org/html/2411.15633v4
[^laion-b16-card]: Hugging Face model card for `laion/CLIP-ViT-B-16-laion2B-s34B-b88K`: https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K
[^laion-b16-config]: OpenCLIP config for `laion/CLIP-ViT-B-16-laion2B-s34B-b88K`: https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K/blame/main/open_clip_config.json
