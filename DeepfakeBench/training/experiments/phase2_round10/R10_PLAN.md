# R10 Strategic Plan

> **Date:** 2026-03-06
> **Production champion:** R9_A (`1551zxa8`) — Holdout AUC 0.9891, OOD AUC 0.9768
> **Fine-tune base:** R8_E (`hu7cen3m`) — Holdout AUC 0.9925, VCD real 82.1%

---

## 1. What Changed Since R9.5

### 1a. New Data: Teams v2

| Metric | Teams v1 | Teams v2 |
|--------|:--------:|:--------:|
| Complete pairs | ~288 | **1,346** |
| Total files | ~11K | **54,234** |
| Size | ~0.8 GB | **3.8 GB** |
| Bucket | `live-deepfake-methods-real-and-fake-frames-cropped-teams` | `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2` |
| Frame format | JPG | JPG |
| Structure | Identical manifests | Identical manifests + root `metadata.json` |

**4.7× more Teams data.** Same structure as v1 — the data source code only needs a bucket name change.

### 1b. WMA JPEG Bug Fix (In Progress)

The WMA client was compressing face crops to JPEG before sending to the inference API. This injected compression artifacts not present in the original frames. **Fix:** send raw bytes (PNG or uncompressed) instead. Once deployed, the inference path becomes: raw WMA crop → `cv2.imdecode` → resize 224×224 → CLIP normalize — effectively PNG-equivalent quality.

**Impact:** After the fix, production images will be *closer* to the PNG training data than to the JPG Teams training data. This reduces the domain gap for all non-Teams sources.

### 1c. Confirmed Findings from R9.5

| Finding | Evidence | R10 Action |
|---------|----------|------------|
| Stability λ monotonically hurts OOD | λ=0 (0.9768) > λ=0.1 (0.9729) > λ=0.3 (0.9661) > λ=0.5 (0.9556) | **λ=0, no label smoothing** |
| FT landscape is narrow | 5/5 FT runs produced identical per-method DF40 scores | **Scratch preferred** |
| facedancer is architectural ceiling | 5× DF40 weight had zero effect (59.1%) | **Accept or go ViT-L-14** |
| Scratch preserves facedancer | R95_D: 68.2% vs FT: 59.1% | **Scratch confirms** |
| R95_D was undercooked | EWI=1 at 12K steps | **22K+ steps for scratch** |

---

## 2. The Three Real Problems

### Problem 1: Lighting Sensitivity (CRITICAL)

Same person, same camera, different background lighting → model output swings 0.1 to 0.9. This is the #1 production failure mode.

**Root cause:** CLIP's frozen top-736 singular components encode lighting information. The rank-32 residual subspace can't also learn lighting invariance — it needs all capacity for real/fake discrimination. ArcFace s=12 amplifies small cosine perturbations (±0.02) into ±8-10pp probability swings.

**Current augmentation is too mild:** R9_A uses `gamma [80,120]`, `brightness 0.25`, `contrast 0.25`, `oneof_p 0.30`. Real webcam lighting swings are much wider.

### Problem 2: Score Jitter Between Similar Frames

R9_A YouTube jitter = 0.039 (target <0.03). Nearly identical frames produce different scores. Related to Problem 1 but also involves:
- ArcFace scale amplifying small cosine perturbations into large probability swings ($s=18$ → 4.5pp per 0.01 cosine shift)
- Resize interpolation sensitivity (tiny pixel shifts → different feature maps)
- The frozen CLIP subspace encoding nuisance factors (lighting, pose) that the trainable residual can't cancel

**Key insight:** R9.5 proved the explicit stability loss (KL on perturbed pairs) hurts generalization because it competes with classification. The solution is to make the model inherently stable through data diversity (augmentation), a gentler probability surface (lower ArcFace s), and smoother optimization (mixup, SWA). See [Section 5](#5-building-an-inherently-stable-model).

### Problem 3: VCD Real Regression

82.1% (R8_E) → 80.3% (R9_A) → 76.9% (R95_B). Adding Teams data dilutes real discrimination. Target: ≥85%.

---

## 3. Pre-R10 Diagnostic Experiments

Run these before any training. ~3 hours total. They sharpen the R10 design.

### Experiment 0A — JPEG Sensitivity Sweep

**Goal:** Quantify how much JPEG compression affects current model scores.

**Method:** Take R9_A checkpoint. Run inference on in-distribution validation PNGs at native quality, then on the same images JPEG-compressed at Q50, Q70, Q85, Q95. Compare AUC and per-method accuracy.

**Outcome:** If AUC drops ≥3pp at Q85, the JPEG bug fix alone will measurably help. If AUC barely moves, JPEG isn't the bottleneck.

**Time:** ~1 hour

### Experiment 0B — Lighting Sensitivity Profile

**Goal:** Map the model's decision surface under lighting perturbations.

**Method:** Take 30 known-real + 30 known-fake face crops from the validation set. Apply systematic brightness multipliers (0.5×, 0.7×, 0.85×, 1.0×, 1.15×, 1.3×, 1.5×, 2.0×) and gamma corrections (0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0). Plot `fake_prob` vs. perturbation.

**Outcome:** Reveals whether lighting pushes reals → fake, fakes → real, or both. Shows the safe operating envelope. Directly informs how wide the R10 augmentation range must be.

**Time:** ~1 hour

### Experiment 0C — Teams v2 Sanity Check

**Goal:** Confirm Teams v2 has same score distribution as v1 under R9_A.

**Method:** Run R9_A on a random 200-sample subset of Teams v2 (100 real, 100 fake). Compare score histograms against Teams v1 eval data.

**Outcome:** Catches structural anomalies (broken crops, new identities with unexpected properties) before committing to a full training run.

**Time:** ~30 min

---

## 4. R10 Experiment Matrix

### Design Principles

1. **Scratch over fine-tune** — the R8_E checkpoint constrains the optimization landscape
2. **λ=0** — stability regularization is counterproductive (R9.5 confirmed)
3. **No label smoothing** — same finding
4. **Aggressive lighting augmentation** — address Problem 1 at the data level
5. **Teams v2 at rebalanced weights** — 4.7× more data means lower weight needed
6. **Longer training** — R95_D was still improving at 12K steps

### Family Weight Rationale

With 1,346 Teams pairs (vs ~288 v1), the old weight of 7.0 would make Teams dominate per-identity selection. Rebalanced:

| Family | R9_A Weight | R10 Weight | Rationale |
|--------|:-----------:|:----------:|-----------|
| `df40_fake` | 0.2 | **0.15** | Further deprioritized |
| `df40_real` | 0.5 | **0.4** | Further deprioritized |
| `visomaster_fake` | 2.0 | **2.5** | Slight bump — core capability |
| `deeplive_non_enhanced_fake` | 2.5 | **2.5** | Unchanged |
| `deeplive_enhanced_fake` | 3.0 | **3.0** | Unchanged |
| `deeplive_teams_fake` | 7.0 | **4.0** | Reduced — 4.7× more data compensates |
| `deeplive_teams_real` | 5.0 | **3.0** | Reduced |
| `realpool_real` | 1.5 | **2.0** | Bumped — real robustness priority |
| `external_real` | 2.0 | **2.5** | Bumped — real robustness priority |

At these weights, for a mixed identity with Teams + DeepLive + VisoMaster: Teams selected ~4/(4+3+2.5+2.5) ≈ 33% of the time (was ~48% with weight 7.0).

### Run Matrix

| Run | Strategy | Steps | Backbone | ArcFace s | Aug | Key Differentiator |
|-----|----------|------:|----------|:---------:|:---:|-------------------|
| **R10_A** | Scratch | 22K | B16-LAION | 10→14 | Wide | **Main bet** — lighting fix + moderate s + Teams v2 |
| **R10_B** | Scratch + mixup α=0.3 | 22K | B16-LAION | 10→14 | Wide | Augmentation + embedding mixup smoothness test |
| **R10_C** | FT from R9_A | 8K | B16-LAION | 6→12 | Narrow | **Cheapest** — Teams v2 data-only domain adaptation |
| **R10_D** | Scratch | 22K | **L-14** | 10→14 | Wide | Capacity ceiling test (facedancer >80%?) |
| **R10_E** | Scratch | 22K | B16-LAION | 10→**12** | Wide | Low-s ablation — max stability, test AUC tradeoff |
| **R10_F** | Scratch | 22K | B16-LAION | 10→14 | **Narrow** | Augmentation isolation (vs R10_A) |
| **R10_G** | FT from R9_A | 8K | B16-LAION | 6→12 | Wide | FT + wider aug (vs R10_C) |

**Clean ablation axes:**
- **R10_A vs R10_F:** Effect of wide augmentation (only aug differs)
- **R10_A vs R10_E:** Effect of ArcFace s endpoint (14 vs 12)
- **R10_A vs R10_B:** Effect of embedding mixup
- **R10_C vs R10_G:** Effect of wider aug in the FT setting
- **R10_A vs R10_D:** Effect of backbone capacity (B16 vs L-14)
- **R10_A vs R10_C/G:** Scratch vs fine-tune strategy

**All 7 runs** share: Teams v2 bucket, λ=0, no label smoothing, rebalanced family weights (Teams 4.0/3.0, reals bumped).

#### R10_A — Main Run (Scratch + Lighting Fix)

The central hypothesis: the model's lighting sensitivity is an augmentation gap, fixable by wider training-time perturbations combined with more on-domain Teams data.

Key parameters:
- **Checkpoint:** None (scratch from CLIP weights)
- **LR:** 2e-4, cosine_with_warmup, 1000 warmup steps
- **ArcFace:** s: 10→14, m: 0.0 (reduced from 18 — stability tradeoff, see Section 5)
- **Teams bucket:** `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`
- **Augmentation changes from R9_A:**
  - `context_variation_gamma_limit: [50, 150]` (was [80,120] — 3× wider swing)
  - `context_variation_brightness: 0.40` (was 0.25)
  - `context_variation_contrast: 0.35` (was 0.25)
  - `context_variation_oneof_p: 0.50` (was 0.30 — half of all images get lighting-perturbed)
  - `context_variation_rotate: 12` (was 10)
- **TeamsCodecSimulation:** p=0.10 (was 0.15 — less needed with more Teams data)

#### R10_B — Maximum Aug Diversity

Same as R10_A, plus:
- `context_variation_gamma_limit: [40, 160]` (even wider)
- `context_variation_oneof_p: 0.65`
- Additional `ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2, hue=0.05)` in the color block
- `RandomToneCurve(scale=0.3)` for non-linear brightness shifts
- Consider: random shadow/vignette overlay

If R10_B beats R10_A on OOD, the answer was always augmentation breadth.

#### R10_C — Quick FT Adaptation (Run First)

Fine-tune R9_A to absorb 4.7× more Teams data. Cheapest experiment.
- **Checkpoint:** R9_A
- **LR:** 3e-5
- **Steps:** 8K
- **Teams weight:** 5.0 (higher than A because purpose is Teams absorption)
- Everything else from R9_A config
- Same augmentation as R9_A (no lighting changes — isolates the Teams data effect)

#### R10_D — ViT-L-14 (Optional)

Same plan as R10_A but ViT-L-14 (hidden=1024, rank=1023, k=1). Tests whether facedancer ceiling lifts with 2× backbone capacity. Only run if facedancer >80% is a production requirement.

---

## 5. Building an Inherently Stable Model

R9.5 proved: `stability_lambda` (KL-divergence between clean and perturbed predictions) monotonically hurts OOD generalization. The reason is structural: the stability loss competes directly with the classification loss — the model can't simultaneously be maximally discriminative and maximally invariant to perturbations when both are expressed as loss terms pulling gradients in different directions.

But the problem is real: jitter = 0.039 on YouTube, and lighting alone swings outputs from 0.1 to 0.9 on the same face. **The model itself must be stable. No inference tricks.**

### Why the Current Model Is Unstable

The instability has three compounding causes:

**Cause 1 — ArcFace scale amplifies everything.**
The output probability is $\text{softmax}(s \cdot \cos\theta)$ where $s$ is the ArcFace scale. R9_A ends at $s=12$, R8_E at $s=18$.

The sensitivity at the decision boundary ($\cos\theta \approx 0$):

| ArcFace s | Δprob per 0.01 cosine shift | Δprob per 0.05 cosine shift |
|:---------:|:---------------------------:|:---------------------------:|
| 8 | ~2pp | ~10pp |
| 12 | ~3pp | ~15pp |
| 18 | ~4.5pp | ~22pp |
| 30 | ~7.5pp | ~37pp |

A small lighting change that moves the CLIP embedding by 0.05 cosine distance (entirely plausible) causes a 15pp probability swing at $s=12$, and 22pp at $s=18$. **The scale is a jitter multiplier.**

**Cause 2 — The frozen subspace carries lighting information.**
CLIP's ViT features encode illumination. The top-736 frozen singular components (which produce the dominant embedding) respond to lighting changes. The rank-32 trainable residual can learn to discriminate real vs. fake, but it cannot cancel out the lighting-sensitivity baked into the frozen 736 directions. The frozen directions dominate the final cosine similarity.

**Cause 3 — Training data has narrow lighting diversity.**
R9_A's augmentation: gamma $[80, 120]$, brightness $\pm 0.25$, applied to only 30% of images. Real webcam conditions swing well beyond this range (backlight, screen glow, window light, office overhead).

### Training-Time Stability Approaches

All of these modify the model's inherent behavior. No inference-time changes.

#### Approach A: Aggressive Augmentation (Primary)

The most proven mechanism. Wider augmentations teach the model that "same face + different lighting/crop = same label" by expanding the training distribution. The cross-entropy loss naturally pushes all augmented versions toward the same prediction — no auxiliary loss needed.

| Augmentation | R9_A | R10 Target | Effect |
|-------------|:----:|:----------:|--------|
| Gamma range | [80,120] | **[50,150]** | 3× wider — covers backlight ↔ overexposure |
| Brightness | ±0.25 | **±0.40** | Handles auto-exposure swings |
| Contrast | ±0.25 | **±0.35** | Handles dynamic range variation |
| Application rate | 30% | **50%** | Half of all training images get perturbed |
| Color jitter hue | None | **±0.05** | Handles white balance / color temp shifts |

This is strictly better than `stability_lambda` for a subtle reason: `stability_lambda` adds noise + crop jitter and penalizes output difference (KL). But it applies to **already-augmented** images, so the model sees `augment(original) → stability_perturb(augment(original))` — an artificial composition that doesn't match real lighting shifts. Wider augmentation directly covers the real perturbation distribution.

**Expected impact:** Reduces the lighting-driven component of jitter (estimated 60-70% of total). Target: 0.039 → 0.025.

#### Approach B: Lower ArcFace Final Scale

This is the most **direct** lever on jitter magnitude and it's pure model configuration.

The R9 lineage used $s: 10 \to 18$ for scratch, $s: 6 \to 12$ for FT. These were chosen for discrimination quality. But $s$ is a double-edged sword: higher $s$ sharpens the decision boundary (better accuracy) but amplifies noise (worse stability).

**Proposal — test $s: 10 \to 14$ for scratch (R10_A):**
- Still higher than the gentle FT schedule ($6 \to 12$)
- $s=14$ at decision boundary: ~3.5pp per 0.01 cosine shift (vs 4.5pp at $s=18$), a 22% jitter reduction purely from scale
- If Experiment 0B reveals that the cosine perturbation from lighting is ~0.03-0.05, then $s=14$ keeps the maximum probability swing under ±18pp (vs ±22pp at $s=18$)

**Risk:** $s=14$ might slightly hurt holdout AUC. But R95_E showed that $s: 10 \to 18$ actually *hurt* OOD vs $s: 6 \to 12$. So a moderate $s=14$ endpoint may actually improve generalization.

**Ablation variant (R10_E):** Run R10_A identically but with $s: 10 \to 12$. If jitter drops substantially with minimal AUC loss, this becomes the production config.

#### Approach C: Mixup in Embedding Space

Mixup generates synthetic training points by linearly interpolating between pairs:

$$\tilde{x} = \lambda x_i + (1-\lambda) x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j, \quad \lambda \sim \text{Beta}(\alpha, \alpha)$$

Applied in the **CLIP embedding space** (after the frozen backbone, before the ArcFace head), this:
1. Fills in the embedding manifold between real training points — the model sees a continuous distribution instead of discrete points
2. Forces the decision boundary to be smooth (can't create a sharp nonlinear boundary between interpolated points)
3. Acts as implicit regularization — well-studied to improve calibration and reduce overconfidence

**Implementation:** ~30 lines in the training step. After backbone extraction, before head:
```python
if mixup_alpha > 0:
    lam = np.random.beta(mixup_alpha, mixup_alpha)
    idx = torch.randperm(embeddings.size(0))
    embeddings = lam * embeddings + (1 - lam) * embeddings[idx]
    labels_a, labels_b = labels, labels[idx]
    loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
```

This is architecturally different from `stability_lambda`: mixup doesn't penalize output difference on near-identical inputs. It forces the embedding-to-output mapping to be *globally smooth*, which indirectly reduces sensitivity to small input variations.

**Suggested α:** 0.2-0.4 (light mixup). Higher α produces more interpolation near 0.5 which heavily regularizes.

**Risk:** Embedding-space mixup may interact poorly with ArcFace's L2-normalization (interpolated embeddings aren't on the unit sphere). Needs testing. Alternative: apply mixup to raw CLIP features before L2 norm.

#### Approach D: Stochastic Weight Averaging (SWA)

Average model weights across the last N training checkpoints:

$$\theta_{\text{SWA}} = \frac{1}{K} \sum_{i=1}^{K} \theta_i$$

SWA finds wider optima that are inherently more stable — the loss landscape around $\theta_{\text{SWA}}$ is flatter, meaning small input perturbations produce smaller loss (and output) changes.

**Implementation:** PyTorch built-in `torch.optim.swa_utils.AveragedModel`. After the main training loop, continue for 2-3K steps with SWA enabled (cyclic or constant LR), then `update_bn()`.

No auxiliary loss, no inference changes — just a different weight configuration that sits in a flatter basin.

**Integration with Effort:** The SVD structure (`weight_main` frozen + `U_residual`, `S_residual`, `V_residual` trainable) is fully compatible with SWA — just average the trainable parameters across checkpoints.

**Expected impact:** SWA typically reduces calibration error by 20-40% and smooths the probability surface. The optimal region found by SWA is more robust to input perturbations because the loss surface is flatter in all directions.

#### Approach E: Spectral Regularization on the Residual

Different from `stability_lambda`. Instead of "output on perturbed input should match output on original input" (which requires paired evaluation and competes with classification), constrain the **maximum sensitivity** of the trainable residual.

The trainable residual contribution is $\Delta W = U_{\text{res}} \cdot \text{diag}(S_{\text{res}}) \cdot V_{\text{res}}^T$. The spectral norm $\|S_{\text{res}}\|_\infty$ (largest singular value) bounds how much the residual can amplify any input perturbation. Capping it constrains the Lipschitz constant of the *trainable component* without touching the frozen backbone.

**Implementation:** During training, after each optimizer step:
```python
with torch.no_grad():
    S_res.clamp_(max=max_spectral_norm)
```

Or as a soft penalty: $\lambda_{\text{spec}} \cdot \max(0, \|S_{\text{res}}\|_\infty - \tau)^2$.

**Key difference from stability_lambda:** This doesn't require any perturbed forward pass. It constrains the model's *capacity to amplify* perturbations structurally, rather than penalizing specific perturbation outcomes. It doesn't compete with the classification loss — it sets an upper bound on how extreme the residual's contribution can be.

**Risk:** Too-tight a spectral bound reduces the residual's discriminative power (it's limited in how much it can separate real/fake). Needs tuning — start with $\tau$ set to 2× the learned $S_{\text{res}}$ value from R9_A, then tighten.

### Stability Approach Ranking

| Approach | Changes Training Loss? | Competes with Classification? | Implementation Complexity | Expected Jitter Reduction |
|----------|:---------------------:|:-----------------------------:|:-------------------------:|:-------------------------:|
| **A. Wider augmentation** | No | No | Config only | **Largest** (lighting component) |
| **B. Lower ArcFace s** | No (just hyperparameter) | Mild tradeoff | Config only | **Moderate** (probability surface) |
| **C. Embedding mixup** | Yes (mixed labels) | No (orthogonal) | ~30 LoC | **Moderate** (smoother boundary) |
| **D. SWA** | No | No | ~50 LoC post-training | **Moderate** (flatter optimum) |
| **E. Spectral regularization** | Yes (soft penalty) | Mild | ~20 LoC | **Unknown** (principled but untested) |

### Recommended R10 Stability Stack (Training Only)

**Tier 1 — All R10 runs (zero risk):**
- Aggressive augmentation (Approach A) — pure config
- Lower ArcFace endpoint: $s: 10 \to 14$ (Approach B) — pure config

**Tier 2 — Ablation run(s):**
- R10_A vs R10_A+SWA (Approach D): same training, apply SWA at the end
- R10_A vs R10_A+Mixup (Approach C): α=0.3 embedding-space mixup

**Tier 3 — Experimental (only if Tier 1+2 insufficient):**
- Spectral regularization (Approach E): needs tuning, may interact with orthogonal loss

**Combined target (Tier 1 only):** jitter from 0.039 → 0.020-0.025 via augmentation + moderate ArcFace scale.

---

## 6. Image Format Strategy

### Decision: Don't Re-Capture Teams as PNG

The Teams frames have two artifact sources convolved:
1. **VP8/VP9 codec processing** from the Teams call (−50.9% sharpness, −77.4% HF energy) — this is the **real signal** we want the model to learn
2. **JPEG save compression** at write time — this is noise, but at Q80+ it's minor compared to (1)

Re-capturing as PNG removes artifact (2) but the frames still have artifact (1) baked in. The benefit is marginal for the cost.

### Should We Train on Both PNG and JPG?

**No.** Once the WMA JPEG bug is fixed, production inputs will be effectively PNG-quality (raw bytes → decode). Training on clean PNG data (90%+ of the training set) with light JPEG augmentation (`ImageCompression Q40-90` already in the pipeline) is sufficient.

### JPG Format in Teams Loader

The Teams frame loader hardcodes `.jpg` at [combined_paired.py L1818](../data/sources/combined_paired.py#L1818):
```python
blob_path = f"{prefix}frame_{idx:04d}.jpg"
```

Teams v2 also uses `.jpg`, so no change needed now. If future data arrives as PNG, make the loader format-agnostic (read from manifest or try both).

---

## 7. DF40 Strategy

Per the priority ranking: real robustness > DeepLive/VisoMaster > Teams domain > DF40.

- **target_source methods (7):** Keep, weight 0.15. These are face-swap methods most relevant to production.
- **source_target methods (10):** Don't add. Reenactment methods are less production-relevant and would dilute the training signal.
- **facedancer ceiling (68%):** Accepted as architectural for ViT-B-16. Only ViT-L-14 or scratch training can push beyond this. Not worth chasing with data rebalancing (R95_F proved this).

---

## 8. Action Items Checklist

### Pre-R10 (Before Any Training)

- [ ] **Fix WMA JPEG bug** — send raw bytes from WMA client to inference API
- [ ] **Run Experiment 0A** — JPEG sensitivity sweep on R9_A with validation PNGs at Q50/Q70/Q85/Q95
- [ ] **Run Experiment 0B** — Lighting sensitivity profile (30 real + 30 fake × 7 brightness × 7 gamma). Key output: cosine distance vs. lighting perturbation → directly informs ArcFace s target
- [ ] **Run Experiment 0C** — Teams v2 sanity check (200 random samples through R9_A)
- [ ] **Verify Teams v2 manifest compatibility** — Confirm `manifest.json` fields match loader expectations: `sample_id`, `strategy`, `original_video_name`, `frame_count`, `pair_complete`

### Code Changes for R10

- [x] **Teams bucket config change** — point to `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`
- [x] **Widen augmentation gamma range** — `context_variation_gamma_limit: [50, 150]` in experiment YAML
- [x] **Increase augmentation probability** — `context_variation_oneof_p: 0.50`
- [x] **Increase brightness/contrast augmentation** — `0.40` / `0.35`
- [ ] **Add hue jitter** — `hue_shift_limit: 0.05` in color block for white-balance invariance
- [x] **Rebalance family weights** — reduce Teams from 7.0/5.0 to 4.0/3.0, bump reals
- [x] **Lower ArcFace s endpoint** — `arcface_s_end: 14` (from 18) in scratch configs
- [x] **Implement embedding-space mixup** — ~30 LoC in `EffortDetector.forward()` + `get_losses()`, gated behind `mixup_alpha` config (default 0.0)
- [ ] **Implement SWA** — PyTorch `AveragedModel` wrapper, run for 2-3K steps post-training
- [ ] **Optional: Add `ColorJitter` and `RandomToneCurve`** to augmentation pipeline for R10_B

### R10 Training Runs (All 7 Ready for Overnight Launch)

- [ ] **R10_A** — Scratch + wide aug + s→14 + Teams v2 (main bet, 22K steps)
- [ ] **R10_B** — R10_A + embedding mixup α=0.3 (smoothness ablation)
- [ ] **R10_C** — FT from R9_A + Teams v2 only (cheapest, 8K steps)
- [ ] **R10_D** — ViT-L-14 scratch + wide aug (capacity ceiling, 22K steps)
- [ ] **R10_E** — R10_A but s→12 (max stability ablation)
- [ ] **R10_F** — R10_A but narrow aug (augmentation isolation)
- [ ] **R10_G** — FT from R9_A + Teams v2 + wide aug (FT + aug test, 8K steps)

### Post-Training

- [ ] **Re-evaluate with JPEG bug fixed** — Run best R10 checkpoint through the fixed WMA pipeline
- [ ] **Measure jitter directly** — YouTube OOD set, compute per-method score std on sequential frames
- [ ] **Docker rebuild** with updated configs and any code changes (mixup, SWA)

### Not Doing

- ~~Re-capture Teams data as PNG~~ — marginal benefit for heavy cost
- ~~Stability lambda~~ — monotonically hurts OOD (confirmed R9.5)
- ~~Label smoothing~~ — same finding
- ~~Test-time augmentation~~ — model must be stable by itself
- ~~Inference-time smoothing changes~~ — not addressing the root cause
- ~~Temperature scaling~~ — cosmetic, doesn't fix the model
- ~~Add source_target DF40 methods~~ — deprioritized
- ~~ArcFace s > 18~~ — hurts OOD (confirmed R95_E)
- ~~Chase facedancer with data rebalancing~~ — architectural ceiling (confirmed R95_F)

---

## 9. Success Criteria

| Metric | R9_A (Current) | R10 Target | Notes |
|--------|:--------------:|:----------:|-------|
| OOD AUC | 0.9768 | **≥ 0.980** | Small improvement is fine if stability improves |
| VCD Real Accuracy | 80.3% | **≥ 85%** | Recovered from Teams dilution |
| Teams EC (holdout) | 70% | **≥ 80%** | 4.7× more data should help |
| YouTube Jitter | 0.039 | **≤ 0.025** | Via augmentation + lower ArcFace s |
| WMA Fake | 99.9% | **≥ 99.5%** | Maintain |
| Lighting Swing (0B profile) | ~0.8 delta | **≤ 0.3 delta** | Maximum prob swing under 2× brightness |
