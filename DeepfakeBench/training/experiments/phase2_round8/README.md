# Phase 2, Round 8 — Target-Domain Focus

> **Date**: Feb 23, 2026
> **Premise**: Time is tight. Stop optimizing for DF40 holdout AUC. Make
> DeepLiveCam and VisoMaster the training focus. Ship a model that works
> on actual video‑conferencing deepfakes.

---

## The Problem (in one paragraph)

After 7 rounds of experimentation, the model is near‑perfect on DF40
(92-99% per‑method) but fails on the actual deployment target:
**VisoMaster overall 69%**, **MINIMAL tier 44% (coin‑flip)**, several
swap models below 50% (GhostFace‑v2 46%, InStyleSwapper256-A 47%).
DeepLiveCam is good (~96%) but cannot regress. VCD reals are
false‑positive'd at ~30%. The model learned DF40's quality profile as a
shortcut; DF40's volume (5,400 pairs / 8 methods) drowns out the ~1,000
DeepLive + VisoMaster samples. **R8 inverts the distribution.**

---

## Key Design Decisions

### 1. VisoMaster Tiers Are Semantically Different — Treat Them So

| Tier | What it means | Training role |
|------|---------------|---------------|
| **STRONG** | Obvious artifacts (color blobs, warping) | Easy positives — helps calibrate but shouldn't dominate. Model already catches some of these. |
| **MODERATE** | Noticeable but not blatant | Core training signal — realistic operational difficulty. |
| **MINIMAL** | Very subtle — passed through the generator but nearly identical to original | **The hardest and most important tier.** If the model catches MINIMAL, it catches everything. But training only on MINIMAL risks the model failing to learn at all (signal too weak early on). |

**Strategy**: Include ALL tiers, but weight MODERATE highest and use
MINIMAL as the key validation metric. Exclude STRONG from family weights
(let it come in via natural sampling but don't upweight it — it's too
easy and its artifacts are non-representative).

> NOTE: The current code treats all VisoMaster as a single `visomaster_fake`
> family key. We don't split by tier in the family weights. The tier
> effect comes purely from which samples exist in the dataset when
> `tiers: null` (= all tiers included). This is fine — the volume balance
> handles itself because the sampling is identity-balanced.

### 2. DeepLiveCam Is the Priority — Despite Less Data

DeepLiveCam is the most realistic deployment scenario (real-time face
swap via webcam). It has ~1K samples across 5 strategies + 2 enhanced.
It already detects well (96%) — the risk is **regression**. Every R8
config must gate on DeepLive TPR ≥ 95%.

The enhanced strategies (GFPGAN-enhanced fakes) are especially important
because enhancement is how adversaries defeat detectors in practice.

### 3. DF40 Becomes a Supplement, Not the Core

DF40 contributes generic "face‑swap awareness" but its quality profile
(smooth, low‑noise, lab‑grade) teaches the wrong shortcuts. **R8 keeps
DF40 but drastically reduces its weight** so it contributes breadth
without dominating gradients.

`faceswap` method is excluded from training (broken data, 54% accuracy
across all rounds — consistently the worst).

### 4. All VisoMaster Swap Models In Training — No Holdout

Previous rounds held out GhostFace‑v2 and Inswapper128 for "OOD
monitoring." But these are the two weakest methods (46% and 55%), and
we're past the exploration phase — we need to *train* on them. **R8
puts all swap models in training.** OOD monitoring uses held-out
identities + external sources (YouTube, WMA) instead.

### 5. More VCD Real Identities in Training

Previous rounds used only 20% of VCD identities (~27 IDs, 800 frames).
R8 bumps to **40%** (~55 IDs) with no per-identity cap. The model
needs to see diverse webcam quality profiles to stop learning
"smooth = real."

### 6. Augmentation: Video‑Conferencing Conditions

The `vcd_targeted` augmentation preset already includes:
- Real‑sharpening (alpha 0.30–0.70, p=0.60) — push DF40 reals toward VCD's sharpness profile
- Real noise injection (p=0.25, var 5–20) — VCD has 2.3× higher HF noise
- Fake degradation (p=0.15) — break the quality‑label correlation

R8 adds **strong context variation** (enabled via override) to simulate:
- Different webcam auto‑exposure (gamma 80–120)
- Office lighting (brightness/contrast ±0.25)
- Camera quality variation (shift/scale/rotate)

---

## Experiment Matrix — 4 Configs

All fine-tuned from **R6_S1 step-4500** (`s3tx3fk4`, AUC 0.9691, EER 0.0674) —
the best EER checkpoint with maximum remaining plasticity. Step 6000 gained only
+0.0003 AUC while worsening EER (0.0674 → 0.0713), suggesting over-commitment
to the DF40-heavy training distribution that R8 is about to invert.

| Config | Key Hypothesis | DF40 weight | Viso weight | DeepLive weight | VCD IDs | Holdout |
|--------|---------------|-------------|-------------|-----------------|---------|---------|
| **R8_A** | Target-heavy: massive shift toward DeepLive+Viso | 0.2 | 3.0 | 4.0 / 5.0 | 40% | None — all swap models in training |
| **R8_B** | DF40-zero: can we drop DF40 entirely? | **0.0** | 3.0 | 4.0 / 5.0 | 40% | None |
| **R8_C** | Middle ground: moderate rebalance | 0.5 | 2.0 | 3.5 / 4.5 | 40% | None |
| **R8_D** | R8_A + VisoMaster MINIMAL+MODERATE only (exclude STRONG) | 0.2 | 3.0 | 4.0 / 5.0 | 40% | None; tiers=MINIMAL,MODERATE |

### Config Details

#### R8_A — "Target-Heavy" (primary hypothesis)
- `df40_fake: 0.2` (was 0.9) — 4.5× reduction
- `visomaster_fake: 3.0` (was 0.8) — 3.75× increase
- `deeplive_non_enhanced_fake: 4.0` (was 2.4) — 1.7× increase
- `deeplive_enhanced_fake: 5.0` (was 3.0) — 1.7× increase
- `external_real: 1.5` (was 0.6) — 2.5× increase
- `realpool_real: 1.5` (was 1.0) — ensures DeepLive/Viso reals aren't underrepresented
- All 9 VisoMaster swap models in training (add GhostFace-v3, InStyleSwapper256-C, SimSwap512)
- Strong context variation enabled
- 10K steps, LR 5e-5 (fine-tune rate)

**What it tests**: Does massively shifting gradient budget toward
deployment-target fakes fix VisoMaster, without breaking DeepLive?

#### R8_B — "DF40-Zero" (ablation)
Same as R8_A but `df40: enabled: false`. No DF40 data at all — train
purely on DeepLive + VisoMaster + diverse reals.

**What it tests**: Is DF40 signal or noise for the deployment target?
If R8_B matches R8_A, DF40 was diluting the training signal.

#### R8_C — "Moderate Rebalance" (conservative)
- `df40_fake: 0.5` — still present but reduced
- `visomaster_fake: 2.0` — solid upweight without extreme
- `deeplive_non_enhanced_fake: 3.5`
- `deeplive_enhanced_fake: 4.5`
- All other settings same as R8_A

**What it tests**: Is the aggressive rebalancing in R8_A too extreme?
Does the model need some DF40 to maintain a diverse decision boundary?

#### R8_D — "Smart VisoMaster Tiers" (tier-aware)
Same as R8_A but:
- `tiers: ["MINIMAL", "MODERATE"]` — exclude STRONG (too-obvious artifacts teach wrong features)
- Hypothesis: STRONG tier fakes with ugly color blobs train the model on
  artifacts that don't generalize. By removing them, the model focuses on
  the subtle manipulation traces that MINIMAL and MODERATE share.

**What it tests**: Does removing easy-but-misleading STRONG tier samples
improve detection of the harder MINIMAL tier?

---

## Success Criteria

### Primary Metrics (must pass ALL for a config to be considered)

| Metric | Minimum | Target | Notes |
|--------|---------|--------|-------|
| DeepLive TPR (all strategies) | **≥ 95%** | ≥ 97% | Cannot regress from R6 |
| DeepLive Enhanced TPR | **≥ 93%** | ≥ 96% | Enhanced fakes are deployment-critical |
| VisoMaster overall fake TPR | **≥ 75%** | ≥ 82% | Was 69% |
| VisoMaster MINIMAL tier TPR | **≥ 55%** | ≥ 65% | Was 44% — THE key metric |
| VCD real accuracy (1-FPR) | **≥ 78%** | ≥ 85% | Was ~70% |
| YouTube real accuracy | **≥ 88%** | ≥ 92% | Was ~85% |

### Secondary Metrics (informational, acceptable regression)

| Metric | Acceptable floor | Notes |
|--------|-----------------|-------|
| DF40 in-dist AUC | ≥ 0.93 | Accept regression — not the deployment target |
| WMA enhanced fake TPR | ≥ 40% | Improvement over 21% baseline welcome but not gated |

### Regression Gates (auto-stop)

If at any checkpoint (2K-step eval):
- DeepLive TPR drops below 90% → early stop
- VCD real accuracy drops below 65% → early stop

---

## Training Details

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base checkpoint | R6_S1 step-4500 (`s3tx3fk4`, AUC 0.9691, EER 0.0674) | Best EER, more plastic than step-6000 (avoids extra 1500 steps of DF40 over-specialization) |
| Learning rate | 5e-5 | Fine-tune rate (not scratch 2e-4) |
| Steps | 10,000 | Longer than R7's 5K to let rebalanced weights converge |
| LR scheduler | cosine_with_warmup, 500 warmup | Gentle start |
| ArcFace | s=10→18, m=0.0 | Same as R6 (proven) |
| Augmentation | quality_targeted_family / vcd_targeted | Proven preset |
| Context variation | **strong** (gamma 80-120, bright/contrast 0.25, OneOf 0.30) | VC lighting diversity |
| Batch | 32 frames, 8 per video | Same as R6 |
| Eval every | 500 steps | Tight monitoring |
| OOD monitoring | Every 1000 steps from step 500 | Early signal |

---

## What We're NOT Doing (and why)

1. **Not training from scratch** — R5 proved lineage matters, time is tight
2. **Not chasing DF40 AUC** — accepting regression on a non-deployment metric
3. **Not using GRL/DANN** — broken in R6, data diversity beat it, too complex for tight timeline
4. **Not adding new augmentation pipelines** — existing vcd_targeted + strong context variation is sufficient
5. **Not holding out VisoMaster methods for OOD** — we're past exploration, we need to train on our weakest methods
6. **Not label smoothing** — adds complexity, R8 is about data composition not loss tricks

---

## Execution Plan

```
1. Create 4 YAML configs (this commit)
2. Run smoke test with R8_A (1K steps, ~30 min)
3. If smoke passes, launch all 4 on Vertex AI overnight
4. Morning analysis: compare on primary metrics
5. Best config → deployment candidate
```
