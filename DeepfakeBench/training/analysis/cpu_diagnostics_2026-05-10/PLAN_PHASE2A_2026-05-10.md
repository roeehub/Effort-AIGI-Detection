# Phase 2A Synthesis — pre-test outcomes route the next experiment
> **Date**: 2026-05-10 14:45
>
> **Predecessor**: PLAN_2026-05-10.md proposed three CPU pre-tests to determine
> which Path (β1 multi-axis GRL, δ1 forgery-localization, γ1 curriculum) the
> empirical evidence supports. This doc reports the pre-test outcomes.
>
> **Companion files**:
> - `outputs/pretest1_multiaxis_grl.csv` (60 rows: 4 ckpts × 3 k × 5 λ)
> - `outputs/pretest3_attention_iou.csv` (4 ckpts)
> - Pre-test 2 was killed at 26 min due to MPS contention; conclusion derived from existing atlas

---

## TL;DR

**Multi-axis GRL has a measurable but bounded lever effect.** On frozen P8A L11 features, multi-axis GRL with compression (k=128) lifts inv_mean from baseline 0.012 to **0.048** at λ=2.0 — a **4× relative lift** but still **half the 0.10 target**. The mechanism works directionally but the achievable ceiling on frozen features is ~0.05, not 0.10.

**Attention is already face-localized across all ckpts.** P8A IoU 0.63, T3 0.59, E2B 0.53. Forgery-localization auxiliary loss has moderate room (could push to 0.85+ with refinement) but the model is NOT in a "uses background features" state — Path δ is refinement, not a structural fix.

**Path β1 (from-scratch + multi-axis GRL) is partially justified — but with tempered expectations.** Frozen-features ceiling is 0.05; from-scratch might reach 0.08-0.12 (encoder has more flexibility); above 0.15 is unlikely on the data we have.

**The most promising path is a HYBRID**: from-scratch + multi-axis GRL + forgery-localization auxiliary loss. The two mechanisms are complementary (GRL shapes feature axes, localization shapes spatial attention).

---

## Pre-test 1 — Multi-axis GRL feasibility

(60 configs: 4 ckpts × {k=128, 256, 768} × {λ=0, 0.5, 1.0, 2.0, 5.0}; ~55 min runtime)

### Headline numbers

Best inv_mean across all λ per (ckpt, k):

| ckpt | k=128 | k=256 | k=768 | baseline (raw) |
|---|---:|---:|---:|---:|
| **P8A** | **+0.048** (λ=2.0, forgery=1.000) | +0.032 | +0.012 | +0.012 |
| MCLIOEXB | +0.044 (λ=2.0) | +0.029 | +0.018 | +0.018 |
| T3_step1500 | +0.029 (λ=0.0) | +0.016 | +0.022 | +0.022 |
| E2B | +0.024 (λ=0.0) | +0.023 | +0.018 | +0.018 |

**Baseline-vs-best lift**:
- P8A: 0.012 → 0.048 (**+0.036**, 4× relative lift)
- MCLIOEXB: 0.018 → 0.044 (+0.026, 2.4× lift)
- T3_step1500: 0.022 → 0.029 (+0.008, 1.3× lift)
- E2B: 0.018 → 0.024 (+0.006, 1.3× lift)

### Reading

**1. The mechanism works directionally.** Adding multi-axis GRL pressure with compression DOES increase inv_mean. Up to 4× relative on P8A. This is real signal, not noise.

**2. The ceiling on frozen features is ~0.05.** Even with optimal (k=128, λ=2.0) tuning, no ckpt's frozen features support inv_mean > 0.05. The 0.10 target was set higher than what's achievable from frozen features.

**3. Compression matters; full-rank doesn't.** k=768 (no compression) → zero lift across all (ckpt, λ). This is a sanity check: the GRL mechanism can only suppress shortcuts by *throwing away dimensions*. A full-rank projection preserves all info.

**4. P8A and MCLIOEXB benefit MORE than E2B and T3.** P8A's L11 has the most "shortcut-separable" subspace; E2B and T3 have features more entangled. This suggests the from-scratch encoder direction matters — P8A's training trajectory created somewhat-separable subspaces that the GRL can find; E2B's CE-only training created tightly-entangled subspaces.

**5. Excessive λ hurts.** At λ=5.0, forgery_AUC starts to drop. The GRL pressure cannibalizes forgery signal. Sweet spot is λ ∈ [0.5, 2.0].

### What this implies for from-scratch GRL training

The pre-test was on FROZEN encoder features with a learnable projection. From-scratch training has the ENCODER itself as trainable, with more degrees of freedom. The achievable ceiling is plausibly higher than 0.05 because:

- The encoder can find DIFFERENT L11 directions during training, not just project from existing ones
- GRL pressure acts throughout training, shaping the encoder representation step by step
- Compression IS available naturally at the bottleneck before the head

Conservative expectation: 0.08-0.12 inv_mean from from-scratch + multi-axis GRL.
Optimistic: 0.15+ if encoder finds genuinely separable subspaces.
Pessimistic: ~0.05 (same ceiling as frozen) if the data fundamentally entangles.

**No way to know without running it.** The pre-test increased confidence the mechanism works, but did not establish a 3× achievable ceiling.

---

## Pre-test 2 — L6 vs L11 head (skipped, conclusion from atlas)

Pre-test was killed at 26 min due to MPS contention with pre-test 1. Conclusion derived from `outputs/job_h_invariance_trajectory.csv`:

| ckpt | L6 inv_mean | L11 inv_mean | Δ |
|---|---:|---:|---:|
| P8A | 0.024 | 0.027 | +0.003 |
| E2B | 0.026 | 0.020 | -0.006 |
| T3_step1500 | 0.025 | 0.024 | -0.001 |
| T3_step2500 | 0.026 | 0.021 | -0.005 |

**Within probe noise** (~0.005-0.010 SD). Reading from L6 instead of L11 doesn't materially change invariance. The architectural variant "head reads from L6" is NOT a structural fix.

(Forgery_AUC at L6 is 0.99+ for all ckpts — L6 is already saturated on real-vs-fake. The shortcut leakage is similar at L6 and L11.)

---

## Pre-test 3 — Attention IoU vs face region

(4 ckpts × ~99 frames; ~10 min runtime)

### Headline numbers

| ckpt | REAL face IoU | FAKE face IoU | face_attn_mass FAKE | centroid in face FAKE |
|---|---:|---:|---:|---:|
| **P8A** | **0.544** | **0.631** | 0.712 | 100% |
| T3_step1500 | 0.524 | 0.591 | 0.682 | 100% |
| T3_step2500 | 0.511 | 0.594 | 0.669 | 100% |
| **E2B** | **0.496** | **0.525** | 0.639 | 100% |

### Reading

**1. All ckpts already attend to the face region.** Centroid lands inside the face bbox for ~100% of fake frames. The attention is NOT scattered across background.

**2. Face_attn_mass on fakes is 64-71%.** Most of the model's attention is on the face for fake classification. This validates that the attention is centered on the right region — just not as tightly bounded as the bbox.

**3. P8A has the tightest face localization** (IoU 0.63 on fakes). Combined with its highest face_attn_mass (0.71), P8A is the most face-focused of the 4 ckpts. This correlates with P8A's substrate-invariance signature — focusing on face features rather than global IQ.

**4. E2B has the loosest** (IoU 0.53). Combined with E2B's highest shortcut leakage in atlas, this is consistent: E2B uses more global/background features.

**5. T3 ckpts sit between.** T3 step1500 → step2500 progression shows decreasing face IoU (0.591 → 0.594... actually similar). T3's face localization is roughly P8A-class.

### What this implies for forgery-localization aux loss

The model is **already** doing face-attention. The auxiliary loss would refine SHAPE within the face region (toward swap-specific subregion: eyes/nose/mouth) rather than redirect from background.

Path δ has **moderate room**:
- IoU 0.63 → 0.85+ is plausible with attention-guidance loss
- But this is REFINEMENT, not a structural redirection
- Less of a game-changer than I initially framed in PLAN_2026-05-10.md

---

## Combined reading: which path is GPU-justified?

### Path β1 (from-scratch + multi-axis GRL): partially justified, tempered expectations

- Frozen-features ceiling: 0.05 inv_mean (Pre-test 1)
- Realistic from-scratch ceiling: 0.08-0.12 (encoder has more flexibility)
- Sweet-spot config: k=128 (compression), λ ∈ [0.5, 2.0]
- Cost: ~$100-150 GPU
- Risk: might not break 0.10 target if data fundamentally entangles forgery + shortcut

### Path δ1 (forgery-localization aux loss): refinement, not game-changer

- Current attention already face-centered (centroid 100%, IoU 0.63 on fakes)
- Aux loss could refine to IoU 0.85+ but not change WHAT the model uses, only HOW LOCALIZED
- Cost: ~$120-180 GPU + mask-pipeline infra
- Risk: small absolute lift in deployment metrics; mostly improves localization not robustness

### Path β1+δ1 hybrid (RECOMMENDED): combined mechanisms with complementary effects

The two mechanisms are orthogonal:
- GRL shapes which feature DIMENSIONS are extractable (axis-decorrelation)
- Localization shapes which SPATIAL REGIONS the attention concentrates on

A hybrid recipe could plausibly achieve:
- inv_mean 0.10+ at L11 (GRL contribution)
- IoU 0.85+ on swap region (localization contribution)
- AND maintain forgery_AUC > 0.97

**Recipe sketch**:
- B16 fresh init from CLIP-DataComp-XL
- CE loss on real/fake at head
- 5× GRL classifiers attached at L11 features for {is_dor, is_chronic, lap_var_q, min_dim_q, color_a_q}, λ_grl ∈ [0.5, 2.0]
- Forgery-localization aux loss: KL(L11_attention, swap_region_mask) for frames with known swap regions
- Hidden bottleneck at L11 (compression to k=128 if architecturally feasible, otherwise full 768 with GRL only)

**Cost**: ~$150-200 GPU + mask-pipeline infra.

**Close criteria** (revised based on Phase 2A):
1. inv_mean ≥ 0.05 at L11 on triptych (matches frozen-features ceiling — minimum)
2. forgery_AUC ≥ 0.97 at L11 (within 0.02 of P8A/E2B baseline)
3. is_chronic_6 AUC ≤ 0.90 at L11 (vs P8A 0.909 — slight encoder-level chronic decoupling)
4. Mean attention-vs-swap-region IoU ≥ 0.75 on fakes (vs current P8A 0.63)
5. F4 deeplive recall ≥ 90% at FPR=10% (matches P8A baseline)
6. F4 viso recall ≥ 70% at FPR=10% (within step1500's range)
7. ≤ 6/92 may6 false-flag rate (matches step1500 floor)

If hybrid hits 5/7+ → it's the candidate. If 3-4/7 → refinement, iterate. If <3/7 → mechanism reading was wrong, revisit.

---

## Risk-adjusted alternative: lower-cost focused experiment

If the hybrid is too ambitious (mask-pipeline infra cost + uncertain λ tuning), a sequential 2-stage approach:

**Stage A — From-scratch + multi-axis GRL ONLY** (~$100): test if pure-GRL gets inv_mean above 0.05 in from-scratch setting.

If Stage A hits inv_mean ≥ 0.07 → proceed to Stage B (add localization aux loss for further refinement).

If Stage A misses → mechanism is GRL-bounded; pivot to curriculum-from-scratch (Path γ1) or accept current ceiling.

**Stage B — Add forgery-localization aux loss** (~$70): refines the Stage A model's spatial attention.

Total: $170 split into two checkpoints. Each is independently informative.

---

## What's UNCHANGED from PLAN_2026-05-10.md

1. **E2B retire urgent** (multi-front empirical refutation; deployment must switch)
2. **Ship P8A primary or step1500 shadow** (deployment recommendation; independent of research)
3. **Defer T4 face_scale_jitter** (refuted by MCLIOEXB result — same face_size_high AUC as P8A)
4. **HDTF promotion-contract bug fix scorecard** (~$15-20 small verification)

---

## Recommendation

Authorize:
1. **Hybrid β1+δ1 GPU experiment** (~$150-200) with the 7 close criteria above. Best expected outcome.
2. **OR** sequential Stage A + Stage B (~$170 split) with Stage B contingent on Stage A inv_mean ≥ 0.07. More cautious; gates the second spend on the first's outcome.

I lean toward the sequential approach — Stage A is the cleanest test of whether multi-axis GRL alone is GPU-worthy. If Stage A clears the 0.07 bar, Stage B's marginal value is high. If Stage A misses, we save the Stage B spend.

What I would NOT recommend:
- Path α (head-only multi-axis GRL on frozen P8A): refuted by Pre-test 1 — frozen-features ceiling is 0.05; head-only retrain inherits the same ceiling with no encoder flexibility.
- Pure Path δ (aux loss alone): attention already face-centered; aux loss is refinement.
- Pure Path γ (curriculum without explicit invariance objective): doesn't address the shortcut entanglement.

---

## Self-correction log (Phase 2A)

- **My PLAN target of inv_mean > 0.10 was too aggressive.** Pre-test 1 ceiling is 0.05 on frozen features. Realistic from-scratch target is 0.07-0.12, not 0.10+.
- **My PLAN forgery-localization framing oversold the room.** Pre-test 3 shows attention is ALREADY face-centered (centroid 100%). Aux loss is refinement, not redirection.
- **The hybrid approach (β+δ) wasn't in PLAN_2026-05-10.md** but is the most evidence-supported path now. The two mechanisms are complementary, not redundant.
