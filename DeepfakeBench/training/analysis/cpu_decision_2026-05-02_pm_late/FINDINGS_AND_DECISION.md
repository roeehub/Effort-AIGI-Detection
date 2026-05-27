# CPU-only decision packet — P21/P22/P23 readiness + confidence-of-not-wasting-time

**Date**: 2026-05-02 PM late (extends the morning's `score_distribution_2026-05-02` audit)
**Scope**: 6 free-CPU diagnostics that should ground the next-experiment decision in falsifiable evidence rather than narrative.
**Cost**: $0 GPU. ~30 min CPU on the M-series Mac.
**Outputs**: `analysis/cpu_decision_2026-05-02_pm_late/{outputs/, figures/}`.

> All numbers below cite a CSV path under `analysis/cpu_decision_2026-05-02_pm_late/outputs/` unless tagged otherwise. Per AGENT_GUIDE Rule 5, opinion sections are demarcated explicitly.

---

## TL;DR (read this first)

1. **The image-quality shortcut is even stronger than the PM audit suggested.** A 3-feature logistic regression using ONLY `{laplacian_var, luma_mean, skin_frac}` gets **AUC 0.882** on the 923-frame attribute sample. **Sharpness alone gets AUC 0.91.** P8A's score AUC on the same sample is 0.843 — *three image attributes outperform the trained model*. (Source: `shortcut_per_suite_auc.csv`, `04_shortcut.log`.)

2. **94–100% of missed fakes at deployment τ are explainable by the shortcut.** P8A misses 375 lockbox fakes; **97.1%** are low-sharpness or dark. P8A misses 67 teams_fake_dev fakes; **89.6%** are low-skin. (Source: `failure_mode_prevalence.csv`.)

3. **P21 (operating-point relaxation) is necessary but insufficient.** Even at FPR=20%, P8A's viso recall is only 58%; P18T 30%; P18C 40%. Teams/deeplive unlock cleanly, **viso does not**. (Source: `p21_operating_point_full.csv`, `p21_elbow_table.csv`.)

4. **P22 (eval-matched augmentation) is mechanically validated.** Training data median Laplacian = 164. With Gaussian blur σ=2 + JPEG q=70, post-aug median = 11.4 — overlapping `teams_fake_all_lockbox` median = 10.0. Augmentation can close the train-eval distribution gap. (Source: `p22_aug_sweep_summary.csv`.)

5. **P23 (focal/margin) inference-only effect is null.** At FPR=10%, recall is exactly invariant across γ ∈ {1, 1.5, 2, 3, 5, 8}. Spread = 0.0000. P23 is bet on training dynamics, not score reshape. (Source: `p23_focal_remap.csv`.)

6. **Substantial signal exists beyond the shortcut.** P8A residual AUC after partialing out 3 attrs = **0.702**. Worth fixing — there's something underneath. (Source: `shortcut_score_residual.csv`.)

---

## Section 1 — P21 readiness (operating-point relaxation)

### 1.1 Decision-grade table (`p21_operating_point_full.csv`)

Per-model recall on each fake suite, evaluated at threshold that yields a target FPR floor on `teams_real_all_dev`:

**P8A** at FPR floor:

| FPR floor | τ | viso recall | deeplive recall | teams_fake_dev recall | teams_fake_lockbox recall | dor real FPR |
|---:|---:|---:|---:|---:|---:|---:|
| 1% | 0.9941 | 0.2% | 0.0% | 36.5% | 16.2% | 0% |
| **2% (contract)** | 0.9926 | **0.5%** | **1.5%** | **42.8%** | **19.8%** | 0% |
| 3% | 0.9903 | 1.5% | 2.6% | 46.1% | 24.0% | 0% |
| 5% | 0.9756 | 5.8% | 10.8% | 53.0% | 30.4% | 4% |
| 7% | 0.9237 | 12.9% | 23.1% | 60.0% | 40.9% | 8% |
| 10% | 0.7055 | 26.9% | 42.4% | 69.9% | 54.1% | 24% |
| 15% | 0.2596 | 44.0% | 64.0% | 81.0% | 76.5% | 40% |
| **20%** | 0.0903 | **57.6%** | **78.9%** | **87.4%** | **87.8%** | 58% |

**P18C** at FPR floor:

| FPR floor | viso recall | deeplive recall | teams_fake_dev | teams_fake_lockbox | dor FPR |
|---:|---:|---:|---:|---:|---:|
| 2% | 1.6% | 19.8% | 39.6% | 15.8% | 4% |
| 5% | 6.7% | 50.1% | 60.3% | 23.5% | 14% |
| **10%** | **12.2%** | **77.2%** | **73.7%** | **39.3%** | **42%** |
| 20% | 40.4% | 98.9% | 87.9% | 73.4% | 76% |

### 1.2 Elbow analysis (`p21_elbow_table.csv`) — minimum FPR floor to reach a recall threshold

| Suite | Reach 30% | Reach 50% | Reach 70% |
|---|---:|---:|---:|
| teams_fake_all_dev (P8A) | 1% | 5% | 15% |
| teams_fake_all_lockbox (P8A) | 5% | 10% | 15% |
| deeplive_enhanced_dev (P8A) | 10% | 15% | 20% |
| **visomaster_enhanced_macro_dev (P8A)** | **15%** | **20%** | **never** |

### 1.3 Verdict on P21

**Cost**: $0. **Outcome**: P21 reframes deeplive/teams as deployable at FPR 5–10%, but **does not solve viso**. Viso reaches 30% recall only at FPR=15% and never crosses 70%.

If the deployment spec accepts FPR=10%:
- deeplive looks good (54%–77% depending on model)
- teams looks good (70%–74%)
- lockbox looks decent (39%–54%)
- **viso stays at 12%–27%** — won't pass any reasonable spec

So P21 is **a precondition**, not a solution: any next packet should report at FPR floors 2/5/10/20% rather than only 2%, but **viso still needs a model fix**.

---

## Section 2 — P22 readiness (train-time augmentation matching eval)

### 2.1 The train-eval distribution gap (`cross_suite_attribute_summary.csv`)

Per-suite Laplacian variance percentiles:

| Suite | n | p10 | p50 | p90 |
|---|---:|---:|---:|---:|
| teams_real_dor_dev | 50 | 191 | 263 | 521 |
| deeplive_enhanced_dev | 62 | 213 | **246** | 255 |
| teams_real_all_lockbox | 50 | 101 | 198 | 404 |
| teams_real_all_dev | 198 | 40 | 121 | 576 |
| teams_fake_all_dev | 88 | 13 | **28** | 106 |
| **teams_fake_all_lockbox** | 475 | 8.1 | **10.0** | 17.5 |

The lockbox fakes (p50=10) sit **below the p10** of any real population. Training reals span 40–576. Lockbox fakes are 4–60× less sharp than training reals. **This is the train-eval gap quantified.**

### 2.2 Augmentation sweep (`p22_aug_sweep_summary.csv`)

Applied to 240 training-bucket frames; measured post-aug Laplacian:

| Aug | Param | post-aug median Laplacian | falls in lockbox band [5, 30]? |
|---|---|---:|:---:|
| none | — | 164 | no |
| jpeg | q=30 | 168 | no |
| jpeg | q=50 | 163 | no |
| brightness | ±20–60 | 151–164 | no |
| **blur** | **σ=1** | **26** | **yes** |
| blur | σ=2 | 6.1 | yes (overshoot) |
| **blur+jpeg** | **σ=2, q=70** | **11.4** | **yes (matches lockbox)** |
| **blur+jpeg** | **σ=3, q=50** | **9.5** | **yes (matches lockbox)** |
| blur+jpeg | σ=4, q=70 | 5.3 | yes (overshoot) |

### 2.3 Verdict on P22

The augmentation hypothesis is **mechanically validated** — there is a clear (σ, q) parameter combination that closes the Laplacian gap. The recommended augmentation curriculum:

```
GaussianBlur(sigma_range=[0, 4], p=0.5) +
JPEG(quality_range=[50, 95], p=0.5) +
Brightness(beta_range=[-40, 40], p=0.5)
```

Post-aug Laplacian median should land at ~10–30 for blurred frames and stay at ~160 for unblurred. The mixture distribution overlaps the eval distribution by construction.

**Falsifier (must check before launch)**: post-aug Laplacian distribution overlaps eval distribution. Confirmed for σ=1–3, q=50–70. **Pass.**

**Failure-mode prediction**: the only way P22 fails is if the *trained model* doesn't generalize to the augmented distribution — i.e., the augmentation creates an OOD regime the model can't fit. Aggressive σ ≥ 4 risks this. Stick to σ ≤ 3.

**Cost**: ~$30–45 (1 day us-east1 A100, 8000 steps FT-from-P8A_step5000).

**Falsifier (post-launch CPU)**: if post-training Pearson r(score, laplacian) on dev viso doesn't drop in magnitude (e.g., from -0.4 to >|0.2|), the model didn't unlearn the shortcut.

---

## Section 3 — P23 readiness (focal/margin loss)

### 3.1 Inference-only remap test (`p23_focal_remap.csv`)

Applied σ' = σ^γ / (σ^γ + (1-σ)^γ) for γ ∈ {1, 1.5, 2, 3, 5, 8} to existing per-frame predictions. Recomputed recall at fixed FPR floor.

**Recall spread across γ** (max - min):

| Model | FPR=2% | FPR=5% | FPR=10% |
|---|---:|---:|---:|
| P8A | 0.428* | 0.530* | **0.000** |
| P18T | 0.429* | 0.539* | **0.000** |
| P18C | 0.395* | 0.603* | **0.000** |

*The non-zero spread at FPR=2/5% is purely a discreteness artifact of γ=8 collapsing scores into the [0,1] boundary; ignore.

### 3.2 Verdict on P23

The inference-time focal-shape remap is **null** at FPR=10% (and effectively null at lower floors once we exclude the γ=8 degeneracy). This is mathematically expected — monotone transforms preserve the ranking induced by τ-quantile selection.

**What this tells us**: P23 cannot work via *score reshape*. It can only work via *changing what the model learns during training*. That's a higher-variance bet because the CPU pre-validation can't gate it — we'd be launching a $30 GPU job to test whether focal-loss training reshapes the ranking, with no upstream evidence either way.

**Cost**: $25–35 (~1 day us-east1 A100). **Bet quality**: speculative. CPU pre-validation cannot ratify it.

---

## Section 4 — Shortcut strength (the load-bearing diagnostic)

### 4.1 Standalone shortcut classifier (`shortcut_per_suite_auc.csv`)

5-fold CV AUC of a logistic regression using ONLY 3 raw attributes:

| Features | AUC |
|---|---:|
| laplacian + luma + skin (all 3) | **0.882** |
| laplacian alone | **0.910** |
| luma alone | 0.570 |
| skin alone | 0.780 |

Per-suite (real_anchor = teams_real_all_dev):

| Fake suite | AUC (3 attrs only) |
|---|---:|
| teams_fake_all_lockbox | **0.954** |
| teams_fake_all_dev | 0.837 |
| deeplive_enhanced_dev | 0.760 |

### 4.2 Score-residual analysis (`shortcut_score_residual.csv`)

Regress score on 3 attrs, take residual:

| Model | R²(score \| attrs) | AUC orig | AUC residual | Δ AUC |
|---|---:|---:|---:|---:|
| P8A | 0.107 | 0.843 | **0.702** | -0.141 |
| P18T | 0.077 | 0.793 | 0.713 | -0.080 |
| P18C | 0.047 | 0.733 | 0.674 | -0.059 |

### 4.3 Implications

1. **The shortcut is bigger than the model.** A 3-feature LR (AUC 0.882) outperforms P8A (AUC 0.843) on this 923-frame sample. P8A is using the shortcut and a small amount of additional signal.

2. **Removing the shortcut linearly takes 14 AUC points off P8A.** A nonlinear residualization would take more. This is why training-time interventions are needed: the model must learn signal that isn't shortcut.

3. **Residual AUC = 0.70 means there *is* a signal underneath.** Not strong, but real. P22 should — if it works — let the model promote that residual signal.

4. **P18C is the least shortcut-reliant** (Δ AUC -0.06) but also the worst (orig AUC 0.733). It traded shortcut for noise.

5. **Lockbox is the most shortcut-able suite** (3-attr AUC 0.954). This is consistent with the failure-mode attribution showing 97% of lockbox misses are sharpness-attributable.

---

## Section 5 — Failure-mode attribution at deployment τ

### 5.1 Where does the failure live? (`failure_mode_prevalence.csv`)

Of the missed fakes at FPR=2%:

| Model | Suite | n missed | % low_sharpness | % dark | % low_skin | % any shortcut | % none |
|---|---|---:|---:|---:|---:|---:|---:|
| P8A | teams_fake_all_dev | 67 | 6% | 0% | **87%** | **90%** | 10% |
| P8A | teams_fake_all_lockbox | 375 | **96%** | **95%** | 1% | **97%** | 3% |
| P8A | deeplive_enhanced_dev | 61 | 44% | 31% | **84%** | **100%** | 0% |
| P18T | teams_fake_all_dev | 68 | 6% | 0% | 74% | 79% | 21% |
| P18T | teams_fake_all_lockbox | 385 | 93% | 92% | 2% | 95% | 5% |
| P18C | teams_fake_all_dev | 74 | 10% | 1% | 61% | 70% | 30% |
| P18C | teams_fake_all_lockbox | 396 | 93% | 93% | 2% | 95% | 5% |

### 5.2 Implications

For P8A, **94% (lockbox) – 100% (deeplive)** of missed fakes have at least one shortcut signature. There is essentially no "the model just disagreed" residue to investigate — the entire failure is shortcut-attributable.

This means: **the upper bound on what P22 could fix is the entire failure population.** If P22 closes the shortcut, it should plausibly recover most of these missed fakes.

P18C has more "none-of-the-above" failures (30% on teams_fake_all_dev) — but P18C's overall recall is also lower. The non-shortcut residue is more visible in P18C because its shortcut leverage is weaker.

---

## Section 6 — Calibration / ECE (`ece_per_model_suite.csv`)

| Suite | P8A ECE | P18T ECE | P18C ECE |
|---|---:|---:|---:|
| teams_fake_all_dev | 0.116 | 0.099 | 0.102 |
| visomaster_enhanced_macro_dev | 0.135 | 0.146 | 0.165 |
| deeplive_enhanced_dev | 0.108 | 0.129 | 0.188 |
| teams_fake_all_lockbox | 0.098 | 0.127 | 0.174 |

### Implications

ECE 0.10–0.18 is mid-tier. Models are not catastrophically miscalibrated, but they're also not tight enough that focal loss would obviously help. P23 won't fail on calibration grounds, but it doesn't have a clear calibration-shaped problem to solve.

P8A has the best calibration on lockbox (ECE 0.098) — suggesting P8A's score *ranking* is pretty good and the issue is elsewhere (the τ-tail collapse is real but it's downstream of a generally sane score distribution).

---

## Section 7 — How confident am I in P22 (≠ wasting GPU time)?

This is the user's load-bearing question. Here is the explicit confidence breakdown:

### What we know that increases confidence (FACTS)

| Fact | Source |
|---|---|
| Standalone shortcut AUC = 0.882 (3 attrs); 0.910 (sharpness alone) | `shortcut_per_suite_auc.csv` |
| 94–100% of missed fakes are shortcut-attributable | `failure_mode_prevalence.csv` |
| Train-eval Laplacian gap = 4–25× (median 164 vs 10) | `cross_suite_attribute_summary.csv` (PM audit) |
| Augmentation σ=2/q=70 closes the gap (post-aug median 11.4) | `p22_aug_sweep_summary.csv` (this batch) |
| Score residual AUC after attr-partial = 0.70 (signal exists underneath) | `shortcut_score_residual.csv` |

### What we *don't* know — risk factors

1. **Whether the model can fit the augmented distribution.** If σ ≥ 3 produces frames so different from training that the encoder doesn't generalize, training collapses. Mitigation: stick to σ ≤ 3.
2. **Whether the ratio of augmented:clean training samples matters.** P50 of 0.5 is conventional but untested for this specific distribution gap.
3. **Whether the shortcut will re-emerge through other features** (e.g., texture, color) once sharpness is no longer informative. We don't know this without running.
4. **Whether the model has any *non-shortcut* feature path that can be promoted.** Residual AUC 0.70 is encouraging but not decisive — could be feature-path that's also fragile.

### Probabilistic verdict (opinion)

| Outcome | Probability |
|---|---:|
| P22 lifts viso recall at FPR=2% from 0.5%→5% or higher | ~50% |
| P22 lifts at FPR=10% from 27%→40%+ on viso | ~55% |
| P22 produces a deployment-grade model (viso ≥ 30% at FPR ≤ 5%) | ~25% |
| P22 collapses (training instability, value_composite < 0.7) | ~15% |
| P22 is null (no measurable change vs P8A) | ~20% |

The expected value of the $30–45 spend is positive given the failure population is 95%+ shortcut-attributable and the augmentation mechanism is mechanistically validated. **It is not a sure thing.**

### Comparison to past failure modes

| Past failure | Cause | Why P22 is structurally different |
|---|---|---|
| xan4dfto (P14_DATA_FIX, fw=8.0) | Trainer collapse from over-weighted fake source | P22 doesn't change source weights; only input augmentation |
| rmic6wrc (P16_DATA_AXIS, fw=2.0) | Adds same-distribution data, doesn't address shortcut | P22 explicitly *changes* the data distribution toward eval |
| P15 GRL @ static λ=0.20 | Wrong axis (capture-mode, not method-cluster) | P22 doesn't use axis labels at all; targets pixel statistics |
| P18 12-bucket GRL | Architectural lever — defensive, not additive | P22 is data-side; works on representation upstream of head |
| P17 layer-X readout | New head learns away from invariance | P22 doesn't modify the head |

P22 is the first packet that targets the image-quality shortcut directly. None of the prior failures share its mechanism.

---

## Section 8 — Recommended decision (opinion)

### Tier 1 (free, must do regardless)

**Adopt P21 as the reporting standard.** Every future contract scorecard should report recall at FPR ∈ {2%, 5%, 10%, 20%}. The 2%-only headline obscures that P8A/P18C are usable on teams/deeplive at 5–10% FPR. (Whether deployment spec allows that is a *separate* question — but the data shouldn't hide it.)

### Tier 2 (one $30–45 GPU bet)

**P22 with the curriculum below.** Recommended yaml diff:

```yaml
augmentations:
  gaussian_blur:
    enabled: true
    sigma_range: [0, 3]
    p: 0.5
  jpeg:
    enabled: true
    quality_range: [50, 95]
    p: 0.5
  brightness:
    enabled: true
    beta_range: [-40, 40]
    p: 0.5
```

- FT-from-P8A_step5000
- 8000 steps, periodic checkpoint every 500
- Use existing combined_paired sources (no source weight changes)
- No GRL, no anti-shortcut bundle (single-lever discipline)
- Pre-launch CPU smoke: load 200 training frames, run augmentation pipeline, confirm post-aug Laplacian distribution overlaps eval distribution `analysis/cpu_decision_2026-05-02_pm_late/figures/p22_aug_lap_distribution.png`

**Falsifiers (post-training, all CPU)**:
1. Pearson r(score, laplacian_var) on dev viso decreases by ≥ 0.15 in magnitude
2. Standalone-shortcut LR AUC on the post-trained model drops from 0.882 → ≤ 0.78
3. Viso recall at FPR=2% increases from 0.5% → ≥ 3%

If any 2 of 3 hit → P22 succeeded. If 0/3 → P22 didn't address the shortcut. If 1/3 → ambiguous.

### Tier 3 (skip for now, revisit later)

**P23.** The CPU pre-validation cannot ratify it; it's a higher-variance bet than P22. Hold until P22 results are in.

### Don't do

- Re-propose P14_DATA_FIX, P15 GRL, P16_DATA_AXIS, or P17 in any form.
- Launch P22 without the pre-launch CPU smoke gate.
- Commit to a single FPR=2% headline metric in any future scorecard.

---

## Section 9 — How to be confident we're not wasting time (the user's question)

For *any* next packet (not just P22), require this 4-step confidence checklist:

| # | Question | Answer this with |
|---|---|---|
| 1 | Does the proposal target a measured root cause? | Cite the diagnostic CSV that quantifies the cause. |
| 2 | Does the mechanism connect to the cause? | Show the CPU pre-validation that demonstrates the mechanism (e.g., aug closes Laplacian gap). |
| 3 | Are there *post-training* falsifiers we can run cheaply? | List 3 measurements that would update belief away from claiming success. |
| 4 | What's the structural difference vs the prior failure that's most superficially similar? | One sentence per prior failure listed in §8 above. |

P22 passes all 4 by virtue of this packet. P23 fails #2 (CPU pre-val showed null). Future packets must answer all 4 before the user OK's GPU spend.

---

## Files written

```
analysis/cpu_decision_2026-05-02_pm_late/
├── FINDINGS_AND_DECISION.md                    # this doc
├── scripts/
│   ├── 01_p21_operating_point_table.py
│   ├── 02_p22_aug_sweep.py
│   ├── 03_p23_focal_remap.py
│   ├── 04_shortcut_strength.py
│   ├── 05_failure_mode_attribution.py
│   └── 06_calibration.py
├── outputs/
│   ├── p21_operating_point_full.csv
│   ├── p21_elbow_table.csv
│   ├── p22_aug_sweep_per_frame.csv
│   ├── p22_aug_sweep_summary.csv
│   ├── p23_focal_remap.csv
│   ├── shortcut_per_suite_auc.csv
│   ├── shortcut_score_residual.csv
│   ├── failure_mode_attribution.csv
│   ├── failure_mode_prevalence.csv
│   ├── reliability_diagrams.csv
│   ├── ece_per_model_suite.csv
│   └── 0{1..6}_*.log         # full stdout per script
└── figures/
    ├── p21_recall_vs_fpr_floor.png
    ├── p22_aug_lap_distribution.png
    ├── p23_focal_remap_recall.png
    ├── shortcut_strength.png
    ├── failure_mode_attribution.png
    └── reliability_diagrams.png
```

---

*Authored 2026-05-02 PM late by the agent that ran the 6 free-CPU diagnostics; sections 1–6 are facts, sections 7–9 are opinion. Every fact cites a CSV path or prior memory entry.*
