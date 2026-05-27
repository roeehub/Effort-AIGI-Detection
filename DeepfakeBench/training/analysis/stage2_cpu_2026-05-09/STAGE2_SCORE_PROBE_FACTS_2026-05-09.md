# Stage 2 score-distribution probe FACTS (2026-05-09)

> **Status: factual-only.** No interpretation. Forbidden words: succeeds,
> fails, wins, promotes, deployment-grade.
>
> Source data: `outputs/stage2_dor_cohort_scores.csv`,
> `outputs/stage2_per_cohort_stats.csv`,
> `outputs/stage2_correlation_vs_p8a.csv` produced by
> `run_score_probe.py` on 2026-05-09 03:21–03:27 local (MPS device,
> 388-frame Dor cohort cached at
> `analysis/dor_encoder_axis_2026-05-08/_cache/`).
>
> Companion docs: existing P2-D Dor encoder-axis FACTS at
> `analysis/dor_encoder_axis_2026-05-08/DOR_ENCODER_AXIS_FACTS_2026-05-08.md`
> (defines the cohort).

## 1. Question

For each of the three Stage 2 slots (S1 REAL_AUG_OFF, S2 LOW_LR_FT,
S3 WEAK_PAIRRANK), does any of step{500, 2500, 4500} reproduce P2-D's
dor-identity-cluster collapse pattern (real-FPR median jumping ~0.02 →
0.27 on `DOR_REAL_LOCKBOX`)?

The 2026-05-08 evening BUNDLE_step500 audit also documented a separate
`Roy_D` 800× score-variance collapse (memory `project_p1_BUNDLE_step500_*`).
This probe checks whether the Stage 2 slots reproduce a similar
identity-conditional collapse on the Dor cohort.

## 2. Method

### 2.1. Cohort

The 388-frame `DOR` cohort from `dor_encoder_axis_2026-05-08`:

| cohort | n | label | source |
|---|---:|---|---|
| DOR_REAL_DEV | 50 | 0 | teams_real_dor_dev |
| DOR_FAKE_DEV | 78 | 1 | teams_capture_dor_shkedi_dev |
| DOR_REAL_LOCKBOX | 100 | 0 | teams_real_all_lockbox ∩ dor_shkedi |
| NON_DOR_REAL_DEV | 80 | 0 | teams_real_all_dev (excl. dor) |
| NON_DOR_FAKE_DEV | 80 | 1 | teams_fake_all_dev (excl. dor) |

### 2.2. Inference

Per ckpt: load via existing detector loader, run forward on each frame
(224×224, CLIP normalization), take fake-class softmax probability. CPU
host (MPS device on Apple M-series). Batch size 32. ~7s per ckpt for 388
frames after model load (~10s for the load).

### 2.3. Reference scores

P8A_REFERENCE_STEP5000, E2B_TOP_N_STEP3200, P2_D_FOURIER_PERIODIC_STEP3000
scores are pre-computed in the cached cohort manifest (run on 2026-05-08).
Stage 2 ckpts S1/S2/S3 × step{500, 2500, 4500} (9 ckpts) scored fresh.

## 3. Per-cohort score median (p50)

(Reading: lower-is-better for `*_REAL_*` cohorts, higher-is-better for
`*_FAKE_*`. P8A is the gold standard.)

| ckpt | DOR_REAL_DEV | DOR_REAL_LOCKBOX | NON_DOR_REAL_DEV | DOR_FAKE_DEV | NON_DOR_FAKE_DEV |
|---|---:|---:|---:|---:|---:|
| **P8A_REFERENCE_STEP5000** | **0.181** | **0.019** | **0.010** | **0.986** | **0.994** |
| E2B_TOP_N_STEP3200 | 0.223 | 0.020 | 0.013 | 0.777 | 0.989 |
| P2_D_FOURIER_PERIODIC_STEP3000 | 0.436 | 0.266 | 0.034 | 0.489 | 0.745 |
| S1_REAL_AUG_OFF step500 | 0.659 | 0.270 | 0.042 | 0.822 | 0.984 |
| S1_REAL_AUG_OFF step2500 | 0.720 | 0.347 | 0.007 | 0.584 | 0.997 |
| S1_REAL_AUG_OFF step4500 | 0.765 | **0.536** | 0.028 | 0.980 | 0.998 |
| S2_LOW_LR_FT step500 | 0.591 | 0.196 | 0.109 | 0.806 | 0.897 |
| S2_LOW_LR_FT step2500 | 0.718 | 0.305 | 0.061 | 0.941 | 0.974 |
| **S2_LOW_LR_FT step4500** | 0.715 | **0.285** | 0.053 | 0.944 | 0.977 |
| S3_WEAK_PAIRRANK step500 | 0.969 | **0.657** | 0.142 | 0.972 | 0.980 |
| S3_WEAK_PAIRRANK step2500 | 0.842 | 0.504 | 0.028 | 0.514 | 0.992 |
| S3_WEAK_PAIRRANK step4500 | 0.906 | 0.517 | 0.021 | 0.582 | 0.996 |

## 4. Per-cohort score std

(Reading: low std = compressed/cluster-collapse signal. P8A `DOR_REAL_DEV`
std=0.347 is the high-discrimination reference. BUNDLE_step500 had
std=0.0005 on Roy_D — 800× compressed.)

| ckpt | DOR_REAL_DEV | DOR_REAL_LOCKBOX | DOR_FAKE_DEV |
|---|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.347 | 0.158 | 0.115 |
| E2B_TOP_N_STEP3200 | 0.294 | 0.172 | 0.207 |
| P2_D_FOURIER_PERIODIC_STEP3000 | 0.176 | 0.166 | 0.124 |
| S1 step500 | 0.296 | 0.237 | 0.194 |
| S1 step2500 | 0.264 | 0.287 | 0.288 |
| S1 step4500 | 0.295 | 0.307 | 0.171 |
| S2 step500 | 0.243 | 0.184 | 0.137 |
| S2 step2500 | 0.269 | 0.270 | 0.129 |
| S2 step4500 | 0.273 | 0.270 | 0.131 |
| **S3 step500** | **0.077** | 0.238 | **0.038** |
| S3 step2500 | 0.217 | 0.265 | 0.200 |
| S3 step4500 | 0.247 | 0.325 | 0.263 |

`S3 step500 DOR_REAL_DEV std = 0.077` is **4.5× compressed vs P8A's 0.347**.
`S3 step500 DOR_FAKE_DEV std = 0.038` is **3.0× compressed vs P8A's 0.115**.
By S3 step2500/4500, std recovers to within 1.4× of P8A.

## 5. Pearson r vs P8A scores per cohort

(Reading: r=1 ⇒ same per-frame ordering as P8A; r=0 ⇒ uncorrelated.)

| cohort | E2B | P2D | S1_500 | S1_2500 | S1_4500 | S2_500 | S2_2500 | S2_4500 | S3_500 | S3_2500 | S3_4500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 0.860 | 0.711 | 0.856 | 0.684 | 0.759 | 0.915 | 0.856 | **0.860** | 0.661 | 0.577 | 0.549 |
| DOR_REAL_LOCKBOX | 0.590 | 0.455 | 0.722 | 0.569 | 0.526 | 0.720 | 0.618 | **0.624** | 0.357 | 0.535 | 0.534 |
| DOR_REAL_DEV | 0.895 | 0.570 | 0.886 | 0.775 | 0.794 | 0.889 | 0.827 | 0.829 | 0.513 | 0.768 | 0.705 |
| DOR_FAKE_DEV | 0.341 | 0.221 | 0.718 | 0.559 | 0.888 | 0.863 | 0.934 | 0.940 | 0.927 | 0.550 | 0.574 |
| NON_DOR_REAL_DEV | 0.541 | 0.681 | 0.818 | 0.809 | 0.792 | 0.915 | 0.826 | 0.825 | 0.707 | 0.797 | 0.711 |
| NON_DOR_FAKE_DEV | 0.827 | 0.802 | 0.941 | 0.953 | 0.910 | 0.957 | 0.894 | 0.904 | 0.847 | 0.921 | 0.838 |

S2_step4500 has the highest `DOR_REAL_LOCKBOX` Pearson r vs P8A among
Stage 2 final ckpts (0.624 vs S1_4500 0.526, S3_4500 0.534, P2D 0.455).

## 6. Trajectory summary per slot

### S1 (REAL_AUG_OFF) — `DOR_REAL_LOCKBOX p50` trajectory
0.270 (step500) → 0.347 (step2500) → **0.536 (step4500)**.
Monotonic degradation. Step4500 is 28× P8A's 0.019.

### S2 (LOW_LR_FT) — `DOR_REAL_LOCKBOX p50` trajectory
0.196 (step500) → 0.305 (step2500) → 0.285 (step4500).
Plateaus by step2500. Step4500 is 15× P8A's 0.019.

### S3 (WEAK_PAIRRANK) — `DOR_REAL_LOCKBOX p50` trajectory
0.657 (step500) → 0.504 (step2500) → 0.517 (step4500).
Highest at step500, partial recovery, plateaus around 0.51. Step4500 is
27× P8A's 0.019.

`DOR_REAL_DEV p50` follows the same pattern but at higher baseline:
P8A 0.181 → S1 0.765 / S2 0.715 / S3 0.906 at step4500.

## 7. Reference table — P8A → P2D regression magnitude per cohort

(For comparison with Stage 2 step4500 ckpts at the same cohort.)

| cohort | P8A | P2D | P2D−P8A | S1_4500−P8A | S2_4500−P8A | S3_4500−P8A |
|---|---:|---:|---:|---:|---:|---:|
| DOR_REAL_LOCKBOX | 0.019 | 0.266 | +0.247 | **+0.517** | +0.266 | +0.498 |
| DOR_REAL_DEV | 0.181 | 0.436 | +0.255 | +0.584 | +0.534 | **+0.725** |
| NON_DOR_REAL_DEV | 0.010 | 0.034 | +0.024 | +0.018 | +0.043 | +0.011 |
| DOR_FAKE_DEV | 0.986 | 0.489 | -0.497 | -0.006 | -0.042 | **-0.404** |
| NON_DOR_FAKE_DEV | 0.994 | 0.745 | -0.249 | +0.004 | -0.017 | +0.002 |

S1_step4500's `DOR_REAL_LOCKBOX` regression magnitude (+0.517) is
**2.1× larger** than P2D's (+0.247). S2_step4500's regression matches
P2D's (+0.266 vs +0.247). S3_step4500 is 2.0× larger than P2D's.

S3 also regresses on `DOR_FAKE_DEV` (Δ=-0.404 vs P8A) — the only Stage 2
ckpt with bidirectional Dor-cohort regression (real-side and fake-side).

## 8. Caveats

- 388-frame cohort. Identity diversity ≠ N=388; small per-cohort sample
  sizes (50-100 per cohort) → CIs not computed here.
- These are CPU/MPS scores from the same image-decode + preprocessing
  pipeline as `dor_encoder_axis_2026-05-08`. Differences from
  trainer's eval-time scores expected at ~1e-4 magnitude (CPU vs CUDA
  numeric).
- This probe measures the dor cohort only. Roy_D (130-frame canary set
  from P1 PE eval) was NOT scored — would require download of the
  Roy_D-cached frames or a fresh download.
- Holdout AUC at training-time was 0.9877–0.9929 across these ckpts;
  per memory `project_train_auc_not_valid_promotion_signal.md` train
  AUC is not deployment-grade. Promotion contract scorecard not run.
- Ckpts are at `gs://training-job-outputs/best_checkpoints/{run_id}/`
  with `run_id` ∈ {`4pf24vo7`, `rravpdb9`, `wc6hvodm`}. All saved
  periodic ckpts present at step{500, 1000, 1500, 2500, 3500, 4500}.
