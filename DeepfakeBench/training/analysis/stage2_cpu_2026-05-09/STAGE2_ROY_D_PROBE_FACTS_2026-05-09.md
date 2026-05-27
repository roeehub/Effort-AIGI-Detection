# Stage 2 Roy_D probe FACTS (2026-05-09)

> **Status: factual-only.** No interpretation. Forbidden words: succeeds,
> fails, wins, promotes, deployment-grade.
>
> Source data: `outputs/roy_d_stage2_per_frame.csv`,
> `outputs/roy_d_stage2_ckpt_stats.csv` produced by `run_roy_d_probe.py`
> on 2026-05-09 03:37–03:38 local (MPS device, 130-frame Roy_D set
> downloaded fresh from GCS).
>
> Companion docs: cohort-level FACTS for the Dor cohort at
> `STAGE2_SCORE_PROBE_FACTS_2026-05-09.md`. Original Roy_D regression
> precedent at `analysis/p1_pe_eval_2026-05-07/roy_d_regression/`.

## 1. Question

P1 BUNDLE_step500 reference: `std=0.0005` on 130-frame Roy_D real set —
800× variance compression vs P8A's 0.405; Pearson r=-0.16 with P8A
(anti-correlated). All 130 frames classified as fake at τ=0.5.

Do any of the Stage 2 step{500, 2500, 4500} ckpts reproduce this
collapse pattern, or its weaker dose-response variants?

## 2. Method

Reused `run_score_probe.py` infrastructure but with the 130-frame Roy_D
set from `analysis/p1_pe_eval_2026-05-07/roy_d_regression/
roy_d_per_frame_scores.csv` (P8A, BUNDLE-step500/3750/4000, PAIRRANK-
step500/6000/6750, E2B reference scores baked in).

Per ckpt: 130 frames preprocessed (224 CLIP), CPU/MPS forward, fake-class
softmax probability extracted. Stage 2 ckpts loaded from local cache
populated by parallel gsutil downloads (~2-3min for all 9 ckpts).

## 3. Per-ckpt stats — full table

(All 130 frames are REAL Roy_D; ideal: low fake-prob, low frac_FP.)

| ckpt | mean | p50 | std | range | r vs P8A | frac_FP @ τ=0.5 |
|---|---:|---:|---:|---:|---:|---:|
| **P8A_REFERENCE_STEP5000** | 0.477 | 0.438 | **0.405** | 0.988 | 1.000 | 0.454 |
| E2B_TOP_N_STEP3200 | 0.353 | 0.286 | 0.305 | 0.956 | 0.507 | 0.331 |
| **P1_BUNDLE_PERIODIC_STEP500** | 0.992 | 0.993 | **0.0005** | 0.004 | **-0.159** | **1.000** |
| P1_BUNDLE_TOP_N_STEP3750 | 0.998 | 1.000 | 0.012 | 0.108 | 0.202 | 1.000 |
| P1_BUNDLE_TOP_N_STEP4000 | 0.994 | 1.000 | 0.031 | 0.250 | 0.226 | 1.000 |
| P1_PAIRRANK_PERIODIC_STEP500 | 0.860 | 0.982 | 0.215 | 0.918 | 0.657 | 0.908 |
| P1_PAIRRANK_TOP_N_STEP6000 | 0.964 | 1.000 | 0.108 | 0.544 | 0.384 | 0.977 |
| P1_PAIRRANK_TOP_N_STEP6750 | 0.962 | 1.000 | 0.111 | 0.611 | 0.395 | 0.977 |
| S1_REAL_AUG_OFF step500 | 0.914 | 0.982 | 0.142 | 0.642 | 0.575 | 0.969 |
| S1_REAL_AUG_OFF step2500 | 0.778 | 0.931 | 0.272 | 0.947 | 0.731 | 0.815 |
| S1_REAL_AUG_OFF step4500 | 0.966 | 0.995 | 0.084 | 0.510 | 0.422 | **0.992** |
| S2_LOW_LR_FT step500 | 0.740 | 0.862 | 0.214 | 0.769 | 0.772 | 0.846 |
| S2_LOW_LR_FT step2500 | 0.922 | 0.974 | 0.127 | 0.632 | 0.488 | 0.962 |
| **S2_LOW_LR_FT step4500** | 0.932 | 0.978 | 0.119 | 0.627 | 0.459 | **0.969** |
| S3_WEAK_PAIRRANK step500 | 0.978 | 0.982 | **0.022** | 0.205 | 0.240 | **1.000** |
| S3_WEAK_PAIRRANK step2500 | 0.944 | 0.990 | 0.113 | 0.547 | 0.485 | 0.977 |
| S3_WEAK_PAIRRANK step4500 | 0.984 | 0.997 | 0.043 | 0.243 | 0.357 | **1.000** |

## 4. Headline metrics — Stage 2 step4500 vs P8A

(All 130 frames are real Roy_D — false-positives only.)

| metric | P8A | S1 step4500 | S2 step4500 | S3 step4500 |
|---|---:|---:|---:|---:|
| frac FP at τ=0.5 | **0.454** | 0.992 | 0.969 | 1.000 |
| score median | 0.438 | 0.995 | 0.978 | 0.997 |
| score std | 0.405 | 0.084 | 0.119 | 0.043 |
| Pearson r vs P8A | 1.000 | 0.422 | 0.459 | 0.357 |

S1 step4500 score-std is **4.8× compressed** vs P8A.
S2 step4500 score-std is **3.4× compressed** vs P8A.
S3 step4500 score-std is **9.4× compressed** vs P8A.

(Reference: BUNDLE_step500 was 800× compressed.)

## 5. Dose-response signal — pair_rank λ axis

Comparing pair_rank step500 ckpts at three λ values + the no-pair_rank
controls:

| ckpt | pair_rank λ | std on Roy_D | std vs P8A |
|---|---:|---:|---:|
| P8A_REFERENCE | 0 | 0.405 | 1.0× |
| S2_LOW_LR_FT step500 | 0 | 0.214 | 1.9× compressed |
| S1_REAL_AUG_OFF step500 | 0 | 0.142 | 2.9× compressed |
| **S3_WEAK_PAIRRANK step500** | **0.05** | **0.022** | **19× compressed** |
| **P1_PAIRRANK_PERIODIC_STEP500** | 0.20 | 0.215 | 1.9× compressed |
| **P1_BUNDLE_PERIODIC_STEP500** | 0.20 + GroupDRO | **0.0005** | **800× compressed** |

S3 (pair_rank λ=0.05) step500 std=0.022 is between the no-pair_rank
controls (0.14-0.21) and the BUNDLE bundle (0.0005). Dose-response is
present in this λ range but non-linear: PAIRRANK-only at λ=0.20 has
std=0.215 (similar to no-pair_rank), but BUNDLE (λ=0.20 + GroupDRO) is
800× compressed.

Reading: pair_rank ALONE at λ=0.20 doesn't compress (PAIRRANK_step500
std=0.215). BUNDLE's compression is GROUPDRO + pair_rank interaction.
S3's compression at λ=0.05 with no GroupDRO suggests pair_rank can
solo-compress at sufficiently low λ, OR that S3's training-step had a
different dynamic (random seed effects).

## 6. Trajectory — score-std on Roy_D over training steps

| slot | step500 | step2500 | step4500 |
|---|---:|---:|---:|
| S1 | 0.142 | 0.272 | 0.084 |
| S2 | 0.214 | 0.127 | 0.119 |
| S3 | 0.022 | 0.113 | 0.043 |

S3 trajectory is non-monotone (recovers between step500 and step2500,
then re-compresses). S1 and S2 trajectories show plateau-like behavior.

## 7. Caveats

- 130-frame single-identity (Roy_D) substrate. The Pillar-2 regression
  measured here is identity-conditional and may not generalize to the
  full lockbox real population.
- All 130 frames are REAL — no fake-recall measurement here. Probe is
  Pillar-2-specific.
- τ=0.5 is the diagnostic threshold; deployment τ for P8A is 0.916 per
  the contract scorecard. At deployment τ=0.916, P8A frac_FP on this
  Roy_D set drops below 5% (per `roy_d_per_frame_scores.csv`'s
  `is_above_tau` column with `tau_selected=0.916`); Stage 2 ckpts have
  not had per-frame contract-τ analysis run yet.
- Reference BUNDLE/PAIRRANK scores in this table are from the existing
  `roy_d_per_frame_scores.csv` (different inference pipeline; expected
  numerical agreement at ~1e-4 magnitude).
- Roy_D was P8A-trained-on; not all chronic-FP identities exhibit the
  same regression magnitude (per `project_chronic_offenders_partition_per_ckpt_2026-05-04.md`).
