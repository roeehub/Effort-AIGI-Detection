# Best-candidate search 2026-05-13 — intermediate findings

## Stage 1 — F4 substrate-cleaning across 38 candidates

Frozen F4 contract from analysis/substrate_cleaning_eval_2026-05-05/:
F4 = drop chronic-6 identities + drop min(W,H)<200 + drop is_no_face.
Calibration: tau picked to give FPR=10% (or 5%) on the F4-cleaned real pool.

Real pool: teams_real_all_dev. F0 n=4564, F4 n=2091 (45.8% kept).

Headline (top 10 by macro_F4_FPR10):

| Rank | ckpt | macro F4@FPR10 | macro F4@FPR5 | viso F4@FPR10 | dl F4@FPR10 | teams_fake F4@FPR10 |
|---|---|---|---|---|---|---|
| 1  | T5C_step3500              | 95.02 | 90.56 | 87.64 | 100.00 | 97.43 |
| 2  | P2D_fourier_step3000      | 94.08 | 89.16 | 85.64 |  99.63 | 96.97 |
| 3  | P1_pairrank_step6750      | 93.44 | 88.30 | 83.45 | 100.00 | 96.87 |
| 4  | P1_pairrank_step6000      | 92.15 | 87.93 | 80.18 | 100.00 | 96.28 |
| 5  | P2D_fourier_step8000      | 92.02 | 82.94 | 80.00 | 100.00 | 96.05 |
| 6  | T3_SLOT1_step2500         | 91.78 | 84.27 | 79.27 | 100.00 | 96.08 |
| 7  | E3_step6600               | 91.18 | 73.53 | 77.64 | 100.00 | 95.89 |
| 8  | P1_bundle_step4000        | 90.63 | 85.33 | 76.55 | 100.00 | 95.33 |
| 9  | P1_bundle_step3750        | 89.96 | 84.26 | 74.91 | 100.00 | 94.97 |
| 10 | T3_SLOT1_step1500         | 89.47 | 85.06 | 73.27 | 100.00 | 95.13 |

For comparison:
- **P8A_step5000** (classic invariance gold standard): rank 25, 83.93 / 72.46
- **E2B_step3200** (currently deployed): rank 29, 72.68 / 64.06
- **SLOT1_LORA_*** (today's batch): rank 33-36, 61-63%

## Stage 2 — Per-chronic-identity extremeness

Chronic-6 = 993 frames = **21.8% of dev real pool**.

Per-identity FPR contribution across 7 reference ckpts (FPR % of identity's own frames passed at deployment-FPR=10% tau on full F0):

| Identity | n | %pool | min_wh | lt200% | P8A | E2B | E3 | T3_S1 | T5C | PA | SLOT1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pc_generator__s22 | 227 | 5.0% |  88 | 100% | **90.7** |  11.5 |   5.7 | **76.7** | 45.8 | 23.3 | **68.3** |
| q__s6             |  54 | 1.2% |  94 | 100% | **94.4** |  59.3 |  16.7 |  44.4 | 33.3 | 16.7 | **90.7** |
| pc_generator__s45 |  91 | 2.0% |  90 | 100% |  57.1 |  49.5 |  17.6 |  29.7 | 49.5 | 39.6 |  59.3 |
| bla_bla_chow      | 311 | 6.8% | 399 |   2% |   7.4 |  49.2 |  53.1 |  13.5 | 21.5 | 51.4 |   0.0 |
| bla_bla_chow__s2  | 180 | 3.9% | 146 |  96% |  17.8 |  37.2 |  75.0 |  28.3 | 32.2 | 36.1 |   1.1 |
| roy_d             | 130 | 2.9% |  NA |   0% |  38.5 |  32.3 |  80.0 | **93.1** | **96.9** | 79.2 |  13.8 |

Mean FPR across all 7 ckpts (extremeness proxy):
1. pc_generator__s22: ~46% — TINY 88px face, universally bad (median min_wh 88, 100% lt200)
2. q__s6:             ~51% — TINY 94px face (median min_wh 94, 100% lt200)
3. pc_generator__s45: ~43% — TINY 90px face (median min_wh 90, 100% lt200)
4. roy_d:             ~62% — no parquet coverage; recipe-specific (T3/T5C-FT vs P8A)
5. bla_bla_chow:      ~28% — WIDE 399px face (other-end-of-distribution OOD)
6. bla_bla_chow__s2:  ~38% — small face (146) + LOW sharpness 55 Laplacian

**The 3 most universally extreme are pc_generator__s22, q__s6, pc_generator__s45** — all
TINY faces (88-94 px), all 100% below the 200-px min_dim threshold the production
pipeline uses. They are not just "hard," they are structurally OOD on face size.

**roy_d** is a different story — it is a *recipe-specific* failure: P8A/SLOT1 handle
it (38%/14% FPR) but every model that includes the T3 SLOT1 data lever
(drop-top-25%-high-IQ-teams-reals) regresses to 80-97% FPR. So roy_d is not
fundamentally extreme; it is what the T3 lever happens to make the encoder
collapse.

## What "extreme" means in production terms

The chronic-3 tiny-face cohort (pc_gen_s22 + q__s6 + pc_gen_s45) = 372 frames at
median min_dim 88-94 px. Production Teams shows faces from real webcams, where
the face crop is essentially always ≥150 px tall and usually ≥250 px. A
sub-100-pixel face crop in a Teams call is **structurally rare** — it means
either an extremely zoomed-out camera or a misdetection at the upstream face
detector. The eval-substrate cropping bug (`project_eval_production_crop_tightness_gap`)
is the leading explanation for why the eval-set has them at all.

So dropping these 3 tiny-face identities from FPR scoring is not "hiding the
bad cases" — it is correcting an eval-substrate artifact. The cost is 8.2% of
eval coverage (372/4564 frames).

Roy_D requires a different argument — it is recipe-specific, not extreme.

## Open: Stage 3

Need to compute progressive-drop scoreboard:
- drop-0 = F0 (no drops); drop-3-tinyface = drop pc_gen_s22 + q__s6 + pc_gen_s45;
  drop-3+roy_d = +roy_d; drop-all-chronic-6 = F4 chronic part.
- For each top-15 candidate, fake_recall@FPR=5% under each drop policy.
- Plus shortcut indicator (IQ Pearson r), latent separability (probe AUC).
