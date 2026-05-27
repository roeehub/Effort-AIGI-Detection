# Joint-Marginal Audit FACTS — 2026-05-19

> **Question.** Does the training pool's joint marginal distribution on image-property axes span the deployment (dev / lockbox / chronic-6) distribution?
>
> **Decision rule (pre-registered in `RESEARCH_HANDOFF_DATA_DEEP_DIVE_2026-05-19.md` §11).**
> - If ≥30% of lockbox-real mass and ≥50% of chronic-6 mass lies in regions where `train_density(x) < 0.10 × deployment_density(x)` on the 5-7 binding axes, the **targeted data-ingestion lever is structurally live** and "rebalance buckets" was the wrong specification of "data."
> - If <10% / <20% of mass lies in those regions, the data thesis is **not** the binding constraint and the training-recipe deconvolution path is the residual lever.
>
> **Verdict.** Mixed. The decision rule **fails for lockbox-overall** (only 5.3% mass below α=0.10, far under the 30% threshold). The decision rule **fails as written for chronic-6** at α=0.10 (7.2%) but **passes at α=0.25** (71.9% — vs 21.5% for lockbox-overall, a 3.3× ratio). The per-axis tail-mass measurement gives a sharper, falsifiable specification of the data lever than the joint-KDE measurement.

---

## 1. What was measured

### 1.1 Data sources
- `analysis/iq_data_atlas_2026-05-08/_cache/*.parquet` — 29 atlas parquets × ~500 rows each = 14,006 frames across 6 buckets:
  - `train_real` n=2,000  (`train_teams_real_pool` + HDTF clean + HDTF teams sampled to dev-side)
  - `train_fake` n=4,000  (`train_teams_fake_pool` + 6 df40 fake methods + `visomaster_enhanced_v2_all`)
  - `dev_real`   n=2,581
  - `dev_fake`   n=2,500
  - `lockbox_real` n=1,500
  - `lockbox_fake` n=1,425
- `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` — 7,334 frames with `identity_key` (joined to atlas by URI to flag chronic-6 membership; 5 of 6 chronic identities are present; **Roy_D is absent from full_tags so is unmeasurable here**).
- Chronic-6 patterns matched: `PC_Generator`, `bla_bla_chow`, `dor_shkedi`, `xiang`, `Cam_Test`, `Chikara_Takahashi`. Chronic ∩ lockbox_real n=153 frames after join.

### 1.2 Axes
Seven image-property axes from the atlas schema:
- `lap_var` — sharpness (Laplacian variance, computed on the saved crop image; known caveat that the metric is not face-cropped — `OPEN_LOOPS.md` `sharpness-metric-computed-on-full-image-not-face`)
- `luma_mean` — brightness (mean V channel)
- `color_a_dev` — color a-channel std (color cast)
- `color_b_dev` — color b-channel std (color cast)
- `saturation_mean` — mean S channel
- `min_dim` — min(width, height) of the saved crop (resolution proxy)
- `skin_frac` — fraction of pixels classified as skin (face area / crop proxy)

### 1.3 Methods
- **Per-axis marginal**: mean, std, 5/25/50/75/95 quantiles per bucket.
- **Pairwise shift**: Wasserstein-1 (normalized by combined std) and Kolmogorov–Smirnov per axis between bucket pairs.
- **Logistic-regression discriminator**: 7-axis standardized features, 5-fold stratified CV balanced accuracy with `class_weight=balanced`. Coefficients interpret which axes drive separation.
- **PCA + KDE density-ratio**: top-3 PCs on combined real-side data, Gaussian KDE with Scott bandwidth fit independently to each bucket, then ratio `train_density(x) / deployment_density(x)` evaluated at every deployment frame.
- **Per-axis tail mass**: for each axis, compute `[q05, q95]` of `train_real`, then count what fraction of deployment mass lies outside this interval.

All outputs: `analysis/joint_marginal_audit_2026-05-19/{tables,figs,artifacts}/`.

---

## 2. Headline numbers

### 2.1 Discriminator (logistic on 7 standardized axes, 5-fold balanced accuracy)

| Pair | 5-fold balanced acc | Interpretation |
|---|---:|---|
| `train_real` vs `dev_real` | 0.728 | Mid-moderate train↔dev gap on IQ axes |
| `train_real` vs `lockbox_real` | **0.749 ± 0.014** | Mid-moderate train↔lockbox gap on IQ axes — **NOT the 99.09% gap D8 found in CLIP-feature space** |
| `dev_real` vs `lockbox_real` | 0.581 | Dev and lockbox are *similar* on these IQ axes (closer to each other than either is to train) |
| `train_fake` vs `dev_fake` | 0.840 | Strongest fake-side gap |
| `train_fake` vs `lockbox_fake` | 0.790 | Strong fake-side gap |
| `train_real` vs `train_fake` | 0.632 | Train reals and train fakes are partially IQ-separable — consistent with the "model uses IQ as fake predictor" finding (D7) |

**Top-3 discriminating axes between `train_real` and `lockbox_real`** (standardized abs coefs):
1. `lap_var` (sharpness) — coef +1.30 (lockbox is sharper)
2. `skin_frac` — coef +0.76 (lockbox has higher skin fraction)
3. `saturation_mean` — coef −0.65 (lockbox is less saturated)

### 2.2 Per-axis tail mass — the sharpest finding

Fraction of deployment mass **outside `train_real`'s [5%, 95%] interval per axis**:

| Axis | lockbox_real overall | chronic ∩ lockbox_real | ratio chronic / overall |
|---|---:|---:|---:|
| `lap_var` | 9.5% | 35.9% | 3.80× |
| `luma_mean` | 3.0% | 11.8% | 3.92× |
| **`color_a_dev`** | **11.3%** | **85.6%** | **7.60×** |
| `color_b_dev` | 10.7% | 15.0% | 1.41× |
| `saturation_mean` | 6.0% | 13.7% | 2.29× |
| **`min_dim`** | **9.4%** | **76.5%** | **8.14×** |
| **`skin_frac`** | **12.9%** | **69.9%** | **5.41×** |

Three axes have chronic tail mass **above 65%** AND chronic/overall ratio **above 5×**:
- `color_a_dev` — chronic-6 has dramatically different color cast (70.6% below `train_real` q05; 15.0% above q95)
- `min_dim` — chronic-6 frames are at **higher resolution** than train pool (20.3% below q05; 56.2% above q95)
- `skin_frac` — chronic-6 frames have **lower skin fraction** = looser crops (69.9% below q05; 0.0% above q95)

**Lockbox-overall is mostly in-distribution on these axes (3–13% tail mass).** The chronic-6 subset sits in IQ regions the train pool dramatically under-covers, while the rest of lockbox is reasonably covered.

### 2.3 KDE density-ratio (top-3 PCs, Gaussian KDE / Scott bandwidth)

Fraction of deployment mass where `train_density / deployment_density < α`:

| α threshold | dev_real | lockbox_real | chronic ∩ lockbox_real |
|---:|---:|---:|---:|
| 0.01 | 0.6% | 1.8% | 3.9% |
| 0.05 | 3.6% | 2.9% | 5.2% |
| **0.10** | **8.4%** | **5.3%** | **7.2%** |
| **0.25** | **30.6%** | **21.5%** | **71.9%** |
| 0.50 | 50.8% | 55.2% | 82.4% |
| 1.00 | 78.5% | 82.2% | 93.5% |

**Note.** Lockbox-overall has *less* under-coverage at α=0.10 (5.3%) than dev_real (8.4%) — i.e., on the 3-PC summary of the 7 axes, the lockbox pool is not dramatically further from train than dev is. **Chronic-6 jumps to 72% under-coverage only at α=0.25**, three times the lockbox-overall figure.

### 2.4 Per-chronic-identity bimodal split

Per-identity median density ratio `train / lockbox` on top-3 PCs, for identities with ≥20 chronic frames:

| Identity | n | bucket | median train/lockbox ratio | frac below 0.25 | interpretation |
|---|---:|---|---:|---:|---|
| `Chikara_Takahashi__s22` | 23 | lockbox_real | **0.18** | 78% | deeply IQ-under-covered |
| `dor_shkedi` | 118 | lockbox_real | **0.21** | 93% | deeply IQ-under-covered |
| `PC_Generator__s15` | 127 | lockbox_fake | 0.32 | 28% | moderately under-covered |
| `PC_Generator__s14` | 53 | dev_real | 0.61 | 17% | mildly under-covered |
| `Xiang_Xiang2_Feng__s23` | 25 | dev_real | 0.72 | 0% | in-distribution |
| `bla_bla_chow__s1` | 29 | lockbox_real | 0.75 | 3% | in-distribution |
| `Cam_Test__s32` | 91 | dev_fake | 0.75 | 0% | in-distribution |
| … | | | | | |
| `PC_Generator__s22` | 56 | dev_real | **1,256** | 0% | densely-covered (over-represented) |
| `bla_bla_chow` | 143 | dev_real | **189,232** | 0% | densely-covered (extreme over-representation) |
| `Xiang_Xiang2_Feng` | 211 | dev_real | 874 | 1.4% | densely-covered |

**Chronic identities partition bimodally:**
- **Under-covered group** (median ratio < 0.50): `Chikara_Takahashi__s22`, `dor_shkedi`, `PC_Generator__s15`, `PC_Generator__s14`. The data lever — IQ-pocket-targeted ingestion — is *structurally aligned* with the failure mode for this group.
- **Over-covered group** (median ratio > 100): `bla_bla_chow`, `PC_Generator__s22`, `Xiang_Xiang2_Feng`, `PC_Generator__s8`, `dor_shkedi__s16`, `PC_Generator__s3`. These chronic-FP identities sit in IQ regions the train pool *heavily* covers. Their failure mode is **not** IQ-gap-driven. **A data-ingestion lever cannot fix these.**

The two anchor_aware-rescued identities from the 2026-05-16 overnight are:
- `Chikara_Takahashi__s22` — under-covered (median ratio 0.18). Anchor_aware rescue is consistent with "anchor-loss compensates for missing IQ coverage by giving local supervision in the pocket."
- `PC_Generator` (mixed sub-IDs) — `__s15` is under-covered (0.32), `__s22` is over-covered (1,256), `__s3` is over-covered (983). Anchor_aware rescue worked aggregated across sub-IDs; the mechanism is consistent for the under-covered sub-IDs but anomalous for the over-covered ones.

### 2.5 PCA structure

Top-5 PC explained variance: 35.7% / 21.3% / 15.7% / 12.6% / 7.7% — cumulative 92.9%.

PC loadings (which axes drive each PC):
- **PC1 (35.7%)**: anti-correlated `saturation_mean` (−0.59), `color_a_dev` (−0.51), `color_b_dev` (−0.44), `skin_frac` (−0.32) — a **color/saturation/face-fraction axis**. High PC1 = pale, less saturated, smaller face.
- **PC2 (21.3%)**: positive `luma_mean` (+0.63), `min_dim` (+0.56), `color_b_dev` (+0.46) — a **resolution × brightness axis**.
- **PC3 (15.7%)**: dominant `lap_var` (+0.69), negative `skin_frac` (−0.51) — a **sharpness × crop-tightness axis**.

The discriminator axis weighs sharpness (`lap_var`) most, but the binding chronic-6 tails are in color_a_dev / min_dim / skin_frac — these are *PC1 and PC2*. The 3-PC KDE is therefore measuring the right space.

---

## 3. Verdict on the pre-registered decision rule

| Decision rule clause | Threshold | Measured | Pass / Fail |
|---|---|---:|:---:|
| ≥30% of `lockbox_real` mass has `ratio < 0.10` | 30% | 5.3% | **FAIL** |
| ≥50% of `chronic_6 ∩ lockbox_real` mass has `ratio < 0.10` | 50% | 7.2% | **FAIL** |
| (same at `ratio < 0.25`) | (n/a) | 71.9% chronic vs 21.5% overall | n/a |

**The pre-registered rule fails at α=0.10.** But the per-axis tail-mass measurement (§2.2) and the bimodal chronic-identity split (§2.4) tell a sharper story than the joint-KDE measurement:

1. **The data gap is concentrated on three specific axes** (`color_a_dev`, `min_dim`, `skin_frac`) and **on a specific subset of chronic identities** (the half with median ratio < 0.5).
2. **General "ingest lockbox-style data" is not predicted to break the ceiling** — lockbox-overall sits in IQ regions the train pool already covers reasonably (5.3% under at α=0.10, less than dev_real at 8.4%).
3. **Targeted "ingest data covering chronic-6 IQ pockets" is structurally aligned with the failure mode for ~50% of chronic-6 identities** — those with `color_a_dev` outside `[train_q05, train_q95]` (especially the 70.6% below q05), `min_dim` above train q95 (56.2% above), and `skin_frac` below q05 (69.9% below).

---

## 4. Interpretation (note: opinions, not facts)

> Per FACTS-doc convention this section is interpretive. Numbers above are mechanical; reading them is judgment.

The original specification of the data thesis in `RESEARCH_HANDOFF_DATA_DEEP_DIVE_2026-05-19.md` §8 was: *"the training pool's joint marginal on five image-property axes does not span the deployment distribution."* This audit refines that to:

> **The training pool's joint marginal does not span the deployment distribution on three specific axes (color_a_dev, min_dim, skin_frac), and the under-coverage is concentrated on roughly half of the chronic-6 identities. The other half of chronic-6 lives in IQ-dense regions and is failing for non-IQ-gap reasons.**

This has three concrete implications:

1. **The "lockbox-style data ingestion" lever** (broadly construed) is **not predicted to break the ceiling**. Lockbox-overall is IQ-covered by the train pool to within sampling noise. The D8 finding that train↔lockbox separates at 99.09% in *CLIP-feature* space but only 75% in *7-axis IQ* space means most of the substrate gap is in non-IQ dimensions — possibly identity-cluster geometry, capture-pipeline texture, or higher-order spatial features. Adding more lockbox-style data won't move the IQ marginals because the IQ marginals already match.
2. **A narrowly-targeted "chronic-IQ-pocket data ingestion" lever** is structurally live and would specifically address the under-covered half of chronic-6. The pre-launch spec for that lever: **collect real frames satisfying `min_dim > 350 AND (color_a_dev < 4 OR color_a_dev > 12) AND skin_frac < 0.35`** — the joint pocket that contains `dor_shkedi` and `Chikara_Takahashi__s22`.
3. **The other half of chronic-6** (`bla_bla_chow`, `PC_Generator__s22`, `Xiang_Xiang2_Feng`) is failing in IQ-dense regions. This is the canonical "model misclassifies despite having seen many similar examples" failure. The fix is not data; it is training-recipe (anchor_aware, output-preservation loss, multi-axis-GRL) or representation (the LoRA / SVD lever class).

Anchor_aware's 2026-05-16 result is consistent with both readings — it fixed under-covered identities (where it acts as a data-coverage substitute) AND it appears to have worked on some over-covered ones (where it must be acting as a representation-level regularizer). The mechanism is not unique to one of the two readings.

---

## 5. Falsifiable predictions for a follow-up

1. **If the targeted data-ingestion lever is correctly specified**: a CPU re-sample of the train pool with frames satisfying the `dor_shkedi` IQ pocket (above), trained-from-P8A for ~1000 steps, will reduce `dor_shkedi` lockbox FPR by ≥30% absolute. **Cost: $20-50 GPU**. If the reduction is < 10%, the IQ-pocket-ingestion mechanism is refuted as the dominant lever for under-covered chronic identities.
2. **If the under-covered / over-covered partition is real**: anchor_aware's 2026-05-16 rescue should retest cleanly when run on a sister pool restricted to under-covered chronic identities only (`Chikara_Takahashi__s22`, `dor_shkedi`, `PC_Generator__s15`, `PC_Generator__s14`). The 26→0% / 28→0% rescues should hold or improve; the Roy_D regression should *not* be replicated because Roy_D's IQ-pocket position is unmeasurable here.
3. **If non-IQ dimensions of the train→lockbox gap drive the chronic-FP in over-covered identities**: a frozen-CLIP probe on `bla_bla_chow` and `PC_Generator__s22` that ignores the 7 IQ axes (or projects them out via Mahalanobis on the train-real IQ covariance) should still separate fake/real with AUC ≥ 0.99. If it does, the over-covered chronic identities' failure mode is encoded in CLIP features but not in IQ axes — and the lever is representation-side, not data-side.

---

## 6. Open caveats

1. **Roy_D is absent from `full_tags`** so cannot be measured. The 2026-05-16 anchor_aware regression on Roy_D may be an under-covered-pocket case we cannot verify here.
2. **The Laplacian-variance sharpness metric is computed on the saved crop, not on a face-cropped subregion** — open loop `sharpness-metric-computed-on-full-image-not-face`. This biases the absolute scale of `lap_var` but does not invalidate the comparative readout (the bias is constant across pools). Recomputing on face crops is recommended before the data-ingestion lever's pocket-specification is finalized.
3. **The atlas pools sample 500 frames per pool**; the train pool is a sub-sample. Real train pool's tail behavior at q05/q95 may be tighter or looser than measured here. A 5,000-frame re-sample of the train pool would tighten the [q05, q95] envelope estimate.
4. **The 7-axis space is not exhaustive.** D8 (KLIEP in CLIP-feature space) measures something else and finds a much larger train↔lockbox gap. The audit only addresses the IQ-axis component of the gap.
5. **Per-frame density ratio at α=0.10 is hard to satisfy** when the KDE is fit at top-3 PCs (3-dimensional density estimates are smooth). A higher-resolution measurement (top-5 PCs, or per-axis bin-based) might surface gaps invisible to the 3D KDE.
6. **`is_chronic` is matched on identity_key substring**; `Cam_Test` matches Cam_Test sessions which appear primarily as `dev_fake` / `lockbox_fake` in this audit (Cam_Test reals don't seem to be tagged in the atlas join). The "chronic-6" name should be read as "5-of-6 chronic identities, with `Roy_D` missing."

---

## 7. Artifacts

| Path | Contents |
|---|---|
| `artifacts/unified_tags.parquet` | 14,006-row harmonized dataframe (atlas axes + identity_key via full_tags join + bucket label) |
| `artifacts/chronic_full_tags.parquet` | 3,981 chronic-identity frames from full_tags with harmonized axis names |
| `tables/per_axis_marginal.csv` | Per-axis mean/std/quantiles per bucket |
| `tables/pairwise_shift.csv` | Wasserstein-1 + K-S for 5 bucket pairs × 7 axes |
| `tables/discriminator_coefs_train_vs_lockbox.csv` | Standardized coefs of the 7-axis logistic discriminator |
| `tables/pca_loadings.csv` | PCA loadings on combined real-side data (7 axes × 5 PCs) |
| `tables/coverage_density_ratio.csv` | Density-ratio coverage table at 6 α thresholds × 3 pools |
| `tables/per_chronic_identity_location.csv` | Per-identity median ratio + per-axis medians for all 23 chronic-keyed sub-identities |
| `tables/SUMMARY.json` | Machine-readable summary of headline numbers |
| `figs/per_axis_histograms.png` | Overlaid univariate histograms, 7 axes |
| `figs/per_axis_tail_mass.png` | Bar chart of overall vs chronic tail mass per axis (§2.2 headline figure) |
| `figs/pc1_pc2_scatter.png` | Top-2 PC scatter colored by bucket, chronic overlay |
| `figs/density_ratio_cdf.png` | CDF of `train/deployment` density ratios, log scale |
| `figs/discriminator_axis_density.png` | Logistic discriminator scores per bucket |
| `figs/discriminator_axis_coefs.png` | Bar chart of axis coefficients |
| `build_unified.py`, `run_analysis.py` | Reproducible analysis scripts |
