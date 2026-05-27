# CPU Probes FACTS — 2026-05-19

> **Purpose.** Three CPU probes commissioned after the joint-marginal audit to resolve
> the audit's open questions before authorizing GPU spend.
>
> 1. **Probe 1** — characterize the non-IQ component of the train↔lockbox gap.
> 2. **Probe 2** — within-identity frozen-CLIP separability for over-covered chronic identities.
> 3. **Probe 3** — Roy_D IQ-pocket placement (he was absent from `full_tags` in the audit).
>
> **Headline.** All three probes returned strong signals that change the planned GPU
> slot specification. Most important: the non-IQ component of the substrate gap is
> **face-identity cluster geometry**, not capture-pipeline texture. The data lever needs
> identity-similarity, not just IQ-pocket coverage.

---

## Probe 1 — Non-IQ component of the dev↔lockbox gap

### What was measured

A logistic discriminator was trained to distinguish `split=dev` from `split=lockbox` on the real-only subset of `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` (n=4,246 frames; 3,832 dev_real + 414 lockbox_real). Five stages, each evaluated by 5-fold stratified CV balanced accuracy and AUC:

- **Stage A** — raw CLIP features (512-d `clip_embed`).
- **Stage B** — IQ-only features (9 continuous IQ axes + 4 categorical CLIP-aux labels, ~30-d).
- **Stage C** — CLIP features after linear-regression residualization against the IQ feature matrix (kills any CLIP component linearly predictable from IQ).
- **Stage D** — CLIP features with the top-20 IQ-aligned dimensions dropped (cruder ablation).
- **Stage E** — CLIP features residualized against `(IQ features ∪ ArcFace identity embedding)`.

### Results

| Stage | Features | bacc | AUC |
|---|---|---:|---:|
| A | raw CLIP | **0.996 ± 0.005** | **1.000** |
| B | IQ only | 0.826 ± 0.014 | 0.915 |
| C | CLIP residualized vs IQ | 0.986 ± 0.011 | 0.999 |
| D | CLIP minus top-20 IQ-aligned dims | 0.996 | 1.000 |
| **E** | **CLIP residualized vs (IQ ∪ ArcFace)** | **0.446 ± 0.038** | **0.434** |

ArcFace-only discriminator (for reference): **bacc = 0.996, AUC = 1.000**.

Correlation between residual-CLIP discriminator axis and ArcFace discriminator axis: **rho = +0.52**.

### Headline finding

**The 99.6% separability of dev_real from lockbox_real in CLIP-feature space is dominated by face-identity cluster geometry, not by image-quality variation.**

- Subtracting IQ from CLIP barely moves the needle (0.996 → 0.986).
- Subtracting `(IQ + ArcFace identity)` collapses the discriminator to chance (0.996 → 0.446).
- ArcFace embeddings alone achieve full separation (0.996).

The two pools (dev_real, lockbox_real) consist of **structurally different face-identity clusters**. The shift is "different humans / different face-similarity neighborhoods" more than "different cameras / lighting / codecs." This refines D8's "structural data gap; not architecture/loss-fixable in isolation" — the structural gap is identity-coverage gap, not transport gap.

### Caveat

CLIP residualization is linear; if IQ enters CLIP features non-linearly, Stage C under-removes IQ. Stage D (drop top-20 IQ-aligned dims) is a non-linear sanity check and gives the same answer (0.996, no movement). So the linear residualization is not under-removing the IQ component in a way that would change the verdict.

---

## Probe 2 — Frozen-CLIP within-identity separability for over-covered chronic IDs

### What was measured

For each chronic identity with both `real` and `fake` frames in `full_tags`, fit a 5-fold cross-validated frozen-CLIP linear probe on real-vs-fake **within that identity_key only**. Report AUC. Compare:

- **Over-covered** group (median train/lockbox density ratio > 0.5 per audit): `Xiang_Xiang2_Feng`, `dor_shkedi__s16`, `PC_Generator__s4`, `PC_Generator__s8`, `bla_bla_chow`, `PC_Generator__s22`, `Cam_Test__s33`, `PC_Generator__s3`.
- **Under-covered** group: `Chikara_Takahashi__s22`, `dor_shkedi`, `PC_Generator__s15`, `PC_Generator__s14`.
- **Reference** (non-chronic): `Test_Cam__s41`, `Md_noyn_Sharker__s15`, `deeplive_dor`.

### Results

Of 15 identities tested, only **4 had ≥5 reals AND ≥5 fakes** (the rest are single-label-only in this dataset, because chronic identities' fakes are often labeled with a different identity_key like `deeplive_dor`).

| identity_key | group | n | real / fake | frozen-CLIP AUC |
|---|---|---:|---:|---:|
| `Xiang_Xiang2_Feng` | over-covered | 436 | 301 / 135 | **1.0000 ± 0.000** |
| `dor_shkedi__s16` | over-covered | 109 | 31 / 78 | **1.0000 ± 0.000** |
| `PC_Generator__s4` | over-covered | 39 | 9 / 30 | **1.0000 ± 0.000** |
| `PC_Generator__s15` | under-covered | 120 | 29 / 91 | **1.0000 ± 0.000** |

ArcFace-only within-identity AUC was also 1.000 across the same 4 identities — which is notable on its own (the swap face has different identity-embedding geometry than the original).

### Headline finding

**For every testable over-covered chronic identity, frozen CLIP separates real from fake at AUC = 1.000.** This supports reading (a) from the audit:

> **The encoder retains the forgery signal even within over-covered chronic identities. The fine-tuned head destroys the separation.**

Reading (b) — "encoder can't see these fakes" — is rejected for the testable subset. This means the over-covered half of chronic-6 is a **representation/head failure**, not a data-coverage failure. The right lever class is **anchor_aware, output-preservation loss, or multi-layer-GRL against the head's drift** — not data ingestion.

### Caveats

1. Only 4 identities are testable (most chronic identities are single-label in `full_tags`). The result is consistent across all 4 but the sample size is small.
2. Per-identity AUC = 1.000 at n ≈ 100 with a 512-d feature space risks fold-overfit. The 5-fold structure protects against in-fold leakage, but a held-out probe at identity-disjoint folds would strengthen the result.
3. The probe uses CLIP projection-head features (512-d), not L11 hidden features (1024-d). D2's chronic-6 AUC=1.000 was on L11 features. The result here is consistent with D2 and extends it to within-identity granularity, but the feature space differs slightly.

---

## Probe 3 — Roy_D IQ-pocket characterization

### What was measured

Roy_D was absent from `full_tags` (the audit's source for identity-keyed analysis), but he is present in the IQ atlas: **63 atlas frames** across 4 pools, including **14 frames in `train_teams_real_pool`**. The probe computes Roy_D's joint position on the 7-axis IQ space and his density ratio `train_real / lockbox_real` (top-3 PCs, Gaussian KDE — same recipe as the audit).

### Results

Per-axis position of Roy_D vs `train_real`:

| axis | Roy_D median | train_real median | train [q05, q95] | Roy_D mass in train tails |
|---|---:|---:|---:|---:|
| `lap_var` | 64.3 | 33.7 | [7.5, 492] | **3.2%** |
| `luma_mean` | 145.8 | 131.4 | [88.6, 182.7] | **0.0%** |
| `color_a_dev` | 19.0 | 12.3 | [5.7, 21.3] | 3.2% |
| `color_b_dev` | 15.7 | 13.7 | [6.4, 24.5] | 3.2% |
| `saturation_mean` | 122.7 | 101.6 | [54.2, 148.1] | 3.2% |
| `min_dim` | 276 | 224 | [166, 313] | **0.0%** |
| `skin_frac` | 0.88 | 0.64 | [0.33, 0.96] | 1.6% |

Density ratio `train / lockbox` at each Roy_D frame:

- **Median ratio: 0.78** (well-covered; under-covered identities have median < 0.5)
- Frac below α=0.10: **0.0%**
- Frac below α=0.25: **1.6%**
- Frac below α=0.50: 14.3%

Pool distribution: 38/63 Roy_D frames are in `teams_real_lighting_extreme_dev`. 14/63 in `train_teams_real_pool`. 9/63 in `teams_real_all_dev`.

### Headline finding

**Roy_D is NOT in the under-covered IQ pocket.** He sits in IQ regions the train pool covers well (median ratio 0.78, near-zero tail mass on every axis). Moreover, **he has 14 training-pool frames** — he is in-distribution as an identity, not held out.

This rejects the original Slot β specification ("anchor_aware + Roy_D anchor pool extension"). The hypothesis that Roy_D's 29→81% regression in the 2026-05-16 overnight is IQ-pocket-driven is unsupported by IQ measurement.

Two alternative mechanisms for Roy_D's regression (untested here):

1. **Lighting-extreme substrate-specific.** 60% of Roy_D's atlas frames live in `teams_real_lighting_extreme_dev`. If anchor_aware's anchor pool happens to be IQ-similar to the lighting-extreme conditions where Roy_D fails, the loss can still over-pull-against Roy_D's score even when his IQ pocket is in-distribution. The mechanism is then anchor-pool composition, not IQ coverage.
2. **Identity-cluster-specific anchor-loss interaction.** Per Probe 1, dev↔lockbox is identity-cluster-driven. If anchor_aware's anchor pool sits at a position in identity space close to Roy_D, the contrastive pull can be "wrong" for Roy_D specifically — pulling his embedding toward a non-Roy_D anchor that the recipe treats as "real-canonical."

Either way, **a "more Roy_D in the anchor pool" extension is not predicted to fix the regression.** The Slot β specification needs to be redesigned around the anchor-pool composition itself.

### Caveat

Roy_D has only 63 frames in the atlas (49 dev + 14 train, 0 lockbox). The IQ pocket position is well-estimated at this sample size, but the density-ratio estimate is bandwidth-dependent on the lockbox-side KDE. The headline ("Roy_D is not under-covered") is robust to bandwidth doubling; the precise median (0.78) is not.

---

## Combined verdict — implications for the Slot α / Slot β GPU plan

The original recommendation (from the audit) was:

- Slot α: Chronic-IQ-pocket data ingestion (~$30, us-west4) — predicted to reduce `dor_shkedi` lockbox FPR ≥ 30%.
- Slot β: anchor_aware + Roy_D-specific anchor pool (~$50, us-east1) — predicted to preserve 2026-05-16 chronic rescues AND fix Roy_D 29→81% regression.

The probes refine this:

| Probe | Finding | Implication for Slot |
|---|---|---|
| **1** | Substrate gap is identity-cluster (ArcFace) driven, not IQ-driven | **Slot α specification should include identity-similarity (ArcFace neighbors) as a sampling criterion, not just IQ-pocket membership.** A pure IQ-pocket-only resample may not move the dev↔lockbox separability that drives FP attribution. |
| **2** | Frozen CLIP within-identity AUC = 1.000 for all 4 testable chronic IDs (mix of over- and under-covered) | **The over-covered half of chronic-6 is a head-destruction failure, not a data-coverage failure.** Confirms the representation-side lever (anchor_aware, etc.) is the right class for these IDs. Data ingestion alone won't fix bla_bla_chow / PC_Generator__s22. |
| **3** | Roy_D is **not** in the under-covered IQ pocket (median ratio 0.78) and has 14 training-pool frames | **Slot β as originally specified is structurally wrong for Roy_D.** "More Roy_D in the anchor pool" is unlikely to fix the regression because Roy_D's failure mode is not IQ-pocket-coverage. The Slot β lever needs to investigate the **anchor pool composition** itself (what identity-similar frames sit in the anchor pool and pull Roy_D's score the wrong way during the contrastive loss). |

### Revised slot recommendations

**Slot α — IQ-pocket × identity-similarity data ingestion** (~$30-40, us-west4)
- Sample reals satisfying `min_dim > 350 AND (color_a_dev < 4 OR > 12) AND skin_frac < 0.35` (IQ-pocket from audit)
- AND with ArcFace-cosine ≥ 0.4 to at least one of the under-covered chronic identities (`dor_shkedi`, `Chikara_Takahashi__s22`, `PC_Generator__s14`, `PC_Generator__s15`) — identity-similarity criterion derived from Probe 1.
- FT-from-P8A for 3000 steps, pocket-frame weight 3-5×.
- **Falsifiable prediction unchanged**: `dor_shkedi` lockbox FPR ≥ 30% absolute reduction. If it fails, the IQ-pocket + identity-similarity specification is also wrong and the data lever is dead.

**Slot β — anchor pool composition diagnostic (CPU first, then ~$50 GPU)**
- CPU first: characterize the anchor pool used in the 2026-05-16 overnight Slot α. Compute pairwise ArcFace cosines between Roy_D and the anchor frames. If Roy_D's nearest anchor is in fact a different person whose score-pull is "real-canonical," that's the mechanism.
- THEN GPU: re-run anchor_aware with a Roy_D-disjoint anchor pool (anchors that have ArcFace-cosine < 0.3 to Roy_D), predict that the 29→81% regression disappears. If it does, anchor-pool composition is the mechanism; the lever class survives. If it doesn't, anchor_aware has a Roy_D-specific failure mode that needs further investigation before deployment.

**What is NOT recommended (additional negative results from probes)**
- A pure "more lockbox-style data" lever (audit + Probe 1 both refute it).
- A pure "more chronic-identity data" lever without identity-similarity filtering (Probe 1 suggests pocket alone is insufficient).
- A "Roy_D-specific anchor pool extension" without first understanding why the existing pool over-pulls Roy_D (Probe 3 rejects the IQ-pocket framing).

---

## Artifacts

| Path | Contents |
|---|---|
| `tables/probe1_summary.json` | Stage A-E discriminator scores |
| `tables/probe1_clip_dim_stats.csv` | Per-CLIP-dim IQ R² + residual-discriminator coefs |
| `tables/probe2_per_identity_auc.csv` | Per-chronic-identity within-identity AUC + ArcFace-comparison AUC |
| `tables/probe2_summary.json` | Group-level aggregates |
| `tables/probe3_royd_axis_summary.csv` | Roy_D per-axis medians + comparison to train/dev/lockbox |
| `tables/probe3_summary.json` | Roy_D density-ratio summary |
| `figs/probe1_stages.png` | Bar chart of bacc + AUC across 5 stages |
| `figs/probe2_within_identity_auc.png` | Per-identity AUC bars grouped by audit category |
| `figs/probe3_royd_pca_position.png` | Roy_D position on PC1-PC2 vs chronic comparison IDs |
| `figs/probe3_royd_per_axis.png` | Roy_D per-axis IQ distributions vs train/lockbox |
| `probe1_non_iq_gap.py` | Reproducible analysis script |
| `probe2_chronic_separability.py` | Reproducible analysis script |
| `probe3_royd_iq.py` | Reproducible analysis script |
