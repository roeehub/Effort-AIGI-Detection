# RESULTS_FACTS_v5 — encoder embedding probe + aug refutation

> **FACTS only.** Final round of CPU diagnostics. Tests whether the proposed
> compositional augmentation actually moves frames toward Roy_D in encoder
> embedding space (the necessary causal claim for the v4 intervention).

## §1. Provenance

- Model: OpenCLIP ViT-B-16 (`datacomp_xl_s13b_b90k`) — the BASE pre-trained
  backbone that T5C and Slot β were fine-tuned from. SVD reparameterization
  + fine-tuned weights were NOT loaded (technical complexity from SVD module
  setup). Vanilla base used as a proxy for the encoder representation space.
- 194 dev PNG frames embedded (130 Roy_D + 29 ilan + 35 orel).
- 100 augmented samples (training reals + compositional aug applied) embedded.
- 100 original training real frames embedded for baseline reference.
- All embeddings 512-dim CLS features.

## §2. Test 1 — encoder DOES separate Roy_D from clean controls

Cosine similarity within and across cohorts (vanilla openclip ViT-B-16):

| pair | cosine sim |
|---|---:|
| within Roy_D (n=130) | **0.9148** (very tight cluster) |
| within ilan+orel (n=64) | 0.7881 |
| across Roy_D ↔ ilan+orel | **0.6537** (far apart) |

Roy_D embeddings form a tight, distinct cluster. ilan+orel are looser but distinct from Roy_D.

## §3. Test 2 — linear probe AUC

Logistic regression on 512-dim embeddings, 5-fold CV:

| target | AUC |
|---|---:|
| Roy_D vs ilan+orel | **1.0000** |
| Slot β over-fire (any identity) | **0.8836** |

The encoder's embedding space alone (no head, no fine-tuning) almost perfectly distinguishes Roy_D from ilan+orel, and predicts Slot β over-fire at AUC 0.88.

## §4. Test 3 — PCA 2D projection

PCA fit on the 194 dev PNG embeddings:
- PC1 explains 42.5% of variance.
- PC2 explains 11.4% of variance.

Per-cohort PC1/PC2 centroids:

| cohort | n | PC1 | PC2 |
|---|---:|---:|---:|
| Roy_D | 130 | **−3.535** ± 0.60 | −0.07 ± 0.42 |
| ilan | 29 | **+7.075** ± 0.44 | −2.47 ± 0.20 |
| orel | 35 | **+7.269** ± 0.77 | +2.29 ± 5.32 |
| training_reals (original) | 100 | +3.513 ± 1.42 | +0.55 ± 0.88 |
| augmented samples | 100 | +3.133 ± 1.46 | +0.67 ± 0.81 |

PC1 separates Roy_D (-3.5) from ilan/orel (+7.1, +7.3) by ~10 units. Training reals sit at +3.5 — between but closer to ilan/orel.

## §5. CRITICAL — augmented samples do NOT move toward Roy_D in encoder space

The augmented samples sit at PC1=+3.1, only 0.4 units left of the original training reals at +3.5. Roy_D is at -3.5. The aug moves frames ~5% of the way toward Roy_D in PC1.

Per-sample cosine similarity to cohort centroids:

| cohort | cos_sim to Roy_D centroid | cos_sim to clean centroid | % closer to Roy_D |
|---|---:|---:|---:|
| Original training reals | 0.6413 | 0.6839 | **24%** |
| **After compositional aug** | 0.6649 | 0.6944 | **30%** |

The augmentation moves 6% MORE of the samples closer to Roy_D than to clean (from 24% → 30%). Median cosine similarity to Roy_D centroid increases from 0.6389 → 0.6643 (+0.025).

## §6. What this means

The property bands identified in v3 (decision tree AUC 0.89 on hand-crafted features) and the encoder's embedding-space discrimination of Roy_D (AUC 1.00 in 512-dim space, PC1=-3.5 vs +7.1) are CORRELATED but not the same axis. Augmenting training reals to inhabit the band region (sharpness <142, lab_a_dev >16, etc.) does NOT move them appreciably in the encoder's principal discrimination axis.

The encoder's PC1 axis is keying on something else — likely face geometry, pose, identity-specific shape features, or higher-order textures that are not capturable by Laplacian variance + LAB statistics + skin-fraction + crop dimensions.

## §7. The intervention proposed in AGENT_PROPOSAL_v4 §4 is mechanistically refuted

A 6% improvement in "% of samples closer to Roy_D" is not enough anchoring to fix a region the model fires 79% on. The proposed compositional augmentation cannot produce training reals that the encoder would recognize as occupying the same feature region as Roy_D.

## §8. Output files

- `outputs/encoder_embeddings_dev_png.csv` — 194 dev PNG frames + 512-dim embeddings
- `outputs/pca_dev_png_2d.csv` — 2D PCA projections for dev only
- `outputs/pca_with_aug_and_training.csv` — extended with 100 aug + 100 training samples
- `outputs/aug_test_properties.csv` — properties of 100 orig + 100 augmented frames

## §9. Limitation

This probe used **vanilla openclip ViT-B-16**, not the SVD-modified fine-tuned
T5C / Slot β encoders. The trained encoders MAY have changed their primary
axis through FT. To definitively confirm, the SVD-aware loader needs to be
built (~2-4h additional engineering). However: PC1 explains 42% of variance
on the dev frames in base CLIP space; substantial PC1 axis shift via FT would
require an unusual amount of representation rewriting, which is not typical
of standard fine-tuning. The base-CLIP result is suggestive but not
definitive evidence about the trained ckpt's encoder.
