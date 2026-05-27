# CLIP-B16 frozen vs P8A frozen — viso linear probe

**Date**: 2026-05-03
**Hypothesis tested**: Did the 5-stage FT chain (CLIP → R12g → RLP6_04 → RLP7_02 → P8A)
collapse the discriminative direction that raw CLIP-B16 had on viso fakes?
**Setup**: Both models forward-passed on the same 1,100 frames (550 viso fake +
550 teams_real_dev sampled, seed=737). 512-d post-projection features.
5-fold StratifiedKFold linear probe, sklearn LogisticRegression, n_jobs=1, MPS for forward.

## Verdict: **PASSED** — raw CLIP-B16 outperforms P8A under classifier-capacity stress.

At default hyperparameters, both models saturate at AUC=1.0 on this probe (the
viso-vs-teams-real bucket gap is trivially separable in any reasonable feature space).
**The signal lives in the stress tests** — what happens when classifier capacity
or training data is constrained.

### C-sweep (regularization stress; smaller C = more regularization)

| C | CLIP-B16 raw AUC | P8A AUC | Δ (CLIP − P8A) |
|---|---:|---:|---:|
| 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| 0.1000 | 1.0000 | 0.9999 | +0.0001 |
| 0.0100 | 1.0000 | 0.9937 | +0.0063 |
| 0.0010 | 1.0000 | 0.9261 | **+0.0739** |
| 0.0001 | 0.9988 | **0.8259** | **+0.1729** |

### N-sweep (training-data starvation; smaller n = harder probe)

| n_train per fold | CLIP-B16 raw AUC | P8A AUC | Δ (CLIP − P8A) |
|---:|---:|---:|---:|
| 256 | 1.0000 | 0.9999 | +0.0001 |
| 128 | 0.9999 | 0.9968 | +0.0031 |
| 64 | 0.9999 | 0.9880 | +0.0118 |
| 32 | 0.9995 | 0.9801 | +0.0194 |
| 16 | 0.9976 | **0.9117** | **+0.0859** |

## Interpretation

Both probes show the same shape: **P8A has the viso signal, but spread across
more orthogonal axes than CLIP**. CLIP holds with very weak classifier
(small C) or very few examples (small n); P8A drops 9–17pp under the same conditions.

This is consistent with three non-exclusive mechanisms:

1. **Effort's orthogonal regularization spread weight across many singular
   directions.** The FT chain's `lambda_reg=0.01` orthogonality pressure pushes
   features to be uncorrelated, which prevents the head from concentrating
   discriminative weight on a single axis.

2. **P8A FT'd toward the teams-domain axis** (its training contract rewarded
   teams in-domain recall + reals-FPR), which is only weakly correlated with
   viso fake-ness. Anti-shortcut FT specifically pushes signal AWAY from
   pixel-quality and identity-cluster axes — but viso fake-ness lives partly
   on those same axes.

3. **5-stage FT ossification.** Each FT stage incrementally narrows the
   activation manifold. By P8A (4 prior stages + 1 current), the representation
   is highly fitted to teams-domain + camera-signature mitigation, with viso
   discriminability as collateral damage.

**All three interpretations support: revisit "scratch from CLIP" or "very-light FT"
as the next packet direction.** Scratch was rejected once in P13 (cross-domain
regression), but P13 ran without:
- The post-2026-04-26 in_proj-SVD gradient fix (per Stream A audit, this matters
  for early-layer capacity)
- The P22 aug curriculum (sharpness + brightness jitter)
- The Stream-C-evidence-backed eval-targeted aug distribution (laplacian 50–150,
  luma 135–185)

A fresh CLIP-init run with these three additions has substantively different
priors than P13 had.

## Caveats

- **viso-fake-vs-viso-real probe was NOT run** — no viso-real frames are
  available in any per-frame CSV under `score_distribution_2026-05-02/raw_reports/`
  or in the triptych frame manifest. The fake-vs-teams-real probe carries a
  bucket gap that both models trivially solve at default hyperparameters; the
  C/N sweeps are what break the tie. Future improvement: run the same probe
  with viso-fake vs proper_visomaster_real or tv2_visomaster_real — would
  isolate viso-method-vs-viso-source-identity rather than mixing in the bucket
  shortcut.

- The probe is on 512-d post-projection features (what the EFFORT head receives).
  Different layer choices would give different verdicts; the deployment-relevant
  dimension is post-projection, so this is the right one for our question.

- **Cheaper alternative worth testing first**: rank-restoration FT (~$40) that
  concentrates signal back onto top-k singular directions (inverse Effort).
  Would test whether the diffusion is reversible without re-pretraining.

## Artefacts

- `outputs/probe_summary.json` — full sweep, headline, verdict
- `outputs/probe_summary.csv` — flat per-row table
- `outputs/clip_b16_raw__features.npz` — 1100 × 512 frozen CLIP features
- `outputs/p8a__features.npz` — 1100 × 512 frozen P8A features
- `outputs/tsne_combined.csv` — viewer-ready (1100 × 2 models, columns include `local_path` and `model`)
- `outputs/tsne_clip_b16_raw.csv`, `outputs/tsne_p8a.csv` — per-model
- `outputs/sample_manifest.csv` — 1100 frames, 0 download failures
- `_frame_cache/` — 1100 cached PNGs (~220 MB)
- `scripts/01_extract_features.py`, `scripts/02_run_probe.py`, `scripts/03_tsne.py`
- `run.log`
