# IQ per-layer probe — FACTS (2026-05-08)

> **Status: factual-only.** No interpretation. Forbidden words: succeeds,
> fails, wins, promotes, deployment-grade.
>
> **Check (a)** of the IQ-deconvolution program's pre-Stage-2a checks. CPU-only
> (MPS-accelerated). No GPU spend.
>
> **Goal**: where in the OpenCLIP B16 encoder is the IQ representation
> concentrated? Output is a per-(ckpt × layer) IQ probe table that informs
> the GRL hook layer choice for Stage 2a.

---

## 1. Method

### 1.1 Frame set

The 800-frame stratified sample from
`analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/sampled_frames.csv`
(predominantly DEV-split lockbox-substrate frames; 713 dev / 87 lockbox; 30+
identities including 98 Dor-related frames). Same sample as the
`intermediate_layer_probe_2026-04-30` precedent.

### 1.2 Encoder feature extraction

Per OpenCLIP `ViT-B-16-DataComp-XL` `transformer.resblocks` indices
{0, 3, 6, 9, 11} (12 blocks total in B16). For each frame:

- Forward through `model.backbone` with hooks capturing the CLS token (token 0)
  at each of the 5 indices.
- 768-dim CLS feature per (frame × layer).

Cached at `_cache/intermediate__{label}__layer{ix:02d}__n800.npz`.

P8A features were already cached in
`analysis/_features_cache_2026-04-30/intermediate__P8A__*` (re-used). E2B and
P2-D-step3000 features extracted fresh on MPS (~15 sec each ckpt × 5 layers).

### 1.3 IQ panel

Per-frame IQ feature panel (matches Stage 1's primary_6):
`[lap_var, min_dim, luma_mean, color_b_dev, edge_mag, skin_frac]`.

The 800-frame triptych sample had 420 frames in the cross-pool atlas
(`analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet`); the remaining
443 frames had IQ features computed inline from local files using the same
cv2/LAB/YCbCr formulas as `build_iq_atlas.py:per_frame_attrs`. Final panel:
**863 frames** (some atlas pools include both real and fake variants of the
same frame_path, hence the +63 over 800).

### 1.4 Probes

Two parallel probes per (ckpt × layer × IQ feature):

1. **Ridge R²** (5-fold CV): regression `iq_feat ~ layer_features` with
   `Ridge(alpha=1.0)` after L2-normalizing layer features. Reported as
   out-of-fold R².
2. **Logistic-regression AUC** (5-fold CV): binary
   `iq_feat > p50 ~ layer_features` with `LogisticRegression(C=1.0, n_jobs=1)`
   on L2-normalized layer features. Reported as out-of-fold AUC.

Multivariate alignment per (ckpt × layer): average per-feature R² and
mean cosine alignment between the IQ-vector prediction and the actual
IQ-vector (both normalized).

`n_jobs=1` per project memory `feedback_sklearn_njobs.md`.

### 1.5 Sanity checks

- Layer 0 (post-patch-embedding + position) features are identical across the
  three ckpts (FT freezes the patch-embedding stage; only resblock
  parameters were modified). The per-layer-0 R² and AUC values are the same
  to 3 decimal places across P8A, E2B, P2-D — confirms the extraction is
  deterministic and the FT delta is post-block-0.

---

## 2. Per-(ckpt × layer × IQ feature) Ridge R²

### 2.1 P8A_REFERENCE_STEP5000

| iq_feature | layer 0 | layer 3 | layer 6 | layer 9 | layer 11 |
|---|---:|---:|---:|---:|---:|
| color_b_dev | 0.207 | 0.375 | 0.669 | 0.577 | 0.561 |
| edge_mag | 0.127 | 0.588 | 0.796 | 0.700 | 0.698 |
| lap_var | 0.149 | 0.422 | 0.620 | 0.545 | 0.583 |
| luma_mean | 0.467 | 0.420 | 0.637 | 0.563 | 0.593 |
| min_dim | 0.185 | 0.776 | 0.894 | 0.828 | 0.809 |
| skin_frac | 0.397 | 0.577 | 0.741 | 0.734 | 0.759 |

### 2.2 E2B_TOP_N_STEP3200

| iq_feature | layer 0 | layer 3 | layer 6 | layer 9 | layer 11 |
|---|---:|---:|---:|---:|---:|
| color_b_dev | 0.207 | 0.391 | 0.703 | 0.609 | 0.756 |
| edge_mag | 0.127 | 0.597 | 0.800 | 0.720 | 0.818 |
| lap_var | 0.149 | 0.407 | 0.628 | 0.585 | 0.723 |
| luma_mean | 0.467 | 0.418 | 0.658 | 0.590 | 0.733 |
| min_dim | 0.184 | 0.789 | 0.895 | 0.811 | 0.858 |
| skin_frac | 0.397 | 0.579 | 0.754 | 0.747 | 0.819 |

### 2.3 P2_D_FOURIER_PERIODIC_STEP3000

| iq_feature | layer 0 | layer 3 | layer 6 | layer 9 | layer 11 |
|---|---:|---:|---:|---:|---:|
| color_b_dev | 0.207 | 0.378 | 0.685 | 0.578 | 0.736 |
| edge_mag | 0.127 | 0.595 | 0.797 | 0.706 | 0.815 |
| lap_var | 0.149 | 0.412 | 0.624 | 0.583 | 0.717 |
| luma_mean | 0.467 | 0.421 | 0.654 | 0.583 | 0.726 |
| min_dim | 0.185 | 0.788 | 0.893 | 0.808 | 0.853 |
| skin_frac | 0.398 | 0.580 | 0.752 | 0.740 | 0.819 |

---

## 3. Per-(ckpt × layer × IQ feature) binary AUC

Binary target: `iq_feat > p50` (high vs low half of the panel).
Discriminator: 5-fold CV `LogisticRegression`, `n_jobs=1`.

### 3.1 P8A_REFERENCE_STEP5000

| iq_feature_bin | layer 0 | layer 3 | layer 6 | layer 9 | layer 11 |
|---|---:|---:|---:|---:|---:|
| color_b_dev | 0.857 | 0.837 | 0.872 | 0.870 | 0.874 |
| edge_mag | 0.730 | 0.860 | 0.915 | 0.893 | 0.867 |
| lap_var | 0.905 | 0.896 | 0.946 | 0.939 | 0.923 |
| luma_mean | 0.915 | 0.837 | 0.896 | 0.884 | 0.878 |
| min_dim | 0.762 | 0.946 | **0.981** | 0.971 | 0.971 |
| skin_frac | 0.939 | 0.942 | 0.951 | 0.945 | 0.936 |

### 3.2 E2B_TOP_N_STEP3200

| iq_feature_bin | layer 0 | layer 3 | layer 6 | layer 9 | layer 11 |
|---|---:|---:|---:|---:|---:|
| color_b_dev | 0.857 | 0.842 | 0.877 | 0.870 | 0.910 |
| edge_mag | 0.730 | 0.854 | 0.908 | 0.879 | 0.922 |
| lap_var | 0.905 | 0.900 | 0.936 | 0.934 | **0.956** |
| luma_mean | 0.916 | 0.838 | 0.886 | 0.872 | 0.927 |
| min_dim | 0.762 | 0.957 | **0.977** | 0.969 | 0.974 |
| skin_frac | 0.940 | 0.946 | 0.949 | 0.953 | 0.964 |

### 3.3 P2_D_FOURIER_PERIODIC_STEP3000

| iq_feature_bin | layer 0 | layer 3 | layer 6 | layer 9 | layer 11 |
|---|---:|---:|---:|---:|---:|
| color_b_dev | 0.857 | 0.840 | 0.871 | 0.866 | 0.901 |
| edge_mag | 0.730 | 0.854 | 0.910 | 0.875 | 0.917 |
| lap_var | 0.905 | 0.897 | 0.940 | 0.928 | 0.956 |
| luma_mean | 0.915 | 0.838 | 0.892 | 0.866 | 0.918 |
| min_dim | 0.762 | 0.957 | **0.976** | 0.961 | 0.968 |
| skin_frac | 0.939 | 0.945 | 0.948 | 0.953 | 0.962 |

---

## 4. Multivariate per-(ckpt × layer) summary

`avg_per_feature_r2` is the unweighted mean of the 6 per-feature R² rows
above. `vector_cosine_alignment` is the mean cosine similarity between the
6-d predicted IQ vector and the 6-d standardized actual IQ vector across all
863 frames.

| ckpt | layer | avg_per_feature_r2 | vector_cosine_alignment |
|---|---:|---:|---:|
| P8A | 0 | 0.255 | 0.669 |
| P8A | 3 | 0.526 | 0.805 |
| P8A | 6 | **0.726** | 0.867 |
| P8A | 9 | 0.658 | 0.842 |
| P8A | 11 | 0.667 | 0.844 |
| E2B | 0 | 0.255 | 0.669 |
| E2B | 3 | 0.530 | 0.814 |
| E2B | 6 | 0.740 | 0.869 |
| E2B | 9 | 0.677 | 0.860 |
| E2B | 11 | **0.785** | 0.888 |
| P2D | 0 | 0.255 | 0.669 |
| P2D | 3 | 0.529 | 0.817 |
| P2D | 6 | 0.729 | 0.867 |
| P2D | 9 | 0.671 | 0.853 |
| P2D | 11 | **0.782** | 0.882 |

### 4.1 Peak-AUC layer per ckpt (max over 6 IQ features)

Per `outputs/iq_perlayer_summary.json`:

| ckpt | layer with peak per-IQ-feature binary AUC |
|---|---:|
| P8A | layer 6 (peak AUC 0.981 on min_dim_bin) |
| E2B | layer 6 (peak AUC 0.977 on min_dim_bin) |
| P2D | layer 6 (peak AUC 0.976 on min_dim_bin) |

---

## 5. Cross-feature observations

(Numerical cross-cuts; no interpretation.)

1. **Layer 0 invariance across ckpts.** All three checkpoints' layer-0 R²
   and AUC values are identical to 3 decimal places (e.g. min_dim AUC =
   0.762 for all 3; skin_frac AUC = 0.939–0.940). Layer 0 is the
   patch-embedding + positional-embedding output, which is unmodified by
   FT for all three ckpts.

2. **`min_dim` saturates earliest.** Binary AUC of `min_dim > p50` reaches
   0.95+ at layer 3 for all three ckpts (P8A 0.946, E2B 0.957, P2D 0.957).
   Layers 6, 9, 11 stay in the 0.96–0.98 band — small additional gain.

3. **`color_b_dev` and `edge_mag` shift at L11 for E2B and P2D but not P8A.**
   P8A L11 R² for color_b_dev = 0.561 vs E2B 0.756 and P2D 0.736 (Δ +0.18).
   P8A L11 R² for edge_mag = 0.698 vs E2B 0.818 and P2D 0.815 (Δ +0.12).
   Both E2B and P2D have stronger L11 representation of color/edge axes
   than P8A.

4. **`lap_var` increases at L11 for E2B and P2D.** P8A L11 lap_var R² =
   0.583 vs E2B 0.723 and P2D 0.717. Same direction as #3.

5. **L9 dip.** All three ckpts show R² LOWER at L9 than at L6 across
   all 6 IQ features. The dip is largest for `color_b_dev` (P8A −0.092,
   E2B −0.094, P2D −0.107) and `lap_var` (P8A −0.075, E2B −0.043, P2D
   −0.041). Then R² recovers at L11.

6. **P8A vs E2B/P2D divergence.** Average per-feature R² at L11:
   P8A 0.667, E2B 0.785, P2D 0.782. The gap is +0.115 / +0.115. Layers
   0–9 differ only marginally across ckpts (Δ < 0.025). The L11
   divergence aligns with memory `project_per_layer_divergence_2026-05-06`'s
   finding that P8A and E2B share encoder through layer 5 and diverge at
   10-11.

7. **E2B and P2D L11 are very close.** L11 R² difference E2B − P2D ≤ 0.020
   on all 6 IQ features (`color_b_dev` 0.020, `edge_mag` 0.003,
   `lap_var` 0.006, `luma_mean` 0.007, `min_dim` 0.005, `skin_frac` 0.000).
   P2D's L11 IQ representation is nearly identical to E2B's, despite
   different training trajectories.

8. **`skin_frac` saturates at L0.** L0 binary AUC = 0.939–0.940 for all 3
   ckpts. Increases to 0.95+ by L3, then plateaus. Patch+positional
   embedding alone discriminates high-skin-frac from low-skin-frac at
   AUC ≥ 0.94.

9. **Multivariate alignment plateaus at L6 for P8A, climbs at L11 for E2B/P2D.**
   P8A vector_cosine_alignment: L0 0.669, L6 0.867, L11 0.844 (slight drop).
   E2B: L0 0.669, L6 0.869, L11 0.888 (continues climbing).
   P2D: L0 0.669, L6 0.867, L11 0.882 (continues climbing).

---

## 6. Coverage caveats

1. **Layer subset.** Only 5 of 12 resblock indices probed (0, 3, 6, 9, 11).
   Layers 1, 2, 4, 5, 7, 8, 10 not measured. Memory `project_per_layer_divergence_2026-05-06`
   measured P8A vs E2B at all 12 resblocks for cosine similarity but did
   not measure IQ probe AUC at the unsampled layers.

2. **Frame set.** 800-frame DEV-substrate sample (713 dev + 87 lockbox).
   The HDTF substrate is not represented; the lockbox-real fraction is
   small (n=87 across all identities). Per-substrate (DEV vs LOCKBOX vs
   HDTF) per-layer probe was not run.

3. **Probe class.** Linear (Ridge / LR). A nonlinear probe (kernel SVM,
   small MLP) might detect IQ-correlated nonlinear patterns the linear
   probe misses, especially at L0 where the AUC for some IQ features is
   ~0.76 (min_dim) — lower than at L6 (0.98) but possibly an artifact of
   the linear probe's expressiveness rather than absent representation.

4. **L1/L2 reading bias.** No guarantee the 800-frame lockbox sample's IQ
   distribution matches the production substrate. The `min_dim` distribution
   in this sample (per `sampled_frames.csv` `face_pixel_area`) covers
   ~50–512 px range, which may differ from any deployment scenario.

---

## 7. Cross-references

- IQ R² decomp Stage 1: `analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_FACTS_2026-05-08.md`.
- IQ data atlas: `analysis/iq_data_atlas_2026-05-08/IQ_ATLAS_FACTS_2026-05-08.md`.
- Per-layer divergence (cosine, no IQ): memory
  `project_per_layer_divergence_2026-05-06.md`.
- Per-layer feature precedent: `analysis/intermediate_layer_probe_2026-04-30/`.
- Driver: `extract_features.py` (idempotent), `run_probe.py` (idempotent).
- Outputs: `outputs/iq_perlayer_r2.csv`, `outputs/iq_perlayer_binary_auc.csv`,
  `outputs/multivar_alignment.csv`, `outputs/iq_perlayer_summary.json`.

---

## 8. Artifacts

- `_cache/intermediate__{P8A,E2B,P2D}__layer{00,03,06,09,11}__n800.npz` —
  per-(ckpt × layer) CLS features, 768-dim. Reusable.
- `_cache/e2b_top_n_step3200.pth` — E2B checkpoint (940 MB; can be deleted
  after probe is finalized; not needed for re-runs of run_probe.py).
- `outputs/iq_perlayer_r2.csv` — 90 rows: 3 ckpts × 5 layers × 6 IQ features.
- `outputs/iq_perlayer_binary_auc.csv` — 90 rows.
- `outputs/multivar_alignment.csv` — 15 rows: 3 ckpts × 5 layers.
- `outputs/iq_perlayer_summary.json` — peak-AUC layer + per-layer summary.
