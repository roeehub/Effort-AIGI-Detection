# L11 atlas inv_mean — FACTS (2026-05-12)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, unfortunately, remarkably.
> Numbers + tables + cross-references only. Interpretation belongs in `../cpu_diagnostics_2026-05-12_stage_a/STAGE_A_PLUS_PACKET_PLAN_OPINIONS_2026-05-12.md`.
>
> **Scope**: extends `cpu_diagnostics_2026-05-11_a3_atlas_composition/ATLAS_COMPOSITION_FACTS_2026-05-11.md` (T4 chronic_6 inv_mean −0.0315 vs P8A finding) to T5C step1500 + T5C step3500 + T6 step1500. Tests the in-flight open loop `t5c-classifier-capacity-mechanism` (`OPEN_LOOPS.md`).
>
> **Inputs**:
> - L11 CLS feature caches under `analysis/iq_perlayer_probe_2026-05-08/_cache/intermediate__{LABEL}__layer11__n800.npz` for labels: P8A, E2B, T3_S1_step1500, T4_L1_step10500, T5C_periodic_step1500, T5C_periodic_step3500, T6_periodic_step1500
> - Triptych panel: `analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/sampled_frames.csv` (800 frames)
> - IQ atlas (6-axis): `analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet`
> - Compute code: `analysis/cpu_diagnostics_2026-05-11_t67_t5c_probe/compute_inv_mean.py`

---

## 1. Method (mirrors A3 2026-05-11 protocol)

For each ckpt × substrate slice, fit a 5-fold cross-validated logistic-regression probe (`sklearn.linear_model.LogisticRegression`, n_jobs=1 per memory `feedback_sklearn_njobs.md`):

- **forgery_auc**: AUC of probe predicting `is_real_vs_fake` from L11 CLS features
- **mean_shortcut_auc**: mean of probe AUCs across 5 shortcut signals — `is_dor`, `is_chronic_6` (where applicable), `lap_var_high`, `min_dim_high`, `face_size_high`
- **inv_mean**: `forgery_auc − mean_shortcut_auc` — higher is more invariant to shortcut while preserving forgery signal

Substrate slices:
- `full_n800` — entire 800-frame triptych
- `dev_only` — split=="dev" rows (n=713)
- `lockbox_only` — split=="lockbox" rows (n=87)
- `chronic_6_only` — chronic_6 identities (n=282)
- `non_chronic_only` — non-chronic identities (n=518)

## 2. Full-triptych inv_mean (sorted desc)

Source: `outputs/L11_inv_mean_summary.csv`.

| Ckpt | n | forgery_auc | mean_shortcut_auc | inv_mean |
|---|---:|---:|---:|---:|
| T4_L1_step10500 | 800 | 0.9869 | 0.9388 | **+0.0482** |
| T6_periodic_step1500 | 800 | 0.9903 | 0.9527 | +0.0376 |
| T5C_periodic_step3500 | 800 | 0.9893 | 0.9559 | +0.0334 |
| P8A | 800 | 0.9738 | 0.9425 | +0.0313 |
| T3_S1_step1500 | 800 | 0.9913 | 0.9625 | +0.0288 |
| T5C_periodic_step1500 | 800 | 0.9873 | 0.9596 | +0.0277 |
| E2B | 800 | 0.9940 | 0.9711 | +0.0229 |

## 3. Per-substrate inv_mean

Source: `outputs/per_ckpt_inv_mean.csv`.

### 3.1 dev_only (n=713; 277 fake, 436 real)

| Ckpt | forgery_auc | mean_shortcut_auc | inv_mean |
|---|---:|---:|---:|
| T4_L1_step10500 | 0.9940 | 0.9333 | +0.0608 |
| P8A | 0.9828 | 0.9318 | +0.0509 |
| T5C_periodic_step3500 | 0.9941 | 0.9569 | +0.0373 |
| T3_S1_step1500 | 0.9958 | 0.9585 | +0.0373 |
| T5C_periodic_step1500 | 0.9932 | 0.9610 | +0.0322 |
| T6_periodic_step1500 | 0.9946 | 0.9526 | +0.0420 |
| E2B | 0.9958 | 0.9688 | +0.0269 |

### 3.2 lockbox_only (n=87; 47 fake, 40 real)

| Ckpt | forgery_auc | mean_shortcut_auc | inv_mean |
|---|---:|---:|---:|
| T3_S1_step1500 | 0.9867 | 0.9905 | −0.0038 |
| T5C_periodic_step1500 | 0.9814 | 0.9913 | −0.0099 |
| T4_L1_step10500 | 0.9766 | 0.9853 | −0.0087 |
| T5C_periodic_step3500 | 0.9847 | 0.9909 | −0.0062 |
| T6_periodic_step1500 | 0.9777 | 0.9810 | −0.0033 |
| P8A | 0.9489 | 0.9679 | −0.0189 |
| E2B | 1.0000 | 0.9972 | +0.0028 |

### 3.3 **chronic_6_only (n=282; 41 fake, 241 real) — the A3-load-bearing metric**

| Ckpt | forgery_auc | mean_shortcut_auc | inv_mean |
|---|---:|---:|---:|
| **P8A** | 0.9915 | 0.9470 | **+0.0445** |
| T6_periodic_step1500 | 0.9856 | 0.9604 | +0.0252 |
| T3_S1_step1500 | 0.9891 | 0.9656 | +0.0235 |
| E2B | 0.9947 | 0.9717 | +0.0230 |
| **T5C_periodic_step3500** | 0.9838 | 0.9614 | **+0.0224** |
| T4_L1_step10500 | 0.9561 | 0.9430 | +0.0130 |
| T5C_periodic_step1500 | 0.9796 | 0.9685 | +0.0110 |

### 3.4 non_chronic_only (n=518; 283 fake, 235 real)

| Ckpt | forgery_auc | mean_shortcut_auc | inv_mean |
|---|---:|---:|---:|
| T4_L1_step10500 | 0.9945 | 0.9424 | +0.0521 |
| T6_periodic_step1500 | 0.9958 | 0.9570 | +0.0388 |
| T5C_periodic_step3500 | 0.9943 | 0.9610 | +0.0332 |
| T3_S1_step1500 | 0.9941 | 0.9637 | +0.0304 |
| T5C_periodic_step1500 | 0.9931 | 0.9659 | +0.0272 |
| E2B | 0.9962 | 0.9703 | +0.0259 |
| P8A | 0.9698 | 0.9508 | +0.0191 |

## 4. chronic_6 inv_mean delta from T4 (open-loop test)

The `chronic_6-feature-regression-on-t4` open loop (`OPEN_LOOPS.md`, severity low/in-progress) records T4_L1_step10500 chronic_6 inv_mean at −0.0315 absolute vs P8A. The proposed T5C lever (hidden_dim 256→1024) was hypothesized to restore chronic_6 invariance. Per §3.3:

| Ckpt | chronic_6 inv_mean | Δ vs P8A (+0.0445) | Δ vs T4 (+0.0130) |
|---|---:|---:|---:|
| P8A | +0.0445 | (reference) | +0.0315 |
| T4_L1_step10500 | +0.0130 | −0.0315 | (reference) |
| T5C_step1500 | +0.0110 | −0.0335 | −0.0020 |
| T5C_step3500 | +0.0224 | −0.0221 | +0.0094 |
| T6_step1500 | +0.0252 | −0.0193 | +0.0122 |
| T3_S1_step1500 | +0.0235 | −0.0210 | +0.0105 |
| E2B | +0.0230 | −0.0215 | +0.0100 |

## 5. Output artifacts

- `outputs/per_ckpt_inv_mean.csv` — 35 rows = 7 ckpts × 5 slices. Columns: ckpt, slice, n, n_fake, n_real, forgery_auc, mean_shortcut_auc, inv_mean, auc_is_dor, auc_is_chronic_6, auc_lap_var_high, auc_min_dim_high, auc_face_size_high.
- `outputs/L11_inv_mean_summary.csv` — 7-row full_n800 summary (§2).
- L11 feature caches at `analysis/iq_perlayer_probe_2026-05-08/_cache/intermediate__{LABEL}__layer11__n800.npz` (extended this session for T5C_step1500, T5C_step3500, T6_step1500 — extraction script `extract_features_l11.py`).

## 6. Caveats

- n=282 for chronic_6_only and n=87 for lockbox_only are small; 95% CI for AUC at p=0.95 ≈ ±0.02 absolute (Mann-Whitney noise band).
- chronic_6 contains 5/6 of the named patterns (`Roy_D` is excluded — 0 frames matched in the 800-frame panel since Roy_D was not in the original triptych sample).
- The shortcut AUC is computed on each axis where labels are well-defined (`is_dor` is degenerate within chronic_6_only since most chronic-6 identities ARE dor-cluster); the inv_mean aggregates across whichever axes are valid per slice.
- `is_chronic_6` axis is omitted in `chronic_6_only` and `non_chronic_only` slices (degenerate).

## 7. Direct observations

1. P8A has the highest chronic_6 inv_mean of the 7 scored ckpts (+0.0445); the next-highest is T6_step1500 (+0.0252) at a delta of −0.0193 absolute (§3.3).
2. T5C_step3500 chronic_6 inv_mean (+0.0224) is between T4_step10500 (+0.0130, the lowest in this set) and P8A (+0.0445) (§3.3, §4).
3. T5C_step3500 lifts chronic_6 inv_mean +0.0094 absolute over T4_step10500 (§4).
4. T5C_step1500 chronic_6 inv_mean (+0.0110) is below T4_step10500's (+0.0130) — among the 7 ckpts the only one below T4 (§3.3).
5. On the lockbox_only slice (n=87), P8A has the lowest inv_mean (−0.0189) and E2B has the highest (+0.0028) (§3.2).
6. On the non_chronic_only slice (n=518), T4_step10500 has the highest inv_mean (+0.0521) and P8A the lowest (+0.0191) (§3.4).
7. On the dev_only slice (n=713), T4_step10500 has the highest inv_mean (+0.0608), and the 6 ckpts including T5C/T6/T3 all fall between +0.0269 and +0.0509 (§3.1).
8. Full-panel inv_mean ranking (§2): T4 > T6 > T5C_step3500 > P8A > T3_S1 > T5C_step1500 > E2B. The full-panel ordering and the chronic_6 ordering disagree on every ckpt except E2B (full lowest = chronic lowest for E2B).
