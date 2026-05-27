# Atlas substrate decomposition + triptych composition audit — FACTS (2026-05-11)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, unfortunately, remarkably.
> Numbers + tables + cross-references only. Interpretation belongs elsewhere.
>
> **Scope**: CPU diagnostic A3. Audits the 800-frame triptych used by `forgery_signal_atlas` / per-layer `inv_mean` and re-fits inv_mean per substrate slice for {`P8A`, `E2B`, `T4_L1_step10500`, `T3_S1_step1500`} on cached L11 features.
>
> **Inputs**:
> - Triptych manifest: `analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/sampled_frames.csv` rows 0..799.
> - L11 feature caches: `analysis/iq_perlayer_probe_2026-05-08/_cache/intermediate__{P8A,E2B,T4_L1_step10500,T3_S1_step1500}__layer11__n800.npz`.
> - Lockbox suite frame reports: `analysis/cpu_diagnostics_2026-05-10/_t4_scorecard_local/teams_{real,fake}_all_lockbox_{p8a_reference_step5000,t4_lambda1_top_n_step10500,t4_lambda2_periodic_step1500}_frames_report.csv`.
> - IQ panel: `analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet` (`frame_path` joined to `gcs_uri`; deduped by `frame_path` before merge).
> - Atlas inv_mean reference: `analysis/cpu_diagnostics_2026-05-10/outputs/L11_inv_mean_with_t4.csv`.
> - Promotion-contract FACTS reference: `analysis/t4_eval_2026-05-11/RESULTS_FACTS_2026-05-11.md`.
>
> **Scripts**:
> - `analysis/cpu_diagnostics_2026-05-11_a3_atlas_composition/_run_a3_atlas_composition.py` — substrate breakdown + per-slice inv_mean.
> - `analysis/cpu_diagnostics_2026-05-11_a3_atlas_composition/_run_a3_head_vs_probe_lockbox.py` — trained-head AUC on the 87 triptych-lockbox frames vs full lockbox suite.

---

## 1. Question

For the 800-frame triptych used to compute `forgery_signal_atlas` and `L11_inv_mean_with_t4`:

1. What is the substrate composition? — Job 1.
2. Recompute `inv_mean` per substrate slice for {P8A, E2B, T4_L1_step10500, T3_S1_step1500}. — Job 2.
3. Compare per-substrate `Δinv_mean(T4 − P8A)` to per-substrate trained-head `ΔAUC` from §7.1 of `RESULTS_FACTS_2026-05-11.md`. — Job 3.
4. Trained-head AUC on the 87 triptych-lockbox frames vs the full lockbox suite. — Job 4.
5. Are any triptych frames also in the lockbox suite? — Job 5.

## 2. Method

- Panel: first 800 rows of `sampled_frames.csv` (same slice the atlas scripts use).
- Substrate columns: `split ∈ {dev, lockbox}` from the manifest; `is_chronic_6 = identity_key matches one of {Roy_D, PC_Generator, bla_bla_chow, Md_noyn_Sharker, dor_shkedi, healthy_dor}`; `is_dor = "dor" in identity_key.lower()`.
- `inv_mean` definition (same as `per_layer_inv_mean.py`): `forgery_AUC − mean(shortcut_AUCs)` where shortcuts = {`is_dor`, `is_chronic_6`, `lap_var_high`, `min_dim_high`, `face_size_high`}.
- All 6 probes (forgery + 5 shortcuts) are 5-fold stratified-CV LR on L2-normalised L11 features (`solver=lbfgs, C=1.0, n_jobs=1`). When a stratum has fewer positives than splits the fold count is reduced to `min(5, n_pos, n_neg)`; below 3 the cell returns NaN and is excluded from the mean.
- Shortcut binarisation thresholds (lap_var_high, min_dim_high, face_size_high) are pool-medians computed on the FULL 800-frame panel (matches the original atlas methodology). The subset masks are applied only to ROWS, not thresholds.
- Slices recomputed: `full_n800`, `dev_only`, `lockbox_only`, `chronic_6_only`, `non_chronic_only`.
- For `chronic_6_only` and `non_chronic_only`, `is_chronic_6` probe is degenerate within the slice and is reported as NaN; mean_shortcut_AUC is averaged over the remaining 4 axes.

## 3. Job 1 — Triptych substrate composition

Source: `triptych_substrate_breakdown.csv`.

### 3.1 Headline

| Slice | n | n_real | n_fake | n_chronic_6 | n_dor | n_identities |
|---|---:|---:|---:|---:|---:|---:|
| all | 800 | 476 | 324 | 282 | 98 | 33 |
| split=dev | 713 | 436 | 277 | 237 | 73 | 28 |
| split=lockbox | 87 | 40 | 47 | 45 | 25 | 5 |

Share: dev = 89.1%, lockbox = 10.9%.

### 3.2 Cross-tab split × label

| Split | label=real | label=fake |
|---|---:|---:|
| dev | 436 | 277 |
| lockbox | 40 | 47 |

### 3.3 Chronic-6 composition

| Slice | n | of which split=dev | of which split=lockbox |
|---|---:|---:|---:|
| is_chronic_6=1 | 282 | 237 | 45 |
| is_chronic_6=0 | 518 | 476 | 42 |

Chronic-6 fraction: 35.2% of the full pool, 33.2% of dev, 51.7% of lockbox.

### 3.4 Method composition

Source: `triptych_method_breakdown.csv`.

Top methods (n ≥ 10): `teams_real` (476), `deeplive_enhanced` (59), `teams_capture_cam_test_s35` (41), `teams_capture_cam_test_s33` (36), `teams_capture_noyn_sharker_s23` (36), `teams_capture_cam_test_s32` (26), `teams_capture_test_cam_s53` (19), `teams_capture_cam_test_s46` (16), `teams_capture_test_cam_s76` (15), `teams_flat_xiang_xiang2_feng` (14), `teams_capture_pc_generator_s3` (13), `teams_capture_test_cam_s73` (11), `teams_capture_pc_generator_s15` (11), `teams_capture_cam_test_s38` (10). Tail: `teams_capture_dor_shkedi_s16` (9), `teams_capture_pc_generator_s9` (5), `teams_capture_pc_generator_s4` (3). Total methods = 17.

No `visomaster_enhanced_*`, no `teams_fake_*_lockbox`, no `teams_real_dor_dev` rows appear under those exact method strings; the lockbox-split rows carry methods `teams_real` (40) + `teams_capture_*` (47).

## 4. Job 2 — Per-substrate inv_mean recomputation

Source: `per_substrate_inv_mean.csv`.

### 4.1 forgery_AUC per ckpt per slice

| Slice | P8A | E2B | T4_L1_step10500 | T3_S1_step1500 |
|---|---:|---:|---:|---:|
| full_n800 | 0.9738 | 0.9940 | 0.9869 | 0.9913 |
| dev_only | 0.9828 | 0.9958 | 0.9940 | 0.9958 |
| lockbox_only (n=87) | 0.9489 | 1.0000 | 0.9766 | 0.9867 |
| chronic_6_only | 0.9915 | 0.9947 | 0.9561 | 0.9891 |
| non_chronic_only | 0.9698 | 0.9962 | 0.9945 | 0.9941 |

### 4.2 inv_mean per ckpt per slice

| Slice | P8A | E2B | T4_L1_step10500 | T3_S1_step1500 |
|---|---:|---:|---:|---:|
| full_n800 | +0.0313 | +0.0229 | +0.0482 | +0.0288 |
| dev_only | +0.0509 | +0.0269 | +0.0608 | +0.0373 |
| lockbox_only (n=87) | −0.0189 | +0.0028 | −0.0087 | −0.0038 |
| chronic_6_only | +0.0445 | +0.0230 | +0.0130 | +0.0235 |
| non_chronic_only | +0.0191 | +0.0259 | +0.0521 | +0.0304 |

Cross-reference: `full_n800` row matches the L11 entries in `analysis/cpu_diagnostics_2026-05-10/outputs/L11_inv_mean_with_t4.csv` within ≤ 0.007 absolute. The two implementations differ slightly because the published atlas uses a panel built by `run_analyses.py` (which can drop rows missing IQ or with `is_no_face`); see §10 caveat.

### 4.3 Δinv_mean (T4_L1_step10500 − P8A) per slice

| Slice | ΔforgeryAUC | Δmean_shortcut | Δinv_mean |
|---|---:|---:|---:|
| full_n800 | +0.0131 | −0.0038 | +0.0169 |
| dev_only | +0.0113 | +0.0014 | +0.0098 |
| lockbox_only (n=87) | +0.0277 | +0.0174 | +0.0103 |
| chronic_6_only | −0.0354 | −0.0039 | −0.0315 |
| non_chronic_only | +0.0246 | −0.0085 | +0.0331 |

### 4.4 Δinv_mean (E2B − P8A) per slice

| Slice | Δinv_mean |
|---|---:|
| full_n800 | −0.0084 |
| dev_only | −0.0240 |
| lockbox_only (n=87) | +0.0217 |
| chronic_6_only | −0.0215 |
| non_chronic_only | +0.0068 |

## 5. Job 3 — Per-substrate inv_mean Δ vs trained-head AUC Δ

Trained-head AUC reference from `RESULTS_FACTS_2026-05-11.md` §7.1 (Mann-Whitney AUC, video-level `avg_video_prob`; positive = fake, negative = real).

### 5.1 Direction-of-effect table

| Cell | Trained-head AUC ΔT4−P8A (full suite, video-level) | Triptych-slice ΔforgeryAUC (LR-probe, frame-level) | Triptych-slice Δinv_mean | Directional consistency |
|---|---:|---:|---:|:--|
| dev (teams_fake vs teams_real) | +0.0369 | dev_only +0.0113 | dev_only +0.0098 | same sign |
| dev (deeplive vs teams_real) | +0.0787 | (no deeplive-only slice) | — | n/a |
| dev (viso vs teams_real) | +0.0815 | (no viso slice in triptych) | — | n/a |
| lockbox (teams_fake vs teams_real, full suite n=253/1361 video-level) | **−0.1736** | **lockbox_only +0.0277** | **lockbox_only +0.0103** | **opposite sign** |

The Job 2 / 3 lockbox-slice triptych LR-probe assigns T4 a HIGHER forgery_AUC than P8A; the full-suite trained-head AUC assigns T4 a LOWER AUC than P8A by 0.1736 absolute. The two assessments disagree in sign on the lockbox slice.

## 6. Job 4 — Trained-head AUC on the 87 triptych-lockbox frames

Source: `trained_head_vs_probe_lockbox.csv`. AUC computed via `sklearn.metrics.roc_auc_score` on `frame_prob` (per-frame trained-head score from the contract reports).

### 6.1 Trained-head AUC, full lockbox vs triptych-lockbox subset

| Ckpt | n_real_full / n_fake_full | trained-head AUC, full lockbox frames | n_real_subset / n_fake_subset | trained-head AUC, triptych-lockbox subset (n=87) |
|---|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 1418 / 425 | 0.9191 | 40 / 47 | 0.8005 |
| T4_LAMBDA1_TOP_N_STEP10500 | 1418 / 425 | 0.7648 | 40 / 47 | 0.7862 |
| T4_LAMBDA2_PERIODIC_STEP1500 | 1418 / 425 | 0.9517 | 40 / 47 | 0.9282 |

### 6.2 Δ trained-head AUC on the 87-frame subset

| Comparison | Δ on subset | Δ on full suite (frame-level) | Δ on full suite (video-level, §7.1) |
|---|---:|---:|---:|
| T4_L1_step10500 − P8A | −0.0143 | −0.1543 | −0.1736 |
| T4_L2_step1500 − P8A | +0.1277 | +0.0326 | (n/a §7.1) |

### 6.3 Triptych-lockbox subset score medians (trained-head)

| Ckpt | real_p50 (n=40) | fake_p50 (n=47) |
|---|---:|---:|
| P8A | 0.0558 | 0.7547 |
| T4_L1_step10500 | 0.4285 | 0.7901 |
| T4_L2_step1500 | 0.6231 | 0.8231 |

## 7. Job 5 — Triptych ↔ lockbox-suite overlap audit

Source: `atlas_lockbox_overlap.csv`. Intersection of `gcs_uri` (triptych) with `frame_path` (lockbox-suite frame reports).

| Comparison | n |
|---|---:|
| triptych unique uris | 800 |
| triptych split=lockbox unique uris | 87 |
| triptych split=dev unique uris | 713 |
| lockbox suite real frames | 1418 |
| lockbox suite fake frames | 425 |
| lockbox suite all frames | 1843 |
| triptych(split=lockbox) ∩ lockbox_suite_real | 40 |
| triptych(split=lockbox) ∩ lockbox_suite_fake | 47 |
| triptych(split=lockbox) ∩ lockbox_suite_all | 87 |
| triptych(split=lockbox) NOT in lockbox_suite | 0 |
| triptych(split=dev) ∩ lockbox_suite_all | 0 |

All 87 triptych-`lockbox` rows are also in the lockbox suite (40 in real, 47 in fake). None of the 713 triptych-`dev` rows are in the lockbox suite.

## 8. Output artifacts

- `triptych_substrate_breakdown.csv` — Job 1.
- `triptych_method_breakdown.csv` — Job 1 method roll-up.
- `per_substrate_inv_mean.csv` — Job 2.
- `trained_head_vs_probe_lockbox.csv` — Job 4.
- `atlas_lockbox_overlap.csv` — Job 5.

## 9. Direct observations

1. The 800-frame triptych is 89.1% `split=dev` and 10.9% `split=lockbox` by row count (§3.1).
2. `forgery_signal_atlas_with_t4.csv` published `inv_mean` for T4_L1_step10500 at L11 = +0.04135; our re-fit on `full_n800` matches within +0.0069 absolute (this run: +0.04819; see §10 caveat for the implementation gap). The published L11 inv_mean for P8A is +0.02662; this run: +0.03132. Both ckpts shift by similar amounts so the published `Δinv_mean(T4 − P8A) = +0.01473` and this run's `+0.01686` agree on rank-order and sign.
3. Recomputed on the 87-frame `lockbox_only` slice, the LR-probe `inv_mean` is negative for 3 of 4 ckpts (P8A −0.0189, T4_L1_step10500 −0.0087, T3_S1_step1500 −0.0038; E2B +0.0028) — the lockbox slice does not reproduce the positive `inv_mean` signal seen on the dev slice (§4.2).
4. On the lockbox slice, the LR-probe `forgery_AUC` increases monotonically with mean-shortcut AUC for these 4 ckpts (P8A 0.949/0.968, E2B 1.000/0.997, T4 0.977/0.985, T3 0.987/0.990), keeping `inv_mean` near or below zero (§4.2).
5. The published-atlas direction `Δinv_mean(T4 − P8A) > 0` holds on every triptych slice except `chronic_6_only` (where it inverts to −0.0315) (§4.3).
6. On the 87 lockbox-only triptych rows, the cross-validated LR-probe assigns T4_L1_step10500 a HIGHER forgery_AUC than P8A (+0.0277), while the trained-head AUC on the SAME 87 rows is LOWER for T4 by 0.0143 (§5.1, §6.1).
7. On the full lockbox suite (frame-level, 1418+425), trained-head AUC drops from P8A 0.9191 to T4_L1_step10500 0.7648 (Δ = −0.1543); the 87-row triptych subset reproduces 9.3% of this drop in absolute magnitude (Δsubset = −0.0143) (§6.2).
8. T4_L1_step10500's `real_p50` trained-head score on the 87 triptych-lockbox reals = 0.4285 vs P8A 0.0558 (§6.3); same direction as the full-suite percentile shift documented in `RESULTS_FACTS_2026-05-11.md` §6.1.
9. All 87 triptych-`lockbox` GCS URIs are present in the lockbox-suite frame reports (40 in real, 47 in fake); no triptych-`dev` rows overlap with the lockbox suite (§7).
10. Triptych contains 0 frames from the `visomaster_enhanced_*` or `teams_real_dor_dev` (those exact method strings) according to the method roll-up; however `is_dor=1` flags 98 rows (12.3% of the pool) on identity-substring (§3.4 + §3.3).

## 10. Caveats

1. **Published vs re-fit inv_mean numerical drift** — published L11 inv_mean for T4_L1_step10500 = 0.04135 (`L11_inv_mean_with_t4.csv`); this run's `full_n800` = 0.04819. The +0.007 gap reflects panel-building differences: `run_analyses.py` drops rows where `is_no_face=True` or where IQ inline-compute failed, and uses a different chronic-6 list materialisation. Sign and rank-order are preserved.
2. **Block-tie inflation does not apply here** — unlike the isotonic-calibration caveat in `RESULTS_FACTS_2026-05-11.md` §2, all probes in this report use raw LR scores; AUC is exact.
3. **LR-probe vs trained-head divergence** — §5.1 and §6.1 disagree in sign on the lockbox slice. This is consistent with the standard observation that LR probes recover feature-level separability while trained-head AUC reflects whether the learned linear classifier actually exploits that separability at the deployed parameters.
4. **n=87 lockbox-only slice is small** — for shortcut probes that have ≤ 10 positives in a fold (e.g. `is_chronic_6` collapses to a single class in some slices), the probe is reported as NaN and excluded from mean_shortcut. Frame-level lockbox AUCs from the contract reports (n=1843, §6.1) are the load-bearing reference.
5. **5-fold CV variance not quantified** — single random_state=0 fit per slice. Repeat-runs could give ±0.005 noise on AUC values.

---

## Summary (self-contained, 5 bullets)

- Bullet 1: The 800-frame triptych is **89.1% dev (713 frames) / 10.9% lockbox (87 frames)**; 35.2% chronic-6 overall, 33.2% inside dev and 51.7% inside the lockbox-split rows. Method-wise it is `teams_real` (476) + 16 method strings dominated by `teams_capture_*` Teams-pipeline variants, plus 59 `deeplive_enhanced` rows. No `visomaster_enhanced_*` rows; no `teams_real_dor_dev` rows under that exact method tag.
- Bullet 2: Per-substrate LR-probe **Δinv_mean (T4_L1_step10500 − P8A) is positive on every slice except chronic_6_only**: full_n800 +0.0169, dev_only +0.0098, lockbox_only +0.0103, non_chronic_only +0.0331, chronic_6_only −0.0315. T4's "inv_mean ↑" signal holds on both the dev-slice and the (small) lockbox-slice of the triptych under the same LR-probe definition.
- Bullet 3: The per-substrate LR-probe Δinv_mean **does NOT track** the trained-head ΔAUC. Trained-head full-suite lockbox AUC drops 0.1736 absolute (P8A 0.9355 → T4 0.7619, §7.1 of `RESULTS_FACTS_2026-05-11.md`); on the same 87 triptych-lockbox frames the trained head moves 0.0143 in the same negative direction, while the cross-validated LR probe on identical L11 features moves +0.0277 in the opposite direction. The disagreement is feature-vs-head, not feature-vs-substrate.
- Bullet 4: All 87 triptych-`lockbox` GCS URIs intersect the lockbox-suite frame reports (40 in real, 47 in fake); 0 triptych-`dev` URIs leak into the lockbox suite. The triptych is sampling 87/1843 = 4.7% of the lockbox-suite frame inventory, and 100% of the triptych's "lockbox" rows are inside the deployed lockbox suite.
- Bullet 5: Caveats: (a) published `L11_inv_mean_with_t4.csv` and this run's `full_n800` differ by +0.0069 absolute (no-face filtering and panel-building differences); (b) n=87 lockbox-only slice degrades 5-fold CV to ≤ 3-fold for shortcuts with few positives; (c) single random_state=0 LR probe — variance not quantified; (d) `viso_macro_enhanced_dev` and `teams_real_dor_dev` are not represented in the triptych under those exact method strings (identity-substring `is_dor=1` flags 98 rows but these are dor-labelled across multiple buckets, not the suite-specific dor-dev cell).
