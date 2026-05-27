# CPU Job 1 — L11 distance map from P8A baseline — FACTS (2026-05-12)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, unfortunately, remarkably.
> Numbers + tables + cross-references only. Interpretation belongs in `STAGE_A_PLUS_PACKET_PLAN_OPINIONS_2026-05-12.md`.
>
> **Scope**: per-frame L2 + cosine distance of each candidate ckpt's L11 CLS features from P8A's L11 CLS features on the 800-frame triptych. Companion to `cpu_diagnostics_2026-05-11_t67_t5c_probe/INV_MEAN_FACTS_2026-05-12.md` (which reports per-slice probe AUCs; this doc reports raw L11 distances).
>
> **Inputs**:
> - L11 feature caches: `analysis/iq_perlayer_probe_2026-05-08/_cache/intermediate__{P8A,T5C_periodic_step3500,T3_S1_step1500,T4_L1_step10500}__layer11__n800.npz`
> - Triptych panel: `analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/sampled_frames.csv`
> - IQ atlas: `analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet`
> - Compute code: `job1_l11_distance_map.py`

---

## 1. Method

For each frame index `i` in the 800-frame triptych (intersected with the common valid_idx across all 4 ckpts):
- `l2_<ckpt>(i) = ||L11_<ckpt>[i] − L11_P8A[i]||₂`
- `cosdist_<ckpt>(i) = 1 − cos(L11_<ckpt>[i], L11_P8A[i])`

Distances are computed in 512-d CLIP B16 L11 CLS feature space (post-ln_post, post-proj).

## 2. Mean L2 distance by cohort

n_common = 800 frames (all 4 ckpts valid).

| Cohort | n | T5C_step3500 | T3_S1_step1500 | T4_L1_step10500 |
|---|---:|---:|---:|---:|
| ALL | 800 | 23.62 | 20.95 | 31.29 |
| real | 476 | 21.41 | 19.80 | 28.12 |
| fake | 324 | 26.86 | 22.66 | 35.96 |
| chronic_6 real | 241 | 20.17 | 17.08 | 26.19 |
| chronic_6 fake | 41 | 30.16 | 28.90 | 40.98 |
| healthy real | 235 | 22.68 | 22.58 | 30.09 |
| healthy fake | 283 | 26.38 | 21.75 | 35.23 |
| lockbox real | 40 | 17.40 | **9.07** | 19.07 |
| lockbox fake | 47 | 25.09 | 18.56 | 27.87 |
| dev real | 436 | 21.77 | 20.78 | 28.95 |
| dev fake | 277 | 27.16 | 23.35 | 37.34 |

## 3. Median cosine distance by cohort

| Cohort | n | T5C_step3500 | T3_S1_step1500 | T4_L1_step10500 |
|---|---:|---:|---:|---:|
| ALL | 800 | 0.5034 | 0.4480 | 0.6547 |
| real | 476 | 0.5057 | 0.4847 | 0.5479 |
| fake | 324 | 0.4989 | 0.3966 | 0.7498 |
| chronic_6 real | 241 | 0.5053 | 0.4013 | 0.5847 |
| chronic_6 fake | 41 | 0.4814 | 0.4556 | 0.7323 |
| healthy real | 235 | 0.5063 | 0.4954 | 0.5198 |
| healthy fake | 283 | 0.5096 | 0.3877 | 0.7534 |
| lockbox real | 40 | 0.5108 | **0.1267** | 0.5743 |
| lockbox fake | 47 | 0.5893 | 0.2944 | 0.7364 |
| dev real | 436 | 0.5055 | 0.4916 | 0.5452 |
| dev fake | 277 | 0.4848 | 0.4048 | 0.7524 |

## 4. Per-chronic-identity L2 distance (T5C_step3500 − P8A on triptych)

| Identity | real_n | mean l2(T5C−P8A) | fake_n | mean l2(T5C−P8A) |
|---|---:|---:|---:|---:|
| Roy_D | 0 | n/a | 0 | n/a |
| PC_Generator | 83 | 20.59 | 32 | 31.71 |
| bla_bla_chow | 51 | 16.69 | 0 | n/a |
| Md_noyn_Sharker | 77 | 22.68 | 0 | n/a |
| dor_shkedi | 30 | 18.47 | 9 | 24.63 |
| healthy_dor | 0 | n/a | 0 | n/a |

(Roy_D and healthy_dor have 0 rows in the triptych sample; see CHRONIC_6 panel coverage caveat in `INV_MEAN_FACTS_2026-05-12.md` §6.)

## 5. L2(T5C_step3500 − P8A) on real frames, binned by IQ quartile

Quartiles computed within the real-cohort subset.

| IQ axis | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| lap_var | 22.95 | 21.55 | 20.12 | 19.82 |
| min_dim | 18.98 | 19.41 | 24.55 | 21.66 |
| color_a_dev | 17.33 | 23.90 | 21.30 | 21.94 |
| saturation_mean | 17.80 | 21.08 | 23.79 | 21.85 |

## 6. Top 10 frames by L2(T5C_step3500 − P8A)

Source: `outputs/l11_distance_per_frame.csv` sorted desc by `l2_T5C_periodic_step3500`.

| chronic_id | is_real | is_lockbox | min_dim | lap_var | color_a_dev | l2(T5C−P8A) | cosdist(T5C−P8A) |
|---|---:|---:|---:|---:|---:|---:|---:|
| (none) | 0 | 0 | n/a | n/a | n/a | 42.59 | 0.359 |
| (none) | 0 | 0 | 336 | 32.20 | 12.21 | 41.90 | 0.382 |
| (none) | 0 | 0 | n/a | n/a | n/a | 41.18 | 0.370 |
| (none) | 0 | 0 | 340 | 32.21 | 12.49 | 40.87 | 0.356 |
| (none) | 0 | 0 | 339 | 32.48 | 12.19 | 40.55 | 0.361 |
| (none) | 0 | 0 | n/a | n/a | n/a | 40.41 | 0.370 |
| (none) | 0 | 0 | n/a | n/a | n/a | 39.45 | 0.366 |
| (none) | 0 | 0 | 329 | 24.84 | 13.13 | 39.21 | 0.381 |
| (none) | 0 | 0 | n/a | n/a | n/a | 39.10 | 0.367 |
| (none) | 0 | 0 | n/a | n/a | n/a | 38.81 | 0.367 |

`is_real=0` is fake; `chronic_id="(none)"` means the frame's `identity_key` does not match any of the 6 chronic patterns.

## 7. Output artifacts

- `outputs/l11_distance_per_frame.csv` — 800 rows × {row_ix, gcs_uri, label, split, identity_key, chronic_id, is_chronic_6, is_real, is_lockbox, min_dim, lap_var, luma_mean, saturation_mean, color_a_dev, l2_<ckpt>, cos_<ckpt>, cosdist_<ckpt> for each of T5C_step3500 / T3_S1_step1500 / T4_L1_step10500}.

## 8. Caveats

- Distance metric is in raw 512-d CLIP L11 CLS feature space, not L2-normalized. Cosine distance is the normalization-invariant alternative.
- Top-10 frames in §6 have `min_dim` reported as `n/a` (IQ atlas join missed); for the 7 of 10 with valid IQ data, all have lap_var ∈ [24.8, 32.5] (low end of sharpness distribution) and color_a_dev ∈ [12.19, 13.13].
- The triptych panel was sampled in 2026-04-30 for embedding probes; its identity composition is not balanced (e.g., 0 Roy_D frames, only 30 dor_shkedi).
- L11 feature extraction for the new ckpts (T5C_step1500, T5C_step3500, T6_step1500) was performed 2026-05-12 via `cpu_diagnostics_2026-05-11_t67_t5c_probe/extract_features_l11.py` — same code path as the pre-existing P8A/T3_S1/T4 caches.

## 9. Direct observations

1. T3_S1_step1500 has the smallest mean L2 distance from P8A across all 11 cohort cells in §2; the smallest specific cell is lockbox_real at L2=9.07 (cosdist=0.13) (§2, §3).
2. T5C_step3500's mean L2 distance from P8A is 23.62 (full triptych); the corresponding cosine distance median is 0.5034 (§2, §3).
3. T4_L1_step10500's mean L2 distance from P8A is 31.29; the corresponding cosine distance median is 0.6547 (§2, §3).
4. Top 10 frames by L2(T5C−P8A) are all is_real=0 (fakes) (§6).
5. On L2(T5C−P8A) by real-cohort IQ quartile: lap_var Q1 (blurriest) = 22.95, lap_var Q4 (sharpest) = 19.82 (§5).
6. Per-chronic-identity L2(T5C−P8A) on real frames ranges 16.69 (bla_bla_chow, n=51) to 22.68 (Md_noyn_Sharker, n=77) (§4).
7. T5C_step3500 cosine distance to P8A on chronic_6 real (0.5053) and healthy real (0.5063) differs by less than 0.001 (§3).
