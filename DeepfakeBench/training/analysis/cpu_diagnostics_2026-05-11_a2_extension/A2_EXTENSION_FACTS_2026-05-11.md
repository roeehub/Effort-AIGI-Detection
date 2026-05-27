# A2 EXTENSION — Linear probe across T4 step ckpts, with `real_dor` added (FACTS, 2026-05-11)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade.
> Extends A2 (`analysis/cpu_diagnostics_2026-05-11_a2_linear_probe/LOCKBOX_PROBE_FACTS_2026-05-11.md`)
> with: (Q1) the 109 `real_dor` lockbox PNG videos now downloaded and included; (Q2) three additional
> T4 step ckpts beyond the original step10500 (periodic step5000, top_n step9000 / step11250).
>
> **Inputs**:
> - T4 ckpts: `analysis/cpu_diagnostics_2026-05-10/_ckpts_t4/{periodic_effort_..._step5000_..., top_n_effort_..._step9000_..., top_n_effort_..._step10500_..., top_n_effort_..._step11250_...}.pth`
> - P8A reference: `analysis/_features_cache_2026-04-30/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth`
> - Lockbox CSVs: `analysis/cpu_diagnostics_2026-05-10/_t4_scorecard_local/teams_{real,fake}_all_lockbox_t4_lambda1_top_n_step10500_frames_report.csv`
> - `real_dor` PNGs (109 lockbox videos): downloaded via `gsutil -m cp` from `gs://teams-faces-data-test-2914-fake-4420-real-feb-28/real/real_dor__*.png` → `_real_dor_png/` (offline cache; 10 MB).
> - Scripts: `run_a2_extension.py`, `run_trained_head_with_dor.py`.
> - Outputs: `lockbox_probe_with_dor_auc.csv` (Q1), `lockbox_probe_per_t4_step.csv` (Q2), `lockbox_probe_summary.csv`, `lockbox_probe_non_dor_only.csv`, `trained_head_with_dor_auc.csv`, `_cache/video_feats_with_dor__*__L11.npz`.

---

## 1. Method

Identical to A2 (§1) except:

- **Sample composition**: `real_dor` source (109 lockbox videos, `.png`) is now included in the real pool. The non-real_dor reals are still proportional-source-stratified to 200 videos (seed=42), so the real pool is `200 + 109 = 309` reals. Fakes unchanged at 253. Total = `562 videos / 752 frames` after `≤4 frames/video` cap.
- **Local mirror mapping**: `gs_to_local()` adds a `real_dor__*.png` → `_real_dor_png/` branch in addition to A2's `.jpg` mirror at `/Users/roeedar/Downloads/faces/r9_feb28_for_checker/{real,fake}/flat/`.
- **Ckpts**: 5 total (4 T4 step ckpts + P8A baseline).
- Linear probe: identical 5-fold StratifiedKFold, `LogisticRegression(C=1.0, max_iter=2000, n_jobs=1)`, `StandardScaler` fit on train fold only, ROC-AUC per fold.

### 1.1 Source composition of probe inputs

| Class | Source identity prefix | n_videos |
|---|---|---:|
| real (label=0) | dor_shkedi | 140 |
| real | bla_bla_chow | 32 |
| real | PC_Generator | 15 |
| real | Chikara_Takahashi | 13 |
| real | **real_dor** (new vs A2) | **109** |
| fake (label=1) | Cam_Test | 191 |
| fake | PC_Generator | 62 |

Total: 309 reals + 253 fakes = **562 videos** (vs A2's 453).

---

## 2. Q1 — Linear probe with `real_dor` included

Source: `lockbox_probe_with_dor_auc.csv` (subset of `lockbox_probe_per_t4_step.csv`).

| ckpt | fold | AUC | n_real | n_fake |
|---|---:|---:|---:|---:|
| T4_LAMBDA1_TOP_N_STEP10500 | 0 | 1.0000 | 62 | 51 |
| T4_LAMBDA1_TOP_N_STEP10500 | 1 | 1.0000 | 62 | 51 |
| T4_LAMBDA1_TOP_N_STEP10500 | 2 | 1.0000 | 62 | 50 |
| T4_LAMBDA1_TOP_N_STEP10500 | 3 | 1.0000 | 62 | 50 |
| T4_LAMBDA1_TOP_N_STEP10500 | 4 | 1.0000 | 61 | 51 |
| P8A_REFERENCE_STEP5000 | 0 | 1.0000 | 62 | 51 |
| P8A_REFERENCE_STEP5000 | 1 | 1.0000 | 62 | 51 |
| P8A_REFERENCE_STEP5000 | 2 | 1.0000 | 62 | 50 |
| P8A_REFERENCE_STEP5000 | 3 | 1.0000 | 62 | 50 |
| P8A_REFERENCE_STEP5000 | 4 | 1.0000 | 61 | 51 |

| ckpt | mean_AUC | std_AUC | Δ vs A2 (no-dor, 453 videos) |
|---|---:|---:|---:|
| T4_LAMBDA1_TOP_N_STEP10500 | 1.0000 | 0.0000 | +0.0000 |
| P8A_REFERENCE_STEP5000 | 1.0000 | 0.0000 | +0.0000 |

A2 reference: T4 1.0000 ± 0.0000, P8A 1.0000 ± 0.0000 on 200+253 = 453 videos (no real_dor).

---

## 3. Q2 — Per-T4-step linear probe (same with-dor lockbox subset)

Source: `lockbox_probe_per_t4_step.csv`, summary `lockbox_probe_summary.csv`.

| ckpt | mean_AUC | std_AUC | min_AUC | max_AUC | n_folds | n_real | n_fake |
|---|---:|---:|---:|---:|---:|---:|---:|
| T4_LAMBDA1_PERIODIC_STEP5000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 5 | 309 | 253 |
| T4_LAMBDA1_TOP_N_STEP9000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 5 | 309 | 253 |
| T4_LAMBDA1_TOP_N_STEP10500 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 5 | 309 | 253 |
| T4_LAMBDA1_TOP_N_STEP11250 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 5 | 309 | 253 |
| P8A_REFERENCE_STEP5000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 5 | 309 | 253 |

All 25 (5 ckpts × 5 folds) per-fold AUCs equal 1.0000 exactly.

---

## 4. Non-`real_dor`-only sensitivity check

Source: `lockbox_probe_non_dor_only.csv`. Same 5 ckpts, same fakes, real pool restricted to non-real_dor (n=200 reals, 40/40/40/40/40 per fold).

| ckpt | mean_AUC | std_AUC | min_AUC | max_AUC |
|---|---:|---:|---:|---:|
| T4_LAMBDA1_PERIODIC_STEP5000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| T4_LAMBDA1_TOP_N_STEP9000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| T4_LAMBDA1_TOP_N_STEP10500 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| T4_LAMBDA1_TOP_N_STEP11250 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| P8A_REFERENCE_STEP5000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

Linear probe AUC = 1.0000 in both `with_dor` (309 reals) and `non_dor_only` (200 reals) regimes for all 5 ckpts.

---

## 5. Trained-head context (Q1 comparison)

Source: `trained_head_with_dor_auc.csv`. Video-level aggregate = mean of `frame_prob` per `video_id`, ROC-AUC via `sklearn.metrics.roc_auc_score`. Not all rows below correspond to the linear-probe subset; reading both helps localize the trained-head AUC drop within the lockbox.

| ckpt | full lockbox AUC (n=1361r+253f) | probe-subset-with-dor AUC (n=309r+253f) | `real_dor`-only-vs-fakes (n=109r+253f) | `non-dor`-only-vs-fakes (n=200r+253f) |
|---|---:|---:|---:|---:|
| T4_LAMBDA1_TOP_N_STEP10500 | 0.7735 | 0.8622 | **0.9842** | 0.7957 |
| P8A_REFERENCE_STEP5000 | 0.9417 | 0.9083 | **0.9901** | 0.8638 |
| Δ (P8A − T4) | +0.1682 | +0.0461 | +0.0059 | +0.0681 |

`Δ` interpretation (FACTS-level): the trained-head AUC gap (P8A − T4) on the with-dor 562-video subset is `+0.0461`, smaller than the full-lockbox `+0.1682`. The `real_dor`-only-vs-fakes trained-head AUC gap is `+0.0059` (T4 0.9842, P8A 0.9901). The `non-dor`-only-vs-fakes gap is `+0.0681` (identical to A2 §3's "453-video probe subset" gap because the non-dor 200 reals are the same A2 sample). A2's claim ("real_dor carries a disproportionate share of T4's lockbox failure") was based on the 0.174 → 0.068 absolute trained-head gap shrink when real_dor was removed; the corresponding measurement here is `+0.1682` (full) → `+0.0681` (non-dor only after dropping all 1138+ dor_shkedi videos), and the trained-head AUC on `real_dor`-only is `0.9842` for T4 and `0.9901` for P8A.

---

## 6. Feature statistics — Q2 ckpt diagnostics

Source: `_cache/video_feats_with_dor__*__L11.npz`. Each ckpt produces (562, 768) video-level features.

| ckpt | n_videos | dim | shape |
|---|---:|---:|---|
| T4_LAMBDA1_PERIODIC_STEP5000 | 562 | 768 | (562, 768) |
| T4_LAMBDA1_TOP_N_STEP9000 | 562 | 768 | (562, 768) |
| T4_LAMBDA1_TOP_N_STEP10500 | 562 | 768 | (562, 768) |
| T4_LAMBDA1_TOP_N_STEP11250 | 562 | 768 | (562, 768) |
| P8A_REFERENCE_STEP5000 | 562 | 768 | (562, 768) |

---

## 7. Cross-references

- A2 primary: `analysis/cpu_diagnostics_2026-05-11_a2_linear_probe/LOCKBOX_PROBE_FACTS_2026-05-11.md` §2 (lockbox probe AUC=1.0000 on 453 videos), §3 (trained-head gap shrink 0.174 → 0.068).
- T4 packet trained-head scorecard: `analysis/t4_eval_2026-05-11/RESULTS_FACTS_2026-05-11.md` §7.1 (T4 lockbox AUC 0.7619 vs P8A 0.9355).
- Per-layer P8A↔E2B divergence: memory `project_per_layer_divergence_2026-05-06.md` (cos→0.32 by L11; behavioral difference concentrated at layers 10-11 + head).
- A1 HDTF generalization: `analysis/.../A1_*` (T4 generalizes to HDTF).
- A3 inv_mean regression on chronic_6: `analysis/.../A3_*` (T4 chronic_6 inv_mean −0.0315 vs P8A).
- T4 substrate-overfit OPINION: memory `project_t4_substrate_overfit_inv_mean_misleading_2026-05-11.md`.

---

## 8. Self-contained summary (5 bullets)

- **Q1** with-dor 5-fold probe AUC: T4_LAMBDA1_TOP_N_STEP10500 = 1.0000 ± 0.0000, P8A_REFERENCE_STEP5000 = 1.0000 ± 0.0000 (n_real=309, n_fake=253, 50 folds across 10 ckpt×fold cells all = 1.0000). A2 no-dor numbers were T4 1.0000 ± 0.0000 / P8A 1.0000 ± 0.0000 on n=453; Δ = 0.0000 in both arms.
- **Q2** per-step T4 probe AUC table (with-dor subset, n_real=309, n_fake=253, 5-fold StratifiedKFold): T4_PERIODIC_STEP5000 = 1.0000; T4_TOP_N_STEP9000 = 1.0000; T4_TOP_N_STEP10500 = 1.0000; T4_TOP_N_STEP11250 = 1.0000; P8A_REF_STEP5000 = 1.0000. All 25 per-fold AUCs equal 1.0000 exactly.
- **Verdict on Q2 (FACTS-only)**: encoder L11 separability AUC=1.0000 is observed at all 4 measured T4 step ckpts (steps 5000-periodic, 9000, 10500, 11250) and at P8A_REF_STEP5000; not step-specific to step10500 within the measured set.
- **Verdict on Q1 (FACTS-only)**: real_dor inclusion does not move the L11 linear-probe AUC from 1.0000 for any of the 5 ckpts (T4 or P8A). Trained-head AUC on real_dor-only-vs-fakes is 0.9842 (T4) / 0.9901 (P8A), i.e. above both ckpts' full-lockbox AUC; the trained-head gap (P8A − T4) on real_dor-only-vs-fakes is +0.0059 vs +0.1682 on full lockbox.
- **Caveats**: (i) the linear probe is overdetermined (dim=768 >> n≈450 train/fold) — A2's Job D regularization sweep is the relevant memorization sanity check, not repeated here for time; (ii) only 4 T4 step ckpts available in `_ckpts_t4/` (5000-periodic + 9000/10500/11250-top_n); other T4 steps (e.g. earlier or λ2 cyclic steps) not tested; (iii) `real_dor`-only-vs-fakes trained-head AUC (0.984/0.990) reflects only 109 reals; cross-substrate, HDTF, and chronic_6 slices are not retested here.

---
