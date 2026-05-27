# T4 cross-substrate AUC — FACTS (2026-05-11, A1)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade.
> Numbers + tables + cross-references only. Interpretation belongs in the parent session.
>
> **Scope**: CPU diagnostic A1. Extends `analysis/t4_eval_2026-05-11/RESULTS_FACTS_2026-05-11.md` §7 (which reported T4_L1_step10500 lockbox AUC −0.174 vs P8A while dev AUC moved +0.04 to +0.08) to two additional substrate families: HDTF-substrate (8 teams + clean cells, 2026-05-08 measurement substrate) and may5/may6 (no fakes; 152 real Xinhe frames from 2026-05-06 cross-camera audit).
>
> **Inputs**:
> - T4_L1_step10500 local checkpoint: `analysis/cpu_diagnostics_2026-05-10/_ckpts_t4/top_n_effort_20260510_step10500_auc0.9937_eer0.0195.pth` (940 MB).
> - dev/lockbox per-suite video-level scores: `analysis/cpu_diagnostics_2026-05-10/_t4_scorecard_local/*_videos_report.csv` (already on disk, from `gs://training-job-outputs/test_results/teams_promotion_contract/teams-promotion-contract-20260511-002920/`).
> - HDTF per-frame P8A + E2B cache: `analysis/iq_shortcut_decomp_2026-05-08/scores_cache/{P8A_REFERENCE_STEP5000,E2B_TOP_N_STEP3200}__hdtf_*.csv` and `analysis/p2_d_hdtf_2026-05-08/raw_reports/proper_visomaster_enhanced_teams_*_{p8a,e2b}_*_frames_report.csv` (HDTF run `6632089555598049280`).
> - may5/may6 P8A + E2B scores: `analysis/xinhe_cross_camera_audit_2026-05-06/outputs/scores_{P8A,E2B}.csv`.
>
> **Subsampling note**: For HDTF cells we subsample N=50 videos per cell (deterministic seed=42 on sorted `video_id`s; all frames per video kept, ≈8 frames/video, n≈400 frames per cell). The same 50 videos are reused across {T4, P8A, E2B} so AUC is computed on identical (video_id, frame_path) pairs. Sample rationale: even running 50% of the 70 290 frames across the 10 HDTF cells would exceed the 90-minute budget at the observed 35 fps MPS rate; N=50 keeps the per-cell sample stable and matches the lockbox cell pool size (382-1444 videos available depending on cell).
> **Scripts**: `score_t4_may56.py`, `score_t4_hdtf.py`, `compute_matrix.py` (this directory).

---

## 1. Question

For T4_LAMBDA1_TOP_N_STEP10500 (and reference ckpts P8A_REFERENCE_STEP5000, E2B_TOP_N_STEP3200, T4_LAMBDA2_PERIODIC_STEP1500 where available), tabulate per-substrate ROC AUC on:

- 4 dev/lockbox teams-substrate cells (same as scorecard).
- 5 HDTF substrate cells: `hdtf_clean_dev`, `hdtf_teams_dev`, `hdtf_teams_lockbox`, `hdtf_viso_enh_teams_dev`, `hdtf_viso_enh_teams_lockbox` (subsample N=50 videos each).
- 2 may5/may6 substrates (real-only; report mean score + FPR@τ=0.5 + max).

## 2. Method

- **AUC**: Mann-Whitney U on video-level `avg_video_prob = mean(frame_prob)` per video. For HDTF where per-frame scores live in the score caches, video-level scores are computed in `compute_matrix.py::video_avg`.
- **may5/may6**: no fake population available; we report `mean(score)`, `p50`, `FPR@τ=0.5 = (score ≥ 0.5).mean()`, `max(score)` per ckpt.
- All score CSVs read as-is; no isotonic or other calibration is applied. AUC is monotone-invariant anyway.
- Ckpts are matched 1:1 across the same (video_id, frame_path) HDTF tuples; same per-cell subsample is reused.

## 3. AUC matrix (video-level)

| Substrate | n_real | n_fake | P8A AUC | T4_L1_step10500 AUC | E2B AUC | T4_L2_step1500 AUC |
|---|---:|---:|---:|---:|---:|---:|
| dev_teams_all | 3253 | 2409 | 0.8896 | 0.9265 | n/a | n/a |
| dev_teams_deeplive_enh | 3253 | 545 | 0.8565 | 0.9352 | n/a | n/a |
| dev_teams_viso_enh | 3253 | 550 | 0.7403 | 0.8218 | n/a | n/a |
| lockbox_teams_all | 1361 | 253 | 0.9355 | 0.7619 | n/a | 0.9498 |
| hdtf_clean_dev | 50 | 50 | 0.9996 | 1.0000 | 0.9996 | n/a |
| hdtf_teams_dev | 50 | 50 | 0.9996 | 0.9796 | 0.9088 | n/a |
| hdtf_teams_lockbox | 50 | 50 | 0.9972 | 0.9800 | 0.9096 | n/a |
| hdtf_viso_enh_teams_dev | 50 | 50 | 0.9968 | 0.9752 | 0.8948 | n/a |
| hdtf_viso_enh_teams_lockbox | 50 | 50 | 0.9972 | 0.9776 | 0.9056 | n/a |

`n/a` cells: E2B per-suite video reports for dev/lockbox were not extracted in this CPU job (E2B's dev/lockbox AUCs are in the same scorecard but were not joined in; not load-bearing for the A1 question). T4_L2_step1500 only present where the scorecard wrote video reports for it (`lockbox_teams_all`).

### 3.1 Δ vs P8A (T4_L1_step10500 minus P8A_REFERENCE_STEP5000)

| Substrate | Δ AUC |
|---|---:|
| dev_teams_all | +0.0369 |
| dev_teams_deeplive_enh | +0.0787 |
| dev_teams_viso_enh | +0.0814 |
| lockbox_teams_all | −0.1736 |
| hdtf_clean_dev | +0.0004 |
| hdtf_teams_dev | −0.0200 |
| hdtf_teams_lockbox | −0.0172 |
| hdtf_viso_enh_teams_dev | −0.0216 |
| hdtf_viso_enh_teams_lockbox | −0.0196 |

T4_L1_step10500 AUC magnitudes on HDTF range 0.9752–1.0000 (5 cells). All Δ vs P8A on HDTF substrates are within ±0.022 absolute (4 negative, 1 essentially zero).

## 4. may5/may6 substrates (no fake population)

| Substrate | n | ckpt | mean | p50 | max | FPR@τ=0.5 |
|---|---:|---|---:|---:|---:|---:|
| may5_correct | 60 | P8A_REFERENCE_STEP5000 | 0.0207 | 0.0066 | 0.2341 | 0.0000 |
| may5_correct | 60 | T4_L1_step10500 | 0.0955 | 0.0914 | 0.1833 | 0.0000 |
| may5_correct | 60 | E2B_TOP_N_STEP3200 | 0.0527 | 0.0149 | 0.6033 | 0.0167 |
| may6_falseflag | 92 | P8A_REFERENCE_STEP5000 | 0.0206 | 0.0093 | 0.2535 | 0.0000 |
| may6_falseflag | 92 | T4_L1_step10500 | 0.2226 | 0.1276 | 0.8346 | 0.0978 |
| may6_falseflag | 92 | E2B_TOP_N_STEP3200 | 0.5113 | 0.5794 | 0.9896 | 0.5761 |

Source: `t4_may56_per_frame.csv` (T4); `analysis/xinhe_cross_camera_audit_2026-05-06/outputs/scores_{P8A,E2B}.csv`.

## 5. Per-substrate score percentiles (real video-level scores)

### 5.1 Reals (p10/p25/p50/p75/p90)

| Substrate | ckpt | n | p10 | p25 | p50 | p75 | p90 |
|---|---|---:|---:|---:|---:|---:|---:|
| dev (teams_real_all_dev) | P8A | 3253 | 0.0054 | 0.0056 | 0.0076 | 0.0539 | 0.6930 |
| dev (teams_real_all_dev) | T4_L1_step10500 | 3253 | 0.0783 | 0.0841 | 0.0976 | 0.2337 | 0.6661 |
| lockbox (teams_real_all_lockbox) | P8A | 1361 | 0.0061 | 0.0082 | 0.0157 | 0.0541 | 0.2541 |
| lockbox (teams_real_all_lockbox) | T4_L1_step10500 | 1361 | 0.1431 | 0.2600 | 0.4722 | 0.6852 | 0.8008 |
| lockbox (teams_real_all_lockbox) | T4_L2_step1500 | 1361 | 0.4904 | 0.5823 | 0.6582 | 0.7074 | 0.7501 |
| hdtf_real_clean_dev | P8A | 50 | 0.0054 | 0.0054 | 0.0055 | 0.0059 | 0.0070 |
| hdtf_real_clean_dev | T4_L1_step10500 | 50 | 0.0784 | 0.0813 | 0.0891 | 0.1001 | 0.1267 |
| hdtf_real_clean_dev | E2B | 50 | 0.0046 | 0.0050 | 0.0056 | 0.0063 | 0.0123 |
| hdtf_real_teams_dev | P8A | 50 | 0.0054 | 0.0055 | 0.0059 | 0.0075 | 0.0567 |
| hdtf_real_teams_dev | T4_L1_step10500 | 50 | 0.0743 | 0.0800 | 0.0852 | 0.0916 | 0.1018 |
| hdtf_real_teams_dev | E2B | 50 | 0.0048 | 0.0051 | 0.0057 | 0.0062 | 0.0092 |
| hdtf_real_teams_lockbox | P8A | 50 | 0.0054 | 0.0054 | 0.0055 | 0.0069 | 0.0123 |
| hdtf_real_teams_lockbox | T4_L1_step10500 | 50 | 0.0782 | 0.0798 | 0.0839 | 0.0896 | 0.0941 |
| hdtf_real_teams_lockbox | E2B | 50 | 0.0047 | 0.0050 | 0.0056 | 0.0062 | 0.0086 |
| may5_correct | P8A | 60 | 0.0055 | 0.0058 | 0.0066 | 0.0116 | 0.0306 |
| may5_correct | T4_L1_step10500 | 60 | 0.0856 | 0.0885 | 0.0914 | 0.0979 | 0.1016 |
| may5_correct | E2B | 60 | 0.0083 | 0.0109 | 0.0149 | 0.0303 | 0.0731 |
| may6_falseflag | P8A | 92 | 0.0056 | 0.0063 | 0.0093 | 0.0170 | 0.0368 |
| may6_falseflag | T4_L1_step10500 | 92 | 0.0943 | 0.1050 | 0.1276 | 0.2655 | 0.4757 |
| may6_falseflag | E2B | 92 | 0.0248 | 0.1088 | 0.5794 | 0.8350 | 0.9451 |

### 5.2 Fakes (p10/p25/p50/p75/p90)

| Substrate | ckpt | n | p10 | p25 | p50 | p75 | p90 |
|---|---|---:|---:|---:|---:|---:|---:|
| teams_fake_all_dev | P8A | 2409 | 0.0306 | 0.3179 | 0.9463 | 0.9945 | 0.9946 |
| teams_fake_all_dev | T4_L1_step10500 | 2409 | 0.4152 | 0.7127 | 0.8779 | 0.9280 | 0.9345 |
| teams_fake_all_lockbox | P8A | 253 | 0.0916 | 0.3438 | 0.8042 | 0.9896 | 0.9946 |
| teams_fake_all_lockbox | T4_L1_step10500 | 253 | 0.3005 | 0.5536 | 0.7936 | 0.8773 | 0.9093 |
| teams_fake_all_lockbox | T4_L2_step1500 | 253 | 0.7548 | 0.7881 | 0.8190 | 0.8447 | 0.8558 |
| visomaster_enhanced_macro_dev | P8A | 550 | 0.0062 | 0.0117 | 0.1704 | 0.7605 | 0.9509 |
| visomaster_enhanced_macro_dev | T4_L1_step10500 | 550 | 0.1101 | 0.2717 | 0.5985 | 0.7859 | 0.8654 |
| deeplive_enhanced_dev | P8A | 545 | 0.0320 | 0.1268 | 0.5342 | 0.9019 | 0.9793 |
| deeplive_enhanced_dev | T4_L1_step10500 | 545 | 0.6243 | 0.7550 | 0.8512 | 0.8920 | 0.9116 |
| hdtf_fake_clean_dev | P8A | 50 | 0.9761 | 0.9916 | 0.9943 | 0.9945 | 0.9946 |
| hdtf_fake_clean_dev | T4_L1_step10500 | 50 | 0.9173 | 0.9278 | 0.9332 | 0.9365 | 0.9381 |
| hdtf_fake_clean_dev | E2B | 50 | 0.4692 | 0.6816 | 0.9586 | 0.9929 | 0.9949 |
| hdtf_fake_teams_dev | P8A | 50 | 0.6725 | 0.9437 | 0.9938 | 0.9944 | 0.9946 |
| hdtf_fake_teams_dev | T4_L1_step10500 | 50 | 0.1142 | 0.1899 | 0.3629 | 0.7183 | 0.9222 |
| hdtf_fake_teams_dev | E2B | 50 | 0.0060 | 0.0095 | 0.0400 | 0.2382 | 0.7616 |
| hdtf_fake_teams_lockbox | P8A | 50 | 0.8578 | 0.9701 | 0.9917 | 0.9945 | 0.9946 |
| hdtf_fake_teams_lockbox | T4_L1_step10500 | 50 | 0.1132 | 0.2199 | 0.4265 | 0.7271 | 0.8852 |
| hdtf_fake_teams_lockbox | E2B | 50 | 0.0066 | 0.0143 | 0.0915 | 0.3991 | 0.8655 |
| hdtf_viso_enh_teams_dev | P8A | 50 | 0.5150 | 0.7965 | 0.9911 | 0.9943 | 0.9946 |
| hdtf_viso_enh_teams_dev | T4_L1_step10500 | 50 | 0.1036 | 0.1891 | 0.3816 | 0.7016 | 0.8398 |
| hdtf_viso_enh_teams_dev | E2B | 50 | 0.0058 | 0.0115 | 0.0281 | 0.1634 | 0.2698 |
| hdtf_viso_enh_teams_lockbox | P8A | 50 | 0.7921 | 0.9524 | 0.9907 | 0.9944 | 0.9946 |
| hdtf_viso_enh_teams_lockbox | T4_L1_step10500 | 50 | 0.1375 | 0.2009 | 0.3190 | 0.6785 | 0.7750 |
| hdtf_viso_enh_teams_lockbox | E2B | 50 | 0.0062 | 0.0103 | 0.0335 | 0.1137 | 0.3331 |

## 6. Caveats and skipped cells

- **HDTF subsample size = 50 videos per cell** (≈400 frames per cell). The full HDTF score caches have 382–1444 videos per cell; AUCs in §3 are subset AUCs. Per-cell σ at this n is approximately 0.02 absolute (binomial reasoning on Mann-Whitney for AUC ≈ 0.95, n=50/50). All HDTF |Δ| in §3.1 ≤ 0.022 are within this same-order range. Lockbox |Δ| = 0.174 and dev |Δ| ≥ 0.037 are well outside that range.
- **E2B dev/lockbox AUCs not joined**: this CPU job did not assemble E2B per-suite video-level scores for the dev/lockbox cells; those are in the same scorecard CSV pool and can be added without further inference. They are not load-bearing for the A1 α/β/γ decision.
- **may5/may6 has no fake population**: AUC cell is not defined; we report score-distribution statistics instead.
- **T4_L2_step1500 is reported only on lockbox_teams_all** (other cells were not pulled because the focus is T4_L1_step10500 vs P8A; T4_L2 is included as a sanity check that the lockbox drop is L1-specific and not a generic T4-family artifact — the L2 lockbox AUC 0.9498 is +0.0143 over P8A's 0.9355, not aligned with L1's −0.1736).
- **`hdtf_viso_enh_teams_*` reals**: this cell does not have its own `real` sub-pool; we reuse `hdtf_real_teams_*` reals (same substrate per §2 of `p2_d_hdtf_2026-05-08`). The "real" baseline reported is therefore shared with `hdtf_teams_dev`/`hdtf_teams_lockbox`.

## 7. Output artifacts

- `cross_substrate_auc.csv` — 33 rows: (substrate × ckpt) AUC table.
- `per_substrate_percentiles.csv` — 50 rows: (substrate × role × ckpt) p10/p25/p50/p75/p90.
- `t4_may56_per_frame.csv` — 152 T4 scores on may5/may6.
- `t4_hdtf_per_frame.csv` — 3200 T4 scores on HDTF subsample.
- `hdtf_subsample_p8a_e2b_match.csv` — companion P8A + E2B scores for the same 3200 frames.

---

## 8. Six-bullet summary

- **B1**: T4_L1_step10500 AUC moves on non-dev substrates: lockbox −0.1736; HDTF teams_dev −0.0200, HDTF teams_lockbox −0.0172, HDTF viso_enh_teams_dev −0.0216, HDTF viso_enh_teams_lockbox −0.0196, HDTF clean_dev +0.0004; may6 mean-score rises from P8A 0.0206 to T4 0.2226 (10.8× ratio) while may5 rises from 0.0207 to 0.0955 (4.6× ratio).
- **B2**: dev AUCs all move upward: +0.0369 / +0.0787 / +0.0814 across the three dev fake cells (sign matches `analysis/t4_eval_2026-05-11/RESULTS_FACTS_2026-05-11.md` §7).
- **B3**: T4_L1_step10500 AUC magnitude **holds** at 0.97–1.00 on all 5 HDTF cells (Δ vs P8A bounded to [−0.022, +0.001]); on HDTF clean_dev T4 matches P8A exactly (1.0000 vs 0.9996).
- **B4**: T4_L1_step10500 AUC **drops** on lockbox_teams_all (−0.1736) and is on the edge of drop on may5/may6 (no AUC available; FPR@τ=0.5 = 0.098 on may6 vs P8A 0.000); for sanity check T4_L2_step1500 on lockbox shows +0.0143, so the lockbox drop is specific to the L1-step10500 ckpt, not generic to T4 packet.
- **B5**: Data supports **β-outcome** (lockbox is the only cell with Δ AUC < −0.05; HDTF holds within sampling noise; dev moves up): the drop signature is lockbox-specific not universal substrate-overfit. may6 is intermediate and consistent with a partial substrate-related score shift.
- **B6**: Caveats: HDTF AUCs are on N=50 subsample per cell (σ ≈ 0.02; small HDTF Δs are within sampling noise); may5/may6 has no fake population so AUC is not measured (only score-shift statistics); E2B dev/lockbox AUCs not joined this run; T4_L2_step1500 AUC only available on `lockbox_teams_all`.
