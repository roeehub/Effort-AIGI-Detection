# Slot 1 (LoRA-P8A) shift profile + τ-recalibration sweep — FACTS (2026-05-13)

> **Status: factual-only.** Numbers + tables + cross-references. No interpretation, no verdict language.
> Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, unfortunately, remarkably.

## 0. Scope and provenance

- **Question**: characterize how Slot 1 LoRA-P8A score distributions differ from P8A_REFERENCE_STEP5000 across the 9-cell contract suite; check whether τ recalibration can match Slot 1 to P8A's operating point.
- **Trigger**: may6 retest (today) reported Spearman r=0.97 + mean delta ≈ +0.21 between Slot 1 and P8A; partial scorecard reported Slot 1 `lb_real_fpr` ≈ 0.39 (21× P8A's 0.0184) at dev-calibrated τ.
- **Inputs**:
  - Frame-level CSVs at `_frame_cache/<suite>_<ckpt>_frames_report.csv`, downloaded from `gs://training-job-outputs/test_results/teams_promotion_contract/r13-overnight-scorecard-2026-05-13/reports/`. 36 files (9 suites × 4 ckpts: P8A + Slot 1 ×3).
  - Video-level CSVs at `../r13_overnight_partial_scorecard_2026-05-13/_reports_cache/<suite>_<ckpt>_videos_report.csv` (canonical scorer-input level).
  - τ values from `_partial_scorecard_2026-05-13.csv` (calibrated on `teams_real_all_dev` at `real_fpr ≤ 0.07`; midpoint convention).
- **Caveat**: my τ-recalibration uses `teams_real_all_dev` only (v3-fix single-constraint), matching the partial scorecard's methodology. Stress-binding τ-bump is not applied.
- **Code**: `run_shift_analysis.py` in this folder. Re-runnable.

## 1. Job 1 — Per-suite mean prob_fake (P8A vs all 3 Slot 1 ckpts)

Frame-level mean of `frame_prob` over all frames per (suite × ckpt). Includes both labels per suite (one of {real, fake} only per suite type).

| Suite | n | P8A mean | S1_step1500 | S1_step3500 | S1_top_n_2000 | Δ(2000−P8A) |
|---|---:|---:|---:|---:|---:|---:|
| teams_real_all_dev (real) | 4564 | 0.1345 | 0.4285 | 0.4089 | 0.4199 | **+0.2854** |
| teams_real_poor_quality_dev (real) | 1303 | 0.0987 | 0.4250 | 0.4039 | 0.4159 | **+0.3172** |
| teams_real_lighting_extreme_dev (real) | 1742 | 0.1264 | 0.4280 | 0.4084 | 0.4194 | **+0.2930** |
| teams_real_all_lockbox (real) | 1418 | 0.0963 | 0.4486 | 0.4372 | 0.4432 | **+0.3468** |
| teams_real_dor_dev (real, n=50) | 50 | 0.3310 | 0.4518 | 0.4415 | 0.4468 | **+0.1158** |
| teams_fake_all_dev (fake) | 3039 | 0.7447 | 0.4563 | 0.4481 | 0.4521 | **−0.2926** |
| visomaster_enhanced_macro_dev (fake) | 550 | 0.3563 | 0.4607 | 0.4542 | 0.4571 | **+0.1008** |
| deeplive_enhanced_dev (fake) | 545 | 0.5224 | 0.4447 | 0.4322 | 0.4388 | **−0.0836** |
| teams_fake_all_lockbox (fake) | 425 | 0.6480 | 0.4470 | 0.4350 | 0.4413 | **−0.2066** |

Source: `_shift_profile_2026-05-13.csv`.

### 1.1 Shift is not constant across substrates

- Real-suite Δ(top_n_2000 − P8A): from **+0.1158** (dor_dev, n=50) to **+0.3468** (real_all_lockbox). Spread 0.231 absolute.
- Fake-suite Δ(top_n_2000 − P8A): from **−0.2926** (fake_all_dev) to **+0.1008** (visomaster). Direction reverses across fake suites.
- The shift mean across the 5 real suites: +0.27 ± 0.10 (1σ across suites).
- The shift mean across the 4 fake suites: −0.13 ± 0.17 (1σ across suites).

### 1.2 Distribution-range collapse (not visible from mean alone)

Per-suite video-level `avg_video_prob` range (min..max):

| Suite | Label | P8A range | Slot 1 top_n_step2000 range |
|---|---|---|---|
| teams_real_all_dev | real | 0.005..0.995 | 0.384..0.484 |
| teams_real_all_lockbox | real | 0.005..0.994 | 0.383..0.475 |
| teams_fake_all_lockbox | fake | 0.013..0.995 | 0.415..0.465 |
| teams_fake_all_dev | fake | 0.005..0.995 | 0.387..0.508 |

P8A span ≈ 0.99 absolute; Slot 1 top_n_step2000 span ≈ 0.07-0.12 (≈ 13× compressed).

### 1.3 Per-frame Spearman rank correlation, P8A vs Slot 1 top_n_step2000

Computed on aligned `frame_path` keys.

| Suite | Label | n | P8A median | S1 median | Δmedian | Spearman r |
|---|---|---:|---:|---:|---:|---:|
| teams_real_all_dev | real | 4564 | 0.0070 | 0.4171 | +0.4101 | 0.4726 |
| teams_real_all_lockbox | real | 1418 | 0.0160 | 0.4460 | +0.4300 | 0.2020 |
| teams_real_dor_dev | real | 50 | 0.1818 | 0.4457 | +0.2638 | 0.4283 |
| teams_real_poor_quality_dev | real | 1303 | 0.0067 | 0.4131 | +0.4064 | 0.2829 |
| teams_real_lighting_extreme_dev | real | 1742 | 0.0085 | 0.4172 | +0.4086 | 0.5199 |
| teams_fake_all_dev | fake | 3039 | 0.9852 | 0.4485 | −0.5368 | 0.6191 |
| teams_fake_all_lockbox | fake | 425 | 0.7882 | 0.4400 | −0.3482 | 0.8554 |
| visomaster_enhanced_macro_dev | fake | 550 | 0.1705 | 0.4579 | +0.2874 | 0.8086 |
| deeplive_enhanced_dev | fake | 545 | 0.5342 | 0.4390 | −0.0952 | 0.4066 |

Spearman ranges 0.20–0.86 across the 9 suites. The high may6 figure (~0.97) is not reproduced on any single contract suite at the per-frame level.

### 1.4 Strongest Slot 1 step

By delta magnitude (|Δ| across reals) and by lb_real_fpr in §3 below: `slot1_lora_p8a_periodic_step1500` has the largest mean-shift on `teams_real_all_lockbox` (+0.3523), but all 3 Slot 1 ckpts behave the same direction and approximate magnitude. `top_n_step2000` sits between step1500 and step3500. For the τ-sweep below I use all 3 ckpts; `top_n_step2000` is the canonical "candidate" per the task.

## 2. Job 2 — τ-recalibration sweep on Slot 1 top_n_step2000

Sweep on video-level `avg_video_prob` (matches scorer methodology). τ grid is densified around 0.42-0.52 where Slot 1 score density is concentrated.

| τ | dev_real_fpr | lb_real_fpr | dor_real_fpr | poor_q_fpr | lighting_fpr | dev_fake_t_recall | dev_fake_v_recall | dev_fake_d_recall | dev_macro_recall | lb_fake_recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.4400 | 0.1660 | 0.6231 | 0.8200 | 0.1018 | 0.0857 | 0.6467 | 0.7618 | 0.4422 | **0.6169** | 0.4941 |
| 0.4450 | 0.1137 | 0.5298 | 0.5600 | 0.0574 | 0.0635 | 0.5301 | 0.7055 | 0.1798 | 0.4718 | 0.3399 |
| **0.4496** (dev-cal) | **0.0704** | **0.3975** | **0.3000** | 0.0336 | 0.0400 | 0.4554 | 0.6382 | 0.0771 | **0.3902** | **0.2609** |
| 0.4500 | 0.0667 | 0.3880 | 0.3000 | 0.0314 | 0.0364 | 0.4467 | 0.6327 | 0.0697 | 0.3830 | 0.2451 |
| 0.4550 | 0.0384 | 0.2425 | 0.1600 | 0.0238 | 0.0228 | 0.3761 | 0.5545 | 0.0147 | 0.3151 | 0.1383 |
| 0.4600 | 0.0191 | 0.1242 | 0.0600 | 0.0141 | 0.0143 | 0.3093 | 0.4782 | 0.0000 | 0.2625 | 0.0593 |
| 0.4650 | 0.0095 | 0.0485 | 0.0000 | 0.0076 | 0.0057 | 0.2561 | 0.3891 | 0.0000 | 0.2151 | 0.0000 |
| 0.4672 (τ_match @0.025) | — | **0.0250** | 0.0000 | 0.0033 | 0.0029 | 0.2367 | 0.3527 | 0.0000 | **0.1965** | **0.0000** |
| 0.4700 | 0.0040 | 0.0096 | 0.0000 | 0.0022 | 0.0036 | 0.2200 | 0.3236 | 0.0000 | 0.1812 | 0.0000 |
| 0.4750 | 0.0015 | 0.0007 | 0.0000 | 0.0011 | 0.0007 | 0.1781 | 0.2582 | 0.0000 | 0.1454 | 0.0000 |
| 0.4800 | 0.0003 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1254 | 0.1764 | 0.0000 | 0.1006 | 0.0000 |
| 0.4900 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0390 | 0.0182 | 0.0000 | 0.0191 | 0.0000 |
| 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0071 | 0.0000 | 0.0000 | 0.0024 | 0.0000 |
| 0.5100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.9143 (P8A τ) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Source: `_tau_recal_2026-05-13.csv`.

### 2.1 τ_match for lb_real_fpr ≤ 0.025 (E2B-comparable Pillar 1)

Computed via midpoint convention on the video-level lockbox-real probs.

| Ckpt | τ_match | lb_real_fpr | lb_fake_recall | dev_macro_recall | dor_real_fpr |
|---|---:|---:|---:|---:|---:|
| slot1_lora_p8a_periodic_step1500 | 0.4695 | 0.0250 | **0.0000** | 0.1954 | 0.0000 |
| slot1_lora_p8a_periodic_step3500 | 0.4662 | 0.0250 | **0.0000** | 0.1961 | 0.0000 |
| slot1_lora_p8a_top_n_step2000 | 0.4672 | 0.0250 | **0.0000** | 0.1954 | 0.0000 |

At τ_match (lb_real_fpr ≤ 0.025):
- `lb_fake_recall = 0.0000` for all 3 Slot 1 ckpts (below 0.30 floor by 0.30 absolute).
- `dev_macro_recall ≈ 0.195` for all 3 Slot 1 ckpts (below 0.30 floor by 0.105 absolute).

Source: `_tau_match_e2b_2026-05-13.csv`.

### 2.2 P8A reference at its dev-cal τ (for direct comparison)

| Metric | P8A @ τ=0.9143 |
|---|---:|
| lb_real_fpr | 0.0184 |
| lb_fake_recall | 0.3874 |
| dev_macro_recall | 0.3003 |
| dor_real_fpr | 0.08 (4/50) |

Source: `../r13_overnight_partial_scorecard_2026-05-13/PARTIAL_RESULTS_FACTS_2026-05-13.md` §3.

### 2.3 Score-distribution span at video level (for context)

| Ckpt | Suite | min | max | n_unique | mean |
|---|---|---:|---:|---:|---:|
| Slot 1 top_n_step2000 | teams_real_all_lockbox | 0.3830 | 0.4754 | 1356/1361 | 0.4432 |
| Slot 1 top_n_step2000 | teams_fake_all_lockbox | 0.4151 | 0.4650 | 253/253 | 0.4412 |
| P8A | teams_real_all_lockbox | 0.005 | 0.994 | — | — |
| P8A | teams_fake_all_lockbox | 0.013 | 0.995 | — | — |

For Slot 1 top_n_step2000 on the lockbox: fake-distribution max (0.4650) is below real-distribution max (0.4754); fake-distribution mean (0.4412) is below real-distribution mean (0.4432).

### 2.4 Per-pair AUC (rank discriminability of fake > real, video-level)

AUC > 0.5 = fakes score higher than reals; AUC = 0.5 = random; AUC < 0.5 = direction inverted.

| Pair | P8A | S1_step1500 | S1_step3500 | S1_top_n_2000 |
|---|---:|---:|---:|---:|
| teams_real_all_dev vs teams_fake_all_dev | **0.8896** | 0.8624 | 0.8664 | 0.8635 |
| teams_real_all_lockbox vs teams_fake_all_lockbox | **0.9355** | 0.4449 | 0.4450 | **0.4442** |
| teams_real_all_dev vs visomaster_enhanced_macro_dev | 0.7403 | 0.8967 | 0.8998 | **0.8974** |
| teams_real_all_dev vs deeplive_enhanced_dev | 0.8565 | 0.8119 | 0.8182 | 0.8136 |

Slot 1 top_n_step2000 vs P8A:
- dev_all AUC drop: 0.8896 → 0.8635 (Δ = −0.026).
- lockbox AUC drop: 0.9355 → 0.4442 (Δ = −0.491; below random).
- visomaster AUC change: 0.7403 → 0.8974 (Δ = +0.157).
- deeplive AUC drop: 0.8565 → 0.8136 (Δ = −0.043).

On the lockbox, Slot 1 ckpts' AUC sits below 0.5 (0.4442 / 0.4449 / 0.4450), i.e. lockbox-fake video probs are on average slightly lower than lockbox-real video probs.

## 3. Job 3 — Per-identity lockbox FPR breakdown (Slot 1 top_n_step2000 @ τ=0.4496)

Source: `_per_identity_lockbox_2026-05-13.csv`.

### 3.1 teams_real_all_lockbox composition

Identity counts (video-level):

| Identity | n_videos | % of lockbox | Chronic-6? |
|---|---:|---:|:-:|
| dor_shkedi | 1138 | 83.6% | Y |
| real_dor | 109 | 8.0% | Y |
| bla_bla_chow | 61 | 4.5% | Y |
| PC_Generator | 28 | 2.1% | Y |
| Chikara_Takahashi | 25 | 1.8% | — |

The lockbox carries 5 distinct identities; 4 of 5 are chronic-6 members. dor_shkedi alone is 83.6% of lockbox-real videos.

### 3.2 Per-identity FPR comparison @ each ckpt's dev-cal τ

| Identity | n | P8A FPR (τ=0.9143) | S1_top_n_2000 FPR (τ=0.4496) | S1_step1500 FPR (τ=0.4543) | S1_step3500 FPR (τ=0.4448) |
|---|---:|---:|---:|---:|---:|
| dor_shkedi | 1138 | 0.0070 (8/1138) | **0.4578** (521/1138) | 0.4552 (518/1138) | 0.4631 (527/1138) |
| PC_Generator | 28 | 0.2857 (8/28) | 0.3571 (10/28) | 0.3571 (10/28) | 0.3571 (10/28) |
| real_dor | 109 | 0.0000 (0/109) | 0.0550 (6/109) | 0.0550 (6/109) | 0.0459 (5/109) |
| Chikara_Takahashi | 25 | **0.3600** (9/25) | 0.1600 (4/25) | 0.1600 (4/25) | 0.2400 (6/25) |
| bla_bla_chow | 61 | 0.0000 (0/61) | 0.0000 (0/61) | 0.0000 (0/61) | 0.0000 (0/61) |
| **Total** | **1361** | **0.0184** (25/1361) | **0.3963** (539/1361) | 0.3954 (538/1361) | 0.4070 (548/1361) |

(Per-ckpt totals here use frame-level alignment in §3.3; the video-level scorecard totals from `_partial_scorecard_2026-05-13.csv` are 0.0184 / 0.3960 / 0.3931 / 0.4026, matching within rounding.)

### 3.3 Concentration on chronic-6 identities

For Slot 1 top_n_step2000 (the canonical task candidate) @ τ=0.4496:

- Total false-positive (video) count: **541** of **1361** real videos (FPR 0.397).
- Chronic-6 share: **531/541 = 98.2%** of FPs.
- dor_shkedi alone: **521/541 = 96.3%** of FPs.

For P8A @ τ=0.9143:

- Total false-positive (video) count: **25** of **1361** (FPR 0.0184).
- Chronic-6 share: 16/25 = 64.0% of FPs (8 dor_shkedi + 8 PC_Generator).
- The non-chronic identity Chikara_Takahashi contributes 9/25 = 36% of P8A's lockbox FPs.

### 3.4 Identity-direction reversal

| Identity | P8A FPR | S1_2000 FPR | Direction |
|---|---:|---:|---|
| dor_shkedi | 0.0070 | 0.4578 | **+65× worse on Slot 1** |
| PC_Generator | 0.2857 | 0.3571 | +1.25× worse on Slot 1 |
| real_dor | 0.0000 | 0.0550 | new FPs on Slot 1 |
| Chikara_Takahashi | 0.3600 | 0.1600 | 2.25× **better** on Slot 1 |
| bla_bla_chow | 0.0000 | 0.0000 | tied at zero |

## 4. Direct observations

1. P8A's per-frame `frame_prob` ranges 0.005–0.995 across all 9 contract suites (frame-level CSVs). Slot 1 top_n_step2000's range is 0.38–0.51 across the same 9 suites (≈ 13× compressed) (§1.2).
2. Slot 1 top_n_step2000 fake-lockbox max prob is 0.465; real-lockbox max prob is 0.475. Fake-lockbox mean (0.4412) is below real-lockbox mean (0.4432) by 0.002 absolute (§2.3).
3. Mean-shift (Slot 1 top_n_step2000 − P8A) is positive on all 5 real suites (+0.116 to +0.347) and negative on 3 of 4 fake suites (−0.084 to −0.293); positive on visomaster (+0.101) (§1).
4. Per-frame Spearman r between Slot 1 top_n_step2000 and P8A ranges 0.20–0.86 across the 9 suites; the highest is `teams_fake_all_lockbox` at 0.85 and `visomaster_enhanced_macro_dev` at 0.81; the lowest is `teams_real_all_lockbox` at 0.20 (§1.3).
5. At τ_match (lb_real_fpr ≤ 0.025, E2B-comparable Pillar 1), all 3 Slot 1 ckpts produce `lb_fake_recall = 0.0000` and `dev_macro_recall ≈ 0.195` (§2.1).
6. At Slot 1 top_n_step2000's dev-cal τ=0.4496, 539/541 = 98.2% of lockbox false-positives come from chronic-6 identities; 521/541 = 96.3% from dor_shkedi alone (§3.3).
7. dor_shkedi identity n=1138 = 83.6% of `teams_real_all_lockbox` (§3.1). P8A fires on 8/1138 dor_shkedi videos (FPR 0.0070); Slot 1 top_n_step2000 fires on 521/1138 (FPR 0.4578) — a 65× increase (§3.2, §3.4).
8. Chikara_Takahashi (non-chronic, n=25): P8A FPR 0.3600 (9/25); Slot 1 top_n_step2000 FPR 0.1600 (4/25). Direction reverses — Slot 1 is better on this identity (§3.4).
9. As τ rises from 0.4496 to 0.4750 in 0.0254 absolute, Slot 1 top_n_step2000 `lb_real_fpr` falls from 0.398 to 0.001 (collapses to zero); `dev_macro_recall` falls from 0.390 to 0.145; `lb_fake_recall` falls from 0.261 to 0.000 (§2).
10. No τ in the 0.30 – 0.99 sweep yields both `lb_real_fpr ≤ 0.025` AND `lb_fake_recall ≥ 0.30` for any of the 3 Slot 1 ckpts (§2). The maximum `lb_fake_recall` value achieved at any τ with `lb_real_fpr ≤ 0.07` is `lb_fake_recall ≈ 0.05` (at τ ≈ 0.4640).
11. The maximum `dev_macro_recall` value achieved at any τ with `lb_real_fpr ≤ 0.07` is `dev_macro_recall ≈ 0.24` (at τ ≈ 0.4640), below the 0.30 floor.
12. Lockbox AUC (`teams_real_all_lockbox` vs `teams_fake_all_lockbox`, video-level) is **0.4442** for Slot 1 top_n_step2000 vs **0.9355** for P8A — a drop of 0.491 absolute, with Slot 1's lockbox AUC sitting below 0.5 (§2.4). Same-direction drops for step1500 (0.4449) and step3500 (0.4450).
13. Dev AUC (`teams_real_all_dev` vs `teams_fake_all_dev`, video-level) is **0.8635** for Slot 1 top_n_step2000 vs **0.8896** for P8A — a drop of 0.026 absolute (§2.4). Slot 1 retains 97% of P8A's dev-substrate discriminability but 47% of P8A's lockbox-substrate discriminability.
14. visomaster_enhanced_macro_dev AUC (vs teams_real_all_dev reals) is **0.8974** for Slot 1 top_n_step2000 vs **0.7403** for P8A — Slot 1 is +0.157 absolute higher on this fake suite (§2.4).

## 5. Output artifacts

- `_frame_cache/` — 36 per-(suite × ckpt) frame-level CSVs (gitignored).
- `_shift_profile_2026-05-13.csv` — Job 1 table.
- `_tau_recal_2026-05-13.csv` — Job 2 τ-sweep, 3 ckpts × 34 τ values.
- `_tau_match_e2b_2026-05-13.csv` — Job 2 precise τ_match for `lb_fpr ≤ 0.025`.
- `_per_identity_lockbox_2026-05-13.csv` — Job 3 identity-level FPR for all 4 ckpts.
- `_results_2026-05-13.json` — consolidated JSON of all three jobs.
- `run_shift_analysis.py` — re-runnable extraction script.
