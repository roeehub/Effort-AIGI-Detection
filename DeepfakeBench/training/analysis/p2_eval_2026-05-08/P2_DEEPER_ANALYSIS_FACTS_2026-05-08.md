# P2 Phase A — deeper analysis FACTS (J1-J5, 2026-05-08)

> **Status: factual-only.** No interpretation. Forbidden words: succeeds, fails,
> wins, promotes, deployment-grade.
>
> **Companion FACTS doc**: [`P2_PHASE_A_VERDICT_FACTS_2026-05-08.md`](P2_PHASE_A_VERDICT_FACTS_2026-05-08.md)
> — promotion-contract ranking + per-suite at contract τ.
>
> **Companion OPINIONS doc**: [`P2_DEEPER_ANALYSIS_OPINIONS_2026-05-08.md`](P2_DEEPER_ANALYSIS_OPINIONS_2026-05-08.md)
> — synthesis, mechanism debate, next-packet design space.
>
> **Source**: 203 frame-level `*_frames_report.csv` files pulled from
> `gs://training-job-outputs/test_results/teams_promotion_contract/p2-scratch-scorecard-2026-05-08/reports/`.
> Driver scripts at `p2_deeper_analysis/build_scoreboard.py` +
> `p2_deeper_analysis/run_analyses.py`. Outputs in
> `p2_deeper_analysis/outputs/`.

---

## 1. Setup

Built unified per-frame scoreboard: 17,634 unique (suite, frame_path) rows × 7
ckpts. Each ckpt has prob_fake on every frame in every suite where the ckpt
was scored. Saved as `outputs/scoreboard.parquet` (0.9 MB).

Run-order: J1 → J2 → J3 → J4 → J5 sequential, ~2 min total runtime on CPU.

---

## 2. J1 — F4-lite τ recalibration

**Definition**: F4-lite filter drops chronic-6 (PC_Generator, Roy_D,
bla_bla_chow, Q_*) frames from the real cohorts. Memory
`project_job14_substrate_clean_2026-05-04.md` notes "chronic-6 axis alone
explains 96-108% of FPR drop in F4". Lex policy applied to F4-lite-filtered
substrate; per-ckpt new τ search; lockbox readout at the new τ also computed
on F4-lite-filtered lockbox real cohort.

Source: `outputs/j1_f4_recalibration.csv`.

| F4 rank | ckpt | F4 τ | F4 primary_fpr | F4 stress_fpr | F4 fake_macro_recall | F4 lockbox_real_fpr | F4 lockbox_fake_recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | P2_C_PAIRRANK_PERIODIC_STEP3000 | 0.971 | 0.0007 | 0.0012 | 0.307 | 0.019 | 0.016 |
| 2 | **P2_D_FOURIER_PERIODIC_STEP3000** | 0.575 | 0.0016 | 0.0049 | **0.336** | **0.078** | **0.769** |
| 3 | P2_D_FOURIER_PERIODIC_STEP8000 | 0.924 | 0.0023 | 0.0074 | 0.301 | 0.024 | 0.261 |
| 4 | E2B_TOP_N_STEP3200 | 0.915 | 0.0023 | 0.0081 | **0.363** | 0.007 | 0.336 |
| 5 | P8A_REFERENCE_STEP5000 | 0.940 | 0.0033 | 0.0074 | 0.305 | 0.011 | 0.374 |
| 6 | P2_C_PAIRRANK_TOP_N_STEP7000 | 0.983 | 0.0043 | 0.0087 | 0.320 | 0.009 | 0.348 |
| 7 | P2_D_FOURIER_TOP_N_STEP19000 | 0.988 | 0.0092 | 0.0242 | 0.302 | 0.008 | 0.216 |

**Direct observations**:

1. F4-lite filter collapses primary_fpr from 5-7% (full substrate) to 0.07-0.92%
   for all ckpts. Chronic-6 frames carry most of the dev primary FPR.
2. D step3000's F4 lockbox_real_fpr drops from 16.4% (full) to 7.8% (F4) —
   chronic-6 frames in the dev real cohort do NOT account for the lockbox FPR
   directly (different substrate); the drop is from re-calibrated τ
   (F4 τ = 0.575 vs full τ = 0.460). At lower-FPR-budget τ on F4, lockbox FPR
   also lowers.
3. D step3000's F4 lockbox_fake_recall = 0.769 — the recall lift survives F4.
4. F4 lex-rank reorders the leaderboard: C step3000 climbs to rank 1
   (its primary_fpr collapses to 0.0007 — the smallest), D step3000 to rank 2.
   P8A drops from rank 1 (full) to rank 5 (F4).
5. E2B retains the highest F4 dev_fake_macro_recall (0.363).

**MODEL_GOALS.md promotion-bar evaluation on F4-lite** (D step3000 vs criteria):
- (1) Beat E2B on dev_fake_macro_recall: D 0.336 < E2B 0.363. Not met.
- (2) Not regress P8A's chronic-FP (≤5pp over P8A's lockbox_real_fpr):
  D 0.078 vs P8A 0.011 → +6.7pp. Not met (over the 5pp bar).
- (3) F1 target (≥0.90 lockbox_fake_recall at FPR ≤ 0.10):
  D 0.769 at FPR 0.078. 0.769 < 0.90. Not met.

**Outcome on F4-lite**: no ckpt clears all three. P2_C_PAIRRANK_PERIODIC_STEP3000
is F4 lex-rank 1, but C also fails the promotion bar (lockbox_fake_recall
0.016 << 0.90).

---

## 3. J2 — FPR-matched τ counterfactual

**Definition**: For each ckpt, find the smallest τ where worst dev stress_fpr
≤ 0.069 (matching P8A's contract-selected stress_fpr 0.0685). Report
fake_macro_recall + lockbox readouts at that τ.

Source: `outputs/j2_fpr_matched_tau.csv`.

| ckpt | τ_matched | stress_fpr_at_τ | dev_fake_macro_recall_at_τ | lockbox_real_fpr_at_τ | lockbox_fake_recall_at_τ |
|---|---:|---:|---:|---:|---:|
| P8A_REFERENCE | 0.915 | 0.069 | 0.326 | 0.019 | 0.412 |
| E2B_TOP_N | 0.763 | 0.068 | **0.498** | 0.018 | 0.581 |
| P2_C_PAIRRANK_TOP_N_STEP7000 | 0.999 | 0.064 | 0.175 | 0.006 | 0.108 |
| P2_C_PAIRRANK_PERIODIC_STEP3000 | 0.849 | 0.069 | 0.480 | 0.080 | 0.092 |
| **P2_D_FOURIER_PERIODIC_STEP3000** | **0.503** | 0.069 | **0.481** | **0.122** | **0.845** |
| P2_D_FOURIER_PERIODIC_STEP8000 | 0.976 | 0.067 | 0.201 | 0.004 | 0.085 |
| P2_D_FOURIER_TOP_N_STEP19000 | 0.991 | 0.057 | 0.206 | 0.001 | 0.073 |

**Direct observations**:

1. At FPR-matched τ (worst stress_fpr ≤ P8A's 0.069):
   - D step3000 retains 84.5% lockbox_fake_recall (vs E2B 58.1%, P8A 41.2%).
   - D step3000 retains 48.1% dev_fake_macro_recall (vs E2B 49.8%, P8A 32.6%).
2. D step3000 lockbox_real_fpr at matched τ = 12.2% (vs P8A 1.9%, E2B 1.8%) —
   6.4× P8A even at matched dev stress_fpr.
3. The recall-vs-FPR ratios across ckpts at matched τ are NOT proportional:
   - D step3000 vs P8A: dev_fake_macro_recall ratio ~1.48×; lockbox_real_fpr
     ratio ~6.4×.
   - This non-proportionality means the recall lift and the FPR cost are
     not perfectly correlated through a single τ slider.

---

## 4. J3 — Per-frame disagreement (D step3000 vs P8A and E2B)

**Definition**: Δ = D_step3000 score − P8A score per frame across all
suites × labels. Source: `outputs/j3_disagreement_summary.csv` and
`outputs/j3_disagreement_top.csv` (top-50 |Δ| each direction).

Per-suite × per-label Δ summary (selected; full table in CSV):

| suite | label | n | mean_Δ | frac_D_higher | frac_D_higher_>0.2 | frac_D_lower_>0.2 |
|---|---:|---:|---:|---:|---:|---:|
| teams_real_all_dev | 0 | 4564 | -0.030 | 0.22 | 0.052 | 0.112 |
| teams_real_poor_quality_dev | 0 | 1303 | -0.002 | 0.16 | 0.039 | 0.066 |
| teams_real_lighting_extreme_dev | 0 | 1742 | +0.014 | 0.21 | 0.100 | 0.079 |
| teams_real_dor_dev | 0 | 50 | **+0.140** | **0.70** | **0.440** | 0.140 |
| **teams_real_all_lockbox** | 0 | 1418 | **+0.184** | **0.92** | **0.474** | 0.055 |
| teams_real_lighting_extreme_lockbox | 0 | 214 | +0.005 | 0.42 | 0.282 | 0.282 |
| teams_real_poor_quality_lockbox | 0 | 31 | -0.244 | 0.06 | 0.000 | 0.581 |
| teams_fake_all_dev | 1 | 3039 | -0.136 | 0.28 | 0.122 | 0.397 |
| teams_fake_all_lockbox | 1 | 425 | +0.035 | 0.38 | 0.360 | 0.257 |
| visomaster_enhanced_macro_dev | 1 | 550 | -0.057 | 0.31 | 0.220 | 0.304 |
| deeplive_enhanced_dev | 1 | 545 | -0.144 | 0.39 | 0.379 | 0.502 |
| teams_capture_dor_shkedi_dev | 1 | 78 | -0.454 | 0.13 | 0.000 | 0.962 |
| teams_capture_dor_shkedi_s16_dev | 1 | 78 | -0.454 | 0.13 | 0.000 | 0.962 |
| teams_capture_pc_generator_dev | 1 | 194 | -0.173 | 0.21 | 0.000 | 0.263 |

**Direct observations**:

1. D step3000 vs P8A on REAL cohorts:
   - On `teams_real_all_lockbox` (1418 frames): mean Δ = +0.184, 92% of frames
     have D higher than P8A, 47% have Δ > +0.2. Asymmetric: 5.5% have Δ < −0.2.
   - On `teams_real_dor_dev` (50 frames): mean Δ = +0.140, 70% D higher,
     44% have Δ > +0.2.
   - On dev real_all / poor_quality / lighting_extreme: mean Δ near 0
     (−0.030, −0.002, +0.014); D-higher fraction <22%.
   - On lockbox poor_quality (n=31, small): mean Δ = −0.244 (D LOWER).

2. D step3000 vs P8A on FAKE cohorts:
   - On `teams_fake_all_lockbox`: mean Δ = +0.035, 38% D higher, 36% have
     Δ > +0.2 AND 26% have Δ < −0.2 (bidirectional).
   - On `teams_capture_dor_shkedi_dev`: mean Δ = −0.454, 96% have Δ < −0.2
     (D systematically LOWER on dor fakes).
   - On `teams_fake_all_dev`: mean Δ = −0.136 (D lower on average).
   - On `deeplive_enhanced_dev`: mean Δ = −0.144 (D lower on average).

3. The "more aggressive on REAL lockbox" pattern (mean Δ +0.184, 92% D higher)
   is asymmetric: D is NOT systematically more aggressive on lockbox fakes
   (mean Δ +0.035, 38% D higher).

---

## 5. J4 — Per-identity breakdown

Source: `outputs/j4_per_identity_real_fpr.csv`,
`outputs/j4_per_identity_fake_recall.csv`. All metrics at each ckpt's
contract-selected τ (per `SELECTED_TAU` from `checkpoint_summary.csv`).

### 5.1 Real-side per-identity (only `teams_real_dor_dev` is in contract)

| ckpt | teams_real_dor_dev FPR |
|---|---:|
| P8A_REFERENCE_STEP5000 | 0.080 |
| E2B_TOP_N_STEP3200 | 0.120 |
| P2_C_PAIRRANK_PERIODIC_STEP3000 | 0.180 |
| **P2_D_FOURIER_PERIODIC_STEP3000** | **0.460** |
| P2_C_PAIRRANK_TOP_N_STEP7000 | 0.040 |
| P2_D_FOURIER_PERIODIC_STEP8000 | 0.040 |
| P2_D_FOURIER_TOP_N_STEP19000 | 0.020 |

### 5.2 Fake-side per-identity capture suites (selected)

| capture suite | n | P8A | E2B | C step3000 | **D step3000** | D step8000 | D step19000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| teams_capture_dor_shkedi_dev | 78 | 0.859 | 0.628 | 0.231 | **0.615** | 0.013 | 0.013 |
| teams_capture_dor_shkedi_s16_dev | 78 | 0.859 | 0.628 | 0.231 | **0.615** | 0.013 | 0.013 |
| teams_capture_noyn_sharker_dev | 324 | 0.707 | 0.963 | 0.049 | **0.562** | 0.296 | 0.679 |
| teams_capture_pc_generator_dev | 194 | 1.000 | 0.969 | 0.897 | 0.902 | 0.814 | 0.794 |
| teams_capture_pc_generator_s3_dev | 118 | 1.000 | 0.949 | 0.856 | 0.864 | 0.754 | 0.712 |
| teams_capture_pc_generator_s4_dev | 30 | 1.000 | 1.000 | 0.933 | **1.000** | 0.867 | 0.900 |
| teams_capture_pc_generator_s9_dev | 46 | 1.000 | 1.000 | 0.978 | 0.935 | 0.935 | 0.935 |
| teams_capture_cam_test_dev | 809 | 0.904 | 0.938 | 0.910 | **0.990** | 0.939 | 0.949 |
| teams_capture_cam_test_s32_dev | 235 | 1.000 | 1.000 | 0.911 | 0.992 | 0.962 | 0.987 |
| teams_capture_cam_test_s35_dev | 365 | 0.792 | 0.995 | 0.986 | **0.992** | 0.975 | 0.981 |
| teams_capture_cam_test_s38_dev | 85 | 1.000 | 1.000 | 0.800 | 0.988 | 0.894 | 0.965 |
| teams_capture_cam_test_s46_dev | 124 | 0.984 | 0.613 | 0.758 | **0.984** | 0.823 | 0.774 |
| teams_capture_test_cam_dev | 404 | 0.990 | 0.993 | 0.542 | **0.923** | 0.861 | 0.795 |
| teams_capture_test_cam_s53_dev | 165 | 0.982 | 1.000 | 0.624 | 0.952 | 0.933 | 0.703 |
| teams_capture_test_cam_s73_dev | 101 | 1.000 | 1.000 | 0.842 | 1.000 | 1.000 | 0.990 |
| teams_capture_test_cam_s76_dev | 138 | 0.993 | 0.978 | 0.225 | 0.833 | 0.674 | 0.761 |
| teams_flat_xiang_xiang2_feng_dev | 135 | 0.074 | 0.474 | 0.178 | 0.200 | 0.089 | 0.059 |

**Direct observations**:

1. `teams_real_dor_dev` FPR for D step3000 = 46.0% vs P8A 8.0% (5.75×).
2. D step3000 fake recall vs P8A by identity:
   - dor_shkedi: 0.615 vs 0.859 → -24.4pp
   - noyn_sharker: 0.562 vs 0.707 → -14.5pp
   - pc_generator (most variants): 0.86-1.00 vs 1.00 → comparable or -10pp
   - cam_test variants: D step3000 ≥ P8A on most variants
   - test_cam variants: D step3000 ≈ P8A or modestly lower
   - xiang_xiang2_feng: D step3000 0.200 vs P8A 0.074 → +12.6pp
3. The pattern is non-uniform: regression on Dor + Noyn_Sharker
   (recall drops 14-24pp); comparable or improvement on most others.

---

## 6. J5 — Per-method recall × IQ signature

Source: `outputs/j5_per_method_recall.csv`. Recall at contract-selected τ per
ckpt. Atlas lap_var p50 attached for the 4 main fake suites where the IQ
atlas measured the same pool.

| suite | n | P8A | E2B | C step3000 | **D step3000** | atlas lap_var p50 |
|---|---:|---:|---:|---:|---:|---|
| **teams_fake_all_lockbox** | 425 | 0.412 | 0.642 | 0.125 | **0.875** | **11.2** (very smooth) |
| visomaster_enhanced_macro_dev | 550 | 0.135 | 0.051 | 0.116 | **0.165** | 81.5 (moderate) |
| deeplive_enhanced_dev | 545 | 0.239 | 0.796 | 0.895 | 0.769 | 171.9 (sharp; HDTF-similar) |
| teams_fake_all_dev | 3039 | 0.604 | 0.735 | 0.572 | **0.696** | 81.5 |
| capture_cam_test_dev (all variants combined) | 1818 | 0.92-1.00 | 0.61-1.00 | 0.76-0.99 | 0.92-1.00 | — |
| capture_test_cam_dev (all variants combined) | 808 | 0.98-1.00 | 0.97-1.00 | 0.22-0.84 | 0.83-1.00 | — |
| capture_dor_shkedi_dev | 78 | 0.859 | 0.628 | 0.231 | 0.615 | — |
| capture_noyn_sharker_dev | 324 | 0.707 | 0.963 | 0.049 | 0.562 | — |
| capture_pc_generator_dev (all variants combined) | 388 | 1.000 | 0.96-1.00 | 0.86-0.98 | 0.86-1.00 | — |
| teams_flat_xiang_xiang2_feng_dev | 135 | 0.074 | 0.474 | 0.178 | 0.200 | — |

**Direct observations**:

1. D step3000 has the highest recall on `teams_fake_all_lockbox` (0.875) —
   the substrate with the lowest atlas lap_var p50 (11.2). P8A on the same:
   0.412.
2. D step3000 has the highest recall on `visomaster_enhanced_macro_dev`
   (0.165) — first material lift over the 13+ packet ceiling at contract τ.
3. D step3000 on `deeplive_enhanced_dev` (atlas lap_var p50 = 171.9, the
   sharpest fake substrate measured): 0.769. C step3000 on the same: 0.895
   (highest); E2B 0.796.
4. D step3000's relative position vs P8A:
   - lockbox_fake (smoothest substrate): +46.3pp over P8A (largest delta)
   - viso_enhanced_macro: +3.0pp over P8A (smallest positive delta)
   - deeplive_enhanced: +53.0pp over P8A (large delta on a sharper substrate)
   - dor_shkedi capture: -24.4pp under P8A (largest negative delta)
5. The recall-vs-IQ-signature correlation on the 3 atlas-measured fake suites
   for D step3000 minus P8A:
   - lockbox (lap_var=11): +0.463
   - visomaster (lap_var=81): +0.030
   - teams_fake_all_dev (lap_var=81): +0.092
   - deeplive (lap_var=172): +0.530
   - The relationship is NOT monotone in lap_var.

---

## 7. Cross-references

- **Verdict + per-suite tables**: `P2_PHASE_A_VERDICT_FACTS_2026-05-08.md`
- **Synthesis + interpretation**: `P2_DEEPER_ANALYSIS_OPINIONS_2026-05-08.md`
- **D5 ckpt-mapping FACTS**: `d1_d4_cpu/D5_CKPT_MAPPING_FACTS_2026-05-08.md`
- **Canary resolution audit**: `d1_d4_cpu/CANARY_RESOLUTION_FACTS_2026-05-08.md`
- **IQ atlas (substrate-level IQ measurements)**:
  `analysis/iq_data_atlas_2026-05-08/IQ_ATLAS_FACTS_2026-05-08.md`
- **Memory entries activated**:
  - `project_p8a_breakthrough.md` — P8A's signature dor invariance pattern.
  - `project_image_quality_shortcut.md` — IQ-axis correlation with score.
  - `project_viso_ceiling_unbroken_10_packets.md` — viso ceiling history.
  - `project_chronic_offenders_partition_per_ckpt_2026-05-04.md` — chronic-6
    partition.
  - `project_lockbox_fpr_dominated_by_webcam_mode.md` — lockbox FPR axis.
  - `project_phase1a_method_cluster_axis_2026-05-01.md` — method-cluster axis.

## 8. Artifacts

- `analysis/p2_eval_2026-05-08/p2_deeper_analysis/build_scoreboard.py`
- `analysis/p2_eval_2026-05-08/p2_deeper_analysis/run_analyses.py`
- `analysis/p2_eval_2026-05-08/p2_deeper_analysis/outputs/scoreboard.parquet` (regenerable)
- `analysis/p2_eval_2026-05-08/p2_deeper_analysis/outputs/j1_f4_recalibration.csv`
- `analysis/p2_eval_2026-05-08/p2_deeper_analysis/outputs/j1_f4_promotion_winner.json`
- `analysis/p2_eval_2026-05-08/p2_deeper_analysis/outputs/j2_fpr_matched_tau.csv`
- `analysis/p2_eval_2026-05-08/p2_deeper_analysis/outputs/j3_disagreement_top.csv`
- `analysis/p2_eval_2026-05-08/p2_deeper_analysis/outputs/j3_disagreement_summary.csv`
- `analysis/p2_eval_2026-05-08/p2_deeper_analysis/outputs/j4_per_identity_real_fpr.csv`
- `analysis/p2_eval_2026-05-08/p2_deeper_analysis/outputs/j4_per_identity_fake_recall.csv`
- `analysis/p2_eval_2026-05-08/p2_deeper_analysis/outputs/j5_per_method_recall.csv`
- 203 frame-level reports cached at `analysis/p2_eval_2026-05-08/p2_deeper_analysis/frame_reports/` (regenerable)
