# P1 follow-up CPU jobs — factual results, 2026-05-07

**Status**: factual-only. No interpretation. Numbers and direct observations.

This doc consolidates 7 CPU follow-ups beyond the F1-F5 audit:
1. Task A — `roy_d` per-frame axis attribution
2. Task B — F2(a) with relaxed missed-fake thresholds
3. Task C — PD-vs-P1 contract-suite comparison
4. Task D — lockbox FPR/recall vs τ curve per ckpt
5. Task E — `face_area_fraction` signed-r direction
6. Counter-theory probe — per-identity regression magnitudes BUNDLE vs PAIRRANK
7. Cross-checks against the bug-fixed `phase_d/run_chronic_filter.py`

Companion CSVs are listed at the bottom.

---

## 1. Task A — roy_d per-frame axis attribution

**Question answered**: Are the 130-frame roy_d FPR transitions (P8A 29% → P1 78-93%) axis-aligned, or axis-orthogonal?

**Method**: For each of the 130 roy_d frames in `teams_real_all_dev`, decoded the cached image (from `f3_color_b_dev/_frame_cache/`) and computed sharpness (Laplacian variance), min_dim (min(W,H)), and color_b_dev (B-channel std). Joined to per-frame scores from each ckpt. Computed Pearson r.

**Output**: `roy_d_regression/roy_d_with_axes.csv` (130 rows × ~12 cols).

### Raw-score r(score, axis) per ckpt — 130 roy_d frames

| ckpt | r(sharpness) | r(min_dim) | r(color_b_dev) |
|---|---:|---:|---:|
| **P8A** | −0.089 | −0.282 | **−0.714** |
| BUNDLE_step500 | +0.081 | +0.215 | +0.195 (sign-flipped) |
| BUNDLE_step3750 | +0.031 | −0.036 | −0.189 |
| BUNDLE_step4000 | +0.035 | −0.030 | −0.199 |
| PAIRRANK_step500 | +0.016 | −0.186 | −0.509 |
| PAIRRANK_step6000 | +0.042 | −0.148 | −0.303 |
| PAIRRANK_step6750 | +0.032 | −0.107 | −0.317 |

### Δscore (P1 − P8A) vs axis r — 130 roy_d frames

| ckpt | r(Δsharpness) | r(Δmin_dim) | r(Δcolor_b_dev) |
|---|---:|---:|---:|
| BUNDLE_step500 | +0.089 | +0.283 | **+0.714** |
| BUNDLE_step3750 | +0.090 | +0.283 | +0.712 |
| BUNDLE_step4000 | +0.093 | +0.284 | +0.709 |
| PAIRRANK_step500 | +0.128 | +0.241 | +0.582 |
| PAIRRANK_step6000 | +0.108 | +0.261 | +0.681 |
| PAIRRANK_step6750 | +0.106 | +0.273 | +0.677 |

### Direct observations
- P8A's score on roy_d correlates strongly negatively with `color_b_dev` (r=−0.714).
- For all 6 P1 ckpts, the (P1 − P8A) score Δ correlates strongly POSITIVELY with `color_b_dev` (r=+0.58 to +0.71).
- The Δ-r magnitude on `color_b_dev` matches P8A's r magnitude almost exactly (~0.71). On `min_dim` and `sharpness` it is much smaller (≤0.28 and ≤0.13).

---

## 2. Task B — F2(a) with relaxed missed-fake thresholds

**Question answered**: F2(a) failed at the strict P8A `frame_prob < 0.5` filter due to baseline saturation. Does relaxing the threshold change the verdict?

**Method**: Reused the F2 computation script with the missed-fake filter relaxed to P8A frame_prob < 0.7 and < 0.9. Output: `f2_pair_rank/relaxed_thresholds.csv`.

### n_pairs at each threshold

| threshold | n_pairs total | sub-lanes with n_pairs > 0 |
|---|---:|---:|
| < 0.5 | 2,286 | 3 (dor_shkedi__s16, test_cam__s76, xiang_xiang2_feng) |
| < 0.7 | 2,993 | 3 (same three) |
| < 0.9 | 4,034 | 4 (the three above + pc_generator__s15) |

### At <0.9 threshold: pc_generator__s15 lane (n=29)

| ckpt | frac_fake_gt_real | lift_rel_pct | passes 30% bar |
|---|---:|---:|:---:|
| P8A | 0.483 | 0.0 | (baseline) |
| BUNDLE_step500 | 0.966 | **+100.0%** | **YES** |
| BUNDLE_step3750 | 0.103 | −78.6% | no |
| BUNDLE_step4000 | 0.069 | −85.7% | no |
| PAIRRANK_step500 | 0.552 | +14.3% | no |
| PAIRRANK_step6000 | 0.552 | +14.3% | no |
| PAIRRANK_step6750 | 0.621 | +28.6% | no |

### At <0.9 threshold: other 3 lanes

| lane | n | P8A | best non-P8A lift |
|---|---:|---:|---|
| dor_shkedi__s16 | 341 | 1.000 | 0% (BUNDLE_500/PAIRRANK_500/PAIRRANK_6750) |
| test_cam__s76 | 27 | 1.000 | 0% (all ckpts at 1.000) |
| xiang_xiang2_feng | 3637 | 0.982 | +1.5% (BUNDLE_500) |

### Direct observations
- At the strict `<0.5` filter, no P1 ckpt passes the 30% relative-lift bar on any lane.
- At `<0.9`, BUNDLE_step500 passes the bar on `pc_generator__s15` (1 lane). YAML criterion requires ≥ 2 lanes.
- BUNDLE_step3750 and step4000 show large negative lift (−79%, −86%) on `pc_generator__s15` at the relaxed threshold — they predict "real beats fake" on previously-missed-and-now-relaxed fakes.
- The other 3 lanes have P8A baselines at 0.97-1.00 (saturated); no headroom for the +30% bar.

---

## 3. Task C — PD-vs-P1 contract-suite comparison

**Question answered**: How does the P1 packet compare to the PD (corr_penalty) packet on the 9 shared contract suites?

**Method**: PD's `unified_scorecard_simple.csv` is a 232-row × 9-col table at τ=0.5 only. P1's contract scorecard uses calibrated τ. Compared at τ=0.5 (apples-to-apples) and at calibrated τ (P1's deployment-relevant readings). Output: `p1_at_tau05_for_pd_compare.csv`.

### Best ckpt per suite at τ=0.5 — PD set vs P1 set

| suite | PD best (value) | PD ckpt | P1 best at τ=0.5 (value) | P1 ckpt |
|---|---:|---|---:|---|
| teams_real_all_dev | 0.0876 (FPR) | deeplive_corr_top_n_step4800 | 0.1008 (FPR) | e2b_top_n_step3200 |
| teams_real_all_lockbox | 0.0617 (FPR) | p8a_reference_step5000 | 0.0628 (FPR) | e2b_top_n_step3200 |
| teams_real_dor_dev | 0.2400 (FPR) | e2b_top_n_step3200 | 0.2400 (FPR) | e2b_top_n_step3200 |
| teams_real_lighting_extreme_dev | 0.1113 (FPR) | p8a_reference_step5000 | 0.1068 (FPR) | p8a_reference_step5000 |
| teams_real_poor_quality_dev | 0.0672 (FPR) | deeplive_corr_top_n_step4800 | 0.0814 (FPR) | p8a_reference_step5000 |
| teams_fake_all_dev | 0.8273 (recall) | viso_corr_top_n_step600 | 0.9552 (recall) | p1_bundle_periodic_step500 |
| teams_fake_all_lockbox | 0.9921 (recall) | viso_corr_periodic_step1000 | 1.0000 (recall) | p1_bundle_periodic_step500 |
| visomaster_enhanced_macro_dev | 0.3564 (recall) | p8a_reference_step5000 | 0.7745 (recall) | p1_bundle_periodic_step500 |
| deeplive_enhanced_dev | 1.0000 (recall) | viso_corr_top_n_step600 | 0.9835 (recall) | p1_pairrank_top_n_step6750 |

### Calibrated-τ (P1) vs PD τ=0.5

| suite | PD best τ=0.5 | P1 BUNDLE_500 (calibrated τ=0.992) | P1 PAIRRANK_500 (calibrated τ=0.768) |
|---|---:|---:|---:|
| teams_real_all_dev (FPR) | 0.0876 | 0.0670 | 0.0609 |
| teams_real_all_lockbox (FPR) | 0.0617 | 0.0331 | 0.0184 |
| teams_real_dor_dev (FPR) | 0.2400 | **0.0400** | 0.1600 |
| teams_real_lighting_extreme_dev (FPR) | 0.1113 | 0.0999 | 0.0992 |
| teams_real_poor_quality_dev (FPR) | 0.0672 | 0.0455 | 0.0347 |
| teams_fake_all_dev (recall) | 0.8273 | 0.2196 | 0.5089 |
| teams_fake_all_lockbox (recall) | 0.9921 | 0.8261 | 0.7075 |
| visomaster_enhanced_macro_dev (recall) | 0.3564 | 0.0055 | 0.2164 |
| deeplive_enhanced_dev (recall) | 1.0000 | 0.0000 | 0.3358 |

### Direct observations
- PD scorecard ckpt set: 8 ckpts (3 deeplive_corr variants, 3 viso_corr variants, P8A, E2B).
- At τ=0.5: P1 BUNDLE_500 leads on 4 fake suites (matched/exceeded PD's best on teams_fake_all_dev, teams_fake_all_lockbox, visomaster_enhanced_macro_dev). PD's viso_corr leads on deeplive_enhanced_dev.
- At calibrated τ: P1 has lower FPR than PD's best τ=0.5 reading on 5 of 5 real suites. Fake recall at calibrated τ collapses on `visomaster_enhanced_macro_dev` (0.005 BUNDLE / 0.216 PAIRRANK) and `deeplive_enhanced_dev` (0.000 BUNDLE / 0.336 PAIRRANK), much lower than PD τ=0.5.
- Apples-to-apples comparison requires PD scorecard at calibrated τ, which is not in `unified_scorecard_simple.csv`. PD's `promotion_contract/` outputs may have it; not pulled in this audit.

---

## 4. Task D — Lockbox FPR/recall vs τ curve per ckpt

**Question answered**: Is the F1 close criterion (lockbox recall ≥ 90% at FPR ≤ 10%) reachable for any P1 ckpt at a non-contract τ?

**Method**: Loaded `teams_real_all_lockbox` and `teams_fake_all_lockbox` per-frame reports for each of 8 ckpts. Swept τ across 86 grid points from 0.5 to 0.9999. For each (ckpt, τ), computed lockbox FPR (over n=1418 reals) and lockbox recall (over n=425 fakes). Output: `lockbox_roc_curves.csv`.

### Best lockbox recall per ckpt under FPR budget

| ckpt | best recall @ FPR≤5% (τ) | best recall @ FPR≤10% (τ) | recall at calibrated τ |
|---|---:|---:|---:|
| P8A | 0.579 (τ=0.64) | 0.664 (τ=0.50) | 0.412 (τ=0.9156, FPR=0.019) |
| E2B | 0.793 (τ=0.56) | 0.833 (τ=0.50) | 0.642 (τ=0.7108, FPR=0.023) |
| **BUNDLE_step500** | **0.859** (τ=0.992) | **0.965** (τ=0.989) | 0.845 (τ=0.9919, FPR=0.038) |
| BUNDLE_step3750 | 0.381 (τ=0.996) | 0.482 (τ=0.982) | 0.212 (τ=0.9994, FPR=0.013) |
| BUNDLE_step4000 | 0.433 (τ=0.995) | 0.536 (τ=0.971) | 0.320 (τ=0.9989, FPR=0.021) |
| **PAIRRANK_step500** | **0.842** (τ=0.60) | **0.915** (τ=0.50) | 0.718 (τ=0.7677, FPR=0.022) |
| PAIRRANK_step6000 | 0.471 (τ=0.94) | 0.558 (τ=0.83) | 0.367 (τ=0.9878, FPR=0.020) |
| PAIRRANK_step6750 | 0.647 (τ=0.97) | 0.758 (τ=0.89) | 0.565 (τ=0.9900, FPR=0.027) |

### BUNDLE_step500 detail across τ

| τ | lockbox FPR | lockbox recall |
|---|---:|---:|
| 0.50 | 0.7137 | 1.0000 |
| 0.90 | 0.3470 | 0.9953 |
| 0.95 | 0.2645 | 0.9953 |
| 0.99 | 0.0663 | 0.9482 |
| 0.992 | 0.0346 | 0.8047 |
| 0.9919 (contract τ) | 0.0367 | 0.8306 |
| 0.999 | 0.0000 | 0.0000 |

### Direct observations
- At FPR ≤ 10%, BUNDLE_step500 (τ=0.989) and PAIRRANK_step500 (τ=0.50) both exceed the F1 90% bar. BUNDLE_step500 reaches 96.5%; PAIRRANK_step500 reaches 91.5%.
- At FPR ≤ 5%, the best is BUNDLE_step500 at 85.9% (τ=0.992) — 4.1pp short of F1.
- At the contract-selected τ, BUNDLE_step500 lockbox recall is 84.5%; the contract chose a slightly lower-FPR τ that gives up 12pp recall vs the F1-passing τ.
- 4 of 8 ckpts have NO τ that gives lockbox recall ≥ 50% within the FPR ≤ 5% budget (BUNDLE_step3750, BUNDLE_step4000, PAIRRANK_step6000, P8A is at 58%). Those ckpts have a structurally narrow lockbox-recall envelope.

---

## 5. Task E — face_area_fraction signed direction

**Question answered**: F3 reported face_area_fraction amplifies +106-266% in absolute terms across P1 ckpts. Does the SIGN of the correlation flip across ckpts × suites?

**Method**: Read `correlations.csv` from the F3 audit (8 ckpts × 6 suites × 3 features). Filtered to `feature == 'face_area_fraction'`. Reported signed Pearson r (not |r|).

### Signed r per (ckpt × suite) — face_area_fraction

| ckpt | deeplive_enh_dev | teams_fake_all_dev | teams_fake_all_lockbox | teams_real_all_dev | teams_real_all_lockbox |
|---|---:|---:|---:|---:|---:|
| P8A | −0.144 | **+0.648** | −0.443 | −0.215 | +0.019 |
| E2B | −0.265 | +0.277 | −0.423 | −0.271 | −0.362 |
| BUNDLE_step500 | −0.086 | +0.362 | −0.010 | −0.376 | **+0.480** |
| BUNDLE_step3750 | −0.073 | +0.121 | −0.240 | −0.412 | +0.211 |
| BUNDLE_step4000 | −0.059 | +0.135 | −0.202 | −0.378 | +0.179 |
| PAIRRANK_step500 | −0.074 | +0.485 | −0.112 | −0.309 | +0.311 |
| PAIRRANK_step6000 | −0.125 | +0.252 | −0.358 | −0.338 | −0.143 |
| PAIRRANK_step6750 | −0.066 | +0.190 | −0.219 | −0.345 | −0.287 |

### Signed mean r per (ckpt × suite_kind) on face_area_fraction

| ckpt | real-side signed mean | fake-side signed mean |
|---|---:|---:|
| P8A | −0.098 | +0.020 |
| E2B | −0.317 | −0.137 |
| BUNDLE_step500 | **+0.052** (sign-flipped) | +0.089 |
| BUNDLE_step3750 | −0.100 | −0.064 |
| BUNDLE_step4000 | −0.100 | −0.042 |
| PAIRRANK_step500 | +0.001 | +0.099 |
| PAIRRANK_step6000 | −0.241 | −0.077 |
| PAIRRANK_step6750 | −0.316 | −0.032 |

### Direct observations
- BUNDLE_step500 has signed mean +0.052 on real-side (sign-FLIPPED vs P8A's −0.098). All other P1 ckpts and E2B have negative signed real-side mean (same direction as P8A).
- The earlier F3 "abs |r|" amplification reading mixed two directions of relationship. BUNDLE_step3750/4000 amplify in P8A's same direction (more negative). BUNDLE_step500 amplifies via sign-flip (positive direction).
- The signed direction varies per suite: teams_fake_all_dev is positive (+0.65 P8A, decreasing across P1 to +0.12-0.49); teams_fake_all_lockbox is negative (−0.44 P8A, mostly preserved or reduced across P1).
- A "shortcut amplification" claim should be qualified by direction; "amplification" in absolute terms can include direction-flips that are structurally different.

---

## 6. Counter-theory probe — per-identity regression magnitudes

**Question answered**: Initial framing: "GroupDRO has a balloon effect on non-target chronic identities" (BUNDLE arm should regress more than PAIRRANK arm on non-targets). Does the data support this claim?

**Method**: Computed Δ FPR (P1 ckpt − P8A) at calibrated τ for every (base_identity × P1 ckpt) cell on `teams_real_all_dev`. 13 base_identities had n ≥ 29 frames. Compared per-identity mean |Δ| between BUNDLE arm (3 ckpts) and PAIRRANK arm (3 ckpts).

**Output**: `counter_theory_per_identity_deltas.csv`.

### Top 13 base_identities × Δ FPR vs P8A at calibrated τ

(Positive Δ = regression. Negative Δ = improvement.)

| identity | n | P8A FPR | BUNDLE Δ (500/3750/4000) | PAIRRANK Δ (500/6000/6750) |
|---|---:|---:|---|---|
| Test_Cam | 1280 | 0.006 | −0.006 / −0.006 / −0.005 | −0.005 / −0.005 / −0.005 |
| **PC_Generator** (chronic-target) | 835 | 0.241 | **−0.241 / −0.238 / −0.229** | −0.214 / −0.187 / −0.189 |
| Md_noyn_Sharker | 682 | 0.003 | −0.003 / −0.003 / −0.003 | −0.003 / −0.003 / −0.003 |
| **bla_bla_chow** (chronic) | 491 | 0.063 | +0.100 / +0.037 / +0.037 | +0.063 / +0.049 / +0.049 |
| Xiang_Xiang2_Feng | 403 | 0.007 | +0.000 / −0.007 / −0.005 | +0.000 / −0.005 / −0.005 |
| dor | 269 | 0.000 | +0.026 / +0.007 / +0.007 | +0.000 / +0.004 / +0.019 |
| Cam_Test | 166 | 0.000 | +0.000 / +0.000 / +0.000 | +0.000 / +0.000 / +0.000 |
| xiang | 159 | 0.000 | +0.050 / +0.006 / +0.019 | +0.094 / +0.019 / +0.044 |
| **Roy_D** (regression) | 130 | 0.292 | **+0.638 / +0.577 / +0.562** | +0.492 / +0.523 / +0.485 |
| **Q** (chronic-target) | 54 | 0.889 | **−0.722 / −0.722 / −0.722** | −0.704 / −0.722 / −0.704 |
| orel | 35 | 0.000 | 0 / 0 / 0 | 0 / 0 / 0 |
| dor_shkedi | 31 | 0.000 | 0 / 0 / 0 | 0 / 0 / 0 |
| ilan | 29 | 0.000 | 0 / 0 / 0 | 0 / 0 / 0 |

### Aggregate test on non-target identities (n_id = 9, excluding chronic targets)

| arm | mean |Δ| | mean signed Δ | |Δ|>5% cells (regr / improv / total) |
|---|---:|---:|---|
| BUNDLE | 0.0058 | +0.0029 | 1 / 0 / 27 |
| PAIRRANK | 0.0080 | +0.0053 | 1 / 0 / 27 |

### Aggregate test on chronic-target identities (n_id = 4)

| arm | mean signed Δ |
|---|---:|
| BUNDLE | **−0.077** (improvement) |
| PAIRRANK | **−0.088** (improvement) |

### Counter-test on non-target identities

- BUNDLE > PAIRRANK in mean |Δ| on identity: 3 of 9 (33.3%)
- Mean (BUNDLE |Δ| − PAIRRANK |Δ|) per identity: −0.0022
- Wilcoxon signed-rank test: stat=4.00, **p=0.8750** (no significant difference)

### Direct observations
- Roy_D: BUNDLE Δ = +0.638/+0.577/+0.562; PAIRRANK Δ = +0.492/+0.523/+0.485. Both arms regress on Roy_D with similar magnitude (BUNDLE slightly higher).
- bla_bla_chow: both arms regress similarly (BUNDLE +0.100/+0.037/+0.037 vs PAIRRANK +0.063/+0.049/+0.049).
- PC_Generator and Q: both arms improve substantially. BUNDLE marginally better on PC_Generator (−0.24 vs −0.20).
- Wilcoxon signed-rank test on non-target |Δ| between arms: p=0.875, no statistically significant difference.

### Implication for the original "GroupDRO balloon" hypothesis

The hypothesis that BUNDLE-arm-specific GroupDRO causes greater non-target regression than PAIRRANK-only is NOT supported by this data. Both arms regress similarly on non-targets. The shared mechanism (pair_rank loss class + post-`2feea58` FT codepath + anchor_aware penalty + stability loss) is the most likely root cause of regressions like Roy_D, not the GroupDRO term specifically.

---

## 7. Bug fixes applied this session

### `trainer/trainer.py:1727` — W&B logging gap fix

Original:
```python
losses = self.calculate_group_dro_loss(data_dict, per_sample_loss)
```

Fixed: after the GroupDRO call, the loop now updates `losses` with the diagnostic keys from `per_sample_losses_dict` (excluding `'overall'`):
```python
losses = self.calculate_group_dro_loss(data_dict, per_sample_loss)
for k, v in per_sample_losses_dict.items():
    if k != 'overall':
        losses[k] = v
```

This preserves the diagnostic scalars (`pair_rank_loss`, `cls_loss`, `corr_penalty_loss`, `feat_norm_loss`, `quality_domain_loss`, `keepsv_loss`, `reg_loss`, etc.) that were silently dropped. The W&B log loop at `trainer.py:1847-1857` already handles both scalar and multi-element tensors correctly. AST + bytecode compile verified.

### `analysis/p1_pe_eval_2026-05-07/phase_d/run_chronic_filter.py` — chronic-id matching fix

Original used `extract_base_identity` (regex strips `__s\d+`/`__seq\d+`) BEFORE the chronic-check, which collapsed `PC_Generator__s22/s45` and `Q__s6` to `PC_Generator` and `Q`, breaking the chronic-id list match. Fix: introduced `video_matches_cid(video_id, cid)` using prefix-on-raw-video_id (case-insensitive), used in both aggregate and per-identity logic.

Re-running the fixed script produces the same numbers as the inline-fixed version: P8A pc_fpr at calibrated τ = 0.629 (vs 0.000 in buggy version), BUNDLE_step500 pc_fpr = 0.000 (Δ +0.629). The original CSVs at `phase_d/{chronic6_aggregate_fpr,per_identity_fpr,pc_generator_cluster_fpr}.csv` are now overwritten with correct numbers.

`_FIXED.csv` files from the inline-fix step remain on disk as forensic provenance.

---

## 8. Companion artifacts

| file | rows | description |
|---|---:|---|
| `roy_d_regression/roy_d_with_axes.csv` | 130 | per-frame scores + computed axes for roy_d |
| `roy_d_regression/roy_d_per_frame_scores.csv` | 1040 | 8 ckpts × 130 frames; raw scores |
| `f2_pair_rank/relaxed_thresholds.csv` | (computed inline, in memory) | F2(a) at <0.5/<0.7/<0.9 missed-fake filters |
| `f2_pair_rank/per_lane_per_ckpt_lift.csv` | 21 | F2(a) at <0.5 (yaml's strict reading) |
| `p1_at_tau05_for_pd_compare.csv` | 72 | P1 at τ=0.5 for PD comparison |
| `lockbox_roc_curves.csv` | 688 | 8 ckpts × 86 τ-points; lockbox FPR + recall |
| `correlations.csv` | 120 | F3 audit signed-r per (ckpt × suite × axis) |
| `counter_theory_per_identity_deltas.csv` | 25 | top-25 base_identities × Δ FPR per ckpt |
| `phase_d/{chronic6_aggregate_fpr,per_identity_fpr,pc_generator_cluster_fpr}.csv` | various | now-correct chronic-6 aggregates (fixed 2026-05-07) |
| `phase_d/{chronic6_aggregate_fpr_FIXED,per_identity_fpr_FIXED}.csv` | various | inline-fix forensic record |

## 9. Cross-references

- `DEEP_DIVE_FACTS_2026-05-07.md` — main consolidating record (some contents from §6, §7 superseded by this doc's bug-fix notes; numbers consistent).
- `RESULTS_F1_F5_FACTS_2026-05-07.md` — original F1-F5 verdicts.
- `AGENT_PROPOSAL_2026-05-07.md` — agent's staked view (separate doc; speculative content lives there).
