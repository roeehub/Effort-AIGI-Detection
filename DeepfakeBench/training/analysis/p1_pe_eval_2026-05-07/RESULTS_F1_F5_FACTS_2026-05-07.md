# P1 Close-Criterion F1-F5 — Computed Verdicts, 2026-05-07

**Status**: F4 + F5 + F3 (partial) + recall-floor forensic complete. F2(a) (pair-rank lift on missed fakes) NOT yet computed — requires pair-coverage convention from `analysis/pair_coverage_audit_2026-05-06/`.

This doc supersedes the earlier "F5 fail" reading in ANALYSIS_DEPRECATED_2026-05-07.md §6 — the original `phase_d/run_chronic_filter.py` had a regex bug that caused PC_Generator and Q chronic identities to be dropped from the chronic-6 aggregate. With prefix matching on raw video_id, F5 PASSES for all BUNDLE ckpts.

---

## F1 — lockbox fake recall ≥ 90% at FPR ≤ 10%

**Verdict: FAIL for every ckpt.**

| ckpt | lockbox_fake_recall | lockbox_real_fpr | F1 |
|---|---:|---:|:---:|
| P8A_REFERENCE_STEP5000 | 0.387 | 0.018 | FAIL |
| E2B_TOP_N_STEP3200 | 0.629 | 0.024 | FAIL |
| P1_BUNDLE_PERIODIC_STEP500 | **0.826** | 0.033 | FAIL (closest) |
| P1_BUNDLE_TOP_N_STEP3750 | 0.154 | 0.013 | FAIL |
| P1_BUNDLE_TOP_N_STEP4000 | 0.261 | 0.021 | FAIL |
| P1_PAIRRANK_PERIODIC_STEP500 | 0.708 | 0.018 | FAIL (winner) |
| P1_PAIRRANK_TOP_N_STEP6000 | 0.324 | 0.020 | FAIL |
| P1_PAIRRANK_TOP_N_STEP6750 | 0.486 | 0.026 | FAIL |

Best: BUNDLE_PERIODIC_STEP500 at 82.6% lockbox recall (still 7.4pp short of bar).

---

## F2(a) — pair-rank lift on previously missed fakes ≥ 30% on ≥ 2 of 5 paired lanes

**Verdict: NOT COMPUTED.**

The script needs the paired training-lane convention from `analysis/pair_coverage_audit_2026-05-06/`. The frames_report.csv schema does not directly expose `(sample_id, frame_idx)` — this is a multi-step join. Deferred.

---

## F2(b) — worst-group recall lift ≥ 20% on chronic / dor-drift cohorts

**Partial data via F5 chronic-6 numbers below + dor probe on 50-frame `teams_real_dor_dev`:**

`teams_real_dor_dev` FPR at calibrated τ (from §2.2 of RESULTS_FACTS_2026-05-07.md):

| ckpt | teams_real_dor_dev FPR (n=50) |
|---|---:|
| P8A | 0.080 (4/50) |
| E2B | 0.120 |
| BUNDLE_step500 | 0.040 |
| BUNDLE_step3750 | 0.060 |
| BUNDLE_step4000 | 0.080 |
| PAIRRANK_step500 | **0.160** (worst) |
| PAIRRANK_step6000 | 0.080 |
| PAIRRANK_step6750 | 0.040 (best, tied) |

dor_dev is FPR-only (no fakes in this suite). "Worst-group recall lift" requires per-group recall on fake suites — the Phase A scorecard doesn't expose this granularity directly. Deferred.

---

## F3 — no untargeted axis amplifies +50%

**Verdict: PARTIAL. P1 PASSES on `sharpness_laplacian` and `min_dim`, FAILS on `face_area_fraction` for all 6 ckpts on the real side.**

Source: `run_audit.py` (3 axes) + `min_dim_correlations.csv` (extension).

### Untargeted-axis |Pearson r| means on REAL suites — Δ vs P8A

| ckpt | sharpness | face_area_fraction | min_dim |
|---|---:|---:|---:|
| (P8A baseline real |r|) | (0.188) | (0.117) | (0.491) |
| E2B | +96.4% AMP | +171.0% AMP | −69.1% |
| P1_BUNDLE_step500 | −38.5% | **+266.1% AMP** | −26.5% |
| P1_BUNDLE_step3750 | +1.0% | **+166.6% AMP** | −61.3% |
| P1_BUNDLE_step4000 | +4.5% | **+138.4% AMP** | −59.7% |
| P1_PAIRRANK_step500 | −33.3% | **+165.4% AMP** | −28.4% |
| P1_PAIRRANK_step6000 | +20.8% | **+105.8% AMP** | −73.0% |
| P1_PAIRRANK_step6750 | +33.7% | **+170.1% AMP** | −70.3% |

`color_b_dev` was not measured (handoff suggested adding it; requires reading source frames for B-channel std — deferred).

Read: P1 substantially **reduced** `min_dim` coupling on reals (the canonical resolution shortcut, P8A baseline 0.491 — the strongest among measured axes for P8A on reals) by 26-73%. Simultaneously, `face_area_fraction` coupling **grew** from a low P8A base (0.117) to 0.24-0.43 across P1 ckpts — a relative amplification >100%. The absolute magnitude of face_area coupling on reals after P1 is still lower than P8A's min_dim baseline; the model traded a stronger high-magnitude shortcut for a stronger relative-but-still-mid-magnitude one.

The F3 close criterion is "no untargeted axis amplifies +50%". By the strict letter, FAIL. By the magnitude-and-direction reading, the picture is "shifted shortcut profile, net reduced resolution coupling".

### Fake-side (recall) — Δ |Pearson r| vs P8A

All P1 ckpts REDUCE coupling on all three axes for fake recall (good). Largest reductions on `face_area_fraction` (down 40-68%) and `min_dim` (down 19-54%).

---

## F4 — HDTF cross-substrate FPR ≤ 5% at calibrated τ

**Verdict: PASS for all 8 ckpts.**

Source: `hdtf/scorecard_calibrated_tau.csv` — applied Phase A's per-ckpt τ to Phase C's per-frame HDTF reports (Phase C's promotion_contract step never ran due to job failure, but the per-frame data was complete).

| ckpt | τ | max HDTF real FPR | worst suite | F4 |
|---|---:|---:|---|:---:|
| P8A_REFERENCE_STEP5000 | 0.9156 | 0.0088 | proper_real_teams_lockbox | PASS |
| E2B_TOP_N_STEP3200 | 0.7108 | 0.0016 | proper_real_clean_dev | PASS |
| P1_BUNDLE_PERIODIC_STEP500 | 0.9919 | 0.0141 | proper_real_clean_lockbox | PASS |
| P1_BUNDLE_TOP_N_STEP3750 | 0.9994 | 0.0010 | proper_real_clean_dev | PASS |
| P1_BUNDLE_TOP_N_STEP4000 | 0.9989 | 0.0013 | proper_real_clean_lockbox | PASS |
| P1_PAIRRANK_PERIODIC_STEP500 | 0.7677 | 0.0069 | proper_real_clean_dev | PASS |
| P1_PAIRRANK_TOP_N_STEP6000 | 0.9878 | 0.0020 | proper_real_clean_lockbox | PASS |
| P1_PAIRRANK_TOP_N_STEP6750 | 0.9900 | 0.0052 | proper_real_clean_lockbox | PASS |

Note: at τ=0.5 (uncalibrated), BUNDLE_step500 had 25% FPR on `proper_real_clean_dev`. At its calibrated τ=0.9919 it drops to 0.67% — comfortably under the 5% F4 bar. **The earlier "BUNDLE_step500 disaster on HDTF" framing was an artifact of reading at uncalibrated τ.**

### HDTF fake recall at calibrated τ — diagnostic

The `*_clean_*` HDTF suites are easy for everyone (94-98% recall for non-step500 ckpts). The `*_teams_*` suites are hard:
- P8A: 83-95% on every HDTF suite (best cross-substrate)
- E2B: collapses on teams subtypes — 12.4% on fake_teams_all_dev, 6.1% on visomaster_enhanced_teams
- P1 BUNDLE step3750/4000: ≈97% clean / 17-40% teams (bimodal)
- P1 PAIRRANK: similar bimodal pattern

The HDTF teams-subtype gap is real for everyone except P8A.

---

## F5 — chronic-FP `pc_generator` cluster failure rate drops ≥ 0.10 absolute (BUNDLE only)

**Verdict: PASS for all 3 BUNDLE ckpts.** Substantially.

Source: `phase_d/chronic6_aggregate_fpr_FIXED.csv` (after fixing the regex bug — original script stripped `__s22/__s45/__s2` session tokens before matching, causing PC_Generator and Q chronic frames to be excluded from the aggregate).

### PC_Generator cluster (s22 + s45) FPR @ calibrated τ

| ckpt | n_pc | pc_fpr | Δ vs P8A | F5 (BUNDLE only) |
|---|---:|---:|---:|:---:|
| P8A_REFERENCE_STEP5000 | 318 | **0.629** | (baseline) | n/a |
| E2B_TOP_N_STEP3200 | 318 | 0.107 | +0.522 | n/a |
| P1_BUNDLE_PERIODIC_STEP500 | 318 | **0.000** | **+0.629** | PASS |
| P1_BUNDLE_TOP_N_STEP3750 | 318 | 0.006 | +0.623 | PASS |
| P1_BUNDLE_TOP_N_STEP4000 | 318 | 0.031 | +0.598 | PASS |
| P1_PAIRRANK_PERIODIC_STEP500 | 318 | 0.066 | +0.563 | n/a |
| P1_PAIRRANK_TOP_N_STEP6000 | 318 | 0.142 | +0.487 | n/a |
| P1_PAIRRANK_TOP_N_STEP6750 | 318 | 0.135 | +0.494 | n/a |

P8A's pc_fpr = 0.629 means **63% of PC_Generator chronic frames are false-positives** at P8A's calibrated τ. BUNDLE drops this to **0-3%**. GroupDRO's `chronic_flag` term did exactly what it was designed to do on this cluster.

### Per-identity breakdown @ calibrated τ — chronic-6 detail

| identity | n | P8A | E2B | B500 | B3750 | B4000 | P500 | P6000 | P6750 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PC_Generator__s22 | 227 | **0.788** | 0.031 | **0.000** | 0.009 | 0.040 | 0.092 | 0.163 | 0.123 |
| PC_Generator__s45 | 91 | 0.231 | 0.297 | 0.000 | 0.000 | 0.011 | 0.000 | 0.088 | 0.165 |
| Q__s6 | 54 | **0.889** | 0.278 | 0.167 | 0.167 | 0.167 | 0.185 | 0.167 | 0.185 |
| bla_bla_chow | 491 | 0.063 | 0.295 | 0.163 | 0.100 | 0.100 | 0.126 | 0.112 | 0.112 |
| bla_bla_chow__s2 | 180 | 0.111 | 0.283 | **0.317** | 0.072 | 0.078 | 0.189 | 0.133 | 0.106 |
| roy_d | 130 | 0.292 | 0.154 | **0.931** | **0.869** | **0.854** | **0.785** | **0.815** | **0.777** |

Mixed signal at the per-identity level:
- **PC_Generator__s22** (the worst-P8A identity at 79%) — fixed entirely by P1 (down to 0-12%).
- **Q__s6** (P8A 89%) — substantially improved by P1 (down to 17-19%).
- **roy_d** (P8A 29%) — **CATASTROPHICALLY WORSE** for P1 (every P1 ckpt is 78-93%; was 29% under P8A).
- **bla_bla_chow** (P8A 6%) — got worse (10-18% across P1).
- **bla_bla_chow__s2** (P8A 11%) — BUNDLE_step500 went to 32%, others mostly OK.

So GroupDRO didn't uniformly help chronic offenders. It traded some fixed identities for one big regression (roy_d) and minor regressions on bla_bla_chow.

### Aggregate chronic-6 FPR @ calibrated τ

| ckpt | n_chronic | chronic_fpr | Δ vs P8A |
|---|---:|---:|---:|
| P8A | 993 | 0.319 | (baseline) |
| E2B | 993 | 0.215 | +0.103 |
| BUNDLE_step500 | 993 | 0.212 | +0.108 |
| BUNDLE_step3750 | 993 | 0.174 | +0.145 |
| BUNDLE_step4000 | 993 | 0.180 | +0.139 |
| PAIRRANK_step500 | 993 | 0.196 | +0.123 |
| PAIRRANK_step6000 | 993 | 0.217 | +0.103 |
| PAIRRANK_step6750 | 993 | 0.211 | +0.108 |

Every P1 ckpt reduces aggregate chronic-6 FPR by ≥10pp (also clears a "softer" F5-like gate even for PAIRRANK arm).

---

## Threshold-grid forensic — why BUNDLE_step3750/4000 selected τ=0.999x

**Verdict: NOT a contract bug. ROC-curve degeneracy on those ckpts.**

Per `arena/score_teams_promotion_contract.py:460-510` (`_threshold_sort_key`): the recall floor is TIERED, not GATED. Tier 0 = budget OK + recall ≥ floor; tier 1 = budget OK + recall < floor; tier 2 = budget violated. Within each tier, sort by (macro_recall desc, threshold desc, primary_fpr asc).

Forensic on BUNDLE_step3750 (5339 grid points):

- 923 grid points satisfy budget (FPR ≤ 0.07 AND stress ≤ 0.10).
- **All 923 budget-OK points have τ ≥ 0.9994**. Below τ=0.9994, FPR exceeds 0.07.
- **0 grid points have macro_recall ≥ 0.30** anywhere in the grid (tier 0 is empty).
- The model's ROC curve is shaped such that lowering τ below 0.9994 spikes FPR faster than recall can grow into the floor band.

So tier 1 is the only available tier. Within tier 1, the sort picks (macro_recall=0.192, threshold=0.9994, FPR=0.053) — that's the contract's selection.

Same shape for BUNDLE_step4000: 0 tier-0 candidates, best tier-1 macro_recall = 0.285.

**The recall-floor flag worked correctly.** Tier 1 ckpts get demoted in the cross-checkpoint ranking — that's why PAIRRANK_PERIODIC_STEP500 (which has many tier-0 candidates at lower τ) is rank-1 instead of any BUNDLE ckpt. The τ=0.999x for BUNDLE step3750/4000 is the legitimate within-tier-1 best, not a τ-tail collapse bug.

The structural read: BUNDLE_step3750/4000 have a **degenerate ROC at the deployment-τ region** — they can't trade FPR for recall smoothly. This is a property of those ckpts, not the contract.

---

## Consolidated F1-F5 verdict per ckpt

| ckpt | F1 | F2(a) | F2(b) | F3 | F4 | F5 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| P8A_REFERENCE_STEP5000 | FAIL | n/a | n/a | (baseline) | PASS | n/a |
| E2B_TOP_N_STEP3200 | FAIL | n/a | n/a | FAIL on face_area | PASS | n/a |
| P1_BUNDLE_PERIODIC_STEP500 | FAIL (82.6%) | TBD | partial | FAIL on face_area | PASS | **PASS** |
| P1_BUNDLE_TOP_N_STEP3750 | FAIL | TBD | partial | FAIL on face_area | PASS | **PASS** |
| P1_BUNDLE_TOP_N_STEP4000 | FAIL | TBD | partial | FAIL on face_area | PASS | **PASS** |
| P1_PAIRRANK_PERIODIC_STEP500 | FAIL (70.8%) | TBD | partial | FAIL on face_area | PASS | n/a |
| P1_PAIRRANK_TOP_N_STEP6000 | FAIL | TBD | partial | FAIL on face_area | PASS | n/a |
| P1_PAIRRANK_TOP_N_STEP6750 | FAIL | TBD | partial | FAIL on face_area | PASS | n/a |

**F1**: not cleared by any ckpt. PAIRRANK_step500 is the contract winner (rank 1) at 70.8% lockbox recall.
**F3**: face_area_fraction amplifies >50% on real side for all P1 ckpts; min_dim and sharpness do NOT amplify.
**F4**: passes universally at calibrated τ.
**F5**: BUNDLE arm passes by very large margin on PC_Generator cluster (drop 60+pp absolute).

The 4-of-4 (BUNDLE-only) gate read: F1 fail + F3 fail (face_area only — P1 traded min_dim shortcut for face_area shortcut) + F4 pass + F5 pass = 2 of 4 pass. F2 still pending.

---

## Updated open questions

1. **F2(a)** — pair-rank lift on missed fakes by paired lane. Needs pair_coverage_audit convention.
2. **`color_b_dev` axis** for F3 — needs frame-level B-channel std computation (memory `project_dor_drift_named_axes_2026-05-06.md` cites this as a load-bearing dor-drift axis). Deferred.
3. **roy_d regression in BUNDLE arm** — chronic identity P8A handled at 29% goes to 78-93% in P1. Mechanism unknown. New open loop.
4. **BUNDLE_step3750/4000 ROC degeneracy** — the model can't trade FPR for recall smoothly at deployment τ. Could be related to BUNDLE's IQ-axis decoupling at step500 (per `axis_decoupling_trajectory.csv`); plausibly the model's score distribution becomes very flat at deployment-relevant magnitudes. Worth a follow-up CPU probe.

---

## Cross-reference

- Companion: `RESULTS_FACTS_2026-05-07.md` (raw scorecard data) and `ANALYSIS_DEPRECATED_2026-05-07.md` (structured analysis — note: the F5-fail framing in §6 is now superseded by this doc).
- Phase D fixed CSV: `phase_d/chronic6_aggregate_fpr_FIXED.csv`, `phase_d/per_identity_fpr_FIXED.csv`.
- Phase D buggy CSV (do not use): `phase_d/chronic6_aggregate_fpr.csv`, `phase_d/per_identity_fpr.csv`, `phase_d/pc_generator_cluster_fpr.csv` (left in place for forensic; original `run_chronic_filter.py` is buggy on the regex strip and should be patched before reuse).
- F4 calibrated-τ CSV: `hdtf/scorecard_calibrated_tau.csv`.
- F3 audit output: `abs_pearson_summary.csv`, `correlations.csv`, `min_dim_correlations.csv`.
- Threshold grid: `scorecard/threshold_grid.csv`.
