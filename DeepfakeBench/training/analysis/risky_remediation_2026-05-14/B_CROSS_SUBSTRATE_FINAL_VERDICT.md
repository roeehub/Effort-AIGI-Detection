# Cross-Substrate Validation of Blend Remediation — FINAL VERDICT

**Date:** 2026-05-14
**Pool:** 13,852 frames across 13 cohorts; 7,331 pass G2(200); 6,940 in two-label production pools
**Models:** T5C step3500 (`jrlldtem`, current ship), P8A step5000 (reference)

---

## TL;DR

After three pool sizes (260 stress → 4035 gate=pass → 13,852 cross-substrate), each progressively more honest:

- **T5C + blend@0.50: STATISTICALLY SIGNIFICANT NEGATIVE on all 3 production cohorts** (sign-test p = 1.0). All bootstrap 95% CIs exclude 0 on the negative side.
- **P8A + blend@0.50: positive on 2 of 3 cohorts** (sign-test p = 0.125, not significant). Wins: teams_dev +0.001, teams_lockbox +0.016. Neutral on dor_cross.

**Conclusion:** Universal blend@0.50 is **NOT a deployable improvement for the current ship candidate (T5C)**. The earlier positive findings were artifacts of small-sample, FP-rich slices.

**Recommended action**: ship T5C step3500 + τ=0.49 + G1+G2(200) **as already specced in `A_SHIP_SPEC_T5C_2026-05-14.md`**. Do not add blend preprocessing.

---

## Methodology summary

Three sequentially-more-honest pools:

| Run | Pool | Result |
|-----|------|--------|
| Stress (260 frames) | hand-selected FPs + clean reals + TP fakes | T5C "+0.01 marginal", misleading |
| Full gate=pass (4035) | only `gate_status='pass'` per lockbox parquet | T5C +0.006 lockbox, P8A +0.05 lockbox — promising-looking |
| Cross-substrate G2-filtered (7331) | all frames passing computed `min(W,H) ≥ 200` | T5C significantly negative on every cohort |

**The third pool is the production-aligned one** because:
- G2(200) is the actual production gate, not the parquet's `gate_status` label
- It covers 13 cohorts including live_prod, dor_cross, may5 — not just lockbox slice
- Includes 992 lockbox reals that weren't in the parquet (but pass G2)

---

## Bootstrap 95% CIs on ΔAUC = AUC(blend) − AUC(orig)

| pool | ckpt | n | n_real | n_fake | AUC_orig | ΔAUC | 95% CI | CI excludes 0 |
|------|------|---|--------|--------|----------|------|--------|---------------|
| **teams_dev** | T5C | 3630 | 2258 | 1372 | 0.9802 | **−0.0016** | [−0.0024, −0.0008] | YES (neg) |
| **teams_dev** | P8A | 3630 | 2258 | 1372 | 0.9925 | **+0.0009** | [+0.0006, +0.0013] | YES (pos) |
| **teams_lockbox** | T5C | 1686 | 1317 | 369 | 0.9085 | **−0.0287** | [−0.0380, −0.0196] | YES (neg) |
| **teams_lockbox** | P8A | 1686 | 1317 | 369 | 0.9292 | **+0.0156** | [+0.0113, +0.0204] | YES (pos) |
| **dor_cross** | T5C | 1624 | 558 | 1066 | 0.9417 | **−0.0189** | [−0.0222, −0.0159] | YES (neg) |
| **dor_cross** | P8A | 1624 | 558 | 1066 | 0.9635 | +0.0001 | [−0.0016, +0.0021] | NO |

Bootstrap: 2000 iterations per pool, paired resampling.

### Sign test across 3 production cohorts
- **T5C: 0/3 positive — sign-test p = 1.0000** (definitively negative across cohorts)
- **P8A: 3/3 positive — sign-test p = 0.1250** (consistent direction; not enough cohorts for α=0.05 significance)

### Per-frame paired Wilcoxon
| ckpt | label | n | median Δ | H1 alternative | p-value |
|------|-------|---|----------|----------------|---------|
| T5C | REAL | 4133 | +0.0057 | Δ < 0 | 1.0 (no evidence) |
| T5C | FAKE | 2807 | −0.0012 | Δ > 0 | 1.0 (no evidence) |
| P8A | REAL | 4133 | +0.0000 | Δ < 0 | 1.0 (no evidence) |
| P8A | FAKE | 2807 | +0.0001 | Δ > 0 | 1.7e-193 (strong, tiny effect) |

P8A's per-frame fake-side shift is statistically real but median 0.0001 means the magnitude is trivial.

---

## Per-cohort score-direction analysis (single-label cohorts)

Frame-level "good direction" (% of frames moving the correct way under blend):

| cohort | n | type | T5C good_dir | P8A good_dir |
|--------|---|------|--------------|--------------|
| dor_evening | 324 | real-only | 10.2% | 23.2% |
| dor_morning | 244 | real-only | 24.6% | 45.1% |
| live_reals_teams_prod | 677 | real-only | 43.4% | 38.3% |
| team_sanity_may5 | 210 | real-only | 10.5% | 12.9% |
| teams_real_all_dev | 4295 | real-only | 27.7% | 45.5% |
| teams_real_all_lockbox | 1418 | real-only | 37.6% | 52.3% |
| teams_real_dor_dev | 50 | real-only | 10.0% | 26.0% |
| dor_fake_local | 605 | fake-only | 61.6% | **95.4%** |
| live_fakes_teams_prod | 1675 | fake-only | 66.2% | **84.8%** |
| teams_fake_all_dev | 1620 | fake-only | 34.9% | 73.3% |
| teams_fake_all_lockbox | 425 | fake-only | 33.4% | 75.1% |
| visomaster_v2_dor | 2073 | fake-only | 23.8% | 72.1% |

**Pattern:**
- T5C real-only cohorts: 10–43% good direction (worse than random across all 7)
- P8A real-only cohorts: 13–52% good direction (mostly worse than random; only teams_lockbox is at coin flip)
- T5C fake-only cohorts: 24–66% good direction (mixed; HURTS visomaster_v2_dor)
- **P8A fake-only cohorts: 72–95% good direction** (strongly consistent fake-detection booster)

**Interpretation:** P8A blend is essentially a "fake-detection-booster" — it strongly lifts fake scores. It does NOT actually reduce false-positive reals; reals are mostly unmoved.

T5C blend hurts because:
1. Real-side scores systematically rise (median Δ = +0.006 on G2-pass reals)
2. Fake-side scores are unchanged or slightly drop (median Δ = -0.001)
3. Net AUC erosion is consistent

---

## Design A — multi-axis selective remediation

Tested 7 lap_var thresholds × 2 directions (apply blend when `lap < T` vs `lap ≥ T`) on combined pool:

**T5C** — every selective variant matched or trailed orig AUC. Best selective: q=0.90 INV (blend only frames with `lap ≥ 344`, top 10%) gives ΔAUC = +0.0003 vs orig — essentially zero. **Selective offers no win.**

**P8A** — selective variants matched universal blend at best (ΔAUC vs universal: −0.0080 to +0.0001). **Universal blend is the dominant strategy; selective adds nothing.**

---

## Design C — soft gate via cheap risk score

Trained logistic regression on (`lap_var`, `luma`, `lab_a_dev`, `lab_b_dev`, `edge_density`) predicting `model_err_at_tau=0.49`. Train pool: `teams_dev`. Test pool: `lockbox + dor_cross` (3310 G2-pass frames).

Risk model train-AUC: T5C 0.786, P8A 0.708 (modest predictive power).

| strategy | T5C test AUC | T5C Δ | P8A test AUC | P8A Δ |
|----------|--------------|-------|--------------|-------|
| orig (no change) | 0.9079 | 0 | 0.9513 | 0 |
| universal blend@0.50 | 0.8888 | **−0.019** | 0.9575 | **+0.006** |
| orig + abstain top 5% risky | 0.9088 | +0.001 | 0.9493 | −0.002 |
| orig + abstain top 10% risky | 0.8994 | −0.009 | 0.9494 | −0.002 |
| blend + abstain top 5% risky | 0.8858 | −0.022 | 0.9556 | +0.004 |
| blend + abstain top 10% risky | 0.8750 | −0.033 | 0.9562 | +0.005 |
| hybrid: blend top 25% risky | 0.9069 | −0.001 | 0.9532 | +0.002 |
| hybrid: blend top 10% risky | 0.9096 | **+0.002** | 0.9515 | 0.000 |

**For T5C**: best Design C strategy ("hybrid: blend top 10% riskiest") gives ΔAUC +0.002 — a trivial gain, not statistically significant given the cohort-level CIs. Effectively no clean lever for T5C.

**For P8A**: universal blend at +0.006 remains the best strategy. Conditional designs don't improve over universal.

---

## What went wrong with the earlier reads

| read | claim | what was actually true |
|------|-------|------------------------|
| Stress pool (260) | "+0.01 marginal" | Pool was FP-heavy, AUC sensitive to which FPs got moved |
| Lockbox @ gate=pass (694) | "T5C +0.006 / P8A +0.050" | Parquet-annotated `pass` reals are only 325 of 1317 G2-passing lockbox reals; the slice happened to favor blend |
| Cross-substrate @ G2(7331) | "T5C significantly negative everywhere" | Honest production-aligned signal |

The methodological lesson: **on any large model (AUC > 0.90), small-pool inference-trick studies routinely produce statistically-significant artifacts of slice selection.** The right validation is multi-cohort cross-substrate with bootstrap CIs.

---

## Recommendations

### For deployment (right now)
**Ship T5C step3500 with τ=0.49 + G1+G2(200) gate as already specced in `A_SHIP_SPEC_T5C_2026-05-14.md`.** Do NOT add the blend preprocessing.

### For T5C improvement (future packets)
The blend doesn't work because T5C's GRL training already partially neutralized the sharpness shortcut. Future packets should NOT try to lean back on the shortcut (the gradient is gone). Alternative directions:
- Train-side: continue the per-axis GRL approach with new axes (color_b_dev, edge_density)
- Architecture-side: LoRA on later layers (already speced — `R13_LORA_L10_L11`)
- Data-side: cross-substrate identity-fresh data for the chronic-FP identities

### For P8A (academic interest)
P8A + blend@0.50 lifts lockbox AUC by +0.016 (CI [+0.011, +0.020]). Operating-point gain at 5% FPR is +4pp recall. Not enough to displace T5C — but if P8A were ever revisited as a deployment candidate, blend would stack.

### For methodology
Add a multi-cohort cross-substrate validation step to any future inference-side or threshold-side claim. Never accept an inference trick on stress-pool or single-cohort data — always run sign-test across ≥3 cohorts and bootstrap CIs.

---

## Artifacts

All in `analysis/risky_remediation_2026-05-14/`:

| file | content |
|------|---------|
| `A_SHIP_SPEC_T5C_2026-05-14.md` | The ship spec for PM (current authoritative) |
| `B_CROSS_SUBSTRATE_FINAL_VERDICT.md` | This document |
| `FINDINGS.md` | Mid-flight stress-pool findings (now superseded) |
| `FINAL_VERDICT.md` | Mid-flight gate=pass-only findings (now superseded) |
| `outputs/all_cohorts_scored.csv` | 13,852 frames × 4 conditions × IQ features |
| `outputs/g2_pass_pool.csv` | 7,331 G2-passing frames (production-eligible) |
| `outputs/per_cohort_analysis.csv` | Per-suite Δscore + frac>0.5 |
| `outputs/single_label_cohorts.csv` | Direction-aware per-cohort shifts |
| `outputs/g2_filtered_per_pool.csv` | G2-filtered per-pool AUC + Wilcoxon p-values |
| `outputs/design_a_per_pool.csv` | Selective remediation per-pool sweep |
| `outputs/design_c_soft_gate_*.csv` | Soft gate strategies per ckpt |
| `outputs/bootstrap_auc_deltas.csv` | 2000-iter bootstrap 95% CIs |
| `run_all_cohorts.py` | Inference runner |
| `apply_g2_and_analyze.py` | G2 filter + per-pool stats |
| `design_a_selective.py` | Selective remediation experiments |
| `design_c_soft_gate.py` | Risk-score-based abstain/hybrid |
| `statistical_validation.py` | Bootstrap CIs + sign tests + Wilcoxon |

## Reproduce

```bash
# Sequential — ~45 min total CPU on M2 mac, $0
python analysis/risky_remediation_2026-05-14/run_all_cohorts.py     # ~30 min
python analysis/risky_remediation_2026-05-14/apply_g2_and_analyze.py # ~30 s
python analysis/risky_remediation_2026-05-14/design_a_selective.py  # ~10 s
python analysis/risky_remediation_2026-05-14/design_c_soft_gate.py  # ~10 s
python analysis/risky_remediation_2026-05-14/statistical_validation.py # ~30 s
```
