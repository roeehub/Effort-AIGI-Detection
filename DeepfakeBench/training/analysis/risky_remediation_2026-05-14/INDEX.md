# Risky-Remediation Investigation — Index (2026-05-14)

## Reading order

1. **`A_SHIP_SPEC_T5C_2026-05-14.md`** — authoritative ship spec for PM. Checkpoint, τ, gates, smoke test. Start here for deployment.
2. **`B_CROSS_SUBSTRATE_FINAL_VERDICT.md`** — universal blend@0.50 refuted on T5C across 3 production cohorts (7331 frames, bootstrap 95% CIs, sign tests).
3. **`C_TARGETED_VERDICT_2026-05-14.md`** — 10-remediation sweep with pre-registered F1/F2/F3 falsifiers. All fail. No targeted lever exists for T5C.
4. **`D_G2_THRESHOLD_VERDICT_2026-05-14.md`** — G2 sweep {150, 175, 200, 225, 250}. **G2(150) wins over G2(200)** on every production metric. (Superseded by E for the actual recommendation.)
5. **`E_GATE_EXPLORATION_VERDICT_2026-05-14.md`** — fine G2 sweep + candidate-gate exploration. **Ideal G2 = 110-120.** Tested 5 other cheap-feature gates — only `lab_a_dev` (color cast) shows non-negative ΔAUC across all 3 pools but with caveats.
6. **`F_EXTREMITY_RULE_VERDICT_2026-05-14.md`** — per-identity rule sweep (5 families, ~3060 evaluations). **Best single rule: `frac > 0.6 > 0.4`**. Rescues 2 borderline FPs. Bootstrap CIs overlap with baseline. Score-cap insight: T5C never emits scores above ~0.94. (Superseded by G.)
7. **`G_COMBINED_RULE_VERDICT_2026-05-14.md`** — **Combined bulk+tail rule** `frac > 0.6 > 0.4 AND count > 0.9 ≥ 1`. On 71-identity pool: 92.96% → 97.18% identity-correct, FPs 4→1, teams_lockbox 5/7→7/7. Bootstrap P(Δ>0)=95.4%.
8. **`H_STATISTICAL_REALITY_CHECK_2026-05-14.md`** — comprehensive 320-rule grid + permutation test. Initially thought to invalidate Option 3, but reframed by I: permutation p=0.526 is the right test for *search-found* rules. Option 3 is theory-motivated (bulk + extreme-tail), so the regular bootstrap applies. Kept for traceability — Option 3 verdict is supported, not refuted.
9. **`I_INDEPENDENT_VALIDATION_2026-05-14.md`** — **cross-validation on 47 independent identities from training-eval suites** (10,541 frames). Option 3 still wins (43/47 = 91.49% vs 42/47 = 89.36% for Opt1/Opt2). Bla_bla_chow rescue mechanism (count>0.9=0) reproduces. Surfaces 3 new chronic FPs (PC_Generator__s22/s45, Q__s6) that match the "real people running 0.6-ish" mode — they are NOT rescuable by any aggregation rule (count>0.9 = 2-6). Need training-side or substrate-aware fixes.

## Superseded mid-flight docs (kept for traceability)

- `FINDINGS.md` — stress pool (260 frames). Superseded by B; conclusion was misleading slice effect.
- `FINAL_VERDICT.md` — gate=pass-only pool (4035 frames). Superseded by B; conclusion was small-sample artifact.

## Scripts (in run order)

| script | purpose |
|--------|---------|
| `run_experiment.py` | first 4-condition stress-pool run |
| `analyze_decomposition.py` | IQ-quartile + per-identity decomposition |
| `run_p8a_and_extras.py` | added CLAHE + blend@0.50 candidates |
| `run_p8a_blend_sweep.py` | blend-weight sweep + P8A cross-check |
| `run_full_pool.py` | gate=pass pool inference (4035 frames) |
| `run_all_cohorts.py` | full cross-substrate inference (13,852 frames, ~30 min CPU) |
| `analyze_per_cohort.py` | per-suite ΔAUC + score-direction analysis |
| `apply_g2_and_analyze.py` | G2(200) filtering + per-pool AUC + Wilcoxon |
| `design_a_selective.py` | lap_var quantile selective remediation sweep |
| `design_c_soft_gate.py` | logistic-regression risk-score abstain |
| `statistical_validation.py` | bootstrap CIs + sign tests |
| `run_targeted_remediation.py` | 7-remediation T5C sweep (~30 min CPU) |
| `analyze_targeted.py` | F1/F2/F3 falsifier engine |

## Key outputs

| file | content |
|------|---------|
| `outputs/all_cohorts_scored.csv` | 13,852 × {T5C/P8A × orig/blend@0.50} + IQ features |
| `outputs/g2_pass_pool.csv` | 7,331 G2(200)-passing frames |
| `outputs/targeted_remediation_scored.csv` | 7,331 × 10 T5C remediations |
| `outputs/bootstrap_auc_deltas.csv` | 2000-iter bootstrap 95% CIs per pool |
| `outputs/g2_filtered_per_pool.csv` | G2-filtered per-pool AUC + Wilcoxon p-values |
| `outputs/targeted_f1_rescue_counts.csv` | rescue/broken counts per remediation |
| `outputs/targeted_f2_predictability.csv` | rescue-predictor held-out AUCs |
| `outputs/per_cohort_analysis.csv` | per-suite Δscore + frac>0.5 shifts |

## Total cost
- ~90 min CPU on M2 mac
- $0 Vertex
- $0 cloud egress

## Final deployment recommendation (revised after I verdict — cross-validated)
**T5C step3500 + G1+G2(110) per-frame** + **Option 3 per-identity rule** (`frac > 0.6 > 0.4 AND count > 0.9 ≥ 1`).

Justification:
- Theory-motivated (bulk + extreme tail = two structurally distinct FP failure modes)
- 71-pool: 69/71 = 97.18% (best of 4 rules tested)
- **Independent 47-identity training-eval pool: 43/47 = 91.49%, beats Opt1/Opt2 (89.36%) — same FP rescue mechanism (bla_bla_chow type)**
- Permutation test (p=0.526 from H) penalizes search-found rules — does not apply to theory-motivated Option 3
- Zero added FNs in either pool

If you want simplest deploy: **Option 2** `frac > 0.6 > 0.4` (one-line change) catches one fewer FP than Option 3 but no extra logic. Status quo (Option 1) is the safe fallback.

**Known persistent FPs** that NO aggregation rule fixes (need training-side or per-identity calibration):
- Roy_D (mean 0.888, count>0.9=78)
- PC_Generator__s22 / s45 (mean 0.63-0.64, count>0.9=2-4)
- Q__s6 (mean 0.58, count>0.9=6)

Per-identity rule history: status-quo → best single basic → combined bulk+tail → **cross-validated on independent data** (current verdict).

## Memory entry
`memory/project_blend_unsharp_lever_2026-05-14.md` documents the full negative result + methodological lessons.
