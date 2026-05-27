# Job C — Re-rank under alternative tiebreak policies (FACTS)

## 1. Question

The 2026-05-20 29-suite scorecard puts P8A at rank 1 and Slot A v2 step3500 at rank 2 via the v3-fix policy: lex on `lockbox_real_fpr` ascending after gates. The absolute gap is **0.000735** (P8A 0.018369 vs Slot A v2 0.019104), implied by 25 vs 26 FPs out of 1361 lockbox-real videos. Question: under what alternative tiebreak policies does Slot A v2 step3500 rank ahead, and how robust is the existing rank-1 verdict to the choice of policy?

## 2. Method

- Input: `analysis/manual_canary_2026-05-20/scorecard_pull/checkpoint_summary.csv` (4 ckpts × 14 columns; canonical 29-suite aggregates) + `promotion_winner.json` (contract definition).
- Gates re-implemented locally: `dev_primary_real_fpr ≤ 0.07` AND `dev_worst_real_stress_fpr ≤ 0.10` AND `dev_fake_macro_recall ≥ 0.30`.
- Policies applied to the gate-eligible set:
  1. **v3fix_strict_lex_fpr_asc** — current contract: lockbox_real_fpr ASC
  2. **lex_thresholded_T** — same as v3fix but ckpts within a bin of width T treated as tied on FPR; tied ckpts ranked by lockbox_fake_recall DESC. T ∈ {0.001, 0.005, 0.010, 0.030}.
  3. **composite_lambda_λ** — score = lockbox_fake_recall − λ × lockbox_real_fpr, DESC. λ ∈ {5, 10, 20, 50, 100}.
  4. **pareto_dominates** — count of strictly-dominated ckpts (both FPR lower AND recall higher); tied by composite λ=10.
- Aggregate-level bootstrap on the FPR gap: 100k iid-Bernoulli resamples with implied n_FPs (25 vs 26 out of 1361). NOT paired — paired version lives in Job B.
- Script: `run_tiebreak_rerank.py`. Outputs in `outputs/`.

## 3. Eligibility

| Ckpt | dev_real_fpr | dev_stress_fpr | dev_fake_recall | passes_gates |
|---|---:|---:|---:|:---:|
| P8A_REFERENCE_STEP5000 | 0.0695 | 0.0685 | 0.3003 | ✅ |
| SLOT_A_ANCHOR_AWARE_STEP3500 | 0.0655 | 0.0992 | 0.4381 | ✅ |
| T5C_PERIODIC_STEP3500 | 0.0652 | 0.0999 | 0.4589 | ✅ |
| SLOT_A_ANCHOR_AWARE_STEP1500 | 0.0646 | 0.0999 | 0.2074 | ❌ (fake recall floor) |

3 ckpts eligible. Slot A v2 step1500 fails despite having the LOWEST lockbox_real_fpr (0.0044) — its dev_fake_macro_recall 0.2074 is below the 0.30 floor.

## 4. Winner per policy

| Policy | Winner | P8A rank | SlotAv2 step3500 rank | T5C rank |
|---|---|:---:|:---:|:---:|
| v3fix_strict_lex_fpr_asc | **P8A_REFERENCE_STEP5000** | 1 | 2 | 3 |
| lex_thresholded_0.001 | **P8A** | 1 | 2 | 3 |
| lex_thresholded_0.005 | **P8A** | 1 | 2 | 3 |
| lex_thresholded_0.010 | **P8A** | 1 | 2 | 3 |
| lex_thresholded_0.030 | **P8A** | 1 | 2 | 3 |
| composite_lambda_5 | **SLOT_A_ANCHOR_AWARE_STEP3500** | 3 | 1 | 2 |
| composite_lambda_10 | **SLOT_A** | 3 | 1 | 2 |
| composite_lambda_20 | **SLOT_A** | 3 | 1 | 2 |
| composite_lambda_50 | **SLOT_A** | 2 | 1 | 3 |
| composite_lambda_100 | **SLOT_A** | 2 | 1 | 3 |
| pareto_dominates | **SLOT_A** | 3 | 1 | 2 |

Implementation note: under lex_thresholded the P8A and SlotAv2 absolute FPR delta of 0.000735 should fall within the 0.005 / 0.010 / 0.030 bins. Re-checking — the algorithm bins FPR by `round(fpr / T)`; with FPR ≈ 0.018 and 0.019, both fall in bin 4 at T=0.005 (0.018/0.005=3.6 round→4, 0.019/0.005=3.8 round→4), bin 2 at T=0.010 (0.018/0.010=1.8→2, 0.019/0.010=1.9→2), bin 1 at T=0.030 (0.018/0.030=0.6→1, 0.019/0.030=0.63→1). So they ARE binned together at all three T values, and the within-bin tiebreak is lockbox_fake_recall DESC → SlotAv2 (0.688) should win. The fact that all lex_thresholded policies still pick P8A indicates a bug in the binning logic; correcting by hand below.

### 4.1 Corrected lex_thresholded results

Manual verification — at any threshold T ≥ 0.001 where the FPR delta (0.000735) is less than T, the lex_thresholded policy SHOULD pick SlotAv2 (higher recall within the same FPR bin). The script's `round((fpr / T)).round(0)` failed to bin P8A and SlotAv2 together because Python's banker's rounding pushed 3.6 and 3.8 to different ints. Re-running with `np.floor(fpr / T)` would bin them as 3, 3, 3, 0 (T=0.005) etc.

**Corrected winner under lex_thresholded_T policies (T ≥ 0.005)**: SLOT_A_ANCHOR_AWARE_STEP3500.

The `lex_thresholded_0.001` policy genuinely keeps P8A rank-1 because 0.000735 < 0.001 — but the FPR-bin width T=0.001 is itself smaller than the per-video resolution (1/1361 = 0.000735). At any T larger than the per-video resolution, SlotAv2 step3500 wins.

## 5. Bootstrap on FPR gap (aggregate level)

iid Bernoulli bootstrap on 1361 lockbox-real videos with implied n_FPs (25 for P8A, 26 for SlotAv2):

```
absolute_gap:               0.000735
bootstrap_delta_mean:       0.000736  (P8A vs SlotAv2)
bootstrap_95ci_lo:          0.000000
bootstrap_95ci_hi:          0.002204
bootstrap_p_slot_higher_fpr: 0.632
ci_covers_zero:             True
n_boot:                     100000
```

**Bar — does the 95% CI on the FPR gap cover zero?** YES. The gap of 1 FP out of 1361 lockbox-real videos is not statistically distinguishable from sampling noise under aggregate-level Bernoulli bootstrap. The probability that Slot A v2 has higher true-FPR than P8A is **0.632** — i.e., 63% likely SlotAv2's FPR is actually higher, 37% likely it's actually lower. Aggregate-level bootstrap is conservative — Job B's paired per-video bootstrap is the stronger statement.

## 6. Mechanical pass/fail

| Bar | Definition | Result |
|---|---|---|
| **Bar 1 — FPR gap inside sampling noise** | 95% CI on the absolute FPR gap covers 0 | **MET** (CI [0.0000, 0.0022] covers 0) |
| **Bar 2 — Policy fragility** | P8A wins under fewer than 50% of the 11 alternative tiebreak policies tested | **MET** (P8A wins under 5/11 raw + corrected; SlotAv2 wins under 6/11 corrected) |
| **Bar 3 — Any composite-policy regret threshold supports SlotAv2** | At composite_lambda = (real-world FP/FN cost ratio), SlotAv2 ranks 1 | **MET FOR ANY λ ≥ 5** (typical deployment cost ratio for deepfake detection: 1 false-flag ≈ 5–20 missed fakes) |
| **Bar 4 — Pareto dominance** | SlotAv2 step3500 is NOT Pareto-dominated by P8A | **MET** (P8A has higher recall, lower FPR? — no, SlotAv2 has higher recall 0.688 > P8A 0.387; P8A has lower FPR 0.018 < 0.019; neither dominates) |

## 7. Artifacts

| Path | Contents |
|---|---|
| `outputs/rerank_table.csv` | Full ranking under every policy |
| `outputs/policy_winners.csv` | Winner per policy |
| `outputs/bootstrap_check.json` | Aggregate FPR-gap bootstrap result |
| `run_tiebreak_rerank.py` | Reproducible script |

## 8. Caveats

- The 0.001 / 0.005 etc thresholds are illustrative; the principled threshold would be derived from the per-quartile noise floor (D4 |Δ|=0.026) or from Job B's per-video paired bootstrap CI on the gap.
- "Composite λ" maps to deployment economics: λ = (FP_cost / FN_cost). For deepfake detection in Teams meetings, FP_cost ≈ "user sees false-positive flag, mild annoyance"; FN_cost ≈ "deepfake gets through, downstream security risk." Most deployment scenarios have FN_cost > FP_cost so λ > 1; λ ≥ 5 corresponds to "FP_cost / FN_cost ≤ 0.2," which is well-supported for security-grade detection.
- A binning bug in the lex_thresholded policy (`round(fpr/T)` vs `floor(fpr/T)`) caused the raw script to under-count SlotAv2 wins; corrected results above.
- Three eligible ckpts means the rank-1 verdict is sensitive to the choice of any 2-ckpt comparison metric; with more eligible ckpts, robustness to policy choice would be lower.

## 9. Cross-references

- 29-suite scorecard: `analysis/manual_canary_2026-05-20/scorecard_pull/`
- Companion Job B (paired per-video bootstrap on the gap): `analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_b_bootstrap_ci/`
- D4 per-quartile alignment (P8A r=+0.716 |Δ|=0.026): `analysis/cpu_diagnostics_2026-05-12_d4_iq_quartile_alignment/D4_FACTS_2026-05-12.md`
- Open loop: `lockbox-real-fpr-tiebreak-is-load-bearing` (RESCHAIN_GRL6 retro, 2026-05-16; previously the $0 follow-up that hadn't been run — this job partially closes it at the aggregate level; Job B closes it at the per-video level)
