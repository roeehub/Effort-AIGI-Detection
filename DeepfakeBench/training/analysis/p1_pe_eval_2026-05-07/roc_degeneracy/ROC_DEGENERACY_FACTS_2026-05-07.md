# ROC degeneracy probe — P1 ckpts on `dev_fake_macro_recall` vs `dev_primary_real_fpr`

**Status**: factual-only. No interpretation. Numbers and direct observations.

**Question being answered**: For each P1 ckpt, what is the achievable recall at FPR ≤ 0.07 across the threshold grid? Does the ckpt have any τ candidate at recall ≥ 0.30 with FPR within budget?

**Method**: Read `analysis/p1_pe_eval_2026-05-07/scorecard/threshold_grid.csv` (8 ckpts × ~5350 grid points each). For each ckpt:
1. Filter to `(dev_primary_real_fpr ≤ 0.07) AND (dev_worst_real_stress_fpr ≤ 0.10)` — call this "budget-OK" region.
2. Within budget-OK, count points with `dev_fake_macro_recall ≥ 0.30` — these are "tier-0" candidates per `arena/score_teams_promotion_contract.py:485-499`.
3. Record τ range, recall range in budget, plus recall at points just below budget.

**Output artifacts**:
- `roc_degeneracy_summary.csv` (8 rows × 11 cols)
- `roc_curve_samples.csv` (8 ckpts × 101 sampled τ points each)

---

## Summary table

| ckpt | n_grid | n_budget_OK | n_tier0 (recall ≥ 0.30) | τ_min in budget | τ_max in budget | max recall in budget | max recall just below budget | mean FPR just below budget |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 5361 | 1257 | **2** | 0.9148 | 1.0000 | **0.3003** | 0.3182 | 0.0719 |
| E2B_TOP_N_STEP3200 | 5599 | 1800 | **561** | 0.7108 | 1.0000 | 0.5085 | 0.5198 | 0.0709 |
| P1_BUNDLE_PERIODIC_STEP500 | 5552 | 720 | **0** | 0.9919 | 1.0000 | 0.0750 | 0.0821 | 0.0682 |
| P1_BUNDLE_TOP_N_STEP3750 | 5339 | 923 | **0** | 0.9994 | 1.0000 | 0.1920 | 0.2142 | 0.0538 |
| P1_BUNDLE_TOP_N_STEP4000 | 5292 | 1032 | **0** | 0.9989 | 1.0000 | 0.2846 | 0.3063 | 0.0554 |
| P1_PAIRRANK_PERIODIC_STEP500 | 5592 | 1408 | **117** | 0.7675 | 1.0000 | 0.3538 | 0.3733 | 0.0645 |
| P1_PAIRRANK_TOP_N_STEP6000 | 5390 | 1281 | **94** | 0.9876 | 1.0000 | 0.3391 | 0.3574 | 0.0621 |
| P1_PAIRRANK_TOP_N_STEP6750 | 5447 | 1273 | **98** | 0.9900 | 1.0000 | 0.3425 | 0.3637 | 0.0613 |

## Direct observations (no interpretation, just what the numbers say)

1. **5 of 8 ckpts have ≥ 94 tier-0 candidates**. 3 of 8 have 0 tier-0 candidates: BUNDLE_step500 (max recall in budget = 0.075), BUNDLE_step3750 (0.192), BUNDLE_step4000 (0.285).

2. **P8A scrapes into tier-0 by 0.0003 absolute** — its max recall in budget is 0.3003, the floor is 0.30. 2 grid points qualify.

3. **BUNDLE_step4000 falls 0.015 absolute short** of tier-0 — max recall in budget is 0.285. Just below budget (mean FPR 0.055), recall reaches 0.306.

4. **All 8 ckpts have τ_max_budget = 1.0000.** This means: within the FPR budget, the contract grid extends to the highest possible τ for every ckpt.

5. **τ_min_budget varies widely**: E2B = 0.7108 (4292 τ values above this in budget); BUNDLE_step3750 = 0.9994 (923 τ values above this in budget). The lower the `τ_min_budget`, the more "room" the ckpt has to trade FPR for recall before hitting the budget ceiling.

6. **For all 3 BUNDLE ckpts: recall just below budget barely exceeds recall in budget.** BUNDLE_step3750: in-budget max=0.192, just-below=0.214 (Δ +0.022 absolute over a ~5pp FPR bump). BUNDLE_step500: 0.075 → 0.082 (Δ +0.007). BUNDLE_step4000: 0.285 → 0.306 (Δ +0.021). The recall vs FPR trade is shallow in this band.

7. **For E2B and 3 PAIRRANK ckpts, recall climbs more readily as FPR is relaxed**. E2B: 0.509 → 0.520 (Δ +0.011 over similar FPR bump — but starting from a much higher base).

8. **The recall floor (0.30) bisects the 8 ckpts:**
   - Above floor (have tier-0): E2B, PAIRRANK 500/6000/6750, P8A (just barely).
   - Below floor (no tier-0): BUNDLE 500/3750/4000.

## Companion data

- `roc_curve_samples.csv` has 101 τ sample points per ckpt for downstream plotting/inspection. Columns: ckpt, threshold, real_fpr, fake_recall, stress_fpr.
- The sort key per `arena/score_teams_promotion_contract.py:494-499` is `(tier, macro_recall_neg, threshold_neg, primary_fpr)`. Within tier 1 (budget OK + recall < floor), the sort prefers max macro_recall, then max threshold, then min primary_fpr.
- For the 3 zero-tier-0 BUNDLE ckpts, the within-tier-1 selected τ is at the high end of the budget range (τ_max_budget = 1.0000 minus a tiny step) because that's where macro_recall in tier 1 is highest given the ROC shape.
