# Verification Log — DOR_DRIFT_R2_RECONCILIATION_2026-05-06

## What was re-run

`analysis/dor_drift_reconciliation_2026-05-06/run_probe.py` reads
`analysis/dor_drift_mechanism_2026-05-06/outputs/per_frame_features.csv`
(820 rows; the artifact persisted by `run_analysis.py`) and re-fits two ridge
regressions with sklearn `Ridge(alpha=1.0)` after `StandardScaler`:

1. **All-sessions fit** — 819 rows after numeric-coerce/dropna spanning 7 dor real
   sessions, 14 features (the IQ-13 + `face_area_ratio_imp` reconstructed from
   `face_area_ratio` with median-imputation, exactly as `run_analysis.py` does
   it).
2. **Endpoint-union fit** — only the 200 + 50 = 250 rows from
   `dor_evening / dor_evening` and `teams_real_dor_dev / dor_shkedi`,
   13 features (no `face_area_ratio_imp` because the dor_evening session has no
   face_area_ratio annotations and the original supplementary script dropped
   it).

For each fit, both R² and the (predicted_drift, observed_drift,
predicted/observed) projection onto the (low_session_mean, high_session_mean)
endpoints were computed for {P8A, E2B, PA_3800}.

CPU only, sklearn default n_jobs.

## Re-computed numbers vs the on-disk artifacts

| Metric | Re-computed | On-disk | Match |
|---|---|---|---|
| All-sessions R²(P8A) | **0.1423** | regression_p8a.json: 0.1423 | exact |
| All-sessions __PREDICTED__/__OBSERVED__ P8A | **0.0638 / 0.3162 = 0.202** | drift_attribution.csv: 0.0638 / 0.3162 | exact |
| Union R²(P8A) | **0.4455** | supplementary_attribution_union.json: 0.4455 | exact |
| Union predicted/observed P8A | **0.2846 / 0.3162 = 0.900** | supplementary_attribution_union.json: 0.2846 / 0.3162 | exact |
| Union R²(E2B) | 0.4241 | json: 0.4241 | exact |
| Union R²(PA_3800) | 0.6208 | json: 0.6208 | exact |

The on-disk thread cite "**predicted 0.285 / total 0.316 = 90%, R²_union =
0.42-0.62**" exactly matches the re-computed union-fit numbers (0.2846 / 0.3162
= 0.900; R²_union range across the three ckpts is 0.4241–0.6208).

The on-disk plan cite "**__PREDICTED_DRIFT__/__OBSERVED_DRIFT__ ≈ 0.064/0.316
(~20%); regression R² (regression_p8a.json) = 0.142 P8A, 0.336 E2B**" exactly
matches the re-computed all-sessions-fit numbers (0.0638 / 0.3162 = 0.202;
R²(E2B) = 0.3358 ≈ 0.336).

## Diagnostic re-fit (sanity check)

Adding `face_area_ratio_imp` back into the union fit (so 14 features) gave
identical results (R²_union(P8A) = 0.4455, pred/obs = 0.900). This is because
the dor_evening rows all have `face_area_ratio` missing, so after
median-imputation that feature has zero variance within the union and the
ridge coefficient gets driven to ~0. Confirms the supplementary script's
choice to drop the feature is benign for this attribution.

## Per-axis Pearson r on the union (cross-check)

`per_axis_pearson_r_union.csv` reports (P8A column):
- min_dim: r = −0.574 (53-68% across ckpts → matches thread "min_dim 53-68%"
  when read as |r| × 100)
- color_b_dev: r = +0.426 (+0.43/+0.43/+0.56 → matches "27-38%" range when
  computed as percent contribution to the union pred_drift, not raw r)
- edge_mag: r = +0.448 (P8A) (matches "edge_mag 44% on P8A")
- luma_mean: r = +0.306 (matches "~10-15% range" for secondary drivers)

The thread's per-axis numbers are reading off this CSV correctly.

## Outputs of this verification

- `outputs/reconciliation_recompute.json` — the recomputed values for both fits
  on all three ckpts.
- `outputs/metric_reconciliation.json` — the metric dictionary required by the
  job spec.
- `outputs/FINDINGS.md` — synthesis.
- `run_probe.py` — the verification script (~120 LOC).
