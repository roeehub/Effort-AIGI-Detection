# Cross-identity color_b_dev — Pearson r per base_identity

**Status**: factual-only. No interpretation. Numbers and direct observations.

**Question being answered**: P8A's r(score, color_b_dev) on Roy_D = -0.7140; the P1_BUNDLE_step500 destruction axis Δ-r = +0.7140. Is the P8A color_b_dev=>real signal Roy_D-specific, or a generalized learned feature?

**Method**: Reused the F3 `per_frame_color_b_dev.csv` (4564 teams_real_all_dev frames, color_b_dev = `np.std(BGR_channel_0)` on 0-255 uint8 decode). Joined per-frame scores from `raw_reports/phase_a/teams_real_all_dev_p8a_reference_step5000_frames_report.csv` and `..._p1_bundle_periodic_step500_frames_report.csv`. Grouped by `base_identity` (via the `extract_base_identity` regex from `phase_d/run_chronic_filter.py`). Per identity, computed three Pearson r values via `numpy.corrcoef` on (P8A_score, color_b_dev), (P1_BUNDLE_step500_score, color_b_dev), and ((P1-P8A) Δ_score, color_b_dev).

**Sample size**: 6 non-Roy_D base_identities (all n ≥ 100 in `teams_real_all_dev`; non-chronic-target). Roy_D included as the anchor row to verify reproduction.

**Compute**: CPU only, num_workers=0, gc.collect() after each identity. No DataLoader.

---

## Per-identity Pearson r

| base_identity | n | r(P8A, color_b_dev) | r(P1_BUNDLE_step500, color_b_dev) | r(Δ_score, color_b_dev) |
| --- | ---: | ---: | ---: | ---: |
| Test_Cam | 1280 | -0.2051 | -0.0635 | +0.0671 |
| PC_Generator | 835 | +0.3360 | +0.1273 | -0.3094 |
| Md_noyn_Sharker | 682 | -0.1381 | -0.2968 | -0.3110 |
| bla_bla_chow | 491 | -0.3346 | -0.0264 | +0.3357 |
| Xiang_Xiang2_Feng | 403 | -0.2234 | +0.1646 | +0.2501 |
| dor | 269 | -0.5998 | +0.0654 | +0.1523 |
| Roy_D | 130 | -0.7141 | +0.1954 | +0.7142 |

## Mechanical pass/fail against |r| > 0.5

Threshold: |r(P8A, color_b_dev)| > 0.5 indicates a strong P8A learned signal on the identity. If the magnitude is consistent across identities, the signal is generalized; if only Roy_D meets threshold, it is identity-specific.

| base_identity | |r(P8A)| | meets |r|>0.5 | |r(P1)| | meets |r|>0.5 | |r(Δ)| | meets |r|>0.5 |
| --- | ---: | :---: | ---: | :---: | ---: | :---: |
| Test_Cam | 0.2051 | no | 0.0635 | no | 0.0671 | no |
| PC_Generator | 0.3360 | no | 0.1273 | no | 0.3094 | no |
| Md_noyn_Sharker | 0.1381 | no | 0.2968 | no | 0.3110 | no |
| bla_bla_chow | 0.3346 | no | 0.0264 | no | 0.3357 | no |
| Xiang_Xiang2_Feng | 0.2234 | no | 0.1646 | no | 0.2501 | no |
| dor | 0.5998 | YES | 0.0654 | no | 0.1523 | no |
| Roy_D | 0.7141 | YES | 0.1954 | no | 0.7142 | YES |

## Direct observations

1. Roy_D anchor row reproduces the prior |r(P8A)| = 0.714 (matches the 0.714 reading in `roy_d_with_axes.csv`).

2. Of 6 non-Roy_D identities tested, 1 meet |r(P8A)| > 0.5; 0 meet |r(P1_BUNDLE_step500)| > 0.5; 0 meet |r(Δ)| > 0.5.

3. Non-Roy_D aggregate: mean |r(P8A)| = 0.306, max |r(P8A)| = 0.600; mean |r(P1_BUNDLE_step500)| = 0.124; mean |r(Δ)| = 0.238.

4. Sign of r(P8A, color_b_dev) on non-Roy_D identities: 5 negative (same sign as Roy_D), 1 non-negative.

5. Roy_D |r(P8A)| = 0.714. Highest non-Roy_D |r(P8A)| = 0.600 on identity = dor. Ratio: 1.19×.

6. Per-identity correspondence between |r(P8A)| and |r(Δ)|:

| base_identity | |r(P8A)| | |r(Δ)| | ratio (Δ/P8A) |
| --- | ---: | ---: | ---: |
| Test_Cam | 0.2051 | 0.0671 | 0.33 |
| PC_Generator | 0.3360 | 0.3094 | 0.92 |
| Md_noyn_Sharker | 0.1381 | 0.3110 | 2.25 |
| bla_bla_chow | 0.3346 | 0.3357 | 1.00 |
| Xiang_Xiang2_Feng | 0.2234 | 0.2501 | 1.12 |
| dor | 0.5998 | 0.1523 | 0.25 |
| Roy_D | 0.7141 | 0.7142 | 1.00 |

## Subsampling

None. All 4564 `teams_real_all_dev` Phase A frames were used; per-identity n is the full count from the merged report.

## Companion artifacts

- `cross_identity_color_b_dev_r.csv` — per-identity Pearson r table (7 rows × 9 cols).
- Source data: `f3_color_b_dev/per_frame_color_b_dev.csv` (color_b_dev), `raw_reports/phase_a/teams_real_all_dev_{p8a_reference_step5000,p1_bundle_periodic_step500}_frames_report.csv` (scores).

## Cross-references

- `roy_d_regression/ROY_D_REGRESSION_FACTS_2026-05-07.md` — original Roy_D r=-0.714 finding.
- `FOLLOWUPS_FACTS_2026-05-07.md` §1 — Task A roy_d axis attribution.
- `FOLLOWUPS_FACTS_2026-05-07.md` §6 — top 13 base_identities × Δ FPR table.