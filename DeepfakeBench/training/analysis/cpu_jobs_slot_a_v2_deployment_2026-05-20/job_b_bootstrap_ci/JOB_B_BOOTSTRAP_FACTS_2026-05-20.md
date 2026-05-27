# Job B — Paired bootstrap CI on lockbox_real_fpr delta — FACTS — 2026-05-20

**Status: factual-only.** No interpretation language ("succeeds", "fails", "wins",
"deployment-grade"). Mechanical pass/fail against pre-stated bars at end.

---

## Question answered

Is the 0.07pp gap on `lockbox_real_fpr` between `P8A_REFERENCE_STEP5000` (0.01837)
and `SLOT_A_ANCHOR_AWARE_STEP3500` (0.01910) within sampling noise on the lockbox
real cohort?

Per-video paired bootstrap of `delta = lockbox_real_fpr[SlotAv2_step3500] - lockbox_real_fpr[P8A_step5000]`
at the contract-frozen thresholds.

---

## Inputs

GCS source prefix:
```
gs://training-job-outputs/test_results/teams_promotion_contract/slot-a-v2-validation-2026-05-20/reports/
```

Files pulled (8 per-video reports + 2 contract artifacts):
- `data/teams_real_all_lockbox_p8a_reference_step5000_videos_report.csv` (1361 rows)
- `data/teams_real_all_lockbox_slot_a_anchor_aware_step3500_videos_report.csv` (1361 rows)
- `data/teams_real_lighting_extreme_lockbox_p8a_reference_step5000_videos_report.csv` (207 rows)
- `data/teams_real_lighting_extreme_lockbox_slot_a_anchor_aware_step3500_videos_report.csv` (207 rows)
- `data/teams_real_poor_quality_lockbox_p8a_reference_step5000_videos_report.csv` (22 rows)
- `data/teams_real_poor_quality_lockbox_slot_a_anchor_aware_step3500_videos_report.csv` (22 rows)
- `data/teams_real_all_dev_p8a_reference_step5000_videos_report.csv` (3253 rows)
- `data/teams_real_all_dev_slot_a_anchor_aware_step3500_videos_report.csv` (3253 rows)
- `data/promotion_winner.json` (contract config + winner)
- `data/checkpoint_summary.csv` (per-ckpt frozen τ + headline metrics)

Schema (per row): `method,label,video_id,avg_video_prob,prediction,is_correct,group_key,family_key`.
Score column used: `avg_video_prob`. Real videos are `label == 0`. All real-lockbox CSVs
contain only `label == 0` rows.

Suite-cardinality structure (verified from `video_id` set membership):
- `teams_real_lighting_extreme_lockbox` (207 vids) ⊂ `teams_real_all_lockbox` (1361 vids)
- `teams_real_poor_quality_lockbox` (22 vids) ⊂ `teams_real_all_lockbox` (1361 vids)
- `lighting_extreme ∩ poor_quality` = 4 vids
- Set-union of all three = 1361 vids (== `teams_real_all_lockbox`)

Direct consequence: the contract's `lockbox_real_fpr` is computed on
`teams_real_all_lockbox` alone (per `promotion_winner.json` → `contract.lockbox_real_suite`);
the union of the three lockbox real suites is identical to `teams_real_all_lockbox`.

---

## Method

### τ-calibration

The contract uses the lex-tiered policy from
`arena/score_teams_promotion_contract.py` (`_threshold_sort_key`):
- Tier 0: rows with `dev_primary_real_fpr ≤ 0.07` AND `dev_worst_real_stress_fpr ≤ 0.10`
  AND `dev_fake_macro_recall ≥ 0.30` (recall floor relaxed from the 0.70 default for
  this scorecard; see `promotion_winner.json` → `contract.target_fake_recall_min = 0.3`).
- Within Tier 0: maximize `dev_fake_macro_recall`, then prefer higher τ, then prefer
  lower primary FPR.

Frozen τ (verbatim from `data/checkpoint_summary.csv`):
| checkpoint_key | selected_threshold |
|---|---:|
| P8A_REFERENCE_STEP5000 | 0.915605 |
| SLOT_A_ANCHOR_AWARE_STEP3500 | 0.787956 |

τ-sanity replication (Job B script `derive_tau_from_dev` — smallest τ in observed dev
score set meeting `dev_primary_real_fpr ≤ 0.07`):
| ckpt | τ_min_meeting_budget | τ_contract_selected | dev_n_real |
|---|---:|---:|---:|
| P8A | 0.913695 | 0.915605 | 3253 |
| SlotAv2 | 0.776480 | 0.787956 | 3253 |

Contract τ is above the budget floor for both ckpts, consistent with the
`prefer higher τ` tiebreaker.

### Bootstrap

- Paired resampling: real video indices resampled with replacement; both ckpts share
  the same resampled index vector (paired bootstrap exploits the fact that the two
  ckpts score the SAME videos).
- N = 10,000 resamples.
- RNG: `numpy.random.default_rng(seed=20260520)`.
- Statistic: `delta = fpr_slotav2 - fpr_p8a`, where each `fpr_*` is computed on the
  resampled video set at that ckpt's frozen τ.
- CI method: percentile (2.5%, 97.5%).

### Scope of bootstrap

Analysis A (primary, equal to the contract metric): bootstrap over the 1361 videos
in `teams_real_all_lockbox`.

Analysis B (task literal "union of 3 lockbox real suites"): the set-union of the
three lockbox real suites is identical to `teams_real_all_lockbox` because both
stress suites are strict subsets (see Inputs §). Analysis B is therefore identical
to Analysis A and is not re-run.

---

## Output artifacts

- `run_bootstrap.py` — bootstrap script (executable from this directory).
- `bootstrap_results.json` — full results (CIs, observed values, bars).
- `data/` — pulled CSVs + contract JSONs (CSV inputs are large; not git-tracked).

---

## Numbers

### Observed point estimates (Analysis A, n_real = 1361)

| Quantity | Value | n_FP / n_real |
|---|---:|---:|
| FPR_P8A at τ=0.915605 | 0.018369 | 25 / 1361 |
| FPR_SlotAv2 at τ=0.787956 | 0.019104 | 26 / 1361 |
| delta (SlotAv2 − P8A) | +0.000735 | +1 video |

Cross-check vs `data/checkpoint_summary.csv`:
- P8A `lockbox_real_fpr` = 0.018369 ✓ (matches contract)
- SlotAv2 `lockbox_real_fpr` = 0.019104 ✓ (matches contract)

### Bootstrap 95% percentile CIs (N=10,000, seed=20260520)

| Quantity | 95% CI | Bootstrap mean |
|---|---:|---:|
| FPR_P8A | [0.011756, 0.025716] | 0.018419 |
| FPR_SlotAv2 | [0.012491, 0.026451] | 0.019036 |
| delta (SlotAv2 − P8A) | [−0.008082, +0.009552] | +0.000616 |

### Sign probability of delta

| Event | Probability |
|---|---:|
| P(delta > 0) | 0.5193 |
| P(delta = 0) | 0.0651 |
| P(delta < 0) | 0.4156 |

The bootstrap distribution of delta is symmetric around 0 to within < 0.001;
P(delta > 0) − P(delta < 0) = +0.104; the +0.000735 observed delta corresponds
to a 1-video swing (25 vs 26 false positives out of 1361).

---

## Direct observations

1. The observed delta of +0.000735 is one false-positive video different out of
   1361 real lockbox videos at each ckpt's contract-frozen τ.
2. The 95% bootstrap CI on delta is [−0.00808, +0.00955]. The interval covers 0.
3. The individual 95% CIs on FPR_P8A ([0.0118, 0.0257]) and FPR_SlotAv2 ([0.0125,
   0.0265]) overlap on the range [0.0125, 0.0257].
4. P(delta > 0) under the bootstrap is 0.519 — within 2 percentage points of 50%.
5. The contract's `lockbox_real_fpr` field, the `lockbox_real_n_videos` field
   (1361), and the bootstrap point estimates agree to the 4th decimal place. The
   numbers fed to the bootstrap reproduce the contract verdict exactly.
6. τ-calibration sanity: contract-selected τ exceeds the budget-floor τ by
   +0.001910 (P8A) and +0.011476 (SlotAv2), consistent with the `prefer higher τ`
   contract tiebreaker. The bootstrap is run at the actual contract-selected τ.

---

## Pre-stated bars

| Bar | Bar text | Inputs | Met? |
|---|---|---|---|
| **Bar 1** (load-bearing) | 95% CI on delta covers 0 | delta CI = [−0.00808, +0.00955] | **MET** |
| **Bar 2** (overlap) | P8A 95% FPR CI overlaps Slot A v2 95% FPR CI | P8A [0.0118, 0.0257]; SlotAv2 [0.0125, 0.0265]; overlap [0.0125, 0.0257] | **MET** |

Both bars MET on the primary analysis (Analysis A = teams_real_all_lockbox = 1361
videos). Analysis B (literal union of 3 lockbox real suites) is identical to
Analysis A because the stress lockbox suites are strict subsets of
`teams_real_all_lockbox`.
