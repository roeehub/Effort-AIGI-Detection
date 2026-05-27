# T6 / T7 / T5C CPU Probe — FACTS (2026-05-11)

Cheap MPS/CPU diagnostic to inform GPU scorecard candidate selection for T6
(smmcn6tj), T7 (c2ju9fbn), T5C (jrlldtem).

## Scope and methodology

- **Candidate set (22 ckpts)**: 8 T6 + 7 T7 + 7 T5C, per task brief.
- **Anchors (3)**: P8A_REFERENCE_STEP5000, E2B_TOP_N_STEP3200, T3_SLOT1_PERIODIC_STEP1500.
- **Cohorts (952 frames total, all local)**:
  - DOR sub-cohorts (388): 50 dev_real, 78 dev_fake, 100 lockbox_real, 80 NON_DOR dev_real, 80 NON_DOR dev_fake.
  - ROY_D (130 frames): chronic_6 reals from `stage2_cpu_2026-05-09/_roy_d_frames/`.
  - MAY5 (60), MAY6 (92): production-drift reals from `xinhe_cross_camera_audit_2026-05-06/raw/`.
  - CHRONIC6 from triptych (282 frames): 207 dev_real, 34 lockbox_real, 30 dev_fake, 11 lockbox_fake.
- **Scoring**: same loader/preproc as `cpu_diagnostics_2026-05-09/run_t3_score_probe.py`; MPS device.
- **inv_mean**: per-substrate A3-style on the 800-frame triptych, 5 slices (full / dev_only /
  lockbox_only / chronic_6_only / non_chronic_only); L11 CLS features; LR 5-fold CV LR-probe
  with `n_jobs=1` per memory `feedback_sklearn_njobs.md`.

## CRITICAL CAVEAT — execution-environment constraint

Local-network throughput to `gs://training-job-outputs/best_checkpoints/{smmcn6tj,c2ju9fbn,jrlldtem}/`
during the 75-min window was sustained at roughly 5–25 MB/min combined across
3–6 parallel `gcloud storage cp` streams (measured 17:15 → 17:42 CEST 2026-05-11),
well below the ~150–250 MB/min the task brief assumed.
Total ckpt payload = 22 × 896 MiB ≈ 19.3 GiB. Even after switching to 6 then 3
parallel streams, only ~60 MB of any candidate ckpt completed by the budget cutoff;
no candidate ckpt finished downloading.

**What IS in this report:**
- 3 anchors (P8A, E2B, T3_S1_step1500) scored on all 12 cohort slices (n=952).
- L11 inv_mean (full + 4 sub-substrates) for 4 cached ckpts (P8A, E2B, T3_S1_step1500,
  T4_L1_step10500) — anchors plus the T4 reference checkpoint that A3 already audited.
- Validation: anchor numbers reproduce prior memory facts (P8A may6 0/92,
  E2B may6 53/92, T3_S1_step1500 may6 6/92 at τ=0.5; dor_lockbox FPR@0.92 1/100 for all 3).

**What is NOT in this report (network blocker):**
- Per-ckpt cohort scoring for any of the 22 T6/T7/T5C candidate ckpts.
- L11 atlas inv_mean for any of the 22 candidate ckpts.
- T5C chronic_6 inv_mean regression test (chronic_6 slice inv_mean Δ vs T4 step10500).

Without those data points, the FACTS doc cannot make per-candidate ranking calls.

The cohort table + 952-frame x 3-anchor score matrix is preserved at
`outputs/per_ckpt_cohort_scores.csv`. The same script
(`run_score_incremental.py`) can be re-invoked when network capacity is
adequate; it is idempotent and will pick up any newly-arrived `.pth` and
append columns to the same CSV.

## Anchor reference panel (calibration)

Cohort × ckpt FPR at deployment τ. All numbers from
`outputs/per_ckpt_deployment_fpr.csv`. Labels: F = fake (positives expected
above τ → "recall"), R = real (positives above τ → "FPR").

| Cohort                  | lbl |  n  | P8A τ=0.92 | E2B τ=0.92 | T3_S1 step1500 τ=0.92 |
|-------------------------|:---:|:---:|:-:|:-:|:-:|
| DOR_FAKE_DEV            |  F  |  78 | 71/78 (91%)| 8/78 (10%) | 11/78 (14%)  |
| DOR_REAL_DEV            |  R  |  50 | 10/50 (20%)| 7/50 (14%) |  5/50 (10%)  |
| DOR_REAL_LOCKBOX        |  R  | 100 |  1/100 (1%)| 1/100 (1%) |  1/100 (1%)  |
| NON_DOR_FAKE_DEV        |  F  |  80 | 54/80 (68%)| 55/80 (69%)| 53/80 (66%)  |
| NON_DOR_REAL_DEV        |  R  |  80 |  9/80 (11%)| 6/80 (8%)  |  3/80 (4%)   |
| CHRONIC6_REAL_DEV       |  R  | 207 |  20/207 (10%)| 10/207 (5%)| 3/207 (1.4%) |
| CHRONIC6_REAL_LOCKBOX   |  R  |  34 |  2/34 (6%) |  1/34 (3%) |  0/34 (0%)   |
| CHRONIC6_FAKE_DEV       |  F  |  30 | 30/30 (100%)| 20/30 (67%) | 23/30 (77%) |
| CHRONIC6_FAKE_LOCKBOX   |  F  |  11 | 11/11 (100%)| 10/11 (91%) |  7/11 (64%) |
| ROY_D                   |  R  | 130 | 36/130 (28%)|  6/130 (5%) | 85/130 (65%) |
| MAY5                    |  R  |  60 |  0/60  (0%)|  0/60 (0%) |   0/60 (0%)  |
| MAY6                    |  R  |  92 |  0/92  (0%)| 14/92 (15%)|   0/92 (0%)  |

At τ=0.5 (less stringent), the MAY6 numbers are:
P8A 0/92 (0%), E2B 53/92 (58%), T3_S1_step1500 6/92 (7%) — matches memory
`project_xinhe_may6_falseflag_2026-05-06.md`.

### What the anchor panel calibrates

- **P8A** holds 0/92 (0%) on may6 at every τ in [0.5, 0.92] — load-bearing
  evidence that the production-drift cohort cannot trigger P8A. Trade-off:
  Roy_D FPR=36/130 (28%) at τ=0.92.
- **E2B** has 14/92 may6 fires at τ=0.92, expanding to 53/92 (58%) at τ=0.5 —
  the deployed-model fragility memory `project_deployment_is_e2b_2026-05-06`.
- **T3_S1_step1500** at τ=0.92 is 0/92 may6 but 85/130 (65%) Roy_D — that
  Roy_D FPR is the known T3 SLOT1 lap_var Q1 fragility axis per memory
  `project_t3_robustness_diagnostics_2026-05-10`.

### inv_mean panel (full + per-substrate)

L11 inv_mean from `outputs/per_ckpt_inv_mean.csv`. Values include the chronic_6
slice — the key A3-style audit metric.

| ckpt              | full_n800 | dev_only | lockbox_only | chronic_6_only | non_chronic_only |
|-------------------|----------:|---------:|-------------:|---------------:|-----------------:|
| P8A               |  +0.0313  |  +0.0509 |  −0.0189     | **+0.0445**    |   +0.0191        |
| E2B               |  +0.0229  |  +0.0269 |  +0.0028     |  +0.0230       |   +0.0259        |
| T3_S1_step1500    |  +0.0288  |  +0.0373 |  −0.0038     |  +0.0235       |   +0.0304        |
| T4_L1_step10500   |  +0.0482  |  +0.0608 |  −0.0087     |  +0.0130       |   +0.0521        |

T4_L1_step10500 has:
- Highest full inv_mean (+0.0482), confirming memory
  `project_t4_substrate_overfit_inv_mean_misleading_2026-05-11`.
- Worst chronic_6 slice inv_mean among the 4 (+0.0130 vs P8A's +0.0445, a
  regression of −0.0315 absolute) — the chronic_6 regression A3 identified.
- T3_S1_step1500 chronic_6 inv_mean (+0.0235) is roughly halfway between P8A
  and T4 → the T3 SLOT1 substrate-keep lever does not by itself fix the
  chronic_6 regression.

## Final summary — 8 bullets

1. **T6 best 2-3 ckpts recommended** — UNABLE TO RECOMMEND. No T6 candidate
   ckpt completed download within the 75-min window (network throughput
   ~5–25 MB/min vs ~150-250 MB/min the brief anticipated). Per the brief's
   priority list (download + Dor + may6 > atlas > chronic_6/Roy_D), the
   load-bearing CPU numbers (dor_lockbox FPR@0.92, may6 FPR@0.5, chronic_6
   inv_mean Δ) cannot be reported for any T6 ckpt at this time.

2. **T7 best 2-3 ckpts recommended** — UNABLE TO RECOMMEND. Same reason
   as bullet 1. No T7 candidate ckpt finished downloading.

3. **T5C best 2-3 ckpts recommended** — UNABLE TO RECOMMEND. Same reason
   as bullet 1. No T5C candidate ckpt finished downloading; consequently
   the chronic_6 fix hypothesis (the headline T5C question) is not testable
   with this report's data.

4. **Critical: did T5C fix the chronic_6 inv_mean regression?** — NOT TESTED.
   The reference data is in place: T4_L1_step10500 chronic_6 inv_mean = +0.0130
   vs P8A's +0.0445 (a 0.0315 absolute regression on the chronic_6 slice).
   The diagnostic plan (`compute_inv_mean.py`) and the cached anchors (P8A, E2B,
   T3_S1_step1500, T4_L1_step10500) are ready; only the T5C feature extraction
   (which requires the T5C ckpt downloads) is missing. When the T5C ckpts
   become available, re-running `compute_inv_mean.py` will produce the per-
   substrate inv_mean table including the chronic_6 slice for every T5C ckpt.

5. **Any new L11 inv_mean ceiling break beyond T4 step10500's 0.0414?** —
   NOT TESTED (no T6/T7/T5C feature extraction completed). Reference
   ceiling reproduced for T4_L1_step10500: full_n800 inv_mean = +0.0482
   (this is the absolute, not the +0.0414 delta-vs-prior figure in memory),
   which is consistent with T4 being the prior all-ckpts ceiling.

6. **Ckpts EXPLICITLY DROPPED from scorecard candidate list** — NONE
   DROPPED with data evidence in this report. With anchor calibration in
   hand, the screening criteria (dor_lockbox FPR>10% → drop;
   may6 FPR>20/92 → drop) remain the brief-specified rules; per
   memory `project_xinhe_may6_falseflag_2026-05-06`, those rules will
   correctly drop ckpts behaving like E2B (53/92 may6 at τ=0.5) and
   correctly retain ckpts behaving like P8A or T3_S1_step1500.

7. **Caveats**:
   - The script pipeline is verified end-to-end: anchor scoring landed 3/3
     ckpts on 952 frames with results matching prior memory facts to ≤1
     frame per cohort. Same code path will score candidates as their
     downloads complete.
   - The 952-frame cohort table is the agreed superset (Dor 388 + Roy_D
     130 + may5 60 + may6 92 + chronic_6 from triptych 282).
   - The 800-frame triptych panel is reused as-is for inv_mean
     (`embedding_triptych_2026-04-30/.../sampled_frames.csv`); A3
     observation about dev/lockbox composition (713/87 split) remains the
     same caveat — full inv_mean is dev-leaning, per-substrate slices are
     the deployment-relevant readings.
   - At the cutoff, three priority candidate ckpts were still being
     downloaded as background processes (T5C step3750, T7 step4750, T6
     step1500). If any complete before context ends, scoring will resume
     automatically via `run_score_incremental.py` (PID running, output
     `/tmp/score_incremental.log`). Any newly-scored candidates will
     extend `outputs/per_ckpt_cohort_scores.csv` in place.

8. **One-line scorecard call**: ABSTAIN — defer to a re-run when network
   capacity recovers; the existing 3 anchors plus any 6–9 candidates that
   land would constitute the scorecard input. Without per-candidate dor
   /may6/chronic_6 numbers, recommending which 6–9 of 22 candidates
   deserve GPU spend would be guesswork. The blocking constraint is
   network egress, not local compute.

## Outputs

- `outputs/per_ckpt_cohort_scores.csv` — per-frame score matrix, 952 × {P8A, E2B, T3_S1_step1500}. Append-target as candidates land.
- `outputs/per_ckpt_cohort_stats.csv` — per-ckpt × per-cohort {mean, p25, p50, p75, p90, std}.
- `outputs/per_ckpt_deployment_fpr.csv` — per-ckpt × per-cohort × τ ∈ {0.5, 0.7, 0.9, 0.92} → n_above_tau.
- `outputs/per_ckpt_inv_mean.csv` — per-ckpt × {full, dev, lockbox, chronic_6, non_chronic} × {forgery_auc, mean_shortcut_auc, inv_mean, per-axis AUCs}.
- `outputs/L11_inv_mean_summary.csv` — full-triptych inv_mean sorted desc (only the 4 cached ckpts).

## Re-run instructions

```
# To resume when network recovers:
cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training
# 1. Re-launch downloads for missing ckpts (any subset of 22)
gcloud storage cp gs://training-job-outputs/best_checkpoints/{smmcn6tj,c2ju9fbn,jrlldtem}/*.pth \
    analysis/cpu_diagnostics_2026-05-11_t67_t5c_probe/_ckpts/{t6,t7,t5c}/
# 2. Re-run the scorer (idempotent — picks up only new ckpts)
python3 analysis/cpu_diagnostics_2026-05-11_t67_t5c_probe/run_score_incremental.py \
    --poll_interval_sec 30 --max_wait_sec 4500 --delete_after_score
# 3. Compute L11 features for each ckpt + extend the inv_mean table
python3 analysis/cpu_diagnostics_2026-05-11_t67_t5c_probe/extract_features_l11.py
python3 analysis/cpu_diagnostics_2026-05-11_t67_t5c_probe/compute_inv_mean.py
```
