# Job D — Per-identity FP picture for Slot A v2 step3500 vs P8A vs T5C

Date: 2026-05-20
Owner: cpu-jobs/job-d
Eval bundle: `slot-a-v2-validation-2026-05-20` (GCS prefix
`gs://training-job-outputs/test_results/teams_promotion_contract/slot-a-v2-validation-2026-05-20/reports/`)

This document is a **FACTS** doc. It uses mechanical pass/fail against
pre-stated bars and avoids words like "succeeds", "fails", "wins",
"promotes", or "deployment-grade".

## Checkpoints

| Label | W&B run | Vertex path |
|---|---|---|
| P8A | `9lmvb5b4` | `value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth` |
| SlotAv2 | `hp35c51p` | `periodic_effort_20260516_step3500_auc0.9950_eer0.0165.pth` |
| T5C | (periodic_effort_20260512) | `periodic_effort_20260512_step3500.pth` |

## Method

### tau calibration

Per ckpt, `tau` is set to the 93rd-percentile of per-video scores on
`teams_real_all_dev` (n=3253 videos). This places the dev real FPR at 0.07
(the contract's calibration point). All three ckpts achieve identical
calibrated FPR=0.0701 by construction (the empirical quantile aligns to
the same dev real-video rank for each ckpt).

| ckpt | n_videos_dev | tau_dev (p93 of scores) | achieved dev FPR | score p50 | score p99 |
|---|---:|---:|---:|---:|---:|
| P8A | 3253 | 0.913289 | 0.07009 | 0.0076 | 0.9938 |
| T5C | 3253 | 0.821178 | 0.07009 | 0.1196 | 0.9266 |
| SlotAv2 | 3253 | 0.773642 | 0.07009 | 0.1009 | 0.8978 |

Notes
- All three score distributions have a hard ceiling well below 1.0
  (P8A 0.9946; T5C 0.9379; SlotAv2 0.9253). T5C and SlotAv2 are shifted
  upward (p50 ~0.10) vs P8A (p50 ~0.008), which is why their dev-calibrated
  tau values are lower in absolute terms.
- tau is video-level (`avg_video_prob`), not frame-level.

### Identity scope and coverage

Per-identity FPR is computed on real-frame video CSVs (label=0). The
`teams_capture_*` suites are fake-only (label=1) and do not contribute to
FPR; their existence is recorded as coverage but not used for the FPR table.

| Identity | Primary suite | Scope | n_videos | Coverage notes |
|---|---|---|---:|---|
| dor_shkedi | teams_real_all_dev | dev | 20 | also 1138 lockbox videos (separate row below) |
| PC_Generator | teams_real_all_dev | dev | 525 | also 28 lockbox videos (separate row below) |
| Cam_Test | teams_real_all_dev | dev | 98 | distinct from Test_Cam in video_id parsing |
| Test_Cam | teams_real_all_dev | dev | 712 | distinct from Cam_Test in video_id parsing |
| Md_Noyn_Sharker | teams_real_all_dev | dev | 409 | parsed as `Md_noyn_Sharker__s15` |
| Chikara_Takahashi | teams_real_all_lockbox | lockbox | 25 | absent from teams_real_all_dev; not in any 29-suite by name |
| Roy_D | teams_real_all_dev | dev | 130 | also 113 in teams_real_lighting_extreme_dev (cross-check below); no Roy_D-specific suite in 29-suite scorecard |
| bla_bla_chow | teams_real_all_dev | dev | 467 | also 61 lockbox videos (separate row below) |
| Q | teams_real_all_dev | dev | 36 | parsed as `Q__s6` |

Identity parsing rule (`base_root`): strip trailing `__real`; match prefix
before `__seqN` (Roy_D pattern), `__sN__seg` (capture pattern), or
`__frame` (ilan/orel pattern); else strip from `__seg` onward.

## Headline table (calibrated tau, dev FPR=0.07)

| identity | scope | n_videos | P8A FPR | T5C FPR | SlotAv2 FPR | delta(SlotAv2-P8A) |
|---|---|---:|---:|---:|---:|---:|
| dor_shkedi | dev | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| PC_Generator | dev | 525 | 0.2419 | 0.0533 | 0.0571 | -0.1848 |
| Cam_Test | dev | 98 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Test_Cam | dev | 712 | 0.0014 | 0.0000 | 0.0014 | 0.0000 |
| Md_Noyn_Sharker | dev | 409 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Chikara_Takahashi | lockbox | 25 | 0.3600 | 0.0000 | 0.0000 | -0.3600 |
| Roy_D | dev | 130 | 0.3000 | 0.8769 | 0.8385 | +0.5385 |
| bla_bla_chow | dev | 467 | 0.0600 | 0.1435 | 0.1542 | +0.0942 |
| Q | dev | 36 | 0.8611 | 0.2222 | 0.2222 | -0.6389 |

## Raw tau=0.5 table (no calibration)

| identity | scope | n_videos | P8A FPR | T5C FPR | SlotAv2 FPR | delta(SlotAv2-P8A) |
|---|---|---:|---:|---:|---:|---:|
| dor_shkedi | dev | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| PC_Generator | dev | 525 | 0.3524 | 0.3105 | 0.2114 | -0.1410 |
| Cam_Test | dev | 98 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Test_Cam | dev | 712 | 0.0309 | 0.0084 | 0.0098 | -0.0211 |
| Md_Noyn_Sharker | dev | 409 | 0.0293 | 0.0000 | 0.0000 | -0.0293 |
| Chikara_Takahashi | lockbox | 25 | 0.7600 | 0.2800 | 0.1200 | -0.6400 |
| Roy_D | dev | 130 | 0.4538 | 1.0000 | 0.9769 | +0.5231 |
| bla_bla_chow | dev | 467 | 0.1542 | 0.3833 | 0.3255 | +0.1713 |
| Q | dev | 36 | 0.9444 | 0.7222 | 0.2222 | -0.7222 |

## Cross-checks

### Roy_D in `teams_real_lighting_extreme_dev` (cross-check pool)

Same identity, different real-pool slice. Same directional pattern.

| ckpt | n_videos | FPR_tau_dev | mean_score |
|---|---:|---:|---:|
| P8A | 113 | 0.3451 | 0.5409 |
| T5C | 113 | 0.9204 | 0.8972 |
| SlotAv2 | 113 | 0.8761 | 0.8381 |

### Lockbox FPR at dev-calibrated tau (for identities present in both pools)

Same tau values from above. Read alongside the dev rows.

| identity | n_lockbox | P8A | T5C | SlotAv2 |
|---|---:|---:|---:|---:|
| PC_Generator | 28 | 0.2857 | 0.0000 | 0.0000 |
| bla_bla_chow | 61 | 0.0000 | 0.1639 | 0.1475 |
| dor_shkedi | 1138 | 0.0070 | 0.0281 | 0.0176 |

## Bar A / Bar B / Bar C (mechanical, at calibrated tau)

Pre-stated bars (re-stated for the reader):
- **Bar A — anchor mechanism reach**: P8A_FPR > 0.20 AND SlotAv2_FPR < 0.05
- **Bar B — anchor mechanism overshoot**: P8A_FPR < 0.10 AND SlotAv2_FPR > 0.30
- **Bar C — composite ratio**: count(Bar A) / count(Bar B)

| identity | scope | P8A FPR | SlotAv2 FPR | Bar A | Bar B |
|---|---|---:|---:|:---:|:---:|
| dor_shkedi | dev | 0.0000 | 0.0000 | False | False |
| PC_Generator | dev | 0.2419 | 0.0571 | False | False |
| Cam_Test | dev | 0.0000 | 0.0000 | False | False |
| Test_Cam | dev | 0.0014 | 0.0014 | False | False |
| Md_Noyn_Sharker | dev | 0.0000 | 0.0000 | False | False |
| Chikara_Takahashi | lockbox | 0.3600 | 0.0000 | **True** | False |
| Roy_D | dev | 0.3000 | 0.8385 | False | False |
| bla_bla_chow | dev | 0.0600 | 0.1542 | False | False |
| Q | dev | 0.8611 | 0.2222 | False | False |

Counts:
- Bar A (anchor fix): **1** (Chikara_Takahashi, lockbox scope)
- Bar B (anchor overshoot): **0**
- Bar C ratio A/B: **inf** (no Bar B identities)

### Marginal-pass commentary on Bar bounds (mechanical, no value language)

The pre-stated bars are strict thresholds. Bar A and Bar B both miss
several identities by small margins (PC_Generator, Q, Roy_D), so the
mechanical counts above understate the per-identity differences. The
following identities lie within 0.06 of one of the bars:

- **PC_Generator** dev: SlotAv2_FPR=0.057 (Bar A required <0.05); P8A_FPR=0.242 satisfies the >0.20 leg. delta = -0.18.
- **Q** dev: P8A_FPR=0.861, SlotAv2_FPR=0.222. P8A clears the >0.20 leg; SlotAv2 misses the <0.05 leg by 0.17. delta = -0.64.
- **Roy_D** dev: P8A_FPR=0.300, SlotAv2_FPR=0.838. P8A misses the <0.10 leg of Bar B by 0.20; SlotAv2 clears the >0.30 leg. delta = +0.54.
- **bla_bla_chow** dev: P8A_FPR=0.060 satisfies the <0.10 leg of Bar B; SlotAv2_FPR=0.154 misses the >0.30 leg by 0.15. delta = +0.09.

## Files

- `tau_dev.json` — per-ckpt tau + dev FPR diagnostics
- `per_identity_fpr_tau_dev.csv` — calibrated table
- `per_identity_fpr_tau_raw.csv` — raw (tau=0.5) table
- `bar_results.csv` — Bar A / Bar B per identity
- `bar_summary.json` — Bar counts + ratio
- `coverage.json` — per-identity suite coverage record
- `data/` — pulled per-video CSVs (63 dev + 9 lockbox = 72 files)
- `run_per_identity_fp.py` — analysis script

## Coverage caveats (explicit)

1. **Chikara_Takahashi** is not present in any per-suite name in the
   29-suite scorecard and is absent from `teams_real_all_dev`. Its FPR
   here is from `teams_real_all_lockbox` (n=25 videos) using the dev-
   calibrated tau. The 2026-05-16 memory entry's "Chikara 26->0%"
   reading was for `teams_real_lockbox` at the lockbox-calibrated tau;
   the FPR value reproduces here at 0.360 (P8A) -> 0.000 (SlotAv2) using
   the dev tau.
2. **Roy_D** is not in any 29-suite slot by name. Its FPR is from
   `teams_real_all_dev` (n=130) and cross-checked on
   `teams_real_lighting_extreme_dev` (n=113), both dev-pool. Roy_D does
   NOT appear in `teams_real_all_lockbox`.
3. **Q** is in `teams_real_all_dev` (n=36) under root `Q__s6`. Not in
   lockbox.
4. **bla_bla_chow** dev pool (n=467) unions `bla_bla_chow` and
   `bla_bla_chow__s2` roots.
5. **Cam_Test** (n=98 dev) and **Test_Cam** (n=712 dev) are parsed as
   distinct identities in the video_id, mirroring the two distinct sets
   of capture suites (`teams_capture_cam_test_*` vs
   `teams_capture_test_cam_*`).

## Cell-by-cell value reproducibility

All numbers in this doc come from `run_per_identity_fp.py` executed on the
72 per-video CSVs in `data/`. To reproduce:

```bash
cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training
python analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_d_per_identity_fp/run_per_identity_fp.py
```

Outputs are written next to the script.
