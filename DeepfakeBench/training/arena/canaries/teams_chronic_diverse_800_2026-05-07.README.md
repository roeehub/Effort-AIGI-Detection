# teams_chronic_diverse_800 canary (2026-05-07)

In-training monitoring canary for the 3-slot from-scratch GPU experiment
(P2 packet, launch 2026-05-07 PM). Forward-passed every 1000 steps.

- Parquet: `arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet`
- Total frames: 800 (target 800)
- Reals: 600 | Fakes: 200
- Sampling seed: 42 (deterministic)

## Source CSVs

All `p8a_reference_score` values are P8A_REFERENCE_STEP5000 `frame_prob`
(per-frame `prob_fake`-equivalent column in the Phase A/C reports).

- Phase A (per-suite frame reports):
  `analysis/p1_pe_eval_2026-05-07/raw_reports/phase_a/<suite>_p8a_reference_step5000_frames_report.csv`
- Phase C (HDTF / proper_real_clean_lockbox):
  `analysis/p1_pe_eval_2026-05-07/raw_reports/phase_c/proper_real_clean_lockbox_p8a_reference_step5000_frames_report.csv`

## Composition

| cohort | target_n | actual_n | delta | p8a_mean | p8a_p50 | p8a_p95 | source_suite |
|---|---:|---:|---:|---:|---:|---:|---|
| chronic_PCGen_s22 | 50 | 50 | +0 | 0.9191 | 0.9810 | 0.9943 | teams_real_all_dev |
| chronic_PCGen_s45 | 50 | 50 | +0 | 0.6752 | 0.7867 | 0.9784 | teams_real_all_dev |
| chronic_Q_s6 | 50 | 50 | +0 | 0.9279 | 0.9911 | 0.9941 | teams_real_all_dev |
| chronic_bla_bla_chow | 50 | 50 | +0 | 0.1021 | 0.0277 | 0.4565 | teams_real_all_dev |
| chronic_bla_bla_chow_s2 | 50 | 50 | +0 | 0.4044 | 0.3152 | 0.9781 | teams_real_all_dev |
| chronic_Roy_D | 50 | 50 | +0 | 0.6060 | 0.7772 | 0.9924 | teams_real_all_dev |
| healthy_test_cam | 50 | 50 | +0 | 0.1038 | 0.0057 | 0.8097 | teams_real_all_dev |
| healthy_md_noyn_sharker | 50 | 50 | +0 | 0.0525 | 0.0075 | 0.1888 | teams_real_all_dev |
| healthy_xiang_xiang2_feng | 50 | 50 | +0 | 0.0136 | 0.0060 | 0.0521 | teams_real_all_dev |
| healthy_dor | 50 | 50 | +0 | 0.0198 | 0.0057 | 0.0282 | teams_real_all_dev |
| healthy_dor_shkedi | 50 | 50 | +0 | 0.3310 | 0.1810 | 0.9550 | teams_real_dor_dev |
| hdtf_clean_real | 50 | 50 | +0 | 0.0068 | 0.0054 | 0.0132 | proper_real_clean_lockbox |
| lockbox_fake | 100 | 100 | +0 | 0.6291 | 0.7401 | 0.9946 | teams_fake_all_lockbox |
| viso_fake | 50 | 50 | +0 | 0.3728 | 0.2288 | 0.9647 | visomaster_enhanced_macro_dev |
| deeplive_fake | 50 | 50 | +0 | 0.4606 | 0.4135 | 0.9925 | deeplive_enhanced_dev |

## Cohort definitions

- **chronic_PCGen_s22 / chronic_PCGen_s45 / chronic_Q_s6 /
  chronic_bla_bla_chow / chronic_bla_bla_chow_s2 / chronic_Roy_D**:
  six chronic FP-tail identities from
  `analysis/p1_pe_eval_2026-05-07/JOINT_TAU_SWEEP_FACTS_2026-05-07.md` /
  Phase D `chronic_flag_definition.json`. Drawn from
  `teams_real_all_dev` via prefix-on-raw-video_id matching (mirrors
  `analysis/p1_pe_eval_2026-05-07/phase_d/run_chronic_filter.py`,
  `video_matches_cid`). `chronic_bla_bla_chow` excludes
  `bla_bla_chow__s2` rows so the cohorts are disjoint.

- **healthy_test_cam, healthy_md_noyn_sharker,
  healthy_xiang_xiang2_feng**: healthy diverse reals from
  `teams_real_all_dev`, prefix-matched.

- **healthy_dor**: from `teams_real_all_dev`, prefix `dor` with
  `dor_shkedi` excluded (since the prefix would otherwise match the
  shkedi rows).

- **healthy_dor_shkedi**: 50 frames from `teams_real_dor_dev` (the
  pre-built 50-frame dor suite). The full suite is taken if size
  permits.

- **hdtf_clean_real**: 50 frames from `proper_real_clean_lockbox`
  (HDTF-style cross-substrate reals, F4-relevant). Phase C report.

- **lockbox_fake**: 100 frames from `teams_fake_all_lockbox`
  (F1 deployment-relevant fakes), video_id-stratified.

- **viso_fake**: 50 frames from `visomaster_enhanced_macro_dev`
  (the structural ceiling cohort — refer to
  `project_viso_ceiling_unbroken_10_packets.md`).

- **deeplive_fake**: 50 frames from `deeplive_enhanced_dev`
  (deeplive method).

## Sampling rule

`stratified_video_sample(seed=42)`:
- If candidate set <= target: take all.
- Else: distribute the target evenly across video_ids
  (floor + remainder), top-up uniformly from leftovers if some videos
  had insufficient frames.

## Schema

| column | dtype | description |
|---|---|---|
| frame_idx | int64 | 0..n-1 |
| frame_path | str | gs:// URL into the existing storage substrate |
| label | int64 | 0 = real, 1 = fake |
| cohort | str | one of the 15 cohort tags above |
| base_identity | str | identity tag for grouping |
| suite | str | source suite name |
| p8a_reference_score | float64 | P8A_REFERENCE_STEP5000 `frame_prob` on this frame |

## Sanity

- Rows missing `frame_path` or with NaN `p8a_reference_score` are
  excluded by the build script.
- Build script: `arena/canaries/build_canary_2026-05-07.py`.
- Run wall-clock: < 30 s (CPU-only).
