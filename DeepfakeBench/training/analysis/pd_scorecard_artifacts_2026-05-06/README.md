# PD packet · promotion-contract scorecard artifacts (2026-05-06)

Self-contained snapshot of the PD (correlation-penalty) packet's Teams promotion-contract scorecard. Hand this directory to a fresh agent for analysis — start with `unified_scorecard_simple.csv`.

## Quick start

| If you want | Open this |
|---|---|
| Full headline numbers (recall / FPR / accuracy at τ=0.5 per suite × ckpt) | `unified_scorecard_simple.csv` (232 rows) |
| Vertex-side rolled-up scorecard (10 suites covered) | `scorecards_resume_only/scorecard.csv` |
| Per-video predictions for any cell | `reports_{original,resume}/<suite>_<ckpt>_videos_report.csv` |
| Which run a given cell came from | `coverage_manifest.csv` |
| Packet design + close criterion | `../../docs/packet_retrospectives/packets/PD.md` |
| Lever-class context | `../../docs/packet_retrospectives/threads/correlation_penalty_loss.md` |

## What this scorecard answers / doesn't answer

**Answers** (from these files, no further work):
- F1-style **Teams headline numbers** (real_FPR / fake_recall at τ=0.5) for each (ckpt × suite). Reads via `unified_scorecard_simple.csv`.
- **Per-video** breakdowns for digging into which identities / sessions a ckpt fails on. Reads via the `reports_*/`*`_videos_report.csv` per cell.
- **Cross-ckpt deltas** for the 8 ckpts in scope (P8A, E2B, 3 PD-deeplive, 3 PD-viso) on each of the 29 contract suites.

**Does NOT answer** (requires separate work — these aren't in this scorecard's scope):
- **F1 lockbox** for `visomaster_enhanced_macro` / `deeplive_enhanced` / `teams_real_dor` — those lockbox suites are NOT in this scorecard's manifest. Only the Teams real/fake/lighting/poor-quality lockbox mirrors are present.
- **F2 (shortcut weakening ≥ 30% on ≥ 2 of 5 axes)** — needs the 5-axis correlation audit, which runs offline against `analysis/pa_pc_eval_2026-05-05/raw_reports/`-style features. Phase 1 baseline is in `../deeplive_viso_corr_eval_2026-05-06/abs_pearson_summary.csv`. Phase 2 (these PD ckpts vs that baseline) has not been run yet.
- **F3 (no untargeted axis +50%)** — same data needed as F2.
- **F4 (HDTF cross-substrate FPR ≤ 5%)** — HDTF substrate is in `arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml`, not this scorecard.

## Why two report directories

The scorecard ran in two phases:

1. **Original run** (`pd-corr-penalty-scorecard-2026-05-06`, Vertex job 8207399447131324416): launched 2026-05-06T11:09:32Z, **FAILED** at 16:27:59Z due to boot-disk exhaustion (disk-leak: per-cell ckpt cache filenames are unique per-suite, so 157 × ~896 MB filled the default 200 GB boot disk). Crashed before writing any scorecard rollup. Wrote per-cell reports for **157 cells** across 19 fully-covered suites + 1 partial suite (`teams_flat_xiang_xiang2_feng_dev`, 5 of 8 ckpts).

2. **Resume run** (`pd-corr-penalty-scorecard-2026-05-06-resume`, Vertex job 4910131201198522368): launched 2026-05-06T17:04:31Z, **SUCCEEDED** at 18:10:38Z (1h6m). Boot disk bumped to 500 GB via `--yaml-template /tmp/vertex_job_template_500gb.yaml` override. Re-ran the 1 partial suite + 9 missing per-capture-mode mini-slices = **80 cells across 10 suites**. Wrote scorecard rollups for those 10 suites.

Combined: **232 unique (suite × ckpt) cells = full 8 × 29 contract coverage**.

## Files

```
pd_scorecard_artifacts_2026-05-06/
├── README.md                          ← you are here
├── unified_scorecard_simple.csv       ← 232 rows; merged rollup; START HERE
├── coverage_manifest.csv              ← per-cell provenance + artifact completeness
├── scorecards_resume_only/            ← cloud-side rollups (resume run only, 10 suites)
│   ├── scorecard.csv                  ← 80 rows; same rows as unified[run==resume] but with extra metadata cols
│   ├── scorecard.wide.csv
│   ├── scorecard.int8_delta.csv
│   └── scorecard.json
├── reports_original/                  ← 628 files / 37 MiB (157 cells × 4 artifacts each)
└── reports_resume/                    ← 320 files / 2.4 MiB (80 cells × 4 artifacts; 3 cells partial — see below)
```

Per-cell artifact types (each cell writes 4 of these):
- `<suite>_<ckpt>_summary_report.txt` — human-readable per-cell summary
- `<suite>_<ckpt>_frames_report.csv` — frame-level scores
- `<suite>_<ckpt>_group_metrics.csv` — group-level rollup (within the cell)
- `<suite>_<ckpt>_videos_report.csv` — per-video labels + scores + correctness (this is the rollup input)

## Schema — `unified_scorecard_simple.csv`

| col | meaning |
|---|---|
| `suite` | Test-suite name (e.g. `teams_real_all_dev`, `visomaster_enhanced_macro_dev`) |
| `ckpt` | Checkpoint key, lowercase (e.g. `p8a_reference_step5000`) |
| `run` | `original` or `resume` — which dir the underlying report came from |
| `n_videos` | Total videos in the cell |
| `n_real` / `n_fake` | Per-label counts |
| `accuracy_at_0p5` | Accuracy with score threshold τ=0.5 |
| `real_fpr_at_0p5` | False-positive rate on real-label videos at τ=0.5 |
| `fake_recall_at_0p5` | Recall on fake-label videos at τ=0.5 |

These are computed by re-aggregating `videos_report.csv` for each cell (or, for the 3 cells where `videos_report.csv` failed to write, taken from the resume `scorecard.csv` confusion-matrix columns directly).

`accuracy_at_0p5` and friends use the threshold τ=0.5 — this is **diagnostic-only** per the launcher (`launch_target_domain_scorecard.sh` prints "these scorecard artifacts are diagnostic-only because they use threshold 0.5"). The deployment threshold is set elsewhere (per-substrate τ-calibration via `arena/launch_teams_promotion_contract.sh`).

## Caveats

1. **Threshold τ=0.5 is diagnostic, not deployment-grade.** For deployment-style numbers, the promotion-contract calibration pipeline (separate from this scorecard) is the authoritative path.
2. **7 cells are partial** (have <4 of 4 artifact types — see `coverage_manifest.csv` columns `has_summary`/`has_frames`/`has_group_metrics`/`has_videos`). Of these, 3 lack `videos_report.csv` but ARE represented in the resume `scorecard.csv` and so flow into `unified_scorecard_simple.csv` correctly.
3. **`teams_flat_xiang_xiang2_feng_dev`** appears in both report dirs — the resume's 8 cells overwrite the original's 5. The unified manifest reflects this de-dup (resume wins). The 5 cells in the original dir for this suite are obsolete; safe to ignore.
4. **The resume's `scorecard.csv` has richer columns** (checkpoint_path, label_mode, split_hint, etc.) for the 10 resume suites. If you need those for the original-run suites, you'd need to either re-derive from per-cell reports or rerun the rollup script over both dirs.

## Checkpoint roster (8 ckpts in scope)

| ckpt key | path | role |
|---|---|---|
| `p8a_reference_step5000` | `gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth` | Production anchor / comparison frame |
| `e2b_top_n_step3200` | `gs://training-job-outputs/best_checkpoints/rmat8lwx/top_n_effort_20260503_step3200_auc0.9863_eer0.0310.pth` | FT base for both PD arms; deployment-equivalent (Pearson r=+1.000 vs production) |
| `deeplive_corr_top_n_step4800` | `gs://training-job-outputs/best_checkpoints/8jgyw1am/top_n_effort_20260506_step4800_auc0.9970_eer0.0113.pth` | PD deeplive arm; highest-AUC ckpt |
| `deeplive_corr_top_n_step1800` | `gs://training-job-outputs/best_checkpoints/8jgyw1am/top_n_effort_20260505_step1800_auc0.9962_eer0.0169.pth` | PD deeplive arm; mid-trajectory |
| `deeplive_corr_periodic_step2000` | `gs://training-job-outputs/best_checkpoints/8jgyw1am/periodic_effort_20260505_step2000_auc0.9939_eer0.0169.pth` | PD deeplive arm; trajectory anchor |
| `viso_corr_top_n_step600` | `gs://training-job-outputs/best_checkpoints/7u3zc5zt/top_n_effort_20260506_step600_auc0.9934_eer0.0273.pth` | PD viso arm; highest-AUC (earliest) |
| `viso_corr_periodic_step2000` | `gs://training-job-outputs/best_checkpoints/7u3zc5zt/periodic_effort_20260506_step2000_auc0.9917_eer0.0182.pth` | PD viso arm; latest periodic (most "fully cooked") |
| `viso_corr_periodic_step1000` | `gs://training-job-outputs/best_checkpoints/7u3zc5zt/periodic_effort_20260506_step1000_auc0.9910_eer0.0228.pth` | PD viso arm; mid-trajectory |

**Both PD arms FT from `e2b_top_n_step3200` with `correlation_penalty(λ=1.0)` on three axes**: `sharpness_laplacian`, `luma_mean`, `face_area_fraction`. Single-lever discipline (no anchor_aware / face_scale_jitter / arcface_head). The arms differ only in whether visomaster training data is enabled (deeplive arm: disabled; viso arm: enabled at fw=4.0 mirroring PA).

## Suite roster (29 suites)

From `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml`:

- 6 dev suites covering all method families: `teams_real_all_dev`, `teams_real_poor_quality_dev`, `teams_real_lighting_extreme_dev`, `teams_fake_all_dev`, `visomaster_enhanced_macro_dev`, `deeplive_enhanced_dev`
- 4 Teams lockbox mirrors: `teams_real_all_lockbox`, `teams_fake_all_lockbox`, `teams_real_poor_quality_lockbox`, `teams_real_lighting_extreme_lockbox`
- 1 Dor diagnostic dev: `teams_real_dor_dev`
- 18 per-capture-mode dev mini-slices (one per Teams capture session): `teams_capture_<participant>_*_dev`, `teams_flat_xiang_xiang2_feng_dev`

## Recommended ways to use this for the next agent

1. **Headline read**: open `unified_scorecard_simple.csv`, pivot by `(suite, ckpt)` for `fake_recall_at_0p5` and `real_fpr_at_0p5`. Compare PD-arm ckpts against the `e2b_top_n_step3200` row per suite to answer "did corr-penalty beat its FT base?".
2. **Per-identity FP digging**: for any suite where `real_fpr_at_0p5` looks high, open the corresponding `videos_report.csv` and group by `video_id` substring (Teams sessions have predictable naming).
3. **Cross-substrate comparison**: P8A is the production anchor; its row per suite is the comparison frame.
4. **F2/F3 audit**: would need to extend `analysis/deeplive_viso_corr_eval_2026-05-06/run_audit.py`'s Phase 2 block (currently commented out per `packets/PD.md` §171-172) and run against these `frames_report.csv` files. Not in this directory's scope.

## Provenance

- Original Vertex job: `8207399447131324416` (us-east1, image 1.3.267, FAILED 2026-05-06T16:27:59Z)
- Resume Vertex job: `4910131201198522368` (us-east1, image 1.3.268 with new in-image suite YAML, SUCCEEDED 2026-05-06T18:10:38Z)
- Resume launcher: bypassed `arena/launch_target_domain_scorecard.sh` and called `scripts/launch/launch_experiment_jobs.sh` directly with `--yaml-template /tmp/vertex_job_template_500gb.yaml` (boot disk override 200 → 500 GB)
- Resume suite manifest (in image 1.3.268): `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor_resume_2026-05-06.yaml`
- Checkpoint map (unchanged across both runs): `arena/checkpoint_maps/teams_target_domain.deeplive_viso_corr_2026-05-06.yaml`
