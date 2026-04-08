# Smoke-Running Agent Handoff

**Date:** April 6, 2026  
**Purpose:** historical handoff created while the first Track A smoke run was still active  
**Scope:** Teams target-domain upgrade, Track A smoke gate, and Track C/S2 split-freeze work

> Historical note:
> This document was originally written before the Track A smoke resolved.
> The smoke later finished `JOB_STATE_SUCCEEDED`.
> Current source-of-truth for the post-smoke state is:
> - `DeepfakeBench/training/docs/TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md`
> - `DeepfakeBench/training/docs/TRACK_A_HANDOFF_2026-04-06.md`
> - `DeepfakeBench/training/docs/TRACK_A_RUNTIME_RUNBOOK_2026-04-06.md`

## 1. Historical Smoke Status Snapshot

Historical tracked smoke job:

- display name: `exp-R13_SMOKE_trackA_teams_enhanced-20260406-123308`
- job id: `3841428920024956928`
- region: `asia-southeast1`
- final state now known: `JOB_STATE_SUCCEEDED`
- create time (UTC): `2026-04-06T10:33:13.444938Z`
- start time (UTC): `2026-04-06T10:41:02Z`
- end time (UTC): `2026-04-06T13:08:48Z`
- final update time (UTC): `2026-04-06T13:08:55.503508Z`

Historical rule at the time:

- do **not** relaunch this smoke while it is still running
- do **not** launch the full-length Track A run until this smoke passes
- do **not** run arena for Track A until a real checkpoint exists

Monitoring commands:

```bash
gcloud ai custom-jobs describe 3841428920024956928 \
  --project=train-cvit2 \
  --region=asia-southeast1

gcloud ai custom-jobs stream-logs 3841428920024956928 \
  --project=train-cvit2 \
  --region=asia-southeast1
```

## 2. Read Order For A New Agent

For current work, prefer the updated post-smoke docs listed above.

Read these in this order. The first three are the active source-of-truth docs.

### Required Source-Of-Truth Docs

1. `DeepfakeBench/training/docs/TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md`
   - Master plan and current status tracker.
   - Read this first for:
     - `S0` / `S1` / `S2` status
     - Track A status
     - Track C/S2 implementation snapshot
     - execution order and non-negotiable checks
   - This is the best single document for "what is done, what is waiting, what is next".

2. `DeepfakeBench/training/docs/TRACK_A_HANDOFF_2026-04-06.md`
   - Narrow Track A handoff.
   - Read this for:
     - active smoke job identity
     - the exact Track A go/no-go gate
     - what not to spend time on while the smoke is still active

3. `DeepfakeBench/training/docs/TRACK_A_RUNTIME_RUNBOOK_2026-04-06.md`
   - Operational runbook for what happens after the smoke passes or fails.
   - Read this for:
     - full-length launch command
     - arena command caveat
     - checkpoint discovery steps
     - post-smoke operational sequence

### Background / Context Docs

4. `DeepfakeBench/training/experiments/phase2_round13/R13_PLAN.md`
   - Historical Round 13 matrix and original rationale.
   - Use this as background only.
   - Important caveat:
     - the currently running smoke is a focused Track A slice, not a full return to the whole original R13 matrix

5. `DeepfakeBench/training/docs/VISOMASTER_ENHANCED_INTEGRATION.md`
   - Older design document for integrating clean enhanced VisoMaster data.
   - Still useful for:
     - family taxonomy
     - augmentation intent
     - historical config choices
   - But it is **not** the final source-of-truth for the new resolver-driven Teams-enhanced path.

6. `DeepfakeBench/training/docs/ENHANCED_TEAMS_GAP_REPORT.md`
   - Motivation document for why enhanced+Teams is a real gap.
   - Read this to understand:
     - why clean enhanced success did not solve Teams passthrough
     - why real Teams-like evaluation and data matter
   - This is problem framing, not an execution doc.

## 3. Non-Doc Artifacts The Next Agent Will Likely Touch

These are not docs, but they are the active configs and templates tied to the work.

### Track A Artifacts

- `DeepfakeBench/training/experiments/phase2_round13/R13_SMOKE_trackA_teams_enhanced.yaml`
  - current smoke config
- `DeepfakeBench/training/experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml`
  - full-length Track A config to launch only after smoke success
- `DeepfakeBench/training/arena/arena_config.track_a.yaml`
  - one-off arena config for the first Track A checkpoint

### Track C / S2 Artifacts

- `DeepfakeBench/training/arena/build_teams_target_domain_manifest.py`
  - builder for the frozen Teams target-domain manifest
- `DeepfakeBench/training/arena/target_domain_suites.teams_manifest.template.yaml`
  - suite template for manifest-based target-domain validation
- `DeepfakeBench/training/arena/prefix_rules.teams_manifest.template.yaml`
  - template for explicit fake-family provenance overrides
- `DeepfakeBench/training/arena/manifests/teams_target_domain_manifest_2026-04-06_frozen.json`
  - first frozen Teams target-domain manifest artifact
- `DeepfakeBench/training/arena/visualize_teams_target_domain_manifest.py`
  - visual audit renderer for frozen manifest rows
- `DeepfakeBench/training/arena/visual_audits/teams_target_domain_manifest_2026-04-06_frozen/index.html`
  - first rendered visual audit gallery for the frozen manifest

### Track A Code Files

- `DeepfakeBench/training/data/sources/visomaster.py`
- `DeepfakeBench/training/data/sources/combined_paired.py`
- `DeepfakeBench/training/tests/test_phase4_family_pipeline.py`

### Track C / S2 Code Files

- `DeepfakeBench/training/data/validation_sources.py`
- `DeepfakeBench/training/validate_custom_sources.py`
- `DeepfakeBench/training/arena/run_target_domain_validation_sequential.py`
- `DeepfakeBench/training/tests/test_validation_external_grouping.py`
- `DeepfakeBench/training/tests/test_target_domain_manifest_builder.py`

## 4. What Is Already Done

### S1 Resolver Audit

Completed and staged:

- local artifacts:
  - `/tmp/enhanced_visomaster_resolver_2026-04-06.json`
  - `/tmp/enhanced_visomaster_resolver_2026-04-06.csv`
- staged GCS artifacts:
  - `gs://training-job-outputs/cache/visomaster/enhanced_visomaster_resolver_2026-04-06.json`
  - `gs://training-job-outputs/cache/visomaster/enhanced_visomaster_resolver_2026-04-06.csv`

Key verified facts:

- `999` enhanced base `sample_id`s audited
- `54` resolve to full `teams_v2` companion
- `943` resolve to `clean_fallback`
- `2` remain unresolved and are excluded

### Track A Core Loader

Implemented and tested:

- new source block: `combined_paired.visomaster_teams_enhanced`
- one merged sample per base `sample_id`
- no 8x enhancer sample explosion
- branch-time choice between:
  - original fake branch
  - one enhanced fake branch
- default branch policy:
  - `p_original = 0.5`
  - otherwise choose one available enhancer uniformly

Targeted verification already run:

```bash
python -m py_compile \
  DeepfakeBench/training/data/sources/visomaster.py \
  DeepfakeBench/training/data/sources/combined_paired.py \
  DeepfakeBench/training/tests/test_phase4_family_pipeline.py

pytest DeepfakeBench/training/tests/test_phase4_family_pipeline.py \
  -k "group_key_mapping_representative_cases or visomaster_teams_enhanced or visomaster_enhanced"
```

Observed result:

- `9 passed`
- `0 failed`

### Track C / S2 Tooling

Implemented and tested:

- manifest-backed validation loading
- split/slice filtering for target-domain manifests
- sequential runner support for manifest suites
- first-pass Teams manifest builder
- suite template and prefix-rule template

Targeted verification already run:

```bash
python -m py_compile \
  DeepfakeBench/training/data/validation_sources.py \
  DeepfakeBench/training/validate_custom_sources.py \
  DeepfakeBench/training/arena/run_target_domain_validation_sequential.py \
  DeepfakeBench/training/arena/build_teams_target_domain_manifest.py \
  DeepfakeBench/training/tests/test_validation_external_grouping.py \
  DeepfakeBench/training/tests/test_target_domain_manifest_builder.py

pytest DeepfakeBench/training/tests/test_validation_external_grouping.py \
  DeepfakeBench/training/tests/test_target_domain_manifest_builder.py
```

Observed result:

- `11 passed`
- `0 failed`

Live discovery probe against `gs://teams-faces-data-test-2914-fake-4420-real-feb-28`:

- grouped videos discovered: `7,276`
- `4,614` real
- `2,662` fake
- the bucket is heterogeneous and contains both:
  - session-capture style names
  - flat upload style names

## 5. TODO For Roee On The Teams Evaluation Freeze

The resolver-driven Teams-enhanced VisoMaster smoke result is now separate work and should be handled in another chat. The next open task in this document is the frozen Teams target-domain scorecard lane.

### Current State

- Frozen manifest built:
  - `DeepfakeBench/training/arena/manifests/teams_target_domain_manifest_2026-04-06_frozen.json`
- Visual audit gallery built:
  - `DeepfakeBench/training/arena/visual_audits/teams_target_domain_manifest_2026-04-06_frozen/index.html`
- Visual audit selection metadata built:
  - `DeepfakeBench/training/arena/visual_audits/teams_target_domain_manifest_2026-04-06_frozen/selection.json`
- Frozen manifest summary:
  - `7,276` grouped videos total
  - `4,614` real
  - `2,662` fake
  - `dev`: `5,662`
  - `lockbox`: `1,614`
  - `teams_fake_unknown`: `0`

### TODO For Roee

1. Open `DeepfakeBench/training/arena/visual_audits/teams_target_domain_manifest_2026-04-06_frozen/index.html` and visually verify the key slices:
   - `teams_real_poor_quality`
   - `teams_real_lighting_extreme`
   - `visomaster_enhanced_macro`
   - `deeplive_enhanced`
   - `teams_capture_cam_test`
   - `teams_capture_pc_generator`
   - `teams_capture_test_cam`
   - `teams_capture_noyn_sharker`
   - `teams_capture_dor_shkedi`
   - `teams_flat_xiang_xiang2_feng`
2. If the gallery looks correct, create a concrete suite manifest from `DeepfakeBench/training/arena/target_domain_suites.teams_manifest.template.yaml` by replacing every `<manifest-path>` with:
   - `DeepfakeBench/training/arena/manifests/teams_target_domain_manifest_2026-04-06_frozen.json`
   - If the scorecard will run outside the local machine, upload the frozen manifest to GCS first and use the `gs://...` path instead.
3. Run a dry run of the sequential target-domain validation scorecard:

   ```bash
   python DeepfakeBench/training/arena/run_target_domain_validation_sequential.py \
     --checkpoints FT7 \
     --checkpoint_map <checkpoint-map.json> \
     --suite_manifest <teams-manifest-suite.yaml> \
     --dry-run
   ```

4. If the dry run is clean, run the real sequential scorecard without `--dry-run`.
5. Only treat the Teams evaluation split as frozen after any visual mislabels or slice-rule issues found in step 1 are corrected.

### Why This Is The Right Next Step

- the frozen Teams target-domain manifest now exists and is ready for evaluation
- the visual audit is already rendered, so slice semantics can be checked before scorecarding
- the next missing artifact is the manifest-based target-domain scorecard, not more bucket plumbing

## 6. What Must Wait For The Smoke Result

Do not do these while the current smoke is still unresolved:

- relaunch the smoke job
- launch `R13_A_trackA_teams_enhanced.yaml`
- run arena for Track A
- tune `p_original`
- retune Track A family weights
- reopen S1 resolver auditing from scratch
- change Teams simulation / Track B scope

## 7. Pass / Fail Branch After The Smoke Resolves

### If The Smoke Passes

1. Launch the full-length Track A baseline:
   - `DeepfakeBench/training/experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml`
2. Wait for the first real checkpoint.
3. Update:
   - `DeepfakeBench/training/arena/arena_config.track_a.yaml`
4. Run arena using checkpoint alias `TRACK_A_CANDIDATE`, not a raw `gs://...pth` URI.
5. Continue Track C scorecard work in parallel.

### If The Smoke Fails

1. Fix the concrete runtime issue only.
2. Re-run the smoke.
3. Do **not** launch the full-length Track A run yet.

## 8. Important Caveats

### Arena CLI Caveat

`./launch_arena.sh --checkpoints ...` filters checkpoint **names** already defined in the arena config. It does **not** accept a raw checkpoint URI directly.

### Fake Provenance Caveat

The flat Teams fake bucket does not fully encode source-family provenance in filenames. The current builder can auto-tag only a few safe prefixes by default. The bundled `prefix_rules.teams_manifest.template.yaml` now freezes the remaining discovered fake lanes to safe provenance slices, but canonical family labels (`deeplive_regular`, `visomaster_original_macro`, etc.) still require stronger provenance evidence before they should replace those session-level labels.

### Git Visibility Caveat

`.gitignore` currently ignores `data/`, so Track A code under:

- `DeepfakeBench/training/data/sources/visomaster.py`
- `DeepfakeBench/training/data/sources/combined_paired.py`

may not appear in normal `git status` / `git diff` even though the code is on disk and tested.

## 9. Minimal Startup Checklist For The Next Agent

1. Read:
   - `TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md`
   - `TRACK_A_HANDOFF_2026-04-06.md`
   - `TRACK_A_RUNTIME_RUNBOOK_2026-04-06.md`
2. Confirm the smoke is still running and do not relaunch it.
3. Open:
   - `prefix_rules.teams_manifest.template.yaml`
   - `target_domain_suites.teams_manifest.template.yaml`
   - `build_teams_target_domain_manifest.py`
4. Continue Track C / S2 provenance freeze work.
5. Only branch back to Track A operational steps after the smoke resolves.
