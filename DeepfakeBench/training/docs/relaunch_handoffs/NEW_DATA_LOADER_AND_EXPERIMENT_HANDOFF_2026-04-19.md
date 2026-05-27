# New Data Loader And Experiment Handoff

**Date:** 2026-04-19  
**Branch:** `teams-relaunch-root-2026-04-17`  
**Head commit at handoff creation:** `7cde78c593d9e9aca654c9eacc90f08903b49b3f`  
**Relevant recent commits:**  
- `7cde78c` `Add fast WT-B startup smoke and cache priming`  
- `f9303eb` `Land WT-B runtime and launcher smoke readiness`

## Start Here

This handoff assumes you are working in the same local workspace and can read
the generated reports referenced below.

Before changing code:

1. read this file fully
2. read the four source-of-truth docs listed below
3. read the generated bucket reports listed below

Current ground truth:

- the new VisoMaster buckets are `proper_data`, not legacy `visomaster_hints`
  and not a `combined_paired` stopgap
- the training-side proper-data path now exists on
  `combined_paired.proper_data` and consumes the WT-F provisional inventory /
  manifest output without collapsing the data into legacy VisoMaster aliases
- the correct clean-versus-Teams row join key is `sample_id`
- Teams manifests are wrappers; semantic clean provenance lives under
  `original_manifest`
- the current Teams snapshot is usable but incomplete; per user report, roughly
  `30%` more Teams-propagated data is still expected to land
- any inventory, manifest, or suite generated before propagation finishes must
  be treated as provisional and regenerated later
- the provisional builder now fails fast if explicit `frame_files` are missing;
  it refuses to synthesize fallback frame names
- the current committed provisional artifacts were regenerated under a strict
  clean-and-Teams `16/16` contract, so the first packet no longer relies on
  ragged-clean residue
- the runtime loader still keeps anchor-index intersection as a defensive
  fallback if a future proper-data manifest admits ragged rows, but that is no
  longer the intended first-packet contract

Immediate next action if continuing implementation:

- use this handoff to plan the first experiment matrix now that both the WT-B
  smoke gate and the proper-data startup smoke have succeeded
- keep the new buckets on the explicit `proper_data` path; do not force them
  through `DeepfakeBench/training/data/sources/visomaster.py`
- before longer experiments, close out the current repo state in git and rerun
  the provisional WT-F artifact build after Teams propagation stabilizes,
  keeping the strict clean-and-Teams `16/16` contract unless we intentionally
  relax it later
- update this same file in place when you finish

## Update This File

This file is the working handoff for the next agent.

When the agent finishes, update **this same file in place** with:

- what bucket structures were received
- how each bucket was classified
- what code was changed
- what tests were run
- what still blocks launch
- the recommended first experiment packet

Do not create a second parallel handoff unless there is a hard blocker that
prevents updating this file.

## Why This Exists

We are converging on the first proper experiment packet for the relaunch.

Current priorities are:

1. finish the WT-B smoke gate cleanly
2. integrate the incoming new data buckets honestly
3. decide how the new data should enter training
4. draft the first experiment packet that combines WT-B with the new data

WT-B is no longer the design blocker. The loader proof boundary is now clean;
the remaining work is experiment planning against a still-provisional data
snapshot plus normal commit / regeneration closeout.

## Source Of Truth Docs

Read these first:

- `DeepfakeBench/training/docs/relaunch_handoffs/WT_B_AND_NEW_DATA_READINESS_2026-04-19.md`
- `DeepfakeBench/training/docs/R13_WTB_WEAK_SIGNAL_RUNBOOK_2026-04-18.md`
- `DeepfakeBench/training/docs/PROPER_DATA_SCHEMA_AND_FUTURE_MANIFESTS_2026-04-17.md`
- `DeepfakeBench/training/docs/relaunch_handoffs/WT-F_2026-04-17.md`

## Generated Artifacts For This Handoff

These were produced during the live bucket inspection and should be read
alongside the docs above:

- `DeepfakeBench/training/arena/inspect_visomaster_proper_buckets.py`
- `DeepfakeBench/training/tests/test_inspect_visomaster_proper_buckets.py`
- `DeepfakeBench/training/arena/reports/visomaster_proper_clean_bucket_census_2026-04-19.json`
- `DeepfakeBench/training/arena/reports/visomaster_proper_teams_bucket_census_2026-04-19.json`
- `DeepfakeBench/training/arena/reports/hdtf_visomaster_clean_vs_teams_join_2026-04-19.json`
- `DeepfakeBench/training/arena/reports/quickclips_visomaster_clean_vs_teams_join_2026-04-19.json`

Important: the Teams reports are based on a live but still incomplete snapshot.
Per user report, roughly `30%` more Teams-propagated data is still expected to
arrive. Anything count-sensitive generated from the current Teams snapshot must
be treated as provisional and regenerated after propagation completes.

Read these code/config surfaces next:

- `DeepfakeBench/training/data/sources/combined_paired.py`
- `DeepfakeBench/training/data/sources/visomaster.py`
- `DeepfakeBench/training/utils/grouping.py`
- `DeepfakeBench/training/tests/test_phase4_family_pipeline.py`
- `DeepfakeBench/training/arena/future_proper_data_inventory.template.yaml`
- `DeepfakeBench/training/arena/build_future_proper_target_domain_manifest.py`
- `DeepfakeBench/training/experiments/phase2_round13/R13_STARTUP_SMOKE_WTB3_weak_signal_hints_plus_teams_hints.yaml`
- `DeepfakeBench/training/experiments/phase2_round13/R13_SMOKE_WTB3_weak_signal_hints_plus_teams_hints.yaml`
- `DeepfakeBench/training/experiments/phase2_round13/R13_WTB1_weak_signal_no_hints.yaml`
- `DeepfakeBench/training/experiments/phase2_round13/R13_WTB2_weak_signal_hints_only.yaml`
- `DeepfakeBench/training/experiments/phase2_round13/R13_WTB3_weak_signal_hints_plus_teams_hints.yaml`

## Current WT-B State

WT-B runtime is already landed and committed.

What exists now:

- explicit `combined_paired.visomaster_hints`
- explicit `combined_paired.visomaster_hints_teams`
- policy-aware partitioning for direct Teams via
  `combined_paired.teams.apply_bad_data_policy: true`
- a launcher-only 2-step startup smoke
- a launcher-only 100-step integration smoke
- shared cache paths for DeepLive, VisoMaster hints, Teams passthrough, and
  external VCD reals

WT-B planning docs already live in:

- `DeepfakeBench/training/docs/R13_WTB_WEAK_SIGNAL_RUNBOOK_2026-04-18.md`
- `DeepfakeBench/training/docs/relaunch_handoffs/WT_B_AND_NEW_DATA_READINESS_2026-04-19.md`

## Live Smoke Snapshot

Snapshot time for the status below:

- `2026-04-19 14:00 CEST`

Active Vertex jobs:

- startup smoke:
  - display name: `exp-R13_STARTUP_SMOKE_WTB3_weak_signal_hints_plus_teams_hints-20260419-125302`
  - custom job id: `9012653657048481792`
  - W&B run id: `8bjcadyr`
- integration smoke:
  - display name: `exp-R13_SMOKE_WTB3_weak_signal_hints_plus_teams_hints-20260419-125308`
  - custom job id: `2921535161029885952`
  - W&B run id: `g6f8fovi`

What has already been proven by the running smokes:

- startup smoke reached `train_step: 2`
- startup smoke already emitted `val_in_dist/overall/auc: 0.4856003244997296`
- startup smoke summary shows:
  - `visomaster_hints_fake: 480`
  - `visomaster_hints_teams_fake: 202`
  - `deeplive_teams_real: 1111`
  - `deeplive_teams_fake: 1111`
- integration smoke reached `train_step: 49`
- integration smoke summary shows:
  - `visomaster_hints_fake: 480`
  - `visomaster_hints_teams_fake: 202`
  - `deeplive_teams_real: 1111`
  - `deeplive_teams_fake: 1111`
  - `external_real: 20`
- shared caches have been written:
  - DeepLive cache at `2026-04-19T11:06:42Z`
  - VisoMaster hints cache at `2026-04-19T11:40:42Z`
  - Teams passthrough cache at `2026-04-19T11:50:45Z`
  - external VCD real cache at `2026-04-19T11:51:00Z`

Known observation from the April 19 integration smoke:

- the smoke W&B summary included `unknown_fake: 20` alongside `external_real: 20`
- this has since been traced to unpaired-real family accounting and fixed in
  the current working tree
- historical pre-2026-04-20 W&B summaries should not be treated as evidence of
  real extra fake-family mass

## What The Next Agent Owns

The next agent should focus on the incoming new buckets.

Primary goals:

1. inspect the bucket structure the user provides
2. classify each bucket into the correct integration path
3. implement the loader or manifest path
4. add the minimum honest tests
5. propose the first experiment packet that includes the new data

The next agent does **not** need to redesign WT-B.

## Decision Rule For Each New Bucket

Use the smallest honest integration path.

### Path A: Extend `combined_paired`

Choose this only if the bucket already behaves like existing training buckets:

- paired real/fake data
- stable sample unit
- stable identity key
- frame layout compatible with existing discovery assumptions
- no need for future `proper_*` semantics

Typical sign:

- it looks like `samples/<sample_id>/frames/{real,fake}` with a usable
  manifest or folder contract

### Path B: WT-F Proper-Data Path

Choose this if the bucket is future proper data rather than legacy hints-like
data.

This is the correct path if:

- the bucket represents clean versus Teams parallel captures
- enhancement state matters semantically
- provenance needs to be explicit
- split hygiene must happen at `base_capture_id` or `split_group_id`
- the bucket should live in `proper_*` lanes rather than hints lanes

If this path is chosen:

- follow
  `DeepfakeBench/training/docs/PROPER_DATA_SCHEMA_AND_FUTURE_MANIFESTS_2026-04-17.md`
- use `future_proper_data_inventory.template.yaml`
- do **not** fold the data into:
  - `visomaster hints`
  - `visomaster hints (teams)`
  - `visomaster_teams_enhanced`

### Path C: Eval-Only Or Unpaired Path

Choose this if the bucket is:

- real-only
- fake-only
- not truly paired
- better suited for target-domain eval or OOD slices than training

If so:

- do not force it into WT-B
- document the correct eval-oriented path instead

## Required Input From The User

For each bucket, capture at least:

- bucket name
- prefix
- one real sample path
- one fake sample path
- one `manifest.json` example if present
- frame extension (`.png`, `.jpg`, or mixed)
- whether the bucket is paired real/fake or real-only
- the stable identity key
- any strategy, method, tier, transport, enhancement, or provenance fields

If multiple new buckets exist, classify them separately. Do not assume they all
share the same loader path.

## Minimum Deliverables

The next agent should leave behind:

1. a bucket-by-bucket classification summary
2. code changes for the chosen loader or manifest path
3. tests covering the new path
4. one recommended launcher smoke to prove the new data loads
5. a first-pass experiment plan for how to use the new data tonight

## Minimum Test Boundary

For each new loader or bucket extension, add:

- one discovery test proving sample selection and identity extraction
- one iteration test proving frame loading and source naming
- one grouping or family-routing test if the bucket creates a new family
- one YAML parse check for the new config surface

If a launcher smoke config is added, document:

- exact YAML path
- why that smoke is sufficient
- what nonzero counts or checkpoints must be seen

## Experiment Planning Constraint

Do not finalize the proper experiment packet until both conditions are true:

1. WT-B smoke gate is clean
2. the new bucket path is integrated and has its own proof boundary

Until then, only draft the matrix.

The likely planning shape is:

- keep the WT-B three-arm family as the weak-signal axis
- add the new data as a second explicit axis only after the loader is honest
- avoid hiding the new data inside a convenience lane that later makes results
  uninterpretable

## Files The Next Agent Will Most Likely Change

Depending on the bucket path, likely files include:

- `DeepfakeBench/training/data/sources/combined_paired.py`
- `DeepfakeBench/training/utils/grouping.py`
- `DeepfakeBench/training/tests/test_phase4_family_pipeline.py`
- `DeepfakeBench/training/arena/future_proper_data_inventory.template.yaml`
- `DeepfakeBench/training/arena/build_future_proper_target_domain_manifest.py`
- one or more new YAMLs under
  `DeepfakeBench/training/experiments/phase2_round13/`
- this handoff file

## Completion Update Template

When the agent finishes, replace the placeholders below.

### Buckets Received

- `hdtf_visomaster_cropped_frames`: reachable clean paired bucket with `1322` sample directories under `samples/<sample_id>/`. Spot check confirms `samples/HDTF20260416_00000/manifest.json` plus `frames/{real,fake}/frame_0000.png` through `frame_0015.png`. Manifest includes stable provenance fields such as `real_id`, `dataset_key=hdtf_20260416`, nested swap-model metadata, and target clip metadata.
- `quickclips_visomaster_cropped_frames`: reachable clean paired bucket with `767` sample directories under the same `samples/<sample_id>/` layout. Spot check confirms `samples/QCLIP20260417R2_00000/manifest.json` plus `frames/{real,fake}/frame_0000.png` through `frame_0015.png`. Manifest includes `real_id`, `dataset_key=quickclips_20260417_20260418_combined`, nested swap-model plus enhancer metadata, and target clip metadata. The sampled manifest describes this set as enhanced-only clean VisoMaster data.
- `hdtf_visomaster_cropped_frames_teams`: reachable Teams-parallel bucket with `835` sample directories under the same `samples/<sample_id>/` layout. Spot check confirms `samples/HDTF20260416_00000/manifest.json` plus `frames/{real,fake}/frame_0000.png` through `frame_0015.png`. The Teams manifest is a wrapper: top-level fields describe the Teams capture (`source=teams_capture`, `pipeline_version=teams_cropped_v1`, `frame_counts.selected_*`), while the full clean semantic payload is preserved under `original_manifest`.
- `quickclips_visomaster_cropped_frames_teams`: reachable Teams-parallel bucket with `520` sample directories under the same layout. Spot check confirms `samples/QCLIP20260417R2_00000/manifest.json` plus `frames/{real,fake}/frame_0000.png` through `frame_0015.png`. Its manifest follows the same wrapper pattern, with clean provenance preserved under `original_manifest`.

### Classification

- `hdtf_visomaster_cropped_frames`: `proper_data`
  Clean half of a future proper-data wave. Honest semantic mapping is `proper_real_clean` + `proper_visomaster_clean`, not legacy `visomaster_hints`.
- `quickclips_visomaster_cropped_frames`: `proper_data`
  Clean half of a future proper-data wave with enhancement semantics in the manifest. Honest semantic mapping is `proper_real_clean` + `proper_visomaster_enhanced_clean`, not `visomaster_enhanced` and not any hint lane.
- `hdtf_visomaster_cropped_frames_teams`: `proper_data`
  Teams half of the same HDTF wave. Honest semantic mapping is `proper_real_teams` + `proper_visomaster_teams`. The converter must keep the outer Teams source semantics and only read `original_manifest` for join and provenance fields.
- `quickclips_visomaster_cropped_frames_teams`: `proper_data`
  Teams half of the enhanced quickclips wave. Honest semantic mapping is `proper_real_teams` + `proper_visomaster_enhanced_teams`.

### Code Changes

- `DeepfakeBench/training/arena/inspect_visomaster_proper_buckets.py`: new pre-integration utility with two commands:
  - `census` for manifest-driven bucket inventory and field-quality checks
  - `validate-join` for clean-versus-Teams join-key validation once the parallel buckets arrive
  - parser now supports both clean manifests and Teams wrapper manifests that place the semantic clean payload under `original_manifest`
  - extracted records now preserve explicit `frame_files` lists so downstream inventory generation can emit exact frame paths without relisting GCS
- `DeepfakeBench/training/tests/test_inspect_visomaster_proper_buckets.py`: focused tests covering nested manifest parsing, duplicate base-key detection, and join-key recommendation behavior.
- `DeepfakeBench/training/arena/build_visomaster_proper_data_artifacts.py`: new provisional WT-F converter that:
  - loads the four incoming buckets directly from GCS manifests
  - intersects clean and Teams rows on exact `sample_id`
  - filters both clean and Teams rows to exact `16/16` for the current
    first-packet artifacts
  - normalizes each kept row pair into a future proper-data capture with explicit clean/Teams variants
  - writes explicit `frame_paths` into the inventory so manifest generation stays local after the initial bucket scan
  - applies explicit current-wave band defaults (`hdtf_20260416 -> high/big_face`, `quickclips_20260417_20260418_combined -> medium/standard`) because the incoming bucket manifests do not yet carry `quality_band` / `face_scale_band`
  - defaults the emitted manifest/suite paths to the canonical runtime artifact
    names already used by configs and Docker packaging
  - renders both the future proper-data manifest and a concrete suite YAML from the WT-F template
- `DeepfakeBench/training/tests/test_build_visomaster_proper_data_artifacts.py`: focused tests covering exact `sample_id` overlap selection, strict clean-and-Teams fixed-frame filtering, canonical output naming, enhanced-method preservation, and rendered suite-YAML validity.
- `DeepfakeBench/training/arena/reports/visomaster_proper_clean_bucket_census_2026-04-19.json`: generated live census report for the two currently available clean buckets.
- `DeepfakeBench/training/arena/reports/visomaster_proper_teams_bucket_census_2026-04-19.json`: generated live census report for the two Teams buckets.
- `DeepfakeBench/training/arena/reports/hdtf_visomaster_clean_vs_teams_join_2026-04-19.json`: generated live join-validation report for HDTF clean versus Teams.
- `DeepfakeBench/training/arena/reports/quickclips_visomaster_clean_vs_teams_join_2026-04-19.json`: generated live join-validation report for quickclips clean versus Teams.
- `DeepfakeBench/training/arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml`: provisional WT-F inventory snapshot built from exact clean-versus-Teams `sample_id` overlap and strict clean-and-Teams `16/16` filtering.
- `DeepfakeBench/training/arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json`: provisional future proper-data manifest rendered from the inventory above.
- `DeepfakeBench/training/arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml`: concrete proper-data suite YAML with the manifest path filled in.
- `DeepfakeBench/training/arena/reports/proper_visomaster_wave_2026_04_19_provisional_build_report.json`: build report recording kept/skipped sample counts, dataset-band defaults, manifest summary, and suite occupancy.
- `DeepfakeBench/training/docs/relaunch_handoffs/NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md`: updated with the live bucket inspection results, current classification, tooling, blockers, and launch guidance.
- `DeepfakeBench/training/data/sources/proper_data.py`: new runtime proper-data loader that:
  - reads the explicit WT-F inventory / manifest contract
  - emits one paired training sample per fake proper-data variant while keeping
    the explicit lane names (`proper_visomaster_clean`,
    `proper_visomaster_enhanced_clean`, `proper_visomaster_teams`,
    `proper_visomaster_enhanced_teams`)
  - loads explicit local or `gs://` frame paths only
  - validates manifest / inventory linkage while treating equivalent local
    inventory references across `/workspace/...`, repo-relative `arena/...`,
    and `DeepfakeBench/training/arena/...` as the same file
- `DeepfakeBench/training/data/sources/combined_paired.py`: training-side plumbing for:
  - `combined_paired.proper_data`
  - proper-data discovery summaries and per-lane counts
  - quality-domain routing for the four proper fake lanes
  - proper-data identity grouping on WT-F `split_group_id` so training-side
    dev/train partitioning matches the future manifest split contract
  - paired iteration from explicit frame paths, with anchor-index intersection
    retained only as a defensive fallback for future ragged manifests
- `DeepfakeBench/training/utils/grouping.py`: explicit grouping / family routing
  for:
  - `proper_visomaster_clean_fake`
  - `proper_visomaster_enhanced_clean_fake`
  - `proper_visomaster_teams_fake`
  - `proper_visomaster_enhanced_teams_fake`
  - `proper_real_clean`
  - `proper_real_teams`
- `DeepfakeBench/training/data/augmentations/pipelines.py`: family-router
  registration for the explicit proper-data fake and real families
- `DeepfakeBench/training/train_sweep.py`: loader-integrity logging for
  `ProperData`, `Proper-data lane counts (all)`, and
  `Proper-data lane counts (train)` plus enabled/zero-sample sanity checks
- `DeepfakeBench/training/tests/test_phase4_family_pipeline.py`: expanded
  training-side tests covering proper-data discovery, counts, grouping,
  iteration on explicit frame paths, ragged-pair behavior, and manifest-path
  validation
- `DeepfakeBench/training/experiments/phase2_round13/R13_STARTUP_SMOKE_PROPER_DATA_WTF_PROVISIONAL.yaml`: tiny 2-step launcher smoke for the new proper-data lanes
- `DeepfakeBench/training/experiments/phase2_round13/R13_STARTUP_SMOKE_WTB3_with_proper_data_unenhanced_provisional.yaml`: tiny combined 2-step launcher smoke for the first `WTB3 + proper_data` arm
- `DeepfakeBench/training/experiments/phase2_round13/R13_WTB3_with_proper_data_unenhanced_provisional.yaml`: first real-packet config for `WTB3` plus the retained unenhanced proper-data lanes
- `DeepfakeBench/training/experiments/phase2_round13/R13_WTB3_with_proper_data_full_snapshot_provisional.yaml`: first real-packet config for `WTB3` plus the full retained provisional proper-data snapshot
- `DeepfakeBench/training/.dockerignore`: adjusted so future proper-data
  target-domain manifests are included in the image build context for remote
  smokes and follow-up launches

### Verification

- `gcloud storage ls gs://hdtf_visomaster_cropped_frames/samples | wc -l`: `1322`
- `gcloud storage ls gs://quickclips_visomaster_cropped_frames/samples | wc -l`: `767`
- `gcloud storage ls gs://hdtf_visomaster_cropped_frames_teams/samples | wc -l`: `835`
- `gcloud storage ls gs://quickclips_visomaster_cropped_frames_teams/samples | wc -l`: `520`
- counts above are a live snapshot from the current shell session on `2026-04-19`, not a declared final Teams total
- `gcloud storage ls gs://hdtf_visomaster_cropped_frames/samples/HDTF20260416_00000/frames/{real,fake}`: both real and fake sides contain `frame_0000.png` through `frame_0015.png`
- `gcloud storage ls gs://quickclips_visomaster_cropped_frames/samples/QCLIP20260417R2_00000/frames/{real,fake}`: both real and fake sides contain `frame_0000.png` through `frame_0015.png`
- `gcloud storage ls gs://hdtf_visomaster_cropped_frames_teams/samples/HDTF20260416_00000/frames/{real,fake}`: both real and fake sides contain `frame_0000.png` through `frame_0015.png`
- `gcloud storage ls gs://quickclips_visomaster_cropped_frames_teams/samples/QCLIP20260417R2_00000/frames/{real,fake}`: both real and fake sides contain `frame_0000.png` through `frame_0015.png`
- `gcloud storage cat gs://hdtf_visomaster_cropped_frames/samples/HDTF20260416_00000/manifest.json`: manifest confirms `real_id`, `dataset_key`, nested swap-model metadata, and clean provenance
- `gcloud storage cat gs://quickclips_visomaster_cropped_frames/samples/QCLIP20260417R2_00000/manifest.json`: manifest confirms `real_id`, `dataset_key`, nested swap-model plus enhancer metadata, and clean provenance
- `gcloud storage cat gs://hdtf_visomaster_cropped_frames_teams/samples/HDTF20260416_00000/manifest.json`: manifest confirms Teams-wrapper fields at top level and the original clean manifest under `original_manifest`
- `gcloud storage cat gs://quickclips_visomaster_cropped_frames_teams/samples/QCLIP20260417R2_00000/manifest.json`: manifest confirms the same wrapper pattern for quickclips Teams data
- `python3 -m pytest DeepfakeBench/training/tests/test_inspect_visomaster_proper_buckets.py`: `3 passed`
- `python3 -m py_compile DeepfakeBench/training/arena/inspect_visomaster_proper_buckets.py DeepfakeBench/training/arena/build_visomaster_proper_data_artifacts.py DeepfakeBench/training/tests/test_inspect_visomaster_proper_buckets.py DeepfakeBench/training/tests/test_build_visomaster_proper_data_artifacts.py`: passed
- `python3 -m pytest DeepfakeBench/training/tests/test_inspect_visomaster_proper_buckets.py DeepfakeBench/training/tests/test_build_visomaster_proper_data_artifacts.py DeepfakeBench/training/tests/test_future_proper_target_domain_manifest_builder.py -q`: `9 passed`
- `python3 -m py_compile DeepfakeBench/training/data/sources/proper_data.py DeepfakeBench/training/data/sources/combined_paired.py DeepfakeBench/training/utils/grouping.py DeepfakeBench/training/data/augmentations/pipelines.py DeepfakeBench/training/train_sweep.py DeepfakeBench/training/arena/build_visomaster_proper_data_artifacts.py DeepfakeBench/training/tests/test_build_visomaster_proper_data_artifacts.py DeepfakeBench/training/tests/test_phase4_family_pipeline.py`: passed
- `python3 -m pytest DeepfakeBench/training/tests/test_build_visomaster_proper_data_artifacts.py DeepfakeBench/training/tests/test_inspect_visomaster_proper_buckets.py DeepfakeBench/training/tests/test_phase4_family_pipeline.py -q`: `40 passed, 5 skipped`
- `python3 DeepfakeBench/training/arena/inspect_visomaster_proper_buckets.py census --source hdtf_visomaster_cropped_frames --source quickclips_visomaster_cropped_frames --output DeepfakeBench/training/arena/reports/visomaster_proper_clean_bucket_census_2026-04-19.json`: report written with `0` load errors on both buckets
- `python3 DeepfakeBench/training/arena/inspect_visomaster_proper_buckets.py census --source hdtf_visomaster_cropped_frames_teams --source quickclips_visomaster_cropped_frames_teams --output DeepfakeBench/training/arena/reports/visomaster_proper_teams_bucket_census_2026-04-19.json`: report written with `0` load errors on both Teams buckets
- `python3 DeepfakeBench/training/arena/inspect_visomaster_proper_buckets.py validate-join --left-source hdtf_visomaster_cropped_frames --right-source hdtf_visomaster_cropped_frames_teams --output DeepfakeBench/training/arena/reports/hdtf_visomaster_clean_vs_teams_join_2026-04-19.json`: `sample_id` is a perfect one-to-one overlap for all `835` Teams rows; `real_id`, `target_video_name`, and `clip_stem` are ambiguous because multiple fake variants share the same base capture
- `python3 DeepfakeBench/training/arena/inspect_visomaster_proper_buckets.py validate-join --left-source quickclips_visomaster_cropped_frames --right-source quickclips_visomaster_cropped_frames_teams --output DeepfakeBench/training/arena/reports/quickclips_visomaster_clean_vs_teams_join_2026-04-19.json`: `sample_id` is a perfect one-to-one overlap for all `520` Teams rows; base-capture keys are ambiguous for the same reason
- `python3 DeepfakeBench/training/arena/build_visomaster_proper_data_artifacts.py`: completed successfully and wrote all four provisional WT-F artifacts using the canonical runtime output names
- live census highlights from `visomaster_proper_clean_bucket_census_2026-04-19.json`:
  - `hdtf_visomaster_cropped_frames`:
    - `sample_id` is fully one-to-one (`1322 / 1322` unique)
    - `real_id` and `clip_stem` collapse to `721` unique base clips with up to `4` fake variants per base clip
    - swap-model counts span `9` models; enhancer counts show `396` unenhanced and `926` enhanced samples
  - `quickclips_visomaster_cropped_frames`:
    - `sample_id` is fully one-to-one (`767 / 767` unique)
    - `real_id` and `clip_stem` collapse to `386` unique base clips with up to `2` fake variants per base clip
    - all observed samples are enhanced; no `enhancer=None` rows were found
- live census highlights from `visomaster_proper_teams_bucket_census_2026-04-19.json`:
  - `hdtf_visomaster_cropped_frames_teams`:
    - `sample_id` is fully one-to-one (`835 / 835` unique)
    - exact clean-versus-Teams overlap is `835` paired sample IDs, leaving `487` clean-only HDTF rows with no Teams counterpart
    - all `9` swap models and all `8` enhancer labels are still represented, but `12` low-frequency clean combo keys are absent from the Teams subset
    - the raw Teams subset still contains `100` underfilled rows; the strict
      clean-and-Teams artifact build retains `707` HDTF pairs after also
      excluding clean-side ragged rows from the overlap
  - `quickclips_visomaster_cropped_frames_teams`:
    - `sample_id` is fully one-to-one (`520 / 520` unique)
    - exact clean-versus-Teams overlap is `520` paired sample IDs, leaving `247` clean-only quickclips rows with no Teams counterpart
    - the full clean combo grid is preserved in the Teams subset
    - the raw Teams subset still contains `13` underfilled rows; the strict
      clean-and-Teams artifact build retains `499` quickclips pairs after also
      excluding clean-side ragged rows from the overlap
- provisional WT-F build highlights from the **current** checked-in
  `proper_visomaster_wave_2026_04_19_provisional_build_report.json` and
  `proper_visomaster_target_domain_manifest_2026-04-19_provisional.json`:
  - inventory currently keeps `1826` exact clean-versus-Teams captures after
    strict clean and Teams filtering:
    - HDTF: `1094` kept, `35` ragged clean rows skipped, `151` ragged Teams
      rows skipped, `42` clean-only rows still unmatched
    - quickclips: `732` kept, `14` ragged clean rows skipped, `16` ragged
      Teams rows skipped, `5` clean-only rows still unmatched
  - generated manifest currently contains `7304` videos across the six
    canonical exact lanes:
    - `proper_real_clean`: `1826`
    - `proper_real_teams`: `1826`
    - `proper_visomaster_clean`: `342`
    - `proper_visomaster_teams`: `342`
    - `proper_visomaster_enhanced_clean`: `1484`
    - `proper_visomaster_enhanced_teams`: `1484`
  - generated split counts are nonzero on both sides: `5776` `dev`, `1528`
    `lockbox`
  - rendered suite YAML has **no empty suites**
  - rendered suite occupancy is nonzero for every canonical proper-data suite,
    including:
    - `proper_real_teams_dev`: `1444`
    - `proper_real_teams_lockbox`: `382`
    - `proper_visomaster_teams_dev`: `262`
    - `proper_visomaster_teams_lockbox`: `80`
    - `proper_visomaster_enhanced_teams_dev`: `1182`
    - `proper_visomaster_enhanced_teams_lockbox`: `302`
  - the current first-packet artifact contract is exact `16/16` on both clean
    and Teams sides; runtime ragged-pair intersection remains only as a
    defensive fallback for future manifests
- `python3 -m pytest DeepfakeBench/training/tests/test_build_visomaster_proper_data_artifacts.py -q`: `6 passed`
- `python3 -m pytest DeepfakeBench/training/tests/test_phase4_family_pipeline.py -q`: `34 passed, 5 skipped`
- `python3 -m py_compile DeepfakeBench/training/data/sources/combined_paired.py DeepfakeBench/training/arena/build_visomaster_proper_data_artifacts.py`: passed
- `gcloud ai custom-jobs describe 9012653657048481792 --region=asia-southeast1 --project=train-cvit2`: `JOB_STATE_SUCCEEDED`, start `2026-04-19T11:01:25Z`, end `2026-04-19T12:03:14Z`
- `gcloud ai custom-jobs describe 2921535161029885952 --region=asia-southeast1 --project=train-cvit2`: `JOB_STATE_SUCCEEDED`, start `2026-04-19T11:01:01Z`, end `2026-04-19T12:23:56Z`
- `gcloud ai custom-jobs describe 2064725331922649088 --region=asia-southeast1 --project=train-cvit2`: `JOB_STATE_SUCCEEDED`, start `2026-04-19T17:54:05Z`, end `2026-04-19T17:58:07Z`
- `./launch_experiment.sh -y phase2r13-experiments asia-southeast1 experiments/phase2_round13/R13_STARTUP_SMOKE_PROPER_DATA_WTF_PROVISIONAL.yaml`: launched the proper-data startup smoke above
- proper-data smoke proof lines from job `2064725331922649088` / W&B `bxay0n66`:
  - `ProperData: enabled=True -> 128 samples`
  - `Proper-data lane counts (all): {'proper_visomaster_clean': 32, 'proper_visomaster_enhanced_clean': 32, 'proper_visomaster_enhanced_teams': 32, 'proper_visomaster_teams': 32}`
  - `Proper-data lane counts (train): {'proper_visomaster_clean': 29, 'proper_visomaster_enhanced_clean': 25, 'proper_visomaster_enhanced_teams': 25, 'proper_visomaster_teams': 29}`
  - `MAX STEPS REACHED: 2/2`
  - checkpoint upload succeeded to `gs://training-job-outputs/phase2r13_experiments/bxay0n66/...`

### Smoke / Launch Follow-Up

- The honest WT-F proof boundary now exists both locally and through the normal
  remote launcher path:
  - inventory: `DeepfakeBench/training/arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml`
  - manifest: `DeepfakeBench/training/arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json`
  - suites: `DeepfakeBench/training/arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml`
  - smoke config: `DeepfakeBench/training/experiments/phase2_round13/R13_STARTUP_SMOKE_PROPER_DATA_WTF_PROVISIONAL.yaml`
- The first combined `WTB3 + proper_data` launch surfaces now also exist in-repo:
  - startup smoke: `DeepfakeBench/training/experiments/phase2_round13/R13_STARTUP_SMOKE_WTB3_with_proper_data_unenhanced_provisional.yaml`
  - real packet arm 1: `DeepfakeBench/training/experiments/phase2_round13/R13_WTB3_with_proper_data_unenhanced_provisional.yaml`
  - real packet arm 2: `DeepfakeBench/training/experiments/phase2_round13/R13_WTB3_with_proper_data_full_snapshot_provisional.yaml`
- Proper-data startup smoke status:
  - custom job id: `2064725331922649088`
  - display name: `exp-R13_STARTUP_SMOKE_PROPER_DATA_WTF_PROVISIONAL-20260419-194708`
  - W&B run id: `bxay0n66`
  - Vertex state: `JOB_STATE_SUCCEEDED`
  - loader proof: nonzero proper-data sample counts plus nonzero counts in all
    four explicit fake proper-data lanes
- User-reported operational note: the current Teams propagation is still incomplete, with roughly `30%` more data from the original new-data buckets expected to appear in the Teams buckets over the next few hours. This is enough to continue converter, inventory, manifest-shape, and loader work now, but it is not a stable final snapshot for count-sensitive artifacts.
- Observed pass boundary from the generated snapshot:
  - nonzero rows now exist for `proper_visomaster_clean`, `proper_visomaster_enhanced_clean`, `proper_real_teams`, `proper_visomaster_teams`, and `proper_visomaster_enhanced_teams`
  - retained rows were constructed from exact `sample_id` overlaps and filtered
    to fixed-frame rows on both sides (`1094` HDTF + `732` quickclips)
  - the rendered proper-data suite file has no empty suites
- Observed pass boundary from the remote training smoke:
  - `ProperData: enabled=True -> 128 samples`
  - all four fake proper-data lanes loaded with nonzero counts in both all/train
  summaries
  - training reached `max_train_steps: 2`
  - checkpoint write succeeded
- Any inventory, manifest, or suite file generated before propagation completes should be treated as a provisional snapshot and regenerated once the Teams buckets finish updating.

### Post-Launch Corrections For Future Packet Work

- the April 19 written `221 / 985 / 2412` packet counts were stale; the live
  RLP1 startup summaries exposed `684` proper fake rows in `RLP1_04` and
  `3186` proper fake rows in `RLP1_05/06/07/08`
- the checked-in builder report / manifest are larger still
  (`342 / 1484 / 1484 / 342` fake-lane totals overall), so future packet docs
  must cite the builder report and then re-check the launched run's startup W&B
  summary
- strict arm-to-arm holdout comparability drifted because the legacy identity
  split used a global shuffled identity list; future comparison configs should
  set `combined_paired.identity_split_mode: "hash_stable"`
- the historical `unknown_fake` / `external_real` oddity was a reporting bug,
  not a real extra fake family; older pre-fix W&B summaries should be read with
  that in mind

### Recommended Experiment Packet

- The technical proof needed to start experiment planning is now satisfied:
  WT-B smoke gate is clean and the proper-data startup smoke is also clean.
- Do not finalize the long-running packet until the provisional proper-data
  artifacts have been regenerated from the stabilized Teams snapshot.
- Draft the first-night packet as a two-axis matrix:
  - axis 1: frozen WT-B weak-signal family (`WTB1`, `WTB2`, `WTB3`)
  - axis 2: explicit `proper_data` add-on as a single `off/on` axis
- Recommended first packet:
  - control: `WTB3` unchanged via `R13_WTB3_weak_signal_hints_plus_teams_hints.yaml`
  - add-on 1: `WTB3 + proper` via `R13_WTB3_with_proper_data_unenhanced_provisional.yaml`, using only the retained unenhanced proper VisoMaster lanes; the launched RLP1 startup summary exposed `684` proper fake rows in this arm (`342 clean + 342 teams`)
  - add-on 2: `WTB3 + full retained proper snapshot` via `R13_WTB3_with_proper_data_full_snapshot_provisional.yaml`, using both unenhanced and enhanced proper lanes; the launched RLP1 startup summary exposed `3186` proper fake rows in this arm (`342 + 1251 + 1251 + 342`)
  - startup smoke for the first combined arm: `R13_STARTUP_SMOKE_WTB3_with_proper_data_unenhanced_provisional.yaml`
  - if the training-side loader needs source-specific increments rather than lane-only increments, do that from the inventory layer by filtering on the `HDTF...` versus `QCLIP...` `base_capture_id` prefixes rather than by hiding the distinction inside a merged lane
- before any rerun or packet-2 launch, rebuild the WT-F artifacts and record
  the startup W&B summary again instead of reusing these copied counts
- The tiny proof run now exists; do not explode the first packet into many
  proper-data sub-arms yet.

### Open Issues

- The current Teams data is still incomplete. Per user report, about `30%` more of the original new-data buckets has not yet been propagated through Teams. The current snapshot is sufficient for continued implementation and pre-integration work, but not for freezing final manifests or final experiment counts.
- The Teams buckets are available, but they are partial subsets rather than full mirrors:
  - HDTF: `1280 / 1322` clean sample IDs have Teams counterparts
  - quickclips: `762 / 767` clean sample IDs have Teams counterparts
  Inventory generation must therefore intersect on `sample_id` rather than assuming every clean row has a Teams pair.
- The Teams manifests are wrapper documents. Semantic provenance fields such as `real_id`, `dataset_key`, target clip metadata, swap model, and enhancer live under `original_manifest`, while the Teams-specific capture state lives in the outer manifest. Any converter must read both layers correctly.
- The raw buckets are not yet perfectly fixed-frame drops:
  - HDTF overlap contains `35` ragged-clean rows and `151` ragged-Teams rows
  - quickclips overlap contains `14` ragged-clean rows and `16` ragged-Teams rows
  The current committed provisional artifacts already filter these out, so the
  first packet now relies on the retained strict subset (`1094 + 732` pairs)
  rather than on ragged-pair fallback behavior.
- The live census shows that `real_id` and `clip_stem` are not one-to-one inside a clean bucket:
  - HDTF: `1322` samples over `721` unique base clips, max multiplicity `4`
  - quickclips: `767` samples over `386` unique base clips, max multiplicity `2`
  This is expected if multiple fake variants share one base real clip, but it means `real_id`, `target_video_name`, and `clip_stem` are not valid clean-versus-Teams row keys. The correct row-level join key is `sample_id`.
- HDTF Teams preserves all swap-model and enhancer labels, but it does not preserve every clean combo key. `12` clean combo keys are absent from the Teams subset. This is acceptable for lane-level proper-data construction, but it means model-by-enhancer balance is not a perfect mirror of the clean bucket.
- Current `visomaster.py` discovery cannot load these buckets unchanged:
  - it only scans `samples/visomaster_`
  - it expects top-level `strategy == "visomaster"`
  - its identity and swap-model extraction assume the legacy `visomaster_<swap_model>_<...>` naming pattern
- The current provisional inventory had to assign `quality_band` / `face_scale_band` using explicit dataset-level defaults because those fields are not present in the incoming bucket manifests:
  - `hdtf_20260416 -> high / big_face`
  - `quickclips_20260417_20260418_combined -> medium / standard`
  This is acceptable for the current provisional manifest/suite proof boundary, but the long-term clean contract should move those band fields upstream into the authoritative capture/inventory metadata rather than hiding them in loader defaults.
- The current proper-data training path now uses WT-F `split_group_id` when it
  forms proper-data identities for train/dev partitioning. That closes the
  launch-blocking split-hygiene gap from the earlier review, but it should be
  preserved if we refactor the loader again later.
- Future comparison packets should also set
  `combined_paired.identity_split_mode: "hash_stable"`. The legacy shuffled
  identity split can move the effective holdout slice when later arms add new
  identities, which weakens strict like-for-like packet comparisons.
- Historical pre-2026-04-20 W&B summaries that show `unknown_fake` alongside
  `external_real` are slightly polluted by a reporting bug in family accounting.
- The current repo state still needs git closeout. The successful proper-data
  smoke used the built working-tree image, not a finalized committed state.
- The current proper-data smoke is only a 2-step launcher proof. It proves
  discovery, loading, grouping, iteration, and checkpoint write, but it is not
  yet a long-duration training result.

### Next Required Steps

- Run the new combined startup smoke
  (`R13_STARTUP_SMOKE_WTB3_with_proper_data_unenhanced_provisional.yaml`) before
  launching the longer first-packet arms.
- Make `combined_paired.identity_split_mode: "hash_stable"` the default stance
  for reruns and future comparison packets so arm membership changes do not move
  the holdout slice.
- If you want source-specific first-night experiments (`HDTF` first, then `quickclips`), plan to do that from the inventory layer using the `HDTF...` / `QCLIP...` `base_capture_id` prefixes until config-level selectors for those source slices are made explicit.
- Because the current Teams propagation is still incomplete, schedule one mandatory regeneration pass of the provisional inventory, manifest, and suite files after the Teams buckets stabilize.
