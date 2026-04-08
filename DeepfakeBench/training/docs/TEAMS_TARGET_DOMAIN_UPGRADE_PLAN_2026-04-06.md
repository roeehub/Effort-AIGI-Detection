# Teams Target-Domain Upgrade Plan

**Date:** April 6, 2026  
**Context:** R12 is strong overall, but deployment is live Microsoft Teams inference on local video streams, not generic holdout classification. The next round should optimize for that reality.

## 0. Progress Tracker

### 0.1 Status Snapshot

- `S0` baseline scorecard: **pending**
  - Shared FP32 + INT8 baseline scorecard is not frozen yet.
  - Current arena output is still FP32-only and not the final target-domain scorecard.

- `S1` enhanced bucket audit: **first full resolver pass completed**
  - Added `DeepfakeBench/training/tools/audit_enhanced_visomaster_resolver.py`.
  - First full audit artifacts were written locally to:
    - `/tmp/enhanced_visomaster_resolver_2026-04-06.json`
    - `/tmp/enhanced_visomaster_resolver_2026-04-06.csv`
  - Vertex-safe staged copies were uploaded to:
    - `gs://training-job-outputs/cache/visomaster/enhanced_visomaster_resolver_2026-04-06.json`
    - `gs://training-job-outputs/cache/visomaster/enhanced_visomaster_resolver_2026-04-06.csv`

- `S2` target-domain split freeze: **first manifest-based implementation completed**
  - Added `DeepfakeBench/training/arena/build_teams_target_domain_manifest.py`.
  - Added manifest-backed validation loading in:
    - `DeepfakeBench/training/data/validation_sources.py`
    - `DeepfakeBench/training/validate_custom_sources.py`
    - `DeepfakeBench/training/arena/run_target_domain_validation_sequential.py`
  - Added a manifest-based suite template:
    - `DeepfakeBench/training/arena/target_domain_suites.teams_manifest.template.yaml`
  - Added an optional explicit provenance template:
    - `DeepfakeBench/training/arena/prefix_rules.teams_manifest.template.yaml`
  - First live discovery probe completed:
    - grouped target-domain videos discovered: `7,276`
    - `4,614` real
    - `2,662` fake
  - `dev` / `lockbox` are now supported in tooling.
  - The bundled prefix-rules template now freezes every currently unresolved fake lane to safe prefix/session provenance labels.
  - Fast live verification now shows `0` fake rows still falling back to `teams_fake_unknown`.
  - A repo-local frozen manifest artifact now exists at:
    - `DeepfakeBench/training/arena/manifests/teams_target_domain_manifest_2026-04-06_frozen.json`
  - The remaining step is to turn this into the standard shared scorecard / comparison path.

- Track A merged Teams-enhanced loader: **full-length Vertex run launched; awaiting first checkpoint + arena**
  - New resolver-driven `combined_paired.visomaster_teams_enhanced` path is now in code.
  - One unified sample is built per base `sample_id`, with fake-branch selection deferred to iteration time.
  - Companion handoff docs created from the implementation thread:
    - `DeepfakeBench/training/docs/TRACK_A_HANDOFF_2026-04-06.md`
    - `DeepfakeBench/training/docs/TRACK_A_RUNTIME_RUNBOOK_2026-04-06.md`
    - `DeepfakeBench/training/docs/SMOKE_RUNNING_AGENT_HANDOFF_2026-04-06.md`
  - First smoke YAML is now wired:
    - `DeepfakeBench/training/experiments/phase2_round13/R13_SMOKE_trackA_teams_enhanced.yaml`
  - First full-length Track A YAML is now drafted:
    - `DeepfakeBench/training/experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml`
  - One-off arena template + runbook are now drafted:
    - `DeepfakeBench/training/arena/arena_config.track_a.yaml`
    - `DeepfakeBench/training/docs/TRACK_A_RUNTIME_RUNBOOK_2026-04-06.md`
  - That smoke config now points at the staged `gs://training-job-outputs/cache/visomaster/enhanced_visomaster_resolver_2026-04-06.json` manifest, so it is Vertex-safe.
  - The first Vertex smoke completed successfully on Vertex and cleared the runtime gate:
    - display name: `exp-R13_SMOKE_trackA_teams_enhanced-20260406-123308`
    - Vertex job id: `3841428920024956928`
    - final state: `JOB_STATE_SUCCEEDED`
    - create time (UTC): `2026-04-06T10:33:13.444938Z`
    - start time (UTC): `2026-04-06T10:41:02Z`
    - end time (UTC): `2026-04-06T13:08:48Z`
    - final Vertex update time (UTC): `2026-04-06T13:08:55.503508Z`
    - W&B run id: `2h0rhxun`
    - checkpoints written under:
      - `gs://training-job-outputs/phase2r13_experiments/2h0rhxun/`
  - Smoke gate facts confirmed from logs:
    - merged source loaded from the staged resolver manifest without resolver / GCS / runtime errors
    - `128` merged `visomaster_teams_enhanced` samples loaded for the smoke slice
    - companion-domain split in that smoke slice:
      - `12` `teams_v2`
      - `116` `clean_fallback`
    - nonzero merged samples appeared in train / val / test logs
    - checkpoints landed successfully in GCS
  - Important interpretation:
    - this smoke was an integration gate for the new data path, not a quality gate for the `100`-step checkpoint
    - the smoke metrics were weak / near-random, so the smoke checkpoint itself should not be treated as a model-quality result or sent to arena
  - The first full-length Track A baseline was launched on Vertex on April 7, 2026:
    - display name: `exp-R13_A_trackA_teams_enhanced-20260407-160710`
    - Vertex job id: `7406731712530481152`
    - full resource name: `projects/700371397073/locations/asia-southeast1/customJobs/7406731712530481152`
    - region: `asia-southeast1`
    - create time (UTC): `2026-04-07T14:07:17.673276Z`
    - start time (UTC): `2026-04-07T14:07:17.834510Z`
    - initial observed state: `JOB_STATE_PENDING`
    - initial observed Vertex update time (UTC): `2026-04-07T14:07:26.861950Z`

- Track B Teams re-characterization: **first current-bucket matched-pair rerun completed**
  - Added:
    - `DeepfakeBench/training/tools/analyze_teams_matched_pairs.py`
    - `DeepfakeBench/training/docs/TRACK_B_MATCHED_PAIR_REPORT_2026-04-07.md`
  - First local rerun artifacts written to:
    - `/tmp/teams_matched_pairs_2026-04-07.json`
    - `/tmp/teams_matched_pairs_2026-04-07.csv`
  - Current readout from that first deterministic sample:
    - ordinary current Teams v1/v2 slices are mixed-to-sharpening, not uniformly blur-dominant
    - the true enhanced-through-Teams slice remains blur / high-frequency-loss dominant
    - the existing single `TeamsCodecSimulation` direction-match only reached:
      - `37.5%` on `teams_v1_real`
      - `50.0%` on `teams_v1_fake`
      - `37.5%` on `teams_v2_real`
      - `50.0%` on `teams_v2_fake`
      - `62.5%` on `visomaster_enhanced_to_teams`
  - Current safe conclusion:
    - keep all Track B work sidecar
    - do not promote the old R9-era single-mode Teams simulator into active `phase2_round13` configs
  - Follow-on sidecar prototype comparison now favors:
    - `teams_codec_simulation.policy: "family_split"`
  - Current next safe task:
    - if Track B gets a training ablation, use the sidecar-only `family_split` policy first

- Track C target-domain + INT8 scorecard: **in progress**
  - Existing arena tooling now supports manifest-based split/slice evaluation.
  - Scorecard export is now wired in `DeepfakeBench/training/arena/run_target_domain_validation_sequential.py`.
  - Custom checkpoint aliases are now supported via `--checkpoint_map` and `--checkpoints ALL`.
  - FP32-vs-INT8 pair deltas are now exported through `scorecard.int8_delta.csv` / `pair_delta_rows`.
  - Concrete local scorecard assets now exist:
    - `DeepfakeBench/training/arena/target_domain_suites.teams_manifest.frozen_2026-04-06.yaml`
    - `DeepfakeBench/training/arena/build_teams_target_domain_suites.py`
    - `DeepfakeBench/training/arena/target_domain_suites.teams_manifest.breakdown_2026-04-07.yaml`
    - `DeepfakeBench/training/arena/checkpoint_maps/teams_target_domain.template.yaml`
    - `DeepfakeBench/training/arena/checkpoint_maps/teams_target_domain.seed_candidates_2026-04-07.yaml`
    - `DeepfakeBench/training/arena/run_teams_target_domain_scorecard.sh`
    - `DeepfakeBench/training/docs/TRACK_C_SCORECARD_RUNBOOK_2026-04-07.md`
  - INT8 checkpoint export / comparison still depends on having the actual INT8 artifacts to score.
  - Per-enhancer VisoMaster rows are still not available from the current frozen manifest; that artifact only preserves `visomaster_enhanced_macro`.

### 0.1A Regrouped Next Tasks (April 7, 2026)

This is the current execution order after the April 6 implementation thread and
the April 7 regroup.

Strict sequential dependencies:

1. Freeze the shared `S0` baseline scorecard in FP32 + INT8.
2. Treat `S1` as complete and do **not** re-audit the enhanced bucket.
3. Treat `S2` manifest tooling as implemented and use the frozen manifest artifact:
   - `DeepfakeBench/training/arena/manifests/teams_target_domain_manifest_2026-04-06_frozen.json`
4. First full-length Track A baseline: **submitted on Vertex April 7, 2026**
   - `DeepfakeBench/training/experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml`
5. Wait for the first real full-length checkpoint.
6. Update:
   - `DeepfakeBench/training/arena/arena_config.track_a.yaml`
7. Run arena on checkpoint alias `TRACK_A_CANDIDATE`.
8. Only after the first full-length + arena readout should we do deep weight tuning, `p_original` tuning, or promote the Track A source swap into more `R13` configs.

Parallel-safe work right now:

- Track A:
  - full-length launch + monitoring
  - then arena once a real checkpoint exists
- Track B:
  - matched-pair Teams re-characterization report
  - no edits to active `phase2_round13` configs unless the side-track earns promotion
- Track C:
  - finish the reusable target-domain scorecard / comparison path
  - keep the frozen manifest and suite flow stable
- Track D:
  - draft-only YAML work is fine, but do **not** finalize until A-C outputs are stable
- Track E:
  - optional ablation only after at least one flat merged-source baseline exists

Things that must wait:

- sending the smoke checkpoint to arena
- re-running the historical smoke
- changing Teams simulation on the active main line
- deep Track A tuning before the first full-length readout lands

### 0.1B Active Agent Ownership (April 7, 2026)

To avoid overlap across parallel worktrees, use this ownership state until the
next regroup.

- `Agent C1` Track C scorecard export: **in progress**
  - scope:
    - `DeepfakeBench/training/arena/run_target_domain_validation_sequential.py`
    - target-domain scorecard export helpers / tests
    - concrete frozen-suite / breakdown-suite / checkpoint-map / runbook assets
  - avoid:
    - Track A loader code
    - Track B augmentation code
    - active `phase2_round13` experiment YAML changes
  - immediate next task:
    - fill the checkpoint map with real baseline INT8 aliases once those artifacts exist
    - run the compact suite manifest for baseline freeze
    - run the breakdown suite manifest when family/method attribution is needed
- `Agent A1` Track A runtime path: **in progress**
  - live job:
    - display name: `exp-R13_A_trackA_teams_enhanced-20260407-160710`
    - Vertex job id: `7406731712530481152`
  - next safe task:
    - monitor until the first full-length checkpoint lands
    - then update arena config once the first checkpoint lands
- `Agent B1` Track B augmentation analysis: **in progress**
  - completed:
    - first current-bucket matched-pair rerun
    - initial Track B report:
      - `DeepfakeBench/training/docs/TRACK_B_MATCHED_PAIR_REPORT_2026-04-07.md`
  - next safe task:
    - first choice for a sidecar ablation is now:
      - `teams_codec_simulation.policy: "family_split"`
    - keep all work sidecar
- `Agent D1` Track D / E experiment drafts: **wait / draft only**
  - may prepare YAML drafts, but should not finalize until A-C outputs stabilize

### 0.2 S1 Audit Snapshot (April 6, 2026)

Re-run command:

```bash
python DeepfakeBench/training/tools/audit_enhanced_visomaster_resolver.py \
  --output-json /tmp/enhanced_visomaster_resolver_2026-04-06.json \
  --output-csv /tmp/enhanced_visomaster_resolver_2026-04-06.csv
```

First full resolver pass summary:

- `999` enhanced base `sample_id`s audited.
- Bucket coverage across those `999` samples:
  - `enhanced-visomaster-cropped`: `999 / 999` enhanced sample manifests, `143,785` enhanced fake frames across 8 enhancers
  - `live-deepfake-methods-real-and-fake-frames-cropped`: `998` real, `998` fake, `997` both
  - `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`: `352` real, `54` fake, `54` both
- Resolution status:
  - `teams_v2_companion`: `54` (`5.4%`)
  - `clean_companion_only`: `943` (`94.4%`)
  - `missing_companion`: `2` (`0.2%`)
- All `999` enhanced manifests currently claim `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`.
- Actual bucket resolution disagrees with that metadata for `943` rows.
- Of the `943` clean-only rows:
  - `297` still have partial Teams-v2 presence (`real` exists, `fake` missing)
  - `646` have no Teams-v2 companion files at all
- `989 / 999` samples have all expected 8 enhancers present.
- `10 / 999` samples are missing at least one expected enhancer.
- Missing-companion sample IDs in the first full pass:
  - `visomaster_CSCS_00399`
  - `visomaster_Inswapper128_08564`

Verified examples from the resolver output:

- `visomaster_CSCS_00007` resolves to a full Teams-v2 companion.
- `visomaster_CSCS_00029` resolves to both Teams-v2 and clean; Teams-v2 wins.
- `visomaster_CSCS_00037` and `visomaster_CSCS_00051` resolve to clean only because Teams-v2 lacks the fake branch.

Immediate implication:

- The raw `real_fake_bucket` metadata is not training-safe.
- Track A should consume the resolver manifest, not the raw enhanced manifests.
- The merged Teams-enhanced loader should prefer Teams-v2 only when both real and fake branches resolve there.
- The new source is now clearly **training-usable as a hybrid merged source**, because clean companions are present for almost all rows.
- But it is **not accurate to treat the whole bucket as a predominantly Teams-native paired source**, because only `54 / 999` rows have a full Teams-v2 companion.
- The loader should carry an explicit companion-domain label such as `teams_v2` vs `clean_fallback` so sampling, reporting, and later ablations can see how much true Teams pairing is actually present.

### 0.3 Track A Implementation Snapshot (April 6, 2026)

Implemented in code:

- `DeepfakeBench/training/data/sources/visomaster.py`
  - Added `VisoMasterTeamsEnhancedSample`.
  - Added `discover_visomaster_teams_enhanced_samples(...)`, which reads the resolver JSON and filters by resolved companion status/domain instead of trusting raw enhanced manifests.
  - Added `load_visomaster_teams_enhanced_frames(...)`, which:
    - always loads real frames from the resolved companion bucket;
    - loads fake frames from either:
      - the resolved original fake branch, or
      - one selected enhancer branch under `enhanced-visomaster-cropped`.

- `DeepfakeBench/training/data/sources/combined_paired.py`
  - Added `create_unified_samples_from_visomaster_teams_enhanced(...)`.
  - Added new source block `combined_paired.visomaster_teams_enhanced`.
  - Added branch-time selection with default `p_original = 0.5`.
  - Added explicit `sampling_family_key` handling so the merged source can be weighted as one family without exploding into 8 separate sample objects.
  - Added `method_variants` support so Group DRO / per-method metrics still know about both:
    - original VisoMaster fake methods, and
    - enhancer-specific fake methods.

Behavior now enforced:

- The resolver manifest is required for the new merged source.
- One base `sample_id` contributes one selected fake branch per sample iteration.
- Companion quality is explicit:
  - real / original-fake from `teams_v2` use the Teams quality domain;
  - clean-fallback branches use the clean VisoMaster quality domain;
  - enhanced fake branches keep the enhanced-family quality domain.

Targeted verification completed:

```bash
python -m py_compile \
  DeepfakeBench/training/data/sources/visomaster.py \
  DeepfakeBench/training/data/sources/combined_paired.py \
  DeepfakeBench/training/tests/test_phase4_family_pipeline.py

pytest DeepfakeBench/training/tests/test_phase4_family_pipeline.py \
  -k "group_key_mapping_representative_cases or visomaster_teams_enhanced or visomaster_enhanced"
```

Result:

- `9 passed`
- `0 failed`
- Real-manifest smoke against `/tmp/enhanced_visomaster_resolver_2026-04-06.json` loaded:
  - `997` merged training-eligible samples
  - `54` `teams_v2`
  - `943` `clean_fallback`

Remaining Track A work:

- decide whether the merged source should keep default sampling-family routing as `visomaster_enhanced_fake` or get its own reporting family later;
- launch the first full-length Track A baseline:
  - `DeepfakeBench/training/experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml`
- once the first full-length checkpoint lands, update:
  - `DeepfakeBench/training/arena/arena_config.track_a.yaml`
- run arena on that real Track A checkpoint via the alias-based path in:
  - `DeepfakeBench/training/docs/TRACK_A_RUNTIME_RUNBOOK_2026-04-06.md`
- decide whether to promote the same source swap into additional Round 13 configs after the first full-length + arena readout.

### 0.4 Agent Handoff

If a new agent picks this up, start here instead of re-auditing the buckets.

What is already done:

- `S1` resolver audit is complete.
- Track A core loader is implemented.
- The main plan doc is updated through this section.

Files with the actual Track A implementation:

- `DeepfakeBench/training/data/sources/visomaster.py`
- `DeepfakeBench/training/data/sources/combined_paired.py`
- `DeepfakeBench/training/tests/test_phase4_family_pipeline.py`

Important implementation facts:

- The new config block is `combined_paired.visomaster_teams_enhanced`.
- It requires `resolver_manifest_uri`.
- It builds one merged sample per base `sample_id`.
- It does **not** explode one base sample into 8 enhanced sample objects.
- At iteration time it chooses:
  - `original` fake from the resolved companion bucket, or
  - one enhancer branch from `enhanced-visomaster-cropped`.
- Default branch mix currently implemented:
  - `p_original = 0.5`
  - otherwise choose one available enhancer uniformly.

Current known-good behavior:

- Real-manifest smoke against `/tmp/enhanced_visomaster_resolver_2026-04-06.json` loads:
  - `997` merged training-eligible samples
  - `54` with `companion_domain = teams_v2`
  - `943` with `companion_domain = clean_fallback`
- The two unresolved rows stay excluded.

Tests already run:

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

Known blocker / repo caveat:

- `.gitignore` currently ignores `data/`, so:
  - `DeepfakeBench/training/data/sources/visomaster.py`
  - `DeepfakeBench/training/data/sources/combined_paired.py`
  do not show up in normal `git status` / `git diff`.
- The code is on disk and tested, but a new agent should be aware that git visibility is incomplete unless ignore rules are adjusted.

Immediate next task for the next agent:

1. Treat the Track A smoke as passed and do **not** relaunch it.
2. Launch the first full-length Track A baseline from:
   - `DeepfakeBench/training/experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml`
3. Wait for the first real checkpoint from that run, then update:
   - `DeepfakeBench/training/arena/arena_config.track_a.yaml`
4. Run arena using checkpoint alias `TRACK_A_CANDIDATE`, not a raw `gs://...pth` URI.

### 0.5 Track C / S2 Implementation Snapshot (April 6, 2026)

Implemented in code:

- `DeepfakeBench/training/data/validation_sources.py`
  - Added `load_external_manifest_videos(...)`.
  - Supports JSON, YAML, or JSONL manifest inputs.
  - Supports filtering by:
    - `label`
    - `split`
    - `slices`
    - `methods`
  - Uses deterministic identity hashing for manifest rows that do not provide numeric IDs.

- `DeepfakeBench/training/validate_custom_sources.py`
  - Added CLI flags:
    - `--external_real_manifest`
    - `--external_real_manifest_split`
    - `--external_real_manifest_slices`
    - `--external_fake_manifest`
    - `--external_fake_manifest_split`
    - `--external_fake_manifest_slices`
  - Manifest mode now takes precedence over raw bucket scanning when both are provided.
  - Real-source registration now derives from the actual loaded validation rows instead of assuming one fixed method per source.

- `DeepfakeBench/training/arena/run_target_domain_validation_sequential.py`
  - Added suite-manifest support for the new external real/fake manifest flags.

- `DeepfakeBench/training/arena/build_teams_target_domain_manifest.py`
  - New builder for the flat Teams benchmark bucket.
  - Groups frames into stable validation videos from:
    - session-capture filenames such as `Cam_Test__s32_103.0_frame_...`
    - flat upload filenames such as `visomaster_enhanced_raw__frame_000675_seq5402.png`
  - Assigns deterministic `dev` / `lockbox` splits from `identity_key`.
  - Emits core real slices:
    - `teams_real_all`
    - `teams_real_poor_quality`
    - `teams_real_lighting_extreme`
  - Emits currently supported fake slices:
    - `teams_fake_all`
    - `visomaster_enhanced_macro`
    - `deeplive_enhanced`
  - Supports optional `--prefix-rules` to freeze additional fake-family provenance without rewriting the builder.

- `DeepfakeBench/training/arena/target_domain_suites.teams_manifest.template.yaml`
  - One-command suite template for the new manifest-based validation path.

- `DeepfakeBench/training/arena/prefix_rules.teams_manifest.template.yaml`
  - Template for freezing ambiguous fake-family provenance with explicit prefix/session overrides.
  - The bundled template now covers all currently discovered unresolved fake lanes with safe session-level provenance slices.

Targeted verification completed:

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

First live discovery probe against `gs://teams-faces-data-test-2914-fake-4420-real-feb-28`:

- grouped validation videos discovered: `7,276`
  - `4,614` real
  - `2,662` fake
- grouping modes seen:
  - `3,643` `flat_upload`
  - `3,569` `session_capture`
  - `64` `fallback_single`
- top discovered fake prefixes by grouped video count:
  - `Cam_Test`: `714`
  - `deeplive_dor`: `545`
  - `visomaster_enhanced_raw`: `275`
  - `visomaster_enhanced_teams`: `275`
  - `PC_Generator`: `237`
  - `Test_Cam`: `220`
- top discovered real prefixes by grouped video count:
  - `dor_shkedi`: `1,158`
  - `Test_Cam`: `712`
  - `PC_Generator`: `553`
  - `bla_bla_chow`: `528`
  - `Md_noyn_Sharker`: `409`

Known limitation in this first pass:

- The flat Teams fake bucket does not fully encode source-family provenance in the filenames.
- The builder therefore only auto-tags:
  - `visomaster_enhanced_*`
  - `deeplive_dor`
- The bundled `prefix_rules.teams_manifest.template.yaml` now supplies a safe provenance freeze for the remaining discovered fake lanes, so live verification no longer leaves rows in `teams_fake_unknown`.

Track A operational sequence after the smoke:

1. Treat the smoke result as known:
   - Vertex job `3841428920024956928` finished `JOB_STATE_SUCCEEDED`
   - the merged resolver-driven source loaded and wrote checkpoints under
     - `gs://training-job-outputs/phase2r13_experiments/2h0rhxun/`
2. Launch the first full-length Track A run using:
   - `DeepfakeBench/training/experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml`
3. After the first full-length checkpoint lands, evaluate it through the normal arena path:
   - update `DeepfakeBench/training/arena/arena_config.track_a.yaml` with the real checkpoint URI;
   - `./launch_arena.sh --config arena_config.track_a.yaml --checkpoints TRACK_A_CANDIDATE`
4. After the first full-length + arena readout, decide whether to fold the same source swap into additional Round 13 configs or keep a dedicated Track A branch.

Recommended first config move:

- Start from:
  - `DeepfakeBench/training/experiments/phase2_round13/R13_A_champion_plus_enhanced.yaml`
- Done in:
  - `DeepfakeBench/training/experiments/phase2_round13/R13_SMOKE_trackA_teams_enhanced.yaml`
- First full-length follow-on draft:
  - `DeepfakeBench/training/experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml`
- That smoke config:
  - adds `combined_paired.visomaster_teams_enhanced`;
  - sets `resolver_manifest_uri` explicitly;
  - disables `combined_paired.visomaster_enhanced` for isolation.
- Arena caveat for the next step:
  - `./launch_arena.sh --checkpoints ...` expects checkpoint names from the selected arena config, not raw `gs://...pth` URIs.
  - Use:
    - `DeepfakeBench/training/arena/arena_config.track_a.yaml`
    - `DeepfakeBench/training/docs/TRACK_A_RUNTIME_RUNBOOK_2026-04-06.md`
- Standard launch entrypoints now available from the training root:
  - `./launch_experiment.sh`
  - `./launch_arena.sh`

Do not spend time next on:

- re-checking whether clean companions exist;
- re-deriving the `54 / 943 / 2` split;
- changing Teams simulation;
- tuning family weights deeply before the first full-length Track A readout lands.

### 0.5 Track A Smoke Result

Track A smoke on Vertex:

- display name: `exp-R13_SMOKE_trackA_teams_enhanced-20260406-123308`
- job id: `3841428920024956928`
- region: `asia-southeast1`
- final state: `JOB_STATE_SUCCEEDED`
- create time (UTC): `2026-04-06T10:33:13.444938Z`
- start time (UTC): `2026-04-06T10:41:02Z`
- end time (UTC): `2026-04-06T13:08:48Z`
- final Vertex update time (UTC): `2026-04-06T13:08:55.503508Z`
- W&B run id: `2h0rhxun`
- checkpoints written:
  - `gs://training-job-outputs/phase2r13_experiments/2h0rhxun/first_best_effort_20260406_ep1_auc0.4680_eer0.5160.pth`
  - `gs://training-job-outputs/phase2r13_experiments/2h0rhxun/top_n_effort_20260406_step50_auc0.4680_eer0.5160.pth`
  - `gs://training-job-outputs/phase2r13_experiments/2h0rhxun/top_n_effort_20260406_step100_auc0.4786_eer0.5184.pth`

Smoke gate outcome:

- runtime / integration gate: **passed**
- the resolver-driven merged source loaded cleanly and produced checkpoints
- the `100`-step checkpoint metrics were weak / near-random and should not be treated as a quality result
- operational meaning:
  - the first full-length Track A launch is now unblocked
  - arena should wait for a real full-length Track A checkpoint, not the smoke checkpoint

## 1. Objective

Build a better post-R12 model for live Microsoft Teams calls.

Primary goal:
- Reduce false positives on real low-quality Teams video from varied webcams, lighting, and network conditions.

Secondary goals:
- Improve detection of `Deeplivecam` regular + enhanced.
- Improve detection of `VisoMaster` regular, with extra focus on `SimSwap512` and enhanced variants.
- Preserve acceptable DF40/general performance.
- Preserve accuracy after INT8 quantization.

## 2. What We Know

### 2.1 Current Best Recipe

`R12_G` is still the practical reference point:
- Scratch ViT-B/16 DataComp-XL.
- Identity-balanced weighted sampling.
- DF40 + DeepLive + VisoMaster + Teams + external VCD reals.
- Family-aware augmentation with `quality_targeted_family` and `vcd_targeted`.
- Strong generic holdout and strong but imperfect OOD behavior.

Important caveat:
- The current training/evaluation loop is still not fully aligned to the production target. `val_holdout` is useful, but it is not the deployment metric.

### 2.2 The Real Deployment Problem

Observed in production-like use:
- False positives on poor real video.
- Sensitivity to lighting / location / camera characteristics.
- Weakness on VisoMaster enhanced faces.
- Weakness on VisoMaster `SimSwap512`.

The repo history supports this diagnosis:
- Data and domain coverage consistently mattered more than clever losses or generic curriculum.
- Lighting and webcam-domain mismatch are already documented as real failure modes.
- The Teams channel itself changes images in structured ways that the current pipeline only partially models.

### 2.3 Data Sources That Matter

Main training/eval universe:
- `DF40` fake and real: useful as a broad negative anchor, but not the deployment domain.
- `DeepLive` clean bucket: important fake family coverage.
- `VisoMaster` clean bucket: important fake family coverage.
- `Teams-v2` bucket: the closest existing production-like paired source.
- `VCD` reals: useful webcam-domain real anchor.
- `teams-faces-data-test-2914-fake-4420-real-feb-28`: strong target-domain benchmark bucket.

### 2.4 Two Different Enhanced VisoMaster Buckets

There are two distinct enhanced buckets and they should not be treated as interchangeable:

1. `gs://visomaster-enhanced-face-cropped/`
- Older loader-compatible bucket.
- Stored per enhancer variant.
- Current code already supports it.
- It behaves like a clean enhanced source, not a Teams-native merged source.
- Current top-level sample count observed: `8788`.

2. `gs://enhanced-visomaster-cropped/`
- Newer bucket.
- Stored by base `sample_id`, with enhancer subfolders under `frames/{enhancer}/`.
- Explicitly claims linkage to the Teams-v2 companion bucket using `sample_id`.
- Current top-level sample count observed: `999`.
- This is the more relevant bucket for the deployment goal because the data was passed through Teams.

### 2.5 Linkage Is Real, But Must Be Audited

The linkage mechanism is valid in principle:
- Example verified by user and by repo-side inspection: `visomaster_CSCS_00007`.
- The enhanced manifest points to:
  - `real` and `fake` from the companion bucket via `sample_id`
  - `enhanced` variants from `enhanced-visomaster-cropped`

However, spot checks also showed a second reality:
- Many enhanced manifests point to Teams-v2 in metadata but do **not** currently resolve to a full Teams-v2 real+fake companion.
- Verified clean-only examples from the first full resolver pass: `visomaster_CSCS_00037`, `visomaster_CSCS_00051`.
- The earlier coarse spot-check that flagged `visomaster_CSCS_00029` as missing was wrong; a side-level audit shows that `visomaster_CSCS_00029` is a valid Teams-v2 companion.
- First full resolver pass summary:
  - clean bucket coverage: `998` real, `998` fake, `997` both
  - Teams-v2 coverage: `352` real, `54` fake, `54` both
  - `teams_v2_companion`: `54`
  - `clean_companion_only`: `943`
  - `missing_companion`: `2`
- Many clean-only rows are only **partially** present in Teams-v2 (`real` exists, `fake` missing), which means side-level existence checks matter.

Working assumption:
- The join key is correct.
- The manifest metadata is not sufficient by itself.
- Companion resolution should be implemented by existence checks, not by trusting the manifest blindly.

This is important because it changes the right conclusion:
- the new bucket is usable for training **if** the loader falls back to the clean companion bucket via the resolver manifest;
- but it is not yet a strongly Teams-native paired source, because full Teams-v2 pairing is sparse.

### 2.6 The Current Sampler Has a Hidden Multiplicity Problem

The current `combined_paired` identity-balanced sampler selects **one sample object per identity per epoch**, then applies family weighting across those sample objects.

That matters for enhanced data:
- If one base sample becomes 8 separate enhanced sample objects, that identity gets 8 extra weighted chances to be served as enhanced.
- That is not the same as “add one more family with a weight.”
- It can quietly distort the effective training distribution.

Conclusion:
- The new Teams-enhanced VisoMaster source should not be integrated as “999 base samples -> 8,000+ independent paired samples” in the same style as the older clean enhanced bucket.
- It should be integrated as a **merged base sample** keyed by `sample_id`, with controlled fake-branch selection at iteration time.

### 2.7 TeamsCodecSimulation Is Useful, But Not Settled

Current state:
- `TeamsCodecSimulation` exists and is wired.
- The old dead-code wiring issue was already fixed after R9.
- The transform was validated against matched before/after Teams frame pairs.

Current implementation models:
- brightness increase,
- contrast increase,
- blur / smoothing,
- JPEG-like compression,
- bilateral deblocking,
- optional chroma blur.

What the repo already says:
- The first bucket-level assumptions were wrong.
- Matched-pair validation was necessary.
- Teams behavior is at least partly bimodal.
- A single fixed transform is only an approximation.

Important implication:
- “Re-enable TeamsCodecSimulation at `p=0.15`” is not a final decision.
- It is a placeholder hypothesis.
- We should investigate the Teams channel again using the newer before/after data and decide whether the right answer is:
  - one better global transform,
  - a two-mode mixture,
  - per-family application,
  - or even an offline synthetic “Teamsified” dataset for selected families.

## 3. Guiding Decisions

1. Optimize for the target domain first.
2. Keep DF40 as a guardrail, not the main target.
3. Treat Teams-native data as more valuable than clean enhanced data.
4. Do not let multi-enhancer multiplicity silently dominate the sampler.
5. Do not assume the current Teams simulator is optimal just because it already exists.
6. Test curriculum once, in a narrow and falsifiable form.

## 4. Success Criteria

Primary success:
- Lower false-positive rate on poor-quality real Teams data.

Secondary success:
- Better recall on:
  - `Deeplivecam` regular,
  - `Deeplivecam` enhanced,
  - `VisoMaster SimSwap512`,
  - `VisoMaster enhanced` macro and per-enhancer.

Guardrails:
- DF40 headline metric may drop a little, but not materially.
- Generic holdout AUC may dip slightly, but should remain close to R12.
- INT8 should retain the target-domain gain.

Recommended ranking order for candidates:
1. `teams_real_poor_quality` FPR
2. target fake macro recall
3. INT8 retention
4. overall Teams-real FPR
5. holdout AUC as tie-breaker only

## 5. Sequential Gates

These tasks should happen in order because downstream work depends on them.

### S0. Freeze Baseline

Goal:
- Freeze the exact `R12_G` baseline and one or two best post-R12 enhanced checkpoints on a shared scorecard.

Deliverables:
- Baseline evaluation report in FP32.
- Baseline evaluation report in INT8.
- Single agreed metric table for later comparisons.

### S1. Audit `enhanced-visomaster-cropped`

Goal:
- Turn the new enhanced bucket from “promising but ambiguous” into a training-safe source.

Required output:
- Resolver manifest with one row per `sample_id`.
- Resolution status:
  - `teams_v2_companion`
  - `clean_companion_only`
  - `missing_companion`
- Available enhancers per sample.
- Whether all 8 enhancers are present.
- Companion frame counts.

Hard rule:
- Training code must use the resolver manifest, not trust the raw manifest metadata.

### S2. Freeze Target-Domain Evaluation Splits

Goal:
- Create a stable `dev` / `lockbox` split for the Teams-like benchmark bucket.

Deliverables:
- Deterministic split by identity/video.
- Deterministic slices:
  - `teams_real_all`
  - `teams_real_poor_quality`
  - `teams_real_lighting_extreme`
  - `deeplive_regular`
  - `deeplive_enhanced`
  - `visomaster_original_macro`
  - `visomaster_simswap512`
  - `visomaster_enhanced_macro`
  - per-enhancer slices if possible

Only after `S1` and `S2` are done should model changes be merged and experiment configs be finalized.

## 6. Parallel Workstreams

These are designed for separate worktrees with minimal write overlap.

### Track A: New Teams-Enhanced Data Source

**Purpose**
- Integrate `enhanced-visomaster-cropped` in a way that matches the production use case and avoids sample explosion.

**Write scope**
- `DeepfakeBench/training/data/sources/visomaster.py`
- `DeepfakeBench/training/data/sources/combined_paired.py`
- tests for the new source

**Core design**
- Add a new config block, e.g. `combined_paired.visomaster_teams_enhanced`.
- Use `sample_id` as the base join key.
- Build merged base samples, not 8 independent enhanced sample objects.
- Each base sample contains:
  - real Teams frames
  - original fake Teams frames
  - available enhanced fake branches
- At iteration time choose exactly one fake branch:
  - original fake branch, or
  - one enhanced branch

**Recommended default branch policy**
- `p_original = 0.5`
- `p_enhanced = 0.5`
- `enhanced_choice = uniform`

**Why**
- This preserves per-enhancer diversity without multiplying the identity weight by 8.

**Acceptance**
- One base sample contributes one fake branch per selection.
- Identity leakage remains impossible.
- Resolver-manifest audit is enforced.

### Track B: Teams Channel Re-Characterization And Augmentation Redesign

**Purpose**
- Investigate the Teams transform more deeply using current before/after data, not just reuse the existing R9-era approximation.

**Write scope**
- `DeepfakeBench/training/data/augmentations/teams_simulation.py`
- `DeepfakeBench/training/tests/test_teams_simulation.py`
- a new analysis/report script or doc under `DeepfakeBench/training/docs/` or `tools/`

**What to investigate**
- Re-run matched-pair analysis on current data:
  - clean -> Teams real/fake
  - VisoMaster original fake -> Teams fake
  - optionally enhanced -> Teams-enhanced if aligned pairs exist
- Break out metrics by:
  - strategy
  - swap model
  - enhancer
  - content family
  - brightness / sharpness regime

**Specific questions**
- Is Teams truly one transform family, or two or more modes?
- Is the existing brightness/contrast overshoot still present?
- Does the sharpness bimodality cluster by method or by input quality?
- Is chroma handling worth modeling?
- Should the transform be:
  - one better preset,
  - a mixture of two presets,
  - or applied selectively by family?

**Recommended deeper target**
- Replace the single hard-coded policy:
  - “non-Teams families only, `p=0.15`”
- with a data-backed decision among:
  - `p=0.10`, `0.15`, `0.25`
  - single-mode simulation
  - mixture simulation
  - or offline cached Teamsified data for selected families

**Acceptance**
- New report using matched pairs from current buckets.
- Chosen augmentation policy justified by measured deltas, not carried over by habit.

### Track C: Target-Domain Evaluation And INT8 Scorecard

**Purpose**
- Make candidate selection reflect the real deployment goal.

**Write scope**
- `DeepfakeBench/training/arena/`
- `DeepfakeBench/training/validate_custom_sources.py`
- eval manifests or runner scripts

**Tasks**
- Build dev/lockbox split for `teams-faces-data-test-*`.
- Compute poor-quality and lighting-extreme real subsets.
- Add scorecard export for:
  - Teams-real FPR,
  - target fake recall,
  - per-method/per-enhancer breakdown,
  - INT8 delta.
- Make the scorecard easy to run across multiple checkpoints.

**Acceptance**
- One command or one short runner produces comparable model tables.

### Track D: Experiment Configs And Launch Matrix

**Purpose**
- Turn the plan into runnable YAMLs once Tracks A-C produce their outputs.

**Write scope**
- `DeepfakeBench/training/experiments/`
- optionally one coordinating doc under `DeepfakeBench/training/docs/`

**Tasks**
- Create a clean experiment family derived from `R12_G`.
- Use corrected seed plumbing explicitly.
- Add configs for:
  - merged Teams-enhanced source baseline
  - heavier enhanced emphasis
  - stronger Teams-real emphasis
  - Teams-simulation ablations driven by Track B
  - one curriculum pair
  - one seed repeat

**Acceptance**
- Config set is consistent, runnable, and directly traceable to the plan.

### Track E: Narrow Curriculum Ablation

**Purpose**
- Test the DF40-first idea once, in a bounded way.

**Write scope**
- experiment YAMLs only
- optional small doc note

**Important background**
- Repo history already suggests flat mixed training generally beat earlier curriculum ideas.
- So this should be a single falsifiable ablation, not the new default.

**Recommended design**
- Stage 1:
  - `DF40 + VCD real`
  - short run
  - moderate augmentation
- Stage 2:
  - initialize from Stage 1
  - full target-domain mixture, including merged Teams-enhanced source

**Matched control**
- One flat run from the same starting checkpoint and same total budget.

**Acceptance**
- Curriculum only survives if it improves target-domain metrics without losing the production fit.

## 7. Recommended Execution Order

### Phase 0: Truth And Baseline
1. `S0` Freeze baseline scorecard.
2. `S1` Audit `enhanced-visomaster-cropped`.
3. `S2` Freeze dev/lockbox evaluation manifests.

### Phase 1: Parallel Implementation
4. Track A implements merged Teams-enhanced loader.
5. Track B re-characterizes Teams channel and proposes augmentation policy.
6. Track C finalizes target-domain + INT8 scorecard tooling.
7. Track D drafts experiment YAMLs in parallel, but only finalizes after A-C outputs land.
8. Track E prepares the narrow curriculum pair.

### Phase 2: Integration
9. Merge Track A first.
10. Merge Track C next so evaluation is stable.
11. Merge Track B once the augmentation policy is selected.
12. Finalize Track D YAMLs.
13. Keep Track E as an optional ablation branch.

### Phase 3: Experiment Batches
14. Launch the flat merged-source baseline first.
15. Launch weight and augmentation ablations next.
16. Launch the curriculum pair only after at least one flat merged-source baseline exists.
17. Promote finalists only using the target-domain scorecard, with INT8 included.

## 8. Initial Experiment Set

These are the first experiments worth running once the implementation tracks land.

### Flat Mainline
- `FT_MERGED_BASE`
  - merged Teams-enhanced source
  - moderate enhanced weight
- `FT_MERGED_ENH_HEAVY`
  - same, heavier enhanced emphasis
- `FT_MERGED_REAL_BOOST`
  - same, more Teams-real / external-real emphasis

### Teams Transform Ablations
- `FT_MERGED_TEAMSIM_SINGLE`
  - best flat baseline + chosen single-mode Teams sim
- `FT_MERGED_TEAMSIM_MIX`
  - if Track B finds a real two-mode benefit
- `FT_MERGED_LIGHTING_PLUS_TEAMSIM`
  - best Teams-sim config + updated lighting block

### Stability / Reproducibility
- `FT_MERGED_BEST_SEED2`
  - exact repeat of best flat config with true seed change

### Curriculum
- `FT_CURRIC_STAGE1`
- `FT_CURRIC_STAGE2`
- `FT_FLAT_MATCHED_CTRL`

## 9. Non-Negotiable Checks

- Do not train the new Teams-enhanced bucket by exploding it into one sample per enhancer variant inside the existing sampler.
- Do not trust `real_fake_bucket` metadata without resolver verification.
- Do not choose winners from generic holdout alone.
- Do not accept a target-domain win that vanishes in INT8.
- Do not assume the existing Teams simulator is final.

## 10. Suggested Agent Split

If using separate worktrees, use these boundaries:

### Agent 1: Data Integration
- Owns Track A only.
- Avoids evaluation and experiment YAML files.

### Agent 2: Teams Simulation Investigation
- Owns Track B only.
- Avoids data loader code and YAMLs.

### Agent 3: Evaluation / Scorecard
- Owns Track C only.
- Avoids augmentation and data loader code.

### Agent 4: Experiment Configs
- Owns Track D and optionally Track E.
- Should wait until the outputs from A-C are stable.

## 11. Bottom Line

The right next round is not just “R12 + more enhanced VisoMaster.”

It is:
- make the production metric primary,
- integrate the new Teams-passed-through enhanced source correctly,
- audit and probably redesign the Teams simulation using current matched-pair data,
- keep DF40 as an anchor instead of a target,
- and test curriculum once, narrowly, instead of betting the round on it.

That is the shortest path to a model that is better for the actual Teams deployment rather than just better on internal validation.
