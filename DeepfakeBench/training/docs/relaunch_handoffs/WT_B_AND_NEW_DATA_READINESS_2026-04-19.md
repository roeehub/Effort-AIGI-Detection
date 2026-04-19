# WT-B And New Data Readiness

## Scope

This note captures the current upgrade state after WT-B code landed and before
the first proper experiment packet that will include incoming new data buckets.

The goal is to make the next steps explicit:

- finish the WT-B smoke through the normal launcher path
- define and integrate the new buckets
- run the first real experiment family only after those gates are clean

## What Is Already Landed

- explicit weak-signal lanes now exist in the training runtime:
  - `combined_paired.visomaster_hints`
  - `combined_paired.visomaster_hints_teams`
- the direct `teams` lane can be policy-filtered so WT-B does not silently
  duplicate retained Teams-played hint rows
- tracked policy artifacts live under
  `DeepfakeBench/training/policy/visomaster_bad_data/`
- runnable WT-B family configs exist:
  - `R13_WTB1_weak_signal_no_hints.yaml`
  - `R13_WTB2_weak_signal_hints_only.yaml`
  - `R13_WTB3_weak_signal_hints_plus_teams_hints.yaml`
- a dedicated launcher smoke config exists:
  - `R13_SMOKE_WTB3_weak_signal_hints_plus_teams_hints.yaml`
- discovery-cache wiring now exists for:
  - DeepLive
  - VisoMaster hints
  - Teams passthrough

## Smoke Path

Smoke for this upgrade should use the normal launcher only:

```bash
cd DeepfakeBench/training
./launch_experiment.sh -y phase2r13-experiments asia-southeast1 \
  experiments/phase2_round13/R13_SMOKE_WTB3_weak_signal_hints_plus_teams_hints.yaml
```

That smoke should be treated as an integration gate, not a quality result.

Current smoke pass criteria:

- nonzero `visomaster_hints_samples`
- nonzero `visomaster_hints_teams_samples`
- nonzero clean direct Teams samples after policy filtering
- training reaches `max_train_steps: 100`
- checkpoint write succeeds

## What Still Must Be Built For New Data

Before the new buckets can be used in proper experiments, each bucket still
needs a concrete source contract.

For each new bucket we need:

- bucket name and prefix
- one real sample path example
- one fake sample path example
- one `manifest.json` example if present
- frame extension (`.png`, `.jpg`, or mixed)
- whether the data is paired real/fake or real-only
- the stable identity key (`sample_id`, `original_video_name`, folder id, etc.)
- any strategy, method, tier, or provenance fields that should survive into
  training reports

## Loader Decision Rule

Use the smallest honest integration path:

- if the bucket already matches the paired `samples/<sample_id>/frames/{real,fake}`
  plus manifest shape, extend `combined_paired`
- if the bucket is a future “proper data” bucket with inventories or manifests,
  keep it aligned with WT-F schema instead of folding it into legacy hints
- if the bucket is eval-only or not truly paired, do not force it into WT-B;
  give it a dedicated eval or unpaired-real path

## What Still Must Be Tested Before Proper Experiments

For each new loader or bucket extension:

- one discovery test proving sample selection and identity extraction
- one iteration test proving frame loading and source naming
- one grouping/family-routing test if the bucket creates new training families
- one YAML parse check for the new config surface
- one launcher smoke with the bucket enabled after the base WT-B smoke passes

Operationally, proper experiments should wait for:

1. the current committed tree to be built into the training image
2. the WT-B smoke to pass remotely
3. the new bucket loader to land with tests
4. one follow-up smoke showing the new bucket actually loads nonzero samples

## Readiness Summary

We are now close to launch-ready for WT-B itself, but not yet for the full
“WT-B plus new buckets” experiment packet.

Current state:

- WT-B code and configs: ready
- WT-B remote smoke: not yet run
- new-bucket loader design: waiting on bucket structure
- proper-experiment matrix: should be finalized only after the first remote
  smoke and the first new-bucket integration are both clean
