# Proper Data Schema And Future Manifests

**Date:** April 17, 2026

## Purpose

Define the canonical way future proper VisoMaster and Teams-parallel data must
enter this repo so it stays provenance-clean, split-hygienic, and directly
scorable by the existing target-domain validation and promotion tooling.

This is a WT-F specification document. It is intentionally about **future
proper data only**. It does not rename or reinterpret the current frozen
evaluation manifests, the current hint lanes, or the old merged
`visomaster_teams_enhanced` lane.

## Why The Current Assumptions Are Too Weak

The current repo already has manifest tooling, but its assumptions are too weak
for future proper data:

1. `build_teams_target_domain_manifest.py` infers provenance from filename
   prefixes and session names.
2. `COORDINATED_CAPTURE_PLAN.md` and `receiver_server.py` only preserve raw
   `sample_id` plus `real|fake` output directories.
3. `COORDINATED_CAPTURE_PLAN.md` still references `upload_teams_to_gcs.py`,
   which is not present in this checkout.
4. The old merged `visomaster_teams_enhanced` lane proved that hiding multiple
   conditions inside one convenience lane makes later interpretation brittle.

So future proper data cannot enter as:

- another prefix-rule freeze
- another flat `sample_id/real|fake` upload
- another merged convenience lane
- another relabel of `visomaster hints`

Future proper data must enter through an **explicit inventory** that declares
the exact condition of every variant.

## Canonical Exact Lanes

These six lanes are the only canonical exact-condition lanes for future proper
data:

| Lane | Label | Transport | Generator family | Enhancement | Meaning |
| --- | --- | --- | --- | --- | --- |
| `proper_real_clean` | real | clean | none | none | clean real anchor |
| `proper_real_teams` | real | Teams | none | none | real after Teams transport |
| `proper_visomaster_clean` | fake | clean | VisoMaster | none | clean VisoMaster fake |
| `proper_visomaster_enhanced_clean` | fake | clean | VisoMaster | enhanced | clean enhanced fake |
| `proper_visomaster_teams` | fake | Teams | VisoMaster | none | VisoMaster fake after Teams transport |
| `proper_visomaster_enhanced_teams` | fake | Teams | VisoMaster | enhanced | enhanced fake after Teams transport |

Interpretation rules:

- clean versus Teams transport is always explicit
- enhancement is always explicit
- the exact target condition `enhanced fake after Teams processing` is
  `proper_visomaster_enhanced_teams`
- none of these lanes may be folded into `visomaster hints`,
  `visomaster hints (teams)`, or `visomaster_teams_enhanced`

## Representation Rules

These are mandatory.

1. The base unit is `base_capture_id`, not `sample_id`.
2. Split hygiene is controlled by `split_group_id` and happens **before**
   clean-versus-Teams or enhanced-versus-unenhanced variants are expanded.
3. All variants derived from the same base capture must share the same
   `base_capture_id` and `split_group_id`.
4. `generator_method` must be explicit for every fake variant.
5. `playback_path` must be explicit for every variant.
   Examples: `direct_capture`, `obs_virtual_cam_to_teams`.
6. The raw capture directory is evidence, not semantics.
   The inventory file is the semantic source of truth.
7. Future proper data uses `proper_*` names only.
   Do not add aliases that reuse current frozen slice names such as
   `teams_fake_all` or `teams_real_all`.

## Inventory Contract

The canonical input is
`DeepfakeBench/training/arena/future_proper_data_inventory.template.yaml`.

Required top-level fields:

- `inventory_version`
- `wave_id`
- `split_seed`
- `lockbox_ratio`
- `source_logs`
- `captures`

Required capture-level fields:

- `base_capture_id`
- `identity_id`
- `capture_session_id`
- `split_group_id`
- `quality_band`
- `face_scale_band`
- `variants`

Required variant-level fields:

- `variant_id`
- `label`
- `transport`
- `enhancement`
- `playback_path`
- exactly one of:
  - `frame_root`
  - `frame_paths`

Extra fake-only fields:

- `generator_family`
- `generator_method`

## Manifest Row Contract

`build_future_proper_target_domain_manifest.py` emits rows with:

- `label`
- `lane`
- `method`
- `video_id`
- `frame_paths`
- `identity`
- `identity_key`
- `split`
- `slices`
- `base_capture_id`
- `identity_id`
- `capture_session_id`
- `quality_band`
- `face_scale_band`
- `transport`
- `generator_family`
- `generator_method`
- `enhancement`
- `playback_path`
- `source_kind`

Important derivations:

- `lane` is computed only from explicit metadata
- `method` is:
  - the real lane name for reals
  - `<lane>__<method_slug>` for fakes
- `split` is computed from `split_group_id`
- `slices` include both exact-condition and rollup tags

## Canonical Eval Slices

Exact-condition eval slices are identical to the exact lane names:

- `proper_real_clean`
- `proper_real_teams`
- `proper_visomaster_clean`
- `proper_visomaster_enhanced_clean`
- `proper_visomaster_teams`
- `proper_visomaster_enhanced_teams`

Rollup eval slices are:

- `proper_real_all`
- `proper_fake_all`
- `proper_fake_clean_all`
- `proper_fake_teams_all`

Method-level exact-condition slices are:

- `proper_visomaster_clean__<method_slug>`
- `proper_visomaster_enhanced_clean__<method_slug>`
- `proper_visomaster_teams__<method_slug>`
- `proper_visomaster_enhanced_teams__<method_slug>`

## Suite Naming Contract With WT-E

To stay compatible with the current validation runner and promotion scorer,
future suite names must continue to use:

`<slice>_<split>`

Examples:

- `proper_real_teams_dev`
- `proper_real_teams_lockbox`
- `proper_fake_teams_all_dev`
- `proper_fake_teams_all_lockbox`
- `proper_visomaster_enhanced_teams_dev`
- `proper_visomaster_enhanced_teams_lockbox`

This keeps future proper-data scoring on the same contract shape already used by
`run_target_domain_validation_sequential.py` and
`score_teams_promotion_contract.py`, but without inventing local aliases.

The concrete template is:

- `DeepfakeBench/training/arena/target_domain_suites.proper_data_future.template.yaml`

## Promotion-Oriented Defaults

If WT-E wants to score future proper data with the existing promotion scorer,
the recommended contract names are:

- `dev_real_suite`: `proper_real_teams_dev`
- `dev_fake_suites`:
  - `proper_fake_teams_all_dev`
  - `proper_visomaster_teams_dev`
  - `proper_visomaster_enhanced_teams_dev`
- `lockbox_real_suite`: `proper_real_teams_lockbox`
- `lockbox_fake_suite`: `proper_fake_teams_all_lockbox`

Interpretation:

- `proper_fake_teams_all_*` is the aggregated Teams-transport fake gate
- `proper_visomaster_teams_*` and
  `proper_visomaster_enhanced_teams_*` remain explicit sub-slices so the exact
  target condition can be scored without ad hoc renaming

## What This Spec Forbids

- no future proper data inside `visomaster hints`
- no future proper data inside `visomaster hints (teams)`
- no future proper data inside the old merged
  `visomaster_teams_enhanced` lane
- no provenance inference from filename prefixes alone
- no scorecard-only aliases that hide the actual proper-data condition

## Checked-In WT-F Artifacts

- schema doc:
  - `DeepfakeBench/training/docs/PROPER_DATA_SCHEMA_AND_FUTURE_MANIFESTS_2026-04-17.md`
- capture/provenance doc:
  - `DeepfakeBench/training/docs/PROPER_DATA_CAPTURE_INVENTORY_AND_PROVENANCE_2026-04-17.md`
- inventory template:
  - `DeepfakeBench/training/arena/future_proper_data_inventory.template.yaml`
- manifest builder:
  - `DeepfakeBench/training/arena/build_future_proper_target_domain_manifest.py`
- future suite template:
  - `DeepfakeBench/training/arena/target_domain_suites.proper_data_future.template.yaml`
