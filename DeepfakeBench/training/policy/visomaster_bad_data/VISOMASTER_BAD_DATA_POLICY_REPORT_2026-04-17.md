# VisoMaster Bad Data Policy Report

**Date:** 2026-04-17

**Manifest:** `DeepfakeBench/training/debug/VISOMASTER_BAD_DATA_POLICY_MANIFEST_2026-04-17.csv`

## Executive Summary

We are **not** treating the historical bad `visomaster_*` bucket data as method-faithful data anymore.

The issue is not that the data is completely signal-free. The issue is that the fake creation on these samples did not work properly, so the outputs can still carry **some** weak hints from the generator stack while **not** being representative examples of the nominal method they claim to be, such as `GhostFace-v3`, `GhostFace-v2`, or `Inswapper128`.

Because of that, this data should no longer be used as if it were correct per-method supervision. Instead:

- keep a small controlled baseline subset and label it **`visomaster hints`**
- keep a small controlled Teams-played subset and label it **`visomaster hints (teams)`**
- **ignore** all remaining bad VisoMaster data in training
- **delete only** the few explicitly named bad datapoints listed below

This is a mitigation step so we do not throw away every weak signal in this family while we wait for more **actually correct VisoMaster** data, which is expected to arrive soon.

## What The New Labels Mean

- **`visomaster hints`**: weak-signal auxiliary data from bad baseline VisoMaster samples. These are **not** correct method-faithful examples of the nominal swap model.
- **`visomaster hints (teams)`**: weak-signal auxiliary data from the same bad VisoMaster family after Teams playback. These carry both bad-VisoMaster signal and Teams-processing signal, so they should be treated as a separate weak label from the baseline subset.

The practical consequence is that retained samples should be consumed only under these new weak labels, **not** under their nominal generator names.

## Operational Policy

1. Keep all non-`CSCS` `pair_complete` Teams-played bad VisoMaster samples, except the explicitly named bad datapoint `visomaster_InStyleSwapper256-B_12029`, and label them **`visomaster hints (teams)`**.
2. From the non-Teams baseline bad VisoMaster pool, take a deterministic random sample of **40** samples per non-`CSCS` method per tier for `MODERATE` and `STRONG`, and label them **`visomaster hints`**.
3. Ignore every other bad VisoMaster sample in training manifests, configs, and experiment definitions.
4. Delete only the explicitly named bad datapoints. Everything else should remain in the buckets but be excluded by manifest policy.

The baseline subset is deterministic random sampling with seed **20260417** so the same manifest can be regenerated reproducibly.

## High-Level Counts

### Unique Bad Source Sample IDs (`live-deepfake-methods-real-and-fake-frames-cropped`)

| Category | Count | Share |
| --- | ---: | ---: |
| Keep as `visomaster hints` | 480 | 8.59% |
| Keep as `visomaster hints (teams)` | 202 | 3.61% |
| Keep total | 682 | 12.20% |
| Ignore | 4904 | 87.74% |
| Delete | 3 | 0.05% |
| Total bad cropped sample IDs | 5589 | 100.00% |

So under the new policy we **keep 682 / 5589 unique bad sample IDs** and **stop using 4907 / 5589** of them. Most of the stopped portion is **ignored**, not physically deleted.

### Training Rows If Baseline And Teams Are Separate Lanes

| Lane | Keep | Ignore | Delete | Total |
| --- | ---: | ---: | ---: | ---: |
| Baseline bad VisoMaster lane | 480 | 5106 | 3 | 5589 |
| Teams-played bad VisoMaster lane | 202 | 34 | 1 | 237 |
| Combined training rows | 682 | 5140 | 4 | 5826 |

This is the larger experiment-planning impact: if someone had been thinking of the bad baseline and the Teams-played bad data as separate usable lanes, the policy now reduces that combined candidate pool from **5826 rows** to **682 rows**, with **5140 rows ignored** and **4 rows deleted**.

## Teams Subset Decision

The Teams-played bad VisoMaster subset is being retained because it is small and still useful as a weak signal, but it must be treated as **`visomaster hints (teams)`**, not as a method-faithful extension of the nominal generator.

- total `pair_complete` bad Teams samples: **237**
- kept as `visomaster hints (teams)`: **202**
- ignored `CSCS`: **34**
- deleted named bad datapoint: **1**

## Baseline Subset Decision

The retained baseline weak-signal subset is sampled only from non-Teams, non-`CSCS`, `MODERATE` and `STRONG` tier samples. `MINIMAL` is excluded, the no-tier cropped-only remainder is excluded, and anything not selected into the deterministic random sample is ignored.

| Method | Tier | Available After Exclusions | Kept |
| --- | --- | ---: | ---: |
| GhostFace-v1 | MODERATE | 274 | 40 |
| GhostFace-v1 | STRONG | 48 | 40 |
| GhostFace-v2 | MODERATE | 316 | 40 |
| GhostFace-v2 | STRONG | 57 | 40 |
| GhostFace-v3 | MODERATE | 317 | 40 |
| GhostFace-v3 | STRONG | 66 | 40 |
| InStyleSwapper256-A | MODERATE | 168 | 40 |
| InStyleSwapper256-A | STRONG | 41 | 40 |
| InStyleSwapper256-B | MODERATE | 209 | 40 |
| InStyleSwapper256-B | STRONG | 67 | 40 |
| Inswapper128 | MODERATE | 218 | 40 |
| Inswapper128 | STRONG | 102 | 40 |

Every method-tier bucket still had at least **40** eligible samples after exclusions, so the planned quota was fully satisfied in all 12 non-`CSCS` method-tier groups.

## Where The Ignored Mass Comes From

Most of the stopped data is being **ignored**, not deleted. The ignored volume comes from five clear sources:

| Ignore Reason | Count |
| --- | ---: |
| Cropped-only samples with no tier-bearing frames companion | 1359 |
| Tier-bearing MINIMAL samples excluded by policy | 1541 |
| Non-Teams CSCS samples excluded | 567 |
| Teams pair-complete CSCS samples excluded | 34 |
| Eligible non-CSCS MODERATE/STRONG baseline samples not selected into the 40-sample quota | 1403 |

This breakdown is why the new policy is such a large contraction. We are excluding:

- the full cropped-only remainder without tier-bearing companion metadata
- the entire `MINIMAL` tier
- all `CSCS`
- the large remainder of eligible `MODERATE` and `STRONG` samples that falls outside the controlled 40-per-method-per-tier quota

## Explicit Deletes

Only the following named datapoints should be deleted from the relevant bad-data buckets. Everything else should stay in storage and be controlled through the manifest.

| Sample ID | Swap Model | Tier | In Cropped | In Frames | In Teams Pair Complete |
| --- | --- | --- | ---: | ---: | ---: |
| `visomaster_GhostFace-v2_04004` | GhostFace-v2 | STRONG | yes | yes | no |
| `visomaster_InStyleSwapper256-B_12029` | InStyleSwapper256-B | STRONG | yes | yes | yes |
| `visomaster_InStyleSwapper256-B_12395` | InStyleSwapper256-B | STRONG | yes | yes | no |

## What The Data Specialist Should Do

- Consume the manifest as the source of truth for this bad VisoMaster family.
- Treat `policy_action == keep` rows with `policy_label == "visomaster hints"` as auxiliary weak-signal baseline data.
- Treat `policy_action == keep` rows with `policy_label == "visomaster hints (teams)"` as auxiliary weak-signal Teams data.
- Treat all `policy_action == ignore` rows as excluded from training.
- Treat all `policy_action == delete` rows as the only samples that should be physically removed from the relevant bad-data buckets.
- Do **not** train the retained rows under their nominal per-method labels. The retained rows are no longer a claim about the correctness of `GhostFace-v1`, `GhostFace-v2`, `GhostFace-v3`, `InStyleSwapper256-A`, `InStyleSwapper256-B`, or `Inswapper128` on those sample IDs.

## Interpretation

This is effectively a relabeling and quarantine policy:

- historical bad VisoMaster data is now being reframed as **weak VisoMaster-hint data**
- the Teams-played subset is being reframed as **weak VisoMaster-hint-plus-Teams data**
- the bulk of the old bad data is no longer eligible for normal method supervision

That is a big experimental change. Any prior or planned experiment that assumed this data was method-faithful should be considered outdated and should be switched over to this manifest-driven policy.

## Forward Looking Note

More actually correct VisoMaster data is planned to arrive soon. When that happens, it should be treated as a clean replacement lane rather than merged back into this weak-signal stopgap bucket policy.
