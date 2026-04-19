# VisoMaster Bad Data Upload Audit

**Date:** April 17, 2026  
**Author:** Codex  
**Purpose:** identify what VisoMaster-derived data should currently be treated as bad or suspect, and list every confirmed place that data was materialized locally or uploaded to GCS.

## Executive Summary

The confirmed bad VisoMaster data is the **historical baseline `visomaster` branch** generated through the old dataset path that enabled the face parser during compositing. That is the branch already proven in the investigation/correction docs to be under-swapped.

As of **April 17, 2026**, that old baseline data is confirmed in three shared Deep-Live-Cam GCS buckets under the `samples/visomaster_*` namespace:

- `gs://live-deepfake-methods-real-and-fake-frames`
- `gs://live-deepfake-methods-real-and-fake-videos`
- `gs://live-deepfake-methods-real-and-fake-frames-cropped`

The exact current bucket state is:

- frames bucket: **4,230** baseline VisoMaster samples
- videos bucket: **5,594** `real.mp4` objects and **4,230** `fake.mp4` objects for baseline VisoMaster samples
- cropped bucket: **5,589** baseline VisoMaster samples

Important update versus older docs:

- the videos bucket now **does** contain `real.mp4` for all **5,594** baseline VisoMaster sample IDs
- only **4,230** of those baseline sample IDs have `fake.mp4` in the videos bucket
- the current bucket metadata uses **`MINIMAL` / `MODERATE` / `STRONG`**, not `LOW / MEDIUM / HIGH`
- tier metadata is currently recoverable for only **4,230 / 5,589** bad cropped samples
  - `MINIMAL`: **1,888**
  - `MODERATE`: **1,776**
  - `STRONG`: **566**
  - missing tier-bearing companion manifest: **1,359**

I also found a second old-path local branch, `visomaster_resolution`, which should be treated as **suspect** because it was rendered out of the same old `VisoMaster/dataset_output` tree. However, I found **no evidence that `visomaster_resolution` was uploaded to any GCS bucket**.

I found **no GCS evidence** for the newer namespaces:

- `visomaster_corrected_*`
- `visomaster_hdtf_20260416_*`

So the only VisoMaster family currently confirmed in GCS is the old baseline `visomaster_*` namespace.

## What Counts As "Bad Data"

Based on the validated conclusion in `VISOMASTER_DATASET_CORRECTION_FINAL_REPORT.md`, the bad data is the output generated through the old VisoMaster dataset path where:

- `face_parser_enabled = True`
- the parser mask was multiplied into the swap mask
- the final composite became too conservative
- the result often looked under-swapped relative to the live-like path

That makes the historical `visomaster` baseline branch the confirmed bad branch.

## Affected Families

### 1. Confirmed bad: `visomaster`

Canonical local manifest:

- `Deep-Live-Cam/dataset/selected_pairs_visomaster.json`

Current local count:

- **5,594** imported baseline pairs

This is the exact bad baseline set that maps:

- `sample_id`
- `original_video`
- `cropped_video`
- `source_face`
- swap model / restorer metadata

### 2. Suspect local-only branch: `visomaster_resolution`

Canonical local manifest:

- `Deep-Live-Cam/dataset/selected_pairs_visomaster_resolution.json`

Current local count:

- **296** imported resolution-study pairs

Why I am treating it as suspect:

- it was imported from `VisoMaster/dataset_output/work_queue_resolution.json`
- it lives in the old `VisoMaster/dataset_output` tree, not the corrected namespace
- it predates the corrected production rollout

What I did **not** find:

- any `visomaster_Inswapper128_res...` sample IDs in any GCS bucket

So this branch exists locally, but not in cloud storage.

### 3. Not bad / not yet uploaded: `visomaster_corrected`

Current local state:

- replay manifest exists with **5,594** target tasks:
  - `VisoMaster/dataset_output_corrected/manifests/replay_manifest.json`
- only **20** staged `.mp4` files currently exist locally in:
  - `VisoMaster/dataset_output_corrected/videos`

What I found in GCS:

- **zero** `visomaster_corrected_*` objects in the frames, videos, and cropped buckets

### 4. Separate newer branch: `visomaster_hdtf_20260416`

Current local state:

- full manifest exists with **1,328** tasks:
  - `D:\visomaster_hdtf_20260416_output\manifests\full_manifest.json`
- **259** `.mp4` files currently exist locally in:
  - `D:\visomaster_hdtf_20260416_output\videos`

What I found in GCS:

- **zero** `visomaster_hdtf_20260416_*` objects in the frames, videos, and cropped buckets

## Confirmed GCS Upload Locations

VisoMaster data was uploaded into the **shared** Deep-Live-Cam buckets, not into separate VisoMaster-only buckets.

### 1. Frames bucket

Bucket:

- `gs://live-deepfake-methods-real-and-fake-frames`

Namespace:

- `samples/visomaster_*`

Verified current count:

- **4,230** `manifest.json` sample markers

What that means:

- the frames bucket contains the exact baseline sample-ID set whose local landmark metadata status is `success` or `partial`
- I verified the GCS sample-ID set matches the local expected set **exactly**

Contents per uploaded sample:

- `manifest.json`
- `frames/real/*.png`
- `frames/fake/*.png`
- `landmarks/real_landmarks.json`
- `landmarks/fake_landmarks.json`
- `landmarks/metadata.json`

What I did not find:

- no `visomaster_resolution`
- no `visomaster_corrected`
- no `visomaster_hdtf_20260416`

### 2. Videos bucket

Bucket:

- `gs://live-deepfake-methods-real-and-fake-videos`

Namespace:

- `samples/visomaster_*`

Verified current counts:

- **5,594** baseline sample IDs with `real.mp4`
- **4,230** baseline sample IDs with `fake.mp4`

Exact set relationships:

- the `fake.mp4` sample-ID set matches the frames-bucket set **exactly**
- the `real.mp4` sample-ID set matches **all 5,594** baseline `visomaster` sample IDs **exactly**

Interpretation:

- the videos bucket now contains a full baseline `real.mp4` backfill
- only the landmark-ready **4,230** subset has baseline fake videos in the videos bucket
- the remaining **1,364** baseline sample directories have `real.mp4` but not `fake.mp4`

What I did not find:

- no `visomaster_resolution`
- no `visomaster_corrected`
- no `visomaster_hdtf_20260416`

### 3. Cropped frames bucket

Bucket:

- `gs://live-deepfake-methods-real-and-fake-frames-cropped`

Namespace:

- `samples/visomaster_*`

Verified current count:

- **5,589** `manifest.json` sample markers

Exact set relationship:

- the GCS sample-ID set matches the local baseline crop-successful set (`success` or `partial`) **exactly**

Interpretation:

- the cropped bucket contains almost the full bad baseline branch
- the missing **5** baseline sample IDs are the known crop failures
- the cropped bucket manifest itself does **not** carry tier data
- current tier enrichment comes only from the full-frames bucket, and therefore only covers the **4,230** cropped bad samples that also exist in `gs://live-deepfake-methods-real-and-fake-frames`

What I did not find:

- no `visomaster_resolution`
- no `visomaster_corrected`
- no `visomaster_hdtf_20260416`

### 3a. Identity-delta tier truth for the bad cropped set

What I verified on April 17, 2026:

- bad cropped sample IDs in `gs://live-deepfake-methods-real-and-fake-frames-cropped`: **5,589**
- matching bad sample IDs with full-frames manifest + tier metadata in `gs://live-deepfake-methods-real-and-fake-frames`: **4,230**
- bad cropped sample IDs with **no** tier-bearing full-frames companion manifest: **1,359**

Current metadata vocabulary:

- `MINIMAL`
- `MODERATE`
- `STRONG`

I did **not** find `LOW` / `MEDIUM` / `HIGH` in the current bucket manifests.

Exact current tier counts for the metadata-bearing subset (`4,230` sample IDs):

- `MINIMAL`: **1,888**
- `MODERATE`: **1,776**
- `STRONG`: **566**

Important interpretation:

- these counts are exact for the tier-bearing subset of the bad cropped namespace
- they are **not** counts for all **5,589** cropped bad samples, because **1,359** cropped sample IDs currently have no matching full-frames manifest from which to read `tier_data.identity_delta_tier`
- all **5,589** cropped bad sample IDs do exist in the videos bucket under `samples/visomaster_*/`, so this is specifically a **tier-metadata coverage gap**, not a missing-sample gap

Examples from the current cropped-only / no-tier subset:

- `visomaster_CSCS_00417`
- `visomaster_GhostFace-v2_04354`
- `visomaster_InStyleSwapper256-A_10388`
- `visomaster_InStyleSwapper256-B_12115`
- `visomaster_InStyleSwapper256-B_12267`

Per-model tier counts inside the metadata-bearing **4,230** subset:

- `CSCS`: `MINIMAL 284`, `MODERATE 166`, `STRONG 151`
- `GhostFace-v1`: `MINIMAL 256`, `MODERATE 294`, `STRONG 52`
- `GhostFace-v2`: `MINIMAL 203`, `MODERATE 333`, `STRONG 65`
- `GhostFace-v3`: `MINIMAL 191`, `MODERATE 340`, `STRONG 71`
- `InStyleSwapper256-A`: `MINIMAL 374`, `MODERATE 183`, `STRONG 44`
- `InStyleSwapper256-B`: `MINIMAL 345`, `MODERATE 225`, `STRONG 78`
- `Inswapper128`: `MINIMAL 235`, `MODERATE 235`, `STRONG 105`

## Confirmed Local Materialization Locations

These are the main local places where the historical bad or suspect data currently exists.

### Old baseline / old-path render roots

- `VisoMaster/dataset_output/videos`
- `VisoMaster/dataset_output/work_queue.json`
- `VisoMaster/dataset_output/videos/dataset_tagged.json`

Notes:

- `selected_pairs_visomaster.json` preserves the **5,594** imported bad baseline pairings
- the same old `dataset_output` tree also contains the separate `work_queue_resolution.json` study branch

### Deep-Live-Cam import manifests

- `Deep-Live-Cam/dataset/selected_pairs_visomaster.json`
- `Deep-Live-Cam/dataset/selected_pairs_visomaster_resolution.json`

Counts:

- baseline bad branch: **5,594**
- suspect local resolution branch: **296**

### Extracted frame roots

Under `D:\DeepLiveCam_Data`:

- `pair_frames_real`
- `pair_frames_fake`
- `pair_frames_metadata`

Observed counts:

- baseline `visomaster` sample IDs present: **5,594**
- `visomaster_resolution` sample IDs present: **296**

### Cropped frame roots

Under `D:\DeepLiveCam_Data`:

- `pair_frames_real_cropped`
- `pair_frames_fake_cropped`
- `pair_frames_cropped_metadata`

Observed counts:

- baseline `visomaster` sample IDs present: **5,594**
- `visomaster_resolution` sample IDs present: **296**

### Landmark root

Under `D:\DeepLiveCam_Data`:

- `face_landmarks`

Observed counts:

- baseline `visomaster` sample IDs present: **4,266** directories total
- among the **5,594** baseline samples:
  - `success`: **3,933**
  - `partial`: **297**
  - `error`: **6**
  - `<no_metadata_file>`: **30**
  - `<missing_dir>`: **1,328**
- `visomaster_resolution` sample IDs present: **0**

Interpretation:

- only the baseline branch reached the landmark stage
- the landmark-ready baseline subset (`success` + `partial` = **4,230**) is exactly the subset that reached the frames bucket and `fake.mp4` in the videos bucket

## Known Failure Edges

### Baseline crop failures

These **5** baseline sample IDs are missing from the cropped bucket because they are recorded crop failures:

- `visomaster_Inswapper128_08565`
- `visomaster_Inswapper128_08566`
- `visomaster_Inswapper128_08567`
- `visomaster_Inswapper128_08568`
- `visomaster_Inswapper128_08569`

Source:

- `Deep-Live-Cam/dataset/crop_failures_visomaster.json`

### Resolution branch did not reach landmarks or GCS

Current local resolution status:

- frame extraction: **296 / 296 success**
- cropping: **295 success + 1 partial**
- landmarks: **296 missing directories**
- GCS: **0** matching objects found in any of the three buckets

## Practical Cleanup / Replacement Meaning

If the goal is to identify what cloud data needs quarantine or replacement, the currently confirmed cloud-resident bad data is:

- the baseline `visomaster_*` namespace in the three shared buckets

Specifically:

- frames bucket: the exact **4,230**-sample landmark-ready baseline set
- videos bucket:
  - **4,230** baseline `fake.mp4` files
  - **5,594** baseline `real.mp4` files
- cropped bucket: the exact **5,589**-sample crop-successful baseline set

If the goal is to identify what still exists locally and should be treated carefully, include:

- `VisoMaster/dataset_output/...` baseline outputs
- `selected_pairs_visomaster.json`
- all `D:\DeepLiveCam_Data\...` materialized `visomaster_*` frame/crop artifacts
- the local-only `visomaster_resolution` branch

Do **not** currently include in the cloud cleanup scope:

- `visomaster_corrected_*`
- `visomaster_hdtf_20260416_*`

I found no evidence those newer namespaces were uploaded to GCS.

## R13_A Track A: Where The Teams-Played Bad Data Enters Training

The relevant experiment is:

- `DeepfakeBench/training/experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml`

This config has **three** separate VisoMaster-related lanes:

1. direct bad baseline VisoMaster:
   - `combined_paired.visomaster`
   - pulls straight from:
     - `gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_*`
     - `gs://live-deepfake-methods-real-and-fake-frames/samples/visomaster_*`
2. direct Teams passthrough:
   - `combined_paired.teams`
   - pulls from:
     - `gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`
3. merged resolver-driven source:
   - `combined_paired.visomaster_teams_enhanced`
   - reads:
     - `gs://training-job-outputs/cache/visomaster/enhanced_visomaster_resolver_2026-04-06.json`
   - and resolves companions in:
     - `gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`
     - or fallback clean companions in:
       - `gs://live-deepfake-methods-real-and-fake-frames-cropped`

The user question here is specifically about the **Teams-connected** bad-data lanes.

### A. Direct Teams passthrough contamination in `combined_paired.teams`

Current bucket truth for `gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`:

- total manifests: **1,646**
- `pair_complete = true` manifests (the exact subset `R13_A` uses): **1,348**
- among that training-visible subset, bad VisoMaster sample IDs under `samples/visomaster_*`: **237**

Important exact relationship:

- all **237** of those `pair_complete` Teams sample IDs are also present in the bad cropped namespace
- equivalently:
  - `237 / 1,348` pair-complete Teams samples used by `combined_paired.teams` are bad VisoMaster samples that were played through Teams

Per-strategy breakdown of those **237** bad `pair_complete` Teams samples:

- `visomaster_CSCS`: **33**
- `visomaster_GhostFace-v1`: **36**
- `visomaster_GhostFace-v2`: **32**
- `visomaster_GhostFace-v3`: **35**
- `visomaster_InStyleSwapper256-A`: **34**
- `visomaster_InStyleSwapper256-B`: **36**
- `visomaster_Inswapper128`: **29**
- `unknown` but still `sample_id` starts with `visomaster_`: **2**

Those two unlabeled-but-still-bad Teams samples are:

- `visomaster_CSCS_00029`
- `visomaster_InStyleSwapper256-A_10033`

What this means for the actual `R13_A_trackA_teams_enhanced` split:

- existing local Track-A data-truth docs show **196** labeled VisoMaster-through-Teams train pairs under the direct Teams source
- adding the verified **2** unlabeled `visomaster_*` train rows means the effective direct Teams contamination in the **train** split is **198** bad VisoMaster-through-Teams paired objects
- because the bucket currently has **237** `pair_complete` bad Teams sample IDs total, the remaining **39** are in val/test combined

Where they live in GCS:

- `gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2/samples/visomaster_*`

Manifest fingerprint of this lane:

- `source: "teams_capture"`
- `pipeline_version: "teams_2.0"`
- `strategy: "visomaster_<SwapModel>"` for the labeled rows
- paired JPG frames under:
  - `samples/{sample_id}/frames/real/frame_XXXX.jpg`
  - `samples/{sample_id}/frames/fake/frame_XXXX.jpg`

### B. Resolver-driven contamination in `combined_paired.visomaster_teams_enhanced`

Current resolver truth from:

- `gs://training-job-outputs/cache/visomaster/enhanced_visomaster_resolver_2026-04-06.json`

Exact current counts:

- audited base sample IDs: **999**
- `teams_v2_companion`: **54**
- `clean_companion_only`: **943**
- `missing_companion`: **2**

Important exact relationship:

- all **54** `teams_v2_companion` rows are a subset of the same **237** `pair_complete` bad direct-Teams VisoMaster sample IDs above
- all **54** are also in the bad cropped namespace

Per-model breakdown of the **54** true Teams companion rows:

- `CSCS`: **5**
- `GhostFace-v1`: **11**
- `GhostFace-v2`: **5**
- `GhostFace-v3`: **11**
- `InStyleSwapper256-A`: **9**
- `InStyleSwapper256-B`: **7**
- `Inswapper128`: **6**

What this means for the actual `R13_A_trackA_teams_enhanced` split:

- existing local Track-A data-truth docs show **42** `teams_v2_companion` merged train pairs
- therefore **12** more true Teams companion rows sit in val/test combined

But the contamination semantics here are different from the direct Teams lane:

- for `visomaster_teams_enhanced`, the real branch always comes from the resolved companion bucket
- the fake branch is:
  - the original Teams fake only when `fake_branch == "original"`
  - otherwise an enhanced fake from `enhanced-visomaster-cropped`
- in `R13_A_trackA_teams_enhanced.yaml`, `p_original = 0.5`

So:

- all **42** train `teams_v2_companion` rows contribute **Teams-domain real frames**
- only about half of their fake-side emissions are expected to come from the original bad Teams-played fake branch on any given epoch
- the other half come from the enhanced bucket and are therefore **not** direct Teams-played bad fakes

Where this lane is anchored:

- resolver row path:
  - `claimed_real_fake_path = samples/{sample_id}/`
- resolved true-Teams companion path:
  - `resolved_companion_bucket = live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`
  - `resolved_companion_path = samples/{sample_id}/`

Examples of verified `teams_v2_companion` sample IDs:

- `visomaster_CSCS_00007`
- `visomaster_CSCS_00019`
- `visomaster_CSCS_00021`
- `visomaster_CSCS_00028`
- `visomaster_GhostFace-v1_02002`

### Bottom line for the Teams-connected bad data in `R13_A`

If the question is "where does bad VisoMaster data show up after being played through Teams?", the answer is:

- **directly** in `combined_paired.teams` via:
  - `gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2/samples/visomaster_*`
  - current exact training-visible bucket count: **237** `pair_complete` sample IDs
  - current Track-A train count: **198** paired objects
- **indirectly** in `combined_paired.visomaster_teams_enhanced` via:
  - resolver rows whose `resolution_status = teams_v2_companion`
  - current exact resolver count: **54** sample IDs
  - current Track-A train count: **42** paired objects
  - only the `fake_branch == original` half of those emit the original Teams-played bad fake frames

So the Teams-connected contamination is real, but it enters through **two different mechanisms**:

- a direct Teams-passthrough lane
- a merged resolver-driven lane whose true Teams subset is small and whose fake branch is only partially Teams-native

## Verification Notes

This audit used both:

- local manifests / metadata in `VisoMaster` and `Deep-Live-Cam`
- direct Google Cloud Storage queries on **April 17, 2026**
- local Track-A truth / composition docs in:
  - `DeepfakeBench/training/docs/research_2026-04-15_round2/02_target_domain_data_truth.md`
  - `DeepfakeBench/training/docs/research_2026-04-15/04_data_composition_and_curriculum_opportunities.md`

The strongest exact-match checks were:

- local baseline landmark-ready set == frames bucket `visomaster_*` set (**4,230** exact match)
- local baseline landmark-ready set == videos bucket `fake.mp4` `visomaster_*` set (**4,230** exact match)
- local baseline full imported set == videos bucket `real.mp4` `visomaster_*` set (**5,594** exact match)
- local baseline crop-successful set == cropped bucket `visomaster_*` set (**5,589** exact match)
- direct Teams VisoMaster `teams_v2_companion` resolver rows == subset of current `pair_complete` Teams `visomaster_*` sample IDs (**54** exact subset match)
