# GCS-side identity-overlap audit — FACTS (2026-05-12)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, unfortunately, remarkably, shortcut-aligned, lucky, confirmed, refuted.
> Numbers + tables + cross-references only. Interpretation belongs in `../cpu_diagnostics_2026-05-12_d1_d5_synthesis/D1_D5_CRITIC_REVIEW_2026-05-12.md` §addendum (open loop `train-bucket-identity-overlap-gcs-audit`).
>
> **Authoring note**: this doc was originally drafted from partial subagent outputs (30 manifests inspected inline by parent agent). It was re-run 2026-05-12 night end-to-end via `run_gcs_audit.py`: 250 manifests pulled across all 10 strategy prefixes, 0 fetch errors. The numbers below reflect the re-run.
>
> **Scope**: closes the open loop `train-bucket-identity-overlap-gcs-audit` opened by the critic-review §addendum 2026-05-12 evening. The question: do any of 8 chronic-cohort identity names (and their case/spelling variants) appear in P8A's Teams training source bucket `gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2/`, either at the sample_id level or in sample manifest fields?
>
> **Identities tested** (8 canonical, 23 patterns including variants): `real_dor`/`Dor`/`dor`/`roee_dor`/`Roee`/`roee`, `Cam_Test`/`cam_test`/`CamTest`/`test_cam`/`Test_Cam`, `PC_Generator`/`pc_generator`/`pcgen`/`PC_Gen`, `dor_shkedi`, `Roy_D`/`roy_d`, `bla_bla_chow`, `Md_noyn_Sharker`/`noyn`/`Md_noyn`, `xiang`.
>
> **Inputs**:
> - GCS bucket: `gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2/` (1346 samples per `bucket_metadata.json`)
> - Eval-side reference: `analysis/train_overlap_audit_2026-05-12/TRAIN_OVERLAP_FACTS_2026-05-12.md`
> - Authenticated as: `roee@dtectvision.ai` (verified via `gcloud auth list`)
>
> **Script**: `run_gcs_audit.py` (end-to-end reproducible; 250 manifests / 10 strategy prefixes / 25 per prefix). Authentication, enumeration, manifest pull, listing+manifest grep, schema audit all automated.

---

## 1. Method

1. **Authentication check**: `gcloud auth list` → `roee@dtectvision.ai` ACTIVE (`outputs/auth.txt`).
2. **Bucket metadata**: `gsutil cat gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2/metadata.json` → 1346 complete_pairs across 10 strategy partitions; 26,461 real frames + 26,427 fake frames; sourced 2026-03-05 via teams-capture pipeline from parent bucket `live-deepfake-methods-real-and-fake-frames-cropped` (`outputs/bucket_metadata.json`).
3. **Sample listing**: `gsutil ls gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2/samples/` → 1646 paths in `outputs/training_bucket_sample_ids.txt` (1646 > 1346 reported as `complete_pairs` in metadata; the 300-row gap is uninvestigated and may include incomplete-pair samples that metadata excluded from totals).
4. **Sample-id naming convention check**: read first 20 entries; sample_ids follow `<strategy>_<NNNNN>` pattern (e.g., `edge_cases_0000`, `minimal_processing_0010`, `visomaster_CSCS_00000`).
5. **Stratified manifest pull** (250 samples = 25 × 10 strategy prefixes): `gsutil cat <sample>/manifest.json` for each; results in `outputs/sample_manifest_examples.json` (10 examples kept), schema in `outputs/manifest_field_schema.txt`, picks in `outputs/_picked_sample_ids.txt`.
6. **Listing-level identity-name grep** (23 patterns × 1646 sample_ids): case-insensitive substring match in Python. Results in `outputs/identity_matches.csv` column `n_listing_matches`.
7. **Manifest-level identity-name grep** (23 patterns × 250 manifests): non-alphanumeric-bounded token match against the JSON-serialized manifest body. Results in `outputs/identity_matches.csv` column `n_manifest_matches`.
8. **Cross-reference**: read `experiments/phase2_round13/R13_RLP8_01_unfreeze_clip_codec.yaml` lines 180-280 to verify which lanes (training vs OOD-readout) reference teams-v2.

---

## 2. Training-bucket sample_id enumeration

Strategy-level partition counts compared between metadata (`complete_pairs`) and `gsutil ls` enumeration:

| Strategy | metadata `complete_pairs` | listing-enumeration | delta |
|---|---:|---:|---:|
| edge_cases | 389 | 389 | 0 |
| minimal_processing | 410 | 410 | 0 |
| quality_enhancement | 312 | 312 | 0 |
| visomaster_CSCS | 33 | 151 | +118 |
| visomaster_GhostFace-v1 | 36 | 145 | +109 |
| visomaster_GhostFace-v2 | 32 | 104 | +72 |
| visomaster_GhostFace-v3 | 35 | 35 | 0 |
| visomaster_InStyleSwapper256-A | 34 | 35 | +1 |
| visomaster_InStyleSwapper256-B | 36 | 36 | 0 |
| visomaster_Inswapper128 | 29 | 29 | 0 |
| **Total** | **1,346** | **1,646** | **+300** |

Frame totals (from metadata): 26,461 real + 26,427 fake.

The 300-sample gap concentrates in 3 visomaster strategy partitions (CSCS, GhostFace-v1, GhostFace-v2); the listing contains samples that the bucket metadata does not count as `complete_pairs`. Whether these extra samples are accessed by the dataloader is a YAML-config question (§5), not visible in the listing alone.

Sample_id naming convention: `<strategy>_<NNNNN>` (4-5 digit zero-padded index). No identity strings appear in any sample_id at the listing level (see §4 grep results).

---

## 3. Sample-manifest field structure (250 manifests)

Source: `outputs/sample_manifest_examples.json` (10 examples kept), `outputs/manifest_field_schema.txt` (union schema).

**Union of all keys observed across 250 manifests** (10 strategies × 25 samples each):

```
fake_complete, frame_count_fake, frame_count_real, image_format,
original_anchor_frames, original_consecutive_frames, original_cropped,
original_cropped_video_name, original_frame_count_fake, original_frame_count_real,
original_has_landmarks, original_original_video_name, original_pipeline_version,
original_trimmed_video_name, pair_complete, pipeline_version, real_complete,
sample_id, source, strategy, teams_fake_duration_s, teams_fake_frames_discarded,
teams_fake_frames_written, teams_real_duration_s, teams_real_frames_discarded,
teams_real_frames_written, uploaded_at
```

No `identity`, `subject`, `person`, `source_identity`, `original_identity_id`, or any related person-level field is present in any of the 250 manifests.

The only fields that can identify the source content are the three video-name fields. Sample of values pulled inline:

| Sample | original_cropped_video_name | original_trimmed_video_name | original_original_video_name |
|---|---|---|---|
| edge_cases_0000 | `edge_cases_0000.mp4` | `trimmed_cropped_lm0hNQmOdFg.mp4` | `cropped_lm0hNQmOdFg.mp4` |
| minimal_processing_0000 | `minimal_processing_0000.mp4` | `trimmed_cropped_mJWOWHvCZW4.mp4` | `cropped_mJWOWHvCZW4.mp4` |
| quality_enhancement_0003 | `quality_enhancement_0003.mp4` | `trimmed_cropped_eSEk_iOeZKk.mp4` | `cropped_eSEk_iOeZKk.mp4` |
| visomaster_Inswapper128_08029 | `Inswapper128_08029_trimmed_cropped_9MVRX_ZlFTM.mp4` | (empty) | `trimmed_cropped_9MVRX_ZlFTM.mp4` |

The source-video identifier in every sampled manifest is an 11-character alphanumeric token matching the YouTube video-ID format (e.g., `lm0hNQmOdFg`, `mJWOWHvCZW4`, `eSEk_iOeZKk`, `9MVRX_ZlFTM`). Sample_id and any per-frame jpg filename derive from the strategy prefix + index, not from the source-video token.

---

## 4. Per-identity match table (23 patterns × 1,646 listings + 250 manifests)

Source: `outputs/identity_matches.csv`. Manifest-level grep uses a non-alphanumeric-bounded token match against the JSON-serialized body (e.g., a sample whose `original_trimmed_video_name` literally contained `dor_` would have produced a hit).

| Canonical identity | Patterns tested | n_listing_matches (of 1,646) | n_manifest_matches (of 250) |
|---|---|---:|---:|
| `real_dor` | real_dor, Dor, dor, roee_dor, Roee, roee | 0 | 0 |
| `Cam_Test` | Cam_Test, cam_test, CamTest, test_cam, Test_Cam | 0 | 0 |
| `PC_Generator` | PC_Generator, pc_generator, pcgen, PC_Gen | 0 | 0 |
| `dor_shkedi` | dor_shkedi | 0 | 0 |
| `Roy_D` | Roy_D, roy_d | 0 | 0 |
| `bla_bla_chow` | bla_bla_chow | 0 | 0 |
| `Md_noyn_Sharker` | Md_noyn_Sharker, noyn, Md_noyn | 0 | 0 |
| `xiang` | xiang | 0 | 0 |

All 23 identity-name patterns: 0 matches at the sample_id level (n=1,646), 0 matches at the manifest-field level (n=250).

---

## 5. Cross-reference to P8A training yaml

`experiments/phase2_round13/R13_RLP8_01_unfreeze_clip_codec.yaml` references the teams-v2 bucket at four loci. Read 2026-05-12 night:

| YAML lane | Lines | Bucket | Loss-bearing? | Notes |
|---|---:|---|---|---|
| `combined_paired.teams` | 181-186 | teams-v2 | YES (training data) | `enabled: true`; this is P8A's Teams training source |
| `combined_paired.visomaster_hints_teams` | 188-192 | teams-v2 | NO (`enabled: false`) | disabled per project memory `visomaster_hints_lanes_bad_data` |
| `ood_monitoring.external_real_sources` | 251-259 | teams-v2 | NO (OOD readout) | `method: teams_ood_real`, `max_videos: 300` |
| `ood_monitoring.external_fake_sources` | 269-278 | teams-v2 | NO (OOD readout) | `method: teams_ood_fake`, `path_exclude_contains: ["/visomaster_"]` |

A separately-named bucket `real-teams-dor-roee` (lines 261-267) hosts dor/roee identity captures and is referenced under `readout_only_external_real_sources` — it is NOT teams-v2 and is NOT loss-bearing during training.

No per-identity filter (allow/exclude) is applied to the `combined_paired.teams` lane. The lane reads all `pair_complete: true` samples from teams-v2 subject to `require_pair_complete: true` and `apply_bad_data_policy: true`.

---

## 6. Output artifacts

- `run_gcs_audit.py` — end-to-end reproducible audit script (1 file, 0 external state).
- `outputs/auth.txt` — `gcloud auth list` output at audit time.
- `outputs/bucket_metadata.json` — bucket-level metadata (strategy breakdown, totals).
- `outputs/training_bucket_sample_ids.txt` — 1,646 `gsutil ls` rows.
- `outputs/_picked_sample_ids.txt` — 250 stratified sample_ids picked for manifest inspection (25 × 10 strategy prefixes).
- `outputs/sample_manifest_examples.json` — 10 example manifest JSON bodies (one per strategy prefix).
- `outputs/manifest_field_schema.txt` — union of keys observed across all 250 manifests.
- `outputs/identity_matches.csv` — 23-row per-pattern match table (listing + manifest grep counts).

---

## 7. Caveats

- **Identity-level overlap is not detectable from manifest fields alone.** The training-bucket samples key on YouTube-video-ID-style 11-character tokens (e.g., `lm0hNQmOdFg`), not person names. The eval bucket `teams-faces-data-test-2914-fake-4420-real-feb-28` keys on person names in per-frame jpg filenames (e.g., `real_dor__frame_*.png`). A simple name grep cannot detect identity overlap because the two keying schemes don't share a join key. Detecting person-level overlap would require: (a) external knowledge mapping each chronic-cohort identity to the YouTube video IDs that contain them, then cross-referencing those tokens against `original_trimmed_video_name`; OR (b) content-side face-embedding comparison between eval and training frames. Neither is in this audit's scope.
- The 1,646-row listing exceeds the 1,346 `complete_pairs` reported in `metadata.json`; the 300-row gap concentrates in three visomaster strategy partitions (CSCS, GhostFace-v1, GhostFace-v2). Whether the dataloader actually reads these extra samples depends on YAML-config flags (`require_pair_complete: true` on line 184 likely excludes them); not investigated further.
- The manifest pull stratified at 25 samples per strategy prefix (250 total). Schema is fully consistent across all 250: 27 keys, no identity-related fields anywhere. n=250 (vs full 1,346 = 18.6% coverage) is sufficient for the structural claim "no identity field exists in the manifest schema" but does not exclude a rare per-sample annotation that does not appear in the chosen 25-per-prefix subset.
- This audit closes the open loop `train-bucket-identity-overlap-gcs-audit` (LOW severity) as: **the manifest-side audit returns null, AND the manifest schema does not contain a field that would let a name-based audit return non-null even if overlap exists**. The null result has two possible interpretations: (i) the 8 chronic-cohort identities are genuinely not in the training data, OR (ii) they ARE in the training data via YouTube source videos that this audit's method cannot link back to person names.
- The audit is read-only; no GCS writes were performed.

---

## 8. Direct observations

1. The teams-v2 training bucket reports 1,346 `complete_pairs` (`metadata.json`) and 1,646 sample directories under `samples/` (`gsutil ls`); the 300-sample gap concentrates in 3 visomaster partitions (§2).
2. Sample_id naming is uniformly `<strategy>_<NNNNN>` across 10 strategy partitions; 0 of 1,646 sample_ids match any of the 23 identity-name patterns (§4).
3. The manifest schema (27 keys, union across 250 inspected manifests, §3) contains no `identity`, `subject`, `person`, `source_identity`, or `original_identity_id` field. The only source-content identifier is `original_trimmed_video_name` / `original_original_video_name`, which holds an 11-character alphanumeric YouTube-video-ID-style token.
4. 0 of 250 inspected manifests contain a token-bounded match for any of the 23 identity-name patterns when the JSON body is searched as a string (§4).
5. The teams-v2 bucket is referenced as P8A training data via the `combined_paired.teams` lane (lines 181-186); the same bucket is referenced for OOD-readout (lines 251-278) but the readout references do not contribute to training loss (§5). The chronic-cohort identity `dor`/`roee` is captured in a separately named bucket `real-teams-dor-roee`, which is wired to `readout_only_external_real_sources` (lines 261-267) and is NOT loss-bearing during training.
6. The training-data anonymization scheme (YouTube-video-ID tokens in manifests, no person-name fields) means a name-based audit cannot resolve "do the 8 chronic-cohort identities appear in teams-v2 training" in either direction (§7).
