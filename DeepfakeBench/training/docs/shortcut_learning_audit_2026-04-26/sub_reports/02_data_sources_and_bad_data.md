# Data Sources, Manifests, and Bad-Data Policies

**Audit date:** 2026-04-26
**Reference config:** `experiments/phase2_round13/R13_P10_SYM_baseline.yaml` (representative P10 entry; the SYM/GRL/baseline P10 yamls share `combined_paired` block bit-for-bit except for a few aug knobs).
**Codebase touchpoints:** `data/sources/combined_paired.py`, `data/sources/{deeplive,df40_paired,visomaster,proper_data}.py`, `data/validation_sources.py`, `data/augmentations/pipelines.py`, `utils/grouping.py`, `policy/visomaster_bad_data/*`, `arena/inventories/*`, `arena/manifests/*`.

---

## 0. Executive verdict

The training data composition is **not symmetric across labels**, and the assembly logic — even though it does correctly identity-split reals — has multiple structural avenues for the model to learn shortcuts that are unrelated to "is this face artificially synthesized":

1. The `family_aware` augmentation router (used in every R13 yaml prior to the explicit P10 SYM packet, and still the **default** in `pipelines.py`) installs an **explicit per-class quality bias**: the fake families are always pushed to lower JPEG floor / heavier blur / more aggressive `OneOf` p, while the real families are pushed in the opposite direction. This is not a discovered correlation, it is hard-coded.
2. Sources are **heavily label-correlated**: `df40` is fake-only (real frames come from FF++ donor identities but that's a single homogeneous pool), `external_training_reals` (VCD) is **real-only**, `external_youtube_avspeech` is **real-only and only used for OOD/stress**, and `proper_visomaster_*_clean` is fake-only — there are no `proper_real_clean_only` samples that aren't also paired with a fake. A model that picked up bucket-level statistics would do quite well on the trainer's own splits and would have very little to do with face-swap detection.
3. Family weights asymmetrically up-weight Teams and Teams-played samples (5x relative to df40) while explicitly down-weighting df40 fake (0.15). This is a deliberate domain-shift bias that biases the loss toward the Teams codec and away from clean methods. It is not in itself a bug, but it does mean the trainer's auc/eer numbers (`AUC≥0.99, EER≤0.02`) heavily reflect df40 performance and substantially weight Teams (which is a small lockbox).
4. Identity-split is correct **for the realpool family** (DeepLive/VisoMaster/visomaster_enhanced share the same `realpool_<videoID>` prefix, so the same source identity can never appear in two splits **within that family**), but it is **not coordinated across pools**. There is no guard that prevents a `dor_shkedi` Teams identity from ending up in lockbox while a clean-domain VisoMaster sample of the same person ends up in train — these would be flagged differently, but the same face would be in both splits.

The single most concrete shortcut to investigate first is **family-aware augmentation routing as a label-leak via image statistics** — see Section 11.A.

---

## 1. Sources enabled in P10 training

P10 reads `data_source: combined_paired`, so the assembly is driven by `combined_paired` block. Below, "Enabled?" means `combined_paired.<source>.enabled: true` in the P10 baseline yaml.

| Source | Real / Fake | What it contains | GCS bucket | Manifest / inventory | Identity-split rule | Enabled in P10? | Family weight (P10) |
|---|---|---|---|---|---|---|---|
| **df40** | both (paired) | Academic deepfake methods: simswap, facedancer, blendface, e4s, inswap, mobileswap, uniface | `df40-frames-recropped-rfa85` | `dataset/df40_pairs/df40-pair-matching.json` (in-repo) | `df40_<target_identity>` prefix; `hash_stable` global split with seed 737 | yes | df40_fake=0.15, df40_real=0.4 |
| **deeplive** | both (paired) | Live-DeepFake / DeepLiveCam captures of swaps. Strategies: edge_cases, minimal_processing, quality_enhancement, edge_cases_enhanced, minimal_processing_enhanced | `live-deepfake-methods-real-and-fake-frames-cropped` (samples/ prefix) | bucket-walked at startup; cached in `gs://training-job-outputs/cache/...` | `realpool_<videoID-from-original_video_name>` (shared with VisoMaster) | yes | deeplive_non_enhanced_fake=2.5, deeplive_enhanced_fake=3.0, deeplive_teams_fake=5.0, deeplive_teams_real=4.0 |
| **visomaster** (legacy bad branch) | both (paired) | Old `face_parser_enabled=True` baseline — confirmed UNDER-SWAPPED (see Section 2). Methods: CSCS, GhostFace v1/v2/v3, InStyleSwapper256-A/B/C, Inswapper128, SimSwap512 | same as deeplive bucket | bad-data policy CSV gates this | `realpool_<identity>` (sample-id parsing) | **disabled** in P10 | n/a |
| **visomaster_hints** | both (paired) | Subset of legacy visomaster relabeled as "weak-signal hints" (480 samples, 40 per non-CSCS method per MODERATE/STRONG tier) | same | bad-data policy CSV | `realpool_<identity>` | **disabled** in P10 | (would be visomaster_hints_fake=2.5) |
| **visomaster_hints_teams** | both (paired) | 202 weak-signal Teams-played samples (the bad data after passing through Teams). | `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2` | bad-data policy CSV | `realpool_<original_video_name parsed>` | **disabled** in P10 | n/a |
| **visomaster_enhanced** | both (paired) | Post-hoc enhanced fakes (GFPGAN/CodeFormer/GPEN-BFR/RestoreFormer applied to legacy VisoMaster outputs) | `visomaster-enhanced-face-cropped` | bucket-walked | `realpool_<identity>` (shared) | **disabled** in P10 | (would be visomaster_enhanced_fake=2.0/3.0) |
| **visomaster_teams_enhanced** | both (paired) | Resolver-driven merged source mixing original-Teams fake with enhanced-clean fake at p_original | `enhanced-visomaster-cropped` + Teams bucket | `gs://training-job-outputs/cache/visomaster/enhanced_visomaster_resolver_2026-04-06.json` | `realpool_<identity>` | **disabled** in P10 | n/a |
| **proper_data** | both (paired) | "Proper" (post-fix, face_parser_enabled=False) VisoMaster captures. Both clean and Teams transports, both with-enhancement and none. 1826 captures, 4 lanes | `hdtf_visomaster_cropped_frames`, `hdtf_visomaster_cropped_frames_teams` (and matching quickclips buckets, in inventory) | `arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml` (180k lines) and `arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json` (329k lines) | `realpool_splitgroup_<split_group_id>` — not raw identity, but inventory's WT-F split-group ID | yes (4 lanes: clean, teams, enhanced_clean, enhanced_teams) | proper_visomaster_clean_fake=1.0, proper_visomaster_teams_fake=2.5, proper_visomaster_enhanced_clean_fake=2.0, proper_visomaster_enhanced_teams_fake=1.0 |
| **teams** (Teams passthrough) | both (paired) | Teams-played pairs from the cropped-teams-v2 bucket. The "real" branch is captured via OBS virtual cam→Teams. The "fake" branch is the same flow but with a deepfake source. | `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2` | walked at startup, cached. Bad-data CSV is applied (`apply_bad_data_policy: true`). | parsed from `original_video_name` and converted to `realpool_<id>` for split coherence | yes | (each Teams sample maps to deeplive_teams_real or deeplive_teams_fake at family-key time, weight 4.0/5.0) |
| **external_training_reals** | real only (unpaired) | VCD webcam reals from effort-collected-data, prefix `real/VCD`, identity is md5 from filename | `effort-collected-data` | bucket-walked at startup with regex `real__VCD__(?P<md5>[a-f0-9]{32})_` | independent identity-split: 40% of identities go into train (`identity_split_seed=737`); rest go into OOD pool | yes | external_real=2.5 |
| **external_youtube_avspeech** | real only (validation/OOD) | YouTube avspeech-derived reals used for lighting/spatial stress OOD eval **only** | `effort-collected-data`, prefix `real/external_youtube_avspeech` | walked at startup | n/a — OOD only | yes (OOD; 200 videos × 6 stress configs) | n/a (eval) |
| **real-teams-dor-roee** | real only (lockbox readout) | Hand-collected Roee/Dor session captures used for lockbox readout. NEVER mixed into train. | `real-teams-dor-roee` (`session_20260424_110139/uniform30`) | folder-walked, max 10 videos | n/a — `readout_only_external_real_sources` | yes (readout-only) | n/a |

Source-level notes:
- The "deeplive" loader actually serves both DeepLive **and** VisoMaster from the same bucket — they share `samples/` prefix; the loader filters by sample-id pattern. They share the `realpool_` identity prefix to enforce identity-split coherence.
- `combined_paired.deeplive.use_landmarks: false` — landmarks are disabled in P10. This matters for augmentation choices (e.g. landmark-aware face-region augmentations are skipped).
- The `visomaster_enhanced.exclude_tiers: ["ARTIFACT"]` filter would exclude clearly-broken enhanced fakes if the source was on. It's off in P10.
- `train_split: 0.85`, `val_split: 0.10`, `test_split: 0.05`. Identity-stratified split. `holdout.mode: "identity"` (not method-holdout).

---

## 2. The VisoMaster bad-data policy

**Files (canonical, tracked):**
- `policy/visomaster_bad_data/VISOMASTER_BAD_DATA_POLICY_MANIFEST_2026-04-17.csv` (one row per `sample_id`, columns include `policy_action` ∈ {keep, ignore, delete}, `policy_label` ∈ {visomaster hints, visomaster hints (teams), ""}, `policy_lane`, `decision_reason`, `tier`, `identity_delta`, etc.)
- `policy/visomaster_bad_data/VISOMASTER_BAD_DATA_POLICY_SUMMARY_2026-04-17.json`
- `policy/visomaster_bad_data/VISOMASTER_BAD_DATA_POLICY_REPORT_2026-04-17.md`
- `policy/visomaster_bad_data/VISOMASTER_BAD_DATA_UPLOAD_AUDIT.md`

**Date / authorship:** Created Apr 17 2026. Policy framework was committed in `b208d17 Freeze WT-A policy truth and lane semantics` (Apr 17 2026, author roeehub). Runtime that consumes the policy was committed in `f9303eb Land WT-B runtime and launcher smoke readiness`. The upload audit was written by "Codex" (the AI agent) on the same day per the audit doc.

**What was excluded and why:**

The historical "visomaster" branch in three GCS buckets (`live-deepfake-methods-real-and-fake-frames`, `…-videos`, `…-cropped`) was generated with the bug `face_parser_enabled=True` in the VisoMaster pipeline, which multiplied a face-parser mask into the swap mask and produced **under-swapped** outputs. The result: faces that were partially swapped, sometimes barely swapped, but tagged as `GhostFace-v3` / `Inswapper128` / etc.

| Action | Count | Rationale |
|---|---:|---|
| `ignore` | **4904** of 5589 unique cropped sample_ids | not method-faithful, exclude from training |
| `keep` (visomaster hints) | 480 | weak-signal baseline, 40 per (non-CSCS method × MODERATE/STRONG tier), seed `20260417` |
| `keep` (visomaster hints (teams)) | 202 | the Teams-played slice, retained as "weak signal + Teams codec" |
| `delete` | 3 | three explicitly named bad datapoints (e.g. `visomaster_GhostFace-v2_04004`) |

**Operational rules (verbatim from the report):**
1. All `CSCS` excluded entirely (the worst-affected method in the failure mode).
2. `MINIMAL` tier excluded entirely (under-swapped → least information).
3. Of `MODERATE` and `STRONG` non-CSCS samples, take a deterministic random 40 per (method, tier) cell with seed `20260417`.
4. The `pair_complete` Teams subset (237 samples) is kept (minus 1 explicitly-named delete = 202) as `visomaster hints (teams)`.
5. Retained samples must NOT be trained under their nominal generator labels — they get the new weak labels `visomaster hints` / `visomaster hints (teams)`.

**Suspicious observation:** in the current P10 yaml, `combined_paired.visomaster.enabled: false` AND `combined_paired.visomaster_hints.enabled: false`. So none of the bad-data, kept or otherwise, makes it into P10 training. The bad-data policy is consequently dormant for P10. However:
- `combined_paired.teams.apply_bad_data_policy: true` is on. The Teams source therefore filters out the 237 `pair_complete` bad VisoMaster sample_ids that landed in Teams (via `_select_clean_teams_samples`). Without this filter, ~237 / 1348 (≈18%) of `pair_complete` Teams samples would be bad VisoMaster outputs played through Teams.

---

## 3. The proper_data inventory

**Files:**
- `arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml` (180792 lines, ~512 KB)
- `arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json` (329049 lines, ~10+ MB)

**Wave/version:** `proper_visomaster_wave_2026_04_19_provisional` — explicitly tagged "provisional" because the Teams propagation was still in flight at the time of build.

**What "proper" means:**
- "Proper" = the **corrected** VisoMaster pipeline — the one that does NOT have the `face_parser_enabled=True` bug. Per the bad-data report: "More actually correct VisoMaster data is planned to arrive soon. When that happens, it should be treated as a clean replacement lane rather than merged back into this weak-signal stopgap bucket policy."
- The inventory packages 1826 base captures (mostly HDTF + quickclips), each with up to 4 transport/enhancement variants per generator method (clean+none, clean+enhanced, teams+none, teams+enhanced) plus a paired real (clean and teams).
- 9 generator methods covered: CSCS, GhostFace-v1/v2/v3, InStyleSwapper256-A/B/C, Inswapper128, SimSwap512.
- For enhanced lanes: 7 face-restorers — codeformer, gfpgan_v1.4, gpen_256/512/1024/2048, restoreformer.

**Counts (from the manifest's `summary` block):**

| Lane | Count |
|---|---:|
| proper_real_clean | 1826 |
| proper_real_teams | 1826 |
| proper_visomaster_clean (fake) | 342 (CSCS=12, GhostFace-v1/2/3, InStyleSwapper256-A/B/C, Inswapper128, SimSwap512) |
| proper_visomaster_teams (fake) | 342 |
| proper_visomaster_enhanced_clean (fake) | 1484 (each method × 7 enhancers) |
| proper_visomaster_enhanced_teams (fake) | 1484 |
| **Total videos** | 7304 |
| **Captures total** | 1826 |

Splits: `dev=5776`, `lockbox=1528` (`lockbox_ratio: 0.2`, `split_seed: 737`). Quality bands: `high=4376, medium=2928`. Face-scale bands: `big_face=4376, standard=2928` (these are flagged "provisional dataset-level defaults" in the inventory header — they are not actual per-capture image-feature measurements).

**What it replaced:**
- The legacy `visomaster` source (the under-swapped one).
- Inventory was built from `arena/build_visomaster_proper_data_artifacts.py`. Census reports under `arena/reports/visomaster_proper_*_bucket_census_2026-04-19.json` and `…/hdtf_visomaster_clean_vs_teams_join_2026-04-19.json` document the underlying clean↔teams alignment.
- Crucially, the inventory keeps **only exact 16/16 clean-vs-teams overlapping rows** (provisional snapshot constraint).

**Identity model:** `realpool_splitgroup_<split_group_id>` where `split_group_id` looks like `RD_Radio14__RD_Radio14_000`. The `proper_data.py` loader explicitly comments: "WT-F split hygiene is defined at the split_group level, not raw identity." This means split-group is a video-segment-level identity (one base capture and its session), not a person-level identity.

Risk: the split-group ID derives from the inventory's own `capture_session_id`. If the same person (e.g. `RD_Radio14`) appears in multiple capture sessions (`RD_Radio14_000`, `RD_Radio14_001`), they get **different** split-group IDs, and so the same person can land in both train and lockbox under different sessions. The inventory-level `identity_id` IS the same (`RD_Radio14`), but it is **not** what is used to gate the split. Confirmation: `proper_data.py` line 329, `identity = f"realpool_splitgroup_{sample.split_group_id}"`.

---

## 4. The visomaster_hints situation

**Status:** disabled in P10 (`combined_paired.visomaster_hints.enabled: false` in `R13_P10_SYM_baseline.yaml` and all other P10 yamls). Same for `visomaster_hints_teams`.

**User note (verbatim):** "hints are bad data, for example."

**Why disabled:**
1. The hints are **deliberately weak signal**. The bad-data report calls them "auxiliary weak-signal data only, not method-faithful supervision." Per the WT-A policy truth freeze (`b208d17`, Apr 17), retained hint rows are only valid under the new lane labels `visomaster hints` / `visomaster hints (teams)` — they do not represent what `GhostFace-v3` or `Inswapper128` actually look like, because they were generated with the face-parser bug.
2. **Replaced by `proper_data`.** The proper-data inventory (`proper_visomaster_wave_2026_04_19_provisional`) provides the same generator coverage (9 methods, with and without enhancement, in both clean and Teams transports) using the **fixed** VisoMaster pipeline. Once proper_data was integrated (commit `77facfc Add proper-data runtime and relaunch review packet`), the weak-signal stopgap was no longer needed for fresh runs.
3. **Per-experiment ablation evidence.** The R13 RLP1_01/02/03 packet was specifically designed to A/B/C the choice: WTB1 (no hints), WTB2 (hints only), WTB3 (hints + Teams hints). RLP1_01 has both hint sources `enabled: false`. RLP1_02 has `visomaster_hints: true, visomaster_hints_teams: false`. RLP1_03 has both `true`. These yamls were committed in `f9303eb Land WT-B runtime and launcher smoke readiness`. The "no_hints" lane has been the production path forward; hints-on lanes were the falsification arms.

**Underlying problem (concretely):**
- A "hint" sample's fake side is a **partially-swapped face** (mask was multiplied by face-parser → only certain pixels got swapped). A model trained to call this "fake" learns "if this image has a strange composite seam over only part of the face, predict fake." That is not a real-world deepfake signature — it is a generation pipeline bug — and a model that picks this up will overfit to **the bug**, not to the swap method.
- The `visomaster_hints_teams` data has a second confound: the bad partial swap was then run through Teams, so the codec fingerprint and the partial-swap fingerprint are co-encoded. A model that reads "Teams-codec + partial composite seam" and calls it fake will not generalize to a clean Teams play of a properly-swapped face.

---

## 5. The Teams data picture

**Bucket:** `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`
**Layout:** `samples/{sample_id}/manifest.json + frames/{real,fake}/frame_NNNN.jpg`
**Discovery:** `discover_teams_passthrough_samples()` lists manifests, requires `pair_complete: true`, supports a discovery cache.

**What's in it (training-time):**
The Teams source `combined_paired.teams` is **paired**: each sample has a real branch and a fake branch captured through Teams. The `combined_paired.teams.apply_bad_data_policy: true` flag means the bad-data filter is applied.

**Identities, cameras, capture conditions** — best surfaced from the Teams target-domain manifest (`arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json`, used for evaluation, not training, but the underlying data is the same population):

| method (slice) | count |
|---|---:|
| `teams_capture_cam_test_s32/s33/s35/s38/s46` | 139 / 191 / 244 / 51 / 89 |
| `teams_capture_dor_shkedi_s16` | 57 |
| `teams_capture_noyn_sharker_s23` | 204 |
| `teams_capture_pc_generator_s3/s4/s9/s15` | 99 / 30 / 46 / 62 |
| `teams_capture_test_cam_s53/s73/s76` | 90 / 55 / 75 |
| `teams_flat_xiang_xiang2_feng` | 135 |
| `deeplive_enhanced` | 545 |
| `visomaster_enhanced_macro` | 550 |
| `teams_real` | 4614 |
| **fake total** | 2662 |
| **real total** | 4614 |

This shows: a small handful of identities (`dor_shkedi`, `noyn_sharker`, `pc_generator`, `xiang_xiang2_feng`, plus `cam_test`/`test_cam` machine pseudo-identities). They are **not** the user's own face (Roee) or his colleague Dor (the `dor_shkedi` slice is a different "Dor"). The only data with the user's actual face is the lockbox-readout-only `real-teams-dor-roee` source.

**Train/dev/lockbox split:**
- The training Teams samples go through the **same** `split_samples_by_identity` as everything else, with `identity_split_mode: hash_stable, split_seed: 737`. The Teams identity is `realpool_<original_video_name>` — i.e. shared `realpool` namespace with DeepLive/VisoMaster.
- **The lockbox suite is separate from the training split.** Lockbox is defined by `target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml` and uses `external_real_manifest_split: "lockbox"` from `teams_target_domain_manifest_2026-04-23_with_dor.json` — so lockbox is the 20% partition of the **manifest's** split (also seed 737, lockbox_ratio 0.2). This is a **different** slicing than the trainer's 0.85/0.10/0.05.

**Is there a Teams-specific identity holdout?** Partially:
- The `arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json` has its own `dev/lockbox` split with seed 737, lockbox_ratio 0.2. It uses `identity_key` (the human-readable identity prefix like `Cam_Test__s32`) for stratification.
- However, the **trainer** uses a different identity prefix (`realpool_<video_id>` derived from `original_video_name`). A given person could in principle land in the training set under one slicing and in the lockbox under the other slicing. The manifest's lockbox is the one that matters for promotion, but it's not the same entity space the trainer's identity-leakage check operates on. **Not verified that the two splits are coherent.**

**Critical observation:** the Teams data has multi-camera structure. The `teams_capture_cam_test_*` and `teams_capture_test_cam_*` "identities" are camera/session pseudo-identities, not people. So "identity holdout" on these specific slices is more accurately "session holdout," not "person holdout."

---

## 6. The DeepLive picture

The DeepLive source provides paired real/fake frames captured through the `Deep-Live-Cam` pipeline. Strategies live as `samples/{strategy_prefix}_*/manifest.json` in the bucket.

**Strategies enabled in P10 (`include_strategies`):**
- `edge_cases`
- `minimal_processing`
- `quality_enhancement`
- `edge_cases_enhanced`
- `minimal_processing_enhanced`

**What each operationally means** (from inspection of `dataset/deeplive_dataset.py`'s `resolve_effective_strategy` and the manifest tagging convention):

| Strategy | Operational meaning |
|---|---|
| `edge_cases` | Difficult capture conditions — extreme angles, partial occlusion, harsh lighting, rapid motion |
| `minimal_processing` | "Cleanest" raw capture — minimal filters and post-processing applied to the live capture |
| `quality_enhancement` | Live capture run through the Deep-Live-Cam quality enhancement (face-restoration / sharpening) at generation time |
| `edge_cases_enhanced` | An `edge_cases` capture that was **subsequently** post-hoc enhanced (face restoration applied AFTER capture) |
| `minimal_processing_enhanced` | A `minimal_processing` capture that was **subsequently** post-hoc enhanced |

The `_enhanced` suffix on `edge_cases_enhanced` / `minimal_processing_enhanced` is detected by the regex `^(?P<base>.+)_enhanced_[0-9]+$` on `sample_id`; this is treated as a separate "effective strategy" by the runtime.

**Quality direction:** the enhancement is **quality-improving** (cleaning) — face restorers (`gfpgan`, `codeformer`, `gpen`, `restoreformer`, etc.) explicitly try to make the image look closer to a clean photographic capture. So `edge_cases_enhanced` should look **cleaner / less artifact-laden** than `edge_cases`. But — because the augmentation router has a special pipeline `deeplive_enhanced_fake` that is **more aggressive in its degradation** than `deeplive_non_enhanced_fake` (see `pipelines.py` lines 1227–1258, with `quality_p + 0.22`, JPEG floor `-12`, downscale floor `-0.10`, etc.) — at training-time the model sees enhanced-fake samples that have been re-degraded **harder** than non-enhanced ones. The augmentation deliberately mostly-undoes the enhancer's quality boost.

This is intentional per the code comment: "Enhancement smooths out GAN artifacts and increases perceived quality, making fakes look closer to real images. Apply aggressive degradation … so the model learns to detect underlying fake structure despite enhancement." Whether that intent is achieved is a separate question.

**Operational bottom line:** the `_enhanced` strategies are quality-improving at generation time but quality-degraded again at augmentation time. The end image distribution as seen by the model is not necessarily cleaner than non-enhanced. Because the enhanced-fake aug is heavier than non-enhanced-fake aug, there is again a **per-class augmentation bias** — if the model detects "this image was downscaled hard and JPEG-compressed hard," it can predict fake without examining face structure.

---

## 7. Anchor pools

Anchor pools are the small, hand-curated pools of real frames used during stress-testing and the per-pool diagnostic. These are **not** the training data — they are the diagnostic apparatus used to answer "does our model false-flag Roee on his Mac webcam?"

**File:** `analysis/configs/dor_roee_combined_2026-04-24.yaml`
**Bucket:** `gs://real-teams-dor-roee/session_20260424_combined_tags_121458_121007/`
**Manifest (local):** `/tmp/dor_roee_combined_2026-04-24/combined_frame_tags.json`

| Pool name | Identity | Camera | Lighting | Virtual BG | Failing/clean |
|---|---|---|---|---|---|
| `dor_laptop_white_clean` | Dor | laptop (built-in) | white-ish | none | clean |
| `dor_laptop_yellow_clean` | Dor | laptop | yellow-ish | none | clean (negative control vs white_clean) |
| `dor_webcam_novb_failing` | Dor | external webcam | normal | none | failing (model false-positives) |
| `dor_webcam_vb_failing` | Dor | external webcam | normal | virtual bg ON | failing |
| `roee_mac_failing` | Roee | Mac laptop | normal | virtual bg ON | failing |
| `roee_windows_clean` | Roee | Windows laptop | normal | none | clean |

Pair semantics defined in the same file:
- A: dor_webcam_novb_failing vs dor_laptop_white_clean — **same person, same lighting, no VBG; only the camera differs.** Cleanest camera-isolation.
- B: dor_webcam_vb_failing vs dor_laptop_white_clean — same person, same white lighting; camera + VBG differ.
- C: roee_mac_failing vs roee_windows_clean — different person, different camera; tests cross-identity replication of the camera-flag pattern.
- D: dor_laptop_yellow_clean vs dor_laptop_white_clean — **negative control** (same camera, only lighting differs; both clean).
- E: dor_webcam_novb_failing vs dor_webcam_vb_failing — negative control (same camera, only VBG differs).

**These pools are the operational evidence cited in `MEMORY.md`'s `project_signature_shortcut_finding.md`** — the same person `dor_shkedi` (training-time Teams identity) shows different model output than `real_dor` (anchor-pool real Dor on Mac webcam). The shortcut is a camera/processing fingerprint, not a face fingerprint.

**Where the anchor pool images live:** in the `real-teams-dor-roee` GCS bucket. They are reachable by training only as `readout_only_external_real_sources` (see Section 1's row for that bucket): max 10 videos, used only for OOD-readout, NEVER mixed into training samples. The `readout_only_*` knob is essential — these images contain the user's actual face, and contaminating training with them would invalidate the diagnostic.

---

## 8. External reals

| Source | Bucket | Prefix | Identity model | Sample cap | Use |
|---|---|---|---|---|---|
| `external_training_reals` (VCD) | `effort-collected-data` | `real/VCD` | regex `real__VCD__(?P<md5>[a-f0-9]{32})_` → md5 acts as identity | `max_frames_per_identity: 15`, `max_total_samples: 1200`, `identity_train_fraction: 0.40` | Training-time real-only unpaired |
| `external_youtube_avspeech` | `effort-collected-data` | `real/external_youtube_avspeech` | grouped by folder (each folder = one video) | `max_videos: 200` per stress slice (general / backlight_dim / warm_harsh / crop_shift / scale / rotation = 6 × 200) | OOD lighting/spatial stress eval ONLY — never in train |
| `real-teams-dor-roee` | `real-teams-dor-roee` | `session_20260424_110139/uniform30` | by_folder | `max_videos: 10` | Lockbox readout ONLY (the user's own face) |

**Provenance:**
- VCD = "Video Conferencing Dataset" — webcam captures. The md5-keyed identity comes from the file name `real__VCD__<md5>_<...>`.
- `external_youtube_avspeech` — derived from av-speech YouTube clips; manifest in `build_quantization_val_manifest.py` (`EXTERNAL_REAL_BUCKET="effort-collected-data"`, `EXTERNAL_REAL_PREFIX="real/external_youtube_avspeech"`).
- `real-teams-dor-roee` is a one-session capture by the user.

**Identity-overlap with anchors:**
- `external_training_reals`: **md5 hashes**. No human-readable identity. Likely no overlap with Dor or Roee anchor pools (those are different captures), but **not auditable from the bucket layout alone**. There is no metadata cross-reference that would let the auditor confirm or refute that, e.g., a VCD md5 happens to be Dor sitting at a different camera.
- `external_youtube_avspeech`: random YouTube speakers; functionally negligible overlap with anchors.
- `real-teams-dor-roee`: by definition contains Dor and Roee. This is **why** it is firewalled to `readout_only_external_real_sources`.

**Sample counts (P10 caps):**
- VCD training: cap=1200 frames, ~80 identities (`identity_train_fraction=0.40` of total ~200 identities, `max_frames_per_identity=15`). Family weight 2.5.
- avspeech: 200 videos × 6 stress slices = 1200 video-equivalents, eval-only.
- dor-roee: 10 videos, eval-only.

---

## 9. Family weights — the asymmetric sampling weights

From P10 baseline `combined_paired.sampling.family_weights`:

| family_key | weight | what it does | comment |
|---|---:|---|---|
| `df40_fake` | **0.15** | Probability mass on df40 fake samples in identity-resampled epochs | Lowest; df40 deliberately suppressed |
| `df40_real` | **0.4** | df40 reals | Suppressed. df40 is treated as a "diversity" pool, not a primary signal |
| `deeplive_non_enhanced_fake` | **2.5** | edge_cases, minimal_processing | Mid weight |
| `deeplive_enhanced_fake` | **3.0** | edge_cases_enhanced, minimal_processing_enhanced, quality_enhancement | Higher; enhanced fakes are the harder positives |
| `deeplive_teams_fake` | **5.0** | Teams-played fake samples | **Highest**, 33x df40_fake |
| `deeplive_teams_real` | **4.0** | Teams-played real samples | Highest real |
| `realpool_real` | **2.5** | DeepLive/VisoMaster real frames | Mid |
| `external_real` | **2.5** | VCD reals | Mid |
| `proper_visomaster_clean_fake` | **1.0** | clean-transport proper-data fakes | Low |
| `proper_visomaster_teams_fake` | **2.5** | Teams-transport proper-data fakes | Mid |
| `proper_visomaster_enhanced_clean_fake` | **2.0** | Enhanced clean | Mid-low |
| `proper_visomaster_enhanced_teams_fake` | **1.0** | Enhanced Teams | Low |

**What these do at training time:** The `CombinedPairedIterableDataset` (combined_paired.py:2519+) computes a per-sample family key at startup, looks up the weight from `identity_family_weights`, and uses these weights to draw sample IDs **stratified by identity**, with each identity's expected frequency proportional to its family weight. The strategy is `identity_resample_weighted`.

Per the code (combined_paired.py:2693-2694): `weight = float(self.config.identity_family_weights.get(family_key) or 1.0)` — unspecified families default to 1.0.

**Asymmetry callouts:**
- `deeplive_teams_fake` at 5.0 is **33×** `df40_fake` at 0.15. So in expectation, every time the dataloader emits one df40_fake sample, it emits ~33 deeplive_teams_fake samples. This is by design — the user wants the model to specialize on Teams — but it means the model's loss signal on the Teams-codec data dominates, and any shortcut available within Teams (camera, codec, identity) is the cheapest gradient direction.
- `deeplive_teams_real` at 4.0 vs `realpool_real` at 2.5: 1.6× heavier. This is consistent with the FPR-on-Teams worry, but combined with the per-class augmentation routing (Section 11.A) it amplifies per-class statistical asymmetry within the Teams group.
- `proper_visomaster_enhanced_teams_fake` at 1.0 — the smallest fake weight. Surprising given that **enhanced-Teams is the closest to a real Teams call carrying a deepfake**. It may be deliberately down-weighted as the most-target-looking and so the most prone to overfitting; **flag for owner check**.

---

## 10. Possible label leaks — CRITICAL

This is the section the user explicitly asked to surface.

### A. Family-aware augmentation router as a per-class quality leak (HIGH RISK)

**Found in:** `data/augmentations/pipelines.py:1148` (`_build_family_quality_pipeline`) and registered in `QualityTargetedFamilyRouter` at line 1585.

The router dispatches each sample to a different albumentations `Compose` based on the **inferred family_key** (which is computed from `(label, source, method)`). Concretely, the family_key is computed at collation time from label and source, and the **augmentation that gets applied is conditional on the label**. This is a hard-coded label leak via image statistics.

Examples of the per-class asymmetry (from `_build_family_quality_pipeline`):

| Family | quality_p adjustment | JPEG floor | Downscale min | Sharpen alpha | Notes |
|---|---|---|---|---|---|
| df40_fake | base `quality_p` | base `jpeg_lower` | base | balanced (0.20–0.45 for moderate) | + optional `fake_extra_degrade_p` heavy-downscale OneOf |
| deeplive_non_enhanced_fake | `quality_p + 0.08`, capped 0.9 | base | base | balanced 0.22 | |
| deeplive_enhanced_fake | `quality_p + 0.22`, capped 0.95 | `jpeg_lower - 10` (chained second compression at `-14`) | `downscale_min - 0.10` | very low (0.08–0.22) | "WMA-style degradation emphasis" |
| visomaster_fake | `quality_p + 0.12`, capped 0.92 | `jpeg_lower - 8` | `downscale_min - 0.08` | low (0.10–0.30) | |
| visomaster_enhanced_fake | `quality_p + 0.24`, capped 0.96 | `jpeg_lower - 12` (chained at `-16`) | `downscale_min - 0.10` | very low (0.06–0.18) | |
| df40_real | (no `OneOf` quality block; replaces with `OneOf([compression>=58, mild noise, dedicated-noise])` at p≥0.20) | floor pinned at **max(58, jpeg_lower)** | n/a (no downscale) | real (0.18–0.45 for moderate); `real_sharpen_p=0.50` default | "ceiling" 95 |
| realpool_real / external_real | OneOf at `quality_p + 0.04`, capped 0.85 | floor at max(52, jpeg_lower) | floor at max(0.62, downscale_min) | real (0.15–0.38 for VCD-targeted) at `max(0.18, real_sharpen_p * 0.4)` | |

Key quantitative asymmetries, normalized to a `vcd_targeted` preset where `quality_p=0.60`, `jpeg_lower=40`, `downscale_min=0.50`:

| Knob | fake side (avg) | real side (avg) | direction |
|---|---|---|---|
| Probability of any quality-degradation OneOf firing | 0.6 to 0.92+ | 0.20 to 0.85 | fakes ALWAYS more likely to be degraded |
| JPEG floor when degraded | 24-40 | 52-58 (min 28 only with chained second pass) | fakes get harsher compression |
| Downscale floor | 0.35 | 0.62 (real route lacks Downscale entirely on df40) | fakes get harsher downscale |
| Sharpen probability | 0.06-0.26 | 0.18-0.50 | reals are sharpened more (higher real_sharpen_p) |

This means: at training time, the average sample-level image statistics (Laplacian variance, JPEG quantization signature, sharpness, noise floor) **carry direct label information**. A linear classifier trained only on those statistics would already get well above chance.

**This is the explicit hypothesis tested by P10's `routing_mode: "symmetric"` mode** (per the P10 commit message: "the family-aware augmentation router IS the camera/processing-signature shortcut source"). The fact that P10 was created suggests the team strongly suspects this.

**Anchor evidence cited in MEMORY.md:** the same person `dor_shkedi` vs `real_dor` flips model output. If the model learned "low-quality codec → fake," then `real_dor` taken on a 4K Mac webcam ≠ `dor_shkedi` taken on a noisy webcam captured through Teams, even though both are real. That's exactly the failure mode this aug router would induce.

### B. Source-label correlation (HIGH RISK)

| Source | Real:Fake split |
|---|---|
| df40 | each `df40-pair` has both real and fake (paired); but the "real" frame is from FF++ donor identities — a single homogeneous pool |
| deeplive (non-Teams) | paired |
| deeplive_teams | paired |
| visomaster_* | paired |
| proper_visomaster_* | paired |
| **external_training_reals (VCD)** | **REAL ONLY** |
| **external_youtube_avspeech** | **REAL ONLY** |
| **real-teams-dor-roee** | **REAL ONLY** |

So the unpaired sources are 100% real. A model that learned "this image came from `effort-collected-data/real/VCD/...` → predict real" without any face content would do well on the trainer's split.

The `UnifiedUnpairedRealSample` dataclass passes `source="external"` (combined_paired.py:135) and the augmentation router routes them to `_build_family_quality_pipeline("external_real", ...)` — same JPEG floor, downscale floor, real-sharpen as the realpool_real route. In other words: **VCD reals** are augmented with the **same `external_real` pipeline** as DeepLive reals, but **never** with the `deeplive_non_enhanced_fake` pipeline. So a VCD image will never see the harsher fake-side aug.

**This means:** at the level of *augmented image statistics*, VCD images are statistically distinguishable from any fake image after augmentation — not because the original VCD images are special, but because they get put through the real-side augmentation chain only. **At training time, "VCD bucket → real" is approximately deterministic.**

### C. File-path conventions

Examples of folder-path-encoded labels (multiple sources):
- DeepLive: `samples/{sample_id}/frames/real/frame_NNNN.png` and `samples/{sample_id}/frames/fake/frame_NNNN.png`
- Teams (cropped-teams-v2): `samples/{sample_id}/frames/real/...` and `samples/{sample_id}/frames/fake/...`
- proper_data inventory: `gs://hdtf_visomaster_cropped_frames/samples/HDTF...`/frames/real/... and `.../frames/fake/...`
- effort-collected-data VCD: `real/VCD/...` (the word `real` is in the path)
- effort-collected-data avspeech: `real/external_youtube_avspeech/...` (path contains `real`)

The image bytes, once decoded, are 224×224 RGB tensors — **the path is not part of the input to the network**. So no direct path leak through the model. However: the path **is** used by the data loader to assign label, and that assignment goes into the augmentation router, which then conditionally augments. So there's no leak from path → image, but there IS a leak from path → label → aug-pipeline-choice → image-statistics. (See Section 10.A.)

### D. EXIF / codec metadata

The frame loaders use `cv2.imdecode(...)` to convert `.jpg/.png` bytes to numpy arrays (`combined_paired.py:822`, `proper_data.py:660`, etc.). `cv2.imdecode` strips EXIF — only the pixel buffer is returned. So **EXIF metadata is not leaked** to the model.

However:
- **JPEG quantization tables are inherent to the pixel statistics** — the recompression artifact pattern is recoverable from pixels alone. So even after EXIF strip, codec lineage is partially recoverable. The training data has multiple JPEG lineages (Teams JPEG quality varies; PNG-vs-JPEG-vs-recompressed-JPEG distribution is non-uniform across sources).
- The `webcam_codec_step = VideoCodecSimulation(...)` is applied to ALL families (it's outside the family-conditional branches), so the JPEG-recompression footprint is partially randomized at training time. But — the **input** lineage to that step is still source-specific (DeepLive PNGs, Teams JPGs, VCD JPGs). The simulator reduces but does not erase the underlying lineage signal.

### E. Cross-source camera asymmetry

Confirmed asymmetry:
- All `real` samples in training come from one of: VCD webcams, DeepLive captures, VisoMaster captures, proper_data captures (HDTF studio + quickclips), Teams-played captures of the above. Each is a small set of cameras / capture environments.
- All `fake` samples come from those same captures with face-swap applied. So real and fake samples in any given source DO share the same upstream capture device. **Within a paired source, camera-level confound is eliminated.**
- BUT: across the joined (df40 + deeplive + visomaster + proper + Teams) pool, **camera distribution by label is non-uniform**:
  - VCD camera distribution → real-only (Section 10.B). Not present on fake side.
  - YouTube avspeech → real-only.
  - Teams (combined fake+real) → both.
  - df40 reals (FF++ donor) → academic-clean. Reals from df40 do not have Teams codec.

In other words: **the only camera that produces fake samples in the user's high-weight Teams family is the Teams playthrough.** Conversely, the only fake samples not from a Teams playthrough are df40, DeepLive (non-Teams), and proper-clean. So a model that learned "this image came from a non-Teams source AND looks like a webcam capture → fake (because it's df40/deeplive/visomaster)" or vice versa could exploit the asymmetry.

---

## 11. Identity overlap across train/dev/lockbox

**Mechanism (combined_paired.py:1186-1340):**

1. Each sample is given an `identity` string at construction time:
   - df40 → `df40_<sample.target_identity>`
   - DeepLive → `realpool_<original_video_name without cropped_ prefix and .mp4 suffix>`
   - VisoMaster → `realpool_<sample.identity>` (where `sample.identity` is parsed from manifest's `original_video_name` if present, else from sample_id)
   - VisoMaster Enhanced / Teams Enhanced / Res Variant → `realpool_<sample.identity>` (same as above)
   - VisoMaster Hints → `realpool_<sample.identity>`
   - VisoMaster Hints Teams → `realpool_<original_video_name parsed>`
   - Teams (the regular passthrough source — file inspected too long, I checked combined_paired.py around the teams section):  also `realpool_<...>` derived from `original_video_name`
   - Proper Data → `realpool_splitgroup_<sample.split_group_id>`
   - external_training_reals (VCD) → `external_vcd_<md5>`

2. `split_samples_by_identity` (combined_paired.py:1186) groups by these identity strings, then splits identity-by-identity using `hash_stable` mode (sha256(`{seed}:{identity}`)) — same identity always lands in the same split across runs of the same seed. There is a **post-condition assert** (line 1334): if any identity overlaps between train, val, test, the function raises.

3. The identity-leak check (line 1334-1338) is **actually run** and would catch in-bucket leaks.

**Where identity-coordination is solid:**
- DeepLive, VisoMaster, VisoMaster Enhanced, VisoMaster Hints — all four share the `realpool_<id>` namespace AND derive `<id>` from the same upstream `original_video_name`. So the same person never crosses splits across these four pools. (From combined_paired.py:233-237 explicit comment: "Use 'realpool_' prefix (NOT 'deeplive_') so that DeepLive and VisoMaster samples from the same source video are grouped together during identity-stratified splitting. This prevents identity leakage where the same real person could appear in train (via DeepLive) and val (via VisoMaster).")

**Where identity-coordination is questionable:**
- **Proper data uses `realpool_splitgroup_<split_group_id>`, not `realpool_<identity>`.** The `split_group_id` is `<identity_id>__<capture_session_id>`. So if `RD_Radio14` has two sessions `RD_Radio14_000` and `RD_Radio14_001`, they get DIFFERENT split-group IDs and could land in DIFFERENT splits — meaning the **same person** can appear in train (one session) and lockbox (another session). The `proper_data.py` author explicitly comments that "WT-F split hygiene is defined at the split_group level, not raw identity," but the consequence — same-person leakage across splits — is real.
- **Teams data identity-key in the trainer** uses `realpool_<original_video_name parsed>`. **Teams data identity-key in the target-domain manifest** (used by lockbox) is `identity_key` like `Cam_Test__s32`. These are different namespaces. If the trainer puts a sample with `realpool_dor_shkedi_blah` into train, and the lockbox has `identity_key=dor_shkedi_s16`, the trainer's leak-check will not see it (different namespace). The promotion contract scorer in `arena/score_teams_promotion_contract.py` uses the manifest's identity_key, so the lockbox slice itself is internally clean. But cross-namespace leak between trainer-train and lockbox is **not gated**.
- **External VCD identities** are md5 hashes. Without a names↔md5 cross-reference, the auditor cannot rule out that one of these md5s corresponds to e.g. Dor in some other capture context. For the audit, treat this as "uncertain — not auditable from the bucket layout."
- **The `realpool_` prefix is a flat namespace.** If the same person appears in DeepLive captures under one `original_video_name` and in VisoMaster captures under a different `original_video_name`, the trainer treats them as different identities. This is a soft leak: same person, two videos, two identities, two splits possible. The codebase comment claims this is mitigated because DeepLive and VisoMaster share the same source-video set, but in practice the mapping depends on whether the upstream `original_video_name` is consistent — a fact the auditor cannot verify from code alone.

**Net summary of identity-leak risk:**

| Cross-pair | Leak gated? | Mechanism |
|---|---|---|
| DeepLive train ↔ VisoMaster val | yes | shared `realpool_<id>` |
| DeepLive train ↔ Teams (passthrough) val | yes (in same trainer split) | shared `realpool_<id>` |
| Trainer train (any) ↔ Teams target-domain manifest lockbox | **NO** | different identity namespace |
| Proper-data train (one capture session) ↔ Proper-data lockbox (another session of same person) | **NO** | split-group, not identity |
| External VCD (md5) ↔ anchor pool (Dor/Roee) | **NOT AUDITABLE** | md5 doesn't carry name |
| df40 train ↔ anything else | yes (df40_ prefix is independent) | trivially separated |

---

## 12. Surfaced suspicions, ranked

1. **Family-aware augmentation router is a hard-coded per-class image-statistics leak.** This is the single most concrete shortcut hypothesis and is exactly what P10 SYM packet was built to test. The fact that the user's own MEMORY notes the camera-signature shortcut and that the team is investigating P10 suggests this is the leading candidate. (Section 10.A.)
2. **Source-label asymmetry on unpaired reals.** External VCD, avspeech, and dor-roee bucket data are real-only. A model can learn "bucket → real" with no face-content reading. (Section 10.B.)
3. **Trainer split ↔ Teams target-domain manifest split are not coordinated.** Identity strings in the two systems live in different namespaces, so the same-person-on-different-day cross-leak between trainer-train and lockbox is not gated. (Section 11.)
4. **Proper-data split is on capture-session, not on person identity.** Same person, two sessions, can appear in train and lockbox. (Section 3 + 11.)
5. **Teams family weights (5.0×) are 33× df40 family weights (0.15×).** The training loss is dominated by Teams; any shortcut available in the Teams data (codec, camera, identity) is the cheapest gradient direction. (Section 9.)
6. **`visomaster_hints` lane disabled in P10 because it is "bad data, for example."** If any pre-P10 checkpoint loaded as `gcs_base_checkpoint` was trained with hints enabled, the bad-data signal is encoded in the base weights. The P10 baseline loads `gs://training-job-outputs/phase2r13_experiments/h2pdu6i5/value_composite_effort_20260424_step23500_auc0.9942_eer0.0169.pth` — not auditable from this report which lineage that checkpoint came from, but worth checking the upstream config. (Section 4.)
7. **Bad-data filter on Teams source assumes the bad-data CSV is current.** If new bad VisoMaster samples enter the cropped-teams-v2 bucket without being added to the CSV, they would silently leak into training as "clean Teams." (Section 2.)
8. **df40 fake at 0.15 weight, df40 real at 0.4 weight — fake/real ratio is 0.375 within df40.** This is unusual; intuitively one would expect ratio 1.0 since the df40 source itself is paired. The asymmetric weight means the trainer effectively over-samples df40 reals relative to df40 fakes — possibly to compensate for some other imbalance, but the rationale is not in-code-commented. **Flag for owner verification.**

---

## Appendix A: Path inventory of files inspected

- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/experiments/phase2_round13/R13_P10_SYM_baseline.yaml`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/experiments/phase2_round13/R13_RLP1_01_FT_WTB1_no_hints_live.yaml`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/experiments/phase2_round13/R13_RLP1_02_FT_WTB2_hints_only_live.yaml`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/experiments/phase2_round13/R13_RLP1_03_FT_WTB3_hints_plus_teams_hints_live.yaml`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/sources/combined_paired.py` (4877 lines)
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/sources/deeplive.py` (700 lines)
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/sources/df40_paired.py` (659 lines)
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/sources/visomaster.py` (1919 lines)
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/sources/proper_data.py` (732 lines)
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/sources/__init__.py`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/validation_sources.py`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/augmentations/pipelines.py` (1912 lines, especially 1148–1418 for family quality pipelines, 1421–1483 for symmetric, 1486–1525 for teams_passthrough, 1530–1714 for the QualityTargetedFamilyRouter)
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/utils/grouping.py`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/policy/visomaster_bad_data/VISOMASTER_BAD_DATA_POLICY_MANIFEST_2026-04-17.csv`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/policy/visomaster_bad_data/VISOMASTER_BAD_DATA_POLICY_SUMMARY_2026-04-17.json`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/policy/visomaster_bad_data/VISOMASTER_BAD_DATA_POLICY_REPORT_2026-04-17.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/policy/visomaster_bad_data/VISOMASTER_BAD_DATA_UPLOAD_AUDIT.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml` (180 792 lines)
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json` (329 049 lines)
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json` (top metadata only)
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/configs/dor_roee_combined_2026-04-24.yaml`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/dataset/deeplive_dataset.py` (top 250 lines, plus enhanced strategy resolver)

## Appendix B: Git references for key decisions

| Commit | Subject | Date | Relevance |
|---|---|---|---|
| `b208d17` | Freeze WT-A policy truth and lane semantics | 2026-04-17 | Establishes the bad-data policy framework |
| `f9303eb` | Land WT-B runtime and launcher smoke readiness | 2026-04-17 | Wires the bad-data policy into combined_paired runtime |
| `d07f06d` | Merge WT-F proper-data schema track | (~2026-04-17 to 04-19) | Creates proper-data inventory schema |
| `22b6819` | docs: add future proper-data schema and manifest builder | (~04-17–19) | Schema docs |
| `77facfc` | Add proper-data runtime and relaunch review packet | 2026-04-19 | Wires proper-data into combined_paired |
| `2c9778b` | Add P10 anti-shortcut packet: symmetric router + GRL slate | 2026-04-25/26 | Adds `routing_mode: symmetric` and the P10 yamls |
| `733b343` / `44743be` / `5533985` / `1fa0360` / `7cde78c` / `f9303eb` | WT-B claim → draft → merge → blocked → smoke → land | 2026-04-17 | Hint-lane runtime came in via WT-B; subsequently disabled in production runs once proper_data was integrated |
