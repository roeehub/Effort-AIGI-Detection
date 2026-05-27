# Train-overlap audit for D5 recurring residual identities — FACTS (2026-05-12)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, unfortunately, remarkably, shortcut-aligned, lucky, confirmed, refuted.
> Numbers + tables + cross-references only. Interpretation belongs in a sibling OPINIONS doc — for this investigation the OPINIONS surface is §addendum in [`../cpu_diagnostics_2026-05-12_d1_d5_synthesis/D1_D5_CRITIC_REVIEW_2026-05-12.md`](../cpu_diagnostics_2026-05-12_d1_d5_synthesis/D1_D5_CRITIC_REVIEW_2026-05-12.md).
>
> **Question tested**: do the three recurring top-residual identities from D5 §4.4 (`real_dor`, `Cam_Test`, `PC_Generator`) appear in P8A's training pool? Per [`D1_D5_CRITIC_REVIEW_TASK_2026-05-12.md`](../../docs/packet_retrospectives/D1_D5_CRITIC_REVIEW_TASK_2026-05-12.md) §4.4 and `D1_D5_OPINIONS_2026-05-12.md` §5 uncertainty #5, this is the load-bearing test that distinguishes "P8A learned identity-cluster memorization" from "P8A learned identity-level invariance from training-pool identities of these same people."
>
> **Inputs**:
> - Eval manifest: `arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json` (7,326 videos)
> - Eval manifest: `arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json` (7,304 videos)
> - Eval manifest: `arena/manifests/visomaster_enhanced_v2_manifest_2026-04-13.json` (2,073 videos)
> - P8A training yaml: `experiments/phase2_round13/R13_RLP8_01_unfreeze_clip_codec.yaml` (FT chain ancestor)
> - All 158 R13 training yamls: `experiments/phase2_round13/*.yaml` (grep-scanned)
> - 2026-05-04 inventory audit: `analysis/inventory_audit_2026-05-04/identity_overlap_report.csv` (3,435 rows)
>
> **Scripts**: `run_train_overlap.py` (this folder).

---

## 1. Method

1. Load the three eval manifests.
2. For each of the 3 critic-review identities (`real_dor`, `Cam_Test`, `PC_Generator`), enumerate all videos in the eval manifests whose `identity_key`, `prefix`, or `method` field substring-matches the identity. Record per-(split, label) breakdown, session_id list, and source bucket of `frame_paths[0]`.
3. Grep all 158 R13 training yamls for the source-bucket strings discovered in step 2.
4. Grep the same yamls for any `gcs_bucket` / `bucket:` / `frames_bucket` directive.
5. Read the P8A training yaml (`R13_RLP8_01_unfreeze_clip_codec.yaml`) lines 180-280 to catalog the four classes of bucket references: `combined_paired.teams`, `combined_paired.proper_data`, `ood_monitoring.external_real_sources`, `ood_monitoring.readout_only_external_real_sources`.
6. Cross-reference the 2026-05-04 inventory audit's `also_in_training` annotation logic (in `analysis/inventory_audit_2026-05-04/run_audit.py:340-365`) against the identity-level frame provenance.

**Constraints**: no GCS-side enumeration is run. Bucket membership for the training-side bucket `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2` is taken from yaml + repo evidence only.

---

## 2. Per-identity eval-side provenance

Source: `outputs/per_identity_eval_provenance.csv`.

| Identity | n_videos | sessions in eval | (split, label) breakdown | Source bucket(s) of `frame_paths` |
|---|---:|---|---|---|
| `real_dor` | 109 | — (no session_id in manifest rows) | `(lockbox, real)`: 109 | `gs://teams-faces-data-test-2914-fake-4420-real-feb-28` |
| `Cam_Test` | 812 | s32, s33, s35, s38, s46 | `(dev, fake)`: 523; `(lockbox, fake)`: 191; `(dev, real)`: 98 | `gs://teams-faces-data-test-2914-fake-4420-real-feb-28` |
| `PC_Generator` | 790 | s3, s4, s8, s9, s13, s14, s15, s22, s34, s45 | `(dev, real)`: 525; `(dev, fake)`: 175; `(lockbox, fake)`: 62; `(lockbox, real)`: 28 | `gs://teams-faces-data-test-2914-fake-4420-real-feb-28` |

Sample first frames:
- `real_dor` — `gs://teams-faces-data-test-2914-fake-4420-real-feb-28/real/real_dor__frame_000096_seq1022.png`
- `Cam_Test` — `gs://teams-faces-data-test-2914-fake-4420-real-feb-28/fake/Cam_Test__s32_0.0_frame_000074_crop_002__8e9ae142.jpg`
- `PC_Generator` — `gs://teams-faces-data-test-2914-fake-4420-real-feb-28/fake/PC_Generator__s15_1507.4_frame_050780_crop_001__46b23355.jpg`

All three identities' eval-frame paths share the same single source bucket: `gs://teams-faces-data-test-2914-fake-4420-real-feb-28`.

---

## 3. Training-bucket cross-reference

### 3.1 Grep of all R13 training yamls

```bash
grep -rn "teams-faces-data-test-2914" experiments/phase2_round13/
# (no output)
```

**Match count: 0 across all 158 R13 yamls.** The eval bucket `teams-faces-data-test-2914-fake-4420-real-feb-28` is not referenced as a training source in any R13 training yaml.

### 3.2 Buckets actually referenced in P8A's R13_RLP8_01

Source: `experiments/phase2_round13/R13_RLP8_01_unfreeze_clip_codec.yaml` lines 120-280. Classified by yaml block.

| Yaml block | Bucket | Role |
|---|---|---|
| `combined_paired.df40` (line ~120) | `df40-frames-recropped-rfa85` | training |
| `combined_paired.deeplive` (lines 141-159) | `live-deepfake-methods-real-and-fake-frames-cropped` | training |
| `combined_paired.deeplive.frames_bucket` (line 142) | `live-deepfake-methods-real-and-fake-frames` | training |
| `combined_paired.visomaster_enhanced` (lines 158-176) | `visomaster-enhanced-face-cropped` / `enhanced-visomaster-cropped` | training |
| `combined_paired.teams` (line 183) | `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2` | training |
| `combined_paired.visomaster_hints_teams` (line 190, `enabled: false`) | `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2` | training (disabled) |
| `combined_paired.proper_data` (line 196-203) | derived via `arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json` → hdtf_visomaster_cropped_frames | training |
| `combined_paired.external_training_reals` (line 206) | `effort-collected-data/real/VCD` | training |
| `ood_monitoring.external_real_sources` (line 252) | `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2` (path_contains `/frames/real/`, max_videos 300) | OOD readout during training |
| `ood_monitoring.readout_only_external_real_sources` (line 262) | `real-teams-dor-roee/session_20260424_110139/uniform30` (max_videos 10) | readout-only (see §3.3) |
| `ood_monitoring.external_fake_sources` (line 270) | `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2` (path_contains `/frames/fake/`, max_videos 300) | OOD readout during training |
| `ood_monitoring.lighting_stress_sources` (line 283+) | `effort-collected-data/real/external_youtube_avspeech` | OOD readout during training |

`exclude_training_identities: true` (line 242) is set on the `ood_monitoring` block; per loader semantics this enforces identity-disjoint sets between training and OOD.

### 3.3 Semantics of `readout_only_external_real_sources`

Source: comment in `analysis/generate_packet6_yamls_2026-04-23.py:308-309`:

> `readout_only_external_real_sources so their FPR is logged but does NOT gate τ`

`readout_only_*` blocks are observed during training (per-step FPR logged to W&B) but not used to compute training gradients. They are not training data.

### 3.4 Training data sources in P8A: classification

| Bucket | Used for training (gradient updates)? | Used for in-training OOD readout? | Used for eval scorecard? |
|---|:--:|:--:|:--:|
| `df40-frames-recropped-rfa85` | yes | no | no |
| `live-deepfake-methods-real-and-fake-frames-cropped` | yes | no | yes |
| `live-deepfake-methods-real-and-fake-frames` (raw) | yes (frames-bucket lookup) | no | no |
| `visomaster-enhanced-face-cropped` / `enhanced-visomaster-cropped` | yes | no | no |
| `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2` | yes (combined_paired.teams) | yes (ood_monitoring, identity-disjoint via `exclude_training_identities`) | no |
| `hdtf_visomaster_cropped_frames` / `_teams` (via proper_data manifest) | yes (proper_data.include_lanes) | no | yes (HDTF Phase C) |
| `effort-collected-data/real/VCD` | yes (external_training_reals, 1,200 frames cap) | yes (lighting stress) | no |
| `real-teams-dor-roee` (session_20260424_110139) | no | readout-only (FPR logged, no gradient) | no |
| `teams-faces-data-test-2914-fake-4420-real-feb-28` (eval bucket) | **no (0 references in any R13 yaml)** | no | yes (Teams target domain manifest) |
| `visomaster-enhanced-face-cropped-v2` | no | no | yes (v2 substrate manifest) |

---

## 4. Cross-reference: 2026-05-04 inventory audit annotation

Source: `analysis/inventory_audit_2026-05-04/identity_overlap_report.csv`. Audit logic at `analysis/inventory_audit_2026-05-04/run_audit.py:340-365`.

The audit's `also_in_training` column for the three critic-review identities:

| Identity | n rows in audit | `also_in_training` annotation | Annotation source (audit code path) |
|---|---:|---|---|
| `real_dor` | 1 (`teams_real_lockbox` cell) | `yes (teams-v2 bucket is training+ood)` | `run_audit.py:344` — branch `method == "teams_real"` |
| `Cam_Test` (5 sessions) | 7 | `yes (teams-v2 bucket is training+ood)` | `run_audit.py:344-345` — branch `method.startswith("teams_capture_")` and `method == "teams_real"` |
| `PC_Generator` (5 sessions) | 12 | `yes (teams-v2 bucket is training+ood)` | `run_audit.py:344-345` — same branches |

The annotation is set **by Teams-target-domain method-string match**, not by any per-identity check on the training-side bucket's contents. From `run_audit.py:340-348`:

```python
if method.startswith("teams_capture_") or method == "teams_real":
    also_train = "yes (teams-v2 bucket is training+ood)"
```

The annotation is therefore a **soft, bucket-family inference**: the audit assumed identities visible in the Teams target-domain manifest (eval-bucket `teams-faces-data-test-2914-...feb-28`) have training-side counterparts in the teams-v2 bucket because both buckets are part of the project's "Teams pipeline" data family. The audit did not run a per-identity check against teams-v2 bucket contents.

---

## 5. Memory `project_clean_teams_same_identity` re-read

Per memory `project_clean_teams_same_identity.md` (read 2026-05-12): "Clean and Teams buckets share identities (paired transport). `*_teams` buckets are recaptures of the `*_clean` bucket's same identities." This is the closest evidence that the SAME PERSON might appear in both training and eval Teams-pipeline buckets.

However, the memory's claim is about clean↔teams pairing within the *training-side* data (live-deepfake-methods bucket family), not about eval-bucket↔training-bucket identity overlap. The "explicit join reports exist" referenced in that memory cover the training-side clean/teams correspondence.

No memory or analysis artifact in this repo (as of 2026-05-12) provides a per-identity verification that the people named `real_dor`, `Cam_Test`, `PC_Generator` in the eval bucket appear under the same or different names in the training bucket `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`.

---

## 6. Output artifacts

Under `analysis/train_overlap_audit_2026-05-12/`:

| Path | Description |
|---|---|
| `outputs/per_identity_eval_provenance.csv` | Per-identity (n_videos, sessions, source_buckets, split/label breakdown) on the eval Teams target-domain manifest |
| `run_train_overlap.py` | Reproducible single-file script (CPU only, no GCS calls) |

---

## 7. Caveats

1. **No GCS-side enumeration was run.** The training-side bucket `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2` was not directly grep'd for the three identity strings. A GCS audit (`gsutil ls gs://.../samples/ | grep -i 'cam_test\|pc_generator\|real_dor'`) would close this. Cost estimate: <10 min + GCS list-API quota.
2. **Sample-id naming convention in the training bucket is not directly enumerable from this repo.** Per `data/sources/combined_paired.py:1038-1090` (`discover_teams_passthrough_samples`), the training bucket exposes `samples/{sample_id}/manifest.json` files; the `sample_id` is opaque from the yaml side. The `manifest.json` per sample may contain `original_video_name` that names the person, but the discovery code's identity-extraction logic uses hashes (`identity: 1800396478` style) not name strings.
3. **The 2026-05-04 inventory audit's `also_in_training` annotation is bucket-family-level** (§4). Its CSV column does not assert per-identity frame overlap.
4. **`exclude_training_identities: true`** (yaml line 242) is set on the ood_monitoring block. If implemented per stated semantics, OOD-side teams-v2 identities are disjoint from training-side teams-v2 identities. Verification of the implementation (in `data/sources/combined_paired.py` or `ood_monitoring/` loader) was not run.
5. **The eval `real_dor` cohort (109 lockbox-real videos)** and the readout-only `real-teams-dor-roee` source (max 10 videos, session 20260424) are likely from different physical capture sessions even if they share the human subject. The eval `real_dor` frame paths are `real_dor__frame_NNNN_seqNNNN.png` while the readout-only source uses `session_20260424_110139/uniform30/...`.
6. **The audit annotation chain (Cam_Test, PC_Generator)**: even if the same humans appear in the training bucket teams-v2 under different session names, the cleanest claim available without GCS audit is "the training pool likely contains the same humans under different captures" — not "the training pool contains the same frame distributions."

---

## 8. Direct observations

1. All 109 `real_dor` videos in the eval Teams target-domain manifest are in the `(lockbox, real)` cell and have their `frame_paths` rooted at `gs://teams-faces-data-test-2914-fake-4420-real-feb-28` (§2).
2. All 812 `Cam_Test` videos in the eval manifest span 5 sessions (s32, s33, s35, s38, s46) with `(dev, fake)`=523 / `(lockbox, fake)`=191 / `(dev, real)`=98 split; `frame_paths` are exclusively in `gs://teams-faces-data-test-2914-fake-4420-real-feb-28` (§2).
3. All 790 `PC_Generator` videos in the eval manifest span 10 sessions (s3, s4, s8, s9, s13, s14, s15, s22, s34, s45) with `(dev, real)`=525 / `(dev, fake)`=175 / `(lockbox, fake)`=62 / `(lockbox, real)`=28 split; `frame_paths` are exclusively in `gs://teams-faces-data-test-2914-fake-4420-real-feb-28` (§2).
4. The eval bucket `teams-faces-data-test-2914-fake-4420-real-feb-28` has 0 references across all 158 R13 training yamls (§3.1, grep result).
5. P8A's training Teams source is a separate bucket `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2` (§3.2, yaml line 183).
6. The same training bucket (`live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`) is also referenced as an OOD readout source with `exclude_training_identities: true` (§3.2, lines 242-272).
7. The `real-teams-dor-roee` bucket is declared `readout_only_external_real_sources` in P8A's yaml (§3.2 line 262), and `readout_only_*` is documented as "FPR is logged but does NOT gate τ" (§3.3, citation `analysis/generate_packet6_yamls_2026-04-23.py:308-309`).
8. The 2026-05-04 inventory audit's `also_in_training` column annotation for each of the three identities (`yes (teams-v2 bucket is training+ood)`) is set by Teams-target-method-string match in audit code (§4, `run_audit.py:340-348`), not by per-identity bucket-content verification.
9. No artifact in this repo (as of 2026-05-12) provides per-identity frame overlap between the eval bucket and the training-side teams-v2 bucket (§5, §7).
10. The cleanest verifiable claim from local-only evidence: the three eval-substrate identities' frames are not in P8A's training pool at the frame level (different bucket); whether the SAME HUMANS appear in the training bucket under different sample_ids requires a GCS-side audit (§7).
