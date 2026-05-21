# Substrate-pair inventory — FACTS (2026-05-22)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, unfortunately, remarkably, shortcut-aligned, lucky, confirmed, refuted, gap-is-wide, gap-is-narrow.
> Numbers + tables + cross-references only. Interpretation belongs elsewhere.
>
> **Scope**: Phase 0 A0.1 — enumerate every usable paired-transport (clean ↔ teams) real-video capture across all on-disk paired-real inventories. Emit a single CSV manifest, a JSON gates summary, and this FACTS document.
>
> **Inputs (READ-ONLY, on-disk only — no GCS calls performed)**:
> 1. `arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml` (180,792 lines; 1,826 captures × 4 variants each)
> 2. `arena/reports/proper_visomaster_wave_2026_04_19_provisional_build_report.json` (per-source kept / skipped tallies)
> 3. `analysis/p16_split_audit_2026-04-30/_cache/enhanced_visomaster_resolver_2026-04-06.json` (999 enhanced-visomaster resolver rows; 54 `teams_v2_companion`, 943 `clean_companion_only`, 2 `missing_companion`)
> 4. `arena/reports/hdtf_visomaster_clean_vs_teams_join_2026-04-19.json` and `arena/reports/quickclips_visomaster_clean_vs_teams_join_2026-04-19.json` (consulted for source-level census; counts mirror the build-report `source_pairs[*]` block)
>
> **Script**: `build_inventory_manifest.py` (end-to-end reproducible; `python3 -m yaml` required).

---

## 1. Method

1. Load the provisional YAML (`yaml.safe_load`); enumerate `captures[*]`.
2. For each capture, locate its two real variants (`label == "real"`): `transport == "clean"` and `transport == "teams"`. Skip any capture missing either real variant. (Provisional inventory only retains 16/16 captures by construction — see build-report `kept_sample_count`.)
3. Derive `clean_bucket`, `clean_prefix`, `teams_bucket`, `teams_prefix` from the parent path of each variant's first `frame_paths` entry, stripping the `gs://` scheme and keeping the trailing `samples/{id}/frames/real/` slash.
4. Per-row frame counts equal `len(frame_paths)` per side (provisional inventory enforces 16/16).
5. Map `base_capture_id` prefix → `source` label: `HDTF*` → `hdtf_visomaster_teams`, `QCLIP*` → `quickclips_visomaster_teams`.
6. Load the enhanced resolver JSON; iterate `rows[*]` keeping only `resolution_status == "teams_v2_companion"` (54 rows). For each:
   - `clean_bucket` = `live-deepfake-methods-real-and-fake-frames-cropped` (the canonical clean-side enhanced-visomaster bucket per the resolver's `config.clean_bucket`).
   - `clean_prefix` = `samples/{sample_id}/frames/real/`.
   - `teams_bucket` = `resolved_companion_bucket` (constant `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`).
   - `teams_prefix` = `resolved_companion_path + "frames/real/"`.
   - `teams_real_frame_count` = `resolved_real_frame_count` (range 17–22; sum 1,050).
   - `clean_real_frame_count` = blank/null (the resolver row schema does not carry a clean-side frame count; `clean_real_exists` is true). Recorded in the `notes` column.
   - `identity_id` falls back to `sample_id` because the resolver row does not embed `manifest.original_video_name`. Following `data/sources/visomaster.py::VisoMasterTeamsEnhancedSample.identity`, when `manifest.original_video_name` is empty the fallback path returns the post-`visomaster_` suffix after the last underscore (5-digit numeric token in this corpus). The script replaces that fallback with the full `sample_id` for traceability.
7. Concatenate provisional rows + resolver rows; write `inventory_manifest.csv` with the 10-column schema.
8. Aggregate counts into `gates_summary.json`. Phase-0 gate threshold: `total_pairs >= 500 → FLOOR_MET`.
9. The `pairs_with_sufficient_frames` count uses ≥5 frames per side and treats `clean_real_frame_count = null` as sufficient (because `clean_real_exists` is true in every kept resolver row, and the 17–22-frame teams-side count is a per-sample floor).

---

## 2. Output artifacts

| Artifact | Path | Size / Shape |
|---|---|---|
| Per-pair manifest CSV | `analysis/substrate_pair_geometry_2026-05-22/inventory_manifest.csv` | 1,880 data rows × 10 cols |
| Gates summary JSON | `analysis/substrate_pair_geometry_2026-05-22/gates_summary.json` | 12-key flat object |
| Builder script | `analysis/substrate_pair_geometry_2026-05-22/build_inventory_manifest.py` | 1 module, CPU-only, no GCS |

`gates_summary.json` headline fields:
- `total_pairs: 1880`
- `phase_0_gate_status: "FLOOR_MET"` (threshold 500)
- `pairs_with_sufficient_frames: 1880` (all kept pairs have ≥16 frames per side, or ≥17 on the teams side for the resolver rows; clean side is treated as sufficient because `clean_real_exists` is true)
- `pairs_clean_count_missing: 54` (every `visomaster_teams_enhanced` row)

---

## 3. Tables

### Table 3.1 — Rows per source

| Source | Pair rows | Distinct `identity_id` | Distinct `base_capture_id` | Clean bucket | Teams bucket |
|---|---:|---:|---:|---|---|
| `hdtf_visomaster_teams` | 1,094 | 328 | 1,094 | `hdtf_visomaster_cropped_frames` | `hdtf_visomaster_cropped_frames_teams` |
| `quickclips_visomaster_teams` | 732 | 377 | 732 | `quickclips_visomaster_cropped_frames` | `quickclips_visomaster_cropped_frames_teams` |
| `visomaster_teams_enhanced` | 54 | 54 (sample-id-derived) | 54 | `live-deepfake-methods-real-and-fake-frames-cropped` | `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2` |
| **TOTAL** | **1,880** | n/a | **1,880** | — | — |

### Table 3.2 — Frames per source and side

| Source | Clean-side frames (sum) | Teams-side frames (sum) | Per-side frames / pair | Notes |
|---|---:|---:|---|---|
| `hdtf_visomaster_teams` | 17,504 | 17,504 | 16 / 16 (all rows) | Provisional inventory drops ragged-teams (151) and ragged-clean (35) HDTF captures. |
| `quickclips_visomaster_teams` | 11,712 | 11,712 | 16 / 16 (all rows) | Provisional inventory drops ragged-teams (16) and ragged-clean (14) quickclips captures. |
| `visomaster_teams_enhanced` | *(not in resolver)* | 1,050 | clean: unknown · teams: 17–22 (mean 19.44) | Per-row `clean_real_frame_count` left blank; `clean_real_exists=true` for all 54 rows. |
| **TOTAL** | **29,216** | **30,266** | — | Clean-side total excludes the 54 enhanced-visomaster rows. |

### Table 3.3 — Top-15 identities by paired-capture count

#### 3.3a `hdtf_visomaster_teams` (top by paired captures; ties broken by enumeration order)

| Rank | `identity_id` | Paired captures | Paired frames (clean + teams) |
|---:|---|---:|---:|
| 1 | `RD_Radio14` | 4 | 128 |
| 2 | `RD_Radio27` | 4 | 128 |
| 3 | `RD_Radio33` | 4 | 128 |
| 4 | `RD_Radio51` | 4 | 128 |
| 5 | `WDA_JoeDonnelly` | 4 | 128 |
| 6 | `WRA_DebFischer2` | 4 | 128 |
| 7 | `WRA_JohnKasich3` | 4 | 128 |
| 8 | `WRA_MarshaBlackburn0` | 4 | 128 |
| 9 | `WRA_PeterRoskam0` | 4 | 128 |
| 10 | `RD_Radio45` | 4 | 128 |
| 11 | `WDA_CatherineCortezMasto` | 4 | 128 |
| 12 | `WDA_DebbieDingell1` | 4 | 128 |
| 13 | `WDA_EricSwalwell` | 4 | 128 |
| 14 | `WRA_CoryGardner` | 4 | 128 |
| 15 | `WRA_DianeBlack1` | 4 | 128 |

HDTF captures-per-identity histogram: `4-cap: 156 ids · 3-cap: 132 ids · 2-cap: 34 ids · 1-cap: 6 ids` (sum 328 ids → 1,094 captures).

#### 3.3b `quickclips_visomaster_teams` (top by paired captures; ties broken by enumeration order)

| Rank | `identity_id` | Paired captures | Paired frames (clean + teams) |
|---:|---|---:|---:|
| 1 | `322` | 2 | 64 |
| 2 | `3265` | 2 | 64 |
| 3 | `3741` | 2 | 64 |
| 4 | `4109` | 2 | 64 |
| 5 | `4308` | 2 | 64 |
| 6 | `5328` | 2 | 64 |
| 7 | `5865` | 2 | 64 |
| 8 | `7504` | 2 | 64 |
| 9 | `7866` | 2 | 64 |
| 10 | `7989` | 2 | 64 |
| 11 | `1066` | 2 | 64 |
| 12 | `1259` | 2 | 64 |
| 13 | `1758` | 2 | 64 |
| 14 | `4072` | 2 | 64 |
| 15 | `4736` | 2 | 64 |

quickclips captures-per-identity histogram: `2-cap: 355 ids · 1-cap: 22 ids` (sum 377 ids → 732 captures). All `identity_id` tokens in quickclips are bare numeric strings (e.g. `"322"`, `"3265"`); the `capture_session_id` field carries the 11-char YouTube ID (e.g. `-jlI8ntZrEY`).

---

## 4. Direct observations

1. **Total paired captures = 1,880** (1,094 HDTF + 732 quickclips + 54 enhanced-visomaster teams_v2_companion); `phase_0_gate_status = FLOOR_MET` against the 500-pair floor.
2. **HDTF + quickclips real-frame totals are symmetric at 16/16** by construction (provisional inventory's `kept_sample_count` requires both real streams at exactly 16 frames). Per the build report: HDTF skipped `151` ragged-teams + `35` ragged-clean captures; quickclips skipped `16` ragged-teams + `14` ragged-clean captures. None of those skipped captures appear in this manifest.
3. **54 `visomaster_teams_enhanced` rows carry no `clean_real_frame_count`** (the resolver row schema only records `resolved_real_frame_count` on the resolved companion side). `clean_real_exists` is `true` for all 54; verifying the clean-side count would require `gsutil ls samples/{sample_id}/frames/real/` against `live-deepfake-methods-real-and-fake-frames-cropped`. Not performed under the CPU-only / no-GCS constraint.
4. **Teams-side frame counts for `visomaster_teams_enhanced`** range 17–22 per sample (mean 19.44, total 1,050 frames). This deviates from the 16-per-side regime used by HDTF/quickclips.
5. **`identity_id` schema is heterogeneous across the three sources**:
   - HDTF: speaker tokens such as `RD_Radio14`, `WDA_CatherineCortezMasto`, `WRA_PeterRoskam0` (328 distinct identities across 1,094 captures; 156 of them appear 4 times).
   - quickclips: bare numeric tokens such as `322`, `3265`, `4109` (377 distinct identities across 732 captures; 355 of them appear 2 times). YouTube source IDs live in the unused `capture_session_id` column.
   - `visomaster_teams_enhanced`: 54 distinct `identity_id` values — but all 54 are simply the row's `sample_id` (`visomaster_<strategy>_<5digit>`) because the resolver row does not embed `manifest.original_video_name`. These are NOT cross-source-comparable identities; they are strategy+index slot identifiers.
6. **No paired-transport rows from non-VisoMaster `deeplive_teams` strategies** appear in this manifest. No canonical paired manifest exists on disk that joins clean-bucket deeplive samples to their teams-v2-bucket counterparts at the `sample_id` level; per `D9_FACTS_2026-05-12.md` §4.1, teams-v2 has 1,346 within-bucket complete pairs (fake / real Teams capture both present) but those are not cross-bucket pairs. This manifest reports 0 contributions from that lane.
7. **Strategy distribution for `visomaster_teams_enhanced` (54 rows)**: `GhostFace-v1`: 11, `GhostFace-v3`: 11, `InStyleSwapper256-A`: 9, `InStyleSwapper256-B`: 7, `Inswapper128`: 6, `CSCS`: 5, `GhostFace-v2`: 5. All 54 rows resolve to the same companion bucket (`...-teams-v2`).
8. **All 1,880 pair rows clear the ≥5-frames-per-side gate**: HDTF and quickclips have 16/16; `visomaster_teams_enhanced` has 17–22 on the teams side (clean side unverified but `clean_real_exists=true`).
9. **Divergence from audit predictions**: zero numeric divergence. Actual totals match the plan's "Expected totals" preview exactly (1,094 + 732 + 54 = 1,880; HDTF clean frames 17,504; quickclips clean frames 11,712; enhanced teams-side frames 1,050). One schema divergence: the plan suggested quickclips `identity_id` tokens such as `0Hn3GMnjzAM_rank01_clip01`; the on-disk YAML actually stores bare numeric strings (`322`, `3265`, …), with the YouTube ID in the separate `capture_session_id` field.
