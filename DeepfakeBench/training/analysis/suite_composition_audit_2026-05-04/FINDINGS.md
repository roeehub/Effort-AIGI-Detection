# Suite composition audit — 2026-05-04 (CPU Job 1 of 3)

Pure metadata join. No model inference. 45 frames_report CSVs from
`analysis/cpu_followups_2026-05-04/raw_reports/` joined against
`arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json`
(the manifest referenced by the authoritative
`arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml`,
which is the canonical 9-suite scorecard Job 2/3 will consume).

The other yamls (`r13_best_megaval_2026-04-13`, `teams_manifest.frozen_2026-04-06`,
`teams_manifest.breakdown_2026-04-07`, `proper_data_future.provisional_2026-04-19`)
are either earlier frozen versions or forward-looking and not what the 45
reports were generated from.

Total frames tagged: 68,180 across 9 suites x 5 ckpts. Coverage:
`source_method` 100%, `teams_passthrough` 100%, `enhancement` 82.6%
(unresolved 17.4% are session-captures and the Xiang flat-upload — see section 4).

---

## 1. Suite-level totals (one ckpt per suite)

| suite                                   | n_frames | n_identities | label  |
|-----------------------------------------|---------:|-------------:|--------|
| `deeplive_enhanced_dev`                 |     545  |       1      | fake   |
| `visomaster_enhanced_macro_dev`         |     550  |       2      | fake   |
| `teams_fake_all_dev`                    |    3039  |      16      | fake   |
| `teams_fake_all_lockbox`                |     425  |       2      | fake   |
| `teams_real_all_dev`                    |    4564  |      24      | real   |
| `teams_real_all_lockbox`                |    1418  |       5      | real   |
| `teams_real_dor_dev`                    |      50  |       1      | real   |
| `teams_real_lighting_extreme_dev`       |    1742  |      20      | real   |
| `teams_real_poor_quality_dev`           |    1303  |      18      | real   |

## 2. CLEAN variants of deeplive / visomaster

Zero clean fakes are evaluated by any of the 9 active scorecard suites.

The manifest's slice taxonomy contains only `deeplive_enhanced` (545) and
`visomaster_enhanced_macro` (550) for fake-creation methods. Tokens like
`*_clean*` exist only in the forward-looking yaml
`target_domain_suites.proper_data_future.provisional_2026-04-19.yaml`
(suites `proper_visomaster_clean_*`, `proper_fake_clean_all_*`), which have
NOT been wired into the production scorecard. Past
`deeplive_enhanced_dev` / `visomaster_enhanced_macro_dev` recall headlines
have therefore measured only the enhanced (sharpened/upscaled) variants;
clean recall is currently unmeasured.

## 3. Pure-transport vs source-method-pure suites

* Source-method-pure (no Teams passthrough):
  `deeplive_enhanced_dev`, `visomaster_enhanced_macro_dev` — both are
  flat-upload enhanced fakes, never went through Teams capture.
* Pure-transport over reals (Teams passthrough = yes):
  `teams_real_all_dev`, `teams_real_all_lockbox`, `teams_real_dor_dev`,
  `teams_real_lighting_extreme_dev`, `teams_real_poor_quality_dev` — all
  100% `teams_real` / `dor_real_passthrough`.
* Mixed: `teams_fake_all_dev`, `teams_fake_all_lockbox`. See section 4.

## 4. Composition of `teams_fake_all_dev` — IT IS A TRANSPORT MIXTURE

`teams_fake_all_dev` is NOT a homogeneous "teams" method. Of its 3,039
frames:

| component                                | frames | % of suite | identities | source_method          | enhancement | teams_passthrough |
|------------------------------------------|-------:|-----------:|-----------:|------------------------|-------------|-------------------|
| `visomaster_enhanced_macro`              |   550  |    18.1%   |          2 | visomaster             | enhanced    | no (flat-upload)  |
| `deeplive_enhanced`                      |   545  |    17.9%   |          1 | deeplive               | enhanced    | no (flat-upload)  |
| 12 x `teams_capture_*` sessions          |  1809  |    59.5%   |         12 | teams_capture_session  | unknown     | yes               |
| `teams_flat_xiang_xiang2_feng`           |   135  |     4.4%   |          1 | other_flat_fake        | unknown     | no (flat-upload)  |

40.5% of `teams_fake_all_dev` is flat-uploaded creation-method fakes that
never traversed the Teams pipeline. Treating "teams_fake_all_dev recall"
as a Teams-transport metric is therefore misleading — about two-fifths of
the recall is just deeplive_enhanced + visomaster_enhanced + a single Xiang
clip evaluated again under a different suite name.

`teams_fake_all_lockbox` is the opposite extreme — only 2 sessions /
2 identities (`Cam_Test__s33` 334 frames + `PC_Generator__s15` 91 frames),
zero deeplive/visomaster enhanced fakes. Lockbox fake recall is essentially
"can you flag these two specific Teams capture sessions".

The teams_capture_* `enhancement` field is reported as unknown (17.4% of
total rows) because the manifest does not record which upstream creation
tool produced each Teams-captured session, so we cannot tell whether each
session is a clean or enhanced swap. We do not fabricate this field.

## 5. Two surprises that change interpretation of past headlines

1. Identity collapse on the "method-pure" suites. `deeplive_enhanced_dev`
   has 545 frames but only 1 distinct identity (`deeplive_dor`).
   `visomaster_enhanced_macro_dev` has 550 frames / 2 identities
   (`visomaster_enhanced_raw` plus a tiny second identity).
   `teams_real_dor_dev` is 50 frames / 1 identity. Recall numbers reported
   on these suites are essentially per-identity and per-session — the
   reported "27% viso ceiling" and "deeplive_enhanced 88%" headlines are
   not aggregating over a diverse identity set; they are 2 and 1 identities
   respectively. This is consistent with the existing memory note about
   per-identity FPR concentration but worth re-stating when comparing
   methods.

2. `teams_fake_all_dev` macro recall is dominated by non-Teams content.
   1,230 of its 3,039 frames (40.5%) are flat-upload deeplive_enhanced /
   visomaster_enhanced / Xiang fakes that bypass the Teams pipeline
   entirely. Any narrative that frames "teams_fake_all" performance as
   evidence about Teams transport robustness is mixing two distinct signals
   (creation-method-pure flat fakes + Teams-captured sessions of swapped
   reals). Decoupling these in Job 2/3 (e.g., recall on
   `teams_passthrough=yes` cells separately from `teams_passthrough=no`
   cells) will give a cleaner read.

---

## Files written

* `frames_tagged.csv` — 68,180 rows x 13 cols (canonical frame-level
  tagged file for Jobs 2 and 3).
* `composition_table.csv` — 12 cells per (suite x source_method x
  enhancement x teams_passthrough x label) with `n_frames` /
  `n_unique_identities`.
* `composition_method_only.csv` — 24 cells per (suite x method) for sanity.
* `audit_stats.json` — coverage and breakdown stats.
* `run_audit.py` — the build script (re-runnable).
