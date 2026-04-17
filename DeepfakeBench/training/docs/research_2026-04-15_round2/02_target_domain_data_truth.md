# Target-Domain Data Truth

## 1. Scope

This file separates three things that kept getting mixed together:

- what exists in the frozen Teams evaluation manifest
- what exists in local training discovery truth
- what the loader can actually emit at training time

The main result is simple and important:

> Local repo truth still does not contain a lane that emits an actually enhanced-through-Teams fake frame.

## 2. Evidence sources used here

- `arena/manifests/teams_target_domain_manifest_2026-04-06_frozen.json`
- `arena/target_domain_suites.teams_manifest.frozen_2026-04-06.yaml`
- `.viewer_cache/discovery/*.json`
- `data/sources/visomaster.py`
- `data/sources/combined_paired.py`

Remote `training-data-viewer` access was not available in this runtime, so local discovery cache was used instead.

## 3. Frozen evaluation-manifest truth

The frozen Teams target-domain manifest contains `7276` videos total:

- `4614` real
- `2662` fake
- `5662` dev
- `1614` lockbox

### Condition inventory in the frozen manifest

| Condition | Dev | Lockbox | What is actually present |
| --- | ---: | ---: | --- |
| Teams real | 3253 | 1361 | Strong real-side target-domain coverage |
| Teams fake | 2409 | 253 | Direct Teams fake coverage exists; lockbox is small |
| Enhanced clean fake | 1095 | 0 | `550` VisoMaster enhanced + `545` DeepLive enhanced, dev-only |
| Enhanced-through-Teams fake | 0 | 0 | No explicit slice and no local manifest row hit combining `teams` and `enhanced` |
| External/webcam-style reals (non-Teams) | 0 | 0 | No separate external-real eval lane in this frozen manifest |

### Important lockbox asymmetry

- `teams_real_all_lockbox`: `1361`
- `teams_fake_all_lockbox`: `253`
- fake lockbox is only two sessions:
  - `teams_capture_cam_test_s33`: `191`
  - `teams_capture_pc_generator_s15`: `62`
- enhanced fake lockbox is absent:
  - `visomaster_enhanced_macro`: `0`
  - `deeplive_enhanced`: `0`

So the eval manifest is strong on real Teams safety, decent but narrow on direct fake lockbox, and still missing enhanced fake lockbox entirely.

## 4. Local training-discovery truth

Using the same source order as `viewer/discovery.py::discover_all`, local discovery truth totals `31064` rows before any config-specific filtering:

- `df40`: `9396`
- `deeplive`: `5800`
- `visomaster`: `11178`
- `visomaster_teams_enhanced`: `1994`
- `teams`: `2696`
- `external_reals`: `0`

With the baseline merged-source split behavior reproduced locally, the global split is:

- `train`: `26588`
- `val_in_dist`: `3016`
- `test`: `1460`

### Direct Teams source (`deeplive_teams`) in the baseline split

| Split | Real pairs | Fake pairs |
| --- | ---: | ---: |
| Train | 1135 | 1135 |
| Val | 130 | 130 |
| Test | 83 | 83 |

Train fake strategy breakdown:

- `edge_cases`: `343`
- `minimal_processing`: `336`
- `quality_enhancement`: `258`
- clean VisoMaster-through-Teams rows:
  - `InStyleSwapper256-A`: `32`
  - `CSCS`: `31`
  - `InStyleSwapper256-B`: `28`
  - `GhostFace-v1`: `27`
  - `GhostFace-v2`: `27`
  - `Inswapper128`: `26`
  - `GhostFace-v3`: `25`
- `unknown`: `2`

This is real Teams transport data, but it is not an enhanced-through-Teams lane.

### Merged VisoMaster-teams-enhanced source (`visomaster_teams_enhanced`) in the baseline split

| Split | Real pairs | Fake pairs |
| --- | ---: | ---: |
| Train | 837 | 837 |
| Val | 111 | 111 |
| Test | 49 | 49 |

Fake-side train breakdown by companion status:

- `teams_v2_companion`: `42`
- `clean_companion_only`: `795`

That means the baseline merged source is dominated by clean-fallback companions, not true Teams companions.

### External/webcam-style reals in local training truth

`external_reals.json` is empty.

Local consequence:

- there is no discovered external real training pool
- any "realboost" effect that depends on those unpaired reals is unsupported locally

## 5. Teamsonly split truth

Round 1 treated teamsonly as if it kept the same global identity split and only removed the clean-fallback VTE rows. That is not what runtime does.

The `companion_domains` filter is applied **before** the identity split. Under `R13_FT15_trackA_scorecard_base_teamsonly.yaml`, local truth becomes:

- `visomaster_teams_enhanced` fake pairs:
  - train: `49`
  - val: `3`
  - test: `2`
- total train identities shrink from `6716` to `5915`

So teamsonly does remove almost all clean-fallback merged pairs, but it also slightly reshuffles the full split.

## 6. Loader emission truth

The key runtime behavior is in `data/sources/visomaster.py` and `data/sources/combined_paired.py`.

### What the VTE loader does

- Status filtering happens first.
- `companion_domain` is then derived:
  - `teams_v2_companion` -> `teams_v2`
  - anything else -> `clean_fallback`
- `companion_domains` filtering happens after that domain derivation.
- At frame emission time:
  - real always comes from the resolved companion bucket
  - fake comes from the companion/original branch only when `branch == original`
  - enhanced branches always come from the clean enhanced bucket

### What that means

- A VTE sample can give:
  - Teams real + Teams original fake
  - clean real + clean original fake
  - Teams real + clean enhanced fake
  - clean real + clean enhanced fake
- A VTE sample does **not** give:
  - Teams real + Teams enhanced fake
  - clean real + Teams enhanced fake

The exact missing condition is therefore not just underrepresented. It is structurally absent.

## 7. The actual data story by requested condition

| Requested condition | Evaluation truth | Training truth | Loader truth |
| --- | --- | --- | --- |
| Teams real | Strong: `3253` dev, `1361` lockbox | Strong direct Teams real pairs: `1135` train | Emitted directly from `deeplive_teams` and as VTE companion reals |
| Teams fake | Stronger than enhanced: `2409` dev, `253` lockbox | Strong direct Teams fake pairs: `1135` train | Emitted directly from `deeplive_teams`; also as VTE original branch for true Teams companions |
| Enhanced clean fake | Present only in dev eval: `1095` total | Strong in baseline VTE: mostly `795` clean-fallback train pairs plus clean enhanced branch | Emitted from the clean enhanced bucket |
| Enhanced-through-Teams fake | Absent locally | Absent locally | Impossible under current branch logic |
| Merged-source clean fallback | Not an eval slice | Dominant in baseline VTE: `795` of `837` train fake pairs | Real companion is clean fallback, enhanced fake still clean |
| External/webcam-style reals | No separate eval lane | `0` discovered locally | No local emitted lane because none were discovered |

## 8. Most important missing conditions

1. No enhanced-through-Teams fake train lane.
2. No enhanced-through-Teams fake eval lane.
3. No enhanced fake lockbox.
4. No local external real training pool.
5. Fake lockbox is narrow and session-concentrated.

## 9. Status

- Established:
  - the repo has real Teams and direct Teams fake coverage
  - the repo has enhanced clean fake coverage
  - the repo does not have enhanced-through-Teams fake coverage
  - local external reals are absent
- Plausible:
  - some historical experiment labels implied richer Teams-enhanced supervision than runtime really delivered
- Still unknown:
  - whether remote buckets or newer manifests now contain enhanced-through-Teams data not present in this local workspace
