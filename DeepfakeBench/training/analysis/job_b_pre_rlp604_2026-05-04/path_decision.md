# Job B Path Decision (Phase 2-4 prep) — 2026-05-04

## Verdict: **Path A** — re-use the already-committed proper-viso suite yaml. **No new suite yaml needed. No image rebuild required for the suite manifest.**

## Deciding evidence

### 1. Runner accepts arbitrary suite-yaml paths via CLI arg.
`arena/run_target_domain_validation_sequential.py` line 923:
```
parser.add_argument("--suite_manifest", type=str, required=True,
                    help="JSON/YAML manifest containing validation suites.")
```
And the launcher script `arena/launch_teams_promotion_contract.sh` line 88-89, 119:
```bash
--suite-manifest)
    SUITE_MANIFEST="$2"; shift 2 ;;
...
SUITE_MANIFEST="$(normalize_container_local_path "${SUITE_MANIFEST}")"
```
The launcher normalizes any repo-relative path to a `/workspace/...` path inside the container, then passes it to the runner. So **any committed file under `arena/` is reachable** without code change.

### 2. Existing yaml has every suite Job B needs.
`arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml` (committed at HEAD = `77facfc Add proper-data runtime and relaunch review packet`) defines all 16 proper-viso suites, including the 8 we need for Job B:

| Lane | Suite name | Manifest slice tag | Videos in HEAD manifest |
|---|---|---|---|
| Real | `proper_real_clean_dev` | `proper_real_clean` | 1242 (split: dev) |
| Real | `proper_real_teams_dev` | `proper_real_teams` | 1242 (split: dev) |
| Real | `proper_real_clean_lockbox` | `proper_real_clean` | (lockbox subset) |
| Real | `proper_real_teams_lockbox` | `proper_real_teams` | (lockbox subset) |
| Fake | `proper_visomaster_clean_dev` | `proper_visomaster_clean` | 234 (split: dev) |
| Fake | `proper_visomaster_teams_dev` | `proper_visomaster_teams` | 234 (split: dev) |
| Fake | `proper_visomaster_enhanced_clean_dev` | `proper_visomaster_enhanced_clean` | 1008 (split: dev) |
| Fake | `proper_visomaster_enhanced_teams_dev` | `proper_visomaster_enhanced_teams` | 1008 (split: dev) |

Source: `git show HEAD:DeepfakeBench/training/arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml | grep -E 'name:'` (lists 16 suites) plus a Counter over the committed manifest's `videos[].slices` field.

### 3. Both yaml and JSON manifest are tracked at HEAD → already inside the most-recent built image.
- `git ls-files`: both files tracked.
- `git log --oneline ...yaml | head`: `77facfc Add proper-data runtime and relaunch review packet`.

### 4. Checkpoint paths verified live on GCS (`gsutil ls`, no download).
All four `.pth` URIs return their canonical paths — see `checkpoint_map_DRAFT.yaml` for the full list.

## Gotchas

### Gotcha 1 — image-currency mtime false positive (CRITICAL for launch)
Both files are **modified in working tree** vs HEAD:
- `arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml`: 34 lines changed (path style: now absolute Mac-host paths instead of repo-relative).
- `arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json`: +226k / -121k lines (manifest content rebuilt locally).

The launcher invokes `scripts/launch/check_image_currency.sh` which compares **local file mtime** (which now exceeds image push time) and would abort with exit 1.

The image actually contains the HEAD-committed content (which has the right slices and content-resolves correctly), so the mtime mismatch is a benign false positive.

**Mitigation:** export `SKIP_IMAGE_CURRENCY_CHECK=1` for this launch only. The launch command in this packet includes that env var. Alternative: `git stash -- arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json`, launch, then `git stash pop` — but the SKIP env var is simpler and reversible.

### Gotcha 2 — promotion-contract default args reference suites that DO NOT EXIST in the proper-viso yaml
`arena/run_target_domain_validation_sequential.py` lines 979-991 default to:
- `--promotion_dev_real_suite teams_real_all_dev`
- `--promotion_dev_fake_suites teams_fake_all_dev,visomaster_enhanced_macro_dev,deeplive_enhanced_dev`
- `--promotion_lockbox_real_suite teams_real_all_lockbox`
- `--promotion_lockbox_fake_suite teams_fake_all_lockbox`

None of those exist in the proper-viso yaml. So we cannot use `--promotion_contract_dir` with the proper-viso yaml unless we override every promotion arg.

**Decision: skip `--promotion_contract_dir` for Job B.** The viso ceiling question is answered directly by `--scorecard_wide_csv` (gives `fake_recall_at_0p5` per checkpoint × suite). No calibrated-contract scoring needed for a binary trajectory-vs-universal verdict.

### Gotcha 3 — checkpoint_map needs to be inside the image
The launcher's `normalize_container_local_path` rewrites `--checkpoint-map` to `/workspace/<repo-relative>` (line 78). Since `analysis/job_b_pre_rlp604_2026-05-04/checkpoint_map_DRAFT.yaml` is committed only locally (not in any image), **the checkpoint_map MUST be moved into a tracked location before launch** — but that requires either (a) a commit + image rebuild (costs the user spend), or (b) inlining the 4 paths via env vars.

**Recommended:** the user can `cp` the draft to `arena/checkpoint_maps/teams_target_domain.pre_rlp6_04_baseline_2026-05-04.yaml`, commit, **then rebuild image** — this is the single rebuild that makes the checkpoint_map reachable inside the container.

OR — quick-and-dirty alternative: pass the 4 paths as `R4_FT*_CKPT` env vars and skip `--checkpoint_map`. But the runner's `_resolve_requested_checkpoint_keys` only allows `FT1..FT8` aliases without a `--checkpoint_map`, which forces opaque names that don't survive in the scorecard. **NOT recommended.**

## Image rebuild requirement: see `image_rebuild_required.txt`

`yes` — to make `arena/checkpoint_maps/teams_target_domain.pre_rlp6_04_baseline_2026-05-04.yaml` reachable inside the container.

The reason is NOT the suite yaml (Path A reuses the already-baked yaml) — it's purely so the new checkpoint_map can be loaded inside `/workspace/arena/checkpoint_maps/`. There is no other way to get the 4-checkpoint map into the image.

## Summary

| Component | Status | Action needed |
|---|---|---|
| Suite manifest yaml | Already in latest image (HEAD-committed) | None — re-use |
| Suite JSON manifest | Already in latest image (HEAD-committed) | None — re-use |
| Checkpoint map | Drafted under `analysis/`, NOT in any image | Promote to `arena/checkpoint_maps/`, commit, rebuild image |
| Checkpoint paths | All 4 live on GCS, verified | None |
| Image-currency check | Would false-flag suite yaml + manifest | Export `SKIP_IMAGE_CURRENCY_CHECK=1` at launch |
| `--promotion_contract_dir` | Cannot use without arg overrides | Omit the flag — wide scorecard answers Job B's question |
