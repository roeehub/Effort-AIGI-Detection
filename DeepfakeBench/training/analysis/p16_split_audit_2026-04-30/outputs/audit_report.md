# P16 Split Audit Report (§10.5)

**Date**: 2026-04-30
**Eval manifest**: `teams_target_domain_manifest_2026-04-06_frozen.json`
**Eval slice/split**: `visomaster_enhanced_macro` / `dev`
**Resolver cache**: `enhanced_visomaster_resolver_2026-04-06.json`

## Verdict: **PASS**

- n_eval_videos: 550
- n_eval_sequence_ids: 275
- n_eval_buckets: 1
- n_eval_frame_paths: 550
- n_train_sample_ids: 999
- n_train_buckets: 2
- n_train_frame_paths: 0

## Checks

### ✅ PASS — bucket_disjointness

train_buckets=2, eval_buckets=1, overlap=0

### ✅ PASS — sequence_id_disjointness

n_eval_sequence_ids=275, n_train_sample_ids=999, matched_pairs=0

### ✅ PASS — frame_path_disjointness

train_frame_paths=0, eval_frame_paths=550, overlap=0
