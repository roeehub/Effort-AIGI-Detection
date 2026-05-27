# Packet A — Wiring verification + minimal yaml diff

Generated: 2026-05-05
Hypothesis: E2B's viso recall is bound by the absence of ENHANCED viso variants in
training data, not by anything fundamental about the model architecture or recipe.

## What's wired in code (verified)

All three sample classes exist in `data/sources/visomaster.py`:
- `VisoMasterSample` (line 61) — V1 clean viso (currently used)
- `VisoMasterEnhancedSample` (line 607) — V1 enhanced viso (NOT in any current packet)
- `VisoMasterTeamsEnhancedSample` (line 741) — V1 enhanced + teams transport (only in failed P14/P16)

`combined_paired.py` has full wiring for all three:
- `create_unified_samples_from_visomaster_enhanced` (line 703)
- `create_unified_samples_from_visomaster_teams_enhanced` (line 761)
- Iteration handlers `_iterate_visomaster_enhanced_sample` (line 2995), `_iterate_visomaster_teams_enhanced_sample` (line 3067)
- Quality-domain mapping at lines 80-87

## GCS resources (verified to exist)

- `gs://visomaster-enhanced-face-cropped/` — V1 enhanced bucket (samples/, cache/)
- `gs://enhanced-visomaster-cropped/` — separate enhanced+teams resolver target
- `gs://training-job-outputs/cache/visomaster/enhanced_visomaster_resolver_2026-04-06.json` — canonical resolver manifest

## What's NOT in any current packet (re-confirmed)

```bash
# E2B / P22 / S2 / S3 viso config (same in all four):
visomaster:
  enabled: true
  gcs_bucket: "live-deepfake-methods-real-and-fake-frames-cropped"  # CLEAN
  swap_models: [9 models]                                            # NOT 16
  tiers: null                                                        # bucket itself is clean
# (no visomaster_enhanced block)
# (no visomaster_teams_enhanced block)
```

## Prior attempts (with confounds)

`R13_P14_DATA_FIX.yaml` (xan4dfto, COLLAPSED):
- visomaster_enhanced.enabled: false (DISABLED)
- visomaster_teams_enhanced.enabled: true, fw=8.0 (heavy weight)
- ALSO: anti-shortcut bundle (jitter + anchor + GRL stacked) — separately net-negative
- ALSO: FT-from-P8A_step5000 base
- Three confounds; not a clean test

`R13_P16_DATA_AXIS.yaml` (rmic6wrc, didn't promote):
- visomaster_enhanced.enabled: false (DISABLED)
- visomaster_teams_enhanced.enabled: true, fw=2.0 (mild weight)
- FT-from-P8A_step5000 base (the 5x-FT pathology applies)
- Two confounds

**Neither tests the clean hypothesis "enable enhanced viso sources from a scratch B16 base at neutral weighting."**

## Packet A — proposed yaml diff vs E2B

Take `R13_E2b_SCRATCH_B16_NO_ARCFACE.yaml` and add this block after the existing `visomaster:` block (between lines 234 and 237). All other settings unchanged.

```yaml
  visomaster_enhanced:
    enabled: true
    gcs_bucket: "visomaster-enhanced-face-cropped"
    original_bucket: "live-deepfake-methods-real-and-fake-frames-cropped"
    exclude_tiers: ["ARTIFACT"]

  visomaster_teams_enhanced:
    enabled: true
    resolver_manifest_uri: "gs://training-job-outputs/cache/visomaster/enhanced_visomaster_resolver_2026-04-06.json"
    enhanced_bucket: "enhanced-visomaster-cropped"
    companion_domains: ["teams_v2"]
    include_statuses: ["teams_v2_companion"]
    anchor_indices: [0, 2, 4, 6, 8, 10, 12, 14]
    p_original: 0.5
    sampling_family_key: "visomaster_enhanced_fake"
```

And in the existing `family_weights:` block (line 256-265), add:

```yaml
      visomaster_enhanced_fake: 4.0    # NEW — same weight as base visomaster_fake.
                                        # No upweight (P14/P16 fw=8.0 and fw=2.0 both
                                        # tested with confounds; default weight is the
                                        # cleanest single-lever change vs E2B baseline.
                                        # This IS still a "more viso" change because
                                        # we're adding data without removing any.)
```

That's the entire delta. ~10 lines added, no other changes.

**Important: explicitly DO NOT add `visomaster_hints_teams.enabled: true`** — memory `project_visomaster_hints_lanes_bad_data` confirms this lane is harmful (face_parser bug residue). This block was already absent in E2B; just don't accidentally inherit it from P14/P16 templates.

## Validation tests before launch

Pre-launch checklist:
1. ✅ `combined_paired.py` wiring exists for both new sources (verified via grep)
2. ✅ GCS buckets exist (`visomaster-enhanced-face-cropped`, `enhanced-visomaster-cropped`)
3. ✅ Resolver manifest exists at the canonical URI
4. ✅ Existing wiring test present: `tests/test_visomaster_enhanced_wiring.py` — re-run before launch
5. ✅ `train_sweep.py:220` does `data_config['combined_paired'] = single_cfg['combined_paired']` — **wholesale dict assignment**, so the new sub-keys pass through with no allowlist needed (no flattening trap)

All wiring infrastructure is ready. Only the yaml diff is needed to launch.

## Why this is genuinely a single-lever change

Vs E2B baseline, this packet changes EXACTLY:
- Two new data sources enabled (both ENHANCED viso variants)
- One new family_weight entry (matching existing viso_fake weight)
- All architecture, optimizer, augmentation, sampling strategy, training schedule, eval — UNCHANGED

This is the cleanest possible test of "does enabling enhanced viso lift viso recall on a scratch B16."

If viso AUC on `visomaster_enhanced_macro_dev` increases by ≥0.05 (i.e., from E2B's 0.68 to ≥0.73), the hypothesis is confirmed and we have a path forward.

If viso AUC is unchanged, the data-availability hypothesis is refuted on this architecture.

If viso AUC drops, something else is wrong (e.g., the V1 enhanced bucket has its own distribution shift — unlikely but possible).

## Cost / time estimate

- Standard E2B-style training packet on a single A100: ~24h, ~$87
- Adding 2 new data sources adds I/O cost but not compute — same per-step time
- Eval scorecard (deployment-honest, all four suites under global τ + gate): ~$5
- Total: ~$92, ~25h

Fits well within 72h budget.

## What this packet does NOT test

- Whether the lift (if any) generalizes to teams-transported viso vs raw viso (will need per-subtype eval)
- Whether `visomaster_enhanced_v2` (the 16-swap-model bucket, not yet wired for training) would lift further — that's v2 work
- Whether HDTF visomaster (705 fresh identities, no wiring) would help — v3 work
- Whether the chronic-6 / webcam FPR problem improves (different problem; needs different lever)
