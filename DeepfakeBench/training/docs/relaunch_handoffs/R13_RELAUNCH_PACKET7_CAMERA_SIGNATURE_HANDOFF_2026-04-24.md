# R13 Packet 7 — Camera/ISP-Signature Shortcut Handoff (2026-04-24)

## Context

RLP6_04 (leader, `value_composite=0.9006`) false-flags real participants on
some webcams. 2-camera controlled tests on 2026-04-24 showed:

| Pool | Score |
|---|---|
| Dor laptop, clean lighting, no VB | 0.02 |
| Dor webcam, same scene, no VB | 0.94 |
| Dor webcam, same scene, with VB | 0.88 |
| Roee Mac webcam, VB on | 0.90 |
| Roee Windows laptop | 0.01 |
| Dor laptop, yellow lighting (control) | 0.04 |

Same subject, same lighting → different camera flips the score. Cross-subject
replication (Roee Mac vs Windows) matches the pattern → **camera/ISP
signature shortcut**, not identity-specific.

Related finding (packet 6): slot-07 learned a dor_shkedi vs real_dor
processing-signature divergence on the same subject. Same class of problem.

## What shipped this session

### WS-P0 — Train/inference interpolation drift (prod bug)
Commit `855871e`. Changed `cv2.INTER_AREA → cv2.INTER_LINEAR` in
`batch_inference_gcs.py:407` and `arena/model_arena.py:472` to match
training (`combined_paired.py:3455`).

`arena/model_arena.py` was the bigger deal: that's the retro-score path
that produced the 0.94 Dor scores. Every retro-score number before this
commit had silent preprocessing drift.

Guarded by `tests/test_inference_train_preprocessing_parity.py` (grep-check
that every `cv2.resize(...interpolation=...)` in the inference files uses
INTER_LINEAR).

### WS-P1 — Per-camera calibration probe
Commit `deac44e`, output at
`analysis/calibration_probe_2026-04-24.summary.json`.

Result: per-pool τ closes **30.1%** of the cross-pool FPR gap (average
across target FPRs 5/10/15/25%). Boundary result — the plan's decision
gate was ≥60% → calibration is right lever, ≤30% → training-aug is right
lever. We're at 0.301, so both in parallel, but training-aug is the
larger remaining signal.

### WS-P2.b — Per-identity reducer
Commit `deac44e`. `arena/postprocess_per_identity.py` groups
`videos_report.csv` by `group_key` (identity) and recomputes
`real_fpr` / `fake_recall` per identity, with `--verify` that per-group
sums equal the aggregate. Covered by
`tests/test_postprocess_per_identity.py`.

Usage:
```
python arena/postprocess_per_identity.py \
    --reports <suite>_<ckpt>_videos_report.csv [more...] \
    --threshold <tau> \
    --out per_identity_<ckpt>.csv --verify
```

### WS-P4.a — Teams-passthrough spatial aug wiring
No code change needed. The `teams_passthrough_special_*` knobs already
exist in `_TEAMS_PASSTHROUGH_DEFAULTS` and `_build_teams_passthrough_pipeline`
(`data/augmentations/pipelines.py`). Enablement is a yaml choice.

### RLP7_04 + RLP7_05 yamls (commit `c4affa1`)
Two new Packet-7 candidates:

| Yaml | Delta vs RLP6_04 |
|---|---|
| `R13_RLP7_04_teams_spatial_only.yaml` | +ShiftScaleRotate on Teams (shift ±3%, scale ±5%, rotate ±3°, p=0.5) |
| `R13_RLP7_05_teams_spatial_plus_codec.yaml` | _04 + `teams_codec_sim_p=0.25` (quality [25,70]) |

Both verified end-to-end: the augmentation override keys survive the
preset-filter in `combined_paired.py:4772` and the constructed Teams
pipeline contains the expected transforms.

## Launch candidates (Packet 7)

Existing (drafted in the prior session):
- `R13_RLP7_01_lighting_aggressive.yaml` — context_variation CCT/gamma/shadow + codec_sim=0.15
- `R13_RLP7_02_codec_aggressive.yaml` — webcam_codec=0.35 + teams_codec_sim=0.40 + heavy jpeg/downscale
- `R13_RLP7_03_combined.yaml` — moderate mix of lighting + codec

New (this session):
- `R13_RLP7_04_teams_spatial_only.yaml` — **isolated spatial-aug test**
- `R13_RLP7_05_teams_spatial_plus_codec.yaml` — spatial + moderate codec

### Suggested launch subset
Given the calibration-probe verdict favoring training-aug, the highest-signal
minimal set is:

1. **RLP7_04 (spatial only)** — cleanest read on whether spatial-geometry
   shortcut is real. If _04 ≥ _02 on camera-signature stress metrics, we
   know spatial was the dominant axis.
2. **RLP7_05 (spatial + codec)** — complementarity test. If _05 > _04 and
   _05 > _02, the two axes combine.
3. **RLP7_02 (codec heavy)** — already the codec-only ceiling reference.

Skip RLP7_01 (lighting) and RLP7_03 (combined) unless budget permits —
the camera-signature controlled data doesn't point at lighting as the
dominant axis (yellow-vs-white clean control stayed near 0).

## Open workstreams (next session)

- **WS-P2.a** — Augmented-validation stress variants. Reuse the eval-stress
  presets at `pipelines.py:1702-1738` (crop_shift, scale, rotation,
  jpeg_q30, color_warm, color_cold) applied deterministically to the
  existing `teams_real_all_dev/lockbox` suites. Produces a 7-column
  per-suite FPR table for retro-scoring the new packet.
- **WS-P3** — Lockbox-scale fingerprint diagnostics. Run
  `analysis/fingerprint_diff.py` with `lockbox_csv_identity` source
  (hundreds of pools) and rank metrics by mean |Spearman ρ| across
  identities. Expensive (2–3 days GCS fetch).
- **WS-P4.b/c** — Additional aug primitives. Gated on WS-P3 ranking the
  top axes. Do not pre-build.

## Decision points for Roee

1. **Which subset to launch?** The suggested 3 (_02, _04, _05) vs the
   full 5 (01–05) vs something in between.
2. **RLP7 seed collisions** — _04 uses seed 744, _05 uses seed 745.
   Existing RLP7_01/02/03 use 740/742/743. Confirm these don't collide
   with currently-running jobs.
3. **Stress-matrix cadence** — worth doing WS-P2.a now so new checkpoints
   retro-score against a fixed yardstick, or launch-first-measure-later?

## Verification snapshot

Tests (all green): `pytest tests/test_inference_train_preprocessing_parity.py tests/test_postprocess_per_identity.py` → 5/5 pass.

Recent commits (teams-relaunch-root-2026-04-17 branch):
- `c4affa1` — Draft RLP7_04 + RLP7_05 yamls
- `deac44e` — Add calibration probe (WS-P1) + per-identity reducer (WS-P2.b)
- `855871e` — Fix train/inference preprocessing drift: INTER_AREA → INTER_LINEAR
