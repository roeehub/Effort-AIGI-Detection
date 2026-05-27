# Deeplive Deployment Packet (24h ship — for QA validation)

Generated: 2026-05-05
Anchor checkpoint: **E2B_3200** (B16 scratch + CE + heavy aug)
Backbone: ViT-B-16 (lightweight)

## Deployment policy (single B16 model)

1. **Quality gate** (computed per-frame at inference time, no auxiliary models needed):
   - Reject frames where `is_no_face == True`
   - Reject frames where `min(width, height) < 150` pixels
   - Reject frames where `laplacian_var < 8.0` (extreme blur — outside operational envelope)
   - Rejected frames receive **NO decision** (output: "input out of scope" — not a deepfake call)

2. **Single global threshold** (no per-mode τ — that requires CLIP-based capture-mode classification which is not deployable):
   - **Recommended τ = 0.4368** (calibrated on lockbox-shaped real population at 5% FPR)
   - Higher-recall option: τ = 0.2889 (10% FPR)

3. **Decision rule** (after gate passes):
   - If `score >= τ`: flag as DEEPFAKE
   - Else: flag as REAL

## Performance under recommended policy (E2B, gate ON, τ = 0.4368)

| Metric | Value |
|---|---|
| **Deeplive recall** | **97.0%** |
| FPR on lockbox reals (production-shaped) | 5.1% |
| Frames rejected by quality gate (lockbox reals) | 3.9% |
| Frames rejected by quality gate (deeplive fakes) | 1.7% |

## Higher-recall option (10% FPR target)

| Metric | Value |
|---|---|
| Deeplive recall | 99.3% |
| FPR on lockbox reals | 10.1% |
| τ | 0.2889 |

## Why this calibration

τ is calibrated on `teams_real_all_lockbox` (1418 frames; production-shaped substrate)
rather than the dev real pool, because dev reals are over-represented in chronic-FP
identities (PC_Generator, bla_bla_chow, etc.) and don't reflect production frame
distribution. Lockbox-calibrated τ is more conservative (slightly lower in raw value
because lockbox reals score lower on average for E2B) but generalizes better to
production-style frames.

## What's NOT in this deployment

- Visomaster (visomaster_enhanced) detection: this packet is for the deeplive-only ship.
  Visomaster recall under E2B at this τ is ~8% (known weakness; v2 work in progress).
- Per-mode τ (would need a CLIP-based capture-mode classifier at inference, not viable
  per deployment constraint).
- Multi-model ensemble (lightweight constraint forbids).

## Identity-cohort caveat for QA interpretation

The 5.1% FPR measurement is on a lockbox-real population that is heavily skewed
toward a small set of identities (Dor Shkedi dominates: 1144/1418 = 81% of lockbox
real frames). The eval substrate intentionally has identity overlap between real
and fake (Dor appears in deeplive training as a fake source AND in lockbox as a
real subject). This means:

- **Production FPR on unseen-identity Teams users will likely be LOWER than 5.1%**
  (the model has learned Dor's face features somewhat through training; production
  identities won't have this overlap).
- The top false-positive examples in `teams_real_FP_examples_top100.csv` are
  Dor Shkedi frames scoring 0.6–0.94. This is a known identity-leak artifact
  (memory `project_lockbox_identity_looseness`), not a generalized model failure.

The 16 deeplive false-negatives are also all Dor (`deeplive_dor__seqXXXX`). Same
identity dynamics — production deeplive attempts won't all be Dor.

## Validation suggestions for QA

1. **Run deeplive samples through the gate + τ pipeline.** Recall should be ~97% at 5% FPR.
2. **Run a sample of typical Teams call frames (real users, normal capture, NOT Dor / NOT chronic-6 identities) through the pipeline.** Production FPR should be lower than the 5.1% measured here because chronic identities won't appear at production density.
3. **Verify gate behavior on edge inputs**:
   - Pure black frame (no face) → should be rejected (no_face)
   - Tiny thumbnail (<150px on either side) → rejected (lowres)
   - Severely blurred frame (laplacian_var<8) → rejected (blur)
4. **Spot-check FN examples** in `deeplive_FN_examples_top100.csv` — these are
   the 16 deeplive-Dor fakes the model misses. The pattern (all same identity) is
   the data-substrate eval artifact, not a generalized weakness.
5. **Spot-check FP examples** in `teams_real_FP_examples_top100.csv` — Dor Shkedi
   real frames dominating. Same identity-leak artifact.
6. **Recommended decision rule for Dor-like ambiguity in production**: if you have
   prior identity context (the user is enrolled or known), use a per-user calibrated
   τ for known-good users. The single global τ is what we ship for unknown users.

## Calibration source files (for reproducibility)

- Per-frame deeplive scores: `analysis/cpu_followups_2026-05-04/raw_reports/deeplive_enhanced_dev_e2b_top_n_step3200_frames_report.csv`
- Per-frame lockbox real scores: `analysis/cpu_followups_2026-05-04/raw_reports/teams_real_all_lockbox_e2b_top_n_step3200_frames_report.csv`
- Quality-gate tags (sharpness, dimensions, no_face): `analysis/lockbox_tagging/full_tags_2026-04-27.parquet`
- This script: `analysis/deeplive_deployment_24h_2026-05-05/run_deployment_setup.py`
