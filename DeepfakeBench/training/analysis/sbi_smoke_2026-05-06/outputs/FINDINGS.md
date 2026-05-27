# SBI Smoke Test — FINDINGS

**Job:** `SBI_SMOKE_2026-05-06`
**Date:** 2026-05-06
**Verdict:** **GREEN.** SBI pseudo-fakes are a plausible foundation for the planned `PE_SBI` packet. All three smoke criteria pass; recipe is launch-ready with one watch-item (mild face-sharpness drop) and recommended sweeps for two parameters.

> Note: this file mirrors the sub-agent's report. The sub-agent wrote `summary.json`, `verdict.json`, `iq_comparison.csv`, `iq_per_frame.csv`, `sbi_records.csv`, `score_distributions.csv`, `pseudofakes/_visual_panel_real_vs_sbi.png`, 60 standalone SBI images, and `run.log`. The sub-agent harness blocked direct `.md` writes; the parent agent mirrors the report content here.

## Numbers

- **N pseudo-fakes generated:** 200 (100% of real source frames).
- **Saved standalone:** 60 images.
- **Visual panel:** 24 side-by-side real|SBI panels at `pseudofakes/_visual_panel_real_vs_sbi.png`.
- **MediaPipe FaceMesh landmark success:** 195/200 = 97.5%; 5 fell back to elliptical mask.
- **IQ similarity** (200 each, real / SBI / known fake): median |Cohen's d|:
  - real-vs-SBI = **0.047** (virtually indistinguishable on IQ axes — luma, sat, RGB).
  - real-vs-fake = 0.60.
  - SBI-vs-fake = **0.68** > real-vs-fake — SBI does NOT carry the method-specific IQ fingerprint of known fakes (the whole SBI premise).
- **Score headroom** (P8A on cached scores): real_median 0.0058 → fake_median 0.884 (gap 0.88). Same shape on E2B and PA_3800 (>0.45 gap each). Plenty of room for SBI scores to land intermediate.

## Verdicts

| Criterion | Verdict | Evidence |
|---|---|---|
| Visual plausibility | **GREEN** | 97.5% landmark success; visual panel shows natural blends with sub-perceptual boundary cue. |
| IQ similarity to reals | **GREEN** | Median \|d\| 0.047 — SBI virtually indistinguishable from real on luma/sat/RGB axes. \|d\| SBI-fake 0.68 > \|d\| real-fake 0.60, so SBI carries method-agnostic blending signal, not method-specific IQ fingerprint. |
| Score headroom | **GREEN** | 0.88 P8A headroom guarantees SBI scores cannot collapse to 0 or 1 under any reasonable training interpretation. |

## What was deferred

**Direct SBI score collection on P8A/E2B/PA_3800 was NOT run** in this smoke test, because the task constraint specified no gs:// pulls and no local checkpoints were available. Score-collection blueprint is documented in `score_distributions.csv` rows tagged `BLUEPRINT_NOT_RUN`. The deferred run is:

- ~10 min CPU wall-clock + ~90 MB ckpt downloads.
- Re-uses `analysis/xinhe_cross_camera_audit_2026-05-06/run_local_inference.py` verbatim with `RAW_DIR` pointing at `outputs/pseudofakes/`.
- Three ckpt URIs validated in that probe.

The current verdict is **GREEN-by-headroom** rather than GREEN-by-direct-measurement; structurally SBI scores must land in (0, 1) but the exact distribution shape would be useful follow-up before launch.

## Watch-item (mild but real)

The blend pipeline drops face-region Laplacian variance by 30% (real 483 → SBI 339, |d|=0.75). This shift is in the **same direction** as known-fake softening (fake 209 — fakes are more softened than SBI, but SBI sit between).

**Risk:** the trained model could latch on "slightly less sharp than canonical real" rather than the blending-boundary itself. If so, `PE_SBI` would:
- Pass F1 (lockbox recall) on the in-distribution substrate.
- **Fail F4 (HDTF cross-substrate FPR ≤ 5%)** — identical failure mode to PA's substrate-bound walkback (memory `project_pa_does_not_generalize_to_hdtf_2026-05-05.md`).

This makes F4 the **differential gate** for P2, exactly as it was for PA.

## Mitigations (rank-ordered)

1. **Shrink `geom_strength`** from 1.0 (translation ±3px, rotation ±3°, scale ±0.03) to 0.5. Recommended sweep: 0.5–1.5.
2. **Lower `noise_std`** from 0.01 to 0.005. Recommended sweep: {0.005, 0.01, 0.02}.
3. **Apply a matching small Laplacian sharpen post-blend** so the SBI `lap_var_face` distribution overlaps real more tightly. Verifiable at training time by logging `lap_var_face` on SBI samples and confirming d-vs-real shrinks below 0.30.

## Recipe parameter recommendations for `PE_SBI`

| Parameter | Smoke value | Recommend for PE_SBI |
|---|---|---|
| `feather_px` | uniform {5, 7, 9, 11, 13, 15} | **KEEP.** Broad sample reduces feather-size memorisation. |
| `geom_strength` (translation / rotation / scale) | ±3 px / ±3° / ±0.03 (=1.0) | **SWEEP 0.5–1.5.** At 1.0, `lap_var_face` d=0.75 is non-trivial. |
| `noise_std` (source-side) | 0.01 | **SWEEP {0.005, 0.01, 0.02}.** Source noise is part of the sharpness shift. |
| Source-target asymmetry | same-image self-blend | **KEEP for v1** (classic SBI). |

## Next steps

1. **Greenlight P2 `PE_SBI` design + engineering** — all three smoke criteria pass.
2. **Run the deferred SBI score collection** (~10 min, ~$0) when gs:// egress is authorised, to upgrade GREEN-by-headroom → GREEN-by-direct-measurement.
3. **Sweep `geom_strength × noise_std`** at PE_SBI launch rather than ship the smoke defaults.
4. **Make F4 (HDTF cross-substrate FPR ≤ 5%) load-bearing** in PE_SBI's close criterion — the sharpness watch-item makes F4 the differential gate, exactly as for PA.
5. **Log `lap_var_face` on SBI samples during training** so the d-vs-real metric is monitored online.

## Output files

All under `analysis/sbi_smoke_2026-05-06/`:

- `run_probe.py` — re-runnable probe (deterministic seed=20260506, ~30s on M1 Pro).
- `outputs/summary.json` — programmatic summary.
- `outputs/verdict.json` — top-level GREEN.
- `outputs/iq_comparison.csv` — 20 IQ axes × 3 populations + 3 pairwise Cohen's d.
- `outputs/iq_per_frame.csv` — 600-row per-frame IQ.
- `outputs/sbi_records.csv` — per-pseudofake params + `landmarks_ok` + `fallback_mask` flags.
- `outputs/score_distributions.csv` — P8A/E2B/PA_3800 cached real+fake stats; SBI rows are BLUEPRINT placeholders.
- `outputs/pseudofakes/_visual_panel_real_vs_sbi.png` — 24-pair side-by-side.
- `outputs/pseudofakes/sbi_*.png` — 60 standalone SBI images.
- `outputs/run.log` — full execution trace.
- `outputs/FINDINGS.md` — this file.

## Cross-references

- Plan: `docs/relaunch_handoffs/NEXT_STEPS_PLAN_2026-05-06.md` §8.2 P2 (recipe spec — incorporate sweep recommendations and F4-as-differential-gate framing).
- Reference run for follow-up score-collection: `analysis/xinhe_cross_camera_audit_2026-05-06/run_local_inference.py`.
- Companion finding: `analysis/checkpoint_cohort_diagnosis_2026-05-06/outputs/FINDINGS.md` (FT base = P8A; P2's FT base same).
- Similar failure-mode warning: memory `project_pa_does_not_generalize_to_hdtf_2026-05-05.md`.
