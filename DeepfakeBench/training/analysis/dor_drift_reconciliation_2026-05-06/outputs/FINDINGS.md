# dor-drift R² Metric Reconciliation — FINDINGS

**Job:** `DOR_DRIFT_R2_RECONCILIATION_2026-05-06`
**Date:** 2026-05-06
**Verdict:** **No contradiction in existing docs.** Three different metrics answer three different questions on the same data. The team's "~90% recovered" cite is correct on the merits; the in-house second-opinion's flag was over-conservative.
**Confidence:** **HIGH** (all three numbers reproduced exactly from `analysis/dor_drift_mechanism_2026-05-06/outputs/per_frame_features.csv`).

> Note: this file was written by the parent agent because the sub-agent harness policy blocked direct `.md` writes from the dispatched probe. `metric_reconciliation.json`, `reconciliation_recompute.json`, `verification_log.md`, and `run_probe.py` (the verification script) are at this directory.

## What each of the three numbers actually measures

| Number | Source | Methodology | What it measures |
|---|---|---|---|
| **R²=0.142** (P8A) | `analysis/dor_drift_mechanism_2026-05-06/outputs/regression_p8a.json` | Ridge fit on all 819 dor real frames across 7 sessions, 14 IQ features, fit-and-score on the same set. | **Within-frame** score variance explained by IQ across the full dor real population. **NOT** cross-session drift. |
| **0.064/0.316 = 20%** (P8A) | `drift_attribution.csv` `__PREDICTED_DRIFT__ / __OBSERVED_DRIFT__` | Take the all-sessions ridge from above, project it onto the `(low_session_mean − high_session_mean)` IQ delta. | All-sessions-fit's prediction of the endpoint score gap. Underestimates because the global fit's coefficients are pulled toward within-session signal (fixed-effects-omitted regime). |
| **0.285/0.316 = 90%, R²_union = 0.42-0.62** (P8A 0.45, E2B 0.42, PA 0.62) | `supplementary_attribution_union.json` | Re-fit ridge on ONLY the 250 frames in (dor_evening + teams_real_dor_dev), 13 features, then project onto the endpoint difference. | Of the cross-session score gap, how much lives in the linear span of named IQ axes. **This is the right number to cite for "X% of drift is explained by named pixel-domain axes."** |

## The right statement to use

> On dor real frames, **~90% of the cross-session score drift between `dor_evening` and `teams_real_dor_dev`** (Δmean P8A = 0.316) is recoverable as a linear projection of named pixel-domain IQ axes (R²_union = 0.45 on P8A; 0.42–0.62 across {P8A, E2B, PA_3800}).
>
> Top single drivers on P8A by univariate Pearson r: **`min_dim`** (resolution, |r|=0.57), **`edge_mag`** (Sobel mean, |r|=0.45), **`color_b_dev`** (LAB B-channel cast, |r|=0.43).

## Caveat to attach

The union fit is calibrated *to* the endpoint difference, so 90% means "the drift lives in a low-dimensional named-IQ subspace," NOT "named IQ is the causal driver." The IQ axes covary with codec / camera / day-of-capture confounders that a 250-row 2-session ridge cannot disentangle.

**90% is necessary, not sufficient, for PD-class success.** A PD-class penalty that zeros correlations on the 13 named axes mechanistically removes ~90% of *this dor session pair's* drift in expectation, but other shortcut dimensions covarying with the named axes may take its place.

## Implication for PD and successor packets

PD currently targets only `sharpness_lap` + `luma_mean` + `face_area_fraction`.

The dor-drift dominant axes are `min_dim` (resolution) and `color_b_dev` (B-channel cast) — **both UN-TARGETED by PD**. These are the high-impact axes most likely to absorb the shifted gradient when PD's sharpness/luma penalties bite. The thread's predicted "shortcut shifting" symptom is structurally expected at these axes.

**Recommendation for PD-class successors (HSIC, AugMix-with-corr-pen, GroupDRO):** include `min_dim` and `color_b_dev` (or `color_b_cast`) in the penalty axes. The chroma-band Fourier probe (Agent 5 result) independently flagged chroma channels — particularly lab_b — as carrying clean shortcut at bands 6-8 with no signal-carrying preserve bands. The two findings converge: B-channel cast is the load-bearing under-targeted axis.

Note that `is_webcam` is NOT deployable as a penalty axis (Teams doesn't surface capture mode at inference; memory `feedback_per_mode_tau_not_deployable.md`), but `min_dim` and `color_b_dev` are both pixel-domain and inferrable from any input frame.

## Was anything in current docs wrong?

- **Thread `processing_signature_shortcut.md` line ~119 ("90% / R²_union 0.42-0.62"):** **correct, exactly matches `supplementary_attribution_union.json`.** Could be tightened with the calibrated-to-endpoint caveat but not factually wrong.
- **Plan `NEXT_STEPS_PLAN_2026-05-06.md` §3.6 (lines around 110-112) and open-question §9.0e:** correctly flagged as needing reconciliation; did not assert a wrong number itself. **Recommend closing §9 entry on this topic** with the union-fit 90% as headline; no contradiction exists, just three different questions on the same data.
- **In-house second-opinion's Round-2 "metric conflation" flag:** technically correct that the three numbers are different, but the implication that the team had made an error was over-conservative. The team's number was the right metric for the right question.

## Outputs

- `metric_reconciliation.json` — structured per-metric definitions.
- `reconciliation_recompute.json` — recomputed values, exact match to source.
- `verification_log.md` — what was re-run and what matched.
- `run_probe.py` — verification script (re-fits both ridges).
- `FINDINGS.md` — this file.

## Cross-references

- Plan: `docs/relaunch_handoffs/NEXT_STEPS_PLAN_2026-05-06.md` §3.6 (annotate to reflect three-metric structure), §9.0e (close).
- Source data: `analysis/dor_drift_mechanism_2026-05-06/outputs/{per_frame_features.csv, regression_p8a.json, drift_attribution.csv, supplementary_attribution_union.json}`.
- Original thread: `docs/packet_retrospectives/threads/processing_signature_shortcut.md` Probe 2 sub-section.
- Companion result: `analysis/chroma_band_fourier_2026-05-06/outputs/FINDINGS.md` — independently confirms chroma B-channel as load-bearing shortcut axis.
- PD packet retro: `docs/packet_retrospectives/packets/PD.md` — note that PD's penalty axes are `sharpness_lap` + `luma_mean` + `face_area_fraction`, and `min_dim` + `color_b_dev` are missing.
