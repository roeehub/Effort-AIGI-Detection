# Chroma-Band Fourier Probe — FINDINGS

**Job:** `CHROMA_BAND_FOURIER_2026-05-06`
**Date:** 2026-05-06
**Verdict:** **CHROMA_DOMINATES_DIFFERENT_BANDS** — chroma extension is required for any Fourier-aug recipe.

> Note: this file was written by the parent agent because the sub-agent harness policy blocked direct `.md` writes from the dispatched probe. CSV outputs and `summary.json` written directly by the sub-agent are at this directory; this file mirrors the sub-agent's report content.

## Substrate

- 152 Xinhe frames (92 may6 / 60 may5) — shortcut task A.
- 200/200 fake/real `teams_*_dev` — manipulation-signal task B.
- All cached locally; no GCS pull.
- 6 channels (R, G, B, L, lab_a, lab_b) × 16 radial FFT bands × 5-fold CV logistic regression, balanced, n_jobs=1.

## Per-channel cleanest-cell bands

Cleanest cell defined per Probe 6 criterion: shortcut_AUC ≥ 0.85 AND signal_AUC < 0.65.

| Channel | Cleanest bands | Argmax shortcut | Signal-carrying (preserve) |
|---|---|---|---|
| R | 9, 10, 12, 13 | 13 (0.989) | 5, 6 |
| G | 9, 10, 12, 13 | 8 (0.977) | 5, 6 |
| B | 9, 10, 12, 13 | 8 (0.977) | 5, 6 |
| L | 9, 12, 13 | 9 (0.953) | 6 |
| **lab_a** | **1, 6, 7, 8, 9, 10** | 7 (0.935) | **none** |
| **lab_b** | **6, 7, 8** | 7 (0.921) | **none** |
| (grayscale ref, Probe 6) | 9, 10, 12, 13 | 9 | 5, 6 |

## Headline answers

### 1. Does the grayscale bands-12-13 cleanest-cell finding generalize to chroma?

**Partially.** R/G/B/L luminance channels reproduce grayscale almost exactly (cleanest at 9-13). Chroma a/b channels do NOT — shortcut peaks at mid-frequency bands 6-8 instead. Specifically:
- lab_a band 13 collapses to AUC 0.71 (vs grayscale band 13 = 0.97).
- lab_b band 13 collapses to AUC 0.62.
- **Grayscale-only band masks miss the chroma shortcut at bands 1, 6, 7.**

### 2. Different/additional cleanest cells on chroma?

**Yes.** lab_a adds bands 1, 6, 7, 8, 10; lab_b adds 6, 7. lab_a uniquely has 6 cleanest cells across the entire low-mid frequency axis. Both chroma channels have **ZERO signal-carrying bands** at any frequency (max signal AUC: lab_a=0.58, lab_b=0.70).

**Implication:** chroma is categorically more separable than luminance for this shortcut/signal partition. The shortcut maps cleanly onto chroma at bands the grayscale probe never measured.

### 3. Is `PE_FOURIER_BAND` viable per-channel?

**Yes, and chroma extension is required.** Recommended per-channel band masks:

| Channel | Randomize | Preserve |
|---|---|---|
| R / G / B / L | {8, 9, 10, 12, 13} | {5, 6} |
| lab_a | {1, 6, 7, 8, 9, 10} | (none required) |
| lab_b | {6, 7, 8} | (none required) |

**Implementation sketch:** BGR → LAB, per-channel FFT amplitude jitter on respective masks, LAB → BGR. ~2× FFT compute relative to grayscale, negligible GPU impact, fits as a training-time augmentation hook.

### 4. Does the may6 chroma-shortcut (sat_std, b_std) align with chroma bands?

**Yes, strongly.** Probe 1's top discriminators (sat_std d=−5.95, b_std d=−2.61) correspond directly to lab_a bands 1, 6-10 carrying the may6/may5 separation while the manipulation-signal task collapses to AUC ~0.45 across the same range. **Chroma-targeted randomization IS the right mechanism** — it hits the correct axis with negligible signal loss.

## Unexpected pattern

`lab_a` shows monotonically degrading shortcut AUC from band 7 (0.94) outward to band 13 (0.71) — the **OPPOSITE** of grayscale where shortcut increases toward high frequency. This means a single radial-band threshold cannot capture the chroma shortcut; it requires explicit per-channel band selection.

`lab_a` band 1 also flags as cleanest (AUC 0.90 / 0.58) — a low-frequency chroma component grayscale entirely misses (grayscale band 1 shortcut AUC 0.69, signal AUC 0.47, not in cleanest cell).

## Implication for `PE_FOURIER_BAND` (plan §8.2 P3)

The recipe sketched in plan §8.2 P3 ("same-label band-limited amplitude randomization on bands 12-13 + 8-9, preserving bands 5-6 and phase universally") was **grayscale-implicit**. With this probe's evidence:

- A grayscale-only implementation leaves the chroma shortcut at bands 1, 6, 7 intact and the ratchet will be smaller than budgeted.
- The correct implementation is per-channel with the masks in section 3 above.

**Updated recipe spec for `PE_FOURIER_BAND`:**
1. Convert input BGR → LAB (per-frame, in the augmentation transform).
2. Apply FFT-amplitude same-label randomization per-channel on the masks above.
3. Convert LAB → BGR.
4. Phase preserved universally on all channels.

## Caveats

- Univariate logistic AUC measures *separability* per band, not multivariate intervention effect. The model trained with the recipe will see combinations; the bands-12-13-clean finding doesn't strictly imply "removing those bands during training preserves the model's manipulation discrimination" — it bounds, but doesn't predict, the intervention effect.
- 152 frames is a small substrate for the shortcut task; per-fold std on the AUCs is reported in the per-channel CSVs.
- The teams_*_dev signal substrate has identity overlap with training (memory `project_visomaster_v2_dor` notes this caveat across all eval cells); absolute peak AUC values are likely inflated by leakage. The relative ordering shortcut-vs-signal per band is the meaningful read.

## File naming note

`lab_a`/`lab_b` rather than `a`/`b` because macOS HFS+ filesystem is case-insensitive and `per_band_aucs_b.csv` would collide with `per_band_aucs_B.csv`. CSV filenames use `lab_a`/`lab_b` prefix; channel labels inside CSVs/JSON match.

## Outputs

- `run_probe.py`
- `per_band_aucs_R.csv`, `per_band_aucs_G.csv`, `per_band_aucs_B.csv`
- `per_band_aucs_L.csv`, `per_band_aucs_lab_a.csv`, `per_band_aucs_lab_b.csv`
- `cleanest_cells_table.csv` (96 rows: channel × band × shortcut_auc × signal_auc × delta × cleanest_cell_flag)
- `summary.json` (per-channel AUC arrays + cross-channel comparison + grayscale ref + verdict)
- `FINDINGS.md` — this file.

## Cross-references

- Plan: `docs/relaunch_handoffs/NEXT_STEPS_PLAN_2026-05-06.md` §3.6 (grayscale Probe 6 caveat now resolved), §8.2 P3 (PE_FOURIER_BAND recipe — needs per-channel update), §9 (open chroma extension question now closed).
- Original Probe 6: `analysis/fourier_band_overlap_2026-05-06/run_probe.py`.
- May6 chroma-loaded shortcut: Probe 1, `analysis/xinhe_cross_camera_audit_2026-05-06/outputs/axis_comparison.csv` (sat_std d=−5.95, b_std d=−2.61).
