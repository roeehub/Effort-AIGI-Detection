# IQ-valley finding: the 364 unreachable v2 viso fakes are in an IQ valley between ckpt sweet spots

**Date authored**: 2026-05-05 (during PA+PC eval wait)
**Source**: re-analysis of `analysis/p8a_signature_decomposition_2026-05-05/viso_cohort_assignments.csv` + `per_cohort_iq_profile.csv` (existing prior CPU analysis)
**Status**: empirical observation; subject to revision based on PA+PC results

---

## Headline

The 364 v2 viso fakes that NO ckpt (P8A_step5000, E2B_top_n_step3200, E3_top_n_step6600) catches at FPR=10% live in an **IQ valley** — a Laplacian-variance range where each individual ckpt's IQ-shortcut signature predicts a low score:

| Cohort | n | Lap p50 | Catch pattern |
|---|---:|---:|---|
| A: caught by ALL 3 | 25 | **13.37** | LOW Lap (E2B's territory wins through) |
| B: P8A+E3 (not E2B) | 26 | 60.73 | MID-HIGH Lap |
| C: P8A only | 93 | **84.76** | HIGH Lap (P8A's sweet spot) |
| **D: missed by all 3** | **364** | **33.36** | **VALLEY** between E2B's and P8A's sweet spots |
| W: E2B+E3 (not P8A) | 4 | 15.71 | LOW Lap |
| X: E2B only | 13 | 14.89 | LOW Lap |
| Y: E3 only | 21 | 18.16 | LOW Lap |
| Z: P8A+E2B (not E3) | 4 | 148.83 | EXTREMELY HIGH Lap |

**Mann-Whitney U between caught_any (n=186) and uncaught_by_any (n=364)**:
- `laplacian_var`: caught median 60.32, uncaught median 33.36, ratio 1.81×, p=6.78e-5
- `luma_p10`: caught 50.59, uncaught 58.62, ratio 0.86×, p=2.54e-9 (uncaught are *brighter*)
- `sobel_edge_mean`: caught 29.81, uncaught 27.35, ratio 1.09×, p=1.03e-5
- `saturation_mean`, `skin_frac`: not significant (p>0.45)

## What this changes about the substrate framing

**Refines** the prior "v2 substrate is bound by IQ shortcut" framing in two ways:

1. **It's not "v2 is too blurry, so all ckpts fail"** — Cohort A (25 frames at Lap p50=13, even more blurry than D) is caught by ALL ckpts. The "low Lap = unreachable" naive read is REFUTED by Cohort A.

2. **Each ckpt has a different IQ sweet spot**. P8A catches HIGH Lap (Cohort C p50=85). E2B/E3 catch LOW Lap (Cohorts A/W/X/Y all p50≤18). The 364 missed-all are in the **Lap=20-50 valley** that no single ckpt's IQ-shortcut covers.

## Mechanism (interpretation, marked clearly)

The IQ shortcut isn't a single direction; it's a learned association that varies per ckpt:
- **P8A**: high Lap → high fake score (per `project_image_quality_shortcut`, viso r=+0.51). Catches sharp viso fakes.
- **E2B**: INVERTED at F1 tau (per `project_iq_gating_viability_2026-05-04`, Q1 low Lap = 52% recall). Catches blurry viso fakes.
- **E3**: similar inverted profile (Cohort A median Lap=13 all-caught includes E3).

The "valley" is the Lap range where each ckpt's learned IQ direction puts these frames OUT of its high-confidence-fake region:
- Too sharp for E2B's "blurry = fake" learning
- Too blurry for P8A's "sharp = fake" learning
- Falls in nobody's high-confidence-fake region → uncaught by any

## Direct implications for PA / PC predictions

**PA is FT-from-E2B**. If PA inherits E2B's IQ shortcut sign (inverted), then:
- PA's sweet spot for viso ≈ Lap 13-18 (Cohort A, W, X, Y range)
- PA likely MISSES the 364 valley frames (Lap ~33) — same failure mode as E2B
- PA's caught set ≈ E2B's caught set (intersect with maybe a few more)

**For PA to break the 364-frame ceiling, it would need to**:
- Either extend its IQ sweet spot UP (toward P8A territory, Lap 60+)
- Or extend its IQ sweet spot DOWN (which it's already in)

The visomaster_enhanced training data adds frames at unknown IQ levels; if those training frames spread across the IQ range, PA could in principle extend its sweet spot. But the prior P-series pattern (P14, P16) shows training-time exposure doesn't survive deployment τ.

**PC's codec aug** explicitly degrades IQ during training (codec compression + lower quality). This might shift PA→PC's IQ sweet spot toward LOWER Lap range, which is where E2B already is. PC would then have a NARROWER catch range, not broader. Counter-intuitive but consistent with the mechanism.

## Testable predictions for PA / PC

1. **PA F0 viso recall** ≈ E2B F0 viso recall ± 3pp (PA inherits E2B's IQ sweet spot; data lever doesn't extend it at deployment τ).
2. **PA's caught viso fakes** will overlap heavily with E2B's caught set (Cohorts A, W, X, Y). Cohort B and C frames remain uncaught.
3. **PC viso recall ≤ PA viso recall** (codec aug narrows the catch range further, not broadens).
4. **F4 substrate cleaning's lift on PA/PC** will be similar to E2B's (~22pp F0→F4 lift).

If observation contradicts (1)-(4): the IQ-valley framing is wrong and we need a different mechanism.

## Remaining uncertainty

- The 364 missed-all could ALSO have other shared properties (identity, swap-model artifacts) beyond Lap. Lap is the strongest single signal but not necessarily the only one.
- PA/PC scores haven't been collected yet — the prediction that PA inherits E2B's IQ profile is testable but not yet tested.
- The IQ-shortcut mechanism is ckpt-specific; new ckpts (PA/PC) might have novel IQ profiles that aren't easily characterized as "P8A-like" or "E2B-like."

## Implication for next-packet planning (if PA/PC verdict is (b))

If PA/PC don't break the IQ-valley structure, the next packet should target the IQ valley directly:
- Train with explicit Lap-stratified augmentation that PUSHES the model to be confident on mid-IQ frames
- Or: train an "IQ-valley specialist" head that's specifically calibrated for Lap 20-50 viso fakes
- Or: use ARGMAX-of-N routing where the router uses Lap as input (Job 12 oracle = 77% suggests this is feasible if router accuracy is high)

## Cross-references

- `analysis/p8a_signature_decomposition_2026-05-05/viso_cohort_assignments.csv` — source data
- `analysis/p8a_signature_decomposition_2026-05-05/per_cohort_iq_profile.csv` — per-cohort IQ stats
- Memory `project_image_quality_shortcut` — P8A's IQ shortcut direction
- Memory `project_iq_gating_viability_2026-05-04` — E2B's INVERTED IQ pattern
- Memory `project_job12_ensemble_ceiling_2026-05-04` — original 364 / 186 / 25 finding

## What this finding does NOT change

- The data-axis lever pattern (P14, P16, S3) is still the strongest predictor for PA's deployment-τ behavior.
- The substrate-specific framing from Job B is still valid (HDTF substrate sharpness is in P8A's sweet spot per memory).
- The chronic-6 FPR finding (Job 11/14) is orthogonal — it's about real-side, not the unreachable-fake-side.
