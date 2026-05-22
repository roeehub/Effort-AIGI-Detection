# CPU-1 Viso-Fake Signature Localization — FACTS (2026-05-23)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, ship, kill, best, worst, unfortunately, remarkably, lucky, confirmed, refuted, shortcut-aligned, gap-is-wide, gap-is-narrow.

## 1. Method

- Checkpoint: Slot A v2 step3500 (`periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth`).
- Forward hook on resblock[11] capturing the full 197-token sequence (CLS + 196 patches in 14x14 grid).
- Three readouts per frame computed by routing each pooled vector through the unchanged classifier head:
  - CLS-pool (1 token).
  - Face-pool: mean of centered 7x7 patch subgrid (49 patches).
  - Non-face-pool: mean of complement (147 patches).
- Per-patch ablation (saliency map): for each of 196 patch positions, replace that patch token with the mean of all 196 patches in the full-patch pool, recompute the head output, Delta = baseline - ablated. Aggregate to a 14x14 saliency map.
- N(viso) = 550 frames; N(deeplive control) = 545 frames.
- Wall time: 502s.

## 2. Per-readout recall table

Calibrated thresholds from the 2026-05-22 face-pool scorecard's `selected_threshold_scorecard.csv`:
- CLS-pool τ = 0.787956
- Face-pool τ = 0.736775

| Suite | CLS @ τ_CLS | Face-pool @ τ_face | Non-face @ τ_CLS | Non-face @ τ_face | Non-face @ τ=0.5 |
|---|---:|---:|---:|---:|---:|
| visomaster_enhanced_macro_dev (n=550) | 0.1836 | 0.0673 | 0.0000 | 0.0127 | 0.4218 |
| deeplive_enhanced_dev (control, n=545) | 0.7817 | 0.6844 | 0.0000 | 0.0514 | 0.9927 |

## 3. Saliency-map summary statistics

Saliency map values: mean(baseline_prob - ablated_prob) across frames per patch position. Higher absolute value = larger Delta = patch contributed more to the head's fake/real decision.

| Suite | mean(saliency) | std | max | min | face-region mass | non-face-region mass | face-frac | non-face-frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| visomaster | -0.000000 | 0.000389 | +0.002431 | -0.000650 | 0.0163 | 0.0430 | 0.2744 | 0.7256 |
| deeplive (control) | +0.000002 | 0.000689 | +0.005620 | -0.001003 | 0.0261 | 0.0599 | 0.3034 | 0.6966 |

## 4. Top-5 most-salient patches (by absolute Delta)

Patch coordinates as (row, col), with row 0 = top of frame, col 0 = left.

### visomaster
| rank | row | col | mean Delta | in-face-region? |
|---|---:|---:|---:|---|
| 1 | 2 | 5 | +0.002431 | False |
| 2 | 2 | 6 | +0.001258 | False |
| 3 | 2 | 1 | +0.000911 | False |
| 4 | 4 | 6 | +0.000803 | True |
| 5 | 8 | 1 | +0.000650 | False |

### deeplive (control)
| rank | row | col | mean Delta | in-face-region? |
|---|---:|---:|---:|---|
| 1 | 6 | 11 | +0.005620 | False |
| 2 | 4 | 2 | +0.003645 | False |
| 3 | 8 | 12 | +0.002976 | False |
| 4 | 8 | 13 | +0.001459 | False |
| 5 | 8 | 1 | -0.001003 | False |

## 5. Close criterion verdict

**Verdict: gamma**

Summary: in-between: cls_recall=0.184 face_recall=0.067 nonface_recall_at_tau_cls=0.000; face_saliency_frac=0.274 nonface_saliency_frac=0.726

Decision rule (verbatim from plan Phase 1):
- alpha: viso non-face-pool recall >= face-pool recall + 0.05 AND non-face saliency mass > 70%
- beta:  viso non-face-pool recall < CLS recall - 0.05 AND face saliency mass > 50%
- gamma: in-between

## 6. Output artifacts

- `viso_patch_saliency_14x14.npy` — 14x14 float64
- `deeplive_patch_saliency_14x14.npy` — 14x14 float64
- `viso_readout_probs.npy` — (n_viso, 4) float64: baseline, cls, face, nonface
- `deeplive_readout_probs.npy` — (n_deeplive, 4) float64
- `_cpu1_complete.json` — sentinel
