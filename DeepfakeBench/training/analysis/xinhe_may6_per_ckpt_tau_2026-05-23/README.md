# Xinhe-may6 re-evaluated at per-ckpt-calibrated τ

Quick CPU follow-up to `analysis/xinhe_may6_t5c_revisit_2026-05-23/` (which used cross-ckpt-const τ) and `analysis/per_ckpt_tau_recal_2026-05-23/` (which provided per-ckpt-calibrated τ for the team-identity bar).

## Method

Loaded existing per-frame scores from `analysis/xinhe_may6_t5c_revisit_2026-05-23/outputs/wide_scores.csv` (5 ckpts × 152 frames; 60 may5_correct + 92 may6_falseflag). Computed FPR at each ckpt's per-ckpt-calibrated τ from `analysis/per_ckpt_tau_recal_2026-05-23/outputs/per_ckpt_summary.csv` (user_bar best_tau column).

## Results at per-ckpt-calibrated τ

| Ckpt | per-ckpt τ | may5 FPR (n=60) | **may6 FPR (n=92)** | mean may6 score |
|---|---:|---:|---:|---:|
| **P8A** | **0.590** | **0.0%** | **0.0%** | 0.021 |
| E2B | 0.718 | 0.0% | **35.9%** | 0.511 |
| T5C | 0.790 | 0.0% | 3.3% | 0.264 |
| SlotAv2_CLS | 0.562 | 0.0% | 1.1% | 0.153 |
| SlotAv2_FACE | 0.681 | 0.0% | **15.2%** | 0.599 |

## Comparison vs cross-ckpt-const τ (for context)

The original `xinhe_may6_t5c_revisit` readout used τ=0.5 / mode A 0.535 / mode B 0.78 / mode C 0.87 across all ckpts. Per-ckpt re-cal mostly tightens or matches:

| Ckpt | cross-τ=0.5 FPR | cross-mode B FPR | **per-ckpt τ FPR** |
|---|---:|---:|---:|
| P8A | 0.0% | 0.0% | 0.0% |
| E2B | 57.6% | 31.5% | 35.9% |
| T5C | 17.4% | 3.3% | 3.3% |
| SlotAv2_CLS | 4.3% | 1.1% | 1.1% |
| SlotAv2_FACE | **91.3%** | 0.0% | **15.2%** |

Key observations:
- **P8A is bulletproof on Xinhe-may6** at its per-ckpt-cal τ (0.0% — matches all prior columns).
- **SlotAv2_FACE: the 91.3% cross-const-τ may6 catastrophe was largely calibration artifact**, but the 15.2% per-ckpt-cal FPR is still 3× over a 5% floor. Face-pool is genuinely fragile on Xinhe-may6 even after calibration.
- **E2B's may6 problem is structural and persists across τ.** At ANY of (cross 0.5, mode B 0.78, per-ckpt 0.72) it's above 30% may6 FPR.
- **T5C and SlotAv2_CLS are clean on may6 at per-ckpt τ**, even though they fail the team-identity bar.

## Implication for production choice

P8A at τ=0.59 is the only ckpt that passes BOTH:
1. The team-identity bar (per-ckpt τ-recal Task #1)
2. The Xinhe-may6 false-flag check (this readout)

This is the strongest support yet for the T5C → P8A production switch recommendation.

## Artifact

- `outputs/per_ckpt_tau_xinhe_may6.csv` — per-(ckpt × population) FPR at per-ckpt-calibrated τ
