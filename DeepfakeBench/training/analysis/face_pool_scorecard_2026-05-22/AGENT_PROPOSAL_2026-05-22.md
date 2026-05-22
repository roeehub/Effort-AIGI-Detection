# Face-Pool Full Scorecard Rerun — AGENT_PROPOSAL (2026-05-22)

Opinion-and-recommendation document. Facts live in RESULTS_FACTS_2026-05-22.md.

## Question

Does swapping the centered-7x7 face-region patch pool for the CLS pool — at inference time only, on a frozen Slot A v2 step3500 head — give a deployment-grade improvement over CLS pool on the same checkpoint?

## Headline numbers (lex policy)

| Metric | CLS pool | Face pool | Δ |
|---|---:|---:|---:|
| `dev_primary_real_fpr` | 0.0655 | 0.0636 | -0.0018 |
| `dev_fake_macro_recall` | 0.4381 | 0.4603 | +0.0222 |
| `lockbox_real_fpr` | 0.0191 | 0.0154 | -0.0037 |
| `lockbox_fake_recall` | 0.6877 | 0.7668 | +0.0791 |

## Recommendation

**SHIP-as-inference-variant** — lockbox_real_fpr non-regressing (Δ≤+0.5pp) AND lockbox_fake_recall up at least +5pp.

## Caveats

- This is a single-ckpt rerun on the same calibration substrate (2026-05-20 validation map). The classifier head was trained on CLS-pool features, so the absolute real-score distribution shifts upward under face-pool — the FPR-budgeted contract τ re-calibrates this away, but the calibration is now on shifted scale.
- The composite-policy result (λ=1.0) names the FP-to-FN cost ratio explicitly. If the composite winner differs from the lex winner, that is a 2026-05-22 policy-question, not a face-pool question.
- The 800-frame canary follow-up gave +6pp at FPR=10% and -4pp at FPR=5%; the per-chronic-identity Roy_D drop (-10pp) was the brightest single number. Full-suite extrapolation depends on whether the chronic subset is representative of the lockbox_real pool — RESULTS table 3a tests this directly.

## Next steps (regardless of recommendation)

- Run face-pool on the broader 4-ckpt panel (P8A, T5C step3500, Slot A v2 step1500, Slot A v2 step3500) to see whether the lift is a Slot-A-v2 idiosyncrasy or a substrate-invariance lever that transfers.
- Probe-side: confirm that the chronic-6 Roy_D drop reproduces in the full-suite lockbox pool — same identity, larger n.