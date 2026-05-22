# CPU-2 Pair-Loss Re-Verification on Slot A v2 — FACTS (2026-05-23)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, ship, kill, best, worst, unfortunately, remarkably, lucky, confirmed, refuted, shortcut-aligned, gap-is-wide, gap-is-narrow.

## 1. Method

- Checkpoint: Slot A v2 step3500 (`periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth`).
- Source: 275 paired (raw, teams) viso fake seq_ids from `analysis/pair_loss_effect_verification_2026-05-05/per_pair_analysis.csv`.
- Frame URIs resolved via `analysis/face_pool_scorecard_2026-05-22/_tmp/visomaster_enhanced_macro_dev_frames.csv` (275 raw + 275 teams = 550 frames).
- Scoring: CLS-pool head readout (apples-to-apples with E2B 2026-05-04 analysis).
- N pairs scored = 275.
- Wall time: 313s.

## 2. Headline table — Slot A v2 vs E2B vs P8A

| Model | mean(raw) | mean(teams) | Δ (teams−raw) | direction |
|---|---:|---:|---:|---|
| Slot A v2 step3500 (new) | 0.5029 | 0.5419 | +0.0390 | teams > raw |
| E2B step3200 (2026-05-04) | 0.0860 | 0.1717 | +0.0857 | teams > raw |
| P8A step5000 (2026-05-04) | 0.4437 | 0.2690 | -0.1747 | raw > teams |

Median(raw) = 0.6496, median(teams) = 0.5915.
Wilcoxon signed-rank (teams vs raw): statistic = 14893.0, p = 0.001986.
(E2B for reference: statistic = 14869.0, p = 0.001868.)

## 3. Per-pair cohort partition

target cohort = teams > τ AND raw ≤ τ (pair-loss would help)
wrong_way cohort = raw > τ AND teams ≤ τ (pair-loss would hurt)

| τ | target | wrong_way | both_caught | both_missed | target Δ | wrong_way Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0 | 0 | 275 | 0 | +nan | +nan |
| 0.10 | 25 | 0 | 250 | 0 | +0.1758 | +nan |
| 0.20 | 62 | 1 | 187 | 25 | +0.2358 | -0.1780 |
| 0.50 | 30 | 19 | 135 | 91 | +0.3573 | -0.3030 |

Comparison to E2B Q3_threshold_sensitivity_e2b_symmetric (raw_caught_teams_missed = target; teams_caught_raw_missed = wrong_way):

| τ | E2B target | E2B wrong_way |
|---:|---:|---:|
| 0.05 | 26 | 62 |
| 0.10 | 17 | 55 |
| 0.20 | 11 | 47 |
| 0.50 | 3 | 29 |

**Note on cohort label semantics:** the 2026-05-04 doc labelled `target` as `raw_caught_teams_missed` (because that paper's pair-loss premise was raw→teams alignment). In the current plan the pair-loss is teams→raw alignment, so `target` = `teams_caught_raw_missed` here. This re-verification reports BOTH cohorts. The decision rule applies to the asymmetry between them at τ=0.20.

## 4. Close criterion verdict

**Verdict: beta**

Summary: sign same direction as E2B on Slot A v2: mean(raw)=0.5029 < mean(teams)=0.5419 (Δ=+0.0390); pair-loss premise does not hold; pivot to GroupDRO substrate-balanced

Decision rule:
- alpha: sign reversed (mean(raw) > mean(teams)) AND target cohort > wrong_way cohort at τ=0.20 → pair-loss fulcrum exists → BACKBONE proceeds with substrate-pair-orthogonal loss
- beta: sign same direction as E2B (mean(raw) < mean(teams)) → pair-loss premise on Slot A v2 too → BACKBONE pivots to GroupDRO substrate-balanced
- gamma: signs cancel (|Δ| < 0.02) → indeterminate; BACKBONE runs with reduced λ_pair=0.15

## 5. Output artifacts

- `pair_scores_slot_a_v2.csv` — per-pair scores
- `RESULTS_FACTS_2026-05-23.md` — this file
