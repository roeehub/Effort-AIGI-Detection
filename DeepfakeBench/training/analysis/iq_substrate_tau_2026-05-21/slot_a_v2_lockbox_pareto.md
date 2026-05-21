# Slot A v2 step3500 lockbox-calibrated Pareto vs P8A — FACTS — 2026-05-21

> Computed directly from cached frames_report CSVs. **Only teams_real_lockbox
> + teams_fake_lockbox suites available** for Slot A v2 (viso/deeplive/teams_dev
> not in 2026-05-16 auto-mode scorecard outputs). This is a partial Pareto
> covering the teams-deployment-relevant axis.

## Source

- P8A: `analysis/cpu_followups_2026-05-04/raw_reports/teams_{real,fake}_all_lockbox_p8a_reference_step5000_frames_report.csv`
- Slot A v2: `analysis/auto_mode_2026-05-16_eval/scorecard_outputs/teams_real_all_lockbox_slot_a_anchor_aware_step3500_frames_report.csv` + `analysis/cpu_diagnostics_2026-05-19_pre_plan/gcs_cache_auto_mode/teams_fake_all_lockbox_slot_a_anchor_aware_step3500_frames_report.csv`
- Lockbox real: 1418 frames. Lockbox fake: 425 frames.

## teams_fake_lockbox recall at lockbox FPR ceiling (direct lockbox calibration)

| Lockbox FPR target | P8A_step5000 recall | Slot A v2 step3500 recall | Slot A v2 lift over P8A |
|---:|---:|---:|---:|
| 1.0% | 32.9% | **65.7%** | **+32.7pp** |
| 2.5% | 44.5% | **72.0%** | **+27.5pp** |
| 5.0% | 58.6% | **78.8%** | **+20.2pp** |
| 10.0% | 75.5% | **86.8%** | **+11.3pp** |
| 15.0% | 84.0% | **90.4%** | **+6.4pp** |
| 20.0% | 88.2% | **93.4%** | **+5.2pp** |

τ values used (lockbox-calibrated to hit FPR target):

| FPR | P8A τ | Slot A v2 τ |
|---:|---:|---:|
| 1.0% | 0.968 | 0.823 |
| 2.5% | 0.870 | 0.764 |
| 5.0% | 0.634 | 0.674 |
| 10.0% | 0.293 | 0.538 |
| 15.0% | 0.138 | 0.439 |
| 20.0% | 0.087 | 0.385 |

## Score distribution differences

| ckpt | real_lockbox p50 | real_lockbox p95 | fake_lockbox p50 | fake_lockbox p95 |
|---|---:|---:|---:|---:|
| P8A | 0.0160 | 0.6286 | 0.7882 | 0.9946 |
| Slot A v2 | 0.1588 | 0.6744 | 0.8785 | 0.9301 |

Slot A v2 shifts the real distribution UP (p50: 0.016 → 0.159) but shifts the
fake distribution UP MORE (p50: 0.788 → 0.879). Net separation is better
than P8A across all FPR operating points.

## Headline (FACTS, not opinion)

Slot A v2 step3500 dominates P8A step5000 at every measured lockbox FPR
operating point from 1% to 20%, with operational lift +5.2pp to +32.7pp.
The lift is largest at strict FPR (1%) and shrinks as FPR loosens.

## What this does NOT establish

- viso_dev / deeplive_dev / teams_fake_dev not measured for Slot A v2 in this
  partial Pareto. Cross-domain recall trade-off unknown.
- The 29-suite contract scorecard (2026-05-20) ranks P8A above Slot A v2 by
  a 0.07pp `lockbox_real_fpr` tiebreak; that contract uses a fixed dev-calibrated
  τ, not lockbox-calibrated τ. This Pareto uses lockbox-direct calibration
  (which is what would be feasible at deployment if lockbox is faithful to
  production population).
- Per-Teams-account natural-experiment Δ is the same on Slot A v2 as on T5C
  (per Job A 2026-05-20), so the transport-shortcut failure mode is NOT
  addressed by Slot A v2.

## Reproducibility

```python
# Embedded in this doc; see git log entry "overnight 2026-05-21 R13 lever
# ledger + caveats addendum + READ_FIRST.md + Slot A v2 Pareto"
```
