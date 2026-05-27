# Training Data Quality Audit — FINDINGS

**Job:** `TRAINING_DATA_QUALITY_AUDIT_2026-05-06`
**Date:** 2026-05-06
**Verdict:** **CONDITIONAL — defer.** Do NOT pull data-cleanup as a primary single-lever packet. PE_QUALITY_FLOOR is at most a smoke-test follow-up after P22's substrate-generalisation is verified.
**Confidence:** medium (resolution-asymmetry caveat materially affects the visomaster-direction finding; native-resolution sample audit would harden).

> Note: this file mirrors the sub-agent's report. The sub-agent wrote `summary.json`, `verdict.json`, `training_iq_distribution.csv` (+ `_all_lanes` version), `production_iq_distribution.csv`, `filter_impact_table.csv` (+ `_all_lanes`), `low_quality_score_distribution.csv`, and `run_probe.py`. The harness blocked direct `.md` writes; the parent agent mirrors the report content here.

## User question (verbatim)

> "I was wondering if perhaps our training data contains some data that is TOO Low quality in a way that really is not representative of deployment since even though deployment may encounter lower quality, We will be able to put some limit to it such that extremely low quality images will simply be rejected... if this is the case where truly the signal is to distorted, then perhaps we want to do some cleanup in our data to just not even train on that. Critically and factually (Historic analysis) Think about this approach."

## Headline answer

The hypothesis is **partially valid on DF40 lanes** but **refuted on the load-bearing visomaster lanes**. The DF40 long tail exists but DF40 is down-weighted (`fw=0.2`); the visomaster lanes (`fw=4.0`, the active load-bearer) are SHARPER than production, not softer.

The most successful prior intervention on viso (P22 augmentation curriculum) went in the **opposite** direction — making training SOFTER yielded 3× macro recall lift. A QUALITY_FLOOR packet that removes soft training samples is structurally inconsistent with the lever that has proven on viso.

## Numbers

### Filter impact at recommended threshold (`min_dim < 200` OR `lap_var_native < 15`, joint)

| Lane group | % below prod_p10 lap_var (49.6) | % below prod_p05 lap_var (15.4) |
|---|---:|---:|
| visomaster (v1 + enhanced + teams) | **5.8%** | **0.3%** |
| df40 + deeplive fakes | **47.9%** | **5.7%** |

Recipe-weighted estimate: **5-15% of training removed**, with the bulk concentrated in down-weighted (`fw=0.2`) df40. Visomaster (`fw=4.0`, the active load-bearer) sees almost no removal.

Active recipe rows audited: **310,340 across 18 lanes**. Production substrates audited: **9 (n=972 real frames)**.

### IQ-shortcut score evidence (FOR cleanup)

From the EVAL-side join `analysis/p8a_lockbox_join_2026-04-27.csv` (n=7,334): on dev real frames, the **bottom-5% lap_var samples have P8A median score 0.32 vs 0.01 for normal frames** — a 32× score inflation. The shortcut is alive in the score data.

### Visomaster direction-flip (AGAINST cleanup)

`train_viso_fake` lap_var p50 = **421** (224×224 cache).
`eval_viso_fake` lap_var p50 = **78** (~360px native).

Even after correcting for the resolution asymmetry (training is post-resize; lap_var is inflated at lower resolution), training viso is **at-or-above production sharpness**. Only 0.3% of viso training falls below prod_p05. The user's mental model — "training has a degraded long tail production never sees" — is REFUTED on viso.

The successful intervention (P22 augmentation curriculum, +3× macro recall on viso per `project_p22_succeeded_2026-05-02.md`) went in the **OPPOSITE** direction: it made training SOFTER, not removed soft training. PE_QUALITY_FLOOR going the other way is structurally inconsistent.

## Two strongest pieces of supporting evidence (FOR cleanup)

1. **The IQ shortcut is alive in the score data.** Bottom-5% lap_var EVAL samples score 32× higher on P8A than normal samples. Removing the analogous training tail could weaken the gradient signal that teaches the shortcut.
2. **Job 14 + P22 sibling-lever evidence.** Job 14 EVAL cleaning lifted P8A viso recall 27% → 67% (`project_job14_substrate_clean_2026-05-04.md`). P22 attacked the same IQ axis via augmentation (making training softer) and produced 3× macro recall. A training-side filter is in the same intervention family.

## Two strongest counter-considerations (AGAINST cleanup)

1. **Direction is wrong on the load-bearing visomaster lane** (see numbers above). The user's premise — "training has degraded samples production never sees" — is refuted on viso. The successful prior intervention (P22) goes in the OPPOSITE direction.
2. **Data-axis lever has failed twice.** P14_DATA_FIX (`xan4dfto`, value_composite=0.126) and P16_DATA_AXIS (`rmic6wrc`, no contract-calibrated viso recall lift above 1.1%). Memory `project_data_axis_lever_pulled_twice_no_lift.md` explicitly warns against another packet in this family without structural distinction. **PE_QUALITY_FLOOR's "removal" framing is weak distinction from P14/P16's "re-weighting" — re-weighting at fw=0 IS removal.**

## Recipe sketch — PE_QUALITY_FLOOR (smoke test only)

Only worth running AFTER P22 substrate-generalisation is verified, AND after the resolution-asymmetry caveat is resolved.

```yaml
data_filter_thresholds:
  min_dim_min: 200          # native-res face crop
  lap_var_min_native: 15    # native res, NOT post-224x224
  applies_to_lanes: [df40, deeplive, visomaster, visomaster_enhanced,
                     visomaster_teams_enhanced, real_pool]
# Everything else IDENTICAL to R13_VISO_CORR_PENALTY.yaml.
```

**Acceptance gate:** PROMOTE iff (a) viso_macro_recall@FPR=10% on dev rises ≥5pp vs P8A_step5000 baseline, AND (b) `teams_real_dor_dev` per-frame FPR stays ≤4% (preserves P8A's dor invariance signature).

**Estimated lift (low confidence):** 0 to +5pp viso recall, bounded above by P22's already-realised 3× lift.

## Critical caveats explicitly engaged

1. Job 14 was EVAL not TRAINING — analogy is suggestive, not proven.
2. IQ shortcut is a model behavior; removing low-IQ training data may not change encoder predisposition.
3. Production p10 anchor is approximate (n=972 across 9 substrates with high spread; dor_morning p10=11, dor_evening p10=78).
4. `project_quality_enhancement_routing_2026-05-05` mislabel bug is separate (correctness) and must be fixed first to avoid lift confounding.
5. Wholesale removal risks overfitting to a clean distribution; `external_youtube_avspeech` real pool has lap_var p10=67 — those soft real-pool samples teach codec robustness.
6. `feedback_per_mode_tau_not_deployable` — filter MUST use content properties (sharpness, resolution, color stats), not metadata-class labels. Recipe respects this.

## Resolution-asymmetry caveat (load-bearing for interpretation)

Visomaster training cache is at 224×224 (post-resize); production is at native ~219-373 px. lap_var measured at lower resolution is INFLATED. The "viso training is sharper than production" finding is qualitatively right but quantitatively soft — a fair native-resolution measurement would likely shrink the train-vs-eval gap from 5.4× to ~1-1.5×.

The DF40 finding (48% below prod_p10) is more robust because DF40 native resolution is also small (256×256 in `df40-frames-recropped-rfa85`); resolution correction would push DF40 lap_var DOWN further at native res.

## What would convert this from CONDITIONAL to YES

A targeted GCS scan over ~5,000 sampled training URIs across all 6 active lanes, computing IQ at NATIVE resolution (pre-resize), would resolve the resolution-asymmetry caveat. Cost: ~$5 GCS egress, CPU-only.

- If viso native lap_var p10 is ALSO above production: current finding holds; recommendation hardens to **NO**.
- If viso native lap_var p10 falls into production's tail: current finding is artifact; recommendation upgrades to **YES**.

## Outputs

All under `analysis/training_data_quality_audit_2026-05-06/`:

- `run_probe.py` — re-runnable, CPU-only, no GCS reads.
- `outputs/training_iq_distribution.csv` — per-active-lane IQ stats (18 lanes, 310k rows).
- `outputs/training_iq_distribution_all_lanes.csv` — all 60 lanes for context.
- `outputs/production_iq_distribution.csv` — per-substrate IQ stats (9 substrates).
- `outputs/filter_impact_table.csv` — per-lane filter fractions at T1/T2/T3.
- `outputs/filter_impact_table_all_lanes.csv` — all-lanes version.
- `outputs/low_quality_score_distribution.csv` — score stats for low-IQ EVAL samples.
- `outputs/summary.json` — top-level numerics + the load-bearing direction-flip finding.
- `outputs/verdict.json` — full structured recommendation.
- `outputs/FINDINGS.md` — this file.

## Cross-references

- Plan: `docs/relaunch_handoffs/NEXT_STEPS_PLAN_2026-05-06.md` §9 (add PE_QUALITY_FLOOR as deferred sub-component).
- Memory: `project_image_quality_shortcut.md`, `project_iq_gating_viability_2026-05-04.md`, `project_job14_substrate_clean_2026-05-04.md`, `project_p22_succeeded_2026-05-02.md`, `project_data_axis_lever_pulled_twice_no_lift.md`, `project_quality_enhancement_routing_2026-05-05.md`.
- Source data: `analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv` (manifest), Probe 1 IQ schema (`analysis/xinhe_cross_camera_audit_2026-05-06/outputs/axis_comparison.csv`), `analysis/p8a_lockbox_join_2026-04-27.csv` (eval-side score join).
