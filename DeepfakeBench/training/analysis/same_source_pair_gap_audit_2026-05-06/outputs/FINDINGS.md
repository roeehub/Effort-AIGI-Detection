# Same-Source Pair Gap Audit — FINDINGS (Phase 0g)

**Job:** `SAME_SOURCE_PAIR_GAP_AUDIT_2026-05-06`
**Date:** 2026-05-06
**Verdict:** **MIXED, leaning LANE_RESTRICTED_LAUNCH on FT-from-P8A.** §8.2's strict gate ("≥2 paired lanes GREEN") is NOT met from cache, but ZERO RED among measured lanes — the cross-product RED was eval-substrate-driven, not training-substrate-driven.

> Note: this file mirrors the sub-agent's report. The sub-agent wrote `summary.json`, `verdict.json`, `same_source_pairs.csv` (4000 rows, cross-product proxy with paired-lane mapping), `coverage_blueprint.md`, and `run_probe.py`. The harness blocked direct `.md` writes; the parent agent mirrors the report content here.

## Headline

Step-1 cache-first audit found **0 tight `(sample_id, frame_idx)` pairs** in any local score cache. Training-only buckets (df40, deeplive, visomaster_v1, visomaster_teams_enhanced, deeplive_teams) are not in any local cache; they're trainer inputs, not evaluation outputs.

Cross-product proxy mapped to paired-lane semantics covers **2 of 6 paired training lanes**. Verdicts on FT-from-P8A:

| Lane | n_pairs | n_missed | P(gap≤0 \| missed) | Verdict | Evidence |
|---|---:|---:|---:|---|---|
| visomaster_enhanced | 2,869 | 765 | **28.1%** | **GREEN** | proxy via `visomaster_v2_dor` fake_suite |
| deeplive | 847 | 68 | 23.5% | AMBER | proxy via `dor_fake_local` (n_missed small) |
| df40 | – | – | – | INSUFFICIENT_DATA | not in any cache; 4,698 base samples is largest unmeasured |
| visomaster_v1 | – | – | – | INSUFFICIENT_DATA | training-only |
| visomaster_teams_enhanced | – | – | – | INSUFFICIENT_DATA | training-only |
| deeplive_teams | – | – | – | INSUFFICIENT_DATA | training-only; family weight 7.0 in yaml (heaviest) |

E2B is RED on both measurable lanes (6.9% / 8.6%) — independently confirms FT-base = P8A (consistent with Agent 3 cohort diagnosis).
PA_3800 is GREEN (43.5%) on viso_enhanced but **informational only** — does not generalize to HDTF (memory `project_pa_does_not_generalize_to_hdtf_2026-05-05.md`).

## Aggregate verdict and reconciliation

**Aggregate: MIXED, leaning LANE_RESTRICTED_LAUNCH.** Below the §8.2 strict gate but with no RED among measured lanes.

**Reconciliation with cross-product RED (Agent 1):** the original aggregate RED (P8A 9.88%) was driven almost entirely by `teams_fake_all_dev` (2.8% inversion rate, n_missed=2,286). **`teams_fake_all_dev` is not one of the 6 paired training lanes** — it's an eval substrate where pair-rank loss never fires during training. Re-aggregating by training-lane (excluding eval-only substrates) flips the P8A verdict on the lanes the loss CAN fire on:
- viso_enhanced 28.1% — GREEN.
- deeplive 23.5% — AMBER.

This is consistent with Probe 4's bimodal cosine-distance finding on viso-enhanced (47% tight-invariant supports tight-pair survival; the catastrophic 11% tail is exactly what pair-rank would target).

**Per-subject heterogeneity reconciliation (Agent 1's per-subject table):**
- `dor_local` 27.6% sits between viso_enhanced (28.1% GREEN) and deeplive (23.5% AMBER). Consistent.
- `viso_v2_inswapper` raw 30-49% is the per-method shard within viso_enhanced; consistent with the 28.1% pooled lane figure.
- `teams_passthrough` 2.8% RED is on eval substrate not a paired training lane; does NOT block P1.

## Recommendation for `PE_PAIR_RANK_DRO`

User decision point (per `feedback_decision_points.md`):

### Option A — Authorize forward-pass tight-pair audit on df40 + deeplive

- Cost: **~$3 / ~40 min on us-east1** (or local A100 if available).
- Buys: 2/6 → 4/6 lane coverage (definitive verdict on df40 — the largest unmeasured lane at 4,698 base samples — plus deeplive direct measurement).
- Risk: minimal; postpones P1 launch by ~1 hour.
- Best $-per-bit option.

### Option B — Lane-restricted P1 launch on FT-from-P8A

- Cost: GPU-week training run (P1).
- Recipe: per Agent 2's coverage finding, **apply pair-rank uniformly across all 6 paired lanes**. The loss contributes 0 on RED-lane batches (harmless); fires non-trivially on viso_enhanced and deeplive.
- Run **P2 (SBI) in parallel** as the no-regret alternative.
- Risk: df40 (largest unmeasured lane, 4,698 base samples; closest to production substrate via deeplive_teams) carries unknown-direction risk. If df40's tight-pair data is RED, pair-rank loss is harmless but adds no value on most of training.

### My read

If time-constrained: **Option B** (P1 + P2 in parallel) — both are FT-from-P8A on the post-`2feea58` codepath; both target the three-pillar gap from different angles; the no-regret pair-rank-on-good-lanes-only application is structurally sound.

If not time-constrained: **Option A** then B (resolve df40 + deeplive first, then launch with confidence).

## Forward-pass blueprint (Option A details)

The sub-agent wrote `coverage_blueprint.md` with:
- Frame-list extraction from `data/sources/df40_paired.py` and `data/sources/deeplive_paired.py` loaders.
- Reuse `analysis/feature_space_2026-04-23/extract_features.py` template (or equivalent).
- Output schema: `analysis/forward_pass_pair_audit_2026-05-07/{p8a, e2b}_paired_scores.npz` with `(sample_id, frame_idx, label, score)` per row.
- Re-run `analysis/same_source_pair_gap_audit_2026-05-06/run_probe.py --tight-pairs <new_csv>` for definitive verdict.
- ~40 min wall-clock on a single A100; ~$3 at us-east1 spot.

## Outputs

- `same_source_pairs.csv` — 4,000 rows, cross-product proxy with paired-lane mapping.
- `summary.json` — programmatic per-lane stats.
- `verdict.json` — top-level decision: MIXED / LANE_RESTRICTED_LAUNCH viable.
- `coverage_blueprint.md` — exact plan for the forward-pass audit (Option A).
- `run_probe.py` — re-runnable; cache-first path active today, tight-pair path activates as soon as `--tight-pairs` CSV is provided.
- `FINDINGS.md` — this file.

## Cross-references

- Plan: `docs/relaunch_handoffs/NEXT_STEPS_PLAN_2026-05-06.md` §8.2 P1 (recipe spec — should reflect lane-restricted launch viability), §8.1 0g (this audit), §8.1 add 0j (forward-pass audit, Option A).
- Companion result: `analysis/pair_gap_audit_2026-05-06/outputs/FINDINGS.md` — cross-product RED reconciled here.
- Companion result: `analysis/pair_coverage_audit_2026-05-06/outputs/FINDINGS.md` — pair-rank-uniform recommendation that grounds Option B.
- Companion result: `analysis/checkpoint_cohort_diagnosis_2026-05-06/outputs/FINDINGS.md` — FT-base = P8A.
- Pair-loss premise refutation that does NOT apply: `project_pair_loss_premise_refuted_2026-05-04.md` (raw-vs-teams transport-invariance, not real-vs-fake same-source ranking).
- Memory: `feedback_decision_points.md` (user reserves judgment calls; present recommendation + tradeoffs and wait).
