# Checkpoint Cohort Diagnosis — FINDINGS

**Job:** `CURRENT_CHECKPOINT_COHORT_DIAGNOSIS_2026-05-06`
**Date:** 2026-05-06
**Verdict:** **FT base for `PE_PAIR_RANK_DRO` = P8A.**
**Confidence:** **HIGH.**

> Note: this file was written by the parent agent because the sub-agent harness policy blocked direct `.md` writes from the dispatched probe. CSV outputs, `summary.json`, `ft_base_recommendation.json`, `paired_frames_with_outcomes.csv`, and `run_probe.py` are at this directory; this file mirrors the sub-agent's report.

## Substrate

- 14,626-frame manifest at `analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv`.
- 10,289 frames (70%) fall in 9 person-clusters with both reals and fakes (pairing key derived in `run_probe.py::_person_cluster`).
- All paired classifications computed at τ ∈ {0.3, 0.5, 0.7} for robustness; τ=0.5 is the headline.

## Aggregate (n=10,289 paired frames at τ=0.5)

| Ckpt | real_OK_fake_OK | real_OK_fake_missed | real_FP_fake_OK | **real_FP_fake_missed** | recall | FPR |
|---|---:|---:|---:|---:|---:|---:|
| P8A | 81.1% | 10.9% | 8.0% | **0.0%** | 80.8% | 9.8% |
| E2B | 63.6% | 9.8% | 23.4% | **3.2%** | 77.2% | 7.2% |
| PA_3800 | 36.5% | 6.8% | 30.4% | **26.3%** | 55.0% | 12.3% |

The `real_FP_fake_missed` cell is **the exact target** of pair-rank loss (paired sample where the model gets BOTH wrong and the fake doesn't outrank its paired real). P8A is at 0% at τ=0.5; E2B at 3.2%; PA at 26.3% (catastrophic).

τ-sweep robustness for the worst cell on P8A: 4.5% / 0.0% / 0.0% at τ ∈ {0.3, 0.5, 0.7}.

## Cohort-by-cohort verdict (P8A vs E2B fail rates)

### P8A wins (decisive, production-relevant)

| Cohort | n | P8A fail | E2B fail | Δ |
|---|---:|---:|---:|---:|
| `live_fakes_teams_prod` | 1,675 | 0.247 | 0.928 | **−0.68** |
| `xinhe_may6_falseflag` | 92 | 0.000 | 0.576 | −0.58 (confirms `project_xinhe_may6_falseflag_2026-05-06.md`) |
| `xinhe` cluster | 1,614 | 0.280 | 0.928 | −0.65 |
| `xiang` cluster | 1,429 | 0.064 | 0.474 | −0.41 |
| `transport=teams_live` | 1,767 | 0.234 | 0.910 | −0.68 |
| `visomaster_v2_dor` suite | 2,073 | 0.262 | 0.440 | −0.18 |
| `enhancer=gpen/gfpgan/codeformer` | 763 | (P8A favoured by 17-27pp) | | |
| `method=ghostface` | 396 | 0.16 | 0.43 | −0.27 |
| `method=simswap` | 79 | 0.27 | 0.89 | −0.62 |

### E2B wins (narrow, real-FPR-dominated)

| Cohort | n | P8A fail | E2B fail | Δ | Disposition |
|---|---:|---:|---:|---:|---|
| `pc_generator` cluster | 1,149 | 0.520 | 0.344 | +0.18 | Chronic-FP identity (memory `project_chronic_offenders_partition_per_ckpt_2026-05-04.md`); covered by GroupDRO `chronic_flag`. |
| `test_cam__s73` cluster | 352 | 0.361 | 0.000 | +0.36 | Single-cluster outlier; covered by GroupDRO `source`. |
| `face_size=far` | 1,452 | 0.424 | 0.289 | +0.13 | P8A more aggressive on small faces. |
| `quality=lo-q` | 2,552 | 0.305 | 0.227 | +0.08 | Quality-band cohort; covered by GroupDRO `quality_band`. |
| `teams_real_all_dev` (real only) | 2,059 | 0.157 | 0.074 | +0.08 | Pure real-FPR cohort, no fake-recall axis at all. |

**All E2B wins are real-FPR-dominated** (none has fake-recall headroom unique to E2B). The proposed `PE_PAIR_RANK_DRO` multi-axis GroupDRO real-side key `source × transport × quality_band × chronic_flag` directly addresses the chronic-flag cohorts (`pc_generator`) and source-outlier cohorts (`test_cam__s73`), reclaiming E2B's narrow wins on the real side.

### PA_3800 — ruled out

26.3% in `real_FP_fake_missed` (worst cell). Consistent with `project_pa_does_not_generalize_to_hdtf_2026-05-05.md`.

## Headroom for `PE_PAIR_RANK_DRO` on P8A

These are the cohorts where P8A still has fake-recall failures and pair-rank has structural leverage (paired-rich substrates):

- `gpen` enhancer: 46.7% missed
- `inswapper` method: 33.1% missed
- `dor_evening_morning` cluster: 17.0% missed
- `live_fakes_teams_prod`: 24.7% missed
- `visomaster_v2_dor`: 26.2% missed

These are the populations where lifting fake-recall while preserving substrate-invariance is the load-bearing improvement.

## Recommendation contingencies

1. **P8A training-date verification (plan §0c)** — RESOLVED 2026-05-06: P8A is **PRE-FIX** with HIGH confidence. The post-`2feea58` codepath has unmeasured headroom for the in-proj-SVD lever. Prefer the post-fix re-trained baseline for the new packet (with cheap grad-audit at step 100/500/1000 to verify the lever is live under the pair-rank+DRO loss).
2. **Multi-axis GroupDRO real-side key** must include `chronic_flag` (covers `pc_generator`) and `source` (covers `test_cam__s73`) to claw back E2B's narrow advantages.

## Outputs

- `cohort_by_method.csv`, `cohort_by_enhancer.csv`, `cohort_by_transport.csv`, `cohort_by_identity.csv` (231 rows), `cohort_by_quality_band.csv`, `cohort_by_face_size_band.csv`, `cohort_by_suite.csv`, `cohort_by_pair_gap_band.csv`, `cohort_by_is_lockbox.csv`, `cohort_by_person_cluster.csv`
- `summary.json` — per-ckpt aggregate cells across τ-sweep.
- `ft_base_recommendation.json` — programmatic per-axis verdict.
- `paired_frames_with_outcomes.csv` — full enriched manifest (10,289 paired frames × outcome and joint-cell columns × 3 ckpts × 3 τ values).
- `run_probe.py` — re-runnable diagnostic (CPU-only, n_jobs=1, no model loaded).
- `FINDINGS.md` — this file.

## Cross-references

- Plan: `docs/relaunch_handoffs/NEXT_STEPS_PLAN_2026-05-06.md` §8.2 P1 (FT base now P8A explicit), §6.5 (advisor argument validated).
- Companion result: `analysis/p8a_training_date_check_2026-05-06/outputs/FINDINGS.md` (P8A pre-fix verdict).
- Pair-gap data input: pending — `analysis/pair_gap_audit_2026-05-06/outputs/` (Agent 1 still running).
- Memory cross-refs: `project_xinhe_may6_falseflag_2026-05-06.md`, `project_chronic_offenders_partition_per_ckpt_2026-05-04.md`, `project_pa_does_not_generalize_to_hdtf_2026-05-05.md`, `project_in_proj_svd_gradient_bug.md`.
