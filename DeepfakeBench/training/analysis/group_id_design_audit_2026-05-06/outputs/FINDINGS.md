# GroupDRO Group-ID Design Audit — FINDINGS

**Job:** `GROUP_ID_DESIGN_AUDIT_2026-05-06`
**Date:** 2026-05-06
**Verdict:** **Recommended pair = (F-B, R-D).** Passes all three DRO-stability thresholds (min group ≥50, spread ≥15pp, max-share ≤30%).

> Note: this file was written by the parent agent because the sub-agent harness policy blocked direct `.md` writes. Sub-agent wrote `candidate_groups.csv`, `pairwise_evaluation.csv`, `chronic_flag_definition.json`, `quality_band_thresholds.json`, `recommendation.json`, and `group_id_python_snippet.py` directly.

## Recommendation

```
Fake-side (F-B): label=fake | method_family | enhancer_family
Real-side (R-D): label=real | source | transport | quality_band | chronic_flag
```

Total groups: **27**. Min group: **68**. Median group: **381**. Max-group share: **0.184** (≤30% rule pass). Avg per-ckpt failure-rate spread: **0.915** (≥15pp rule pass). Composite score: **0.878**.

## Why F-B over F-A / F-C / F-D / F-E

- **F-A** (`label=fake | method`) is baseline; insufficient spread.
- **F-B** (add `enhancer_family`) lifts avg spread from 0.68 → 0.78 (gpen 47% missed vs none 9% on P8A). Wins on stats.
- **F-C** (add `transport`) adds only ~1pp spread but pushes smallest group below the 50-frame floor.
- **F-D / F-E** (add `quality_band`) silently degenerates because **67% of fake rows have `quality=unknown`** — quality_band collapses to a synonym for "is_visomaster_or_external."

The advisor's full fake-side proposal (`method × enhancer × transport × quality_band`) was DEMOTED to F-B because the data doesn't support those extra dimensions. Net win on spread is small; net cost on group fragmentation is large.

## Why R-D over R-A / R-B / R-C

- **R-A** (`label=real | source`) violates max-share rule: `teams_real_dev` source = 56% of reals.
- **R-B** (add `transport`) still has max-share above 30%.
- **R-C** (add `chronic_flag` to R-B) brings max-share to 0.43 — still above 0.30.
- **R-D** (add `quality_band`) splits the dominant group to **0.18 max-share**. Wins.

**The `chronic_flag` axis is the load-bearing real-side lever.** It lifts real-side avg ckpt spread from 0.11 (R-A) to 0.61 (R-C). Without it, GroupDRO has nothing to do on the real side.

## Chronic-6 list (definitive)

From `memory/project_chronic_offenders_partition_per_ckpt_2026-05-04.md` plus manifest verification:

- `bla_bla_chow`
- `bla_bla_chow__s2`
- `PC_Generator__s22`
- `PC_Generator__s45`
- `roy_d`
- `Q__s6`

Match rule: exact-or-substring on `base_identity`.

**1,297 real rows match. 0 fake rows match.** All chronic-flagged identities are real-side; fake-side chronic_flag would be uniformly False, so it's correctly omitted from F-B.

## Train/eval stability caveat

The `is_lockbox` proxy is structurally weak: lockbox covers only `teams_real_all_lockbox` + `teams_fake_all_lockbox` (both `teams_capture` transport). Cosine similarity between train and eval group distributions caps around 0.10 across **all** candidate schemes — this is a substrate-disjointness fact, not a grouping flaw.

R-D's `eval_train_recall = 0.33` means the eval slice exercises 5 of 15 real-side groups, **including the load-bearing chronic-flag groups** (`PC_Generator__s22` is in `teams_real_all_lockbox`).

## External lane handling

`external_vcd_real` (1,200 unpaired reals, ~80 identities) should set `is_unpaired_real=True` (per Agent 2 PAIR_COVERAGE_AUDIT) to suppress pair-rank but still contribute to GroupDRO via its own real group: `real | external_vcd_real | webcam_codec | unknown | regular`. Codified in the python snippet.

## Quality_band thresholds (when usable)

When `quality` is known (33% of fakes, near-100% of reals), the chosen ternary cutoff is in `quality_band_thresholds.json`. When unknown, default to `unknown` band. R-D uses `quality_band` because it has signal on the real side; F-B excludes it from the fake side because it silently degenerates.

## Implementation notes for `trainer/mixins/group_dro.py`

The current implementation uses `method_mapping` (str→int) and reads `method_id` from `data_dict`. Replace with:

```python
# At config-load time, walk the manifest once and build:
group_id_mapping = build_group_id_mapping(manifest_df)
# Returns dict: group_str_key -> int_idx

# At forward time, data_dict carries:
data_dict["group_id"] = LongTensor of shape (B,)
data_dict["is_unpaired_real"] = BoolTensor of shape (B,)  # for skip rule
```

- ~27 groups means EMA buffer is tiny.
- Default `ema_alpha=0.1` and `beta=3.0`.
- Suggest **100-step EMA warmup** before applying group-weighted loss.

## Outputs

- `candidate_groups.csv` — 9 single-side schemes (F-A through F-E and R-A through R-D) with stats.
- `pairwise_evaluation.csv` — 20 (F, R) pairs ranked by composite.
- `chronic_flag_definition.json` — explicit chronic-6 list with provenance.
- `quality_band_thresholds.json` — chosen quality_band cutoffs.
- `recommendation.json` — headline + top-3 pairs.
- `group_id_python_snippet.py` — ready-to-paste; includes `is_chronic`, `quality_band`, `make_group_id`, `build_group_id_mapping`.
- `run_probe.py` (one level up) — CPU-only, ~10s, re-runnable.
- `FINDINGS.md` — this file.

## Cross-references

- Plan: `docs/relaunch_handoffs/NEXT_STEPS_PLAN_2026-05-06.md` §8.2 P1 (DRO recipe), §6.4 (advisor's multi-axis grouping argument).
- Companion result: `analysis/checkpoint_cohort_diagnosis_2026-05-06/outputs/FINDINGS.md` (Agent 3, identifies which cohorts the GroupDRO key needs to separate).
- Companion result: `analysis/pair_coverage_audit_2026-05-06/outputs/FINDINGS.md` (Agent 2, identifies `external_vcd_real` skip rule).
- Memory: `project_chronic_offenders_partition_per_ckpt_2026-05-04.md` (chronic-6 origin).
- Code: `trainer/mixins/group_dro.py` (current method-level GroupDRO).
