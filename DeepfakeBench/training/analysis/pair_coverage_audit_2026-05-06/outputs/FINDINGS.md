# Pair Coverage Audit — FINDINGS

**Job:** `PAIR_COVERAGE_AUDIT_2026-05-06`
**Date:** 2026-05-06
**Verdict:** **Pair-rank loss is broadly applicable.** ~91–97% of training is paired (depending on the metric); all 6 paired lanes share the same pair key. Recommend applying uniformly with a one-line skip rule for the unpaired-real lane.

> Note: this file mirrors the sub-agent's report. The sub-agent wrote `pairing_semantics_notes.md` (1-page reference for the pair-rank recipe) and CSVs / summary; this FINDINGS file is added by the parent agent to keep the file structure consistent with the plan's expectations.

## Headline coverage

| Metric | Value |
|---|---|
| Paired fraction (base samples) | ~**91.5%** (12,888 / 14,088) |
| Paired fraction (frames per epoch) | ~**96.6%** |
| Paired fraction (effective batch share by yaml `family_weights`) | ~**93.3%** |

Pair-rank fires on essentially every batch.

## Lanes WITH paired structure (pair-rank applies)

All six paired lanes use **the same pair key**: `(sample_id, frame_idx)` with opposite `label`. The iterator yields real then fake back-to-back.

| Lane | Paired base samples | Family weight | Notes |
|---|---:|---:|---|
| `df40` | 4,698 | (per yaml) | 7 of 17 methods enabled in ship yaml; ~954 identities |
| `deeplive` | 4,067 | (per yaml) | 5 strategies (3 clean + 2 enhanced) enabled |
| `visomaster_v1_base` | 342 (lower bound; up to ~430) | (per yaml) | 9 swap models enabled |
| `visomaster_enhanced` | 1,484 | (per yaml) | 8 enhancers, clean transport |
| `visomaster_teams_enhanced` | 997 | (per yaml) | only 54 (5.4%) have true `teams_v2` real companion; 943 fall back to clean. Fake branch picked per-epoch from N+1 options |
| `deeplive_teams` | ~1,300 (approx) | **7.0 — heaviest in yaml** | Teams-passthrough JPGs, 5 strategies; closest to production |

## Lane WITHOUT paired structure (skip pair-rank)

- `external_vcd_real` — 1,200 frames cap (~80 identities). `UnifiedUnpairedRealSample` iterator yields `label=0` only with no `frame_idx`-matched fake. Pair-rank loss term must be skipped when `is_unpaired_real=True`. CE still applies.

## Disabled / inactive

- `visomaster_hints`, `visomaster_hints_teams` — explicitly disabled in ship yaml (verified bad data; memory `project_visomaster_hints_lanes_bad_data.md`).
- `proper_data` lane — not enabled in `R13_VISO_CORR_PENALTY` (would be paired if reactivated).

## Recommendation: apply uniformly across all 6 paired lanes

**Do NOT restrict to "DF40 + VisoMaster + DeepLive only, drop Teams passthrough."** Rationale:

1. All 6 paired lanes share identical pair structure (`sample_id`+`frame_idx` key, opposite labels) — uniform application is the simplest correct implementation.
2. Restricting away `deeplive_teams` would punt on the binding metric — `deeplive_teams` carries the highest family weight (7.0) and is the closest substrate to production.
3. The unpaired `external_vcd_real` lane already needs special handling (`is_unpaired_real=True` flag) — a one-line skip-guard, not a real decision boundary.
4. The `visomaster_teams_enhanced` per-epoch fake-branch rotation is a feature, not a bug: pair-rank fires on whichever branch is sampled, and the loss sees all enhancers across training.
5. **The 2026-05-04 `project_pair_loss_premise_refuted_2026-05-04.md` memory does NOT transfer to PE_PAIR_RANK_DRO.** That memory refuted *raw vs teams-transport symmetric consistency* on viso enhanced. PE_PAIR_RANK_DRO is *real-vs-fake same-source ranking* — an orthogonal premise. The external-advisor synthesis already argued this; the coverage probe corroborates by showing the actual pair structure in the loader.

## Reconciliation with `PAIR_GAP_AUDIT_2026-05-06`

Both findings are correct, measuring different things:

- `PAIR_GAP_AUDIT` measured pair_gap inversion rate on **cross-product pairs within identity** (capped at 4k pairs/subject) on the eval manifest. Verdict: RED on aggregate (P8A 9.88% / E2B 5.63% / PA 17.79% inversion rate among missed fakes).
- `PAIR_COVERAGE_AUDIT` measured the **actual training-loader pair structure**: tight same-source pairs (`sample_id`, `frame_idx`, opposite label). 91-97% of training is paired this way.

The cross-product audit's RED verdict does NOT directly apply to the tight training pairs because the loose cross-product massively over-counts the number of real-fake comparisons relative to what the loss would actually compute.

**The right go/no-go for `PE_PAIR_RANK_DRO` is `SAME_SOURCE_PAIR_GAP_AUDIT_2026-05-06` (Phase 0g)** — extracting cached scores at training-time pair indices on the ~5,379 frame-level same-source pairs in DF40/DeepLive/VisoMaster lanes. Until that runs, P1's signal-vs-no-signal verdict is open.

## Pair-rank loss design implications

- **Pair detection at batch time:** find items in same batch sharing `(sample_id, frame_idx)` AND opposite labels. Both are present in the same forward pass under default collate. The current `combined_paired_collate_fn` (line 3503 of `combined_paired.py`) groups by `(sample_id, label)` but does not preserve the explicit real-fake link as a first-class field. **The PE recipe should add a tag** on the dict (e.g., `pair_id = (sample_id, frame_idx)`) so the loss can find pairs in O(N) without label-side joins.
- **Skip rule:** items with `is_unpaired_real=True` contribute 0 to pair-rank; CE only.

## Major lanes where paired data is MISSING that we should add

None on the PE critical path. The one structural gap is `visomaster_teams_enhanced` having only 5.4% true `teams_v2` companion coverage — but PE doesn't *need* matched-transport pair-rank (it ranks real vs fake on score, not transport-invariance). Filling the missing 945 `teams_v2` companions would be a future packet's input, not PE's.

## Caveats

- `deeplive_teams` count (~1,300) is approximate; exact requires GCS listing.
- VisoMaster V1 base recorded as 342 lower bound (proper-viso clean count proxy); memory inventory upper bound is ~430.
- External VCD identity count (~80) is a 0.40 split estimate.
- None of these change the headline 91–97% paired-fraction conclusion.

## Outputs

- `run_probe.py` — audit script (CPU-only, no GCS / model loading; uses cached metadata).
- `coverage_by_method.csv` — per-method × lane × transport × enhancer counts.
- `coverage_by_enhancer.csv` — aggregated by enhancer family.
- `coverage_by_transport.csv` — aggregated by transport (`raw_clean` / `teams_v2` / `webcam_codec`).
- `coverage_by_identity.csv` — per-lane unique-identity counts (paired vs real-only).
- `coverage_summary.json` — full lane breakdown + totals + conclusions.
- `pairing_semantics_notes.md` — 1-page reference for the pair-rank recipe (top level of the audit dir, not in `outputs/`).
- `FINDINGS.md` — this file.

## Cross-references

- Plan: `docs/relaunch_handoffs/NEXT_STEPS_PLAN_2026-05-06.md` §8.2 P1 (recipe should apply uniformly), §6.5 (advisor's pair-rank argument corroborated).
- Companion result: `analysis/pair_gap_audit_2026-05-06/outputs/FINDINGS.md` — cross-product RED verdict; same-source audit (Phase 0g) is the right go/no-go.
- Pair-loss prior refutation correctly NOT transferring: `project_pair_loss_premise_refuted_2026-05-04.md` (different premise; raw-vs-teams transport-invariance, not real-vs-fake same-source).
- Code paths: `data/sources/combined_paired.py` (`CombinedPairedIterableDataset`, collate at line 3503), `data/sources/df40_paired.py`, etc.
