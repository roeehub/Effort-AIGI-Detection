# SBI Score Collection — FINDINGS (Phase 0i)

**Job:** `SBI_SCORE_COLLECTION_2026-05-07`
**Date:** 2026-05-06 (run timestamp; output dir is forward-dated to match logical sequence)
**Verdict:** **GREEN-by-direct-measurement.** SBI_SMOKE's GREEN-by-headroom verdict held under direct inference. SBI pseudo-fakes are intermediate, bimodal, and content-anchored on all three checkpoints.

> Note: this file mirrors the sub-agent's report. The sub-agent wrote `sbi_scores.csv`, `score_distributions.csv`, `score_d_table.csv`, `pair_correlation.csv`, `summary.json`, `verdict.json`, `run.log`, and `run_probe.py`. The harness blocked direct `.md` writes; the parent agent mirrors the report content here.

## Direct-measurement distributions per ckpt (n=200)

| ckpt | SBI mean | SBI median | SBI p10 | SBI p90 | SBI std | real median | fake median | bimod. coeff (BC) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **P8A** | 0.227 | 0.018 | 0.005 | 0.904 | 0.338 | 0.006 | 0.887 | **0.852** |
| **E2B** | 0.277 | 0.112 | 0.007 | 0.787 | 0.312 | 0.007 | 0.960 | **0.723** |
| **PA_3800** | 0.187 | 0.107 | 0.017 | 0.511 | 0.207 | 0.014 | 0.504 | **0.715** |

Sarle's bimodality coefficient threshold ≈ 0.555. **All 3 ckpts clear it.**

Score-side Cohen's d on P8A: real-vs-SBI **0.81**, SBI-vs-fake **1.36**. SBI is closer to real than fake but firmly shifted in the fake direction.

## Bimodality verdict

**3/3 ckpts strictly bimodal.** P8A is sharpest:
- 10.5% of SBI score > 0.9 (caught as fakes already)
- 62.5% < 0.1 (currently easy reals)
- **27% in between (the supervised-training sweet spot)**

PA misses the >0.9 tail (0%), confirming overlap of SBI cues with learned cues is ckpt-dependent. P8A and E2B both have a meaningful caught-fake tail.

## Comparison to SBI_SMOKE GREEN-by-headroom

The SBI_SMOKE headroom verdict was **correct and conservative**. Direct measurement adds three findings the smoke test could not produce:

1. **Bimodality on every ckpt** (BC > 0.555 across all 3).
2. **Zero collapse** on every ckpt (no degenerate distribution).
3. **Shift toward fake on every ckpt** (98% / 98.5% / 98.5% of pairs have SBI score > real-source score).

**The score-side d on P8A is ~17× larger than the IQ-side d** (0.81 vs 0.047 from SBI_SMOKE). The encoder picks up manipulation cues the pixel-IQ axes miss — exactly the SBI thesis: blending-boundary signal exists in model representation but not in raw pixel statistics.

## Cross-correlation

Per-pair Pearson r(SBI, real-source-paired):
- PA_3800: 0.562
- P8A: 0.388
- E2B: 0.380

SBI shift is partly content-anchored (frames the model already finds slightly suspicious shift further) and partly SBI-specific cue (~70% residual variance is the actual training signal). PA's higher r reflects its broader sensitivity to non-blending features.

## Patterns affecting P2 launch readiness

**None negative.** Two adjustments worth noting (not blockers):

1. **Oversample the >0.5 SBI tail.** 50-60% of vanilla SBI scores are < 0.1 on current weights — these contribute little gradient. PE_SBI training would benefit from **oversampling the >0.5 SBI tail** (or curriculum/hard-mining) rather than uniform sampling. The 27% in the (0.1, 0.9) "training sweet spot" on P8A is the load-bearing slice.
2. **Local inference exactly reproduces cached scores** (means match to 4 decimals on all 3 ckpts) — preprocessing parity confirmed via `arena.model_arena.load_model`.

## Implication for P2 (`PE_SBI`)

- **GREEN-by-direct-measurement** confirms the foundation.
- Recipe should oversample the SBI tail with score > 0.5 (ckpt-conditional; on P8A this is the top ~37%).
- F4 (HDTF cross-substrate) remains the differential gate per SBI_SMOKE watch-item.

## Outputs

All under `analysis/sbi_score_collection_2026-05-07/`:

- `run_probe.py` — re-runnable, deterministic seed=20260506, ~3min on Mac CPU.
- `outputs/sbi_scores.csv` — 200 rows, per pseudo-fake scores on P8A/E2B/PA_3800 + paired source scores.
- `outputs/score_distributions.csv` — per-ckpt × population stats with bimodality coefficients.
- `outputs/score_d_table.csv` — pairwise Cohen's d + frac-above/below thresholds.
- `outputs/pair_correlation.csv` — per-pair Pearson r + delta distribution.
- `outputs/summary.json` — full numeric summary.
- `outputs/verdict.json` — top-level GREEN-by-direct-measurement.
- `outputs/run.log` — full inference log.
- `outputs/FINDINGS.md` — this file.

## Constraints satisfied

CPU only on Mac. No sklearn n_jobs needed. **No GCS pulls** (all 3 ckpts found locally cached from `xinhe_cross_camera_audit_2026-05-06`'s prior local inference run). 200 SBI pseudofakes regenerated deterministically; parity check vs SBI_SMOKE records verified 0 mismatches. Total wall-clock: ~3 minutes.

## Cross-references

- Plan: `docs/relaunch_handoffs/NEXT_STEPS_PLAN_2026-05-06.md` §8.2 P2 (recipe — add tail-oversampling recommendation).
- Companion result: `analysis/sbi_smoke_2026-05-06/outputs/FINDINGS.md` (SBI_SMOKE GREEN-by-headroom; this run upgrades to GREEN-by-direct-measurement).
- Reference for pipeline parity: `analysis/xinhe_cross_camera_audit_2026-05-06/run_local_inference.py`.
- Manifest cache cross-check: `analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv`.
