# TTA POC on May6 — FINDINGS (Phase 0a, tight scope)

**Job:** `TTA_POC_MAY6_2026-05-06`
**Date:** 2026-05-06
**Verdict:** **NOT VIABLE** as a production patch for may6 false-flag pattern. E2B may6 FPR 57.6% → 56.5% with 4-view TTA. P8A may6 invariance preserved.

> Note: this file mirrors the sub-agent's report. The sub-agent wrote `tta_scores.csv`, `tta_scores_E2B.csv`, `tta_scores_P8A.csv`, `fpr_comparison.csv`, `per_view_breakdown.csv`, `summary.json`, `verdict.json`, `probe_run.log`, and `run_probe.py`. The harness blocked direct `.md` writes; the parent agent mirrors the report content here.

## Verdict table (τ=0.5)

| ckpt | substrate | n | single-view FPR | TTA FPR | Δ |
|---|---|---:|---:|---:|---:|
| **E2B** | **may6_falseflag** | 92 | **0.576** | **0.565** | **−0.011** |
| E2B | may5_correct | 60 | 0.017 | 0.017 | 0.000 |
| E2B | dor_morning | 50 | 0.020 | 0.020 | 0.000 |
| E2B | dor_evening | 50 | 0.000 | 0.000 | 0.000 |
| P8A | may6_falseflag | 92 | 0.000 | 0.000 | 0.000 |
| P8A | may5_correct | 60 | 0.000 | 0.017 | +0.017 |
| P8A | dor_morning | 50 | 0.100 | 0.020 | −0.080 |
| P8A | dor_evening | 50 | 0.000 | 0.000 | 0.000 |

V1 baseline FPR on may6 E2B (0.576) reproduces the morning's `xinhe_cross_camera_audit_2026-05-06` result (0.576) exactly — pipeline parity confirmed.

P8A may6 invariance preserved (0% → 0%). TTA didn't hurt anywhere meaningfully (P8A may5 saw a single new FP added — TTA_NEUTRAL_INVARIANT classification).

Net E2B rescue on may6: **7 V1-FPs rescued, 6 new FPs introduced — net 1 frame.**

## Why TTA fails

May6 false-flags are **confident**:
- median V1 fake-prob = 0.58
- q90 = 0.95
- mean across-view std = 0.10
- 16% of frames have view-std > 0.2 (i.e., 84% of frames TTA can barely move)

Score quantile compression is asymmetric and small: q50 drops 0.58 → 0.55, but **q25 actually rises +0.15** (TTA pushes some clean-looking frames *up*). TTA's effect is symmetric and narrow — not a damping mechanism for confident misclassifications.

Only **4 of 92** may6 frames sit in the V1 ∈ [0.45, 0.55] band where TTA has any leverage to flip the τ=0.5 decision. The remaining 88 frames are either confidently fake (53) or confidently real (~35) — TTA lacks the magnitude to move them.

Consistent with `project_dor_drift_named_axes_2026-05-06`: may6 drift lives on a named-axis ridge (`min_dim`, `color_b_dev`, `edge_mag`) at ~90% explained variance per the union-fit ridge. **Pixel-level local jitter operates on the wrong axis** — it's a perceptual-noise lever, but the may6 shortcut is a structural axis-level shift.

## Recipe — which views are load-bearing?

All 4 views are highly correlated on may6 E2B:
- r(V1, V2_flip) = 0.85
- r(V1, V3_shift+scale) = 0.81
- r(V1, V4_shift+scale+flip) = 0.73

Mean |Δ view vs V1|: V2=0.13, V3=0.15, V4=0.18. V4 gives the most independent signal but its standalone FPR (0.554) is essentially V1's (0.576). **No single augmentation drives substantive correction.**

## Recommendation

**Do not productionize this recipe.** Larger augmentation magnitudes (±10px / ±10% scale / rotation / brightness) might add separation but risk inflating FPR on the clean substrates where E2B is already invariant. The math on may6 suggests even doubled view-count won't reach the <20% FPR target.

## Closes and re-routes

This POC **closes the TTA branch as NOT VIABLE** for the may6 production false-flag pattern. The `deployment-vs-p8a-substrate-tradeoff-not-quantified` open loop remains unresolved; remaining levers are:

1. **Substrate-aware τ-raise on may6-like inputs** — OFFLINE-only per per-mode-τ-not-deployable constraint (Teams doesn't surface capture mode at inference). Useful for evaluation; not deployable.
2. **Deployment-side switch E2B → P8A on may6-class substrates** — concrete deployment-side action. Trade: P8A's higher live-prod FPR or lower fake recall on some cohorts (per `CHECKPOINT_COHORT_DIAGNOSIS` Agent 3, P8A wins 7/9 cohort axes). **Now the primary production-pain mitigation option since TTA is dead.**
3. **IQ gating per `project_image_quality_shortcut.md`** — the more principled fix; uses sharpness/luma quartile to gate which model fires. P8A recall increases monotonically with sharpness (Q1 36% → Q4 98%); E2B is INVERTED (Q1 52% → Q3 5%) — IQ gating is a P8A-specific lever, non-starter for E2B.

## Outputs

All under `analysis/tta_poc_may6_2026-05-06/`:

- `run_probe.py` — re-runnable; auto-detects cached checkpoints in `/var/folders/.../arena_ckpt_*/`.
- `outputs/tta_scores.csv` — 504 rows (252 frames × 2 ckpts), per-view + mean + std.
- `outputs/tta_scores_E2B.csv`, `outputs/tta_scores_P8A.csv` — per-ckpt detail.
- `outputs/fpr_comparison.csv` — substrate × ckpt FPR table.
- `outputs/per_view_breakdown.csv` — per-view FPR + r(V1, Vk) + rescued/new-FP counts.
- `outputs/summary.json` — top-level booleans (`tta_damps_may6_E2B_below_20pct: false`, `tta_damps_may6_E2B_meaningfully: false`, `P8A_may6_invariance_preserved: true`).
- `outputs/verdict.json` — per-cell `TTA_NEUTRAL` / `TTA_NEUTRAL_INVARIANT`.
- `outputs/probe_run.log` — full execution log.
- `outputs/FINDINGS.md` — this file.

Total run cost: $0 (Mac CPU; ~120s inference + ~30s model load × 2 ckpts).

## Cross-references

- Plan: `docs/relaunch_handoffs/NEXT_STEPS_PLAN_2026-05-06.md` §8.1 0a (close TTA branch), §9 deployment-vs-P8A-substrate-tradeoff (TTA option removed; remaining levers re-ranked).
- Memory: `project_dor_drift_named_axes_2026-05-06.md` (drift lives on named axes — explains why TTA fails).
- Memory: `project_image_quality_shortcut.md` (IQ-axis monotone separation; P8A only).
- Reference for pipeline parity: `analysis/xinhe_cross_camera_audit_2026-05-06/run_local_inference.py`.
- Memory: `feedback_per_mode_tau_not_deployable.md` (per-substrate τ is offline-only).
