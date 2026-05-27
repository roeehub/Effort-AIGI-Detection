# Step250 (Slot 1 Head-Retrain) Quick Eval — FACTS
**Date**: 2026-05-13
**Eval folder**: `analysis/r13_overnight_head_retrain_step250_eval_2026-05-13/`
**Ckpt**: `gs://training-job-outputs/best_checkpoints/fz84lq5k/top_n_effort_20260513_step250_auc0.9847_eer0.0437.pth`
**W&B run**: `fz84lq5k`
**Vertex job**: `6221202060298158080`
**Reference base ckpt**: Slot 1 `top_n_step2000` (W&B `gf6l06rf`)
**Harness**: local CPU/MPS, mirrors `analysis/r13_overnight_may6_retest_2026-05-13/run_may6_retest.py`
**LoRA wrap**: applied BEFORE state_dict load (rank=16, alpha=32, layers=[10,11], modules=[attn.in_proj/out_proj, mlp.c_fc/c_proj]); 0 missing, 0 unexpected on load

## Suites scored

| Suite | n_frames | n_videos | Source |
|---|---|---|---|
| `may6_falseflag` | 92 | 92 | `analysis/xinhe_cross_camera_audit_2026-05-06/raw/may6` |
| `teams_real_all_lockbox` | 1418 | 1361 | GCS via `arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json` (split=lockbox, slice=teams_real_all) |
| `teams_real_all_dev` | 4564 | 3253 | same manifest, split=dev |

dor_shkedi cohort = 1138 videos / 1170 frames inside `teams_real_all_lockbox`. Identity_key `dor_shkedi` is the only cohort filter applied; `real_dor` is reported separately as a sanity check.

## may6 production-drift retest at τ=0.5

| Ckpt | n_fired/92 | p50 | p90 | p99 | max | Spearman vs P8A |
|---|---|---|---|---|---|---|
| P8A_step5000 (precedent) | 0 | 0.0093 | 0.0368 | 0.1479 | 0.2535 | 1.0000 |
| Slot 1 `top_n_step2000` (as-is) | 5 | 0.1765 | 0.3523 | 0.7275 | 0.7764 | 0.9826 |
| **step250 (head-retrain)** | **5** | **0.2034** | **0.3711** | **0.7146** | **0.7618** | **0.9673** |

Spearman r(step250 may6 vs Slot 1 as-is may6) = **0.9992** — rank order on may6 essentially preserved across the head retrain. The 5 firings are the same 5 frames.

## τ=0.5 readout — `teams_real_all_lockbox` and `teams_real_all_dev`

| Ckpt | Suite | Cohort | n_fired | n | FPR |
|---|---|---|---|---|---|
| step250 (head-retrain) | lockbox | all | 117 | 1361 | 8.60% |
| step250 (head-retrain) | lockbox | dor_shkedi | 52 | 1138 | 4.57% |
| Slot 1 step2000 (as-is) | lockbox | all | 0 | 1361 | 0.00% |
| Slot 1 step2000 (as-is) | lockbox | dor_shkedi | 0 | 1138 | 0.00% |
| step250 (head-retrain) | dev | all | 608 | 3253 | 18.69% |
| Slot 1 step2000 (as-is) | dev | all | 0 | 3253 | 0.00% |

NOTE: Slot 1's full real_dev + real_lockbox score distribution is bounded above by ~0.484 (max p50 0.448 / max p99 ~0.464), so τ=0.5 trivially gives 0/N firings on Slot 1. The 45.8% dor_shkedi FPR reported in the task description for Slot 1 was at its calibrated τ=0.4496 (gives dev FPR=6.95%), not τ=0.5.

## Calibrated-τ readout (each ckpt's own τ that gives dev_real_all FPR ≤ 0.07)

Calibration target: `teams_real_all_dev` video-level real FPR ≤ 0.07. v3-fix policy (smallest τ that satisfies the floor).

| Ckpt | τ | dev_real_FPR | lockbox all FPR | lockbox dor_shkedi FPR |
|---|---|---|---|---|
| **step250 (head-retrain)** | **0.7796** | **6.95%** (226/3253) | **1.54%** (21/1361) | **0.26%** (3/1138) |
| Slot 1 step2000 (as-is) | 0.4496 | 6.95% (226/3253) | 39.60% (539/1361) | 45.61% (519/1138) |

Both ckpts meet the FPR target by construction. step250's tau is +0.330 higher than Slot 1's, but its score scale is wider (max ~0.79 vs ~0.48).

## Per-identity lockbox FPR at calibrated τ

| Identity | n_videos | step250 (τ=0.7796) | Slot 1 (τ=0.4496) |
|---|---|---|---|
| dor_shkedi | 1138 | 3 (0.26%) | 521 (45.78%) |
| real_dor | 109 | 0 (0.00%) | 6 (5.50%) |
| bla_bla_chow__s1 | 61 | 0 (0.00%) | 0 (0.00%) |
| PC_Generator__s15 | 28 | 2 (7.14%) | 10 (35.71%) |
| Chikara_Takahashi__s22 | 25 | 16 (64.00%) | 4 (16.00%) |
| **TOTAL** | **1361** | **21 (1.54%)** | **541 (39.75%)** |

step250 dramatically improves dor_shkedi, real_dor, PC_Generator. Regresses on Chikara_Takahashi__s22 (4 → 16 of 25 videos). The Chikara cohort is small (25 vid); the absolute over-fire delta is +12 vs the −518 dor recovery, so net change is overwhelmingly favorable.

## Paired score comparison on dor_shkedi (n=1138 common videos)

| Stat | Slot 1 step2000 (as-is) | step250 (head-retrain) |
|---|---|---|
| p50 | 0.4480 | 0.2106 |
| p90 | 0.4625 | 0.3471 |
| p99 | 0.4702 | 0.7148 |
| max | 0.4754 | 0.7824 |

Per-video Spearman r(slot1, step250 on dor_shkedi) = **0.1169** — the rank order changed substantively. Mechanism: head retrain didn't just shift scores down uniformly; it redistributed which dor_shkedi videos are scored most fake-like.

## Mechanism evidence summary

1. **Encoder unchanged**: load_state_dict reports 0 missing / 0 unexpected (LoRA wrapper applied pre-load); seller log confirms LoRA params populate from ckpt with non-zero values.
2. **may6 rank preserved across head retrain**: Spearman r=0.9992 vs Slot 1 step2000 on per-frame scores; same 5/92 firings at τ=0.5.
3. **Score scale expanded**: Slot 1 step2000 dor_shkedi p50=0.45, max=0.48 (compressed near boundary). step250 p50=0.21, max=0.78 (wider).
4. **dor_shkedi catastrophe closed**: 45.78% → 0.26% at calibrated τ.
5. **Other chronic identities partially regress**: Chikara_Takahashi (n=25) jumps 16% → 64%; PC_Generator stays high but improves; bla_bla_chow stays at 0%.
6. **Dev behavior intact**: same dev_real FPR 6.95% at calibrated τ; same 226/3253 firings.

## Files

- `outputs/scores_step250_may6.csv` (92 rows, per-frame)
- `outputs/scores_step250_lockbox_all.csv` (1418 rows, per-frame; includes identity_key)
- `outputs/scores_step250_dev_real_all.csv` (4564 rows, per-frame; includes identity_key)
- `outputs/summary_step250_vs_slot1.csv` (13 rows, side-by-side comparison)
- `outputs/summary_step250_may6.csv` (1 row, may6 stats)
- `_cache/top_n_effort_20260513_step250_auc0.9847_eer0.0437.pth` (897 MB)
- `run_step250_eval.py` (eval script; reproducible by deleting outputs/ and rerunning)
