# Local CPU inference findings — P8A / E2B / PA_3800 on the 92+60 Xinhe set

**Date**: 2026-05-06
**Inputs**: 92 false-flagged Xinhe real frames (`gs://live-fakes-teams-prod/real/session_20260506_125113/xinhe-may6-real-false-flag-1/`) + 60 caught-correctly Xinhe real frames (`team_sanity_may5__Xinhe`). Both populations are file-prefixed `Generator PC` (same nominal hardware, different days).
**Inference**: local CPU on Mac, batch=8, num_workers=2. Identical preprocessing to training (INTER_LINEAR resize, BGR2RGB, CLIP normalize) via `arena.model_arena.load_model`. Script: `run_local_inference.py`.

## Headline result

**The deployed production model is E2B (or a near-clone).** Pearson r between the session-metadata `score` field (deployment readout) and the local E2B inference is **+1.000** on the 92 may6 frames. Every may6 frame deployment flagged at 0.5+ is also flagged by E2B at 0.5+. The local E2B run reproduces the false-flag distribution exactly.

**P8A does NOT exhibit the same false-flag pattern.** P8A scores both may5 (n=60) and may6 (n=92) at 0% FPR@0.5, with mean 0.021 on both populations.

| ckpt | may5_correct (n=60) | may6_falseflag (n=92) | population AUC |
|---|---|---|---|
|  | mean / p50 / p90 / max / FPR@0.5 | mean / p50 / p90 / max / FPR@0.5 | (may6=1 vs may5=0) |
| **P8A** | 0.021 / 0.007 / 0.031 / 0.234 / **0.000** | 0.021 / 0.009 / 0.037 / 0.254 / **0.000** | **0.619** (no shift) |
| **E2B** | 0.053 / 0.015 / 0.073 / 0.603 / 0.017 | 0.511 / 0.579 / 0.945 / 0.990 / **0.576** | **0.892** |
| **PA_3800** | 0.054 / 0.027 / 0.126 / 0.430 / 0.000 | 0.269 / 0.191 / 0.695 / 0.921 / 0.163 | 0.883 |
| Deployment | (not measured) | mean ~0.83-0.93, RED on most | ≡ E2B |
| IQ-LR (Probe 1 reference) | — | — | **1.000** |

## Cross-correlation: model agreement with deployment on may6 (n=92)

| ckpt | Pearson r vs deploy_score | model_high (≥0.5) | deploy_high (≥0.5) | both_high |
|---|---|---|---|---|
| **P8A** | +0.359 | **0** | 53 | **0** |
| **E2B** | **+1.000** | 53 | 53 | 53 |
| PA_3800 | +0.757 | 15 | 53 | 15 |

E2B's score-frame-by-frame match with deployment is exact. P8A doesn't reproduce the false-flag pattern at all (zero frames cross 0.5).

## IQ-axis correlations within may6 (Pearson r vs score)

| axis | P8A | E2B | PA_3800 | deploy |
|---|---|---|---|---|
| `lap_var_face` | +0.111 | **+0.331** | **+0.479** | +0.331 |
| `luma_mean` | −0.109 | −0.275 | −0.277 | −0.275 |
| `sat_std` | −0.054 | −0.234 | −0.246 | −0.234 |
| `hf_ratio_face` | +0.111 | +0.230 | **+0.362** | +0.230 |
| `face_area` | +0.050 | −0.073 | −0.131 | −0.073 |

E2B and PA_3800 both read the IQ axes that Probe 1 identified as discriminating between may5 and may6. P8A's correlations are uniformly weaker — P8A is reading a different signal class on this substrate.

## What this means

1. **The user's lived experience of "Xinhe across cameras flips" is E2B's behavior.** Deployment = E2B; deployment false-flags Xinhe-may6 at 57.6% FPR. Same identity, same nominal hardware, one day later: 0.5+ score on 53 of 92 frames.
2. **P8A would not produce that flip on this substrate.** P8A scores 0% FPR@0.5 on may6, no frames cross 0.5 — the may6 distribution is invisible to P8A's learned discriminator.
3. **P8A and E2B have different learned shortcut axes.** Probe 4 (paired-feature consistency) found similar — E2B reverses sign on viso pairs while P8A does not. This audit confirms P8A and E2B have learned non-overlapping shortcut representations on Generator PC frames specifically.
4. **PA_3800 is a hybrid.** PA was FT-from-E2B; it inherits some of E2B's IQ-axis sensitivity (correlations match direction, smaller magnitude) but with reduced FPR@0.5 (16.3% vs 57.6%).

## Open question this raises

**Is the production deployment intentionally on E2B?** If yes, the user has the option of switching deployment to P8A on this substrate-class with no retraining. The trade-off (per memory `project_p8a_frame_level_auc_2026-04-29.md` and prior packets):
- P8A has higher real-side FPR robustness on the Generator PC drift axis (this finding).
- P8A has **lower** production-fake recall on `live_fakes_teams_prod` per `analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv` — at single-τ matched, P8A 54.3% vs E2B 74.4% on prod fakes at τ=0.5.
- P8A has lower viso recall on v2 substrate, but matches E2B on HDTF (the production-distribution proxy).

The right deployment model depends on which failure mode you care more about — the false-flag pattern observed today (E2B has it, P8A doesn't) or the missed-fake rate (E2B catches more, P8A catches fewer).

## Outputs

- `scores_P8A.csv`, `scores_E2B.csv`, `scores_PA_3800.csv` — per-frame `prob_fake` for each ckpt
- `scores_with_features.csv` — joined: per-frame scores + IQ features + deployment score + population label (152 rows × 36 cols)
- `inference.log` — run log
- `run_local_inference.py` — the inference script (reuses `arena.model_arena.load_model`)
