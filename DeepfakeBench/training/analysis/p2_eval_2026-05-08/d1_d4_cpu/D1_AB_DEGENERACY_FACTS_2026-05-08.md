# D1 — Slot A / Slot B local-inference distribution FACTS (2026-05-08)

> **Status: factual-only.** No interpretation. Forbidden words: succeeds, fails, wins,
> promotes, deployment-grade.
>
> Source: `scores/slot{A,B}_top_n_step500.csv` × `scores/_canary_meta.csv` produced by
> `run_inference.py` on the 800-frame canary parquet.

## Question

What does Slot A's `top_n_step500` and Slot B's `top_n_step500` produce on the canary
800-frame substrate? In particular: distribution shape, label-conditional means, and
per-cohort breakdown.

## Method

Local CPU inference (PyTorch CPU, batch 32, 25 batches × 800 frames). Same 224×224
preprocessing as the trainer (CLIP mean/std normalization). Per-frame `prob_fake`
saved to `scores/<ckpt_id>.csv`.

## Slot A `top_n_step500` (auc=0.6802) — distribution stats

Source: `outputs/d1_distribution_stats.csv` filter `ckpt=slotA_top_n_step500`.

| scope | n | mean | p05 | p50 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 800 | 0.4525 | 0.2978 | 0.4652 | 0.5949 | 0.1988 | 0.6759 |
| label 0 (real) | 600 | 0.4346 | 0.2873 | 0.4265 | 0.6019 | 0.1988 | 0.6759 |
| label 1 (fake) | 200 | 0.5061 | 0.3962 | 0.5164 | 0.5741 | 0.3499 | 0.6084 |

Per-cohort means (label-flagged):

| cohort | label | n | mean | p50 | p95 |
|---|---:|---:|---:|---:|---:|
| chronic_PCGen_s22 | real | 50 | 0.5135 | 0.5278 | 0.5995 |
| chronic_PCGen_s45 | real | 50 | 0.4938 | 0.4854 | 0.5739 |
| chronic_Q_s6 | real | 50 | 0.3925 | 0.4037 | 0.5048 |
| chronic_Roy_D | real | 50 | 0.5283 | 0.5340 | 0.5993 |
| chronic_bla_bla_chow | real | 50 | 0.5217 | 0.5163 | 0.5783 |
| chronic_bla_bla_chow_s2 | real | 50 | 0.4569 | 0.4577 | 0.5471 |
| hdtf_clean_real | real | 50 | 0.3681 | 0.3563 | 0.4716 |
| healthy_dor | real | 50 | 0.4185 | 0.4170 | 0.4863 |
| healthy_dor_shkedi | real | 50 | 0.5208 | 0.5260 | 0.5867 |
| healthy_md_noyn_sharker | real | 50 | 0.4189 | 0.4129 | 0.5100 |
| healthy_test_cam | real | 50 | 0.3194 | 0.3197 | 0.4014 |
| healthy_xiang_xiang2_feng | real | 50 | 0.3589 | 0.3568 | 0.4252 |
| deeplive_fake | fake | 50 | 0.4977 | 0.5118 | 0.5736 |
| lockbox_fake | fake | 100 | 0.5101 | 0.5234 | 0.5704 |
| viso_fake | fake | 50 | 0.5066 | 0.5168 | 0.5824 |

(Cohort means computed live from `outputs/d1_distribution_stats.csv` rows where
`scope=cohort`.)

## Slot B `top_n_step500` (auc=0.6800) — distribution stats

| scope | n | mean | p05 | p50 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 800 | 0.5638 | 0.4499 | 0.5662 | 0.6564 | 0.3528 | 0.7117 |
| label 0 (real) | 600 | 0.5597 | 0.4383 | 0.5618 | 0.6636 | 0.3528 | 0.7117 |
| label 1 (fake) | 200 | 0.5759 | 0.4844 | 0.5886 | 0.6392 | 0.4597 | 0.6661 |

(Slot B per-cohort table omitted for brevity — see
`outputs/d1_distribution_stats.csv` filter `ckpt=slotB_top_n_step500, scope=cohort`.)

## Cross-comparison vs in-training canary

Per-slot canary fires logged at `_step=7000` (Slot A) and `_step=6000` (Slot B). The
canary fires at those `_step`s are NOT the same model state as `top_n_step500`. The
in-training canary scored:

| metric | Slot A canary (`_step=7000`) | Slot A local (`top_n_step500`) | Slot B canary (`_step=6000`) | Slot B local (`top_n_step500`) |
|---|---:|---:|---:|---:|
| reals score_p50 | 0.4994 | 0.4265 | 0.5003 | 0.5618 |
| reals score_p95 | 0.4994 | 0.6019 | 0.5005 | 0.6636 |
| reals score_mean | 0.4991 | 0.4346 | 0.5003 | 0.5597 |
| fakes score_p50 | 0.4994 | 0.5164 | 0.5002 | 0.5886 |
| fakes score_mean | 0.4994 | 0.5061 | 0.5004 | 0.5759 |

The in-training canary's near-uniform 0.499/0.500 distribution is NOT reproduced by
local inference of the saved `top_n_step500` ckpt. Slot A `top_n_step500` shows a
real/fake mean gap of +0.072 (fakes higher) with std 0.10 on reals; Slot B shows a
gap of +0.016 with std 0.07 on reals.

The two ckpt files (`top_n_step500` for A and B) and the model states at canary
fire-time (`_step=7000` for A, `_step=6000` for B) are different model states. The
ratio between yaml `optimizer_step` and W&B `_step` is unverified at this level.

## Direct observations

1. Slot A `top_n_step500` real-distribution range is [0.20, 0.68]; fake range [0.35, 0.61].
   Real-fake mean gap = +0.072 (fakes higher).
2. Slot B `top_n_step500` real-distribution range is [0.35, 0.71]; fake range [0.46, 0.67].
   Real-fake mean gap = +0.016 (fakes higher).
3. Slot A's per-cohort real means range from 0.32 (`healthy_test_cam`) to 0.53
   (`chronic_Roy_D`). Slot A's per-cohort spread is ~0.20.
4. Slot B's score range [0.35, 0.71] is shifted right of Slot A's [0.20, 0.68]; both
   midpoints are ~0.45-0.55.
5. The in-training canary's `score_p95_on_reals=0.4994` (A) and `0.5005` (B) at
   single-fire are NOT reproduced by local inference of `top_n_step500`. The canary
   fires reflect later model states; `top_n_step500` reflects the earliest top-1 ckpt
   selected by the value-composite tracker.

## Artifacts

- `outputs/d1_distribution_stats.csv` (108 rows = 6 ckpts × 18 scopes)
- `scores/slot{A,B}_top_n_step500.csv` — per-frame prob_fake
- `figs/d3_hist_*.png` — overlaid histograms incl. Slot A/B
- Driver: `analyze_scores.py` (function `d1_distribution_stats`)
