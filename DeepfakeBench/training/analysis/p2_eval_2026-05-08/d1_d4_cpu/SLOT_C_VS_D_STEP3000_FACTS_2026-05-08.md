# Slot C step3000 — matched-step counterfactual to Slot D step3000 FACTS (2026-05-08)

> **Status: factual-only.** No interpretation. Forbidden words: succeeds, fails, wins,
> promotes, deployment-grade.
>
> Source: `scores/slotC_periodic_step3000.csv` (newly generated) ×
> `scores/slotD_periodic_step3000.csv` (D5) × `scores/_canary_meta.csv`. Slot C
> ckpt: `gs://training-job-outputs/best_checkpoints/oaur8odo/periodic_effort_20260507_step3000_auc0.9364_eer0.1416.pth`.

## Question

D5 found Slot D `periodic_step3000` to be a distinct low-saturation operating point
(`score_p95_on_reals=0.883`, `lockbox_recall_at_FPR_10pct=0.37` — highest across all
6 saved D ckpts). Is this a property of optimizer-step 3000 in any from-scratch run,
or is it specific to Slot D's Fourier-aug training regime? The matched-step
counterfactual is Slot C `periodic_step3000` (PAIRRANK_ONLY, same step label, no
Fourier aug).

## Method

Score Slot C `periodic_step3000` on the same 800-frame canary parquet (CPU PyTorch,
batch 32, ~50s inference). Compute the same per-ckpt headline stats. Compare to
Slot D `periodic_step3000` from D5.

## Headline distribution stats

| metric | Slot C step3000 | Slot D step3000 | Δ (D − C) |
|---|---:|---:|---:|
| `score_p50_on_reals` | 0.0441 | 0.227 | +0.18 |
| `score_p95_on_reals` | **0.9983** | **0.8828** | −0.116 |
| `score_mean_on_reals` | 0.249 | 0.295 | +0.05 |
| `score_p50_on_fakes` | 0.509 | 0.666 | +0.16 |
| `score_mean_on_fakes` | 0.510 | (≈0.7 inferred) | ~+0.2 |
| `tau_at_FPR_10pct` | 0.959 | 0.779 | −0.18 |
| `lockbox_recall_at_FPR_10pct` | **0.02** | **0.37** | **+0.35** |
| `viso_recall_at_FPR_10pct` | 0.10 | (logged 0.18 at canary _step=3000) | — |
| `deeplive_recall_at_FPR_10pct` | 0.88 | (logged 0.96 at canary _step=3000) | — |
| `max_per_identity_mean_score` | 0.984 | 0.757 | −0.23 |

## Chronic-6 means

| identity | Slot C step3000 | Slot D step3000 | Δ (D − C) | P8A reference |
|---|---:|---:|---:|---:|
| PC_Generator__s22 | 0.073 | 0.282 | +0.21 | 0.92 |
| PC_Generator__s45 | 0.150 | 0.587 | +0.44 | 0.68 |
| Q__s6 | 0.292 | 0.394 | +0.10 | 0.93 |
| Roy_D | **0.869** | 0.757 | −0.11 | 0.06 |
| bla_bla_chow | 0.548 | 0.359 | −0.19 | 0.10 |
| bla_bla_chow__s2 | 0.397 | 0.375 | −0.02 | 0.40 |

Slot C step3000 has **higher** chronic activation on Roy_D, bla_bla_chow than Slot D
step3000 (despite both being at the same optimizer step). Slot D step3000 has
**higher** chronic activation on PC_Generator__s22/s45 than Slot C step3000.

## Per-method recall@FPR_10pct comparison

|  | C step3000 | D step3000 | C step7000 | D step19000 |
|---|---:|---:|---:|---:|
| `lockbox_fake` recall | 0.02 | **0.37** | 0.94 (canary _step=12000) | 0.20 |
| `viso_fake` recall | 0.10 | 0.18 | 0.72 | 0.62 |
| `deeplive_fake` recall | 0.88 | 0.96 | 1.00 | 0.98 |

(canary-substrate readouts; Phase A scorecard will give the 29-suite production view.)

## Direct observations

1. Both ckpts are at optimizer step 3000 in their respective from-scratch runs.
   Slot C uses pair_rank only; Slot D uses Fourier band-amp aug only. No other
   delta in training recipe or yaml at the lever level.
2. Slot C step3000 has `score_p95_on_reals=0.998` (saturated). Slot D step3000
   has `score_p95_on_reals=0.883` (sub-0.95, the only D ckpt with this property).
3. The `lockbox_recall_at_FPR_10pct` gap on the canary substrate is 0.02 (C) vs
   0.37 (D) — a factor of ~18×.
4. The `tau_at_FPR_10pct` for C step3000 is 0.959 (very high — driven by score_p95
   saturation at 0.998); for D step3000 it is 0.779.
5. Chronic-6 saturation pattern is NOT identical between C step3000 and D step3000:
   C is higher on Roy_D + bla_bla_chow; D is higher on PC_Generator. Both diverge
   from P8A on different identities.
6. `max_per_identity_mean_score` is 0.984 (C step3000) vs 0.757 (D step3000) —
   Slot C step3000 has at least one identity scoring near-1.0 mean prob_fake;
   Slot D step3000 does not.

## Cross-reference

- Slot D step3000 stats: `D5_CKPT_MAPPING_FACTS_2026-05-08.md` §"Per-ckpt headline stats"
- Slot C step7000 stats: `D3_HISTOGRAM_FACTS_2026-05-08.md` §"Per-cohort distribution overlap"
- Slot D step19000 stats: same as above
- P8A reference per-cohort means: same as above

## Artifacts

- `scores/slotC_periodic_step3000.csv` — 800 per-frame predictions
- Companion docs: `D5_CKPT_MAPPING_FACTS_2026-05-08.md`, `D3_HISTOGRAM_FACTS_2026-05-08.md`
