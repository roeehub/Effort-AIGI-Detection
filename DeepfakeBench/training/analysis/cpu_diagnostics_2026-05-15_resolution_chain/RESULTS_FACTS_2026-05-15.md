# Resolution-chain sensitivity probe — FACTS (2026-05-15)

> **FACTS only.** No verdicts; no recommended packets. Numerical findings.
> Driver: `scripts/run_probe.py`. Analysis: `scripts/analyze.py`.
> Panel: 388 cohort frames from `analysis/dor_encoder_axis_2026-05-08/_cache/frames/`
> spanning 13 identities (10 chronic-cohort + visomaster_enhanced fakes).

## §1. Method

For each of 388 frames, produce 21 variants:
- 1 baseline: cv2.INTER_LINEAR resize from native crop directly to 224×224
- 20 perturbed: 5 downsample sizes ∈ {64, 96, 128, 160, 192} × 4 kernels ∈
  {LINEAR, CUBIC, AREA, LANCZOS4}. Downsample to size then upsample back to
  224 using the SAME kernel both ways (chain-kernel-consistent).

Score every variant on 3 ckpts via the full detector forward (head produces
prob_fake): T5C step3500 (`jrlldtem/periodic_step3500`), P8A step5000
(`9lmvb5b4/value_composite_step5000`), E2B step3200 (`rmat8lwx/top_n_step3200`).

Total: 24,444 forward passes (388 × 21 × 3) on MPS, ~7.5 min wall-time.

## §2. Headline: median score range across the 20 perturbations is enormous on real cohorts for P8A and E2B

`score_range = max(score) − min(score)` across the 20 perturbed variants per
frame. Median across the cohort:

| ckpt | DOR_FAKE_DEV | DOR_REAL_DEV | DOR_REAL_LOCKBOX | NON_DOR_FAKE_DEV | NON_DOR_REAL_DEV |
|---|---:|---:|---:|---:|---:|
| P8A_step5000 | 0.42 | **0.89** | **0.98** | 0.32 | **0.98** |
| E2B_step3200 | 0.52 | **0.90** | **0.97** | 0.10 | **0.93** |
| T5C_step3500 | 0.63 | **0.58** | **0.54** | 0.27 | **0.72** |

**Reading**: on real cohorts (chronic + non-chronic, dev + lockbox), the
median P8A/E2B score range across 20 resolution-chain perturbations of the
same content is **0.89–0.98**. For T5C, **0.54–0.72**. The same real frame,
fed through different downsample→upsample chains, can score anywhere from
near-0 to near-1 on P8A and E2B; T5C compresses this swing by ~30–45%.

## §3. Score range distribution by ckpt (across all 388 frames)

| ckpt | p50 | p90 | max |
|---|---:|---:|---:|
| P8A | 0.969 | 0.988 | 0.989 |
| E2B | 0.847 | 0.982 | 0.989 |
| T5C | 0.601 | 0.794 | 0.840 |

## §4. Flip rate across τ thresholds (% of frames where ANY perturbation crosses τ)

| ckpt | n_frames | τ=0.5 | τ=0.49 (T5C ship τ) | τ=0.7 | τ=0.9 (~P8A ship τ) |
|---|---:|---:|---:|---:|---:|
| P8A | 388 | 70.6% | 70.1% | 78.9% | **85.3%** |
| E2B | 388 | 73.5% | 72.7% | 81.9% | 84.8% |
| T5C | 388 | 75.5% | 74.7% | **87.6%** | **57.7%** |

Notes:
- At τ=0.5 / τ=0.49, all three ckpts flip ~70–76% of frames. Comparable.
- At τ=0.7, T5C is the worst (87.6%) — its score-distribution mass concentrates
  near 0.7 so perturbations cross that boundary readily.
- At τ=0.9 — close to P8A's contract-calibrated τ=0.916 — T5C flips 57.7%
  vs P8A's 85.3%. T5C's score distribution is denser away from 0.9, so high-τ
  is operationally more stable on T5C than P8A.

## §5. Per-size mean score on REAL frames (label=0, n=690 rows per size per ckpt)

The dependence of score on the chosen downsample size, averaged over kernels:

| down_size | P8A mean | E2B mean | T5C mean |
|---:|---:|---:|---:|
| 64  | **0.70** | 0.54 | 0.58 |
| 96  | **0.68** | 0.58 | 0.67 |
| 128 | **0.59** | 0.47 | 0.65 |
| 160 | 0.37 | 0.30 | 0.55 |
| 192 | 0.23 | 0.23 | 0.44 |

The trend is monotone for P8A/E2B: **smaller source resolution → higher
fake score on reals**. P8A goes from 0.70 at down_size=64 to 0.23 at
down_size=192, a 0.47 absolute mean-score swing JUST FROM CHANGING THE
PERTURBATION SIZE. E2B similar (0.54 → 0.23). T5C is flatter
(0.58 → 0.44; 0.14 absolute swing) and non-monotone (peaks at 96/128).

On FAKE frames (label=1), all three ckpts are relatively flat
(P8A 0.85–0.92, E2B 0.73–0.86, T5C 0.65–0.78). The instability is asymmetric:
it lives on the real-frame side.

## §6. Variance attribution per ckpt (2-way: down_size vs kernel)

Fraction of per-frame perturbation variance explained by each axis (mean over 388 frames):

| ckpt | size_frac | kernel_frac | interaction_frac |
|---|---:|---:|---:|
| P8A | 37.6% | 23.4% | 39.0% |
| T5C | 42.4% | 21.0% | 36.5% |
| E2B | 35.3% | **38.4%** | 26.3% |

Reading:
- For P8A and T5C, source-resolution is the dominant single axis (~40%).
- For E2B, kernel is co-equal with size (38% vs 35%). E2B is unusually
  sensitive to which interpolation kernel is used at the same size.
- Interaction terms are large (26–39%) across all three — the joint
  (size, kernel) pair matters, not just each axis independently.

## §7. Per-identity score range (top instability per ckpt)

Identities with largest p90 score range on T5C (the most stable ckpt):

| identity | n | T5C range_p50 | T5C range_p90 | T5C score_baseline_p50 |
|---|---:|---:|---:|---:|
| xiang | 3 | 0.82 | 0.82 | 0.15 |
| Test_Cam | 32 | 0.67 | 0.82 | 0.13 |
| PC_Generator | 22 | 0.70 | 0.81 | 0.42 |
| Md_noyn_Sharker | 11 | 0.72 | 0.79 | 0.09 |
| bla_bla_chow | 15 | 0.71 | 0.77 | 0.44 |

For P8A and E2B, **every** non-saturated identity has range_p90 ≥ 0.94 —
i.e., 90% of the frames on that identity see at least one perturbation that
flips them across nearly the full [0,1] range. The identities that
escape this are those with baseline score ≥ 0.99 (Cam_Test on P8A/E2B,
Noyn_sharker on P8A) — they can only flip downward, and even that is
substantial (Cam_Test on E2B has range_p50=0.05, range_p90=0.75).

## §8. Per-frame ckpt disagreement (across the 21 variants × 3 ckpts grid)

For each frame, max stdev across ckpts at the same variant:

| cohort | mean ckpt_std_max | mean ckpt_std_p50 |
|---|---:|---:|
| DOR_REAL_LOCKBOX | 0.393 | 0.215 |
| DOR_REAL_DEV | 0.337 | 0.144 |
| NON_DOR_REAL_DEV | 0.322 | 0.124 |
| DOR_FAKE_DEV | 0.293 | 0.171 |
| NON_DOR_FAKE_DEV | 0.240 | 0.104 |

The 3 ckpts disagree most on the lockbox real cohort. At least one
perturbation per frame produces a P8A/T5C/E2B std of ≥ 0.39 on average for
lockbox reals — i.e., the ckpts assign substantially different scores to
the same perturbation of the same content.

## §9. Output files

- `outputs/per_frame_per_variant.parquet` — 24,444 rows: full score grid
- `outputs/per_frame_summary.parquet` — 1,164 rows: per (frame, ckpt) stats
- `outputs/range_by_cohort.csv`
- `outputs/range_by_identity.csv`
- `outputs/anova_per_ckpt.csv`
- `outputs/flip_rate_per_ckpt.csv`
- `outputs/per_size_mean_score_reals.csv`
- `outputs/per_size_mean_score_fakes.csv`
- `outputs/ckpt_disagreement_per_frame.csv`

## §10. Caveats

1. Cohort is identity-heavy on dor_shkedi (228/388 frames) — global numbers
   are weighted toward this identity. Per-identity tables (§7) decompose.
2. Source frames are the ones in `_cache/frames/` (eval-substrate crops);
   their native resolutions vary identity by identity. Perturbations are
   relative to that native size, not to a production-capture baseline.
3. Kernel set is cv2's 4 standard interpolations. Production capture
   pipelines may use different kernels (Teams' specific resampling has not
   been audited at this level).
4. The chain forces kernel consistency (same down + up kernel). Mixed
   kernels (down with one, up with another) would add variants but reduce
   per-cell sample size; not measured here.
5. All scores are at the model's raw output; deployment-runtime aggregation
   (per-window decisions, IQ gating) is downstream and could attenuate the
   instability before it reaches the user.
