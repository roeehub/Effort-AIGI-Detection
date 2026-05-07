# roy_d regression — score distribution + per-frame deltas across ckpts

**Status**: factual-only. No interpretation. Numbers and direct observations.

**Question being answered**: P8A had 29% FPR on roy_d (130 frames in `teams_real_all_dev`); all 6 P1 ckpts have 78-93% FPR on roy_d. What does the per-frame score distribution look like? Where in the scoring distribution did the regression occur?

**Method**: Read `analysis/p1_pe_eval_2026-05-07/raw_reports/phase_a/teams_real_all_dev_<ckpt>_frames_report.csv` for each of 8 ckpts; filter to `video_id` starting with `roy_d` (case-insensitive). Compute distribution stats (p25/p50/p75/p90/p95/max/mean) of `frame_prob` per ckpt + the fraction at-or-above the calibrated `selected_threshold` from Phase A.

**Output artifacts**:
- `roy_d_per_frame_scores.csv` — 1040 rows (8 ckpts × 130 frames) with per-frame `frame_prob`, `tau_selected`, `is_above_tau`.

**Sample size**: 130 unique video_ids (each has 1 frame in this suite); same 130 frames scored by all 8 ckpts.

---

## Score distribution at calibrated τ

| ckpt | τ_selected | p25 | p50 | p75 | p90 | p95 | max | mean | frac ≥ τ_selected (= FPR) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.9156 | 0.0416 | 0.4381 | 0.9396 | 0.9865 | 0.9905 | 0.9938 | 0.4774 | **0.2923** |
| E2B_TOP_N_STEP3200 | 0.7108 | 0.0526 | 0.2857 | 0.6193 | 0.8244 | 0.9169 | 0.9617 | 0.3529 | 0.1538 |
| P1_BUNDLE_PERIODIC_STEP500 | 0.9919 | **0.9922** | 0.9925 | 0.9927 | 0.9929 | 0.9930 | 0.9932 | **0.9924** | **0.9308** |
| P1_BUNDLE_TOP_N_STEP3750 | 0.9994 | 0.9998 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.9978 | **0.8692** |
| P1_BUNDLE_TOP_N_STEP4000 | 0.9989 | 0.9997 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.9939 | 0.8538 |
| P1_PAIRRANK_PERIODIC_STEP500 | 0.7677 | 0.8194 | 0.9822 | 0.9877 | 0.9884 | 0.9885 | 0.9888 | 0.8599 | **0.7846** |
| P1_PAIRRANK_TOP_N_STEP6000 | 0.9878 | 0.9967 | 0.9998 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.9639 | 0.8154 |
| P1_PAIRRANK_TOP_N_STEP6750 | 0.9900 | 0.9931 | 0.9998 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.9615 | 0.7769 |

## Per-frame Δ score: BUNDLE_step500 minus P8A

n_frames matched = 130 (all roy_d frames present in both reports).

| stat | Δ value |
|---|---:|
| p25 | +0.0528 |
| p50 | +0.5545 |
| p75 | +0.9513 |
| max | +0.9856 |
| min | −0.0017 |
| mean | +0.5151 |

## Per-video binarized FPR @ τ_selected — full table

`roy_d_per_frame_scores.csv` has the raw data. 130 video_ids; each is a single frame; binarized columns show 0 (frame_prob < τ_sel) or 1 (frame_prob ≥ τ_sel) per ckpt.

Counts of P8A-vs-BUNDLE transitions across the 130 frames (verified via pivot-and-count on `roy_d_per_frame_scores.csv`):

| transition | BUNDLE_step500 | BUNDLE_step3750 | BUNDLE_step4000 |
|---|---:|---:|---:|
| P8A=0 (correct) → BUNDLE=0 (still correct) | 6 | 17 | 19 |
| P8A=0 (correct) → BUNDLE=1 (regression) | **86** | **75** | **73** |
| P8A=1 (FP) → BUNDLE=0 (recovered) | 3 | 0 | 0 |
| P8A=1 (FP) → BUNDLE=1 (still FP) | 35 | 38 | 38 |
| **Total** | 130 | 130 | 130 |

Sanity: P8A FPR @ τ=0.9156 = 38/130 = 0.292; BUNDLE_step500 FPR @ τ=0.9919 = 121/130 = 0.931 (= 86 + 35); BUNDLE_step3750 FPR = 113/130 = 0.869 (= 75 + 38); BUNDLE_step4000 FPR = 111/130 = 0.854 (= 73 + 38). Matches the per-identity table.

## Direct observations

1. **P8A score distribution on roy_d is bimodal**: p25=0.042, p50=0.438, p75=0.940. Spread spans almost the full [0, 1] range. Mean = 0.477.

2. **BUNDLE_step500 score distribution on roy_d is collapsed near 0.99**: p25 through p95 all sit between 0.992 and 0.993. Mean = 0.992. Range = 0.001.

3. **BUNDLE_step3750 and step4000 are near-saturated** at ~0.9999. Range across p25-p95 is 0.0002.

4. **PAIRRANK_step500 distribution is between P8A and BUNDLE_step500**: p25=0.819, p50=0.982, p75=0.988. Wider than BUNDLE but still upper-shifted vs P8A.

5. **For 130 roy_d frames, BUNDLE_step500 produces a higher score than P8A on ~99% of frames** (only 1 frame had a negative Δ; min Δ = −0.0017).

6. **38 of 130 roy_d frames are P8A-FPs at τ=0.9156**. Under BUNDLE_step500 at τ=0.9919, 35 of those remain FPs (3 recovered to score < τ). Under BUNDLE_step3750/step4000, 0 recovered (all 38 still FP).

7. **Of the 92 P8A-correct (score < τ) frames, 86 became BUNDLE_step500 FPs** (93.5%). 75 became BUNDLE_step3750 FPs (81.5%). 73 became BUNDLE_step4000 FPs (79.3%).

8. **Per-video scan of the per-frame transitions** (in `roy_d_per_frame_scores.csv`) shows a small set of roy_d videos that all 8 ckpts agree are real (score < τ_selected for all): seq782, 1487, 1595, 1615, 1623, 1635, 2656 — 7 frames where every ckpt outputs near-0. These represent "stable real" roy_d examples.

## Companion data

- `roy_d_per_frame_scores.csv` — 1040 rows (8 ckpts × 130 frames). Columns: `ckpt, video_id, frame_path, frame_prob, tau_selected, is_above_tau`.
- The 130 roy_d video_ids are all of the form `Roy_D__seq<N>__real`. Each video_id has exactly 1 frame in this report.

## Cross-references

- The chronic-6 roy_d aggregate stats: `phase_d/per_identity_fpr_FIXED.csv` (P8A 0.292, BUNDLE_step500 0.931).
- `analysis/p1_pe_eval_2026-05-07/dor_invariance_2026-05-07/scores_full.csv` — does NOT contain roy_d frames (different substrate, only dor reals).
- The frame_path values in this CSV can be joined to `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` for IQ-axis values per roy_d frame if a follow-up axis-attribution analysis is desired.
