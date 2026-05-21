# Phase 0 A0.2 — Multi-Layer CLIP Feature Geometry on Substrate-Pair Inventory — FACTS (2026-05-22)

> **Status: factual-only.** Forbidden words (none present below): succeeds, fails, wins, loses, promotes, deployment-grade, ship, kill, best, worst, unfortunately, remarkably, lucky, confirmed, refuted, shortcut-aligned, gap-is-wide, gap-is-narrow.
> Numbers + tables + cross-references only. Interpretation lives in `AGENT_PROPOSAL_2026-05-22.md`.

---

## 1. Method

1. Loaded the 1,880-pair inventory `inventory_manifest.csv` (A0.1 artifact). Of those, 1,826 carry a non-null `clean_real_frame_count`; the remaining 54 `visomaster_teams_enhanced` rows have an unknown clean-side count and are skipped here.
2. Selected the first `N=3` frames per pair per side (`min(N, clean_count, teams_count)`); produced a `5,478 × 2`-row frame manifest (clean + teams sides).
3. Downloaded each frame from GCS, decoded with OpenCV, resized to `224×224`, normalized with OpenCLIP mean/std `(0.481, 0.458, 0.408)` / `(0.269, 0.261, 0.276)`, cached as a single fp16 tensor on disk (`_cache_frames_<side>.pt`, ~1.65 GB per side). Decode rate: clean ~340 ms / frame, teams ~330 ms / frame. Decode failures: clean = 3 (pair_ids 285 only, listed as `OpenCV imdecode_ assertion failed` — likely zero-byte blob), teams = 0.
4. Cached frames per side: clean `5,475` valid (after dropping 3 decode-failed pair-ids), teams `5,478` valid. After intersecting `pair_id` keys across sides, **1,825 paired pair_ids** had both `clean[frame_idx=0]` and `teams[frame_idx=0]` present and were used for the `cos_pair` and KLIEP-projection metrics.
5. Three checkpoints loaded via `batch_inference_gcs.load_model`:
   - `P8A_step5000` = `gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth`
   - `SlotAv2_step3500` = `gs://training-job-outputs/best_checkpoints/hp35c51p/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth`
   - `T5C_step3500` = `gs://training-job-outputs/best_checkpoints/jrlldtem/periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth`
6. Registered forward hooks on `model.backbone.visual.transformer.resblocks[L]` for `L ∈ {0, 4, 8, 11}` (verified n_blocks = 12); CLS token extracted via `output[0]` (seq-first) per the wrapper convention.
7. Single forward pass per side per ckpt captured all 4 layers' CLS tokens and the head probability `pred["prob"]`. Forward throughput (32-frame batch, MPS, M-series Mac): ~575 frames / s peak (smoke); ~58 frames / s end-to-end with disk reads (full run).
8. Per-cell metrics (3 ckpts × 4 layers = 12 cells):
   - `cos_pair`: cosine of L2-normalized `clean[frame_idx=0]` and `teams[frame_idx=0]`, mean over the 1,825 matched pair_ids.
   - `cos_within_same`: off-diagonal mean of the full `k × k` cosine gram matrix among the 3 same-side frames, averaged across both sides and across pair_ids that retain `≥ 2` valid frames.
   - `cos_cross_id`: cosine between `clean[frame_idx=0]` of two distinct random pair_ids, drawn `N_cross = 2,000` times (random `default_rng(42)`).
   - `delta_pair_vs_within = cos_pair − cos_within_same`.
   - `score_corr`: Pearson correlation of `(prob_clean[frame_idx=0], prob_teams[frame_idx=0])` over the 1,825 matched pair_ids (one value per ckpt; the head produces one prob per frame and is invariant to the L hook).
9. **KLIEP axis** (L11 only): refit on the cached CLIP-frozen L11 features from `analysis/cpu_diagnostics_2026-05-12_d8_head_retrain_substrate_balanced/outputs/clip_frozen_l11__n4839.npz` (2,000 dev_real + 414 lockbox_real, L2-normalized, `LogisticRegression(C=1.0, max_iter=2000, class_weight=balanced, n_jobs=1)`; mirrors `analysis/cpu_diagnostics_2026-05-12_d10_training_pool_position/run_d10.py:fit_kliep_discriminator`). Refit accuracy on the 2,414-sample classifier = `0.9909`. The discriminator unit-vector `w_hat = w / ||w||` was projected onto per-pair difference vectors `(L2-norm(teams[0]) − L2-norm(clean[0]))`. Stored as 5,475-row CSV (1,825 pair_ids × 3 ckpts).
10. **Compute environment**: macOS Darwin 24.3.0, Python 3 (`anaconda3`), torch 2.11.0 + MPS, OpenCV 4.11.0, 32 GB RAM, local SSD. No CUDA, no GPU job. Cache fp16 dtype, `n_frames = 3`, `batch_size = 32`. Total wall time: ~76 minutes including frame-cache build (33 min × 2 sides decode + 5 min × 2 sides blob-listing); pure forward + metric time = ~12 minutes.

---

## 2. Output artifacts

| Artifact | Path | Size / Shape |
|---|---|---|
| 12-cell metrics CSV | `analysis/substrate_pair_geometry_2026-05-22/per_ckpt_layer_cosines.csv` | 12 data rows × 12 cols |
| Per-pair L11 KLIEP projections | `analysis/substrate_pair_geometry_2026-05-22/kliep_axis_projections.csv` | 5,475 data rows × 4 cols |
| Phase-0 gate verdict JSON | `analysis/substrate_pair_geometry_2026-05-22/_gate_verdict.json` | 4-key object |
| KLIEP `w_hat` unit-vector | `analysis/substrate_pair_geometry_2026-05-22/_kliep_w_hat.npy` | 768-dim float64 |
| KLIEP metadata | `analysis/substrate_pair_geometry_2026-05-22/_kliep_metadata.json` | accuracy, bias, w_norm |
| Per-frame features (12 × 2 = 24 files) | `analysis/substrate_pair_geometry_2026-05-22/feats/{ckpt}_L{L}_{side}.npy` | each ~5,475×768 float32 |
| Per-frame head probs (3 × 2 = 6 files) | `analysis/substrate_pair_geometry_2026-05-22/scores/{ckpt}_{side}.npy` | each ~5,475 float32 |
| Frame caches | `_cache_frames_{clean,teams}.pt` | 1,648 + 1,649 MB fp16 |
| Per-frame manifests | `_cache_frames_meta_{clean,teams}.parquet` | 5,475 / 5,478 rows |
| Builder script | `analysis/substrate_pair_geometry_2026-05-22/run_phase0_geometry.py` | 825 lines, single module |
| Run log | `analysis/substrate_pair_geometry_2026-05-22/_run_phase0.log` | 53 KB |

---

## 3. Headline 12-cell table — `per_ckpt_layer_cosines.csv`

| ckpt_key | layer | cos_pair | cos_within_same | cos_cross_id | Δ = cos_pair − cos_within | KLIEP μ | KLIEP σ | score_corr | n_pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **P8A_step5000** | L0 | 0.9996 | 0.9998 | 0.9978 | −0.0002 | — | — | 0.2787 | 1825 |
| P8A_step5000 | L4 | 0.9824 | 0.9979 | 0.9857 | −0.0155 | — | — | 0.2787 | 1825 |
| P8A_step5000 | L8 | 0.9784 | 0.9945 | 0.9687 | −0.0161 | — | — | 0.2787 | 1825 |
| **P8A_step5000** | **L11** | **0.8668** | **0.9492** | **0.8683** | **−0.0824** | **+0.0052** | 0.0195 | 0.2787 | 1825 |
| **SlotAv2_step3500** | L0 | 0.9996 | 0.9998 | 0.9978 | −0.0002 | — | — | 0.3685 | 1825 |
| SlotAv2_step3500 | L4 | 0.9826 | 0.9979 | 0.9856 | −0.0153 | — | — | 0.3685 | 1825 |
| SlotAv2_step3500 | L8 | 0.9757 | 0.9942 | 0.9643 | −0.0185 | — | — | 0.3685 | 1825 |
| **SlotAv2_step3500** | **L11** | **0.8673** | **0.9447** | **0.7201** | **−0.0774** | **+0.0135** | 0.0261 | 0.3685 | 1825 |
| **T5C_step3500** | L0 | 0.9996 | 0.9998 | 0.9978 | −0.0002 | — | — | 0.4314 | 1825 |
| T5C_step3500 | L4 | 0.9824 | 0.9979 | 0.9856 | −0.0155 | — | — | 0.4314 | 1825 |
| T5C_step3500 | L8 | 0.9758 | 0.9939 | 0.9628 | −0.0182 | — | — | 0.4314 | 1825 |
| **T5C_step3500** | **L11** | **0.8654** | **0.9401** | **0.7143** | **−0.0748** | **+0.0148** | 0.0230 | 0.4314 | 1825 |

(Bolded rows are the SlotAv2_step3500/L11 gate input and the L0+L11 reference cells.)

---

## 4. Phase-0 gate verdict

Gate definition (from `analysis/substrate_pair_geometry_2026-05-22/run_phase0_geometry.py:768-794` and the master plan):

| Condition (at SlotAv2_step3500 / L11) | Verdict |
|---|---|
| `delta ≥ −0.02 AND |kliep_proj_mean| ≤ 0.2` | NO FULCRUM |
| `delta ≤ −0.05 AND |kliep_proj_mean| ≥ 0.4` | STRONG FULCRUM |
| middle | AMBIGUOUS |

At the SlotAv2_step3500 / L11 cell:
- `delta_pair_vs_within = −0.0774` (≤ −0.05; satisfies the STRONG-FULCRUM half-condition on cluster contraction)
- `|kliep_projection_mean| = 0.0135` (≤ 0.2; satisfies the NO-FULCRUM half-condition on axis alignment, AND does not meet the ≥0.4 STRONG threshold)

Neither full gate clause holds → **verdict = AMBIGUOUS**. Persisted in `_gate_verdict.json`.

KLIEP classifier accuracy on its training data (2,000 dev_real + 414 lockbox_real, L2-normalized CLIP-frozen L11 features): `0.9909`.

---

## 5. Per-pair KLIEP projection distribution

Aggregates from `kliep_axis_projections.csv` (5,475 rows = 1,825 pair_ids × 3 ckpts):

| ckpt_key | n | mean | std | median | q25 | q75 | p5 | p95 | % |proj|>0.05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P8A_step5000 | 1825 | +0.00518 | 0.01951 | +0.00584 | −0.00517 | +0.01558 | −0.02649 | +0.03710 | 2.2% |
| SlotAv2_step3500 | 1825 | +0.01350 | 0.02611 | +0.00947 | −0.00196 | +0.02463 | −0.01869 | +0.06377 | 9.9% |
| T5C_step3500 | 1825 | +0.01477 | 0.02299 | +0.01243 | +0.00088 | +0.02688 | −0.01648 | +0.05533 | 8.1% |

By inventory source:

| ckpt_key | source | n | mean | median |
|---|---|---:|---:|---:|
| P8A_step5000 | hdtf_visomaster_teams | 1093 | +0.00768 | +0.00724 |
| P8A_step5000 | quickclips_visomaster_teams | 732 | +0.00145 | +0.00198 |
| SlotAv2_step3500 | hdtf_visomaster_teams | 1093 | +0.01382 | +0.00852 |
| SlotAv2_step3500 | quickclips_visomaster_teams | 732 | +0.01303 | +0.01282 |
| T5C_step3500 | hdtf_visomaster_teams | 1093 | +0.01588 | +0.01210 |
| T5C_step3500 | quickclips_visomaster_teams | 732 | +0.01312 | +0.01319 |

(Zero `visomaster_teams_enhanced` rows in the projection table — those 54 pairs were skipped at A0.2 entry because the resolver row lacks a clean-side frame count.)

---

## 6. Direct observations

1. **The 3 ckpts produce near-identical L0/L4/L8 geometry.** `cos_pair`, `cos_within_same`, and `Δ` at L0–L8 differ by < 0.003 across all three ckpts. Divergence appears only at L11. This is consistent with P8A, T5C, T5C-base, and SlotAv2 all sharing the same OpenCLIP backbone with the same shallow-layer SVD residuals applied, and only diverging in their L11 + head training history.

2. **L11 `cos_pair` is essentially identical across the 3 ckpts** (P8A 0.8668, SlotAv2 0.8673, T5C 0.8654 — span = 0.002). `cos_within_same` is also nearly identical (0.940–0.949). The `Δ = −0.077 to −0.082` spread is < 0.008.

3. **`cos_cross_id` diverges across the 3 ckpts** at L11 only (P8A 0.8683, SlotAv2 0.7201, T5C 0.7143). P8A's `cos_cross_id` is ~0.15 higher than the other two; its L11 features cluster more tightly across distinct identities than SlotAv2/T5C. (P8A L11 `cos_cross_id` of 0.868 is approximately equal to its `cos_pair` of 0.867.)

4. **Per-pair KLIEP projections are small in magnitude across all 3 ckpts.** Mean ranges from +0.005 (P8A) to +0.015 (T5C). The p5 and p95 percentiles fall inside ±0.07 for all three. The fraction of pairs with `|proj| > 0.05` runs 2.2 % (P8A), 9.9 % (SlotAv2), 8.1 % (T5C).

5. **KLIEP projection mean has the same sign across all 3 ckpts (positive).** The per-pair difference vector `(teams − clean)` projects, on average, in the same direction as the dev_real → lockbox_real discriminator's positive class (lockbox). Mean magnitudes range +0.005 to +0.015. The same-sign behavior holds across both inventory sources for SlotAv2 and T5C; for P8A, the projection magnitude is much smaller on quickclips (+0.0015) than HDTF (+0.0077).

6. **`score_corr` (Pearson) ranges 0.28–0.43 across the 3 ckpts.** P8A is lowest (0.28), T5C highest (0.43). All three are well below the in-substrate baseline `cos_within_same` of ~0.94 at L11.

7. **`cos_pair` at L11 (~0.87) is below `cos_within_same` at L11 (~0.94) by ~0.07.** It is also approximately equal to `cos_cross_id` for P8A (0.87 vs 0.87) but markedly above `cos_cross_id` for SlotAv2 (0.87 vs 0.72) and T5C (0.87 vs 0.71). That is, at L11 the cross-substrate pair distance is comparable to a different-identity same-substrate sample for P8A but is smaller than a different-identity same-substrate sample for SlotAv2/T5C.

8. **Cross-substrate cluster contraction is present at L4 and L8 already.** `cos_pair < cos_within_same` by ~0.015 at L4 and ~0.016–0.019 at L8 across all 3 ckpts. The L11 gap (~0.077–0.082) is roughly 4–5× the L8 gap and is the dominant single-layer contribution.

9. **The 54 `visomaster_teams_enhanced` pairs are absent from this analysis.** As recorded in `INVENTORY_FACTS_2026-05-22.md` §1.6, the resolver row schema for those pairs lacks `clean_real_frame_count`; the script's `select_pair_frames()` filters them out.

10. **Decode failure rate: 3 frames of 10,956 attempted (0.027 %).** All 3 failures are on the clean side at pair-ids around frame index 855–857 (the OpenCV `imdecode_` assertion on empty buffer indicates an empty / corrupted JPEG blob on GCS). Net impact on per-cell metrics: 2 pair_ids were dropped from the matched-pair set (clean had only `frame_idx ∈ {1, 2}` available for those pair_ids while `frame_idx = 0` was missing); `n_pairs = 1,825` of the 1,826 fully-paired inventory rows entered the L11 cells.

11. **Compute resource ratio of forward vs decode.** Pure forward time per ckpt (10,953 frames both sides, MPS) = ~196 s. Frame-cache build (one-time, reusable across all 3 ckpts) = ~3,400 s. Frame I/O dominated wall time by ~17×. The cache is reusable for any future Phase-0-class probe and need not be rebuilt unless `n_frames` is increased.
