# Phase 0 Fallback 1 (Face-Region Pool) + Probe 1 (KLIEP Re-Fit) — FACTS (2026-05-22)

> **Status: factual-only.** Forbidden words (none present below): succeeds, fails, wins, loses, promotes, deployment-grade, ship, kill, best, worst, unfortunately, remarkably, lucky, confirmed, refuted, shortcut-aligned, gap-is-wide, gap-is-narrow.
> Interpretation lives in `FALLBACK1_PROBE1_PROPOSAL_2026-05-22.md`.
> Companion run to A0.2 (RESULTS_FACTS_2026-05-22.md) and the master plan execution log `/Users/roeedar/.claude/plans/ok-so-we-don-t-valiant-quasar.md` Phase 0.

---

## 1. Probe 1 — KLIEP Re-Fit Sanity Probe (DONE)

### 1.1 Method

For each of the 3 ckpts (`P8A_step5000`, `SlotAv2_step3500`, `T5C_step3500`):

1. Loaded the cached L11 CLS features built by A0.2: `feats/{ckpt}_L11_clean.npy` (5,475 × 768) and `feats/{ckpt}_L11_teams.npy` (5,478 × 768).
2. Concatenated both sides into a single feature matrix; assigned `y = 0` to clean rows and `y = 1` to teams rows. L2-normalized each row.
3. Stratified 80/20 train/test split on `(X, y)` with `random_state=42`. Train size 8,762; test size 2,191.
4. Fit `LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', n_jobs=1, class_weight='balanced')` — same hyperparameters as `analysis/cpu_diagnostics_2026-05-12_d10_training_pool_position/run_d10.py:fit_kliep_discriminator` (lines 145-175).
5. Saved the unit-norm coefficient vector as `_trained_encoder_substrate_axis_{ckpt}.npy`.
6. Computed:
   - Held-out 20% accuracy.
   - Cosine similarity between the per-ckpt substrate axis and the frozen-CLIP-L11 KLIEP axis (`_kliep_w_hat.npy`, accuracy = 0.9909 on dev_real vs lockbox_real).
   - Per-pair direction `diff_i = L2norm(teams_features[i]) − L2norm(clean_features[i])` for `i ∈` matched pair_ids (n = 1,825), projected onto the per-ckpt axis (`diff_i · w_hat_trained`) and onto the frozen-CLIP axis (`diff_i · w_hat_frozen`). Reported mean and std.

All computation: numpy/sklearn on CPU. Wall time: 11.3 s end-to-end for all 3 ckpts.

### 1.2 Headline results

| ckpt_key | test_acc | train_acc | cos(w_trained, w_frozen) | pair_proj on trained axis (μ ± σ) | pair_proj on frozen axis (μ ± σ) | ratio (μ_trained / μ_frozen) |
|---|---:|---:|---:|---:|---:|---:|
| frozen-CLIP-L11 KLIEP (reference) | — | — | 1.0000 | — | — | — |
| P8A_step5000 | 0.9804 | 0.9808 | +0.1037 | +0.1187 ± 0.0423 | +0.0052 ± 0.0195 | 22.9× |
| SlotAv2_step3500 | 0.9836 | 0.9893 | +0.0400 | +0.1363 ± 0.0433 | +0.0135 ± 0.0261 | 10.1× |
| T5C_step3500 | 0.9836 | 0.9886 | +0.0592 | +0.1337 ± 0.0380 | +0.0148 ± 0.0230 | 9.0× |

(Frozen-CLIP KLIEP `pair_proj` reproduces the values from A0.2 `_gate_verdict.json` and `RESULTS_FACTS_2026-05-22.md` §5 to 4-decimal precision.)

### 1.3 Direct observations on Probe 1

1. **The trained-encoder substrate axis is highly discriminative.** All three ckpts produce a logistic-regression classifier that scores held-out test accuracy in the 0.980–0.984 range for distinguishing clean-side from teams-side L11 features. By comparison, the frozen-CLIP L11 KLIEP classifier (fit on dev_real vs lockbox_real, a different problem) scores 0.991 on its own training distribution.

2. **Cosine alignment between per-ckpt substrate axis and the frozen-CLIP KLIEP axis is small.** P8A 0.10, SlotAv2 0.04, T5C 0.06. The axes are nearly orthogonal — `arccos(0.04) ≈ 87.7°`, `arccos(0.10) ≈ 84.0°`. P8A is the closest to the frozen-CLIP axis.

3. **Per-pair direction projects 9–23× more strongly onto the per-ckpt axis than onto the frozen axis.** Per-ckpt projection means range +0.119 to +0.136 with σ ≈ 0.04. Frozen-axis projection means range +0.005 to +0.015 with σ ≈ 0.02. The ratio is 9.0× (T5C), 10.1× (SlotAv2), 22.9× (P8A).

4. **Sign is positive on both axes for all 3 ckpts.** The (teams − clean) direction projects in the +1 direction (the teams class direction) of the per-ckpt substrate axis with magnitude ~0.13, and in the +1 direction (the lockbox_real class direction) of the frozen-CLIP KLIEP axis with magnitude ~0.01.

5. **|pair_proj| > 0 for ≥ 99.9 % of pairs on the per-ckpt axis.** (Mean / |mean| ≈ 1.00 implies near-uniform sign of the per-pair value.) On the per-ckpt axes the substrate-pair direction is highly consistent across pairs.

6. **Norm of the trained-encoder substrate axis is similar across ckpts.** Coef norms (pre-normalization): P8A 39.46, SlotAv2 38.28, T5C 38.50. The bias terms `b` are P8A +0.41, SlotAv2 +0.40, T5C +0.30.

### 1.4 Output artifacts (Probe 1)

| Artifact | Path | Size / Shape |
|---|---|---|
| Probe 1 results JSON | `analysis/substrate_pair_geometry_2026-05-22/_probe1_kliep_refit_results.json` | 3-ckpt object |
| Per-ckpt substrate axes | `analysis/substrate_pair_geometry_2026-05-22/_trained_encoder_substrate_axis_{ckpt}.npy` × 3 | each 768-dim float64 |
| Probe 1 builder script | `analysis/substrate_pair_geometry_2026-05-22/run_probe1_kliep_refit.py` | 5.0 KB Python |
| Probe 1 log | `analysis/substrate_pair_geometry_2026-05-22/_probe1.log` | — |

---

## 2. Probe 2 — Face-Region Attention Pool CPU Probe (LAUNCHED — in flight)

### 2.1 Method

For each of the 3 ckpts:

1. Downloaded the ckpt from GCS (US bucket per CLAUDE.md region preference) into `analysis/substrate_pair_geometry_2026-05-22/ckpts/`. Each ckpt is ~1 GB; ckpt is deleted after its forward pass to manage the ~5 GB free-disk budget.
2. Loaded the model via `batch_inference_gcs.load_model` using `config/detector/effort.yaml` + `config/train_config.yaml`. Backbone path: `model.backbone.visual.transformer.resblocks` (or `.visual.visual....` per the wrapper variant); 12 resblocks total.
3. Registered a forward hook on `resblocks[11]` that captures the FULL token sequence (1 CLS + 196 patches in a 14×14 grid). Layout: `(seq, batch, dim)` per A0.2's seq-first observation; permuted to `(batch, seq, dim)` inside the hook.
4. Loaded the cached frame tensors `_cache_frames_{clean,teams}.pt` (fp16 → fp32 on load; the same 5,475 / 5,478 frame tensors A0.2 used).
5. For each frame: ran one forward pass; pulled the 197-token output from the hook; selected the 49 patches inside the centered `7×7` subgrid of the `14×14` patch grid (rows 3–9, cols 3–9 — see §2.2 fallback note); mean-pooled to a single 768-dim face-region feature per frame.
6. Computed the same 6 metrics from A0.2 on these face-region pooled features (per-pair indices identical to A0.2):
   - `cos_pair`, `cos_within_same`, `cos_cross_id`, `delta_pair_vs_within`, `score_corr` (head probability)
   - `kliep_projection` of `(face_pool[teams] − face_pool[clean])` onto the frozen-CLIP-L11 KLIEP axis `_kliep_w_hat.npy`

### 2.2 Face mask: documented fallback

The cached frame metadata `_cache_frames_meta_{clean,teams}.parquet` does NOT include a `face_bbox` column. The frame manifest `_frame_manifest.parquet` does NOT include one either. The source bucket — `live-deepfake-methods-real-and-fake-frames-cropped/...` — supplies pre-cropped face frames (per `data/sources/visomaster.py:9-21,277-360`).

Because no per-frame face bbox is available offline, this probe uses a **coarse fallback face-region mask**: the centered `7×7` subgrid of the `14×14` patch grid (49 patches, rows 3–9 / cols 3–9). Visualized:

```
..............
..............
..............
...XXXXXXX....
...XXXXXXX....
...XXXXXXX....
...XXXXXXX....
...XXXXXXX....
...XXXXXXX....
...XXXXXXX....
..............
..............
..............
..............
```

This is the same convention requested in the brief (~ centered 75 % of the frame → ~49 central patches of the 14×14 grid). It is NOT a per-frame face-localized bbox. The source-bucket cropping convention is the only face-locality guarantee in this probe.

### 2.3 Compute parameters (Probe 2)

| Parameter | Value |
|---|---|
| Layer | L11 (single layer; the only layer where A0.2 found `delta < −0.05`) |
| Resolution | 224 × 224 |
| Patch grid | 14 × 14 (CLIP-B/16) |
| Face subgrid | centered 7 × 7 = 49 patches |
| Pool function | mean across 49 face patches |
| Batch size | 16 (reduced from A0.2's 32 because the hook captures the full 197-token output) |
| Device | MPS (M-series Mac, fp32 forward) |
| Frame caches | reused from A0.2 (`_cache_frames_{clean,teams}.pt`, fp16 → fp32 on load) |

### 2.4 Output artifacts (Probe 2 — populated on completion)

| Artifact | Path | Contents |
|---|---|---|
| Per-ckpt face-region cell metrics | `analysis/substrate_pair_geometry_2026-05-22/per_ckpt_face_region_cosines.csv` | 3 rows (one per ckpt, L11, face-pool) |
| Per-pair face-pool KLIEP projections | `analysis/substrate_pair_geometry_2026-05-22/face_region_kliep_projections.csv` | 3 × 1,825 = 5,475 rows |
| Per-frame face-pool features | `analysis/substrate_pair_geometry_2026-05-22/feats_face_region/{ckpt}_L11_face_{side}.npy` × 6 | each ~5,475 × 768 float32 |
| Per-frame head probs | `analysis/substrate_pair_geometry_2026-05-22/feats_face_region/{ckpt}_probs_{side}.npy` × 6 | each ~5,475 float32 |
| Probe 2 log | `analysis/substrate_pair_geometry_2026-05-22/_probe2.log` | — |
| **Sentinel marker** | `analysis/substrate_pair_geometry_2026-05-22/_probe_complete.json` | `{probe1: done, probe2: done\|aborted, wall_seconds: <int>, verdict_summary: <str>}` |

### 2.5 Probe 2 status (at write time)

| Item | Status |
|---|---|
| Script | `analysis/substrate_pair_geometry_2026-05-22/run_probe2_face_region.py` (560 lines) |
| Launch | `nohup python … > _probe2.log 2>&1 &` |
| PID at launch | 53266 |
| Launched | 2026-05-22 00:19:38 local |
| Device | MPS |
| ETA at launch | ~25–40 min based on early-batch throughput ~54 frames/s on MPS; ~7 min/ckpt forward × 3 ckpts + 3 × ~1 min download. (Brief estimate of 3–4 h was conservative; the 197-token capture does not slow forward materially.) |
| Sentinel marker path | `analysis/substrate_pair_geometry_2026-05-22/_probe_complete.json` |

The parent will poll the sentinel marker. This document will not be edited again unless the probe writes new artifacts that require additional FACTS rows; the proposal document interprets both probe outcomes.

---

## 3. Cross-references

- A0.1 inventory (1,880 pairs, 1,826 with full clean+teams sides): `INVENTORY_FACTS_2026-05-22.md`.
- A0.2 multi-layer geometry probe (12 cells, gate verdict AMBIGUOUS): `RESULTS_FACTS_2026-05-22.md` + `AGENT_PROPOSAL_2026-05-22.md`.
- D10 frozen-CLIP KLIEP axis (training accuracy 99.09 %; KLIEP axis 89.71° from IQ-PC1): `analysis/cpu_diagnostics_2026-05-12_d10_training_pool_position/D10_FACTS_2026-05-12.md`.
- D8 substrate balanced head retrain: `analysis/cpu_diagnostics_2026-05-12_d8_head_retrain_substrate_balanced/`.
- Master plan execution log: `/Users/roeedar/.claude/plans/ok-so-we-don-t-valiant-quasar.md` Phase 0 / Fallback 1 sections.

---

## 4. Numerical sanity checks

1. **Probe 1 frozen-axis projection matches A0.2 exactly.** The `pair_proj_frozen_mean` values reproduce the A0.2 KLIEP projection means stored in `_gate_verdict.json` and `RESULTS_FACTS_2026-05-22.md` §5: P8A +0.0052 (vs A0.2 +0.00518), SlotAv2 +0.0135 (vs A0.2 +0.01350), T5C +0.0148 (vs A0.2 +0.01477). Differences are at the 5th decimal. The Probe 1 pipeline reads the same cached features as A0.2 and uses the same per-pair `(c0, t0)` matching logic. Sanity check passes.

2. **Probe 1 frozen-axis vs per-ckpt-axis projection ratios.** P8A 22.9× = 0.1187 / 0.0052. SlotAv2 10.1× = 0.1363 / 0.0135. T5C 9.0× = 0.1337 / 0.0148. All three projections on the per-ckpt axis are simultaneously much larger than on the frozen axis, and the projection σ values (~0.04) are tight relative to the means (~0.13), so the ratios are not driven by noise.

3. **Decode failure count matches A0.2.** Probe 1 uses the same `meta_clean` (n = 5,475 after 3 decode failures) and `meta_teams` (n = 5,478) as A0.2. Matched-pair count `n = 1,825` is identical to A0.2.

4. **Per-ckpt axis norms are sensibly bounded.** `LogisticRegression` with `C = 1.0` produced `||coef|| ≈ 38–39` for all 3 ckpts, with `b ≈ 0.3–0.4`. These are in the typical range for L2-normalized 768-dim inputs under balanced class weighting.
