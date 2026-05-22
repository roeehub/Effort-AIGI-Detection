# Face-Pool Canary Follow-Up — Slot A v2 step3500 — FACTS (2026-05-22)

> **Status: factual-only.** Forbidden words (none present below): succeeds, fails, wins, loses, promotes, deployment-grade, ship, kill, best, worst, unfortunately, remarkably, lucky, confirmed, refuted, shortcut-aligned, gap-is-wide, gap-is-narrow.
> Numbers + tables + cross-references only. Interpretation lives in `AGENT_PROPOSAL_2026-05-22.md`.

---

## 1. Method

1. Reused the 800-frame canary tensor `analysis/manual_canary_2026-05-20/frames.pt` (shape `(800, 3, 224, 224)`, CLIP-normalized) and the per-frame metadata `analysis/manual_canary_2026-05-20/frames_meta.parquet` (labels: 600 real, 200 fake; cohorts include `lockbox_fake` and `chronic_*`).
2. Loaded checkpoint `analysis/manual_canary_2026-05-20/ckpts/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth` (Slot A v2 step3500, gs URI `gs://training-job-outputs/best_checkpoints/hp35c51p/...`) via `batch_inference_gcs.load_model` against `config/detector/effort.yaml` + `config/train_config.yaml`. Backbone resolved to OpenCLIP `ViT-B-16-DataComp-XL`, hidden_size 512 (post-projection), patch grid 14×14, transformer width 768.
3. Installed a forward hook on `model.backbone.visual.transformer.resblocks[11]` that captures the full 197-token sequence `(B, 197, 768)` per batch (handles seq-first layout via permute when `output.shape[0] >= output.shape[1]`). Indexing convention matches `analysis/substrate_pair_geometry_2026-05-22/run_probe2_face_region.py`.
4. Monkey-patched `model.backbone.forward` for the duration of inference. The patched forward runs the original (which fires the hook), discards the CLS-pool result, then computes the face-pool readout:
   - Drop the CLS token (`tokens[:, 1:, :]`, shape `(B, 196, 768)`).
   - `index_select` the centered 7×7 patch subgrid = 49 face-region patches (mask: rows {3..9} × cols {3..9} of the 14×14 grid).
   - `mean(dim=1)` → `(B, 768)`.
   - Apply the same post-block ops the CLS path uses: `visual.ln_post(...)` then matmul `... @ visual.proj` (`(768,) @ (768, 512) → (512,)`).
   - Return as `{"pooler_output": face_pool}` to the unchanged ArcFace head (`in_features=512`).
5. Hook + monkey-patch are restored on `__exit__`; production code (`detectors/effort_detector.py`, `batch_inference_gcs.py`) was not modified.
6. Forwarded all 800 canary frames at `batch_size=16` on MPS in `torch.no_grad()`, recorded `pred["prob"]` (softmax fake-class). 14.0 s total forward time (after a 13.5 s model load).
7. Aggregated metrics by re-importing `aggregate_metrics` verbatim from `analysis/manual_canary_2026-05-20/score_canary.py` (no edits) so the FPR-5 % / FPR-10 % τ grids and per-identity means are computed identically to the baseline.

**Compute environment**: macOS Darwin 24.3.0, Python 3 (`pyenv shim`), torch 2.11.0 + MPS, 32 GB RAM. No CUDA, no GPU job, no GCS download (ckpt already cached from 2026-05-20). Total wall time: 28 s (load + score).

---

## 2. Output artifacts

| Artifact | Path | Size / Shape |
|---|---|---|
| Face-pool aggregate metrics | `analysis/face_pool_canary_2026-05-22/outputs/SLOT_A_V2_STEP3500_face_pool.json` | 34-key object |
| Face-pool per-frame probs | `analysis/face_pool_canary_2026-05-22/outputs/SLOT_A_V2_STEP3500_face_pool.scores.npy` | `(800,)` float64 |
| Completion sentinel | `analysis/face_pool_canary_2026-05-22/outputs/_done.json` | 3-key object |
| Scorer source | `analysis/face_pool_canary_2026-05-22/score_canary_face_pool.py` | 233 lines |
| CLS baseline (reference) | `analysis/manual_canary_2026-05-20/outputs/SLOT_A_V2_STEP3500.json` | 33-key object |
| CLS baseline per-frame probs | `analysis/manual_canary_2026-05-20/outputs/SLOT_A_V2_STEP3500.scores.npy` | `(800,)` float64 |

---

## 3. Headline metric table — CLS pool vs face pool

Same ckpt (Slot A v2 step3500), same 800-frame canary, same `aggregate_metrics` function. Only the pooling op feeding the classifier changes.

| Metric | CLS pool (baseline) | Face pool (new) | Δ (face − CLS) |
|---|---:|---:|---:|
| `lockbox_recall_at_FPR_5pct` | 0.5900 | 0.5500 | **−0.0400** |
| `lockbox_recall_at_FPR_10pct` | 0.6100 | 0.6700 | **+0.0600** |
| `max_per_identity_mean_score` (over 6 chronic ids; lower = less over-fire) | 0.8559 | 0.7552 | **−0.1006** |
| `mean_per_identity_mean_score` (over 6 chronic ids) | 0.5298 | 0.6624 | **+0.1326** |
| `score_p50_on_reals` | 0.1750 | 0.5676 | +0.3926 |
| `score_p95_on_reals` | 0.8898 | 0.7879 | −0.1020 |
| `score_mean_on_reals` | 0.3421 | 0.5511 | +0.2090 |
| `score_std_on_reals` | 0.3053 | 0.1592 | −0.1461 |
| `score_p50_on_fakes` | 0.8358 | 0.7430 | −0.0928 |
| `score_mean_on_fakes` | 0.7511 | 0.7217 | −0.0294 |
| `score_p05_on_fakes` | 0.2819 | 0.5055 | +0.2237 |
| `recall_at_tau05/lockbox_fake` | 0.9100 | 1.0000 | +0.0900 |
| `recall_at_tau05/viso_fake` | 0.6600 | 0.8200 | +0.1600 |
| `recall_at_tau05/deeplive_fake` | 1.0000 | 1.0000 | +0.0000 |
| `lockbox_tau_at_FPR_10pct` | 0.8740 | 0.7561 | −0.1178 |
| `wilcoxon_pval_vs_p8a_reals` | 1.9 × 10⁻⁴ | 1.1 × 10⁻⁴¹ | (no longer comparable to P8A under same calibration) |
| `mean_score_drift_vs_p8a_reals` | −0.0047 | +0.2042 | +0.2089 |
| `abs_mean_score_drift_vs_p8a_reals` | 0.2521 | 0.3589 | +0.1068 |

Frame count, real / fake split, and seed are identical between rows.

---

## 4. Per-chronic-identity table — mean real-frame score (over the 800-frame canary's `chronic_*` cohorts)

Lower = less over-fire on the chronic-identity real frames (the substrate-shortcut signal Slot A v2 was designed to attack). Each entry is the mean head probability over the canary frames where `cohort.startswith("chronic_")` and `label == 0` for that identity.

| Chronic identity | CLS pool (baseline) | Face pool (new) | Δ (face − CLS) |
|---|---:|---:|---:|
| `Roy_D` | 0.8559 | 0.7552 | **−0.1006** |
| `PC_Generator__s22` | 0.4696 | 0.6527 | **+0.1831** |
| `PC_Generator__s45` | 0.6949 | 0.7149 | +0.0199 |
| `Q__s6` | 0.2522 | 0.6188 | **+0.3666** |
| `bla_bla_chow` | 0.4072 | 0.6406 | **+0.2334** |
| `bla_bla_chow__s2` | 0.4989 | 0.5921 | +0.0932 |
| (1-of-6 below baseline) | — | — | — |
| (5-of-6 above baseline) | — | — | — |

---

## 5. Score-distribution table

Distribution of per-frame head probability across the 800 canary frames, broken out by `label`.

| Statistic | Reals (n=600), CLS | Reals (n=600), Face | Fakes (n=200), CLS | Fakes (n=200), Face |
|---|---:|---:|---:|---:|
| min | 0.0497 | 0.4123 | 0.1062 | 0.4848 |
| p05 | 0.0626 | 0.4322 | 0.2819 | 0.5055 |
| p25 | 0.0822 | 0.4498 | 0.5946 | 0.6464 |
| p50 (median) | 0.1750 | 0.5676 | 0.8358 | 0.7430 |
| p75 | 0.6021 | 0.6840 | 0.8862 | 0.7980 |
| p95 | 0.8898 | 0.7879 | 0.9226 | 0.8474 |
| max | 0.9450 | 0.8262 | 0.9417 | 0.8758 |
| mean | 0.3421 | 0.5511 | 0.7511 | 0.7217 |
| std | 0.3053 | 0.1592 | 0.1772 | 0.0784 |

`p05/p25/p75` values for the face row computed below (CSV in tail of `_done.json`'s sibling `.scores.npy` array; numbers above derived live from `numpy.percentile`).

---

## 6. Operating-point τ grid (faces of contract calibration)

The contract τ is the largest threshold τ such that `(real_scores > τ).mean() ≤ FPR_target`. The face-pool τ moves dramatically because the real-score distribution shifts upward (mean 0.34 → 0.55, std 0.31 → 0.16).

| FPR target | CLS τ | Face τ | CLS lockbox-recall at τ | Face lockbox-recall at τ |
|---:|---:|---:|---:|---:|
| 5 % | not reported in JSON | not reported in JSON | 0.5900 | 0.5500 |
| 10 % | 0.8740 | 0.7561 | 0.6100 | 0.6700 |

The `recall_at_tau05` rows (which use the un-calibrated fixed threshold τ = 0.5) are NOT contract metrics — they read the absolute probability and so move purely because the face-pool real-distribution shifts upward; they are reported here for completeness only.

---

## 7. Cross-reference

- Probe 2 representation-geometry (Slot A v2 step3500): `analysis/substrate_pair_geometry_2026-05-22/per_ckpt_face_region_cosines.csv`
  - `cos_pair`: CLS 0.8673 → Face 0.9626 (+0.0953)
  - `delta_pair_vs_within`: CLS −0.0774 → Face −0.0160 (+0.0614)
  - KLIEP μ (frozen-CLIP substrate axis): CLS +0.0135 → Face +0.0019 (−0.0116)
- Baseline ckpt panel comparison (CLS pool only): `analysis/manual_canary_2026-05-20/outputs/summary.csv` — Slot A v2 step3500 is the rank-1 entry on the canary contract metrics.
- Mechanism note: per chronic-identity pool composition is documented in `arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet`. The 6 chronic identities each contribute `n` real frames per identity (n=20–50 depending on identity); the canary fakes are `lockbox_fake` cohort frames spanning the same substrate transports.

---

## 8. Reproducibility

```bash
cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training
python analysis/face_pool_canary_2026-05-22/score_canary_face_pool.py \
    --output-dir analysis/face_pool_canary_2026-05-22/outputs \
    --ckpt analysis/manual_canary_2026-05-20/ckpts/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth
```

- Single CKPT, MPS, no network, deterministic (the only stochastic element is `aggregate_metrics`'s τ-grid construction which is a fixed-grid `np.linspace(0, 100, 401)`).
- Wall time on M-series Mac, 32 GB RAM: 28 s total (13.5 s load + 14.0 s score + ~0.5 s metrics).
