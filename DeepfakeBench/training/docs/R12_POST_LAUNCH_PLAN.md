# R12 Phase 2 — Post-Launch Improvements Plan

**Date:** March 8, 2026  
**Status:** Planning → Implementation  
**Predecessor:** R12 A/B/C/D launched (aug fixes, CCT, timeout, SVD capacity ablation)

---

## Background

### The Core Problem

Our EFFORT detector achieves **AUC=0.9942** on the holdout validation set (R9_D, W&B `m7etxxnp`), yet
production deployment on Microsoft Teams video calls suffers from **false positives on real faces** and
**score instability** (±10pp frame-to-frame swings on identical participants).

The root cause is a **domain gap** between training/validation data and the production environment.
Evidence:

| Property | Training Reals | Production (Teams) | Gap |
|---|---|---|---|
| Mean brightness | 104 | 160 | +54% |
| R/B channel ratio | 1.4–1.6 | ~1.2 | Different white balance |
| Contrast (std) | Higher | Lower (compressed) | Codec flattening |
| EER threshold | ~0.45 | ~0.77 | **Probability space not calibrated** |

The threshold gap — documented extensively in
[PHASE2_SUMMARY_R8_R11.md](experiments/PHASE2_SUMMARY_R8_R11.md) and
[R9_PLAN.md](experiments/phase2_round9/R9_PLAN.md) §1 — means the model's
decision boundary learned from clean training data doesn't transfer to
production conditions.  The lighting analysis in
[docs/Simulating Varied Indoor Lighting for Robust Real‑vs‑Fake Face Classification.pdf](docs/Simulating%20Varied%20Indoor%20Lighting%20for%20Robust%20Real‑vs‑Fake%20Face%20Classification.pdf)
and [docs/LIGHTING_ROBUSTNESS_REPORT.md](docs/LIGHTING_ROBUSTNESS_REPORT.md) confirms
that brightness, colour temperature, and contrast are the dominant confounds.

### What R12 Already Addresses (Launched)

The R12 experiments (launched tonight) tackle the **augmentation side** of the gap:

- **OneOf removal:** Context variation transforms (gamma, brightness, shift, CCT) now fire
  independently instead of mutually exclusively.  
  Changed in [data/augmentations/pipelines.py](data/augmentations/pipelines.py) `_build_context_variation_block()`.
- **ColorTemperatureShift:** New transform simulating 2700–8000K white balance via Kelvin→RGB gains.  
  Added to [data/augmentations/transforms.py](data/augmentations/transforms.py).
- **Asymmetric brightness** (-0.20, +0.60) to cover bright webcam faces (mean ~160).
- **SVD capacity sweep:** R12_B (k=64) and R12_D (k=128) test whether
  32 trainable singular directions per layer is a bottleneck.
- **Vertex AI timeout:** 24h → 72h so scratch runs (30K steps, ~40h) don't get killed.

### What R12 Does NOT Fix (This Plan)

Five structural issues remain that cannot be solved by augmentation alone:

1. **Checkpointing is blind to OOD metrics** — best checkpoint is selected on holdout AUC only
2. **No post-hoc calibration in production** — raw probabilities are used with a fixed threshold
3. **Quality domain head (GRL) is broken** — domain labels never propagated from data sources
4. **No test-time augmentation (TTA)** — single forward pass per frame
5. **Teams holdout set is too small** — ~14 videos, not statistically meaningful

---

## Relevant Files Reference

| File | Role |
|---|---|
| [trainer/trainer.py](trainer/trainer.py) | Main trainer with mixin composition, validation flow, OOD monitoring |
| [trainer/mixins/checkpointing.py](trainer/mixins/checkpointing.py) | `init_checkpointing()`, `save_ckpt()`, top-N logic |
| [trainer/mixins/validation.py](trainer/mixins/validation.py) | Validation evaluation mixin |
| [detectors/effort_detector.py](detectors/effort_detector.py) | `EffortDetector`, `SVDResidualLinear`, `QualityDomainHead`, `GradientReversalLayer` |
| [app4.py](app4.py) | FastAPI inference endpoint (v4) |
| [wma/server.py](wma/server.py) | WMA inference server |
| [run_r8_calibration_fit.py](run_r8_calibration_fit.py) | Offline Platt scaling + isotonic calibration pipeline |
| [run_r8_score_manifest.py](run_r8_score_manifest.py) | Score manifest with checkpoint |
| [launch_r8_calibration.sh](launch_r8_calibration.sh) | Calibration pipeline orchestrator |
| [data/sources/](data/sources/) | Data source implementations (none propagate `quality_domain` labels) |
| [docs/LIGHTING_ROBUSTNESS_REPORT.md](docs/LIGHTING_ROBUSTNESS_REPORT.md) | Production lighting analysis |
| [docs/Simulating Varied Indoor Lighting...pdf](docs/Simulating%20Varied%20Indoor%20Lighting%20for%20Robust%20Real‑vs‑Fake%20Face%20Classification.pdf) | Lighting simulation paper |
| [experiments/PHASE2_SUMMARY_R8_R11.md](experiments/PHASE2_SUMMARY_R8_R11.md) | R8–R11 experiment summary |
| [experiments/WINNING_RUNS_REGISTRY.md](experiments/WINNING_RUNS_REGISTRY.md) | All winning run records |
| [experiments/phase2_round6/R6_EXPERIMENT_REPORT.md](experiments/phase2_round6/R6_EXPERIMENT_REPORT.md) | R6 report (GRL bug documented) |

---

## Task 1: OOD-Inclusive Checkpointing

### Why This Is #1

The trainer currently saves checkpoints **only** when `val_holdout` AUC improves
([trainer.py L1926–1975](trainer/trainer.py)).  OOD metrics (computed in
`ood_monitoring_epoch()` at [trainer.py L2169–2292](trainer/trainer.py)) are
logged to W&B but **never influence which checkpoint is kept**.

This means:
- The "best" checkpoint maximizes in-distribution performance
- OOD performance (which correlates with production) can regress silently
- The EER threshold gap (0.45 in-dist vs 0.77 OOD) can widen without penalty

### Current Architecture

```
_run_validation(epoch, step_cnt)
  ├── test_epoch(..., val_in_dist_loader, is_primary_metric=False)     # logs only
  ├── test_epoch(..., val_holdout_loader, is_primary_metric=True)      # triggers save_ckpt
  ├── compute at-indist-threshold metrics for holdout
  ├── compute unified threshold
  ├── check lesson gate
  └── _run_ood_monitoring(epoch, step_cnt, indist_threshold)           # logs only, no save
```

Key state variables (in [checkpointing.py L34–41](trainer/mixins/checkpointing.py)):
```python
self.top_n_checkpoints: List[Dict] = []
self.top_n_size = 6
self.best_val_metric = -1.0   # tracks holdout AUC
self.best_val_epoch = -1
```

### Implementation Plan

**Goal:** Keep a *separate* top-N checkpoint list ranked by a composite OOD-aware metric,
alongside the existing holdout-only list (so we don't break anything).

- [ ] **1a.** Add `self.best_ood_composite = -1.0` and `self.top_n_ood_checkpoints = []` to `init_checkpointing()` in [checkpointing.py](trainer/mixins/checkpointing.py)
- [ ] **1b.** Make `ood_monitoring_epoch()` in [trainer.py](trainer/trainer.py) **return** the overall OOD AUC (currently returns nothing)
- [ ] **1c.** In `_run_ood_monitoring()`, capture the returned OOD AUC and pass it back to `_run_validation()`
- [ ] **1d.** In `_run_validation()`, after OOD monitoring, compute `composite = hmean(holdout_auc, ood_auc)` (harmonic mean — penalizes large gaps). If composite > `self.best_ood_composite`, save an `ood_best` checkpoint using the existing `save_ckpt()` method with a new prefix.
- [ ] **1e.** Add config knob `checkpointing.ood_composite_enabled: true` (default: `false` for backward compatibility) and `checkpointing.ood_composite_weight: 0.5` (blend parameter)
- [ ] **1f.** Log the composite metric to W&B as `val_primary/ood_composite`
- [ ] **1g.** Add the new config keys to R12 experiment YAMLs

### Design Decisions

- **Harmonic mean** (not arithmetic) because it penalizes cases where one metric
  is high and the other is low — exactly the failure mode we're fixing.
- **Separate checkpoint list:** The holdout-only list remains untouched so we don't
  break early stopping or existing ablation logic.
- **Same `save_ckpt()` method** with a different prefix (`ood_composite` vs `top_n`)
  so GCS cleanup and top-N eviction work identically.
- **Only triggers when OOD actually runs** (every `ood_monitoring_every_steps`).
  On steps where OOD doesn't run, the composite metric is not updated.

---

## Task 2: Wire Calibration into Production Inference

### Why

The R8 calibration pipeline ([run_r8_calibration_fit.py](run_r8_calibration_fit.py))
fits Platt scaling and isotonic regression on scored data, producing a
`calibrator_bundle.json`.  But [app4.py](app4.py) uses raw probabilities
with a hardcoded threshold — the calibrator is never loaded or applied.

After R12 checkpoints are produced, we need:
1. Run the calibration pipeline on the best R12 checkpoint
2. Load the calibrator in `app4.py` at startup
3. Apply calibration to raw model probabilities before thresholding

### Implementation Plan

- [ ] **2a.** Add a `--calibrator` CLI arg / env var (`CALIBRATOR_GCS_PATH`) to `app4.py` that loads a `calibrator_bundle.json` at startup
- [ ] **2b.** Add a `calibrate(raw_prob)` function that applies the loaded Platt/isotonic transform
- [ ] **2c.** Wire `calibrate()` into both `/check_frame` and `/check_video` endpoints, before thresholding
- [ ] **2d.** Log both raw and calibrated probabilities in responses (backward-compatible: add `calibrated_prob` field, keep `fake_prob` as raw)
- [ ] **2e.** Update the default threshold to be calibration-aware (if calibrator loaded, use the calibrator's optimal threshold from the bundle)

### Notes

- The calibrator bundle format (from `run_r8_calibration_fit.py`) contains
  `{'platt': {'a': float, 'b': float}, 'isotonic': <sklearn object>, 'optimal_threshold': float}`
- We'll use **Platt scaling** by default (deterministic, fast, no sklearn dependency at inference)
- This is a post-R12-training task — needs a checkpoint to score against

---

## Task 3: Fix GRL Quality Domain Head Label Propagation

### Why

The `QualityDomainHead` ([effort_detector.py L236–272](detectors/effort_detector.py))
implements gradient reversal for domain-invariant features.  It was designed to encourage
the CLIP backbone to produce features that are useful for real/fake classification but
**not** useful for distinguishing data-source domains (DF40 vs DeepLive vs Teams vs VCD).

However, as documented in [R6_EXPERIMENT_REPORT.md](experiments/phase2_round6/R6_EXPERIMENT_REPORT.md):
> "GRL was broken throughout R6 (quality_domain_loss = 0 due to label propagation bug)"

**Root cause:** The data sources in `data/sources/` never populate `quality_domain` in the
data dict.  The `QualityDomainHead.DOMAIN_MAP` expects:
```python
{"df40": 0, "external": 1, "deeplive": 2, "visomaster": 2, "youtube": 3}
```
But no data source creates this key.  Since R6, the fix was to add
`quality_domain_require_labels: true` which raises a RuntimeError if labels
are missing — effectively keeping the head disabled.

> **UPDATE (implementation):** On investigation, `quality_domain` label propagation was
> *already fixed* since R6 — all 11 yield points in `combined_paired.py` propagate
> `quality_domain` via `_quality_domain_for_source()` using `QUALITY_DOMAIN_MAP`.
> The collate function outputs it as `torch.long`. Comprehensive unit tests exist in
> `test_unpaired_reals_and_grl.py`. The detector's `DOMAIN_MAP` was synced to include
> `"deeplive_teams": 1` (webcam_codec domain). The head simply needs to be **enabled**
> via config (`use_quality_domain_head: true`), which R12_E does.

### Implementation Plan

- [ ] **3a.** In the combined_paired data source, add `quality_domain` label to each sample based on the source family/method string. Map using the existing `DOMAIN_MAP`.
- [ ] **3b.** Verify the label flows through the dataloader collation into the batch `data_dict`
- [ ] **3c.** Add a unit test that a combined_paired batch contains `quality_domain` with correct shape and range
- [ ] **3d.** Enable the head in one R12 YAML (or create an R12_E variant) with `use_quality_domain_head: true` and `quality_domain_loss_weight: 0.1`
- [ ] **3e.** Verify GRL lambda annealing works ([trainer.py L231–248](trainer/trainer.py) `_update_quality_domain_lambda()`)

### Risk

Low — the GRL head is architecturally complete and loss computation is implemented.
The only missing piece is the data-source label propagation.  If this degrades
holdout AUC, we simply set `quality_domain_loss_weight: 0.0` and the head
becomes a no-op.

---

## Task 4: Test-Time Augmentation (TTA) in Inference

### Why

A single forward pass per frame means the model's prediction is sensitive to:
- Exact face crop coordinates (YOLO jitter)
- JPEG compression artifacts
- Sub-pixel alignment differences

TTA (horizontal flip + average) is a well-known technique that:
- Reduces score variance by ~30% (from literature on face recognition)
- Often gives ~1–2pp AUC boost for free
- Is especially effective when the model has learned asymmetric features

### Implementation Plan

- [ ] **4a.** Add a `tta_inference()` utility function that takes `image_tensor [B, C, H, W]`, runs the model on `[original, horizontally_flipped]`, and returns the averaged probability
- [ ] **4b.** Add `--tta` / `TTA_ENABLED` env var to `app4.py`
- [ ] **4c.** Wire TTA into `/check_frame` endpoint
- [ ] **4d.** Wire TTA into the video inference path (per-frame TTA before temporal pooling)
- [ ] **4e.** Benchmark latency impact (should be ~2× per frame — acceptable for our use case since inference is not real-time-critical)

### Design

```python
def tta_inference(model, image_tensor, device):
    """TTA: original + horizontal flip, average probabilities."""
    flipped = torch.flip(image_tensor, dims=[-1])  # flip W dimension
    batch = torch.cat([image_tensor, flipped], dim=0)
    with torch.inference_mode():
        preds = model({'image': batch}, inference=True)
    probs = preds['prob'].view(2, -1).mean(dim=0)  # average over augmentations
    return probs
```

---

## Task 5: Expand Teams OOD Holdout

### Why

The Teams data currently goes into the **same combined_paired pool** as all
other data, split by identity at 85/10/5%.  With ~1300 video pairs in
`live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`, only ~65
videos end up in validation and ~65 in test — and they're mixed into the
holdout set with DF40, DeepLive, etc.

There is **no dedicated Teams OOD evaluation**.  The OOD set currently contains
`external_youtube_avspeech`, `zoom_vcd_real`, and `wma_failure_fake` — none
from Teams.

For statistically meaningful production-domain metrics, we need a dedicated
Teams OOD holdout of 200+ videos.

### Implementation Plan

- [ ] **5a.** Add a `teams_ood` external source to the `ood_monitoring` config in experiment YAMLs, pointing to the Teams v2 bucket with a held-out identity split
- [ ] **5b.** Verify the OOD loader picks up the Teams data and logs per-method metrics with a `teams_*` prefix
- [ ] **5c.** Ensure Teams OOD identities are excluded from training (the `exclude_training_identities` flag should handle this)
- [ ] **5d.** Update R12 YAMLs (if still running, update via W&B config override; otherwise create R12.1 configs)

---

## Implementation Order & Priority

| # | Task | Impact | Effort | Depends On |
|---|------|--------|--------|------------|
| **1** | OOD-inclusive checkpointing | 🔴 High | ~1.5h | None |
| **2** | Calibration in production | 🔴 High | ~1h | R12 checkpoint |
| **3** | Fix GRL domain head | 🟡 Medium | ~1h | Data source code |
| **4** | TTA in inference | 🟡 Medium | ~30min | None |
| **5** | Teams OOD holdout | 🟢 Low | ~30min | Teams v2 bucket |

Tasks 1 and 4 are independent and can be done right now.  
Task 2 requires a trained R12 checkpoint (wait ~22h for R12_C).  
Task 3 requires understanding the combined_paired data source internals.  
Task 5 is a config change.

---

## Checklist

- [x] Task 1a: Add OOD composite state to `init_checkpointing()`
- [x] Task 1b: Return OOD AUC from `ood_monitoring_epoch()`
- [x] Task 1c: Capture OOD AUC in `_run_ood_monitoring()`
- [x] Task 1d: Compute composite + save in `_run_validation()`
- [x] Task 1e: Add config knobs
- [x] Task 1f: Log composite to W&B
- [x] Task 1g: Update R12 YAMLs
- [x] Task 2a: Add calibrator loading to `app4.py` — `PlattCalibrator` class + `CALIBRATOR_GCS_PATH`/`CALIBRATOR_LOCAL_PATH` env vars
- [x] Task 2b: Add `calibrate()` function — Platt scaling: `sigmoid(a * logit(p) + b)`
- [x] Task 2c: Wire into endpoints — `/check_frame`, `/check_video`, `/check_video_from_gcp`
- [x] Task 2d: Add calibrated_prob to responses — `FrameInferResponse.calibrated_prob`, `MultiDecisionResponse.calibrated_frame_probs`
- [x] Task 2e: Calibration-aware threshold — uses `calibrator.optimal_threshold` when loaded
- [x] Task 3a: ~~Add `quality_domain` to data sources~~ — **Already implemented**: all 11 yield points in `combined_paired.py` propagate `quality_domain` via `_quality_domain_for_source()`. Collate handles it at L1949. Fixed since R6.
- [x] Task 3b: Verify label flows through collation — verified: collate outputs `quality_domain` as `torch.long` tensor
- [x] Task 3c: Unit test — **Already exists**: `test_unpaired_reals_and_grl.py` has `TestQualityDomainMap`, `TestCollateQualityDomain`, `TestGradientReversalLayer`, `TestQualityDomainHead`
- [x] Task 3d: Enable in experiment YAML — created `R12_E_grl_teams_ood.yaml` with `use_quality_domain_head: true`, `quality_domain_loss_weight: 0.1`
- [x] Task 3e: Verify GRL lambda annealing — `_update_quality_domain_lambda()` in trainer.py L231 confirmed working (sigmoid schedule)
- [x] Task 4a: `tta_inference()` utility — implemented in `app4.py`
- [x] Task 4b: TTA env var in `app4.py` — `TTA_ENABLED` env var
- [x] Task 4c: Wire into `/check_frame` — TTA replaces single forward pass when enabled
- [x] Task 4d: Wire into video inference — per-frame TTA in `/check_video` and `/check_video_from_gcp`
- [ ] Task 4e: Benchmark latency — *requires deployment; expected ~2× per frame*
- [x] Task 5a: Add Teams OOD source to config — `teams_ood_real` + `teams_ood_fake` in R12_E `ood_monitoring`
- [ ] Task 5b: Verify OOD loader — *requires running R12_E on Vertex AI*
- [x] Task 5c: Verify identity exclusion — `exclude_training_identities: true` in R12_E config
- [x] Task 5d: Update R12 YAMLs — created `R12_E_grl_teams_ood.yaml`
