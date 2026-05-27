# Domain-confusion linear probe (2026-04-30)

## Purpose

Direct test of whether **Slot 2 (P15 GRL, run `w5tky6ss`)** successfully
flattened the quality-domain manifold in the backbone's [CLS] features. If
the gradient-reversal head bit, training a frozen-feature 3-way logistic
regression should yield much lower macro-OVR AUC on Slot 2 than on
**P8A baseline**.

This is the canonical "did GRL succeed at its job?" test and matches the
DANN-style architecture documented in `detectors/effort_detector.py:236-274`
and `data/sources/combined_paired.py:65-82` (QUALITY_DOMAIN_MAP).

## What it does

1. Sample N frames per domain from the lockbox parquet (mapping `method` →
   quality-domain ID using `assign_domain_default`).
2. Extract [CLS] features from each checkpoint with the same preprocessing
   pipeline as `batch_inference_gcs.py` (cv2 INTER_LINEAR + CLIP normalize).
3. Train a multinomial `LogisticRegression` (n_jobs=1; never -1) with
   stratified 5-fold cross-validation, predicting the quality-domain.
4. Report 3-way macro-OVR AUC and per-class OOF confusion matrix.

## Inputs

```
python domain_probe.py \
  --ckpts P8A=gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_..._step5000_....pth \
          SLOT2=gs://training-job-outputs/phase2r13_experiments/w5tky6ss/value_composite_effort_..._step....pth \
  --sample_per_domain 300 \
  --domains 0 1 2 \
  --output_dir ./outputs/probe_p8a_vs_slot2/
```

## Outputs

- **summary.json**: full per-checkpoint result dict (macro AUC, per-class
  AUC, confusion matrix, fold-level metrics).
- **per_fold_per_class_auc.csv**: long-form CSV for downstream analysis;
  fold=-1 rows are the OOF (out-of-fold) aggregate per class.
- **confusion_matrices.png**: side-by-side normalized confusion matrices,
  one panel per checkpoint.
- **macro_auc_bar.png**: macro-AUC bar chart with chance baseline.
- **Stdout**: macro-AUC P8A vs Slot 2, the headline single number per ckpt.

## Expected runtime (M-series Mac CPU)

- ~900 frames × 2 checkpoints × 1 forward pass = 1800 forward passes.
- ViT-B-16 on CPU at batch=64: ~5-7 minutes per checkpoint.
- LogisticRegression with 5-fold CV: <30s.
- **Total: ~12-15 minutes** end-to-end.
- On `mps`: ~3-4 minutes.
- On CUDA: ~1-2 minutes.

## Interpretation

| Outcome | Reading |
|---|---|
| P8A macro-AUC ≈ 0.95, Slot 2 macro-AUC ≤ 0.55 | GRL worked. Backbone is now domain-invariant. |
| P8A macro-AUC ≈ 0.95, Slot 2 macro-AUC ≥ 0.85 | GRL did NOT bite. Reversed gradient was insufficient or λ too small. |
| Slot 2 macro-AUC drops below chance (~0.30) | Possible label leak in fold construction — re-run with `--seed` change. |
| Confusion off-diagonal mass between domains 1↔2 collapses on Slot 2 | The webcam ↔ studio distinction is what got flattened (load-bearing direction per `project_signature_shortcut_finding.md`). |

## Caveats

- Lockbox parquet contains primarily `teams_*` (domain 1) and
  `deeplive_enhanced` (domain 2). **Domain 0 (df40) will have 0 rows** in
  the default parquet → script will warn and run a 2-class probe instead.
  To get domain 0, supply `--parquet` pointing at a parquet that includes
  df40 frames with cached `local_path`.
- The probe is **linear** by design (matches the GRL theoretical guarantee).
  A non-linear MLP probe could expose residual domain info that the linear
  head misses. Out of scope for this script.
- Features are **L2-normalized via StandardScaler before fitting**, so the
  probe sees direction-only signals. Norm-based shortcuts would not show
  up as macro-AUC differences here — separate scripts handle that.
