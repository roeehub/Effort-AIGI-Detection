# Embedding triptych: P8A | Slot 2 | Slot 3 (2026-04-30)

## Purpose

Visual narrative figure for the wiki/handoff. A 3x3 grid:
- **Rows**: P8A baseline | Slot 2 (GRL) | Slot 3 (jitter)
- **Columns**: real/fake coloring | clip_capture_mode | face_pixel_area bucket

Within a row, geometry is shaped by what THAT checkpoint's [CLS] features
encode. Across rows, you see how each intervention reshaped the manifold
along the three load-bearing axes.

## What it does

1. Stratified-sample N frames from the lockbox parquet across
   (label, method, clip_capture_mode).
2. For each checkpoint, extract [CLS] features (cached to
   `--cache_dir`).
3. Project to 2D via TSNE (default, perplexity=30) or UMAP
   (`--reducer umap`).
4. Render a 3x3 grid of scatter plots; each row uses the same checkpoint's
   embedding, each column uses a different coloring.

## Inputs

```
python triptych.py \
  --ckpts P8A=gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_..._step5000_....pth \
          SLOT2=gs://training-job-outputs/phase2r13_experiments/w5tky6ss/value_composite_effort_..._step....pth \
          SLOT3=gs://training-job-outputs/phase2r13_experiments/mclioexb/value_composite_effort_20260429_step500_auc0.9797_eer0.0521.pth \
  --n_samples 800 \
  --reducer tsne \
  --output_dir ./outputs/triptych/
```

For UMAP: install `umap-learn` first (`pip install umap-learn`), then pass
`--reducer umap`.

## Outputs

- **triptych_grid_tsne.png** (or `_umap.png`): the headline 3x3 figure.
- **triptych_coords_tsne.csv**: per-checkpoint per-frame 2D coords + all
  metadata columns (so the user can re-render plots without re-extracting
  features).
- **sampled_frames.csv**: the stratified sample manifest, useful for
  reproducibility.

## Expected runtime (M-series Mac CPU)

- 800 frames × 3 checkpoints = 2400 forward passes.
- ViT-B-16 on CPU at batch=64: ~7-10 minutes per checkpoint = ~25-30
  minutes for feature extraction.
- TSNE on 800 × 768-dim points: ~30-60 seconds per checkpoint = ~2-3
  minutes for reduction.
- **Total: ~30-40 minutes** end-to-end on first run; ~3 minutes on rerun
  with cached features.
- On `mps`: ~10-15 minutes first run.
- On CUDA: ~3-5 minutes first run.

## Interpretation

### Column 1 (real/fake coloring)

| Outcome | Reading |
|---|---|
| All 3 rows show clean real/fake separation | All checkpoints retain primary-task discriminability. Sanity check passed. |
| Slot 2 / Slot 3 row shows much weaker real/fake separation than P8A | The intervention degraded the primary task; check W&B AUC. |

### Column 2 (clip_capture_mode coloring)

| Outcome | Reading |
|---|---|
| P8A: webcam cluster sits as a dense island; Slot 2: webcam cluster bleeds into rest | GRL worked — capture-mode invariance achieved. (Project north-star outcome.) |
| All 3 rows: webcam stays a dense island | Neither intervention attacked the right axis. |
| Slot 3: webcam cluster also softens | Suggests jitter is incidentally softening capture-mode features (interesting but not the targeted lever). |

### Column 3 (face_pixel_area bucket coloring)

| Outcome | Reading |
|---|---|
| P8A: face-size buckets cleanly separate; Slot 3: buckets bleed into each other | jitter@0.50 worked — face-size leak attenuated. |
| All 3 rows: buckets cleanly separate | Face-size leak intact across all interventions. |
| Slot 2 row also shows softening | GRL is incidentally softening face-size features (gain across both axes). |

## Caveats

- TSNE coordinates are **not directly comparable across rows** (each row
  is its own non-linear projection). Within-row geometry is meaningful;
  across-row distances are not.
- `--n_samples 800` is the recommended floor for TSNE perplexity 30. Below
  ~300 samples, TSNE structure becomes unstable. Above ~2000, runtime
  becomes prohibitive on CPU.
- Stratified sampling is **proportional** to the parquet's stratum sizes;
  rare strata may have only 1-2 representatives. For balanced-by-design
  sampling, use the `--seed` arg and check `sampled_frames.csv`.
- UMAP is faster but its preserves global vs local structure differently;
  for visual storytelling stick with TSNE unless you have a specific
  reason.
