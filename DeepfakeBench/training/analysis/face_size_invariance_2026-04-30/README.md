# Face-size invariance probe (2026-04-30)

## Purpose

Validate whether **Slot 3 (P14 jitter@0.50, run `mclioexb`)** flattened the
score-vs-tightness curve compared to **P8A baseline (run `9lmvb5b4`)**. The
04-27 audit (`analysis/crop_shortcut_2026-04-27/`) found that 25/47 (53%) of
production-honest frames flipped predicted label across crop tightness
t in [0.7, 1.5]. If `face_scale_jitter.scale_limit=0.50` worked, this rate
should drop substantially on the Slot 3 checkpoint.

## What it does

For each frame in `--frames_dir`:
1. Generate N tightness variants via center-crop / pad-with-edge-replication
   (mirrors `analysis/crop_shortcut_2026-04-27/crop_sweep.py:make_variant`).
2. Score each variant with the loaded checkpoint via the standard
   `model({'image': ...}, inference=True)` interface.
3. Record per-(frame, tightness) `prob_fake` and the predicted label at
   threshold 0.5.

## Inputs

```
python face_size_invariance.py \
  --ckpt gs://...value_composite_effort_..._step5000_....pth \
  --frames_dir /path/to/47-frame/dir \
  --tightness_grid 0.7 0.85 1.0 1.15 1.5 \
  --output_csv ./outputs/p8a_invariance.csv \
  --output_png ./outputs/p8a_invariance.png
```

Re-run with the Slot 3 ckpt and the same `--frames_dir` to compare.

## Outputs

- **CSV** (`--output_csv`): one row per (frame, tightness) with
  `prob_fake` and `predicted_label`.
- **PNG** (`--output_png`): 47 thin lines (alpha=0.3) overlaid with a thick
  median line; flip-rate annotated in the title.
- **Stdout**: total flip rate, adjacent-pair flip counts, median |Δprob_fake|.

## Expected runtime (M-series Mac CPU)

- 47 frames × 5 tightness levels = 235 forward passes.
- ViT-B-16 on CPU ≈ ~0.4-0.6s per forward pass at batch=64.
- Total: **~3-5 minutes per checkpoint** including PIL transforms.
- On `mps` device: ~30-60 seconds per checkpoint.
- On CUDA (if running on Vertex): ~5-10 seconds.

## Interpretation

| Outcome | Reading |
|---|---|
| P8A flip rate ≈ 53%, Slot 3 flip rate ≤ 20% | jitter@0.50 worked — face-size leak attenuated |
| P8A flip rate ≈ 53%, Slot 3 flip rate ≥ 40% | jitter@0.50 did NOT bite — leak intact |
| Median \|Δprob_fake\| drops from ~0.3 to <0.1 on Slot 3 | curve flattened — jitter generalized |
| Both checkpoints flat (low flip rate) | crop-shortcut may have been reduced upstream — verify with ground-truth labels per frame |

## Caveats

- **Frames must be already face-cropped** to the same convention as training
  (square crop, face centered). The `_prod_cache/frames/` directory used by
  `population_sweep.py` is the canonical source.
- Very tight crops (t > 1.5) may eliminate the face entirely; very loose
  crops (t < 0.5) lose enough context that the prediction becomes
  meaningless. Stick to t in [0.7, 1.5] unless investigating extreme regimes.
- Threshold 0.5 is arbitrary — for a stricter readout, also examine the
  prob_fake delta directly (the `median |Δprob_fake|` summary).
