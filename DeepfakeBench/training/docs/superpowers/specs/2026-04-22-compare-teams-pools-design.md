# Spec: `compare_teams_pools.py` — diagnose why the model is weak on `teams_ood`

**Date**: 2026-04-22
**Author**: Claude (sonnet-4-7) + roee
**Status**: Approved for implementation
**Branch**: `teams-relaunch-root-2026-04-17`

## Problem

The current packet-3 leader (slot 05 low_arc+spatial, `value_composite_effort_20260422_step2500_*.pth`) is strong overall but consistently underperforms on the `teams_ood` OOD pool (per-pool TPR ≈ 0.66 at the global tau, vs > 0.85 on every other real pool). `teams_ood_*` is the older "live-deepfake-methods Teams v2" capture and does **not** overlap with current training Teams data. The newer Teams supervision in training comes from the `proper_visomaster` wave (HDTF + QuickClips rendered through the visomaster pipeline with Teams transport), which landed 2026-04-19.

We want a diagnostic script that takes representative samples from both pools, surfaces concrete distributional differences (image stats, face geometry, model embeddings, model confidence), and produces a self-contained HTML report that lets a human form hypotheses about how to change the training regime to cover both.

The script's success metric is **legibility, not automation** — it must let the user *see* the differences. Recommendations are a follow-up session with the report in hand.

## Non-goals

- No automated training-regime recommendation.
- No GCS output. Local artifact only; user copies wherever they want.
- No per-identity drill-down (deferred — only build it if the failures gallery surfaces identity-clustered artifacts).
- No production hardening, no Vertex AI packaging. One-off local diagnostic.

## Data plane

Two pools × two labels = **four groups**:

| Group | Pool tag | GCS bucket(s) | Pre-filter |
|------|----------|---------------|------------|
| `teams_ood_real` | older live-capture | `gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2/samples/<capture>/frames/real/` | None (held-out identity split is already enforced upstream — we sample from whatever's in the bucket since the script is not used in training) |
| `teams_ood_fake` | older live-capture | `gs://.../samples/<capture>/frames/fake/` | None |
| `proper_visomaster_teams_real` | new visomaster Teams transport | `gs://hdtf_visomaster_cropped_frames_teams/...` + `gs://quickclips_visomaster_cropped_frames_teams/...`, real lane | None |
| `proper_visomaster_teams_fake` | new visomaster Teams transport | same buckets, fake lane | None |

Exact bucket sub-paths are read from the canonical inventory `arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml` and the existing OOD source manifest. The script must not hardcode sub-paths beyond the bucket root — it should resolve real/fake folders by the same convention the training pipeline uses.

### Sampling

- **Per group**: 150 videos, 8 frames per video → **1200 frames per group**, 4800 total.
- Video-level seeded sample (seed=737, matches training default) so reruns hit cache.
- Frame selection inside a video: deterministic 8-frame slice (the same convention `create_data_pipeline` uses for OOD scoring).
- If a video has fewer than 8 frames, take all available; record the actual count.

### Local cache

- `~/.cache/teams_pool_diff/<pool>/<video_id>/<frame>.jpg`
- Cache key: `(bucket, sub-path, video_id)`. Re-runs skip download if cache hit.
- Cache size budget: ~4800 frames × ~50 KB = ~250 MB. Acceptable.

## Five passes

Each pass is a function that consumes a `frames_df` (path + group + label + video_id) and emits a section of `stats.json`. The HTML builder is the only consumer of all five sections.

### Pass 1 — Low-level image stats

Per frame, computed on the decoded numpy array:

- `brightness` — mean luminance (rec601)
- `contrast` — std of luminance
- `sharpness` — variance of Laplacian (cv2.Laplacian)
- `colorfulness` — Hasler-Süsstrunk metric
- `saturation` — mean S in HSV
- `resolution_w`, `resolution_h` — pixel dimensions
- `bytes_per_pixel` — file size / (W×H), JPEG-compression proxy
- `warmth` — mean R minus mean B (proxy for color temperature)

Aggregated per group: `mean`, `median`, `p5`, `p95`, plus pairwise **two-sample KS-distance** for each feature (4 groups → 6 pairs, but we only care about the 2 cross-pool same-label pairs: real-vs-real, fake-vs-fake).

### Pass 2 — Face geometry

MediaPipe `face_mesh` (CPU) on each frame. Per frame:

- `face_detected` — bool
- `face_bbox_area_ratio` — face bounding box / crop area
- `inter_eye_distance_ratio` — IED in pixels / crop width
- `yaw_deg`, `pitch_deg`, `roll_deg` — head pose from landmark PnP
- `landmark_confidence` — mediapipe's score

Aggregated same way as pass 1, plus a `face_detected_rate` per group.

### Pass 3 — Model embeddings + confidence

- Load checkpoint: P3 leader slot 05, `gs://training-job-outputs/phase2r13_experiments/w92amaaa/value_composite_effort_20260422_step2500_auc0.9895_eer0.0116.pth`.
- Reuse the load pattern from `retro_score_value_composite._load_state_dict_into_model` (which itself mirrors `rerun_validation.py:169-202`). Don't reinvent.
- Preprocessing must match training: resize 224, ImageNet normalize.
- CPU inference, batch 16. Expected runtime: ~10 min for 4800 frames on a Mac.
- Per frame: `embedding` (512d, penultimate layer) + `fake_prob` (sigmoid of head logit).
- Per group: centroid embedding, within-group L2 dispersion, pairwise cosine distance between the 4 centroids.
- Per group: confidence histogram bins [0.0, 0.1, ..., 1.0].

Embeddings stored in `raw_scores.parquet` (one row per frame: group, label, video_id, frame_path, fake_prob, 512 embedding floats). The HTML uses this for UMAP and the gallery.

### Pass 4 — Hard-sample selection

For each of the 4 groups, pick three buckets:

- **Confidently wrong** — top 30 by `|fake_prob − target| > 0.9` (reals scored ≥0.9 fake / fakes scored ≤0.1 fake)
- **Uncertain** — top 30 with `0.4 ≤ fake_prob ≤ 0.6`, sampled by closeness to 0.5
- **Confidently right** — top 30 by `|fake_prob − target| < 0.1`, sampled at random within that band

Total: **360 flagged frames**. Each gets a 224×224 thumbnail saved under `thumbnails/<group>/<bucket>/<video_id>_<frame>.jpg`.

### Pass 5 — HTML report

Single self-contained `report.html` (no external assets — base64-embedded thumbnails, inline matplotlib SVGs):

1. **Executive summary** — sortable table: feature name × KS-distance (real-vs-real and fake-vs-fake), highlighting features with KS > 0.2.
2. **Stats distributions** — paired histograms (teams_ood vs proper_visomaster) for each pass-1 feature, separately for real and fake labels.
3. **Geometry distributions** — same layout for pass-2 features.
4. **Model section** — confidence histograms per group, 2D UMAP of embeddings colored by group, centroid distance heatmap.
5. **Failures gallery** — for each group, three thumbnail strips (confidently-wrong, uncertain, confidently-right). Each thumbnail captioned with `fake_prob`, `face_detected`, video_id.

## Architecture

Single file: `analysis/compare_teams_pools.py`. Sections marked with banner comments:

```
# === Section 1: CLI + paths ===
# === Section 2: Sampling + GCS download (cached) ===
# === Section 3: Pass 1 — image stats ===
# === Section 4: Pass 2 — face geometry ===
# === Section 5: Pass 3 — model embeddings + confidence ===
# === Section 6: Pass 4 — hard-sample selection ===
# === Section 7: Pass 5 — HTML report ===
# === Section 8: main() ===
```

Functions are pure where reasonable. Each pass:

```python
def run_pass_<n>(frames_df: pd.DataFrame, *, output_dir: Path, **kwargs) -> dict
```

Returns a JSON-serializable dict that the report consumes. Each pass also writes its own `stats_pass_<n>.json` so a `--skip-pass` rerun can reload prior results.

### CLI

```
python analysis/compare_teams_pools.py \
  --checkpoint gs://training-job-outputs/phase2r13_experiments/w92amaaa/value_composite_effort_20260422_step2500_auc0.9895_eer0.0116.pth \
  --videos-per-group 150 \
  --frames-per-video 8 \
  --output-dir scratch/teams_pool_diff \
  [--skip-pass stats|geometry|model|gallery]   # repeatable
  [--cache-dir ~/.cache/teams_pool_diff]
  [--seed 737]
```

`--output-dir` gets a timestamped subdir per run.

### Dependencies

Already present (assumed): `torch`, `torchvision`, `numpy`, `matplotlib`, `Pillow`, `pandas`, `google-cloud-storage`, `opencv-python`.

**New (pip-only, no system deps):**
- `mediapipe` (face landmarks)
- `umap-learn` (2D embedding viz)
- `pyarrow` (parquet for raw scores)

## Error handling

- **GCS auth failure** — first GCS call gets a try/except that prints `gcloud auth application-default login`-style instructions and exits 2.
- **MediaPipe no face** — record `face_detected=False`, contribute `NaN` to geometry features (numpy-safe in aggregation), still appears in pass-1 stats.
- **Checkpoint state-dict mismatch** — print missing/unexpected keys diff, exit 3. Don't silently load a partial model.
- **Per-pass crashes** — wrap each `run_pass_<n>` in a try/except; on failure, log + write `stats_pass_<n>_FAILED.json` + continue. Report renders "(pass failed: <error>)" in that section. The user can `--skip-pass` and rerun the failing one in isolation.
- **Cache corruption** — single corrupt frame logged + skipped; downstream passes use only successfully-decoded frames. If a video drops below 4 usable frames, drop the video entirely and log it.

## What we don't build (explicitly)

- No identity-level analysis (deferred until failures gallery suggests it's worth it).
- No multi-checkpoint comparison (just P3 slot 05; trivially extensible by re-running with `--checkpoint` change).
- No statistical-significance hypothesis testing beyond KS-distance values (no p-value gymnastics).
- No interactive web UI (static HTML only).
- No automated CI integration. Manual diagnostic.

## Acceptance criteria

1. Running the CLI as documented produces `scratch/teams_pool_diff/<timestamp>/report.html` plus `stats.json`, `raw_scores.parquet`, and `thumbnails/`.
2. The HTML opens in a browser and renders all 5 sections without errors (modulo any pass marked "failed" if data was unavailable).
3. The executive summary table includes at least 8 features and lists which side of each comparison is "more" (e.g. "teams_ood is darker / more compressed / lower-resolution").
4. Re-running with the same args is fully cached: zero new GCS reads, zero new model inference (idempotent on `<timestamp>`-disambiguated output).
5. `--skip-pass model` runs in <2 min on a Mac (no checkpoint download, no inference); the report still renders the pass-1 / pass-2 sections.
