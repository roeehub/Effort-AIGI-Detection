# Teams Video Capture — Reconstruction & Validation

## What We're Doing

We played 1,179 DeepLiveCam sample pairs (real + fake videos, 2,358 total) through an OBS → Microsoft Teams pipeline overnight. The goal is to create Teams-augmented training data: the same deepfake/real faces, but with realistic video conferencing artifacts (H.264 compression, YUV 4:2:0 chroma subsampling, auto-brightness, denoising, resolution scaling). This augmented data will be used to train our Effort deepfake detector to generalize to video-call scenarios.

On the playback side (Mac), `obs_full_pipeline.py` played each video through OBS with celebrity-face separator images between clips, logging exact timestamps to `playback_log.jsonl`. On the capture side (this Windows machine), capture software recorded face crops from the Teams call into a session directory with JSON sidecar files containing timestamps.

## The 5,000 Face Cap Bug

The overnight run completed successfully — all 2,358 videos played over 11.1 hours. However, the capture software had a **5,000 face crop limit** that we didn't know about. It stopped saving faces after ~35 minutes into the 11-hour session, capturing only ~125 out of 2,358 videos worth of faces.

**We need to re-run overnight with the cap removed/raised**, but first we need to validate the entire reconstruction pipeline using the 5,000 faces we already have. This is enough data (~60 complete sample pairs) to verify everything works end-to-end before committing to another 11-hour run.

## What Needs to Be Validated

There are three scripts to run, in order:

### 1. `reconstruct_teams_dataset.py` — Timestamp-based assignment

This script reads `playback_log.jsonl` (timestamps of when each video played) and the face crop JSON sidecars (timestamps of when each face was captured), then assigns each face to its source video by matching timestamps. It outputs a paired directory structure: `teams_dataset/{sample_id}/{real|fake}/frame_NNN.jpg`.

**What to verify:** The dry-run report should show ~5,000 faces distributed across ~125 video segments with ~40 faces each. If most faces land in "unassigned", the clock offset between the two machines needs adjusting (use `--clock-offset`).

### 2. `verify_assignment.py` — Visual sanity check

This downloads the original YOLO-cropped frames from GCS and places them side-by-side with the Teams-captured faces in image grids. It generates an `index.html` you can open in a browser.

**What to verify:** For each sample, the left column (original, green border) and right column (Teams-captured, blue border) should show **the same person**. If they show different people, the timestamp alignment is wrong and the `--clock-offset` parameter needs tuning.

### 3. `compare_original_vs_teams.py` — Property comparison

This compares statistical properties of original GCS frames vs Teams-captured frames. The metrics are chosen specifically for what Teams video conferencing changes:

- **Sharpness** (Laplacian variance) — should decrease (H.264 blur)
- **Blockiness** (8×8 macro-block boundary detection) — should increase (H.264 artifacts)
- **Chroma blur ratio** (chroma vs luma sharpness) — should decrease (YUV 4:2:0 subsampling)
- **High-frequency energy ratio** (DCT analysis) — should decrease (compression removes detail)
- **Noise level** (MAD-based estimate) — should decrease (Teams denoising)
- **Color channel means** — may shift (auto-white-balance, YUV conversion)

**What to verify:** The delta table should show the expected directions above. If blockiness doesn't increase or sharpness doesn't decrease, something is wrong with the capture pipeline.

## Commands

```bash
# Install dependencies
pip install numpy Pillow scipy matplotlib google-cloud-storage
gcloud auth application-default login

# Step 1: Dry run to check assignment
python reconstruct_teams_dataset.py --session-dir <SESSION_DIR> --playback-log playback_log.jsonl --dry-run

# Step 1b: If most faces are unassigned, try adjusting clock offset
python reconstruct_teams_dataset.py --session-dir <SESSION_DIR> --playback-log playback_log.jsonl --clock-offset 12.0 --dry-run

# Step 2: Run reconstruction for real
python reconstruct_teams_dataset.py --session-dir <SESSION_DIR> --playback-log playback_log.jsonl

# Step 3: Visual verification (opens in browser)
python verify_assignment.py --teams-dir teams_dataset --max-samples 20

# Step 4: Property comparison
python compare_original_vs_teams.py --teams-dir teams_dataset --max-samples 20
```

Replace `<SESSION_DIR>` with the path to the capture session directory (e.g. `C:\Users\...\session_20260225_004520`).

## Files Needed

These files should have been copied from the Mac:

- `reconstruct_teams_dataset.py` — reconstruction script
- `compare_original_vs_teams.py` — property comparison script
- `verify_assignment.py` — visual verification script
- `playback_log.jsonl` — timestamps from the OBS playback session
