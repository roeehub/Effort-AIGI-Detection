# Teams Video Pipeline — System Report

## Goal

Collect face-crop training data by passing deepfake detection videos through a real Microsoft Teams call and capturing the face crops produced by the WMA (Windows Media Analyzer) debug inspector. This creates a dataset of `{real, fake}` face-crop pairs that have been through the Teams video pipeline — encoding artifacts, compression, resolution changes — matching what the detector will see in production.

Each sample pair in GCS has a `real.mp4` and `fake.mp4`. The pipeline plays each video through OBS → Virtual Camera → Teams call → WMA face detection, and collects the resulting face crops on Machine B.

---

## Architecture

```
Machine A (Mac)                          Machine B (Windows)
┌──────────────────────┐                ┌──────────────────────────────┐
│  obs_coordinated_    │   HTTP API     │  receiver_server.py          │
│  sender.py           │◄──────────────►│  (FastAPI on port 8080)      │
│                      │                │                              │
│  - Downloads video   │                │  - Watches WMA faces_dir     │
│    from GCS          │                │  - Baseline snapshot sync    │
│  - Plays via OBS     │                │  - Caches JPG bytes in RAM   │
│  - Controls scenes   │                │  - Writes to teams_dataset/  │
│  - Polls receiver    │                │                              │
└──────┬───────────────┘                └──────────────┬───────────────┘
       │                                               │
       ▼                                               ▼
   OBS Studio                                    WMA Debug Inspector
   (Virtual Camera) ──► Teams Call ──►          (face crop output)
```

---

## Files

### 1. `obs_coordinated_sender.py` (~935 lines) — Machine A

**Purpose:** Orchestrates the entire pipeline from the Mac side. Downloads videos from GCS, plays them through OBS into a Teams call, and coordinates with the receiver to ensure each segment's face crops are captured before moving on.

**Key flow per segment:**
1. **Green screen** (5s) — flushes WMA pipeline of previous video's faces
2. **Download video** from GCS (overlapped with green screen time)
3. **Start playback** in OBS → switches to VideoPlayback scene
4. **Verify playing** — confirms OBS media state is PLAYING (retries 3x)
5. **Pipeline settle** (2s) — waits for OBS → Teams → WMA propagation
6. **Signal receiver** — calls `POST /segment/start` (video already playing!)
7. **Poll** `GET /segment/status` every 1s until COMPLETE or timeout

**Key components:**
- `ResilientOBS` — OBS WebSocket wrapper with auto-reconnect on disconnect
- `ReceiverClient` — HTTP client for all receiver API calls
- `play_segment_with_retry()` — retries failed segments up to 3x
- `discover_samples()` / `build_playlist()` — GCS sample discovery
- `write_rerun_manifest()` — saves failed samples for re-run
- Deferred retry pass — second pass over all first-pass failures
- `--limit N` flag — process only N pairs (for testing)
- `--rerun-manifest` — re-run only specific failed samples
- Auto-skip of already-completed samples on receiver

**Why the ordering matters:** The sender signals the receiver AFTER the video is playing and settled. This ensures the receiver's baseline snapshot captures the directory state while the correct video's faces are flowing through WMA — eliminating the timing race that caused wrong-face contamination.

---

### 2. `receiver_server.py` (~990 lines) — Machine B

**Purpose:** FastAPI server that watches the WMA debug inspector's face crop output directory. When signaled by the sender, it takes a baseline snapshot of the directory, detects new face crops appearing, caches their bytes in memory, and writes them to the output dataset.

**Key flow per segment:**
1. `POST /segment/start` → takes full `{filename: mtime}` snapshot of faces_dir
2. Watcher thread `tick()` every 250ms:
   - Scans for files with changed mtime (new or overwritten since baseline)
   - Validates: `.jpg` exists, file size ≥ 5KB
   - **Reads JPG bytes into RAM** before marking complete (critical safety)
   - When ≥16 valid cached frames → marks COMPLETE
3. Background `_finalize_worker`:
   - Filters bbox outliers (rejects faces >60px from median position)
   - Caps at 30 frames (evenly spaced)
   - **Writes from cached bytes** — never reads from live faces_dir

**Key safety layers:**
| Layer | Protection |
|-------|-----------|
| Cache-on-detect | JPG bytes read into memory BEFORE COMPLETE is set |
| Write from memory | `_finalize_worker` uses `dst.write_bytes(cached_bytes)` |
| Baseline sanity | >90 changed files → re-snapshots baseline, skips tick |
| Bbox outlier filter | Rejects face crops with deviant position |

**API endpoints:**
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness check |
| `/session/start` | POST | Initialize session with playlist |
| `/session/end` | POST | Save session log, return summary |
| `/segment/start` | POST | Start collecting for a sample/type |
| `/segment/status` | GET | Current segment state + frame counts |
| `/session/progress` | GET | Overall session stats |
| `/completed-samples` | GET | Samples with both real+fake on disk |
| `/dataset/samples` | GET | Inventory of collected samples |
| `/dataset/frames/{id}/{side}` | GET | Base64-encoded frames for validation |
| `/dataset/session-log` | GET | Full session log |

**Modes:**
- Normal: auto-discovers latest WMA session directory
- `--test`: creates temp directory, drops fake frames, self-validates
- `--session-dir`: explicit session directory

---

### 3. `test_pipeline.py` (~507 lines) — Machine A

**Purpose:** Pre-flight verification suite. Run before an overnight collection to confirm the full sender↔receiver pipeline is healthy.

**Test levels:**

| Level | Tests | What it checks |
|-------|-------|---------------|
| `quick` (default) | 7 | Health, session, segment signals, polling, progress, completed-samples, session end |
| `stress` | 11 | All quick tests + repeated session cycles, rapid segment starts, concurrent polling |
| `full` | N | Actually runs N collection cycles (requires video playing) |

**Key behaviors:**
- Auto-detects test mode (checks if faces_dir contains "receiver_test" or "Temp")
- In test mode, collection cycle tests SKIP instead of FAIL (expected: no real WMA)
- 8-second timeout for collection tests in test mode (vs 45s normally)
- Colored terminal output with timing per test

**Usage:**
```bash
python test_pipeline.py --receiver-url http://10.0.0.19:8080              # quick
python test_pipeline.py --receiver-url http://10.0.0.19:8080 --level stress
python test_pipeline.py --receiver-url http://10.0.0.19:8080 --level full --pairs 3
```

---

### 4. `validate_teams_dataset.py` (~674 lines) — Machine A or B

**Purpose:** Post-run validation and visual comparison tool. Audits the collected dataset for completeness and generates an HTML report comparing Teams face crops against GCS ground truth.

**What it checks:**
- Directory structure: each sample has `real/` and `fake/` subdirectories
- Frame counts: flags samples with <16 or significantly unbalanced frame counts
- File integrity: checks for zero-byte or suspiciously small JPGs
- **GCS comparison** (`--compare-gcs`): fetches original face crops from GCS bucket, generates side-by-side HTML

**GCS comparison HTML:**
- Left: Teams face crops (collected by pipeline)
- Right: GCS original face crops (ground truth)
- Shows ALL frames (no subsampling)
- Self-contained HTML with base64-embedded images
- Caches GCS downloads in `gcs_comparison_cache/` for fast re-runs

**Modes:**
- `--dataset-dir <path>`: reads directly from local teams_dataset directory
- `--receiver-url <url>`: fetches data remotely via receiver API endpoints

**Usage:**
```bash
# Local (on Machine B):
python validate_teams_dataset.py --dataset-dir C:\...\teams_dataset --compare-gcs

# Remote (from Machine A):
python validate_teams_dataset.py --receiver-url http://10.0.0.19:8080 --compare-gcs
```

---

## GCS Bucket Structure

| Bucket | Contents | Path pattern |
|--------|----------|-------------|
| `live-deepfake-methods-real-and-fake-videos` | Source videos | `samples/{sample_id}/{real\|fake}.mp4` |
| `live-deepfake-methods-real-and-fake-frames-cropped` | Ground truth face crops | `samples/{sample_id}/frames/{real\|fake}/frame_NNNN.png` |

All strategies (edge_cases, minimal_processing, quality_enhancement, visomaster_*) are in the same buckets. Project: `train-cvit2`.

---

## Output Structure

```
teams_dataset/
├── edge_cases_0000/
│   ├── real/
│   │   ├── frame_0000.jpg
│   │   ├── frame_0001.jpg
│   │   └── ... (16+ frames)
│   └── fake/
│       ├── frame_0000.jpg
│       └── ... (16+ frames)
├── edge_cases_0001/
│   ├── real/
│   └── fake/
└── ...
```

---

## Operational Workflow

```
1. Start WMA + Teams call on Machine B
2. Start receiver:     python receiver_server.py --output-dir <dir> --port 8080
3. Pre-flight test:    python test_pipeline.py --receiver-url http://B:8080 --level stress
4. Quick E2E test:     python obs_coordinated_sender.py --receiver-url http://B:8080 --limit 3
5. Validate:           python validate_teams_dataset.py --dataset-dir <dir>/teams_dataset --compare-gcs
6. Overnight run:      python obs_coordinated_sender.py --receiver-url http://B:8080
7. Morning validate:   python validate_teams_dataset.py --dataset-dir <dir>/teams_dataset --compare-gcs
```

---

## Bug History & Fixes

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Wrong faces in output | `shutil.copy2` from live dir in background thread; files overwritten by next video before copy runs | Cache JPG bytes in RAM before COMPLETE; write from memory |
| Signal-before-play race | Receiver warmup expired before video reached WMA | Restructured: play video → settle → then signal receiver |
| Empty baseline → flood | Transient filesystem error → all files look "new" | Sanity guard: >90 changed files → re-snapshot baseline |
| OBS WebSocket drops | Long-running sessions disconnect | `ResilientOBS` wrapper with auto-reconnect |
| Segment timeouts | Single attempt, no retry | `play_segment_with_retry()` up to 3x + deferred retry pass |
| Slow dir scanning | `os.listdir()` + stat on large circular buffer | `os.scandir()` with DirEntry caching |
