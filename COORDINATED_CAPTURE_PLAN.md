# Coordinated Teams Data Collection — Implementation Plan

**Created:** February 28, 2026  
**Purpose:** Replace blind timestamp-based reconstruction with real-time API-coordinated capture.

---

## Overview

Two scripts work in tandem across two machines during a Teams call:

| Machine | Role | Script | Purpose |
|---|---|---|---|
| **Machine A** (Sender) | Plays videos via OBS → Teams virtual camera | `obs_coordinated_sender.py` | Plays real/fake videos, signals receiver, waits for confirmation |
| **Machine B** (Receiver / `dtect_dev1`) | Captures faces via WMA app | `receiver_server.py` | Watches WMA debug face crops, organizes them per sample, signals readiness |

The WMA app's real capture pipeline (screen capture → YOLO → gRPC → debug inspector) stays **completely untouched**. The receiver server is a passive observer that reads the face crops WMA writes to disk.

---

## Architecture

```
Machine A (Sender)                          Machine B (Receiver / dtect_dev1)
┌─────────────────────┐                     ┌────────────────────────────────────┐
│ obs_coordinated_    │                     │  WMA App (C++)                     │
│ sender.py           │                     │    Screen Capture → YOLO → gRPC   │
│                     │                     │         │                          │
│ 1. Show GreenScreen │                     │  python_backend/server.py          │
│ 2. POST /segment/   │ ──── HTTP ────────► │    --debug-inspect                 │
│    start            │                     │    DebugInspector writes:          │
│ 3. Play video.mp4   │  Teams Video Call   │    participants/Roy D/faces/       │
│    via OBS→VCam     │ ◄═══════════════════│      frame_000042_seq1537.jpg      │
│ 4. Poll GET /segment│ ──── HTTP ────────► │      frame_000042_seq1537.json     │
│    /status          │                     │         │                          │
│ 5. status.complete  │ ◄─── HTTP ──────────│  receiver_server.py (port 8080)    │
│    == true → next   │                     │    Watches faces/ dir              │
│                     │                     │    Counts new frames               │
│ Repeat for all      │                     │    Copies to teams_dataset/        │
│ 901 sample pairs    │                     │    Reports status via HTTP API     │
└─────────────────────┘                     └────────────────────────────────────┘
```

---

## Receiver Server (Machine B) — `receiver_server.py`

### What It Does

- **Watches** `<session_dir>/participants/<participant>/faces/` for new `.json` files (the `.json` is written after the `.jpg`, so its appearance means the frame is fully written)
- **Tracks** a per-segment state machine: `IDLE → WARMING_UP → COLLECTING → COMPLETE`
- **Copies** collected frames into `<output_dir>/teams_dataset/<sample_id>/<type>/frame_NNNN.jpg`
- **Exposes** an HTTP API for the sender to control and query

### State Machine Per Segment

```
IDLE ──POST /segment/start──► WARMING_UP ──(3s timer)──► COLLECTING ──(≥16 frames)──► COMPLETE
                                   │                         │
                               (ignore frames)          (count + copy frames)
```

- **WARMING_UP (3 seconds):** Teams video pipeline has latency — the first few frames after a scene switch may show transition artifacts (partial green, previous face, blur). These are saved but not counted.
- **COLLECTING:** Every new face crop after warmup is counted toward the threshold and copied to the output directory.
- **COMPLETE:** ≥16 clean post-warmup frames captured. Sender can move on.

### CLI

```bash
python receiver_server.py \
  --session-dir "C:\...\debug_sessions\session_XXXXXXXX_XXXXXX" \
  --output-dir "C:\Users\dtect_dev1\Desktop\teams_capture_output" \
  --participant "Roy D" \
  --port 8080 \
  --min-frames 16 \
  --warmup-seconds 3.0
```

If `--session-dir` is omitted, auto-discovers the latest `session_*` directory under `C:\Users\dtect_dev1\Desktop\wma_debug\wma\debug_sessions\`.

### API Endpoints

| Method | Path | Body / Response |
|---|---|---|
| `POST /segment/start` | `{sample_id, type, strategy, index}` → `{ok, baseline_frames}` | Tell receiver a new segment is starting |
| `GET /segment/status` | → `{sample_id, type, state, warmup_frames, collected_frames, complete}` | Poll for segment completion |
| `GET /session/progress` | → `{completed_segments, total_expected, completed_pairs, current, elapsed_s}` | Overall session stats |
| `POST /session/start` | `{playlist: [...], participant}` | Initialize session (optional, for metadata) |
| `POST /session/end` | → `{summary}` | Finalize + write summary |
| `GET /health` | → `{status: "ok", session_dir, watching}` | Liveness check |

---

## Coordinated Sender (Machine A) — `obs_coordinated_sender.py`

### What It Does

- Downloads videos from GCS one-at-a-time
- Plays each through OBS virtual camera into Teams
- Signals receiver before/after each segment
- Waits for receiver to confirm capture before moving on
- Green screen between every segment (no face → clean break)

### Per-Segment Flow

```
1. Switch OBS → GreenScreen scene
2. Hold 2 seconds (drain previous face detections)
3. POST http://<machine_b>:8080/segment/start
   body: {sample_id: "edge_cases_0042", type: "real", strategy: "edge_cases", index: 83}
4. Wait 1 second  
5. Switch OBS → VideoPlayback scene (start playing the video)
6. Poll GET /segment/status every 1 second:
     - state: "warming_up" → keep waiting (3s warmup)
     - state: "collecting" → frames being counted
     - state: "complete"   → ≥16 frames captured, DONE
7. On "complete" → move to next segment
8. Timeout after 45s → log warning, continue anyway
```

### Per-Sample Flow

```
For each sample_id:
  [GreenScreen 2s] → signal(real) → play real.mp4 → poll until complete
  [GreenScreen 2s] → signal(fake) → play fake.mp4 → poll until complete
```

### Input Modes

```bash
# Full playlist (all 901 pairs):
python obs_coordinated_sender.py --password myPass \
  --receiver-url http://192.168.X.X:8080 \
  --playlist playlist.json

# Rerun only failed samples:
python obs_coordinated_sender.py --password myPass \
  --receiver-url http://192.168.X.X:8080 \
  --rerun-manifest rerun_manifest.json

# Resume after crash (skip first N segments):
python obs_coordinated_sender.py --password myPass \
  --receiver-url http://192.168.X.X:8080 \
  --playlist playlist.json --resume-from 42

# Test mode (2 samples per strategy, short warmup):
python obs_coordinated_sender.py --password myPass \
  --receiver-url http://192.168.X.X:8080 \
  --playlist playlist.json --test
```

### Time Estimate

| Phase | Duration | Count | Total |
|---|---|---|---|
| Green screen hold | 2s | 1,802 segments | 3,604s |
| Signal + settle | 1s | 1,802 | 1,802s |
| Warmup | 3s | 1,802 | 5,406s |
| Capture 16 frames (~3-5 fps through Teams) | ~4s | 1,802 | ~7,208s |
| **Total** | | | **~5 hours** |

Much faster than the original 8.5h fixed-duration approach, and with guaranteed quality.

---

## Output Directory Structure

Matches what `upload_teams_to_gcs.py` expects:

```
<output_dir>/
  teams_dataset/
    edge_cases_0000/
      real/
        frame_0000.jpg     ← 16+ face crops
        frame_0001.jpg
        ...
      fake/
        frame_0000.jpg
        ...
    edge_cases_0001/
      real/
      fake/
    ...
  warmup_frames/             ← frames captured during warmup (diagnostics)
    edge_cases_0000/
      real/
      fake/
  session_log.json           ← per-segment capture stats  
  playlist.json              ← copy of input playlist
```

---

## Prerequisites

### Machine B (Receiver — `dtect_dev1`)

```bash
pip install fastapi uvicorn
```

Running infrastructure:
- WMA app capturing Teams window
- `python_backend/server.py --debug-inspect` writing face crops
- Firewall rule for port 8080:
  ```powershell
  netsh advfirewall firewall add rule name="Teams Data Collection" dir=in action=allow protocol=TCP localport=8080
  ```

### Machine A (Sender)

```bash
pip install obsws-python google-cloud-storage requests
```

Running infrastructure:
- OBS Studio with WebSocket server enabled (port 4455)
- OBS virtual camera active and selected as Teams camera
- GreenScreen + VideoPlayback scenes (auto-created by script)

---

## Test Procedure

### Quick Local Test (Machine B only, no Teams needed)

```bash
# 1. Start receiver in test mode (creates fake session dir with test frames):
python receiver_server.py --test --port 8080

# 2. In another terminal, simulate sender requests:
python test_receiver.py --port 8080
```

The test script will:
1. POST `/segment/start` for a fake sample
2. Drop test `.jpg` + `.json` files into the watched directory
3. Poll `/segment/status` until complete
4. Verify frames were copied to output

### Integration Test (Both machines, Teams call)

1. Start Teams call between machines
2. Machine B: Start WMA + backend + receiver_server.py
3. Machine A: Run sender with `--test` flag (2 samples per strategy, ~2 min)
4. Verify `teams_dataset/` has correct structure
5. Run `upload_teams_to_gcs.py --dry-run` against output

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Sender can't reach receiver | Retries 3× then skips segment with warning |
| Segment times out (45s, no 16 frames) | Logs warning, saves partial, continues |
| WMA stops producing frames | `/segment/status` stays in COLLECTING; sender sees timeout eventually |
| Receiver crashes mid-session | Sender gets connection errors; resume-from saves progress |
| OBS disconnects | Sender catches exception, logs resume index, exits |
| Disk full on receiver | File copy fails, logged as error, segment marked partial |

---

## Files to Create

| File | Machine | Purpose |
|---|---|---|
| `receiver_server.py` | B (dtect_dev1) | ✅ FastAPI server watching WMA face crops |
| `test_receiver.py` | B (dtect_dev1) | ✅ Local test script for receiver |
| `obs_coordinated_sender.py` | A (broadcast) | Coordinated OBS playback sender |
| `COORDINATED_CAPTURE_PLAN.md` | Both | This document |
