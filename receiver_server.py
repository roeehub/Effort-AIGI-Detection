#!/usr/bin/env python3
"""
Receiver Server — Coordinated Teams Data Collection (Machine B)

Watches the WMA debug inspector's face crop output directory for a specific
participant and exposes an HTTP API that the sender (Machine A) uses to
coordinate video playback with face capture.

The WMA app's capture pipeline stays completely untouched.  This server is
a passive observer that reads the .jpg/.json files WMA writes to disk.

Usage:
  # Normal mode (auto-discover latest session):
  python receiver_server.py --output-dir "C:\\capture_output" --port 8080

  # Explicit session directory:
  python receiver_server.py \\
    --session-dir "C:\\...\\debug_sessions\\session_20260228_XXXXXX" \\
    --output-dir "C:\\capture_output" --port 8080

  # Test mode (creates a fake session dir, drops test frames automatically):
  python receiver_server.py --test --port 8080
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("receiver")


# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_SESSION_BASE = Path(r"C:\Users\dtect_dev1\Desktop\wma_debug\wma\debug_sessions")
DEFAULT_PARTICIPANT = "Roy D"
DEFAULT_MIN_FRAMES = 16
DEFAULT_WARMUP_S = 3.0
DEFAULT_PORT = 8080
POLL_INTERVAL_S = 0.25          # how often to check for new files
MIN_FILE_SIZE_BYTES = 5000      # ignore tiny/corrupt files (conservative; upload uses 15KB)
SEGMENT_TIMEOUT_S = 90.0       # auto-fail a segment after this long (aligned w/ sender 60s + margin)
BBOX_OUTLIER_THRESHOLD = 60     # max bbox-center deviation from median (pixels) before rejection
MAX_COLLECT_FRAMES = 30         # cap frames to copy (avoids 100+ frame copies that block the API)


# ── Data models ─────────────────────────────────────────────────────────────

class SegmentState(str, Enum):
    IDLE = "idle"
    WARMING_UP = "warming_up"
    COLLECTING = "collecting"
    COMPLETE = "complete"
    TIMEOUT = "timeout"


class SegmentStartRequest(BaseModel):
    sample_id: str
    type: str               # "real" or "fake"
    strategy: str = ""
    index: int = -1


class SessionStartRequest(BaseModel):
    playlist: list[dict] = []
    participant: str = DEFAULT_PARTICIPANT


# ── Core state ──────────────────────────────────────────────────────────────

class ReceiverState:
    """Holds all mutable state — one instance shared by the watcher thread and API."""

    def __init__(
        self,
        faces_dir: Path,
        output_dir: Path,
        participant: str,
        min_frames: int,
        warmup_seconds: float,
    ):
        self.faces_dir = faces_dir
        self.output_dir = output_dir
        self.participant = participant
        self.min_frames = min_frames
        self.warmup_seconds = warmup_seconds

        # Current segment
        self.seg_state = SegmentState.IDLE
        self.seg_sample_id: str = ""
        self.seg_type: str = ""
        self.seg_strategy: str = ""
        self.seg_index: int = -1
        self.seg_start_time: float = 0.0
        self.seg_baseline: dict[str, float] = {}     # snapshot {filename: mtime} taken at segment start
        self.seg_collected_frames: list[str] = []    # files counted toward threshold
        self.seg_frame_data: dict[str, bytes] = {}   # {json_name: jpg_bytes} cached at detection time

        # Session-level stats
        self.session_start_time: float = time.time()
        self.completed_segments: int = 0
        self.completed_pairs: int = 0
        self.segment_log: list[dict] = []            # per-segment capture stats
        self.playlist: list[dict] = []
        self.total_expected: int = 0

        self.lock = threading.Lock()

    # ── File scanning ──

    def _scan_recent_files(self, after: float) -> list[str]:
        """Return .json filenames modified after the given timestamp.

        Uses os.scandir() for ~5x speed vs listdir+stat on large directories.
        WMA reuses filenames in a circular buffer, so mtime-based detection
        is the only reliable approach.
        """
        try:
            if not self.faces_dir.exists():
                return []
            result = []
            faces_str = str(self.faces_dir)
            with os.scandir(faces_str) as it:
                for entry in it:
                    if not entry.name.endswith(".json"):
                        continue
                    try:
                        stat = entry.stat()
                        if stat.st_mtime > after:
                            result.append(entry.name)
                    except OSError:
                        pass
            return sorted(result)
        except OSError:
            return []

    def _corresponding_jpg(self, json_name: str) -> str:
        """frame_000042_seq1537.json → frame_000042_seq1537.jpg"""
        return json_name[:-5] + ".jpg"

    def _is_valid_frame(self, json_name: str) -> bool:
        """Check the .jpg companion exists and is large enough."""
        jpg_path = self.faces_dir / self._corresponding_jpg(json_name)
        try:
            return jpg_path.exists() and jpg_path.stat().st_size >= MIN_FILE_SIZE_BYTES
        except OSError:
            return False

    def _read_bbox_center(self, json_name: str) -> tuple[float, float] | None:
        """Read the bbox center (cx, cy) from a face crop's companion JSON."""
        json_path = self.faces_dir / json_name
        try:
            with open(json_path) as f:
                meta = json.load(f)
            bb = meta.get("bbox", {})
            x, y, w, h = bb.get("x", 0), bb.get("y", 0), bb.get("width", 0), bb.get("height", 0)
            if w > 0 and h > 0:
                return (x + w / 2.0, y + h / 2.0)
        except (OSError, json.JSONDecodeError, KeyError):
            pass
        return None

    def _filter_bbox_outliers(self, json_names: list[str]) -> tuple[list[str], list[str]]:
        """Remove frames whose bbox center is far from the group median.

        Returns (good_frames, rejected_frames).
        This catches stale face crops from previous videos that were written
        to disk after the baseline snapshot (race condition).
        """
        if len(json_names) < 3:
            return json_names, []

        # Collect bbox centers
        centers: list[tuple[str, float, float]] = []
        no_meta: list[str] = []  # keep frames with no metadata (benefit of doubt)
        for jn in json_names:
            c = self._read_bbox_center(jn)
            if c:
                centers.append((jn, c[0], c[1]))
            else:
                no_meta.append(jn)

        if len(centers) < 3:
            return json_names, []

        # Compute median center
        xs = sorted(c[1] for c in centers)
        ys = sorted(c[2] for c in centers)
        med_x = xs[len(xs) // 2]
        med_y = ys[len(ys) // 2]

        good = []
        rejected = []
        for jn, cx, cy in centers:
            dist = ((cx - med_x) ** 2 + (cy - med_y) ** 2) ** 0.5
            if dist > BBOX_OUTLIER_THRESHOLD:
                rejected.append(jn)
                log.warning(
                    f"REJECTED outlier frame {jn}: center=({cx:.0f},{cy:.0f}) "
                    f"median=({med_x:.0f},{med_y:.0f}) dist={dist:.0f}px"
                )
            else:
                good.append(jn)

        good.extend(no_meta)
        return good, rejected

    # ── Directory snapshot (baseline-based sync) ──

    def _full_snapshot(self) -> dict[str, float]:
        """Take a snapshot of {filename: mtime} for all .json files in faces_dir.

        This is the "baseline" — any file whose mtime later changes relative
        to this snapshot (or any new file that appears) is a genuinely new
        capture from the currently-playing video.
        """
        snapshot: dict[str, float] = {}
        try:
            if not self.faces_dir.exists():
                return snapshot
            with os.scandir(str(self.faces_dir)) as it:
                for entry in it:
                    if not entry.name.endswith(".json"):
                        continue
                    try:
                        snapshot[entry.name] = entry.stat().st_mtime
                    except OSError:
                        pass
        except OSError:
            pass
        return snapshot

    def _scan_changed_since_baseline(self, baseline: dict[str, float]) -> list[str]:
        """Return .json filenames that are new or modified since the baseline.

        A file is 'changed' if:
          - It didn't exist in the baseline (brand-new file), OR
          - Its mtime has increased since the baseline (circular buffer slot
            was overwritten with a new face crop).
        """
        try:
            if not self.faces_dir.exists():
                return []
            result = []
            with os.scandir(str(self.faces_dir)) as it:
                for entry in it:
                    if not entry.name.endswith(".json"):
                        continue
                    try:
                        st = entry.stat()
                        prev_mtime = baseline.get(entry.name)
                        if prev_mtime is None:
                            # Brand-new file — not in baseline
                            result.append(entry.name)
                        elif st.st_mtime > prev_mtime + 0.001:
                            # File was overwritten (mtime increased)
                            result.append(entry.name)
                    except OSError:
                        pass
            return sorted(result)
        except OSError:
            return []

    # ── Segment lifecycle ──

    def start_segment(self, sample_id: str, vid_type: str, strategy: str, index: int) -> dict:
        with self.lock:
            # Take a full directory snapshot as the baseline.  The sender
            # calls this AFTER the video is already playing and the pipeline
            # has settled, so any file change after this snapshot must be a
            # new face crop from the current video.
            self.seg_baseline = self._full_snapshot()
            self.seg_sample_id = sample_id
            self.seg_type = vid_type
            self.seg_strategy = strategy
            self.seg_index = index
            self.seg_start_time = time.time()
            self.seg_collected_frames = []
            self.seg_frame_data = {}  # fresh cache for this segment
            # Go directly to COLLECTING — no warmup needed because:
            # 1. Green screen has flushed the WMA pipeline
            # 2. The video is already playing and has settled
            # 3. The baseline snapshot excludes all pre-existing data
            self.seg_state = SegmentState.COLLECTING

            if len(self.seg_baseline) == 0:
                log.warning(
                    f"EMPTY BASELINE for {sample_id}/{vid_type} — "
                    f"faces_dir may not exist yet or scan failed"
                )

            log.info(
                f"SEGMENT START  sample={sample_id} type={vid_type} "
                f"index={index}  baseline={len(self.seg_baseline)} files"
            )
            return {
                "ok": True,
                "baseline_files": len(self.seg_baseline),
            }

    def _cache_frame_bytes(self, json_names: list[str],
                           existing_cache: dict[str, bytes]) -> dict[str, bytes]:
        """Read JPG bytes for newly detected files into memory.

        Only reads files not already in existing_cache.  This is called
        BEFORE marking COMPLETE, while the sender's video is still playing
        and the files on disk still contain the correct face.  This
        eliminates the race where a background copy reads files that WMA
        has already overwritten with the next video's faces.
        """
        cache = dict(existing_cache)
        for jn in json_names:
            if jn in cache:
                continue  # already cached
            jpg_path = self.faces_dir / self._corresponding_jpg(jn)
            try:
                data = jpg_path.read_bytes()
                if len(data) >= MIN_FILE_SIZE_BYTES:
                    cache[jn] = data
            except OSError:
                pass
        return cache

    def tick(self):
        """Called by the watcher thread every POLL_INTERVAL_S.

        Uses baseline-snapshot comparison: any file whose mtime changed
        (or that didn't exist) relative to the snapshot taken at segment
        start is a new capture from the currently-playing video.

        KEY SAFETY INVARIANT: JPG bytes are read into memory BEFORE the
        segment is marked COMPLETE.  The sender only sees COMPLETE after
        the bytes are cached, so when the sender moves on and WMA starts
        overwriting files, we are no longer reading from the live dir.
        """
        # ── Phase 1: snapshot state (brief lock) ──
        with self.lock:
            if self.seg_state in (SegmentState.IDLE, SegmentState.COMPLETE, SegmentState.TIMEOUT):
                return
            state = self.seg_state
            start_time = self.seg_start_time
            baseline = dict(self.seg_baseline)   # copy for lock-free scan
            min_frames = self.min_frames
            cached = dict(self.seg_frame_data)   # existing cache snapshot

        now = time.time()

        # Timeout guard
        if now - start_time > SEGMENT_TIMEOUT_S:
            with self.lock:
                if self.seg_state in (SegmentState.COMPLETE, SegmentState.TIMEOUT):
                    return
                log.warning(
                    f"TIMEOUT  sample={self.seg_sample_id}/{self.seg_type}  "
                    f"collected={len(self.seg_collected_frames)} frames  "
                    f"cached={len(self.seg_frame_data)}"
                )
                self.seg_state = SegmentState.TIMEOUT
                self._schedule_finalize(timed_out=True)
            return

        # ── Phase 2: scan for changes since baseline (NO lock held) ──
        changed = self._scan_changed_since_baseline(baseline)
        valid_new = [f for f in changed if self._is_valid_frame(f)]

        # ── Phase 2b: sanity check — too many "new" files? ──
        if len(valid_new) > MAX_COLLECT_FRAMES * 3:
            log.warning(
                f"SUSPICIOUS: {len(valid_new)} changed files detected "
                f"(baseline had {len(baseline)} files). "
                f"Possible empty/stale baseline — re-snapshotting."
            )
            # Re-take baseline and skip this tick
            new_baseline = self._full_snapshot()
            with self.lock:
                if self.seg_state != state:
                    return
                self.seg_baseline = new_baseline
            return

        # ── Phase 3: cache JPG bytes for new files (NO lock, disk I/O) ──
        # This reads from the live faces_dir while video is STILL PLAYING.
        # The sender won't see COMPLETE until Phase 4, so the files are
        # guaranteed to contain the current video's faces.
        new_cache = self._cache_frame_bytes(valid_new, cached)

        # ── Phase 4: update state (brief lock) ──
        with self.lock:
            if self.seg_state != state:
                return

            self.seg_collected_frames = valid_new
            self.seg_frame_data = new_cache
            # Count frames that actually have cached bytes
            n_cached = sum(1 for f in valid_new if f in new_cache)
            n = len(valid_new)

            if n_cached >= min_frames:
                log.info(
                    f"COMPLETE  sample={self.seg_sample_id}/{self.seg_type}  "
                    f"detected={n} cached={n_cached} frames  "
                    f"(baseline: {len(baseline)} files)"
                )
                self.seg_state = SegmentState.COMPLETE
                self._schedule_finalize(timed_out=False)

    # ── Background finalization ──

    def _schedule_finalize(self, timed_out: bool):
        """Snapshot data and spawn background thread for bbox filter + copy.

        MUST be called while holding self.lock.  Does minimal work under
        the lock — just copies scalar/list values, then starts a daemon
        thread that does all expensive I/O.
        """
        snap = dict(
            sample_id=self.seg_sample_id,
            type=self.seg_type,
            strategy=self.seg_strategy,
            index=self.seg_index,
            start_time=self.seg_start_time,
            collected_frames=list(self.seg_collected_frames),
            frame_data=dict(self.seg_frame_data),  # CACHED JPG bytes
            timed_out=timed_out,
            baseline_size=len(self.seg_baseline),
        )
        # Clear the cache from state (free memory)
        self.seg_frame_data = {}
        threading.Thread(
            target=self._finalize_worker, args=(snap,), daemon=True
        ).start()

    def _finalize_worker(self, snap: dict):
        """Background thread: bbox filter, cap, update stats, copy files.

        All expensive I/O (reading JSON metadata, copying JPEGs) runs here
        without holding the lock.  The lock is only acquired briefly to
        update session-level stats.
        """
        sample_id = snap["sample_id"]
        vid_type = snap["type"]
        raw_collected = snap["collected_frames"]
        frame_data = snap.get("frame_data", {})  # {json_name: jpg_bytes}
        timed_out = snap["timed_out"]
        baseline_size = snap.get("baseline_size", 0)

        # Only keep frames that have cached bytes
        cached_collected = [f for f in raw_collected if f in frame_data]
        if len(cached_collected) < len(raw_collected):
            log.warning(
                f"MISSING CACHE: {len(raw_collected) - len(cached_collected)}/"
                f"{len(raw_collected)} frames had no cached bytes for "
                f"{sample_id}/{vid_type} — using only cached frames"
            )

        # ── Filter bbox outliers (no lock — reads JSON files) ──
        good_frames, rejected = self._filter_bbox_outliers(cached_collected)
        if rejected:
            log.warning(
                f"FILTERED {len(rejected)}/{len(cached_collected)} outlier frames "
                f"for {sample_id}/{vid_type}"
            )

        # ── Cap to MAX_COLLECT_FRAMES (evenly spaced) ──
        if len(good_frames) > MAX_COLLECT_FRAMES:
            step = len(good_frames) / MAX_COLLECT_FRAMES
            collected = [good_frames[int(i * step)] for i in range(MAX_COLLECT_FRAMES)]
            log.info(f"CAPPED {len(good_frames)} → {len(collected)} frames for {sample_id}/{vid_type}")
        else:
            collected = good_frames

        # ── Update stats (brief lock) ──
        with self.lock:
            seg_entry = {
                "sample_id": sample_id,
                "type": vid_type,
                "strategy": snap["strategy"],
                "index": snap["index"],
                "baseline_files": baseline_size,
                "raw_collected": len(raw_collected),
                "cached_frames": len(cached_collected),
                "rejected_outliers": len(rejected),
                "copied_frames": len(collected),
                "timed_out": timed_out,
                "duration_s": round(time.time() - snap["start_time"], 1),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            self.segment_log.append(seg_entry)
            self.completed_segments += 1

            # Track pair completion
            completed_samples: dict[str, set[str]] = {}
            for entry in self.segment_log:
                sid = entry["sample_id"]
                if sid not in completed_samples:
                    completed_samples[sid] = set()
                completed_samples[sid].add(entry["type"])
            self.completed_pairs = sum(
                1 for types in completed_samples.values()
                if "real" in types and "fake" in types
            )

        # ── Write frames from CACHED BYTES (no lock, no live-dir reads) ──
        dest_dir = self.output_dir / "teams_dataset" / sample_id / vid_type
        # Clear any existing files (retries / previous runs)
        if dest_dir.exists():
            for old in dest_dir.iterdir():
                try:
                    old.unlink()
                except OSError:
                    pass
        dest_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for i, json_name in enumerate(collected):
            jpg_bytes = frame_data.get(json_name)
            if jpg_bytes is None:
                continue  # shouldn't happen — already filtered
            dst = dest_dir / f"frame_{i:04d}.jpg"
            try:
                dst.write_bytes(jpg_bytes)
                written += 1
            except OSError as e:
                log.error(f"Failed to write {dst}: {e}")
        log.info(
            f"SAVED  {sample_id}/{vid_type}: {written}/{len(collected)} frames → {dest_dir}  "
            f"(from cached bytes, baseline={baseline_size})"
        )

    def get_segment_status(self) -> dict:
        with self.lock:
            return {
                "sample_id": self.seg_sample_id,
                "type": self.seg_type,
                "state": self.seg_state.value,
                "warmup_frames": 0,  # Legacy field — no warmup in baseline approach
                "baseline_files": len(self.seg_baseline),
                "collected_frames": len(self.seg_collected_frames),
                "min_frames": self.min_frames,
                "complete": self.seg_state == SegmentState.COMPLETE,
                "timed_out": self.seg_state == SegmentState.TIMEOUT,
                "elapsed_s": round(time.time() - self.seg_start_time, 1) if self.seg_start_time else 0,
            }

    def get_session_progress(self) -> dict:
        with self.lock:
            return {
                "completed_segments": self.completed_segments,
                "total_expected": self.total_expected,
                "completed_pairs": self.completed_pairs,
                "current": {
                    "sample_id": self.seg_sample_id,
                    "type": self.seg_type,
                    "state": self.seg_state.value,
                },
                "elapsed_s": round(time.time() - self.session_start_time, 1),
                "segment_log_count": len(self.segment_log),
            }

    def save_session_log(self):
        """Write session_log.json to output dir."""
        log_path = self.output_dir / "session_log.json"
        summary = {
            "session_start": datetime.fromtimestamp(
                self.session_start_time, tz=timezone.utc
            ).isoformat(),
            "session_end": datetime.now(timezone.utc).isoformat(),
            "participant": self.participant,
            "min_frames": self.min_frames,
            "warmup_seconds": self.warmup_seconds,
            "completed_segments": self.completed_segments,
            "completed_pairs": self.completed_pairs,
            "faces_dir": str(self.faces_dir),
            "output_dir": str(self.output_dir),
            "segments": self.segment_log,
        }
        with open(log_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Session log saved: {log_path}")


# ── Watcher thread ──────────────────────────────────────────────────────────

def watcher_thread(state: ReceiverState, stop_event: threading.Event):
    """Background thread that polls the faces directory for new files."""
    log.info(f"Watcher started — polling {state.faces_dir} every {POLL_INTERVAL_S}s")
    while not stop_event.is_set():
        state.tick()
        stop_event.wait(POLL_INTERVAL_S)
    log.info("Watcher stopped.")


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(title="Teams Data Collection Receiver", version="1.0")
_state: ReceiverState | None = None
_stop_event = threading.Event()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "session_dir": str(_state.faces_dir.parent.parent) if _state else None,
        "faces_dir": str(_state.faces_dir) if _state else None,
        "watching": _state is not None,
        "participant": _state.participant if _state else None,
    }


@app.post("/session/start")
def session_start(req: SessionStartRequest):
    if _state is None:
        raise HTTPException(500, "Receiver not initialized")
    with _state.lock:
        _state.playlist = req.playlist
        _state.total_expected = len(req.playlist)
        _state.session_start_time = time.time()
    log.info(f"Session started with {len(req.playlist)} playlist entries")
    return {"ok": True, "total_expected": len(req.playlist)}


@app.post("/session/end")
def session_end():
    if _state is None:
        raise HTTPException(500, "Receiver not initialized")
    _state.save_session_log()
    return {
        "ok": True,
        "summary": {
            "completed_segments": _state.completed_segments,
            "completed_pairs": _state.completed_pairs,
            "segment_log": _state.segment_log,
        },
    }


@app.post("/segment/start")
def segment_start(req: SegmentStartRequest):
    if _state is None:
        raise HTTPException(500, "Receiver not initialized")
    if req.type not in ("real", "fake"):
        raise HTTPException(400, f"Invalid type: {req.type!r}  (expected 'real' or 'fake')")
    result = _state.start_segment(req.sample_id, req.type, req.strategy, req.index)
    return result


@app.get("/segment/status")
def segment_status():
    if _state is None:
        raise HTTPException(500, "Receiver not initialized")
    return _state.get_segment_status()


@app.get("/session/progress")
def session_progress():
    if _state is None:
        raise HTTPException(500, "Receiver not initialized")
    return _state.get_session_progress()


MIN_FRAMES_COMPLETE = 3   # a side is "done" if it has >= this many frames

@app.get("/completed-samples")
def completed_samples():
    """Scan teams_dataset/ on disk and return sample_ids that have both real+fake with enough frames."""
    if _state is None:
        raise HTTPException(500, "Receiver not initialized")
    td = _state.output_dir / "teams_dataset"
    if not td.exists():
        return {"completed": [], "partial": [], "total_dirs": 0}

    completed = []
    partial = []
    for d in sorted(td.iterdir()):
        if not d.is_dir():
            continue
        real_dir = d / "real"
        fake_dir = d / "fake"
        rc = len(list(real_dir.glob("*.jpg"))) if real_dir.exists() else 0
        fc = len(list(fake_dir.glob("*.jpg"))) if fake_dir.exists() else 0
        if rc >= MIN_FRAMES_COMPLETE and fc >= MIN_FRAMES_COMPLETE:
            completed.append(d.name)
        elif rc > 0 or fc > 0:
            partial.append({"sample_id": d.name, "real": rc, "fake": fc})

    return {
        "completed": completed,
        "partial": partial,
        "total_dirs": len(completed) + len(partial),
    }


# ── Session directory discovery ─────────────────────────────────────────────

def find_latest_session(base: Path) -> Path | None:
    """Find the most recently created session_YYYYMMDD_HHMMSS directory."""
    if not base.exists():
        return None
    sessions = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("session_")],
        key=lambda d: d.name,
        reverse=True,
    )
    return sessions[0] if sessions else None


def resolve_faces_dir(session_dir: Path, participant: str) -> Path:
    """Locate the participant's faces directory within a session."""
    faces = session_dir / "participants" / participant / "faces"
    return faces


# ── Test mode ───────────────────────────────────────────────────────────────

def run_test_mode(port: int):
    """
    Self-contained test: creates a fake session dir, starts the server,
    spawns a thread that simulates face crops appearing, and validates the flow.
    """
    import io

    log.info("=" * 60)
    log.info("TEST MODE — simulated session")
    log.info("=" * 60)

    tmp_root = Path(tempfile.mkdtemp(prefix="receiver_test_"))
    session_dir = tmp_root / "session_test"
    session_dir.mkdir()
    faces_dir = session_dir / "participants" / DEFAULT_PARTICIPANT / "faces"
    faces_dir.mkdir(parents=True)
    output_dir = tmp_root / "output"
    output_dir.mkdir()

    log.info(f"Temp session dir:  {session_dir}")
    log.info(f"Faces dir:         {faces_dir}")
    log.info(f"Output dir:        {output_dir}")

    global _state
    _state = ReceiverState(
        faces_dir=faces_dir,
        output_dir=output_dir,
        participant=DEFAULT_PARTICIPANT,
        min_frames=4,           # low threshold for test
        warmup_seconds=2.0,     # short warmup for test
    )

    stop = threading.Event()
    t = threading.Thread(target=watcher_thread, args=(_state, stop), daemon=True)
    t.start()

    def _fake_frame_dropper():
        """Simulate WMA writing face crops to disk — runs alongside the server."""
        import requests
        time.sleep(2)  # let server start

        base_url = f"http://127.0.0.1:{port}"
        log.info("[TEST] Checking health...")
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            log.info(f"[TEST] Health: {r.json()}")
        except Exception as e:
            log.error(f"[TEST] Cannot reach server: {e}")
            return

        # --- Test one segment ---
        sample_id = "test_sample_001"
        vid_type = "real"

        log.info(f"[TEST] Starting segment: {sample_id}/{vid_type}")
        r = requests.post(f"{base_url}/segment/start", json={
            "sample_id": sample_id,
            "type": vid_type,
            "strategy": "test",
            "index": 0,
        })
        log.info(f"[TEST] Start response: {r.json()}")

        # Drop frames during warmup (these should NOT be counted)
        log.info("[TEST] Dropping 3 warmup frames...")
        for i in range(3):
            time.sleep(0.3)
            _write_fake_frame(faces_dir, 1000 + i, 5000 + i)

        # Wait for warmup to end
        time.sleep(2.5)

        # Drop frames during collection (these SHOULD be counted)
        log.info("[TEST] Dropping 6 collection frames...")
        for i in range(6):
            time.sleep(0.3)
            _write_fake_frame(faces_dir, 2000 + i, 6000 + i)

        # Poll status
        for _ in range(20):
            time.sleep(0.5)
            r = requests.get(f"{base_url}/segment/status")
            status = r.json()
            log.info(
                f"[TEST] Status: state={status['state']}  "
                f"warmup={status['warmup_frames']}  "
                f"collected={status['collected_frames']}  "
                f"complete={status['complete']}"
            )
            if status["complete"]:
                break

        # Check output
        out_real = output_dir / "teams_dataset" / sample_id / vid_type
        if out_real.exists():
            frames = sorted(out_real.glob("*.jpg"))
            log.info(f"[TEST] OUTPUT: {len(frames)} frames in {out_real}")
            for f in frames:
                log.info(f"[TEST]   {f.name}  ({f.stat().st_size:,} bytes)")
        else:
            log.warning(f"[TEST] No output directory at {out_real}")

        # End session
        r = requests.post(f"{base_url}/session/end")
        log.info(f"[TEST] Session end: {json.dumps(r.json()['summary'], indent=2)}")

        log.info("")
        log.info("=" * 60)
        log.info("TEST COMPLETE — check output above for correctness")
        log.info(f"  Expected: >= 4 frames in {out_real}")
        log.info(f"  Warmup frames: {output_dir / 'warmup_frames' / sample_id / vid_type}")
        log.info("=" * 60)
        log.info("")
        log.info("Server still running — you can also test manually with curl.")
        log.info(f"  curl http://127.0.0.1:{port}/health")
        log.info(f"  curl http://127.0.0.1:{port}/session/progress")

    # Start the frame dropper in a background thread
    dropper = threading.Thread(target=_fake_frame_dropper, daemon=True)
    dropper.start()

    log.info(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

    stop.set()


def _write_fake_frame(faces_dir: Path, frame_num: int, seq_num: int):
    """Write a fake face crop .jpg + .json pair to simulate WMA output."""
    stem = f"frame_{frame_num:06d}_seq{seq_num}"

    # Write a minimal valid JPEG (small red square)
    # Smallest valid JPEG header + some padding to exceed MIN_FILE_SIZE_BYTES
    try:
        import cv2
        import numpy as np
        img = np.random.randint(50, 200, (120, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpg_bytes = buf.tobytes()
    except ImportError:
        # Fallback: write raw bytes large enough to pass size check
        jpg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 8000 + b"\xff\xd9"

    jpg_path = faces_dir / f"{stem}.jpg"
    with open(jpg_path, "wb") as f:
        f.write(jpg_bytes)

    # Write companion JSON (mirrors debug_inspector format)
    meta = {
        "timestamp_ms": int(time.time() * 1000),
        "sequence_number": seq_num,
        "participant_id_raw": DEFAULT_PARTICIPANT,
        "participant_id_clean": DEFAULT_PARTICIPANT,
        "image_path": f"participants/{DEFAULT_PARTICIPANT}/faces/{stem}.jpg",
        "bbox": {"x": 100, "y": 80, "width": 100, "height": 120},
        "confidence": 0.91,
        "has_video": True,
        "frame_size": [100, 120],
        "file_size_bytes": len(jpg_bytes),
        "score": None,
        "verdict": None,
        "batch_id": None,
    }
    json_path = faces_dir / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Receiver Server — Coordinated Teams Data Collection"
    )
    parser.add_argument(
        "--session-dir", type=str, default=None,
        help="WMA debug session directory. If omitted, auto-discovers the latest session.",
    )
    parser.add_argument(
        "--session-base", type=str,
        default=str(DEFAULT_SESSION_BASE),
        help=f"Base directory containing session_* folders (default: {DEFAULT_SESSION_BASE})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save organized output (teams_dataset/). "
             "Default: creates a timestamped dir on Desktop.",
    )
    parser.add_argument(
        "--participant", type=str, default=DEFAULT_PARTICIPANT,
        help=f"Participant name to watch (default: {DEFAULT_PARTICIPANT!r})",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"HTTP server port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--min-frames", type=int, default=DEFAULT_MIN_FRAMES,
        help=f"Frames required per segment (default: {DEFAULT_MIN_FRAMES})",
    )
    parser.add_argument(
        "--warmup-seconds", type=float, default=DEFAULT_WARMUP_S,
        help=f"Seconds to ignore frames after segment start (default: {DEFAULT_WARMUP_S})",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run in test mode with simulated face crops",
    )
    args = parser.parse_args()

    # ── Test mode shortcut ──
    if args.test:
        run_test_mode(args.port)
        return

    # ── Resolve session directory ──
    if args.session_dir:
        session_dir = Path(args.session_dir)
    else:
        log.info(f"Auto-discovering latest session in {args.session_base} ...")
        session_dir_maybe = find_latest_session(Path(args.session_base))
        if session_dir_maybe is None:
            sys.exit(f"No session directories found in {args.session_base}")
        session_dir = session_dir_maybe
        log.info(f"  → Using: {session_dir.name}")

    if not session_dir.exists():
        sys.exit(f"Session directory does not exist: {session_dir}")

    faces_dir = resolve_faces_dir(session_dir, args.participant)
    log.info(f"Session dir: {session_dir}")
    log.info(f"Faces dir:   {faces_dir}")

    if not faces_dir.exists():
        log.warning(
            f"Faces directory does not exist yet: {faces_dir}\n"
            f"  This is OK if WMA hasn't started capture yet — "
            f"the watcher will detect it once it appears."
        )

    # ── Resolve output directory ──
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(
            rf"C:\Users\dtect_dev1\Desktop\teams_capture_{ts}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output dir:  {output_dir}")

    # ── Initialize state ──
    global _state
    _state = ReceiverState(
        faces_dir=faces_dir,
        output_dir=output_dir,
        participant=args.participant,
        min_frames=args.min_frames,
        warmup_seconds=args.warmup_seconds,
    )

    # ── Start watcher thread ──
    stop = threading.Event()
    t = threading.Thread(target=watcher_thread, args=(_state, stop), daemon=True)
    t.start()

    # ── Start HTTP server ──
    log.info("")
    log.info("=" * 60)
    log.info(f"  Receiver Server listening on  0.0.0.0:{args.port}")
    log.info(f"  Participant:  {args.participant}")
    log.info(f"  Min frames:   {args.min_frames}")
    log.info(f"  Warmup:       {args.warmup_seconds}s")
    log.info(f"  Output:       {output_dir}")
    log.info("=" * 60)
    log.info("")

    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        stop.set()
        _state.save_session_log()
        log.info("Done.")


if __name__ == "__main__":
    main()
