#!/usr/bin/env python3
"""
OBS Coordinated Sender — Machine A

Plays DeepLive real+fake videos through OBS (→ Teams virtual camera) with
API-based coordination with a receiver server on Machine B.

Instead of blind timed playback, this script:
  1. Shows green screen (no face → clean break)
  2. Signals the receiver: "I'm about to play sample X / real"
  3. Plays the video via OBS
  4. Polls the receiver until it confirms ≥N frames captured
  5. Moves to the next segment

Prerequisites:
  pip install obsws-python google-cloud-storage requests

Usage:
  # Test mode (2 samples per strategy):
  python obs_coordinated_sender.py --password myPass \\
    --receiver-url http://192.168.X.X:8080 --test

  # Full run with playlist:
  python obs_coordinated_sender.py --password myPass \\
    --receiver-url http://192.168.X.X:8080

  # Rerun failed samples only:
  python obs_coordinated_sender.py --password myPass \\
    --receiver-url http://192.168.X.X:8080 \\
    --rerun-manifest rerun_manifest.json

  # Resume after crash (auto-skips already-completed samples):
  python obs_coordinated_sender.py --password myPass \\
    --receiver-url http://192.168.X.X:8080
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import obsws_python as obs
except ImportError:
    sys.exit("Missing obsws-python. Install: pip install obsws-python")

try:
    from google.cloud import storage as gcs
except ImportError:
    sys.exit("Missing google-cloud-storage. Install: pip install google-cloud-storage")

try:
    import requests
except ImportError:
    sys.exit("Missing requests. Install: pip install requests")


# ── Config ──────────────────────────────────────────────────────────────────

GCS_PROJECT = "train-cvit2"
VIDEOS_BUCKET = "live-deepfake-methods-real-and-fake-videos"

SCENE_GREEN = "GreenScreen"
SCENE_VIDEO = "VideoPlayback"
SOURCE_VIDEO = "VideoPlayer"
SOURCE_GREEN_COLOR = "GreenColor"
GREEN_HEX = 0xFF00FF00  # ABGR format for OBS — pure green

# If non-empty, only include these strategies (in this order).
# If empty, auto-discover all strategies from GCS.
STRATEGY_FILTER: list[str] = []

# Visomaster strategies get subsampled (there are ~600 each, way too many).
# Full run: take this fraction of each visomaster group.
# Test run: take VISOMASTER_TEST_N per visomaster group (vs 3 for normals).
VISO_SAMPLE_FRACTION = 0.06   # 6% ≈ 36 samples per visomaster group
VISO_TEST_N = 2
NORMAL_TEST_N = 6

# Timing
GREEN_HOLD_S = 2.0          # seconds of green screen between segments
POST_SIGNAL_WAIT_S = 1.0    # seconds after signaling receiver before starting video
POLL_INTERVAL_S = 1.0        # how often to poll receiver status
SEGMENT_TIMEOUT_S = 90.0     # give up on a segment after this many seconds


# ── Logging ─────────────────────────────────────────────────────────────────

class PlaybackLogger:
    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "a")

    def log(self, event: str, **kwargs):
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **kwargs,
        }
        self._f.write(json.dumps(entry) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


# ── Receiver API client ────────────────────────────────────────────────────

class ReceiverClient:
    """HTTP client for the Machine B receiver server."""

    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def session_start(self, playlist: list[dict], participant: str = "") -> dict:
        r = requests.post(
            f"{self.base_url}/session/start",
            json={"playlist": playlist, "participant": participant},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def session_end(self) -> dict:
        r = requests.post(f"{self.base_url}/session/end", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def segment_start(self, sample_id: str, vid_type: str,
                      strategy: str, index: int) -> dict:
        r = requests.post(
            f"{self.base_url}/segment/start",
            json={
                "sample_id": sample_id,
                "type": vid_type,
                "strategy": strategy,
                "index": index,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def segment_status(self) -> dict:
        r = requests.get(f"{self.base_url}/segment/status", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def segment_finalize(self) -> dict:
        """Explicitly finalize the current segment (flush frames to disk)."""
        r = requests.post(f"{self.base_url}/segment/finalize", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def session_progress(self) -> dict:
        r = requests.get(f"{self.base_url}/session/progress", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def completed_samples(self) -> tuple[set[str], dict[str, set[str]]]:
        """Query receiver for completed and partial sample info.

        Returns (fully_completed_ids, {sample_id: set of missing types}).
        e.g. (complete_set, {"edge_cases_0042": {"fake"}, "mp_0010": {"real"}})
        """
        r = requests.get(f"{self.base_url}/completed-samples", timeout=30.0)
        r.raise_for_status()
        data = r.json()
        complete = set(data.get("completed", []))
        # Build dict of partial samples → which side(s) are MISSING
        partials: dict[str, set[str]] = {}
        for p in data.get("partial", []):
            sid = p["sample_id"]
            missing = set()
            if p.get("real", 0) < 3:
                missing.add("real")
            if p.get("fake", 0) < 3:
                missing.add("fake")
            if missing:
                partials[sid] = missing
        return complete, partials


# ── GCS helpers ─────────────────────────────────────────────────────────────

def discover_samples(bucket) -> tuple[list[str], dict[str, list[str]]]:
    """Discover all strategies and samples from GCS.
    Returns (strategy_order, {strategy: [sample_ids]}).
    Only includes samples that have BOTH real.mp4 and fake.mp4.
    """
    print("Discovering samples in GCS bucket ...")
    # Pass 1: find all sample_ids that have real.mp4
    has_real: set[str] = set()
    has_fake: set[str] = set()
    for blob in bucket.list_blobs(prefix="samples/"):
        parts = blob.name.split("/")
        if len(parts) >= 3:
            sample_id = parts[1]
            if parts[2] == "real.mp4":
                has_real.add(sample_id)
            elif parts[2] == "fake.mp4":
                has_fake.add(sample_id)

    # Only keep samples with both files
    complete = has_real & has_fake
    incomplete = (has_real | has_fake) - complete
    if incomplete:
        print(f"  WARNING: {len(incomplete)} samples missing real or fake — skipping")

    raw: dict[str, list[str]] = {}
    for sample_id in complete:
        m = re.match(r"^(.+?)_(\d{4,})$", sample_id)
        if m:
            strat = m.group(1)
            if STRATEGY_FILTER and strat not in STRATEGY_FILTER:
                continue
            raw.setdefault(strat, []).append(sample_id)

    # Deterministic order: use STRATEGY_FILTER order if set, else alphabetical
    if STRATEGY_FILTER:
        order = [s for s in STRATEGY_FILTER if s in raw]
    else:
        order = sorted(raw.keys())

    strategy_samples = {s: sorted(raw[s]) for s in order}
    for strat in order:
        print(f"  {strat}: {len(strategy_samples[strat])} samples")
    total = sum(len(v) for v in strategy_samples.values())
    print(f"  Total: {total} samples ({total * 2} videos)")
    return order, strategy_samples


def build_playlist(strategy_order: list[str], strategy_samples: dict[str, list[str]]) -> list[dict]:
    playlist = []
    idx = 0
    for strat in strategy_order:
        for sample_id in strategy_samples.get(strat, []):
            for vid_type in ("real", "fake"):
                playlist.append({
                    "index": idx,
                    "sample_id": sample_id,
                    "type": vid_type,
                    "strategy": strat,
                })
                idx += 1
    return playlist


def download_video(bucket, sample_id: str, vid_type: str, tmp_dir: str) -> str:
    gcs_path = f"samples/{sample_id}/{vid_type}.mp4"
    local_path = os.path.join(tmp_dir, f"{sample_id}__{vid_type}.mp4")
    if os.path.exists(local_path):
        return local_path
    blob = bucket.blob(gcs_path)
    blob.download_to_filename(local_path)
    return local_path


def cleanup_video(local_path: str):
    try:
        if os.path.exists(local_path):
            os.remove(local_path)
    except OSError:
        pass


# ── OBS helpers ─────────────────────────────────────────────────────────────

def get_scene_names(cl: obs.ReqClient) -> list[str]:
    resp = cl.get_scene_list()
    return [s["sceneName"] for s in resp.scenes]


def _fit_source_to_canvas(cl: obs.ReqClient, scene: str, source: str):
    try:
        video = cl.get_video_settings()
        canvas_w = float(video.base_width)
        canvas_h = float(video.base_height)
        resp = cl.get_scene_item_id(scene, source)
        item_id = resp.scene_item_id
        cl.set_scene_item_transform(scene, item_id, {
            "positionX": 0.0, "positionY": 0.0,
            "boundsType": "OBS_BOUNDS_STRETCH",
            "boundsWidth": canvas_w, "boundsHeight": canvas_h,
            "boundsAlignment": 0,
        })
    except Exception as e:
        print(f"  Warning: Could not auto-scale {source}: {e}")


def ensure_scenes(cl: obs.ReqClient):
    existing = get_scene_names(cl)
    if SCENE_GREEN not in existing:
        print(f"Creating scene '{SCENE_GREEN}' ...")
        cl.create_scene(SCENE_GREEN)
        cl.create_input(
            SCENE_GREEN, SOURCE_GREEN_COLOR, "color_source_v3",
            {"color": GREEN_HEX, "width": 1920, "height": 1080}, True,
        )
    if SCENE_VIDEO not in existing:
        print(f"Creating scene '{SCENE_VIDEO}' ...")
        cl.create_scene(SCENE_VIDEO)
        cl.create_input(
            SCENE_VIDEO, SOURCE_VIDEO, "ffmpeg_source",
            {"local_file": "", "looping": True, "restart_on_activate": True}, True,
        )


def set_video_file(cl: obs.ReqClient, local_path: str):
    cl.set_input_settings(
        SOURCE_VIDEO,
        {"local_file": local_path, "looping": True, "restart_on_activate": True},
        True,
    )
    time.sleep(0.5)
    _fit_source_to_canvas(cl, SCENE_VIDEO, SOURCE_VIDEO)


def restart_media(cl: obs.ReqClient):
    cl.trigger_media_input_action(
        SOURCE_VIDEO, "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_RESTART"
    )


# ── Segment playback with coordination ─────────────────────────────────────

def play_segment_coordinated(
    cl: obs.ReqClient,
    receiver: ReceiverClient,
    bucket,
    entry: dict,
    tmp_dir: str,
    logger: PlaybackLogger,
) -> dict:
    """
    Play one video segment with receiver coordination.
    Returns a result dict with status.
    """
    sample_id = entry["sample_id"]
    vid_type = entry["type"]
    strategy = entry["strategy"]
    index = entry["index"]

    # 1. Show green screen (video gap — clears stale pipeline data)
    cl.set_current_program_scene(SCENE_GREEN)
    logger.log("green_screen", index=index, duration_s=GREEN_HOLD_S)
    time.sleep(GREEN_HOLD_S)

    # 2. Pre-download the video DURING green screen so it's ready to play
    #    immediately after signaling the receiver.  This avoids wasting
    #    warmup frames on green-screen / stale content.
    try:
        local_path = download_video(bucket, sample_id, vid_type, tmp_dir)
    except Exception as e:
        msg = f"Download failed for {sample_id}/{vid_type}.mp4: {e}"
        print(f"    SKIP: {msg}")
        logger.log("download_failed", index=index, sample_id=sample_id,
                    type=vid_type, error=str(e))
        return {"status": "download_failed", "error": str(e)}

    # 3. Load the video into OBS (but keep green screen visible)
    set_video_file(cl, local_path)

    # 4. Signal receiver — warmup begins NOW
    try:
        resp = receiver.segment_start(sample_id, vid_type, strategy, index)
        logger.log("segment_signal_sent", index=index, sample_id=sample_id,
                    type=vid_type, receiver_response=resp)
    except Exception as e:
        logger.log("segment_signal_failed", index=index, error=str(e))
        print(f"    WARNING: Could not signal receiver: {e}")
        cleanup_video(local_path)
        return {"status": "signal_failed", "error": str(e)}

    # 5. Brief pause, then switch to video — warmup absorbs this gap
    time.sleep(POST_SIGNAL_WAIT_S)
    cl.set_current_program_scene(SCENE_VIDEO)
    restart_media(cl)       # force-restart — OBS may be in "ended" state from prev video

    logger.log("video_start", index=index, sample_id=sample_id, type=vid_type,
               strategy=strategy)
    print(f"    Playing {sample_id}/{vid_type}.mp4 ...")

    # 4. Poll receiver until complete or timeout
    poll_start = time.time()
    last_status = {}

    while True:
        elapsed = time.time() - poll_start
        if elapsed > SEGMENT_TIMEOUT_S:
            print(f"    TIMEOUT after {elapsed:.0f}s — moving on")
            logger.log("segment_timeout", index=index, last_status=last_status)
            cleanup_video(local_path)
            return {"status": "timeout", "last_status": last_status}

        try:
            status = receiver.segment_status()
            last_status = status
        except Exception as e:
            print(f"    Poll error: {e}")
            time.sleep(POLL_INTERVAL_S)
            continue

        state = status.get("state", "unknown")
        collected = status.get("collected_frames", 0)
        warmup = status.get("warmup_frames", 0)

        if status.get("complete"):
            # Immediately finalize to flush frames to disk
            try:
                receiver.segment_finalize()
            except Exception as e:
                print(f"    WARNING: finalize failed: {e}")
            duration = time.time() - poll_start
            print(f"    COMPLETE: {collected} frames in {duration:.1f}s  "
                  f"(+{warmup} warmup)")
            logger.log("segment_complete", index=index, sample_id=sample_id,
                        type=vid_type, collected=collected, warmup=warmup,
                        duration_s=round(duration, 1))
            cleanup_video(local_path)
            return {"status": "complete", "collected": collected}

        if status.get("timed_out"):
            print(f"    Receiver timeout: {collected} frames")
            logger.log("segment_receiver_timeout", index=index,
                        collected=collected)
            cleanup_video(local_path)
            return {"status": "receiver_timeout", "collected": collected}

        # Progress indicator
        if state in ("warming_up", "discarding"):
            print(f"\r    [{state}] warmup frames: {warmup}         ", end="", flush=True)
        else:
            print(f"\r    [{state}] collected: {collected}         ", end="", flush=True)

        time.sleep(POLL_INTERVAL_S)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OBS Coordinated Sender — Machine A"
    )
    parser.add_argument("--host", default="localhost", help="OBS WebSocket host")
    parser.add_argument("--port", type=int, default=4455, help="OBS WebSocket port")
    parser.add_argument("--password", default=os.environ.get("OBS_WS_PASSWORD", ""),
                        help="OBS WebSocket password (leave empty if auth disabled)")
    parser.add_argument("--receiver-url", required=True,
                        help="Receiver server URL — either the old receiver_server.py "
                             "or the backend data-mode API "
                             "(e.g. http://192.168.1.100:8080)")
    parser.add_argument("--rerun-manifest", type=str, default=None,
                        help="Path to rerun_manifest.json (only stream listed samples)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Disable auto-skip of already-completed samples")
    parser.add_argument("--partials-only", action="store_true",
                        help="Only send the missing side of partial pairs (skip everything else)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for playlist.json and playback_log.jsonl")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: 2 samples per strategy")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build playlist, check receiver, don't play")
    parser.add_argument("--participant", type=str, default="",
                        help="Participant name filter (only collect this person's crops). "
                             "Empty string = no filter. Example: --participant 'Roy D'")
    args = parser.parse_args()

    print("=" * 60)
    print("OBS Coordinated Sender")
    print("=" * 60)

    # ── 1. Check receiver health ──
    print(f"\nChecking receiver at {args.receiver_url} ...")
    receiver = ReceiverClient(args.receiver_url)
    try:
        health = receiver.health()
        mode = health.get("mode", "receiver")
        if mode == "data":
            print(f"  Backend DATA MODE: output={health.get('watching')}, "
                  f"participant={health.get('participant')}")
        else:
            print(f"  Receiver OK: watching={health.get('watching')}, "
                  f"participant={health.get('participant')}")
    except Exception as e:
        sys.exit(f"Cannot reach receiver: {e}\n"
                 f"Make sure the receiver (receiver_server.py or backend --data-mode) "
                 f"is running on Machine B.")

    # ── 2. Discover samples and build playlist ──
    client = gcs.Client(project=GCS_PROJECT)
    bucket = client.bucket(VIDEOS_BUCKET)

    if args.rerun_manifest:
        with open(args.rerun_manifest) as f:
            manifest = json.load(f)
        rerun_list = manifest["rerun_samples"]
        print(f"\nRERUN MODE: {len(rerun_list)} samples from {args.rerun_manifest}")
        strategy_samples: dict[str, list[str]] = {}
        for entry in rerun_list:
            strategy_samples.setdefault(entry["strategy"], []).append(entry["sample_id"])
        strategy_order = sorted(strategy_samples.keys())
        for strat in strategy_order:
            strategy_samples[strat].sort()
    else:
        strategy_order, strategy_samples = discover_samples(bucket)

    # Subsample visomaster strategies (full run: 3%, test: 1 each)
    if not args.rerun_manifest:
        for strat in strategy_order:
            if strat.startswith("visomaster_") and not args.test:
                full = strategy_samples[strat]
                n_keep = max(1, int(len(full) * VISO_SAMPLE_FRACTION))
                strategy_samples[strat] = full[:n_keep]
                print(f"  {strat}: subsampled {len(full)} → {n_keep} ({VISO_SAMPLE_FRACTION:.0%})")

    if args.test:
        print(f"\nTEST MODE: {NORMAL_TEST_N}/strategy (visomaster: {VISO_TEST_N})")
        for strat in strategy_order:
            n = VISO_TEST_N if strat.startswith("visomaster_") else NORMAL_TEST_N
            strategy_samples[strat] = strategy_samples[strat][:n]

    playlist = build_playlist(strategy_order, strategy_samples)
    n_samples = len(playlist) // 2

    # Save playlist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    playlist_path = output_dir / "playlist.json"
    with open(playlist_path, "w") as f:
        json.dump(playlist, f, indent=2)
    print(f"\nPlaylist: {len(playlist)} entries ({n_samples} sample pairs)")

    # Time estimate (rough)
    per_segment_s = GREEN_HOLD_S + POST_SIGNAL_WAIT_S + 3.0 + 4.0  # green + signal + warmup + capture
    total_s = len(playlist) * per_segment_s
    print(f"Estimated time: ~{total_s / 3600:.1f} hours")

    if args.dry_run:
        print("\nDry run complete.")
        return

    # ── 3. Connect to OBS ──
    print(f"\nConnecting to OBS at {args.host}:{args.port} ...")
    try:
        # If password is empty, pass empty string (works with auth disabled)
        cl = obs.ReqClient(host=args.host, port=args.port, password=args.password or "")
    except Exception as e:
        sys.exit(f"OBS connection failed: {e}")
    print("  Connected.")
    ensure_scenes(cl)

    # ── 4. Signal receiver session start ──
    try:
        receiver.session_start(playlist, participant=args.participant)
        print(f"  Receiver session started (participant={args.participant!r}).")
    except Exception as e:
        print(f"  Warning: Could not start receiver session: {e}")

    # ── 5. Prepare ──
    log_path = output_dir / "playback_log.jsonl"
    logger = PlaybackLogger(str(log_path))
    # ── 5b. Auto-skip already-completed samples ──
    skip_ids: set[str] = set()
    partial_missing: dict[str, set[str]] = {}   # sample_id → {"real"} or {"fake"}
    if not args.no_skip:
        print("\nQuerying receiver for already-completed samples ...")
        try:
            skip_ids, partial_missing = receiver.completed_samples()
            if skip_ids:
                print(f"  Will skip {len(skip_ids)} already-completed samples")
            if partial_missing:
                print(f"  {len(partial_missing)} partial pairs found")
            if not skip_ids and not partial_missing:
                print("  No completed samples on receiver — processing all.")
            if args.partials_only:
                if not partial_missing:
                    print("  No partials to fix — nothing to do.")
                    return
                print(f"  --partials-only: will ONLY send {len(partial_missing)} missing sides")
        except Exception as e:
            print(f"  Warning: Could not query completed samples: {e}")
            print("  Proceeding without skip (all samples will be processed).")
            if args.partials_only:
                print("  Cannot run --partials-only without receiver info. Aborting.")
                return

    logger.log("session_start", playlist_entries=len(playlist),
               test_mode=args.test, skip_count=len(skip_ids))

    tmp_dir = tempfile.mkdtemp(prefix="obs_sender_")
    skipped = 0
    completed = 0
    started = time.time()
    n_remaining = n_samples - len(skip_ids)

    # ── 6. Main loop ──
    try:
        for entry in playlist:
            idx = entry["index"]
            sample_id = entry["sample_id"]
            vid_type = entry["type"]

            # Auto-skip completed samples (both sides done)
            if sample_id in skip_ids:
                if vid_type == "fake":
                    skipped += 1
                continue

            # --partials-only: skip anything that's NOT a missing side
            if args.partials_only:
                if sample_id not in partial_missing:
                    if vid_type == "fake":
                        skipped += 1
                    continue
                if vid_type not in partial_missing[sample_id]:
                    continue
            # Normal mode: skip already-completed side of partial pairs
            elif sample_id in partial_missing:
                if vid_type not in partial_missing[sample_id]:
                    continue

            # Progress header — print once per sample (on whichever side runs first)
            is_first_side = (vid_type == "real") or (
                sample_id in partial_missing and "real" not in partial_missing[sample_id]
            )
            if is_first_side:
                elapsed = time.time() - started
                pairs_done = completed
                if pairs_done > 0:
                    eta_h = (elapsed / pairs_done * (n_remaining - pairs_done)) / 3600
                else:
                    eta_h = total_s / 3600
                partial_tag = ""
                if sample_id in partial_missing:
                    needed = ", ".join(sorted(partial_missing[sample_id]))
                    partial_tag = f"  [partial — need {needed}]"
                print(f"\n[{idx}/{len(playlist)}]  {sample_id}  "
                      f"(pair {completed + 1}/{n_remaining}, "
                      f"skipped {skipped}, ETA: {eta_h:.1f}h){partial_tag}")

            result = play_segment_coordinated(
                cl, receiver, bucket, entry, tmp_dir, logger
            )

            if vid_type == "fake":
                completed += 1

    except KeyboardInterrupt:
        elapsed = time.time() - started
        print(f"\n\nInterrupted after {elapsed / 3600:.1f}h")
        print(f"Completed {completed} pairs, skipped {skipped}.  Just re-run to resume.")
        logger.log("interrupted", index=idx, completed=completed, skipped=skipped)
    except Exception as e:
        elapsed = time.time() - started
        print(f"\n\nError at index {idx}: {e}")
        print(f"Completed {completed} pairs, skipped {skipped}.  Just re-run to resume.")
        logger.log("error", index=idx, error=str(e), completed=completed, skipped=skipped)
        raise
    else:
        elapsed = time.time() - started
        print(f"\n\nDone in {elapsed / 3600:.1f}h — "
              f"{completed} pairs captured, {skipped} skipped (already complete)")
        logger.log("session_complete", pairs=completed, skipped=skipped)
    finally:
        # Show green screen at the end
        try:
            cl.set_current_program_scene(SCENE_GREEN)
        except Exception:
            pass

        # End receiver session
        try:
            summary = receiver.session_end()
            print(f"Receiver summary: {json.dumps(summary.get('summary', {}), indent=2)}")
        except Exception as e:
            print(f"Warning: Could not end receiver session: {e}")

        logger.close()
        cl.disconnect()

        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        print(f"\nLog: {log_path}")
        print(f"Playlist: {playlist_path}")


if __name__ == "__main__":
    main()
