#!/usr/bin/env python3
"""
OBS Simple Player — plays real OR fake videos from GCS through OBS virtual camera.

Usage:
  # Play all fake videos:
  python obs_simple_player.py --password myPass --label fake

  # Play all real videos:
  python obs_simple_player.py --password myPass --label real

  # Test mode (few samples per strategy):
  python obs_simple_player.py --password myPass --label fake --test

  # Only specific strategies:
  python obs_simple_player.py --password myPass --label fake --strategies deeplive visomaster_gan

  # Dry run (list what would play):
  python obs_simple_player.py --password myPass --label real --dry-run
"""

import argparse
import json
import os
import random
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


# ── Config ──────────────────────────────────────────────────────────────────

GCS_PROJECT = "train-cvit2"
VIDEOS_BUCKET = "live-deepfake-methods-real-and-fake-videos"

SCENE_GREEN = "GreenScreen"
SCENE_VIDEO = "VideoPlayback"
SOURCE_VIDEO = "VideoPlayer"
SOURCE_GREEN_COLOR = "GreenColor"
GREEN_HEX = 0xFF00FF00  # ABGR format for OBS — pure green

VISO_SAMPLE_FRACTION = 0.06
VISO_TEST_N = 1
NORMAL_TEST_N = 3

GREEN_HOLD_S = 0.3        # seconds of green screen between videos (just enough for source swap)
VIDEO_PLAY_S = 8.0         # seconds to play each video


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


# ── GCS helpers ─────────────────────────────────────────────────────────────

def discover_samples(bucket, label: str, strategies_filter: list[str] | None = None
                     ) -> tuple[list[str], dict[str, list[str]]]:
    """Discover strategies and samples from GCS that have the requested label video."""
    print(f"Discovering '{label}' samples in GCS bucket ...")
    has_file: set[str] = set()
    for blob in bucket.list_blobs(prefix="samples/"):
        parts = blob.name.split("/")
        if len(parts) >= 3 and parts[2] == f"{label}.mp4":
            has_file.add(parts[1])

    raw: dict[str, list[str]] = {}
    for sample_id in has_file:
        m = re.match(r"^(.+?)_(\d{4,})$", sample_id)
        if m:
            strat = m.group(1)
            if strategies_filter and strat not in strategies_filter:
                continue
            raw.setdefault(strat, []).append(sample_id)

    order = sorted(raw.keys())
    if strategies_filter:
        order = [s for s in strategies_filter if s in raw]

    strategy_samples = {s: sorted(raw[s]) for s in order}
    for strat in order:
        print(f"  {strat}: {len(strategy_samples[strat])} samples")
    total = sum(len(v) for v in strategy_samples.values())
    print(f"  Total: {total} videos")
    return order, strategy_samples


def download_video(bucket, sample_id: str, label: str, tmp_dir: str) -> str:
    gcs_path = f"samples/{sample_id}/{label}.mp4"
    local_path = os.path.join(tmp_dir, f"{sample_id}__{label}.mp4")
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


def set_cut_transition(cl: obs.ReqClient):
    """Set OBS scene transition to Cut (instant)."""
    try:
        cl.set_current_scene_transition("Cut")
        print("  Transition set to Cut (instant).")
    except Exception as e:
        print(f"  Warning: Could not set Cut transition: {e}")


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


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OBS Simple Player — play real or fake videos from GCS")
    parser.add_argument("--host", default="localhost", help="OBS WebSocket host")
    parser.add_argument("--port", type=int, default=4455, help="OBS WebSocket port")
    parser.add_argument("--password", default=os.environ.get("OBS_WS_PASSWORD", ""),
                        help="OBS WebSocket password")
    parser.add_argument("--label", required=True, choices=["real", "fake"],
                        help="Which video type to play")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Only play these strategies (space-separated)")
    parser.add_argument("--play-seconds", type=float, default=VIDEO_PLAY_S,
                        help=f"Seconds to play each video (default: {VIDEO_PLAY_S})")
    parser.add_argument("--green-seconds", type=float, default=GREEN_HOLD_S,
                        help=f"Seconds of green screen between videos (default: {GREEN_HOLD_S})")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for log files")
    parser.add_argument("--test", action="store_true",
                        help=f"Test mode: {NORMAL_TEST_N}/strategy ({VISO_TEST_N} for visomaster)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would play without connecting to OBS")
    parser.add_argument("--shuffle", action="store_true",
                        help="Randomize playlist order")
    args = parser.parse_args()

    print("=" * 60)
    print(f"OBS Simple Player — label={args.label}")
    print("=" * 60)

    # ── 1. Discover samples ──
    client = gcs.Client(project=GCS_PROJECT)
    bucket = client.bucket(VIDEOS_BUCKET)
    strategy_order, strategy_samples = discover_samples(bucket, args.label, args.strategies)

    # Subsample visomaster
    for strat in strategy_order:
        if strat.startswith("visomaster_") and not args.test:
            full = strategy_samples[strat]
            n_keep = max(1, int(len(full) * VISO_SAMPLE_FRACTION))
            strategy_samples[strat] = full[:n_keep]
            print(f"  {strat}: subsampled {len(full)} → {n_keep}")

    if args.test:
        print(f"\nTEST MODE: {NORMAL_TEST_N}/strategy (visomaster: {VISO_TEST_N})")
        for strat in strategy_order:
            n = VISO_TEST_N if strat.startswith("visomaster_") else NORMAL_TEST_N
            strategy_samples[strat] = strategy_samples[strat][:n]

    # Build flat playlist
    playlist = []
    for strat in strategy_order:
        for sample_id in strategy_samples.get(strat, []):
            playlist.append({"sample_id": sample_id, "strategy": strat})

    if args.shuffle:
        random.shuffle(playlist)
        print("  Playlist shuffled.")

    total = len(playlist)
    est_s = total * (args.play_seconds + args.green_seconds)
    print(f"\n{total} videos to play — estimated ~{est_s / 60:.1f} min")

    if args.dry_run:
        for i, entry in enumerate(playlist):
            print(f"  {i + 1:4d}  {entry['strategy']:30s}  {entry['sample_id']}")
        print("\nDry run complete.")
        return

    # ── 2. Connect to OBS ──
    print(f"\nConnecting to OBS at {args.host}:{args.port} ...")
    try:
        cl = obs.ReqClient(host=args.host, port=args.port, password=args.password or "")
    except Exception as e:
        sys.exit(f"OBS connection failed: {e}")
    print("  Connected.")
    ensure_scenes(cl)
    set_cut_transition(cl)

    # ── 3. Play loop ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"simple_playback_{args.label}.jsonl"
    logger = PlaybackLogger(str(log_path))
    logger.log("session_start", label=args.label, total=total, test_mode=args.test)

    tmp_dir = tempfile.mkdtemp(prefix="obs_simple_")
    completed = 0
    started = time.time()

    # Pre-download the first video before entering the loop
    next_local_path = None
    if playlist:
        first = playlist[0]
        try:
            next_local_path = download_video(bucket, first["sample_id"], args.label, tmp_dir)
        except Exception as e:
            print(f"  Pre-download failed for first video: {e}")

    try:
        for i, entry in enumerate(playlist):
            sample_id = entry["sample_id"]
            strategy = entry["strategy"]

            elapsed = time.time() - started
            if completed > 0:
                eta_m = (elapsed / completed * (total - completed)) / 60
            else:
                eta_m = est_s / 60
            print(f"\n[{i + 1}/{total}]  {strategy}/{sample_id}  "
                  f"(ETA: {eta_m:.1f}m)")

            # Use pre-downloaded file, or download now if pre-download failed
            local_path = next_local_path
            next_local_path = None
            if local_path is None:
                try:
                    local_path = download_video(bucket, sample_id, args.label, tmp_dir)
                except Exception as e:
                    print(f"  SKIP (download failed): {e}")
                    logger.log("download_failed", sample_id=sample_id, error=str(e))
                    # Still try to pre-download the next one
                    if i + 1 < total:
                        nxt = playlist[i + 1]
                        try:
                            next_local_path = download_video(bucket, nxt["sample_id"], args.label, tmp_dir)
                        except Exception:
                            pass
                    continue

            # Brief green screen (just enough for the source swap)
            cl.set_current_program_scene(SCENE_GREEN)
            time.sleep(args.green_seconds)

            # Play
            set_video_file(cl, local_path)
            cl.set_current_program_scene(SCENE_VIDEO)
            restart_media(cl)
            logger.log("video_start", sample_id=sample_id, strategy=strategy)
            print(f"  Playing for {args.play_seconds}s ...")

            # Pre-download the NEXT video while this one plays
            if i + 1 < total:
                nxt = playlist[i + 1]
                try:
                    next_local_path = download_video(bucket, nxt["sample_id"], args.label, tmp_dir)
                except Exception as e:
                    print(f"  Warning: pre-download of next video failed: {e}")
                    next_local_path = None

            time.sleep(args.play_seconds)

            cleanup_video(local_path)
            completed += 1
            logger.log("video_done", sample_id=sample_id, strategy=strategy)

    except KeyboardInterrupt:
        elapsed = time.time() - started
        print(f"\n\nInterrupted after {elapsed / 60:.1f}min — {completed}/{total} played")
        logger.log("interrupted", completed=completed)
    else:
        elapsed = time.time() - started
        print(f"\n\nDone in {elapsed / 60:.1f}min — {completed}/{total} played")
        logger.log("session_complete", completed=completed)
    finally:
        try:
            cl.set_current_program_scene(SCENE_GREEN)
        except Exception:
            pass
        logger.close()
        cl.disconnect()
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
