#!/usr/bin/env python3
"""
OBS Video Player — Test Script

Downloads a real DeepLive video from GCS, loops it for a fixed duration
(default 20 s) via a virtual camera, then cleans up. Green screen shown before/after.

Prerequisites:
  1. OBS Studio running with WebSocket server enabled (Tools → WebSocket Server Settings)
     - Default port: 4455, set a password
  2. OBS scenes configured (the script will create them if missing):
     - "GreenScreen" scene with a green Color Source
     - "VideoPlayback" scene with a Media Source named "VideoPlayer"
  3. pip install obsws-python google-cloud-storage

Usage:
  python obs_video_player.py                          # pick first DeepLive sample
  python obs_video_player.py --sample-id edge_cases_0042
  python obs_video_player.py --password myObsPassword
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

try:
    import obsws_python as obs
except ImportError:
    sys.exit(
        "Missing obsws-python. Install with:\n"
        "  pip install obsws-python"
    )

try:
    from google.cloud import storage as gcs
except ImportError:
    sys.exit(
        "Missing google-cloud-storage. Install with:\n"
        "  pip install google-cloud-storage"
    )


# ── Config ──────────────────────────────────────────────────────────────────

GCS_PROJECT = "train-cvit2"
VIDEOS_BUCKET = "live-deepfake-methods-real-and-fake-videos"

# OBS object names (created automatically if they don't exist)
SCENE_GREEN = "GreenScreen"
SCENE_VIDEO = "VideoPlayback"
SOURCE_VIDEO = "VideoPlayer"
SOURCE_GREEN_COLOR = "GreenColor"

GREEN_HEX = 0xFF00FF00  # ABGR format for OBS — pure green


# ── GCS helpers ─────────────────────────────────────────────────────────────

def pick_sample_id(bucket, sample_id: str | None) -> str:
    """Return a valid sample_id, either the one requested or the first DeepLive one found."""
    if sample_id:
        # Verify it exists
        blob = bucket.blob(f"samples/{sample_id}/real.mp4")
        if not blob.exists():
            sys.exit(f"Sample '{sample_id}' not found in gs://{VIDEOS_BUCKET}/samples/")
        return sample_id

    print("No --sample-id given, discovering first available DeepLive sample...")
    for blob in bucket.list_blobs(prefix="samples/edge_cases_", max_results=50):
        if blob.name.endswith("/real.mp4"):
            # blob.name = "samples/edge_cases_0000/real.mp4"
            found = blob.name.split("/")[1]
            print(f"  → Using: {found}")
            return found

    sys.exit(f"No edge_cases_* samples found in gs://{VIDEOS_BUCKET}/samples/")


def download_video(bucket, sample_id: str, tmp_dir: str) -> str:
    """Download real.mp4 to a temp directory. Returns local path."""
    gcs_path = f"samples/{sample_id}/real.mp4"
    local_path = os.path.join(tmp_dir, f"{sample_id}__real.mp4")

    print(f"Downloading gs://{VIDEOS_BUCKET}/{gcs_path} ...")
    blob = bucket.blob(gcs_path)
    blob.download_to_filename(local_path)

    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"  → {local_path}  ({size_mb:.1f} MB)")
    return local_path


# ── OBS helpers ─────────────────────────────────────────────────────────────

def get_scene_names(cl: obs.ReqClient) -> list[str]:
    resp = cl.get_scene_list()
    return [s["sceneName"] for s in resp.scenes]


def ensure_scenes(cl: obs.ReqClient):
    """Create GreenScreen and VideoPlayback scenes + sources if missing."""
    existing = get_scene_names(cl)

    # ── GreenScreen scene ──
    if SCENE_GREEN not in existing:
        print(f"Creating scene '{SCENE_GREEN}' ...")
        cl.create_scene(SCENE_GREEN)
        # Add a green Color Source (OBS v30+ input kind: color_source_v3)
        cl.create_input(
            SCENE_GREEN,
            SOURCE_GREEN_COLOR,
            "color_source_v3",
            {"color": GREEN_HEX, "width": 1920, "height": 1080},
            True,
        )
    else:
        print(f"Scene '{SCENE_GREEN}' already exists.")

    # ── VideoPlayback scene ──
    if SCENE_VIDEO not in existing:
        print(f"Creating scene '{SCENE_VIDEO}' ...")
        cl.create_scene(SCENE_VIDEO)
        # Add a Media Source (ffmpeg_source) — file will be set later
        cl.create_input(
            SCENE_VIDEO,
            SOURCE_VIDEO,
            "ffmpeg_source",
            {"local_file": "", "looping": False, "restart_on_activate": True},
            True,
        )
    else:
        print(f"Scene '{SCENE_VIDEO}' already exists.")


def set_video_file(cl: obs.ReqClient, local_path: str):
    """Point the VideoPlayer media source at a local file and scale to fill canvas."""
    cl.set_input_settings(
        SOURCE_VIDEO,
        {"local_file": local_path, "looping": False, "restart_on_activate": True},
        True,
    )
    # Give OBS a moment to load the file and detect its resolution
    time.sleep(0.5)
    _fit_source_to_canvas(cl, SCENE_VIDEO, SOURCE_VIDEO)


def _fit_source_to_canvas(cl: obs.ReqClient, scene: str, source: str):
    """Force a source to fill the full canvas (1280×720) regardless of native resolution."""
    try:
        # Get canvas size from OBS video settings
        video = cl.get_video_settings()
        canvas_w = float(video.base_width)
        canvas_h = float(video.base_height)

        resp = cl.get_scene_item_id(scene, source)
        item_id = resp.scene_item_id

        cl.set_scene_item_transform(
            scene,
            item_id,
            {
                "positionX": 0.0,
                "positionY": 0.0,
                "boundsType": "OBS_BOUNDS_STRETCH",
                "boundsWidth": canvas_w,
                "boundsHeight": canvas_h,
                "boundsAlignment": 0,
            },
        )
        print(f"  → Video scaled to fill {int(canvas_w)}×{int(canvas_h)} canvas")
    except Exception as e:
        print(f"  ⚠ Could not auto-scale: {e} — resize manually in OBS")


def get_media_duration_ms(cl: obs.ReqClient) -> int:
    """Get current media source duration in ms."""
    resp = cl.get_media_input_status(SOURCE_VIDEO)
    return resp.media_duration or 0


def get_media_cursor_ms(cl: obs.ReqClient) -> int:
    """Get current playback position in ms."""
    resp = cl.get_media_input_status(SOURCE_VIDEO)
    return resp.media_cursor or 0


def get_media_state(cl: obs.ReqClient) -> str:
    """Get media state: OBS_MEDIA_STATE_PLAYING, _PAUSED, _ENDED, etc."""
    resp = cl.get_media_input_status(SOURCE_VIDEO)
    return resp.media_state or "OBS_MEDIA_STATE_NONE"


def restart_media(cl: obs.ReqClient):
    """Restart playback from the beginning."""
    cl.trigger_media_input_action(SOURCE_VIDEO, "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_RESTART")


def wait_for_playback_end(cl: obs.ReqClient, timeout_s: float = 300):
    """Block until the media source finishes playing."""
    time.sleep(0.5)  # let OBS start
    while True:
        state = get_media_state(cl)
        if state in ("OBS_MEDIA_STATE_ENDED", "OBS_MEDIA_STATE_STOPPED", "OBS_MEDIA_STATE_NONE"):
            break
        cursor = get_media_cursor_ms(cl)
        duration = get_media_duration_ms(cl)
        if duration > 0:
            pct = min(cursor / duration * 100, 100)
            print(f"\r  ▶ Playing: {cursor // 1000}s / {duration // 1000}s  ({pct:.0f}%)", end="", flush=True)
        time.sleep(0.5)
    print()  # newline after progress


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OBS Video Player — GCS DeepLive test")
    parser.add_argument("--sample-id", type=str, default=None,
                        help="GCS sample_id (e.g. edge_cases_0042). Auto-picks one if omitted.")
    parser.add_argument("--host", type=str, default="localhost", help="OBS WebSocket host")
    parser.add_argument("--port", type=int, default=4455, help="OBS WebSocket port")
    parser.add_argument("--password", type=str, default=os.environ.get("OBS_WS_PASSWORD", ""),
                        help="OBS WebSocket password (or set OBS_WS_PASSWORD env var)")
    parser.add_argument("--play-duration", type=float, default=20.0,
                        help="Total seconds to loop the video (default: 20)")
    parser.add_argument("--green-hold-s", type=float, default=3.0,
                        help="Seconds to hold green screen before/after video")
    parser.add_argument("--keep-video", action="store_true",
                        help="Don't delete the downloaded video after playback")
    args = parser.parse_args()

    # ── 1. Connect to GCS and download video ───
    print("=" * 60)
    print("OBS Video Player — Test Script")
    print("=" * 60)

    client = gcs.Client(project=GCS_PROJECT)
    bucket = client.bucket(VIDEOS_BUCKET)

    sample_id = pick_sample_id(bucket, args.sample_id)

    tmp_dir = tempfile.mkdtemp(prefix="obs_player_")
    video_path = download_video(bucket, sample_id, tmp_dir)

    # ── 2. Connect to OBS ───
    print(f"\nConnecting to OBS WebSocket at {args.host}:{args.port} ...")
    try:
        cl = obs.ReqClient(host=args.host, port=args.port, password=args.password)
    except Exception as e:
        sys.exit(
            f"Failed to connect to OBS WebSocket: {e}\n"
            f"Make sure OBS is running and WebSocket server is enabled "
            f"(Tools → WebSocket Server Settings)."
        )
    print("  → Connected to OBS.")

    # ── 3. Set up scenes ───
    ensure_scenes(cl)

    # ── 4. Show green screen ───
    print(f"\n🟩 Green screen — holding {args.green_hold_s}s ...")
    cl.set_current_program_scene(SCENE_GREEN)
    time.sleep(args.green_hold_s)

    # ── 5. Play video N times ───
    set_video_file(cl, video_path)

    # Brief pause to let OBS load the media file
    time.sleep(1.0)

    duration_ms = get_media_duration_ms(cl)
    vid_secs = duration_ms / 1000.0
    play_dur = args.play_duration
    print(f"\nVideo: {sample_id}/real.mp4  (duration: {vid_secs:.1f}s)")
    print(f"Looping for {play_dur:.0f}s total...\n")

    loop = 0
    wall_start = time.time()

    while True:
        elapsed = time.time() - wall_start
        remaining = play_dur - elapsed
        if remaining <= 0:
            break

        loop += 1
        print(f"── Loop {loop}  (elapsed {elapsed:.1f}s / {play_dur:.0f}s) ──")

        if loop == 1:
            cl.set_current_program_scene(SCENE_VIDEO)
        else:
            restart_media(cl)

        # Wait for this iteration to finish, but stop early if wall time exceeded
        time.sleep(0.5)  # let OBS start
        while True:
            state = get_media_state(cl)
            if state in ("OBS_MEDIA_STATE_ENDED", "OBS_MEDIA_STATE_STOPPED", "OBS_MEDIA_STATE_NONE"):
                break
            if time.time() - wall_start >= play_dur:
                break
            cursor = get_media_cursor_ms(cl)
            dur = get_media_duration_ms(cl)
            if dur > 0:
                pct = min(cursor / dur * 100, 100)
                print(f"\r  ▶ Playing: {cursor // 1000}s / {dur // 1000}s  ({pct:.0f}%)", end="", flush=True)
            time.sleep(0.5)
        print()  # newline

    total_elapsed = time.time() - wall_start
    print(f"\nPlayed {loop} loop(s) in {total_elapsed:.1f}s")

    # ── 6. Back to green screen ───
    print(f"\n🟩 Green screen — holding {args.green_hold_s}s ...")
    cl.set_current_program_scene(SCENE_GREEN)
    time.sleep(args.green_hold_s)

    # ── 7. Cleanup ───
    cl.disconnect()

    if args.keep_video:
        print(f"\nVideo kept at: {video_path}")
    else:
        os.remove(video_path)
        os.rmdir(tmp_dir)
        print(f"\nDeleted: {video_path}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
