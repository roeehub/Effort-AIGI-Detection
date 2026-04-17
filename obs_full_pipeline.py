#!/usr/bin/env python3
"""
OBS Full Pipeline — Overnight DeepLive Video Playback

Plays all DeepLive real+fake videos through OBS (→ Teams virtual camera)
with celebrity face separators to enable post-hoc stream segmentation.

Separator Roles:
  A_sep.jpg  — Video separator: shown between every clip (3s)
  B_sep.jpg  — Real↔Fake divider: between real and fake of same sample (3s)
  C_sep.jpg  — Strategy group header: at start of each strategy group (7s)

Playback Order (deterministic):
  [C 7s]  ← strategy header (edge_cases)
    [A 3s] → edge_cases_0000/real.mp4 (12s) → [B 3s] → edge_cases_0000/fake.mp4 (12s)
    [A 3s] → edge_cases_0001/real.mp4 (12s) → [B 3s] → edge_cases_0001/fake.mp4 (12s)
    ...
  [C 7s]  ← strategy header (minimal_processing)
    ...
  [C 7s]  ← strategy header (quality_enhancement)
    ...

Outputs:
  playlist.json      — full ordered manifest (generated before playback)
  playback_log.jsonl — real-time timestamped events

Prerequisites:
  pip install obsws-python google-cloud-storage

Usage:
  # Test mode (2 samples per strategy, short durations):
  python obs_full_pipeline.py --password myPass --test

  # Full overnight run:
  python obs_full_pipeline.py --password myPass

  # Resume after crash:
  python obs_full_pipeline.py --password myPass --resume-from 42
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


# ── Config ──────────────────────────────────────────────────────────────────

GCS_PROJECT = "train-cvit2"
VIDEOS_BUCKET = "live-deepfake-methods-real-and-fake-videos"

SEPARATOR_DIR = Path(__file__).parent / "DeepfakeBench" / "training" / "debug" / "face_seperators"
SEP_A = str(SEPARATOR_DIR / "A_sep.jpg")   # video separator
SEP_B = str(SEPARATOR_DIR / "B_sep.jpg")   # real↔fake divider
SEP_C = str(SEPARATOR_DIR / "C_sep.jpg")   # strategy group header

# OBS scene/source names
SCENE_VIDEO = "VideoPlayback"
SCENE_SEP_A = "SeparatorA"
SCENE_SEP_B = "SeparatorB"
SCENE_SEP_C = "SeparatorC"

SOURCE_VIDEO = "VideoPlayer"
SOURCE_IMG_A = "ImgSepA"
SOURCE_IMG_B = "ImgSepB"
SOURCE_IMG_C = "ImgSepC"

# Strategy ordering (deterministic)
STRATEGY_ORDER = ["edge_cases", "minimal_processing", "quality_enhancement"]


# ── Logging ─────────────────────────────────────────────────────────────────

class PlaybackLogger:
    """Append-only JSONL logger with timestamps."""

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

def discover_samples(bucket) -> dict[str, list[str]]:
    """
    Scan the GCS bucket and return {strategy: [sample_id, ...]} sorted.
    Only includes strategies in STRATEGY_ORDER.
    """
    print("Discovering samples in GCS bucket ...")
    strategy_samples: dict[str, list[str]] = {s: [] for s in STRATEGY_ORDER}

    for blob in bucket.list_blobs(prefix="samples/"):
        parts = blob.name.split("/")
        if len(parts) >= 3 and parts[2] == "real.mp4":
            sample_id = parts[1]
            m = re.match(r"^(.+?)_(\d{4,})$", sample_id)
            if m and m.group(1) in strategy_samples:
                strategy_samples[m.group(1)].append(sample_id)

    for strat in STRATEGY_ORDER:
        strategy_samples[strat].sort()
        print(f"  {strat}: {len(strategy_samples[strat])} samples")

    total = sum(len(v) for v in strategy_samples.values())
    print(f"  Total: {total} samples ({total * 2} videos)")
    return strategy_samples


def build_playlist(strategy_samples: dict[str, list[str]]) -> list[dict]:
    """Build ordered playlist: for each sample, real then fake."""
    playlist = []
    idx = 0
    for strat in STRATEGY_ORDER:
        for sample_id in strategy_samples[strat]:
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
    """Download a video to tmp_dir. Returns local path."""
    gcs_path = f"samples/{sample_id}/{vid_type}.mp4"
    local_path = os.path.join(tmp_dir, f"{sample_id}__{vid_type}.mp4")

    if os.path.exists(local_path):
        return local_path

    blob = bucket.blob(gcs_path)
    blob.download_to_filename(local_path)
    return local_path


def cleanup_video(local_path: str):
    """Remove a downloaded video file."""
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
    """Force a source to stretch-fill the OBS canvas."""
    try:
        video = cl.get_video_settings()
        canvas_w = float(video.base_width)
        canvas_h = float(video.base_height)

        resp = cl.get_scene_item_id(scene, source)
        item_id = resp.scene_item_id

        cl.set_scene_item_transform(scene, item_id, {
            "positionX": 0.0,
            "positionY": 0.0,
            "boundsType": "OBS_BOUNDS_STRETCH",
            "boundsWidth": canvas_w,
            "boundsHeight": canvas_h,
            "boundsAlignment": 0,
        })
    except Exception as e:
        print(f"  ⚠ Could not auto-scale {source}: {e}")


def ensure_scenes(cl: obs.ReqClient):
    """Create all required OBS scenes and sources if missing."""
    existing = get_scene_names(cl)

    # ── Video playback scene ──
    if SCENE_VIDEO not in existing:
        print(f"Creating scene '{SCENE_VIDEO}' ...")
        cl.create_scene(SCENE_VIDEO)
        cl.create_input(
            SCENE_VIDEO, SOURCE_VIDEO, "ffmpeg_source",
            {"local_file": "", "looping": False, "restart_on_activate": True},
            True,
        )
    else:
        print(f"Scene '{SCENE_VIDEO}' exists.")

    # ── Separator image scenes ──
    sep_configs = [
        (SCENE_SEP_A, SOURCE_IMG_A, SEP_A),
        (SCENE_SEP_B, SOURCE_IMG_B, SEP_B),
        (SCENE_SEP_C, SOURCE_IMG_C, SEP_C),
    ]
    for scene_name, source_name, img_path in sep_configs:
        if scene_name not in existing:
            print(f"Creating scene '{scene_name}' with {Path(img_path).name} ...")
            cl.create_scene(scene_name)
            cl.create_input(
                scene_name, source_name, "image_source",
                {"file": img_path},
                True,
            )
            # Scale image to fill canvas
            time.sleep(0.3)
            _fit_source_to_canvas(cl, scene_name, source_name)
        else:
            print(f"Scene '{scene_name}' exists.")


def show_separator(cl: obs.ReqClient, sep_type: str, duration_s: float, logger: PlaybackLogger):
    """Switch to a separator scene and hold for duration_s."""
    scene_map = {"A": SCENE_SEP_A, "B": SCENE_SEP_B, "C": SCENE_SEP_C}
    scene = scene_map[sep_type]
    cl.set_current_program_scene(scene)
    logger.log("separator", marker=sep_type, duration_s=duration_s)
    time.sleep(duration_s)


def set_video_file(cl: obs.ReqClient, local_path: str):
    """Point the VideoPlayer media source at a local file and scale to fill canvas."""
    cl.set_input_settings(
        SOURCE_VIDEO,
        {"local_file": local_path, "looping": False, "restart_on_activate": True},
        True,
    )
    time.sleep(0.5)
    _fit_source_to_canvas(cl, SCENE_VIDEO, SOURCE_VIDEO)


def get_media_state(cl: obs.ReqClient) -> str:
    resp = cl.get_media_input_status(SOURCE_VIDEO)
    return resp.media_state or "OBS_MEDIA_STATE_NONE"


def get_media_duration_ms(cl: obs.ReqClient) -> int:
    resp = cl.get_media_input_status(SOURCE_VIDEO)
    return resp.media_duration or 0


def get_media_cursor_ms(cl: obs.ReqClient) -> int:
    resp = cl.get_media_input_status(SOURCE_VIDEO)
    return resp.media_cursor or 0


def restart_media(cl: obs.ReqClient):
    cl.trigger_media_input_action(SOURCE_VIDEO, "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_RESTART")


def play_video_for_duration(cl: obs.ReqClient, local_path: str, play_duration_s: float,
                            entry: dict, logger: PlaybackLogger):
    """Play a video looping for play_duration_s seconds."""
    set_video_file(cl, local_path)
    time.sleep(0.5)

    dur_ms = get_media_duration_ms(cl)
    vid_secs = dur_ms / 1000.0

    logger.log("video_start", index=entry["index"], sample_id=entry["sample_id"],
               type=entry["type"], strategy=entry["strategy"],
               video_duration_s=round(vid_secs, 1), play_duration_s=play_duration_s)

    sid = entry["sample_id"]
    vtype = entry["type"]
    print(f"    ▶ {sid}/{vtype}.mp4  ({vid_secs:.1f}s vid, looping {play_duration_s:.0f}s)")

    cl.set_current_program_scene(SCENE_VIDEO)

    loop = 0
    wall_start = time.time()

    while True:
        elapsed = time.time() - wall_start
        if elapsed >= play_duration_s:
            break

        loop += 1
        if loop > 1:
            restart_media(cl)

        # Wait for this playback to end or wall time to expire
        time.sleep(0.3)
        while True:
            state = get_media_state(cl)
            if state in ("OBS_MEDIA_STATE_ENDED", "OBS_MEDIA_STATE_STOPPED", "OBS_MEDIA_STATE_NONE"):
                break
            if time.time() - wall_start >= play_duration_s:
                break
            time.sleep(0.4)

    total_elapsed = time.time() - wall_start
    logger.log("video_end", index=entry["index"], loops=loop,
               actual_duration_s=round(total_elapsed, 1))


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OBS Full Pipeline — Overnight DeepLive playback")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=4455)
    parser.add_argument("--password", default=os.environ.get("OBS_WS_PASSWORD", ""))
    parser.add_argument("--play-duration", type=float, default=12.0,
                        help="Seconds to loop each video (default: 12)")
    parser.add_argument("--sep-a-duration", type=float, default=3.0,
                        help="Video separator duration in seconds (default: 3)")
    parser.add_argument("--sep-b-duration", type=float, default=3.0,
                        help="Real↔Fake divider duration in seconds (default: 3)")
    parser.add_argument("--sep-c-duration", type=float, default=7.0,
                        help="Strategy header duration in seconds (default: 7)")
    parser.add_argument("--resume-from", type=int, default=None,
                        help="Resume from playlist index (skip earlier entries)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for playlist.json and playback_log.jsonl")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: 2 samples per strategy, shorter durations")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build playlist and print time estimate, don't play")
    parser.add_argument("--rerun-manifest", type=str, default=None,
                        help="Path to rerun_manifest.json — only stream listed samples")

    args = parser.parse_args()

    # Test mode overrides
    if args.test:
        args.play_duration = 6.0
        args.sep_a_duration = 2.0
        args.sep_b_duration = 2.0
        args.sep_c_duration = 4.0

    # Verify separator images exist
    for label, path in [("A", SEP_A), ("B", SEP_B), ("C", SEP_C)]:
        if not os.path.exists(path):
            sys.exit(f"Separator image not found: {path}")
        print(f"  Separator {label}: {path}")

    # ── 1. Discover samples and build playlist ──
    print("\n" + "=" * 60)
    print("OBS Full Pipeline — DeepLive Overnight Playback")
    print("=" * 60)

    client = gcs.Client(project=GCS_PROJECT)
    bucket = client.bucket(VIDEOS_BUCKET)

    # Load rerun manifest metadata (needed for logging even before playlist)
    manifest = None
    if args.rerun_manifest:
        with open(args.rerun_manifest) as f:
            manifest = json.load(f)

    if args.rerun_manifest:
        # ── Rerun mode: load manifest instead of full discovery ──
        rerun_list = manifest["rerun_samples"]
        print(f"\n🔄 RERUN MODE: loading {args.rerun_manifest}")
        print(f"   {len(rerun_list)} sample pairs to re-stream")

        # Build strategy_samples from manifest (preserves strategy grouping)
        strategy_samples = {s: [] for s in STRATEGY_ORDER}
        for entry in rerun_list:
            strategy_samples[entry["strategy"]].append(entry["sample_id"])

        # Sort within each strategy for deterministic ordering
        for strat in STRATEGY_ORDER:
            strategy_samples[strat].sort()

        # In test mode, limit to 2 samples per strategy
        if args.test:
            print("\n⚡ TEST MODE: limiting to 2 samples per strategy")
            for strat in STRATEGY_ORDER:
                strategy_samples[strat] = strategy_samples[strat][:2]

        for strat in STRATEGY_ORDER:
            print(f"  {strat}: {len(strategy_samples[strat])} samples")
        total = sum(len(v) for v in strategy_samples.values())
        print(f"  Total: {total} samples ({total * 2} videos)")

        playlist = build_playlist(strategy_samples)
    else:
        # ── Normal mode: discover all samples from GCS ──
        strategy_samples = discover_samples(bucket)

        # In test mode, limit to 2 samples per strategy
        if args.test:
            print("\n⚡ TEST MODE: limiting to 2 samples per strategy")
            for strat in STRATEGY_ORDER:
                strategy_samples[strat] = strategy_samples[strat][:2]

        playlist = build_playlist(strategy_samples)

    # Save playlist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    playlist_path = output_dir / "playlist.json"
    with open(playlist_path, "w") as f:
        json.dump(playlist, f, indent=2)
    print(f"\nPlaylist saved: {playlist_path}  ({len(playlist)} entries)")

    # Time estimate
    n_samples = len(playlist) // 2
    n_strategies = sum(1 for s in STRATEGY_ORDER if strategy_samples[s])
    vid_time = len(playlist) * args.play_duration
    sep_a_time = n_samples * args.sep_a_duration           # A before each sample pair
    sep_b_time = n_samples * args.sep_b_duration           # B between real and fake
    sep_c_time = n_strategies * args.sep_c_duration        # C at strategy boundaries
    total_s = vid_time + sep_a_time + sep_b_time + sep_c_time
    total_h = total_s / 3600

    print(f"\nTime estimate:")
    print(f"  Videos:       {len(playlist)} × {args.play_duration}s = {vid_time:.0f}s")
    print(f"  Sep A (clip): {n_samples} × {args.sep_a_duration}s = {sep_a_time:.0f}s")
    print(f"  Sep B (r↔f):  {n_samples} × {args.sep_b_duration}s = {sep_b_time:.0f}s")
    print(f"  Sep C (strat):{n_strategies} × {args.sep_c_duration}s = {sep_c_time:.0f}s")
    print(f"  ─────────────────────────────────")
    print(f"  Total:        {total_s:.0f}s  ({total_h:.1f} hours)")

    if args.dry_run:
        print("\n🏁 Dry run complete. No playback.")
        return

    # ── 2. Connect to OBS ──
    print(f"\nConnecting to OBS at {args.host}:{args.port} ...")
    try:
        cl = obs.ReqClient(host=args.host, port=args.port, password=args.password)
    except Exception as e:
        sys.exit(f"OBS connection failed: {e}")
    print("  → Connected.")

    ensure_scenes(cl)

    # ── 3. Prepare logger ──
    log_path = output_dir / "playback_log.jsonl"
    logger = PlaybackLogger(str(log_path))
    logger.log("session_start", playlist_entries=len(playlist),
               play_duration=args.play_duration,
               test_mode=args.test,
               resume_from=args.resume_from,
               rerun_mode=bool(args.rerun_manifest),
               rerun_source=manifest["metadata"]["source_session"]
                   if manifest else None)

    # ── 4. Create temp dir for downloads ──
    tmp_dir = tempfile.mkdtemp(prefix="obs_pipeline_")
    print(f"Temp dir: {tmp_dir}")

    # ── 5. Main playback loop ──
    resume_idx = args.resume_from or 0
    current_strategy = None
    prev_sample_id = None

    started = time.time()
    completed_pairs = 0
    total_pairs = n_samples

    try:
        for entry in playlist:
            idx = entry["index"]
            sample_id = entry["sample_id"]
            vid_type = entry["type"]
            strategy = entry["strategy"]

            # Skip if resuming
            if idx < resume_idx:
                if vid_type == "fake":
                    completed_pairs += 1
                continue

            # ── Strategy header (C separator) ──
            if strategy != current_strategy:
                current_strategy = strategy
                n_strat = len(strategy_samples[strategy])
                print(f"\n{'=' * 60}")
                print(f"🔷 STRATEGY: {strategy}  ({n_strat} samples)")
                print(f"{'=' * 60}")
                logger.log("strategy_header", strategy=strategy, count=n_strat)
                show_separator(cl, "C", args.sep_c_duration, logger)

            # ── Video separator (A) — before each sample's real video ──
            if vid_type == "real":
                elapsed_total = time.time() - started
                remaining_pairs = total_pairs - completed_pairs
                eta_s = (elapsed_total / max(completed_pairs, 1)) * remaining_pairs if completed_pairs > 0 else total_s
                eta_h = eta_s / 3600
                print(f"\n  [{idx}/{len(playlist)}]  "
                      f"Sample: {sample_id}  "
                      f"(pair {completed_pairs + 1}/{total_pairs}, "
                      f"ETA: {eta_h:.1f}h)")
                show_separator(cl, "A", args.sep_a_duration, logger)

            # ── Real↔Fake divider (B) — between real and fake of same sample ──
            if vid_type == "fake" and prev_sample_id == sample_id:
                show_separator(cl, "B", args.sep_b_duration, logger)

            # ── Download and play ──
            local_path = download_video(bucket, sample_id, vid_type, tmp_dir)
            play_video_for_duration(cl, local_path, args.play_duration, entry, logger)

            # Cleanup previous video to save disk space
            cleanup_video(local_path)

            prev_sample_id = sample_id

            if vid_type == "fake":
                completed_pairs += 1

    except KeyboardInterrupt:
        elapsed_total = time.time() - started
        print(f"\n\n⚠️  Interrupted at index {idx} after {elapsed_total / 3600:.1f}h")
        print(f"   Resume with: --resume-from {idx}")
        logger.log("interrupted", index=idx, elapsed_h=round(elapsed_total / 3600, 2))
    except Exception as e:
        elapsed_total = time.time() - started
        print(f"\n\n❌ Error at index {idx}: {e}")
        print(f"   Resume with: --resume-from {idx}")
        logger.log("error", index=idx, error=str(e), elapsed_h=round(elapsed_total / 3600, 2))
        raise
    else:
        elapsed_total = time.time() - started
        print(f"\n\n✅ All {len(playlist)} videos played in {elapsed_total / 3600:.1f}h")
        logger.log("session_complete", elapsed_h=round(elapsed_total / 3600, 2),
                    pairs_completed=completed_pairs)
    finally:
        logger.close()
        cl.disconnect()
        # Clean up temp dir
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\nLog: {log_path}")
        print(f"Playlist: {playlist_path}")


if __name__ == "__main__":
    main()
