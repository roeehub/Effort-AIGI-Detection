#!/usr/bin/env python3
"""
Reconstruct Teams-augmented dataset from captured face crops.

Takes a capture session directory (from the Windows capture software) and
the playback_log.jsonl from OBS, and assigns each captured face to its
source video (sample_id + real/fake) using timestamp alignment.

Outputs:
  - Assignment report (how many faces per video segment, separators found, etc.)
  - teams_dataset/ directory mirroring the original paired structure
  - assignment.json mapping every face → (sample_id, type, strategy)

Usage:
  # Point to the capture session dir (copy it over or mount it):
  python reconstruct_teams_dataset.py \\
      --session-dir /path/to/session_20260225_004520 \\
      --playback-log playback_log.jsonl \\
      --output-dir teams_dataset

  # Dry run — just report statistics, don't copy files:
  python reconstruct_teams_dataset.py \\
      --session-dir /path/to/session_20260225_004520 \\
      --playback-log playback_log.jsonl \\
      --dry-run
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class TimeInterval:
    """A time interval representing a playback segment."""
    start_ts: float       # Unix timestamp (seconds)
    end_ts: float         # Unix timestamp (seconds)
    event_type: str       # "video" or "separator"
    # For video segments:
    playlist_index: int | None = None
    sample_id: str | None = None
    vid_type: str | None = None   # "real" or "fake"
    strategy: str | None = None
    # For separator segments:
    marker: str | None = None     # "A", "B", or "C"


@dataclass
class FaceCrop:
    """A captured face crop with its metadata."""
    frame_name: str          # e.g. "frame_000079_seq963"
    image_path: str          # relative path in session dir
    timestamp_s: float       # Unix timestamp in seconds
    confidence: float
    frame_size: tuple[int, int]
    file_size_bytes: int
    score: float | None = None
    json_path: str = ""


@dataclass
class VideoSegment:
    """A video segment with its assigned faces."""
    interval: TimeInterval
    faces: list[FaceCrop] = field(default_factory=list)


# ── Parse playback log ─────────────────────────────────────────────────────

def parse_playback_log(log_path: str) -> list[TimeInterval]:
    """
    Parse playback_log.jsonl and build time intervals for every segment
    (both video and separator).

    Returns intervals sorted by start_ts.
    """
    # Load all events from the LAST full session
    events = []
    with open(log_path) as f:
        for line in f:
            e = json.loads(line)
            # Reset on each full session start (not test sessions)
            if e.get("event") == "session_start" and not e.get("test_mode", False):
                events = []
            events.append(e)

    if not events:
        sys.exit("No full session found in playback log.")

    # Build intervals
    intervals: list[TimeInterval] = []

    # Helper to parse ISO timestamp to Unix seconds
    def ts_to_unix(iso_str: str) -> float:
        dt = datetime.fromisoformat(iso_str)
        return dt.timestamp()

    i = 0
    while i < len(events):
        e = events[i]

        if e.get("event") == "separator":
            start = ts_to_unix(e["ts"])
            duration = e["duration_s"]
            intervals.append(TimeInterval(
                start_ts=start,
                end_ts=start + duration,
                event_type="separator",
                marker=e["marker"],
            ))

        elif e.get("event") == "video_start":
            start = ts_to_unix(e["ts"])
            # Find matching video_end
            end_ts = start + 15.0  # fallback
            for j in range(i + 1, min(i + 10, len(events))):
                if events[j].get("event") == "video_end" and events[j].get("index") == e["index"]:
                    end_ts = ts_to_unix(events[j]["ts"])
                    break
            intervals.append(TimeInterval(
                start_ts=start,
                end_ts=end_ts,
                event_type="video",
                playlist_index=e["index"],
                sample_id=e["sample_id"],
                vid_type=e["type"],
                strategy=e["strategy"],
            ))

        i += 1

    intervals.sort(key=lambda x: x.start_ts)
    return intervals


# ── Parse face crops ────────────────────────────────────────────────────────

def load_face_crops(session_dir: str) -> list[FaceCrop]:
    """
    Load all face crop JSON sidecars from the session directory.
    Looks in participants/*/faces/*.json
    """
    faces = []
    session_path = Path(session_dir)

    participants_dir = session_path / "participants"
    if not participants_dir.exists():
        sys.exit(f"No 'participants' directory in {session_dir}")

    for participant_dir in participants_dir.iterdir():
        if not participant_dir.is_dir():
            continue
        faces_dir = participant_dir / "faces"
        if not faces_dir.exists():
            continue

        for json_file in sorted(faces_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                frame_name = json_file.stem
                faces.append(FaceCrop(
                    frame_name=frame_name,
                    image_path=data.get("image_path", ""),
                    timestamp_s=data["timestamp_ms"] / 1000.0,
                    confidence=data.get("confidence", 0.0),
                    frame_size=tuple(data.get("frame_size", [0, 0])),
                    file_size_bytes=data.get("file_size_bytes", 0),
                    score=data.get("score"),
                    json_path=str(json_file),
                ))
            except (json.JSONDecodeError, KeyError) as exc:
                print(f"  ⚠ Skipping {json_file.name}: {exc}")

    faces.sort(key=lambda f: f.timestamp_s)
    return faces


# ── Assign faces to intervals ──────────────────────────────────────────────

TRANSITION_MARGIN_S = 1.0  # seconds to exclude around boundaries

def assign_faces(faces: list[FaceCrop], intervals: list[TimeInterval],
                 margin_s: float = TRANSITION_MARGIN_S) -> dict:
    """
    Assign each face to a time interval.

    Returns:
        {
            "video_segments": {playlist_index: VideoSegment, ...},
            "separator_faces": [FaceCrop, ...],
            "transition_faces": [FaceCrop, ...],  # near boundaries
            "unassigned_faces": [FaceCrop, ...],   # outside all intervals
        }
    """
    video_segments: dict[int, VideoSegment] = {}
    separator_faces: list[FaceCrop] = []
    transition_faces: list[FaceCrop] = []
    unassigned_faces: list[FaceCrop] = []

    # Pre-build video segments
    for iv in intervals:
        if iv.event_type == "video" and iv.playlist_index is not None:
            video_segments[iv.playlist_index] = VideoSegment(interval=iv)

    # For each face, find its interval
    for face in faces:
        t = face.timestamp_s
        matched = False

        for iv in intervals:
            if iv.start_ts <= t <= iv.end_ts:
                # Check if it's within margin of boundaries (transition frame)
                dist_to_start = t - iv.start_ts
                dist_to_end = iv.end_ts - t
                is_transition = (dist_to_start < margin_s) or (dist_to_end < margin_s)

                if is_transition:
                    transition_faces.append(face)
                elif iv.event_type == "separator":
                    separator_faces.append(face)
                elif iv.event_type == "video" and iv.playlist_index is not None:
                    video_segments[iv.playlist_index].faces.append(face)
                else:
                    unassigned_faces.append(face)

                matched = True
                break

        if not matched:
            unassigned_faces.append(face)

    return {
        "video_segments": video_segments,
        "separator_faces": separator_faces,
        "transition_faces": transition_faces,
        "unassigned_faces": unassigned_faces,
    }


# ── Output ──────────────────────────────────────────────────────────────────

def print_report(result: dict, intervals: list[TimeInterval], playlist: list[dict]):
    """Print a detailed reconstruction report."""
    segments = result["video_segments"]
    sep_faces = result["separator_faces"]
    trans_faces = result["transition_faces"]
    unassigned = result["unassigned_faces"]

    total_faces = sum(len(s.faces) for s in segments.values()) + len(sep_faces) + len(trans_faces) + len(unassigned)

    print("\n" + "=" * 60)
    print("RECONSTRUCTION REPORT")
    print("=" * 60)

    # Overall stats
    print(f"\nTotal faces processed:     {total_faces}")
    print(f"  Assigned to videos:      {sum(len(s.faces) for s in segments.values())}")
    print(f"  Separator faces:         {len(sep_faces)}")
    print(f"  Transition (boundary):   {len(trans_faces)}")
    print(f"  Unassigned:              {len(unassigned)}")

    # Videos with faces
    videos_with_faces = {idx: seg for idx, seg in segments.items() if seg.faces}
    print(f"\nVideo segments with faces: {len(videos_with_faces)} / {len(segments)}")

    # Per-sample summary (pair real+fake together)
    sample_pairs: dict[str, dict] = defaultdict(lambda: {"real": 0, "fake": 0, "strategy": ""})
    for idx, seg in videos_with_faces.items():
        iv = seg.interval
        sample_pairs[iv.sample_id][iv.vid_type] = len(seg.faces)
        sample_pairs[iv.sample_id]["strategy"] = iv.strategy

    print(f"Samples with at least one face: {len(sample_pairs)}")

    # Distribution of faces per video
    face_counts = [len(seg.faces) for seg in segments.values() if seg.faces]
    if face_counts:
        print(f"\nFaces per video (with data):")
        print(f"  Min:    {min(face_counts)}")
        print(f"  Max:    {max(face_counts)}")
        print(f"  Mean:   {sum(face_counts)/len(face_counts):.1f}")
        print(f"  Median: {sorted(face_counts)[len(face_counts)//2]}")

    # Separator breakdown
    sep_by_marker: dict[str, int] = defaultdict(int)
    for iv in intervals:
        if iv.event_type == "separator":
            sep_by_marker[iv.marker] = sep_by_marker.get(iv.marker, 0) + 1
    print(f"\nSeparator intervals in playback log:")
    for marker in ("A", "B", "C"):
        print(f"  {marker}: {sep_by_marker.get(marker, 0)} intervals")
    print(f"Separator faces captured: {len(sep_faces)}")

    # Sample-level detail table (first 20)
    print(f"\n{'─' * 60}")
    print(f"{'Sample':<30} {'Real':>6} {'Fake':>6} {'Strategy'}")
    print(f"{'─' * 60}")
    for sid in sorted(sample_pairs.keys())[:30]:
        p = sample_pairs[sid]
        print(f"{sid:<30} {p['real']:>6} {p['fake']:>6} {p['strategy']}")
    if len(sample_pairs) > 30:
        print(f"  ... and {len(sample_pairs) - 30} more samples")
    print(f"{'─' * 60}")

    # Samples with >= 16 faces for both real and fake
    complete = [sid for sid, p in sample_pairs.items()
                if p["real"] >= 16 and p["fake"] >= 16]
    partial = [sid for sid, p in sample_pairs.items()
               if (p["real"] > 0 or p["fake"] > 0) and (p["real"] < 16 or p["fake"] < 16)]
    print(f"\nComplete pairs (≥16 real + ≥16 fake): {len(complete)}")
    print(f"Partial pairs (some data, <16 each):  {len(partial)}")

    return sample_pairs


def write_dataset(result: dict, session_dir: str, output_dir: str,
                  max_frames: int = 16):
    """
    Copy face images into a paired directory structure:
      output_dir/
        {sample_id}/
          real/
            frame_000.jpg
            frame_001.jpg
            ...
          fake/
            frame_000.jpg
            ...
    Also writes assignment.json with full metadata.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    session_path = Path(session_dir)

    segments = result["video_segments"]
    assignment = []

    for idx, seg in sorted(segments.items()):
        if not seg.faces:
            continue

        iv = seg.interval
        sample_dir = output_path / iv.sample_id / iv.vid_type
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Take up to max_frames, evenly spaced
        faces = seg.faces
        if len(faces) > max_frames:
            step = len(faces) / max_frames
            faces = [faces[int(i * step)] for i in range(max_frames)]

        for i, face in enumerate(faces):
            # Source: resolve the JPG path from session dir
            # image_path is like "participants/Roy D/faces/frame_000079_seq963.jpg"
            src = session_path / face.image_path
            dst = sample_dir / f"frame_{i:03d}.jpg"

            if src.exists():
                shutil.copy2(src, dst)
            else:
                print(f"  ⚠ Missing: {src}")

            assignment.append({
                "output_path": str(dst.relative_to(output_path)),
                "source_frame": face.frame_name,
                "source_image_path": face.image_path,
                "timestamp_s": face.timestamp_s,
                "confidence": face.confidence,
                "frame_size": list(face.frame_size),
                "file_size_bytes": face.file_size_bytes,
                "score": face.score,
                "playlist_index": iv.playlist_index,
                "sample_id": iv.sample_id,
                "type": iv.vid_type,
                "strategy": iv.strategy,
            })

    # Write assignment manifest
    manifest_path = output_path / "assignment.json"
    with open(manifest_path, "w") as f:
        json.dump(assignment, f, indent=2)
    print(f"\nDataset written to: {output_dir}")
    print(f"  Assignment manifest: {manifest_path}")
    print(f"  Total frames copied: {len(assignment)}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct Teams-augmented dataset from captured face crops")
    parser.add_argument("--session-dir", required=True,
                        help="Path to the capture session directory (e.g. session_20260225_004520/)")
    parser.add_argument("--playback-log", default="playback_log.jsonl",
                        help="Path to OBS playback_log.jsonl")
    parser.add_argument("--playlist", default="playlist.json",
                        help="Path to playlist.json")
    parser.add_argument("--output-dir", default="teams_dataset",
                        help="Output directory for reconstructed dataset")
    parser.add_argument("--max-frames", type=int, default=16,
                        help="Maximum frames to keep per video segment (default: 16)")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Seconds to exclude near segment boundaries (default: 1.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print report, don't copy files")
    parser.add_argument("--clock-offset", type=float, default=0.0,
                        help="Seconds to add to capture timestamps to align with playback "
                             "(positive = capture clock was ahead)")
    args = parser.parse_args()

    print("=" * 60)
    print("Teams Dataset Reconstruction")
    print("=" * 60)

    # ── 1. Parse playback log ──
    print(f"\nLoading playback log: {args.playback_log}")
    intervals = parse_playback_log(args.playback_log)
    n_video = sum(1 for iv in intervals if iv.event_type == "video")
    n_sep = sum(1 for iv in intervals if iv.event_type == "separator")
    print(f"  {n_video} video intervals, {n_sep} separator intervals")

    if intervals:
        span_h = (intervals[-1].end_ts - intervals[0].start_ts) / 3600
        print(f"  Time span: {span_h:.1f} hours")

    # ── 2. Load playlist ──
    print(f"\nLoading playlist: {args.playlist}")
    with open(args.playlist) as f:
        playlist = json.load(f)
    print(f"  {len(playlist)} entries")

    # ── 3. Load face crops ──
    print(f"\nLoading face crops from: {args.session_dir}")
    faces = load_face_crops(args.session_dir)
    print(f"  {len(faces)} face crops loaded")

    if not faces:
        sys.exit("No face crops found.")

    # Apply clock offset
    if args.clock_offset != 0:
        print(f"  Applying clock offset: {args.clock_offset:+.1f}s")
        for face in faces:
            face.timestamp_s += args.clock_offset

    # Time range of faces
    face_span = faces[-1].timestamp_s - faces[0].timestamp_s
    print(f"  Time span: {face_span:.0f}s ({face_span/60:.1f} min)")
    print(f"  First face: {datetime.fromtimestamp(faces[0].timestamp_s, tz=timezone.utc).isoformat()}")
    print(f"  Last face:  {datetime.fromtimestamp(faces[-1].timestamp_s, tz=timezone.utc).isoformat()}")

    # ── 4. Assign faces to intervals ──
    print(f"\nAssigning faces to intervals (margin={args.margin}s) ...")
    result = assign_faces(faces, intervals, margin_s=args.margin)

    # ── 5. Report ──
    sample_pairs = print_report(result, intervals, playlist)

    # ── 6. Write dataset ──
    if not args.dry_run:
        write_dataset(result, args.session_dir, args.output_dir,
                      max_frames=args.max_frames)
    else:
        print("\n🏁 Dry run — no files copied.")


if __name__ == "__main__":
    main()
