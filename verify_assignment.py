#!/usr/bin/env python3
"""
Visual verification of Teams dataset reconstruction.

Creates side-by-side image grids showing:
  LEFT column  = original YOLO-cropped frames from GCS
  RIGHT column = Teams-captured face crops

For each sample, you can eyeball whether they're from the same video
(same person, same scene, same lighting). This catches assignment errors
caused by clock misalignment or wrong timestamp matching.

Usage:
  python verify_assignment.py \
      --teams-dir teams_dataset \
      --output-dir verification_grids \
      --max-samples 20 \
      --frames-per-sample 4

Output:
  verification_grids/
    DL_0001_real.jpg     # side-by-side grid
    DL_0001_fake.jpg
    DL_0002_real.jpg
    ...
    index.html           # single-page viewer for quick scrolling
"""

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from google.cloud import storage as gcs
except ImportError:
    sys.exit("Missing google-cloud-storage. Install: pip install google-cloud-storage")


# ── Config ──────────────────────────────────────────────────────────────────

GCS_PROJECT = "train-cvit2"
FRAMES_BUCKET = "live-deepfake-methods-real-and-fake-frames-cropped"

THUMB_SIZE = 224          # thumbnail size for each cell
PADDING = 8              # pixels between images
LABEL_HEIGHT = 28        # height of text labels
BG_COLOR = (30, 30, 30)  # dark background
ORIG_BORDER = (0, 180, 0)    # green border for originals
TEAMS_BORDER = (0, 120, 255)  # blue border for Teams frames
BORDER_WIDTH = 3


# ── GCS helpers ─────────────────────────────────────────────────────────────

def download_original_frames(bucket, sample_id: str, vid_type: str,
                             tmp_dir: str, max_frames: int = 4) -> list[str]:
    """Download original cropped frames from GCS."""
    prefix = f"samples/{sample_id}/{vid_type}/"
    blobs = list(bucket.list_blobs(prefix=prefix, max_results=max_frames * 3))
    image_blobs = [b for b in blobs if b.name.endswith((".jpg", ".jpeg", ".png"))]

    if not image_blobs:
        return []

    # Take evenly spaced subset
    if len(image_blobs) > max_frames:
        step = len(image_blobs) / max_frames
        image_blobs = [image_blobs[int(i * step)] for i in range(max_frames)]

    local_paths = []
    for blob in image_blobs:
        fname = blob.name.replace("/", "_")
        local_path = os.path.join(tmp_dir, fname)
        blob.download_to_filename(local_path)
        local_paths.append(local_path)

    return local_paths


# ── Grid building ───────────────────────────────────────────────────────────

def make_thumbnail(img_path: str, size: int = THUMB_SIZE,
                   border_color: tuple = None) -> Image.Image:
    """Load, resize to square thumbnail, optionally add colored border."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)

    if border_color:
        draw = ImageDraw.Draw(img)
        bw = BORDER_WIDTH
        for i in range(bw):
            draw.rectangle([i, i, size - 1 - i, size - 1 - i],
                           outline=border_color)
    return img


def add_label(text: str, width: int, height: int = LABEL_HEIGHT,
              color: tuple = (255, 255, 255)) -> Image.Image:
    """Create a text label image."""
    label = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(label)

    # Try to find a decent font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (width - tw) // 2
    y = (height - th) // 2
    draw.text((x, y), text, fill=color, font=font)
    return label


def build_comparison_grid(orig_paths: list[str], teams_paths: list[str],
                          sample_id: str, vid_type: str,
                          playlist_index: int | None = None) -> Image.Image:
    """
    Build a side-by-side grid image:

    ┌─────────────────────────────────────────┐
    │  Sample: DL_0001 | Type: real | Idx: 0  │
    ├───────────────────┬─────────────────────┤
    │ Original (GCS)    │ Teams-captured       │
    ├───────────────────┼─────────────────────┤
    │ [thumb] [thumb]   │ [thumb] [thumb]      │
    │ [thumb] [thumb]   │ [thumb] [thumb]      │
    └───────────────────┴─────────────────────┘
    """
    n_orig = len(orig_paths)
    n_teams = len(teams_paths)
    n_rows = max(n_orig, n_teams)

    if n_rows == 0:
        return None

    # Thumbnails
    orig_thumbs = [make_thumbnail(p, border_color=ORIG_BORDER) for p in orig_paths]
    teams_thumbs = [make_thumbnail(p, border_color=TEAMS_BORDER) for p in teams_paths]

    # Dimensions
    cell = THUMB_SIZE + PADDING
    col_width = cell + PADDING
    divider_width = 4
    total_width = col_width + divider_width + col_width + PADDING
    header_height = LABEL_HEIGHT * 2 + PADDING
    total_height = header_height + n_rows * cell + PADDING

    canvas = Image.new("RGB", (total_width, total_height), BG_COLOR)

    # Header
    idx_str = f" | Playlist #{playlist_index}" if playlist_index is not None else ""
    header = add_label(f"{sample_id}  |  {vid_type}{idx_str}", total_width)
    canvas.paste(header, (0, 0))

    orig_label = add_label(f"Original (GCS) [{n_orig}]", col_width, color=(0, 220, 0))
    teams_label = add_label(f"Teams-captured [{n_teams}]", col_width, color=(80, 160, 255))
    canvas.paste(orig_label, (PADDING, LABEL_HEIGHT))
    canvas.paste(teams_label, (col_width + divider_width, LABEL_HEIGHT))

    # Draw divider line
    draw = ImageDraw.Draw(canvas)
    x_div = col_width + divider_width // 2
    draw.line([(x_div, header_height), (x_div, total_height)],
              fill=(100, 100, 100), width=divider_width)

    # Place thumbnails
    y_start = header_height

    for row in range(n_rows):
        y = y_start + row * cell
        if row < len(orig_thumbs):
            canvas.paste(orig_thumbs[row], (PADDING, y))
        if row < len(teams_thumbs):
            canvas.paste(teams_thumbs[row], (col_width + divider_width + PADDING, y))

    return canvas


# ── HTML index ──────────────────────────────────────────────────────────────

def write_html_index(image_files: list[dict], output_dir: str):
    """Write an index.html for easy browsing of all comparison grids."""
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>Teams Assignment Verification</title>",
        "<style>",
        "  body { background: #1a1a1a; color: #ddd; font-family: monospace; padding: 20px; }",
        "  h1 { color: #fff; }",
        "  .grid { display: flex; flex-wrap: wrap; gap: 16px; }",
        "  .card { background: #2a2a2a; border-radius: 8px; padding: 8px; }",
        "  .card img { max-width: 500px; border-radius: 4px; }",
        "  .card .label { text-align: center; padding: 4px; font-size: 13px; }",
        "  .match { color: #4caf50; }",
        "  .mismatch { color: #f44336; }",
        "  .legend { margin-bottom: 20px; padding: 12px; background: #222; border-radius: 8px; }",
        "  .legend span { margin-right: 20px; }",
        "  .green { color: #00dc00; }",
        "  .blue { color: #5090ff; }",
        "</style>",
        "</head><body>",
        "<h1>Teams Assignment Verification</h1>",
        "<div class='legend'>",
        "  <span class='green'>■ Green border = Original (GCS)</span>",
        "  <span class='blue'>■ Blue border = Teams-captured</span>",
        "  <br><br>",
        "  <b>Check:</b> Do the left and right columns show the same person/scene?",
        "  If yes, the timestamp assignment is correct.",
        "</div>",
        "<div class='grid'>",
    ]

    for item in image_files:
        fname = item["filename"]
        sid = item["sample_id"]
        vtype = item["type"]
        html_parts.append(f"""
        <div class='card'>
          <div class='label'><b>{sid}</b> / {vtype}</div>
          <img src='{fname}' alt='{sid} {vtype}'>
        </div>
        """)

    html_parts.extend(["</div>", "</body></html>"])

    html_path = Path(output_dir) / "index.html"
    with open(html_path, "w") as f:
        f.write("\n".join(html_parts))
    print(f"\nHTML viewer: {html_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visual verification of Teams dataset reconstruction")
    parser.add_argument("--teams-dir", required=True,
                        help="Path to reconstructed teams_dataset/ directory")
    parser.add_argument("--assignment", default=None,
                        help="Path to assignment.json (default: teams_dir/assignment.json)")
    parser.add_argument("--output-dir", default="verification_grids",
                        help="Output directory for grid images + HTML viewer")
    parser.add_argument("--max-samples", type=int, default=20,
                        help="Max samples to verify (default: 20)")
    parser.add_argument("--frames-per-sample", type=int, default=4,
                        help="Frames per sample to show (default: 4)")
    args = parser.parse_args()

    teams_path = Path(args.teams_dir)
    assignment_path = args.assignment or str(teams_path / "assignment.json")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Visual Assignment Verification")
    print("=" * 60)

    # Load assignment
    print(f"\nLoading assignment: {assignment_path}")
    with open(assignment_path) as f:
        assignment = json.load(f)
    print(f"  {len(assignment)} assigned frames")

    # Group by (sample_id, type)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for entry in assignment:
        key = (entry["sample_id"], entry["type"])
        groups[key].append(entry)

    # Unique samples
    sample_ids = sorted(set(e["sample_id"] for e in assignment))
    samples_to_verify = sample_ids[:args.max_samples]
    print(f"  Verifying {len(samples_to_verify)} of {len(sample_ids)} samples")

    # Connect to GCS
    print(f"\nConnecting to GCS bucket: {FRAMES_BUCKET}")
    client = gcs.Client(project=GCS_PROJECT)
    bucket = client.bucket(FRAMES_BUCKET)

    tmp_dir = tempfile.mkdtemp(prefix="verify_frames_")
    image_files = []

    for i, sample_id in enumerate(samples_to_verify):
        for vid_type in ("real", "fake"):
            key = (sample_id, vid_type)
            if key not in groups:
                continue

            entries = groups[key]

            # Get Teams frames
            teams_frame_paths = []
            for entry in entries[:args.frames_per_sample]:
                p = teams_path / entry["output_path"]
                if p.exists():
                    teams_frame_paths.append(str(p))

            if not teams_frame_paths:
                continue

            # Get playlist index for labeling
            playlist_idx = entries[0].get("playlist_index")

            # Download originals from GCS
            orig_paths = download_original_frames(
                bucket, sample_id, vid_type, tmp_dir,
                max_frames=args.frames_per_sample)

            if not orig_paths:
                print(f"  ⚠ No originals for {sample_id}/{vid_type}")
                # Still make a grid showing only Teams side
                # (helps catch cases where the sample_id might be wrong)

            # Build grid
            grid = build_comparison_grid(
                orig_paths, teams_frame_paths,
                sample_id, vid_type,
                playlist_index=playlist_idx)

            if grid:
                fname = f"{sample_id}_{vid_type}.jpg"
                grid_path = output_dir / fname
                grid.save(grid_path, quality=92)
                image_files.append({
                    "filename": fname,
                    "sample_id": sample_id,
                    "type": vid_type,
                    "n_orig": len(orig_paths),
                    "n_teams": len(teams_frame_paths),
                })

                print(f"  ✓ [{i+1}/{len(samples_to_verify)}] {sample_id}/{vid_type}  "
                      f"({len(orig_paths)} orig, {len(teams_frame_paths)} teams)")

    # Write HTML index
    if image_files:
        write_html_index(image_files, str(output_dir))
        print(f"\n{'=' * 60}")
        print(f"Generated {len(image_files)} verification grids")
        print(f"Open {output_dir / 'index.html'} in a browser to review")
        print(f"{'=' * 60}")
    else:
        print("\n⚠ No grids generated.")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
