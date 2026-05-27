#!/usr/bin/env python3
"""Render a single PNG contact sheet of lockbox failure frames for slot-07.

Pockets (rows), 12 frames each:
  1. cam_test_s33 FAKES (lockbox) — 12 lowest-prob-fake videos, each's lowest-prob frame
  2. cam_test_s32 FAKES (dev)     — 12 random reference frames (working session)
  3. dor_shkedi REALS (lockbox)   — 12 highest-prob-fake videos, each's highest-prob frame
  4. real_dor  REALS (lockbox)    — 12 random reference frames (clean Dor pool)

Output: analysis/lockbox_failure_contact_2026-04-24.png  (+ ...index.json)
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Reuse proven GCS helpers from the manifest visualizer.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from arena.visualize_teams_target_domain_manifest import (  # noqa: E402
    _read_bytes_from_path,
)

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

OUT_DIR = REPO_ROOT / "analysis"
OUT_PNG = OUT_DIR / "lockbox_failure_contact_2026-04-24.png"
OUT_JSON = OUT_DIR / "lockbox_failure_contact_2026-04-24.index.json"
FRAME_CACHE = REPO_ROOT / "analysis" / "_lockbox_contact_cache_2026-04-24"

LOCAL = Path("/tmp/r13_analysis/frames")
MANIFEST = REPO_ROOT / "arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json"

FAKE_LB_CSV = LOCAL / "teams_fake_all_lockbox_r13_rlp5_07_e3_seedb_frames_report.csv"
REAL_LB_CSV = LOCAL / "teams_real_all_lockbox_r13_rlp5_07_e3_seedb_frames_report.csv"
FAKE_DEV_CSV = LOCAL / "teams_fake_all_dev_r13_rlp5_07_e3_seedb_frames_report.csv"

N_PER_ROW = 12
THUMB = 192
CAPTION_H = 32
PAD = 8
ROW_LABEL_W = 220
BG = (18, 18, 18)
CAP_BG = (40, 40, 40)
TEXT = (235, 235, 235)
MUTED = (170, 170, 170)
FAKE_BORDER = (214, 78, 78)
REAL_BORDER = (36, 170, 85)
MISCLASS_ACCENT = (255, 180, 40)  # yellow border for misclassified
FAKE_REF_BORDER = (120, 60, 60)  # dimmer for reference row
REAL_REF_BORDER = (60, 100, 60)
SEED = 42


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                pass
    return ImageFont.load_default()


def load_video_identity_map() -> Dict[str, str]:
    """video_id -> identity_key."""
    data = json.loads(MANIFEST.read_text())
    return {v["video_id"]: v["identity_key"] for v in data["videos"]}


def load_frames(path: Path) -> List[Dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "method": row["method"],
                "label": int(row["label"]),
                "video_id": row["video_id"],
                "frame_path": row["frame_path"],
                "frame_prob": float(row["frame_prob"]),
            })
    return rows


def group_by_video(rows: List[Dict]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for r in rows:
        out.setdefault(r["video_id"], []).append(r)
    return out


def pick_videos_by_avg(grouped: Dict[str, List[Dict]], mode: str, n: int) -> List[str]:
    """Pick n videos by their average frame_prob. mode = 'lowest' or 'highest' or 'random'."""
    video_avgs = [
        (vid, sum(r["frame_prob"] for r in frames) / len(frames))
        for vid, frames in grouped.items()
    ]
    if mode == "lowest":
        video_avgs.sort(key=lambda x: x[1])
    elif mode == "highest":
        video_avgs.sort(key=lambda x: -x[1])
    else:
        rng = random.Random(SEED)
        rng.shuffle(video_avgs)
    return [vid for vid, _ in video_avgs[:n]]


def pick_representative_frame(frames: List[Dict], mode: str) -> Dict:
    """From a video's frames, pick one frame to show.

    For failing fakes: lowest prob (the most 'real-looking' frame).
    For failing reals: highest prob (the most 'fake-looking' frame).
    For reference: pick the median-prob frame (typical).
    """
    if mode == "lowest":
        return min(frames, key=lambda r: r["frame_prob"])
    if mode == "highest":
        return max(frames, key=lambda r: r["frame_prob"])
    sorted_frames = sorted(frames, key=lambda r: r["frame_prob"])
    return sorted_frames[len(sorted_frames) // 2]


def short_id(vid: str) -> str:
    # Keep it short enough for a caption under a 192px thumb.
    if len(vid) <= 26:
        return vid
    return vid[:22] + "..."


def cache_frame(gs_path: str) -> Path:
    FRAME_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(gs_path.encode()).hexdigest()
    local = FRAME_CACHE / f"{key}.jpg"
    if not local.exists():
        data = _read_bytes_from_path(gs_path)
        local.write_bytes(data)
    return local


def build_pockets() -> List[Dict]:
    """Return list of pocket dicts with 'title', 'color', 'picks' (list of 12 frame rows)."""
    vid2ident = load_video_identity_map()

    fake_lb = load_frames(FAKE_LB_CSV)
    real_lb = load_frames(REAL_LB_CSV)
    fake_dev = load_frames(FAKE_DEV_CSV)

    # Pocket 1: cam_test_s33 fakes (lockbox) — worst-recall
    s33 = [r for r in fake_lb if r["method"] == "teams_capture_cam_test_s33"]
    s33_grouped = group_by_video(s33)
    s33_videos = pick_videos_by_avg(s33_grouped, "lowest", N_PER_ROW)
    p1 = [pick_representative_frame(s33_grouped[v], "lowest") for v in s33_videos]

    # Pocket 2: cam_test_s32 fakes (dev) — working reference
    s32 = [r for r in fake_dev if r["method"] == "teams_capture_cam_test_s32"]
    s32_grouped = group_by_video(s32)
    s32_videos = pick_videos_by_avg(s32_grouped, "random", N_PER_ROW)
    p2 = [pick_representative_frame(s32_grouped[v], "median") for v in s32_videos]

    # Pocket 3: dor_shkedi reals (lockbox) — worst FPs
    dor = [r for r in real_lb if vid2ident.get(r["video_id"]) == "dor_shkedi"]
    dor_grouped = group_by_video(dor)
    dor_videos = pick_videos_by_avg(dor_grouped, "highest", N_PER_ROW)
    p3 = [pick_representative_frame(dor_grouped[v], "highest") for v in dor_videos]

    # Pocket 4: real_dor reals (lockbox) — clean reference
    rd = [r for r in real_lb if vid2ident.get(r["video_id"]) == "real_dor"]
    rd_grouped = group_by_video(rd)
    rd_videos = pick_videos_by_avg(rd_grouped, "random", N_PER_ROW)
    p4 = [pick_representative_frame(rd_grouped[v], "median") for v in rd_videos]

    return [
        {
            "title": "FAKE cam_test_s33 (LOCKBOX)  —  MISSED by model",
            "subtitle": f"Picked: 12 lowest avg_prob videos, each's lowest-prob frame",
            "border": MISCLASS_ACCENT,
            "label": "fake",
            "picks": p1,
        },
        {
            "title": "FAKE cam_test_s32 (DEV)  —  reference (caught fine)",
            "subtitle": "Picked: 12 random videos, median-prob frame",
            "border": FAKE_REF_BORDER,
            "label": "fake",
            "picks": p2,
        },
        {
            "title": "REAL dor_shkedi (LOCKBOX)  —  FALSELY FLAGGED as fake",
            "subtitle": "Picked: 12 highest avg_prob videos, each's highest-prob frame",
            "border": MISCLASS_ACCENT,
            "label": "real",
            "picks": p3,
        },
        {
            "title": "REAL real_dor (LOCKBOX)  —  reference (clean)",
            "subtitle": "Picked: 12 random videos, median-prob frame",
            "border": REAL_REF_BORDER,
            "label": "real",
            "picks": p4,
        },
    ]


def render(pockets: List[Dict]):
    title_h = 28
    subtitle_h = 18
    row_header_h = title_h + subtitle_h + 6
    row_h = row_header_h + THUMB + CAPTION_H + PAD * 2
    sheet_w = PAD + ROW_LABEL_W + N_PER_ROW * (THUMB + PAD) + PAD
    sheet_h = PAD + len(pockets) * row_h + PAD

    sheet = Image.new("RGB", (sheet_w, sheet_h), BG)
    draw = ImageDraw.Draw(sheet)
    font_row = load_font(15)
    font_sub = load_font(11)
    font_cap = load_font(10)

    index_entries = []

    for row_idx, pocket in enumerate(pockets):
        y0 = PAD + row_idx * row_h
        # Row header (left margin text)
        draw.text((PAD, y0 + 4), pocket["title"], fill=TEXT, font=font_row)
        draw.text((PAD, y0 + 4 + title_h - 10), pocket["subtitle"], fill=MUTED, font=font_sub)

        # Thumbnails
        x0 = PAD + ROW_LABEL_W
        for i, frame_row in enumerate(pocket["picks"]):
            try:
                local = cache_frame(frame_row["frame_path"])
                img = Image.open(local).convert("RGB")
                img.thumbnail((THUMB, THUMB), Image.LANCZOS)
                # Center-crop/pad to THUMB x THUMB
                canvas = Image.new("RGB", (THUMB, THUMB), (0, 0, 0))
                canvas.paste(img, ((THUMB - img.width) // 2, (THUMB - img.height) // 2))
                img = canvas
            except Exception as e:
                img = Image.new("RGB", (THUMB, THUMB), (60, 0, 0))
                d2 = ImageDraw.Draw(img)
                d2.text((8, 80), f"ERR:\n{e}"[:100], fill=(255, 200, 200), font=font_cap)

            thumb_x = x0 + i * (THUMB + PAD)
            thumb_y = y0 + row_header_h + PAD
            sheet.paste(img, (thumb_x, thumb_y))

            # Border
            border = pocket["border"]
            for t in range(3):
                draw.rectangle(
                    [thumb_x - 1 - t, thumb_y - 1 - t, thumb_x + THUMB + t, thumb_y + THUMB + t],
                    outline=border,
                )

            # Caption: video_id + prob
            cap_y = thumb_y + THUMB + 2
            cap_box = (thumb_x, cap_y, thumb_x + THUMB, cap_y + CAPTION_H)
            draw.rectangle(cap_box, fill=CAP_BG)
            vid_short = short_id(frame_row["video_id"])
            prob = frame_row["frame_prob"]
            prob_color = FAKE_BORDER if prob > 0.5 else REAL_BORDER
            draw.text((thumb_x + 4, cap_y + 2), vid_short, fill=TEXT, font=font_cap)
            draw.text(
                (thumb_x + 4, cap_y + 16), f"prob_fake={prob:.3f}", fill=prob_color, font=font_cap
            )

            index_entries.append({
                "row": row_idx,
                "col": i,
                "pocket_title": pocket["title"],
                "video_id": frame_row["video_id"],
                "frame_path": frame_row["frame_path"],
                "frame_prob": prob,
                "method": frame_row["method"],
                "label": pocket["label"],
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT_PNG)
    OUT_JSON.write_text(json.dumps(index_entries, indent=2))
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_JSON}")
    print(f"Sheet: {sheet_w}x{sheet_h}px, {len(index_entries)} frames")


def main():
    pockets = build_pockets()
    for p in pockets:
        print(f"  {p['title']}: {len(p['picks'])} frames")
    render(pockets)


if __name__ == "__main__":
    main()
