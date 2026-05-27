#!/usr/bin/env python3
"""High-resolution side-by-side of 6 dor_shkedi (failed) vs 6 real_dor (clean) frames.

Purpose: visual verification of whether the two "Dor" pools are the same person,
and what the quality/lighting/crop differences are between them.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from arena.visualize_teams_target_domain_manifest import _read_bytes_from_path  # noqa: E402

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

LOCAL = Path("/tmp/r13_analysis/frames")
MANIFEST = REPO_ROOT / "arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json"
CACHE = REPO_ROOT / "analysis" / "_dor_closeup_cache_2026-04-24"
OUT = REPO_ROOT / "analysis" / "dor_closeup_2026-04-24.png"

THUMB = 384
CAP_H = 44
PAD = 10
HEADER_H = 36


def load_font(size):
    for f in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(f):
            try:
                return ImageFont.truetype(f, size)
            except Exception:
                pass
    return ImageFont.load_default()


def cache(gs_path):
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(gs_path.encode()).hexdigest()
    local = CACHE / f"{key}.jpg"
    if not local.exists():
        local.write_bytes(_read_bytes_from_path(gs_path))
    return local


def main():
    m = json.loads(MANIFEST.read_text())
    v2ident = {v["video_id"]: v["identity_key"] for v in m["videos"]}
    vdata = {v["video_id"]: v for v in m["videos"]}

    real_csv = LOCAL / "teams_real_all_lockbox_r13_rlp5_07_e3_seedb_frames_report.csv"
    rows = []
    with open(real_csv) as f:
        for r in csv.DictReader(f):
            rows.append({
                "vid": r["video_id"],
                "path": r["frame_path"],
                "prob": float(r["frame_prob"]),
            })

    ds = [r for r in rows if v2ident.get(r["vid"]) == "dor_shkedi"]
    rd = [r for r in rows if v2ident.get(r["vid"]) == "real_dor"]

    # Pick 6 highest-prob dor_shkedi (model says fake) — worst offenders
    ds_sorted = sorted(ds, key=lambda x: -x["prob"])[:6]
    # Pick 6 random real_dor (clean)
    rng = random.Random(42)
    rd_sample = rng.sample(rd, min(6, len(rd)))

    # Layout: 2 rows of 6 thumbs
    ncol = 6
    row_h = HEADER_H + THUMB + CAP_H + PAD * 2
    sheet_w = PAD + ncol * (THUMB + PAD) + PAD
    sheet_h = PAD + 2 * row_h + PAD

    sheet = Image.new("RGB", (sheet_w, sheet_h), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    font_h = load_font(18)
    font_c = load_font(13)

    def draw_row(y0, title, color, picks):
        draw.text((PAD, y0 + 4), title, fill=(235, 235, 235), font=font_h)
        for i, rowd in enumerate(picks):
            x = PAD + i * (THUMB + PAD)
            y = y0 + HEADER_H + PAD
            try:
                local = cache(rowd["path"])
                img = Image.open(local).convert("RGB")
                img.thumbnail((THUMB, THUMB), Image.LANCZOS)
                canvas = Image.new("RGB", (THUMB, THUMB), (0, 0, 0))
                canvas.paste(img, ((THUMB - img.width) // 2, (THUMB - img.height) // 2))
                img = canvas
            except Exception as e:
                img = Image.new("RGB", (THUMB, THUMB), (80, 0, 0))
                ImageDraw.Draw(img).text((8, 8), str(e)[:80], fill=(255, 200, 200))
            sheet.paste(img, (x, y))
            for t in range(3):
                draw.rectangle([x - 1 - t, y - 1 - t, x + THUMB + t, y + THUMB + t], outline=color)
            # Caption
            cap_box = (x, y + THUMB + 2, x + THUMB, y + THUMB + 2 + CAP_H)
            draw.rectangle(cap_box, fill=(40, 40, 40))
            ident = v2ident.get(rowd["vid"], "?")
            vid_data = vdata.get(rowd["vid"], {})
            seq = vid_data.get("sequence_id", "?")
            qm = vid_data.get("quality_metrics") or {}
            sharp = qm.get("sharpness_tenengrad", 0.0)
            bright = qm.get("mean_brightness", 0.0)
            draw.text(
                (x + 4, y + THUMB + 4),
                f"ident={ident}  {seq}  prob={rowd['prob']:.3f}",
                fill=(235, 235, 235),
                font=font_c,
            )
            draw.text(
                (x + 4, y + THUMB + 22),
                f"sharp={sharp:.2f}  bright={bright:.1f}",
                fill=(170, 170, 170),
                font=font_c,
            )

    draw_row(
        PAD,
        "dor_shkedi  (LOCKBOX real, identity #732954601) — 6 WORST-flagged (model thinks FAKE)",
        (255, 180, 40),
        ds_sorted,
    )
    draw_row(
        PAD + row_h,
        "real_dor   (LOCKBOX real, identity #249494207) — 6 random (model correctly says REAL)",
        (36, 170, 85),
        rd_sample,
    )

    sheet.save(OUT)
    print(f"Wrote {OUT}  ({sheet_w}x{sheet_h})")


if __name__ == "__main__":
    main()
