#!/usr/bin/env python3
"""Contact sheet of enhanced-fake pockets the model also struggles with.

Pockets (3 rows, 12 thumbs each):
  1. visomaster_enhanced_macro_dev — 2.9% recall (fails badly). Pick 12 lowest-prob videos' lowest frame.
  2. deeplive_enhanced_dev         — 9.0% recall (also fails). Same selection.
  3. teams_fake_all_dev (caught)   — reference. 12 random-prob videos, median frame.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from arena.visualize_teams_target_domain_manifest import _read_bytes_from_path  # noqa: E402

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

LOCAL = Path("/tmp/r13_analysis/frames")
OUT = REPO_ROOT / "analysis" / "enhanced_fakes_contact_2026-04-24.png"
CACHE = REPO_ROOT / "analysis" / "_enh_fake_cache_2026-04-24"

VISO = LOCAL / "visomaster_enhanced_macro_dev_r13_rlp5_07_e3_seedb_frames_report.csv"
DEEPLIVE = LOCAL / "deeplive_enhanced_dev_r13_rlp5_07_e3_seedb_frames_report.csv"
TEAMS_FAKE = LOCAL / "teams_fake_all_dev_r13_rlp5_07_e3_seedb_frames_report.csv"

N = 12
THUMB = 192
CAP_H = 32
PAD = 8
ROW_LABEL_W = 220
HEADER_H = 30
BG = (18, 18, 18)


def font(size):
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


def load(path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out.append({
                "method": r["method"],
                "video_id": r["video_id"],
                "path": r["frame_path"],
                "prob": float(r["frame_prob"]),
            })
    return out


def group(rows):
    out: Dict[str, List] = {}
    for r in rows:
        out.setdefault(r["video_id"], []).append(r)
    return out


def pick_worst_first(grouped, n, mode="lowest"):
    avgs = [(v, sum(x["prob"] for x in fs) / len(fs)) for v, fs in grouped.items()]
    avgs.sort(key=lambda x: x[1] if mode == "lowest" else -x[1])
    chosen = [v for v, _ in avgs[:n]]
    return [min(grouped[v], key=lambda x: x["prob"]) for v in chosen]


def pick_random_median(grouped, n, seed=42):
    rng = random.Random(seed)
    vids = rng.sample(list(grouped.keys()), min(n, len(grouped)))
    out = []
    for v in vids:
        fs = sorted(grouped[v], key=lambda r: r["prob"])
        out.append(fs[len(fs) // 2])
    return out


def main():
    viso = load(VISO)
    deeplive = load(DEEPLIVE)
    teams_fake = load(TEAMS_FAKE)

    # viso: method=visomaster_enhanced_macro
    viso_g = group([r for r in viso if r["method"] == "visomaster_enhanced_macro"])
    # deeplive: method=deeplive_enhanced
    deep_g = group([r for r in deeplive if r["method"] == "deeplive_enhanced"])
    # teams_fake: all methods, pick any
    tf_g = group(teams_fake)

    pockets = [
        {
            "title": "visomaster_enhanced_macro_dev — MISSED (2.9% recall)",
            "subtitle": f"12 lowest-avg-prob videos, lowest-prob frame each",
            "picks": pick_worst_first(viso_g, N, "lowest"),
            "border": (255, 180, 40),
        },
        {
            "title": "deeplive_enhanced_dev — MISSED (9.0% recall)",
            "subtitle": f"12 lowest-avg-prob videos, lowest-prob frame each",
            "picks": pick_worst_first(deep_g, N, "lowest"),
            "border": (255, 180, 40),
        },
        {
            "title": "teams_fake_all_dev — reference (38% recall; caught these)",
            "subtitle": f"12 random videos, median-prob frame",
            "picks": pick_random_median(tf_g, N),
            "border": (120, 60, 60),
        },
    ]

    row_h = HEADER_H + THUMB + CAP_H + PAD * 2
    sheet_w = PAD + ROW_LABEL_W + N * (THUMB + PAD) + PAD
    sheet_h = PAD + len(pockets) * row_h + PAD
    sheet = Image.new("RGB", (sheet_w, sheet_h), BG)
    draw = ImageDraw.Draw(sheet)
    fh = font(14)
    fs = font(11)
    fc = font(10)

    for row_idx, p in enumerate(pockets):
        y0 = PAD + row_idx * row_h
        draw.text((PAD, y0 + 4), p["title"], fill=(235, 235, 235), font=fh)
        draw.text((PAD, y0 + 20), p["subtitle"], fill=(170, 170, 170), font=fs)
        for i, r in enumerate(p["picks"]):
            x = PAD + ROW_LABEL_W + i * (THUMB + PAD)
            y = y0 + HEADER_H + PAD
            try:
                local = cache(r["path"])
                img = Image.open(local).convert("RGB")
                img.thumbnail((THUMB, THUMB), Image.LANCZOS)
                canvas = Image.new("RGB", (THUMB, THUMB), (0, 0, 0))
                canvas.paste(img, ((THUMB - img.width) // 2, (THUMB - img.height) // 2))
                img = canvas
            except Exception as e:
                img = Image.new("RGB", (THUMB, THUMB), (80, 0, 0))
                ImageDraw.Draw(img).text((8, 8), str(e)[:70], fill=(255, 200, 200))
            sheet.paste(img, (x, y))
            for t in range(3):
                draw.rectangle([x - 1 - t, y - 1 - t, x + THUMB + t, y + THUMB + t], outline=p["border"])
            cap_y = y + THUMB + 2
            draw.rectangle((x, cap_y, x + THUMB, cap_y + CAP_H), fill=(40, 40, 40))
            vid = r["video_id"][:26]
            prob = r["prob"]
            pcolor = (214, 78, 78) if prob > 0.5 else (36, 170, 85)
            draw.text((x + 4, cap_y + 2), vid, fill=(235, 235, 235), font=fc)
            draw.text((x + 4, cap_y + 16), f"prob={prob:.3f}", fill=pcolor, font=fc)

    sheet.save(OUT)
    print(f"Wrote {OUT}  ({sheet_w}x{sheet_h})")


if __name__ == "__main__":
    main()
