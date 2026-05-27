#!/usr/bin/env python3
"""4-identity side-by-side, same session (2026-04-24 Teams capture).

All 4 are REAL. Model says 3 real, 1 fake (Dor). Same pipeline, same session.
This contact sheet is the decisive visual for ruling out or confirming a
face-specific vs pipeline-specific failure mode.
"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/tmp/dor_session_20260424")
OUT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/session_20260424_4people_compare.png")

IDENTS = [
    ("Xiang_Xiang2_Feng", "REAL (model: real)", (36, 170, 85)),
    ("Xinhe_XH68_Wang", "REAL (model: real)", (36, 170, 85)),
    ("tester_tester", "REAL (model: real)", (36, 170, 85)),
    ("dor_shkedi", "REAL (model: FAKE p>0.9)", (255, 180, 40)),
]

N = 12
THUMB = 256
CAP_H = 34
HEADER_H = 40
PAD = 10
LABEL_W = 240
BG = (18, 18, 18)


def font(size):
    for p in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def pick(d: Path, n: int):
    files = sorted(d.glob("frame_*.png"))
    if len(files) <= n:
        return files
    idx = [round(i * (len(files) - 1) / (n - 1)) for i in range(n)]
    return [files[i] for i in idx]


def main():
    row_h = HEADER_H + THUMB + CAP_H + PAD * 2
    w = PAD + LABEL_W + N * (THUMB + PAD) + PAD
    h = PAD + len(IDENTS) * row_h + PAD
    sheet = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(sheet)
    fh = font(20)
    fs = font(13)
    fc = font(11)

    draw.text((PAD, 4), f"session_20260424_110139 — same capture session, all 4 REAL", fill=(235, 235, 235), font=fh)

    for row_idx, (ident, status, color) in enumerate(IDENTS):
        y0 = PAD + row_idx * row_h
        draw.text((PAD, y0 + 6), ident, fill=(235, 235, 235), font=fh)
        draw.text((PAD, y0 + 28), status, fill=color, font=fs)
        frames = pick(ROOT / ident, N)
        for i, fp in enumerate(frames):
            x = PAD + LABEL_W + i * (THUMB + PAD)
            y = y0 + HEADER_H + PAD
            try:
                img = Image.open(fp).convert("RGB")
                img.thumbnail((THUMB, THUMB), Image.LANCZOS)
                canvas = Image.new("RGB", (THUMB, THUMB), (0, 0, 0))
                canvas.paste(img, ((THUMB - img.width) // 2, (THUMB - img.height) // 2))
                img = canvas
            except Exception as e:
                img = Image.new("RGB", (THUMB, THUMB), (80, 0, 0))
                ImageDraw.Draw(img).text((6, 6), str(e)[:50], fill=(255, 200, 200), font=fc)
            sheet.paste(img, (x, y))
            for t in range(3):
                draw.rectangle([x - 1 - t, y - 1 - t, x + THUMB + t, y + THUMB + t], outline=color)
            cap_y = y + THUMB + 2
            draw.rectangle((x, cap_y, x + THUMB, cap_y + CAP_H), fill=(40, 40, 40))
            draw.text((x + 4, cap_y + 2), fp.name[:32], fill=(215, 215, 215), font=fc)

    sheet.save(OUT)
    print(f"Wrote {OUT}  ({w}x{h})")


if __name__ == "__main__":
    main()
