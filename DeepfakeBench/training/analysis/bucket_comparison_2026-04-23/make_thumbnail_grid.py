"""Render a single composite image with N thumbnails per source, labeled.
Makes the distribution gaps visible, not just numerical."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).parent
CACHE = HERE / "cache"
OUT = HERE / "thumbnails_grid.png"

TILE_W = 180  # thumbnail size
TILE_H = 180
PAD = 4
LABEL_H = 22
COLS = 6  # thumbnails per row

# Provenance-based ordering so related sources sit together.
ORDER = [
    "deeplive_non_enh_fake", "deeplive_non_enh_real",
    "deeplive_enh_fake", "deeplive_enh_real",
    "dl_bucket_visomaster_fake", "dl_bucket_visomaster_real",
    "tv2_deeplive_fake", "tv2_deeplive_real",
    "tv2_visomaster_fake", "tv2_visomaster_real",
    "proper_visomaster_clean_fake", "proper_real_clean__paired",
    "proper_visomaster_teams_fake", "proper_real_teams__paired",
    "proper_visomaster_enhanced_clean_fake",
    "proper_visomaster_enhanced_teams_fake",
    "visomaster_enhanced_v2_fake",
    "external_vcd_real",
    "external_youtube_avspeech_real",
    "wma_failure_fake",
]


def letterbox_to(img, w, h):
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 30, dtype=np.uint8)
    y = (h - nh) // 2
    x = (w - nw) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def render_source_strip(source: str) -> np.ndarray:
    d = CACHE / source
    files = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")) + list(d.glob("*.jpeg")))
    if not files:
        # placeholder
        strip = np.full((LABEL_H + TILE_H + PAD * 2, TILE_W * COLS + PAD * (COLS + 1), 3), 60, dtype=np.uint8)
        cv2.putText(strip, f"{source}: no samples", (8, LABEL_H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        return strip

    files = files[:COLS]
    width = TILE_W * COLS + PAD * (COLS + 1)
    height = LABEL_H + TILE_H + PAD * 2
    strip = np.full((height, width, 3), 30, dtype=np.uint8)

    # label
    lbl = f"{source}  (n={len(files)})"
    cv2.putText(strip, lbl, (8, LABEL_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (245, 245, 245), 1, cv2.LINE_AA)

    for i, f in enumerate(files):
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        tile = letterbox_to(img, TILE_W, TILE_H)
        # overlay native resolution in corner
        cv2.putText(tile, f"{w}x{h}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 255, 0), 1, cv2.LINE_AA)
        y0 = LABEL_H + PAD
        x0 = PAD + i * (TILE_W + PAD)
        strip[y0:y0 + TILE_H, x0:x0 + TILE_W] = tile
    return strip


def main():
    strips = []
    for src in ORDER:
        strip = render_source_strip(src)
        strips.append(strip)
    grid = np.vstack(strips)
    cv2.imwrite(str(OUT), grid)
    print(f"Wrote {OUT}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
