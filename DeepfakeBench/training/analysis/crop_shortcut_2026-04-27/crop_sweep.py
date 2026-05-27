"""Crop-tightness sensitivity sweep.

For a given input image, generate variants at multiple "tightness" levels and
run `check-frame` on each. Reports prob_fake at each level.

Tightness model:
  tightness > 1.0 → tighter than input. Take a center crop at side length
                    (1/tightness)*W, then resize back to input size. Face
                    occupies a larger fraction of the frame.
  tightness = 1.0 → input as-is.
  tightness < 1.0 → looser than input. Embed input at fraction `tightness`
                    centered in a canvas of input size, padded with replicated
                    edge pixels. Face occupies a smaller fraction.

Usage:
  python -m analysis.crop_shortcut_2026-04-27.crop_sweep <image> [<image2> ...]
  python -m analysis.crop_shortcut_2026-04-27.crop_sweep --tag <label> <image>

Output:
  Writes `sweep_outputs/<basename>_t<tightness>.png` for each variant.
  Prints a CSV-style table to stdout and appends a JSONL line to
  `crop_sweep_results.jsonl`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT = REPO / "analysis/crop_shortcut_2026-04-27/sweep_outputs"
RESULTS = REPO / "analysis/crop_shortcut_2026-04-27/crop_sweep_results.jsonl"
OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_TIGHTNESS = [0.6, 0.75, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75]


def make_variant(img: Image.Image, tightness: float) -> Image.Image:
    if abs(tightness - 1.0) < 1e-6:
        return img.copy()
    W, H = img.size
    if tightness > 1.0:
        # Tighter: center-crop (1/t) side, resize back.
        new_w = max(1, int(round(W / tightness)))
        new_h = max(1, int(round(H / tightness)))
        x0 = (W - new_w) // 2
        y0 = (H - new_h) // 2
        sub = img.crop((x0, y0, x0 + new_w, y0 + new_h))
        return sub.resize((W, H), Image.LANCZOS)
    # Looser: shrink to fraction `tightness` of the canvas, place centered with edge-replication padding.
    inner_w = max(1, int(round(W * tightness)))
    inner_h = max(1, int(round(H * tightness)))
    inner = img.resize((inner_w, inner_h), Image.LANCZOS)
    canvas = Image.new(img.mode, (W, H), (0, 0, 0) if img.mode in ("RGB", "RGBA") else 0)
    pad_x = (W - inner_w) // 2
    pad_y = (H - inner_h) // 2
    # Replicate edges of `inner` outward to fill the canvas.
    # Top band
    if pad_y > 0:
        top = inner.crop((0, 0, inner_w, 1)).resize((inner_w, pad_y))
        canvas.paste(top, (pad_x, 0))
    # Bottom band
    if pad_y > 0 and (H - pad_y - inner_h) > 0:
        bot = inner.crop((0, inner_h - 1, inner_w, inner_h)).resize((inner_w, H - pad_y - inner_h))
        canvas.paste(bot, (pad_x, pad_y + inner_h))
    # Left band
    if pad_x > 0:
        lf = inner.crop((0, 0, 1, inner_h)).resize((pad_x, inner_h))
        canvas.paste(lf, (0, pad_y))
    # Right band
    if pad_x > 0 and (W - pad_x - inner_w) > 0:
        rt = inner.crop((inner_w - 1, 0, inner_w, inner_h)).resize((W - pad_x - inner_w, inner_h))
        canvas.paste(rt, (pad_x + inner_w, pad_y))
    # Corners (replicate corner pixels)
    if pad_x > 0 and pad_y > 0:
        tl = inner.crop((0, 0, 1, 1)).resize((pad_x, pad_y))
        canvas.paste(tl, (0, 0))
    if pad_x > 0 and pad_y > 0 and (W - pad_x - inner_w) > 0:
        tr = inner.crop((inner_w - 1, 0, inner_w, 1)).resize((W - pad_x - inner_w, pad_y))
        canvas.paste(tr, (pad_x + inner_w, 0))
    if pad_x > 0 and pad_y > 0 and (H - pad_y - inner_h) > 0:
        bl = inner.crop((0, inner_h - 1, 1, inner_h)).resize((pad_x, H - pad_y - inner_h))
        canvas.paste(bl, (0, pad_y + inner_h))
    if (
        pad_x > 0 and pad_y > 0
        and (W - pad_x - inner_w) > 0 and (H - pad_y - inner_h) > 0
    ):
        br = inner.crop((inner_w - 1, inner_h - 1, inner_w, inner_h)).resize(
            (W - pad_x - inner_w, H - pad_y - inner_h)
        )
        canvas.paste(br, (pad_x + inner_w, pad_y + inner_h))
    canvas.paste(inner, (pad_x, pad_y))
    return canvas


_PROB_RE = re.compile(r'"confidence"\s*:\s*([0-9.]+)')
_LABEL_RE = re.compile(r'"pred_label"\s*:\s*"([^"]+)"')


def run_check_frame(path: Path) -> tuple[float | None, str | None, str]:
    try:
        out = subprocess.run(
            ["check-frame", str(path)],
            check=False, capture_output=True, text=True, timeout=60,
        )
    except Exception as e:
        return None, None, f"ERR_RUN: {e}"
    text = (out.stdout or "") + (out.stderr or "")
    label_m = _LABEL_RE.search(text)
    prob_m = _PROB_RE.search(text)
    if not (label_m and prob_m):
        return None, None, f"ERR_PARSE: {text[:200]}"
    label = label_m.group(1)
    # check-frame's "confidence" field is prob_fake directly (not confidence-in-label).
    # Verified: pred=REAL, conf=0.30 means prob_fake=0.30 — pred is just `conf >= 0.5`.
    prob_fake = float(prob_m.group(1))
    return prob_fake, label, text.strip()


def sweep(img_path: Path, tag: str, tightnesses: list[float]) -> list[dict]:
    img = Image.open(img_path).convert("RGB")
    base = img_path.stem
    rows = []
    for t in tightnesses:
        v = make_variant(img, t)
        out_path = OUT / f"{base}__t{t:.2f}.png"
        v.save(out_path)
        prob, pred, _ = run_check_frame(out_path)
        rows.append({
            "source": str(img_path),
            "tag": tag,
            "tightness": t,
            "variant_path": str(out_path),
            "prob_fake": prob,
            "pred_label": pred,
        })
        print(f"  t={t:.2f}  prob_fake={prob}  pred={pred}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="Input image paths")
    ap.add_argument("--tag", default="", help="Group tag for this run (e.g., 'real_dor' or 'deeplive_dor')")
    ap.add_argument(
        "--tightness", default=",".join(str(t) for t in DEFAULT_TIGHTNESS),
        help="Comma-separated tightness factors",
    )
    args = ap.parse_args()

    tightnesses = [float(x) for x in args.tightness.split(",")]
    all_rows = []
    for img_p in args.images:
        p = Path(img_p)
        if not p.exists():
            print(f"[skip] not found: {p}", file=sys.stderr)
            continue
        print(f"\n=== {args.tag or 'untagged'} : {p.name} ===")
        rows = sweep(p, args.tag, tightnesses)
        all_rows.extend(rows)

    with open(RESULTS, "a") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n[crop-sweep] appended {len(all_rows)} rows to {RESULTS}")


if __name__ == "__main__":
    main()
