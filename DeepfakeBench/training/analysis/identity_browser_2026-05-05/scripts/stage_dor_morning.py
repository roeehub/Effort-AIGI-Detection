"""Stage dor_morning frames into the identity browser.

Frames are local-only (from user's Downloads), so we synthesize a stable URI
prefix to use as the score-lookup key.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
SRC_ROOT = ROOT / "analysis/dor_morning_2026-05-05"
SRC_FRAMES = SRC_ROOT / "raw/all/dor_morning"
SRC_SCORES = SRC_ROOT / "scores"
DST = ROOT / "analysis/identity_browser_2026-05-05"

URI_PREFIX = "gs://local/faces_dor/morning/"
SUITE = "dor_morning"
BUCKET = "local"
LABEL = 0  # all real

CKPT_MAP = {
    "P8A_REFERENCE_STEP5000": "P8A",
    "E2B_TOP_N_STEP3200": "E2B",
    "PA_TOP_N_STEP3800": "PA_3800",
}

THUMB_MAX_EDGE = 256


def main() -> int:
    (DST / "data").mkdir(parents=True, exist_ok=True)

    # 1. Manifest -----------------------------------------------------------
    manifest_rows = []
    for img in sorted(SRC_FRAMES.iterdir()):
        if img.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        manifest_rows.append({
            "suite": SUITE,
            "video_id": SUITE,           # single-group; collapses to one section
            "frame_path": f"{URI_PREFIX}{img.name}",
            "identity": SUITE,
            "label": LABEL,
            "is_lockbox": False,
            "bucket": BUCKET,
        })
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = DST / "data" / "dor_morning_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"  manifest: {len(manifest_df)} rows -> {manifest_path}")

    # 2. Score CSVs --------------------------------------------------------
    raw_prefix = "raw/all/dor_morning/"
    for src_name, dst_name in CKPT_MAP.items():
        src_csv = SRC_SCORES / f"{src_name}.csv"
        if not src_csv.exists():
            print(f"  WARN missing {src_csv}", file=sys.stderr)
            continue
        df = pd.read_csv(src_csv)
        def to_uri(fp: str) -> str:
            stem = Path(fp).name
            return f"{URI_PREFIX}{stem}"
        df["frame_path"] = df["frame_path"].astype(str).map(to_uri)
        out_path = DST / "data" / f"dor_morning_scores_{dst_name}.csv"
        df[["frame_path", "frame_prob"]].to_csv(out_path, index=False)
        print(f"  scores [{dst_name}]: {len(df)} rows -> {out_path}")

    # 3. Stage frames + thumbs ---------------------------------------------
    n_frames = 0
    n_thumbs = 0
    for row in manifest_rows:
        src_name = Path(row["frame_path"]).name
        target_name = f"{SUITE}__{src_name}"
        src_img = SRC_FRAMES / src_name
        dst_img = DST / "frames" / SUITE / target_name
        dst_jpg = (DST / "thumbs" / SUITE /
                   Path(target_name).with_suffix(".jpg").name)
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_jpg.parent.mkdir(parents=True, exist_ok=True)
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
            n_frames += 1
        if not dst_jpg.exists():
            with Image.open(src_img) as im:
                im = im.convert("RGB")
                im.thumbnail((THUMB_MAX_EDGE, THUMB_MAX_EDGE), Image.LANCZOS)
                im.save(dst_jpg, "JPEG", quality=85)
            n_thumbs += 1

    print(f"  staged: {n_frames} frames copied, {n_thumbs} thumbs generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
