"""Stage visomaster-enhanced-face-cropped-v2 (16 swap models) into the browser.

Each swap-model dir becomes its own base_identity (dor_fake_<swap>) under suite
visomaster_v2_dor, label=1.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
SRC_ROOT = ROOT / "analysis/visomaster_v2_2026-05-05"
SRC_FRAMES = SRC_ROOT / "raw/all"
SRC_SCORES = SRC_ROOT / "scores"
DST = ROOT / "analysis/identity_browser_2026-05-05"

GS_PREFIX = "gs://visomaster-enhanced-face-cropped-v2/fake/"
SUITE = "visomaster_v2_dor"
BUCKET = "visomaster-enhanced-face-cropped-v2"
LABEL = 1

CKPT_MAP = {
    "P8A_REFERENCE_STEP5000": "P8A",
    "E2B_TOP_N_STEP3200": "E2B",
    "PA_TOP_N_STEP3800": "PA_3800",
}

THUMB_MAX_EDGE = 256


def main() -> int:
    (DST / "data").mkdir(parents=True, exist_ok=True)

    # ---- Manifest ----
    manifest_rows = []
    for ident_dir in sorted(SRC_FRAMES.iterdir()):
        if not ident_dir.is_dir():
            continue
        # ident_dir.name = "dor_fake_ghostface_v1" — strip prefix to get swap key
        swap = ident_dir.name[len("dor_fake_"):]
        for img in sorted(ident_dir.iterdir()):
            if img.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            uri = f"{GS_PREFIX}{swap}/{img.name}"
            manifest_rows.append({
                "suite": SUITE,
                "video_id": ident_dir.name,
                "frame_path": uri,
                "identity": ident_dir.name,
                "label": LABEL,
                "is_lockbox": False,
                "bucket": BUCKET,
            })
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = DST / "data" / "visomaster_v2_dor_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"  manifest: {len(manifest_df)} rows, "
          f"{manifest_df['identity'].nunique()} swap models -> {manifest_path}")

    # ---- Score CSVs ----
    raw_prefix = "raw/all/"
    for src_name, dst_name in CKPT_MAP.items():
        src_csv = SRC_SCORES / f"{src_name}.csv"
        df = pd.read_csv(src_csv)
        # frame_path "raw/all/dor_fake_ghostface_v1/<file>.png" -> gs URI
        def to_uri(fp: str) -> str:
            p = Path(fp)
            ident_dir = p.parent.name  # dor_fake_<swap>
            swap = ident_dir[len("dor_fake_"):]
            return f"{GS_PREFIX}{swap}/{p.name}"
        df["frame_path"] = df["frame_path"].astype(str).map(to_uri)
        out_path = DST / "data" / f"visomaster_v2_dor_scores_{dst_name}.csv"
        df[["frame_path", "frame_prob"]].to_csv(out_path, index=False)
        print(f"  scores [{dst_name}]: {len(df)} rows -> {out_path}")

    # ---- Stage frames + thumbs ----
    n_frames = 0
    n_thumbs = 0
    for row in manifest_rows:
        ident = row["video_id"]
        src_name = Path(row["frame_path"]).name
        target_name = f"{SUITE}__{src_name}"
        # Re-derive local source path
        swap = ident[len("dor_fake_"):]
        src_img = SRC_FRAMES / ident / src_name
        dst_img = DST / "frames" / ident / target_name
        dst_jpg = (DST / "thumbs" / ident /
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
