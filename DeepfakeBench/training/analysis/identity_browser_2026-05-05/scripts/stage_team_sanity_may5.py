"""Stage the 5-identity team sanity bucket into the identity browser.

Reads existing team_sanity_check scores + local PNGs and produces:
  data/team_sanity_may5_manifest.csv
  data/team_sanity_may5_scores_<CKPT>.csv  (3 ckpts: P8A, E2B, PA_3800)
  frames/team_may5__<Identity>/team_sanity_may5__<basename>.png  (copied)
  thumbs/team_may5__<Identity>/team_sanity_may5__<basename>.jpg  (256-px thumb)

After this runs, build_browser.py can be invoked with both manifests + the new
score CSVs and --skip-download --skip-thumb.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
SRC_FRAMES = ROOT / "analysis/team_sanity_check_2026-05-05/frames"
SRC_SCORES = ROOT / "analysis/team_sanity_check_2026-05-05/scores"
DST = ROOT / "analysis/identity_browser_2026-05-05"

GS_PREFIX = "gs://real-teams-dor-roee/Roee-Dor-Xiang-Xinhe-noyn-may5/"
SUITE = "team_sanity_may5"
BUCKET = "real-teams-dor-roee"

# Source ckpt CSV name -> output ckpt name in browser
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
    for ident_dir in sorted(SRC_FRAMES.iterdir()):
        if not ident_dir.is_dir():
            continue
        identity = ident_dir.name  # Dor, Noyn, Roee, Xiang, Xinhe
        base_video_id = f"team_may5__{identity}"
        for png in sorted(ident_dir.glob("*.png")):
            rel_in_bucket = f"frames/{identity}/{png.name}"
            gs_uri = f"{GS_PREFIX}{rel_in_bucket}"
            manifest_rows.append({
                "suite": SUITE,
                "video_id": base_video_id,
                "frame_path": gs_uri,
                "identity": identity,
                "label": 0,
                "is_lockbox": False,
                "bucket": BUCKET,
            })
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = DST / "data" / "team_sanity_may5_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"  manifest: {len(manifest_df)} rows -> {manifest_path}")

    # 2. Score CSVs ---------------------------------------------------------
    for src_name, dst_name in CKPT_MAP.items():
        src_csv = SRC_SCORES / f"{src_name}.csv"
        if not src_csv.exists():
            print(f"  WARN missing {src_csv}", file=sys.stderr)
            continue
        df = pd.read_csv(src_csv)
        # frame_path was 'frames/Dor/dor shkedi__...png'; turn into gs:// URI
        df["frame_path"] = GS_PREFIX + df["frame_path"].astype(str)
        out_path = DST / "data" / f"team_sanity_may5_scores_{dst_name}.csv"
        df[["frame_path", "frame_prob"]].to_csv(out_path, index=False)
        print(f"  scores [{dst_name}]: {len(df)} rows -> {out_path}")

    # 3. Stage frames + thumbs ---------------------------------------------
    n_frames = 0
    n_thumbs = 0
    for row in manifest_rows:
        identity = row["identity"]
        base_video_id = row["video_id"]
        gs_uri = row["frame_path"]
        src_name = Path(gs_uri).name  # e.g., "dor shkedi__frame_000004_seq10.png"
        target_name = f"{SUITE}__{src_name}"

        src_png = SRC_FRAMES / identity / src_name
        dst_png = DST / "frames" / base_video_id / target_name
        dst_jpg = DST / "thumbs" / base_video_id / Path(target_name).with_suffix(".jpg").name

        dst_png.parent.mkdir(parents=True, exist_ok=True)
        dst_jpg.parent.mkdir(parents=True, exist_ok=True)

        if not dst_png.exists():
            shutil.copy2(src_png, dst_png)
            n_frames += 1

        if not dst_jpg.exists():
            with Image.open(src_png) as im:
                im = im.convert("RGB")
                im.thumbnail((THUMB_MAX_EDGE, THUMB_MAX_EDGE), Image.LANCZOS)
                im.save(dst_jpg, "JPEG", quality=85)
            n_thumbs += 1

    print(f"  staged: {n_frames} frames copied, {n_thumbs} thumbs generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
