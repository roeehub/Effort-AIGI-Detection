"""Stage the live-fakes-teams-prod bucket into the identity browser.

Reads scored CSVs from analysis/live_fakes_teams_prod_2026-05-05/scores/ and
local PNGs from analysis/live_fakes_teams_prod_2026-05-05/raw/<session>/<tag>/
and produces:
  data/live_fakes_manifest.csv
  data/live_fakes_scores_<CKPT>.csv  (3 ckpts: P8A, E2B, PA_3800)
  frames/live_prod__<tag>/live_fakes_teams_prod__<basename>.png
  thumbs/live_prod__<tag>/live_fakes_teams_prod__<basename>.jpg

Each tag (e.g., xiang-fake-1, xinhe-fake-8-glasses) becomes its own
base_identity, so the browser groups them per-variant.

After this runs, build_browser.py is invoked with all 3 manifests + new score
CSVs and --skip-download --skip-thumb.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
SRC_ROOT = ROOT / "analysis/live_fakes_teams_prod_2026-05-05"
SRC_FRAMES = SRC_ROOT / "raw/session_20260414_112354"
SRC_SCORES = SRC_ROOT / "scores"
DST = ROOT / "analysis/identity_browser_2026-05-05"

GS_PREFIX = "gs://live-fakes-teams-prod/fake/session_20260414_112354/"
SUITE = "live_fakes_teams_prod"
BUCKET = "live-fakes-teams-prod"
LABEL = 1  # all fake

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
    for tag_dir in sorted(SRC_FRAMES.iterdir()):
        if not tag_dir.is_dir() or tag_dir.name == "metadata":
            continue
        tag = tag_dir.name  # xiang-fake-1, xinhe-fake-8-glasses, ...
        base_video_id = f"live_prod__{tag}"
        for png in sorted(tag_dir.glob("*.png")):
            rel_in_bucket = f"{tag}/{png.name}"
            gs_uri = f"{GS_PREFIX}{rel_in_bucket}"
            manifest_rows.append({
                "suite": SUITE,
                "video_id": base_video_id,
                "frame_path": gs_uri,
                "identity": tag,
                "label": LABEL,
                "is_lockbox": False,
                "bucket": BUCKET,
            })
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = DST / "data" / "live_fakes_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"  manifest: {len(manifest_df)} rows, "
          f"{manifest_df['identity'].nunique()} tags -> {manifest_path}")

    # 2. Score CSVs ---------------------------------------------------------
    # Inference wrote frame_path = "raw/session_20260414_112354/<tag>/<file>"
    # We need to map to gs:// URI used by the browser manifest.
    raw_prefix = "raw/session_20260414_112354/"
    for src_name, dst_name in CKPT_MAP.items():
        src_csv = SRC_SCORES / f"{src_name}.csv"
        if not src_csv.exists():
            print(f"  WARN missing {src_csv}", file=sys.stderr)
            continue
        df = pd.read_csv(src_csv)
        # Strip "raw/session_20260414_112354/" prefix and prepend gs:// URI
        def to_gs(fp: str) -> str:
            if fp.startswith(raw_prefix):
                return GS_PREFIX + fp[len(raw_prefix):]
            return GS_PREFIX + fp
        df["frame_path"] = df["frame_path"].astype(str).map(to_gs)
        out_path = DST / "data" / f"live_fakes_scores_{dst_name}.csv"
        df[["frame_path", "frame_prob"]].to_csv(out_path, index=False)
        print(f"  scores [{dst_name}]: {len(df)} rows -> {out_path}")

    # 3. Stage frames + thumbs ---------------------------------------------
    n_frames = 0
    n_thumbs = 0
    for row in manifest_rows:
        tag = row["identity"]
        base_video_id = row["video_id"]
        gs_uri = row["frame_path"]
        src_name = Path(gs_uri).name
        target_name = f"{SUITE}__{src_name}"

        src_png = SRC_FRAMES / tag / src_name
        dst_png = DST / "frames" / base_video_id / target_name
        dst_jpg = (DST / "thumbs" / base_video_id /
                   Path(target_name).with_suffix(".jpg").name)

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
