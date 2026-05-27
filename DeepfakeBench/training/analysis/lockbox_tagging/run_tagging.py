"""End-to-end tagger: fetch frames, run layers, derive flags, write parquet."""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from analysis.lockbox_tagging.io_utils import (
    FrameRow,
    fetch_frame,
    select_frames,
)
from analysis.lockbox_tagging.layers.face_geometry import compute_face_geometry
from analysis.lockbox_tagging.layers.identity import compute_identity
from analysis.lockbox_tagging.layers.quality import compute_quality
from analysis.lockbox_tagging.layers.semantic import compute_semantic

# Same blur cutoff as training/process_veo2.py:BLUR_THRESHOLD so analysis is comparable.
BLUR_THRESHOLD = 100.0
SMALL_FACE_PIXELS = 96 * 96
# Calibrated against observed lockbox distribution (yaw p99≈24°, pitch p99≈35°).
EXTREME_YAW_DEG = 20.0
EXTREME_PITCH_DEG = 30.0


def derive_flags(row: dict) -> dict:
    """Per-row boolean flags. The ArcFace-NN-based outlier flag is computed in
    analyze.py because it requires the full dataset (k-NN over all embeddings)."""
    sharpness = row.get("sharpness_laplacian")
    face_pixel_area = row.get("face_pixel_area")
    yaw = row.get("yaw_deg")
    pitch = row.get("pitch_deg")
    face_count = row.get("face_count") or 0
    capture = row.get("clip_capture_mode")
    quality = row.get("clip_quality")
    bv_mean = row.get("brightness_v_mean")

    is_low_quality = bool(
        (face_pixel_area is not None and face_pixel_area < SMALL_FACE_PIXELS)
        or (sharpness is not None and sharpness < BLUR_THRESHOLD)
        or (bv_mean is not None and (bv_mean < 25 or bv_mean > 235))
        or (quality in ("pixelated", "blurry"))
    )
    is_pose_extreme = bool(
        (yaw is not None and abs(yaw) > EXTREME_YAW_DEG)
        or (pitch is not None and abs(pitch) > EXTREME_PITCH_DEG)
    )
    is_no_face = bool(face_count == 0)
    is_likely_screen_capture = bool(capture in ("screen", "phone_screen", "screen_recording"))
    return {
        "is_low_quality": is_low_quality,
        "is_pose_extreme": is_pose_extreme,
        "is_no_face": is_no_face,
        "is_likely_screen_capture": is_likely_screen_capture,
    }


def tag_frame(frame: FrameRow, local: Path, layers: set[str]) -> dict:
    """Run requested layers on a single fetched frame."""
    rec: dict = {**asdict(frame), "local_path": str(local)}
    if "quality" in layers:
        rec.update(compute_quality(local))
    if "face" in layers:
        rec.update(compute_face_geometry(local))
    if "identity" in layers:
        bbox = None
        if rec.get("face_bbox_x") is not None:
            bbox = (
                rec["face_bbox_x"],
                rec["face_bbox_y"],
                rec["face_bbox_w"],
                rec["face_bbox_h"],
            )
        rec.update(compute_identity(local, face_bbox=bbox))
    if "semantic" in layers:
        rec.update(compute_semantic(local))
    rec.update(derive_flags(rec))
    return rec


def run(
    scope: str,
    limit: int | None,
    layers: set[str],
    out_path: Path,
    download_workers: int = 16,
) -> pd.DataFrame:
    frames = select_frames(scope=scope, limit=limit)
    print(f"[run] {len(frames)} frames in scope={scope}")

    print(f"[run] fetching frames (workers={download_workers}) ...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=download_workers) as ex:
        local_paths = list(ex.map(fetch_frame, frames))
    print(f"[run] fetched in {time.time() - t0:.1f}s")

    print(f"[run] tagging with layers={sorted(layers)} ...")
    t0 = time.time()
    records = []
    for i, (frame, local) in enumerate(zip(frames, local_paths)):
        if i and i % 25 == 0:
            print(f"[run]   {i}/{len(frames)} (elapsed {time.time() - t0:.0f}s)")
        records.append(tag_frame(frame, local, layers))
    print(f"[run] tagged in {time.time() - t0:.1f}s")

    df = pd.DataFrame.from_records(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[run] wrote {len(df)} rows to {out_path}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="lockbox_with_preds",
                    choices=["lockbox_with_preds", "lockbox_all", "all_preds"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--layers", default="quality,face,identity,semantic",
                    help="Comma-separated subset of {quality,face,identity,semantic}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    layers = {x.strip() for x in args.layers.split(",") if x.strip()}
    run(args.scope, args.limit, layers, args.out, download_workers=args.workers)


if __name__ == "__main__":
    main()
