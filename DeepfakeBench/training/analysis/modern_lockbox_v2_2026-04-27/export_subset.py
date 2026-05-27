"""Export modern_lockbox_real_v2 subset as a frame-list yaml + per-video aggregation.

Produces:
- modern_lockbox_real_v2_frames.yaml: list of gcs_uri's that pass the filter.
- modern_lockbox_real_v2_videos.yaml: list of video_id's where >50% of frames pass.
- modern_lockbox_fake_v2_frames.yaml: same filter on fakes (for paired recall).
"""
from __future__ import annotations

import yaml
from pathlib import Path

import pandas as pd


REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
JOIN_CSV = REPO / "analysis/crop_shortcut_2026-04-27/p8a_lockbox_join_2026-04-27.csv"
OUT_DIR = REPO / "analysis/modern_lockbox_v2_2026-04-27"


def main() -> None:
    df = pd.read_csv(JOIN_CSV)
    lockbox = df[df["split"] == "lockbox"].copy()

    # Apply v2_recommended filter inline (avoid module-name import issue).
    mask = (
        (lockbox["clip_capture_mode"] != "webcam")
        & (lockbox["clip_capture_mode"] != "screen")
        & (lockbox["face_area_ratio"] >= 0.10)
        & (~lockbox["is_pose_extreme"].fillna(False))
        & (~lockbox["is_no_face"].fillna(False))
    )
    keep = lockbox[mask].copy()
    real = keep[keep["label"] == "real"]
    fake = keep[keep["label"] == "fake"]

    real_frames = sorted(real["gcs_uri"].dropna().tolist())
    fake_frames = sorted(fake["gcs_uri"].dropna().tolist())

    # Per-video: video kept if >=50% of its lockbox frames pass.
    by_video = lockbox.groupby(["video_id", "label"]).agg(
        n_total=("gcs_uri", "size"),
        n_kept=("gcs_uri", lambda s: int(mask.loc[s.index].sum())),
    ).reset_index()
    by_video["frac_kept"] = by_video["n_kept"] / by_video["n_total"]
    kept_videos = by_video[by_video["frac_kept"] >= 0.5].copy()
    real_videos = sorted(kept_videos[kept_videos["label"] == "real"]["video_id"].tolist())
    fake_videos = sorted(kept_videos[kept_videos["label"] == "fake"]["video_id"].tolist())

    out_real_frames = OUT_DIR / "modern_lockbox_real_v2_frames.yaml"
    out_fake_frames = OUT_DIR / "modern_lockbox_fake_v2_frames.yaml"
    out_real_videos = OUT_DIR / "modern_lockbox_real_v2_videos.yaml"
    out_fake_videos = OUT_DIR / "modern_lockbox_fake_v2_videos.yaml"

    yaml.safe_dump({
        "subset_name": "modern_lockbox_real_v2",
        "split": "lockbox",
        "label": "real",
        "filter_rule": (
            "clip_capture_mode not in {webcam, screen} AND "
            "face_area_ratio >= 0.10 AND "
            "not is_pose_extreme AND not is_no_face"
        ),
        "n_frames": len(real_frames),
        "frames": real_frames,
    }, open(out_real_frames, "w"), sort_keys=False)
    yaml.safe_dump({
        "subset_name": "modern_lockbox_fake_v2",
        "split": "lockbox",
        "label": "fake",
        "filter_rule": "(same as modern_lockbox_real_v2)",
        "n_frames": len(fake_frames),
        "frames": fake_frames,
    }, open(out_fake_frames, "w"), sort_keys=False)
    yaml.safe_dump({
        "subset_name": "modern_lockbox_real_v2_videos",
        "aggregation_rule": "video kept if >=50% of its frames pass v2 filter",
        "n_videos": len(real_videos),
        "videos": real_videos,
    }, open(out_real_videos, "w"), sort_keys=False)
    yaml.safe_dump({
        "subset_name": "modern_lockbox_fake_v2_videos",
        "aggregation_rule": "video kept if >=50% of its frames pass v2 filter",
        "n_videos": len(fake_videos),
        "videos": fake_videos,
    }, open(out_fake_videos, "w"), sort_keys=False)

    print(f"wrote {out_real_frames}: {len(real_frames)} frames")
    print(f"wrote {out_fake_frames}: {len(fake_frames)} frames")
    print(f"wrote {out_real_videos}: {len(real_videos)} videos")
    print(f"wrote {out_fake_videos}: {len(fake_videos)} videos")


if __name__ == "__main__":
    main()
