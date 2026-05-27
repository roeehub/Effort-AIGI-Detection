#!/usr/bin/env python3
"""Adapt Stream B's tsne_combined.csv into the viewer's expected schema.

Stream B's CSV: frame_path, label, source, family_key, local_path, x, y, model
Viewer wants:   frame_path, label, method, identity_key, video_id,
                gcs_uri, tsne_x, tsne_y, checkpoint

This script preserves Stream B's outputs verbatim and produces a parallel
viewer-ready file.
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
TRAINING_DIR = HERE.parent.parent

SRC = TRAINING_DIR / "analysis/clip_vs_p8a_viso_2026-05-03/outputs/tsne_combined.csv"
OUT_P8A = HERE / "outputs/tsne_for_viewer__p8a.csv"
OUT_CLIP = HERE / "outputs/tsne_for_viewer__clip_b16_raw.csv"
OUT_BOTH = HERE / "outputs/tsne_for_viewer__combined.csv"

# Map Stream B's `model` column to the viewer's `checkpoint` column;
# values must match a run's `aliases` field for the manifold filter to admit them.
MODEL_TO_CHECKPOINT = {
    "p8a": "P8A",
    "clip_b16_raw": "CLIP_B16_RAW",
}


def adapt_row(row: dict) -> dict:
    return {
        "frame_path": row["frame_path"],
        "local_path": row.get("local_path", ""),
        "label": row["label"],  # 1=fake, 0=real
        "method": row.get("source", "unknown"),  # viso_fake or teams_real_dev
        "identity_key": "",
        "video_id": "",
        "gcs_uri": row["frame_path"],  # same as frame_path; helps frame proxy fallback
        "family_key": row.get("family_key", ""),
        "tsne_x": row["x"],
        "tsne_y": row["y"],
        "checkpoint": MODEL_TO_CHECKPOINT.get(row["model"], row["model"]),
    }


def main() -> None:
    with SRC.open() as fh:
        reader = csv.DictReader(fh)
        rows = [adapt_row(r) for r in reader]

    fields = list(rows[0].keys())
    p8a_rows = [r for r in rows if r["checkpoint"] == "P8A"]
    clip_rows = [r for r in rows if r["checkpoint"] == "CLIP_B16_RAW"]

    for path, subset in [(OUT_P8A, p8a_rows), (OUT_CLIP, clip_rows), (OUT_BOTH, rows)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(subset)

    print(f"Wrote {OUT_P8A.relative_to(TRAINING_DIR)}: {len(p8a_rows)} rows (checkpoint=P8A)")
    print(f"Wrote {OUT_CLIP.relative_to(TRAINING_DIR)}: {len(clip_rows)} rows (checkpoint=CLIP_B16_RAW)")
    print(f"Wrote {OUT_BOTH.relative_to(TRAINING_DIR)}: {len(rows)} rows (both checkpoints)")


if __name__ == "__main__":
    main()
