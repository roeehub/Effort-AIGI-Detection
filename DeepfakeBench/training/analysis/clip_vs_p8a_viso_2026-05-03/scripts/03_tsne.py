#!/usr/bin/env python3
"""Step 3 — t-SNE on (CLIP-raw + P8A) features for the viewer manifold panel.

Outputs:
  outputs/tsne_clip_b16_raw.csv (frame_path,x,y,label,source,model)
  outputs/tsne_p8a.csv          (frame_path,x,y,label,source,model)
  outputs/tsne_combined.csv     (concatenation, with extra `model` column)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

REPO_ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
)
HERE = REPO_ROOT / "analysis" / "clip_vs_p8a_viso_2026-05-03"
OUTPUTS = HERE / "outputs"

SEED = 737

logger = logging.getLogger("clip-vs-p8a-tsne")


def run_tsne(features: np.ndarray) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=SEED,
        n_jobs=1,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(features)


def main() -> int:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
                        handlers=[logging.FileHandler(HERE / "run.log", mode="a"), logging.StreamHandler(sys.stdout)])

    clip = np.load(OUTPUTS / "clip_b16_raw__features.npz", allow_pickle=True)
    p8a = np.load(OUTPUTS / "p8a__features.npz", allow_pickle=True)

    fp_clip = np.array([str(s) for s in clip["frame_path"]])
    fp_p8a = np.array([str(s) for s in p8a["frame_path"]])
    common = sorted(set(fp_clip.tolist()) & set(fp_p8a.tolist()))
    logger.info("aligning on %d common frames", len(common))

    def to_dict(d):
        out = {}
        for k in d.files:
            arr = d[k]
            if arr.dtype == object:
                arr = np.array([str(s) for s in arr])
            out[k] = arr
        return out

    def filter_to(d, keep):
        keep_set = set(keep)
        fp = d["frame_path"]
        mask = np.array([fp_i in keep_set for fp_i in fp])
        order = np.argsort(fp[mask])
        out = {}
        for k, arr in d.items():
            arr = arr[mask][order]
            out[k] = arr
        return out

    clipf = filter_to(to_dict(clip), common)
    p8af = filter_to(to_dict(p8a), common)
    assert (clipf["frame_path"] == p8af["frame_path"]).all(), "alignment mismatch"

    logger.info("Running TSNE on CLIP-raw features (n=%d, dim=%d)",
                clipf["features"].shape[0], clipf["features"].shape[1])
    coords_clip = run_tsne(clipf["features"].astype(np.float32))

    logger.info("Running TSNE on P8A features (n=%d, dim=%d)",
                p8af["features"].shape[0], p8af["features"].shape[1])
    coords_p8a = run_tsne(p8af["features"].astype(np.float32))

    base = pd.DataFrame(
        {
            "frame_path": clipf["frame_path"].astype(str),
            "label": clipf["label"].astype(int),
            "source": clipf["source"].astype(str),
            "family_key": clipf["family_key"].astype(str),
            "local_path": clipf["local_path"].astype(str),
        }
    )

    df_clip = base.copy()
    df_clip["x"] = coords_clip[:, 0]
    df_clip["y"] = coords_clip[:, 1]
    df_clip["model"] = "clip_b16_raw"

    df_p8a = base.copy()
    df_p8a["x"] = coords_p8a[:, 0]
    df_p8a["y"] = coords_p8a[:, 1]
    df_p8a["model"] = "p8a"

    df_clip.to_csv(OUTPUTS / "tsne_clip_b16_raw.csv", index=False)
    df_p8a.to_csv(OUTPUTS / "tsne_p8a.csv", index=False)
    pd.concat([df_clip, df_p8a], ignore_index=True).to_csv(OUTPUTS / "tsne_combined.csv", index=False)

    logger.info("Wrote tsne_clip_b16_raw.csv, tsne_p8a.csv, tsne_combined.csv (%d rows each per-model)", len(df_clip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
