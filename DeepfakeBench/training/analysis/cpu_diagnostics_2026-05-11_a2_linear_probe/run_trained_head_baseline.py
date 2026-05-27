"""Sanity baseline — compute trained-head video-level AUC on the SAME video subset
used by the linear probe, for direct comparability with probe AUC.

This isolates whether the AUC=1.0 probe result is real or sample-size artifact.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
LOCKBOX_DIR = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-10/_t4_scorecard_local"
OUT_DIR = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-11_a2_linear_probe"
CACHE_DIR = OUT_DIR / "_cache"

CKPTS = {
    "T4_LAMBDA1_TOP_N_STEP10500": "t4_lambda1_top_n_step10500",
    "P8A_REFERENCE_STEP5000": "p8a_reference_step5000",
}

logger = logging.getLogger("a2-headbaseline")


def head_auc(ckpt_tag: str, video_ids_subset: set[str] | None = None) -> dict:
    real_csv = LOCKBOX_DIR / f"teams_real_all_lockbox_{ckpt_tag}_frames_report.csv"
    fake_csv = LOCKBOX_DIR / f"teams_fake_all_lockbox_{ckpt_tag}_frames_report.csv"
    real_df = pd.read_csv(real_csv)
    fake_df = pd.read_csv(fake_csv)

    # Aggregate to video-level via mean of frame_prob
    real_vid = real_df.groupby("video_id")["frame_prob"].mean().reset_index()
    fake_vid = fake_df.groupby("video_id")["frame_prob"].mean().reset_index()

    full = {"all_real_n": len(real_vid), "all_fake_n": len(fake_vid),
            "all_auc": roc_auc_score(
                np.concatenate([np.zeros(len(real_vid)), np.ones(len(fake_vid))]),
                np.concatenate([real_vid["frame_prob"].values, fake_vid["frame_prob"].values]),
            )}

    if video_ids_subset is not None:
        rsub = real_vid[real_vid["video_id"].isin(video_ids_subset)]
        fsub = fake_vid[fake_vid["video_id"].isin(video_ids_subset)]
        full["subset_real_n"] = len(rsub)
        full["subset_fake_n"] = len(fsub)
        if len(rsub) >= 2 and len(fsub) >= 2:
            full["subset_auc"] = roc_auc_score(
                np.concatenate([np.zeros(len(rsub)), np.ones(len(fsub))]),
                np.concatenate([rsub["frame_prob"].values, fsub["frame_prob"].values]),
            )
    return full


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    # Load probe input video ids (T4 cache as canonical — both probes use same)
    cache = np.load(CACHE_DIR / "video_feats__T4_LAMBDA1_TOP_N_STEP10500__L11.npz", allow_pickle=True)
    probe_vids = set(cache["video_ids"].tolist())
    logger.info("probe used %d videos", len(probe_vids))

    rows = []
    for ckpt_name, ckpt_tag in CKPTS.items():
        out = head_auc(ckpt_tag, probe_vids)
        out["ckpt"] = ckpt_name
        rows.append(out)
        logger.info("[%s] full_lockbox: %d reals + %d fakes → AUC=%.4f",
                    ckpt_name, out["all_real_n"], out["all_fake_n"], out["all_auc"])
        if "subset_auc" in out:
            logger.info("[%s] probe_subset:   %d reals + %d fakes → AUC=%.4f",
                        ckpt_name, out["subset_real_n"], out["subset_fake_n"], out["subset_auc"])
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "trained_head_baseline_auc.csv", index=False)
    print(df)


if __name__ == "__main__":
    raise SystemExit(main())
