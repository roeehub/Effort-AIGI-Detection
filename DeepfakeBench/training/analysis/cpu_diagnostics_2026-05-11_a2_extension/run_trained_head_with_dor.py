"""Compute trained-head AUC on the same 309-real + 253-fake subset (now including real_dor)."""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
LOCKBOX_DIR = REPO / "analysis/cpu_diagnostics_2026-05-10/_t4_scorecard_local"
OUT_DIR = REPO / "analysis/cpu_diagnostics_2026-05-11_a2_extension"

PAIRS = {
    "T4_LAMBDA1_TOP_N_STEP10500": (
        "teams_real_all_lockbox_t4_lambda1_top_n_step10500_frames_report.csv",
        "teams_fake_all_lockbox_t4_lambda1_top_n_step10500_frames_report.csv",
    ),
    "P8A_REFERENCE_STEP5000": (
        "teams_real_all_lockbox_p8a_reference_step5000_frames_report.csv",
        "teams_fake_all_lockbox_p8a_reference_step5000_frames_report.csv",
    ),
}

# Reproduce probe subset = same video_ids used in run_a2_extension.py
cache = np.load(OUT_DIR / "_cache" / "video_feats_with_dor__T4_LAMBDA1_TOP_N_STEP10500__L11.npz",
                allow_pickle=True)
subset_vids = set(cache["video_ids"].tolist())

rows = []
for label, (rcsv, fcsv) in PAIRS.items():
    real = pd.read_csv(LOCKBOX_DIR / rcsv)
    fake = pd.read_csv(LOCKBOX_DIR / fcsv)
    # video-level: mean of frame_prob per video_id
    real_vid = real.groupby("video_id")["frame_prob"].mean().reset_index()
    fake_vid = fake.groupby("video_id")["frame_prob"].mean().reset_index()
    real_vid["label"] = 0
    fake_vid["label"] = 1
    all_vid = pd.concat([real_vid, fake_vid], axis=0, ignore_index=True)
    # full lockbox
    auc_full = roc_auc_score(all_vid["label"], all_vid["frame_prob"])
    n_real_full = (all_vid["label"] == 0).sum()
    n_fake_full = (all_vid["label"] == 1).sum()
    # subset = video_ids used in probe (309 reals + 253 fakes)
    sub = all_vid[all_vid["video_id"].isin(subset_vids)]
    auc_sub = roc_auc_score(sub["label"], sub["frame_prob"])
    n_real_sub = (sub["label"] == 0).sum()
    n_fake_sub = (sub["label"] == 1).sum()
    # also: only the real_dor cohort + fakes
    dor_only = all_vid[all_vid["video_id"].str.startswith("real_dor") | (all_vid["label"] == 1)]
    dor_only = dor_only[dor_only["video_id"].isin(subset_vids)]
    auc_dor = roc_auc_score(dor_only["label"], dor_only["frame_prob"])
    n_real_dor = ((dor_only["label"] == 0)).sum()
    n_fake_dor = ((dor_only["label"] == 1)).sum()
    # non_dor only
    non_dor = sub[~sub["video_id"].str.startswith("real_dor")]
    auc_non_dor = roc_auc_score(non_dor["label"], non_dor["frame_prob"])
    n_real_nd = (non_dor["label"] == 0).sum()
    n_fake_nd = (non_dor["label"] == 1).sum()
    rows.append({
        "ckpt": label,
        "auc_full_lockbox": auc_full, "n_real_full": int(n_real_full), "n_fake_full": int(n_fake_full),
        "auc_probe_subset_with_dor": auc_sub, "n_real_subset": int(n_real_sub), "n_fake_subset": int(n_fake_sub),
        "auc_real_dor_only_vs_fakes": auc_dor, "n_real_dor": int(n_real_dor), "n_fake_dor": int(n_fake_dor),
        "auc_non_dor_only_vs_fakes": auc_non_dor, "n_real_non_dor": int(n_real_nd), "n_fake_non_dor": int(n_fake_nd),
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_DIR / "trained_head_with_dor_auc.csv", index=False)
print(out_df.to_string())
