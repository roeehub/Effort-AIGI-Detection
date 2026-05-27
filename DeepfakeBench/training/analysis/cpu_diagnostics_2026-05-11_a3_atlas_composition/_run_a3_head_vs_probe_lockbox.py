"""Compare TRAINED-HEAD AUC vs LR-probe AUC on the 87 triptych-lockbox frames.

This is a follow-up to the per-substrate inv_mean recomputation. The atlas's
forgery_auc is an LR PROBE on L11 features (cross-validated). Production uses
the TRAINED HEAD on the full lockbox suite.

We compute trained-head AUC on:
  - the 87 triptych-lockbox frames (the slice the atlas actually saw)
  - the full lockbox suite (1418 real + 425 fake frames)
to quantify the gap between "what the atlas sees" and "what the contract sees".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT = REPO / "analysis" / "cpu_diagnostics_2026-05-11_a3_atlas_composition"
SCORE_DIR = REPO / "analysis" / "cpu_diagnostics_2026-05-10" / "_t4_scorecard_local"
SAMPLED = REPO / "analysis" / "embedding_triptych_2026-04-30" / "outputs" / "triptych_p8a_slot2_slot3" / "sampled_frames.csv"


def main():
    samp = pd.read_csv(SAMPLED, usecols=["gcs_uri", "split"]).iloc[:800].reset_index(drop=True)
    lockbox_triptych_uris = set(samp[samp["split"] == "lockbox"]["gcs_uri"].tolist())

    rows = []

    for ckpt_label, file_tag in [
        ("P8A_REFERENCE_STEP5000", "p8a_reference_step5000"),
        ("T4_LAMBDA1_TOP_N_STEP10500", "t4_lambda1_top_n_step10500"),
        ("T4_LAMBDA2_PERIODIC_STEP1500", "t4_lambda2_periodic_step1500"),
    ]:
        real_rpt = pd.read_csv(SCORE_DIR / f"teams_real_all_lockbox_{file_tag}_frames_report.csv")
        fake_rpt = pd.read_csv(SCORE_DIR / f"teams_fake_all_lockbox_{file_tag}_frames_report.csv")

        # Full-lockbox trained-head AUC (frame-level)
        labels_full = np.concatenate([np.zeros(len(real_rpt)), np.ones(len(fake_rpt))])
        scores_full = np.concatenate([real_rpt["frame_prob"].values, fake_rpt["frame_prob"].values])
        full_auc = float(roc_auc_score(labels_full, scores_full))

        # Triptych-lockbox subset trained-head AUC
        real_in = real_rpt[real_rpt["frame_path"].isin(lockbox_triptych_uris)]
        fake_in = fake_rpt[fake_rpt["frame_path"].isin(lockbox_triptych_uris)]
        if len(real_in) and len(fake_in):
            labels_sub = np.concatenate([np.zeros(len(real_in)), np.ones(len(fake_in))])
            scores_sub = np.concatenate([real_in["frame_prob"].values, fake_in["frame_prob"].values])
            sub_auc = float(roc_auc_score(labels_sub, scores_sub))
            real_p50 = float(np.median(real_in["frame_prob"].values))
            fake_p50 = float(np.median(fake_in["frame_prob"].values))
        else:
            sub_auc = float("nan")
            real_p50 = fake_p50 = float("nan")

        rows.append({
            "ckpt": ckpt_label,
            "n_real_full": len(real_rpt),
            "n_fake_full": len(fake_rpt),
            "trained_head_AUC_full_lockbox_frames": full_auc,
            "n_real_triptych_subset": len(real_in),
            "n_fake_triptych_subset": len(fake_in),
            "trained_head_AUC_triptych_lockbox_subset": sub_auc,
            "triptych_real_p50_score": real_p50,
            "triptych_fake_p50_score": fake_p50,
        })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / "trained_head_vs_probe_lockbox.csv", index=False)
    print(tbl.to_string(index=False, float_format="%.4f"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
