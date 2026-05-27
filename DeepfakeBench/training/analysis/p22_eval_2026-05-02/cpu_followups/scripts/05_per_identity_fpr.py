"""Job E — per-identity P22 lockbox FPR concentration.

Is P22 step8k's lockbox FPR concentrated on specific identities or modes?
The pre-existing memory (`project_lockbox_fpr_dominated_by_webcam_mode`) says
webcam-mode reals drive 65.7% of lockbox FPR for P8A. Does the same pattern
hold for P22 step8k, or has P22 broken into a new failure mode?
"""
from __future__ import annotations

import sys; sys.path.insert(0, "analysis/p22_eval_2026-05-02/cpu_followups/scripts")
import re

import numpy as np
import pandas as pd

from _common import OUT, load_per_frame_scores

CKPTS = ["P8A_REFERENCE_STEP5000", "P18T_GRL_TREATMENT_STEP4000",
         "P22_AUG_STEP1000", "P22_AUG_STEP4000", "P22_AUG_STEP8000"]


def parse_identity(video_id: str) -> str:
    """video_id formats:
       'Chikara_Takahashi__s22__seg_10.0__real' → 'Chikara_Takahashi'
       'dor_shkedi__seq5775__real' → 'dor_shkedi'
       'PC_Generator' → 'PC_Generator'
    """
    if pd.isna(video_id):
        return "<missing>"
    parts = re.split(r"__s\d+|__seg_|__seq\d+", video_id)
    return parts[0] if parts else video_id


def parse_session(video_id: str) -> str:
    """Capture mode hint from video_id, if available (e.g. 's22' is session)."""
    if pd.isna(video_id):
        return "<missing>"
    m = re.search(r"__s(\d+)", video_id)
    return f"s{m.group(1)}" if m else "<no_session>"


def per_identity_fpr(ckpt, suite="teams_real_all_lockbox"):
    df = load_per_frame_scores(ckpt, suite)
    if df is None: return None
    # Calibrate τ on dev
    real_dev = load_per_frame_scores(ckpt, "teams_real_all_dev")
    if real_dev is None: return None
    rd_sorted = np.sort(real_dev["frame_prob"].to_numpy())
    floor = 0.02
    idx = int(np.ceil(len(rd_sorted) * (1 - floor)))
    tau = float(rd_sorted[min(idx, len(rd_sorted) - 1)])

    df["identity"] = df["video_id"].apply(parse_identity)
    df["session"] = df["video_id"].apply(parse_session)
    df["is_FP"] = (df["frame_prob"] >= tau).astype(int)

    # Per-identity FPR (frame-level, not video-level — for granularity)
    by_id = df.groupby("identity").agg(
        n_frames=("frame_prob", "count"),
        n_FP=("is_FP", "sum"),
        mean_score=("frame_prob", "mean"),
    ).reset_index()
    by_id["fpr_per_id"] = by_id["n_FP"] / by_id["n_frames"]
    by_id["ckpt"] = ckpt
    return by_id, tau


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Per-id table per ckpt
    all_id_rows = []
    print("=" * 80)
    print("Identity-level lockbox FPR (P22 step8k vs P8A) — top false-positives")
    print("=" * 80)
    summary_rows = []
    for ckpt in CKPTS:
        result = per_identity_fpr(ckpt)
        if result is None: continue
        by_id, tau = result
        all_id_rows.append(by_id)
        # Top-10 FPR identities
        top = by_id.sort_values("n_FP", ascending=False).head(10)
        print(f"\n{ckpt}  (τ={tau:.3f}, total FPs across all identities: {by_id.n_FP.sum()})")
        print(top[["identity", "n_frames", "n_FP", "fpr_per_id", "mean_score"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))

        # Concentration metric: what % of FPs come from top-K identities
        sorted_by_fp = by_id.sort_values("n_FP", ascending=False)
        total_fp = by_id.n_FP.sum()
        n_ids_with_fp = (by_id.n_FP > 0).sum()
        if total_fp > 0:
            top1_share = sorted_by_fp.iloc[0].n_FP / total_fp
            top5_share = sorted_by_fp.iloc[:5].n_FP.sum() / total_fp
            top10_share = sorted_by_fp.iloc[:10].n_FP.sum() / total_fp
        else:
            top1_share = top5_share = top10_share = float("nan")
        summary_rows.append({"ckpt": ckpt, "tau": tau, "total_fp": int(total_fp),
                             "total_identities": int(len(by_id)),
                             "identities_with_any_fp": int(n_ids_with_fp),
                             "top1_id_share_of_FPs": float(top1_share),
                             "top5_id_share_of_FPs": float(top5_share),
                             "top10_id_share_of_FPs": float(top10_share)})

    full_id = pd.concat(all_id_rows, ignore_index=True)
    full_id.to_csv(OUT / "05_per_identity_fpr.csv", index=False)
    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(OUT / "05_per_identity_fpr_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("Concentration of lockbox FPR across identities")
    print("=" * 80)
    print(sdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
