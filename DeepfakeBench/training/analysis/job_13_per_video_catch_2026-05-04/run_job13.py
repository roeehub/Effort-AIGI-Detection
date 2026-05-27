"""
Job 13 - Per-video catch-rate analysis on visomaster_enhanced_macro_dev.

Pure CPU. Cached scores only. n_jobs=1 throughout.

Treats P8A, E2B_3200, E3_6600 as symmetric candidates.
"""
import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Constants from FACTS Section 1
# ---------------------------------------------------------------------------
RAW_REPORTS = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/"
    "DeepfakeBench/training/analysis/cpu_followups_2026-05-04/raw_reports"
)
OUT_DIR = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/"
    "DeepfakeBench/training/analysis/job_13_per_video_catch_2026-05-04"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPTS = {
    "P8A": {
        "csv": "visomaster_enhanced_macro_dev_p8a_reference_step5000_frames_report.csv",
        "tau_fpr10": 0.7052,
    },
    "E2B_3200": {
        "csv": "visomaster_enhanced_macro_dev_e2b_top_n_step3200_frames_report.csv",
        "tau_fpr10": 0.5075,
    },
    "E3_6600": {
        "csv": "visomaster_enhanced_macro_dev_e3_top_n_step6600_frames_report.csv",
        "tau_fpr10": 0.8526,
    },
}

GAP_THRESHOLD = 10  # frame_num gap > this starts a new derived sequence


# ---------------------------------------------------------------------------
# Loading & sequence derivation
# ---------------------------------------------------------------------------
def parse_frame_path(p):
    fname = p.rsplit("/", 1)[-1]
    m = re.match(
        r"visomaster_enhanced_(raw|teams)__frame_(\d+)_seq(\d+)\.png", fname
    )
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), int(m.group(3))


def load_one(ckpt_name):
    cfg = CKPTS[ckpt_name]
    df = pd.read_csv(RAW_REPORTS / cfg["csv"])
    parsed = df["frame_path"].apply(parse_frame_path)
    df["subtype"] = parsed.apply(lambda x: x[0])
    df["frame_num"] = parsed.apply(lambda x: x[1])
    df["seq_num"] = parsed.apply(lambda x: x[2])
    df["ckpt"] = ckpt_name
    df["tau"] = cfg["tau_fpr10"]
    df["caught"] = (df["frame_prob"] > cfg["tau_fpr10"]).astype(int)
    return df


def derive_sequence_groups(df):
    """
    Group frames into derived video-sequences using frame_num contiguity within
    a subtype.  A new group starts whenever the sorted-frame_num gap exceeds
    GAP_THRESHOLD.

    NOTE: video_id field in the CSV is *per-frame unique*, so it cannot serve
    as a sequence grouping key.  This function reconstructs sequence groupings
    from the regular frame_num spacing observed in the manifest (step ~2 within
    a sequence, large gaps between sequences).
    """
    out = []
    for subtype, sub_df in df.groupby("subtype"):
        sub_df = sub_df.sort_values("frame_num").reset_index(drop=True)
        diffs = sub_df["frame_num"].diff()
        breaks = (diffs > GAP_THRESHOLD).cumsum().fillna(0).astype(int)
        sub_df["seq_group_idx"] = breaks
        sub_df["derived_video_id"] = (
            "viso_"
            + subtype
            + "_seqgrp"
            + sub_df["seq_group_idx"].astype(str).str.zfill(2)
            + "_f"
            + sub_df.groupby("seq_group_idx")["frame_num"]
            .transform("min")
            .astype(int)
            .astype(str)
            + "-"
            + sub_df.groupby("seq_group_idx")["frame_num"]
            .transform("max")
            .astype(int)
            .astype(str)
        )
        out.append(sub_df)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 1+2 - per-video per-ckpt catch
# ---------------------------------------------------------------------------
def per_video_per_ckpt(all_df):
    rows = []
    for (vid, subtype, ckpt), g in all_df.groupby(
        ["derived_video_id", "subtype", "ckpt"]
    ):
        scores = g["frame_prob"].values
        tau = g["tau"].iloc[0]
        n_frames = len(g)
        n_caught = int((scores > tau).sum())
        rows.append(
            {
                "video_id": vid,
                "subtype": subtype,
                "ckpt": ckpt,
                "n_frames": n_frames,
                "n_caught": n_caught,
                "catch_rate": n_caught / n_frames if n_frames else 0.0,
                "mean_score": float(np.mean(scores)),
                "p10": float(np.percentile(scores, 10)),
                "p50": float(np.percentile(scores, 50)),
                "p90": float(np.percentile(scores, 90)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["ckpt", "subtype", "video_id"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3 - bimodality
# ---------------------------------------------------------------------------
def bimodality_table(per_v):
    bins = np.linspace(0, 1, 11)  # 10 bins from 0..1
    rows = []
    for ckpt, g in per_v.groupby("ckpt"):
        hist, _ = np.histogram(g["catch_rate"], bins=bins)
        for i, c in enumerate(hist):
            rows.append(
                {
                    "ckpt": ckpt,
                    "bin_lo": float(bins[i]),
                    "bin_hi": float(bins[i + 1]),
                    "count": int(c),
                }
            )
    return pd.DataFrame(rows)


def bimodality_summary(per_v):
    rows = []
    for ckpt, g in per_v.groupby("ckpt"):
        n_total = len(g)
        n_lt10 = int((g["catch_rate"] < 0.10).sum())
        n_gt90 = int((g["catch_rate"] > 0.90).sum())
        n_mixed = int(
            ((g["catch_rate"] > 0.40) & (g["catch_rate"] < 0.60)).sum()
        )
        rows.append(
            {
                "ckpt": ckpt,
                "n_videos": n_total,
                "n_catch_lt_0.10": n_lt10,
                "n_catch_gt_0.90": n_gt90,
                "n_mixed_0.4_0.6": n_mixed,
                "frac_near_0": n_lt10 / n_total if n_total else 0.0,
                "frac_near_1": n_gt90 / n_total if n_total else 0.0,
                "frac_mixed": n_mixed / n_total if n_total else 0.0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4 - cross-ckpt agreement at video level
# ---------------------------------------------------------------------------
def cross_ckpt_table(per_v):
    pivot_rate = per_v.pivot_table(
        index=["video_id", "subtype"], columns="ckpt", values="catch_rate"
    ).reset_index()
    pivot_rate = pivot_rate.rename(
        columns={
            "P8A": "P8A_catch_rate",
            "E2B_3200": "E2B_catch_rate",
            "E3_6600": "E3_catch_rate",
        }
    )
    pivot_rate["max_minus_min"] = pivot_rate[
        ["P8A_catch_rate", "E2B_catch_rate", "E3_catch_rate"]
    ].max(axis=1) - pivot_rate[
        ["P8A_catch_rate", "E2B_catch_rate", "E3_catch_rate"]
    ].min(
        axis=1
    )

    def klass(row):
        rates = [
            row["P8A_catch_rate"],
            row["E2B_catch_rate"],
            row["E3_catch_rate"],
        ]
        all_high = all(r > 0.5 for r in rates)
        all_low = all(r < 0.5 for r in rates)
        if all_high:
            return "all_high_>0.5"
        if all_low:
            return "all_low_<0.5"
        return "mixed"

    pivot_rate["agreement_class"] = pivot_rate.apply(klass, axis=1)
    return pivot_rate


def cross_ckpt_correlations(cross_df):
    """Pearson r between ckpts at the video level (1 datum per video)."""
    pairs = [
        ("P8A_catch_rate", "E2B_catch_rate"),
        ("P8A_catch_rate", "E3_catch_rate"),
        ("E2B_catch_rate", "E3_catch_rate"),
    ]
    out = {}
    for a, b in pairs:
        x = cross_df[a].values
        y = cross_df[b].values
        if np.std(x) == 0 or np.std(y) == 0:
            r, p = (np.nan, np.nan)
        else:
            r, p = pearsonr(x, y)
        out[f"{a}__vs__{b}"] = {"r": float(r), "p": float(p), "n": int(len(x))}
    return out


def high_disagreement_videos(cross_df, all_df):
    """
    Videos where one ckpt catches > 0.5 and another < 0.1.
    Provide frame-level details.
    """
    rates = cross_df[
        ["P8A_catch_rate", "E2B_catch_rate", "E3_catch_rate"]
    ].values
    flagged = []
    for i, r in enumerate(rates):
        if (r.max() > 0.5) and (r.min() < 0.1):
            flagged.append(cross_df.iloc[i]["video_id"])
    if not flagged:
        return pd.DataFrame()
    flagged_set = set(flagged)
    detail = all_df[all_df["derived_video_id"].isin(flagged_set)].copy()
    keep_cols = [
        "derived_video_id",
        "subtype",
        "ckpt",
        "frame_num",
        "seq_num",
        "frame_prob",
        "tau",
        "caught",
        "frame_path",
    ]
    detail = detail[keep_cols].rename(
        columns={"derived_video_id": "video_id"}
    )
    return detail.sort_values(["video_id", "ckpt", "frame_num"]).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Step 5 - subtype-stratified
# ---------------------------------------------------------------------------
def subtype_stratified_summaries(per_v, cross_df):
    rows = []
    for subtype, g in per_v.groupby("subtype"):
        for ckpt, gc in g.groupby("ckpt"):
            n_total = len(gc)
            rows.append(
                {
                    "subtype": subtype,
                    "ckpt": ckpt,
                    "n_videos": n_total,
                    "n_catch_lt_0.10": int((gc["catch_rate"] < 0.10).sum()),
                    "n_catch_gt_0.90": int((gc["catch_rate"] > 0.90).sum()),
                    "n_mixed_0.4_0.6": int(
                        (
                            (gc["catch_rate"] > 0.40)
                            & (gc["catch_rate"] < 0.60)
                        ).sum()
                    ),
                    "mean_catch_rate": float(gc["catch_rate"].mean()),
                    "median_catch_rate": float(gc["catch_rate"].median()),
                }
            )
    bm = pd.DataFrame(rows)

    # Per-subtype cross-ckpt correlations
    corr_rows = []
    for subtype, g in cross_df.groupby("subtype"):
        if len(g) < 3:
            continue
        for a, b, label in [
            ("P8A_catch_rate", "E2B_catch_rate", "P8A__vs__E2B"),
            ("P8A_catch_rate", "E3_catch_rate", "P8A__vs__E3"),
            ("E2B_catch_rate", "E3_catch_rate", "E2B__vs__E3"),
        ]:
            x = g[a].values
            y = g[b].values
            if np.std(x) == 0 or np.std(y) == 0:
                r, p = (np.nan, np.nan)
            else:
                r, p = pearsonr(x, y)
            corr_rows.append(
                {
                    "subtype": subtype,
                    "pair": label,
                    "r": float(r),
                    "p": float(p),
                    "n": int(len(g)),
                }
            )
    return bm, pd.DataFrame(corr_rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    frames = []
    for ckpt in CKPTS:
        frames.append(load_one(ckpt))
    raw_all = pd.concat(frames, ignore_index=True)

    # Derive sequences from frame_num gaps within subtype
    p8a_only = raw_all[raw_all["ckpt"] == "P8A"].copy()
    seq_p8a = derive_sequence_groups(p8a_only)
    # Build mapping: (subtype, frame_num, seq_num) -> derived_video_id
    key_cols = ["subtype", "frame_num", "seq_num"]
    mapping = seq_p8a[key_cols + ["derived_video_id", "seq_group_idx"]]
    all_df = raw_all.merge(mapping, on=key_cols, how="left")

    # ---- Step 1 + 2
    per_v = per_video_per_ckpt(all_df)
    per_v.to_csv(OUT_DIR / "per_video_per_ckpt_catch.csv", index=False)

    # ---- Step 3
    bm = bimodality_table(per_v)
    bm.to_csv(OUT_DIR / "per_video_bimodality.csv", index=False)
    bm_sum = bimodality_summary(per_v)
    bm_sum.to_csv(OUT_DIR / "per_video_bimodality_summary.csv", index=False)

    # ---- Step 4
    cross = cross_ckpt_table(per_v)
    cross.to_csv(OUT_DIR / "per_video_cross_ckpt.csv", index=False)
    corrs = cross_ckpt_correlations(cross)
    with open(OUT_DIR / "per_video_cross_ckpt_correlations.json", "w") as f:
        json.dump(corrs, f, indent=2)

    high_dis = high_disagreement_videos(cross, all_df)
    high_dis.to_csv(OUT_DIR / "high_disagreement_videos.csv", index=False)

    # ---- Step 5
    sub_bm, sub_corr = subtype_stratified_summaries(per_v, cross)
    sub_bm.to_csv(OUT_DIR / "per_subtype_bimodality_summary.csv", index=False)
    sub_corr.to_csv(
        OUT_DIR / "per_subtype_cross_ckpt_correlations.csv", index=False
    )

    # ---- Sequence-grouping audit (transparent table)
    seq_audit_rows = []
    for (subtype, idx), g in all_df[all_df["ckpt"] == "P8A"].groupby(
        ["subtype", "seq_group_idx"]
    ):
        seq_audit_rows.append(
            {
                "subtype": subtype,
                "seq_group_idx": int(idx),
                "n_frames": len(g),
                "frame_num_min": int(g["frame_num"].min()),
                "frame_num_max": int(g["frame_num"].max()),
                "seq_num_min": int(g["seq_num"].min()),
                "seq_num_max": int(g["seq_num"].max()),
                "derived_video_id": g["derived_video_id"].iloc[0],
            }
        )
    pd.DataFrame(seq_audit_rows).to_csv(
        OUT_DIR / "sequence_grouping_audit.csv", index=False
    )

    # Console echo
    print("== Bimodality summary ==")
    print(bm_sum.to_string(index=False))
    print()
    print("== Cross-ckpt video-level correlations ==")
    print(json.dumps(corrs, indent=2))
    print()
    print("== Subtype-stratified bimodality ==")
    print(sub_bm.to_string(index=False))
    print()
    print("== Subtype cross-ckpt correlations ==")
    print(sub_corr.to_string(index=False))
    print()
    print("== High-disagreement videos: distinct video count =", high_dis["video_id"].nunique() if len(high_dis) else 0)

    return per_v, bm_sum, cross, corrs, sub_bm, sub_corr, high_dis


if __name__ == "__main__":
    main()
