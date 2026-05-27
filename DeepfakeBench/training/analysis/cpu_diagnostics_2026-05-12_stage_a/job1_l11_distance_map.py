"""CPU Job 1 — L11 distance map from P8A baseline.

For each candidate ckpt (T5C step3500, T3_SLOT1 step1500, T4_L1 step10500), compute
per-frame ||L11_ckpt - L11_P8A||₂ on the 800-frame triptych. Bin by:
  - cohort (chronic_6 vs healthy vs fake)
  - IQ axis quartile (lap_var, min_dim, color_a_dev, saturation_mean)
  - per-identity (chronic_6 sub-cohort)
  - real-vs-fake

Goal: determine where P8A and T5C's L11 representations have diverged. This drives the
choice of anchor cohort for the proposed L11-anchor-loss packet.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
CACHE = REPO / "analysis" / "iq_perlayer_probe_2026-05-08" / "_cache"
OUT_DIR = REPO / "analysis" / "cpu_diagnostics_2026-05-12_stage_a" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLED_CSV = REPO / "analysis" / "embedding_triptych_2026-04-30" / "outputs" / "triptych_p8a_slot2_slot3" / "sampled_frames.csv"
ATLAS = REPO / "analysis" / "iq_data_atlas_2026-05-08" / "outputs" / "per_frame.parquet"

CHRONIC_6 = ["Roy_D", "PC_Generator", "bla_bla_chow",
             "Md_noyn_Sharker", "dor_shkedi", "healthy_dor"]


def load_l11(label: str):
    p = CACHE / f"intermediate__{label}__layer11__n800.npz"
    blob = np.load(p)
    return blob["features"].astype(np.float32), blob["valid_idx"].astype(np.int64)


def build_panel():
    cols = ["gcs_uri", "label", "split", "identity_key", "session_id", "video_id", "method", "local_path",
            "width", "height", "is_no_face", "face_pixel_area"]
    df = pd.read_csv(SAMPLED_CSV, usecols=cols).iloc[:800].reset_index(drop=True)
    df["row_ix"] = np.arange(len(df))

    def chronic_match(s):
        if not isinstance(s, str):
            return ""
        for p in CHRONIC_6:
            if p.lower() in s.lower():
                return p
        return ""

    df["chronic_id"] = df["identity_key"].apply(chronic_match)
    df["is_chronic_6"] = (df["chronic_id"] != "").astype(int)
    df["is_lockbox"] = (df["split"] == "lockbox").astype(int)
    df["is_real"] = (df["label"] == "real").astype(int)

    # IQ join
    atlas = pd.read_parquet(ATLAS)[["frame_path", "min_dim", "lap_var", "luma_mean",
                                    "saturation_mean", "color_a_dev"]].drop_duplicates("frame_path", keep="first")
    df = df.merge(atlas, left_on="gcs_uri", right_on="frame_path", how="left")
    return df


def compute_distances(panel, ckpts):
    """For each row in panel that has valid features, compute ||L11_ckpt - L11_P8A||₂."""
    feat_dict = {}
    for ckpt in ckpts:
        feat, vidx = load_l11(ckpt)
        feat_dict[ckpt] = (feat, vidx)

    # Align all ckpts to common valid_idx
    common = set(feat_dict["P8A"][1].tolist())
    for ckpt in ckpts:
        common &= set(feat_dict[ckpt][1].tolist())
    common_arr = np.array(sorted(common))
    print(f"Common valid frames across all ckpts: {len(common_arr)}")

    distances = {}
    for ckpt in ckpts:
        feat, vidx = feat_dict[ckpt]
        # Map common_arr to position in vidx
        positions = np.searchsorted(vidx, common_arr)
        feat_sub = feat[positions]
        distances[ckpt] = feat_sub

    # Compute distances from P8A
    p8a_feat = distances["P8A"]
    out_panel = panel.set_index("row_ix").loc[common_arr].reset_index()
    for ckpt in ckpts:
        if ckpt == "P8A":
            continue
        diff = distances[ckpt] - p8a_feat
        l2 = np.linalg.norm(diff, axis=1)
        # Cosine distance
        norm_p = np.linalg.norm(p8a_feat, axis=1)
        norm_c = np.linalg.norm(distances[ckpt], axis=1)
        cos = (distances[ckpt] * p8a_feat).sum(axis=1) / (norm_p * norm_c + 1e-9)
        out_panel[f"l2_{ckpt}"] = l2
        out_panel[f"cos_{ckpt}"] = cos
        out_panel[f"cosdist_{ckpt}"] = 1.0 - cos
    return out_panel


def main():
    panel = build_panel()
    ckpts = ["P8A", "T5C_periodic_step3500", "T3_S1_step1500", "T4_L1_step10500"]
    df = compute_distances(panel, ckpts)
    df.to_csv(OUT_DIR / "l11_distance_per_frame.csv", index=False)
    print(f"\nWrote {len(df)} rows to l11_distance_per_frame.csv")

    # Summary tables
    print("\n=== L2 distance from P8A's L11 features (mean by cohort) ===")
    print(f"{'Cohort':<32} {'n':>5}  " + "  ".join(f"{c[:18]:>20}" for c in ckpts if c != "P8A"))
    cohorts = [
        ("ALL", df.index >= 0),
        ("real", df["is_real"] == 1),
        ("fake", df["is_real"] == 0),
        ("chronic_6_real", (df["is_chronic_6"] == 1) & (df["is_real"] == 1)),
        ("chronic_6_fake", (df["is_chronic_6"] == 1) & (df["is_real"] == 0)),
        ("healthy_real", (df["is_chronic_6"] == 0) & (df["is_real"] == 1)),
        ("healthy_fake", (df["is_chronic_6"] == 0) & (df["is_real"] == 0)),
        ("lockbox_real", (df["is_lockbox"] == 1) & (df["is_real"] == 1)),
        ("lockbox_fake", (df["is_lockbox"] == 1) & (df["is_real"] == 0)),
        ("dev_real", (df["is_lockbox"] == 0) & (df["is_real"] == 1)),
        ("dev_fake", (df["is_lockbox"] == 0) & (df["is_real"] == 0)),
    ]
    for cname, mask in cohorts:
        sub = df[mask]
        if len(sub) == 0:
            continue
        vals = []
        for ckpt in ckpts:
            if ckpt == "P8A":
                continue
            vals.append(f"{sub[f'l2_{ckpt}'].mean():>20.4f}")
        print(f"{cname:<32} {len(sub):>5}  " + "  ".join(vals))

    print("\n=== Cosine DISTANCE (1-cos) from P8A's L11 (median by cohort) ===")
    print(f"{'Cohort':<32} {'n':>5}  " + "  ".join(f"{c[:18]:>20}" for c in ckpts if c != "P8A"))
    for cname, mask in cohorts:
        sub = df[mask]
        if len(sub) == 0:
            continue
        vals = []
        for ckpt in ckpts:
            if ckpt == "P8A":
                continue
            vals.append(f"{sub[f'cosdist_{ckpt}'].median():>20.4f}")
        print(f"{cname:<32} {len(sub):>5}  " + "  ".join(vals))

    # Per-chronic-identity breakdown
    print("\n=== Per-chronic-identity L2 distance (T5C step3500 - P8A) ===")
    for ident in CHRONIC_6:
        sub = df[df["chronic_id"] == ident]
        sub_real = sub[sub["is_real"] == 1]
        sub_fake = sub[sub["is_real"] == 0]
        n_r = len(sub_real)
        n_f = len(sub_fake)
        d_r = sub_real["l2_T5C_periodic_step3500"].mean() if n_r else np.nan
        d_f = sub_fake["l2_T5C_periodic_step3500"].mean() if n_f else np.nan
        print(f"  {ident:<25} real_n={n_r:>3}  l2(T5C-P8A)={d_r:>7.4f}  "
              f"fake_n={n_f:>3}  l2(T5C-P8A)={d_f:>7.4f}")

    # By IQ axis quartile (on real cohort only)
    print("\n=== L2(T5C-P8A) by IQ quartile on real frames (chronic_6 + healthy) ===")
    for axis in ["lap_var", "min_dim", "color_a_dev", "saturation_mean"]:
        sub = df[(df["is_real"] == 1) & df[axis].notna()].copy()
        qs = sub[axis].quantile([0.25, 0.5, 0.75]).tolist()
        bins = pd.cut(sub[axis], bins=[-np.inf] + qs + [np.inf], labels=["Q1", "Q2", "Q3", "Q4"])
        sub["qbin"] = bins
        means = sub.groupby("qbin", observed=True)["l2_T5C_periodic_step3500"].mean()
        print(f"  {axis:<20}: " + "  ".join(f"Q{i+1}={v:.3f}" for i, v in enumerate(means)))

    # Rank-1 distance frames
    print("\n=== Top 10 frames where T5C step3500 has LARGEST L11 drift from P8A ===")
    top = df.nlargest(10, "l2_T5C_periodic_step3500")[
        ["chronic_id", "is_real", "is_lockbox", "min_dim", "lap_var", "color_a_dev",
         "l2_T5C_periodic_step3500", "cosdist_T5C_periodic_step3500"]]
    print(top.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
