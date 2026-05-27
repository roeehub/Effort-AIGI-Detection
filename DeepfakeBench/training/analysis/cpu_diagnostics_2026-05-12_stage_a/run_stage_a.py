"""Stage A: three CPU probes on existing per-frame contract data.

Probes:
  1. Per-IQ-bin policy probe: at what IQ thresholds does T5C step3500
     strictly dominate P8A on aggregate FPR + recall?
  2. P8A + T5C ensemble policy probe: best min/max/mean/per-IQ-routing.
  3. Per-frame disagreement audit: where do P8A and T5C disagree by > 0.5?
     Do disagreements concentrate on chronic_6 / Roy_D?

Inputs:
  - 27 per-frame contract reports (9 suites x 3 ckpts) under _scorecard_reports/
  - IQ atlas at analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet

Outputs:
  - outputs/per_iq_bin_policy.csv
  - outputs/ensemble_policy_grid.csv
  - outputs/disagreement_audit.csv
  - STAGE_A_FACTS.md
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPORTS_DIR = THIS_DIR / "_scorecard_reports"
OUT_DIR = THIS_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ATLAS_PARQUET = THIS_DIR.parent / "iq_data_atlas_2026-05-08" / "outputs" / "per_frame.parquet"

CKPT_TAGS = {
    "P8A": "p8a_reference_step5000",
    "T5C_step3500": "t5c_periodic_step3500",
    "T3_S1_step1500": "t3_slot1_periodic_step1500",
}

SUITES = [
    "teams_real_all_dev",
    "teams_real_poor_quality_dev",
    "teams_real_lighting_extreme_dev",
    "teams_real_all_lockbox",
    "teams_real_dor_dev",
    "teams_fake_all_dev",
    "teams_fake_all_lockbox",
    "visomaster_enhanced_macro_dev",
    "deeplive_enhanced_dev",
]

REAL_SUITES = {s for s in SUITES if "fake" not in s}
FAKE_SUITES = set(SUITES) - REAL_SUITES


def load_unified_frames() -> pd.DataFrame:
    """Build per-frame matrix with score columns for each ckpt, stacked across all suites."""
    per_suite = []
    for suite in SUITES:
        suite_df = None
        for ckpt_label, ckpt_tag in CKPT_TAGS.items():
            path = REPORTS_DIR / f"{suite}_{ckpt_tag}_frames_report.csv"
            df = pd.read_csv(path)
            df = df.rename(columns={"frame_prob": ckpt_label})
            if suite_df is None:
                suite_df = df[["frame_path", "label", "video_id", "group_key", "family_key", ckpt_label]].copy()
            else:
                add = df[["frame_path", ckpt_label]].drop_duplicates("frame_path")
                suite_df = suite_df.merge(add, on="frame_path", how="outer")
        suite_df["suite"] = suite
        per_suite.append(suite_df)
    out = pd.concat(per_suite, ignore_index=True)
    cols = ["suite", "frame_path", "label", "video_id", "group_key", "family_key"] + list(CKPT_TAGS.keys())
    out = out[cols]
    return out


def attach_iq(df: pd.DataFrame) -> pd.DataFrame:
    """Join with IQ atlas on frame_path."""
    atlas = pd.read_parquet(ATLAS_PARQUET)
    cols = ["frame_path", "min_dim", "max_dim", "lap_var", "luma_mean", "luma_std",
            "saturation_mean", "color_a_dev", "color_b_dev", "edge_mag", "skin_frac", "contrast_l"]
    atlas = atlas[cols].drop_duplicates("frame_path", keep="first")
    merged = df.merge(atlas, on="frame_path", how="left")
    return merged


def add_identity_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Extract identity prefix from filename + add chronic_6 / dor flags."""
    chronic_6 = ["Roy_D", "PC_Generator", "bla_bla_chow",
                 "Md_noyn_Sharker", "dor_shkedi", "healthy_dor"]
    df["is_chronic_6"] = df["frame_path"].fillna("").str.contains(
        "|".join(chronic_6), case=False, regex=True).astype(int)
    df["is_roy_d"] = df["frame_path"].fillna("").str.contains("Roy_D", case=True).astype(int)
    df["is_dor"] = df["frame_path"].fillna("").str.contains("dor", case=False).astype(int)
    return df


# ---- Probe 1: Per-IQ-bin policy ----

def probe_per_iq_bin(df: pd.DataFrame) -> pd.DataFrame:
    """For each IQ axis × quartile × ckpt × τ, compute FPR on reals and recall on fakes."""
    axes = ["min_dim", "lap_var", "color_a_dev", "saturation_mean", "luma_mean"]
    taus = [0.5, 0.7, 0.9]
    rows = []
    # Only use reals from teams_real_all_dev for calibration (matches contract policy)
    # and lockbox + dor for OOD test
    cohort_groups = [
        ("dev_real", df[df["suite"].isin(["teams_real_all_dev", "teams_real_poor_quality_dev",
                                          "teams_real_lighting_extreme_dev"])]),
        ("lockbox_real", df[df["suite"] == "teams_real_all_lockbox"]),
        ("dor_real_dev", df[df["suite"] == "teams_real_dor_dev"]),
        ("teams_fake_dev", df[df["suite"] == "teams_fake_all_dev"]),
        ("lockbox_fake", df[df["suite"] == "teams_fake_all_lockbox"]),
        ("viso_enh", df[df["suite"] == "visomaster_enhanced_macro_dev"]),
        ("deeplive_enh", df[df["suite"] == "deeplive_enhanced_dev"]),
    ]
    for axis in axes:
        for cohort_name, cohort_df in cohort_groups:
            valid = cohort_df.dropna(subset=[axis])
            if len(valid) == 0:
                continue
            # Quartile thresholds computed PER axis using dev_real cohort baseline
            base_axis = df[df["suite"] == "teams_real_all_dev"][axis].dropna()
            q1, q2, q3 = base_axis.quantile([0.25, 0.50, 0.75]).tolist()
            for qlabel, mask in [
                ("Q1", valid[axis] <= q1),
                ("Q2", (valid[axis] > q1) & (valid[axis] <= q2)),
                ("Q3", (valid[axis] > q2) & (valid[axis] <= q3)),
                ("Q4", valid[axis] > q3),
            ]:
                sub = valid[mask]
                n = len(sub)
                if n == 0:
                    continue
                for ckpt in CKPT_TAGS:
                    for tau in taus:
                        positive_rate = (sub[ckpt] >= tau).mean()
                        rows.append(dict(axis=axis, cohort=cohort_name, qbin=qlabel,
                                        q_low=valid[axis].min(), q_high=valid[axis].max(),
                                        n=n, ckpt=ckpt, tau=tau,
                                        positive_rate=positive_rate))
    return pd.DataFrame(rows)


# ---- Probe 2: Ensemble policy ----

def probe_ensemble_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Score ensemble policies at multiple τ. Real cohorts → FPR; fake → recall."""
    rows = []
    # Compute ensemble columns
    df = df.copy()
    df["min_P8A_T5C"] = df[["P8A", "T5C_step3500"]].min(axis=1)
    df["max_P8A_T5C"] = df[["P8A", "T5C_step3500"]].max(axis=1)
    df["mean_P8A_T5C"] = df[["P8A", "T5C_step3500"]].mean(axis=1)
    # IQ-gated routing: if min_dim < 200 use P8A (safer at low IQ), else T5C
    df["routed_min_dim"] = np.where(df["min_dim"] < 200, df["P8A"], df["T5C_step3500"])
    # Lap-var-gated: low lap_var → P8A
    df["routed_lap_var"] = np.where(df["lap_var"] < 100, df["P8A"], df["T5C_step3500"])
    # Chronic-gated: chronic identities → P8A
    df["routed_chronic"] = np.where(df["is_chronic_6"] == 1, df["P8A"], df["T5C_step3500"])
    # Combined gate: chronic OR low IQ → P8A
    df["routed_chronic_or_lowiq"] = np.where(
        (df["is_chronic_6"] == 1) | (df["min_dim"] < 200) | (df["lap_var"] < 100),
        df["P8A"], df["T5C_step3500"])

    policies = ["P8A", "T5C_step3500", "T3_S1_step1500",
                "min_P8A_T5C", "max_P8A_T5C", "mean_P8A_T5C",
                "routed_min_dim", "routed_lap_var", "routed_chronic",
                "routed_chronic_or_lowiq"]

    cohorts = {
        "teams_real_all_dev": "real",
        "teams_real_poor_quality_dev": "real",
        "teams_real_lighting_extreme_dev": "real",
        "teams_real_all_lockbox": "real",
        "teams_real_dor_dev": "real",
        "teams_fake_all_dev": "fake",
        "teams_fake_all_lockbox": "fake",
        "visomaster_enhanced_macro_dev": "fake",
        "deeplive_enhanced_dev": "fake",
    }
    for tau in [0.5, 0.7, 0.85, 0.9, 0.95]:
        for suite, role in cohorts.items():
            sub = df[df["suite"] == suite]
            n = len(sub)
            if n == 0:
                continue
            for policy in policies:
                pos = (sub[policy] >= tau).sum()
                rows.append(dict(suite=suite, role=role, tau=tau, policy=policy,
                                n=n, n_positive=int(pos), rate=pos/n))
    return pd.DataFrame(rows)


# ---- Probe 3: Disagreement audit ----

def probe_disagreement(df: pd.DataFrame) -> dict:
    """Identify frames where |P8A - T5C| > 0.5. Characterize them."""
    df = df.copy()
    df["disagree_magnitude"] = (df["P8A"] - df["T5C_step3500"]).abs()
    big = df[df["disagree_magnitude"] > 0.5].copy()
    big["direction"] = np.where(big["P8A"] > big["T5C_step3500"], "P8A>T5C", "T5C>P8A")

    n_total = len(df)
    n_big = len(big)
    summary = {
        "total_frames": n_total,
        "big_disagreement_frames": n_big,
        "big_disagreement_pct": 100 * n_big / n_total if n_total else 0,
    }
    # By cohort
    cohort_split = big.groupby(["suite", "direction"]).size().unstack(fill_value=0)
    # By chronic flag
    chronic_split = big.groupby(["is_chronic_6", "direction"]).size().unstack(fill_value=0)
    # By Roy_D
    royd_split = big.groupby(["is_roy_d", "direction"]).size().unstack(fill_value=0)
    return summary, cohort_split, chronic_split, royd_split, big


def main():
    print("=== Loading frame reports ===")
    df = load_unified_frames()
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Suites: {df['suite'].value_counts().to_dict()}")

    print("\n=== Attaching IQ atlas ===")
    df = attach_iq(df)
    print(f"  IQ atlas join: {df['min_dim'].notna().sum()} / {len(df)} have min_dim")

    print("\n=== Adding identity flags ===")
    df = add_identity_cohort(df)
    print(f"  chronic_6 flagged: {df['is_chronic_6'].sum()} / {len(df)}")
    print(f"  Roy_D flagged: {df['is_roy_d'].sum()} / {len(df)}")

    print("\n=== Probe 1: Per-IQ-bin policy ===")
    p1 = probe_per_iq_bin(df)
    p1.to_csv(OUT_DIR / "per_iq_bin_policy.csv", index=False)
    print(f"  Wrote {len(p1)} rows")

    print("\n=== Probe 2: Ensemble policy ===")
    p2 = probe_ensemble_policy(df)
    p2.to_csv(OUT_DIR / "ensemble_policy_grid.csv", index=False)
    print(f"  Wrote {len(p2)} rows")

    print("\n=== Probe 3: Disagreement audit ===")
    summary, cohort_split, chronic_split, royd_split, big_df = probe_disagreement(df)
    big_df.to_csv(OUT_DIR / "disagreement_frames.csv", index=False)
    print(f"  Summary: {summary}")
    print(f"  Cohort split:\n{cohort_split}")
    print(f"  Chronic split (0=non-chronic, 1=chronic_6):\n{chronic_split}")
    print(f"  Roy_D split:\n{royd_split}")

    # Save unified data
    df.to_csv(OUT_DIR / "unified_frame_matrix.csv", index=False)
    print(f"\n=== Saved unified matrix at {OUT_DIR / 'unified_frame_matrix.csv'} ===")


if __name__ == "__main__":
    main()
