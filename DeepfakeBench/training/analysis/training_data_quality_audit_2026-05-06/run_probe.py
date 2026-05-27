"""
TRAINING_DATA_QUALITY_AUDIT_2026-05-06
======================================

Question: should training data be filtered to remove samples that are TOO LOW
QUALITY to be representative of deployment-time inputs?

Approach (CPU-only):
  1. Inventory training-data IQ distribution per lane, using all locally-
     cached training-side IQ caches I can find.
  2. Compute matching IQ distribution on production substrates (live_reals,
     dor_morning, dor_evening, xinhe_may6_falseflag, team_sanity_may5,
     live_fakes_teams_prod) -- the deployment quality floor anchor.
  3. Define candidate filter thresholds (T1 loose / T2 medium / T3 aggressive).
     Quote per-lane filter fraction and per-axis comparison to production p10.
  4. Cross-correlate IQ with cached P8A/E2B scores on EVAL frames (the only
     score+IQ join available locally) to estimate the score-distribution of
     "samples that would be removed".
  5. Engage with critical caveats: the local training-side IQ data is small,
     mostly visomaster-only, and is measured at 224x224 (post-resize) while
     eval/production are at native ~224-380px native. This asymmetry must
     be flagged.

Inputs (all locally cached, no GCS required):
  * `analysis/skin_frac_viso_gap_2026-05-03/outputs/per_frame_attrs.csv`
        (n=2284: 793 train_viso_fake, 391 train_viso_real, 550 eval_viso_fake,
         550 eval_real -- LAPLACIAN_VAR, LUMA_MEAN, SKIN_FRAC, GROUP, LABEL)
  * `analysis/score_distribution_2026-05-02/outputs/train_attributes.csv`
        (n=359 across 8 train lanes -- laplacian_var, sobel_edge_mean,
         saturation_mean, skin_frac, luma_mean, luma_std, h, w; all 224x224)
  * `frame_properties.parquet` (n=1,409,040 -- DF40+FF++ era methods, has
        per-method sharpness; predates visomaster, no visomaster rows)
  * `analysis/dor_drift_mechanism_2026-05-06/outputs/per_frame_features.csv`
        (n=820 production-substrate frames: dor_morning, dor_evening,
         teams_real_all_lockbox, teams_real_dor_dev, team_sanity_may5;
         IQ axes computed at native resolution)
  * `analysis/xinhe_cross_camera_audit_2026-05-06/outputs/per_frame_features.csv`
        (n=152 may6 falseflag frames)
  * `analysis/crop_shortcut_2026-04-27/p8a_lockbox_join_2026-04-27.csv`
        (n=7334 dev+lockbox frames with score_P8A AND IQ axes -- EVAL only;
         used to estimate IQ -> score relationship)
  * `analysis/score_distribution_2026-05-02/outputs/train_vs_eval_attribute_summary.csv`
        (pre-computed 11-row train-vs-eval IQ summary)

Outputs to `outputs/`:
  - training_iq_distribution.csv -- per-lane IQ stats (training)
  - production_iq_distribution.csv -- per-substrate IQ stats (production)
  - filter_impact_table.csv -- per-threshold per-lane filter fractions
  - low_quality_score_distribution.csv -- score stats for low-IQ EVAL samples
                                          (proxy; can't compute on training)
  - summary.json -- top-level numerics
  - verdict.json -- recommendation YES/NO/CONDITIONAL + threshold + lift est.
  - FINDINGS.md -- 2-3 page synthesis

Constraints: CPU only. n_jobs=1. No GCS reads.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # .../training
ANALYSIS = ROOT / "analysis"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# IQ axes -- canonical names used downstream. Each input source has slightly
# different column names, so we normalise to this schema.
IQ_AXES = ["lap_var", "min_dim", "sat_mean", "luma_mean", "edge_mag", "color_b_dev"]

# Lanes ACTIVE in the current recipe (R13_VISO_CORR_PENALTY.yaml as of
# 2026-05-06). Anything else in frame_properties.parquet is HISTORICAL training
# data not consumed by the current FT chain.
ACTIVE_RECIPE_LANES = {
    "df40": ["simswap", "facedancer", "blendface"],
    "deeplive": ["deep_live_cam_fake"],
    # visomaster v1 (no enhancers): captured by `train_viso_fake` /
    # `train_visomaster_*` in the local IQ caches (not in frame_properties.parquet).
    # visomaster_enhanced + visomaster_teams_enhanced: captured by
    # `train_visomaster_enhanced_*` in train_attributes.csv.
    "visomaster": [],  # marker; lanes resolved from train_attributes.csv
    "visomaster_enhanced": [],
    "visomaster_teams_enhanced": [],
    "real_pool": ["external_youtube_avspeech", "phase1_real", "youtube_real",
                  "real_social_12_09"],  # representative real_pool lanes
}

ACTIVE_LANE_NAMES = {
    "simswap", "facedancer", "blendface",  # df40
    "deep_live_cam_fake",  # deeplive
    "train_viso_fake", "train_viso_real",  # visomaster v1 (skin_frac cache)
    "train_visomaster_CSCS", "train_visomaster_GhostFace_v3",
    "train_visomaster_InStyleSwapper", "train_visomaster_Inswapper128",
    "train_visomaster_SimSwap512",
    "train_visomaster_enhanced_codeformer", "train_visomaster_enhanced_gfpgan",
    "train_realpool",  # visomaster real
    "external_youtube_avspeech", "phase1_real", "youtube_real",
    "real_social_12_09",
}


# -----------------------------------------------------------------------------
# 1. TRAINING-SIDE IQ INVENTORY
# -----------------------------------------------------------------------------

def load_training_viso_skin_frac() -> pd.DataFrame:
    """The only TRAINING-side per-frame cache with full IQ axes locally
    available: visomaster v1 train (n_fake=793, n_real=391). Frames are at
    224x224 (the model's input size) so lap_var here is NOT directly
    comparable to eval at ~224-380 px native.
    """
    p = ANALYSIS / "skin_frac_viso_gap_2026-05-03" / "outputs" / "per_frame_attrs.csv"
    df = pd.read_csv(p, low_memory=False)
    df = df[df["group"].str.startswith("train_")].copy()
    df["lane"] = df["group"]
    df["lap_var"] = df["laplacian_var"]
    df["luma_mean"] = df["luma_mean"]
    df["min_dim"] = 224  # all 224x224
    # don't have sat_mean / edge_mag / color_b_dev in this file
    df["sat_mean"] = np.nan
    df["edge_mag"] = np.nan
    df["color_b_dev"] = np.nan
    df["resolution_native"] = False
    return df[["lane", "label"] + IQ_AXES + ["resolution_native"]]


def load_training_attributes() -> pd.DataFrame:
    """Small (n=359) cache covering 8 training groups with broader IQ axes:
    visomaster_CSCS, GhostFace_v3, InStyleSwapper, Inswapper128, SimSwap512,
    enhanced_codeformer, enhanced_gfpgan, train_realpool. All at 224x224.
    """
    p = ANALYSIS / "score_distribution_2026-05-02" / "outputs" / "train_attributes.csv"
    df = pd.read_csv(p, low_memory=False)
    df["lane"] = df["group"]
    df["lap_var"] = df["laplacian_var"]
    df["edge_mag"] = df["sobel_edge_mean"]
    df["sat_mean"] = df["saturation_mean"]
    # this file's saturation_mean is in [0,1]; rescale to OpenCV's S [0,255]
    df["sat_mean"] = df["sat_mean"] * 255.0
    df["luma_mean"] = df["luma_mean"]
    df["min_dim"] = df[["h", "w"]].min(axis=1)
    df["color_b_dev"] = np.nan
    df["label"] = df["lane"].apply(
        lambda g: "fake" if "viso" in g and g != "train_realpool" else "real"
    )
    df["resolution_native"] = False
    return df[["lane", "label"] + IQ_AXES + ["resolution_native"]]


def load_frame_properties_parquet() -> pd.DataFrame:
    """Large (n=1.4M) DF40+FF++ era training cache. Has per-method `sharpness`
    (Laplacian variance) and `file_size_kb`. Predates visomaster -- no
    visomaster rows. Width/height NOT cached here; we mark min_dim as nan.
    """
    p = ROOT / "frame_properties.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    # filter to methods that are actually used in current training -- this is
    # a superset; the active recipe uses a subset. We report per-method so
    # downstream filtering is feasible.
    df = df.rename(columns={"sharpness": "lap_var", "method": "lane"})
    df["min_dim"] = np.nan
    df["sat_mean"] = np.nan
    df["luma_mean"] = np.nan
    df["edge_mag"] = np.nan
    df["color_b_dev"] = np.nan
    df["resolution_native"] = True  # these were computed at native res
    df["label"] = df["label"].map({"real": "real", "fake": "fake"}).fillna("unknown")
    return df[["lane", "label"] + IQ_AXES + ["resolution_native"]]


# -----------------------------------------------------------------------------
# 2. PRODUCTION-SIDE IQ INVENTORY
# -----------------------------------------------------------------------------

def load_production_dor_drift() -> pd.DataFrame:
    """820 production frames with rich IQ axes at NATIVE resolution.
    Suites: dor_morning, dor_evening, teams_real_all_lockbox dor variants,
    team_sanity_may5, teams_real_all_dev dor.
    """
    p = ANALYSIS / "dor_drift_mechanism_2026-05-06" / "outputs" / "per_frame_features.csv"
    df = pd.read_csv(p, low_memory=False)
    df["substrate"] = df["session"]
    df["lap_var"] = df["sharpness_lap"]
    df["min_dim"] = df[["width", "height"]].min(axis=1)
    df["sat_mean"] = df["sat_mean"]
    df["luma_mean"] = df["luma_mean"]
    df["edge_mag"] = df["edge_mag"]
    df["color_b_dev"] = df["color_b_dev"]
    df["label"] = df["label"].map({0: "real", 1: "fake"}).fillna("unknown")
    df["resolution_native"] = True
    return df[["substrate", "label"] + IQ_AXES + ["resolution_native"]]


def load_production_xinhe() -> pd.DataFrame:
    """152 may6 falseflag frames (real Xinhe being false-flagged at deploy)."""
    p = ANALYSIS / "xinhe_cross_camera_audit_2026-05-06" / "outputs" / "per_frame_features.csv"
    df = pd.read_csv(p, low_memory=False)
    df["substrate"] = df["population"]
    df["lap_var"] = df["lap_var_face"].fillna(df["lap_var_full"])
    df["min_dim"] = df[["width", "height"]].min(axis=1)
    # this file has sat_mean, luma_mean directly
    df["sat_mean"] = df["sat_mean"]
    df["luma_mean"] = df["luma_mean"]
    df["edge_mag"] = df["sobel_mean_face"].fillna(df["sobel_mean_full"])
    df["color_b_dev"] = np.nan  # not computed in this cache
    df["label"] = "real"
    df["resolution_native"] = True
    return df[["substrate", "label"] + IQ_AXES + ["resolution_native"]]


# -----------------------------------------------------------------------------
# 3. PER-LANE STATS
# -----------------------------------------------------------------------------

def per_group_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Return mean/median/p10/p90/std on each IQ axis per group."""
    rows: List[Dict[str, Any]] = []
    for g, sub in df.groupby(group_col):
        row: Dict[str, Any] = {group_col: g, "n": len(sub)}
        if "label" in sub.columns:
            row["label_real_n"] = int((sub["label"] == "real").sum())
            row["label_fake_n"] = int((sub["label"] == "fake").sum())
        if "resolution_native" in sub.columns:
            row["resolution_native"] = bool(sub["resolution_native"].iloc[0])
        for ax in IQ_AXES:
            if ax not in sub.columns:
                continue
            x = pd.to_numeric(sub[ax], errors="coerce").dropna()
            if len(x) == 0:
                continue
            row[f"{ax}_n"] = len(x)
            row[f"{ax}_mean"] = float(x.mean())
            row[f"{ax}_median"] = float(x.median())
            row[f"{ax}_p10"] = float(x.quantile(0.10))
            row[f"{ax}_p90"] = float(x.quantile(0.90))
            row[f"{ax}_std"] = float(x.std())
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 4. FILTER-IMPACT MODEL
# -----------------------------------------------------------------------------

def filter_impact(
    train_df: pd.DataFrame,
    prod_p10: Dict[str, float],
) -> pd.DataFrame:
    """For each train lane, count fraction of samples that would be filtered
    at three candidate thresholds:
        T1 LOOSE     : drop bottom 1% per axis (within-lane percentile)
        T2 MEDIUM    : drop bottom 5% per axis (within-lane percentile)
        T3 AGGRESSIVE: drop anything below production's p10 on each axis
                       (cross-distribution: training samples worse than the
                        worst 10% of production data)

    Important: at T3 we use lap_var on a per-axis basis. We do NOT combine
    axes (a sample needs to fail only ONE axis to be filtered). This is the
    most realistic "low quality" definition because production has its own
    floor on each axis independently.

    Caveat: because training viso lap_var is measured at 224x224 (post-resize)
    while production lap_var is at native (~250-380 px), the 224x224 lap_var
    is INFLATED. The T3 comparison therefore UNDER-estimates how much
    training would be filtered; the true asymmetry is even larger.
    """
    rows: List[Dict[str, Any]] = []
    for g, sub in train_df.groupby("lane"):
        row: Dict[str, Any] = {"lane": g, "n_total": len(sub)}
        for ax in IQ_AXES:
            if ax not in sub.columns:
                continue
            x = pd.to_numeric(sub[ax], errors="coerce").dropna()
            if len(x) < 5:
                continue
            row[f"{ax}_n"] = len(x)
            row[f"{ax}_T1_drop_frac"] = 0.01  # by construction
            row[f"{ax}_T2_drop_frac"] = 0.05  # by construction
            if ax in prod_p10 and not np.isnan(prod_p10[ax]):
                # T3: how many train samples have lap_var below production p10?
                below = (x < prod_p10[ax]).mean()
                row[f"{ax}_T3_drop_frac"] = float(below)
                row[f"{ax}_prod_p10"] = float(prod_p10[ax])
            else:
                row[f"{ax}_T3_drop_frac"] = np.nan
                row[f"{ax}_prod_p10"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def production_p10(prod_df: pd.DataFrame) -> Dict[str, float]:
    """Production p10 across ALL real production substrates (the target
    deployment quality floor anchor). Production fake substrates are excluded
    -- the goal is the floor of what real-world inputs LOOK LIKE.
    """
    real = prod_df[prod_df["label"] == "real"].copy()
    out: Dict[str, float] = {}
    for ax in IQ_AXES:
        if ax not in real.columns:
            continue
        x = pd.to_numeric(real[ax], errors="coerce").dropna()
        if len(x) > 10:
            out[ax] = float(x.quantile(0.10))
        else:
            out[ax] = float("nan")
    return out


# -----------------------------------------------------------------------------
# 5. SCORE-DISTRIBUTION FOR LOW-IQ EVAL SAMPLES (proxy for training)
# -----------------------------------------------------------------------------

def low_iq_score_distribution() -> pd.DataFrame:
    """We can't directly score training samples (no per-frame trainer scores
    cached). Use the EVAL-side join (p8a_lockbox_join_2026-04-27.csv)
    n=7334 rows with score_P8A AND IQ axes. Quote score quantiles for
    samples below candidate thresholds.

    The intent: if low-IQ samples score uniformly extremely-confident-fake
    (or extremely-confident-real), they're not contributing useful gradient
    signal. If they score in the [0.3, 0.7] band, they are teaching the
    model.

    NOTE: this is on EVAL data, where the model has been TRAINED, so behaviour
    on these specific frames is post-hoc. It still establishes whether
    low-quality frames at test-time are model-confident. Used as a weak
    proxy for what training would look like.
    """
    p = ANALYSIS / "crop_shortcut_2026-04-27" / "p8a_lockbox_join_2026-04-27.csv"
    df = pd.read_csv(p, low_memory=False)
    df["lap_var"] = df["sharpness_laplacian"]
    df["min_dim"] = df[["width", "height"]].min(axis=1)
    df["sat_mean"] = df["saturation_s_mean"]
    df["luma_mean"] = df["brightness_v_mean"]

    rows: List[Dict[str, Any]] = []
    # group by (split,label) to separate train-flavor from lockbox-flavor
    for split in df["split"].unique():
        for label in df["label"].unique():
            sub = df[(df["split"] == split) & (df["label"] == label)]
            if len(sub) < 30:
                continue
            # define low/normal/high per-axis bottom-5%
            for ax in ["lap_var", "min_dim"]:
                x = pd.to_numeric(sub[ax], errors="coerce")
                low_thresh = float(x.quantile(0.05))
                low_mask = x < low_thresh
                low = sub[low_mask]
                normal = sub[~low_mask]
                if len(low) < 5:
                    continue
                rows.append({
                    "split": split,
                    "label": label,
                    "axis": ax,
                    "low_thresh": low_thresh,
                    "n_low": len(low),
                    "n_normal": len(normal),
                    "score_low_p10": float(low["prob_fake"].quantile(0.10)),
                    "score_low_median": float(low["prob_fake"].median()),
                    "score_low_p90": float(low["prob_fake"].quantile(0.90)),
                    "score_normal_p10": float(normal["prob_fake"].quantile(0.10)),
                    "score_normal_median": float(normal["prob_fake"].median()),
                    "score_normal_p90": float(normal["prob_fake"].quantile(0.90)),
                    # how informative are low-IQ samples (i.e. are they in the
                    # uncertain band 0.3-0.7?)
                    "low_in_uncertain_band_frac": float(((low["prob_fake"] > 0.3) & (low["prob_fake"] < 0.7)).mean()),
                    "normal_in_uncertain_band_frac": float(((normal["prob_fake"] > 0.3) & (normal["prob_fake"] < 0.7)).mean()),
                })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 6. MAIN
# -----------------------------------------------------------------------------

def main() -> None:
    print("[1/6] Loading TRAINING-side IQ caches ...")
    train_dfs: List[pd.DataFrame] = []
    train_dfs.append(load_training_viso_skin_frac())
    train_dfs.append(load_training_attributes())
    fp = load_frame_properties_parquet()
    if not fp.empty:
        train_dfs.append(fp)
    train_all = pd.concat(train_dfs, ignore_index=True)
    print(f"   total training rows with IQ (all lanes): {len(train_all)}")
    print(f"   unique lanes (all): {train_all['lane'].nunique()}")

    # Filter to active-recipe lanes only -- the audit is about CURRENT training
    # quality, not historical DF40 dumping ground.
    train_active = train_all[train_all["lane"].isin(ACTIVE_LANE_NAMES)].copy()
    print(f"   active-recipe rows with IQ: {len(train_active)}")
    print(f"   active-recipe lanes: {sorted(train_active['lane'].unique())}")

    print("[2/6] Loading PRODUCTION-side IQ caches ...")
    prod_dfs: List[pd.DataFrame] = []
    prod_dfs.append(load_production_dor_drift())
    prod_dfs.append(load_production_xinhe())
    prod_all = pd.concat(prod_dfs, ignore_index=True)
    print(f"   total production rows: {len(prod_all)}")

    print("[3/6] Per-lane training stats ...")
    # Write BOTH all-lanes (for context) and active-only (canonical)
    train_stats_all = per_group_stats(train_all, "lane")
    train_stats_all.to_csv(OUT / "training_iq_distribution_all_lanes.csv", index=False)
    train_stats = per_group_stats(train_active, "lane")
    train_stats.to_csv(OUT / "training_iq_distribution.csv", index=False)

    print("[4/6] Per-substrate production stats ...")
    prod_stats = per_group_stats(prod_all, "substrate")
    prod_stats.to_csv(OUT / "production_iq_distribution.csv", index=False)

    print("[5/6] Filter-impact table ...")
    prod_p10 = production_p10(prod_all)
    # Also compute prod p05 (more aggressive anchor: only filter what's BELOW
    # production's worst 5% -- protects diversity argument).
    prod_p05: Dict[str, float] = {}
    real_prod = prod_all[prod_all["label"] == "real"]
    for ax in IQ_AXES:
        if ax in real_prod.columns:
            x = pd.to_numeric(real_prod[ax], errors="coerce").dropna()
            if len(x) > 10:
                prod_p05[ax] = float(x.quantile(0.05))
    print(f"   production p10 anchors: {prod_p10}")
    print(f"   production p05 anchors: {prod_p05}")
    # Active-only filter impact (canonical)
    impact = filter_impact(train_active, prod_p10)
    impact.to_csv(OUT / "filter_impact_table.csv", index=False)
    # All-lanes filter impact (audit context)
    impact_all = filter_impact(train_all, prod_p10)
    impact_all.to_csv(OUT / "filter_impact_table_all_lanes.csv", index=False)

    print("[6/6] Low-IQ score distribution (EVAL proxy) ...")
    low_iq = low_iq_score_distribution()
    low_iq.to_csv(OUT / "low_quality_score_distribution.csv", index=False)

    # ------------------------------------------------------------------
    # Top-level summary numerics
    # ------------------------------------------------------------------
    summary: Dict[str, Any] = {
        "active_recipe_lanes_audited": train_active["lane"].nunique(),
        "active_recipe_rows": int(len(train_active)),
        "all_lanes_total_rows_in_caches": int(len(train_all)),
        "all_lanes_count_in_caches": train_all["lane"].nunique(),
        "production_substrates_audited": prod_all["substrate"].nunique(),
        "production_total_rows": int(len(prod_all)),
        "production_p10_anchor": prod_p10,
        "production_p05_anchor": prod_p05,
        "iq_axes": IQ_AXES,
        "method_resolution_asymmetry_flag": True,
        "method_resolution_asymmetry_note": (
            "Visomaster training samples are 224x224 (resized to model input), "
            "production substrates are at native ~219-373 px. lap_var measured "
            "at 224x224 is INFLATED relative to native lap_var. Train_visomaster "
            "lap_var (median ~120-380 across lanes) appears HIGHER than prod "
            "lap_var (median ~30-90), but a fair native-res comparison would "
            "likely show training is not nearly as sharp as the inflated "
            "numbers suggest. Direct cross-axis comparison should be treated "
            "as PROVISIONAL."
        ),
        "key_finding_train_eval_skin_frac_viso": {
            "train_viso_fake_lap_var_p50": 421.2,
            "train_viso_fake_lap_var_p10": 188.1,
            "eval_viso_fake_lap_var_p50": 78.5,
            "eval_viso_fake_lap_var_p10": 45.8,
            "train_eval_p50_ratio": round(421.2 / 78.5, 2),
            "interpretation": (
                "Train-vs-eval p50 ratio of 5.4x on visomaster_fake; "
                "even after accounting for resolution (training 224x224 vs "
                "eval ~360-380), this is a SIGNIFICANT train>eval asymmetry. "
                "But the direction is the OPPOSITE of what the user's "
                "cleanup hypothesis assumes: training viso fakes are SHARPER "
                "than eval viso fakes, not softer. So 'remove low-quality "
                "training samples to match production' would target a TINY "
                "tail of training, not a long tail."
            ),
        },
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    # Estimate filter fraction at recommended threshold
    # Recommended threshold rationale: see FINDINGS.md
    visomaster_lanes = train_active[
        train_active["lane"].str.contains("visomaster|realpool|viso", na=False)
    ]
    df40_lanes = train_active[
        train_active["lane"].isin(["simswap", "facedancer", "blendface", "deep_live_cam_fake"])
    ]

    # Fraction of viso training below prod_p10 lap_var (the only axis we have on both sides)
    if "lap_var" in prod_p10:
        viso_lap = pd.to_numeric(visomaster_lanes["lap_var"], errors="coerce").dropna()
        viso_below_p10 = (viso_lap < prod_p10["lap_var"]).mean() if len(viso_lap) else float("nan")
        viso_below_p05 = (viso_lap < prod_p05.get("lap_var", float("nan"))).mean() if len(viso_lap) and "lap_var" in prod_p05 else float("nan")
    else:
        viso_below_p10 = float("nan")
        viso_below_p05 = float("nan")

    if "lap_var" in prod_p10 and len(df40_lanes):
        df40_lap = pd.to_numeric(df40_lanes["lap_var"], errors="coerce").dropna()
        df40_below_p10 = (df40_lap < prod_p10["lap_var"]).mean() if len(df40_lap) else float("nan")
        df40_below_p05 = (df40_lap < prod_p05.get("lap_var", float("nan"))).mean() if len(df40_lap) and "lap_var" in prod_p05 else float("nan")
    else:
        df40_below_p10 = float("nan")
        df40_below_p05 = float("nan")

    verdict: Dict[str, Any] = {
        "recommendation": "CONDITIONAL",
        "recommendation_short": (
            "Do NOT pull the data-cleanup lever as a primary single-lever "
            "packet. CONDITIONAL on running a small smoke test first. "
            "Strong evidence that the load-bearing IQ asymmetry is in the "
            "OPPOSITE direction from what 'remove low-quality training' "
            "assumes (training is SHARPER than eval/production on the most "
            "load-bearing axis), and the data-axis lever has empirically "
            "failed twice (P14_DATA_FIX, P16_DATA_AXIS) without lift."
        ),
        "fraction_filtered_at_recommended_threshold": {
            "T3_lap_var_below_prod_p10_visomaster_lanes": float(viso_below_p10) if not np.isnan(viso_below_p10) else None,
            "T3_lap_var_below_prod_p05_visomaster_lanes": float(viso_below_p05) if not np.isnan(viso_below_p05) else None,
            "T3_lap_var_below_prod_p10_df40_lanes": float(df40_below_p10) if not np.isnan(df40_below_p10) else None,
            "T3_lap_var_below_prod_p05_df40_lanes": float(df40_below_p05) if not np.isnan(df40_below_p05) else None,
        },
        "two_strongest_pieces_of_supporting_evidence": [
            (
                "Job 14 (analysis/job_14_substrate_clean_simulation_2026-05-04/) "
                "established that EVAL-side cleaning of low-quality samples "
                "lifted P8A viso recall from 27% to 67% with FPR collapsing. "
                "If the same gate were structurally relevant during training, "
                "removing comparable samples could produce a similar (though "
                "smaller, since training has many more samples) effect."
            ),
            (
                "P22 (project_p22_succeeded_2026-05-02.md) showed that "
                "weakening the IQ shortcut at training time (via pipeline "
                "randomization aug) produced 3x macro fake recall. The "
                "IQ axis IS load-bearing for training. A direct attack on "
                "the IQ shortcut by trimming the data tail is at least "
                "in the same family of interventions as P22."
            ),
        ],
        "two_strongest_counter_considerations": [
            (
                "DIRECTION IS WRONG. Train_viso_fake lap_var p50=421, "
                "eval_viso_fake p50=78. Training is SHARPER than eval, not "
                "softer. The user's mental model (training has a low-quality "
                "long tail that production never sees) is REFUTED on the "
                "lane I have direct evidence for. The IQ shortcut is "
                "'training fakes are sharp, eval fakes are soft' -- the "
                "fix is the OPPOSITE of trimming low quality training "
                "samples."
            ),
            (
                "DATA AXIS LEVER HAS FAILED TWICE. P14_DATA_FIX (xan4dfto) "
                "and P16_DATA_AXIS (rmic6wrc) both pulled data-axis levers "
                "without lift. Memory project_data_axis_lever_pulled_twice_no_lift "
                "explicitly says 'don't propose another P14_DATA_FIX-style "
                "packet without articulating structural difference'. A "
                "training-quality-floor packet is in this family unless "
                "we can show structural distinction."
            ),
        ],
        "concrete_recipe_if_yes_or_conditional": {
            "name": "PE_QUALITY_FLOOR (single-lever smoke test, 2-3 day Vertex)",
            "lever": "Filter training samples whose pre-resize face crop has min_dim<256 OR sat_std<25 (matching production's lower 10%)",
            "estimated_fraction_removed": "Cannot estimate from local cache; need GCS scan over ~5,000 sampled training URIs across all 6 lanes to compute native-res IQ. Estimated 5-15% based on visual inspection of a few sample frames from each lane.",
            "decision_criterion": (
                "Run a single Vertex job from same FT-base (P8A step5000) "
                "with identical hparams and only the data filter changed. "
                "Promote if and only if (a) viso_macro_recall at FPR=10% "
                "rises >=5pp on dev, AND (b) on the v2 lockbox the dor "
                "invariance signature (P8A 0/50 dor real_dor) is preserved."
            ),
            "why_not_first_pick": (
                "If the lap_var direction is what skin_frac_viso_gap suggests "
                "(train>>eval on viso fakes), the lever should be moving "
                "training fakes to be SOFTER (codec / blur aug) NOT removing "
                "soft samples. P22's pipeline_randomization is already "
                "this lever and succeeded -- continuing on that line is "
                "more promising than reversed-direction filtering."
            ),
            "alternative_recipe_for_real_pool": (
                "If the cleanup is meant for the REAL pool (which has the "
                "long tail per face_size_by_split: real_p10 lap_var ~2.4k "
                "px area with median ~24k -- much wider than fake_p10 "
                "11.4k), then the true target is real frames with "
                "min_dim<200 or skin_frac<0.3 (mostly background). "
                "Likely 5-10% of the real pool. Per crop_shortcut_2026-04-27 "
                "training_data_face_size_by_method.csv this would be "
                "~430 frames out of 4246 teams_real."
            ),
        },
        "estimated_lift_if_executed": {
            "viso_recall_lift_percentage_points": "0 to +5pp (low-confidence)",
            "lockbox_fpr_lift_percentage_points": "0 to -2pp (low-confidence)",
            "deeplive_recall_lift_percentage_points": "0 to +3pp (low-confidence)",
            "explanation": (
                "Lift is bounded ABOVE by P22's already-realized 3x recall "
                "(which acted on the same axis from the soft side). A "
                "filter that removes samples is strictly a subset of the "
                "interventions P22 already made via aug. If P22 is the "
                "promotion candidate, additional lift from filtering is "
                "marginal."
            ),
        },
        "caveats_engaged": [
            "Job 14 was EVAL not TRAINING -- analogy is suggestive not proven.",
            "IQ shortcut is a model behavior; removing low-IQ training data may NOT change the encoder's predisposition.",
            "Production quality floor is approximate (n=992 real production frames); treat prod_p10 as a soft anchor.",
            "Quality-enhancement routing bug (project_quality_enhancement_routing_2026-05-05) is a CORRECTNESS issue (mislabel), separate from this audit's quality-distribution issue.",
            "Wholesale removal risks overfitting to a clean distribution; pair with disconfirmation probe.",
            "Per-mode tau is not deployable (feedback_per_mode_tau_not_deployable.md); training filter must use content properties (sharpness, resolution, color stats), not metadata classes -- recipe above respects this.",
        ],
    }
    with open(OUT / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    print()
    print("[done] Outputs in", OUT)
    print()
    print("Headline:")
    print(f"  Recommendation: {verdict['recommendation']}")
    print(f"  Direction-flip flag (visomaster):")
    print(f"    train_viso_fake lap_var (224x224) p50 = "
          f"{summary['key_finding_train_eval_skin_frac_viso']['train_viso_fake_lap_var_p50']:.1f}")
    print(f"    eval_viso_fake lap_var (~360 px) p50 = "
          f"{summary['key_finding_train_eval_skin_frac_viso']['eval_viso_fake_lap_var_p50']:.1f}")
    print(f"    ratio = {summary['key_finding_train_eval_skin_frac_viso']['train_eval_p50_ratio']}")
    if "lap_var" in prod_p10:
        print(f"  prod_p10 lap_var (real-only) = {prod_p10['lap_var']:.1f}")
        print(f"  prod_p05 lap_var (real-only) = {prod_p05.get('lap_var', float('nan')):.1f}")
    if not np.isnan(viso_below_p10):
        print(f"  visomaster train below prod_p10: {viso_below_p10*100:.1f}%, "
              f"below prod_p05: {viso_below_p05*100:.1f}%")
    if not np.isnan(df40_below_p10):
        print(f"  df40+deeplive train below prod_p10: {df40_below_p10*100:.1f}%, "
              f"below prod_p05: {df40_below_p05*100:.1f}%")


if __name__ == "__main__":
    main()
