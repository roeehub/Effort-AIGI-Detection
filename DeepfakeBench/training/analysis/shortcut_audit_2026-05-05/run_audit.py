"""
CPU-only shortcut audit for E2B and P8A checkpoints.

Reads frame-score CSVs, joins with tag parquets by gcs_uri==frame_path, and
characterizes how predictions co-vary with capture-condition features
(sharpness, face area, capture mode, lighting, source resolution, quality).

Outputs into analysis/shortcut_audit_2026-05-05/.
"""
from __future__ import annotations

import logging
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ------------------------------------------------------------ paths / config

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT_DIR = ROOT / "analysis/shortcut_audit_2026-05-05"
OUT_DIR.mkdir(parents=True, exist_ok=True)

E2B_DIR = ROOT / "analysis/cpu_followups_2026-05-04/raw_reports"
P8A_DIR = ROOT / "analysis/score_distribution_2026-05-02/raw_reports"

CKPTS = {
    "e2b": {"dir": E2B_DIR, "tag": "e2b_top_n_step3200"},
    "p8a": {"dir": P8A_DIR, "tag": "p8a_reference_step5000"},
}

SUITES = [
    "visomaster_enhanced_macro_dev",
    "deeplive_enhanced_dev",
    "teams_fake_all_dev",
    "teams_fake_all_lockbox",
    "teams_real_all_dev",
    "teams_real_all_lockbox",
    "teams_real_dor_dev",
    "teams_real_poor_quality_dev",
    "teams_real_lighting_extreme_dev",
]

# Suite -> kind ("fake" -> recall metric, "real" -> FPR metric)
SUITE_KIND = {
    "visomaster_enhanced_macro_dev": "fake",
    "deeplive_enhanced_dev": "fake",
    "teams_fake_all_dev": "fake",
    "teams_fake_all_lockbox": "fake",
    "teams_real_all_dev": "real",
    "teams_real_all_lockbox": "real",
    "teams_real_dor_dev": "real",
    "teams_real_poor_quality_dev": "real",
    "teams_real_lighting_extreme_dev": "real",
}

CALIB_SUITE = "teams_real_all_dev"
TAU_LEVELS = {"tau_fpr_10": 0.10, "tau_fpr_05": 0.05}

# ------------------------------------------------------------ logging

LOG_PATH = OUT_DIR / "run.log"
# fresh log
if LOG_PATH.exists():
    LOG_PATH.unlink()

logger = logging.getLogger("shortcut_audit")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_PATH)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
logger.addHandler(sh)


def report_path(ckpt: str, suite: str) -> Path:
    cfg = CKPTS[ckpt]
    return cfg["dir"] / f"{suite}_{cfg['tag']}_frames_report.csv"


# ------------------------------------------------------------ load tags

def load_tags() -> pd.DataFrame:
    full = pd.read_parquet(ROOT / "analysis/lockbox_tagging/full_tags_2026-04-27.parquet")
    lock = pd.read_parquet(ROOT / "analysis/lockbox_tagging/lockbox_tags_2026-04-27.parquet")
    keep_cols = [
        "gcs_uri", "label", "split", "identity_key", "method",
        "width", "height",
        "sharpness_laplacian", "brightness_v_mean", "saturation_s_mean",
        "face_pixel_area", "face_area_ratio",
        "clip_quality", "clip_capture_mode", "clip_lighting",
        "is_low_quality", "is_no_face", "is_likely_screen_capture",
    ]
    fcols = [c for c in keep_cols if c in full.columns]
    lcols = [c for c in keep_cols if c in lock.columns]
    full = full[fcols].copy()
    lock = lock[lcols].copy()
    combined = pd.concat([full, lock], ignore_index=True).drop_duplicates(subset=["gcs_uri"])
    logger.info("Loaded tags: full=%d lock=%d combined=%d", len(full), len(lock), len(combined))
    return combined


# ------------------------------------------------------------ stratification

def add_quartile(df: pd.DataFrame, src: str, dst: str, ref: pd.DataFrame | None = None):
    """Assign quartile labels based on the same boundaries as `ref` (or self)."""
    series = df[src]
    base = ref[src] if ref is not None else series
    base = base.dropna()
    if len(base) < 4:
        df[dst] = pd.NA
        return df
    qs = base.quantile([0.25, 0.5, 0.75]).values
    edges = [-np.inf, qs[0], qs[1], qs[2], np.inf]
    labels = ["Q1", "Q2", "Q3", "Q4"]
    df[dst] = pd.cut(series, bins=edges, labels=labels, include_lowest=True)
    return df


# ------------------------------------------------------------ metrics

def quantile_threshold(scores: np.ndarray, fpr: float) -> float:
    """tau s.t. P(score>=tau on real)=fpr."""
    if len(scores) == 0:
        return float("nan")
    # frame is positive (caught) if frame_prob >= tau; FPR = mean(real_score >= tau)
    return float(np.quantile(scores, 1.0 - fpr))


def per_axis_table(joined: pd.DataFrame, axis_col: str, taus: dict[str, float], suite_kind: str) -> pd.DataFrame:
    """Aggregate at each bucket of `axis_col`. Produces one row per (bucket, tau)."""
    rows = []
    if axis_col not in joined.columns:
        return pd.DataFrame()
    for bucket, gdf in joined.groupby(axis_col, dropna=False, observed=True):
        n_total = len(gdf)
        if n_total == 0:
            continue
        scores = gdf["frame_prob"].values
        for tau_name, tau in taus.items():
            n_caught = int((scores >= tau).sum())
            metric = n_caught / n_total
            rows.append({
                "axis": axis_col,
                "bucket": str(bucket) if not (isinstance(bucket, float) and math.isnan(bucket)) else "NA",
                "n_total": n_total,
                "n_caught": n_caught,
                "metric_value": metric,  # recall if fake suite; FPR if real suite
                "metric_kind": "recall" if suite_kind == "fake" else "fpr",
                "tau_name": tau_name,
                "tau_value": tau,
            })
    return pd.DataFrame(rows)


def correlations_for(joined: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        if feat not in joined.columns:
            continue
        sub = joined[["frame_prob", feat]].dropna()
        if len(sub) < 5:
            rows.append({
                "feature": feat,
                "pearson_r": float("nan"),
                "pearson_p": float("nan"),
                "spearman_rho": float("nan"),
                "spearman_p": float("nan"),
                "n": len(sub),
            })
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pr = stats.pearsonr(sub["frame_prob"].values, sub[feat].values)
                pearson_r, pearson_p = float(pr.statistic), float(pr.pvalue)
            except Exception:
                pearson_r = pearson_p = float("nan")
            try:
                sr = stats.spearmanr(sub["frame_prob"].values, sub[feat].values)
                spearman_rho, spearman_p = float(sr.statistic), float(sr.pvalue)
            except Exception:
                spearman_rho = spearman_p = float("nan")
        rows.append({
            "feature": feat,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
            "n": len(sub),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------ join helper

def join_report_with_tags(report: pd.DataFrame, tags: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Returns (joined_df, coverage_pct)."""
    joined = report.merge(tags, left_on="frame_path", right_on="gcs_uri", how="left")
    cov = joined["gcs_uri"].notna().mean()
    return joined, cov


# ------------------------------------------------------------ main pipeline

AXES = [
    "face_area_quartile",
    "sharpness_quartile",
    "clip_capture_mode",
    "clip_lighting",
    "source_resolution_bucket",
    "clip_quality",
    "is_low_quality",
]

CONT_FEATS = ["sharpness_laplacian", "face_area_fraction", "luma_mean", "skin_frac"]


def synthesize_features(df: pd.DataFrame, calib_quartile_ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add derived columns: face_area_fraction (alias), luma_mean, skin_frac (none here),
    quartile bands, and source_resolution_bucket."""
    df = df.copy()
    # face area fraction alias
    if "face_area_ratio" in df.columns:
        df["face_area_fraction"] = df["face_area_ratio"]
    # luma alias
    if "brightness_v_mean" in df.columns:
        df["luma_mean"] = df["brightness_v_mean"]
    # skin_frac doesn't exist in this parquet; leave as NaN-only column for the schema
    df["skin_frac"] = np.nan

    # source resolution bucket: bucket on width*height
    if "width" in df.columns and "height" in df.columns:
        wh = (df["width"].astype("float") * df["height"].astype("float"))

        def _b(v):
            if pd.isna(v):
                return pd.NA
            if v < 60_000:
                return "<300x200"
            if v < 100_000:
                return "300-100k"
            if v < 200_000:
                return "100-200k"
            if v < 350_000:
                return "200-350k"
            return ">350k"
        df["source_resolution_bucket"] = wh.apply(_b)

    # is_low_quality boolean already exists; cast to str
    if "is_low_quality" in df.columns:
        df["is_low_quality"] = df["is_low_quality"].apply(
            lambda v: "True" if bool(v) else "False" if v is not None and not pd.isna(v) else pd.NA
        )

    # quartile binning: use calibration reference if provided, else self
    df = add_quartile(df, "face_area_ratio", "face_area_quartile", ref=calib_quartile_ref)
    df = add_quartile(df, "sharpness_laplacian", "sharpness_quartile", ref=calib_quartile_ref)
    return df


def main():
    logger.info("=== shortcut audit start ===")
    tags = load_tags()
    # use the calibration suite real-frames as reference quartile basis (so the
    # same Q1/Q2/Q3/Q4 boundaries apply across suites for stratification)

    # First pass: load every report once, joined; will need calib report twice
    joined_by_ckpt_suite: dict[tuple[str, str], pd.DataFrame] = {}
    coverage_rows = []

    for ckpt, cfg in CKPTS.items():
        for suite in SUITES:
            rp = report_path(ckpt, suite)
            if not rp.exists():
                logger.warning("MISSING report %s", rp)
                continue
            rep = pd.read_csv(rp)
            joined, cov = join_report_with_tags(rep, tags)
            coverage_rows.append({
                "ckpt": ckpt,
                "suite": suite,
                "n_report": len(rep),
                "n_joined": int(joined["gcs_uri"].notna().sum()),
                "coverage_pct": round(100 * cov, 2),
            })
            joined_by_ckpt_suite[(ckpt, suite)] = joined
            logger.info("loaded %s / %s: n=%d cov=%.1f%%", ckpt, suite, len(rep), 100 * cov)

    cov_df = pd.DataFrame(coverage_rows)
    cov_df.to_csv(OUT_DIR / "coverage.csv", index=False)
    logger.info("coverage.csv written")

    # Compute calibration thresholds from calib suite (quartile basis: REAL only, dropping rows without tags is FINE
    # for axis stratifications, but tau MUST come from full report regardless of tag availability)
    taus_per_ckpt: dict[str, dict[str, float]] = {}
    calib_quartile_ref_per_ckpt: dict[str, pd.DataFrame] = {}
    for ckpt in CKPTS:
        joined_calib = joined_by_ckpt_suite.get((ckpt, CALIB_SUITE))
        if joined_calib is None:
            logger.error("no calib suite for %s", ckpt)
            continue
        scores = joined_calib["frame_prob"].dropna().values
        tau10 = quantile_threshold(scores, 0.10)
        tau05 = quantile_threshold(scores, 0.05)
        taus_per_ckpt[ckpt] = {"tau_fpr_10": tau10, "tau_fpr_05": tau05}
        # build quartile ref from frames *with tags* on calib suite
        ref = joined_calib[joined_calib["gcs_uri"].notna()].copy()
        calib_quartile_ref_per_ckpt[ckpt] = ref
        logger.info("calib %s tau10=%.4f tau05=%.4f n_calib_total=%d n_with_tags=%d",
                    ckpt, tau10, tau05, len(joined_calib), len(ref))

    # ---------------------------------------------------------- per-axis table
    per_axis_rows = []
    for ckpt in CKPTS:
        ref = calib_quartile_ref_per_ckpt.get(ckpt)
        for suite in SUITES:
            joined = joined_by_ckpt_suite.get((ckpt, suite))
            if joined is None:
                continue
            kind = SUITE_KIND[suite]
            # For axis stratifications, drop rows missing tags
            j = joined[joined["gcs_uri"].notna()].copy()
            if len(j) == 0:
                logger.warning("axis: skipping %s/%s (no tagged rows)", ckpt, suite)
                continue
            j = synthesize_features(j, calib_quartile_ref=ref)
            for axis in AXES:
                t = per_axis_table(j, axis, taus_per_ckpt[ckpt], kind)
                if len(t) == 0:
                    continue
                t["ckpt"] = ckpt
                t["suite"] = suite
                per_axis_rows.append(t)
    per_axis_df = pd.concat(per_axis_rows, ignore_index=True) if per_axis_rows else pd.DataFrame()
    if len(per_axis_df):
        per_axis_df = per_axis_df[[
            "ckpt", "suite", "axis", "bucket", "n_total", "n_caught",
            "metric_kind", "metric_value", "tau_name", "tau_value",
        ]]
    per_axis_df.to_csv(OUT_DIR / "per_axis_recall_fpr.csv", index=False)
    logger.info("per_axis_recall_fpr.csv: %d rows", len(per_axis_df))

    # ---------------------------------------------------------- correlations
    corr_rows = []
    for ckpt in CKPTS:
        for suite in SUITES:
            joined = joined_by_ckpt_suite.get((ckpt, suite))
            if joined is None:
                continue
            j = joined[joined["gcs_uri"].notna()].copy()
            if len(j) == 0:
                continue
            j = synthesize_features(j, calib_quartile_ref=None)
            cdf = correlations_for(j, CONT_FEATS)
            if len(cdf) == 0:
                continue
            cdf["ckpt"] = ckpt
            cdf["suite"] = suite
            corr_rows.append(cdf)
    corr_df = pd.concat(corr_rows, ignore_index=True) if corr_rows else pd.DataFrame()
    if len(corr_df):
        corr_df = corr_df[["ckpt", "suite", "feature", "n", "pearson_r", "pearson_p", "spearman_rho", "spearman_p"]]
    corr_df.to_csv(OUT_DIR / "correlations.csv", index=False)
    logger.info("correlations.csv: %d rows", len(corr_df))

    # ---------------------------------------------------------- 2D heatmap
    # NOTE: deeplive face_area_ratio is tightly clustered near 0.12 (vs teams_real
    # which spans 0.07-0.91). If we use the calibration-suite quartile boundaries
    # all deeplive frames collapse into Q1. So we ALSO emit a local-binned variant
    # ("local_q") so the 2D heatmap has informative 4x4 coverage.
    heat_rows = []
    for ckpt in CKPTS:
        joined = joined_by_ckpt_suite.get((ckpt, "deeplive_enhanced_dev"))
        if joined is None:
            continue
        j = joined[joined["gcs_uri"].notna()].copy()
        # Apply BOTH global (calib-anchored) and local (deeplive-anchored) bins
        j_global = synthesize_features(j.copy(), calib_quartile_ref=calib_quartile_ref_per_ckpt.get(ckpt))
        j_local = synthesize_features(j.copy(), calib_quartile_ref=None)  # self-anchored
        tau10 = taus_per_ckpt[ckpt]["tau_fpr_10"]

        for binning_name, jdf in [("global_q", j_global), ("local_q", j_local)]:
            for fa in ["Q1", "Q2", "Q3", "Q4"]:
                for sh in ["Q1", "Q2", "Q3", "Q4"]:
                    sub = jdf[(jdf["face_area_quartile"].astype(str) == fa)
                              & (jdf["sharpness_quartile"].astype(str) == sh)]
                    n_total = len(sub)
                    if n_total == 0:
                        heat_rows.append({"ckpt": ckpt, "binning": binning_name,
                                          "face_area_quartile": fa, "sharpness_quartile": sh,
                                          "n_total": 0, "n_caught": 0, "recall": float("nan"),
                                          "tau_value": tau10})
                        continue
                    n_caught = int((sub["frame_prob"].values >= tau10).sum())
                    heat_rows.append({"ckpt": ckpt, "binning": binning_name,
                                      "face_area_quartile": fa, "sharpness_quartile": sh,
                                      "n_total": n_total, "n_caught": n_caught,
                                      "recall": n_caught / n_total, "tau_value": tau10})
    pd.DataFrame(heat_rows).to_csv(OUT_DIR / "face_area_x_sharpness_deeplive.csv", index=False)
    logger.info("face_area_x_sharpness_deeplive.csv written: %d rows", len(heat_rows))

    # ---------------------------------------------------------- per-identity FPR
    pi_rows = []
    for ckpt in CKPTS:
        joined = joined_by_ckpt_suite.get((ckpt, "teams_real_all_lockbox"))
        if joined is None:
            continue
        j = joined[joined["gcs_uri"].notna()].copy()
        if "identity_key" not in j.columns:
            continue
        tau10 = taus_per_ckpt[ckpt]["tau_fpr_10"]
        for ident, gdf in j.groupby("identity_key"):
            n_total = len(gdf)
            n_caught = int((gdf["frame_prob"].values >= tau10).sum())
            pi_rows.append({
                "ckpt": ckpt,
                "identity_key": ident,
                "n_frames": n_total,
                "n_caught": n_caught,
                "fpr": n_caught / n_total,
                "tau_value": tau10,
                "tau_name": "tau_fpr_10",
            })
    pi_df = pd.DataFrame(pi_rows).sort_values(["ckpt", "fpr"], ascending=[True, False])
    pi_df.to_csv(OUT_DIR / "per_identity_fpr_lockbox.csv", index=False)
    logger.info("per_identity_fpr_lockbox.csv written: %d rows", len(pi_df))

    # ---------------------------------------------------------- axis variance ranking
    rank_rows = []
    target_axes = ["face_area_quartile", "sharpness_quartile",
                   "clip_capture_mode", "clip_lighting", "source_resolution_bucket"]
    for ckpt in CKPTS:
        # On deeplive: max_minus_min recall at tau_fpr_10
        for axis in target_axes:
            sub = per_axis_df[(per_axis_df["ckpt"] == ckpt)
                              & (per_axis_df["suite"] == "deeplive_enhanced_dev")
                              & (per_axis_df["axis"] == axis)
                              & (per_axis_df["tau_name"] == "tau_fpr_10")]
            if len(sub) == 0:
                continue
            # require buckets with >=10 frames
            sub2 = sub[sub["n_total"] >= 10]
            if len(sub2) < 2:
                continue
            recall_range = float(sub2["metric_value"].max() - sub2["metric_value"].min())
            rank_rows.append({
                "ckpt": ckpt,
                "suite": "deeplive_enhanced_dev",
                "axis": axis,
                "kind": "recall",
                "n_buckets_used": int(len(sub2)),
                "max_minus_min": recall_range,
                "max_value": float(sub2["metric_value"].max()),
                "min_value": float(sub2["metric_value"].min()),
            })
        # On teams_real_all_dev: max_minus_min FPR at tau_fpr_10
        for axis in target_axes:
            sub = per_axis_df[(per_axis_df["ckpt"] == ckpt)
                              & (per_axis_df["suite"] == "teams_real_all_dev")
                              & (per_axis_df["axis"] == axis)
                              & (per_axis_df["tau_name"] == "tau_fpr_10")]
            if len(sub) == 0:
                continue
            sub2 = sub[sub["n_total"] >= 10]
            if len(sub2) < 2:
                continue
            fpr_range = float(sub2["metric_value"].max() - sub2["metric_value"].min())
            rank_rows.append({
                "ckpt": ckpt,
                "suite": "teams_real_all_dev",
                "axis": axis,
                "kind": "fpr",
                "n_buckets_used": int(len(sub2)),
                "max_minus_min": fpr_range,
                "max_value": float(sub2["metric_value"].max()),
                "min_value": float(sub2["metric_value"].min()),
            })
    rank_df = pd.DataFrame(rank_rows)
    rank_df = rank_df.sort_values(["ckpt", "kind", "max_minus_min"], ascending=[True, True, False])
    rank_df.to_csv(OUT_DIR / "axis_variance_ranking.csv", index=False)
    logger.info("axis_variance_ranking.csv written: %d rows", len(rank_df))

    # ---------------------------------------------------------- write tau summary
    tau_rows = []
    for ckpt in CKPTS:
        for nm, val in taus_per_ckpt[ckpt].items():
            tau_rows.append({"ckpt": ckpt, "tau_name": nm, "tau_value": val})
    pd.DataFrame(tau_rows).to_csv(OUT_DIR / "tau_calibration.csv", index=False)
    logger.info("tau_calibration.csv written")

    logger.info("=== shortcut audit done ===")


if __name__ == "__main__":
    main()
