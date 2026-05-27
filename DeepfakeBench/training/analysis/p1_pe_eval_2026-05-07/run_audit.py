"""
CPU-only 5-axis shortcut audit — F2/F3 audit for P1 PE_PAIR_RANK_DRO
scorecard (2026-05-07).

Run AFTER the Phase A scorecard's per-frame reports land in
analysis/p1_pe_eval_2026-05-07/raw_reports/. Compares the P1 BUNDLE +
PAIRRANK ckpts against P8A + E2B baselines on Pearson |r| against the
F2/F3 axes.

F2 close criterion: shortcut weakening >= 30% on >= 2 of 5 axes vs E2B
F3 close criterion: no untargeted axis amplifies +50%

Cloned from analysis/deeplive_viso_corr_eval_2026-05-06/run_audit.py;
only CKPTS + paths differ.
"""
from __future__ import annotations

import logging
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ------------------------------------------------------------ paths / config

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT_DIR = ROOT / "analysis/p1_pe_eval_2026-05-07"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Populated by `gcloud storage cp -r gs://.../p1-pe-pair-rank-scorecard-2026-05-07/reports/ <NEW_DIR>`
# after the Phase A scorecard finishes.
NEW_DIR = OUT_DIR / "raw_reports/phase_a"

CKPTS = {
    "p8a":                          {"dir": NEW_DIR, "tag": "p8a_reference_step5000"},
    "e2b":                          {"dir": NEW_DIR, "tag": "e2b_top_n_step3200"},
    "p1_bundle_periodic_500":       {"dir": NEW_DIR, "tag": "p1_bundle_periodic_step500"},
    "p1_bundle_top_n_3750":         {"dir": NEW_DIR, "tag": "p1_bundle_top_n_step3750"},
    "p1_bundle_top_n_4000":         {"dir": NEW_DIR, "tag": "p1_bundle_top_n_step4000"},
    "p1_pairrank_periodic_500":     {"dir": NEW_DIR, "tag": "p1_pairrank_periodic_step500"},
    "p1_pairrank_top_n_6000":       {"dir": NEW_DIR, "tag": "p1_pairrank_top_n_step6000"},
    "p1_pairrank_top_n_6750":       {"dir": NEW_DIR, "tag": "p1_pairrank_top_n_step6750"},
}

# Suites available in pa_pc_eval_2026-05-05/raw_reports
SUITES = [
    "visomaster_enhanced_macro_dev",
    "deeplive_enhanced_dev",
    "teams_fake_all_dev",
    "teams_fake_all_lockbox",
    "teams_real_all_dev",
    "teams_real_all_lockbox",
]

SUITE_KIND = {
    "visomaster_enhanced_macro_dev": "fake",
    "deeplive_enhanced_dev": "fake",
    "teams_fake_all_dev": "fake",
    "teams_fake_all_lockbox": "fake",
    "teams_real_all_dev": "real",
    "teams_real_all_lockbox": "real",
}

CALIB_SUITE = "teams_real_all_dev"
TAU_LEVELS = {"tau_fpr_10": 0.10, "tau_fpr_05": 0.05}

# ------------------------------------------------------------ logging

LOG_PATH = OUT_DIR / "run.log"
if LOG_PATH.exists():
    LOG_PATH.unlink()

logger = logging.getLogger("corr_audit")
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


# ------------------------------------------------------------ helpers

def add_quartile(df: pd.DataFrame, src: str, dst: str, ref: pd.DataFrame | None = None):
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


def quantile_threshold(scores: np.ndarray, fpr: float) -> float:
    if len(scores) == 0:
        return float("nan")
    return float(np.quantile(scores, 1.0 - fpr))


def per_axis_table(joined: pd.DataFrame, axis_col: str, taus: dict[str, float], suite_kind: str) -> pd.DataFrame:
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
                "metric_value": metric,
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
            rows.append({"feature": feat, "pearson_r": float("nan"), "pearson_p": float("nan"),
                         "spearman_rho": float("nan"), "spearman_p": float("nan"), "n": len(sub)})
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
        rows.append({"feature": feat, "pearson_r": pearson_r, "pearson_p": pearson_p,
                     "spearman_rho": spearman_rho, "spearman_p": spearman_p, "n": len(sub)})
    return pd.DataFrame(rows)


def join_report_with_tags(report: pd.DataFrame, tags: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    joined = report.merge(tags, left_on="frame_path", right_on="gcs_uri", how="left")
    cov = joined["gcs_uri"].notna().mean()
    return joined, cov


# ------------------------------------------------------------ feature synth

AXES = [
    "face_area_quartile",
    "sharpness_quartile",
    "clip_capture_mode",
    "clip_lighting",
    "source_resolution_bucket",
    "clip_quality",
    "is_low_quality",
]

CONT_FEATS = ["sharpness_laplacian", "face_area_fraction", "luma_mean"]


def synthesize_features(df: pd.DataFrame, calib_quartile_ref: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df.copy()
    if "face_area_ratio" in df.columns:
        df["face_area_fraction"] = df["face_area_ratio"]
    if "brightness_v_mean" in df.columns:
        df["luma_mean"] = df["brightness_v_mean"]

    if "width" in df.columns and "height" in df.columns:
        wh = (df["width"].astype("float") * df["height"].astype("float"))
        def _b(v):
            if pd.isna(v):
                return pd.NA
            if v < 60_000: return "<300x200"
            if v < 100_000: return "300-100k"
            if v < 200_000: return "100-200k"
            if v < 350_000: return "200-350k"
            return ">350k"
        df["source_resolution_bucket"] = wh.apply(_b)

    if "is_low_quality" in df.columns:
        df["is_low_quality"] = df["is_low_quality"].apply(
            lambda v: "True" if bool(v) else "False" if v is not None and not pd.isna(v) else pd.NA
        )

    df = add_quartile(df, "face_area_ratio", "face_area_quartile", ref=calib_quartile_ref)
    df = add_quartile(df, "sharpness_laplacian", "sharpness_quartile", ref=calib_quartile_ref)
    return df


# ------------------------------------------------------------ main

def main():
    logger.info("=== corr-penalty baseline audit start (n_ckpts=%d, n_suites=%d) ===",
                len(CKPTS), len(SUITES))
    tags = load_tags()

    joined_by_ckpt_suite: dict[tuple[str, str], pd.DataFrame] = {}
    coverage_rows = []

    for ckpt in CKPTS:
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

    pd.DataFrame(coverage_rows).to_csv(OUT_DIR / "coverage.csv", index=False)
    logger.info("coverage.csv written")

    # ---------------------------------------------------------- tau calibration
    taus_per_ckpt: dict[str, dict[str, float]] = {}
    calib_ref_per_ckpt: dict[str, pd.DataFrame] = {}
    for ckpt in CKPTS:
        joined_calib = joined_by_ckpt_suite.get((ckpt, CALIB_SUITE))
        if joined_calib is None:
            logger.error("no calib suite for %s", ckpt)
            continue
        scores = joined_calib["frame_prob"].dropna().values
        tau10 = quantile_threshold(scores, 0.10)
        tau05 = quantile_threshold(scores, 0.05)
        taus_per_ckpt[ckpt] = {"tau_fpr_10": tau10, "tau_fpr_05": tau05}
        ref = joined_calib[joined_calib["gcs_uri"].notna()].copy()
        calib_ref_per_ckpt[ckpt] = ref
        logger.info("calib %s tau10=%.4f tau05=%.4f n_calib_total=%d n_with_tags=%d",
                    ckpt, tau10, tau05, len(joined_calib), len(ref))

    # ---------------------------------------------------------- per-axis table
    per_axis_rows = []
    for ckpt in CKPTS:
        ref = calib_ref_per_ckpt.get(ckpt)
        for suite in SUITES:
            joined = joined_by_ckpt_suite.get((ckpt, suite))
            if joined is None:
                continue
            kind = SUITE_KIND[suite]
            j = joined[joined["gcs_uri"].notna()].copy()
            if len(j) == 0:
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

    # ---------------------------------------------------------- correlations (the headline F2/F3 table)
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

    # ---------------------------------------------------------- F2/F3 baseline summary table
    # For each ckpt, compute |r| on each of the 3 corr-penalty axes against
    # frame_prob, on real-frame suites (where shortcut bias matters most for FPR)
    # and fake suites (where it dominates recall variance).
    summary_rows = []
    for ckpt in CKPTS:
        for suite_kind_filter in ["fake", "real"]:
            target_suites = [s for s in SUITES if SUITE_KIND[s] == suite_kind_filter]
            for feat in CONT_FEATS:
                vals = []
                for suite in target_suites:
                    sub = corr_df[(corr_df["ckpt"] == ckpt)
                                  & (corr_df["suite"] == suite)
                                  & (corr_df["feature"] == feat)]
                    if len(sub) and not pd.isna(sub["pearson_r"].iloc[0]):
                        vals.append(abs(float(sub["pearson_r"].iloc[0])))
                if vals:
                    summary_rows.append({
                        "ckpt": ckpt,
                        "suite_kind": suite_kind_filter,
                        "feature": feat,
                        "abs_pearson_mean": float(np.mean(vals)),
                        "abs_pearson_max": float(np.max(vals)),
                        "n_suites": len(vals),
                    })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "abs_pearson_summary.csv", index=False)
    logger.info("abs_pearson_summary.csv: %d rows", len(summary_df))

    # tau summary
    tau_rows = []
    for ckpt in CKPTS:
        for nm, val in taus_per_ckpt.get(ckpt, {}).items():
            tau_rows.append({"ckpt": ckpt, "tau_name": nm, "tau_value": val})
    pd.DataFrame(tau_rows).to_csv(OUT_DIR / "tau_calibration.csv", index=False)
    logger.info("tau_calibration.csv written")

    logger.info("=== corr-penalty baseline audit done ===")


if __name__ == "__main__":
    main()
