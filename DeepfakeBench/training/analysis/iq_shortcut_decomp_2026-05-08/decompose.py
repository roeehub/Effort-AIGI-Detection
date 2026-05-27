"""IQ-shortcut R² decomposition (Stage 1 of IQ deconvolution program).

Per `docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md`
§4.1, this driver:

1. Joins per-frame IQ features (from the cross-pool atlas at
   `analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet`) to per-frame
   model scores from the Phase A scorecard (P2 verdict, 2026-05-08) and Phase C
   HDTF scorecard (P1 PE, 2026-05-07).
2. For each (ckpt × pool-group) cell: fits sklearn LinearRegression
   (n_jobs=1, per memory `feedback_sklearn_njobs.md`) with
   `model_score = β · IQ_features + ε`.
3. Reports R², residual `score_resid = score - β · IQ`, fake-vs-real AUC of the
   raw score, and fake-vs-real AUC of the residual.

Outputs:
  outputs/iq_decomp.csv
  outputs/iq_decomp.json
  outputs/per_frame_residuals.parquet
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
ATLAS_PARQUET = THIS_DIR.parent / "iq_data_atlas_2026-05-08" / "outputs" / "per_frame.parquet"
SCORES_CACHE = THIS_DIR / "scores_cache"
OUTPUTS = THIS_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
SCORES_CACHE.mkdir(parents=True, exist_ok=True)

PHASE_A_GS = "gs://training-job-outputs/test_results/teams_promotion_contract/p2-scratch-scorecard-2026-05-08/reports"
PHASE_C_GS = "gs://training-job-outputs/test_results/teams_promotion_contract/p1-pe-hdtf-scorecard-2026-05-07/reports"

# Checkpoint key → suite-prefix in the report file names
CKPTS = {
    "P8A_REFERENCE_STEP5000": "p8a_reference_step5000",
    "E2B_TOP_N_STEP3200": "e2b_top_n_step3200",
    "P2_D_FOURIER_PERIODIC_STEP3000": "p2_d_fourier_periodic_step3000",
}

# (atlas_pool, label) → list of (gs_prefix, suite_basename) it can be sourced from.
# label is 0 for real, 1 for fake. atlas_pool name matches the parquet's `pool` column.
PHASE_A_SUITES = {
    # dev — pooled real
    "teams_real_all_dev": (0, "teams_real_all_dev"),
    "teams_real_poor_quality_dev": (0, "teams_real_poor_quality_dev"),
    "teams_real_lighting_extreme_dev": (0, "teams_real_lighting_extreme_dev"),
    "teams_real_dor_dev": (0, "teams_real_dor_dev"),
    # dev fakes
    "teams_fake_all_dev": (1, "teams_fake_all_dev"),
    "visomaster_enhanced_macro_dev": (1, "visomaster_enhanced_macro_dev"),
    "deeplive_enhanced_dev": (1, "deeplive_enhanced_dev"),
    # lockbox
    "teams_real_all_lockbox": (0, "teams_real_all_lockbox"),
    "teams_fake_all_lockbox": (1, "teams_fake_all_lockbox"),
}

PHASE_C_SUITES = {
    "hdtf_real_clean_dev": (0, "proper_real_clean_dev"),
    "hdtf_real_clean_lockbox": (0, "proper_real_clean_lockbox"),
    "hdtf_real_teams_dev": (0, "proper_real_teams_dev"),
    "hdtf_real_teams_lockbox": (0, "proper_real_teams_lockbox"),
    "hdtf_fake_clean_dev": (1, "proper_fake_clean_all_dev"),
    "hdtf_fake_clean_lockbox": (1, "proper_fake_clean_all_lockbox"),
    "hdtf_fake_teams_dev": (1, "proper_fake_teams_all_dev"),
    "hdtf_fake_teams_lockbox": (1, "proper_fake_teams_all_lockbox"),
}

# Phase C scorecard only ran on P1 candidates + P8A + E2B; P2-D HDTF scores DO
# NOT exist as of 2026-05-08. Skip P2_D from HDTF cells.
PHASE_C_AVAILABLE_CKPTS = {"P8A_REFERENCE_STEP5000", "E2B_TOP_N_STEP3200"}

# IQ features used in the regression (per proposal §4.1 plus 1 brightness)
IQ_FEATURES = [
    "lap_var",
    "min_dim",
    "luma_mean",
    "color_b_dev",
    "edge_mag",
    "skin_frac",
]

# Sensitivity-check expanded feature set (reported separately)
IQ_FEATURES_EXPANDED = IQ_FEATURES + [
    "color_a_dev",
    "luma_std",
    "contrast_l",
    "saturation_mean",
]

# Pool-groups: each is a list of (atlas_pool, label_override_or_none).
# Each pool-group must contain both real and fake frames so we can compute AUC.
POOL_GROUPS = {
    "DEV_TEAMS_PRIMARY": ["teams_real_all_dev", "teams_fake_all_dev"],
    "DEV_DEEPLIVE_VS_REAL": ["teams_real_all_dev", "deeplive_enhanced_dev"],
    "DEV_VISO_VS_REAL": ["teams_real_all_dev", "visomaster_enhanced_macro_dev"],
    "DEV_TEAMS_STRESS_VS_FAKE": [
        "teams_real_lighting_extreme_dev",
        "teams_real_poor_quality_dev",
        "teams_fake_all_dev",
    ],
    "LOCKBOX_TEAMS": ["teams_real_all_lockbox", "teams_fake_all_lockbox"],
    "HDTF_CLEAN_DEV": ["hdtf_real_clean_dev", "hdtf_fake_clean_dev"],
    "HDTF_CLEAN_LOCKBOX": ["hdtf_real_clean_lockbox", "hdtf_fake_clean_lockbox"],
    "HDTF_TEAMS_DEV": ["hdtf_real_teams_dev", "hdtf_fake_teams_dev"],
    "HDTF_TEAMS_LOCKBOX": ["hdtf_real_teams_lockbox", "hdtf_fake_teams_lockbox"],
}


# -----------------------------------------------------------------------------
# Score download / caching
# -----------------------------------------------------------------------------


def gsutil_cat(gs_uri: str) -> str:
    """Single-process gsutil cat. Returns CSV text."""
    p = subprocess.run(
        ["gsutil", "cat", gs_uri],
        check=True,
        capture_output=True,
        text=True,
    )
    return p.stdout


def cache_path(ckpt_key: str, atlas_pool: str) -> Path:
    return SCORES_CACHE / f"{ckpt_key}__{atlas_pool}.csv"


def fetch_scores(
    ckpt_key: str, atlas_pool: str, suite_basename: str, gs_prefix: str
) -> pd.DataFrame:
    """Download (or load from cache) per-frame scores for one (ckpt, suite)."""
    cache = cache_path(ckpt_key, atlas_pool)
    if cache.exists():
        return pd.read_csv(cache)
    suite_prefix = CKPTS[ckpt_key]
    uri = f"{gs_prefix}/{suite_basename}_{suite_prefix}_frames_report.csv"
    print(f"[fetch] {ckpt_key} / {atlas_pool}  <-  {uri}")
    text = gsutil_cat(uri)
    cache.write_text(text)
    return pd.read_csv(cache)


# -----------------------------------------------------------------------------
# Build joined per-frame panel
# -----------------------------------------------------------------------------


def build_joined_panel() -> pd.DataFrame:
    """Returns long-format panel: one row per (ckpt × atlas_pool × frame_path).

    Columns: ckpt, atlas_pool, label, frame_path, frame_prob, + all IQ features.
    """
    print(f"[load atlas] {ATLAS_PARQUET}")
    atlas = pd.read_parquet(ATLAS_PARQUET)
    atlas = atlas[["frame_path", "pool"] + IQ_FEATURES_EXPANDED].copy()
    atlas = atlas.rename(columns={"pool": "atlas_pool"})
    print(f"[load atlas] rows={len(atlas):,}, pools={atlas['atlas_pool'].nunique()}")

    rows = []
    for ckpt_key in CKPTS:
        is_p2d = ckpt_key == "P2_D_FOURIER_PERIODIC_STEP3000"
        # Phase A (dev + lockbox)
        for atlas_pool, (label, suite_basename) in PHASE_A_SUITES.items():
            scores = fetch_scores(ckpt_key, atlas_pool, suite_basename, PHASE_A_GS)
            scores = scores[["frame_path", "frame_prob"]].copy()
            scores["ckpt"] = ckpt_key
            scores["atlas_pool"] = atlas_pool
            scores["label"] = label
            rows.append(scores)
        # Phase C (HDTF) — only for P8A and E2B
        if ckpt_key in PHASE_C_AVAILABLE_CKPTS:
            for atlas_pool, (label, suite_basename) in PHASE_C_SUITES.items():
                scores = fetch_scores(ckpt_key, atlas_pool, suite_basename, PHASE_C_GS)
                scores = scores[["frame_path", "frame_prob"]].copy()
                scores["ckpt"] = ckpt_key
                scores["atlas_pool"] = atlas_pool
                scores["label"] = label
                rows.append(scores)

    scored = pd.concat(rows, ignore_index=True)
    print(f"[scored] total rows pre-join: {len(scored):,}")

    # Inner join on (atlas_pool, frame_path). Atlas has a sample, scored has all
    # frames in the suite; the join keeps only the frames that have IQ features.
    joined = scored.merge(atlas, on=["frame_path", "atlas_pool"], how="inner")
    print(f"[joined] rows after inner join: {len(joined):,}")
    return joined


# -----------------------------------------------------------------------------
# Per-cell decomposition
# -----------------------------------------------------------------------------


@dataclass
class CellResult:
    ckpt: str
    pool_group: str
    n_total: int
    n_real: int
    n_fake: int
    feature_set: str
    r2: float
    raw_auc: float
    residual_auc: float
    score_mean_real: float
    score_mean_fake: float
    score_p50_real: float
    score_p50_fake: float
    resid_mean_real: float
    resid_mean_fake: float
    coef: dict


def fit_one_cell(
    df: pd.DataFrame, ckpt: str, pool_group: str, feats: list[str], feat_set_name: str
) -> tuple[CellResult, np.ndarray]:
    """Fit OLS, compute R² + residual AUC. Returns result + per-row residuals."""
    X = df[feats].to_numpy(dtype=np.float64)
    y = df["frame_prob"].to_numpy(dtype=np.float64)
    labels = df["label"].to_numpy()

    reg = LinearRegression(n_jobs=1)
    reg.fit(X, y)
    y_hat = reg.predict(X)
    resid = y - y_hat
    r2 = reg.score(X, y)

    raw_auc = (
        roc_auc_score(labels, y) if (labels.min() == 0 and labels.max() == 1) else float("nan")
    )
    resid_auc = (
        roc_auc_score(labels, resid)
        if (labels.min() == 0 and labels.max() == 1)
        else float("nan")
    )

    is_real = labels == 0
    is_fake = labels == 1
    score_mean_real = float(np.mean(y[is_real])) if is_real.any() else float("nan")
    score_mean_fake = float(np.mean(y[is_fake])) if is_fake.any() else float("nan")
    score_p50_real = float(np.median(y[is_real])) if is_real.any() else float("nan")
    score_p50_fake = float(np.median(y[is_fake])) if is_fake.any() else float("nan")
    resid_mean_real = float(np.mean(resid[is_real])) if is_real.any() else float("nan")
    resid_mean_fake = float(np.mean(resid[is_fake])) if is_fake.any() else float("nan")

    coef = {f: float(c) for f, c in zip(feats, reg.coef_)}
    coef["__intercept__"] = float(reg.intercept_)

    res = CellResult(
        ckpt=ckpt,
        pool_group=pool_group,
        n_total=len(df),
        n_real=int(is_real.sum()),
        n_fake=int(is_fake.sum()),
        feature_set=feat_set_name,
        r2=float(r2),
        raw_auc=float(raw_auc),
        residual_auc=float(resid_auc),
        score_mean_real=score_mean_real,
        score_mean_fake=score_mean_fake,
        score_p50_real=score_p50_real,
        score_p50_fake=score_p50_fake,
        resid_mean_real=resid_mean_real,
        resid_mean_fake=resid_mean_fake,
        coef=coef,
    )
    return res, resid


def main() -> None:
    panel = build_joined_panel()
    panel.to_parquet(OUTPUTS / "joined_panel.parquet", index=False)
    print(f"[save] {OUTPUTS / 'joined_panel.parquet'}")

    rows: list[dict] = []
    resid_rows: list[pd.DataFrame] = []

    for ckpt in CKPTS:
        for pg_name, atlas_pools in POOL_GROUPS.items():
            sub = panel[(panel["ckpt"] == ckpt) & (panel["atlas_pool"].isin(atlas_pools))].copy()
            if len(sub) == 0:
                continue
            if sub["label"].nunique() < 2:
                # Skip cells without both classes (HDTF for P2-D etc.)
                continue
            for feat_set_name, feats in [
                ("primary_6", IQ_FEATURES),
                ("expanded_10", IQ_FEATURES_EXPANDED),
            ]:
                # Drop rows with NaN in any feature
                clean = sub.dropna(subset=feats + ["frame_prob"])
                if clean["label"].nunique() < 2 or len(clean) < 30:
                    continue
                res, resid = fit_one_cell(clean, ckpt, pg_name, feats, feat_set_name)
                rows.append(res.__dict__)
                if feat_set_name == "primary_6":
                    rdf = clean[["frame_path", "atlas_pool", "label", "frame_prob"]].copy()
                    rdf["ckpt"] = ckpt
                    rdf["pool_group"] = pg_name
                    rdf["score_resid"] = resid
                    resid_rows.append(rdf)

    df_results = pd.DataFrame(rows)
    df_results.to_csv(OUTPUTS / "iq_decomp.csv", index=False)
    print(f"[save] {OUTPUTS / 'iq_decomp.csv'} rows={len(df_results)}")

    # Compact JSON without coef nesting
    j = []
    for d in rows:
        d2 = dict(d)
        d2["coef"] = d["coef"]
        j.append(d2)
    (OUTPUTS / "iq_decomp.json").write_text(json.dumps(j, indent=2))
    print(f"[save] {OUTPUTS / 'iq_decomp.json'}")

    if resid_rows:
        per_frame_resid = pd.concat(resid_rows, ignore_index=True)
        per_frame_resid.to_parquet(OUTPUTS / "per_frame_residuals.parquet", index=False)
        print(f"[save] per_frame_residuals.parquet rows={len(per_frame_resid):,}")


if __name__ == "__main__":
    main()
