"""D6 — Empirical orthogonality baseline for LR-probe direction angles.

Tests the load-bearing question raised in `D1_D5_CRITIC_REVIEW_2026-05-12.md`
§3.1, §6: what is the empirical noise floor on angle measurements between
LR-probe coefficient vectors that are measuring the SAME thing on independent
random subsets of the same data? Without this, the 4.8°-8.7° / 9°-13° drifts
reported in `D2_FACTS_2026-05-12.md` §3 cannot be sized against measurement
variance.

Method per ckpt × sub_cohort:
  For each bootstrap b in 1..B (B=50):
    - Random stratified 50/50 split of the cohort (by label).
    - Fit LR(C=1.0, max_iter=2000) on each half -> w1, w2; unit-normalize.
    - Compute angle_rf = arccos(|<w1, w2>|) * 180/pi.
    - For each of 6 IQ axes: fit LinearRegression(features -> z-scored axis)
      on each half -> 6 axis vectors per half; stack 6x768; SVD; take first
      right-singular-vector as IQ-PC1 per half; angle between halves' PC1s.
    - For each half: angle between that half's w_rf and that half's IQ-PC1
      (= D2 measurement replicated on a half-cohort).

Reports for each metric: mean, std, p10, p25, p50, p75, p90 of the bootstrap
distribution, plus the D2 single-split reference and a z-score.

CPU-only; sklearn n_jobs=1 throughout.

Inputs:
  - L11 caches: analysis/iq_perlayer_probe_2026-05-08/_cache/
    intermediate__{LABEL}__layer11__n800.npz (5 ckpts)
  - Triptych metadata: analysis/embedding_triptych_2026-04-30/outputs/
    triptych_p8a_slot2_slot3/sampled_frames.csv
  - IQ atlas: analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet
  - IQ inline cv2 fallback.

Outputs:
  - outputs/bootstrap_rf_direction_stability.csv
  - outputs/bootstrap_iq_pc1_stability.csv
  - outputs/bootstrap_d2_replicate.csv
  - outputs/d2_vs_empirical_comparison.csv
  - outputs/_summary.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

THIS_DIR = Path(__file__).resolve().parent
OUTPUTS = THIS_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

CACHE_DIR = REPO_ROOT / "analysis" / "iq_perlayer_probe_2026-05-08" / "_cache"
SAMPLED_CSV = (
    REPO_ROOT
    / "analysis"
    / "embedding_triptych_2026-04-30"
    / "outputs"
    / "triptych_p8a_slot2_slot3"
    / "sampled_frames.csv"
)
ATLAS_PARQUET = (
    REPO_ROOT / "analysis" / "iq_data_atlas_2026-05-08" / "outputs" / "per_frame.parquet"
)

IQ_AXES = ["lap_var", "min_dim", "color_a_dev", "color_b_dev", "saturation_mean", "luma_mean"]

CHRONIC_6_TOKENS = ["Roy_D", "PC_Generator", "bla_bla_chow",
                    "Md_noyn_Sharker", "dor_shkedi", "healthy_dor"]

CKPT_LABELS = ["CLIP_FROZEN", "P8A", "E2B", "T5C_periodic_step3500", "T3_S1_step1500"]
DISPLAY_NAMES = {
    "CLIP_FROZEN": "CLIP_FROZEN",
    "P8A": "P8A",
    "E2B": "E2B",
    "T5C_periodic_step3500": "T5C_step3500",
    "T3_S1_step1500": "T3_S1_step1500",
}

# Hard-coded D2 single-split angles for direct comparison (chronic_6, full,
# non_chronic). Source: D2_FACTS_2026-05-12.md §3 / outputs/angle_to_iq_pc1.csv.
D2_ANGLES_TO_PC1: Dict[Tuple[str, str], float] = {
    ("CLIP_FROZEN", "full"): 89.85586995462728,
    ("CLIP_FROZEN", "chronic_6"): 83.67638051765091,
    ("CLIP_FROZEN", "non_chronic"): 88.45812567974298,
    ("P8A", "full"): 89.25249422763902,
    ("P8A", "chronic_6"): 78.53815614415434,
    ("P8A", "non_chronic"): 89.40261520615357,
    ("E2B", "full"): 89.73514629116676,
    ("E2B", "chronic_6"): 74.94718858805753,
    ("E2B", "non_chronic"): 88.91051560731628,
    ("T5C_step3500", "full"): 89.24933220562022,
    ("T5C_step3500", "chronic_6"): 75.18791030283309,
    ("T5C_step3500", "non_chronic"): 88.9245586110462,
    ("T3_S1_step1500", "full"): 89.9748612654135,
    ("T3_S1_step1500", "chronic_6"): 77.09123859433018,
    ("T3_S1_step1500", "non_chronic"): 89.25736197985707,
}

# Bootstrap configuration.
B_BOOTSTRAPS_DEFAULT = 50
RANDOM_SEED = 20260512

logger = logging.getLogger("d6-empirical-orthogonality")


# ---------------------------------------------------------------------------
# IQ panel (same construction as D2).
# ---------------------------------------------------------------------------
def compute_iq_inline(local_paths: List[str]) -> pd.DataFrame:
    """Compute the 6 IQ axes inline via cv2/LAB (matches D2)."""
    import cv2

    rows = []
    for p in local_paths:
        try:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                rows.append({k: np.nan for k in IQ_AXES})
                continue
            h, w = img.shape[:2]
            min_dim = float(min(h, w))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation_mean = float(hsv[..., 1].astype(np.float32).mean())
            luma_mean = float(hsv[..., 2].astype(np.float32).mean())
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            color_a_dev = float(np.abs(lab[..., 1] - 128.0).mean())
            color_b_dev = float(np.abs(lab[..., 2] - 128.0).mean())
            rows.append({
                "lap_var": lap_var,
                "min_dim": min_dim,
                "color_a_dev": color_a_dev,
                "color_b_dev": color_b_dev,
                "saturation_mean": saturation_mean,
                "luma_mean": luma_mean,
            })
        except Exception:
            rows.append({k: np.nan for k in IQ_AXES})
    return pd.DataFrame(rows)


def build_iq_panel() -> pd.DataFrame:
    df = pd.read_csv(
        SAMPLED_CSV,
        usecols=["gcs_uri", "label", "split", "identity_key", "local_path"],
    )
    df = df.iloc[:800].reset_index(drop=True)
    df["row_ix"] = np.arange(len(df))

    df["is_chronic_6"] = df["gcs_uri"].fillna("").str.contains(
        "|".join(CHRONIC_6_TOKENS), case=False, regex=True
    ).astype(int)

    atlas = pd.read_parquet(ATLAS_PARQUET)[["frame_path"] + IQ_AXES].copy()
    atlas = atlas.drop_duplicates(subset=["frame_path"], keep="first")
    merged = df.merge(atlas, left_on="gcs_uri", right_on="frame_path", how="left")
    n_in_atlas = merged[IQ_AXES[0]].notna().sum()
    n_missing = len(merged) - int(n_in_atlas)
    logger.info("IQ atlas join: %d in atlas, %d need inline cv2",
                n_in_atlas, n_missing)

    if n_missing > 0:
        missing_mask = merged[IQ_AXES[0]].isna()
        missing_paths = merged.loc[missing_mask, "local_path"].tolist()
        inline = compute_iq_inline(missing_paths)
        for col in IQ_AXES:
            merged.loc[missing_mask, col] = inline[col].values
    return merged


# ---------------------------------------------------------------------------
# Feature loading.
# ---------------------------------------------------------------------------
def load_l11_features(label: str, n: int = 800) -> Tuple[np.ndarray, np.ndarray]:
    p = CACHE_DIR / f"intermediate__{label}__layer11__n{n}.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing cache: {p}")
    blob = np.load(p)
    return blob["features"].astype(np.float32), blob["valid_idx"].astype(np.int64)


# ---------------------------------------------------------------------------
# Direction extraction (same primitives as D2).
# ---------------------------------------------------------------------------
def fit_rf_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LR fit; return unit-normalized w_rf. NaN-safe."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=1, solver="lbfgs")
    clf.fit(X, y)
    w = clf.coef_[0].astype(np.float64)
    n = np.linalg.norm(w)
    if n <= 0:
        return w
    return w / n


def fit_axis_direction(X: np.ndarray, axis_values: np.ndarray) -> np.ndarray:
    """LinearRegression(features -> z-scored axis). Return unit vector."""
    from sklearn.linear_model import LinearRegression

    mu = float(axis_values.mean())
    sd = float(axis_values.std())
    if sd <= 0:
        return np.zeros(X.shape[1], dtype=np.float64)
    z = (axis_values - mu) / sd
    reg = LinearRegression(n_jobs=1)
    reg.fit(X, z)
    w = reg.coef_.astype(np.float64)
    n = np.linalg.norm(w)
    if n <= 0:
        return w
    return w / n


def cosine_angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Unsigned angle between u and v in degrees. Uses |cos|."""
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu <= 0 or nv <= 0:
        return float("nan")
    c = float(np.dot(u, v) / (nu * nv))
    c = min(1.0, max(-1.0, c))
    return float(np.degrees(np.arccos(abs(c))))


def compute_iq_pc1(X: np.ndarray, panel: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    """Fit 6 axis directions, stack 6 x d, SVD, return first right-singular-vec."""
    w_axes: List[np.ndarray] = []
    for ax in IQ_AXES:
        ax_vals = panel[ax].values[idx].astype(np.float32)
        try:
            w_ax = fit_axis_direction(X[idx], ax_vals)
        except Exception:
            w_ax = np.zeros(X.shape[1], dtype=np.float64)
        w_axes.append(w_ax)
    W = np.stack(w_axes, axis=0)
    norms = np.linalg.norm(W, axis=1)
    valid_rows = W[norms > 1e-12]
    if valid_rows.shape[0] < 2:
        return np.zeros(X.shape[1], dtype=np.float64)
    U, S, Vt = np.linalg.svd(valid_rows, full_matrices=False)
    return Vt[0]


def stratified_split_indices(
    cohort_idx: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified 50/50 split of `cohort_idx` by `y` (binary). Returns two
    disjoint index arrays."""
    half1, half2 = [], []
    for cls in [0, 1]:
        cls_idx = cohort_idx[y == cls]
        perm = rng.permutation(len(cls_idx))
        cls_shuf = cls_idx[perm]
        cut = len(cls_shuf) // 2
        half1.extend(cls_shuf[:cut])
        half2.extend(cls_shuf[cut:cut + (len(cls_shuf) - cut)])
    return np.array(half1, dtype=np.int64), np.array(half2, dtype=np.int64)


# ---------------------------------------------------------------------------
# Bootstrap per ckpt × cohort.
# ---------------------------------------------------------------------------
def bootstrap_one_cell(
    label: str,
    cohort_name: str,
    X: np.ndarray,
    y: np.ndarray,
    panel: pd.DataFrame,
    cohort_idx: np.ndarray,
    B: int,
    base_seed: int,
) -> List[dict]:
    """Run B bootstraps on (X, y, panel) restricted to cohort_idx.

    Returns a list of dicts, one row per bootstrap, with:
      - angle_rf (between half1's w_rf and half2's w_rf)
      - angle_iqpc1 (between half1's IQ-PC1 and half2's IQ-PC1)
      - angle_d2_h1 (half1's w_rf vs half1's IQ-PC1)
      - angle_d2_h2 (half2's w_rf vs half2's IQ-PC1)
    """
    rows: List[dict] = []
    if len(cohort_idx) < 20:
        logger.warning("[%s/%s] cohort too small (n=%d); skipping",
                       label, cohort_name, len(cohort_idx))
        return rows

    # Pull local cohort arrays.
    Xc = X[cohort_idx]
    yc = y[cohort_idx]
    # Map cohort_idx -> position in X (we need to fit IQ-PC1 using ax_vals
    # from the FULL panel via cohort_idx); but compute_iq_pc1 takes (X, panel,
    # idx) where idx is into X. So we'll pass X and indices into X directly.

    n_c = len(cohort_idx)
    n_real = int((yc == 0).sum())
    n_fake = int((yc == 1).sum())
    if n_real < 5 or n_fake < 5:
        logger.warning("[%s/%s] insufficient labels real=%d fake=%d; skipping",
                       label, cohort_name, n_real, n_fake)
        return rows

    rng = np.random.default_rng(base_seed)

    t0 = time.time()
    for b in range(B):
        # Stratified 50/50 split of cohort_idx into half1/half2 (indices into X).
        # Sub-seed per bootstrap so each iteration is reproducible.
        sub_rng = np.random.default_rng(base_seed * 1_000_003 + b)
        # First map positions in cohort to labels for stratification.
        cohort_positions = np.arange(n_c)
        h1_pos, h2_pos = stratified_split_indices(cohort_positions, yc, sub_rng)
        if len(h1_pos) < 5 or len(h2_pos) < 5:
            logger.warning("[%s/%s/b=%d] half too small; skipping",
                           label, cohort_name, b)
            continue
        # Confirm both halves have both classes.
        if len(np.unique(yc[h1_pos])) < 2 or len(np.unique(yc[h2_pos])) < 2:
            logger.warning("[%s/%s/b=%d] half missing class; skipping",
                           label, cohort_name, b)
            continue

        # Map back to X indices.
        h1_idx_X = cohort_idx[h1_pos]
        h2_idx_X = cohort_idx[h2_pos]

        # Fit w_rf on each half.
        try:
            w1_rf = fit_rf_direction(Xc[h1_pos], yc[h1_pos])
            w2_rf = fit_rf_direction(Xc[h2_pos], yc[h2_pos])
        except Exception as e:
            logger.warning("[%s/%s/b=%d] w_rf fit failed: %s",
                           label, cohort_name, b, e)
            continue

        angle_rf = cosine_angle_deg(w1_rf, w2_rf)

        # Fit IQ-PC1 on each half. compute_iq_pc1 uses (X, panel, idx) where idx
        # indexes into X. So pass full X and h1_idx_X / h2_idx_X.
        try:
            pc1_h1 = compute_iq_pc1(X, panel, h1_idx_X)
            pc1_h2 = compute_iq_pc1(X, panel, h2_idx_X)
        except Exception as e:
            logger.warning("[%s/%s/b=%d] IQ-PC1 fit failed: %s",
                           label, cohort_name, b, e)
            continue

        angle_iqpc1 = cosine_angle_deg(pc1_h1, pc1_h2)

        # D2 replicate per half: w_rf vs IQ-PC1 within the same half.
        angle_d2_h1 = cosine_angle_deg(w1_rf, pc1_h1)
        angle_d2_h2 = cosine_angle_deg(w2_rf, pc1_h2)

        rows.append({
            "ckpt": DISPLAY_NAMES.get(label, label),
            "sub_cohort": cohort_name,
            "bootstrap_b": b,
            "n_half1": int(len(h1_pos)),
            "n_half2": int(len(h2_pos)),
            "angle_rf_deg": angle_rf,
            "angle_iqpc1_deg": angle_iqpc1,
            "angle_d2_h1_deg": angle_d2_h1,
            "angle_d2_h2_deg": angle_d2_h2,
        })

        if (b + 1) % 10 == 0:
            elapsed = time.time() - t0
            logger.info("  [%s/%s] b=%d/%d (%.1fs elapsed)",
                        label, cohort_name, b + 1, B, elapsed)

    return rows


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------
def summarize_distribution(values: List[float]) -> dict:
    """Return dict with mean, std, p10, p25, p50, p75, p90 of a numeric list."""
    arr = np.array([v for v in values if v is not None and not np.isnan(v)],
                   dtype=np.float64)
    if len(arr) == 0:
        return {
            "n_valid": 0,
            "mean": np.nan, "std": np.nan,
            "p10": np.nan, "p25": np.nan, "p50": np.nan,
            "p75": np.nan, "p90": np.nan,
        }
    return {
        "n_valid": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main(B_bootstraps: int = B_BOOTSTRAPS_DEFAULT) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    logger.info("D6 — empirical orthogonality baseline; B=%d bootstraps",
                B_bootstraps)
    t_start = time.time()

    # ----- 1. Build IQ panel.
    panel = build_iq_panel()
    logger.info("panel rows=%d; chronic_6 count=%d",
                len(panel), int(panel["is_chronic_6"].sum()))

    # ----- 2. Per-ckpt bootstrap.
    all_rows: List[dict] = []
    cells_processed: List[Tuple[str, str]] = []

    for label_idx, label in enumerate(CKPT_LABELS):
        try:
            feats, vidx = load_l11_features(label, n=800)
        except FileNotFoundError as e:
            logger.error("missing cache for %s: %s", label, e)
            continue

        # Build aligned panel subset.
        sub_panel = panel.iloc[vidx].reset_index(drop=True)
        sub_panel["y"] = (sub_panel["label"] == "fake").astype(int)

        # Filter to IQ-complete rows (same logic as D2).
        iq_ok = sub_panel[IQ_AXES].notna().all(axis=1)
        keep_mask = iq_ok.values
        X_keep = feats[keep_mask]
        panel_keep = sub_panel.loc[keep_mask].reset_index(drop=True)
        y_keep = panel_keep["y"].values
        n_keep = len(panel_keep)
        logger.info("[%s] feature rows=%d, IQ-complete=%d, feat_dim=%d",
                    label, len(sub_panel), n_keep, X_keep.shape[1])

        # Define sub-cohorts (positions in panel_keep / X_keep).
        cohorts: Dict[str, np.ndarray] = {
            "full": np.arange(n_keep, dtype=np.int64),
            "chronic_6": np.where(panel_keep["is_chronic_6"].values == 1)[0],
            "non_chronic": np.where(panel_keep["is_chronic_6"].values == 0)[0],
        }

        for cohort_name, idx in cohorts.items():
            if len(idx) < 20:
                logger.info("[%s/%s] cohort n=%d too small; skipping",
                            label, cohort_name, len(idx))
                continue

            t_cell_start = time.time()
            cell_seed = RANDOM_SEED + 100 * label_idx + (
                {"full": 0, "chronic_6": 1, "non_chronic": 2}[cohort_name]
            )
            rows = bootstrap_one_cell(
                label=label,
                cohort_name=cohort_name,
                X=X_keep,
                y=y_keep,
                panel=panel_keep,
                cohort_idx=idx,
                B=B_bootstraps,
                base_seed=cell_seed,
            )
            t_cell = time.time() - t_cell_start
            logger.info("[%s/%s] done; %d rows in %.1fs",
                        label, cohort_name, len(rows), t_cell)
            all_rows.extend(rows)
            cells_processed.append((DISPLAY_NAMES.get(label, label), cohort_name))

    if len(all_rows) == 0:
        logger.error("No bootstrap rows produced; aborting.")
        return 1

    # ----- 3. Build per-cell distribution summaries.
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUTPUTS / "_raw_bootstrap_rows.csv", index=False)
    logger.info("wrote raw bootstrap rows: %d", len(df_all))

    # Per-cell summary for each metric.
    rf_rows: List[dict] = []
    iqpc1_rows: List[dict] = []
    d2_rows: List[dict] = []
    d2_compare_rows: List[dict] = []

    for (ckpt_disp, cohort_name), grp in df_all.groupby(["ckpt", "sub_cohort"]):
        rf_summary = summarize_distribution(grp["angle_rf_deg"].tolist())
        iqpc1_summary = summarize_distribution(grp["angle_iqpc1_deg"].tolist())
        # For D2 replicate, pool both halves together (each bootstrap contributes 2 values).
        d2_values = grp["angle_d2_h1_deg"].tolist() + grp["angle_d2_h2_deg"].tolist()
        d2_summary = summarize_distribution(d2_values)

        rf_rows.append({
            "ckpt": ckpt_disp,
            "sub_cohort": cohort_name,
            "n_bootstraps": int(len(grp)),
            **{k: rf_summary[k] for k in
               ["n_valid", "mean", "std", "p10", "p25", "p50", "p75", "p90"]},
        })
        iqpc1_rows.append({
            "ckpt": ckpt_disp,
            "sub_cohort": cohort_name,
            "n_bootstraps": int(len(grp)),
            **{k: iqpc1_summary[k] for k in
               ["n_valid", "mean", "std", "p10", "p25", "p50", "p75", "p90"]},
        })
        d2_rows.append({
            "ckpt": ckpt_disp,
            "sub_cohort": cohort_name,
            "n_bootstraps": int(len(grp)),
            **{k: d2_summary[k] for k in
               ["n_valid", "mean", "std", "p10", "p25", "p50", "p75", "p90"]},
        })

        # D2-vs-empirical comparison: where does D2's single-split angle
        # sit relative to the bootstrap "D2-replicate" distribution?
        d2_lookup_key = (ckpt_disp, cohort_name)
        d2_ang = D2_ANGLES_TO_PC1.get(d2_lookup_key, np.nan)
        if not np.isnan(d2_ang) and d2_summary["std"] > 0:
            z = (d2_ang - d2_summary["mean"]) / d2_summary["std"]
        else:
            z = np.nan
        d2_compare_rows.append({
            "ckpt": ckpt_disp,
            "sub_cohort": cohort_name,
            "d2_angle_deg": d2_ang,
            "bootstrap_mean": d2_summary["mean"],
            "bootstrap_std": d2_summary["std"],
            "bootstrap_p10": d2_summary["p10"],
            "bootstrap_p90": d2_summary["p90"],
            "z_score": z,
        })

    df_rf = pd.DataFrame(rf_rows).sort_values(["sub_cohort", "ckpt"])
    df_iqpc1 = pd.DataFrame(iqpc1_rows).sort_values(["sub_cohort", "ckpt"])
    df_d2 = pd.DataFrame(d2_rows).sort_values(["sub_cohort", "ckpt"])
    df_compare = pd.DataFrame(d2_compare_rows).sort_values(["sub_cohort", "ckpt"])

    df_rf.to_csv(OUTPUTS / "bootstrap_rf_direction_stability.csv", index=False)
    df_iqpc1.to_csv(OUTPUTS / "bootstrap_iq_pc1_stability.csv", index=False)
    df_d2.to_csv(OUTPUTS / "bootstrap_d2_replicate.csv", index=False)
    df_compare.to_csv(OUTPUTS / "d2_vs_empirical_comparison.csv", index=False)
    logger.info("wrote 4 summary CSVs")

    # ----- 4. Summary json.
    summary = {
        "B_bootstraps_target": int(B_bootstraps),
        "n_panel": int(len(panel)),
        "n_chronic_6": int(panel["is_chronic_6"].sum()),
        "ckpts_processed": sorted({c for c, _ in cells_processed}),
        "cells_processed": [list(c) for c in cells_processed],
        "elapsed_seconds": float(time.time() - t_start),
        "feature_dim": int(X_keep.shape[1]) if "X_keep" in locals() else None,
        "random_baseline_angle_deg_d768": float(
            np.degrees(np.arccos(np.sqrt(1.0 / 768)))
        ),
    }
    with open(OUTPUTS / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("D6 done in %.1fs", summary["elapsed_seconds"])

    # ----- 5. Console summary.
    print("\n=== bootstrap_rf_direction_stability.csv ===")
    print(df_rf.to_string(index=False))
    print("\n=== bootstrap_iq_pc1_stability.csv ===")
    print(df_iqpc1.to_string(index=False))
    print("\n=== bootstrap_d2_replicate.csv ===")
    print(df_d2.to_string(index=False))
    print("\n=== d2_vs_empirical_comparison.csv ===")
    print(df_compare.to_string(index=False))

    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--B", type=int, default=B_BOOTSTRAPS_DEFAULT,
                        help="Number of bootstrap iterations per cell.")
    args = parser.parse_args()
    raise SystemExit(main(B_bootstraps=args.B))
