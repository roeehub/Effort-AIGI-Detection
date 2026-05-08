"""Per-layer IQ probe (check (a) of IQ-deconvolution program, 2026-05-08).

Goal: determine where in the OpenCLIP B16 encoder the IQ representation is
concentrated.

Method:
  1. For each (ckpt, layer) cell, fit a multivariate linear probe
     `IQ_features ← layer_features` per primary_6 IQ feature individually
     (R²) and as a vector (multivariate cosine alignment).
  2. For each (ckpt, layer), fit a binary discriminator
     `is_high_lap_var ← layer_features` (split at lap_var p50) and report AUC.
     Repeat for `is_high_min_dim`, `is_high_color_b_dev`, `is_high_edge_mag`,
     `is_high_skin_frac` for completeness.

Inputs:
  - Per-layer cached features at `_cache/intermediate__{label}__layer{ix:02d}__n800.npz`
    (extracted via extract_features.py)
  - IQ atlas frame-level panel
    (`analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet`)
  - Triptych sampled-frames metadata
    (`analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/sampled_frames.csv`)
    — provides the `local_path` ↔ `gcs_uri` mapping for joining IQ features.

Joining: the triptych sample's `local_path` corresponds to a specific
`gcs_uri`. The IQ atlas's `frame_path` is a GCS URI. We use the triptych's
`gcs_uri` as the join key to attach IQ features per frame in the 800-frame
sample. Frames in the triptych sample whose URI is NOT in the IQ atlas
will have the IQ features computed inline (CV2 + LAB) so we can probe the
full 800.

Per memory `feedback_sklearn_njobs.md`, all sklearn calls use n_jobs=1.

Outputs:
  - `outputs/iq_perlayer_r2.csv`: rows = (ckpt × layer × IQ_feature). Cols:
    R² of `iq_feat ~ layer_features` via Ridge(alpha=1.0).
  - `outputs/iq_perlayer_binary_auc.csv`: rows = (ckpt × layer × IQ_feature_bin).
    Cols: AUC for binary `iq_feat > p50 ~ layer_features` via 5-fold CV LR.
  - `outputs/multivar_alignment.csv`: rows = (ckpt × layer). Cols: multivar
    R² + multivar cosine alignment (predicted-vs-actual normed).
  - `outputs/iq_perlayer_summary.json`: peak-AUC layer per ckpt, summary.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

THIS_DIR = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = THIS_DIR / "_cache"
OUTPUTS = THIS_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

ATLAS_PARQUET = (
    REPO_ROOT / "analysis" / "iq_data_atlas_2026-05-08" / "outputs" / "per_frame.parquet"
)
SAMPLED_CSV = (
    REPO_ROOT
    / "analysis"
    / "embedding_triptych_2026-04-30"
    / "outputs"
    / "triptych_p8a_slot2_slot3"
    / "sampled_frames.csv"
)

LAYERS = [0, 3, 6, 9, 11]
LABELS_TO_CKPTS = {
    "P8A": "P8A_REFERENCE_STEP5000",
    "E2B": "E2B_TOP_N_STEP3200",
    "P2D": "P2_D_FOURIER_PERIODIC_STEP3000",
}

# Primary IQ features (matches Stage 1 IQ_DECOMP_FACTS primary_6).
PRIMARY_6 = ["lap_var", "min_dim", "luma_mean", "color_b_dev", "edge_mag", "skin_frac"]

logger = logging.getLogger("iq-perlayer-probe")


def compute_iq_panel_inline(local_paths: List[str]) -> pd.DataFrame:
    """Compute the same primary_6 IQ features as the atlas, inline from local
    files. Used for frames that aren't in the GCS-keyed atlas.

    Reuses the formulas from
    `analysis/iq_data_atlas_2026-05-08/build_iq_atlas.py:per_frame_attrs`
    (kept consistent with `dor_drift_mechanism_2026-05-06/run_analysis.py:compute_iq_axes`).
    """
    import cv2

    rows = []
    for p in local_paths:
        try:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                rows.append({k: np.nan for k in PRIMARY_6})
                continue
            h, w = img.shape[:2]
            min_dim = float(min(h, w))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            v = hsv[..., 2].astype(np.float32)
            luma_mean = float(v.mean())
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_mag = float(np.sqrt(sx**2 + sy**2).mean())
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            color_b_dev = float(np.abs(lab[..., 2] - 128.0).mean())
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            cr = ycrcb[..., 1]
            cb = ycrcb[..., 2]
            y_chan = ycrcb[..., 0]
            skin = (
                (y_chan > 80)
                & (cr >= 133) & (cr <= 173)
                & (cb >= 77) & (cb <= 127)
            )
            skin_frac = float(skin.mean())
            rows.append({
                "lap_var": lap_var,
                "min_dim": min_dim,
                "luma_mean": luma_mean,
                "color_b_dev": color_b_dev,
                "edge_mag": edge_mag,
                "skin_frac": skin_frac,
            })
        except Exception:
            rows.append({k: np.nan for k in PRIMARY_6})
    return pd.DataFrame(rows)


def build_iq_panel_for_sample() -> pd.DataFrame:
    """Build per-frame IQ panel for the 800-frame triptych sample.

    Strategy:
      1. Try to join the triptych's gcs_uri against the IQ atlas's frame_path.
      2. For frames not in the atlas, compute IQ features inline.
    """
    sampled = pd.read_csv(
        SAMPLED_CSV,
        usecols=[
            "gcs_uri",
            "label",
            "split",
            "identity_key",
            "session_id",
            "video_id",
            "method",
            "local_path",
            "sharpness_laplacian",
            "brightness_v_mean",
            "face_pixel_area",
        ],
    )
    sampled = sampled.iloc[:800].reset_index(drop=True)
    sampled["row_ix"] = np.arange(len(sampled))

    atlas = pd.read_parquet(ATLAS_PARQUET)[["frame_path"] + PRIMARY_6].copy()
    # Merge: triptych gcs_uri vs atlas frame_path (both GCS URIs).
    merged = sampled.merge(
        atlas,
        left_on="gcs_uri",
        right_on="frame_path",
        how="left",
    )
    n_in_atlas = merged["lap_var"].notna().sum()
    n_missing = len(merged) - n_in_atlas
    logger.info("triptych ∩ IQ atlas: %d frames in atlas, %d need inline computation",
                n_in_atlas, n_missing)

    if n_missing > 0:
        missing_idx = merged[merged["lap_var"].isna()].index
        logger.info("computing IQ features inline for %d frames", len(missing_idx))
        inline = compute_iq_panel_inline(merged.loc[missing_idx, "local_path"].tolist())
        for col in PRIMARY_6:
            merged.loc[missing_idx, col] = inline[col].values
    return merged


def load_layer_features(label: str, layer_ix: int, n: int = 800) -> tuple[np.ndarray, np.ndarray]:
    p = LOCAL_CACHE_DIR / f"intermediate__{label}__layer{layer_ix:02d}__n{n}.npz"
    blob = np.load(p)
    return blob["features"].astype(np.float32), blob["valid_idx"].astype(np.int64)


def fit_ridge_r2(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[float, np.ndarray]:
    """5-fold CV Ridge regression. Returns (CV-R², held-out predictions)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros_like(y, dtype=np.float64)
    for tr, te in kf.split(X):
        clf = Ridge(alpha=alpha)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict(X[te])
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float(((y - oof) ** 2).sum())
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return r2, oof


def fit_logistic_auc(X: np.ndarray, y: np.ndarray) -> float:
    """5-fold CV logistic-regression AUC. n_jobs=1 per project memory."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    logger.info("building IQ panel for 800-frame triptych sample…")
    panel = build_iq_panel_for_sample()
    logger.info("panel rows=%d, IQ NaN counts:\n%s", len(panel), panel[PRIMARY_6].isna().sum().to_string())

    # Drop rows with any IQ NaN (safest).
    keep = panel[PRIMARY_6].notna().all(axis=1).values
    n_keep = int(keep.sum())
    logger.info("kept %d/%d frames with full IQ panel", n_keep, len(panel))
    panel_keep = panel[keep].reset_index(drop=True)
    keep_row_ix = panel_keep["row_ix"].values

    # Standardize IQ features (within sample).
    iq_arr = panel_keep[PRIMARY_6].values.astype(np.float32)
    iq_mean = iq_arr.mean(axis=0, keepdims=True)
    iq_std = iq_arr.std(axis=0, keepdims=True) + 1e-12
    iq_z = (iq_arr - iq_mean) / iq_std
    # Binary p50 splits for AUC.
    iq_bin = (iq_arr > np.median(iq_arr, axis=0, keepdims=True)).astype(np.int32)

    r2_rows = []
    auc_rows = []
    multivar_rows = []
    summary_per_ckpt = {}

    for label, ckpt_key in LABELS_TO_CKPTS.items():
        logger.info("=== %s (%s) ===", label, ckpt_key)
        per_layer_peak_auc = {}
        per_layer_avg_r2 = {}

        for ix in LAYERS:
            feats, valid_idx = load_layer_features(label, ix)
            # Align: select only rows present in panel_keep.
            valid_to_pos = {int(v): r for r, v in enumerate(valid_idx)}
            sel_pos = [valid_to_pos[int(rx)] for rx in keep_row_ix if int(rx) in valid_to_pos]
            X = feats[sel_pos]
            # also subset panel to matching rows
            panel_aligned = panel_keep.loc[
                [i for i, rx in enumerate(keep_row_ix) if int(rx) in valid_to_pos]
            ].reset_index(drop=True)
            iq_arr_l = panel_aligned[PRIMARY_6].values.astype(np.float32)
            iq_z_l = (iq_arr_l - iq_arr_l.mean(axis=0)) / (iq_arr_l.std(axis=0) + 1e-12)
            iq_bin_l = (iq_arr_l > np.median(iq_arr_l, axis=0)).astype(np.int32)

            n_aligned = X.shape[0]
            if n_aligned < 50:
                logger.warning("[%s layer=%d] only %d frames; skipping", label, ix, n_aligned)
                continue

            # L2-normalize layer features (cosine-style) — improves Ridge stability.
            X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

            # Per-feature Ridge R²
            r2_per_feat = {}
            preds_per_feat = {}
            for k, fname in enumerate(PRIMARY_6):
                r2, preds = fit_ridge_r2(X_n, iq_z_l[:, k], alpha=1.0)
                r2_per_feat[fname] = r2
                preds_per_feat[fname] = preds
                r2_rows.append({
                    "ckpt": label,
                    "layer": ix,
                    "iq_feature": fname,
                    "r2": r2,
                    "n": n_aligned,
                })

            # Multivariate alignment: predicted IQ vector vs actual IQ vector.
            P = np.stack([preds_per_feat[f] for f in PRIMARY_6], axis=1)
            # Average per-feature R² as scalar summary.
            mvar_r2_avg = float(np.mean([r2_per_feat[f] for f in PRIMARY_6]))
            # Cosine alignment per row, then mean.
            P_n = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
            T = iq_z_l
            T_n = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
            cos_align = float(np.einsum("ij,ij->i", P_n, T_n).mean())
            multivar_rows.append({
                "ckpt": label,
                "layer": ix,
                "n": n_aligned,
                "avg_per_feature_r2": mvar_r2_avg,
                "vector_cosine_alignment": cos_align,
            })

            # Binary AUC per IQ axis (high vs low p50 split).
            peak_layer_auc = 0.0
            for k, fname in enumerate(PRIMARY_6):
                auc = fit_logistic_auc(X_n, iq_bin_l[:, k])
                auc_rows.append({
                    "ckpt": label,
                    "layer": ix,
                    "iq_feature_bin": fname,
                    "auc": auc,
                    "n": n_aligned,
                })
                if auc > peak_layer_auc:
                    peak_layer_auc = auc
            per_layer_peak_auc[ix] = peak_layer_auc
            per_layer_avg_r2[ix] = mvar_r2_avg
            logger.info("  layer=%d  avg_per_feat_R²=%.3f  cos_align=%.3f  peak_bin_AUC=%.3f",
                        ix, mvar_r2_avg, cos_align, peak_layer_auc)

        if per_layer_peak_auc:
            peak_layer = int(max(per_layer_peak_auc, key=per_layer_peak_auc.get))
            summary_per_ckpt[label] = {
                "peak_layer_by_avg_bin_auc": peak_layer,
                "per_layer_peak_auc": per_layer_peak_auc,
                "per_layer_avg_r2": per_layer_avg_r2,
            }

    # Save outputs.
    pd.DataFrame(r2_rows).to_csv(OUTPUTS / "iq_perlayer_r2.csv", index=False)
    pd.DataFrame(auc_rows).to_csv(OUTPUTS / "iq_perlayer_binary_auc.csv", index=False)
    pd.DataFrame(multivar_rows).to_csv(OUTPUTS / "multivar_alignment.csv", index=False)
    with open(OUTPUTS / "iq_perlayer_summary.json", "w") as f:
        json.dump({"per_ckpt": summary_per_ckpt}, f, indent=2)

    logger.info("outputs saved to %s", OUTPUTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
