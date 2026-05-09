"""D1 — L11 (and other layers) shortcut-subspace ablation.

For each (ckpt × layer) cell with cached features:
  1. Build a shortcut-features matrix X_short = [n × k] from named axes:
     {lap_var, min_dim, luma_mean, color_b_dev, edge_mag, skin_frac,
      face_pixel_area, is_dor, is_chronic_6}.
  2. Z-score X_short.
  3. Residualize encoder features against shortcut features:
     X_enc_res = X_enc - X_short @ pinv(X_short^T X_short) @ X_short^T @ X_enc
     (linear projection of L11 onto orthogonal complement of shortcut span)
  4. Fit logistic regression for real/fake on:
     (a) raw X_enc (baseline)
     (b) X_enc_res (residualized)
     (c) X_short alone (lower bound — what the shortcuts alone predict)
  5. Compare AUCs.

Reading: if (b) >> (c), there is encoder signal beyond shortcuts. If (b) ≈ (c),
the encoder's discrimination is the shortcut combination.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
LOCAL_CACHE = THIS_DIR / "_cache"
PRIOR_PERLAYER_CACHE = REPO_ROOT / "analysis" / "iq_perlayer_probe_2026-05-08" / "_cache"
OUTPUTS = THIS_DIR / "outputs"

REFERENCE_CKPTS = ["P8A", "E2B", "P2D"]
STAGE2_CKPTS = [
    "S1_step500", "S1_step2500", "S1_step4500",
    "S2_step500", "S2_step2500", "S2_step4500",
    "S3_step500", "S3_step2500", "S3_step4500",
]
ALL_CKPTS = REFERENCE_CKPTS + STAGE2_CKPTS
LAYERS = [0, 3, 6, 9, 11]

CHRONIC_6_PATTERNS = ["Roy_D", "PC_Generator", "bla_bla_chow",
                     "Md_noyn_Sharker", "dor_shkedi", "healthy_dor"]
PRIMARY_6 = ["lap_var", "min_dim", "luma_mean", "color_b_dev", "edge_mag", "skin_frac"]

logger = logging.getLogger("d1-shortcut-ablation")


def feature_path(ckpt: str, layer: int, n: int = 800) -> Path:
    p_local = LOCAL_CACHE / f"intermediate__{ckpt}__layer{layer:02d}__n{n}.npz"
    if p_local.exists():
        return p_local
    return PRIOR_PERLAYER_CACHE / f"intermediate__{ckpt}__layer{layer:02d}__n{n}.npz"


def chronic_match(s) -> bool:
    if not isinstance(s, str):
        return False
    return any(p.lower() in s.lower() for p in CHRONIC_6_PATTERNS)


def is_dor(s) -> bool:
    if not isinstance(s, str):
        return False
    return "dor" in s.lower()


def build_panel() -> pd.DataFrame:
    """Re-build the same 800-frame panel used by run_analyses.py, with
    the IQ axes joined."""
    sampled = pd.read_csv(
        REPO_ROOT / "analysis" / "embedding_triptych_2026-04-30"
        / "outputs" / "triptych_p8a_slot2_slot3" / "sampled_frames.csv",
        usecols=["gcs_uri", "label", "split", "identity_key", "method",
                 "local_path", "face_pixel_area", "width", "height", "is_no_face"],
    ).iloc[:800].reset_index(drop=True)
    sampled["row_ix"] = np.arange(len(sampled))

    atlas = pd.read_parquet(
        REPO_ROOT / "analysis" / "iq_data_atlas_2026-05-08" / "outputs" / "per_frame.parquet"
    )[["frame_path"] + PRIMARY_6].drop_duplicates(subset=["frame_path"])
    merged = sampled.merge(atlas, left_on="gcs_uri", right_on="frame_path", how="left")

    # Inline-fill missing IQ.
    missing = merged["lap_var"].isna()
    if missing.any():
        import cv2
        for ix in merged.index[missing]:
            try:
                img = cv2.imread(str(merged.loc[ix, "local_path"]), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                luma_mean = float(hsv[..., 2].astype(np.float32).mean())
                sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                edge_mag = float(np.sqrt(sx**2 + sy**2).mean())
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
                color_b_dev = float(np.abs(lab[..., 2] - 128.0).mean())
                ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                skin = ((ycrcb[..., 0] > 80) & (ycrcb[..., 1] >= 133) &
                        (ycrcb[..., 1] <= 173) & (ycrcb[..., 2] >= 77) &
                        (ycrcb[..., 2] <= 127))
                merged.loc[ix, "lap_var"] = lap_var
                merged.loc[ix, "min_dim"] = float(min(h, w))
                merged.loc[ix, "luma_mean"] = luma_mean
                merged.loc[ix, "edge_mag"] = edge_mag
                merged.loc[ix, "color_b_dev"] = color_b_dev
                merged.loc[ix, "skin_frac"] = float(skin.mean())
            except Exception:
                pass

    nan_min = merged["min_dim"].isna()
    if nan_min.any():
        merged.loc[nan_min, "min_dim"] = merged.loc[nan_min, ["width", "height"]].min(axis=1)

    merged["is_dor"] = merged["identity_key"].apply(is_dor).astype(int)
    merged["is_chronic_6"] = merged["identity_key"].apply(chronic_match).astype(int)
    merged["is_fake_int"] = (merged["label"] == "fake").astype(int)
    return merged


def fit_lr_auc(X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    if X.size == 0 or X.shape[1] == 0:
        return float("nan")
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return float("nan")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(len(y), dtype=np.float64)
    try:
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(C=1.0, max_iter=3000, n_jobs=1, solver="lbfgs")
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        return float(roc_auc_score(y, oof))
    except Exception:
        return float("nan")


def residualize(X_enc: np.ndarray, X_short: np.ndarray) -> np.ndarray:
    """Project X_enc onto the orthogonal complement of X_short's column span."""
    # Normalize X_short columns (already z-scored upstream typically).
    # Compute hat = X_short @ pinv(X_short).
    # X_enc_res = X_enc - hat @ X_enc = (I - hat) @ X_enc.
    # Numerically stable via lstsq.
    coefs, *_ = np.linalg.lstsq(X_short, X_enc, rcond=None)
    pred = X_short @ coefs  # n × d
    return X_enc - pred


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    panel = build_panel()
    logger.info("panel rows=%d", len(panel))

    # Build shortcut feature matrix (with intercept).
    short_cols = PRIMARY_6 + ["face_pixel_area", "is_dor", "is_chronic_6"]
    short_panel = panel[short_cols + ["row_ix", "is_fake_int"]].dropna()
    short_panel = short_panel.reset_index(drop=True)
    logger.info("non-null shortcut rows=%d (dropped %d for NaN)",
                len(short_panel), len(panel) - len(short_panel))

    X_short_raw = short_panel[short_cols].values.astype(np.float64)
    # z-score continuous cols; binary cols already 0/1.
    means = X_short_raw.mean(axis=0)
    stds = X_short_raw.std(axis=0) + 1e-12
    X_short_z = (X_short_raw - means) / stds
    # Append intercept column.
    X_short = np.hstack([X_short_z, np.ones((len(X_short_z), 1))])

    y = short_panel["is_fake_int"].values.astype(int)
    keep_row_ix = short_panel["row_ix"].values

    # Baseline: AUC of shortcut features alone for real/fake.
    shortcut_alone_auc = fit_lr_auc(X_short, y)
    logger.info("shortcut-only AUC for real/fake: %.4f", shortcut_alone_auc)

    rows = []
    for ckpt in ALL_CKPTS:
        for layer in LAYERS:
            try:
                blob = np.load(feature_path(ckpt, layer))
                feats = blob["features"].astype(np.float32)
                valid_idx = blob["valid_idx"].astype(np.int64)
            except Exception as exc:
                logger.warning("%s L%d feature load failed: %s", ckpt, layer, exc)
                continue
            valid_to_pos = {int(v): r for r, v in enumerate(valid_idx)}
            sel_pos = [valid_to_pos[int(rx)] for rx in keep_row_ix
                       if int(rx) in valid_to_pos]
            if len(sel_pos) < len(keep_row_ix) * 0.8:
                logger.warning("%s L%d alignment poor (%d/%d), skipping",
                               ckpt, layer, len(sel_pos), len(keep_row_ix))
                continue
            X_enc = feats[sel_pos]
            mask_keep = np.array([int(rx) in valid_to_pos for rx in keep_row_ix])
            y_aligned = y[mask_keep]
            X_short_aligned = X_short[mask_keep]

            # L2-normalize encoder features (cosine-style; matches earlier probes).
            X_enc_n = X_enc / (np.linalg.norm(X_enc, axis=1, keepdims=True) + 1e-12)
            # Residualize against shortcut features.
            X_enc_res = residualize(X_enc_n, X_short_aligned)
            # Renormalize the residual.
            X_enc_res_n = X_enc_res / (np.linalg.norm(X_enc_res, axis=1, keepdims=True) + 1e-12)

            auc_raw = fit_lr_auc(X_enc_n, y_aligned)
            auc_res = fit_lr_auc(X_enc_res_n, y_aligned)
            rows.append({
                "ckpt": ckpt, "layer": layer,
                "n": int(len(y_aligned)),
                "auc_raw_encoder": auc_raw,
                "auc_residualized": auc_res,
                "auc_shortcut_only": shortcut_alone_auc,
                "auc_drop_after_residualize": auc_raw - auc_res,
            })
            logger.info("[%s L%d] raw=%.3f res=%.3f drop=%.3f",
                        ckpt, layer, auc_raw, auc_res, auc_raw - auc_res)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS / "d1_shortcut_subspace_ablation.csv", index=False)
    logger.info("wrote %s (%d rows)", OUTPUTS / "d1_shortcut_subspace_ablation.csv", len(df))

    # Headlines.
    print("\n" + "=" * 80)
    print("D1 — RESIDUALIZED encoder AUC for real/fake at each layer")
    print(f"     (shortcut-only AUC reference = {shortcut_alone_auc:.4f})")
    print("=" * 80)
    pd.options.display.float_format = "{:.4f}".format
    pivot = df.pivot_table(index="ckpt", columns="layer", values="auc_residualized")
    print(pivot.reindex(index=ALL_CKPTS).to_string())

    print("\n" + "=" * 80)
    print("D1 — AUC drop after residualization (raw - residualized) per layer")
    print("=" * 80)
    drop_pivot = df.pivot_table(index="ckpt", columns="layer",
                                 values="auc_drop_after_residualize")
    print(drop_pivot.reindex(index=ALL_CKPTS).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
