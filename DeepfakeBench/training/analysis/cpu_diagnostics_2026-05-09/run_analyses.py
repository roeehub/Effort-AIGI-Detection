"""Combined CPU analysis on the 800-frame triptych.

Reads:
  - cached features at _cache/intermediate__{label}__layer{XX}__n800.npz
    (12 ckpts: P8A, E2B, P2D + 9 Stage 2; 5 layers each)
  - scores at outputs/stage2_triptych_scores.csv
  - metadata from analysis/embedding_triptych_2026-04-30/.../sampled_frames.csv

Produces (under outputs/):
  - forgery_signal_atlas.csv  (ckpt × layer × signal × AUC)
  - score_per_identity_per_ckpt.csv
  - cross_substrate_variance.csv
  - f4_filter_triptych.csv
  - iq_decomp_triptych.csv
  - ANALYSIS_FACTS_2026-05-09.md (factual aggregation)

Per memory feedback_sklearn_njobs.md: n_jobs=1 throughout.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
LOCAL_CACHE = THIS_DIR / "_cache"
PRIOR_PERLAYER_CACHE = REPO_ROOT / "analysis" / "iq_perlayer_probe_2026-05-08" / "_cache"
OUTPUTS = THIS_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

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

# Reference ckpts (cached at iq_perlayer_probe_2026-05-08/_cache).
REFERENCE_CKPTS = ["P8A", "E2B", "P2D"]
STAGE2_CKPTS = [
    "S1_step500", "S1_step2500", "S1_step4500",
    "S2_step500", "S2_step2500", "S2_step4500",
    "S3_step500", "S3_step2500", "S3_step4500",
]
ALL_CKPTS = REFERENCE_CKPTS + STAGE2_CKPTS
LAYERS = [0, 3, 6, 9, 11]

# Chronic-6 substring patterns (per memory).
CHRONIC_6_PATTERNS = [
    "Roy_D", "PC_Generator", "bla_bla_chow",
    "Md_noyn_Sharker", "dor_shkedi", "healthy_dor",
]

PRIMARY_6 = ["lap_var", "min_dim", "luma_mean", "color_b_dev", "edge_mag", "skin_frac"]

logger = logging.getLogger("cpu-diagnostics")


# ============================================================================
# Cache helpers
# ============================================================================
def feature_path(ckpt: str, layer: int, n: int = 800) -> Path:
    """Resolve feature .npz for either reference or Stage 2 ckpt."""
    p_local = LOCAL_CACHE / f"intermediate__{ckpt}__layer{layer:02d}__n{n}.npz"
    if p_local.exists():
        return p_local
    p_prior = PRIOR_PERLAYER_CACHE / f"intermediate__{ckpt}__layer{layer:02d}__n{n}.npz"
    if p_prior.exists():
        return p_prior
    return p_local  # will fail on np.load


def load_features(ckpt: str, layer: int, n: int = 800) -> tuple[np.ndarray, np.ndarray]:
    p = feature_path(ckpt, layer, n)
    blob = np.load(p)
    return blob["features"].astype(np.float32), blob["valid_idx"].astype(np.int64)


# ============================================================================
# Build the metadata + IQ panel for the 800-frame triptych
# ============================================================================
def chronic_match(s) -> bool:
    if not isinstance(s, str):
        return False
    sl = s.lower()
    return any(p.lower() in sl for p in CHRONIC_6_PATTERNS)


def is_dor(identity_key) -> bool:
    if not isinstance(identity_key, str):
        return False
    return "dor" in identity_key.lower()


def build_panel() -> pd.DataFrame:
    sampled = pd.read_csv(
        SAMPLED_CSV,
        usecols=[
            "gcs_uri", "label", "split", "identity_key", "session_id", "video_id",
            "method", "local_path", "sharpness_laplacian", "brightness_v_mean",
            "face_pixel_area", "width", "height", "is_no_face",
        ],
    ).iloc[:800].reset_index(drop=True)
    sampled["row_ix"] = np.arange(len(sampled))

    # Try to attach atlas IQ features.
    try:
        atlas = pd.read_parquet(ATLAS_PARQUET)[["frame_path"] + PRIMARY_6].copy()
        merged = sampled.merge(atlas, left_on="gcs_uri", right_on="frame_path", how="left")
    except Exception as exc:
        logger.warning("atlas not joinable (%s); inline computation falls back", exc)
        merged = sampled.copy()
        for c in PRIMARY_6:
            merged[c] = np.nan

    # Compute IQ features inline for any missing rows.
    missing = merged["lap_var"].isna()
    if missing.any():
        logger.info("computing IQ inline for %d frames missing from atlas", int(missing.sum()))
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

    # If atlas's min_dim was filled, prefer it; otherwise fill from width/height.
    nan_min = merged["min_dim"].isna()
    if nan_min.any():
        merged.loc[nan_min, "min_dim"] = merged.loc[nan_min, ["width", "height"]].min(axis=1)

    # Derived flags.
    merged["is_dor"] = merged["identity_key"].apply(is_dor).astype(int)
    merged["is_chronic_6"] = merged["identity_key"].apply(chronic_match).astype(int)
    merged["is_lockbox"] = (merged["split"] == "lockbox").astype(int)
    return merged


# ============================================================================
# (A) Forgery-signal atlas: per-layer × per-signal AUC matrix
# ============================================================================
def compute_signal_atlas(panel: pd.DataFrame) -> pd.DataFrame:
    """For each ckpt × layer, fit LR probes for {real_vs_fake, is_dor, is_chronic_6,
    lap_var_high, min_dim_high, face_size_high}. Return long table of AUCs.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    # Build target columns.
    panel = panel.copy()
    panel["is_real_vs_fake"] = (panel["label"] == "fake").astype(int)
    # Use sample-conditional medians so each cell has balanced targets.
    panel["lap_var_high"] = (panel["lap_var"] > panel["lap_var"].median()).astype(int)
    panel["min_dim_high"] = (panel["min_dim"] > panel["min_dim"].median()).astype(int)
    if panel["face_pixel_area"].notna().any():
        panel["face_size_high"] = (
            panel["face_pixel_area"] > panel["face_pixel_area"].median()
        ).astype(int)
    else:
        panel["face_size_high"] = 0
    panel["is_chronic_6_int"] = panel["is_chronic_6"].astype(int)

    SIGNALS = [
        ("is_real_vs_fake", "is_real_vs_fake"),
        ("is_dor", "is_dor"),
        ("is_chronic_6", "is_chronic_6_int"),
        ("lap_var_high", "lap_var_high"),
        ("min_dim_high", "min_dim_high"),
        ("face_size_high", "face_size_high"),
    ]

    rows = []
    for ckpt in ALL_CKPTS:
        for layer in LAYERS:
            try:
                feats, valid_idx = load_features(ckpt, layer)
            except Exception as exc:
                logger.warning("%s layer=%d feature load failed: %s", ckpt, layer, exc)
                continue
            # Align panel to valid_idx.
            valid_set = set(int(v) for v in valid_idx.tolist())
            mask = panel["row_ix"].isin(valid_set).values
            panel_aligned = panel[mask].reset_index(drop=True)
            valid_to_pos = {int(v): r for r, v in enumerate(valid_idx)}
            sel_pos = [valid_to_pos[int(rx)] for rx in panel_aligned["row_ix"].values]
            X = feats[sel_pos]
            X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

            for sig_name, sig_col in SIGNALS:
                if sig_col not in panel_aligned.columns:
                    continue
                y = panel_aligned[sig_col].values.astype(int)
                # Filter to non-null rows.
                valid_y = ~pd.isna(y)
                if valid_y.sum() < 20:
                    continue
                X_y = X_n[valid_y]
                y_y = y[valid_y]
                if y_y.sum() < 5 or (len(y_y) - y_y.sum()) < 5:
                    # Not enough positives or negatives — skip.
                    rows.append({
                        "ckpt": ckpt, "layer": layer, "signal": sig_name,
                        "n": int(len(y_y)),
                        "n_pos": int(y_y.sum()), "auc": float("nan"),
                    })
                    continue
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
                oof = np.zeros(len(y_y), dtype=np.float64)
                try:
                    for tr, te in skf.split(X_y, y_y):
                        clf = LogisticRegression(C=1.0, max_iter=3000, n_jobs=1, solver="lbfgs")
                        clf.fit(X_y[tr], y_y[tr])
                        oof[te] = clf.predict_proba(X_y[te])[:, 1]
                    auc = float(roc_auc_score(y_y, oof))
                except Exception as exc:
                    logger.warning("%s layer=%d signal=%s probe failed: %s",
                                   ckpt, layer, sig_name, exc)
                    auc = float("nan")
                rows.append({
                    "ckpt": ckpt, "layer": layer, "signal": sig_name,
                    "n": int(len(y_y)),
                    "n_pos": int(y_y.sum()), "auc": auc,
                })
        logger.info("[%s] forgery-signal atlas done", ckpt)
    return pd.DataFrame(rows)


# ============================================================================
# (B) Score-based analyses
# ============================================================================
def score_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Join the cached scores file + reference-ckpt scores from elsewhere."""
    triptych_scores = pd.read_csv(THIS_DIR / "outputs" / "stage2_triptych_scores.csv")
    # The triptych_scores already has Stage 2 per-ckpt cols; merge into panel.
    out = panel.merge(triptych_scores[["row_ix"] + STAGE2_CKPTS],
                      on="row_ix", how="left")
    # Reference scores: pull from the existing iq_perlayer cached features?
    # We don't have P8A/E2B/P2D scores on the triptych cached. Need to
    # extract them inline, but skipping for now; analyses that DO need them
    # will read from the original score sources.
    return out


def f4_filter_recompute(panel_with_scores: pd.DataFrame) -> pd.DataFrame:
    """Apply F4-style cleaning. Compute real-FPR @ τ=0.5 per ckpt for each filter."""
    rows = []
    df = panel_with_scores.copy()
    df["is_fake"] = (df["label"] == "fake").astype(int)
    df["min_dim_eff"] = df[["width", "height"]].min(axis=1)

    filters = {
        "F0_full": np.ones(len(df), dtype=bool),
        "F1_no_chronic_6": (df["is_chronic_6"] == 0).values,
        "F2_no_chronic_no_noface": ((df["is_chronic_6"] == 0)
                                     & (df["is_no_face"].fillna(False) == False)).values,
        "F4_no_chronic_no_noface_min_dim_ge200": (
            (df["is_chronic_6"] == 0)
            & (df["is_no_face"].fillna(False) == False)
            & (df["min_dim_eff"] >= 200)
        ).values,
    }
    for tag, mask in filters.items():
        sub = df[mask]
        n_real = int((sub["is_fake"] == 0).sum())
        n_fake = int((sub["is_fake"] == 1).sum())
        for ckpt in STAGE2_CKPTS:
            if ckpt not in sub.columns:
                continue
            scores = sub[ckpt].values
            real_mask = (sub["is_fake"] == 0).values
            fake_mask = (sub["is_fake"] == 1).values
            real_fpr = float((scores[real_mask] >= 0.5).mean()) if real_mask.sum() > 0 else float("nan")
            fake_recall = float((scores[fake_mask] >= 0.5).mean()) if fake_mask.sum() > 0 else float("nan")
            real_p50 = float(np.median(scores[real_mask])) if real_mask.sum() > 0 else float("nan")
            fake_p50 = float(np.median(scores[fake_mask])) if fake_mask.sum() > 0 else float("nan")
            rows.append({
                "filter": tag, "ckpt": ckpt,
                "n": len(sub), "n_real": n_real, "n_fake": n_fake,
                "real_fpr_at_0p5": real_fpr,
                "fake_recall_at_0p5": fake_recall,
                "real_p50_score": real_p50,
                "fake_p50_score": fake_p50,
            })
    return pd.DataFrame(rows)


def cross_substrate_variance(panel_with_scores: pd.DataFrame) -> pd.DataFrame:
    """For each identity present in BOTH dev and lockbox splits, compute the
    score difference per ckpt (median lockbox - median dev). High |delta| =
    substrate-sensitive ckpt for that identity.
    """
    df = panel_with_scores.copy()
    rows = []
    # Per-ckpt: per-identity-and-substrate medians.
    for ckpt in STAGE2_CKPTS:
        if ckpt not in df.columns:
            continue
        per = df.groupby(["identity_key", "split"])[ckpt].agg(
            ["count", "median", "std"]
        ).reset_index()
        # Pivot to dev vs lockbox.
        dev = per[per["split"] == "dev"].set_index("identity_key")
        lbx = per[per["split"] == "lockbox"].set_index("identity_key")
        common = sorted(set(dev.index) & set(lbx.index))
        for ident in common:
            d_med = dev.loc[ident, "median"]
            l_med = lbx.loc[ident, "median"]
            d_n = int(dev.loc[ident, "count"])
            l_n = int(lbx.loc[ident, "count"])
            rows.append({
                "ckpt": ckpt, "identity_key": ident,
                "n_dev": d_n, "n_lockbox": l_n,
                "median_dev": float(d_med), "median_lockbox": float(l_med),
                "delta_lockbox_minus_dev": float(l_med - d_med),
            })
    return pd.DataFrame(rows)


def per_identity_score(panel_with_scores: pd.DataFrame) -> pd.DataFrame:
    """For each ckpt, compute per-identity score median + count, marking is_dor /
    is_chronic_6 / label-mix.
    """
    df = panel_with_scores.copy()
    df["is_real"] = (df["label"] == "real").astype(int)
    rows = []
    for ckpt in STAGE2_CKPTS:
        if ckpt not in df.columns:
            continue
        for ident in df["identity_key"].dropna().unique():
            sub = df[df["identity_key"] == ident]
            real_n = int(sub["is_real"].sum())
            fake_n = len(sub) - real_n
            scores = sub[ckpt].values
            real_med = float(np.median(scores[sub["is_real"] == 1])) if real_n > 0 else float("nan")
            fake_med = float(np.median(scores[sub["is_real"] == 0])) if fake_n > 0 else float("nan")
            rows.append({
                "ckpt": ckpt, "identity_key": ident,
                "n_real": real_n, "n_fake": fake_n,
                "real_median": real_med, "fake_median": fake_med,
                "is_dor": int(is_dor(ident)),
                "is_chronic_6": int(chronic_match(ident)),
            })
    return pd.DataFrame(rows)


# ============================================================================
# (C) IQ decomp on triptych for Stage 2 ckpts
# ============================================================================
def iq_decomp_triptych(panel_with_scores: pd.DataFrame) -> pd.DataFrame:
    """Stage 1-style decomp: regress score on PRIMARY_6 IQ axes per (ckpt × pool).
    Only Stage 2 ckpts have triptych scores cached locally; reference ckpts'
    decomp is in the existing iq_shortcut_decomp_2026-05-08 outputs.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import roc_auc_score

    df = panel_with_scores.copy()
    df["is_fake_int"] = (df["label"] == "fake").astype(int)
    pools = {
        "ALL": np.ones(len(df), dtype=bool),
        "DEV": (df["split"] == "dev").values,
        "LOCKBOX": (df["split"] == "lockbox").values,
        "DEV_NO_CHRONIC": ((df["split"] == "dev") & (df["is_chronic_6"] == 0)).values,
    }
    rows = []
    for ckpt in STAGE2_CKPTS:
        if ckpt not in df.columns:
            continue
        for pool_name, mask in pools.items():
            sub = df[mask]
            sub_iq = sub[PRIMARY_6 + [ckpt, "is_fake_int"]].dropna()
            if len(sub_iq) < 30:
                continue
            X = sub_iq[PRIMARY_6].values
            X_z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
            y = sub_iq[ckpt].values
            try:
                lr = LinearRegression(n_jobs=1)
                lr.fit(X_z, y)
                pred = lr.predict(X_z)
                ss_res = float(((y - pred) ** 2).sum())
                ss_tot = float(((y - y.mean()) ** 2).sum())
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                resid = y - pred
            except Exception:
                continue
            # Discriminative: residual AUC (fake vs real on residual).
            labels = sub_iq["is_fake_int"].values
            if labels.sum() < 5 or (len(labels) - labels.sum()) < 5:
                resid_auc = float("nan")
                raw_auc = float("nan")
            else:
                try:
                    resid_auc = float(roc_auc_score(labels, resid))
                    raw_auc = float(roc_auc_score(labels, y))
                except Exception:
                    resid_auc = float("nan")
                    raw_auc = float("nan")
            rows.append({
                "ckpt": ckpt, "pool": pool_name, "n": int(len(sub_iq)),
                "iq_r2": r2, "raw_auc": raw_auc, "resid_auc": resid_auc,
                "delta_auc": (raw_auc - resid_auc) if (np.isfinite(raw_auc) and np.isfinite(resid_auc)) else float("nan"),
            })
    return pd.DataFrame(rows)


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    panel = build_panel()
    logger.info("panel rows=%d (label dist: %s, split dist: %s)",
                len(panel),
                panel["label"].value_counts().to_dict(),
                panel["split"].value_counts().to_dict())

    # (A) Forgery-signal atlas.
    logger.info("=== (A) Forgery-signal atlas ===")
    atlas_df = compute_signal_atlas(panel)
    atlas_df.to_csv(OUTPUTS / "forgery_signal_atlas.csv", index=False)
    logger.info("wrote forgery_signal_atlas.csv (%d rows)", len(atlas_df))

    # (B) Score-based.
    panel_w_scores = score_table(panel)
    logger.info("=== (B) F4 filter recompute ===")
    f4_df = f4_filter_recompute(panel_w_scores)
    f4_df.to_csv(OUTPUTS / "f4_filter_triptych.csv", index=False)
    logger.info("wrote f4_filter_triptych.csv (%d rows)", len(f4_df))

    logger.info("=== (B') cross-substrate variance ===")
    cs_df = cross_substrate_variance(panel_w_scores)
    cs_df.to_csv(OUTPUTS / "cross_substrate_variance.csv", index=False)
    logger.info("wrote cross_substrate_variance.csv (%d rows)", len(cs_df))

    logger.info("=== (B'') per-identity scores ===")
    pi_df = per_identity_score(panel_w_scores)
    pi_df.to_csv(OUTPUTS / "score_per_identity_per_ckpt.csv", index=False)
    logger.info("wrote score_per_identity_per_ckpt.csv (%d rows)", len(pi_df))

    # (C) IQ decomp for Stage 2 ckpts on triptych.
    logger.info("=== (C) IQ decomp on triptych (Stage 2 only) ===")
    iq_df = iq_decomp_triptych(panel_w_scores)
    iq_df.to_csv(OUTPUTS / "iq_decomp_triptych.csv", index=False)
    logger.info("wrote iq_decomp_triptych.csv (%d rows)", len(iq_df))

    # ====================================================================
    # Console headlines
    # ====================================================================
    pd.options.display.float_format = "{:.4f}".format
    print("\n" + "=" * 80)
    print("FORGERY-SIGNAL ATLAS — peak AUC per (ckpt, signal)")
    print("=" * 80)
    pa = atlas_df.dropna(subset=["auc"])
    peak = pa.loc[pa.groupby(["ckpt", "signal"])["auc"].idxmax()]
    pivot = peak.pivot_table(index="ckpt", columns="signal", values="auc")
    pivot = pivot.reindex(index=ALL_CKPTS)
    print(pivot.to_string())

    print("\n" + "=" * 80)
    print("FORGERY-SIGNAL ATLAS — at L11 only, per ckpt")
    print("=" * 80)
    l11 = atlas_df[atlas_df["layer"] == 11].pivot_table(
        index="ckpt", columns="signal", values="auc")
    l11 = l11.reindex(index=ALL_CKPTS)
    print(l11.to_string())

    print("\n" + "=" * 80)
    print("F4 FILTER (triptych): real-FPR @ τ=0.5")
    print("=" * 80)
    f4_pivot = f4_df.pivot_table(index="ckpt", columns="filter",
                                  values="real_fpr_at_0p5")
    print(f4_pivot.reindex(index=STAGE2_CKPTS).to_string())

    print("\n" + "=" * 80)
    print("F4 FILTER (triptych): fake-recall @ τ=0.5")
    print("=" * 80)
    f4_recall = f4_df.pivot_table(index="ckpt", columns="filter",
                                   values="fake_recall_at_0p5")
    print(f4_recall.reindex(index=STAGE2_CKPTS).to_string())

    print("\n" + "=" * 80)
    print("IQ DECOMP (triptych): R² of score ~ 6 IQ axes, ALL pool")
    print("=" * 80)
    iq_pivot = iq_df.pivot_table(index="ckpt", columns="pool", values="iq_r2")
    print(iq_pivot.reindex(index=STAGE2_CKPTS).to_string())

    print("\n" + "=" * 80)
    print("IQ DECOMP (triptych): residual AUC (fake-vs-real after IQ removed)")
    print("=" * 80)
    iq_resid = iq_df.pivot_table(index="ckpt", columns="pool", values="resid_auc")
    print(iq_resid.reindex(index=STAGE2_CKPTS).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
