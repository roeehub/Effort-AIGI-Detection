"""Compute L11 atlas inv_mean (full + per-substrate) for T6/T7/T5C ckpts.

Uses cached L11 features at iq_perlayer_probe_2026-05-08/_cache/.
Reuses A3 pattern from cpu_diagnostics_2026-05-11_a3_atlas_composition/.

Outputs:
  - per_ckpt_inv_mean.csv      — full-triptych and per-substrate inv_mean per ckpt
  - L11_inv_mean_summary.csv   — sorted full inv_mean table
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT = REPO / "analysis" / "cpu_diagnostics_2026-05-11_t67_t5c_probe" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLED_CSV = REPO / "analysis" / "embedding_triptych_2026-04-30" / "outputs" / "triptych_p8a_slot2_slot3" / "sampled_frames.csv"
FEAT_CACHE = REPO / "analysis" / "iq_perlayer_probe_2026-05-08" / "_cache"

ATLAS_PARQUET = REPO / "analysis" / "iq_data_atlas_2026-05-08" / "outputs" / "per_frame.parquet"
PRIMARY_6_IQ = ["lap_var", "min_dim", "luma_mean", "color_b_dev", "edge_mag", "skin_frac"]
CHRONIC_6_PATTERNS = [
    "Roy_D", "PC_Generator", "bla_bla_chow",
    "Md_noyn_Sharker", "dor_shkedi", "healthy_dor",
]

# Target ckpts (anchors first, then candidates)
TARGET_CKPTS = [
    "P8A", "E2B", "T3_S1_step1500", "T4_L1_step10500",  # references for context
    "T6_periodic_step500", "T6_periodic_step1500", "T6_periodic_step2500",
    "T6_periodic_step3500", "T6_periodic_step4500", "T6_top_n_step3250",
    "T6_top_n_step6000", "T6_top_n_step10250",
    "T7_periodic_step500", "T7_periodic_step1500", "T7_periodic_step2500",
    "T7_periodic_step3500", "T7_periodic_step5000",
    "T7_top_n_step4250", "T7_top_n_step4750",
    "T5C_periodic_step500", "T5C_periodic_step1500", "T5C_periodic_step2500",
    "T5C_periodic_step3500", "T5C_periodic_step5000",
    "T5C_top_n_step2750", "T5C_top_n_step3750",
]

logger = logging.getLogger("inv-mean")


def chronic_match(s) -> bool:
    if not isinstance(s, str):
        return False
    sl = s.lower()
    return any(p.lower() in sl for p in CHRONIC_6_PATTERNS)


def is_dor_fn(identity_key) -> bool:
    if not isinstance(identity_key, str):
        return False
    return "dor" in identity_key.lower()


def build_panel():
    cols = ["gcs_uri", "label", "split", "identity_key",
            "session_id", "video_id", "method",
            "local_path", "width", "height", "is_no_face",
            "face_pixel_area"]
    df = pd.read_csv(SAMPLED_CSV, usecols=cols).iloc[:800].reset_index(drop=True)
    df["row_ix"] = np.arange(len(df))
    df["is_dor"] = df["identity_key"].apply(is_dor_fn).astype(int)
    df["is_chronic_6"] = df["identity_key"].apply(chronic_match).astype(int)
    df["is_lockbox"] = (df["split"] == "lockbox").astype(int)
    df["is_real_vs_fake"] = (df["label"] == "fake").astype(int)

    try:
        atlas = pd.read_parquet(ATLAS_PARQUET)[["frame_path"] + PRIMARY_6_IQ].copy()
        atlas = atlas.drop_duplicates(subset=["frame_path"], keep="first")
        df = df.merge(atlas, left_on="gcs_uri", right_on="frame_path", how="left")
    except Exception as exc:
        logger.warning("atlas not joinable: %s", exc)
        for c in PRIMARY_6_IQ:
            df[c] = np.nan

    missing = df["lap_var"].isna()
    if missing.any():
        import cv2
        for ix in df.index[missing]:
            try:
                img = cv2.imread(str(df.loc[ix, "local_path"]), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                df.loc[ix, "lap_var"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                df.loc[ix, "min_dim"] = float(min(h, w))
            except Exception:
                pass

    nan_min = df["min_dim"].isna() if "min_dim" in df.columns else None
    if nan_min is not None and nan_min.any():
        df.loc[nan_min, "min_dim"] = df.loc[nan_min, ["width", "height"]].min(axis=1)

    return df


def feat_path(label: str, layer: int = 11) -> Path:
    return FEAT_CACHE / f"intermediate__{label}__layer{layer:02d}__n800.npz"


def load_features(label: str, layer: int = 11):
    p = feat_path(label, layer)
    if not p.exists():
        return None, None
    blob = np.load(p)
    return blob["features"].astype(np.float32), blob["valid_idx"].astype(np.int64)


def fit_probe_auc(X, y, n_splits=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y, dtype=int)
    if len(y) < 20:
        return float("nan")
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return float("nan")
    actual_splits = min(n_splits, int(y.sum()), int(len(y) - y.sum()))
    if actual_splits < 3:
        return float("nan")
    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=3000, n_jobs=1, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    try:
        return float(roc_auc_score(y, oof))
    except Exception:
        return float("nan")


def per_substrate_inv_mean(panel: pd.DataFrame):
    panel = panel.copy()
    panel["lap_var_high"] = (panel["lap_var"] > panel["lap_var"].median()).astype(int)
    panel["min_dim_high"] = (panel["min_dim"] > panel["min_dim"].median()).astype(int)
    if panel["face_pixel_area"].notna().any():
        panel["face_size_high"] = (
            panel["face_pixel_area"] > panel["face_pixel_area"].median()
        ).astype(int)
    else:
        panel["face_size_high"] = 0

    SIGNALS = [
        ("is_real_vs_fake", "is_real_vs_fake"),
        ("is_dor", "is_dor"),
        ("is_chronic_6", "is_chronic_6"),
        ("lap_var_high", "lap_var_high"),
        ("min_dim_high", "min_dim_high"),
        ("face_size_high", "face_size_high"),
    ]
    SHORTCUTS = ["is_dor", "is_chronic_6", "lap_var_high", "min_dim_high", "face_size_high"]

    SLICES = {
        "full_n800":           pd.Series(True, index=panel.index),
        "dev_only":            (panel["split"] == "dev"),
        "lockbox_only":        (panel["split"] == "lockbox"),
        "chronic_6_only":      (panel["is_chronic_6"] == 1),
        "non_chronic_only":    (panel["is_chronic_6"] == 0),
    }

    rows = []
    for ckpt_name in TARGET_CKPTS:
        feats, valid_idx = load_features(ckpt_name)
        if feats is None:
            logger.warning("[%s] no L11 cache, skipping", ckpt_name)
            continue
        valid_to_pos = {int(v): r for r, v in enumerate(valid_idx)}
        valid_set = set(valid_to_pos.keys())

        for slice_name, slice_mask in SLICES.items():
            sub = panel[slice_mask].copy()
            sub = sub[sub["row_ix"].isin(valid_set)].reset_index(drop=True)
            if len(sub) < 30:
                continue
            sel_pos = [valid_to_pos[int(rx)] for rx in sub["row_ix"].values]
            X = feats[sel_pos]
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

            sig_aucs = {}
            for sig_label, sig_col in SIGNALS:
                if sig_col not in sub.columns:
                    sig_aucs[sig_label] = float("nan")
                    continue
                y = sub[sig_col].values.astype(int)
                sig_aucs[sig_label] = fit_probe_auc(X, y)

            forgery_auc = sig_aucs.get("is_real_vs_fake", float("nan"))
            shortcut_vals = [sig_aucs.get(s, float("nan")) for s in SHORTCUTS]
            shortcut_vals_clean = [v for v in shortcut_vals if not (v is None or np.isnan(v))]
            mean_shortcut = float(np.mean(shortcut_vals_clean)) if shortcut_vals_clean else float("nan")
            inv_mean = (forgery_auc - mean_shortcut) if (not np.isnan(forgery_auc) and not np.isnan(mean_shortcut)) else float("nan")

            row = {
                "ckpt": ckpt_name,
                "slice": slice_name,
                "n": len(sub),
                "n_fake": int(sub["is_real_vs_fake"].sum()),
                "n_real": int(len(sub) - sub["is_real_vs_fake"].sum()),
                "forgery_auc": forgery_auc,
                "mean_shortcut_auc": mean_shortcut,
                "inv_mean": inv_mean,
            }
            for s in SHORTCUTS:
                row[f"auc_{s}"] = sig_aucs.get(s, float("nan"))
            rows.append(row)
            logger.info("  [%s] slice=%s n=%d forgery=%.3f inv_mean=%+.4f",
                        ckpt_name, slice_name, len(sub), forgery_auc, inv_mean)

    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    logger.info("building panel")
    panel = build_panel()
    logger.info("panel: %d frames", len(panel))

    logger.info("computing per-substrate inv_mean (5 slices × %d ckpts)", len(TARGET_CKPTS))
    tbl = per_substrate_inv_mean(panel)
    tbl.to_csv(OUT / "per_ckpt_inv_mean.csv", index=False)
    logger.info("wrote per_ckpt_inv_mean.csv (%d rows)", len(tbl))

    # Summary: full_n800 only, sorted
    full = tbl[tbl["slice"] == "full_n800"][["ckpt", "n", "forgery_auc", "mean_shortcut_auc", "inv_mean"]].sort_values("inv_mean", ascending=False)
    full.to_csv(OUT / "L11_inv_mean_summary.csv", index=False)
    print("\n=== full_n800 inv_mean (sorted) ===")
    print(full.to_string(index=False, float_format="%.4f"))

    # chronic_6 slice
    c6 = tbl[tbl["slice"] == "chronic_6_only"][["ckpt", "n", "forgery_auc", "mean_shortcut_auc", "inv_mean"]].sort_values("inv_mean", ascending=False)
    print("\n=== chronic_6_only inv_mean (sorted) ===")
    print(c6.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    raise SystemExit(main())
