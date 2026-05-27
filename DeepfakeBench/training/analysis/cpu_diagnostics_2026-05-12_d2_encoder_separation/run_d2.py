"""D2 — Encoder L11 real-vs-fake direction vs IQ-axis alignment.

Hypothesis tested: prior diagnostic A2-ext (2026-05-11) found T5C step3500's L11
features give LR-probe AUC = 1.000 on lockbox. But linear probes find ANY
separating direction. If that direction is IQ-aligned (cos angle ~0° between
`w_real_fake` and the IQ-axis direction), the encoder hasn't learned forgery —
it has learned IQ shortcuts. If IQ-orthogonal (angle ~90°), the encoder has
genuine forgery-aligned separation.

Method per ckpt (P8A, E2B, T5C step3500, T3_S1 step1500, optional CLIP-frozen):
  E1 — fit `LogisticRegression(C=1.0, max_iter=2000)` on (L11 features → label).
       Extract w_rf = coef_[0], normalize. Report 5-fold CV AUC.
  E2 — for each of 6 IQ axes (lap_var, min_dim, color_a_dev, color_b_dev,
       saturation_mean, luma_mean):
         z-score axis, fit LinearRegression (features → axis), extract w_axis,
         normalize.
  E3 — for each (ckpt, axis): angle_deg = arccos(|<w_rf, w_axis>|) * 180/pi.
       Also: stack 6 w_axis into 6x768, SVD, take first PC as "overall IQ
       direction", compute angle to w_rf.
  E4 — repeat E1-E3 restricted to (a) chronic_6 frames only, (b) non-chronic
       frames only.

Inputs:
  - L11 feature caches (n=800, 768-dim):
    analysis/iq_perlayer_probe_2026-05-08/_cache/intermediate__<LABEL>__layer11__n800.npz
  - Triptych metadata:
    analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/sampled_frames.csv
  - IQ atlas:
    analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet
  - Inline IQ computation fallback for frames not in atlas (cv2/LAB axes).

Outputs (written to ./outputs/):
  - angles_per_ckpt_per_axis.csv
  - angle_to_iq_pc1.csv
  - probe_aucs.csv

CPU-only by policy (memory `feedback_sklearn_njobs.md`). All sklearn n_jobs=1.
Optional CLIP-frozen extraction uses MPS if available, but is bounded
(skipped if it fails for any reason).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from collections import OrderedDict
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

# Six chronic-6 identity tokens (same definition as
# `cpu_diagnostics_2026-05-12_stage_a/run_stage_a.py:add_identity_cohort`).
CHRONIC_6_TOKENS = ["Roy_D", "PC_Generator", "bla_bla_chow",
                    "Md_noyn_Sharker", "dor_shkedi", "healthy_dor"]

# FT ckpt L11 cache labels. CLIP-frozen is added at runtime if extraction works.
FT_CKPT_LABELS = ["P8A", "E2B", "T5C_periodic_step3500", "T3_S1_step1500"]
# Friendly display names for the FACTS doc + CSVs.
DISPLAY_NAMES = {
    "CLIP_FROZEN": "CLIP_FROZEN",
    "P8A": "P8A",
    "E2B": "E2B",
    "T5C_periodic_step3500": "T5C_step3500",
    "T3_S1_step1500": "T3_S1_step1500",
}

logger = logging.getLogger("d2-encoder-separation")


# ---------------------------------------------------------------------------
# IQ panel assembly (atlas join + inline cv2 fallback).
# ---------------------------------------------------------------------------
def compute_iq_inline(local_paths: List[str]) -> pd.DataFrame:
    """Compute the 6 IQ axes inline via cv2/LAB.

    Formulas match `analysis/iq_data_atlas_2026-05-08/build_iq_atlas.py` and
    `analysis/iq_perlayer_probe_2026-05-08/run_probe.py:compute_iq_panel_inline`
    where they overlap. Adds color_a_dev and saturation_mean which are also
    in the atlas schema (not just the perlayer-probe's PRIMARY_6).
    """
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
            # a* and b* are in [0, 255] (cv2 convention) with neutral at 128.
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
    """Build per-frame IQ panel for the 800-frame triptych sample.

    Returns a DataFrame with one row per triptych frame, columns:
      row_ix, gcs_uri, label_str ('real'/'fake'), is_chronic_6 (0/1),
      lap_var, min_dim, color_a_dev, color_b_dev, saturation_mean, luma_mean.
    """
    df = pd.read_csv(
        SAMPLED_CSV,
        usecols=["gcs_uri", "label", "split", "identity_key", "local_path"],
    )
    df = df.iloc[:800].reset_index(drop=True)
    df["row_ix"] = np.arange(len(df))

    # is_chronic_6 from gcs_uri filename match (same regex as stage_a).
    df["is_chronic_6"] = df["gcs_uri"].fillna("").str.contains(
        "|".join(CHRONIC_6_TOKENS), case=False, regex=True
    ).astype(int)

    # Try atlas join. Atlas has dups per frame_path (frame can be in multiple
    # pools); take first match.
    atlas = pd.read_parquet(ATLAS_PARQUET)[["frame_path"] + IQ_AXES].copy()
    atlas = atlas.drop_duplicates(subset=["frame_path"], keep="first")
    merged = df.merge(atlas, left_on="gcs_uri", right_on="frame_path", how="left")
    n_in_atlas = merged[IQ_AXES[0]].notna().sum()
    n_missing = len(merged) - int(n_in_atlas)
    logger.info("IQ atlas join: %d in atlas, %d need inline cv2", n_in_atlas, n_missing)

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
    """Load L11 cached features for a ckpt label.

    Returns (features [n, 768], valid_idx [n]).
    """
    p = CACHE_DIR / f"intermediate__{label}__layer11__n{n}.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing cache: {p}")
    blob = np.load(p)
    return blob["features"].astype(np.float32), blob["valid_idx"].astype(np.int64)


# ---------------------------------------------------------------------------
# CLIP-frozen extraction (optional, MPS-accelerated, bounded).
# ---------------------------------------------------------------------------
def extract_clip_frozen_l11_features(local_paths: List[str]) -> Optional[np.ndarray]:
    """Try to extract OpenCLIP B16 frozen L11 features for the 800 frames.

    Returns features array [n, 768] on success, None on failure.

    Uses local cached weights at
    `weights/CLIP-ViT-B-16-DataComp.XL-s13B-b90K/open_clip_pytorch_model.bin`
    if present; falls back to open_clip download otherwise.
    """
    try:
        import torch
        import open_clip
        import cv2
    except Exception as e:
        logger.warning("CLIP-frozen extraction blocked: missing import: %s", e)
        return None

    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

    # Pick device.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("CLIP-frozen device: %s", device)

    # Try local cached weights first.
    local_pretrained = (
        REPO_ROOT / "weights" / "CLIP-ViT-B-16-DataComp.XL-s13B-b90K"
        / "open_clip_pytorch_model.bin"
    )
    try:
        if local_pretrained.exists():
            logger.info("loading OpenCLIP B16 from local cache: %s", local_pretrained)
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained=str(local_pretrained)
            )
        else:
            logger.info("local cache not found; downloading datacomp_xl_s13b_b90k")
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained="datacomp_xl_s13b_b90k"
            )
    except Exception as e:
        logger.warning("CLIP-frozen blocked: model load failed: %s", e)
        return None

    model = model.to(device).eval()
    visual = model.visual
    if not (hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks")):
        logger.warning("CLIP-frozen blocked: cannot locate transformer.resblocks")
        return None

    captured: List[np.ndarray] = []

    def hook(_module, _input, output):
        # output shape: either (seq, batch, hidden) or (batch, seq, hidden).
        if output.dim() == 3:
            if output.shape[0] >= output.shape[1]:
                cls = output[0]  # (seq, batch, hidden) -> CLS = output[0]
            else:
                cls = output[:, 0]  # (batch, seq, hidden) -> CLS = output[:, 0]
        elif output.dim() == 2:
            cls = output
        else:
            raise RuntimeError(f"unexpected resblock output: {tuple(output.shape)}")
        captured.append(cls.detach().cpu().to(torch.float32).numpy())

    handle = visual.transformer.resblocks[11].register_forward_hook(hook)

    try:
        # Preprocess + batched forward.
        feats_per_frame: List[Optional[np.ndarray]] = [None] * len(local_paths)
        batch_size = 16
        pending: List[Tuple[int, "torch.Tensor"]] = []

        for i, p in enumerate(local_paths):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
            pending.append((i, torch.from_numpy(img.transpose(2, 0, 1))))
        logger.info("CLIP-frozen: %d/%d frames loaded", len(pending), len(local_paths))

        for j in range(0, len(pending), batch_size):
            chunk = pending[j : j + batch_size]
            batch_idx = [c[0] for c in chunk]
            batch = torch.stack([c[1] for c in chunk]).to(device, non_blocking=True)
            captured.clear()
            with torch.inference_mode():
                _ = visual(batch)
            if not captured:
                logger.warning("CLIP-frozen: no capture for batch %d", j // batch_size)
                continue
            cls_batch = captured[0]
            if cls_batch.shape[0] != len(batch_idx):
                logger.warning("CLIP-frozen: capture shape mismatch (%d vs %d)",
                               cls_batch.shape[0], len(batch_idx))
                continue
            for k, ii in enumerate(batch_idx):
                feats_per_frame[ii] = cls_batch[k]
            if (j // batch_size) % 5 == 0:
                logger.info("  batch %d/%d", j // batch_size + 1,
                            (len(pending) + batch_size - 1) // batch_size)
    except Exception as e:
        handle.remove()
        logger.warning("CLIP-frozen extraction failed mid-forward: %s", e)
        return None
    finally:
        handle.remove()

    # Filter to frames where we got features.
    out = []
    for i, f in enumerate(feats_per_frame):
        if f is None:
            # Use zero vec so row_ix alignment stays — we'll filter out later.
            out.append(None)
        else:
            out.append(f)

    # Return only frames where features exist; record valid_idx via the caller.
    arr = np.stack([f for f in out if f is not None], axis=0)
    valid_idx = np.array([i for i, f in enumerate(out) if f is not None], dtype=np.int64)
    logger.info("CLIP-frozen extracted %d/%d features", arr.shape[0], len(local_paths))

    # Save to cache so it can be reused.
    cache_out = CACHE_DIR / f"intermediate__CLIP_FROZEN__layer11__n{len(local_paths)}.npz"
    try:
        np.savez_compressed(cache_out, features=arr.astype(np.float32),
                            valid_idx=valid_idx)
        logger.info("CLIP-frozen cached to %s", cache_out)
    except Exception as e:
        logger.warning("CLIP-frozen cache save failed: %s (continuing)", e)

    return arr, valid_idx


# ---------------------------------------------------------------------------
# Direction extraction.
# ---------------------------------------------------------------------------
def fit_rf_direction(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit LR (C=1.0, max_iter=2000). Return (w_rf_unit, 5fold_CV_AUC).

    n_jobs=1 throughout per project memory.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    # Full-fit direction.
    clf_full = LogisticRegression(C=1.0, max_iter=2000, n_jobs=1, solver="lbfgs")
    clf_full.fit(X, y)
    w = clf_full.coef_[0].astype(np.float64)
    n = np.linalg.norm(w)
    if n <= 0:
        return w, float("nan")
    w_unit = w / n

    # 5-fold CV AUC.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=1, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y, oof))
    return w_unit, auc


def fit_axis_direction(X: np.ndarray, axis_values: np.ndarray) -> np.ndarray:
    """Fit LinearRegression(features -> axis_value) and return unit-normalized
    coef vector (768-d)."""
    from sklearn.linear_model import LinearRegression

    # Z-score axis values first.
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
    """Angle in degrees between u and v, using |cos| (axis-direction, not arrow).

    Returns NaN if either norm is 0.
    """
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu <= 0 or nv <= 0:
        return float("nan")
    c = float(np.dot(u, v) / (nu * nv))
    c = min(1.0, max(-1.0, c))
    return float(np.degrees(np.arccos(abs(c))))


# ---------------------------------------------------------------------------
# Main per-ckpt analysis.
# ---------------------------------------------------------------------------
def analyze_one_ckpt(
    label: str,
    features: np.ndarray,
    valid_idx_in_panel: np.ndarray,
    panel: pd.DataFrame,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """For each sub-cohort in {full, chronic_6, non_chronic}:
       - Fit w_rf via LR, record AUC (5-fold CV) on the sub-cohort.
       - Fit w_axis per IQ axis via LR, compute angle_deg(w_rf, w_axis).
       - Stack 6 w_axis, SVD -> PC1 = "overall IQ direction", compute angle.

    Inputs:
      features: (n_feats, 768) — npz['features'] array
      valid_idx_in_panel: (n_feats,) — indices into the 800-row panel that the
                           features correspond to (npz['valid_idx']).
      panel: 800-row DataFrame with label/cohort/IQ columns.

    Returns (angle_rows, pc1_rows, auc_rows) — lists of dicts.
    """
    angle_rows: List[dict] = []
    pc1_rows: List[dict] = []
    auc_rows: List[dict] = []

    # Build aligned (features, panel_subset).
    # valid_idx_in_panel is indices into the 800-frame panel.
    sub_panel = panel.iloc[valid_idx_in_panel].reset_index(drop=True)
    # Map label -> 0/1 (real=0, fake=1).
    sub_panel["y"] = (sub_panel["label"] == "fake").astype(int)
    # Filter to rows with non-null IQ (all 6 axes).
    iq_ok = sub_panel[IQ_AXES].notna().all(axis=1)
    keep_mask = iq_ok.values
    n_total = len(sub_panel)
    n_keep = int(keep_mask.sum())
    logger.info("[%s] feature rows=%d, IQ-complete=%d", label, n_total, n_keep)
    if n_keep < 20:
        logger.warning("[%s] too few IQ-complete rows; skipping", label)
        return angle_rows, pc1_rows, auc_rows

    X_keep = features[keep_mask]
    panel_keep = sub_panel.loc[keep_mask].reset_index(drop=True)

    # Define sub-cohort indices into panel_keep (0..n_keep-1).
    cohorts: Dict[str, np.ndarray] = {
        "full": np.arange(n_keep, dtype=np.int64),
        "chronic_6": np.where(panel_keep["is_chronic_6"].values == 1)[0],
        "non_chronic": np.where(panel_keep["is_chronic_6"].values == 0)[0],
    }

    for cohort_name, idx in cohorts.items():
        n_c = len(idx)
        if n_c < 20:
            logger.info("[%s/%s] cohort n=%d too small; skipping", label, cohort_name, n_c)
            continue
        Xc = X_keep[idx]
        yc = panel_keep["y"].values[idx]
        n_real = int((yc == 0).sum())
        n_fake = int((yc == 1).sum())
        if n_real < 5 or n_fake < 5:
            logger.info("[%s/%s] real=%d fake=%d insufficient; skipping",
                        label, cohort_name, n_real, n_fake)
            continue

        # E1: w_rf + 5-fold CV AUC.
        try:
            w_rf, probe_auc = fit_rf_direction(Xc, yc)
        except Exception as e:
            logger.warning("[%s/%s] LR failed: %s", label, cohort_name, e)
            continue
        auc_rows.append({
            "ckpt": DISPLAY_NAMES.get(label, label),
            "sub_cohort": cohort_name,
            "n_frames": int(n_c),
            "n_real": n_real,
            "n_fake": n_fake,
            "probe_auc_5fold": probe_auc,
        })

        # E2 + E3: w_axis per axis, angle to w_rf.
        w_axes: List[np.ndarray] = []
        for ax in IQ_AXES:
            ax_vals = panel_keep[ax].values[idx].astype(np.float32)
            try:
                w_ax = fit_axis_direction(Xc, ax_vals)
            except Exception as e:
                logger.warning("[%s/%s/%s] axis fit failed: %s",
                               label, cohort_name, ax, e)
                w_ax = np.zeros(Xc.shape[1], dtype=np.float64)
            w_axes.append(w_ax)
            ang = cosine_angle_deg(w_rf, w_ax)
            angle_rows.append({
                "ckpt": DISPLAY_NAMES.get(label, label),
                "sub_cohort": cohort_name,
                "axis": ax,
                "angle_deg": ang,
                "n_frames": int(n_c),
            })

        # E3-PC1: SVD on stacked w_axes (6 x 768) -> first right-sing-vec = PC1.
        W = np.stack(w_axes, axis=0)  # (6, 768)
        # Drop axes that came back as zero (failed fits).
        norms = np.linalg.norm(W, axis=1)
        valid_rows = W[norms > 1e-12]
        if valid_rows.shape[0] < 2:
            logger.warning("[%s/%s] <2 valid axis weights; skipping PC1", label, cohort_name)
            pc1_angle = float("nan")
            n_valid_axes = int(valid_rows.shape[0])
        else:
            try:
                U, S, Vt = np.linalg.svd(valid_rows, full_matrices=False)
                pc1 = Vt[0]  # (768,)
                pc1_angle = cosine_angle_deg(w_rf, pc1)
            except Exception as e:
                logger.warning("[%s/%s] SVD failed: %s", label, cohort_name, e)
                pc1_angle = float("nan")
            n_valid_axes = int(valid_rows.shape[0])
        pc1_rows.append({
            "ckpt": DISPLAY_NAMES.get(label, label),
            "sub_cohort": cohort_name,
            "angle_to_iq_pc1_deg": pc1_angle,
            "n_axes_used": n_valid_axes,
            "n_frames": int(n_c),
            "probe_auc_5fold": probe_auc,
        })

    return angle_rows, pc1_rows, auc_rows


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    logger.info("D2 — encoder L11 separation direction vs IQ-axis alignment")

    # ----- 1. Build IQ panel. -----
    panel = build_iq_panel()
    logger.info("panel rows=%d; chronic_6 count=%d",
                len(panel), int(panel["is_chronic_6"].sum()))

    # ----- 2. Attempt CLIP-frozen extraction (optional, bounded). -----
    clip_frozen_blocked_reason: Optional[str] = None
    clip_frozen_cache = CACHE_DIR / "intermediate__CLIP_FROZEN__layer11__n800.npz"
    clip_frozen_features: Optional[np.ndarray] = None
    clip_frozen_valid_idx: Optional[np.ndarray] = None
    if clip_frozen_cache.exists():
        logger.info("CLIP_FROZEN cache hit: %s", clip_frozen_cache)
        blob = np.load(clip_frozen_cache)
        clip_frozen_features = blob["features"].astype(np.float32)
        clip_frozen_valid_idx = blob["valid_idx"].astype(np.int64)
    else:
        try:
            paths = pd.read_csv(SAMPLED_CSV)["local_path"].iloc[:800].tolist()
            result = extract_clip_frozen_l11_features(paths)
            if result is None:
                clip_frozen_blocked_reason = "extract_clip_frozen_l11_features returned None"
            else:
                clip_frozen_features, clip_frozen_valid_idx = result
        except Exception as e:
            clip_frozen_blocked_reason = f"extraction raised: {e}"
            logger.warning("CLIP-frozen extraction blocked: %s", e)

    # ----- 3. Per-ckpt analysis. -----
    all_angle_rows: List[dict] = []
    all_pc1_rows: List[dict] = []
    all_auc_rows: List[dict] = []

    # CLIP-frozen first if available.
    if clip_frozen_features is not None and clip_frozen_valid_idx is not None:
        a, p, u = analyze_one_ckpt(
            "CLIP_FROZEN", clip_frozen_features, clip_frozen_valid_idx, panel
        )
        all_angle_rows.extend(a)
        all_pc1_rows.extend(p)
        all_auc_rows.extend(u)

    for label in FT_CKPT_LABELS:
        try:
            feats, vidx = load_l11_features(label, n=800)
        except FileNotFoundError as e:
            logger.error("missing cache for %s: %s", label, e)
            continue
        a, p, u = analyze_one_ckpt(label, feats, vidx, panel)
        all_angle_rows.extend(a)
        all_pc1_rows.extend(p)
        all_auc_rows.extend(u)

    # ----- 4. Write outputs. -----
    df_angles = pd.DataFrame(all_angle_rows)
    df_pc1 = pd.DataFrame(all_pc1_rows)
    df_auc = pd.DataFrame(all_auc_rows)

    df_angles.to_csv(OUTPUTS / "angles_per_ckpt_per_axis.csv", index=False)
    df_pc1.to_csv(OUTPUTS / "angle_to_iq_pc1.csv", index=False)
    df_auc.to_csv(OUTPUTS / "probe_aucs.csv", index=False)
    logger.info("wrote %d angle rows, %d pc1 rows, %d AUC rows",
                len(df_angles), len(df_pc1), len(df_auc))

    # ----- 5. Print compact summary. -----
    if len(df_auc) > 0:
        print("\n=== Probe AUCs (5-fold CV) ===")
        print(df_auc.to_string(index=False))
    if len(df_pc1) > 0:
        print("\n=== Angle to IQ PC1 (overall IQ direction) ===")
        print(df_pc1.to_string(index=False))
    if len(df_angles) > 0:
        print("\n=== Angle per axis (full cohort only) ===")
        print(df_angles[df_angles["sub_cohort"] == "full"].to_string(index=False))
    if clip_frozen_blocked_reason is not None:
        print(f"\n=== CLIP-frozen blocked: {clip_frozen_blocked_reason} ===")
    # Persist a summary json for the FACTS doc.
    summary = {
        "n_panel": int(len(panel)),
        "n_chronic_6": int(panel["is_chronic_6"].sum()),
        "clip_frozen_status": "ok" if clip_frozen_features is not None else f"blocked: {clip_frozen_blocked_reason}",
        "ckpts_processed": sorted({r["ckpt"] for r in all_auc_rows}),
    }
    with open(OUTPUTS / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
