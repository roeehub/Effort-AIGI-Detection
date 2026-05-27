"""D11 — Name the dev-vs-lockbox real KLIEP axis.

Continuation of D8/D10. D8 showed dev-real (n=2000) and lockbox-real (n=414)
are 99.09% linearly separable in CLIP-frozen ViT-B/16 L11 feature space. D10
showed the discriminator axis is 89.71° from the 6-IQ-axis PC1 — i.e., the
separation is not explained by the 6 measured IQ axes.

D11 enumerates per-frame metadata (manifest + IQ atlas + pixel-derived stats),
regresses the KLIEP scalar projection on each field individually and jointly,
and reports the R² that each field / cumulative model explains.

Operational constraints (per AGENT_GUIDE + memory):
  - n_jobs=1 in every sklearn call.
  - CPU only; uses cached CLIP-frozen L11 features from D8 npz.
  - Forbidden words excluded in FACTS doc; numerical only.
  - No re-extraction of CLIP features.

Inputs:
  - D8 cache npz (4839×768 CLIP-frozen L11 CLS features).
  - lockbox_tagging/full_tags_2026-04-27.parquet (per-frame manifest).
  - iq_data_atlas_2026-05-08/outputs/per_frame.parquet (IQ atlas).
  - teams_target_domain_manifest_2026-04-23_with_dor.json (video manifest).
  - local frame cache /analysis/lockbox_tagging/_frame_cache/.

Outputs:
  - per_frame_projection_metadata.csv — long-form: gcs_uri × all fields × proj.
  - cat_field_anova.csv — per-categorical-field R² (one-way ANOVA on KLIEP proj).
  - cont_field_correlation.csv — per-continuous-field Pearson r and R².
  - cumulative_regression_models.csv — R² for: manifest-only / +pixel-stats.
  - extreme_frames.csv — 5 most-dev-aligned + 5 most-lockbox-aligned frames.
  - clip_frozen_pixel_stats__n2414.npz — pixel-stats cache for the 2414 reals.
  - _summary.json — machine-readable summary of all rows above.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
)
THIS_DIR = REPO_ROOT / "analysis" / "cpu_diagnostics_2026-05-13_d11_dev_lockbox_axis"
OUTPUTS = THIS_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
LOG_PATH = THIS_DIR / "_run.log"

D8_CACHE = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-12_d8_head_retrain_substrate_balanced/outputs/clip_frozen_l11__n4839.npz"
LOCKBOX_PARQUET = REPO_ROOT / "analysis/lockbox_tagging/full_tags_2026-04-27.parquet"
IQ_ATLAS_PARQUET = REPO_ROOT / "analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet"
TEAMS_MANIFEST = REPO_ROOT / "arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json"
PIXEL_STATS_CACHE = OUTPUTS / "clip_frozen_pixel_stats__n2414.npz"

SEED = 42

logger = logging.getLogger("d11")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, mode="w"),
            logging.StreamHandler(),
        ],
    )


# ---------------------------------------------------------------------------
# 1. Replay the D8 dev/lockbox sampling.
# ---------------------------------------------------------------------------
def replay_dev_lockbox_paths() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Same as D8 / D10 — seed=42 stratified 2000/2000 dev, full lockbox."""
    df = pd.read_parquet(LOCKBOX_PARQUET)
    df = df[df["local_path"].astype(str).str.len() > 0].reset_index(drop=True)
    dev_df = df[df["split"] == "dev"].copy()
    dev_real = dev_df[dev_df["label"] == "real"]
    dev_fake = dev_df[dev_df["label"] == "fake"]
    rng = np.random.default_rng(seed=42)
    dev_real_idx = rng.choice(len(dev_real), size=2000, replace=False)
    dev_fake_idx = rng.choice(len(dev_fake), size=2000, replace=False)
    dev_sample = pd.concat(
        [dev_real.iloc[dev_real_idx], dev_fake.iloc[dev_fake_idx]]
    ).reset_index(drop=True)
    lb_df = df[df["split"] == "lockbox"].copy().reset_index(drop=True)
    return dev_sample, lb_df


def load_real_features() -> Dict[str, np.ndarray]:
    """Return CLIP-frozen real features (dev + lockbox), their labels, and
    metadata DataFrame in the same order as features."""
    d8 = np.load(D8_CACHE, allow_pickle=True)
    feats = d8["features"].astype(np.float32)
    assert feats.shape[0] == 4839, f"D8 cache shape mismatch: {feats.shape}"

    dev_sample, lb_df = replay_dev_lockbox_paths()
    assert len(dev_sample) == 4000
    assert len(lb_df) == 839

    dev_labels = (dev_sample["label"] == "fake").astype(np.int64).to_numpy()
    lb_labels = (lb_df["label"] == "fake").astype(np.int64).to_numpy()

    clip_dev = feats[:4000]
    clip_lb = feats[4000:]

    dev_real_mask = (dev_labels == 0)
    lb_real_mask = (lb_labels == 0)

    dev_real_feats = clip_dev[dev_real_mask]
    lb_real_feats = clip_lb[lb_real_mask]
    dev_real_meta = dev_sample[dev_real_mask].reset_index(drop=True).copy()
    lb_real_meta = lb_df[lb_real_mask].reset_index(drop=True).copy()

    dev_real_meta["pool"] = "dev_real"
    lb_real_meta["pool"] = "lockbox_real"

    feats_all = np.concatenate([dev_real_feats, lb_real_feats], axis=0)
    meta_all = pd.concat([dev_real_meta, lb_real_meta], axis=0).reset_index(drop=True)
    is_lockbox = np.concatenate([
        np.zeros(len(dev_real_feats), dtype=np.int64),
        np.ones(len(lb_real_feats), dtype=np.int64),
    ])
    logger.info(
        "loaded: dev_real=%d lockbox_real=%d total=%d",
        len(dev_real_feats), len(lb_real_feats), len(feats_all),
    )
    return {
        "feats": feats_all,
        "meta": meta_all,
        "is_lockbox": is_lockbox,
    }


# ---------------------------------------------------------------------------
# 2. KLIEP discriminator (D8 estimator C / D10 method).
# ---------------------------------------------------------------------------
def fit_kliep(feats: np.ndarray, y: np.ndarray) -> Dict:
    from sklearn.linear_model import LogisticRegression

    X = feats.astype(np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    clf = LogisticRegression(
        C=1.0, max_iter=2000, solver="lbfgs", n_jobs=1, class_weight="balanced"
    )
    clf.fit(X, y)
    w = clf.coef_[0].astype(np.float64)
    b = float(clf.intercept_[0])
    acc = float(clf.score(X, y))
    proj = X @ w + b
    return {"w": w, "b": b, "accuracy": acc, "proj": proj.astype(np.float64)}


# ---------------------------------------------------------------------------
# 3. Merge teams_target_domain_manifest fields onto frames.
# ---------------------------------------------------------------------------
def load_teams_manifest_frame_index() -> pd.DataFrame:
    """Return one row per frame_path with manifest-level fields
    (source_kind, slices, prefix, identity, sequence_id)."""
    with open(TEAMS_MANIFEST) as f:
        data = json.load(f)
    rows = []
    for v in data["videos"]:
        slices = v.get("slices") or []
        slice_str = "|".join(sorted(set(slices)))
        for fp in v.get("frame_paths", []):
            rows.append({
                "gcs_uri": fp,
                "manifest_source_kind": v.get("source_kind"),
                "manifest_slices": slice_str,
                "manifest_prefix": v.get("prefix"),
                "manifest_sequence_id": v.get("sequence_id"),
                "manifest_method": v.get("method"),
                "manifest_segment_id": v.get("segment_id"),
                "manifest_video_id": v.get("video_id"),
            })
    return pd.DataFrame(rows)


def merge_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    """Join manifest and IQ atlas onto the lockbox_tagging meta."""
    logger.info("loading teams target-domain manifest")
    manifest_df = load_teams_manifest_frame_index()
    n_before_m = len(manifest_df)
    manifest_df = manifest_df.drop_duplicates(subset=["gcs_uri"], keep="first")
    logger.info("manifest rows: %d -> %d after dedupe", n_before_m, len(manifest_df))

    logger.info("loading IQ atlas")
    atlas = pd.read_parquet(IQ_ATLAS_PARQUET)
    # The atlas frame_path uses gcs_uri; rename to align.
    atlas = atlas.rename(columns={"frame_path": "gcs_uri"})
    atlas_join = atlas[[
        "gcs_uri", "lap_var", "min_dim", "max_dim", "luma_mean", "luma_std",
        "edge_mag", "saturation_mean", "contrast_l", "color_a_dev",
        "color_b_dev", "skin_frac", "bytes",
    ]].copy()
    atlas_join.columns = [
        "gcs_uri", "atlas_lap_var", "atlas_min_dim", "atlas_max_dim",
        "atlas_luma_mean", "atlas_luma_std", "atlas_edge_mag",
        "atlas_saturation_mean", "atlas_contrast_l", "atlas_color_a_dev",
        "atlas_color_b_dev", "atlas_skin_frac", "atlas_bytes",
    ]
    # Atlas has the same frame in multiple pool/role rows — collapse to first
    # occurrence per gcs_uri so the merge does not multiply rows.
    n_before = len(atlas_join)
    atlas_join = atlas_join.drop_duplicates(subset=["gcs_uri"], keep="first")
    logger.info("atlas dedupe: %d -> %d rows", n_before, len(atlas_join))

    merged = meta.merge(manifest_df, on="gcs_uri", how="left")
    merged = merged.merge(atlas_join, on="gcs_uri", how="left")
    logger.info("merged meta shape: %s", merged.shape)
    # Quick coverage report
    for c in ("manifest_source_kind", "manifest_slices", "atlas_lap_var"):
        cov = merged[c].notna().sum() if c in merged else 0
        logger.info("  coverage %s: %d / %d (%.1f%%)",
                    c, cov, len(merged), 100.0 * cov / max(1, len(merged)))
    return merged


# ---------------------------------------------------------------------------
# 4. Per-frame pixel-level statistics.
# ---------------------------------------------------------------------------
def extract_pixel_stats(paths: List[str]) -> np.ndarray:
    """For each local_path: compute additional pixel-level features.

    Returns array (n, 12) with:
      [0] fft_low_band  (mean log|F| in radial bins 0..2)
      [1] fft_mid_band  (mean log|F| in bins 3..7)
      [2] fft_high_band (mean log|F| in bins 8..13)
      [3] fft_xhigh_band (mean log|F| in bins 14..15)
      [4] fft_high_minus_low (band difference)
      [5] corr_R_G       (Pearson corr of R and G pixel values)
      [6] corr_R_B       (Pearson corr of R and B pixel values)
      [7] corr_G_B       (Pearson corr of G and B pixel values)
      [8] hist_entropy   (Shannon entropy of luminance histogram)
      [9] dct_blockiness (mean |DCT(8x8 block boundary minus midblock)|)
     [10] sensor_noise   (variance of laplacian-highpass residuals)
     [11] highfreq_energy_log (log of total |F| in bins 12..15)

    Uses RGB image at 224x224 (CLIP-style preprocessing analog).
    Returns NaN row if image cannot be loaded.
    """
    import cv2

    RES = 224
    N_RADIAL = 16
    # build radial bin map
    yy, xx = np.mgrid[:RES, :RES]
    cx = cy = (RES - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_max = r.max()
    rbins = np.clip((r / r_max * N_RADIAL).astype(int), 0, N_RADIAL - 1)

    band_masks = {
        "low": np.isin(rbins, [0, 1, 2]),
        "mid": np.isin(rbins, [3, 4, 5, 6, 7]),
        "high": np.isin(rbins, [8, 9, 10, 11, 12, 13]),
        "xhigh": np.isin(rbins, [14, 15]),
        "veryhigh": np.isin(rbins, [12, 13, 14, 15]),
    }

    out = np.full((len(paths), 12), np.nan, dtype=np.float64)
    t0 = time.time()
    last_log = t0
    for i, p in enumerate(paths):
        try:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)  # BGR
            if img is None:
                continue
            img = cv2.resize(img, (RES, RES), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # FFT
            F = np.fft.fft2(gray)
            F = np.fft.fftshift(F)
            mag = np.log1p(np.abs(F)).astype(np.float64)
            low = float(mag[band_masks["low"]].mean())
            mid = float(mag[band_masks["mid"]].mean())
            high = float(mag[band_masks["high"]].mean())
            xhigh = float(mag[band_masks["xhigh"]].mean())
            veryhigh_energy = float(np.log1p(np.abs(F)[band_masks["veryhigh"]].sum()))

            # Channel correlations
            R = img_rgb[..., 0].ravel()
            G = img_rgb[..., 1].ravel()
            B = img_rgb[..., 2].ravel()
            def safe_corr(a, b):
                a_ = a - a.mean()
                b_ = b - b.mean()
                da = float(np.sqrt((a_*a_).sum()))
                db = float(np.sqrt((b_*b_).sum()))
                if da <= 0 or db <= 0:
                    return float("nan")
                return float((a_*b_).sum() / (da*db))
            corr_RG = safe_corr(R, G)
            corr_RB = safe_corr(R, B)
            corr_GB = safe_corr(G, B)

            # Luminance histogram entropy
            hist, _ = np.histogram(gray, bins=64, range=(0.0, 255.0))
            p = hist.astype(np.float64) / max(1, hist.sum())
            p = p[p > 0]
            ent = float(-np.sum(p * np.log2(p)))

            # Block-DCT JPEG-like blockiness proxy:
            # average absolute pixel-difference across 8x8 block boundaries minus
            # within-block. RES=224 = 28×8, so use the first 224-1 columns/rows
            # and compare boundary differences vs within-block differences.
            # (Approximation of JPEG blockiness from Wang/Bovik 2002.)
            def blockiness(g):
                g = g.astype(np.float32)
                # horizontal direction: diff between adjacent cols j and j+1
                # for j = 0..222. Boundary cols j s.t. j%8 == 7 (i.e., j=7,15,...,223
                # — last boundary index 223 not in range so cap at 215). Indices in
                # full diff array [0..222]: boundary positions = 7, 15, ..., 215
                # (28 positions); within positions = the other 195.
                h_diff = np.abs(np.diff(g, axis=1))  # (H, W-1)
                cols_full = np.arange(g.shape[1] - 1)
                h_boundary_mask = (cols_full % 8 == 7)
                h_boundary = h_diff[:, h_boundary_mask]
                h_within = h_diff[:, ~h_boundary_mask]
                v_diff = np.abs(np.diff(g, axis=0))  # (H-1, W)
                rows_full = np.arange(g.shape[0] - 1)
                v_boundary_mask = (rows_full % 8 == 7)
                v_boundary = v_diff[v_boundary_mask, :]
                v_within = v_diff[~v_boundary_mask, :]
                return float(
                    (h_boundary.mean() - h_within.mean()
                     + v_boundary.mean() - v_within.mean()) / 2.0
                )
            block = blockiness(gray)

            # Sensor-noise proxy: variance of high-pass-filtered residual
            blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
            residual = gray - blur
            sensor = float(residual.var())

            out[i, 0] = low
            out[i, 1] = mid
            out[i, 2] = high
            out[i, 3] = xhigh
            out[i, 4] = high - low
            out[i, 5] = corr_RG
            out[i, 6] = corr_RB
            out[i, 7] = corr_GB
            out[i, 8] = ent
            out[i, 9] = block
            out[i, 10] = sensor
            out[i, 11] = veryhigh_energy
        except Exception as exc:  # noqa: BLE001
            logger.warning("pixel-stats failure %s: %s", p, exc)

        now = time.time()
        if now - last_log > 5.0:
            logger.info(
                "  pixel-stats progress: %d/%d (%.1f%%) elapsed %.1fs",
                i + 1, len(paths), 100.0 * (i + 1) / len(paths), now - t0,
            )
            last_log = now
    elapsed = time.time() - t0
    logger.info("pixel-stats done in %.1fs (%d frames)", elapsed, len(paths))
    return out


PIXEL_FIELDS = [
    "px_fft_low", "px_fft_mid", "px_fft_high", "px_fft_xhigh",
    "px_fft_high_minus_low",
    "px_corr_RG", "px_corr_RB", "px_corr_GB",
    "px_hist_entropy", "px_blockiness", "px_sensor_noise",
    "px_veryhigh_energy_log",
]


def compute_or_load_pixel_stats(meta: pd.DataFrame) -> np.ndarray:
    """Cache-aware loader for pixel statistics."""
    if PIXEL_STATS_CACHE.exists():
        logger.info("loading cached pixel-stats: %s", PIXEL_STATS_CACHE)
        d = np.load(PIXEL_STATS_CACHE, allow_pickle=True)
        feats = d["pixel_stats"]
        gcs_cached = list(d["gcs_uris"])
        # Validate ordering
        meta_uris = meta["gcs_uri"].tolist()
        if gcs_cached == meta_uris:
            logger.info("pixel-stats cache validated: %d rows", len(feats))
            return feats
        logger.warning(
            "pixel-stats cache ordering mismatch (cached %d, meta %d) — recomputing",
            len(gcs_cached), len(meta_uris),
        )
    paths = meta["local_path"].tolist()
    logger.info("computing pixel-stats for %d frames", len(paths))
    pixel_stats = extract_pixel_stats(paths)
    np.savez_compressed(
        PIXEL_STATS_CACHE,
        pixel_stats=pixel_stats,
        gcs_uris=np.array(meta["gcs_uri"].tolist(), dtype=object),
        field_names=np.array(PIXEL_FIELDS, dtype=object),
    )
    logger.info("pixel-stats cache saved: %s", PIXEL_STATS_CACHE)
    return pixel_stats


# ---------------------------------------------------------------------------
# 5. Per-field univariate analyses.
# ---------------------------------------------------------------------------
def cat_r2(values: np.ndarray, proj: np.ndarray) -> Tuple[float, int, int]:
    """One-way ANOVA R²: explained-variance fraction for a categorical field.

    Returns (R², n_used, n_categories_used).
    """
    s = pd.Series(values)
    mask = s.notna()
    if mask.sum() < 5:
        return float("nan"), int(mask.sum()), 0
    s = s[mask]
    p = proj[mask.values]
    cats = s.unique()
    if len(cats) < 2:
        return 0.0, int(mask.sum()), len(cats)
    grand_mean = float(p.mean())
    ss_between = 0.0
    ss_total = float(((p - grand_mean) ** 2).sum())
    for c in cats:
        g = p[s.values == c]
        if len(g) == 0:
            continue
        ss_between += len(g) * (g.mean() - grand_mean) ** 2
    if ss_total <= 0:
        return 0.0, int(mask.sum()), len(cats)
    return float(ss_between / ss_total), int(mask.sum()), len(cats)


def cat_summary(values: np.ndarray, proj: np.ndarray, top_k: int = 10) -> List[Dict]:
    """For each top category (by count), report mean projection."""
    s = pd.Series(values)
    mask = s.notna()
    if mask.sum() == 0:
        return []
    s = s[mask]
    p = proj[mask.values]
    counts = s.value_counts()
    rows = []
    for cat, n in counts.head(top_k).items():
        g = p[s.values == cat]
        rows.append({
            "category": str(cat),
            "n": int(n),
            "mean_proj": float(g.mean()),
            "std_proj": float(g.std(ddof=1)) if len(g) > 1 else float("nan"),
            "median_proj": float(np.median(g)),
        })
    return rows


def cont_r2(values: np.ndarray, proj: np.ndarray) -> Tuple[float, float, int]:
    """Pearson r² and r between continuous field and projection."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(v) & np.isfinite(proj)
    n = int(mask.sum())
    if n < 5:
        return float("nan"), float("nan"), n
    a = v[mask]
    b = proj[mask]
    if a.std() <= 0 or b.std() <= 0:
        return 0.0, 0.0, n
    r = float(np.corrcoef(a, b)[0, 1])
    return r * r, r, n


# ---------------------------------------------------------------------------
# 6. Cumulative regression: R² for KLIEP_proj ~ all-available-metadata.
# ---------------------------------------------------------------------------
def build_design_matrix(meta: pd.DataFrame, fields: List[str], cat_fields: List[str], top_k_cat: int = 30) -> Tuple[np.ndarray, List[str]]:
    """One-hot encode `cat_fields` (capped at top_k_cat categories), include
    `fields` continuous. NaN -> 0 with an indicator column."""
    cols = []
    names = []
    for f in fields:
        if f not in meta.columns:
            continue
        v = pd.to_numeric(meta[f], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(v)
        ind = mask.astype(np.float64)
        # mean-impute (over non-na). If all na, skip.
        if mask.sum() == 0:
            continue
        mu = v[mask].mean()
        sd = v[mask].std()
        v_filled = np.where(mask, v, mu)
        if sd > 0:
            v_z = (v_filled - mu) / sd
        else:
            v_z = v_filled - mu
        cols.append(v_z)
        names.append(f)
        cols.append(ind)
        names.append(f + "__present")
    for f in cat_fields:
        if f not in meta.columns:
            continue
        s = meta[f].astype("string").fillna("__missing__")
        counts = s.value_counts()
        # cap at top_k_cat to avoid identity-key explosion
        top_cats = list(counts.head(top_k_cat).index)
        for cat in top_cats:
            cols.append((s == cat).astype(np.float64).to_numpy())
            names.append(f"{f}={cat}")
    if not cols:
        return np.zeros((len(meta), 0), dtype=np.float64), names
    X = np.stack(cols, axis=1)
    return X, names


def cumulative_r2(X: np.ndarray, y: np.ndarray) -> Dict:
    """Fit Ridge(alpha=1.0, n_jobs=1) and return R² (training) + n_features.

    Ridge instead of LinearRegression because the design matrix for the larger
    models is rank-deficient (one-hot identity categoricals + low-support
    levels). Reported R² is in-sample R² of the ridge fit; cumulative_r2_cv
    is the load-bearing generalization statistic.
    """
    from sklearn.linear_model import Ridge
    if X.shape[1] == 0:
        return {"r2": float("nan"), "n_features": 0}
    reg = Ridge(alpha=1.0)
    reg.fit(X, y)
    pred = reg.predict(X)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {"r2": r2, "n_features": int(X.shape[1])}


def cumulative_r2_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict:
    """5-fold CV R² (held-out prediction) using Ridge(alpha=1.0)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    if X.shape[1] == 0:
        return {"cv_r2": float("nan"), "n_features": 0}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    preds = np.zeros_like(y)
    for tr, te in kf.split(X):
        reg = Ridge(alpha=1.0)
        reg.fit(X[tr], y[tr])
        preds[te] = reg.predict(X[te])
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    cv_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {"cv_r2": cv_r2, "n_features": int(X.shape[1])}


# ---------------------------------------------------------------------------
# 7. Main.
# ---------------------------------------------------------------------------
def main():
    setup_logging()
    t_start = time.time()
    logger.info("D11 — name the dev-vs-lockbox KLIEP axis — start")

    # ----- load features + metadata
    pools = load_real_features()
    feats = pools["feats"]
    meta = pools["meta"]
    is_lockbox = pools["is_lockbox"]
    logger.info("feats shape: %s; is_lockbox: dev=%d lockbox=%d",
                feats.shape, int((is_lockbox == 0).sum()), int((is_lockbox == 1).sum()))

    # ----- refit KLIEP on the same n=2414 set
    kliep = fit_kliep(feats, is_lockbox)
    proj = kliep["proj"]
    logger.info("KLIEP refit: balanced-acc=%.4f |w|=%.4f b=%.4f",
                kliep["accuracy"], float(np.linalg.norm(kliep["w"])), kliep["b"])
    # ----- sanity check: pool means should match D10
    logger.info("  dev_real mean proj: %.4f", float(proj[is_lockbox == 0].mean()))
    logger.info("  lockbox_real mean proj: %.4f", float(proj[is_lockbox == 1].mean()))

    # ----- merge manifest + atlas
    meta_full = merge_metadata(meta)
    meta_full["kliep_proj"] = proj
    meta_full["is_lockbox"] = is_lockbox

    # ----- compute pixel-level statistics
    pixel_stats = compute_or_load_pixel_stats(meta_full)
    for j, name in enumerate(PIXEL_FIELDS):
        meta_full[name] = pixel_stats[:, j]

    # ----- categorical and continuous field definitions
    CAT_FIELDS = [
        "identity_key", "session_id", "method", "video_id",
        "clip_capture_mode", "clip_quality", "clip_occlusion", "clip_lighting",
        "image_mode", "actual_format", "is_low_quality", "is_pose_extreme",
        "is_no_face", "is_likely_screen_capture", "has_alpha",
        "manifest_source_kind", "manifest_slices", "manifest_prefix",
        "manifest_sequence_id", "manifest_method",
    ]
    CONT_FIELDS = [
        "width", "height", "aspect_ratio",
        "sharpness_laplacian", "brightness_v_mean", "brightness_v_std",
        "contrast_rms", "saturation_s_mean",
        "is_clipped_highlights", "face_count",
        "face_pixel_area", "face_area_ratio",
        "face_bbox_x", "face_bbox_y", "face_bbox_w", "face_bbox_h",
        "yaw_deg", "pitch_deg", "roll_deg",
        "eye_aspect_ratio_left", "eye_aspect_ratio_right", "mouth_aspect_ratio",
        "arcface_norm",
        "clip_quality_prob", "clip_capture_mode_prob",
        "clip_occlusion_prob", "clip_lighting_prob",
        "file_bytes",
        "atlas_lap_var", "atlas_min_dim", "atlas_max_dim", "atlas_luma_mean",
        "atlas_luma_std", "atlas_edge_mag", "atlas_saturation_mean",
        "atlas_contrast_l", "atlas_color_a_dev", "atlas_color_b_dev",
        "atlas_skin_frac", "atlas_bytes",
    ] + PIXEL_FIELDS

    # ----- per-categorical R²
    # Note: extremely-high-cardinality fields (k > n/8) and dev/lockbox-disjoint
    # fields (n_categories_shared == 0) achieve high R² by construction
    # (overfit / definitional separation) and should be read accordingly.
    logger.info("[5a] categorical R² scan")
    cat_rows = []
    for f in CAT_FIELDS:
        if f not in meta_full.columns:
            continue
        r2, n, k = cat_r2(meta_full[f].to_numpy(), proj)
        # Also report dev vs lockbox category overlap fraction
        if f in meta_full.columns:
            s_dev = meta_full.loc[meta_full["is_lockbox"] == 0, f].dropna().unique()
            s_lb = meta_full.loc[meta_full["is_lockbox"] == 1, f].dropna().unique()
            shared = set(map(str, s_dev)) & set(map(str, s_lb))
            n_shared = len(shared)
        else:
            n_shared = 0
        # high_cardinality flag: degenerate near-perfect fit warning
        hc = (k > max(1, n // 8))
        cat_rows.append({
            "field": f,
            "n_used": n,
            "n_categories": k,
            "r2_anova": r2,
            "n_categories_shared_dev_lockbox": n_shared,
            "high_cardinality_flag": bool(hc),
            "dev_lockbox_disjoint_categories_flag": bool(n_shared == 0 and k > 1),
        })
        logger.info("  cat %s: n=%d, k=%d, shared=%d, R²=%.4f%s",
                    f, n, k, n_shared, r2,
                    " [HC]" if hc else (" [DISJOINT]" if (n_shared == 0 and k > 1) else ""))
    df_cat = pd.DataFrame(cat_rows).sort_values("r2_anova", ascending=False)
    df_cat.to_csv(OUTPUTS / "cat_field_anova.csv", index=False)

    # ----- per-categorical mean proj for top categories (for the largest ones)
    logger.info("[5b] per-categorical mean proj — top categories per field")
    cat_summary_rows = []
    for f in df_cat.head(8)["field"]:  # top 8 fields by R²
        rows = cat_summary(meta_full[f].to_numpy(), proj, top_k=12)
        for r in rows:
            r["field"] = f
            cat_summary_rows.append(r)
    pd.DataFrame(cat_summary_rows).to_csv(OUTPUTS / "cat_field_top_categories.csv", index=False)

    # ----- per-continuous R²
    logger.info("[5c] continuous R² + Pearson r scan")
    cont_rows = []
    for f in CONT_FIELDS:
        if f not in meta_full.columns:
            continue
        r2, r, n = cont_r2(meta_full[f].to_numpy(), proj)
        cont_rows.append({
            "field": f,
            "n_used": n,
            "pearson_r": r,
            "r2": r2,
        })
        logger.info("  cont %s: n=%d, r=%.4f, R²=%.4f", f, n, r, r2)
    df_cont = pd.DataFrame(cont_rows).sort_values("r2", ascending=False)
    df_cont.to_csv(OUTPUTS / "cont_field_correlation.csv", index=False)

    # ----- cumulative regression models
    logger.info("[6] cumulative regression models")
    y = proj.astype(np.float64)

    # Strategy: build progressively richer designs.
    MANIFEST_ONLY_CONT = [
        "width", "height", "aspect_ratio",
        "sharpness_laplacian", "brightness_v_mean", "brightness_v_std",
        "contrast_rms", "saturation_s_mean",
        "is_clipped_highlights", "face_count",
        "face_pixel_area", "face_area_ratio",
        "face_bbox_x", "face_bbox_y", "face_bbox_w", "face_bbox_h",
        "yaw_deg", "pitch_deg", "roll_deg",
        "eye_aspect_ratio_left", "eye_aspect_ratio_right", "mouth_aspect_ratio",
        "arcface_norm",
        "clip_quality_prob", "clip_capture_mode_prob",
        "clip_occlusion_prob", "clip_lighting_prob",
        "file_bytes",
    ]
    MANIFEST_ONLY_CAT = [
        "clip_capture_mode", "clip_quality", "clip_occlusion", "clip_lighting",
        "is_low_quality", "is_pose_extreme", "is_no_face", "is_likely_screen_capture",
        "has_alpha", "manifest_source_kind", "manifest_slices", "manifest_prefix",
    ]
    # NOTE: video_id (k=2054 on n=2414) and manifest_sequence_id (k=555 on
    # n=592) are excluded from the cumulative design because they are
    # near-bijective with frames and would explain ~100% of any per-frame target
    # by construction. They are still scored in the per-field §5a R² scan with
    # the high_cardinality_flag.
    IDENTITY_CAT = ["identity_key", "session_id", "method", "manifest_method"]

    # 6a — manifest IQ / clip-tag fields only (no identity, no pixel-level extras)
    X1, n1 = build_design_matrix(meta_full, MANIFEST_ONLY_CONT, MANIFEST_ONLY_CAT, top_k_cat=30)
    r1_full = cumulative_r2(X1, y)
    r1_cv = cumulative_r2_cv(X1, y, n_splits=5)
    logger.info("  (a) manifest+clip-tag (no identity, no pixel-extras): "
                "R²=%.4f cv_R²=%.4f n_features=%d",
                r1_full["r2"], r1_cv["cv_r2"], r1_full["n_features"])

    # 6b — add identity-class categoricals
    X2, n2 = build_design_matrix(
        meta_full, MANIFEST_ONLY_CONT,
        MANIFEST_ONLY_CAT + IDENTITY_CAT,
        top_k_cat=30,
    )
    r2_full = cumulative_r2(X2, y)
    r2_cv = cumulative_r2_cv(X2, y, n_splits=5)
    logger.info("  (b) +identity categoricals (top 30 per field): "
                "R²=%.4f cv_R²=%.4f n_features=%d",
                r2_full["r2"], r2_cv["cv_r2"], r2_full["n_features"])

    # 6c — add atlas IQ continuous fields
    X3, n3 = build_design_matrix(
        meta_full,
        MANIFEST_ONLY_CONT + [
            "atlas_lap_var", "atlas_min_dim", "atlas_max_dim", "atlas_luma_mean",
            "atlas_luma_std", "atlas_edge_mag", "atlas_saturation_mean",
            "atlas_contrast_l", "atlas_color_a_dev", "atlas_color_b_dev",
            "atlas_skin_frac", "atlas_bytes",
        ],
        MANIFEST_ONLY_CAT + IDENTITY_CAT,
        top_k_cat=30,
    )
    r3_full = cumulative_r2(X3, y)
    r3_cv = cumulative_r2_cv(X3, y, n_splits=5)
    logger.info("  (c) +atlas-IQ-extras: R²=%.4f cv_R²=%.4f n_features=%d",
                r3_full["r2"], r3_cv["cv_r2"], r3_full["n_features"])

    # 6d — add pixel-level extras
    X4, n4 = build_design_matrix(
        meta_full,
        MANIFEST_ONLY_CONT + [
            "atlas_lap_var", "atlas_min_dim", "atlas_max_dim", "atlas_luma_mean",
            "atlas_luma_std", "atlas_edge_mag", "atlas_saturation_mean",
            "atlas_contrast_l", "atlas_color_a_dev", "atlas_color_b_dev",
            "atlas_skin_frac", "atlas_bytes",
        ] + PIXEL_FIELDS,
        MANIFEST_ONLY_CAT + IDENTITY_CAT,
        top_k_cat=30,
    )
    r4_full = cumulative_r2(X4, y)
    r4_cv = cumulative_r2_cv(X4, y, n_splits=5)
    logger.info("  (d) +pixel-level FFT/corr/sensor/blockiness: "
                "R²=%.4f cv_R²=%.4f n_features=%d",
                r4_full["r2"], r4_cv["cv_r2"], r4_full["n_features"])

    # 6e — no-identity baseline (manifest + atlas + pixel, no identity categoricals)
    X5, n5 = build_design_matrix(
        meta_full,
        MANIFEST_ONLY_CONT + [
            "atlas_lap_var", "atlas_min_dim", "atlas_max_dim", "atlas_luma_mean",
            "atlas_luma_std", "atlas_edge_mag", "atlas_saturation_mean",
            "atlas_contrast_l", "atlas_color_a_dev", "atlas_color_b_dev",
            "atlas_skin_frac", "atlas_bytes",
        ] + PIXEL_FIELDS,
        MANIFEST_ONLY_CAT,
        top_k_cat=30,
    )
    r5_full = cumulative_r2(X5, y)
    r5_cv = cumulative_r2_cv(X5, y, n_splits=5)
    logger.info("  (e) all-but-identity: R²=%.4f cv_R²=%.4f n_features=%d",
                r5_full["r2"], r5_cv["cv_r2"], r5_full["n_features"])

    cumulative_rows = [
        {"model": "a_manifest_clip_tag", "n_features": r1_full["n_features"],
         "r2_full": r1_full["r2"], "r2_cv5": r1_cv["cv_r2"],
         "uses_identity": False, "uses_pixel_extras": False,
         "uses_atlas_iq": False},
        {"model": "b_plus_identity_cat", "n_features": r2_full["n_features"],
         "r2_full": r2_full["r2"], "r2_cv5": r2_cv["cv_r2"],
         "uses_identity": True, "uses_pixel_extras": False,
         "uses_atlas_iq": False},
        {"model": "c_plus_atlas_iq", "n_features": r3_full["n_features"],
         "r2_full": r3_full["r2"], "r2_cv5": r3_cv["cv_r2"],
         "uses_identity": True, "uses_pixel_extras": False,
         "uses_atlas_iq": True},
        {"model": "d_plus_pixel_extras", "n_features": r4_full["n_features"],
         "r2_full": r4_full["r2"], "r2_cv5": r4_cv["cv_r2"],
         "uses_identity": True, "uses_pixel_extras": True,
         "uses_atlas_iq": True},
        {"model": "e_all_minus_identity", "n_features": r5_full["n_features"],
         "r2_full": r5_full["r2"], "r2_cv5": r5_cv["cv_r2"],
         "uses_identity": False, "uses_pixel_extras": True,
         "uses_atlas_iq": True},
    ]
    df_cum = pd.DataFrame(cumulative_rows)
    df_cum.to_csv(OUTPUTS / "cumulative_regression_models.csv", index=False)

    # ----- extreme frames
    logger.info("[7] extreme-frame inspection")
    order = np.argsort(proj)
    extreme_rows = []
    EX_COLS = [
        "gcs_uri", "local_path", "pool", "is_lockbox", "kliep_proj",
        "identity_key", "session_id", "method", "video_id",
        "manifest_source_kind", "manifest_slices",
        "clip_capture_mode", "clip_quality", "clip_occlusion", "clip_lighting",
        "width", "height", "aspect_ratio", "face_pixel_area", "face_area_ratio",
        "sharpness_laplacian", "yaw_deg", "pitch_deg", "roll_deg",
        "brightness_v_mean", "brightness_v_std", "contrast_rms",
        "arcface_norm", "file_bytes",
    ] + PIXEL_FIELDS
    EX_COLS = [c for c in EX_COLS if c in meta_full.columns]
    for rank, ix in enumerate(order[:5]):
        row = {"rank": rank + 1, "side": "most_dev_aligned"}
        for c in EX_COLS:
            v = meta_full.iloc[ix][c]
            row[c] = float(v) if isinstance(v, (np.floating, np.integer, float, int)) and not isinstance(v, bool) else (str(v) if v is not None else "")
        extreme_rows.append(row)
    for rank, ix in enumerate(order[-5:][::-1]):
        row = {"rank": rank + 1, "side": "most_lockbox_aligned"}
        for c in EX_COLS:
            v = meta_full.iloc[ix][c]
            row[c] = float(v) if isinstance(v, (np.floating, np.integer, float, int)) and not isinstance(v, bool) else (str(v) if v is not None else "")
        extreme_rows.append(row)
    pd.DataFrame(extreme_rows).to_csv(OUTPUTS / "extreme_frames.csv", index=False)
    for r in extreme_rows[:5]:
        logger.info("  most-dev  rank %d proj=%.3f  id=%s  cap=%s",
                    r["rank"], r["kliep_proj"], r.get("identity_key", ""),
                    r.get("clip_capture_mode", ""))
    for r in extreme_rows[5:]:
        logger.info("  most-lock rank %d proj=%.3f  id=%s  cap=%s",
                    r["rank"], r["kliep_proj"], r.get("identity_key", ""),
                    r.get("clip_capture_mode", ""))

    # ----- save per-frame projection+metadata
    KEEP_COLS = ["gcs_uri", "local_path", "pool", "is_lockbox", "kliep_proj",
                 "identity_key", "session_id", "method", "video_id",
                 "manifest_source_kind", "manifest_slices",
                 "clip_capture_mode", "clip_quality", "clip_occlusion", "clip_lighting",
                 "width", "height", "aspect_ratio", "face_pixel_area",
                 "face_area_ratio", "sharpness_laplacian",
                 "yaw_deg", "pitch_deg", "roll_deg",
                 "brightness_v_mean", "brightness_v_std", "contrast_rms",
                 "saturation_s_mean", "arcface_norm",
                 "clip_quality_prob", "clip_capture_mode_prob",
                 "clip_occlusion_prob", "clip_lighting_prob",
                 "is_low_quality", "is_pose_extreme", "is_no_face",
                 "is_likely_screen_capture",
                 "atlas_lap_var", "atlas_min_dim", "atlas_color_a_dev",
                 "atlas_color_b_dev", "atlas_saturation_mean", "atlas_luma_mean",
                 "atlas_edge_mag", "atlas_contrast_l", "atlas_skin_frac",
                 "file_bytes",
                 ] + PIXEL_FIELDS
    KEEP_COLS = [c for c in KEEP_COLS if c in meta_full.columns]
    meta_full[KEEP_COLS].to_csv(OUTPUTS / "per_frame_projection_metadata.csv", index=False)

    # ----- top-N highest-R² fields summary
    top_cat = df_cat.head(10).to_dict(orient="records")
    top_cont = df_cont.head(10).to_dict(orient="records")
    summary = {
        "n_dev_real": int((is_lockbox == 0).sum()),
        "n_lockbox_real": int((is_lockbox == 1).sum()),
        "kliep_balanced_accuracy": float(kliep["accuracy"]),
        "kliep_w_norm": float(np.linalg.norm(kliep["w"])),
        "kliep_b": float(kliep["b"]),
        "dev_real_mean_proj": float(proj[is_lockbox == 0].mean()),
        "lockbox_real_mean_proj": float(proj[is_lockbox == 1].mean()),
        "top10_categorical_by_r2": top_cat,
        "top10_continuous_by_r2": top_cont,
        "cumulative_models": cumulative_rows,
    }
    with open(OUTPUTS / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: str(x))

    logger.info("D11 done in %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
