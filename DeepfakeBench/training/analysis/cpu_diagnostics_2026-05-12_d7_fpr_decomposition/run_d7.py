"""D7 — FPR decomposition by substrate_distance vs IQ vector vs chronic_indicator.

Task brief
----------
For P8A_step5000 (and T5C_step3500 if features are cached), decompose eval-real
FPR at dev-calibrated 5% τ into three feature blocks:

  1. substrate_distance: mean cos-dist in CLIP-frozen B16 L11 feature space to
     k=10 nearest training reals. Reference set = subsample of ~2000 training
     reals (constrained by local-cache availability).
  2. IQ vector: 6-axis IQ position (lap_var, min_dim, color_a_dev, color_b_dev,
     saturation_mean, luma_mean). Z-scored.
  3. chronic_indicator: binary, identity ∈ chronic_6.

Outputs
-------
- outputs/per_frame_features.csv  (per eval real)
- outputs/per_block_partial_r2.csv
- outputs/coefficient_table.csv
- outputs/block_correlation_matrix.csv
- outputs/_summary.json

Methodology notes
-----------------
- CLIP_FROZEN extraction uses MPS device, OpenCLIP ViT-B-16 with local cached
  weights (CLIP-ViT-B-16-DataComp.XL-s13B-b90K).
- substrate_distance = mean cos-dist to k=10 NN training reals (cos-dist =
  1 - cos-sim). One scalar per eval frame.
- Dev-cal τ: 95th percentile of P8A/T5C scores on `teams_real_all_dev`.
- is_FP: real frames in any eval suite scoring > τ.
- Logistic regression: sklearn.LogisticRegression, C=1.0, max_iter=2000,
  n_jobs=1, solver=lbfgs.
- Nested-model partial-R²: full model McFadden pseudo-R² minus reduced-model
  pseudo-R² (drop one block at a time). Also report ΔAUC.
- Pearson correlation between blocks: substrate_distance scalar, IQ-PC1 scalar,
  chronic_indicator scalar.
- All sklearn n_jobs=1 per `feedback_sklearn_njobs.md`.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
THIS_DIR = REPO_ROOT / "analysis" / "cpu_diagnostics_2026-05-12_d7_fpr_decomposition"
OUT_DIR = THIS_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = THIS_DIR / "_run.log"

UNIFIED_MATRIX = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-12_stage_a/outputs/unified_frame_matrix.csv"
TRAIN_REAL_POOL_PARQUET = REPO_ROOT / "analysis/iq_data_atlas_2026-05-08/_cache/train_teams_real_pool.parquet"

CLIP_WEIGHTS = REPO_ROOT / "weights/CLIP-ViT-B-16-DataComp.XL-s13B-b90K/open_clip_pytorch_model.bin"

# CLIP normalization
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# 6 IQ axes
IQ_AXES = ["lap_var", "min_dim", "color_a_dev", "color_b_dev", "saturation_mean", "luma_mean"]

# Chronic-6 identities
CHRONIC_6_TOKENS = ["Roy_D", "PC_Generator", "bla_bla_chow",
                    "Md_noyn_Sharker", "dor_shkedi", "healthy_dor"]

# Calibration suite (dev real, broad cohort)
CALIB_SUITE = "teams_real_all_dev"

# Real eval suites (excluding calibration, but we'll include them all and report)
REAL_SUITES = [
    "teams_real_all_dev",
    "teams_real_lighting_extreme_dev",
    "teams_real_all_lockbox",
    "teams_real_poor_quality_dev",
    "teams_real_dor_dev",
]

CKPTS = ["P8A", "T5C_step3500"]

# Logistic regression hyper-parameters
LR_KW = dict(C=1.0, max_iter=2000, n_jobs=1, solver="lbfgs")

# Substrate-distance NN
K_NN = 10

SEED = 42
N_TRAIN_REAL_REFERENCE = 2000  # task spec — but capped by local availability

# Frame caches (locally cached jpeg/png)
FRAME_CACHE_DIRS = [
    "analysis/lockbox_tagging/_frame_cache",
    "analysis/job7_head_retrain_2026-05-04/_frame_cache",
    "analysis/codec_aug_verification_2026-05-05/_frame_cache",
    "analysis/skin_frac_viso_gap_2026-05-03/_frame_cache",
    "analysis/move1_frozen_probe_2026-05-01/_frame_cache",
    "analysis/clip_vs_p8a_viso_2026-05-03/_frame_cache",
    "analysis/p2_eval_2026-05-08/d1_d4_cpu/_frame_cache",
    "analysis/p1_pe_eval_2026-05-07/f3_color_b_dev/_frame_cache",
    "analysis/pa_pc_eval_2026-05-05/_frame_cache_hard_suites",
]

# CLIP_FROZEN cache (we cache features here so reruns are cheap)
CLIP_FEATS_CACHE = THIS_DIR / "_clip_l11_features.npz"


def setup_logging() -> logging.Logger:
    fmt = "%(asctime)s %(levelname)s %(name)s :: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler(sys.stdout),
    ])
    return logging.getLogger("d7-fpr-decomp")


def build_local_index(logger: logging.Logger) -> Dict[str, str]:
    """basename -> abs local path across all known frame caches."""
    cache_files: List[str] = []
    for d in FRAME_CACHE_DIRS:
        full = REPO_ROOT / d
        if not full.exists():
            logger.info("  cache dir missing: %s (skipping)", full)
            continue
        n_before = len(cache_files)
        cache_files.extend(glob.glob(str(full / "**/*.jpg"), recursive=True))
        cache_files.extend(glob.glob(str(full / "**/*.png"), recursive=True))
        logger.info("  cache dir %s: %d files", d, len(cache_files) - n_before)
    basename_to_path: Dict[str, str] = {}
    for p in cache_files:
        basename_to_path[os.path.basename(p)] = p
    logger.info("  total unique cached basenames: %d", len(basename_to_path))
    return basename_to_path


def load_eval_panel(logger: logging.Logger, local_index: Dict[str, str]) -> pd.DataFrame:
    """Load unified frame matrix, filter to real eval frames with local path
    AND complete 6-IQ row. Attach is_chronic_6 (recomputed for safety)."""
    df = pd.read_csv(UNIFIED_MATRIX)
    logger.info("[panel] unified matrix rows=%d", len(df))
    # Real frames only
    df = df[df["label"] == 0].copy()
    logger.info("[panel] real eval rows=%d", len(df))
    # Attach local path
    df["basename"] = df["frame_path"].str.split("/").str[-1]
    df["local_path"] = df["basename"].map(local_index)
    n_with_local = df["local_path"].notna().sum()
    logger.info("[panel] real rows with local file: %d/%d", n_with_local, len(df))
    # Require all 6 IQ axes present
    df = df.dropna(subset=IQ_AXES).copy()
    logger.info("[panel] real with full IQ panel: %d", len(df))
    # Require local file present
    df = df[df["local_path"].notna()].copy()
    logger.info("[panel] real with local file AND IQ: %d", len(df))

    # is_chronic_6 (use the column from unified matrix if present, else derive)
    if "is_chronic_6" in df.columns:
        logger.info("[panel] using existing is_chronic_6 col")
    else:
        pat = "|".join(CHRONIC_6_TOKENS)
        df["is_chronic_6"] = df["frame_path"].fillna("").str.contains(
            pat, case=False, regex=True
        ).astype(int)
    n_chronic = int(df["is_chronic_6"].sum())
    logger.info("[panel] chronic_6 frames: %d/%d", n_chronic, len(df))
    df.reset_index(drop=True, inplace=True)
    df["row_ix"] = np.arange(len(df))
    return df


def load_training_reals(logger: logging.Logger, local_index: Dict[str, str]) -> pd.DataFrame:
    """Load training real pool from iq_data_atlas cache + filter to locally cached frames."""
    train_pool = pd.read_parquet(TRAIN_REAL_POOL_PARQUET)
    train_pool["basename"] = train_pool["frame_path"].str.split("/").str[-1]
    train_pool["local_path"] = train_pool["basename"].map(local_index)
    n_with_local = train_pool["local_path"].notna().sum()
    logger.info("[train_real] %d in train_teams_real_pool; %d locally cached",
                len(train_pool), n_with_local)
    train_pool = train_pool[train_pool["local_path"].notna()].reset_index(drop=True)
    return train_pool


def extract_clip_l11_features(
    paths: List[str],
    logger: logging.Logger,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """OpenCLIP ViT-B/16 L11 CLS feature extraction.

    Returns (features [N, 768], valid_idx [N] — indices into input list)
    """
    import torch
    import open_clip
    import cv2

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("[clip] device: %s", device)

    if not CLIP_WEIGHTS.exists():
        raise FileNotFoundError(f"missing CLIP weights: {CLIP_WEIGHTS}")
    logger.info("[clip] loading weights: %s", CLIP_WEIGHTS)
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(CLIP_WEIGHTS)
    )
    model = model.to(device).eval()
    visual = model.visual

    captured: List[np.ndarray] = []

    def hook(_module, _inp, output):
        if output.dim() == 3:
            if output.shape[0] >= output.shape[1]:
                cls = output[0]
            else:
                cls = output[:, 0]
        elif output.dim() == 2:
            cls = output
        else:
            raise RuntimeError(f"unexpected output: {tuple(output.shape)}")
        captured.append(cls.detach().cpu().to(torch.float32).numpy())

    handle = visual.transformer.resblocks[11].register_forward_hook(hook)

    feats_per_frame: List[Optional[np.ndarray]] = [None] * len(paths)
    pending: List[Tuple[int, "torch.Tensor"]] = []

    # Preprocess
    t_pre = time.time()
    for i, p in enumerate(paths):
        try:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
            pending.append((i, torch.from_numpy(img.transpose(2, 0, 1))))
        except Exception as e:
            logger.warning("  preprocess failed for %s: %s", p, e)
    t_pre = time.time() - t_pre
    logger.info("[clip] %d/%d frames preprocessed (%.1f s)", len(pending), len(paths), t_pre)

    # Batched forward
    t_fwd = time.time()
    n_batches = (len(pending) + batch_size - 1) // batch_size
    for b in range(n_batches):
        chunk = pending[b * batch_size : (b + 1) * batch_size]
        batch_idx = [c[0] for c in chunk]
        batch = torch.stack([c[1] for c in chunk]).to(device, non_blocking=True)
        captured.clear()
        with torch.inference_mode():
            _ = visual(batch)
        if not captured:
            logger.warning("  empty capture for batch %d", b)
            continue
        cls_batch = captured[0]
        if cls_batch.shape[0] != len(batch_idx):
            logger.warning("  capture shape mismatch (%d vs %d) at batch %d",
                           cls_batch.shape[0], len(batch_idx), b)
            continue
        for k, ii in enumerate(batch_idx):
            feats_per_frame[ii] = cls_batch[k]
        if b % 20 == 0:
            logger.info("  batch %d/%d", b + 1, n_batches)
    handle.remove()
    t_fwd = time.time() - t_fwd
    logger.info("[clip] forward done (%.1f s)", t_fwd)

    valid_idx = np.array([i for i, f in enumerate(feats_per_frame) if f is not None],
                         dtype=np.int64)
    if len(valid_idx) == 0:
        return np.zeros((0, 768), dtype=np.float32), valid_idx
    arr = np.stack([feats_per_frame[i] for i in valid_idx], axis=0).astype(np.float32)
    logger.info("[clip] extracted %d/%d features", arr.shape[0], len(paths))
    return arr, valid_idx


def compute_substrate_distance(
    eval_feats: np.ndarray,
    train_feats: np.ndarray,
    k: int,
    logger: logging.Logger,
) -> np.ndarray:
    """For each eval feature, mean cosine distance to its k-NN train features.

    Returns array of shape (n_eval,).
    """
    if eval_feats.shape[1] != train_feats.shape[1]:
        raise ValueError(f"feature-dim mismatch eval={eval_feats.shape[1]} vs train={train_feats.shape[1]}")
    # L2-normalize
    eval_norm = eval_feats / np.linalg.norm(eval_feats, axis=1, keepdims=True).clip(min=1e-12)
    train_norm = train_feats / np.linalg.norm(train_feats, axis=1, keepdims=True).clip(min=1e-12)
    # cos sim matrix (n_eval, n_train)
    cs = eval_norm @ train_norm.T
    # cos dist = 1 - cos sim, take k smallest distances (= k largest sims)
    # We pick top-k sims per row, take mean, convert to distance.
    k_eff = min(k, cs.shape[1])
    top_k_sims = np.partition(cs, -k_eff, axis=1)[:, -k_eff:]
    mean_sim_topk = top_k_sims.mean(axis=1)
    mean_dist_topk = 1.0 - mean_sim_topk
    logger.info("[substrate] eval=%d train=%d k=%d  mean dist=%.4f  min=%.4f  max=%.4f",
                eval_feats.shape[0], train_feats.shape[0], k_eff,
                float(mean_dist_topk.mean()), float(mean_dist_topk.min()),
                float(mean_dist_topk.max()))
    return mean_dist_topk.astype(np.float64)


def fit_lr_pseudoR2(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = SEED,
) -> Dict[str, float]:
    """Fit LR with intercept, return McFadden pseudo-R², deviance, AUC.

    McFadden pseudo-R^2 = 1 - LL_model / LL_null
    LL_null = log-likelihood of intercept-only model (= base-rate prediction)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, log_loss

    n = len(y)
    p_base = float(y.mean())
    # LL_null using base-rate p_base for every sample
    eps = 1e-12
    ll_null = -log_loss(y, np.full(n, p_base), labels=[0, 1], normalize=False)

    if X.shape[1] == 0:
        # Intercept-only model — same as null
        ll_model = ll_null
        auc = 0.5
    else:
        clf = LogisticRegression(**LR_KW)
        clf.fit(X, y)
        p_hat = clf.predict_proba(X)[:, 1].clip(eps, 1 - eps)
        ll_model = -log_loss(y, p_hat, labels=[0, 1], normalize=False)
        try:
            auc = float(roc_auc_score(y, p_hat))
        except Exception:
            auc = float("nan")

    deviance_model = -2.0 * ll_model
    deviance_null = -2.0 * ll_null
    if ll_null == 0:
        mcfadden = float("nan")
    else:
        mcfadden = 1.0 - (ll_model / ll_null)
    return {
        "n": n,
        "k_features": int(X.shape[1]),
        "ll_model": float(ll_model),
        "ll_null": float(ll_null),
        "deviance_model": float(deviance_model),
        "deviance_null": float(deviance_null),
        "mcfadden_r2": float(mcfadden),
        "auc": float(auc),
        "p_base": p_base,
    }


def fit_lr_full(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, float, "sklearn.linear_model.LogisticRegression"]:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(**LR_KW)
    clf.fit(X, y)
    coefs = clf.coef_[0]
    intercept = float(clf.intercept_[0])
    return coefs, intercept, clf


def standardize_columns(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score columns. Returns (Z, mu, sd)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd_safe = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd_safe, mu, sd


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    logger = setup_logging()
    t0 = time.time()
    logger.info("=" * 80)
    logger.info("D7 — FPR decomposition: substrate × IQ × chronic")
    logger.info("=" * 80)

    # ---- 1. Build local frame index ----
    logger.info("[step 1] building local frame index")
    local_index = build_local_index(logger)

    # ---- 2. Load eval panel + training-real reference ----
    logger.info("[step 2] loading eval panel")
    eval_df = load_eval_panel(logger, local_index)
    logger.info("[step 2] loading training-real reference")
    train_real = load_training_reals(logger, local_index)

    # Subsample training reals to N_TRAIN_REAL_REFERENCE (capped at available)
    n_train_avail = len(train_real)
    n_train_use = min(N_TRAIN_REAL_REFERENCE, n_train_avail)
    rng = np.random.default_rng(SEED)
    if n_train_use < n_train_avail:
        idx = rng.choice(n_train_avail, size=n_train_use, replace=False)
        train_real = train_real.iloc[idx].reset_index(drop=True)
    logger.info("[step 2] using %d training reals as reference set", len(train_real))

    # ---- 3. Calibrate τ_5pct per ckpt on teams_real_all_dev ----
    logger.info("[step 3] calibrating dev_5pct τ per ckpt")
    full_uni = pd.read_csv(UNIFIED_MATRIX)
    calib_pool = full_uni[full_uni["suite"] == CALIB_SUITE].copy()
    tau_5pct: Dict[str, float] = {}
    for ckpt in CKPTS:
        scores = calib_pool[ckpt].dropna().values
        tau = float(np.quantile(scores, 0.95))
        tau_5pct[ckpt] = tau
        logger.info("  %s: tau_5pct = %.6f (n_calib=%d)", ckpt, tau, len(scores))

    # ---- 4. Compute is_FP per ckpt per eval real ----
    logger.info("[step 4] computing is_FP per eval real")
    for ckpt in CKPTS:
        eval_df[f"is_FP_{ckpt}"] = (eval_df[ckpt] > tau_5pct[ckpt]).astype(int)
    for ckpt in CKPTS:
        n_fp = int(eval_df[f"is_FP_{ckpt}"].sum())
        rate = n_fp / len(eval_df) * 100
        logger.info("  %s: %d FP / %d = %.2f%%", ckpt, n_fp, len(eval_df), rate)

    # ---- 5. Extract CLIP_FROZEN L11 features for eval reals + training reals ----
    logger.info("[step 5] CLIP_FROZEN L11 feature extraction")
    # Cache file holds both eval and train features.
    if CLIP_FEATS_CACHE.exists():
        logger.info("  cache hit: %s — loading", CLIP_FEATS_CACHE)
        blob = np.load(CLIP_FEATS_CACHE, allow_pickle=True)
        eval_feats = blob["eval_feats"]
        eval_valid_idx = blob["eval_valid_idx"]
        train_feats = blob["train_feats"]
        train_valid_idx = blob["train_valid_idx"]
        cached_eval_basenames = list(blob["eval_basenames"])
        cached_train_basenames = list(blob["train_basenames"])
        # Sanity-check that cached eval set matches current eval_df
        if (cached_eval_basenames != eval_df["basename"].tolist()
                or cached_train_basenames != train_real["basename"].tolist()):
            logger.warning("  cached basenames mismatch — re-extracting")
            CLIP_FEATS_CACHE.unlink()
        else:
            logger.info("  cache loaded: eval=%d feats, train=%d feats",
                        eval_feats.shape[0], train_feats.shape[0])
    if not CLIP_FEATS_CACHE.exists():
        # Eval features
        logger.info("  extracting eval features (n=%d)", len(eval_df))
        eval_paths = eval_df["local_path"].tolist()
        eval_feats, eval_valid_idx = extract_clip_l11_features(eval_paths, logger)
        # Training features
        logger.info("  extracting training features (n=%d)", len(train_real))
        train_paths = train_real["local_path"].tolist()
        train_feats, train_valid_idx = extract_clip_l11_features(train_paths, logger)
        np.savez_compressed(
            CLIP_FEATS_CACHE,
            eval_feats=eval_feats,
            eval_valid_idx=eval_valid_idx,
            train_feats=train_feats,
            train_valid_idx=train_valid_idx,
            eval_basenames=np.array(eval_df["basename"].tolist(), dtype=object),
            train_basenames=np.array(train_real["basename"].tolist(), dtype=object),
        )
        logger.info("  cached to %s", CLIP_FEATS_CACHE)

    # Map eval features back into eval_df via valid_idx
    if eval_feats.shape[0] != len(eval_df):
        logger.warning("  eval feature count (%d) != eval_df rows (%d); filtering",
                       eval_feats.shape[0], len(eval_df))
        eval_df = eval_df.iloc[eval_valid_idx].reset_index(drop=True)
    if train_feats.shape[0] != len(train_real):
        train_real = train_real.iloc[train_valid_idx].reset_index(drop=True)
    logger.info("  final aligned eval n=%d, train n=%d", len(eval_df), len(train_real))

    # ---- 6. Substrate distance per eval real ----
    logger.info("[step 6] computing substrate_distance (k=%d NN to train reals)", K_NN)
    sub_dist = compute_substrate_distance(eval_feats, train_feats, K_NN, logger)
    eval_df["substrate_distance"] = sub_dist

    # ---- 7. Build feature matrix for logistic regression ----
    # Features: 6 IQ axes (z-scored) + substrate_distance (z-scored)
    #         + chronic_indicator (binary)
    logger.info("[step 7] building feature matrices + fitting models")
    # Build raw 10-feature matrix
    X_raw = eval_df[IQ_AXES + ["substrate_distance", "is_chronic_6"]].to_numpy(dtype=float)
    # Z-score IQ and substrate (cols 0..6), leave chronic at 0/1
    cols_to_z = list(range(7))  # 6 IQ + substrate_distance
    X = X_raw.copy()
    z_mu = np.zeros(X.shape[1])
    z_sd = np.ones(X.shape[1])
    for j in cols_to_z:
        z_mu[j] = X_raw[:, j].mean()
        z_sd[j] = X_raw[:, j].std(ddof=0) or 1.0
        X[:, j] = (X_raw[:, j] - z_mu[j]) / z_sd[j]
    feature_names = IQ_AXES + ["substrate_distance", "is_chronic_6"]
    block_map = {
        "iq": list(range(0, 6)),
        "substrate": [6],
        "chronic": [7],
    }
    logger.info("  feature matrix shape: %s", X.shape)
    logger.info("  feature names: %s", feature_names)

    # Per ckpt, fit logistic regressions
    per_block_rows = []
    coef_rows = []
    summary_per_ckpt: Dict[str, Dict] = {}

    for ckpt in CKPTS:
        y = eval_df[f"is_FP_{ckpt}"].values.astype(np.int64)
        n_fp = int(y.sum())
        logger.info("\n[%s] fitting LR; n=%d, FP=%d (%.2f%%)",
                    ckpt, len(y), n_fp, 100 * y.mean())
        # Skip if too few positives
        if n_fp < 5:
            logger.warning("  too few FP; skipping")
            continue

        # Full model
        full = fit_lr_pseudoR2(X, y)
        coefs, intercept, clf = fit_lr_full(X, y, feature_names)
        for j, fn in enumerate(feature_names):
            coef_rows.append({
                "ckpt": ckpt,
                "feature": fn,
                "block": ("iq" if j < 6 else ("substrate" if j == 6 else "chronic")),
                "coef": float(coefs[j]),
                "abs_coef": float(abs(coefs[j])),
            })
        coef_rows.append({
            "ckpt": ckpt,
            "feature": "_intercept",
            "block": "_intercept",
            "coef": float(intercept),
            "abs_coef": float(abs(intercept)),
        })

        # Reduced models — drop one block at a time
        logger.info("  full model: pseudo-R²=%.4f, AUC=%.4f", full["mcfadden_r2"], full["auc"])
        block_deltas = {}
        for block_name, block_cols in block_map.items():
            keep_cols = [c for c in range(X.shape[1]) if c not in block_cols]
            X_red = X[:, keep_cols]
            red = fit_lr_pseudoR2(X_red, y)
            delta_r2 = full["mcfadden_r2"] - red["mcfadden_r2"]
            delta_auc = full["auc"] - red["auc"]
            logger.info("  drop %s: pseudo-R²=%.4f (Δ=%.4f), AUC=%.4f (Δ=%.4f)",
                        block_name, red["mcfadden_r2"], delta_r2, red["auc"], delta_auc)
            per_block_rows.append({
                "ckpt": ckpt,
                "block_dropped": block_name,
                "full_pseudo_r2": full["mcfadden_r2"],
                "reduced_pseudo_r2": red["mcfadden_r2"],
                "delta_pseudo_r2": delta_r2,
                "full_auc": full["auc"],
                "reduced_auc": red["auc"],
                "delta_auc": delta_auc,
                "n_features_full": full["k_features"],
                "n_features_reduced": red["k_features"],
                "n_obs": full["n"],
                "n_fp": n_fp,
                "fp_rate": full["p_base"],
            })
            block_deltas[block_name] = {
                "delta_pseudo_r2": delta_r2,
                "delta_auc": delta_auc,
            }
        # Also fit single-block-only models for context
        for block_name, block_cols in block_map.items():
            X_only = X[:, block_cols]
            only = fit_lr_pseudoR2(X_only, y)
            logger.info("  only %s (%d feats): pseudo-R²=%.4f, AUC=%.4f",
                        block_name, X_only.shape[1], only["mcfadden_r2"], only["auc"])
            per_block_rows.append({
                "ckpt": ckpt,
                "block_dropped": f"_only_{block_name}",
                "full_pseudo_r2": full["mcfadden_r2"],
                "reduced_pseudo_r2": only["mcfadden_r2"],
                "delta_pseudo_r2": full["mcfadden_r2"] - only["mcfadden_r2"],
                "full_auc": full["auc"],
                "reduced_auc": only["auc"],
                "delta_auc": full["auc"] - only["auc"],
                "n_features_full": full["k_features"],
                "n_features_reduced": only["k_features"],
                "n_obs": full["n"],
                "n_fp": n_fp,
                "fp_rate": full["p_base"],
            })

        summary_per_ckpt[ckpt] = {
            "n_obs": full["n"],
            "n_fp": n_fp,
            "fp_rate": full["p_base"],
            "full_pseudo_r2": full["mcfadden_r2"],
            "full_auc": full["auc"],
            "tau_5pct_dev_cal": tau_5pct[ckpt],
            "block_deltas": block_deltas,
        }

    # ---- 8. Per-block correlation matrix ----
    logger.info("[step 8] block correlation matrix (substrate vs IQ-PC1 vs chronic)")
    # IQ to scalar: PC1 of z-scored IQ block
    iq_z = X[:, block_map["iq"]]
    _, _, Vt = np.linalg.svd(iq_z, full_matrices=False)
    pc1 = iq_z @ Vt[0]  # project onto PC1
    sub_z = X[:, block_map["substrate"][0]]
    chr_v = X[:, block_map["chronic"][0]]
    block_vec_df = pd.DataFrame({
        "substrate_z": sub_z,
        "iq_pc1": pc1,
        "chronic_indicator": chr_v,
    })
    corr = block_vec_df.corr(method="pearson")
    logger.info("  Pearson correlations:\n%s", corr.to_string())

    # Also per-IQ-axis vs substrate vs chronic (more granular)
    full_corr_cols = IQ_AXES + ["substrate_distance", "is_chronic_6"]
    full_corr_mat = pd.DataFrame(
        X_raw, columns=full_corr_cols
    ).corr(method="pearson")
    logger.info("  full per-feature correlation (head):\n%s",
                full_corr_mat.round(3).to_string())

    # ---- 9. Write outputs ----
    logger.info("[step 9] writing CSVs")
    # Per-frame features + FP flags
    out_cols = ["row_ix", "suite", "frame_path", "basename", "is_chronic_6"] + IQ_AXES \
        + ["substrate_distance"] + CKPTS + [f"is_FP_{c}" for c in CKPTS]
    eval_df[out_cols].to_csv(OUT_DIR / "per_frame_features.csv", index=False)
    logger.info("  wrote per_frame_features.csv (%d rows)", len(eval_df))

    pd.DataFrame(per_block_rows).to_csv(
        OUT_DIR / "per_block_partial_r2.csv", index=False
    )
    logger.info("  wrote per_block_partial_r2.csv")

    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(OUT_DIR / "coefficient_table.csv", index=False)
    logger.info("  wrote coefficient_table.csv")

    # Block-vector correlation matrix + per-feature correlation matrix
    corr.to_csv(OUT_DIR / "block_correlation_matrix.csv")
    full_corr_mat.to_csv(OUT_DIR / "feature_correlation_matrix.csv")
    logger.info("  wrote block_correlation_matrix.csv + feature_correlation_matrix.csv")

    # Summary JSON
    summary = {
        "n_eval_real_used": int(len(eval_df)),
        "n_train_real_reference": int(len(train_real)),
        "k_nn": K_NN,
        "tau_5pct_dev_cal": tau_5pct,
        "per_ckpt": summary_per_ckpt,
        "block_correlations": corr.to_dict(),
        "per_suite_n_real": eval_df["suite"].value_counts().to_dict(),
        "per_suite_n_chronic": eval_df.groupby("suite")["is_chronic_6"].sum().to_dict(),
        "elapsed_seconds": time.time() - t0,
    }
    with open(OUT_DIR / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("  wrote _summary.json")

    logger.info("\n%s\nDONE  elapsed=%.1f s\n%s", "=" * 80, time.time() - t0, "=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
