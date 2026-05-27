"""D8 — head retrain with substrate-balanced real pool (2026-05-12).

Question
--------
If we retrain P8A's head with the real-pool importance-weighted to match the
eval-real distribution in CLIP-frozen feature space, does the DEV→LOCKBOX
probe transfer rise vs the unweighted baseline?

Method (per brief)
------------------
1. Feature inputs:
   - P8A frozen-encoder L11 features for 4000 dev + 839 lockbox frames
     from `analysis/_features_cache_2026-04-30/layer_validation__P8A__4000_839__layers_3_6_11.npz`.
   - CLIP-frozen (OpenCLIP B16, DataComp.XL) L11 CLS features for the same
     4839 frames (re-extracted on demand, MPS-batched, cached on disk).
2. Substrate balance in CLIP-frozen space:
   - PCA reduce to 32 dim (fit on combined dev+lockbox features for shared basis).
   - KDE on (a) dev reals (n=2000), (b) eval reals = lockbox reals (n=414).
   - For each dev real x_i: w_i = density_eval(x_i) / max(density_train(x_i), ε)
     with ε=1e-6, clipped to [0.01, 100]. Report w distribution stats.
3. Head retrain on P8A features:
   - Baseline = unweighted LR(C=1.0, max_iter=2000, solver=lbfgs) on
     (dev features, dev labels).
   - Substrate-balanced = same LR but with sample_weight = w_i for dev reals,
     w = 1.0 for dev fakes.
4. Probe transfer AUCs:
   - DEV→LOCKBOX: train on dev, score lockbox (both heads).
   - LOCKBOX→DEV: train on lockbox, score dev (one direction symmetric, since
     the brief asks both heads in step 4; for LOCKBOX→DEV we report only the
     unweighted baseline since there's no separate "training-real" pool when
     the source is lockbox).
   - Bootstrap 95% CI (100 resamples, seed 42).
5. Contract metrics (dev-trained heads only):
   - τ calibrated to 5% FPR on dev reals (per-head).
   - Lockbox FPR at that τ; dev real recall (= 1 − FPR_dev); dev fake recall;
     macro fake recall on lockbox at that τ.

Constraints
-----------
- n_jobs=1 throughout (memory feedback_sklearn_njobs.md).
- Deterministic seed 42.
- FACTS-doc rules: numbers only, no interpretation in this script's output text.
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

P8A_CACHE = (
    REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
    / "layer_validation__P8A__4000_839__layers_3_6_11.npz"
)
LOCKBOX_PARQUET = (
    REPO_ROOT / "analysis" / "lockbox_tagging" / "full_tags_2026-04-27.parquet"
)
CLIP_FROZEN_CACHE = (
    THIS_DIR / "outputs" / "clip_frozen_l11__n4839.npz"
)
SEED = 42
N_BOOTSTRAP = 100
PCA_DIM = 32
# Per brief: clip weights to [0.01, 100]. Empirical caveat documented in §6: in
# 32-d Silverman-KDE space the dev-real density dwarfs the lockbox-real density
# at every dev real, driving every w to the clip floor of 0.01. We additionally
# report a low-dim (PCA_DIM_LOW) variant and a classifier-based importance
# estimator (KLIEP-style logistic-regression density ratio) as robustness
# checks. Both are reported alongside the brief's primary estimator.
PCA_DIM_LOW = 8
BW_FACTOR = 2.0  # multiplicative factor on Silverman's rule for the low-dim KDE
W_CLIP_LO, W_CLIP_HI = 0.01, 100.0
EPS_DENS = 1e-6
FPR_TARGET = 0.05

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

logger = logging.getLogger("d8")


# ---------------------------------------------------------------------------
# 1. Load P8A cache, replay the same sampling for paths.
# ---------------------------------------------------------------------------
def replay_dev_lockbox_paths() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Replay the dev/lockbox sampling from
    scaled_layer3_validation.py with seed=42 so the resulting frame order
    matches the cached P8A L11 features in `layer_validation__P8A__4000_839...`.
    """
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


def load_p8a_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (dev_feats, dev_labels, lb_feats, lb_labels) for L11.
    dev_feats shape (4000, 768); lb_feats shape (839, 768).
    """
    blob = np.load(P8A_CACHE, allow_pickle=True)
    return (
        blob["dev_layer_11_feats"].astype(np.float32),
        blob["dev_labels"].astype(np.int64),
        blob["lb_layer_11_feats"].astype(np.float32),
        blob["lb_labels"].astype(np.int64),
    )


# ---------------------------------------------------------------------------
# 2. CLIP-frozen extraction.
# ---------------------------------------------------------------------------
def extract_clip_frozen_l11(local_paths: List[str]) -> np.ndarray:
    """Extract OpenCLIP B16 L11 CLS features for the given local image paths.

    Returns features array of shape (N, 768). Raises on any unrecoverable
    error so D8 can stop instead of synthesizing substitutes.
    """
    import torch
    import open_clip
    import cv2

    # Pick device.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("CLIP-frozen device: %s", device)

    local_pretrained = (
        REPO_ROOT / "weights" / "CLIP-ViT-B-16-DataComp.XL-s13B-b90K"
        / "open_clip_pytorch_model.bin"
    )
    if not local_pretrained.exists():
        raise FileNotFoundError(
            f"CLIP weights not found at {local_pretrained}; "
            "D8 requires this checkpoint."
        )
    logger.info("loading OpenCLIP B16 from %s", local_pretrained)
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(local_pretrained)
    )
    model = model.to(device).eval()
    visual = model.visual
    if not (hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks")):
        raise RuntimeError("cannot locate transformer.resblocks on CLIP visual")

    captured: List[np.ndarray] = []

    def hook(_module, _input, output):
        if output.dim() == 3:
            if output.shape[0] >= output.shape[1]:
                cls = output[0]
            else:
                cls = output[:, 0]
        elif output.dim() == 2:
            cls = output
        else:
            raise RuntimeError(f"unexpected resblock output: {tuple(output.shape)}")
        captured.append(cls.detach().cpu().to(torch.float32).numpy())

    handle = visual.transformer.resblocks[11].register_forward_hook(hook)
    feats_per_frame: List[Optional[np.ndarray]] = [None] * len(local_paths)
    batch_size = 32

    try:
        # Preprocess on the fly per batch (avoid OOM for 4839 frames).
        pending_imgs: List[torch.Tensor] = []
        pending_idx: List[int] = []
        n_batches = (len(local_paths) + batch_size - 1) // batch_size
        for i, p in enumerate(local_paths):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("cv2.imread None for %s", p)
                continue
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (
                img - np.array(CLIP_MEAN, dtype=np.float32)
            ) / np.array(CLIP_STD, dtype=np.float32)
            pending_imgs.append(torch.from_numpy(img.transpose(2, 0, 1)))
            pending_idx.append(i)
            if len(pending_imgs) >= batch_size:
                _run_batch(model, device, pending_imgs, pending_idx, captured, feats_per_frame)
                pending_imgs, pending_idx = [], []
                bj = i // batch_size
                if bj % 10 == 0:
                    logger.info("  batch %d / %d", bj + 1, n_batches)
        if pending_imgs:
            _run_batch(model, device, pending_imgs, pending_idx, captured, feats_per_frame)
    finally:
        handle.remove()

    n_ok = sum(1 for f in feats_per_frame if f is not None)
    logger.info("CLIP-frozen ok %d/%d frames", n_ok, len(local_paths))
    if n_ok != len(local_paths):
        # Fill missing with zeros and record indices in run.log, but require all to succeed.
        missing_idx = [i for i, f in enumerate(feats_per_frame) if f is None]
        raise RuntimeError(
            f"CLIP-frozen extraction missing {len(missing_idx)} frames: first few {missing_idx[:10]}"
        )
    return np.stack(feats_per_frame, axis=0).astype(np.float32)


def _run_batch(model, device, imgs: List, idx_list: List[int],
               captured: List[np.ndarray], dest: List[Optional[np.ndarray]]) -> None:
    import torch
    batch = torch.stack(imgs).to(device, non_blocking=True)
    captured.clear()
    with torch.inference_mode():
        _ = model.visual(batch)
    if not captured:
        raise RuntimeError(f"no capture for batch starting at idx {idx_list[0]}")
    cls_batch = captured[0]
    if cls_batch.shape[0] != len(idx_list):
        raise RuntimeError(
            f"capture shape mismatch ({cls_batch.shape[0]} vs {len(idx_list)})"
        )
    for k, ii in enumerate(idx_list):
        dest[ii] = cls_batch[k]


# ---------------------------------------------------------------------------
# 3. Importance weights via KDE in PCA-reduced CLIP-frozen space.
# ---------------------------------------------------------------------------
def compute_importance_weights_classifier(
    clip_dev_real: np.ndarray, clip_lb_real: np.ndarray, n_dev_real: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """KLIEP-style: fit a logistic regression discriminating dev-real (y=0) vs
    lockbox-real (y=1) on raw CLIP-frozen features (l2-normalized). For each
    dev-real x_i, the importance ratio p_eval(x)/p_train(x) is proportional to
    p(y=1|x) / p(y=0|x).
    """
    from sklearn.linear_model import LogisticRegression

    X = np.concatenate([clip_dev_real, clip_lb_real], axis=0)
    y = np.concatenate([np.zeros(len(clip_dev_real)), np.ones(len(clip_lb_real))]).astype(np.int64)
    # l2-normalize (cosine geometry of CLIP).
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    # Balanced class weights so the larger dev-real pool doesn't dominate.
    clf = LogisticRegression(
        C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs", class_weight="balanced",
    )
    clf.fit(X, y)
    # Probability of being eval (lockbox-real) for dev rows.
    p = clf.predict_proba(X[:n_dev_real])[:, 1]
    eps = 1e-6
    w = p / np.maximum(1.0 - p, eps)
    # Rescale so the mean weight on dev reals is 1 (purely cosmetic; LR is
    # scale-invariant on the magnitude up to a constant).
    w = w * (n_dev_real / max(w.sum(), 1e-12))
    w_clipped = np.clip(w, W_CLIP_LO, W_CLIP_HI)
    n_top1 = max(1, int(np.ceil(0.01 * n_dev_real)))
    n_top10 = max(1, int(np.ceil(0.10 * n_dev_real)))
    w_sorted = np.sort(w_clipped)[::-1]
    total_mass = float(w_sorted.sum())
    stats = {
        "estimator": "classifier_KLIEP",
        "n_dev_real": n_dev_real,
        "unclipped_min": float(np.min(w)),
        "unclipped_p5": float(np.percentile(w, 5)),
        "unclipped_median": float(np.median(w)),
        "unclipped_p95": float(np.percentile(w, 95)),
        "unclipped_max": float(np.max(w)),
        "clipped_min": float(np.min(w_clipped)),
        "clipped_p5": float(np.percentile(w_clipped, 5)),
        "clipped_median": float(np.median(w_clipped)),
        "clipped_p95": float(np.percentile(w_clipped, 95)),
        "clipped_max": float(np.max(w_clipped)),
        "n_clipped_lo": int((w <= W_CLIP_LO).sum()),
        "n_clipped_hi": int((w >= W_CLIP_HI).sum()),
        "top1pct_mass_share": float(w_sorted[:n_top1].sum() / total_mass),
        "top10pct_mass_share": float(w_sorted[:n_top10].sum() / total_mass),
        "total_clipped_mass": total_mass,
        "effective_sample_size": float((w_sorted.sum() ** 2) / (w_sorted ** 2).sum()),
        "lr_disc_train_balanced_acc": float(clf.score(X, y)),
    }
    return w_clipped, stats


def compute_importance_weights_lowdim_kde(
    clip_dev_real: np.ndarray, clip_lb_real: np.ndarray, n_dev_real: int,
    pca_dim: int = PCA_DIM_LOW, bw_factor: float = BW_FACTOR,
) -> Tuple[np.ndarray, Dict[str, float], float]:
    """KDE-based ratio in low-dim PCA space with bandwidth-inflated KDE.

    Returns (weights_clipped, stats, pca_var_explained).
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KernelDensity

    X_all = np.concatenate([clip_dev_real, clip_lb_real], axis=0)
    pca = PCA(n_components=pca_dim, random_state=SEED)
    pca.fit(X_all)
    dev_pca = pca.transform(clip_dev_real)
    lb_pca = pca.transform(clip_lb_real)

    def silverman(X: np.ndarray) -> float:
        n, d = X.shape
        sigma = float(np.median(X.std(axis=0)))
        bw = sigma * (n ** (-1.0 / (d + 4)))
        return max(bw, 1e-6)

    bw_train = silverman(dev_pca) * bw_factor
    bw_eval = silverman(lb_pca) * bw_factor

    kde_train = KernelDensity(kernel="gaussian", bandwidth=bw_train)
    kde_train.fit(dev_pca)
    kde_eval = KernelDensity(kernel="gaussian", bandwidth=bw_eval)
    kde_eval.fit(lb_pca)
    log_p_train = kde_train.score_samples(dev_pca)
    log_p_eval = kde_eval.score_samples(dev_pca)
    w = np.exp(log_p_eval - log_p_train)
    w_clipped = np.clip(w, W_CLIP_LO, W_CLIP_HI)

    n_top1 = max(1, int(np.ceil(0.01 * n_dev_real)))
    n_top10 = max(1, int(np.ceil(0.10 * n_dev_real)))
    w_sorted = np.sort(w_clipped)[::-1]
    total_mass = float(w_sorted.sum())
    stats = {
        "estimator": f"lowdim_KDE(d={pca_dim}, bw*{bw_factor})",
        "n_dev_real": n_dev_real,
        "unclipped_min": float(np.min(w)),
        "unclipped_p5": float(np.percentile(w, 5)),
        "unclipped_median": float(np.median(w)),
        "unclipped_p95": float(np.percentile(w, 95)),
        "unclipped_max": float(np.max(w)),
        "clipped_min": float(np.min(w_clipped)),
        "clipped_p5": float(np.percentile(w_clipped, 5)),
        "clipped_median": float(np.median(w_clipped)),
        "clipped_p95": float(np.percentile(w_clipped, 95)),
        "clipped_max": float(np.max(w_clipped)),
        "n_clipped_lo": int((w <= W_CLIP_LO).sum()),
        "n_clipped_hi": int((w >= W_CLIP_HI).sum()),
        "top1pct_mass_share": float(w_sorted[:n_top1].sum() / total_mass),
        "top10pct_mass_share": float(w_sorted[:n_top10].sum() / total_mass),
        "total_clipped_mass": total_mass,
        "effective_sample_size": float((w_sorted.sum() ** 2) / (w_sorted ** 2).sum()),
        "kde_bw_train": bw_train,
        "kde_bw_eval": bw_eval,
        "pca_dim": pca_dim,
        "pca_var_explained": float(pca.explained_variance_ratio_.sum()),
    }
    return w_clipped, stats, float(pca.explained_variance_ratio_.sum())


def compute_importance_weights(
    clip_dev: np.ndarray, dev_labels: np.ndarray, clip_lb: np.ndarray, lb_labels: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
    """Substrate-balance the dev-real pool to match lockbox-real distribution
    in CLIP-frozen 32-PCA space.

    Returns:
      weights : np.ndarray shape (n_dev,) — weight for each dev row (dev fakes
                receive weight 1.0; dev reals receive the importance ratio).
      stats   : dict with min/p5/median/p95/max + top-1% / top-10% mass fractions.
      pca_components : shape (PCA_DIM, 768) for record.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KernelDensity

    # PCA fit on dev+lockbox (combined for a shared basis).
    X_all = np.concatenate([clip_dev, clip_lb], axis=0)
    # Center for stability.
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    pca.fit(X_all)
    dev_pca = pca.transform(clip_dev)
    lb_pca = pca.transform(clip_lb)
    logger.info("PCA fit: explained variance sum (32-d) = %.4f", float(pca.explained_variance_ratio_.sum()))

    dev_real_mask = (dev_labels == 0)
    lb_real_mask = (lb_labels == 0)
    n_dev_real = int(dev_real_mask.sum())
    n_lb_real = int(lb_real_mask.sum())

    # KDE bandwidth: Silverman's rule for d-dim Gaussian KDE on the train side.
    def silverman(X: np.ndarray) -> float:
        n, d = X.shape
        sigma = float(np.median(X.std(axis=0)))
        bw = sigma * (n ** (-1.0 / (d + 4)))
        return max(bw, 1e-6)

    bw_train = silverman(dev_pca[dev_real_mask])
    bw_eval = silverman(lb_pca[lb_real_mask])
    logger.info("KDE bandwidth (Silverman): train=%.4g eval=%.4g", bw_train, bw_eval)

    kde_train = KernelDensity(kernel="gaussian", bandwidth=bw_train)
    kde_train.fit(dev_pca[dev_real_mask])
    kde_eval = KernelDensity(kernel="gaussian", bandwidth=bw_eval)
    kde_eval.fit(lb_pca[lb_real_mask])

    # Score all dev reals under both densities.
    log_p_train = kde_train.score_samples(dev_pca[dev_real_mask])  # (n_dev_real,)
    log_p_eval = kde_eval.score_samples(dev_pca[dev_real_mask])    # (n_dev_real,)

    # Per brief: w_i = density_eval(x_i) / max(density_train(x_i), eps), clipped.
    p_train = np.exp(log_p_train)
    p_eval = np.exp(log_p_eval)
    w_real = p_eval / np.maximum(p_train, EPS_DENS)
    w_real_clipped = np.clip(w_real, W_CLIP_LO, W_CLIP_HI)

    # Build full dev weight vector.
    weights = np.ones(len(dev_labels), dtype=np.float64)
    weights[dev_real_mask] = w_real_clipped

    # Stats.
    w_unclipped_stats = {
        "n_dev_real": n_dev_real,
        "n_lb_real": n_lb_real,
        "unclipped_min": float(np.min(w_real)),
        "unclipped_p5": float(np.percentile(w_real, 5)),
        "unclipped_median": float(np.median(w_real)),
        "unclipped_p95": float(np.percentile(w_real, 95)),
        "unclipped_max": float(np.max(w_real)),
        "clipped_min": float(np.min(w_real_clipped)),
        "clipped_p5": float(np.percentile(w_real_clipped, 5)),
        "clipped_median": float(np.median(w_real_clipped)),
        "clipped_p95": float(np.percentile(w_real_clipped, 95)),
        "clipped_max": float(np.max(w_real_clipped)),
        "n_clipped_lo": int((w_real <= W_CLIP_LO).sum()),
        "n_clipped_hi": int((w_real >= W_CLIP_HI).sum()),
        "kde_bw_train": bw_train,
        "kde_bw_eval": bw_eval,
        "pca_dim": PCA_DIM,
        "pca_var_explained": float(pca.explained_variance_ratio_.sum()),
    }
    # Mass concentration: of the post-clip weight sum, what fraction sits in
    # the top-1% / top-10% of dev reals?
    w_sorted = np.sort(w_real_clipped)[::-1]
    total_mass = float(w_sorted.sum())
    n_top1 = max(1, int(np.ceil(0.01 * n_dev_real)))
    n_top10 = max(1, int(np.ceil(0.10 * n_dev_real)))
    w_unclipped_stats["top1pct_mass_share"] = float(w_sorted[:n_top1].sum() / total_mass)
    w_unclipped_stats["top10pct_mass_share"] = float(w_sorted[:n_top10].sum() / total_mass)
    w_unclipped_stats["total_clipped_mass"] = total_mass
    w_unclipped_stats["effective_sample_size"] = float(
        (w_sorted.sum() ** 2) / (w_sorted ** 2).sum()
    )
    return weights, w_unclipped_stats, pca.components_


# ---------------------------------------------------------------------------
# 4. Probe transfer + contract metrics.
# ---------------------------------------------------------------------------
def fit_head(
    X_train: np.ndarray, y_train: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> "LogisticRegression":
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        C=1.0, max_iter=2000, n_jobs=1, solver="lbfgs"
    )
    if sample_weight is not None:
        clf.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        clf.fit(X_train, y_train)
    return clf


def score_to_prob(clf, X: np.ndarray) -> np.ndarray:
    """Use predict_proba (column 1 = P(fake)) for τ-calibration; use
    decision_function for AUC."""
    return clf.predict_proba(X)[:, 1]


def auc_with_bootstrap(
    y_true: np.ndarray, y_score: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED
) -> Tuple[float, float, float]:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    auc = float(roc_auc_score(y_true, y_score))
    boots = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if yt.min() == yt.max():
            continue
        boots.append(float(roc_auc_score(yt, ys)))
    if len(boots) < 2:
        return auc, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, float(lo), float(hi)


def calibrate_tau_fpr(
    y_real_scores: np.ndarray, fpr_target: float = FPR_TARGET
) -> float:
    """Pick τ so that fraction(real_score >= τ) ≈ fpr_target.
    Use the smallest τ at which FPR ≤ target (so the boundary is achievable);
    equivalent to the (1-fpr) quantile of real scores from above.
    """
    sorted_scores = np.sort(y_real_scores)
    # We want the threshold such that the top (fpr_target * n) reals are over τ.
    n = len(sorted_scores)
    k = int(np.ceil((1.0 - fpr_target) * n))
    if k >= n:
        return float(sorted_scores[-1] + 1e-9)
    tau = float(sorted_scores[k])
    return tau


def fpr_at_tau(scores: np.ndarray, labels: np.ndarray, tau: float) -> float:
    real = scores[labels == 0]
    if len(real) == 0:
        return float("nan")
    return float((real >= tau).mean())


def recall_at_tau(scores: np.ndarray, labels: np.ndarray, tau: float) -> float:
    fake = scores[labels == 1]
    if len(fake) == 0:
        return float("nan")
    return float((fake >= tau).mean())


# ---------------------------------------------------------------------------
# Main pipeline.
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        handlers=[
            logging.FileHandler(THIS_DIR / "_run.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    t0 = time.time()
    logger.info("D8 — head retrain (substrate-balanced) — start")

    # 1. Paths + labels.
    dev_sample, lb_df = replay_dev_lockbox_paths()
    logger.info("dev_sample: %d rows; lb_df: %d rows", len(dev_sample), len(lb_df))

    # 2. P8A cached features.
    dev_p8a, dev_labels, lb_p8a, lb_labels = load_p8a_features()
    logger.info("P8A L11: dev shape %s, lockbox shape %s", dev_p8a.shape, lb_p8a.shape)
    assert dev_p8a.shape[0] == 4000 and lb_p8a.shape[0] == 839, "cache shape mismatch"
    assert (dev_labels == (dev_sample["label"] == "fake").astype(np.int64).to_numpy()).all()
    assert (lb_labels == (lb_df["label"] == "fake").astype(np.int64).to_numpy()).all()

    # 3. CLIP-frozen features (extract or load).
    paths_dev = dev_sample["local_path"].tolist()
    paths_lb = lb_df["local_path"].tolist()
    all_paths = paths_dev + paths_lb
    if CLIP_FROZEN_CACHE.exists():
        logger.info("CLIP-frozen cache hit: %s", CLIP_FROZEN_CACHE)
        blob = np.load(CLIP_FROZEN_CACHE)
        clip_feats = blob["features"].astype(np.float32)
        if clip_feats.shape[0] != len(all_paths):
            raise RuntimeError(
                f"CLIP-frozen cache size mismatch: {clip_feats.shape[0]} vs {len(all_paths)}"
            )
    else:
        logger.info("extracting CLIP-frozen L11 for %d frames", len(all_paths))
        clip_feats = extract_clip_frozen_l11(all_paths)
        np.savez_compressed(CLIP_FROZEN_CACHE, features=clip_feats.astype(np.float32))
        logger.info("CLIP-frozen cached: %s (%s)", CLIP_FROZEN_CACHE, clip_feats.shape)
    clip_dev = clip_feats[:len(paths_dev)]
    clip_lb = clip_feats[len(paths_dev):]
    assert clip_dev.shape[0] == 4000 and clip_lb.shape[0] == 839

    # 4. Importance weights in CLIP-frozen PCA(32) space.
    # Primary estimator (per brief): PCA(32) Silverman-KDE.
    weights_primary, w_stats_primary, pca_comp = compute_importance_weights(
        clip_dev, dev_labels, clip_lb, lb_labels
    )
    logger.info("PRIMARY weights stats: %s", json.dumps(w_stats_primary, indent=2))

    # Robustness 1: low-dim (PCA=8) KDE with 2x bandwidth.
    dev_real_mask = dev_labels == 0
    lb_real_mask = lb_labels == 0
    w_lowdim_real, w_stats_lowdim, _ = compute_importance_weights_lowdim_kde(
        clip_dev[dev_real_mask], clip_lb[lb_real_mask], int(dev_real_mask.sum()),
    )
    weights_lowdim = np.ones(len(dev_labels), dtype=np.float64)
    weights_lowdim[dev_real_mask] = w_lowdim_real
    logger.info("LOWDIM-KDE weights stats: %s", json.dumps(w_stats_lowdim, indent=2))

    # Robustness 2: classifier (KLIEP-style) density ratio.
    w_clf_real, w_stats_clf = compute_importance_weights_classifier(
        clip_dev[dev_real_mask], clip_lb[lb_real_mask], int(dev_real_mask.sum()),
    )
    weights_clf = np.ones(len(dev_labels), dtype=np.float64)
    weights_clf[dev_real_mask] = w_clf_real
    logger.info("CLASSIFIER-KLIEP weights stats: %s", json.dumps(w_stats_clf, indent=2))

    # Save weight stats table.
    pd.DataFrame([w_stats_primary, w_stats_lowdim, w_stats_clf]).to_csv(
        OUTPUTS / "weight_distribution_stats.csv", index=False
    )

    # Save per-frame weights (for traceability).
    pd.DataFrame({
        "row_ix_in_dev": np.arange(len(weights_primary)),
        "label_int": dev_labels,
        "gcs_uri": dev_sample["gcs_uri"].values,
        "identity_key": dev_sample["identity_key"].values,
        "weight_primary_pca32_silverman": weights_primary,
        "weight_lowdim_pca8_silverman_bw2x": weights_lowdim,
        "weight_classifier_kliep": weights_clf,
    }).to_csv(OUTPUTS / "per_frame_weights_dev.csv", index=False)

    # 5. Fit heads on P8A L11 (dev as training pool).
    logger.info("fitting unweighted head")
    clf_unweighted = fit_head(dev_p8a, dev_labels)

    logger.info("fitting balanced-primary head (PCA32 Silverman KDE)")
    clf_balanced = fit_head(dev_p8a, dev_labels, sample_weight=weights_primary)

    logger.info("fitting balanced-lowdim head (PCA8 Silverman x2 KDE)")
    clf_lowdim = fit_head(dev_p8a, dev_labels, sample_weight=weights_lowdim)

    logger.info("fitting balanced-classifier head (KLIEP)")
    clf_clf = fit_head(dev_p8a, dev_labels, sample_weight=weights_clf)

    # 6. D3-style probe transfer AUCs.
    logger.info("computing DEV→LOCKBOX AUCs")
    score_lb_unw = score_to_prob(clf_unweighted, lb_p8a)
    score_lb_bal_primary = score_to_prob(clf_balanced, lb_p8a)
    score_lb_bal_lowdim = score_to_prob(clf_lowdim, lb_p8a)
    score_lb_bal_clf = score_to_prob(clf_clf, lb_p8a)
    auc_unw, lo_unw, hi_unw = auc_with_bootstrap(lb_labels, score_lb_unw)
    auc_bal, lo_bal, hi_bal = auc_with_bootstrap(lb_labels, score_lb_bal_primary)
    auc_low, lo_low, hi_low = auc_with_bootstrap(lb_labels, score_lb_bal_lowdim)
    auc_clf, lo_clf, hi_clf = auc_with_bootstrap(lb_labels, score_lb_bal_clf)
    logger.info("DEV→LOCKBOX unweighted    AUC = %.4f [%.4f, %.4f]", auc_unw, lo_unw, hi_unw)
    logger.info("DEV→LOCKBOX bal-primary   AUC = %.4f [%.4f, %.4f]", auc_bal, lo_bal, hi_bal)
    logger.info("DEV→LOCKBOX bal-lowdim    AUC = %.4f [%.4f, %.4f]", auc_low, lo_low, hi_low)
    logger.info("DEV→LOCKBOX bal-classifier AUC = %.4f [%.4f, %.4f]", auc_clf, lo_clf, hi_clf)

    # LOCKBOX→DEV reference (unweighted only — no separate "training-real" pool
    # to importance-weight when source is lockbox).
    logger.info("computing LOCKBOX→DEV AUC (unweighted reference)")
    clf_lb_to_dev = fit_head(lb_p8a, lb_labels)
    score_dev_lb = score_to_prob(clf_lb_to_dev, dev_p8a)
    auc_l2d, lo_l2d, hi_l2d = auc_with_bootstrap(dev_labels, score_dev_lb)
    logger.info("LOCKBOX→DEV unweighted    AUC = %.4f [%.4f, %.4f]", auc_l2d, lo_l2d, hi_l2d)

    # 7. In-sample CV reference (dev, unweighted vs weighted).
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    def cv_auc(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]) -> float:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = np.zeros(len(y), dtype=np.float64)
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=1, solver="lbfgs")
            if w is not None:
                clf.fit(X[tr], y[tr], sample_weight=w[tr])
            else:
                clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        return float(roc_auc_score(y, oof))

    cv_unw = cv_auc(dev_p8a, dev_labels, None)
    cv_bal = cv_auc(dev_p8a, dev_labels, weights_primary)
    cv_low = cv_auc(dev_p8a, dev_labels, weights_lowdim)
    cv_clf = cv_auc(dev_p8a, dev_labels, weights_clf)
    cv_lb_unw = cv_auc(lb_p8a, lb_labels, None)
    logger.info(
        "dev CV: unweighted=%.4f bal-primary=%.4f bal-lowdim=%.4f bal-classifier=%.4f",
        cv_unw, cv_bal, cv_low, cv_clf,
    )
    logger.info("lockbox CV unweighted=%.4f", cv_lb_unw)

    aucs_df = pd.DataFrame([
        {"head": "P8A_dev_unweighted",          "direction": "DEV→LOCKBOX", "n_train": int(dev_labels.size),
         "n_test": int(lb_labels.size), "auc": auc_unw, "ci_low": lo_unw, "ci_high": hi_unw},
        {"head": "P8A_dev_bal_primary_pca32",   "direction": "DEV→LOCKBOX", "n_train": int(dev_labels.size),
         "n_test": int(lb_labels.size), "auc": auc_bal, "ci_low": lo_bal, "ci_high": hi_bal},
        {"head": "P8A_dev_bal_lowdim_pca8",     "direction": "DEV→LOCKBOX", "n_train": int(dev_labels.size),
         "n_test": int(lb_labels.size), "auc": auc_low, "ci_low": lo_low, "ci_high": hi_low},
        {"head": "P8A_dev_bal_classifier_KLIEP","direction": "DEV→LOCKBOX", "n_train": int(dev_labels.size),
         "n_test": int(lb_labels.size), "auc": auc_clf, "ci_low": lo_clf, "ci_high": hi_clf},
        {"head": "P8A_lb_unweighted",           "direction": "LOCKBOX→DEV", "n_train": int(lb_labels.size),
         "n_test": int(dev_labels.size), "auc": auc_l2d, "ci_low": lo_l2d, "ci_high": hi_l2d},
        {"head": "P8A_dev_unweighted_CV5",          "direction": "DEV→DEV_CV", "n_train": int(dev_labels.size),
         "n_test": int(dev_labels.size), "auc": cv_unw, "ci_low": float("nan"), "ci_high": float("nan")},
        {"head": "P8A_dev_bal_primary_CV5",         "direction": "DEV→DEV_CV", "n_train": int(dev_labels.size),
         "n_test": int(dev_labels.size), "auc": cv_bal, "ci_low": float("nan"), "ci_high": float("nan")},
        {"head": "P8A_dev_bal_lowdim_CV5",          "direction": "DEV→DEV_CV", "n_train": int(dev_labels.size),
         "n_test": int(dev_labels.size), "auc": cv_low, "ci_low": float("nan"), "ci_high": float("nan")},
        {"head": "P8A_dev_bal_classifier_CV5",      "direction": "DEV→DEV_CV", "n_train": int(dev_labels.size),
         "n_test": int(dev_labels.size), "auc": cv_clf, "ci_low": float("nan"), "ci_high": float("nan")},
        {"head": "P8A_lb_unweighted_CV5",           "direction": "LB→LB_CV",   "n_train": int(lb_labels.size),
         "n_test": int(lb_labels.size), "auc": cv_lb_unw, "ci_low": float("nan"), "ci_high": float("nan")},
    ])
    aucs_df.to_csv(OUTPUTS / "head_transfer_aucs.csv", index=False)

    # 8. Contract metrics at FPR=5% τ (dev-trained heads).
    logger.info("contract metrics at FPR=%.2f%% τ (dev-calibrated)", FPR_TARGET * 100)
    rows = []
    for head_name, clf, scores_dev, scores_lb in [
        ("P8A_dev_unweighted", clf_unweighted,
         score_to_prob(clf_unweighted, dev_p8a), score_lb_unw),
        ("P8A_dev_bal_primary_pca32", clf_balanced,
         score_to_prob(clf_balanced, dev_p8a), score_lb_bal_primary),
        ("P8A_dev_bal_lowdim_pca8", clf_lowdim,
         score_to_prob(clf_lowdim, dev_p8a), score_lb_bal_lowdim),
        ("P8A_dev_bal_classifier_KLIEP", clf_clf,
         score_to_prob(clf_clf, dev_p8a), score_lb_bal_clf),
    ]:
        scores_dev_real = scores_dev[dev_labels == 0]
        tau = calibrate_tau_fpr(scores_dev_real, FPR_TARGET)
        # Achieved FPR on dev (target is 5% but discretization may differ).
        dev_fpr = fpr_at_tau(scores_dev, dev_labels, tau)
        dev_real_recall = 1.0 - dev_fpr
        dev_fake_recall = recall_at_tau(scores_dev, dev_labels, tau)
        lb_fpr = fpr_at_tau(scores_lb, lb_labels, tau)
        lb_real_recall = 1.0 - lb_fpr
        lb_fake_recall = recall_at_tau(scores_lb, lb_labels, tau)
        rows.append({
            "head": head_name,
            "tau_dev_5fpr": tau,
            "dev_fpr_at_tau": dev_fpr,
            "dev_real_recall_at_tau": dev_real_recall,
            "dev_fake_recall_at_tau": dev_fake_recall,
            "lockbox_fpr_at_tau": lb_fpr,
            "lockbox_real_recall_at_tau": lb_real_recall,
            "lockbox_fake_recall_at_tau": lb_fake_recall,
        })
        logger.info(
            "  head=%s τ=%.4f dev_fpr=%.4f dev_fake_recall=%.4f lb_fpr=%.4f lb_fake_recall=%.4f",
            head_name, tau, dev_fpr, dev_fake_recall, lb_fpr, lb_fake_recall,
        )
    pd.DataFrame(rows).to_csv(OUTPUTS / "contract_metrics_fpr5pct.csv", index=False)

    # 9. Save head coefficients (for inspection / posterior diagnosis).
    head_coef = pd.DataFrame({
        "feature_idx": np.arange(dev_p8a.shape[1]),
        "coef_unweighted": clf_unweighted.coef_[0],
        "coef_bal_primary_pca32": clf_balanced.coef_[0],
        "coef_bal_lowdim_pca8": clf_lowdim.coef_[0],
        "coef_bal_classifier_kliep": clf_clf.coef_[0],
    })
    head_coef.to_csv(OUTPUTS / "head_coefficients.csv", index=False)

    # 10. Save a summary JSON for the FACTS doc.
    summary = {
        "n_dev": int(dev_labels.size),
        "n_dev_real": int((dev_labels == 0).sum()),
        "n_dev_fake": int((dev_labels == 1).sum()),
        "n_lockbox": int(lb_labels.size),
        "n_lockbox_real": int((lb_labels == 0).sum()),
        "n_lockbox_fake": int((lb_labels == 1).sum()),
        "weight_stats_primary": w_stats_primary,
        "weight_stats_lowdim": w_stats_lowdim,
        "weight_stats_classifier": w_stats_clf,
        "auc_dev_to_lockbox": {
            "unweighted": {"auc": float(auc_unw), "ci": [float(lo_unw), float(hi_unw)]},
            "bal_primary_pca32": {"auc": float(auc_bal), "ci": [float(lo_bal), float(hi_bal)]},
            "bal_lowdim_pca8": {"auc": float(auc_low), "ci": [float(lo_low), float(hi_low)]},
            "bal_classifier_kliep": {"auc": float(auc_clf), "ci": [float(lo_clf), float(hi_clf)]},
        },
        "auc_lockbox_to_dev_unweighted": float(auc_l2d),
        "auc_lockbox_to_dev_unweighted_ci": [float(lo_l2d), float(hi_l2d)],
        "cv5_dev_unweighted": float(cv_unw),
        "cv5_dev_bal_primary": float(cv_bal),
        "cv5_dev_bal_lowdim": float(cv_low),
        "cv5_dev_bal_classifier": float(cv_clf),
        "cv5_lockbox_unweighted": float(cv_lb_unw),
        "contract_rows": rows,
        "elapsed_sec": float(time.time() - t0),
    }
    with open(OUTPUTS / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("D8 complete in %.1f sec", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
