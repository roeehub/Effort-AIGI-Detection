#!/usr/bin/env python3
"""Localize the deepfake signal in the FFT — amplitude vs phase.

Goal: decide whether FACT/APR-S-style Fourier amplitude randomization is safe.
We run logistic regression on three feature representations of `teams_fake_all_dev`
(label=1) vs `teams_real_all_dev` (label=0):

  * AMPLITUDE: log(|F|+1) -> 16 radial bands x 8 angle bins = 128 feats
  * PHASE: angle(F) -> per-band circular variance over the same 16x8 grid = 128 feats
  * PIXEL-BASELINE: pooled grayscale stats (mean/std/skew/kurt of intensity, edge density)

Plus a SHUFFLE control on the amplitude classifier.

CPU-only, n_jobs=2 max for sklearn. multiprocessing.Pool(4) for FFT extraction.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import logging
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------- Paths ----------
TRAINING_DIR = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
PROBE_DIR = TRAINING_DIR / "analysis/amp_vs_phase_probe_2026-05-06"
RAW_DIR = PROBE_DIR / "raw"
OUT_DIR = PROBE_DIR / "outputs"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

HARD_CACHE = TRAINING_DIR / "analysis/pa_pc_eval_2026-05-05/_frame_cache_hard_suites"
LOCKBOX_CACHE = TRAINING_DIR / "analysis/lockbox_tagging/_frame_cache"

FAKE_REPORT = TRAINING_DIR / "analysis/pa_pc_eval_2026-05-05/raw_reports/teams_fake_all_dev_p8a_reference_step5000_frames_report.csv"
REAL_REPORT = TRAINING_DIR / "analysis/pa_pc_eval_2026-05-05/raw_reports/teams_real_all_dev_p8a_reference_step5000_frames_report.csv"

# ---------- Constants ----------
RES = 224
N_RADIAL = 16
N_ANGLE = 8
N_FEATS = N_RADIAL * N_ANGLE  # 128
N_PER_CLASS = 2500
SEED = 42

# ---------- Logging ----------
logger = logging.getLogger("amp-vs-phase")


def setup_logging():
    log_path = OUT_DIR / "run.log"
    fh = logging.FileHandler(str(log_path), mode="w")
    sh = logging.StreamHandler(sys.stderr)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)


# ---------- Cache lookup ----------
def md5_path(cache_dir: Path, gs_uri: str) -> Path:
    blob = gs_uri[5:].split("/", 1)[1]
    name = Path(blob).name
    h = hashlib.md5(blob.encode()).hexdigest()
    return cache_dir / h[:2] / h[2:4] / name


def lookup(gs_uri: str) -> Tuple[Optional[Path], str]:
    for cache, tag in [(HARD_CACHE, "hard_cache"), (LOCKBOX_CACHE, "lockbox_cache")]:
        p = md5_path(cache, gs_uri)
        if p.exists() and p.stat().st_size > 0:
            return p, tag
    return None, "missing"


def gsutil_cp(gs_uri: str, target: Path) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return True
    try:
        r = subprocess.run(
            ["gsutil", "-q", "cp", gs_uri, str(target)],
            check=False, capture_output=True, text=True, timeout=60,
        )
        return r.returncode == 0 and target.exists() and target.stat().st_size > 0
    except Exception:
        return False


def download_missing(items: List[Tuple[str, Path]], max_workers: int = 24) -> int:
    """Download (gs_uri, target) tuples in parallel. Returns ok count."""
    n_ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(gsutil_cp, gs, p): (gs, p) for gs, p in items}
        for i, fut in enumerate(as_completed(futs)):
            try:
                ok = fut.result()
            except Exception:
                ok = False
            if ok:
                n_ok += 1
            if (i + 1) % 100 == 0 or (i + 1) == len(items):
                logger.info("    download: %d/%d (ok=%d)", i + 1, len(items), n_ok)
    return n_ok


# ---------- Sampling ----------
@dataclass
class Sample:
    label: int
    video_id: str
    method: str
    local_path: Path


def build_sample(seed: int = SEED, do_download_viso: bool = True) -> List[Sample]:
    rng = random.Random(seed)

    # Load reports
    fk = pd.read_csv(FAKE_REPORT)
    rl = pd.read_csv(REAL_REPORT)
    logger.info("source counts: fake=%d, real=%d", len(fk), len(rl))

    # Drop visomaster if downloads not desired
    if not do_download_viso:
        fk = fk[fk["method"] != "visomaster_enhanced_macro"].copy()
        logger.info("excluded visomaster_enhanced_macro: fake=%d", len(fk))

    # Resolve cached paths
    fk["_path_pair"] = fk["frame_path"].map(lookup)
    rl["_path_pair"] = rl["frame_path"].map(lookup)

    # For viso: download if missing
    if do_download_viso:
        viso_missing = fk[(fk["method"] == "visomaster_enhanced_macro")
                          & fk["_path_pair"].apply(lambda x: x[0] is None)]
        if len(viso_missing):
            logger.info("downloading %d visomaster frames", len(viso_missing))
            t0 = time.time()
            items = []
            for _, r in viso_missing.iterrows():
                gs = r["frame_path"]
                p = md5_path(HARD_CACHE, gs)
                items.append((gs, p))
            n_ok = download_missing(items, max_workers=24)
            logger.info("viso download done: ok=%d/%d in %.1fs",
                        n_ok, len(items), time.time() - t0)
            # Re-resolve
            fk["_path_pair"] = fk["frame_path"].map(lookup)

    # Drop unresolvable
    fk = fk[fk["_path_pair"].apply(lambda x: x[0] is not None)].copy()
    rl = rl[rl["_path_pair"].apply(lambda x: x[0] is not None)].copy()
    logger.info("after cache resolve: fake=%d, real=%d", len(fk), len(rl))

    fk["local_path"] = fk["_path_pair"].apply(lambda x: x[0])
    rl["local_path"] = rl["_path_pair"].apply(lambda x: x[0])

    # Per-class sampling stratified by video_id where possible
    # For fakes: sample N_PER_CLASS keeping method-balanced as much as we can
    n_fake_target = min(N_PER_CLASS, len(fk))
    n_real_target = min(N_PER_CLASS, len(rl))

    # Stratified-by-method fake sampling: proportional to method count, capped
    fk_indices = []
    methods = fk["method"].value_counts()
    proportions = (methods / methods.sum() * n_fake_target).round().astype(int)
    total_p = proportions.sum()
    if total_p < n_fake_target:
        # bump biggest method
        top = proportions.idxmax()
        proportions[top] += (n_fake_target - total_p)
    elif total_p > n_fake_target:
        top = proportions.idxmax()
        proportions[top] -= (total_p - n_fake_target)
    for m, n_take in proportions.items():
        sub = fk[fk["method"] == m]
        if n_take >= len(sub):
            fk_indices.extend(sub.index.tolist())
        else:
            fk_indices.extend(rng.sample(list(sub.index), int(n_take)))
    fk_sample = fk.loc[fk_indices].reset_index(drop=True)

    # Reals: random sample
    if n_real_target < len(rl):
        rl_sample = rl.sample(n=n_real_target, random_state=seed).reset_index(drop=True)
    else:
        rl_sample = rl.reset_index(drop=True)

    samples: List[Sample] = []
    for _, r in fk_sample.iterrows():
        samples.append(Sample(label=1, video_id=str(r["video_id"]),
                              method=str(r["method"]), local_path=Path(r["local_path"])))
    for _, r in rl_sample.iterrows():
        samples.append(Sample(label=0, video_id=str(r["video_id"]),
                              method=str(r["method"]), local_path=Path(r["local_path"])))
    rng.shuffle(samples)
    logger.info("final sample: total=%d (fake=%d, real=%d), unique videos=%d",
                len(samples),
                sum(s.label == 1 for s in samples),
                sum(s.label == 0 for s in samples),
                len(set(s.video_id for s in samples)))
    return samples


# ---------- Feature extraction ----------
def _make_band_index_map(res: int = RES, n_radial: int = N_RADIAL,
                        n_angle: int = N_ANGLE) -> Tuple[np.ndarray, np.ndarray]:
    """Return (band_radial_idx, band_angle_idx) arrays of shape (res, res), int32."""
    cy, cx = res / 2.0, res / 2.0
    yy, xx = np.mgrid[0:res, 0:res]
    dy = yy - cy
    dx = xx - cx
    rho = np.sqrt(dx * dx + dy * dy)
    rmax = np.sqrt(2) * (res / 2.0) * 0.999  # avoid OOB
    radial = np.clip((rho / rmax * n_radial).astype(np.int32), 0, n_radial - 1)
    theta = np.arctan2(dy, dx)  # [-pi, pi]
    # Use unsigned angle (0..pi) since FFT magnitude/phase are not strictly symmetric for real images
    # but we want to capture orientation in a balanced way: use full [-pi, pi] mapped to 0..n_angle
    angle = ((theta + np.pi) / (2 * np.pi) * n_angle).astype(np.int32)
    angle = np.clip(angle, 0, n_angle - 1)
    return radial, angle


# Precomputed once per process
_BAND_R = None
_BAND_A = None
_BAND_FLAT = None  # combined index in [0, n_radial*n_angle)


def _init_worker(res: int = RES, n_radial: int = N_RADIAL, n_angle: int = N_ANGLE):
    global _BAND_R, _BAND_A, _BAND_FLAT
    _BAND_R, _BAND_A = _make_band_index_map(res, n_radial, n_angle)
    _BAND_FLAT = (_BAND_R * n_angle + _BAND_A).astype(np.int32)


def _extract_features_one(args: Tuple[int, str]) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Returns (idx, amp_feats[128], phase_feats[128], pixel_feats[6])."""
    idx, path = args
    global _BAND_R, _BAND_A, _BAND_FLAT
    if _BAND_R is None:
        _init_worker()

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return idx, None, None, None
    if img.shape != (RES, RES):
        img = cv2.resize(img, (RES, RES), interpolation=cv2.INTER_LINEAR)

    img_f = img.astype(np.float32) / 255.0

    # Pixel baseline features
    flat = img_f.ravel()
    px_mean = float(flat.mean())
    px_std = float(flat.std())
    px_skew = float(scipy_stats.skew(flat))
    px_kurt = float(scipy_stats.kurtosis(flat))
    # Edge density via Laplacian variance and Sobel magnitude mean
    lap = cv2.Laplacian(img_f, cv2.CV_32F)
    lap_var = float(lap.var())
    sobel_x = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag_mean = float(np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y).mean())
    pixel_feats = np.array(
        [px_mean, px_std, px_skew, px_kurt, lap_var, sobel_mag_mean],
        dtype=np.float32,
    )

    # FFT (centered)
    F = np.fft.fftshift(np.fft.fft2(img_f))
    amp = np.log1p(np.abs(F)).astype(np.float32)  # log(|F|+1)
    phase = np.angle(F).astype(np.float32)

    # ---- Amplitude band features: mean log-amp per (radial, angle) band ----
    # Use np.bincount for speed
    flat_idx = _BAND_FLAT.ravel()
    amp_flat = amp.ravel()
    sum_amp = np.bincount(flat_idx, weights=amp_flat, minlength=N_FEATS)
    counts = np.bincount(flat_idx, minlength=N_FEATS).astype(np.float32)
    counts[counts == 0] = 1.0
    amp_feats = (sum_amp / counts).astype(np.float32)

    # ---- Phase band features: circular variance per band ----
    # circ_var = 1 - |mean(e^{i*theta})|
    cos_t = np.cos(phase, dtype=np.float32).ravel()
    sin_t = np.sin(phase, dtype=np.float32).ravel()
    sum_cos = np.bincount(flat_idx, weights=cos_t, minlength=N_FEATS)
    sum_sin = np.bincount(flat_idx, weights=sin_t, minlength=N_FEATS)
    mean_cos = sum_cos / counts
    mean_sin = sum_sin / counts
    R = np.sqrt(mean_cos * mean_cos + mean_sin * mean_sin)
    phase_feats = (1.0 - R).astype(np.float32)

    return idx, amp_feats, phase_feats, pixel_feats


def extract_all_features(samples: List[Sample], n_workers: int = 4
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Returns X_amp (n,128), X_phase (n,128), X_pixel (n,6), y (n,), groups (n,), failed_idx list."""
    n = len(samples)
    X_amp = np.zeros((n, N_FEATS), dtype=np.float32)
    X_phase = np.zeros((n, N_FEATS), dtype=np.float32)
    X_pixel = np.zeros((n, 6), dtype=np.float32)
    y = np.array([s.label for s in samples], dtype=np.int32)
    # Map video_id -> integer group id
    vid_map: Dict[str, int] = {}
    groups = np.zeros(n, dtype=np.int32)
    for i, s in enumerate(samples):
        if s.video_id not in vid_map:
            vid_map[s.video_id] = len(vid_map)
        groups[i] = vid_map[s.video_id]
    logger.info("groups: %d unique video ids", len(vid_map))

    failed: List[int] = []
    args_list = [(i, str(s.local_path)) for i, s in enumerate(samples)]
    t0 = time.time()
    with mp.Pool(n_workers, initializer=_init_worker) as pool:
        for j, (idx, a, p, x) in enumerate(pool.imap_unordered(
                _extract_features_one, args_list, chunksize=32)):
            if a is None:
                failed.append(idx)
                continue
            X_amp[idx] = a
            X_phase[idx] = p
            X_pixel[idx] = x
            if (j + 1) % 500 == 0 or (j + 1) == n:
                elapsed = time.time() - t0
                rate = (j + 1) / max(elapsed, 1e-6)
                logger.info("    fft extract %d/%d  %.1f fps  %.1fs",
                            j + 1, n, rate, elapsed)
    logger.info("feature extraction done: %d failed", len(failed))

    if failed:
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[failed] = False
        X_amp = X_amp[keep_mask]
        X_phase = X_phase[keep_mask]
        X_pixel = X_pixel[keep_mask]
        y = y[keep_mask]
        groups = groups[keep_mask]
    return X_amp, X_phase, X_pixel, y, groups, failed


# ---------- CV ----------
def fit_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, name: str,
           n_splits: int = 5, seed: int = SEED, shuffle_y: bool = False
           ) -> Tuple[np.ndarray, Dict[str, float], LogisticRegression]:
    """Run StratifiedGroupKFold (or StratifiedKFold if groups missing). Return fold AUCs, summary, and a final-fit model on all data."""
    if shuffle_y:
        rng = np.random.default_rng(seed)
        y = y.copy()
        rng.shuffle(y)

    # Try StratifiedGroupKFold first; if any fold ends up unbalanced (single class), fall back
    use_group = True
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_aucs = []
        for fold, (tr, te) in enumerate(sgkf.split(X, y, groups)):
            if len(np.unique(y[te])) < 2:
                use_group = False
                break
        if not use_group:
            logger.warning("[%s] StratifiedGroupKFold yielded single-class folds; falling back to StratifiedKFold", name)
    except Exception as e:
        logger.warning("[%s] StratifiedGroupKFold failed (%s); falling back to StratifiedKFold", name, e)
        use_group = False

    if use_group:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)

    fold_aucs = []
    for fold, (tr, te) in enumerate(split_iter):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, class_weight="balanced",
                                      solver="liblinear", max_iter=2000,
                                      random_state=seed)),
        ])
        pipe.fit(X[tr], y[tr])
        prob = pipe.predict_proba(X[te])[:, 1]
        try:
            auc = roc_auc_score(y[te], prob)
        except ValueError:
            auc = float("nan")
        fold_aucs.append(auc)
        logger.info("    [%s] fold %d  AUC=%.4f  n_train=%d  n_test=%d  (group=%s)",
                    name, fold, auc, len(tr), len(te), use_group)
    fold_aucs = np.array(fold_aucs, dtype=np.float64)
    summary = {
        "mean": float(np.nanmean(fold_aucs)),
        "std": float(np.nanstd(fold_aucs, ddof=1)) if np.sum(~np.isnan(fold_aucs)) > 1 else 0.0,
        "n_folds": int(np.sum(~np.isnan(fold_aucs))),
        "use_group": use_group,
    }
    logger.info("[%s] AUC = %.4f +/- %.4f over %d folds", name,
                summary["mean"], summary["std"], summary["n_folds"])

    # Final fit for coefficient inspection (use all data, no holdout)
    final = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, class_weight="balanced",
                                  solver="liblinear", max_iter=2000,
                                  random_state=seed)),
    ])
    final.fit(X, y)
    return fold_aucs, summary, final


def write_top_features(model: Pipeline, n_radial: int, n_angle: int,
                       out_path: Path, k: int = 20, name: str = ""):
    coefs = model.named_steps["lr"].coef_.ravel()  # (128,)
    # band_index = radial * n_angle + angle
    band_radial = np.arange(N_FEATS) // n_angle
    band_angle = np.arange(N_FEATS) % n_angle
    order = np.argsort(-np.abs(coefs))
    rows = []
    for rank, i in enumerate(order[:k]):
        rows.append({
            "rank": rank + 1,
            "feature_idx": int(i),
            "radial_band": int(band_radial[i]),
            "angle_bin": int(band_angle[i]),
            "coef": float(coefs[i]),
            "abs_coef": float(abs(coefs[i])),
        })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info("[%s] wrote top-%d features -> %s", name, k, out_path)
    return rows


def write_aucs(records: List[Dict], path: Path):
    fieldnames = ["name"] + [f"fold_{i}_auc" for i in range(5)] + ["mean", "std", "n_folds", "use_group"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            row = {"name": r["name"]}
            for i in range(5):
                row[f"fold_{i}_auc"] = float(r["folds"][i]) if i < len(r["folds"]) else ""
            row["mean"] = r["summary"]["mean"]
            row["std"] = r["summary"]["std"]
            row["n_folds"] = r["summary"]["n_folds"]
            row["use_group"] = r["summary"]["use_group"]
            w.writerow(row)
    logger.info("wrote AUC table -> %s", path)


# ---------- Verdict ----------
def make_verdict(amp_auc: float, phase_auc: float, pixel_auc: float, shuffle_auc: float
                 ) -> Tuple[str, str]:
    if abs(shuffle_auc - 0.5) > 0.05:
        return "INVALID-LEAKAGE", (
            f"Shuffle control AUC={shuffle_auc:.3f} indicates leakage "
            f"(should be ~0.5). Verdicts on amp/phase are not trustworthy."
        )
    # Decision rules
    if amp_auc >= 0.85 and phase_auc < 0.70:
        return "POISON", (
            f"Amplitude carries the manipulation signal "
            f"(amp AUC={amp_auc:.3f}, phase AUC={phase_auc:.3f}). "
            f"FACT/APR-S amplitude randomization would destroy this signal. "
            f"Recommend band-limited or partial mixing only."
        )
    if phase_auc >= 0.85 and amp_auc < 0.70:
        return "GOLD", (
            f"Phase carries the manipulation signal "
            f"(phase AUC={phase_auc:.3f}, amp AUC={amp_auc:.3f}). "
            f"Amplitude is safe to randomize. Greenlight FACT/APR-S."
        )
    if amp_auc >= 0.85 and phase_auc >= 0.85 and abs(amp_auc - phase_auc) <= 0.05:
        return "MIXED", (
            f"Signal is distributed across amp and phase "
            f"(amp={amp_auc:.3f}, phase={phase_auc:.3f}). "
            f"FACT/APR-S gives up some signal; conservative use only "
            f"(small beta, mid-frequency band only)."
        )
    return "OTHER", (
        f"AUCs do not match any clean rule (amp={amp_auc:.3f}, phase={phase_auc:.3f}, "
        f"pixel={pixel_auc:.3f}). Manual interpretation required."
    )


# ---------- Main ----------
def main():
    setup_logging()
    t_total_0 = time.time()
    logger.info("=" * 70)
    logger.info("AMP vs PHASE PROBE — teams_fake_all_dev vs teams_real_all_dev")
    logger.info("=" * 70)

    # 1) Build sample
    samples = build_sample(seed=SEED, do_download_viso=True)

    # 2) Extract features (4 workers)
    n_workers = 4
    logger.info("extracting features with %d workers ...", n_workers)
    X_amp, X_phase, X_pixel, y, groups, failed = extract_all_features(samples, n_workers=n_workers)
    logger.info("matrices: X_amp=%s, X_phase=%s, X_pixel=%s, y=%s, groups uniq=%d",
                X_amp.shape, X_phase.shape, X_pixel.shape, y.shape, len(np.unique(groups)))
    if X_amp.shape[0] < 100:
        logger.error("too few features extracted; aborting")
        sys.exit(1)

    # 3) CV runs
    auc_records = []

    for name, X in [("amplitude", X_amp), ("phase", X_phase), ("pixel-baseline", X_pixel)]:
        logger.info("--- CV: %s ---", name)
        folds, summary, model = fit_cv(X, y, groups, name=name, n_splits=5, seed=SEED, shuffle_y=False)
        auc_records.append({"name": name, "folds": folds, "summary": summary, "model": model})

    # 4) Shuffle control (on amplitude — the most likely strong signal)
    logger.info("--- CV: amplitude-SHUFFLED (control) ---")
    folds_sh, summary_sh, _ = fit_cv(X_amp, y, groups, name="amp-shuffle",
                                     n_splits=5, seed=SEED, shuffle_y=True)
    auc_records.append({"name": "amp-shuffle", "folds": folds_sh, "summary": summary_sh, "model": None})

    # 5) Write outputs
    write_aucs(auc_records, OUT_DIR / "amp_phase_aucs.csv")

    amp_top = write_top_features(auc_records[0]["model"], N_RADIAL, N_ANGLE,
                                 OUT_DIR / "amp_top_features.csv", k=20, name="amplitude")
    phase_top = write_top_features(auc_records[1]["model"], N_RADIAL, N_ANGLE,
                                   OUT_DIR / "phase_top_features.csv", k=20, name="phase")

    # 6) Verdict + FINDINGS
    amp_auc = auc_records[0]["summary"]["mean"]
    phase_auc = auc_records[1]["summary"]["mean"]
    pixel_auc = auc_records[2]["summary"]["mean"]
    shuffle_auc = auc_records[3]["summary"]["mean"]
    verdict, rationale = make_verdict(amp_auc, phase_auc, pixel_auc, shuffle_auc)
    logger.info("VERDICT: %s", verdict)
    logger.info("rationale: %s", rationale)

    # Per-band frequency interpretation
    amp_radial_dist = {}
    for r in amp_top:
        amp_radial_dist.setdefault(r["radial_band"], 0)
        amp_radial_dist[r["radial_band"]] += 1
    phase_radial_dist = {}
    for r in phase_top:
        phase_radial_dist.setdefault(r["radial_band"], 0)
        phase_radial_dist[r["radial_band"]] += 1

    def band_label(b: int) -> str:
        # 0..N_RADIAL-1, lower = LF, higher = HF
        if b < N_RADIAL // 4:
            return "LF"
        if b < (3 * N_RADIAL) // 4:
            return "MF"
        return "HF"

    amp_class = {"LF": 0, "MF": 0, "HF": 0}
    for b, c in amp_radial_dist.items():
        amp_class[band_label(b)] += c
    phase_class = {"LF": 0, "MF": 0, "HF": 0}
    for b, c in phase_radial_dist.items():
        phase_class[band_label(b)] += c

    # FINDINGS doc
    findings_path = OUT_DIR / "FINDINGS.md"
    n_total = X_amp.shape[0]
    n_fake = int((y == 1).sum())
    n_real = int((y == 0).sum())
    n_groups = len(np.unique(groups))
    use_group = auc_records[0]["summary"]["use_group"]

    lines = []
    lines.append("# Amp vs Phase Probe — FINDINGS\n")
    lines.append(f"**Date**: 2026-05-06\n")
    lines.append(f"**Question**: Where in the FFT does the deepfake signal live "
                 f"on `teams_fake_all_dev` vs `teams_real_all_dev`?\n")
    lines.append("\n## Verdict: **" + verdict + "**\n")
    lines.append(f"\n{rationale}\n")
    lines.append("\n## Setup\n")
    lines.append(f"- Total frames scored: **{n_total}** "
                 f"({n_fake} fake, {n_real} real)\n")
    lines.append(f"- Unique videos / groups: **{n_groups}**\n")
    lines.append(f"- Resolution: 224x224 grayscale\n")
    lines.append(f"- FFT bands: {N_RADIAL} radial x {N_ANGLE} angular = "
                 f"{N_FEATS} features per representation\n")
    lines.append(f"- Classifier: LogisticRegression (liblinear, C=1.0, "
                 f"class_weight=balanced) inside StandardScaler pipeline\n")
    lines.append(f"- CV: 5-fold "
                 f"{'StratifiedGroupKFold (group=video_id)' if use_group else 'StratifiedKFold (group fold not feasible)'}\n")
    if failed:
        lines.append(f"- Frames that failed to load: {len(failed)}\n")

    lines.append("\n## AUC table\n")
    lines.append("| Representation | Mean AUC | Std | Folds | Per-fold AUCs |\n")
    lines.append("|---|---|---|---|---|\n")
    for rec in auc_records:
        per = ", ".join(f"{x:.3f}" for x in rec["folds"])
        lines.append(f"| {rec['name']} | {rec['summary']['mean']:.4f} "
                     f"| {rec['summary']['std']:.4f} "
                     f"| {rec['summary']['n_folds']} | {per} |\n")

    lines.append("\n## Per-band interpretation\n")
    lines.append(f"Radial bands are indexed 0 (DC / lowest freq) to {N_RADIAL - 1} "
                 f"(highest freq). Coarse classes:\n")
    lines.append(f"- LF = bands 0..{N_RADIAL // 4 - 1}\n")
    lines.append(f"- MF = bands {N_RADIAL // 4}..{(3 * N_RADIAL) // 4 - 1}\n")
    lines.append(f"- HF = bands {(3 * N_RADIAL) // 4}..{N_RADIAL - 1}\n")
    lines.append("\n### Amplitude — top-20 |coef| radial-band class breakdown\n")
    lines.append(f"- LF: {amp_class['LF']} of 20\n")
    lines.append(f"- MF: {amp_class['MF']} of 20\n")
    lines.append(f"- HF: {amp_class['HF']} of 20\n")
    lines.append("\nTop-5 amplitude features:\n")
    for r in amp_top[:5]:
        lines.append(f"- band r={r['radial_band']:2d}, theta={r['angle_bin']}, "
                     f"coef={r['coef']:+.4f}\n")

    lines.append("\n### Phase — top-20 |coef| radial-band class breakdown\n")
    lines.append(f"- LF: {phase_class['LF']} of 20\n")
    lines.append(f"- MF: {phase_class['MF']} of 20\n")
    lines.append(f"- HF: {phase_class['HF']} of 20\n")
    lines.append("\nTop-5 phase features:\n")
    for r in phase_top[:5]:
        lines.append(f"- band r={r['radial_band']:2d}, theta={r['angle_bin']}, "
                     f"coef={r['coef']:+.4f}\n")

    lines.append("\n## Decision rule recap\n")
    lines.append("- **POISON**: amp >= 0.85 AND phase < 0.70 -> Fourier amp aug destroys signal\n")
    lines.append("- **GOLD**: phase >= 0.85 AND amp < 0.70 -> Fourier amp aug greenlit\n")
    lines.append("- **MIXED**: both >= 0.85 within 0.05 -> conservative, mid-band only\n")

    lines.append("\n## Files\n")
    lines.append("- `outputs/amp_phase_aucs.csv` — fold AUCs and summary\n")
    lines.append("- `outputs/amp_top_features.csv` — top-20 amplitude coefficients\n")
    lines.append("- `outputs/phase_top_features.csv` — top-20 phase coefficients\n")
    lines.append("- `outputs/run.log` — full run log\n")

    findings_path.write_text("".join(lines))
    logger.info("wrote FINDINGS -> %s", findings_path)

    # Summary JSON for easy parsing
    summary_json = {
        "verdict": verdict,
        "rationale": rationale,
        "amp_auc": amp_auc,
        "phase_auc": phase_auc,
        "pixel_auc": pixel_auc,
        "shuffle_auc": shuffle_auc,
        "n_total": int(n_total),
        "n_fake": n_fake,
        "n_real": n_real,
        "n_groups": int(n_groups),
        "use_group_kfold": bool(use_group),
        "amp_radial_class_top20": amp_class,
        "phase_radial_class_top20": phase_class,
        "n_failed_frames": len(failed),
        "wallclock_seconds": time.time() - t_total_0,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary_json, indent=2))
    logger.info("DONE in %.1fs", time.time() - t_total_0)
    logger.info("=" * 70)
    return summary_json


if __name__ == "__main__":
    # cap blas threads to be polite to the parallel agents
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    main()
