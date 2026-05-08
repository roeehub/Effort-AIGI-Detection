#!/usr/bin/env python3
"""Build a unified, cross-pool image-quality (IQ) data atlas.

Driver for the user request 2026-05-08: produce side-by-side IQ-property
distributions for ALL data pools (training, dev, lockbox, HDTF, canary), so the
user can decide:

  1. whether to filter "ultra-bad" frames out of TRAINING,
  2. whether to standardize on F4-style filtering for EVAL/lockbox readouts,
  3. where to set the deployment-side IQ-gate threshold.

Method:

  - Sample N frames per pool (default 500), download in parallel via
    `gsutil -m`, then decode + measure features in a bounded multiprocessing
    Pool (8 workers; never n_jobs=-1 — see project memory).
  - Feature set is the canonical seven from
    `analysis/score_distribution_2026-05-02/cross_suite_attribute_extraction.py`
    (`per_frame_attrs`) with `min_dim`, `max_dim`, `aspect_ratio`,
    `color_b_dev` added — matches the chronic6_fingerprint and dor_drift
    feature panels.
  - Idempotent: each pool writes a parquet to `_cache/<pool>.parquet`. Re-runs
    skip pools that are already cached.

Outputs:

  - `outputs/per_frame.parquet`     — concatenated long table, all pools
  - `outputs/per_pool_summary.csv`   — one row per pool, summary stats
  - `outputs/cross_pool_compare.csv` — pivoted: rows=pool, columns=metric x p05/p50/p95
  - `figs/histogram_<metric>_all_pools.png`            — overlaid histograms
  - `figs/pool_<pool>_quad.png`                         — 4-panel per pool
  - `figs/train_vs_eval_vs_lockbox_<metric>.png`        — group overlay

This is a CPU-only diagnostic. No GPU.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")
log = logging.getLogger("iq_atlas")

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
THIS = ROOT / "analysis" / "iq_data_atlas_2026-05-08"
OUT = THIS / "outputs"
FIG = THIS / "figs"
CACHE_DL = THIS / "_cache"  # per-pool parquet caches + downloaded frames
DL_ROOT = Path("/tmp/iq_atlas_cache")  # temp frame downloads

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
CACHE_DL.mkdir(parents=True, exist_ok=True)
DL_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_N_PER_POOL = 500
SEED = 5505
N_WORKERS = 8

# ───────────────────────────────────────────────────────────────────────────
# Feature extraction (single source of truth)
# ───────────────────────────────────────────────────────────────────────────


def _luminance(arr: np.ndarray) -> np.ndarray:
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)


def per_frame_attrs(p: Path) -> Optional[Dict[str, float]]:
    """Compute IQ feature panel for one image. Returns None on failure.

    Formulas match `analysis/dor_drift_mechanism_2026-05-06/run_analysis.py`
    exactly so the production-reference rows fold in directly without unit
    mismatch:

    - `lap_var`: cv2.Laplacian on grayscale, variance
    - `luma_mean` / `luma_std`: HSV V channel mean / std
    - `edge_mag`: Sobel ksize=3 magnitude mean
    - `saturation_mean`: HSV S channel mean
    - `color_b_dev`: `mean(|LAB.B - 128|)` (deviation from neutral, NOT RGB-B dispersion)
    - `color_a_dev`: `mean(|LAB.A - 128|)`
    - `skin_frac`: YCbCr skin range fraction (kept for cross_suite compatibility)

    `min_dim`, `max_dim`, `aspect_ratio` are added.
    """
    try:
        import cv2
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap_var = float(lap.var())

        v = hsv[..., 2]
        luma_mean = float(v.mean())
        luma_std = float(v.std())

        sat_mean = float(hsv[..., 1].mean())
        contrast_l = float(lab[..., 0].std())

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = float(np.sqrt(gx * gx + gy * gy).mean())

        a_dev = float(np.abs(lab[..., 1] - 128).mean())
        b_dev = float(np.abs(lab[..., 2] - 128).mean())

        # Skin fraction (RGB-derived, kept for cross_suite back-compat)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        R, G, B = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cr = (R - Y) * 0.713 + 128.0
        Cb = (B - Y) * 0.564 + 128.0
        skin = (Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127)

        return {
            "h": int(h), "w": int(w),
            "min_dim": int(min(h, w)),
            "max_dim": int(max(h, w)),
            "aspect_ratio": float(max(h, w) / max(1, min(h, w))),
            "luma_mean": luma_mean,
            "luma_std": luma_std,
            "lap_var": lap_var,
            "edge_mag": edge_mag,
            "saturation_mean": sat_mean,
            "contrast_l": contrast_l,
            "color_a_dev": a_dev,
            "color_b_dev": b_dev,
            "skin_frac": float(skin.mean()),
            "bytes": int(p.stat().st_size),
        }
    except Exception:
        return None


def _worker(args):
    """Module-level worker, picklable for mp.Pool."""
    local_path, frame_path = args
    rec = per_frame_attrs(Path(local_path))
    if rec is None:
        return None
    rec["frame_path"] = frame_path
    return rec


# ───────────────────────────────────────────────────────────────────────────
# gsutil parallel download
# ───────────────────────────────────────────────────────────────────────────


def _local_name_for(uri: str) -> str:
    """Build a collision-free local basename for a URI.

    df40 + HDTF use repeating `frame_0000.png` / `000.png` filenames across
    many video subdirs. Hash the URI tail to avoid collisions.
    """
    import hashlib
    bn = uri.split("/")[-1]
    # Keep extension; prefix with hash of URI for uniqueness
    ext = ""
    for cand in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        if bn.lower().endswith(cand):
            ext = cand
            break
    h = hashlib.md5(uri.encode()).hexdigest()[:12]
    # Keep partial human-readable name for debugging
    safe = bn.replace("/", "_")[:80]
    return f"{h}__{safe}" + ("" if safe.lower().endswith(ext) else ext)


def gsutil_parallel_download(uris: Sequence[str], dest_dir: Path,
                             max_workers: int = 24) -> List[Tuple[str, Path]]:
    """Download URIs to dest_dir/<unique-name>. Returns [(uri, local_path)] only
    for files that ended up present and non-empty.

    Strategy: split URIs by basename (`gsutil -m cp -I` puts everything in one
    dir, which collides on df40 / HDTF where many video subdirs share
    `frame_0000.png` / `000.png`). For the bulk, fall back to per-URI parallel
    `gsutil cp` via ThreadPoolExecutor (slower but safe).

    For plain teams_faces sources (where every frame already has a unique
    crypto-hash suffix in the basename), use the much faster
    `gsutil -m cp -I` bulk approach.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Detect if URIs are basename-unique (heuristic: count unique basenames vs total)
    seen_bn = set()
    dup = False
    for uri in uris:
        bn = uri.split("/")[-1]
        if bn in seen_bn:
            dup = True
            break
        seen_bn.add(bn)

    have: List[Tuple[str, Path]] = []
    todo: List[Tuple[str, Path]] = []
    for uri in uris:
        local = dest_dir / _local_name_for(uri)
        if local.exists() and local.stat().st_size > 0:
            have.append((uri, local))
        else:
            todo.append((uri, local))

    # Per-URI parallel download — needed when basenames collide. Faster path
    # would be `gsutil -m cp -I` to a single dir, but that breaks on df40.
    def _one(pair: Tuple[str, Path]) -> Optional[Tuple[str, Path]]:
        uri, local = pair
        try:
            r = subprocess.run(["gsutil", "-q", "cp", uri, str(local)],
                               capture_output=True, text=True, timeout=60)
            if r.returncode != 0 or not local.exists() or local.stat().st_size == 0:
                return None
            return (uri, local)
        except Exception:
            return None

    results: List[Tuple[str, Path]] = list(have)
    if todo:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for r in ex.map(_one, todo):
                if r is not None:
                    results.append(r)
    return results


# ───────────────────────────────────────────────────────────────────────────
# Pool resolution: build a flat list of (pool_name, role, [URIs])
# ───────────────────────────────────────────────────────────────────────────


def _videos_to_uris(videos: List[dict], filt) -> List[str]:
    out: List[str] = []
    for v in videos:
        if not filt(v):
            continue
        for fp in v.get("frame_paths", []):
            if isinstance(fp, str):
                out.append(fp)
    return out


def _slice_filter(slice_name: str, split: Optional[str] = None, label: Optional[str] = None):
    def _f(v):
        if split is not None and v.get("split") != split:
            return False
        if label is not None and v.get("label") != label:
            return False
        slices = v.get("slices") or []
        if slice_name not in slices and v.get("method") != slice_name and v.get("lane") != slice_name:
            return False
        return True
    return _f


def collect_eval_pools() -> List[Tuple[str, str, List[str]]]:
    """Returns [(pool_name, role, [uris])] for the eval / dev / lockbox pools
    drawn from the three main manifests.

    role ∈ {train_real, train_fake, dev_real, dev_fake, lockbox_real, lockbox_fake,
              hdtf_real, hdtf_fake, hdtf_clean_fake, hdtf_teams_fake, canary_*,
              prod_ref_*}
    """
    pools: List[Tuple[str, str, List[str]]] = []

    # --- Teams target-domain manifest (with dor) ---
    teams_path = ROOT / "arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json"
    with open(teams_path) as f:
        tm = json.load(f)
    videos = tm["videos"]

    pools.append(("teams_real_all_dev", "dev_real",
                  _videos_to_uris(videos, _slice_filter("teams_real_all", "dev"))))
    pools.append(("teams_real_all_lockbox", "lockbox_real",
                  _videos_to_uris(videos, _slice_filter("teams_real_all", "lockbox"))))
    pools.append(("teams_real_lighting_extreme_dev", "dev_real",
                  _videos_to_uris(videos, _slice_filter("teams_real_lighting_extreme", "dev"))))
    pools.append(("teams_real_poor_quality_dev", "dev_real",
                  _videos_to_uris(videos, _slice_filter("teams_real_poor_quality", "dev"))))
    pools.append(("teams_fake_all_dev", "dev_fake",
                  _videos_to_uris(videos, _slice_filter("teams_fake_all", "dev"))))
    pools.append(("teams_fake_all_lockbox", "lockbox_fake",
                  _videos_to_uris(videos, _slice_filter("teams_fake_all", "lockbox"))))
    pools.append(("deeplive_enhanced_dev", "dev_fake",
                  _videos_to_uris(videos, _slice_filter("deeplive_enhanced", "dev"))))
    pools.append(("visomaster_enhanced_macro_dev", "dev_fake",
                  _videos_to_uris(videos, _slice_filter("visomaster_enhanced_macro", "dev"))))
    # dor_shkedi specifically (chronic FP target)
    pools.append(("teams_real_dor_dev", "dev_real",
                  _videos_to_uris(videos, lambda v: v.get("split") == "dev"
                                  and v.get("label") == "real"
                                  and "dor_shkedi" in str(v.get("identity_key", "")).lower())))

    # --- Proper-visomaster (HDTF) ---
    hdtf_path = ROOT / "arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json"
    with open(hdtf_path) as f:
        hm = json.load(f)
    hvideos = hm["videos"]
    # split real_clean, real_teams, fake_clean, fake_teams; both dev and lockbox
    pools.append(("hdtf_real_clean_dev", "hdtf_real",
                  _videos_to_uris(hvideos, _slice_filter("proper_real_clean", "dev"))))
    pools.append(("hdtf_real_clean_lockbox", "hdtf_real",
                  _videos_to_uris(hvideos, _slice_filter("proper_real_clean", "lockbox"))))
    pools.append(("hdtf_real_teams_dev", "hdtf_real",
                  _videos_to_uris(hvideos, _slice_filter("proper_real_teams", "dev"))))
    pools.append(("hdtf_real_teams_lockbox", "hdtf_real",
                  _videos_to_uris(hvideos, _slice_filter("proper_real_teams", "lockbox"))))
    pools.append(("hdtf_fake_clean_dev", "hdtf_fake",
                  _videos_to_uris(hvideos, _slice_filter("proper_fake_clean_all", "dev"))))
    pools.append(("hdtf_fake_clean_lockbox", "hdtf_fake",
                  _videos_to_uris(hvideos, _slice_filter("proper_fake_clean_all", "lockbox"))))
    pools.append(("hdtf_fake_teams_dev", "hdtf_fake",
                  _videos_to_uris(hvideos, _slice_filter("proper_fake_teams_all", "dev"))))
    pools.append(("hdtf_fake_teams_lockbox", "hdtf_fake",
                  _videos_to_uris(hvideos, _slice_filter("proper_fake_teams_all", "lockbox"))))

    # --- Visomaster enhanced v2 (training-pool fake, large) ---
    v2_path = ROOT / "arena/manifests/visomaster_enhanced_v2_manifest_2026-04-13.json"
    with open(v2_path) as f:
        v2 = json.load(f)
    v2v = v2["videos"]
    pools.append(("visomaster_enhanced_v2_all", "train_fake",
                  _videos_to_uris(v2v, lambda v: True)))

    return pools


def list_gcs_prefix(bucket: str, prefix: str, n_target: int, exts: Sequence[str]) -> List[str]:
    """Use `gsutil ls -r` to enumerate up to ~10x n_target candidates from a prefix,
    returns full URIs filtered to extensions.
    """
    uri = f"gs://{bucket}/{prefix.rstrip('/')}/**"
    try:
        r = subprocess.run(["gsutil", "ls", uri], capture_output=True, text=True, timeout=180)
    except Exception:
        return []
    if r.returncode != 0:
        return []
    out: List[str] = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line.startswith("gs://"):
            continue
        if not any(line.lower().endswith(e) for e in exts):
            continue
        out.append(line)
        if len(out) >= n_target * 30:  # cap enumeration
            break
    return out


def list_gcs_top_dirs(bucket: str, prefix: str, max_dirs: int = 200) -> List[str]:
    uri = f"gs://{bucket}/{prefix.rstrip('/')}/"
    try:
        r = subprocess.run(["gsutil", "ls", uri], capture_output=True, text=True, timeout=60)
    except Exception:
        return []
    if r.returncode != 0:
        return []
    out: List[str] = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.endswith("/") and line.startswith("gs://"):
            out.append(line)
        if len(out) >= max_dirs:
            break
    return out


def collect_training_pools(n_per_pool: int) -> List[Tuple[str, str, List[str]]]:
    """Sample-on-the-fly training pools by listing GCS prefixes. Capped lists
    so this never blows up enumeration cost.
    """
    pools: List[Tuple[str, str, List[str]]] = []

    rng = random.Random(SEED)

    # 1. Teams training bucket (real / fake from the production-style cropped pool)
    log.info("[collect] teams_real (training source)...")
    uris = list_gcs_prefix("teams-faces-data-test-2914-fake-4420-real-feb-28", "real",
                            n_per_pool, [".jpg", ".jpeg", ".png"])
    rng.shuffle(uris)
    pools.append(("train_teams_real_pool", "train_real", uris[: n_per_pool * 4]))

    log.info("[collect] teams_fake (training source)...")
    uris = list_gcs_prefix("teams-faces-data-test-2914-fake-4420-real-feb-28", "fake",
                            n_per_pool, [".jpg", ".jpeg", ".png"])
    rng.shuffle(uris)
    pools.append(("train_teams_fake_pool", "train_fake", uris[: n_per_pool * 4]))

    # 2. df40 (training-pool fake; multiple methods)
    log.info("[collect] df40 fake methods...")
    df40_methods = ["simswap", "facedancer", "blendface", "e4s", "inswap", "uniface"]
    for meth in df40_methods:
        # df40 has nested dirs: fake/<method>/<videoid>/<frame>.png
        # Enumerate top-N video dirs, then sample frames within
        vid_dirs = list_gcs_top_dirs("df40-frames-recropped-rfa85", f"fake/{meth}", max_dirs=80)
        rng.shuffle(vid_dirs)
        sampled: List[str] = []
        for vd in vid_dirs[:30]:
            try:
                r = subprocess.run(["gsutil", "ls", vd], capture_output=True, text=True, timeout=30)
                if r.returncode != 0:
                    continue
                frames = [l.strip() for l in r.stdout.splitlines() if l.strip().startswith("gs://")
                          and l.strip().lower().endswith((".png", ".jpg"))]
                rng.shuffle(frames)
                sampled.extend(frames[:25])
                if len(sampled) >= n_per_pool * 3:
                    break
            except Exception:
                continue
        pools.append((f"train_df40_{meth}", "train_fake", sampled))

    # 3. df40 real (Celeb-real, FF++, YouTube-real)
    log.info("[collect] df40 real (Celeb-real, FF++, YouTube-real)...")
    for src in ["Celeb-real", "FaceForensics++", "YouTube-real"]:
        vid_dirs = list_gcs_top_dirs("df40-frames-recropped-rfa85", f"real/{src}", max_dirs=80)
        rng.shuffle(vid_dirs)
        sampled = []
        for vd in vid_dirs[:30]:
            try:
                r = subprocess.run(["gsutil", "ls", vd], capture_output=True, text=True, timeout=30)
                if r.returncode != 0:
                    continue
                frames = [l.strip() for l in r.stdout.splitlines() if l.strip().startswith("gs://")
                          and l.strip().lower().endswith((".png", ".jpg"))]
                rng.shuffle(frames)
                sampled.extend(frames[:25])
                if len(sampled) >= n_per_pool * 3:
                    break
            except Exception:
                continue
        # df40 nested: real/<src>/<videoid>/<frame>.png — also possible flat
        if not sampled:
            # fallback: directly list the prefix
            uris = list_gcs_prefix("df40-frames-recropped-rfa85", f"real/{src}",
                                    n_per_pool, [".png", ".jpg"])
            rng.shuffle(uris)
            sampled = uris[: n_per_pool * 3]
        pools.append((f"train_df40_real_{src.lower().replace('+', '').replace('-', '_')}",
                      "train_real", sampled))

    return pools


def collect_canary_pools() -> List[Tuple[str, str, List[str]]]:
    """Reuse the verified canary CSV directly — no resample needed."""
    csv = ROOT / "analysis/p2_eval_2026-05-08/d1_d4_cpu/outputs/canary_resolution_per_frame.csv"
    if not csv.exists():
        log.warning("canary CSV missing: %s", csv)
        return []
    df = pd.read_csv(csv)
    # Build pseudo-pools from cohort + label
    pools: List[Tuple[str, str, List[str]]] = []
    # We don't have URI re-list here (csv only has frame_idx + cohort + dims + lap_var);
    # so we DON'T treat canary as a sample pool; it gets folded in from the metadata
    # CSV directly during merge. Marker pool:
    return []


def collect_production_reference_pools() -> List[Tuple[str, str, List[str]]]:
    """Production reference: pull from the dor_drift_mechanism per_frame_features
    CSV, which already has min_dim/sharpness_lap/luma/color_b_dev computed for
    real-world dor frames. This is the closest thing to a production reference
    we have without raw production-frame logs.

    Returns empty URI list — the merge step pulls these features directly from
    that CSV.
    """
    return []


# ───────────────────────────────────────────────────────────────────────────
# Per-pool sampling + measurement (the workhorse)
# ───────────────────────────────────────────────────────────────────────────


def measure_pool(pool_name: str, role: str, uris: Sequence[str], n_target: int) -> Optional[pd.DataFrame]:
    cache_path = CACHE_DL / f"{pool_name}.parquet"
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            log.info("[%s] cached parquet (%d rows) — skipping", pool_name, len(df))
            return df
        except Exception:
            log.warning("[%s] cache parquet unreadable, redoing", pool_name)

    if not uris:
        log.warning("[%s] no URIs available, skipping", pool_name)
        return None

    rng = random.Random(SEED + hash(pool_name) % 1000003)
    uris = list(uris)
    rng.shuffle(uris)
    # Oversample by 1.5x to absorb download / decode losses
    sample = uris[: int(n_target * 1.5)]

    pool_dl_dir = DL_ROOT / pool_name
    pool_dl_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log.info("[%s] downloading %d URIs ...", pool_name, len(sample))
    pairs = gsutil_parallel_download(sample, pool_dl_dir, max_workers=24)
    log.info("[%s]   downloaded %d/%d in %.1fs", pool_name, len(pairs), len(sample), time.time() - t0)

    if not pairs:
        log.warning("[%s] no successful downloads", pool_name)
        return None

    # Trim to n_target
    if len(pairs) > n_target:
        pairs = pairs[:n_target]

    # Measure with mp.Pool(8)
    work = [(str(local), uri) for (uri, local) in pairs]
    t0 = time.time()
    log.info("[%s] decoding + measuring %d frames ...", pool_name, len(work))
    rows: List[Optional[Dict[str, float]]] = []
    try:
        with mp.Pool(processes=N_WORKERS) as pool:
            for r in pool.imap_unordered(_worker, work, chunksize=8):
                rows.append(r)
    except Exception as e:
        log.warning("[%s] mp.Pool error %s, falling back to serial", pool_name, e)
        for w in work:
            rows.append(_worker(w))
    rows = [r for r in rows if r is not None]
    log.info("[%s]   measured %d/%d in %.1fs", pool_name, len(rows), len(work), time.time() - t0)

    if not rows:
        log.warning("[%s] no successful measurements", pool_name)
        return None

    df = pd.DataFrame(rows)
    df["pool"] = pool_name
    df["role"] = role
    df.to_parquet(cache_path, index=False)
    log.info("[%s] wrote %s (%d rows)", pool_name, cache_path, len(df))

    # Free disk
    try:
        shutil.rmtree(pool_dl_dir, ignore_errors=True)
    except Exception:
        pass
    return df


# ───────────────────────────────────────────────────────────────────────────
# Folding in pre-computed sources (canary, prod-ref, existing cross_suite)
# ───────────────────────────────────────────────────────────────────────────


def fold_canary(per_frame: List[pd.DataFrame]):
    """The canary CSV from `verify_canary_resolution.py` only has
    h/w/min_dim/max_dim/lap_var/bytes — not the full feature panel. We can
    still use it for the resolution + sharpness comparison; we mark missing
    features as NaN.
    """
    csv = ROOT / "analysis/p2_eval_2026-05-08/d1_d4_cpu/outputs/canary_resolution_per_frame.csv"
    if not csv.exists():
        log.warning("canary CSV missing")
        return
    df = pd.read_csv(csv)
    df = df.rename(columns={"label": "_label_canary"})
    df["aspect_ratio"] = df["max_dim"] / df["min_dim"].clip(lower=1)
    # Build pool name from cohort + chronic membership
    chronic_ids = {"PC_Generator__s22", "PC_Generator__s45", "Q__s6", "Roy_D",
                   "bla_bla_chow", "bla_bla_chow__s2"}
    df["pool"] = np.where(df["base_identity"].isin(chronic_ids),
                           "canary_chronic_real",
                           np.where(df["_label_canary"] == 1, "canary_fake", "canary_other_real"))
    df["role"] = np.where(df["pool"] == "canary_chronic_real", "canary_chronic_real",
                           np.where(df["pool"] == "canary_fake", "canary_fake", "canary_other_real"))
    df["frame_path"] = "canary_cache/" + df["frame_idx"].astype(str)
    keep = ["pool", "role", "h", "w", "min_dim", "max_dim", "aspect_ratio",
            "lap_var", "bytes", "frame_path"]
    per_frame.append(df[keep])
    log.info("[canary] folded %d rows from canary CSV", len(df))


def fold_production_reference(per_frame: List[pd.DataFrame]):
    """Fold in the dor_drift per_frame_features as a production-reference proxy.

    Sessions used:
      - dor_evening / dor_morning           — local (laptop) recordings of dor (real)
      - team_sanity_may5 / team_may5__Dor   — real teams capture of dor
    """
    csv = ROOT / "analysis/dor_drift_mechanism_2026-05-06/outputs/per_frame_features.csv"
    if not csv.exists():
        log.warning("dor_drift CSV missing")
        return
    df = pd.read_csv(csv)
    sess_to_pool = {
        "dor_evening / dor_evening": ("prod_ref_dor_evening_local", "prod_ref_real"),
        "dor_morning / dor_morning": ("prod_ref_dor_morning_local", "prod_ref_real"),
        "team_sanity_may5 / team_may5__Dor": ("prod_ref_dor_may5_teams", "prod_ref_real"),
    }
    rows = []
    for sess, (pool, role) in sess_to_pool.items():
        sub = df[df["session"] == sess].copy()
        if sub.empty:
            continue
        sub["pool"] = pool
        sub["role"] = role
        sub = sub.rename(columns={
            "sharpness_lap": "lap_var",
            "luma_mean": "luma_mean",
            "luma_std": "luma_std",
            "edge_mag": "edge_mag",
            "color_b_dev": "color_b_dev",
            "sat_mean": "saturation_mean",
        })
        sub["aspect_ratio"] = (sub["height"].astype(float) / sub["width"].astype(float).clip(lower=1)
                               .where(sub["height"] >= sub["width"],
                                      sub["width"].astype(float) / sub["height"].astype(float).clip(lower=1)))
        # Properly compute aspect_ratio as max/min:
        sub["max_dim"] = np.maximum(sub["width"], sub["height"])
        sub["aspect_ratio"] = sub["max_dim"].astype(float) / sub["min_dim"].astype(float).clip(lower=1)
        sub["h"] = sub["height"].astype(int)
        sub["w"] = sub["width"].astype(int)
        sub["bytes"] = np.nan  # not available
        sub["skin_frac"] = np.nan
        keep = ["pool", "role", "h", "w", "min_dim", "max_dim", "aspect_ratio",
                "luma_mean", "luma_std", "lap_var", "edge_mag", "saturation_mean",
                "color_b_dev", "skin_frac", "bytes", "frame_path"]
        sub = sub[[c for c in keep if c in sub.columns]]
        rows.append(sub)
        log.info("[prod_ref] session=%s n=%d", sess, len(sub))
    if rows:
        per_frame.extend(rows)


# ───────────────────────────────────────────────────────────────────────────
# Summary + figures
# ───────────────────────────────────────────────────────────────────────────


METRICS = ["min_dim", "lap_var", "luma_mean", "color_b_dev", "edge_mag", "aspect_ratio", "bytes"]


def per_pool_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pool, g in df.groupby("pool"):
        role = g["role"].iloc[0]
        row = {"pool": pool, "role": role, "n": len(g)}
        for m in METRICS:
            if m not in g.columns:
                row[f"{m}_p05"] = np.nan
                row[f"{m}_p50"] = np.nan
                row[f"{m}_p95"] = np.nan
                row[f"{m}_mean"] = np.nan
                continue
            s = g[m].dropna()
            if s.empty:
                row[f"{m}_p05"] = np.nan
                row[f"{m}_p50"] = np.nan
                row[f"{m}_p95"] = np.nan
                row[f"{m}_mean"] = np.nan
                continue
            row[f"{m}_p05"] = float(s.quantile(0.05))
            row[f"{m}_p50"] = float(s.quantile(0.50))
            row[f"{m}_p95"] = float(s.quantile(0.95))
            row[f"{m}_mean"] = float(s.mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out.sort_values(["role", "pool"]).reset_index(drop=True)
    return out


def cross_pool_compare(df: pd.DataFrame) -> pd.DataFrame:
    """Pivoted: rows=pool, columns=metric x p05/p50/p95."""
    rows = []
    for pool, g in df.groupby("pool"):
        row = {"pool": pool, "role": g["role"].iloc[0], "n": len(g)}
        for m in METRICS:
            if m not in g.columns:
                continue
            s = g[m].dropna()
            if s.empty:
                continue
            row[f"{m}__p05"] = float(s.quantile(0.05))
            row[f"{m}__p50"] = float(s.quantile(0.50))
            row[f"{m}__p95"] = float(s.quantile(0.95))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["role", "pool"])


def make_figures(df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pools_sorted = sorted(df["pool"].unique())
    role_order = ["train_real", "train_fake", "dev_real", "dev_fake",
                  "lockbox_real", "lockbox_fake", "hdtf_real", "hdtf_fake",
                  "canary_chronic_real", "canary_other_real", "canary_fake",
                  "prod_ref_real"]

    # 1. Histograms: one plot per metric, overlay all pools (subsample if too many)
    for m in METRICS:
        if m not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(11, 6))
        for pool in pools_sorted:
            g = df[df["pool"] == pool][m].dropna()
            if len(g) < 5:
                continue
            ax.hist(g, bins=40, alpha=0.35, label=f"{pool} (n={len(g)})", density=True)
        ax.set_xlabel(m)
        ax.set_ylabel("density")
        ax.set_title(f"{m} — all pools (density)")
        ax.legend(fontsize=6, ncol=2, loc="best")
        plt.tight_layout()
        plt.savefig(FIG / f"histogram_{m}_all_pools.png", dpi=110)
        plt.close()

    # 2. Per-pool 4-panel quad
    for pool in pools_sorted:
        g = df[df["pool"] == pool]
        if g.empty:
            continue
        fig, axs = plt.subplots(2, 2, figsize=(10, 7))
        for ax, met in zip(axs.flat, ["min_dim", "lap_var", "luma_mean", "color_b_dev"]):
            if met not in g.columns:
                ax.text(0.5, 0.5, f"{met}: N/A", transform=ax.transAxes, ha="center")
                continue
            s = g[met].dropna()
            if s.empty:
                ax.text(0.5, 0.5, f"{met}: empty", transform=ax.transAxes, ha="center")
                continue
            ax.hist(s, bins=30, alpha=0.7)
            ax.set_title(f"{met}  p05={s.quantile(0.05):.1f}  p50={s.quantile(0.50):.1f}  p95={s.quantile(0.95):.1f}")
        fig.suptitle(f"{pool}  (n={len(g)})")
        plt.tight_layout()
        plt.savefig(FIG / f"pool_{pool}_quad.png", dpi=110)
        plt.close()

    # 3. Train vs Eval vs Lockbox grouped overlays (the headline)
    grp_map = {
        "train": ["train_real", "train_fake"],
        "dev":    ["dev_real", "dev_fake"],
        "lockbox": ["lockbox_real", "lockbox_fake"],
        "hdtf":   ["hdtf_real", "hdtf_fake"],
        "canary": ["canary_chronic_real", "canary_other_real", "canary_fake"],
        "prod_ref": ["prod_ref_real"],
    }
    for m in ["min_dim", "lap_var", "color_b_dev"]:
        if m not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(11, 6))
        for grp_name, roles in grp_map.items():
            sub = df[df["role"].isin(roles)][m].dropna()
            if len(sub) < 10:
                continue
            ax.hist(sub, bins=40, alpha=0.35, label=f"{grp_name} (n={len(sub)})", density=True)
        ax.set_xlabel(m)
        ax.set_ylabel("density")
        ax.set_title(f"{m} — train vs eval vs lockbox vs hdtf vs canary vs prod_ref")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIG / f"train_vs_eval_vs_lockbox_{m}.png", dpi=110)
        plt.close()


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N_PER_POOL,
                        help="frames sampled per pool (default 500)")
    parser.add_argument("--pools-only", type=str, default=None,
                        help="comma-separated pool names to run (debug)")
    parser.add_argument("--skip-training", action="store_true",
                        help="skip training pools (use only manifest-defined pools)")
    args = parser.parse_args()

    log.info("Building IQ atlas: n_per_pool=%d, workers=%d", args.n, N_WORKERS)

    pools: List[Tuple[str, str, List[str]]] = []
    pools.extend(collect_eval_pools())
    if not args.skip_training:
        log.info("Collecting training pools (involves gsutil ls)...")
        try:
            pools.extend(collect_training_pools(args.n))
        except Exception as e:
            log.warning("training-pool collection failed: %s — continuing without training pools", e)

    log.info("Total pools queued: %d", len(pools))
    for pn, role, uris in pools:
        log.info("  %s [%s]  %d candidate URIs", pn, role, len(uris))

    if args.pools_only:
        wanted = set([s.strip() for s in args.pools_only.split(",")])
        pools = [(pn, r, u) for (pn, r, u) in pools if pn in wanted]
        log.info("Filtered to %d pools: %s", len(pools), [p[0] for p in pools])

    per_frame: List[pd.DataFrame] = []
    failed_pools: List[str] = []
    t_global = time.time()
    for pn, role, uris in pools:
        try:
            df_pool = measure_pool(pn, role, uris, args.n)
            if df_pool is not None and not df_pool.empty:
                per_frame.append(df_pool)
            else:
                failed_pools.append(pn)
        except Exception as e:
            log.warning("[%s] failed: %s", pn, e)
            failed_pools.append(pn)

    # Fold in pre-computed pools (canary, production reference)
    fold_canary(per_frame)
    fold_production_reference(per_frame)

    if not per_frame:
        log.error("No pools produced data — aborting")
        sys.exit(1)

    df_all = pd.concat(per_frame, ignore_index=True)
    df_all.to_parquet(OUT / "per_frame.parquet", index=False)
    log.info("Wrote per_frame.parquet (%d rows)", len(df_all))

    summary = per_pool_summary(df_all)
    summary.to_csv(OUT / "per_pool_summary.csv", index=False)
    log.info("Wrote per_pool_summary.csv (%d pools)", len(summary))

    cross = cross_pool_compare(df_all)
    cross.to_csv(OUT / "cross_pool_compare.csv", index=False)
    log.info("Wrote cross_pool_compare.csv")

    log.info("Building figures ...")
    make_figures(df_all)

    elapsed = time.time() - t_global
    log.info("Total runtime: %.1f minutes", elapsed / 60)
    log.info("Failed pools: %s", failed_pools)
    log.info("Done.")


if __name__ == "__main__":
    main()
