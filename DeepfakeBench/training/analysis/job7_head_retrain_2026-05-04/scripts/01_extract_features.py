#!/usr/bin/env python3
"""Job 7 — Stage 1: extract P8A frozen 512-d post-projection features for the
suites listed in the task. Uses MPS (Apple Silicon) when available; falls back
to CPU. Reuses local frame cache where present and reuses the existing
clip_vs_p8a_viso_2026-05-03 cached features for the (550 viso + 550 reals)
sub-frame they cover.

Per-suite output: analysis/job7_head_retrain_2026-05-04/frozen_features/<suite>_p8a_features.npz
with arrays: features (N, 512), labels (N,), frame_paths (N,), identities (N,),
local_paths (N,), and meta (json) — also computes IQ features per frame
inline so we can reuse them for H1/H2/H3 in Stage 2 (luma_p10/p90/mean,
laplacian_var, sobel_edge_mean, saturation_mean, skin_frac).

Suites to extract (per task):
    teams_real_all_dev (4564), visomaster_enhanced_macro_dev (550),
    deeplive_enhanced_dev (545), teams_fake_all_dev (3039),
    teams_real_all_lockbox (1418), teams_fake_all_lockbox (425).
v2 held-out frames (visomaster_enhanced_v2 manifest, 2073) are extracted
into a separate npz for v2 evaluation.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HERE = REPO_ROOT / "analysis" / "job7_head_retrain_2026-05-04"
FRAME_CACHE = HERE / "_frame_cache"
SHARED_CACHE = REPO_ROOT / "analysis" / "clip_vs_p8a_viso_2026-05-03" / "_frame_cache"
FROZEN = HERE / "frozen_features"
LOG_PATH = HERE / "run_extract.log"

P8A_CKPT = (
    REPO_ROOT
    / "analysis"
    / "_features_cache_2026-04-30"
    / "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth"
)
DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "config" / "detector" / "effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config" / "train_config.yaml"

# CLIP normalisation (P8A is OpenCLIP visual stem so same stats).
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

REPORTS_DIR = REPO_ROOT / "analysis" / "cpu_followups_2026-05-04" / "raw_reports"
SUITE_REPORTS: Dict[str, Path] = {
    "teams_real_all_dev": REPORTS_DIR / "teams_real_all_dev_p8a_reference_step5000_frames_report.csv",
    "visomaster_enhanced_macro_dev": REPORTS_DIR / "visomaster_enhanced_macro_dev_p8a_reference_step5000_frames_report.csv",
    "deeplive_enhanced_dev": REPORTS_DIR / "deeplive_enhanced_dev_p8a_reference_step5000_frames_report.csv",
    "teams_fake_all_dev": REPORTS_DIR / "teams_fake_all_dev_p8a_reference_step5000_frames_report.csv",
    "teams_real_all_lockbox": REPORTS_DIR / "teams_real_all_lockbox_p8a_reference_step5000_frames_report.csv",
    "teams_fake_all_lockbox": REPORTS_DIR / "teams_fake_all_lockbox_p8a_reference_step5000_frames_report.csv",
}

V2_MANIFEST = REPO_ROOT / "arena" / "manifests" / "visomaster_enhanced_v2_manifest_2026-04-13.json"

logger = logging.getLogger("job7-extract")


# -----------------------------------------------------------------------------
# Cache + download helpers (re-use shared cache).
# -----------------------------------------------------------------------------
def _local_name_for(uri: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in uri[len("gs://"):])[-200:]


def cached_path(uri: str) -> Path:
    # Try shared cache (1100 frames already there) first, else local cache.
    name = _local_name_for(uri)
    shared = SHARED_CACHE / name
    if shared.exists():
        return shared
    return FRAME_CACHE / name


def gsutil_cp_one(uri: str, dest: Path) -> Tuple[str, bool]:
    if dest.exists():
        return uri, True
    try:
        res = subprocess.run(
            ["gsutil", "-q", "cp", uri, str(dest)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        return uri, res.returncode == 0
    except Exception:
        return uri, False


def download_frames(uris: List[str], max_workers: int = 16) -> List[Path]:
    FRAME_CACHE.mkdir(parents=True, exist_ok=True)
    locals_: List[Path] = [cached_path(u) for u in uris]
    todo = [(u, p) for u, p in zip(uris, locals_) if not p.exists()]
    logger.info(
        "  cache: %d total, %d cached, %d to download",
        len(uris), len(uris) - len(todo), len(todo),
    )
    if not todo:
        return locals_
    n_done = 0
    n_fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(gsutil_cp_one, u, p): (u, p) for (u, p) in todo}
        for fut in as_completed(futs):
            uri, ok = fut.result()
            n_done += 1
            if not ok:
                n_fail += 1
            if n_done % 200 == 0 or n_done == len(todo):
                logger.info(
                    "    download progress %d/%d (%d failed) elapsed %.1fs",
                    n_done, len(todo), n_fail, time.time() - t0,
                )
    return [cached_path(u) for u in uris]


# -----------------------------------------------------------------------------
# Image preprocess + IQ feature compute (single forward pass through the file).
# -----------------------------------------------------------------------------
def load_and_features(local_path: Path, resolution: int = 224) -> Optional[Tuple[torch.Tensor, np.ndarray]]:
    img = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    # IQ on full-resolution crop (before model resize) — this is what the
    # score_distribution_2026-05-02/crop_attributes.csv was computed on.
    iq = compute_iq_features(img)
    img224 = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img224 = cv2.cvtColor(img224, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img224 = (img224 - CLIP_MEAN) / CLIP_STD
    return torch.from_numpy(img224.transpose(2, 0, 1)), iq


def compute_iq_features(img_bgr: np.ndarray) -> np.ndarray:
    """Return (luma_mean, luma_p10, luma_p90, laplacian_var, sobel_edge_mean,
    saturation_mean, skin_frac).
    Mirrors score_distribution_2026-05-02 output column semantics.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    luma_mean = float(gray.mean())
    luma_p10 = float(np.percentile(gray, 10))
    luma_p90 = float(np.percentile(gray, 90))
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    laplacian_var = float(lap.var())
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_edge_mean = float(np.sqrt(sx ** 2 + sy ** 2).mean())
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturation_mean = float(hsv[..., 1].mean())
    # skin_frac: HSV-based heuristic (matches typical OpenCV skin masks; close
    # in spirit to score_distribution_2026-05-02 ranges).
    H, S, V = cv2.split(hsv)
    skin_mask = ((H >= 0) & (H <= 25) & (S >= 30) & (S <= 200) & (V >= 60) & (V <= 255))
    skin_frac = float(skin_mask.mean())
    return np.array([luma_mean, luma_p10, luma_p90, laplacian_var,
                     sobel_edge_mean, saturation_mean, skin_frac], dtype=np.float32)


IQ_COLS = ["luma_mean", "luma_p10", "luma_p90", "laplacian_var",
           "sobel_edge_mean", "saturation_mean", "skin_frac"]


# -----------------------------------------------------------------------------
# P8A model loader.
# -----------------------------------------------------------------------------
def load_p8a_model(device: torch.device):
    from detectors import DETECTOR
    with open(DEFAULT_DETECTOR_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    with open(DEFAULT_TRAIN_CONFIG, "r") as f:
        train_cfg = yaml.safe_load(f)
    cfg.update(train_cfg)
    ckpt = torch.load(str(P8A_CKPT), map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model_config = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
    for k, v in model_config.items():
        if k != "current_arcface_s":
            cfg[k] = v
    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)
    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])
    clean_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    logger.info("  P8A load: missing=%d unexpected=%d", len(missing), len(unexpected))
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Identity extraction from frame_path / video_id.
# -----------------------------------------------------------------------------
CHRONIC_PATTERNS = {
    "bla_bla_chow__s2": re.compile(r"bla_bla_chow__?s2(?!\d)", re.IGNORECASE),
    "bla_bla_chow":     re.compile(r"bla_bla_chow", re.IGNORECASE),
    "PC_Generator__s22": re.compile(r"PC_Generator__?s22(?!\d)", re.IGNORECASE),
    "PC_Generator__s45": re.compile(r"PC_Generator__?s45(?!\d)", re.IGNORECASE),
    "Roy_D":             re.compile(r"Roy_D|roy_d", re.IGNORECASE),
    "Q__s6":             re.compile(r"Q__?s6(?!\d)", re.IGNORECASE),
}


def identity_for(video_id: str, frame_path: str) -> str:
    s = f"{video_id}|{frame_path}"
    for ident, pat in CHRONIC_PATTERNS.items():
        if pat.search(s):
            return ident
    # default identity = first 2 underscore-separated tokens of video_id
    parts = str(video_id).split("__")
    return "__".join(parts[:2]) if len(parts) >= 2 else parts[0]


# -----------------------------------------------------------------------------
# Main per-suite extraction.
# -----------------------------------------------------------------------------
def extract_for_suite(model: torch.nn.Module, device: torch.device, suite: str,
                       df: pd.DataFrame, batch_size: int = 32) -> Dict[str, np.ndarray]:
    paths_uri = df["frame_path"].tolist()
    locals_ = download_frames(paths_uri)
    df = df.copy()
    df["local_path"] = [str(p) if p.exists() else "" for p in locals_]
    df["identity"] = [identity_for(v, p) for v, p in zip(df["video_id"], df["frame_path"])]
    df_valid = df[df["local_path"] != ""].reset_index(drop=True)
    logger.info("  %s: %d/%d frames available locally", suite, len(df_valid), len(df))

    feats: List[np.ndarray] = []
    iqs:   List[np.ndarray] = []
    valid_idx: List[int] = []
    pending: List[Tuple[int, torch.Tensor, np.ndarray]] = []
    for i, (lp, fp) in enumerate(zip(df_valid["local_path"].tolist(),
                                      df_valid["frame_path"].tolist())):
        out = load_and_features(Path(lp))
        if out is None:
            continue
        t, iq = out
        pending.append((i, t, iq))
    logger.info("  P8A forward: %d frames", len(pending))

    t_start = time.time()
    for j in range(0, len(pending), batch_size):
        chunk = pending[j: j + batch_size]
        idxs = [c[0] for c in chunk]
        x = torch.stack([c[1] for c in chunk]).to(device)
        iq_arr = np.stack([c[2] for c in chunk], axis=0)
        with torch.inference_mode():
            out = model({"image": x}, inference=True)
            f = out["feat"].detach().to("cpu").to(torch.float32).numpy()
        feats.append(f)
        iqs.append(iq_arr)
        valid_idx.extend(idxs)
        if (j // batch_size) % 20 == 0:
            elapsed = time.time() - t_start
            done = j + len(chunk)
            rate = done / max(1e-6, elapsed)
            logger.info("    %s batch %d / %d (%.1f frames/s)",
                        suite, j // batch_size + 1,
                        (len(pending) + batch_size - 1) // batch_size, rate)
    if not feats:
        raise RuntimeError(f"No features extracted for {suite}")
    X = np.concatenate(feats, axis=0)
    IQ = np.concatenate(iqs, axis=0)
    df_kept = df_valid.iloc[valid_idx].reset_index(drop=True)
    return {
        "features": X.astype(np.float32),
        "iq_features": IQ.astype(np.float32),
        "labels": df_kept["label"].astype(np.int32).to_numpy(),
        "frame_paths": df_kept["frame_path"].astype(str).to_numpy(),
        "video_ids": df_kept["video_id"].astype(str).to_numpy(),
        "identities": df_kept["identity"].astype(str).to_numpy(),
        "local_paths": df_kept["local_path"].astype(str).to_numpy(),
        "family_keys": df_kept.get("family_key", pd.Series([""] * len(df_kept))).astype(str).to_numpy(),
        "scores_p8a_existing": df_kept["frame_prob"].astype(np.float32).to_numpy(),
    }


def maybe_load_existing_p8a_cache() -> Dict[str, np.ndarray]:
    """Returns dict frame_path -> 512-d feature for the 1100 frames already
    extracted in clip_vs_p8a_viso_2026-05-03/outputs/p8a__features.npz."""
    npz = REPO_ROOT / "analysis" / "clip_vs_p8a_viso_2026-05-03" / "outputs" / "p8a__features.npz"
    if not npz.exists():
        return {}
    z = np.load(npz, allow_pickle=True)
    out = {}
    for fp, feat in zip(z["frame_path"], z["features"]):
        out[str(fp)] = np.asarray(feat, dtype=np.float32)
    logger.info("Loaded %d cached P8A features from clip_vs_p8a_viso_2026-05-03", len(out))
    return out


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    FROZEN.mkdir(parents=True, exist_ok=True)
    FRAME_CACHE.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(LOG_PATH); sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s %(levelname)s :: %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.handlers = [fh, sh]; logger.setLevel(logging.INFO)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logger.info("device=%s", device)

    cached_p8a = maybe_load_existing_p8a_cache()

    # Order suites: small ones first to fail fast, big ones (teams_fake_all_dev,
    # teams_real_all_dev) last so a partial run still gives us 4/6.
    # NOTE 2026-05-04: per task priority (teams_real_all_dev mandatory, then v2,
    # then teams_fake_all_dev), v2 should run before teams_fake_all_dev so we
    # have held-out family-level eval even if teams_fake_all_dev is cut short.
    suite_order = [
        "visomaster_enhanced_macro_dev",   # 550
        "deeplive_enhanced_dev",            # 545
        "teams_fake_all_lockbox",           # 425
        "teams_real_all_lockbox",           # 1418
        "teams_real_all_dev",               # 4564
        "teams_fake_all_dev",               # 3039 (de-prioritise per task; v2 first)
    ]

    model = load_p8a_model(device)

    for suite in suite_order:
        out_path = FROZEN / f"{suite}_p8a_features.npz"
        if out_path.exists():
            logger.info("[%s] features already cached at %s — skipping", suite, out_path)
            continue
        report = SUITE_REPORTS[suite]
        if not report.exists():
            logger.warning("[%s] report missing %s", suite, report)
            continue
        logger.info("=" * 80)
        logger.info("Suite: %s", suite)
        df = pd.read_csv(report, low_memory=False)
        logger.info("  %d rows, label counts=%s", len(df), df["label"].value_counts().to_dict())

        # Reuse already-cached features where the frame_path is in cached_p8a:
        # skip the model forward in those cases.
        df["from_cache"] = df["frame_path"].isin(cached_p8a)
        n_cache_hits = int(df["from_cache"].sum())
        logger.info("  cache hits (P8A feats already extracted): %d", n_cache_hits)

        df_to_extract = df[~df["from_cache"]].reset_index(drop=True)
        if len(df_to_extract) > 0:
            t0 = time.time()
            res = extract_for_suite(model, device, suite, df_to_extract, batch_size=32)
            logger.info("  forward elapsed: %.1fs", time.time() - t0)
        else:
            res = None

        # Combine cached + freshly-extracted features.
        df_full = df.copy()
        df_full["identity"] = [identity_for(v, p) for v, p in zip(df_full["video_id"], df_full["frame_path"])]
        # Cache hits: pull features by frame_path; we still need IQ (requires
        # local image read) — so download those frames too.
        if n_cache_hits > 0:
            cache_uris = df_full[df_full["from_cache"]]["frame_path"].tolist()
            cache_locals = download_frames(cache_uris)
            df_cache = df_full[df_full["from_cache"]].reset_index(drop=True)
            df_cache["local_path"] = [str(p) if p.exists() else "" for p in cache_locals]
            df_cache = df_cache[df_cache["local_path"] != ""].reset_index(drop=True)
            X_cache = np.stack([cached_p8a[fp] for fp in df_cache["frame_path"]], axis=0)
            iq_cache = []
            for lp in df_cache["local_path"]:
                img = cv2.imread(lp, cv2.IMREAD_COLOR)
                iq_cache.append(compute_iq_features(img) if img is not None else np.zeros(7, dtype=np.float32))
            iq_cache = np.stack(iq_cache, axis=0)
        else:
            X_cache = np.zeros((0, 512), dtype=np.float32)
            iq_cache = np.zeros((0, 7), dtype=np.float32)
            df_cache = df_full.iloc[:0].assign(local_path="").reset_index(drop=True)

        # Concatenate.
        if res is not None:
            X = np.concatenate([X_cache, res["features"]], axis=0).astype(np.float32)
            IQ = np.concatenate([iq_cache, res["iq_features"]], axis=0).astype(np.float32)
            labels = np.concatenate([
                df_cache["label"].astype(np.int32).to_numpy(),
                res["labels"],
            ], axis=0)
            frame_paths = np.concatenate([
                df_cache["frame_path"].astype(str).to_numpy(),
                res["frame_paths"],
            ], axis=0)
            video_ids = np.concatenate([
                df_cache["video_id"].astype(str).to_numpy(),
                res["video_ids"],
            ], axis=0)
            identities = np.concatenate([
                df_cache["identity"].astype(str).to_numpy(),
                res["identities"],
            ], axis=0)
            local_paths = np.concatenate([
                df_cache["local_path"].astype(str).to_numpy(),
                res["local_paths"],
            ], axis=0)
            family_keys = np.concatenate([
                df_cache.get("family_key", pd.Series([""] * len(df_cache))).astype(str).to_numpy(),
                res["family_keys"],
            ], axis=0)
            scores_p8a = np.concatenate([
                df_cache["frame_prob"].astype(np.float32).to_numpy(),
                res["scores_p8a_existing"],
            ], axis=0)
        else:
            X = X_cache; IQ = iq_cache
            labels = df_cache["label"].astype(np.int32).to_numpy()
            frame_paths = df_cache["frame_path"].astype(str).to_numpy()
            video_ids = df_cache["video_id"].astype(str).to_numpy()
            identities = df_cache["identity"].astype(str).to_numpy()
            local_paths = df_cache["local_path"].astype(str).to_numpy()
            family_keys = df_cache.get("family_key", pd.Series([""] * len(df_cache))).astype(str).to_numpy()
            scores_p8a = df_cache["frame_prob"].astype(np.float32).to_numpy()

        np.savez_compressed(
            out_path,
            features=X, iq_features=IQ, iq_cols=np.array(IQ_COLS),
            labels=labels, frame_paths=frame_paths, video_ids=video_ids,
            identities=identities, local_paths=local_paths,
            family_keys=family_keys, scores_p8a_existing=scores_p8a,
            suite=np.array(suite),
        )
        logger.info("[%s] saved %s features=%s", suite, out_path, X.shape)

    # v2 held-out.
    v2_out = FROZEN / "visomaster_enhanced_v2_p8a_features.npz"
    if not v2_out.exists():
        logger.info("=" * 80)
        logger.info("Suite: visomaster_enhanced_v2 (held-out)")
        with open(V2_MANIFEST) as f:
            mani = json.load(f)
        rows = []
        for v in mani["videos"]:
            method = v.get("method", "")
            for fp in v.get("frame_paths", []):
                rows.append({
                    "frame_path": fp,
                    "video_id": v.get("video_id", ""),
                    "label": 1,
                    "method": method,
                    "family_key": method,
                    "frame_prob": 0.0,  # we'll fill from npz, no reference
                })
        df_v2 = pd.DataFrame(rows)
        logger.info("  v2 manifest: %d frames across %d videos / %d methods",
                    len(df_v2), df_v2["video_id"].nunique(),
                    df_v2["method"].nunique())
        # Use simple extract (no cache hits expected).
        res = extract_for_suite(model, device, "visomaster_enhanced_v2", df_v2, batch_size=32)
        np.savez_compressed(
            v2_out,
            features=res["features"], iq_features=res["iq_features"],
            iq_cols=np.array(IQ_COLS),
            labels=res["labels"], frame_paths=res["frame_paths"],
            video_ids=res["video_ids"], identities=res["identities"],
            local_paths=res["local_paths"], family_keys=res["family_keys"],
            scores_p8a_existing=res["scores_p8a_existing"],
            suite=np.array("visomaster_enhanced_v2"),
        )
        logger.info("[visomaster_enhanced_v2] saved %s features=%s",
                    v2_out, res["features"].shape)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
