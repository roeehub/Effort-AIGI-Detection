#!/usr/bin/env python3
"""Step 1 — frame caching + frozen-feature extraction.

Compares raw CLIP-ViT-B-16 (LAION DataComp-XL pretrained) to P8A frozen
features on (visomaster_enhanced_macro_dev fakes vs teams_real_all_dev reals).

Both models output 512-d post-projection (visual.proj) features.

Outputs:
  outputs/clip_b16_raw__features.npz
  outputs/p8a__features.npz
  outputs/sample_manifest.csv      (per-frame metadata: frame_path, label, source, family_key, local_path)
"""
from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HERE = REPO_ROOT / "analysis" / "clip_vs_p8a_viso_2026-05-03"
FRAME_CACHE = HERE / "_frame_cache"
OUTPUTS = HERE / "outputs"
LOG_PATH = HERE / "run.log"

VISO_FAKES_CSV = (
    REPO_ROOT
    / "analysis"
    / "score_distribution_2026-05-02"
    / "raw_reports"
    / "visomaster_enhanced_macro_dev_p8a_reference_step5000_frames_report.csv"
)
TEAMS_REALS_CSV = (
    REPO_ROOT
    / "analysis"
    / "score_distribution_2026-05-02"
    / "raw_reports"
    / "teams_real_all_dev_p8a_reference_step5000_frames_report.csv"
)

CLIP_WEIGHTS = REPO_ROOT / "weights" / "CLIP-ViT-B-16-DataComp.XL-s13B-b90K"
CLIP_BIN = CLIP_WEIGHTS / "open_clip_pytorch_model.bin"
P8A_CKPT = (
    REPO_ROOT
    / "analysis"
    / "_features_cache_2026-04-30"
    / "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth"
)

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "config" / "detector" / "effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config" / "train_config.yaml"

SEED = 737
N_REALS_TARGET = 550  # cap teams_real_all_dev to balance vs 550 viso fakes

logger = logging.getLogger("clip-vs-p8a")


# -----------------------------------------------------------------------------
# Frame caching via gsutil.
# -----------------------------------------------------------------------------
def _local_name_for(uri: str) -> str:
    h = "".join(c if c.isalnum() else "_" for c in uri[len("gs://"):])[-200:]
    return h


def cached_path(uri: str) -> Path:
    return FRAME_CACHE / _local_name_for(uri)


def gsutil_cp_one(uri: str, dest: Path) -> Tuple[str, bool]:
    if dest.exists():
        return uri, True
    try:
        # -q makes gsutil quiet
        res = subprocess.run(
            ["gsutil", "-q", "cp", uri, str(dest)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        if res.returncode != 0:
            return uri, False
        return uri, True
    except Exception:
        return uri, False


def download_frames(uris: List[str], max_workers: int = 8) -> List[Path]:
    """Returns list of local paths, None for failures (wrapped: returns missing
    placeholders that should be filtered out by the caller)."""
    FRAME_CACHE.mkdir(parents=True, exist_ok=True)
    locals_: List[Path] = [cached_path(u) for u in uris]
    todo = [(u, p) for u, p in zip(uris, locals_) if not p.exists()]
    logger.info(
        "Frame cache: %d total, %d cached, %d to download",
        len(uris),
        len(uris) - len(todo),
        len(todo),
    )
    if not todo:
        return locals_

    n_done = 0
    n_fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(gsutil_cp_one, u, p): (u, p) for (u, p) in todo}
        for fut in as_completed(futs):
            u, p = futs[fut]
            uri, ok = fut.result()
            n_done += 1
            if not ok:
                n_fail += 1
            if n_done % 50 == 0 or n_done == len(todo):
                logger.info(
                    "  download progress: %d/%d (%d failed) elapsed %.1fs",
                    n_done, len(todo), n_fail, time.time() - t0,
                )
    logger.info("Downloads done: %d / %d failed", n_fail, len(todo))
    return locals_


# -----------------------------------------------------------------------------
# Sample selection.
# -----------------------------------------------------------------------------
def select_balanced_samples() -> pd.DataFrame:
    df_fake = pd.read_csv(VISO_FAKES_CSV, low_memory=False)
    df_real = pd.read_csv(TEAMS_REALS_CSV, low_memory=False)
    logger.info("Loaded %d viso fakes, %d teams reals (raw counts)", len(df_fake), len(df_real))

    # Use all viso fakes (550 expected).
    df_fake = df_fake.copy()
    df_fake["source"] = "viso_fake"

    # Stratify reals by family_key so we get a representative sample.
    rng = np.random.default_rng(SEED)
    if "family_key" in df_real.columns and df_real["family_key"].nunique() > 1:
        groups = df_real.groupby("family_key")
        per_group = max(1, N_REALS_TARGET // len(groups))
        sampled_pieces = []
        for k, g in groups:
            n = min(len(g), per_group)
            idx = rng.choice(len(g), size=n, replace=False)
            sampled_pieces.append(g.iloc[idx])
        df_real_s = pd.concat(sampled_pieces, ignore_index=True)
        if len(df_real_s) > N_REALS_TARGET:
            keep = rng.choice(len(df_real_s), size=N_REALS_TARGET, replace=False)
            df_real_s = df_real_s.iloc[keep].reset_index(drop=True)
        elif len(df_real_s) < N_REALS_TARGET:
            # Backfill from remaining frames (random)
            seen_paths = set(df_real_s["frame_path"].tolist())
            remainder = df_real[~df_real["frame_path"].isin(seen_paths)]
            n_more = N_REALS_TARGET - len(df_real_s)
            if len(remainder) >= n_more:
                add_idx = rng.choice(len(remainder), size=n_more, replace=False)
                df_real_s = pd.concat([df_real_s, remainder.iloc[add_idx]], ignore_index=True)
        df_real = df_real_s
    else:
        idx = rng.choice(len(df_real), size=min(N_REALS_TARGET, len(df_real)), replace=False)
        df_real = df_real.iloc[idx].reset_index(drop=True)
    df_real = df_real.copy()
    df_real["source"] = "teams_real_dev"

    df = pd.concat([df_fake, df_real], ignore_index=True)
    logger.info("Manifest: %d fakes, %d reals (total %d)",
                int((df["label"] == 1).sum()),
                int((df["label"] == 0).sum()),
                len(df))
    return df


# -----------------------------------------------------------------------------
# Image preprocessing.
# -----------------------------------------------------------------------------
def load_and_preprocess(local_path: Path, resolution: int = 224) -> Optional[torch.Tensor]:
    import cv2
    img = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - CLIP_MEAN) / CLIP_STD
    return torch.from_numpy(img.transpose(2, 0, 1))


# -----------------------------------------------------------------------------
# Models.
# -----------------------------------------------------------------------------
def load_raw_clip_visual(device: torch.device):
    """Load OpenCLIP ViT-B-16 with raw DataComp-XL weights, return visual tower
    that emits 512-d post-projection features.
    """
    import open_clip
    logger.info("Building OpenCLIP ViT-B-16 (no pretrained), then loading raw weights from %s", CLIP_BIN)
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained=None)
    sd = torch.load(str(CLIP_BIN), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info("Raw CLIP load: missing=%d unexpected=%d", len(missing), len(unexpected))
    if missing:
        logger.info("  example missing: %s", missing[:3])
    if unexpected:
        logger.info("  example unexpected: %s", unexpected[:3])
    visual = model.visual.to(device).eval()
    return visual


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
    logger.info("P8A load: missing=%d unexpected=%d", len(missing), len(unexpected))
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Feature extraction.
# -----------------------------------------------------------------------------
def extract_clip_visual_features(
    visual: torch.nn.Module,
    paths: List[Optional[Path]],
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    valid: List[int] = []
    pending: List[Tuple[int, torch.Tensor]] = []
    for i, p in enumerate(paths):
        if p is None or not p.exists():
            continue
        t = load_and_preprocess(p)
        if t is None:
            continue
        pending.append((i, t))
    logger.info("CLIP-raw forward: %d valid frames", len(pending))
    for j in range(0, len(pending), batch_size):
        chunk = pending[j : j + batch_size]
        idxs = [c[0] for c in chunk]
        x = torch.stack([c[1] for c in chunk]).to(device)
        with torch.inference_mode():
            f = visual(x)  # 512-d post-projection
            if isinstance(f, (tuple, list)):
                f = f[0]
        feats.append(f.detach().to("cpu").to(torch.float32).numpy())
        valid.extend(idxs)
        if (j // batch_size) % 10 == 0:
            logger.info("  CLIP raw batch %d / %d", j // batch_size + 1, (len(pending) + batch_size - 1) // batch_size)
    if not feats:
        return np.zeros((0, 0), dtype=np.float32), np.array([], dtype=np.int64)
    return np.concatenate(feats, axis=0), np.array(valid, dtype=np.int64)


def extract_p8a_features(
    model: torch.nn.Module,
    paths: List[Optional[Path]],
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    valid: List[int] = []
    pending: List[Tuple[int, torch.Tensor]] = []
    for i, p in enumerate(paths):
        if p is None or not p.exists():
            continue
        t = load_and_preprocess(p)
        if t is None:
            continue
        pending.append((i, t))
    logger.info("P8A forward: %d valid frames", len(pending))
    for j in range(0, len(pending), batch_size):
        chunk = pending[j : j + batch_size]
        idxs = [c[0] for c in chunk]
        x = torch.stack([c[1] for c in chunk]).to(device)
        with torch.inference_mode():
            out = model({"image": x}, inference=True)
            f = out["feat"].detach().to("cpu").to(torch.float32).numpy()
        feats.append(f)
        valid.extend(idxs)
        if (j // batch_size) % 10 == 0:
            logger.info("  P8A batch %d / %d", j // batch_size + 1, (len(pending) + batch_size - 1) // batch_size)
    if not feats:
        return np.zeros((0, 0), dtype=np.float32), np.array([], dtype=np.int64)
    return np.concatenate(feats, axis=0), np.array(valid, dtype=np.int64)


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------
def main() -> int:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FRAME_CACHE.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(LOG_PATH)
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.handlers = [fh, sh]
    logger.setLevel(logging.INFO)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logger.info("device=%s", device)

    # 1. Build manifest + download frames.
    manifest_path = OUTPUTS / "sample_manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        logger.info("Loaded existing manifest: %d rows", len(df))
    else:
        df = select_balanced_samples()
        df.to_csv(manifest_path, index=False)
    uris = df["frame_path"].tolist()

    locals_ = download_frames(uris, max_workers=8)
    df["local_path"] = [str(p) if p.exists() else "" for p in locals_]
    df.to_csv(manifest_path, index=False)

    # Drop rows whose download failed.
    n_total = len(df)
    df_valid = df[df["local_path"] != ""].reset_index(drop=True)
    n_valid = len(df_valid)
    logger.info("After download: %d / %d frames available locally", n_valid, n_total)
    n_fakes = int((df_valid["label"] == 1).sum())
    n_reals = int((df_valid["label"] == 0).sum())
    logger.info("  fakes=%d reals=%d", n_fakes, n_reals)
    if n_fakes < 400 or n_reals < 400:
        logger.error("Too many failed downloads (need >=400 of each); aborting")
        return 1

    paths_local: List[Optional[Path]] = [Path(p) for p in df_valid["local_path"].tolist()]

    # 2. Extract raw CLIP features.
    out_clip = OUTPUTS / "clip_b16_raw__features.npz"
    if out_clip.exists():
        logger.info("CLIP-raw features cached at %s, reusing", out_clip)
    else:
        visual = load_raw_clip_visual(device)
        feats, valid = extract_clip_visual_features(visual, paths_local, device, batch_size=32)
        df_kept = df_valid.iloc[valid].reset_index(drop=True)
        np.savez_compressed(
            out_clip,
            features=feats.astype(np.float32),
            label=df_kept["label"].astype(np.int32).to_numpy(),
            frame_path=df_kept["frame_path"].astype(str).to_numpy(),
            source=df_kept["source"].astype(str).to_numpy(),
            family_key=df_kept.get("family_key", pd.Series([""] * len(df_kept))).astype(str).to_numpy(),
            local_path=df_kept["local_path"].astype(str).to_numpy(),
        )
        logger.info("Saved %s (features %s, n=%d)", out_clip, feats.shape, len(df_kept))
        del visual
        if device.type == "mps":
            torch.mps.empty_cache()

    # 3. Extract P8A features.
    out_p8a = OUTPUTS / "p8a__features.npz"
    if out_p8a.exists():
        logger.info("P8A features cached at %s, reusing", out_p8a)
    else:
        model = load_p8a_model(device)
        feats, valid = extract_p8a_features(model, paths_local, device, batch_size=32)
        df_kept = df_valid.iloc[valid].reset_index(drop=True)
        np.savez_compressed(
            out_p8a,
            features=feats.astype(np.float32),
            label=df_kept["label"].astype(np.int32).to_numpy(),
            frame_path=df_kept["frame_path"].astype(str).to_numpy(),
            source=df_kept["source"].astype(str).to_numpy(),
            family_key=df_kept.get("family_key", pd.Series([""] * len(df_kept))).astype(str).to_numpy(),
            local_path=df_kept["local_path"].astype(str).to_numpy(),
        )
        logger.info("Saved %s (features %s, n=%d)", out_p8a, feats.shape, len(df_kept))
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
