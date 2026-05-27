#!/usr/bin/env python3
"""Score PA_TOP_N_STEP3800 on the 3 hard real teams suites that were missing
from the original pa_pc_eval_2026-05-05 run:

  - teams_real_dor_dev (50 frames)
  - teams_real_poor_quality_dev (1303 frames)
  - teams_real_lighting_extreme_dev (1742 frames)

Inputs:
  - frame paths (gs:// URIs) extracted from the P8A reference frame reports
    under analysis/cpu_followups_2026-05-04/raw_reports/
  - PA_TOP_N_STEP3800 checkpoint:
    gs://training-job-outputs/best_checkpoints/26u8bn1t/top_n_effort_20260504_step3800_auc0.9892_eer0.0402.pth
  - cached frames where available:
      * 50 dor frames: analysis/score_distribution_2026-05-02/outputs/cross_suite_samples/teams_real_dor_dev/
      * poor_quality / lighting_extreme: analysis/lockbox_tagging/_frame_cache (md5 hashed)

Outputs (cpu_followups schema):
  analysis/pa_pc_eval_2026-05-05/raw_reports/teams_real_dor_dev_pa_top_n_step3800_frames_report.csv
  analysis/pa_pc_eval_2026-05-05/raw_reports/teams_real_poor_quality_dev_pa_top_n_step3800_frames_report.csv
  analysis/pa_pc_eval_2026-05-05/raw_reports/teams_real_lighting_extreme_dev_pa_top_n_step3800_frames_report.csv

Plus log:
  analysis/pa_pc_eval_2026-05-05/raw_reports/pa_3800_hard_suites_inference_2026-05-05.log

CPU only. Uses cv2.INTER_LINEAR (post-fix preprocessing parity).
"""
from __future__ import annotations

import csv
import gc
import hashlib
import logging
import os
import sys
import subprocess
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
import yaml

TRAINING_DIR = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from detectors import DETECTOR  # noqa: E402

logger = logging.getLogger("pa3800-hard")

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

CKPT_NAME = "PA_TOP_N_STEP3800"
CKPT_GCS = "gs://training-job-outputs/best_checkpoints/26u8bn1t/top_n_effort_20260504_step3800_auc0.9892_eer0.0402.pth"

SUITES = [
    {
        "name": "teams_real_dor_dev",
        "src_csv": TRAINING_DIR / "analysis/cpu_followups_2026-05-04/raw_reports/teams_real_dor_dev_p8a_reference_step5000_frames_report.csv",
        "out_csv": TRAINING_DIR / "analysis/pa_pc_eval_2026-05-05/raw_reports/teams_real_dor_dev_pa_top_n_step3800_frames_report.csv",
    },
    {
        "name": "teams_real_poor_quality_dev",
        "src_csv": TRAINING_DIR / "analysis/cpu_followups_2026-05-04/raw_reports/teams_real_poor_quality_dev_p8a_reference_step5000_frames_report.csv",
        "out_csv": TRAINING_DIR / "analysis/pa_pc_eval_2026-05-05/raw_reports/teams_real_poor_quality_dev_pa_top_n_step3800_frames_report.csv",
    },
    {
        "name": "teams_real_lighting_extreme_dev",
        "src_csv": TRAINING_DIR / "analysis/cpu_followups_2026-05-04/raw_reports/teams_real_lighting_extreme_dev_p8a_reference_step5000_frames_report.csv",
        "out_csv": TRAINING_DIR / "analysis/pa_pc_eval_2026-05-05/raw_reports/teams_real_lighting_extreme_dev_pa_top_n_step3800_frames_report.csv",
    },
]

# Caches (read-only lookups, in priority order)
LOCKBOX_CACHE = TRAINING_DIR / "analysis/lockbox_tagging/_frame_cache"
DOR_CACHE = TRAINING_DIR / "analysis/score_distribution_2026-05-02/outputs/cross_suite_samples"

# New cache for any frames we have to download (so future re-runs hit it)
HARD_CACHE = TRAINING_DIR / "analysis/pa_pc_eval_2026-05-05/_frame_cache_hard_suites"
HARD_CACHE.mkdir(parents=True, exist_ok=True)

CKPT_DIR = TRAINING_DIR / "analysis/pa_pc_eval_2026-05-05/checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def lockbox_cache_path(gs_uri: str) -> Path:
    """Recreate analysis/lockbox_tagging/io_utils._cache_path layout (md5 of blob_path)."""
    assert gs_uri.startswith("gs://"), gs_uri
    blob = gs_uri[5:].split("/", 1)[1]
    name = Path(blob).name
    h = hashlib.md5(blob.encode()).hexdigest()
    return LOCKBOX_CACHE / h[:2] / h[2:4] / name


def dor_cache_path(gs_uri: str, suite: str) -> Optional[Path]:
    name = Path(gs_uri).name
    p = DOR_CACHE / suite / name
    return p if p.exists() else None


def hard_cache_path(gs_uri: str) -> Path:
    assert gs_uri.startswith("gs://"), gs_uri
    blob = gs_uri[5:].split("/", 1)[1]
    name = Path(blob).name
    h = hashlib.md5(blob.encode()).hexdigest()
    return HARD_CACHE / h[:2] / h[2:4] / name


def resolve_local_path(gs_uri: str, suite: str) -> Tuple[Optional[Path], str]:
    """Return (local_path, source_tag) or (None, 'missing')."""
    # Try lockbox cache
    p = lockbox_cache_path(gs_uri)
    if p.exists() and p.stat().st_size > 0:
        return p, "lockbox_cache"
    # Try dor cross_suite_samples
    if suite == "teams_real_dor_dev":
        p = dor_cache_path(gs_uri, suite)
        if p is not None:
            return p, "dor_cross_suite"
    # Try hard cache (downloaded this session)
    p = hard_cache_path(gs_uri)
    if p.exists() and p.stat().st_size > 0:
        return p, "hard_cache"
    return None, "missing"


def gsutil_download_one(gs_uri: str) -> Optional[Path]:
    """Download to hard_cache if not already present. Returns local path or None on failure."""
    target = hard_cache_path(gs_uri)
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            ["gsutil", "-q", "cp", gs_uri, str(target)],
            check=False, capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            return None
        if not target.exists() or target.stat().st_size == 0:
            return None
        return target
    except Exception:
        return None


def download_missing_parallel(missing: List[str], max_workers: int = 32) -> Dict[str, Optional[Path]]:
    out: Dict[str, Optional[Path]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(gsutil_download_one, gs): gs for gs in missing}
        n_done = 0
        for fut in as_completed(futs):
            gs = futs[fut]
            try:
                out[gs] = fut.result()
            except Exception as e:
                out[gs] = None
                logger.warning("download exception: %s -> %s", gs, e)
            n_done += 1
            if n_done % 100 == 0 or n_done == len(missing):
                ok = sum(1 for v in out.values() if v is not None)
                logger.info("    download progress %d/%d (ok=%d)", n_done, len(missing), ok)
    return out


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    detector_config = TRAINING_DIR / "config" / "detector" / "effort.yaml"
    train_config = TRAINING_DIR / "config" / "train_config.yaml"
    with open(detector_config, "r") as f:
        cfg = yaml.safe_load(f)
    with open(train_config, "r") as f:
        train_cfg = yaml.safe_load(f)
    cfg.update(train_cfg)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        model_config = ckpt.get("model_config", {})
        for k, v in model_config.items():
            if k != "current_arcface_s":
                cfg[k] = v
    else:
        state_dict = ckpt
        model_config = {}

    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)

    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])
            logger.info("    Restored ArcFace s=%.3f", model_config["current_arcface_s"])

    clean_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing:
        logger.debug("    Missing keys: %d", len(missing))
    if unexpected:
        logger.debug("    Unexpected keys: %d", len(unexpected))

    model.eval()
    return model


class FrameDataset(data.Dataset):
    """Reads pre-cropped face images, resizes to 224 with INTER_LINEAR, applies CLIP norm."""
    def __init__(self, items: List[Tuple[Path, int]], resolution: int = 224):
        self.items = items
        self.resolution = resolution
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        local_path, gi = self.items[idx]
        img_bgr = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return torch.zeros(3, self.resolution, self.resolution), idx, 0
        img_bgr = cv2.resize(img_bgr, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(img_rgb), idx, 1


def score_items(
    model: torch.nn.Module,
    items: List[Tuple[Path, int]],
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[Dict[int, float], int]:
    if not items:
        return {}, 0
    dataset = FrameDataset(items)
    loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    probs: Dict[int, float] = {}
    n_load_failed = 0
    t0 = time.time()
    n_batches = len(loader)
    for bi, (images, indices, ok) in enumerate(loader):
        images = images.to(device)
        with torch.inference_mode():
            outputs = model({"image": images}, inference=True)
            batch_probs = outputs["prob"].detach().cpu().numpy().reshape(-1)
        for j, gi in enumerate(indices.numpy()):
            if int(ok[j]) == 0:
                n_load_failed += 1
                continue
            probs[int(items[int(gi)][1])] = float(batch_probs[j])
        if (bi + 1) % 10 == 0 or (bi + 1) == n_batches:
            elapsed = time.time() - t0
            done = len(probs)
            fps = done / max(elapsed, 1e-6)
            logger.info("    batch %d/%d  scored=%d/%d  %.2f fps  %.1fs",
                        bi + 1, n_batches, done, len(items), fps, elapsed)
    return probs, n_load_failed


def process_suite(model: torch.nn.Module, suite: Dict[str, Any], device: torch.device,
                  batch_size: int, num_workers: int) -> Dict[str, Any]:
    name = suite["name"]
    src_csv = suite["src_csv"]
    out_csv = suite["out_csv"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SUITE %s", name)
    logger.info("  reading source: %s", src_csv)
    rows = list(csv.DictReader(open(src_csv)))
    n_total = len(rows)
    logger.info("  %d rows", n_total)

    # Resolve local paths
    resolved: List[Tuple[int, Path]] = []
    sources: Dict[str, int] = {}
    missing: List[Tuple[int, str]] = []
    for i, r in enumerate(rows):
        gs = r["frame_path"]
        local, src_tag = resolve_local_path(gs, name)
        sources[src_tag] = sources.get(src_tag, 0) + 1
        if local is None:
            missing.append((i, gs))
        else:
            resolved.append((i, local))
    logger.info("  cache lookup: %s", dict(sources))
    logger.info("  resolved=%d, missing=%d", len(resolved), len(missing))

    # Download any missing
    if missing:
        logger.info("  downloading %d missing frames in parallel...", len(missing))
        t_dl = time.time()
        gs_uris = [m[1] for m in missing]
        out = download_missing_parallel(gs_uris, max_workers=32)
        n_dl_ok = sum(1 for v in out.values() if v is not None)
        n_dl_fail = sum(1 for v in out.values() if v is None)
        logger.info("  download: ok=%d, failed=%d in %.1fs", n_dl_ok, n_dl_fail,
                    time.time() - t_dl)
        for i, gs in missing:
            p = out.get(gs)
            if p is not None and p.exists() and p.stat().st_size > 0:
                resolved.append((i, p))

    logger.info("  total resolvable for inference: %d/%d", len(resolved), n_total)

    # Score
    items: List[Tuple[Path, int]] = [(p, idx) for idx, p in resolved]
    t_inf = time.time()
    probs_by_idx, n_load_fail = score_items(
        model, items, device, batch_size=batch_size, num_workers=num_workers
    )
    inf_dt = time.time() - t_inf
    logger.info("  inference done: %d frames scored, %d load-fail, %.1fs (%.2f fps)",
                len(probs_by_idx), n_load_fail, inf_dt, len(probs_by_idx) / max(inf_dt, 1e-6))

    # Write CSV in same schema as source: method, label, video_id, frame_path, frame_prob, group_key, family_key
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["method", "label", "video_id", "frame_path", "frame_prob", "group_key", "family_key"]
        )
        w.writeheader()
        n_written = 0
        n_missing_score = 0
        for i, r in enumerate(rows):
            prob = probs_by_idx.get(i)
            if prob is None:
                n_missing_score += 1
                # Skip rows we couldn't score (consistent with honest reporting)
                continue
            w.writerow({
                "method": r["method"],
                "label": r["label"],
                "video_id": r["video_id"],
                "frame_path": r["frame_path"],
                "frame_prob": f"{prob:.8f}",
                "group_key": r["group_key"],
                "family_key": r["family_key"],
            })
            n_written += 1
    logger.info("  wrote %d rows (skipped %d unscored) -> %s", n_written, n_missing_score, out_csv)

    # Stats
    if n_written > 0:
        scored = np.array([probs_by_idx[i] for i in range(len(rows)) if i in probs_by_idx], dtype=np.float64)
        stats = {
            "n_total": n_total,
            "n_scored": int(scored.size),
            "n_missing_score": int(n_missing_score),
            "mean": float(scored.mean()),
            "p90": float(np.percentile(scored, 90)),
            "p99": float(np.percentile(scored, 99)),
            "frac_above_0_5": float((scored >= 0.5).mean()),
            "frac_above_0_7": float((scored >= 0.7).mean()),
            "inference_seconds": inf_dt,
            "fps": len(probs_by_idx) / max(inf_dt, 1e-6),
        }
    else:
        stats = {"n_total": n_total, "n_scored": 0, "n_missing_score": n_missing_score}
    logger.info("  stats: %s", stats)
    return {"name": name, "stats": stats, "n_load_fail": n_load_fail, "out_csv": str(out_csv)}


def main():
    log_path = TRAINING_DIR / "analysis/pa_pc_eval_2026-05-05/raw_reports/pa_3800_hard_suites_inference_2026-05-05.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_path), mode="w")
    sh = logging.StreamHandler(sys.stderr)
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.setLevel(logging.INFO)
    logger.addHandler(fh); logger.addHandler(sh)

    # Set torch threads conservatively (Mac, 10 cores, 32GB RAM)
    n_threads = min(8, os.cpu_count() or 8)
    torch.set_num_threads(n_threads)

    logger.info("=" * 70)
    logger.info("PA_3800 hard-suites inference run starting")
    logger.info("=" * 70)
    logger.info("ckpt=%s", CKPT_NAME)
    logger.info("ckpt_gcs=%s", CKPT_GCS)
    logger.info("torch threads=%d", torch.get_num_threads())

    overall_t0 = time.time()
    device = torch.device("cpu")

    # Download the checkpoint
    local_ckpt = CKPT_DIR / Path(CKPT_GCS).name
    if not local_ckpt.exists() or local_ckpt.stat().st_size < 1_000_000:
        logger.info("downloading checkpoint to %s ...", local_ckpt)
        t_dl = time.time()
        subprocess.run(["gsutil", "-q", "cp", CKPT_GCS, str(local_ckpt)], check=True)
        logger.info("  ckpt download: %.1fs (%.1f MB)",
                    time.time() - t_dl, local_ckpt.stat().st_size / 1e6)
    else:
        logger.info("ckpt cached at %s (%.1f MB)", local_ckpt, local_ckpt.stat().st_size / 1e6)

    t_load = time.time()
    model = load_model(local_ckpt, device)
    logger.info("model loaded in %.1fs", time.time() - t_load)

    results = []
    for suite in SUITES:
        try:
            r = process_suite(model, suite, device, batch_size=16, num_workers=2)
            results.append(r)
        except Exception as e:
            logger.exception("SUITE %s FAILED: %s", suite["name"], e)
            results.append({"name": suite["name"], "error": str(e)})
        gc.collect()

    overall_dt = time.time() - overall_t0
    logger.info("=" * 70)
    logger.info("ALL DONE in %.1fs (%.1f min)", overall_dt, overall_dt / 60)
    logger.info("=" * 70)
    logger.info("SUMMARY")
    for r in results:
        logger.info("  %s: %s", r.get("name"),
                    r.get("stats") if "stats" in r else "FAILED: " + str(r.get("error")))


if __name__ == "__main__":
    main()
