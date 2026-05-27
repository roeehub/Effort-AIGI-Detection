"""Local CPU inference for the 92 xinhe-may6 + 60 may5 reference frames.

Loads P8A, E2B, PA_3800 checkpoints from GCS, runs forward in inference_mode
on CPU, writes per-ckpt CSV to outputs/.

Reuses arena/model_arena.load_model so preprocessing matches training/eval
(INTER_LINEAR resize, BGR2RGB, CLIP normalization).

Usage:
    cd analysis/xinhe_cross_camera_audit_2026-05-06
    python run_local_inference.py
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]  # training/
sys.path.insert(0, str(REPO_ROOT))

from arena.model_arena import (  # noqa: E402
    CLIP_MEAN, CLIP_STD,
    _download_checkpoint,
    load_model,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("xinhe_local_inference")

THIS_DIR = Path(__file__).resolve().parent
RAW_DIR = THIS_DIR / "raw"
OUT_DIR = THIS_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DETECTOR_CONFIG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO_ROOT / "config/defaults.yaml"

CKPTS = {
    "P8A": "gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth",
    "E2B": "gs://training-job-outputs/best_checkpoints/rmat8lwx/top_n_effort_20260503_step3200_auc0.9863_eer0.0310.pth",
    "PA_3800": "gs://training-job-outputs/best_checkpoints/26u8bn1t/top_n_effort_20260504_step3800_auc0.9892_eer0.0402.pth",
    "T3_S1_STEP1500": "gs://training-job-outputs/best_checkpoints/bxnuz22g/periodic_effort_20260509_step1500_auc0.9817_eer0.0745.pth",
    "T3_S1_STEP2500": "gs://training-job-outputs/best_checkpoints/bxnuz22g/periodic_effort_20260509_step2500_auc0.9886_eer0.0311.pth",
}

DEVICE = torch.device("cpu")
RESOLUTION = 224
BATCH_SIZE = 8
NUM_WORKERS = 2  # Mac n_jobs caveat (memory feedback_sklearn_njobs.md)


class LocalFrameDataset(Dataset):
    """Local-disk version of GCSFrameDataset — same preprocessing."""

    def __init__(self, frame_paths: list[Path], resolution: int = RESOLUTION):
        self.paths = list(frame_paths)
        self.resolution = resolution
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            log.warning("decode-fail: %s", path)
            return torch.zeros(3, self.resolution, self.resolution), idx
        img_bgr = cv2.resize(img_bgr, (self.resolution, self.resolution),
                             interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(img_rgb), idx


def collect_frames() -> list[tuple[str, Path]]:
    """Return list of (population_label, path) tuples for all 152 frames."""
    out: list[tuple[str, Path]] = []
    may6 = sorted((RAW_DIR / "may6").glob("*.png")) + sorted((RAW_DIR / "may6").glob("*.jpg"))
    for p in may6:
        out.append(("may6_falseflag", p))
    may5 = sorted((RAW_DIR / "may5").glob("*.png")) + sorted((RAW_DIR / "may5").glob("*.jpg"))
    for p in may5:
        out.append(("may5_correct", p))
    log.info("collected: may6=%d, may5=%d, total=%d", len(may6), len(may5), len(out))
    return out


def score_one_ckpt(ckpt_name: str, ckpt_uri: str, frames: Iterable[tuple[str, Path]]) -> Path:
    """Download + load + score one ckpt. Returns CSV path."""
    out_csv = OUT_DIR / f"scores_{ckpt_name}.csv"
    if out_csv.exists():
        log.info("[%s] CSV exists, skipping: %s", ckpt_name, out_csv)
        return out_csv

    log.info("[%s] downloading checkpoint ...", ckpt_name)
    t0 = time.time()
    local_ckpt = _download_checkpoint(ckpt_uri)
    log.info("[%s] download in %.1fs -> %s", ckpt_name, time.time() - t0, local_ckpt)

    log.info("[%s] loading model on %s ...", ckpt_name, DEVICE)
    t0 = time.time()
    model = load_model(str(local_ckpt), str(DETECTOR_CONFIG), str(TRAIN_CONFIG), DEVICE)
    log.info("[%s] model loaded in %.1fs", ckpt_name, time.time() - t0)

    frames_list = list(frames)
    populations = [pop for pop, _ in frames_list]
    paths = [p for _, p in frames_list]

    dataset = LocalFrameDataset(paths)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    n = len(paths)
    probs = np.zeros(n, dtype=np.float32)
    log.info("[%s] running inference: %d frames, batch=%d, num_workers=%d",
             ckpt_name, n, BATCH_SIZE, NUM_WORKERS)
    t0 = time.time()
    for batch_idx, (images, indices) in enumerate(loader):
        images = images.to(DEVICE)
        with torch.inference_mode():
            outputs = model({"image": images}, inference=True)
            batch_probs = outputs["prob"].detach().cpu().numpy().reshape(-1)
        for i, gi in enumerate(indices.numpy()):
            probs[int(gi)] = float(batch_probs[i])
        elapsed = time.time() - t0
        done = (batch_idx + 1) * BATCH_SIZE
        log.info("[%s] batch %d (~%d/%d) elapsed=%.1fs (%.1f fps)",
                 ckpt_name, batch_idx + 1, min(done, n), n,
                 elapsed, min(done, n) / max(elapsed, 1e-3))

    log.info("[%s] inference done in %.1fs", ckpt_name, time.time() - t0)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["population", "frame_path", "frame_basename", "prob_fake"])
        for (pop, path), prob in zip(frames_list, probs):
            w.writerow([pop, str(path), path.name, f"{prob:.6f}"])
    log.info("[%s] wrote -> %s", ckpt_name, out_csv)

    # Free the model before loading the next
    del model
    return out_csv


def main() -> int:
    log.info("=" * 70)
    log.info("Local CPU inference: P8A + E2B + PA_3800 on xinhe-may6 + may5")
    log.info("=" * 70)
    if not DETECTOR_CONFIG.exists():
        log.error("missing detector config: %s", DETECTOR_CONFIG)
        return 2
    if not TRAIN_CONFIG.exists():
        log.error("missing train config: %s", TRAIN_CONFIG)
        return 2
    frames = collect_frames()
    if not frames:
        log.error("no frames found")
        return 2
    for name, uri in CKPTS.items():
        try:
            score_one_ckpt(name, uri, frames)
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] FAILED: %s", name, exc, exc_info=True)
    log.info("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
