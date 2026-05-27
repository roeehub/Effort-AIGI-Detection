"""may6 production-drift retest for P2_D_FOURIER_PERIODIC_STEP3000.

Adapted from run_t5c_step3500_may6.py. Same harness, different ckpt.
P2D = B16 scratch from CLIP + band-limited Fourier amp aug (run 89tt9xyz).
"""
from __future__ import annotations

import csv
import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arena.model_arena import CLIP_MEAN, CLIP_STD  # noqa: E402
from detectors import DETECTOR  # noqa: E402

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("p2d_step3000_may6")

CKPT_ALIAS = "P2D_FOURIER_PERIODIC_STEP3000"
CKPT_GCS = "gs://training-job-outputs/best_checkpoints/89tt9xyz/periodic_effort_20260507_step3000_auc0.9738_eer0.0823.pth"
CKPT_BASENAME = "periodic_effort_20260507_step3000_auc0.9738_eer0.0823.pth"
CKPT_LOCAL = THIS_DIR / "_ckpt_cache" / CKPT_BASENAME

RAW_DIR = REPO_ROOT / "analysis/xinhe_cross_camera_audit_2026-05-06/raw/may6"
PRIOR_OUT_DIR = REPO_ROOT / "analysis/xinhe_cross_camera_audit_2026-05-06/outputs"
P8A_CSV = PRIOR_OUT_DIR / "scores_P8A.csv"
OUT_DIR = THIS_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DETECTOR_CONFIG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO_ROOT / "config/defaults.yaml"

try:
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
except Exception:
    DEVICE = torch.device("cpu")

RESOLUTION = 224
BATCH_SIZE = 8
NUM_WORKERS = 2


class LocalFrameDataset(Dataset):
    def __init__(self, frame_paths, resolution: int = RESOLUTION):
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


def collect_may6_frames():
    paths = sorted(RAW_DIR.glob("*.png")) + sorted(RAW_DIR.glob("*.jpg"))
    log.info("collected: may6=%d frames from %s", len(paths), RAW_DIR)
    return paths


def download_ckpt():
    import subprocess
    CKPT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    if CKPT_LOCAL.exists():
        log.info("ckpt cached: %s", CKPT_LOCAL)
        return
    log.info("downloading %s -> %s", CKPT_GCS, CKPT_LOCAL)
    subprocess.run(["gsutil", "-q", "cp", CKPT_GCS, str(CKPT_LOCAL)], check=True)
    log.info("ckpt download done")


def load_p2d_model(ckpt_path: Path, device: torch.device):
    """Build EffortDetector matching the saved P2D model_config + load state."""
    import yaml

    with open(DETECTOR_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    with open(TRAIN_CONFIG, "r") as f:
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

    cfg["multi_axis_grl"] = {"enabled": False}

    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)
    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])

    clean_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    log.info("  state_dict load: %d missing, %d unexpected", len(missing), len(unexpected))
    if unexpected:
        log.info("  unexpected (first 5): %s", unexpected[:5])
    if missing:
        log.warning("  missing (first 5): %s", missing[:5])
    model.eval()
    return model


def score():
    download_ckpt()
    if not CKPT_LOCAL.exists():
        log.error("ckpt download failed")
        return None

    out_csv = OUT_DIR / f"scores_{CKPT_ALIAS}.csv"
    if out_csv.exists():
        log.info("CSV exists, reusing: %s", out_csv)
        return out_csv

    frames = collect_may6_frames()
    log.info("loading P2D step3000 on %s ...", DEVICE)
    t0 = time.time()
    model = load_p2d_model(CKPT_LOCAL, DEVICE)
    log.info("model loaded in %.1fs", time.time() - t0)

    ds = LocalFrameDataset(frames)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=False)

    n = len(frames)
    probs = np.zeros(n, dtype=np.float32)
    log.info("running inference: %d frames, batch=%d, device=%s", n, BATCH_SIZE, DEVICE)
    t0 = time.time()
    for batch_idx, (images, indices) in enumerate(loader):
        images = images.to(DEVICE)
        with torch.inference_mode():
            outputs = model({"image": images}, inference=True)
            batch_probs = outputs["prob"].detach().cpu().numpy().reshape(-1)
        for i, gi in enumerate(indices.numpy()):
            probs[int(gi)] = float(batch_probs[i])
    log.info("inference done in %.1fs", time.time() - t0)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["population", "frame_path", "frame_basename", "prob_fake"])
        for path, prob in zip(frames, probs):
            w.writerow(["may6_falseflag", str(path), path.name, f"{prob:.6f}"])
    log.info("wrote -> %s", out_csv)
    return out_csv


def summarize(out_csv: Path):
    df = pd.read_csv(out_csv)
    probs = df["prob_fake"].to_numpy()
    n_fired = int((probs > 0.5).sum())
    log.info("=== may6 retest verdict for P2D step3000 ===")
    log.info("n=%d, n_fired@0.5=%d (%.1f%%), p50=%.4f, p90=%.4f, p99=%.4f, max=%.4f",
             len(probs), n_fired, 100*n_fired/len(probs),
             float(np.median(probs)), float(np.quantile(probs, 0.9)),
             float(np.quantile(probs, 0.99)), float(probs.max()))

    # Update may6_retest_table.csv
    table_csv = OUT_DIR / "may6_retest_table.csv"
    if table_csv.exists():
        t = pd.read_csv(table_csv)
        t = t[t["ckpt"] != CKPT_ALIAS]
        new_row = {
            "ckpt": CKPT_ALIAS,
            "source": "best-candidate-search follow-up",
            "n": len(probs),
            "n_fired@0.5": n_fired,
            "p50": float(np.median(probs)),
            "p90": float(np.quantile(probs, 0.9)),
            "p99": float(np.quantile(probs, 0.99)),
            "max": float(probs.max()),
        }
        t = pd.concat([t, pd.DataFrame([new_row])], ignore_index=True)
        t.to_csv(table_csv, index=False)
        log.info("updated %s", table_csv)


def main():
    out_csv = score()
    if out_csv:
        summarize(out_csv)


if __name__ == "__main__":
    main()
