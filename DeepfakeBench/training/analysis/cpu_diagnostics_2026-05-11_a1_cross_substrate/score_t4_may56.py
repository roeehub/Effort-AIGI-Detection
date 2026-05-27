"""Score T4_LAMBDA1_TOP_N_STEP10500 on the 152 may5/may6 reference frames.

Adapted from `analysis/xinhe_cross_camera_audit_2026-05-06/run_local_inference.py`.
T4 ckpt is already local at
`analysis/cpu_diagnostics_2026-05-10/_ckpts_t4/top_n_effort_20260510_step10500_auc0.9937_eer0.0195.pth`.

Writes per-frame scores to
`analysis/cpu_diagnostics_2026-05-11_a1_cross_substrate/t4_may56_per_frame.csv`.
"""

from __future__ import annotations

import csv
import logging
import sys
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

from arena.model_arena import CLIP_MEAN, CLIP_STD, load_model  # noqa: E402

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("t4_may56")

THIS_DIR = Path(__file__).resolve().parent
XINHE_DIR = REPO_ROOT / "analysis" / "xinhe_cross_camera_audit_2026-05-06"
RAW_DIR = XINHE_DIR / "raw"
OUT_CSV = THIS_DIR / "t4_may56_per_frame.csv"

T4_LOCAL = REPO_ROOT / "analysis" / "cpu_diagnostics_2026-05-10" / "_ckpts_t4" / "top_n_effort_20260510_step10500_auc0.9937_eer0.0195.pth"
DETECTOR_CONFIG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO_ROOT / "config/defaults.yaml"

# Try MPS first (Mac M-series), fallback to CPU.
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
RESOLUTION = 224
BATCH_SIZE = 16
NUM_WORKERS = 2


class LocalFrameDataset(Dataset):
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
    out: list[tuple[str, Path]] = []
    may6 = sorted((RAW_DIR / "may6").glob("*.png")) + sorted((RAW_DIR / "may6").glob("*.jpg"))
    for p in may6:
        out.append(("may6_falseflag", p))
    may5 = sorted((RAW_DIR / "may5").glob("*.png")) + sorted((RAW_DIR / "may5").glob("*.jpg"))
    for p in may5:
        out.append(("may5_correct", p))
    log.info("collected: may6=%d, may5=%d, total=%d", len(may6), len(may5), len(out))
    return out


def main() -> int:
    log.info("=" * 70)
    log.info("Local inference: T4_LAMBDA1_TOP_N_STEP10500 on may5+may6")
    log.info("device=%s", DEVICE)
    log.info("=" * 70)
    if not T4_LOCAL.exists():
        log.error("missing ckpt: %s", T4_LOCAL)
        return 2
    frames = collect_frames()
    if not frames:
        log.error("no frames")
        return 2

    log.info("loading model ...")
    t0 = time.time()
    model = load_model(str(T4_LOCAL), str(DETECTOR_CONFIG), str(TRAIN_CONFIG), DEVICE)
    log.info("model loaded in %.1fs", time.time() - t0)

    populations = [pop for pop, _ in frames]
    paths = [p for _, p in frames]

    dataset = LocalFrameDataset(paths)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=False)

    n = len(paths)
    probs = np.zeros(n, dtype=np.float32)
    log.info("inference: %d frames, batch=%d", n, BATCH_SIZE)
    t0 = time.time()
    for batch_idx, (images, indices) in enumerate(loader):
        images = images.to(DEVICE)
        with torch.inference_mode():
            outputs = model({"image": images}, inference=True)
            batch_probs = outputs["prob"].detach().cpu().numpy().reshape(-1)
        for i, gi in enumerate(indices.numpy()):
            probs[int(gi)] = float(batch_probs[i])
    log.info("inference in %.1fs (%.1f fps)", time.time() - t0, n / max(time.time() - t0, 1e-3))

    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["population", "frame_path", "frame_basename", "prob_fake"])
        for (pop, path), prob in zip(frames, probs):
            w.writerow([pop, str(path), path.name, f"{prob:.6f}"])
    log.info("wrote -> %s", OUT_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
