"""Score the 92 may6 + 60 may5 frames on T5C, Slot A v2 CLS, and Slot A v2 face-pool.

Reuses the existing preprocessing from the 2026-05-06 audit (INTER_LINEAR resize,
BGR2RGB, CLIP normalization) for byte-equivalence with the cached P8A/E2B scores.

Outputs per-ckpt CSVs to outputs/scores_<KEY>.csv with columns:
    population, frame_path, frame_basename, prob_fake

Usage:
    python analysis/xinhe_may6_t5c_revisit_2026-05-23/scripts/score_may6_may5.py
    python analysis/xinhe_may6_t5c_revisit_2026-05-23/scripts/score_may6_may5.py --only T5C
    python analysis/xinhe_may6_t5c_revisit_2026-05-23/scripts/score_may6_may5.py --only T5C,SLOT_A_V2_CLS,SLOT_A_V2_FACE_POOL
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analysis/face_pool_canary_2026-05-22"))

from arena.model_arena import CLIP_MEAN, CLIP_STD  # noqa: E402
from batch_inference_gcs import load_model  # noqa: E402

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("xinhe_t5c_revisit")

THIS_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "analysis/xinhe_cross_camera_audit_2026-05-06/raw"
OUT_DIR = THIS_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DETECTOR_CFG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CFG = REPO_ROOT / "config/train_config.yaml"
DETECTOR_CFG_FALLBACK = REPO_ROOT / "config/defaults.yaml"

LOCAL_CKPT_DIR = REPO_ROOT / "analysis/manual_canary_2026-05-20/ckpts"
PRIOR_E2B_CKPT = REPO_ROOT / "analysis/team_identity_deploy_readout_2026-05-23/_ckpts/E2B_TOP_N_STEP3200.pth"

# ckpt_key: (path, use_face_pool)
# We re-score all 5 ckpts for byte-equivalence verification, even though P8A/E2B
# CSVs already exist (cached) in xinhe_cross_camera_audit_2026-05-06/outputs/.
CKPTS = {
    "P8A_REFERENCE_STEP5000": (
        LOCAL_CKPT_DIR / "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth",
        False,
    ),
    "E2B_TOP_N_STEP3200": (PRIOR_E2B_CKPT, False),
    "T5C_PERIODIC_STEP3500": (
        LOCAL_CKPT_DIR / "periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth",
        False,
    ),
    "SLOT_A_V2_CLS_STEP3500": (
        LOCAL_CKPT_DIR / "periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth",
        False,
    ),
    "SLOT_A_V2_FACE_POOL_STEP3500": (
        LOCAL_CKPT_DIR / "periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth",
        True,
    ),
}

RESOLUTION = 224
BATCH_SIZE = 8
NUM_WORKERS = 2


class LocalFrameDataset(Dataset):
    """Local-disk dataset matching arena.model_arena GCSFrameDataset preprocessing."""

    def __init__(self, frame_paths: List[Path], resolution: int = RESOLUTION):
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


def collect_frames() -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    may6 = sorted((RAW_DIR / "may6").glob("*.png")) + sorted((RAW_DIR / "may6").glob("*.jpg"))
    for p in may6:
        out.append(("may6_falseflag", p))
    may5 = sorted((RAW_DIR / "may5").glob("*.png")) + sorted((RAW_DIR / "may5").glob("*.jpg"))
    for p in may5:
        out.append(("may5_correct", p))
    log.info("collected: may6=%d, may5=%d, total=%d", len(may6), len(may5), len(out))
    return out


def score_one_ckpt(ckpt_key: str, ckpt_path: Path, use_face_pool: bool,
                   frames: List[Tuple[str, Path]], device: torch.device) -> Path:
    out_csv = OUT_DIR / f"scores_{ckpt_key}.csv"
    if out_csv.exists():
        log.info("[%s] CSV exists, skipping: %s", ckpt_key, out_csv)
        return out_csv

    if not ckpt_path.exists():
        log.error("[%s] ckpt not found: %s", ckpt_key, ckpt_path)
        return out_csv

    log.info("[%s] loading model on %s ...", ckpt_key, device)
    t0 = time.time()
    # Pick a detector config that exists
    det_cfg = DETECTOR_CFG if DETECTOR_CFG.exists() else DETECTOR_CFG_FALLBACK
    # Pick a train config that exists
    train_cfg = TRAIN_CFG
    if not train_cfg.exists():
        train_cfg = REPO_ROOT / "config/defaults.yaml"
    model = load_model(str(ckpt_path), str(det_cfg), str(train_cfg), device)
    model.eval()
    log.info("[%s] model loaded in %.1fs", ckpt_key, time.time() - t0)

    populations = [pop for pop, _ in frames]
    paths = [p for _, p in frames]

    dataset = LocalFrameDataset(paths)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
    )

    n = len(paths)
    probs = np.zeros(n, dtype=np.float32)

    if use_face_pool:
        from score_canary_face_pool import FacePoolMonkeyPatch, L11_LAYER
        cm = FacePoolMonkeyPatch(model, layer=L11_LAYER)
    else:
        cm = nullcontext()

    log.info("[%s] running inference: %d frames batch=%d num_workers=%d face_pool=%s",
             ckpt_key, n, BATCH_SIZE, NUM_WORKERS, use_face_pool)
    t0 = time.time()
    with torch.inference_mode():
        with cm:
            for batch_idx, (images, indices) in enumerate(loader):
                images = images.to(device)
                out = model({"image": images}, inference=True)
                if isinstance(out, dict) and "prob" in out:
                    p = out["prob"].detach().cpu().numpy().reshape(-1)
                else:
                    if isinstance(out, dict):
                        logits = out.get("cls") or out.get("raw_logits") or out.get("logits")
                    else:
                        logits = out
                    if logits is None:
                        raise RuntimeError(
                            f"could not extract logits/prob from {list(out.keys()) if isinstance(out, dict) else type(out)}"
                        )
                    if logits.dim() == 3:
                        logits = logits.mean(dim=1)
                    p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
                for i, gi in enumerate(indices.numpy()):
                    probs[int(gi)] = float(p[i])

                if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(loader):
                    elapsed = time.time() - t0
                    done = min((batch_idx + 1) * BATCH_SIZE, n)
                    log.info("[%s] batch %d/%d (~%d/%d) elapsed=%.1fs (%.1f fps)",
                             ckpt_key, batch_idx + 1, len(loader),
                             done, n, elapsed, done / max(elapsed, 1e-3))

    log.info("[%s] inference done in %.1fs", ckpt_key, time.time() - t0)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["population", "frame_path", "frame_basename", "prob_fake"])
        for (pop, path), prob in zip(frames, probs):
            w.writerow([pop, str(path), path.name, f"{prob:.6f}"])
    log.info("[%s] wrote -> %s", ckpt_key, out_csv)

    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return out_csv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated ckpt keys; default = all 5")
    ap.add_argument("--device", default="auto", help="cpu | mps | cuda | auto")
    args = ap.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log.info("device=%s", device)

    frames = collect_frames()
    if not frames:
        log.error("no frames found at %s", RAW_DIR)
        return 2

    keys_to_run = list(CKPTS.keys())
    if args.only:
        keys_to_run = [k.strip() for k in args.only.split(",")]

    for k in keys_to_run:
        if k not in CKPTS:
            log.warning("unknown ckpt key: %s — skipping", k)
            continue
        path, use_face_pool = CKPTS[k]
        try:
            score_one_ckpt(k, path, use_face_pool, frames, device)
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] FAILED: %s", k, exc, exc_info=True)

    log.info("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
