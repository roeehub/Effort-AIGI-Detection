"""Cross-cohort drift eval — run T5C step3500 + P2D step3000 on
team_sanity_check_2026-05-05 frames.

The team_sanity_check_2026-05-05 panel scored P8A, E2B, E3, PA, PC across 210
local frames (Dor + Noyn + Roee + Xiang + Xinhe). T5C and P2D are missing.

This script adapts run_t5c_step3500_may6.py to the team-sanity frames panel.
Both ckpts already cached locally (downloaded earlier this session).

Usage (run after wakeup; ~5 min per ckpt on MPS):
  python analysis/r13_overnight_may6_retest_2026-05-13/run_t5c_p2d_team_sanity.py
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
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arena.model_arena import CLIP_MEAN, CLIP_STD  # noqa: E402
from detectors import DETECTOR  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("team_sanity_t5c_p2d")

TEAM_SANITY_DIR = REPO_ROOT / "analysis/team_sanity_check_2026-05-05/frames"
OUT_DIR = THIS_DIR / "outputs"
DETECTOR_CONFIG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO_ROOT / "config/defaults.yaml"

CKPT_CACHE = THIS_DIR / "_ckpt_cache"

CKPTS = {
    "T5C_PERIODIC_STEP3500": CKPT_CACHE / "periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth",
    "P2D_FOURIER_PERIODIC_STEP3000": CKPT_CACHE / "periodic_effort_20260507_step3000_auc0.9738_eer0.0823.pth",
}

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class FrameDataset(Dataset):
    def __init__(self, paths, resolution=224):
        self.paths = list(paths)
        self.resolution = resolution
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return torch.zeros(3, self.resolution, self.resolution), idx
        img_bgr = cv2.resize(img_bgr, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        return self.transform(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)), idx


def collect_frames():
    paths = []
    for d in sorted(TEAM_SANITY_DIR.iterdir()):
        if d.is_dir():
            for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
                paths.append((str(p), d.name))
    log.info(f"collected {len(paths)} frames across {len(set(p[1] for p in paths))} identities")
    return paths


def load_model(ckpt_path):
    import yaml
    with open(DETECTOR_CONFIG, "r") as f: cfg = yaml.safe_load(f)
    with open(TRAIN_CONFIG, "r") as f: cfg.update(yaml.safe_load(f))
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model_cfg = ckpt.get("model_config", {})
    for k, v in model_cfg.items():
        if k != "current_arcface_s": cfg[k] = v
    cfg["multi_axis_grl"] = {"enabled": False}
    model = DETECTOR[cfg["model_name"]](cfg).to(DEVICE)
    if model_cfg.get("use_arcface_head") and "current_arcface_s" in model_cfg:
        if hasattr(model.head, "s"):
            model.head.s.data.fill_(model_cfg["current_arcface_s"])
    clean = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
    missing, unexpected = model.load_state_dict(clean, strict=False)
    log.info(f"  load: {len(missing)} missing, {len(unexpected)} unexpected")
    model.eval()
    return model


def score_ckpt(alias, ckpt_path, frames):
    if not ckpt_path.exists():
        log.error(f"missing ckpt: {ckpt_path}"); return None
    out_csv = OUT_DIR / f"team_sanity_scores_{alias}.csv"
    log.info(f"loading {alias} on {DEVICE}...")
    model = load_model(ckpt_path)
    log.info("loaded")
    ds = FrameDataset([p[0] for p in frames])
    loader = DataLoader(ds, batch_size=8, num_workers=2, shuffle=False)
    probs = np.zeros(len(frames), dtype=np.float32)
    t0 = time.time()
    for images, indices in loader:
        with torch.inference_mode():
            out = model({"image": images.to(DEVICE)}, inference=True)["prob"].detach().cpu().numpy().reshape(-1)
        for i, gi in enumerate(indices.numpy()):
            probs[int(gi)] = float(out[i])
    log.info(f"  inference done in {time.time()-t0:.1f}s")
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_path", "identity", "frame_prob"])
        for (path, ident), p in zip(frames, probs):
            w.writerow([path, ident, f"{p:.6f}"])
    log.info(f"wrote -> {out_csv}")
    return out_csv


def summarize():
    print()
    print("=== Team-sanity per-identity scores (frac > 0.50) ===")
    for alias in CKPTS:
        out = OUT_DIR / f"team_sanity_scores_{alias}.csv"
        if not out.exists():
            print(f"  {alias}: NO DATA"); continue
        df = pd.read_csv(out)
        df["frame_prob"] = pd.to_numeric(df["frame_prob"], errors="coerce")
        per_id = df.groupby("identity")["frame_prob"].apply(
            lambda s: (s > 0.50).mean()).round(3)
        print(f"  {alias}: {dict(per_id)}")


def main():
    frames = collect_frames()
    for alias, ckpt in CKPTS.items():
        score_ckpt(alias, ckpt, frames)
    summarize()


if __name__ == "__main__":
    main()
