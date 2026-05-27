"""P8A cross-check + blend-ratio sweep on T5C.

Blend ratios to test: 0.25, 0.35, 0.50, 0.65, 0.75 (= weight on unsharp)
The 0.50 was the winner in the previous run. Sweep to find sweet spot.
"""
from __future__ import annotations

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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arena.model_arena import CLIP_MEAN, CLIP_STD  # noqa: E402
from detectors import DETECTOR  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("blend_sweep")

T5C_CKPT = REPO_ROOT / "analysis/r13_overnight_may6_retest_2026-05-13/_ckpt_cache/periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth"
P8A_CKPT = REPO_ROOT / "analysis/p2_eval_2026-05-08/d1_d4_cpu/ckpts/p8a_step5000.pth"

OUT_DIR = THIS_DIR / "outputs"
DETECTOR_CONFIG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO_ROOT / "config/defaults.yaml"

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
RES = 224


def make_blend(weight):
    def _f(img):
        sharp = cv2.addWeighted(img, 1.5, cv2.GaussianBlur(img, (5, 5), 1.0), -0.5, 0)
        return cv2.addWeighted(sharp, weight, img, 1.0 - weight, 0)
    return _f


def rem_orig(img):
    return img


REMEDIATIONS = OrderedDict([
    ("orig", rem_orig),
    ("blend_025", make_blend(0.25)),
    ("blend_035", make_blend(0.35)),
    ("blend_050", make_blend(0.50)),
    ("blend_065", make_blend(0.65)),
    ("blend_075", make_blend(0.75)),
])


class RemDataset(Dataset):
    def __init__(self, paths, remediation):
        self.paths = paths
        self.remediation = remediation
        self.transform = T.Compose([T.ToTensor(),
                                     T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]), cv2.IMREAD_COLOR)
        if img is None:
            return torch.zeros(3, RES, RES), idx
        img = cv2.resize(img, (RES, RES), interpolation=cv2.INTER_LINEAR)
        img = self.remediation(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), idx


def load_model(ckpt_path):
    import yaml
    with open(DETECTOR_CONFIG) as f: cfg = yaml.safe_load(f)
    with open(TRAIN_CONFIG) as f: cfg.update(yaml.safe_load(f))
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


def score(model, paths, remediation, name):
    ds = RemDataset(paths, remediation)
    loader = DataLoader(ds, batch_size=8, num_workers=0)
    probs = np.zeros(len(paths), dtype=np.float32)
    t0 = time.time()
    for images, indices in loader:
        with torch.inference_mode():
            out = model({"image": images.to(DEVICE)}, inference=True)["prob"]
            out = out.detach().cpu().numpy().reshape(-1)
        for i, gi in enumerate(indices.numpy()):
            probs[int(gi)] = float(out[i])
    log.info(f"  [{name}] {len(paths)} in {time.time()-t0:.1f}s")
    return probs


def run_ckpt(alias, ckpt_path, pool, out_name):
    log.info(f"=== {alias} ===")
    model = load_model(ckpt_path)
    paths = pool["local"].tolist()
    out = pool.copy()
    for cond, fn in REMEDIATIONS.items():
        out[f"{alias}_{cond}"] = score(model, paths, fn, f"{alias}/{cond}")
    out.to_csv(OUT_DIR / out_name, index=False)
    return out


def summarize(df, alias):
    risky = df[df["pool"] == "risky_real"]
    clean = df[df["pool"] == "clean_real"]
    fake = df[df["pool"] == "tp_fake"]
    mixed = df[df["pool"].isin(["risky_real", "clean_real", "tp_fake"])]
    hard = df[df["pool"].isin(["risky_real", "tp_fake"])]
    orig_col = f"{alias}_orig"
    print()
    print("=" * 86)
    print(f"== {alias}  (n: risky={len(risky)} clean={len(clean)} fake={len(fake)})")
    print("=" * 86)
    hdr = f"{'condition':<14} {'AUC(mix)':<10} {'ΔAUC':<10} {'AUC(hard)':<11} {'ΔAUC_h':<10} {'Δrisky':<10} {'Δclean':<10} {'Δfake':<10}"
    print(hdr)
    auc_orig = roc_auc_score(mixed["label"], mixed[orig_col])
    auc_orig_h = roc_auc_score(hard["label"], hard[orig_col])
    for cond in REMEDIATIONS:
        col = f"{alias}_{cond}"
        auc_m = roc_auc_score(mixed["label"], mixed[col])
        auc_h = roc_auc_score(hard["label"], hard[col])
        d_risky = (risky[col] - risky[orig_col]).mean()
        d_clean = (clean[col] - clean[orig_col]).mean()
        d_fake = (fake[col] - fake[orig_col]).mean()
        mark = "✓" if (auc_m > auc_orig) else "·"
        print(f"{cond:<14} {auc_m:<10.4f} {auc_m-auc_orig:<+10.4f} "
              f"{auc_h:<11.4f} {auc_h-auc_orig_h:<+10.4f} "
              f"{d_risky:<+10.4f} {d_clean:<+10.4f} {d_fake:<+10.4f} {mark}")


def main():
    pool = pd.read_csv(OUT_DIR / "scores_per_frame.csv")
    log.info(f"pool: {len(pool)} frames")

    t5c = run_ckpt("T5C", T5C_CKPT, pool, "blend_sweep_t5c.csv")
    summarize(t5c, "T5C")

    if not P8A_CKPT.exists():
        log.error(f"P8A NOT FOUND at {P8A_CKPT}")
        return
    p8a = run_ckpt("P8A", P8A_CKPT, pool, "blend_sweep_p8a.csv")
    summarize(p8a, "P8A")


if __name__ == "__main__":
    main()
