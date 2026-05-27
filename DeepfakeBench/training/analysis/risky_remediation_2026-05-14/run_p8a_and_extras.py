"""Cross-check on P8A + add 2 more remediation candidates (CLAHE + blend).

Reuses the same pool as the T5C run for direct comparison.
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
log = logging.getLogger("p8a_extras")

# P8A ckpt — find it
P8A_CANDIDATES = [
    REPO_ROOT / "analysis/r13_overnight_atlas_2026-05-13/_cache",
    REPO_ROOT / "analysis/team_sanity_check_2026-05-05",
    REPO_ROOT / "analysis/cpu_followups_2026-05-04",
]

# Search for P8A
def find_p8a():
    for root in P8A_CANDIDATES:
        if not root.exists():
            continue
        for p in root.rglob("*p8a*step5000*.pth"):
            return p
        for p in root.rglob("*p8a*5000*.pth"):
            return p
    # Broader search
    for p in REPO_ROOT.rglob("*p8a_reference_step5000*.pth"):
        return p
    return None

P8A_CKPT = find_p8a()
T5C_CKPT = REPO_ROOT / "analysis/r13_overnight_may6_retest_2026-05-13/_ckpt_cache/periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth"

OUT_DIR = THIS_DIR / "outputs"
DETECTOR_CONFIG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO_ROOT / "config/defaults.yaml"

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
RES = 224

# ============================================================================
# Reuse pool from previous run
# ============================================================================


def load_pool():
    pool = pd.read_csv(OUT_DIR / "scores_per_frame.csv")
    return pool


# ============================================================================
# Remediations (incl. 2 new candidates)
# ============================================================================


def rem_orig(img):
    return img


def rem_unsharp_05(img, amount=0.5, ksize=5, sigma=1.0):
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def rem_clahe(img):
    """CLAHE on the L channel of Lab — local contrast enhancement without global shift."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)


def rem_blend_unsharp(img, weight=0.5):
    """Blend original with mildly-sharpened: img = w*sharp + (1-w)*orig."""
    sharp = rem_unsharp_05(img, amount=0.5)
    return cv2.addWeighted(sharp, weight, img, 1.0 - weight, 0)


REMEDIATIONS = OrderedDict([
    ("orig", rem_orig),
    ("unsharp_05", rem_unsharp_05),
    ("clahe", rem_clahe),
    ("blend_unsharp_05", rem_blend_unsharp),
])


# ============================================================================
# Dataset + model
# ============================================================================


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
    loader = DataLoader(ds, batch_size=8, num_workers=2)
    probs = np.zeros(len(paths), dtype=np.float32)
    t0 = time.time()
    for images, indices in loader:
        with torch.inference_mode():
            out = model({"image": images.to(DEVICE)}, inference=True)["prob"]
            out = out.detach().cpu().numpy().reshape(-1)
        for i, gi in enumerate(indices.numpy()):
            probs[int(gi)] = float(out[i])
    log.info(f"  [{name}] {len(paths)} frames in {time.time()-t0:.1f}s")
    return probs


# ============================================================================
# Main
# ============================================================================


def run_ckpt(model_alias, ckpt_path, pool, out_csv_name):
    log.info(f"=== {model_alias} ===")
    log.info(f"ckpt: {ckpt_path}")
    model = load_model(ckpt_path)
    paths = pool["local"].tolist()
    out = pool.copy()
    for cond, fn in REMEDIATIONS.items():
        out[f"{model_alias}_score_{cond}"] = score(model, paths, fn, f"{model_alias}/{cond}")
    out.to_csv(OUT_DIR / out_csv_name, index=False)
    log.info(f"wrote -> {OUT_DIR / out_csv_name}")
    return out


def summarize(df, alias):
    """Compute AUC + per-pool deltas under each condition."""
    real = df[df["pool"].isin(["risky_real", "clean_real"])]
    risky = df[df["pool"] == "risky_real"]
    fake = df[df["pool"] == "tp_fake"]
    mixed = df[df["pool"].isin(["risky_real", "clean_real", "tp_fake"])]
    hard = df[df["pool"].isin(["risky_real", "tp_fake"])]
    orig_col = f"{alias}_score_orig"
    print()
    print("=" * 78)
    print(f"== {alias}")
    print("=" * 78)
    print(f"{'condition':<20} {'AUC(mix)':<10} {'AUC(hard)':<10} {'Δrisky':<10} {'Δclean':<10} {'Δtpfake':<10} {'risky<.5':<10} {'tp>.5':<10}")
    for cond in REMEDIATIONS:
        col = f"{alias}_score_{cond}"
        if cond == "orig":
            print(f"{cond:<20} "
                  f"{roc_auc_score(mixed['label'], mixed[col]):<10.4f} "
                  f"{roc_auc_score(hard['label'], hard[col]):<10.4f} "
                  f"{'0.0000':<10} {'0.0000':<10} {'0.0000':<10} "
                  f"{float((risky[col] < 0.5).mean()):<10.4f} "
                  f"{float((fake[col] > 0.5).mean()):<10.4f}")
            continue
        d_risky = float((risky[col] - risky[orig_col]).mean())
        d_clean = float((df[df['pool']=='clean_real'][col] - df[df['pool']=='clean_real'][orig_col]).mean())
        d_fake = float((fake[col] - fake[orig_col]).mean())
        print(f"{cond:<20} "
              f"{roc_auc_score(mixed['label'], mixed[col]):<10.4f} "
              f"{roc_auc_score(hard['label'], hard[col]):<10.4f} "
              f"{d_risky:<10.4f} {d_clean:<10.4f} {d_fake:<10.4f} "
              f"{float((risky[col] < 0.5).mean()):<10.4f} "
              f"{float((fake[col] > 0.5).mean()):<10.4f}")
    return df


def main():
    pool = load_pool()
    log.info(f"pool: {len(pool)} frames")

    # Run T5C with the new conditions (orig + unsharp_05 already in CSV but recompute for parity)
    t5c = run_ckpt("T5C", T5C_CKPT, pool, "p8a_extras_t5c.csv")
    summarize(t5c, "T5C")

    if P8A_CKPT is None:
        log.warning("P8A ckpt NOT FOUND locally — skipping P8A cross-check")
        log.warning("Expected somewhere with name *p8a*step5000*.pth")
        return
    log.info(f"\np8a ckpt found: {P8A_CKPT}")
    p8a = run_ckpt("P8A", P8A_CKPT, pool, "p8a_extras_p8a.csv")
    summarize(p8a, "P8A")


if __name__ == "__main__":
    main()
