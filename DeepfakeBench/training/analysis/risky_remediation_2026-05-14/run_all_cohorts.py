"""Full cross-substrate inference: T5C step3500 + P8A step5000, orig + blend@0.50.

Pool: all 13852 frames in grouped_manifest_v2.csv that have local files.
Covers 12+ cohorts: in-sample dev, lockbox, live_prod (reals & fakes),
visomaster_v2, dor_evening/morning, dor_fake_local, team_sanity_may5, etc.

Also computes cheap IQ features (lap_var, luma, color_a/b_dev) per frame —
needed for Design A (selective) + Design C (soft gate).

Expected runtime: ~30 min on M2 mac CPU.
Output: outputs/all_cohorts_scored.csv (one row per frame, all conditions + IQ features).
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
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arena.model_arena import CLIP_MEAN, CLIP_STD  # noqa: E402
from detectors import DETECTOR  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("all_cohorts")

T5C_CKPT = REPO_ROOT / "analysis/r13_overnight_may6_retest_2026-05-13/_ckpt_cache/periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth"
P8A_CKPT = REPO_ROOT / "analysis/p2_eval_2026-05-08/d1_d4_cpu/ckpts/p8a_step5000.pth"
MANIFEST = REPO_ROOT / "analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv"
FRAMES_ROOT = REPO_ROOT / "analysis/identity_browser_2026-05-05/frames"

OUT_DIR = THIS_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DETECTOR_CONFIG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO_ROOT / "config/defaults.yaml"

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
RES = 224


def local_path(row):
    src_name = Path(row["frame_path"]).name
    return FRAMES_ROOT / row["base_identity"] / f"{row['suite']}__{src_name}"


def blend_unsharp(img, w=0.5):
    sharp = cv2.addWeighted(img, 1.5, cv2.GaussianBlur(img, (5, 5), 1.0), -0.5, 0)
    return cv2.addWeighted(sharp, w, img, 1.0 - w, 0)


def rem_orig(img): return img
def rem_blend_050(img): return blend_unsharp(img, 0.5)


REMEDIATIONS = OrderedDict([("orig", rem_orig), ("blend_050", rem_blend_050)])


def iq_features(img):
    """Compute pixel-domain features on a 224x224 BGR image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    luma = float(np.mean(gray))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_dev = float(np.mean(np.abs(lab[..., 1] - 128)))
    b_dev = float(np.mean(np.abs(lab[..., 2] - 128)))
    edge = cv2.Canny(gray, 50, 150)
    edge_density = float(edge.sum() / (255.0 * gray.size))
    return {
        "iq_lap_var": lap_var, "iq_luma": luma,
        "iq_lab_a_dev": a_dev, "iq_lab_b_dev": b_dev,
        "iq_edge_density": edge_density,
    }


class FrameDataset(Dataset):
    """Loads frames + applies remediation + returns tensor."""
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
    ds = FrameDataset(paths, remediation)
    loader = DataLoader(ds, batch_size=8, num_workers=0)
    probs = np.zeros(len(paths), dtype=np.float32)
    t0 = time.time()
    n_batches = 0
    for images, indices in loader:
        with torch.inference_mode():
            out = model({"image": images.to(DEVICE)}, inference=True)["prob"]
            out = out.detach().cpu().numpy().reshape(-1)
        for i, gi in enumerate(indices.numpy()):
            probs[int(gi)] = float(out[i])
        n_batches += 1
        if n_batches % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / n_batches * (len(paths) // 8 - n_batches)
            log.info(f"    [{name}] {n_batches*8}/{len(paths)} ({elapsed:.0f}s, ETA {eta:.0f}s)")
    log.info(f"  [{name}] DONE {len(paths)} in {time.time()-t0:.1f}s")
    return probs


def main():
    log.info("loading manifest...")
    m = pd.read_csv(MANIFEST)
    m["local"] = m.apply(local_path, axis=1).astype(str)
    m["exists"] = m["local"].apply(lambda p: Path(p).exists())
    log.info(f"manifest: {len(m)} total; {m['exists'].sum()} have local files")
    pool = m[m["exists"]].reset_index(drop=True)

    log.info("\nCohort breakdown (n by suite × label):")
    print(pool.groupby(["suite", "label"]).size().to_string())

    paths = pool["local"].tolist()

    # IQ features (one-pass)
    log.info(f"\ncomputing IQ features on {len(paths)} frames...")
    iq_rows = []
    t0 = time.time()
    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            iq_rows.append({k: float("nan") for k in
                            ["iq_lap_var", "iq_luma", "iq_lab_a_dev",
                             "iq_lab_b_dev", "iq_edge_density"]})
            continue
        img = cv2.resize(img, (RES, RES), interpolation=cv2.INTER_LINEAR)
        iq_rows.append(iq_features(img))
        if (i+1) % 2000 == 0:
            log.info(f"  IQ: {i+1}/{len(paths)} ({time.time()-t0:.0f}s)")
    iq_df = pd.DataFrame(iq_rows)
    for c in iq_df.columns:
        pool[c] = iq_df[c].values
    log.info(f"IQ features added in {time.time()-t0:.1f}s")

    # T5C
    log.info("\n=== T5C step3500 ===")
    model = load_model(T5C_CKPT)
    for cond, fn in REMEDIATIONS.items():
        pool[f"T5C_{cond}"] = score(model, paths, fn, f"T5C/{cond}")
    pool.to_csv(OUT_DIR / "all_cohorts_t5c.csv", index=False)

    # P8A
    log.info("\n=== P8A step5000 ===")
    del model
    model = load_model(P8A_CKPT)
    for cond, fn in REMEDIATIONS.items():
        pool[f"P8A_{cond}"] = score(model, paths, fn, f"P8A/{cond}")
    pool.to_csv(OUT_DIR / "all_cohorts_scored.csv", index=False)
    log.info(f"\nwrote -> {OUT_DIR / 'all_cohorts_scored.csv'}  ({len(pool)} rows)")


if __name__ == "__main__":
    main()
