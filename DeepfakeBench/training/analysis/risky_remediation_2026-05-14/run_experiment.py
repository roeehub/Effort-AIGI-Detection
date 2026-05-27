"""Risky-frame detection + cheap remediation experiment.

Hypothesis
----------
Cheap, deterministic pre-processing transforms (gray-world white-balance,
unsharp mask, gamma normalize) can shift the score of false-positive reals
toward the correct (low-score) region without dragging true-positive fakes
equally. If real-side score drop > fake-side score drop, the mixed-pool AUC
goes up and we have a real lever (not just a tau shift).

Pool
----
- 100 risky_real   (label=0, T5C step3500 > 0.7, gate=pass, chronic ids)
-  80 clean_real   (label=0, T5C step3500 < 0.2, gate=pass, non-chronic ids)
-  80 tp_fake      (label=1, T5C step3500 > 0.7, gate=pass)

Conditions
----------
- orig:         identity transform
- wb_grayworld: gray-world white-balance correction
- unsharp_05:   unsharp mask, amount=0.5
- gamma_norm:   gamma adjusted to push luma median toward 0.5

Outputs
-------
- outputs/scores_per_frame.csv   (one row per frame, all conditions)
- outputs/pool.csv               (selected pool with manifest info)
- outputs/summary.json           (AUC + per-group score deltas)

Usage
-----
  python analysis/risky_remediation_2026-05-14/run_experiment.py
"""
from __future__ import annotations

import csv
import json
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
log = logging.getLogger("risky_remed")

MANIFEST = REPO_ROOT / "analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv"
FRAMES_ROOT = REPO_ROOT / "analysis/identity_browser_2026-05-05/frames"
CKPT = REPO_ROOT / "analysis/r13_overnight_may6_retest_2026-05-13/_ckpt_cache/periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth"

OUT_DIR = THIS_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DETECTOR_CONFIG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO_ROOT / "config/defaults.yaml"

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
RES = 224
SEED = 9501

# ============================================================================
# Pool selection
# ============================================================================


def local_path(row) -> Path:
    src_name = Path(row["frame_path"]).name
    return FRAMES_ROOT / row["base_identity"] / f"{row['suite']}__{src_name}"


def build_pool(n_risky=100, n_clean=80, n_tp_fake=80):
    rng = np.random.default_rng(SEED)
    m = pd.read_csv(MANIFEST)
    m = m[m["gate_status"] == "pass"].copy()

    # Risky reals: label=0, T5C>0.7
    risky = m[(m["label"] == 0) & (m["score_T5C"] > 0.7)].copy()
    risky["pool"] = "risky_real"
    if len(risky) > n_risky:
        # Stratified by base_identity, sample proportional to identity count
        risky = risky.sample(n=n_risky, random_state=SEED)

    # Clean reals: label=0, T5C<0.2
    clean = m[(m["label"] == 0) & (m["score_T5C"] < 0.2)].copy()
    clean["pool"] = "clean_real"
    # Stratify across identities — pick at most 10 per identity
    clean_groups = clean.groupby("base_identity", group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), 10), random_state=SEED)
    )
    if len(clean_groups) > n_clean:
        clean_groups = clean_groups.sample(n=n_clean, random_state=SEED)
    clean = clean_groups

    # TP fakes: label=1, T5C>0.7
    tp_fake = m[(m["label"] == 1) & (m["score_T5C"] > 0.7)].copy()
    tp_fake["pool"] = "tp_fake"
    # Stratify across base_identity — pick at most 10 per identity
    fake_groups = tp_fake.groupby("base_identity", group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), 10), random_state=SEED)
    )
    if len(fake_groups) > n_tp_fake:
        fake_groups = fake_groups.sample(n=n_tp_fake, random_state=SEED)
    tp_fake = fake_groups

    pool = pd.concat([risky, clean, tp_fake], ignore_index=True)
    pool["local"] = pool.apply(local_path, axis=1).astype(str)
    pool["exists"] = pool["local"].apply(lambda p: Path(p).exists())
    log.info(f"pool: risky={len(risky)} clean={len(clean)} tp_fake={len(tp_fake)} "
             f"total={len(pool)} all_exist={pool['exists'].all()}")
    pool.to_csv(OUT_DIR / "pool.csv", index=False)
    return pool[pool["exists"]].reset_index(drop=True)


# ============================================================================
# IQ features (per frame)
# ============================================================================


def iq_features(img_bgr):
    """Return dict of pixel-domain statistics computed on the input BGR image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lum = float(np.mean(gray))
    # Lab color space — a* and b* deviations from neutral (a=0, b=0)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_dev = float(np.mean(np.abs(lab[..., 1] - 128)))
    b_dev = float(np.mean(np.abs(lab[..., 2] - 128)))
    h, w = gray.shape[:2]
    return {
        "lap_var": lap_var,
        "luma_mean": lum,
        "lab_a_dev": a_dev,
        "lab_b_dev": b_dev,
        "h": h,
        "w": w,
        "min_dim": min(h, w),
    }


# ============================================================================
# Remediation transforms (in-place on BGR uint8 images)
# ============================================================================


def remediate_orig(img_bgr):
    return img_bgr


def remediate_wb_grayworld(img_bgr):
    """Gray-world white balance: scale each channel so its mean equals the
    grand mean (target gray = 128). Caps multipliers to avoid pathological
    rescales on saturated images."""
    img = img_bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)  # B, G, R
    target = float(means.mean())
    scales = target / np.clip(means, 1e-6, None)
    scales = np.clip(scales, 0.5, 2.0)
    img = img * scales[None, None, :]
    return np.clip(img, 0, 255).astype(np.uint8)


def remediate_unsharp_05(img_bgr, amount=0.5, ksize=5, sigma=1.0):
    """Mild unsharp mask: img + amount * (img - blur(img))."""
    blurred = cv2.GaussianBlur(img_bgr, (ksize, ksize), sigma)
    sharp = cv2.addWeighted(img_bgr, 1.0 + amount, blurred, -amount, 0)
    return sharp


def remediate_gamma_norm(img_bgr, target_luma=0.5):
    """Gamma adjusted so luminance median moves toward target_luma * 255."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    med = float(np.median(gray))
    med = max(med, 1e-3)
    # gamma s.t. med^gamma = target  ->  gamma = log(target)/log(med)
    gamma = float(np.log(max(target_luma, 1e-3)) / np.log(med))
    gamma = float(np.clip(gamma, 0.5, 2.0))
    img = (img_bgr.astype(np.float32) / 255.0) ** gamma
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


REMEDIATIONS = OrderedDict([
    ("orig", remediate_orig),
    ("wb_grayworld", remediate_wb_grayworld),
    ("unsharp_05", remediate_unsharp_05),
    ("gamma_norm", remediate_gamma_norm),
])


# ============================================================================
# Dataset: produces (image, idx) batches under a chosen remediation
# ============================================================================


class RemediationFrameDataset(Dataset):
    def __init__(self, paths, remediation_fn):
        self.paths = list(paths)
        self.remediation = remediation_fn
        self.transform = T.Compose([
            T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]), cv2.IMREAD_COLOR)
        if img is None:
            return torch.zeros(3, RES, RES), idx
        # Resize FIRST, then remediate at training resolution (closer to deploy semantics)
        img = cv2.resize(img, (RES, RES), interpolation=cv2.INTER_LINEAR)
        img = self.remediation(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), idx


# ============================================================================
# Model loading + scoring
# ============================================================================


def load_model(ckpt_path):
    import yaml
    with open(DETECTOR_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    with open(TRAIN_CONFIG, "r") as f:
        cfg.update(yaml.safe_load(f))
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model_cfg = ckpt.get("model_config", {})
    for k, v in model_cfg.items():
        if k != "current_arcface_s":
            cfg[k] = v
    cfg["multi_axis_grl"] = {"enabled": False}
    model = DETECTOR[cfg["model_name"]](cfg).to(DEVICE)
    if model_cfg.get("use_arcface_head") and "current_arcface_s" in model_cfg:
        if hasattr(model.head, "s"):
            model.head.s.data.fill_(model_cfg["current_arcface_s"])
    clean = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
    missing, unexpected = model.load_state_dict(clean, strict=False)
    log.info(f"  load: {len(missing)} missing, {len(unexpected)} unexpected keys")
    model.eval()
    return model


def score_condition(model, paths, remediation_fn, condition_name, batch_size=8):
    ds = RemediationFrameDataset(paths, remediation_fn)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2, shuffle=False)
    probs = np.zeros(len(paths), dtype=np.float32)
    t0 = time.time()
    for images, indices in loader:
        with torch.inference_mode():
            out = model({"image": images.to(DEVICE)}, inference=True)["prob"]
            out = out.detach().cpu().numpy().reshape(-1)
        for i, gi in enumerate(indices.numpy()):
            probs[int(gi)] = float(out[i])
    log.info(f"  [{condition_name}] scored {len(paths)} frames in {time.time()-t0:.1f}s")
    return probs


# ============================================================================
# Analysis
# ============================================================================


def analyze(pool, score_cols):
    """Compute mixed-pool AUC + per-pool score deltas for each condition."""
    real_mask = pool["label"] == 0
    fake_mask = pool["label"] == 1
    risky_mask = pool["pool"] == "risky_real"
    clean_mask = pool["pool"] == "clean_real"
    tp_mask = pool["pool"] == "tp_fake"

    summary = {"n_risky": int(risky_mask.sum()), "n_clean": int(clean_mask.sum()),
               "n_tp_fake": int(tp_mask.sum()), "conditions": {}}
    orig_scores = pool["score_orig"].values

    for col in score_cols:
        cond = col.replace("score_", "")
        s = pool[col].values
        # AUC on real-vs-fake mixed pool
        try:
            auc_all = float(roc_auc_score(pool["label"].values, s))
        except Exception:
            auc_all = float("nan")
        # AUC on (risky_real + tp_fake) only — the "high-conf misfire vs
        # correct-fake" pool
        sub_mask = risky_mask | tp_mask
        try:
            auc_hard = float(roc_auc_score(pool.loc[sub_mask, "label"].values,
                                            s[sub_mask]))
        except Exception:
            auc_hard = float("nan")
        # Score deltas vs orig
        delta = s - orig_scores
        summary["conditions"][cond] = {
            "auc_all": auc_all,
            "auc_hard": auc_hard,
            "delta_risky_real_mean": float(np.mean(delta[risky_mask])),
            "delta_risky_real_median": float(np.median(delta[risky_mask])),
            "delta_clean_real_mean": float(np.mean(delta[clean_mask])),
            "delta_tp_fake_mean": float(np.mean(delta[tp_mask])),
            "delta_tp_fake_median": float(np.median(delta[tp_mask])),
            "risky_below_50_frac": float(np.mean(s[risky_mask] < 0.5)),
            "tp_fake_above_50_frac": float(np.mean(s[tp_mask] > 0.5)),
        }
    return summary


# ============================================================================
# Main
# ============================================================================


def main():
    log.info(f"device: {DEVICE}")
    log.info(f"ckpt: {CKPT}")
    pool = build_pool()
    log.info(f"pool size (existing files): {len(pool)}")

    # Compute IQ features on originals (before any remediation)
    log.info("computing IQ features on originals...")
    iq_rows = []
    for path in pool["local"]:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            iq_rows.append({k: float("nan") for k in
                            ["lap_var", "luma_mean", "lab_a_dev", "lab_b_dev",
                             "h", "w", "min_dim"]})
            continue
        # Resize to training res before measuring — matches what model sees
        img = cv2.resize(img, (RES, RES), interpolation=cv2.INTER_LINEAR)
        iq_rows.append(iq_features(img))
    iq_df = pd.DataFrame(iq_rows)
    for c in iq_df.columns:
        pool[f"iq_{c}"] = iq_df[c].values
    log.info(f"IQ features added: {list(iq_df.columns)}")

    # Load model once
    log.info("loading T5C step3500...")
    model = load_model(CKPT)

    # Score each condition
    paths = pool["local"].tolist()
    for cond_name, fn in REMEDIATIONS.items():
        col = f"score_{cond_name}"
        log.info(f"scoring under condition: {cond_name}")
        pool[col] = score_condition(model, paths, fn, cond_name)

    # Save per-frame CSV
    out_csv = OUT_DIR / "scores_per_frame.csv"
    pool.to_csv(out_csv, index=False)
    log.info(f"wrote per-frame scores -> {out_csv}")

    # Analyze
    score_cols = [f"score_{c}" for c in REMEDIATIONS]
    summary = analyze(pool, score_cols)
    out_json = OUT_DIR / "summary.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"wrote summary -> {out_json}")

    # Console headline
    print()
    print("=" * 70)
    print(f"POOL: risky_real={summary['n_risky']}  clean_real={summary['n_clean']}  tp_fake={summary['n_tp_fake']}")
    print("=" * 70)
    print(f"{'condition':<16} {'AUC(all)':<10} {'AUC(hard)':<10} {'Δrisky':<10} {'Δclean':<10} {'Δtpfake':<10} {'risky<0.5':<10} {'tp>0.5':<10}")
    for cond, s in summary["conditions"].items():
        print(f"{cond:<16} {s['auc_all']:<10.4f} {s['auc_hard']:<10.4f} "
              f"{s['delta_risky_real_mean']:<10.4f} {s['delta_clean_real_mean']:<10.4f} "
              f"{s['delta_tp_fake_mean']:<10.4f} {s['risky_below_50_frac']:<10.4f} "
              f"{s['tp_fake_above_50_frac']:<10.4f}")
    print()


if __name__ == "__main__":
    main()
