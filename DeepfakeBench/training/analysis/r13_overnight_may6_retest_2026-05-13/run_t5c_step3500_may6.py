"""may6 production-drift retest for T5C_PERIODIC_STEP3500.

Companion to run_may6_retest.py. Scores the 92 fresh real Xinhe-may6 frames
on T5C step3500 (jrlldtem job, R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.yaml) at
τ=0.5 using the same preprocessing + forward-pass path.

T5C step3500 architecture (relevant to inference):
  - vit_b_16_laion_datacomp (ViT-B-16-DataComp-XL, pretrained datacomp_xl_s13b_b90k)
  - apply_svd_to_in_proj=true, apply_svd_to_mlp=true (SVD residuals)
  - unfreeze_final_proj=true, unfreeze_final_ln=true
  - multi_axis_grl ENABLED at training (chronic_flag, is_dor,
    sharpness_laplacian_high, color_a_approx_dev_high) with hidden_dim=1024,
    bottleneck_dim=128 — but this block is NOT used at inference (the
    detector's `forward` only invokes self.multi_axis_grl_block when
    `not inference`, see detectors/effort_detector.py line 1843). The saved
    state_dict's `multi_axis_grl_block.*` keys are dropped via
    load_state_dict(strict=False), same handling as run_score_probe.py
    precedent (analysis/cpu_diagnostics_2026-05-11_t67_t5c_probe/).
  - Classifier head is nn.Linear(hidden_size=512, 2) — independent of the
    multi_axis_grl hidden_dim (which only affects the dropped aux block).

Outputs:
  outputs/scores_T5C_PERIODIC_STEP3500.csv   (frame-level scores, same schema
    as the precedent harness)
  Appends row to outputs/may6_retest_table.csv
  Updates MAY6_RETEST_FACTS_2026-05-13.md

Usage:
  python analysis/r13_overnight_may6_retest_2026-05-13/run_t5c_step3500_may6.py
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
log = logging.getLogger("t5c_step3500_may6")

CKPT_ALIAS = "T5C_PERIODIC_STEP3500"
CKPT_BASENAME = "periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth"
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
    """Identical preprocessing path to run_may6_retest.py::LocalFrameDataset."""

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


def load_t5c_model(ckpt_path: Path, device: torch.device):
    """Build EffortDetector matching the T5C T4-multi-axis-GRL recipe and load
    the trained state_dict. multi_axis_grl_block.* keys in the saved state are
    expected to be 'unexpected' at inference time (the block is not constructed
    here, since it isn't needed for the prob output)."""
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

    # Ensure multi_axis_grl is OFF at inference construction — saved
    # model_config does not include the multi_axis_grl block (see
    # trainer/trainer.py save_ckpt; only model_name/arcface/etc are saved
    # in model_config). The default detector init reads cfg.get('multi_axis_grl', {})
    # which is {} → use_multi_axis_grl=False. We assert that explicitly for safety.
    cfg["multi_axis_grl"] = {"enabled": False}

    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)
    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])

    clean_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    grl_unexpected = [k for k in unexpected if "multi_axis_grl_block" in k]
    other_unexpected = [k for k in unexpected if "multi_axis_grl_block" not in k]
    log.info(
        "  state_dict load: %d missing, %d unexpected "
        "(multi_axis_grl_block: %d, other: %d)",
        len(missing), len(unexpected), len(grl_unexpected), len(other_unexpected),
    )
    if other_unexpected:
        log.warning("  unexpected non-GRL keys (first 5): %s", other_unexpected[:5])
    if missing:
        log.warning("  missing keys (first 5): %s", missing[:5])
    model.eval()
    return model


def score_t5c():
    if not CKPT_LOCAL.exists():
        log.error("ckpt not present yet: %s", CKPT_LOCAL)
        return None

    out_csv = OUT_DIR / f"scores_{CKPT_ALIAS}.csv"
    if out_csv.exists():
        log.info("CSV exists, reusing: %s", out_csv)
        return out_csv

    frames = collect_may6_frames()
    if len(frames) != 92:
        log.warning("expected 92 may6 frames, found %d", len(frames))

    log.info("loading T5C step3500 on %s ...", DEVICE)
    t0 = time.time()
    model = load_t5c_model(CKPT_LOCAL, DEVICE)
    log.info("model loaded in %.1fs", time.time() - t0)

    ds = LocalFrameDataset(frames)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=False)

    n = len(frames)
    probs = np.zeros(n, dtype=np.float32)
    log.info("running inference: %d frames, batch=%d, device=%s",
             n, BATCH_SIZE, DEVICE)
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
        log.info("batch %d (~%d/%d) elapsed=%.1fs",
                 batch_idx + 1, min(done, n), n, elapsed)
    log.info("inference done in %.1fs", time.time() - t0)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["population", "frame_path", "frame_basename", "prob_fake"])
        for path, prob in zip(frames, probs):
            w.writerow(["may6_falseflag", str(path), path.name, f"{prob:.6f}"])
    log.info("wrote -> %s", out_csv)
    return out_csv


def summarize_and_agreement(scores_csv: Path):
    """Compute headline stats + agreement vs P8A; return dict."""
    df = pd.read_csv(scores_csv)
    df = df[df["population"] == "may6_falseflag"].reset_index(drop=True)
    probs = df["prob_fake"].to_numpy()
    n = len(probs)
    n_fired = int((probs > 0.5).sum())
    p50 = float(np.median(probs))
    p90 = float(np.quantile(probs, 0.9))
    p99 = float(np.quantile(probs, 0.99))
    mx = float(probs.max())

    # Agreement vs P8A
    p8a = pd.read_csv(P8A_CSV)
    p8a = p8a[p8a["population"] == "may6_falseflag"].reset_index(drop=True)
    merged = df.merge(p8a, on="frame_basename", suffixes=("_t5c", "_p8a"))
    assert len(merged) == n, f"join lost frames: {len(merged)} vs {n}"
    spearman_r, spearman_p = spearmanr(merged["prob_fake_t5c"], merged["prob_fake_p8a"])
    pearson_r, pearson_p = pearsonr(merged["prob_fake_t5c"], merged["prob_fake_p8a"])
    delta = merged["prob_fake_t5c"] - merged["prob_fake_p8a"]
    mean_delta = float(delta.mean())
    median_delta = float(delta.median())
    max_delta = float(delta.max())
    min_delta = float(delta.min())

    # Frame where the disagreement is the largest
    max_delta_row = merged.loc[delta.idxmax()]
    min_delta_row = merged.loc[delta.idxmin()]

    # Band counts
    band_counts = {}
    for low, high, label in [(0.0, 0.1, "[0.0,0.1)"),
                              (0.1, 0.3, "[0.1,0.3)"),
                              (0.3, 0.5, "[0.3,0.5)"),
                              (0.5, 0.7, "[0.5,0.7)"),
                              (0.7, 1.001, "[0.7,1.0]")]:
        band_counts[label] = int(((probs >= low) & (probs < high)).sum())

    p8a_fires = int((merged["prob_fake_p8a"] > 0.5).sum())
    t5c_fires = n_fired
    both_fire = int(((merged["prob_fake_t5c"] > 0.5) & (merged["prob_fake_p8a"] > 0.5)).sum())
    t5c_only = int(((merged["prob_fake_t5c"] > 0.5) & (merged["prob_fake_p8a"] <= 0.5)).sum())
    p8a_only = int(((merged["prob_fake_t5c"] <= 0.5) & (merged["prob_fake_p8a"] > 0.5)).sum())

    return {
        "n": n, "n_fired": n_fired, "p50": p50, "p90": p90, "p99": p99, "max": mx,
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "mean_delta": mean_delta, "median_delta": median_delta,
        "max_delta": max_delta, "min_delta": min_delta,
        "max_delta_frame": str(max_delta_row["frame_basename"]),
        "max_delta_p8a": float(max_delta_row["prob_fake_p8a"]),
        "max_delta_t5c": float(max_delta_row["prob_fake_t5c"]),
        "min_delta_frame": str(min_delta_row["frame_basename"]),
        "min_delta_p8a": float(min_delta_row["prob_fake_p8a"]),
        "min_delta_t5c": float(min_delta_row["prob_fake_t5c"]),
        "p8a_fires": p8a_fires, "t5c_fires": t5c_fires,
        "both_fire": both_fire, "t5c_only_fire": t5c_only, "p8a_only_fire": p8a_only,
        "band_counts": band_counts,
    }


def append_summary_row(summary: dict):
    """Append/update a T5C row in may6_retest_table.csv (preserve other rows)."""
    table_csv = OUT_DIR / "may6_retest_table.csv"
    df = pd.read_csv(table_csv) if table_csv.exists() else pd.DataFrame(
        columns=["ckpt", "source", "n", "n_fired@0.5", "p50", "p90", "p99", "max"]
    )
    df = df[df["ckpt"] != CKPT_ALIAS]  # drop prior T5C row, if any
    new_row = {
        "ckpt": CKPT_ALIAS,
        "source": "this retest (t5c followup)",
        "n": summary["n"],
        "n_fired@0.5": summary["n_fired"],
        "p50": summary["p50"],
        "p90": summary["p90"],
        "p99": summary["p99"],
        "max": summary["max"],
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(table_csv, index=False)
    log.info("updated -> %s", table_csv)


def main():
    log.info("=" * 70)
    log.info("may6 production-drift retest — T5C_PERIODIC_STEP3500")
    log.info("device=%s, batch=%d, workers=%d", DEVICE, BATCH_SIZE, NUM_WORKERS)
    log.info("=" * 70)

    if not CKPT_LOCAL.exists():
        log.error("ckpt missing: %s", CKPT_LOCAL)
        return 2

    scores_csv = score_t5c()
    if scores_csv is None:
        return 2

    summary = summarize_and_agreement(scores_csv)
    append_summary_row(summary)

    log.info("=" * 80)
    log.info("T5C step3500 may6 RESULT")
    log.info("=" * 80)
    log.info("  n_fired@0.5: %d/%d", summary["n_fired"], summary["n"])
    log.info("  p50=%.4f  p90=%.4f  p99=%.4f  max=%.4f",
             summary["p50"], summary["p90"], summary["p99"], summary["max"])
    log.info("  vs P8A: Spearman r=%.4f (p=%.3g), Pearson r=%.4f (p=%.3g)",
             summary["spearman_r"], summary["spearman_p"],
             summary["pearson_r"], summary["pearson_p"])
    log.info("  delta(T5C−P8A): mean=%+.4f median=%+.4f min=%+.4f max=%+.4f",
             summary["mean_delta"], summary["median_delta"],
             summary["min_delta"], summary["max_delta"])
    log.info("  band counts: %s", summary["band_counts"])
    log.info("  fires: P8A=%d/%d, T5C=%d/%d, both=%d, T5C-only=%d, P8A-only=%d",
             summary["p8a_fires"], summary["n"], summary["t5c_fires"], summary["n"],
             summary["both_fire"], summary["t5c_only_fire"], summary["p8a_only_fire"])

    # Persist summary JSON for later auditability
    import json
    summary_path = OUT_DIR / "t5c_step3500_may6_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info("wrote -> %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
