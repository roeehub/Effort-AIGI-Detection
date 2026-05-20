"""may6 production-drift cohort, scored with VideoCodecSimulation aug applied.

PURPOSE: Predict whether the 3 launched packets will preserve P8A-style
may6 robustness OR compression-trap. Two scenarios:

  (a) aug-applied may6 scores ≈ no-aug may6 scores — aug is invariant on
      production frames; predicted training behavior: preserves robustness.
  (b) aug-applied may6 scores >> no-aug — aug pushes production frames
      INTO false-flag region; predicted training behavior: trained ckpt will
      false-flag may6 even at deployment-τ (compression trap).

Per Job F (2026-05-20), reference no-aug scores:
  - P8A_STEP5000      : 0/92 above τ_cal (within MPS noise)
  - SLOT_A_V2_STEP3500: 4/92 above τ_cal (preserves P8A-level robustness)
  - T5C_STEP3500      : 16/92 above τ_cal (4× more false-flags)

This script measures the no-aug baseline (sanity vs Job F) + aug-applied
scores per ckpt × 3 seeds per frame.

Outputs:
  outputs/may6_with_codec_summary.json
  outputs/may6_with_codec_per_frame.csv

Runtime: ~5-8 min on MPS / ~15 min on CPU (92 frames × 3 ckpts × (1 baseline + 3 aug) = ~1100 forward passes).
"""
from __future__ import annotations

import sys, json, time, csv
from pathlib import Path
import numpy as np
import cv2
import torch
import torchvision.transforms as T

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
sys.path.insert(0, str(REPO))

from arena.model_arena import CLIP_MEAN, CLIP_STD, load_model  # noqa
from data.augmentations.transforms import VideoCodecSimulation  # noqa

OUT_DIR = REPO / "analysis/codec_restoration_gates_2026-05-21" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAY6_DIR = REPO / "analysis/xinhe_cross_camera_audit_2026-05-06/raw/may6"
LOCAL_CKPT_CACHE = REPO / "analysis/slot_b_property_shortcut_2026-05-16/ckpt_cache"
DETECTOR_CONFIG = REPO / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO / "config/defaults.yaml"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")
RES = 224

CKPTS = [
    {
        "name": "SLOT_A_V2_STEP3500",
        "local_path": LOCAL_CKPT_CACHE / "periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth",
    },
    {
        "name": "T5C_STEP3500",
        "local_path": LOCAL_CKPT_CACHE / "periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth",
    },
]

# Get all may6 frames
FRAME_PATHS = sorted(MAY6_DIR.glob("*.png"))
print(f"may6 frames: {len(FRAME_PATHS)}")


def load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (RES, RES), interpolation=cv2.INTER_LINEAR)


def bgr_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])
    return transform(img_rgb).unsqueeze(0)


def apply_codec(img_bgr: np.ndarray, seed: int) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    np.random.seed(seed)
    import random as _r
    _r.seed(seed)
    aug = VideoCodecSimulation(codec_quality=(20, 65), p=1.0, always_apply=True)
    out_rgb = aug.apply(img_rgb)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def score(model, img_bgr: np.ndarray) -> float:
    x = bgr_to_tensor(img_bgr).to(DEVICE)
    with torch.inference_mode():
        out = model({"image": x}, inference=True)
    return float(out["prob"].detach().cpu().numpy().reshape(-1)[0])


per_frame_rows = []
summary = {}

for c in CKPTS:
    name = c["name"]
    print(f"\n=== {name} ===")
    t0 = time.time()
    model = load_model(str(c["local_path"]), str(DETECTOR_CONFIG), str(TRAIN_CONFIG), DEVICE)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    # Pre-load all frames once (small data, all in memory)
    print(f"  loading 92 may6 frames ...")
    frames_bgr = [load_bgr(p) for p in FRAME_PATHS]

    # Baseline scores (no aug)
    baseline_scores = []
    for i, img in enumerate(frames_bgr):
        s = score(model, img)
        baseline_scores.append(s)
        per_frame_rows.append({
            "ckpt": name, "frame": FRAME_PATHS[i].name, "variant": "no_aug", "seed": -1, "score": s,
        })
    base_arr = np.array(baseline_scores)
    print(f"  Baseline: mean={base_arr.mean():.4f} p50={np.percentile(base_arr, 50):.4f} p95={np.percentile(base_arr, 95):.4f} max={base_arr.max():.4f}")
    print(f"           frac > 0.50 = {(base_arr > 0.50).mean():.3f}  frac > 0.70 = {(base_arr > 0.70).mean():.3f}  frac > 0.85 = {(base_arr > 0.85).mean():.3f}")

    # Aug scores (3 seeds per frame)
    aug_scores_by_seed = {0: [], 1: [], 2: []}
    for seed in [0, 1, 2]:
        for i, img in enumerate(frames_bgr):
            aug_bgr = apply_codec(img, seed=seed * 1000 + i)
            s = score(model, aug_bgr)
            aug_scores_by_seed[seed].append(s)
            per_frame_rows.append({
                "ckpt": name, "frame": FRAME_PATHS[i].name, "variant": "codec_aug", "seed": seed, "score": s,
            })

    # Combine all aug seeds
    all_aug = np.array(sum(aug_scores_by_seed.values(), []))
    print(f"  Aug:      mean={all_aug.mean():.4f} p50={np.percentile(all_aug, 50):.4f} p95={np.percentile(all_aug, 95):.4f} max={all_aug.max():.4f}")
    print(f"           frac > 0.50 = {(all_aug > 0.50).mean():.3f}  frac > 0.70 = {(all_aug > 0.70).mean():.3f}  frac > 0.85 = {(all_aug > 0.85).mean():.3f}")

    # Per-frame paired delta (median across 3 seeds)
    aug_median_per_frame = np.median(np.array([aug_scores_by_seed[s] for s in [0, 1, 2]]), axis=0)
    delta = aug_median_per_frame - base_arr
    print(f"  Δ (aug_median - baseline) per frame: mean={delta.mean():+.4f}  p50={np.percentile(delta, 50):+.4f}  worst (most increasing)={delta.max():+.4f}")

    summary[name] = {
        "baseline": {
            "mean": float(base_arr.mean()),
            "p50": float(np.percentile(base_arr, 50)),
            "p95": float(np.percentile(base_arr, 95)),
            "max": float(base_arr.max()),
            "frac_above_0.50": float((base_arr > 0.50).mean()),
            "frac_above_0.70": float((base_arr > 0.70).mean()),
            "frac_above_0.85": float((base_arr > 0.85).mean()),
        },
        "aug_codec_q20_65": {
            "mean": float(all_aug.mean()),
            "p50": float(np.percentile(all_aug, 50)),
            "p95": float(np.percentile(all_aug, 95)),
            "max": float(all_aug.max()),
            "frac_above_0.50": float((all_aug > 0.50).mean()),
            "frac_above_0.70": float((all_aug > 0.70).mean()),
            "frac_above_0.85": float((all_aug > 0.85).mean()),
        },
        "paired_delta_mean": float(delta.mean()),
        "paired_delta_p50": float(np.percentile(delta, 50)),
        "paired_delta_worst_increase": float(delta.max()),
        "n_frames": len(FRAME_PATHS),
    }
    del model

with open(OUT_DIR / "may6_with_codec_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
with open(OUT_DIR / "may6_with_codec_per_frame.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ckpt", "frame", "variant", "seed", "score"])
    w.writeheader()
    w.writerows(per_frame_rows)

print("\n=== Final summary ===")
for name, d in summary.items():
    print(f"\n{name}")
    print(f"  Baseline mean={d['baseline']['mean']:.4f}  frac>0.50={d['baseline']['frac_above_0.50']:.3f}  frac>0.85={d['baseline']['frac_above_0.85']:.3f}")
    print(f"  Aug      mean={d['aug_codec_q20_65']['mean']:.4f}  frac>0.50={d['aug_codec_q20_65']['frac_above_0.50']:.3f}  frac>0.85={d['aug_codec_q20_65']['frac_above_0.85']:.3f}")
    print(f"  Paired Δ mean={d['paired_delta_mean']:+.4f}  worst_increase={d['paired_delta_worst_increase']:+.4f}")

print(f"\nWrote: {OUT_DIR / 'may6_with_codec_summary.json'}")
print(f"Wrote: {OUT_DIR / 'may6_with_codec_per_frame.csv'}")
