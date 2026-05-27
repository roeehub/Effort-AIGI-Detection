"""Step 2 — Run T5C step3500 (and comparison ckpts) inference on the two face crops.

Loads checkpoints via arena.model_arena.load_model (matches training preprocessing:
INTER_LINEAR resize, BGR→RGB, CLIP normalization).

Output:
  outputs/inference_results.json
  outputs/inference_results.csv
"""
from __future__ import annotations

import sys, os, json, time
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.transforms as T

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
sys.path.insert(0, str(REPO))

from arena.model_arena import CLIP_MEAN, CLIP_STD, _download_checkpoint, load_model  # noqa

CROPS_DIR = Path("analysis/teams_account_natural_experiment_2026-05-19/crops")
OUT_DIR = Path("analysis/teams_account_natural_experiment_2026-05-19/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CKPT_CACHE = Path("analysis/slot_b_property_shortcut_2026-05-16/ckpt_cache")

DETECTOR_CONFIG = REPO / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO / "config/defaults.yaml"
DEVICE = torch.device("cpu")
RES = 224

# Checkpoints to evaluate.
# T5C_STEP3500 is the user's "current deployment" per the 2026-05-19 prompt.
# Add P8A (production anchor / rank-1) and SLOT_A_RESCHAIN_STEP3500 (the most-recent
# training-side lever that bit; 2026-05-16 overnight) for context.
CKPTS = [
    {
        "name": "T5C_STEP3500_DEPLOY",
        "note": "user's current deployment per 2026-05-19 prompt",
        "local_path": LOCAL_CKPT_CACHE / "periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth",
        "gcs_uri": "gs://training-job-outputs/best_checkpoints/jrlldtem/periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth",
    },
    {
        "name": "P8A_STEP5000",
        "note": "production anchor (rank-1 on contract), 2026-04-24",
        "local_path": None,
        "gcs_uri": "gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth",
    },
    {
        "name": "SLOT_B_6AXIS_GRL_STEP3500",
        "note": "2026-05-16 6-axis GRL on T5C base",
        "local_path": LOCAL_CKPT_CACHE / "periodic_effort_20260516_step3500_auc0.9910_eer0.0175.pth",
        "gcs_uri": "gs://training-job-outputs/best_checkpoints/gwntcld0/periodic_effort_20260516_step3500_auc0.9910_eer0.0175.pth",
    },
]

# Two crops — same person, same camera, different Teams accounts
CROPS = [
    {"label": "Roy_D", "path": CROPS_DIR / "face_roy_d.png"},
    {"label": "Guest", "path": CROPS_DIR / "face_guest.png"},
]


def preprocess(raw_bgr_path: Path) -> torch.Tensor:
    """Match the training preprocessing: cv2.imread (BGR) → INTER_LINEAR resize → BGR2RGB → CLIP normalize."""
    img_bgr = cv2.imread(str(raw_bgr_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(raw_bgr_path)
    img_bgr = cv2.resize(img_bgr, (RES, RES), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])
    return transform(img_rgb).unsqueeze(0)  # (1, 3, 224, 224)


def resolve_ckpt(c: dict) -> Path:
    """Return a local path for the checkpoint, downloading from GCS if needed."""
    if c["local_path"] is not None and c["local_path"].exists():
        print(f"  using local cache: {c['local_path']}")
        return c["local_path"]
    print(f"  downloading: {c['gcs_uri']}")
    return Path(_download_checkpoint(c["gcs_uri"]))


def score_ckpt(c: dict, crops: list[dict]) -> dict:
    print(f"\n=== {c['name']} ({c['note']}) ===")
    t0 = time.time()
    local_ckpt = resolve_ckpt(c)
    print(f"  loading model ...")
    model = load_model(str(local_ckpt), str(DETECTOR_CONFIG), str(TRAIN_CONFIG), DEVICE)
    model.eval()
    print(f"  model loaded in {time.time()-t0:.1f}s")
    results = {}
    for crop in crops:
        x = preprocess(crop["path"]).to(DEVICE)
        with torch.inference_mode():
            out = model({"image": x}, inference=True)
        prob_fake = float(out["prob"].detach().cpu().numpy().reshape(-1)[0])
        # Also get cls logit if available
        if "cls" in out:
            cls = out["cls"].detach().cpu().numpy().reshape(-1).tolist()
        else:
            cls = None
        results[crop["label"]] = {"prob_fake": prob_fake, "cls": cls}
        print(f"    {crop['label']:>8}  prob_fake = {prob_fake:.6f}")
    # Pairwise delta + tau-readouts
    p_roy = results["Roy_D"]["prob_fake"]
    p_guest = results["Guest"]["prob_fake"]
    print(f"  Δ(Guest - Roy_D) = {p_guest - p_roy:+.6f}")
    print(f"  ratio Guest/Roy_D = {p_guest / max(p_roy, 1e-9):.3f}")
    # Flip indicators at common deployment thresholds
    for tau in [0.50, 0.70, 0.80, 0.90, 0.95, 0.98]:
        roy_flag = "FAKE" if p_roy >= tau else "real"
        guest_flag = "FAKE" if p_guest >= tau else "real"
        sym = "⚠ FLIP" if (p_roy >= tau) != (p_guest >= tau) else "same"
        print(f"    @ τ={tau:.2f}: Roy_D={roy_flag:<4}  Guest={guest_flag:<4}  {sym}")
    del model
    return results


all_results = {}
for c in CKPTS:
    try:
        all_results[c["name"]] = score_ckpt(c, CROPS)
    except Exception as e:
        print(f"  FAILED on {c['name']}: {e}")
        all_results[c["name"]] = {"error": str(e)}

# Save
with open(OUT_DIR / "inference_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

# CSV
import csv
with open(OUT_DIR / "inference_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ckpt", "label", "prob_fake", "delta_guest_minus_roy", "flip_at_tau_0.5", "flip_at_tau_0.9"])
    for name, r in all_results.items():
        if "error" in r:
            continue
        for lab in ["Roy_D", "Guest"]:
            w.writerow([name, lab, f"{r[lab]['prob_fake']:.6f}", "", "", ""])
        delta = r["Guest"]["prob_fake"] - r["Roy_D"]["prob_fake"]
        flip05 = (r["Roy_D"]["prob_fake"] >= 0.5) != (r["Guest"]["prob_fake"] >= 0.5)
        flip09 = (r["Roy_D"]["prob_fake"] >= 0.9) != (r["Guest"]["prob_fake"] >= 0.9)
        w.writerow([name, "DELTA", "", f"{delta:+.6f}", str(flip05), str(flip09)])

print("\n=== Final summary ===")
print(json.dumps(all_results, indent=2))
