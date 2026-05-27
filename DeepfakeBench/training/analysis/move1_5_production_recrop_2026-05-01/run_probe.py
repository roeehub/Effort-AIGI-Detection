"""Move 1.5 — Production-tightness re-crop and re-score probe (2026-05-01).

Question: does re-cropping eval-substrate frames at production tightness shift
P8A's score distribution materially?

Production tightness convention (from recrop_dataset.py:39):
  TARGET_RFA = 0.85
  i.e. the face bounding box covers 85% of the image area.
  Production crop: square centered on face bbox; side = sqrt(face_area/0.85);
  resize to 224x224.

This probe:
  * Loads the 800-frame triptych sample CSV (lockbox + dev).
  * Stratifies by clip_capture_mode (~50/mode, ~100 total) — favors lockbox
    rows when available (the 800-frame CSV is ~87% dev), and includes a
    'webcam' bucket which the wiki flags as FPR-dominant.
  * For each frame, builds two model-input tensors:
    A) "eval-substrate as-is": the source frame resized to 224x224 (matches
       the cached pipeline that produced the CSV's prob_fake column).
    B) "production tight crop": square crop centered on face bbox at
       TARGET_RFA=0.85, resized to 224x224.
  * Scores both with the P8A checkpoint
    (analysis/_features_cache_2026-04-30/value_composite_effort_20260424_step5000_*.pth).
  * Reports per-frame Δprob_fake distribution, per-mode shift, FPR/recall at
    τ=0.5, τ=0.92 (corrected v3 contract policy), and τ=0.9741 (P8A
    operating point used in the existing thread).

Constraints honored: CPU/MPS only, no Vertex, no GCS downloads, no commits.
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import yaml
from PIL import Image

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from detectors import DETECTOR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("move1.5")

# ---------------------------------------------------------------------------
CSV = REPO / "analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/sampled_frames.csv"
CKPT = REPO / "analysis/_features_cache_2026-04-30/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth"
DETECTOR_CFG = REPO / "config/detector/effort.yaml"
TRAIN_CFG = REPO / "config/train_config.yaml"
OUT = REPO / "analysis/move1_5_production_recrop_2026-05-01/outputs"

RESOLUTION = 224
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# Production crop tightness (recrop_dataset.py:39). Face bbox area / image area = 0.85.
TARGET_RFA = 0.85

# Stratification target — capture modes and frames per mode.
PER_MODE_TARGET = 50  # aim ~50/mode; will downsample if fewer available
RANDOM_SEED = 42

# Operating points to score.
TAUS = {
    "tau_0.5": 0.5,
    "tau_0.92": 0.92,        # corrected v3 contract policy point
    "tau_0.9741": 0.9741,    # P8A historical operating point
}


# ---------------------------------------------------------------------------
# Crop builders
# ---------------------------------------------------------------------------
def asis_to_tensor(img: Image.Image) -> torch.Tensor:
    """Resize whole image to 224x224 and normalize."""
    arr = np.array(img.convert("RGB"))
    arr = cv2.resize(arr, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LINEAR)
    t = T.Compose([T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])
    return t(arr)


def production_crop_to_tensor(
    img: Image.Image,
    bbox_x: float,
    bbox_y: float,
    bbox_w: float,
    bbox_h: float,
) -> tuple[torch.Tensor, dict]:
    """Bbox-centered square crop at TARGET_RFA=0.85, resized to 224x224.

    Mirrors recrop_dataset.crop_to_target_rfa, but uses the CSV's MediaPipe
    bbox columns (which are tight axis-aligned bboxes of facial landmarks).
    Returns (tensor, debug-info-dict).
    """
    arr = np.array(img.convert("RGB"))
    H, W = arr.shape[:2]
    fx, fy, fw, fh = float(bbox_x), float(bbox_y), float(bbox_w), float(bbox_h)
    face_area = fw * fh
    debug = {
        "img_w": W,
        "img_h": H,
        "face_w": fw,
        "face_h": fh,
        "native_far": face_area / (W * H) if (W * H) > 0 else float("nan"),
    }
    if face_area <= 0 or (W * H) <= 0:
        # fallback to as-is — treat as a pass-through
        debug["fallback_asis"] = True
        return asis_to_tensor(img), debug

    new_side = math.sqrt(face_area / (TARGET_RFA + 1e-6))
    cx = fx + fw / 2.0
    cy = fy + fh / 2.0
    nx0 = int(round(cx - new_side / 2.0))
    ny0 = int(round(cy - new_side / 2.0))
    nx1 = int(round(cx + new_side / 2.0))
    ny1 = int(round(cy + new_side / 2.0))
    nx0c = max(0, nx0)
    ny0c = max(0, ny0)
    nx1c = min(W, nx1)
    ny1c = min(H, ny1)
    debug["crop_side_target"] = float(new_side)
    debug["crop_box_clamped"] = [nx0c, ny0c, nx1c, ny1c]
    debug["crop_box_unclamped"] = [nx0, ny0, nx1, ny1]
    debug["clamped_to_edge"] = bool(
        nx0 != nx0c or ny0 != ny0c or nx1 != nx1c or ny1 != ny1c
    )
    crop = arr[ny0c:ny1c, nx0c:nx1c]
    if crop.size == 0:
        debug["fallback_asis"] = True
        return asis_to_tensor(img), debug
    debug["realized_far"] = face_area / (crop.shape[0] * crop.shape[1]) if crop.size else float("nan")
    crop = cv2.resize(crop, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)
    t = T.Compose([T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])
    return t(crop), debug


# ---------------------------------------------------------------------------
# Model loading (mirrors phase2_recrop_rescore.load_model)
# ---------------------------------------------------------------------------
def load_model(device: torch.device) -> torch.nn.Module:
    logger.info("Loading checkpoint from %s", CKPT)
    with open(DETECTOR_CFG) as f:
        cfg = yaml.safe_load(f)
    with open(TRAIN_CFG) as f:
        cfg.update(yaml.safe_load(f))

    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model_config = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
    for k, v in model_config.items():
        if k != "current_arcface_s":
            cfg[k] = v

    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)
    if model_config.get("use_arcface_head") and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])
    clean = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:
        logger.debug("Missing keys: %d", len(missing))
    if unexpected:
        logger.debug("Unexpected keys: %d", len(unexpected))
    model.eval()
    logger.info("Model loaded on %s", device)
    return model


def score_batch(model: torch.nn.Module, tensors: List[torch.Tensor], device: torch.device) -> np.ndarray:
    if not tensors:
        return np.array([])
    batch = torch.stack(tensors).to(device, non_blocking=True)
    with torch.inference_mode():
        out = model({"image": batch}, inference=True)
        return out["prob"].detach().cpu().numpy().reshape(-1)


# ---------------------------------------------------------------------------
# Stratified sample
# ---------------------------------------------------------------------------
def stratified_sample(df: pd.DataFrame, per_mode: int, seed: int) -> pd.DataFrame:
    """Take ~per_mode rows per clip_capture_mode. Prefer lockbox rows when both splits available."""
    df = df.copy()
    df = df[df["local_path"].notna()]
    df = df[df["face_bbox_w"].notna() & df["face_bbox_h"].notna()]
    df = df[df["face_bbox_w"] > 0]
    df = df[df["face_bbox_h"] > 0]
    df = df[~df["is_no_face"].fillna(False)]
    chunks = []
    rng = np.random.default_rng(seed)
    for mode, grp in df.groupby("clip_capture_mode"):
        n = min(len(grp), per_mode)
        # Bias toward lockbox rows up to half the slot, then fill with dev
        lock = grp[grp["split"] == "lockbox"]
        dev = grp[grp["split"] == "dev"]
        n_lock = min(len(lock), n // 2)
        n_dev = n - n_lock
        n_dev = min(n_dev, len(dev))
        deficit = n - (n_lock + n_dev)
        if deficit > 0:
            # backfill from lockbox if dev was the bottleneck
            extra_lock = min(deficit, len(lock) - n_lock)
            n_lock += extra_lock
        sampled_lock = lock.sample(n=n_lock, random_state=int(rng.integers(0, 1 << 31))) if n_lock else lock.iloc[:0]
        sampled_dev = dev.sample(n=n_dev, random_state=int(rng.integers(0, 1 << 31))) if n_dev else dev.iloc[:0]
        chunks.append(pd.concat([sampled_lock, sampled_dev]))
    out = pd.concat(chunks).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("MPS not available, using CPU")

    raw = pd.read_csv(CSV, low_memory=False)
    logger.info("CSV rows: %d", len(raw))

    sample = stratified_sample(raw, per_mode=PER_MODE_TARGET, seed=RANDOM_SEED)
    logger.info("Sampled %d frames stratified by clip_capture_mode", len(sample))
    mode_counts = sample.groupby(["clip_capture_mode", "split"]).size().unstack(fill_value=0)
    logger.info("Per-mode breakdown:\n%s", mode_counts.to_string())

    model = load_model(device)

    rows = []
    asis_tensors: List[torch.Tensor] = []
    prod_tensors: List[torch.Tensor] = []
    record_meta: List[dict] = []

    t0 = time.time()
    for _, row in sample.iterrows():
        try:
            img = Image.open(row["local_path"]).convert("RGB")
        except Exception as e:
            logger.warning("Cannot open %s: %s", row["local_path"], e)
            continue
        asis_t = asis_to_tensor(img)
        prod_t, debug = production_crop_to_tensor(
            img, row["face_bbox_x"], row["face_bbox_y"], row["face_bbox_w"], row["face_bbox_h"],
        )
        asis_tensors.append(asis_t)
        prod_tensors.append(prod_t)
        record_meta.append({
            "frame_id": Path(row["local_path"]).stem,
            "local_path": row["local_path"],
            "label": row["label"],
            "split": row["split"],
            "method": row.get("method"),
            "clip_capture_mode": row.get("clip_capture_mode"),
            "native_far": row.get("face_area_ratio"),
            "csv_prob_fake": row.get("prob_fake"),
            "img_w": debug["img_w"],
            "img_h": debug["img_h"],
            "face_w": debug["face_w"],
            "face_h": debug["face_h"],
            "computed_native_far": debug.get("native_far"),
            "crop_side_target": debug.get("crop_side_target"),
            "crop_box_clamped": debug.get("crop_box_clamped"),
            "clamped_to_edge": debug.get("clamped_to_edge"),
            "realized_far": debug.get("realized_far"),
            "fallback_asis": debug.get("fallback_asis", False),
        })

    logger.info("Built %d (asis, prod) pairs in %.1fs", len(asis_tensors), time.time() - t0)

    # Score both arms
    BATCH = 16
    asis_probs = []
    prod_probs = []
    t0 = time.time()
    for i in range(0, len(asis_tensors), BATCH):
        asis_probs.extend(score_batch(model, asis_tensors[i:i + BATCH], device).tolist())
        prod_probs.extend(score_batch(model, prod_tensors[i:i + BATCH], device).tolist())
        if (i // BATCH) % 10 == 0:
            done = min(i + BATCH, len(asis_tensors))
            elapsed = time.time() - t0
            eta = elapsed / max(done, 1) * (len(asis_tensors) - done)
            logger.info("Scored %d/%d (ETA %.0fs)", done, len(asis_tensors), eta)
    logger.info("Scoring done in %.1fs", time.time() - t0)

    for i, m in enumerate(record_meta):
        m["asis_prob_fake"] = float(asis_probs[i])
        m["prod_prob_fake"] = float(prod_probs[i])
        m["delta_prob_fake"] = float(prod_probs[i] - asis_probs[i])

    df = pd.DataFrame(record_meta)
    csv_path = OUT / "recrop_per_frame.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)

    # Summary
    summary: dict = {
        "n_frames_total": int(len(df)),
        "checkpoint": CKPT.name,
        "production_crop_spec": {
            "target_rfa": TARGET_RFA,
            "source": "recrop_dataset.py:39",
            "convention": "square crop centered on face bbox; side = sqrt(face_area/TARGET_RFA); resize to 224x224",
            "bbox_source_in_csv": "MediaPipe FaceMesh landmark bbox (face_bbox_x/y/w/h)",
        },
        "delta_prob_fake_distribution": {
            "median": float(np.median(df["delta_prob_fake"])),
            "p25": float(np.percentile(df["delta_prob_fake"], 25)),
            "p75": float(np.percentile(df["delta_prob_fake"], 75)),
            "min": float(df["delta_prob_fake"].min()),
            "max": float(df["delta_prob_fake"].max()),
            "mean": float(df["delta_prob_fake"].mean()),
            "std": float(df["delta_prob_fake"].std()),
            "abs_median": float(np.median(np.abs(df["delta_prob_fake"]))),
            "abs_p75": float(np.percentile(np.abs(df["delta_prob_fake"]), 75)),
            "abs_p90": float(np.percentile(np.abs(df["delta_prob_fake"]), 90)),
            "n_frames_delta_gt_0.10": int((np.abs(df["delta_prob_fake"]) > 0.10).sum()),
            "n_frames_delta_gt_0.30": int((np.abs(df["delta_prob_fake"]) > 0.30).sum()),
        },
        "per_capture_mode": {},
        "per_label": {},
        "tau_metrics": {},
        "csv_consistency_check": {},
    }

    # Per capture mode
    for mode, grp in df.groupby("clip_capture_mode"):
        sub = {
            "n": int(len(grp)),
            "n_real": int((grp["label"] == "real").sum()),
            "n_fake": int((grp["label"] == "fake").sum()),
            "delta_median": float(np.median(grp["delta_prob_fake"])),
            "delta_abs_median": float(np.median(np.abs(grp["delta_prob_fake"]))),
            "delta_p25": float(np.percentile(grp["delta_prob_fake"], 25)),
            "delta_p75": float(np.percentile(grp["delta_prob_fake"], 75)),
            "asis_mean_prob_fake": float(grp["asis_prob_fake"].mean()),
            "prod_mean_prob_fake": float(grp["prod_prob_fake"].mean()),
        }
        summary["per_capture_mode"][str(mode)] = sub

    # Per label
    for lab, grp in df.groupby("label"):
        sub = {
            "n": int(len(grp)),
            "delta_median": float(np.median(grp["delta_prob_fake"])),
            "delta_abs_median": float(np.median(np.abs(grp["delta_prob_fake"]))),
            "asis_mean_prob_fake": float(grp["asis_prob_fake"].mean()),
            "prod_mean_prob_fake": float(grp["prod_prob_fake"].mean()),
        }
        summary["per_label"][str(lab)] = sub

    # Tau metrics
    for tau_name, tau_val in TAUS.items():
        reals = df[df["label"] == "real"]
        fakes = df[df["label"] == "fake"]
        asis_fpr = float((reals["asis_prob_fake"] >= tau_val).mean()) if len(reals) else float("nan")
        prod_fpr = float((reals["prod_prob_fake"] >= tau_val).mean()) if len(reals) else float("nan")
        asis_rec = float((fakes["asis_prob_fake"] >= tau_val).mean()) if len(fakes) else float("nan")
        prod_rec = float((fakes["prod_prob_fake"] >= tau_val).mean()) if len(fakes) else float("nan")
        # Per-mode FPR for the load-bearing webcam slice
        per_mode_fpr = {}
        for mode, grp in reals.groupby("clip_capture_mode"):
            per_mode_fpr[str(mode)] = {
                "n_real": int(len(grp)),
                "asis_fpr": float((grp["asis_prob_fake"] >= tau_val).mean()) if len(grp) else float("nan"),
                "prod_fpr": float((grp["prod_prob_fake"] >= tau_val).mean()) if len(grp) else float("nan"),
            }
        summary["tau_metrics"][tau_name] = {
            "tau": tau_val,
            "n_real": int(len(reals)),
            "n_fake": int(len(fakes)),
            "asis_fpr": asis_fpr,
            "prod_fpr": prod_fpr,
            "fpr_delta": prod_fpr - asis_fpr,
            "asis_recall": asis_rec,
            "prod_recall": prod_rec,
            "recall_delta": prod_rec - asis_rec,
            "per_mode_fpr": per_mode_fpr,
        }

    # CSV-consistency check: how close is asis_prob_fake to the cached CSV prob_fake?
    valid = df.dropna(subset=["csv_prob_fake"])
    if len(valid):
        diffs = (valid["asis_prob_fake"] - valid["csv_prob_fake"]).abs()
        summary["csv_consistency_check"] = {
            "n": int(len(valid)),
            "median_abs_diff": float(diffs.median()),
            "p95_abs_diff": float(np.percentile(diffs, 95)),
            "max_abs_diff": float(diffs.max()),
            "note": "asis_prob_fake should match csv_prob_fake closely; large diff would indicate model-load drift.",
        }

    json_path = OUT / "recrop_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved %s", json_path)

    # Plot
    logger.info("Generating plot ...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Move 1.5 — Production-tightness re-crop vs eval-as-is (P8A, n=%d)" % len(df))

    # (1) histogram of delta
    ax = axes[0]
    reals = df[df["label"] == "real"]["delta_prob_fake"]
    fakes = df[df["label"] == "fake"]["delta_prob_fake"]
    bins = np.linspace(-1, 1, 41)
    ax.hist(reals, bins=bins, alpha=0.55, color="steelblue", label=f"real (n={len(reals)})")
    ax.hist(fakes, bins=bins, alpha=0.55, color="darkorange", label=f"fake (n={len(fakes)})")
    ax.axvline(0, color="k", ls="--", lw=0.7)
    ax.set_xlabel("Δprob_fake = prod − asis")
    ax.set_ylabel("count")
    ax.set_title("Δprob_fake distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2) scatter asis vs prod per frame, color by label
    ax = axes[1]
    for lab, color in [("real", "steelblue"), ("fake", "darkorange")]:
        sub = df[df["label"] == lab]
        ax.scatter(sub["asis_prob_fake"], sub["prod_prob_fake"], alpha=0.5, s=18, color=color, label=lab)
    ax.plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.6)
    for tau_name, tau_val in TAUS.items():
        ax.axhline(tau_val, ls=":", lw=0.6, alpha=0.4)
        ax.axvline(tau_val, ls=":", lw=0.6, alpha=0.4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("asis prob_fake (eval-substrate as-is)")
    ax.set_ylabel("prod prob_fake (RFA=0.85 face crop)")
    ax.set_title("Per-frame asis vs prod")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (3) per-mode delta box-plot
    ax = axes[2]
    modes = sorted(df["clip_capture_mode"].dropna().unique().tolist())
    box_data = [df[df["clip_capture_mode"] == m]["delta_prob_fake"].values for m in modes]
    bp = ax.boxplot(box_data, labels=modes, showmeans=True, vert=True)
    ax.axhline(0, color="k", ls="--", lw=0.6, alpha=0.5)
    ax.set_ylabel("Δprob_fake (prod − asis)")
    ax.set_title("Δprob_fake by capture mode")
    ax.grid(True, alpha=0.3)
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)

    plt.tight_layout()
    png_path = OUT / "score_shift_distribution.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", png_path)

    # Stdout summary
    print()
    print("=" * 70)
    print("MOVE 1.5 SUMMARY")
    print("=" * 70)
    print(f"Frames scored : {len(df)}")
    print(f"Production-tightness target_rfa : {TARGET_RFA} (recrop_dataset.py:39)")
    cv = summary.get("csv_consistency_check") or {}
    if cv:
        print(f"asis vs cached csv prob_fake median |diff| : {cv['median_abs_diff']:.4f}, "
              f"p95 {cv['p95_abs_diff']:.4f}, max {cv['max_abs_diff']:.4f}")
    d = summary["delta_prob_fake_distribution"]
    print(f"Δprob_fake median {d['median']:+.4f}  p25 {d['p25']:+.4f}  p75 {d['p75']:+.4f}")
    print(f"           |Δ| median {d['abs_median']:.4f}  p75 {d['abs_p75']:.4f}  p90 {d['abs_p90']:.4f}")
    print(f"           frames with |Δ|>0.10 : {d['n_frames_delta_gt_0.10']}/{len(df)}")
    print(f"           frames with |Δ|>0.30 : {d['n_frames_delta_gt_0.30']}/{len(df)}")
    print()
    print("Per capture mode:")
    print(f"{'mode':<20s} {'n':>4s} {'real':>4s} {'fake':>4s} {'Δmed':>9s} {'|Δ|med':>9s} "
          f"{'asis<f>':>9s} {'prod<f>':>9s}")
    for mode, sub in summary["per_capture_mode"].items():
        print(f"{mode:<20s} {sub['n']:>4d} {sub['n_real']:>4d} {sub['n_fake']:>4d} "
              f"{sub['delta_median']:>+9.4f} {sub['delta_abs_median']:>9.4f} "
              f"{sub['asis_mean_prob_fake']:>9.4f} {sub['prod_mean_prob_fake']:>9.4f}")
    print()
    print("τ-FPR / Recall (real vs fake):")
    for tau_name, info in summary["tau_metrics"].items():
        print(f"  {tau_name} (τ={info['tau']:.4f}): "
              f"n_real={info['n_real']} n_fake={info['n_fake']}")
        print(f"    FPR    asis={info['asis_fpr']:.4f}  prod={info['prod_fpr']:.4f}  "
              f"Δ={info['fpr_delta']:+.4f}")
        print(f"    Recall asis={info['asis_recall']:.4f}  prod={info['prod_recall']:.4f}  "
              f"Δ={info['recall_delta']:+.4f}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
