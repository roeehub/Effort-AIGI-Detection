"""Phase 2 — Re-crop frames live and re-score with P8A.

NEVER touches GCS data. Reads original local frame images, applies tightness
transforms in memory, runs P8A inference on the resulting pixels. The on-disk
files are never written or modified.

Two experiments in one script:
  (A) Fixed-tightness sweep: score every frame at t ∈ TIGHTNESS_GRID, report
      aggregate FPR/recall at each level. Shows the full sensitivity curve.
  (B) Normalize-to-target: for each frame whose native face_area_ratio < TARGET,
      compute the t that brings it to TARGET (t = sqrt(TARGET / native_far)),
      apply it; frames already >= TARGET get t=1.0. Answers: "if production
      enforces min face_area_ratio via re-crop, what is FPR/recall?"

Tightness convention (mirrors crop_shortcut_2026-04-27/crop_sweep.py):
  t > 1.0 → tighter (center-crop 1/t side, resize back — face fills more)
  t = 1.0 → native crop
  t < 1.0 → looser (shrink + edge-replicate pad)

Usage:
  python3 analysis/quality_gate_2026-04-30/phase2_recrop_rescore.py [options]

Options:
  --split {lockbox,dev,all}   Default: all
  --max_frames N              Cap N frames per split (for quick testing)
  --batch_size N              Default: 32
  --device {mps,cpu}          Default: auto-detect MPS, else cpu
  --target_far FLOAT          Normalize-to-target face_area_ratio (default 0.45)

Outputs (analysis/quality_gate_2026-04-30/outputs/):
  phase2_per_frame_scores.csv        — (frame_id, label, native_far, tightness, prob_fake)
  phase2_tightness_fpr_table.csv     — aggregate FPR/recall at each tightness
  phase2_normalize_result.json       — headline: FPR/recall after normalization
  phase2_tightness_plot.png
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

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
logger = logging.getLogger("phase2-recrop")

# ---------------------------------------------------------------------------
CSV = REPO / "analysis/crop_shortcut_2026-04-27/p8a_lockbox_join_2026-04-27.csv"
DEFAULT_CKPT = REPO / "analysis/_features_cache_2026-04-30/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth"
DETECTOR_CFG = REPO / "config/detector/effort.yaml"
TRAIN_CFG = REPO / "config/train_config.yaml"
OUT = REPO / "analysis/quality_gate_2026-04-30/outputs"

TAU = 0.9741
SCORE_COL = "prob_fake_p8a"
RESOLUTION = 224
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

TIGHTNESS_GRID = [0.70, 0.85, 1.00, 1.20, 1.50]


# ---------------------------------------------------------------------------
# Image helpers — never write any pixels back to disk
# ---------------------------------------------------------------------------
def make_variant(img: Image.Image, tightness: float) -> Image.Image:
    """Apply tightness transform in memory only. Source image is never modified."""
    if abs(tightness - 1.0) < 1e-6:
        return img.copy()
    W, H = img.size
    if tightness > 1.0:
        new_w = max(1, int(round(W / tightness)))
        new_h = max(1, int(round(H / tightness)))
        x0 = (W - new_w) // 2
        y0 = (H - new_h) // 2
        sub = img.crop((x0, y0, x0 + new_w, y0 + new_h))
        return sub.resize((W, H), Image.LANCZOS)
    # Looser: shrink + edge-replicate pad
    inner_w = max(1, int(round(W * tightness)))
    inner_h = max(1, int(round(H * tightness)))
    inner = img.resize((inner_w, inner_h), Image.LANCZOS)
    canvas = Image.new(img.mode, (W, H))
    pad_x = (W - inner_w) // 2
    pad_y = (H - inner_h) // 2
    if pad_y > 0:
        top = inner.crop((0, 0, inner_w, 1)).resize((inner_w, pad_y))
        canvas.paste(top, (pad_x, 0))
    if pad_y > 0 and (H - pad_y - inner_h) > 0:
        bot = inner.crop((0, inner_h - 1, inner_w, inner_h)).resize((inner_w, H - pad_y - inner_h))
        canvas.paste(bot, (pad_x, pad_y + inner_h))
    if pad_x > 0:
        lf = inner.crop((0, 0, 1, inner_h)).resize((pad_x, inner_h))
        canvas.paste(lf, (0, pad_y))
    if pad_x > 0 and (W - pad_x - inner_w) > 0:
        rt = inner.crop((inner_w - 1, 0, inner_w, inner_h)).resize((W - pad_x - inner_w, inner_h))
        canvas.paste(rt, (pad_x + inner_w, pad_y))
    if pad_x > 0 and pad_y > 0:
        canvas.paste(inner.crop((0, 0, 1, 1)).resize((pad_x, pad_y)), (0, 0))
    if pad_x > 0 and pad_y > 0 and (W - pad_x - inner_w) > 0:
        canvas.paste(inner.crop((inner_w - 1, 0, inner_w, 1)).resize((W - pad_x - inner_w, pad_y)),
                     (pad_x + inner_w, 0))
    if pad_x > 0 and pad_y > 0 and (H - pad_y - inner_h) > 0:
        canvas.paste(inner.crop((0, inner_h - 1, 1, inner_h)).resize((pad_x, H - pad_y - inner_h)),
                     (0, pad_y + inner_h))
    if (pad_x > 0 and pad_y > 0 and (W - pad_x - inner_w) > 0 and (H - pad_y - inner_h) > 0):
        canvas.paste(
            inner.crop((inner_w - 1, inner_h - 1, inner_w, inner_h)).resize(
                (W - pad_x - inner_w, H - pad_y - inner_h)),
            (pad_x + inner_w, pad_y + inner_h),
        )
    canvas.paste(inner, (pad_x, pad_y))
    return canvas


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB"))
    arr = cv2.resize(arr, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LINEAR)
    t = T.Compose([T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])
    return t(arr)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(device: torch.device, ckpt_path: Path) -> torch.nn.Module:
    logger.info("Loading checkpoint from %s", ckpt_path)
    with open(DETECTOR_CFG) as f:
        cfg = yaml.safe_load(f)
    with open(TRAIN_CFG) as f:
        cfg.update(yaml.safe_load(f))

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
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


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def score_batch(model: torch.nn.Module, tensors: List[torch.Tensor], device: torch.device) -> np.ndarray:
    batch = torch.stack(tensors).to(device, non_blocking=True)
    with torch.inference_mode():
        out = model({"image": batch}, inference=True)
        return out["prob"].detach().cpu().numpy().reshape(-1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["lockbox", "dev", "all"], default="all")
    p.add_argument("--max_frames", type=int, default=None,
                   help="Cap frames per split (for quick testing)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", choices=["mps", "cpu"], default=None)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                   help="Local checkpoint path to evaluate")
    p.add_argument("--checkpoint_label", default="P8A",
                   help="Human-readable label used in plots and summaries")
    p.add_argument("--out_prefix", default="phase2",
                   help="Output filename prefix. Default preserves original P8A filenames.")
    p.add_argument("--target_far", type=float, default=0.45,
                   help="Target face_area_ratio for normalize-to-target experiment")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("MPS not available, using CPU")

    OUT.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load metadata
    # -----------------------------------------------------------------------
    logger.info("Loading CSV metadata ...")
    raw = pd.read_csv(CSV, low_memory=False)

    if args.split != "all":
        raw = raw[raw["split"] == args.split].copy()
    logger.info("  %d frames after split filter (%s)", len(raw), args.split)

    # Drop no-face rows (no bbox to work with)
    raw = raw[~raw["is_no_face"].fillna(False)].copy()
    raw = raw[raw["local_path"].notna()].copy()
    logger.info("  %d frames with face bbox and local path", len(raw))

    if args.max_frames:
        # Stratified subsample: keep balance across (split, label)
        chunks = []
        per_group = args.max_frames // 4
        for (sp, lb), grp in raw.groupby(["split", "label"]):
            chunks.append(grp.sample(min(len(grp), per_group), random_state=42))
        raw = pd.concat(chunks).reset_index(drop=True)
        logger.info("  Subsampled to %d frames (--max_frames %d)", len(raw), args.max_frames)

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    model = load_model(device, args.ckpt)

    # -----------------------------------------------------------------------
    # Experiment A: fixed tightness sweep
    # -----------------------------------------------------------------------
    logger.info("Experiment A: fixed tightness sweep over %s ...", TIGHTNESS_GRID)
    n_pairs = len(raw) * len(TIGHTNESS_GRID)
    logger.info("  Total (frame × tightness) pairs: %d", n_pairs)

    t0 = time.time()
    pending: list[dict] = []

    for _, row in raw.iterrows():
        try:
            img = Image.open(row["local_path"]).convert("RGB")
        except Exception as e:
            logger.warning("Cannot open %s: %s", row["local_path"], e)
            continue
        for t in TIGHTNESS_GRID:
            variant = make_variant(img, t)
            tensor = pil_to_tensor(variant)
            pending.append({
                "frame_id": Path(row["local_path"]).stem,
                "local_path": row["local_path"],
                "label": row["label"],
                "split": row["split"],
                "native_far": row["face_area_ratio"],
                "cached_prob_fake_p8a": row[SCORE_COL],
                "clip_capture_mode": row["clip_capture_mode"],
                "tightness": float(t),
                "_tensor": tensor,
            })

    logger.info("  Scoring %d pairs in batches of %d ...", len(pending), args.batch_size)
    rows_a: list[dict] = []
    for i in range(0, len(pending), args.batch_size):
        chunk = pending[i : i + args.batch_size]
        probs = score_batch(model, [c["_tensor"] for c in chunk], device)
        for c, p in zip(chunk, probs):
            rows_a.append({k: v for k, v in c.items() if k != "_tensor"} | {"prob_fake": float(p)})
        if (i // args.batch_size) % 50 == 0:
            elapsed = time.time() - t0
            pct = (i + len(chunk)) / len(pending)
            eta = elapsed / pct * (1 - pct) if pct > 0 else 0
            logger.info("  %.1f%% done, ETA %.0fs", pct * 100, eta)

    scores_df = pd.DataFrame(rows_a)
    native_scores = (
        scores_df[scores_df["tightness"] == 1.0]
        .set_index(["frame_id", "split"])["prob_fake"]
        .to_dict()
    )
    csv_path = OUT / f"{args.out_prefix}_per_frame_scores.csv"
    scores_df.drop(columns=["_tensor"], errors="ignore").to_csv(csv_path, index=False)
    logger.info("Saved per-frame scores to %s", csv_path)

    # -----------------------------------------------------------------------
    # Aggregate: FPR / recall at each tightness
    # -----------------------------------------------------------------------
    agg_rows = []
    for split_filter in (["lockbox", "dev"] if args.split == "all" else [args.split]):
        sub_split = scores_df[scores_df["split"] == split_filter]
        native_split = sub_split[sub_split["tightness"] == 1.0]
        native_reals = native_split[native_split["label"] == "real"]["prob_fake"]
        native_fakes = native_split[native_split["label"] == "fake"]["prob_fake"]
        nat_fpr = (native_reals >= TAU).mean() if len(native_reals) else float("nan")
        nat_rec = (native_fakes >= TAU).mean() if len(native_fakes) else float("nan")
        for t in TIGHTNESS_GRID:
            sub_t = sub_split[sub_split["tightness"] == t]
            reals = sub_t[sub_t["label"] == "real"]["prob_fake"]
            fakes = sub_t[sub_t["label"] == "fake"]["prob_fake"]
            fpr = (reals >= TAU).mean() if len(reals) else float("nan")
            rec = (fakes >= TAU).mean() if len(fakes) else float("nan")
            agg_rows.append({
                "split": split_filter,
                "tightness": t,
                "n_real": len(reals),
                "n_fake": len(fakes),
                "fpr": round(float(fpr), 4),
                "recall": round(float(rec), 4),
                "native_fpr": round(float(nat_fpr), 4),
                "native_recall": round(float(nat_rec), 4),
                "fpr_delta": round(float(fpr) - float(nat_fpr), 4),
                "recall_delta": round(float(rec) - float(nat_rec), 4),
            })

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(OUT / f"{args.out_prefix}_tightness_fpr_table.csv", index=False)
    logger.info("Tightness FPR table:")
    print(agg_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # Experiment B: normalize-to-target face_area_ratio
    # -----------------------------------------------------------------------
    target_far = args.target_far
    logger.info("Experiment B: normalize-to-target face_area_ratio >= %.2f ...", target_far)

    # For each frame: compute t needed to reach target_far from native_far.
    # t = sqrt(target_far / native_far); clamp to [1.0, 2.5] (only tighten, never loosen).
    norm_rows: list[dict] = []
    pending_b: list[dict] = []

    for _, row in raw.iterrows():
        native_far = row["face_area_ratio"]
        frame_id = Path(row["local_path"]).stem
        native_prob = native_scores.get((frame_id, row["split"]))
        if pd.isna(native_far) or native_far <= 0:
            continue
        if native_prob is None:
            logger.warning("Missing native score for %s (%s); skipping normalize row", frame_id, row["split"])
            continue
        if native_far >= target_far:
            # Already meets the criterion — use native score without re-inference
            norm_rows.append({
                "frame_id": frame_id,
                "label": row["label"],
                "split": row["split"],
                "native_far": native_far,
                "applied_t": 1.0,
                "estimated_far_after": native_far,
                "prob_fake": native_prob,
                "native_prob_fake": native_prob,
                "cached_prob_fake_p8a": row[SCORE_COL],
                "clip_capture_mode": row["clip_capture_mode"],
                "from_cache": True,
            })
        else:
            t_needed = min(2.5, math.sqrt(target_far / native_far))
            try:
                img = Image.open(row["local_path"]).convert("RGB")
            except Exception as e:
                logger.warning("Cannot open %s: %s", row["local_path"], e)
                continue
            variant = make_variant(img, t_needed)
            tensor = pil_to_tensor(variant)
            pending_b.append({
                "frame_id": frame_id,
                "label": row["label"],
                "split": row["split"],
                "native_far": native_far,
                "applied_t": t_needed,
                "estimated_far_after": min(1.0, native_far * t_needed ** 2),
                "native_prob_fake": native_prob,
                "cached_prob_fake_p8a": row[SCORE_COL],
                "clip_capture_mode": row["clip_capture_mode"],
                "from_cache": False,
                "_tensor": tensor,
            })

    logger.info("  B: %d frames from cache (already tight), %d need re-inference",
                len(norm_rows), len(pending_b))

    if pending_b:
        logger.info("  B: scoring %d loose frames ...", len(pending_b))
        for i in range(0, len(pending_b), args.batch_size):
            chunk = pending_b[i : i + args.batch_size]
            probs = score_batch(model, [c["_tensor"] for c in chunk], device)
            for c, p in zip(chunk, probs):
                norm_rows.append({k: v for k, v in c.items() if k != "_tensor"} | {"prob_fake": float(p)})

    norm_df = pd.DataFrame(norm_rows)

    # Compute FPR/recall before and after normalization, per split
    norm_result: dict = {"target_far": target_far, "splits": {}}
    for split_filter in (["lockbox", "dev"] if args.split == "all" else [args.split]):
        sub = norm_df[norm_df["split"] == split_filter]
        reals = sub[sub["label"] == "real"]
        fakes = sub[sub["label"] == "fake"]
        after_fpr = (reals["prob_fake"] >= TAU).mean() if len(reals) else float("nan")
        after_rec = (fakes["prob_fake"] >= TAU).mean() if len(fakes) else float("nan")
        before_fpr = (reals["native_prob_fake"] >= TAU).mean() if len(reals) else float("nan")
        before_rec = (fakes["native_prob_fake"] >= TAU).mean() if len(fakes) else float("nan")
        n_tightened = int((sub["from_cache"] == False).sum())  # noqa: E712
        norm_result["splits"][split_filter] = {
            "n_real": len(reals), "n_fake": len(fakes),
            "n_frames_tightened": n_tightened,
            "frac_tightened": round(n_tightened / len(sub), 4) if len(sub) else float("nan"),
            "before_fpr": round(float(before_fpr), 4),
            "before_recall": round(float(before_rec), 4),
            "after_fpr": round(float(after_fpr), 4),
            "after_recall": round(float(after_rec), 4),
            "fpr_delta": round(float(after_fpr) - float(before_fpr), 4),
            "recall_delta": round(float(after_rec) - float(before_rec), 4),
        }
        logger.info("  B %s: before FPR=%.3f recall=%.3f  after FPR=%.3f recall=%.3f  "
                    "(%d/%d frames re-cropped = %.1f%%)",
                    split_filter, before_fpr, before_rec, after_fpr, after_rec,
                    n_tightened, len(sub), n_tightened / len(sub) * 100 if len(sub) else 0)

    norm_csv = OUT / f"{args.out_prefix}_normalize_per_frame.csv"
    norm_df.drop(columns=["_tensor"], errors="ignore").to_csv(norm_csv, index=False)
    normalize_json = OUT / f"{args.out_prefix}_normalize_result.json"
    with open(normalize_json, "w") as f:
        json.dump(norm_result, f, indent=2)
    logger.info("Saved normalize result to %s", normalize_json)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    logger.info("Generating plots ...")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f"Phase 2 — Re-crop + Re-score ({args.checkpoint_label}, τ={TAU})", fontsize=13)

    colors = {"lockbox": "steelblue", "dev": "darkorange"}
    for split_filter in (["lockbox", "dev"] if args.split == "all" else [args.split]):
        sub_agg = agg_df[agg_df["split"] == split_filter]

        # Panel 1: FPR vs tightness
        ax = axes[0]
        ax.plot(sub_agg["tightness"], sub_agg["fpr"], "o-", color=colors.get(split_filter, "gray"),
                label=split_filter, ms=6)
        ax.axhline(sub_agg["native_fpr"].iloc[0], color=colors.get(split_filter, "gray"),
                   ls="--", lw=1, alpha=0.6)

        # Panel 2: Recall vs tightness
        ax = axes[1]
        ax.plot(sub_agg["tightness"], sub_agg["recall"], "s-", color=colors.get(split_filter, "gray"),
                label=split_filter, ms=6)
        ax.axhline(sub_agg["native_recall"].iloc[0], color=colors.get(split_filter, "gray"),
                   ls="--", lw=1, alpha=0.6)

    axes[0].axhline(0.05, color="orange", ls=":", lw=1, label="5% FPR")
    axes[0].axhline(0.01, color="green", ls=":", lw=1, label="1% FPR")
    axes[0].set_xlabel("Tightness factor applied to all frames")
    axes[0].set_ylabel("FPR @ τ")
    axes[0].set_title("FPR vs Tightness (real frames)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(1.0, color="gray", ls=":", lw=0.8)

    axes[1].set_xlabel("Tightness factor applied to all frames")
    axes[1].set_ylabel("Fake recall @ τ")
    axes[1].set_title("Recall vs Tightness (fake frames)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(1.0, color="gray", ls=":", lw=0.8)

    # Panel 3: before/after normalize-to-target scatter (per frame)
    ax = axes[2]
    reals_norm = norm_df[norm_df["label"] == "real"]
    fakes_norm = norm_df[norm_df["label"] == "fake"]
    ax.scatter(reals_norm["native_prob_fake"], reals_norm["prob_fake"],
               alpha=0.2, s=8, color="steelblue", label="real")
    ax.scatter(fakes_norm["native_prob_fake"], fakes_norm["prob_fake"],
               alpha=0.2, s=8, color="darkorange", label="fake")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="no change")
    ax.axhline(TAU, color="red", ls=":", lw=0.8, label=f"τ={TAU}")
    ax.axvline(TAU, color="red", ls=":", lw=0.8)
    ax.set_xlabel("Native prob_fake (t=1.0 re-score)")
    ax.set_ylabel(f"Re-scored prob_fake (after normalize to far≥{target_far})")
    ax.set_title(f"Before vs After Normalization (far≥{target_far})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = OUT / f"{args.out_prefix}_tightness_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved plot to %s", plot_path)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 65)
    print("PHASE 2 SUMMARY")
    print("=" * 65)
    print(f"Checkpoint : {args.checkpoint_label} ({args.ckpt.name})")
    print(f"Device     : {device}")
    print(f"τ          : {TAU}")
    print()
    print("Experiment A — Fixed tightness:")
    print(agg_df[["split", "tightness", "n_real", "fpr", "native_fpr",
                  "fpr_delta", "recall", "native_recall", "recall_delta"]].to_string(index=False))
    print()
    print(f"Experiment B — Normalize-to-target far≥{target_far}:")
    for split_filter, info in norm_result["splits"].items():
        print(f"  {split_filter}: {info['n_frames_tightened']}/{info['n_real'] + info['n_fake']} "
              f"frames re-cropped ({info['frac_tightened']:.1%})")
        print(f"    FPR:    {info['before_fpr']:.3f} → {info['after_fpr']:.3f}  "
              f"(Δ={info['fpr_delta']:+.3f})")
        print(f"    Recall: {info['before_recall']:.3f} → {info['after_recall']:.3f}  "
              f"(Δ={info['recall_delta']:+.3f})")
    print()
    print("Phase 2 complete.")


if __name__ == "__main__":
    main()
