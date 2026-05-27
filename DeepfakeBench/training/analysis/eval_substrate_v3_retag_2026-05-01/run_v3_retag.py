"""Phase 2B — Full eval-substrate retag at production tightness (2026-05-01).

Extends Move 1.5 (190-frame stratified probe) to the full lockbox (839 rows).
Output: a v3-substrate eval CSV/parquet that future scorecards can use to
re-run promotion-contract numbers on a substrate that honestly translates
to production-FPR.

Pipeline mirrors analysis/move1_5_production_recrop_2026-05-01/run_probe.py:
  asis: full image -> 224x224 LINEAR resize -> CLIP normalize
        (bit-equivalent to phase2 t=1.0 / canonical eval pipeline)
  prod: square crop centered on bbox at TARGET_RFA=0.85 -> 224x224 AREA
        resize -> CLIP normalize.

Source data: analysis/lockbox_tagging/full_tags_2026-04-27.parquet
Hygiene: drop is_no_face=True, missing bbox, missing local_path, missing
          on disk. Keep only split=='lockbox'.

Constraints honored: CPU/MPS only, no Vertex, no GCS downloads, no commits,
no n_jobs=-1 sklearn calls.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import List

import cv2
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
logger = logging.getLogger("phase2b.v3_retag")

# ---------------------------------------------------------------------------
SOURCE_PARQUET = REPO / "analysis/lockbox_tagging/full_tags_2026-04-27.parquet"
CKPT = REPO / "analysis/_features_cache_2026-04-30/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth"
DETECTOR_CFG = REPO / "config/detector/effort.yaml"
TRAIN_CFG = REPO / "config/train_config.yaml"
OUT = REPO / "analysis/eval_substrate_v3_retag_2026-05-01/outputs"

RESOLUTION = 224
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# Production crop tightness (recrop_dataset.py:39).
TARGET_RFA = 0.85

# Operating points scored.
TAUS = {
    "tau_0.5": 0.5,
    "tau_0.92": 0.92,        # corrected v3 contract policy point
    "tau_0.9741": 0.9741,    # P8A historical operating point
}

# Inference batch size.
BATCH = 16


# ---------------------------------------------------------------------------
# Crop builders (verbatim from move1.5/run_probe.py)
# ---------------------------------------------------------------------------
def asis_to_tensor(img: Image.Image) -> torch.Tensor:
    """Resize whole image to 224x224 (LINEAR) and CLIP-normalize."""
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
    """Bbox-centered square crop at TARGET_RFA=0.85, AREA-resized to 224x224.

    Mirrors recrop_dataset.crop_to_target_rfa, but uses the parquet's
    MediaPipe bbox columns (face_bbox_x/y/w/h).
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
        "native_face_area_ratio": face_area / (W * H) if (W * H) > 0 else float("nan"),
    }
    if face_area <= 0 or (W * H) <= 0:
        debug["fallback_asis"] = True
        debug["realized_face_area_ratio_at_rfa_0.85"] = float("nan")
        debug["clamped_to_edge"] = False
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
        debug["realized_face_area_ratio_at_rfa_0.85"] = float("nan")
        return asis_to_tensor(img), debug
    debug["realized_face_area_ratio_at_rfa_0.85"] = (
        face_area / (crop.shape[0] * crop.shape[1]) if crop.size else float("nan")
    )
    crop = cv2.resize(crop, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)
    t = T.Compose([T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])
    return t(crop), debug


# ---------------------------------------------------------------------------
# Model loading (mirrors move1.5/run_probe.load_model)
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
# Hygiene + load
# ---------------------------------------------------------------------------
def load_lockbox() -> pd.DataFrame:
    raw = pd.read_parquet(SOURCE_PARQUET)
    logger.info("Source parquet rows: %d", len(raw))
    n0 = len(raw)
    df = raw[raw["split"] == "lockbox"].copy()
    n_lock = len(df)
    df = df[df["local_path"].notna()]
    df = df[df["face_bbox_x"].notna() & df["face_bbox_y"].notna()
            & df["face_bbox_w"].notna() & df["face_bbox_h"].notna()]
    df = df[df["face_bbox_w"] > 0]
    df = df[df["face_bbox_h"] > 0]
    df = df[~df["is_no_face"].fillna(False)]
    n_after_meta = len(df)
    # File-on-disk filter
    exists_mask = df["local_path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))
    n_missing_disk = int((~exists_mask).sum())
    df = df[exists_mask].reset_index(drop=True)
    logger.info(
        "Hygiene: %d total -> %d lockbox -> %d after metadata filter -> %d on-disk (dropped %d missing-on-disk)",
        n0, n_lock, n_after_meta, len(df), n_missing_disk,
    )
    return df, {
        "n_source_rows": n0,
        "n_lockbox_rows": n_lock,
        "n_after_metadata_filter": n_after_meta,
        "n_missing_on_disk": n_missing_disk,
        "n_kept_for_inference": int(len(df)),
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------
def compute_aggregate_metrics(df: pd.DataFrame) -> dict:
    out = {
        "n_total": int(len(df)),
        "n_real": int((df["label"] == "real").sum()),
        "n_fake": int((df["label"] == "fake").sum()),
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
    }

    for mode, grp in df.groupby("clip_capture_mode", dropna=False):
        sub = {
            "n": int(len(grp)),
            "n_real": int((grp["label"] == "real").sum()),
            "n_fake": int((grp["label"] == "fake").sum()),
            "delta_median": float(np.median(grp["delta_prob_fake"])),
            "delta_abs_median": float(np.median(np.abs(grp["delta_prob_fake"]))),
            "delta_p25": float(np.percentile(grp["delta_prob_fake"], 25)),
            "delta_p75": float(np.percentile(grp["delta_prob_fake"], 75)),
            "asis_mean_prob_fake": float(grp["prob_fake_asis"].mean()),
            "prod_mean_prob_fake": float(grp["prob_fake_prod"].mean()),
        }
        out["per_capture_mode"][str(mode)] = sub

    for lab, grp in df.groupby("label"):
        sub = {
            "n": int(len(grp)),
            "delta_median": float(np.median(grp["delta_prob_fake"])),
            "delta_abs_median": float(np.median(np.abs(grp["delta_prob_fake"]))),
            "asis_mean_prob_fake": float(grp["prob_fake_asis"].mean()),
            "prod_mean_prob_fake": float(grp["prob_fake_prod"].mean()),
        }
        out["per_label"][str(lab)] = sub

    # Per-method recall delta (fake-only)
    out["per_method_recall_delta"] = {}
    fakes_only = df[df["label"] == "fake"]
    for method, grp in fakes_only.groupby("method", dropna=False):
        per_method = {"n": int(len(grp))}
        for tau_name, tau in TAUS.items():
            asis_rec = float((grp["prob_fake_asis"] >= tau).mean())
            prod_rec = float((grp["prob_fake_prod"] >= tau).mean())
            per_method[tau_name] = {
                "asis_recall": asis_rec,
                "prod_recall": prod_rec,
                "recall_delta": prod_rec - asis_rec,
            }
        out["per_method_recall_delta"][str(method)] = per_method

    # Per-identity FPR shift (real-only) — top 10 worsening at tau_0.9741
    reals_only = df[df["label"] == "real"]
    per_id_rows = []
    for ident, grp in reals_only.groupby("identity_key", dropna=False):
        ident_row = {
            "identity_key": str(ident),
            "n_real": int(len(grp)),
        }
        for tau_name, tau in TAUS.items():
            asis_fpr = float((grp["prob_fake_asis"] >= tau).mean())
            prod_fpr = float((grp["prob_fake_prod"] >= tau).mean())
            ident_row[f"{tau_name}_asis_fpr"] = asis_fpr
            ident_row[f"{tau_name}_prod_fpr"] = prod_fpr
            ident_row[f"{tau_name}_fpr_delta"] = prod_fpr - asis_fpr
        per_id_rows.append(ident_row)
    if per_id_rows:
        per_id_df = pd.DataFrame(per_id_rows).sort_values("tau_0.9741_fpr_delta", ascending=False)
        out["per_identity_fpr_shift_top10_worsening_at_tau_0.9741"] = (
            per_id_df.head(10).to_dict(orient="records")
        )
        out["per_identity_count"] = int(len(per_id_rows))

    # Tau-level pooled FPR / Recall + per-mode FPR
    for tau_name, tau in TAUS.items():
        reals = df[df["label"] == "real"]
        fakes = df[df["label"] == "fake"]
        asis_fpr = float((reals["prob_fake_asis"] >= tau).mean()) if len(reals) else float("nan")
        prod_fpr = float((reals["prob_fake_prod"] >= tau).mean()) if len(reals) else float("nan")
        asis_rec = float((fakes["prob_fake_asis"] >= tau).mean()) if len(fakes) else float("nan")
        prod_rec = float((fakes["prob_fake_prod"] >= tau).mean()) if len(fakes) else float("nan")
        per_mode_fpr = {}
        for mode, grp in reals.groupby("clip_capture_mode", dropna=False):
            per_mode_fpr[str(mode)] = {
                "n_real": int(len(grp)),
                "asis_fpr": float((grp["prob_fake_asis"] >= tau).mean()) if len(grp) else float("nan"),
                "prod_fpr": float((grp["prob_fake_prod"] >= tau).mean()) if len(grp) else float("nan"),
                "fpr_delta": (
                    float((grp["prob_fake_prod"] >= tau).mean())
                    - float((grp["prob_fake_asis"] >= tau).mean())
                ) if len(grp) else float("nan"),
            }
        out["tau_metrics"][tau_name] = {
            "tau": tau,
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

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("MPS not available, using CPU")

    df_in, hygiene = load_lockbox()
    n_in = len(df_in)
    if n_in == 0:
        logger.error("No frames passed hygiene; aborting.")
        sys.exit(1)

    model = load_model(device)

    rows: List[dict] = []
    asis_tensors: List[torch.Tensor] = []
    prod_tensors: List[torch.Tensor] = []

    t0 = time.time()
    for i, row in df_in.iterrows():
        try:
            img = Image.open(row["local_path"]).convert("RGB")
        except Exception as e:
            logger.warning("Cannot open %s: %s", row["local_path"], e)
            continue
        asis_t = asis_to_tensor(img)
        prod_t, debug = production_crop_to_tensor(
            img, row["face_bbox_x"], row["face_bbox_y"],
            row["face_bbox_w"], row["face_bbox_h"],
        )
        asis_tensors.append(asis_t)
        prod_tensors.append(prod_t)
        rows.append({
            "frame_id": Path(row["local_path"]).stem,
            "original_local_path": row["local_path"],
            "label": row["label"],
            "split": row["split"],
            "method": row.get("method"),
            "identity_key": row.get("identity_key"),
            "session_id": row.get("session_id"),
            "video_id": row.get("video_id"),
            "clip_capture_mode": row.get("clip_capture_mode"),
            "clip_capture_mode_prob": row.get("clip_capture_mode_prob"),
            "face_bbox_x": float(row["face_bbox_x"]),
            "face_bbox_y": float(row["face_bbox_y"]),
            "face_bbox_w": float(row["face_bbox_w"]),
            "face_bbox_h": float(row["face_bbox_h"]),
            "img_w": int(debug["img_w"]),
            "img_h": int(debug["img_h"]),
            "native_face_area_ratio": debug.get("native_face_area_ratio"),
            "realized_face_area_ratio_at_rfa_0.85": debug.get("realized_face_area_ratio_at_rfa_0.85"),
            "crop_side_target": debug.get("crop_side_target"),
            "crop_box_clamped": debug.get("crop_box_clamped"),
            "clamped_to_edge": debug.get("clamped_to_edge"),
            "fallback_asis": debug.get("fallback_asis", False),
            "csv_prob_fake": row.get("prob_fake"),  # parquet's stale upstream score
        })
        if (i + 1) % 100 == 0:
            logger.info("Built %d/%d (asis,prod) pairs", i + 1, n_in)
    logger.info("Built %d (asis, prod) pairs in %.1fs", len(asis_tensors), time.time() - t0)

    # Score both arms
    asis_probs: List[float] = []
    prod_probs: List[float] = []
    t0 = time.time()
    n = len(asis_tensors)
    for i in range(0, n, BATCH):
        asis_probs.extend(score_batch(model, asis_tensors[i:i + BATCH], device).tolist())
        prod_probs.extend(score_batch(model, prod_tensors[i:i + BATCH], device).tolist())
        if (i // BATCH) % 5 == 0:
            done = min(i + BATCH, n)
            elapsed = time.time() - t0
            eta = elapsed / max(done, 1) * (n - done)
            logger.info("Scored %d/%d (ETA %.0fs)", done, n, eta)
    logger.info("Scoring done in %.1fs", time.time() - t0)

    for i, m in enumerate(rows):
        m["prob_fake_asis"] = float(asis_probs[i])
        m["prob_fake_prod"] = float(prod_probs[i])
        m["delta_prob_fake"] = float(prod_probs[i] - asis_probs[i])

    df_out = pd.DataFrame(rows)
    parquet_path = OUT / "eval_substrate_v3_retag.parquet"
    csv_path = OUT / "eval_substrate_v3_retag.csv"
    df_out.to_parquet(parquet_path, index=False)
    df_out.to_csv(csv_path, index=False)
    logger.info("Saved %s and %s", parquet_path, csv_path)

    # Aggregate metrics
    agg = compute_aggregate_metrics(df_out)
    agg["hygiene"] = hygiene
    agg["checkpoint"] = CKPT.name
    agg["production_crop_spec"] = {
        "target_rfa": TARGET_RFA,
        "source": "recrop_dataset.py:39",
        "convention": "square crop centered on face bbox; side=sqrt(face_area/TARGET_RFA); resize to 224x224 (AREA)",
        "bbox_source": "MediaPipe FaceMesh landmark bbox in lockbox_tagging parquet (face_bbox_x/y/w/h)",
    }

    json_path = OUT / "aggregate_metrics_v3_retag.json"
    with open(json_path, "w") as f:
        json.dump(agg, f, indent=2)
    logger.info("Saved %s", json_path)

    # Stdout summary
    print()
    print("=" * 70)
    print("PHASE 2B v3-RETAG SUMMARY")
    print("=" * 70)
    print(f"Lockbox frames scored : {len(df_out)} (real {agg['n_real']} / fake {agg['n_fake']})")
    print(f"Production target RFA : {TARGET_RFA} (recrop_dataset.py:39)")
    d = agg["delta_prob_fake_distribution"]
    print(f"Delta prob_fake median {d['median']:+.4f}  p25 {d['p25']:+.4f}  p75 {d['p75']:+.4f}")
    print(f"           |Delta| median {d['abs_median']:.4f}  p75 {d['abs_p75']:.4f}  p90 {d['abs_p90']:.4f}")
    print(f"           frames with |Delta|>0.10 : {d['n_frames_delta_gt_0.10']}/{len(df_out)}")
    print()
    print("Per capture mode (sorted by n):")
    print(f"{'mode':<20s} {'n':>5s} {'real':>5s} {'fake':>5s} {'Dmed':>9s} {'|D|med':>9s} "
          f"{'asis<f>':>9s} {'prod<f>':>9s}")
    pm = sorted(agg["per_capture_mode"].items(), key=lambda x: -x[1]["n"])
    for mode, sub in pm:
        print(f"{mode:<20s} {sub['n']:>5d} {sub['n_real']:>5d} {sub['n_fake']:>5d} "
              f"{sub['delta_median']:>+9.4f} {sub['delta_abs_median']:>9.4f} "
              f"{sub['asis_mean_prob_fake']:>9.4f} {sub['prod_mean_prob_fake']:>9.4f}")
    print()
    print("Tau-FPR / Recall (real vs fake):")
    for tau_name, info in agg["tau_metrics"].items():
        print(f"  {tau_name} (tau={info['tau']:.4f}): "
              f"n_real={info['n_real']} n_fake={info['n_fake']}")
        print(f"    FPR    asis={info['asis_fpr']:.4f}  prod={info['prod_fpr']:.4f}  "
              f"D={info['fpr_delta']:+.4f}")
        print(f"    Recall asis={info['asis_recall']:.4f}  prod={info['prod_recall']:.4f}  "
              f"D={info['recall_delta']:+.4f}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
