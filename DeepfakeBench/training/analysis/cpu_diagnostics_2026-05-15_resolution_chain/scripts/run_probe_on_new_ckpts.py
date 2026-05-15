"""Re-run the resolution-chain probe on the overnight 2026-05-15 ckpts.

Same panel, same 21 variants per frame, same scoring infra as `run_probe.py`.
Difference: scores the two new ckpts (Slot α + Slot β) and writes their
per-frame summary into a separate output dir so the original 3-ckpt baseline
is preserved.

Usage (after overnight training jobs complete):
  python run_probe_on_new_ckpts.py \
      --slot-a-ckpt gs://<run_id>/<latest_periodic>.pth \
      --slot-b-ckpt gs://<run_id>/<latest_periodic>.pth

The script downloads ckpts to _cache/ if not already local, then scores.
"""
from __future__ import annotations
import argparse
import os
import sys
import logging
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
import cv2

THIS_DIR = Path(__file__).resolve().parent
PROBE_DIR = THIS_DIR.parent
REPO_ROOT = PROBE_DIR.parent.parent
OUT_DIR = PROBE_DIR / "outputs_new_ckpts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = PROBE_DIR / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = PROBE_DIR / "_run_new_ckpts.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("resolution_chain_new")

sys.path.insert(0, str(REPO_ROOT))

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
DETECTOR_CONFIG = REPO_ROOT / "config" / "detector" / "effort.yaml"
TRAIN_CONFIG = REPO_ROOT / "config" / "train_config.yaml"
COHORT_MANIFEST = REPO_ROOT / "analysis" / "dor_encoder_axis_2026-05-08" / "_cache" / "cohort_manifest.csv"
FRAMES_DIR = REPO_ROOT / "analysis" / "dor_encoder_axis_2026-05-08" / "_cache" / "frames"

DOWN_SIZES = [64, 96, 128, 160, 192]
KERNELS = {
    "LINEAR": cv2.INTER_LINEAR,
    "CUBIC": cv2.INTER_CUBIC,
    "AREA": cv2.INTER_AREA,
    "LANCZOS4": cv2.INTER_LANCZOS4,
}
BATCH_SIZE = 16
TARGET_RES = 224


def gcs_download(gcs_uri: str) -> Path:
    local = CACHE_DIR / Path(gcs_uri).name
    if local.exists():
        logger.info("ckpt already cached: %s", local)
        return local
    logger.info("downloading %s → %s", gcs_uri, local)
    subprocess.check_call(["gsutil", "cp", gcs_uri, str(local)])
    return local


def load_effort_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    from detectors import DETECTOR
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
    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)
    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])
    clean_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing or unexpected:
        logger.info("    load_state_dict: %d missing, %d unexpected", len(missing), len(unexpected))
    model.eval()
    return model


def normalize_to_tensor(img_224: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_224, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - CLIP_MEAN) / CLIP_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).contiguous()


def make_variants(img_native_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    out: List[Tuple[str, np.ndarray]] = []
    base = cv2.resize(img_native_bgr, (TARGET_RES, TARGET_RES), interpolation=cv2.INTER_LINEAR)
    out.append(("baseline|LINEAR", base))
    for size in DOWN_SIZES:
        for k_name, k_cv in KERNELS.items():
            down = cv2.resize(img_native_bgr, (size, size), interpolation=k_cv)
            up = cv2.resize(down, (TARGET_RES, TARGET_RES), interpolation=k_cv)
            out.append((f"{size}|{k_name}", up))
    return out


def load_cohort() -> pd.DataFrame:
    df = pd.read_csv(COHORT_MANIFEST)
    local_paths = []
    for gcs_path in df["frame_path"]:
        base = Path(gcs_path).name
        local = FRAMES_DIR / base
        if not local.exists():
            candidates = list(FRAMES_DIR.glob(f"{Path(base).stem}*"))
            local = candidates[0] if candidates else None
        local_paths.append(str(local) if local else "")
    df["local_path"] = local_paths
    df = df[df["local_path"] != ""].reset_index(drop=True)
    df["identity"] = df["local_path"].apply(
        lambda p: Path(p).name.split("__")[0].split("_real_")[0].split("_frame_")[0]
    )
    return df


def score_one_ckpt(label: str, ckpt_path: Path, frame_tensors_by_variant: Dict, device: torch.device):
    logger.info("loading %s from %s", label, ckpt_path.name)
    model = load_effort_model(ckpt_path, device)
    scores_by_variant: Dict[str, np.ndarray] = {}
    for variant_key, tensors in frame_tensors_by_variant.items():
        n = len(tensors)
        scores = np.zeros(n, dtype=np.float32)
        for j in range(0, n, BATCH_SIZE):
            batch = torch.stack(tensors[j : j + BATCH_SIZE]).to(device, non_blocking=True)
            with torch.inference_mode():
                pred = model({"image": batch}, inference=True)
            probs = pred["prob"].detach().cpu().to(torch.float32).numpy()
            scores[j : j + len(probs)] = probs
        scores_by_variant[variant_key] = scores
    del model
    return scores_by_variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot-a-ckpt", required=True, help="GCS URI to Slot α ckpt")
    ap.add_argument("--slot-b-ckpt", required=True, help="GCS URI to Slot β ckpt")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("device=%s", device)

    slot_a_local = gcs_download(args.slot_a_ckpt)
    slot_b_local = gcs_download(args.slot_b_ckpt)

    df = load_cohort()
    logger.info("loaded %d cohort frames", len(df))

    frame_tensors_by_variant: Dict[str, List[torch.Tensor]] = {}
    for idx, row in df.iterrows():
        img = cv2.imread(row["local_path"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        for variant_key, img_224 in make_variants(img):
            frame_tensors_by_variant.setdefault(variant_key, []).append(normalize_to_tensor(img_224))

    ckpts = {"SLOT_A_RESCHAIN": slot_a_local, "SLOT_B_6AXIS_GRL": slot_b_local}
    all_rows: List[dict] = []
    for label, path in ckpts.items():
        scores_by_variant = score_one_ckpt(label, path, frame_tensors_by_variant, device)
        for variant_key, scores in scores_by_variant.items():
            size_str, kernel = variant_key.split("|")
            down_size = int(size_str) if size_str != "baseline" else -1
            for i, row in df.iterrows():
                if i >= len(scores):
                    break
                all_rows.append({
                    "frame_path": row["frame_path"],
                    "cohort": row["cohort"],
                    "identity": row["identity"],
                    "label": int(row["label"]),
                    "down_size": down_size,
                    "kernel": kernel,
                    "variant_key": variant_key,
                    "ckpt": label,
                    "score": float(scores[i]),
                })

    full = pd.DataFrame(all_rows)
    full.to_parquet(OUT_DIR / "per_frame_per_variant_new_ckpts.parquet", index=False)
    logger.info("wrote new ckpts grid (%d rows)", len(full))

    perturbed = full[full["down_size"] != -1].copy()
    summary_rows = []
    for (frame_path, ckpt), g in perturbed.groupby(["frame_path", "ckpt"]):
        scores = g["score"].values
        sizes = g["down_size"].values.astype(np.float32)
        baseline = full[(full["frame_path"] == frame_path) & (full["ckpt"] == ckpt)
                        & (full["down_size"] == -1)]["score"].iloc[0]
        meta = g.iloc[0]
        res_corr = float(np.corrcoef(sizes, scores)[0, 1]) if len(set(sizes)) > 1 else np.nan
        summary_rows.append({
            "frame_path": frame_path, "ckpt": ckpt,
            "cohort": meta["cohort"], "identity": meta["identity"], "label": int(meta["label"]),
            "score_baseline": float(baseline),
            "score_min": float(scores.min()), "score_max": float(scores.max()),
            "score_mean": float(scores.mean()), "score_std": float(scores.std(ddof=0)),
            "score_range": float(scores.max() - scores.min()), "res_corr": res_corr,
        })
    pd.DataFrame(summary_rows).to_parquet(OUT_DIR / "per_frame_summary_new_ckpts.parquet", index=False)
    logger.info("DONE")


if __name__ == "__main__":
    main()
