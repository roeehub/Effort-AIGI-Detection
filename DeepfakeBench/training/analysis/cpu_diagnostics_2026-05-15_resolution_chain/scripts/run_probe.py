"""Resolution-chain sensitivity probe (2026-05-15).

Tests whether models give consistent scores under downsample->upsample chains
that simulate the production behaviour the user reported: same face content at
different screen capture sizes produces wildly different scores.

Method:
  For each of 388 cohort frames (5 cohorts spanning chronic-6 + healthy +
  visomaster_enhanced fakes), produce 21 variants:
    1 baseline (cv2 INTER_LINEAR resize from native crop to 224, the canonical
                preprocessing path used by the scorecard)
    5 downsample sizes (64, 96, 128, 160, 192) x 4 upsample kernels
                       (LINEAR, CUBIC, AREA, LANCZOS4) = 20 variants

  Then score every variant on T5C step3500, P8A step5000, E2B step3200 via
  the full detector forward (gets prob_fake from the head).

Outputs:
  outputs/per_frame_per_variant.parquet
    columns: frame_path, cohort, identity, label,
             down_size (int or NaN for baseline),
             kernel (str or 'baseline'),
             ckpt (str), score (float)
  outputs/per_frame_summary.parquet
    columns: frame_path, cohort, identity, label, ckpt,
             score_baseline, score_min, score_max, score_std,
             score_range = max - min,
             res_corr  = Pearson r(score, down_size)
"""
from __future__ import annotations
import os
import sys
import logging
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
OUT_DIR = PROBE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = PROBE_DIR / "_run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("resolution_chain")

sys.path.insert(0, str(REPO_ROOT))

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

DETECTOR_CONFIG = REPO_ROOT / "config" / "detector" / "effort.yaml"
TRAIN_CONFIG = REPO_ROOT / "config" / "train_config.yaml"

COHORT_MANIFEST = REPO_ROOT / "analysis" / "dor_encoder_axis_2026-05-08" / "_cache" / "cohort_manifest.csv"
FRAMES_DIR = REPO_ROOT / "analysis" / "dor_encoder_axis_2026-05-08" / "_cache" / "frames"

CKPTS: Dict[str, Path] = {
    "P8A_step5000": REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
                    / "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth",
    "T5C_step3500": PROBE_DIR / "_cache" / "periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth",
    "E2B_step3200": PROBE_DIR / "_cache" / "top_n_effort_20260503_step3200_auc0.9863_eer0.0310.pth",
}

DOWN_SIZES = [64, 96, 128, 160, 192]
KERNELS: Dict[str, int] = {
    "LINEAR": cv2.INTER_LINEAR,
    "CUBIC": cv2.INTER_CUBIC,
    "AREA": cv2.INTER_AREA,
    "LANCZOS4": cv2.INTER_LANCZOS4,
}

BATCH_SIZE = 16
TARGET_RES = 224


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
    """img_224 is BGR uint8 of shape (224,224,3). Return CHW float32 tensor."""
    rgb = cv2.cvtColor(img_224, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - CLIP_MEAN) / CLIP_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).contiguous()


def make_variants(img_native_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Produce 21 variants per native crop.

    - baseline: cv2.INTER_LINEAR resize directly from native to 224
    - 5 sizes x 4 kernels: downsample to size with given kernel, then
      upsample back to 224 with that SAME kernel (so the chain stays
      kernel-consistent; if we mixed kernels we'd attribute to the wrong axis)
    """
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
    # extract identity from local file name (the dor_encoder script saved frames
    # with original basename, sometimes prefixed by an identity hash). Match by
    # basename to the local _cache/frames dir.
    local_paths = []
    for gcs_path in df["frame_path"]:
        # The local cache uses the basename of the gcs URI (possibly with a
        # hash suffix on .jpg from team_sanity_check). Most are direct basename.
        base = Path(gcs_path).name
        local = FRAMES_DIR / base
        if not local.exists():
            # Try permissive match by stem.
            candidates = list(FRAMES_DIR.glob(f"{Path(base).stem}*"))
            local = candidates[0] if candidates else None
        local_paths.append(str(local) if local else "")
    df["local_path"] = local_paths
    n_missing = (df["local_path"] == "").sum()
    logger.info("cohort manifest: %d rows, %d missing local frames", len(df), n_missing)
    df = df[df["local_path"] != ""].reset_index(drop=True)
    # parse identity from filename
    df["identity"] = df["local_path"].apply(
        lambda p: Path(p).name.split("__")[0].split("_real_")[0].split("_frame_")[0]
    )
    return df


def score_one_ckpt(
    ckpt_label: str,
    ckpt_path: Path,
    frame_tensors_by_variant: Dict[str, List[torch.Tensor]],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Returns {variant_key: np.ndarray of scores, shape (n_frames,)}."""
    logger.info("loading %s from %s", ckpt_label, ckpt_path.name)
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
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores_by_variant


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    logger.info("device=%s", device)

    df = load_cohort()
    n_frames = len(df)
    logger.info("loaded %d cohort frames", n_frames)

    # Preload+preprocess all variants once into memory (21 * 388 = 8148 tensors
    # at 224x224x3 float32 = ~5GB; OK on this Mac).
    logger.info("preprocessing 21 variants per frame (%d total tensors)...", n_frames * 21)
    frame_tensors_by_variant: Dict[str, List[torch.Tensor]] = {}
    for idx, row in df.iterrows():
        img = cv2.imread(row["local_path"], cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("imread failed for %s, skipping", row["local_path"])
            continue
        variants = make_variants(img)
        for variant_key, img_224 in variants:
            tensor = normalize_to_tensor(img_224)
            frame_tensors_by_variant.setdefault(variant_key, []).append(tensor)
        if (idx + 1) % 50 == 0:
            logger.info("  preprocessed %d/%d frames", idx + 1, n_frames)

    logger.info("preprocessing done. variants: %d", len(frame_tensors_by_variant))

    # Score each ckpt on every variant.
    all_rows: List[dict] = []
    for ckpt_label, ckpt_path in CKPTS.items():
        scores_by_variant = score_one_ckpt(ckpt_label, ckpt_path, frame_tensors_by_variant, device)
        for variant_key, scores in scores_by_variant.items():
            if "|" in variant_key:
                size_str, kernel = variant_key.split("|")
                down_size = int(size_str) if size_str != "baseline" else -1
            else:
                size_str, kernel = variant_key, "LINEAR"
                down_size = -1
            for i, row in df.iterrows():
                if i >= len(scores):
                    break
                all_rows.append({
                    "frame_path": row["frame_path"],
                    "local_path": row["local_path"],
                    "cohort": row["cohort"],
                    "identity": row["identity"],
                    "label": int(row["label"]),
                    "down_size": down_size,
                    "kernel": kernel,
                    "variant_key": variant_key,
                    "ckpt": ckpt_label,
                    "score": float(scores[i]),
                })

    full = pd.DataFrame(all_rows)
    full_path = OUT_DIR / "per_frame_per_variant.parquet"
    full.to_parquet(full_path, index=False)
    logger.info("wrote %s (%d rows)", full_path, len(full))

    # Summary: per (frame, ckpt) stats across the 20 perturbed variants
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
            "frame_path": frame_path,
            "ckpt": ckpt,
            "cohort": meta["cohort"],
            "identity": meta["identity"],
            "label": int(meta["label"]),
            "score_baseline": float(baseline),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std(ddof=0)),
            "score_range": float(scores.max() - scores.min()),
            "res_corr": res_corr,
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "per_frame_summary.parquet"
    summary.to_parquet(summary_path, index=False)
    logger.info("wrote %s (%d rows)", summary_path, len(summary))

    logger.info("DONE")


if __name__ == "__main__":
    main()
