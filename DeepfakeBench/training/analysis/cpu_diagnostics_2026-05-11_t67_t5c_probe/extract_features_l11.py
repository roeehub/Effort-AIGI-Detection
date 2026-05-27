"""Extract L11-only CLS features for T6/T7/T5C ckpts on the 800-frame triptych.

Time-budget-aware variant of extract_t4_features.py: extracts ONLY layer 11
since that is the load-bearing layer for inv_mean. Skips L0/L3/L6/L9 to save
~5× wall-clock.

Writes feature .npz into iq_perlayer_probe_2026-05-08/_cache/ so
extend_atlas with the existing pipeline still works.
"""
from __future__ import annotations

import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

CACHE_DIR = REPO_ROOT / "analysis" / "iq_perlayer_probe_2026-05-08" / "_cache"
SAMPLED_CSV = (
    REPO_ROOT / "analysis" / "embedding_triptych_2026-04-30"
    / "outputs" / "triptych_p8a_slot2_slot3" / "sampled_frames.csv"
)
DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "config" / "detector" / "effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config" / "train_config.yaml"

CKPT_BASE = Path(__file__).resolve().parent / "_ckpts"

# Candidates - all 22 + 3 anchors.
TARGET_CKPTS: Dict[str, Path] = {
    # T6
    "T6_periodic_step500":  CKPT_BASE / "t6" / "periodic_effort_20260511_step500_auc0.9780_eer0.0520.pth",
    "T6_periodic_step1500": CKPT_BASE / "t6" / "periodic_effort_20260511_step1500_auc0.9903_eer0.0340.pth",
    "T6_periodic_step2500": CKPT_BASE / "t6" / "periodic_effort_20260511_step2500_auc0.9926_eer0.0240.pth",
    "T6_periodic_step3500": CKPT_BASE / "t6" / "periodic_effort_20260511_step3500_auc0.9918_eer0.0220.pth",
    "T6_periodic_step4500": CKPT_BASE / "t6" / "periodic_effort_20260511_step4500_auc0.9940_eer0.0200.pth",
    "T6_top_n_step3250":    CKPT_BASE / "t6" / "top_n_effort_20260511_step3250_auc0.9940_eer0.0240.pth",
    "T6_top_n_step6000":    CKPT_BASE / "t6" / "top_n_effort_20260511_step6000_auc0.9943_eer0.0200.pth",
    "T6_top_n_step10250":   CKPT_BASE / "t6" / "top_n_effort_20260511_step10250_auc0.9972_eer0.0180.pth",
    # T7
    "T7_periodic_step500":  CKPT_BASE / "t7" / "periodic_effort_20260511_step500_auc0.9782_eer0.0787.pth",
    "T7_periodic_step1500": CKPT_BASE / "t7" / "periodic_effort_20260511_step1500_auc0.9857_eer0.0476.pth",
    "T7_periodic_step2500": CKPT_BASE / "t7" / "periodic_effort_20260511_step2500_auc0.9929_eer0.0311.pth",
    "T7_periodic_step3500": CKPT_BASE / "t7" / "periodic_effort_20260511_step3500_auc0.9874_eer0.0497.pth",
    "T7_periodic_step5000": CKPT_BASE / "t7" / "periodic_effort_20260511_step5000_auc0.9946_eer0.0228.pth",
    "T7_top_n_step4250":    CKPT_BASE / "t7" / "top_n_effort_20260511_step4250_auc0.9944_eer0.0248.pth",
    "T7_top_n_step4750":    CKPT_BASE / "t7" / "top_n_effort_20260511_step4750_auc0.9956_eer0.0207.pth",
    # T5C
    "T5C_periodic_step500":  CKPT_BASE / "t5c" / "periodic_effort_20260511_step500_auc0.9894_eer0.0285.pth",
    "T5C_periodic_step1500": CKPT_BASE / "t5c" / "periodic_effort_20260511_step1500_auc0.9874_eer0.0570.pth",
    "T5C_periodic_step2500": CKPT_BASE / "t5c" / "periodic_effort_20260511_step2500_auc0.9926_eer0.0373.pth",
    "T5C_periodic_step3500": CKPT_BASE / "t5c" / "periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth",
    "T5C_periodic_step5000": CKPT_BASE / "t5c" / "periodic_effort_20260511_step5000_auc0.9937_eer0.0285.pth",
    "T5C_top_n_step2750":    CKPT_BASE / "t5c" / "top_n_effort_20260511_step2750_auc0.9940_eer0.0219.pth",
    "T5C_top_n_step3750":    CKPT_BASE / "t5c" / "top_n_effort_20260511_step3750_auc0.9948_eer0.0154.pth",
}

# Pre-cached anchors are already in iq_perlayer_probe_2026-05-08/_cache.
# P8A__layer11, E2B__layer11, T3_S1_step1500__layer11 — no need to redo.

LAYERS = [11]  # L11 only — time-budget-aware
N_FRAMES = 800
BATCH_SIZE = 16

logger = logging.getLogger("extract-t67-t5c")


def load_effort_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    import yaml
    from detectors import DETECTOR

    with open(DEFAULT_DETECTOR_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    with open(DEFAULT_TRAIN_CONFIG, "r") as f:
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
    if unexpected:
        logger.info("  dropped %d unexpected keys (e.g. %s)", len(unexpected), unexpected[:3])
    model.eval()
    return model


def get_resblocks(model: torch.nn.Module) -> torch.nn.ModuleList:
    if hasattr(model.backbone, "visual"):
        visual = model.backbone.visual
    else:
        visual = model.backbone
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        return visual.transformer.resblocks
    raise RuntimeError("Could not find transformer.resblocks")


def load_and_preprocess(local_path: Path, resolution: int = 224) -> torch.Tensor | None:
    import cv2
    img = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1))


def extract_per_layer_features(
    model, paths: List[str], layers: List[int], device: torch.device, batch_size: int = 16
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    resblocks = get_resblocks(model)
    captured: Dict[int, List[np.ndarray]] = {ix: [] for ix in layers}

    def make_hook(ix: int):
        def hook(_m, _i, output):
            if output.dim() == 3:
                cls = output[0] if output.shape[0] >= output.shape[1] else output[:, 0]
            elif output.dim() == 2:
                cls = output
            else:
                raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
            captured[ix].append(cls.detach().cpu().to(torch.float32).numpy())
        return hook

    handles = [resblocks[ix].register_forward_hook(make_hook(ix)) for ix in layers]
    try:
        valid: List[int] = []
        pending: List[Tuple[int, torch.Tensor]] = []
        for i, p in enumerate(paths):
            t = load_and_preprocess(Path(p))
            if t is not None:
                pending.append((i, t))
        logger.info("loaded %d/%d valid frames", len(pending), len(paths))

        for j in range(0, len(pending), batch_size):
            chunk = pending[j : j + batch_size]
            batch_idx = [c[0] for c in chunk]
            batch = torch.stack([c[1] for c in chunk]).to(device, non_blocking=True)
            with torch.inference_mode():
                _ = model.backbone(batch)
            valid.extend(batch_idx)
        for ix in layers:
            captured[ix] = np.concatenate(captured[ix], axis=0) if captured[ix] else np.zeros((0, 0), dtype=np.float32)
    finally:
        for h in handles:
            h.remove()
    return captured, np.array(valid, dtype=np.int64)


def cache_path(label: str, layer_ix: int, n: int) -> Path:
    return CACHE_DIR / f"intermediate__{label}__layer{layer_ix:02d}__n{n}.npz"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("device=%s", device)

    sampled = pd.read_csv(SAMPLED_CSV).iloc[: N_FRAMES].reset_index(drop=True)
    paths = sampled["local_path"].tolist()
    logger.info("loaded %d frames", len(paths))

    for label, ckpt_path in TARGET_CKPTS.items():
        all_cached = all(cache_path(label, ix, N_FRAMES).exists() for ix in LAYERS)
        if all_cached:
            logger.info("[%s] all layers cached, skipping", label)
            continue
        if not ckpt_path.exists():
            logger.warning("[%s] ckpt missing: %s", label, ckpt_path)
            continue
        if ckpt_path.stat().st_size < 100_000_000:
            logger.warning("[%s] ckpt too small (%d bytes), skipping (partial download?)",
                           label, ckpt_path.stat().st_size)
            continue
        logger.info("[%s] loading model from %s", label, ckpt_path)
        try:
            model = load_effort_model(ckpt_path, device)
        except Exception as exc:
            logger.error("[%s] load failed: %s", label, exc)
            continue
        logger.info("[%s] extracting L%s", label, LAYERS)
        feats, valid_idx = extract_per_layer_features(model, paths, LAYERS, device, BATCH_SIZE)
        for ix in LAYERS:
            np.savez_compressed(
                cache_path(label, ix, N_FRAMES),
                features=feats[ix].astype(np.float32),
                valid_idx=valid_idx,
            )
            logger.info("  saved %s (shape=%s)", cache_path(label, ix, N_FRAMES).name, feats[ix].shape)
        del model
    logger.info("done")


if __name__ == "__main__":
    raise SystemExit(main())
