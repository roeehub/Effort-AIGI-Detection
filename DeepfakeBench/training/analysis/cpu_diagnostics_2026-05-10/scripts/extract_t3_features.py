"""Extract per-layer CLS features for T3_SLOT1 step1500/step2500 on the
800-frame triptych — extends the existing forgery-signal atlas.

Reuses the iq_perlayer_probe_2026-05-08 extraction pattern.

Outputs feature .npz files into iq_perlayer_probe_2026-05-08/_cache/
to match the schema the existing run_analyses.py expects.
"""
from __future__ import annotations

import logging
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
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

T3_CKPT_DIR = REPO_ROOT / "analysis" / "cpu_diagnostics_2026-05-10" / "_ckpts_t3"

TARGET_CKPTS: Dict[str, Path] = {
    "T3_S1_step1500": T3_CKPT_DIR / "T3_SLOT1_step1500.pth",
    "T3_S1_step2500": T3_CKPT_DIR / "T3_SLOT1_step2500.pth",
}

LAYERS = [0, 3, 6, 9, 11]
N_FRAMES = 800
BATCH_SIZE = 16

logger = logging.getLogger("extract-t3")


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
    model.load_state_dict(clean_state, strict=False)
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
            if (j // batch_size) % 5 == 0:
                logger.info("  batch %d/%d", j // batch_size + 1, (len(pending) + batch_size - 1) // batch_size)
        for ix in layers:
            captured[ix] = np.concatenate(captured[ix], axis=0) if captured[ix] else np.zeros((0,0), dtype=np.float32)
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
        logger.info("[%s] loading model from %s", label, ckpt_path)
        model = load_effort_model(ckpt_path, device)
        logger.info("[%s] extracting layers %s", label, LAYERS)
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
