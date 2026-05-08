"""Per-layer CLS feature extraction for the IQ-perlayer-probe (check (a)).

Extracts CLS-token features at OpenCLIP B16 transformer.resblocks layers
{0, 3, 6, 9, 11} for the three target ckpts:
  - P8A_REFERENCE (already cached at analysis/_features_cache_2026-04-30/intermediate__P8A__layer*__n800.npz)
  - E2B_TOP_N_STEP3200 (downloaded to ./_cache/e2b_top_n_step3200.pth)
  - P2_D_FOURIER_PERIODIC_STEP3000 (existing local at analysis/p2_eval_2026-05-08/d1_d4_cpu/ckpts/slotD_periodic_step3000.pth)

Frame set: the 800-frame triptych sample
(`analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/sampled_frames.csv`),
which is mostly DEV-split lockbox-substrate frames (713 dev / 87 lockbox)
covering 30+ identities including 98 Dor frames.

Reuses the per-layer extraction logic from
`analysis/intermediate_layer_probe_2026-04-30/intermediate_layer_probe.py`,
including the OpenCLIP resblocks hook and CLIP normalization.

CPU-only by policy (no GPU spend authorized for these checks).
Auto-detects MPS for ~5x speedup on M-series Mac. Idempotent: per-layer cache
files at `_cache/intermediate__{ckpt_label}__layer{ix:02d}__n{n}.npz` skip
re-extraction.

Usage:
    python3 analysis/iq_perlayer_probe_2026-05-08/extract_features.py \\
        --layers 0,3,6,9,11 --device mps
"""
from __future__ import annotations

import argparse
import logging
import shutil
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

THIS_DIR = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = THIS_DIR / "_cache"
PRIOR_CACHE_DIR = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
SAMPLED_CSV = (
    REPO_ROOT
    / "analysis"
    / "embedding_triptych_2026-04-30"
    / "outputs"
    / "triptych_p8a_slot2_slot3"
    / "sampled_frames.csv"
)
DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "config" / "detector" / "effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config" / "train_config.yaml"

# Target ckpts. Each maps to absolute pth path on disk.
TARGET_CKPTS: Dict[str, Path] = {
    "P8A_REFERENCE_STEP5000": (
        PRIOR_CACHE_DIR / "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth"
    ),
    "E2B_TOP_N_STEP3200": (LOCAL_CACHE_DIR / "e2b_top_n_step3200.pth"),
    "P2_D_FOURIER_PERIODIC_STEP3000": (
        REPO_ROOT
        / "analysis"
        / "p2_eval_2026-05-08"
        / "d1_d4_cpu"
        / "ckpts"
        / "slotD_periodic_step3000.pth"
    ),
}

# Short labels used in cache filename.
LABELS = {
    "P8A_REFERENCE_STEP5000": "P8A",
    "E2B_TOP_N_STEP3200": "E2B",
    "P2_D_FOURIER_PERIODIC_STEP3000": "P2D",
}

logger = logging.getLogger("iq-perlayer-extract")


# -----------------------------------------------------------------------------
# Model build + checkpoint load.
# -----------------------------------------------------------------------------
def load_effort_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Build an EffortDetector with the canonical config + load state."""
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
    """Locate the OpenCLIP transformer.resblocks under the wrapper."""
    if hasattr(model.backbone, "visual"):
        visual = model.backbone.visual
    else:
        visual = model.backbone
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        return visual.transformer.resblocks
    raise RuntimeError("Could not find transformer.resblocks under model.backbone(.visual)")


# -----------------------------------------------------------------------------
# Image loading + preprocessing.
# -----------------------------------------------------------------------------
def load_and_preprocess(local_path: Path, resolution: int = 224) -> torch.Tensor | None:
    import cv2

    img = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1))


# -----------------------------------------------------------------------------
# Per-layer feature extraction with hooks.
# -----------------------------------------------------------------------------
def extract_per_layer_features(
    model: torch.nn.Module,
    paths: List[str],
    layers: List[int],
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Returns (per_layer_features, valid_idx_into_input_paths)."""
    resblocks = get_resblocks(model)
    if max(layers) >= len(resblocks):
        raise ValueError(
            f"requested layer {max(layers)} but resblocks length is {len(resblocks)}"
        )

    captured: Dict[int, List[np.ndarray]] = {ix: [] for ix in layers}

    def make_hook(ix: int):
        def hook(_module, _input, output):
            if output.dim() == 3:
                if output.shape[0] >= output.shape[1]:
                    cls = output[0]
                else:
                    cls = output[:, 0]
            elif output.dim() == 2:
                cls = output
            else:
                raise RuntimeError(f"unexpected output shape from resblock: {tuple(output.shape)}")
            captured[ix].append(cls.detach().cpu().to(torch.float32).numpy())

        return hook

    handles = [resblocks[ix].register_forward_hook(make_hook(ix)) for ix in layers]
    try:
        valid: List[int] = []
        pending: List[Tuple[int, torch.Tensor]] = []
        for i, p in enumerate(paths):
            t = load_and_preprocess(Path(p))
            if t is None:
                continue
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
                logger.info("  processed batch %d/%d", j // batch_size + 1, (len(pending) + batch_size - 1) // batch_size)
        for ix in layers:
            captured[ix] = (
                np.concatenate(captured[ix], axis=0)
                if captured[ix]
                else np.zeros((0, 0), dtype=np.float32)
            )
    finally:
        for h in handles:
            h.remove()
    return captured, np.array(valid, dtype=np.int64)


def cache_path_for_layer(label: str, layer_ix: int, n_samples: int) -> Path:
    return LOCAL_CACHE_DIR / f"intermediate__{label}__layer{layer_ix:02d}__n{n_samples}.npz"


def prior_cache_path_for_layer(label: str, layer_ix: int, n_samples: int) -> Path:
    return PRIOR_CACHE_DIR / f"intermediate__{label}__layer{layer_ix:02d}__n{n_samples}.npz"


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-layer feature extraction for IQ probe (check a)")
    ap.add_argument("--layers", default="0,3,6,9,11")
    ap.add_argument("--max_frames", type=int, default=800)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpts", default=None,
                    help="comma-separated subset of ckpts to extract; default = all 3")
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    layers = sorted({int(x) for x in args.layers.split(",") if x.strip()})

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("device=%s, layers=%s, max_frames=%d", device, layers, args.max_frames)

    if not SAMPLED_CSV.exists():
        logger.error("Sampled-frames CSV missing: %s", SAMPLED_CSV)
        return 2
    sampled = pd.read_csv(SAMPLED_CSV)
    sampled = sampled.iloc[: args.max_frames].reset_index(drop=True)
    paths = sampled["local_path"].tolist()
    n_samples = len(paths)
    logger.info("loaded %d frames", n_samples)

    if args.ckpts:
        wanted = [c.strip() for c in args.ckpts.split(",") if c.strip()]
        ckpts = {k: v for k, v in TARGET_CKPTS.items() if k in wanted}
    else:
        ckpts = TARGET_CKPTS

    for ckpt_key, ckpt_path in ckpts.items():
        label = LABELS[ckpt_key]
        # First check prior cache (for P8A); copy if found.
        all_layers_cached_locally = all(
            cache_path_for_layer(label, ix, n_samples).exists() for ix in layers
        )
        if all_layers_cached_locally and not args.no_cache:
            logger.info("[%s] all layers cached locally", label)
            continue
        all_layers_in_prior = all(
            prior_cache_path_for_layer(label, ix, n_samples).exists() for ix in layers
        )
        if all_layers_in_prior and not args.no_cache:
            logger.info("[%s] all layers in prior cache; copying to local", label)
            for ix in layers:
                src = prior_cache_path_for_layer(label, ix, n_samples)
                dst = cache_path_for_layer(label, ix, n_samples)
                shutil.copy(src, dst)
            continue

        if not ckpt_path.exists():
            logger.error("ckpt missing: %s", ckpt_path)
            return 2

        logger.info("[%s] loading model from %s", label, ckpt_path)
        model = load_effort_model(ckpt_path, device)
        layer_feats, valid_idx = extract_per_layer_features(
            model, paths, layers, device, batch_size=args.batch_size
        )
        for ix in layers:
            np.savez_compressed(
                cache_path_for_layer(label, ix, n_samples),
                features=layer_feats[ix].astype(np.float32),
                valid_idx=valid_idx,
            )
        logger.info("[%s] cached %d layers (%d frames)", label, len(layers), n_samples)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
