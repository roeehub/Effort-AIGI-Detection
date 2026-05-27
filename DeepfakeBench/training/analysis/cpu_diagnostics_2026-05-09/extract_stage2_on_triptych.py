"""Extract Stage 2 (S1/S2/S3 step{500,2500,4500}) features + scores on the
800-frame triptych sample. Combines two passes (scoring + per-layer feature
extraction) into one forward-pass loop per ckpt.

Outputs:
  outputs/stage2_triptych_scores.csv
    columns: row_ix, gcs_uri, label, split, identity_key, method,
             {S1|S2|S3}_step{500|2500|4500}  (fake-class softmax prob)

  _cache/intermediate__{S1|S2|S3}_step{500|2500|4500}__layer{XX}__n800.npz
    keys: features (n,768), valid_idx (n,)
    matches the schema the existing iq_perlayer_probe scripts expect.

Reuses logic from analysis/iq_perlayer_probe_2026-05-08/extract_features.py
and analysis/stage2_cpu_2026-05-09/run_score_probe.py.
"""
from __future__ import annotations

import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

THIS_DIR = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = THIS_DIR / "_cache"
OUTPUTS = THIS_DIR / "outputs"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

STAGE2_CKPT_DIR = REPO_ROOT / "analysis" / "stage2_cpu_2026-05-09" / "_ckpts"
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

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
LAYERS = [0, 3, 6, 9, 11]

CKPTS_TO_EXTRACT: Dict[str, Path] = {
    "S1_step500": STAGE2_CKPT_DIR / "S1_step500.pth",
    "S1_step2500": STAGE2_CKPT_DIR / "S1_step2500.pth",
    "S1_step4500": STAGE2_CKPT_DIR / "S1_step4500.pth",
    "S2_step500": STAGE2_CKPT_DIR / "S2_step500.pth",
    "S2_step2500": STAGE2_CKPT_DIR / "S2_step2500.pth",
    "S2_step4500": STAGE2_CKPT_DIR / "S2_step4500.pth",
    "S3_step500": STAGE2_CKPT_DIR / "S3_step500.pth",
    "S3_step2500": STAGE2_CKPT_DIR / "S3_step2500.pth",
    "S3_step4500": STAGE2_CKPT_DIR / "S3_step4500.pth",
}

logger = logging.getLogger("stage2-triptych-extract")


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
    raise RuntimeError("Could not find transformer.resblocks under model.backbone(.visual)")


def load_and_preprocess(local_path: Path, resolution: int = 224) -> torch.Tensor | None:
    import cv2
    img = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1))


def cache_path_for_layer(label: str, layer_ix: int, n_samples: int) -> Path:
    return LOCAL_CACHE_DIR / f"intermediate__{label}__layer{layer_ix:02d}__n{n_samples}.npz"


def extract_per_ckpt(
    ckpt_label: str,
    ckpt_path: Path,
    paths: List[Path],
    device: torch.device,
    layers: List[int],
    batch_size: int = 16,
    n_samples: int = 800,
) -> Tuple[np.ndarray, Dict[int, np.ndarray], np.ndarray]:
    """Returns (scores, per_layer_features, valid_idx).

    scores shape: (n_samples,)
    per_layer_features[ix] shape: (n_valid, dim)
    valid_idx shape: (n_valid,)
    """
    model = load_effort_model(ckpt_path, device)
    resblocks = get_resblocks(model)

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
    scores = np.full(n_samples, np.nan, dtype=np.float64)
    valid: List[int] = []

    try:
        # Pre-load all valid frames.
        pending: List[Tuple[int, torch.Tensor]] = []
        for i, p in enumerate(paths):
            t = load_and_preprocess(Path(p))
            if t is not None:
                pending.append((i, t))
        logger.info("[%s] valid frames: %d/%d", ckpt_label, len(pending), len(paths))

        for j in range(0, len(pending), batch_size):
            chunk = pending[j : j + batch_size]
            batch_idx = [c[0] for c in chunk]
            batch = torch.stack([c[1] for c in chunk]).to(device, non_blocking=True)
            with torch.no_grad():
                # Forward through full detector so that we ALSO capture the head logits.
                # Hooks fire inside model.backbone.
                data = {"image": batch}
                try:
                    pred = model(data, inference=True)
                except TypeError:
                    pred = model(data)
                if isinstance(pred, dict):
                    logits = pred.get("cls", pred.get("logits", pred.get("score")))
                else:
                    logits = pred
                if logits is None:
                    raise RuntimeError("model returned no logits")
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            for k, idx in enumerate(batch_idx):
                scores[idx] = float(probs[k])
            valid.extend(batch_idx)
            if (j // batch_size) % 10 == 0:
                logger.info("  [%s] batch %d/%d", ckpt_label,
                            j // batch_size + 1,
                            (len(pending) + batch_size - 1) // batch_size)
        per_layer = {
            ix: (np.concatenate(captured[ix], axis=0) if captured[ix]
                 else np.zeros((0, 0), dtype=np.float32))
            for ix in layers
        }
    finally:
        for h in handles:
            h.remove()
        del model

    return scores, per_layer, np.array(valid, dtype=np.int64)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    t0 = time.time()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("device=%s, layers=%s", device, LAYERS)

    sampled = pd.read_csv(
        SAMPLED_CSV,
        usecols=["gcs_uri", "label", "split", "identity_key",
                 "session_id", "video_id", "method",
                 "local_path", "sharpness_laplacian",
                 "brightness_v_mean", "face_pixel_area",
                 "width", "height", "is_no_face"],
    )
    sampled = sampled.iloc[:800].reset_index(drop=True)
    sampled["row_ix"] = np.arange(len(sampled))
    paths = [Path(p) for p in sampled["local_path"].tolist()]
    n_samples = len(paths)
    logger.info("loaded %d frames", n_samples)

    score_cols: Dict[str, np.ndarray] = {}

    for label, ckpt_path in CKPTS_TO_EXTRACT.items():
        # Skip if all layers + scores already cached.
        if all(cache_path_for_layer(label, ix, n_samples).exists() for ix in LAYERS):
            score_cache = LOCAL_CACHE_DIR / f"scores_{label}__n{n_samples}.npy"
            if score_cache.exists():
                logger.info("[%s] full cache hit; skipping forward pass", label)
                score_cols[label] = np.load(score_cache)
                continue
        if not ckpt_path.exists():
            logger.error("[%s] missing ckpt: %s", label, ckpt_path)
            continue
        logger.info("[%s] forward pass…", label)
        t1 = time.time()
        scores, per_layer, valid_idx = extract_per_ckpt(
            label, ckpt_path, paths, device, LAYERS, batch_size=16, n_samples=n_samples
        )
        for ix in LAYERS:
            np.savez_compressed(
                cache_path_for_layer(label, ix, n_samples),
                features=per_layer[ix].astype(np.float32),
                valid_idx=valid_idx,
            )
        np.save(LOCAL_CACHE_DIR / f"scores_{label}__n{n_samples}.npy", scores)
        score_cols[label] = scores
        logger.info("[%s] done in %.1fs (mean=%.3f, p50=%.3f)",
                    label, time.time() - t1, np.nanmean(scores), np.nanmedian(scores))

    # Build the per-frame table with all Stage 2 + reference ckpt scores.
    per_frame = sampled[["row_ix", "gcs_uri", "label", "split",
                         "identity_key", "method"]].copy()
    for label, scores in score_cols.items():
        per_frame[label] = scores
    per_frame.to_csv(OUTPUTS / "stage2_triptych_scores.csv", index=False)
    logger.info("wrote outputs/stage2_triptych_scores.csv")

    logger.info("total elapsed: %.1f min", (time.time() - t0) / 60.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
