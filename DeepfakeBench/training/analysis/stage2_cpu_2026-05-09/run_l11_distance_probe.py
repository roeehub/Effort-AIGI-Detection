"""L11 feature-distance probe — pre-flight for option 3a (L11 anchor-loss FT).

Question: For each Stage 2 step4500 ckpt, how far has L11 CLS feature
drifted from P8A on the 130-frame Roy_D set? If the drift is large, an
L11 anchor lever has a clear target.

Reuses extraction infrastructure from `dor_encoder_axis_2026-05-08/
build_cohort_features.py`. Pulls hook on `model.backbone.transformer.
resblocks[11]` and captures the CLS token's pre-projection 768-d
representation per frame.

Output:
  outputs/l11_distance_per_frame.csv
  outputs/l11_distance_per_ckpt.csv

Per-ckpt summary: mean cosine-distance(P8A, ckpt) on the 130 Roy_D frames,
its std, and the correlation with the score-shift |fake_prob - P8A_fake_prob|.
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

THIS_DIR = Path(__file__).resolve().parent
CKPT_DIR = THIS_DIR / "_ckpts"
FRAMES_DIR = THIS_DIR / "_roy_d_frames"
OUTPUTS = THIS_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

ROY_D_REF_CSV = (
    REPO_ROOT / "analysis" / "p1_pe_eval_2026-05-07" / "roy_d_regression" / "roy_d_per_frame_scores.csv"
)
P8A_LOCAL_CKPT = (
    REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
    / "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth"
)

DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "config" / "detector" / "effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config" / "train_config.yaml"

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

CKPTS_TO_PROBE: Dict[str, Path] = {
    "P8A": P8A_LOCAL_CKPT,
    "S1_step500": CKPT_DIR / "S1_step500.pth",
    "S1_step2500": CKPT_DIR / "S1_step2500.pth",
    "S1_step4500": CKPT_DIR / "S1_step4500.pth",
    "S2_step500": CKPT_DIR / "S2_step500.pth",
    "S2_step2500": CKPT_DIR / "S2_step2500.pth",
    "S2_step4500": CKPT_DIR / "S2_step4500.pth",
    "S3_step500": CKPT_DIR / "S3_step500.pth",
    "S3_step2500": CKPT_DIR / "S3_step2500.pth",
    "S3_step4500": CKPT_DIR / "S3_step4500.pth",
}

logger = logging.getLogger("l11-probe")


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


def extract_l11_cls(
    model: torch.nn.Module,
    paths: List[Path],
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """Returns (n, 768) L11 CLS features."""
    resblocks = get_resblocks(model)
    capture: List[np.ndarray] = []

    def hook11(_module, _input, output):
        if output.dim() == 3:
            if output.shape[0] >= output.shape[1]:
                cls = output[0]  # (seq, B, dim) layout
            else:
                cls = output[:, 0]  # (B, seq, dim) layout
        elif output.dim() == 2:
            cls = output
        else:
            raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
        capture.append(cls.detach().cpu().to(torch.float32).numpy())

    handle = resblocks[11].register_forward_hook(hook11)
    try:
        pending: List[Tuple[int, torch.Tensor]] = []

        def flush(batch):
            if not batch:
                return
            x = torch.stack([b[1] for b in batch]).to(device)
            with torch.no_grad():
                model({"image": x}, inference=True)

        for i, p in enumerate(paths):
            t = load_and_preprocess(Path(p))
            if t is not None:
                pending.append((i, t))
            if len(pending) >= batch_size:
                flush(pending)
                pending = []
        flush(pending)
    finally:
        handle.remove()

    if not capture:
        return np.zeros((0, 768), dtype=np.float32)
    return np.concatenate(capture, axis=0)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")

    if not P8A_LOCAL_CKPT.exists():
        logger.error("P8A ckpt missing at %s — cannot establish baseline", P8A_LOCAL_CKPT)
        return 2

    # Frame paths.
    frames = sorted(FRAMES_DIR.glob("Roy_D__frame_*.png"))
    logger.info("Roy_D frames: %d", len(frames))
    if len(frames) == 0:
        logger.error("No Roy_D frames in %s", FRAMES_DIR)
        return 2

    # Device.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("device: %s", device)

    # Extract L11 features per ckpt.
    feats: Dict[str, np.ndarray] = {}
    for label, p in CKPTS_TO_PROBE.items():
        if not p.exists():
            logger.warning("ckpt missing: %s", p)
            continue
        logger.info("[%s] extracting L11 on %d frames", label, len(frames))
        model = load_effort_model(p, device)
        f = extract_l11_cls(model, frames, device, batch_size=32)
        feats[label] = f
        logger.info("[%s] feats shape=%s", label, f.shape)
        del model

    if "P8A" not in feats:
        logger.error("P8A features not extracted; aborting")
        return 2

    p8a_f = feats["P8A"]  # (n, 768)

    # Per-frame cos distance vs P8A.
    rows = []
    summary = []
    for label, f in feats.items():
        if label == "P8A":
            continue
        # Align lengths.
        n = min(len(p8a_f), len(f))
        a = p8a_f[:n]
        b = f[:n]
        # Per-frame cos distance.
        a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        cos_sim = (a_n * b_n).sum(axis=1)
        cos_dist = 1.0 - cos_sim
        l2 = np.linalg.norm(a - b, axis=1)
        for i in range(n):
            rows.append({
                "ckpt": label,
                "frame_basename": frames[i].name,
                "cos_dist_l11_vs_P8A": float(cos_dist[i]),
                "l2_dist_l11_vs_P8A": float(l2[i]),
            })
        summary.append({
            "ckpt": label,
            "n": n,
            "cos_dist_mean": float(cos_dist.mean()),
            "cos_dist_p50": float(np.median(cos_dist)),
            "cos_dist_p90": float(np.percentile(cos_dist, 90)),
            "cos_dist_max": float(cos_dist.max()),
            "l2_mean": float(l2.mean()),
            "l2_p90": float(np.percentile(l2, 90)),
        })

    pd.DataFrame(rows).to_csv(OUTPUTS / "l11_distance_per_frame.csv", index=False)
    summary_df = pd.DataFrame(summary).sort_values(by="cos_dist_mean")
    summary_df.to_csv(OUTPUTS / "l11_distance_per_ckpt.csv", index=False)

    print("\n" + "=" * 80)
    print("L11 CLS cosine-distance vs P8A on 130-frame Roy_D set")
    print("(higher cos_dist = more L11 drift; ideal anchor target = high & correlated with score regression)")
    print("=" * 80)
    pd.options.display.float_format = "{:.4f}".format
    print(summary_df.to_string(index=False))
    print("\n[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
