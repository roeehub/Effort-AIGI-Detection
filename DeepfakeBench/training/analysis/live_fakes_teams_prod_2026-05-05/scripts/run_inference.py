"""
Score live-fakes-teams-prod (1482 fake frames, 18 tags) with 3 browser ckpts.

Adapted from analysis/team_sanity_check_2026-05-05/scripts/run_team_sanity_check.py
with:
  - device = MPS (Apple Silicon) with CPU fallback
  - frames root = analysis/live_fakes_teams_prod_2026-05-05/raw/session_20260414_112354/
  - identity = first-level dir name (e.g., xiang-fake-1, xinhe-fake-8-glasses)
  - skips the 'metadata' dir
  - only 3 ckpts: P8A_REFERENCE_STEP5000, E2B_TOP_N_STEP3200, PA_TOP_N_STEP3800

Output:
  scores/<CKPT>.csv  with cols: frame_path, identity, frame_prob

Run from training/ root:
  python analysis/live_fakes_teams_prod_2026-05-05/scripts/run_inference.py
"""
from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
import yaml

TRAINING_DIR = Path(__file__).resolve().parents[3]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from detectors import DETECTOR  # noqa: E402

logger = logging.getLogger("live-fakes-inference")

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

CKPTS: List[Dict[str, str]] = [
    {
        "name": "P8A_REFERENCE_STEP5000",
        "gcs": "gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth",
        "arch": "B16",
    },
    {
        "name": "E2B_TOP_N_STEP3200",
        "gcs": "gs://training-job-outputs/best_checkpoints/rmat8lwx/top_n_effort_20260503_step3200_auc0.9863_eer0.0310.pth",
        "arch": "B16_scratch",
    },
    {
        "name": "PA_TOP_N_STEP3800",
        "gcs": "gs://training-job-outputs/best_checkpoints/26u8bn1t/top_n_effort_20260504_step3800_auc0.9892_eer0.0402.pth",
        "arch": "B16",
    },
]

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def discover_frames(frames_root: Path) -> List[Tuple[str, Path]]:
    """List (identity, frame_path), where identity = first-level dir name.

    Skips the 'metadata' dir.  Only returns image files.
    """
    out = []
    for d in sorted(frames_root.iterdir()):
        if not d.is_dir():
            continue
        if d.name == "metadata":
            continue
        for fp in sorted(d.iterdir()):
            if fp.suffix.lower() in IMG_EXTS:
                out.append((d.name, fp))
    return out


class LocalFrameDataset(data.Dataset):
    def __init__(self, items: List[Tuple[str, Path]], resolution: int = 224):
        self.items = items
        self.resolution = resolution
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        ident, fp = self.items[idx]
        img_bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return torch.zeros(3, self.resolution, self.resolution), idx
        img_bgr = cv2.resize(img_bgr, (self.resolution, self.resolution),
                             interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(img_rgb), idx


def gsutil_cp(gcs_uri: str, local_path: Path) -> None:
    import subprocess
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > 1024 * 1024:
        logger.info("    cached: %s (%.1f MB)", local_path.name, local_path.stat().st_size / 1e6)
        return
    logger.info("    downloading %s ...", gcs_uri)
    subprocess.run(["gsutil", "-q", "cp", gcs_uri, str(local_path)], check=True)
    logger.info("    -> %s (%.1f MB)", local_path.name, local_path.stat().st_size / 1e6)


def load_model(
    checkpoint_path: Path,
    detector_config: Path,
    train_config: Path,
    device: torch.device,
) -> torch.nn.Module:
    with open(detector_config, "r") as f:
        cfg = yaml.safe_load(f)
    with open(train_config, "r") as f:
        train_cfg = yaml.safe_load(f)
    cfg.update(train_cfg)

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
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
            logger.info("    Restored ArcFace s=%.3f", model_config["current_arcface_s"])

    clean_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing:
        logger.debug("    Missing keys: %d", len(missing))
    if unexpected:
        logger.debug("    Unexpected keys: %d", len(unexpected))

    model.eval()
    return model


def score_all(
    model: torch.nn.Module,
    items: List[Tuple[str, Path]],
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 0,
) -> List[float]:
    dataset = LocalFrameDataset(items)
    loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    probs: List[Optional[float]] = [None] * len(items)
    t0 = time.time()
    n_batches = len(loader)
    for bi, (images, indices) in enumerate(loader):
        images = images.to(device)
        with torch.inference_mode():
            outputs = model({"image": images}, inference=True)
            batch_probs = outputs["prob"].detach().cpu().numpy().reshape(-1)
        for i, gi in enumerate(indices.numpy()):
            probs[int(gi)] = float(batch_probs[i])
        if (bi + 1) % 5 == 0 or (bi + 1) == n_batches:
            elapsed = time.time() - t0
            done = sum(1 for p in probs if p is not None)
            fps = done / max(elapsed, 1e-6)
            logger.info("    batch %d/%d  done=%d/%d  %.2f fps  %.1fs",
                        bi + 1, n_batches, done, len(items), fps, elapsed)
    return [float(p) if p is not None else float("nan") for p in probs]


def pick_device(prefer_mps: bool) -> torch.device:
    if not prefer_mps:
        return torch.device("cpu")
    if not torch.backends.mps.is_available():
        logger.info("MPS not available; using CPU")
        return torch.device("cpu")
    if not torch.backends.mps.is_built():
        logger.info("MPS not built; using CPU")
        return torch.device("cpu")
    logger.info("Using MPS (Apple Silicon GPU)")
    return torch.device("mps")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str,
                        default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--frames-subdir", type=str,
                        default="raw/session_20260414_112354")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--limit-ckpts", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="Skip ckpts whose scores CSV already exists")
    parser.add_argument("--detector-config", type=str,
                        default=str(TRAINING_DIR / "config" / "detector" / "effort.yaml"))
    parser.add_argument("--train-config", type=str,
                        default=str(TRAINING_DIR / "config" / "train_config.yaml"))
    args = parser.parse_args()

    root = Path(args.root)
    frames_root = root / args.frames_subdir
    scores_dir = root / "scores"
    ckpts_dir = root / "checkpoints"
    log_path = root / "run.log"

    scores_dir.mkdir(parents=True, exist_ok=True)
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(str(log_path))
    sh = logging.StreamHandler(sys.stderr)
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.setLevel(logging.INFO)
    logger.addHandler(fh); logger.addHandler(sh)

    device = pick_device(prefer_mps=(args.device == "mps"))
    logger.info("=" * 70)
    logger.info("Live-fakes-teams-prod inference starting")
    logger.info("=" * 70)
    logger.info("device=%s, batch_size=%d, num_workers=%d",
                device, args.batch_size, args.num_workers)

    items = discover_frames(frames_root)
    logger.info("Discovered %d frames across %d identities",
                len(items), len({i for i, _ in items}))
    by_ident: Dict[str, int] = {}
    for ident, _ in items:
        by_ident[ident] = by_ident.get(ident, 0) + 1
    for k, v in sorted(by_ident.items()):
        logger.info("  %s: %d", k, v)

    ckpts = CKPTS if args.limit_ckpts <= 0 else CKPTS[: args.limit_ckpts]

    overall_t0 = time.time()
    timing: Dict[str, float] = {}
    failed: List[Tuple[str, str]] = []

    for ckpt_idx, ckpt in enumerate(ckpts):
        name = ckpt["name"]
        gcs = ckpt["gcs"]
        per_ckpt_csv = scores_dir / f"{name}.csv"
        logger.info("-" * 70)
        logger.info("[%d/%d] %s (%s)", ckpt_idx + 1, len(ckpts), name, ckpt["arch"])

        if args.resume and per_ckpt_csv.exists():
            logger.info("    RESUMED (file exists): %s", per_ckpt_csv.name)
            continue

        local_ckpt_path = ckpts_dir / Path(gcs).name
        try:
            gsutil_cp(gcs, local_ckpt_path)
        except Exception as e:
            logger.error("    DOWNLOAD FAILED: %s", e)
            failed.append((name, f"download: {e}"))
            continue

        try:
            t_load = time.time()
            model = load_model(local_ckpt_path,
                               Path(args.detector_config),
                               Path(args.train_config),
                               device)
            logger.info("    model loaded in %.1fs", time.time() - t_load)
        except Exception as e:
            logger.error("    MODEL LOAD FAILED: %s", e)
            failed.append((name, f"load: {e}"))
            try:
                local_ckpt_path.unlink()
            except Exception:
                pass
            continue

        try:
            t_inf = time.time()
            probs = score_all(model, items, device,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers)
            dt = time.time() - t_inf
            logger.info("    inference done in %.1fs (%.1f fps)",
                        dt, len(items) / max(dt, 1e-6))
            timing[name] = dt
        except Exception as e:
            logger.error("    INFERENCE FAILED on %s: %s", device, e)
            if str(device) == "mps":
                logger.info("    falling back to CPU for this ckpt")
                model = model.to("cpu")
                try:
                    t_inf = time.time()
                    probs = score_all(model, items, torch.device("cpu"),
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers)
                    dt = time.time() - t_inf
                    logger.info("    CPU fallback done in %.1fs (%.1f fps)",
                                dt, len(items) / max(dt, 1e-6))
                    timing[name] = dt
                except Exception as e2:
                    logger.error("    CPU FALLBACK ALSO FAILED: %s", e2)
                    failed.append((name, f"inference: {e}; cpu_fallback: {e2}"))
                    probs = [float("nan")] * len(items)
            else:
                failed.append((name, f"inference: {e}"))
                probs = [float("nan")] * len(items)

        # frame_path stored RELATIVE to root for downstream tooling parity
        with open(per_ckpt_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame_path", "identity", "frame_prob"])
            w.writeheader()
            for (ident, fp), p in zip(items, probs):
                w.writerow({
                    "frame_path": str(fp.relative_to(root)),
                    "identity": ident,
                    "frame_prob": f"{p:.8f}" if not np.isnan(p) else "",
                })
        logger.info("    wrote %s", per_ckpt_csv.name)

        del model
        gc.collect()
        try:
            local_ckpt_path.unlink()
            logger.info("    removed local ckpt to save disk")
        except Exception:
            pass

    overall_dt = time.time() - overall_t0
    logger.info("=" * 70)
    logger.info("All ckpts done in %.1fs (%.1f min)", overall_dt, overall_dt / 60)
    logger.info("Per-ckpt timing:")
    for k, v in timing.items():
        logger.info("  %s: %.1fs (%.1f fps)", k, v, len(items) / max(v, 1e-6))
    if failed:
        logger.error("FAILED ckpts (%d):", len(failed))
        for name, why in failed:
            logger.error("  %s: %s", name, why)
        return 2
    logger.info("No failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
