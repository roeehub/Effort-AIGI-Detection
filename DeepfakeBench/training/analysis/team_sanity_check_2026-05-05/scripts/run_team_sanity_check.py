#!/usr/bin/env python3
"""
Team sanity-check: score 9 candidate ckpts on 210 REAL team frames.

Frames are pre-cropped face images (~130-170px square) from the standard
crop pipeline (filename: <identity>__frame_NNNNNN_seqNN.png). We resize to
224 with cv2.INTER_LINEAR (post-fix preprocessing parity, matches
combined_paired.py:3455) and apply CLIP normalization.

Loads ckpts one at a time (Mac local CPU memory budget), scores all 210
frames, writes per-ckpt CSVs, then aggregates per-identity FPR at three
tau levels (0.5, FPR=10%, FPR=5%).

CPU only. No CUDA. No MPS.
"""
from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import sys
import tempfile
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

# Add training dir to path so detector imports resolve
TRAINING_DIR = Path(__file__).resolve().parents[3]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from detectors import DETECTOR  # noqa: E402

logger = logging.getLogger("team-sanity-check")

# CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# ---------------------------------------------------------------------------
# Checkpoints under test
# ---------------------------------------------------------------------------
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
        "name": "E3_TOP_N_STEP6600",
        "gcs": "gs://training-job-outputs/best_checkpoints/jzroefab/top_n_effort_20260503_step6600_auc0.9972_eer0.0093.pth",
        "arch": "L14_scratch",
    },
    {
        "name": "PA_TOP_N_STEP5600",
        "gcs": "gs://training-job-outputs/best_checkpoints/26u8bn1t/top_n_effort_20260504_step5600_auc0.9909_eer0.0282.pth",
        "arch": "B16",
    },
    {
        "name": "PA_TOP_N_STEP3800",
        "gcs": "gs://training-job-outputs/best_checkpoints/26u8bn1t/top_n_effort_20260504_step3800_auc0.9892_eer0.0402.pth",
        "arch": "B16",
    },
    {
        "name": "PA_PERIODIC_STEP5000",
        "gcs": "gs://training-job-outputs/best_checkpoints/26u8bn1t/periodic_effort_20260504_step5000_auc0.9869_eer0.0322.pth",
        "arch": "B16",
    },
    {
        "name": "PC_TOP_N_STEP7400",
        "gcs": "gs://training-job-outputs/best_checkpoints/0ujswaad/top_n_effort_20260504_step7400_auc0.9871_eer0.0415.pth",
        "arch": "B16",
    },
    {
        "name": "PC_TOP_N_STEP5400",
        "gcs": "gs://training-job-outputs/best_checkpoints/0ujswaad/top_n_effort_20260504_step5400_auc0.9863_eer0.0459.pth",
        "arch": "B16",
    },
    {
        "name": "PC_PERIODIC_STEP5000",
        "gcs": "gs://training-job-outputs/best_checkpoints/0ujswaad/periodic_effort_20260504_step5000_auc0.9862_eer0.0415.pth",
        "arch": "B16",
    },
]

# ---------------------------------------------------------------------------
# Tau values, sourced from:
#   analysis/shortcut_audit_2026-05-05/tau_calibration.csv (P8A, E2B)
#   docs/relaunch_handoffs/HANDOFF_FINAL_SYNTHESIS_2026-05-05.md (PA, PC)
# E3 has no published per-substrate tau yet; mark it MISSING and we will
# also compute an empirical fallback later by rank within these 210 frames
# (NOT a true FPR=10% calibration since these frames were not in any dev
# pool — but lets us still report something).
# ---------------------------------------------------------------------------
TAU_VALUES: Dict[str, Dict[str, Optional[float]]] = {
    # ckpt: { tau_05_default: 0.5, tau_fpr10: <val>, tau_fpr05: <val> }
    "P8A_REFERENCE_STEP5000":   {"tau_default":  0.5, "tau_fpr10": 0.7050346434116362, "tau_fpr05": 0.9752188086509702},
    "E2B_TOP_N_STEP3200":       {"tau_default":  0.5, "tau_fpr10": 0.5064434647560113, "tau_fpr05": 0.7410529732704153},
    "E3_TOP_N_STEP6600":        {"tau_default":  0.5, "tau_fpr10": None,                "tau_fpr05": None},
    "PA_TOP_N_STEP5600":        {"tau_default":  0.5, "tau_fpr10": 0.9907,              "tau_fpr05": None},
    "PA_TOP_N_STEP3800":        {"tau_default":  0.5, "tau_fpr10": 0.8773,              "tau_fpr05": None},
    "PA_PERIODIC_STEP5000":     {"tau_default":  0.5, "tau_fpr10": 0.9846,              "tau_fpr05": None},
    "PC_TOP_N_STEP7400":        {"tau_default":  0.5, "tau_fpr10": 0.9832,              "tau_fpr05": None},
    "PC_TOP_N_STEP5400":        {"tau_default":  0.5, "tau_fpr10": 0.9908,              "tau_fpr05": None},
    "PC_PERIODIC_STEP5000":     {"tau_default":  0.5, "tau_fpr10": 0.9761,              "tau_fpr05": None},
}

TAU_SOURCE: Dict[str, Dict[str, str]] = {
    "P8A_REFERENCE_STEP5000":   {"tau_default": "fixed_default", "tau_fpr10": "shortcut_audit_2026-05-05/tau_calibration.csv", "tau_fpr05": "shortcut_audit_2026-05-05/tau_calibration.csv"},
    "E2B_TOP_N_STEP3200":       {"tau_default": "fixed_default", "tau_fpr10": "shortcut_audit_2026-05-05/tau_calibration.csv", "tau_fpr05": "shortcut_audit_2026-05-05/tau_calibration.csv"},
    "E3_TOP_N_STEP6600":        {"tau_default": "fixed_default", "tau_fpr10": "MISSING_no_published_tau",                       "tau_fpr05": "MISSING_no_published_tau"},
    "PA_TOP_N_STEP5600":        {"tau_default": "fixed_default", "tau_fpr10": "HANDOFF_FINAL_SYNTHESIS_2026-05-05.md MODERATE", "tau_fpr05": "MISSING_no_LOOSE_or_05_published"},
    "PA_TOP_N_STEP3800":        {"tau_default": "fixed_default", "tau_fpr10": "HANDOFF_FINAL_SYNTHESIS_2026-05-05.md MODERATE", "tau_fpr05": "MISSING_no_LOOSE_or_05_published"},
    "PA_PERIODIC_STEP5000":     {"tau_default": "fixed_default", "tau_fpr10": "HANDOFF_FINAL_SYNTHESIS_2026-05-05.md MODERATE", "tau_fpr05": "MISSING_no_LOOSE_or_05_published"},
    "PC_TOP_N_STEP7400":        {"tau_default": "fixed_default", "tau_fpr10": "HANDOFF_FINAL_SYNTHESIS_2026-05-05.md MODERATE", "tau_fpr05": "MISSING_no_LOOSE_or_05_published"},
    "PC_TOP_N_STEP5400":        {"tau_default": "fixed_default", "tau_fpr10": "HANDOFF_FINAL_SYNTHESIS_2026-05-05.md MODERATE", "tau_fpr05": "MISSING_no_LOOSE_or_05_published"},
    "PC_PERIODIC_STEP5000":     {"tau_default": "fixed_default", "tau_fpr10": "HANDOFF_FINAL_SYNTHESIS_2026-05-05.md MODERATE", "tau_fpr05": "MISSING_no_LOOSE_or_05_published"},
}


# ---------------------------------------------------------------------------
# Frame discovery
# ---------------------------------------------------------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg"}


def discover_frames(frames_root: Path) -> List[Tuple[str, Path]]:
    """Return list of (identity, frame_path) tuples, sorted within identity."""
    out = []
    for d in sorted(frames_root.iterdir()):
        if not d.is_dir():
            continue
        for fp in sorted(d.iterdir()):
            if fp.suffix.lower() in IMG_EXTS:
                out.append((d.name, fp))
    return out


# ---------------------------------------------------------------------------
# Dataset (pre-cropped face images on disk)
# ---------------------------------------------------------------------------
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
        # INTER_LINEAR matches post-fix preprocessing parity (combined_paired.py:3455)
        img_bgr = cv2.resize(img_bgr, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(img_rgb), idx


# ---------------------------------------------------------------------------
# Checkpoint download (gsutil-based, no Python GCS dep needed locally)
# ---------------------------------------------------------------------------
def gsutil_cp(gcs_uri: str, local_path: Path) -> None:
    import subprocess
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > 1024 * 1024:
        logger.info("    cached: %s (%.1f MB)", local_path.name, local_path.stat().st_size / 1e6)
        return
    logger.info("    downloading %s ...", gcs_uri)
    subprocess.run(
        ["gsutil", "-q", "cp", gcs_uri, str(local_path)],
        check=True,
    )
    logger.info("    -> %s (%.1f MB)", local_path.name, local_path.stat().st_size / 1e6)


# ---------------------------------------------------------------------------
# Model loading (mirrors batch_inference_gcs.load_model)
# ---------------------------------------------------------------------------
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

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
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

    # Restore ArcFace scale if applicable
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


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def score_all(
    model: torch.nn.Module,
    items: List[Tuple[str, Path]],
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 0,
) -> List[float]:
    dataset = LocalFrameDataset(items)
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    probs = [None] * len(items)
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


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(
    items: List[Tuple[str, Path]],
    scores_by_ckpt: Dict[str, List[float]],
    out_dir: Path,
) -> None:
    """Write all_scores.csv, per_identity_per_ckpt_score_stats.csv,
    per_identity_per_ckpt_fpr.csv, tau_values_used.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # all_scores.csv (long)
    rows: List[Dict[str, Any]] = []
    for ckpt_name, probs in scores_by_ckpt.items():
        for (ident, fp), p in zip(items, probs):
            rows.append({
                "ckpt": ckpt_name,
                "identity": ident,
                "frame_path": str(fp.relative_to(out_dir.parent)),
                "frame_prob": p,
            })
    with open(out_dir / "all_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ckpt", "identity", "frame_path", "frame_prob"])
        w.writeheader()
        w.writerows(rows)

    # tau_values_used.csv
    tau_rows = []
    for ckpt_name in scores_by_ckpt:
        for tau_level in ("tau_default", "tau_fpr10", "tau_fpr05"):
            tau_rows.append({
                "ckpt": ckpt_name,
                "tau_level": tau_level,
                "tau_value": TAU_VALUES[ckpt_name].get(tau_level),
                "source": TAU_SOURCE[ckpt_name].get(tau_level, ""),
            })
    with open(out_dir / "tau_values_used.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ckpt", "tau_level", "tau_value", "source"])
        w.writeheader()
        w.writerows(tau_rows)

    # Identity grouping
    idents_in_order = []
    seen = set()
    for ident, _ in items:
        if ident not in seen:
            idents_in_order.append(ident)
            seen.add(ident)

    # per-identity stats (per ckpt)
    stats_rows = []
    for ckpt_name, probs in scores_by_ckpt.items():
        # per identity
        for ident in idents_in_order:
            ident_probs = np.array([
                p for (i, _), p in zip(items, probs) if i == ident and not np.isnan(p)
            ], dtype=np.float64)
            if len(ident_probs) == 0:
                stats_rows.append({"ckpt": ckpt_name, "identity": ident, "n": 0, "mean": "", "median": "", "p90": "", "max": ""})
                continue
            stats_rows.append({
                "ckpt": ckpt_name,
                "identity": ident,
                "n": int(len(ident_probs)),
                "mean":   f"{ident_probs.mean():.6f}",
                "median": f"{np.median(ident_probs):.6f}",
                "p90":    f"{np.percentile(ident_probs, 90):.6f}",
                "max":    f"{ident_probs.max():.6f}",
            })
        # all
        all_probs = np.array([p for p in probs if not np.isnan(p)], dtype=np.float64)
        stats_rows.append({
            "ckpt": ckpt_name,
            "identity": "_ALL",
            "n": int(len(all_probs)),
            "mean":   f"{all_probs.mean():.6f}"   if len(all_probs) else "",
            "median": f"{np.median(all_probs):.6f}" if len(all_probs) else "",
            "p90":    f"{np.percentile(all_probs, 90):.6f}" if len(all_probs) else "",
            "max":    f"{all_probs.max():.6f}"    if len(all_probs) else "",
        })
    with open(out_dir / "per_identity_per_ckpt_score_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ckpt", "identity", "n", "mean", "median", "p90", "max"])
        w.writeheader()
        w.writerows(stats_rows)

    # per-identity FPR by tau level
    fpr_rows = []
    for ckpt_name, probs in scores_by_ckpt.items():
        for tau_level in ("tau_default", "tau_fpr10", "tau_fpr05"):
            tau = TAU_VALUES[ckpt_name].get(tau_level)
            for ident in idents_in_order + ["_ALL"]:
                if ident == "_ALL":
                    ident_probs = np.array([p for p in probs if not np.isnan(p)], dtype=np.float64)
                else:
                    ident_probs = np.array([
                        p for (i, _), p in zip(items, probs)
                        if i == ident and not np.isnan(p)
                    ], dtype=np.float64)

                n = int(len(ident_probs))
                if tau is None or n == 0:
                    fpr_rows.append({
                        "ckpt": ckpt_name, "identity": ident,
                        "tau_level": tau_level, "tau_value": "" if tau is None else tau,
                        "n": n, "n_above_tau": "", "fpr": "",
                    })
                    continue
                n_above = int((ident_probs >= tau).sum())
                fpr = n_above / n
                fpr_rows.append({
                    "ckpt": ckpt_name, "identity": ident,
                    "tau_level": tau_level, "tau_value": f"{tau:.6f}",
                    "n": n, "n_above_tau": n_above, "fpr": f"{fpr:.6f}",
                })
    with open(out_dir / "per_identity_per_ckpt_fpr.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ckpt", "identity", "tau_level", "tau_value", "n", "n_above_tau", "fpr"])
        w.writeheader()
        w.writerows(fpr_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]),
                        help="Workspace root (analysis/team_sanity_check_2026-05-05/)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 to keep memory budget tight)")
    parser.add_argument("--limit-ckpts", type=int, default=0,
                        help="If >0, only run the first N ckpts (debug)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip ckpts whose scores CSV already exists")
    parser.add_argument("--detector-config", type=str,
                        default=str(TRAINING_DIR / "config" / "detector" / "effort.yaml"))
    parser.add_argument("--train-config", type=str,
                        default=str(TRAINING_DIR / "config" / "train_config.yaml"))
    args = parser.parse_args()

    root = Path(args.root)
    frames_dir = root / "frames"
    scores_dir = root / "scores"
    ckpts_dir = root / "checkpoints"
    log_path = root / "run.log"

    scores_dir.mkdir(parents=True, exist_ok=True)
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    # Logger: file + stderr
    fh = logging.FileHandler(str(log_path))
    sh = logging.StreamHandler(sys.stderr)
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.setLevel(logging.INFO)
    logger.addHandler(fh); logger.addHandler(sh)

    device = torch.device("cpu")
    logger.info("=" * 70)
    logger.info("Team sanity-check run starting")
    logger.info("=" * 70)
    logger.info("device=%s, threads=%d, batch_size=%d, num_workers=%d",
                device, torch.get_num_threads(), args.batch_size, args.num_workers)

    items = discover_frames(frames_dir)
    logger.info("Discovered %d frames across %d identities",
                len(items), len({i for i, _ in items}))
    by_ident: Dict[str, int] = {}
    for ident, _ in items:
        by_ident[ident] = by_ident.get(ident, 0) + 1
    for k, v in sorted(by_ident.items()):
        logger.info("  %s: %d", k, v)

    ckpts = CKPTS if args.limit_ckpts <= 0 else CKPTS[: args.limit_ckpts]

    scores_by_ckpt: Dict[str, List[float]] = {}
    failed_ckpts: List[Tuple[str, str]] = []

    overall_t0 = time.time()
    for ckpt_idx, ckpt in enumerate(ckpts):
        name = ckpt["name"]
        gcs = ckpt["gcs"]
        arch = ckpt["arch"]
        per_ckpt_csv = scores_dir / f"{name}.csv"
        logger.info("-" * 70)
        logger.info("[%d/%d] %s (%s)", ckpt_idx + 1, len(ckpts), name, arch)

        # Resume support
        if args.resume and per_ckpt_csv.exists():
            try:
                rows = list(csv.DictReader(open(per_ckpt_csv)))
                if len(rows) == len(items):
                    probs = [float(r["frame_prob"]) for r in rows]
                    scores_by_ckpt[name] = probs
                    logger.info("    RESUMED from %s (%d rows)", per_ckpt_csv.name, len(rows))
                    continue
            except Exception as e:
                logger.warning("    resume failed (%s); will re-score", e)

        local_ckpt_path = ckpts_dir / Path(gcs).name

        try:
            gsutil_cp(gcs, local_ckpt_path)
        except Exception as e:
            logger.error("    DOWNLOAD FAILED: %s", e)
            failed_ckpts.append((name, f"download_failed: {e}"))
            scores_by_ckpt[name] = [float("nan")] * len(items)
            with open(per_ckpt_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["frame_path", "identity", "frame_prob"])
                w.writeheader()
                for ident, fp in items:
                    w.writerow({"frame_path": str(fp.relative_to(root)), "identity": ident, "frame_prob": ""})
            continue

        try:
            t_load = time.time()
            model = load_model(local_ckpt_path, Path(args.detector_config), Path(args.train_config), device)
            logger.info("    model loaded in %.1fs", time.time() - t_load)
        except Exception as e:
            logger.error("    MODEL LOAD FAILED: %s", e)
            failed_ckpts.append((name, f"load_failed: {e}"))
            scores_by_ckpt[name] = [float("nan")] * len(items)
            with open(per_ckpt_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["frame_path", "identity", "frame_prob"])
                w.writeheader()
                for ident, fp in items:
                    w.writerow({"frame_path": str(fp.relative_to(root)), "identity": ident, "frame_prob": ""})
            # cleanup ckpt
            try:
                local_ckpt_path.unlink()
            except Exception:
                pass
            continue

        try:
            t_inf = time.time()
            probs = score_all(model, items, device, batch_size=args.batch_size, num_workers=args.num_workers)
            logger.info("    inference done in %.1fs", time.time() - t_inf)
        except Exception as e:
            logger.error("    INFERENCE FAILED: %s", e)
            failed_ckpts.append((name, f"inference_failed: {e}"))
            probs = [float("nan")] * len(items)

        scores_by_ckpt[name] = probs

        # write per-ckpt CSV
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

        # free
        del model
        gc.collect()
        # delete the ckpt locally to keep disk usage modest (B16 ~600MB, L14 ~1.5GB)
        try:
            local_ckpt_path.unlink()
            logger.info("    removed local ckpt to save disk")
        except Exception:
            pass

    overall_dt = time.time() - overall_t0
    logger.info("=" * 70)
    logger.info("All ckpts done in %.1fs (%.1f min)", overall_dt, overall_dt / 60)
    if failed_ckpts:
        logger.error("FAILED ckpts (%d):", len(failed_ckpts))
        for name, why in failed_ckpts:
            logger.error("  %s: %s", name, why)
    else:
        logger.info("No failures")

    # Aggregate
    aggregate(items, scores_by_ckpt, root)
    logger.info("Aggregation written to %s", root)

    return 0 if not failed_ckpts else 2


if __name__ == "__main__":
    raise SystemExit(main())
