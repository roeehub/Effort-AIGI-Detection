"""TTA POC: 4-view test-time augmentation on may6 false-flag substrate.

Goal: damp same-person production drift on the may6 substrate.
Production deployment ≡ E2B fires at ~57% FPR on may6 reals; if 4-view
TTA on E2B can damp this to <20%, TTA becomes a no-training-required
production patch.

Substrates (all label=real):
  - may6_falseflag (92 frames)
  - may5_correct  (60 frames)
  - dor_morning   (50 frames, control 1)
  - dor_evening   (50 frames, control 2)

Checkpoints:
  - E2B (deployment ≡ E2B per project_deployment_is_e2b_2026-05-06.md)
  - P8A (FT-base candidate; should be invariant on may6)

Views (4 total):
  V1: original (no augmentation)
  V2: horizontal flip
  V3: small shift (+2px, +2px) + small scale (×1.02), no flip
  V4: same shift+scale + horizontal flip

Output score = mean of 4 view scores. Std of 4 = uncertainty signal.

Outputs land in analysis/tta_poc_may6_2026-05-06/outputs/.
CPU only. ~10-20 minutes on M1/M2.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]  # training/
sys.path.insert(0, str(REPO_ROOT))

from arena.model_arena import (  # noqa: E402
    CLIP_MEAN, CLIP_STD,
    _download_checkpoint,
    load_model,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("tta_poc")

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Reuse the may6 / may5 frames from the cross-camera audit
XINHE_AUDIT_DIR = REPO_ROOT / "analysis/xinhe_cross_camera_audit_2026-05-06/raw"
DOR_MORNING_DIR = REPO_ROOT / "analysis/dor_morning_2026-05-05/raw/all/dor_morning"
DOR_EVENING_DIR = REPO_ROOT / "analysis/dor_evening_2026-05-05/raw/all/dor_evening"

DETECTOR_CONFIG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO_ROOT / "config/defaults.yaml"

CKPTS = {
    "E2B": "gs://training-job-outputs/best_checkpoints/rmat8lwx/top_n_effort_20260503_step3200_auc0.9863_eer0.0310.pth",
    "P8A": "gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth",
}

DEVICE = torch.device("cpu")
RESOLUTION = 224
BATCH_SIZE = 8
NUM_WORKERS = 0  # keep simple for 4 views; on Mac n_jobs=-1 forbidden

DOR_CONTROL_N = 50  # cap each control to N frames
SUBSTRATE_LABELS = ["may6_falseflag", "may5_correct", "dor_morning", "dor_evening"]

# TTA recipe (small magnitudes only; ±2px ≈ 0.9% of 224px, ±2% scale)
SHIFT_PX = 2
SCALE_FACTOR = 1.02


def collect_frames() -> list[tuple[str, str, Path]]:
    """Return list of (substrate, identity, path) tuples for all 4 substrates.
    Identity is parsed from filename so per-frame logs are tractable.
    """
    out: list[tuple[str, str, Path]] = []

    # may6 false-flag (92 frames, all "Generator PC" identity)
    may6 = sorted((XINHE_AUDIT_DIR / "may6").glob("*.png")) + \
           sorted((XINHE_AUDIT_DIR / "may6").glob("*.jpg"))
    for p in may6:
        out.append(("may6_falseflag", "Generator_PC", p))

    # may5 correct (60 frames, same identity)
    may5 = sorted((XINHE_AUDIT_DIR / "may5").glob("*.png")) + \
           sorted((XINHE_AUDIT_DIR / "may5").glob("*.jpg"))
    for p in may5:
        out.append(("may5_correct", "Generator_PC", p))

    # dor_morning, dor_evening (cap at DOR_CONTROL_N each)
    morning = sorted(DOR_MORNING_DIR.glob("*.png")) + sorted(DOR_MORNING_DIR.glob("*.jpg"))
    morning = morning[:DOR_CONTROL_N]
    for p in morning:
        out.append(("dor_morning", "dor", p))

    evening = sorted(DOR_EVENING_DIR.glob("*.png")) + sorted(DOR_EVENING_DIR.glob("*.jpg"))
    evening = evening[:DOR_CONTROL_N]
    for p in evening:
        out.append(("dor_evening", "dor", p))

    counts = {sub: 0 for sub in SUBSTRATE_LABELS}
    for sub, _, _ in out:
        counts[sub] += 1
    log.info("collected: %s, total=%d", counts, len(out))
    return out


# ─── TTA augmentations ──────────────────────────────────────────────────

def view1_original(img_bgr: np.ndarray, res: int) -> np.ndarray:
    """V1: original (resize only)."""
    return cv2.resize(img_bgr, (res, res), interpolation=cv2.INTER_LINEAR)


def view2_hflip(img_bgr: np.ndarray, res: int) -> np.ndarray:
    """V2: horizontal flip after resize."""
    resized = cv2.resize(img_bgr, (res, res), interpolation=cv2.INTER_LINEAR)
    return cv2.flip(resized, 1)


def _shift_scale(img_bgr: np.ndarray, res: int, shift_px: int, scale: float) -> np.ndarray:
    """Apply small (shift, scale) then resize back to res×res.
    Shift is +shift_px in both x and y. Scale is around image center.
    """
    h0, w0 = img_bgr.shape[:2]
    # First resize to slightly larger than res×scale, then center crop to res
    target_size = int(round(res * scale))  # e.g., 228 for scale=1.02
    resized = cv2.resize(img_bgr, (target_size, target_size),
                         interpolation=cv2.INTER_LINEAR)
    # Crop offset: center + shift_px (clamped)
    cx = (target_size - res) // 2 + shift_px
    cy = (target_size - res) // 2 + shift_px
    cx = max(0, min(target_size - res, cx))
    cy = max(0, min(target_size - res, cy))
    return resized[cy:cy + res, cx:cx + res]


def view3_shift_scale(img_bgr: np.ndarray, res: int) -> np.ndarray:
    """V3: small shift (+2px) + small scale (×1.02), no flip."""
    return _shift_scale(img_bgr, res, SHIFT_PX, SCALE_FACTOR)


def view4_shift_scale_hflip(img_bgr: np.ndarray, res: int) -> np.ndarray:
    """V4: same shift+scale as V3, then horizontal flip."""
    return cv2.flip(_shift_scale(img_bgr, res, SHIFT_PX, SCALE_FACTOR), 1)


VIEW_FNS = [view1_original, view2_hflip, view3_shift_scale, view4_shift_scale_hflip]
VIEW_NAMES = ["v1_original", "v2_hflip", "v3_shift_scale", "v4_shift_scale_hflip"]


class TTAFrameDataset(Dataset):
    """Loads each frame as 4 views stacked → tensor of shape (4, 3, H, W)."""

    def __init__(self, frame_paths: list[Path], resolution: int = RESOLUTION):
        self.paths = list(frame_paths)
        self.resolution = resolution
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            log.warning("decode-fail: %s", path)
            zero = torch.zeros(4, 3, self.resolution, self.resolution)
            return zero, idx

        views = []
        for fn in VIEW_FNS:
            v_bgr = fn(img_bgr, self.resolution)
            v_rgb = cv2.cvtColor(v_bgr, cv2.COLOR_BGR2RGB)
            views.append(self.normalize(v_rgb))
        return torch.stack(views, dim=0), idx  # (4, 3, H, W)


def _find_cached_checkpoint(ckpt_uri: str) -> str | None:
    """Look in /var/folders/.../arena_ckpt_*/ for already-downloaded copy
    of this URI's basename. Returns local path or None if not found."""
    import glob
    basename = Path(ckpt_uri).name
    candidates = glob.glob(f"/var/folders/**/arena_ckpt_*/{basename}", recursive=True)
    candidates += glob.glob(f"/tmp/arena_ckpt_*/{basename}")
    for c in candidates:
        if Path(c).exists() and Path(c).stat().st_size > 100_000_000:  # >100MB safety
            return c
    return None


def score_one_ckpt(ckpt_name: str, ckpt_uri: str, frames: list[tuple[str, str, Path]]) -> Path:
    """Download + load + score one ckpt with 4-view TTA. Returns CSV path."""
    out_csv = OUT_DIR / f"tta_scores_{ckpt_name}.csv"
    if out_csv.exists():
        log.info("[%s] CSV exists, skipping: %s", ckpt_name, out_csv)
        return out_csv

    cached = _find_cached_checkpoint(ckpt_uri)
    if cached is not None:
        log.info("[%s] using cached checkpoint: %s", ckpt_name, cached)
        local_ckpt = cached
    else:
        log.info("[%s] downloading checkpoint ...", ckpt_name)
        t0 = time.time()
        local_ckpt = _download_checkpoint(ckpt_uri)
        log.info("[%s] download in %.1fs -> %s", ckpt_name, time.time() - t0, local_ckpt)

    log.info("[%s] loading model on %s ...", ckpt_name, DEVICE)
    t0 = time.time()
    model = load_model(str(local_ckpt), str(DETECTOR_CONFIG), str(TRAIN_CONFIG), DEVICE)
    log.info("[%s] model loaded in %.1fs", ckpt_name, time.time() - t0)

    paths = [p for _, _, p in frames]
    substrates = [s for s, _, _ in frames]
    identities = [i for _, i, _ in frames]

    dataset = TTAFrameDataset(paths)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    n = len(paths)
    n_views = len(VIEW_FNS)
    view_probs = np.zeros((n, n_views), dtype=np.float32)
    log.info("[%s] running TTA inference: %d frames × %d views, batch=%d",
             ckpt_name, n, n_views, BATCH_SIZE)
    t0 = time.time()
    for batch_idx, (images_4v, indices) in enumerate(loader):
        # images_4v: (B, 4, 3, H, W). Flatten views into batch dim.
        B, V, C, H, W = images_4v.shape
        flat = images_4v.reshape(B * V, C, H, W).to(DEVICE)
        with torch.inference_mode():
            outputs = model({"image": flat}, inference=True)
            batch_probs = outputs["prob"].detach().cpu().numpy().reshape(-1)
        # Reshape back to (B, V)
        batch_view_probs = batch_probs.reshape(B, V)
        for i, gi in enumerate(indices.numpy()):
            view_probs[int(gi)] = batch_view_probs[i]
        elapsed = time.time() - t0
        done = (batch_idx + 1) * BATCH_SIZE
        log.info("[%s] batch %d (~%d/%d) elapsed=%.1fs (%.1f frames/s)",
                 ckpt_name, batch_idx + 1, min(done, n), n,
                 elapsed, min(done, n) / max(elapsed, 1e-3))

    log.info("[%s] TTA inference done in %.1fs", ckpt_name, time.time() - t0)

    # Compute mean and std
    means = view_probs.mean(axis=1)
    stds = view_probs.std(axis=1)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "substrate", "identity", "frame_path", "frame_basename",
            "ckpt", "label",
            "v1_score", "v2_score", "v3_score", "v4_score",
            "mean_score", "std_score",
        ])
        for k, (sub, ident, path) in enumerate(frames):
            w.writerow([
                sub, ident, str(path), path.name, ckpt_name, "real",
                f"{view_probs[k, 0]:.6f}",
                f"{view_probs[k, 1]:.6f}",
                f"{view_probs[k, 2]:.6f}",
                f"{view_probs[k, 3]:.6f}",
                f"{means[k]:.6f}",
                f"{stds[k]:.6f}",
            ])
    log.info("[%s] wrote -> %s", ckpt_name, out_csv)

    del model
    return out_csv


def aggregate_results(per_ckpt_csvs: dict[str, Path]) -> None:
    """Combine all per-ckpt CSVs into one tta_scores.csv plus FPR comparison."""
    # ─── Combined per-frame CSV ───
    all_rows = []
    for ckpt_name, csv_path in per_ckpt_csvs.items():
        with csv_path.open("r") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                all_rows.append(r)

    combined_csv = OUT_DIR / "tta_scores.csv"
    with combined_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    log.info("wrote combined: %s (%d rows)", combined_csv, len(all_rows))

    # ─── FPR comparison: substrate × ckpt × {single_view_FPR, tta_FPR, delta} ───
    fpr_rows = []
    summary: dict = {
        "ckpts": list(per_ckpt_csvs.keys()),
        "substrates": SUBSTRATE_LABELS,
        "tau": 0.5,
        "by_substrate": {},
    }
    verdict_rows: list[dict] = []

    for ckpt_name in per_ckpt_csvs.keys():
        ckpt_rows = [r for r in all_rows if r["ckpt"] == ckpt_name]
        for sub in SUBSTRATE_LABELS:
            sub_rows = [r for r in ckpt_rows if r["substrate"] == sub]
            if not sub_rows:
                continue
            v1_scores = np.array([float(r["v1_score"]) for r in sub_rows])
            mean_scores = np.array([float(r["mean_score"]) for r in sub_rows])
            std_scores = np.array([float(r["std_score"]) for r in sub_rows])

            single_fpr = float((v1_scores >= 0.5).mean())
            tta_fpr = float((mean_scores >= 0.5).mean())
            delta = tta_fpr - single_fpr

            # |Δ score| view1 vs mean
            abs_delta = float(np.mean(np.abs(mean_scores - v1_scores)))

            fpr_rows.append({
                "ckpt": ckpt_name,
                "substrate": sub,
                "n": len(sub_rows),
                "single_view_FPR": f"{single_fpr:.4f}",
                "tta_FPR": f"{tta_fpr:.4f}",
                "delta_FPR": f"{delta:+.4f}",
                "single_view_p50": f"{float(np.median(v1_scores)):.4f}",
                "tta_p50": f"{float(np.median(mean_scores)):.4f}",
                "mean_abs_delta_score": f"{abs_delta:.4f}",
                "mean_view_std": f"{float(np.mean(std_scores)):.4f}",
            })
            summary["by_substrate"].setdefault(sub, {})[ckpt_name] = {
                "n": len(sub_rows),
                "single_view_FPR": single_fpr,
                "tta_FPR": tta_fpr,
                "delta_FPR": delta,
                "mean_abs_delta_score": abs_delta,
                "mean_view_std": float(np.mean(std_scores)),
                "single_view_p50": float(np.median(v1_scores)),
                "tta_p50": float(np.median(mean_scores)),
            }

            # Verdict logic: TTA_HELPS if FPR drops by ≥0.10 absolute on a
            # substrate where baseline had FPR>0.10. TTA_HURTS if FPR
            # rises by ≥0.05. TTA_NEUTRAL otherwise.
            if single_fpr <= 0.10:
                # Already invariant — only check it doesn't regress
                if delta >= 0.05:
                    verdict = "TTA_HURTS_INVARIANT"
                else:
                    verdict = "TTA_NEUTRAL_INVARIANT"
            else:
                if delta <= -0.10:
                    verdict = "TTA_HELPS"
                elif delta >= 0.05:
                    verdict = "TTA_HURTS"
                else:
                    verdict = "TTA_NEUTRAL"
            verdict_rows.append({
                "ckpt": ckpt_name,
                "substrate": sub,
                "single_view_FPR": single_fpr,
                "tta_FPR": tta_fpr,
                "delta_FPR": delta,
                "verdict": verdict,
            })

    fpr_csv = OUT_DIR / "fpr_comparison.csv"
    with fpr_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fpr_rows[0].keys()))
        w.writeheader()
        for r in fpr_rows:
            w.writerow(r)
    log.info("wrote FPR table: %s", fpr_csv)

    # Top-line summary booleans
    e2b_may6 = summary["by_substrate"].get("may6_falseflag", {}).get("E2B", {})
    p8a_may6 = summary["by_substrate"].get("may6_falseflag", {}).get("P8A", {})
    summary["headline"] = {
        "tta_damps_may6_E2B_below_20pct": (
            e2b_may6.get("tta_FPR", 1.0) < 0.20
            if e2b_may6 else None
        ),
        "tta_damps_may6_E2B_meaningfully": (
            (e2b_may6.get("single_view_FPR", 0) - e2b_may6.get("tta_FPR", 0)) >= 0.10
            if e2b_may6 else None
        ),
        "P8A_may6_invariance_preserved": (
            p8a_may6.get("tta_FPR", 1.0) <= 0.05
            if p8a_may6 else None
        ),
        "may6_E2B_single_view_FPR": e2b_may6.get("single_view_FPR"),
        "may6_E2B_tta_FPR": e2b_may6.get("tta_FPR"),
        "may6_P8A_single_view_FPR": p8a_may6.get("single_view_FPR"),
        "may6_P8A_tta_FPR": p8a_may6.get("tta_FPR"),
    }

    summary_json = OUT_DIR / "summary.json"
    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info("wrote summary: %s", summary_json)

    verdict_json = OUT_DIR / "verdict.json"
    with verdict_json.open("w") as f:
        json.dump(verdict_rows, f, indent=2)
    log.info("wrote verdicts: %s", verdict_json)


def main() -> int:
    log.info("=" * 70)
    log.info("TTA POC — 4-view test-time augmentation on may6 substrate")
    log.info("=" * 70)
    if not DETECTOR_CONFIG.exists():
        log.error("missing detector config: %s", DETECTOR_CONFIG)
        return 2
    if not TRAIN_CONFIG.exists():
        log.error("missing train config: %s", TRAIN_CONFIG)
        return 2
    frames = collect_frames()
    if not frames:
        log.error("no frames found")
        return 2

    per_ckpt_csvs: dict[str, Path] = {}
    for name, uri in CKPTS.items():
        try:
            csv_path = score_one_ckpt(name, uri, frames)
            per_ckpt_csvs[name] = csv_path
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] FAILED: %s", name, exc, exc_info=True)
            return 1

    aggregate_results(per_ckpt_csvs)
    log.info("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
