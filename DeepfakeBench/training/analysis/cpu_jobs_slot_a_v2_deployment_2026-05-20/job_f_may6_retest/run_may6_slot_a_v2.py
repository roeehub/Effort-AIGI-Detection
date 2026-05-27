"""may6 production-drift retest for Slot A v2 step3500 (+ P8A harness sanity).

Adapted from `analysis/r13_overnight_may6_retest_2026-05-13/run_t5c_step3500_may6.py`.

Slot A v2 step3500 (`hp35c51p`) is an FT from T5C step3500 with `anchor_aware`
enabled (no LoRA). Classifier head hidden_dim=1024 (T5C lineage). The saved
state_dict's `multi_axis_grl_block.*` and any `anchor_*` aux keys are dropped
via `load_state_dict(strict=False)` — those aux heads are not invoked when
`forward(..., inference=True)`.

Outputs:
  outputs/scores_slot_a_v2_step3500_may6.csv  — per-frame fake probabilities
  outputs/scores_p8a_sanity_may6.csv          — P8A re-score (harness drift check)
  outputs/may6_retest_table_with_slot_a_v2.csv — comparison table

Usage:
  python analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_f_may6_retest/run_may6_slot_a_v2.py
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from arena.model_arena import CLIP_MEAN, CLIP_STD  # noqa: E402
from batch_inference_gcs import load_model  # noqa: E402

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("may6_slot_a_v2")

# Local ckpt paths (already-cached by manual_canary_2026-05-20 + slot_b probe jobs)
SLOT_A_V2_CKPT = REPO_ROOT / (
    "analysis/manual_canary_2026-05-20/ckpts/"
    "periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth"
)
P8A_CKPT = REPO_ROOT / (
    "analysis/manual_canary_2026-05-20/ckpts/"
    "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth"
)

RAW_DIR = REPO_ROOT / "analysis/xinhe_cross_camera_audit_2026-05-06/raw/may6"
PRIOR_OUT_DIR = REPO_ROOT / "analysis/xinhe_cross_camera_audit_2026-05-06/outputs"
PRIOR_P8A_CSV = PRIOR_OUT_DIR / "scores_P8A.csv"

OUT_DIR = THIS_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Match the prior may6 harness (run_may6_retest.py uses defaults.yaml + effort.yaml).
DETECTOR_CONFIG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO_ROOT / "config/defaults.yaml"

try:
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
except Exception:
    DEVICE = torch.device("cpu")

RESOLUTION = 224
BATCH_SIZE = 8
NUM_WORKERS = 2


class LocalFrameDataset(Dataset):
    """Identical preprocessing path to run_may6_retest.py::LocalFrameDataset."""

    def __init__(self, frame_paths, resolution: int = RESOLUTION):
        self.paths = list(frame_paths)
        self.resolution = resolution
        self.transform = T.Compose([
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
            return torch.zeros(3, self.resolution, self.resolution), idx
        img_bgr = cv2.resize(img_bgr, (self.resolution, self.resolution),
                             interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(img_rgb), idx


def collect_may6_frames():
    paths = sorted(RAW_DIR.glob("*.png")) + sorted(RAW_DIR.glob("*.jpg"))
    log.info("collected: may6=%d frames from %s", len(paths), RAW_DIR)
    return paths


def score_ckpt(alias: str, ckpt_path: Path, frames: list[Path], out_csv: Path):
    """Load + score one ckpt on all may6 frames. Returns the score array."""
    log.info("[%s] loading model on %s from %s ...", alias, DEVICE, ckpt_path)
    t0 = time.time()
    model = load_model(str(ckpt_path), str(DETECTOR_CONFIG), str(TRAIN_CONFIG), DEVICE)
    log.info("[%s] model loaded in %.1fs", alias, time.time() - t0)

    ds = LocalFrameDataset(frames)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=False)

    n = len(frames)
    probs = np.zeros(n, dtype=np.float32)
    log.info("[%s] running inference: %d frames, batch=%d, device=%s",
             alias, n, BATCH_SIZE, DEVICE)
    t0 = time.time()
    for batch_idx, (images, indices) in enumerate(loader):
        images = images.to(DEVICE)
        with torch.inference_mode():
            outputs = model({"image": images}, inference=True)
            batch_probs = outputs["prob"].detach().cpu().numpy().reshape(-1)
        for i, gi in enumerate(indices.numpy()):
            probs[int(gi)] = float(batch_probs[i])
        elapsed = time.time() - t0
        done = (batch_idx + 1) * BATCH_SIZE
        log.info("[%s] batch %d (~%d/%d) elapsed=%.1fs",
                 alias, batch_idx + 1, min(done, n), n, elapsed)
    log.info("[%s] inference done in %.1fs", alias, time.time() - t0)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["population", "frame_path", "frame_basename", "prob_fake"])
        for path, prob in zip(frames, probs):
            w.writerow(["may6_falseflag", str(path), path.name, f"{prob:.6f}"])
    log.info("[%s] wrote -> %s", alias, out_csv)

    del model
    return probs


def percentile_pack(probs: np.ndarray) -> dict:
    return {
        "n": int(len(probs)),
        "n_fired@0.5": int((probs > 0.5).sum()),
        "p50": float(np.median(probs)),
        "p90": float(np.quantile(probs, 0.9)),
        "p99": float(np.quantile(probs, 0.99)),
        "max": float(probs.max()),
    }


def write_comparison_table(slot_a_v2_probs: np.ndarray, p8a_resc_probs: np.ndarray):
    """Append the SLOT_A_V2 row to a copy of the prior may6 retest table format."""
    # Prior baselines + candidates table (the established schema)
    prior_table = REPO_ROOT / "analysis/r13_overnight_may6_retest_2026-05-13/outputs/may6_retest_table.csv"
    rows = []
    if prior_table.exists():
        prior_df = pd.read_csv(prior_table)
        for _, r in prior_df.iterrows():
            rows.append({
                "ckpt": str(r["ckpt"]),
                "source": str(r["source"]),
                "n": int(r["n"]),
                "n_fired@0.5": int(r["n_fired@0.5"]),
                "p50": float(r["p50"]),
                "p90": float(r["p90"]),
                "p99": float(r["p99"]),
                "max": float(r["max"]),
            })
    else:
        log.warning("prior table not found: %s — table will only contain Slot A v2 row", prior_table)

    p_sa = percentile_pack(slot_a_v2_probs)
    rows.append({"ckpt": "SLOT_A_V2_STEP3500", "source": "job_f (this retest)", **p_sa})

    p_re = percentile_pack(p8a_resc_probs)
    rows.append({"ckpt": "P8A_STEP5000_HARNESS_SANITY", "source": "job_f (harness re-score)", **p_re})

    out_csv = OUT_DIR / "may6_retest_table_with_slot_a_v2.csv"
    fieldnames = ["ckpt", "source", "n", "n_fired@0.5", "p50", "p90", "p99", "max"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log.info("wrote table -> %s", out_csv)
    return out_csv


def compare_p8a_resc_vs_prior(p8a_resc_probs: np.ndarray, frames: list[Path]) -> dict:
    """Compare the re-scored P8A to the prior CSV (numerical-noise sanity check)."""
    prior = pd.read_csv(PRIOR_P8A_CSV)
    prior = prior[prior["population"] == "may6_falseflag"].reset_index(drop=True)
    # Align by frame basename in sorted order (same ordering by spec)
    resc_df = pd.DataFrame({
        "frame_basename": [p.name for p in frames],
        "prob_fake_resc": p8a_resc_probs,
    })
    merged = resc_df.merge(prior[["frame_basename", "prob_fake"]],
                           on="frame_basename", how="inner")
    if len(merged) != len(p8a_resc_probs):
        log.warning("p8a sanity join lost frames: %d vs %d", len(merged), len(p8a_resc_probs))
    delta = (merged["prob_fake_resc"] - merged["prob_fake"]).to_numpy()
    return {
        "n_joined": int(len(merged)),
        "n_fired_resc": int((merged["prob_fake_resc"] > 0.5).sum()),
        "n_fired_prior": int((merged["prob_fake"] > 0.5).sum()),
        "max_abs_delta": float(np.max(np.abs(delta))),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
    }


def main():
    log.info("=" * 70)
    log.info("may6 production-drift retest — Slot A v2 step3500 + P8A sanity")
    log.info("device=%s, batch=%d, workers=%d", DEVICE, BATCH_SIZE, NUM_WORKERS)
    log.info("=" * 70)

    for p in (SLOT_A_V2_CKPT, P8A_CKPT):
        if not p.exists():
            log.error("ckpt missing: %s", p)
            return 2

    frames = collect_may6_frames()
    if len(frames) != 92:
        log.warning("expected 92 may6 frames, found %d", len(frames))

    # Score Slot A v2 step3500
    slot_a_v2_csv = OUT_DIR / "scores_slot_a_v2_step3500_may6.csv"
    if slot_a_v2_csv.exists():
        log.info("Slot A v2 CSV exists, reusing: %s", slot_a_v2_csv)
        sa_df = pd.read_csv(slot_a_v2_csv)
        sa_df = sa_df[sa_df["population"] == "may6_falseflag"].reset_index(drop=True)
        slot_a_v2_probs = sa_df["prob_fake"].to_numpy(dtype=np.float32)
    else:
        slot_a_v2_probs = score_ckpt("SLOT_A_V2_STEP3500", SLOT_A_V2_CKPT, frames, slot_a_v2_csv)

    # Score P8A sanity
    p8a_resc_csv = OUT_DIR / "scores_p8a_sanity_may6.csv"
    if p8a_resc_csv.exists():
        log.info("P8A sanity CSV exists, reusing: %s", p8a_resc_csv)
        p8a_df = pd.read_csv(p8a_resc_csv)
        p8a_df = p8a_df[p8a_df["population"] == "may6_falseflag"].reset_index(drop=True)
        p8a_resc_probs = p8a_df["prob_fake"].to_numpy(dtype=np.float32)
    else:
        p8a_resc_probs = score_ckpt("P8A_SANITY_STEP5000", P8A_CKPT, frames, p8a_resc_csv)

    # Stats and comparison
    sa_stats = percentile_pack(slot_a_v2_probs)
    p8a_stats = percentile_pack(p8a_resc_probs)
    p8a_drift = compare_p8a_resc_vs_prior(p8a_resc_probs, frames)

    log.info("=" * 80)
    log.info("SLOT_A_V2_STEP3500 — may6 result")
    log.info("=" * 80)
    for k, v in sa_stats.items():
        log.info("  %s = %s", k, v)

    log.info("=" * 80)
    log.info("P8A re-score (harness sanity)")
    log.info("=" * 80)
    for k, v in p8a_stats.items():
        log.info("  %s = %s", k, v)
    log.info("vs prior P8A CSV: max_abs_delta=%.6f mean_abs_delta=%.6f n_fired_resc=%d (prior=%d)",
             p8a_drift["max_abs_delta"], p8a_drift["mean_abs_delta"],
             p8a_drift["n_fired_resc"], p8a_drift["n_fired_prior"])

    summary_path = OUT_DIR / "slot_a_v2_may6_summary.json"
    with summary_path.open("w") as f:
        json.dump({
            "slot_a_v2_step3500": sa_stats,
            "p8a_sanity_step5000": p8a_stats,
            "p8a_harness_drift_vs_prior": p8a_drift,
        }, f, indent=2)
    log.info("wrote summary -> %s", summary_path)

    write_comparison_table(slot_a_v2_probs, p8a_resc_probs)
    log.info("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
