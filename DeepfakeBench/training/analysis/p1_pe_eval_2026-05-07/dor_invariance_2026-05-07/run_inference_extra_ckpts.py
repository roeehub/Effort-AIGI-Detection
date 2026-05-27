"""
Extension of run_inference.py — scores the 180-frame dor invariance manifest
with the 4 additional P1 ckpts to complete the trajectory.

Already scored (in scores_full.csv): P8A, E2B, P1_BUNDLE_step4000,
P1_PAIRRANK_step6750.

Adds (4 new score columns appended in place):
    - P1_BUNDLE_PERIODIC_STEP500    (tznuar61)
    - P1_BUNDLE_TOP_N_STEP3750      (tznuar61)
    - P1_PAIRRANK_PERIODIC_STEP500  (s2mp5fxm)
    - P1_PAIRRANK_TOP_N_STEP6000    (s2mp5fxm)

Per CLAUDE.md / handoff: APPEND new columns, don't regenerate the 4 already
present. Reuses Phase E's local ckpt cache; downloads any missing ones.

Usage:
    cd analysis/p1_pe_eval_2026-05-07/dor_invariance_2026-05-07
    python run_inference_extra_ckpts.py
"""
from __future__ import annotations

import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from arena.model_arena import (  # noqa: E402
    FrameRecord,
    GCSFrameDataset,
    _download_checkpoint,
    load_model,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")
log = logging.getLogger("dor_inv_extra")

THIS_DIR = Path(__file__).resolve().parent
MANIFEST = THIS_DIR / "manifest.csv"
SCORES_FULL = THIS_DIR / "scores_full.csv"
DETECTOR_CONFIG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO_ROOT / "config/defaults.yaml"
PHASE_E_CACHE = REPO_ROOT / "analysis/p1_pe_eval_2026-05-07/weight_delta/ckpt_cache"

# 4 new ckpts (the 2 already done in scores_full.csv are deliberately omitted).
NEW_CKPTS: dict[str, str] = {
    "P1_BUNDLE_PERIODIC_STEP500":
        "gs://training-job-outputs/best_checkpoints/tznuar61/periodic_effort_20260506_step500_auc0.9784_eer0.0735.pth",
    "P1_BUNDLE_TOP_N_STEP3750":
        "gs://training-job-outputs/best_checkpoints/tznuar61/top_n_effort_20260506_step3750_auc0.9947_eer0.0168.pth",
    "P1_PAIRRANK_PERIODIC_STEP500":
        "gs://training-job-outputs/best_checkpoints/s2mp5fxm/periodic_effort_20260506_step500_auc0.9848_eer0.0500.pth",
    "P1_PAIRRANK_TOP_N_STEP6000":
        "gs://training-job-outputs/best_checkpoints/s2mp5fxm/top_n_effort_20260507_step6000_auc0.9939_eer0.0095.pth",
}

DEVICE = torch.device("cpu")
RESOLUTION = 224
BATCH_SIZE = 32
NUM_WORKERS = 0


def parse_gcs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(uri)
    bucket, blob = uri[5:].split("/", 1)
    return bucket, blob


def manifest_to_records(df: pd.DataFrame) -> list[FrameRecord]:
    records = []
    for _, row in df.iterrows():
        bucket, blob = parse_gcs(row["frame_path"])
        records.append(FrameRecord(
            bucket=bucket, blob_path=blob, label=int(row["label"]),
            method="dor_inv", video_id=row["variant"],
            frame_name=Path(blob).name,
        ))
    return records


def cached_ckpt_path(uri: str) -> Path:
    name = Path(uri).name
    cached = PHASE_E_CACHE / name
    if cached.exists():
        log.info("ckpt cached locally: %s", cached)
        return cached
    log.info("ckpt NOT cached, downloading: %s", uri)
    return Path(_download_checkpoint(uri))


def score(ckpt_name: str, ckpt_uri: str, records: list[FrameRecord]) -> np.ndarray:
    log.info("[%s] resolving ckpt", ckpt_name)
    local = cached_ckpt_path(ckpt_uri)
    log.info("[%s] loading model from %s", ckpt_name, local.name)
    t0 = time.time()
    model = load_model(str(local), str(DETECTOR_CONFIG), str(TRAIN_CONFIG), DEVICE)
    log.info("[%s] loaded in %.1fs", ckpt_name, time.time() - t0)

    ds = GCSFrameDataset(records, resolution=RESOLUTION)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                        shuffle=False, pin_memory=False)

    n = len(records)
    probs = np.full(n, np.nan, dtype=np.float32)
    log.info("[%s] inference on %d frames", ckpt_name, n)
    t0 = time.time()
    for batch_idx, (images, idxs) in enumerate(loader):
        with torch.inference_mode():
            out = model({"image": images.to(DEVICE)}, inference=True)
            bp = out["prob"].detach().cpu().numpy().reshape(-1)
        for i, gi in enumerate(idxs.numpy()):
            probs[int(gi)] = float(bp[i])
    log.info("[%s] done in %.1fs", ckpt_name, time.time() - t0)
    del model, ds, loader
    gc.collect()
    return probs


def main() -> None:
    if not SCORES_FULL.exists():
        raise FileNotFoundError(
            f"Expected {SCORES_FULL} from prior run_inference.py invocation"
        )
    df = pd.read_csv(SCORES_FULL)
    log.info("loaded existing scores_full.csv: %d rows, cols=%s",
             len(df), list(df.columns))
    if len(df) != 180:
        raise RuntimeError(f"expected 180 rows, got {len(df)}")

    records = manifest_to_records(df)

    # Skip ckpts already present (idempotent).
    for ckpt_name, uri in NEW_CKPTS.items():
        col = f"score_{ckpt_name}"
        if col in df.columns:
            log.info("[%s] column already present, skipping", ckpt_name)
            continue
        df[col] = score(ckpt_name, uri, records)
        # Write partial after every ckpt for crash safety.
        df.to_csv(SCORES_FULL, index=False)
        log.info("appended %s, scores_full.csv now has %d cols",
                 col, len(df.columns))

    log.info("FINAL: scores_full.csv has %d rows × %d cols",
             len(df), len(df.columns))
    log.info("score columns: %s", [c for c in df.columns if c.startswith("score_")])

    # Per-variant FPR @ τ=0.5 for the 6 P1 ckpts (extension table).
    p1_score_cols = [c for c in df.columns if c.startswith("score_P1_")]
    p1_score_cols = sorted(p1_score_cols, key=lambda c: c.lower())
    log.info("p1 score columns for extended FPR table: %s", p1_score_cols)

    print()
    print("=" * 100)
    print("DOR INVARIANCE @ τ=0.5  —  6 P1 ckpts × 6 dor variants  (FPR; lower is better)")
    print("=" * 100)
    summary = df.groupby("variant")[p1_score_cols].apply(
        lambda g: pd.Series({
            f"{c}_FPR": float((g[c] >= 0.5).mean()) for c in p1_score_cols
        })
    )
    summary["n"] = df.groupby("variant").size()
    print(summary.round(4).to_string())

    summary.to_csv(THIS_DIR / "per_variant_fpr_extended.csv")
    print()
    print(f"=> wrote {SCORES_FULL} and per_variant_fpr_extended.csv")


if __name__ == "__main__":
    main()
