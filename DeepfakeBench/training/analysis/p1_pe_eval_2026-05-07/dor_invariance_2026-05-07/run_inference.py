"""
Dor invariance probe for P1 (PE_PAIR_RANK_DRO).
2026-05-07.

Built because Phase A.5's `dor_evening` / `dor_morning` / `dor_fake_local`
suites in `analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv`
all have stale `gs://local/...` placeholder paths and silently returned
zero-tensor scores. The dor invariance question (does P1 preserve P8A's
0% FPR on Dor reals across capture conditions?) is load-bearing for the
P1 verdict and not measurable from grouped_manifest_v2 in its current
state.

This script enumerates 180 currently-extant Dor real frames spread across
6 capture conditions in `gs://real-teams-dor-roee/`:

  - dor_laptop_whiteish (n=30) — laptop camera, white-shifted lighting
  - dor_laptop_yellowish (n=30) — laptop camera, yellow-shifted lighting
  - dor_webcam_no_vbg (n=30) — *the canonical 2-camera-test substrate*
    (memory project_signature_shortcut_finding.md: P8A 0.02, RLP6_04 0.94)
  - dor_webcam_with_vbg (n=30) — webcam + virtual background
  - dor_session_0411 (n=30) — uniform30 sample, 2026-04-11 session
  - dor_session_0424 (n=30) — uniform30 sample, 2026-04-24 session

Scores 4 ckpts: P8A, E2B (deployed), P1 BUNDLE step4000, P1 PAIRRANK step6750.
Reuses Phase E's local ckpt cache. All 180 frames have valid GCS paths
(verified via gcloud storage ls).

Usage:
    cd analysis/p1_pe_eval_2026-05-07/dor_invariance_2026-05-07
    python run_inference.py
"""
from __future__ import annotations

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
log = logging.getLogger("dor_inv")

THIS_DIR = Path(__file__).resolve().parent
MANIFEST = THIS_DIR / "manifest.csv"
DETECTOR_CONFIG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO_ROOT / "config/defaults.yaml"
PHASE_E_CACHE = REPO_ROOT / "analysis/p1_pe_eval_2026-05-07/weight_delta/ckpt_cache"

CKPTS: dict[str, str] = {
    "P8A":                  "gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth",
    "E2B":                  "gs://training-job-outputs/best_checkpoints/rmat8lwx/top_n_effort_20260503_step3200_auc0.9863_eer0.0310.pth",
    "P1_BUNDLE_step4000":   "gs://training-job-outputs/best_checkpoints/tznuar61/top_n_effort_20260507_step4000_auc0.9951_eer0.0210.pth",
    "P1_PAIRRANK_step6750": "gs://training-job-outputs/best_checkpoints/s2mp5fxm/top_n_effort_20260507_step6750_auc0.9949_eer0.0119.pth",
}

DEVICE = torch.device("cpu")
RESOLUTION = 224
BATCH_SIZE = 32
NUM_WORKERS = 4


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
        return cached
    return Path(_download_checkpoint(uri))


def score(ckpt_name: str, ckpt_uri: str, records: list[FrameRecord]) -> np.ndarray:
    log.info("[%s] resolving ckpt", ckpt_name)
    local = cached_ckpt_path(ckpt_uri)
    log.info("[%s] loading model", ckpt_name)
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
    del model
    return probs


def main() -> None:
    df = pd.read_csv(MANIFEST)
    log.info("manifest: %d rows, %d variants", len(df), df["variant"].nunique())
    records = manifest_to_records(df)

    score_cols: list[str] = []
    for ckpt_name, uri in CKPTS.items():
        col = f"score_{ckpt_name}"
        score_cols.append(col)
        df[col] = score(ckpt_name, uri, records)
        df.to_csv(THIS_DIR / "scores_partial.csv", index=False)

    df.to_csv(THIS_DIR / "scores_full.csv", index=False)
    log.info("wrote scores_full.csv")

    # Per-variant FPR @ τ=0.5 (all frames are reals, label=0).
    print()
    print("=" * 100)
    print("DOR INVARIANCE @ τ=0.5  (real-pool FPR; lower is better)")
    print("=" * 100)
    summary = df.groupby("variant")[score_cols].apply(
        lambda g: pd.Series({
            f"{c}_FPR": float((g[c] >= 0.5).mean()) for c in score_cols
        })
    )
    summary["n"] = df.groupby("variant").size()
    print(summary.round(4).to_string())

    # All-Dor row
    all_row = {f"{c}_FPR": float((df[c] >= 0.5).mean()) for c in score_cols}
    all_row["n"] = len(df)
    print()
    print("ALL DOR (combined):")
    for k, v in all_row.items():
        if k == "n":
            print(f"  n: {v}")
        else:
            print(f"  {k}: {v:.4f}")

    summary.to_csv(THIS_DIR / "per_variant_fpr.csv")
    print()
    print(f"=> wrote {THIS_DIR / 'scores_full.csv'} and per_variant_fpr.csv")


if __name__ == "__main__":
    main()
