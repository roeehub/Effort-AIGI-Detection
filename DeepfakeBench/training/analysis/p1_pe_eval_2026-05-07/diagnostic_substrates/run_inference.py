"""
Phase A.5 — diagnostic-substrate inference for P1 (PE_PAIR_RANK_DRO).
2026-05-07.

Why: the 29-suite contract (Phase A) frozen 2026-04-23 does not include the
substrates that drove this week's investigation — xinhe_may6_falseflag,
live_*_teams_prod variants, dor_evening/dor_morning, team_sanity_may5,
dor_fake_local, extra. Plan §3.1 tabulates them; without scoring them here
P1's verdict on the failure modes that motivated it is unmeasured.

Method: load grouped_manifest_v2.csv (already has score_P8A / score_E2B /
score_PA_3800), filter to suites NOT in the contract, run CPU inference
on the 2 P1 final-best ckpts (BUNDLE step4000, PAIRRANK step6750), append
score_P1_BUNDLE_step4000 / score_P1_PAIRRANK_step6750 columns idempotently,
and emit a per-substrate comparison report at τ=0.5.

Usage:
    cd analysis/p1_pe_eval_2026-05-07/diagnostic_substrates
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

REPO_ROOT = Path(__file__).resolve().parents[3]  # DeepfakeBench/training
sys.path.insert(0, str(REPO_ROOT))

from arena.model_arena import (  # noqa: E402
    FrameRecord,
    GCSFrameDataset,
    _download_checkpoint,
    load_model,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")
log = logging.getLogger("phase_a5")

# ----------------------------------------------------------------- paths

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR
GROUPED_MANIFEST = REPO_ROOT / "analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv"
DETECTOR_CONFIG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO_ROOT / "config/defaults.yaml"

# Reuse cached ckpts from Phase E.
PHASE_E_CACHE = REPO_ROOT / "analysis/p1_pe_eval_2026-05-07/weight_delta/ckpt_cache"

# ----------------------------------------------------------------- ckpts

CKPTS: dict[str, str] = {
    "P1_BUNDLE_step4000":   "gs://training-job-outputs/best_checkpoints/tznuar61/top_n_effort_20260507_step4000_auc0.9951_eer0.0210.pth",
    "P1_PAIRRANK_step6750": "gs://training-job-outputs/best_checkpoints/s2mp5fxm/top_n_effort_20260507_step6750_auc0.9949_eer0.0119.pth",
}

# Suites in the 29-suite contract — Phase A scores these on GPU. Skip here.
CONTRACT_SUITES = {
    "teams_real_all_dev",
    "teams_real_all_lockbox",
    "teams_fake_all_dev",
    "teams_fake_all_lockbox",
    "teams_real_dor_dev",
}

DEVICE = torch.device("cpu")
RESOLUTION = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

# ----------------------------------------------------------------- helpers

def parse_gcs_path(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"not a gs:// URI: {uri}")
    no_scheme = uri[5:]
    bucket, blob = no_scheme.split("/", 1)
    return bucket, blob


def manifest_to_records(df: pd.DataFrame) -> list[FrameRecord]:
    records = []
    for _, row in df.iterrows():
        bucket, blob = parse_gcs_path(row["frame_path"])
        records.append(FrameRecord(
            bucket=bucket,
            blob_path=blob,
            label=int(row["label"]),
            method="diagnostic",
            video_id=str(row.get("video_id", "")),
            frame_name=Path(blob).name,
        ))
    return records


def cached_ckpt_path(uri: str) -> Path:
    """Reuse Phase E's ckpt cache to avoid re-downloads."""
    name = Path(uri).name
    cached = PHASE_E_CACHE / name
    if cached.exists():
        return cached
    return Path(_download_checkpoint(uri))


def score_ckpt(ckpt_name: str, ckpt_uri: str, df: pd.DataFrame) -> np.ndarray:
    log.info("[%s] resolving ckpt", ckpt_name)
    local = cached_ckpt_path(ckpt_uri)
    log.info("[%s] using ckpt: %s", ckpt_name, local)

    log.info("[%s] loading model on %s", ckpt_name, DEVICE)
    t0 = time.time()
    model = load_model(str(local), str(DETECTOR_CONFIG), str(TRAIN_CONFIG), DEVICE)
    log.info("[%s] model loaded in %.1fs", ckpt_name, time.time() - t0)

    records = manifest_to_records(df)
    dataset = GCSFrameDataset(records, resolution=RESOLUTION)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    n = len(records)
    probs = np.full(n, np.nan, dtype=np.float32)
    log.info("[%s] inference: n=%d batch=%d workers=%d", ckpt_name, n, BATCH_SIZE, NUM_WORKERS)
    t0 = time.time()
    last_log = t0
    for batch_idx, (images, indices) in enumerate(loader):
        images = images.to(DEVICE)
        with torch.inference_mode():
            outputs = model({"image": images}, inference=True)
            batch_probs = outputs["prob"].detach().cpu().numpy().reshape(-1)
        for i, gi in enumerate(indices.numpy()):
            probs[int(gi)] = float(batch_probs[i])
        now = time.time()
        if now - last_log >= 30 or (batch_idx + 1) * BATCH_SIZE >= n:
            done = min((batch_idx + 1) * BATCH_SIZE, n)
            elapsed = now - t0
            rate = done / max(elapsed, 1e-3)
            eta = (n - done) / max(rate, 1e-3)
            log.info("[%s] batch %d  done=%d/%d  elapsed=%.0fs  fps=%.1f  eta=%.0fs",
                     ckpt_name, batch_idx + 1, done, n, elapsed, rate, eta)
            last_log = now

    log.info("[%s] inference done in %.1fs", ckpt_name, time.time() - t0)
    del model
    return probs


def per_suite_report(df: pd.DataFrame, score_cols: list[str], tau: float = 0.5) -> pd.DataFrame:
    """Per-suite FPR (label==0) and recall (label==1) at fixed τ for all score cols."""
    rows = []
    for suite, gdf in df.groupby("suite", observed=True):
        n = len(gdf)
        n_real = int((gdf["label"] == 0).sum())
        n_fake = int((gdf["label"] == 1).sum())
        row = {"suite": suite, "n_total": n, "n_real": n_real, "n_fake": n_fake}
        for col in score_cols:
            scores = gdf[col].values
            mask_real = (gdf["label"] == 0).values & ~np.isnan(scores)
            mask_fake = (gdf["label"] == 1).values & ~np.isnan(scores)
            fpr = (scores[mask_real] >= tau).mean() if mask_real.sum() > 0 else float("nan")
            rec = (scores[mask_fake] >= tau).mean() if mask_fake.sum() > 0 else float("nan")
            row[f"{col}_FPR"] = round(fpr, 4) if not np.isnan(fpr) else float("nan")
            row[f"{col}_recall"] = round(rec, 4) if not np.isnan(rec) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("suite")


# ----------------------------------------------------------------- main

def main() -> None:
    log.info("=" * 70)
    log.info("Phase A.5 — diagnostic substrates inference (P1 BUNDLE + PAIRRANK)")
    log.info("=" * 70)

    if not GROUPED_MANIFEST.exists():
        log.error("manifest missing: %s", GROUPED_MANIFEST)
        sys.exit(2)

    df = pd.read_csv(GROUPED_MANIFEST)
    log.info("loaded manifest: %d rows, %d suites", len(df), df["suite"].nunique())

    is_diag = ~df["suite"].isin(CONTRACT_SUITES)
    diag = df[is_diag].copy().reset_index(drop=True)
    log.info("filtered to diagnostic substrates: %d rows", len(diag))
    log.info("substrates kept: %s", sorted(diag["suite"].unique()))

    score_cols = []
    for ckpt_name, ckpt_uri in CKPTS.items():
        col = f"score_{ckpt_name}"
        score_cols.append(col)
        log.info("=== scoring ckpt: %s ===", ckpt_name)
        probs = score_ckpt(ckpt_name, ckpt_uri, diag)
        diag[col] = probs
        # checkpoint partial output in case of crash
        diag.to_csv(OUT_DIR / f"diagnostic_scores_partial_{ckpt_name}.csv", index=False)
        log.info("saved partial scores for %s (n=%d)", ckpt_name, len(diag))

    # idempotent merge back into the master manifest (replace cols if exist)
    log.info("merging scores into master manifest")
    master = pd.read_csv(GROUPED_MANIFEST)
    join_keys = ["suite", "frame_path"]
    keep_cols = join_keys + score_cols
    merged = master.merge(diag[keep_cols], on=join_keys, how="left", suffixes=("", "__new"))
    for col in score_cols:
        if f"{col}__new" in merged.columns:
            merged[col] = merged[f"{col}__new"]
            merged.drop(columns=[f"{col}__new"], inplace=True)

    out_csv = OUT_DIR / "grouped_manifest_v2_with_p1.csv"
    merged.to_csv(out_csv, index=False)
    log.info("wrote: %s", out_csv)

    # Per-substrate comparison report at τ=0.5
    all_score_cols = [c for c in [
        "score_P8A", "score_E2B", "score_PA_3800",
        "score_P1_BUNDLE_step4000", "score_P1_PAIRRANK_step6750",
    ] if c in merged.columns]
    report = per_suite_report(merged[is_diag.values | ~is_diag.values], all_score_cols, tau=0.5)
    report_csv = OUT_DIR / "per_suite_comparison_tau_0.5.csv"
    report.to_csv(report_csv)
    log.info("wrote: %s", report_csv)
    print()
    print("=" * 100)
    print(f"PER-SUITE COMPARISON @ τ=0.5  (FPR=label0; recall=label1)")
    print("=" * 100)
    print(report.to_string())


if __name__ == "__main__":
    main()
