#!/usr/bin/env python3
"""Post-fix re-score of RLP6_04 on the 6 Dor/Roee pools (pre-launch gate).

Goal: score the RLP6_04 checkpoint against the 6 Dor/Roee pool folders in
gs://real-teams-dor-roee/session_20260424_combined_tags_121458_121007/, using
the NOW-FIXED arena inference path (INTER_LINEAR), and compare to the pre-fix
means in combined_frame_tags.json (which came from the deploy server at
http://34.16.217.28:8999, preprocessing unknown).

Writes per-frame CSV + a summary JSON:
  analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json

Run:
  python analysis/rlp6_04_postfix_rescore_2026-04-24.py \
      [--device cpu|mps|cuda] [--batch-size 16] [--num-workers 0]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_TRAINING_ROOT = _HERE.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

import numpy as np

# Reuse the fixed inference path
from arena.model_arena import (
    FrameRecord,
    GCSFrameDataset,
    _download_checkpoint,
    load_model,
)

BUCKET = "real-teams-dor-roee"
PREFIX = "session_20260424_combined_tags_121458_121007"

POOLS = [
    "dor-real-laptop-correct-no-virtual-bg-whiteish",
    "dor-real-laptop-correct-no-virtual-bg-yellowish",
    "roee-real-windows-laptop-correct",
    "dor-real-webcam-false-flag",
    "dor-real-webcam-false-flag-no-virtual-bg",
    "roee-mac-laptop-false-flag-virtual-bg",
]

PREFIX_MEANS = {
    "dor-real-laptop-correct-no-virtual-bg-whiteish": 0.018,
    "dor-real-laptop-correct-no-virtual-bg-yellowish": 0.040,
    "roee-real-windows-laptop-correct": 0.007,
    "dor-real-webcam-false-flag": 0.878,
    "dor-real-webcam-false-flag-no-virtual-bg": 0.940,
    "roee-mac-laptop-false-flag-virtual-bg": 0.900,
}

CHECKPOINT_GCS = (
    "gs://training-job-outputs/phase2r13_experiments/h2pdu6i5/"
    "value_composite_effort_20260424_step23500_auc0.9942_eer0.0169.pth"
)
DETECTOR_CFG = str(_TRAINING_ROOT / "config" / "detector" / "effort.yaml")
TRAIN_CFG = str(_TRAINING_ROOT / "config" / "defaults.yaml")

OUT_JSON = _HERE.parent / "rlp6_04_postfix_rescore_2026-04-24.summary.json"
OUT_CSV = _HERE.parent / "rlp6_04_postfix_rescore_2026-04-24.per_frame.csv"


def discover_pool_records(client, pool: str) -> list[FrameRecord]:
    bucket = client.bucket(BUCKET)
    prefix = f"{PREFIX}/{pool}/"
    records = []
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue
        if not blob.name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        fname = os.path.basename(blob.name)
        records.append(
            FrameRecord(
                bucket=BUCKET,
                blob_path=blob.name,
                label=0,  # all real
                method="real",
                video_id=pool,
                frame_name=fname,
                strategy=pool,
            )
        )
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    import torch
    from google.cloud import storage

    if args.device == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)
    print(f"[info] device: {device_str}", flush=True)

    print(f"[info] downloading checkpoint: {CHECKPOINT_GCS}", flush=True)
    t0 = time.time()
    local_ckpt = _download_checkpoint(CHECKPOINT_GCS)
    print(f"[info] checkpoint local: {local_ckpt} ({time.time()-t0:.1f}s)", flush=True)

    print("[info] loading model ...", flush=True)
    model = load_model(local_ckpt, DETECTOR_CFG, TRAIN_CFG, device)

    print("[info] discovering frames in all 6 pools ...", flush=True)
    client = storage.Client()
    pool_records = {p: discover_pool_records(client, p) for p in POOLS}
    for p, recs in pool_records.items():
        print(f"  {p}: {len(recs)} frames")

    all_records: list[FrameRecord] = []
    for p in POOLS:
        all_records.extend(pool_records[p])
    print(f"[info] total frames: {len(all_records)}", flush=True)

    dataset = GCSFrameDataset(all_records)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device_str == "cuda"),
    )

    probs = np.zeros(len(all_records), dtype=np.float32)
    statuses = ["ok"] * len(all_records)
    t0 = time.time()
    for bi, (images, indices) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        with torch.inference_mode():
            outputs = model({"image": images}, inference=True)
            batch_probs = outputs["prob"].detach().cpu().numpy().reshape(-1)
        for i, gi in enumerate(indices.numpy()):
            probs[int(gi)] = float(batch_probs[i])
            if bool(images[i].sum().item() == 0.0):
                statuses[int(gi)] = "failed_decode"
        if (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
            elapsed = time.time() - t0
            rate = ((bi + 1) * args.batch_size) / max(elapsed, 1e-6)
            print(f"  batch {bi+1}/{len(loader)} ({elapsed:.1f}s, {rate:.1f} fr/s)",
                  flush=True)

    # Write per-frame CSV
    import csv
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool", "blob_path", "frame_name", "prob_fake", "status"])
        for i, rec in enumerate(all_records):
            w.writerow([rec.video_id, rec.blob_path, rec.frame_name,
                        f"{probs[i]:.6f}", statuses[i]])
    print(f"[info] wrote per-frame CSV: {OUT_CSV}", flush=True)

    # Aggregate per-pool
    summary = {
        "run_generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": CHECKPOINT_GCS,
        "n_frames_total": len(all_records),
        "device": device_str,
        "pools": [],
    }
    print("\n" + "=" * 96)
    print(f"{'pool':<50} {'N':>3} {'pre':>7} {'post':>7} {'delta':>7} {'min':>6} {'max':>6}")
    print("-" * 96)
    for p in POOLS:
        recs = pool_records[p]
        idxs = [i for i, rec in enumerate(all_records) if rec.video_id == p]
        ps = [float(probs[i]) for i in idxs if statuses[i] == "ok"]
        n_ok = len(ps)
        pre_mean = PREFIX_MEANS[p]
        mean = statistics.mean(ps) if ps else None
        std = statistics.stdev(ps) if len(ps) > 1 else 0.0
        mn = min(ps) if ps else None
        mx = max(ps) if ps else None
        frac_gt_09 = sum(1 for x in ps if x > 0.9) / len(ps) if ps else None
        frac_lt_01 = sum(1 for x in ps if x < 0.1) / len(ps) if ps else None

        delta = (mean - pre_mean) if mean is not None else None
        entry = {
            "pool": p,
            "n_frames": n_ok,
            "pre_fix_mean": pre_mean,
            "post_fix_mean": mean,
            "delta_post_minus_pre": delta,
            "std": std,
            "min": mn,
            "max": mx,
            "frac_gt_0_9": frac_gt_09,
            "frac_lt_0_1": frac_lt_01,
        }
        summary["pools"].append(entry)
        print(f"{p:<50} {n_ok:>3} {pre_mean:>7.3f} "
              f"{(mean if mean is not None else float('nan')):>7.3f} "
              f"{(delta if delta is not None else float('nan')):>+7.3f} "
              f"{(mn if mn is not None else float('nan')):>6.3f} "
              f"{(mx if mx is not None else float('nan')):>6.3f}")

    # Decision-tree evaluation (mirrors handoff step 2)
    webcam = next(p for p in summary["pools"]
                  if p["pool"] == "dor-real-webcam-false-flag-no-virtual-bg")
    webcam_delta = webcam.get("delta_post_minus_pre")
    if webcam_delta is None:
        verdict = "no_data"
    elif webcam_delta <= -0.30:
        verdict = "preprocessing_dominated__descope_aug_and_rerun_probe"
    elif webcam_delta <= -0.10:
        verdict = "preprocessing_partial__launch_1or2_experiments"
    elif webcam_delta >= 0.0:
        verdict = "post_fix_scores_not_lower__investigate"
    else:  # -0.10 < delta < 0.0 (drop under 10 pts)
        verdict = "fully_structural__launch_full_subset"
    summary["decision_tree"] = {
        "anchor_pool": "dor-real-webcam-false-flag-no-virtual-bg",
        "anchor_delta": webcam_delta,
        "verdict": verdict,
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\n[info] wrote summary JSON: {OUT_JSON}")
    print(f"[verdict] anchor_delta={webcam_delta!r} -> {verdict}")


if __name__ == "__main__":
    main()
