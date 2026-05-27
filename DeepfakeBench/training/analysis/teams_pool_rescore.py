#!/usr/bin/env python3
"""Parameterized re-score of any checkpoint on the 6 Dor/Roee pools.

Usage:
  python analysis/teams_pool_rescore.py \
      --checkpoint gs://training-job-outputs/phase2r13_experiments/<run_id>/value_composite_*.pth \
      --label rlp7_02 \
      [--device cpu|mps|cuda] [--batch-size 16] [--num-workers 0]

Writes:
  analysis/pool_rescore_<label>.summary.json
  analysis/pool_rescore_<label>.per_frame.csv

Baseline to diff against: analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json
(use post_fix_mean field per pool).
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

# RLP6_04 post-fix baseline (from analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json).
# All new runs are compared against these means.
RLP6_04_BASELINE = {
    "dor-real-laptop-correct-no-virtual-bg-whiteish": 0.012597115151584149,
    "dor-real-laptop-correct-no-virtual-bg-yellowish": 0.014436461481576164,
    "roee-real-windows-laptop-correct": 0.006001271059115728,
    "dor-real-webcam-false-flag": 0.9640364785989125,
    "dor-real-webcam-false-flag-no-virtual-bg": 0.9316849172115326,
    "roee-mac-laptop-false-flag-virtual-bg": 0.7514556576808293,
}

ANCHOR_POOL = "dor-real-webcam-false-flag-no-virtual-bg"

# "Correct real" pools — i.e. real frames the model SHOULD score low.
# Used by the per-step monitor to penalise FPR regression on these.
CORRECT_REAL_POOLS = (
    "dor-real-laptop-correct-no-virtual-bg-whiteish",
    "dor-real-laptop-correct-no-virtual-bg-yellowish",
    "roee-real-windows-laptop-correct",
)

DETECTOR_CFG = str(_TRAINING_ROOT / "config" / "detector" / "effort.yaml")
TRAIN_CFG = str(_TRAINING_ROOT / "config" / "defaults.yaml")


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
                label=0,
                method="real",
                video_id=pool,
                frame_name=fname,
                strategy=pool,
            )
        )
    return records


# ── Reusable in-memory anchor monitor ─────────────────────────────────
# Used by ``trainer/trainer.py::_run_validation`` to compute per-step pool
# metrics directly against the live model (no checkpoint round-trip). The CLI
# above is preserved for offline re-scoring. Heavy imports are deferred so
# importing this module never forces torch/cv2/google-cloud at import time.

_ANCHOR_FRAMES_INDEX_NAME = "frames_index.json"


def _anchor_local_image_path(cache_dir, pool: str, frame_name: str):
    from pathlib import Path as _Path

    return _Path(cache_dir) / pool / frame_name


def cache_anchor_pools_locally(cache_dir, pools=None, force_redownload: bool = False):
    """Discover & cache the 6 anchor pool frames to ``cache_dir`` once.

    Returns a dict ``{pool_name: [{"local_path": str, "frame_name": str}, ...]}``.
    Subsequent calls reuse ``cache_dir`` without re-listing GCS or re-downloading
    bytes. The on-disk index file is at ``cache_dir/frames_index.json``.
    """
    import json as _json
    from pathlib import Path as _Path

    cache_dir = _Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / _ANCHOR_FRAMES_INDEX_NAME

    if pools is None:
        pools = POOLS

    if index_path.exists() and not force_redownload:
        try:
            idx = _json.loads(index_path.read_text())
            # Sanity-check that listed files actually still exist on disk.
            ok = True
            for pool, recs in idx.items():
                for r in recs:
                    if not _Path(r["local_path"]).exists():
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return idx
        except Exception:
            pass  # fall through and rebuild

    from google.cloud import storage  # heavy import deferred

    client = storage.Client()
    out: dict[str, list[dict]] = {}
    for pool in pools:
        pool_dir = cache_dir / pool
        pool_dir.mkdir(parents=True, exist_ok=True)
        recs = discover_pool_records(client, pool)
        items: list[dict] = []
        for rec in recs:
            local_path = _anchor_local_image_path(cache_dir, pool, rec.frame_name)
            if not local_path.exists() or force_redownload:
                blob = client.bucket(rec.bucket).blob(rec.blob_path)
                blob.download_to_filename(str(local_path))
            items.append({
                "local_path": str(local_path),
                "frame_name": rec.frame_name,
                "blob_path": rec.blob_path,
                "bucket": rec.bucket,
            })
        out[pool] = items

    index_path.write_text(_json.dumps(out, indent=2))
    return out


class _LocalAnchorDataset:
    """Tiny torch dataset reading cached anchor PNG/JPG files from disk.

    Mirrors ``arena.model_arena.GCSFrameDataset`` preprocessing exactly
    (CLIP normalize, INTER_LINEAR 224×224, RGB) to keep parity with both
    training preprocessing and the CLI re-scorer.
    """

    def __init__(self, items: list[dict], resolution: int = 224):
        import cv2  # noqa: F401  — verify available
        import torchvision.transforms as T

        self.items = items
        self.resolution = resolution
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import cv2
        import torch

        item = self.items[idx]
        img_bgr = cv2.imread(item["local_path"], cv2.IMREAD_COLOR)
        if img_bgr is None:
            return torch.zeros(3, self.resolution, self.resolution), idx
        img_bgr = cv2.resize(
            img_bgr, (self.resolution, self.resolution),
            interpolation=cv2.INTER_LINEAR,
        )
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(img_rgb), idx


def compute_anchor_metrics(
    model,
    device,
    anchor_cache_dir,
    batch_size: int = 16,
    num_workers: int = 0,
):
    """Run inference on the 6 anchor pools using the in-memory ``model``.

    Args:
        model: an Effort detector with a ``__call__({"image": batch}, inference=True)``
            signature whose output dict has ``"prob"``. Should already be in
            eval mode; this function does NOT toggle ``.train()``.
        device: torch device to move batches onto.
        anchor_cache_dir: local directory where pool frames are cached. Created
            on first call (downloads from GCS); reused thereafter.
        batch_size, num_workers: dataloader settings. ``num_workers=0`` keeps the
            call thread-safe inside training validation hooks.

    Returns:
        dict with:
          - ``per_pool_mean``: {pool: float}
          - ``per_pool_frac_gt_0_9``: {pool: float}
          - ``per_pool_n``: {pool: int}
          - ``anchor_mean``: float — mean prob_fake on the anchor pool
          - ``anchor_frac_gt_0_9``: float
          - ``max_correct_real_mean``: float — worst mean across "correct real" pools
          - ``spread_mean``: float — mean prob_fake across ALL frames
          - ``composite``: float — ``(1 - anchor_mean) - max(0, max_correct_real_mean - 0.02)``
          - ``n_frames_total``: int
    """
    import numpy as _np
    import torch

    cache_index = cache_anchor_pools_locally(anchor_cache_dir)

    # Flatten while keeping pool ownership for aggregation.
    all_items: list[dict] = []
    pool_indices: dict[str, list[int]] = {p: [] for p in POOLS}
    for pool in POOLS:
        for it in cache_index.get(pool, []):
            pool_indices[pool].append(len(all_items))
            all_items.append(it)

    if not all_items:
        return {
            "per_pool_mean": {},
            "per_pool_frac_gt_0_9": {},
            "per_pool_n": {p: 0 for p in POOLS},
            "anchor_mean": float("nan"),
            "anchor_frac_gt_0_9": float("nan"),
            "max_correct_real_mean": float("nan"),
            "spread_mean": float("nan"),
            "composite": float("nan"),
            "n_frames_total": 0,
        }

    dataset = _LocalAnchorDataset(all_items)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(str(device).startswith("cuda")),
    )

    probs = _np.zeros(len(all_items), dtype=_np.float32)
    with torch.inference_mode():
        for images, indices in loader:
            images = images.to(device, non_blocking=True)
            outputs = model({"image": images}, inference=True)
            batch_probs = outputs["prob"].detach().cpu().numpy().reshape(-1)
            for i, gi in enumerate(indices.numpy()):
                probs[int(gi)] = float(batch_probs[i])

    per_pool_mean: dict[str, float] = {}
    per_pool_frac_gt: dict[str, float] = {}
    per_pool_n: dict[str, int] = {}
    for pool in POOLS:
        idxs = pool_indices[pool]
        if not idxs:
            per_pool_mean[pool] = float("nan")
            per_pool_frac_gt[pool] = float("nan")
            per_pool_n[pool] = 0
            continue
        ps = probs[idxs]
        per_pool_mean[pool] = float(_np.mean(ps))
        per_pool_frac_gt[pool] = float(_np.mean(ps > 0.9))
        per_pool_n[pool] = int(ps.size)

    anchor_mean = per_pool_mean.get(ANCHOR_POOL, float("nan"))
    anchor_frac = per_pool_frac_gt.get(ANCHOR_POOL, float("nan"))

    correct_means = [per_pool_mean[p] for p in CORRECT_REAL_POOLS
                     if not _np.isnan(per_pool_mean.get(p, float("nan")))]
    max_correct = float(max(correct_means)) if correct_means else float("nan")

    spread_mean = float(_np.mean(probs)) if probs.size else float("nan")

    if _np.isnan(anchor_mean) or _np.isnan(max_correct):
        composite = float("nan")
    else:
        composite = float((1.0 - anchor_mean) - max(0.0, max_correct - 0.02))

    return {
        "per_pool_mean": per_pool_mean,
        "per_pool_frac_gt_0_9": per_pool_frac_gt,
        "per_pool_n": per_pool_n,
        "anchor_mean": float(anchor_mean),
        "anchor_frac_gt_0_9": float(anchor_frac),
        "max_correct_real_mean": float(max_correct),
        "spread_mean": spread_mean,
        "composite": float(composite),
        "n_frames_total": int(probs.size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="GCS URL to the checkpoint .pth file")
    ap.add_argument("--label", required=True,
                    help="Run label for output filenames (e.g. rlp7_02)")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    import torch
    from google.cloud import storage

    out_json = _HERE.parent / f"pool_rescore_{args.label}.summary.json"
    out_csv = _HERE.parent / f"pool_rescore_{args.label}.per_frame.csv"

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
    print(f"[info] label: {args.label}", flush=True)
    print(f"[info] device: {device_str}", flush=True)

    print(f"[info] downloading checkpoint: {args.checkpoint}", flush=True)
    t0 = time.time()
    local_ckpt = _download_checkpoint(args.checkpoint)
    print(f"[info] checkpoint local: {local_ckpt} ({time.time()-t0:.1f}s)",
          flush=True)

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

    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool", "blob_path", "frame_name", "prob_fake", "status"])
        for i, rec in enumerate(all_records):
            w.writerow([rec.video_id, rec.blob_path, rec.frame_name,
                        f"{probs[i]:.6f}", statuses[i]])
    print(f"[info] wrote per-frame CSV: {out_csv}", flush=True)

    summary = {
        "run_generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": args.label,
        "checkpoint": args.checkpoint,
        "n_frames_total": len(all_records),
        "device": device_str,
        "baseline_source": "rlp6_04_postfix_rescore_2026-04-24.summary.json",
        "pools": [],
    }
    print("\n" + "=" * 104)
    print(f"{'pool':<50} {'N':>3} {'rlp6_04':>8} {'this':>8} {'delta':>8} "
          f"{'min':>6} {'max':>6}")
    print("-" * 104)
    for p in POOLS:
        idxs = [i for i, rec in enumerate(all_records) if rec.video_id == p]
        ps = [float(probs[i]) for i in idxs if statuses[i] == "ok"]
        n_ok = len(ps)
        mean = statistics.mean(ps) if ps else None
        std = statistics.stdev(ps) if len(ps) > 1 else 0.0
        mn = min(ps) if ps else None
        mx = max(ps) if ps else None
        frac_gt_09 = sum(1 for x in ps if x > 0.9) / len(ps) if ps else None
        frac_lt_01 = sum(1 for x in ps if x < 0.1) / len(ps) if ps else None

        baseline = RLP6_04_BASELINE[p]
        delta = (mean - baseline) if mean is not None else None
        summary["pools"].append({
            "pool": p,
            "n_frames": n_ok,
            "rlp6_04_baseline_mean": baseline,
            "this_run_mean": mean,
            "delta_vs_rlp6_04": delta,
            "std": std,
            "min": mn,
            "max": mx,
            "frac_gt_0_9": frac_gt_09,
            "frac_lt_0_1": frac_lt_01,
        })
        print(f"{p:<50} {n_ok:>3} {baseline:>8.3f} "
              f"{(mean if mean is not None else float('nan')):>8.3f} "
              f"{(delta if delta is not None else float('nan')):>+8.3f} "
              f"{(mn if mn is not None else float('nan')):>6.3f} "
              f"{(mx if mx is not None else float('nan')):>6.3f}")

    anchor_entry = next(e for e in summary["pools"] if e["pool"] == ANCHOR_POOL)
    anchor_delta = anchor_entry.get("delta_vs_rlp6_04")
    # Three-pillar readout:
    # pillar-A = real pools stay low (false-positive regression guard)
    # pillar-B = anchor pool (false-flag webcam) drops
    # pillar-C = still-flagged fake-like pools (none here, all 6 are real)
    real_correct_pools = [
        "dor-real-laptop-correct-no-virtual-bg-whiteish",
        "dor-real-laptop-correct-no-virtual-bg-yellowish",
        "roee-real-windows-laptop-correct",
    ]
    real_correct_means = [e["this_run_mean"] for e in summary["pools"]
                          if e["pool"] in real_correct_pools
                          and e["this_run_mean"] is not None]
    max_real_correct = max(real_correct_means) if real_correct_means else None
    fpr_regression = (max_real_correct is not None
                      and max_real_correct > 0.10)  # >10% mean on "correct" pools is bad

    if anchor_delta is None:
        verdict = "no_data"
    elif anchor_delta <= -0.50:
        verdict = "strong_close__shortcut_likely_broken"
    elif anchor_delta <= -0.20:
        verdict = "moderate_close__shortcut_partially_addressed"
    elif anchor_delta <= -0.05:
        verdict = "marginal_close__small_improvement"
    elif anchor_delta < 0.05:
        verdict = "no_meaningful_change"
    else:
        verdict = "regression__anchor_false_flag_got_worse"
    if fpr_regression:
        verdict += "__BUT_REAL_POOL_REGRESSION"

    summary["decision_tree"] = {
        "anchor_pool": ANCHOR_POOL,
        "anchor_delta_vs_rlp6_04": anchor_delta,
        "max_real_correct_mean": max_real_correct,
        "fpr_regression_on_correct_pools": fpr_regression,
        "verdict": verdict,
    }

    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[info] wrote summary JSON: {out_json}")
    print(f"[verdict] anchor_delta={anchor_delta!r} "
          f"max_real_correct={max_real_correct!r} -> {verdict}")


if __name__ == "__main__":
    main()
