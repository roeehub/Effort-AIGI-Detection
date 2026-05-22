"""Face-pool re-scoring of all 9 promotion-contract suites for Slot A v2 step3500.

Pipeline:
1. Reuse the 2026-05-20 frames_report.csv (one row per (video,frame) with
   GCS frame_path, label, video_id, group_key, family_key, method). These
   are the EXACT frames the CLS-pool baseline scored — we only swap the
   pooling op feeding the classifier head.
2. For each of the 9 suites, stream frames from GCS through a face-pool
   monkey-patched Effort detector (centered 7x7 of 14x14 patch grid -> 49
   patches; mean -> ln_post -> visual.proj; classifier head unchanged).
3. Aggregate per video_id (mean of per-frame face-pool probs) and emit a
   <suite>_<checkpoint_key>.lower()_videos_report.csv matching the schema
   that arena/score_teams_promotion_contract.py expects.
4. Discard intermediate frame_report CSVs between suites; nothing else is
   cached locally (frames are streamed in-memory).

Usage:
    python analysis/face_pool_scorecard_2026-05-22/score_face_pool_suites.py \
        --report-root analysis/face_pool_scorecard_2026-05-22/reports \
        --ckpt analysis/manual_canary_2026-05-20/ckpts/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth \
        --checkpoint-key SLOT_A_ANCHOR_AWARE_STEP3500 \
        --suites-root analysis/face_pool_scorecard_2026-05-22/_tmp \
        --log-dir analysis/face_pool_scorecard_2026-05-22/_logs
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as T
from google.cloud import storage

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Reuse existing pieces
from batch_inference_gcs import CLIP_MEAN, CLIP_STD, load_model  # noqa: E402

# Reuse face-pool monkey-patch from the canary script (same module).
sys.path.insert(0, str(REPO_ROOT / "analysis/face_pool_canary_2026-05-22"))
from score_canary_face_pool import FacePoolMonkeyPatch, L11_LAYER  # noqa: E402


DETECTOR_CFG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CFG = REPO_ROOT / "config/train_config.yaml"

# The 9 suites and the local manifest (frames_report.csv) filenames.
SUITES: Tuple[str, ...] = (
    "teams_real_all_dev",
    "teams_real_poor_quality_dev",
    "teams_real_lighting_extreme_dev",
    "teams_fake_all_dev",
    "visomaster_enhanced_macro_dev",
    "deeplive_enhanced_dev",
    "teams_real_all_lockbox",
    "teams_fake_all_lockbox",
    "teams_real_dor_dev",
)


logger = logging.getLogger("face_pool_scorer")


def _parse_gs_uri(uri: str) -> Tuple[str, str]:
    s = uri.removeprefix("gs://")
    bucket, _, blob = s.partition("/")
    return bucket, blob


class FrameURIDataset(data.Dataset):
    """Load (224x224 RGB, CLIP-normalized) face-cropped images from GCS URIs.

    Each row carries the full row metadata so we can reconstruct the
    videos_report from per-frame probabilities.
    """

    def __init__(self, rows: List[Dict[str, str]], resolution: int = 224):
        self.rows = rows
        self.resolution = resolution
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
        self._client = None
        self._buckets: Dict[str, object] = {}

    def _get_bucket(self, bucket_name: str):
        if bucket_name not in self._buckets:
            if self._client is None:
                self._client = storage.Client()
            self._buckets[bucket_name] = self._client.bucket(bucket_name)
        return self._buckets[bucket_name]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.rows[idx]
        frame_path = row["frame_path"]
        bucket_name, blob_path = _parse_gs_uri(frame_path)
        bucket = self._get_bucket(bucket_name)
        blob = bucket.blob(blob_path)
        try:
            img_bytes = blob.download_as_bytes()
        except Exception as e:
            logger.warning("Failed to download %s: %s", frame_path, e)
            return torch.zeros(3, self.resolution, self.resolution), idx
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            logger.warning("Failed to decode %s", frame_path)
            return torch.zeros(3, self.resolution, self.resolution), idx
        img_bgr = cv2.resize(
            img_bgr,
            (self.resolution, self.resolution),
            interpolation=cv2.INTER_LINEAR,
        )
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(img_rgb), idx


@torch.no_grad()
def score_frames_face_pool(
    model: torch.nn.Module,
    rows: List[Dict[str, str]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    """Return per-frame face-pool fake probabilities aligned to `rows`."""
    dataset = FrameURIDataset(rows)
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    probs = np.zeros(len(rows), dtype=np.float64)
    failed_mask = np.zeros(len(rows), dtype=bool)
    model.eval()
    t0 = time.time()
    n_done = 0
    with FacePoolMonkeyPatch(model, layer=L11_LAYER):
        for batch_idx, (images, indices) in enumerate(loader):
            images_dev = images.to(device, non_blocking=True)
            out = model({"image": images_dev}, inference=True)
            if isinstance(out, dict):
                if "prob" in out:
                    p = out["prob"].detach().cpu().numpy().reshape(-1)
                else:
                    logits = None
                    for k in ("cls", "raw_logits", "logits", "classifier_logits", "pred_logits"):
                        if k in out:
                            logits = out[k]
                            break
                    if logits is None:
                        raise RuntimeError("could not extract logits from model output")
                    if logits.dim() == 3:
                        logits = logits.mean(dim=1)
                    p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            else:
                p = torch.softmax(out, dim=-1)[:, 1].detach().cpu().numpy()
            zero_mask = (images.flatten(1).abs().sum(dim=1).cpu().numpy() == 0.0)
            idx_arr = indices.numpy()
            probs[idx_arr] = p
            failed_mask[idx_arr] = zero_mask
            n_done += len(idx_arr)
            if (batch_idx + 1) % 10 == 0:
                fps = n_done / max(time.time() - t0, 1e-3)
                logger.info(
                    "    batch %d/%d  done=%d/%d  %.1f fps  elapsed=%.1fs",
                    batch_idx + 1,
                    len(loader),
                    n_done,
                    len(rows),
                    fps,
                    time.time() - t0,
                )
    logger.info(
        "    scored %d frames in %.1fs (%.1f fps), %d failed-decode",
        len(rows),
        time.time() - t0,
        len(rows) / max(time.time() - t0, 1e-3),
        int(failed_mask.sum()),
    )
    return probs


def aggregate_to_videos(
    frame_rows: List[Dict[str, str]],
    frame_probs: np.ndarray,
    threshold: float = 0.5,
) -> List[Dict[str, object]]:
    """Average per-frame fake-prob into per-video prob; preserve metadata.

    Returns rows ordered by first-occurrence of each video_id so the
    output CSV matches the convention of the baseline videos_report.csv.
    """
    by_video: Dict[str, Dict[str, object]] = {}
    order: List[str] = []
    for row, prob in zip(frame_rows, frame_probs):
        vid = row["video_id"]
        if vid not in by_video:
            by_video[vid] = {
                "method": row.get("method", ""),
                "label": int(row.get("label", 0)),
                "video_id": vid,
                "probs": [],
                "group_key": row.get("group_key", ""),
                "family_key": row.get("family_key", ""),
            }
            order.append(vid)
        by_video[vid]["probs"].append(float(prob))
    out_rows: List[Dict[str, object]] = []
    for vid in order:
        v = by_video[vid]
        avg = float(np.mean(v["probs"])) if v["probs"] else 0.0
        prediction = int(avg >= threshold)
        is_correct = int(prediction == int(v["label"]))
        out_rows.append({
            "method": v["method"],
            "label": int(v["label"]),
            "video_id": vid,
            "avg_video_prob": f"{avg:.8f}",
            "prediction": prediction,
            "is_correct": is_correct,
            "group_key": v["group_key"],
            "family_key": v["family_key"],
        })
    return out_rows


def write_videos_report(rows: List[Dict[str, object]], path: Path) -> None:
    fieldnames = [
        "method",
        "label",
        "video_id",
        "avg_video_prob",
        "prediction",
        "is_correct",
        "group_key",
        "family_key",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("    wrote %s (%d rows)", path, len(rows))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-root", required=True, help="Local output dir for videos_report CSVs")
    ap.add_argument("--ckpt", required=True, help="Local Slot A v2 ckpt path")
    ap.add_argument(
        "--checkpoint-key",
        default="SLOT_A_ANCHOR_AWARE_STEP3500",
        help="Key used in output filename (will be lowercased)",
    )
    ap.add_argument(
        "--suites-root",
        required=True,
        help="Local dir containing <suite>_frames.csv (the 9 manifests pre-downloaded)",
    )
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    ap.add_argument(
        "--start-suite",
        default=None,
        help="If set, skip suites alphabetically (by SUITES order) until this one. "
        "Used for resume-after-crash.",
    )
    args = ap.parse_args()

    # logging
    log_dir = Path(args.log_dir) if args.log_dir else None
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "score_face_pool_suites.log", mode="a"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=handlers,
    )

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("device=%s", device)

    # Load model once
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        logger.error("ckpt not found: %s", ckpt_path)
        return 1
    logger.info("loading model from %s", ckpt_path)
    t0 = time.time()
    model = load_model(str(ckpt_path), str(DETECTOR_CFG), str(TRAIN_CFG), device)
    logger.info("model loaded in %.1fs", time.time() - t0)

    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    suites_root = Path(args.suites_root)

    overall_t0 = time.time()
    suite_summary: List[Dict[str, object]] = []
    started = args.start_suite is None
    for suite in SUITES:
        if not started:
            if suite == args.start_suite:
                started = True
            else:
                logger.info("[skip] %s (before start-suite=%s)", suite, args.start_suite)
                continue
        manifest_path = suites_root / f"{suite}_frames.csv"
        if not manifest_path.exists():
            logger.error("missing manifest: %s — ABORT", manifest_path)
            return 2
        df = pd.read_csv(manifest_path)
        n_frames = len(df)
        n_videos = df["video_id"].nunique() if "video_id" in df.columns else None
        logger.info("=" * 70)
        logger.info(
            "[suite] %s  frames=%d  videos=%s",
            suite, n_frames, n_videos,
        )
        rows = df.to_dict(orient="records")
        t_suite = time.time()
        probs = score_frames_face_pool(
            model, rows, device, args.batch_size, args.num_workers,
        )
        video_rows = aggregate_to_videos(rows, probs)
        out_path = report_root / f"{suite}_{args.checkpoint_key.lower()}_videos_report.csv"
        write_videos_report(video_rows, out_path)
        elapsed = time.time() - t_suite
        suite_summary.append({
            "suite": suite,
            "frames": int(n_frames),
            "videos": int(n_videos) if n_videos is not None else -1,
            "wall_seconds": round(elapsed, 1),
            "videos_report": str(out_path),
        })
        # Free frame manifest from memory between suites
        del df, rows, probs, video_rows
        logger.info("[suite] %s done in %.1fs", suite, elapsed)

    total_wall = time.time() - overall_t0
    logger.info("ALL SUITES DONE in %.1fs", total_wall)

    # Per-suite summary JSON next to the videos_reports
    summary_json = report_root / f"_face_pool_scoring_summary_{args.checkpoint_key.lower()}.json"
    with open(summary_json, "w") as f:
        json.dump(
            {
                "checkpoint_key": args.checkpoint_key,
                "ckpt_path": str(ckpt_path),
                "total_wall_seconds": round(total_wall, 1),
                "suites": suite_summary,
            },
            f,
            indent=2,
        )
    logger.info("wrote %s", summary_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
