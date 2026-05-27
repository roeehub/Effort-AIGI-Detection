"""Score Slot 1 head-retrain step250 on the remaining 6 contract suites.

Already scored by `run_step250_eval.py`:
  - teams_real_all_dev (n=3253 videos)
  - teams_real_all_lockbox (n=1361 videos)
  - may6_falseflag (n=92 frames)

Remaining 7 suites needed for the deployment curve (mirrors the T5C deployment
curve harness):
  - teams_real_poor_quality_dev   (n=923 vids,   1303 frames)
  - teams_real_lighting_extreme_dev (n=1401 vids, 1742 frames)
  - teams_fake_all_dev            (n=2409 vids,  3039 frames)
  - visomaster_enhanced_macro_dev (n=550 vids,   550 frames)
  - deeplive_enhanced_dev         (n=545 vids,   545 frames)
  - teams_fake_all_lockbox        (n=253 vids,   425 frames)
  - teams_real_dor_dev            (n=50 vids,    50 frames)

Total: 7,654 frames across 7 suites.

Re-uses `_load_model_with_lora` from run_step250_eval.py for parity.

Outputs:
  outputs/scores_step250_<suite>.csv  (per-frame)
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_DIR))

from arena.model_arena import GCSFrameDataset, FrameRecord  # noqa: E402
from run_step250_eval import (  # noqa: E402
    _load_model_with_lora,
    LORA_CFG_R13,
    STEP250_LOCAL,
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    RESOLUTION,
    MANIFEST_PATH,
    OUT_DIR,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("step250_remaining")


# 7 remaining contract suites: (suite_name, manifest_split, manifest_slice, label)
REMAINING_SUITES = [
    ("teams_real_poor_quality_dev",      "dev",     "teams_real_poor_quality",     0),
    ("teams_real_lighting_extreme_dev",  "dev",     "teams_real_lighting_extreme", 0),
    ("teams_fake_all_dev",               "dev",     "teams_fake_all",              1),
    ("visomaster_enhanced_macro_dev",    "dev",     "visomaster_enhanced_macro",   1),
    ("deeplive_enhanced_dev",            "dev",     "deeplive_enhanced",           1),
    ("teams_fake_all_lockbox",           "lockbox", "teams_fake_all",              1),
    ("teams_real_dor_dev",               "dev",     "dor",                         0),
]


def load_manifest_videos(split: str, slice_name: str) -> list[dict]:
    with open(MANIFEST_PATH) as f:
        m = json.load(f)
    out = []
    for v in m["videos"]:
        if v["split"] != split:
            continue
        if slice_name not in v.get("slices", []):
            continue
        out.append(v)
    return out


def videos_to_records(videos: list[dict], label: int) -> list[FrameRecord]:
    recs: list[FrameRecord] = []
    for v in videos:
        for fp in v["frame_paths"]:
            assert fp.startswith("gs://"), fp
            no_scheme = fp[5:]
            bucket, blob_path = no_scheme.split("/", 1)
            recs.append(FrameRecord(
                bucket=bucket,
                blob_path=blob_path,
                label=label,
                method=v.get("method", "unknown"),
                video_id=v["video_id"],
                frame_name=Path(blob_path).name,
                strategy="",
                extra={"identity_key": v.get("identity_key", "")},
            ))
    return recs


def score_records(model: torch.nn.Module, records: list[FrameRecord],
                  out_csv: Path, label_pop: str) -> Path:
    if out_csv.exists():
        log.info("[%s] CSV exists, skipping: %s", out_csv.name, out_csv)
        return out_csv

    n = len(records)
    log.info("[%s] scoring %d frames ...", out_csv.name, n)

    dataset = GCSFrameDataset(records, resolution=RESOLUTION)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    probs = np.zeros(n, dtype=np.float32)
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
        if (now - last_log) >= 30 or (batch_idx + 1) % 25 == 0:
            done = (batch_idx + 1) * BATCH_SIZE
            elapsed = now - t0
            fps = min(done, n) / max(elapsed, 1e-3)
            log.info("[%s] batch %d ~%d/%d elapsed=%.1fs (%.1f fps)",
                     out_csv.name, batch_idx + 1, min(done, n), n, elapsed, fps)
            last_log = now

    log.info("[%s] inference done in %.1fs", out_csv.name, time.time() - t0)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "population", "video_id", "identity_key", "gcs_uri",
            "frame_name", "label", "prob_fake",
        ])
        for rec, prob in zip(records, probs):
            w.writerow([
                label_pop, rec.video_id, rec.extra.get("identity_key", ""),
                f"gs://{rec.bucket}/{rec.blob_path}", rec.frame_name,
                rec.label, f"{prob:.6f}",
            ])
    log.info("[%s] wrote -> %s", out_csv.name, out_csv)
    return out_csv


def main() -> int:
    log.info("=" * 80)
    log.info("Score step250 on remaining 6 contract suites")
    log.info("device=%s, batch=%d, workers=%d", DEVICE, BATCH_SIZE, NUM_WORKERS)
    log.info("=" * 80)

    if not STEP250_LOCAL.exists():
        log.error("step250 cache missing: %s — run run_step250_eval.py first", STEP250_LOCAL)
        return 1

    log.info("loading step250 model with LoRA wrap ...")
    t0 = time.time()
    model = _load_model_with_lora(str(STEP250_LOCAL), DEVICE, LORA_CFG_R13)
    log.info("model loaded in %.1fs", time.time() - t0)

    suite_dir = OUT_DIR
    suite_dir.mkdir(parents=True, exist_ok=True)

    t_global = time.time()
    for suite_name, split, slice_name, label in REMAINING_SUITES:
        videos = load_manifest_videos(split, slice_name)
        recs = videos_to_records(videos, label=label)
        log.info("[%s] %d videos, %d frames", suite_name, len(videos), len(recs))
        out_csv = suite_dir / f"scores_step250_{suite_name}.csv"
        score_records(model, recs, out_csv, label_pop=suite_name)
    log.info("All suites done in %.1fs", time.time() - t_global)

    return 0


if __name__ == "__main__":
    sys.exit(main())
