#!/usr/bin/env python3
"""Score the 5 selected P2 ckpts on the 800-frame canary parquet.

Outputs per-ckpt per-frame CSV at
``analysis/p2_eval_2026-05-08/d1_d4_cpu/scores/<ckpt_id>.csv``.

Reuses the model-loading logic of ``test_checkpoint_local.py`` (CPU-safe).
The canary parquet at ``arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet``
already carries P8A reference scores in ``p8a_reference_score``.

Frame caching: frames are downloaded once into ``./_frame_cache/`` keyed by URI
and reused across ckpts.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))

from detectors import DETECTOR  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO,
                    datefmt="%H:%M:%S")
log = logging.getLogger("d1_d4_inference")

THIS = Path(__file__).resolve().parent
CKPT_DIR = THIS / "ckpts"
SCORES_DIR = THIS / "scores"
FRAME_CACHE = THIS / "_frame_cache"
SCORES_DIR.mkdir(parents=True, exist_ok=True)
FRAME_CACHE.mkdir(parents=True, exist_ok=True)

CANARY_PARQUET = REPO / "arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet"

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
_TRANSFORM = T.Compose([T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])

CKPTS = [
    ("slotA_top_n_step500", CKPT_DIR / "slotA_top_n_step500.pth"),
    ("slotB_top_n_step500", CKPT_DIR / "slotB_top_n_step500.pth"),
    ("slotC_top_n_step7000", CKPT_DIR / "slotC_top_n_step7000.pth"),
    ("slotD_top_n_step6000", CKPT_DIR / "slotD_top_n_step6000.pth"),
    ("slotD_top_n_step19000", CKPT_DIR / "slotD_top_n_step19000.pth"),
]


def load_detector(weights_path: Path, device: torch.device) -> torch.nn.Module:
    """Load an EffortDetector ckpt; mirrors test_checkpoint_local.load_detector_cpu."""
    import yaml
    ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        model_config = ckpt.get("model_config", {})
    else:
        state_dict = ckpt
        model_config = {}
    base_cfg_path = REPO / "config" / "detector" / "effort.yaml"
    if base_cfg_path.exists():
        with open(base_cfg_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {"model_name": "effort"}
    if model_config:
        for k, v in model_config.items():
            if k == "current_arcface_s":
                continue
            cfg[k] = v
    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)
    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])
    state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning("missing keys (%d): %s ...", len(missing), missing[:3])
    if unexpected:
        log.warning("unexpected keys (%d): %s ...", len(unexpected), unexpected[:3])
    model.eval()
    return model


def gcs_download(uri: str, dst: Path) -> bool:
    if dst.exists():
        return True
    cp = subprocess.run(["gcloud", "storage", "cp", uri, str(dst)],
                        capture_output=True, text=True, timeout=60)
    return cp.returncode == 0 and dst.exists()


def download_all_frames(df: pd.DataFrame, n_workers: int = 16) -> dict[str, Path]:
    items = []
    for fp in df["frame_path"]:
        basename = fp.split("/")[-1]
        items.append((fp, FRAME_CACHE / basename))
    have = sum(1 for _, dst in items if dst.exists())
    log.info("frame cache: %d / %d already present", have, len(items))
    pending = [(uri, dst) for uri, dst in items if not dst.exists()]
    if pending:
        log.info("downloading %d frames with %d workers ...", len(pending), n_workers)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(gcs_download, uri, dst): (uri, dst) for uri, dst in pending}
            done = 0
            for fut in as_completed(futs):
                done += 1
                if done % 50 == 0:
                    log.info("  downloaded %d / %d", done, len(pending))
    out = {uri: dst for uri, dst in items if dst.exists()}
    log.info("frame cache ready: %d / %d", len(out), len(items))
    return out


def preprocess_one(path: Path) -> torch.Tensor | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (224, 224):
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    return _TRANSFORM(rgb)


@torch.inference_mode()
def score_ckpt(model: torch.nn.Module, df: pd.DataFrame, frame_paths: dict[str, Path],
               device: torch.device, batch_size: int = 32) -> pd.DataFrame:
    rows = []
    tensors: list[torch.Tensor] = []
    metas: list[tuple] = []
    for _, row in df.iterrows():
        local_p = frame_paths.get(row["frame_path"])
        if local_p is None:
            rows.append({"frame_idx": row["frame_idx"], "frame_path": row["frame_path"],
                         "prob_fake": float("nan"), "decode_fail": True})
            continue
        t = preprocess_one(local_p)
        if t is None:
            rows.append({"frame_idx": row["frame_idx"], "frame_path": row["frame_path"],
                         "prob_fake": float("nan"), "decode_fail": True})
            continue
        tensors.append(t)
        metas.append((row["frame_idx"], row["frame_path"]))

    log.info("  %d frames preprocessed; running %d batches of %d",
             len(tensors), (len(tensors) + batch_size - 1) // batch_size, batch_size)
    t0 = time.time()
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i:i + batch_size]).to(device)
        preds = model({"image": batch}, inference=True)
        probs = preds["prob"].detach().cpu().float().numpy().reshape(-1)
        for (frame_idx, frame_path), p in zip(metas[i:i + batch_size], probs):
            rows.append({"frame_idx": frame_idx, "frame_path": frame_path,
                         "prob_fake": float(p), "decode_fail": False})
    log.info("  inference done in %.1f sec", time.time() - t0)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of canary frames (debug; 0=all)")
    args = parser.parse_args()
    device = torch.device(args.device)
    log.info("=" * 70)
    log.info("D1-D4 inference: %d ckpts on canary 800-frame parquet (device=%s)", len(CKPTS), device)
    log.info("=" * 70)
    df = pd.read_parquet(CANARY_PARQUET)
    if args.limit:
        df = df.head(args.limit)
    log.info("canary parquet: %d rows; cohorts=%s",
             len(df), sorted(df["cohort"].unique()))
    frame_paths = download_all_frames(df)
    keep_meta = df[["frame_idx", "frame_path", "label", "cohort", "base_identity",
                    "suite", "p8a_reference_score"]].copy()
    keep_meta.to_csv(SCORES_DIR / "_canary_meta.csv", index=False)
    log.info("wrote canary meta -> %s", SCORES_DIR / "_canary_meta.csv")
    for ckpt_id, weights_path in CKPTS:
        if not weights_path.exists():
            log.warning("skipping %s — file not found", weights_path)
            continue
        out_csv = SCORES_DIR / f"{ckpt_id}.csv"
        if out_csv.exists():
            log.info("[%s] skip — output already exists", ckpt_id)
            continue
        log.info("[%s] loading weights ...", ckpt_id)
        t_load = time.time()
        model = load_detector(weights_path, device)
        log.info("[%s] loaded in %.1fs", ckpt_id, time.time() - t_load)
        df_scores = score_ckpt(model, df, frame_paths, device, args.batch_size)
        df_scores.to_csv(out_csv, index=False)
        log.info("[%s] -> %s (n=%d non-nan=%d)",
                 ckpt_id, out_csv, len(df_scores), df_scores["prob_fake"].notna().sum())
        del model
    log.info("DONE.")


if __name__ == "__main__":
    main()
