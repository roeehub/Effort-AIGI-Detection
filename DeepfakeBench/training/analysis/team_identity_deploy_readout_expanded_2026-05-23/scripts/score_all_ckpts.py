"""Score the master inventory on all 5 ckpts in a single pass.

For each ckpt: load once, score the frames it needs, save per-frame CSV.

Strategy:
- We score all frames on Slot A v2 (CLS) and Slot A v2 (face-pool) — these are
  the new ckpts under evaluation.
- For P8A, E2B, T5C we score ONLY frames where `needs_fresh_score=True` in
  master_inventory.csv (i.e., not already cached in grouped_manifest_v2). The
  cached scores will be merged in by the analyzer downstream.
- Mac-Roee cohort is skipped entirely for fresh scoring (Slot A v2 + face-pool
  also skipped) — Mac is info-only via cached scores from grouped_manifest_v2.

Output per ckpt: outputs/<CKPT_KEY>_pool_scores.per_frame.csv with columns
    base_identity, frame_path, prob_fake, status

The analyzer (analyze_team_deploy.py) merges these with cached-score columns
from master_inventory.csv to assemble the per-(ckpt, frame) full table.

Usage:
    python analysis/team_identity_deploy_readout_expanded_2026-05-23/scripts/score_all_ckpts.py
    python analysis/team_identity_deploy_readout_expanded_2026-05-23/scripts/score_all_ckpts.py --only SLOT_A_ANCHOR_AWARE_STEP3500
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analysis/face_pool_canary_2026-05-22"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from arena.model_arena import FrameRecord, GCSFrameDataset, logger  # noqa: E402
from batch_inference_gcs import load_model  # noqa: E402
from local_frame_resolver import resolve_local  # noqa: E402


class HybridFrameDataset(GCSFrameDataset):
    """Drop-in replacement for GCSFrameDataset that also resolves gs://local/... paths.

    For records with bucket == 'local', delegates to local_frame_resolver.resolve_local()
    to find the file on disk; otherwise behaves identically to GCSFrameDataset.
    """

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_bgr = None
        uri = f"gs://{rec.bucket}/{rec.blob_path}"

        # Local-bucket short-circuit
        if rec.bucket == "local":
            local_path = resolve_local(uri)
            if local_path is not None:
                img_bgr = cv2.imread(local_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                logger.warning("Failed local resolve: %s", uri)
        else:
            client = self._get_client()
            bucket = client.bucket(rec.bucket)
            blob = bucket.blob(rec.blob_path)
            try:
                data = blob.download_as_bytes()
                arr = np.frombuffer(data, dtype=np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                img_bgr = None
            # GCS-miss fallback: try local-resolver (covers manifest paths that
            # use cohort-name aliases instead of the real session-* directory layout,
            # e.g. live-fakes-teams-prod/real/tester_roee_real_2026-03-06/ → local copy)
            if img_bgr is None:
                local_path = resolve_local(uri)
                if local_path is not None:
                    img_bgr = cv2.imread(local_path, cv2.IMREAD_COLOR)

        if img_bgr is None:
            logger.warning("Failed to decode: gs://%s/%s", rec.bucket, rec.blob_path)
            return torch.zeros(3, self.resolution, self.resolution), idx

        img_bgr = cv2.resize(img_bgr, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb)
        return tensor, idx

DETECTOR_CFG = REPO_ROOT / "config/detector/effort.yaml"
TRAIN_CFG = REPO_ROOT / "config/train_config.yaml"

LOCAL_CKPT_DIR = REPO_ROOT / "analysis/manual_canary_2026-05-20/ckpts"
PRIOR_E2B_CKPT = REPO_ROOT / "analysis/team_identity_deploy_readout_2026-05-23/_ckpts/E2B_TOP_N_STEP3200.pth"

CKPTS = {
    # ckpt_key: (path, score_only_fresh_flag)
    "P8A_REFERENCE_STEP5000": (
        LOCAL_CKPT_DIR / "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth",
        True,  # only score frames where needs_fresh_score=True
    ),
    "E2B_TOP_N_STEP3200": (PRIOR_E2B_CKPT, True),
    "T5C_PERIODIC_STEP3500": (
        LOCAL_CKPT_DIR / "periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth",
        True,
    ),
    "SLOT_A_ANCHOR_AWARE_STEP3500": (
        LOCAL_CKPT_DIR / "periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth",
        False,  # score all non-Mac frames
    ),
    "SLOT_A_ANCHOR_AWARE_STEP3500_FACE_POOL": (
        LOCAL_CKPT_DIR / "periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth",
        False,  # score all non-Mac frames + apply face-pool monkey-patch
    ),
}

OUTPUT_DIR = REPO_ROOT / "analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs"
INV_CSV = OUTPUT_DIR / "master_inventory.csv"

BATCH_SIZE = 32


def build_records_from_inventory(rows: pd.DataFrame) -> List[FrameRecord]:
    recs: List[FrameRecord] = []
    for _, r in rows.iterrows():
        # frame_path is a full gs:// URI
        fp = r["frame_path"]
        if fp.startswith("gs://"):
            no_scheme = fp[5:]
            bkt, blob = no_scheme.split("/", 1)
        else:
            bkt = r["bucket"]
            blob = fp
        recs.append(FrameRecord(
            bucket=bkt,
            blob_path=blob,
            label=int(r["label"]),
            method="real" if int(r["label"]) == 0 else "fake",
            video_id=r["base_identity"],
            frame_name=Path(blob).name,
            strategy=r["base_identity"],
        ))
    return recs


def score_one_ckpt(ckpt_key: str, ckpt_path: Path, score_only_fresh: bool,
                   inv: pd.DataFrame, device: torch.device,
                   use_face_pool: bool = False) -> None:
    out_csv = OUTPUT_DIR / f"{ckpt_key}_scores.per_frame.csv"
    if out_csv.exists():
        print(f"[skip] {ckpt_key} -> {out_csv} exists", flush=True)
        return
    if not ckpt_path.exists():
        print(f"[FAIL] {ckpt_key}: ckpt not found at {ckpt_path}", flush=True)
        return

    # Filter inventory rows
    if score_only_fresh:
        # P8A/E2B/T5C only need fresh scoring on rows where needs_fresh_score=True
        rows = inv[inv.needs_fresh_score].copy()
    else:
        # Slot A v2 (CLS + face-pool) — score all non-Mac frames (Mac-Roee is info-only)
        rows = inv[inv.human != "Roee_Mac"].copy()
    rows = rows.reset_index(drop=True)

    print(f"\n=== {ckpt_key} ({len(rows)} frames) ===", flush=True)
    if len(rows) == 0:
        # Write empty CSV with header
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["base_identity", "frame_path", "prob_fake", "status"])
        return

    recs = build_records_from_inventory(rows)
    dataset = HybridFrameDataset(recs)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    t0 = time.time()
    model = load_model(str(ckpt_path), str(DETECTOR_CFG), str(TRAIN_CFG), device)
    print(f"[load] {time.time() - t0:.1f}s", flush=True)
    model.eval()

    probs = np.zeros(len(recs), dtype=np.float32)
    statuses = ["ok"] * len(recs)
    t_inf = time.time()

    if use_face_pool:
        from score_canary_face_pool import FacePoolMonkeyPatch, L11_LAYER
        cm = FacePoolMonkeyPatch(model, layer=L11_LAYER)
    else:
        from contextlib import nullcontext
        cm = nullcontext()

    with torch.inference_mode():
        with cm:
            for bi, (images, indices) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                out = model({"image": images}, inference=True)
                if isinstance(out, dict) and "prob" in out:
                    p = out["prob"].detach().cpu().numpy().reshape(-1)
                else:
                    if isinstance(out, dict):
                        logits = out.get("cls") or out.get("raw_logits") or out.get("logits")
                    else:
                        logits = out
                    if logits is None:
                        raise RuntimeError(
                            f"could not extract logits from {list(out.keys()) if isinstance(out, dict) else type(out)}"
                        )
                    if logits.dim() == 3:
                        logits = logits.mean(dim=1)
                    p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
                for i, gi in enumerate(indices.numpy()):
                    probs[int(gi)] = float(p[i])
                    if bool(images[i].abs().sum().item() == 0.0):
                        statuses[int(gi)] = "failed_decode"
                if (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
                    elapsed = time.time() - t_inf
                    fps = (bi + 1) * BATCH_SIZE / elapsed
                    eta_s = elapsed / (bi + 1) * (len(loader) - bi - 1)
                    print(
                        f"  batch {bi+1}/{len(loader)} ({elapsed:.0f}s, {fps:.1f}fps, ETA {eta_s/60:.1f}min)",
                        flush=True,
                    )
    print(f"[inf] {time.time() - t_inf:.0f}s", flush=True)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["base_identity", "frame_path", "prob_fake", "status"])
        for i, rec in enumerate(recs):
            full_path = f"gs://{rec.bucket}/{rec.blob_path}"
            w.writerow([rec.video_id, full_path, f"{probs[i]:.6f}", statuses[i]])
    print(f"[ok] wrote {out_csv}")
    del model
    if device.type == "mps":
        torch.mps.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated ckpt keys to score (default: all 5)")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INV_CSV.exists():
        print(f"[FAIL] inventory CSV not found: {INV_CSV}")
        return 1

    inv = pd.read_csv(INV_CSV, low_memory=False)
    print(f"[main] loaded {len(inv)} frames from {INV_CSV}", flush=True)
    print(f"[main] needs_fresh_score=True: {inv.needs_fresh_score.sum()}", flush=True)
    print(f"[main] non-Mac frames: {(inv.human != 'Roee_Mac').sum()}", flush=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[main] device={device}", flush=True)

    keys_to_run = list(CKPTS.keys())
    if args.only:
        keys_to_run = [k.strip() for k in args.only.split(",")]

    for ck in keys_to_run:
        if ck not in CKPTS:
            print(f"[warn] unknown ckpt key: {ck} — skipping")
            continue
        path, score_only_fresh = CKPTS[ck]
        use_face_pool = (ck == "SLOT_A_ANCHOR_AWARE_STEP3500_FACE_POOL")
        score_one_ckpt(ck, path, score_only_fresh, inv, device, use_face_pool=use_face_pool)

    print("\n[main] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
