#!/usr/bin/env python3
"""PATH_A_LAUNCH_2026-05-07 — paired-feature extraction for Phase 0h + 0j.

Combined forward-pass extraction at the 5,311 unique paired training-pair
indices in `analysis/pair_gap_audit_2026-05-06/outputs/pair_gaps.csv`. Used to
unblock two downstream probes:

  * Phase 0h (FROZEN_PAIR_HEAD_PROBE) — needs frozen final-layer features
    for {P8A, E2B, CLIP_B16_raw} at every paired frame so we can fit a small
    head with vs without pair-rank loss and localise the gap signal
    (head-side → cheaper retrain; encoder-side → full FT).
  * Phase 0j (forward-pass tight-pair audit) — needs binary scores at the
    same paired frames so the same-source pair-gap probe can run on df40 +
    deeplive lanes (currently 2/6 covered → expanding to 4/6).

Output schema (one NPZ per checkpoint):

    features:    float32 (N, D)  — `out["feat"]` for Effort detectors,
                                   raw `visual(x)` post-projection for the
                                   bare CLIP_B16 baseline. D = 512 for both.
    scores:      float32 (N,)    — `out["prob"]` for Effort; sigmoid of
                                   first-coord post-projection for the
                                   raw-CLIP baseline (placeholder, not used
                                   downstream — head-probe will fit its own).
    frame_path:  object  (N,)    — gs:// URI matching pair_gaps.csv.
    label:       int32   (N,)    — 0 = real, 1 = fake (parsed from CSV
                                   columns: real_path → 0, fake_path → 1).
    pair_id:     object  (N,)    — semicolon-joined list of pair_ids that
                                   the frame participates in (a single frame
                                   can be in many cross-product pairs).

Compatible with `analysis/frozen_pair_head_probe_2026-05-06/run_probe.py`
(it loads `features`, `frame_path`, `label`).

Usage (Vertex container):
    python3 analysis/path_a_launch_2026-05-07/extract_paired_features.py \\
        --pair_gaps_csv gs://training-job-outputs/path_a_inputs_2026-05-07/pair_gaps.csv \\
        --output_prefix gs://training-job-outputs/analysis_outputs/frozen_pair_features_2026-05-07/ \\
        --p8a_ckpt gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth \\
        --e2b_ckpt gs://training-job-outputs/best_checkpoints/rmat8lwx/top_n_effort_20260503_step3200_auc0.9863_eer0.0310.pth \\
        --include_clip_raw \\
        --batch_size 64 \\
        --num_workers 8
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
from google.cloud import storage

HERE = Path(__file__).resolve().parent
TRAINING_ROOT = HERE.parent.parent  # .../training/
sys.path.insert(0, str(TRAINING_ROOT))

from detectors import DETECTOR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("path-a-extract")

# =============================================================================
# Constants — preprocessing parity with arena/feature_space extractors.
# =============================================================================
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
RESOLUTION = 224
SEED = 737

DETECTOR_CONFIG: dict = {
    "model_name": "effort",
    "backbone": {
        "name": "vit_b_16_laion_datacomp",
        "variant": "ViT-B-16-DataComp-XL",
        "source": "laion",
        "model_name": "ViT-B-16",
        "pretrained": "datacomp_xl_s13b_b90k",
        "hidden_size": 512,
        "resolution": 224,
        "apply_svd_to_in_proj": True,
    },
    "rank": 736,
    "lambda_reg": 0.01,
    "use_arcface_head": True,
    "arcface_m": 0.15,
    "arcface_s": 12.0,
    "label_smoothing": 0.0,
    "stability_lambda": 0.0,
    "normalize_features_before_head": False,
    "mixup_alpha": 0.0,
    "use_quality_head": False,
    "gcs_assets": {
        "clip_backbone": {
            "gcs_path": "gs://base-checkpoints/effort-aigi/CLIP-ViT-B-16-DataComp.XL-s13B-b90K/",
            "local_path": "./weights/CLIP-ViT-B-16-DataComp.XL-s13B-b90K/",
            "files": ["open_clip_config.json", "open_clip_pytorch_model.bin"],
        }
    },
}


# =============================================================================
# GCS helpers (lifted from feature_space_2026-04-23/extract_features.py).
# =============================================================================
def parse_gs_uri(uri: str) -> Tuple[str, str]:
    assert uri.startswith("gs://"), f"Expected gs:// URI: {uri}"
    bucket, _, blob = uri[5:].partition("/")
    return bucket, blob


def download_to_local(uri: str, suffix: str = "") -> str:
    bucket, blob_name = parse_gs_uri(uri)
    client = storage.Client()
    blob = client.bucket(bucket).blob(blob_name)
    if not blob.exists(client=client):
        raise FileNotFoundError(f"Not found: {uri}")
    local_dir = tempfile.mkdtemp(prefix="path_a_")
    local_path = os.path.join(local_dir, Path(blob_name).name + suffix)
    logger.info("Downloading %s → %s (%.1f MB)", uri, local_path, (blob.size or 0) / 1e6)
    blob.download_to_filename(local_path)
    return local_path


def upload_file(local_path: str, gcs_uri: str) -> None:
    bucket, blob = parse_gs_uri(gcs_uri)
    storage.Client().bucket(bucket).blob(blob).upload_from_filename(local_path)
    logger.info("Uploaded %s → %s", local_path, gcs_uri)


def download_backbone_weights(cfg: dict) -> None:
    assets = cfg["gcs_assets"]["clip_backbone"]
    gcs_path = assets["gcs_path"]
    local_path = Path(assets["local_path"]).resolve()
    local_path.mkdir(parents=True, exist_ok=True)
    if all((local_path / f).exists() for f in assets["files"]):
        logger.info("CLIP backbone already present at %s", local_path)
        return
    client = storage.Client()
    bucket_name, blob_prefix = parse_gs_uri(gcs_path.rstrip("/"))
    for f in assets["files"]:
        dst = local_path / f
        if dst.exists():
            continue
        src = f"{blob_prefix}/{f}" if blob_prefix else f
        logger.info("Downloading %s → %s", f, dst)
        client.bucket(bucket_name).blob(src).download_to_filename(str(dst))
    logger.info("Backbone ready at %s", local_path)


# =============================================================================
# Frame manifest from pair_gaps.csv.
# =============================================================================
def build_frame_manifest(csv_path: str) -> pd.DataFrame:
    """Return a DataFrame with one row per UNIQUE frame URI (real or fake),
    annotated with label (0/1) and the semicolon-joined list of pair_ids the
    frame participates in.
    """
    pg = pd.read_csv(csv_path)
    logger.info("Loaded pair_gaps.csv: %d rows, %d columns", len(pg), pg.shape[1])

    rows: Dict[str, Dict] = {}
    pair_ids_per_path: Dict[str, List[str]] = defaultdict(list)
    for _, r in pg.iterrows():
        rp = r["real_path"]
        fp = r["fake_path"]
        pid = str(r["pair_id"])
        pair_ids_per_path[rp].append(pid)
        pair_ids_per_path[fp].append(pid)
        if rp not in rows:
            rows[rp] = {"frame_path": rp, "label": 0}
        if fp not in rows:
            rows[fp] = {"frame_path": fp, "label": 1}

    out = pd.DataFrame(list(rows.values()))
    out["pair_id"] = out["frame_path"].map(
        lambda p: ";".join(pair_ids_per_path[p])
    )
    n_real = int((out["label"] == 0).sum())
    n_fake = int((out["label"] == 1).sum())
    logger.info("Unique frames: %d total (%d real, %d fake)", len(out), n_real, n_fake)
    return out


# =============================================================================
# Effort model load (parity with arena/model_arena.load_model).
# =============================================================================
def load_effort_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    cfg = dict(DETECTOR_CONFIG)
    download_backbone_weights(cfg)

    logger.info("Building Effort detector...")
    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)

    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        model_config = ckpt.get("model_config", {})
    else:
        state_dict = ckpt
        model_config = {}
    clean = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    missing, unexpected = model.load_state_dict(clean, strict=False)
    logger.info(
        "ckpt load: missing=%d unexpected=%d (first missing: %s)",
        len(missing), len(unexpected), list(missing)[:3]
    )

    if model_config.get("use_arcface_head") and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])
            logger.info("Restored ArcFace s=%.3f", model_config["current_arcface_s"])

    model.eval()
    return model


def load_clip_b16_raw(device: torch.device) -> torch.nn.Module:
    """Load OpenCLIP ViT-B-16 with raw DataComp-XL weights (no fine-tuning).
    Returns the visual tower; output is 512-d post-projection CLS.
    """
    cfg = dict(DETECTOR_CONFIG)
    download_backbone_weights(cfg)
    bin_path = Path(cfg["gcs_assets"]["clip_backbone"]["local_path"]).resolve() / "open_clip_pytorch_model.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"CLIP weights not found at {bin_path}")

    import open_clip
    logger.info("Building OpenCLIP ViT-B-16 (no pretrained), then loading raw DataComp-XL weights...")
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained=None)
    sd = torch.load(str(bin_path), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info("Raw CLIP load: missing=%d unexpected=%d", len(missing), len(unexpected))
    visual = model.visual.to(device).eval()
    return visual


# =============================================================================
# Dataset (GCS-streaming, deterministic indexing).
# =============================================================================
class GCSFrameDataset(data.Dataset):
    def __init__(self, uris: List[str], resolution: int = RESOLUTION):
        self.uris = uris
        self.resolution = resolution
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
        self._client: Optional[storage.Client] = None
        self._bucket_cache: Dict[str, object] = {}

    def _bucket(self, name: str):
        if self._client is None:
            self._client = storage.Client()
        if name not in self._bucket_cache:
            self._bucket_cache[name] = self._client.bucket(name)
        return self._bucket_cache[name]

    def __len__(self) -> int:
        return len(self.uris)

    def __getitem__(self, idx: int):
        uri = self.uris[idx]
        try:
            bucket_name, blob_name = parse_gs_uri(uri)
            blob = self._bucket(bucket_name).blob(blob_name)
            img_bytes = blob.download_as_bytes()
        except Exception as e:
            logger.warning("Download failed %s: %s", uri, e)
            return torch.zeros(3, self.resolution, self.resolution), idx, 0

        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return torch.zeros(3, self.resolution, self.resolution), idx, 0
        img_bgr = cv2.resize(img_bgr, (self.resolution, self.resolution),
                             interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(img_rgb), idx, 1


# =============================================================================
# Forward passes — returns features + scores aligned with input order.
# =============================================================================
def extract_with_effort(
    model: torch.nn.Module,
    uris: List[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (features [N,D], scores [N], ok [N]).
    features = out['feat'], scores = out['prob']  (sigmoid-1 of binary head).
    """
    dataset = GCSFrameDataset(uris)
    loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    feats: List[Optional[np.ndarray]] = [None] * len(uris)
    scores: List[float] = [0.0] * len(uris)
    ok: List[int] = [0] * len(uris)

    t0 = time.time()
    n_done = 0
    for images, indices, valid_flags in loader:
        images = images.to(device, non_blocking=True)
        with torch.inference_mode():
            out = model({"image": images}, inference=True)
            f = out["feat"].detach().cpu().numpy()
            p = out["prob"].detach().cpu().numpy().reshape(-1)
        for j, idx_j in enumerate(indices.numpy()):
            ix = int(idx_j)
            if int(valid_flags[j]) == 1:
                feats[ix] = f[j]
                scores[ix] = float(p[j])
                ok[ix] = 1
        n_done += len(indices)
        if n_done % (batch_size * 20) == 0 or n_done >= len(uris):
            logger.info("  effort batch progress: %d/%d (%.1f fps)",
                        n_done, len(uris),
                        n_done / max(time.time() - t0, 1e-3))

    # Backfill any missing rows with zeros so the array is rectangular.
    D = next((f.shape[0] for f in feats if f is not None), 512)
    for i, f in enumerate(feats):
        if f is None:
            feats[i] = np.zeros(D, dtype=np.float32)
    features = np.stack(feats).astype(np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)
    ok_arr = np.asarray(ok, dtype=np.int32)
    dt = time.time() - t0
    logger.info("  effort done: %d/%d ok in %.1fs (%.1f fps)",
                int(ok_arr.sum()), len(uris), dt, len(uris) / max(dt, 1e-3))
    return features, scores_arr, ok_arr


def extract_with_clip_raw(
    visual: torch.nn.Module,
    uris: List[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (features [N,D], scores [N], ok [N]).
    features = visual(x) post-projection (512-d).
    scores   = placeholder (sigmoid of feature[0]); not used downstream — the
               head probe trains its own head on top of these frozen features.
    """
    dataset = GCSFrameDataset(uris)
    loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    feats: List[Optional[np.ndarray]] = [None] * len(uris)
    scores: List[float] = [0.0] * len(uris)
    ok: List[int] = [0] * len(uris)

    t0 = time.time()
    n_done = 0
    for images, indices, valid_flags in loader:
        images = images.to(device, non_blocking=True)
        with torch.inference_mode():
            f = visual(images)
            if isinstance(f, (tuple, list)):
                f = f[0]
            f_np = f.detach().cpu().numpy()
            p_np = torch.sigmoid(f[:, 0]).detach().cpu().numpy()
        for j, idx_j in enumerate(indices.numpy()):
            ix = int(idx_j)
            if int(valid_flags[j]) == 1:
                feats[ix] = f_np[j]
                scores[ix] = float(p_np[j])
                ok[ix] = 1
        n_done += len(indices)
        if n_done % (batch_size * 20) == 0 or n_done >= len(uris):
            logger.info("  clip_raw batch progress: %d/%d (%.1f fps)",
                        n_done, len(uris),
                        n_done / max(time.time() - t0, 1e-3))

    D = next((f.shape[0] for f in feats if f is not None), 512)
    for i, f in enumerate(feats):
        if f is None:
            feats[i] = np.zeros(D, dtype=np.float32)
    features = np.stack(feats).astype(np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)
    ok_arr = np.asarray(ok, dtype=np.int32)
    dt = time.time() - t0
    logger.info("  clip_raw done: %d/%d ok in %.1fs (%.1f fps)",
                int(ok_arr.sum()), len(uris), dt, len(uris) / max(dt, 1e-3))
    return features, scores_arr, ok_arr


# =============================================================================
# Save NPZ + upload.
# =============================================================================
def save_and_upload(
    output_uri: str,
    features: np.ndarray,
    scores: np.ndarray,
    frame_paths: List[str],
    labels: np.ndarray,
    pair_ids: List[str],
    ok: np.ndarray,
    meta: dict,
) -> None:
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
        npz_path = tf.name
    np.savez_compressed(
        npz_path,
        features=features.astype(np.float32),
        scores=scores.astype(np.float32),
        frame_path=np.asarray(frame_paths, dtype=object),
        label=labels.astype(np.int32),
        pair_id=np.asarray(pair_ids, dtype=object),
        ok=ok.astype(np.int32),
        meta_json=np.array(json.dumps(meta), dtype=object),
    )
    upload_file(npz_path, output_uri)
    os.unlink(npz_path)


# =============================================================================
# Main.
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair_gaps_csv", required=True,
                    help="GCS URI or local path to pair_gaps.csv (5,311 unique frames).")
    ap.add_argument("--output_prefix", required=True,
                    help="GCS prefix for {p8a,e2b,clip_b16_raw}_paired_features.npz")
    ap.add_argument("--p8a_ckpt", required=True, help="GCS URI of P8A checkpoint .pth")
    ap.add_argument("--e2b_ckpt", required=True, help="GCS URI of E2B checkpoint .pth")
    ap.add_argument("--include_clip_raw", action="store_true",
                    help="Also extract raw OpenCLIP ViT-B-16 baseline features.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    if not args.output_prefix.endswith("/"):
        args.output_prefix = args.output_prefix + "/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Output prefix: %s", args.output_prefix)

    # 1. Pull pair_gaps.csv.
    if args.pair_gaps_csv.startswith("gs://"):
        local_csv = download_to_local(args.pair_gaps_csv)
    else:
        local_csv = args.pair_gaps_csv
    manifest = build_frame_manifest(local_csv)
    uris = manifest["frame_path"].tolist()
    labels = manifest["label"].astype(np.int32).to_numpy()
    pair_ids = manifest["pair_id"].tolist()

    # Persist manifest alongside features (one upload, shared across ckpts).
    manifest_local = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False).name
    manifest.to_csv(manifest_local, index=False)
    upload_file(manifest_local, args.output_prefix + "frame_manifest.csv")
    os.unlink(manifest_local)

    common_meta = {
        "pair_gaps_csv": args.pair_gaps_csv,
        "n_frames": int(len(manifest)),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "resolution": RESOLUTION,
        "clip_mean": CLIP_MEAN,
        "clip_std": CLIP_STD,
    }

    # 2. P8A.
    logger.info("=" * 70)
    logger.info("Extracting P8A features (%s)", args.p8a_ckpt)
    logger.info("=" * 70)
    p8a_local = download_to_local(args.p8a_ckpt)
    p8a_model = load_effort_model(p8a_local, device)
    feats, scores, ok = extract_with_effort(
        p8a_model, uris, device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    save_and_upload(
        args.output_prefix + "p8a_paired_features.npz",
        feats, scores, uris, labels, pair_ids, ok,
        meta={**common_meta, "ckpt_kind": "p8a", "ckpt_uri": args.p8a_ckpt,
              "feature_source": "out['feat'] post-projection"},
    )
    del p8a_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 3. E2B.
    logger.info("=" * 70)
    logger.info("Extracting E2B features (%s)", args.e2b_ckpt)
    logger.info("=" * 70)
    e2b_local = download_to_local(args.e2b_ckpt)
    e2b_model = load_effort_model(e2b_local, device)
    feats, scores, ok = extract_with_effort(
        e2b_model, uris, device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    save_and_upload(
        args.output_prefix + "e2b_paired_features.npz",
        feats, scores, uris, labels, pair_ids, ok,
        meta={**common_meta, "ckpt_kind": "e2b", "ckpt_uri": args.e2b_ckpt,
              "feature_source": "out['feat'] post-projection"},
    )
    del e2b_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 4. CLIP_B16_raw (optional).
    if args.include_clip_raw:
        logger.info("=" * 70)
        logger.info("Extracting CLIP_B16_raw baseline features (no FT)")
        logger.info("=" * 70)
        visual = load_clip_b16_raw(device)
        feats, scores, ok = extract_with_clip_raw(
            visual, uris, device,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        save_and_upload(
            args.output_prefix + "clip_b16_raw_paired_features.npz",
            feats, scores, uris, labels, pair_ids, ok,
            meta={**common_meta, "ckpt_kind": "clip_b16_raw",
                  "ckpt_uri": "openclip/ViT-B-16/datacomp_xl_s13b_b90k",
                  "feature_source": "visual(x) post-projection (frozen baseline)"},
        )
        del visual
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("=" * 70)
    logger.info("Done — all features uploaded under %s", args.output_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
