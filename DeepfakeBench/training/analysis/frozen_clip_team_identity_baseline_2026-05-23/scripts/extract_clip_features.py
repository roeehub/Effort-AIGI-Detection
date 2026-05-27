"""Extract OpenCLIP B16 DataComp.XL L11 CLS features for the team-identity cohort frames.

Reads from `analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/per_frame_full.csv`
and extracts features for all `deploy_relevant=True` frames (5,941 frames as of 2026-05-23).

Preprocessing matches D8 exactly:
- 224x224 cv2 INTER_LINEAR resize
- CLIP mean/std normalization (the OpenCLIP norms)
- Forward hook on visual.transformer.resblocks[11], CLS token (index 0)
- MPS-batched at 32 per batch

For local-bucket frames, uses the local_frame_resolver from the expanded readout.
For GCS frames, uses google-cloud-storage to download bytes.

Caches features to outputs/clip_frozen_l11__team_identity_n{N}.npz with arrays:
  - features: (N, 768) float32
  - frame_paths: (N,) <U
  - labels: (N,) int64 (0=real, 1=fake)
  - humans: (N,) <U
  - roles: (N,) <U
  - deploy_relevant: (N,) bool
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPANDED_READOUT = REPO_ROOT / "analysis/team_identity_deploy_readout_expanded_2026-05-23"
THIS_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = THIS_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Make the team-identity readout scripts importable for local_frame_resolver
sys.path.insert(0, str(EXPANDED_READOUT / "scripts"))
from local_frame_resolver import resolve_local  # noqa: E402

PER_FRAME_CSV = EXPANDED_READOUT / "outputs/per_frame_full.csv"

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
BATCH_SIZE = 32

logger = logging.getLogger("extract_clip")


def load_clip_model():
    import torch
    import open_clip

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("CLIP-frozen device: %s", device)

    local_pretrained = (
        REPO_ROOT / "weights" / "CLIP-ViT-B-16-DataComp.XL-s13B-b90K"
        / "open_clip_pytorch_model.bin"
    )
    if not local_pretrained.exists():
        raise FileNotFoundError(f"CLIP weights not found at {local_pretrained}")
    logger.info("loading OpenCLIP B16 from %s", local_pretrained)
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(local_pretrained)
    )
    model = model.to(device).eval()
    return model, device


def make_hook(captured: List[np.ndarray]):
    def hook(_module, _input, output):
        import torch
        if output.dim() == 3:
            if output.shape[0] >= output.shape[1]:
                cls = output[0]
            else:
                cls = output[:, 0]
        elif output.dim() == 2:
            cls = output
        else:
            raise RuntimeError(f"unexpected resblock output: {tuple(output.shape)}")
        captured.append(cls.detach().cpu().to(torch.float32).numpy())
    return hook


def resolve_frame_path(frame_path: str) -> Optional[str]:
    """Return a local filesystem path for frame_path, or None if not available locally."""
    # Try local resolver first for both gs://local/ and gs://live-fakes-teams-prod/real/
    rp = resolve_local(frame_path)
    if rp is not None and Path(rp).exists():
        return rp
    return None


def fetch_gcs(uri: str, client_cache: dict):
    """Download GCS bytes; return BGR np.ndarray or None."""
    import cv2
    try:
        if "storage_client" not in client_cache:
            from google.cloud import storage  # type: ignore
            client_cache["storage_client"] = storage.Client()
        client = client_cache["storage_client"]
    except Exception as e:
        logger.error("gcs client init failed: %s", e)
        return None
    no_scheme = uri[len("gs://"):]
    bucket, blob_path = no_scheme.split("/", 1)
    try:
        b = client.bucket(bucket).blob(blob_path)
        data = b.download_as_bytes()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.warning("gcs miss %s: %s", uri, e)
        return None


def preprocess_bgr(img_bgr: np.ndarray):
    import cv2
    img = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
    return img.transpose(2, 0, 1)


def _run_batch(model, device, imgs, idx_list, captured, dest):
    import torch
    batch = torch.from_numpy(np.stack(imgs, axis=0).astype(np.float32)).to(device, non_blocking=True)
    captured.clear()
    with torch.inference_mode():
        _ = model.visual(batch)
    if not captured:
        raise RuntimeError(f"no capture for batch starting at idx {idx_list[0]}")
    cls_batch = captured[0]
    if cls_batch.shape[0] != len(idx_list):
        raise RuntimeError(
            f"capture shape mismatch ({cls_batch.shape[0]} vs {len(idx_list)})"
        )
    for k, ii in enumerate(idx_list):
        dest[ii] = cls_batch[k]


def extract_all(df: pd.DataFrame, out_npz: Path):
    import cv2
    import torch

    model, device = load_clip_model()
    visual = model.visual
    captured: List[np.ndarray] = []
    handle = visual.transformer.resblocks[11].register_forward_hook(make_hook(captured))

    frame_paths = df["frame_path"].tolist()
    labels = df["label"].astype(np.int64).to_numpy()
    humans = df["human"].astype(str).to_numpy()
    roles = df["role"].astype(str).to_numpy()
    deploy_flags = df["deploy_relevant"].astype(bool).to_numpy()

    n = len(frame_paths)
    feats_per_frame: List[Optional[np.ndarray]] = [None] * n
    pending_imgs: List = []
    pending_idx: List[int] = []
    client_cache: dict = {}
    t0 = time.time()
    n_local = 0
    n_gcs = 0
    n_failed = 0
    n_done = 0

    try:
        for i, fp in enumerate(frame_paths):
            # 1) Try local-resolver
            local_path = resolve_frame_path(fp)
            img_bgr = None
            if local_path is not None:
                img_bgr = cv2.imread(local_path, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    n_local += 1
            # 2) If not local or failed, try GCS for gs:// URIs (skip gs://local/)
            if img_bgr is None and fp.startswith("gs://") and not fp.startswith("gs://local/"):
                img_bgr = fetch_gcs(fp, client_cache)
                if img_bgr is not None:
                    n_gcs += 1
            if img_bgr is None:
                # Last-resort: try the live-fakes-teams-prod local fallback
                if fp.startswith("gs://live-fakes-teams-prod/"):
                    # already tried via resolve_frame_path
                    pass
                logger.warning("FAIL to load %s", fp)
                n_failed += 1
                continue

            pending_imgs.append(preprocess_bgr(img_bgr))
            pending_idx.append(i)
            if len(pending_imgs) >= BATCH_SIZE:
                _run_batch(model, device, pending_imgs, pending_idx, captured, feats_per_frame)
                pending_imgs, pending_idx = [], []
                n_done += BATCH_SIZE
                if n_done % (BATCH_SIZE * 10) == 0:
                    elapsed = time.time() - t0
                    fps = n_done / max(elapsed, 1e-6)
                    eta_s = elapsed / max(n_done, 1) * (n - n_done)
                    logger.info(
                        "  done %d/%d  fps=%.1f  elapsed=%.0fs  eta=%.1fmin  local=%d gcs=%d failed=%d",
                        n_done, n, fps, elapsed, eta_s / 60.0, n_local, n_gcs, n_failed,
                    )
        if pending_imgs:
            _run_batch(model, device, pending_imgs, pending_idx, captured, feats_per_frame)
    finally:
        handle.remove()

    n_ok = sum(1 for f in feats_per_frame if f is not None)
    logger.info(
        "extraction done in %.1fs. n=%d ok=%d failed=%d (local=%d gcs=%d)",
        time.time() - t0, n, n_ok, n - n_ok, n_local, n_gcs,
    )

    # Pack into arrays. For failed frames, fill with NaN so downstream can mask.
    feats = np.full((n, 768), np.nan, dtype=np.float32)
    for i, f in enumerate(feats_per_frame):
        if f is not None:
            feats[i] = f

    np.savez_compressed(
        out_npz,
        features=feats,
        frame_paths=np.array(frame_paths, dtype=object),
        labels=labels,
        humans=humans,
        roles=roles,
        deploy_relevant=deploy_flags,
    )
    logger.info("wrote %s (%.1f MB)", out_npz, out_npz.stat().st_size / 1e6)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        handlers=[
            logging.FileHandler(THIS_DIR / "outputs/_extract.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    if not PER_FRAME_CSV.exists():
        logger.error("missing %s", PER_FRAME_CSV)
        return 1

    df = pd.read_csv(PER_FRAME_CSV, low_memory=False)
    logger.info("loaded %d rows from %s", len(df), PER_FRAME_CSV)
    # Subset deploy-relevant only (5,941 frames)
    df_deploy = df[df.deploy_relevant.astype(bool)].reset_index(drop=True)
    logger.info("deploy-relevant rows: %d", len(df_deploy))

    out_npz = OUTPUTS / f"clip_frozen_l11__team_identity_n{len(df_deploy)}.npz"
    if out_npz.exists():
        logger.info("CACHE HIT: %s already exists. Set rm to re-extract.", out_npz)
        return 0

    extract_all(df_deploy, out_npz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
