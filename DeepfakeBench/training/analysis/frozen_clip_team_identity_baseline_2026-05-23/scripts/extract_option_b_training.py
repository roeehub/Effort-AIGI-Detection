"""Extract OpenCLIP B16 L11 CLS features for a sample of the actual training corpus (Option B).

Samples N=3000 real + 3000 fake frames from frame_properties.parquet (the training manifest the
FT'd ckpts saw), extracts frozen-CLIP features identically to D8 and to extract_clip_features.py.

Sampling: stratified by method (so we get diversity across deep-live-cam, simswap, neuraltextures, etc.).
Within method, uniform random.

Output: outputs/clip_frozen_l11__training_optionB_n{N}.npz with arrays
  features (N,768), labels (N,), methods (N,), paths (N,)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = THIS_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

TRAIN_MANIFEST = REPO_ROOT / "frame_properties.parquet"

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
BATCH_SIZE = 32

logger = logging.getLogger("extract_optionb")


def stratified_sample(df: pd.DataFrame, label: str, n_target: int, rng: np.random.Generator) -> pd.DataFrame:
    """Per-method stratified sample. Return up to n_target rows of the given label."""
    sub = df[df.label == label]
    methods = sub.method.value_counts().to_dict()
    # Allocate per-method quotas proportional to method size, with a floor of 5
    total = sum(methods.values())
    out_dfs = []
    for m, n_m in methods.items():
        quota = max(5, int(round(n_target * n_m / total)))
        quota = min(quota, n_m)
        m_sub = sub[sub.method == m]
        if len(m_sub) <= quota:
            out_dfs.append(m_sub)
        else:
            idx = rng.choice(len(m_sub), size=quota, replace=False)
            out_dfs.append(m_sub.iloc[idx])
    pooled = pd.concat(out_dfs, axis=0).reset_index(drop=True)
    if len(pooled) > n_target:
        # Final cap with uniform sample
        keep = rng.choice(len(pooled), size=n_target, replace=False)
        pooled = pooled.iloc[keep].reset_index(drop=True)
    return pooled


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
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(local_pretrained)
    )
    model = model.to(device).eval()
    return model, device


def fetch_gcs(uri: str, client_cache: dict):
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


def run_batch(model, device, imgs, idx_list, captured, dest):
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


def extract(df: pd.DataFrame, out_npz: Path):
    import torch
    model, device = load_clip_model()
    visual = model.visual
    captured = []
    def hook(_m, _i, output):
        if output.dim() == 3:
            if output.shape[0] >= output.shape[1]:
                cls = output[0]
            else:
                cls = output[:, 0]
        else:
            cls = output
        captured.append(cls.detach().cpu().to(torch.float32).numpy())
    handle = visual.transformer.resblocks[11].register_forward_hook(hook)

    paths = df["path"].tolist()
    labels_str = df["label"].tolist()
    methods = df["method"].tolist()
    labels_int = np.array([1 if x == "fake" else 0 for x in labels_str], dtype=np.int64)
    n = len(paths)
    feats = [None] * n
    pending_imgs, pending_idx = [], []
    client_cache: dict = {}
    t0 = time.time()
    n_ok = 0
    n_fail = 0
    import cv2

    try:
        for i, p in enumerate(paths):
            img = fetch_gcs(p, client_cache)
            if img is None:
                n_fail += 1
                continue
            pending_imgs.append(preprocess_bgr(img))
            pending_idx.append(i)
            n_ok += 1
            if len(pending_imgs) >= BATCH_SIZE:
                run_batch(model, device, pending_imgs, pending_idx, captured, feats)
                pending_imgs, pending_idx = [], []
                done_now = n_ok
                if done_now % (BATCH_SIZE * 10) == 0:
                    elapsed = time.time() - t0
                    fps = done_now / max(elapsed, 1e-6)
                    eta_s = elapsed / max(done_now, 1) * (n - done_now - n_fail)
                    logger.info(
                        "  done %d/%d  fps=%.1f  fail=%d  elapsed=%.0fs  eta=%.1fmin",
                        done_now, n, fps, n_fail, elapsed, eta_s / 60.0,
                    )
        if pending_imgs:
            run_batch(model, device, pending_imgs, pending_idx, captured, feats)
    finally:
        handle.remove()

    logger.info("done in %.1fs  ok=%d  fail=%d", time.time() - t0, n_ok, n_fail)

    feats_arr = np.full((n, 768), np.nan, dtype=np.float32)
    for i, f in enumerate(feats):
        if f is not None:
            feats_arr[i] = f

    np.savez_compressed(
        out_npz,
        features=feats_arr,
        paths=np.array(paths, dtype=object),
        labels=labels_int,
        methods=np.array(methods, dtype=object),
    )
    logger.info("wrote %s (%.1f MB)", out_npz, out_npz.stat().st_size / 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=3000)
    ap.add_argument("--n-fake", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        handlers=[
            logging.FileHandler(OUTPUTS / "_extract_optionb.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    logger.info("loading training manifest %s", TRAIN_MANIFEST)
    df = pd.read_parquet(TRAIN_MANIFEST)
    logger.info("manifest: %d rows", len(df))

    # Drop rows with non-gs:// paths or empty
    df = df[df.path.astype(str).str.startswith("gs://")].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    real_sample = stratified_sample(df, "real", args.n_real, rng)
    fake_sample = stratified_sample(df, "fake", args.n_fake, rng)
    sample = pd.concat([real_sample, fake_sample], axis=0).reset_index(drop=True)
    logger.info("sample size: %d real + %d fake = %d", len(real_sample), len(fake_sample), len(sample))

    out_npz = OUTPUTS / f"clip_frozen_l11__training_optionB_n{len(sample)}.npz"
    if out_npz.exists():
        logger.info("CACHE HIT: %s", out_npz)
        return 0

    extract(sample, out_npz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
