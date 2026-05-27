#!/usr/bin/env python3
"""Feature-space extraction across 20 data sources for a given checkpoint.

Runs on Vertex AI with an A100 GPU. Samples N images per source (deterministic
with SEED=737), forward-passes them through the detector to collect the 512-dim
post-backbone features and the classifier probs, then uploads one .npz per
source to GCS.

Pairs with ./compute_distances.py, which runs locally after the Vertex job
finishes to compute pairwise Fréchet / MMD / centroid distances.

Usage (inside the training container, on Vertex):
    python3 analysis/feature_space_2026-04-23/extract_features.py \
        --checkpoint gs://.../top_n_effort_*.pth \
        --output_prefix gs://training-job-outputs/feature_space_analysis/<run_id>/ \
        --n_per_source 150
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
from google.cloud import storage

HERE = Path(__file__).resolve().parent
TRAINING_ROOT = HERE.parent.parent  # .../training/
sys.path.insert(0, str(TRAINING_ROOT))

from detectors import DETECTOR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("feature-space-extract")

# =============================================================================
# Constants
# =============================================================================
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

DEFAULT_MANIFEST = (
    TRAINING_ROOT
    / "arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json"
)

DEFAULT_N_PER_SOURCE = 150
OVERSAMPLE = 2
SEED = 737

DEEPLIVE_NON_ENHANCED = ("edge_cases", "minimal_processing", "quality_enhancement")
DEEPLIVE_ENHANCED = ("edge_cases_enhanced", "minimal_processing_enhanced")
VISOMASTER_IN_LIVE_BUCKET = (
    "visomaster_CSCS", "visomaster_GhostFace-v1", "visomaster_GhostFace-v2",
    "visomaster_GhostFace-v3", "visomaster_InStyleSwapper256-A",
    "visomaster_InStyleSwapper256-B", "visomaster_InStyleSwapper256-C",
    "visomaster_Inswapper128", "visomaster_SimSwap512",
)
IMG_EXTS = (".png", ".jpg", ".jpeg")

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
# Source sampling (pure Python, no gsutil) — mirrors ../bucket_comparison_2026-04-23/sample_and_analyze.py
# =============================================================================
@dataclass
class Source:
    name: str
    provenance: str
    label: str
    uris: list


def _list_dirs(client: storage.Client, bucket: str, prefix: str) -> list:
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"
    it = client.list_blobs(bucket, prefix=prefix, delimiter="/")
    _ = list(it)
    return [f"gs://{bucket}/{p}" for p in it.prefixes]


def _list_direct(client: storage.Client, bucket: str, prefix: str) -> list:
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"
    uris = []
    for blob in client.list_blobs(bucket, prefix=prefix, delimiter="/"):
        if blob.name.endswith("/"):
            continue
        if "/" in blob.name[len(prefix):]:
            continue
        if blob.name.lower().endswith(IMG_EXTS):
            uris.append(f"gs://{bucket}/{blob.name}")
    return uris


def _rnd(seq, n, rng):
    if len(seq) <= n:
        return list(seq)
    return rng.sample(list(seq), n)


def _session_matches_strategy(session_url: str, strategies: tuple) -> bool:
    name = session_url.rstrip("/").split("/")[-1]
    best = None
    for s in strategies:
        if name.startswith(s + "_"):
            rest = name[len(s) + 1:]
            if rest.isdigit():
                if best is None or len(s) > len(best):
                    best = s
    return best is not None


def _resolve_deeplive_like(client, bucket, strategies, subdirs_to_try, child, n, rng):
    sessions = _list_dirs(client, bucket, "samples/")
    matching = [s for s in sessions if _session_matches_strategy(s, strategies)]
    if not matching:
        return []
    chosen = _rnd(matching, subdirs_to_try, rng)
    uris = []
    for sess in chosen:
        _, _, rest = sess[5:].partition("/")
        frames = _list_direct(client, bucket, f"{rest}frames/{child}/")
        if not frames:
            continue
        pick = _rnd(frames, max(2, n // subdirs_to_try), rng)
        uris.extend(pick)
        if len(uris) >= n:
            break
    return uris[:n]


def build_proper_lane_uris(manifest_path: Path, lane: str, label_filter: str, n: int, rng) -> list:
    with open(manifest_path) as f:
        m = json.load(f)
    videos = m["videos"]
    if label_filter == "fake":
        matched = [v for v in videos if v.get("lane") == lane]
    else:
        target_ids = {v.get("identity_id") for v in videos if v.get("lane") == lane}
        target_lane = "proper_real_teams" if lane.endswith("_teams") else "proper_real_clean"
        matched = [
            v for v in videos
            if v.get("label") == "real"
            and v.get("identity_id") in target_ids
            and v.get("lane") == target_lane
        ]
    if not matched:
        return []
    rng.shuffle(matched)
    uris = []
    for v in matched:
        paths = v.get("frame_paths") or []
        if not paths:
            continue
        uris.append(rng.choice(paths))
        if len(uris) >= n:
            break
    return uris


def build_external_vcd_uris(client, n, rng):
    top = _list_dirs(client, "effort-collected-data", "real/VCD/")
    uris = []
    for d in _rnd(top, 8, rng):
        _, _, rest = d[5:].partition("/")
        frames = _list_direct(client, "effort-collected-data", rest)
        if not frames:
            subs = _list_dirs(client, "effort-collected-data", rest)
            for s in subs[:2]:
                _, _, srest = s[5:].partition("/")
                frames.extend(_list_direct(client, "effort-collected-data", srest))
        if frames:
            uris.extend(_rnd(frames, max(3, n // 8), rng))
        if len(uris) >= n:
            break
    return uris[:n]


def build_external_avspeech_uris(client, n, rng):
    top = _list_dirs(client, "effort-collected-data", "real/external_youtube_avspeech/")
    uris = []
    for d in _rnd(top, 16, rng):
        _, _, rest = d[5:].partition("/")
        frames = _list_direct(client, "effort-collected-data", rest)
        if frames:
            uris.extend(_rnd(frames, max(2, n // 16), rng))
        if len(uris) >= n:
            break
    return uris[:n]


def build_wma_fake_uris(client, n, rng):
    top = _list_dirs(client, "effort-collected-data", "wma_validation/enhanced_fake/")
    uris = []
    for d in _rnd(top, 6, rng):
        _, _, rest = d[5:].partition("/")
        frames = _list_direct(client, "effort-collected-data", rest)
        if not frames:
            subs = _list_dirs(client, "effort-collected-data", rest)
            for s in subs[:3]:
                _, _, srest = s[5:].partition("/")
                frames.extend(_list_direct(client, "effort-collected-data", srest))
        if frames:
            uris.extend(_rnd(frames, max(5, n // 6), rng))
        if len(uris) >= n:
            break
    return uris[:n]


def build_enhanced_v2_uris(client, n, rng):
    methods = _list_dirs(client, "visomaster-enhanced-face-cropped-v2", "fake/")
    uris = []
    for d in _rnd(methods, 12, rng):
        _, _, rest = d[5:].partition("/")
        frames = _list_direct(client, "visomaster-enhanced-face-cropped-v2", rest)
        if frames:
            uris.extend(_rnd(frames, max(2, n // 12), rng))
        if len(uris) >= n:
            break
    return uris[:n]


def discover_sources(manifest_path: Path, n_per_source: int, seed: int) -> List[Source]:
    rng = random.Random(seed)
    n = n_per_source * OVERSAMPLE
    client = storage.Client()

    sources: List[Source] = []
    DL = "live-deepfake-methods-real-and-fake-frames-cropped"
    TV2 = "live-deepfake-methods-real-and-fake-frames-cropped-teams-v2"

    for label in ("fake", "real"):
        uris = _resolve_deeplive_like(client, DL, DEEPLIVE_NON_ENHANCED, 8, label, n, rng)
        sources.append(Source(f"deeplive_non_enh_{label}",
                              "user_created_training__fam=deeplive_non_enhanced", label, uris))
        uris = _resolve_deeplive_like(client, DL, DEEPLIVE_ENHANCED, 8, label, n, rng)
        sources.append(Source(f"deeplive_enh_{label}",
                              "user_created_training__fam=deeplive_enhanced", label, uris))
        uris = _resolve_deeplive_like(client, DL, VISOMASTER_IN_LIVE_BUCKET, 8, label, n, rng)
        sources.append(Source(f"dl_bucket_visomaster_{label}",
                              "user_created_UNUSED__visomaster_in_dl_bucket", label, uris))

    for label in ("fake", "real"):
        uris = _resolve_deeplive_like(client, TV2, DEEPLIVE_NON_ENHANCED, 8, label, n, rng)
        sources.append(Source(f"tv2_deeplive_{label}",
                              "user_created_training+ood__fam=deeplive_teams", label, uris))
        uris = _resolve_deeplive_like(client, TV2, VISOMASTER_IN_LIVE_BUCKET, 8, label, n, rng)
        sources.append(Source(f"tv2_visomaster_{label}",
                              "user_created_training+ood__fam=deeplive_teams", label, uris))

    for lane in ("proper_visomaster_clean", "proper_visomaster_teams",
                 "proper_visomaster_enhanced_clean", "proper_visomaster_enhanced_teams"):
        prov = ("user_created_UNUSED__proper" if lane == "proper_visomaster_enhanced_clean"
                else "user_created_training__proper")
        uris = build_proper_lane_uris(manifest_path, lane, "fake", n, rng)
        sources.append(Source(f"{lane}_fake", prov, "fake", uris))
    for lane in ("proper_visomaster_clean", "proper_visomaster_teams"):
        uris = build_proper_lane_uris(manifest_path, lane, "real", n, rng)
        short = lane.replace("proper_visomaster_", "proper_real_") + "__paired"
        sources.append(Source(short, "user_created_training__proper", "real", uris))

    sources.append(Source("visomaster_enhanced_v2_fake", "candidate_new_data", "fake",
                          build_enhanced_v2_uris(client, n, rng)))
    sources.append(Source("external_vcd_real", "external_collected__ood_gate", "real",
                          build_external_vcd_uris(client, n, rng)))
    sources.append(Source("external_youtube_avspeech_real", "external_collected__ood_gate", "real",
                          build_external_avspeech_uris(client, n, rng)))
    sources.append(Source("wma_failure_fake", "external_collected__ood_gate", "fake",
                          build_wma_fake_uris(client, n, rng)))

    return sources


# =============================================================================
# GCS helpers
# =============================================================================
def parse_gs_uri(uri: str) -> Tuple[str, str]:
    assert uri.startswith("gs://"), f"Expected gs:// URI: {uri}"
    bucket, _, blob = uri[5:].partition("/")
    return bucket, blob


def download_checkpoint(uri: str) -> str:
    bucket, blob_name = parse_gs_uri(uri)
    client = storage.Client()
    blob = client.bucket(bucket).blob(blob_name)
    if blob.exists(client=client) and blob_name.endswith(".pth"):
        pass  # exact path given
    else:
        # directory: pick the latest value_composite_* checkpoint (highest step),
        # fall back to ood_composite_*, fall back to top_n_effort_*
        if not blob_name.endswith("/"):
            blob_name = blob_name + "/"
        buckets_by_prefix = {"value_composite": [], "ood_composite": [], "top_n_effort": []}
        for b in client.list_blobs(bucket, prefix=blob_name):
            if not b.name.endswith(".pth"):
                continue
            base = Path(b.name).name
            for key in buckets_by_prefix:
                if base.startswith(key + "_"):
                    buckets_by_prefix[key].append(b)
                    break
        def _step(b):
            base = Path(b.name).name
            for part in base.split("_"):
                if part.startswith("step"):
                    try:
                        return int(part[4:].rstrip(".pth"))
                    except ValueError:
                        return -1
            return -1
        chosen = None
        for key in ("value_composite", "ood_composite", "top_n_effort"):
            if buckets_by_prefix[key]:
                buckets_by_prefix[key].sort(key=_step, reverse=True)
                chosen = buckets_by_prefix[key][0]
                logger.info("Auto-selected %s checkpoint: %s", key, chosen.name)
                break
        if chosen is None:
            raise FileNotFoundError(f"No checkpoint under {uri}")
        blob = chosen
    local_dir = tempfile.mkdtemp(prefix="fs_ckpt_")
    local_path = os.path.join(local_dir, Path(blob.name).name)
    logger.info("Downloading checkpoint gs://%s/%s → %s (%.1f MB)",
                blob.bucket.name, blob.name, local_path,
                (blob.size or 0) / 1e6)
    blob.download_to_filename(local_path)
    return local_path


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
# Model
# =============================================================================
def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    cfg = dict(DETECTOR_CONFIG)
    download_backbone_weights(cfg)

    logger.info("Building detector…")
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
    if missing:
        logger.info("Missing keys: %d (first 5: %s)", len(missing), list(missing)[:5])
    if unexpected:
        logger.info("Unexpected keys: %d (first 5: %s)", len(unexpected), list(unexpected)[:5])

    if model_config.get("use_arcface_head") and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])
            logger.info("Restored ArcFace s=%.3f", model_config["current_arcface_s"])

    model.eval()
    return model


# =============================================================================
# Dataset + extraction
# =============================================================================
class GCSUriDataset(data.Dataset):
    def __init__(self, uris: List[str], resolution: int = 224):
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
        bucket_name, blob_name = parse_gs_uri(uri)
        blob = self._bucket(bucket_name).blob(blob_name)
        try:
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


def extract_source(model, uris, device, batch_size, num_workers) -> Dict[str, np.ndarray]:
    dataset = GCSUriDataset(uris)
    loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    feats_by_idx = [None] * len(uris)
    probs_by_idx = [0.0] * len(uris)
    ok_by_idx = [0] * len(uris)

    t0 = time.time()
    for images, indices, ok in loader:
        images = images.to(device, non_blocking=True)
        with torch.inference_mode():
            out = model({"image": images}, inference=True)
            feat = out["feat"].detach().cpu().numpy()
            prob = out["prob"].detach().cpu().numpy().reshape(-1)
        for j, idx_j in enumerate(indices.numpy()):
            feats_by_idx[int(idx_j)] = feat[j]
            probs_by_idx[int(idx_j)] = float(prob[j])
            ok_by_idx[int(idx_j)] = int(ok[j])

    valid = [i for i, f in enumerate(feats_by_idx) if f is not None]
    features = np.stack([feats_by_idx[i] for i in valid]) if valid else np.zeros((0, 512))
    probs = np.array([probs_by_idx[i] for i in valid])
    ok_arr = np.array([ok_by_idx[i] for i in valid], dtype=np.uint8)
    dt = time.time() - t0
    logger.info("  processed %d images in %.1fs (%.1f fps); ok=%d",
                len(valid), dt, len(valid) / max(dt, 1e-3), int(ok_arr.sum()))
    return {"features": features, "probs": probs, "ok": ok_arr, "valid_idx": np.array(valid)}


def upload_file(local_path: str, gcs_uri: str) -> None:
    bucket, blob = parse_gs_uri(gcs_uri)
    storage.Client().bucket(bucket).blob(blob).upload_from_filename(local_path)
    logger.info("Uploaded %s → %s", local_path, gcs_uri)


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_prefix", required=True)
    ap.add_argument("--n_per_source", type=int, default=DEFAULT_N_PER_SOURCE)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--manifest_path", type=str, default=str(DEFAULT_MANIFEST))
    args = ap.parse_args()

    if not args.output_prefix.endswith("/"):
        args.output_prefix = args.output_prefix + "/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    ckpt_path = download_checkpoint(args.checkpoint)
    model = load_model(ckpt_path, device)

    logger.info("Sampling URIs (n_per_source=%d, seed=%d)…",
                args.n_per_source, args.seed)
    sources = discover_sources(
        manifest_path=Path(args.manifest_path),
        n_per_source=args.n_per_source,
        seed=args.seed,
    )
    for s in sources:
        logger.info("  %-45s %-60s %-5s n=%d", s.name, s.provenance, s.label, len(s.uris))

    manifest_out = {
        "checkpoint": args.checkpoint,
        "n_per_source": args.n_per_source,
        "seed": args.seed,
        "sources": [asdict(s) for s in sources],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(manifest_out, f, indent=2)
        manifest_local = f.name
    upload_file(manifest_local, args.output_prefix + "sampling_manifest.json")
    os.unlink(manifest_local)

    for s in sources:
        if not s.uris:
            logger.warning("Source %s has no URIs — skipping", s.name)
            continue
        logger.info("Processing %s (%d URIs) …", s.name, len(s.uris))
        result = extract_source(model, s.uris, device,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
            tmp_path = tf.name
        np.savez_compressed(
            tmp_path,
            features=result["features"].astype(np.float32),
            probs=result["probs"].astype(np.float32),
            ok=result["ok"],
            valid_idx=result["valid_idx"],
            uris=np.array(s.uris, dtype=object),
            name=s.name,
            provenance=s.provenance,
            label=s.label,
        )
        upload_file(tmp_path, args.output_prefix + f"{s.name}.npz")
        os.unlink(tmp_path)

    logger.info("Done — results at %s", args.output_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
