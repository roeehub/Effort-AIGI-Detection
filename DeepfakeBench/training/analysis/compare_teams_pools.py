"""
compare_teams_pools.py — local diagnostic for distributional differences
between the `teams_ood` OOD pool and the `proper_visomaster_teams` training
pool. Spec: docs/superpowers/specs/2026-04-22-compare-teams-pools-design.md.

Usage:
    python analysis/compare_teams_pools.py \\
        --checkpoint gs://.../value_composite_*.pth \\
        --videos-per-group 150 --frames-per-video 8 \\
        --output-dir scratch/teams_pool_diff
"""
from __future__ import annotations

# === Section 1: CLI + paths =================================================

import argparse
import datetime as _dt
import logging
import sys
from pathlib import Path

logger = logging.getLogger("compare_teams_pools")

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "teams_pool_diff"
DEFAULT_OUTPUT_DIR = Path("scratch") / "teams_pool_diff"
DEFAULT_CHECKPOINT = (
    "gs://training-job-outputs/phase2r13_experiments/w92amaaa/"
    "value_composite_effort_20260422_step2500_auc0.9895_eer0.0116.pth"
)
DEFAULT_SEED = 737
KNOWN_PASSES = ("stats", "geometry", "model", "gallery")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compare_teams_pools",
        description="Diagnose distributional differences between teams_ood "
                    "and proper_visomaster_teams pools.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="GCS URI of the checkpoint to use for embeddings/confidence.")
    p.add_argument("--videos-per-group", type=int, default=150)
    p.add_argument("--frames-per-video", type=int, default=8)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--skip-pass", action="append", default=[],
                   choices=KNOWN_PASSES,
                   help="Skip a pass by name (repeatable).")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p


def _make_run_dir(output_dir: Path) -> Path:
    """Create a timestamped subdir under output_dir and return it."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = output_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "thumbnails").mkdir()
    return run_dir


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


# === Section 2: Sampling + GCS download (cached) ============================

import dataclasses
import hashlib
import random
from typing import Iterable

# Lazy GCS client — populated on first call.
_GCS_CLIENT = None


@dataclasses.dataclass(frozen=True)
class GroupDef:
    """A pool to compare. `buckets` lists gs:// roots that each contain a
    `samples/<video_id>/frames/<label>/...` subtree.
    """
    name: str
    label: str  # "real" | "fake"
    buckets: tuple[str, ...]


# All four groups share the layout `<bucket>/samples/<video>/frames/<label>/<frame>.*`.
# Verified 2026-04-22.
GROUP_DEFINITIONS: dict[str, GroupDef] = {
    "teams_ood_real": GroupDef(
        name="teams_ood_real",
        label="real",
        buckets=(
            "gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2",
        ),
    ),
    "teams_ood_fake": GroupDef(
        name="teams_ood_fake",
        label="fake",
        buckets=(
            "gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2",
        ),
    ),
    "proper_visomaster_teams_real": GroupDef(
        name="proper_visomaster_teams_real",
        label="real",
        buckets=(
            "gs://hdtf_visomaster_cropped_frames_teams",
            "gs://quickclips_visomaster_cropped_frames_teams",
        ),
    ),
    "proper_visomaster_teams_fake": GroupDef(
        name="proper_visomaster_teams_fake",
        label="fake",
        buckets=(
            "gs://hdtf_visomaster_cropped_frames_teams",
            "gs://quickclips_visomaster_cropped_frames_teams",
        ),
    ),
}


def _sample_video_ids(pool: Iterable[str], n: int, seed: int) -> list[str]:
    """Deterministic sample of n video ids from pool. Returns sorted IDs.

    If pool has fewer than n items, returns the whole pool sorted.
    """
    items = sorted(set(pool))
    if len(items) <= n:
        return items
    rng = random.Random(seed)
    return sorted(rng.sample(items, n))


def _gcs_client():
    """Lazy GCS client; surfaces a helpful auth message on failure."""
    global _GCS_CLIENT
    if _GCS_CLIENT is not None:
        return _GCS_CLIENT
    try:
        from google.cloud import storage
        _GCS_CLIENT = storage.Client()
        return _GCS_CLIENT
    except Exception as exc:
        logger.error(
            "GCS client init failed: %s. "
            "Run `gcloud auth application-default login` and retry.",
            exc,
        )
        raise SystemExit(2)


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    assert uri.startswith("gs://"), f"not a gs:// URI: {uri}"
    bucket, _, prefix = uri[5:].partition("/")
    return bucket, prefix


def _list_video_dirs_in_bucket(bucket_uri: str) -> list[str]:
    """Return video subdirectory names under <bucket>/samples/.

    Uses delimiter listing — does NOT enumerate the full file tree.
    Returns just the leaf folder names (e.g. 'edge_cases_0000').
    """
    client = _gcs_client()
    bucket_name, _ = _parse_gs_uri(bucket_uri)
    bucket = client.bucket(bucket_name)
    iterator = client.list_blobs(bucket, prefix="samples/", delimiter="/")
    # Force iteration to populate prefixes.
    list(iterator)
    out: list[str] = []
    for prefix in iterator.prefixes:
        # prefix looks like 'samples/edge_cases_0000/'
        rel = prefix[len("samples/"):].rstrip("/")
        if rel:
            out.append(rel)
    return sorted(out)


def _list_frames_in_video(
    bucket_uri: str, video_id: str, label: str,
) -> list[tuple[str, str]]:
    """Return [(blob_name, gs_uri), ...] for frames under a video's label folder."""
    client = _gcs_client()
    bucket_name, _ = _parse_gs_uri(bucket_uri)
    bucket = client.bucket(bucket_name)
    prefix = f"samples/{video_id}/frames/{label}/"
    out: list[tuple[str, str]] = []
    for blob in client.list_blobs(bucket, prefix=prefix):
        if blob.name.lower().endswith((".jpg", ".jpeg", ".png")):
            out.append((blob.name, f"gs://{bucket_name}/{blob.name}"))
    out.sort()
    return out


def _download_frames_for_video(
    bucket_uri: str, video_id: str, label: str,
    n_frames: int, cache_dir: Path,
) -> list[Path]:
    """Download up to n_frames evenly-spaced frames from a single video subdir.

    Returns local file paths in order. Idempotent: skips downloads if the
    cache already has the same file count for this video.
    """
    cache_key = hashlib.blake2b(
        f"{bucket_uri}|{video_id}|{label}".encode(), digest_size=8,
    ).hexdigest()
    local_dir = cache_dir / cache_key
    local_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(local_dir.glob("*"))
    existing = [p for p in existing if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if len(existing) >= n_frames:
        return existing[:n_frames]
    frames = _list_frames_in_video(bucket_uri, video_id, label)
    if not frames:
        return []
    if len(frames) <= n_frames:
        chosen = frames
    else:
        step = len(frames) / n_frames
        chosen = [frames[int(i * step)] for i in range(n_frames)]
    client = _gcs_client()
    bucket_name, _ = _parse_gs_uri(bucket_uri)
    bucket = client.bucket(bucket_name)
    out: list[Path] = []
    for blob_name, _gs in chosen:
        local_path = local_dir / Path(blob_name).name
        if not local_path.exists():
            bucket.blob(blob_name).download_to_filename(str(local_path))
        out.append(local_path)
    return out


def collect_frames(
    *, videos_per_group: int, frames_per_video: int,
    cache_dir: Path, seed: int,
):
    """Top-level: enumerate, sample, download, return a DataFrame.

    Columns: group, label, video_id, bucket_uri, frame_path
    """
    import pandas as pd
    rows: list[dict] = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    for gname, gdef in GROUP_DEFINITIONS.items():
        candidates: list[tuple[str, str]] = []  # (bucket_uri, video_id)
        for bucket_uri in gdef.buckets:
            try:
                vids = _list_video_dirs_in_bucket(bucket_uri)
            except Exception as exc:
                logger.warning("listing %s failed: %s", bucket_uri, exc)
                continue
            candidates.extend((bucket_uri, v) for v in vids)
        keys = [f"{b}|{v}" for b, v in candidates]
        chosen_keys = set(_sample_video_ids(keys, n=videos_per_group, seed=seed))
        sampled_count = 0
        for (bucket_uri, video_id), key in zip(candidates, keys):
            if key not in chosen_keys:
                continue
            paths = _download_frames_for_video(
                bucket_uri, video_id, gdef.label,
                frames_per_video, cache_dir,
            )
            if not paths:
                continue
            sampled_count += 1
            for p in paths:
                rows.append({
                    "group": gname,
                    "label": gdef.label,
                    "video_id": video_id,
                    "bucket_uri": bucket_uri,
                    "frame_path": str(p),
                })
        logger.info("group=%s — %d videos sampled, %d frames downloaded",
                    gname, sampled_count,
                    sum(1 for r in rows if r["group"] == gname))
    return pd.DataFrame(rows)


# === Section 3: Pass 1 — image stats ========================================


def _colorfulness(img_rgb):
    """Hasler-Süsstrunk colorfulness metric on a HxWx3 uint8 RGB image."""
    import numpy as np
    r = img_rgb[..., 0].astype(np.float32)
    g = img_rgb[..., 1].astype(np.float32)
    b = img_rgb[..., 2].astype(np.float32)
    rg = r - g
    yb = 0.5 * (r + g) - b
    sigma_rgyb = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2))
    mu_rgyb = float(np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return sigma_rgyb + 0.3 * mu_rgyb


def _image_stats_from_array(img_rgb, *, file_size_bytes: int) -> dict:
    """Per-frame low-level stats. Input: HxWx3 uint8 RGB array."""
    import cv2
    import numpy as np
    h, w = img_rgb.shape[:2]
    luma = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[..., 1] / 255.0
    laplacian = cv2.Laplacian(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F,
    )
    return {
        "brightness": float(luma.mean()),
        "contrast": float(luma.std()),
        "sharpness": float(laplacian.var()),
        "colorfulness": float(_colorfulness(img_rgb)),
        "saturation": float(sat.mean()),
        "resolution_w": int(w),
        "resolution_h": int(h),
        "bytes_per_pixel": float(file_size_bytes / max(1, w * h)),
        "warmth": float(img_rgb[..., 0].mean() - img_rgb[..., 2].mean()),
    }


def _ks_distance(a, b) -> float:
    """Two-sample Kolmogorov-Smirnov distance (max |F_a(x) - F_b(x)|)."""
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), grid, side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _aggregate_per_group(df, feature_cols: list[str]) -> dict:
    import numpy as np
    agg: dict[str, dict] = {}
    for g in sorted(df["group"].unique()):
        sub = df[df["group"] == g]
        agg[g] = {}
        for f in feature_cols:
            col = sub[f].to_numpy(dtype=np.float64)
            col = col[np.isfinite(col)]
            if col.size == 0:
                agg[g][f] = {"mean": None, "median": None, "p5": None,
                             "p95": None, "n": 0}
                continue
            agg[g][f] = {
                "mean": float(np.mean(col)),
                "median": float(np.median(col)),
                "p5": float(np.percentile(col, 5)),
                "p95": float(np.percentile(col, 95)),
                "n": int(col.size),
            }
    return agg


_KS_PAIRS = (
    ("teams_ood_real", "proper_visomaster_teams_real"),
    ("teams_ood_fake", "proper_visomaster_teams_fake"),
)


def _cross_pool_ks(df, feature_cols: list[str]) -> dict:
    groups_present = set(df["group"].unique())
    ks: dict[str, dict] = {}
    for left, right in _KS_PAIRS:
        if left not in groups_present or right not in groups_present:
            continue
        ks[f"{left}__vs__{right}"] = {
            f: _ks_distance(
                df[df["group"] == left][f].to_numpy(),
                df[df["group"] == right][f].to_numpy(),
            )
            for f in feature_cols
        }
    return ks


def run_pass_stats(frames_df, *, run_dir: Path) -> dict:
    """Pass 1: per-frame low-level stats + per-group aggregates + KS pairs."""
    import json
    import pandas as pd
    from PIL import Image
    rows = []
    for _, row in frames_df.iterrows():
        path = Path(row["frame_path"])
        try:
            with Image.open(path) as im:
                import numpy as np
                img = np.array(im.convert("RGB"))
            s = _image_stats_from_array(img, file_size_bytes=path.stat().st_size)
        except Exception as exc:
            logger.warning("stats failed for %s: %s", path, exc)
            continue
        s.update({k: row[k] for k in ("group", "label", "video_id", "frame_path")})
        rows.append(s)
    if not rows:
        out = {"per_group": {}, "ks": {}, "feature_cols": [], "n_frames": 0}
        (run_dir / "stats_pass_stats.json").write_text(json.dumps(out, indent=2))
        return out
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns
                    if c not in ("group", "label", "video_id", "frame_path")]
    agg = _aggregate_per_group(df, feature_cols)
    ks = _cross_pool_ks(df, feature_cols)
    out = {"per_group": agg, "ks": ks, "feature_cols": feature_cols,
           "n_frames": int(len(df))}
    (run_dir / "stats_pass_stats.json").write_text(json.dumps(out, indent=2))
    df.to_parquet(run_dir / "raw_stats.parquet", index=False)
    return out


# === Section 4: Pass 2 — face geometry (mediapipe) ==========================

# MediaPipe face_mesh canonical landmark indices:
# nose_tip=1, chin=152, left_eye_outer=33, right_eye_outer=263,
# left_mouth=61, right_mouth=291.
_LEFT_EYE_IDX = 33
_RIGHT_EYE_IDX = 263
_NOSE_IDX = 1


_FACE_MESH = None


def _get_face_mesh():
    """Lazy singleton mediapipe FaceMesh — avoids per-frame XNNPACK init logs."""
    global _FACE_MESH
    if _FACE_MESH is None:
        import os
        os.environ.setdefault("GLOG_minloglevel", "2")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import mediapipe as mp
        _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=False,
        )
    return _FACE_MESH


def _face_geometry_from_array(img_rgb) -> dict:
    """Per-frame face geometry via MediaPipe face_mesh."""
    import math
    import numpy as np
    nan = float("nan")
    out = {
        "face_detected": False,
        "face_bbox_area_ratio": nan,
        "inter_eye_distance_ratio": nan,
        "yaw_deg": nan,
        "pitch_deg": nan,
        "roll_deg": nan,
    }
    try:
        mesh = _get_face_mesh()
    except ImportError:
        logger.warning("mediapipe not installed — geometry pass returns no-face rows")
        return out
    h, w = img_rgb.shape[:2]
    result = mesh.process(img_rgb)
    if not result.multi_face_landmarks:
        return out
    lms = result.multi_face_landmarks[0].landmark
    pts = np.array([(lm.x * w, lm.y * h) for lm in lms], dtype=np.float32)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    bbox_area = float(max(0.0, xmax - xmin) * max(0.0, ymax - ymin))
    crop_area = float(w * h)
    left_eye = pts[_LEFT_EYE_IDX]
    right_eye = pts[_RIGHT_EYE_IDX]
    ied = float(np.linalg.norm(right_eye - left_eye))
    roll = math.degrees(math.atan2(
        float(right_eye[1] - left_eye[1]),
        float(right_eye[0] - left_eye[0]),
    ))
    nose = pts[_NOSE_IDX]
    eye_mid_x = float((left_eye[0] + right_eye[0]) / 2)
    eye_mid_y = float((left_eye[1] + right_eye[1]) / 2)
    yaw = math.degrees(math.atan2(
        float(nose[0] - eye_mid_x),
        max(1e-3, ied / 2),
    ))
    pitch = math.degrees(math.atan2(
        float(nose[1] - eye_mid_y),
        max(1e-3, ied / 2),
    ))
    out.update({
        "face_detected": True,
        "face_bbox_area_ratio": bbox_area / max(1.0, crop_area),
        "inter_eye_distance_ratio": ied / max(1.0, float(w)),
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "roll_deg": roll,
    })
    return out


def run_pass_geometry(frames_df, *, run_dir: Path) -> dict:
    """Pass 2: per-frame face geometry + per-group aggregates + KS pairs."""
    import json
    import numpy as np
    import pandas as pd
    from PIL import Image
    rows = []
    for _, row in frames_df.iterrows():
        path = Path(row["frame_path"])
        try:
            with Image.open(path) as im:
                img = np.array(im.convert("RGB"))
            g = _face_geometry_from_array(img)
        except Exception as exc:
            logger.warning("geometry failed for %s: %s", path, exc)
            continue
        g.update({k: row[k] for k in ("group", "label", "video_id", "frame_path")})
        rows.append(g)
    if not rows:
        out = {"per_group": {}, "ks": {}, "feature_cols": [], "n_frames": 0}
        (run_dir / "stats_pass_geometry.json").write_text(json.dumps(out, indent=2))
        return out
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns
                    if c not in ("group", "label", "video_id", "frame_path",
                                 "face_detected")]
    agg = _aggregate_per_group(df, feature_cols)
    # Layer in face_detected_rate per group.
    for g in agg:
        sub = df[df["group"] == g]
        agg[g]["face_detected_rate"] = float(sub["face_detected"].mean())
    ks = _cross_pool_ks(df, feature_cols)
    out = {"per_group": agg, "ks": ks, "feature_cols": feature_cols,
           "n_frames": int(len(df))}
    (run_dir / "stats_pass_geometry.json").write_text(json.dumps(out, indent=2))
    df.to_parquet(run_dir / "raw_geometry.parquet", index=False)
    return out


# === Section 5: Pass 3 — model embeddings + confidence ======================

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _preprocess_image(img_rgb):
    """Resize to 224x224 and CLIP-normalize. Returns (3, 224, 224) float tensor.

    Matches the preprocessing used in data/batching/combined_paired.py
    (CLIP mean/std, not ImageNet).
    """
    import cv2
    import numpy as np
    import torch
    img = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    for c in range(3):
        arr[..., c] = (arr[..., c] - _CLIP_MEAN[c]) / _CLIP_STD[c]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _download_checkpoint_if_gcs(uri: str, cache_dir: Path) -> Path:
    """If uri is a gs:// path, download to cache_dir and return local path."""
    if not uri.startswith("gs://"):
        return Path(uri)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / Path(uri).name
    if local.exists() and local.stat().st_size > 1_000_000:
        logger.info("checkpoint cache hit: %s", local)
        return local
    bucket_name, blob_name = _parse_gs_uri(uri)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    logger.info("downloading checkpoint: %s", uri)
    bucket.blob(blob_name).download_to_filename(str(local))
    logger.info("checkpoint downloaded: %s (%.1f MB)",
                local, local.stat().st_size / (1024 * 1024))
    return local


def _load_state_dict_into_model(model, checkpoint_data, saved_config):
    """Inlined copy of retro_score_value_composite._load_state_dict_into_model.

    Avoids importing retro_score, which transitively imports `data.sources`
    and pulls in `torchdata` (not installed on Mac dev hosts).
    """
    from collections import OrderedDict
    if isinstance(checkpoint_data, dict) and "state_dict" in checkpoint_data:
        state_dict = checkpoint_data["state_dict"]
    else:
        state_dict = checkpoint_data

    if saved_config.get("use_arcface_head", False) and "current_arcface_s" in saved_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            current_s = saved_config["current_arcface_s"]
            model.head.s.data.fill_(current_s)
            logger.info("Restored ArcFace s parameter: %s", current_s)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s%s", len(missing), missing[:5],
                       "..." if len(missing) > 5 else "")
    if unexpected:
        logger.warning("Unexpected keys (%d): %s%s", len(unexpected),
                       unexpected[:5], "..." if len(unexpected) > 5 else "")


def _load_model_for_inference(ckpt_local: Path):
    """Load checkpoint using the embedded config + DETECTOR registry.

    Returns (model, embedding_fn). embedding_fn takes a (B, 3, 224, 224) tensor
    and returns (embeddings_np, fake_probs_np).
    """
    import torch
    import sys as _sys
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in _sys.path:
        _sys.path.insert(0, str(repo_root))
    from detectors import DETECTOR

    saved = torch.load(ckpt_local, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or "model_config" not in saved:
        raise RuntimeError(
            f"Checkpoint {ckpt_local} has no embedded `model_config`. "
            "This diagnostic requires checkpoints saved by the live training "
            "pipeline (which embeds the model_config dict)."
        )
    saved_cfg = saved["model_config"]
    if "model_name" not in saved_cfg:
        raise RuntimeError(
            f"Checkpoint model_config missing `model_name`. Keys present: "
            f"{sorted(saved_cfg.keys())[:20]}"
        )
    logger.info("Instantiating detector: %s", saved_cfg["model_name"])
    model = DETECTOR[saved_cfg["model_name"]](saved_cfg)
    _load_state_dict_into_model(model, saved, saved_cfg)
    model.eval()

    @torch.no_grad()
    def embedding_fn(batch):
        out = model({"image": batch}, inference=True)
        feat = out["feat"].cpu().numpy()
        prob = out["prob"].cpu().numpy()
        return feat, prob

    return model, embedding_fn


def run_pass_model(
    frames_df, *, run_dir: Path, checkpoint_uri: str, cache_dir: Path,
    batch_size: int = 16,
) -> dict:
    """Pass 3: load model, score every frame, write embeddings + probs."""
    import json
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    ckpt_local = _download_checkpoint_if_gcs(
        checkpoint_uri, cache_dir / "checkpoints",
    )
    _, embedding_fn = _load_model_for_inference(ckpt_local)

    paths = frames_df["frame_path"].tolist()
    n = len(paths)
    embeddings_list: list = []
    probs_list: list = []
    valid_idx: list[int] = []
    batch: list = []
    batch_idx: list[int] = []

    def _flush_batch():
        if not batch:
            return
        stacked = torch.stack(batch, dim=0)
        emb, pb = embedding_fn(stacked)
        embeddings_list.append(emb)
        probs_list.append(pb)
        valid_idx.extend(batch_idx)
        batch.clear()
        batch_idx.clear()

    for i, path in enumerate(paths):
        try:
            with Image.open(path) as im:
                img = np.array(im.convert("RGB"))
            t = _preprocess_image(img)
        except Exception as exc:
            logger.warning("preprocess failed for %s: %s", path, exc)
            continue
        batch.append(t)
        batch_idx.append(i)
        if len(batch) == batch_size:
            _flush_batch()
            if (len(valid_idx) // batch_size) % 10 == 0:
                logger.info("model pass: %d/%d frames scored", len(valid_idx), n)
    _flush_batch()

    if not embeddings_list:
        out = {"per_group": {}, "centroid_distances": {}, "n_frames": 0,
               "embedding_dim": 0}
        (run_dir / "stats_pass_model.json").write_text(json.dumps(out, indent=2))
        return out

    emb_arr = np.concatenate(embeddings_list, axis=0)
    prob_arr = np.concatenate(probs_list, axis=0)
    df = frames_df.iloc[valid_idx].reset_index(drop=True).copy()
    df["fake_prob"] = prob_arr
    for d in range(emb_arr.shape[1]):
        df[f"emb_{d:03d}"] = emb_arr[:, d]
    df.to_parquet(run_dir / "raw_scores.parquet", index=False)

    per_group: dict[str, dict] = {}
    centroids: dict[str, np.ndarray] = {}
    for g in sorted(df["group"].unique()):
        sub = df[df["group"] == g]
        emb = sub[[c for c in df.columns if c.startswith("emb_")]].to_numpy()
        centroids[g] = emb.mean(axis=0)
        per_group[g] = {
            "fake_prob_mean": float(sub["fake_prob"].mean()),
            "fake_prob_median": float(sub["fake_prob"].median()),
            "fake_prob_p05": float(sub["fake_prob"].quantile(0.05)),
            "fake_prob_p95": float(sub["fake_prob"].quantile(0.95)),
            "embedding_dispersion_l2": float(
                np.linalg.norm(emb - centroids[g], axis=1).mean()
            ),
            "n": int(len(sub)),
        }
    centroid_distances: dict[str, float] = {}
    keys = sorted(centroids.keys())
    for i, gi in enumerate(keys):
        for gj in keys[i + 1:]:
            ci, cj = centroids[gi], centroids[gj]
            denom = float(np.linalg.norm(ci) * np.linalg.norm(cj) + 1e-12)
            cos = float(np.dot(ci, cj) / denom)
            centroid_distances[f"{gi}__{gj}"] = 1.0 - cos
    out = {
        "per_group": per_group,
        "centroid_distances": centroid_distances,
        "n_frames": int(len(df)),
        "embedding_dim": int(emb_arr.shape[1]),
    }
    (run_dir / "stats_pass_model.json").write_text(json.dumps(out, indent=2))
    return out


# === Section 6: Pass 4 — hard-sample selection ==============================


def _select_hard_samples(scored_df, *, per_bucket: int = 30, seed: int = 737):
    """Select 3 buckets of frames per group: wrong, uncertain, right.

    `scored_df` must contain columns: group, label, video_id, frame_path, fake_prob.
    Returns a DataFrame with an added `bucket` column.
    """
    import numpy as np
    import pandas as pd
    out_rows = []
    rng = np.random.default_rng(seed)
    for g in sorted(scored_df["group"].unique()):
        sub = scored_df[scored_df["group"] == g].copy()
        if sub.empty:
            continue
        label = sub["label"].iloc[0]
        target = 0.0 if label == "real" else 1.0
        sub["err"] = (sub["fake_prob"] - target).abs()
        wrong = (sub[sub["err"] > 0.9]
                 .sort_values("err", ascending=False)
                 .head(per_bucket)
                 .assign(bucket="confidently_wrong"))
        uncert_pool = sub[(sub["fake_prob"] >= 0.4) & (sub["fake_prob"] <= 0.6)].copy()
        if not uncert_pool.empty:
            uncert_pool["dist_to_half"] = (uncert_pool["fake_prob"] - 0.5).abs()
            uncert = (uncert_pool
                      .sort_values("dist_to_half")
                      .head(per_bucket)
                      .drop(columns="dist_to_half")
                      .assign(bucket="uncertain"))
        else:
            uncert = sub.head(0).assign(bucket="uncertain")
        right_pool = sub[sub["err"] < 0.1]
        if len(right_pool) > per_bucket:
            ix = rng.choice(len(right_pool), size=per_bucket, replace=False)
            right = right_pool.iloc[sorted(ix)]
        else:
            right = right_pool
        right = right.assign(bucket="confidently_right")
        out_rows.append(pd.concat([wrong, uncert, right], ignore_index=True))
    if not out_rows:
        return pd.DataFrame()
    result = pd.concat(out_rows, ignore_index=True)
    if "err" in result.columns:
        result = result.drop(columns=["err"])
    return result


def run_pass_gallery(scored_df, *, run_dir: Path, per_bucket: int = 30) -> dict:
    """Pass 4: select hard samples and copy thumbnails into run_dir/thumbnails/."""
    import json
    from PIL import Image
    if scored_df is None or len(scored_df) == 0:
        out = {"selected": 0, "rows": [], "per_bucket": per_bucket,
               "note": "no scored frames — model pass skipped or failed"}
        (run_dir / "stats_pass_gallery.json").write_text(json.dumps(out, indent=2))
        return out
    sel = _select_hard_samples(scored_df, per_bucket=per_bucket, seed=DEFAULT_SEED)
    thumb_dir = run_dir / "thumbnails"
    thumb_dir.mkdir(exist_ok=True)
    rows = []
    for _, row in sel.iterrows():
        src = Path(row["frame_path"])
        dst_dir = thumb_dir / row["group"] / row["bucket"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{row['video_id']}__{src.stem}.jpg"
        try:
            with Image.open(src) as im:
                im.thumbnail((224, 224))
                im.convert("RGB").save(dst, format="JPEG", quality=85)
        except Exception as exc:
            logger.warning("thumbnail failed for %s: %s", src, exc)
            continue
        rows.append({
            "group": row["group"], "label": row["label"], "bucket": row["bucket"],
            "video_id": row["video_id"], "fake_prob": float(row["fake_prob"]),
            "thumbnail_rel_path": str(dst.relative_to(run_dir)),
        })
    out = {"selected": len(rows), "rows": rows, "per_bucket": per_bucket}
    (run_dir / "stats_pass_gallery.json").write_text(json.dumps(out, indent=2))
    return out


# === Section 7: Pass 5 — HTML report ========================================


def _ks_summary_table(stats_section: dict, geometry_section: dict) -> str:
    rows: list[tuple[str, str, str, float]] = []  # (pass, pair, feature, ks)
    for source_name, section in (("stats", stats_section),
                                 ("geometry", geometry_section)):
        for pair, ks_dict in (section or {}).get("ks", {}).items():
            for feat, ks in ks_dict.items():
                if ks != ks:  # NaN
                    continue
                rows.append((source_name, pair, feat, float(ks)))
    rows.sort(key=lambda r: r[3], reverse=True)
    if not rows:
        return "<p><em>No KS distances computed (no overlapping data).</em></p>"
    lines = ['<table class="summary"><thead><tr>'
             '<th>Pass</th><th>Pair</th><th>Feature</th><th>KS distance</th>'
             '</tr></thead><tbody>']
    for src, pair, feat, ks in rows:
        cls = ' class="hi"' if ks > 0.2 else ""
        lines.append(f'<tr{cls}><td>{src}</td><td>{pair}</td>'
                     f'<td>{feat}</td><td>{ks:.3f}</td></tr>')
    lines.append("</tbody></table>")
    return "\n".join(lines)


def _per_group_means_table(section: dict, title: str,
                           feature_cols: list[str] | None = None) -> str:
    pg = (section or {}).get("per_group", {})
    if not pg:
        return f"<p><em>No data for {title}.</em></p>"
    feats = feature_cols or (section or {}).get("feature_cols", [])
    if not feats:
        return f"<p><em>No features for {title}.</em></p>"
    head = "<th>Group</th>" + "".join(f"<th>{f}</th>" for f in feats)
    body = []
    for g, stats in pg.items():
        cells = [f"<td>{g}</td>"]
        for f in feats:
            entry = stats.get(f)
            if isinstance(entry, dict):
                v = entry.get("mean")
            elif isinstance(entry, (int, float)):
                v = entry
            else:
                v = None
            cells.append(f"<td>{v:.3f}</td>" if isinstance(v, (int, float)) else "<td>-</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f"<h3>{title}</h3>"
            f"<table><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>")


def _gallery_section_html(gallery_section: dict) -> str:
    rows = (gallery_section or {}).get("rows", [])
    if not rows:
        return "<p><em>No gallery data — model pass was skipped or empty.</em></p>"
    by_group: dict[str, dict[str, list]] = {}
    for r in rows:
        by_group.setdefault(r["group"], {}).setdefault(r["bucket"], []).append(r)
    parts = []
    for grp in sorted(by_group):
        parts.append(f"<h3>{grp}</h3>")
        for bucket in ("confidently_wrong", "uncertain", "confidently_right"):
            items = by_group[grp].get(bucket, [])
            parts.append(f"<h4>{bucket} ({len(items)} frames)</h4>")
            if not items:
                parts.append('<p><em>(no frames in this bucket)</em></p>')
                continue
            parts.append('<div class="thumbs">')
            for r in items:
                rel = r["thumbnail_rel_path"]
                parts.append(
                    f'<figure><img src="{rel}" alt="{grp}/{bucket}">'
                    f'<figcaption>p={r["fake_prob"]:.2f}<br>'
                    f'{r["video_id"]}</figcaption></figure>'
                )
            parts.append("</div>")
    return "\n".join(parts)


def build_html_report(
    *,
    run_dir: Path,
    stats_section: dict,
    geometry_section: dict,
    model_section: dict,
    gallery_section: dict,
) -> Path:
    """Compose the single-file HTML report. Returns the written path."""
    css = """
    body { font-family: -apple-system, system-ui, sans-serif; max-width: 1200px;
           margin: 2em auto; padding: 0 1em; color: #222; }
    h1 { border-bottom: 2px solid #888; padding-bottom: 0.3em; }
    h2 { margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }
    table { border-collapse: collapse; margin: 1em 0; font-size: 0.9em; }
    th, td { border: 1px solid #ccc; padding: 0.3em 0.6em; text-align: left; }
    th { background: #f4f4f4; }
    tr.hi td { background: #fff4e0; font-weight: bold; }
    .thumbs { display: flex; flex-wrap: wrap; gap: 0.5em; margin: 0.5em 0 1.5em; }
    .thumbs figure { margin: 0; padding: 0; text-align: center; }
    .thumbs img { width: 112px; height: 112px; object-fit: cover; border: 1px solid #ddd; }
    .thumbs figcaption { font-size: 0.7em; color: #555; max-width: 112px; }
    """
    n_stats = (stats_section or {}).get("n_frames", 0)
    n_geom = (geometry_section or {}).get("n_frames", 0)
    n_model = (model_section or {}).get("n_frames", 0)
    centroid_html = ""
    cd = (model_section or {}).get("centroid_distances", {})
    if cd:
        centroid_html = "<h3>Centroid cosine distances (1 - cos)</h3><table><tbody>" + "".join(
            f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in sorted(cd.items())
        ) + "</tbody></table>"
    model_summary_html = _per_group_means_table(
        {"per_group": (model_section or {}).get("per_group", {})},
        "Model fake-probability + embedding dispersion",
        feature_cols=["fake_prob_mean", "fake_prob_median",
                      "fake_prob_p05", "fake_prob_p95",
                      "embedding_dispersion_l2"],
    )
    body = f"""
    <h1>Teams pool comparison: teams_ood vs proper_visomaster_teams</h1>
    <p>Run dir: <code>{run_dir}</code></p>
    <p>Frames analyzed — stats: {n_stats}, geometry: {n_geom}, model: {n_model}.</p>

    <h2>Executive summary</h2>
    <p>KS distances across cross-pool same-label pairs. Rows highlighted in
    orange exceed 0.2 (a notable distributional shift).</p>
    {_ks_summary_table(stats_section, geometry_section)}

    <h2>Image stats</h2>
    {_per_group_means_table(stats_section, "Image stats — per-group means")}

    <h2>Face geometry</h2>
    {_per_group_means_table(geometry_section, "Face geometry — per-group means")}

    <h2>Model section</h2>
    {model_summary_html}
    {centroid_html}

    <h2>Failures gallery</h2>
    {_gallery_section_html(gallery_section)}
    """
    html = (f"<!doctype html><meta charset='utf-8'>"
            f"<title>teams pool diff</title>"
            f"<style>{css}</style>{body}")
    out = run_dir / "report.html"
    out.write_text(html)
    return out


# === Section 8: main() ======================================================


def main(argv: list[str] | None = None) -> int:
    import json
    args = _build_arg_parser().parse_args(argv)
    _configure_logging(args.log_level)
    run_dir = _make_run_dir(args.output_dir)
    logger.info("Run dir: %s", run_dir)
    logger.info("Skipped passes: %s", args.skip_pass or "(none)")

    frames_df = collect_frames(
        videos_per_group=args.videos_per_group,
        frames_per_video=args.frames_per_video,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    n_groups = frames_df["group"].nunique() if len(frames_df) else 0
    logger.info("collected %d frames across %d groups", len(frames_df), n_groups)

    stats_section: dict = {}
    geometry_section: dict = {}
    model_section: dict = {}
    gallery_section: dict = {}

    if "stats" not in args.skip_pass and len(frames_df):
        try:
            stats_section = run_pass_stats(frames_df, run_dir=run_dir)
        except Exception as exc:
            logger.exception("stats pass failed: %s", exc)
            stats_section = {"error": str(exc)}

    if "geometry" not in args.skip_pass and len(frames_df):
        try:
            geometry_section = run_pass_geometry(frames_df, run_dir=run_dir)
        except Exception as exc:
            logger.exception("geometry pass failed: %s", exc)
            geometry_section = {"error": str(exc)}

    scored_df = None
    if "model" not in args.skip_pass and len(frames_df):
        try:
            model_section = run_pass_model(
                frames_df, run_dir=run_dir, checkpoint_uri=args.checkpoint,
                cache_dir=args.cache_dir,
            )
            import pandas as pd
            scored_path = run_dir / "raw_scores.parquet"
            if scored_path.exists():
                scored_df = pd.read_parquet(scored_path)
        except Exception as exc:
            logger.exception("model pass failed: %s", exc)
            model_section = {"error": str(exc)}

    if "gallery" not in args.skip_pass and scored_df is not None:
        try:
            gallery_section = run_pass_gallery(scored_df, run_dir=run_dir)
        except Exception as exc:
            logger.exception("gallery pass failed: %s", exc)
            gallery_section = {"error": str(exc)}

    html_path = build_html_report(
        run_dir=run_dir,
        stats_section=stats_section,
        geometry_section=geometry_section,
        model_section=model_section,
        gallery_section=gallery_section,
    )
    summary = {
        "run_dir": str(run_dir),
        "n_frames_collected": int(len(frames_df)),
        "report_path": str(html_path),
        "skipped_passes": list(args.skip_pass),
        "checkpoint": args.checkpoint,
    }
    (run_dir / "stats.json").write_text(json.dumps(summary, indent=2))
    logger.info("done — report at %s", html_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
