#!/usr/bin/env python3
"""
Build a diverse validation manifest for quantization error measurement.

Purpose:
    Compare FP32 vs quantized model outputs on a fixed, reproducible set of
    frames drawn from the SAME sources used in production validation
    (validate_custom_sources.py).

Data sources (mirrors R3 / B16 / L14 validation):
    1. DF40 Paired — target→source + source→target (17 methods + ff++ real)
       Bucket: gs://df40-frames-recropped-rfa85/
    2. DeepLiveCam — edge_cases, minimal_processing, quality_enhancement
       Bucket: gs://live-deepfake-methods-real-and-fake-frames-cropped/
    3. VisoMaster — 9 swap models + real pairs + tier enrichment
       Bucket: gs://live-deepfake-methods-real-and-fake-frames-cropped/
    4. External Real — YouTube AVSpeech
       Bucket: gs://effort-collected-data/real/external_youtube_avspeech/

Design:
    ~4,700 frames total, balanced sampling from each source.
    Deterministic (SEED=42) so the manifest is reproducible.
    Split into two non-overlapping subsets (by video_id):
      - calibration (~10%): used to determine quantization scale/zero-point
      - validation  (~90%): used to measure FP32-vs-quantized error

Usage:
    python build_quantization_val_manifest.py [--dry-run] [--output-dir /tmp/quant_val]
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
SEED = 42
GCS_PROJECT = "train-cvit2"

# DF40 paired data
DF40_PAIR_JSON = "dataset/df40_pairs/df40-pair-matching.json"
DF40_BUCKET = "df40-frames-recropped-rfa85"

# DeepLive / VisoMaster cropped frames
DEEPLIVE_BUCKET = "live-deepfake-methods-real-and-fake-frames-cropped"

# VisoMaster full-frames bucket (for tier metadata)
VISOMASTER_FRAMES_BUCKET = "live-deepfake-methods-real-and-fake-frames"

# External real data
EXTERNAL_REAL_BUCKET = "effort-collected-data"
EXTERNAL_REAL_PREFIX = "real/external_youtube_avspeech"

# Frames per video/identity to sample
MAX_FRAMES_PER_VIDEO = 4

# Per-identity frame indices (DeepLive/VisoMaster have 16 frames: 0-15)
DEEPLIVE_FRAME_INDICES = list(range(16))

# ─── DF40 Paired ──────────────────────────────────────────────────────────────

def build_df40_entries(pair_json_path: str, max_videos_per_method: int = 40) -> list[dict]:
    """Sample from DF40 paired validation data (both orientations).

    Reads the pair-matching JSON, samples videos per method, and builds
    frame entries for both real and fake sides of each pair.
    """
    with open(pair_json_path) as f:
        data = json.load(f)

    pairs = data.get("pairs", [])
    methods = data.get("methods", [])
    method_orientation = data.get("method_orientation", {})

    # The JSON stores bucket as "gs://df40-frames-recropped-rfa85" —
    # normalise to just the bucket name for the 'bucket' column.
    raw_bucket = data.get("bucket", DF40_BUCKET)
    bucket_name = raw_bucket.replace("gs://", "").strip("/")

    logger.info("  DF40 pairs JSON: %d pairs across %d methods", len(pairs), len(methods))

    # Group pairs by method
    pairs_by_method = defaultdict(list)
    for pair in pairs:
        pairs_by_method[pair["method"]].append(pair)

    entries = []
    random.seed(SEED)

    for method in sorted(methods):
        method_pairs = pairs_by_method.get(method, [])
        if not method_pairs:
            logger.warning("  DF40 method '%s' has no pairs, skipping", method)
            continue

        # Sample videos
        n_sample = min(max_videos_per_method, len(method_pairs))
        sampled = random.sample(method_pairs, n_sample)
        orientation = method_orientation.get(method, "source_target")

        for pair in sampled:
            video_id = pair["pair_id"]
            fake_info = pair["fake"]  # {path, frame_count, frames}
            real_info = pair["real"]  # {identity, source, path, frame_count, frames}

            fake_base = fake_info["path"].rstrip("/")
            real_base = real_info["path"].rstrip("/")
            fake_frames = fake_info["frames"]
            real_frames = real_info["frames"]

            # The pair JSON paths are full gs:// URIs already, e.g.:
            #   gs://df40-frames-recropped-rfa85/fake/blendface/001_870/
            # We need to split into gcs_uri (full) and path (object key only).
            bucket_prefix = f"gs://{bucket_name}/"

            # Sample frames (max MAX_FRAMES_PER_VIDEO, limited by whichever side has fewer)
            n_available = min(len(fake_frames), len(real_frames))
            if n_available == 0:
                continue
            n_frames = min(MAX_FRAMES_PER_VIDEO, n_available)
            frame_indices = sorted(random.sample(range(n_available), n_frames))

            for idx in frame_indices:
                fake_gcs_uri = f"{fake_base}/{fake_frames[idx]}"
                real_gcs_uri = f"{real_base}/{real_frames[idx]}"

                # Strip gs://bucket/ prefix to get the object key
                fake_obj_key = fake_gcs_uri.replace(bucket_prefix, "", 1) if fake_gcs_uri.startswith(bucket_prefix) else fake_gcs_uri
                real_obj_key = real_gcs_uri.replace(bucket_prefix, "", 1) if real_gcs_uri.startswith(bucket_prefix) else real_gcs_uri

                # Fake frame
                entries.append({
                    "gcs_uri": fake_gcs_uri,
                    "bucket": bucket_name,
                    "path": fake_obj_key,
                    "label": "fake",
                    "method": method,
                    "data_source": "df40_paired",
                    "video_id": video_id,
                    "orientation": orientation,
                    "diversity_category": f"df40_{method}",
                })

                # Paired real frame
                entries.append({
                    "gcs_uri": real_gcs_uri,
                    "bucket": bucket_name,
                    "path": real_obj_key,
                    "label": "real",
                    "method": "faceforensics++",
                    "data_source": "df40_paired",
                    "video_id": f"real_{video_id}",
                    "orientation": orientation,
                    "diversity_category": "df40_real_ff",
                })

        logger.info("  DF40 method '%s': %d videos × %d frames (%s)",
                     method, n_sample, MAX_FRAMES_PER_VIDEO, orientation)

    return entries


# ─── DeepLive ─────────────────────────────────────────────────────────────────

# Known identity counts per strategy (from GCS exploration)
DEEPLIVE_STRATEGY_COUNTS = {
    "edge_cases":           (434, 4),   # 4-digit zero-padded: edge_cases_0000
    "minimal_processing":   (425, 4),
    "quality_enhancement":  (320, 4),
}

def build_deeplive_entries(
    bucket_name: str,
    strategies: list[str],
    max_identities_per_strategy: int = 30,
    frames_per_identity: int = 3,
) -> list[dict]:
    """Build entries from DeepLiveCam paired data using deterministic paths.

    Each identity has real + fake frames at:
      gs://{bucket}/samples/{identity_id}/frames/{real|fake}/frame_XXXX.png
    """
    entries = []
    random.seed(SEED + 1)  # Different seed offset for variety

    for strategy in strategies:
        total_available, digit_width = DEEPLIVE_STRATEGY_COUNTS.get(strategy, (0, 4))
        if total_available == 0:
            logger.warning("  Unknown DeepLive strategy '%s', skipping", strategy)
            continue

        n_sample = min(max_identities_per_strategy, total_available)
        sampled_indices = sorted(random.sample(range(total_available), n_sample))

        for idx in sampled_indices:
            identity_id = f"{strategy}_{idx:0{digit_width}d}"

            # Sample frame indices
            chosen_frames = sorted(random.sample(DEEPLIVE_FRAME_INDICES,
                                                  min(frames_per_identity, len(DEEPLIVE_FRAME_INDICES))))

            for frame_idx in chosen_frames:
                fake_key = f"samples/{identity_id}/frames/fake/frame_{frame_idx:04d}.png"
                real_key = f"samples/{identity_id}/frames/real/frame_{frame_idx:04d}.png"

                # Fake
                entries.append({
                    "gcs_uri": f"gs://{bucket_name}/{fake_key}",
                    "bucket": bucket_name,
                    "path": fake_key,
                    "label": "fake",
                    "method": f"deeplive_{strategy}",
                    "data_source": "deeplive",
                    "video_id": identity_id,
                    "diversity_category": f"deeplive_{strategy}",
                })
                # Real (paired)
                entries.append({
                    "gcs_uri": f"gs://{bucket_name}/{real_key}",
                    "bucket": bucket_name,
                    "path": real_key,
                    "label": "real",
                    "method": f"deeplive_{strategy}",
                    "data_source": "deeplive",
                    "video_id": identity_id,
                    "diversity_category": f"deeplive_{strategy}_real",
                })

        logger.info("  DeepLive '%s': %d identities × %d frames (total avail: %d)",
                     strategy, n_sample, frames_per_identity, total_available)

    return entries


# ─── VisoMaster ───────────────────────────────────────────────────────────────

VISOMASTER_MODEL_COUNTS = {
    "CSCS":                  (601, 5),  # 5-digit: visomaster_CSCS_00000
    "GhostFace-v1":          (602, 5),
    "GhostFace-v2":          (601, 5),
    "GhostFace-v3":          (602, 5),
    "InStyleSwapper256-A":   (601, 5),
    "InStyleSwapper256-B":   (648, 5),
    "InStyleSwapper256-C":   (675, 5),
    "Inswapper128":          (575, 5),
    "SimSwap512":            (576, 5),
}

def build_visomaster_entries(
    bucket_name: str,
    swap_models: list[str] | None = None,
    max_identities_per_model: int = 10,
    frames_per_identity: int = 3,
) -> list[dict]:
    """Build entries from VisoMaster paired data using deterministic paths.

    Same frame layout as DeepLive:
      gs://{bucket}/samples/visomaster_{model}_{NNNNN}/frames/{real|fake}/frame_XXXX.png
    """
    if swap_models is None:
        swap_models = list(VISOMASTER_MODEL_COUNTS.keys())

    entries = []
    random.seed(SEED + 2)  # Different seed offset

    for model in swap_models:
        total_available, digit_width = VISOMASTER_MODEL_COUNTS.get(model, (0, 5))
        if total_available == 0:
            logger.warning("  Unknown VisoMaster model '%s', skipping", model)
            continue

        n_sample = min(max_identities_per_model, total_available)
        sampled_indices = sorted(random.sample(range(total_available), n_sample))

        for idx in sampled_indices:
            identity_id = f"visomaster_{model}_{idx:0{digit_width}d}"

            chosen_frames = sorted(random.sample(DEEPLIVE_FRAME_INDICES,
                                                  min(frames_per_identity, len(DEEPLIVE_FRAME_INDICES))))

            for frame_idx in chosen_frames:
                fake_key = f"samples/{identity_id}/frames/fake/frame_{frame_idx:04d}.png"
                real_key = f"samples/{identity_id}/frames/real/frame_{frame_idx:04d}.png"

                # Fake
                entries.append({
                    "gcs_uri": f"gs://{bucket_name}/{fake_key}",
                    "bucket": bucket_name,
                    "path": fake_key,
                    "label": "fake",
                    "method": f"visomaster_{model}",
                    "data_source": "visomaster",
                    "video_id": identity_id,
                    "diversity_category": f"visomaster_{model}",
                })
                # Real (paired)
                entries.append({
                    "gcs_uri": f"gs://{bucket_name}/{real_key}",
                    "bucket": bucket_name,
                    "path": real_key,
                    "label": "real",
                    "method": "visomaster_real",
                    "data_source": "visomaster",
                    "video_id": identity_id,
                    "diversity_category": "visomaster_real",
                })

        logger.info("  VisoMaster '%s': %d identities × %d frames (total avail: %d)",
                     model, n_sample, frames_per_identity, total_available)

    return entries


# ─── External Real (YouTube AVSpeech) ─────────────────────────────────────────

def build_external_real_entries(
    bucket_name: str,
    prefix: str,
    max_videos: int = 100,
    frames_per_video: int = 4,
) -> list[dict]:
    """Sample external real videos from GCS.

    Structure: gs://{bucket}/{prefix}/{video_id}/{frame}.png
    """
    client = storage.Client(project=GCS_PROJECT)
    bucket = client.bucket(bucket_name)

    logger.info("  Listing external real videos from gs://%s/%s ...", bucket_name, prefix)

    # Group blobs by video folder
    videos = defaultdict(list)
    for blob in bucket.list_blobs(prefix=prefix + "/"):
        parts = blob.name.split("/")
        if len(parts) >= 4 and blob.name.lower().endswith((".png", ".jpg", ".jpeg")):
            video_id = parts[-2]  # folder name = video_id
            videos[video_id].append(blob.name)

    logger.info("  Found %d videos with frames", len(videos))

    # Sample videos
    random.seed(SEED + 3)
    video_ids = sorted(videos.keys())
    n_sample = min(max_videos, len(video_ids))
    sampled_video_ids = random.sample(video_ids, n_sample)

    entries = []
    for vid_id in sampled_video_ids:
        frames = sorted(videos[vid_id])
        n_frames = min(frames_per_video, len(frames))
        chosen = random.sample(frames, n_frames)

        for blob_name in chosen:
            entries.append({
                "gcs_uri": f"gs://{bucket_name}/{blob_name}",
                "bucket": bucket_name,
                "path": blob_name,
                "label": "real",
                "method": "external_youtube_avspeech",
                "data_source": "external_real",
                "video_id": vid_id,
                "diversity_category": "external_youtube_avspeech",
            })

    logger.info("  External real: %d videos × up to %d frames = %d entries",
                 n_sample, frames_per_video, len(entries))
    return entries


# ─── Calibration / Validation Split ───────────────────────────────────────────

def split_calibration_validation(
    df: pd.DataFrame,
    calibration_fraction: float = 0.10,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split manifest into calibration and validation sets BY VIDEO.

    The split is stratified per diversity_category so every category is
    represented in both subsets.  All frames from a given video go to
    exactly one subset (no video leakage).

    Calibration data is used during quantization (PTQ) to determine optimal
    scale and zero-point per tensor.  It must NOT overlap with the
    validation data used to measure quantization error.

    Args:
        df: Full manifest DataFrame.
        calibration_fraction: Fraction of videos to assign to calibration
            (default 0.10 — typically 200-500 frames is plenty for PTQ).
        seed: Random seed for reproducibility.

    Returns:
        (calibration_df, validation_df) — non-overlapping DataFrames.
    """
    rng = random.Random(seed + 100)  # offset from other seeds

    cal_indices = []
    val_indices = []

    for cat, group in df.groupby("diversity_category"):
        # Get unique videos in this category
        video_ids = sorted(group["video_id"].unique())
        n_cal = max(1, round(len(video_ids) * calibration_fraction))

        rng.shuffle(video_ids)
        cal_videos = set(video_ids[:n_cal])
        val_videos = set(video_ids[n_cal:])

        cal_mask = group["video_id"].isin(cal_videos)
        cal_indices.extend(group.index[cal_mask].tolist())
        val_indices.extend(group.index[~cal_mask].tolist())

    cal_df = df.loc[cal_indices].reset_index(drop=True)
    val_df = df.loc[val_indices].reset_index(drop=True)

    # Sanity: no video overlap
    overlap = set(cal_df["video_id"]) & set(val_df["video_id"])
    if overlap:
        # Paired data (DF40) can have real_<pair_id> and <pair_id> for same
        # identity — that's OK (different label), but same video_id shouldn't
        # appear in both splits.
        logger.warning("Video overlap between cal/val: %d (may be paired real/fake)", len(overlap))

    logger.info("\n── Calibration / Validation Split ──")
    logger.info("  Calibration: %d frames, %d videos", len(cal_df), cal_df["video_id"].nunique())
    logger.info("  Validation:  %d frames, %d videos", len(val_df), val_df["video_id"].nunique())
    logger.info("  Cal label dist: %s", cal_df["label"].value_counts().to_dict())
    logger.info("  Val label dist: %s", val_df["label"].value_counts().to_dict())
    logger.info("  Cal categories: %d / %d",
                 cal_df["diversity_category"].nunique(),
                 df["diversity_category"].nunique())

    return cal_df, val_df


# ─── Main ─────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, title: str = "QUANTIZATION VALIDATION MANIFEST SUMMARY") -> dict:
    """Print and return a summary of the manifest."""
    summary = {
        "total_frames": len(df),
        "unique_videos": int(df["video_id"].nunique()),
        "label_distribution": df["label"].value_counts().to_dict(),
        "data_source_distribution": df["data_source"].value_counts().to_dict(),
        "diversity_categories": {},
    }

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Total frames:  {len(df)}")
    print(f"Unique videos: {df['video_id'].nunique()}")

    print(f"\nLabel distribution:")
    for label, count in sorted(df["label"].value_counts().items()):
        print(f"  {label}: {count} ({100*count/len(df):.1f}%)")

    print(f"\nData source distribution:")
    for src, count in sorted(df["data_source"].value_counts().items()):
        print(f"  {src}: {count}")

    print(f"\nDiversity category breakdown:")
    for cat in sorted(df["diversity_category"].unique()):
        sub = df[df["diversity_category"] == cat]
        vids = sub["video_id"].nunique()
        label = sub["label"].iloc[0]
        ds = sub["data_source"].iloc[0]
        print(f"  [{label:4s}] {cat:45s} | {len(sub):4d} frames | {vids:4d} videos | {ds}")
        summary["diversity_categories"][cat] = {
            "frames": len(sub),
            "videos": vids,
            "label": label,
            "data_source": ds,
        }

    print("=" * 80)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Build quantization validation manifest from the same sources used in production validation."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show budget plan without building manifest.")
    parser.add_argument("--no-external-real", action="store_true",
                        help="Skip external real (avoids GCS listing).")
    parser.add_argument("--output-dir", type=str, default="/tmp/quant_val_manifest",
                        help="Output directory for manifest files.")
    parser.add_argument("--upload", action="store_true",
                        help="Upload output to GCS after building.")
    parser.add_argument("--gcs-dest", type=str,
                        default="gs://df40-frames/manifests/quant_val/",
                        help="GCS destination for upload.")
    # Tuning knobs
    parser.add_argument("--df40-videos-per-method", type=int, default=25,
                        help="Max DF40 videos per method (default 25).")
    parser.add_argument("--deeplive-identities-per-strategy", type=int, default=25,
                        help="Max DeepLive identities per strategy (default 25).")
    parser.add_argument("--visomaster-identities-per-model", type=int, default=8,
                        help="Max VisoMaster identities per swap model (default 8).")
    parser.add_argument("--external-real-videos", type=int, default=120,
                        help="Max external real videos (default 120).")
    parser.add_argument("--calibration-fraction", type=float, default=0.10,
                        help="Fraction of videos for calibration split (default 0.10).")

    args = parser.parse_args()

    # ── Resolve paths ──
    script_dir = Path(__file__).parent
    pair_json = script_dir / DF40_PAIR_JSON
    if not pair_json.exists():
        logger.error("DF40 pair JSON not found: %s", pair_json)
        return

    # ── Dry run ──
    if args.dry_run:
        print("=" * 80)
        print("QUANTIZATION VALIDATION MANIFEST — DRY RUN")
        print("=" * 80)
        print()
        print("Data sources (same as validate_custom_sources.py):")
        print()
        print("1. DF40 Paired (target→source + source→target)")
        print(f"   Bucket: gs://{DF40_BUCKET}/")
        print(f"   17 methods × {args.df40_videos_per_method} videos × {MAX_FRAMES_PER_VIDEO} frames")
        est_df40 = 17 * args.df40_videos_per_method * MAX_FRAMES_PER_VIDEO * 2  # ×2 for real+fake
        print(f"   Estimated: ~{est_df40} frames (real + fake)")
        print()
        print("2. DeepLiveCam (edge_cases, minimal_processing, quality_enhancement)")
        print(f"   Bucket: gs://{DEEPLIVE_BUCKET}/")
        est_dl = 3 * args.deeplive_identities_per_strategy * 3 * 2  # 3 strategies, 3 frames, ×2
        print(f"   3 strategies × {args.deeplive_identities_per_strategy} identities × 3 frames × 2 (real+fake)")
        print(f"   Estimated: ~{est_dl} frames")
        print()
        print("3. VisoMaster (9 swap models)")
        print(f"   Bucket: gs://{DEEPLIVE_BUCKET}/")
        est_vm = 9 * args.visomaster_identities_per_model * 3 * 2
        print(f"   9 models × {args.visomaster_identities_per_model} identities × 3 frames × 2 (real+fake)")
        print(f"   Estimated: ~{est_vm} frames")
        print()
        print("4. External Real (YouTube AVSpeech)")
        print(f"   Bucket: gs://{EXTERNAL_REAL_BUCKET}/{EXTERNAL_REAL_PREFIX}/")
        est_er = args.external_real_videos * MAX_FRAMES_PER_VIDEO
        print(f"   {args.external_real_videos} videos × {MAX_FRAMES_PER_VIDEO} frames")
        print(f"   Estimated: ~{est_er} frames")
        print()
        total = est_df40 + est_dl + est_vm + est_er
        print(f"TOTAL ESTIMATED: ~{total} frames")
        cal_est = round(total * args.calibration_fraction)
        print()
        print(f"Calibration / Validation split ({args.calibration_fraction:.0%} by video):")
        print(f"  Calibration (PTQ): ~{cal_est} frames")
        print(f"  Validation (error): ~{total - cal_est} frames")
        return

    # ── Build manifest ──
    all_entries = []

    # 1. DF40 Paired
    logger.info("── 1. DF40 Paired ──")
    df40_entries = build_df40_entries(
        str(pair_json),
        max_videos_per_method=args.df40_videos_per_method,
    )
    all_entries.extend(df40_entries)
    logger.info("  Total DF40: %d entries", len(df40_entries))

    # 2. DeepLive
    logger.info("\n── 2. DeepLiveCam ──")
    deeplive_entries = build_deeplive_entries(
        DEEPLIVE_BUCKET,
        strategies=["edge_cases", "minimal_processing", "quality_enhancement"],
        max_identities_per_strategy=args.deeplive_identities_per_strategy,
        frames_per_identity=3,
    )
    all_entries.extend(deeplive_entries)
    logger.info("  Total DeepLive: %d entries", len(deeplive_entries))

    # 3. VisoMaster
    logger.info("\n── 3. VisoMaster ──")
    visomaster_entries = build_visomaster_entries(
        DEEPLIVE_BUCKET,
        swap_models=None,  # all 9 models
        max_identities_per_model=args.visomaster_identities_per_model,
        frames_per_identity=3,
    )
    all_entries.extend(visomaster_entries)
    logger.info("  Total VisoMaster: %d entries", len(visomaster_entries))

    # 4. External Real
    if not args.no_external_real:
        logger.info("\n── 4. External Real (YouTube AVSpeech) ──")
        external_entries = build_external_real_entries(
            EXTERNAL_REAL_BUCKET,
            EXTERNAL_REAL_PREFIX,
            max_videos=args.external_real_videos,
            frames_per_video=MAX_FRAMES_PER_VIDEO,
        )
        all_entries.extend(external_entries)
        logger.info("  Total External Real: %d entries", len(external_entries))

    # ── Assemble DataFrame ──
    df = pd.DataFrame(all_entries)
    # Ensure consistent types
    for col in ["video_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # ── Calibration / Validation split ──
    cal_df, val_df = split_calibration_validation(
        df, calibration_fraction=args.calibration_fraction,
    )

    # ── Output ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_files = []  # track for upload

    def _save_split(split_df: pd.DataFrame, prefix: str, title: str):
        """Save a split as parquet + CSV + paths JSON + summary JSON."""
        pq = out_dir / f"{prefix}_manifest.parquet"
        split_df.to_parquet(pq, index=False)
        output_files.append(pq)
        logger.info("Saved %s parquet: %s", prefix, pq)

        csv = out_dir / f"{prefix}_manifest.csv"
        split_df.to_csv(csv, index=False)
        output_files.append(csv)

        paths = out_dir / f"{prefix}_paths.json"
        with open(paths, "w") as f:
            json.dump(split_df["path"].tolist(), f, indent=2)
        output_files.append(paths)
        logger.info("Saved %d paths: %s", len(split_df), paths)

        summary = print_summary(split_df, title=title)
        sj = out_dir / f"{prefix}_summary.json"
        with open(sj, "w") as f:
            json.dump(summary, f, indent=2)
        output_files.append(sj)

    # Full manifest (for reference)
    _save_split(df, "quant_full", "FULL MANIFEST (before split)")

    # Calibration split
    _save_split(cal_df, "quant_calibration", "CALIBRATION SET (for PTQ scale/zero-point)")

    # Validation split
    _save_split(val_df, "quant_validation", "VALIDATION SET (for FP32 vs quantized error)")

    # Upload
    if args.upload:
        import subprocess
        gcs_dest = args.gcs_dest.rstrip("/")
        for local_file in output_files:
            dest = f"{gcs_dest}/{local_file.name}"
            logger.info("Uploading %s → %s", local_file, dest)
            subprocess.run(["gsutil", "cp", str(local_file), dest], check=True)
        logger.info("✅ All files uploaded to %s", gcs_dest)


if __name__ == "__main__":
    main()
