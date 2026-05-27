#!/usr/bin/env python3
"""Probe-battery feature extraction.

Reuses the source-discovery + model-load + per-source extraction from
``analysis/feature_space_2026-04-23/extract_features.py``, then concatenates
everything into a single unified ``features.npz`` with per-frame labels
(source_name, provenance, real/fake, source_bucket). That single file is what
``run_linear_probes.py`` consumes.

Only the output layout differs from the parent script — the discovery + model
plumbing is shared so the two scripts stay byte-identical on every axis that
matters.

Usage (inside the training container, on Vertex):
    python3 analysis/probe_battery_2026-04-26/extract_features_for_probes.py \\
        --checkpoint gs://.../value_composite_effort_*.pth \\
        --output_prefix gs://training-job-outputs/probe_battery_2026-04-26/<run_id>/ \\
        --n_per_source 150
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TRAINING_ROOT = HERE.parent.parent
sys.path.insert(0, str(TRAINING_ROOT))
sys.path.insert(0, str(TRAINING_ROOT / "analysis" / "feature_space_2026-04-23"))

from extract_features import (  # noqa: E402  -- shared building blocks
    DEFAULT_MANIFEST,
    DEFAULT_N_PER_SOURCE,
    SEED,
    discover_sources,
    download_checkpoint,
    extract_source,
    load_model,
    upload_file,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("probe-feature-extract")


# 10-class source bucket grouping. Probes train against this label.
# Anything not listed defaults to its own source name (so we never silently
# drop a source); regroup later if the bucket count drifts from 10.
SOURCE_BUCKET_MAP: Dict[str, str] = {
    "deeplive_non_enh_fake":              "deeplive_non_enhanced",
    "deeplive_non_enh_real":              "deeplive_non_enhanced",
    "deeplive_enh_fake":                  "deeplive_enhanced",
    "deeplive_enh_real":                  "deeplive_enhanced",
    "dl_bucket_visomaster_fake":          "dl_bucket_visomaster",
    "dl_bucket_visomaster_real":          "dl_bucket_visomaster",
    "tv2_deeplive_fake":                  "tv2_deeplive",
    "tv2_deeplive_real":                  "tv2_deeplive",
    "tv2_visomaster_fake":                "tv2_visomaster",
    "tv2_visomaster_real":                "tv2_visomaster",
    "proper_visomaster_clean_fake":       "proper_visomaster",
    "proper_visomaster_teams_fake":       "proper_visomaster",
    "proper_real_clean__paired":          "proper_visomaster",
    "proper_real_teams__paired":          "proper_visomaster",
    "proper_visomaster_enhanced_clean_fake": "proper_visomaster_enhanced",
    "proper_visomaster_enhanced_teams_fake": "proper_visomaster_enhanced",
    "visomaster_enhanced_v2_fake":        "visomaster_enhanced_v2",
    "external_vcd_real":                  "external_real",
    "external_youtube_avspeech_real":     "external_real",
    "wma_failure_fake":                   "external_fake",
}


def _bucket_for(source_name: str) -> str:
    return SOURCE_BUCKET_MAP.get(source_name, source_name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="GCS URI of the .pth checkpoint")
    ap.add_argument("--output_prefix", required=True, help="GCS prefix to upload features.npz under")
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

    feat_chunks: List[np.ndarray] = []
    prob_chunks: List[np.ndarray] = []
    source_names: List[str] = []
    provenances: List[str] = []
    labels: List[str] = []
    buckets: List[str] = []
    uris: List[str] = []

    for s in sources:
        if not s.uris:
            logger.warning("Source %s has no URIs — skipping", s.name)
            continue
        logger.info("Processing %s (%d URIs) …", s.name, len(s.uris))
        result = extract_source(model, s.uris, device,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)
        if result["features"].shape[0] == 0:
            logger.warning("Source %s extracted 0 features — skipping", s.name)
            continue
        n = result["features"].shape[0]
        feat_chunks.append(result["features"].astype(np.float32))
        prob_chunks.append(result["probs"].astype(np.float32))
        bucket = _bucket_for(s.name)
        source_names.extend([s.name] * n)
        provenances.extend([s.provenance] * n)
        labels.extend([s.label] * n)
        buckets.extend([bucket] * n)
        kept = [s.uris[int(i)] for i in result["valid_idx"]]
        uris.extend(kept)

    if not feat_chunks:
        raise RuntimeError("No features extracted from any source.")

    features = np.concatenate(feat_chunks, axis=0)
    probs = np.concatenate(prob_chunks, axis=0)

    logger.info("Total frames: %d across %d sources, %d buckets",
                features.shape[0], len(set(source_names)), len(set(buckets)))

    manifest_out = {
        "checkpoint": args.checkpoint,
        "n_per_source": args.n_per_source,
        "seed": args.seed,
        "n_frames_total": int(features.shape[0]),
        "source_bucket_map": SOURCE_BUCKET_MAP,
        "sources": [asdict(s) for s in sources],
        "bucket_counts": {b: int(buckets.count(b)) for b in sorted(set(buckets))},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(manifest_out, f, indent=2)
        manifest_local = f.name
    upload_file(manifest_local, args.output_prefix + "sampling_manifest.json")
    os.unlink(manifest_local)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
        npz_path = tf.name
    np.savez_compressed(
        npz_path,
        features=features,
        probs=probs,
        source_name=np.array(source_names, dtype=object),
        provenance=np.array(provenances, dtype=object),
        label=np.array(labels, dtype=object),
        source_bucket=np.array(buckets, dtype=object),
        uri=np.array(uris, dtype=object),
    )
    upload_file(npz_path, args.output_prefix + "features.npz")
    os.unlink(npz_path)

    logger.info("Done — results at %s", args.output_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
