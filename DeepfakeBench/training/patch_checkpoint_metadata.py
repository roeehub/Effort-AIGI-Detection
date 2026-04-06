#!/usr/bin/env python3
"""
Patch checkpoint metadata for reliable reloads.

Usage examples:
  python patch_checkpoint_metadata.py --checkpoint local.pth
  python patch_checkpoint_metadata.py --checkpoint gs://bucket/old.pth --metadata metadata.yaml --output gs://bucket/new.pth
  python patch_checkpoint_metadata.py --checkpoint local.pth --set backbone.variant=ViT-L-14 --set backbone.source=openai --output fixed.pth
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from google.api_core import exceptions
from google.cloud import storage


def is_gcs_path(path: str) -> bool:
    return path.startswith("gs://")


def split_gcs_path(gcs_path: str) -> Tuple[str, str]:
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {gcs_path}")
    bucket_name = gcs_path.split("gs://", 1)[1].split("/", 1)[0]
    blob_name = gcs_path.split(f"gs://{bucket_name}/", 1)[1]
    return bucket_name, blob_name


def download_from_gcs(gcs_path: str, local_path: str) -> None:
    bucket_name, blob_name = split_gcs_path(gcs_path)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)


def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    bucket_name, blob_name = split_gcs_path(gcs_path)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)


def load_metadata_file(path: str) -> Dict[str, Any]:
    with open(path, "r") as handle:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(handle) or {}
        return json.load(handle)


def deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def parse_set_values(values: List[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --set value (expected key=value): {item}")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return overrides


def infer_arcface_from_state_dict(state_dict: Dict[str, Any]) -> bool:
    return "head.s" in state_dict


def has_backbone_path(model_config: Dict[str, Any]) -> bool:
    if model_config.get("backbone_path"):
        return True
    backbone = model_config.get("backbone", {}) or {}
    if backbone.get("local_path_override") or backbone.get("huggingface_id"):
        return True
    gcs_assets = model_config.get("gcs_assets", {}) or {}
    clip_backbone = gcs_assets.get("clip_backbone", {}) or {}
    return bool(clip_backbone.get("gcs_path") or clip_backbone.get("local_path"))


def find_missing(model_config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    missing: List[str] = []
    recommended: List[str] = []

    if not model_config.get("model_name"):
        missing.append("model_config.model_name")

    if "use_arcface_head" not in model_config:
        missing.append("model_config.use_arcface_head")
    else:
        if model_config.get("use_arcface_head"):
            if "arcface_s" not in model_config:
                missing.append("model_config.arcface_s")
            if "arcface_m" not in model_config:
                missing.append("model_config.arcface_m")
            if "current_arcface_s" not in model_config:
                recommended.append("model_config.current_arcface_s")

    if "rank" not in model_config:
        missing.append("model_config.rank")

    backbone = model_config.get("backbone", {}) or {}
    if not backbone:
        missing.append("model_config.backbone")
    else:
        if not backbone.get("source"):
            missing.append("model_config.backbone.source")
        if not (backbone.get("variant") or backbone.get("model_name")):
            missing.append("model_config.backbone.variant_or_model_name")
        if "resolution" not in backbone:
            recommended.append("model_config.backbone.resolution")
        if "hidden_size" not in backbone:
            recommended.append("model_config.backbone.hidden_size")

    if not has_backbone_path(model_config):
        missing.append("model_config.backbone_path_or_gcs_assets.clip_backbone")

    if "mean" not in model_config:
        recommended.append("model_config.mean")
    if "std" not in model_config:
        recommended.append("model_config.std")

    return missing, recommended


def load_checkpoint(path: str, trust_checkpoint: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    weights_only = not trust_checkpoint
    if weights_only:
        try:
            safe_globals = torch.serialization.safe_globals  # type: ignore[attr-defined]
        except AttributeError:
            checkpoint_data = torch.load(path, map_location="cpu", weights_only=True)
        else:
            with safe_globals([np.core.multiarray.scalar]):
                checkpoint_data = torch.load(path, map_location="cpu", weights_only=True)
    else:
        checkpoint_data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint_data, dict) and "state_dict" in checkpoint_data:
        model_config = checkpoint_data.get("model_config", {}) or {}
        return checkpoint_data, model_config
    return {"state_dict": checkpoint_data}, {}


def save_checkpoint(checkpoint_data: Dict[str, Any], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(checkpoint_data, output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch checkpoint metadata and re-save.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint (.pth) file (local or gs://).")
    parser.add_argument("--output", help="Output checkpoint path (local or gs://).")
    parser.add_argument("--metadata", help="YAML/JSON file with model_config fields to merge.")
    parser.add_argument("--set", action="append", default=[], help="Override a field (dot.path=value).")
    parser.add_argument("--apply-inferred-arcface", action="store_true",
                        help="If use_arcface_head is missing, infer from state_dict and apply.")
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="Load checkpoint with weights_only=False (unsafe for untrusted files).")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Allow saving even if required metadata is missing.")

    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_checkpoint = args.checkpoint
        if is_gcs_path(args.checkpoint):
            local_checkpoint = os.path.join(tmp_dir, "checkpoint.pth")
            try:
                download_from_gcs(args.checkpoint, local_checkpoint)
            except exceptions.GoogleAPICallError as exc:
                print(f"ERROR: Failed to download checkpoint from GCS: {exc}", file=sys.stderr)
                return 1

        checkpoint_data, model_config = load_checkpoint(local_checkpoint, trust_checkpoint=args.trust_checkpoint)
        state_dict = checkpoint_data.get("state_dict", {})

        if args.metadata:
            overrides = load_metadata_file(args.metadata)
            if "model_config" in overrides:
                overrides = overrides["model_config"] or {}
            deep_merge(model_config, overrides)

        if args.set:
            overrides = parse_set_values(args.set)
            deep_merge(model_config, overrides)

        if "use_arcface_head" not in model_config:
            inferred = infer_arcface_from_state_dict(state_dict)
            if inferred and not args.apply_inferred_arcface:
                print("Hint: state_dict contains head.s (ArcFace detected). Use --apply-inferred-arcface or --set use_arcface_head=true.")
            if inferred and args.apply_inferred_arcface:
                model_config["use_arcface_head"] = True

        model_config["metadata_version"] = 2
        model_config["metadata_updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        checkpoint_data["model_config"] = model_config

        missing, recommended = find_missing(model_config)

        if missing:
            print("Missing required metadata:")
            for item in missing:
                print(f"  - {item}")
        else:
            print("All required metadata is present.")

        if recommended:
            print("Missing recommended metadata:")
            for item in recommended:
                print(f"  - {item}")

        if args.output:
            if missing and not args.allow_incomplete:
                print("ERROR: Required metadata missing. Provide overrides or use --allow-incomplete.", file=sys.stderr)
                return 2

            if is_gcs_path(args.output):
                tmp_output = os.path.join(tmp_dir, "patched_checkpoint.pth")
                save_checkpoint(checkpoint_data, tmp_output)
                try:
                    upload_to_gcs(tmp_output, args.output)
                except exceptions.GoogleAPICallError as exc:
                    print(f"ERROR: Failed to upload checkpoint to GCS: {exc}", file=sys.stderr)
                    return 1
                print(f"Saved checkpoint to {args.output}")
            else:
                save_checkpoint(checkpoint_data, args.output)
                print(f"Saved checkpoint to {args.output}")
        else:
            print("No output path provided; nothing was saved.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
