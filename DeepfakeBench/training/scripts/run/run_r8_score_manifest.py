#!/usr/bin/env python3
"""
Score a labeled manifest with an R8 checkpoint and export raw probabilities.

Input manifest CSV must include at least:
  - path (image path)
  - label (0/1 or real/fake)
  - identity (identity token)

Output CSV includes all input columns + raw_prob + score_status.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

try:
    import yaml
except ImportError:
    yaml = None

try:
    from google.cloud import storage
except ImportError:
    storage = None

try:
    import video_preprocessor
except ImportError:
    video_preprocessor = None

try:
    from detectors import DETECTOR
except ImportError:
    DETECTOR = None


def _parse_gs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    no_scheme = uri[5:]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    blob = parts[1] if len(parts) > 1 else ""
    return bucket, blob


def _download_if_gcs(path_or_uri: str) -> Tuple[str, Optional[str]]:
    if not path_or_uri.startswith("gs://"):
        return path_or_uri, None

    if storage is None:
        raise ImportError("google-cloud-storage is required for gs:// checkpoint paths")

    bucket_name, blob_name = _parse_gs_uri(path_or_uri)
    local_dir = tempfile.mkdtemp(prefix="r8_ckpt_")
    local_path = str(Path(local_dir) / Path(blob_name).name)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists(client=client):
        raise FileNotFoundError(f"Checkpoint not found in GCS: {path_or_uri}")
    blob.download_to_filename(local_path)
    return local_path, local_dir


def _resolve_device(device_arg: str) -> torch.device:
    device_arg = device_arg.lower().strip()
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda requested, but CUDA is unavailable")
    return torch.device(device_arg)


def _load_merged_config(detector_config: str, train_config: str) -> Dict:
    with open(detector_config, "r") as f:
        cfg = yaml.safe_load(f)
    with open(train_config, "r") as f:
        cfg.update(yaml.safe_load(f))
    return cfg


def _load_detector(cfg: Dict, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        model_config = ckpt.get("model_config", {})
        for k, v in model_config.items():
            if k != "current_arcface_s":
                cfg[k] = v
    else:
        state_dict = ckpt
        model_config = {}

    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)

    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])

    clean_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    return model


def _parse_label(value: str) -> int:
    text = str(value).strip().lower()
    if text in {"0", "real", "r", "false", "negative"}:
        return 0
    if text in {"1", "fake", "f", "true", "positive"}:
        return 1
    try:
        v = int(float(text))
    except Exception as exc:
        raise ValueError(f"Unsupported label value: {value}") from exc
    if v not in (0, 1):
        raise ValueError(f"Label must be 0 or 1, got: {value}")
    return v


def _read_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def _resolve_path(path_value: str, root_dir: Optional[str]) -> str:
    p = Path(path_value)
    if p.is_absolute():
        return str(p)
    if root_dir:
        return str((Path(root_dir) / p).resolve())
    return str(p.resolve())


def _preprocess_image(
    *,
    img_bgr: np.ndarray,
    recrop: bool,
    yolo_conf_threshold: float,
) -> np.ndarray:
    if recrop:
        face = video_preprocessor.extract_yolo_face(img_bgr, yolo_conf_threshold)
        if face is None:
            raise RuntimeError("face_not_found")
        return face
    return cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)


def _run_batch(
    *,
    model: torch.nn.Module,
    device: torch.device,
    tensors: List[torch.Tensor],
) -> List[float]:
    if not tensors:
        return []
    batch = torch.cat(tensors, dim=0).to(device)
    with torch.inference_mode():
        preds = model({"image": batch}, inference=True)
        probs = preds["prob"].detach().squeeze().cpu().numpy()
    if isinstance(probs, np.ndarray):
        if probs.ndim == 0:
            return [float(probs.item())]
        return [float(x) for x in probs.reshape(-1)]
    return [float(probs)]


def _write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score calibration manifest with checkpoint")
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Local path or gs:// path to checkpoint")

    parser.add_argument("--detector_config", type=str, default="config/detector/effort.yaml")
    parser.add_argument("--train_config", type=str, default="config/train_config.yaml")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])

    parser.add_argument("--path_col", type=str, default="path")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--identity_col", type=str, default="identity")
    parser.add_argument("--root_dir", type=str, default=None,
                        help="Optional root for relative paths in manifest")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--recrop", action="store_true", default=False,
                        help="Run YOLO face detection before scoring")
    parser.add_argument("--yolo_conf_threshold", type=float, default=0.20)

    parser.add_argument("--failures_csv", type=str, default="",
                        help="Optional path for failed rows")
    args = parser.parse_args()

    missing = []
    if cv2 is None:
        missing.append("cv2")
    if np is None:
        missing.append("numpy")
    if torch is None:
        missing.append("torch")
    if yaml is None:
        missing.append("pyyaml")
    if video_preprocessor is None:
        missing.append("video_preprocessor module")
    if DETECTOR is None:
        missing.append("detectors module")
    if missing:
        raise ImportError(
            "Missing required runtime dependencies: "
            + ", ".join(missing)
            + ". Run from the training environment/container."
        )

    manifest_rows = _read_manifest(args.manifest_csv)

    for col in (args.path_col, args.label_col, args.identity_col):
        if col not in manifest_rows[0]:
            raise ValueError(f"Missing required column in manifest: {col}")

    device = _resolve_device(args.device)
    print(f"[run_r8_score_manifest] using device: {device}")

    local_ckpt, temp_dir = _download_if_gcs(args.checkpoint)
    try:
        config = _load_merged_config(args.detector_config, args.train_config)
        model = _load_detector(config, local_ckpt, device)

        if args.recrop:
            video_preprocessor.initialize_yolo_model()

        transform = video_preprocessor._get_transform()

        output_rows: List[Dict[str, str]] = []
        failure_rows: List[Dict[str, str]] = []

        pending_tensors: List[torch.Tensor] = []
        pending_meta: List[Dict[str, str]] = []

        def _flush() -> None:
            nonlocal pending_tensors, pending_meta
            probs = _run_batch(model=model, device=device, tensors=pending_tensors)
            if len(probs) != len(pending_meta):
                raise RuntimeError("Batch output size mismatch")
            for meta, prob in zip(pending_meta, probs):
                row = dict(meta)
                row["raw_prob"] = f"{prob:.8f}"
                row["score_status"] = "ok"
                output_rows.append(row)
            pending_tensors = []
            pending_meta = []

        for row in manifest_rows:
            path_value = row[args.path_col]
            resolved_path = _resolve_path(path_value, args.root_dir)

            try:
                label_int = _parse_label(row[args.label_col])
                _ = row[args.identity_col]  # validation only

                img = cv2.imread(resolved_path)
                if img is None:
                    raise RuntimeError("image_read_failed")

                processed = _preprocess_image(
                    img_bgr=img,
                    recrop=args.recrop,
                    yolo_conf_threshold=args.yolo_conf_threshold,
                )

                rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                tensor = transform(rgb).unsqueeze(0)

                meta = dict(row)
                meta[args.path_col] = resolved_path
                meta[args.label_col] = str(label_int)

                pending_tensors.append(tensor)
                pending_meta.append(meta)

                if len(pending_tensors) >= args.batch_size:
                    _flush()

            except Exception as exc:
                fail = dict(row)
                fail[args.path_col] = resolved_path
                fail["score_status"] = "failed"
                fail["error"] = str(exc)
                failure_rows.append(fail)

        if pending_tensors:
            _flush()

        base_fields = list(manifest_rows[0].keys())
        out_fields = base_fields + ["raw_prob", "score_status"]
        _write_csv(args.output_csv, output_rows, out_fields)

        if args.failures_csv:
            fail_fields = list(manifest_rows[0].keys()) + ["score_status", "error"]
            _write_csv(args.failures_csv, failure_rows, fail_fields)

        print("[run_r8_score_manifest] done")
        print(f"  manifest      : {Path(args.manifest_csv).resolve()}")
        print(f"  output_csv    : {Path(args.output_csv).resolve()}")
        print(f"  scored_ok     : {len(output_rows)}")
        print(f"  failed        : {len(failure_rows)}")

        if not output_rows:
            raise RuntimeError("No rows were successfully scored.")

    finally:
        if temp_dir:
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
