#!/usr/bin/env python3
"""
Analyze matched before/after Teams pairs from the current buckets.

This is a Track B sidecar tool. It measures image-space deltas on matched
pairs from:

- clean original -> Teams v1 (real / fake)
- clean original -> Teams v2 (real / fake)
- clean enhanced VisoMaster fake -> Teams-enhanced VisoMaster fake

It can also compare the current ``TeamsCodecSimulation`` output against the
real deltas so augmentation decisions stay tied to measurements rather than
historical assumptions.

Example:
    python DeepfakeBench/training/tools/analyze_teams_matched_pairs.py \
      --output-json /tmp/teams_matched_pairs_2026-04-07.json \
      --output-csv /tmp/teams_matched_pairs_2026-04-07.csv \
      --standard-max-samples 64 \
      --enhanced-max-base-samples 24 \
      --sim-repeats 3
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

try:
    from google.cloud import storage as gcs
except ImportError as exc:  # pragma: no cover - environment requirement
    raise SystemExit(
        "Missing google-cloud-storage. Install with: pip install google-cloud-storage"
    ) from exc


ORIGINAL_BUCKET = "live-deepfake-methods-real-and-fake-frames-cropped"
TEAMS_V1_BUCKET = "live-deepfake-methods-real-and-fake-frames-cropped-teams"
TEAMS_V2_BUCKET = "live-deepfake-methods-real-and-fake-frames-cropped-teams-v2"
VISOMASTER_ENHANCED_BUCKET = "visomaster-enhanced-face-cropped"
VISOMASTER_TEAMS_ENHANCED_BUCKET = "enhanced-visomaster-cropped"
DEFAULT_PROJECT = "train-cvit2"
DEFAULT_FRAME_INDICES = (0, 2, 4, 6, 8, 10, 12, 14)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
PCT_DELTA_SIGN_EPS = 1.0


@dataclass(frozen=True)
class FramePair:
    group: str
    sample_id: str
    strategy: str
    swap_model: str
    side: str
    enhancer: str
    source_bucket: str
    source_blob: str
    target_bucket: str
    target_blob: str
    frame_stem: str


def _load_teams_simulation_module():
    training_dir = Path(__file__).resolve().parent.parent
    module_path = training_dir / "data" / "augmentations" / "teams_simulation.py"
    spec = importlib.util.spec_from_file_location("teams_simulation", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import TeamsCodecSimulation from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_simulator(args: argparse.Namespace):
    mod = _load_teams_simulation_module()
    policy = str(args.sim_policy or "legacy_single")
    if policy in {"legacy", "legacy_single", "single"}:
        return mod.TeamsCodecSimulation(always_apply=True, p=1.0), "legacy_single"
    if policy in {"adaptive", "adaptive_mixture", "mixture"}:
        return (
            mod.TeamsAdaptiveCodecSimulation(
                always_apply=True,
                p=1.0,
                ordinary_mode_probability_non_enhanced=float(
                    args.adaptive_ordinary_probability_non_enhanced
                ),
                ordinary_mode_probability_enhanced=float(
                    args.adaptive_ordinary_probability_enhanced
                ),
            ),
            "adaptive_mixture",
        )
    if policy in {"family_split", "hybrid"}:
        return (
            mod.TeamsHybridCodecSimulation(
                always_apply=True,
                p=1.0,
                ordinary_mode_probability_non_enhanced=float(
                    args.adaptive_ordinary_probability_non_enhanced
                ),
                ordinary_mode_probability_enhanced=float(
                    args.adaptive_ordinary_probability_enhanced
                ),
            ),
            "family_split",
        )
    raise ValueError(
        f"Unknown sim policy '{policy}'. "
        "Choose from: legacy_single, adaptive_mixture, family_split"
    )


def _infer_sim_family_key(pair: FramePair) -> str | None:
    if pair.group == "visomaster_enhanced_to_teams":
        return "visomaster_enhanced_fake"
    if pair.side == "real":
        return "realpool_real"
    if pair.sample_id.startswith("visomaster_"):
        return "visomaster_fake"
    strategy = (pair.strategy or "").lower()
    if "enhancement" in strategy:
        return "deeplive_enhanced_fake"
    return "deeplive_non_enhanced_fake"


def compute_sharpness(img_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap))


def compute_brightness(img_rgb: np.ndarray) -> float:
    return float(np.mean(img_rgb.astype(np.float64)))


def compute_contrast(img_rgb: np.ndarray) -> float:
    return float(np.std(img_rgb.astype(np.float64)))


def compute_noise(img_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.median(np.abs(lap)) / 0.6745)


def compute_hf_energy(img_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if gray.shape[0] % 2 == 1:
        gray = gray[:-1, :]
    if gray.shape[1] % 2 == 1:
        gray = gray[:, :-1]
    if gray.size == 0:
        return 0.0
    dct_coeffs = cv2.dct(gray)
    total_energy = float(np.sum(dct_coeffs * dct_coeffs))
    if total_energy < 1e-10:
        return 0.0
    h, w = dct_coeffs.shape
    hf = dct_coeffs[h // 2 :, w // 2 :]
    hf_energy = float(np.sum(hf * hf))
    return hf_energy / total_energy


def compute_chroma_blur(img_rgb: np.ndarray) -> float:
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float64)
    luma = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    luma_var = float(np.var(cv2.Laplacian(luma, cv2.CV_64F)))
    if luma_var < 1e-6:
        return 1.0
    cr_var = float(np.var(cv2.Laplacian(cr, cv2.CV_64F)))
    cb_var = float(np.var(cv2.Laplacian(cb, cv2.CV_64F)))
    return float((cr_var + cb_var) / (2.0 * luma_var))


def compute_blockiness(img_rgb: np.ndarray, block_size: int = 8) -> float:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    h_diff = np.abs(np.diff(gray, axis=0))
    v_diff = np.abs(np.diff(gray, axis=1))
    h_boundary = list(range(block_size - 1, h_diff.shape[0], block_size))
    v_boundary = list(range(block_size - 1, v_diff.shape[1], block_size))
    if not h_boundary or not v_boundary:
        return 1.0
    h_boundary_energy = float(np.mean(h_diff[h_boundary, :]))
    v_boundary_energy = float(np.mean(v_diff[:, v_boundary]))

    h_mask = np.ones(h_diff.shape[0], dtype=bool)
    h_mask[h_boundary] = False
    v_mask = np.ones(v_diff.shape[1], dtype=bool)
    v_mask[v_boundary] = False

    h_non_energy = float(np.mean(h_diff[h_mask, :])) if np.any(h_mask) else 1.0
    v_non_energy = float(np.mean(v_diff[:, v_mask])) if np.any(v_mask) else 1.0
    return float((h_boundary_energy + v_boundary_energy) / max(h_non_energy + v_non_energy, 1e-6))


def compute_bpp(img_rgb: np.ndarray) -> float:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        return 0.0
    h, w = img_rgb.shape[:2]
    return float(len(encoded) * 8.0 / max(h * w, 1))


ALL_METRICS = {
    "sharpness": compute_sharpness,
    "brightness": compute_brightness,
    "contrast": compute_contrast,
    "noise": compute_noise,
    "hf_energy": compute_hf_energy,
    "chroma_blur": compute_chroma_blur,
    "blockiness": compute_blockiness,
    "bpp": compute_bpp,
}


def compute_all_metrics(img_rgb: np.ndarray) -> dict[str, float]:
    return {name: fn(img_rgb) for name, fn in ALL_METRICS.items()}


def pct_delta(base: float, target: float) -> float:
    if abs(base) < 1e-10:
        return 0.0
    return 100.0 * (target - base) / abs(base)


def _safe_mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(np.mean(vals)) if vals else 0.0


def _metric_mode_label(deltas: list[float]) -> str:
    if not deltas:
        return "no_data"
    positive = sum(1 for x in deltas if x > PCT_DELTA_SIGN_EPS) / len(deltas)
    negative = sum(1 for x in deltas if x < -PCT_DELTA_SIGN_EPS) / len(deltas)
    if positive >= 0.2 and negative >= 0.2:
        return "mixed"
    if negative >= 0.8:
        return "mostly_negative"
    if positive >= 0.8:
        return "mostly_positive"
    return "lean_negative" if negative >= positive else "lean_positive"


def _parse_frame_indices(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return DEFAULT_FRAME_INDICES
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _list_sample_ids(bucket: gcs.Bucket) -> list[str]:
    blobs = bucket.list_blobs(prefix="samples/", delimiter="/")
    sample_ids: list[str] = []
    for page in blobs.pages:
        for prefix in sorted(page.prefixes):
            sample_id = prefix.rstrip("/").split("/")[-1]
            if sample_id:
                sample_ids.append(sample_id)
    return sample_ids


def _list_frame_map(
    bucket: gcs.Bucket,
    prefix: str,
    cache: dict[tuple[str, str], dict[str, str]],
) -> dict[str, str]:
    key = (bucket.name, prefix)
    if key in cache:
        return cache[key]
    frame_map: dict[str, str] = {}
    for blob in bucket.list_blobs(prefix=prefix):
        if not blob.name.lower().endswith(IMAGE_SUFFIXES):
            continue
        stem = Path(blob.name).stem
        frame_map[stem] = blob.name
    cache[key] = frame_map
    return frame_map


def _sorted_common_frame_stems(
    left: dict[str, str],
    right: dict[str, str],
) -> list[str]:
    common = set(left) & set(right)
    return sorted(common, key=lambda stem: int(stem.split("_")[-1]))


def _select_frame_stems(stems: list[str], frame_indices: tuple[int, ...]) -> list[str]:
    selected = [stems[i] for i in frame_indices if i < len(stems)]
    return selected if selected else stems[: min(len(stems), 4)]


def _parse_visomaster_swap_model(sample_id: str) -> str:
    parts = sample_id.split("_")
    if len(parts) >= 3 and parts[0] == "visomaster":
        return parts[1]
    return ""


def _discover_standard_pairs(
    client: gcs.Client,
    original_bucket_name: str,
    teams_bucket_name: str,
    side: str,
    group: str,
    frame_indices: tuple[int, ...],
    max_samples: int,
    seed: int,
) -> list[FramePair]:
    original_bucket = client.bucket(original_bucket_name)
    teams_bucket = client.bucket(teams_bucket_name)
    frame_cache: dict[tuple[str, str], dict[str, str]] = {}

    sample_ids = _list_sample_ids(teams_bucket)
    rng = random.Random(seed)
    rng.shuffle(sample_ids)

    pairs: list[FramePair] = []
    selected_samples = 0
    for sample_id in sample_ids:
        manifest_blob = teams_bucket.blob(f"samples/{sample_id}/manifest.json")
        if not manifest_blob.exists():
            continue
        try:
            manifest = json.loads(manifest_blob.download_as_text())
        except Exception:
            continue

        source_map = _list_frame_map(
            original_bucket,
            f"samples/{sample_id}/frames/{side}/",
            frame_cache,
        )
        target_map = _list_frame_map(
            teams_bucket,
            f"samples/{sample_id}/frames/{side}/",
            frame_cache,
        )
        common = _sorted_common_frame_stems(source_map, target_map)
        if not common:
            continue

        strategy = str(manifest.get("strategy") or "unknown")
        swap_model = _parse_visomaster_swap_model(sample_id)
        for frame_stem in _select_frame_stems(common, frame_indices):
            pairs.append(
                FramePair(
                    group=group,
                    sample_id=sample_id,
                    strategy=strategy,
                    swap_model=swap_model,
                    side=side,
                    enhancer="",
                    source_bucket=original_bucket_name,
                    source_blob=source_map[frame_stem],
                    target_bucket=teams_bucket_name,
                    target_blob=target_map[frame_stem],
                    frame_stem=frame_stem,
                )
            )
        selected_samples += 1
        if selected_samples >= max_samples:
            break

    return pairs


def _discover_enhanced_pairs(
    client: gcs.Client,
    clean_enhanced_bucket_name: str,
    teams_enhanced_bucket_name: str,
    frame_indices: tuple[int, ...],
    max_base_samples: int,
    seed: int,
) -> list[FramePair]:
    clean_bucket = client.bucket(clean_enhanced_bucket_name)
    teams_bucket = client.bucket(teams_enhanced_bucket_name)
    frame_cache: dict[tuple[str, str], dict[str, str]] = {}

    sample_ids = _list_sample_ids(teams_bucket)
    rng = random.Random(seed)
    rng.shuffle(sample_ids)

    pairs: list[FramePair] = []
    selected_bases = 0
    for sample_id in sample_ids:
        manifest_blob = teams_bucket.blob(f"samples/{sample_id}/manifest.json")
        if not manifest_blob.exists():
            continue
        try:
            manifest = json.loads(manifest_blob.download_as_text())
        except Exception:
            continue
        enhancers = manifest.get("enhancers") or {}
        if not enhancers:
            continue

        strategy = str(manifest.get("strategy") or "visomaster")
        swap_model = _parse_visomaster_swap_model(sample_id)
        base_added = False
        for enhancer in sorted(enhancers):
            clean_sample_id = f"{sample_id}_enhanced_{enhancer}"
            source_map = _list_frame_map(
                clean_bucket,
                f"samples/{clean_sample_id}/frames/fake/",
                frame_cache,
            )
            target_map = _list_frame_map(
                teams_bucket,
                f"samples/{sample_id}/frames/{enhancer}/",
                frame_cache,
            )
            common = _sorted_common_frame_stems(source_map, target_map)
            if not common:
                continue
            for frame_stem in _select_frame_stems(common, frame_indices):
                pairs.append(
                    FramePair(
                        group="visomaster_enhanced_to_teams",
                        sample_id=sample_id,
                        strategy=strategy,
                        swap_model=swap_model,
                        side="fake",
                        enhancer=enhancer,
                        source_bucket=clean_enhanced_bucket_name,
                        source_blob=source_map[frame_stem],
                        target_bucket=teams_enhanced_bucket_name,
                        target_blob=target_map[frame_stem],
                        frame_stem=frame_stem,
                    )
                )
            base_added = True
        if base_added:
            selected_bases += 1
        if selected_bases >= max_base_samples:
            break

    return pairs


def _download_image_rgb(
    client: gcs.Client,
    bucket_name: str,
    blob_name: str,
) -> np.ndarray:
    bucket = client.bucket(bucket_name)
    data = bucket.blob(blob_name).download_as_bytes()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode gs://{bucket_name}/{blob_name}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _summarize_rows(
    rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
    include_sim: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        grouped[key].append(row)

    summaries: list[dict[str, Any]] = []
    for key, bucket_rows in sorted(grouped.items()):
        summary = {field: value for field, value in zip(key_fields, key)}
        summary["n_frames"] = len(bucket_rows)
        summary["n_samples"] = len({row["sample_id"] for row in bucket_rows})
        metrics: dict[str, Any] = {}
        direction_total = 0
        direction_match_count = 0
        abs_errors: list[float] = []

        for metric in ALL_METRICS:
            src_vals = [row["metrics"][metric]["source"] for row in bucket_rows]
            tgt_vals = [row["metrics"][metric]["target"] for row in bucket_rows]
            real_deltas = [pct_delta(src, tgt) for src, tgt in zip(src_vals, tgt_vals)]
            metric_summary: dict[str, Any] = {
                "source_mean": _safe_mean(src_vals),
                "target_mean": _safe_mean(tgt_vals),
                "delta_real_mean": pct_delta(_safe_mean(src_vals), _safe_mean(tgt_vals)),
                "delta_real_frame_mean": _safe_mean(real_deltas),
                "delta_real_frame_std": float(np.std(real_deltas)) if real_deltas else 0.0,
                "delta_real_p10": float(np.percentile(real_deltas, 10)) if real_deltas else 0.0,
                "delta_real_p50": float(np.percentile(real_deltas, 50)) if real_deltas else 0.0,
                "delta_real_p90": float(np.percentile(real_deltas, 90)) if real_deltas else 0.0,
                "positive_frac": sum(1 for x in real_deltas if x > PCT_DELTA_SIGN_EPS) / len(real_deltas)
                if real_deltas else 0.0,
                "negative_frac": sum(1 for x in real_deltas if x < -PCT_DELTA_SIGN_EPS) / len(real_deltas)
                if real_deltas else 0.0,
                "mode_label": _metric_mode_label(real_deltas),
            }

            if include_sim:
                sim_vals = [row["metrics"][metric]["sim"] for row in bucket_rows]
                sim_deltas = [pct_delta(src, sim) for src, sim in zip(src_vals, sim_vals)]
                delta_real = metric_summary["delta_real_mean"]
                delta_sim = pct_delta(_safe_mean(src_vals), _safe_mean(sim_vals))
                same_direction = (
                    True if abs(delta_real) < 0.5 else (delta_sim > 0) == (delta_real > 0)
                )
                metric_summary.update({
                    "sim_mean": _safe_mean(sim_vals),
                    "delta_sim_mean": delta_sim,
                    "delta_sim_frame_mean": _safe_mean(sim_deltas),
                    "delta_sim_frame_std": float(np.std(sim_deltas)) if sim_deltas else 0.0,
                    "abs_error": abs(delta_sim - delta_real),
                    "direction_match": same_direction,
                })
                if abs(delta_real) >= 0.5:
                    direction_total += 1
                    direction_match_count += int(same_direction)
                abs_errors.append(metric_summary["abs_error"])

            metrics[metric] = metric_summary

        summary["metrics"] = metrics
        if include_sim:
            summary["sim_direction_accuracy"] = (
                direction_match_count / direction_total if direction_total else 1.0
            )
            summary["sim_mean_abs_error"] = _safe_mean(abs_errors)
        summaries.append(summary)

    return summaries


def _flatten_metric_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        shared = {
            key: value
            for key, value in summary.items()
            if key not in {"metrics"}
        }
        for metric_name, metric_summary in summary["metrics"].items():
            row = dict(shared)
            row["metric"] = metric_name
            row.update(metric_summary)
            rows.append(row)
    return rows


def _write_json(path: str, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    client = gcs.Client(project=args.project)
    frame_indices = _parse_frame_indices(args.frame_indices)

    include_sim = not args.skip_simulation
    teams_sim = None
    sim_policy = None
    if include_sim:
        teams_sim, sim_policy = _build_simulator(args)

    standard_specs = [
        ("teams_v1_real", TEAMS_V1_BUCKET, "real", args.seed + 11),
        ("teams_v1_fake", TEAMS_V1_BUCKET, "fake", args.seed + 13),
        ("teams_v2_real", TEAMS_V2_BUCKET, "real", args.seed + 17),
        ("teams_v2_fake", TEAMS_V2_BUCKET, "fake", args.seed + 19),
    ]

    frame_pairs: list[FramePair] = []
    for group, teams_bucket, side, seed in standard_specs:
        discovered = _discover_standard_pairs(
            client=client,
            original_bucket_name=ORIGINAL_BUCKET,
            teams_bucket_name=teams_bucket,
            side=side,
            group=group,
            frame_indices=frame_indices,
            max_samples=args.standard_max_samples,
            seed=seed,
        )
        print(f"{group}: discovered {len(discovered)} frame pairs")
        frame_pairs.extend(discovered)

    if not args.skip_enhanced:
        discovered = _discover_enhanced_pairs(
            client=client,
            clean_enhanced_bucket_name=VISOMASTER_ENHANCED_BUCKET,
            teams_enhanced_bucket_name=VISOMASTER_TEAMS_ENHANCED_BUCKET,
            frame_indices=frame_indices,
            max_base_samples=args.enhanced_max_base_samples,
            seed=args.seed + 23,
        )
        print(f"visomaster_enhanced_to_teams: discovered {len(discovered)} frame pairs")
        frame_pairs.extend(discovered)

    if not frame_pairs:
        raise RuntimeError("No matched frame pairs discovered.")

    rows: list[dict[str, Any]] = []
    for idx, pair in enumerate(frame_pairs, start=1):
        source_img = _download_image_rgb(client, pair.source_bucket, pair.source_blob)
        target_img = _download_image_rgb(client, pair.target_bucket, pair.target_blob)
        source_metrics = compute_all_metrics(source_img)
        target_metrics = compute_all_metrics(target_img)

        metric_bundle: dict[str, dict[str, float]] = {
            name: {
                "source": source_metrics[name],
                "target": target_metrics[name],
            }
            for name in ALL_METRICS
        }

        if include_sim and teams_sim is not None:
            family_key = _infer_sim_family_key(pair)
            family_apply = getattr(teams_sim, "apply_for_family", None)
            if callable(family_apply):
                sim_images = [
                    family_apply(source_img, family_key=family_key)
                    for _ in range(args.sim_repeats)
                ]
            else:
                sim_images = [teams_sim(image=source_img)["image"] for _ in range(args.sim_repeats)]
            sim_mean = np.mean(sim_images, axis=0).astype(np.uint8)
            sim_metrics = compute_all_metrics(sim_mean)
            for name in ALL_METRICS:
                metric_bundle[name]["sim"] = sim_metrics[name]

        rows.append({
            "group": pair.group,
            "sample_id": pair.sample_id,
            "strategy": pair.strategy,
            "swap_model": pair.swap_model,
            "side": pair.side,
            "enhancer": pair.enhancer,
            "frame_stem": pair.frame_stem,
            "source_bucket": pair.source_bucket,
            "source_blob": pair.source_blob,
            "target_bucket": pair.target_bucket,
            "target_blob": pair.target_blob,
            "metrics": metric_bundle,
        })

        if idx % 50 == 0 or idx == len(frame_pairs):
            print(f"processed {idx}/{len(frame_pairs)} frame pairs")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "project": args.project,
            "standard_max_samples": args.standard_max_samples,
            "enhanced_max_base_samples": args.enhanced_max_base_samples,
            "frame_indices": list(frame_indices),
            "include_simulation": include_sim,
            "sim_policy": sim_policy,
            "adaptive_ordinary_probability_non_enhanced": (
                args.adaptive_ordinary_probability_non_enhanced
                if sim_policy in {"adaptive_mixture", "family_split"}
                else None
            ),
            "adaptive_ordinary_probability_enhanced": (
                args.adaptive_ordinary_probability_enhanced
                if sim_policy in {"adaptive_mixture", "family_split"}
                else None
            ),
            "sim_repeats": args.sim_repeats,
            "seed": args.seed,
        },
        "pair_counts": {
            group: sum(1 for row in rows if row["group"] == group)
            for group in sorted({row["group"] for row in rows})
        },
        "summaries": {
            "overall": _summarize_rows(rows, ("group",), include_sim=include_sim),
            "by_strategy": _summarize_rows(rows, ("group", "strategy"), include_sim=include_sim),
            "by_swap_model": _summarize_rows(
                [row for row in rows if row["swap_model"]],
                ("group", "swap_model"),
                include_sim=include_sim,
            ),
            "by_enhancer": _summarize_rows(
                [row for row in rows if row["enhancer"]],
                ("group", "enhancer"),
                include_sim=include_sim,
            ),
        },
    }

    if args.output_json:
        _write_json(args.output_json, payload)
    if args.output_csv:
        flat_rows = _flatten_metric_summaries(payload["summaries"]["overall"])
        _write_csv(args.output_csv, flat_rows)

    print("\nGroup summary:")
    for summary in payload["summaries"]["overall"]:
        group = summary["group"]
        sharpness = summary["metrics"]["sharpness"]["delta_real_mean"]
        brightness = summary["metrics"]["brightness"]["delta_real_mean"]
        noise = summary["metrics"]["noise"]["delta_real_mean"]
        line = (
            f"  {group:<30} n_frames={summary['n_frames']:<4d} "
            f"sharpness={sharpness:+6.1f}% "
            f"brightness={brightness:+6.1f}% "
            f"noise={noise:+6.1f}%"
        )
        if include_sim:
            line += (
                f" sim_dir_acc={summary['sim_direction_accuracy'] * 100:5.1f}% "
                f"sim_mae={summary['sim_mean_abs_error']:5.1f}"
            )
        print(line)

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze current matched Teams pairs from live GCS buckets."
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--standard-max-samples", type=int, default=64)
    parser.add_argument("--enhanced-max-base-samples", type=int, default=24)
    parser.add_argument("--frame-indices", default="0,2,4,6,8,10,12,14")
    parser.add_argument("--seed", type=int, default=737)
    parser.add_argument("--sim-repeats", type=int, default=3)
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument(
        "--sim-policy",
        default="legacy_single",
        choices=("legacy_single", "adaptive_mixture", "family_split"),
    )
    parser.add_argument("--adaptive-ordinary-probability-non-enhanced", type=float, default=0.75)
    parser.add_argument("--adaptive-ordinary-probability-enhanced", type=float, default=0.25)
    parser.add_argument("--skip-enhanced", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
