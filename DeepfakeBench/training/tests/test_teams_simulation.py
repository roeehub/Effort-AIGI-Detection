#!/usr/bin/env python3
"""
Validate TeamsCodecSimulation against real Teams-transmitted frames.

Downloads matched triplets:
  1. Original frame (clean PNG from source bucket)
  2. Real Teams frame (JPG from Teams bucket — actual Teams passthrough)
  3. Simulated Teams frame (TeamsCodecSimulation applied to original)

Computes 8 metrics from the R9 plan and prints a comparison table:
  - Original → Real Teams delta  (ground truth)
  - Original → Simulated delta   (our augmentation)
  - Match %: how close simulated is to real

Usage:
  cd DeepfakeBench/training
  python -m tests.test_teams_simulation --num-samples 10 --num-frames 8
  python -m tests.test_teams_simulation --num-samples 10 --save-images test_output/

Requirements:
  pip install numpy opencv-python Pillow scipy google-cloud-storage albumentations==0.4.6
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.signal import convolve2d

# Add training dir to path so we can import our augmentation
_training_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _training_dir)

# Direct import to avoid pulling in the full data.augmentations package
# (which requires torch, albumentations matching albucore, etc.)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "teams_simulation",
    os.path.join(_training_dir, "data", "augmentations", "teams_simulation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TeamsCodecSimulation = _mod.TeamsCodecSimulation

try:
    from google.cloud import storage as gcs
except ImportError:
    sys.exit("Missing google-cloud-storage. Install: pip install google-cloud-storage")


# ── Constants ───────────────────────────────────────────────────────────────

ORIGINAL_BUCKET = "live-deepfake-methods-real-and-fake-frames-cropped"
TEAMS_BUCKET = "live-deepfake-methods-real-and-fake-frames-cropped-teams"
PROJECT = "train-cvit2"

# Ground-truth deltas measured from 18 samples × 132 paired frames (validated)
GROUND_TRUTH_DELTAS = {
    "sharpness":     -50.9,   # % change — Teams blurs (original plan was WRONG: +37.8)
    "brightness":    +19.0,
    "contrast":      +3.5,
    "noise":         -11.8,   # Teams introduces no noise (plan was WRONG: +8.3)
    "hf_energy":     -77.4,
    "chroma_blur":   +5.3,    # noisy metric (std=98.4%)
    "blockiness":    -1.2,    # near-zero (std=7.5%)
    "bpp":           -8.7,
}

LAP_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


# ── Metric functions ────────────────────────────────────────────────────────

def compute_sharpness(img_rgb: np.ndarray) -> float:
    """Laplacian variance (higher = sharper)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    filtered = convolve2d(gray, LAP_KERNEL, mode="valid")
    return float(np.var(filtered))


def compute_brightness(img_rgb: np.ndarray) -> float:
    """Mean pixel value across all channels."""
    return float(np.mean(img_rgb.astype(np.float64)))


def compute_contrast(img_rgb: np.ndarray) -> float:
    """Pixel std across all channels."""
    return float(np.std(img_rgb.astype(np.float64)))


def compute_noise(img_rgb: np.ndarray) -> float:
    """MAD-based noise estimate from high-pass filtered grayscale."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    filtered = convolve2d(gray, LAP_KERNEL, mode="valid")
    sigma = float(np.median(np.abs(filtered)) / 0.6745)
    return sigma


def compute_hf_energy(img_rgb: np.ndarray) -> float:
    """Ratio of high-frequency DCT energy to total."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    from scipy.fftpack import dct
    dct_coeffs = dct(dct(gray, axis=0, norm='ortho'), axis=1, norm='ortho')
    h, w = dct_coeffs.shape
    total_energy = np.sum(dct_coeffs ** 2)
    if total_energy < 1e-10:
        return 0.0
    # High-freq: bottom-right quadrant
    hf_energy = np.sum(dct_coeffs[h // 2:, w // 2:] ** 2)
    return float(hf_energy / total_energy)


def compute_chroma_blur(img_rgb: np.ndarray) -> float:
    """Chroma sharpness / luma sharpness ratio. Lower = more chroma blur."""
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float64)
    y_ch = ycrcb[:, :, 0]
    cr_ch = ycrcb[:, :, 1]
    cb_ch = ycrcb[:, :, 2]
    luma_var = float(np.var(convolve2d(y_ch, LAP_KERNEL, mode="valid")))
    cr_var = float(np.var(convolve2d(cr_ch, LAP_KERNEL, mode="valid")))
    cb_var = float(np.var(convolve2d(cb_ch, LAP_KERNEL, mode="valid")))
    if luma_var < 1e-6:
        return 1.0
    return float((cr_var + cb_var) / (2.0 * luma_var))


def compute_blockiness(img_rgb: np.ndarray, block_size: int = 8) -> float:
    """Ratio of boundary-edge-energy to non-boundary-edge-energy."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    h, w = gray.shape
    h_diff = np.abs(np.diff(gray, axis=0))
    v_diff = np.abs(np.diff(gray, axis=1))
    h_boundary = list(range(block_size - 1, h - 1, block_size))
    v_boundary = list(range(block_size - 1, w - 1, block_size))
    if not h_boundary or not v_boundary:
        return 1.0
    h_b_e = np.mean(h_diff[h_boundary, :])
    v_b_e = np.mean(v_diff[:, v_boundary])
    h_non = [r for r in range(h_diff.shape[0]) if r not in set(h_boundary)]
    v_non = [c for c in range(v_diff.shape[1]) if c not in set(v_boundary)]
    h_nb_e = np.mean(h_diff[h_non, :]) if h_non else 1.0
    v_nb_e = np.mean(v_diff[:, v_non]) if v_non else 1.0
    return float((h_b_e + v_b_e) / max(h_nb_e + v_nb_e, 1e-6))


def compute_bpp(img_rgb: np.ndarray) -> float:
    """Bits-per-pixel when JPEG-encoded at quality 95."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    n_bytes = len(encoded)
    h, w = img_rgb.shape[:2]
    return float(n_bytes * 8.0 / max(h * w, 1))


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


# ── GCS helpers ─────────────────────────────────────────────────────────────

def download_image_from_gcs(client: gcs.Client, bucket_name: str,
                            blob_path: str) -> np.ndarray:
    """Download image from GCS and return as RGB numpy array."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    data = blob.download_as_bytes()
    nparr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to decode image: {bucket_name}/{blob_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def discover_complete_samples(client: gcs.Client, max_samples: int) -> list[str]:
    """Find sample IDs that have pair_complete=true in Teams bucket."""
    bucket = client.bucket(TEAMS_BUCKET)
    # List all manifest.json files
    prefix = "samples/"
    blobs = bucket.list_blobs(prefix=prefix, delimiter="/")
    
    # Collect sample prefixes
    sample_prefixes = []
    # Need to iterate over pages to get prefixes
    for page in blobs.pages:
        sample_prefixes.extend(page.prefixes)
    
    complete = []
    for sp in sample_prefixes:
        sample_id = sp.rstrip("/").split("/")[-1]
        manifest_blob = bucket.blob(f"samples/{sample_id}/manifest.json")
        if not manifest_blob.exists():
            continue
        manifest = json.loads(manifest_blob.download_as_string())
        if manifest.get("pair_complete", False):
            complete.append(sample_id)
            if len(complete) >= max_samples:
                break
    return complete


# ── Main ────────────────────────────────────────────────────────────────────

@dataclass
class FrameMetrics:
    original: dict[str, float] = field(default_factory=dict)
    teams_real: dict[str, float] = field(default_factory=dict)
    teams_sim: dict[str, float] = field(default_factory=dict)


def pct_delta(base: float, target: float) -> float:
    if abs(base) < 1e-10:
        return 0.0
    return 100.0 * (target - base) / abs(base)


def run_validation(args):
    print("=" * 80)
    print("TeamsCodecSimulation Validation")
    print("=" * 80)
    print(f"\nDownloading {args.num_samples} samples × {args.num_frames} frames from GCS...")
    print(f"  Original bucket: {ORIGINAL_BUCKET}")
    print(f"  Teams bucket:    {TEAMS_BUCKET}\n")

    client = gcs.Client(project=PROJECT)

    # Discover complete samples
    sample_ids = discover_complete_samples(client, args.num_samples)
    print(f"Found {len(sample_ids)} complete samples: {sample_ids[:5]}{'...' if len(sample_ids) > 5 else ''}\n")

    if not sample_ids:
        print("ERROR: No complete samples found in Teams bucket!")
        return

    # Create augmentation (always_apply=True so every frame gets transformed)
    teams_sim = TeamsCodecSimulation(
        brightness_limit=(0.02, 0.10),
        contrast_limit=(0.10, 0.22),
        blur_sigma=(0.35, 0.85),
        jpeg_quality=(72, 88),
        chroma_blur_ksize=0,
        deblock_d=5,
        deblock_sigma_color=30.0,
        deblock_sigma_space=30.0,
        always_apply=True,
        p=1.0,
    )

    # Collect per-frame metrics
    all_metrics: list[FrameMetrics] = []
    save_dir = Path(args.save_images) if args.save_images else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    n_downloaded = 0
    for si, sample_id in enumerate(sample_ids):
        # Use real frames (not fake) for this test — codec effects are the same
        for fi in range(args.num_frames):
            frame_name = f"frame_{fi:04d}"
            try:
                # 1. Original (PNG)
                orig = download_image_from_gcs(
                    client, ORIGINAL_BUCKET,
                    f"samples/{sample_id}/frames/real/{frame_name}.png"
                )
                # 2. Teams real (JPG)
                teams = download_image_from_gcs(
                    client, TEAMS_BUCKET,
                    f"samples/{sample_id}/frames/real/{frame_name}.jpg"
                )
            except Exception as e:
                print(f"  SKIP {sample_id}/{frame_name}: {e}")
                continue

            # 3. Simulated = TeamsCodecSimulation(original)
            # Run multiple times and average to reduce randomness
            sim_runs = []
            for _ in range(args.sim_repeats):
                sim_runs.append(teams_sim(image=orig)["image"])

            # Use mean of multiple runs for metric computation
            sim_mean = np.mean(sim_runs, axis=0).astype(np.uint8)
            # But also keep a single run for visual comparison
            sim_single = sim_runs[0]

            fm = FrameMetrics(
                original=compute_all_metrics(orig),
                teams_real=compute_all_metrics(teams),
                teams_sim=compute_all_metrics(sim_mean),
            )
            all_metrics.append(fm)
            n_downloaded += 1

            # Save side-by-side images
            if save_dir and fi < 3:  # Save first 3 frames per sample
                _save_comparison(save_dir, sample_id, frame_name,
                                 orig, teams, sim_single)

            if n_downloaded % 10 == 0:
                print(f"  Processed {n_downloaded} frames...")

    print(f"\nTotal frames processed: {n_downloaded}\n")
    if not all_metrics:
        print("ERROR: No frames were successfully processed!")
        return

    # ── Compute aggregate deltas ──
    metric_names = list(ALL_METRICS.keys())

    # Collect raw values
    orig_vals = {m: [] for m in metric_names}
    real_vals = {m: [] for m in metric_names}
    sim_vals = {m: [] for m in metric_names}

    for fm in all_metrics:
        for m in metric_names:
            orig_vals[m].append(fm.original[m])
            real_vals[m].append(fm.teams_real[m])
            sim_vals[m].append(fm.teams_sim[m])

    # Print results table
    print("=" * 100)
    print(f"{'METRIC':<14} {'ORIGINAL':>10} {'REAL TEAMS':>12} {'SIM TEAMS':>12} "
          f"{'Δ REAL %':>10} {'Δ SIM %':>10} {'PLAN Δ %':>10} {'SIM MATCH':>12}")
    print("-" * 100)

    for m in metric_names:
        o_mean = np.mean(orig_vals[m])
        r_mean = np.mean(real_vals[m])
        s_mean = np.mean(sim_vals[m])

        delta_real = pct_delta(o_mean, r_mean)
        delta_sim = pct_delta(o_mean, s_mean)
        delta_plan = GROUND_TRUTH_DELTAS.get(m, 0.0)

        # Match quality: how close is sim delta to real delta?
        if abs(delta_real) > 0.1:
            match_pct = 100.0 * (1 - abs(delta_sim - delta_real) / abs(delta_real))
            match_str = f"{match_pct:+.0f}%"
        else:
            match_str = "N/A"

        # Direction check
        same_dir = (delta_sim > 0) == (delta_real > 0) if abs(delta_real) > 0.5 else True
        dir_symbol = "✓" if same_dir else "✗ DIR"

        print(f"{m:<14} {o_mean:>10.2f} {r_mean:>12.2f} {s_mean:>12.2f} "
              f"{delta_real:>+10.1f} {delta_sim:>+10.1f} {delta_plan:>+10.1f} "
              f"{match_str:>8} {dir_symbol:>3}")

    print("-" * 100)

    # ── Direction match summary ──
    print("\n" + "=" * 80)
    print("DIRECTION MATCH SUMMARY")
    print("=" * 80)
    n_correct = 0
    n_total = 0
    for m in metric_names:
        o_mean = np.mean(orig_vals[m])
        r_mean = np.mean(real_vals[m])
        s_mean = np.mean(sim_vals[m])
        delta_real = pct_delta(o_mean, r_mean)
        delta_sim = pct_delta(o_mean, s_mean)
        if abs(delta_real) > 0.5:
            n_total += 1
            same = (delta_sim > 0) == (delta_real > 0)
            n_correct += int(same)
            print(f"  {m:<14}: Real Δ={delta_real:+.1f}%  Sim Δ={delta_sim:+.1f}%  "
                  f"{'✓ MATCH' if same else '✗ MISMATCH'}")
    print(f"\n  Direction accuracy: {n_correct}/{n_total} "
          f"({100*n_correct/max(n_total,1):.0f}%)")

    # ── Per-metric distribution (optional detail) ──
    if args.verbose:
        print("\n" + "=" * 80)
        print("PER-FRAME DELTA DISTRIBUTIONS")
        print("=" * 80)
        for m in metric_names:
            deltas_real = [pct_delta(o, r) for o, r in zip(orig_vals[m], real_vals[m]) if abs(o) > 1e-10]
            deltas_sim = [pct_delta(o, s) for o, s in zip(orig_vals[m], sim_vals[m]) if abs(o) > 1e-10]
            if deltas_real and deltas_sim:
                print(f"\n  {m}:")
                print(f"    Real Teams: mean={np.mean(deltas_real):+.1f}%  "
                      f"std={np.std(deltas_real):.1f}%  "
                      f"[{np.percentile(deltas_real,5):+.1f}, {np.percentile(deltas_real,95):+.1f}]")
                print(f"    Simulated:  mean={np.mean(deltas_sim):+.1f}%  "
                      f"std={np.std(deltas_sim):.1f}%  "
                      f"[{np.percentile(deltas_sim,5):+.1f}, {np.percentile(deltas_sim,95):+.1f}]")

    if save_dir:
        print(f"\n  Side-by-side images saved to: {save_dir}/")


def _save_comparison(save_dir: Path, sample_id: str, frame_name: str,
                     orig: np.ndarray, teams: np.ndarray, sim: np.ndarray):
    """Save a side-by-side comparison of original, real Teams, and simulated."""
    # Resize all to same dimensions for comparison
    h = min(orig.shape[0], teams.shape[0], sim.shape[0])
    w = min(orig.shape[1], teams.shape[1], sim.shape[1])

    def resize(img):
        return cv2.resize(img, (w, h)) if img.shape[:2] != (h, w) else img

    orig_r = resize(orig)
    teams_r = resize(teams)
    sim_r = resize(sim)

    # Create side-by-side with labels
    gap = 4
    label_h = 30
    canvas_w = w * 3 + gap * 2
    canvas_h = h + label_h
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # Place images
    canvas[label_h:label_h + h, 0:w] = orig_r
    canvas[label_h:label_h + h, w + gap:2 * w + gap] = teams_r
    canvas[label_h:label_h + h, 2 * w + 2 * gap:3 * w + 2 * gap] = sim_r

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.putText(canvas_bgr, "Original", (10, 20), font, 0.5, (0, 0, 0), 1)
    cv2.putText(canvas_bgr, "Real Teams", (w + gap + 10, 20), font, 0.5, (0, 0, 200), 1)
    cv2.putText(canvas_bgr, "Simulated", (2 * w + 2 * gap + 10, 20), font, 0.5, (200, 0, 0), 1)

    out_path = save_dir / f"{sample_id}_{frame_name}.jpg"
    cv2.imwrite(str(out_path), canvas_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate TeamsCodecSimulation against real Teams frames")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of Teams samples to download (default: 10)")
    parser.add_argument("--num-frames", type=int, default=8,
                        help="Frames per sample to compare (default: 8)")
    parser.add_argument("--sim-repeats", type=int, default=5,
                        help="Run TeamsCodecSimulation N times per frame and average "
                             "metrics to reduce randomness (default: 5)")
    parser.add_argument("--save-images", type=str, default=None,
                        help="Save side-by-side comparison images to this directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-frame delta distributions")
    args = parser.parse_args()
    run_validation(args)


if __name__ == "__main__":
    main()
