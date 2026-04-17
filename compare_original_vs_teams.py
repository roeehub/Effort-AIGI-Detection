#!/usr/bin/env python3
"""
Compare original GCS frames vs Teams-augmented frames.

After reconstruct_teams_dataset.py has created the teams_dataset/ directory,
this script downloads matching original frames from GCS and compares properties:
  - Resolution / aspect ratio differences
  - File size / JPEG compression level
  - Color distribution (histogram comparison)
  - Frequency domain analysis (DCT energy, high-freq content)
  - Pixel-level statistics (brightness, contrast, sharpness)

This helps verify that the Teams pipeline introduces measurable augmentation
(compression, color shifts, resolution changes) that our model should learn.

Usage:
  python compare_original_vs_teams.py \\
      --teams-dir teams_dataset \\
      --assignment teams_dataset/assignment.json \\
      --output-dir comparison_results \\
      --max-samples 20

Prerequisites:
  pip install numpy Pillow scipy matplotlib google-cloud-storage
"""

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

try:
    from google.cloud import storage as gcs
except ImportError:
    sys.exit("Missing google-cloud-storage. Install: pip install google-cloud-storage")

try:
    from scipy import fftpack
except ImportError:
    fftpack = None
    print("⚠ scipy not installed — skipping frequency analysis")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("⚠ matplotlib not installed — skipping plots")


# ── Config ──────────────────────────────────────────────────────────────────

GCS_PROJECT = "train-cvit2"
FRAMES_BUCKET = "live-deepfake-methods-real-and-fake-frames-cropped"


# ── Image analysis functions ────────────────────────────────────────────────

def image_stats(img: Image.Image) -> dict:
    """Compute basic image statistics."""
    arr = np.array(img).astype(np.float32)

    stats = {
        "width": img.width,
        "height": img.height,
        "aspect_ratio": round(img.width / max(img.height, 1), 3),
        "mean_brightness": float(np.mean(arr)),
        "std_brightness": float(np.std(arr)),
    }

    # Per-channel means
    if arr.ndim == 3 and arr.shape[2] >= 3:
        stats["mean_r"] = float(np.mean(arr[:, :, 0]))
        stats["mean_g"] = float(np.mean(arr[:, :, 1]))
        stats["mean_b"] = float(np.mean(arr[:, :, 2]))

    return stats


def sharpness_score(img: Image.Image) -> float:
    """Laplacian variance as a sharpness measure."""
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float64)
    # Laplacian kernel
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    from scipy.signal import convolve2d
    filtered = convolve2d(arr, lap, mode="valid")
    return float(np.var(filtered))


def blockiness_score(img: Image.Image, block_size: int = 8) -> float:
    """
    Detect H.264 macro-block artifacts by measuring edge energy
    along block boundaries vs non-boundary positions.

    Higher score = more visible block boundaries = more compression.
    Returns ratio of boundary-edge-energy to non-boundary-edge-energy.
    A pristine image ≈ 1.0; a blocky image > 1.0.
    """
    gray = np.array(img.convert("L"), dtype=np.float64)
    h, w = gray.shape

    # Horizontal edges (diff along rows)
    h_diff = np.abs(np.diff(gray, axis=0))  # shape (h-1, w)
    # Vertical edges (diff along cols)
    v_diff = np.abs(np.diff(gray, axis=1))  # shape (h, w-1)

    # Boundary rows/cols (every block_size pixels)
    h_boundary_rows = list(range(block_size - 1, h - 1, block_size))
    v_boundary_cols = list(range(block_size - 1, w - 1, block_size))

    if not h_boundary_rows or not v_boundary_cols:
        return 1.0

    # Energy at boundaries
    h_boundary_energy = np.mean(h_diff[h_boundary_rows, :])
    v_boundary_energy = np.mean(v_diff[:, v_boundary_cols])

    # Energy at non-boundaries
    h_non_rows = [r for r in range(h_diff.shape[0]) if r not in set(h_boundary_rows)]
    v_non_cols = [c for c in range(v_diff.shape[1]) if c not in set(v_boundary_cols)]

    h_non_energy = np.mean(h_diff[h_non_rows, :]) if h_non_rows else 1.0
    v_non_energy = np.mean(v_diff[:, v_non_cols]) if v_non_cols else 1.0

    # Ratio: >1 means block boundaries are more visible than expected
    ratio = ((h_boundary_energy + v_boundary_energy) /
             max(h_non_energy + v_non_energy, 1e-6))
    return float(round(ratio, 4))


def chroma_blur_ratio(img: Image.Image) -> float:
    """
    Measure chroma sharpness relative to luma sharpness.

    YUV 4:2:0 (used by Teams/H.264) halves chroma resolution,
    so Teams frames should have lower chroma sharpness vs luma.
    Returns chroma_sharpness / luma_sharpness. Lower = more chroma blur.
    """
    from scipy.signal import convolve2d
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

    arr = np.array(img.convert("YCbCr"), dtype=np.float64)
    y_ch = arr[:, :, 0]   # Luma
    cb_ch = arr[:, :, 1]  # Chroma blue
    cr_ch = arr[:, :, 2]  # Chroma red

    luma_var = float(np.var(convolve2d(y_ch, lap, mode="valid")))
    cb_var = float(np.var(convolve2d(cb_ch, lap, mode="valid")))
    cr_var = float(np.var(convolve2d(cr_ch, lap, mode="valid")))

    chroma_var = (cb_var + cr_var) / 2.0
    if luma_var < 1e-6:
        return 1.0
    return float(round(chroma_var / luma_var, 4))


def noise_estimate(img: Image.Image) -> float:
    """
    Estimate noise level using median absolute deviation of
    high-pass filtered image. Teams denoising should lower this.
    """
    gray = np.array(img.convert("L"), dtype=np.float64)
    # 3x3 high-pass via Laplacian
    from scipy.signal import convolve2d
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    filtered = convolve2d(gray, lap, mode="valid")
    # Robust noise estimate (MAD-based sigma)
    sigma = float(np.median(np.abs(filtered)) / 0.6745)
    return round(sigma, 3)


def color_histogram(img: Image.Image, bins: int = 64) -> dict:
    """Compute normalized color histograms per channel."""
    arr = np.array(img)
    result = {}
    if arr.ndim == 3 and arr.shape[2] >= 3:
        for ch, name in enumerate(["r", "g", "b"]):
            hist, _ = np.histogram(arr[:, :, ch], bins=bins, range=(0, 256))
            hist = hist.astype(np.float64) / max(hist.sum(), 1)
            result[name] = hist
    return result


def histogram_distance(hist_a: dict, hist_b: dict) -> dict:
    """Compute histogram distances (L1, chi-squared) between two histograms."""
    distances = {}
    for ch in ("r", "g", "b"):
        if ch in hist_a and ch in hist_b:
            a = hist_a[ch]
            b = hist_b[ch]
            # L1 distance
            l1 = float(np.sum(np.abs(a - b)))
            # Chi-squared distance
            denom = a + b
            denom[denom == 0] = 1
            chi2 = float(np.sum((a - b) ** 2 / denom))
            distances[f"{ch}_l1"] = round(l1, 4)
            distances[f"{ch}_chi2"] = round(chi2, 4)
    return distances


def frequency_energy(img: Image.Image) -> dict:
    """DCT-based frequency analysis."""
    if fftpack is None:
        return {}

    gray = np.array(img.convert("L"), dtype=np.float64)

    # 2D DCT
    dct = fftpack.dct(fftpack.dct(gray, axis=0, norm="ortho"), axis=1, norm="ortho")
    dct_abs = np.abs(dct)

    h, w = dct_abs.shape
    # Split into quadrants: low-freq (top-left quarter) vs high-freq (rest)
    qh, qw = h // 4, w // 4
    low_freq = dct_abs[:qh, :qw]
    high_freq_energy = float(np.sum(dct_abs) - np.sum(low_freq))
    total_energy = float(np.sum(dct_abs))

    return {
        "total_dct_energy": round(total_energy, 2),
        "high_freq_ratio": round(high_freq_energy / max(total_energy, 1), 4),
        "low_freq_energy": round(float(np.sum(low_freq)), 2),
    }


def jpeg_file_quality_estimate(file_path: str) -> dict:
    """Estimate JPEG quality from file size ratio."""
    file_size = os.path.getsize(file_path)
    img = Image.open(file_path)
    pixels = img.width * img.height
    # Bits per pixel
    bpp = (file_size * 8) / max(pixels, 1)
    return {
        "file_size_bytes": file_size,
        "pixels": pixels,
        "bits_per_pixel": round(bpp, 3),
    }


# ── GCS helpers ─────────────────────────────────────────────────────────────

def download_original_frames(bucket, sample_id: str, vid_type: str,
                             tmp_dir: str, max_frames: int = 4) -> list[str]:
    """
    Download a few original cropped frames from GCS for comparison.
    Returns list of local paths.
    """
    prefix = f"samples/{sample_id}/{vid_type}/"
    blobs = list(bucket.list_blobs(prefix=prefix, max_results=max_frames * 2))

    # Filter to .jpg/.png files only
    image_blobs = [b for b in blobs if b.name.endswith((".jpg", ".jpeg", ".png"))]

    if not image_blobs:
        return []

    # Take evenly spaced subset
    if len(image_blobs) > max_frames:
        step = len(image_blobs) / max_frames
        image_blobs = [image_blobs[int(i * step)] for i in range(max_frames)]

    local_paths = []
    for blob in image_blobs:
        fname = blob.name.replace("/", "_")
        local_path = os.path.join(tmp_dir, fname)
        blob.download_to_filename(local_path)
        local_paths.append(local_path)

    return local_paths


# ── Comparison ──────────────────────────────────────────────────────────────

def compare_pair(original_paths: list[str], teams_paths: list[str]) -> dict:
    """
    Compare a set of original frames vs Teams-augmented frames.
    Returns aggregated statistics.
    """
    orig_stats_list = []
    teams_stats_list = []

    for path in original_paths:
        img = Image.open(path).convert("RGB")
        stats = image_stats(img)
        stats.update(jpeg_file_quality_estimate(path))
        stats.update(frequency_energy(img))
        try:
            stats["sharpness"] = sharpness_score(img)
            stats["blockiness"] = blockiness_score(img)
            stats["chroma_blur_ratio"] = chroma_blur_ratio(img)
            stats["noise_level"] = noise_estimate(img)
        except Exception:
            pass
        orig_stats_list.append(stats)

    for path in teams_paths:
        img = Image.open(path).convert("RGB")
        stats = image_stats(img)
        stats.update(jpeg_file_quality_estimate(path))
        stats.update(frequency_energy(img))
        try:
            stats["sharpness"] = sharpness_score(img)
            stats["blockiness"] = blockiness_score(img)
            stats["chroma_blur_ratio"] = chroma_blur_ratio(img)
            stats["noise_level"] = noise_estimate(img)
        except Exception:
            pass
        teams_stats_list.append(stats)

    # Average stats
    def avg_stats(stats_list):
        if not stats_list:
            return {}
        result = {}
        for key in stats_list[0]:
            vals = [s[key] for s in stats_list if s.get(key) is not None and isinstance(s[key], (int, float))]
            if vals:
                result[key] = round(sum(vals) / len(vals), 4)
        return result

    orig_avg = avg_stats(orig_stats_list)
    teams_avg = avg_stats(teams_stats_list)

    # Compute deltas — only for Teams-relevant metrics
    # Skip metrics that reflect crop differences, not Teams processing
    SKIP_DELTA = {"width", "height", "aspect_ratio", "pixels"}
    comparison = {"original": orig_avg, "teams": teams_avg, "delta": {}}
    for key in orig_avg:
        if key in SKIP_DELTA:
            continue
        if key in teams_avg and orig_avg[key] != 0:
            delta = teams_avg[key] - orig_avg[key]
            pct = (delta / abs(orig_avg[key])) * 100 if orig_avg[key] != 0 else 0
            comparison["delta"][key] = {
                "absolute": round(delta, 4),
                "percent": round(pct, 2),
            }

    # Color histogram comparison
    if original_paths and teams_paths:
        orig_img = Image.open(original_paths[0]).convert("RGB")
        teams_img = Image.open(teams_paths[0]).convert("RGB")
        orig_hist = color_histogram(orig_img)
        teams_hist = color_histogram(teams_img)
        if orig_hist and teams_hist:
            comparison["histogram_distance"] = histogram_distance(orig_hist, teams_hist)

    return comparison


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_comparison_summary(all_comparisons: list[dict], output_dir: str):
    """Create summary plots of original vs Teams properties."""
    if not HAS_MPL or not all_comparisons:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect key metrics across all samples
    metrics = ["sharpness", "blockiness", "chroma_blur_ratio",
               "bits_per_pixel", "high_freq_ratio", "noise_level"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        orig_vals = []
        teams_vals = []
        for comp in all_comparisons:
            o = comp.get("original", {}).get(metric)
            t = comp.get("teams", {}).get(metric)
            if o is not None and t is not None:
                orig_vals.append(o)
                teams_vals.append(t)

        if orig_vals:
            ax.scatter(orig_vals, teams_vals, alpha=0.6, s=20)
            lo = min(min(orig_vals), min(teams_vals))
            hi = max(max(orig_vals), max(teams_vals))
            ax.plot([lo, hi], [lo, hi], "r--", alpha=0.5, label="y=x")
            ax.set_xlabel("Original")
            ax.set_ylabel("Teams")
            ax.set_title(metric.replace("_", " ").title())
            ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = output_path / "original_vs_teams_scatter.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved: {plot_path}")

    # Bar chart of average deltas
    delta_means = defaultdict(list)
    for comp in all_comparisons:
        for metric, d in comp.get("delta", {}).items():
            if "percent" in d:
                delta_means[metric].append(d["percent"])

    if delta_means:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = sorted(delta_means.keys())
        means = [np.mean(delta_means[k]) for k in labels]
        stds = [np.std(delta_means[k]) for k in labels]
        x = range(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=3, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("% Change (Teams vs Original)")
        ax.set_title("Teams Pipeline Effect on Frame Properties")
        ax.axhline(y=0, color="k", linewidth=0.5)
        plt.tight_layout()
        delta_path = output_path / "teams_delta_barchart.png"
        plt.savefig(delta_path, dpi=150)
        plt.close()
        print(f"Plot saved: {delta_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare original GCS frames vs Teams-augmented frames")
    parser.add_argument("--teams-dir", required=True,
                        help="Path to reconstructed teams_dataset/ directory")
    parser.add_argument("--assignment", default=None,
                        help="Path to assignment.json (default: teams_dir/assignment.json)")
    parser.add_argument("--output-dir", default="comparison_results",
                        help="Output directory for comparison results")
    parser.add_argument("--max-samples", type=int, default=20,
                        help="Max sample pairs to compare (default: 20)")
    parser.add_argument("--frames-per-sample", type=int, default=4,
                        help="Frames per sample to compare (default: 4)")
    args = parser.parse_args()

    teams_path = Path(args.teams_dir)
    assignment_path = args.assignment or str(teams_path / "assignment.json")

    print("=" * 60)
    print("Original vs Teams Frame Comparison")
    print("=" * 60)

    # Load assignment
    print(f"\nLoading assignment: {assignment_path}")
    with open(assignment_path) as f:
        assignment = json.load(f)
    print(f"  {len(assignment)} assigned frames")

    # Group by (sample_id, type)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for entry in assignment:
        key = (entry["sample_id"], entry["type"])
        groups[key].append(entry)

    # Unique samples
    sample_ids = sorted(set(e["sample_id"] for e in assignment))
    print(f"  {len(sample_ids)} unique samples")

    # Limit samples
    samples_to_compare = sample_ids[:args.max_samples]
    print(f"  Comparing first {len(samples_to_compare)} samples")

    # Connect to GCS
    print(f"\nConnecting to GCS bucket: {FRAMES_BUCKET}")
    client = gcs.Client(project=GCS_PROJECT)
    bucket = client.bucket(FRAMES_BUCKET)

    tmp_dir = tempfile.mkdtemp(prefix="compare_frames_")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_comparisons = []
    detailed_results = []

    for i, sample_id in enumerate(samples_to_compare):
        for vid_type in ("real", "fake"):
            key = (sample_id, vid_type)
            if key not in groups:
                continue

            entries = groups[key]
            teams_frame_paths = []
            for entry in entries[:args.frames_per_sample]:
                p = teams_path / entry["output_path"]
                if p.exists():
                    teams_frame_paths.append(str(p))

            if not teams_frame_paths:
                continue

            # Download originals
            orig_paths = download_original_frames(
                bucket, sample_id, vid_type, tmp_dir,
                max_frames=args.frames_per_sample)

            if not orig_paths:
                print(f"  ⚠ No originals found for {sample_id}/{vid_type}")
                continue

            # Compare
            comp = compare_pair(orig_paths, teams_frame_paths)
            comp["sample_id"] = sample_id
            comp["type"] = vid_type
            comp["n_original"] = len(orig_paths)
            comp["n_teams"] = len(teams_frame_paths)
            all_comparisons.append(comp)

            detailed_results.append({
                "sample_id": sample_id,
                "type": vid_type,
                "comparison": comp,
            })

            status = "✓" if comp.get("delta") else "⚠"
            print(f"  {status} [{i+1}/{len(samples_to_compare)}] "
                  f"{sample_id}/{vid_type}  "
                  f"({len(orig_paths)} orig, {len(teams_frame_paths)} teams)")

    # ── Summary ──
    if all_comparisons:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)

        # Aggregate deltas
        delta_agg: dict[str, list[float]] = defaultdict(list)
        for comp in all_comparisons:
            for metric, d in comp.get("delta", {}).items():
                delta_agg[metric].append(d["percent"])

        print(f"\n{'Metric':<25} {'Mean Δ%':>10} {'Std Δ%':>10} {'Direction'}")
        print("─" * 60)
        for metric in sorted(delta_agg.keys()):
            vals = delta_agg[metric]
            mean = np.mean(vals)
            std = np.std(vals)
            direction = "↑ Teams higher" if mean > 0 else "↓ Teams lower"
            print(f"{metric:<25} {mean:>+10.2f} {std:>10.2f}   {direction}")

        # Save detailed results
        results_path = output_dir / "comparison_details.json"
        with open(results_path, "w") as f:
            json.dump(detailed_results, f, indent=2, default=str)
        print(f"\nDetailed results: {results_path}")

        # Plots
        plot_comparison_summary(all_comparisons, str(output_dir))
    else:
        print("\n⚠ No comparisons made — check paths and data availability.")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
