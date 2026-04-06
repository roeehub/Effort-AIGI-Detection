#!/usr/bin/env python3
"""
Visual Demo: VideoCodecSimulation augmentation effect.

Loads REAL cached images from Phase 1 analysis (VCD, YouTube, DF40, Webcam),
applies the VideoCodecSimulation transform at multiple severity levels, and
produces publication-quality plots comparing:

  1. Before/after image grid (originals vs codec-simulated)
  2. Frequency spectrum shift — PSD slope and high-freq ratio distributions
  3. Distribution overlap: augmented DF40 reals vs VCD/YouTube/Webcam reals
  4. Radar chart of key metrics across all sources + augmented

Output:  analysis_results/plots/codec_simulation_demo/
         - 01_before_after_grid.png
         - 02_frequency_shift.png
         - 03_distribution_overlap.png
         - 04_radar_comparison.png

Does NOT require albumentations — implements the 5-step codec chain directly
from cv2/numpy so it runs in the base conda environment.
"""

import sys
import os
import cv2
import numpy as np
import random
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "weights" / "quality_analysis_samples"
OUTPUT_DIR = SCRIPT_DIR.parent.parent / "analysis_results" / "plots" / "codec_simulation_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "df40_real": "DF40 Training Reals",
    "youtube_real": "YouTube AVSpeech",
    "vcd_real": "Zoom VCD (OOD)",
    "real_or_virtual": "Webcam Tests",
}


# ─────────────────────────────────────────────
# VideoCodecSimulation — standalone (no albumentations needed)
# ─────────────────────────────────────────────
def apply_codec_simulation(img, codec_quality=50):
    """
    Apply the 5-step codec simulation chain.
    Mirrors VideoCodecSimulation.apply() but without albumentations dependency.
    
    Args:
        img: RGB uint8 numpy array (H, W, 3)
        codec_quality: Overall quality [0=worst, 100=best]
    
    Returns:
        Degraded image (same shape/dtype)
    """
    h, w = img.shape[:2]
    result = img.copy()
    
    q = codec_quality
    severity = 1.0 - (q / 100.0)  # 0=pristine, 1=worst
    
    # Step 1: Resolution reduction
    scale = random.uniform(0.5 + (1.0 - severity) * 0.1, 0.85)
    scale = min(scale, 1.0)
    if scale < 0.95:
        new_h, new_w = max(16, int(h * scale)), max(16, int(w * scale))
        small = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_AREA)
        interp = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC])
        result = cv2.resize(small, (w, h), interpolation=interp)
    
    # Step 2: Bilateral filter (deblocking)
    d = random.randint(5, 11)
    if d % 2 == 0:
        d += 1
    sigma_c = random.uniform(40, 90) * (0.5 + 0.5 * severity)
    sigma_s = random.uniform(40, 90) * (0.5 + 0.5 * severity)
    result = cv2.bilateralFilter(result, d, sigma_c, sigma_s)
    
    # Step 3: Block quantization artifacts
    bs = 8
    strength = random.uniform(0.3, 1.5) * severity
    if strength > 0.05 and h > bs * 2 and w > bs * 2:
        result_f = result.astype(np.float32)
        for col in range(bs, w - 1, bs):
            noise_col = np.random.normal(0, strength, (h, 1, 3))
            result_f[:, col:col+1] += noise_col.astype(np.float32)
        for row in range(bs, h - 1, bs):
            noise_row = np.random.normal(0, strength, (1, w, 3))
            result_f[row:row+1, :] += noise_row.astype(np.float32)
        result = np.clip(result_f, 0, 255).astype(np.uint8)
    
    # Step 4: Frequency-shaped codec noise
    noise_std = random.uniform(3.0, 12.0) * severity
    if noise_std > 0.5:
        noise = np.random.normal(0, noise_std, result.shape).astype(np.float32)
        blur_k = random.choice([3, 5, 7])
        noise_lf = cv2.GaussianBlur(noise, (blur_k, blur_k), 0)
        noise_hf = noise - noise_lf
        current_std = noise_hf.std()
        if current_std > 0.1:
            noise_hf = noise_hf * (noise_std / current_std)
        result = np.clip(result.astype(np.float32) + noise_hf, 0, 255).astype(np.uint8)
    
    # Step 5: JPEG I-frame compression
    jpeg_q = int(q * 0.8 + 10)
    jpeg_q = max(25, min(95, jpeg_q))
    _, enc = cv2.imencode('.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
    result = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    
    return result


# ─────────────────────────────────────────────
# Quality metrics (same as Phase 1)
# ─────────────────────────────────────────────
def compute_quality_metrics(img_rgb):
    """Compute the key frequency/sharpness metrics we care about."""
    h, w = img_rgb.shape[:2]
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    props = {}
    
    # Sharpness
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    props["sharpness_tenengrad"] = float(np.sqrt(gx**2 + gy**2).mean())
    
    # Edge density
    edges = cv2.Canny(img_gray, 100, 200)
    props["edge_density"] = float(edges.sum() / 255.0) / float(w * h)
    
    # Frequency spectrum (224×224 basis)
    gray_224 = cv2.resize(img_gray, (224, 224))
    f_transform = np.fft.fft2(gray_224.astype(np.float64))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    Y, X = np.ogrid[:224, :224]
    radius = np.sqrt((X - 112)**2 + (Y - 112)**2)
    total = magnitude.sum()
    if total > 0:
        props["freq_low_ratio"] = float(magnitude[radius < 20].sum() / total)
        props["freq_high_ratio"] = float(magnitude[radius >= 60].sum() / total)
        props["freq_high_to_low"] = float(
            magnitude[radius >= 60].sum() / max(magnitude[radius < 20].sum(), 1e-8)
        )
    
    # PSD slope (vectorized — no nested Python loops)
    max_r = int(np.sqrt(2) * 112)
    r_int = radius.astype(int)
    r_flat = r_int.ravel()
    mag_flat = magnitude.ravel()
    valid_mask = r_flat < max_r
    radial_profile = np.bincount(r_flat[valid_mask], weights=mag_flat[valid_mask], minlength=max_r).astype(np.float64)
    radial_count = np.bincount(r_flat[valid_mask], minlength=max_r).astype(np.float64)
    radial_count[radial_count == 0] = 1
    radial_profile /= radial_count
    
    valid = (radial_profile[5:100] > 0)
    if valid.sum() > 10:
        freqs = np.arange(5, 100)[valid]
        power = radial_profile[5:100][valid]
        slope, _, r_value, _, _ = stats.linregress(np.log10(freqs), np.log10(power))
        props["psd_slope"] = float(slope)
        props["psd_r_squared"] = float(r_value ** 2)
    else:
        props["psd_slope"] = 0.0
        props["psd_r_squared"] = 0.0
    
    # Effective resolution
    sorted_mag = np.sort(magnitude.flatten())[::-1]
    cumsum = np.cumsum(sorted_mag)
    if cumsum[-1] > 0:
        idx_90 = np.searchsorted(cumsum, 0.90 * cumsum[-1])
        props["effective_resolution_90pct"] = float(idx_90) / float(len(sorted_mag))
    
    # JPEG compressibility
    _, jpg_buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 95])
    props["jpeg_compressibility"] = float(len(jpg_buf)) / float(w * h * 3)
    
    # Blockiness
    if h > 16 and w > 16:
        gray_f = img_gray.astype(np.float64)
        h_bound_cols = np.arange(8, w - 1, 8)
        h_non_bound_cols = np.array([c for c in range(1, w - 1) if c % 8 != 0])
        if len(h_bound_cols) > 0 and len(h_non_bound_cols) > 0:
            h_bd = np.abs(gray_f[:, h_bound_cols] - gray_f[:, h_bound_cols - 1]).mean()
            rng = np.random.RandomState(0)
            sc = rng.choice(h_non_bound_cols, size=min(len(h_non_bound_cols), len(h_bound_cols)*3), replace=False)
            h_nb = np.abs(gray_f[:, sc] - gray_f[:, sc - 1]).mean()
            props["blockiness_h_ratio"] = float(h_bd / max(h_nb, 1e-8))
        else:
            props["blockiness_h_ratio"] = 1.0
    
    return props


# ─────────────────────────────────────────────
# Load cached images
# ─────────────────────────────────────────────
def load_cached_images(source_key, max_n=100):
    """Load images from Phase 1 cache directory."""
    src_dir = CACHE_DIR / source_key
    if not src_dir.exists():
        print(f"  WARNING: Cache dir not found: {src_dir}")
        return []
    
    files = sorted(src_dir.iterdir())
    files = [f for f in files if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if len(files) > max_n:
        rng = random.Random(42)
        files = rng.sample(files, max_n)
    
    images = []
    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)
    return images


def compute_metrics_batch(images, label=""):
    """Compute quality metrics for a batch of images."""
    results = []
    for i, img in enumerate(images):
        if label:
            print(f"\r  {label}: {i+1}/{len(images)}", end="", flush=True)
        results.append(compute_quality_metrics(img))
    if label:
        print()
    return results


# ─────────────────────────────────────────────
# PLOT 1: Before/After Image Grid
# ─────────────────────────────────────────────
def plot_before_after_grid(images_by_source, n_show=3):
    """Show original images alongside codec-simulated versions at 3 severity levels."""
    print("Generating Plot 1: Before/After Grid...")
    
    # Pick images from different sources
    samples = []
    for source_key in ["df40_real", "vcd_real", "youtube_real"]:
        imgs = images_by_source.get(source_key, [])
        if len(imgs) >= n_show:
            rng = random.Random(123)
            samples.extend([(img, SOURCES[source_key]) for img in rng.sample(imgs, 1)])
    
    if not samples:
        print("  No images to show!")
        return
    
    severities = [
        ("Original", None),
        ("Light\n(q=70)", 70),
        ("Moderate\n(q=50)", 50),
        ("Heavy\n(q=25)", 25),
    ]
    
    n_rows = len(samples)
    n_cols = len(severities)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for row, (img, src_label) in enumerate(samples):
        for col, (sev_label, q) in enumerate(severities):
            ax = axes[row, col]
            if q is None:
                show_img = img
            else:
                show_img = apply_codec_simulation(img, codec_quality=q)
            
            ax.imshow(show_img)
            ax.set_xticks([])
            ax.set_yticks([])
            
            if row == 0:
                ax.set_title(sev_label, fontsize=13, fontweight='bold', pad=10)
            if col == 0:
                ax.set_ylabel(src_label, fontsize=11, rotation=90, labelpad=15)
    
    fig.suptitle("VideoCodecSimulation — Before & After at Multiple Severity Levels",
                 fontsize=15, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUTPUT_DIR / "01_before_after_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────
# PLOT 2: Frequency Shift Distributions
# ─────────────────────────────────────────────
def plot_frequency_shift(df40_images, n_aug=80):
    """Show how augmentation shifts the frequency profile of DF40 reals."""
    print("Generating Plot 2: Frequency Shift...")
    
    # Compute metrics for original DF40 images
    subset = df40_images[:n_aug]
    print(f"  Computing metrics for {len(subset)} original DF40 images...")
    orig_metrics = compute_metrics_batch(subset, "Original")
    
    # Apply codec simulation at different quality levels and compute metrics
    aug_metrics_by_q = {}
    for q_label, q_range in [("Light (q∈[60,80])", (60, 80)),
                              ("Moderate (q∈[35,55])", (35, 55)),
                              ("Heavy (q∈[15,35])", (15, 35))]:
        print(f"  Applying {q_label}...")
        aug_metrics = []
        for img in subset:
            q = random.randint(q_range[0], q_range[1])
            aug_img = apply_codec_simulation(img, codec_quality=q)
            aug_metrics.append(compute_quality_metrics(aug_img))
        aug_metrics_by_q[q_label] = aug_metrics
    
    # Plot distributions for key metrics
    key_metrics = [
        ("psd_slope", "PSD Slope", "flatter = more codec-like"),
        ("freq_high_ratio", "High-Freq Ratio", "higher = more HF noise"),
        ("sharpness_tenengrad", "Tenengrad Sharpness", "lower = softer edges"),
        ("edge_density", "Edge Density", "lower = fewer sharp edges"),
        ("effective_resolution_90pct", "Effective Resolution (90%)", "higher = more dispersed energy"),
        ("jpeg_compressibility", "JPEG Compressibility", "higher = more redundancy"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors_aug = ['#2196F3', '#FF9800', '#F44336']
    
    for idx, (metric, title, subtitle) in enumerate(key_metrics):
        ax = axes[idx]
        
        # Original
        orig_vals = [m[metric] for m in orig_metrics if metric in m]
        ax.hist(orig_vals, bins=25, alpha=0.6, color='#4CAF50', label='DF40 Original',
                density=True, edgecolor='white', linewidth=0.5)
        
        # Augmented versions
        for (q_label, metrics_list), color in zip(aug_metrics_by_q.items(), colors_aug):
            vals = [m[metric] for m in metrics_list if metric in m]
            ax.hist(vals, bins=25, alpha=0.4, color=color, label=q_label,
                    density=True, edgecolor='white', linewidth=0.5)
        
        ax.set_title(f"{title}\n({subtitle})", fontsize=11, fontweight='bold')
        ax.set_xlabel(metric.replace("_", " "), fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        
        if idx == 0:
            ax.legend(fontsize=8, loc='upper left')
    
    fig.suptitle("Frequency Shift: DF40 Reals → After Codec Simulation\n"
                 "Green = original training reals | Colors = augmented at 3 severity levels",
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = OUTPUT_DIR / "02_frequency_shift.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────
# PLOT 3: Distribution Overlap with OOD Sources
# ─────────────────────────────────────────────
def plot_distribution_overlap(metrics_by_source, aug_metrics):
    """Compare augmented DF40 reals against VCD/YouTube/Webcam distributions."""
    print("Generating Plot 3: Distribution Overlap...")
    
    key_metrics = [
        ("psd_slope", "PSD Slope"),
        ("freq_high_ratio", "High-Freq Energy Ratio"),
        ("sharpness_tenengrad", "Tenengrad Sharpness"),
        ("edge_density", "Edge Density"),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    source_colors = {
        "df40_real": ("#4CAF50", "DF40 Original"),
        "vcd_real": ("#F44336", "Zoom VCD (OOD)"),
        "youtube_real": ("#2196F3", "YouTube (OOD)"),
        "real_or_virtual": ("#9C27B0", "Webcam Tests"),
    }
    
    for idx, (metric, title) in enumerate(key_metrics):
        ax = axes[idx]
        
        # Plot each source
        for source_key, (color, label) in source_colors.items():
            if source_key in metrics_by_source:
                vals = [m[metric] for m in metrics_by_source[source_key] if metric in m]
                if vals:
                    ax.hist(vals, bins=30, alpha=0.35, color=color, label=label,
                            density=True, edgecolor='white', linewidth=0.5)
        
        # Augmented DF40 (the whole point — does it overlap with VCD?)
        aug_vals = [m[metric] for m in aug_metrics if metric in m]
        ax.hist(aug_vals, bins=30, alpha=0.5, color='#FF9800', label='DF40 + Codec Sim',
                density=True, edgecolor='black', linewidth=2.5, histtype='step', linestyle='--')
        # Also filled with very low alpha
        ax.hist(aug_vals, bins=30, alpha=0.15, color='#FF9800',
                density=True, edgecolor='none')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(metric.replace("_", " "), fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        
        if idx == 0:
            ax.legend(fontsize=8, loc='best')
    
    fig.suptitle(
        "Distribution Overlap: Can Codec Simulation Bridge the DF40→VCD Gap?\n"
        "Orange dashed = DF40 reals AFTER codec simulation (moderate, q∈[30,60])\n"
        "Goal: orange should overlap with red (VCD) = model sees VCD-like inputs during training",
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out_path = OUTPUT_DIR / "03_distribution_overlap.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────
# PLOT 4: Radar Chart Comparison
# ─────────────────────────────────────────────
def plot_radar_comparison(metrics_by_source, aug_metrics):
    """Spider/radar chart comparing mean metrics across sources."""
    print("Generating Plot 4: Radar Comparison...")
    
    radar_metrics = [
        "psd_slope", "freq_high_ratio", "freq_low_ratio", 
        "sharpness_tenengrad", "edge_density",
        "effective_resolution_90pct", "jpeg_compressibility",
    ]
    radar_labels = [
        "PSD Slope\n(flatter→)", "High-Freq\nRatio ↑", "Low-Freq\nRatio ↓",
        "Tenengrad\nSharpness", "Edge\nDensity",
        "Effective\nResolution", "JPEG\nCompressibility",
    ]
    
    # Compute means for each source
    source_means = {}
    for source_key, metrics_list in metrics_by_source.items():
        means = {}
        for m in radar_metrics:
            vals = [x[m] for x in metrics_list if m in x]
            means[m] = np.mean(vals) if vals else 0
        source_means[source_key] = means
    
    # Augmented means
    aug_means = {}
    for m in radar_metrics:
        vals = [x[m] for x in aug_metrics if m in x]
        aug_means[m] = np.mean(vals) if vals else 0
    
    # Normalize all values to [0, 1] range for radar chart
    all_vals = {}
    for m in radar_metrics:
        all_v = [source_means[s][m] for s in source_means] + [aug_means[m]]
        all_vals[m] = (min(all_v), max(all_v))
    
    def normalize(means):
        normed = []
        for m in radar_metrics:
            vmin, vmax = all_vals[m]
            if vmax > vmin:
                normed.append((means[m] - vmin) / (vmax - vmin))
            else:
                normed.append(0.5)
        return normed
    
    # Setup radar
    N = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    
    # Plot each source
    plot_configs = [
        ("df40_real", "#4CAF50", "DF40 Training Reals", "-", 1.5),
        ("vcd_real", "#F44336", "Zoom VCD (OOD)", "-", 2.0),
        ("youtube_real", "#2196F3", "YouTube (OOD)", "-", 1.5),
        ("real_or_virtual", "#9C27B0", "Webcam Tests", "--", 1.5),
    ]
    
    for source_key, color, label, ls, lw in plot_configs:
        if source_key in source_means:
            vals = normalize(source_means[source_key])
            vals += vals[:1]
            ax.plot(angles, vals, color=color, linewidth=lw, linestyle=ls, label=label)
            ax.fill(angles, vals, color=color, alpha=0.08)
    
    # Augmented
    aug_normed = normalize(aug_means)
    aug_normed += aug_normed[:1]
    ax.plot(angles, aug_normed, color='#FF9800', linewidth=2.5, linestyle='--',
            label='DF40 + Codec Sim', marker='o', markersize=6)
    ax.fill(angles, aug_normed, color='#FF9800', alpha=0.12)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color='grey')
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
    
    ax.set_title("Quality Fingerprint Radar — All Real Sources + Augmented\n"
                 "Orange dashed should approach Red (VCD) while staying distinct from Green (DF40 original)",
                 fontsize=12, fontweight='bold', pad=30)
    
    out_path = OUTPUT_DIR / "04_radar_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────
# PLOT 5: Single-Image Deep Dive 
# ─────────────────────────────────────────────
def plot_single_image_deep_dive(img, source_label="DF40 Real"):
    """Show one image at 5 severity levels with frequency spectrum visualization."""
    print("Generating Plot 5: Single-Image Deep Dive...")
    
    qualities = [("Original", None), ("q=80", 80), ("q=60", 60), ("q=40", 40), ("q=20", 20)]
    
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(2, 5, height_ratios=[1.2, 1], hspace=0.3)
    
    for col, (label, q) in enumerate(qualities):
        # Apply codec sim
        if q is None:
            result = img
        else:
            result = apply_codec_simulation(img, codec_quality=q)
        
        # Top row: image
        ax_img = fig.add_subplot(gs[0, col])
        ax_img.imshow(result)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        
        # Compute metrics for annotation
        m = compute_quality_metrics(result)
        ax_img.set_title(f"{label}\nPSD={m.get('psd_slope', 0):.3f}  HF={m.get('freq_high_ratio', 0):.3f}",
                         fontsize=10, fontweight='bold')
        
        # Bottom row: frequency spectrum (log magnitude)
        ax_fft = fig.add_subplot(gs[1, col])
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        gray_224 = cv2.resize(gray, (224, 224))
        f_shift = np.fft.fftshift(np.fft.fft2(gray_224.astype(np.float64)))
        log_mag = np.log1p(np.abs(f_shift))
        ax_fft.imshow(log_mag, cmap='inferno', vmin=log_mag.min(), vmax=log_mag.max() * 0.7)
        ax_fft.set_xticks([])
        ax_fft.set_yticks([])
        ax_fft.set_xlabel("FFT Magnitude", fontsize=9)
    
    fig.suptitle(f"Single-Image Deep Dive: {source_label}\n"
                 f"Top: visual effect | Bottom: FFT spectrum | "
                 f"Note how codec sim flattens PSD slope and raises HF ratio",
                 fontsize=13, fontweight='bold')
    
    out_path = OUTPUT_DIR / "05_single_image_deep_dive.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
def main():
    print("=" * 70)
    print("VideoCodecSimulation — Visual Demo")
    print("=" * 70)
    print()
    
    # Load cached images from Phase 1
    print("Loading cached images from Phase 1 analysis...")
    images_by_source = {}
    for source_key in SOURCES:
        imgs = load_cached_images(source_key, max_n=100)
        print(f"  {source_key}: {len(imgs)} images loaded")
        images_by_source[source_key] = imgs
    
    total = sum(len(v) for v in images_by_source.values())
    if total == 0:
        print("\nERROR: No cached images found. Run analyze_real_quality_distributions.py first.")
        sys.exit(1)
    print(f"\nTotal: {total} images loaded\n")
    
    # Compute metrics for all sources
    print("Computing quality metrics for each source...")
    metrics_by_source = {}
    for source_key, imgs in images_by_source.items():
        metrics_by_source[source_key] = compute_metrics_batch(imgs, SOURCES[source_key])
    
    # Apply codec simulation to DF40 reals (the training set)
    print("\nApplying codec simulation to DF40 reals (moderate: q∈[30,60])...")
    df40_imgs = images_by_source.get("df40_real", [])
    aug_metrics = []
    for i, img in enumerate(df40_imgs):
        print(f"\r  Augmenting: {i+1}/{len(df40_imgs)}", end="", flush=True)
        q = random.randint(30, 60)
        aug_img = apply_codec_simulation(img, codec_quality=q)
        aug_metrics.append(compute_quality_metrics(aug_img))
    print()
    
    # Generate all plots
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70 + "\n")
    
    plot_before_after_grid(images_by_source)
    plot_frequency_shift(df40_imgs, n_aug=min(50, len(df40_imgs)))
    plot_distribution_overlap(metrics_by_source, aug_metrics)
    plot_radar_comparison(metrics_by_source, aug_metrics)
    
    # Single-image deep dive (use a DF40 real)
    if df40_imgs:
        plot_single_image_deep_dive(df40_imgs[0], "DF40 Training Real")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: Does Codec Simulation Bridge the Gap?")
    print("=" * 70)
    
    key_metrics = ["psd_slope", "freq_high_ratio", "sharpness_tenengrad", "edge_density",
                   "effective_resolution_90pct"]
    
    header = f"{'Metric':<30s} {'DF40 Orig':>12s} {'DF40+Codec':>12s} {'VCD (OOD)':>12s} {'YouTube':>12s} {'Webcam':>12s}"
    print(header)
    print("-" * len(header))
    
    for metric in key_metrics:
        vals = {}
        for source_key, ml in metrics_by_source.items():
            v = [m[metric] for m in ml if metric in m]
            vals[source_key] = np.mean(v) if v else float('nan')
        aug_v = [m[metric] for m in aug_metrics if metric in m]
        aug_mean = np.mean(aug_v) if aug_v else float('nan')
        
        print(f"{metric:<30s} {vals.get('df40_real', float('nan')):>12.4f} "
              f"{aug_mean:>12.4f} "
              f"{vals.get('vcd_real', float('nan')):>12.4f} "
              f"{vals.get('youtube_real', float('nan')):>12.4f} "
              f"{vals.get('real_or_virtual', float('nan')):>12.4f}")
    
    print(f"\nPlots saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
