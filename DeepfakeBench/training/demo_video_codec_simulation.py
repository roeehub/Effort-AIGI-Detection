#!/usr/bin/env python3
"""
Visual demo of VideoCodecSimulation transform.

Produces a multi-panel figure showing:
  1. Before/after image comparison at multiple severity levels
  2. PSD (Power Spectral Density) curves overlay — original vs codec sim
  3. Frequency metric bar charts comparing codec sim output to VCD/YouTube/DF40 reference
  4. Severity sweep: how metrics shift as codec quality decreases

Runs locally — uses only cv2, numpy, scipy, matplotlib (NO albumentations needed).
The core codec simulation logic is extracted directly from the transform class.

Usage:
    cd DeepfakeBench/training
    python demo_video_codec_simulation.py [--output demo_codec_sim.png] [--seed 42]
"""

import argparse
import os
import sys
import cv2
import numpy as np
import random
from scipy import stats

# ─── Output directory ────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "analysis_results", "plots", "codec_sim_demo")


# ═══════════════════════════════════════════════════════════════════════════════
# Core codec simulation logic (extracted from VideoCodecSimulation.apply)
# No albumentations dependency — pure cv2/numpy
# ═══════════════════════════════════════════════════════════════════════════════

def apply_codec_simulation(
    img,
    codec_quality=50,
    downscale_range=(0.5, 0.85),
    bilateral_d=(5, 11),
    bilateral_sigma_color=(40, 90),
    bilateral_sigma_space=(40, 90),
    block_size=8,
    block_strength=(0.3, 1.5),
    codec_noise_std=(3.0, 12.0),
    jpeg_quality=None,
):
    """Apply full codec simulation chain. codec_quality: 0=worst, 100=best."""
    h, w = img.shape[:2]
    result = img.copy()
    q = codec_quality
    severity = 1.0 - (q / 100.0)

    # Step 1: Resolution reduction
    scale = random.uniform(
        downscale_range[0] + (1.0 - severity) * 0.1,
        downscale_range[1],
    )
    scale = min(scale, 1.0)
    if scale < 0.95:
        new_h, new_w = max(16, int(h * scale)), max(16, int(w * scale))
        small = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_AREA)
        interp = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC])
        result = cv2.resize(small, (w, h), interpolation=interp)

    # Step 2: Bilateral filter (deblocking)
    d = random.randint(bilateral_d[0], bilateral_d[1])
    if d % 2 == 0:
        d += 1
    sigma_c = random.uniform(bilateral_sigma_color[0], bilateral_sigma_color[1])
    sigma_s = random.uniform(bilateral_sigma_space[0], bilateral_sigma_space[1])
    sigma_c *= (0.5 + 0.5 * severity)
    sigma_s *= (0.5 + 0.5 * severity)
    result = cv2.bilateralFilter(result, d, sigma_c, sigma_s)

    # Step 3: Block quantization artifacts
    bs = block_size
    strength = random.uniform(block_strength[0], block_strength[1]) * severity
    if strength > 0.05 and h > bs * 2 and w > bs * 2:
        result_f = result.astype(np.float32)
        for col in range(bs, w - 1, bs):
            noise_col = np.random.normal(0, strength, (h, 1, result.shape[2] if result.ndim == 3 else 1))
            if result.ndim == 2:
                noise_col = noise_col[:, :, 0]
            result_f[:, col:col + 1] += noise_col.astype(np.float32)
        for row in range(bs, h - 1, bs):
            noise_row = np.random.normal(0, strength, (1, w, result.shape[2] if result.ndim == 3 else 1))
            if result.ndim == 2:
                noise_row = noise_row[:, :, 0]
            result_f[row:row + 1, :] += noise_row.astype(np.float32)
        result = np.clip(result_f, 0, 255).astype(np.uint8)

    # Step 4: Codec noise (frequency-shaped)
    n_std = random.uniform(codec_noise_std[0], codec_noise_std[1]) * severity
    if n_std > 0.5:
        noise = np.random.normal(0, n_std, result.shape).astype(np.float32)
        blur_k = random.choice([3, 5, 7])
        noise_lf = cv2.GaussianBlur(noise, (blur_k, blur_k), 0)
        noise_hf = noise - noise_lf
        current_std = noise_hf.std()
        if current_std > 0.1:
            noise_hf = noise_hf * (n_std / current_std)
        result = np.clip(result.astype(np.float32) + noise_hf, 0, 255).astype(np.uint8)

    # Step 5: JPEG compression (I-frame)
    if jpeg_quality is not None:
        jpeg_q = jpeg_quality
    else:
        jpeg_q = int(q * 0.8 + 10)
        jpeg_q = max(25, min(95, jpeg_q))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q]
    _, enc = cv2.imencode('.jpg', result, encode_param)
    result = cv2.imdecode(enc, cv2.IMREAD_COLOR if result.ndim == 3 else cv2.IMREAD_GRAYSCALE)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Frequency analysis helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_radial_psd(img_gray, size=224):
    """Compute radial PSD profile and slope."""
    gray = cv2.resize(img_gray, (size, size))
    f_transform = np.fft.fft2(gray.astype(np.float64))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift) ** 2  # power

    cy, cx = size // 2, size // 2
    max_r = int(np.sqrt(2) * cx)
    Y, X = np.ogrid[:size, :size]
    r_map = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)

    radial_profile = np.zeros(max_r)
    radial_count = np.zeros(max_r)
    for ry in range(size):
        for rx in range(size):
            r = r_map[ry, rx]
            if r < max_r:
                radial_profile[r] += magnitude[ry, rx]
                radial_count[r] += 1
    radial_count[radial_count == 0] = 1
    radial_profile /= radial_count

    # Compute slope on log-log scale
    rng = slice(5, 100)
    valid = radial_profile[rng] > 0
    slope = 0.0
    if valid.sum() > 10:
        freqs = np.arange(5, 100)[valid]
        power = radial_profile[rng][valid]
        slope, _, _, _, _ = stats.linregress(np.log10(freqs), np.log10(power))

    return radial_profile, slope


def compute_freq_ratios(img_gray, size=224):
    """Compute low/mid/high frequency energy ratios."""
    gray = cv2.resize(img_gray, (size, size))
    f_transform = np.fft.fft2(gray.astype(np.float64))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    Y, X = np.ogrid[:size, :size]
    radius = np.sqrt((X - size // 2) ** 2 + (Y - size // 2) ** 2)
    total = magnitude.sum()
    if total == 0:
        return 0.0, 0.0, 0.0

    low = float(magnitude[radius < 30].sum() / total)
    mid = float(magnitude[(radius >= 30) & (radius < 60)].sum() / total)
    high = float(magnitude[radius >= 60].sum() / total)
    return low, mid, high


def compute_sharpness(img_gray):
    """Tenengrad sharpness (Sobel gradient magnitude)."""
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def compute_all_metrics(img):
    """Compute all frequency metrics for an image."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, slope = compute_radial_psd(gray)
    low, mid, high = compute_freq_ratios(gray)
    sharp = compute_sharpness(gray)
    return {
        "psd_slope": slope,
        "freq_low_ratio": low,
        "freq_mid_ratio": mid,
        "freq_high_ratio": high,
        "freq_high_to_low": high / max(low, 1e-10),
        "sharpness_tenengrad": sharp,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic face image generator
# ═══════════════════════════════════════════════════════════════════════════════

def make_synthetic_face(size=224, seed=42):
    """
    Create a realistic-ish synthetic face image with skin tones, features,
    and texture. Enough structure for meaningful frequency analysis.
    """
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Base skin-tone gradient
    for c, base in enumerate([140, 175, 210]):  # BGR: blueish, warm skin
        grad_h = np.linspace(base - 20, base + 20, size).reshape(-1, 1)
        grad_w = np.linspace(base - 10, base + 10, size).reshape(1, -1)
        img[:, :, c] = np.clip((grad_h + grad_w) / 2, 0, 255).astype(np.uint8)

    # Face oval
    cv2.ellipse(img, (size // 2, size // 2), (size // 3, size // 2 - 20),
                0, 0, 360, (160, 190, 220), -1)

    # Eyes
    for ex in [size // 2 - 30, size // 2 + 30]:
        cv2.ellipse(img, (ex, size // 2 - 20), (18, 10), 0, 0, 360, (80, 60, 50), -1)
        cv2.circle(img, (ex, size // 2 - 20), 7, (40, 30, 20), -1)
        cv2.circle(img, (ex + 2, size // 2 - 22), 3, (220, 220, 230), -1)

    # Eyebrows
    for ex in [size // 2 - 30, size // 2 + 30]:
        cv2.ellipse(img, (ex, size // 2 - 38), (22, 5), 0, 0, 180, (70, 50, 40), 2)

    # Nose
    pts = np.array([[size // 2, size // 2 - 5],
                     [size // 2 - 8, size // 2 + 20],
                     [size // 2 + 8, size // 2 + 20]], dtype=np.int32)
    cv2.polylines(img, [pts], False, (130, 160, 180), 2)

    # Mouth
    cv2.ellipse(img, (size // 2, size // 2 + 40), (25, 10), 0, 0, 180, (100, 120, 180), 2)
    cv2.ellipse(img, (size // 2, size // 2 + 40), (25, 8), 0, 180, 360, (120, 100, 90), 1)

    # Hair region (top)
    cv2.rectangle(img, (size // 2 - size // 3, 0), (size // 2 + size // 3, size // 4), (40, 35, 30), -1)
    cv2.ellipse(img, (size // 2, size // 4), (size // 3, size // 6), 0, 0, 180, (40, 35, 30), -1)

    # Texture noise (simulates skin pores / fine details)
    noise = rng.normal(0, 6, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Slight Gaussian blur to make it look natural
    img = cv2.GaussianBlur(img, (3, 3), 0.5)

    return img


# ═══════════════════════════════════════════════════════════════════════════════
# Reference values from Phase 1 quality fingerprint analysis
# ═══════════════════════════════════════════════════════════════════════════════

REFERENCE_DATA = {
    "DF40 Reals\n(training)": {
        "psd_slope": -2.071,
        "freq_high_ratio": 0.145,
        "freq_high_to_low": 0.258,
        "sharpness_tenengrad": None,  # not directly comparable with synthetic
    },
    "YouTube\nAVSpeech": {
        "psd_slope": -1.942,
        "freq_high_ratio": 0.166,
        "freq_high_to_low": 0.335,
        "sharpness_tenengrad": None,
    },
    "Webcam\nTests": {
        "psd_slope": -1.742,
        "freq_high_ratio": 0.228,
        "freq_high_to_low": 0.526,
        "sharpness_tenengrad": None,
    },
    "Zoom VCD\n(target)": {
        "psd_slope": -1.665,
        "freq_high_ratio": 0.249,
        "freq_high_to_low": 0.609,
        "sharpness_tenengrad": None,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def create_demo_figure(seed=42, output_path=None):
    """Create the full multi-panel demo figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch

    random.seed(seed)
    np.random.seed(seed)

    # Generate base image
    img_orig = make_synthetic_face(224, seed)

    # Apply codec sim at 3 severity levels
    severities = {
        "Light\n(q=70)": 70,
        "Moderate\n(q=45)": 45,
        "Heavy\n(q=20)": 20,
    }

    results = {}
    for label, q in severities.items():
        # Average over multiple runs for stable metrics
        metrics_list = []
        sample_img = None
        for trial in range(20):
            random.seed(seed + trial * 100)
            np.random.seed(seed + trial * 100)
            out = apply_codec_simulation(img_orig, codec_quality=q)
            if trial == 0:
                sample_img = out
            metrics_list.append(compute_all_metrics(out))

        avg_metrics = {}
        for key in metrics_list[0]:
            avg_metrics[key] = np.mean([m[key] for m in metrics_list])

        results[label] = {"image": sample_img, "quality": q, "metrics": avg_metrics}

    orig_metrics = compute_all_metrics(img_orig)

    # ── Figure layout ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 16), facecolor="white")
    fig.suptitle(
        "VideoCodecSimulation — Visual Demo & Frequency Analysis",
        fontsize=18, fontweight="bold", y=0.98,
    )
    fig.text(0.5, 0.955,
             "Simulating webcam/video-call codec artifacts to bridge the VCD quality gap",
             ha="center", fontsize=12, color="gray")

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35,
                           top=0.92, bottom=0.06, left=0.06, right=0.96)

    # ── Row 1: Before/After images ───────────────────────────────────
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    ax_orig.set_title("Original", fontsize=13, fontweight="bold")
    ax_orig.axis("off")

    for i, (label, data) in enumerate(results.items()):
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(cv2.cvtColor(data["image"], cv2.COLOR_BGR2RGB))
        ax.set_title(f"Codec Sim {label}", fontsize=12, fontweight="bold")
        ax.axis("off")

    # ── Row 2 left: PSD curves overlay ───────────────────────────────
    ax_psd = fig.add_subplot(gs[1, 0:2])

    gray_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    prof_orig, slope_orig = compute_radial_psd(gray_orig)

    colors_sev = ["#2196F3", "#FF9800", "#E53935"]
    ax_psd.loglog(np.arange(1, len(prof_orig)), prof_orig[1:],
                  color="black", linewidth=2.5, label=f"Original (slope={slope_orig:.2f})",
                  alpha=0.9)

    for (label, data), color in zip(results.items(), colors_sev):
        gray = cv2.cvtColor(data["image"], cv2.COLOR_BGR2GRAY)
        prof, slope = compute_radial_psd(gray)
        ax_psd.loglog(np.arange(1, len(prof)), prof[1:],
                      color=color, linewidth=1.8,
                      label=f"{label.replace(chr(10), ' ')} (slope={slope:.2f})",
                      alpha=0.8)

    ax_psd.set_xlabel("Spatial Frequency (cycles/image)", fontsize=11)
    ax_psd.set_ylabel("Power", fontsize=11)
    ax_psd.set_title("Power Spectral Density — Original vs Codec Sim", fontsize=13, fontweight="bold")
    ax_psd.legend(fontsize=9, loc="upper right")
    ax_psd.set_xlim(1, 150)
    ax_psd.grid(True, alpha=0.3)

    # ── Row 2 right: Frequency metric comparison bars ─────────────────
    ax_bars = fig.add_subplot(gs[1, 2:4])

    metric_keys = ["psd_slope", "freq_high_ratio", "freq_high_to_low"]
    metric_labels = ["PSD Slope\n(less negative = flatter)", "High-Freq Ratio\n(higher = more HF energy)",
                     "High/Low Ratio\n(higher = codec-like)"]
    n_groups = len(metric_keys)
    n_bars = 2 + len(results)  # original + severities + ... we'll add reference bands instead

    x = np.arange(n_groups)
    width = 0.18

    # Original
    orig_vals = [orig_metrics[k] for k in metric_keys]
    ax_bars.bar(x - 1.5 * width, orig_vals, width, label="Original (synthetic)",
                color="#9E9E9E", edgecolor="black", linewidth=0.5)

    # Codec sim severities
    for i, ((label, data), color) in enumerate(zip(results.items(), colors_sev)):
        vals = [data["metrics"][k] for k in metric_keys]
        ax_bars.bar(x + (i - 0.5) * width, vals, width,
                    label=f"Codec Sim {label.replace(chr(10), ' ')}",
                    color=color, edgecolor="black", linewidth=0.5, alpha=0.85)

    # Reference bands (from Phase 1 analysis)
    for j, k in enumerate(metric_keys):
        vcd_val = REFERENCE_DATA["Zoom VCD\n(target)"][k]
        if vcd_val is not None:
            ax_bars.plot([j - 0.3, j + 0.6], [vcd_val, vcd_val],
                         color="red", linewidth=2, linestyle="--", alpha=0.8)
            ax_bars.annotate("VCD ref", (j + 0.62, vcd_val), fontsize=8,
                             color="red", va="center", fontweight="bold")

        yt_val = REFERENCE_DATA["YouTube\nAVSpeech"][k]
        if yt_val is not None:
            ax_bars.plot([j - 0.3, j + 0.6], [yt_val, yt_val],
                         color="green", linewidth=1.5, linestyle=":", alpha=0.7)
            ax_bars.annotate("YouTube ref", (j + 0.62, yt_val), fontsize=7,
                             color="green", va="center")

    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(metric_labels, fontsize=10)
    ax_bars.set_title("Frequency Metrics — Codec Sim vs Real-World References", fontsize=13, fontweight="bold")
    ax_bars.legend(fontsize=8, loc="upper left", ncol=2)
    ax_bars.grid(True, axis="y", alpha=0.3)

    # ── Row 3 left: Severity sweep ──────────────────────────────────
    ax_sweep = fig.add_subplot(gs[2, 0:2])

    quality_levels = list(range(10, 91, 5))
    sweep_metrics = {k: [] for k in ["psd_slope", "freq_high_ratio", "sharpness_tenengrad"]}

    for q in quality_levels:
        trial_metrics = []
        for trial in range(10):
            random.seed(seed + trial * 200 + q)
            np.random.seed(seed + trial * 200 + q)
            out = apply_codec_simulation(img_orig, codec_quality=q)
            trial_metrics.append(compute_all_metrics(out))
        for k in sweep_metrics:
            sweep_metrics[k].append(np.mean([m[k] for m in trial_metrics]))

    color_map = {"psd_slope": "#E53935", "freq_high_ratio": "#2196F3", "sharpness_tenengrad": "#4CAF50"}
    label_map = {"psd_slope": "PSD Slope", "freq_high_ratio": "High-Freq Ratio", "sharpness_tenengrad": "Sharpness (Tenengrad)"}

    ax_sweep_twin = ax_sweep.twinx()

    for k in ["psd_slope", "freq_high_ratio"]:
        ax_sweep.plot(quality_levels, sweep_metrics[k], color=color_map[k],
                      linewidth=2.5, marker="o", markersize=4, label=label_map[k])
    ax_sweep_twin.plot(quality_levels, sweep_metrics["sharpness_tenengrad"],
                       color=color_map["sharpness_tenengrad"], linewidth=2.5,
                       marker="s", markersize=4, label=label_map["sharpness_tenengrad"],
                       linestyle="--")

    # Reference lines
    ax_sweep.axhline(y=REFERENCE_DATA["Zoom VCD\n(target)"]["psd_slope"],
                     color="#E53935", linestyle=":", alpha=0.5, linewidth=1.5)
    ax_sweep.annotate("VCD PSD slope", (12, REFERENCE_DATA["Zoom VCD\n(target)"]["psd_slope"] + 0.02),
                      fontsize=8, color="#E53935", alpha=0.7)
    ax_sweep.axhline(y=REFERENCE_DATA["Zoom VCD\n(target)"]["freq_high_ratio"],
                     color="#2196F3", linestyle=":", alpha=0.5, linewidth=1.5)
    ax_sweep.annotate("VCD high-freq", (12, REFERENCE_DATA["Zoom VCD\n(target)"]["freq_high_ratio"] + 0.005),
                      fontsize=8, color="#2196F3", alpha=0.7)

    ax_sweep.set_xlabel("Codec Quality (lower = more degradation)", fontsize=11)
    ax_sweep.set_ylabel("PSD Slope / Freq Ratio", fontsize=11)
    ax_sweep_twin.set_ylabel("Sharpness (Tenengrad)", fontsize=11, color=color_map["sharpness_tenengrad"])
    ax_sweep.set_title("Severity Sweep — Metrics vs Codec Quality", fontsize=13, fontweight="bold")
    ax_sweep.invert_xaxis()
    ax_sweep.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax_sweep.get_legend_handles_labels()
    lines2, labels2 = ax_sweep_twin.get_legend_handles_labels()
    ax_sweep.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    # ── Row 3 right: Pipeline diagram + summary table ─────────────────
    ax_info = fig.add_subplot(gs[2, 2:4])
    ax_info.axis("off")

    # Pipeline diagram
    steps = [
        ("1. Downscale\n→ Upscale", "#BBDEFB", "Simulates low\ncapture resolution"),
        ("2. Bilateral\nFilter", "#C8E6C9", "H.264/H.265\ndeblocking"),
        ("3. Block\nQuantization", "#FFE0B2", "8x8 boundary\ndiscontinuities"),
        ("4. Freq-Shaped\nNoise", "#F8BBD0", "Ringing &\nmosquito noise"),
        ("5. JPEG\nCompression", "#D1C4E9", "I-frame\nsimulation"),
    ]

    for i, (title, color, desc) in enumerate(steps):
        x_pos = 0.02 + i * 0.195
        box = FancyBboxPatch((x_pos, 0.72), 0.17, 0.22,
                              boxstyle="round,pad=0.01", facecolor=color,
                              edgecolor="gray", linewidth=1.5,
                              transform=ax_info.transAxes)
        ax_info.add_patch(box)
        ax_info.text(x_pos + 0.085, 0.87, title, transform=ax_info.transAxes,
                     ha="center", va="center", fontsize=9, fontweight="bold")
        ax_info.text(x_pos + 0.085, 0.76, desc, transform=ax_info.transAxes,
                     ha="center", va="center", fontsize=7.5, color="gray")
        if i < len(steps) - 1:
            ax_info.annotate("", xy=(x_pos + 0.19, 0.83), xytext=(x_pos + 0.17, 0.83),
                             arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
                             transform=ax_info.transAxes)

    ax_info.text(0.5, 0.98, "VideoCodecSimulation Pipeline",
                 transform=ax_info.transAxes, ha="center", fontsize=13, fontweight="bold")

    # Summary table
    table_data = [
        ["Preset", "Probability", "Quality Range", "Use Case"],
        ["Light", "10%", "(40, 80)", "R5/R6 default"],
        ["Moderate", "15%", "(30, 75)", "Balanced"],
        ["Strong", "22%", "(25, 70)", "Max robustness"],
    ]

    table = ax_info.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="lower center",
        bbox=[0.05, 0.0, 0.9, 0.42],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E3F2FD")
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("gray")

    ax_info.text(0.5, 0.47, "Augmentation Presets (quality_targeted_family)",
                 transform=ax_info.transAxes, ha="center", fontsize=11, fontweight="bold")

    # ── Save ──────────────────────────────────────────────────────────
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "codec_simulation_demo.png")

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\n{'=' * 60}")
    print(f"  Demo figure saved to: {output_path}")
    print(f"{'=' * 60}")

    # Also print metric summary
    print(f"\n{'─' * 60}")
    print("  Frequency Metrics Summary")
    print(f"{'─' * 60}")
    print(f"  {'Source':<25} {'PSD Slope':>12} {'HF Ratio':>12} {'HF/LF':>12}")
    print(f"  {'─' * 25} {'─' * 12} {'─' * 12} {'─' * 12}")
    print(f"  {'Original (synthetic)':<25} {orig_metrics['psd_slope']:>12.3f} {orig_metrics['freq_high_ratio']:>12.4f} {orig_metrics['freq_high_to_low']:>12.3f}")
    for label, data in results.items():
        m = data["metrics"]
        lbl = label.replace("\n", " ")
        print(f"  {'Codec Sim ' + lbl:<25} {m['psd_slope']:>12.3f} {m['freq_high_ratio']:>12.4f} {m['freq_high_to_low']:>12.3f}")
    print(f"  {'─' * 25} {'─' * 12} {'─' * 12} {'─' * 12}")
    print(f"  {'VCD Reference':<25} {-1.665:>12.3f} {0.249:>12.4f} {0.609:>12.3f}")
    print(f"  {'YouTube Reference':<25} {-1.942:>12.3f} {0.166:>12.4f} {0.335:>12.3f}")
    print(f"  {'DF40 Train Reals':<25} {-2.071:>12.3f} {0.145:>12.4f} {0.258:>12.3f}")
    print(f"{'─' * 60}")

    plt.close(fig)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoCodecSimulation visual demo")
    parser.add_argument("--output", type=str, default=None, help="Output path for the figure")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    create_demo_figure(seed=args.seed, output_path=args.output)
