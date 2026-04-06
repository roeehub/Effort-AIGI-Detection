#!/usr/bin/env python3
"""
Real vs Fake brightness/colour skew analysis for DF40 training data.

Computes per-image stats (brightness, std, R/B ratio, contrast) for real
and fake samples, broken down by method family, and plots distributions
side by side.  Also overlays Roee's real-world captures to show where
production data lands relative to training real/fake.
"""

import os
import sys
import csv
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def compute_stats(img_bgr):
    """Compute lighting-related stats from a BGR image."""
    img = img_bgr.astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    mean_bright = gray.mean()
    std_bright = gray.std()
    rb_ratio = (r.mean() + 1e-6) / (b.mean() + 1e-6)
    contrast_rms = gray.std() / (gray.mean() + 1e-6)
    return {
        "mean_brightness": mean_bright,
        "std_brightness": std_bright,
        "rb_ratio": rb_ratio,
        "contrast_rms": contrast_rms,
    }


def load_and_stat(directory, resize=224):
    """Load all images from directory, compute stats."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    results = []
    for p in sorted(Path(directory).rglob("*")):
        if p.suffix.lower() not in exts:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (resize, resize))
        s = compute_stats(img)
        s["filename"] = p.name
        # Extract method family from filename
        s["method"] = p.name.split("_")[0] if "_" in p.name else "unknown"
        results.append(s)
    return results


def categorize_fake(filename):
    """Map fake filename to method family."""
    name = filename.lower()
    gan_methods = ["stylegan2", "stylegan3", "styleganxl", "vqgan"]
    diffusion_methods = ["ddim", "rddm", "dit", "sit"]
    swap_methods = ["facedancer", "simswap", "inswap", "faceswap", "e4s",
                    "blendface", "uniface", "mobileswap"]
    reenact_methods = ["fomm", "mraa", "sadtalker", "wav2lip", "facevid2vid",
                       "pirender", "hyperreenact", "lia", "tpsm", "mcnet",
                       "one", "danet"]

    for m in gan_methods:
        if name.startswith(m):
            return "GAN"
    for m in diffusion_methods:
        if name.startswith(m):
            return "Diffusion"
    for m in swap_methods:
        if name.startswith(m):
            return "FaceSwap"
    for m in reenact_methods:
        if name.startswith(m):
            return "Reenact"
    return "Other"


def main():
    real_dir = "/tmp/df40_real_vs_fake/real"
    fake_dir = "/tmp/df40_real_vs_fake/fake"
    real_captures_dir = "/Users/roeedar/Downloads/roee_light"
    output_path = "/tmp/real_vs_fake_skew.png"

    print("Loading real training images...")
    real_stats = load_and_stat(real_dir)
    print(f"  {len(real_stats)} real images")

    print("Loading fake training images...")
    fake_stats = load_and_stat(fake_dir)
    print(f"  {len(fake_stats)} fake images")

    # Categorize fakes
    for s in fake_stats:
        s["family"] = categorize_fake(s["filename"])

    families = defaultdict(list)
    for s in fake_stats:
        families[s["family"]].append(s)

    print("\nFake breakdown by family:")
    for fam, items in sorted(families.items()):
        print(f"  {fam}: {len(items)}")

    # Load real-world captures
    real_captures = []
    if os.path.isdir(real_captures_dir):
        real_captures = load_and_stat(real_captures_dir)
        print(f"\nReal-world captures: {len(real_captures)}")

    # =========================================================================
    # PLOT 1: Real vs Fake overall distributions (4 metrics)
    # =========================================================================
    metrics = [
        ("mean_brightness", "Mean Brightness"),
        ("std_brightness", "Brightness Std Dev"),
        ("rb_ratio", "R/B Ratio (colour temperature proxy)"),
        ("contrast_rms", "RMS Contrast"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "DF40 Training Data: Real vs Fake Brightness/Colour Skew\n"
        f"Real: {len(real_stats)} images | Fake: {len(fake_stats)} images "
        f"(GAN:{len(families['GAN'])}, Diff:{len(families['Diffusion'])}, "
        f"Swap:{len(families['FaceSwap'])}, Reen:{len(families['Reenact'])})",
        fontsize=13, fontweight="bold",
    )
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, metrics):
        real_vals = np.array([s[key] for s in real_stats])
        fake_vals = np.array([s[key] for s in fake_stats])

        # Per-family
        fam_vals = {}
        for fam in ["GAN", "Diffusion", "FaceSwap", "Reenact"]:
            if families[fam]:
                fam_vals[fam] = np.array([s[key] for s in families[fam]])

        all_vals = np.concatenate([real_vals, fake_vals])
        if real_captures:
            cap_vals = np.array([s[key] for s in real_captures])
            all_vals = np.concatenate([all_vals, cap_vals])

        lo, hi = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
        margin = (hi - lo) * 0.1
        bins = np.linspace(lo - margin, hi + margin, 45)

        # Real distribution
        ax.hist(real_vals, bins=bins, alpha=0.55, density=True,
                label=f"Real (n={len(real_vals)}, μ={real_vals.mean():.1f})",
                color="steelblue", edgecolor="white", linewidth=0.5)

        # Fake overall
        ax.hist(fake_vals, bins=bins, alpha=0.40, density=True,
                label=f"Fake ALL (n={len(fake_vals)}, μ={fake_vals.mean():.1f})",
                color="crimson", edgecolor="white", linewidth=0.5)

        # Real mean and fake mean as vertical lines
        ax.axvline(real_vals.mean(), color="steelblue", linewidth=2,
                   linestyle="-", alpha=0.8)
        ax.axvline(fake_vals.mean(), color="crimson", linewidth=2,
                   linestyle="-", alpha=0.8)

        # Overlay real captures
        if real_captures:
            for j, rv in enumerate(cap_vals):
                ax.axvline(rv, color="gold", linewidth=1.2, linestyle="--",
                           alpha=0.6,
                           label="Your captures" if j == 0 else None)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7.5, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"\nSaved → {output_path}")
    plt.close(fig)

    # =========================================================================
    # PLOT 2: Per-family breakdown (box plots)
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11))
    fig2.suptitle(
        "DF40 Brightness/Colour by Method Family\n"
        "(+ Roee's real-world captures in gold)",
        fontsize=13, fontweight="bold",
    )
    axes2 = axes2.flatten()

    family_colors = {
        "Real": "steelblue",
        "GAN": "red",
        "Diffusion": "darkorange",
        "FaceSwap": "mediumpurple",
        "Reenact": "green",
        "Your captures": "gold",
    }

    for ax, (key, title) in zip(axes2, metrics):
        data = []
        labels = []
        colors = []

        # Real
        real_vals = [s[key] for s in real_stats]
        data.append(real_vals)
        labels.append(f"Real\n(n={len(real_vals)})")
        colors.append("steelblue")

        for fam in ["GAN", "Diffusion", "FaceSwap", "Reenact"]:
            if families[fam]:
                vals = [s[key] for s in families[fam]]
                data.append(vals)
                labels.append(f"{fam}\n(n={len(vals)})")
                colors.append(family_colors[fam])

        if real_captures:
            cap_vals = [s[key] for s in real_captures]
            data.append(cap_vals)
            labels.append(f"Your caps\n(n={len(cap_vals)})")
            colors.append("gold")

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                        showfliers=True, flierprops=dict(markersize=3))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    box_path = output_path.replace(".png", "_boxplot.png")
    fig2.savefig(box_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved → {box_path}")
    plt.close(fig2)

    # =========================================================================
    # Print summary table
    # =========================================================================
    print("\n" + "=" * 100)
    print("SUMMARY: Mean ± Std of each metric by category")
    print("=" * 100)
    header = f"{'Category':<20}"
    for _, title in metrics:
        header += f"  {title:>25}"
    print(header)
    print("-" * 100)

    categories = [("Real", real_stats)]
    for fam in ["GAN", "Diffusion", "FaceSwap", "Reenact"]:
        if families[fam]:
            categories.append((fam, families[fam]))
    categories.append(("Fake ALL", fake_stats))
    if real_captures:
        categories.append(("Your captures", real_captures))

    for cat_name, cat_stats in categories:
        row = f"{cat_name:<20}"
        for key, _ in metrics:
            vals = np.array([s[key] for s in cat_stats])
            row += f"  {vals.mean():10.1f} ± {vals.std():<10.1f}"
        print(row)

    print()

    # Key diagnostic: is fake systematically brighter?
    real_bright = np.array([s["mean_brightness"] for s in real_stats])
    fake_bright = np.array([s["mean_brightness"] for s in fake_stats])
    diff = fake_bright.mean() - real_bright.mean()
    direction = "BRIGHTER" if diff > 0 else "DIMMER"
    print(f">>> Fake images are on average {abs(diff):.1f} units {direction} than real")
    print(f"    Real mean brightness: {real_bright.mean():.1f} (std {real_bright.std():.1f})")
    print(f"    Fake mean brightness: {fake_bright.mean():.1f} (std {fake_bright.std():.1f})")

    if real_captures:
        cap_bright = np.array([s["mean_brightness"] for s in real_captures])
        print(f"    Your captures mean:   {cap_bright.mean():.1f} (std {cap_bright.std():.1f})")
        # Where do captures land relative to real/fake?
        overlap_with_real = np.mean((cap_bright >= np.percentile(real_bright, 5)) &
                                    (cap_bright <= np.percentile(real_bright, 95))) * 100
        overlap_with_fake = np.mean((cap_bright >= np.percentile(fake_bright, 5)) &
                                    (cap_bright <= np.percentile(fake_bright, 95))) * 100
        print(f"\n    Your captures overlap with training REAL [5%-95%]: {overlap_with_real:.0f}%")
        print(f"    Your captures overlap with training FAKE [5%-95%]: {overlap_with_fake:.0f}%")
        if overlap_with_fake > overlap_with_real:
            print("    ⚠️  Your captures sit CLOSER to the FAKE distribution than REAL!")
            print("    → This explains why the model might flag you as fake under certain lighting.")

    # Save CSV
    csv_path = output_path.replace(".png", ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "family", "filename",
                         "mean_brightness", "std_brightness", "rb_ratio", "contrast_rms"])
        for s in real_stats:
            writer.writerow(["real", "Real", s["filename"],
                             s["mean_brightness"], s["std_brightness"],
                             s["rb_ratio"], s["contrast_rms"]])
        for s in fake_stats:
            writer.writerow(["fake", s["family"], s["filename"],
                             s["mean_brightness"], s["std_brightness"],
                             s["rb_ratio"], s["contrast_rms"]])
        for s in real_captures:
            writer.writerow(["capture", "YourCaptures", s["filename"],
                             s["mean_brightness"], s["std_brightness"],
                             s["rb_ratio"], s["contrast_rms"]])
    print(f"\nSaved raw data → {csv_path}")


if __name__ == "__main__":
    main()
