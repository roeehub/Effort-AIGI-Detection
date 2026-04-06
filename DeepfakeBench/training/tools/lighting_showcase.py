#!/usr/bin/env python3
"""
Lighting Augmentation Showcase & Comparison Tool
=================================================

Three modes:

1. SHOWCASE — Take a face crop, apply each proposed lighting augmentation,
   render a visual grid.  Optionally include CLIP cosine distance from the
   original for each variant.

2. COMPARE  — You provide a folder of real captures (same face, different
   room lighting).  The script profiles each capture's brightness/color,
   then tries to recreate those conditions from a "neutral" reference using
   augmentations.  Side-by-side comparison shows whether synthetic
   augmentations land in the same perceptual/embedding neighbourhood as the
   real lighting changes.

3. AUDIT   — Take a batch of training images, apply randomised current-
   pipeline and proposed augmentations, and plot distributions of image
   properties (brightness, contrast, R/B ratio).  If you also provide
   real-world captures, their stats are overlaid so you can see whether
   the augmented distributions actually cover production conditions.

Usage:
------
# Showcase mode — grid of augmentations applied to a single image
python tools/lighting_showcase.py showcase \
    --image path/to/face_crop.jpg \
    --output lighting_showcase.png

# Showcase with CLIP cosine distances
python tools/lighting_showcase.py showcase \
    --image path/to/face_crop.jpg \
    --clip-weights ./weights/CLIP-ViT-B-16-DataComp.XL-s13B-b90K/ \
    --output lighting_showcase.png

# Compare mode — real captures vs synthetic augmentations
python tools/lighting_showcase.py compare \
    --captures-dir ./my_lighting_captures/ \
    --reference ./my_lighting_captures/neutral.jpg \
    --clip-weights ./weights/CLIP-ViT-B-16-DataComp.XL-s13B-b90K/ \
    --output lighting_compare.png

# Audit mode — statistical coverage of augmentations on training images
python tools/lighting_showcase.py audit \
    --images-dir ./path/to/training/crops/ \
    --max-images 200 \
    --output lighting_audit.png

# Audit with real-world captures overlaid
python tools/lighting_showcase.py audit \
    --images-dir ./path/to/training/crops/ \
    --real-captures-dir ./my_lighting_captures/ \
    --output lighting_audit.png

Requirements:
  pip install numpy opencv-python-headless Pillow matplotlib
  (optional for CLIP) pip install open_clip_torch torch
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

import random

import cv2
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================================
# Lighting Augmentation Transforms
# ============================================================================
# Each transform takes a uint8 HWC BGR numpy array and returns the same.
# They are intentionally standalone — no albumentations dependency — so this
# script works without the full training environment.
# ============================================================================


def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction: output = (input/255)^gamma * 255."""
    lut = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(img, lut)


def apply_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Multiply pixel values by `factor`, clamp to [0, 255]."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Adjust contrast around mean intensity."""
    mean = img.mean()
    return np.clip(mean + (img.astype(np.float32) - mean) * factor, 0, 255).astype(
        np.uint8
    )


def apply_cct(img: np.ndarray, cct_kelvin: int) -> np.ndarray:
    """
    Simulate a colour-temperature (CCT) shift by applying per-channel gains.

    Uses Tanner Helland's RGB-from-Kelvin approximation (widely used in
    graphics / photography tools).  The gain is computed relative to a
    6500 K daylight reference so that 6500 K ≈ identity.

    Lower CCT → warmer (more red/yellow, less blue).
    Higher CCT → cooler (more blue, less red).
    """
    def _kelvin_to_rgb(kelvin: int):
        """Return (R, G, B) float gains for a given colour temperature."""
        temp = kelvin / 100.0
        # Red
        if temp <= 66:
            r = 255.0
        else:
            r = 329.698727446 * ((temp - 60) ** -0.1332047592)
            r = max(0.0, min(255.0, r))
        # Green
        if temp <= 66:
            g = 99.4708025861 * np.log(temp) - 161.1195681661
        else:
            g = 288.1221695283 * ((temp - 60) ** -0.0755148492)
        g = max(0.0, min(255.0, g))
        # Blue
        if temp >= 66:
            b = 255.0
        elif temp <= 19:
            b = 0.0
        else:
            b = 138.5177312231 * np.log(temp - 10) - 305.0447927307
            b = max(0.0, min(255.0, b))
        return r, g, b

    ref_r, ref_g, ref_b = _kelvin_to_rgb(6500)  # daylight reference
    tgt_r, tgt_g, tgt_b = _kelvin_to_rgb(cct_kelvin)

    # Gains relative to daylight (so 6500K → ~identity)
    gain_r = tgt_r / ref_r
    gain_g = tgt_g / ref_g
    gain_b = tgt_b / max(ref_b, 1e-6)

    # img is BGR
    out = img.astype(np.float32)
    out[:, :, 0] *= gain_b  # B
    out[:, :, 1] *= gain_g  # G
    out[:, :, 2] *= gain_r  # R
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_shadow(
    img: np.ndarray,
    direction: str = "left",
    intensity: float = 0.5,
    softness: float = 0.3,
) -> np.ndarray:
    """
    Apply a synthetic directional shadow across the face.

    Supports 8 directions: 4 cardinal + 4 diagonal.
    Vectorised version (no per-pixel Python loop).

    `direction`:  'left', 'right', 'top', 'bottom',
                  'top_left', 'top_right', 'bottom_left', 'bottom_right'
    `intensity`:  0 = no shadow, 1 = fully black
    `softness`:   fraction of image width/height used for gradient falloff
    """
    h, w = img.shape[:2]

    def _make_1d(length, invert=False):
        transition = max(1, int(length * softness))
        shadow_len = length - transition
        shadow_val = 1.0 - intensity
        profile = np.concatenate([
            np.full(shadow_len, shadow_val, dtype=np.float32),
            np.linspace(shadow_val, 1.0, transition, dtype=np.float32),
        ])
        if invert:
            profile = profile[::-1].copy()
        return profile

    if direction in ("left", "right"):
        profile_h = _make_1d(w, invert=(direction == "right"))
        mask = np.ones((h, 1), dtype=np.float32) * profile_h[np.newaxis, :]
    elif direction in ("top", "bottom"):
        profile_v = _make_1d(h, invert=(direction == "bottom"))
        mask = profile_v[:, np.newaxis] * np.ones((1, w), dtype=np.float32)
    else:
        # Diagonal: combine H + V profiles, take element-wise minimum.
        h_invert = direction.endswith("right")
        v_invert = direction.startswith("bottom")
        profile_h = _make_1d(w, invert=h_invert)
        profile_v = _make_1d(h, invert=v_invert)
        mask_h = np.ones((h, 1), dtype=np.float32) * profile_h[np.newaxis, :]
        mask_v = profile_v[:, np.newaxis] * np.ones((1, w), dtype=np.float32)
        mask = np.minimum(mask_h, mask_v)

    out = img.astype(np.float32) * mask[:, :, np.newaxis]
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_compound_lighting(
    img: np.ndarray,
    gamma: float = 1.0,
    brightness: float = 1.0,
    contrast: float = 1.0,
    cct: int = 6500,
) -> np.ndarray:
    """Apply gamma + brightness + contrast + CCT in sequence (compound)."""
    out = apply_gamma(img, gamma)
    out = apply_brightness(out, brightness)
    out = apply_contrast(out, contrast)
    out = apply_cct(out, cct)
    return out


# ============================================================================
# Image Statistics
# ============================================================================


def compute_image_stats(img_bgr: np.ndarray) -> dict:
    """Compute lighting-relevant statistics for a BGR image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b, g, r = (
        img_bgr[:, :, 0].astype(np.float32),
        img_bgr[:, :, 1].astype(np.float32),
        img_bgr[:, :, 2].astype(np.float32),
    )
    mean_intensity = gray.mean()
    std_intensity = gray.std()

    # Approximate CCT from R/B ratio (very rough heuristic)
    rb_ratio = (r.mean() + 1e-6) / (b.mean() + 1e-6)

    return {
        "mean_brightness": float(mean_intensity),
        "std_brightness": float(std_intensity),
        "mean_r": float(r.mean()),
        "mean_g": float(g.mean()),
        "mean_b": float(b.mean()),
        "rb_ratio": float(rb_ratio),
        "contrast_rms": float(gray.std() / (gray.mean() + 1e-6)),
    }


# ============================================================================
# CLIP embedding (optional)
# ============================================================================


def load_clip_model(weights_path: str, device: str = "cpu"):
    """Load OpenCLIP B-16 model and return (model, preprocess, device)."""
    try:
        import torch
        import open_clip
    except ImportError:
        print("ERROR: open_clip_torch and torch are required for CLIP distances.")
        print("  pip install open_clip_torch torch")
        sys.exit(1)

    weight_file = None
    if os.path.isdir(weights_path):
        for f in os.listdir(weights_path):
            if f.endswith(".bin") or f.endswith(".pt") or f.endswith(".safetensors"):
                weight_file = os.path.join(weights_path, f)
                break
    elif os.path.isfile(weights_path):
        weight_file = weights_path

    if weight_file:
        print(f"Loading CLIP from local weights: {weight_file}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained=weight_file
        )
    else:
        print("Loading CLIP from OpenCLIP hub (ViT-B-16, datacomp_xl_s13b_b90k)...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="datacomp_xl_s13b_b90k"
        )

    model = model.to(device).eval()
    return model, preprocess, device


def get_clip_embedding(model, preprocess, device, img_bgr: np.ndarray) -> np.ndarray:
    """Get L2-normalised CLIP visual embedding for a BGR numpy image."""
    import torch

    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    tensor = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().flatten()


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity."""
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(1.0 - sim)


# ============================================================================
# SHOWCASE MODE
# ============================================================================


# Define the augmentation grid — each entry is (label, transform_fn)
SHOWCASE_AUGMENTATIONS = [
    # --- Current pipeline transforms ---
    ("Original", lambda img: img),
    ("Gamma 0.5\n(current wide)", lambda img: apply_gamma(img, 0.5)),
    ("Gamma 1.5\n(current wide)", lambda img: apply_gamma(img, 1.5)),
    ("Bright ×0.6\n(current wide)", lambda img: apply_brightness(img, 0.6)),
    ("Bright ×1.4\n(current wide)", lambda img: apply_brightness(img, 1.4)),
    # --- PROPOSED: CCT / white balance ---
    ("CCT 2700K\n(warm lamp)", lambda img: apply_cct(img, 2700)),
    ("CCT 4000K\n(warm LED)", lambda img: apply_cct(img, 4000)),
    ("CCT 8000K\n(overcast cool)", lambda img: apply_cct(img, 8000)),
    # --- PROPOSED: Directional shadow ---
    ("Shadow left\n(window right)", lambda img: apply_shadow(img, "left", 0.5, 0.35)),
    ("Shadow right\n(window left)", lambda img: apply_shadow(img, "right", 0.5, 0.35)),
    ("Shadow top\n(overhead off)", lambda img: apply_shadow(img, "top", 0.4, 0.3)),
    # --- PROPOSED: Compound (what really happens indoors) ---
    (
        "Compound: dim warm\n(evening lamp)",
        lambda img: apply_compound_lighting(img, gamma=1.3, brightness=0.7, cct=2700),
    ),
    (
        "Compound: bright cool\n(daylight window)",
        lambda img: apply_compound_lighting(img, gamma=0.8, brightness=1.3, cct=7500),
    ),
    (
        "Compound: backlit\n(window behind)",
        lambda img: apply_compound_lighting(img, gamma=1.5, brightness=0.55, contrast=0.8, cct=5500),
    ),
    (
        "Compound: overhead\n(office fluorescent)",
        lambda img: apply_compound_lighting(img, gamma=0.9, brightness=1.1, contrast=1.15, cct=4200),
    ),
]


def run_showcase(args):
    """Generate a grid of augmented images from a single face crop."""
    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: Cannot read image: {args.image}")
        sys.exit(1)

    # Resize for display
    display_size = 224
    img = cv2.resize(img, (display_size, display_size))

    # Load CLIP if requested
    clip_model = None
    if args.clip_weights:
        clip_model, clip_preprocess, clip_device = load_clip_model(
            args.clip_weights, "cuda" if _has_cuda() else "cpu"
        )
        ref_emb = get_clip_embedding(clip_model, clip_preprocess, clip_device, img)

    # Apply all augmentations
    results = []
    for label, fn in SHOWCASE_AUGMENTATIONS:
        aug_img = fn(img)
        stats = compute_image_stats(aug_img)
        cos_dist = None
        if clip_model is not None:
            emb = get_clip_embedding(clip_model, clip_preprocess, clip_device, aug_img)
            cos_dist = cosine_distance(ref_emb, emb)
        results.append((label, aug_img, stats, cos_dist))

    # Plot grid
    n = len(results)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(ncols * 4.2, nrows * 5.0))
    fig.suptitle(
        "Lighting Augmentation Showcase\nTop: current pipeline transforms | Bottom: proposed additions",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.45, wspace=0.25)

    for idx, (label, aug_img, stats, cos_dist) in enumerate(results):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        ax.imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
        ax.set_xticks([])
        ax.set_yticks([])

        subtitle = f"μ={stats['mean_brightness']:.0f}  σ={stats['std_brightness']:.0f}  R/B={stats['rb_ratio']:.2f}"
        if cos_dist is not None:
            subtitle += f"\nCLIP cos_dist={cos_dist:.4f}"
            # Color the border by cosine distance
            color = "green" if cos_dist < 0.05 else ("orange" if cos_dist < 0.10 else "red")
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xlabel(subtitle, fontsize=7.5, color="gray")

    # Hide empty subplots
    for idx in range(n, nrows * ncols):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        ax.axis("off")

    out_path = args.output or "lighting_showcase.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved showcase grid → {out_path}")
    plt.close(fig)

    # Also print a text table
    print(f"\n{'Label':<30} {'Mean':>5} {'Std':>5} {'R/B':>5}", end="")
    if clip_model:
        print(f"  {'CosDist':>8}", end="")
    print()
    print("-" * 60)
    for label, _, stats, cos_dist in results:
        lbl = label.replace("\n", " / ")
        print(
            f"{lbl:<30} {stats['mean_brightness']:5.0f} {stats['std_brightness']:5.0f} {stats['rb_ratio']:5.2f}",
            end="",
        )
        if cos_dist is not None:
            print(f"  {cos_dist:8.4f}", end="")
        print()


# ============================================================================
# COMPARE MODE
# ============================================================================


def _estimate_augmentation_params(ref_stats: dict, target_stats: dict) -> dict:
    """
    Given stats of a reference (neutral) image and a target (real capture),
    estimate augmentation params that would transform the reference to look
    like the target.  Returns a dict of approximate transform parameters.
    """
    # Brightness ratio
    brightness_ratio = target_stats["mean_brightness"] / (
        ref_stats["mean_brightness"] + 1e-6
    )

    # Contrast ratio (std / mean)
    contrast_ratio = target_stats["contrast_rms"] / (ref_stats["contrast_rms"] + 1e-6)

    # Estimate CCT from R/B shift direction
    # Higher R/B → warmer; lower R/B → cooler
    rb_delta = target_stats["rb_ratio"] - ref_stats["rb_ratio"]
    # Map rb_delta to a CCT offset from 6500K (rough heuristic)
    # +0.3 rb_delta ≈ -2000K, -0.3 ≈ +2000K
    est_cct = int(6500 - rb_delta * 6000)
    est_cct = max(2000, min(10000, est_cct))

    # Gamma estimate: if target is darker with same content, gamma > 1
    # Rough: gamma ≈ log(target_mean/255) / log(ref_mean/255)
    ref_norm = ref_stats["mean_brightness"] / 255.0
    tgt_norm = target_stats["mean_brightness"] / 255.0
    if ref_norm > 0.01 and tgt_norm > 0.01:
        est_gamma = np.log(tgt_norm + 1e-6) / np.log(ref_norm + 1e-6)
        est_gamma = max(0.3, min(3.0, est_gamma))
    else:
        est_gamma = 1.0

    return {
        "brightness": float(np.clip(brightness_ratio, 0.3, 3.0)),
        "contrast": float(np.clip(contrast_ratio, 0.5, 2.0)),
        "cct": est_cct,
        "gamma": float(est_gamma),
    }


def run_compare(args):
    """Compare real lighting captures vs synthetic augmentations."""
    captures_dir = Path(args.captures_dir)
    if not captures_dir.is_dir():
        print(f"ERROR: --captures-dir is not a directory: {captures_dir}")
        sys.exit(1)

    ref_path = args.reference
    if not ref_path:
        print("ERROR: --reference is required for compare mode.")
        sys.exit(1)

    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        print(f"ERROR: Cannot read reference image: {ref_path}")
        sys.exit(1)

    display_size = 224
    ref_img = cv2.resize(ref_img, (display_size, display_size))
    ref_stats = compute_image_stats(ref_img)

    # Collect capture images (skip the reference itself)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    capture_paths = sorted(
        [
            p
            for p in captures_dir.iterdir()
            if p.suffix.lower() in exts and str(p) != str(Path(ref_path).resolve())
        ]
    )

    if not capture_paths:
        print(f"ERROR: No images found in {captures_dir}")
        sys.exit(1)

    print(f"Reference: {ref_path}")
    print(f"Captures:  {len(capture_paths)} images in {captures_dir}")

    # Load CLIP if requested
    clip_model = None
    if args.clip_weights:
        clip_model, clip_preprocess, clip_device = load_clip_model(
            args.clip_weights, "cuda" if _has_cuda() else "cpu"
        )
        ref_emb = get_clip_embedding(clip_model, clip_preprocess, clip_device, ref_img)

    # Process each capture
    rows = []  # (name, real_img, synth_img, real_stats, synth_stats, params, cos_dist_real, cos_dist_synth)
    for cap_path in capture_paths:
        cap_img = cv2.imread(str(cap_path))
        if cap_img is None:
            continue
        cap_img = cv2.resize(cap_img, (display_size, display_size))
        cap_stats = compute_image_stats(cap_img)

        # Estimate params to recreate this lighting from the reference
        params = _estimate_augmentation_params(ref_stats, cap_stats)

        # Apply compound augmentation to the reference
        synth_img = apply_compound_lighting(
            ref_img,
            gamma=params["gamma"],
            brightness=params["brightness"],
            contrast=params["contrast"],
            cct=params["cct"],
        )
        synth_stats = compute_image_stats(synth_img)

        cos_real = cos_synth = None
        if clip_model is not None:
            emb_real = get_clip_embedding(
                clip_model, clip_preprocess, clip_device, cap_img
            )
            emb_synth = get_clip_embedding(
                clip_model, clip_preprocess, clip_device, synth_img
            )
            cos_real = cosine_distance(ref_emb, emb_real)
            cos_synth = cosine_distance(ref_emb, emb_synth)

        rows.append(
            (
                cap_path.stem,
                cap_img,
                synth_img,
                cap_stats,
                synth_stats,
                params,
                cos_real,
                cos_synth,
            )
        )

    # Plot: for each capture, show [Reference | Real capture | Synthetic recreation]
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(13, n * 4.2 + 1.5))
    if n == 1:
        axes = axes[np.newaxis, :]  # ensure 2D

    fig.suptitle(
        "Real Lighting Captures vs. Synthetic Augmentation Recreation",
        fontsize=14,
        fontweight="bold",
    )

    col_titles = ["Reference (neutral)", "Real capture", "Synthetic recreation"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=10)

    for i, (name, real_img, synth_img, real_s, synth_s, params, cos_r, cos_s) in enumerate(rows):
        # Reference
        ax_ref = axes[i, 0]
        ax_ref.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
        ax_ref.set_xticks([])
        ax_ref.set_yticks([])
        ax_ref.set_ylabel(name, fontsize=9, fontweight="bold", rotation=0, labelpad=60, va="center")

        # Real capture
        ax_real = axes[i, 1]
        ax_real.imshow(cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB))
        ax_real.set_xticks([])
        ax_real.set_yticks([])
        subtitle = f"μ={real_s['mean_brightness']:.0f} R/B={real_s['rb_ratio']:.2f}"
        if cos_r is not None:
            subtitle += f"\nCLIP cos_dist={cos_r:.4f}"
        ax_real.set_xlabel(subtitle, fontsize=7.5, color="gray")

        # Synthetic recreation
        ax_syn = axes[i, 2]
        ax_syn.imshow(cv2.cvtColor(synth_img, cv2.COLOR_BGR2RGB))
        ax_syn.set_xticks([])
        ax_syn.set_yticks([])
        p = params
        subtitle = (
            f"μ={synth_s['mean_brightness']:.0f} R/B={synth_s['rb_ratio']:.2f}\n"
            f"γ={p['gamma']:.2f} bright={p['brightness']:.2f} CCT={p['cct']}K"
        )
        if cos_s is not None:
            subtitle += f"\nCLIP cos_dist={cos_s:.4f}"
            # Highlight how close synthetic is to real
            gap = abs(cos_r - cos_s) if cos_r is not None else 0
            color = "green" if gap < 0.02 else ("orange" if gap < 0.05 else "red")
            for spine in ax_syn.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
        ax_syn.set_xlabel(subtitle, fontsize=7.5, color="gray")

    out_path = args.output or "lighting_compare.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"\nSaved comparison grid → {out_path}")
    plt.close(fig)

    # Text summary
    print(f"\n{'Capture':<20} {'Real μ':>7} {'Synth μ':>8} {'Real R/B':>9} {'Synth R/B':>10} {'Est CCT':>8}", end="")
    if clip_model:
        print(f"  {'CosD_real':>10} {'CosD_synth':>11} {'Gap':>6}", end="")
    print()
    print("-" * 100)
    for name, _, _, real_s, synth_s, params, cos_r, cos_s in rows:
        print(
            f"{name:<20} {real_s['mean_brightness']:7.1f} {synth_s['mean_brightness']:8.1f} "
            f"{real_s['rb_ratio']:9.2f} {synth_s['rb_ratio']:10.2f} {params['cct']:8d}",
            end="",
        )
        if cos_r is not None:
            gap = abs(cos_r - cos_s)
            print(f"  {cos_r:10.4f} {cos_s:11.4f} {gap:6.4f}", end="")
        print()

    if clip_model:
        gaps = [abs(r[6] - r[7]) for r in rows if r[6] is not None]
        if gaps:
            print(f"\nMean |cos_dist_real - cos_dist_synth| = {np.mean(gaps):.4f}")
            print("(Lower = augmentation better recreates real embedding shift)")


# ============================================================================
# AUDIT MODE
# ============================================================================

# Randomised augmentation "strategies" — each draws params from a range
# and applies them.  This mirrors what a training pipeline would do.

def _random_current_pipeline(img: np.ndarray) -> np.ndarray:
    """Simulate the pre-R12 training pipeline's lighting augmentation.

    Matches the old context_variation_block: OneOf(gamma, brightness, shift)
    with the ranges from R10_A / R11_C.
    """
    choice = random.choice(["gamma", "brightness", "none"])
    if choice == "gamma":
        gamma = random.uniform(0.50, 1.50)
        return apply_gamma(img, gamma)
    elif choice == "brightness":
        factor = 1.0 + random.uniform(-0.40, 0.40)
        return apply_brightness(img, factor)
    return img


def _random_r12_pipeline(img: np.ndarray) -> np.ndarray:
    """Simulate the R12 production pipeline (compound, no OneOf).

    Matches ``_build_context_variation_block()`` with ``vcd_targeted`` preset:
    independent gamma, asymmetric brightness, CCT — all at p=0.15 each.
    No shadow or gamma-up (those are off in R12).
    """
    # RandomGamma: gamma_limit (70, 130) → albumentations γ = sampled/100
    if random.random() < 0.15:
        gamma = random.uniform(0.70, 1.30)
        img = apply_gamma(img, gamma)
    # RandomBrightnessContrast: brightness (-0.20, +0.60), contrast 0.25
    if random.random() < 0.15:
        brightness = 1.0 + random.uniform(-0.20, 0.60)
        img = apply_brightness(img, brightness)
        contrast = 1.0 + random.uniform(-0.25, 0.25)
        img = apply_contrast(img, contrast)
    # CCT: (2700, 8000) at p=0.15
    if random.random() < 0.15:
        cct = random.randint(2700, 8000)
        img = apply_cct(img, cct)
    return img


def _random_proposed_pipeline(img: np.ndarray, shadow_p: float = 0.10,
                              gamma_up_p: float = 0.12) -> np.ndarray:
    """Simulate the proposed pipeline: R12 base + shadow + gamma-up.

    Adds the two new transforms implemented in the lighting robustness work:
    - ``DirectionalShadow``: uneven indoor lighting simulation
    - ``GammaUp``: dedicated brightness push for the 104→160-205 gap

    Parameters mirror the suggested YAML config so the audit output
    directly predicts what training would see.
    """
    # Start with R12 baseline transforms
    img = _random_r12_pipeline(img)

    # DirectionalShadow at shadow_p
    if random.random() < shadow_p:
        directions = ["left", "right", "top", "bottom",
                       "top_left", "top_right", "bottom_left", "bottom_right"]
        direction = random.choice(directions)
        intensity = random.uniform(0.15, 0.45)
        softness = random.uniform(0.20, 0.50)
        img = apply_shadow(img, direction, intensity, softness)

    # GammaUp at gamma_up_p  (gamma < 1 → always brighter)
    if random.random() < gamma_up_p:
        gamma = random.uniform(0.45, 0.85)
        img = apply_gamma(img, gamma)

    return img


def run_audit(args):
    """Statistical coverage audit of augmentations over a batch of images."""
    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        print(f"ERROR: --images-dir is not a directory: {images_dir}")
        sys.exit(1)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_paths = sorted(
        [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    )
    max_images = args.max_images or 200
    if len(all_paths) > max_images:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(all_paths), max_images, replace=False)
        all_paths = [all_paths[i] for i in sorted(indices)]

    print(f"Audit: {len(all_paths)} images from {images_dir}")

    display_size = 224
    repeats = args.repeats or 5  # augmentations per image

    shadow_p = args.shadow_p
    gamma_up_p = args.gamma_up_p
    print(f"  Proposed params: shadow_p={shadow_p}, gamma_up_p={gamma_up_p}")

    # Collect stats: original, R12 (current production), proposed
    stats_original = []
    stats_r12 = []
    stats_proposed = []

    for p in all_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (display_size, display_size))
        stats_original.append(compute_image_stats(img))

        for _ in range(repeats):
            aug_r12 = _random_r12_pipeline(img.copy())
            stats_r12.append(compute_image_stats(aug_r12))

            aug_proposed = _random_proposed_pipeline(img.copy(), shadow_p=shadow_p,
                                                      gamma_up_p=gamma_up_p)
            stats_proposed.append(compute_image_stats(aug_proposed))

    print(f"  Originals:       {len(stats_original)}")
    print(f"  R12 aug:         {len(stats_r12)} ({repeats}× per image)")
    print(f"  Proposed aug:    {len(stats_proposed)} ({repeats}× per image)")

    # Optionally load real captures
    stats_real_captures = []
    if args.real_captures_dir:
        cap_dir = Path(args.real_captures_dir)
        if cap_dir.is_dir():
            for cp in sorted(cap_dir.iterdir()):
                if cp.suffix.lower() in exts:
                    cimg = cv2.imread(str(cp))
                    if cimg is not None:
                        cimg = cv2.resize(cimg, (display_size, display_size))
                        stats_real_captures.append(compute_image_stats(cimg))
            print(f"  Real captures:   {len(stats_real_captures)}")

    # Extract arrays for plotting
    def _extract(stats_list, key):
        return np.array([s[key] for s in stats_list])

    metrics = [
        ("mean_brightness", "Mean Brightness"),
        ("std_brightness", "Brightness Std Dev"),
        ("rb_ratio", "R/B Ratio (≈ colour temperature)"),
        ("contrast_rms", "RMS Contrast"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Augmentation Coverage Audit\n"
        f"{len(stats_original)} training images × {repeats} augmentations each",
        fontsize=14,
        fontweight="bold",
    )
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, metrics):
        orig_vals = _extract(stats_original, key)
        r12_vals = _extract(stats_r12, key)
        proposed_vals = _extract(stats_proposed, key)

        # Determine common bin range
        all_vals = np.concatenate([orig_vals, r12_vals, proposed_vals])
        if stats_real_captures:
            real_vals = _extract(stats_real_captures, key)
            all_vals = np.concatenate([all_vals, real_vals])
        lo, hi = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
        margin = (hi - lo) * 0.1
        bins = np.linspace(lo - margin, hi + margin, 50)

        ax.hist(
            orig_vals, bins=bins, alpha=0.45, density=True,
            label=f"Original (n={len(orig_vals)})", color="steelblue", edgecolor="white", linewidth=0.5,
        )
        ax.hist(
            r12_vals, bins=bins, alpha=0.35, density=True,
            label=f"R12 production (n={len(r12_vals)})", color="orange", edgecolor="white", linewidth=0.5,
        )
        ax.hist(
            proposed_vals, bins=bins, alpha=0.35, density=True,
            label=f"Proposed (n={len(proposed_vals)})", color="green", edgecolor="white", linewidth=0.5,
        )

        # Overlay real captures as vertical lines
        if stats_real_captures:
            real_vals = _extract(stats_real_captures, key)
            for j, rv in enumerate(real_vals):
                ax.axvline(
                    rv, color="red", linewidth=1.5, linestyle="--",
                    alpha=0.7,
                    label="Real captures" if j == 0 else None,
                )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = args.output or "lighting_audit.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"\nSaved audit distributions → {out_path}")
    plt.close(fig)

    # --- Coverage summary table ---
    print(f"\n{'Metric':<30} {'Orig [5%–95%]':>20} {'R12 [5%–95%]':>20} {'Proposed [5%–95%]':>20}", end="")
    if stats_real_captures:
        print(f"  {'Real captures [min–max]':>25}", end="")
    print()
    print("-" * 125)

    for key, title in metrics:
        orig_vals = _extract(stats_original, key)
        r12_vals = _extract(stats_r12, key)
        proposed_vals = _extract(stats_proposed, key)

        o5, o95 = np.percentile(orig_vals, [5, 95])
        c5, c95 = np.percentile(r12_vals, [5, 95])
        p5, p95 = np.percentile(proposed_vals, [5, 95])

        print(
            f"{title:<30} {o5:8.2f} – {o95:<8.2f}  {c5:8.2f} – {c95:<8.2f}  {p5:8.2f} – {p95:<8.2f}",
            end="",
        )

        if stats_real_captures:
            real_vals = _extract(stats_real_captures, key)
            rmin, rmax = real_vals.min(), real_vals.max()
            print(f"  {rmin:10.2f} – {rmax:<10.2f}", end="")

            # Check coverage: do R12/proposed ranges cover real captures?
            r12_covers = np.mean((real_vals >= c5) & (real_vals <= c95)) * 100
            proposed_covers = np.mean((real_vals >= p5) & (real_vals <= p95)) * 100
            print(f"  R12:{r12_covers:3.0f}% Proposed:{proposed_covers:3.0f}%", end="")

        print()

    if stats_real_captures:
        print("\n(Coverage %: fraction of real capture values falling within augmentation's [5%–95%] range)")

    # --- Save raw stats to CSV ---
    csv_path = str(Path(out_path).with_suffix(".csv"))
    try:
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source", "image_idx"] + [k for k, _ in metrics])
            for i, s in enumerate(stats_original):
                writer.writerow(["original", i] + [s[k] for k, _ in metrics])
            for i, s in enumerate(stats_r12):
                writer.writerow(["r12_aug", i] + [s[k] for k, _ in metrics])
            for i, s in enumerate(stats_proposed):
                writer.writerow(["proposed_aug", i] + [s[k] for k, _ in metrics])
            for i, s in enumerate(stats_real_captures):
                writer.writerow(["real_capture", i] + [s[k] for k, _ in metrics])
        print(f"Saved raw stats → {csv_path}")
    except Exception as e:
        print(f"Warning: could not save CSV: {e}")


# ============================================================================
# Utility
# ============================================================================


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Lighting Augmentation Showcase & Comparison Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- Showcase ---
    p_show = sub.add_parser("showcase", help="Grid of augmented images from one face crop")
    p_show.add_argument("--image", required=True, help="Path to a face crop image")
    p_show.add_argument("--output", default=None, help="Output image path (default: lighting_showcase.png)")
    p_show.add_argument(
        "--clip-weights",
        default=None,
        help="Path to OpenCLIP weights dir or file (optional — adds cosine distances)",
    )

    # --- Compare ---
    p_comp = sub.add_parser("compare", help="Real captures vs synthetic augmentations")
    p_comp.add_argument("--captures-dir", required=True, help="Directory of real captures (same face, different lighting)")
    p_comp.add_argument("--reference", required=True, help="Path to the 'neutral' reference image from the captures")
    p_comp.add_argument("--output", default=None, help="Output image path (default: lighting_compare.png)")
    p_comp.add_argument(
        "--clip-weights",
        default=None,
        help="Path to OpenCLIP weights dir or file (optional — adds cosine distances)",
    )

    # --- Audit ---
    p_audit = sub.add_parser("audit", help="Statistical coverage audit of augmentations on training images")
    p_audit.add_argument("--images-dir", required=True, help="Directory of training face crops (searched recursively)")
    p_audit.add_argument("--max-images", type=int, default=200, help="Max images to sample (default: 200)")
    p_audit.add_argument("--repeats", type=int, default=5, help="Augmentations per image (default: 5)")
    p_audit.add_argument("--real-captures-dir", default=None, help="Overlay stats from real-world lighting captures (optional)")
    p_audit.add_argument("--shadow-p", type=float, default=0.10, help="DirectionalShadow probability for proposed pipeline (default: 0.10)")
    p_audit.add_argument("--gamma-up-p", type=float, default=0.12, help="GammaUp probability for proposed pipeline (default: 0.12)")
    p_audit.add_argument("--output", default=None, help="Output image path (default: lighting_audit.png)")

    args = parser.parse_args()
    if args.mode == "showcase":
        run_showcase(args)
    elif args.mode == "compare":
        run_compare(args)
    elif args.mode == "audit":
        run_audit(args)


if __name__ == "__main__":
    main()
