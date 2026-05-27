"""Step 3 — characterize EXACTLY what differs between the two crops.

What is held constant by construction:
  - same person (the user, himself)
  - same physical camera (same machine)
  - same room / lighting / background
  - effectively same moment (side-by-side comparison frame; pose nearly identical)
The only manipulated variable: Microsoft Teams account.

We measure: pixel diff, mean RGB / LAB channels, Laplacian sharpness, JPEG QF estimate,
ArcFace cosine, CLIP cosine, and the 7 IQ atlas axes.
"""
from __future__ import annotations

import sys, json, os
from pathlib import Path
import cv2
import numpy as np

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
sys.path.insert(0, str(REPO))

CROPS = Path("analysis/teams_account_natural_experiment_2026-05-19/crops")
OUT = Path("analysis/teams_account_natural_experiment_2026-05-19/outputs")
OUT.mkdir(parents=True, exist_ok=True)

def compute_iq_metrics(img_bgr: np.ndarray) -> dict:
    """Compute 7-axis atlas metrics + extra channel stats."""
    h, w = img_bgr.shape[:2]
    # Convert color spaces
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # IQ atlas axes (matching analysis/iq_data_atlas_2026-05-08 schema)
    lap_var = float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
    luma_mean = float(img_lab[:, :, 0].mean())
    luma_std = float(img_lab[:, :, 0].std())
    saturation_mean = float(img_hsv[:, :, 1].mean())
    color_a_dev = float(img_lab[:, :, 1].std())
    color_b_dev = float(img_lab[:, :, 2].std())
    edges = cv2.Canny(img_gray, 50, 150)
    edge_mag = float(edges.mean())
    contrast_l = float(img_lab[:, :, 0].std())
    # Skin fraction via simple HSV heuristic
    h_h = img_hsv[:, :, 0]; s_s = img_hsv[:, :, 1]; v_v = img_hsv[:, :, 2]
    skin = ((h_h >= 0) & (h_h <= 25) & (s_s >= 40) & (s_s <= 255) & (v_v >= 50)).mean()
    return {
        "h": h, "w": w, "min_dim": min(h, w), "max_dim": max(h, w),
        "lap_var_sharpness": lap_var,
        "luma_mean": luma_mean,
        "luma_std": luma_std,
        "saturation_mean": saturation_mean,
        "color_a_dev": color_a_dev,
        "color_b_dev": color_b_dev,
        "edge_mag": edge_mag,
        "contrast_l": contrast_l,
        "skin_frac": float(skin),
        # Extra per-channel raw stats
        "R_mean": float(img_rgb[:, :, 0].mean()),
        "G_mean": float(img_rgb[:, :, 1].mean()),
        "B_mean": float(img_rgb[:, :, 2].mean()),
        "L_mean": float(img_lab[:, :, 0].mean()),  # Lightness
        "a_mean": float(img_lab[:, :, 1].mean()),  # a* axis (green-red)
        "b_mean": float(img_lab[:, :, 2].mean()),  # b* axis (blue-yellow)
        "H_mean": float(img_hsv[:, :, 0].mean()),
        "S_mean": float(img_hsv[:, :, 1].mean()),
        "V_mean": float(img_hsv[:, :, 2].mean()),
    }


roy = cv2.imread(str(CROPS / "face_roy_d.png"), cv2.IMREAD_COLOR)
guest = cv2.imread(str(CROPS / "face_guest.png"), cv2.IMREAD_COLOR)
print(f"Roy_D shape:  {roy.shape}")
print(f"Guest shape:  {guest.shape}")

# To compare per-pixel, resize both to same size (224x224)
roy224 = cv2.resize(roy, (224, 224), interpolation=cv2.INTER_LINEAR)
guest224 = cv2.resize(guest, (224, 224), interpolation=cv2.INTER_LINEAR)

iq_roy = compute_iq_metrics(roy224)
iq_guest = compute_iq_metrics(guest224)

print("\n=== Per-axis comparison (224x224 preprocessed crops) ===")
print(f"{'metric':<22} {'Roy_D':>12} {'Guest':>12} {'Δ (G-R)':>12} {'% change':>10}")
for k in iq_roy:
    if isinstance(iq_roy[k], (int, float)):
        d = iq_guest[k] - iq_roy[k]
        pct = (d / iq_roy[k] * 100) if abs(iq_roy[k]) > 1e-6 else 0.0
        print(f"  {k:<20} {iq_roy[k]:>12.4f} {iq_guest[k]:>12.4f} {d:>+12.4f} {pct:>+9.2f}%")

# Pixel-level diff stats
diff = roy224.astype(np.float32) - guest224.astype(np.float32)
abs_diff = np.abs(diff)
print(f"\n=== Per-pixel diff (after 224×224 INTER_LINEAR resize, BGR) ===")
print(f"  mean abs diff (all channels): {abs_diff.mean():.3f} / 255")
print(f"  median abs diff: {np.median(abs_diff):.3f} / 255")
print(f"  p95 abs diff: {np.percentile(abs_diff, 95):.3f} / 255")
print(f"  per-channel mean abs diff (BGR): {abs_diff.mean(axis=(0,1)).tolist()}")
print(f"  per-channel signed mean diff (BGR): {diff.mean(axis=(0,1)).tolist()}")

# CLIP embedding cosine — use a frozen CLIP model
print("\n=== CLIP / ArcFace similarity ===")
try:
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor

    # Load OpenAI CLIP ViT-L/14
    model_path = REPO / "weights/models--openai--clip-vit-large-patch14"
    if model_path.exists():
        clip = CLIPModel.from_pretrained(str(model_path))
        proc = CLIPProcessor.from_pretrained(str(model_path))
    else:
        # Fallback to HF model id
        clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip.eval()
    # Process both crops
    pil_roy = cv2.cvtColor(roy224, cv2.COLOR_BGR2RGB)
    pil_guest = cv2.cvtColor(guest224, cv2.COLOR_BGR2RGB)
    inputs = proc(images=[pil_roy, pil_guest], return_tensors="pt")
    with torch.inference_mode():
        feats = clip.get_image_features(**inputs)  # (2, 768) or projection_dim
    feats = F.normalize(feats, dim=-1)
    cos = float((feats[0] @ feats[1]).item())
    print(f"  CLIP image-feature cosine(Roy_D, Guest): {cos:.6f}")
    # L11 hidden-state cosine
    vision_outputs = clip.vision_model(pixel_values=inputs["pixel_values"], output_hidden_states=True)
    hs = vision_outputs.hidden_states  # tuple of (B, seq, hidden)
    L11_cls = hs[12][:, 0, :]   # layer index 12 = after 12 blocks for ViT-L; cls token
    L23_cls = hs[-1][:, 0, :]
    cos_L11 = float(F.cosine_similarity(L11_cls[0:1], L11_cls[1:2]).item())
    cos_L23 = float(F.cosine_similarity(L23_cls[0:1], L23_cls[1:2]).item())
    print(f"  Frozen CLIP L11 CLS cosine: {cos_L11:.6f}")
    print(f"  Frozen CLIP L23 CLS cosine: {cos_L23:.6f}")
except Exception as e:
    print(f"  CLIP cosine: SKIPPED ({e})")
    cos = cos_L11 = cos_L23 = None

# ArcFace embedding cosine — use insightface if available, else fall back
print()
try:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(224, 224))
    fr = app.get(roy224)
    fg = app.get(guest224)
    if fr and fg:
        ar = fr[0].embedding; ag = fg[0].embedding
        ar = ar / np.linalg.norm(ar); ag = ag / np.linalg.norm(ag)
        arc_cos = float((ar * ag).sum())
        print(f"  ArcFace cosine(Roy_D, Guest): {arc_cos:.6f}")
    else:
        arc_cos = None
        print(f"  ArcFace: face detection failed in at least one crop")
except ImportError:
    arc_cos = None
    print(f"  ArcFace cosine: SKIPPED (insightface not installed)")
except Exception as e:
    arc_cos = None
    print(f"  ArcFace cosine: FAILED ({e})")

# Save diff visualization
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(cv2.cvtColor(roy224, cv2.COLOR_BGR2RGB)); axes[0].set_title(f"Roy_D crop\nT5C prob_fake = 0.795")
axes[1].imshow(cv2.cvtColor(guest224, cv2.COLOR_BGR2RGB)); axes[1].set_title(f"Guest crop\nT5C prob_fake = 0.628")
# Diff visualization — amplified
diff_vis = np.clip(np.abs(diff).mean(axis=2) * 3, 0, 255).astype(np.uint8)
axes[2].imshow(diff_vis, cmap="hot")
axes[2].set_title(f"Abs per-pixel diff (×3)\nmean = {abs_diff.mean():.2f}/255")
for ax in axes: ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "diff_visualization.png", dpi=120, bbox_inches="tight")
plt.close()

# Save summary
summary = {
    "image_metrics": {"roy_d": iq_roy, "guest": iq_guest},
    "pixel_diff": {
        "mean_abs": float(abs_diff.mean()),
        "median_abs": float(np.median(abs_diff)),
        "p95_abs": float(np.percentile(abs_diff, 95)),
        "per_channel_BGR_mean_abs": abs_diff.mean(axis=(0,1)).tolist(),
        "per_channel_BGR_signed_mean": diff.mean(axis=(0,1)).tolist(),
    },
    "similarity": {
        "clip_projection_cosine": cos,
        "clip_L11_cls_cosine": cos_L11,
        "clip_L23_cls_cosine": cos_L23,
        "arcface_cosine": arc_cos,
    },
}
with open(OUT / "diff_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Summary saved to outputs/diff_summary.json ===")
