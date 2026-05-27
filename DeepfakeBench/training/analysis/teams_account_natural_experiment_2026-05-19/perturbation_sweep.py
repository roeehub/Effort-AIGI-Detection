"""Cheap follow-up — per-axis controlled perturbation sweep on Roy_D crop.

For each of 7 axes (gamma/brightness, Gaussian blur, R/G/B channel scale, saturation,
contrast), apply a monotone range of perturbations to the Roy_D crop and score with
T5C step3500. Tells us which axis the model is most sensitive to.

Also runs a "reverse-engineering" experiment: apply a bundled perturbation
(blur + dim + decontrast) that approximates the Roy_D → Guest shift, and check
if the perturbed Roy_D's prob_fake matches the actual Guest's prob_fake (0.628).
"""
from __future__ import annotations

import sys, json, time
from pathlib import Path
import numpy as np
import cv2
import torch
import torchvision.transforms as T

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
sys.path.insert(0, str(REPO))

from arena.model_arena import CLIP_MEAN, CLIP_STD, load_model  # noqa

CROPS = Path("analysis/teams_account_natural_experiment_2026-05-19/crops")
OUT = Path("analysis/teams_account_natural_experiment_2026-05-19/outputs")
OUT.mkdir(parents=True, exist_ok=True)
LOCAL_CKPT_CACHE = Path("analysis/slot_b_property_shortcut_2026-05-16/ckpt_cache")

DETECTOR_CONFIG = REPO / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO / "config/defaults.yaml"
DEVICE = torch.device("cpu")
RES = 224

CKPTS = {
    "T5C_STEP3500": LOCAL_CKPT_CACHE / "periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth",
    "SLOT_B_6AXIS_GRL": LOCAL_CKPT_CACHE / "periodic_effort_20260516_step3500_auc0.9910_eer0.0175.pth",
}

# Source crops (raw resolution; we'll resize at scoring time)
roy_bgr = cv2.imread(str(CROPS / "face_roy_d.png"), cv2.IMREAD_COLOR)
guest_bgr = cv2.imread(str(CROPS / "face_guest.png"), cv2.IMREAD_COLOR)
roy_224 = cv2.resize(roy_bgr, (RES, RES), interpolation=cv2.INTER_LINEAR)
guest_224 = cv2.resize(guest_bgr, (RES, RES), interpolation=cv2.INTER_LINEAR)


def score(img_bgr: np.ndarray, model) -> float:
    """Preprocess and score one BGR crop."""
    img = cv2.resize(img_bgr, (RES, RES), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])
    x = transform(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        out = model({"image": x}, inference=True)
    return float(out["prob"].detach().cpu().numpy().reshape(-1)[0])


# ============================================================================
# Perturbation library
# ============================================================================
def perturb_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction; gamma<1 brightens, gamma>1 darkens."""
    f = (img.astype(np.float32) / 255.0) ** gamma
    return np.clip(f * 255, 0, 255).astype(np.uint8)


def perturb_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with given sigma."""
    if sigma <= 0:
        return img.copy()
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(img, (k, k), sigma)


def perturb_channel(img: np.ndarray, channel: str, scale: float) -> np.ndarray:
    """Multiplicative scale on one BGR channel."""
    out = img.astype(np.float32).copy()
    idx = {"B": 0, "G": 1, "R": 2}[channel]
    out[:, :, idx] = np.clip(out[:, :, idx] * scale, 0, 255)
    return out.astype(np.uint8)


def perturb_saturation(img: np.ndarray, scale: float) -> np.ndarray:
    """Scale saturation in HSV; scale<1 = more grey, scale>1 = more saturated."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def perturb_contrast(img: np.ndarray, scale: float) -> np.ndarray:
    """Scale contrast around 128 mid-gray."""
    f = img.astype(np.float32)
    f = (f - 128) * scale + 128
    return np.clip(f, 0, 255).astype(np.uint8)


def perturb_brightness_add(img: np.ndarray, offset: float) -> np.ndarray:
    """Additive brightness offset (in [-50, 50] grayscale levels)."""
    f = img.astype(np.float32) + offset
    return np.clip(f, 0, 255).astype(np.uint8)


# Sweeps to run on Roy_D (baseline)
SWEEPS = {
    "gamma":         {"fn": perturb_gamma,        "values": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7]},
    "blur_sigma":    {"fn": perturb_blur,         "values": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]},
    "R_scale":       {"fn": lambda i, v: perturb_channel(i, "R", v), "values": [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]},
    "G_scale":       {"fn": lambda i, v: perturb_channel(i, "G", v), "values": [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]},
    "B_scale":       {"fn": lambda i, v: perturb_channel(i, "B", v), "values": [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]},
    "saturation":    {"fn": perturb_saturation,   "values": [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7]},
    "contrast":      {"fn": perturb_contrast,     "values": [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7]},
    "brightness_add":{"fn": perturb_brightness_add,"values": [-50, -30, -20, -10, 0, 10, 20, 30, 50]},
}


def iq_metrics(img_bgr: np.ndarray) -> dict:
    """Light per-axis metrics for tracking."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    return {
        "lap_var": float(cv2.Laplacian(g, cv2.CV_64F).var()),
        "luma_mean": float(lab[:, :, 0].mean()),
        "sat_mean": float(hsv[:, :, 1].mean()),
        "color_a_dev": float(lab[:, :, 1].std()),
        "R_mean": float(img_bgr[:, :, 2].mean()),
        "G_mean": float(img_bgr[:, :, 1].mean()),
        "B_mean": float(img_bgr[:, :, 0].mean()),
    }


# Baseline scores (Roy_D, Guest, on each ckpt)
print("=== Loading models ===")
models = {}
for name, path in CKPTS.items():
    print(f"  loading {name}")
    t0 = time.time()
    models[name] = load_model(str(path), str(DETECTOR_CONFIG), str(TRAIN_CONFIG), DEVICE)
    models[name].eval()
    print(f"    loaded in {time.time()-t0:.1f}s")

baselines = {}
for name, model in models.items():
    baselines[name] = {
        "Roy_D":   score(roy_bgr, model),
        "Guest":   score(guest_bgr, model),
    }
print("\n=== Baselines ===")
print(json.dumps(baselines, indent=2))

# ============================================================================
# Sweep
# ============================================================================
results = []
for axis_name, spec in SWEEPS.items():
    print(f"\n=== Sweep {axis_name} ===")
    fn = spec["fn"]
    for v in spec["values"]:
        perturbed = fn(roy_bgr, v)
        iq = iq_metrics(perturbed)
        for ckpt_name, model in models.items():
            p = score(perturbed, model)
            results.append({
                "axis": axis_name, "value": v,
                "ckpt": ckpt_name,
                "prob_fake": p,
                **iq,
            })
        # Print for T5C only
        t5c_p = results[-len(models)]["prob_fake"]
        slot_b_p = results[-len(models)+1]["prob_fake"] if len(models) > 1 else None
        print(f"  {axis_name}={v:>6}  lap_var={iq['lap_var']:>7.2f}  luma={iq['luma_mean']:>6.1f}  T5C={t5c_p:.4f}  SlotB={slot_b_p:.4f}")

import pandas as pd
df = pd.DataFrame(results)
df.to_csv(OUT / "perturbation_sweep.csv", index=False)
print(f"\nWrote {OUT / 'perturbation_sweep.csv'} ({len(df)} rows)")

# ============================================================================
# Per-axis sensitivity summary
# ============================================================================
print("\n=== Sensitivity per axis (T5C only) ===")
print(f"{'axis':<18} {'baseline':>10} {'min_p':>8} {'max_p':>8} {'swing':>8} {'value_min':>10} {'value_max':>10}")
for axis in SWEEPS:
    sub = df[(df["axis"] == axis) & (df["ckpt"] == "T5C_STEP3500")]
    base = baselines["T5C_STEP3500"]["Roy_D"]
    pmin, pmax = sub["prob_fake"].min(), sub["prob_fake"].max()
    swing = pmax - pmin
    vmin = float(sub.loc[sub["prob_fake"].idxmin(), "value"])
    vmax = float(sub.loc[sub["prob_fake"].idxmax(), "value"])
    print(f"  {axis:<16} {base:>10.4f} {pmin:>8.4f} {pmax:>8.4f} {swing:>8.4f} {vmin:>10.3f} {vmax:>10.3f}")

# ============================================================================
# Bundle perturbation — try to reproduce Roy_D → Guest shift
# ============================================================================
print("\n=== Reverse-engineering: simulate Guest pipeline on Roy_D ===")
# Observed Roy_D → Guest changes:
#  lap_var:    113 → 56 (factor 0.50) -- apply Gaussian blur
#  luma_mean:  159 → 139 (factor 0.876)  -- apply gamma > 1 or scale down
#  contrast:   60 → 46 (factor 0.762)
#  R_mean:     180 → 153 (factor 0.85)
#  G_mean:     145 → 126 (factor 0.87)
#  B_mean:     133 → 113 (factor 0.85)
# Net effect roughly:  scale all channels by ~0.85 (uniform dim) + Gaussian blur sigma ~1.5

# Try a few candidate bundles
def gauss_then_scale(img, sigma, scale):
    """Blur then scale all channels."""
    img2 = perturb_blur(img, sigma) if sigma > 0 else img.copy()
    f = img2.astype(np.float32) * scale
    return np.clip(f, 0, 255).astype(np.uint8)


bundle_recipes = {
    "blur_sigma=1.0_scale=0.85":  lambda i: gauss_then_scale(i, 1.0, 0.85),
    "blur_sigma=1.5_scale=0.85":  lambda i: gauss_then_scale(i, 1.5, 0.85),
    "blur_sigma=2.0_scale=0.85":  lambda i: gauss_then_scale(i, 2.0, 0.85),
    "blur_sigma=1.5_scale=0.80":  lambda i: gauss_then_scale(i, 1.5, 0.80),
    "blur_sigma=1.5_only":        lambda i: perturb_blur(i, 1.5),
    "scale=0.85_only":            lambda i: gauss_then_scale(i, 0.0, 0.85),
    "blur_sigma=2.5_scale=0.80":  lambda i: gauss_then_scale(i, 2.5, 0.80),
}

bundle_results = []
for name, fn in bundle_recipes.items():
    perturbed = fn(roy_bgr)
    iq = iq_metrics(perturbed)
    row = {"recipe": name, **iq}
    for ckpt_name, model in models.items():
        row[f"prob_fake_{ckpt_name}"] = score(perturbed, model)
    bundle_results.append(row)
bundle_df = pd.DataFrame(bundle_results)
bundle_df.to_csv(OUT / "bundle_sweep.csv", index=False)
print(bundle_df.to_string(index=False))
print()
print(f"Target Guest scores: T5C={baselines['T5C_STEP3500']['Guest']:.4f}  SlotB={baselines['SLOT_B_6AXIS_GRL']['Guest']:.4f}")
print(f"Baseline Roy_D scores: T5C={baselines['T5C_STEP3500']['Roy_D']:.4f}  SlotB={baselines['SLOT_B_6AXIS_GRL']['Roy_D']:.4f}")

# ============================================================================
# CLIP CLS cosine (fixed)
# ============================================================================
print("\n=== Frozen CLIP CLS cosine ===")
try:
    from transformers import CLIPModel, CLIPProcessor
    import torch.nn.functional as F
    model_path = REPO / "weights/models--openai--clip-vit-large-patch14"
    if model_path.exists():
        clip = CLIPModel.from_pretrained(str(model_path))
        proc = CLIPProcessor.from_pretrained(str(model_path))
    else:
        clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip.eval()
    pil_roy = cv2.cvtColor(roy_224, cv2.COLOR_BGR2RGB)
    pil_guest = cv2.cvtColor(guest_224, cv2.COLOR_BGR2RGB)
    inputs = proc(images=[pil_roy, pil_guest], return_tensors="pt")
    with torch.inference_mode():
        # Use vision_model with output_hidden_states properly
        vout = clip.vision_model(pixel_values=inputs["pixel_values"], output_hidden_states=True, return_dict=True)
        hs = vout.hidden_states  # tuple of length num_layers+1
        # ViT-L has 24 transformer blocks; hs[i] = after block i
        # CLS token is index 0
        L11 = hs[12][:, 0, :]
        L23 = hs[-1][:, 0, :]
        proj = clip.get_image_features(**inputs)
    cos_L11 = float(F.cosine_similarity(L11[0:1], L11[1:2]).item())
    cos_L23 = float(F.cosine_similarity(L23[0:1], L23[1:2]).item())
    cos_proj = float(F.cosine_similarity(F.normalize(proj[0:1]), F.normalize(proj[1:2])).item())
    print(f"  L11 CLS cosine(Roy_D, Guest):       {cos_L11:.6f}")
    print(f"  L23 CLS cosine(Roy_D, Guest):       {cos_L23:.6f}")
    print(f"  Projection-head cosine(Roy_D, Guest): {cos_proj:.6f}")
except Exception as e:
    cos_L11 = cos_L23 = cos_proj = None
    print(f"  CLIP cosine FAILED: {e}")
    import traceback; traceback.print_exc()

# ============================================================================
# ArcFace cosine via insightface
# ============================================================================
print("\n=== ArcFace cosine via insightface ===")
try:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    # Run on the FULL PANELS, not the cropped faces (insightface includes its own detector)
    roy_panel = cv2.imread(str(CROPS / "panel_roy_d.png"), cv2.IMREAD_COLOR)
    guest_panel = cv2.imread(str(CROPS / "panel_guest.png"), cv2.IMREAD_COLOR)
    fr = app.get(roy_panel)
    fg = app.get(guest_panel)
    print(f"  Roy_D faces detected: {len(fr)}")
    print(f"  Guest faces detected: {len(fg)}")
    if fr and fg:
        ar = fr[0].embedding
        ag = fg[0].embedding
        ar = ar / np.linalg.norm(ar)
        ag = ag / np.linalg.norm(ag)
        arc_cos = float((ar * ag).sum())
        print(f"  ArcFace cosine(Roy_D, Guest): {arc_cos:.6f}")
        # Also dimensionality
        print(f"  ArcFace embedding dim: {len(ar)}")
    else:
        arc_cos = None
        print(f"  ArcFace: face detection failed")
except Exception as e:
    arc_cos = None
    print(f"  ArcFace FAILED: {e}")
    import traceback; traceback.print_exc()


# ============================================================================
# Summary
# ============================================================================
final_summary = {
    "baselines": baselines,
    "clip_cosines": {
        "L11_cls": cos_L11,
        "L23_cls": cos_L23,
        "projection_head": cos_proj,
    },
    "arcface_cosine": arc_cos,
}
with open(OUT / "followup_summary.json", "w") as f:
    json.dump(final_summary, f, indent=2)

print("\n=== FINAL SUMMARY ===")
print(json.dumps(final_summary, indent=2))
print("\nDONE")
