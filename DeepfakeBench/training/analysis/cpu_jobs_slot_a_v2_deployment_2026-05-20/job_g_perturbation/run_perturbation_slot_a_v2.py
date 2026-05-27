"""Job G — replicate 2026-05-19 per-axis perturbation sweep on Slot A v2 step3500.

Reference: analysis/teams_account_natural_experiment_2026-05-19/perturbation_sweep.py
Grids: SAME 8 axes × same per-axis levels (verbatim), so T5C swing numbers are direct
comparators.

Step 5 — operational direction check applies the measured Roy_D → Guest
single-axis factors individually:
  - R ×0.85   (R_mean 180.9 → 153.5)
  - G ×0.87   (G_mean 145.1 → 126.4)
  - B ×0.85   (B_mean 133.5 → 113.4)
  - blur σ=1.0   (lap_var 113 → ~56)

Outputs:
  outputs/perturbation_sweep_slot_a_v2.csv     # schema = 2026-05-19 perturbation_sweep.csv
  outputs/operational_single_axis_slot_a_v2.csv
  outputs/swing_comparison.csv
  outputs/summary.json
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

CROPS = REPO / "analysis/teams_account_natural_experiment_2026-05-19/crops"
OUT = REPO / "analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_g_perturbation/outputs"
OUT.mkdir(parents=True, exist_ok=True)

DETECTOR_CONFIG = REPO / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO / "config/defaults.yaml"
DEVICE = torch.device("cpu")
RES = 224

# Slot A v2 step3500 — FT from T5C, no LoRA, classifier hidden_dim=1024 (same as T5C)
SLOT_A_V2_CKPT = REPO / "analysis/manual_canary_2026-05-20/ckpts/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth"

CKPT_NAME = "SLOT_A_V2_STEP3500"

# T5C swings from the 2026-05-19 sweep (computed from
# analysis/teams_account_natural_experiment_2026-05-19/outputs/perturbation_sweep.csv).
# These match the task's reference table.
T5C_SWINGS_REFERENCE = {
    "gamma":          0.092,
    "blur_sigma":     0.373,
    "R_scale":        0.259,
    "G_scale":        0.678,
    "B_scale":        0.497,
    "saturation":     0.030,
    "contrast":       0.428,
    "brightness_add": 0.199,
}

# Source crops
roy_bgr = cv2.imread(str(CROPS / "face_roy_d.png"), cv2.IMREAD_COLOR)
guest_bgr = cv2.imread(str(CROPS / "face_guest.png"), cv2.IMREAD_COLOR)
assert roy_bgr is not None, f"missing {CROPS / 'face_roy_d.png'}"
assert guest_bgr is not None, f"missing {CROPS / 'face_guest.png'}"


def score(img_bgr: np.ndarray, model) -> float:
    img = cv2.resize(img_bgr, (RES, RES), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])
    x = transform(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        out = model({"image": x}, inference=True)
    return float(out["prob"].detach().cpu().numpy().reshape(-1)[0])


# ============================================================================
# Perturbation library — VERBATIM from 2026-05-19 perturbation_sweep.py
# ============================================================================
def perturb_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    f = (img.astype(np.float32) / 255.0) ** gamma
    return np.clip(f * 255, 0, 255).astype(np.uint8)


def perturb_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img.copy()
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(img, (k, k), sigma)


def perturb_channel(img: np.ndarray, channel: str, scale: float) -> np.ndarray:
    out = img.astype(np.float32).copy()
    idx = {"B": 0, "G": 1, "R": 2}[channel]
    out[:, :, idx] = np.clip(out[:, :, idx] * scale, 0, 255)
    return out.astype(np.uint8)


def perturb_saturation(img: np.ndarray, scale: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def perturb_contrast(img: np.ndarray, scale: float) -> np.ndarray:
    f = img.astype(np.float32)
    f = (f - 128) * scale + 128
    return np.clip(f, 0, 255).astype(np.uint8)


def perturb_brightness_add(img: np.ndarray, offset: float) -> np.ndarray:
    f = img.astype(np.float32) + offset
    return np.clip(f, 0, 255).astype(np.uint8)


# VERBATIM grids from 2026-05-19 perturbation_sweep.py
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


# ============================================================================
# Load Slot A v2 step3500
# ============================================================================
print(f"=== Loading {CKPT_NAME} ===")
t0 = time.time()
model = load_model(str(SLOT_A_V2_CKPT), str(DETECTOR_CONFIG), str(TRAIN_CONFIG), DEVICE)
model.eval()
print(f"  loaded in {time.time()-t0:.1f}s")

# Baselines
print("\n=== Baselines (Slot A v2 step3500) ===")
baseline_roy = score(roy_bgr, model)
baseline_guest = score(guest_bgr, model)
print(f"  Roy_D: {baseline_roy:.4f}")
print(f"  Guest: {baseline_guest:.4f}")

# ============================================================================
# Per-axis sweep — same schema as 2026-05-19 perturbation_sweep.csv
# ============================================================================
results = []
for axis_name, spec in SWEEPS.items():
    print(f"\n=== Sweep {axis_name} ===")
    fn = spec["fn"]
    for v in spec["values"]:
        perturbed = fn(roy_bgr, v)
        iq = iq_metrics(perturbed)
        p = score(perturbed, model)
        results.append({
            "axis": axis_name, "value": v,
            "ckpt": CKPT_NAME,
            "prob_fake": p,
            **iq,
        })
        print(f"  {axis_name}={v:>6}  lap_var={iq['lap_var']:>7.2f}  luma={iq['luma_mean']:>6.1f}  prob_fake={p:.4f}")

import pandas as pd
df = pd.DataFrame(results)
df.to_csv(OUT / "perturbation_sweep_slot_a_v2.csv", index=False)
print(f"\nWrote {OUT / 'perturbation_sweep_slot_a_v2.csv'} ({len(df)} rows)")

# ============================================================================
# Step 4 — swing summary + comparison vs T5C
# ============================================================================
print(f"\n=== Slot A v2 per-axis swings (vs T5C reference) ===")
swing_rows = []
print(f"{'axis':<16} {'min_p':>8} {'max_p':>8} {'swing_new':>10} {'swing_T5C':>10} {'delta':>10} {'pct':>10}")
for axis in SWEEPS:
    sub = df[df["axis"] == axis]
    pmin, pmax = float(sub["prob_fake"].min()), float(sub["prob_fake"].max())
    swing_new = pmax - pmin
    swing_t5c = T5C_SWINGS_REFERENCE[axis]
    delta = swing_new - swing_t5c
    pct = (delta / swing_t5c) if swing_t5c > 0 else float("nan")
    vmin = float(sub.loc[sub["prob_fake"].idxmin(), "value"])
    vmax = float(sub.loc[sub["prob_fake"].idxmax(), "value"])
    swing_rows.append({
        "axis": axis,
        "slot_a_v2_min_prob": pmin,
        "slot_a_v2_max_prob": pmax,
        "slot_a_v2_swing": swing_new,
        "slot_a_v2_value_min": vmin,
        "slot_a_v2_value_max": vmax,
        "t5c_swing_reference": swing_t5c,
        "delta_swing": delta,
        "delta_over_t5c": pct,
    })
    print(f"  {axis:<14} {pmin:>8.4f} {pmax:>8.4f} {swing_new:>10.4f} {swing_t5c:>10.4f} {delta:>+10.4f} {pct:>+10.2%}")

swing_df = pd.DataFrame(swing_rows)
swing_df.to_csv(OUT / "swing_comparison.csv", index=False)
print(f"\nWrote {OUT / 'swing_comparison.csv'}")

# ============================================================================
# Step 5 — operational single-axis check: Roy_D → Guest factors individually
# ============================================================================
print("\n=== Step 5: operational single-axis (Roy_D → Guest measured factors) ===")
operational_perturbs = [
    ("baseline_roy",       lambda i: i.copy(),                      None),
    ("R_x0.85",            lambda i: perturb_channel(i, "R", 0.85), "R 180.9 → 153.5 (×0.849)"),
    ("G_x0.87",            lambda i: perturb_channel(i, "G", 0.87), "G 145.1 → 126.4 (×0.871)"),
    ("B_x0.85",            lambda i: perturb_channel(i, "B", 0.85), "B 133.5 → 113.4 (×0.849)"),
    ("blur_sigma_1.0",     lambda i: perturb_blur(i, 1.0),           "lap_var 113 → ~56 (factor 0.5)"),
    ("baseline_guest",     lambda i: guest_bgr.copy(),               "actual Guest crop (reference)"),
]
op_rows = []
print(f"{'recipe':<22} {'prob_fake':>10} {'lap_var':>8} {'R_mean':>8} {'G_mean':>8} {'B_mean':>8}   note")
for name, fn, note in operational_perturbs:
    perturbed = fn(roy_bgr)
    iq = iq_metrics(perturbed)
    p = score(perturbed, model)
    op_rows.append({
        "recipe": name, "prob_fake": p,
        "delta_vs_baseline": p - baseline_roy,
        **iq,
        "note": note or "",
    })
    print(f"  {name:<20} {p:>10.4f} {iq['lap_var']:>8.2f} {iq['R_mean']:>8.2f} {iq['G_mean']:>8.2f} {iq['B_mean']:>8.2f}   {note or ''}")
op_df = pd.DataFrame(op_rows)
op_df.to_csv(OUT / "operational_single_axis_slot_a_v2.csv", index=False)
print(f"\nWrote {OUT / 'operational_single_axis_slot_a_v2.csv'}")

# Direction check: which single axis moves prob_fake furthest in the Guest direction
guest_direction = "down" if baseline_guest < baseline_roy else "up"
print(f"\nDirection from Roy_D baseline {baseline_roy:.4f} to Guest {baseline_guest:.4f}: {guest_direction}")
single_axis_only = op_df[~op_df["recipe"].isin(["baseline_roy", "baseline_guest"])].copy()
single_axis_only["abs_delta"] = single_axis_only["delta_vs_baseline"].abs()
strongest = single_axis_only.sort_values("abs_delta", ascending=False).iloc[0]
print(f"Strongest single-axis driver: {strongest['recipe']}  delta={strongest['delta_vs_baseline']:+.4f}")

# ============================================================================
# Step 6 — Mechanical bars
# ============================================================================
g_swing_new = float(swing_df.loc[swing_df["axis"] == "G_scale", "slot_a_v2_swing"].iloc[0])
g_swing_t5c = T5C_SWINGS_REFERENCE["G_scale"]
mean_swing_new = float(swing_df["slot_a_v2_swing"].mean())
mean_swing_t5c = float(np.mean(list(T5C_SWINGS_REFERENCE.values())))

bar1 = g_swing_new < 0.50 * g_swing_t5c
bar2 = mean_swing_new < 0.50 * mean_swing_t5c
bar3_violations = []
for row in swing_df.to_dict(orient="records"):
    if row["t5c_swing_reference"] > 0 and row["slot_a_v2_swing"] > 1.5 * row["t5c_swing_reference"]:
        bar3_violations.append({
            "axis": row["axis"],
            "slot_a_v2_swing": row["slot_a_v2_swing"],
            "t5c_swing": row["t5c_swing_reference"],
            "ratio_vs_t5c": row["slot_a_v2_swing"] / row["t5c_swing_reference"],
        })

print("\n=== Mechanical bars ===")
print(f"Bar 1 (G_scale swing < 50% of T5C 0.678):  Slot A v2 G_scale swing = {g_swing_new:.4f}  threshold = {0.50*g_swing_t5c:.4f}  ->  {'MET' if bar1 else 'NOT MET'}")
print(f"Bar 2 (mean 8-axis swing < 50% of T5C mean {mean_swing_t5c:.4f}):  Slot A v2 mean = {mean_swing_new:.4f}  threshold = {0.50*mean_swing_t5c:.4f}  ->  {'MET' if bar2 else 'NOT MET'}")
if bar3_violations:
    print(f"Bar 3 (any axis swing > T5C × 1.5):  AMPLIFIED on {len(bar3_violations)} axis/axes  ->  TRIGGERED")
    for v in bar3_violations:
        print(f"    {v['axis']}: Slot A v2 {v['slot_a_v2_swing']:.4f} vs T5C {v['t5c_swing']:.4f}  ratio={v['ratio_vs_t5c']:.2f}x")
else:
    print("Bar 3 (any axis swing > T5C × 1.5):  no axis triggered amplification  ->  NOT TRIGGERED")

# ============================================================================
# Summary JSON
# ============================================================================
summary = {
    "ckpt": CKPT_NAME,
    "ckpt_path": str(SLOT_A_V2_CKPT),
    "baselines": {"Roy_D": baseline_roy, "Guest": baseline_guest},
    "g_scale_swing_new": g_swing_new,
    "g_scale_swing_t5c_reference": g_swing_t5c,
    "mean_swing_new": mean_swing_new,
    "mean_swing_t5c_reference": mean_swing_t5c,
    "bar1_g_scale_invariance_met": bool(bar1),
    "bar2_composite_invariance_met": bool(bar2),
    "bar3_amplification_triggered": bool(bar3_violations),
    "bar3_amplification_axes": bar3_violations,
    "swing_comparison": swing_df.to_dict(orient="records"),
    "operational_single_axis": op_df.to_dict(orient="records"),
    "strongest_single_axis_driver": {
        "recipe": strongest["recipe"],
        "delta_vs_baseline": float(strongest["delta_vs_baseline"]),
    },
    "guest_direction_from_roy": guest_direction,
}
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nWrote {OUT / 'summary.json'}")
print("\nDONE")
