"""CPU Gates for R13_T5C_ANCHOR_AWARE_PLUS_CODEC_2026-05-21.

Three pre-launch sanity gates on the Roy_D natural-experiment crop:

  Gate 1 — natural-experiment VALIDITY.
    Apply VideoCodecSimulation (codec_quality=(20,65), the P8A teams_codec_sim
    setting) to face_roy_d.png and score with Slot A v2 step3500 + T5C step3500.
    PASS if the median augmented Roy_D score lands within ±0.10 of Guest's
    reference score (Slot A v2: 0.640; T5C: 0.628) — i.e. the aug at least
    half-closes the per-Teams-account transport gap.

  Gate 3 — saturation swing sanity.
    Sweep VideoCodecSimulation across codec_quality (q=15, 30, 50, 70, 90).
    Report mean / max swing in Slot A v2's prob_fake. PASS if max swing across
    realistic q range ≤ 0.30 (i.e. the aug doesn't bimodally flip the model;
    Slot A v2's full Roy_D↔Guest swing 0.168 is the natural baseline).

  Gate 2 (AUC preservation) deferred to in-flight wandb canary on the
  3 launched GPU jobs — runs every 500 steps with built-in fake-recall and
  chronic-FP monitors.

Output:
  analysis/codec_restoration_gates_2026-05-21/gates_summary.json
  analysis/codec_restoration_gates_2026-05-21/gates_per_run.csv
"""
from __future__ import annotations

import sys, json, time, csv
from pathlib import Path
import numpy as np
import cv2
import torch
import torchvision.transforms as T

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
sys.path.insert(0, str(REPO))

from arena.model_arena import CLIP_MEAN, CLIP_STD, _download_checkpoint, load_model  # noqa
from data.augmentations.transforms import VideoCodecSimulation  # noqa

OUT_DIR = REPO / "analysis/codec_restoration_gates_2026-05-21"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR = REPO / "analysis/teams_account_natural_experiment_2026-05-19/crops"
LOCAL_CKPT_CACHE = REPO / "analysis/slot_b_property_shortcut_2026-05-16/ckpt_cache"
DETECTOR_CONFIG = REPO / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO / "config/defaults.yaml"
DEVICE = torch.device("cpu")
RES = 224

# Reference Roy_D + Guest crops from the 2026-05-19 natural experiment
ROY_D_PATH = CROPS_DIR / "face_roy_d.png"
GUEST_PATH = CROPS_DIR / "face_guest.png"

# Reference scores from the 2026-05-19 natural experiment + Job A (2026-05-20)
# T5C step3500   : Roy_D=0.795 Guest=0.628 (Δ=−0.168)
# Slot A v2 step3500: Roy_D≈0.808 Guest≈0.640 (Δ identical to T5C to 4dp per Job A)
REFERENCE_SCORES = {
    "SLOT_A_V2_STEP3500": {"Roy_D": 0.808, "Guest": 0.640, "delta": -0.168},
    "T5C_STEP3500":       {"Roy_D": 0.795, "Guest": 0.628, "delta": -0.168},
}

CKPTS = [
    {
        "name": "SLOT_A_V2_STEP3500",
        "local_path": LOCAL_CKPT_CACHE / "periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth",
        "gcs_uri":    "gs://training-job-outputs/best_checkpoints/hp35c51p/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth",
    },
    {
        "name": "T5C_STEP3500",
        "local_path": LOCAL_CKPT_CACHE / "periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth",
        "gcs_uri":    "gs://training-job-outputs/best_checkpoints/jrlldtem/periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth",
    },
]


def load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    # Resize to 224 with INTER_LINEAR (matches training preprocessing)
    return cv2.resize(img, (RES, RES), interpolation=cv2.INTER_LINEAR)


def bgr_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])
    return transform(img_rgb).unsqueeze(0)


def apply_codec(img_bgr: np.ndarray, codec_quality: tuple[int, int], seed: int = 0) -> np.ndarray:
    """Apply VideoCodecSimulation matching training's albumentations call.

    The training pipeline passes RGB to albumentations. We follow suit:
    BGR → RGB → augment → BGR.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    np.random.seed(seed)
    import random as _random
    _random.seed(seed)
    aug = VideoCodecSimulation(codec_quality=codec_quality, p=1.0, always_apply=True)
    out_rgb = aug.apply(img_rgb)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def score(model, img_bgr: np.ndarray) -> float:
    x = bgr_to_tensor(img_bgr).to(DEVICE)
    with torch.inference_mode():
        out = model({"image": x}, inference=True)
    return float(out["prob"].detach().cpu().numpy().reshape(-1)[0])


def resolve_ckpt(c: dict) -> Path:
    if c["local_path"].exists():
        return c["local_path"]
    return Path(_download_checkpoint(c["gcs_uri"]))


# ─── Main ─────────────────────────────────────────────────────────────
print("=" * 70)
print(f"Codec Restoration CPU Gates — {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
print("=" * 70)

# Load both crops once (constant baseline)
roy_d_bgr = load_bgr(ROY_D_PATH)
guest_bgr = load_bgr(GUEST_PATH)
print(f"Roy_D crop: {roy_d_bgr.shape}, Guest crop: {guest_bgr.shape}")
print(f"Reference: {json.dumps(REFERENCE_SCORES, indent=2)}")

per_run_rows = []
gate_summary = {}

for c in CKPTS:
    name = c["name"]
    print(f"\n=== {name} ===")
    t0 = time.time()
    local = resolve_ckpt(c)
    print(f"  loading from {local} ...")
    model = load_model(str(local), str(DETECTOR_CONFIG), str(TRAIN_CONFIG), DEVICE)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    # Baseline scores (no aug — should match reference table within preprocessing precision)
    base_roy = score(model, roy_d_bgr)
    base_guest = score(model, guest_bgr)
    print(f"  Baseline (no aug): Roy_D={base_roy:.4f} Guest={base_guest:.4f} Δ={base_guest-base_roy:+.4f}")
    print(f"  Reference:        Roy_D={REFERENCE_SCORES[name]['Roy_D']:.4f} Guest={REFERENCE_SCORES[name]['Guest']:.4f}")

    per_run_rows.append({
        "ckpt": name, "crop": "Roy_D",        "codec_quality": "no_aug", "seed": -1, "score": base_roy,
    })
    per_run_rows.append({
        "ckpt": name, "crop": "Guest",        "codec_quality": "no_aug", "seed": -1, "score": base_guest,
    })

    # ─── GATE 1: natural-experiment validity (codec_quality=[20, 65], P8A dose) ─
    print(f"\n  -- Gate 1: codec_quality=(20, 65), 12 seeds --")
    g1_aug_scores = []
    for seed in range(12):
        aug_bgr = apply_codec(roy_d_bgr, codec_quality=(20, 65), seed=seed)
        s = score(model, aug_bgr)
        g1_aug_scores.append(s)
        per_run_rows.append({
            "ckpt": name, "crop": "Roy_D_aug", "codec_quality": "20-65", "seed": seed, "score": s,
        })
        print(f"    seed={seed:2d}  score={s:.4f}  Δ_vs_base={s-base_roy:+.4f}  Δ_vs_guest={s-base_guest:+.4f}")
    g1_med = float(np.median(g1_aug_scores))
    g1_mean = float(np.mean(g1_aug_scores))
    g1_min, g1_max = float(min(g1_aug_scores)), float(max(g1_aug_scores))
    guest_gap = abs(g1_med - base_guest)
    g1_pass = guest_gap <= 0.10
    print(f"  Gate 1 SUMMARY: aug_median={g1_med:.4f}  guest_ref={base_guest:.4f}  |gap|={guest_gap:.4f}")
    print(f"  Gate 1 VERDICT: {'PASS' if g1_pass else 'FAIL'} (criterion: |aug_med − guest_baseline| ≤ 0.10)")

    # ─── GATE 3: saturation swing across codec_quality ─
    print(f"\n  -- Gate 3: codec_quality sweep [15..90], 3 seeds each --")
    g3_data = {}
    for q_lo, q_hi, label in [
        (10, 20, "q=10-20 (extreme)"),
        (20, 35, "q=20-35 (aggressive)"),
        (30, 50, "q=30-50 (medium)"),
        (50, 70, "q=50-70 (light)"),
        (70, 90, "q=70-90 (very light)"),
    ]:
        scores_this_band = []
        for seed in range(3):
            aug_bgr = apply_codec(roy_d_bgr, codec_quality=(q_lo, q_hi), seed=100 + seed)
            s = score(model, aug_bgr)
            scores_this_band.append(s)
            per_run_rows.append({
                "ckpt": name, "crop": "Roy_D_aug", "codec_quality": f"{q_lo}-{q_hi}", "seed": 100 + seed, "score": s,
            })
        m = float(np.median(scores_this_band))
        g3_data[label] = {"min": float(min(scores_this_band)), "max": float(max(scores_this_band)),
                          "median": m, "n": 3, "q_range": [q_lo, q_hi]}
        print(f"    {label:>22s}  scores={[f'{s:.3f}' for s in scores_this_band]}  median={m:.4f}")

    # Across all bands, what is the max-min swing of medians?
    band_medians = [v["median"] for v in g3_data.values()]
    g3_swing = float(max(band_medians) - min(band_medians))
    g3_pass = g3_swing <= 0.30
    print(f"  Gate 3 SUMMARY: median-of-medians swing across q-bands = {g3_swing:.4f}")
    print(f"  Gate 3 VERDICT: {'PASS' if g3_pass else 'FAIL'} (criterion: swing ≤ 0.30)")

    gate_summary[name] = {
        "baseline_roy_d": base_roy,
        "baseline_guest": base_guest,
        "baseline_delta": base_guest - base_roy,
        "reference_roy_d": REFERENCE_SCORES[name]["Roy_D"],
        "reference_guest": REFERENCE_SCORES[name]["Guest"],
        "gate1": {
            "aug_median": g1_med,
            "aug_mean": g1_mean,
            "aug_min": g1_min,
            "aug_max": g1_max,
            "guest_gap_abs": guest_gap,
            "n_seeds": len(g1_aug_scores),
            "pass": g1_pass,
            "criterion": "|aug_median - guest_baseline| <= 0.10",
        },
        "gate3": {
            "per_band": g3_data,
            "band_median_swing": g3_swing,
            "pass": g3_pass,
            "criterion": "band_median_swing <= 0.30",
        },
    }

    del model
    print()

# Persist
with open(OUT_DIR / "gates_summary.json", "w") as f:
    json.dump(gate_summary, f, indent=2)
with open(OUT_DIR / "gates_per_run.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ckpt", "crop", "codec_quality", "seed", "score"])
    w.writeheader()
    w.writerows(per_run_rows)

print("=" * 70)
print("ALL GATES COMPLETE")
print("=" * 70)
for name, g in gate_summary.items():
    print(f"  {name}:")
    print(f"    Gate 1 (validity):  {'PASS' if g['gate1']['pass'] else 'FAIL'}  "
          f"aug_med={g['gate1']['aug_median']:.4f}  guest_ref={g['baseline_guest']:.4f}  "
          f"gap={g['gate1']['guest_gap_abs']:.4f}")
    print(f"    Gate 3 (saturation): {'PASS' if g['gate3']['pass'] else 'FAIL'}  "
          f"swing={g['gate3']['band_median_swing']:.4f}")

print(f"\nWrote: {OUT_DIR / 'gates_summary.json'}")
print(f"Wrote: {OUT_DIR / 'gates_per_run.csv'}")
