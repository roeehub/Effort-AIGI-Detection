"""Morning verdict: score the 3 trained step3500 ckpts on natural-experiment + may6.

Three FT-from-base ckpts to compare against Slot A v2 step3500 reference:
  Slot 1 (anchor=ON  + codec=0.40 on Slot A v2): gs://.../ohcx4w0x/...step3500...
  Slot 2 (anchor=ON  + codec=0.20 on Slot A v2): gs://.../xg35bae4/...step3500...
  Slot 3 (anchor=OFF + codec=0.40 on T5C):       gs://.../21alo5iu/...step3500...

Reference values (Slot A v2 step3500 + T5C step3500):
  Natural-experiment Roy_D / Guest / Δ:
    Slot A v2 = 0.7668 / 0.5987 / -0.1681
    T5C       = 0.7952 / 0.6275 / -0.1677
  may6 frac>0.5:
    Slot A v2 = 0.043
    T5C       = 0.174

Verdict decision tree (per MORNING_DECISION_TREE):
  Branch A (ship Slot 1)        : Slot 1 reduces nat-exp Δ ≤0.10 AND lockbox-shaped sanity holds
  Branch B (codec binding)      : Slot 3 ≈ Slot 1 on natural-exp AND Slot 3 better than Slot 1
  Branch C (compression trap)   : Slot 1 score drift bad on Roy_D OR may6 frac>0.5 regresses
  Branch D (codec doesn't work) : All 3 fail to reduce nat-exp Δ
  Branch E (light dose wins)    : Slot 2 better than Slot 1

Outputs:
  outputs/verdict_natural_experiment.json   — per-ckpt Roy_D/Guest scores + Δ
  outputs/verdict_may6.json                 — per-ckpt may6 cohort statistics
  outputs/VERDICT_FACTS_2026-05-21.md       — written conclusion
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

OUT_DIR = REPO / "analysis/codec_restoration_gates_2026-05-21" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR = REPO / "analysis/teams_account_natural_experiment_2026-05-19/crops"
MAY6_DIR = REPO / "analysis/xinhe_cross_camera_audit_2026-05-06/raw/may6"
LOCAL_CKPT_CACHE = REPO / "analysis/codec_restoration_gates_2026-05-21" / "ckpt_cache"
LOCAL_CKPT_CACHE.mkdir(parents=True, exist_ok=True)
DETECTOR_CONFIG = REPO / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO / "config/defaults.yaml"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")
RES = 224

CKPTS = [
    {
        "name": "SLOT_1_ANCHOR_CODEC040_STEP3500",
        "gcs_uri": "gs://training-job-outputs/best_checkpoints/ohcx4w0x/periodic_effort_20260521_step3500_auc0.9940_eer0.0267.pth",
        "anchor": True, "codec_p": 0.40, "ft_base": "Slot A v2 step3500",
    },
    {
        "name": "SLOT_2_ANCHOR_CODEC020_STEP3500",
        "gcs_uri": "gs://training-job-outputs/best_checkpoints/xg35bae4/periodic_effort_20260521_step3500_auc0.9951_eer0.0062.pth",
        "anchor": True, "codec_p": 0.20, "ft_base": "Slot A v2 step3500",
    },
    {
        "name": "SLOT_3_NOANCHOR_CODEC040_STEP3500",
        "gcs_uri": "gs://training-job-outputs/best_checkpoints/21alo5iu/periodic_effort_20260521_step3500_auc0.9960_eer0.0165.pth",
        "anchor": False, "codec_p": 0.40, "ft_base": "T5C step3500",
    },
]

REFERENCE = {
    "SLOT_A_V2_STEP3500_REF": {"Roy_D": 0.7668, "Guest": 0.5987, "delta": -0.1681, "may6_frac_gt05": 0.043},
    "T5C_STEP3500_REF":       {"Roy_D": 0.7952, "Guest": 0.6275, "delta": -0.1677, "may6_frac_gt05": 0.174},
}

ROY_D_PATH = CROPS_DIR / "face_roy_d.png"
GUEST_PATH = CROPS_DIR / "face_guest.png"
MAY6_PATHS = sorted(MAY6_DIR.glob("*.png"))
print(f"may6 frames: {len(MAY6_PATHS)}")


def load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (RES, RES), interpolation=cv2.INTER_LINEAR)


def bgr_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])
    return transform(img_rgb).unsqueeze(0)


def score(model, img_bgr: np.ndarray) -> float:
    x = bgr_to_tensor(img_bgr).to(DEVICE)
    with torch.inference_mode():
        out = model({"image": x}, inference=True)
    return float(out["prob"].detach().cpu().numpy().reshape(-1)[0])


def cache_ckpt(gcs_uri: str) -> Path:
    fname = gcs_uri.rsplit("/", 1)[-1]
    local = LOCAL_CKPT_CACHE / fname
    if not local.exists():
        print(f"  downloading {gcs_uri} → {local} ...")
        import subprocess
        subprocess.run(["gsutil", "cp", gcs_uri, str(local)], check=True)
    else:
        print(f"  cache hit: {local}")
    return local


# ─── Main ─────────────────────────────────────────────────────────────
print("=" * 70)
print(f"Codec Restoration VERDICT — {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
print("=" * 70)

roy_d_bgr = load_bgr(ROY_D_PATH)
guest_bgr = load_bgr(GUEST_PATH)
may6_bgrs = [load_bgr(p) for p in MAY6_PATHS]
print(f"Loaded: Roy_D + Guest + {len(may6_bgrs)} may6 frames")

natexp_results = {}
may6_results = {}

for c in CKPTS:
    name = c["name"]
    print(f"\n=== {name} (anchor={c['anchor']}, codec_p={c['codec_p']}, FT from {c['ft_base']}) ===")
    t0 = time.time()
    local = cache_ckpt(c["gcs_uri"])
    model = load_model(str(local), str(DETECTOR_CONFIG), str(TRAIN_CONFIG), DEVICE)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    # Natural experiment
    p_roy = score(model, roy_d_bgr)
    p_guest = score(model, guest_bgr)
    delta = p_guest - p_roy
    print(f"  Natural exp: Roy_D={p_roy:.4f} Guest={p_guest:.4f} Δ={delta:+.4f}")
    natexp_results[name] = {
        "Roy_D": p_roy, "Guest": p_guest, "delta": delta,
        "anchor": c["anchor"], "codec_p": c["codec_p"], "ft_base": c["ft_base"],
    }

    # may6 cohort
    print(f"  Scoring may6 cohort ({len(may6_bgrs)} frames) ...")
    may6_scores = [score(model, img) for img in may6_bgrs]
    arr = np.array(may6_scores)
    may6_results[name] = {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "frac_above_0.50": float((arr > 0.50).mean()),
        "frac_above_0.70": float((arr > 0.70).mean()),
        "frac_above_0.85": float((arr > 0.85).mean()),
        "n_frames": len(may6_bgrs),
    }
    print(f"  may6: mean={arr.mean():.4f} p95={np.percentile(arr, 95):.4f} frac>0.5={(arr > 0.50).mean():.3f} frac>0.85={(arr > 0.85).mean():.3f}")
    del model

# Persist
with open(OUT_DIR / "verdict_natural_experiment.json", "w") as f:
    json.dump({"trained": natexp_results, "reference": REFERENCE}, f, indent=2)
with open(OUT_DIR / "verdict_may6.json", "w") as f:
    json.dump({"trained": may6_results, "reference": REFERENCE}, f, indent=2)

# Print verdict-ready summary
print("\n" + "=" * 70)
print("VERDICT SUMMARY")
print("=" * 70)
print(f"\n{'ckpt':<40s} {'Roy_D':>8s} {'Guest':>8s} {'Δ':>8s} {'may6 frac>0.5':>13s}")
for name in ["SLOT_A_V2_STEP3500_REF", "T5C_STEP3500_REF"]:
    r = REFERENCE[name]
    print(f"{name:<40s} {r['Roy_D']:>8.4f} {r['Guest']:>8.4f} {r['delta']:>+8.4f} {r['may6_frac_gt05']:>13.3f} (ref)")
for name, n in natexp_results.items():
    m = may6_results[name]
    print(f"{name:<40s} {n['Roy_D']:>8.4f} {n['Guest']:>8.4f} {n['delta']:>+8.4f} {m['frac_above_0.50']:>13.3f}")

# Decision-tree branch
print("\n" + "=" * 70)
print("DECISION-TREE BRANCH ASSESSMENT")
print("=" * 70)
slot_a_v2_delta = -0.1681
for name, n in natexp_results.items():
    delta_close = abs(n['delta']) - 0.10
    direction = "CLOSED" if abs(n['delta']) <= 0.05 else ("PARTIAL" if abs(n['delta']) <= 0.10 else "NOT CLOSED")
    delta_red_pct = (1 - abs(n['delta']) / abs(slot_a_v2_delta)) * 100 if slot_a_v2_delta != 0 else 0
    print(f"\n{name}:")
    print(f"  Natural-exp Δ = {n['delta']:+.4f} (vs reference {slot_a_v2_delta:+.4f}; reduced by {delta_red_pct:+.1f}%)")
    print(f"  Gap closure: {direction} (criterion: |Δ| ≤ 0.05)")
    m = may6_results[name]
    may6_ok = m['frac_above_0.50'] <= 0.10
    print(f"  may6 frac>0.5 = {m['frac_above_0.50']:.3f} ({'OK' if may6_ok else 'REGRESSED'}; criterion: ≤ 0.10)")

print(f"\nWrote: {OUT_DIR / 'verdict_natural_experiment.json'}")
print(f"Wrote: {OUT_DIR / 'verdict_may6.json'}")
