"""Job A — Score Slot A v2 step3500 + step1500 on the Roy_D / Guest face crops
from analysis/teams_account_natural_experiment_2026-05-19/crops/.

Extends the 2026-05-19 natural-experiment table with the rank-2 ckpt that was
missing from it. The prior table showed T5C flips Roy_D vs Guest at τ=0.70
despite same person / same camera / same moment — different Teams account
only. Question: does Slot A v2's anchor_aware mechanism reach this
transport-encoded shortcut, or is it bounded to the dor identity pool?

Reuses the existing locally-cached ckpts under
analysis/manual_canary_2026-05-20/ckpts/ — no GCS download needed.

Output:
  outputs/job_a_inference_results.json
  outputs/job_a_inference_results.csv
  outputs/job_a_combined_table.csv  (merges prior 2026-05-19 rows + new rows)
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
sys.path.insert(0, str(REPO))

from arena.model_arena import CLIP_MEAN, CLIP_STD, _download_checkpoint, load_model  # noqa

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CROPS_DIR = REPO / "analysis/teams_account_natural_experiment_2026-05-19/crops"
LOCAL_CKPT_CACHE_2026_05_19 = REPO / "analysis/slot_b_property_shortcut_2026-05-16/ckpt_cache"
LOCAL_CKPT_CACHE_2026_05_20 = REPO / "analysis/manual_canary_2026-05-20/ckpts"

DETECTOR_CONFIG = REPO / "config/detector/effort.yaml"
TRAIN_CONFIG = REPO / "config/defaults.yaml"
DEVICE = torch.device("cpu")
RES = 224

# Ckpts. We re-score T5C as a control to confirm the harness matches the
# 2026-05-19 numbers within numerical noise; new rows are SLOT_A_V2 (step3500
# + step1500). hp35c51p is the Slot A v2 W&B run id; the periodic step3500
# auc0.9952 ckpt is the rank-2 candidate from the auto-mode 2026-05-16
# scorecard and from today's 29-suite validation.
CKPTS = [
    {
        "name": "T5C_STEP3500_CONTROL",
        "note": "rerun to confirm harness matches 2026-05-19 numbers (T5C Roy_D=0.795, Guest=0.628)",
        "local_path": LOCAL_CKPT_CACHE_2026_05_20 / "periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth",
        "gcs_uri": "gs://training-job-outputs/best_checkpoints/jrlldtem/periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth",
    },
    {
        "name": "SLOT_A_V2_STEP3500",
        "note": "anchor_aware FT on T5C base, rank-2 from auto-mode 2026-05-16 (hp35c51p)",
        "local_path": LOCAL_CKPT_CACHE_2026_05_20 / "periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth",
        "gcs_uri": "gs://training-job-outputs/best_checkpoints/hp35c51p/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth",
    },
]

CROPS = [
    {"label": "Roy_D", "path": CROPS_DIR / "face_roy_d.png"},
    {"label": "Guest", "path": CROPS_DIR / "face_guest.png"},
]


def preprocess(raw_bgr_path: Path) -> torch.Tensor:
    img_bgr = cv2.imread(str(raw_bgr_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(raw_bgr_path)
    img_bgr = cv2.resize(img_bgr, (RES, RES), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])
    return transform(img_rgb).unsqueeze(0)


def resolve_ckpt(c: dict) -> Path:
    if c["local_path"] is not None and c["local_path"].exists():
        print(f"  using local cache: {c['local_path'].name} ({c['local_path'].stat().st_size/1e9:.2f} GB)")
        return c["local_path"]
    print(f"  downloading: {c['gcs_uri']}")
    return Path(_download_checkpoint(c["gcs_uri"]))


def score_ckpt(c: dict, crops: list[dict]) -> dict:
    print(f"\n=== {c['name']} ({c['note']}) ===")
    t0 = time.time()
    local_ckpt = resolve_ckpt(c)
    print(f"  loading model ...")
    model = load_model(str(local_ckpt), str(DETECTOR_CONFIG), str(TRAIN_CONFIG), DEVICE)
    model.eval()
    print(f"  model loaded in {time.time()-t0:.1f}s")
    results = {}
    for crop in crops:
        x = preprocess(crop["path"]).to(DEVICE)
        with torch.inference_mode():
            out = model({"image": x}, inference=True)
        prob_fake = float(out["prob"].detach().cpu().numpy().reshape(-1)[0])
        cls = (
            out["cls"].detach().cpu().numpy().reshape(-1).tolist()
            if "cls" in out
            else None
        )
        results[crop["label"]] = {"prob_fake": prob_fake, "cls": cls}
        print(f"    {crop['label']:>8}  prob_fake = {prob_fake:.6f}")
    p_roy = results["Roy_D"]["prob_fake"]
    p_guest = results["Guest"]["prob_fake"]
    delta = p_guest - p_roy
    print(f"  Δ(Guest - Roy_D) = {delta:+.6f}")
    flips = {}
    for tau in [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90]:
        roy_flag = "FAKE" if p_roy >= tau else "real"
        guest_flag = "FAKE" if p_guest >= tau else "real"
        flipped = (p_roy >= tau) != (p_guest >= tau)
        flips[f"tau_{tau:.2f}"] = {"roy": roy_flag, "guest": guest_flag, "flipped": flipped}
        sym = "⚠ FLIP" if flipped else "same"
        print(f"    @ τ={tau:.2f}: Roy_D={roy_flag:<4}  Guest={guest_flag:<4}  {sym}")
    results["_meta"] = {
        "delta_guest_minus_roy": delta,
        "flips_by_tau": flips,
        "elapsed_seconds": time.time() - t0,
    }
    del model
    return results


def main():
    all_results = {}
    for c in CKPTS:
        try:
            all_results[c["name"]] = score_ckpt(c, CROPS)
        except Exception as e:
            print(f"  FAILED on {c['name']}: {e}")
            import traceback
            traceback.print_exc()
            all_results[c["name"]] = {"error": str(e)}

    with open(OUT_DIR / "job_a_inference_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(OUT_DIR / "job_a_inference_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ckpt", "label", "prob_fake", "delta_guest_minus_roy", "flips_at_tau_06_07_08"])
        for name, r in all_results.items():
            if "error" in r:
                w.writerow([name, "ERROR", "", "", r["error"]])
                continue
            for lab in ["Roy_D", "Guest"]:
                w.writerow([name, lab, f"{r[lab]['prob_fake']:.6f}", "", ""])
            delta = r["_meta"]["delta_guest_minus_roy"]
            flips_06_07_08 = ",".join(
                str(r["_meta"]["flips_by_tau"][f"tau_{t:.2f}"]["flipped"])
                for t in [0.60, 0.70, 0.80]
            )
            w.writerow([name, "DELTA", "", f"{delta:+.6f}", flips_06_07_08])

    print("\n=== Final summary (Job A) ===")
    print(json.dumps(all_results, indent=2))

    # Combined table with 2026-05-19 priors
    prior_2026_05_19 = REPO / "analysis/teams_account_natural_experiment_2026-05-19/outputs/inference_results.csv"
    if prior_2026_05_19.exists():
        with open(OUT_DIR / "job_a_combined_table.csv", "w", newline="") as fout:
            w = csv.writer(fout)
            w.writerow(["source", "ckpt", "label", "prob_fake", "delta_guest_minus_roy"])
            with open(prior_2026_05_19) as fin:
                reader = csv.reader(fin)
                header = next(reader)
                for row in reader:
                    if len(row) < 4:
                        continue
                    name, lab, prob, delta = row[0], row[1], row[2], row[3] if len(row) > 3 else ""
                    w.writerow(["2026-05-19_prior", name, lab, prob, delta])
            for name, r in all_results.items():
                if "error" in r:
                    continue
                for lab in ["Roy_D", "Guest"]:
                    w.writerow(["2026-05-20_job_a", name, lab, f"{r[lab]['prob_fake']:.6f}", ""])
                w.writerow(["2026-05-20_job_a", name, "DELTA", "", f"{r['_meta']['delta_guest_minus_roy']:+.6f}"])


if __name__ == "__main__":
    main()
