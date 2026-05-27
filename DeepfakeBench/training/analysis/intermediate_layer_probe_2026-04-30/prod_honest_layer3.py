"""Phase B: evaluate the layer-3 head on the 180-frame production-honest substrate.

The 180-frame production-honest substrate
(`analysis/deployment_honest_eval_2026-04-27/_prod_cache/frames/`) is the
true deployment-relevant set. Frames are organized by 6 source groups:

  dor-real-laptop-correct-no-virtual-bg-whiteish     (LOW risk: P8A 0% fake)
  dor-real-laptop-correct-no-virtual-bg-yellowish    (LOW: P8A 10%)
  dor-real-webcam-false-flag                          (HIGH: P8A 80% fake)
  dor-real-webcam-false-flag-no-virtual-bg            (HIGH: P8A 80%)
  roee-mac-laptop-false-flag-virtual-bg               (HIGHEST: P8A 90%)
  roee-real-windows-laptop-correct                    (LOW: P8A 0%)

All 180 frames are REAL. The current model's high-fake-rate on the
high-risk groups is the deployment block.

This script:
  1. Extracts P8A layer-3 [CLS] features for all 180 frames.
  2. Loads the cached layer-3 dev features (4000 stratified frames from
     scaled_layer3_validation).
  3. Trains an LR head on dev features, evaluates on the 180 frames.
  4. Reports per-source-group fake rates at the same operating point.

If the layer-3 head reduces the high-risk-group fake rate from ~90% to
something operationally tolerable (say <30%), the GPU experiment is a
strong recommendation.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

CACHE_DIR = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
PROD_FRAMES_DIR = REPO_ROOT / "analysis" / "deployment_honest_eval_2026-04-27" / "_prod_cache" / "frames"
PROD_TAGS_JSON = REPO_ROOT / "analysis" / "deployment_honest_eval_2026-04-27" / "_prod_cache" / "combined_frame_tags.json"
DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "config" / "detector" / "effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config" / "train_config.yaml"
OUTPUT_DIR = REPO_ROOT / "analysis" / "intermediate_layer_probe_2026-04-30" / "outputs"

P8A_CKPT = "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth"


def load_p8a(device: torch.device) -> torch.nn.Module:
    import yaml
    from detectors import DETECTOR
    with open(DEFAULT_DETECTOR_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    with open(DEFAULT_TRAIN_CONFIG, "r") as f:
        cfg.update(yaml.safe_load(f))
    ckpt = torch.load(str(CACHE_DIR / P8A_CKPT), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        for k, v in ckpt.get("model_config", {}).items():
            if k != "current_arcface_s":
                cfg[k] = v
    else:
        state_dict = ckpt
    model = DETECTOR[cfg["model_name"]](cfg).to(device).eval()
    clean = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    model.load_state_dict(clean, strict=False)
    return model


def get_resblocks(model):
    visual = model.backbone.visual if hasattr(model.backbone, "visual") else model.backbone
    return visual.transformer.resblocks


def load_and_preprocess(path: Path, resolution=224):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array(CLIP_MEAN, np.float32)) / np.array(CLIP_STD, np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1))


def extract_layer_features(model, paths, layer_ix, device, batch_size=32):
    resblocks = get_resblocks(model)
    captured = []

    def hook(_m, _inp, output):
        if output.dim() == 3:
            cls = output[0] if output.shape[0] >= output.shape[1] else output[:, 0]
        else:
            cls = output
        captured.append(cls.detach().cpu().to(torch.float32).numpy())

    handle = resblocks[layer_ix].register_forward_hook(hook)
    try:
        valid = np.zeros(len(paths), dtype=bool)
        chunks = []
        idxs = []
        for i, p in enumerate(paths):
            t = load_and_preprocess(Path(p))
            if t is None:
                continue
            chunks.append(t)
            idxs.append(i)
            if len(chunks) >= batch_size:
                batch = torch.stack(chunks).to(device, non_blocking=True)
                with torch.inference_mode():
                    _ = model.backbone(batch)
                for j in idxs:
                    valid[j] = True
                chunks, idxs = [], []
        if chunks:
            batch = torch.stack(chunks).to(device, non_blocking=True)
            with torch.inference_mode():
                _ = model.backbone(batch)
            for j in idxs:
                valid[j] = True
        feats = np.concatenate(captured, axis=0) if captured else np.zeros((0, 0), np.float32)
    finally:
        handle.remove()
    return feats, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layers", default="3,6,11")
    args = ap.parse_args()
    device = torch.device(args.device)
    layers = [int(x) for x in args.layers.split(",")]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover the 180 frames + their source groups.
    rows = []
    for source_group_dir in sorted(PROD_FRAMES_DIR.iterdir()):
        if not source_group_dir.is_dir():
            continue
        for frame in sorted(source_group_dir.iterdir()):
            if frame.suffix.lower() in (".jpg", ".png", ".jpeg"):
                rows.append({"source_group": source_group_dir.name, "path": str(frame)})
    df = pd.DataFrame(rows)
    print(f"Discovered {len(df)} prod-honest frames across {df['source_group'].nunique()} groups")
    print(df.groupby("source_group").size().to_string())

    # Extract features.
    model = load_p8a(device)
    paths = df["path"].tolist()
    per_layer_feats = {}
    for layer in layers:
        cache_path = CACHE_DIR / f"prod_honest_180__P8A__layer{layer:02d}.npz"
        if cache_path.exists():
            blob = np.load(cache_path)
            per_layer_feats[layer] = blob["features"]
            print(f"[layer {layer}] cached n={len(blob['features'])}")
        else:
            print(f"[layer {layer}] extracting {len(paths)} frames...", flush=True)
            feats, valid = extract_layer_features(model, paths, layer, device)
            np.savez_compressed(cache_path, features=feats, valid=valid)
            per_layer_feats[layer] = feats
            print(f"[layer {layer}] done; cached → {cache_path}")
    df = df[:len(per_layer_feats[layers[0]])].reset_index(drop=True)

    # Load cached dev features for training the head.
    cached_paths = sorted(CACHE_DIR.glob("layer_validation__P8A__*.npz"))
    if not cached_paths:
        print("ERROR: no scaled validation cache found; run scaled_layer3_validation.py first")
        return 1
    valid_cache = cached_paths[-1]
    print(f"Using cache: {valid_cache.name}")
    cache = np.load(valid_cache)
    dev_labels = cache["dev_labels"]
    print(f"Dev train set: n={len(dev_labels)} ({(dev_labels==0).sum()} real, {(dev_labels==1).sum()} fake)")

    # Train head per layer; predict on prod-honest 180.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve

    results = {}
    for layer in layers:
        if f"dev_layer_{layer}_feats" not in cache.files:
            print(f"layer {layer} dev features missing in cache — skipping")
            continue
        F_dev = cache[f"dev_layer_{layer}_feats"]
        F_prod = per_layer_feats[layer]
        F_dev_n = F_dev / (np.linalg.norm(F_dev, axis=1, keepdims=True) + 1e-12)
        F_prod_n = F_prod / (np.linalg.norm(F_prod, axis=1, keepdims=True) + 1e-12)

        clf = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
        clf.fit(F_dev_n, dev_labels)
        # Calibrate τ such that on lockbox (also cached) FPR=5% — but we lack
        # lockbox labels here. Instead use τ at FPR=5% on dev OOF.
        # Use lockbox features from the same cache for τ calibration.
        if f"lb_layer_{layer}_feats" in cache.files:
            F_lb = cache[f"lb_layer_{layer}_feats"]
            F_lb_n = F_lb / (np.linalg.norm(F_lb, axis=1, keepdims=True) + 1e-12)
            lb_labels = cache["lb_labels"]
            lb_proba = clf.predict_proba(F_lb_n)[:, 1]
            # Pick τ at FPR = 5% on lockbox
            fpr, tpr, thr = roc_curve(lb_labels, lb_proba)
            target_fpr = 0.05
            eligible = np.where(fpr <= target_fpr + 1e-12)[0]
            if len(eligible):
                idx = eligible[np.argmax(tpr[eligible])]
                tau_5pct = float(thr[idx])
                lb_recall_at_5pct = float(tpr[idx])
                lb_fpr_at_5pct = float(fpr[idx])
            else:
                tau_5pct = 0.5
                lb_recall_at_5pct = 0.0
                lb_fpr_at_5pct = 0.0
            lb_auc = float(roc_auc_score(lb_labels, lb_proba))
        else:
            tau_5pct = 0.5
            lb_recall_at_5pct = lb_fpr_at_5pct = lb_auc = float("nan")

        # Predict prod-honest.
        prod_proba = clf.predict_proba(F_prod_n)[:, 1]
        df[f"layer{layer}_prob_fake"] = prod_proba
        df[f"layer{layer}_pred_at_tau5pct"] = (prod_proba >= tau_5pct).astype(int)

        per_group = df.groupby("source_group").agg(
            n=(f"layer{layer}_pred_at_tau5pct", "size"),
            fake_rate=(f"layer{layer}_pred_at_tau5pct", "mean"),
            mean_prob=(f"layer{layer}_prob_fake", "mean"),
        ).reset_index()

        results[f"layer_{layer}"] = {
            "tau_5pct_lockbox": tau_5pct,
            "lockbox_AUC": lb_auc,
            "lockbox_FPR_at_tau": lb_fpr_at_5pct,
            "lockbox_recall_at_tau": lb_recall_at_5pct,
            "prod_180_per_group": per_group.to_dict(orient="records"),
            "prod_180_overall_fake_rate": float(df[f"layer{layer}_pred_at_tau5pct"].mean()),
        }

        print(f"\n=== Layer {layer} ===")
        print(f"  τ@5%FPR_lb = {tau_5pct:.4f}; lb AUC={lb_auc:.4f} fpr={lb_fpr_at_5pct:.4f} rec={lb_recall_at_5pct:.4f}")
        print(f"  prod_180 overall fake_rate at τ: {df[f'layer{layer}_pred_at_tau5pct'].mean():.3f}")
        print(per_group.to_string(index=False))

    out_json = OUTPUT_DIR / "prod_honest_layer3.json"
    out_csv = OUTPUT_DIR / "prod_honest_layer3.csv"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    df.to_csv(out_csv, index=False)
    print(f"\noutputs → {out_json}, {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
