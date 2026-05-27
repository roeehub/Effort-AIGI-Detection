"""Scale validation of the layer-3 dev→lockbox transfer finding.

The 800-sample probe (per_layer_split_probe.py) showed:
  P8A layer 3:  dev_AUC=0.94, lb_AUC=0.97, transfer dev→lb AUC=0.95
  P8A layer 6:  dev_AUC=0.99, lb_AUC=0.997, transfer dev→lb AUC=0.39
  P8A layer 11: dev_AUC=0.98, lb_AUC=0.94, transfer dev→lb AUC=0.66

Lockbox n=87 was small. Scale this up to the full 839-frame lockbox + a
larger dev sample, extracting P8A layer-3 [CLS] features and re-running
the dev→lockbox transfer test.

Outputs:
  outputs/scaled_layer3_validation.json
  outputs/scaled_layer3_features_p8a.npz   (cached features)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

CACHE_DIR = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
PARQUET = REPO_ROOT / "analysis" / "lockbox_tagging" / "full_tags_2026-04-27.parquet"
OUTPUT_DIR = REPO_ROOT / "analysis" / "intermediate_layer_probe_2026-04-30" / "outputs"
DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "config" / "detector" / "effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config" / "train_config.yaml"

P8A_CKPT = "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth"


def load_p8a(device: torch.device) -> torch.nn.Module:
    import yaml
    from detectors import DETECTOR
    with open(DEFAULT_DETECTOR_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    with open(DEFAULT_TRAIN_CONFIG, "r") as f:
        cfg.update(yaml.safe_load(f))
    ckpt_path = CACHE_DIR / P8A_CKPT
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
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


def extract_layer_features(model, paths, layer_ix, device, batch_size=32, log_every=1000):
    """Returns (features [N,D], valid_mask [orig_N])."""
    resblocks = get_resblocks(model)
    captured = []

    def hook(_m, _inp, output):
        if output.dim() == 3:
            cls = output[0] if output.shape[0] >= output.shape[1] else output[:, 0]
        elif output.dim() == 2:
            cls = output
        else:
            raise RuntimeError(f"unexpected output shape {output.shape}")
        captured.append(cls.detach().cpu().to(torch.float32).numpy())

    handle = resblocks[layer_ix].register_forward_hook(hook)
    try:
        valid = np.zeros(len(paths), dtype=bool)
        chunks = []
        chunk_idx = []
        for i, p in enumerate(paths):
            t = load_and_preprocess(Path(p))
            if t is None:
                continue
            chunks.append(t)
            chunk_idx.append(i)

            if len(chunks) >= batch_size:
                batch = torch.stack(chunks).to(device, non_blocking=True)
                with torch.inference_mode():
                    _ = model.backbone(batch)
                for j in chunk_idx:
                    valid[j] = True
                chunks, chunk_idx = [], []
                if (i + 1) % log_every == 0:
                    print(f"  processed {i+1}/{len(paths)} (valid so far: {valid.sum()})", flush=True)
        if chunks:
            batch = torch.stack(chunks).to(device, non_blocking=True)
            with torch.inference_mode():
                _ = model.backbone(batch)
            for j in chunk_idx:
                valid[j] = True

        feats = np.concatenate(captured, axis=0) if captured else np.zeros((0, 0), np.float32)
    finally:
        handle.remove()
    return feats, valid


def transfer_eval(train_feats, train_labels, test_feats, test_labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    A = train_feats / (np.linalg.norm(train_feats, axis=1, keepdims=True) + 1e-12)
    B = test_feats / (np.linalg.norm(test_feats, axis=1, keepdims=True) + 1e-12)
    clf = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
    clf.fit(A, train_labels)
    proba = clf.predict_proba(B)[:, 1]
    fpr, tpr, _ = roc_curve(test_labels, proba)
    auc = float(roc_auc_score(test_labels, proba))
    out = {"auc": auc, "n_train": len(train_labels), "n_test": len(test_labels)}
    for tgt in (0.02, 0.05, 0.10):
        eligible = np.where(fpr <= tgt + 1e-12)[0]
        out[f"rec@fpr_{tgt:.2f}"] = float(tpr[eligible[np.argmax(tpr[eligible])]]) if len(eligible) else 0.0
    return out, proba


def cv_eval(feats, labels, n_splits=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold
    if min(labels.sum(), (1-labels).sum()) < n_splits:
        return None
    A = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    oof = np.zeros(len(labels))
    for tr, te in skf.split(A, labels):
        clf = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
        clf.fit(A[tr], labels[tr])
        oof[te] = clf.predict_proba(A[te])[:, 1]
    fpr, tpr, _ = roc_curve(labels, oof)
    out = {"auc": float(roc_auc_score(labels, oof))}
    for tgt in (0.02, 0.05, 0.10):
        eligible = np.where(fpr <= tgt + 1e-12)[0]
        out[f"rec@fpr_{tgt:.2f}"] = float(tpr[eligible[np.argmax(tpr[eligible])]]) if len(eligible) else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="3", help="comma-separated layer indices")
    ap.add_argument("--n_dev", type=int, default=4000, help="stratified dev sample (50/50 real/fake)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--no_cache", action="store_true")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    device = torch.device(args.device)
    print(f"device={device}, layers={layers}, n_dev={args.n_dev}", flush=True)

    df = pd.read_parquet(PARQUET)
    df = df[df["local_path"].astype(str).str.len() > 0].reset_index(drop=True)
    print(f"parquet loaded: {len(df)} rows; dev={int((df['split']=='dev').sum())} lockbox={int((df['split']=='lockbox').sum())}", flush=True)

    # Stratified dev sample.
    dev_df = df[df["split"] == "dev"].copy()
    dev_real = dev_df[dev_df["label"] == "real"]
    dev_fake = dev_df[dev_df["label"] == "fake"]
    rng = np.random.default_rng(seed=42)
    dev_real_idx = rng.choice(len(dev_real), size=min(args.n_dev // 2, len(dev_real)), replace=False)
    dev_fake_idx = rng.choice(len(dev_fake), size=min(args.n_dev // 2, len(dev_fake)), replace=False)
    dev_sample = pd.concat([
        dev_real.iloc[dev_real_idx],
        dev_fake.iloc[dev_fake_idx],
    ]).reset_index(drop=True)

    lb_df = df[df["split"] == "lockbox"].copy().reset_index(drop=True)
    print(f"dev_sample: n={len(dev_sample)} (real={(dev_sample['label']=='real').sum()}, fake={(dev_sample['label']=='fake').sum()})")
    print(f"lockbox: n={len(lb_df)} (real={(lb_df['label']=='real').sum()}, fake={(lb_df['label']=='fake').sum()})")

    cache_key = f"layer_validation__P8A__{args.n_dev}_{len(lb_df)}__layers_{'_'.join(map(str, layers))}.npz"
    cache_path = CACHE_DIR / cache_key

    if cache_path.exists() and not args.no_cache:
        print(f"loading cached: {cache_path}", flush=True)
        blob = np.load(cache_path, allow_pickle=True)
        per_layer_feats = {int(k.split("_")[2]): blob[k] for k in blob.files
                           if k.startswith("dev_layer_") or k.startswith("lb_layer_")}
        # Reconstruct properly
        per_layer_feats = {}
        for layer in layers:
            per_layer_feats[layer] = {
                "dev": blob[f"dev_layer_{layer}_feats"],
                "lb": blob[f"lb_layer_{layer}_feats"],
            }
        dev_labels = blob["dev_labels"]
        lb_labels = blob["lb_labels"]
    else:
        model = load_p8a(device)

        all_paths_dev = dev_sample["local_path"].tolist()
        all_paths_lb = lb_df["local_path"].tolist()

        per_layer_feats = {}
        cache_dict = {}
        # Extract dev for each layer.
        for layer in layers:
            print(f"\n[dev] extracting P8A layer {layer} for {len(all_paths_dev)} frames...", flush=True)
            feats_dev, valid_dev = extract_layer_features(model, all_paths_dev, layer, device, args.batch_size)
            print(f"[dev] valid: {valid_dev.sum()}/{len(all_paths_dev)}; feature shape: {feats_dev.shape}", flush=True)

            print(f"\n[lb] extracting P8A layer {layer} for {len(all_paths_lb)} frames...", flush=True)
            feats_lb, valid_lb = extract_layer_features(model, all_paths_lb, layer, device, args.batch_size)
            print(f"[lb] valid: {valid_lb.sum()}/{len(all_paths_lb)}; feature shape: {feats_lb.shape}", flush=True)

            # Subset labels to valid only.
            per_layer_feats[layer] = {
                "dev": feats_dev,
                "lb": feats_lb,
                "valid_dev": valid_dev,
                "valid_lb": valid_lb,
            }
            cache_dict[f"dev_layer_{layer}_feats"] = feats_dev
            cache_dict[f"lb_layer_{layer}_feats"] = feats_lb
            cache_dict[f"dev_layer_{layer}_valid"] = valid_dev
            cache_dict[f"lb_layer_{layer}_valid"] = valid_lb

        # All layers should agree on valid mask (same paths). Use the first.
        valid_dev_first = per_layer_feats[layers[0]]["valid_dev"]
        valid_lb_first = per_layer_feats[layers[0]]["valid_lb"]
        dev_labels = (dev_sample["label"].astype(str) == "fake").astype(np.int64).to_numpy()[valid_dev_first]
        lb_labels = (lb_df["label"].astype(str) == "fake").astype(np.int64).to_numpy()[valid_lb_first]
        cache_dict["dev_labels"] = dev_labels
        cache_dict["lb_labels"] = lb_labels
        np.savez_compressed(cache_path, **cache_dict)
        print(f"\ncached features → {cache_path}", flush=True)

    # Per-layer evaluation.
    print()
    print("=" * 92)
    print(f"SCALED LAYER VALIDATION  ({P8A_CKPT})")
    print("=" * 92)
    print(f"{'layer':>6} {'dev_cv_AUC':>12} {'dev@.05':>9} {'lb_cv_AUC':>12} {'lb@.05':>9} {'transfer_AUC':>14} {'tr@.05':>8} {'tr@.10':>8}")
    print("-" * 92)
    rows = []
    for layer in layers:
        feats_dev = per_layer_feats[layer]["dev"]
        feats_lb = per_layer_feats[layer]["lb"]
        within_dev = cv_eval(feats_dev, dev_labels)
        within_lb = cv_eval(feats_lb, lb_labels, n_splits=5)
        transfer, _ = transfer_eval(feats_dev, dev_labels, feats_lb, lb_labels)
        rows.append({
            "layer": layer,
            "n_dev": len(feats_dev), "n_lb": len(feats_lb),
            "within_dev": within_dev, "within_lb": within_lb, "transfer_dev2lb": transfer,
        })
        print(f"{layer:>6d} "
              f"{within_dev['auc']:>12.4f} {within_dev['rec@fpr_0.05']:>9.4f} "
              f"{within_lb['auc']:>12.4f} {within_lb['rec@fpr_0.05']:>9.4f} "
              f"{transfer['auc']:>14.4f} {transfer['rec@fpr_0.05']:>8.4f} {transfer['rec@fpr_0.10']:>8.4f}")
    print("=" * 92)

    out_path = OUTPUT_DIR / "scaled_layer3_validation.json"
    with open(out_path, "w") as f:
        json.dump({"P8A_ckpt": P8A_CKPT, "rows": rows}, f, indent=2)
    print(f"  outputs → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
