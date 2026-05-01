"""Extract P18 BACKBONE features (final [CLS]) on the 800-frame eval substrate.

This is the missing piece that the head-direction probe couldn't address: did
GRL flatten the encoder's [CLS] manifold? P8A's frozen features can't tell us;
we need P18's live encoder features.

Method:
1. Load the P18 ckpt's full model state (backbone + head).
2. Run inference on the 800-frame substrate (using the cached `local_path`
   in sampled_frames.csv → no GCS).
3. Extract the final [CLS] features (post-encoder, pre-head).
4. Save as analysis/p18_probe_2026-05-01/outputs/p18_feats__<run_id>__<step>.npz
5. Run the 12-class method-LR macro-OVR AUC on those features (plus same
   axes as probe_p18_ckpt.py).

Usage:
    python3 extract_p18_features.py --ckpt <local_path> --label <id> --arm treatment

Slow-ish: ~5-10 min CPU per ckpt, ~1-2 min on MPS. Worth doing once both
jobs reach final ckpts.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("p18-extract")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CACHE_DIR = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
SAMPLED_CSV = (
    REPO_ROOT / "analysis" / "embedding_triptych_2026-04-30" / "outputs"
    / "triptych_p8a_slot2_slot3" / "sampled_frames.csv"
)
OUTPUT_DIR = REPO_ROOT / "analysis" / "p18_probe_2026-05-01" / "outputs"

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711])


def load_image_clip_normalize(local_path: str, target=(224, 224)) -> np.ndarray:
    """Load image with OpenCV, resize to 224x224 LINEAR, BGR→RGB, CLIP normalize."""
    img = cv2.imread(local_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, target, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - CLIP_MEAN) / CLIP_STD
    img = img.transpose(2, 0, 1).astype(np.float32)  # CHW
    return img


def detect_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_effort_detector_from_ckpt(ckpt_path: Path, device: torch.device):
    """Construct an EffortDetector with the same recipe as the ckpt and load its state."""
    from detectors.effort_detector import EffortDetector

    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ck.get("state_dict") or ck.get("model_state_dict") or ck
    model_config = ck.get("model_config", {})

    # Default to the P8A recipe; override with model_config when present.
    config = {
        "rank": 736,
        "lambda_reg": 0.01,
        "use_arcface_head": True,
        "arcface_s": 6.0,
        "arcface_m": 0.15,
        "s_start": 6.0,
        "s_end": 12.0,
        "anneal_steps": 4000,
        "backbone": {
            "name": "vit_b_16_laion_datacomp",
            "variant": "ViT-B-16-DataComp-XL",
            "source": "laion",
            "model_name": "ViT-B-16",
            "pretrained": "datacomp_xl_s13b_b90k",
            "hidden_size": 512,
            "resolution": 224,
            "apply_svd_to_in_proj": True,
            "unfreeze_final_proj": True,
            "unfreeze_final_ln": True,
            "apply_svd_to_mlp": True,
        },
        "use_quality_domain_head": False,  # Doesn't matter for inference
    }
    config.update(model_config)
    if "backbone" not in config or not isinstance(config["backbone"], dict):
        config["backbone"] = {
            "name": "vit_b_16_laion_datacomp",
            "hidden_size": 512,
            "resolution": 224,
            "apply_svd_to_in_proj": True,
            "unfreeze_final_proj": True,
            "unfreeze_final_ln": True,
            "apply_svd_to_mlp": True,
        }

    logger.info(f"Building EffortDetector with config from ckpt...")
    model = EffortDetector(config=config)
    # Strip "module." prefix if DDP
    new_state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        logger.warning(f"Missing keys (truncated): {missing[:3]}{'...' if len(missing)>3 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys (truncated): {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")
    model.eval()
    model.to(device)
    return model


def extract_cls_features(model, df_valid: pd.DataFrame, device: torch.device, batch_size=16):
    """Run model.backbone on each frame, extract final [CLS] features.

    The EffortDetector's backbone is a CLIP visual encoder; the [CLS] token
    is what `backbone(...)` returns when the model is in inference mode.
    """
    feats = np.zeros((len(df_valid), 512), dtype=np.float32)
    batch_imgs = []
    batch_idx = []

    def flush_batch():
        if not batch_imgs:
            return
        x = torch.from_numpy(np.stack(batch_imgs)).to(device)
        with torch.no_grad():
            out = model.backbone(x)
        # EffortDetector convention: backbone returns dict with 'pooler_output'
        # which is the [CLS] feature post final-LN (per detector.forward
        # at effort_detector.py:847 and 1219).
        if isinstance(out, dict):
            out = out["pooler_output"]
        elif isinstance(out, tuple):
            out = out[0]
        feats[batch_idx] = out.detach().cpu().float().numpy()
        batch_imgs.clear()
        batch_idx.clear()

    for i, row in df_valid.iterrows():
        img = load_image_clip_normalize(row["local_path"])
        if img is None:
            continue
        batch_imgs.append(img)
        batch_idx.append(i)
        if len(batch_imgs) >= batch_size:
            flush_batch()
    flush_batch()
    return feats


def run_method_lr_probe(feats: np.ndarray, df_valid: pd.DataFrame, label: np.ndarray):
    """12-class method-LR macro-OVR AUC analog. Uses StandardScaler matching
    the canonical domain_probe.py."""
    from sklearn.preprocessing import StandardScaler
    from data.sources.method_domain_map import lookup_method_domain_with_label

    method = df_valid["method"].astype(str).to_numpy()

    def to_bucket(m: str, lbl: int) -> int:
        m = (m or "").strip().lower()
        if m == "teams_real":
            return 11
        if m.startswith("teams_capture") or m.startswith("teams_flat"):
            return 3
        if m == "deeplive_enhanced":
            return 2
        if m.startswith("deeplive_"):
            return lookup_method_domain_with_label(m, "deeplive", lbl)
        return lookup_method_domain_with_label(m, "visomaster", lbl)

    domain_label = np.array(
        [to_bucket(method[i], int(label[i])) for i in range(len(method))], dtype=np.int64
    )
    bucket_counts = Counter(domain_label.tolist())
    keep_mask = np.array([bucket_counts[b] >= 5 for b in domain_label])
    f = feats[keep_mask]
    y = domain_label[keep_mask]
    kept_buckets = sorted(set(y.tolist()))

    if len(kept_buckets) < 2:
        return None, dict(bucket_counts)

    scaler = StandardScaler()
    f_norm = scaler.fit_transform(f)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(y), len(kept_buckets)))
    bucket_to_idx = {b: i for i, b in enumerate(kept_buckets)}
    for tr, te in skf.split(f_norm, y):
        clf = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs", multi_class="ovr")
        clf.fit(f_norm[tr], y[tr])
        cls_to_col = {c: i for i, c in enumerate(clf.classes_)}
        proba = clf.predict_proba(f_norm[te])
        for c, col in cls_to_col.items():
            oof_proba[te, bucket_to_idx[c]] = proba[:, col]
    aucs = []
    per_bucket_auc = {}
    for b in kept_buckets:
        i = bucket_to_idx[b]
        y_bin = (y == b).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        auc = float(roc_auc_score(y_bin, oof_proba[:, i]))
        aucs.append(auc)
        per_bucket_auc[str(b)] = auc
    macro_auc = float(np.mean(aucs)) if aucs else None
    return macro_auc, per_bucket_auc, dict(bucket_counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = detect_device()
    logger.info(f"Using device: {device}")

    # Load eval-substrate metadata + filter to valid local_paths
    df = pd.read_csv(SAMPLED_CSV).iloc[:800].reset_index(drop=True)
    df["has_local"] = df["local_path"].apply(lambda p: isinstance(p, str) and Path(p).exists())
    df_valid = df[df["has_local"]].reset_index(drop=True).iloc[:800]
    logger.info(f"Eval substrate: {len(df_valid)} frames with local cache.")
    label = (df_valid["label"].astype(str) == "fake").astype(int).to_numpy()

    # Build model + extract features
    model = build_effort_detector_from_ckpt(Path(args.ckpt), device)
    logger.info(f"Extracting [CLS] features (batch_size={args.batch_size})...")
    feats = extract_cls_features(model, df_valid, device, batch_size=args.batch_size)
    logger.info(f"Features shape: {feats.shape}")

    # Save raw features
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_npz = OUTPUT_DIR / f"p18_feats__{args.label}.npz"
    np.savez_compressed(out_npz, features=feats, valid_idx=df_valid.index.to_numpy())
    logger.info(f"Saved features: {out_npz}")

    # Run 12-class method-LR probe
    logger.info("Running 12-class method-LR macro-OVR AUC probe on P18 features...")
    macro_auc, per_bucket_auc, bucket_counts = run_method_lr_probe(feats, df_valid, label)
    print()
    print("=" * 80)
    print(f"P18 ENCODER-SIDE PROBE — {args.arm.upper()} {args.label}")
    print("=" * 80)
    print(f"  Bucket counts: {bucket_counts}")
    print(f"  Per-bucket AUC: {per_bucket_auc}")
    print(f"  Macro-OVR AUC (P18 features, 12-class): {macro_auc:.4f}" if macro_auc is not None else "  Macro AUC: NA")
    print()
    print(f"  Comparison vs P8A baseline (Phase 3.5 smoke gate): 0.998")
    print(f"  Comparison vs P8A on P8A-extracted (this probe pipeline): 0.912")
    if macro_auc is not None:
        if macro_auc < 0.65:
            verdict = "BITES STRONGLY (encoder substantially flatter)"
        elif macro_auc < 0.85:
            verdict = "BITES (material reduction in domain discriminability)"
        elif macro_auc < 0.95:
            verdict = "PARTIAL bite (some flattening; not decisive)"
        else:
            verdict = "DOES NOT BITE (encoder still discriminates 12-class methods)"
        print(f"  → {verdict}")

    # Save probe result
    out_json = OUTPUT_DIR / f"p18_method_lr_probe__{args.label}.json"
    with open(out_json, "w") as f:
        json.dump({
            "label": args.label,
            "arm": args.arm,
            "macro_ovr_auc": macro_auc,
            "per_bucket_auc": per_bucket_auc,
            "bucket_counts": {str(k): int(v) for k, v in bucket_counts.items()},
            "p8a_baseline_method_lr_auc": 0.912,
            "p8a_smoke_gate_baseline": 0.998,
        }, f, indent=2)
    logger.info(f"Saved probe result: {out_json}")


if __name__ == "__main__":
    main()
