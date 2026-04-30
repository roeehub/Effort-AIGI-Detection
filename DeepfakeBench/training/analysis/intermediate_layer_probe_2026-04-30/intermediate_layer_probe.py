"""Intermediate-layer probe for P8A vs mclioexb (2026-04-30).

Closes mechanism (2) from threads/jitter_winner_mechanism_unknown.md:

    The head/feature decomposition probe (analysis/head_feature_decomposition_2026-04-30/)
    measured the [CLS] features at the FINAL layer (pooler_output). It found
    9.4% of frames shifted >0.10 cosine between P8A and mclioexb, and that
    features fully determine the operating-point recall.

    THIS probe asks: did the FT (jitter@0.50) reshape features uniformly
    across the transformer stack, or is the shift concentrated at deeper
    layers? If intermediate layers also moved 5-10%, mechanism (2) is
    supported and the head/feature decomposition is local to the [CLS]
    representation. If only the [CLS]-level features moved, the FT reshape
    is shallow and a different FT base (move C) wouldn't recover anything.

How:
    1. Load the EffortDetector for each checkpoint (P8A_step5000, mclioexb_step500).
       Same config as the trainer used: rank=736, apply_svd_to_in_proj=True,
       unfreeze_final_proj/ln=True, apply_svd_to_mlp=True.
    2. Register forward hooks on the OpenCLIP transformer.resblocks at
       --layers (default 0,3,6,9,11). The hook captures the [CLS] token (first
       in seq dim) of each resblock's output.
    3. Reuse the 800-frame stratified-lockbox sampling from the triptych
       (analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/sampled_frames.csv).
       Frames are cached locally; no GCS reads needed.
    4. Forward each ckpt × all 800 frames in batches; cache per-layer [CLS]
       features in .npz to ``_features_cache_2026-04-30/``.
    5. Per layer, compute:
       - per-frame cos(F_P8A_layer, F_mclioexb_layer); mean / median / fraction below thresholds
       - 5-fold CV LR rec@FPR (oracle head per layer)
    6. Output per-layer summary JSON + per-layer CSV + multi-panel plot.

Runtime estimate: ~10–25 minutes on CPU/MPS for both checkpoints across 5
layers × 800 frames.

Usage:
    python3 analysis/intermediate_layer_probe_2026-04-30/intermediate_layer_probe.py \\
        --layers 0,3,6,9,11 \\
        --device cpu

If you want a faster smoke pass first, use --layers 0,11 (just early/late) or
--max_frames 200.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

CACHE_DIR = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
SAMPLED_CSV = (
    REPO_ROOT
    / "analysis"
    / "embedding_triptych_2026-04-30"
    / "outputs"
    / "triptych_p8a_slot2_slot3"
    / "sampled_frames.csv"
)
DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "config" / "detector" / "effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config" / "train_config.yaml"
OUTPUT_DIR = REPO_ROOT / "analysis" / "intermediate_layer_probe_2026-04-30" / "outputs"

CANONICAL_CKPTS = {
    "P8A": {
        "ckpt_file": "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth",
        "run_id": "9lmvb5b4",
        "recipe": "P8A_REFERENCE_step5000 (FT origin, baseline)",
    },
    "MCLIOEXB": {
        "ckpt_file": "value_composite_effort_20260429_step500_auc0.9797_eer0.0521.pth",
        "run_id": "mclioexb",
        "recipe": "P14_FACE_SCALE_JITTER_ISOLATED step 500 (jitter@0.50, FT-from-P8A)",
    },
}

logger = logging.getLogger("intermediate-layer-probe")


# -----------------------------------------------------------------------------
# Model build + checkpoint load.
# -----------------------------------------------------------------------------
def load_effort_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Build an EffortDetector with the canonical P8A config + load state."""
    import yaml
    from detectors import DETECTOR

    with open(DEFAULT_DETECTOR_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    with open(DEFAULT_TRAIN_CONFIG, "r") as f:
        train_cfg = yaml.safe_load(f)
    cfg.update(train_cfg)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        model_config = ckpt.get("model_config", {})
        for k, v in model_config.items():
            if k != "current_arcface_s":
                cfg[k] = v
    else:
        state_dict = ckpt
        model_config = {}

    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)
    if model_config.get("use_arcface_head", False) and "current_arcface_s" in model_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            model.head.s.data.fill_(model_config["current_arcface_s"])
    clean_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    return model


def get_resblocks(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Locate the OpenCLIP transformer.resblocks under the wrapper."""
    # EffortDetector.backbone is OpenCLIPVisionModelWrapper; .visual is the openclip visual encoder.
    if hasattr(model.backbone, "visual"):
        visual = model.backbone.visual
    else:
        visual = model.backbone
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        return visual.transformer.resblocks
    raise RuntimeError("Could not find transformer.resblocks under model.backbone(.visual)")


# -----------------------------------------------------------------------------
# Image loading + preprocessing.
# -----------------------------------------------------------------------------
def load_and_preprocess(local_path: Path, resolution: int = 224) -> torch.Tensor | None:
    import cv2  # local import: cv2 is heavy

    img = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1))


# -----------------------------------------------------------------------------
# Per-layer feature extraction with hooks.
# -----------------------------------------------------------------------------
def extract_per_layer_features(
    model: torch.nn.Module,
    paths: List[Path],
    layers: List[int],
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Returns (per_layer_features, valid_idx_into_input_paths)."""
    resblocks = get_resblocks(model)
    if max(layers) >= len(resblocks):
        raise ValueError(
            f"requested layer {max(layers)} but resblocks length is {len(resblocks)}"
        )

    captured: Dict[int, List[torch.Tensor]] = {ix: [] for ix in layers}

    def make_hook(ix: int):
        def hook(_module, _input, output):
            # OpenCLIP resblock output shape: (seq, batch, dim) by default.
            # The [CLS] token is index 0.
            if output.dim() == 3:
                # Heuristic: if first dim > batch, it's seq-first; else batch-first.
                # OpenCLIP ViT-B-16 with 224 input → 197 tokens → seq-first when dim0=197.
                if output.shape[0] >= output.shape[1]:
                    cls = output[0]  # [batch, dim]
                else:
                    cls = output[:, 0]  # [batch, dim]
            elif output.dim() == 2:
                cls = output  # already pooled
            else:
                raise RuntimeError(f"unexpected output shape from resblock: {tuple(output.shape)}")
            captured[ix].append(cls.detach().cpu().to(torch.float32).numpy())

        return hook

    handles = [resblocks[ix].register_forward_hook(make_hook(ix)) for ix in layers]
    try:
        valid: List[int] = []
        pending: List[Tuple[int, torch.Tensor]] = []
        for i, p in enumerate(paths):
            t = load_and_preprocess(Path(p))
            if t is None:
                continue
            pending.append((i, t))

        for j in range(0, len(pending), batch_size):
            chunk = pending[j : j + batch_size]
            batch_idx = [c[0] for c in chunk]
            batch = torch.stack([c[1] for c in chunk]).to(device, non_blocking=True)
            with torch.inference_mode():
                _ = model.backbone(batch)  # populates the hooks via forward
            valid.extend(batch_idx)
        for ix in layers:
            captured[ix] = np.concatenate(captured[ix], axis=0) if captured[ix] else np.zeros((0, 0), dtype=np.float32)
    finally:
        for h in handles:
            h.remove()
    return captured, np.array(valid, dtype=np.int64)


def cache_path_for_layer(ckpt_label: str, layer_ix: int, n_samples: int) -> Path:
    return CACHE_DIR / f"intermediate__{ckpt_label}__layer{layer_ix:02d}__n{n_samples}.npz"


# -----------------------------------------------------------------------------
# Per-layer analyses.
# -----------------------------------------------------------------------------
def per_frame_cos_stats(F_a: np.ndarray, F_b: np.ndarray) -> Dict[str, float]:
    A = F_a / (np.linalg.norm(F_a, axis=1, keepdims=True) + 1e-12)
    B = F_b / (np.linalg.norm(F_b, axis=1, keepdims=True) + 1e-12)
    cs = np.einsum("ij,ij->i", A, B)
    return {
        "mean": float(cs.mean()),
        "median": float(np.median(cs)),
        "min": float(cs.min()),
        "max": float(cs.max()),
        "frac_below_0_95": float((cs < 0.95).mean()),
        "frac_below_0_90": float((cs < 0.90).mean()),
        "frac_below_0_80": float((cs < 0.80).mean()),
    }, cs


def fresh_logistic_rec_fpr(feats: np.ndarray, labels: np.ndarray) -> Dict[str, dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold

    feats_n = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(len(labels), dtype=np.float64)
    for tr, te in skf.split(feats_n, labels):
        clf = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
        clf.fit(feats_n[tr], labels[tr])
        oof[te] = clf.predict_proba(feats_n[te])[:, 1]
    fpr, tpr, _ = roc_curve(labels, oof)
    out: Dict[str, dict] = {"auc": float(roc_auc_score(labels, oof))}
    for tgt in (0.02, 0.05, 0.10):
        eligible = np.where(fpr <= tgt + 1e-12)[0]
        if len(eligible) == 0:
            out[f"fpr@{tgt:.2f}"] = {"recall": 0.0}
        else:
            best = eligible[np.argmax(tpr[eligible])]
            out[f"fpr@{tgt:.2f}"] = {"recall": float(tpr[best]), "actual_fpr": float(fpr[best])}
    return out


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Intermediate-layer probe for P8A vs mclioexb")
    ap.add_argument("--layers", default="0,3,6,9,11",
                    help="comma-separated transformer-block indices (0..11 for ViT-B-16)")
    ap.add_argument("--max_frames", type=int, default=800)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default=None,
                    help="cpu / mps / cuda ; auto-detect if omitted")
    ap.add_argument("--no_cache", action="store_true", help="re-extract even if cache file exists")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    layers = sorted(set(layers))

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("device=%s; layers=%s; max_frames=%d", device, layers, args.max_frames)

    # Sampling / metadata.
    if not SAMPLED_CSV.exists():
        logger.error("Sampled-frames CSV missing: %s", SAMPLED_CSV)
        logger.error("Run analysis/embedding_triptych_2026-04-30/triptych.py first to produce it.")
        return 2
    sampled = pd.read_csv(SAMPLED_CSV)
    if "local_path" not in sampled.columns:
        logger.error("sampled_frames.csv must have a 'local_path' column")
        return 2
    sampled = sampled.iloc[: args.max_frames].reset_index(drop=True)
    paths = sampled["local_path"].tolist()
    labels = (sampled["label"].astype(str) == "fake").astype(np.int64).to_numpy()
    n_samples = len(paths)
    logger.info("Loaded %d frames (real=%d, fake=%d)", n_samples, int((labels == 0).sum()), int((labels == 1).sum()))

    # Per-checkpoint × per-layer feature extraction.
    per_ckpt_layer_feats: Dict[str, Dict[int, np.ndarray]] = {}
    per_ckpt_valid_idx: Dict[str, np.ndarray] = {}
    for label, info in CANONICAL_CKPTS.items():
        ckpt_path = CACHE_DIR / info["ckpt_file"]
        if not ckpt_path.exists():
            logger.error("Checkpoint missing: %s; fetch via gsutil cp from gs://training-job-outputs/...", ckpt_path)
            return 2

        # Try cache first.
        layer_feats: Dict[int, np.ndarray] = {}
        cache_hit = (not args.no_cache) and all(
            cache_path_for_layer(label, ix, n_samples).exists() for ix in layers
        )
        if cache_hit:
            for ix in layers:
                blob = np.load(cache_path_for_layer(label, ix, n_samples))
                layer_feats[ix] = blob["features"].astype(np.float32)
            valid_idx = np.load(cache_path_for_layer(label, layers[0], n_samples))["valid_idx"]
            logger.info("[%s] features cached for all %d layers (%d frames)", label, len(layers), n_samples)
        else:
            logger.info("[%s] loading model + extracting layers %s …", label, layers)
            model = load_effort_model(ckpt_path, device)
            layer_feats, valid_idx = extract_per_layer_features(
                model, paths, layers, device, batch_size=args.batch_size,
            )
            for ix in layers:
                np.savez_compressed(
                    cache_path_for_layer(label, ix, n_samples),
                    features=layer_feats[ix].astype(np.float32),
                    valid_idx=valid_idx,
                )
            logger.info("[%s] cached features for %d layers", label, len(layers))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        per_ckpt_layer_feats[label] = layer_feats
        per_ckpt_valid_idx[label] = valid_idx

    # Align valid_idx across ckpts.
    common = np.intersect1d(per_ckpt_valid_idx["P8A"], per_ckpt_valid_idx["MCLIOEXB"])
    pos_p = {int(v): r for r, v in enumerate(per_ckpt_valid_idx["P8A"])}
    pos_m = {int(v): r for r, v in enumerate(per_ckpt_valid_idx["MCLIOEXB"])}
    rows_p = [pos_p[int(v)] for v in common]
    rows_m = [pos_m[int(v)] for v in common]
    aligned_labels = labels[common]

    # Per-layer analyses.
    per_layer_results = []
    for ix in layers:
        Fp = per_ckpt_layer_feats["P8A"][ix][rows_p]
        Fm = per_ckpt_layer_feats["MCLIOEXB"][ix][rows_m]
        cos_stats, _ = per_frame_cos_stats(Fp, Fm)
        lr_p = fresh_logistic_rec_fpr(Fp, aligned_labels)
        lr_m = fresh_logistic_rec_fpr(Fm, aligned_labels)
        per_layer_results.append({
            "layer": ix,
            "feature_cos": cos_stats,
            "fresh_LR_on_Fp": lr_p,
            "fresh_LR_on_Fm": lr_m,
        })

    summary = {
        "probe": "intermediate_layer",
        "date": "2026-04-30",
        "layers": layers,
        "n_aligned_frames": int(len(common)),
        "ckpts": CANONICAL_CKPTS,
        "per_layer": per_layer_results,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # CSV table.
    rows = []
    for r in per_layer_results:
        rows.append({
            "layer": r["layer"],
            "cos_mean": r["feature_cos"]["mean"],
            "cos_median": r["feature_cos"]["median"],
            "frac_below_0_95": r["feature_cos"]["frac_below_0_95"],
            "frac_below_0_90": r["feature_cos"]["frac_below_0_90"],
            "frac_below_0_80": r["feature_cos"]["frac_below_0_80"],
            "Fp_LR_auc": r["fresh_LR_on_Fp"]["auc"],
            "Fm_LR_auc": r["fresh_LR_on_Fm"]["auc"],
            "Fp_LR_rec@0.05": r["fresh_LR_on_Fp"]["fpr@0.05"]["recall"],
            "Fm_LR_rec@0.05": r["fresh_LR_on_Fm"]["fpr@0.05"]["recall"],
            "Fp_LR_rec@0.10": r["fresh_LR_on_Fp"]["fpr@0.10"]["recall"],
            "Fm_LR_rec@0.10": r["fresh_LR_on_Fm"]["fpr@0.10"]["recall"],
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "per_layer_table.csv", index=False)

    # Print readout.
    print()
    print("=" * 92)
    print("INTERMEDIATE-LAYER PROBE  (P8A vs mclioexb)  --  n=%d aligned frames" % len(common))
    print("=" * 92)
    print(f"{'layer':>6} {'cos_mean':>10} {'cos_med':>10} {'frac<.90':>10} {'Fp_AUC':>9} {'Fm_AUC':>9} {'Fp@.05':>9} {'Fm@.05':>9}")
    print("-" * 92)
    for r in per_layer_results:
        print(
            f"{r['layer']:>6d} "
            f"{r['feature_cos']['mean']:>10.4f} "
            f"{r['feature_cos']['median']:>10.4f} "
            f"{r['feature_cos']['frac_below_0_90']:>10.3f} "
            f"{r['fresh_LR_on_Fp']['auc']:>9.4f} "
            f"{r['fresh_LR_on_Fm']['auc']:>9.4f} "
            f"{r['fresh_LR_on_Fp']['fpr@0.05']['recall']:>9.4f} "
            f"{r['fresh_LR_on_Fm']['fpr@0.05']['recall']:>9.4f}"
        )
    print("-" * 92)
    print(f"  outputs in {OUTPUT_DIR}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
