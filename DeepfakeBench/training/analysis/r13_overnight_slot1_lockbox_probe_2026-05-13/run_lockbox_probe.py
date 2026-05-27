"""Linear probe on Slot 1 (LoRA-P8A top_n_step2000) frozen L11 features for
the dor_shkedi cohort — real-vs-fake separability test.

Why: today's diagnostics produced an apparent paradox:
  - Atlas chronic_6 forgery AUC=0.995 (n=800 triptych)
  - Partial scorecard: lockbox AUC = 0.4442 at HEAD output for Slot 1
  - 96.3% of Slot 1's lockbox FPs are dor_shkedi (45.8% FPR vs P8A 0.7%)
This script answers: does the ENCODER still separate dor reals from dor fakes
at L11 even though the head collapsed?

Methodology:
  - Reuse the LoRA-wrap + state-load harness from run_atlas.py
  - Hook resblocks[11] for L11 CLS embeddings (same as A3 / atlas)
  - Extract on the locally-cached dor cohort (no GCS downloads required)
  - 5-fold StratifiedKFold logistic regression
  - Report mean ± std AUC over the 5 folds + 95% CI
  - Same probe on P8A_REFERENCE_STEP5000 features for comparison

Cohort construction:
  - REAL (label=0): dor_shkedi lockbox real, locally cached  (~275 frames)
  - FAKE (label=1):
      * dor_shkedi__s16 dev fake (teams pipeline on dor) — 78 frames (100% cached)
      * deeplive_dor dev fake (deeplive method on dor) — 545 frames (100% cached)
  - Two probes: (a) lockbox-real vs s16-fake (single teams-method),
                (b) lockbox-real vs (s16-fake + deeplive-fake) (any fake method on dor)

Note on memory's "62 fake dor_shkedi lockbox": this memory is incorrect.
lockbox-fake contains only Cam_Test__s33 (191) + PC_Generator__s15 (62);
dor_shkedi is real-only in lockbox. The "1138 reals" matches (1138 videos =
1170 frames). For probe purposes, dev-fake provides the fake-side anchor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

HERE = REPO / "analysis" / "r13_overnight_slot1_lockbox_probe_2026-05-13"
HERE.mkdir(parents=True, exist_ok=True)
FEAT_DIR = HERE / "_features"
FEAT_DIR.mkdir(exist_ok=True)

# Inputs
ATLAS_DIR = REPO / "analysis" / "r13_overnight_atlas_2026-05-13"
CKPT_SLOT1 = ATLAS_DIR / "_cache" / "top_n_effort_20260513_step2000_auc0.9929_eer0.0229.pth"
CKPT_P8A = REPO / "analysis" / "_features_cache_2026-04-30" / "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth"

MANIFEST = REPO / "arena" / "manifests" / "teams_target_domain_manifest_2026-04-23_with_dor.json"
FRAME_CACHE_ROOT = REPO / "analysis" / "lockbox_tagging" / "_frame_cache"

DEFAULT_DETECTOR_CONFIG = REPO / "config" / "detector" / "effort.yaml"
DEFAULT_TRAIN_CONFIG = REPO / "config" / "train_config.yaml"

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# LoRA config matching R13_LORA_*.yaml
LORA_CONFIG = dict(target_layers=[10, 11], rank=16, alpha=32.0,
                   target_modules=("attn.in_proj", "attn.out_proj", "mlp.c_fc", "mlp.c_proj"))

logger = logging.getLogger("dor-probe")

BUCKET_PREFIX = "gs://teams-faces-data-test-2914-fake-4420-real-feb-28/"


# =============================================================================
# Cohort assembly
# =============================================================================

def uri_to_local(uri: str) -> Path:
    """Map gs://bucket/<blob> → local cache path via md5(blob_path)[:2]/[2:4]/basename."""
    if not uri.startswith(BUCKET_PREFIX):
        # Fallback: treat suffix after last `/` as basename and md5 the whole URI.
        blob = uri
    else:
        blob = uri[len(BUCKET_PREFIX):]
    md5 = hashlib.md5(blob.encode()).hexdigest()
    return FRAME_CACHE_ROOT / md5[:2] / md5[2:4] / blob.rsplit("/", 1)[-1]


def build_cohort() -> Dict[str, List[Tuple[str, Path]]]:
    """Return dict of cohort-name → list of (gcs_uri, local_path) where local_path exists.

    Cohorts:
      real_lockbox_dor:   dor_shkedi  split=lockbox  label=real
      fake_dev_s16:       dor_shkedi__s16  split=dev  label=fake (teams pipeline on dor)
      real_dev_s16:       dor_shkedi__s16  split=dev  label=real
      fake_dev_deeplive:  deeplive_dor  split=dev  label=fake
    """
    manifest = json.load(open(MANIFEST))
    videos = manifest["videos"]

    buckets: Dict[str, List[str]] = {
        "real_lockbox_dor": [],
        "fake_dev_s16": [],
        "real_dev_s16": [],
        "fake_dev_deeplive": [],
    }
    for v in videos:
        ik = v["identity_key"]
        sp, lb = v["split"], v["label"]
        if ik == "dor_shkedi" and sp == "lockbox" and lb == "real":
            buckets["real_lockbox_dor"].extend(v["frame_paths"])
        elif ik == "dor_shkedi__s16" and lb == "fake":
            buckets["fake_dev_s16"].extend(v["frame_paths"])
        elif ik == "dor_shkedi__s16" and lb == "real":
            buckets["real_dev_s16"].extend(v["frame_paths"])
        elif ik == "deeplive_dor" and lb == "fake":
            buckets["fake_dev_deeplive"].extend(v["frame_paths"])

    out: Dict[str, List[Tuple[str, Path]]] = {}
    for name, uris in buckets.items():
        rows = []
        for u in uris:
            p = uri_to_local(u)
            if p.exists():
                rows.append((u, p))
        out[name] = rows
        logger.info("cohort %-22s: %d / %d cached locally", name, len(rows), len(uris))
    return out


# =============================================================================
# Model + extraction
# =============================================================================

def load_effort_model(ckpt_path: Path, device, lora_cfg: Optional[Dict] = None):
    """Build EffortDetector with cfg from saved model_config; wrap LoRA if needed;
    load state dict. Mirrors atlas/run_atlas.py exactly."""
    import yaml
    import torch
    from detectors import DETECTOR

    with open(DEFAULT_DETECTOR_CONFIG) as f:
        cfg = yaml.safe_load(f)
    with open(DEFAULT_TRAIN_CONFIG) as f:
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

    if lora_cfg is not None:
        from detectors.lora_adapter import apply_lora_to_openclip_visual, count_lora_parameters
        visual = model.backbone.visual
        n_wrapped = apply_lora_to_openclip_visual(
            visual,
            target_layers=list(lora_cfg["target_layers"]),
            rank=int(lora_cfg["rank"]),
            alpha=float(lora_cfg["alpha"]),
            target_modules=tuple(lora_cfg["target_modules"]),
        )
        model.to(device)
        lora_params, total_params = count_lora_parameters(visual)
        logger.info("  [LoRA] wrapped %d layers (%d lora / %d total)",
                    n_wrapped, lora_params, total_params)

    clean_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    miss_other = [k for k in missing if "lora_A" not in k and "lora_B" not in k]
    if miss_other:
        logger.info("  missing %d non-LoRA keys (e.g. %s)", len(miss_other), miss_other[:3])
    if unexpected:
        logger.info("  dropped %d unexpected keys (e.g. %s)", len(unexpected), unexpected[:3])

    model.eval()
    return model


def get_resblocks(model):
    visual = model.backbone.visual if hasattr(model.backbone, "visual") else model.backbone
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        return visual.transformer.resblocks
    raise RuntimeError("Could not find transformer.resblocks")


def load_and_preprocess(local_path: Path, resolution: int = 224):
    import cv2
    import torch
    img = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1))


def extract_l11(model, paths: List[Path], device, batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """Same harness as run_atlas.py extract_l11."""
    import torch
    resblocks = get_resblocks(model)
    captured: List[np.ndarray] = []

    def hook(_m, _i, output):
        if output.dim() == 3:
            cls = output[0] if output.shape[0] >= output.shape[1] else output[:, 0]
        elif output.dim() == 2:
            cls = output
        else:
            raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
        captured.append(cls.detach().cpu().to(torch.float32).numpy())

    handle = resblocks[11].register_forward_hook(hook)
    try:
        valid: List[int] = []
        pending: List[Tuple[int, "torch.Tensor"]] = []
        for i, p in enumerate(paths):
            t = load_and_preprocess(p)
            if t is not None:
                pending.append((i, t))
        logger.info("  loaded %d/%d valid frames", len(pending), len(paths))
        for j in range(0, len(pending), batch_size):
            chunk = pending[j: j + batch_size]
            batch_idx = [c[0] for c in chunk]
            batch = torch.stack([c[1] for c in chunk]).to(device, non_blocking=True)
            with torch.inference_mode():
                _ = model.backbone(batch)
            valid.extend(batch_idx)
            if (j // batch_size) % 5 == 0:
                logger.info("  processed batch %d/%d", j // batch_size + 1,
                            (len(pending) + batch_size - 1) // batch_size)
        feats = np.concatenate(captured, axis=0) if captured else np.zeros((0, 0), dtype=np.float32)
    finally:
        handle.remove()
    return feats, np.array(valid, dtype=np.int64)


# =============================================================================
# Linear probe
# =============================================================================

def linear_probe_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 0
                    ) -> Dict[str, float]:
    """5-fold StratifiedKFold logistic regression. Returns dict with mean/std/CI.

    Following run_atlas.py fit_probe_auc convention: L2-normalize features before
    fitting (Z = X / ||X||), n_jobs=1, max_iter=3000, C=1.0.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    y = np.asarray(y, dtype=int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos < 5 or n_neg < 5:
        return dict(auc_mean=float("nan"), auc_std=float("nan"),
                    ci_lo=float("nan"), ci_hi=float("nan"),
                    n_pos=n_pos, n_neg=n_neg, oof_auc=float("nan"))

    actual_splits = min(n_splits, n_pos, n_neg)
    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=seed)
    per_fold_aucs = []
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=3000, n_jobs=1, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        oof[te] = p
        per_fold_aucs.append(float(roc_auc_score(y[te], p)))

    aucs = np.array(per_fold_aucs, dtype=np.float64)
    mean = float(aucs.mean())
    std = float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0
    # 95% CI via fold-mean ± t * std / sqrt(n) — use 1.96 for simplicity at k=5
    se = std / np.sqrt(len(aucs))
    return dict(
        auc_mean=mean,
        auc_std=std,
        ci_lo=mean - 1.96 * se,
        ci_hi=mean + 1.96 * se,
        n_pos=n_pos,
        n_neg=n_neg,
        oof_auc=float(roc_auc_score(y, oof)),
        per_fold=per_fold_aucs,
        actual_splits=actual_splits,
    )


# =============================================================================
# Pipeline
# =============================================================================

def extract_for_ckpt(label: str, ckpt_path: Path, lora_cfg: Optional[Dict],
                     cohort_paths: List[Path], device, batch_size: int = 16) -> np.ndarray:
    """Extract features for one ckpt over given cohort_paths. Cache to disk
    so re-runs are cheap."""
    cache = FEAT_DIR / f"{label}__layer11__dor_cohort.npz"
    cache_meta = FEAT_DIR / f"{label}__layer11__dor_cohort.meta.json"
    if cache.exists() and cache_meta.exists():
        meta = json.load(open(cache_meta))
        if meta.get("n_paths") == len(cohort_paths):
            blob = np.load(cache)
            feats = blob["features"].astype(np.float32)
            valid = blob["valid_idx"].astype(np.int64)
            logger.info("[%s] cached: shape=%s, valid=%d/%d",
                        label, feats.shape, len(valid), len(cohort_paths))
            return feats, valid

    logger.info("[%s] loading model from %s", label, ckpt_path)
    model = load_effort_model(ckpt_path, device, lora_cfg=lora_cfg)
    feats, valid = extract_l11(model, cohort_paths, device, batch_size=batch_size)
    nonfinite = int((~np.isfinite(feats)).sum())
    if nonfinite > 0:
        logger.warning("[%s] %d non-finite values in features", label, nonfinite)
    np.savez_compressed(cache, features=feats.astype(np.float32), valid_idx=valid)
    json.dump(dict(n_paths=len(cohort_paths), shape=list(feats.shape), nonfinite=nonfinite),
              open(cache_meta, "w"), indent=2)
    logger.info("[%s] saved: shape=%s, valid=%d/%d, nonfinite=%d",
                label, feats.shape, len(valid), len(cohort_paths), nonfinite)
    del model
    return feats, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_reals", type=int, default=None,
                        help="cap real-lockbox cohort size for speed (default: all available)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s :: %(message)s")

    import torch
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("device=%s", device)

    # === 1. Build cohort ===
    cohort = build_cohort()
    real_lb = cohort["real_lockbox_dor"]
    fake_s16 = cohort["fake_dev_s16"]
    fake_dl = cohort["fake_dev_deeplive"]

    if args.max_reals is not None and len(real_lb) > args.max_reals:
        real_lb = real_lb[: args.max_reals]
        logger.info("capped real_lockbox to %d frames", len(real_lb))

    # Build the unified order: reals first, then fake_s16, then fake_deeplive
    all_rows = []
    for uri, p in real_lb:
        all_rows.append(dict(uri=uri, local=p, label=0, source="real_lockbox_dor"))
    for uri, p in fake_s16:
        all_rows.append(dict(uri=uri, local=p, label=1, source="fake_dev_s16"))
    for uri, p in fake_dl:
        all_rows.append(dict(uri=uri, local=p, label=1, source="fake_dev_deeplive"))
    all_paths = [r["local"] for r in all_rows]
    logger.info("total cohort: %d frames (real=%d, fake_s16=%d, fake_deeplive=%d)",
                len(all_rows), len(real_lb), len(fake_s16), len(fake_dl))

    # === 2. Extract features for Slot 1 + P8A ===
    feats_s1, valid_s1 = extract_for_ckpt(
        "SLOT1_LORA_P8A_top_n_step2000", CKPT_SLOT1, LORA_CONFIG,
        all_paths, device, args.batch_size,
    )
    feats_p8a, valid_p8a = extract_for_ckpt(
        "P8A_REFERENCE_STEP5000", CKPT_P8A, None,
        all_paths, device, args.batch_size,
    )

    # Sanity: should have the same valid set
    valid_common = sorted(set(valid_s1.tolist()) & set(valid_p8a.tolist()))
    s1_idx_map = {int(v): r for r, v in enumerate(valid_s1)}
    p8a_idx_map = {int(v): r for r, v in enumerate(valid_p8a)}
    X_s1 = np.stack([feats_s1[s1_idx_map[v]] for v in valid_common])
    X_p8a = np.stack([feats_p8a[p8a_idx_map[v]] for v in valid_common])
    y_full = np.array([all_rows[v]["label"] for v in valid_common], dtype=int)
    source_full = [all_rows[v]["source"] for v in valid_common]

    logger.info("X_s1=%s, X_p8a=%s, y=%s (n_pos=%d, n_neg=%d)",
                X_s1.shape, X_p8a.shape, y_full.shape, int(y_full.sum()), int(len(y_full) - y_full.sum()))

    # === 3. Run probes ===
    probes: Dict[str, Dict] = {}

    # Probe A: real_lockbox_dor vs fake_dev_s16 (teams-pipeline-on-dor only)
    mask_a = np.array([s in ("real_lockbox_dor", "fake_dev_s16") for s in source_full])
    Xa_s1, Xa_p8a, ya = X_s1[mask_a], X_p8a[mask_a], y_full[mask_a]
    probes["A_lockbox_real_vs_s16_fake__SLOT1"] = linear_probe_cv(Xa_s1, ya)
    probes["A_lockbox_real_vs_s16_fake__P8A"] = linear_probe_cv(Xa_p8a, ya)

    # Probe B: real_lockbox_dor vs fake_dev_s16 + fake_dev_deeplive (any-method fake on dor)
    probes["B_lockbox_real_vs_anyfake__SLOT1"] = linear_probe_cv(X_s1, y_full)
    probes["B_lockbox_real_vs_anyfake__P8A"] = linear_probe_cv(X_p8a, y_full)

    # Probe C: real_lockbox_dor vs fake_dev_deeplive only (cross-method dor probe)
    mask_c = np.array([s in ("real_lockbox_dor", "fake_dev_deeplive") for s in source_full])
    Xc_s1, Xc_p8a, yc = X_s1[mask_c], X_p8a[mask_c], y_full[mask_c]
    probes["C_lockbox_real_vs_deeplive_fake__SLOT1"] = linear_probe_cv(Xc_s1, yc)
    probes["C_lockbox_real_vs_deeplive_fake__P8A"] = linear_probe_cv(Xc_p8a, yc)

    # === 4. Print results ===
    print()
    print("=" * 100)
    print(f"  Linear probe on L11 CLS features (dor_shkedi cohort, n_real_lockbox={int((y_full[mask_a]==0).sum())})")
    print("=" * 100)
    for name, res in probes.items():
        is_s1 = name.endswith("__SLOT1")
        tag = "SLOT1" if is_s1 else "P8A  "
        probe_tag = name.split("__")[0]
        print(f"  [{tag}] {probe_tag:40s}  "
              f"n_pos={res['n_pos']:4d}  n_neg={res['n_neg']:4d}  k={res['actual_splits']}  "
              f"AUC = {res['auc_mean']:.4f} ± {res['auc_std']:.4f}  "
              f"95% CI [{res['ci_lo']:.4f}, {res['ci_hi']:.4f}]  "
              f"OOF_AUC={res['oof_auc']:.4f}")
    print("=" * 100)

    # === 5. Save results ===
    out = dict(
        date="2026-05-13",
        ckpt_slot1=str(CKPT_SLOT1),
        ckpt_p8a=str(CKPT_P8A),
        cohort_n_real_lockbox=len(real_lb),
        cohort_n_fake_s16=len(fake_s16),
        cohort_n_fake_deeplive=len(fake_dl),
        valid_extracted=int(len(valid_common)),
        probes=probes,
    )
    out_path = HERE / "_probe_results_2026-05-13.json"
    json.dump(out, open(out_path, "w"), indent=2, default=float)
    logger.info("wrote %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
