"""P18 corrective probes — actually test the load-bearing axis.

The earlier `extract_p18_features.py` measured 3-class inter-bucket macro-OVR
AUC, which is the wrong granularity. The shortcut Phase 1A identified is
intra-bucket-3 (dor_shkedi-style identities vs other Teams identities, both
inside deeplive_teams). This probe runs the right tests:

1. **Phase 1A re-run on P18 L3 features** — re-extract layer-3 [CLS] from each
   arm's encoder, fit fresh-LR fake-vs-real on dev, fit substrate-classifier
   directions, compute cosines between trained head's L3 readout direction
   and substrate axes. Directly comparable to P17 verdict numbers.

   Caveat: P18's trained head consumes [CLS] of FINAL layer, not L3. So we
   can't directly extract a "P18 L3 head direction" — there is no L3 head.
   What we CAN do: train a fresh LR on each arm's L3 features predicting
   fake/real, then check that LR direction's alignment with substrate axes.
   This tells us how much substrate-shortcut signal sits in the L3 features
   themselves under each arm.

2. **Within-bucket-3 LR**: dor_shkedi (39 frames in eval) vs other Teams reals
   (subset of bucket 11/3) on each arm's [CLS] features.

3. **Per-frame score distribution comparison** on dor_shkedi lockbox frames.
   Use the cached prob_fake from the arm's training (W&B per-frame logging
   IF present), or re-score using the deployed model — but we already have
   per-frame predictions from extract_p18_features's [CLS] features +
   head.weight extraction.

CPU only. n_jobs=1.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("p18-corrective")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CACHE_DIR = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
SAMPLED_CSV = (
    REPO_ROOT / "analysis" / "embedding_triptych_2026-04-30" / "outputs"
    / "triptych_p8a_slot2_slot3" / "sampled_frames.csv"
)
OUTPUT_DIR = REPO_ROOT / "analysis" / "p18_probe_2026-05-01" / "outputs"

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711])


def l2norm(x, axis):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def cos(a, b):
    if a is None or b is None:
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return None
    return float(np.dot(a, b) / (na * nb))


def detect_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image_clip_normalize(local_path: str, target=(224, 224)) -> Optional[np.ndarray]:
    img = cv2.imread(local_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, target, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - CLIP_MEAN) / CLIP_STD
    return img.transpose(2, 0, 1).astype(np.float32)


def build_effort_detector_from_ckpt(ckpt_path: Path, device: torch.device):
    """Construct an EffortDetector and load state. Configures with P8A defaults
    if model_config is incomplete in the saved ckpt."""
    from detectors.effort_detector import EffortDetector
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ck.get("state_dict") or ck.get("model_state_dict") or ck
    model_config = ck.get("model_config", {})
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
        "use_quality_domain_head": False,
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
    model = EffortDetector(config=config)
    new_state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(new_state, strict=False)
    model.eval()
    model.to(device)
    return model, ck


def extract_l3_features(model, df_valid: pd.DataFrame, device: torch.device, batch_size=16):
    """Run model.backbone with a forward hook on resblocks[3] to capture L3 [CLS]."""
    captured = {}
    handle = None

    def hook(module, inp, out):
        # CLIP visual: out is (seq_len, batch, dim) in pre-LN-pre-final-proj space
        # OR (batch, seq_len, dim) depending on version. Detect dynamically.
        if isinstance(out, tuple):
            out = out[0]
        # Standard open_clip ViT outputs (batch, seq_len, dim) at a resblock.
        # CLS token is at position 0.
        captured["x"] = out

    # Locate resblocks[3] in the backbone
    block = None
    for name, mod in model.backbone.named_modules():
        if name.endswith("transformer.resblocks.3") or name.endswith("resblocks.3"):
            block = mod
            break
    if block is None:
        raise RuntimeError("Could not find resblocks[3] in backbone")
    handle = block.register_forward_hook(hook)

    feats_l3 = np.zeros((len(df_valid), 768), dtype=np.float32)  # ViT-B-16 hidden=768
    batch_imgs = []
    batch_idx = []

    def flush_batch():
        if not batch_imgs:
            return
        x = torch.from_numpy(np.stack(batch_imgs)).to(device)
        with torch.no_grad():
            _ = model.backbone(x)
        cap = captured["x"]
        # cap is (batch, seq_len, dim) — extract CLS at position 0
        if cap.dim() == 3:
            if cap.shape[0] == len(batch_imgs):
                cls = cap[:, 0, :]
            else:
                # (seq_len, batch, dim)
                cls = cap[0, :, :]
        else:
            raise RuntimeError(f"Unexpected feature shape: {cap.shape}")
        feats_l3[batch_idx] = cls.detach().cpu().float().numpy()
        batch_imgs.clear()
        batch_idx.clear()

    try:
        for i, row in df_valid.iterrows():
            img = load_image_clip_normalize(row["local_path"])
            if img is None:
                continue
            batch_imgs.append(img)
            batch_idx.append(i)
            if len(batch_imgs) >= batch_size:
                flush_batch()
        flush_batch()
    finally:
        if handle is not None:
            handle.remove()
    return feats_l3


def fit_substrate_directions(feats_n: np.ndarray, df_valid: pd.DataFrame, label: np.ndarray):
    """Mirror Phase 1A's substrate-classifier direction probe."""
    capture = df_valid["clip_capture_mode"].astype(str).to_numpy()
    method = df_valid["method"].astype(str).to_numpy()
    identity = df_valid["identity_key"].astype(str).to_numpy()
    is_lockbox = (df_valid["split"].astype(str) == "lockbox").to_numpy()
    is_dev = ~is_lockbox

    axes = {
        "is_lockbox": is_lockbox,
        "is_webcam": capture == "webcam",
        "is_dor_shkedi": np.array([("dor_shkedi" in i) for i in identity]),
        "is_deeplive_enhanced": method == "deeplive_enhanced",
        "is_teams_capture": np.array([("teams_capture" in m) for m in method]),
    }
    directions = {}
    for name, y in axes.items():
        if y.sum() < 5 or (~y).sum() < 5:
            directions[name] = None
            continue
        lr = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
        lr.fit(feats_n, y.astype(int))
        directions[name] = lr.coef_.flatten().astype(np.float64)

    # Fresh-LR fake-vs-real direction (substrate-INVARIANT reference)
    lr_fr = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
    lr_fr.fit(feats_n[is_dev], label[is_dev])
    directions["fresh_LR_fake_vs_real"] = lr_fr.coef_.flatten().astype(np.float64)

    # ALSO compute lockbox transfer AUC of fresh-LR (Phase 1A baseline = 0.9495 on P8A L3)
    fresh_lb_scores = lr_fr.predict_proba(feats_n[is_lockbox])[:, 1]
    fresh_lb_auc = roc_auc_score(label[is_lockbox], fresh_lb_scores) if len(np.unique(label[is_lockbox])) >= 2 else None

    return directions, fresh_lb_auc


def within_bucket_3_dor_lr(feats_n: np.ndarray, df_valid: pd.DataFrame):
    """Within bucket-3 (deeplive_teams = teams_capture content): can a linear
    classifier distinguish dor_shkedi-identity frames from other identities?

    This tests the actual shortcut axis Phase 1A identified.
    """
    method = df_valid["method"].astype(str).to_numpy()
    identity = df_valid["identity_key"].astype(str).to_numpy()
    label = (df_valid["label"].astype(str) == "fake").astype(int).to_numpy()

    # Bucket-3 mask: teams_capture content (Phase 2C convention)
    bucket3 = np.array([("teams_capture" in m or "teams_flat" in m) for m in method])
    is_dor = np.array([("dor_shkedi" in i) for i in identity])
    in_bucket3 = bucket3
    n_dor = int((in_bucket3 & is_dor).sum())
    n_other = int((in_bucket3 & ~is_dor).sum())

    if n_dor < 5 or n_other < 5:
        return None, {"n_dor": n_dor, "n_other": n_other, "warning": "insufficient"}

    f_sub = feats_n[in_bucket3]
    y_sub = is_dor[in_bucket3].astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y_sub))
    for tr, te in skf.split(f_sub, y_sub):
        clf = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
        clf.fit(f_sub[tr], y_sub[tr])
        oof[te] = clf.predict_proba(f_sub[te])[:, 1]
    auc = float(roc_auc_score(y_sub, oof))
    return auc, {"n_dor": n_dor, "n_other": n_other, "label_real_vs_fake": Counter(label[in_bucket3].tolist())}


def score_with_head(feats: np.ndarray, ckpt: dict) -> np.ndarray:
    """Apply the trained head to extract per-frame margins (fake - real)."""
    state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    head_w = None
    for key in ("head.weight", "module.head.weight"):
        if key in state:
            head_w = state[key].cpu().numpy().astype(np.float64)
            break
    if head_w is None:
        for k, v in state.items():
            if "head" in k and k.endswith(".weight") and v.ndim == 2 and v.shape[0] == 2:
                head_w = v.cpu().numpy().astype(np.float64)
                break
    if head_w is None:
        raise RuntimeError("no head.weight in ckpt")
    config = ckpt.get("model_config", {})
    use_arc = bool(config.get("use_arcface_head", True))
    if use_arc:
        feats_n = l2norm(feats.astype(np.float64), axis=1)
        w_n = l2norm(head_w, axis=1)
        logits = feats_n @ w_n.T
    else:
        logits = feats.astype(np.float64) @ head_w.T
        hb = state.get("head.bias")
        if hb is not None:
            logits = logits + hb.cpu().numpy()[None, :]
    margin = logits[:, 1] - logits[:, 0]
    return margin


def per_frame_dor_score_distribution(margins: np.ndarray, df_valid: pd.DataFrame):
    """Compare prob_fake distribution on dor_shkedi vs other Teams reals (lockbox)."""
    identity = df_valid["identity_key"].astype(str).to_numpy()
    method = df_valid["method"].astype(str).to_numpy()
    label = (df_valid["label"].astype(str) == "fake").astype(int).to_numpy()
    is_lockbox = (df_valid["split"].astype(str) == "lockbox").to_numpy()
    is_dor = np.array([("dor_shkedi" in i) for i in identity])
    is_real = label == 0
    is_teams_real = np.array([m == "teams_real" for m in method])

    # Sigmoid-like normalization
    prob_fake = 1.0 / (1.0 + np.exp(-margins))

    # Lockbox reals: dor_shkedi vs other identities
    lb_real_dor = is_lockbox & is_real & is_dor
    lb_real_nondor = is_lockbox & is_real & ~is_dor

    summary = {
        "lockbox_real_dor_n": int(lb_real_dor.sum()),
        "lockbox_real_dor_mean_prob_fake": float(np.mean(prob_fake[lb_real_dor])) if lb_real_dor.sum() > 0 else None,
        "lockbox_real_dor_median": float(np.median(prob_fake[lb_real_dor])) if lb_real_dor.sum() > 0 else None,
        "lockbox_real_nondor_n": int(lb_real_nondor.sum()),
        "lockbox_real_nondor_mean_prob_fake": float(np.mean(prob_fake[lb_real_nondor])) if lb_real_nondor.sum() > 0 else None,
        "lockbox_real_nondor_median": float(np.median(prob_fake[lb_real_nondor])) if lb_real_nondor.sum() > 0 else None,
    }
    # FPR at τ=0.5, 0.92, 0.974 on each subset
    for name, mask in [("dor", lb_real_dor), ("nondor", lb_real_nondor)]:
        if mask.sum() > 0:
            for tau in [0.5, 0.92, 0.9741]:
                fp = float((prob_fake[mask] >= tau).sum())
                summary[f"lockbox_real_{name}_fpr_at_tau_{tau}"] = fp / int(mask.sum())
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()

    device = detect_device()
    logger.info(f"Using device: {device}")

    df = pd.read_csv(SAMPLED_CSV).iloc[:800].reset_index(drop=True)
    df["has_local"] = df["local_path"].apply(lambda p: isinstance(p, str) and Path(p).exists())
    df_valid = df[df["has_local"]].reset_index(drop=True).iloc[:800]
    logger.info(f"Eval substrate: {len(df_valid)} frames with local cache.")
    label = (df_valid["label"].astype(str) == "fake").astype(int).to_numpy()

    # Build model + extract L3 features
    model, ck = build_effort_detector_from_ckpt(Path(args.ckpt), device)
    logger.info("Extracting L3 [CLS] features (forward hook on resblocks[3])...")
    feats_l3 = extract_l3_features(model, df_valid, device, batch_size=16)
    logger.info(f"L3 features: {feats_l3.shape}")

    # ALSO get final [CLS] for within-bucket probe
    logger.info("Extracting final [CLS] features...")
    feats_cls = np.zeros((len(df_valid), 512), dtype=np.float32)
    batch_imgs, batch_idx = [], []

    def flush_cls():
        if not batch_imgs:
            return
        x = torch.from_numpy(np.stack(batch_imgs)).to(device)
        with torch.no_grad():
            out = model.backbone(x)
        if isinstance(out, dict):
            out = out["pooler_output"]
        feats_cls[batch_idx] = out.detach().cpu().float().numpy()
        batch_imgs.clear()
        batch_idx.clear()
    for i, row in df_valid.iterrows():
        img = load_image_clip_normalize(row["local_path"])
        if img is None:
            continue
        batch_imgs.append(img)
        batch_idx.append(i)
        if len(batch_imgs) >= 16:
            flush_cls()
    flush_cls()

    # === Probe 1: Phase 1A re-run on P18 L3 features ===
    logger.info("\n=== Probe 1: Phase 1A re-run on P18 L3 features ===")
    feats_l3_n = l2norm(feats_l3.astype(np.float64), axis=1)
    substrate_dirs, fresh_lb_auc = fit_substrate_directions(feats_l3_n, df_valid, label)
    print()
    print(f"  Fresh-LR L3 dev→lockbox transfer AUC: {fresh_lb_auc:.4f} (P8A baseline = 0.9495)")

    # Compare fresh-LR direction vs each substrate axis
    fresh_dir = substrate_dirs.get("fresh_LR_fake_vs_real")
    print(f"  Fresh-LR direction's alignment with substrate axes (in P18 L3 normalized space):")
    cosines_fresh = {}
    for name in ["is_lockbox", "is_webcam", "is_dor_shkedi", "is_deeplive_enhanced", "is_teams_capture"]:
        c = cos(fresh_dir, substrate_dirs.get(name))
        cosines_fresh[name] = c
        if c is not None:
            print(f"    cos(fresh_LR, {name:<25}) = {c:+.4f}")

    # === Probe 2: Within-bucket-3 dor_shkedi-vs-non-dor LR on FINAL [CLS] ===
    logger.info("\n=== Probe 2: Within-bucket-3 dor_shkedi-vs-non-dor LR (final [CLS]) ===")
    feats_cls_n = StandardScaler().fit_transform(feats_cls)
    bucket3_auc, bucket3_meta = within_bucket_3_dor_lr(feats_cls_n, df_valid)
    if bucket3_auc is not None:
        print(f"  Within-bucket-3 dor-vs-other LR macro-OVR AUC = {bucket3_auc:.4f}")
        print(f"    n_dor={bucket3_meta['n_dor']}, n_other={bucket3_meta['n_other']}")
    else:
        print(f"  SKIPPED — {bucket3_meta}")

    # === Probe 2b: Same on L3 features ===
    bucket3_l3_auc, bucket3_l3_meta = within_bucket_3_dor_lr(feats_l3_n, df_valid)
    if bucket3_l3_auc is not None:
        print(f"  Within-bucket-3 dor-vs-other LR (L3 features) AUC = {bucket3_l3_auc:.4f}")

    # === Probe 3: Per-frame score distribution ===
    logger.info("\n=== Probe 3: Per-frame score distribution on dor lockbox ===")
    margins = score_with_head(feats_cls, ck)
    score_summary = per_frame_dor_score_distribution(margins, df_valid)
    print(f"  Lockbox real dor_shkedi (n={score_summary['lockbox_real_dor_n']}):")
    if score_summary["lockbox_real_dor_n"] > 0:
        print(f"    mean prob_fake = {score_summary['lockbox_real_dor_mean_prob_fake']:.4f}")
        print(f"    median        = {score_summary['lockbox_real_dor_median']:.4f}")
        print(f"    FPR @ τ=0.50  = {score_summary.get('lockbox_real_dor_fpr_at_tau_0.5', 0.0):.4f}")
        print(f"    FPR @ τ=0.92  = {score_summary.get('lockbox_real_dor_fpr_at_tau_0.92', 0.0):.4f}")
        print(f"    FPR @ τ=0.974 = {score_summary.get('lockbox_real_dor_fpr_at_tau_0.9741', 0.0):.4f}")
    print(f"  Lockbox real other-Teams (n={score_summary['lockbox_real_nondor_n']}):")
    if score_summary["lockbox_real_nondor_n"] > 0:
        print(f"    mean prob_fake = {score_summary['lockbox_real_nondor_mean_prob_fake']:.4f}")
        print(f"    median        = {score_summary['lockbox_real_nondor_median']:.4f}")
        print(f"    FPR @ τ=0.50  = {score_summary.get('lockbox_real_nondor_fpr_at_tau_0.5', 0.0):.4f}")
        print(f"    FPR @ τ=0.92  = {score_summary.get('lockbox_real_nondor_fpr_at_tau_0.92', 0.0):.4f}")
        print(f"    FPR @ τ=0.974 = {score_summary.get('lockbox_real_nondor_fpr_at_tau_0.9741', 0.0):.4f}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"corrective_probes__{args.label}.json"
    with open(out, "w") as f:
        json.dump({
            "label": args.label,
            "arm": args.arm,
            "fresh_lr_l3_dev_to_lockbox_auc": fresh_lb_auc,
            "p8a_baseline_fresh_lr_l3_lockbox_auc": 0.9495,
            "fresh_lr_direction_cos_substrate_axes": cosines_fresh,
            "within_bucket3_dor_vs_other_cls_auc": bucket3_auc,
            "within_bucket3_dor_vs_other_l3_auc": bucket3_l3_auc,
            "within_bucket3_meta": bucket3_meta,
            "per_frame_score_summary": score_summary,
        }, f, indent=2, default=str)
    logger.info(f"Saved: {out}")


if __name__ == "__main__":
    main()
