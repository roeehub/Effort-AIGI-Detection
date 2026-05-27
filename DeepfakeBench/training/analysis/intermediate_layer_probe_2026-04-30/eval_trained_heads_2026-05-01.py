"""Apply the TRAINED ArcFace head to cached layer-3 features and measure
dev / lockbox / transfer AUC.

The substrate-invariance hypothesis test:
  Offline probe (per_layer_split_probe.py) found that a fresh LR on layer-3
  P8A features achieves dev→lockbox transfer AUC = 0.95.

  Question: does the TRAINED head from the L3 ArcFace run (melp4mol) preserve
  this transfer property, or does the trainer's head re-learn the dev-substrate
  shortcut?

  This script answers that. If transfer AUC ≥ 0.85, hypothesis is alive at the
  trained-head level. If transfer AUC ≪ within-dev AUC, the head learned
  substrate, not label.

Run:
    python3 analysis/intermediate_layer_probe_2026-04-30/eval_trained_heads_2026-05-01.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
SAMPLED_CSV = (
    REPO_ROOT
    / "analysis"
    / "embedding_triptych_2026-04-30"
    / "outputs"
    / "triptych_p8a_slot2_slot3"
    / "sampled_frames.csv"
)
OUTPUT_DIR = REPO_ROOT / "analysis" / "intermediate_layer_probe_2026-04-30" / "outputs"

CKPT_DIR = Path("/tmp/p17_ckpts")

# (label, ckpt path, layer in cache)
ARMS = [
    ("L3_ARCFACE_melp4mol_step2088", CKPT_DIR / "melp4mol_step2088.pth", 3),
    ("L3_LINEAR_nqvfz44v_step1285", CKPT_DIR / "nqvfz44v_step1285.pth", 3),
    # L4 needs layer-4 feature extraction (not in cache); skipping for now.
]

N_SAMPLES = 800


def l2_normalize(x: np.ndarray, axis: int) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def metrics_at(labels: np.ndarray, scores: np.ndarray) -> dict:
    out = {"auc": float(roc_auc_score(labels, scores))}
    fpr, tpr, _ = roc_curve(labels, scores)
    for tgt in (0.05, 0.10):
        eligible = np.where(fpr <= tgt + 1e-12)[0]
        out[f"rec@fpr_{tgt:.2f}"] = (
            float(tpr[eligible[np.argmax(tpr[eligible])]]) if len(eligible) else 0.0
        )
    return out


def apply_trained_head(
    features: np.ndarray,
    head_weight: np.ndarray,
    *,
    use_arcface_head: bool,
    head_bias: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the trained head and return fake-vs-real margin.

    ArcFace recipe (use_arcface_head=True, normalize_features_before_head=True):
        f_n = features / ||features||
        w_n = head.weight / ||head.weight||
        logits = s * (f_n @ w_n.T)
        scalar s drops out of AUC ranking → omitted

    Plain Linear recipe (use_arcface_head=False, normalize_features_before_head=False):
        logits = features @ head.weight.T + head.bias
        head.bias is constant per class → adds (b[1]-b[0]) to every margin
        AUC and rec@FPR are invariant under that constant shift, but we still
        include it so downstream consumers see the actual margin.
    """
    if use_arcface_head:
        f_n = l2_normalize(features, axis=1)         # (N, 768)
        w_n = l2_normalize(head_weight, axis=1)      # (2, 768)
        logits = f_n @ w_n.T                          # (N, 2)
    else:
        logits = features @ head_weight.T             # (N, 2)
        if head_bias is not None:
            logits = logits + head_bias[None, :]
    return logits[:, 1] - logits[:, 0]                # fake-vs-real margin


def main():
    df = pd.read_csv(SAMPLED_CSV).iloc[:N_SAMPLES].reset_index(drop=True)
    label_int = (df["label"].astype(str) == "fake").astype(int).to_numpy()
    is_lb = (df["split"].astype(str) == "lockbox").to_numpy()
    is_dv = (df["split"].astype(str) == "dev").to_numpy()
    print(f"sample: dev={is_dv.sum()} lockbox={is_lb.sum()}  "
          f"dev_fake={(label_int.astype(bool) & is_dv).sum()} "
          f"dev_real={((~label_int.astype(bool)) & is_dv).sum()} "
          f"lb_fake={(label_int.astype(bool) & is_lb).sum()} "
          f"lb_real={((~label_int.astype(bool)) & is_lb).sum()}")
    print()

    rows = []
    for arm_label, ckpt_path, layer in ARMS:
        if not ckpt_path.exists():
            print(f"SKIP {arm_label}: ckpt not found at {ckpt_path}")
            continue
        cache_path = CACHE_DIR / f"intermediate__P8A__layer{layer:02d}__n{N_SAMPLES}.npz"
        if not cache_path.exists():
            print(f"SKIP {arm_label}: cache not found at {cache_path}")
            continue

        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        head_w = ck["state_dict"]["head.weight"].cpu().numpy().astype(np.float32)  # (2, 768)
        use_arcface = bool(ck["model_config"].get("use_arcface_head", True))
        if use_arcface:
            head_s = float(ck["state_dict"]["head.s"].cpu().numpy())
            head_b = None
        else:
            head_s = None
            hb = ck["state_dict"].get("head.bias")
            head_b = hb.cpu().numpy().astype(np.float32) if hb is not None else None
        ckpt_step = ck.get("training_step")
        ckpt_auc = ck.get("auc")
        cur_s = ck["model_config"].get("current_arcface_s")

        blob = np.load(cache_path)
        feats = blob["features"].astype(np.float32)
        valid_idx = blob["valid_idx"].astype(np.int64)
        lbl = label_int[valid_idx]
        is_dv_v = is_dv[valid_idx]
        is_lb_v = is_lb[valid_idx]

        scores = apply_trained_head(
            feats, head_w, use_arcface_head=use_arcface, head_bias=head_b
        )
        m_dev = metrics_at(lbl[is_dv_v], scores[is_dv_v])
        m_lb = metrics_at(lbl[is_lb_v], scores[is_lb_v])
        m_all = metrics_at(lbl, scores)

        recipe = "ARCFACE" if use_arcface else "LINEAR"
        head_s_str = f"{head_s:.4f}" if head_s is not None else "n/a"
        head_b_str = (
            f"{head_b.tolist()}" if head_b is not None else "n/a"
        )
        print(f"=== {arm_label} ===")
        print(f"  recipe: {recipe}  ckpt: {ckpt_path.name}  reported_AUC={ckpt_auc:.4f}")
        print(f"  head.s={head_s_str}  current_s={cur_s}  head.bias={head_b_str}")
        print(f"  feats:  {feats.shape}  norm_mean={np.linalg.norm(feats, axis=1).mean():.3f}")
        print(f"  head.w: {head_w.shape}  norm_per_class={np.linalg.norm(head_w, axis=1)}")
        print(f"  WITHIN-DEV     n={is_dv_v.sum()}  AUC={m_dev['auc']:.4f}  "
              f"rec@5%FPR={m_dev['rec@fpr_0.05']:.4f}  rec@10%FPR={m_dev['rec@fpr_0.10']:.4f}")
        print(f"  WITHIN-LOCKBOX n={is_lb_v.sum()}  AUC={m_lb['auc']:.4f}  "
              f"rec@5%FPR={m_lb['rec@fpr_0.05']:.4f}  rec@10%FPR={m_lb['rec@fpr_0.10']:.4f}")
        print(f"  TRANSFER (dev-trained head, lockbox-tested)  AUC={m_lb['auc']:.4f}  "
              f"≡ WITHIN-LOCKBOX (since the head IS dev-trained)")
        print(f"  ALL n={len(lbl)}  AUC={m_all['auc']:.4f}")
        print()

        rows.append({
            "arm": arm_label,
            "recipe": recipe,
            "ckpt": ckpt_path.name,
            "ckpt_reported_auc": ckpt_auc,
            "head_s": head_s,
            "head_bias": head_b.tolist() if head_b is not None else None,
            "n_dev": int(is_dv_v.sum()),
            "n_lockbox": int(is_lb_v.sum()),
            "within_dev_auc": m_dev["auc"],
            "within_dev_rec_at_5fpr": m_dev["rec@fpr_0.05"],
            "lockbox_auc": m_lb["auc"],
            "lockbox_rec_at_5fpr": m_lb["rec@fpr_0.05"],
            "all_auc": m_all["auc"],
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "trained_head_eval_2026-05-01.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {out_path}")

    # Reference comparison line (from per_layer_split_probe.csv):
    print()
    print("Reference (fresh LR on same features, layer 3):")
    print("  P8A within-dev  AUC=0.9390   within-lockbox  AUC=0.9729   transfer  AUC=0.9495")
    print("If trained-head lockbox AUC ≥ 0.85 → substrate-invariance preserved.")
    print("If trained-head lockbox AUC ≪ within-dev → head learned dev shortcut.")


if __name__ == "__main__":
    sys.exit(main())
