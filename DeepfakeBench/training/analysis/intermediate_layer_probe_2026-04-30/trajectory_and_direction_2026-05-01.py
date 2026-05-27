"""P17 trained-head validation pack — trajectory + bootstrap CIs + fresh-LR direction.

Three things at once:

  1. TRAJECTORY: probe multiple ckpts per arm to see how lockbox AUC evolves
     across training. Tests the "step asymmetry" / "early-clean then drift"
     hypothesis. If anti-correlation appears late, early-stopping is a
     candidate recipe. If it's there from epoch 1, the trainer's signal is
     intrinsically substrate-flipping at this readout level.

  2. BOOTSTRAP CIs: resample lockbox (n=87) with replacement, recompute AUC,
     1000 iterations, report 95% CI. Confirms the 0.187 / 0.066 numbers are
     not statistical artifacts of small N.

  3. FRESH-LR DIRECTION COMPARISON: train fresh LR on dev features (the
     offline-probe baseline that gives lockbox AUC ~0.95). Compare its
     per-sample scores to each trained head's scores via Pearson r and
     Spearman rho, on dev and lockbox separately. Tells us *how* the trained
     heads diverge from the substrate-invariant signal — same direction with
     noise, orthogonal, or anti-aligned?

Run:
    python3 analysis/intermediate_layer_probe_2026-04-30/trajectory_and_direction_2026-05-01.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

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
N_SAMPLES = 800
N_BOOT = 1000

# (label, ckpt path, layer in cache, recipe-hint-string for ordering)
ARMS = [
    # ArcFace L3 trajectory
    ("L3_ARCFACE_ep1_auc0.5006",       CKPT_DIR / "first_best_effort_20260501_ep1_auc0.5006_eer0.4926.pth", 3),
    ("L3_ARCFACE_step1687_auc0.6804",  CKPT_DIR / "top_n_effort_20260501_step1687_auc0.6804_eer0.3775.pth", 3),
    ("L3_ARCFACE_step1928_auc0.7216",  CKPT_DIR / "top_n_effort_20260501_step1928_auc0.7216_eer0.3440.pth", 3),
    ("L3_ARCFACE_step2088_auc0.7390",  CKPT_DIR / "melp4mol_step2088.pth", 3),
    # LINEAR L3 trajectory (note: step883 was pruned by keep_last_n=3 before we could grab it)
    ("L3_LINEAR_ep1_auc0.4994",        CKPT_DIR / "first_best_effort_20260501_ep1_auc0.4994_eer0.4926.pth", 3),
    ("L3_LINEAR_step1044_auc0.6713",   CKPT_DIR / "top_n_effort_20260501_step1044_auc0.6713_eer0.3762.pth", 3),
    ("L3_LINEAR_step1205_auc0.7090",   CKPT_DIR / "top_n_effort_20260501_step1205_auc0.7090_eer0.3481.pth", 3),
    ("L3_LINEAR_step1285_auc0.7277",   CKPT_DIR / "nqvfz44v_step1285.pth", 3),
]


def l2norm(x, axis):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def auc_with_boot(labels, scores, n_boot=N_BOOT, seed=42):
    """Point AUC + bootstrap 95% CI (resample with replacement)."""
    if len(np.unique(labels)) < 2:
        return None, None, None, 0
    point = float(roc_auc_score(labels, scores))
    rng = np.random.default_rng(seed)
    n = len(labels)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], scores[idx]))
    aucs = np.array(aucs)
    return point, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)), len(aucs)


def apply_head(features, head_w, head_b, use_arcface):
    """Trained-head fake-vs-real margin under the trainer's recipe."""
    if use_arcface:
        f = l2norm(features, axis=1)
        w = l2norm(head_w, axis=1)
        logits = f @ w.T
    else:
        logits = features @ head_w.T
        if head_b is not None:
            logits = logits + head_b[None, :]
    return logits[:, 1] - logits[:, 0]


def fresh_lr_oof_dev_and_lb(feats_dv, lbl_dv, feats_lb):
    """Reproduce the offline reference probe.
    Returns:
      dev_scores (out-of-fold predict_proba[:,1] from 5-fold CV on dev)
      lb_scores (predict_proba[:,1] from LR trained on full dev)
      lr_full   (the LR fit on full dev — for weight extraction)
    The reference probe normalizes features before LR; we do the same.
    """
    feats_dv_n = l2norm(feats_dv, axis=1)
    feats_lb_n = l2norm(feats_lb, axis=1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    dev_oof = np.zeros(len(lbl_dv), dtype=np.float64)
    for tr, te in skf.split(feats_dv_n, lbl_dv):
        clf = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
        clf.fit(feats_dv_n[tr], lbl_dv[tr])
        dev_oof[te] = clf.predict_proba(feats_dv_n[te])[:, 1]
    lr_full = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
    lr_full.fit(feats_dv_n, lbl_dv)
    lb_scores = lr_full.predict_proba(feats_lb_n)[:, 1]
    return dev_oof, lb_scores, lr_full


def main():
    df = pd.read_csv(SAMPLED_CSV).iloc[:N_SAMPLES].reset_index(drop=True)
    label_int = (df["label"].astype(str) == "fake").astype(int).to_numpy()
    is_lb_all = (df["split"].astype(str) == "lockbox").to_numpy()
    is_dv_all = (df["split"].astype(str) == "dev").to_numpy()
    print(f"sample: dev={is_dv_all.sum()} lockbox={is_lb_all.sum()}")
    print(f"        dev fake/real={(label_int.astype(bool) & is_dv_all).sum()}/"
          f"{((~label_int.astype(bool)) & is_dv_all).sum()}  "
          f"lb fake/real={(label_int.astype(bool) & is_lb_all).sum()}/"
          f"{((~label_int.astype(bool)) & is_lb_all).sum()}")

    # === Fresh-LR baseline (computed once) ===
    cache_l3 = CACHE_DIR / f"intermediate__P8A__layer03__n{N_SAMPLES}.npz"
    blob = np.load(cache_l3)
    feats_l3 = blob["features"].astype(np.float32)
    valid_idx = blob["valid_idx"].astype(np.int64)
    lbl = label_int[valid_idx]
    is_dv_v = is_dv_all[valid_idx]
    is_lb_v = is_lb_all[valid_idx]

    print("\n=== Fresh-LR reference (dev-trained, normalized features) ===")
    fresh_dev_scores, fresh_lb_scores, lr_full = fresh_lr_oof_dev_and_lb(
        feats_l3[is_dv_v], lbl[is_dv_v], feats_l3[is_lb_v]
    )
    fresh_dev_auc, fresh_dev_lo, fresh_dev_hi, _ = auc_with_boot(lbl[is_dv_v], fresh_dev_scores)
    fresh_lb_auc,  fresh_lb_lo,  fresh_lb_hi,  _ = auc_with_boot(lbl[is_lb_v], fresh_lb_scores)
    print(f"  dev OOF AUC = {fresh_dev_auc:.4f}  CI95=[{fresh_dev_lo:.4f}, {fresh_dev_hi:.4f}]")
    print(f"  lb     AUC = {fresh_lb_auc:.4f}  CI95=[{fresh_lb_lo:.4f}, {fresh_lb_hi:.4f}]")
    lr_dir = lr_full.coef_.flatten()  # (768,) — direction in NORMALIZED feature space
    print(f"  LR coef ||w|| = {np.linalg.norm(lr_dir):.4f}")

    # === Per-arm trained-head probe with CI + correlation to fresh LR ===
    rows = []
    for arm_label, ckpt_path, layer in ARMS:
        if not ckpt_path.exists():
            print(f"\nSKIP {arm_label}: ckpt not found at {ckpt_path}")
            continue

        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        head_w = ck["state_dict"]["head.weight"].cpu().numpy().astype(np.float32)
        use_arcface = bool(ck["model_config"].get("use_arcface_head", True))
        if use_arcface:
            head_b = None
            head_dir_in_norm_space = (
                l2norm(head_w[1:2], axis=1)[0] - l2norm(head_w[0:1], axis=1)[0]
            )
        else:
            hb = ck["state_dict"].get("head.bias")
            head_b = hb.cpu().numpy().astype(np.float32) if hb is not None else None
            # LINEAR operates on raw features; project a normalized-space
            # comparison by treating its raw direction as the decision dir.
            head_dir_in_norm_space = head_w[1] - head_w[0]
        recipe = "ARCFACE" if use_arcface else "LINEAR"

        scores = apply_head(feats_l3, head_w, head_b, use_arcface)
        s_dev = scores[is_dv_v]
        s_lb = scores[is_lb_v]

        dev_auc, dev_lo, dev_hi, _ = auc_with_boot(lbl[is_dv_v], s_dev)
        lb_auc,  lb_lo,  lb_hi,  _ = auc_with_boot(lbl[is_lb_v], s_lb)

        # Correlation to fresh-LR scores (substrate-invariant signal)
        r_dev_p, _ = pearsonr(s_dev, fresh_dev_scores)
        r_dev_s, _ = spearmanr(s_dev, fresh_dev_scores)
        r_lb_p,  _ = pearsonr(s_lb, fresh_lb_scores)
        r_lb_s,  _ = spearmanr(s_lb, fresh_lb_scores)

        # Cosine similarity between directions in NORMALIZED feature space.
        # For ArcFace this is exact (the recipe lives in normed space).
        # For LINEAR this is approximate (recipe is in raw space) — useful
        # for "is the learned direction even pointing the same way?"
        cos_lr_head = float(np.dot(lr_dir, head_dir_in_norm_space) /
                            (np.linalg.norm(lr_dir) * np.linalg.norm(head_dir_in_norm_space) + 1e-12))

        print(f"\n=== {arm_label}  ({recipe}) ===")
        print(f"  WITHIN-DEV     n={int(is_dv_v.sum())}  AUC={dev_auc:.4f}  CI95=[{dev_lo:.4f}, {dev_hi:.4f}]")
        print(f"  WITHIN-LOCKBOX n={int(is_lb_v.sum())}  AUC={lb_auc:.4f}  CI95=[{lb_lo:.4f}, {lb_hi:.4f}]")
        print(f"  Correlation to fresh-LR scores:")
        print(f"     dev    Pearson r={r_dev_p:+.3f}  Spearman ρ={r_dev_s:+.3f}")
        print(f"     lockbox Pearson r={r_lb_p:+.3f}  Spearman ρ={r_lb_s:+.3f}")
        print(f"  Cosine(LR_dir, head_decision_dir) [normed-space approx] = {cos_lr_head:+.3f}")

        rows.append({
            "arm": arm_label,
            "recipe": recipe,
            "ckpt": ckpt_path.name,
            "use_arcface_head": use_arcface,
            "n_dev": int(is_dv_v.sum()),
            "n_lb": int(is_lb_v.sum()),
            "dev_auc": dev_auc, "dev_ci_lo": dev_lo, "dev_ci_hi": dev_hi,
            "lb_auc":  lb_auc,  "lb_ci_lo":  lb_lo,  "lb_ci_hi":  lb_hi,
            "pearson_r_dev_vs_freshLR":  float(r_dev_p),
            "spearman_r_dev_vs_freshLR": float(r_dev_s),
            "pearson_r_lb_vs_freshLR":   float(r_lb_p),
            "spearman_r_lb_vs_freshLR":  float(r_lb_s),
            "cosine_dir_vs_freshLR":     cos_lr_head,
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUTPUT_DIR / "trajectory_and_direction_2026-05-01.json"
    out_csv = OUTPUT_DIR / "trajectory_and_direction_2026-05-01.csv"
    with open(out_json, "w") as f:
        json.dump({
            "fresh_lr_baseline": {
                "dev_auc": fresh_dev_auc, "dev_ci_lo": fresh_dev_lo, "dev_ci_hi": fresh_dev_hi,
                "lb_auc":  fresh_lb_auc,  "lb_ci_lo":  fresh_lb_lo,  "lb_ci_hi":  fresh_lb_hi,
            },
            "arms": rows,
        }, f, indent=2)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Summary table
    print()
    print("=" * 130)
    print("TRAJECTORY + BOOTSTRAP CI + FRESH-LR DIRECTION COMPARISON")
    print("=" * 130)
    print(f"FRESH-LR baseline (substrate-invariant ref):  "
          f"dev AUC = {fresh_dev_auc:.4f} [CI {fresh_dev_lo:.3f}, {fresh_dev_hi:.3f}]   "
          f"lb AUC = {fresh_lb_auc:.4f} [CI {fresh_lb_lo:.3f}, {fresh_lb_hi:.3f}]")
    print("-" * 130)
    hdr = f"{'arm':<36} {'recipe':<8} {'dev_AUC':>8} {'dev_CI':>17} {'lb_AUC':>8} {'lb_CI':>17} {'r_lb':>6} {'cos_dir':>8}"
    print(hdr)
    print("-" * 130)
    for r in rows:
        print(f"{r['arm']:<36} {r['recipe']:<8} "
              f"{r['dev_auc']:>8.4f} [{r['dev_ci_lo']:.3f},{r['dev_ci_hi']:.3f}] "
              f"{r['lb_auc']:>8.4f} [{r['lb_ci_lo']:.3f},{r['lb_ci_hi']:.3f}] "
              f"{r['pearson_r_lb_vs_freshLR']:+6.3f} {r['cosine_dir_vs_freshLR']:+8.3f}")
    print("=" * 130)
    print(f"  outputs: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
