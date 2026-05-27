"""Head-vs-feature decomposition probe (2026-04-30).

Question: is the calibration cliff that shows up in `mclioexb`'s production-grade
scorecard (does NOT promote — see `analysis/scorecard_mclioexb_2026-04-30/`) caused
by FT-from-P8A reshaping the [CLS] features in a calibration-hostile way, or is
it inherited from P8A's frozen features (in which case the head + scale shift is
the entire mclioexb-vs-P8A story)?

Decomposition. Both checkpoints use ArcFace heads:

    cosine = normalize(features) @ normalize(head.weight).T
    logits = head.s * cosine
    prob_fake = softmax(logits)[:, 1]

So the model factorises into (a) features, (b) head weight directions, (c) scale s.
This script reuses the cached 800-frame stratified-lockbox features from
`analysis/_features_cache_2026-04-30/triptych_features__{P8A,SLOT3_JITTER}__n800.npz`,
loads each checkpoint's head, then computes prob_fake under all combinations:

    natural_p   = head_p (W_p, s_p) on F_p           -- P8A baseline
    natural_m   = head_m (W_m, s_m) on F_m           -- mclioexb baseline
    Wp_on_Fm    = head_p (W_p, s_p) on F_m           -- swap features
    Wm_on_Fp    = head_m (W_m, s_m) on F_p           -- swap features
    Wp_sm_Fp    = (W_p, s_m) on F_p                  -- swap scale only
    Wm_sp_Fm    = (W_m, s_p) on F_m                  -- swap scale only

For each variant we compute fake-recall at fixed real-FPR (0.02, 0.05, 0.10) on
the 800-frame substrate (476 real / 324 fake). We also fit a fresh logistic
regression on each feature set as an "ideal head" upper bound, and report
feature/weight geometry diffs.

Reading guide:

    natural_p recall@0.02 vs natural_m recall@0.02     -- the cliff itself
    Wp_on_Fp vs Wp_on_Fm                               -- pure feature shift effect
    Wp_on_Fp vs Wm_on_Fp                               -- pure head shift effect
    natural_p vs Wp_sm_Fp                              -- scale-only ablation
    fresh-logistic-on-Fp vs natural_p                  -- is P8A's head sub-optimal on its own features?
    fresh-logistic-on-Fm vs natural_m                  -- is mclioexb's head sub-optimal on its own features?

If the cliff is HEAD-side (head-side / calibration loss in P16 is high-value):
    natural_m recall@0.02 << natural_p recall@0.02
    fresh-logistic-on-Fm recall@0.02 ~ fresh-logistic-on-Fp recall@0.02 (features have similar separability)
    Wp_on_Fm recall@0.02 ~ natural_p recall@0.02 (P8A head still works on mclioexb features)

If the cliff is FEATURE-side (substrate or different FT base in P16):
    fresh-logistic-on-Fm recall@0.02 << fresh-logistic-on-Fp recall@0.02
    Wp_on_Fm recall@0.02 ~ natural_m recall@0.02 (head can't fix it)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CACHE_DIR = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
SAMPLED_CSV = (
    REPO_ROOT
    / "analysis"
    / "embedding_triptych_2026-04-30"
    / "outputs"
    / "triptych_p8a_slot2_slot3"
    / "sampled_frames.csv"
)

# Canonical (label, ckpt-filename, run_id) mapping. Filenames are the local
# files already downloaded by the triptych / face-size-invariance probes.
CANONICAL_CKPTS = {
    "P8A": {
        "ckpt_file": "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth",
        "run_id": "9lmvb5b4",
        "recipe": "P8A_REFERENCE_step5000 (FT origin, baseline)",
    },
    "MCLIOEXB": {
        "ckpt_file": "value_composite_effort_20260429_step500_auc0.9797_eer0.0521.pth",
        "run_id": "mclioexb",
        "recipe": "P14_FACE_SCALE_JITTER_ISOLATED step 500 (jitter@0.50, FT-from-P8A_step5000)",
    },
}

logger = logging.getLogger("head-feature-decomp")


# -----------------------------------------------------------------------------
# Data loading.
# -----------------------------------------------------------------------------
def load_cached_features(label: str) -> Tuple[np.ndarray, np.ndarray]:
    cache_label = {"P8A": "P8A", "MCLIOEXB": "SLOT3_JITTER"}[label]
    p = CACHE_DIR / f"triptych_features__{cache_label}__n800.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"Cached features not found: {p}. Re-run the embedding triptych first."
        )
    blob = np.load(p, allow_pickle=False)
    return blob["features"].astype(np.float32), blob["valid_idx"]


def load_head_from_ckpt(ckpt_file: str) -> Tuple[np.ndarray, float]:
    p = CACHE_DIR / ckpt_file
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found locally: {p}")
    state = torch.load(str(p), map_location="cpu", weights_only=False)
    sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    sd = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())
    if "head.weight" not in sd:
        raise KeyError(f"head.weight missing from {p} state_dict")
    weight = sd["head.weight"].detach().cpu().numpy().astype(np.float32)
    # ArcMarginProduct registers s as a buffer; current_arcface_s in model_config
    # is the post-warmup deployed scale. Prefer the buffer if present.
    if "head.s" in sd:
        s = float(sd["head.s"].detach().cpu().item())
    else:
        mc = state.get("model_config", {}) if isinstance(state, dict) else {}
        s = float(mc.get("current_arcface_s", 30.0))
    return weight, s


# -----------------------------------------------------------------------------
# Head forward (matches detectors/effort_detector.py:42-54 inference path).
# -----------------------------------------------------------------------------
def head_forward_np(weight: np.ndarray, s: float, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised numpy implementation. Returns (prob_fake, raw_cosine)."""
    f_t = torch.from_numpy(feats)
    w_t = torch.from_numpy(weight)
    cos = F.linear(F.normalize(f_t, dim=1), F.normalize(w_t, dim=1))
    logits = float(s) * cos
    prob = torch.softmax(logits, dim=1)[:, 1]
    return prob.numpy(), cos.numpy()


# -----------------------------------------------------------------------------
# Metrics.
# -----------------------------------------------------------------------------
def recall_at_fpr(probs: np.ndarray, labels_int: np.ndarray, fpr_targets: list[float]) -> Dict[str, dict]:
    """labels_int: 1 = fake, 0 = real."""
    fpr, tpr, thr = roc_curve(labels_int, probs)
    out: Dict[str, dict] = {}
    for tgt in fpr_targets:
        # First operating point with FPR <= target (largest such tau, lowest recall valid).
        eligible = np.where(fpr <= tgt + 1e-12)[0]
        if len(eligible) == 0:
            out[f"fpr@{tgt:.2f}"] = {"recall": 0.0, "tau": None, "actual_fpr": 0.0}
            continue
        best = eligible[np.argmax(tpr[eligible])]
        out[f"fpr@{tgt:.2f}"] = {
            "recall": float(tpr[best]),
            "tau": float(thr[best]) if np.isfinite(thr[best]) else None,
            "actual_fpr": float(fpr[best]),
        }
    return out


def auc_score(probs: np.ndarray, labels_int: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels_int, probs))


def fit_fresh_logistic(feats: np.ndarray, labels_int: np.ndarray) -> np.ndarray:
    """5-fold CV out-of-fold prob_fake on L2-normalised features. n_jobs=1
    intentionally — see memory feedback_sklearn_njobs.md."""
    from sklearn.model_selection import StratifiedKFold

    feats_n = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    oof = np.zeros(len(labels_int), dtype=np.float64)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(feats_n, labels_int):
        clf = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
        clf.fit(feats_n[tr], labels_int[tr])
        oof[te] = clf.predict_proba(feats_n[te])[:, 1]
    return oof


# -----------------------------------------------------------------------------
# Plotting.
# -----------------------------------------------------------------------------
def plot_prob_distributions(
    variant_probs: Dict[str, np.ndarray],
    labels_int: np.ndarray,
    output_png: Path,
) -> None:
    n = len(variant_probs)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.0 * rows))
    axes = axes.flatten()
    for i, (name, probs) in enumerate(variant_probs.items()):
        ax = axes[i]
        bins = np.linspace(0, 1, 41)
        ax.hist(probs[labels_int == 0], bins=bins, alpha=0.55, color="#4477aa", label=f"real (n={int((labels_int==0).sum())})")
        ax.hist(probs[labels_int == 1], bins=bins, alpha=0.55, color="#cc6677", label=f"fake (n={int((labels_int==1).sum())})")
        ax.axvline(0.5, color="black", linestyle="--", alpha=0.4)
        ax.axvline(0.97, color="red", linestyle=":", alpha=0.6, label="cliff τ≈0.97")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("prob_fake")
        ax.set_yscale("log")
        ax.legend(fontsize=7, loc="upper center")
    for j in range(len(variant_probs), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Head × feature decomposition — prob_fake distributions on 800-frame stratified lockbox", y=1.0)
    fig.tight_layout()
    fig.savefig(output_png, dpi=120)
    plt.close(fig)


def plot_recall_fpr(
    variant_probs: Dict[str, np.ndarray],
    labels_int: np.ndarray,
    output_png: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    palette = ["#cc6677", "#4477aa", "#117733", "#882255", "#88ccee", "#ddcc77", "#aa4499", "#999933"]
    for i, (name, probs) in enumerate(variant_probs.items()):
        fpr, tpr, _ = roc_curve(labels_int, probs)
        ax.plot(fpr, tpr, "-", color=palette[i % len(palette)], linewidth=1.3, label=name, alpha=0.85)
    for x in (0.02, 0.05, 0.10):
        ax.axvline(x, color="grey", linestyle=":", alpha=0.4)
        ax.text(x, 0.02, f"FPR={x}", rotation=90, color="grey", fontsize=8)
    ax.set_xlabel("FPR (real false-positive rate)")
    ax.set_ylabel("TPR / fake recall")
    ax.set_xlim(0, 0.30)
    ax.set_ylim(0, 1.02)
    ax.set_title("ROC tail — operating region (FPR ≤ 0.30)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_png, dpi=120)
    plt.close(fig)


def plot_feature_similarity(F_p: np.ndarray, F_m: np.ndarray, labels_int: np.ndarray, output_png: Path) -> None:
    Fp_n = F_p / (np.linalg.norm(F_p, axis=1, keepdims=True) + 1e-12)
    Fm_n = F_m / (np.linalg.norm(F_m, axis=1, keepdims=True) + 1e-12)
    cos_sim = np.einsum("ij,ij->i", Fp_n, Fm_n)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(-0.2, 1.0, 60)
    axes[0].hist(cos_sim, bins=bins, color="#117733", alpha=0.8)
    axes[0].set_title(
        f"Per-frame cos(F_P8A, F_mclioexb)\n"
        f"mean={cos_sim.mean():.3f}  median={np.median(cos_sim):.3f}  min={cos_sim.min():.3f}"
    )
    axes[0].set_xlabel("cosine similarity")
    axes[0].axvline(1.0, color="black", linestyle=":", alpha=0.4)
    axes[1].hist(cos_sim[labels_int == 0], bins=bins, alpha=0.6, color="#4477aa", label="real")
    axes[1].hist(cos_sim[labels_int == 1], bins=bins, alpha=0.6, color="#cc6677", label="fake")
    axes[1].set_title("by label")
    axes[1].set_xlabel("cosine similarity")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=120)
    plt.close(fig)
    return cos_sim


def head_geometry(W_p: np.ndarray, W_m: np.ndarray, s_p: float, s_m: float) -> Dict[str, float]:
    """L2-normalised columns: ArcFace head normalises weight rows in forward."""
    Wp_n = W_p / (np.linalg.norm(W_p, axis=1, keepdims=True) + 1e-12)  # [2, 512]
    Wm_n = W_m / (np.linalg.norm(W_m, axis=1, keepdims=True) + 1e-12)
    # Per-class cosine between P8A and mclioexb weight directions
    real_cos = float((Wp_n[0] * Wm_n[0]).sum())
    fake_cos = float((Wp_n[1] * Wm_n[1]).sum())
    # Within-model real-vs-fake direction angle
    intra_p = float((Wp_n[0] * Wp_n[1]).sum())
    intra_m = float((Wm_n[0] * Wm_n[1]).sum())
    return {
        "weight_cos_real_class_P8A_vs_mclioexb": real_cos,
        "weight_cos_fake_class_P8A_vs_mclioexb": fake_cos,
        "intra_real_vs_fake_cos_P8A": intra_p,
        "intra_real_vs_fake_cos_mclioexb": intra_m,
        "scale_s_P8A": float(s_p),
        "scale_s_mclioexb": float(s_m),
        "scale_ratio_mclioexb_over_P8A": float(s_m / s_p),
    }


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Head-vs-feature decomposition probe")
    ap.add_argument("--output_dir", type=Path, default=REPO_ROOT / "analysis" / "head_feature_decomposition_2026-04-30" / "outputs")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading cached features and metadata")
    F_p, idx_p = load_cached_features("P8A")
    F_m, idx_m = load_cached_features("MCLIOEXB")
    if not (np.array_equal(idx_p, idx_m)):
        common = np.intersect1d(idx_p, idx_m)
        logger.warning("valid_idx differ; intersecting to %d frames", len(common))
        pos_p = {int(v): r for r, v in enumerate(idx_p)}
        pos_m = {int(v): r for r, v in enumerate(idx_m)}
        rows_p = [pos_p[int(v)] for v in common]
        rows_m = [pos_m[int(v)] for v in common]
        F_p = F_p[rows_p]
        F_m = F_m[rows_m]
        valid_idx = common
    else:
        valid_idx = idx_p

    sampled = pd.read_csv(SAMPLED_CSV)
    meta = sampled.iloc[valid_idx].reset_index(drop=True)
    labels_int = (meta["label"].astype(str) == "fake").astype(np.int64).to_numpy()
    logger.info("aligned: %d frames (%d real, %d fake)", len(meta), int((labels_int == 0).sum()), int((labels_int == 1).sum()))

    logger.info("Loading heads from checkpoints")
    W_p, s_p = load_head_from_ckpt(CANONICAL_CKPTS["P8A"]["ckpt_file"])
    W_m, s_m = load_head_from_ckpt(CANONICAL_CKPTS["MCLIOEXB"]["ckpt_file"])
    logger.info("P8A head: weight=%s, s=%.4f", W_p.shape, s_p)
    logger.info("mclioexb head: weight=%s, s=%.4f", W_m.shape, s_m)

    logger.info("Computing the 6 head×feature combinations")
    variants: Dict[str, np.ndarray] = {}
    variants["natural_P8A          (Wp,sp,Fp)"], _ = head_forward_np(W_p, s_p, F_p)
    variants["natural_mclioexb     (Wm,sm,Fm)"], _ = head_forward_np(W_m, s_m, F_m)
    variants["Wp_on_Fm             (Wp,sp,Fm)"], _ = head_forward_np(W_p, s_p, F_m)
    variants["Wm_on_Fp             (Wm,sm,Fp)"], _ = head_forward_np(W_m, s_m, F_p)
    variants["Wp_with_sm_on_Fp     (Wp,sm,Fp)"], _ = head_forward_np(W_p, s_m, F_p)
    variants["Wm_with_sp_on_Fm     (Wm,sp,Fm)"], _ = head_forward_np(W_m, s_p, F_m)

    logger.info("Fitting fresh 5-fold CV logistic regressions on each feature set (n_jobs=1)")
    variants["fresh_LR_on_Fp       (oracle head)"] = fit_fresh_logistic(F_p, labels_int)
    variants["fresh_LR_on_Fm       (oracle head)"] = fit_fresh_logistic(F_m, labels_int)

    logger.info("Computing recall@FPR and AUC for every variant")
    metrics: Dict[str, dict] = {}
    fpr_targets = [0.02, 0.05, 0.10]
    for name, probs in variants.items():
        metrics[name.strip()] = {
            "auc": auc_score(probs, labels_int),
            **recall_at_fpr(probs, labels_int, fpr_targets),
        }

    logger.info("Computing feature similarity")
    cos_sim = plot_feature_similarity(F_p, F_m, labels_int, args.output_dir / "feature_similarity.png")
    feat_stats = {
        "per_frame_cos_F_P8A_vs_F_mclioexb_mean": float(cos_sim.mean()),
        "per_frame_cos_F_P8A_vs_F_mclioexb_median": float(np.median(cos_sim)),
        "per_frame_cos_min": float(cos_sim.min()),
        "per_frame_cos_max": float(cos_sim.max()),
        "frac_below_0_95": float((cos_sim < 0.95).mean()),
        "frac_below_0_90": float((cos_sim < 0.90).mean()),
        "frac_below_0_80": float((cos_sim < 0.80).mean()),
    }

    geom = head_geometry(W_p, W_m, s_p, s_m)

    logger.info("Plotting")
    plot_prob_distributions(variants, labels_int, args.output_dir / "prob_distributions.png")
    plot_recall_fpr(variants, labels_int, args.output_dir / "recall_fpr_curves.png")

    # Per-frame CSV.
    out_df = meta[["local_path", "label", "method", "clip_capture_mode", "identity_key"]].copy()
    out_df["per_frame_cos_Fp_vs_Fm"] = cos_sim
    for name, probs in variants.items():
        out_df[f"prob_fake__{name.split()[0]}"] = probs
    out_df.to_csv(args.output_dir / "per_frame_probs.csv", index=False)

    # Summary JSON.
    summary = {
        "probe": "head_vs_feature_decomposition",
        "date": "2026-04-30",
        "ckpts": CANONICAL_CKPTS,
        "n_frames": int(len(meta)),
        "n_real": int((labels_int == 0).sum()),
        "n_fake": int((labels_int == 1).sum()),
        "feature_geometry": feat_stats,
        "head_geometry": geom,
        "metrics_per_variant": metrics,
    }
    summary["interpretation"] = derive_interpretation(metrics, feat_stats, geom)

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 78)
    print("HEAD × FEATURE DECOMPOSITION  (2026-04-30, P8A vs mclioexb)")
    print("=" * 78)
    print(f"  n_frames         : {len(meta)}  (real={int((labels_int==0).sum())}  fake={int((labels_int==1).sum())})")
    print(f"  scale s          : P8A={s_p:.3f}   mclioexb={s_m:.3f}   ratio={s_m/s_p:.3f}")
    print(f"  head weight cos  : real-class={geom['weight_cos_real_class_P8A_vs_mclioexb']:.4f}  "
          f"fake-class={geom['weight_cos_fake_class_P8A_vs_mclioexb']:.4f}")
    print(f"  feature cos (per-frame) : mean={feat_stats['per_frame_cos_F_P8A_vs_F_mclioexb_mean']:.4f}  "
          f"median={feat_stats['per_frame_cos_F_P8A_vs_F_mclioexb_median']:.4f}  "
          f"frac<0.90={feat_stats['frac_below_0_90']:.3f}")
    print()
    print(f"{'variant':<42} {'AUC':>7} {'rec@.02':>9} {'rec@.05':>9} {'rec@.10':>9}")
    print("-" * 78)
    for name, m in metrics.items():
        print(
            f"{name:<42} {m['auc']:>7.4f} "
            f"{m['fpr@0.02']['recall']:>9.4f} "
            f"{m['fpr@0.05']['recall']:>9.4f} "
            f"{m['fpr@0.10']['recall']:>9.4f}"
        )
    print()
    print("Interpretation:")
    for line in summary["interpretation"]:
        print("  - " + line)
    print()
    print(f"Outputs in {args.output_dir}")
    print("=" * 78)
    return 0


def derive_interpretation(metrics: dict, feat_stats: dict, geom: dict) -> list[str]:
    out: list[str] = []
    nat_p = metrics["natural_P8A          (Wp,sp,Fp)"]
    nat_m = metrics["natural_mclioexb     (Wm,sm,Fm)"]
    wp_fm = metrics["Wp_on_Fm             (Wp,sp,Fm)"]
    wm_fp = metrics["Wm_on_Fp             (Wm,sm,Fp)"]
    fresh_p = metrics["fresh_LR_on_Fp       (oracle head)"]
    fresh_m = metrics["fresh_LR_on_Fm       (oracle head)"]
    wp_sm_fp = metrics["Wp_with_sm_on_Fp     (Wp,sm,Fp)"]

    delta_natural = nat_p["fpr@0.02"]["recall"] - nat_m["fpr@0.02"]["recall"]
    out.append(
        f"natural_P8A vs natural_mclioexb at FPR≤0.02: "
        f"P8A={nat_p['fpr@0.02']['recall']:.3f} vs mclioexb={nat_m['fpr@0.02']['recall']:.3f} "
        f"(Δ={delta_natural:+.3f}). The cliff = {delta_natural:+.3f}."
    )

    head_swap_effect_on_Fp = wm_fp["fpr@0.02"]["recall"] - nat_p["fpr@0.02"]["recall"]
    feat_swap_effect_on_Wp = wp_fm["fpr@0.02"]["recall"] - nat_p["fpr@0.02"]["recall"]
    out.append(
        f"On P8A features (F_p), swapping head P8A→mclioexb shifts recall@.02 by "
        f"{head_swap_effect_on_Fp:+.3f} (HEAD-side contribution)."
    )
    out.append(
        f"With P8A head (W_p), swapping features F_p→F_m shifts recall@.02 by "
        f"{feat_swap_effect_on_Wp:+.3f} (FEATURE-side contribution)."
    )

    if abs(feat_swap_effect_on_Wp) > 1.5 * abs(head_swap_effect_on_Fp) and feat_swap_effect_on_Wp < -0.05:
        out.append("=> Cliff is dominantly FEATURE-side: jitter@0.50 reshaped features in a calibration-hostile way.")
    elif abs(head_swap_effect_on_Fp) > 1.5 * abs(feat_swap_effect_on_Wp) and head_swap_effect_on_Fp < -0.05:
        out.append("=> Cliff is dominantly HEAD-side: features are still ~P8A-grade; mclioexb head/scale is the issue.")
    else:
        out.append("=> Cliff is mixed/intermediate between head and feature; neither single swap explains it.")

    out.append(
        f"Fresh oracle logistic on F_p: rec@.02={fresh_p['fpr@0.02']['recall']:.3f}; "
        f"on F_m: rec@.02={fresh_m['fpr@0.02']['recall']:.3f}. "
        f"Δ_fresh={fresh_p['fpr@0.02']['recall'] - fresh_m['fpr@0.02']['recall']:+.3f}."
    )
    if fresh_p["fpr@0.02"]["recall"] - fresh_m["fpr@0.02"]["recall"] > 0.05:
        out.append("    Fresh-LR confirms F_m has WORSE separability than F_p at the operating point — feature-side problem persists even with an oracle head.")
    elif fresh_m["fpr@0.02"]["recall"] - fresh_p["fpr@0.02"]["recall"] > 0.05:
        out.append("    Fresh-LR finds F_m has BETTER separability than F_p — mclioexb's natural head is mis-calibrated; head-side fix could recover.")
    else:
        out.append("    Fresh-LR finds F_p and F_m have ~equal separability; cliff is neither feature-quality nor head-direction (probably calibration / scale).")

    scale_only = wp_sm_fp["fpr@0.02"]["recall"] - nat_p["fpr@0.02"]["recall"]
    out.append(
        f"Scale-only swap (W_p, s_m, F_p): rec@.02 shifts by {scale_only:+.3f} relative to natural P8A — "
        f"isolates effect of mclioexb's smaller s ({geom['scale_s_mclioexb']:.2f} vs P8A's {geom['scale_s_P8A']:.2f})."
    )

    if feat_stats["per_frame_cos_F_P8A_vs_F_mclioexb_mean"] > 0.95:
        out.append(f"Features are nearly identical (per-frame cos mean={feat_stats['per_frame_cos_F_P8A_vs_F_mclioexb_mean']:.3f}) — backbone barely moved during FT.")
    elif feat_stats["per_frame_cos_F_P8A_vs_F_mclioexb_mean"] > 0.80:
        out.append(f"Features moved modestly during FT (per-frame cos mean={feat_stats['per_frame_cos_F_P8A_vs_F_mclioexb_mean']:.3f}).")
    else:
        out.append(f"Features moved substantially during FT (per-frame cos mean={feat_stats['per_frame_cos_F_P8A_vs_F_mclioexb_mean']:.3f}).")

    return out


if __name__ == "__main__":
    raise SystemExit(main())
