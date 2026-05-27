"""Diagnostics A + B + C + E from HANDOFF_2026-05-02_P18_CORRECTIVE.md.

A. Bootstrap CIs on within-bucket-3 dor-vs-other LR AUC (n_dor=9, n_other=256).
B. Per-frame paired test on 25 dor_shkedi lockbox real frames.
C. ArcFace s-scaling (multiply margin by s ≈ 9.75 for P8A or 12.0 for P18) and
   re-evaluate FPR @ τ=0.92, 0.974.
E. Within-bucket-3 dor-vs-other LR at intermediate layers L3, L6, L9, L11 to
   localize where the GRL effect lives.

Inputs: cached features in analysis/_features_cache_2026-04-30/ (produced by
extract_all_arms_layers.py).

CPU-only. n_jobs=1. Deterministic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CACHE_DIR = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
SAMPLED_CSV = REPO_ROOT / "analysis" / "embedding_triptych_2026-04-30" / "outputs" / "triptych_p8a_slot2_slot3" / "sampled_frames.csv"
OUTPUT_DIR = REPO_ROOT / "analysis" / "p18_probe_2026-05-01" / "outputs"

CKPTS = {
    "P8A": REPO_ROOT / "analysis" / "_features_cache_2026-04-30" / "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth",
    "P18T": Path("/tmp/p18_ckpts/xpbvc1e4__periodic_step4000.pth"),
    "P18C": Path("/tmp/p18_ckpts/rgt4kw2u__periodic_step4000.pth"),
}

LAYERS = [3, 6, 9, 11]
N_BOOTSTRAP = 1000
RNG_SEED = 42

ARCFACE_S = {
    "P8A": 9.749250411987305,
    "P18T": 11.998499870300293,
    "P18C": 11.998499870300293,
}


def l2norm_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def load_cached_layer(arm: str, layer: int, n: int) -> np.ndarray:
    p = CACHE_DIR / f"intermediate__{arm}__layer{layer:02d}__n{n}.npz"
    return np.load(p, allow_pickle=True)["features"]


def load_cached_cls(arm: str, n: int) -> np.ndarray:
    p = CACHE_DIR / f"final_cls__{arm}__n{n}.npz"
    return np.load(p, allow_pickle=True)["features"]


def load_head_weight(arm: str) -> np.ndarray:
    """Load the trained head's weight matrix (2x512) for ArcFace cosine scoring."""
    ck = torch.load(str(CKPTS[arm]), map_location="cpu", weights_only=False)
    state = ck.get("state_dict") or ck.get("model_state_dict") or ck
    for key in ("head.weight", "module.head.weight"):
        if key in state:
            return state[key].cpu().numpy().astype(np.float64)
    for k, v in state.items():
        if "head" in k and k.endswith(".weight") and v.ndim == 2 and v.shape[0] == 2:
            return v.cpu().numpy().astype(np.float64)
    raise RuntimeError(f"No head.weight in {arm} ckpt")


def compute_oof_dor_vs_other_lr(feats: np.ndarray, df_valid: pd.DataFrame, seed: int = 0,
                                  norm: str = "stdscaler_subset") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Within-bucket-3 dor-vs-other LR via 5-fold StratifiedKFold.

    norm:
      "stdscaler_subset" (default for CLS) — subtract subset mean, divide by subset std.
        Matches original within_bucket_3_dor_lr semantics for CLS.
      "l2norm_row" (for intermediate layers) — l2-normalize each row.
        Matches original substrate_classifier_direction probe at L3, where
        feats_l3_n = l2norm(feats_l3, axis=1) was used. Avoids overfitting at
        high D / low N because the LR operates on the unit-sphere geometry.
    """
    method = df_valid["method"].astype(str).to_numpy()
    identity = df_valid["identity_key"].astype(str).to_numpy()
    in_bucket3 = np.array([("teams_capture" in m or "teams_flat" in m) for m in method])
    is_dor = np.array([("dor_shkedi" in i) for i in identity])

    if norm == "stdscaler_subset":
        f_sub = feats[in_bucket3]
        mu, sigma = f_sub.mean(axis=0, keepdims=True), f_sub.std(axis=0, keepdims=True) + 1e-12
        f_sub = (f_sub - mu) / sigma
    elif norm == "l2norm_row":
        f_sub = l2norm_rows(feats[in_bucket3].astype(np.float64))
    else:
        raise ValueError(f"unknown norm: {norm}")
    y = is_dor[in_bucket3].astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(f_sub, y):
        clf = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
        clf.fit(f_sub[tr], y[tr])
        oof[te] = clf.predict_proba(f_sub[te])[:, 1]
    bucket3_idx = np.where(in_bucket3)[0]
    return oof, y, bucket3_idx


def bootstrap_auc(oof: np.ndarray, y: np.ndarray, n_iter: int = 1000, seed: int = 42) -> Dict[str, float]:
    """Bootstrap CI on AUC of the given OOF predictions. Resamples (oof, y) with replacement."""
    rng = np.random.RandomState(seed)
    n = len(y)
    aucs = []
    point = float(roc_auc_score(y, oof))
    for _ in range(n_iter):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(y[idx], oof[idx])))
        except ValueError:
            continue
    aucs = np.array(aucs)
    return {
        "point_auc": point,
        "n_valid_iters": int(len(aucs)),
        "ci_lo_2.5": float(np.percentile(aucs, 2.5)),
        "ci_hi_97.5": float(np.percentile(aucs, 97.5)),
        "median": float(np.median(aucs)),
        "std": float(np.std(aucs)),
    }


def paired_bootstrap_delta(oof_a: np.ndarray, oof_b: np.ndarray, y: np.ndarray, n_iter: int = 1000, seed: int = 42) -> Dict[str, float]:
    """Bootstrap CI on AUC_a - AUC_b using THE SAME resampled indices each iter."""
    rng = np.random.RandomState(seed)
    n = len(y)
    deltas = []
    for _ in range(n_iter):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            a = roc_auc_score(y[idx], oof_a[idx])
            b = roc_auc_score(y[idx], oof_b[idx])
            deltas.append(a - b)
        except ValueError:
            continue
    deltas = np.array(deltas)
    return {
        "point_delta": float(roc_auc_score(y, oof_a) - roc_auc_score(y, oof_b)),
        "n_valid_iters": int(len(deltas)),
        "ci_lo_2.5": float(np.percentile(deltas, 2.5)),
        "ci_hi_97.5": float(np.percentile(deltas, 97.5)),
        "median": float(np.median(deltas)),
        "p_two_sided_zero_in_ci": bool(np.percentile(deltas, 2.5) <= 0 <= np.percentile(deltas, 97.5)),
    }


def score_with_arcface_s(feats_cls: np.ndarray, head_w: np.ndarray, s: float) -> np.ndarray:
    """Deployed-style scoring: prob_fake = sigmoid(s * (cos(fake) - cos(real)))."""
    fn = l2norm_rows(feats_cls.astype(np.float64))
    wn = l2norm_rows(head_w)
    cos_logits = fn @ wn.T  # (N, 2) — class 0=real, 1=fake (per existing code)
    margin = cos_logits[:, 1] - cos_logits[:, 0]
    return 1.0 / (1.0 + np.exp(-s * margin))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n = 800
    df = pd.read_csv(SAMPLED_CSV).iloc[:n].reset_index(drop=True)
    df["has_local"] = df["local_path"].apply(lambda p: isinstance(p, str) and Path(p).exists())
    df_valid = df[df["has_local"]].reset_index(drop=True).iloc[:n]
    n = len(df_valid)
    print(f"df_valid: {n} rows")

    method = df_valid["method"].astype(str).to_numpy()
    identity = df_valid["identity_key"].astype(str).to_numpy()
    label = (df_valid["label"].astype(str) == "fake").astype(int).to_numpy()
    is_lockbox = (df_valid["split"].astype(str) == "lockbox").to_numpy()
    is_dor = np.array([("dor_shkedi" in i) for i in identity])
    is_real = label == 0
    in_bucket3 = np.array([("teams_capture" in m or "teams_flat" in m) for m in method])

    print(f"  in_bucket3 (Teams capture content): n={in_bucket3.sum()}")
    print(f"  in_bucket3 + is_dor: n={(in_bucket3 & is_dor).sum()}")
    print(f"  is_lockbox + is_real + is_dor: n={(is_lockbox & is_real & is_dor).sum()}")
    print(f"  is_lockbox + is_real + ~is_dor (other Teams reals): n={(is_lockbox & is_real & ~is_dor).sum()}")

    # Load all features
    cls = {}
    layers_cache: Dict[str, Dict[int, np.ndarray]] = {}
    for arm in CKPTS:
        cls[arm] = load_cached_cls(arm, n)
        print(f"  loaded {arm} CLS: {cls[arm].shape}")
        layers_cache[arm] = {}
        for L in LAYERS:
            layers_cache[arm][L] = load_cached_layer(arm, L, n)
            print(f"  loaded {arm} L{L:02d}: {layers_cache[arm][L].shape}")

    results: Dict = {}

    # ============================================================
    # Diagnostic A: Bootstrap CIs on within-bucket-3 LR AUC at CLS
    # ============================================================
    print("\n=== Diagnostic A: Bootstrap CIs on within-bucket-3 dor-vs-other LR @ CLS ===")
    oof_per_arm: Dict[str, np.ndarray] = {}
    y_ref = None
    for arm in CKPTS:
        oof, y, b3_idx = compute_oof_dor_vs_other_lr(cls[arm], df_valid, seed=RNG_SEED, norm="stdscaler_subset")
        oof_per_arm[arm] = oof
        if y_ref is None:
            y_ref = y
        ci = bootstrap_auc(oof, y, n_iter=N_BOOTSTRAP, seed=RNG_SEED)
        print(f"  {arm}: AUC={ci['point_auc']:.4f}, 95% CI=[{ci['ci_lo_2.5']:.4f}, {ci['ci_hi_97.5']:.4f}], std={ci['std']:.4f}")
        results.setdefault("A_bootstrap_cls", {})[arm] = ci

    # Pairwise delta CIs at CLS
    print("  Paired delta bootstrap @ CLS:")
    for a, b in [("P18T", "P18C"), ("P18T", "P8A"), ("P18C", "P8A")]:
        delta = paired_bootstrap_delta(oof_per_arm[a], oof_per_arm[b], y_ref, n_iter=N_BOOTSTRAP, seed=RNG_SEED)
        zero_in = "ZERO IN CI" if delta["p_two_sided_zero_in_ci"] else "delta != 0"
        print(f"    {a}-{b}: Δ={delta['point_delta']:+.4f}, 95% CI=[{delta['ci_lo_2.5']:+.4f}, {delta['ci_hi_97.5']:+.4f}]  {zero_in}")
        results.setdefault("A_paired_delta_cls", {})[f"{a}-{b}"] = delta

    # ============================================================
    # Diagnostic E: same probe at intermediate layers
    # ============================================================
    print("\n=== Diagnostic E: Within-bucket-3 LR AUC at intermediate layers (l2norm_row) ===")
    for L in LAYERS:
        print(f"  Layer {L:02d}:")
        oof_layer: Dict[str, np.ndarray] = {}
        y_layer = None
        for arm in CKPTS:
            oof, y, _ = compute_oof_dor_vs_other_lr(layers_cache[arm][L], df_valid, seed=RNG_SEED, norm="l2norm_row")
            oof_layer[arm] = oof
            if y_layer is None:
                y_layer = y
            ci = bootstrap_auc(oof, y, n_iter=N_BOOTSTRAP, seed=RNG_SEED)
            print(f"    {arm}: AUC={ci['point_auc']:.4f}, 95% CI=[{ci['ci_lo_2.5']:.4f}, {ci['ci_hi_97.5']:.4f}]")
            results.setdefault(f"E_layer{L:02d}_bootstrap", {})[arm] = ci
        for a, b in [("P18T", "P18C"), ("P18T", "P8A"), ("P18C", "P8A")]:
            delta = paired_bootstrap_delta(oof_layer[a], oof_layer[b], y_layer, n_iter=N_BOOTSTRAP, seed=RNG_SEED)
            zero_in = "ZERO IN CI" if delta["p_two_sided_zero_in_ci"] else "delta != 0"
            print(f"      Δ {a}-{b}={delta['point_delta']:+.4f}, 95% CI=[{delta['ci_lo_2.5']:+.4f}, {delta['ci_hi_97.5']:+.4f}]  {zero_in}")
            results.setdefault(f"E_layer{L:02d}_paired_delta", {})[f"{a}-{b}"] = delta

    # ============================================================
    # Diagnostic C: ArcFace s-scaling on CLS scores
    # ============================================================
    print("\n=== Diagnostic C: Score with ArcFace s-scaling and recompute FPR @ τ=0.92, 0.974 ===")
    head_weights = {arm: load_head_weight(arm) for arm in CKPTS}
    score_per_arm: Dict[str, Dict[str, np.ndarray]] = {}
    for arm in CKPTS:
        s = ARCFACE_S[arm]
        head_w = head_weights[arm]
        # Compare unscaled (sigmoid(margin)) vs scaled (sigmoid(s * margin))
        prob_unscaled = score_with_arcface_s(cls[arm], head_w, s=1.0)
        prob_scaled = score_with_arcface_s(cls[arm], head_w, s=s)
        score_per_arm[arm] = {"unscaled": prob_unscaled, "scaled": prob_scaled, "s": s}
        print(f"  {arm} (s={s:.2f}):")
        for label_str, mask in [("dor_lockbox_real", is_lockbox & is_real & is_dor),
                                  ("nondor_teams_lockbox_real", is_lockbox & is_real & ~is_dor & np.array([m == "teams_real" for m in method]))]:
            for tau in [0.5, 0.92, 0.974]:
                fpr_u = float((prob_unscaled[mask] >= tau).mean()) if mask.sum() > 0 else None
                fpr_s = float((prob_scaled[mask] >= tau).mean()) if mask.sum() > 0 else None
                results.setdefault("C_fpr", {}).setdefault(arm, {}).setdefault(label_str, {})[f"tau_{tau}_unscaled"] = fpr_u
                results["C_fpr"][arm][label_str][f"tau_{tau}_scaled"] = fpr_s
            print(f"    {label_str} (n={int(mask.sum())}):")
            print(f"      mean_prob (unscaled): {float(prob_unscaled[mask].mean()) if mask.sum() else None:.4f}")
            print(f"      mean_prob (scaled s={s:.2f}): {float(prob_scaled[mask].mean()) if mask.sum() else None:.4f}")
            for tau in [0.5, 0.92, 0.974]:
                fpr_u = float((prob_unscaled[mask] >= tau).mean()) if mask.sum() > 0 else None
                fpr_s = float((prob_scaled[mask] >= tau).mean()) if mask.sum() > 0 else None
                print(f"      FPR @ τ={tau}: unscaled={fpr_u:.4f}, scaled={fpr_s:.4f}")

    # ============================================================
    # Diagnostic B: Per-frame paired test on dor lockbox reals
    # ============================================================
    print("\n=== Diagnostic B: Per-frame paired Wilcoxon signed-rank tests ===")
    dor_mask = is_lockbox & is_real & is_dor
    print(f"  n_dor_lockbox_real = {int(dor_mask.sum())}")
    nondor_mask = is_lockbox & is_real & ~is_dor & np.array([m == "teams_real" for m in method])
    print(f"  n_nondor_teams_lockbox_real = {int(nondor_mask.sum())}")

    for use_scaling, scaling_label in [(False, "unscaled"), (True, "scaled")]:
        print(f"  -- {scaling_label} --")
        for mask, mask_name in [(dor_mask, "dor"), (nondor_mask, "nondor_teams")]:
            for a, b in [("P18T", "P18C"), ("P18T", "P8A"), ("P18C", "P8A")]:
                key = "scaled" if use_scaling else "unscaled"
                pa = score_per_arm[a][key][mask]
                pb = score_per_arm[b][key][mask]
                if len(pa) < 5 or np.allclose(pa, pb):
                    continue
                # Wilcoxon signed-rank, two-sided
                try:
                    stat, pval = wilcoxon(pa, pb, zero_method="zsplit", alternative="two-sided")
                except ValueError as e:
                    stat, pval = None, None
                    print(f"    {mask_name} {a}-{b}: wilcoxon err: {e}")
                    continue
                med_diff = float(np.median(pa - pb))
                print(f"    {mask_name} ({a}-{b}): median Δp_fake={med_diff:+.4f}, Wilcoxon W={stat:.1f}, p={pval:.4f}, n={int(mask.sum())}")
                results.setdefault(f"B_paired_{scaling_label}", {}).setdefault(mask_name, {})[f"{a}-{b}"] = {
                    "median_delta": med_diff,
                    "wilcoxon_W": float(stat) if stat is not None else None,
                    "p_two_sided": float(pval) if pval is not None else None,
                    "n": int(mask.sum()),
                }

    # ============================================================
    # Save full results
    # ============================================================
    out = OUTPUT_DIR / "diagnostics_abce_2026-05-02.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
