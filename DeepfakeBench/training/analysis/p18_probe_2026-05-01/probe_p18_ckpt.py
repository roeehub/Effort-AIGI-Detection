"""Probe a P18 checkpoint — extracts decision direction + computes alignment
with Phase 1A substrate axes + 12-class GRL bite metric.

Usage:
    python3 analysis/p18_probe_2026-05-01/probe_p18_ckpt.py \
        --ckpt <local_path_to_pth> \
        --label <run_id_or_step> \
        --arm <treatment|control>

Adds one row per probe to:
    analysis/p18_probe_2026-05-01/outputs/trajectory.json (append-mode)
    analysis/p18_probe_2026-05-01/outputs/trajectory.csv  (append-mode)

What it does:

1. Loads the ckpt's `head.weight` (and `head.s` for ArcFace) and computes the
   decision direction in normalized feature space (matches the substrate-
   classifier-direction probe from Phase 1A).

2. Reuses the cached final-layer P8A features (triptych_features__P8A__n800.npz,
   500-dim if it's the P8A backbone). Note: P18 trains the FULL backbone
   (P8A's unfreeze recipe), so by the end-of-run the backbone shifts. The
   probe direction we compute IS in the final-layer space, but cosines vs
   substrate-classifier directions computed on P8A features are an
   approximation. For trajectory purposes (early ckpts close to P8A) the
   approximation is tight; late ckpts may need a re-extracted feature cache
   (deferred — heavier).

3. Computes:
   - cos(P18_head_dir, fresh_LR_fake_vs_real_dir) — does the head direction
     contain the substrate-invariant signal?
   - cos(P18_head_dir, is_dor_shkedi_dir) — does it still align with the
     identity-cluster axis Phase 1A pinpointed? Lower = better.
   - cos(P18_head_dir, is_deeplive_enhanced_dir) — same for the other axis.
   - cos(P18_head_dir, is_webcam_dir) and is_phone_screen — control axes.
   - Domain-confusion AUC analog: 12-class LR on the P8A cached features
     predicting method-domain. If P18's training pushed encoder features in
     a way that still leaves [CLS] linearly separable at 0.998, GRL didn't
     bite. (Note: this is a P8A-feature-space proxy. The TRUE domain probe
     needs to re-extract features with the P18 backbone on the same 800
     frames; deferred to Phase 4.4 for final ckpts.)

4. Reports per-checkpoint verdict against the α/β/γ thresholds:
   - α: cos(head, dor_shkedi) ≤ 0.05 AND domain-AUC drops < 0.85.
   - β: head moved off dor_shkedi axis but domain-AUC stayed ~0.95.
   - γ: head still aligned with dor_shkedi (cos ≥ 0.10) and domain-AUC
     stayed ~0.998. → 12-class GRL also wrong axis.

Designed to be incremental: each invocation appends a row. Final synthesis
script (`synthesize_p18_trajectory.py`) reads the json/csv and produces the
final verdict.

CPU only. n_jobs=1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))

CACHE_DIR = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
SAMPLED_CSV = (
    REPO_ROOT / "analysis" / "embedding_triptych_2026-04-30" / "outputs"
    / "triptych_p8a_slot2_slot3" / "sampled_frames.csv"
)
P8A_FEATS = CACHE_DIR / "triptych_features__P8A__n800.npz"

OUTPUT_DIR = REPO_ROOT / "analysis" / "p18_probe_2026-05-01" / "outputs"


def l2norm(x, axis):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def cos(a, b):
    if a is None or b is None:
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return None
    return float(np.dot(a, b) / (na * nb))


def extract_head_direction(ckpt_path: Path) -> Tuple[np.ndarray, str, dict]:
    """Extract the decision direction (unit-vector difference between fake and
    real class weight rows) from a saved EffortDetector ckpt. Returns
    (direction_in_normalized_feature_space, recipe_name, metadata)."""
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ck.get("state_dict") or ck.get("model_state_dict") or ck
    config = ck.get("model_config", {}) or ck.get("config", {})

    # Find head.weight (may be under "head.weight" or "module.head.weight").
    head_w = None
    for key in ("head.weight", "module.head.weight"):
        if key in state:
            head_w = state[key].cpu().numpy().astype(np.float64)
            break
    if head_w is None:
        # Try any *head*.weight key
        for key, val in state.items():
            if "head" in key and key.endswith(".weight") and val.ndim == 2 and val.shape[0] == 2:
                head_w = val.cpu().numpy().astype(np.float64)
                break
    if head_w is None:
        raise RuntimeError(f"Could not locate 2-class head.weight in {ckpt_path}")

    use_arcface = bool(config.get("use_arcface_head", True))
    recipe = "ARCFACE" if use_arcface else "LINEAR"
    if use_arcface:
        direction = l2norm(head_w[1:2], axis=1)[0] - l2norm(head_w[0:1], axis=1)[0]
    else:
        direction = head_w[1] - head_w[0]

    meta = {
        "recipe": recipe,
        "head_weight_shape": list(head_w.shape),
        "use_quality_domain_head": bool(config.get("use_quality_domain_head", False)),
        "quality_domain_count": int(config.get("quality_domain_count", 4)),
    }
    return direction, recipe, meta


def fit_substrate_lrs(feats_n: np.ndarray, df_valid: pd.DataFrame, label: np.ndarray):
    """Fit substrate-classifier directions on normalized P8A features.
    Returns dict of axis_name → direction_vector."""
    from sklearn.linear_model import LogisticRegression

    capture = df_valid["clip_capture_mode"].astype(str).to_numpy()
    method = df_valid["method"].astype(str).to_numpy()
    identity = df_valid["identity_key"].astype(str).to_numpy()
    is_lockbox = (df_valid["split"].astype(str) == "lockbox").to_numpy()
    is_dev = ~is_lockbox

    axes = {
        "is_lockbox": is_lockbox,
        "is_webcam": capture == "webcam",
        "is_phone_screen": capture == "phone_screen",
        "is_normal_photo": capture == "normal_photo",
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

    # fresh-LR fake-vs-real direction (substrate-INVARIANT reference)
    lr_fr = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
    lr_fr.fit(feats_n[is_dev], label[is_dev])
    directions["fresh_LR_fake_vs_real"] = lr_fr.coef_.flatten().astype(np.float64)

    return directions


def fit_method_12class_lr_macro_auc(
    feats_n: np.ndarray, df_valid: pd.DataFrame, label: np.ndarray
) -> Tuple[Optional[float], dict]:
    """12-class method-domain LR on cached P8A features. Returns
    (macro_ovr_auc_proxy, per_bucket_n)."""
    sys.path.insert(0, str(REPO_ROOT))
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
        [to_bucket(method[i], int(label[i])) for i in range(len(method))],
        dtype=np.int64,
    )

    from collections import Counter
    bucket_counts = Counter(domain_label.tolist())
    keep_mask = np.array([bucket_counts[b] >= 5 for b in domain_label])
    f = feats_n[keep_mask]
    y = domain_label[keep_mask]
    kept_buckets = sorted(set(y.tolist()))

    if len(kept_buckets) < 2:
        return None, dict(bucket_counts)

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(y), len(kept_buckets)))
    bucket_to_idx = {b: i for i, b in enumerate(kept_buckets)}
    for tr, te in skf.split(f, y):
        clf = LogisticRegression(
            C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs", multi_class="ovr"
        )
        clf.fit(f[tr], y[tr])
        cls_to_col = {c: i for i, c in enumerate(clf.classes_)}
        proba = clf.predict_proba(f[te])
        for c, col in cls_to_col.items():
            oof_proba[te, bucket_to_idx[c]] = proba[:, col]
    aucs = []
    for b in kept_buckets:
        i = bucket_to_idx[b]
        y_bin = (y == b).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        aucs.append(float(roc_auc_score(y_bin, oof_proba[:, i])))
    macro_auc = float(np.mean(aucs)) if aucs else None
    return macro_auc, {str(k): int(v) for k, v in bucket_counts.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Local path to .pth")
    parser.add_argument("--label", required=True, help="Identifier (run_id + step)")
    parser.add_argument("--arm", required=True, choices=["treatment", "control", "p8a", "other"])
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Ckpt not found: {ckpt_path}")

    # Load eval-substrate features (P8A backbone, final [CLS])
    blob = np.load(P8A_FEATS)
    feats = blob["features"].astype(np.float64)
    valid_idx = (
        blob["valid_idx"].astype(np.int64)
        if "valid_idx" in blob.files else np.arange(len(feats))
    )
    df = pd.read_csv(SAMPLED_CSV).iloc[: 800].reset_index(drop=True)
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    label = (df_valid["label"].astype(str) == "fake").astype(int).to_numpy()

    feats_n = l2norm(feats, axis=1)

    # Substrate-classifier directions (in P8A normalized feature space)
    print(f"Fitting substrate-classifier directions on {len(feats_n)} P8A features...")
    substrate_dirs = fit_substrate_lrs(feats_n, df_valid, label)

    # Extract P18 ckpt's decision direction
    print(f"Extracting head direction from {ckpt_path.name}...")
    p18_dir, recipe, meta = extract_head_direction(ckpt_path)
    print(f"  recipe: {recipe}")
    print(f"  head.weight shape: {meta['head_weight_shape']}")
    print(f"  use_quality_domain_head: {meta['use_quality_domain_head']}")
    print(f"  quality_domain_count: {meta['quality_domain_count']}")

    # Compute cosines
    cosines = {}
    for name, dir_vec in substrate_dirs.items():
        cosines[name] = cos(p18_dir, dir_vec)

    # 12-class method LR macro-OVR AUC (proxy for "did GRL bite at the
    # encoder's [CLS] manifold?" — but on P8A features, so the answer for
    # late P18 ckpts is approximate)
    print("Computing 12-class method-LR proxy AUC...")
    macro_auc, bucket_counts = fit_method_12class_lr_macro_auc(
        feats_n, df_valid, label
    )

    print(f"\n=== {args.arm.upper()} ckpt {args.label} ===")
    print(f"  cos(P18_head_dir, fresh_LR_fake_vs_real)  = {cosines['fresh_LR_fake_vs_real']:+.4f}"
          if cosines['fresh_LR_fake_vs_real'] is not None else "  cos vs fresh_LR: NA")
    print(f"  cos(P18_head_dir, is_dor_shkedi)          = {cosines['is_dor_shkedi']:+.4f}"
          if cosines['is_dor_shkedi'] is not None else "  cos vs is_dor_shkedi: NA")
    print(f"  cos(P18_head_dir, is_deeplive_enhanced)   = {cosines['is_deeplive_enhanced']:+.4f}"
          if cosines['is_deeplive_enhanced'] is not None else "  cos vs is_deeplive_enhanced: NA")
    print(f"  cos(P18_head_dir, is_webcam)              = {cosines['is_webcam']:+.4f}"
          if cosines['is_webcam'] is not None else "")
    print(f"  cos(P18_head_dir, is_lockbox)             = {cosines['is_lockbox']:+.4f}"
          if cosines['is_lockbox'] is not None else "")
    print(f"  12-class method-LR macro-AUC (P8A feats)  = {macro_auc:.4f}"
          if macro_auc is not None else "  macro AUC: NA")

    # Verdict per α/β/γ thresholds
    cos_dor = abs(cosines.get("is_dor_shkedi") or 0.0)
    if macro_auc is None:
        verdict = "INCONCLUSIVE"
    elif cos_dor <= 0.05 and macro_auc < 0.85:
        verdict = "α (passes)"
    elif cos_dor <= 0.05 and macro_auc < 0.95:
        verdict = "β (partial — head moved but domain-AUC stayed high)"
    elif cos_dor <= 0.05:
        verdict = "β (partial — head moved but encoder still discriminates)"
    elif macro_auc < 0.85:
        verdict = "β (partial — encoder flatter but head still on dor axis)"
    else:
        verdict = "γ (no bite)"

    print(f"  → preliminary verdict: {verdict}")

    # Append to trajectory CSV/JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "label": args.label,
        "arm": args.arm,
        "ckpt_path": str(ckpt_path),
        **{f"cos_{k}": v for k, v in cosines.items()},
        "method_lr_macro_auc": macro_auc,
        "recipe": recipe,
        "use_quality_domain_head": meta["use_quality_domain_head"],
        "quality_domain_count": meta["quality_domain_count"],
        "verdict_preliminary": verdict,
    }

    # Append to JSON (read-modify-write)
    json_path = OUTPUT_DIR / "trajectory.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = {"rows": []}
    data["rows"].append(row)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    # Append to CSV
    csv_path = OUTPUT_DIR / "trajectory.csv"
    df_row = pd.DataFrame([row])
    if csv_path.exists():
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, index=False)
    print(f"\n  appended to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
