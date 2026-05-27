"""Phase 3.5 pre-launch smoke gate — frozen-feature method-LR baseline.

Question: do P8A's [CLS] features linearly support 12-class method
discrimination? If macro-OVR AUC >= 0.85, GRL has signal to attack. If < 0.70,
the GRL classifier won't have enough gradient and the packet is pre-doomed.

Uses cached P8A final-layer features (triptych_features__P8A__n800.npz)
because the trained-head GRL operates on the FINAL layer, not L3. The 800
frames are a stratified eval-substrate sample (713 dev + 87 lockbox).

Caveats:
- Sample is from eval bucket, not training distribution. Still a useful
  smoke check because the GRL classifier fires at training time and the
  same encoder is used.
- Some Phase 3 buckets (7, 8, 9) are RESERVED in the 12-bucket map (Phase 2C
  audit) — the eval substrate may not have samples from those.
- Bucket 11 (realpool_real) requires the label-aware wrapper.

Output: macro-OVR AUC + per-bucket counts and AUCs.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import sys
REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
sys.path.insert(0, str(REPO_ROOT))

from data.sources.method_domain_map import (
    METHOD_DOMAIN_NAMES,
    lookup_method_domain_with_label,
)


CACHE_DIR = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
SAMPLED_CSV = (
    REPO_ROOT
    / "analysis"
    / "embedding_triptych_2026-04-30"
    / "outputs"
    / "triptych_p8a_slot2_slot3"
    / "sampled_frames.csv"
)
P8A_FEATS = CACHE_DIR / "triptych_features__P8A__n800.npz"
OUTPUT_DIR = REPO_ROOT / "analysis" / "method_class_audit_2026-05-01" / "outputs"


def map_eval_method_to_bucket(method_str: str, label: int) -> int:
    """Map eval-substrate method strings to 12-bucket map. Eval methods are
    things like 'teams_real', 'deeplive_enhanced', 'teams_capture_cam_test_s35'.
    Use a simple synthetic source-string assignment for the lookup."""
    m = (method_str or "").strip().lower()
    if m == "teams_real":
        return 11  # realpool_real (lockbox real, non-df40 non-external)
    if m.startswith("teams_capture"):
        return 3  # deeplive_teams (Teams passthrough; matches dor_shkedi cluster)
    if m == "teams_flat" or m.startswith("teams_flat"):
        return 3
    if m == "deeplive_enhanced":
        return 2  # deeplive_enhanced — Phase 1A axis
    if m.startswith("deeplive_"):
        # Try the canonical lookup
        return lookup_method_domain_with_label(
            method=f"deeplive_{m.split('_', 1)[1] if '_' in m else m}",
            source="deeplive",
            label=label,
        )
    # Try canonical lookup for everything else
    return lookup_method_domain_with_label(method=m, source="visomaster", label=label)


def main():
    df = pd.read_csv(SAMPLED_CSV).iloc[:800].reset_index(drop=True)
    blob = np.load(P8A_FEATS)
    feats = blob["features"].astype(np.float64)
    valid_idx = blob["valid_idx"].astype(np.int64) if "valid_idx" in blob.files else np.arange(len(feats))
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    label = (df_valid["label"].astype(str) == "fake").astype(int).to_numpy()
    method = df_valid["method"].astype(str).to_numpy()

    # Map every frame to a 12-bucket label
    domain_label = np.array([
        map_eval_method_to_bucket(method[i], label[i]) for i in range(len(method))
    ], dtype=np.int64)

    # Counts per bucket
    bucket_counts = Counter(domain_label.tolist())
    print("Bucket counts (12-class):")
    for b in sorted(bucket_counts.keys()):
        print(f"  bucket {b:>2} ({METHOD_DOMAIN_NAMES.get(b, '?'):<25}): {bucket_counts[b]:>4} frames")

    # Drop buckets with < 5 samples (too small for meaningful AUC)
    keep_mask = np.array([bucket_counts[b] >= 5 for b in domain_label])
    f = feats[keep_mask]
    y = domain_label[keep_mask]
    kept_buckets = sorted(set(y.tolist()))
    print(f"\nKeeping {keep_mask.sum()}/{len(domain_label)} frames across {len(kept_buckets)} buckets with ≥ 5 samples")
    print(f"  Kept buckets: {kept_buckets}")

    # Standardize features (matches the canonical domain_probe.py)
    scaler = StandardScaler()
    f_norm = scaler.fit_transform(f)

    # 5-fold stratified CV with multi-class LR (one-vs-rest)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(y), len(kept_buckets)))
    bucket_to_idx = {b: i for i, b in enumerate(kept_buckets)}

    for fold, (tr, te) in enumerate(skf.split(f_norm, y)):
        clf = LogisticRegression(
            C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs",
            multi_class="ovr",
        )
        clf.fit(f_norm[tr], y[tr])
        # Map class labels to column positions
        cls_to_col = {c: i for i, c in enumerate(clf.classes_)}
        proba = clf.predict_proba(f_norm[te])
        for c, col in cls_to_col.items():
            oof_proba[te, bucket_to_idx[c]] = proba[:, col]
        print(f"  fold {fold} done (train {len(tr)}, test {len(te)})")

    # Macro-OVR AUC across kept buckets
    per_bucket_auc = {}
    for b in kept_buckets:
        i = bucket_to_idx[b]
        y_bin = (y == b).astype(int)
        if len(np.unique(y_bin)) < 2:
            per_bucket_auc[b] = None
            continue
        auc = float(roc_auc_score(y_bin, oof_proba[:, i]))
        per_bucket_auc[b] = auc
        print(f"  bucket {b:>2} ({METHOD_DOMAIN_NAMES.get(b, '?'):<25}): AUC={auc:.4f}  n_pos={int(y_bin.sum())}")

    valid_aucs = [v for v in per_bucket_auc.values() if v is not None]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else None

    print()
    print("=" * 80)
    print(f"VERDICT — Phase 3.5 method-LR baseline")
    print("=" * 80)
    if macro_auc is None:
        print("  MACRO-OVR AUC: NA (insufficient data)")
        print("  → SMOKE GATE FAILED — cannot launch P18 without signal evidence.")
        verdict = "FAILED"
    elif macro_auc >= 0.85:
        print(f"  MACRO-OVR AUC: {macro_auc:.4f}")
        print("  → SMOKE GATE PASSED — GRL has signal to attack at the 12-class level.")
        verdict = "PASSED"
    elif macro_auc >= 0.70:
        print(f"  MACRO-OVR AUC: {macro_auc:.4f}")
        print("  → SMOKE GATE MARGINAL — signal exists but borderline.")
        verdict = "MARGINAL"
    else:
        print(f"  MACRO-OVR AUC: {macro_auc:.4f}")
        print("  → SMOKE GATE FAILED — GRL has insufficient signal at this granularity.")
        verdict = "FAILED"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "method_lr_baseline_2026-05-01.json", "w") as fjson:
        json.dump({
            "n_samples_used": int(keep_mask.sum()),
            "n_samples_total": int(len(domain_label)),
            "kept_buckets": kept_buckets,
            "bucket_counts": {str(k): int(v) for k, v in bucket_counts.items()},
            "per_bucket_auc": {str(k): v for k, v in per_bucket_auc.items()},
            "macro_ovr_auc": macro_auc,
            "verdict": verdict,
            "threshold_pass": 0.85,
            "threshold_marginal": 0.70,
        }, fjson, indent=2)
    print(f"\n  output: {OUTPUT_DIR / 'method_lr_baseline_2026-05-01.json'}")
    return 0 if verdict == "PASSED" else (1 if verdict == "MARGINAL" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
