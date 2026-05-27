"""Finish Move 1 — within-bucket fake/real AUC + cross-bucket transfer.

The Move 1 sub-agent's run_probe.py wrote the cached features but stopped
mid-run before writing the final results. This script picks up from those
cached files and finishes the analysis the original probe was meant to do.

Inputs:
  - train_bucket_feats.npz (from run_probe.py)
  - train_bucket_samples.csv (from run_probe.py; columns include sample_id,
    swap_model, side, frame_index, gcs_uri, local_path)
  - analysis/_features_cache_2026-04-30/triptych_features__P8A__n800.npz
    (eval-bucket features)
  - analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/
    sampled_frames.csv (eval-bucket metadata)

Outputs:
  - probe_results.json
  - probe_results.csv

Sklearn n_jobs=1 enforced.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
PROBE_DIR = REPO_ROOT / "analysis" / "move1_frozen_probe_2026-05-01"
OUTPUT_DIR = PROBE_DIR / "outputs"

EVAL_FEATS_NPZ = REPO_ROOT / "analysis" / "_features_cache_2026-04-30" / "triptych_features__P8A__n800.npz"
SAMPLED_CSV = (
    REPO_ROOT / "analysis" / "embedding_triptych_2026-04-30" / "outputs"
    / "triptych_p8a_slot2_slot3" / "sampled_frames.csv"
)


def auc_safe(y, s):
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def main():
    # === Load training-bucket features ===
    train_npz = OUTPUT_DIR / "train_bucket_feats.npz"
    train_csv = OUTPUT_DIR / "train_bucket_samples.csv"
    train_blob = np.load(train_npz)
    train_feats = train_blob["features"].astype(np.float32)
    train_valid = train_blob["valid_idx"].astype(np.int64) if "valid_idx" in train_blob.files else None
    train_meta_full = pd.read_csv(train_csv)
    if train_valid is not None and len(train_valid) != len(train_meta_full):
        train_meta = train_meta_full.iloc[train_valid].reset_index(drop=True)
    else:
        train_meta = train_meta_full
    print(f"train: feats={train_feats.shape}  meta={train_meta.shape}")
    print(f"  train side counts: {Counter(train_meta['side'].astype(str))}")
    print(f"  train swap_models: {dict(Counter(train_meta['swap_model'].astype(str)))}")

    # === Load eval-bucket features ===
    eval_blob = np.load(EVAL_FEATS_NPZ)
    eval_feats = eval_blob["features"].astype(np.float32)
    eval_valid = eval_blob["valid_idx"].astype(np.int64) if "valid_idx" in eval_blob.files else np.arange(len(eval_feats))
    eval_meta_full = pd.read_csv(SAMPLED_CSV).iloc[: len(eval_blob["features"])].reset_index(drop=True)
    eval_meta = eval_meta_full.iloc[eval_valid].reset_index(drop=True)
    eval_label = (eval_meta["label"].astype(str) == "fake").astype(int).to_numpy()
    eval_method = eval_meta["method"].astype(str).to_numpy()
    eval_identity = eval_meta["identity_key"].astype(str).to_numpy()

    print(f"eval: feats={eval_feats.shape}  meta={eval_meta.shape}")
    print(f"  eval label counts: {Counter(eval_label)}")
    print(f"  eval method top: {dict(Counter(eval_method).most_common(5))}")

    # ---- Within eval-bucket fake/real AUC (no held-out; in-sample) ----
    # eval-bucket fake/real AUC with grouped split by identity to avoid leakage
    print("\n=== Within EVAL-bucket fake/real AUC (grouped by identity) ===")
    feats_n = eval_feats / (np.linalg.norm(eval_feats, axis=1, keepdims=True) + 1e-12)
    gkf = GroupKFold(n_splits=5)
    eval_oof = np.zeros(len(eval_label), dtype=np.float64)
    for fold_idx, (tr, te) in enumerate(gkf.split(feats_n, eval_label, groups=eval_identity)):
        if len(np.unique(eval_label[tr])) < 2:
            continue
        clf = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
        clf.fit(feats_n[tr], eval_label[tr])
        eval_oof[te] = clf.predict_proba(feats_n[te])[:, 1]
    eval_within_auc = auc_safe(eval_label, eval_oof)
    print(f"  eval-bucket within fake/real OOF AUC = {eval_within_auc:.4f}")

    # ---- Within train-bucket fake/real AUC (grouped by sample_id) ----
    train_label = (train_meta["side"].astype(str) == "fake").astype(int).to_numpy()
    # Use sample_id as identity proxy; if same identity has both real and fake frames, we group them.
    train_groups = train_meta["sample_id"].astype(str).to_numpy()
    print("\n=== Within TRAIN-bucket fake/real AUC (grouped by sample_id) ===")
    train_n = train_feats / (np.linalg.norm(train_feats, axis=1, keepdims=True) + 1e-12)
    gkf2 = GroupKFold(n_splits=5)
    tr_oof = np.zeros(len(train_label), dtype=np.float64)
    used_any = False
    for tr, te in gkf2.split(train_n, train_label, groups=train_groups):
        if len(np.unique(train_label[tr])) < 2 or len(np.unique(train_label[te])) < 2:
            continue
        clf = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
        clf.fit(train_n[tr], train_label[tr])
        tr_oof[te] = clf.predict_proba(train_n[te])[:, 1]
        used_any = True
    if used_any and len(np.unique(train_label)) == 2:
        train_within_auc = auc_safe(train_label, tr_oof)
        print(f"  train-bucket within fake/real OOF AUC = {train_within_auc:.4f}")
    else:
        # Fallback: random shuffle split (less ideal, but gives a number)
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        tr_oof = np.zeros(len(train_label), dtype=np.float64)
        for tr, te in skf.split(train_n, train_label):
            clf = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
            clf.fit(train_n[tr], train_label[tr])
            tr_oof[te] = clf.predict_proba(train_n[te])[:, 1]
        train_within_auc = auc_safe(train_label, tr_oof)
        print(f"  train-bucket within fake/real OOF AUC (stratified KFold fallback) = {train_within_auc:.4f}")

    # ---- Cross-bucket transfer: train on train-bucket, test on eval-bucket ----
    print("\n=== Cross-bucket transfer: train on TRAIN-bucket viso, test on EVAL-bucket ===")
    # Limit eval samples to those that look like 'viso-style' (deeplive_enhanced + teams_capture? -- conservative: all)
    clf_x = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
    clf_x.fit(train_n, train_label)
    eval_xfer_scores = clf_x.predict_proba(feats_n)[:, 1]
    eval_xfer_auc = auc_safe(eval_label, eval_xfer_scores)
    print(f"  cross-bucket transfer AUC (train→eval, all eval) = {eval_xfer_auc:.4f}")

    # Per-eval-method cross-bucket transfer AUC (interesting per-method)
    print("\n  per-eval-method transfer AUC:")
    method_aucs = {}
    fake_methods_in_eval = set()
    for m in np.unique(eval_method):
        mask = eval_method == m
        if mask.sum() < 4:
            continue
        # We need both real and fake to compute AUC; skip pure-real or pure-fake methods
        sub_label = eval_label[mask]
        if len(np.unique(sub_label)) < 2:
            continue
        sub_auc = auc_safe(sub_label, eval_xfer_scores[mask])
        method_aucs[m] = (int(mask.sum()), sub_auc)
        if sub_label.sum() > 0:
            fake_methods_in_eval.add(m)
    for m, (n, a) in sorted(method_aucs.items(), key=lambda kv: -kv[1][0]):
        print(f"    {m:<40} n={n:>3}  AUC={a:.4f}" if a is not None else f"    {m:<40} n={n:>3}  AUC=NA")

    # Also compute pooled real-vs-fake using ONLY 'teams_real' as real and method-by-method as fake
    pooled_results = {}
    real_mask = eval_method == "teams_real"
    real_scores = eval_xfer_scores[real_mask]
    real_n = int(real_mask.sum())
    print(f"\n  pooled vs teams_real (n_real={real_n}):")
    for m in sorted({mm for mm in np.unique(eval_method) if mm != "teams_real"}):
        m_mask = eval_method == m
        if m_mask.sum() < 5:
            continue
        # Treat method m as 'fake' (label=1), teams_real as 'real' (label=0)
        y = np.concatenate([np.zeros(real_n), np.ones(int(m_mask.sum()))])
        s = np.concatenate([real_scores, eval_xfer_scores[m_mask]])
        if len(np.unique(y)) < 2:
            continue
        a = auc_safe(y, s)
        pooled_results[m] = (int(m_mask.sum()), a)

    # Final results dict
    results = {
        "n_train": int(len(train_feats)),
        "n_eval": int(len(eval_feats)),
        "train_label_counts": {str(k): int(v) for k, v in Counter(train_label).items()},
        "eval_label_counts": {str(k): int(v) for k, v in Counter(eval_label).items()},
        "train_swap_models": {str(k): int(v) for k, v in Counter(train_meta["swap_model"]).items()},
        "bucket_discrimination_auc_mean": 0.9160,    # from log
        "bucket_discrimination_auc_std": 0.0192,     # from log
        "identity_only_control_auc_kept32": 0.9851,  # from log (n_classes_kept=32)
        "identity_only_control_auc_kept33": 0.9872,  # from log (min_per_class=2)
        "eval_within_fake_real_oof_auc_grouped": eval_within_auc,
        "train_within_fake_real_oof_auc_grouped": train_within_auc,
        "cross_bucket_transfer_auc_train_to_eval": eval_xfer_auc,
        "per_eval_method_transfer_auc": {m: {"n": n, "auc": a} for m, (n, a) in method_aucs.items()},
        "pooled_vs_teams_real": {m: {"n_fake": n, "auc": a} for m, (n, a) in pooled_results.items()},
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame([
        {"k": k, "v": v} for k, v in results.items()
        if not isinstance(v, dict)
    ]).to_csv(OUTPUT_DIR / "probe_results.csv", index=False)

    print()
    print("=" * 80)
    print("VERDICT (Move 1)")
    print("=" * 80)
    print(f"  Bucket discrimination AUC (grouped):       {results['bucket_discrimination_auc_mean']:.4f}  ± {results['bucket_discrimination_auc_std']:.4f}")
    print(f"  Identity-only control AUC (multi-class):  {results['identity_only_control_auc_kept33']:.4f}")
    print(f"  Eval within fake/real OOF AUC (grouped):  {eval_within_auc:.4f}" if eval_within_auc is not None else "  Eval within fake/real OOF AUC: NA")
    print(f"  Train within fake/real OOF AUC (grouped): {train_within_auc:.4f}" if train_within_auc is not None else "  Train within fake/real OOF AUC: NA")
    print(f"  Cross-bucket transfer AUC (train→eval):   {eval_xfer_auc:.4f}" if eval_xfer_auc is not None else "  Cross-bucket transfer AUC: NA")
    print()
    print("  Per PLAN.md §9 P1 outcome ladder:")
    print(f"    - identity_control_auc {results['identity_only_control_auc_kept33']:.3f} >= 0.70 -> AMBIGUOUS")
    print("    - probe results identity-confounded; do NOT use as P14_DATA_FIX gate")
    print(f"  outputs: {OUTPUT_DIR / 'probe_results.json'}")


if __name__ == "__main__":
    main()
