"""Per-layer label probe split by dev vs lockbox.

The key question: P8A's layer-6 features show label AUC 0.995 on the
800-frame mixed-substrate sample. But that sample is 89% dev (713/800).
The pattern we care about is precisely "high AUC on dev, fails on lockbox".

This script asks: at layer 6, does the label probe trained AND TESTED on
lockbox-only do well? Equivalently, does cross-substrate training (train
on dev, test on lockbox) hold up?

Three tests per layer per ckpt:
  - Within-dev 5-fold CV
  - Within-lockbox 5-fold CV (n=87, small but informative)
  - Train on dev, test on lockbox (the substrate-transfer test)

If the train-on-dev → test-on-lockbox AUC is high at layer 6 but low at
layer 11, then the head is the substrate-overfitting culprit.

If both layer 6 and layer 11 collapse on cross-substrate, then the
backbone itself encodes substrate, and a head retrain alone won't save
us.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
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

CKPTS = ["P8A", "MCLIOEXB"]
LAYERS = [3, 6, 9, 11]
N_SAMPLES = 800


def normalize(F):
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)


def cv_metrics(feats, labels, n_splits=5):
    feats_n = normalize(feats)
    if len(np.unique(labels)) < 2 or labels.sum() < n_splits or (1 - labels).sum() < n_splits:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    oof = np.zeros(len(labels), dtype=np.float64)
    for tr, te in skf.split(feats_n, labels):
        clf = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
        clf.fit(feats_n[tr], labels[tr])
        oof[te] = clf.predict_proba(feats_n[te])[:, 1]
    fpr, tpr, _ = roc_curve(labels, oof)
    out = {"auc": float(roc_auc_score(labels, oof))}
    for tgt in (0.05, 0.10):
        eligible = np.where(fpr <= tgt + 1e-12)[0]
        out[f"rec@fpr_{tgt:.2f}"] = float(tpr[eligible[np.argmax(tpr[eligible])]]) if len(eligible) else 0.0
    return out


def transfer_metrics(train_feats, train_labels, test_feats, test_labels):
    """Train on (train_*), test on (test_*), report AUC and rec@FPR."""
    if len(np.unique(test_labels)) < 2:
        return None
    clf = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs")
    clf.fit(normalize(train_feats), train_labels)
    proba = clf.predict_proba(normalize(test_feats))[:, 1]
    fpr, tpr, _ = roc_curve(test_labels, proba)
    out = {"auc": float(roc_auc_score(test_labels, proba))}
    for tgt in (0.05, 0.10):
        eligible = np.where(fpr <= tgt + 1e-12)[0]
        out[f"rec@fpr_{tgt:.2f}"] = float(tpr[eligible[np.argmax(tpr[eligible])]]) if len(eligible) else 0.0
    return out


def main():
    df = pd.read_csv(SAMPLED_CSV).iloc[:N_SAMPLES].reset_index(drop=True)
    label_int = (df["label"].astype(str) == "fake").astype(int).to_numpy()
    is_lockbox = (df["split"].astype(str) == "lockbox").to_numpy()
    is_dev = (df["split"].astype(str) == "dev").to_numpy()
    print(f"dev={is_dev.sum()}, lockbox={is_lockbox.sum()}, "
          f"dev_real={((~label_int.astype(bool)) & is_dev).sum()}, "
          f"dev_fake={(label_int.astype(bool) & is_dev).sum()}, "
          f"lb_real={((~label_int.astype(bool)) & is_lockbox).sum()}, "
          f"lb_fake={(label_int.astype(bool) & is_lockbox).sum()}")

    rows = []
    for ckpt in CKPTS:
        for layer in LAYERS:
            blob = np.load(CACHE_DIR / f"intermediate__{ckpt}__layer{layer:02d}__n{N_SAMPLES}.npz")
            feats = blob["features"].astype(np.float32)
            valid_idx = blob["valid_idx"].astype(np.int64)
            assert len(feats) == len(valid_idx)
            lbl = label_int[valid_idx]
            is_lb = is_lockbox[valid_idx]
            is_dv = is_dev[valid_idx]

            within_dev = cv_metrics(feats[is_dv], lbl[is_dv])
            within_lb = cv_metrics(feats[is_lb], lbl[is_lb], n_splits=3)  # small N
            transfer = transfer_metrics(feats[is_dv], lbl[is_dv], feats[is_lb], lbl[is_lb])

            row = {
                "ckpt": ckpt,
                "layer": layer,
                "within_dev_AUC": within_dev["auc"] if within_dev else None,
                "within_dev_rec@fpr_0.05": within_dev["rec@fpr_0.05"] if within_dev else None,
                "within_lb_AUC": within_lb["auc"] if within_lb else None,
                "within_lb_rec@fpr_0.05": within_lb["rec@fpr_0.05"] if within_lb else None,
                "transfer_dev_to_lb_AUC": transfer["auc"] if transfer else None,
                "transfer_dev_to_lb_rec@fpr_0.05": transfer["rec@fpr_0.05"] if transfer else None,
            }
            rows.append(row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "per_layer_split_probe.json", "w") as f:
        json.dump(rows, f, indent=2)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "per_layer_split_probe.csv", index=False)

    print()
    print("=" * 110)
    print("PER-LAYER × SPLIT LABEL PROBE")
    print("=" * 110)
    print(f"{'ckpt':<10} {'layer':>5} {'dev_AUC':>9} {'dev@.05':>9} {'lb_AUC':>9} {'lb@.05':>9} {'tr_AUC':>9} {'tr@.05':>9}")
    print("-" * 110)
    for r in rows:
        print(f"{r['ckpt']:<10} {r['layer']:>5d} "
              f"{(r['within_dev_AUC'] or 0):>9.4f} {(r['within_dev_rec@fpr_0.05'] or 0):>9.4f} "
              f"{(r['within_lb_AUC'] or 0):>9.4f} {(r['within_lb_rec@fpr_0.05'] or 0):>9.4f} "
              f"{(r['transfer_dev_to_lb_AUC'] or 0):>9.4f} {(r['transfer_dev_to_lb_rec@fpr_0.05'] or 0):>9.4f}")
    print("=" * 110)
    print(f"  outputs: {OUTPUT_DIR}/per_layer_split_probe.{{json,csv}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
