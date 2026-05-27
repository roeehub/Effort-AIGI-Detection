"""Per-layer domain probe — extends intermediate_layer_probe.

The intermediate-layer probe (this directory) showed that:
  - layer-0 to layer-9 features are nearly identical between P8A and mclioexb
    (cos >= 0.995); only layer 11 ([CLS]) shows ~5% drift.
  - Label separability PEAKS at layer 6 (rec@5%FPR ~ 0.97) and DECAYS at
    layer 11 (rec@5%FPR ~ 0.83).

The natural follow-up: at which layer does the capture/source domain
information become linearly decodable? If domain decodability is high at
layer 6 too, then the shortcut is structural to the backbone and a head
swap won't help. If domain decodability ramps up sharply between layer 6
and layer 11, then the head is where the shortcut lives — and a clean
head retrain on layer-6 features can break the shortcut while preserving
98+% of label info.

This script reads the cached intermediate-layer features (no model
forwards, no GPU) and runs a 5-fold-CV LR domain probe per layer for
each ckpt.

Output:
  analysis/intermediate_layer_probe_2026-04-30/outputs/per_layer_domain_probe.json
  analysis/intermediate_layer_probe_2026-04-30/outputs/per_layer_domain_probe.csv

Usage:
    python3 analysis/intermediate_layer_probe_2026-04-30/per_layer_domain_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

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
LAYERS = [0, 3, 6, 9, 11]
N_SAMPLES = 800


def load_layer_features(ckpt: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    blob = np.load(CACHE_DIR / f"intermediate__{ckpt}__layer{layer:02d}__n{N_SAMPLES}.npz")
    return blob["features"].astype(np.float32), blob["valid_idx"].astype(np.int64)


def domain_probe(feats: np.ndarray, labels_int: np.ndarray) -> dict:
    feats_n = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    n_classes = int(labels_int.max() + 1)
    oof_proba = np.zeros((len(labels_int), n_classes), dtype=np.float64)
    for tr, te in skf.split(feats_n, labels_int):
        clf = LogisticRegression(C=1.0, max_iter=5000, n_jobs=1, solver="lbfgs",
                                  multi_class="multinomial")
        clf.fit(feats_n[tr], labels_int[tr])
        oof_proba[te] = clf.predict_proba(feats_n[te])
    # macro OvR AUC
    aucs = []
    for c in range(n_classes):
        y = (labels_int == c).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        aucs.append(roc_auc_score(y, oof_proba[:, c]))
    return {"macro_ovr_auc": float(np.mean(aucs)), "per_class_auc": [float(a) for a in aucs]}


def main():
    df = pd.read_csv(SAMPLED_CSV).iloc[:N_SAMPLES].reset_index(drop=True)
    print(f"Loaded {len(df)} frames")

    capture_le = LabelEncoder().fit(df["clip_capture_mode"].astype(str))
    capture_int = capture_le.transform(df["clip_capture_mode"].astype(str))
    print(f"capture_mode classes: {dict(zip(capture_le.classes_, np.bincount(capture_int)))}")

    label_int = (df["label"].astype(str) == "fake").astype(int).to_numpy()

    rows = []
    for ckpt in CKPTS:
        for layer in LAYERS:
            feats, valid_idx = load_layer_features(ckpt, layer)
            cap_aligned = capture_int[valid_idx]
            lbl_aligned = label_int[valid_idx]

            cap_res = domain_probe(feats, cap_aligned)
            lbl_res = domain_probe(feats, lbl_aligned)

            rows.append({
                "ckpt": ckpt,
                "layer": layer,
                "n_samples": len(feats),
                "capture_mode_macro_auc": cap_res["macro_ovr_auc"],
                "label_auc": lbl_res["macro_ovr_auc"],
                "capture_per_class_auc": cap_res["per_class_auc"],
            })
            print(f"[{ckpt}] layer {layer:>2}: capture_macro_AUC={cap_res['macro_ovr_auc']:.4f} | label_AUC={lbl_res['macro_ovr_auc']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUTPUT_DIR / "per_layer_domain_probe.json"
    out_csv = OUTPUT_DIR / "per_layer_domain_probe.csv"
    with open(out_json, "w") as f:
        json.dump({"capture_classes": capture_le.classes_.tolist(), "rows": rows}, f, indent=2)
    pd.DataFrame([{k: v for k, v in r.items() if k != "capture_per_class_auc"} for r in rows]).to_csv(out_csv, index=False)

    # Pretty table.
    print()
    print("=" * 92)
    print("PER-LAYER DOMAIN PROBE  (capture_mode 5-class, 5-fold CV linear LR)")
    print("=" * 92)
    print(f"{'ckpt':<10} {'layer':>6} {'capture_macro_AUC':>20} {'label_AUC':>12}")
    print("-" * 92)
    for r in rows:
        print(f"{r['ckpt']:<10} {r['layer']:>6d} {r['capture_mode_macro_auc']:>20.4f} {r['label_auc']:>12.4f}")
    print("=" * 92)
    print(f"  outputs: {out_json}, {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
