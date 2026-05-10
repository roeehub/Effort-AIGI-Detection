"""Extend forgery atlas with MCLIOEXB (face_scale_jitter winner) + P18T/P18C (GRL variants).

Critical for T4 thesis testing:
  - If MCLIOEXB has LOWER face_size_high AUC at L11 vs P8A → face_scale_jitter
    weakens the face-size shortcut at the encoder level → T4 hypothesis supported.
  - If MCLIOEXB has SIMILAR face_size_high AUC → jitter doesn't weaken the
    shortcut at the encoder level → T4 lever is at a different mechanism.

P18T/P18C inform the multi-axis-GRL hypothesis:
  - If P18T has lower method-cluster AUC at L11 vs P8A → GRL bites encoder.
  - If similar → GRL was head-level only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Note: features are at PRIOR_PERLAYER_CACHE = analysis/_features_cache_2026-04-30/
# but the run_analyses.py loader expects them in iq_perlayer_probe_2026-05-08/_cache/.
# Need to handle this.

PRIOR_CACHE = REPO_ROOT / "analysis" / "_features_cache_2026-04-30"
LOCAL_CACHE = REPO_ROOT / "analysis" / "iq_perlayer_probe_2026-05-08" / "_cache"

import shutil

EXTRAS = ["MCLIOEXB", "P18T", "P18C"]
LAYERS = [0, 3, 6, 9, 11]

# Copy missing caches from prior to local
for label in EXTRAS:
    for layer in LAYERS:
        src = PRIOR_CACHE / f"intermediate__{label}__layer{layer:02d}__n800.npz"
        dst = LOCAL_CACHE / f"intermediate__{label}__layer{layer:02d}__n800.npz"
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
            print(f"Copied {src.name}")
        elif not src.exists():
            print(f"MISSING: {src.name}")

sys.path.insert(0, str(REPO_ROOT / "analysis" / "cpu_diagnostics_2026-05-09"))
from run_analyses import build_panel, load_features  # noqa

OUT = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-10/outputs"
EXISTING_ATLAS = OUT / "forgery_signal_atlas_extended.csv"


def compute_signal_atlas_for_ckpts(panel, ckpts):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    panel = panel.copy()
    panel["is_real_vs_fake"] = (panel["label"] == "fake").astype(int)
    panel["lap_var_high"] = (panel["lap_var"] > panel["lap_var"].median()).astype(int)
    panel["min_dim_high"] = (panel["min_dim"] > panel["min_dim"].median()).astype(int)
    if panel["face_pixel_area"].notna().any():
        panel["face_size_high"] = (panel["face_pixel_area"] > panel["face_pixel_area"].median()).astype(int)
    else:
        panel["face_size_high"] = 0
    panel["is_chronic_6_int"] = panel["is_chronic_6"].astype(int)

    SIGNALS = [
        ("is_real_vs_fake", "is_real_vs_fake"), ("is_dor", "is_dor"),
        ("is_chronic_6", "is_chronic_6_int"), ("lap_var_high", "lap_var_high"),
        ("min_dim_high", "min_dim_high"), ("face_size_high", "face_size_high"),
    ]
    rows = []
    for ckpt in ckpts:
        for layer in LAYERS:
            try:
                feats, valid_idx = load_features(ckpt, layer)
            except Exception as exc:
                print(f"  {ckpt} L{layer}: {exc}")
                continue
            valid_set = set(int(v) for v in valid_idx.tolist())
            mask = panel["row_ix"].isin(valid_set).values
            panel_aligned = panel[mask].reset_index(drop=True)
            valid_to_pos = {int(v): r for r, v in enumerate(valid_idx)}
            sel_pos = [valid_to_pos[int(rx)] for rx in panel_aligned["row_ix"].values]
            X = feats[sel_pos]
            X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            for sig_name, sig_col in SIGNALS:
                if sig_col not in panel_aligned.columns:
                    continue
                y = panel_aligned[sig_col].values.astype(int)
                if len(y) < 20 or y.sum() < 5 or (len(y) - y.sum()) < 5:
                    rows.append({"ckpt": ckpt, "layer": layer, "signal": sig_name,
                                 "n": int(len(y)), "n_pos": int(y.sum()), "auc": float("nan")})
                    continue
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
                oof = np.zeros(len(y), dtype=np.float64)
                try:
                    for tr, te in skf.split(X_n, y):
                        clf = LogisticRegression(C=1.0, max_iter=3000, n_jobs=1, solver="lbfgs")
                        clf.fit(X_n[tr], y[tr])
                        oof[te] = clf.predict_proba(X_n[te])[:, 1]
                    auc = float(roc_auc_score(y, oof))
                except Exception:
                    auc = float("nan")
                rows.append({"ckpt": ckpt, "layer": layer, "signal": sig_name,
                             "n": int(len(y)), "n_pos": int(y.sum()), "auc": auc})
    return pd.DataFrame(rows)


def main():
    print("Building panel...")
    panel = build_panel()
    print(f"\nComputing atlas for {EXTRAS}...")
    new_atlas = compute_signal_atlas_for_ckpts(panel, EXTRAS)
    print(f"\n{len(new_atlas)} new atlas rows")

    if EXISTING_ATLAS.exists():
        existing = pd.read_csv(EXISTING_ATLAS)
        # Drop any of EXTRAS already there
        existing = existing[~existing["ckpt"].isin(EXTRAS)]
        combined = pd.concat([existing, new_atlas], ignore_index=True)
        combined.to_csv(OUT / "forgery_signal_atlas_extended.csv", index=False)
        print(f"Wrote: {len(combined)} rows ({combined['ckpt'].nunique()} ckpts)")

if __name__ == "__main__":
    main()
