"""Extend the forgery_signal_atlas with T3_S1 step1500 + step2500.

Reuses the panel-building + signal-probe logic from
`analysis/cpu_diagnostics_2026-05-09/run_analyses.py` but runs ONLY for
T3 ckpts and appends to the existing CSV.

Outputs:
  - analysis/cpu_diagnostics_2026-05-10/outputs/forgery_signal_atlas_extended.csv
    (full atlas: original 12 ckpts + T3 step1500 + T3 step2500)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the existing build_panel and load_features helpers
sys.path.insert(0, str(REPO_ROOT / "analysis" / "cpu_diagnostics_2026-05-09"))
from run_analyses import build_panel, load_features, LAYERS  # noqa

OUT = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-10/outputs"
EXISTING_ATLAS = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-09/outputs/forgery_signal_atlas.csv"

T3_CKPTS = ["T3_S1_step1500", "T3_S1_step2500"]


def compute_signal_atlas_for_ckpts(panel: pd.DataFrame, ckpts: list[str]) -> pd.DataFrame:
    """Same logic as compute_signal_atlas() but limited to specified ckpts."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    panel = panel.copy()
    panel["is_real_vs_fake"] = (panel["label"] == "fake").astype(int)
    panel["lap_var_high"] = (panel["lap_var"] > panel["lap_var"].median()).astype(int)
    panel["min_dim_high"] = (panel["min_dim"] > panel["min_dim"].median()).astype(int)
    if panel["face_pixel_area"].notna().any():
        panel["face_size_high"] = (
            panel["face_pixel_area"] > panel["face_pixel_area"].median()
        ).astype(int)
    else:
        panel["face_size_high"] = 0
    panel["is_chronic_6_int"] = panel["is_chronic_6"].astype(int)

    SIGNALS = [
        ("is_real_vs_fake", "is_real_vs_fake"),
        ("is_dor", "is_dor"),
        ("is_chronic_6", "is_chronic_6_int"),
        ("lap_var_high", "lap_var_high"),
        ("min_dim_high", "min_dim_high"),
        ("face_size_high", "face_size_high"),
    ]

    rows = []
    for ckpt in ckpts:
        for layer in LAYERS:
            try:
                feats, valid_idx = load_features(ckpt, layer)
            except Exception as exc:
                print(f"  {ckpt} layer={layer} feature load failed: {exc}")
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
                except Exception as exc:
                    print(f"  {ckpt} layer={layer} signal={sig_name} probe failed: {exc}")
                    auc = float("nan")
                rows.append({"ckpt": ckpt, "layer": layer, "signal": sig_name,
                             "n": int(len(y)), "n_pos": int(y.sum()), "auc": auc})
                print(f"  {ckpt} L{layer:02d} {sig_name:18s}: AUC={auc:.3f} (n={len(y)}, n_pos={y.sum()})")
    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    print("Building panel...")
    panel = build_panel()
    print(f"Panel: {len(panel)} rows")
    print(f"\nComputing forgery atlas for T3 ckpts...")
    new_atlas = compute_signal_atlas_for_ckpts(panel, T3_CKPTS)
    print(f"\nNew atlas rows: {len(new_atlas)}")

    # Combine with existing atlas
    if EXISTING_ATLAS.exists():
        existing = pd.read_csv(EXISTING_ATLAS)
        combined = pd.concat([existing, new_atlas], ignore_index=True)
        combined.to_csv(OUT / "forgery_signal_atlas_extended.csv", index=False)
        print(f"Wrote extended atlas: {len(combined)} rows ({existing['ckpt'].nunique()} + {new_atlas['ckpt'].nunique()} ckpts)")
    else:
        new_atlas.to_csv(OUT / "forgery_signal_atlas_extended.csv", index=False)

if __name__ == "__main__":
    main()
