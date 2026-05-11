"""Extend the forgery_signal_atlas with T4 multi-axis-L11-GRL ckpts.

The load-bearing question: did T4 (FT-from-P8A + T3 SLOT1 data lever +
multi-axis L11 encoder GRL) move L11 inv_mean above the 0.031 ceiling
shared by all 7 prior trained ckpts (P8A, E2B, T3 family, MCLIOEXB,
P18T, P18C)? Pre-test 1 predicted ~4× lift on P8A frozen features
(0.012 → 0.048 at k=128, λ=2.0). T4 is the GPU-scale realization.

Reuses the panel-building + signal-probe logic from
analysis/cpu_diagnostics_2026-05-09/run_analyses.py.

Outputs:
  - analysis/cpu_diagnostics_2026-05-10/outputs/forgery_signal_atlas_with_t4.csv
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "analysis" / "cpu_diagnostics_2026-05-09"))
from run_analyses import build_panel, load_features, LAYERS  # noqa

OUT = REPO_ROOT / "analysis/cpu_diagnostics_2026-05-10/outputs"
EXISTING_ATLAS = OUT / "forgery_signal_atlas_extended.csv"

T4_CKPTS = [
    "T4_L1_step5000", "T4_L1_step9000", "T4_L1_step10500", "T4_L1_step11250",
    "T4_L2_step1500", "T4_L2_step2500",
    "T5A_C1_step3000", "T5A_C1_step3500", "T5A_C1_step4250",
    "T5A_C1_step6500", "T5A_C1_step8000", "T5A_C1_step8500",
]


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


def compute_inv_mean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute inv_mean per ckpt at L11 = forgery_AUC - mean(shortcut_AUCs)."""
    SHORTCUT_SIGNALS = ["is_dor", "is_chronic_6", "lap_var_high", "min_dim_high", "face_size_high"]
    rows = []
    for ckpt in df["ckpt"].unique():
        sub = df[(df["ckpt"] == ckpt) & (df["layer"] == 11)]
        forgery_row = sub[sub["signal"] == "is_real_vs_fake"]
        if forgery_row.empty:
            continue
        forgery_auc = float(forgery_row["auc"].iloc[0])
        shortcut_aucs = []
        per_axis = {}
        for sig in SHORTCUT_SIGNALS:
            sig_row = sub[sub["signal"] == sig]
            if not sig_row.empty:
                v = float(sig_row["auc"].iloc[0])
                if not np.isnan(v):
                    shortcut_aucs.append(v)
                    per_axis[sig] = v
        if not shortcut_aucs:
            continue
        inv_mean = forgery_auc - np.mean(shortcut_aucs)
        row = {"ckpt": ckpt, "layer": 11, "forgery_auc": forgery_auc, "inv_mean": inv_mean}
        row.update(per_axis)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("inv_mean", ascending=False)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    print("Building panel...")
    panel = build_panel()
    print(f"Panel: {len(panel)} rows")
    print(f"\nComputing forgery atlas for T4 ckpts ({T4_CKPTS})...")
    new_atlas = compute_signal_atlas_for_ckpts(panel, T4_CKPTS)
    print(f"\nNew atlas rows: {len(new_atlas)}")

    if EXISTING_ATLAS.exists():
        existing = pd.read_csv(EXISTING_ATLAS)
        combined = pd.concat([existing, new_atlas], ignore_index=True)
        out_path = OUT / "forgery_signal_atlas_with_t4.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nWrote combined atlas: {len(combined)} rows ({combined['ckpt'].nunique()} ckpts)")
        print(f"  → {out_path}")
    else:
        out_path = OUT / "forgery_signal_atlas_with_t4.csv"
        new_atlas.to_csv(out_path, index=False)
        combined = new_atlas

    # Compute inv_mean table — the key deliverable
    print("\n" + "=" * 70)
    print("L11 inv_mean table (sorted desc)")
    print("=" * 70)
    inv_table = compute_inv_mean_table(combined)
    print(inv_table.to_string(index=False, float_format="%.4f"))
    inv_path = OUT / "L11_inv_mean_with_t4.csv"
    inv_table.to_csv(inv_path, index=False)
    print(f"\n  → {inv_path}")

    # Specifically highlight T4 vs ceiling
    print("\n" + "=" * 70)
    print("T4 vs prior ceiling")
    print("=" * 70)
    prior = inv_table[~inv_table["ckpt"].str.startswith("T4")]["inv_mean"].max()
    print(f"Prior ckpt max inv_mean (ceiling): {prior:.4f}")
    for _, r in inv_table[inv_table["ckpt"].str.startswith("T4")].iterrows():
        delta = r['inv_mean'] - prior
        symbol = "↑" if delta > 0 else "↓"
        pct = (delta / prior) * 100 if prior > 0 else float('nan')
        print(f"  {r['ckpt']:25s} inv_mean={r['inv_mean']:.4f}  vs ceiling: {delta:+.4f} ({pct:+.1f}%) {symbol}")

if __name__ == "__main__":
    main()
