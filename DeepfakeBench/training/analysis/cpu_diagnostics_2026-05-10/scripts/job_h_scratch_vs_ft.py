"""Job H — From-scratch (E2B) vs FT-from-P8A representation comparison.

Question (user-posed): could a from-scratch (or curriculum) recipe produce a
better model than FT-from-P8A? E2B is the from-scratch reference; P8A is the
FT-chain reference. Compare their per-layer representations:
  1. Per-layer cosine similarity (P8A vs E2B on shared frames)
  2. Per-layer probe AUC for forgery / shortcuts (already in atlas)
  3. Where does P8A's L11 invariance arise? Layer where P8A separates
     real/fake without separating identity/IQ?
  4. Does E2B at any layer have similar "clean" forgery-only signal?

Reuses cached features at iq_perlayer_probe_2026-05-08/_cache/intermediate__*.npz
and the existing forgery_signal_atlas.csv.

Outputs:
  - outputs/job_h_layer_cosine_p8a_e2b.csv (per layer)
  - outputs/job_h_invariance_score.csv (per ckpt × layer: forgery_AUC - shortcut_AUC)
  - outputs/job_h_PROFILE.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
CACHE = ROOT / "analysis/iq_perlayer_probe_2026-05-08/_cache"
ATLAS_CSV = ROOT / "analysis/cpu_diagnostics_2026-05-09/outputs/forgery_signal_atlas.csv"
OUT = ROOT / "analysis/cpu_diagnostics_2026-05-10/outputs"

LAYERS = [0, 3, 6, 9, 11]


def load_layer_features(label: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    p = CACHE / f"intermediate__{label}__layer{layer:02d}__n800.npz"
    data = np.load(p)
    return data["features"], data["valid_idx"]


def cosine_per_frame(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return (A_n * B_n).sum(axis=1)


def main():
    print("=== Job H — Scratch vs FT representation comparison ===\n")

    # Load atlas
    atlas = pd.read_csv(ATLAS_CSV)
    print(f"Loaded forgery atlas: {len(atlas)} rows, {atlas['ckpt'].nunique()} ckpts")
    print(f"Available ckpts: {sorted(atlas['ckpt'].unique())}\n")

    # ============================================================
    # 1. Per-layer cosine similarity P8A vs E2B
    # ============================================================
    print("=== 1. Per-layer cosine sim (P8A vs E2B) ===")
    cosine_rows = []
    for layer in LAYERS:
        try:
            P, P_idx = load_layer_features("P8A", layer)
            E, E_idx = load_layer_features("E2B", layer)
            common = sorted(set(P_idx.tolist()) & set(E_idx.tolist()))
            P_pos = {int(v): r for r, v in enumerate(P_idx)}
            E_pos = {int(v): r for r, v in enumerate(E_idx)}
            P_sub = P[[P_pos[c] for c in common]]
            E_sub = E[[E_pos[c] for c in common]]
            cos = cosine_per_frame(P_sub, E_sub)
            cosine_rows.append({
                "layer": layer,
                "n": len(common),
                "cos_p25": float(np.percentile(cos, 25)),
                "cos_p50": float(np.percentile(cos, 50)),
                "cos_p75": float(np.percentile(cos, 75)),
                "cos_min": float(np.min(cos)),
                "cos_max": float(np.max(cos)),
            })
            print(f"  Layer {layer}: n={len(common)}, cos p50={np.percentile(cos, 50):.3f}, p25={np.percentile(cos, 25):.3f}, p75={np.percentile(cos, 75):.3f}")
        except Exception as exc:
            print(f"  Layer {layer}: ERROR {exc}")
    pd.DataFrame(cosine_rows).to_csv(OUT / "job_h_layer_cosine_p8a_e2b.csv", index=False)

    # ============================================================
    # 2. Invariance score per ckpt x layer = forgery_AUC - max_shortcut_AUC
    # An "ideal" representation has high forgery_AUC (real/fake) AND
    # low shortcut_AUC (the model can't predict identity/IQ from features).
    # ============================================================
    print("\n=== 2. Forgery-vs-shortcut per layer per ckpt ===")
    pivot = atlas.pivot_table(index=["ckpt", "layer"], columns="signal", values="auc").reset_index()
    print(f"Atlas signals: {[c for c in pivot.columns if c not in ['ckpt', 'layer']]}")

    # Compute invariance score: forgery_AUC - mean(shortcut_AUCs)
    shortcut_signals = ["is_dor", "is_chronic_6", "lap_var_high", "min_dim_high", "face_size_high"]
    forgery_signal = "is_real_vs_fake"
    invariance_rows = []
    for _, r in pivot.iterrows():
        forgery = r.get(forgery_signal, np.nan)
        shortcuts = [r.get(s, np.nan) for s in shortcut_signals]
        shortcuts_clean = [s for s in shortcuts if not pd.isna(s)]
        max_shortcut = max(shortcuts_clean) if shortcuts_clean else np.nan
        mean_shortcut = np.mean(shortcuts_clean) if shortcuts_clean else np.nan
        invariance_rows.append({
            "ckpt": r["ckpt"],
            "layer": int(r["layer"]),
            "forgery_auc": forgery,
            "max_shortcut_auc": max_shortcut,
            "mean_shortcut_auc": mean_shortcut,
            "invariance_score_max": forgery - max_shortcut if not pd.isna(forgery) and not pd.isna(max_shortcut) else np.nan,
            "invariance_score_mean": forgery - mean_shortcut if not pd.isna(forgery) and not pd.isna(mean_shortcut) else np.nan,
            **{s: r.get(s, np.nan) for s in shortcut_signals + [forgery_signal]},
        })
    inv_df = pd.DataFrame(invariance_rows)
    inv_df.to_csv(OUT / "job_h_invariance_score.csv", index=False)

    # Headline table: P8A vs E2B per layer
    print("\nForgery/Shortcut AUC at L11 per ckpt:")
    l11 = inv_df[inv_df["layer"] == 11].sort_values("invariance_score_max", ascending=False)
    cols_show = ["ckpt", "forgery_auc", "is_dor", "is_chronic_6", "lap_var_high", "min_dim_high"]
    print(l11[cols_show].to_string(index=False))

    print("\nForgery/Shortcut AUC at L0 per ckpt (baseline before training):")
    l0 = inv_df[inv_df["layer"] == 0]
    print(l0[cols_show].to_string(index=False))

    # ============================================================
    # 3. Layer-by-layer trajectory: P8A vs E2B forgery AUC
    # ============================================================
    print("\n=== 3. Layer-by-layer forgery_auc trajectory ===")
    md = ["# Job H — From-scratch (E2B) vs FT-from-P8A representation comparison", ""]
    md.append("## 1. Per-layer cosine similarity P8A vs E2B")
    md.append("")
    md.append("| layer | n | cos p25 | cos p50 | cos p75 | cos min |")
    md.append("|---:|---:|---:|---:|---:|---:|")
    for r in cosine_rows:
        md.append(f"| {r['layer']} | {r['n']} | {r['cos_p25']:.3f} | {r['cos_p50']:.3f} | {r['cos_p75']:.3f} | {r['cos_min']:.3f} |")
    md.append("")
    md.append("## 2. Forgery vs shortcut AUC per layer")
    md.append("")
    pivot_p = inv_df[inv_df["ckpt"].isin(["P8A", "E2B"])].pivot_table(
        index="layer", columns="ckpt",
        values=["forgery_auc", "is_dor", "is_chronic_6", "lap_var_high", "min_dim_high"],
    )
    md.append("### P8A trajectory")
    md.append("")
    md.append("| layer | forgery_auc | is_dor | is_chronic_6 | lap_var_high | min_dim_high | invariance_max |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    for layer in LAYERS:
        row = inv_df[(inv_df["ckpt"] == "P8A") & (inv_df["layer"] == layer)].iloc[0] if len(inv_df[(inv_df["ckpt"] == "P8A") & (inv_df["layer"] == layer)]) > 0 else None
        if row is not None:
            md.append(f"| {layer} | {row['forgery_auc']:.3f} | {row['is_dor']:.3f} | {row['is_chronic_6']:.3f} | {row['lap_var_high']:.3f} | {row['min_dim_high']:.3f} | {row['invariance_score_max']:.3f} |")
    md.append("")
    md.append("### E2B trajectory")
    md.append("")
    md.append("| layer | forgery_auc | is_dor | is_chronic_6 | lap_var_high | min_dim_high | invariance_max |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    for layer in LAYERS:
        row = inv_df[(inv_df["ckpt"] == "E2B") & (inv_df["layer"] == layer)].iloc[0] if len(inv_df[(inv_df["ckpt"] == "E2B") & (inv_df["layer"] == layer)]) > 0 else None
        if row is not None:
            md.append(f"| {layer} | {row['forgery_auc']:.3f} | {row['is_dor']:.3f} | {row['is_chronic_6']:.3f} | {row['lap_var_high']:.3f} | {row['min_dim_high']:.3f} | {row['invariance_score_max']:.3f} |")
    md.append("")

    md.append("## 3. Reading")
    md.append("")
    md.append("**Invariance score** (forgery_auc - max_shortcut_auc) at each layer:")
    md.append("- Higher = more 'forgery-only' signal at that layer")
    md.append("- Lower = forgery signal is co-mingled with shortcuts")
    md.append("- A 'good' representation has high forgery_auc AND low shortcut_auc (= high invariance)")
    md.append("")
    md.append("**Cosine similarity** (P8A vs E2B same frame, same layer):")
    md.append("- Higher = both ckpts produce similar representations at that layer")
    md.append("- Lower = ckpts have diverged at that layer")
    md.append("- Interpretation: shows where P8A's FT chain pulled away from E2B's scratch path")

    (OUT / "job_h_PROFILE.md").write_text("\n".join(md))
    print(f"\nWrote outputs.")

if __name__ == "__main__":
    main()
