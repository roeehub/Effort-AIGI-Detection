"""Per-layer feature-distance analysis: P8A vs Slot A v2 vs T5C on substrate-pair pool.

Uses cached features from analysis/substrate_pair_geometry_2026-05-22/feats/
  - 3 ckpts × 4 layers × {clean, teams} substrate
  - ~5475 clean + 5478 teams frames per (ckpt × layer)

Outputs feature-distance metrics that inform the output-preservation aux loss spec:
  - Per-(ckpt-pair × layer): cosine similarity, MSE, normalized MSE
  - Substrate-pair Δ within each ckpt (clean ↔ teams of same identity)
  - Cross-ckpt feature drift on identical frames

Goal: identify which layer's drift most accounts for the per-ckpt behavior delta,
informs (a) which layer to regularize on, (b) which reference encoder to use.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
FEATS = ROOT / "analysis/substrate_pair_geometry_2026-05-22/feats"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

CKPTS = ["P8A_step5000", "SlotAv2_step3500", "T5C_step3500"]
LAYERS = ["L0", "L4", "L8", "L11"]
SUBSTRATES = ["clean", "teams"]


def load_feats(ckpt: str, layer: str, sub: str) -> np.ndarray:
    return np.load(FEATS / f"{ckpt}_{layer}_{sub}.npy")


def cos_sim_per_row(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity per row (A[i] vs B[i])."""
    norms_A = np.linalg.norm(A, axis=1, keepdims=True)
    norms_B = np.linalg.norm(B, axis=1, keepdims=True)
    A_n = A / (norms_A + 1e-12)
    B_n = B / (norms_B + 1e-12)
    return (A_n * B_n).sum(axis=1)


def cross_ckpt_drift(ckpt_a: str, ckpt_b: str, layer: str) -> dict:
    """Compute feature drift between two ckpts at a layer, averaged over clean + teams."""
    Ac = load_feats(ckpt_a, layer, "clean")
    Bc = load_feats(ckpt_b, layer, "clean")
    At = load_feats(ckpt_a, layer, "teams")
    Bt = load_feats(ckpt_b, layer, "teams")
    # assume same row alignment within (clean) and within (teams) — they came from the same script
    if Ac.shape != Bc.shape or At.shape != Bt.shape:
        return {"error": f"shape mismatch {ckpt_a} vs {ckpt_b}: {Ac.shape}/{Bc.shape}, {At.shape}/{Bt.shape}"}

    cos_clean = cos_sim_per_row(Ac, Bc)
    cos_teams = cos_sim_per_row(At, Bt)
    mse_clean = float(((Ac - Bc) ** 2).mean())
    mse_teams = float(((At - Bt) ** 2).mean())
    # Normalize MSE by mean feature variance (per-encoder average) so layers/encoders comparable
    var_a = float(np.var(np.concatenate([Ac, At], axis=0)))
    var_b = float(np.var(np.concatenate([Bc, Bt], axis=0)))
    norm_mse_clean = mse_clean / (0.5 * (var_a + var_b))
    norm_mse_teams = mse_teams / (0.5 * (var_a + var_b))

    return {
        "ckpt_a": ckpt_a, "ckpt_b": ckpt_b, "layer": layer,
        "n_clean": len(Ac), "n_teams": len(At),
        "cos_clean_mean": float(cos_clean.mean()), "cos_clean_std": float(cos_clean.std()),
        "cos_teams_mean": float(cos_teams.mean()), "cos_teams_std": float(cos_teams.std()),
        "mse_clean": mse_clean, "mse_teams": mse_teams,
        "norm_mse_clean": norm_mse_clean, "norm_mse_teams": norm_mse_teams,
        "var_a_avg": var_a, "var_b_avg": var_b,
    }


def within_ckpt_substrate_delta(ckpt: str, layer: str) -> dict:
    """Compute how much (clean → teams) shifts features within the same ckpt at a layer.

    We don't know exact pairing between clean and teams rows (the parquet inventory has
    multi-frame-per-identity); we use *centroid* distance.
    """
    C = load_feats(ckpt, layer, "clean")
    T = load_feats(ckpt, layer, "teams")
    centroid_C = C.mean(axis=0)
    centroid_T = T.mean(axis=0)
    cos_centroid = float(centroid_C @ centroid_T / (np.linalg.norm(centroid_C) * np.linalg.norm(centroid_T) + 1e-12))
    mse_centroid = float(((centroid_C - centroid_T) ** 2).mean())
    # also intra-substrate vs inter-substrate variance ratio
    var_within_C = float(np.var(C, axis=0).mean())
    var_within_T = float(np.var(T, axis=0).mean())
    var_centroid_diff = float((centroid_C - centroid_T).var())
    return {
        "ckpt": ckpt, "layer": layer,
        "cos_centroid_clean_vs_teams": cos_centroid,
        "mse_centroid_clean_vs_teams": mse_centroid,
        "var_within_clean": var_within_C,
        "var_within_teams": var_within_T,
        "var_centroid_diff": var_centroid_diff,
    }


def main() -> None:
    cross_rows = []
    within_rows = []

    print("=== Cross-ckpt feature drift per layer ===")
    for layer in LAYERS:
        for ckpt_a, ckpt_b in combinations(CKPTS, 2):
            r = cross_ckpt_drift(ckpt_a, ckpt_b, layer)
            if "error" in r:
                print(f"  ERROR: {r['error']}")
                continue
            cross_rows.append(r)
            print(f"  {layer:3s}  {ckpt_a:18s} vs {ckpt_b:18s}  "
                  f"cos_clean={r['cos_clean_mean']:.4f}±{r['cos_clean_std']:.4f}  "
                  f"cos_teams={r['cos_teams_mean']:.4f}±{r['cos_teams_std']:.4f}  "
                  f"norm_mse_clean={r['norm_mse_clean']:.4f}  norm_mse_teams={r['norm_mse_teams']:.4f}")

    print("\n=== Within-ckpt substrate-pair delta (clean → teams) per layer ===")
    for layer in LAYERS:
        for ckpt in CKPTS:
            r = within_ckpt_substrate_delta(ckpt, layer)
            within_rows.append(r)
            print(f"  {layer:3s}  {ckpt:18s}  cos_centroid={r['cos_centroid_clean_vs_teams']:.4f}  "
                  f"mse_centroid={r['mse_centroid_clean_vs_teams']:.6f}  "
                  f"var_centroid_diff={r['var_centroid_diff']:.6f}  "
                  f"var_within_clean={r['var_within_clean']:.6f}")

    pd.DataFrame(cross_rows).to_csv(OUT / "cross_ckpt_drift.csv", index=False)
    pd.DataFrame(within_rows).to_csv(OUT / "within_ckpt_substrate_delta.csv", index=False)

    # ---- Layer-wise summary: which layer has the most cross-ckpt drift? ----
    print("\n=== Layer-wise cross-ckpt drift summary ===")
    cross_df = pd.DataFrame(cross_rows)
    layer_summary = cross_df.groupby("layer").agg(
        mean_cos_clean=("cos_clean_mean", "mean"),
        mean_cos_teams=("cos_teams_mean", "mean"),
        mean_norm_mse_clean=("norm_mse_clean", "mean"),
        mean_norm_mse_teams=("norm_mse_teams", "mean"),
    ).reset_index()
    print(layer_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    layer_summary.to_csv(OUT / "layer_summary.csv", index=False)

    # ---- P8A vs Slot A v2 specifically — the candidate reference for output-preservation ----
    print("\n=== P8A vs Slot A v2 (candidate output-preservation reference) ===")
    pas = cross_df[(cross_df["ckpt_a"] == "P8A_step5000") & (cross_df["ckpt_b"] == "SlotAv2_step3500")]
    print(pas[["layer", "cos_clean_mean", "cos_teams_mean", "norm_mse_clean", "norm_mse_teams"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
