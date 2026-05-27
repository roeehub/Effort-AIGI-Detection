"""
Job E — Per-ckpt chronic-6 partition matrix (all 5 ckpts).

Replicates Job A across P8A, T5C, Slot α steps, Slot β.
6 (chronic identity) × 5 (ckpts) matrix of FPR + IQ-pocket location.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402


def main() -> int:
    log_f = common.LOGS / "job_e.log"
    sys.stdout = open(log_f, "w", buffering=1)
    sys.stderr = sys.stdout

    print("=== Job E: per-ckpt chronic-6 partition matrix ===")
    df = common.load_lockbox_frames("teams_real_all_lockbox")
    df_real = df[df["label"] == 0].copy()
    print(f"Lockbox real frames: {len(df_real)}")

    bim = common.load_bimodal_partition()

    rows = []
    for ckpt_key, _slug, tau in common.CKPTS:
        score_col = f"prob_{ckpt_key}"
        df_real[f"overfire_{ckpt_key}"] = df_real[score_col] >= tau

    # Per-identity, per-ckpt
    pieces = []
    for ckpt_key, _slug, tau in common.CKPTS:
        score_col = f"prob_{ckpt_key}"
        agg = (
            df_real.groupby("identity_key")
            .agg(
                n_frames=("frame_path", "count"),
                n_overfire=(f"overfire_{ckpt_key}", "sum"),
                mean_prob=(score_col, "mean"),
                is_chronic=("is_chronic", "first"),
            )
            .reset_index()
        )
        agg["fpr_at_tau"] = agg["n_overfire"] / agg["n_frames"]
        agg["ckpt"] = ckpt_key
        agg["tau"] = tau
        pieces.append(agg)
    long_df = pd.concat(pieces, ignore_index=True)

    # Save long form
    long_with_bim = long_df.merge(
        bim[["identity_key", "median_ratio", "coverage_class"]],
        on="identity_key",
        how="left",
    )
    long_with_bim.to_csv(common.OUT / "job_e_per_identity_per_ckpt_long.csv", index=False)
    print(f"\nSaved long-form: {common.OUT / 'job_e_per_identity_per_ckpt_long.csv'} ({len(long_with_bim)} rows)")

    # Wide pivot — FPR matrix
    fpr_pivot = long_df.pivot_table(
        index="identity_key", columns="ckpt", values="fpr_at_tau"
    ).reset_index()
    # Add coverage class
    fpr_pivot = fpr_pivot.merge(bim[["identity_key", "coverage_class", "median_ratio"]], on="identity_key", how="left")
    fpr_pivot["is_chronic"] = fpr_pivot["identity_key"].apply(common.is_chronic)
    # Per-row max FPR for sorting
    ckpt_cols = [c[0] for c in common.CKPTS]
    fpr_pivot["max_fpr_across_ckpts"] = fpr_pivot[ckpt_cols].max(axis=1)
    fpr_pivot = fpr_pivot.sort_values("max_fpr_across_ckpts", ascending=False).reset_index(drop=True)
    fpr_pivot.to_csv(common.OUT / "job_e_fpr_matrix_wide.csv", index=False)
    print(f"\nSaved wide FPR matrix: {common.OUT / 'job_e_fpr_matrix_wide.csv'}")

    # Chronic-only view
    print("\n=== CHRONIC IDENTITIES — FPR PER CKPT ===")
    chronic_view = fpr_pivot[fpr_pivot["is_chronic"] == True].head(25)
    print(chronic_view.to_string(index=False))

    # Count summary: per-ckpt where over-fires concentrate
    print("\n=== TOTAL OVER-FIRES BY (CKPT × COVERAGE_CLASS) ===")
    over_by_cov = long_with_bim.groupby(["ckpt", "coverage_class"], dropna=False).agg(
        total_overfire=("n_overfire", "sum"),
    ).reset_index().pivot(index="ckpt", columns="coverage_class", values="total_overfire").fillna(0).astype(int)
    print(over_by_cov.to_string())
    over_by_cov.to_csv(common.OUT / "job_e_overfire_by_ckpt_coverage.csv")

    # Per-ckpt fraction of over-fires on chronic
    print("\n=== PER-CKPT TOTAL FPR + CHRONIC SHARE OF OVER-FIRES ===")
    rows2 = []
    total_frames = len(df_real)
    for ckpt_key, _slug, tau in common.CKPTS:
        n_over = int(df_real[f"overfire_{ckpt_key}"].sum())
        n_over_chronic = int((df_real[f"overfire_{ckpt_key}"] & df_real["is_chronic"]).sum())
        rows2.append({
            "ckpt": ckpt_key,
            "tau": tau,
            "n_frames": total_frames,
            "n_overfire": n_over,
            "fpr": n_over / total_frames if total_frames else 0.0,
            "n_overfire_chronic": n_over_chronic,
            "chronic_share_of_overfire": (n_over_chronic / n_over) if n_over else 0.0,
        })
    summary = pd.DataFrame(rows2)
    print(summary.to_string(index=False))
    summary.to_csv(common.OUT / "job_e_per_ckpt_summary.csv", index=False)

    print("\n=== Job E complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
