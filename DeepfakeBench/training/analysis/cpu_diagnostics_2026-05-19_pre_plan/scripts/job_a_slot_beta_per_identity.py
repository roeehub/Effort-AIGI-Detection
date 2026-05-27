"""
Job A — Slot β step3500 per-identity lockbox FPR, cross-tabbed with the
joint-marginal §2.4 bimodal partition (under-covered vs over-covered).

Tests the highest-severity open loop:
`slot-b-viso-lift-mechanism-and-lockbox-fpr-localization` (HIGH, 2026-05-16).
If FPR concentrates on under-covered chronic identities → rule-rescuable
(per memory `project_blend_unsharp_lever_2026-05-14`); if over-covered →
non-IQ-fixable; if mixed → need both data + recipe interventions.

Selected τ for Slot β step3500: 0.816038.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402


def main() -> int:
    common.LOGS.mkdir(exist_ok=True)
    log_f = common.LOGS / "job_a.log"
    sys.stdout = open(log_f, "w", buffering=1)
    sys.stderr = sys.stdout

    print("=== Job A: Slot β step3500 per-identity lockbox FPR × bimodal partition ===")
    df = common.load_lockbox_frames("teams_real_all_lockbox")
    print(f"Loaded {len(df)} lockbox-real frames across 5 ckpts")
    print(f"Columns: {list(df.columns)}")

    tau_slot_b = 0.816038
    score_col = "prob_SLOT_B_6AXIS_GRL_STEP3500"

    df_real = df[df["label"] == 0].copy()
    print(f"Real frames: {len(df_real)}")
    df_real["slot_b_overfire"] = df_real[score_col] >= tau_slot_b

    # Per-identity FPR
    grp = (
        df_real.groupby("identity_key")
        .agg(
            n_frames=("frame_path", "count"),
            n_overfire=("slot_b_overfire", "sum"),
            mean_prob=(score_col, "mean"),
            p90_prob=(score_col, lambda x: float(x.quantile(0.9))),
            is_chronic=("is_chronic", "first"),
        )
        .reset_index()
    )
    grp["fpr_at_tau"] = grp["n_overfire"] / grp["n_frames"]
    grp["overfire_share_of_total"] = grp["n_overfire"] / grp["n_overfire"].sum()
    grp = grp.sort_values("n_overfire", ascending=False).reset_index(drop=True)

    # Join with bimodal partition (per_chronic_identity_location)
    bim = common.load_bimodal_partition()
    print(f"Loaded bimodal partition: {len(bim)} identities")
    grp_with_bim = grp.merge(
        bim[["identity_key", "median_ratio", "coverage_class", "median_lap_var", "median_color_a_dev", "median_skin_frac", "median_min_dim"]],
        on="identity_key",
        how="left",
    )

    # Save full
    out_full = common.OUT / "job_a_per_identity_slot_b.csv"
    grp_with_bim.to_csv(out_full, index=False)
    print(f"\nSaved per-identity table: {out_full}")

    # Headline tables
    print("\n=== TOP 20 IDENTITIES BY SLOT β OVER-FIRE COUNT ===")
    print(grp_with_bim.head(20).to_string(index=False))

    print("\n=== TOTAL OVER-FIRES BY COVERAGE CLASS ===")
    by_cov = grp_with_bim.groupby("coverage_class", dropna=False).agg(
        n_identities=("identity_key", "count"),
        total_frames=("n_frames", "sum"),
        total_overfire=("n_overfire", "sum"),
    ).reset_index()
    by_cov["share_of_overfire"] = by_cov["total_overfire"] / by_cov["total_overfire"].sum()
    print(by_cov.to_string(index=False))
    by_cov.to_csv(common.OUT / "job_a_overfire_by_coverage_class.csv", index=False)

    # Chronic-only subset
    print("\n=== CHRONIC ONLY ===")
    chronic = grp_with_bim[grp_with_bim["is_chronic"] == True].copy()
    chronic_sorted = chronic.sort_values("n_overfire", ascending=False)
    print(chronic_sorted.to_string(index=False))
    chronic_sorted.to_csv(common.OUT / "job_a_chronic_only.csv", index=False)

    # Summary stats
    total_real = len(df_real)
    total_overfire = int(df_real["slot_b_overfire"].sum())
    print(f"\n=== SUMMARY ===")
    print(f"Lockbox real frames:     {total_real}")
    print(f"Slot β over-fires (τ={tau_slot_b}): {total_overfire}")
    print(f"Overall FPR:             {total_overfire / total_real:.4f}")
    print(f"Identities with ≥1 over-fire: {(grp_with_bim['n_overfire'] > 0).sum()}")
    chronic_overfire = int(chronic["n_overfire"].sum())
    print(f"Chronic over-fires:      {chronic_overfire} ({chronic_overfire/total_overfire:.1%} of total)")
    nonchronic_overfire = total_overfire - chronic_overfire
    print(f"Non-chronic over-fires:  {nonchronic_overfire}")

    print("\n=== Job A complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
