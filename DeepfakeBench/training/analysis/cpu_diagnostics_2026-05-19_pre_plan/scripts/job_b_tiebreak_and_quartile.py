"""
Job B — Tiebreak audit + per-IQ-quartile FPR decomposition.

Closes open loop `lockbox-real-fpr-tiebreak-is-load-bearing` (MEDIUM, 2026-05-16).
- Re-rank with dev_fake_macro_recall as tiebreak (vs current ascending lockbox_real_fpr).
- Per-IQ-quartile FPR decomposition on lockbox reals, at each ckpt's selected τ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402


def main() -> int:
    log_f = common.LOGS / "job_b.log"
    sys.stdout = open(log_f, "w", buffering=1)
    sys.stderr = sys.stdout

    print("=== Job B: tiebreak audit + per-IQ-quartile FPR decomposition ===")

    # 1) Tiebreak audit
    summary = pd.read_csv(common.GCS_CACHE / "promotion_contract/checkpoint_summary.csv")
    print("\nCheckpoint summary loaded:")
    print(summary[[
        "checkpoint_key", "selected_threshold", "dev_primary_real_fpr",
        "dev_worst_real_stress_fpr", "dev_fake_macro_recall", "lockbox_real_fpr",
        "lockbox_fake_recall", "visomaster_enhanced_macro_dev__fake_recall", "promotion_rank"
    ]].to_string(index=False))

    # All ckpts pass gate1/2/3 except Slot α step3500 (dev_fake_macro_recall=0.226 < 0.30 floor).
    # Tiebreak among the 4 all-pass ckpts:
    floor = 0.30
    stress_ceil = 0.10
    real_fpr_budget = 0.07

    summary["gate_floor_pass"] = summary["dev_fake_macro_recall"] >= floor
    summary["gate_stress_pass"] = summary["dev_worst_real_stress_fpr"] <= stress_ceil + 1e-9
    summary["gate_real_fpr_pass"] = summary["dev_primary_real_fpr"] <= real_fpr_budget + 1e-9
    summary["all_pass"] = (
        summary["gate_floor_pass"] & summary["gate_stress_pass"] & summary["gate_real_fpr_pass"]
    )

    def add_rank(col: str, by: list[str], ascending: list[bool]) -> None:
        # Stable sort, then assign 1..N based on sorted position via index.
        sorted_idx = summary.sort_values(by=by, ascending=ascending).index.tolist()
        rank_map = {idx: i + 1 for i, idx in enumerate(sorted_idx)}
        summary[col] = summary.index.map(rank_map)

    add_rank("rank_v3fix_lockbox_asc", ["all_pass", "lockbox_real_fpr"], [False, True])
    add_rank("rank_alt_recall_desc", ["all_pass", "dev_fake_macro_recall"], [False, False])
    add_rank("rank_alt_lockbox_fake_desc", ["all_pass", "lockbox_fake_recall"], [False, False])
    add_rank("rank_alt_viso_desc", ["all_pass", "visomaster_enhanced_macro_dev__fake_recall"], [False, False])

    for k in [1.0, 2.0, 5.0, 10.0]:
        col = f"composite_k{k:g}"
        summary[col] = summary["lockbox_fake_recall"] - k * summary["lockbox_real_fpr"]
        add_rank(f"rank_composite_k{k:g}", ["all_pass", col], [False, False])

    ranking_cols = [
        "checkpoint_key", "all_pass", "dev_fake_macro_recall", "lockbox_real_fpr",
        "lockbox_fake_recall", "visomaster_enhanced_macro_dev__fake_recall",
        "rank_v3fix_lockbox_asc", "rank_alt_recall_desc", "rank_alt_lockbox_fake_desc",
        "rank_alt_viso_desc",
        "rank_composite_k1", "rank_composite_k2", "rank_composite_k5", "rank_composite_k10",
    ]
    print("\n=== RANKING COMPARISON ===")
    print(summary[ranking_cols].to_string(index=False))
    summary[ranking_cols].to_csv(common.OUT / "job_b_ranking_comparison.csv", index=False)

    # 2) Per-IQ-quartile FPR decomposition
    print("\n=== PER-IQ-QUARTILE FPR DECOMPOSITION ===")
    df_lock = common.load_lockbox_frames("teams_real_all_lockbox")
    df_real = df_lock[df_lock["label"] == 0].copy()
    print(f"Lockbox real frames available: {len(df_real)}")

    # Join with full_tags for IQ axes
    full_tags = common.load_lockbox_full_tags()
    print(f"Full tags rows: {len(full_tags)}; cols: {list(full_tags.columns)[:15]}...")
    # Join on gcs_uri ↔ frame_path
    join_col = "gcs_uri" if "gcs_uri" in full_tags.columns else "frame_path"
    df_real_iq = df_real.merge(
        full_tags[[join_col, "sharpness_laplacian", "brightness_v_mean", "face_pixel_area", "saturation_s_mean", "jpeg_qf_estimate"]].rename(columns={join_col: "frame_path"}),
        on="frame_path",
        how="left",
    )
    matched = df_real_iq["sharpness_laplacian"].notna().sum()
    print(f"Frames matched on IQ tags: {matched} / {len(df_real_iq)}")

    iq_axes = ["sharpness_laplacian", "brightness_v_mean", "face_pixel_area", "saturation_s_mean", "jpeg_qf_estimate"]

    quartile_rows = []
    for axis in iq_axes:
        if df_real_iq[axis].notna().sum() < 200:
            continue
        # Quartile boundaries on the panel
        q = df_real_iq[axis].quantile([0.25, 0.50, 0.75]).to_dict()
        df_real_iq[f"{axis}_quartile"] = pd.cut(
            df_real_iq[axis], bins=[-np.inf, q[0.25], q[0.50], q[0.75], np.inf],
            labels=["Q1_low", "Q2", "Q3", "Q4_high"]
        )
        for ckpt_key, _slug, tau in common.CKPTS:
            score_col = f"prob_{ckpt_key}"
            df_real_iq[f"_over_{ckpt_key}"] = (df_real_iq[score_col] >= tau).astype(int)
            for quartile, g in df_real_iq.groupby(f"{axis}_quartile", observed=True):
                if len(g) == 0:
                    continue
                fpr = g[f"_over_{ckpt_key}"].mean()
                quartile_rows.append({
                    "axis": axis,
                    "quartile": str(quartile),
                    "ckpt": ckpt_key,
                    "tau": tau,
                    "n_frames": len(g),
                    "n_overfire": int(g[f"_over_{ckpt_key}"].sum()),
                    "fpr": fpr,
                })

    quart_df = pd.DataFrame(quartile_rows)
    quart_df.to_csv(common.OUT / "job_b_per_iq_quartile_fpr.csv", index=False)
    print(f"\nSaved quartile FPR table: {common.OUT / 'job_b_per_iq_quartile_fpr.csv'} ({len(quart_df)} rows)")

    # Pivot for readability
    for axis in iq_axes:
        sub = quart_df[quart_df["axis"] == axis]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="ckpt", columns="quartile", values="fpr")
        # Reorder columns
        col_order = [c for c in ["Q1_low", "Q2", "Q3", "Q4_high"] if c in pivot.columns]
        pivot = pivot[col_order]
        print(f"\n=== FPR BY {axis.upper()} QUARTILE ===")
        print(pivot.round(4).to_string())

    print("\n=== Job B complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
