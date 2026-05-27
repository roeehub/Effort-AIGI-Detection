"""
Extension diagnostics 2026-05-20 — discovered that auto-mode-scorecard-2026-05-16
(separate from the overnight scorecard) includes Slot A v2 anchor_aware and
Slot B real_rebalance ckpts. Yesterday's pre-plan analysis missed these.

This script extends Jobs A / E / B / F / J to include all 9 ckpts across both
scorecards. The Slot A v2 step3500 ckpt is rank-2 in the auto-mode-scorecard
with lockbox_real_fpr=0.0191 (0.07pp behind P8A's 0.0184) — the strongest
single deployment candidate seen yet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as common_overnight  # noqa: E402


HERE = common_overnight.HERE
GCS_AUTO = HERE / "gcs_cache_auto_mode"
OUT = HERE / "outputs_2026-05-20"
LOGS = HERE / "logs"
OUT.mkdir(exist_ok=True)

# Auto-mode ckpts (added today).
CKPTS_AUTO = [
    ("SLOT_A_ANCHOR_AWARE_STEP1500", "slot_a_anchor_aware_step1500", 0.930254),
    ("SLOT_A_ANCHOR_AWARE_STEP3500", "slot_a_anchor_aware_step3500", 0.787956),
    ("SLOT_B_REAL_REBAL_STEP1500", "slot_b_real_rebal_step1500", 0.956587),
    ("SLOT_B_REAL_REBAL_STEP3500", "slot_b_real_rebal_step3500", 0.839983),
]
# Overnight ckpts inherited from common.CKPTS.
ALL_CKPTS = list(common_overnight.CKPTS) + CKPTS_AUTO


def load_auto_lockbox() -> pd.DataFrame:
    pieces = []
    for key, slug, tau in CKPTS_AUTO:
        path = GCS_AUTO / f"teams_real_all_lockbox_{slug}_frames_report.csv"
        df = pd.read_csv(path)
        df = df.rename(columns={"frame_prob": f"prob_{key}"})
        if not pieces:
            pieces.append(df[["frame_path", "video_id", "label", f"prob_{key}"]])
        else:
            pieces.append(df[["frame_path", f"prob_{key}"]])
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on="frame_path", how="outer")
    return out


def load_all_lockbox_real() -> pd.DataFrame:
    """Merge overnight + auto scorecards into a single 9-ckpt wide DataFrame."""
    overnight = common_overnight.load_lockbox_frames("teams_real_all_lockbox")
    overnight_real = overnight[overnight["label"] == 0].copy()
    auto = load_auto_lockbox()
    auto_real = auto[auto["label"] == 0].copy()
    auto_cols = [c for c in auto_real.columns if c.startswith("prob_")]
    merged = overnight_real.merge(auto_real[["frame_path"] + auto_cols], on="frame_path", how="inner")
    print(f"Overnight rows: {len(overnight_real)}, auto rows: {len(auto_real)}, merged inner: {len(merged)}")
    return merged


def job_a_extended(merged: pd.DataFrame, bim: pd.DataFrame) -> None:
    """Per-identity lockbox FPR for ALL 9 ckpts."""
    print("\n\n========================================")
    print("Job A extended — per-identity lockbox FPR for all 9 ckpts")
    print("========================================")

    rows = []
    for ckpt, slug, tau in ALL_CKPTS:
        sc = f"prob_{ckpt}"
        over = merged[sc] >= tau
        per_ident = merged.assign(_over=over.astype(int)).groupby("identity_key").agg(
            n_frames=("frame_path", "count"),
            n_overfire=("_over", "sum"),
        ).reset_index()
        per_ident["ckpt"] = ckpt
        per_ident["tau"] = tau
        per_ident["fpr_at_tau"] = per_ident["n_overfire"] / per_ident["n_frames"]
        rows.append(per_ident)
    long = pd.concat(rows, ignore_index=True)

    bim_lookup = bim[["identity_key", "median_ratio", "coverage_class"]]
    long = long.merge(bim_lookup, on="identity_key", how="left")

    long.to_csv(OUT / "ext_job_a_per_ckpt_per_identity.csv", index=False)
    print(f"Saved long: {OUT / 'ext_job_a_per_ckpt_per_identity.csv'} ({len(long)} rows)")

    # Wide FPR matrix (chronic only)
    long_with_chronic = long.copy()
    long_with_chronic["is_chronic"] = long_with_chronic["identity_key"].apply(common_overnight.is_chronic)
    chronic_long = long_with_chronic[long_with_chronic["is_chronic"]]
    wide = chronic_long.pivot_table(index="identity_key", columns="ckpt", values="fpr_at_tau").reset_index()
    wide = wide.merge(
        bim_lookup, on="identity_key", how="left"
    )
    cols_ordered = ["identity_key"] + [c[0] for c in ALL_CKPTS] + ["coverage_class", "median_ratio"]
    cols_present = [c for c in cols_ordered if c in wide.columns]
    wide = wide[cols_present]
    wide.to_csv(OUT / "ext_job_a_chronic_fpr_matrix_9ckpts.csv", index=False)

    print("\n=== Per-ckpt totals (all-identity sum / lockbox_real_fpr) ===")
    summary = []
    n_total = len(merged)
    for ckpt, slug, tau in ALL_CKPTS:
        n_over = int((merged[f"prob_{ckpt}"] >= tau).sum())
        n_over_chronic = int(((merged[f"prob_{ckpt}"] >= tau) & merged["is_chronic"]).sum())
        summary.append({
            "ckpt": ckpt,
            "tau": tau,
            "n_frames": n_total,
            "n_overfire": n_over,
            "fpr": n_over / n_total,
            "n_overfire_chronic": n_over_chronic,
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT / "ext_job_a_per_ckpt_summary_9ckpts.csv", index=False)
    print(summary_df.to_string(index=False))

    print("\n=== Slot A v2 step3500 per-identity decomp ===")
    s_a_v2 = long[long["ckpt"] == "SLOT_A_ANCHOR_AWARE_STEP3500"].sort_values("n_overfire", ascending=False)
    chronic_only = s_a_v2[s_a_v2["identity_key"].apply(common_overnight.is_chronic)]
    print(chronic_only.head(20).to_string(index=False))


def job_b_extended_ranking() -> None:
    """Tiebreak audit including all 6 ckpts in the auto-mode scorecard."""
    print("\n\n========================================")
    print("Job B extended — tiebreak audit with auto-mode 6 ckpts")
    print("========================================")
    s = pd.read_csv(GCS_AUTO / "checkpoint_summary.csv")
    s = s.rename(columns={
        "visomaster_enhanced_macro_dev__fake_recall": "viso_recall",
    })
    floor = 0.30
    stress = 0.10
    rfpr_budget = 0.07
    s["all_pass"] = (
        (s["dev_fake_macro_recall"] >= floor) &
        (s["dev_worst_real_stress_fpr"] <= stress + 1e-9) &
        (s["dev_primary_real_fpr"] <= rfpr_budget + 1e-9)
    )

    def add_rank(col, by, asc):
        sorted_idx = s.sort_values(by=by, ascending=asc).index.tolist()
        rank_map = {idx: i + 1 for i, idx in enumerate(sorted_idx)}
        s[col] = s.index.map(rank_map)

    add_rank("rank_v3fix", ["all_pass", "lockbox_real_fpr"], [False, True])
    add_rank("rank_recall_desc", ["all_pass", "dev_fake_macro_recall"], [False, False])
    add_rank("rank_lockbox_fake_desc", ["all_pass", "lockbox_fake_recall"], [False, False])
    add_rank("rank_viso_desc", ["all_pass", "viso_recall"], [False, False])
    for k in [1.0, 2.0, 5.0, 10.0]:
        col = f"composite_k{k:g}"
        s[col] = s["lockbox_fake_recall"] - k * s["lockbox_real_fpr"]
        add_rank(f"rank_composite_k{k:g}", ["all_pass", col], [False, False])

    cols = ["checkpoint_key", "all_pass", "dev_fake_macro_recall", "lockbox_real_fpr",
            "lockbox_fake_recall", "viso_recall",
            "rank_v3fix", "rank_recall_desc", "rank_lockbox_fake_desc", "rank_viso_desc",
            "rank_composite_k1", "rank_composite_k2", "rank_composite_k5", "rank_composite_k10"]
    print(s[cols].to_string(index=False))
    s[cols].to_csv(OUT / "ext_job_b_ranking_9ckpts.csv", index=False)


def job_l_ensemble_routing(merged: pd.DataFrame) -> None:
    """Ensemble routing simulation (P8A + T5C-family + Slot A v2)."""
    print("\n\n========================================")
    print("Job L — Ensemble routing simulation on lockbox real")
    print("========================================")
    print(f"N lockbox real frames (intersection across all 9 ckpts): {len(merged)}")

    # Compute per-ckpt over-fire bool at each ckpt's τ
    for ckpt, slug, tau in ALL_CKPTS:
        merged[f"_over_{ckpt}"] = (merged[f"prob_{ckpt}"] >= tau).astype(int)

    # Per-ckpt FPR
    per_ckpt_fpr = {}
    for ckpt, slug, tau in ALL_CKPTS:
        per_ckpt_fpr[ckpt] = float(merged[f"_over_{ckpt}"].mean())

    # Ensemble rules
    rules = []

    # Pair: P8A ∧ T5C (BOTH must fire)
    p8a = "P8A_REFERENCE_STEP5000"
    pairs = [
        ("P8A", "T5C_PERIODIC_STEP3500"),
        ("P8A", "SLOT_A_ANCHOR_AWARE_STEP3500"),
        ("P8A", "SLOT_A_ANCHOR_AWARE_STEP1500"),  # the ultra-low-FPR mid-step
        ("P8A", "SLOT_B_6AXIS_GRL_STEP3500"),  # Slot β (highest viso)
        ("P8A", "SLOT_B_REAL_REBAL_STEP3500"),
        ("T5C_PERIODIC_STEP3500", "SLOT_A_ANCHOR_AWARE_STEP3500"),
    ]
    for a_key, b_key in pairs:
        a_full = p8a if a_key == "P8A" else a_key
        col_a = f"_over_{a_full}"
        col_b = f"_over_{b_key}"
        # AND-ensemble: both fire → over_fire
        and_fpr = float(((merged[col_a] == 1) & (merged[col_b] == 1)).mean())
        or_fpr = float(((merged[col_a] == 1) | (merged[col_b] == 1)).mean())
        rules.append({
            "rule": f"{a_key} AND {b_key}",
            "fpr": and_fpr,
            "fpr_vs_single_a": and_fpr - per_ckpt_fpr[a_full],
            "fpr_vs_single_b": and_fpr - per_ckpt_fpr[b_key],
            "kind": "AND",
        })
        rules.append({
            "rule": f"{a_key} OR  {b_key}",
            "fpr": or_fpr,
            "fpr_vs_single_a": or_fpr - per_ckpt_fpr[a_full],
            "fpr_vs_single_b": or_fpr - per_ckpt_fpr[b_key],
            "kind": "OR",
        })

    rules_df = pd.DataFrame(rules)
    rules_df.to_csv(OUT / "ext_job_l_ensemble_rules.csv", index=False)
    print("\n=== Pairwise ensemble FPR (real) ===")
    print(rules_df.to_string(index=False))

    # Now compute the SAME rules' effect on LOCKBOX FAKE recall to see the trade.
    # Need fake-side frames_reports — load them quickly.
    print("\n=== Loading lockbox fake frames_reports for both scorecards ===")
    fake_pieces = []
    for cks in [common_overnight.CKPTS, CKPTS_AUTO]:
        cache = common_overnight.GCS_CACHE if cks == common_overnight.CKPTS else GCS_AUTO
        # Lockbox fake frames_reports
        for ckpt, slug, tau in cks:
            p = cache / f"teams_fake_all_lockbox_{slug}_frames_report.csv"
            if p.exists():
                df = pd.read_csv(p)
                df = df.rename(columns={"frame_prob": f"prob_{ckpt}"})
                fake_pieces.append((ckpt, df[["frame_path", "label", f"prob_{ckpt}"]]))
    if fake_pieces:
        fake_merged = fake_pieces[0][1].copy()
        for ckpt, piece in fake_pieces[1:]:
            fake_merged = fake_merged.merge(piece.drop(columns=["label"]), on="frame_path", how="outer")
        print(f"Lockbox fake merged: {len(fake_merged)} rows; cols: {len(fake_merged.columns)}")
        for ckpt, slug, tau in ALL_CKPTS:
            sc = f"prob_{ckpt}"
            if sc in fake_merged.columns:
                fake_merged[f"_over_{ckpt}"] = (fake_merged[sc] >= tau).astype(int)

        fake_n = len(fake_merged)
        per_ckpt_recall = {ckpt: float(fake_merged[f"_over_{ckpt}"].mean()) for ckpt, _, _ in ALL_CKPTS if f"_over_{ckpt}" in fake_merged.columns}

        ens_recall_rows = []
        for a_key, b_key in pairs:
            a_full = p8a if a_key == "P8A" else a_key
            ca, cb = f"_over_{a_full}", f"_over_{b_key}"
            if ca not in fake_merged.columns or cb not in fake_merged.columns:
                continue
            and_rec = float(((fake_merged[ca] == 1) & (fake_merged[cb] == 1)).mean())
            or_rec = float(((fake_merged[ca] == 1) | (fake_merged[cb] == 1)).mean())
            ens_recall_rows.append({
                "rule": f"{a_key} AND {b_key}",
                "fake_recall": and_rec,
                "recall_vs_single_a": and_rec - per_ckpt_recall[a_full],
                "kind": "AND",
            })
            ens_recall_rows.append({
                "rule": f"{a_key} OR  {b_key}",
                "fake_recall": or_rec,
                "recall_vs_single_a": or_rec - per_ckpt_recall[a_full],
                "kind": "OR",
            })
        ens_rec = pd.DataFrame(ens_recall_rows)
        ens_rec.to_csv(OUT / "ext_job_l_ensemble_recall.csv", index=False)
        print("\n=== Pairwise ensemble FAKE recall ===")
        print(ens_rec.to_string(index=False))

        # Combine FPR + recall
        combo = rules_df.merge(ens_rec[["rule", "fake_recall"]], on="rule", how="left")
        combo["composite_k1"] = combo["fake_recall"] - 1.0 * combo["fpr"]
        combo["composite_k2"] = combo["fake_recall"] - 2.0 * combo["fpr"]
        combo = combo.sort_values("composite_k1", ascending=False)
        combo.to_csv(OUT / "ext_job_l_ensemble_combo.csv", index=False)
        print("\n=== Combined (FPR, recall, composites) — ranked by k=1 composite ===")
        print(combo.to_string(index=False))


def main() -> int:
    log_f = LOGS / "extension_2026-05-20.log"
    sys.stdout = open(log_f, "w", buffering=1)
    sys.stderr = sys.stdout

    print("=== Extension diagnostics 2026-05-20 ===")
    print("Auto-mode scorecard (missed yesterday): 6 ckpts including Slot A v2 + Slot B real_rebal")
    print()

    merged = load_all_lockbox_real()
    bim = common_overnight.load_bimodal_partition()
    job_a_extended(merged, bim)
    job_b_extended_ranking()
    job_l_ensemble_routing(merged)
    print("\n=== Extension complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
