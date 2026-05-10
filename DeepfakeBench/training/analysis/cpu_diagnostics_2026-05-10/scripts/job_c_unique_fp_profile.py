"""Job C — Profile catastrophic-FP frame clusters per ckpt.

Joins unique catastrophic-FP frames (where one ckpt scores >0.9 alone) with
the IQ atlas + identity metadata to characterize each ckpt's failure mode:
  - Identity / session distribution
  - IQ axis profile (lap_var, min_dim, color_a_dev, sat_mean, luma_mean)
  - IQ-gate retention at multiple thresholds
  - Comparison to the broad cohort statistics

Critical question: are P8A's 11 unique catastrophic-FPs gate-shielded
(low-IQ frames the deployment IQ gate would abstain on) or production-relevant
(passing the gate, structural risk)?

Inputs:
  - outputs/job_b_unique_FPs_*.csv (per-ckpt unique catastrophic-FP frame paths)
  - iq_data_atlas_2026-05-08/outputs/per_frame.parquet (IQ axes per frame)

Outputs:
  - outputs/job_c_fp_profile_per_ckpt.csv (rows = ckpt × identity_session × axis stats)
  - outputs/job_c_fp_profile_summary.csv (per-ckpt aggregate)
  - outputs/job_c_fp_gate_retention.csv (gate retention rates)
  - outputs/job_c_fp_PROFILE.md (human-readable summary)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT = ROOT / "analysis/cpu_diagnostics_2026-05-10/outputs"
ATLAS = ROOT / "analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet"

CKPTS = {
    "P8A": "p8a_only",
    "E2B": "e2b_only",
    "T3_S1_step1500": "t3_step1500_only",
    "T3_S1_step2500": "t3_step2500_only",
    "shared_all_4": "shared_all_4",
}

# Gate thresholds
GATES = {
    "very_lenient": (30, 100),
    "lenient": (50, 150),
    "medium": (100, 200),
    "strict": (200, 200),
}

CHRONIC_6 = ["Roy_D", "PC_Generator", "bla_bla_chow", "Md_noyn_Sharker", "Test_Cam", "Xiang_Xiang2_Feng"]

IDENTITY_RE = re.compile(r"/real/([A-Za-z_0-9]+__s\d+)_")


def parse_identity_session(path: str) -> tuple[str, str]:
    m = IDENTITY_RE.search(path)
    if not m:
        return ("unknown", "unknown")
    full = m.group(1)
    parts = full.rsplit("__s", 1)
    if len(parts) == 2:
        return (parts[0], "s" + parts[1])
    return (full, "s?")


def is_chronic(identity: str) -> bool:
    base = identity.replace("_", "").lower()
    for c in CHRONIC_6:
        if c.replace("_", "").lower() in base:
            return True
    # Special case for dor variants (Dor identity family)
    if base.startswith("dor") or base.startswith("xiang"):
        return True
    return False


def passes_gate(lap_var: float, min_dim: float, gate_lap: int, gate_min_dim: int) -> bool:
    if pd.isna(lap_var) or pd.isna(min_dim):
        return False
    return lap_var >= gate_lap and min_dim >= gate_min_dim


def main():
    print(f"Loading IQ atlas...")
    atlas = pd.read_parquet(ATLAS)
    print(f"  atlas rows={len(atlas)}, columns={list(atlas.columns)[:15]}")

    # Determine the right join key column
    if "frame_path" in atlas.columns:
        join_col = "frame_path"
    elif "path" in atlas.columns:
        join_col = "path"
    elif "gcs_uri" in atlas.columns:
        join_col = "gcs_uri"
    else:
        # Try first string column
        for c in atlas.columns:
            if atlas[c].dtype == object:
                join_col = c
                break
    print(f"  using join_col={join_col}")

    # IQ axes we want
    iq_cols = ["lap_var", "min_dim", "color_a_dev", "color_b_dev", "luma_mean", "saturation_mean"]
    available_iq = [c for c in iq_cols if c in atlas.columns]
    print(f"  available IQ cols: {available_iq}")

    rows = []
    summaries = []
    gate_rows = []

    for ckpt_label, suffix in CKPTS.items():
        path = OUT / f"job_b_unique_FPs_{suffix}.csv"
        if not path.exists():
            print(f"  SKIP {ckpt_label}: {path} not found")
            continue
        df = pd.read_csv(path)
        n = len(df)
        if n == 0:
            print(f"  {ckpt_label}: 0 unique FPs (nothing to profile)")
            summaries.append({
                "ckpt": ckpt_label, "n_unique_fps": 0,
                "n_chronic_6": 0, "n_non_chronic": 0,
                "lap_var_p50": None, "min_dim_p50": None,
                "color_a_dev_p50": None, "saturation_mean_p50": None,
            })
            continue
        print(f"\n=== {ckpt_label} ({n} unique FPs) ===")

        # Parse identity & session
        df[["identity", "session"]] = df["frame_path"].apply(
            lambda p: pd.Series(parse_identity_session(p))
        )
        df["is_chronic"] = df["identity"].apply(is_chronic)

        # Join with IQ atlas
        joined = df.merge(atlas[[join_col] + available_iq],
                          left_on="frame_path", right_on=join_col, how="left")
        n_joined = joined[available_iq[0]].notna().sum()
        print(f"  IQ join rate: {n_joined}/{n} ({100*n_joined/n:.0f}%)")

        # Per-identity-session breakdown
        breakdown = joined.groupby(["identity", "session", "is_chronic"]).agg(
            n=("frame_path", "count"),
            mean_score=("P8A" if ckpt_label == "P8A" else "E2B" if ckpt_label == "E2B"
                       else "T3_S1_step1500" if "step1500" in ckpt_label
                       else "T3_S1_step2500", "mean"),
            **{f"{c}_p50": (c, "median") for c in available_iq},
        ).reset_index()
        for _, r in breakdown.iterrows():
            print(f"    {r['identity']:30s} session={r['session']:5s} chronic={r['is_chronic']}  n={int(r['n']):3d}  lap_var_p50={r.get('lap_var_p50', np.nan):.1f}  min_dim_p50={r.get('min_dim_p50', np.nan):.0f}  color_a_dev_p50={r.get('color_a_dev_p50', np.nan):.2f}")
            row = {"ckpt": ckpt_label, **r.to_dict()}
            rows.append(row)

        # Aggregate summary
        n_chronic = joined["is_chronic"].sum()
        n_non = (~joined["is_chronic"]).sum()
        summary = {
            "ckpt": ckpt_label, "n_unique_fps": n,
            "n_chronic_6": int(n_chronic), "n_non_chronic": int(n_non),
            "pct_chronic": 100.0 * n_chronic / n,
        }
        for c in available_iq:
            summary[f"{c}_p50"] = float(joined[c].median()) if joined[c].notna().any() else None
            summary[f"{c}_mean"] = float(joined[c].mean()) if joined[c].notna().any() else None
        summaries.append(summary)
        print(f"\n  Summary: {n_chronic}/{n} chronic-6 ({100*n_chronic/n:.0f}%)")
        for c in available_iq:
            if joined[c].notna().any():
                print(f"    {c}: p50={joined[c].median():.2f}, mean={joined[c].mean():.2f}")

        # Gate retention analysis
        for gate_name, (lap_thr, mdim_thr) in GATES.items():
            passing = joined.apply(
                lambda r: passes_gate(r.get("lap_var"), r.get("min_dim"), lap_thr, mdim_thr), axis=1
            ).sum()
            retain_pct = 100.0 * passing / n_joined if n_joined > 0 else 0
            print(f"    Gate {gate_name} ({lap_thr},{mdim_thr}): retains {int(passing)}/{n_joined} = {retain_pct:.1f}%")
            gate_rows.append({
                "ckpt": ckpt_label, "gate": gate_name,
                "lap_thr": lap_thr, "min_dim_thr": mdim_thr,
                "n_joined": int(n_joined), "n_passing": int(passing),
                "retention_pct": retain_pct,
            })

    # Save outputs
    pd.DataFrame(rows).to_csv(OUT / "job_c_fp_profile_per_ckpt.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "job_c_fp_profile_summary.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(OUT / "job_c_fp_gate_retention.csv", index=False)
    print(f"\nWrote outputs to {OUT}")

if __name__ == "__main__":
    main()
