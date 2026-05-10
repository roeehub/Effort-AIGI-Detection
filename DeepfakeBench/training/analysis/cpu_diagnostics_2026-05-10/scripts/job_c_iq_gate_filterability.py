"""Job C — IQ-gate filterability check.

Question: does the proposed IQ gate (sharpness + min_dim) filter out the
chronic-6 + color-axis identities that T3_step2500 regresses on? Or do
they pass through to be wrongly scored?

If gate filters them: step2500 is shippable under IQ-gate policy.
If gate doesn't filter them: step2500's regression is production-relevant
and P8A or step1500 are safer.

Method:
1. For multiple candidate (lap_var, min_dim) gate thresholds:
2. Apply gate to teams_real_all_dev real cohort
3. Count reals retained / filtered by chronic-6 status + color quartile

Output: filterability table at multiple gate thresholds.

Usage: python job_c_iq_gate_filterability.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT = ROOT / "analysis/cpu_diagnostics_2026-05-10/outputs"
IQ_ATLAS = ROOT / "analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet"

REAL_REPORT = ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/teams_real_all_dev_p8a_reference_step5000_frames_report.csv"

CHRONIC_6 = ["Roy_D", "PC_Generator", "bla_bla_chow", "Md_noyn_Sharker", "Test_Cam", "Xiang_Xiang2_Feng"]


def identity_from_path(p: str) -> str:
    """Extract identity from frame_path like .../{identity}__sX_..."""
    base = Path(p).name
    # Heuristic: identity is the prefix up to '__s' or first '_'
    m = re.match(r"([A-Za-z0-9_]+?)(?:__s\d+|_s\d+|_\d{2,})", base)
    if m:
        return m.group(1)
    return base.split("_")[0]


def main():
    print("Loading IQ atlas + real cohort...")
    iq = pd.read_parquet(IQ_ATLAS)
    real = pd.read_csv(REAL_REPORT, usecols=["frame_path", "frame_prob", "video_id"])
    print(f"  IQ atlas: {len(iq):,} rows; real cohort: {len(real):,} rows")

    # Join on frame_path
    df = real.merge(
        iq[["frame_path", "lap_var", "min_dim", "color_a_dev", "saturation_mean"]],
        on="frame_path", how="left"
    )
    n_with_iq = df["lap_var"].notna().sum()
    print(f"  Joined: {len(df)} rows, {n_with_iq} with IQ ({n_with_iq/len(df)*100:.1f}%)")

    # Identity extraction
    df["identity"] = df["frame_path"].apply(identity_from_path)
    df["is_chronic_6"] = df["identity"].apply(lambda x: any(c in x for c in CHRONIC_6))

    # Color-axis Q4 (top 25% color_a_dev) — Roy_D-equivalent population
    iq_with_color = df.dropna(subset=["color_a_dev"])
    color_q3 = iq_with_color["color_a_dev"].quantile(0.75)
    df["is_color_q4"] = df["color_a_dev"] >= color_q3

    # Chronic-6 frame counts in IQ-with-data subset
    iq_subset = df.dropna(subset=["lap_var", "min_dim"])
    print(f"\nReal cohort breakdown (frames with IQ data, n={len(iq_subset)}):")
    n_chronic = iq_subset["is_chronic_6"].sum()
    n_color_q4 = iq_subset["is_color_q4"].sum()
    print(f"  chronic_6 frames: {n_chronic} ({n_chronic/len(iq_subset)*100:.1f}%)")
    print(f"  color_a_dev Q4 (warm-color, Roy_D-type): {n_color_q4} ({n_color_q4/len(iq_subset)*100:.1f}%)")
    chronic_in_q4 = ((iq_subset["is_chronic_6"]) & (iq_subset["is_color_q4"])).sum()
    print(f"  chronic_6 AND color Q4: {chronic_in_q4}")

    # Per-chronic-identity IQ profile
    print(f"\nChronic-6 per-identity IQ profile (mean):")
    for c in CHRONIC_6:
        sub = iq_subset[iq_subset["identity"].str.contains(c, na=False)]
        if len(sub) == 0:
            continue
        print(f"  {c:20s} n={len(sub):4d}  lap_var={sub['lap_var'].mean():6.1f}  min_dim={sub['min_dim'].mean():6.1f}  color_a_dev={sub['color_a_dev'].mean():6.2f}  sat_mean={sub['saturation_mean'].mean():6.2f}")

    # Apply gates at various thresholds
    print("\n" + "=" * 95)
    print("IQ-gate filterability — at each threshold, count of reals retained vs filtered")
    print("=" * 95)
    gates = [
        ("strict",   {"min_lap_var": 200, "min_min_dim": 200}),
        ("medium",   {"min_lap_var": 100, "min_min_dim": 200}),
        ("lenient",  {"min_lap_var": 50,  "min_min_dim": 150}),
        ("very_lenient", {"min_lap_var": 30, "min_min_dim": 100}),
        ("res_only_200", {"min_lap_var": 0, "min_min_dim": 200}),
        ("sharp_only_100", {"min_lap_var": 100, "min_min_dim": 0}),
    ]

    rows = []
    for name, params in gates:
        keep = (iq_subset["lap_var"] >= params["min_lap_var"]) & (iq_subset["min_dim"] >= params["min_min_dim"])
        kept = iq_subset[keep]
        filt = iq_subset[~keep]
        # Categories: chronic_6, color_q4, neither
        c_kept = kept["is_chronic_6"].sum()
        c_filt = filt["is_chronic_6"].sum()
        cq4_kept = kept["is_color_q4"].sum()
        cq4_filt = filt["is_color_q4"].sum()
        rows.append({
            "gate_name": name,
            "min_lap_var": params["min_lap_var"],
            "min_min_dim": params["min_min_dim"],
            "n_kept": int(len(kept)),
            "n_filtered": int(len(filt)),
            "frac_kept": float(len(kept) / len(iq_subset)),
            "n_chronic_kept": int(c_kept),
            "n_chronic_filtered": int(c_filt),
            "frac_chronic_filtered": float(c_filt / max(1, c_kept + c_filt)),
            "n_color_q4_kept": int(cq4_kept),
            "n_color_q4_filtered": int(cq4_filt),
            "frac_color_q4_filtered": float(cq4_filt / max(1, cq4_kept + cq4_filt)),
        })
        print(f"\nGate '{name}'  lap_var≥{params['min_lap_var']}  min_dim≥{params['min_min_dim']}:")
        print(f"  total reals kept: {len(kept)}/{len(iq_subset)} ({len(kept)/len(iq_subset)*100:.1f}%)")
        print(f"  chronic-6 kept: {c_kept}/{c_kept+c_filt} ({c_kept/(c_kept+c_filt)*100:.1f}% — gate filters {c_filt/(c_kept+c_filt)*100:.1f}%)")
        print(f"  color_a_dev Q4 kept: {cq4_kept}/{cq4_kept+cq4_filt} ({cq4_kept/(cq4_kept+cq4_filt)*100:.1f}% — gate filters {cq4_filt/(cq4_kept+cq4_filt)*100:.1f}%)")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "job_c_iq_gate_filterability.csv", index=False)
    print(f"\nWrote {OUT / 'job_c_iq_gate_filterability.csv'}")

    # Per-chronic-identity gate retention at "medium" gate
    print("\n" + "=" * 95)
    print("Per-chronic-identity gate retention (at lap_var≥100, min_dim≥200):")
    print("=" * 95)
    keep_med = (iq_subset["lap_var"] >= 100) & (iq_subset["min_dim"] >= 200)
    for c in CHRONIC_6:
        ident_mask = iq_subset["identity"].str.contains(c, na=False)
        if not ident_mask.any():
            continue
        sub = iq_subset[ident_mask]
        kept = sub[keep_med]
        filt = sub[~keep_med]
        print(f"  {c:20s} kept: {len(kept):4d}/{len(sub):4d} ({len(kept)/len(sub)*100:.1f}%)  filtered: {len(filt):4d}")


if __name__ == "__main__":
    main()
