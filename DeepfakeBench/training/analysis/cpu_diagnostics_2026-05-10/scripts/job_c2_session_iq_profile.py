"""Job C extension — session-level IQ profile lookup for unique catastrophic-FPs.

The 11 P8A unique catastrophic-FP frames aren't in the IQ atlas (sparse 39%
coverage). But the (identity, session) pairs THEY ARE FROM exist in the atlas
with other frame samples. This script characterizes each unique-FP frame's
SESSION-level IQ profile as a proxy.

The session is the natural unit of capture-pipeline character (one camera, one
lighting setup, one room). If P8A's failure is session-correlated (rather than
random per-frame), the session IQ profile is the relevant quantity.

Outputs:
  - outputs/job_c2_session_iq_profiles.csv (per (ckpt, identity, session))
  - outputs/job_c2_PROFILE.md (human summary)
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
    "T3_S1_step2500": "t3_step2500_only",
    "shared_all_4": "shared_all_4",
}

CHRONIC_6 = {"Roy_D", "PC_Generator", "bla_bla_chow", "Md_noyn_Sharker", "Test_Cam", "Xiang_Xiang2_Feng"}
DOR_FAMILY = {"dor_shkedi", "healthy_dor", "dor"}

IDENTITY_RE = re.compile(r"/real/([A-Za-z_0-9]+__s\d+)_")
IDENTITY_NOSESSION_RE = re.compile(r"/real/([A-Za-z_0-9]+)__s")


def parse_id_session(p: str) -> tuple[str, str]:
    """Extract identity__session token from frame path."""
    m = re.search(r"/real/([A-Za-z_0-9]+__s\d+)_", p)
    if m:
        return m.group(1), "matched"
    # Fallback parse
    m2 = re.search(r"/real/([A-Za-z_0-9]+)_", p)
    if m2:
        return m2.group(1), "fallback"
    return ("unknown", "fail")


def is_chronic(id_session: str) -> bool:
    base = id_session.split("__s")[0]
    if base in CHRONIC_6:
        return True
    if base in DOR_FAMILY:
        return True
    return False


def main():
    print("Loading IQ atlas...")
    atlas = pd.read_parquet(ATLAS)
    # Add an id_session column from frame_path
    atlas["id_session"] = atlas["frame_path"].apply(lambda p: parse_id_session(p)[0])
    print(f"  atlas rows={len(atlas)}, unique id_sessions={atlas['id_session'].nunique()}")

    # Keep only relevant cols
    iq_cols = ["lap_var", "min_dim", "color_a_dev", "color_b_dev", "luma_mean", "saturation_mean"]
    avail = [c for c in iq_cols if c in atlas.columns]

    rows = []
    md_lines = ["# Job C2 — Session-level IQ profile of unique catastrophic FPs",
                "",
                "Pulls IQ atlas statistics for each (identity, session) pair containing a unique catastrophic-FP frame. The unique-FP frames themselves may not be in the atlas (39% coverage), but the same (identity, session) pair has other atlas-sampled frames whose distribution is informative.",
                ""]

    for ckpt_label, suffix in CKPTS.items():
        path = OUT / f"job_b_unique_FPs_{suffix}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        n = len(df)
        if n == 0:
            continue
        df["id_session"] = df["frame_path"].apply(lambda p: parse_id_session(p)[0])
        df["is_chronic"] = df["id_session"].apply(is_chronic)

        unique_sessions = df["id_session"].value_counts()
        md_lines.append(f"## {ckpt_label} — {n} unique catastrophic-FPs")
        md_lines.append("")
        md_lines.append("| identity__session | n_FPs | chronic | n_atlas_samples | lap_var p50 | min_dim p50 | color_a_dev p50 | sat p50 | passes medium_gate? |")
        md_lines.append("|---|---:|:---:|---:|---:|---:|---:|---:|:---:|")

        for id_sess, n_fps in unique_sessions.items():
            chronic = is_chronic(id_sess)
            atlas_subset = atlas[atlas["id_session"] == id_sess]
            n_atlas = len(atlas_subset)
            row = {
                "ckpt": ckpt_label,
                "id_session": id_sess,
                "n_unique_FPs": int(n_fps),
                "is_chronic_6": chronic,
                "n_atlas_samples": int(n_atlas),
            }
            if n_atlas > 0:
                for c in avail:
                    if c in atlas_subset.columns:
                        row[f"{c}_p50"] = float(atlas_subset[c].median())
                        row[f"{c}_mean"] = float(atlas_subset[c].mean())
                # Gate retention at medium (lap_var>=100, min_dim>=200)
                gate_pass = ((atlas_subset["lap_var"] >= 100) & (atlas_subset["min_dim"] >= 200)).mean()
                row["medium_gate_retention"] = float(gate_pass)
                # Strict
                strict_pass = ((atlas_subset["lap_var"] >= 200) & (atlas_subset["min_dim"] >= 200)).mean()
                row["strict_gate_retention"] = float(strict_pass)
                med_str = f"{100*gate_pass:.0f}%" if gate_pass > 0 else "0%"
                md_lines.append(
                    f"| {id_sess} | {n_fps} | {'Y' if chronic else 'N'} | {n_atlas} | "
                    f"{row.get('lap_var_p50', np.nan):.0f} | {row.get('min_dim_p50', np.nan):.0f} | "
                    f"{row.get('color_a_dev_p50', np.nan):.1f} | {row.get('saturation_mean_p50', np.nan):.0f} | "
                    f"{med_str} |"
                )
            else:
                md_lines.append(f"| {id_sess} | {n_fps} | {'Y' if chronic else 'N'} | 0 | - | - | - | - | n/a |")
            rows.append(row)
        md_lines.append("")

    pd.DataFrame(rows).to_csv(OUT / "job_c2_session_iq_profiles.csv", index=False)
    (OUT / "job_c2_PROFILE.md").write_text("\n".join(md_lines))
    print(f"\nWrote {len(rows)} rows to job_c2_session_iq_profiles.csv")
    print(f"Wrote markdown to job_c2_PROFILE.md")
    # Print summary
    print("\n" + "\n".join(md_lines[3:]))

if __name__ == "__main__":
    main()
