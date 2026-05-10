"""Job C2 — Post-IQ-gate FPR per ckpt.

The actually-deployable question: AFTER the IQ gate filters, what's each
ckpt's FPR on the surviving real cohort? At multiple gate thresholds.

This separates "FPR on raw eval substrate" from "FPR on production-honest
substrate (post gate)" — the latter is what production users would see.

Usage: python job_c2_post_gate_fpr.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT = ROOT / "analysis/cpu_diagnostics_2026-05-10/outputs"
IQ_ATLAS = ROOT / "analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet"

REAL_REPORTS = {
    "P8A": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/teams_real_all_dev_p8a_reference_step5000_frames_report.csv",
    "E2B": ROOT / "analysis/cpu_followups_2026-05-04/raw_reports/teams_real_all_dev_e2b_top_n_step3200_frames_report.csv",
    "T3_S1_step1500": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step1500/teams_real_all_dev_t3_slot1_periodic_step1500_frames_report.csv",
    "T3_S1_step2500": ROOT / "analysis/cpu_diagnostics_2026-05-09/_t3_f4_inputs/slot1_step2500/teams_real_all_dev_t3_slot1_periodic_step2500_frames_report.csv",
}

GATES = [
    ("none",         {"min_lap_var": 0,   "min_min_dim": 0}),
    ("very_lenient", {"min_lap_var": 30,  "min_min_dim": 100}),
    ("lenient",      {"min_lap_var": 50,  "min_min_dim": 150}),
    ("medium",       {"min_lap_var": 100, "min_min_dim": 200}),
    ("strict",       {"min_lap_var": 200, "min_min_dim": 200}),
]


def main():
    iq = pd.read_parquet(IQ_ATLAS)[["frame_path", "lap_var", "min_dim", "color_a_dev"]]
    rows = []

    # Determine color_q4 threshold globally
    color_q3_global = iq["color_a_dev"].quantile(0.75)

    for ckpt, path in REAL_REPORTS.items():
        real = pd.read_csv(path, usecols=["frame_path", "frame_prob"])
        joined = real.merge(iq, on="frame_path", how="left")
        with_iq = joined.dropna(subset=["lap_var", "min_dim"]).copy()
        with_iq["is_color_q4"] = with_iq["color_a_dev"] >= color_q3_global

        for gate_name, params in GATES:
            mask = (with_iq["lap_var"] >= params["min_lap_var"]) & (with_iq["min_dim"] >= params["min_min_dim"])
            kept = with_iq[mask]
            n_kept = len(kept)
            if n_kept == 0:
                continue

            # FPR @ multiple thresholds, ON THE GATE-PASSING SUBSET
            for tau in [0.5, 0.7, 0.9]:
                fpr = (kept["frame_prob"] > tau).mean()
                rows.append({
                    "ckpt": ckpt,
                    "gate": gate_name,
                    "min_lap_var": params["min_lap_var"],
                    "min_min_dim": params["min_min_dim"],
                    "n_kept": n_kept,
                    "tau": tau,
                    "fpr_overall": float(fpr),
                    "n_above_tau": int((kept["frame_prob"] > tau).sum()),
                })

            # FPR on color_q4 subset of gate-passing reals
            color_q4_kept = kept[kept["is_color_q4"]]
            if len(color_q4_kept) > 0:
                for tau in [0.5, 0.7, 0.9]:
                    fpr = (color_q4_kept["frame_prob"] > tau).mean()
                    rows.append({
                        "ckpt": ckpt,
                        "gate": f"{gate_name} + color_q4_only",
                        "min_lap_var": params["min_lap_var"],
                        "min_min_dim": params["min_min_dim"],
                        "n_kept": len(color_q4_kept),
                        "tau": tau,
                        "fpr_overall": float(fpr),
                        "n_above_tau": int((color_q4_kept["frame_prob"] > tau).sum()),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "job_c2_post_gate_fpr.csv", index=False)
    print(f"Wrote {OUT / 'job_c2_post_gate_fpr.csv'}")

    # Headline: FPR @ tau=0.5 on full gate-passing real cohort
    print("\n" + "=" * 95)
    print("FPR @ tau=0.5 on gate-PASSING real cohort (production-honest reals after IQ gate)")
    print("=" * 95)
    sub = df[(df["tau"] == 0.5) & (~df["gate"].str.contains("color_q4"))]
    pivot = sub.pivot(index="gate", columns="ckpt", values="fpr_overall")[["P8A", "E2B", "T3_S1_step1500", "T3_S1_step2500"]]
    pivot = pivot.reindex(["none", "very_lenient", "lenient", "medium", "strict"])
    print((pivot * 100).round(2).astype(str) + "%")

    # Same for color_q4 (warm-color subset) — production-honest warm-color reals
    print("\n" + "=" * 95)
    print("FPR @ tau=0.5 on gate-passing AND color_a_dev Q4 (warm-color, Roy_D-type)")
    print("=" * 95)
    sub = df[(df["tau"] == 0.5) & (df["gate"].str.contains("color_q4"))]
    pivot = sub.pivot(index="gate", columns="ckpt", values="fpr_overall")[["P8A", "E2B", "T3_S1_step1500", "T3_S1_step2500"]]
    print((pivot * 100).round(2).astype(str) + "%")

    # Reduction in FPR from gate (ratio of pre-gate to post-gate)
    print("\n" + "=" * 95)
    print("FPR reduction from gating (pre-gate FPR / post-gate FPR), tau=0.5")
    print("=" * 95)
    pre = df[(df["tau"] == 0.5) & (df["gate"] == "none")][["ckpt", "fpr_overall"]].set_index("ckpt")
    for gate_name in ["very_lenient", "lenient", "medium", "strict"]:
        post = df[(df["tau"] == 0.5) & (df["gate"] == gate_name)][["ckpt", "fpr_overall"]].set_index("ckpt")
        ratio = pre["fpr_overall"] / post["fpr_overall"].replace(0, np.nan)
        print(f"\nGate '{gate_name}':")
        for ckpt in ["P8A", "E2B", "T3_S1_step1500", "T3_S1_step2500"]:
            pre_v = pre.loc[ckpt, "fpr_overall"]
            post_v = post.loc[ckpt, "fpr_overall"] if ckpt in post.index else float("nan")
            print(f"  {ckpt:18s}: pre={pre_v*100:.2f}%  post={post_v*100:.2f}%  reduction={pre_v/(post_v if post_v > 0 else 1):.2f}x")


if __name__ == "__main__":
    main()
