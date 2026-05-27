"""Extension: add may6 production-drift FPR to the deployment Pareto analysis.

For each ckpt × candidate operating point (τ chosen to give a target
lockbox_real_fpr), compute may6 fired count / 92. This is the third
deployment-relevant cohort beyond dev_real / lockbox_real.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT_DIR = REPO / "analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_h_deployment_pareto/outputs"
FIG_DIR = REPO / "analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_h_deployment_pareto/figs"

# Per-frame may6 score files
MAY6_FILES = {
    "P8A": REPO / "analysis/xinhe_cross_camera_audit_2026-05-06/outputs/scores_P8A.csv",
    "E2B": REPO / "analysis/xinhe_cross_camera_audit_2026-05-06/outputs/scores_E2B.csv",
    "T5C": REPO / "analysis/r13_overnight_may6_retest_2026-05-13/outputs/scores_T5C_PERIODIC_STEP3500.csv",
    "SlotAv2": REPO / "analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_f_may6_retest/outputs/scores_slot_a_v2_step3500_may6.csv",
}

# Lockbox per-video reports
LOCKBOX_REAL = {
    "P8A": REPO / "analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_b_bootstrap_ci/data/teams_real_all_lockbox_p8a_reference_step5000_videos_report.csv",
    "T5C": REPO / "analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_d_per_identity_fp/data/teams_real_all_lockbox_t5c_periodic_step3500_videos_report.csv",
    "SlotAv2": REPO / "analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_b_bootstrap_ci/data/teams_real_all_lockbox_slot_a_anchor_aware_step3500_videos_report.csv",
    "E2B": "AGG",  # special handling — E2B is per-frame cache, aggregate
}
LOCKBOX_FAKE = {
    "P8A": REPO / "analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_h_deployment_pareto/data/teams_fake_all_lockbox_p8a_reference_step5000_videos_report.csv",
    "T5C": REPO / "analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_h_deployment_pareto/data/teams_fake_all_lockbox_t5c_periodic_step3500_videos_report.csv",
    "SlotAv2": REPO / "analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_h_deployment_pareto/data/teams_fake_all_lockbox_slot_a_anchor_aware_step3500_videos_report.csv",
    "E2B": "AGG",
}


def load_e2b_lockbox(suite_name):
    path = REPO / f"analysis/iq_shortcut_decomp_2026-05-08/scores_cache/E2B_TOP_N_STEP3200__{suite_name}.csv"
    raw = pd.read_csv(path)
    return (raw.groupby(["method", "label", "video_id"], dropna=False)
              .agg(avg_video_prob=("frame_prob", "mean"))
              .reset_index())


def load_lockbox_scores(ckpt: str, kind: str):
    src = (LOCKBOX_REAL if kind == "real" else LOCKBOX_FAKE)[ckpt]
    if src == "AGG":
        return load_e2b_lockbox(f"teams_{kind}_all_lockbox")["avg_video_prob"].values
    return pd.read_csv(src)["avg_video_prob"].values


def load_may6_scores(ckpt: str):
    df = pd.read_csv(MAY6_FILES[ckpt])
    sub = df[df["population"] == "may6_falseflag"]
    return sub["prob_fake"].values


# Sweep τ on each ckpt's may6 scores; also note its lockbox-cal τ for target_fpr
def main():
    ckpts = list(MAY6_FILES.keys())
    target_fprs = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    # 1) For each ckpt, compute may6_fired at lockbox-cal τ for each target FPR
    rows = []
    raw_curves = []
    for ckpt in ckpts:
        lock_real = load_lockbox_scores(ckpt, "real")
        lock_fake = load_lockbox_scores(ckpt, "fake")
        may6 = load_may6_scores(ckpt)
        n_may6 = len(may6)
        print(f"  {ckpt}: n_lockbox_real={len(lock_real)}, n_lockbox_fake={len(lock_fake)}, n_may6={n_may6}")

        sorted_desc = np.sort(lock_real)[::-1]
        n = len(sorted_desc)
        for tfp in target_fprs:
            n_above = max(1, int(tfp * n))
            tau = float(sorted_desc[n_above - 1])
            achieved_fpr = float((lock_real >= tau).mean())
            lock_rec = float((lock_fake >= tau).mean())
            may6_fired = int((may6 >= tau).sum())
            rows.append({
                "ckpt": ckpt,
                "target_lockbox_fpr": tfp,
                "tau": tau,
                "achieved_lockbox_fpr": achieved_fpr,
                "lockbox_fake_recall": lock_rec,
                "may6_fired": may6_fired,
                "may6_total": n_may6,
                "may6_fpr": may6_fired / n_may6,
            })

        # Full τ-sweep for plotting
        for tau in np.linspace(0.01, 0.999, 200):
            raw_curves.append({
                "ckpt": ckpt,
                "tau": float(tau),
                "lockbox_real_fpr": float((lock_real >= tau).mean()),
                "lockbox_fake_recall": float((lock_fake >= tau).mean()),
                "may6_fpr": float((may6 >= tau).mean()),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "may6_at_operating_points.csv", index=False)

    df_curves = pd.DataFrame(raw_curves)
    df_curves.to_csv(OUT_DIR / "may6_sweep_curves.csv", index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("OPERATING-POINT TABLE — may6_fpr at each lockbox-cal τ")
    print("=" * 80)
    print(f"{'ckpt':<10} {'target_lockbox_fpr':>18} {'τ':>8} {'achieved_fpr':>13} {'lockbox_recall':>15} {'may6_fired':>12} {'may6_fpr':>10}")
    for _, r in df.iterrows():
        print(f"{r['ckpt']:<10} {r['target_lockbox_fpr']:>18.4f} {r['tau']:>8.4f} {r['achieved_lockbox_fpr']:>13.4f} {r['lockbox_fake_recall']:>15.4f} {int(r['may6_fired']):>9}/92 {r['may6_fpr']:>10.4f}")

    # Plot: 3-axis trade-off
    # X = lockbox_fake_recall (the "good" metric — want high)
    # Y1 = may6_fpr (cost on production-drift — want low)
    plt.figure(figsize=(11, 7))
    colors = {"P8A": "C0", "SlotAv2": "C1", "T5C": "C2", "E2B": "C3"}
    for ckpt in ckpts:
        sub = df_curves[df_curves["ckpt"] == ckpt].sort_values("lockbox_fake_recall")
        plt.plot(sub["lockbox_fake_recall"], sub["may6_fpr"], "-",
                 color=colors[ckpt], linewidth=2, label=ckpt, alpha=0.85)
    # Annotate the lockbox-cal=10% operating points
    for _, r in df[df["target_lockbox_fpr"] == 0.10].iterrows():
        plt.plot(r["lockbox_fake_recall"], r["may6_fpr"], "o",
                 color=colors[r["ckpt"]], markersize=10, markeredgecolor="black")
    plt.xlabel("lockbox_fake_recall (good — higher is better)")
    plt.ylabel("may6_fpr (bad — lower is better)")
    plt.title("Production-drift cost vs lockbox catching power\n(dots: τ at lockbox_fpr=10%)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_may6_vs_lockbox_recall.png", dpi=110)
    plt.close()

    # Plot: lockbox_fpr → may6_fpr correlation
    plt.figure(figsize=(10, 7))
    for ckpt in ckpts:
        sub = df_curves[df_curves["ckpt"] == ckpt].sort_values("lockbox_real_fpr")
        plt.plot(sub["lockbox_real_fpr"], sub["may6_fpr"], "-",
                 color=colors[ckpt], linewidth=2, label=ckpt, alpha=0.85)
    plt.xlabel("lockbox_real_fpr (calibration)")
    plt.ylabel("may6_fpr (production-drift fire rate)")
    plt.title("Lockbox FPR ↔ may6 FPR correlation per ckpt")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 0.30)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_lockbox_fpr_to_may6_fpr.png", dpi=110)
    plt.close()


if __name__ == "__main__":
    main()
