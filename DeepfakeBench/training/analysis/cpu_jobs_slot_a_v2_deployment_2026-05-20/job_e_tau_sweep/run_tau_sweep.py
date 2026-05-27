"""Job E — τ-sweep operating-point analysis on the 800-frame manual-canary panel.

Per ckpt, sweep τ from 0.30 to 0.99 in steps of 0.005 and compute per-suite
recall (for fake suites) and FPR (for real suites) at each τ. Produces
operating-point comparison plots: each ckpt has a different (FPR, recall)
curve; the curves let us read off "what τ gives Slot A v2 the same FPR as
P8A, and what recall does it get there?".

Inputs:
  analysis/manual_canary_2026-05-20/outputs/<CKPT>.scores.npy (per-frame
    prob_fake for 8 ckpts × 800 frames each)
  analysis/manual_canary_2026-05-20/frames_meta.parquet (per-frame label /
    cohort / suite / base_identity)

Outputs:
  analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_e_tau_sweep/outputs/
    tau_sweep_table.csv          - per-ckpt × τ × suite metrics
    operating_points.csv         - per-ckpt: τ that matches P8A's FPR, recall at that τ
    figs/                        - matplotlib plots (one per metric)
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
CANARY_DIR = REPO / "analysis/manual_canary_2026-05-20"
OUT_DIR = REPO / "analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_e_tau_sweep/outputs"
FIG_DIR = OUT_DIR / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


CKPT_FILES = {
    "P8A_REFERENCE_STEP5000": "P8A_REFERENCE_STEP5000.scores.npy",
    "T5C_PERIODIC_STEP3500": "T5C_PERIODIC_STEP3500.scores.npy",
    "SLOT_A_V2_STEP3500": "SLOT_A_V2_STEP3500.scores.npy",
    "SLOT_1_6AXIS_ANCHOR_STEP1500": "SLOT_1_6AXIS_ANCHOR_STEP1500.scores.npy",
    "SLOT_1_6AXIS_ANCHOR_STEP2500": "SLOT_1_6AXIS_ANCHOR_STEP2500.scores.npy",
    "SLOT_1_6AXIS_ANCHOR_STEP3500": "SLOT_1_6AXIS_ANCHOR_STEP3500.scores.npy",
    "SLOT_2_LORA_L8_L9_STEP2500": "SLOT_2_LORA_L8_L9_STEP2500.scores.npy",
    "SLOT_3_5AXIS_NOLUMA_STEP3500": "SLOT_3_5AXIS_NOLUMA_STEP3500.scores.npy",
}


def main():
    meta = pd.read_parquet(CANARY_DIR / "frames_meta.parquet")
    print(f"Loaded meta: {meta.shape}, columns={meta.columns.tolist()}")
    print(f"Unique cohorts: {meta['cohort'].unique().tolist()}")
    print(f"Unique suites: {meta['suite'].unique().tolist()}")
    print(f"label distribution: {meta['label'].value_counts().to_dict()}")

    ckpt_scores = {}
    for key, fname in CKPT_FILES.items():
        path = CANARY_DIR / "outputs" / fname
        if not path.exists():
            print(f"WARN: missing {path}")
            continue
        s = np.load(path)
        if len(s) != len(meta):
            print(f"WARN: {key} has {len(s)} scores vs {len(meta)} meta rows")
        ckpt_scores[key] = s

    print(f"Loaded {len(ckpt_scores)} ckpts")

    # τ sweep
    taus = np.arange(0.30, 0.991, 0.005)
    print(f"Sweeping {len(taus)} τ values from {taus[0]:.3f} to {taus[-1]:.3f}")

    suites = sorted(meta["suite"].unique())
    rows = []
    for ckpt, scores in ckpt_scores.items():
        for suite in suites:
            mask = meta["suite"] == suite
            suite_scores = scores[mask.values]
            suite_labels = meta.loc[mask, "label"].values  # 0=real, 1=fake
            n_real = int((suite_labels == 0).sum())
            n_fake = int((suite_labels == 1).sum())
            for tau in taus:
                n_above = int((suite_scores >= tau).sum())
                n_real_above = int(((suite_scores >= tau) & (suite_labels == 0)).sum())
                n_fake_above = int(((suite_scores >= tau) & (suite_labels == 1)).sum())
                rows.append({
                    "ckpt": ckpt,
                    "suite": suite,
                    "tau": float(tau),
                    "n_real": n_real,
                    "n_fake": n_fake,
                    "n_above_tau": n_above,
                    "real_fpr": n_real_above / max(n_real, 1) if n_real > 0 else np.nan,
                    "fake_recall": n_fake_above / max(n_fake, 1) if n_fake > 0 else np.nan,
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "tau_sweep_table.csv", index=False)
    print(f"Wrote {len(df)} rows to tau_sweep_table.csv")

    # Operating points: τ that gives each ckpt the SAME real_fpr as P8A on each suite
    print("\n=== Operating points (match P8A real_fpr per suite) ===")
    op_rows = []
    for suite in suites:
        p8a_df = df[(df["ckpt"] == "P8A_REFERENCE_STEP5000") & (df["suite"] == suite)]
        if p8a_df.empty:
            continue
        # P8A's real_fpr at τ=0.5
        p8a_at_05 = p8a_df[p8a_df["tau"].between(0.498, 0.502)]
        if p8a_at_05.empty:
            continue
        target_fpr = float(p8a_at_05.iloc[0]["real_fpr"])
        if np.isnan(target_fpr):
            continue
        for ckpt in ckpt_scores.keys():
            ck_df = df[(df["ckpt"] == ckpt) & (df["suite"] == suite)].copy()
            if ck_df.empty:
                continue
            # Find τ where this ckpt achieves real_fpr ≤ target_fpr (with margin)
            tol = 0.005
            ck_df["abs_diff"] = (ck_df["real_fpr"] - target_fpr).abs()
            best = ck_df.iloc[ck_df["abs_diff"].argmin()]
            op_rows.append({
                "suite": suite,
                "ckpt": ckpt,
                "p8a_target_fpr_at_05": target_fpr,
                "matched_tau": float(best["tau"]),
                "achieved_fpr": float(best["real_fpr"]),
                "fake_recall_at_matched_tau": float(best["fake_recall"]),
                "n_real": int(best["n_real"]),
                "n_fake": int(best["n_fake"]),
            })
    op_df = pd.DataFrame(op_rows)
    op_df.to_csv(OUT_DIR / "operating_points.csv", index=False)

    # Lockbox-specific operating-point summary
    lockbox_summary = []
    fake_suites = [s for s in suites if "fake" in s and "lockbox" in s]
    real_suites = [s for s in suites if "real" in s and "lockbox" in s]
    print(f"Lockbox fake suites: {fake_suites}")
    print(f"Lockbox real suites: {real_suites}")

    # At dev-calibrated τ (τ that achieves dev real_fpr = 0.05), what's the lockbox recall?
    print("\n=== Calibrated-τ summary (τ s.t. teams_real_all_dev FPR ≤ 0.05) ===")
    print(f"{'ckpt':<35} {'τ_cal':>8} {'dev_fpr':>8} {'lockbox_fpr':>12} {'lockbox_recall':>15}")
    cal_rows = []
    for ckpt in ckpt_scores.keys():
        dev_df = df[(df["ckpt"] == ckpt) & (df["suite"] == "teams_real_all_dev")].sort_values("tau")
        if dev_df.empty:
            continue
        # find smallest τ such that dev_fpr ≤ 0.05
        below = dev_df[dev_df["real_fpr"] <= 0.05]
        if below.empty:
            tau_cal = float(dev_df["tau"].max())
            dev_fpr = float(dev_df["real_fpr"].iloc[-1])
        else:
            tau_cal = float(below["tau"].iloc[0])
            dev_fpr = float(below["real_fpr"].iloc[0])
        lock_real_df = df[(df["ckpt"] == ckpt) & (df["suite"] == "teams_real_all_lockbox")
                          & (df["tau"].between(tau_cal - 1e-4, tau_cal + 1e-4))]
        lock_fake_df = df[(df["ckpt"] == ckpt) & (df["suite"] == "teams_fake_all_lockbox")
                          & (df["tau"].between(tau_cal - 1e-4, tau_cal + 1e-4))]
        lock_fpr = float(lock_real_df["real_fpr"].iloc[0]) if not lock_real_df.empty else np.nan
        lock_rec = float(lock_fake_df["fake_recall"].iloc[0]) if not lock_fake_df.empty else np.nan
        cal_rows.append({
            "ckpt": ckpt,
            "tau_cal_dev_fpr_05": tau_cal,
            "dev_real_fpr_achieved": dev_fpr,
            "lockbox_real_fpr_at_tau_cal": lock_fpr,
            "lockbox_fake_recall_at_tau_cal": lock_rec,
        })
        print(f"{ckpt:<35} {tau_cal:>8.4f} {dev_fpr:>8.4f} {lock_fpr:>12.4f} {lock_rec:>15.4f}")
    pd.DataFrame(cal_rows).to_csv(OUT_DIR / "dev_cal_05pct_summary.csv", index=False)

    # Plot 1: per-ckpt real-FPR-vs-τ on teams_real_all_dev + teams_real_all_lockbox
    plt.figure(figsize=(12, 6))
    for ckpt in ckpt_scores.keys():
        if "ANCHOR" in ckpt and "STEP3500" not in ckpt and "SLOT_A" not in ckpt:
            continue
        for suite, ls in [("teams_real_all_dev", "-"), ("teams_real_all_lockbox", "--")]:
            sub = df[(df["ckpt"] == ckpt) & (df["suite"] == suite)].sort_values("tau")
            if sub.empty:
                continue
            plt.plot(sub["tau"], sub["real_fpr"], linestyle=ls,
                     label=f"{ckpt[:25]} / {suite[-10:]}", alpha=0.6)
    plt.xlabel("τ")
    plt.ylabel("real_fpr")
    plt.title("Real-FPR vs τ — dev (solid) vs lockbox (dashed)")
    plt.legend(loc="upper right", fontsize=7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_real_fpr_vs_tau.png", dpi=110)
    plt.close()

    # Plot 2: ROC-style — real_fpr vs lockbox_fake_recall per ckpt (parametric in τ)
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(ckpt_scores)))
    for color, ckpt in zip(colors, ckpt_scores.keys()):
        lock_real = df[(df["ckpt"] == ckpt) & (df["suite"] == "teams_real_all_lockbox")].sort_values("tau")
        lock_fake = df[(df["ckpt"] == ckpt) & (df["suite"] == "teams_fake_all_lockbox")].sort_values("tau")
        merged = pd.merge(lock_real[["tau", "real_fpr"]],
                          lock_fake[["tau", "fake_recall"]], on="tau")
        plt.plot(merged["real_fpr"], merged["fake_recall"], "-", color=color,
                 label=ckpt[:30], alpha=0.85, linewidth=1.5)
        # mark τ=0.5
        m05 = merged[merged["tau"].between(0.498, 0.502)]
        if not m05.empty:
            plt.plot(m05.iloc[0]["real_fpr"], m05.iloc[0]["fake_recall"], "o",
                     color=color, markersize=8, markeredgecolor="black")
    plt.xlabel("lockbox_real_fpr")
    plt.ylabel("lockbox_fake_recall")
    plt.title("Operating-point frontier (lockbox) — dots mark τ=0.5")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_lockbox_operating_frontier.png", dpi=110)
    plt.close()

    # Plot 3: same but for dev suites
    plt.figure(figsize=(10, 8))
    for color, ckpt in zip(colors, ckpt_scores.keys()):
        dev_real = df[(df["ckpt"] == ckpt) & (df["suite"] == "teams_real_all_dev")].sort_values("tau")
        # union all dev fake suites
        dev_fake_suites = [s for s in suites if "fake" in s and "dev" in s]
        for fake_suite in dev_fake_suites:
            dev_fake = df[(df["ckpt"] == ckpt) & (df["suite"] == fake_suite)].sort_values("tau")
            merged = pd.merge(dev_real[["tau", "real_fpr"]],
                              dev_fake[["tau", "fake_recall"]], on="tau")
            plt.plot(merged["real_fpr"], merged["fake_recall"], "-", color=color,
                     label=f"{ckpt[:20]} / {fake_suite[-15:]}",
                     alpha=0.6, linewidth=1)
    plt.xlabel("dev_real_fpr (teams_real_all_dev)")
    plt.ylabel("dev_fake_recall")
    plt.title("Dev operating-point frontier")
    plt.legend(loc="lower right", fontsize=6)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_dev_operating_frontier.png", dpi=110)
    plt.close()

    print(f"\nWrote {len(df)} rows to tau_sweep_table.csv")
    print(f"Wrote {len(op_df)} rows to operating_points.csv")
    print(f"Wrote {len(cal_rows)} rows to dev_cal_05pct_summary.csv")
    print(f"Plots in {FIG_DIR}")


if __name__ == "__main__":
    main()
