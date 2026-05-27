"""Job H — Optimal deployment configuration analysis given current artifacts.

Goal: identify the single best (ckpt, τ) deployment configuration that we can
recommend right now without further training.

Computes:
  1. Lockbox-calibrated naive-global-τ Pareto curves per ckpt — sweeps τ to
     achieve lockbox_real_fpr ∈ {0.5%, 1%, 2%, 5%, 10%, 15%, 20%, 25%}.
  2. Multi-cohort operating-point table per (ckpt × target_lockbox_fpr) — at
     each calibration point, report: lockbox_fake_recall, dev_real_fpr,
     teams_fake_all_dev_recall, viso_enh_recall, deeplive_enh_recall.
  3. Per-identity FP at each candidate τ for chronic identities (Roy_D,
     PC_Generator, Chikara_Takahashi, Q, bla_bla_chow, dor_shkedi).
  4. Pareto dominance check between ckpt pairs.
  5. Score-ensemble Pareto: mean / max / min of P8A + Slot A v2 frame-level
     probs.
  6. Best τ per FP/FN cost ratio λ — argmax{lockbox_fake_recall − λ × lockbox_real_fpr}.

Loads (all per-video, video_id-paired):
  - From 2026-05-20 scorecard reports (P8A, T5C, Slot A v2 ANCHOR_AWARE step3500):
      * teams_real_all_lockbox / teams_fake_all_lockbox
      * teams_real_all_dev / teams_fake_all_dev
      * visomaster_enhanced_macro_dev / deeplive_enhanced_dev
  - From 2026-05-08 E2B per-frame cache (aggregated to per-video by mean):
      * E2B teams_real_all_lockbox / teams_fake_all_lockbox / dev variants

Writes:
  outputs/pareto_curves.csv          — (ckpt, τ, lockbox_fpr, lockbox_recall, ...) for all τ
  outputs/calibrated_operating_pts.csv — table at target lockbox_fpr ∈ {1%, 2%, 5%, 10%, 15%, 20%}
  outputs/per_identity_at_op_pts.csv — Roy_D / PCGen / etc. FPRs at each candidate τ
  outputs/ensemble_pareto.csv        — score-fusion Pareto
  outputs/best_per_lambda.csv        — best (ckpt, τ) per cost-ratio λ
  outputs/pareto_dominance.json      — pairwise dominance verdicts
  outputs/headline.json              — single-number summary for the user
  figs/                              — Pareto + per-cohort plots
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
THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
OUT_DIR = THIS_DIR / "outputs"
FIG_DIR = THIS_DIR / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Reused data dirs
JOB_B_DATA = REPO / "analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_b_bootstrap_ci/data"
JOB_D_DATA = REPO / "analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_d_per_identity_fp/data"
E2B_CACHE = REPO / "analysis/iq_shortcut_decomp_2026-05-08/scores_cache"
MAY6_DIR_2026_05_13 = REPO / "analysis/r13_overnight_may6_retest_2026-05-13/outputs"
MAY6_DIR_2026_05_06 = REPO / "analysis/xinhe_cross_camera_audit_2026-05-06/outputs"
JOB_F_OUT = REPO / "analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_f_may6_retest/outputs"

CKPTS = {
    "P8A": "p8a_reference_step5000",
    "SlotAv2": "slot_a_anchor_aware_step3500",
    "T5C": "t5c_periodic_step3500",
}
E2B_KEY = "E2B_TOP_N_STEP3200"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_2026_05_20_report(suite: str, ckpt_slug: str) -> pd.DataFrame:
    """Per-video reports from the 2026-05-20 scorecard. Schema:
       method,label,video_id,avg_video_prob,prediction,is_correct,group_key,family_key
    """
    # Check several dirs that may contain this report
    candidates = [
        THIS_DIR / "data" / f"{suite}_{ckpt_slug}_videos_report.csv",
        JOB_B_DATA / f"{suite}_{ckpt_slug}_videos_report.csv",
        JOB_D_DATA / f"{suite}_{ckpt_slug}_videos_report.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"No report for {suite} / {ckpt_slug}; checked {candidates}")


def load_e2b_per_video(suite_short: str) -> pd.DataFrame:
    """Per-frame E2B cache from 2026-05-08 aggregated to per-video by mean.
    Schema in: method,label,video_id,frame_path,frame_prob,group_key,family_key
    Returns: method,label,video_id,avg_video_prob (synthesized)
    """
    cache_path = E2B_CACHE / f"{E2B_KEY}__{suite_short}.csv"
    if not cache_path.exists():
        raise FileNotFoundError(f"E2B cache miss: {cache_path}")
    df = pd.read_csv(cache_path)
    # Aggregate frame_prob → avg_video_prob per (label, video_id)
    agg = (
        df.groupby(["method", "label", "video_id", "group_key", "family_key"], dropna=False)
          .agg(avg_video_prob=("frame_prob", "mean"), n_frames=("frame_prob", "size"))
          .reset_index()
    )
    return agg


def collect_suite(suite: str, suite_short: str | None = None) -> dict[str, pd.DataFrame]:
    """Return dict ckpt_label → per-video df for a suite. E2B path uses
    `suite_short` if provided (cache filename differs)."""
    out = {}
    for label, slug in CKPTS.items():
        try:
            out[label] = load_2026_05_20_report(suite, slug)
        except FileNotFoundError as e:
            print(f"  WARN: missing {label} {suite}: {e}")
    suite_e2b = suite_short or suite
    try:
        out["E2B"] = load_e2b_per_video(suite_e2b)
    except FileNotFoundError as e:
        print(f"  WARN: no E2B cache for {suite_e2b}: {e}")
    return out


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

TAU_GRID = np.linspace(0.001, 0.999, 999)


def fpr_at_tau(scores: np.ndarray, tau: float) -> float:
    """Reals only — fpr at τ = fraction of scores ≥ τ."""
    if len(scores) == 0:
        return float("nan")
    return float((scores >= tau).mean())


def recall_at_tau(scores: np.ndarray, tau: float) -> float:
    """Fakes only — recall at τ = fraction of scores ≥ τ."""
    if len(scores) == 0:
        return float("nan")
    return float((scores >= tau).mean())


def tau_for_target_fpr(real_scores: np.ndarray, target_fpr: float) -> float:
    """Find τ such that FPR(real_scores, τ) ≤ target_fpr.
    Returns the smallest τ satisfying this (the natural calibration point).
    """
    if len(real_scores) == 0:
        return float("nan")
    sorted_desc = np.sort(real_scores)[::-1]
    # We want the (target_fpr * n)-th largest score; τ = that value + epsilon
    n = len(sorted_desc)
    n_above = int(np.floor(target_fpr * n))
    if n_above == 0:
        # No real should be above τ → τ = max+ε
        return float(sorted_desc[0] + 1e-9)
    if n_above >= n:
        return float(sorted_desc[-1] - 1e-9)
    # τ between the n_above-th and (n_above+1)-th sorted-desc score
    tau = float((sorted_desc[n_above - 1] + sorted_desc[n_above]) / 2.0)
    return tau


# ---------------------------------------------------------------------------
# 1. Lockbox-calibrated Pareto curves
# ---------------------------------------------------------------------------

def build_pareto_curves():
    print("\n=== Building Pareto curves on lockbox ===")
    lock_real = collect_suite("teams_real_all_lockbox")
    lock_fake = collect_suite("teams_fake_all_lockbox")

    rows = []
    for ckpt in sorted(set(lock_real) & set(lock_fake)):
        r_scores = lock_real[ckpt]["avg_video_prob"].values
        f_scores = lock_fake[ckpt]["avg_video_prob"].values
        for tau in TAU_GRID:
            rows.append({
                "ckpt": ckpt,
                "tau": float(tau),
                "lockbox_real_fpr": fpr_at_tau(r_scores, tau),
                "lockbox_fake_recall": recall_at_tau(f_scores, tau),
                "n_real": len(r_scores),
                "n_fake": len(f_scores),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "pareto_curves.csv", index=False)
    return df, lock_real, lock_fake


# ---------------------------------------------------------------------------
# 2. Calibrated operating points (lockbox FPR-calibrated)
# ---------------------------------------------------------------------------

def calibrated_operating_points(lock_real, lock_fake):
    print("\n=== Computing calibrated operating points ===")
    other_suites = {
        "teams_real_all_dev": collect_suite("teams_real_all_dev"),
        "teams_fake_all_dev": collect_suite("teams_fake_all_dev"),
        "visomaster_enhanced_macro_dev": collect_suite("visomaster_enhanced_macro_dev"),
        "deeplive_enhanced_dev": collect_suite("deeplive_enhanced_dev"),
    }

    target_fprs = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
    rows = []
    for ckpt in sorted(set(lock_real) & set(lock_fake)):
        r_scores = lock_real[ckpt]["avg_video_prob"].values
        f_scores = lock_fake[ckpt]["avg_video_prob"].values
        for tfp in target_fprs:
            tau = tau_for_target_fpr(r_scores, tfp)
            actual_fpr = fpr_at_tau(r_scores, tau)
            cell = {
                "ckpt": ckpt,
                "target_lockbox_fpr": tfp,
                "tau": tau,
                "achieved_lockbox_fpr": actual_fpr,
                "lockbox_fake_recall": recall_at_tau(f_scores, tau),
            }
            for suite_name, suite_dict in other_suites.items():
                if ckpt not in suite_dict:
                    cell[f"{suite_name}_metric"] = float("nan")
                    continue
                sub = suite_dict[ckpt]
                # Reals → fpr, fakes → recall (use label column)
                reals = sub.loc[sub["label"] == 0, "avg_video_prob"].values
                fakes = sub.loc[sub["label"] == 1, "avg_video_prob"].values
                if "_real_" in suite_name:
                    cell[f"{suite_name}_metric"] = fpr_at_tau(reals, tau) if len(reals) else float("nan")
                else:
                    # all fake suites
                    target_scores = fakes if len(fakes) > 0 else sub["avg_video_prob"].values
                    cell[f"{suite_name}_metric"] = recall_at_tau(target_scores, tau)
            rows.append(cell)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "calibrated_operating_pts.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 3. Per-identity FPR at candidate operating points
# ---------------------------------------------------------------------------

def per_identity_at_op_points(operating_df):
    print("\n=== Per-identity FPR at operating points ===")
    # Use Job D's per-identity per-video reports
    chronic_suites = {
        "Roy_D_dev": "teams_real_all_dev",            # filter by base_identity = Roy_D
        "PC_Generator_dev": "teams_capture_pc_generator_dev",
        "Chikara_Takahashi_lockbox": "teams_real_all_lockbox",  # filter by base_identity
        "Q_dev": "teams_real_all_dev",                 # filter by base_identity = Q
        "bla_bla_chow_dev": "teams_real_all_dev",      # filter
        "dor_shkedi_dev": "teams_capture_dor_shkedi_dev",
    }
    # base_identity is the prefix of video_id before "__"
    identity_filters = {
        "Roy_D_dev": "Roy_D",
        "PC_Generator_dev": None,  # whole suite
        "Chikara_Takahashi_lockbox": "Chikara_Takahashi",
        "Q_dev": "Q",
        "bla_bla_chow_dev": "bla_bla_chow",
        "dor_shkedi_dev": None,
    }

    cached = {}
    def get_suite(suite):
        if suite not in cached:
            cached[suite] = collect_suite(suite)
        return cached[suite]

    rows = []
    for _, op in operating_df.iterrows():
        ckpt = op["ckpt"]
        tau = op["tau"]
        target_fpr = op["target_lockbox_fpr"]
        row = {"ckpt": ckpt, "target_lockbox_fpr": target_fpr, "tau": tau}
        for label, suite in chronic_suites.items():
            sd = get_suite(suite)
            if ckpt not in sd:
                row[label] = float("nan")
                continue
            sub = sd[ckpt]
            filt = identity_filters[label]
            if filt is not None:
                sub = sub[sub["video_id"].str.startswith(filt + "__") | (sub["video_id"] == filt)]
            reals = sub.loc[sub["label"] == 0, "avg_video_prob"].values
            row[label] = fpr_at_tau(reals, tau) if len(reals) else float("nan")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_identity_at_op_pts.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 4. Pareto dominance check
# ---------------------------------------------------------------------------

def pareto_dominance(curves_df):
    print("\n=== Pareto dominance ===")
    # For each ckpt pair (A, B), check: does A dominate B?
    # A dominates B if at every FPR level, A_recall(τ_a) ≥ B_recall(τ_b) for matched FPR.
    # Approach: at a grid of FPR targets, look up each ckpt's recall.
    fpr_grid = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20]
    ckpts = sorted(curves_df["ckpt"].unique())
    by_ckpt = {}
    for ckpt in ckpts:
        sub = curves_df[curves_df["ckpt"] == ckpt].sort_values("tau")
        recalls = []
        for tfp in fpr_grid:
            below = sub[sub["lockbox_real_fpr"] <= tfp]
            if below.empty:
                recalls.append(0.0)
            else:
                # Best recall achievable at ≤ this FPR
                recalls.append(float(below["lockbox_fake_recall"].max()))
        by_ckpt[ckpt] = dict(zip(fpr_grid, recalls))

    dom = {}
    for a in ckpts:
        for b in ckpts:
            if a == b:
                continue
            wins = sum(1 for f in fpr_grid if by_ckpt[a][f] > by_ckpt[b][f])
            ties = sum(1 for f in fpr_grid if by_ckpt[a][f] == by_ckpt[b][f])
            losses = sum(1 for f in fpr_grid if by_ckpt[a][f] < by_ckpt[b][f])
            dom[f"{a}_vs_{b}"] = {
                "a_wins_at_fpr_levels": wins,
                "ties": ties,
                "b_wins_at_fpr_levels": losses,
                "dominance": "A_strictly" if (losses == 0 and wins > 0) else
                              ("B_strictly" if (wins == 0 and losses > 0) else "no_dominance"),
                "recall_at_fpr_levels_a": by_ckpt[a],
                "recall_at_fpr_levels_b": by_ckpt[b],
            }
    with open(OUT_DIR / "pareto_dominance.json", "w") as f:
        json.dump(dom, f, indent=2)
    return dom


# ---------------------------------------------------------------------------
# 5. Score-ensemble Pareto (P8A + Slot A v2 fusion)
# ---------------------------------------------------------------------------

def ensemble_pareto(lock_real, lock_fake):
    print("\n=== Score-fusion ensemble Pareto ===")
    if "P8A" not in lock_real or "SlotAv2" not in lock_real:
        print("  Insufficient ckpts for ensemble")
        return pd.DataFrame()

    # Merge on video_id within each suite
    def merge_scores(suite_dict):
        p8a = suite_dict["P8A"][["video_id", "avg_video_prob"]].rename(
            columns={"avg_video_prob": "p8a_score"})
        slot = suite_dict["SlotAv2"][["video_id", "avg_video_prob"]].rename(
            columns={"avg_video_prob": "slot_score"})
        return pd.merge(p8a, slot, on="video_id", how="inner")

    r = merge_scores(lock_real)
    f = merge_scores(lock_fake)
    fusions = {
        "mean": lambda a, b: (a + b) / 2.0,
        "max": np.maximum,
        "min": np.minimum,
        "geomean": lambda a, b: np.sqrt(np.maximum(a, 1e-9) * np.maximum(b, 1e-9)),
    }

    rows = []
    for fname, fn in fusions.items():
        r_fused = fn(r["p8a_score"].values, r["slot_score"].values)
        f_fused = fn(f["p8a_score"].values, f["slot_score"].values)
        for tau in TAU_GRID:
            rows.append({
                "fusion": fname,
                "tau": float(tau),
                "lockbox_real_fpr": fpr_at_tau(r_fused, tau),
                "lockbox_fake_recall": recall_at_tau(f_fused, tau),
                "n_real": len(r_fused),
                "n_fake": len(f_fused),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "ensemble_pareto.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 6. Best (ckpt, τ) per FP/FN cost ratio λ
# ---------------------------------------------------------------------------

def best_per_lambda(curves_df, ensemble_df):
    print("\n=== Best (ckpt, τ) per cost ratio λ ===")
    lambdas = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100]
    rows = []
    # Single ckpts
    for ckpt in sorted(curves_df["ckpt"].unique()):
        sub = curves_df[curves_df["ckpt"] == ckpt].copy()
        for lam in lambdas:
            sub["score"] = sub["lockbox_fake_recall"] - lam * sub["lockbox_real_fpr"]
            best = sub.loc[sub["score"].idxmax()]
            rows.append({
                "config": ckpt,
                "lambda": lam,
                "tau": float(best["tau"]),
                "lockbox_real_fpr": float(best["lockbox_real_fpr"]),
                "lockbox_fake_recall": float(best["lockbox_fake_recall"]),
                "score": float(best["score"]),
            })
    # Ensembles
    if not ensemble_df.empty:
        for fusion in ensemble_df["fusion"].unique():
            sub = ensemble_df[ensemble_df["fusion"] == fusion].copy()
            for lam in lambdas:
                sub["score"] = sub["lockbox_fake_recall"] - lam * sub["lockbox_real_fpr"]
                best = sub.loc[sub["score"].idxmax()]
                rows.append({
                    "config": f"ENSEMBLE_{fusion}",
                    "lambda": lam,
                    "tau": float(best["tau"]),
                    "lockbox_real_fpr": float(best["lockbox_real_fpr"]),
                    "lockbox_fake_recall": float(best["lockbox_fake_recall"]),
                    "score": float(best["score"]),
                })
    df = pd.DataFrame(rows)
    # For each λ, mark the global winner
    df["is_winner"] = False
    for lam in lambdas:
        idx = df[df["lambda"] == lam]["score"].idxmax()
        df.at[idx, "is_winner"] = True
    df.to_csv(OUT_DIR / "best_per_lambda.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------

def plot_pareto_curves(curves_df, ensemble_df):
    plt.figure(figsize=(11, 8))
    colors = {"P8A": "C0", "SlotAv2": "C1", "T5C": "C2", "E2B": "C3"}
    for ckpt in sorted(curves_df["ckpt"].unique()):
        sub = curves_df[curves_df["ckpt"] == ckpt].sort_values("lockbox_real_fpr")
        plt.plot(sub["lockbox_real_fpr"], sub["lockbox_fake_recall"],
                 "-", color=colors.get(ckpt, "gray"), linewidth=2, label=ckpt, alpha=0.85)
    if not ensemble_df.empty:
        for fname, ls in [("mean", ":"), ("max", "--"), ("min", "-.")]:
            sub = ensemble_df[ensemble_df["fusion"] == fname].sort_values("lockbox_real_fpr")
            plt.plot(sub["lockbox_real_fpr"], sub["lockbox_fake_recall"],
                     ls, color="black", linewidth=1, label=f"ENS_{fname}", alpha=0.7)

    # Other agent's reference: P8A 75.5% recall at 10% lockbox FPR
    plt.axvline(0.10, color="gray", linestyle=":", alpha=0.4)
    plt.axhline(0.755, color="gray", linestyle=":", alpha=0.4)
    plt.plot(0.10, 0.755, "kx", markersize=12, markeredgewidth=2,
             label="prior agent ref: P8A 75.5%/10%")

    plt.xlabel("lockbox_real_fpr (calibrated on teams_real_all_lockbox, n=1361)")
    plt.ylabel("lockbox_fake_recall (teams_fake_all_lockbox, n=253)")
    plt.title("Lockbox-calibrated naive-global-τ Pareto curves")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 0.30)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_lockbox_pareto.png", dpi=110)
    plt.close()

    # Zoomed-in low-FPR region
    plt.figure(figsize=(10, 7))
    for ckpt in sorted(curves_df["ckpt"].unique()):
        sub = curves_df[curves_df["ckpt"] == ckpt].sort_values("lockbox_real_fpr")
        plt.plot(sub["lockbox_real_fpr"], sub["lockbox_fake_recall"],
                 "-", color=colors.get(ckpt, "gray"), linewidth=2, label=ckpt, alpha=0.85)
    plt.axvline(0.05, color="gray", linestyle=":", alpha=0.4, label="5% FPR")
    plt.axvline(0.10, color="gray", linestyle=":", alpha=0.4, label="10% FPR")
    plt.xlabel("lockbox_real_fpr")
    plt.ylabel("lockbox_fake_recall")
    plt.title("Lockbox Pareto — low-FPR zoom")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 0.10)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_lockbox_pareto_low_fpr_zoom.png", dpi=110)
    plt.close()


def plot_per_identity(per_id_df):
    if per_id_df.empty:
        return
    identities = [c for c in per_id_df.columns if c not in ["ckpt", "target_lockbox_fpr", "tau"]]
    n = len(identities)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    axes = axes.flatten()
    colors = {"P8A": "C0", "SlotAv2": "C1", "T5C": "C2", "E2B": "C3"}
    for ax, ident in zip(axes, identities):
        for ckpt in sorted(per_id_df["ckpt"].unique()):
            sub = per_id_df[per_id_df["ckpt"] == ckpt]
            ax.plot(sub["target_lockbox_fpr"], sub[ident],
                    "-o", color=colors.get(ckpt, "gray"), label=ckpt, alpha=0.8)
        ax.set_title(ident)
        ax.set_xlabel("target_lockbox_fpr")
        ax.set_ylabel("identity-specific FPR")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.25)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7)
    fig.suptitle("Per-identity FPR at each lockbox-calibrated operating point")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_per_identity_at_op_pts.png", dpi=110)
    plt.close()


def plot_lambda_winners(lambda_df):
    plt.figure(figsize=(11, 6))
    for config in sorted(lambda_df["config"].unique()):
        sub = lambda_df[lambda_df["config"] == config]
        plt.plot(sub["lambda"], sub["score"], "-o", label=config, alpha=0.8)
    plt.xscale("log")
    plt.xlabel("FP/FN cost ratio λ (log scale)")
    plt.ylabel("max score = recall − λ × fpr")
    plt.title("Optimal score vs cost ratio λ for each config")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_lambda_winners.png", dpi=110)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    curves_df, lock_real, lock_fake = build_pareto_curves()
    print(f"  Pareto curves: {len(curves_df)} rows across {curves_df['ckpt'].nunique()} ckpts")

    operating_df = calibrated_operating_points(lock_real, lock_fake)
    print(f"  Operating points: {len(operating_df)} rows")

    per_id_df = per_identity_at_op_points(operating_df)
    print(f"  Per-identity rows: {len(per_id_df)}")

    dom = pareto_dominance(curves_df)
    ensemble_df = ensemble_pareto(lock_real, lock_fake)
    print(f"  Ensemble rows: {len(ensemble_df)}")

    lambda_df = best_per_lambda(curves_df, ensemble_df)

    # Headline values
    headline = {}
    for ckpt in sorted(curves_df["ckpt"].unique()):
        sub = curves_df[curves_df["ckpt"] == ckpt]
        # Recall at lockbox_fpr <= 0.10
        below_10 = sub[sub["lockbox_real_fpr"] <= 0.10]
        recall_at_10pct = float(below_10["lockbox_fake_recall"].max()) if not below_10.empty else 0.0
        below_05 = sub[sub["lockbox_real_fpr"] <= 0.05]
        recall_at_5pct = float(below_05["lockbox_fake_recall"].max()) if not below_05.empty else 0.0
        below_02 = sub[sub["lockbox_real_fpr"] <= 0.02]
        recall_at_2pct = float(below_02["lockbox_fake_recall"].max()) if not below_02.empty else 0.0
        # AUC-style integral (sum of recall over the Pareto curve up to FPR=0.25)
        below_25 = sub[sub["lockbox_real_fpr"] <= 0.25].sort_values("lockbox_real_fpr")
        if len(below_25) >= 2:
            x = below_25["lockbox_real_fpr"].values
            y = below_25["lockbox_fake_recall"].values
            auc_to_25 = float(np.trapz(y, x) / 0.25)
        else:
            auc_to_25 = 0.0
        headline[ckpt] = {
            "recall_at_lockbox_fpr_2pct": recall_at_2pct,
            "recall_at_lockbox_fpr_5pct": recall_at_5pct,
            "recall_at_lockbox_fpr_10pct": recall_at_10pct,
            "auc_recall_below_fpr_25pct": auc_to_25,
            "n_real_lockbox": int(sub.iloc[0]["n_real"]),
            "n_fake_lockbox": int(sub.iloc[0]["n_fake"]),
        }
    # Ensemble headline
    if not ensemble_df.empty:
        for fname in ["mean", "max", "min", "geomean"]:
            sub = ensemble_df[ensemble_df["fusion"] == fname]
            below_10 = sub[sub["lockbox_real_fpr"] <= 0.10]
            r10 = float(below_10["lockbox_fake_recall"].max()) if not below_10.empty else 0.0
            below_05 = sub[sub["lockbox_real_fpr"] <= 0.05]
            r05 = float(below_05["lockbox_fake_recall"].max()) if not below_05.empty else 0.0
            below_02 = sub[sub["lockbox_real_fpr"] <= 0.02]
            r02 = float(below_02["lockbox_fake_recall"].max()) if not below_02.empty else 0.0
            below_25 = sub[sub["lockbox_real_fpr"] <= 0.25].sort_values("lockbox_real_fpr")
            if len(below_25) >= 2:
                auc_25 = float(np.trapz(below_25["lockbox_fake_recall"].values,
                                         below_25["lockbox_real_fpr"].values) / 0.25)
            else:
                auc_25 = 0.0
            headline[f"ENSEMBLE_{fname}"] = {
                "recall_at_lockbox_fpr_2pct": r02,
                "recall_at_lockbox_fpr_5pct": r05,
                "recall_at_lockbox_fpr_10pct": r10,
                "auc_recall_below_fpr_25pct": auc_25,
                "n_real_lockbox": int(sub.iloc[0]["n_real"]),
                "n_fake_lockbox": int(sub.iloc[0]["n_fake"]),
            }
    with open(OUT_DIR / "headline.json", "w") as f:
        json.dump(headline, f, indent=2)

    plot_pareto_curves(curves_df, ensemble_df)
    plot_per_identity(per_id_df)
    plot_lambda_winners(lambda_df)

    print("\n" + "=" * 80)
    print("HEADLINE — recall at fixed lockbox FPR levels")
    print("=" * 80)
    cols = ["recall_at_lockbox_fpr_2pct", "recall_at_lockbox_fpr_5pct",
            "recall_at_lockbox_fpr_10pct", "auc_recall_below_fpr_25pct"]
    print(f"{'config':<25} " + " ".join(f"{c.replace('recall_at_lockbox_fpr_', '@'):>11}" for c in cols[:3]) + f"{'AUC<25%':>11}")
    for cfg, vals in headline.items():
        print(f"{cfg:<25} " + " ".join(f"{vals[c]:>11.4f}" for c in cols))

    print("\n" + "=" * 80)
    print("BEST CONFIG PER COST RATIO λ")
    print("=" * 80)
    winners = lambda_df[lambda_df["is_winner"]]
    for _, r in winners.iterrows():
        print(f"  λ={r['lambda']:>5.1f}: {r['config']:<22}  τ={r['tau']:.3f}  "
              f"fpr={r['lockbox_real_fpr']:.4f}  recall={r['lockbox_fake_recall']:.4f}  score={r['score']:.4f}")

    print("\n" + "=" * 80)
    print("PARETO DOMINANCE (lockbox Pareto across FPR grid)")
    print("=" * 80)
    seen = set()
    for k, v in dom.items():
        a, b = k.split("_vs_")
        pair = tuple(sorted([a, b]))
        if pair in seen:
            continue
        seen.add(pair)
        rev = f"{b}_vs_{a}"
        if rev in dom:
            print(f"  {a:>10} vs {b:<10}: {a}_wins={v['a_wins_at_fpr_levels']:>2}  ties={v['ties']:>2}  {b}_wins={v['b_wins_at_fpr_levels']:>2}  →  {v['dominance']}")

    return headline


if __name__ == "__main__":
    main()
