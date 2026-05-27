"""Score distribution + slice composition forensics for P8A / P18T / P18C.

Reads the 27 per-frame reports pulled from the 2026-05-02 D contract scorecard
(gs://training-job-outputs/test_results/teams_promotion_contract/p18-corrective-contract-20260502-103127/reports)
and produces:

  outputs/combined_frames.parquet      — all 27 reports merged
  outputs/combined_videos.parquet      — video-level aggregation (max frame_prob)
  outputs/quantiles_frame.csv          — per-(suite,model) quantiles, frame-level
  outputs/quantiles_video.csv          — per-(suite,model) quantiles, video-level
  outputs/method_breakdown.csv         — per-method counts + recall at deployed τ
  outputs/viso_winners.csv             — the N viso fakes P8A scores above contract τ
  outputs/recall_curve_by_suite.csv    — per-(suite,model,τ) cumulative recall
  outputs/figures/hist_<suite>_<model>.png    — histogram per (suite,model)
  outputs/figures/kde_overlay_<model>.png      — all suites overlaid per model
  outputs/figures/method_recall_curves_<model>.png — recall vs τ per method, one model
  outputs/figures/suite_recall_curves.png       — fake suites: recall vs τ, all models

Outputs land under analysis/score_distribution_2026-05-02/outputs/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_reports"
OUT = ROOT / "outputs"
FIG = OUT / "figures"

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# Selected (contract-deployed) τ per arm — from FACTS §1.2 / d_results scorecard.
DEPLOYED_TAU = {
    "P8A":  0.990946,
    "P18T": 0.993590,
    "P18C": 0.994625,
}

# Suite groupings for plots.
FAKE_SUITES = [
    "teams_fake_all_dev",
    "teams_fake_all_lockbox",
    "visomaster_enhanced_macro_dev",
    "deeplive_enhanced_dev",
]
REAL_SUITES = [
    "teams_real_all_dev",
    "teams_real_all_lockbox",
    "teams_real_dor_dev",
    "teams_real_lighting_extreme_dev",
    "teams_real_poor_quality_dev",
]

MODELS = ["P8A", "P18T", "P18C"]


def _parse_filename(name: str) -> Tuple[str, str]:
    base = name.replace("_frames_report.csv", "")
    if base.endswith("_p8a_reference_step5000"):
        return base[: -len("_p8a_reference_step5000")], "P8A"
    if base.endswith("_p18t_grl_treatment_step4000"):
        return base[: -len("_p18t_grl_treatment_step4000")], "P18T"
    if base.endswith("_p18c_no_grl_control_step4000"):
        return base[: -len("_p18c_no_grl_control_step4000")], "P18C"
    raise ValueError(f"Unparseable filename: {name}")


def load_all() -> pd.DataFrame:
    rows = []
    for f in sorted(RAW.glob("*_frames_report.csv")):
        suite, model = _parse_filename(f.name)
        df = pd.read_csv(f)
        df["suite"] = suite
        df["model"] = model
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["frame_prob"] = out["frame_prob"].astype(float)
    out["label"] = out["label"].astype(int)
    return out


def quantiles(df: pd.DataFrame, key_col: str = "frame_prob") -> pd.DataFrame:
    qs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    grouped = df.groupby(["suite", "model"])
    rows = []
    for (suite, model), g in grouped:
        rec = {
            "suite": suite,
            "model": model,
            "n": len(g),
            "label": int(g["label"].iloc[0]) if len(g) else -1,
            "mean": g[key_col].mean(),
            "std": g[key_col].std(),
        }
        for q in qs:
            rec[f"p{int(q*100)}" if q < 1.0 else "max"] = float(np.quantile(g[key_col], q))
        # Also: fraction above each common threshold.
        for tau in [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]:
            rec[f"frac_ge_{tau:g}"] = float((g[key_col] >= tau).mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["suite", "model"])


def to_video_level(df: pd.DataFrame) -> pd.DataFrame:
    # Use MAX prob_fake across frames per video — typical "any frame says fake" rule.
    g = df.groupby(["suite", "model", "video_id"], as_index=False).agg(
        frame_prob=("frame_prob", "max"),
        label=("label", "first"),
        method=("method", "first"),
        n_frames=("frame_prob", "size"),
    )
    return g


def method_breakdown(df_frames: pd.DataFrame) -> pd.DataFrame:
    """Per (suite, model, method) counts and recall@deployed τ."""
    rows = []
    for (suite, model, method), g in df_frames.groupby(["suite", "model", "method"]):
        tau = DEPLOYED_TAU.get(model, 0.5)
        is_real = (g["label"] == 0).all()
        is_fake = (g["label"] == 1).all()
        if is_real:
            metric_name, metric_val = "FPR_at_deployed_tau", float((g["frame_prob"] >= tau).mean())
        elif is_fake:
            metric_name, metric_val = "RECALL_at_deployed_tau", float((g["frame_prob"] >= tau).mean())
        else:
            metric_name, metric_val = "MIXED", -1.0
        rows.append({
            "suite": suite,
            "model": model,
            "method": method,
            "n_frames": len(g),
            "n_videos": g["video_id"].nunique(),
            "median_prob": float(g["frame_prob"].median()),
            "mean_prob": float(g["frame_prob"].mean()),
            metric_name: metric_val,
            "deployed_tau": tau,
        })
    return pd.DataFrame(rows).sort_values(["suite", "model", "method"])


def viso_winners(df_frames: pd.DataFrame) -> pd.DataFrame:
    """For each (model, suite=visomaster_enhanced_macro_dev), the frames above deployed τ."""
    rows = []
    for model in MODELS:
        sub = df_frames[(df_frames["suite"] == "visomaster_enhanced_macro_dev") & (df_frames["model"] == model)]
        tau = DEPLOYED_TAU[model]
        winners = sub[sub["frame_prob"] >= tau].copy()
        winners["model"] = model
        winners["deployed_tau"] = tau
        rows.append(winners)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def recall_curves(df_frames: pd.DataFrame) -> pd.DataFrame:
    """For each (suite, model), recall(τ) over a τ grid covering deployment range."""
    taus = np.concatenate([
        np.linspace(0.0, 0.95, 20, endpoint=False),  # coarse low end
        np.linspace(0.95, 1.0, 51, endpoint=True),    # fine deployment range
    ])
    taus = np.unique(np.round(taus, 5))
    rows = []
    for (suite, model), g in df_frames.groupby(["suite", "model"]):
        is_real = (g["label"] == 0).all()
        for tau in taus:
            frac = float((g["frame_prob"] >= tau).mean())
            rows.append({
                "suite": suite,
                "model": model,
                "tau": float(tau),
                "metric_name": "FPR" if is_real else "RECALL",
                "metric_value": frac,
                "n": len(g),
            })
    return pd.DataFrame(rows)


# ────────── Plotting ──────────

def plot_hist_per_suite_model(df_frames: pd.DataFrame) -> None:
    bins = np.linspace(0, 1, 41)
    for suite in sorted(df_frames["suite"].unique()):
        for model in MODELS:
            sub = df_frames[(df_frames["suite"] == suite) & (df_frames["model"] == model)]
            if sub.empty:
                continue
            tau = DEPLOYED_TAU[model]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(sub["frame_prob"], bins=bins, color="steelblue", edgecolor="black", alpha=0.8)
            ax.axvline(tau, color="red", linestyle="--", linewidth=1.5, label=f"deployed τ={tau:.4f}")
            ax.axvline(0.5, color="grey", linestyle=":", linewidth=1, label="τ=0.5")
            n_above = int((sub["frame_prob"] >= tau).sum())
            label = "fake" if sub["label"].iloc[0] == 1 else "real"
            metric = "RECALL" if label == "fake" else "FPR"
            ax.set_title(
                f"{suite}  ·  {model}  ·  n={len(sub)} {label}\n"
                f"{metric}@τ_deployed = {n_above}/{len(sub)} = {n_above/len(sub):.3f}"
            )
            ax.set_xlabel("frame_prob (post-sigmoid)")
            ax.set_ylabel("count")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.5)
            fig.tight_layout()
            fig.savefig(FIG / f"hist_{suite}_{model}.png", dpi=110)
            plt.close(fig)


def plot_kde_overlay(df_frames: pd.DataFrame) -> None:
    """For each model, KDE overlay of all FAKE suites in one panel; real suites in another."""
    from scipy.stats import gaussian_kde

    def _plot_overlay(suites: List[str], label: str, tag: str):
        for model in MODELS:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            xs = np.linspace(0, 1, 400)
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
            for i, suite in enumerate(suites):
                sub = df_frames[(df_frames["suite"] == suite) & (df_frames["model"] == model)]
                if sub.empty:
                    continue
                vals = sub["frame_prob"].to_numpy()
                if vals.std() < 1e-6:
                    continue
                kde = gaussian_kde(vals, bw_method=0.06)
                ys = kde(xs)
                ax.plot(xs, ys, label=f"{suite} (n={len(sub)})", color=colors[i % len(colors)], linewidth=1.6)
                ax.fill_between(xs, 0, ys, alpha=0.10, color=colors[i % len(colors)])
            ax.axvline(DEPLOYED_TAU[model], color="red", linestyle="--", linewidth=1.5,
                       label=f"deployed τ={DEPLOYED_TAU[model]:.4f}")
            ax.axvline(0.5, color="grey", linestyle=":", linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_xlabel("frame_prob (post-sigmoid)")
            ax.set_ylabel("density")
            ax.set_title(f"{label} suites — score density · {model}")
            ax.legend(loc="upper left", fontsize=8)
            fig.tight_layout()
            fig.savefig(FIG / f"kde_overlay_{tag}_{model}.png", dpi=110)
            plt.close(fig)

    _plot_overlay(FAKE_SUITES, "Fake", "fake")
    _plot_overlay(REAL_SUITES, "Real", "real")


def plot_method_recall_curves(df_frames: pd.DataFrame) -> None:
    """For teams_fake_all_dev: per-method recall vs τ. This is THE key shape-of-failure plot."""
    suite = "teams_fake_all_dev"
    sub_all = df_frames[df_frames["suite"] == suite]
    if sub_all.empty:
        return
    methods = sorted(sub_all["method"].unique(), key=lambda m: -(sub_all["method"] == m).sum())
    # Pick top methods by frame count + always include viso/deeplive for contrast.
    top = list(dict.fromkeys(["visomaster_enhanced_macro", "deeplive_enhanced"] + methods))[:14]
    for model in MODELS:
        fig, ax = plt.subplots(figsize=(9, 5))
        taus = np.linspace(0.5, 1.0, 101)
        cmap = plt.get_cmap("tab20")
        for i, method in enumerate(top):
            g = sub_all[(sub_all["method"] == method) & (sub_all["model"] == model)]
            if g.empty:
                continue
            recalls = [(g["frame_prob"] >= t).mean() for t in taus]
            ax.plot(taus, recalls, label=f"{method} (n={len(g)})", color=cmap(i % 20), linewidth=1.4)
        ax.axvline(DEPLOYED_TAU[model], color="red", linestyle="--", linewidth=1.5,
                   label=f"deployed τ={DEPLOYED_TAU[model]:.4f}")
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("threshold τ on frame_prob")
        ax.set_ylabel("recall (frame-level, fakes only)")
        ax.set_title(f"teams_fake_all_dev — per-method recall vs τ · {model}")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / f"method_recall_curves_{model}.png", dpi=110)
        plt.close(fig)


def plot_suite_recall_curves(df_frames: pd.DataFrame) -> None:
    """Fake suites: recall(τ) — one panel per model, all suites overlaid."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    taus = np.linspace(0.5, 1.0, 101)
    colors = {
        "teams_fake_all_dev": "#1f77b4",
        "teams_fake_all_lockbox": "#9467bd",
        "visomaster_enhanced_macro_dev": "#d62728",
        "deeplive_enhanced_dev": "#2ca02c",
    }
    for ax, model in zip(axes, MODELS):
        for suite in FAKE_SUITES:
            g = df_frames[(df_frames["suite"] == suite) & (df_frames["model"] == model)]
            if g.empty:
                continue
            recalls = [(g["frame_prob"] >= t).mean() for t in taus]
            ax.plot(taus, recalls, label=f"{suite} (n={len(g)})", color=colors.get(suite), linewidth=1.6)
        ax.axvline(DEPLOYED_TAU[model], color="red", linestyle="--", linewidth=1.2,
                   label=f"deployed τ={DEPLOYED_TAU[model]:.4f}")
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("τ")
        ax.set_title(model)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("frame-level recall")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle("Fake-suite recall vs τ — all three models")
    fig.tight_layout()
    fig.savefig(FIG / "suite_recall_curves.png", dpi=120)
    plt.close(fig)


def plot_real_fpr_curves(df_frames: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    taus = np.linspace(0.5, 1.0, 101)
    colors = {
        "teams_real_all_dev": "#1f77b4",
        "teams_real_all_lockbox": "#9467bd",
        "teams_real_dor_dev": "#d62728",
        "teams_real_lighting_extreme_dev": "#2ca02c",
        "teams_real_poor_quality_dev": "#ff7f0e",
    }
    for ax, model in zip(axes, MODELS):
        for suite in REAL_SUITES:
            g = df_frames[(df_frames["suite"] == suite) & (df_frames["model"] == model)]
            if g.empty:
                continue
            fpr = [(g["frame_prob"] >= t).mean() for t in taus]
            ax.plot(taus, fpr, label=f"{suite} (n={len(g)})", color=colors.get(suite), linewidth=1.6)
        ax.axvline(DEPLOYED_TAU[model], color="red", linestyle="--", linewidth=1.2,
                   label=f"deployed τ={DEPLOYED_TAU[model]:.4f}")
        ax.axhline(0.02, color="black", linestyle=":", linewidth=1, alpha=0.7, label="2% FPR floor")
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("τ")
        ax.set_title(model)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("frame-level FPR")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle("Real-suite FPR vs τ — all three models")
    fig.tight_layout()
    fig.savefig(FIG / "real_fpr_curves.png", dpi=120)
    plt.close(fig)


def plot_viso_video_score_strip(df_frames: pd.DataFrame) -> None:
    """For visomaster_enhanced_macro_dev: scatter of per-video max score, sorted, all models overlaid.

    Tells us whether viso "winners" are concentrated at one tail or scattered.
    """
    suite = "visomaster_enhanced_macro_dev"
    fig, ax = plt.subplots(figsize=(11, 5))
    for model, color in [("P8A", "#1f77b4"), ("P18T", "#ff7f0e"), ("P18C", "#2ca02c")]:
        g = df_frames[(df_frames["suite"] == suite) & (df_frames["model"] == model)].copy()
        if g.empty:
            continue
        # Each viso video has 1 frame, so frame == video.
        g_sorted = g.sort_values("frame_prob").reset_index(drop=True)
        ax.scatter(np.arange(len(g_sorted)), g_sorted["frame_prob"], s=8, alpha=0.6, color=color, label=model)
        ax.axhline(DEPLOYED_TAU[model], color=color, linestyle="--", linewidth=1, alpha=0.4)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.6, label="τ=0.5")
    ax.set_xlabel("video rank (sorted by frame_prob)")
    ax.set_ylabel("frame_prob")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{suite} (n=550) — per-video max score, sorted ascending")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "viso_video_score_strip.png", dpi=120)
    plt.close(fig)


def main():
    print("[load] reading all 27 frame reports...")
    df_frames = load_all()
    print(f"[load] total frames: {len(df_frames):,}")
    print(f"[load] suites × models: {df_frames.groupby(['suite','model']).size().shape[0]}")

    print("[aggregate] video-level (max prob across frames)...")
    df_videos = to_video_level(df_frames)
    print(f"[aggregate] total videos: {len(df_videos):,}")

    print("[save] parquet...")
    df_frames.to_parquet(OUT / "combined_frames.parquet", index=False)
    df_videos.to_parquet(OUT / "combined_videos.parquet", index=False)

    print("[quantiles] frame-level...")
    quantiles(df_frames).to_csv(OUT / "quantiles_frame.csv", index=False)

    print("[quantiles] video-level...")
    quantiles(df_videos).to_csv(OUT / "quantiles_video.csv", index=False)

    print("[breakdown] per-method...")
    method_breakdown(df_frames).to_csv(OUT / "method_breakdown.csv", index=False)

    print("[viso] winners (frames ≥ deployed τ)...")
    viso_winners(df_frames).to_csv(OUT / "viso_winners.csv", index=False)

    print("[curves] recall curves...")
    recall_curves(df_frames).to_csv(OUT / "recall_curve_by_suite.csv", index=False)

    print("[plot] per-(suite,model) histograms...")
    plot_hist_per_suite_model(df_frames)

    print("[plot] KDE overlays...")
    plot_kde_overlay(df_frames)

    print("[plot] method recall curves on teams_fake_all_dev...")
    plot_method_recall_curves(df_frames)

    print("[plot] suite recall curves...")
    plot_suite_recall_curves(df_frames)

    print("[plot] real-suite FPR curves...")
    plot_real_fpr_curves(df_frames)

    print("[plot] viso per-video score strip...")
    plot_viso_video_score_strip(df_frames)

    # Headline summary.
    print("\n" + "=" * 80)
    print("HEADLINE SUMMARY (frame-level, at deployed τ)")
    print("=" * 80)
    headline = []
    for suite in FAKE_SUITES + REAL_SUITES:
        for model in MODELS:
            g = df_frames[(df_frames["suite"] == suite) & (df_frames["model"] == model)]
            if g.empty:
                continue
            tau = DEPLOYED_TAU[model]
            metric = "RECALL" if g["label"].iloc[0] == 1 else "FPR"
            frac = float((g["frame_prob"] >= tau).mean())
            mean_p = g["frame_prob"].mean()
            p50 = g["frame_prob"].median()
            p99 = float(np.quantile(g["frame_prob"], 0.99))
            headline.append({
                "suite": suite, "model": model, "n": len(g),
                f"{metric}@deployed_tau": frac,
                "mean_prob": mean_p, "p50": p50, "p99": p99, "deployed_tau": tau,
            })
    summary_df = pd.DataFrame(headline)
    summary_df.to_csv(OUT / "headline_summary.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"\n[done] artifacts in {OUT}")


if __name__ == "__main__":
    main()
