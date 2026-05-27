"""
Job C — Fake-vs-real AUC on resolution-chain panel for all 5 ckpts.

Per AGENT_GUIDE Rule 6 (added 2026-05-19): distinguishes encoder-level
resolution invariance (AUC preserved) from score-distribution compression
(AUC reduced). The 2026-05-15 probe reported only score_range; this script
adds the missing AUC measurement that the post-mortem flagged as essential.

Inputs:
- analysis/cpu_diagnostics_2026-05-15_resolution_chain/outputs/per_frame_per_variant.parquet
  (baseline: P8A_step5000, T5C_step3500, E2B_step3200; 24444 rows variant-level)
- analysis/cpu_diagnostics_2026-05-15_resolution_chain/outputs_new_ckpts/per_frame_per_variant_new_ckpts.parquet
  (new ckpts: SLOT_A_RESCHAIN steps 1500/3500, SLOT_B_6AXIS_GRL; 16296 rows variant-level)

Outputs:
- outputs/job_c_auc_by_ckpt_size.csv — (ckpt, down_size, kernel, AUC) — fine-grained
- outputs/job_c_auc_summary.csv — per (ckpt, down_size) median AUC across kernels
- outputs/job_c_baseline_auc.csv — baseline (no aug) per-ckpt AUC
- figs/job_c_auc_vs_size.png — line chart per ckpt
- logs/job_c.log
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis/cpu_diagnostics_2026-05-19_pre_plan"

BASELINE_PARQUET = ROOT / "analysis/cpu_diagnostics_2026-05-15_resolution_chain/outputs/per_frame_per_variant.parquet"
NEW_PARQUET = ROOT / "analysis/cpu_diagnostics_2026-05-15_resolution_chain/outputs_new_ckpts/per_frame_per_variant_new_ckpts.parquet"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_combined() -> pd.DataFrame:
    df1 = pd.read_parquet(BASELINE_PARQUET)
    df2 = pd.read_parquet(NEW_PARQUET)
    log(f"Baseline parquet: {df1.shape} | columns {list(df1.columns)}")
    log(f"New parquet:      {df2.shape} | columns {list(df2.columns)}")
    # Align columns
    common = sorted(set(df1.columns) & set(df2.columns))
    df1 = df1[common].copy()
    df2 = df2[common].copy()
    df = pd.concat([df1, df2], ignore_index=True)
    log(f"Combined: {df.shape}")
    log(f"Ckpts: {sorted(df['ckpt'].unique())}")
    log(f"Cohorts: {sorted(df['cohort'].unique())}")
    log(f"Label values: {sorted(df['label'].unique())}")
    return df


def auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_auc_per_variant(df: pd.DataFrame) -> pd.DataFrame:
    """Per (ckpt, down_size, kernel) — AUC across all panel frames (388 frames)."""
    rows = []
    for (ckpt, dsize, kernel), g in df.groupby(["ckpt", "down_size", "kernel"]):
        auc = auc_safe(g["label"].to_numpy(), g["score"].to_numpy())
        n_real = int((g["label"] == 0).sum())
        n_fake = int((g["label"] == 1).sum())
        rows.append(
            {
                "ckpt": ckpt,
                "down_size": dsize,
                "kernel": kernel,
                "n_frames": len(g),
                "n_real": n_real,
                "n_fake": n_fake,
                "auc": auc,
                "mean_real_score": float(g.loc[g["label"] == 0, "score"].mean()),
                "mean_fake_score": float(g.loc[g["label"] == 1, "score"].mean()),
                "score_std_real": float(g.loc[g["label"] == 0, "score"].std()),
                "score_std_fake": float(g.loc[g["label"] == 1, "score"].std()),
            }
        )
    return pd.DataFrame(rows).sort_values(["ckpt", "down_size", "kernel"]).reset_index(drop=True)


def baseline_auc(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ckpt baseline AUC at variant_key='baseline' (no aug)."""
    base = df[df["variant_key"] == "baseline"].copy()
    if base.empty:
        # Probe encodes baseline as down_size=-1 (no augmentation applied)
        base = df[df["down_size"] == -1].copy()
    if base.empty:
        log("WARNING: no baseline rows found by any heuristic")
        return pd.DataFrame(columns=["ckpt", "baseline_auc"])
    rows = []
    for ckpt, g in base.groupby("ckpt"):
        auc = auc_safe(g["label"].to_numpy(), g["score"].to_numpy())
        rows.append(
            {
                "ckpt": ckpt,
                "n_frames": len(g),
                "n_real": int((g["label"] == 0).sum()),
                "n_fake": int((g["label"] == 1).sum()),
                "baseline_auc": auc,
                "mean_real_score": float(g.loc[g["label"] == 0, "score"].mean()),
                "mean_fake_score": float(g.loc[g["label"] == 1, "score"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_size(per_variant: pd.DataFrame) -> pd.DataFrame:
    """Median across kernels per (ckpt, down_size)."""
    agg = (
        per_variant.groupby(["ckpt", "down_size"])
        .agg(
            n_kernels=("kernel", "nunique"),
            median_auc=("auc", "median"),
            min_auc=("auc", "min"),
            max_auc=("auc", "max"),
            mean_real_score=("mean_real_score", "mean"),
            mean_fake_score=("mean_fake_score", "mean"),
        )
        .reset_index()
    )
    return agg.sort_values(["ckpt", "down_size"]).reset_index(drop=True)


def plot_auc_vs_size(summary: pd.DataFrame, base: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ckpt_order = sorted(summary["ckpt"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(ckpt_order)))
    for color, ckpt in zip(colors, ckpt_order):
        sub = summary[summary["ckpt"] == ckpt].sort_values("down_size")
        ax.plot(sub["down_size"], sub["median_auc"], "o-", label=ckpt, color=color)
        # Add baseline AUC as horizontal dotted line
        b = base[base["ckpt"] == ckpt]
        if not b.empty and not np.isnan(b["baseline_auc"].iloc[0]):
            ax.axhline(b["baseline_auc"].iloc[0], color=color, linestyle=":", alpha=0.5)
    ax.set_xlabel("downsample target size (px)")
    ax.set_ylabel("fake-vs-real AUC (median across kernels)")
    ax.set_title("AUC vs resolution-chain perturbation\n(dotted = no-aug baseline)")
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    log(f"Saved figure: {out_path}")


def main() -> int:
    log_file = OUT / "logs/job_c.log"
    sys.stdout = open(log_file, "w", buffering=1)
    sys.stderr = sys.stdout

    log("=== Job C: fake-vs-real AUC on resolution-chain panel ===")
    df = load_combined()

    per_variant = compute_auc_per_variant(df)
    out1 = OUT / "outputs/job_c_auc_by_ckpt_size.csv"
    per_variant.to_csv(out1, index=False)
    log(f"\nSaved per-variant AUC: {out1} ({len(per_variant)} rows)")

    base = baseline_auc(df)
    out2 = OUT / "outputs/job_c_baseline_auc.csv"
    base.to_csv(out2, index=False)
    log(f"Saved baseline AUC:    {out2}")
    log("\n=== BASELINE AUC PER CKPT ===")
    log(base.to_string(index=False))

    summary = summarize_by_size(per_variant)
    out3 = OUT / "outputs/job_c_auc_summary.csv"
    summary.to_csv(out3, index=False)
    log(f"\nSaved per-size summary: {out3}")
    log("\n=== AUC SUMMARY BY CKPT × DOWN_SIZE ===")
    log(summary.to_string(index=False))

    # Delta vs baseline
    if not base.empty:
        delta_rows = []
        for ckpt in summary["ckpt"].unique():
            b = base[base["ckpt"] == ckpt]
            if b.empty or np.isnan(b["baseline_auc"].iloc[0]):
                continue
            b_auc = b["baseline_auc"].iloc[0]
            sub = summary[summary["ckpt"] == ckpt]
            for _, r in sub.iterrows():
                delta_rows.append(
                    {
                        "ckpt": ckpt,
                        "down_size": int(r["down_size"]),
                        "baseline_auc": b_auc,
                        "median_aug_auc": r["median_auc"],
                        "auc_delta": r["median_auc"] - b_auc,
                    }
                )
        if delta_rows:
            delta_df = pd.DataFrame(delta_rows)
            out4 = OUT / "outputs/job_c_auc_delta_vs_baseline.csv"
            delta_df.to_csv(out4, index=False)
            log(f"\nSaved AUC delta: {out4}")
            log("\n=== AUC DELTA (median_aug - baseline) ===")
            log(delta_df.to_string(index=False))

    plot_auc_vs_size(summary, base, OUT / "figs/job_c_auc_vs_size.png")
    log("\n=== Job C complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
