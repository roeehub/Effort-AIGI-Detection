#!/usr/bin/env python3
"""D1-D3 analyses on per-frame canary scores produced by run_inference.py.

Inputs:
  scores/_canary_meta.csv  (frame metadata + p8a_reference_score)
  scores/<ckpt_id>.csv     (per-frame prob_fake)

Outputs:
  outputs/d1_distribution_stats.csv     (per-ckpt × label/cohort stats)
  outputs/d2_slotD_pair_compare.csv     (per-frame step6000 vs step19000 deltas)
  outputs/d2_slotD_per_cohort.csv       (cohort-level summary)
  outputs/d3_distribution_overlap.csv   (per-ckpt × cohort histograms)
  figs/d3_hist_<cohort>.png             (per-cohort overlaid histograms)
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO,
                    datefmt="%H:%M:%S")
log = logging.getLogger("analyze_scores")

THIS = Path(__file__).resolve().parent
SCORES = THIS / "scores"
OUT = THIS / "outputs"
FIGS = THIS / "figs"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

CKPTS = [
    "slotA_top_n_step500", "slotB_top_n_step500", "slotC_top_n_step7000",
    "slotD_top_n_step6000", "slotD_top_n_step19000",
]


def quantile_table(s: pd.Series) -> dict:
    s = s.dropna().astype(float)
    if len(s) == 0:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def load_all() -> pd.DataFrame:
    meta = pd.read_csv(SCORES / "_canary_meta.csv")
    df = meta.copy()
    for c in CKPTS:
        path = SCORES / f"{c}.csv"
        if not path.exists():
            log.warning("missing %s -> skip", path)
            continue
        s = pd.read_csv(path)[["frame_idx", "frame_path", "prob_fake"]]
        s = s.rename(columns={"prob_fake": c})
        df = df.merge(s, on=["frame_idx", "frame_path"], how="left")
    return df


def d1_distribution_stats(df: pd.DataFrame):
    """Per-ckpt × (overall, label, cohort) summary stats."""
    rows = []
    score_cols = [c for c in CKPTS if c in df.columns] + ["p8a_reference_score"]
    for ckpt in score_cols:
        rows.append({"ckpt": ckpt, "scope": "all", "key": "all",
                     **quantile_table(df[ckpt])})
        for lab, sub in df.groupby("label"):
            rows.append({"ckpt": ckpt, "scope": "label", "key": int(lab),
                         **quantile_table(sub[ckpt])})
        for coh, sub in df.groupby("cohort"):
            rows.append({"ckpt": ckpt, "scope": "cohort", "key": coh,
                         **quantile_table(sub[ckpt])})
    out = pd.DataFrame(rows)
    out_csv = OUT / "d1_distribution_stats.csv"
    out.to_csv(out_csv, index=False)
    log.info("wrote %s (n=%d)", out_csv, len(out))
    print()
    print("D1 — overall + per-label distribution preview:")
    preview = out[(out["scope"].isin(["all", "label"]))].copy()
    print(preview[["ckpt", "scope", "key", "n", "mean", "p05", "p50", "p95", "min", "max"]]
          .round(4).to_string(index=False))


def d2_slotD_pair_compare(df: pd.DataFrame):
    """Per-frame Slot D step6000 vs step19000."""
    a = "slotD_top_n_step6000"
    b = "slotD_top_n_step19000"
    if a not in df.columns or b not in df.columns:
        log.warning("Slot D pair missing -> skip D2")
        return
    sub = df.dropna(subset=[a, b]).copy()
    sub["delta_19k_minus_6k"] = sub[b] - sub[a]
    pair_csv = OUT / "d2_slotD_pair_compare.csv"
    sub[["frame_idx", "frame_path", "label", "cohort", "base_identity", "suite",
         a, b, "delta_19k_minus_6k", "p8a_reference_score"]].to_csv(pair_csv, index=False)
    log.info("wrote %s (n=%d)", pair_csv, len(sub))
    rows = []
    for (coh, lab), grp in sub.groupby(["cohort", "label"]):
        rows.append({"cohort": coh, "label": int(lab), "n": len(grp),
                     "step6000_mean": float(grp[a].mean()),
                     "step19000_mean": float(grp[b].mean()),
                     "delta_mean": float(grp["delta_19k_minus_6k"].mean()),
                     "delta_p50": float(grp["delta_19k_minus_6k"].quantile(0.5)),
                     "delta_min": float(grp["delta_19k_minus_6k"].min()),
                     "delta_max": float(grp["delta_19k_minus_6k"].max()),
                     "p8a_mean": float(grp["p8a_reference_score"].mean())})
    coh_csv = OUT / "d2_slotD_per_cohort.csv"
    pd.DataFrame(rows).to_csv(coh_csv, index=False)
    log.info("wrote %s", coh_csv)
    print()
    print("D2 — per-cohort step19000 vs step6000 (Slot D) — sorted by abs(delta_mean):")
    cdf = pd.DataFrame(rows).copy()
    cdf["abs_delta"] = cdf["delta_mean"].abs()
    print(cdf.sort_values("abs_delta", ascending=False).round(4).to_string(index=False))


def d3_histograms(df: pd.DataFrame):
    """Per-cohort overlaid histograms of all ckpts + P8A."""
    score_cols = [c for c in CKPTS if c in df.columns] + ["p8a_reference_score"]
    cohorts = sorted(df["cohort"].unique())
    rows = []
    for ckpt in score_cols:
        for coh in cohorts:
            sub = df[df["cohort"] == coh][ckpt].dropna()
            if not len(sub):
                continue
            rows.append({"ckpt": ckpt, "cohort": coh, "n": int(len(sub)),
                         **quantile_table(sub)})
    csv_path = OUT / "d3_distribution_overlap.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log.info("wrote %s", csv_path)
    bins = np.linspace(0, 1, 41)
    for coh in cohorts:
        sub = df[df["cohort"] == coh]
        if len(sub) < 5:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        for ckpt in score_cols:
            vals = sub[ckpt].dropna().values
            if not len(vals):
                continue
            ax.hist(vals, bins=bins, histtype="step", label=ckpt, alpha=0.85, linewidth=1.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("prob_fake")
        ax.set_ylabel("count (frames)")
        ax.set_title(f"cohort={coh} (n={len(sub)}, label={int(sub['label'].iloc[0])})")
        ax.legend(fontsize=7, loc="upper center")
        fig.tight_layout()
        fig.savefig(FIGS / f"d3_hist_{coh}.png", dpi=110)
        plt.close(fig)
    # Also a single grand-overlay figure split by label
    for lab in sorted(df["label"].unique()):
        fig, ax = plt.subplots(figsize=(8, 4))
        for ckpt in score_cols:
            vals = df[df["label"] == lab][ckpt].dropna().values
            if not len(vals):
                continue
            ax.hist(vals, bins=bins, histtype="step", label=ckpt, alpha=0.85, linewidth=1.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("prob_fake")
        ax.set_ylabel("count (frames)")
        ax.set_title(f"label={int(lab)} (n={int((df['label']==lab).sum())})")
        ax.legend(fontsize=7, loc="upper center")
        fig.tight_layout()
        fig.savefig(FIGS / f"d3_hist_label_{int(lab)}.png", dpi=110)
        plt.close(fig)
    log.info("wrote %d cohort + 2 label histogram PNGs to %s", len(cohorts), FIGS)


def main():
    df = load_all()
    log.info("merged frame table: %s rows × %s cols", *df.shape)
    d1_distribution_stats(df)
    d2_slotD_pair_compare(df)
    d3_histograms(df)
    log.info("DONE.")


if __name__ == "__main__":
    main()
