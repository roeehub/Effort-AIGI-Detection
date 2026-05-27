"""Paired raw-vs-teams substrate analysis on visomaster_enhanced_macro_dev.

The 550-frame slice is exactly 50/50 raw vs teams substrate, with the same 275
sequence IDs in both halves (paired recapture). This script:

  1. Identifies the pairs.
  2. Computes per-pair raw_score, teams_score, delta = teams - raw.
  3. Plots paired-scatter (raw vs teams) per model.
  4. Plots delta histograms per model.
  5. Categorises pairs into {both_caught, raw_only, teams_only, both_missed} at τ=0.5.
  6. Selects a curated set of 4 frames each from {raw_only, teams_only, both_caught,
     both_missed} for visual inspection.
  7. Saves CSV + PNGs.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
SAMPLES = OUT / "viso_paired_samples"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SAMPLES.mkdir(parents=True, exist_ok=True)

DEPLOYED_TAU = {"P8A": 0.990946, "P18T": 0.99359, "P18C": 0.994625}

PATTERN = re.compile(r"^.*/visomaster_enhanced_(raw|teams)__frame_(\d+)_(seq\d+)\.png$")


def parse_subtype(p: str):
    m = PATTERN.match(p)
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), m.group(3)


def build_pairs(df_frames: pd.DataFrame) -> pd.DataFrame:
    viso = df_frames[df_frames["suite"] == "visomaster_enhanced_macro_dev"].copy()
    parsed = viso["frame_path"].apply(parse_subtype)
    viso["subtype"] = parsed.apply(lambda x: x[0])
    viso["frame_num"] = parsed.apply(lambda x: x[1])
    viso["seq_id"] = parsed.apply(lambda x: x[2])
    pivot = viso.pivot_table(
        index=["model", "seq_id"], columns="subtype",
        values=["frame_prob", "frame_path"], aggfunc="first"
    ).reset_index()
    pivot.columns = [f"{a}_{b}" if b else a for a, b in pivot.columns]
    pivot["delta"] = pivot["frame_prob_teams"] - pivot["frame_prob_raw"]
    return pivot


def plot_paired_scatter(pairs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, model in zip(axes, ["P8A", "P18T", "P18C"]):
        sub = pairs[pairs["model"] == model]
        ax.scatter(sub["frame_prob_raw"], sub["frame_prob_teams"], s=12, alpha=0.55, color="steelblue")
        ax.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1, alpha=0.7, label="y=x")
        tau = DEPLOYED_TAU[model]
        ax.axvline(tau, color="grey", linestyle=":", linewidth=1, alpha=0.5)
        ax.axhline(tau, color="grey", linestyle=":", linewidth=1, alpha=0.5)
        ax.axvline(0.5, color="black", linestyle=":", linewidth=0.6, alpha=0.4)
        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.6, alpha=0.4)
        r = sub[["frame_prob_raw", "frame_prob_teams"]].corr().iloc[0, 1]
        ax.set_title(
            f"{model} · Pearson r={r:.3f}\n"
            f"raw mean={sub['frame_prob_raw'].mean():.3f}  teams mean={sub['frame_prob_teams'].mean():.3f}"
        )
        ax.set_xlabel("raw frame_prob")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("teams frame_prob")
    fig.suptitle("Paired viso scores: raw vs teams (n=275 sequences each)")
    fig.tight_layout()
    fig.savefig(FIG / "viso_paired_scatter.png", dpi=120)
    plt.close(fig)


def plot_delta_hist(pairs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    bins = np.linspace(-1, 1, 41)
    for ax, model in zip(axes, ["P8A", "P18T", "P18C"]):
        sub = pairs[pairs["model"] == model]
        ax.hist(sub["delta"], bins=bins, color="purple", edgecolor="black", alpha=0.75)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(
            f"{model} — Δ = teams - raw\n"
            f"mean={sub['delta'].mean():+.3f}  median={sub['delta'].median():+.3f}"
        )
        ax.set_xlabel("teams - raw")
        ax.set_xlim(-1, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("count")
    fig.suptitle("Per-sequence Δ score (teams substrate vs raw substrate)")
    fig.tight_layout()
    fig.savefig(FIG / "viso_paired_delta.png", dpi=120)
    plt.close(fig)


def categorize(pairs: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    p = pairs.copy()
    raw_high = p["frame_prob_raw"] >= threshold
    teams_high = p["frame_prob_teams"] >= threshold
    p["category"] = np.where(raw_high & teams_high, "both_caught",
                     np.where(raw_high & ~teams_high, "raw_only",
                     np.where(~raw_high & teams_high, "teams_only",
                              "both_missed")))
    return p


def select_samples(categorized: pd.DataFrame, model: str, n_per_cat: int = 4) -> Dict[str, pd.DataFrame]:
    sub = categorized[categorized["model"] == model]
    out = {}
    rng = np.random.RandomState(737)
    for cat in ["both_caught", "raw_only", "teams_only", "both_missed"]:
        cands = sub[sub["category"] == cat]
        if len(cands) == 0:
            out[cat] = pd.DataFrame()
            continue
        idx = rng.choice(len(cands), min(n_per_cat, len(cands)), replace=False)
        out[cat] = cands.iloc[idx].reset_index(drop=True)
    return out


def download_frames(samples: Dict[str, pd.DataFrame], model: str) -> List[Dict]:
    """Download both raw and teams variants for each sample. Build manifest for viewer."""
    manifest_rows = []
    for cat, df in samples.items():
        if df.empty:
            continue
        for _, row in df.iterrows():
            for subtype in ["raw", "teams"]:
                src = row[f"frame_path_{subtype}"]
                seq = row["seq_id"]
                local = SAMPLES / model / cat / f"{seq}__{subtype}.png"
                local.parent.mkdir(parents=True, exist_ok=True)
                if not local.exists():
                    subprocess.run(["gsutil", "-q", "cp", src, str(local)], check=False)
                manifest_rows.append({
                    "model": model,
                    "category": cat,
                    "seq_id": seq,
                    "subtype": subtype,
                    "raw_score": float(row["frame_prob_raw"]),
                    "teams_score": float(row["frame_prob_teams"]),
                    "delta": float(row["delta"]),
                    "gcs_uri": src,
                    "local_path": str(local.relative_to(ROOT.parent.parent)) if local.exists() else "",
                    "downloaded": local.exists(),
                })
    return manifest_rows


def plot_sample_grid(samples: Dict[str, pd.DataFrame], model: str) -> None:
    """Render a 4-row × 8-col grid: 4 categories × (4 sequences × 2 subtypes) for one model."""
    from PIL import Image
    cats = ["both_caught", "raw_only", "teams_only", "both_missed"]
    n_per = 4
    fig, axes = plt.subplots(len(cats), n_per * 2, figsize=(16, 12))
    if len(cats) == 1:
        axes = np.array([axes])
    for r, cat in enumerate(cats):
        df = samples[cat]
        for c in range(n_per):
            ax_raw = axes[r, c * 2]
            ax_teams = axes[r, c * 2 + 1]
            if c < len(df):
                seq = df.iloc[c]["seq_id"]
                raw_score = df.iloc[c]["frame_prob_raw"]
                teams_score = df.iloc[c]["frame_prob_teams"]
                for ax, subtype, score in [(ax_raw, "raw", raw_score), (ax_teams, "teams", teams_score)]:
                    p = SAMPLES / model / cat / f"{seq}__{subtype}.png"
                    if p.exists():
                        try:
                            ax.imshow(Image.open(p))
                        except Exception:
                            pass
                    ax.set_title(f"{subtype} · {score:.2f}", fontsize=7)
                    ax.set_xticks([])
                    ax.set_yticks([])
                if c == 0:
                    ax_raw.set_ylabel(cat, fontsize=10)
            else:
                ax_raw.axis("off")
                ax_teams.axis("off")
    fig.suptitle(
        f"viso paired samples · {model} · 4 categories × 4 sequences × {{raw, teams}}\n"
        f"Each pair is the SAME source video processed two ways. Score in title.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG / f"viso_paired_grid_{model}.png", dpi=110)
    plt.close(fig)


def main():
    print("[load] reading combined frames...")
    df_frames = pd.read_parquet(OUT / "combined_frames.parquet")

    print("[pairs] building paired raw/teams table...")
    pairs = build_pairs(df_frames)
    pairs.to_csv(OUT / "viso_pairs.csv", index=False)
    print(f"[pairs] total rows: {len(pairs)} (= 275 seqs × 3 models)")

    print("[plot] paired scatter + delta hist...")
    plot_paired_scatter(pairs)
    plot_delta_hist(pairs)

    print("[categorize] @ τ=0.5...")
    categorized = categorize(pairs, threshold=0.5)
    cat_summary = categorized.groupby(["model", "category"]).size().unstack(fill_value=0)
    cat_summary.to_csv(OUT / "viso_pair_category_counts.csv")
    print(cat_summary.to_string())

    print("\n[samples] selecting 4 sequences × 4 categories per model...")
    all_manifest = []
    for model in ["P8A", "P18T", "P18C"]:
        print(f"[samples] {model} ...")
        samples = select_samples(categorized, model, n_per_cat=4)
        print(f"[samples] {model}: " + ", ".join(f"{c}={len(d)}" for c, d in samples.items()))
        manifest_rows = download_frames(samples, model)
        all_manifest.extend(manifest_rows)
        print(f"[samples] {model}: rendering grid...")
        plot_sample_grid(samples, model)

    pd.DataFrame(all_manifest).to_csv(OUT / "viso_paired_sample_manifest.csv", index=False)
    print(f"[samples] total downloaded: {sum(1 for r in all_manifest if r['downloaded'])} / {len(all_manifest)}")
    print(f"[done] outputs in {OUT}")


if __name__ == "__main__":
    main()
