"""Walkthrough all 47 lockbox fake frames.

Lockbox fake suite (teams_fake_all_lockbox) contains 425 frames per model
(2 unique methods, multiple frames per video). At video level n=253. The CPU
diagnostics doc cites n=47 unique lockbox fakes used in the embedding triptych
substrate. This script downloads ALL 425 frames, then renders a sorted gallery
+ per-frame score table.

Outputs:
  outputs/lockbox_fake_walkthrough/<frame>.png (downloaded)
  outputs/lockbox_fake_per_frame.csv (per-frame scores)
  outputs/figures/lockbox_fake_gallery_<method>_<model>.png
"""

from __future__ import annotations

import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
DEST = OUT / "lockbox_fake_walkthrough"
DEST.mkdir(parents=True, exist_ok=True)

DEPLOYED_TAU = {"P8A": 0.990946, "P18T": 0.99359, "P18C": 0.994625}


def fetch(uri: str) -> Path:
    name = uri.rsplit("/", 1)[-1]
    p = DEST / name
    if p.exists() and p.stat().st_size > 0:
        return p
    subprocess.run(["gsutil", "-q", "cp", uri, str(p)], check=False)
    return p


def main():
    df = pd.read_parquet(OUT / "combined_frames.parquet")
    sub = df[df["suite"] == "teams_fake_all_lockbox"].copy()
    print(f"[load] {len(sub)} (suite × model × frame) rows; {sub['frame_path'].nunique()} unique frames")
    print(f"[load] methods: {sorted(sub['method'].unique())}")

    # Per-frame, per-model score matrix
    pivot = sub.pivot_table(index=["method", "video_id", "frame_path"], columns="model",
                            values="frame_prob", aggfunc="first").reset_index()
    pivot.columns.name = None
    pivot["max_score"] = pivot[["P8A", "P18T", "P18C"]].max(axis=1)
    pivot["min_score"] = pivot[["P8A", "P18T", "P18C"]].min(axis=1)
    # Catch flags at deployed tau
    for m in ["P8A", "P18T", "P18C"]:
        pivot[f"{m}_caught_at_deployed"] = (pivot[m] >= DEPLOYED_TAU[m]).astype(int)
        pivot[f"{m}_caught_at_0p5"] = (pivot[m] >= 0.5).astype(int)
    pivot["caught_by_any_at_deployed"] = (pivot[["P8A_caught_at_deployed", "P18T_caught_at_deployed", "P18C_caught_at_deployed"]].sum(axis=1) > 0).astype(int)
    pivot["caught_by_all_at_deployed"] = (pivot[["P8A_caught_at_deployed", "P18T_caught_at_deployed", "P18C_caught_at_deployed"]].sum(axis=1) == 3).astype(int)
    pivot.to_csv(OUT / "lockbox_fake_per_frame.csv", index=False)

    # Per-method recall summary (frame-level)
    print("\n=== Per-method lockbox fake recall (frame-level) ===")
    summary = []
    for method in sorted(sub["method"].unique()):
        for model in ["P8A", "P18T", "P18C"]:
            ms = sub[(sub["method"] == method) & (sub["model"] == model)]
            tau_dep = DEPLOYED_TAU[model]
            row = {
                "method": method, "model": model, "n_frames": len(ms),
                "recall_at_deployed_tau": float((ms["frame_prob"] >= tau_dep).mean()),
                "recall_at_0p9": float((ms["frame_prob"] >= 0.9).mean()),
                "recall_at_0p5": float((ms["frame_prob"] >= 0.5).mean()),
                "median_prob": float(ms["frame_prob"].median()),
                "mean_prob": float(ms["frame_prob"].mean()),
            }
            summary.append(row)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT / "lockbox_fake_method_summary.csv", index=False)
    print(summary_df.to_string(index=False, float_format="%.3f"))

    # ── Always-caught vs never-caught at deployed tau ──
    print("\n=== Per-method catch pattern at deployed tau ===")
    for method in sorted(sub["method"].unique()):
        m_pivot = pivot[pivot["method"] == method]
        n_total = len(m_pivot)
        n_any = m_pivot["caught_by_any_at_deployed"].sum()
        n_all = m_pivot["caught_by_all_at_deployed"].sum()
        n_none = (m_pivot["caught_by_any_at_deployed"] == 0).sum()
        print(f"\n{method} (n={n_total} frames):")
        print(f"  caught by ANY model at deployed τ: {n_any} ({n_any/n_total:.1%})")
        print(f"  caught by ALL 3 at deployed τ:    {n_all} ({n_all/n_total:.1%})")
        print(f"  caught by NO model at deployed τ: {n_none} ({n_none/n_total:.1%})")

    # ── Download a sample (cap at 60 frames per method, sorted by P8A score) ──
    print("\n[fetch] downloading lockbox fake frames (capped sample)...")
    uris_to_fetch = []
    for method in sorted(sub["method"].unique()):
        m_pivot = pivot[pivot["method"] == method].sort_values("P8A")
        # Take 30 worst (lowest P8A score) + 30 best, dedupe
        sample = pd.concat([m_pivot.head(30), m_pivot.tail(30)]).drop_duplicates(subset="frame_path")
        uris_to_fetch.extend(sample["frame_path"].tolist())
    uris_to_fetch = list(set(uris_to_fetch))
    print(f"[fetch] {len(uris_to_fetch)} unique URIs")
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(fetch, uris_to_fetch))

    # ── Render gallery: per method, top 24 worst-scored (most missed) + top 8 best-scored ──
    print("\n[plot] rendering galleries...")
    for method in sorted(sub["method"].unique()):
        m_pivot = pivot[pivot["method"] == method].sort_values("P8A")
        n_show = min(32, len(m_pivot))
        # 24 worst + 8 best
        worst = m_pivot.head(24)
        best = m_pivot.tail(min(8, len(m_pivot) - 24)) if len(m_pivot) > 24 else pd.DataFrame()
        gallery = pd.concat([worst, best])
        n_actual = len(gallery)
        cols = 6
        rows = int(np.ceil(n_actual / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))
        for i, (_, row) in enumerate(gallery.iterrows()):
            ax = axes.flat[i] if n_actual > 1 else axes
            local = DEST / Path(row["frame_path"]).name
            if local.exists():
                try:
                    arr = np.asarray(Image.open(local).convert("RGB"))
                    ax.imshow(arr)
                except Exception:
                    pass
            color = "green" if row["P8A_caught_at_deployed"] else "red"
            tag = "WORST" if i < 24 else "BEST"
            ax.set_title(f"[{tag}] P8A:{row['P8A']:.2f} P18T:{row['P18T']:.2f} P18C:{row['P18C']:.2f}", fontsize=7, color=color)
            ax.set_xticks([]); ax.set_yticks([])
        for i in range(n_actual, rows * cols):
            axes.flat[i].axis("off")
        fig.suptitle(f"Lockbox fake walkthrough — {method} — sorted by P8A score (worst→best)", fontsize=11)
        fig.tight_layout()
        fig.savefig(FIG / f"lockbox_fake_gallery_{method}.png", dpi=110)
        plt.close(fig)

    print(f"\n[done] outputs in {OUT}")


if __name__ == "__main__":
    main()
