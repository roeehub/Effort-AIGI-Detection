"""Visual identity-comparison gallery.

Builds a side-by-side gallery showing 8 frames each from:
  - dor (dev real, 0% FPR)
  - dor_shkedi (lockbox real, FPR driver)
  - real_dor (lockbox real, separate naming)
  - deeplive_dor (deeplive_enhanced_dev FAKE, presumably the dor person)
  - Roy_D (dev real, 4.6% FPR)
  - Q (dev real, 51.9% FPR)
  - Cam_Test (the lockbox cam_test_s33 cropped subject)
  - bla_bla_chow (dev real, 0.4% FPR)

If dor / dor_shkedi / real_dor / deeplive_dor are visually the same person, that's
the identity-overlap reality. If Roy_D and Q look similar to each other, that may
hint at a shared characteristic the model fixates on.

Per-identity prob_fake histogram also plotted.
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
DEST = OUT / "identity_comparison"
DEST.mkdir(parents=True, exist_ok=True)


def parse_identity(p: str) -> str:
    name = p.rsplit("/", 1)[-1]
    if "__" in name:
        return name.split("__", 1)[0]
    return name.rsplit(".", 1)[0]


def fetch(uri: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gsutil", "-q", "cp", uri, str(dest)], check=False)
    return dest


def main():
    df = pd.read_parquet(OUT / "combined_frames.parquet")

    # Identity, suite, label
    targets = [
        ("dor", "teams_real_all_dev", 0),
        ("dor_shkedi", "teams_real_all_lockbox", 0),
        ("real_dor", "teams_real_all_lockbox", 0),
        ("deeplive_dor", "deeplive_enhanced_dev", 1),  # This is the deeplive identity_key prefix
        ("Roy_D", "teams_real_all_dev", 0),
        ("Q", "teams_real_all_dev", 0),
        ("Cam_Test", "teams_real_all_dev", 0),
        ("bla_bla_chow", "teams_real_all_dev", 0),
    ]

    # Collect 8 frames per (identity, suite)
    all_frames = []  # (label_str, frame_paths_list, p8a_scores)
    for ident_str, suite, label in targets:
        sub = df[(df["suite"] == suite) & (df["model"] == "P8A") & (df["label"] == label)].copy()
        sub["ident_parsed"] = sub["frame_path"].apply(parse_identity)
        # Try exact identity match
        matches = sub[sub["ident_parsed"] == ident_str]
        if matches.empty:
            # For deeplive_dor, the parser prefix may differ (e.g., 'deeplive_dor' vs 'deeplive')
            matches = sub[sub["ident_parsed"].str.startswith(ident_str)]
        if matches.empty:
            print(f"  [warn] no frames for {ident_str} in {suite}")
            continue
        rng = np.random.RandomState(737)
        idx = rng.choice(len(matches), min(8, len(matches)), replace=False)
        sample = matches.iloc[idx]
        frames = []
        scores = []
        for _, row in sample.iterrows():
            uri = row["frame_path"]
            local = DEST / ident_str / Path(uri).name
            frames.append((uri, local))
            scores.append((row["frame_prob"], row["video_id"]))
        all_frames.append((f"{ident_str}\n({suite[:30]})\nlabel={label}", frames, scores))

    # Parallel download
    print("[fetch] downloading all sample frames...")
    todo = []
    for _, frames, _ in all_frames:
        for uri, local in frames:
            todo.append((uri, local))
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(lambda args: fetch(args[0], args[1]), todo))

    # Render gallery: one row per identity, 8 columns
    n_rows = len(all_frames)
    n_cols = 8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.4, n_rows * 2.7))
    if n_rows == 1:
        axes = np.array([axes])
    for r, (label_str, frames, scores) in enumerate(all_frames):
        for c in range(n_cols):
            ax = axes[r, c]
            if c < len(frames):
                _, local = frames[c]
                if local.exists():
                    try:
                        arr = np.asarray(Image.open(local).convert("RGB"))
                        ax.imshow(arr)
                    except Exception:
                        pass
                score, vid = scores[c]
                ax.set_title(f"P8A:{score:.2f}", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label_str, fontsize=8, rotation=0, ha="right", va="center", labelpad=110)
    fig.suptitle(
        "Identity comparison gallery — dor variants + FPR drivers + lockbox cam_test_s33 subject\n"
        "Same person across rows? Same person between dor/dor_shkedi/real_dor/deeplive_dor?",
        fontsize=11, y=0.99
    )
    fig.tight_layout()
    fig.savefig(FIG / "identity_comparison_gallery.png", dpi=110)
    plt.close(fig)

    # ────────── Per-identity score histograms (P8A, P18T, P18C overlaid) ──────────
    # Use the full per-frame data, not just the 8-sample subset
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax_idx, (ident_str, suite, label) in enumerate(targets):
        ax = axes.flat[ax_idx]
        sub = df[(df["suite"] == suite) & (df["label"] == label)].copy()
        sub["ident_parsed"] = sub["frame_path"].apply(parse_identity)
        matches = sub[sub["ident_parsed"] == ident_str]
        if matches.empty:
            matches = sub[sub["ident_parsed"].str.startswith(ident_str)]
        if matches.empty:
            ax.set_title(f"{ident_str} — no frames"); ax.axis("off"); continue
        n_per_model = len(matches[matches["model"] == "P8A"])
        bins = np.linspace(0, 1, 30)
        for model, color in [("P8A", "#1f77b4"), ("P18T", "#ff7f0e"), ("P18C", "#2ca02c")]:
            m_data = matches[matches["model"] == model]["frame_prob"]
            ax.hist(m_data, bins=bins, alpha=0.45, color=color, label=f"{model} (n={len(m_data)})", edgecolor="black")
        ax.axvline(0.5, color="grey", linestyle=":", alpha=0.5)
        ax.set_title(f"{ident_str} (suite={suite[:25]}, label={'fake' if label==1 else 'real'}, n={n_per_model})", fontsize=9)
        ax.set_xlabel("frame_prob")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Per-identity prob_fake distribution (P8A/P18T/P18C overlaid)")
    fig.tight_layout()
    fig.savefig(FIG / "identity_score_histograms.png", dpi=120)
    plt.close(fig)

    print(f"[done] outputs in {OUT}")


if __name__ == "__main__":
    main()
