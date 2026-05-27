"""Feature-space embedding of the 550 viso frames using a pretrained
ImageNet ResNet-50 encoder, projected to 2D via t-SNE/PCA (sklearn-only).

Question: do uncatchable pairs cluster separately from catchable pairs in
generic image-feature space? If yes, there's a structural visual difference
that even an off-the-shelf encoder picks up. If no, the difference is more
subtle than pretrained features can resolve.

Output:
  outputs/viso_embeddings_resnet50.npy (550 x 2048)
  outputs/figures/embedding_pca_2d.png
  outputs/figures/embedding_tsne_2d.png
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torchvision.models as tvm
import torchvision.transforms as T

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FRAMES = OUT / "viso_full_paired"

PATTERN = re.compile(r"visomaster_enhanced_(raw|teams)__frame_(\d+)_(seq\d+)\.png")


def main():
    print("[load] resnet50 ImageNet pretrained...")
    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval()

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    files = sorted(FRAMES.glob("visomaster_enhanced_*.png"))
    print(f"[extract] {len(files)} frames")

    feats = []
    metadata = []
    batch = []
    batch_meta = []
    BATCH_SIZE = 16

    def flush():
        if not batch:
            return
        with torch.no_grad():
            x = torch.stack(batch)
            f = model(x).numpy()
        feats.append(f)
        metadata.extend(batch_meta)
        batch.clear()
        batch_meta.clear()

    for i, p in enumerate(files):
        m = PATTERN.search(p.name)
        if not m:
            continue
        subtype, frame_num, seq_id = m.groups()
        try:
            img = Image.open(p).convert("RGB")
            batch.append(transform(img))
            batch_meta.append({"seq_id": seq_id, "subtype": subtype, "filename": p.name})
        except Exception as e:
            print(f"  [skip] {p.name}: {e}")
            continue
        if len(batch) >= BATCH_SIZE:
            flush()
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(files)}]")
    flush()

    feats = np.concatenate(feats, axis=0)
    print(f"[extract] feature matrix: {feats.shape}")
    np.save(OUT / "viso_embeddings_resnet50.npy", feats)

    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(OUT / "viso_embeddings_resnet50_meta.csv", index=False)

    # Merge with pair categorisation
    pairs = pd.read_csv(OUT / "viso_pairs.csv")
    p8a_pairs = pairs[pairs["model"] == "P8A"].copy()
    rh = p8a_pairs["frame_prob_raw"] >= 0.5
    th = p8a_pairs["frame_prob_teams"] >= 0.5
    p8a_pairs["category_p8a"] = np.where(rh & th, "both_caught",
                              np.where(rh & ~th, "raw_only",
                              np.where(~rh & th, "teams_only", "both_missed")))
    pivot = pairs.pivot_table(index="seq_id", columns="model", values="frame_prob_raw", aggfunc="first")
    pivot.columns = [f"raw_score_{c}" for c in pivot.columns]
    pivot["never_caught_raw"] = (pivot < 0.5).all(axis=1)
    pivot = pivot.reset_index()

    # Per-frame score for the actual subtype shown
    pairs_long = pairs[pairs["model"] == "P8A"][["seq_id", "frame_prob_raw", "frame_prob_teams"]]

    meta_df = meta_df.merge(p8a_pairs[["seq_id", "category_p8a", "frame_prob_raw", "frame_prob_teams"]], on="seq_id", how="left")
    meta_df["frame_prob"] = np.where(meta_df["subtype"] == "raw", meta_df["frame_prob_raw"], meta_df["frame_prob_teams"])

    # PCA
    print("\n[PCA] computing 2D PCA...")
    pca = PCA(n_components=2, random_state=737)
    pca_xy = pca.fit_transform(feats)
    print(f"[PCA] explained variance: {pca.explained_variance_ratio_}")

    # t-SNE
    print("\n[t-SNE] computing 2D t-SNE (may take ~1 min)...")
    tsne = TSNE(n_components=2, random_state=737, perplexity=30, n_iter=1000, init="pca")
    tsne_xy = tsne.fit_transform(feats)

    meta_df["pca_x"] = pca_xy[:, 0]
    meta_df["pca_y"] = pca_xy[:, 1]
    meta_df["tsne_x"] = tsne_xy[:, 0]
    meta_df["tsne_y"] = tsne_xy[:, 1]
    meta_df.to_csv(OUT / "viso_embeddings_resnet50_meta.csv", index=False)

    # Plot: t-SNE colored by P8A score
    for proj_name, xs_ys in [("pca", (pca_xy[:, 0], pca_xy[:, 1])), ("tsne", (tsne_xy[:, 0], tsne_xy[:, 1]))]:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        # Color by subtype
        ax = axes[0, 0]
        for st, color in [("raw", "blue"), ("teams", "orange")]:
            mask = meta_df["subtype"] == st
            ax.scatter(xs_ys[0][mask], xs_ys[1][mask], s=15, alpha=0.6, color=color, label=st)
        ax.set_title(f"{proj_name.upper()} — colored by subtype")
        ax.legend()
        # Color by P8A score
        ax = axes[0, 1]
        sc = ax.scatter(xs_ys[0], xs_ys[1], s=15, alpha=0.7, c=meta_df["frame_prob"], cmap="RdYlGn")
        plt.colorbar(sc, ax=ax, label="P8A prob_fake")
        ax.set_title(f"{proj_name.upper()} — colored by P8A score (red=missed, green=caught)")
        # Color by category
        ax = axes[1, 0]
        cat_colors = {"both_caught": "#2ca02c", "raw_only": "#1f77b4", "teams_only": "#ff7f0e", "both_missed": "#d62728"}
        for cat, color in cat_colors.items():
            mask = meta_df["category_p8a"] == cat
            if mask.sum() == 0:
                continue
            ax.scatter(xs_ys[0][mask], xs_ys[1][mask], s=15, alpha=0.6, color=color, label=f"{cat} (n={mask.sum()//2})")
        ax.set_title(f"{proj_name.upper()} — colored by P8A pair category (at τ=0.5)")
        ax.legend(fontsize=8)
        # Color by sequence ID hash for visual structure
        ax = axes[1, 1]
        seq_colors = pd.factorize(meta_df["seq_id"])[0]
        ax.scatter(xs_ys[0], xs_ys[1], s=15, alpha=0.6, c=seq_colors, cmap="hsv")
        ax.set_title(f"{proj_name.upper()} — colored by seq_id (hashed). Pairs should be CLOSE if features agnostic to substrate")
        for ax in axes.flat:
            ax.set_xlabel(f"{proj_name}_x")
            ax.set_ylabel(f"{proj_name}_y")
            ax.grid(alpha=0.3)
        fig.suptitle(f"Viso 550-frame ResNet50 ImageNet feature embedding — {proj_name.upper()}")
        fig.tight_layout()
        fig.savefig(FIG / f"embedding_{proj_name}_2d.png", dpi=120)
        plt.close(fig)

    # Pair distance: for each seq, distance(raw, teams) in feature space
    print("\n[pairs] computing per-seq raw↔teams feature distance...")
    pair_dist = []
    for seq, g in meta_df.groupby("seq_id"):
        if len(g) != 2:
            continue
        raw_idx = g[g["subtype"] == "raw"].index[0]
        teams_idx = g[g["subtype"] == "teams"].index[0]
        d = float(np.linalg.norm(feats[raw_idx] - feats[teams_idx]))
        pair_dist.append({
            "seq_id": seq,
            "raw_teams_feature_distance": d,
            "category_p8a": g["category_p8a"].iloc[0],
            "raw_score_p8a": g["frame_prob_raw"].iloc[0],
            "teams_score_p8a": g["frame_prob_teams"].iloc[0],
        })
    pdf = pd.DataFrame(pair_dist)
    pdf.to_csv(OUT / "viso_pair_feature_distance.csv", index=False)

    print("\n=== Per-pair feature distance distribution by category ===")
    summary = pdf.groupby("category_p8a")["raw_teams_feature_distance"].agg(["count", "mean", "median", "std"])
    print(summary)

    # Plot distance distribution per category
    fig, ax = plt.subplots(figsize=(10, 5))
    cat_order = ["both_caught", "raw_only", "teams_only", "both_missed"]
    cat_colors = {"both_caught": "#2ca02c", "raw_only": "#1f77b4", "teams_only": "#ff7f0e", "both_missed": "#d62728"}
    data_per_cat = [pdf[pdf["category_p8a"] == c]["raw_teams_feature_distance"].dropna().tolist() for c in cat_order]
    bp = ax.boxplot(data_per_cat, tick_labels=cat_order, patch_artist=True)
    for patch, c in zip(bp["boxes"], cat_order):
        patch.set_facecolor(cat_colors[c])
        patch.set_alpha(0.6)
    ax.set_ylabel("ResNet50 feature distance: raw ↔ teams (per pair)")
    ax.set_title("Per-pair raw↔teams feature distance by P8A category")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "embedding_pair_distance.png", dpi=120)
    plt.close(fig)

    print(f"\n[done] outputs in {OUT}")


if __name__ == "__main__":
    main()
