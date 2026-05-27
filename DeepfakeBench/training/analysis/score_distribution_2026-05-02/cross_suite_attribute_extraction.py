"""Download samples and compute crop attributes for:

  - All 425 teams_fake_all_lockbox frames (covers Diag #7)
  - 16 frames each from Roy_D, Q, PC_Generator, dor in teams_real_all_dev (Diag #8)
  - 50-frame samples from each remaining suite for cross-suite landscape (Diag #9)

Then runs:

  - Cam_test_s33 vs pc_generator_s15 attribute distributions (Diag #7)
  - Identity gallery rendering (Diag #8)
  - Cross-suite distribution plots (Diag #9)
  - Score-attribute Pearson r per (suite, model) (Diag #10)

All outputs land under outputs/cross_suite_*.csv and figures/cross_suite_*.png.
"""

from __future__ import annotations

import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import laplace, sobel
from scipy.stats import pearsonr, mannwhitneyu

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
DEST = OUT / "cross_suite_samples"
DEST.mkdir(parents=True, exist_ok=True)
IDENTITY_DEST = OUT / "identity_gallery"
IDENTITY_DEST.mkdir(parents=True, exist_ok=True)
LOCKBOX_DEST = OUT / "lockbox_fake_walkthrough"

DEPLOYED_TAU = {"P8A": 0.990946, "P18T": 0.99359, "P18C": 0.994625}

FAKE_SUITES = ["teams_fake_all_dev", "teams_fake_all_lockbox", "visomaster_enhanced_macro_dev", "deeplive_enhanced_dev"]
REAL_SUITES = ["teams_real_all_dev", "teams_real_all_lockbox", "teams_real_dor_dev",
               "teams_real_lighting_extreme_dev", "teams_real_poor_quality_dev"]
ALL_SUITES = FAKE_SUITES + REAL_SUITES

FPR_DRIVERS = {"Roy_D": 16, "Q": 16, "PC_Generator": 16, "dor": 16, "bla_bla_chow": 16,
               "Test_Cam": 8, "Md_noyn_Sharker": 8}


def fetch(uri: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gsutil", "-q", "cp", uri, str(dest)], check=False)
    return dest


def luminance(arr: np.ndarray) -> np.ndarray:
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)


def per_frame_attrs(p: Path) -> Dict[str, float]:
    arr = np.asarray(Image.open(p).convert("RGB"))
    luma = luminance(arr)
    sx = sobel(luma, axis=0); sy = sobel(luma, axis=1)
    sobel_mag = np.sqrt(sx**2 + sy**2)
    rgb_n = arr.astype(np.float32) / 255.0
    cmax = rgb_n.max(axis=2); cmin = rgb_n.min(axis=2)
    sat = np.where(cmax > 0, (cmax - cmin) / (cmax + 1e-8), 0.0)
    R, G, B = arr[..., 0].astype(np.float32), arr[..., 1].astype(np.float32), arr[..., 2].astype(np.float32)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 128.0
    Cb = (B - Y) * 0.564 + 128.0
    skin = (Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127)
    return {
        "h": int(arr.shape[0]), "w": int(arr.shape[1]),
        "luma_mean": float(luma.mean()), "luma_std": float(luma.std()),
        "laplacian_var": float(laplace(luma).var()),
        "sobel_edge_mean": float(sobel_mag.mean()),
        "saturation_mean": float(sat.mean()),
        "skin_frac": float(skin.mean()),
        "filesize": int(p.stat().st_size),
    }


def main():
    df = pd.read_parquet(OUT / "combined_frames.parquet")

    # ────────── Stage 1: Build URI lists ──────────
    print("[stage1] building URI list...")
    uri_specs: List[Tuple[str, str, Path]] = []  # (suite_or_id, uri, local_dest)

    # 1a: All 425 lockbox fake frames (already partially cached)
    for uri in df[df["suite"] == "teams_fake_all_lockbox"]["frame_path"].unique():
        local = LOCKBOX_DEST / Path(uri).name
        uri_specs.append(("teams_fake_all_lockbox", uri, local))

    # 1b: 16-frame samples per FPR-driver identity (from teams_real_all_dev)
    real_dev = df[(df["suite"] == "teams_real_all_dev") & (df["model"] == "P8A")].copy()
    real_dev["identity"] = real_dev["frame_path"].apply(
        lambda p: p.rsplit("/", 1)[-1].split("__frame_")[0] if "__frame_" in p else "unknown"
    )
    for ident, n in FPR_DRIVERS.items():
        sub = real_dev[real_dev["identity"] == ident]
        if sub.empty:
            print(f"  [warn] no frames for identity '{ident}'")
            continue
        rng = np.random.RandomState(737)
        idx = rng.choice(len(sub), min(n, len(sub)), replace=False)
        for _, row in sub.iloc[idx].iterrows():
            local = IDENTITY_DEST / ident / Path(row["frame_path"]).name
            uri_specs.append((f"identity_{ident}", row["frame_path"], local))

    # 1c: 30-frame samples per remaining suite (real + fake suites, cross-suite landscape)
    for suite in ALL_SUITES:
        sub = df[(df["suite"] == suite) & (df["model"] == "P8A")]
        if sub.empty:
            continue
        rng = np.random.RandomState(737)
        n_sample = min(50, len(sub))
        idx = rng.choice(len(sub), n_sample, replace=False)
        for _, row in sub.iloc[idx].iterrows():
            local = DEST / suite / Path(row["frame_path"]).name
            uri_specs.append((f"crosssuite_{suite}", row["frame_path"], local))

    print(f"[stage1] total URI tasks: {len(uri_specs)}")

    # ────────── Stage 2: Parallel download ──────────
    print("[stage2] downloading...")
    todo = [(uri, dest) for _, uri, dest in uri_specs if not (dest.exists() and dest.stat().st_size > 0)]
    print(f"[stage2] need to download {len(todo)} (cached: {len(uri_specs) - len(todo)})")
    with ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(lambda args: fetch(args[0], args[1]), todo))

    # ────────── Stage 3: Compute attributes ──────────
    print("\n[stage3] computing per-frame attributes...")
    rows = []
    for tag, uri, dest in uri_specs:
        if not (dest.exists() and dest.stat().st_size > 0):
            continue
        try:
            attrs = per_frame_attrs(dest)
        except Exception as e:
            continue
        # Get score for this frame across all 3 models (if it's in the contract scorecard)
        scored = df[df["frame_path"] == uri]
        for model in ["P8A", "P18T", "P18C"]:
            ms = scored[scored["model"] == model]
            if not ms.empty:
                attrs[f"score_{model}"] = float(ms["frame_prob"].iloc[0])
                attrs[f"label_{model}"] = int(ms["label"].iloc[0])
                attrs["suite"] = ms["suite"].iloc[0]
                attrs["method"] = ms["method"].iloc[0]
                attrs["video_id"] = ms["video_id"].iloc[0]
        attrs["tag"] = tag
        attrs["frame_path"] = uri
        rows.append(attrs)
    attr_df = pd.DataFrame(rows)
    attr_df.to_csv(OUT / "cross_suite_attributes.csv", index=False)
    print(f"[stage3] {len(attr_df)} frames with attributes")

    # ────────── Diag #7: Lockbox cam_test_s33 vs pc_generator_s15 attribute distributions ──────────
    print("\n=== Diag #7: lockbox attribute comparison ===")
    lock = attr_df[attr_df["suite"] == "teams_fake_all_lockbox"].copy()
    if not lock.empty:
        print(f"lockbox frames with attrs: {len(lock)}")
        print(f"by method:\n{lock['method'].value_counts()}")

        cam = lock[lock["method"] == "teams_capture_cam_test_s33"]
        pc = lock[lock["method"] == "teams_capture_pc_generator_s15"]
        # For cam_test_s33: compare caught vs missed (P8A) within
        cam_caught = cam[cam["score_P8A"] >= 0.5]
        cam_missed = cam[cam["score_P8A"] < 0.5]
        print(f"\ncam_test_s33: caught={len(cam_caught)} missed={len(cam_missed)}")

        # Mann-Whitney
        results = []
        for col in ["luma_mean", "luma_std", "laplacian_var", "sobel_edge_mean", "saturation_mean", "skin_frac", "h"]:
            if cam_caught.empty or cam_missed.empty:
                continue
            try:
                u, p = mannwhitneyu(cam_caught[col], cam_missed[col], alternative="two-sided")
            except Exception:
                u, p = float("nan"), float("nan")
            results.append({
                "feature": col,
                "cam_caught_mean": float(cam_caught[col].mean()),
                "cam_missed_mean": float(cam_missed[col].mean()),
                "delta": float(cam_caught[col].mean() - cam_missed[col].mean()),
                "MWU_p": float(p),
                "significant": bool(p < 0.05),
            })
        # cam vs pc method comparison
        for col in ["luma_mean", "luma_std", "laplacian_var", "sobel_edge_mean", "saturation_mean", "skin_frac", "h"]:
            if cam.empty or pc.empty:
                continue
            try:
                u, p = mannwhitneyu(cam[col], pc[col], alternative="two-sided")
            except Exception:
                u, p = float("nan"), float("nan")
            results.append({
                "feature": col,
                "cam_caught_mean": float(cam[col].mean()),
                "cam_missed_mean": float(pc[col].mean()),
                "delta": float(cam[col].mean() - pc[col].mean()),
                "MWU_p": float(p),
                "significant": bool(p < 0.05),
                "comparison": "cam_test_s33 vs pc_generator_s15"
            })
        sig_df = pd.DataFrame(results)
        sig_df.to_csv(OUT / "lockbox_attribute_significance.csv", index=False)
        print("\nResults:")
        print(sig_df.to_string(index=False, float_format="%.3f"))

        # Plot lockbox cam vs pc distributions
        plot_features = ["luma_mean", "laplacian_var", "sobel_edge_mean", "saturation_mean"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax, col in zip(axes.flat, plot_features):
            if cam.empty or pc.empty:
                ax.axis("off"); continue
            all_vals = pd.concat([cam[col], pc[col]])
            bins = np.linspace(all_vals.quantile(0.02), all_vals.quantile(0.98), 25)
            ax.hist(pc[col], bins=bins, alpha=0.55, color="green", label=f"pc_generator_s15 (n={len(pc)})", edgecolor="black")
            ax.hist(cam[col], bins=bins, alpha=0.55, color="red", label=f"cam_test_s33 (n={len(cam)})", edgecolor="black")
            try:
                _, p = mannwhitneyu(cam[col], pc[col], alternative="two-sided")
                sig = " *" if p < 0.05 else ""
                ax.set_title(f"{col}\np={p:.4f}{sig}")
            except Exception:
                ax.set_title(col)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.3)
        fig.suptitle("Lockbox fake — cam_test_s33 (mostly missed) vs pc_generator_s15 (mostly caught)")
        fig.tight_layout()
        fig.savefig(FIG / "lockbox_cam_vs_pc_attrs.png", dpi=110)
        plt.close(fig)

    # ────────── Diag #8: Identity gallery ──────────
    print("\n=== Diag #8: identity gallery ===")
    identity_groups = [t for t in attr_df["tag"].unique() if t.startswith("identity_")]
    if identity_groups:
        rows_g = len(identity_groups)
        cols_g = 8
        fig, axes = plt.subplots(rows_g, cols_g, figsize=(cols_g * 2.4, rows_g * 2.7))
        if rows_g == 1:
            axes = np.array([axes])
        for r, tag in enumerate(sorted(identity_groups)):
            ident = tag.replace("identity_", "")
            sub = attr_df[attr_df["tag"] == tag].head(cols_g)
            for c in range(cols_g):
                ax = axes[r, c]
                if c < len(sub):
                    row = sub.iloc[c]
                    p = IDENTITY_DEST / ident / Path(row["frame_path"]).name
                    if p.exists():
                        try:
                            arr = np.asarray(Image.open(p).convert("RGB"))
                            ax.imshow(arr)
                        except Exception:
                            pass
                    title = f"P8A:{row.get('score_P8A', float('nan')):.2f}\nP18C:{row.get('score_P18C', float('nan')):.2f}"
                    ax.set_title(title, fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(ident, fontsize=10, rotation=0, ha="right", va="center", labelpad=80)
        fig.suptitle("FPR-driver identity gallery — teams_real_all_dev (8 random frames per identity, with model scores)")
        fig.tight_layout()
        fig.savefig(FIG / "identity_gallery.png", dpi=110)
        plt.close(fig)

    # Per-identity attribute summary
    if identity_groups:
        ident_summary = []
        for tag in identity_groups:
            sub = attr_df[attr_df["tag"] == tag]
            ident = tag.replace("identity_", "")
            ident_summary.append({
                "identity": ident,
                "n_sampled": len(sub),
                "mean_h": float(sub["h"].mean()),
                "mean_w": float(sub["w"].mean()),
                "mean_luma": float(sub["luma_mean"].mean()),
                "mean_laplacian_var": float(sub["laplacian_var"].mean()),
                "mean_sobel_edge": float(sub["sobel_edge_mean"].mean()),
                "mean_score_P8A": float(sub["score_P8A"].mean()),
                "mean_score_P18T": float(sub["score_P18T"].mean()),
                "mean_score_P18C": float(sub["score_P18C"].mean()),
            })
        pd.DataFrame(ident_summary).to_csv(OUT / "identity_attribute_summary.csv", index=False)
        print(pd.DataFrame(ident_summary).to_string(index=False, float_format="%.3f"))

    # ────────── Diag #9: Cross-suite landscape ──────────
    print("\n=== Diag #9: cross-suite attribute landscape ===")
    plot_features = ["luma_mean", "laplacian_var", "sobel_edge_mean", "h"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    suite_palette = {
        "teams_fake_all_dev": "#1f77b4", "teams_fake_all_lockbox": "#9467bd",
        "visomaster_enhanced_macro_dev": "#d62728", "deeplive_enhanced_dev": "#2ca02c",
        "teams_real_all_dev": "#17becf", "teams_real_all_lockbox": "#bcbd22",
        "teams_real_dor_dev": "#e377c2", "teams_real_lighting_extreme_dev": "#7f7f7f",
        "teams_real_poor_quality_dev": "#ff7f0e",
    }
    for ax, col in zip(axes.flat, plot_features):
        suite_data = []
        labels = []
        colors = []
        for suite in ALL_SUITES:
            sub = attr_df[attr_df["suite"] == suite]
            if sub.empty:
                continue
            suite_data.append(sub[col].dropna().tolist())
            labels.append(f"{suite}\n(n={len(sub)})")
            colors.append(suite_palette.get(suite, "#888"))
        if not suite_data:
            ax.axis("off"); continue
        bp = ax.boxplot(suite_data, tick_labels=labels, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Cross-suite attribute distributions (sampled frames)")
    fig.tight_layout()
    fig.savefig(FIG / "cross_suite_attr_box.png", dpi=110)
    plt.close(fig)

    # Per-suite attribute summary
    suite_summary = []
    for suite in ALL_SUITES:
        sub = attr_df[attr_df["suite"] == suite]
        if sub.empty:
            continue
        suite_summary.append({
            "suite": suite,
            "n_sampled": len(sub),
            "mean_h": float(sub["h"].mean()),
            "mean_luma": float(sub["luma_mean"].mean()),
            "mean_laplacian_var": float(sub["laplacian_var"].mean()),
            "median_laplacian_var": float(sub["laplacian_var"].median()),
            "mean_sobel": float(sub["sobel_edge_mean"].mean()),
            "P8A_mean_score": float(sub["score_P8A"].mean()),
            "P8A_median_score": float(sub["score_P8A"].median()),
            "label": int(sub["label_P8A"].iloc[0]) if not sub.empty else -1,
        })
    pd.DataFrame(suite_summary).to_csv(OUT / "cross_suite_attribute_summary.csv", index=False)
    print(pd.DataFrame(suite_summary).to_string(index=False, float_format="%.3f"))

    # ────────── Diag #10: Score-attribute correlations per suite, per model ──────────
    print("\n=== Diag #10: score-attribute correlations ===")
    corr_rows = []
    attr_cols = ["luma_mean", "luma_std", "laplacian_var", "sobel_edge_mean", "saturation_mean", "skin_frac", "h"]
    for suite in ALL_SUITES:
        for model in ["P8A", "P18T", "P18C"]:
            sub = attr_df[(attr_df["suite"] == suite) & attr_df[f"score_{model}"].notna()]
            if len(sub) < 10:
                continue
            for col in attr_cols:
                vals = sub[col].dropna()
                scores = sub[f"score_{model}"].dropna()
                idx = vals.index.intersection(scores.index)
                if len(idx) < 10:
                    continue
                try:
                    r, p = pearsonr(vals.loc[idx], scores.loc[idx])
                except Exception:
                    r, p = float("nan"), float("nan")
                corr_rows.append({
                    "suite": suite, "model": model, "feature": col,
                    "n": int(len(idx)),
                    "pearson_r": float(r),
                    "p": float(p),
                    "significant_5pct": bool(p < 0.05),
                })
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT / "score_attribute_correlations.csv", index=False)
    print("\nTop 20 strongest correlations (by |r|):")
    print(corr_df.reindex(corr_df["pearson_r"].abs().sort_values(ascending=False).index).head(20).to_string(index=False, float_format="%.4f"))

    # Plot: heatmap of |r| per (suite × feature) for P8A
    p8a_corr = corr_df[corr_df["model"] == "P8A"].pivot_table(index="suite", columns="feature", values="pearson_r")
    fig, ax = plt.subplots(figsize=(12, 6))
    import matplotlib.colors as mcolors
    im = ax.imshow(p8a_corr.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(p8a_corr.columns)))
    ax.set_xticklabels(p8a_corr.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(p8a_corr.index)))
    ax.set_yticklabels(p8a_corr.index, fontsize=9)
    for i in range(p8a_corr.shape[0]):
        for j in range(p8a_corr.shape[1]):
            v = p8a_corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="black" if abs(v) < 0.4 else "white")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("P8A: score vs crop-attribute correlation per suite")
    fig.tight_layout()
    fig.savefig(FIG / "score_attribute_corr_heatmap_P8A.png", dpi=120)
    plt.close(fig)

    print(f"\n[done] outputs in {OUT}")


if __name__ == "__main__":
    main()
