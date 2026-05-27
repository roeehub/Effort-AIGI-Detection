"""Train vs eval visomaster visual comparison.

Compares:
  - TRAINING enhanced viso: gs://visomaster-enhanced-face-cropped/samples/visomaster_<METHOD>_<NNNNN>_enhanced_<ENHANCER>/frames/fake/
  - TRAINING base viso (no enhancement): gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_<METHOD>_<NNNNN>/frames/fake/
  - EVAL enhanced viso: gs://teams-faces-data-test-2914-fake-4420-real-feb-28/fake/visomaster_enhanced_(raw|teams)__*

These are different buckets with different naming conventions. The question:
do the training and eval frames look obviously different?

Outputs:
  outputs/train_vs_eval/<group>/*.png  (downloaded samples)
  outputs/train_vs_eval/comparison_grid.png
  outputs/train_vs_eval/manifest.csv
  outputs/figures/train_vs_eval_pixel_stats.png
"""

from __future__ import annotations

import json
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

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
DEST = OUT / "train_vs_eval"
DEST.mkdir(parents=True, exist_ok=True)

GROUPS = {
    "train_base_CSCS": [
        # Training base (no enhancement) viso
        "gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_CSCS_00000/frames/fake/frame_0001.png",
        "gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_CSCS_00001/frames/fake/frame_0001.png",
        "gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_CSCS_00002/frames/fake/frame_0001.png",
        "gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_CSCS_00003/frames/fake/frame_0001.png",
    ],
    "train_base_GhostFace_v3": [
        "gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_GhostFace-v3_06000/frames/fake/frame_0001.png",
        "gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_GhostFace-v3_06001/frames/fake/frame_0001.png",
        "gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_GhostFace-v3_06002/frames/fake/frame_0001.png",
        "gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_GhostFace-v3_06003/frames/fake/frame_0001.png",
    ],
    "train_base_SimSwap512": [
        "gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_SimSwap512_16000/frames/fake/frame_0001.png",
        "gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_SimSwap512_16001/frames/fake/frame_0001.png",
        "gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_SimSwap512_16002/frames/fake/frame_0001.png",
        "gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_SimSwap512_16003/frames/fake/frame_0001.png",
    ],
    "train_enhanced_codeformer": [
        "gs://visomaster-enhanced-face-cropped/samples/visomaster_CSCS_00007_enhanced_codeformer/frames/fake/frame_0001.png",
        "gs://visomaster-enhanced-face-cropped/samples/visomaster_CSCS_00021_enhanced_codeformer/frames/fake/frame_0001.png",
    ],
    "train_enhanced_gfpgan": [
        "gs://visomaster-enhanced-face-cropped/samples/visomaster_CSCS_00007_enhanced_gfpgan/frames/fake/frame_0001.png",
        "gs://visomaster-enhanced-face-cropped/samples/visomaster_CSCS_00019_enhanced_gfpgan/frames/fake/frame_0001.png",
    ],
    "train_enhanced_gpen-2048": [
        "gs://visomaster-enhanced-face-cropped/samples/visomaster_CSCS_00007_enhanced_gpen-2048/frames/fake/frame_0001.png",
    ],
    # Will fill in eval samples dynamically below
}


def fetch_one(uri: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gsutil", "-q", "cp", uri, str(dest)], check=False)
    return dest


def fetch_all(groups: Dict[str, List[str]]) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        for group, uris in groups.items():
            paths = []
            for uri in uris:
                local = DEST / group / Path(uri).name
                paths.append(local)
                ex.submit(fetch_one, uri, local)
            out[group] = paths
        ex.shutdown(wait=True)
    return out


def main():
    # Fill in eval samples dynamically — pick 4 raw + 4 teams from already-cached pairs
    cached = sorted((OUT / "viso_full_paired").glob("visomaster_enhanced_raw__*.png"))[:4]
    cached_teams = sorted((OUT / "viso_full_paired").glob("visomaster_enhanced_teams__*.png"))[:4]
    GROUPS["eval_enhanced_raw"] = [f"file://{p}" for p in cached]
    GROUPS["eval_enhanced_teams"] = [f"file://{p}" for p in cached_teams]

    # Local symlinks for eval (no fetch needed)
    for group_name in ("eval_enhanced_raw", "eval_enhanced_teams"):
        (DEST / group_name).mkdir(parents=True, exist_ok=True)
        for uri in GROUPS[group_name]:
            src = Path(uri.replace("file://", ""))
            link = DEST / group_name / src.name
            if not link.exists():
                try:
                    link.symlink_to(src)
                except FileExistsError:
                    pass

    print("[fetch] downloading training samples...")
    download_groups = {k: v for k, v in GROUPS.items() if not k.startswith("eval_")}
    fetched = fetch_all(download_groups)

    # Wait for downloads to finalise
    import time
    time.sleep(3)

    # Build manifest
    manifest_rows = []
    for group, uris in GROUPS.items():
        for uri in uris:
            if uri.startswith("file://"):
                local = Path(uri.replace("file://", ""))
            else:
                local = DEST / group / Path(uri).name
            row = {"group": group, "uri": uri, "local_path": str(local)}
            if local.exists():
                arr = np.asarray(Image.open(local).convert("RGB"))
                row.update({
                    "shape_h": int(arr.shape[0]),
                    "shape_w": int(arr.shape[1]),
                    "luma_mean": float((0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).mean()),
                    "size_bytes": local.stat().st_size,
                })
                row["downloaded"] = True
            else:
                row["downloaded"] = False
            manifest_rows.append(row)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(DEST / "manifest.csv", index=False)

    print("\n=== Manifest summary ===")
    summary = manifest.groupby("group").agg(
        n=("local_path", "size"),
        n_downloaded=("downloaded", "sum"),
        mean_luma=("luma_mean", "mean"),
        mean_h=("shape_h", "mean"),
        mean_w=("shape_w", "mean"),
        mean_size=("size_bytes", "mean"),
    ).round(2)
    print(summary)
    summary.to_csv(DEST / "group_summary.csv")

    # ---- Render comparison grid: rows = groups, cols = up to 4 samples
    groups_to_show = list(GROUPS.keys())
    n_cols = max(len(v) for v in GROUPS.values())
    fig, axes = plt.subplots(len(groups_to_show), n_cols, figsize=(n_cols * 3.5, len(groups_to_show) * 3.5))
    for r, group in enumerate(groups_to_show):
        uris = GROUPS[group]
        for c in range(n_cols):
            ax = axes[r, c] if len(groups_to_show) > 1 else axes[c]
            if c < len(uris):
                if uris[c].startswith("file://"):
                    p = Path(uris[c].replace("file://", ""))
                else:
                    p = DEST / group / Path(uris[c]).name
                if p.exists():
                    try:
                        img = Image.open(p)
                        arr = np.asarray(img.convert("RGB"))
                        ax.imshow(arr)
                        ax.set_title(f"luma={arr.mean():.0f} {arr.shape[0]}×{arr.shape[1]}", fontsize=7)
                    except Exception as exc:
                        ax.text(0.5, 0.5, f"err: {exc}", ha="center", transform=ax.transAxes, fontsize=8)
                else:
                    ax.text(0.5, 0.5, "missing", ha="center", transform=ax.transAxes, fontsize=10)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(group, fontsize=10, rotation=0, ha="right", va="center", labelpad=80)
    fig.suptitle("Train vs Eval visomaster — visual comparison", fontsize=14, y=0.99)
    fig.tight_layout()
    fig.savefig(FIG / "train_vs_eval_grid.png", dpi=110)
    plt.close(fig)

    # ---- Pixel-level stats per group
    valid = manifest[manifest["downloaded"] == True].copy()
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    groups_present = valid["group"].unique().tolist()
    pos = np.arange(len(groups_present))
    means = [valid[valid["group"] == g]["luma_mean"].mean() for g in groups_present]
    stds = [valid[valid["group"] == g]["luma_mean"].std() for g in groups_present]
    ax.bar(pos, means, yerr=stds, capsize=4, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_xticks(pos)
    ax.set_xticklabels(groups_present, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("mean luminance (per frame)")
    ax.set_title("Mean luminance per group — train vs eval visomaster")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "train_vs_eval_luma.png", dpi=110)
    plt.close(fig)

    print(f"\n[done] outputs in {DEST}")
    print(f"[grid] {FIG / 'train_vs_eval_grid.png'}")
    print(f"[luma] {FIG / 'train_vs_eval_luma.png'}")


if __name__ == "__main__":
    main()
