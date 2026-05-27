#!/usr/bin/env python3
"""Face-size-leak scatter on P8A scorecard reports.

Per eval-cohort ("bucket"): scatter face_pixel_area vs avg_video_prob,
colored real/fake. Pearson correlation magnitude per real/fake split.

Inputs:
- /tmp/p8a_reports_2026-04-29/*p8a_step5000_videos_report.csv (from
  gs://training-job-outputs/.../p8a-review-scorecard-20260425/reports/)
- /tmp/p8a_reports_2026-04-29/*p8a_step5000_frames_report.csv
- analysis/lockbox_tagging/full_tags_2026-04-27.parquet (face_pixel_area
  per gcs_uri)

Output:
- analysis/face_size_leak_scatter_2026-04-29/<bucket>.png (per cohort)
- analysis/face_size_leak_scatter_2026-04-29/grid.png (multi-bucket)
- analysis/face_size_leak_scatter_2026-04-29/per_bucket_verdicts.csv
- analysis/face_size_leak_scatter_2026-04-29/summary.json

Aggregation: per-video face_pixel_area = MEAN of frame face_pixel_area.
Pearson computed via numpy/scipy; sklearn n_jobs avoided.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
REPORTS_DIR = Path("/tmp/p8a_reports_2026-04-29")
OUT_DIR = ROOT / "analysis/face_size_leak_scatter_2026-04-29"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARQUET = ROOT / "analysis/lockbox_tagging/full_tags_2026-04-27.parquet"


def verdict(r: float | None) -> str:
    if r is None or (isinstance(r, float) and (math.isnan(r) or math.isinf(r))):
        return "n/a"
    a = abs(r)
    if a > 0.5:
        return "strong"
    if a >= 0.3:
        return "moderate"
    if a >= 0.1:
        return "weak"
    return "none"


def safe_pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float | None, float | None]:
    if len(x) < 3:
        return None, None
    if np.std(x) == 0 or np.std(y) == 0:
        return None, None
    r, p = pearsonr(x, y)
    return float(r), float(p)


def list_buckets() -> List[str]:
    """Find videos_report files under REPORTS_DIR; bucket = stem before
    `_p8a_step5000_videos_report.csv`."""
    out: List[str] = []
    for p in sorted(REPORTS_DIR.glob("*p8a_step5000_videos_report.csv")):
        name = p.name
        suffix = "_p8a_step5000_videos_report.csv"
        bucket = name[: -len(suffix)]
        out.append(bucket)
    return out


def load_videos_report(bucket: str) -> pd.DataFrame:
    p = REPORTS_DIR / f"{bucket}_p8a_step5000_videos_report.csv"
    return pd.read_csv(p)


def load_frames_report(bucket: str) -> pd.DataFrame:
    p = REPORTS_DIR / f"{bucket}_p8a_step5000_frames_report.csv"
    return pd.read_csv(p)


def build_video_face_area(frames: pd.DataFrame, parq_idx: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Join frames -> parquet face_pixel_area, then aggregate to per-video MEAN.

    Returns (per_video_df, diag) where per_video_df has columns
    [video_id, face_pixel_area_mean, n_frames_matched].
    """
    merged = frames.merge(
        parq_idx[["gcs_uri", "face_pixel_area"]],
        left_on="frame_path",
        right_on="gcs_uri",
        how="left",
    )
    n_total = len(merged)
    n_matched = int(merged["face_pixel_area"].notna().sum())
    diag = {
        "n_frames_total": n_total,
        "n_frames_matched": n_matched,
        "match_pct": (n_matched / n_total * 100.0) if n_total else 0.0,
    }
    matched = merged[merged["face_pixel_area"].notna()].copy()
    grp = (
        matched.groupby("video_id")["face_pixel_area"]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": "face_pixel_area_mean", "size": "n_frames_matched"})
    )
    return grp, diag


def make_scatter(ax, df_join: pd.DataFrame, title: str) -> Dict:
    """Scatter and return per-class Pearson stats."""
    df_real = df_join[df_join["label"] == 0]
    df_fake = df_join[df_join["label"] == 1]
    # Plot
    if len(df_real):
        ax.scatter(
            df_real["face_pixel_area_mean"],
            df_real["avg_video_prob"],
            c="tab:blue",
            alpha=0.5,
            s=14,
            edgecolors="none",
            label=f"real (n={len(df_real)})",
        )
    if len(df_fake):
        ax.scatter(
            df_fake["face_pixel_area_mean"],
            df_fake["avg_video_prob"],
            c="tab:red",
            alpha=0.5,
            s=14,
            edgecolors="none",
            label=f"fake (n={len(df_fake)})",
        )
    r_real, p_real = safe_pearson(
        df_real["face_pixel_area_mean"].to_numpy(), df_real["avg_video_prob"].to_numpy()
    )
    r_fake, p_fake = safe_pearson(
        df_fake["face_pixel_area_mean"].to_numpy(), df_fake["avg_video_prob"].to_numpy()
    )
    v_real = verdict(r_real)
    v_fake = verdict(r_fake)
    txt_real = f"{r_real:.3f}" if r_real is not None else "n/a"
    txt_fake = f"{r_fake:.3f}" if r_fake is not None else "n/a"
    sub = f"r_real={txt_real} ({v_real})  r_fake={txt_fake} ({v_fake})"

    ax.set_xlabel("face_pixel_area (per-video mean)")
    ax.set_ylabel("avg_video_prob (P8A)")
    ax.set_title(f"{title}\n{sub}", fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    return {
        "n_real": int(len(df_real)),
        "n_fake": int(len(df_fake)),
        "r_real": r_real,
        "p_real": p_real,
        "verdict_real": v_real,
        "r_fake": r_fake,
        "p_fake": p_fake,
        "verdict_fake": v_fake,
    }


def main() -> None:
    print(f"[scatter] reports dir: {REPORTS_DIR}")
    print(f"[scatter] parquet:    {PARQUET}")
    parq = pd.read_parquet(PARQUET)[["gcs_uri", "face_pixel_area"]]
    print(f"[scatter] parquet rows: {len(parq)}")

    buckets = list_buckets()
    print(f"[scatter] found {len(buckets)} buckets")

    results: Dict[str, Dict] = {}
    skipped: Dict[str, str] = {}

    # First pass: process each bucket and stash per-video joined frames + stats
    per_bucket_join: Dict[str, pd.DataFrame] = {}

    for bucket in buckets:
        try:
            videos = load_videos_report(bucket)
            frames = load_frames_report(bucket)
        except FileNotFoundError as e:
            skipped[bucket] = f"missing report: {e}"
            continue
        if "avg_video_prob" not in videos.columns or "label" not in videos.columns:
            skipped[bucket] = "videos_report missing avg_video_prob/label"
            continue

        face_video, diag = build_video_face_area(frames, parq)
        if diag["n_frames_matched"] == 0:
            skipped[bucket] = (
                f"no parquet match for any frame "
                f"(n_frames_total={diag['n_frames_total']})"
            )
            continue

        join = videos.merge(face_video, on="video_id", how="inner")
        join = join.dropna(subset=["face_pixel_area_mean", "avg_video_prob"])
        if len(join) < 5:
            skipped[bucket] = f"after-join sample too small (n={len(join)})"
            continue

        per_bucket_join[bucket] = join
        results[bucket] = {
            "n_videos_report": int(len(videos)),
            "n_frames_total": diag["n_frames_total"],
            "n_frames_matched": diag["n_frames_matched"],
            "frame_match_pct": diag["match_pct"],
            "n_videos_after_join": int(len(join)),
        }
        print(
            f"[scatter] bucket={bucket}  videos={len(videos)}  "
            f"frames_matched={diag['n_frames_matched']}/{diag['n_frames_total']} "
            f"({diag['match_pct']:.1f}%)  joined_videos={len(join)}"
        )

    # Per-bucket plots
    saved_pngs: List[str] = []
    rows = []
    for bucket, join in per_bucket_join.items():
        fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
        stats = make_scatter(ax, join, bucket)
        out_png = OUT_DIR / f"{bucket}.png"
        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)
        saved_pngs.append(str(out_png))
        results[bucket].update(stats)
        rows.append(
            {
                "bucket": bucket,
                "n_real": stats["n_real"],
                "n_fake": stats["n_fake"],
                "r_real": stats["r_real"],
                "verdict_real": stats["verdict_real"],
                "r_fake": stats["r_fake"],
                "verdict_fake": stats["verdict_fake"],
                "n_frames_matched": results[bucket]["n_frames_matched"],
                "n_frames_total": results[bucket]["n_frames_total"],
            }
        )

    # Pooled mixed-class plots (real + fake combined) per regime,
    # so we can see the cross-class face-size leak signal that
    # individual single-class buckets cannot show on their own.
    pooled_groups = {
        "pooled_dev_fake_real": [
            "teams_fake_all_dev",
            "teams_real_all_dev",
            "deeplive_enhanced_dev",
        ],
        "pooled_lockbox_fake_real": [
            "teams_fake_all_lockbox",
            "teams_real_all_lockbox",
        ],
    }
    pooled_results: Dict[str, Dict] = {}
    for group_name, bucket_list in pooled_groups.items():
        parts = []
        for b in bucket_list:
            if b in per_bucket_join:
                parts.append(per_bucket_join[b].assign(_src=b))
        if not parts:
            continue
        pooled = pd.concat(parts, ignore_index=True)
        if len(pooled) < 5:
            continue
        fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
        stats = make_scatter(ax, pooled, group_name)
        # Cross-class Pearson on the pooled set (real+fake mixed)
        x = pooled["face_pixel_area_mean"].to_numpy()
        y = pooled["avg_video_prob"].to_numpy()
        r_all, p_all = safe_pearson(x, y)
        v_all = verdict(r_all)
        title_old = ax.get_title()
        all_txt = f"{r_all:.3f}" if r_all is not None else "n/a"
        ax.set_title(
            f"{title_old}\nPOOLED real+fake: r_all={all_txt} ({v_all})",
            fontsize=10,
        )
        out_png = OUT_DIR / f"{group_name}.png"
        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)
        saved_pngs.append(str(out_png))
        pooled_results[group_name] = {
            "buckets": bucket_list,
            "n_total": int(len(pooled)),
            "n_real": stats["n_real"],
            "n_fake": stats["n_fake"],
            "r_real": stats["r_real"],
            "verdict_real": stats["verdict_real"],
            "r_fake": stats["r_fake"],
            "verdict_fake": stats["verdict_fake"],
            "r_pooled_all": r_all,
            "verdict_pooled_all": v_all,
        }

    # Multi-bucket grid
    if per_bucket_join:
        n = len(per_bucket_join)
        cols = 3
        rows_n = math.ceil(n / cols)
        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 5.5, rows_n * 4.2), dpi=120)
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]
        for i, (bucket, join) in enumerate(per_bucket_join.items()):
            make_scatter(axes_flat[i], join, bucket)
        for j in range(len(per_bucket_join), len(axes_flat)):
            axes_flat[j].axis("off")
        fig.suptitle(
            "P8A face-size leak scatter — face_pixel_area (per-video mean) vs avg_video_prob",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        grid_png = OUT_DIR / "grid.png"
        fig.savefig(grid_png)
        plt.close(fig)
        saved_pngs.append(str(grid_png))

    # Verdict CSV
    df_v = pd.DataFrame(rows)
    df_v.to_csv(OUT_DIR / "per_bucket_verdicts.csv", index=False)

    summary = {
        "buckets_attempted": list(buckets),
        "buckets_processed": list(per_bucket_join.keys()),
        "buckets_skipped": skipped,
        "results": results,
        "pooled_results": pooled_results,
        "saved_pngs": saved_pngs,
        "verdict_thresholds": {
            "strong": "|r| > 0.5",
            "moderate": "0.3 <= |r| <= 0.5",
            "weak": "0.1 <= |r| < 0.3",
            "none": "|r| < 0.1",
        },
        "aggregation": "per-video face_pixel_area = MEAN of frame face_pixel_area",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print()
    print("=== Per-bucket verdicts ===")
    print(df_v.to_string(index=False))
    print()
    print(f"[scatter] saved {len(saved_pngs)} PNG(s) to {OUT_DIR}")
    print(f"[scatter] summary: {OUT_DIR/'summary.json'}")
    print(f"[scatter] verdicts: {OUT_DIR/'per_bucket_verdicts.csv'}")


if __name__ == "__main__":
    main()
