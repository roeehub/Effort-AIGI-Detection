"""Phase 2 — within-video / within-stream score-spread.

For our score data the per-`video_id` rows are 1-4 frames each (these are
*video segments* not full videos). To get true within-stream variance we
GROUP BY SESSION (e.g., 'Cam_Test__s32') and order by (seg_id, frame_idx).

Per stream: n, min, max, median, IQR, std, frac_above_τ for τ in
{0.3,0.5,0.7,0.95,0.98}, has_above_0.95, has_above_0.98.

Aggregated to suite-level percentiles. Override-rule pre-check is the headline
output: `P(>=1 frame above 0.98 | real_video)` vs `P(... | fake_video)`.

Outputs:
  data/<ckpt>/within_video_stats.csv
  data/<ckpt>/override_rule_precheck.csv
  findings/<ckpt>/phase2_within_video.md
  figures/<ckpt>/score_traces_sample.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from _common import ensure_dirs, get_ckpt_cfg, load_suite_csv  # noqa: E402
from aggregator_sim import (  # noqa: E402
    extract_frame_index,
    extract_seg_id,
    extract_session,
)


THRESHOLDS = [0.3, 0.5, 0.7, 0.95, 0.98]


def stream_stats(scores: np.ndarray) -> dict:
    if len(scores) == 0:
        return {}
    s = scores.astype(np.float64)
    out = {
        "n_frames": int(len(s)),
        "min": float(s.min()),
        "max": float(s.max()),
        "median": float(np.median(s)),
        "iqr": float(np.percentile(s, 75) - np.percentile(s, 25)),
        "std": float(s.std()),
    }
    for t in THRESHOLDS:
        out[f"frac_above_{t}"] = float((s >= t).mean())
    out["has_above_0.95"] = bool((s >= 0.95).any())
    out["has_above_0.98"] = bool((s >= 0.98).any())
    return out


def build_streams_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per stream (= session + label). For deeplive-style suites
    we fall back to chunks-of-32 as in aggregator_sim.build_streams_for_suite,
    but here we use the natural session grouping where it exists.
    """
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["session"] = df["video_id"].astype(str).map(extract_session)
    df["seg_id"] = df["frame_path"].astype(str).map(extract_seg_id)
    df["frame_idx"] = df["frame_path"].astype(str).map(extract_frame_index)
    rows = []
    for sess, sub in df.groupby("session", sort=False):
        ordered = sub.sort_values(["seg_id", "frame_idx"])
        s = ordered["frame_prob"].to_numpy(dtype=np.float64)
        st = stream_stats(s)
        st["stream_id"] = sess
        st["suite"] = df["suite"].iloc[0] if "suite" in df.columns else None
        st["label_class"] = df["label_class"].iloc[0] if "label_class" in df.columns else None
        rows.append(st)
    return pd.DataFrame(rows)


def aggregate_to_suite(streams_df: pd.DataFrame) -> pd.DataFrame:
    """Suite-level summary of stream-level stats (percentiles)."""
    if streams_df.empty:
        return pd.DataFrame()
    out_rows = []
    for (suite, lab), sub in streams_df.groupby(["suite", "label_class"], sort=False):
        n_streams = len(sub)
        rec = {
            "suite": suite,
            "label_class": lab,
            "n_streams": int(n_streams),
            "frames_per_stream_p50": float(sub["n_frames"].median()),
            "frames_per_stream_p95": float(sub["n_frames"].quantile(0.95)),
            "median_p50": float(sub["median"].median()),
            "iqr_p50": float(sub["iqr"].median()),
            "std_p50": float(sub["std"].median()),
        }
        for t in THRESHOLDS:
            rec[f"frac_above_{t}_p50"] = float(sub[f"frac_above_{t}"].median())
            rec[f"frac_above_{t}_p95"] = float(sub[f"frac_above_{t}"].quantile(0.95))
        rec["pct_streams_with_any_frame_above_0.95"] = float(sub["has_above_0.95"].mean())
        rec["pct_streams_with_any_frame_above_0.98"] = float(sub["has_above_0.98"].mean())
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def plot_score_traces(streams_dict: dict, out_path: Path, ckpt_full: str) -> None:
    """Plot up to 10 example streams (5 real, 5 fake)."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 5))
    real_picks = [s for s in streams_dict.get("real", []) if len(s) >= 8][:5]
    fake_picks = [s for s in streams_dict.get("fake", []) if len(s) >= 8][:5]
    for i in range(5):
        ax = axes[0, i]
        if i < len(real_picks):
            n, label, scores = real_picks[i]
            ax.plot(scores, color="C0")
            ax.axhline(0.5, color="gray", linestyle=":", lw=0.8)
            ax.set_ylim(0, 1)
            ax.set_title(f"REAL: {label}\nn={n}", fontsize=7)
            ax.tick_params(labelsize=6)
        else:
            ax.axis("off")

        ax = axes[1, i]
        if i < len(fake_picks):
            n, label, scores = fake_picks[i]
            ax.plot(scores, color="C3")
            ax.axhline(0.5, color="gray", linestyle=":", lw=0.8)
            ax.set_ylim(0, 1)
            ax.set_title(f"FAKE: {label}\nn={n}", fontsize=7)
            ax.tick_params(labelsize=6)
        else:
            ax.axis("off")
    fig.suptitle(f"Phase 2 — sample stream traces — {ckpt_full}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(ckpt_name: str) -> None:
    cfg = get_ckpt_cfg(ckpt_name)
    dirs = ensure_dirs(ckpt_name)
    full_name = cfg["full_name"]
    t0 = time.time()

    all_streams = []
    sample_traces = {"real": [], "fake": []}
    for suite_name in cfg["suites"].keys():
        try:
            df = load_suite_csv(cfg, suite_name)
        except FileNotFoundError:
            continue
        sdf = build_streams_table(df)
        if sdf.empty:
            continue
        sdf["suite"] = suite_name
        all_streams.append(sdf)

        # Pull a few sample traces for the plot
        df = df.copy()
        df["session"] = df["video_id"].astype(str).map(extract_session)
        df["seg_id"] = df["frame_path"].astype(str).map(extract_seg_id)
        df["frame_idx"] = df["frame_path"].astype(str).map(extract_frame_index)
        for sess, sub in df.groupby("session", sort=False):
            ordered = sub.sort_values(["seg_id", "frame_idx"])
            s = ordered["frame_prob"].to_numpy(dtype=np.float64)
            if len(s) >= 16:
                bucket = "real" if df["label_class"].iloc[0] == "real" else "fake"
                if len(sample_traces[bucket]) < 5:
                    sample_traces[bucket].append((len(s), f"{suite_name}/{sess}", s))

    streams_df = pd.concat(all_streams, ignore_index=True) if all_streams else pd.DataFrame()
    streams_df.to_csv(dirs["data"] / "within_video_stats.csv", index=False)

    suite_summary = aggregate_to_suite(streams_df)
    suite_summary.to_csv(dirs["data"] / "within_video_suite_summary.csv", index=False)

    # Override rule pre-check: per-suite P(any frame >= τ)
    or_rows = []
    for thr in [0.95, 0.98]:
        for (suite, lab), sub in streams_df.groupby(["suite", "label_class"]):
            or_rows.append(
                {
                    "suite": suite,
                    "label_class": lab,
                    "threshold": thr,
                    "n_streams": int(len(sub)),
                    "p_any_frame_above_thr": float((sub[f"frac_above_{thr}"] > 0).mean()),
                }
            )
    or_df = pd.DataFrame(or_rows)
    or_df.to_csv(dirs["data"] / "override_rule_precheck.csv", index=False)

    plot_score_traces(sample_traces, dirs["figures"] / "score_traces_sample.png", full_name)

    # Markdown summary.
    md = [f"# Phase 2 — within-stream score spread — {full_name}", ""]
    md.append(
        "**NOTE on grouping**: per-`video_id` rows in the score reports are short segments "
        "(1-4 frames). True within-stream variance comes from grouping by session "
        "(e.g., 'Cam_Test__s32') and ordering by (seg_id, frame_idx). This phase reports "
        "stream-level stats at the session granularity. For deeplive_enhanced_dev where all "
        "545 frames share a single session, treat each `video_id` as its own length-1 stream.\n"
    )
    md.append("## Suite-level summary\n")
    md.append("| suite | label | n_streams | frames p50 | frames p95 | median p50 | IQR p50 | std p50 | frac>=0.5 p50 | frac>=0.5 p95 | pct_w/_any>=0.95 | pct_w/_any>=0.98 |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in suite_summary.iterrows():
        md.append(
            f"| {r['suite']} | {r['label_class']} | {int(r['n_streams'])} | "
            f"{r['frames_per_stream_p50']:.0f} | {r['frames_per_stream_p95']:.0f} | "
            f"{r['median_p50']:.3f} | {r['iqr_p50']:.3f} | {r['std_p50']:.3f} | "
            f"{r['frac_above_0.5_p50']:.3f} | {r['frac_above_0.5_p95']:.3f} | "
            f"{r['pct_streams_with_any_frame_above_0.95']:.3f} | "
            f"{r['pct_streams_with_any_frame_above_0.98']:.3f} |"
        )

    md.append("\n## Override-rule pre-check\n")
    md.append("`P(any frame above τ | label_class)` — high gap between fake and real means the override is safe.\n")
    md.append("| suite | label | τ | n | P(any≥τ) |")
    md.append("|---|---|---:|---:|---:|")
    for _, r in or_df.iterrows():
        md.append(
            f"| {r['suite']} | {r['label_class']} | {r['threshold']:.2f} | "
            f"{int(r['n_streams'])} | {r['p_any_frame_above_thr']:.3f} |"
        )

    md.append(f"\n_Wall time: {time.time() - t0:.1f}s_")
    (dirs["findings"] / "phase2_within_video.md").write_text("\n".join(md))
    print(f"[phase2] {full_name} done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    main(a.ckpt)
