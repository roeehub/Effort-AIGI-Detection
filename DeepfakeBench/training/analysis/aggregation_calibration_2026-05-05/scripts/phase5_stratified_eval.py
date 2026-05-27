"""Phase 5 — Cross-condition robustness check on the Phase 4 optimal policy.

Re-evaluate the chosen policy stratified by parquet tags:
  clip_capture_mode, face_area_quartile, sharpness_quartile, clip_lighting,
  is_low_quality, is_no_face

Per (suite, stratum) report n_streams, n_flagged, rate. Flag any stratum where
recall drops > 10pp or FPR rises > 5pp vs the global readout.

Outputs:
  data/<ckpt>/stratified_eval.csv
  findings/<ckpt>/phase5_stratified.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from _common import ensure_dirs, get_ckpt_cfg, load_ckpts_yaml, load_suite_csv  # noqa: E402
from aggregator_sim import (  # noqa: E402
    extract_frame_index,
    extract_seg_id,
    extract_session,
    simulate_video,
)


STRAT_COLUMNS = [
    "clip_capture_mode",
    "face_area_quartile",
    "sharpness_quartile",
    "clip_lighting",
    "is_low_quality",
    "is_no_face",
]


def _add_derived_strat_cols(pq: pd.DataFrame) -> pd.DataFrame:
    """Derive face_area_quartile and sharpness_quartile from raw columns."""
    pq = pq.copy()
    if "face_pixel_area" in pq.columns and "face_area_quartile" not in pq.columns:
        try:
            pq["face_area_quartile"] = pd.qcut(
                pq["face_pixel_area"].astype(float), q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
            )
        except Exception:
            pq["face_area_quartile"] = None
    if "sharpness_laplacian" in pq.columns and "sharpness_quartile" not in pq.columns:
        try:
            pq["sharpness_quartile"] = pd.qcut(
                pq["sharpness_laplacian"].astype(float), q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
            )
        except Exception:
            pq["sharpness_quartile"] = None
    return pq


def load_parquet_join(parquet_path: str, df_scores: pd.DataFrame) -> pd.DataFrame:
    """Join score CSV (frame_path) to parquet metadata. Try gcs_uri then basename."""
    pq = pd.read_parquet(parquet_path)
    pq = _add_derived_strat_cols(pq)
    keep = [c for c in (["gcs_uri", "blob_path", "local_path"] + STRAT_COLUMNS) if c in pq.columns]
    pq = pq[keep].copy()
    pq["basename"] = pq["gcs_uri"].astype(str).str.rsplit("/", n=1).str[-1] if "gcs_uri" in pq.columns else None

    df = df_scores.copy()
    df["basename"] = df["frame_path"].astype(str).str.rsplit("/", n=1).str[-1]

    # Strict: try gcs_uri exact join first
    if "gcs_uri" in pq.columns:
        m = df.merge(
            pq.drop(columns=[c for c in ["blob_path", "local_path", "basename"] if c in pq.columns]),
            left_on="frame_path", right_on="gcs_uri", how="left",
        )
        # Pick the first strat col we have
        present = [c for c in STRAT_COLUMNS if c in m.columns]
        if present and m[present[0]].notna().mean() > 0.5:
            return m
    # Fallback: basename join
    sub_cols = ["basename"] + [c for c in STRAT_COLUMNS if c in pq.columns]
    pq_b = pq[sub_cols].drop_duplicates(subset=["basename"])
    m2 = df.merge(pq_b, on="basename", how="left", suffixes=("", "_pq"))
    return m2


def stream_score_arrays_with_strat(
    df_joined: pd.DataFrame,
) -> List[tuple]:
    """Return list of (session, scores_array, strat_dict)."""
    df = df_joined.copy()
    df["session"] = df["video_id"].astype(str).map(extract_session)
    df["seg_id"] = df["frame_path"].astype(str).map(extract_seg_id)
    df["frame_idx"] = df["frame_path"].astype(str).map(extract_frame_index)
    out = []
    for sess, sub in df.groupby("session", sort=False):
        ordered = sub.sort_values(["seg_id", "frame_idx"])
        scores = ordered["frame_prob"].to_numpy(dtype=np.float64)
        strat = {}
        for col in STRAT_COLUMNS:
            if col in ordered.columns:
                vals = ordered[col].dropna().tolist()
                # Mode, fall back to first-non-null
                strat[col] = vals[0] if vals else None
        out.append((sess, scores, strat))
    return out


def main(ckpt_name: str) -> None:
    cfg = get_ckpt_cfg(ckpt_name)
    dirs = ensure_dirs(ckpt_name)
    full_name = cfg["full_name"]
    parquet_path = load_ckpts_yaml()["parquet"]["path"]
    if not Path(parquet_path).is_absolute():
        from _common import REPO_ROOT
        parquet_path = str(REPO_ROOT / parquet_path)

    optimal_path = dirs["data"] / "optimal_policy.json"
    if not optimal_path.exists():
        print(f"[phase5] optimal_policy.json missing — run phase4 first")
        return
    sel = json.loads(optimal_path.read_text())
    chosen = sel.get("chosen")
    if chosen is None:
        (dirs["findings"] / "phase5_stratified.md").write_text(
            f"# Phase 5 — stratified eval — {full_name}\n\nNo chosen policy from Phase 4.\n"
        )
        return

    policy = {
        "window_size": chosen["window_size"],
        "strategy": chosen["strategy"],
        "strategy_params": chosen["strategy_params"],
        "override_rule": chosen["override_rule"],
    }
    print(f"[phase5] Using policy {chosen['policy_id']}: {policy}")

    t0 = time.time()
    rows = []
    join_coverage_rows = []
    for suite_name in cfg["suites"].keys():
        try:
            df = load_suite_csv(cfg, suite_name)
        except FileNotFoundError:
            continue
        joined = load_parquet_join(parquet_path, df)
        n_total = len(joined)
        # Coverage report
        for col in STRAT_COLUMNS:
            if col in joined.columns:
                pct = joined[col].notna().mean()
                join_coverage_rows.append(
                    {"suite": suite_name, "strat_col": col, "join_coverage": float(pct)}
                )
        streams = stream_score_arrays_with_strat(joined)

        # Global readout (no stratification) for reference.
        n_streams = len(streams)
        n_flagged = sum(1 for _, s, _ in streams if simulate_video(s, policy))
        rows.append(
            {
                "suite": suite_name,
                "label_class": df["label_class"].iloc[0],
                "stratum_col": "(global)",
                "stratum_val": "(all)",
                "n_streams": int(n_streams),
                "n_flagged": int(n_flagged),
                "rate": float(n_flagged / n_streams) if n_streams else float("nan"),
            }
        )

        # Per-stratum readouts.
        for col in STRAT_COLUMNS:
            if col not in joined.columns:
                continue
            # Bucket streams by their dominant value of `col`.
            buckets: Dict[str, list] = {}
            for sess, scores, strat in streams:
                v = strat.get(col)
                if v is None or pd.isna(v):
                    continue
                buckets.setdefault(str(v), []).append((sess, scores))
            for v, members in buckets.items():
                n_s = len(members)
                if n_s == 0:
                    continue
                n_f = sum(1 for _, s in members if simulate_video(s, policy))
                rows.append(
                    {
                        "suite": suite_name,
                        "label_class": df["label_class"].iloc[0],
                        "stratum_col": col,
                        "stratum_val": v,
                        "n_streams": int(n_s),
                        "n_flagged": int(n_f),
                        "rate": float(n_f / n_s) if n_s else float("nan"),
                    }
                )

    res = pd.DataFrame(rows)
    res.to_csv(dirs["data"] / "stratified_eval.csv", index=False)
    cov = pd.DataFrame(join_coverage_rows)
    cov.to_csv(dirs["data"] / "stratified_join_coverage.csv", index=False)

    # Build markdown.
    md = [f"# Phase 5 — cross-condition robustness — {full_name}", ""]
    md.append(f"**Chosen policy**: `{chosen['strategy']}` W={chosen['window_size']} "
              f"params={chosen['strategy_params']} override=`{chosen['override_rule']}`\n")

    # For each suite show global rate then per-stratum deltas
    md.append("## Per-stratum vs global rate\n")
    md.append("Δ shows stratum_rate − global_rate. For real suites: Δ>+5pp = FPR rises (bad); "
              "for fake suites: Δ<-10pp = recall drops (bad).\n")

    flagged_rows = []
    for suite_name in cfg["suites"].keys():
        sub = res[res["suite"] == suite_name]
        if sub.empty:
            continue
        glob = sub[sub["stratum_col"] == "(global)"].iloc[0]
        md.append(f"\n### {suite_name} ({glob['label_class']}) — global rate = {glob['rate']:.3f} (n={int(glob['n_streams'])})\n")
        md.append("| stratum_col | stratum_val | n | rate | Δ vs global |")
        md.append("|---|---|---:|---:|---:|")
        per_strat = sub[sub["stratum_col"] != "(global)"].copy()
        per_strat["delta"] = per_strat["rate"] - glob["rate"]
        per_strat = per_strat.sort_values(["stratum_col", "stratum_val"])
        for _, r in per_strat.iterrows():
            md.append(
                f"| {r['stratum_col']} | {r['stratum_val']} | {int(r['n_streams'])} | "
                f"{r['rate']:.3f} | {r['delta']:+.3f} |"
            )
            # Threshold-flag
            if r["label_class"] == "real" and r["delta"] > 0.05:
                flagged_rows.append((suite_name, r["stratum_col"], r["stratum_val"], "FPR rises >5pp", r["delta"]))
            if r["label_class"] == "fake" and r["delta"] < -0.10:
                flagged_rows.append((suite_name, r["stratum_col"], r["stratum_val"], "recall drops >10pp", r["delta"]))

    md.append("\n## Robustness flags\n")
    if flagged_rows:
        md.append("| suite | stratum_col | stratum_val | issue | Δ |")
        md.append("|---|---|---|---|---:|")
        for x in flagged_rows:
            md.append(f"| {x[0]} | {x[1]} | {x[2]} | {x[3]} | {x[4]:+.3f} |")
    else:
        md.append("**No strata exceed the robustness thresholds.** Policy holds across conditions.")

    md.append("\n## Parquet join coverage\n")
    md.append("| suite | strat_col | coverage |")
    md.append("|---|---|---:|")
    for _, r in cov.iterrows():
        md.append(f"| {r['suite']} | {r['strat_col']} | {r['join_coverage']:.3f} |")

    md.append(f"\n_Wall time: {time.time() - t0:.1f}s_")
    (dirs["findings"] / "phase5_stratified.md").write_text("\n".join(md))
    print(f"[phase5] {full_name} done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    main(a.ckpt)
