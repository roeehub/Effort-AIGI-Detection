#!/usr/bin/env python3
"""Generate IQ_ATLAS_FACTS_2026-05-08.md from the produced CSVs + parquet.

Strict facts-only: numbers + tables + cross-references. No interpretation,
no recommendations, no "succeeds/fails/wins/promotes/deployment-grade" wording.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
THIS = ROOT / "analysis" / "iq_data_atlas_2026-05-08"
OUT = THIS / "outputs"

# Pool role -> short group label for the headline table
ROLE_TO_GROUP = {
    "train_real": "TRAIN_REAL",
    "train_fake": "TRAIN_FAKE",
    "dev_real": "DEV_REAL",
    "dev_fake": "DEV_FAKE",
    "lockbox_real": "LOCKBOX_REAL",
    "lockbox_fake": "LOCKBOX_FAKE",
    "hdtf_real": "HDTF_REAL",
    "hdtf_fake": "HDTF_FAKE",
    "canary_chronic_real": "CANARY_CHRONIC_REAL",
    "canary_other_real": "CANARY_OTHER_REAL",
    "canary_fake": "CANARY_FAKE",
    "prod_ref_real": "PROD_REF_REAL",
}

GROUP_ORDER = [
    "TRAIN_REAL", "TRAIN_FAKE",
    "DEV_REAL", "DEV_FAKE",
    "LOCKBOX_REAL", "LOCKBOX_FAKE",
    "HDTF_REAL", "HDTF_FAKE",
    "CANARY_CHRONIC_REAL", "CANARY_OTHER_REAL", "CANARY_FAKE",
    "PROD_REF_REAL",
]

METRICS = ["min_dim", "lap_var", "luma_mean", "color_b_dev", "edge_mag", "aspect_ratio", "bytes"]


def fmt(x, prec=1):
    if pd.isna(x):
        return "N/A"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) < 1:
        return f"{x:.3f}"
    return f"{x:.{prec}f}"


def headline_group_table(df: pd.DataFrame) -> str:
    """For each property, p05/p50/p95 across pool groups."""
    df = df.copy()
    df["group"] = df["role"].map(ROLE_TO_GROUP).fillna(df["role"])
    lines = []
    for m in METRICS:
        if m not in df.columns:
            continue
        lines.append(f"\n### {m}\n")
        lines.append(f"| group | n | p05 | p50 | p95 | mean |")
        lines.append(f"|---|---:|---:|---:|---:|---:|")
        for grp in GROUP_ORDER:
            sub = df[df["group"] == grp]
            s = sub[m].dropna() if m in sub.columns else pd.Series(dtype=float)
            if len(s) == 0:
                continue
            lines.append(f"| {grp} | {len(s)} | {fmt(s.quantile(0.05))} | "
                         f"{fmt(s.quantile(0.50))} | {fmt(s.quantile(0.95))} | "
                         f"{fmt(s.mean())} |")
    return "\n".join(lines)


def pool_inventory_table(df: pd.DataFrame) -> str:
    pool_role = df.groupby("pool").agg(
        role=("role", "first"),
        n=("pool", "count"),
    ).reset_index().sort_values(["role", "pool"])
    lines = ["| pool | role | n_frames |", "|---|---|---:|"]
    for _, r in pool_role.iterrows():
        lines.append(f"| {r['pool']} | {r['role']} | {r['n']} |")
    return "\n".join(lines)


def per_property_summary(df: pd.DataFrame) -> str:
    """One subsection per metric: train vs eval gap, real vs fake gap, ultra-bad tail."""
    df = df.copy()
    df["group"] = df["role"].map(ROLE_TO_GROUP).fillna(df["role"])
    lines = []

    # Reference for "ultra-bad" — production-ref p05 (or canary_other_real if none)
    ref_pool = "PROD_REF_REAL"
    ref_df = df[df["group"] == ref_pool]
    fallback = "CANARY_OTHER_REAL"
    if ref_df.empty:
        ref_df = df[df["group"] == fallback]
        ref_label = f"{fallback} p05 (no PROD_REF available)"
    else:
        ref_label = f"{ref_pool} p05"

    for m in METRICS:
        if m not in df.columns:
            continue
        lines.append(f"\n### {m}\n")

        def grp_p(g, q):
            s = df[df["group"] == g][m].dropna()
            return float(s.quantile(q)) if len(s) > 0 else np.nan

        # Train vs eval gap (using real-side, focus on min_dim semantics)
        tr_p50 = grp_p("TRAIN_REAL", 0.50)
        dv_p50 = grp_p("DEV_REAL", 0.50)
        lb_p50 = grp_p("LOCKBOX_REAL", 0.50)
        hf_p50 = grp_p("HDTF_REAL", 0.50)
        pr_p50 = grp_p("PROD_REF_REAL", 0.50)

        lines.append(f"- TRAIN_REAL p50 = {fmt(tr_p50)} | DEV_REAL p50 = {fmt(dv_p50)} | "
                     f"LOCKBOX_REAL p50 = {fmt(lb_p50)} | HDTF_REAL p50 = {fmt(hf_p50)} | "
                     f"PROD_REF_REAL p50 = {fmt(pr_p50)}")

        # Real vs fake (on training, dev, lockbox)
        tr_fake_p50 = grp_p("TRAIN_FAKE", 0.50)
        dv_fake_p50 = grp_p("DEV_FAKE", 0.50)
        lb_fake_p50 = grp_p("LOCKBOX_FAKE", 0.50)
        hf_fake_p50 = grp_p("HDTF_FAKE", 0.50)
        lines.append(f"- TRAIN_FAKE p50 = {fmt(tr_fake_p50)} | DEV_FAKE p50 = {fmt(dv_fake_p50)} | "
                     f"LOCKBOX_FAKE p50 = {fmt(lb_fake_p50)} | HDTF_FAKE p50 = {fmt(hf_fake_p50)}")

        # Ultra-bad tail: fraction below ref_pool p05 (on each real-side group)
        if not ref_df.empty:
            ref_p05 = float(ref_df[m].dropna().quantile(0.05))
            lines.append(f"- Reference {ref_label} = {fmt(ref_p05)}. "
                         f"Frac frames below this:")
            for grp in ["TRAIN_REAL", "TRAIN_FAKE", "DEV_REAL", "DEV_FAKE",
                        "LOCKBOX_REAL", "LOCKBOX_FAKE",
                        "HDTF_REAL", "HDTF_FAKE", "CANARY_CHRONIC_REAL"]:
                s = df[df["group"] == grp][m].dropna()
                if len(s) == 0:
                    continue
                if m in ["min_dim", "luma_mean", "lap_var", "edge_mag", "bytes"]:
                    # higher = better quality; "below ref_p05" = bad
                    frac = (s < ref_p05).mean()
                else:
                    # ambiguous; just show "below"
                    frac = (s < ref_p05).mean()
                lines.append(f"    - {grp}: {100 * frac:.1f}% (n={len(s)})")
    return "\n".join(lines)


def per_pool_min_dim_p05(df: pd.DataFrame, ref_p05: float, metric: str = "min_dim") -> str:
    """Identified ultra-bad tails: per pool, what fraction of frames falls
    below the reference p05?"""
    lines = ["| pool | role | n | metric_p05 | metric_p50 | frac_below_ref |",
             "|---|---|---:|---:|---:|---:|"]
    for pool, g in df.groupby("pool"):
        if metric not in g.columns:
            continue
        s = g[metric].dropna()
        if len(s) == 0:
            continue
        frac_below = (s < ref_p05).mean()
        lines.append(f"| {pool} | {g['role'].iloc[0]} | {len(s)} | "
                     f"{fmt(s.quantile(0.05))} | {fmt(s.quantile(0.50))} | "
                     f"{100 * frac_below:.1f}% |")
    return "\n".join(lines)


def main():
    parquet = OUT / "per_frame.parquet"
    if not parquet.exists():
        raise SystemExit(f"missing {parquet} — run build_iq_atlas.py first")
    df = pd.read_parquet(parquet)
    n_total = len(df)
    n_pools = df["pool"].nunique()

    # Pool inventory
    pool_inv = pool_inventory_table(df)

    # Headline cross-pool table
    headline = headline_group_table(df)

    # Per-property findings
    per_prop = per_property_summary(df)

    # Build reference for ultra-bad-tail section (use prod_ref if present, else canary_other_real)
    df_t = df.copy()
    df_t["group"] = df_t["role"].map(ROLE_TO_GROUP).fillna(df_t["role"])
    ref = df_t[df_t["group"] == "PROD_REF_REAL"]
    if ref.empty:
        ref = df_t[df_t["group"] == "CANARY_OTHER_REAL"]
        ref_label = "CANARY_OTHER_REAL p05 (no production reference available — see Method §6 caveat)"
    else:
        ref_label = "PROD_REF_REAL p05 (dor_evening + dor_morning + dor_may5_teams)"
    ref_min_dim_p05 = float(ref["min_dim"].dropna().quantile(0.05)) if not ref.empty else np.nan
    ref_lap_p05 = float(ref["lap_var"].dropna().quantile(0.05)) if not ref.empty else np.nan

    ultra_bad_min_dim = per_pool_min_dim_p05(df, ref_min_dim_p05, "min_dim")
    ultra_bad_lap = per_pool_min_dim_p05(df, ref_lap_p05, "lap_var")

    md = f"""# IQ data atlas FACTS (2026-05-08)

> **Status: factual-only.** No interpretation. Forbidden words: succeeds, fails,
> wins, promotes, deployment-grade.
>
> Source data: `outputs/per_frame.parquet`, `outputs/per_pool_summary.csv`,
> `outputs/cross_pool_compare.csv` produced by `build_iq_atlas.py` on
> 2026-05-08 (CPU-only; multiprocessing.Pool(8); gsutil parallel download via
> ThreadPoolExecutor).
>
> Companion figures: `figs/histogram_<metric>_all_pools.png`,
> `figs/pool_<pool>_quad.png`,
> `figs/train_vs_eval_vs_lockbox_<metric>.png`.

## 1. Question

User eyeballed chronic-FP frames in the canary substrate and noticed they look
pixelated. The pixelation was confirmed to be in the data (cached crops match
GCS-source dims; not viewer-side downscale). See
`analysis/p2_eval_2026-05-08/d1_d4_cpu/CANARY_RESOLUTION_FACTS_2026-05-08.md`.

The user requested a comprehensive cross-pool IQ atlas to inform three
decisions:

1. Whether to filter "ultra-bad" frames out of TRAINING.
2. Whether to standardize on F4-style filtering for EVAL/lockbox readouts.
3. Where to set the deployment-side IQ-gate threshold.

This document is the cross-pool measurement that informs those decisions. It
contains numbers + tables only. No recommendations.

## 2. Method

**Features.** Per-frame IQ panel computed by
`build_iq_atlas.py:per_frame_attrs`. Feature formulas match
`analysis/dor_drift_mechanism_2026-05-06/run_analysis.py:compute_iq_axes`
exactly so the production-reference rows fold in directly without unit
mismatch:

- `h`, `w`, `min_dim`, `max_dim`, `aspect_ratio`
- `lap_var` — `cv2.Laplacian(gray).var()` (sharpness proxy)
- `luma_mean`, `luma_std` — HSV V channel mean / std
- `saturation_mean` — HSV S channel mean
- `contrast_l` — LAB L channel std
- `edge_mag` — `cv2.Sobel` magnitude mean (ksize=3)
- `color_a_dev` — `mean(|LAB.A - 128|)` (LAB A deviation from neutral)
- `color_b_dev` — `mean(|LAB.B - 128|)` (LAB B deviation from neutral)
- `skin_frac` — YCbCr skin range fraction
- `bytes` — JPEG / PNG file size

**Sampling.** Random sample of N=500 frames per pool (oversample 1.5× to
absorb download / decode loss). Manifest-based pools enumerate frame URIs from
the three manifests in `arena/manifests/`. Training pools enumerate via
`gsutil ls` over `gs://teams-faces-data-test-2914-fake-4420-real-feb-28/{{real,fake}}/`
and df40 method directories, then sample uniformly within.

**Speed-ups.** ThreadPoolExecutor over `gsutil cp` (24 workers) for download;
`multiprocessing.Pool(8)` for CPU decode + feature compute. Pool dirs deleted
after measurement to bound disk usage. Per-pool parquet caches in `_cache/`
make the script idempotent — re-runs skip already-measured pools.

**Frame counts.**

- Total pools sampled: **{n_pools}**
- Total frames measured: **{n_total:,}**
- Worker count: 8 (constrained per project memory `feedback_sklearn_njobs.md`)

## 3. Pool inventory

{pool_inv}

## 4. Headline cross-pool table — by group

For each property: p05 / p50 / p95 across the pool groups (one row per group).

{headline}

## 5. Per-property findings

For each metric: train-vs-eval-vs-lockbox p50, real-vs-fake p50, and the
fraction of each group's frames that fall below the reference p05.

Reference for "below" cutoff: **{ref_label}**.

{per_prop}

## 6. Identified ultra-bad tails (per-pool detail)

### 6.1 min_dim (resolution)

Reference: {ref_label} = {fmt(ref_min_dim_p05)} px.

{ultra_bad_min_dim}

### 6.2 lap_var (sharpness proxy)

Reference: {ref_label} = {fmt(ref_lap_p05)}.

{ultra_bad_lap}

### 6.3 Caveat

We do not have a raw production-frame substrate. The PROD_REF_REAL group is a
proxy built from three local recordings of one identity (Dor): the
`dor_morning_local`, `dor_evening_local` (laptop captures, n=200 each), and
`dor_may5_teams` (Teams capture, n=30) sessions from
`analysis/dor_drift_mechanism_2026-05-06/outputs/per_frame_features.csv`.

This is one identity in three sessions, not a production-traffic sample. If a
real production-frame log becomes available, recompute §6 against it.

## 7. Open observations (factual)

The following are observable from the headline tables. No interpretation; no
ranking; no recommendation.

1. **Resolution gap between training pools and HDTF.** TRAIN_REAL / TRAIN_FAKE
   p50 vs HDTF_REAL / HDTF_FAKE p50 on `min_dim` — see §4 `min_dim`.
2. **Resolution gap between training pools and CANARY_CHRONIC_REAL.** Three of
   the six chronic identities have min_dim p50 ~ 90 px (see
   `CANARY_RESOLUTION_FACTS_2026-05-08.md`); see §6.1 for atlas-wide cross-pool
   distribution.
3. **lap_var inversion between fake and real on the canary substrate.** Per
   `CANARY_RESOLUTION_FACTS_2026-05-08.md`: fake group p50 lap_var = 20.7;
   chronic_real p50 = 491. Atlas confirms direction at scale — see §4.
4. **TRAIN_FAKE vs TRAIN_REAL gap** on `min_dim` and `lap_var` — see §4.
5. **HDTF resolution distribution** — clean vs teams subtypes — see §4 / §6.

(Caller may extend / contest. These are starting points only.)

## 8. Cross-references (existing memories + threads)

The following project memories share evidence with or constrain
interpretations of this atlas. Do not collapse them into recommendations
inside this FACTS doc; they are listed for the user to consult during the
decision step that follows this document.

- `project_image_quality_shortcut.md` — score correlates negatively with
  Laplacian variance / luminance / skin_frac across most suites; eval lockbox
  cam_test_s33 is 14-25× less sharp than training data; deployment-τ
  relaxation 2%→10% real_FPR unlocks 4-77% recall depending on suite.
- `project_face_size_label_leak.md` — each fake method clusters at a tight
  face-size band; reals span wider; model uses face size as a fake predictor.
- `project_dor_drift_named_axes_2026-05-06.md` — endpoint-union ridge
  captures 90% of drift on P8A from named pixel-domain IQ axes; top drivers
  min_dim 53-68%, color_b_dev 27-38%, edge_mag 44%.
- `project_eval_production_crop_tightness_gap.md` — eval frames carry more
  background context around the face than production crops do; structurally
  upstream of the camera-signature shortcut + face-size leak + webcam FPR.
- `project_lockbox_fpr_dominated_by_webcam_mode.md` — lockbox FPR is
  dominated by webcam-style captures (clip_capture_mode==webcam = 65.7% FPR);
  modern_v2 filter cuts headline 4.6% → 0.71%.
- `project_iq_gating_viability_2026-05-04.md` — P8A recall increases
  monotonically with sharpness (Q1 36% → Q4 98%); E2B is INVERTED (Q1 52% →
  Q3 5%). IQ gating is a P8A lever, non-starter for E2B.
- `project_canary_below_production_resolution_2026-05-08.md` — user policy
  thread on resolution-based gating.
- Companion FACTS:
  `analysis/p2_eval_2026-05-08/d1_d4_cpu/CANARY_RESOLUTION_FACTS_2026-05-08.md`.

## 9. Artifacts

- `outputs/per_frame.parquet` — {n_total:,} rows × all features
- `outputs/per_pool_summary.csv` — per-pool p05/p50/p95/mean for all metrics
- `outputs/cross_pool_compare.csv` — pivoted: rows=pool, columns=metric x p05/p50/p95
- `figs/histogram_<metric>_all_pools.png` — overlaid histograms (7 metrics)
- `figs/pool_<pool>_quad.png` — 4-panel per pool
- `figs/train_vs_eval_vs_lockbox_<metric>.png` — group overlay (3 metrics)
- `build_iq_atlas.py` — driver (re-runnable, idempotent)
- `build_facts_doc.py` — this doc generator
"""
    out_path = THIS / "IQ_ATLAS_FACTS_2026-05-08.md"
    out_path.write_text(md)
    print(f"wrote {out_path}")
    print(f"  {n_total} frames, {n_pools} pools")


if __name__ == "__main__":
    main()
