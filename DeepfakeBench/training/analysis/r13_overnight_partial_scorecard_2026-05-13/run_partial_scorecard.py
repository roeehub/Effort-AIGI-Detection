"""Partial scorecard mining for r13-overnight-scorecard-2026-05-13.

Reads per-suite _videos_report.csv files from the in-flight Vertex job,
calibrates per-ckpt τ on teams_real_all_dev (video-level avg_video_prob,
real_fpr ≤ 0.07), and tabulates partial Pillar-3 metrics.

Inputs:
  ./_reports_cache/<suite>_<ckpt>_videos_report.csv

Outputs:
  ./_partial_scorecard_<date>.csv (sorted ascending by lockbox_real_fpr)
  Console-readable markdown table

Policy: v3-fix (target_real_fpr=0.07; floors only used for diagnostic gating).
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent
CACHE = HERE / "_reports_cache"
DATE = "2026-05-13"


CKPTS = [
    # anchors first
    "p8a_reference_step5000",
    "e2b_top_n_step3200",
    "t5c_periodic_step3500",
    # slot 1: LoRA-P8A
    "slot1_lora_p8a_periodic_step1500",
    "slot1_lora_p8a_periodic_step3500",
    "slot1_lora_p8a_top_n_step2000",
    # slot 2: LoRA-T5C
    "slot2_lora_t5c_periodic_step1500",
    "slot2_lora_t5c_periodic_step2500",
    "slot2_lora_t5c_periodic_step3500",
    # slot 3: T5C+jitter@0.30
    "slot3_t5c_jitter030_periodic_step1500",
    "slot3_t5c_jitter030_periodic_step3500",
    "slot3_t5c_jitter030_top_n_step4500",
    # slot 4: B16-scratch+Fourier
    "slot4_b16sc_fourier_periodic_step5000",
    "slot4_b16sc_fourier_periodic_step6000",
    "slot4_b16sc_fourier_top_n_step10000",
]

# 9-cell contract suites we care about (subset of the 29-suite manifest).
CONTRACT_SUITES = [
    "teams_real_all_dev",            # τ-calibration cell
    "teams_real_poor_quality_dev",   # stress
    "teams_real_lighting_extreme_dev",  # stress
    "teams_fake_all_dev",            # macro recall
    "visomaster_enhanced_macro_dev", # macro recall
    "deeplive_enhanced_dev",         # macro recall
    "teams_real_all_lockbox",        # readout
    "teams_fake_all_lockbox",        # readout
    "teams_real_dor_dev",            # n=50 chronic
]

TAU_TARGET_REAL_FPR = 0.07
TAU_FALLBACK = 0.5
RECALL_FLOOR = 0.30
STRESS_FPR_CEIL = 0.10


def load_videos_report(suite: str, ckpt: str) -> Optional[list[dict]]:
    """Load a videos_report.csv into a list of {video_id, label, avg_video_prob}.

    Returns None if file missing.
    """
    path = CACHE / f"{suite}_{ckpt}_videos_report.csv"
    if not path.exists():
        return None
    rows: list[dict] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "video_id": r["video_id"],
                "label": int(r["label"]),
                "avg_video_prob": float(r["avg_video_prob"]),
            })
    return rows


def fpr_at_tau(rows: list[dict], tau: float) -> float:
    """Fraction of label=0 (real) videos with avg_video_prob >= tau."""
    reals = [r for r in rows if r["label"] == 0]
    if not reals:
        return float("nan")
    n_fp = sum(1 for r in reals if r["avg_video_prob"] >= tau)
    return n_fp / len(reals)


def recall_at_tau(rows: list[dict], tau: float) -> float:
    """Fraction of label=1 (fake) videos with avg_video_prob >= tau."""
    fakes = [r for r in rows if r["label"] == 1]
    if not fakes:
        return float("nan")
    n_tp = sum(1 for r in fakes if r["avg_video_prob"] >= tau)
    return n_tp / len(fakes)


def calibrate_tau(rows: list[dict], target_fpr: float = TAU_TARGET_REAL_FPR) -> Optional[dict]:
    """Find the smallest τ on teams_real_all_dev such that real_fpr ≤ target_fpr.

    Matches the scorer's v3-fix convention: pick τ that yields the largest possible
    real_fpr satisfying ≤ target_fpr. This is the "smallest τ" (lowest threshold)
    that still satisfies the FPR floor strictly.

    Cross-check vs official T5C scorecard 2026-05-12: P8A_REFERENCE_STEP5000
    published τ=0.915605, attained_fpr=0.06947. Our calibration matches within
    ties to ~mid-point between rank 225 and 226 of the sorted probs.

    Returns dict {tau, attained_fpr, n_reals}, or None if rows is None.
    """
    if rows is None:
        return None
    reals = [r for r in rows if r["label"] == 0]
    if not reals:
        return None
    probs = sorted([r["avg_video_prob"] for r in reals], reverse=True)
    n_reals = len(probs)
    # We want the largest n_fp satisfying n_fp/n ≤ target_fpr → n_fp ≤ floor(target_fpr * n).
    max_n_fp = int(target_fpr * n_reals)  # floor
    # We seek τ such that exactly max_n_fp probs satisfy prob >= τ.
    # Sorted descending, τ is set just above probs[max_n_fp] (so probs[0..max_n_fp-1] qualify,
    # giving FPR = max_n_fp / n_reals).
    if max_n_fp >= n_reals:
        tau = 0.0
    elif max_n_fp == 0:
        # τ = max prob + tiny ε; we use the max prob exactly, which yields FPR = (n with prob>=max)/n.
        # In the absence of ties at the top, that's 1/n. For our purposes, use the max+ tiny ε
        # represented as 1.0001*max+1e-9 to avoid floats. Approximation: take τ exactly.
        tau = max(probs) + 1e-9
    else:
        # Use midpoint between probs[max_n_fp-1] (the last to be accepted as FP) and
        # probs[max_n_fp] (the first to be rejected). This yields a clean threshold that
        # admits exactly max_n_fp false positives.
        upper = probs[max_n_fp - 1]
        lower = probs[max_n_fp]
        tau = (upper + lower) / 2.0
    attained_fpr = fpr_at_tau(rows, tau)
    return {"tau": tau, "attained_fpr": attained_fpr, "n_reals": n_reals}


@dataclass
class PerCkptMetrics:
    ckpt: str
    tau: Optional[float] = None
    tau_source: str = "MISSING"  # "calibrated", "fallback_0.5", "MISSING"
    dev_real_fpr_at_tau: Optional[float] = None
    n_reals_dev: Optional[int] = None
    poor_quality_real_fpr: Optional[float] = None
    lighting_extreme_real_fpr: Optional[float] = None
    teams_fake_all_dev_recall: Optional[float] = None
    visomaster_enh_macro_dev_recall: Optional[float] = None
    deeplive_enh_dev_recall: Optional[float] = None
    dev_fake_macro_recall: Optional[float] = None  # mean of the three
    lockbox_real_fpr: Optional[float] = None
    lockbox_fake_recall: Optional[float] = None
    n_real_lockbox: Optional[int] = None
    n_fake_lockbox: Optional[int] = None
    teams_real_dor_dev_fpr: Optional[float] = None
    n_dor_dev: Optional[int] = None
    cells_missing: list[str] = field(default_factory=list)


def compute_metrics(ckpt: str) -> PerCkptMetrics:
    m = PerCkptMetrics(ckpt=ckpt)

    cal_rows = load_videos_report("teams_real_all_dev", ckpt)
    if cal_rows is None:
        m.cells_missing.append("teams_real_all_dev")
        m.tau = TAU_FALLBACK
        m.tau_source = "fallback_0.5"
    else:
        cal = calibrate_tau(cal_rows)
        if cal is None:
            m.tau = TAU_FALLBACK
            m.tau_source = "fallback_0.5"
        else:
            m.tau = cal["tau"]
            m.tau_source = "calibrated"
            m.dev_real_fpr_at_tau = cal["attained_fpr"]
            m.n_reals_dev = cal["n_reals"]

    tau = m.tau

    # Compute per-suite metrics at this τ
    def get_fpr(suite: str) -> Optional[float]:
        rows = load_videos_report(suite, ckpt)
        if rows is None:
            m.cells_missing.append(suite)
            return None
        return fpr_at_tau(rows, tau)

    def get_recall(suite: str) -> Optional[float]:
        rows = load_videos_report(suite, ckpt)
        if rows is None:
            m.cells_missing.append(suite)
            return None
        return recall_at_tau(rows, tau)

    def get_n_reals(suite: str) -> Optional[int]:
        rows = load_videos_report(suite, ckpt)
        if rows is None:
            return None
        return sum(1 for r in rows if r["label"] == 0)

    def get_n_fakes(suite: str) -> Optional[int]:
        rows = load_videos_report(suite, ckpt)
        if rows is None:
            return None
        return sum(1 for r in rows if r["label"] == 1)

    m.poor_quality_real_fpr = get_fpr("teams_real_poor_quality_dev")
    m.lighting_extreme_real_fpr = get_fpr("teams_real_lighting_extreme_dev")
    m.teams_fake_all_dev_recall = get_recall("teams_fake_all_dev")
    m.visomaster_enh_macro_dev_recall = get_recall("visomaster_enhanced_macro_dev")
    m.deeplive_enh_dev_recall = get_recall("deeplive_enhanced_dev")
    if (m.teams_fake_all_dev_recall is not None
            and m.visomaster_enh_macro_dev_recall is not None
            and m.deeplive_enh_dev_recall is not None):
        m.dev_fake_macro_recall = (
            m.teams_fake_all_dev_recall
            + m.visomaster_enh_macro_dev_recall
            + m.deeplive_enh_dev_recall
        ) / 3.0
    m.lockbox_real_fpr = get_fpr("teams_real_all_lockbox")
    m.lockbox_fake_recall = get_recall("teams_fake_all_lockbox")
    m.n_real_lockbox = get_n_reals("teams_real_all_lockbox")
    m.n_fake_lockbox = get_n_fakes("teams_fake_all_lockbox")
    m.teams_real_dor_dev_fpr = get_fpr("teams_real_dor_dev")
    m.n_dor_dev = get_n_reals("teams_real_dor_dev")
    return m


def main() -> None:
    metrics = [compute_metrics(c) for c in CKPTS]

    # Coverage matrix
    coverage = {}
    for c in CKPTS:
        coverage[c] = {}
        for s in CONTRACT_SUITES:
            rows = load_videos_report(s, c)
            coverage[c][s] = "OK" if rows is not None else "MISSING"
    (HERE / f"_coverage_{DATE}.json").write_text(json.dumps(coverage, indent=2))

    # Write a CSV with full metric grid.
    out_csv = HERE / f"_partial_scorecard_{DATE}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ckpt",
            "tau",
            "tau_source",
            "dev_real_fpr_at_tau",
            "n_reals_dev",
            "poor_quality_real_fpr",
            "lighting_extreme_real_fpr",
            "teams_fake_all_dev_recall",
            "visomaster_enh_macro_dev_recall",
            "deeplive_enh_dev_recall",
            "dev_fake_macro_recall",
            "lockbox_real_fpr",
            "lockbox_fake_recall",
            "n_real_lockbox",
            "n_fake_lockbox",
            "teams_real_dor_dev_fpr",
            "n_dor_dev",
            "cells_missing",
        ])
        for m in metrics:
            w.writerow([
                m.ckpt,
                f"{m.tau:.6f}" if m.tau is not None else "",
                m.tau_source,
                f"{m.dev_real_fpr_at_tau:.4f}" if m.dev_real_fpr_at_tau is not None else "MISSING",
                m.n_reals_dev or "",
                f"{m.poor_quality_real_fpr:.4f}" if m.poor_quality_real_fpr is not None else "MISSING",
                f"{m.lighting_extreme_real_fpr:.4f}" if m.lighting_extreme_real_fpr is not None else "MISSING",
                f"{m.teams_fake_all_dev_recall:.4f}" if m.teams_fake_all_dev_recall is not None else "MISSING",
                f"{m.visomaster_enh_macro_dev_recall:.4f}" if m.visomaster_enh_macro_dev_recall is not None else "MISSING",
                f"{m.deeplive_enh_dev_recall:.4f}" if m.deeplive_enh_dev_recall is not None else "MISSING",
                f"{m.dev_fake_macro_recall:.4f}" if m.dev_fake_macro_recall is not None else "MISSING",
                f"{m.lockbox_real_fpr:.4f}" if m.lockbox_real_fpr is not None else "MISSING",
                f"{m.lockbox_fake_recall:.4f}" if m.lockbox_fake_recall is not None else "MISSING",
                m.n_real_lockbox or "",
                m.n_fake_lockbox or "",
                f"{m.teams_real_dor_dev_fpr:.4f}" if m.teams_real_dor_dev_fpr is not None else "MISSING",
                m.n_dor_dev or "",
                ";".join(m.cells_missing),
            ])

    # Sort ascending by lockbox_real_fpr; ckpts with missing lockbox go to bottom.
    def lb_key(m: PerCkptMetrics) -> tuple:
        if m.lockbox_real_fpr is None:
            return (1, 0.0)
        return (0, m.lockbox_real_fpr)

    sorted_metrics = sorted(metrics, key=lb_key)

    # Print leaderboard
    print("=" * 100)
    print(f"PARTIAL LEADERBOARD — sorted ascending by lockbox_real_fpr ({DATE})")
    print("=" * 100)
    hdr = (
        f"{'ckpt':40s} | {'τ':>7s} | {'src':>10s} | "
        f"{'lb_fpr':>7s} | {'lb_rec':>7s} | {'dev_mac':>7s} | "
        f"{'viso':>7s} | {'dl_enh':>7s} | {'tf_dev':>7s} | "
        f"{'dor':>6s} | missing"
    )
    print(hdr)
    print("-" * len(hdr))
    for m in sorted_metrics:
        def fmt(x, w=7, p=4):
            if x is None:
                return "MISSING".rjust(w)
            return f"{x:>{w}.{p}f}"
        print(
            f"{m.ckpt:40s} | "
            f"{fmt(m.tau, 7, 4)} | "
            f"{m.tau_source:>10s} | "
            f"{fmt(m.lockbox_real_fpr)} | "
            f"{fmt(m.lockbox_fake_recall)} | "
            f"{fmt(m.dev_fake_macro_recall)} | "
            f"{fmt(m.visomaster_enh_macro_dev_recall)} | "
            f"{fmt(m.deeplive_enh_dev_recall)} | "
            f"{fmt(m.teams_fake_all_dev_recall)} | "
            f"{fmt(m.teams_real_dor_dev_fpr, 6, 2)} | "
            f"{';'.join(m.cells_missing) if m.cells_missing else '-'}"
        )

    # Also dump JSON for downstream
    out_json = HERE / f"_partial_scorecard_{DATE}.json"
    out_json.write_text(json.dumps([asdict(m) for m in sorted_metrics], indent=2))
    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {HERE / f'_coverage_{DATE}.json'}")


if __name__ == "__main__":
    main()
