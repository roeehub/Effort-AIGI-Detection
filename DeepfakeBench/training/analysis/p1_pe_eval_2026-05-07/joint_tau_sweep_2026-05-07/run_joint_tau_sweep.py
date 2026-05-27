"""Joint tau sweep for P1 BUNDLE_step500 — F1+F4+F5 simultaneous-overlap probe.

Goal: determine whether a single global tau in [0.95, 0.999] simultaneously
satisfies F1 (lockbox fake recall >= 90%), F4 (max HDTF real FPR <= 5%), and
F5 (max per-chronic-identity FPR <= 10%).

Method: read per-frame frame_prob CSVs from phase_a/phase_c, sweep tau on a
100-point linear grid in [0.95, 0.999], compute the relevant metrics at each
tau, then intersect the per-criterion qualifying tau bands.

Chronic identity match: case-insensitive prefix match on the raw video_id
(NOT regex-stripped base_identity) — same rule as the bug-fixed
`phase_d/run_chronic_filter.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
RAW_A = ROOT / "analysis/p1_pe_eval_2026-05-07/raw_reports/phase_a"
RAW_C = ROOT / "analysis/p1_pe_eval_2026-05-07/raw_reports/phase_c"
OUT_DIR = ROOT / "analysis/p1_pe_eval_2026-05-07/joint_tau_sweep_2026-05-07"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_TAG = "p1_bundle_periodic_step500"

# Suites
LOCKBOX_REAL_SUITE = "teams_real_all_lockbox"
LOCKBOX_FAKE_SUITE = "teams_fake_all_lockbox"

HDTF_REAL_SUITES = [
    "proper_real_clean_dev",
    "proper_real_clean_lockbox",
    "proper_real_teams_dev",
    "proper_real_teams_lockbox",
]
HDTF_DIR = RAW_C  # phase_c

CHRONIC_IDS = [
    "PC_Generator__s22",
    "PC_Generator__s45",
    "Q__s6",
    "bla_bla_chow",
    "bla_bla_chow__s2",
    "roy_d",
]
CHRONIC_SOURCE_SUITE = "teams_real_all_dev"  # phase_a

# Bars
F1_RECALL_FLOOR = 0.90
F4_FPR_CEIL = 0.05
F5_FPR_CEIL = 0.10  # per identity

TAU_GRID = np.linspace(0.95, 0.999, 100)


def video_matches_cid(video_id: str, cid: str) -> bool:
    """Case-insensitive prefix match (same as phase_d FIXED rule)."""
    if not isinstance(video_id, str):
        return False
    vl = video_id.lower()
    cl = cid.lower()
    return vl == cl or vl.startswith(cl + "_") or vl.startswith(cl + "__")


def report_path_a(suite: str, ckpt: str) -> Path:
    return RAW_A / f"{suite}_{ckpt}_frames_report.csv"


def report_path_c(suite: str, ckpt: str) -> Path:
    return RAW_C / f"{suite}_{ckpt}_frames_report.csv"


def load_scores(path: Path) -> np.ndarray:
    df = pd.read_csv(path, usecols=["frame_prob"])
    return df["frame_prob"].to_numpy(dtype=np.float64)


def fpr_at_tau(scores: np.ndarray, tau: float) -> float:
    if scores.size == 0:
        return float("nan")
    return float((scores >= tau).sum()) / float(scores.size)


def recall_at_tau(scores: np.ndarray, tau: float) -> float:
    return fpr_at_tau(scores, tau)  # for fake-only suite, "fpr" of pos-class == recall


def main() -> int:
    # ---------------------- Load lockbox real & fake (BUNDLE_step500) ----------------------
    lb_real_path = report_path_a(LOCKBOX_REAL_SUITE, CKPT_TAG)
    lb_fake_path = report_path_a(LOCKBOX_FAKE_SUITE, CKPT_TAG)
    print(f"Loading {lb_real_path.name} ...")
    lb_real_scores = load_scores(lb_real_path)
    print(f"  n_real_lockbox = {lb_real_scores.size}")
    print(f"Loading {lb_fake_path.name} ...")
    lb_fake_scores = load_scores(lb_fake_path)
    print(f"  n_fake_lockbox = {lb_fake_scores.size}")

    # ---------------------- Load HDTF real suites ----------------------
    hdtf_real_scores: dict[str, np.ndarray] = {}
    for suite in HDTF_REAL_SUITES:
        path = report_path_c(suite, CKPT_TAG)
        print(f"Loading HDTF real {path.name} ...")
        hdtf_real_scores[suite] = load_scores(path)
        print(f"  n = {hdtf_real_scores[suite].size}")

    # ---------------------- Load chronic identities (from teams_real_all_dev) ----------------------
    chronic_path = report_path_a(CHRONIC_SOURCE_SUITE, CKPT_TAG)
    print(f"Loading chronic source {chronic_path.name} ...")
    chronic_full = pd.read_csv(chronic_path, usecols=["video_id", "frame_prob"])
    print(f"  n_total_real = {len(chronic_full)}")

    chronic_subsets: dict[str, np.ndarray] = {}
    for cid in CHRONIC_IDS:
        mask = chronic_full["video_id"].apply(lambda v, c=cid: video_matches_cid(v, c))
        sub_scores = chronic_full.loc[mask, "frame_prob"].to_numpy(dtype=np.float64)
        chronic_subsets[cid] = sub_scores
        print(f"  chronic {cid}: n = {sub_scores.size}")

    # ---------------------- Sweep tau ----------------------
    rows = []
    for tau in TAU_GRID:
        lb_real_fpr = fpr_at_tau(lb_real_scores, tau)
        lb_fake_recall = recall_at_tau(lb_fake_scores, tau)
        hdtf_per_suite = {s: fpr_at_tau(arr, tau) for s, arr in hdtf_real_scores.items()}
        max_hdtf_fpr = max(hdtf_per_suite.values())
        chronic_per_id = {c: fpr_at_tau(arr, tau) for c, arr in chronic_subsets.items()}
        max_chronic_fpr = max(chronic_per_id.values())

        f1_pass = lb_fake_recall >= F1_RECALL_FLOOR
        f4_pass = max_hdtf_fpr <= F4_FPR_CEIL
        f5_pass = max_chronic_fpr <= F5_FPR_CEIL
        all_pass = f1_pass and f4_pass and f5_pass

        row = {
            "tau": tau,
            "lockbox_real_fpr": lb_real_fpr,
            "lockbox_fake_recall": lb_fake_recall,
            "max_hdtf_fpr": max_hdtf_fpr,
            "max_chronic_fpr": max_chronic_fpr,
            "f1_pass": f1_pass,
            "f4_pass": f4_pass,
            "f5_pass": f5_pass,
            "all_pass": all_pass,
        }
        for s, v in hdtf_per_suite.items():
            row[f"hdtf_{s}_fpr"] = v
        for c, v in chronic_per_id.items():
            row[f"chronic_{c}_fpr"] = v
        rows.append(row)

    sweep_df = pd.DataFrame(rows)
    sweep_path = OUT_DIR / "tau_sweep_bundle_step500.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"Wrote {sweep_path}")

    # ---------------------- Summarize ----------------------
    f1_taus = sweep_df.loc[sweep_df["f1_pass"], "tau"]
    f4_taus = sweep_df.loc[sweep_df["f4_pass"], "tau"]
    f5_taus = sweep_df.loc[sweep_df["f5_pass"], "tau"]
    overlap_taus = sweep_df.loc[sweep_df["all_pass"], "tau"]

    summary = {
        "n_grid": len(sweep_df),
        "tau_min": float(TAU_GRID[0]),
        "tau_max": float(TAU_GRID[-1]),
        "f1_band": (float(f1_taus.min()), float(f1_taus.max())) if len(f1_taus) else None,
        "f4_band": (float(f4_taus.min()), float(f4_taus.max())) if len(f4_taus) else None,
        "f5_band": (float(f5_taus.min()), float(f5_taus.max())) if len(f5_taus) else None,
        "overlap_band": (float(overlap_taus.min()), float(overlap_taus.max())) if len(overlap_taus) else None,
        "n_f1": int(len(f1_taus)),
        "n_f4": int(len(f4_taus)),
        "n_f5": int(len(f5_taus)),
        "n_overlap": int(len(overlap_taus)),
    }

    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Save summary as JSON-like dict
    pd.Series(summary, dtype=object).to_csv(OUT_DIR / "tau_sweep_summary.csv", header=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
