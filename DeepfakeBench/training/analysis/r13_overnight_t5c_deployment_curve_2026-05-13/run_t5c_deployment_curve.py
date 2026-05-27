"""T5C_PERIODIC_STEP3500 deployment τ-tradeoff curve.

Builds a deployment-decision data sheet showing (real_fpr, fake_recall) at
several candidate τ values for T5C step3500, alongside matched-τ rows for
P8A_REFERENCE_STEP5000 and the currently-deployed E2B_TOP_N_STEP3200.

Inputs (video-level, scorer-canonical):
  ../r13_overnight_partial_scorecard_2026-05-13/_reports_cache/
    <suite>_t5c_periodic_step3500_videos_report.csv
    <suite>_p8a_reference_step5000_videos_report.csv
    <suite>_e2b_top_n_step3200_videos_report.csv

Outputs (this folder):
  _t5c_pareto_<date>.csv          T5C τ-sweep (all metrics)
  _comparison_<date>.csv          T5C vs P8A vs E2B at each τ (long form)
  _recommended_tau_<date>.csv     single-row at recommended τ
  _per_identity_lockbox_<date>.csv  identity FPR breakdown at recommended τ
  _results_<date>.json            JSON dump of all above
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent
VIDEO_CACHE = HERE.parent / "r13_overnight_partial_scorecard_2026-05-13" / "_reports_cache"
DATE = "2026-05-13"

CONTRACT_SUITES = [
    "teams_real_all_dev",
    "teams_real_poor_quality_dev",
    "teams_real_lighting_extreme_dev",
    "teams_fake_all_dev",
    "visomaster_enhanced_macro_dev",
    "deeplive_enhanced_dev",
    "teams_real_all_lockbox",
    "teams_fake_all_lockbox",
    "teams_real_dor_dev",
    "teams_real_poor_quality_lockbox",
]

# Anchors of interest
T5C_KEY = "t5c_periodic_step3500"
P8A_KEY = "p8a_reference_step5000"
E2B_KEY = "e2b_top_n_step3200"

CKPTS = [T5C_KEY, P8A_KEY, E2B_KEY]

# Calibrated τ values from the partial scorecard (dev-cal at real_fpr ≤ 0.07
# on teams_real_all_dev, midpoint convention).
TAU_CAL = {
    P8A_KEY: 0.914263,
    E2B_KEY: 0.695595,
    T5C_KEY: 0.821989,
}

# τ grid per task spec (12 user-specified values)
TAU_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.83, 0.85, 0.87, 0.90, 0.92, 0.94, 0.96]


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def load_videos(suite: str, ckpt: str) -> Optional[list[dict]]:
    path = VIDEO_CACHE / f"{suite}_{ckpt}_videos_report.csv"
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "video_id": r["video_id"],
                "label": int(r["label"]),
                "avg_video_prob": float(r["avg_video_prob"]),
            })
    return rows


def parse_identity(video_id: str) -> str:
    """Extract identity from video_id like `dor_shkedi__seq0007__real` or
    `Cam_Test__s32__seg_309.0__real`.
    """
    parts = video_id.split("__")
    if len(parts) >= 2:
        return parts[0]
    return video_id


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def fpr_at_tau(rows: list[dict], tau: float) -> float:
    reals = [r for r in rows if r["label"] == 0]
    if not reals:
        return float("nan")
    return sum(1 for r in reals if r["avg_video_prob"] >= tau) / len(reals)


def recall_at_tau(rows: list[dict], tau: float) -> float:
    fakes = [r for r in rows if r["label"] == 1]
    if not fakes:
        return float("nan")
    return sum(1 for r in fakes if r["avg_video_prob"] >= tau) / len(fakes)


def n_real(rows: list[dict]) -> int:
    return sum(1 for r in rows if r["label"] == 0)


def n_fake(rows: list[dict]) -> int:
    return sum(1 for r in rows if r["label"] == 1)


# -----------------------------------------------------------------------------
# Step 1: τ-sweep on T5C step3500
# -----------------------------------------------------------------------------

def metrics_at_tau(suite_rows: dict, tau: float) -> dict:
    """All Pareto-relevant metrics at a single τ."""
    out = {"tau": tau}
    # Per-suite real FPRs
    out["dev_real_fpr"] = fpr_at_tau(suite_rows["teams_real_all_dev"], tau)
    out["dor_dev_fpr"] = fpr_at_tau(suite_rows["teams_real_dor_dev"], tau)
    out["poor_quality_dev_fpr"] = fpr_at_tau(suite_rows["teams_real_poor_quality_dev"], tau)
    out["lighting_extreme_dev_fpr"] = fpr_at_tau(suite_rows["teams_real_lighting_extreme_dev"], tau)
    out["lockbox_real_fpr"] = fpr_at_tau(suite_rows["teams_real_all_lockbox"], tau)
    out["poor_quality_lockbox_fpr"] = fpr_at_tau(suite_rows["teams_real_poor_quality_lockbox"], tau)
    # Per-suite fake recalls
    out["teams_fake_dev_recall"] = recall_at_tau(suite_rows["teams_fake_all_dev"], tau)
    out["viso_enh_dev_recall"] = recall_at_tau(suite_rows["visomaster_enhanced_macro_dev"], tau)
    out["deeplive_enh_dev_recall"] = recall_at_tau(suite_rows["deeplive_enhanced_dev"], tau)
    out["dev_macro_recall"] = (
        out["teams_fake_dev_recall"]
        + out["viso_enh_dev_recall"]
        + out["deeplive_enh_dev_recall"]
    ) / 3.0
    out["lockbox_fake_recall"] = recall_at_tau(suite_rows["teams_fake_all_lockbox"], tau)
    return out


def load_all_suites(ckpt: str) -> dict:
    """Load video CSVs for one ckpt across all contract suites."""
    suite_rows = {}
    for s in CONTRACT_SUITES:
        suite_rows[s] = load_videos(s, ckpt)
    return suite_rows


# -----------------------------------------------------------------------------
# Step 4: recommended τ selection
# -----------------------------------------------------------------------------

def find_recommended_tau(suite_rows: dict, lb_fpr_ceiling: float = 0.025,
                        tau_max: float = 0.99) -> dict:
    """Find best (lb_fake_recall, dev_macro_recall) at lb_real_fpr ≤ ceiling.

    Search strategy: precise τ from sorted lb_real probs (matching the scorer's
    calibration logic). For ceiling=0.025 with n=1361 reals, max_n_fp = 34.

    Returns the full metric row at that τ.
    """
    v_lb = suite_rows["teams_real_all_lockbox"]
    reals = [r for r in v_lb if r["label"] == 0]
    probs = sorted([r["avg_video_prob"] for r in reals], reverse=True)
    n = len(probs)
    max_n_fp = int(lb_fpr_ceiling * n)
    if max_n_fp >= n:
        tau_match = 0.0
    elif max_n_fp == 0:
        tau_match = max(probs) + 1e-9
    else:
        upper = probs[max_n_fp - 1]
        lower = probs[max_n_fp]
        tau_match = (upper + lower) / 2.0
    return metrics_at_tau(suite_rows, tau_match), tau_match


# -----------------------------------------------------------------------------
# Step 5: per-identity sanity check at recommended τ
# -----------------------------------------------------------------------------

def per_identity_breakdown(suite_rows: dict, suite_name: str, tau: float,
                           threshold_warn: float = 0.30) -> list[dict]:
    """For a single real-suite, decompose FPR by identity at given τ.
    """
    chronic = {"dor_shkedi", "Roy_D", "PC_Generator", "bla_bla_chow", "xiang", "dor"}
    rows = suite_rows[suite_name]
    per_id = defaultdict(lambda: {"n_videos": 0, "n_fp_videos": 0})
    for r in rows:
        if r["label"] != 0:
            continue
        ident = parse_identity(r["video_id"])
        per_id[ident]["n_videos"] += 1
        if r["avg_video_prob"] >= tau:
            per_id[ident]["n_fp_videos"] += 1

    records = []
    for ident, d in per_id.items():
        fpr = d["n_fp_videos"] / d["n_videos"] if d["n_videos"] > 0 else 0.0
        records.append({
            "identity": ident,
            "suite": suite_name,
            "is_chronic_6": ident in chronic,
            "n_videos": d["n_videos"],
            "n_fp_videos": d["n_fp_videos"],
            "fpr": fpr,
            "exceeds_30pct": fpr > threshold_warn,
        })
    return sorted(records, key=lambda r: -r["fpr"])


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> None:
    print("=" * 100)
    print(f"T5C_PERIODIC_STEP3500 DEPLOYMENT τ-TRADEOFF CURVE — {DATE}")
    print("=" * 100)

    # Load all per-ckpt suite rows
    rows_by_ckpt = {ckpt: load_all_suites(ckpt) for ckpt in CKPTS}

    # Print sample sizes for sanity
    s_lb_real = rows_by_ckpt[T5C_KEY]["teams_real_all_lockbox"]
    s_lb_fake = rows_by_ckpt[T5C_KEY]["teams_fake_all_lockbox"]
    s_dev_real = rows_by_ckpt[T5C_KEY]["teams_real_all_dev"]
    s_dor = rows_by_ckpt[T5C_KEY]["teams_real_dor_dev"]
    print(f"\nSample sizes (T5C, video-level): "
          f"dev_real n={n_real(s_dev_real)}, lockbox_real n={n_real(s_lb_real)}, "
          f"lockbox_fake n={n_fake(s_lb_fake)}, dor_dev n={n_real(s_dor)}")
    print(f"Calibrated τ (partial scorecard): "
          f"T5C={TAU_CAL[T5C_KEY]:.4f}, P8A={TAU_CAL[P8A_KEY]:.4f}, "
          f"E2B={TAU_CAL[E2B_KEY]:.4f}")

    # Step 1: τ-sweep on T5C step3500
    t5c_pareto = [metrics_at_tau(rows_by_ckpt[T5C_KEY], tau) for tau in TAU_GRID]
    # Also include the calibrated τ for T5C and the recommended τ later
    t5c_pareto_with_cal = sorted(
        t5c_pareto + [metrics_at_tau(rows_by_ckpt[T5C_KEY], TAU_CAL[T5C_KEY])],
        key=lambda r: r["tau"]
    )

    # Step 3: same τ grid for P8A and E2B
    p8a_pareto = [metrics_at_tau(rows_by_ckpt[P8A_KEY], tau) for tau in TAU_GRID]
    e2b_pareto = [metrics_at_tau(rows_by_ckpt[E2B_KEY], tau) for tau in TAU_GRID]

    # E2B at its OWN calibrated τ — this is the operating point used in production
    e2b_at_cal = metrics_at_tau(rows_by_ckpt[E2B_KEY], TAU_CAL[E2B_KEY])
    p8a_at_cal = metrics_at_tau(rows_by_ckpt[P8A_KEY], TAU_CAL[P8A_KEY])
    t5c_at_cal = metrics_at_tau(rows_by_ckpt[T5C_KEY], TAU_CAL[T5C_KEY])

    # Step 4: recommended τ for T5C — best (lb_fake_recall, dev_macro_recall) at
    # lb_real_fpr ≤ 0.025 (E2B-comparable Pillar 1 ceiling).
    t5c_rec, t5c_rec_tau = find_recommended_tau(rows_by_ckpt[T5C_KEY], lb_fpr_ceiling=0.025)

    # Also: comparison rows at the recommended τ for P8A & E2B
    p8a_at_rec = metrics_at_tau(rows_by_ckpt[P8A_KEY], t5c_rec_tau)
    e2b_at_rec = metrics_at_tau(rows_by_ckpt[E2B_KEY], t5c_rec_tau)

    # Step 5: per-identity sanity check at recommended τ for T5C
    # Lockbox only has 5 distinct identities; also breakdown teams_real_all_dev,
    # poor_quality_dev, lighting_extreme_dev, dor_dev for fuller coverage.
    t5c_per_id_by_suite = {
        s: per_identity_breakdown(rows_by_ckpt[T5C_KEY], s, t5c_rec_tau, threshold_warn=0.30)
        for s in [
            "teams_real_all_lockbox",
            "teams_real_all_dev",
            "teams_real_poor_quality_dev",
            "teams_real_lighting_extreme_dev",
            "teams_real_dor_dev",
            "teams_real_poor_quality_lockbox",
        ]
    }
    # Pool across suites: identity -> aggregate (sum n_videos, sum n_fp_videos)
    pool_per_id = defaultdict(lambda: {"n_videos": 0, "n_fp_videos": 0,
                                       "suites": []})
    chronic = {"dor_shkedi", "Roy_D", "PC_Generator", "bla_bla_chow", "xiang", "dor"}
    for suite_name, recs in t5c_per_id_by_suite.items():
        for rec in recs:
            ident = rec["identity"]
            pool_per_id[ident]["n_videos"] += rec["n_videos"]
            pool_per_id[ident]["n_fp_videos"] += rec["n_fp_videos"]
            pool_per_id[ident]["suites"].append(
                f"{suite_name}={rec['n_fp_videos']}/{rec['n_videos']}"
            )
    t5c_per_id_pooled = []
    for ident, d in pool_per_id.items():
        fpr = d["n_fp_videos"] / d["n_videos"] if d["n_videos"] > 0 else 0.0
        t5c_per_id_pooled.append({
            "identity": ident,
            "is_chronic_6": ident in chronic,
            "n_videos": d["n_videos"],
            "n_fp_videos": d["n_fp_videos"],
            "fpr": fpr,
            "exceeds_30pct": fpr > 0.30,
            "per_suite_breakdown": "; ".join(d["suites"]),
        })
    t5c_per_id_pooled.sort(key=lambda r: -r["fpr"])

    # ---------------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------------
    # Pareto table CSV (T5C-only τ-sweep + cal row)
    with open(HERE / f"_t5c_pareto_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tau", "dev_real_fpr", "lockbox_real_fpr", "dor_dev_fpr",
            "poor_quality_dev_fpr", "lighting_extreme_dev_fpr",
            "poor_quality_lockbox_fpr",
            "teams_fake_dev_recall", "viso_enh_dev_recall",
            "deeplive_enh_dev_recall", "dev_macro_recall",
            "lockbox_fake_recall",
        ])
        for r in t5c_pareto_with_cal:
            w.writerow([
                f"{r['tau']:.4f}",
                f"{r['dev_real_fpr']:.4f}",
                f"{r['lockbox_real_fpr']:.4f}",
                f"{r['dor_dev_fpr']:.4f}",
                f"{r['poor_quality_dev_fpr']:.4f}",
                f"{r['lighting_extreme_dev_fpr']:.4f}",
                f"{r['poor_quality_lockbox_fpr']:.4f}",
                f"{r['teams_fake_dev_recall']:.4f}",
                f"{r['viso_enh_dev_recall']:.4f}",
                f"{r['deeplive_enh_dev_recall']:.4f}",
                f"{r['dev_macro_recall']:.4f}",
                f"{r['lockbox_fake_recall']:.4f}",
            ])

    # Comparison CSV (long form: ckpt × τ)
    with open(HERE / f"_comparison_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ckpt", "tau", "tau_kind",
            "dev_real_fpr", "lockbox_real_fpr", "dor_dev_fpr",
            "dev_macro_recall", "lockbox_fake_recall",
            "teams_fake_dev_recall", "viso_enh_dev_recall",
            "deeplive_enh_dev_recall",
        ])
        # Common τ grid for all 3
        for tau_idx, tau in enumerate(TAU_GRID):
            for ckpt, sweep in [(T5C_KEY, t5c_pareto), (P8A_KEY, p8a_pareto), (E2B_KEY, e2b_pareto)]:
                r = sweep[tau_idx]
                w.writerow([
                    ckpt, f"{tau:.4f}", "grid",
                    f"{r['dev_real_fpr']:.4f}",
                    f"{r['lockbox_real_fpr']:.4f}",
                    f"{r['dor_dev_fpr']:.4f}",
                    f"{r['dev_macro_recall']:.4f}",
                    f"{r['lockbox_fake_recall']:.4f}",
                    f"{r['teams_fake_dev_recall']:.4f}",
                    f"{r['viso_enh_dev_recall']:.4f}",
                    f"{r['deeplive_enh_dev_recall']:.4f}",
                ])
        # Each ckpt at its own calibrated τ
        for ckpt, r in [(T5C_KEY, t5c_at_cal), (P8A_KEY, p8a_at_cal), (E2B_KEY, e2b_at_cal)]:
            w.writerow([
                ckpt, f"{TAU_CAL[ckpt]:.4f}", "calibrated",
                f"{r['dev_real_fpr']:.4f}",
                f"{r['lockbox_real_fpr']:.4f}",
                f"{r['dor_dev_fpr']:.4f}",
                f"{r['dev_macro_recall']:.4f}",
                f"{r['lockbox_fake_recall']:.4f}",
                f"{r['teams_fake_dev_recall']:.4f}",
                f"{r['viso_enh_dev_recall']:.4f}",
                f"{r['deeplive_enh_dev_recall']:.4f}",
            ])
        # All 3 ckpts at recommended τ
        for ckpt, r in [(T5C_KEY, t5c_rec), (P8A_KEY, p8a_at_rec), (E2B_KEY, e2b_at_rec)]:
            w.writerow([
                ckpt, f"{t5c_rec_tau:.4f}", "recommended_t5c",
                f"{r['dev_real_fpr']:.4f}",
                f"{r['lockbox_real_fpr']:.4f}",
                f"{r['dor_dev_fpr']:.4f}",
                f"{r['dev_macro_recall']:.4f}",
                f"{r['lockbox_fake_recall']:.4f}",
                f"{r['teams_fake_dev_recall']:.4f}",
                f"{r['viso_enh_dev_recall']:.4f}",
                f"{r['deeplive_enh_dev_recall']:.4f}",
            ])

    # Recommended τ CSV (single row + comparators)
    with open(HERE / f"_recommended_tau_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ckpt", "tau", "dev_real_fpr", "lockbox_real_fpr", "dor_dev_fpr",
            "dev_macro_recall", "lockbox_fake_recall",
            "teams_fake_dev_recall", "viso_enh_dev_recall",
            "deeplive_enh_dev_recall",
        ])
        for ckpt, r in [(T5C_KEY, t5c_rec), (P8A_KEY, p8a_at_rec), (E2B_KEY, e2b_at_rec)]:
            w.writerow([
                ckpt, f"{t5c_rec_tau:.4f}",
                f"{r['dev_real_fpr']:.4f}",
                f"{r['lockbox_real_fpr']:.4f}",
                f"{r['dor_dev_fpr']:.4f}",
                f"{r['dev_macro_recall']:.4f}",
                f"{r['lockbox_fake_recall']:.4f}",
                f"{r['teams_fake_dev_recall']:.4f}",
                f"{r['viso_enh_dev_recall']:.4f}",
                f"{r['deeplive_enh_dev_recall']:.4f}",
            ])

    # Per-identity CSV — lockbox-only
    with open(HERE / f"_per_identity_lockbox_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "identity", "is_chronic_6", "n_videos", "n_fp_videos", "fpr",
            "exceeds_30pct",
        ])
        for rec in t5c_per_id_by_suite["teams_real_all_lockbox"]:
            w.writerow([
                rec["identity"],
                "Y" if rec["is_chronic_6"] else "",
                rec["n_videos"], rec["n_fp_videos"],
                f"{rec['fpr']:.4f}",
                "Y" if rec["exceeds_30pct"] else "",
            ])

    # Cross-ckpt per-identity comparison (T5C vs P8A vs E2B) at recommended τ
    cross_records = []
    suites_all = [
        "teams_real_all_lockbox", "teams_real_all_dev",
        "teams_real_poor_quality_dev", "teams_real_lighting_extreme_dev",
        "teams_real_dor_dev", "teams_real_poor_quality_lockbox",
    ]
    idents_all = set()
    for s in suites_all:
        for r in rows_by_ckpt[T5C_KEY][s]:
            if r["label"] == 0:
                idents_all.add(parse_identity(r["video_id"]))

    def pool_fpr(ckpt: str, tau: float, ident: str) -> tuple:
        n = 0
        fp = 0
        for s in suites_all:
            for r in rows_by_ckpt[ckpt][s]:
                if r["label"] == 0 and parse_identity(r["video_id"]) == ident:
                    n += 1
                    if r["avg_video_prob"] >= tau:
                        fp += 1
        return (fp, n, fp / n if n > 0 else 0.0)

    for ident in idents_all:
        rec = {"identity": ident, "is_chronic_6": ident in chronic}
        for ckpt, tau, label in [
            (T5C_KEY, t5c_rec_tau, "T5C_at_rec_tau"),
            (P8A_KEY, t5c_rec_tau, "P8A_at_rec_tau"),
            (E2B_KEY, t5c_rec_tau, "E2B_at_rec_tau"),
            (T5C_KEY, TAU_CAL[T5C_KEY], "T5C_at_cal_tau"),
            (P8A_KEY, TAU_CAL[P8A_KEY], "P8A_at_cal_tau"),
            (E2B_KEY, TAU_CAL[E2B_KEY], "E2B_at_cal_tau"),
        ]:
            fp, n, fpr = pool_fpr(ckpt, tau, ident)
            rec[f"{label}_fpr"] = fpr
            rec[f"{label}_fp_n"] = f"{fp}/{n}"
        cross_records.append(rec)
    cross_records.sort(key=lambda r: -r["T5C_at_rec_tau_fpr"])

    with open(HERE / f"_per_identity_cross_ckpt_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "identity", "is_chronic_6",
            "T5C_at_rec_tau_fpr", "T5C_at_rec_tau_fp_n",
            "P8A_at_rec_tau_fpr", "P8A_at_rec_tau_fp_n",
            "E2B_at_rec_tau_fpr", "E2B_at_rec_tau_fp_n",
            "T5C_at_cal_tau_fpr", "T5C_at_cal_tau_fp_n",
            "P8A_at_cal_tau_fpr", "P8A_at_cal_tau_fp_n",
            "E2B_at_cal_tau_fpr", "E2B_at_cal_tau_fp_n",
        ])
        for r in cross_records:
            w.writerow([
                r["identity"], "Y" if r["is_chronic_6"] else "",
                f"{r['T5C_at_rec_tau_fpr']:.4f}", r["T5C_at_rec_tau_fp_n"],
                f"{r['P8A_at_rec_tau_fpr']:.4f}", r["P8A_at_rec_tau_fp_n"],
                f"{r['E2B_at_rec_tau_fpr']:.4f}", r["E2B_at_rec_tau_fp_n"],
                f"{r['T5C_at_cal_tau_fpr']:.4f}", r["T5C_at_cal_tau_fp_n"],
                f"{r['P8A_at_cal_tau_fpr']:.4f}", r["P8A_at_cal_tau_fp_n"],
                f"{r['E2B_at_cal_tau_fpr']:.4f}", r["E2B_at_cal_tau_fp_n"],
            ])

    # Per-identity CSV — pooled across all real suites
    with open(HERE / f"_per_identity_pooled_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "identity", "is_chronic_6", "n_videos_pooled", "n_fp_videos_pooled",
            "fpr_pooled", "exceeds_30pct", "per_suite_breakdown",
        ])
        for rec in t5c_per_id_pooled:
            w.writerow([
                rec["identity"],
                "Y" if rec["is_chronic_6"] else "",
                rec["n_videos"], rec["n_fp_videos"],
                f"{rec['fpr']:.4f}",
                "Y" if rec["exceeds_30pct"] else "",
                rec["per_suite_breakdown"],
            ])

    # JSON dump
    payload = {
        "date": DATE,
        "ckpts": CKPTS,
        "tau_calibrated": TAU_CAL,
        "t5c_pareto": t5c_pareto_with_cal,
        "comparison_grid": {
            T5C_KEY: t5c_pareto,
            P8A_KEY: p8a_pareto,
            E2B_KEY: e2b_pareto,
        },
        "at_each_calibrated_tau": {
            T5C_KEY: t5c_at_cal,
            P8A_KEY: p8a_at_cal,
            E2B_KEY: e2b_at_cal,
        },
        "recommended_tau": {
            "tau": t5c_rec_tau,
            "lb_fpr_ceiling": 0.025,
            T5C_KEY: t5c_rec,
            P8A_KEY: p8a_at_rec,
            E2B_KEY: e2b_at_rec,
        },
        "per_identity_lockbox_t5c_at_rec_tau": t5c_per_id_by_suite["teams_real_all_lockbox"],
        "per_identity_pooled_t5c_at_rec_tau": t5c_per_id_pooled,
        "per_identity_by_suite_t5c_at_rec_tau": t5c_per_id_by_suite,
    }
    (HERE / f"_results_{DATE}.json").write_text(json.dumps(payload, indent=2))

    # ---------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------
    print("\n--- Step 1+2: T5C step3500 Pareto curve ---")
    print(f"{'τ':>7s} {'dev_fpr':>8s} {'lb_fpr':>7s} {'dor_fpr':>8s} "
          f"{'macro_R':>8s} {'lb_R':>7s} {'teams_R':>8s} {'viso_R':>7s} {'dl_R':>7s}")
    for r in t5c_pareto_with_cal:
        tag = "  (cal)" if abs(r["tau"] - TAU_CAL[T5C_KEY]) < 1e-4 else ""
        print(f"{r['tau']:>7.4f} {r['dev_real_fpr']:>8.4f} "
              f"{r['lockbox_real_fpr']:>7.4f} {r['dor_dev_fpr']:>8.4f} "
              f"{r['dev_macro_recall']:>8.4f} {r['lockbox_fake_recall']:>7.4f} "
              f"{r['teams_fake_dev_recall']:>8.4f} {r['viso_enh_dev_recall']:>7.4f} "
              f"{r['deeplive_enh_dev_recall']:>7.4f}{tag}")

    print("\n--- Step 3: Comparison at common τ grid (T5C vs P8A vs E2B) ---")
    for tau_idx, tau in enumerate(TAU_GRID):
        t5c_r = t5c_pareto[tau_idx]
        p8a_r = p8a_pareto[tau_idx]
        e2b_r = e2b_pareto[tau_idx]
        print(f"\nτ = {tau:.4f}")
        print(f"  {'ckpt':<25s} {'lb_fpr':>7s} {'lb_R':>7s} {'macro_R':>8s} {'dor_fpr':>8s}")
        for ckpt, r in [("T5C_STEP3500", t5c_r), ("P8A_STEP5000", p8a_r), ("E2B_STEP3200", e2b_r)]:
            print(f"  {ckpt:<25s} {r['lockbox_real_fpr']:>7.4f} "
                  f"{r['lockbox_fake_recall']:>7.4f} {r['dev_macro_recall']:>8.4f} "
                  f"{r['dor_dev_fpr']:>8.4f}")

    print("\n--- Each ckpt at its OWN calibrated τ ---")
    for ckpt, r in [(T5C_KEY, t5c_at_cal), (P8A_KEY, p8a_at_cal), (E2B_KEY, e2b_at_cal)]:
        print(f"  {ckpt:<35s} τ={TAU_CAL[ckpt]:.4f} "
              f"lb_fpr={r['lockbox_real_fpr']:.4f} "
              f"lb_R={r['lockbox_fake_recall']:.4f} "
              f"macro_R={r['dev_macro_recall']:.4f} "
              f"dor_fpr={r['dor_dev_fpr']:.4f}")

    print(f"\n--- Step 4: Recommended τ for T5C (best at lb_real_fpr ≤ 0.025) ---")
    print(f"  τ_recommended = {t5c_rec_tau:.6f}")
    print(f"  T5C @ τ_rec: lb_fpr={t5c_rec['lockbox_real_fpr']:.4f} "
          f"lb_R={t5c_rec['lockbox_fake_recall']:.4f} "
          f"macro_R={t5c_rec['dev_macro_recall']:.4f} "
          f"dor_fpr={t5c_rec['dor_dev_fpr']:.4f} "
          f"dev_fpr={t5c_rec['dev_real_fpr']:.4f}")
    print(f"  P8A @ τ_rec={t5c_rec_tau:.4f}: lb_fpr={p8a_at_rec['lockbox_real_fpr']:.4f} "
          f"lb_R={p8a_at_rec['lockbox_fake_recall']:.4f} "
          f"macro_R={p8a_at_rec['dev_macro_recall']:.4f}")
    print(f"  E2B @ τ_rec={t5c_rec_tau:.4f}: lb_fpr={e2b_at_rec['lockbox_real_fpr']:.4f} "
          f"lb_R={e2b_at_rec['lockbox_fake_recall']:.4f} "
          f"macro_R={e2b_at_rec['dev_macro_recall']:.4f}")

    print(f"\n--- Step 5: Per-identity T5C @ recommended τ={t5c_rec_tau:.4f} (lockbox reals) ---")
    t5c_per_id_lb = t5c_per_id_by_suite["teams_real_all_lockbox"]
    over_30_lb = [r for r in t5c_per_id_lb if r["exceeds_30pct"]]
    over_20_lb = [r for r in t5c_per_id_lb if r["fpr"] > 0.20 and not r["exceeds_30pct"]]
    print(f"  Identities with FPR > 30%: {len(over_30_lb)}")
    for rec in over_30_lb:
        chronic = " CHRONIC" if rec["is_chronic_6"] else ""
        print(f"    {rec['identity']:<25s} n={rec['n_videos']:3d} "
              f"fp={rec['n_fp_videos']:3d} fpr={rec['fpr']:.4f}{chronic}")
    print(f"  Identities with FPR ∈ (20%, 30%]: {len(over_20_lb)}")
    for rec in over_20_lb:
        chronic = " CHRONIC" if rec["is_chronic_6"] else ""
        print(f"    {rec['identity']:<25s} n={rec['n_videos']:3d} "
              f"fp={rec['n_fp_videos']:3d} fpr={rec['fpr']:.4f}{chronic}")
    print(f"  Lockbox total identities: {len(t5c_per_id_lb)}; "
          f"with any FP: {sum(1 for r in t5c_per_id_lb if r['n_fp_videos'] > 0)}")

    # Per-identity full table on each suite (rich coverage)
    print("\n--- Per-identity T5C @ recommended τ — POOLED across real suites ---")
    print(f"{'identity':<25s} {'n_pooled':>8s} {'fp_pooled':>9s} {'fpr':>8s} {'chronic':>8s}")
    for rec in t5c_per_id_pooled:
        if rec["n_videos"] < 5:
            continue
        chronic = "Y" if rec["is_chronic_6"] else ""
        flag = " >30%" if rec["exceeds_30pct"] else ""
        print(f"  {rec['identity']:<23s} {rec['n_videos']:>8d} {rec['n_fp_videos']:>9d} "
              f"{rec['fpr']:>8.4f} {chronic:>8s}{flag}")
    over_30_pooled = [r for r in t5c_per_id_pooled if r["exceeds_30pct"] and r["n_videos"] >= 5]
    print(f"\n  Pooled identities with FPR > 30% (n>=5): {len(over_30_pooled)}")
    for rec in over_30_pooled:
        print(f"    {rec['identity']:<25s} "
              f"fpr={rec['fpr']:.4f} ({rec['n_fp_videos']}/{rec['n_videos']})")
    if not over_30_pooled:
        print("    [none]")

    # E2B-vs-T5C headline math
    print("\n--- HEADLINE: T5C @ recommended τ vs E2B @ E2B's calibrated τ (production) ---")
    delta_lb_fpr_pp = (t5c_rec["lockbox_real_fpr"] - e2b_at_cal["lockbox_real_fpr"]) * 100
    delta_lb_R_pp = (t5c_rec["lockbox_fake_recall"] - e2b_at_cal["lockbox_fake_recall"]) * 100
    delta_macro_R_pp = (t5c_rec["dev_macro_recall"] - e2b_at_cal["dev_macro_recall"]) * 100
    delta_dor_pp = (t5c_rec["dor_dev_fpr"] - e2b_at_cal["dor_dev_fpr"]) * 100
    print(f"  ΔFPR (lockbox real)     = {delta_lb_fpr_pp:+.2f} pp  "
          f"(T5C {t5c_rec['lockbox_real_fpr']*100:.2f}% vs E2B {e2b_at_cal['lockbox_real_fpr']*100:.2f}%)")
    print(f"  Δrecall (lockbox fake)  = {delta_lb_R_pp:+.2f} pp  "
          f"(T5C {t5c_rec['lockbox_fake_recall']*100:.2f}% vs E2B {e2b_at_cal['lockbox_fake_recall']*100:.2f}%)")
    print(f"  Δmacro_recall (dev)     = {delta_macro_R_pp:+.2f} pp  "
          f"(T5C {t5c_rec['dev_macro_recall']*100:.2f}% vs E2B {e2b_at_cal['dev_macro_recall']*100:.2f}%)")
    print(f"  Δdor_dev_fpr            = {delta_dor_pp:+.2f} pp  "
          f"(T5C {t5c_rec['dor_dev_fpr']*100:.2f}% vs E2B {e2b_at_cal['dor_dev_fpr']*100:.2f}%)")

    print(f"\nWritten outputs:")
    print(f"  {HERE / f'_t5c_pareto_{DATE}.csv'}")
    print(f"  {HERE / f'_comparison_{DATE}.csv'}")
    print(f"  {HERE / f'_recommended_tau_{DATE}.csv'}")
    print(f"  {HERE / f'_per_identity_lockbox_{DATE}.csv'}")
    print(f"  {HERE / f'_results_{DATE}.json'}")


if __name__ == "__main__":
    main()
