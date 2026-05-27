"""T5C_TOP_N_STEP3750 deployment τ-tradeoff curve.

Sister script to ../r13_overnight_t5c_deployment_curve_2026-05-13/run_t5c_deployment_curve.py
that targets the T5C step3750 ckpt instead of step3500.

Inputs (video-level, scorer-canonical):
  ./_reports_cache/<suite>_<ckpt>_videos_report.csv
    Where ckpts are:
      - t5c_top_n_step3750 (TARGET)
      - t5c_periodic_step3500 (sister ckpt comparison)
      - p8a_reference_step5000
      - e2b_top_n_step3200
  Source GCS bucket: t67-t5c-scorecard-2026-05-11/reports/

Outputs (this folder):
  _t5c_step3750_pareto_<date>.csv          T5C step3750 τ-sweep (all metrics)
  _comparison_<date>.csv                   T5C step3750 vs step3500 vs P8A vs E2B at each τ (long form)
  _recommended_tau_<date>.csv              single-row at recommended τ for each ckpt
  _per_identity_lockbox_<date>.csv         identity FPR breakdown at recommended τ
  _per_identity_pooled_<date>.csv          identity FPR pooled across real suites at recommended τ
  _per_identity_cross_ckpt_<date>.csv      per-id FPR for all 4 ckpts at rec-τ and cal-τ
  _results_<date>.json                     JSON dump of all above
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent
VIDEO_CACHE = HERE / "_reports_cache"
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
T5C_3750_KEY = "t5c_top_n_step3750"          # PRIMARY TARGET
T5C_3500_KEY = "t5c_periodic_step3500"       # sister ckpt
P8A_KEY = "p8a_reference_step5000"
E2B_KEY = "e2b_top_n_step3200"

CKPTS = [T5C_3750_KEY, T5C_3500_KEY, P8A_KEY, E2B_KEY]

# Calibrated τ values from t67-t5c-scorecard-2026-05-11 (v3-fix policy with
# stress-binding τ-bump). Values taken from
# analysis/t6_t7_t5c_scorecard_eval_2026-05-12/RESULTS_FACTS_2026-05-12.md §3.2:
#   P8A: 0.9156, E2B: 0.7108, T5C step3500: 0.8309, T5C step3750: 0.8240
TAU_CAL = {
    P8A_KEY: 0.9156,
    E2B_KEY: 0.7108,
    T5C_3500_KEY: 0.8309,
    T5C_3750_KEY: 0.8240,
}

# Task-spec τ grid for T5C step3750
TAU_GRID = [0.50, 0.70, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94]


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


def metrics_at_tau(suite_rows: dict, tau: float) -> dict:
    """All Pareto-relevant metrics at a single τ."""
    out = {"tau": tau}
    out["dev_real_fpr"] = fpr_at_tau(suite_rows["teams_real_all_dev"], tau)
    out["dor_dev_fpr"] = fpr_at_tau(suite_rows["teams_real_dor_dev"], tau)
    out["poor_quality_dev_fpr"] = fpr_at_tau(suite_rows["teams_real_poor_quality_dev"], tau)
    out["lighting_extreme_dev_fpr"] = fpr_at_tau(suite_rows["teams_real_lighting_extreme_dev"], tau)
    out["lockbox_real_fpr"] = fpr_at_tau(suite_rows["teams_real_all_lockbox"], tau)
    out["poor_quality_lockbox_fpr"] = fpr_at_tau(suite_rows["teams_real_poor_quality_lockbox"], tau)
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
    suite_rows = {}
    for s in CONTRACT_SUITES:
        suite_rows[s] = load_videos(s, ckpt)
    return suite_rows


# -----------------------------------------------------------------------------
# recommended τ selection
# -----------------------------------------------------------------------------

def find_recommended_tau(suite_rows: dict, lb_fpr_ceiling: float = 0.025,
                        tau_max: float = 0.99) -> tuple[dict, float]:
    """Find best (lb_fake_recall, dev_macro_recall) at lb_real_fpr ≤ ceiling.

    For ceiling=0.025 with n=1361 reals, max_n_fp = floor(0.025*1361) = 34.
    Returns (metrics_at_tau, tau_match).
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
# Per-identity sanity check at recommended τ
# -----------------------------------------------------------------------------

CHRONIC = {"dor_shkedi", "Roy_D", "PC_Generator", "bla_bla_chow", "xiang", "dor"}


def per_identity_breakdown(suite_rows: dict, suite_name: str, tau: float,
                           threshold_warn: float = 0.30) -> list[dict]:
    rows = suite_rows[suite_name]
    if rows is None:
        return []
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
            "is_chronic_6": ident in CHRONIC,
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
    print(f"T5C_TOP_N_STEP3750 DEPLOYMENT τ-TRADEOFF CURVE — {DATE}")
    print("=" * 100)

    rows_by_ckpt = {ckpt: load_all_suites(ckpt) for ckpt in CKPTS}

    s_lb_real = rows_by_ckpt[T5C_3750_KEY]["teams_real_all_lockbox"]
    s_lb_fake = rows_by_ckpt[T5C_3750_KEY]["teams_fake_all_lockbox"]
    s_dev_real = rows_by_ckpt[T5C_3750_KEY]["teams_real_all_dev"]
    s_dor = rows_by_ckpt[T5C_3750_KEY]["teams_real_dor_dev"]
    print(f"\nSample sizes (T5C step3750, video-level): "
          f"dev_real n={n_real(s_dev_real)}, lockbox_real n={n_real(s_lb_real)}, "
          f"lockbox_fake n={n_fake(s_lb_fake)}, dor_dev n={n_real(s_dor)}")
    print(f"Calibrated τ (t67-t5c-scorecard): "
          f"T5C_3750={TAU_CAL[T5C_3750_KEY]:.4f}, "
          f"T5C_3500={TAU_CAL[T5C_3500_KEY]:.4f}, "
          f"P8A={TAU_CAL[P8A_KEY]:.4f}, "
          f"E2B={TAU_CAL[E2B_KEY]:.4f}")

    # ---- Step 1: τ-sweep on T5C step3750 ----
    t5c_3750_pareto = [metrics_at_tau(rows_by_ckpt[T5C_3750_KEY], tau) for tau in TAU_GRID]
    t5c_3750_pareto_with_cal = sorted(
        t5c_3750_pareto + [metrics_at_tau(rows_by_ckpt[T5C_3750_KEY], TAU_CAL[T5C_3750_KEY])],
        key=lambda r: r["tau"]
    )

    # ---- Step 2: same τ grid for all 4 ckpts ----
    pareto_by_ckpt = {
        ckpt: [metrics_at_tau(rows_by_ckpt[ckpt], tau) for tau in TAU_GRID]
        for ckpt in CKPTS
    }
    at_cal_by_ckpt = {
        ckpt: metrics_at_tau(rows_by_ckpt[ckpt], TAU_CAL[ckpt]) for ckpt in CKPTS
    }

    # ---- Step 3: recommended τ for T5C step3750 ----
    t5c_3750_rec, t5c_3750_rec_tau = find_recommended_tau(
        rows_by_ckpt[T5C_3750_KEY], lb_fpr_ceiling=0.025
    )
    rec_at_rec_tau = {
        ckpt: metrics_at_tau(rows_by_ckpt[ckpt], t5c_3750_rec_tau) for ckpt in CKPTS
    }

    # ---- Step 4: per-identity sanity check at recommended τ ----
    suites_real = [
        "teams_real_all_lockbox",
        "teams_real_all_dev",
        "teams_real_poor_quality_dev",
        "teams_real_lighting_extreme_dev",
        "teams_real_dor_dev",
        "teams_real_poor_quality_lockbox",
    ]
    t5c_3750_per_id_by_suite = {
        s: per_identity_breakdown(rows_by_ckpt[T5C_3750_KEY], s, t5c_3750_rec_tau,
                                  threshold_warn=0.30)
        for s in suites_real
    }
    pool_per_id = defaultdict(lambda: {"n_videos": 0, "n_fp_videos": 0, "suites": []})
    for suite_name, recs in t5c_3750_per_id_by_suite.items():
        for rec in recs:
            ident = rec["identity"]
            pool_per_id[ident]["n_videos"] += rec["n_videos"]
            pool_per_id[ident]["n_fp_videos"] += rec["n_fp_videos"]
            pool_per_id[ident]["suites"].append(
                f"{suite_name}={rec['n_fp_videos']}/{rec['n_videos']}"
            )
    t5c_3750_per_id_pooled = []
    for ident, d in pool_per_id.items():
        fpr = d["n_fp_videos"] / d["n_videos"] if d["n_videos"] > 0 else 0.0
        t5c_3750_per_id_pooled.append({
            "identity": ident,
            "is_chronic_6": ident in CHRONIC,
            "n_videos": d["n_videos"],
            "n_fp_videos": d["n_fp_videos"],
            "fpr": fpr,
            "exceeds_30pct": fpr > 0.30,
            "per_suite_breakdown": "; ".join(d["suites"]),
        })
    t5c_3750_per_id_pooled.sort(key=lambda r: -r["fpr"])

    # ---- Step 5: cross-ckpt per-identity comparison at rec-τ ----
    idents_all = set()
    for s in suites_real:
        for r in rows_by_ckpt[T5C_3750_KEY][s]:
            if r["label"] == 0:
                idents_all.add(parse_identity(r["video_id"]))

    def pool_fpr(ckpt: str, tau: float, ident: str) -> tuple:
        n = 0
        fp = 0
        for s in suites_real:
            for r in rows_by_ckpt[ckpt][s]:
                if r["label"] == 0 and parse_identity(r["video_id"]) == ident:
                    n += 1
                    if r["avg_video_prob"] >= tau:
                        fp += 1
        return (fp, n, fp / n if n > 0 else 0.0)

    cross_records = []
    for ident in idents_all:
        rec = {"identity": ident, "is_chronic_6": ident in CHRONIC}
        for ckpt in [T5C_3750_KEY, T5C_3500_KEY, P8A_KEY, E2B_KEY]:
            fp, n, fpr = pool_fpr(ckpt, t5c_3750_rec_tau, ident)
            rec[f"{ckpt}_at_rec_tau_fpr"] = fpr
            rec[f"{ckpt}_at_rec_tau_fp_n"] = f"{fp}/{n}"
        for ckpt in [T5C_3750_KEY, T5C_3500_KEY, P8A_KEY, E2B_KEY]:
            fp, n, fpr = pool_fpr(ckpt, TAU_CAL[ckpt], ident)
            rec[f"{ckpt}_at_cal_tau_fpr"] = fpr
            rec[f"{ckpt}_at_cal_tau_fp_n"] = f"{fp}/{n}"
        cross_records.append(rec)
    cross_records.sort(key=lambda r: -r[f"{T5C_3750_KEY}_at_rec_tau_fpr"])

    # ---- Write outputs ----
    # Pareto CSV
    with open(HERE / f"_t5c_step3750_pareto_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tau", "dev_real_fpr", "lockbox_real_fpr", "dor_dev_fpr",
            "poor_quality_dev_fpr", "lighting_extreme_dev_fpr",
            "poor_quality_lockbox_fpr",
            "teams_fake_dev_recall", "viso_enh_dev_recall",
            "deeplive_enh_dev_recall", "dev_macro_recall",
            "lockbox_fake_recall",
        ])
        for r in t5c_3750_pareto_with_cal:
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
        for tau_idx, tau in enumerate(TAU_GRID):
            for ckpt in CKPTS:
                r = pareto_by_ckpt[ckpt][tau_idx]
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
        for ckpt in CKPTS:
            r = at_cal_by_ckpt[ckpt]
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
        for ckpt in CKPTS:
            r = rec_at_rec_tau[ckpt]
            w.writerow([
                ckpt, f"{t5c_3750_rec_tau:.4f}", "recommended_t5c_3750",
                f"{r['dev_real_fpr']:.4f}",
                f"{r['lockbox_real_fpr']:.4f}",
                f"{r['dor_dev_fpr']:.4f}",
                f"{r['dev_macro_recall']:.4f}",
                f"{r['lockbox_fake_recall']:.4f}",
                f"{r['teams_fake_dev_recall']:.4f}",
                f"{r['viso_enh_dev_recall']:.4f}",
                f"{r['deeplive_enh_dev_recall']:.4f}",
            ])

    # Recommended τ CSV
    with open(HERE / f"_recommended_tau_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ckpt", "tau", "dev_real_fpr", "lockbox_real_fpr", "dor_dev_fpr",
            "dev_macro_recall", "lockbox_fake_recall",
            "teams_fake_dev_recall", "viso_enh_dev_recall",
            "deeplive_enh_dev_recall",
        ])
        for ckpt in CKPTS:
            r = rec_at_rec_tau[ckpt]
            w.writerow([
                ckpt, f"{t5c_3750_rec_tau:.4f}",
                f"{r['dev_real_fpr']:.4f}",
                f"{r['lockbox_real_fpr']:.4f}",
                f"{r['dor_dev_fpr']:.4f}",
                f"{r['dev_macro_recall']:.4f}",
                f"{r['lockbox_fake_recall']:.4f}",
                f"{r['teams_fake_dev_recall']:.4f}",
                f"{r['viso_enh_dev_recall']:.4f}",
                f"{r['deeplive_enh_dev_recall']:.4f}",
            ])

    # Per-identity lockbox CSV
    with open(HERE / f"_per_identity_lockbox_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "identity", "is_chronic_6", "n_videos", "n_fp_videos", "fpr",
            "exceeds_30pct",
        ])
        for rec in t5c_3750_per_id_by_suite["teams_real_all_lockbox"]:
            w.writerow([
                rec["identity"],
                "Y" if rec["is_chronic_6"] else "",
                rec["n_videos"], rec["n_fp_videos"],
                f"{rec['fpr']:.4f}",
                "Y" if rec["exceeds_30pct"] else "",
            ])

    # Per-identity pooled CSV
    with open(HERE / f"_per_identity_pooled_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "identity", "is_chronic_6", "n_videos_pooled", "n_fp_videos_pooled",
            "fpr_pooled", "exceeds_30pct", "per_suite_breakdown",
        ])
        for rec in t5c_3750_per_id_pooled:
            w.writerow([
                rec["identity"],
                "Y" if rec["is_chronic_6"] else "",
                rec["n_videos"], rec["n_fp_videos"],
                f"{rec['fpr']:.4f}",
                "Y" if rec["exceeds_30pct"] else "",
                rec["per_suite_breakdown"],
            ])

    # Cross-ckpt per-identity CSV (includes step3500 as 4th column)
    with open(HERE / f"_per_identity_cross_ckpt_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["identity", "is_chronic_6"]
        for ckpt in CKPTS:
            header += [f"{ckpt}_at_rec_tau_fpr", f"{ckpt}_at_rec_tau_fp_n"]
        for ckpt in CKPTS:
            header += [f"{ckpt}_at_cal_tau_fpr", f"{ckpt}_at_cal_tau_fp_n"]
        w.writerow(header)
        for r in cross_records:
            row = [r["identity"], "Y" if r["is_chronic_6"] else ""]
            for ckpt in CKPTS:
                row += [f"{r[f'{ckpt}_at_rec_tau_fpr']:.4f}",
                       r[f"{ckpt}_at_rec_tau_fp_n"]]
            for ckpt in CKPTS:
                row += [f"{r[f'{ckpt}_at_cal_tau_fpr']:.4f}",
                       r[f"{ckpt}_at_cal_tau_fp_n"]]
            w.writerow(row)

    # JSON dump
    payload = {
        "date": DATE,
        "ckpts": CKPTS,
        "tau_calibrated": TAU_CAL,
        "t5c_3750_pareto": t5c_3750_pareto_with_cal,
        "comparison_grid": {ckpt: pareto_by_ckpt[ckpt] for ckpt in CKPTS},
        "at_each_calibrated_tau": at_cal_by_ckpt,
        "recommended_tau": {
            "tau": t5c_3750_rec_tau,
            "lb_fpr_ceiling": 0.025,
            **{ckpt: rec_at_rec_tau[ckpt] for ckpt in CKPTS},
        },
        "per_identity_lockbox_t5c_3750_at_rec_tau":
            t5c_3750_per_id_by_suite["teams_real_all_lockbox"],
        "per_identity_pooled_t5c_3750_at_rec_tau": t5c_3750_per_id_pooled,
        "per_identity_by_suite_t5c_3750_at_rec_tau": t5c_3750_per_id_by_suite,
    }
    (HERE / f"_results_{DATE}.json").write_text(json.dumps(payload, indent=2))

    # ---- Print summary ----
    print("\n--- Step 1: T5C step3750 Pareto curve ---")
    print(f"{'τ':>7s} {'dev_fpr':>8s} {'lb_fpr':>7s} {'dor_fpr':>8s} "
          f"{'macro_R':>8s} {'lb_R':>7s} {'teams_R':>8s} {'viso_R':>7s} {'dl_R':>7s}")
    for r in t5c_3750_pareto_with_cal:
        tag = "  (cal)" if abs(r["tau"] - TAU_CAL[T5C_3750_KEY]) < 1e-4 else ""
        print(f"{r['tau']:>7.4f} {r['dev_real_fpr']:>8.4f} "
              f"{r['lockbox_real_fpr']:>7.4f} {r['dor_dev_fpr']:>8.4f} "
              f"{r['dev_macro_recall']:>8.4f} {r['lockbox_fake_recall']:>7.4f} "
              f"{r['teams_fake_dev_recall']:>8.4f} {r['viso_enh_dev_recall']:>7.4f} "
              f"{r['deeplive_enh_dev_recall']:>7.4f}{tag}")

    print("\n--- Step 2: Each ckpt at its OWN calibrated τ (operational point) ---")
    for ckpt in CKPTS:
        r = at_cal_by_ckpt[ckpt]
        print(f"  {ckpt:<30s} τ={TAU_CAL[ckpt]:.4f} "
              f"lb_fpr={r['lockbox_real_fpr']:.4f} "
              f"lb_R={r['lockbox_fake_recall']:.4f} "
              f"macro_R={r['dev_macro_recall']:.4f} "
              f"dor_fpr={r['dor_dev_fpr']:.4f}")

    print(f"\n--- Step 3: Recommended τ for T5C step3750 (best at lb_real_fpr ≤ 0.025) ---")
    print(f"  τ_recommended = {t5c_3750_rec_tau:.6f}")
    for ckpt in CKPTS:
        r = rec_at_rec_tau[ckpt]
        print(f"  {ckpt:<30s} @ τ_rec: lb_fpr={r['lockbox_real_fpr']:.4f} "
              f"lb_R={r['lockbox_fake_recall']:.4f} "
              f"macro_R={r['dev_macro_recall']:.4f} "
              f"dor_fpr={r['dor_dev_fpr']:.4f}")

    print(f"\n--- Step 4: Per-identity T5C step3750 @ rec τ={t5c_3750_rec_tau:.4f} (lockbox reals) ---")
    t5c_lb = t5c_3750_per_id_by_suite["teams_real_all_lockbox"]
    over_30_lb = [r for r in t5c_lb if r["exceeds_30pct"]]
    over_20_lb = [r for r in t5c_lb if r["fpr"] > 0.20 and not r["exceeds_30pct"]]
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
    print(f"  Lockbox total identities: {len(t5c_lb)}; "
          f"with any FP: {sum(1 for r in t5c_lb if r['n_fp_videos'] > 0)}")

    print("\n--- Per-identity T5C step3750 @ rec τ — POOLED across real suites ---")
    print(f"{'identity':<25s} {'n_pooled':>8s} {'fp_pooled':>9s} {'fpr':>8s} {'chronic':>8s}")
    for rec in t5c_3750_per_id_pooled:
        if rec["n_videos"] < 5:
            continue
        chronic = "Y" if rec["is_chronic_6"] else ""
        flag = " >30%" if rec["exceeds_30pct"] else ""
        print(f"  {rec['identity']:<23s} {rec['n_videos']:>8d} {rec['n_fp_videos']:>9d} "
              f"{rec['fpr']:>8.4f} {chronic:>8s}{flag}")
    over_30_pooled = [r for r in t5c_3750_per_id_pooled
                     if r["exceeds_30pct"] and r["n_videos"] >= 5]
    print(f"\n  Pooled identities with FPR > 30% (n>=5): {len(over_30_pooled)}")
    for rec in over_30_pooled:
        print(f"    {rec['identity']:<25s} "
              f"fpr={rec['fpr']:.4f} ({rec['n_fp_videos']}/{rec['n_videos']})")
    if not over_30_pooled:
        print("    [none]")

    # E2B-vs-T5C-step3750 headline math
    print("\n--- HEADLINE: T5C step3750 @ rec-τ vs each comparator @ its own cal-τ ---")
    base = rec_at_rec_tau[T5C_3750_KEY]
    print(f"  Base: T5C step3750 @ τ={t5c_3750_rec_tau:.4f}: "
          f"lb_fpr={base['lockbox_real_fpr']:.4f} "
          f"lb_R={base['lockbox_fake_recall']:.4f} "
          f"macro_R={base['dev_macro_recall']:.4f} "
          f"dor_fpr={base['dor_dev_fpr']:.4f}")
    for comp_ckpt in [T5C_3500_KEY, P8A_KEY, E2B_KEY]:
        c = at_cal_by_ckpt[comp_ckpt]
        d_lb_fpr = (base["lockbox_real_fpr"] - c["lockbox_real_fpr"]) * 100
        d_lb_R = (base["lockbox_fake_recall"] - c["lockbox_fake_recall"]) * 100
        d_macro = (base["dev_macro_recall"] - c["dev_macro_recall"]) * 100
        d_dor = (base["dor_dev_fpr"] - c["dor_dev_fpr"]) * 100
        print(f"\n  vs {comp_ckpt} @ τ={TAU_CAL[comp_ckpt]:.4f}:")
        print(f"    ΔFPR (lockbox real)    = {d_lb_fpr:+.2f} pp")
        print(f"    Δrecall (lockbox fake) = {d_lb_R:+.2f} pp")
        print(f"    Δmacro_recall (dev)    = {d_macro:+.2f} pp")
        print(f"    Δdor_dev_fpr           = {d_dor:+.2f} pp")

    print(f"\nWritten outputs:")
    print(f"  {HERE / f'_t5c_step3750_pareto_{DATE}.csv'}")
    print(f"  {HERE / f'_comparison_{DATE}.csv'}")
    print(f"  {HERE / f'_recommended_tau_{DATE}.csv'}")
    print(f"  {HERE / f'_per_identity_lockbox_{DATE}.csv'}")
    print(f"  {HERE / f'_per_identity_pooled_{DATE}.csv'}")
    print(f"  {HERE / f'_per_identity_cross_ckpt_{DATE}.csv'}")
    print(f"  {HERE / f'_results_{DATE}.json'}")


if __name__ == "__main__":
    main()
