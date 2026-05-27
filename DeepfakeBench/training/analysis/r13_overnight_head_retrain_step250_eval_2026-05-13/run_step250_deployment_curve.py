"""Slot-1 head-retrain step250 deployment τ-tradeoff curve.

Builds a deployment-decision data sheet showing (real_fpr, fake_recall) at
several candidate τ values for step250 (W&B run `fz84lq5k`), alongside
matched-τ rows for P8A_REFERENCE_STEP5000, T5C_PERIODIC_STEP3500, and the
currently-deployed E2B_TOP_N_STEP3200.

Methodology mirrors `analysis/r13_overnight_t5c_deployment_curve_2026-05-13/
run_t5c_deployment_curve.py`. Key adaptations:

  - step250 video-level CSVs come from `./outputs/scores_step250_<suite>.csv`
    (this folder; per-frame). Aggregated to per-video `avg_video_prob`.
  - P8A / T5C / E2B come from the partial-scorecard cache (per-video).

Inputs:
  ./outputs/scores_step250_<suite>.csv (per-frame)
  ../r13_overnight_partial_scorecard_2026-05-13/_reports_cache/<suite>_<ckpt>_videos_report.csv

Outputs (this folder):
  _step250_pareto_<date>.csv
  _comparison_<date>.csv
  _recommended_tau_<date>.csv
  _per_identity_lockbox_<date>.csv
  _per_identity_pooled_<date>.csv
  _per_identity_cross_ckpt_<date>.csv
  _chikara_context_<date>.csv
  _results_<date>.json
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent
STEP250_DIR = HERE / "outputs"
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
]

STEP250_KEY = "step250"
P8A_KEY = "p8a_reference_step5000"
E2B_KEY = "e2b_top_n_step3200"
T5C_KEY = "t5c_periodic_step3500"

CKPTS = [STEP250_KEY, P8A_KEY, E2B_KEY, T5C_KEY]

# Calibrated τ from partial scorecard. step250's τ comes from its own dev FPR
# calibration (target real_fpr ≤ 0.07 on teams_real_all_dev).
TAU_CAL = {
    P8A_KEY: 0.914263,
    E2B_KEY: 0.695595,
    T5C_KEY: 0.821989,
    # step250 cal-τ computed in main() once dev_real_all is loaded.
}

# τ grid per task spec
TAU_GRID = [0.50, 0.60, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.90]


# -----------------------------------------------------------------------------
# I/O — load step250 (per-frame -> per-video) and anchors (per-video).
# -----------------------------------------------------------------------------

def load_step250_videos(suite: str) -> Optional[list[dict]]:
    """step250 CSVs are per-frame in ./outputs/. Aggregate to per-video.

    The dev/lockbox suites have already-scored CSVs from `run_step250_eval.py`:
        scores_step250_lockbox_all.csv    -> teams_real_all_lockbox
        scores_step250_dev_real_all.csv   -> teams_real_all_dev
    Others come from score_step250_remaining_suites.py:
        scores_step250_<suite>.csv
    """
    name_map = {
        "teams_real_all_lockbox": "scores_step250_lockbox_all.csv",
        "teams_real_all_dev":     "scores_step250_dev_real_all.csv",
    }
    fname = name_map.get(suite, f"scores_step250_{suite}.csv")
    path = STEP250_DIR / fname
    if not path.exists():
        return None
    # Aggregate per-frame -> per-video avg prob
    by_v: dict[str, list[dict]] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            vid = r["video_id"]
            by_v.setdefault(vid, []).append({
                "label": int(r["label"]),
                "avg_video_prob": float(r["prob_fake"]),
                "identity_key": r.get("identity_key", ""),
            })
    rows = []
    for vid, lst in by_v.items():
        probs = [r["avg_video_prob"] for r in lst]
        rows.append({
            "video_id": vid,
            "label": lst[0]["label"],
            "avg_video_prob": float(np.mean(probs)),
            "identity_key": lst[0]["identity_key"],
        })
    return rows


def load_anchor_videos(suite: str, ckpt: str) -> Optional[list[dict]]:
    """Anchor ckpts: per-video CSVs from partial-scorecard cache."""
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
                # video_id like `Cam_Test__s32__seg_308.0__real` or `dor_shkedi__seqXXXX__real`;
                # both have identity_key at the front. Use `parse_identity` for cross-CSV parity.
            })
    return rows


def parse_identity(video_id: str) -> str:
    """Identity = everything before the first `__seg_` or `__seq` marker.

    Differs from the T5C deployment-curve harness's first-token split (which
    collapses `Q__s6__seg_43.0__real` to `Q`). This identity_key matches the
    manifest's `identity_key` field (e.g. `Q__s6`, `PC_Generator__s22`).
    Falls back to first `__`-split token if neither marker is present.
    """
    for sep in ("__seq", "__seg_"):
        if sep in video_id:
            return video_id.split(sep)[0]
    parts = video_id.split("__")
    if len(parts) >= 2:
        return parts[0]
    return video_id


def attach_identity_keys(rows: list[dict]) -> None:
    """If rows lack `identity_key` (anchor CSVs), parse from video_id."""
    for r in rows:
        if "identity_key" not in r or not r["identity_key"]:
            r["identity_key"] = parse_identity(r["video_id"])


def load_all_suites(ckpt_key: str) -> dict[str, list[dict]]:
    """Load all 9 contract suites for one ckpt. Includes identity_key on each row."""
    suite_rows: dict[str, list[dict]] = {}
    for s in CONTRACT_SUITES:
        if ckpt_key == STEP250_KEY:
            rows = load_step250_videos(s)
        else:
            rows = load_anchor_videos(s, ckpt_key)
        if rows is None:
            suite_rows[s] = None
            continue
        attach_identity_keys(rows)
        suite_rows[s] = rows
    return suite_rows


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def fpr_at_tau(rows: list[dict], tau: float) -> float:
    if rows is None:
        return float("nan")
    reals = [r for r in rows if r["label"] == 0]
    if not reals:
        return float("nan")
    return sum(1 for r in reals if r["avg_video_prob"] >= tau) / len(reals)


def recall_at_tau(rows: list[dict], tau: float) -> float:
    if rows is None:
        return float("nan")
    fakes = [r for r in rows if r["label"] == 1]
    if not fakes:
        return float("nan")
    return sum(1 for r in fakes if r["avg_video_prob"] >= tau) / len(fakes)


def n_real(rows) -> int:
    if rows is None: return 0
    return sum(1 for r in rows if r["label"] == 0)


def n_fake(rows) -> int:
    if rows is None: return 0
    return sum(1 for r in rows if r["label"] == 1)


def metrics_at_tau(suite_rows: dict, tau: float) -> dict:
    out = {"tau": tau}
    out["dev_real_fpr"] = fpr_at_tau(suite_rows.get("teams_real_all_dev"), tau)
    out["dor_dev_fpr"] = fpr_at_tau(suite_rows.get("teams_real_dor_dev"), tau)
    out["poor_quality_dev_fpr"] = fpr_at_tau(suite_rows.get("teams_real_poor_quality_dev"), tau)
    out["lighting_extreme_dev_fpr"] = fpr_at_tau(suite_rows.get("teams_real_lighting_extreme_dev"), tau)
    out["lockbox_real_fpr"] = fpr_at_tau(suite_rows.get("teams_real_all_lockbox"), tau)
    out["teams_fake_dev_recall"] = recall_at_tau(suite_rows.get("teams_fake_all_dev"), tau)
    out["viso_enh_dev_recall"] = recall_at_tau(suite_rows.get("visomaster_enhanced_macro_dev"), tau)
    out["deeplive_enh_dev_recall"] = recall_at_tau(suite_rows.get("deeplive_enhanced_dev"), tau)
    out["lockbox_fake_recall"] = recall_at_tau(suite_rows.get("teams_fake_all_lockbox"), tau)
    parts = [out["teams_fake_dev_recall"], out["viso_enh_dev_recall"], out["deeplive_enh_dev_recall"]]
    parts = [p for p in parts if not (isinstance(p, float) and (p != p))]  # filter NaN
    out["dev_macro_recall"] = float(np.mean(parts)) if parts else float("nan")
    return out


# -----------------------------------------------------------------------------
# τ calibration (smallest τ such that dev_real_fpr ≤ target).
# -----------------------------------------------------------------------------

def calibrate_tau_dev(suite_rows: dict, target_fpr: float = 0.07) -> float:
    rows = suite_rows.get("teams_real_all_dev")
    if rows is None:
        return 0.5
    reals = [r for r in rows if r["label"] == 0]
    probs = sorted([r["avg_video_prob"] for r in reals], reverse=True)
    n = len(probs)
    max_n_fp = int(target_fpr * n)
    if max_n_fp >= n:
        return 0.0
    if max_n_fp == 0:
        return max(probs) + 1e-9
    upper = probs[max_n_fp - 1]
    lower = probs[max_n_fp]
    return (upper + lower) / 2.0


def find_recommended_tau(suite_rows: dict, lb_fpr_ceiling: float = 0.025) -> tuple[dict, float]:
    """τ that puts lockbox_real_fpr at the ceiling."""
    rows = suite_rows.get("teams_real_all_lockbox")
    reals = [r for r in rows if r["label"] == 0]
    probs = sorted([r["avg_video_prob"] for r in reals], reverse=True)
    n = len(probs)
    max_n_fp = int(lb_fpr_ceiling * n)
    if max_n_fp >= n:
        tau = 0.0
    elif max_n_fp == 0:
        tau = max(probs) + 1e-9
    else:
        upper = probs[max_n_fp - 1]
        lower = probs[max_n_fp]
        tau = (upper + lower) / 2.0
    return metrics_at_tau(suite_rows, tau), tau


# -----------------------------------------------------------------------------
# Per-identity decomposition
# -----------------------------------------------------------------------------

def per_identity_breakdown(suite_rows: dict, suite: str, tau: float) -> list[dict]:
    chronic = {"dor_shkedi", "Roy_D", "PC_Generator", "bla_bla_chow", "xiang", "dor"}
    rows = suite_rows.get(suite)
    if rows is None:
        return []
    per_id = defaultdict(lambda: {"n_videos": 0, "n_fp_videos": 0})
    for r in rows:
        if r["label"] != 0:
            continue
        ident = r.get("identity_key") or parse_identity(r["video_id"])
        per_id[ident]["n_videos"] += 1
        if r["avg_video_prob"] >= tau:
            per_id[ident]["n_fp_videos"] += 1
    recs = []
    for ident, d in per_id.items():
        fpr = d["n_fp_videos"] / d["n_videos"] if d["n_videos"] > 0 else 0.0
        # Strip suffixes like __s22 from identity for chronic-6 match.
        base = ident.split("__")[0] if "__" in ident else ident
        is_chronic = base in chronic
        recs.append({
            "identity": ident,
            "suite": suite,
            "is_chronic_6": is_chronic,
            "n_videos": d["n_videos"],
            "n_fp_videos": d["n_fp_videos"],
            "fpr": fpr,
            "exceeds_30pct": fpr > 0.30,
        })
    return sorted(recs, key=lambda r: -r["fpr"])


def pool_per_identity(suite_rows: dict, tau: float,
                      suites: list[str]) -> list[dict]:
    chronic = {"dor_shkedi", "Roy_D", "PC_Generator", "bla_bla_chow", "xiang", "dor"}
    pool = defaultdict(lambda: {"n_videos": 0, "n_fp_videos": 0, "per_suite": []})
    for s in suites:
        rows = suite_rows.get(s)
        if rows is None:
            continue
        per_suite_for_id = defaultdict(lambda: [0, 0])  # ident -> [n, fp]
        for r in rows:
            if r["label"] != 0:
                continue
            ident = r.get("identity_key") or parse_identity(r["video_id"])
            per_suite_for_id[ident][0] += 1
            if r["avg_video_prob"] >= tau:
                per_suite_for_id[ident][1] += 1
        for ident, (n_v, n_fp) in per_suite_for_id.items():
            pool[ident]["n_videos"] += n_v
            pool[ident]["n_fp_videos"] += n_fp
            pool[ident]["per_suite"].append(f"{s}={n_fp}/{n_v}")
    out = []
    for ident, d in pool.items():
        fpr = d["n_fp_videos"] / d["n_videos"] if d["n_videos"] > 0 else 0.0
        base = ident.split("__")[0] if "__" in ident else ident
        out.append({
            "identity": ident,
            "is_chronic_6": base in chronic,
            "n_videos": d["n_videos"],
            "n_fp_videos": d["n_fp_videos"],
            "fpr": fpr,
            "exceeds_30pct": fpr > 0.30,
            "per_suite_breakdown": "; ".join(d["per_suite"]),
        })
    return sorted(out, key=lambda r: -r["fpr"])


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> None:
    print("=" * 100)
    print(f"STEP250 DEPLOYMENT τ-TRADEOFF CURVE — {DATE}")
    print("=" * 100)

    # Load suites per ckpt
    rows_by_ckpt = {ckpt: load_all_suites(ckpt) for ckpt in CKPTS}

    # Sample sizes
    print(f"\nSample sizes (step250, video-level):")
    for s in CONTRACT_SUITES:
        rs = rows_by_ckpt[STEP250_KEY].get(s)
        if rs is None:
            print(f"  {s:40s} MISSING")
        else:
            print(f"  {s:40s} n_real={n_real(rs):4d}  n_fake={n_fake(rs):4d}  total={len(rs):4d}")

    # Calibrate step250's τ on its own dev (target 0.07)
    tau_step250_cal = calibrate_tau_dev(rows_by_ckpt[STEP250_KEY], 0.07)
    TAU_CAL[STEP250_KEY] = tau_step250_cal
    print(f"\nCalibrated τ:")
    for ckpt in CKPTS:
        print(f"  {ckpt:40s} τ={TAU_CAL[ckpt]:.6f}")

    # Step 1: τ-sweep on step250 (grid + cal + recommended)
    step250_pareto = [metrics_at_tau(rows_by_ckpt[STEP250_KEY], tau) for tau in TAU_GRID]
    step250_at_cal = metrics_at_tau(rows_by_ckpt[STEP250_KEY], tau_step250_cal)

    # Step 4: recommended τ (lb_real_fpr ≤ 0.025)
    step250_rec, step250_rec_tau = find_recommended_tau(rows_by_ckpt[STEP250_KEY], lb_fpr_ceiling=0.025)
    print(f"\nRecommended τ for step250 (lb_real_fpr ≤ 0.025): τ={step250_rec_tau:.6f}")
    print(f"  step250 @ rec-τ: lb_real_fpr={step250_rec['lockbox_real_fpr']:.4f} "
          f"lb_fake_R={step250_rec['lockbox_fake_recall']:.4f} "
          f"macro_R={step250_rec['dev_macro_recall']:.4f} "
          f"dor_fpr={step250_rec['dor_dev_fpr']:.4f}")

    # Comparison rows at rec-τ for other ckpts
    p8a_at_rec = metrics_at_tau(rows_by_ckpt[P8A_KEY], step250_rec_tau)
    e2b_at_rec = metrics_at_tau(rows_by_ckpt[E2B_KEY], step250_rec_tau)
    t5c_at_rec = metrics_at_tau(rows_by_ckpt[T5C_KEY], step250_rec_tau)

    # Each ckpt at its OWN cal-τ
    step250_at_cal_tau = metrics_at_tau(rows_by_ckpt[STEP250_KEY], tau_step250_cal)
    p8a_at_cal = metrics_at_tau(rows_by_ckpt[P8A_KEY], TAU_CAL[P8A_KEY])
    e2b_at_cal = metrics_at_tau(rows_by_ckpt[E2B_KEY], TAU_CAL[E2B_KEY])
    t5c_at_cal = metrics_at_tau(rows_by_ckpt[T5C_KEY], TAU_CAL[T5C_KEY])

    # Step 5: per-identity decomposition at recommended τ for step250
    per_id_by_suite = {
        s: per_identity_breakdown(rows_by_ckpt[STEP250_KEY], s, step250_rec_tau)
        for s in [
            "teams_real_all_lockbox",
            "teams_real_all_dev",
            "teams_real_poor_quality_dev",
            "teams_real_lighting_extreme_dev",
            "teams_real_dor_dev",
        ]
    }
    pool_suites = [
        "teams_real_all_lockbox", "teams_real_all_dev",
        "teams_real_poor_quality_dev", "teams_real_lighting_extreme_dev",
        "teams_real_dor_dev",
    ]
    per_id_pooled = pool_per_identity(rows_by_ckpt[STEP250_KEY], step250_rec_tau, pool_suites)

    # Roy_D pooled FPR — the KEY metric
    roy_d_pooled = next((r for r in per_id_pooled if r["identity"] == "Roy_D"), None)
    print(f"\nRoy_D pooled FPR @ step250 rec-τ: ", end="")
    if roy_d_pooled:
        print(f"{roy_d_pooled['fpr']:.4f} ({roy_d_pooled['n_fp_videos']}/{roy_d_pooled['n_videos']})")
        print(f"  per-suite: {roy_d_pooled['per_suite_breakdown']}")
    else:
        print("MISSING")

    # Cross-ckpt per-identity at rec-τ
    cross_records = []
    idents_all = set()
    for s in pool_suites:
        rs = rows_by_ckpt[STEP250_KEY].get(s)
        if rs is None: continue
        for r in rs:
            if r["label"] == 0:
                idents_all.add(r.get("identity_key") or parse_identity(r["video_id"]))

    def pool_fpr_for_ckpt(ckpt: str, tau: float, ident: str) -> tuple:
        n = 0; fp = 0
        for s in pool_suites:
            rs = rows_by_ckpt[ckpt].get(s)
            if rs is None: continue
            for r in rs:
                if r["label"] != 0: continue
                r_ident = r.get("identity_key") or parse_identity(r["video_id"])
                if r_ident == ident:
                    n += 1
                    if r["avg_video_prob"] >= tau:
                        fp += 1
        return (fp, n, fp / n if n > 0 else 0.0)

    chronic = {"dor_shkedi", "Roy_D", "PC_Generator", "bla_bla_chow", "xiang", "dor"}
    for ident in sorted(idents_all):
        base = ident.split("__")[0] if "__" in ident else ident
        rec = {"identity": ident, "is_chronic_6": base in chronic}
        for label, ckpt, tau in [
            ("step250_rec",  STEP250_KEY, step250_rec_tau),
            ("p8a_rec",      P8A_KEY,     step250_rec_tau),
            ("e2b_rec",      E2B_KEY,     step250_rec_tau),
            ("t5c_rec",      T5C_KEY,     step250_rec_tau),
            ("step250_cal",  STEP250_KEY, tau_step250_cal),
            ("p8a_cal",      P8A_KEY,     TAU_CAL[P8A_KEY]),
            ("e2b_cal",      E2B_KEY,     TAU_CAL[E2B_KEY]),
            ("t5c_cal",      T5C_KEY,     TAU_CAL[T5C_KEY]),
        ]:
            fp, n, fpr = pool_fpr_for_ckpt(ckpt, tau, ident)
            rec[f"{label}_fpr"] = fpr
            rec[f"{label}_fp_n"] = f"{fp}/{n}"
        cross_records.append(rec)
    cross_records.sort(key=lambda r: -r["step250_rec_fpr"])

    # Chikara_Takahashi context: lockbox-only and dev cells at multiple ckpts.
    chikara_context = []
    for ckpt in CKPTS:
        # Use rec-τ for step250 and the matched-τ for others (also at each own cal)
        rs = rows_by_ckpt[ckpt].get("teams_real_all_lockbox")
        if rs is None: continue
        chikara_videos = [r for r in rs if (r.get("identity_key") or parse_identity(r["video_id"])).startswith("Chikara")]
        n_v = len(chikara_videos)
        probs = sorted([r["avg_video_prob"] for r in chikara_videos], reverse=True)
        for label, tau in [("rec", step250_rec_tau), ("cal", TAU_CAL[ckpt])]:
            n_fp = sum(1 for r in chikara_videos if r["avg_video_prob"] >= tau)
            chikara_context.append({
                "ckpt": ckpt,
                "tau_label": label,
                "tau": tau,
                "suite": "teams_real_all_lockbox",
                "n_videos": n_v,
                "n_fp": n_fp,
                "fpr": n_fp / n_v if n_v > 0 else 0.0,
                "prob_max": probs[0] if probs else float("nan"),
                "prob_p90": probs[max(0, int(0.10 * n_v))] if probs else float("nan"),
                "prob_p50": probs[n_v // 2] if probs else float("nan"),
                "prob_min": probs[-1] if probs else float("nan"),
            })

    # -------- Write outputs --------
    # Pareto (step250 only)
    with open(HERE / f"_step250_pareto_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tau", "dev_real_fpr", "lockbox_real_fpr", "dor_dev_fpr",
            "poor_quality_dev_fpr", "lighting_extreme_dev_fpr",
            "teams_fake_dev_recall", "viso_enh_dev_recall",
            "deeplive_enh_dev_recall", "dev_macro_recall", "lockbox_fake_recall",
        ])
        rows_to_write = [metrics_at_tau(rows_by_ckpt[STEP250_KEY], t) for t in TAU_GRID]
        rows_to_write.append(step250_at_cal_tau)
        rows_to_write.append(step250_rec)
        for r in sorted(rows_to_write, key=lambda r: r["tau"]):
            w.writerow([
                f"{r['tau']:.6f}",
                f"{r['dev_real_fpr']:.4f}",
                f"{r['lockbox_real_fpr']:.4f}",
                f"{r['dor_dev_fpr']:.4f}",
                f"{r['poor_quality_dev_fpr']:.4f}",
                f"{r['lighting_extreme_dev_fpr']:.4f}",
                f"{r['teams_fake_dev_recall']:.4f}",
                f"{r['viso_enh_dev_recall']:.4f}",
                f"{r['deeplive_enh_dev_recall']:.4f}",
                f"{r['dev_macro_recall']:.4f}",
                f"{r['lockbox_fake_recall']:.4f}",
            ])

    # Comparison long-form: ckpt × τ
    with open(HERE / f"_comparison_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ckpt", "tau", "tau_kind",
            "dev_real_fpr", "lockbox_real_fpr", "dor_dev_fpr",
            "dev_macro_recall", "lockbox_fake_recall",
            "teams_fake_dev_recall", "viso_enh_dev_recall", "deeplive_enh_dev_recall",
            "poor_quality_dev_fpr", "lighting_extreme_dev_fpr",
        ])
        # grid
        for tau in TAU_GRID:
            for ckpt in CKPTS:
                r = metrics_at_tau(rows_by_ckpt[ckpt], tau)
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
                    f"{r['poor_quality_dev_fpr']:.4f}",
                    f"{r['lighting_extreme_dev_fpr']:.4f}",
                ])
        # cal
        for ckpt, r in [(STEP250_KEY, step250_at_cal_tau), (P8A_KEY, p8a_at_cal),
                        (E2B_KEY, e2b_at_cal), (T5C_KEY, t5c_at_cal)]:
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
                f"{r['poor_quality_dev_fpr']:.4f}",
                f"{r['lighting_extreme_dev_fpr']:.4f}",
            ])
        # rec-τ
        for ckpt, r in [(STEP250_KEY, step250_rec), (P8A_KEY, p8a_at_rec),
                        (E2B_KEY, e2b_at_rec), (T5C_KEY, t5c_at_rec)]:
            w.writerow([
                ckpt, f"{step250_rec_tau:.4f}", "recommended",
                f"{r['dev_real_fpr']:.4f}",
                f"{r['lockbox_real_fpr']:.4f}",
                f"{r['dor_dev_fpr']:.4f}",
                f"{r['dev_macro_recall']:.4f}",
                f"{r['lockbox_fake_recall']:.4f}",
                f"{r['teams_fake_dev_recall']:.4f}",
                f"{r['viso_enh_dev_recall']:.4f}",
                f"{r['deeplive_enh_dev_recall']:.4f}",
                f"{r['poor_quality_dev_fpr']:.4f}",
                f"{r['lighting_extreme_dev_fpr']:.4f}",
            ])

    # Recommended τ summary (single row each)
    with open(HERE / f"_recommended_tau_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ckpt", "tau", "lockbox_real_fpr", "lockbox_fake_recall",
            "dev_macro_recall", "dor_dev_fpr",
            "teams_fake_dev_recall", "viso_enh_dev_recall", "deeplive_enh_dev_recall",
        ])
        for ckpt, r in [(STEP250_KEY, step250_rec), (P8A_KEY, p8a_at_rec),
                        (E2B_KEY, e2b_at_rec), (T5C_KEY, t5c_at_rec)]:
            w.writerow([
                ckpt, f"{step250_rec_tau:.4f}",
                f"{r['lockbox_real_fpr']:.4f}",
                f"{r['lockbox_fake_recall']:.4f}",
                f"{r['dev_macro_recall']:.4f}",
                f"{r['dor_dev_fpr']:.4f}",
                f"{r['teams_fake_dev_recall']:.4f}",
                f"{r['viso_enh_dev_recall']:.4f}",
                f"{r['deeplive_enh_dev_recall']:.4f}",
            ])

    # Per-identity lockbox-only
    with open(HERE / f"_per_identity_lockbox_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["identity", "is_chronic_6", "n_videos", "n_fp_videos", "fpr", "exceeds_30pct"])
        for rec in per_id_by_suite["teams_real_all_lockbox"]:
            w.writerow([
                rec["identity"],
                "Y" if rec["is_chronic_6"] else "",
                rec["n_videos"], rec["n_fp_videos"],
                f"{rec['fpr']:.4f}",
                "Y" if rec["exceeds_30pct"] else "",
            ])

    # Per-identity pooled
    with open(HERE / f"_per_identity_pooled_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "identity", "is_chronic_6", "n_videos_pooled", "n_fp_videos_pooled",
            "fpr_pooled", "exceeds_30pct", "per_suite_breakdown",
        ])
        for rec in per_id_pooled:
            w.writerow([
                rec["identity"], "Y" if rec["is_chronic_6"] else "",
                rec["n_videos"], rec["n_fp_videos"],
                f"{rec['fpr']:.4f}",
                "Y" if rec["exceeds_30pct"] else "",
                rec["per_suite_breakdown"],
            ])

    # Per-identity cross-ckpt
    with open(HERE / f"_per_identity_cross_ckpt_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        labels = ["step250_rec", "p8a_rec", "e2b_rec", "t5c_rec",
                  "step250_cal", "p8a_cal", "e2b_cal", "t5c_cal"]
        head = ["identity", "is_chronic_6"]
        for L in labels:
            head += [f"{L}_fpr", f"{L}_fp_n"]
        w.writerow(head)
        for r in cross_records:
            row = [r["identity"], "Y" if r["is_chronic_6"] else ""]
            for L in labels:
                row += [f"{r[f'{L}_fpr']:.4f}", r[f"{L}_fp_n"]]
            w.writerow(row)

    # Chikara context CSV
    with open(HERE / f"_chikara_context_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ckpt", "tau_label", "tau", "suite", "n_videos", "n_fp", "fpr",
                    "prob_max", "prob_p90", "prob_p50", "prob_min"])
        for c in chikara_context:
            w.writerow([
                c["ckpt"], c["tau_label"], f"{c['tau']:.4f}", c["suite"],
                c["n_videos"], c["n_fp"], f"{c['fpr']:.4f}",
                f"{c['prob_max']:.4f}", f"{c['prob_p90']:.4f}",
                f"{c['prob_p50']:.4f}", f"{c['prob_min']:.4f}",
            ])

    # JSON dump
    payload = {
        "date": DATE,
        "step250_rec_tau": step250_rec_tau,
        "tau_cal": TAU_CAL,
        "step250_pareto_grid": step250_pareto,
        "step250_at_cal": step250_at_cal_tau,
        "step250_at_rec": step250_rec,
        "comparison_at_rec": {
            STEP250_KEY: step250_rec,
            P8A_KEY: p8a_at_rec,
            E2B_KEY: e2b_at_rec,
            T5C_KEY: t5c_at_rec,
        },
        "at_each_cal": {
            STEP250_KEY: step250_at_cal_tau,
            P8A_KEY: p8a_at_cal,
            E2B_KEY: e2b_at_cal,
            T5C_KEY: t5c_at_cal,
        },
        "per_id_lockbox_step250_rec": per_id_by_suite["teams_real_all_lockbox"],
        "per_id_pooled_step250_rec": per_id_pooled,
        "cross_ckpt_per_id": cross_records,
        "chikara_context": chikara_context,
    }
    (HERE / f"_results_{DATE}.json").write_text(json.dumps(payload, indent=2))

    # ---------- Print summary ----------
    print("\n--- Step 1: step250 Pareto curve ---")
    hdr = f"{'τ':>7s} {'dev_fpr':>8s} {'lb_fpr':>7s} {'dor_fpr':>8s} {'macro_R':>8s} {'lb_R':>7s} {'teams_R':>8s} {'viso_R':>7s} {'dl_R':>7s}"
    print(hdr)
    rows_all = [metrics_at_tau(rows_by_ckpt[STEP250_KEY], t) for t in TAU_GRID] + [step250_at_cal_tau, step250_rec]
    for r in sorted(rows_all, key=lambda r: r["tau"]):
        tag = ""
        if abs(r["tau"] - tau_step250_cal) < 1e-6: tag = "  (cal)"
        if abs(r["tau"] - step250_rec_tau) < 1e-6: tag = "  (rec)"
        print(f"{r['tau']:>7.4f} {r['dev_real_fpr']:>8.4f} {r['lockbox_real_fpr']:>7.4f} "
              f"{r['dor_dev_fpr']:>8.4f} {r['dev_macro_recall']:>8.4f} {r['lockbox_fake_recall']:>7.4f} "
              f"{r['teams_fake_dev_recall']:>8.4f} {r['viso_enh_dev_recall']:>7.4f} {r['deeplive_enh_dev_recall']:>7.4f}{tag}")

    print(f"\n--- Headline @ recommended τ (step250 forced to lb_real_fpr ≤ 0.025) ---")
    print(f"  τ = {step250_rec_tau:.6f}")
    for ckpt, r in [(STEP250_KEY, step250_rec), (P8A_KEY, p8a_at_rec),
                    (E2B_KEY, e2b_at_rec), (T5C_KEY, t5c_at_rec)]:
        print(f"  {ckpt:30s} lb_fpr={r['lockbox_real_fpr']:.4f} "
              f"lb_R={r['lockbox_fake_recall']:.4f} macro_R={r['dev_macro_recall']:.4f} "
              f"dor_fpr={r['dor_dev_fpr']:.4f}")

    print(f"\n--- Each ckpt at its OWN cal-τ (operational) ---")
    for ckpt, r in [(STEP250_KEY, step250_at_cal_tau), (P8A_KEY, p8a_at_cal),
                    (E2B_KEY, e2b_at_cal), (T5C_KEY, t5c_at_cal)]:
        print(f"  {ckpt:30s} τ={TAU_CAL[ckpt]:.4f} "
              f"lb_fpr={r['lockbox_real_fpr']:.4f} lb_R={r['lockbox_fake_recall']:.4f} "
              f"macro_R={r['dev_macro_recall']:.4f} dor_fpr={r['dor_dev_fpr']:.4f}")

    print(f"\n--- Per-identity step250 @ rec-τ (pooled, sorted by FPR) ---")
    n_over_30 = 0
    for rec in per_id_pooled:
        if rec["n_videos"] < 5: continue
        flag = " >30%" if rec["exceeds_30pct"] else ""
        if rec["exceeds_30pct"]: n_over_30 += 1
        chronic_tag = " CHRONIC" if rec["is_chronic_6"] else ""
        print(f"  {rec['identity']:<25s} n={rec['n_videos']:4d} fp={rec['n_fp_videos']:4d} "
              f"fpr={rec['fpr']:.4f}{flag}{chronic_tag}")
    print(f"\n  Identities (n≥5) with FPR > 30%: {n_over_30}")

    print(f"\n--- Chikara_Takahashi__s22 context (lockbox; n=25) ---")
    print(f"{'ckpt':<30s} {'τ_lbl':>6s} {'τ':>8s} {'n_fp/n':>10s} {'fpr':>7s} {'pmax':>6s} {'pmed':>6s}")
    for c in chikara_context:
        print(f"  {c['ckpt']:<28s} {c['tau_label']:>6s} {c['tau']:>8.4f} "
              f"{c['n_fp']:>4d}/{c['n_videos']:<4d}  {c['fpr']:>5.2%}  "
              f"{c['prob_max']:>5.2f}  {c['prob_p50']:>5.2f}")

    print(f"\nWritten:")
    for fn in ["_step250_pareto", "_comparison", "_recommended_tau",
               "_per_identity_lockbox", "_per_identity_pooled",
               "_per_identity_cross_ckpt", "_chikara_context", "_results"]:
        print(f"  {HERE / f'{fn}_{DATE}.csv' if fn != '_results' else HERE / f'{fn}_{DATE}.json'}")


if __name__ == "__main__":
    main()
