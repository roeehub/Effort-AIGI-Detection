"""
PD vs P1 head-to-head builder.

Pulls per-video probability reports from the PD scorecard (which were emitted
without an aggregated calibrated-tau scorecard) and computes the same
contract-style calibrated-tau metrics that P1's selected_threshold_scorecard.csv
already contains. Then assembles a single head-to-head table on the 9 contract
suites for these checkpoints:
  - P8A_REFERENCE_STEP5000   (baseline, present in both eval runs)
  - E2B_TOP_N_STEP3200       (FT-base for both PD and P1, deployment ckpt)
  - DEEPLIVE_CORR_TOP_N_STEP4800   (PD deeplive arm, highest AUC)
  - DEEPLIVE_CORR_TOP_N_STEP1800   (PD deeplive arm, mid-trajectory)
  - DEEPLIVE_CORR_PERIODIC_STEP2000(PD deeplive arm, periodic anchor)
  - VISO_CORR_TOP_N_STEP600       (PD viso arm, highest AUC)
  - VISO_CORR_PERIODIC_STEP1000   (PD viso arm, mid-trajectory)
  - VISO_CORR_PERIODIC_STEP2000   (PD viso arm, latest)
  - P1_BUNDLE_PERIODIC_STEP500     (P1, bundle arm)
  - P1_PAIRRANK_PERIODIC_STEP500   (P1, pairrank arm; promotion winner)

Calibration policy (matches P1's promotion contract):
  contract.target_real_fpr = 0.07         (on teams_real_all_dev)
  contract.target_stress_fpr = 0.10       (on poor_quality + lighting_extreme)
  contract.target_fake_recall_min = 0.30  (informational floor)

  selected_threshold = smallest tau in candidate grid such that:
    real_fpr(teams_real_all_dev)        <= 0.07 AND
    real_fpr(teams_real_poor_quality_dev) <= 0.10 AND
    real_fpr(teams_real_lighting_extreme_dev) <= 0.10
  ties broken by maximizing dev_fake_macro_recall

  candidate grid: unique percentiles (0..100 in 0.1 steps) of the union of
  per-video probabilities on teams_real_all_dev + the three dev fake suites.

Output: head_to_head_at_calibrated_tau.csv (one row per (ckpt,suite)),
        head_to_head_pivot.csv (wide form),
        PD_VS_P1_FACTS_2026-05-07.md
"""
import csv
import os
import json
import math
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
PD_REPORTS = ROOT / "raw" / "pd_videos_reports"
P1_SCORECARD = ROOT.parent / "p1_pe_eval_2026-05-07" / "scorecard" / "selected_threshold_scorecard.csv"

CKPT_KEYS = {
    "p8a_reference_step5000": "P8A_REFERENCE_STEP5000",
    "e2b_top_n_step3200": "E2B_TOP_N_STEP3200",
    "deeplive_corr_periodic_step2000": "DEEPLIVE_CORR_PERIODIC_STEP2000",
    "deeplive_corr_top_n_step1800": "DEEPLIVE_CORR_TOP_N_STEP1800",
    "deeplive_corr_top_n_step4800": "DEEPLIVE_CORR_TOP_N_STEP4800",
    "viso_corr_periodic_step1000": "VISO_CORR_PERIODIC_STEP1000",
    "viso_corr_periodic_step2000": "VISO_CORR_PERIODIC_STEP2000",
    "viso_corr_top_n_step600": "VISO_CORR_TOP_N_STEP600",
}

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

DEV_FAKE_SUITES = ["teams_fake_all_dev", "visomaster_enhanced_macro_dev", "deeplive_enhanced_dev"]
DEV_REAL_SUITE = "teams_real_all_dev"
DEV_REAL_STRESS_SUITES = ["teams_real_poor_quality_dev", "teams_real_lighting_extreme_dev"]
TARGET_REAL_FPR = 0.07
TARGET_STRESS_FPR = 0.10

def load_videos_report(suite, ckpt_lower):
    p = PD_REPORTS / f"{suite}_{ckpt_lower}_videos_report.csv"
    if not p.exists():
        return None
    rows = []
    with p.open() as f:
        for row in csv.DictReader(f):
            rows.append({
                "video_id": row["video_id"],
                "label": int(row["label"]),
                "prob": float(row["avg_video_prob"]),
            })
    return rows

def fpr_at_tau(rows, tau):
    """rows must be all label==0 (real)."""
    n = len(rows)
    if n == 0: return 0.0
    fp = sum(1 for r in rows if r["prob"] > tau)
    return fp / n

def recall_at_tau(rows, tau):
    """rows must be all label==1 (fake)."""
    n = len(rows)
    if n == 0: return 0.0
    tp = sum(1 for r in rows if r["prob"] > tau)
    return tp / n

def calibrate_tau_for_ckpt(ckpt_lower):
    """Match P1's contract calibration. Returns (tau, candidate_count, dev_macro_recall)."""
    real_dev = load_videos_report(DEV_REAL_SUITE, ckpt_lower)
    poor = load_videos_report("teams_real_poor_quality_dev", ckpt_lower)
    lightext = load_videos_report("teams_real_lighting_extreme_dev", ckpt_lower)
    fakes = {s: load_videos_report(s, ckpt_lower) for s in DEV_FAKE_SUITES}

    if any(x is None for x in (real_dev, poor, lightext)) or any(v is None for v in fakes.values()):
        return None, 0, None

    # candidate grid: percentiles 0..100 step 0.1 of (real_dev ∪ all dev fakes) + sentinel 0,1
    union_probs = [r["prob"] for r in real_dev]
    for s in DEV_FAKE_SUITES:
        union_probs.extend(r["prob"] for r in fakes[s])
    union_probs.sort()
    grid = set([0.0, 1.0])
    n = len(union_probs)
    for i in range(1001):
        q = i / 1000.0
        idx = min(int(q * (n - 1)), n - 1)
        grid.add(union_probs[idx])
    # Also dense linear grid for tighter calibration
    for i in range(0, 10001):
        grid.add(i / 10000.0)
    grid = sorted(grid)

    best = None
    for tau in grid:
        fpr_real = fpr_at_tau(real_dev, tau)
        if fpr_real > TARGET_REAL_FPR:
            continue
        fpr_poor = fpr_at_tau(poor, tau)
        if fpr_poor > TARGET_STRESS_FPR:
            continue
        fpr_le = fpr_at_tau(lightext, tau)
        if fpr_le > TARGET_STRESS_FPR:
            continue
        # all gates pass: maximize dev_macro_fake_recall
        recalls = [recall_at_tau(fakes[s], tau) for s in DEV_FAKE_SUITES]
        macro = sum(recalls) / len(recalls)
        # Tie break: prefer smaller tau (more sensitive) when macro equal
        key = (-macro, tau)
        if best is None or key < best[0]:
            best = (key, tau, macro)
    if best is None:
        return None, len(grid), None
    return best[1], len(grid), best[2]

def emit_pd_rows():
    """One row per (ckpt, suite) on PD ckpts. Computes calibrated-tau metrics."""
    out = []
    for ckpt_lower, ckpt_pretty in CKPT_KEYS.items():
        tau, cand_count, macro = calibrate_tau_for_ckpt(ckpt_lower)
        if tau is None:
            print(f"WARN: no tau found for {ckpt_lower}")
            continue
        for suite in CONTRACT_SUITES:
            rows = load_videos_report(suite, ckpt_lower)
            if rows is None:
                continue
            n = len(rows)
            n_real = sum(1 for r in rows if r["label"] == 0)
            n_fake = sum(1 for r in rows if r["label"] == 1)
            real_fpr = fpr_at_tau([r for r in rows if r["label"] == 0], tau) if n_real > 0 else None
            fake_recall = recall_at_tau([r for r in rows if r["label"] == 1], tau) if n_fake > 0 else None
            out.append({
                "checkpoint_key": ckpt_pretty,
                "suite_name": suite,
                "threshold": tau,
                "n_videos": n,
                "n_real": n_real,
                "n_fake": n_fake,
                "real_fpr": real_fpr,
                "fake_recall": fake_recall,
                "threshold_candidate_count": cand_count,
                "dev_fake_macro_recall_at_calibrated_tau": macro,
            })
    return out

def load_p1_rows():
    """Pull P1 ckpts on the contract suites at their calibrated tau (already computed)."""
    out = []
    target_ckpts = {"P1_BUNDLE_PERIODIC_STEP500", "P1_PAIRRANK_PERIODIC_STEP500"}
    with P1_SCORECARD.open() as f:
        for row in csv.DictReader(f):
            ck = row["checkpoint_key"]
            if ck not in target_ckpts:
                continue
            if row["suite_name"] not in CONTRACT_SUITES:
                continue
            out.append({
                "checkpoint_key": ck,
                "suite_name": row["suite_name"],
                "threshold": float(row["threshold"]),
                "n_videos": int(row["n_videos"]),
                "n_real": int(row["n_real"]),
                "n_fake": int(row["n_fake"]),
                "real_fpr": float(row["real_fpr"]) if row["real_fpr"] else None,
                "fake_recall": float(row["fake_recall"]) if row["fake_recall"] else None,
                "threshold_candidate_count": "from_p1_scorecard",
                "dev_fake_macro_recall_at_calibrated_tau": None,  # filled below
            })
    return out

def main():
    pd_rows = emit_pd_rows()
    p1_rows = load_p1_rows()

    # Compute P1 dev macro fake recall from rows
    for ck in {r["checkpoint_key"] for r in p1_rows}:
        recalls = []
        for s in DEV_FAKE_SUITES:
            for r in p1_rows:
                if r["checkpoint_key"] == ck and r["suite_name"] == s:
                    if r["fake_recall"] is not None:
                        recalls.append(r["fake_recall"])
        macro = sum(recalls) / len(recalls) if recalls else None
        for r in p1_rows:
            if r["checkpoint_key"] == ck:
                r["dev_fake_macro_recall_at_calibrated_tau"] = macro

    # P1's E2B and P8A baselines from P1 scorecard (the ones we already have)
    base_rows = []
    with P1_SCORECARD.open() as f:
        for row in csv.DictReader(f):
            ck = row["checkpoint_key"]
            if ck not in {"P8A_REFERENCE_STEP5000", "E2B_TOP_N_STEP3200"}:
                continue
            if row["suite_name"] not in CONTRACT_SUITES:
                continue
            base_rows.append({
                "checkpoint_key": ck + "__from_P1_run",
                "suite_name": row["suite_name"],
                "threshold": float(row["threshold"]),
                "n_videos": int(row["n_videos"]),
                "n_real": int(row["n_real"]),
                "n_fake": int(row["n_fake"]),
                "real_fpr": float(row["real_fpr"]) if row["real_fpr"] else None,
                "fake_recall": float(row["fake_recall"]) if row["fake_recall"] else None,
                "threshold_candidate_count": "from_p1_scorecard",
                "dev_fake_macro_recall_at_calibrated_tau": None,
            })
    for ck in {r["checkpoint_key"] for r in base_rows}:
        recalls = []
        for s in DEV_FAKE_SUITES:
            for r in base_rows:
                if r["checkpoint_key"] == ck and r["suite_name"] == s:
                    if r["fake_recall"] is not None:
                        recalls.append(r["fake_recall"])
        macro = sum(recalls) / len(recalls) if recalls else None
        for r in base_rows:
            if r["checkpoint_key"] == ck:
                r["dev_fake_macro_recall_at_calibrated_tau"] = macro

    all_rows = pd_rows + p1_rows + base_rows

    # Long-form output
    out_long = ROOT / "head_to_head_at_calibrated_tau.csv"
    with out_long.open("w") as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            for r in all_rows:
                w.writerow(r)

    # Wide-form (pivot)
    out_wide = ROOT / "head_to_head_pivot.csv"
    by_ck = defaultdict(dict)
    for r in all_rows:
        by_ck[r["checkpoint_key"]][r["suite_name"]] = r
    with out_wide.open("w") as f:
        cols = ["checkpoint_key", "threshold", "dev_fake_macro_recall_at_calibrated_tau"]
        for s in CONTRACT_SUITES:
            cols.extend([f"{s}__metric", f"{s}__value", f"{s}__n_videos"])
        w = csv.writer(f)
        w.writerow(cols)
        ckpt_order = [
            "P8A_REFERENCE_STEP5000",
            "P8A_REFERENCE_STEP5000__from_P1_run",
            "E2B_TOP_N_STEP3200",
            "E2B_TOP_N_STEP3200__from_P1_run",
            "DEEPLIVE_CORR_PERIODIC_STEP2000",
            "DEEPLIVE_CORR_TOP_N_STEP1800",
            "DEEPLIVE_CORR_TOP_N_STEP4800",
            "VISO_CORR_PERIODIC_STEP1000",
            "VISO_CORR_PERIODIC_STEP2000",
            "VISO_CORR_TOP_N_STEP600",
            "P1_BUNDLE_PERIODIC_STEP500",
            "P1_PAIRRANK_PERIODIC_STEP500",
        ]
        for ck in ckpt_order:
            if ck not in by_ck:
                continue
            data = by_ck[ck]
            any_row = next(iter(data.values()))
            row = [ck, f"{any_row['threshold']:.5f}", f"{any_row['dev_fake_macro_recall_at_calibrated_tau']:.4f}" if any_row['dev_fake_macro_recall_at_calibrated_tau'] is not None else ""]
            for s in CONTRACT_SUITES:
                if s not in data:
                    row.extend(["", "", ""])
                    continue
                rd = data[s]
                if rd["n_real"] > 0:
                    metric = "real_fpr"
                    value = rd["real_fpr"]
                else:
                    metric = "fake_recall"
                    value = rd["fake_recall"]
                row.extend([metric, f"{value:.4f}" if value is not None else "", str(rd["n_videos"])])
            w.writerow(row)

    print(f"Wrote {out_long}")
    print(f"Wrote {out_wide}")
    print(f"Total rows: {len(all_rows)}")

if __name__ == "__main__":
    main()
