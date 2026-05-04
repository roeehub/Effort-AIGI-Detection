#!/usr/bin/env python3
"""
PA + PC-codec full eval orchestrator.

Pulls scorecards + frame reports from GCS, runs F4 substrate cleaning and
per-substrate tau calibration on each ckpt, and writes a comparison table
against the E2B / P8A baselines.

Run as:
    python3 analysis/pa_pc_eval_2026-05-05/run_full_eval.py

Assumes:
    - PA+PC F0 contract scorecard at gs://training-job-outputs/test_results/pa_pc_promotion_contract_2026-05-05/
    - F4 reference baselines at analysis/substrate_cleaning_eval_2026-05-05/
    - per-substrate tau tool at analysis/per_substrate_tau_calibration_2026-05-05/
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT_DIR = ROOT / "analysis/pa_pc_eval_2026-05-05"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GCS_BASE = "gs://training-job-outputs/test_results/pa_pc_promotion_contract_2026-05-05/pa-pc-promotion-contract-2026-05-05"
CKPTS = [
    "P8A_REFERENCE_STEP5000",
    "E2B_TOP_N_STEP3200",
    "PA_TOP_N_STEP5600",
    "PA_TOP_N_STEP3800",
    "PA_PERIODIC_STEP5000",
    "PC_TOP_N_STEP7400",
    "PC_TOP_N_STEP5400",
    "PC_PERIODIC_STEP5000",
]

# F4 needs these 4 suites per ckpt
F4_SUITES = ["teams_real_all_dev", "visomaster_enhanced_macro_dev", "deeplive_enhanced_dev", "teams_fake_all_dev"]
# per-substrate tau needs an extra lockbox bucket
PSUBSTRATE_EXTRA = ["teams_real_all_lockbox", "teams_fake_all_lockbox"]


def run(cmd, **kw):
    print(f"+ {' '.join(str(x) for x in cmd)}")
    return subprocess.run(cmd, check=False, **kw)


def gsutil_cat(uri):
    r = subprocess.run(["gsutil", "cat", uri], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return r.stdout


def download_reports(ckpt_keys):
    """Download frame-report CSVs for the suites we need into raw_reports/.
    Filename format: <suite>_<ckpt_lowercase>_frames_report.csv
    """
    raw_dir = OUT_DIR / "raw_reports"
    raw_dir.mkdir(exist_ok=True)
    suites = list(set(F4_SUITES) | set(PSUBSTRATE_EXTRA))
    paths_by_ckpt = defaultdict(dict)
    for ckpt in ckpt_keys:
        ck_low = ckpt.lower()
        for suite in suites:
            fname = f"{suite}_{ck_low}_frames_report.csv"
            local = raw_dir / fname
            if local.exists() and local.stat().st_size > 100:
                paths_by_ckpt[ckpt][suite] = str(local)
                continue
            uri = f"{GCS_BASE}/reports/{fname}"
            r = subprocess.run(["gsutil", "cp", uri, str(local)], capture_output=True, text=True)
            if r.returncode == 0:
                paths_by_ckpt[ckpt][suite] = str(local)
            else:
                print(f"  MISSING: {fname} (likely suite not in this run's manifest)", file=sys.stderr)
    return paths_by_ckpt


def download_scorecards():
    """Download the scorecard CSVs (long + wide)."""
    sc_dir = OUT_DIR
    for fname in ["scorecard.csv", "scorecard.wide.csv", "scorecard.json"]:
        uri = f"{GCS_BASE}/diagnostic_scorecard/{fname}"
        local = sc_dir / fname
        if local.exists():
            continue
        r = subprocess.run(["gsutil", "cp", uri, str(local)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  WARN: could not download {fname}", file=sys.stderr)


def run_f4(ckpt, paths):
    """Run substrate cleaning eval; returns parsed summary dict or None."""
    out_dir = OUT_DIR / "f4_outputs"
    out_dir.mkdir(exist_ok=True)
    real_csv = paths.get("teams_real_all_dev")
    if not real_csv:
        return None
    cmd = [
        "python3", str(ROOT / "analysis/substrate_cleaning_eval_2026-05-05/run_clean_eval.py"),
        "--ckpt-name", ckpt.lower(),
        "--real-csv", real_csv,
        "--out-dir", str(out_dir),
    ]
    for suite in ["visomaster_enhanced_macro_dev", "deeplive_enhanced_dev", "teams_fake_all_dev"]:
        path = paths.get(suite)
        if path:
            cmd += ["--fake-csv", f"{suite}={path}"]
    r = run(cmd, capture_output=True, text=True)
    summary_path = out_dir / f"{ckpt.lower()}_summary.json"
    if not summary_path.exists():
        print(f"  F4 FAIL for {ckpt}: {r.stderr[:500]}", file=sys.stderr)
        return None
    with open(summary_path) as f:
        return json.load(f)


def run_psubstrate(ckpt, paths):
    """Run per-substrate tau calibration; returns parsed recommendations or None."""
    out_dir = OUT_DIR / "psubstrate_outputs" / ckpt.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    real_dev = paths.get("teams_real_all_dev")
    real_lockbox = paths.get("teams_real_all_lockbox")
    if not real_dev:
        return None
    cmd = [
        "python3", str(ROOT / "analysis/per_substrate_tau_calibration_2026-05-05/run_calibration.py"),
        "--real-dev", real_dev,
        "--out", str(out_dir),
    ]
    if real_lockbox:
        cmd += ["--real-lockbox", real_lockbox]
    for nm, suite_key in [
        ("viso_dev", "visomaster_enhanced_macro_dev"),
        ("deeplive_dev", "deeplive_enhanced_dev"),
        ("teams_fake_dev", "teams_fake_all_dev"),
        ("teams_fake_lockbox", "teams_fake_all_lockbox"),
    ]:
        path = paths.get(suite_key)
        if path:
            cmd += ["--fake-suite", f"name={nm} path={path}"]
    r = run(cmd, capture_output=True, text=True)
    rec_path = out_dir / "tau_recommendations.json"
    if not rec_path.exists():
        print(f"  psubstrate FAIL for {ckpt}: {r.stderr[:500]}", file=sys.stderr)
        return None
    with open(rec_path) as f:
        return json.load(f)


def parse_long_scorecard():
    """Parse scorecard.csv (long form) into {ckpt: {suite: row}}."""
    sc_path = OUT_DIR / "scorecard.csv"
    if not sc_path.exists():
        return {}
    out = defaultdict(dict)
    with open(sc_path) as f:
        for row in csv.DictReader(f):
            out[row["checkpoint_key"]][row["suite_name"]] = row
    return out


def main():
    print("=== Step 1: download scorecards ===")
    download_scorecards()
    sc = parse_long_scorecard()

    print("=== Step 2: download frame reports ===")
    paths_by_ckpt = download_reports(CKPTS)

    print("=== Step 3: run F4 per ckpt ===")
    f4_summaries = {}
    for ckpt in CKPTS:
        s = run_f4(ckpt, paths_by_ckpt.get(ckpt, {}))
        if s:
            f4_summaries[ckpt] = s

    print("=== Step 4: run per-substrate tau per ckpt ===")
    psub_recs = {}
    for ckpt in CKPTS:
        s = run_psubstrate(ckpt, paths_by_ckpt.get(ckpt, {}))
        if s:
            psub_recs[ckpt] = s

    print("=== Step 5: build comparison table ===")
    # F0 at tau=0.5 (raw scorecard)
    f0_tab = []
    for ckpt in CKPTS:
        if ckpt not in sc:
            continue
        row = {"ckpt": ckpt}
        for suite in ["visomaster_enhanced_macro_dev", "deeplive_enhanced_dev", "teams_fake_all_dev",
                      "teams_real_all_dev", "teams_real_all_lockbox", "teams_fake_all_lockbox"]:
            r = sc[ckpt].get(suite, {})
            if r:
                if "real" in suite:
                    row[suite] = float(r.get("real_fpr_at_0p5", 0) or 0)
                else:
                    row[suite] = float(r.get("fake_recall_at_0p5", 0) or 0)
        f0_tab.append(row)
    with open(OUT_DIR / "f0_at_tau_0p5.json", "w") as f:
        json.dump(f0_tab, f, indent=2)

    # F4 summaries — extract recall at FPR=10%
    f4_tab = []
    for ckpt, summ in f4_summaries.items():
        row = {"ckpt": ckpt, "tau_F4_at_FPR10": summ.get("tau_F4_at_FPR_10pct"),
               "fpr_F4_pct": summ.get("fpr_F4_pct"), "n_real_F4": summ.get("n_real_frames_F4")}
        for s in summ.get("per_suite", []):
            row[s["fake_suite"] + "_F4_FPR10"] = s.get("recall_at_tau_F4_FPR10_pct")
            row[s["fake_suite"] + "_F0"] = s.get("recall_at_tau_F0_pct")
        f4_tab.append(row)
    with open(OUT_DIR / "f4_summary.json", "w") as f:
        json.dump(f4_tab, f, indent=2)

    # per-substrate tau recommendations
    psub_tab = []
    for ckpt, rec in psub_recs.items():
        row = {"ckpt": ckpt}
        for sel in ["tau_strict", "tau_moderate", "tau_loose"]:
            sa = rec.get("selections", {}).get(sel, {}).get("substrate_aware_single_tau", {})
            row[sel + "_tau"] = sa.get("tau")
            row[sel + "_worst_substrate_fpr"] = sa.get("dev_worst_substrate_fpr")
            for r_key in ["viso_dev", "deeplive_dev", "teams_fake_dev", "teams_fake_lockbox"]:
                row[f"{sel}_{r_key}"] = sa.get("recall", {}).get(r_key)
        psub_tab.append(row)
    with open(OUT_DIR / "psubstrate_summary.json", "w") as f:
        json.dump(psub_tab, f, indent=2)

    print("=== Done. Outputs in analysis/pa_pc_eval_2026-05-05/ ===")
    print(f"  scorecard.csv: {(OUT_DIR / 'scorecard.csv').exists()}")
    print(f"  f0_at_tau_0p5.json: {(OUT_DIR / 'f0_at_tau_0p5.json').exists()}")
    print(f"  f4_summary.json: {(OUT_DIR / 'f4_summary.json').exists()}")
    print(f"  psubstrate_summary.json: {(OUT_DIR / 'psubstrate_summary.json').exists()}")


if __name__ == "__main__":
    main()
