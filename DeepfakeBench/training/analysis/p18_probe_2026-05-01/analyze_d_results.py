"""Pull D's contract scorecard outputs from GCS and analyze.

Run after Vertex job 4550151094264659968 reaches SUCCEEDED.
Pulls promotion_contract.json + checkpoint_summary.csv + selected_threshold_scorecard.csv
+ promotion_winner.json. Writes a Markdown summary.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
GCS_ROOT = "gs://training-job-outputs/test_results/teams_promotion_contract/p18-corrective-contract-20260502-103127"
LOCAL_PULL_DIR = REPO_ROOT / "analysis" / "p18_probe_2026-05-01" / "d_results"

ARTIFACTS = [
    "promotion_contract/promotion_contract.json",
    "promotion_contract/promotion_winner.json",
    "promotion_contract/checkpoint_summary.csv",
    "promotion_contract/selected_threshold_scorecard.csv",
    "promotion_contract/threshold_grid.csv",
    "diagnostic_scorecard/scorecard.json",
    "diagnostic_scorecard/scorecard.csv",
]


def pull_artifact(name: str) -> Path:
    src = f"{GCS_ROOT}/{name}"
    dst = LOCAL_PULL_DIR / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  pulling {src} → {dst}")
    subprocess.run(["gsutil", "cp", src, str(dst)], check=False)
    return dst


def main():
    LOCAL_PULL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Pulling artifacts from {GCS_ROOT}")
    for a in ARTIFACTS:
        pull_artifact(a)
    print("\n=== promotion_contract.json ===")
    pc_path = LOCAL_PULL_DIR / "promotion_contract" / "promotion_contract.json"
    if pc_path.exists():
        try:
            with open(pc_path) as f:
                pc = json.load(f)
            print(json.dumps(pc, indent=2)[:4000])
        except json.JSONDecodeError as e:
            print(f"  parse error: {e}")
            print(pc_path.read_text()[:2000])
    else:
        print("  (file not present)")

    print("\n=== promotion_winner.json ===")
    pw_path = LOCAL_PULL_DIR / "promotion_contract" / "promotion_winner.json"
    if pw_path.exists():
        try:
            print(json.dumps(json.load(open(pw_path)), indent=2)[:2000])
        except json.JSONDecodeError:
            print(pw_path.read_text()[:1000])
    else:
        print("  (file not present)")

    print("\n=== checkpoint_summary.csv (head) ===")
    cs_path = LOCAL_PULL_DIR / "promotion_contract" / "checkpoint_summary.csv"
    if cs_path.exists():
        try:
            cs = pd.read_csv(cs_path)
            print(cs.to_string())
        except Exception as e:
            print(f"  parse error: {e}")
            print(cs_path.read_text()[:2000])
    else:
        print("  (file not present)")

    print("\n=== selected_threshold_scorecard.csv (head) ===")
    ss_path = LOCAL_PULL_DIR / "promotion_contract" / "selected_threshold_scorecard.csv"
    if ss_path.exists():
        try:
            ss = pd.read_csv(ss_path)
            print(ss.head(50).to_string())
        except Exception as e:
            print(f"  parse error: {e}")
    else:
        print("  (file not present)")


if __name__ == "__main__":
    main()
