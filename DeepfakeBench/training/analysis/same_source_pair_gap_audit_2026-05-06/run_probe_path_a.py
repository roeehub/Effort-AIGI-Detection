"""SAME_SOURCE_PAIR_GAP_AUDIT_2026-05-06 — Phase 0j Path-A wrapper.

Re-run pair-gap audit using FRESH scores from frozen-feature NPZ extraction
(Vertex job 3077166152858730496). Replaces the cross-product proxy that
returned MIXED in run_probe.py.

Joins NPZ scores -> pair_gaps.csv via frame_path, computes
P(pair_gap <= 0 | missed_fake) per training lane.

CPU-only; no model forward pass; no sklearn calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GREEN_THRESHOLD = 0.25
AMBER_THRESHOLD = 0.10


def verdict_for(p_le_0):
    if p_le_0 is None or (isinstance(p_le_0, float) and np.isnan(p_le_0)):
        return "INSUFFICIENT_DATA"
    if p_le_0 > GREEN_THRESHOLD:
        return "GREEN"
    if p_le_0 >= AMBER_THRESHOLD:
        return "AMBER"
    return "RED"


def load_fresh_scores(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    fp = d["frame_path"]
    sc = d["scores"]
    ok = d["ok"]
    mask = ok == 1
    return dict(zip(fp[mask].tolist(), sc[mask].tolist()))


def lane_assignment(row):
    """Map a pair_gaps row to a runbook lane.

    Runbook's 6 paired training lanes:
      df40, deeplive_v1, deeplive_v2, viso_v1, viso_enhanced, viso_teams_enhanced

    Mapping rules:
      - df40: not present in pair_gaps.csv (all viso/deeplive/teams)
      - deeplive_v1: dor_fake_local AND method ends '_regular'
      - deeplive_v2: dor_fake_local AND method ends '_enhanced'
      - viso_v1: not present in pair_gaps.csv (only viso_v2_*)
      - viso_enhanced: visomaster_v2_dor (raw transport)
      - viso_teams_enhanced: not present in pair_gaps.csv (no teams transport on viso)

    Additional informative lanes (NOT in the 6-lane request, included for ground-truth context):
      - teams_passthrough_dev: teams_fake_all_dev
      - teams_passthrough_lockbox: teams_fake_all_lockbox
      - extra: extra (xiang/xinghe deeplive variants)
      - live_fakes_teams_prod: production fakes
    """
    fs = row["fake_suite"]
    method = row.get("method", "")
    if fs == "dor_fake_local":
        if isinstance(method, str) and method.endswith("_enhanced"):
            return "deeplive_v2"
        return "deeplive_v1"
    if fs == "visomaster_v2_dor":
        return "viso_enhanced"
    if fs == "teams_fake_all_dev":
        return "teams_passthrough_dev"
    if fs == "teams_fake_all_lockbox":
        return "teams_passthrough_lockbox"
    if fs == "extra":
        return "extra"
    if fs == "live_fakes_teams_prod":
        return "live_fakes_teams_prod"
    return "unknown"


# Runbook-specified 6 lanes (target output)
RUNBOOK_LANES = [
    "df40",
    "deeplive_v1",
    "deeplive_v2",
    "viso_v1",
    "viso_enhanced",
    "viso_teams_enhanced",
]

# All lanes we can report on (subset of runbook + informative additions)
ALL_LANES = RUNBOOK_LANES + [
    "teams_passthrough_dev",
    "teams_passthrough_lockbox",
    "extra",
    "live_fakes_teams_prod",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p8a_npz", required=True)
    ap.add_argument("--e2b_npz", required=True)
    ap.add_argument("--pair_gaps_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--miss_threshold", type=float, default=0.5,
                    help="fake_score < threshold counts as missed_fake")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pg = pd.read_csv(args.pair_gaps_csv)
    pg["lane"] = pg.apply(lane_assignment, axis=1)

    p8a_scores = load_fresh_scores(Path(args.p8a_npz))
    e2b_scores = load_fresh_scores(Path(args.e2b_npz))

    pg["real_score_fresh_P8A"] = pg["real_path"].map(p8a_scores)
    pg["fake_score_fresh_P8A"] = pg["fake_path"].map(p8a_scores)
    pg["real_score_fresh_E2B"] = pg["real_path"].map(e2b_scores)
    pg["fake_score_fresh_E2B"] = pg["fake_path"].map(e2b_scores)

    pg["pair_gap_fresh_P8A"] = pg["fake_score_fresh_P8A"] - pg["real_score_fresh_P8A"]
    pg["pair_gap_fresh_E2B"] = pg["fake_score_fresh_E2B"] - pg["real_score_fresh_E2B"]

    # Fallback to cached pair_gaps.csv columns when fresh scores not available
    # (gs://local placeholder paths). This is the same data the prior audit used.
    pg["pair_gap_used_P8A"] = pg["pair_gap_fresh_P8A"].fillna(pg["pair_gap_P8A"])
    pg["fake_score_used_P8A"] = pg["fake_score_fresh_P8A"].fillna(pg["fake_score_P8A"])
    pg["pair_gap_used_E2B"] = pg["pair_gap_fresh_E2B"].fillna(pg["pair_gap_E2B"])
    pg["fake_score_used_E2B"] = pg["fake_score_fresh_E2B"].fillna(pg["fake_score_E2B"])

    # Track score source: 'fresh' (NPZ) vs 'cached' (pair_gaps.csv)
    pg["score_source_P8A"] = np.where(pg["pair_gap_fresh_P8A"].notna(), "fresh", "cached")
    pg["score_source_E2B"] = np.where(pg["pair_gap_fresh_E2B"].notna(), "fresh", "cached")

    pg["covered_P8A"] = pg["pair_gap_used_P8A"].notna()
    pg["covered_E2B"] = pg["pair_gap_used_E2B"].notna()

    pg["missed_fake_P8A"] = pg["fake_score_used_P8A"] < args.miss_threshold
    pg["missed_fake_E2B"] = pg["fake_score_used_E2B"] < args.miss_threshold

    # Per-lane verdicts
    lane_results = {}
    for lane in ALL_LANES:
        sub = pg[pg["lane"] == lane]
        lane_results[lane] = {"in_runbook_six": lane in RUNBOOK_LANES, "by_ckpt": {}}
        if len(sub) == 0:
            for ckpt in ["P8A", "E2B"]:
                lane_results[lane]["by_ckpt"][ckpt] = {
                    "n_pairs_in_csv": 0,
                    "n_pairs_with_fresh_scores": 0,
                    "n_missed_fakes": 0,
                    "p_pair_gap_le_0_given_missed": None,
                    "verdict": "INSUFFICIENT_DATA",
                    "reason": "lane not present in pair_gaps.csv",
                }
            continue
        for ckpt in ["P8A", "E2B"]:
            cov_col = f"covered_{ckpt}"
            src_col = f"score_source_{ckpt}"
            covered = sub[sub[cov_col]]
            n_covered = len(covered)
            n_fresh = int((sub[src_col] == "fresh").sum())
            n_cached = int((sub[src_col] == "cached").sum())
            if n_covered == 0:
                lane_results[lane]["by_ckpt"][ckpt] = {
                    "n_pairs_in_csv": int(len(sub)),
                    "n_pairs_covered": 0,
                    "n_pairs_fresh": 0,
                    "n_pairs_cached": 0,
                    "n_missed_fakes": 0,
                    "p_pair_gap_le_0_given_missed": None,
                    "verdict": "INSUFFICIENT_DATA",
                    "reason": "no scores available",
                }
                continue
            miss_col = f"missed_fake_{ckpt}"
            gap_col = f"pair_gap_used_{ckpt}"
            missed = covered[covered[miss_col]]
            n_missed = len(missed)
            if n_missed == 0:
                lane_results[lane]["by_ckpt"][ckpt] = {
                    "n_pairs_in_csv": int(len(sub)),
                    "n_pairs_covered": int(n_covered),
                    "n_pairs_fresh": n_fresh,
                    "n_pairs_cached": n_cached,
                    "n_missed_fakes": 0,
                    "p_pair_gap_le_0_given_missed": None,
                    "verdict": "INSUFFICIENT_DATA",
                    "reason": "no missed fakes (fake_score >= 0.5 for all pairs)",
                }
                continue
            p_le_0 = float((missed[gap_col] <= 0).mean())
            lane_results[lane]["by_ckpt"][ckpt] = {
                "n_pairs_in_csv": int(len(sub)),
                "n_pairs_covered": int(n_covered),
                "n_pairs_fresh": n_fresh,
                "n_pairs_cached": n_cached,
                "n_missed_fakes": int(n_missed),
                "p_pair_gap_le_0_given_missed": p_le_0,
                "verdict": verdict_for(p_le_0),
                "score_source": "fresh" if n_fresh > 0 else "cached",
                "reason": None,
            }

    # Write per-lane CSV
    rows = []
    for lane in ALL_LANES:
        for ckpt in ["P8A", "E2B"]:
            r = lane_results[lane]["by_ckpt"][ckpt]
            rows.append({
                "lane": lane,
                "in_runbook_six": lane_results[lane]["in_runbook_six"],
                "ckpt": ckpt,
                **r,
            })
    pd.DataFrame(rows).to_csv(out / "per_lane_per_ckpt.csv", index=False)

    # Summary JSON
    n_green_p8a = sum(1 for l in RUNBOOK_LANES if lane_results[l]["by_ckpt"]["P8A"]["verdict"] == "GREEN")
    n_amber_p8a = sum(1 for l in RUNBOOK_LANES if lane_results[l]["by_ckpt"]["P8A"]["verdict"] == "AMBER")
    n_red_p8a = sum(1 for l in RUNBOOK_LANES if lane_results[l]["by_ckpt"]["P8A"]["verdict"] == "RED")
    n_insuf_p8a = sum(1 for l in RUNBOOK_LANES if lane_results[l]["by_ckpt"]["P8A"]["verdict"] == "INSUFFICIENT_DATA")

    n_green_e2b = sum(1 for l in RUNBOOK_LANES if lane_results[l]["by_ckpt"]["E2B"]["verdict"] == "GREEN")
    n_amber_e2b = sum(1 for l in RUNBOOK_LANES if lane_results[l]["by_ckpt"]["E2B"]["verdict"] == "AMBER")
    n_red_e2b = sum(1 for l in RUNBOOK_LANES if lane_results[l]["by_ckpt"]["E2B"]["verdict"] == "RED")
    n_insuf_e2b = sum(1 for l in RUNBOOK_LANES if lane_results[l]["by_ckpt"]["E2B"]["verdict"] == "INSUFFICIENT_DATA")

    summary = {
        "audit_name": "SAME_SOURCE_PAIR_GAP_AUDIT_PATH_A_2026-05-07",
        "phase": "0j",
        "source_npz_p8a": str(args.p8a_npz),
        "source_npz_e2b": str(args.e2b_npz),
        "source_pair_gaps_csv": str(args.pair_gaps_csv),
        "miss_threshold": args.miss_threshold,
        "decision_rule": {
            "green_threshold_p_le_0": GREEN_THRESHOLD,
            "amber_threshold_p_le_0": AMBER_THRESHOLD,
            "min_green_lanes_for_unconditional_p1": 2,
        },
        "runbook_lanes": RUNBOOK_LANES,
        "lane_results": lane_results,
        "runbook_six_summary": {
            "P8A": {
                "n_green": n_green_p8a,
                "n_amber": n_amber_p8a,
                "n_red": n_red_p8a,
                "n_insufficient": n_insuf_p8a,
            },
            "E2B": {
                "n_green": n_green_e2b,
                "n_amber": n_amber_e2b,
                "n_red": n_red_e2b,
                "n_insufficient": n_insuf_e2b,
            },
        },
        "p1_recommendation": {
            "P8A": (
                "LAUNCH_PE_PAIR_RANK_DRO" if n_green_p8a >= 2 else
                "LANE_RESTRICTED_LAUNCH" if n_green_p8a == 1 else
                "DEMOTE_P1"
            ),
            "E2B": (
                "LAUNCH_PE_PAIR_RANK_DRO" if n_green_e2b >= 2 else
                "LANE_RESTRICTED_LAUNCH" if n_green_e2b == 1 else
                "DEMOTE_P1"
            ),
        },
    }

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Wrote {out / 'summary.json'}")
    print(f"Wrote {out / 'per_lane_per_ckpt.csv'}")
    print()
    print("=== Phase 0j (Path A) per-lane verdicts ===")
    for lane in RUNBOOK_LANES:
        p8a_v = lane_results[lane]["by_ckpt"]["P8A"]
        e2b_v = lane_results[lane]["by_ckpt"]["E2B"]
        print(f"  {lane:30s}  P8A: {p8a_v['verdict']:18s} (n_missed={p8a_v['n_missed_fakes']}, p_le_0={p8a_v['p_pair_gap_le_0_given_missed']})")
        print(f"  {'':30s}  E2B: {e2b_v['verdict']:18s} (n_missed={e2b_v['n_missed_fakes']}, p_le_0={e2b_v['p_pair_gap_le_0_given_missed']})")
    print()
    print("=== Informative additional lanes (not in runbook six) ===")
    for lane in [l for l in ALL_LANES if l not in RUNBOOK_LANES]:
        p8a_v = lane_results[lane]["by_ckpt"]["P8A"]
        e2b_v = lane_results[lane]["by_ckpt"]["E2B"]
        print(f"  {lane:30s}  P8A: {p8a_v['verdict']:18s} (n_missed={p8a_v['n_missed_fakes']}, p_le_0={p8a_v['p_pair_gap_le_0_given_missed']})")
        print(f"  {'':30s}  E2B: {e2b_v['verdict']:18s} (n_missed={e2b_v['n_missed_fakes']}, p_le_0={e2b_v['p_pair_gap_le_0_given_missed']})")


if __name__ == "__main__":
    main()
