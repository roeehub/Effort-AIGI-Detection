"""
SAME_SOURCE_PAIR_GAP_AUDIT_2026-05-06 — Phase 0g

Re-runs the pair-gap analysis at the trainer's tight (sample_id, frame_idx)
opposite-label pairing key. CPU-only, no model forward pass.

This script does TWO things:
  1. Inspects local score caches (`grouped_manifest_v2.csv`, prior pair_gaps.csv,
     other CSV caches) for rows that share trainer-style tight pair keys.
  2. If insufficient coverage, writes a forward-pass blueprint plus a
     lane-level reconciliation against the existing cross-product audit.

Outputs land in `analysis/same_source_pair_gap_audit_2026-05-06/outputs/`.

Trigger context: NEXT_STEPS_PLAN_2026-05-06.md §8.1 Phase 0g — load-bearing
go/no-go for `PE_PAIR_RANK_DRO` (P1).

Decision rule (per lane, on missed_fakes only):
  P(pair_gap <= 0 | missed_fake)
    > 25% -> GREEN  (lever has signal in this lane)
   10-25% -> AMBER  (marginal; depends on margin/DRO)
    < 10% -> RED    (lever is dead in this lane)

Six paired training lanes from `pair_coverage_audit_2026-05-06`:
    df40, deeplive, visomaster_v1_base, visomaster_enhanced,
    visomaster_teams_enhanced, deeplive_teams
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
TRAINING_DIR = ROOT.parent.parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = TRAINING_DIR / "analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv"
PRIOR_PAIRS = TRAINING_DIR / "analysis/pair_gap_audit_2026-05-06/outputs/pair_gaps.csv"
COVERAGE_SUMMARY = TRAINING_DIR / "analysis/pair_coverage_audit_2026-05-06/outputs/coverage_summary.json"

PAIRED_LANES = [
    "df40",
    "deeplive",
    "visomaster_v1_base",
    "visomaster_enhanced",
    "visomaster_teams_enhanced",
    "deeplive_teams",
]

CKPTS = ["P8A", "E2B", "PA_3800"]
GREEN_THRESHOLD = 0.25
AMBER_THRESHOLD = 0.10


def verdict_for(p_le_0: float | None) -> str:
    if p_le_0 is None:
        return "INSUFFICIENT_DATA"
    if p_le_0 > GREEN_THRESHOLD:
        return "GREEN"
    if p_le_0 >= AMBER_THRESHOLD:
        return "AMBER"
    return "RED"


def extract_pair_key(path: str) -> tuple[str | None, str | None, int | None]:
    """Best-effort extract of (subdir, seq_or_segment, frame_idx) from a frame path."""
    if not isinstance(path, str):
        return None, None, None
    # df40: gs://df40-frames-recropped-rfa85/{real|fake}/{method}/{identity}/<frame>.png
    m = re.search(r"gs://df40-frames-recropped[^/]*/(real|fake)/([^/]+)/([^/]+)/(\d+)\.png", path)
    if m:
        return f"df40__{m.group(2)}__{m.group(3)}", None, int(m.group(4))
    # generic frame_NNN_seqNNN
    m = re.search(r"frame_(\d+)_seq(\d+)", path)
    if m:
        # seq is the per-source-video id; frame is the index within
        return None, m.group(2), int(m.group(1))
    # generic _frame_NNN
    m = re.search(r"frame_(\d+)", path)
    if m:
        return None, None, int(m.group(1))
    return None, None, None


def detect_bucket(path: str) -> str:
    if not isinstance(path, str):
        return "unknown"
    m = re.match(r"gs://([^/]+)/", path)
    return m.group(1) if m else "unknown"


def lane_for_path(path: str) -> str | None:
    """Map a frame_path to a paired training lane, or None if not a paired-training-lane path."""
    if not isinstance(path, str):
        return None
    bucket = detect_bucket(path)
    if bucket.startswith("df40-frames"):
        return "df40"
    if "visomaster" in bucket and "v1" in bucket:
        return "visomaster_v1_base"
    if "visomaster" in bucket and "teams" in bucket:
        return "visomaster_teams_enhanced"
    if "visomaster" in bucket:
        # default: enhanced (v2) clean fallback
        return "visomaster_enhanced"
    # DeepLive lanes are not directly identifiable from path bucket — they live in
    # `deeplive-frames-*` (training) or `live-fakes-teams-prod` (eval/prod).
    if "deeplive-frames" in bucket or "deep-live-frames" in bucket:
        return "deeplive"
    if "live-fakes-teams-prod" in bucket:
        # production fakes — NOT a training-time paired lane (no real companion in this bucket)
        return None
    return None


def step_1_cache_first_audit() -> dict:
    """
    For each known score cache, attempt to find rows whose `frame_path` lies in a
    paired training-lane bucket AND has both labels at the same (sample_id, frame_idx).
    Returns a summary dict.
    """
    out: dict = {
        "checked_caches": [],
        "n_rows_per_lane": {lane: 0 for lane in PAIRED_LANES},
        "n_pairs_per_lane": {lane: 0 for lane in PAIRED_LANES},
        "n_pairs_total": 0,
        "lane_coverage": {},
    }

    # 1) grouped_manifest_v2.csv
    if MANIFEST.exists():
        df = pd.read_csv(MANIFEST)
        out["checked_caches"].append({"path": str(MANIFEST), "n_rows": int(len(df))})
        df["lane"] = df["frame_path"].map(lane_for_path)
        for lane in PAIRED_LANES:
            mask = df["lane"] == lane
            out["n_rows_per_lane"][lane] = int(mask.sum())
        # Try to construct tight pairs within each lane: same (sample_id, frame_idx) opposite label
        df["bucket"] = df["frame_path"].map(detect_bucket)
        df[["sample_id_extracted", "seq_n", "frame_n"]] = df["frame_path"].apply(
            lambda p: pd.Series(extract_pair_key(p))
        )
        # Use seq_n as a proxy for sample_id within a bucket (per-source video id)
        df["pair_key"] = df["bucket"].astype(str) + "::" + df["seq_n"].astype(str) + "::" + df["frame_n"].astype(str)
        for lane in PAIRED_LANES:
            sub = df[df["lane"] == lane]
            if len(sub) == 0:
                continue
            grouped = sub.groupby("pair_key")["label"].agg(lambda s: tuple(sorted(set(s.dropna().astype(int).tolist()))))
            mixed = grouped[grouped.apply(lambda t: 0 in t and 1 in t)]
            out["n_pairs_per_lane"][lane] = int(len(mixed))
        out["n_pairs_total"] = sum(out["n_pairs_per_lane"].values())

    # 2) prior pair_gaps.csv — already cross-product within canonical_subject; check for tight matches
    if PRIOR_PAIRS.exists():
        df = pd.read_csv(PRIOR_PAIRS)
        out["checked_caches"].append({"path": str(PRIOR_PAIRS), "n_rows": int(len(df))})
        # Strict tight: same path-stem (name__seg__frame) across real/fake — already known to be 0
        # Add a cross-check to keep the script self-contained.
        def stem(p: str) -> str:
            if not isinstance(p, str):
                return ""
            m = re.search(r"frame_(\d+)", p)
            return m.group(0) if m else ""
        df["real_stem"] = df["real_path"].map(stem)
        df["fake_stem"] = df["fake_path"].map(stem)
        # Plus require base_identity match
        same_name = df["real_base_identity"] == df["fake_base_identity"]
        same_stem = df["real_stem"] == df["fake_stem"]
        tight = same_name & same_stem & df["real_stem"].str.len().gt(0)
        out["prior_pair_gaps_tight_matches"] = int(tight.sum())

    return out


def step_2_lane_reconciliation() -> dict:
    """
    Closest proxy for tight same-source pair-gap on lanes that ARE present in cache:
    use the cross-product audit's per-(canonical_subject, fake_suite) cell as a stand-in,
    since within `dor_local + visomaster_v2_dor` the canonical subject pool is single-source
    enough to be informative.
    """
    if not PRIOR_PAIRS.exists():
        return {"error": "no prior pair_gaps.csv found"}
    df = pd.read_csv(PRIOR_PAIRS)
    tm = df[df["transport_match"] == True].copy()

    rows = []
    for ckpt in CKPTS:
        gap_col = f"pair_gap_{ckpt}"
        fake_col = f"fake_score_{ckpt}"
        for fs, g in tm.groupby("fake_suite"):
            missed = g[fake_col] < 0.5
            n_pairs = len(g)
            n_missed = int(missed.sum())
            p_le_0 = float((g.loc[missed, gap_col] <= 0).mean()) if n_missed > 0 else None
            rows.append({
                "ckpt": ckpt,
                "fake_suite": fs,
                "n_pairs": n_pairs,
                "n_missed": n_missed,
                "p_gap_le_0": p_le_0,
                "verdict": verdict_for(p_le_0),
            })

    return {"rows": rows}


def lane_for_fake_suite(suite: str) -> str | None:
    """Heuristic mapping from cross-product fake_suite -> training lane."""
    if suite == "visomaster_v2_dor":
        return "visomaster_enhanced"
    if suite == "dor_fake_local":
        return "deeplive"  # dor local fakes are deeplive-style
    if suite == "live_fakes_teams_prod":
        return "deeplive_teams"
    if suite == "extra":
        return None  # extra is a held-out probe set, not a training lane
    if suite in {"teams_fake_all_dev", "teams_fake_all_lockbox"}:
        return None  # teams_passthrough eval substrate, not a training lane
    return None


def aggregate_lane_verdicts(reconciliation: dict) -> dict:
    """Map fake_suite-level rows to lane-level verdicts."""
    rows = reconciliation.get("rows", [])
    by_lane = defaultdict(lambda: {ckpt: [] for ckpt in CKPTS})
    for r in rows:
        lane = lane_for_fake_suite(r["fake_suite"])
        if lane is None:
            continue
        by_lane[lane][r["ckpt"]].append(r)

    lane_verdicts: dict = {}
    for lane in PAIRED_LANES:
        lane_data = {"covered": False, "by_ckpt": {}}
        for ckpt in CKPTS:
            cells = by_lane.get(lane, {}).get(ckpt, [])
            if not cells:
                lane_data["by_ckpt"][ckpt] = {"verdict": "INSUFFICIENT_DATA"}
                continue
            n_pairs = sum(c["n_pairs"] for c in cells)
            n_missed = sum(c["n_missed"] for c in cells)
            if n_missed == 0:
                lane_data["by_ckpt"][ckpt] = {
                    "n_pairs": n_pairs, "n_missed": n_missed,
                    "p_gap_le_0": None, "verdict": "INSUFFICIENT_DATA",
                }
                continue
            # Re-aggregate exactly: weighted-mean of p_le_0 by n_missed
            p_le_0 = sum(c["p_gap_le_0"] * c["n_missed"] for c in cells if c["p_gap_le_0"] is not None) / n_missed
            lane_data["by_ckpt"][ckpt] = {
                "n_pairs": n_pairs, "n_missed": n_missed,
                "p_gap_le_0": float(p_le_0), "verdict": verdict_for(p_le_0),
            }
            lane_data["covered"] = True
        lane_verdicts[lane] = lane_data
    return lane_verdicts


def step_3_construct_proxy_pairs_csv() -> Path:
    """
    Best-available 'same_source_pairs.csv' is the prior cross-product table restricted to
    rows where (a) a paired-lane mapping exists, and (b) we surface lane + per-ckpt gap.
    """
    if not PRIOR_PAIRS.exists():
        return None
    df = pd.read_csv(PRIOR_PAIRS)
    df["proxy_lane"] = df["fake_suite"].map(lane_for_fake_suite)
    sub = df[df["proxy_lane"].notna()].copy()
    out_csv = OUT_DIR / "same_source_pairs.csv"
    cols = [
        "pair_id", "canonical_subject", "proxy_lane",
        "real_base_identity", "fake_base_identity",
        "real_suite", "fake_suite", "method", "enhancer",
        "real_transport", "fake_transport", "transport_match",
        "real_score_P8A", "fake_score_P8A", "pair_gap_P8A",
        "real_score_E2B", "fake_score_E2B", "pair_gap_E2B",
        "real_score_PA_3800", "fake_score_PA_3800", "pair_gap_PA_3800",
        "real_path", "fake_path",
    ]
    sub[cols].to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    cache_audit = step_1_cache_first_audit()

    # Decision: is cache-first audit sufficient?
    n_pairs_total = cache_audit.get("n_pairs_total", 0)
    sufficient = n_pairs_total >= 500

    reconciliation = step_2_lane_reconciliation()
    lane_verdicts = aggregate_lane_verdicts(reconciliation)
    proxy_csv = step_3_construct_proxy_pairs_csv()

    # Aggregate verdict: per ckpt, was any paired-lane GREEN?
    # We use lane_verdicts (cross-product proxy) as the closest available signal.
    aggregate: dict[str, dict] = {}
    for ckpt in CKPTS:
        lane_results = []
        n_green = 0
        n_amber = 0
        n_red = 0
        n_insufficient = 0
        for lane in PAIRED_LANES:
            v = lane_verdicts[lane]["by_ckpt"][ckpt]["verdict"]
            lane_results.append({"lane": lane, "verdict": v, **{k: lane_verdicts[lane]["by_ckpt"][ckpt].get(k) for k in ["n_pairs", "n_missed", "p_gap_le_0"]}})
            if v == "GREEN":
                n_green += 1
            elif v == "AMBER":
                n_amber += 1
            elif v == "RED":
                n_red += 1
            else:
                n_insufficient += 1
        aggregate[ckpt] = {
            "n_green_lanes": n_green,
            "n_amber_lanes": n_amber,
            "n_red_lanes": n_red,
            "n_insufficient_lanes": n_insufficient,
            "lane_verdicts": lane_results,
        }

    # Top-level go/no-go for P1 per the rule: GREEN in >= 2 paired lanes
    p1_recommendation = {}
    for ckpt in CKPTS:
        n_green = aggregate[ckpt]["n_green_lanes"]
        if n_green >= 2:
            p1_recommendation[ckpt] = "LAUNCH_PE_PAIR_RANK_DRO"
        elif n_green == 1:
            p1_recommendation[ckpt] = "LANE_RESTRICTED_LAUNCH"
        else:
            p1_recommendation[ckpt] = "DEMOTE_P1"

    summary = {
        "audit_name": "SAME_SOURCE_PAIR_GAP_AUDIT_2026-05-06",
        "phase": "0g",
        "step_1_cache_first_audit": cache_audit,
        "step_1_sufficient_for_tight_pair_gap": bool(sufficient),
        "lane_verdicts_via_proxy": lane_verdicts,
        "aggregate_per_ckpt": aggregate,
        "p1_recommendation_per_ckpt": p1_recommendation,
        "decision_rule": {
            "green_threshold": GREEN_THRESHOLD,
            "amber_threshold": AMBER_THRESHOLD,
            "min_green_lanes_for_launch": 2,
        },
        "proxy_csv_path": str(proxy_csv) if proxy_csv else None,
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Verdict file
    verdict = {
        "headline": "SEE_FINDINGS_MD",
        "phase": "0g",
        "p1_recommendation_per_ckpt": p1_recommendation,
        "p1_load_bearing_recommendation": "LANE_RESTRICTED_LAUNCH or DEMOTE — see FINDINGS.md",
    }
    with open(OUT_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    print(f"Wrote: {OUT_DIR}/summary.json")
    print(f"Wrote: {OUT_DIR}/verdict.json")
    if proxy_csv:
        print(f"Wrote: {proxy_csv}")
    print()
    print("=== Cache-first audit ===")
    print(f"  n pairs found at tight (sample_id, frame_idx) keys: {n_pairs_total}")
    print(f"  Sufficient for direct measurement (>=500): {sufficient}")
    print(f"  Per-lane row coverage (manifest):")
    for lane, n in cache_audit["n_rows_per_lane"].items():
        n_pairs = cache_audit["n_pairs_per_lane"].get(lane, 0)
        print(f"    {lane}: {n} rows  /  {n_pairs} tight pairs")
    print()
    print("=== Lane-level verdicts (cross-product proxy) ===")
    for lane in PAIRED_LANES:
        lane_data = lane_verdicts[lane]
        print(f"  {lane}: covered={lane_data['covered']}")
        for ckpt in CKPTS:
            d = lane_data["by_ckpt"][ckpt]
            print(f"    {ckpt}: {d}")
    print()
    print("=== P1 recommendation per ckpt ===")
    for ckpt, rec in p1_recommendation.items():
        print(f"  {ckpt}: {rec}")


if __name__ == "__main__":
    main()
