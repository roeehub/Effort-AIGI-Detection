"""F2 pair-rank close criterion computation.

Pair source: analysis/pair_gap_audit_2026-05-06/outputs/pair_gaps.csv,
filtered to eval-substrate cross-product rows (real_suite in
{teams_real_all_dev, teams_real_all_lockbox}; fake_suite in
{teams_fake_all_dev, teams_fake_all_lockbox}).

Score source: analysis/p1_pe_eval_2026-05-07/raw_reports/phase_a/<suite>_<ckpt>_frames_report.csv

Pairing convention: frame-level cross-product within canonical_subject
(per pair_gap_audit). Each canonical_subject is treated as a "lane".
Mapping to the 6 paired training lanes is incomplete — see the
caveats section in the markdown report.

Metric:
  frac_fake_gt_real per (lane, ckpt) = mean(1[fake_score > real_score])
  lift_abs = frac_p1 - frac_p8a
  lift_rel_pct = (frac_p1 - frac_p8a) / frac_p8a * 100
  pass_30pct = lift_rel_pct >= 30.0

"Previously-missed fakes" filter: fake's P8A frame_prob < 0.5.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
)
PHASE_A_DIR = REPO_ROOT / "analysis/p1_pe_eval_2026-05-07/raw_reports/phase_a"
OUT_DIR = REPO_ROOT / "analysis/p1_pe_eval_2026-05-07/f2_pair_rank"
PAIR_GAP_CSV = REPO_ROOT / "analysis/pair_gap_audit_2026-05-06/outputs/pair_gaps.csv"

# All P1 ckpts we are comparing vs P8A baseline.
P1_CKPTS = [
    "p1_bundle_periodic_step500",
    "p1_bundle_top_n_step3750",
    "p1_bundle_top_n_step4000",
    "p1_pairrank_periodic_step500",
    "p1_pairrank_top_n_step6000",
    "p1_pairrank_top_n_step6750",
]
BASELINE_CKPT = "p8a_reference_step5000"


def load_frame_scores(report_path: Path) -> Dict[str, float]:
    """Load frame_path -> frame_prob from a Phase A frames_report.csv."""
    out: Dict[str, float] = {}
    with report_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            fp = row["frame_path"]
            try:
                out[fp] = float(row["frame_prob"])
            except (KeyError, ValueError):
                continue
    return out


def merge_score_maps(*maps: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for m in maps:
        out.update(m)
    return out


def load_phase_a_for_ckpt(ckpt: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return (real_scores, fake_scores) for the eval-substrate teams suites
    used by the pair manifest.

    Real suites: teams_real_all_dev, teams_real_all_lockbox.
    Fake suites: teams_fake_all_dev, teams_fake_all_lockbox.
    """
    real_dev = PHASE_A_DIR / f"teams_real_all_dev_{ckpt}_frames_report.csv"
    real_lb = PHASE_A_DIR / f"teams_real_all_lockbox_{ckpt}_frames_report.csv"
    fake_dev = PHASE_A_DIR / f"teams_fake_all_dev_{ckpt}_frames_report.csv"
    fake_lb = PHASE_A_DIR / f"teams_fake_all_lockbox_{ckpt}_frames_report.csv"
    return (
        merge_score_maps(load_frame_scores(real_dev), load_frame_scores(real_lb)),
        merge_score_maps(load_frame_scores(fake_dev), load_frame_scores(fake_lb)),
    )


def load_pairs() -> List[dict]:
    """Load pair manifest restricted to eval-substrate cross-product rows."""
    out = []
    keep_real = {"teams_real_all_dev", "teams_real_all_lockbox"}
    keep_fake = {"teams_fake_all_dev", "teams_fake_all_lockbox"}
    with PAIR_GAP_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["real_suite"] in keep_real and row["fake_suite"] in keep_fake:
                out.append(row)
    return out


def compute_per_lane(
    pairs: List[dict],
    real_scores: Dict[str, float],
    fake_scores: Dict[str, float],
    p8a_real_scores: Dict[str, float],
    p8a_fake_scores: Dict[str, float],
) -> Dict[str, dict]:
    """Compute (n_pairs, n_fake_gt_real) per lane (canonical_subject).

    Restricts to "previously-missed fakes": those where the fake_path's
    P8A frame_prob < 0.5 (P8A reference scores). The lane membership and
    pair manifest is held constant; only the threshold is fixed by P8A's
    decision.
    """
    per_lane = defaultdict(lambda: {"n_pairs": 0, "n_fake_gt_real": 0, "n_unscored": 0})
    for row in pairs:
        lane = row["canonical_subject"]
        rp = row["real_path"]
        fp = row["fake_path"]
        # Filter: previously-missed fake = P8A fake score < 0.5
        p8a_fp_score = p8a_fake_scores.get(fp)
        if p8a_fp_score is None:
            per_lane[lane]["n_unscored"] += 1
            continue
        if p8a_fp_score >= 0.5:
            continue  # not previously missed
        rp_score = real_scores.get(rp)
        fp_score = fake_scores.get(fp)
        if rp_score is None or fp_score is None:
            per_lane[lane]["n_unscored"] += 1
            continue
        per_lane[lane]["n_pairs"] += 1
        if fp_score > rp_score:
            per_lane[lane]["n_fake_gt_real"] += 1
    return per_lane


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading pair manifest...", file=sys.stderr)
    pairs = load_pairs()
    print(f"  Loaded {len(pairs)} eval-substrate cross-product pairs", file=sys.stderr)

    print("[2/4] Loading P8A baseline scores...", file=sys.stderr)
    p8a_real, p8a_fake = load_phase_a_for_ckpt(BASELINE_CKPT)
    print(
        f"  P8A reals scored: {len(p8a_real)}; fakes scored: {len(p8a_fake)}",
        file=sys.stderr,
    )

    # Compute P8A baseline per-lane
    p8a_per_lane = compute_per_lane(pairs, p8a_real, p8a_fake, p8a_real, p8a_fake)

    print("[3/4] Loading P1 ckpts and computing per-lane lift...", file=sys.stderr)
    rows = []
    # Always emit P8A reference rows first
    for lane, stats in sorted(p8a_per_lane.items()):
        n_pairs = stats["n_pairs"]
        n_fake_gt_real = stats["n_fake_gt_real"]
        if n_pairs == 0:
            frac = float("nan")
        else:
            frac = n_fake_gt_real / n_pairs
        rows.append({
            "lane": lane,
            "ckpt": BASELINE_CKPT,
            "n_pairs": n_pairs,
            "n_fake_gt_real": n_fake_gt_real,
            "frac_fake_gt_real": f"{frac:.6f}" if n_pairs > 0 else "nan",
            "lift_abs": "0.000000",
            "lift_rel_pct": "0.000000",
        })

    p1_per_ckpt_per_lane: Dict[str, Dict[str, dict]] = {}
    for ckpt in P1_CKPTS:
        real, fake = load_phase_a_for_ckpt(ckpt)
        per_lane = compute_per_lane(pairs, real, fake, p8a_real, p8a_fake)
        p1_per_ckpt_per_lane[ckpt] = per_lane
        for lane, stats in sorted(per_lane.items()):
            n_pairs = stats["n_pairs"]
            n_fake_gt_real = stats["n_fake_gt_real"]
            p8a_stats = p8a_per_lane.get(lane, {"n_pairs": 0, "n_fake_gt_real": 0})
            p8a_n = p8a_stats["n_pairs"]
            if n_pairs == 0 or p8a_n == 0:
                frac = float("nan") if n_pairs == 0 else n_fake_gt_real / n_pairs
                lift_abs = float("nan")
                lift_rel = float("nan")
            else:
                frac = n_fake_gt_real / n_pairs
                p8a_frac = p8a_stats["n_fake_gt_real"] / p8a_n
                lift_abs = frac - p8a_frac
                if p8a_frac == 0:
                    lift_rel = float("inf") if frac > 0 else 0.0
                else:
                    lift_rel = (frac - p8a_frac) / p8a_frac * 100.0
            rows.append({
                "lane": lane,
                "ckpt": ckpt,
                "n_pairs": n_pairs,
                "n_fake_gt_real": n_fake_gt_real,
                "frac_fake_gt_real": f"{frac:.6f}" if n_pairs > 0 else "nan",
                "lift_abs": f"{lift_abs:.6f}" if lift_abs == lift_abs else "nan",
                "lift_rel_pct": (
                    f"{lift_rel:.6f}" if lift_rel == lift_rel and lift_rel != float("inf")
                    else ("inf" if lift_rel == float("inf") else "nan")
                ),
            })

    out_csv = OUT_DIR / "per_lane_per_ckpt_lift.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lane", "ckpt", "n_pairs", "n_fake_gt_real",
                "frac_fake_gt_real", "lift_abs", "lift_rel_pct",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[4/4] Wrote {out_csv}", file=sys.stderr)

    # Pass count per ckpt @ 30% relative-lift bar
    pass_summary = {}
    for ckpt in P1_CKPTS:
        n_pass = 0
        n_total_lanes_with_data = 0
        per_lane = p1_per_ckpt_per_lane[ckpt]
        for lane, stats in per_lane.items():
            n_pairs = stats["n_pairs"]
            p8a_stats = p8a_per_lane.get(lane, {"n_pairs": 0, "n_fake_gt_real": 0})
            p8a_n = p8a_stats["n_pairs"]
            if n_pairs == 0 or p8a_n == 0:
                continue
            n_total_lanes_with_data += 1
            frac = stats["n_fake_gt_real"] / n_pairs
            p8a_frac = p8a_stats["n_fake_gt_real"] / p8a_n
            if p8a_frac == 0:
                if frac > 0:
                    n_pass += 1  # infinite lift
                continue
            lift_rel = (frac - p8a_frac) / p8a_frac * 100.0
            if lift_rel >= 30.0:
                n_pass += 1
        pass_summary[ckpt] = {
            "n_pass": n_pass,
            "n_lanes_with_data": n_total_lanes_with_data,
        }

    pass_csv = OUT_DIR / "pass_summary.csv"
    with pass_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ckpt", "n_pass_30pct_relative", "n_lanes_with_data"],
        )
        writer.writeheader()
        for ckpt, s in pass_summary.items():
            writer.writerow({
                "ckpt": ckpt,
                "n_pass_30pct_relative": s["n_pass"],
                "n_lanes_with_data": s["n_lanes_with_data"],
            })
    print(f"Wrote {pass_csv}", file=sys.stderr)

    # Print summary to stdout
    print("\n=== F2 PASS SUMMARY (30% relative-lift bar) ===")
    print(f"{'ckpt':<32}  {'n_pass':>7}  {'n_lanes_with_data':>18}")
    for ckpt, s in pass_summary.items():
        print(f"{ckpt:<32}  {s['n_pass']:>7}  {s['n_lanes_with_data']:>18}")


if __name__ == "__main__":
    main()
