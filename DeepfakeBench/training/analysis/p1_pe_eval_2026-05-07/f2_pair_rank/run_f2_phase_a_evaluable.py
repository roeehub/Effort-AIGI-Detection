"""F2 reformulated to be Phase-A-evaluable.

Original F2 (per `R13_P1_BUNDLE_FT_FROM_P8A.yaml` and
`NEXT_STEPS_PLAN_2026-05-06.md` §8.2):
  ">=30% relative pair-rank lift on >=2 of 6 paired training lanes among
   previously-missed fakes"

That criterion was not testable on Phase A substrate because 5/6
yaml-named paired training lanes have zero coverage in Phase A, and the
one (deeplive_teams) that does is saturated at P8A baseline frac >=
0.97 across every active sub-lane. See:
  analysis/p1_pe_eval_2026-05-07/f2_pair_rank/F2_PAIR_RANK_FACTS_2026-05-07.md

Reformulation tested here (P8A-gated, suite-axis):
  ">=30% relative lift on per-frame pair-gap probability
   P(fake_score > real_score) on (fake, real) pairs where P8A's
   fake_score on the fake frame is < 0.5 (P8A confident-wrong on the
   fake)."

Pairing convention here:
  - For each fake suite F, identify the P8A-missed fake frames in F
    (fake frames where P8A's frame_prob < 0.5).
  - The real pool is the union of all real-suite frames available in
    Phase A (all label==0 frames, dev + lockbox).
  - For each missed fake frame f, the per-frame "pair-gap probability"
    is mean over reals r of 1[score(f) > score(r)]. This is the
    standard unbiased P(fake>real | missed) estimator using all
    available reals.
  - The suite-level metric is the mean of that per-frame quantity over
    all P8A-missed fake frames in F.
  - For each ckpt c (including P8A), this gives one number per
    (suite, ckpt). Lift_rel_pct(c, F) = (val_c - val_p8a) / val_p8a *
    100.
  - Pass per ckpt: lift_rel_pct >= 30 on >=2 of N evaluable fake suites
    (suites where val_p8a > 0 and number of P8A-missed frames > 0).

Eval-substrate fake suite axis (top-level, non-overlapping per fake
file scope):
  1. deeplive_enhanced_dev
  2. teams_fake_all_dev
  3. teams_fake_all_lockbox
  4. teams_flat_xiang_xiang2_feng_dev
  5. visomaster_enhanced_macro_dev

Per-subject teams_capture_*_dev and teams_capture_*_s*_dev suites are
subsets of teams_fake_all_dev / teams_fake_all_lockbox by frame_path
and are NOT counted as separate suites here (would double-count).

Output:
  - pairing.csv: long-form per-(fake_path, ckpt) per-suite frame-level
    pair-gap probability values (one row per (suite, fake_path, ckpt)).
  - per_suite_per_ckpt_lift.csv: per-(suite, ckpt) summary (n_missed,
    val_pair_gap_prob, lift_abs, lift_rel_pct, passes_30pct).
  - pass_summary.csv: per-ckpt count of evaluable suites passing 30pct
    bar.

CPU only. No randomization (deterministic given the inputs).
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
ALL_CKPTS = [BASELINE_CKPT] + P1_CKPTS

# Real-suite scopes (label==0 frame_path domains). Lockbox + dev
# combined. These are the same Phase A substrates the P1 packet was
# scored against.
REAL_SUITES = [
    "teams_real_all_dev",
    "teams_real_all_lockbox",
    "teams_real_dor_dev",
    "teams_real_lighting_extreme_dev",
    "teams_real_lighting_extreme_lockbox",
    "teams_real_poor_quality_dev",
    "teams_real_poor_quality_lockbox",
]

# Fake-suite axis: top-level non-overlapping suites scored on Phase A.
# Per-subject teams_capture_*_dev variants are subsets of
# teams_fake_all_dev (verified by frame_path overlap) and would
# duplicate-count if included.
FAKE_SUITES = [
    "deeplive_enhanced_dev",
    "teams_fake_all_dev",
    "teams_fake_all_lockbox",
    "teams_flat_xiang_xiang2_feng_dev",
    "visomaster_enhanced_macro_dev",
]

# Threshold that defines "previously missed" by P8A.
P8A_MISSED_THRESHOLD = 0.5


def load_frame_scores(report_path: Path) -> Dict[str, float]:
    """Load frame_path -> frame_prob from a Phase A frames_report.csv."""
    out: Dict[str, float] = {}
    if not report_path.exists():
        return out
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


def load_suite_scores(suite: str, ckpt: str) -> Dict[str, float]:
    """Load frame_path -> frame_prob for a (suite, ckpt) pair."""
    rp = PHASE_A_DIR / f"{suite}_{ckpt}_frames_report.csv"
    return load_frame_scores(rp)


def load_real_pool_scores(ckpt: str) -> Dict[str, float]:
    """Union of all real suites' frame_path -> frame_prob for ckpt."""
    maps = [load_suite_scores(s, ckpt) for s in REAL_SUITES]
    return merge_score_maps(*maps)


def compute_pair_gap_per_frame(
    fake_paths: Iterable[str],
    fake_scores: Dict[str, float],
    real_scores_sorted: List[float],
) -> List[Tuple[str, float, int, int]]:
    """For each fake_path, compute mean(1[fake_score > real_score]) over
    the entire real pool. Returns list of (fake_path, frac, n_real_lt,
    n_real_total)."""
    import bisect

    out = []
    n_real = len(real_scores_sorted)
    for fp in fake_paths:
        fs = fake_scores.get(fp)
        if fs is None or n_real == 0:
            continue
        # bisect_left returns count of reals strictly < fs
        n_real_lt = bisect.bisect_left(real_scores_sorted, fs)
        # If there are ties at fs, count them as not-greater (strict >).
        # bisect_right - bisect_left = count equal to fs.
        n_eq = bisect.bisect_right(real_scores_sorted, fs) - n_real_lt
        # strict gt count = bisect_left (reals < fs)
        n_gt_strict = n_real_lt
        # mean = n_gt_strict / n_real
        out.append((fp, n_gt_strict / n_real, n_real_lt, n_real))
        _ = n_eq  # not used but kept for clarity
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading P8A baseline scores...", file=sys.stderr)
    p8a_real = load_real_pool_scores(BASELINE_CKPT)
    print(
        f"  P8A real pool size: {len(p8a_real)} unique frame_paths",
        file=sys.stderr,
    )

    p8a_fake_per_suite = {
        s: load_suite_scores(s, BASELINE_CKPT) for s in FAKE_SUITES
    }
    for s, m in p8a_fake_per_suite.items():
        print(f"  P8A fake suite {s}: {len(m)} frames", file=sys.stderr)

    # Identify P8A-missed fake_paths per fake suite.
    print(f"[2/5] Filtering P8A-missed fakes (frame_prob < {P8A_MISSED_THRESHOLD})...",
          file=sys.stderr)
    missed_per_suite: Dict[str, List[str]] = {}
    for s in FAKE_SUITES:
        missed = [fp for fp, score in p8a_fake_per_suite[s].items()
                  if score < P8A_MISSED_THRESHOLD]
        missed_per_suite[s] = sorted(missed)
        print(f"  {s}: {len(missed)} / {len(p8a_fake_per_suite[s])} P8A-missed",
              file=sys.stderr)

    # For each ckpt, load real-pool scores and per-fake-suite scores;
    # compute the per-frame pair-gap prob over the real pool restricted
    # to the missed-fake set per suite.
    print("[3/5] Computing per-frame pair-gap probability per ckpt × suite...",
          file=sys.stderr)
    pairing_rows = []  # long-form (suite, ckpt, fake_path, frac, n_real)
    per_ckpt_per_suite_summary: Dict[Tuple[str, str], dict] = {}

    for ckpt in ALL_CKPTS:
        print(f"  ckpt={ckpt}", file=sys.stderr)
        real_scores = load_real_pool_scores(ckpt)
        # Sort real scores for fast count via bisect.
        real_scores_sorted = sorted(real_scores.values())
        n_real = len(real_scores_sorted)
        for suite in FAKE_SUITES:
            fake_scores = load_suite_scores(suite, ckpt)
            missed = missed_per_suite[suite]
            per_frame = compute_pair_gap_per_frame(
                missed, fake_scores, real_scores_sorted
            )
            for fp, frac, n_lt, _n_real in per_frame:
                pairing_rows.append({
                    "suite": suite,
                    "ckpt": ckpt,
                    "fake_path": fp,
                    "fake_score": f"{fake_scores[fp]:.6f}",
                    "frac_real_below_fake": f"{frac:.6f}",
                    "n_real_below_strict": n_lt,
                    "n_real_pool": _n_real,
                })
            n = len(per_frame)
            mean_frac = (sum(t[1] for t in per_frame) / n) if n > 0 else float("nan")
            per_ckpt_per_suite_summary[(suite, ckpt)] = {
                "n_missed_p8a_fakes": len(missed),
                "n_scored": n,
                "mean_pair_gap_prob": mean_frac,
                "n_real_pool": n_real,
            }

    # Write pairing CSV.
    pairing_csv = OUT_DIR / "phase_a_evaluable_pairing.csv"
    with pairing_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "suite", "ckpt", "fake_path", "fake_score",
                "frac_real_below_fake", "n_real_below_strict", "n_real_pool",
            ],
        )
        writer.writeheader()
        for r in pairing_rows:
            writer.writerow(r)
    print(f"[4/5] Wrote {pairing_csv}", file=sys.stderr)

    # Per-suite × per-ckpt summary with lift vs P8A baseline.
    print("[5/5] Summarizing lifts...", file=sys.stderr)
    summary_rows = []
    for suite in FAKE_SUITES:
        p8a_val = per_ckpt_per_suite_summary[(suite, BASELINE_CKPT)]["mean_pair_gap_prob"]
        for ckpt in ALL_CKPTS:
            row = per_ckpt_per_suite_summary[(suite, ckpt)]
            val = row["mean_pair_gap_prob"]
            # lift only meaningful if both are finite and p8a_val > 0
            if (val != val) or (p8a_val != p8a_val):
                lift_abs = float("nan")
                lift_rel = float("nan")
            elif p8a_val == 0:
                lift_abs = val - p8a_val
                lift_rel = float("inf") if val > 0 else 0.0
            else:
                lift_abs = val - p8a_val
                lift_rel = (val - p8a_val) / p8a_val * 100.0
            passes_30 = (lift_rel == lift_rel) and (
                lift_rel >= 30.0 if lift_rel != float("inf") else True
            )
            summary_rows.append({
                "suite": suite,
                "ckpt": ckpt,
                "n_missed_p8a_fakes": row["n_missed_p8a_fakes"],
                "n_scored": row["n_scored"],
                "n_real_pool": row["n_real_pool"],
                "mean_pair_gap_prob": (
                    f"{val:.6f}" if val == val else "nan"
                ),
                "lift_abs": (
                    f"{lift_abs:.6f}" if lift_abs == lift_abs else "nan"
                ),
                "lift_rel_pct": (
                    "inf" if lift_rel == float("inf")
                    else (f"{lift_rel:.6f}" if lift_rel == lift_rel else "nan")
                ),
                "passes_30pct": "yes" if (
                    ckpt != BASELINE_CKPT and passes_30
                ) else "no",
            })

    summary_csv = OUT_DIR / "phase_a_evaluable_per_suite_per_ckpt_lift.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "suite", "ckpt", "n_missed_p8a_fakes", "n_scored",
                "n_real_pool", "mean_pair_gap_prob", "lift_abs",
                "lift_rel_pct", "passes_30pct",
            ],
        )
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print(f"  Wrote {summary_csv}", file=sys.stderr)

    # Pass-summary per ckpt: count evaluable suites where lift_rel_pct >=
    # 30. Evaluable = n_missed > 0 and p8a_val > 0.
    pass_summary = {}
    n_evaluable = 0
    for suite in FAKE_SUITES:
        p8a_row = per_ckpt_per_suite_summary[(suite, BASELINE_CKPT)]
        if p8a_row["n_missed_p8a_fakes"] > 0 and p8a_row["mean_pair_gap_prob"] != 0 and p8a_row["mean_pair_gap_prob"] == p8a_row["mean_pair_gap_prob"]:
            n_evaluable += 1

    for ckpt in P1_CKPTS:
        n_pass = 0
        for suite in FAKE_SUITES:
            p8a_row = per_ckpt_per_suite_summary[(suite, BASELINE_CKPT)]
            ck_row = per_ckpt_per_suite_summary[(suite, ckpt)]
            if p8a_row["n_missed_p8a_fakes"] == 0:
                continue
            p8a_val = p8a_row["mean_pair_gap_prob"]
            ck_val = ck_row["mean_pair_gap_prob"]
            if p8a_val != p8a_val or ck_val != ck_val:
                continue
            if p8a_val == 0:
                if ck_val > 0:
                    n_pass += 1
                continue
            lift_rel = (ck_val - p8a_val) / p8a_val * 100.0
            if lift_rel >= 30.0:
                n_pass += 1
        pass_summary[ckpt] = {
            "n_pass_30pct_relative": n_pass,
            "n_evaluable_suites": n_evaluable,
        }

    pass_csv = OUT_DIR / "phase_a_evaluable_pass_summary.csv"
    with pass_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ckpt", "n_pass_30pct_relative", "n_evaluable_suites",
                        "passes_2_of_N"],
        )
        writer.writeheader()
        for ckpt, s in pass_summary.items():
            writer.writerow({
                "ckpt": ckpt,
                "n_pass_30pct_relative": s["n_pass_30pct_relative"],
                "n_evaluable_suites": s["n_evaluable_suites"],
                "passes_2_of_N": (
                    "yes" if s["n_pass_30pct_relative"] >= 2 else "no"
                ),
            })
    print(f"  Wrote {pass_csv}", file=sys.stderr)

    print("\n=== F2 Phase-A-evaluable PASS SUMMARY (>=30%% relative lift, >=2 evaluable suites) ===")
    print(f"{'ckpt':<32}  {'n_pass':>7}  {'n_eval':>7}  {'>=2-of-N':>10}")
    for ckpt, s in pass_summary.items():
        flag = "yes" if s["n_pass_30pct_relative"] >= 2 else "no"
        print(f"{ckpt:<32}  {s['n_pass_30pct_relative']:>7}  {s['n_evaluable_suites']:>7}  {flag:>10}")


if __name__ == "__main__":
    main()
