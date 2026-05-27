"""Slot 1 (LoRA-P8A) score-shift profile + τ-recalibration sweep.

For the 9 contract suites scored by the in-flight r13-overnight scorecard,
analyze whether Slot 1 LoRA ckpts apply a uniform additive shift over P8A
(simple τ-fix viable) or an asymmetric shift across substrates (head-retrain
needed).

Inputs:
  ./_frame_cache/<suite>_<ckpt>_frames_report.csv  (frame-level prob_fake)
  ../r13_overnight_partial_scorecard_2026-05-13/_reports_cache/<suite>_<ckpt>_videos_report.csv
   (video-level avg_video_prob; used for τ calibration matching the scorer)

Outputs:
  ./_shift_profile_<date>.csv         per-suite mean-prob deltas
  ./_tau_recal_<date>.csv             τ-sweep on Slot 1 top_n_step2000
  ./_per_identity_lockbox_<date>.csv  identity-level lockbox FPR breakdown
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent
FRAME_CACHE = HERE / "_frame_cache"
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

CKPTS = [
    "p8a_reference_step5000",
    "slot1_lora_p8a_periodic_step1500",
    "slot1_lora_p8a_periodic_step3500",
    "slot1_lora_p8a_top_n_step2000",
]

# Pre-computed τ from partial scorecard
TAU_CAL = {
    "p8a_reference_step5000": 0.9143,
    "slot1_lora_p8a_periodic_step1500": 0.4543,
    "slot1_lora_p8a_periodic_step3500": 0.4448,
    "slot1_lora_p8a_top_n_step2000": 0.4496,
}


def load_frames(suite: str, ckpt: str) -> Optional[list[dict]]:
    path = FRAME_CACHE / f"{suite}_{ckpt}_frames_report.csv"
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "video_id": r["video_id"],
                "label": int(r["label"]),
                "frame_path": r["frame_path"],
                "frame_prob": float(r["frame_prob"]),
            })
    return rows


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

    Strategy: identity = everything before the first `__sNN` (session id) or
    `__seqNNNN` token, or before the first `__seg_` token.
    """
    # Split on the substrings that mark non-identity tokens
    # Common patterns:
    # - dor_shkedi__seq0007__real
    # - Cam_Test__s32__seg_309.0__real
    # - Chikara_Takahashi__s22__seg_10.0__real
    parts = video_id.split("__")
    if len(parts) >= 2:
        return parts[0]
    return video_id


# -----------------------------------------------------------------------------
# Job 1: per-suite score-shift profile
# -----------------------------------------------------------------------------

def job1_shift_profile() -> dict:
    """Compute mean frame_prob per (suite × ckpt), split by label.

    Output table: rows = suites, cols = (P8A mean | slot1_step1500 mean |
    slot1_step3500 mean | slot1_top_n_step2000 mean | delta for each Slot 1
    vs P8A).
    """
    out = {}
    for suite in CONTRACT_SUITES:
        out[suite] = {}
        for ckpt in CKPTS:
            rows = load_frames(suite, ckpt)
            if rows is None:
                out[suite][ckpt] = {"mean_all": None, "mean_real": None, "mean_fake": None, "n": 0}
                continue
            all_probs = [r["frame_prob"] for r in rows]
            real_probs = [r["frame_prob"] for r in rows if r["label"] == 0]
            fake_probs = [r["frame_prob"] for r in rows if r["label"] == 1]
            out[suite][ckpt] = {
                "mean_all": sum(all_probs) / len(all_probs),
                "mean_real": sum(real_probs) / len(real_probs) if real_probs else None,
                "mean_fake": sum(fake_probs) / len(fake_probs) if fake_probs else None,
                "n": len(rows),
                "n_real": len(real_probs),
                "n_fake": len(fake_probs),
            }
    return out


# -----------------------------------------------------------------------------
# Job 2: τ-recalibration sweep on Slot 1 top_n_step2000
# -----------------------------------------------------------------------------

def fpr_at_tau(rows: list[dict], tau: float, prob_key: str = "avg_video_prob") -> float:
    reals = [r for r in rows if r["label"] == 0]
    if not reals:
        return float("nan")
    return sum(1 for r in reals if r[prob_key] >= tau) / len(reals)


def recall_at_tau(rows: list[dict], tau: float, prob_key: str = "avg_video_prob") -> float:
    fakes = [r for r in rows if r["label"] == 1]
    if not fakes:
        return float("nan")
    return sum(1 for r in fakes if r[prob_key] >= tau) / len(fakes)


def job2_tau_sweep(ckpt: str = "slot1_lora_p8a_top_n_step2000") -> dict:
    """Sweep τ values and report key metrics. Uses video-level avg_video_prob
    to match the scorer.

    τ range chosen to bracket Slot 1's dev-cal τ (~0.45) upward to
    near-deployment-comparable (0.99).
    """
    # Reasonable sweep covering Slot 1 calibrated τ (0.45) up through 0.99
    # Dense around the bimodal pile-up at ~0.45-0.50 (sigmoid saturation).
    tau_grid = [
        0.30, 0.35, 0.40,
        0.43, 0.44, 0.445, 0.4496, 0.450, 0.455, 0.46, 0.465, 0.47, 0.475,
        0.48, 0.485, 0.49, 0.495, 0.499, 0.50, 0.501, 0.505, 0.51, 0.52,
        0.55, 0.60, 0.70, 0.80, 0.90, 0.9143, 0.92, 0.94, 0.96, 0.98, 0.99,
    ]

    results = []

    # Load relevant per-suite video CSVs once
    suite_rows = {}
    for s in ["teams_real_all_dev", "teams_real_all_lockbox", "teams_fake_all_dev",
              "teams_fake_all_lockbox", "visomaster_enhanced_macro_dev",
              "deeplive_enhanced_dev", "teams_real_dor_dev",
              "teams_real_poor_quality_dev", "teams_real_lighting_extreme_dev"]:
        suite_rows[s] = load_videos(s, ckpt)

    for tau in tau_grid:
        row = {"ckpt": ckpt, "tau": tau}
        row["dev_real_fpr"] = fpr_at_tau(suite_rows["teams_real_all_dev"], tau)
        row["lb_real_fpr"] = fpr_at_tau(suite_rows["teams_real_all_lockbox"], tau)
        row["dor_real_fpr"] = fpr_at_tau(suite_rows["teams_real_dor_dev"], tau)
        row["poor_quality_real_fpr"] = fpr_at_tau(suite_rows["teams_real_poor_quality_dev"], tau)
        row["lighting_extreme_real_fpr"] = fpr_at_tau(suite_rows["teams_real_lighting_extreme_dev"], tau)
        row["dev_fake_recall_teams"] = recall_at_tau(suite_rows["teams_fake_all_dev"], tau)
        row["dev_fake_recall_viso"] = recall_at_tau(suite_rows["visomaster_enhanced_macro_dev"], tau)
        row["dev_fake_recall_deeplive"] = recall_at_tau(suite_rows["deeplive_enhanced_dev"], tau)
        row["dev_macro_recall"] = (
            (row["dev_fake_recall_teams"] + row["dev_fake_recall_viso"] + row["dev_fake_recall_deeplive"]) / 3.0
        )
        row["lb_fake_recall"] = recall_at_tau(suite_rows["teams_fake_all_lockbox"], tau)
        results.append(row)
    return results


def find_tau_match(results: list[dict], target_lb_fpr: float = 0.025) -> Optional[dict]:
    """Find smallest τ with lb_real_fpr ≤ target."""
    # Sort by τ ascending; first one with lb_real_fpr ≤ target
    sorted_r = sorted(results, key=lambda r: r["tau"])
    for r in sorted_r:
        if r["lb_real_fpr"] <= target_lb_fpr:
            return r
    return None


def find_tau_for_dev_fpr(rows: list[dict], target_fpr: float, prob_key: str = "avg_video_prob") -> float:
    """Calibrate τ to a target real_fpr on dev (matching scorer's midpoint convention)."""
    reals = [r for r in rows if r["label"] == 0]
    if not reals:
        return float("nan")
    probs = sorted([r[prob_key] for r in reals], reverse=True)
    n = len(probs)
    max_n_fp = int(target_fpr * n)
    if max_n_fp >= n:
        return 0.0
    if max_n_fp == 0:
        return max(probs) + 1e-9
    upper = probs[max_n_fp - 1]
    lower = probs[max_n_fp]
    return (upper + lower) / 2.0


# -----------------------------------------------------------------------------
# Job 3: per-identity lockbox FPR breakdown
# -----------------------------------------------------------------------------

def job3_per_identity_lockbox() -> dict:
    """For teams_real_all_lockbox, decompose FPR by identity for each ckpt at
    its dev-cal τ (frame-level basis).
    """
    out = {}
    chronic = {"dor_shkedi", "Roy_D", "PC_Generator", "bla_bla_chow", "xiang", "dor"}

    for ckpt in CKPTS:
        rows = load_frames("teams_real_all_lockbox", ckpt)
        # We also want video-level at calibrated τ
        v_rows = load_videos("teams_real_all_lockbox", ckpt)
        tau = TAU_CAL[ckpt]
        per_id_video = defaultdict(lambda: {"n_videos": 0, "n_fp_videos": 0,
                                            "n_frames": 0, "n_fp_frames": 0})
        # Video-level decomposition (canonical for scorecard FPR)
        for r in v_rows:
            if r["label"] != 0:
                continue
            ident = parse_identity(r["video_id"])
            per_id_video[ident]["n_videos"] += 1
            if r["avg_video_prob"] >= tau:
                per_id_video[ident]["n_fp_videos"] += 1
        # Frame-level decomposition
        for r in rows:
            if r["label"] != 0:
                continue
            ident = parse_identity(r["video_id"])
            per_id_video[ident]["n_frames"] += 1
            if r["frame_prob"] >= tau:
                per_id_video[ident]["n_fp_frames"] += 1

        id_records = []
        for ident, d in per_id_video.items():
            fpr_v = d["n_fp_videos"] / d["n_videos"] if d["n_videos"] > 0 else None
            fpr_f = d["n_fp_frames"] / d["n_frames"] if d["n_frames"] > 0 else None
            is_chronic = ident in chronic
            id_records.append({
                "identity": ident,
                "is_chronic": is_chronic,
                "n_videos": d["n_videos"],
                "n_fp_videos": d["n_fp_videos"],
                "fpr_video": fpr_v,
                "n_frames": d["n_frames"],
                "n_fp_frames": d["n_fp_frames"],
                "fpr_frame": fpr_f,
            })

        out[ckpt] = id_records
    return out


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> None:
    shift = job1_shift_profile()

    # Job 1 CSV
    with open(HERE / f"_shift_profile_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suite", "ckpt", "n_total", "n_real", "n_fake",
                    "mean_all", "mean_real", "mean_fake"])
        for suite in CONTRACT_SUITES:
            for ckpt in CKPTS:
                d = shift[suite][ckpt]
                w.writerow([suite, ckpt, d.get("n", 0),
                            d.get("n_real", 0), d.get("n_fake", 0),
                            f"{d['mean_all']:.4f}" if d["mean_all"] is not None else "",
                            f"{d['mean_real']:.4f}" if d.get("mean_real") is not None else "",
                            f"{d['mean_fake']:.4f}" if d.get("mean_fake") is not None else ""])

    # Job 2 CSVs (all Slot 1 ckpts so we can pick strongest)
    all_sweeps = {}
    for slot1_ckpt in ["slot1_lora_p8a_periodic_step1500",
                       "slot1_lora_p8a_periodic_step3500",
                       "slot1_lora_p8a_top_n_step2000"]:
        all_sweeps[slot1_ckpt] = job2_tau_sweep(slot1_ckpt)

    with open(HERE / f"_tau_recal_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ckpt", "tau", "dev_real_fpr", "lb_real_fpr", "dor_real_fpr",
                    "poor_quality_real_fpr", "lighting_extreme_real_fpr",
                    "dev_fake_recall_teams", "dev_fake_recall_viso",
                    "dev_fake_recall_deeplive", "dev_macro_recall",
                    "lb_fake_recall"])
        for ckpt, sweep in all_sweeps.items():
            for r in sweep:
                w.writerow([
                    r["ckpt"], f"{r['tau']:.4f}",
                    f"{r['dev_real_fpr']:.4f}", f"{r['lb_real_fpr']:.4f}",
                    f"{r['dor_real_fpr']:.4f}",
                    f"{r['poor_quality_real_fpr']:.4f}",
                    f"{r['lighting_extreme_real_fpr']:.4f}",
                    f"{r['dev_fake_recall_teams']:.4f}",
                    f"{r['dev_fake_recall_viso']:.4f}",
                    f"{r['dev_fake_recall_deeplive']:.4f}",
                    f"{r['dev_macro_recall']:.4f}",
                    f"{r['lb_fake_recall']:.4f}",
                ])

    # Also: find τ_match for each Slot 1 ckpt (lb_fpr ≤ 0.025 = E2B-comparable)
    matches = {}
    for ckpt, sweep in all_sweeps.items():
        matches[ckpt] = find_tau_match(sweep, target_lb_fpr=0.025)

    # Cross-check: compute precise τ_match from video-level data (rather than grid).
    # For each ckpt, recalibrate τ on teams_real_all_lockbox to FPR ≤ 0.025 (= E2B-comparable).
    precise_matches = {}
    for ckpt in ["slot1_lora_p8a_periodic_step1500",
                 "slot1_lora_p8a_periodic_step3500",
                 "slot1_lora_p8a_top_n_step2000"]:
        v_lb = load_videos("teams_real_all_lockbox", ckpt)
        # find smallest τ such that lb FPR ≤ 0.025
        # Sort lb-real probs descending, take floor(0.025 * n) as max_n_fp.
        reals = [r for r in v_lb if r["label"] == 0]
        probs = sorted([r["avg_video_prob"] for r in reals], reverse=True)
        n = len(probs)
        max_n_fp = int(0.025 * n)
        if max_n_fp >= n:
            tau_match = 0.0
        elif max_n_fp == 0:
            tau_match = max(probs) + 1e-9
        else:
            upper = probs[max_n_fp - 1]
            lower = probs[max_n_fp]
            tau_match = (upper + lower) / 2.0

        # Evaluate all metrics at this τ
        suite_rows = {}
        for s in ["teams_real_all_dev", "teams_real_all_lockbox",
                  "teams_fake_all_dev", "teams_fake_all_lockbox",
                  "visomaster_enhanced_macro_dev", "deeplive_enhanced_dev",
                  "teams_real_dor_dev", "teams_real_poor_quality_dev",
                  "teams_real_lighting_extreme_dev"]:
            suite_rows[s] = load_videos(s, ckpt)
        row = {"ckpt": ckpt, "tau_match": tau_match}
        row["dev_real_fpr"] = fpr_at_tau(suite_rows["teams_real_all_dev"], tau_match)
        row["lb_real_fpr"] = fpr_at_tau(suite_rows["teams_real_all_lockbox"], tau_match)
        row["dor_real_fpr"] = fpr_at_tau(suite_rows["teams_real_dor_dev"], tau_match)
        row["poor_quality_real_fpr"] = fpr_at_tau(suite_rows["teams_real_poor_quality_dev"], tau_match)
        row["lighting_extreme_real_fpr"] = fpr_at_tau(suite_rows["teams_real_lighting_extreme_dev"], tau_match)
        row["dev_fake_recall_teams"] = recall_at_tau(suite_rows["teams_fake_all_dev"], tau_match)
        row["dev_fake_recall_viso"] = recall_at_tau(suite_rows["visomaster_enhanced_macro_dev"], tau_match)
        row["dev_fake_recall_deeplive"] = recall_at_tau(suite_rows["deeplive_enhanced_dev"], tau_match)
        row["dev_macro_recall"] = (
            (row["dev_fake_recall_teams"] + row["dev_fake_recall_viso"] + row["dev_fake_recall_deeplive"]) / 3.0
        )
        row["lb_fake_recall"] = recall_at_tau(suite_rows["teams_fake_all_lockbox"], tau_match)
        precise_matches[ckpt] = row

    with open(HERE / f"_tau_match_e2b_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ckpt", "tau_match", "dev_real_fpr", "lb_real_fpr",
                    "dor_real_fpr", "poor_quality_real_fpr",
                    "lighting_extreme_real_fpr", "dev_fake_recall_teams",
                    "dev_fake_recall_viso", "dev_fake_recall_deeplive",
                    "dev_macro_recall", "lb_fake_recall"])
        for ckpt, r in precise_matches.items():
            w.writerow([
                r["ckpt"], f"{r['tau_match']:.4f}",
                f"{r['dev_real_fpr']:.4f}", f"{r['lb_real_fpr']:.4f}",
                f"{r['dor_real_fpr']:.4f}",
                f"{r['poor_quality_real_fpr']:.4f}",
                f"{r['lighting_extreme_real_fpr']:.4f}",
                f"{r['dev_fake_recall_teams']:.4f}",
                f"{r['dev_fake_recall_viso']:.4f}",
                f"{r['dev_fake_recall_deeplive']:.4f}",
                f"{r['dev_macro_recall']:.4f}",
                f"{r['lb_fake_recall']:.4f}",
            ])

    # Job 3
    per_id = job3_per_identity_lockbox()
    with open(HERE / f"_per_identity_lockbox_{DATE}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ckpt", "identity", "is_chronic_6", "n_videos", "n_fp_videos",
                    "fpr_video", "n_frames", "n_fp_frames", "fpr_frame"])
        for ckpt in CKPTS:
            for rec in per_id[ckpt]:
                w.writerow([
                    ckpt, rec["identity"],
                    "Y" if rec["is_chronic"] else "",
                    rec["n_videos"], rec["n_fp_videos"],
                    f"{rec['fpr_video']:.4f}" if rec["fpr_video"] is not None else "",
                    rec["n_frames"], rec["n_fp_frames"],
                    f"{rec['fpr_frame']:.4f}" if rec["fpr_frame"] is not None else "",
                ])

    # JSON dump for downstream
    out_json = {
        "date": DATE,
        "shift_profile": shift,
        "tau_match_e2b_comparable": precise_matches,
        "matches_from_grid": {k: v for k, v in matches.items() if v is not None},
    }
    (HERE / f"_results_{DATE}.json").write_text(json.dumps(out_json, indent=2, default=str))

    # Print top-line summary
    print("=" * 90)
    print(f"SLOT 1 SHIFT ANALYSIS — {DATE}")
    print("=" * 90)
    print("\n--- Job 1: Per-suite mean prob_fake (P8A vs Slot 1) ---")
    print(f"{'suite':40s} {'P8A':>8s} {'S1_1500':>8s} {'S1_3500':>8s} {'S1_2000':>8s} {'Δ(2000-P8A)':>12s}")
    for suite in CONTRACT_SUITES:
        d_p8a = shift[suite]["p8a_reference_step5000"]
        d_2000 = shift[suite]["slot1_lora_p8a_top_n_step2000"]
        d_1500 = shift[suite]["slot1_lora_p8a_periodic_step1500"]
        d_3500 = shift[suite]["slot1_lora_p8a_periodic_step3500"]
        delta = (d_2000["mean_all"] - d_p8a["mean_all"]) if (d_2000["mean_all"] is not None and d_p8a["mean_all"] is not None) else None
        print(
            f"{suite:40s} {d_p8a['mean_all']:>8.4f} {d_1500['mean_all']:>8.4f} "
            f"{d_3500['mean_all']:>8.4f} {d_2000['mean_all']:>8.4f} "
            f"{delta:>+12.4f}"
        )

    print("\n--- Job 2: τ_match (lb_fpr ≤ 0.025) for each Slot 1 ckpt ---")
    print(f"{'ckpt':45s} {'τ_match':>8s} {'lb_fpr':>7s} {'lb_rec':>7s} {'dev_mac':>8s} {'dor_fpr':>8s}")
    for ckpt, r in precise_matches.items():
        print(
            f"{ckpt:45s} {r['tau_match']:>8.4f} "
            f"{r['lb_real_fpr']:>7.4f} {r['lb_fake_recall']:>7.4f} "
            f"{r['dev_macro_recall']:>8.4f} {r['dor_real_fpr']:>8.4f}"
        )

    print("\n--- Job 3: chronic-6 share of lockbox FPR (top_n_step2000) ---")
    top_recs = per_id["slot1_lora_p8a_top_n_step2000"]
    # Total fp_videos overall
    total_fp_v = sum(r["n_fp_videos"] for r in top_recs)
    chronic_fp_v = sum(r["n_fp_videos"] for r in top_recs if r["is_chronic"])
    total_v = sum(r["n_videos"] for r in top_recs)
    print(f"Total n_videos (real): {total_v}; total fp_videos: {total_fp_v}; "
          f"chronic-6 fp_videos: {chronic_fp_v} ({chronic_fp_v / total_fp_v * 100:.1f}% of fp)")
    print()
    print("Top 15 identities by fp_videos:")
    top_recs_sorted = sorted(top_recs, key=lambda r: -r["n_fp_videos"])[:15]
    for rec in top_recs_sorted:
        flag = " CHRONIC" if rec["is_chronic"] else ""
        print(
            f"  {rec['identity']:25s} n_videos={rec['n_videos']:4d} "
            f"n_fp={rec['n_fp_videos']:4d} fpr={rec['fpr_video']:.4f}{flag}"
        )

    print("\nWritten outputs:")
    print(f"  {HERE / f'_shift_profile_{DATE}.csv'}")
    print(f"  {HERE / f'_tau_recal_{DATE}.csv'}")
    print(f"  {HERE / f'_tau_match_e2b_{DATE}.csv'}")
    print(f"  {HERE / f'_per_identity_lockbox_{DATE}.csv'}")
    print(f"  {HERE / f'_results_{DATE}.json'}")


if __name__ == "__main__":
    main()
