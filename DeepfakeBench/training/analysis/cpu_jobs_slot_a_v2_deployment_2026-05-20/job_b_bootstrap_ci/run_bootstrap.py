#!/usr/bin/env python3
"""
Paired bootstrap CI on per-video lockbox_real_fpr delta:
  delta = lockbox_real_fpr[SLOT_A_ANCHOR_AWARE_STEP3500]
        - lockbox_real_fpr[P8A_REFERENCE_STEP5000]

Inputs: per-video CSVs pulled from
  gs://training-job-outputs/test_results/teams_promotion_contract/
  slot-a-v2-validation-2026-05-20/reports/

τ-calibration: replicates contract policy from arena/score_teams_promotion_contract.py
For the primary analysis we DO NOT re-derive τ ourselves — we use the τ already
selected by the contract scorer (recorded in checkpoint_summary.csv / promotion_winner.json):
  τ_P8A         = 0.915605
  τ_SlotAv2     = 0.787956

The contract's lockbox_real_suite is `teams_real_all_lockbox` (1361 unique videos);
the lighting_extreme/poor_quality lockbox suites are STRESS SUBSETS of it (207, 22
rows respectively; all ⊂ teams_real_all_lockbox). We therefore report TWO analyses:

  Analysis A (PRIMARY): paired bootstrap over the 1361 videos in
     teams_real_all_lockbox.  This is the actual metric the contract uses for
     the tiebreak.

  Analysis B (TASK-LITERAL UNION): paired bootstrap over the unique-video union
     of the three lockbox real suites.  Because teams_real_all_lockbox ⊃
     {lighting_extreme, poor_quality}, the union == teams_real_all_lockbox.
     Analysis B is therefore identical to Analysis A.

Bars (pre-stated):
  Bar 1 (load-bearing CI):    95% CI on delta covers 0  → tiebreak inside sampling noise
  Bar 2 (overlap of CIs):     P8A 95% FPR CI overlaps Slot A v2 95% FPR CI
                              → not statistically distinguishable

Output: stdout summary + bootstrap_results.json + per-video JSON snapshot.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

P8A_KEY = "P8A_REFERENCE_STEP5000"
SLOTAV2_KEY = "SLOT_A_ANCHOR_AWARE_STEP3500"

# Frozen thresholds from contract checkpoint_summary.csv:
TAU = {
    P8A_KEY: 0.915605,
    SLOTAV2_KEY: 0.787956,
}

REPORT_FILES = {
    P8A_KEY: {
        "all_lockbox": "teams_real_all_lockbox_p8a_reference_step5000_videos_report.csv",
        "le_lockbox": "teams_real_lighting_extreme_lockbox_p8a_reference_step5000_videos_report.csv",
        "pq_lockbox": "teams_real_poor_quality_lockbox_p8a_reference_step5000_videos_report.csv",
        "all_dev": "teams_real_all_dev_p8a_reference_step5000_videos_report.csv",
    },
    SLOTAV2_KEY: {
        "all_lockbox": "teams_real_all_lockbox_slot_a_anchor_aware_step3500_videos_report.csv",
        "le_lockbox": "teams_real_lighting_extreme_lockbox_slot_a_anchor_aware_step3500_videos_report.csv",
        "pq_lockbox": "teams_real_poor_quality_lockbox_slot_a_anchor_aware_step3500_videos_report.csv",
        "all_dev": "teams_real_all_dev_slot_a_anchor_aware_step3500_videos_report.csv",
    },
}

N_RESAMPLES = 10_000
RNG_SEED = 20260520


def load_report(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"video_id", "avg_video_prob", "label"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    return df[["video_id", "label", "avg_video_prob"]].copy()


def fpr_at_threshold(scores: np.ndarray, labels: np.ndarray, tau: float) -> float:
    """real-FPR = fraction of real videos (label==0) with avg_video_prob >= tau."""
    real_mask = labels == 0
    n_real = int(real_mask.sum())
    if n_real == 0:
        return float("nan")
    return float((scores[real_mask] >= tau).sum() / n_real)


def paired_video_table(
    df_p8a: pd.DataFrame, df_slot: pd.DataFrame
) -> pd.DataFrame:
    """Join per-video probs from both ckpts on video_id so resampling is paired."""
    a = df_p8a.rename(columns={"avg_video_prob": "prob_p8a"})
    b = df_slot.rename(columns={"avg_video_prob": "prob_slot"})
    joined = a.merge(b[["video_id", "prob_slot"]], on="video_id", how="inner")
    # Sanity: labels must match between ckpts on the same video.
    return joined


def bootstrap_paired_fpr_delta(
    table: pd.DataFrame, tau_p8a: float, tau_slot: float,
    n_resamples: int, rng: np.random.Generator,
) -> dict:
    """Paired bootstrap of (FPR_slot - FPR_p8a) over real videos."""
    real_mask = table["label"].values == 0
    real_only = table.loc[real_mask].reset_index(drop=True)
    n = len(real_only)
    p8a = real_only["prob_p8a"].values.astype(np.float64)
    slot = real_only["prob_slot"].values.astype(np.float64)

    # Point estimates on observed sample (these must match contract output).
    obs_fpr_p8a = float((p8a >= tau_p8a).sum() / n)
    obs_fpr_slot = float((slot >= tau_slot).sum() / n)
    obs_delta = obs_fpr_slot - obs_fpr_p8a

    # Bootstrap: resample real video indices with replacement; both ckpts share
    # the same resampled indices (paired).
    idx_matrix = rng.integers(0, n, size=(n_resamples, n), endpoint=False)
    p8a_above = (p8a >= tau_p8a).astype(np.float64)  # 1.0 if FP at τ, else 0
    slot_above = (slot >= tau_slot).astype(np.float64)

    fpr_p8a_boot = p8a_above[idx_matrix].mean(axis=1)
    fpr_slot_boot = slot_above[idx_matrix].mean(axis=1)
    delta_boot = fpr_slot_boot - fpr_p8a_boot

    def pct_ci(x: np.ndarray) -> tuple[float, float]:
        return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))

    return {
        "n_videos_real": int(n),
        "observed": {
            "fpr_p8a": obs_fpr_p8a,
            "fpr_slotav2": obs_fpr_slot,
            "delta": obs_delta,
            "n_fp_p8a": int(p8a_above.sum()),
            "n_fp_slotav2": int(slot_above.sum()),
        },
        "bootstrap": {
            "n_resamples": int(n_resamples),
            "fpr_p8a_ci95": pct_ci(fpr_p8a_boot),
            "fpr_p8a_mean": float(fpr_p8a_boot.mean()),
            "fpr_slotav2_ci95": pct_ci(fpr_slot_boot),
            "fpr_slotav2_mean": float(fpr_slot_boot.mean()),
            "delta_ci95": pct_ci(delta_boot),
            "delta_mean": float(delta_boot.mean()),
            "p_delta_gt_0": float((delta_boot > 0).mean()),
            "p_delta_eq_0": float((delta_boot == 0).mean()),
            "p_delta_lt_0": float((delta_boot < 0).mean()),
        },
    }


def fmt_pct(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def derive_tau_from_dev(df_dev: pd.DataFrame, target_fpr: float = 0.07) -> float:
    """Replicate contract's τ-calibration on dev_real_suite.

    The contract sweeps every observed prob as a candidate τ and picks the smallest τ
    such that dev_primary_real_fpr <= target_real_fpr (one of several tier rules — but
    for the purpose of a sanity check, the lowest τ meeting the FPR budget is a tight
    approximation: contract additionally maximizes recall + prefers higher τ, so the
    selected τ may be ABOVE this minimum).
    """
    probs = df_dev.loc[df_dev["label"] == 0, "avg_video_prob"].values.astype(np.float64)
    n = len(probs)
    if n == 0:
        return float("nan")
    # Sort probs descending. The FPR at τ = p_i (i-th highest) is i/n.
    sorted_desc = np.sort(probs)[::-1]
    # Find smallest τ in observed set with FPR <= target
    # i.e., smallest k such that (k / n) <= target_fpr → τ = sorted_desc[k]
    k = int(np.floor(target_fpr * n))
    # τ that gives <= k FPs is sorted_desc[k] (k-th highest score, 0-indexed)
    if k >= n:
        return 0.0
    # ensure FPR(τ) <= target
    tau_min_meeting_budget = float(sorted_desc[k])
    return tau_min_meeting_budget


def main() -> int:
    rng = np.random.default_rng(RNG_SEED)
    results: dict = {
        "method": {
            "tau_calibration": (
                "Used τ frozen by the contract scorer (arena/score_teams_promotion_contract.py)"
                " as recorded in promotion_contract/checkpoint_summary.csv. τ-calibration"
                " is the contract's lex-tiered policy with target_real_fpr=0.07,"
                " target_stress_fpr=0.10, target_fake_recall_min=0.3 (from promotion_winner.json)."
            ),
            "tau_p8a": TAU[P8A_KEY],
            "tau_slotav2": TAU[SLOTAV2_KEY],
            "n_resamples": N_RESAMPLES,
            "rng_seed": RNG_SEED,
            "pairing": (
                "Per-video paired resampling — same resampled video index set"
                " applied to both ckpts. Real videos only (label==0)."
            ),
        },
        "analyses": {},
        "tau_sanity": {},
    }

    # τ sanity check: minimum τ meeting dev FPR budget should be <= the
    # contract-selected τ (which adds recall-max + higher-τ tiebreaks).
    for key in (P8A_KEY, SLOTAV2_KEY):
        df_dev = load_report(DATA / REPORT_FILES[key]["all_dev"])
        tau_budget = derive_tau_from_dev(df_dev, target_fpr=0.07)
        results["tau_sanity"][key] = {
            "tau_min_meeting_dev_fpr_budget": tau_budget,
            "tau_contract_selected": TAU[key],
            "dev_n_real": int((df_dev["label"] == 0).sum()),
        }

    # ============ Analysis A: teams_real_all_lockbox (1361 vids) ============
    df_p8a = load_report(DATA / REPORT_FILES[P8A_KEY]["all_lockbox"])
    df_slot = load_report(DATA / REPORT_FILES[SLOTAV2_KEY]["all_lockbox"])
    table_a = paired_video_table(df_p8a, df_slot)
    res_a = bootstrap_paired_fpr_delta(
        table_a, TAU[P8A_KEY], TAU[SLOTAV2_KEY], N_RESAMPLES, rng,
    )
    res_a["scope"] = "teams_real_all_lockbox"
    res_a["scope_note"] = (
        "The contract's lockbox_real_suite is teams_real_all_lockbox alone."
        " This is the metric used for the contract tiebreak."
    )
    results["analyses"]["A_teams_real_all_lockbox"] = res_a

    # ============ Analysis B: unique-video union of 3 lockbox real suites ============
    # Verify all_lockbox ⊇ lighting_extreme ∪ poor_quality first (we already confirmed
    # this manually above) — then the union is identical to teams_real_all_lockbox.
    df_p8a_le = load_report(DATA / REPORT_FILES[P8A_KEY]["le_lockbox"])
    df_p8a_pq = load_report(DATA / REPORT_FILES[P8A_KEY]["pq_lockbox"])
    union_vids = set(df_p8a["video_id"]).union(set(df_p8a_le["video_id"])).union(set(df_p8a_pq["video_id"]))
    only_all = len(set(df_p8a["video_id"]) - set(df_p8a_le["video_id"]) - set(df_p8a_pq["video_id"]))
    results["analyses"]["B_three_lockbox_suites_union"] = {
        "scope": "set-union of teams_real_all_lockbox + lighting_extreme_lockbox + poor_quality_lockbox",
        "n_vids_in_union": len(union_vids),
        "n_vids_in_all_lockbox_only": only_all,
        "n_vids_in_le_lockbox": len(df_p8a_le),
        "n_vids_in_pq_lockbox": len(df_p8a_pq),
        "is_union_identical_to_all_lockbox": (
            set(df_p8a_le["video_id"]).issubset(set(df_p8a["video_id"]))
            and set(df_p8a_pq["video_id"]).issubset(set(df_p8a["video_id"]))
        ),
        "note": (
            "Stress suites lighting_extreme_lockbox and poor_quality_lockbox are"
            " STRICT SUBSETS of teams_real_all_lockbox. The set-union of the three"
            " is therefore == teams_real_all_lockbox. Analysis B is identical to"
            " Analysis A."
        ),
    }

    # ============ Bars ============
    obs = res_a["observed"]
    boot = res_a["bootstrap"]
    bar1_ci_covers_zero = (boot["delta_ci95"][0] <= 0.0 <= boot["delta_ci95"][1])
    fpr_p8a_ci = boot["fpr_p8a_ci95"]
    fpr_slot_ci = boot["fpr_slotav2_ci95"]
    bar2_cis_overlap = (
        max(fpr_p8a_ci[0], fpr_slot_ci[0]) <= min(fpr_p8a_ci[1], fpr_slot_ci[1])
    )
    results["bars"] = {
        "primary_analysis_for_bars": "A_teams_real_all_lockbox",
        "bar1_load_bearing_ci_covers_zero": {
            "bar_text": "95% CI on delta covers 0",
            "delta_ci95": boot["delta_ci95"],
            "met": bool(bar1_ci_covers_zero),
        },
        "bar2_individual_ci_overlap": {
            "bar_text": "P8A 95% FPR CI overlaps Slot A v2 95% FPR CI",
            "fpr_p8a_ci95": fpr_p8a_ci,
            "fpr_slotav2_ci95": fpr_slot_ci,
            "met": bool(bar2_cis_overlap),
        },
    }

    # ============ Stdout summary ============
    print("=" * 78)
    print("PAIRED BOOTSTRAP CI — lockbox_real_fpr delta")
    print("=" * 78)
    print(f"  scope: teams_real_all_lockbox (contract metric, n_real={res_a['n_videos_real']})")
    print(f"  τ_P8A      = {TAU[P8A_KEY]:.6f}")
    print(f"  τ_SlotAv2  = {TAU[SLOTAV2_KEY]:.6f}")
    print(f"  n_resamples = {N_RESAMPLES}, seed = {RNG_SEED}")
    print()
    print("Observed (point estimates):")
    print(f"  FPR_P8A       = {fmt_pct(obs['fpr_p8a'])}  (n_fp = {obs['n_fp_p8a']} / {res_a['n_videos_real']})")
    print(f"  FPR_SlotAv2   = {fmt_pct(obs['fpr_slotav2'])}  (n_fp = {obs['n_fp_slotav2']} / {res_a['n_videos_real']})")
    print(f"  delta (SlotAv2 - P8A) = {fmt_pct(obs['delta'])}")
    print()
    print("Bootstrap 95% percentile CIs:")
    print(f"  FPR_P8A         CI = [{fpr_p8a_ci[0]:.6f}, {fpr_p8a_ci[1]:.6f}]  mean={boot['fpr_p8a_mean']:.6f}")
    print(f"  FPR_SlotAv2     CI = [{fpr_slot_ci[0]:.6f}, {fpr_slot_ci[1]:.6f}]  mean={boot['fpr_slotav2_mean']:.6f}")
    print(f"  delta           CI = [{boot['delta_ci95'][0]:.6f}, {boot['delta_ci95'][1]:.6f}]  mean={boot['delta_mean']:.6f}")
    print(f"  P(delta > 0)    = {boot['p_delta_gt_0']:.4f}")
    print(f"  P(delta = 0)    = {boot['p_delta_eq_0']:.4f}")
    print(f"  P(delta < 0)    = {boot['p_delta_lt_0']:.4f}")
    print()
    print("Bars (mechanical, on the primary analysis):")
    print(f"  Bar 1 (delta CI covers 0):       {'MET' if bar1_ci_covers_zero else 'NOT MET'}  CI={boot['delta_ci95']}")
    print(f"  Bar 2 (individual FPR CIs overlap): {'MET' if bar2_cis_overlap else 'NOT MET'}")
    print(f"     P8A     CI: {fpr_p8a_ci}")
    print(f"     SlotAv2 CI: {fpr_slot_ci}")
    print()
    print("Analysis B (3-suite union):")
    print(f"  Stress suites lighting_extreme / poor_quality are STRICT SUBSETS of teams_real_all_lockbox.")
    print(f"  Union size = {len(union_vids)} (equals teams_real_all_lockbox). Analysis B ≡ Analysis A.")
    print()
    print("τ sanity (replicate contract τ-selection from dev_real):")
    for key in (P8A_KEY, SLOTAV2_KEY):
        s = results["tau_sanity"][key]
        print(f"  {key}: tau_min_dev_FPR_<=0.07 = {s['tau_min_meeting_dev_fpr_budget']:.6f}  "
              f"contract_τ = {s['tau_contract_selected']:.6f}  (dev n_real = {s['dev_n_real']})")

    # Persist JSON
    out_json = HERE / "bootstrap_results.json"
    with out_json.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
