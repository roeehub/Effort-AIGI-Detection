"""P8A v3-substrate frame-level scorecard analog (2026-05-01).

Apply the canonical promotion-contract threshold-grid + recall-floor selection
policy (arena/score_teams_promotion_contract.py) to per-frame asis vs prod
scores from analysis/eval_substrate_v3_retag_2026-05-01/. Goal: an honest
production-FPR baseline for P18 to beat.

Caveat: the canonical contract uses VIDEO-level avg_video_prob and selects τ on
DEV, then reads out lockbox. Here we have only per-frame scores on the lockbox
sample (no separate dev pool), so this is a frame-level lockbox-internal
Pareto. We label outputs as such. The asis-vs-prod *delta* is the load-bearing
signal — it isolates "production-tight crop" effect from "selection sample".

CPU only. n_jobs=1. No GCS, no Vertex, no commits.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
V3_PARQUET = REPO / "analysis/eval_substrate_v3_retag_2026-05-01/outputs/eval_substrate_v3_retag.parquet"
TAGS_PARQUET = REPO / "analysis/lockbox_tagging/full_tags_2026-04-27.parquet"
OUT = REPO / "analysis/p8a_v3_substrate_scorecard_2026-05-01/outputs"

# Canonical contract defaults (score_teams_promotion_contract.py:281-291).
CONTRACT_TARGET_REAL_FPR = 0.07
CONTRACT_TARGET_RECALL_FLOOR_DEFAULT = 0.70   # default 2026-04-29
CONTRACT_TARGET_RECALL_FLOOR_RELAXED = 0.30   # relaxed sweep policy

# Canonical scorer constructs grid from set of unique probs in dev suites; we
# cap at ~5000 quantile-spaced points to match the canonical scale.
GRID_MAX_POINTS = 5000

# Recall reference points for Pareto table.
PARETO_RECALL_TARGETS = [0.20, 0.30, 0.50, 0.70]

# Capture modes to break out individually.
CAPTURE_MODES_OF_INTEREST = ("normal_photo", "phone_screen", "webcam")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("p8a_v3_substrate_scorecard")


# ---------------------------------------------------------------------------
# Load and merge
# ---------------------------------------------------------------------------
def load_v3_with_tags() -> pd.DataFrame:
    v3 = pd.read_parquet(V3_PARQUET)
    logger.info("Loaded v3 substrate: %d rows", len(v3))

    tags = pd.read_parquet(TAGS_PARQUET)
    tags_lock = tags[tags["split"] == "lockbox"][
        ["local_path", "is_pose_extreme", "is_no_face", "face_area_ratio"]
    ].copy()
    logger.info("Loaded tags lockbox subset: %d rows", len(tags_lock))

    merged = v3.merge(
        tags_lock,
        left_on="original_local_path",
        right_on="local_path",
        how="left",
    )
    n_merged_with_tags = merged["face_area_ratio"].notna().sum()
    logger.info(
        "Merged: %d rows; %d with tags (modern_v2 columns)",
        len(merged), n_merged_with_tags,
    )
    return merged


def apply_modern_v2_filter(df: pd.DataFrame) -> pd.DataFrame:
    """modern_v2: mode not in {webcam, screen} AND face_area_ratio >= 0.10
    AND not is_pose_extreme AND not is_no_face.
    """
    keep = (
        (~df["clip_capture_mode"].isin({"webcam", "screen"}))
        & (df["face_area_ratio"].fillna(-1.0) >= 0.10)
        & (~df["is_pose_extreme"].fillna(False))
        & (~df["is_no_face"].fillna(False))
    )
    return df[keep].copy()


# ---------------------------------------------------------------------------
# Threshold grid
# ---------------------------------------------------------------------------
def build_threshold_grid(probs: np.ndarray, max_points: int = GRID_MAX_POINTS) -> np.ndarray:
    """Quantile-spaced grid bounded by 0 and 1.

    Mirrors the canonical scorer: include 0 and 1 plus the unique prob values.
    Cap at max_points by quantile-spacing if needed.
    """
    uniq = np.unique(probs)
    if len(uniq) <= max_points:
        grid = np.concatenate([[0.0], uniq, [1.0]])
    else:
        qs = np.linspace(0.0, 1.0, max_points)
        quantile_pts = np.quantile(probs, qs)
        grid = np.concatenate([[0.0], quantile_pts, [1.0]])
    return np.unique(np.clip(grid, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Scorecard core
# ---------------------------------------------------------------------------
def per_threshold_metrics(
    real_probs: np.ndarray,
    fake_probs: np.ndarray,
    grid: np.ndarray,
) -> pd.DataFrame:
    """For each tau in grid, compute fpr (on reals) and recall (on fakes)."""
    rows = []
    for tau in grid:
        fpr = float((real_probs >= tau).mean()) if len(real_probs) else float("nan")
        recall = float((fake_probs >= tau).mean()) if len(fake_probs) else float("nan")
        rows.append({"threshold": float(tau), "real_fpr": fpr, "fake_recall": recall})
    return pd.DataFrame(rows)


def select_threshold_with_floor(
    grid_df: pd.DataFrame,
    target_fpr: float,
    target_recall_floor: float,
) -> Dict:
    """Apply canonical contract sort policy (frame-level analog).

    Tier 0: real_fpr <= target_fpr AND fake_recall >= floor.
    Tier 1: real_fpr <= target_fpr but recall < floor.
    Tier 2: real_fpr > target_fpr.

    Within tier: maximize fake_recall, then prefer higher tau (more
    conservative), then lower fpr.
    """
    if grid_df.empty:
        return {"selected_threshold": None, "reason": "empty grid"}

    rows = grid_df.copy()
    rows["fpr_ok"] = rows["real_fpr"] <= target_fpr + 1e-9
    rows["recall_ok"] = rows["fake_recall"] >= target_recall_floor - 1e-9
    # tier 0: both, tier 1: fpr ok recall fail, tier 2: fpr violates
    def _tier(r):
        if r["fpr_ok"] and r["recall_ok"]:
            return 0
        if r["fpr_ok"]:
            return 1
        return 2
    rows["tier"] = rows.apply(_tier, axis=1)

    # Sort: tier asc, fake_recall desc, threshold desc, real_fpr asc
    rows = rows.sort_values(
        by=["tier", "fake_recall", "threshold", "real_fpr"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    best = rows.iloc[0]
    return {
        "selected_threshold": float(best["threshold"]),
        "selected_real_fpr": float(best["real_fpr"]),
        "selected_fake_recall": float(best["fake_recall"]),
        "selected_tier": int(best["tier"]),
        "selected_meets_recall_floor": bool(best["recall_ok"]),
        "selected_meets_fpr_budget": bool(best["fpr_ok"]),
        "n_grid_points": int(len(grid_df)),
        "n_tier0_candidates": int((rows["tier"] == 0).sum()),
        "n_tier1_candidates": int((rows["tier"] == 1).sum()),
    }


def compute_pareto_at_recall_targets(
    grid_df: pd.DataFrame,
    targets: List[float] = PARETO_RECALL_TARGETS,
) -> List[Dict]:
    """For each target recall, find the lowest FPR achievable with recall >= target."""
    out = []
    for tgt in targets:
        eligible = grid_df[grid_df["fake_recall"] >= tgt - 1e-9]
        if eligible.empty:
            out.append({
                "target_recall": tgt,
                "achievable": False,
                "min_fpr": None,
                "threshold_at_min_fpr": None,
                "actual_recall": None,
                "max_recall_at_grid": float(grid_df["fake_recall"].max()) if not grid_df.empty else None,
            })
            continue
        # Minimize fpr, then maximize threshold (conservative tie-break).
        eligible = eligible.sort_values(by=["real_fpr", "threshold"], ascending=[True, False])
        best = eligible.iloc[0]
        out.append({
            "target_recall": tgt,
            "achievable": True,
            "min_fpr": float(best["real_fpr"]),
            "threshold_at_min_fpr": float(best["threshold"]),
            "actual_recall": float(best["fake_recall"]),
        })
    return out


# ---------------------------------------------------------------------------
# Per-mode and per-method
# ---------------------------------------------------------------------------
def per_mode_pareto(
    df: pd.DataFrame,
    score_col: str,
    targets: List[float] = PARETO_RECALL_TARGETS,
) -> pd.DataFrame:
    """Per-capture-mode FPR-vs-recall Pareto.

    Splits reals by mode. For modes with no fakes (most reals), recall comes
    from the global fake set (all fakes — since we want "FPR contributed by
    this capture mode at recall=X across all fakes").
    """
    fake_probs = df[df["label"] == "fake"][score_col].to_numpy()
    rows = []
    for mode in CAPTURE_MODES_OF_INTEREST:
        mode_reals = df[(df["label"] == "real") & (df["clip_capture_mode"] == mode)]
        n_real = len(mode_reals)
        if n_real == 0:
            continue
        real_probs = mode_reals[score_col].to_numpy()

        grid = build_threshold_grid(np.concatenate([real_probs, fake_probs]))
        grid_df = per_threshold_metrics(real_probs, fake_probs, grid)
        pareto = compute_pareto_at_recall_targets(grid_df, targets)
        for p in pareto:
            rows.append({
                "score_arm": score_col,
                "mode": mode,
                "n_real": n_real,
                "n_fake_global": int(len(fake_probs)),
                "target_recall": p["target_recall"],
                "achievable": p["achievable"],
                "min_fpr": p.get("min_fpr"),
                "threshold_at_min_fpr": p.get("threshold_at_min_fpr"),
                "actual_recall": p.get("actual_recall"),
            })
    return pd.DataFrame(rows)


def per_method_recall_at_tau(
    df: pd.DataFrame,
    score_col: str,
    tau: float,
) -> pd.DataFrame:
    fakes = df[df["label"] == "fake"]
    rows = []
    for method, grp in fakes.groupby("method", dropna=False):
        recall = float((grp[score_col] >= tau).mean()) if len(grp) else float("nan")
        rows.append({
            "score_arm": score_col,
            "method": str(method),
            "n_fake": int(len(grp)),
            "tau": tau,
            "recall": recall,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def score_arm(
    df: pd.DataFrame,
    score_col: str,
    arm_name: str,
) -> Dict:
    """Build the full scorecard for one score arm (asis or prod) on a given subset."""
    real_probs = df[df["label"] == "real"][score_col].to_numpy()
    fake_probs = df[df["label"] == "fake"][score_col].to_numpy()

    grid = build_threshold_grid(np.concatenate([real_probs, fake_probs]))
    grid_df = per_threshold_metrics(real_probs, fake_probs, grid)

    # Apply contract-style selection at default 0.70 floor and at 0.30 floor.
    sel_default = select_threshold_with_floor(
        grid_df,
        target_fpr=CONTRACT_TARGET_REAL_FPR,
        target_recall_floor=CONTRACT_TARGET_RECALL_FLOOR_DEFAULT,
    )
    sel_relaxed = select_threshold_with_floor(
        grid_df,
        target_fpr=CONTRACT_TARGET_REAL_FPR,
        target_recall_floor=CONTRACT_TARGET_RECALL_FLOOR_RELAXED,
    )

    pareto = compute_pareto_at_recall_targets(grid_df, PARETO_RECALL_TARGETS)

    return {
        "arm_name": arm_name,
        "score_col": score_col,
        "n_real": int(len(real_probs)),
        "n_fake": int(len(fake_probs)),
        "selection_default_floor_0.70": sel_default,
        "selection_relaxed_floor_0.30": sel_relaxed,
        "pareto_at_recall_targets": pareto,
        "grid_summary": {
            "n_grid_points": int(len(grid_df)),
            "min_fpr": float(grid_df["real_fpr"].min()),
            "max_fpr": float(grid_df["real_fpr"].max()),
            "min_recall": float(grid_df["fake_recall"].min()),
            "max_recall": float(grid_df["fake_recall"].max()),
        },
        "_grid_df": grid_df,  # carry through for csv export
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_v3_with_tags()
    df_modern_v2 = apply_modern_v2_filter(df)
    logger.info(
        "modern_v2 filter: %d -> %d (%d real / %d fake)",
        len(df), len(df_modern_v2),
        int((df_modern_v2["label"]=="real").sum()),
        int((df_modern_v2["label"]=="fake").sum()),
    )

    full_results: Dict = {
        "metadata": {
            "checkpoint": "value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth",
            "checkpoint_alias": "P8A_REFERENCE_STEP5000",
            "substrate": "v3-retag (production-tight crop, RFA=0.85)",
            "substrate_source": "analysis/eval_substrate_v3_retag_2026-05-01/outputs/eval_substrate_v3_retag.parquet",
            "n_lockbox_frames": int(len(df)),
            "n_lockbox_frames_modern_v2": int(len(df_modern_v2)),
            "contract_target_real_fpr": CONTRACT_TARGET_REAL_FPR,
            "contract_recall_floor_default": CONTRACT_TARGET_RECALL_FLOOR_DEFAULT,
            "contract_recall_floor_relaxed": CONTRACT_TARGET_RECALL_FLOOR_RELAXED,
            "pareto_recall_targets": PARETO_RECALL_TARGETS,
            "selection_caveat": (
                "Frame-level analog. Canonical scorer uses VIDEO-level avg_video_prob "
                "and selects tau on DEV pool; this analog uses lockbox frames as both "
                "selection and readout (no held-out dev within v3 substrate). The "
                "asis-vs-prod delta is the load-bearing signal; absolute numbers are "
                "not directly comparable to canonical video-level scorecards."
            ),
        },
        "canonical_p8a_reference_video_level": {
            "source": "analysis/policy_reruns_2026-04-29_floor_0p70/promotion_contract.json",
            "checkpoint_key": "P8A_REFERENCE_STEP5000",
            "selected_threshold": 0.915605,
            "lockbox_real_fpr_video_level": 0.018369,
            "lockbox_fake_recall_video_level": 0.387352,
            "dev_fake_macro_recall": 0.30028,
            "lockbox_real_n_videos": 1361,
            "lockbox_fake_n_videos": 253,
        },
        "scorecards": {},
    }

    # Score four configurations: (asis vs prod) x (full vs modern_v2).
    configs = [
        ("full", df, "prob_fake_asis", "asis_full"),
        ("full", df, "prob_fake_prod", "prod_full"),
        ("modern_v2", df_modern_v2, "prob_fake_asis", "asis_modern_v2"),
        ("modern_v2", df_modern_v2, "prob_fake_prod", "prod_modern_v2"),
    ]

    grid_csv_rows: List[Dict] = []
    for subset_name, sub_df, score_col, arm_name in configs:
        logger.info("Scoring arm: %s on subset=%s (n=%d)", arm_name, subset_name, len(sub_df))
        scorecard = score_arm(sub_df, score_col, arm_name)
        grid_df = scorecard.pop("_grid_df")
        # Append grid rows for csv
        for _, r in grid_df.iterrows():
            grid_csv_rows.append({
                "subset": subset_name,
                "arm": arm_name,
                "score_col": score_col,
                "threshold": float(r["threshold"]),
                "real_fpr": float(r["real_fpr"]),
                "fake_recall": float(r["fake_recall"]),
            })
        scorecard["subset"] = subset_name
        full_results["scorecards"][arm_name] = scorecard

    # Per-mode Pareto (asis vs prod, full only — modern_v2 excludes webcam)
    logger.info("Computing per-mode Pareto curves...")
    per_mode_asis = per_mode_pareto(df, "prob_fake_asis")
    per_mode_asis["subset"] = "full"
    per_mode_prod = per_mode_pareto(df, "prob_fake_prod")
    per_mode_prod["subset"] = "full"
    per_mode_df = pd.concat([per_mode_asis, per_mode_prod], ignore_index=True)
    per_mode_csv_path = OUT / "per_mode_pareto.csv"
    per_mode_df.to_csv(per_mode_csv_path, index=False)
    logger.info("Wrote %s (%d rows)", per_mode_csv_path, len(per_mode_df))

    # Per-method recall at the default-floor selected tau for each arm/subset
    logger.info("Computing per-method recall at selected taus...")
    per_method_rows: List[Dict] = []
    for subset_name, sub_df, score_col, arm_name in configs:
        sc = full_results["scorecards"][arm_name]
        tau_default = sc["selection_default_floor_0.70"].get("selected_threshold")
        tau_relaxed = sc["selection_relaxed_floor_0.30"].get("selected_threshold")
        if tau_default is not None:
            df_pm = per_method_recall_at_tau(sub_df, score_col, tau_default)
            df_pm["subset"] = subset_name
            df_pm["floor_policy"] = "default_0.70"
            per_method_rows.append(df_pm)
        if tau_relaxed is not None:
            df_pm = per_method_recall_at_tau(sub_df, score_col, tau_relaxed)
            df_pm["subset"] = subset_name
            df_pm["floor_policy"] = "relaxed_0.30"
            per_method_rows.append(df_pm)
    per_method_df = pd.concat(per_method_rows, ignore_index=True) if per_method_rows else pd.DataFrame()
    per_method_csv_path = OUT / "per_method_recall.csv"
    per_method_df.to_csv(per_method_csv_path, index=False)
    logger.info("Wrote %s (%d rows)", per_method_csv_path, len(per_method_df))

    # Save grid csv
    grid_csv_df = pd.DataFrame(grid_csv_rows)
    grid_csv_path = OUT / "v3_scorecard.csv"
    grid_csv_df.to_csv(grid_csv_path, index=False)
    logger.info("Wrote %s (%d grid rows)", grid_csv_path, len(grid_csv_df))

    # Save full json summary
    json_path = OUT / "v3_scorecard_summary.json"
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    logger.info("Wrote %s", json_path)

    # Stdout summary
    print()
    print("=" * 78)
    print("P8A v3-substrate FRAME-LEVEL SCORECARD ANALOG")
    print("=" * 78)
    print(f"Lockbox frames (full)      : {len(df)} ({(df['label']=='real').sum()} real / {(df['label']=='fake').sum()} fake)")
    print(f"Lockbox frames (modern_v2) : {len(df_modern_v2)} "
          f"({(df_modern_v2['label']=='real').sum()} real / {(df_modern_v2['label']=='fake').sum()} fake)")
    print()
    print("Canonical asis (video-level, source: 2026-04-29 P8A scorecard):")
    print(f"  tau=0.9156  lockbox_real_fpr=0.0184  lockbox_fake_recall=0.387  dev_fake_macro_recall=0.300")
    print()
    for arm_name, sc in full_results["scorecards"].items():
        print(f"-- ARM: {arm_name} (n_real={sc['n_real']}, n_fake={sc['n_fake']}) --")
        sd = sc["selection_default_floor_0.70"]
        sr = sc["selection_relaxed_floor_0.30"]
        print(f"  Default 0.70 floor selection:")
        print(f"    tau={sd.get('selected_threshold')}  fpr={sd.get('selected_real_fpr')}  "
              f"recall={sd.get('selected_fake_recall')}  tier={sd.get('selected_tier')}  "
              f"meets_floor={sd.get('selected_meets_recall_floor')}  "
              f"meets_budget={sd.get('selected_meets_fpr_budget')}")
        print(f"    n_tier0={sd.get('n_tier0_candidates')}  n_tier1={sd.get('n_tier1_candidates')}")
        print(f"  Relaxed 0.30 floor selection:")
        print(f"    tau={sr.get('selected_threshold')}  fpr={sr.get('selected_real_fpr')}  "
              f"recall={sr.get('selected_fake_recall')}  tier={sr.get('selected_tier')}  "
              f"meets_floor={sr.get('selected_meets_recall_floor')}  "
              f"meets_budget={sr.get('selected_meets_fpr_budget')}")
        print(f"  Pareto (target_recall, achievable_fpr):")
        for p in sc["pareto_at_recall_targets"]:
            ach = "yes" if p["achievable"] else "NO "
            mfp = f"{p['min_fpr']:.4f}" if p['achievable'] else "-----"
            ar = f"{p['actual_recall']:.4f}" if p['achievable'] else "-----"
            print(f"    target={p['target_recall']:.2f}  ach={ach}  min_fpr={mfp}  actual_recall={ar}")
        print()

    # Write final report
    report_md = build_report(full_results, per_mode_df, per_method_df)
    (REPO / "analysis/p8a_v3_substrate_scorecard_2026-05-01/REPORT.md").write_text(report_md)
    logger.info("Wrote REPORT.md")


# ---------------------------------------------------------------------------
# REPORT.md builder
# ---------------------------------------------------------------------------
def _fmt(x, fmt=".4f"):
    if x is None:
        return "n/a"
    try:
        return format(float(x), fmt)
    except Exception:
        return str(x)


def build_report(full: Dict, per_mode_df: pd.DataFrame, per_method_df: pd.DataFrame) -> str:
    md = full["metadata"]
    sc = full["scorecards"]
    canon = full["canonical_p8a_reference_video_level"]

    asis_full = sc["asis_full"]
    prod_full = sc["prod_full"]
    asis_mv2 = sc["asis_modern_v2"]
    prod_mv2 = sc["prod_modern_v2"]

    # Helper to extract pareto rows
    def pareto_lookup(arm_sc, target):
        for p in arm_sc["pareto_at_recall_targets"]:
            if abs(p["target_recall"] - target) < 1e-9:
                return p
        return None

    # Headline numbers
    asis_full_30 = pareto_lookup(asis_full, 0.30)
    prod_full_30 = pareto_lookup(prod_full, 0.30)
    asis_full_70 = pareto_lookup(asis_full, 0.70)
    prod_full_70 = pareto_lookup(prod_full, 0.70)

    sd_prod = prod_full["selection_default_floor_0.70"]
    sr_prod = prod_full["selection_relaxed_floor_0.30"]
    sd_asis = asis_full["selection_default_floor_0.70"]
    sr_asis = asis_full["selection_relaxed_floor_0.30"]

    # Compute "headline gap" — production overestimate vs asis at recall=0.30
    fpr_gap_30 = None
    if asis_full_30 and prod_full_30 and asis_full_30["achievable"] and prod_full_30["achievable"]:
        fpr_gap_30 = prod_full_30["min_fpr"] - asis_full_30["min_fpr"]

    # Pivot per-mode for a compact table
    pm_table_lines = []
    for mode in CAPTURE_MODES_OF_INTEREST:
        for tgt in PARETO_RECALL_TARGETS:
            asis_row = per_mode_df[
                (per_mode_df["mode"] == mode)
                & (per_mode_df["score_arm"] == "prob_fake_asis")
                & (abs(per_mode_df["target_recall"] - tgt) < 1e-9)
            ]
            prod_row = per_mode_df[
                (per_mode_df["mode"] == mode)
                & (per_mode_df["score_arm"] == "prob_fake_prod")
                & (abs(per_mode_df["target_recall"] - tgt) < 1e-9)
            ]
            if asis_row.empty or prod_row.empty:
                continue
            ar = asis_row.iloc[0]
            pr = prod_row.iloc[0]
            pm_table_lines.append(
                f"| {mode} | n={int(ar['n_real'])} | {tgt:.2f} | "
                f"{_fmt(ar.get('min_fpr'))} | {_fmt(pr.get('min_fpr'))} | "
                f"{_fmt((pr.get('min_fpr') or 0) - (ar.get('min_fpr') or 0), '+.4f') if ar.get('min_fpr') is not None and pr.get('min_fpr') is not None else 'n/a'} |"
            )
    pm_table = "\n".join(pm_table_lines)

    # Per-method table at default-floor (or relaxed if default not selected)
    method_lines = []
    methods = sorted(per_method_df["method"].unique())
    for method in methods:
        for arm in ["prob_fake_asis", "prob_fake_prod"]:
            for floor in ["default_0.70", "relaxed_0.30"]:
                sub = per_method_df[
                    (per_method_df["method"] == method)
                    & (per_method_df["score_arm"] == arm)
                    & (per_method_df["subset"] == "full")
                    & (per_method_df["floor_policy"] == floor)
                ]
                if sub.empty:
                    continue
                row = sub.iloc[0]
                method_lines.append(
                    f"| {method} | {int(row['n_fake'])} | {arm} | {floor} | "
                    f"{_fmt(row['tau'])} | {_fmt(row['recall'])} |"
                )

    # Build report text
    lines = []
    lines.append("# P8A v3-Substrate Scorecard Analog (2026-05-01)")
    lines.append("")
    lines.append("## TL;DR")
    lines.append("")
    lines.append(
        f"Frame-level promotion-contract analog applied to P8A_step5000 lockbox "
        f"({md['n_lockbox_frames']} frames) on the v3-substrate (production-tight RFA=0.85 crop)."
    )
    lines.append("")
    lines.append(
        f"- **Default-floor (0.70) selection on prod arm**: "
        f"tier={sd_prod['selected_tier']} "
        f"({'PASSES' if sd_prod['selected_meets_recall_floor'] and sd_prod['selected_meets_fpr_budget'] else 'FAILS'} "
        f"contract). tau={_fmt(sd_prod['selected_threshold'])} "
        f"recall={_fmt(sd_prod['selected_fake_recall'])} fpr={_fmt(sd_prod['selected_real_fpr'])}."
    )
    lines.append(
        f"- **Relaxed-floor (0.30) selection on prod arm**: "
        f"tier={sr_prod['selected_tier']} "
        f"({'PASSES' if sr_prod['selected_meets_recall_floor'] and sr_prod['selected_meets_fpr_budget'] else 'FAILS'} "
        f"contract). tau={_fmt(sr_prod['selected_threshold'])} "
        f"recall={_fmt(sr_prod['selected_fake_recall'])} fpr={_fmt(sr_prod['selected_real_fpr'])}."
    )
    if asis_full_30 and prod_full_30:
        lines.append(
            f"- **Production-FPR gap at recall=0.30**: asis "
            f"min_fpr={_fmt(asis_full_30.get('min_fpr'))} -> "
            f"prod min_fpr={_fmt(prod_full_30.get('min_fpr'))} "
            f"(delta={_fmt(fpr_gap_30, '+.4f')})."
        )
    if asis_full_70 and prod_full_70:
        lines.append(
            f"- **Recall=0.70 reachability**: "
            f"asis "
            f"{'achievable' if asis_full_70['achievable'] else 'UNACHIEVABLE'} "
            f"(min_fpr={_fmt(asis_full_70.get('min_fpr'))}); "
            f"prod "
            f"{'achievable' if prod_full_70['achievable'] else 'UNACHIEVABLE'} "
            f"(min_fpr={_fmt(prod_full_70.get('min_fpr'))})."
        )
    lines.append(
        f"- **Canonical (video-level) reference**: tau=0.9156 lockbox_real_fpr=0.0184 "
        f"lockbox_fake_recall=0.387 dev_fake_macro_recall=0.300 (FAILS 0.70 floor)."
    )
    lines.append("")

    lines.append("## Method")
    lines.append("")
    lines.append(
        f"Loaded the v3-retag parquet ({md['n_lockbox_frames']} rows) and merged with "
        f"`analysis/lockbox_tagging/full_tags_2026-04-27.parquet` on `original_local_path` "
        f"to bring in `is_pose_extreme`, `is_no_face`, `face_area_ratio`. Built a quantile-spaced "
        f"threshold grid of up to {GRID_MAX_POINTS} points over the union of asis+prod prob "
        f"distributions."
    )
    lines.append("")
    lines.append(
        "For each tau, computed FPR (over reals) and recall (over fakes). Selection mirrors "
        "`arena/score_teams_promotion_contract._threshold_sort_key`: tier 0 = both FPR<=0.07 "
        "and recall>=floor; tier 1 = FPR<=0.07 only; tier 2 = FPR violated. Within tier, "
        "max recall, then highest tau, then lowest fpr."
    )
    lines.append("")
    lines.append(
        "**Caveat**: canonical scorer is video-level on dev pool then reads out lockbox. This "
        "analog is frame-level on lockbox-only — same underlying logic, but absolute numbers "
        "are not directly comparable to the canonical scorecard. Asis-vs-prod *delta* is the "
        "load-bearing signal."
    )
    lines.append("")

    lines.append("## Asis-vs-Prod Pareto Headline (full lockbox)")
    lines.append("")
    lines.append("| recall_target | asis min_fpr | prod min_fpr | delta (prod - asis) |")
    lines.append("|---|---|---|---|")
    for tgt in PARETO_RECALL_TARGETS:
        a = pareto_lookup(asis_full, tgt)
        p = pareto_lookup(prod_full, tgt)
        if a is None or p is None:
            continue
        a_fpr = a.get("min_fpr") if a["achievable"] else None
        p_fpr = p.get("min_fpr") if p["achievable"] else None
        delta = (p_fpr - a_fpr) if (a_fpr is not None and p_fpr is not None) else None
        lines.append(
            f"| {tgt:.2f} | {_fmt(a_fpr) if a['achievable'] else 'UNACH'} | "
            f"{_fmt(p_fpr) if p['achievable'] else 'UNACH'} | "
            f"{_fmt(delta, '+.4f') if delta is not None else 'n/a'} |"
        )
    lines.append("")

    lines.append("### modern_v2 subset")
    lines.append("")
    lines.append("| recall_target | asis min_fpr | prod min_fpr | delta |")
    lines.append("|---|---|---|---|")
    for tgt in PARETO_RECALL_TARGETS:
        a = pareto_lookup(asis_mv2, tgt)
        p = pareto_lookup(prod_mv2, tgt)
        if a is None or p is None:
            continue
        a_fpr = a.get("min_fpr") if a["achievable"] else None
        p_fpr = p.get("min_fpr") if p["achievable"] else None
        delta = (p_fpr - a_fpr) if (a_fpr is not None and p_fpr is not None) else None
        lines.append(
            f"| {tgt:.2f} | {_fmt(a_fpr) if a['achievable'] else 'UNACH'} | "
            f"{_fmt(p_fpr) if p['achievable'] else 'UNACH'} | "
            f"{_fmt(delta, '+.4f') if delta is not None else 'n/a'} |"
        )
    lines.append("")

    lines.append("## Per-Capture-Mode (asis vs prod, full lockbox)")
    lines.append("")
    lines.append("| mode | n_real | recall | asis min_fpr | prod min_fpr | delta |")
    lines.append("|---|---|---|---|---|---|")
    if pm_table:
        lines.append(pm_table)
    lines.append("")

    lines.append("## Per-Method Recall at Selected Taus (full lockbox)")
    lines.append("")
    lines.append("| method | n_fake | arm | floor | tau | recall |")
    lines.append("|---|---|---|---|---|---|")
    if method_lines:
        for ml in method_lines:
            lines.append(ml)
    lines.append("")

    lines.append("## Implication for P18 Baseline Floor")
    lines.append("")
    lines.append(
        "The asis-vs-prod delta isolates how much the canonical scorer overstates P8A's "
        "production-FPR. Any P18 candidate must (a) close the asis-prod FPR gap (or beat the "
        "prod-arm number directly), and (b) achieve recall=0.30 at fpr<=prod_full_30 to be a "
        "credible improvement over P8A on production-honest crops. Numbers above set that "
        "floor."
    )
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
