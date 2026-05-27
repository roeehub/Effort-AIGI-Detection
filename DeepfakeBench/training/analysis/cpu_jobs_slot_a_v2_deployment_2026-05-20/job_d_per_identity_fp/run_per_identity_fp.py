#!/usr/bin/env python3
"""
Job D — per-identity FPR audit.

Computes per-identity false-positive rate (FPR) for P8A vs Slot A v2 step3500
vs T5C step3500 at two threshold regimes:
  1. tau_dev: per-ckpt 93rd-percentile of `teams_real_all_dev` per-video scores
              (i.e. dev_real_FPR == 0.07). This is the contract-calibrated tau.
  2. tau_raw: tau = 0.5 (no calibration).

Identity scope
--------------
The capture suites `teams_capture_*_dev` are fake-only (label=1) and therefore
do not produce FPR numbers. Per-identity FPR is computed on real-only suites:
  - dev:      `teams_real_all_dev` (label=0). All eight requested identities are
              probed here when present. Md_noyn_Sharker is parsed from
              `teams_real_all_dev` (no per-identity dev suite of reals exists).
  - lockbox:  `teams_real_all_lockbox` (label=0). Chikara_Takahashi lives ONLY
              in lockbox. We report its lockbox FPR with an explicit
              "scope: lockbox" flag.

Identity = parsed prefix of `video_id`, stripping `__real`, `__seq<N>`,
`__sN__seg`, `__frame` suffixes (see `base_root`). Note that the dev `Q__s6`
videos all have root `Q`, and `bla_bla_chow` covers both `bla_bla_chow` and
`bla_bla_chow__s2` in the dev pool (filter root = "bla_bla_chow"), which
union to 467 videos.

Coverage caveats (Step 1 fact):
  - Chikara_Takahashi: not in any 29-suite by name AND absent from
    `teams_real_all_dev`. Reported from `teams_real_all_lockbox` only.
  - Roy_D: present in `teams_real_all_dev` (130 videos) and
    `teams_real_lighting_extreme_dev`. Reported using `teams_real_all_dev`
    (with `teams_real_lighting_extreme_dev` cross-checked).
  - Q: present in `teams_real_all_dev` as `Q__s6` (36 videos).
  - bla_bla_chow: present in `teams_real_all_dev` (467 videos).
  - Md_Noyn_Sharker: 409 videos in `teams_real_all_dev` as `Md_noyn_Sharker__s15`.
    The capture suites `teams_capture_noyn_sharker_*` carry FAKES, not reals;
    they are NOT used for FPR.

Outputs:
- tau_dev.json: per-ckpt tau values + diagnostic dev FPR achieved
- per_identity_fpr_tau_dev.csv
- per_identity_fpr_tau_raw.csv
- bar_results.csv
- bar_summary.json
- coverage.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT

CKPTS = {
    "P8A": "p8a_reference_step5000",
    "T5C": "t5c_periodic_step3500",
    "SlotAv2": "slot_a_anchor_aware_step3500",
}

# Each identity points to a list of (suite, scope) pairs. The first entry is
# the primary source; secondary entries are coverage cross-checks (not unioned
# into the primary FPR).
IDENTITY_SPEC = {
    "dor_shkedi": {
        "primary_suite": "teams_real_all_dev",
        "scope": "dev",
        "filter_root": ["dor_shkedi"],
    },
    "PC_Generator": {
        "primary_suite": "teams_real_all_dev",
        "scope": "dev",
        "filter_root": ["PC_Generator"],
    },
    "Cam_Test": {
        "primary_suite": "teams_real_all_dev",
        "scope": "dev",
        "filter_root": ["Cam_Test"],
    },
    "Test_Cam": {
        "primary_suite": "teams_real_all_dev",
        "scope": "dev",
        "filter_root": ["Test_Cam"],
    },
    "Md_Noyn_Sharker": {
        "primary_suite": "teams_real_all_dev",
        "scope": "dev",
        "filter_root": ["Md_noyn_Sharker"],
    },
    "Chikara_Takahashi": {
        "primary_suite": "teams_real_all_lockbox",
        "scope": "lockbox",
        "filter_root": ["Chikara_Takahashi"],
        "coverage_note": (
            "absent from teams_real_all_dev; reported on lockbox only. "
            "tau is dev-calibrated."
        ),
    },
    "Roy_D": {
        "primary_suite": "teams_real_all_dev",
        "scope": "dev",
        "filter_root": ["Roy_D"],
        "coverage_note": (
            "Roy_D appears as `Roy_D__seqNNNN__real` in teams_real_all_dev "
            "(130 videos) and also in teams_real_lighting_extreme_dev (cross-"
            "check). 29-suite 2026-05-20 scorecard does not include a Roy_D-"
            "specific suite."
        ),
    },
    "bla_bla_chow": {
        "primary_suite": "teams_real_all_dev",
        "scope": "dev",
        "filter_root": ["bla_bla_chow"],
    },
    "Q": {
        "primary_suite": "teams_real_all_dev",
        "scope": "dev",
        "filter_root": ["Q"],
    },
}

DEV_REAL_FPR_TARGET = 0.07  # contract-calibrated dev real FPR


def base_root(v: str) -> str:
    """Return root-level identity from video_id."""
    v = re.sub(r"__real$", "", v)
    m = re.match(r"(.+?)__seq\d+", v)
    if m:
        return m.group(1)
    m = re.match(r"(.+?)__s\d+__seg", v)
    if m:
        return m.group(1)
    m = re.match(r"(.+?)__frame", v)
    if m:
        return m.group(1)
    return re.sub(r"__seg.*", "", v)


def load_videos(suite: str, ckpt_slug: str) -> pd.DataFrame | None:
    path = DATA / f"{suite}_{ckpt_slug}_videos_report.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["base_root"] = df["video_id"].apply(base_root)
    df["suite"] = suite
    return df


def calibrate_tau(ckpt_slug: str) -> tuple[float, dict]:
    """tau such that fraction of teams_real_all_dev scores above tau == 0.07."""
    df = load_videos("teams_real_all_dev", ckpt_slug)
    scores = df["avg_video_prob"].to_numpy()
    tau = float(np.quantile(scores, 1 - DEV_REAL_FPR_TARGET))
    achieved = float((scores > tau).mean())
    return tau, {
        "n_videos_dev": int(len(scores)),
        "tau_dev_p93": tau,
        "achieved_dev_fpr": achieved,
        "score_p50": float(np.quantile(scores, 0.5)),
        "score_p95": float(np.quantile(scores, 0.95)),
        "score_p99": float(np.quantile(scores, 0.99)),
        "score_max": float(scores.max()),
    }


def gather_identity(spec: dict, ckpt_slug: str) -> pd.DataFrame:
    df = load_videos(spec["primary_suite"], ckpt_slug)
    if df is None:
        return pd.DataFrame(columns=["video_id", "avg_video_prob", "base_root"])
    targets = spec["filter_root"]
    return df[df["base_root"].isin(targets)].copy()


def main():
    # Step 2: tau per ckpt
    tau_table = {}
    diagnostics = {}
    for label, slug in CKPTS.items():
        tau, diag = calibrate_tau(slug)
        tau_table[label] = tau
        diagnostics[label] = diag
    (OUT / "tau_dev.json").write_text(
        json.dumps(
            {
                "method": "tau = quantile(scores_teams_real_all_dev, 1 - 0.07)",
                "dev_real_fpr_target": DEV_REAL_FPR_TARGET,
                "ckpts": diagnostics,
                "tau_values": tau_table,
            },
            indent=2,
        )
    )

    # Step 3: per-identity tables
    rows_dev = []
    rows_raw = []
    coverage = {}
    for identity, spec in IDENTITY_SPEC.items():
        cell_dev = {"identity": identity, "scope": spec["scope"]}
        cell_raw = {"identity": identity, "scope": spec["scope"]}
        n_per_ckpt = {}
        for label, slug in CKPTS.items():
            df = gather_identity(spec, slug)
            n = len(df)
            n_per_ckpt[label] = n
            if n == 0:
                cell_dev[f"{label}_FPR"] = np.nan
                cell_raw[f"{label}_FPR"] = np.nan
                continue
            scores = df["avg_video_prob"].to_numpy()
            tau = tau_table[label]
            cell_dev[f"{label}_FPR"] = float((scores > tau).mean())
            cell_raw[f"{label}_FPR"] = float((scores > 0.5).mean())
        cell_dev["n_videos"] = max(n_per_ckpt.values()) if n_per_ckpt else 0
        cell_raw["n_videos"] = cell_dev["n_videos"]
        if len(set(n_per_ckpt.values())) > 1:
            cell_dev["n_videos_note"] = f"varies: {n_per_ckpt}"
        # deltas
        if not (
            pd.isna(cell_dev.get("SlotAv2_FPR")) or pd.isna(cell_dev.get("P8A_FPR"))
        ):
            cell_dev["delta_SlotAv2_minus_P8A"] = (
                cell_dev["SlotAv2_FPR"] - cell_dev["P8A_FPR"]
            )
            cell_raw["delta_SlotAv2_minus_P8A"] = (
                cell_raw["SlotAv2_FPR"] - cell_raw["P8A_FPR"]
            )
        rows_dev.append(cell_dev)
        rows_raw.append(cell_raw)
        coverage[identity] = {
            "primary_suite": spec["primary_suite"],
            "scope": spec["scope"],
            "filter_root": spec["filter_root"],
            "n_videos_per_ckpt": n_per_ckpt,
            "coverage_note": spec.get("coverage_note", ""),
        }

    df_dev = pd.DataFrame(rows_dev)
    df_raw = pd.DataFrame(rows_raw)
    df_dev.to_csv(OUT / "per_identity_fpr_tau_dev.csv", index=False)
    df_raw.to_csv(OUT / "per_identity_fpr_tau_raw.csv", index=False)
    (OUT / "coverage.json").write_text(json.dumps(coverage, indent=2))

    # Step 4: Bar A / Bar B / Bar C
    bar_rows = []
    for row in rows_dev:
        identity = row["identity"]
        p8a = row.get("P8A_FPR", np.nan)
        slot = row.get("SlotAv2_FPR", np.nan)
        if pd.isna(p8a) or pd.isna(slot):
            bar_a = False
            bar_b = False
            note = "no data for one or both ckpts"
        else:
            bar_a = (p8a > 0.20) and (slot < 0.05)
            bar_b = (p8a < 0.10) and (slot > 0.30)
            note = ""
        bar_rows.append(
            {
                "identity": identity,
                "scope": row["scope"],
                "P8A_FPR": p8a,
                "SlotAv2_FPR": slot,
                "Bar_A_anchor_fix": bar_a,
                "Bar_B_anchor_overshoot": bar_b,
                "n_videos": row.get("n_videos", 0),
                "note": note,
            }
        )
    df_bars = pd.DataFrame(bar_rows)
    df_bars.to_csv(OUT / "bar_results.csv", index=False)
    bar_a_count = int(df_bars["Bar_A_anchor_fix"].sum())
    bar_b_count = int(df_bars["Bar_B_anchor_overshoot"].sum())
    bar_ratio = (
        float(bar_a_count) / bar_b_count if bar_b_count else float("inf")
    )
    (OUT / "bar_summary.json").write_text(
        json.dumps(
            {
                "Bar_A_anchor_fix_count": bar_a_count,
                "Bar_B_anchor_overshoot_count": bar_b_count,
                "Bar_C_ratio_A_over_B": bar_ratio,
            },
            indent=2,
        )
    )
    print("=== tau_dev ===")
    print(json.dumps({k: round(v, 6) for k, v in tau_table.items()}, indent=2))
    print("\n=== per_identity_fpr_tau_dev ===")
    print(df_dev.to_string(index=False))
    print("\n=== per_identity_fpr_tau_raw (tau=0.5) ===")
    print(df_raw.to_string(index=False))
    print("\n=== bar_results ===")
    print(df_bars.to_string(index=False))
    print(f"\nBar A count: {bar_a_count}")
    print(f"Bar B count: {bar_b_count}")
    print(f"Bar C ratio A/B: {bar_ratio}")

    # --- cross-check tables
    cross_rows = []
    # Roy_D in lighting_extreme dev (Roy_D-specific cross-check pool)
    for label, slug in CKPTS.items():
        df = load_videos("teams_real_lighting_extreme_dev", slug)
        if df is None:
            continue
        sub = df[df["base_root"] == "Roy_D"]
        if len(sub) == 0:
            continue
        cross_rows.append(
            {
                "identity": "Roy_D",
                "cross_pool": "teams_real_lighting_extreme_dev",
                "ckpt": label,
                "n_videos": len(sub),
                "FPR_tau_dev": float((sub["avg_video_prob"] > tau_table[label]).mean()),
                "mean_score": float(sub["avg_video_prob"].mean()),
            }
        )
    # Lockbox FPR for shared identities at dev tau
    for ident in ["PC_Generator", "bla_bla_chow", "dor_shkedi"]:
        for label, slug in CKPTS.items():
            df = load_videos("teams_real_all_lockbox", slug)
            if df is None:
                continue
            sub = df[df["base_root"] == ident]
            if len(sub) == 0:
                continue
            cross_rows.append(
                {
                    "identity": ident,
                    "cross_pool": "teams_real_all_lockbox",
                    "ckpt": label,
                    "n_videos": len(sub),
                    "FPR_tau_dev": float(
                        (sub["avg_video_prob"] > tau_table[label]).mean()
                    ),
                    "mean_score": float(sub["avg_video_prob"].mean()),
                }
            )
    df_cross = pd.DataFrame(cross_rows)
    df_cross.to_csv(OUT / "cross_check_tables.csv", index=False)
    print("\n=== cross_check_tables ===")
    print(df_cross.to_string(index=False))


if __name__ == "__main__":
    main()
