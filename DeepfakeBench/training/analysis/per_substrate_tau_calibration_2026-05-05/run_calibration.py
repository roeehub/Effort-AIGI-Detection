"""Per-substrate tau calibration for offline deployment-tau picking.

Background
----------
We ship a single global tau at deployment (no per-mode tau in production —
substrate is unobservable at inference time). But to PICK that tau, we want
to satisfy a worst-substrate FPR ceiling on a held-out real-frame set, not a
single-substrate FPR. P8A's score distribution on real frames varies
systematically by substrate, so calibrating tau against any one substrate
gives a misleading FPR (under- or over-controlled).

This tool: given a per-frame score CSV for ONE checkpoint, plus a substrate
manifest (how to bucket reals into substrate types) and a list of fake suites,
sweeps a tau grid and reports per-substrate FPR + per-suite recall at each tau.
Then picks tau_strict / tau_moderate / tau_loose at worst-substrate FPR ceilings
of 5% / 10% / 20%.

Inputs
------
The simplest form is a SINGLE CSV with columns:
    frame_path  score  label  [substrate]  [suite]

If `substrate` and `suite` are not present, they are derived from the
substrate manifest (capture-mode parquet lookup) and the file path layout
(e.g. one file per fake-suite report).

Outputs
-------
tau_sweep.csv             one row per tau, all per-substrate FPR + per-suite
                          recall columns
tau_recommendations.json  the three deployable taus + per-suite recall at each
substrate_assignment.csv  per-frame substrate label, for audit

Reference run
-------------
Built-in `--profile p8a_reference_step5000` packs the right inputs to
reproduce the Job-7 21pp lift claim on P8A (substrate-aware vs naive global
calibration, FPR<=5% on real lockbox).

Usage
-----
# Reference run on P8A:
python3 run_calibration.py --profile p8a_reference_step5000 \\
    --out reference_run_p8a/

# New checkpoint:
python3 run_calibration.py \\
    --real-dev path/to/teams_real_dev_<ckpt>.csv \\
    --real-lockbox path/to/teams_real_lockbox_<ckpt>.csv \\
    --fake-suite name=viso path=path/to/viso_dev_<ckpt>.csv \\
    --fake-suite name=deeplive path=path/to/deeplive_dev_<ckpt>.csv \\
    --fake-suite name=teams_fake_dev path=path/to/teams_fake_dev_<ckpt>.csv \\
    --fake-suite name=teams_fake_lockbox path=path/to/teams_fake_lockbox_<ckpt>.csv \\
    --substrate-source capture_mode_parquet \\
    --tags-parquet analysis/lockbox_tagging/full_tags_2026-04-27.parquet \\
    --score-col frame_prob --label-col label --path-col frame_path \\
    --out my_ckpt_calibration/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Defaults — pinned to repo paths so the tool works headless.
# ----------------------------------------------------------------------
ROOT = "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training"
DEFAULT_TAGS_PARQUET = f"{ROOT}/analysis/lockbox_tagging/full_tags_2026-04-27.parquet"
DEFAULT_RAW_REPORTS = f"{ROOT}/analysis/cpu_followups_2026-05-04/raw_reports"

# Capture-mode buckets we treat as substrates. Order = report column order.
DEFAULT_CAPTURE_MODES = [
    "normal_photo",
    "webcam",
    "phone_screen",
    "screen",
    "screen_recording",
]

DEFAULT_TAU_GRID_LO = 0.05
DEFAULT_TAU_GRID_HI = 0.95
DEFAULT_TAU_GRID_N = 100

FPR_TARGETS = {
    "tau_strict": 0.05,
    "tau_moderate": 0.10,
    "tau_loose": 0.20,
}

P8A_PROFILE = {
    "real_dev": f"{DEFAULT_RAW_REPORTS}/teams_real_all_dev_p8a_reference_step5000_frames_report.csv",
    "real_lockbox": f"{DEFAULT_RAW_REPORTS}/teams_real_all_lockbox_p8a_reference_step5000_frames_report.csv",
    "fake_suites": [
        ("viso_dev", f"{DEFAULT_RAW_REPORTS}/visomaster_enhanced_macro_dev_p8a_reference_step5000_frames_report.csv"),
        ("deeplive_dev", f"{DEFAULT_RAW_REPORTS}/deeplive_enhanced_dev_p8a_reference_step5000_frames_report.csv"),
        ("teams_fake_dev", f"{DEFAULT_RAW_REPORTS}/teams_fake_all_dev_p8a_reference_step5000_frames_report.csv"),
        ("teams_fake_lockbox", f"{DEFAULT_RAW_REPORTS}/teams_fake_all_lockbox_p8a_reference_step5000_frames_report.csv"),
    ],
    "score_col": "frame_prob",
    "label_col": "label",
    "path_col": "frame_path",
}


# ----------------------------------------------------------------------
# Substrate assignment
# ----------------------------------------------------------------------
def load_substrate_lookup(tags_parquet: str) -> pd.DataFrame:
    """Return DataFrame[frame_path, substrate, split].

    Substrate := clip_capture_mode bucket. Frames not in the parquet
    fall back to substrate='unknown' downstream.
    """
    if not os.path.exists(tags_parquet):
        raise FileNotFoundError(f"tags parquet missing: {tags_parquet}")
    cols = ["gcs_uri", "clip_capture_mode", "split"]
    tags = pd.read_parquet(tags_parquet, columns=cols)
    tags = tags.rename(columns={"gcs_uri": "frame_path", "clip_capture_mode": "substrate"})
    return tags


def attach_substrate(
    df: pd.DataFrame,
    substrate_lookup: pd.DataFrame,
    path_col: str,
) -> pd.DataFrame:
    """Add 'substrate' column. Frames not in lookup get 'unknown'."""
    df = df.copy()
    df = df.rename(columns={path_col: "frame_path"})
    df = df.merge(substrate_lookup[["frame_path", "substrate"]], on="frame_path", how="left")
    df["substrate"] = df["substrate"].fillna("unknown")
    return df


# ----------------------------------------------------------------------
# Score loading
# ----------------------------------------------------------------------
@dataclass
class ScoreFrame:
    name: str  # e.g. "real_dev", "real_lockbox", "viso_dev"
    df: pd.DataFrame  # columns: frame_path, score, label, substrate
    role: str  # "real" or "fake"


def load_score_csv(
    path: str,
    score_col: str,
    label_col: str,
    path_col: str,
) -> pd.DataFrame:
    """Load a frames_report CSV and standardize columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"score CSV missing: {path}")
    df = pd.read_csv(path)
    missing = [c for c in [score_col, label_col, path_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; have {list(df.columns)}")
    df = df.rename(columns={score_col: "score", label_col: "label", path_col: "frame_path"})
    return df[["frame_path", "score", "label"]]


# ----------------------------------------------------------------------
# Tau sweep
# ----------------------------------------------------------------------
def build_tau_grid(
    score_frames: List[ScoreFrame],
    n: int,
    lo: float,
    hi: float,
    quantile_anchored: bool = True,
) -> np.ndarray:
    """Mostly an evenly-spaced grid over [lo, hi]. If quantile_anchored, add
    extra grid points at the empirical real-score quantiles so we don't miss
    the FPR-target-crossing tau by interpolation slack."""
    grid = np.linspace(lo, hi, n)
    if quantile_anchored:
        anchors = []
        for sf in score_frames:
            if sf.role != "real":
                continue
            qs = sf.df["score"].quantile(np.linspace(0.5, 0.999, 50)).values
            anchors.append(qs)
        if anchors:
            extra = np.unique(np.concatenate(anchors))
            grid = np.unique(np.concatenate([grid, extra]))
    return grid


def per_substrate_fpr(
    real_df: pd.DataFrame, tau: float, substrates: List[str]
) -> Dict[str, float]:
    """For each substrate, fraction of reals with score >= tau."""
    out = {}
    for s in substrates:
        sub = real_df[real_df["substrate"] == s]
        if len(sub) == 0:
            out[s] = float("nan")
        else:
            out[s] = float((sub["score"] >= tau).mean())
    return out


def per_suite_recall(
    fake_frames: List[ScoreFrame], tau: float
) -> Dict[str, float]:
    """For each fake-suite frame, fraction with score >= tau."""
    out = {}
    for sf in fake_frames:
        if len(sf.df) == 0:
            out[sf.name] = float("nan")
        else:
            out[sf.name] = float((sf.df["score"] >= tau).mean())
    return out


def global_fpr(real_df: pd.DataFrame, tau: float) -> float:
    if len(real_df) == 0:
        return float("nan")
    return float((real_df["score"] >= tau).mean())


def sweep(
    real_dev_df: pd.DataFrame,
    real_lockbox_df: Optional[pd.DataFrame],
    fake_frames: List[ScoreFrame],
    tau_grid: np.ndarray,
    substrates: List[str],
) -> pd.DataFrame:
    """For each tau, compute per-substrate FPR (on real DEV — this is what
    you calibrate on) plus per-suite recall plus global FPR plus, if
    provided, real-lockbox FPR (substrate-bucketed and global)."""
    rows = []
    for tau in tau_grid:
        row = {"tau": float(tau)}
        # Calibration substrate (dev)
        dev_per_substrate = per_substrate_fpr(real_dev_df, tau, substrates)
        for s, v in dev_per_substrate.items():
            row[f"fpr_dev_{s}"] = v
        # Worst dev substrate
        finite = [v for v in dev_per_substrate.values() if not np.isnan(v)]
        row["fpr_dev_worst_substrate"] = max(finite) if finite else float("nan")
        row["fpr_dev_global"] = global_fpr(real_dev_df, tau)
        # Held-out (lockbox) — diagnostic only
        if real_lockbox_df is not None:
            lock_per_substrate = per_substrate_fpr(real_lockbox_df, tau, substrates)
            for s, v in lock_per_substrate.items():
                row[f"fpr_lockbox_{s}"] = v
            finite_lk = [v for v in lock_per_substrate.values() if not np.isnan(v)]
            row["fpr_lockbox_worst_substrate"] = max(finite_lk) if finite_lk else float("nan")
            row["fpr_lockbox_global"] = global_fpr(real_lockbox_df, tau)
        # Recall per suite
        suite_recall = per_suite_recall(fake_frames, tau)
        for s, v in suite_recall.items():
            row[f"recall_{s}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Recommendations
# ----------------------------------------------------------------------
def pick_tau(sweep_df: pd.DataFrame, fpr_target: float, worst_col: str) -> Optional[Dict]:
    """Lowest tau where worst_col <= fpr_target. Returns the row as dict, or
    None if no tau satisfies the ceiling."""
    eligible = sweep_df[sweep_df[worst_col] <= fpr_target + 1e-9]
    if eligible.empty:
        return None
    # Lowest tau = most permissive yet still meeting ceiling = max recall
    pick = eligible.sort_values("tau").iloc[0]
    return pick.to_dict()


def naive_global_pick_tau(
    sweep_df: pd.DataFrame, fpr_target: float, global_col: str = "fpr_dev_global"
) -> Optional[Dict]:
    """Naive baseline: pick tau using global pooled FPR (no substrate awareness)."""
    eligible = sweep_df[sweep_df[global_col] <= fpr_target + 1e-9]
    if eligible.empty:
        return None
    pick = eligible.sort_values("tau").iloc[0]
    return pick.to_dict()


def per_mode_tau_policy(
    real_dev_df: pd.DataFrame,
    real_lockbox_df: Optional[pd.DataFrame],
    fake_frames: List[ScoreFrame],
    substrates: List[str],
    fpr_target: float,
    fake_substrate_lookup: Optional[pd.DataFrame] = None,
) -> Dict:
    """Compare against an ORACLE per-mode-tau policy: pick a different tau per
    substrate so that within-substrate FPR equals fpr_target. NOT shippable
    (substrate is not observable at deployment) but it shows the offline
    upper-bound recall.

    Reports per-substrate tau, the implied lockbox FPR if those tau were
    applied (only on frames whose substrate is known), and total fake recall
    summed over fake suites with weighting = sample size."""
    per_mode_tau = {}
    for s in substrates:
        sub = real_dev_df[real_dev_df["substrate"] == s]
        if len(sub) == 0:
            per_mode_tau[s] = None
        else:
            per_mode_tau[s] = float(sub["score"].quantile(1.0 - fpr_target))
    out = {
        "fpr_target_per_substrate": fpr_target,
        "tau_per_substrate_dev_calibrated": per_mode_tau,
        "note": (
            "ORACLE policy — NOT deployable. Requires per-frame substrate at "
            "inference. Reported as upper bound for substrate-aware tuning lift."
        ),
    }
    if real_lockbox_df is not None:
        # Apply each substrate's tau only to its own frames in lockbox.
        flagged = 0
        covered = 0
        for s, t in per_mode_tau.items():
            if t is None:
                continue
            mask = real_lockbox_df["substrate"] == s
            covered += int(mask.sum())
            flagged += int(((real_lockbox_df.loc[mask, "score"] >= t)).sum())
        if covered > 0:
            out["lockbox_fpr_under_dev_per_mode_tau"] = flagged / covered
        else:
            out["lockbox_fpr_under_dev_per_mode_tau"] = float("nan")
        out["lockbox_n_real_covered"] = covered
        out["lockbox_n_real_total"] = len(real_lockbox_df)

    # Apply per-mode tau to fake suites only for frames whose substrate is known.
    if fake_substrate_lookup is not None:
        suite_recalls = {}
        for sf in fake_frames:
            d = sf.df.merge(
                fake_substrate_lookup[["frame_path", "substrate"]],
                on="frame_path", how="left",
            )
            d["substrate"] = d["substrate"].fillna("unknown")
            n_fake = len(d)
            n_caught = 0
            n_with_substrate = 0
            for s, t in per_mode_tau.items():
                if t is None:
                    continue
                mask = d["substrate"] == s
                n_with_substrate += int(mask.sum())
                n_caught += int((d.loc[mask, "score"] >= t).sum())
            if n_with_substrate > 0:
                suite_recalls[sf.name] = {
                    "recall_on_substrate_known": n_caught / n_with_substrate,
                    "n_frames_substrate_known": int(n_with_substrate),
                    "n_frames_total": int(n_fake),
                }
            else:
                suite_recalls[sf.name] = {
                    "recall_on_substrate_known": float("nan"),
                    "n_frames_substrate_known": 0,
                    "n_frames_total": int(n_fake),
                }
        out["per_suite_recall_under_dev_per_mode_tau"] = suite_recalls
    return out


def build_recommendations(
    sweep_df: pd.DataFrame,
    fake_suites: List[ScoreFrame],
    real_lockbox_attached: bool,
    real_dev_df: Optional[pd.DataFrame] = None,
    real_lockbox_df: Optional[pd.DataFrame] = None,
    substrates: Optional[List[str]] = None,
    substrate_lookup: Optional[pd.DataFrame] = None,
) -> Dict:
    rec = {
        "fpr_targets": FPR_TARGETS,
        "method": (
            "Three deployable taus picked by lowest tau where worst-substrate "
            "FPR on REAL DEV is at-or-below the FPR target. Substrate-aware "
            "single-tau policy: at deployment we cannot observe substrate, so "
            "we pick a single tau that bounds worst-substrate FPR. Naive "
            "baseline = lowest tau where global pooled FPR meets target "
            "(under-controls per-substrate FPR when score distributions differ "
            "across substrates). Per-mode tau is also reported as an oracle "
            "upper-bound (NOT deployable)."
        ),
        "selections": {},
    }
    for label, target in FPR_TARGETS.items():
        sub = pick_tau(sweep_df, target, worst_col="fpr_dev_worst_substrate")
        naive = naive_global_pick_tau(sweep_df, target)
        entry = {
            "fpr_target_worst_substrate": target,
            "substrate_aware_single_tau": _slim(sub, fake_suites, real_lockbox_attached),
            "naive_global_pooled_single_tau": _slim(naive, fake_suites, real_lockbox_attached),
        }
        if sub is not None and naive is not None:
            recall_lift = {}
            for sf in fake_suites:
                k = f"recall_{sf.name}"
                if k in sub and k in naive:
                    recall_lift[sf.name] = float(sub[k]) - float(naive[k])
            entry["recall_lift_substrate_aware_minus_naive_pp"] = {
                k: round(v * 100.0, 4) for k, v in recall_lift.items()
            }
            # Lockbox FPR comparison — substrate-aware should be lower & safer
            if real_lockbox_attached and "fpr_lockbox_global" in sub:
                entry["lockbox_fpr_substrate_aware_pct"] = round(
                    float(sub["fpr_lockbox_global"]) * 100.0, 4
                )
                entry["lockbox_fpr_naive_pct"] = round(
                    float(naive["fpr_lockbox_global"]) * 100.0, 4
                )
        # Per-mode oracle — reproduces Job-7-style lift
        if real_dev_df is not None and substrates is not None:
            entry["per_mode_tau_oracle"] = per_mode_tau_policy(
                real_dev_df=real_dev_df,
                real_lockbox_df=real_lockbox_df,
                fake_frames=fake_suites,
                substrates=substrates,
                fpr_target=target,
                fake_substrate_lookup=substrate_lookup,
            )
        rec["selections"][label] = entry
    return rec


def _slim(d: Optional[Dict], fake_suites: List[ScoreFrame], real_lockbox_attached: bool) -> Optional[Dict]:
    if d is None:
        return None
    out = {
        "tau": float(d["tau"]),
        "fpr_dev_worst_substrate": float(d["fpr_dev_worst_substrate"]),
        "fpr_dev_global": float(d["fpr_dev_global"]),
        "per_substrate_fpr_dev": {
            k.replace("fpr_dev_", ""): float(v)
            for k, v in d.items()
            if k.startswith("fpr_dev_")
            and k not in {"fpr_dev_worst_substrate", "fpr_dev_global"}
        },
        "per_suite_recall": {
            sf.name: float(d[f"recall_{sf.name}"]) for sf in fake_suites if f"recall_{sf.name}" in d
        },
    }
    if real_lockbox_attached:
        out["fpr_lockbox_worst_substrate"] = float(d.get("fpr_lockbox_worst_substrate", float("nan")))
        out["fpr_lockbox_global"] = float(d.get("fpr_lockbox_global", float("nan")))
        out["per_substrate_fpr_lockbox"] = {
            k.replace("fpr_lockbox_", ""): float(v)
            for k, v in d.items()
            if k.startswith("fpr_lockbox_")
            and k not in {"fpr_lockbox_worst_substrate", "fpr_lockbox_global"}
        }
    return out


# ----------------------------------------------------------------------
# Manifest writer
# ----------------------------------------------------------------------
def write_substrate_manifest(
    out_path: str,
    substrates: List[str],
    substrate_lookup: pd.DataFrame,
    score_frames: List[ScoreFrame],
    tags_parquet: str,
):
    """Persist substrate definition for reproducibility."""
    counts_per_frame_set = {}
    for sf in score_frames:
        counts_per_frame_set[sf.name] = (
            sf.df["substrate"].value_counts(dropna=False).to_dict()
        )
    manifest = {
        "substrate_source": "clip_capture_mode (from lockbox_tagging parquet)",
        "tags_parquet": tags_parquet,
        "substrates": substrates,
        "substrate_counts_in_parquet": substrate_lookup["substrate"]
        .value_counts(dropna=False)
        .to_dict(),
        "substrate_counts_per_score_frame": counts_per_frame_set,
        "notes": (
            "Frames not present in the tags parquet are bucketed as 'unknown' "
            "and appear under fpr_dev_unknown / fpr_lockbox_unknown sweep cols. "
            "Treat them as a real-frame substrate; if their FPR is the worst, "
            "selected tau is conservative w.r.t. them."
        ),
    }
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def run_pipeline(
    real_dev_path: str,
    real_lockbox_path: Optional[str],
    fake_suite_paths: List[Tuple[str, str]],
    score_col: str,
    label_col: str,
    path_col: str,
    out_dir: str,
    tags_parquet: str,
    substrates: List[str],
    tau_grid_n: int,
    tau_grid_lo: float,
    tau_grid_hi: float,
):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Substrate lookup
    print(f"[1/5] Loading substrate lookup from {tags_parquet} ...")
    substrate_lookup = load_substrate_lookup(tags_parquet)
    print(f"  {len(substrate_lookup)} tagged frames")

    # 2. Load real DEV (calibration)
    print(f"[2/5] Loading real DEV scores from {real_dev_path}")
    real_dev_raw = load_score_csv(real_dev_path, score_col, label_col, path_col)
    real_dev = attach_substrate(real_dev_raw, substrate_lookup, "frame_path")
    real_dev_sf = ScoreFrame(name="real_dev", df=real_dev, role="real")
    print(f"  real_dev: {len(real_dev)} frames; substrate breakdown: "
          f"{real_dev['substrate'].value_counts(dropna=False).to_dict()}")

    # 3. Optional real LOCKBOX (held-out diagnostic)
    real_lockbox = None
    real_lockbox_sf = None
    if real_lockbox_path:
        print(f"[3/5] Loading real LOCKBOX scores from {real_lockbox_path}")
        real_lockbox_raw = load_score_csv(real_lockbox_path, score_col, label_col, path_col)
        real_lockbox = attach_substrate(real_lockbox_raw, substrate_lookup, "frame_path")
        real_lockbox_sf = ScoreFrame(name="real_lockbox", df=real_lockbox, role="real")
        print(f"  real_lockbox: {len(real_lockbox)} frames; substrate breakdown: "
              f"{real_lockbox['substrate'].value_counts(dropna=False).to_dict()}")
    else:
        print("[3/5] No real_lockbox path supplied; held-out FPR not computed.")

    # 4. Load fake suites
    print(f"[4/5] Loading {len(fake_suite_paths)} fake suite(s) ...")
    fake_frames: List[ScoreFrame] = []
    for name, path in fake_suite_paths:
        f = load_score_csv(path, score_col, label_col, path_col)
        # No substrate attach needed — recall is suite-level.
        fake_frames.append(ScoreFrame(name=name, df=f, role="fake"))
        print(f"  {name}: {len(f)} fake frames")

    # 5. Sweep
    score_frames_for_grid: List[ScoreFrame] = [real_dev_sf]
    if real_lockbox_sf is not None:
        score_frames_for_grid.append(real_lockbox_sf)
    tau_grid = build_tau_grid(
        score_frames_for_grid, n=tau_grid_n, lo=tau_grid_lo, hi=tau_grid_hi
    )
    print(f"[5/5] Sweeping {len(tau_grid)} tau values "
          f"(linspace n={tau_grid_n} + quantile anchors)")
    sweep_df = sweep(
        real_dev_df=real_dev,
        real_lockbox_df=real_lockbox,
        fake_frames=fake_frames,
        tau_grid=tau_grid,
        substrates=substrates,
    )
    sweep_csv = os.path.join(out_dir, "tau_sweep.csv")
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"  wrote {sweep_csv} ({len(sweep_df)} rows)")

    # Substrate manifest
    score_frames_with_substrate = [real_dev_sf]
    if real_lockbox_sf is not None:
        score_frames_with_substrate.append(real_lockbox_sf)
    manifest_path = os.path.join(out_dir, "substrate_manifest.json")
    write_substrate_manifest(
        manifest_path, substrates, substrate_lookup, score_frames_with_substrate,
        tags_parquet=tags_parquet,
    )
    print(f"  wrote {manifest_path}")

    # Per-frame substrate audit
    audit_rows = []
    for sf in score_frames_with_substrate:
        d = sf.df.copy()
        d["score_frame"] = sf.name
        audit_rows.append(d[["score_frame", "frame_path", "score", "label", "substrate"]])
    audit_df = pd.concat(audit_rows, ignore_index=True)
    audit_path = os.path.join(out_dir, "substrate_assignment.csv")
    audit_df.to_csv(audit_path, index=False)
    print(f"  wrote {audit_path}")

    # Recommendations
    rec = build_recommendations(
        sweep_df,
        fake_frames,
        real_lockbox_attached=(real_lockbox is not None),
        real_dev_df=real_dev,
        real_lockbox_df=real_lockbox,
        substrates=substrates,
        substrate_lookup=substrate_lookup,
    )
    rec_path = os.path.join(out_dir, "tau_recommendations.json")
    with open(rec_path, "w") as f:
        json.dump(rec, f, indent=2, default=str)
    print(f"  wrote {rec_path}")

    return sweep_df, rec


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--profile",
        choices=["p8a_reference_step5000"],
        help="Pin all input paths to a known checkpoint reference run.",
    )
    p.add_argument("--real-dev", help="Path to real DEV scores CSV (calibration).")
    p.add_argument(
        "--real-lockbox",
        help="Path to held-out real LOCKBOX scores CSV (diagnostic, optional).",
    )
    p.add_argument(
        "--fake-suite",
        action="append",
        default=[],
        help="Repeatable. Format: name=foo path=/abs/path.csv",
    )
    p.add_argument("--score-col", default="frame_prob")
    p.add_argument("--label-col", default="label")
    p.add_argument("--path-col", default="frame_path")
    p.add_argument(
        "--substrate-source",
        choices=["capture_mode_parquet"],
        default="capture_mode_parquet",
        help="Currently only one substrate source is supported.",
    )
    p.add_argument("--tags-parquet", default=DEFAULT_TAGS_PARQUET)
    p.add_argument(
        "--substrates",
        nargs="*",
        default=DEFAULT_CAPTURE_MODES,
        help="Substrate names to evaluate. Default: capture-mode buckets.",
    )
    p.add_argument("--tau-grid-n", type=int, default=DEFAULT_TAU_GRID_N)
    p.add_argument("--tau-grid-lo", type=float, default=DEFAULT_TAU_GRID_LO)
    p.add_argument("--tau-grid-hi", type=float, default=DEFAULT_TAU_GRID_HI)
    p.add_argument("--out", required=True, help="Output directory.")
    return p.parse_args(argv)


def parse_fake_suite_arg(s: str) -> Tuple[str, str]:
    parts = dict(kv.split("=", 1) for kv in s.split() if "=" in kv)
    if "name" not in parts or "path" not in parts:
        raise ValueError(f"--fake-suite arg must be 'name=X path=Y', got: {s}")
    return parts["name"], parts["path"]


def main(argv=None):
    args = parse_args(argv)

    if args.profile == "p8a_reference_step5000":
        real_dev = P8A_PROFILE["real_dev"]
        real_lockbox = P8A_PROFILE["real_lockbox"]
        fake_suite_paths = list(P8A_PROFILE["fake_suites"])
        score_col = P8A_PROFILE["score_col"]
        label_col = P8A_PROFILE["label_col"]
        path_col = P8A_PROFILE["path_col"]
    else:
        if not args.real_dev:
            raise SystemExit("--real-dev is required when --profile is not used")
        real_dev = args.real_dev
        real_lockbox = args.real_lockbox
        fake_suite_paths = [parse_fake_suite_arg(s) for s in args.fake_suite]
        score_col = args.score_col
        label_col = args.label_col
        path_col = args.path_col

    sweep_df, rec = run_pipeline(
        real_dev_path=real_dev,
        real_lockbox_path=real_lockbox,
        fake_suite_paths=fake_suite_paths,
        score_col=score_col,
        label_col=label_col,
        path_col=path_col,
        out_dir=args.out,
        tags_parquet=args.tags_parquet,
        substrates=args.substrates,
        tau_grid_n=args.tau_grid_n,
        tau_grid_lo=args.tau_grid_lo,
        tau_grid_hi=args.tau_grid_hi,
    )

    print("\n=== Recommendations ===")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
