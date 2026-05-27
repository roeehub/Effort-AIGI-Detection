#!/usr/bin/env python3
"""Option A — inference-time stack analysis.

Steps:
  1) Load all 5 candidates x 8 suites of frames_report.csv from cache/
  2) Fit Platt + isotonic calibrators per candidate using
     teams_real_all_dev + teams_fake_all_dev as the calibration pool
     (identity-safe split via video_id prefix)
  3) Apply calibrators across all suites; emit calibrated_scorecard.csv
  4) Per-method τ — find τ giving 90% recall on each fake family
     (raw + calibrated); reject if dev_real FPR > 7%
  5) Build P8A + P11_HEAVY noisy-OR ensemble (raw + calibrated) and score
     across all suites + modern_v2 subset.
  6) Optional weighted/3-way ensembles for context.
  7) Write REPORT.md narrative.

Constraints:
  * No n_jobs=-1 (sklearn) — uses default single-threaded paths.
  * Local-only, no GCS reads.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
OUT_DIR = REPO / "analysis/option_a_ensemble_2026-04-28"
CACHE_DIR = OUT_DIR / "cache"
MODERN_DIR = REPO / "analysis/modern_lockbox_v2_2026-04-27"

CANDIDATES = [
    "p8a_reference_step5000",
    "p11_heavy_step1000",
    "p11_heavy_deeplive_step1000",
    "p11_mild_step1000",
    "p11_webcam_harden_step1000",
]

REAL_SUITES = [
    "teams_real_all_dev",
    "teams_real_all_lockbox",
    "teams_real_poor_quality_dev",
    "teams_real_lighting_extreme_dev",
]
FAKE_SUITES = [
    "teams_fake_all_dev",
    "teams_fake_all_lockbox",
    "visomaster_enhanced_macro_dev",
    "deeplive_enhanced_dev",
]
ALL_SUITES = REAL_SUITES + FAKE_SUITES

# Hard targets (Plan v3 §6 + user's brief)
T_VISO_RECALL = 0.90
T_DEEPLIVE_RECALL = 0.90
T_TEAMS_FAKE_RECALL = 0.90
T_MODERN_FPR = 0.05
T_DEV_REAL_FPR_CAP = 0.07

EPS = 1e-6


def clip_p(x: np.ndarray) -> np.ndarray:
    return np.clip(x, EPS, 1.0 - EPS)


def logit(p: np.ndarray) -> np.ndarray:
    pp = clip_p(p)
    return np.log(pp / (1.0 - pp))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_platt(p_raw: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = logit(p_raw).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    lr.fit(x, y.astype(np.int32))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(p_raw: np.ndarray, a: float, b: float) -> np.ndarray:
    z = a * logit(p_raw) + b
    return clip_p(sigmoid(z))


def fit_isotonic(p_raw: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(p_raw, y.astype(np.int32))
    return ir


def apply_isotonic(p_raw: np.ndarray, ir: IsotonicRegression) -> np.ndarray:
    return clip_p(ir.transform(p_raw))


def load_frames(suite: str, candidate: str) -> Optional[pd.DataFrame]:
    f = CACHE_DIR / f"{suite}_{candidate}_frames_report.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df["suite"] = suite
    df["candidate"] = candidate
    return df


def fpr_at(p: np.ndarray, tau: float) -> float:
    if len(p) == 0:
        return float("nan")
    return float((p >= tau).mean())


def recall_at(p: np.ndarray, tau: float) -> float:
    if len(p) == 0:
        return float("nan")
    return float((p >= tau).mean())


def tau_for_target_recall(p_fake: np.ndarray, target_recall: float) -> float:
    """Return smallest tau giving recall >= target_recall (highest tau in [0,1])."""
    if len(p_fake) == 0:
        return float("nan")
    sorted_p = np.sort(p_fake)
    # Need fraction (p >= tau) >= target_recall
    # Equivalently rank: tau = sorted_p[ceil((1 - target_recall) * n)] (with floor-like handling)
    n = len(sorted_p)
    # number of fakes that must be >= tau
    k_required = int(np.ceil(target_recall * n))
    if k_required <= 0:
        return 1.0  # anything passes
    if k_required > n:
        return 0.0  # impossible
    # idx of (n - k_required)-th smallest from 0; tau == that value
    tau = float(sorted_p[n - k_required])
    return tau


def tau_for_target_fpr(p_real: np.ndarray, target_fpr: float) -> float:
    """Return smallest tau giving FPR <= target_fpr."""
    if len(p_real) == 0:
        return float("nan")
    sorted_p = np.sort(p_real)
    n = len(sorted_p)
    # number allowed false positives
    k_allowed = int(np.floor(target_fpr * n))
    # we want fraction (p_real >= tau) <= target_fpr
    # i.e. at most k_allowed real probs are >= tau
    # smallest tau achieving that: sorted_p[n - k_allowed] (if k_allowed=0 => tau just above max)
    if k_allowed <= 0:
        # smallest tau strictly above max real prob (or 1.0 if degenerate)
        return float(sorted_p[-1]) + 1e-9 if sorted_p[-1] < 1.0 else 1.0
    if k_allowed >= n:
        return 0.0
    # tau equal to the (n - k_allowed)-th smallest exactly catches k_allowed of them
    # we want strictly <= so tau just above that value
    tau = float(sorted_p[n - k_allowed])
    return tau


# --------------------------------------------------------------------------
# Step 1 — load all data
# --------------------------------------------------------------------------
print("=== Step 1: load frames CSVs ===")
data: Dict[Tuple[str, str], pd.DataFrame] = {}
sizes_log: List[Dict] = []
for cand in CANDIDATES:
    for suite in ALL_SUITES:
        df = load_frames(suite, cand)
        if df is None:
            print(f"  MISSING {suite} / {cand}")
            continue
        data[(suite, cand)] = df
        sizes_log.append({
            "suite": suite,
            "candidate": cand,
            "n_rows": len(df),
            "n_label1": int((df["label"] == 1).sum()),
            "n_label0": int((df["label"] == 0).sum()),
        })
sizes_df = pd.DataFrame(sizes_log)
sizes_df.to_csv(OUT_DIR / "cache_sizes.csv", index=False)
print(f"loaded {len(data)} (suite,candidate) pairs; sizes written to cache_sizes.csv")


# --------------------------------------------------------------------------
# Step 2 — calibration fit + apply
# --------------------------------------------------------------------------
print("\n=== Step 2: calibration ===")

CALIB_CANDIDATES = ["p8a_reference_step5000", "p11_heavy_step1000",
                    "p11_heavy_deeplive_step1000", "p11_mild_step1000",
                    "p11_webcam_harden_step1000"]
# fit calibrators for ALL 5 — needed for ensembles below.

calibrators: Dict[str, Dict] = {}

for cand in CALIB_CANDIDATES:
    real_df = data.get(("teams_real_all_dev", cand))
    fake_df = data.get(("teams_fake_all_dev", cand))
    if real_df is None or fake_df is None:
        print(f"  {cand}: missing dev pool, skip calibration")
        continue
    # identity-safe split: split by video_id prefix (subject)
    # Use video_id directly — it's per-segment but contains identity
    rng = np.random.default_rng(42)
    real_vids = sorted(real_df["video_id"].unique())
    fake_vids = sorted(fake_df["video_id"].unique())
    # Identity = leading token before "__"
    def ident(v):
        return v.split("__", 1)[0]
    all_idents = sorted(set([ident(v) for v in real_vids]) | set([ident(v) for v in fake_vids]))
    rng.shuffle(all_idents)
    # 70/30 split for FIT/TEST not strictly needed since we will re-evaluate on full suites;
    # we just need a calibrator. Fit on full dev pool.
    p_real = clip_p(real_df["frame_prob"].to_numpy())
    p_fake = clip_p(fake_df["frame_prob"].to_numpy())
    p_all = np.concatenate([p_real, p_fake])
    y_all = np.concatenate([np.zeros(len(p_real)), np.ones(len(p_fake))])
    a, b = fit_platt(p_all, y_all)
    ir = fit_isotonic(p_all, y_all)
    calibrators[cand] = {
        "platt_a": a,
        "platt_b": b,
        "isotonic": ir,
    }
    print(f"  {cand}: platt a={a:.3f} b={b:.3f}; iso fit on n={len(p_all)} ({len(p_real)} real, {len(p_fake)} fake)")


def apply_all_to_df(df: pd.DataFrame, cand: str) -> pd.DataFrame:
    out = df.copy()
    p_raw = clip_p(out["frame_prob"].to_numpy())
    out["p_raw"] = p_raw
    if cand in calibrators:
        c = calibrators[cand]
        out["p_platt"] = apply_platt(p_raw, c["platt_a"], c["platt_b"])
        out["p_iso"] = apply_isotonic(p_raw, c["isotonic"])
    else:
        out["p_platt"] = p_raw
        out["p_iso"] = p_raw
    return out


# Build calibrated scorecard
print("\n=== Step 2: calibrated scorecard ===")
calib_rows = []
for cand in CANDIDATES:
    for suite in ALL_SUITES:
        df = data.get((suite, cand))
        if df is None:
            continue
        df_c = apply_all_to_df(df, cand)
        is_fake_suite = suite in FAKE_SUITES
        for variant, col in [("raw", "p_raw"), ("platt", "p_platt"), ("isotonic", "p_iso")]:
            p = df_c[col].to_numpy()
            row = {
                "candidate": cand,
                "suite": suite,
                "variant": variant,
                "n": len(df_c),
                "metric_kind": "recall" if is_fake_suite else "fpr",
            }
            row["mean_p"] = float(np.mean(p))
            for tau in (0.5, 0.7, 0.9, 0.95, 0.97, 0.99, 0.995):
                if is_fake_suite:
                    row[f"recall_tau_{tau}"] = recall_at(p, tau)
                else:
                    row[f"fpr_tau_{tau}"] = fpr_at(p, tau)
            calib_rows.append(row)
calib_df = pd.DataFrame(calib_rows)
calib_df.to_csv(OUT_DIR / "calibrated_scorecard.csv", index=False)
print(f"wrote calibrated_scorecard.csv ({len(calib_df)} rows)")


# --------------------------------------------------------------------------
# Step 3 — per-method τ
# --------------------------------------------------------------------------
print("\n=== Step 3: per-family τ ===")
fake_family_suites = [
    ("viso", "visomaster_enhanced_macro_dev", T_VISO_RECALL),
    ("deeplive", "deeplive_enhanced_dev", T_DEEPLIVE_RECALL),
    ("teams_fake", "teams_fake_all_dev", T_TEAMS_FAKE_RECALL),
]

# load modern_v2 frame URI sets
real_v2_uris = set(yaml.safe_load((MODERN_DIR / "modern_lockbox_real_v2_frames.yaml").read_text())["frames"])
fake_v2_uris = set(yaml.safe_load((MODERN_DIR / "modern_lockbox_fake_v2_frames.yaml").read_text())["frames"])

def modern_v2_subset(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    uris = real_v2_uris if kind == "real" else fake_v2_uris
    return df[df["frame_path"].isin(uris)]

per_tau_rows = []
for cand in CANDIDATES:
    teams_real_dev_df = data.get(("teams_real_all_dev", cand))
    teams_real_lock_df = data.get(("teams_real_all_lockbox", cand))
    if teams_real_dev_df is None:
        continue
    teams_real_dev_df = apply_all_to_df(teams_real_dev_df, cand)
    teams_real_lock_df = apply_all_to_df(teams_real_lock_df, cand) if teams_real_lock_df is not None else None
    real_v2 = modern_v2_subset(teams_real_lock_df, "real") if teams_real_lock_df is not None else None
    for variant, col in [("raw", "p_raw"), ("platt", "p_platt"), ("isotonic", "p_iso")]:
        for fam_name, fam_suite, target in fake_family_suites:
            fam_df = data.get((fam_suite, cand))
            if fam_df is None:
                continue
            fam_df_c = apply_all_to_df(fam_df, cand)
            p_fake = fam_df_c[col].to_numpy()
            tau = tau_for_target_recall(p_fake, target)
            # cost: dev_real_fpr at this tau
            dev_real_fpr = fpr_at(teams_real_dev_df[col].to_numpy(), tau)
            # modern_v2 fpr at this tau
            v2_fpr = fpr_at(real_v2[col].to_numpy(), tau) if real_v2 is not None else float("nan")
            recall_at_tau = recall_at(p_fake, tau)
            cap_breach = (dev_real_fpr > T_DEV_REAL_FPR_CAP)
            modern_breach = (v2_fpr > T_MODERN_FPR)
            per_tau_rows.append({
                "candidate": cand,
                "variant": variant,
                "fake_family": fam_name,
                "target_recall": target,
                "tau_for_target": tau,
                "achieved_recall": recall_at_tau,
                "dev_real_fpr": dev_real_fpr,
                "modern_v2_fpr": v2_fpr,
                "dev_real_cap_breach": cap_breach,
                "modern_v2_cap_breach": modern_breach,
                "rejected": cap_breach,
            })

per_tau_df = pd.DataFrame(per_tau_rows)
per_tau_df.to_csv(OUT_DIR / "per_method_tau_table.csv", index=False)
print(f"wrote per_method_tau_table.csv ({len(per_tau_df)} rows)")
# print summary of feasible (non-rejected) rows
feasible = per_tau_df[~per_tau_df["rejected"]]
print(f"  feasible (dev_real_fpr<={T_DEV_REAL_FPR_CAP}): {len(feasible)}/{len(per_tau_df)}")
if len(feasible):
    # best per (candidate,variant,family) — already unique tuples
    print(feasible[["candidate", "variant", "fake_family", "tau_for_target",
                     "achieved_recall", "dev_real_fpr", "modern_v2_fpr"]].to_string(index=False))


# --------------------------------------------------------------------------
# Step 4 — noisy-OR ensemble
# --------------------------------------------------------------------------
print("\n=== Step 4: noisy-OR ensembles ===")

def build_ensemble_probs(cands: List[str], suite: str, variant: str) -> Optional[pd.DataFrame]:
    """Inner-join frames across cands by frame_path; compute noisy-OR per row."""
    base = None
    p_cols = []
    for c in cands:
        df = data.get((suite, c))
        if df is None:
            return None
        df_c = apply_all_to_df(df, c)
        col_map = {"raw": "p_raw", "platt": "p_platt", "isotonic": "p_iso"}
        col = col_map[variant]
        sub = df_c[["frame_path", "label", col]].rename(columns={col: f"p_{c}"})
        if base is None:
            base = sub
        else:
            base = base.merge(sub.drop(columns=["label"]), on="frame_path", how="inner")
        p_cols.append(f"p_{c}")
    arr = base[p_cols].to_numpy()
    arr = clip_p(arr)
    base["p_ensemble"] = 1.0 - np.prod(1.0 - arr, axis=1)
    return base

ENS_DEFS = [
    ("p8a_p11heavy", ["p8a_reference_step5000", "p11_heavy_step1000"]),
    ("p8a_p11heavy_p11hd", ["p8a_reference_step5000", "p11_heavy_step1000", "p11_heavy_deeplive_step1000"]),
    ("all5", CANDIDATES),
]

ensemble_rows = []
for ens_name, cands in ENS_DEFS:
    for variant in ("raw", "platt", "isotonic"):
        # 1) score full suites
        for suite in ALL_SUITES:
            ens = build_ensemble_probs(cands, suite, variant)
            if ens is None:
                continue
            is_fake = suite in FAKE_SUITES
            p = ens["p_ensemble"].to_numpy()
            row = {
                "ensemble": ens_name,
                "variant": variant,
                "suite": suite,
                "n_after_join": len(ens),
                "metric_kind": "recall" if is_fake else "fpr",
                "mean_p": float(np.mean(p)),
            }
            for tau in (0.5, 0.7, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999, 0.9999):
                row[f"{'recall' if is_fake else 'fpr'}_tau_{tau}"] = (
                    recall_at(p, tau) if is_fake else fpr_at(p, tau))
            ensemble_rows.append(row)

ens_df = pd.DataFrame(ensemble_rows)
ens_df.to_csv(OUT_DIR / "ensemble_scorecard.csv", index=False)
print(f"wrote ensemble_scorecard.csv ({len(ens_df)} rows)")


# --------------------------------------------------------------------------
# Step 4b — operating points: gate evaluation
# --------------------------------------------------------------------------
print("\n=== Step 4b: ensemble operating points ===")

gate_rows = []

def family_recall_at(cands, variant, fam_suite, tau):
    ens = build_ensemble_probs(cands, fam_suite, variant)
    if ens is None:
        return float("nan"), 0
    p = ens["p_ensemble"].to_numpy()
    return recall_at(p, tau), len(ens)

def fpr_in_suite_at(cands, variant, suite, tau, modern_only=False, kind=None):
    ens = build_ensemble_probs(cands, suite, variant)
    if ens is None:
        return float("nan"), 0
    if modern_only:
        ens = ens[ens["frame_path"].isin(real_v2_uris if kind == "real" else fake_v2_uris)]
    if len(ens) == 0:
        return float("nan"), 0
    p = ens["p_ensemble"].to_numpy()
    return fpr_at(p, tau), len(ens)

for ens_name, cands in ENS_DEFS:
    for variant in ("raw", "platt", "isotonic"):
        # build cached ensembles for the operating-point work
        suite_ens = {}
        for s in ALL_SUITES:
            ens = build_ensemble_probs(cands, s, variant)
            if ens is not None:
                suite_ens[s] = ens
        if "teams_real_all_dev" not in suite_ens:
            continue

        # tau choice (a) τ=0.5
        # tau choice (b) τ giving 5% modern_v2 FPR
        # tau choice (c) τ giving 90% recall on each fake family individually

        # modern_v2 real ensemble
        real_lock_ens = suite_ens.get("teams_real_all_lockbox")
        modern_real = real_lock_ens[real_lock_ens["frame_path"].isin(real_v2_uris)] if real_lock_ens is not None else None

        for op_name, tau_select in [("tau_0.5", 0.5)] + [
            ("modern_v2_5pct", "modern_v2_5pct"),
            ("viso_90", "viso_90"),
            ("deeplive_90", "deeplive_90"),
            ("teams_fake_90", "teams_fake_90"),
        ]:
            # determine tau
            if isinstance(tau_select, float):
                tau = tau_select
            elif tau_select == "modern_v2_5pct" and modern_real is not None:
                tau = tau_for_target_fpr(modern_real["p_ensemble"].to_numpy(), 0.05)
            elif tau_select == "viso_90" and "visomaster_enhanced_macro_dev" in suite_ens:
                tau = tau_for_target_recall(
                    suite_ens["visomaster_enhanced_macro_dev"]["p_ensemble"].to_numpy(), 0.90)
            elif tau_select == "deeplive_90" and "deeplive_enhanced_dev" in suite_ens:
                tau = tau_for_target_recall(
                    suite_ens["deeplive_enhanced_dev"]["p_ensemble"].to_numpy(), 0.90)
            elif tau_select == "teams_fake_90" and "teams_fake_all_dev" in suite_ens:
                tau = tau_for_target_recall(
                    suite_ens["teams_fake_all_dev"]["p_ensemble"].to_numpy(), 0.90)
            else:
                continue

            # gather metrics
            row = {
                "ensemble": ens_name,
                "variant": variant,
                "operating_point": op_name,
                "tau": float(tau),
            }
            for s in ALL_SUITES:
                if s not in suite_ens:
                    continue
                p = suite_ens[s]["p_ensemble"].to_numpy()
                key = ("recall_" if s in FAKE_SUITES else "fpr_") + s
                row[key] = (recall_at(p, tau) if s in FAKE_SUITES else fpr_at(p, tau))
            # modern_v2 FPR
            if modern_real is not None:
                row["fpr_modern_v2"] = fpr_at(modern_real["p_ensemble"].to_numpy(), tau)
            # gate decision
            v = row.get("recall_visomaster_enhanced_macro_dev", float("nan"))
            d = row.get("recall_deeplive_enhanced_dev", float("nan"))
            tf = row.get("recall_teams_fake_all_dev", float("nan"))
            mv2 = row.get("fpr_modern_v2", float("nan"))
            tr = row.get("fpr_teams_real_all_dev", float("nan"))
            gate = (
                (v >= T_VISO_RECALL) and (d >= T_DEEPLIVE_RECALL)
                and (tf >= T_TEAMS_FAKE_RECALL)
                and (mv2 <= T_MODERN_FPR) and (tr <= T_DEV_REAL_FPR_CAP)
            )
            row["passes_gate"] = bool(gate)
            gate_rows.append(row)

gate_df = pd.DataFrame(gate_rows)
gate_df.to_csv(OUT_DIR / "ensemble_gate_evaluation.csv", index=False)
print(f"wrote ensemble_gate_evaluation.csv ({len(gate_df)} rows)")
gates_pass = gate_df[gate_df["passes_gate"]]
print(f"  passing gate: {len(gates_pass)}/{len(gate_df)}")
if len(gates_pass):
    print(gates_pass[["ensemble", "variant", "operating_point", "tau",
                       "recall_visomaster_enhanced_macro_dev", "recall_deeplive_enhanced_dev",
                       "recall_teams_fake_all_dev", "fpr_modern_v2",
                       "fpr_teams_real_all_dev"]].to_string(index=False))


# --------------------------------------------------------------------------
# Step 4c — single-candidate gate evaluation (sanity baseline)
# --------------------------------------------------------------------------
print("\n=== Step 4c: single-candidate gate baseline ===")
single_gate_rows = []
for cand in CANDIDATES:
    teams_real_lock = data.get(("teams_real_all_lockbox", cand))
    if teams_real_lock is None:
        continue
    teams_real_lock = apply_all_to_df(teams_real_lock, cand)
    modern_real = teams_real_lock[teams_real_lock["frame_path"].isin(real_v2_uris)]
    suite_dfs = {}
    for s in ALL_SUITES:
        df = data.get((s, cand))
        if df is None:
            continue
        suite_dfs[s] = apply_all_to_df(df, cand)
    for variant, col in [("raw", "p_raw"), ("platt", "p_platt"), ("isotonic", "p_iso")]:
        # operating points
        for op_name, tau in [
            ("tau_0.5", 0.5),
            ("tau_modern5", tau_for_target_fpr(modern_real[col].to_numpy(), 0.05)),
        ]:
            row = {
                "candidate": cand,
                "variant": variant,
                "operating_point": op_name,
                "tau": float(tau),
            }
            for s, df in suite_dfs.items():
                p = df[col].to_numpy()
                key = ("recall_" if s in FAKE_SUITES else "fpr_") + s
                row[key] = (recall_at(p, tau) if s in FAKE_SUITES else fpr_at(p, tau))
            row["fpr_modern_v2"] = fpr_at(modern_real[col].to_numpy(), tau)
            v = row.get("recall_visomaster_enhanced_macro_dev", float("nan"))
            d = row.get("recall_deeplive_enhanced_dev", float("nan"))
            tf = row.get("recall_teams_fake_all_dev", float("nan"))
            mv2 = row.get("fpr_modern_v2", float("nan"))
            tr = row.get("fpr_teams_real_all_dev", float("nan"))
            gate = (
                (v >= T_VISO_RECALL) and (d >= T_DEEPLIVE_RECALL)
                and (tf >= T_TEAMS_FAKE_RECALL)
                and (mv2 <= T_MODERN_FPR) and (tr <= T_DEV_REAL_FPR_CAP)
            )
            row["passes_gate"] = bool(gate)
            single_gate_rows.append(row)
single_gate_df = pd.DataFrame(single_gate_rows)
single_gate_df.to_csv(OUT_DIR / "single_candidate_gate_evaluation.csv", index=False)
print(f"wrote single_candidate_gate_evaluation.csv ({len(single_gate_df)} rows)")
sg_pass = single_gate_df[single_gate_df["passes_gate"]]
print(f"  single-candidate passing gate: {len(sg_pass)}/{len(single_gate_df)}")


# --------------------------------------------------------------------------
# Step 5 — weighted ensemble grid (optional)
# --------------------------------------------------------------------------
print("\n=== Step 5: weighted noisy-OR (P8A + P11_HEAVY) ===")

def build_weighted(cands: List[str], weights: List[float], suite: str, variant: str) -> Optional[pd.DataFrame]:
    """Weighted noisy-OR variant: 1 - prod((1 - p_i)^w_i) — equivalent to scaling logits."""
    base = None
    p_cols = []
    for c in cands:
        df = data.get((suite, c))
        if df is None:
            return None
        df_c = apply_all_to_df(df, c)
        col_map = {"raw": "p_raw", "platt": "p_platt", "isotonic": "p_iso"}
        col = col_map[variant]
        sub = df_c[["frame_path", "label", col]].rename(columns={col: f"p_{c}"})
        if base is None:
            base = sub
        else:
            base = base.merge(sub.drop(columns=["label"]), on="frame_path", how="inner")
        p_cols.append(f"p_{c}")
    arr = clip_p(base[p_cols].to_numpy())
    w = np.asarray(weights)
    log_neg = np.log(1.0 - arr) * w[None, :]
    base["p_weighted"] = 1.0 - np.exp(np.sum(log_neg, axis=1))
    return base

w_rows = []
for variant in ("platt", "isotonic"):
    for w_p8a in (0.3, 0.4, 0.5, 0.6, 0.7):
        w_h = 1.0 - w_p8a
        cands = ["p8a_reference_step5000", "p11_heavy_step1000"]
        weights = [w_p8a, w_h]
        # compute on key suites
        suite_ens = {}
        for s in ALL_SUITES:
            e = build_weighted(cands, weights, s, variant)
            if e is not None:
                suite_ens[s] = e
        if not suite_ens:
            continue
        # cache modern_v2 real
        real_lock = suite_ens.get("teams_real_all_lockbox")
        modern_real = real_lock[real_lock["frame_path"].isin(real_v2_uris)] if real_lock is not None else None
        # operating point: τ giving 5% modern_v2 FPR
        if modern_real is None:
            continue
        tau = tau_for_target_fpr(modern_real["p_weighted"].to_numpy(), 0.05)
        row = {"variant": variant, "w_p8a": w_p8a, "w_p11h": w_h, "tau_modern5": float(tau)}
        for s, e in suite_ens.items():
            p = e["p_weighted"].to_numpy()
            key = ("recall_" if s in FAKE_SUITES else "fpr_") + s
            row[key] = (recall_at(p, tau) if s in FAKE_SUITES else fpr_at(p, tau))
        row["fpr_modern_v2"] = fpr_at(modern_real["p_weighted"].to_numpy(), tau)
        v = row.get("recall_visomaster_enhanced_macro_dev", float("nan"))
        d = row.get("recall_deeplive_enhanced_dev", float("nan"))
        tf = row.get("recall_teams_fake_all_dev", float("nan"))
        mv2 = row.get("fpr_modern_v2", float("nan"))
        tr = row.get("fpr_teams_real_all_dev", float("nan"))
        gate = ((v >= T_VISO_RECALL) and (d >= T_DEEPLIVE_RECALL)
                and (tf >= T_TEAMS_FAKE_RECALL)
                and (mv2 <= T_MODERN_FPR) and (tr <= T_DEV_REAL_FPR_CAP))
        row["passes_gate"] = bool(gate)
        w_rows.append(row)
w_df = pd.DataFrame(w_rows)
w_df.to_csv(OUT_DIR / "weighted_ensemble_grid.csv", index=False)
print(f"wrote weighted_ensemble_grid.csv ({len(w_df)} rows)")
print(f"  weighted passing gate: {int(w_df['passes_gate'].sum() if len(w_df) else 0)}/{len(w_df)}")


# --------------------------------------------------------------------------
# Step 6 — emit summary JSON for narrative report
# --------------------------------------------------------------------------
summary = {
    "n_candidates": len(CANDIDATES),
    "n_suites": len(ALL_SUITES),
    "modern_v2_real_n": len(real_v2_uris),
    "modern_v2_fake_n": len(fake_v2_uris),
    "single_candidate_gate_passing": int(single_gate_df["passes_gate"].sum()),
    "ensemble_gate_passing": int(gate_df["passes_gate"].sum()),
    "weighted_gate_passing": int(w_df["passes_gate"].sum() if len(w_df) else 0),
}
(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nDONE — wrote summary.json: {summary}")
