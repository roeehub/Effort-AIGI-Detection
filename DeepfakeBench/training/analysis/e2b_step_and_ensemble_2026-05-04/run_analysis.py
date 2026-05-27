"""CPU-only diagnostic to inform tonight's deployment decision.

Job 1: Score-distribution profile across E2B steps for each suite.
Job 2: P8A_step5000 + E2B_step3200 ensemble (per-frame), 3 rules.

Inputs from analysis/cpu_followups_2026-05-04/raw_reports/:
- *_e2b_(top_n_)?step<STEP>_group_metrics.csv -> mean_prob/p50/p90 (per-video)
- *_e2b_top_n_step3200_frames_report.csv      -> per-frame scores
- *_p8a_reference_step5000_frames_report.csv  -> per-frame scores

Outputs in this directory:
- score_distribution_profile.csv
- ensemble_results.csv
- (FINDINGS.md is hand-written based on these)
"""
from __future__ import annotations

from pathlib import Path
import csv
import sys

import numpy as np
import pandas as pd

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
RAW = ROOT / "analysis/cpu_followups_2026-05-04/raw_reports"
OUT = ROOT / "analysis/e2b_step_and_ensemble_2026-05-04"
OUT.mkdir(exist_ok=True)

STEPS = [3000, 3200, 3400, 4400, 6400, 6800, 7400]

REAL_SUITES = ["teams_real_all_dev", "teams_real_all_lockbox"]
FAKE_SUITES = ["deeplive_enhanced_dev", "teams_fake_all_dev", "teams_fake_all_lockbox"]
ALL_SUITES = REAL_SUITES + FAKE_SUITES


def e2b_group_path(suite: str, step: int) -> Path:
    """Return the existing group_metrics path for a given E2B step.

    Step 3000 uses the older pattern '_e2b_step3000_'; 6000 same.
    Other steps use '_e2b_top_n_step<STEP>_'. We probe both.
    """
    cand1 = RAW / f"{suite}_e2b_top_n_step{step}_group_metrics.csv"
    cand2 = RAW / f"{suite}_e2b_step{step}_group_metrics.csv"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    return cand1  # nonexistent; caller will skip


def job1_distribution_profile() -> pd.DataFrame:
    """For each suite x step, report aggregate per-video score stats.

    The group_metrics CSV gives mean_prob, p50_prob, p90_prob (over n_videos).
    No p10/p25/p75 — be explicit about that. Spread proxy: p90 - mean.
    For multi-group fake suites we keep all groups (each is a row).
    """
    rows = []
    for suite in ALL_SUITES:
        for step in STEPS:
            p = e2b_group_path(suite, step)
            if not p.exists():
                rows.append({
                    "suite": suite, "step": step, "group_key": None,
                    "n_videos": None, "mean_prob": None,
                    "p50_prob": None, "p90_prob": None,
                    "spread_p90_minus_mean": None, "fnr_or_fpr": None,
                    "note": "missing",
                })
                continue
            df = pd.read_csv(p)
            for _, r in df.iterrows():
                gk = r["group_key"]
                # FPR for real groups, FNR for fake groups
                if "real" in str(r["family_key"]):
                    err_col, err_name = "fpr", "fpr"
                else:
                    err_col, err_name = "fnr", "fnr"
                rows.append({
                    "suite": suite, "step": step,
                    "group_key": gk,
                    "n_videos": int(r["n_videos"]),
                    "mean_prob": float(r["mean_prob"]),
                    "p50_prob": float(r["p50_prob"]),
                    "p90_prob": float(r["p90_prob"]),
                    "spread_p90_minus_mean": float(r["p90_prob"]) - float(r["mean_prob"]),
                    "fnr_or_fpr": f"{err_name}={float(r[err_col]):.4f}",
                    "note": "",
                })
    return pd.DataFrame(rows)


# ---------- Job 2 ----------

def load_frames(suite: str, ckpt_tag: str) -> pd.DataFrame:
    p = RAW / f"{suite}_{ckpt_tag}_frames_report.csv"
    if not p.exists():
        print(f"MISSING: {p.name}", file=sys.stderr)
        return pd.DataFrame()
    df = pd.read_csv(p)
    return df[["frame_path", "label", "frame_prob"]].copy()


def join_e2b_p8a(suite: str) -> pd.DataFrame:
    e2b = load_frames(suite, "e2b_top_n_step3200")
    p8a = load_frames(suite, "p8a_reference_step5000")
    if e2b.empty or p8a.empty:
        return pd.DataFrame()
    e2b = e2b.rename(columns={"frame_prob": "s_e2b"})
    p8a = p8a.rename(columns={"frame_prob": "s_p8a", "label": "label_p8a"})
    j = e2b.merge(p8a[["frame_path", "s_p8a", "label_p8a"]],
                  on="frame_path", how="inner")
    if (j["label"] != j["label_p8a"]).any():
        n_bad = int((j["label"] != j["label_p8a"]).sum())
        print(f"WARNING {suite}: {n_bad} frames have mismatched labels", file=sys.stderr)
    j = j.drop(columns=["label_p8a"])
    return j


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_ensembles(df: pd.DataFrame) -> dict[str, np.ndarray]:
    s_e = df["s_e2b"].to_numpy()
    s_p = df["s_p8a"].to_numpy()
    return {
        "e2b_only": s_e,
        "p8a_only": s_p,
        "min": np.minimum(s_e, s_p),
        "mean": 0.5 * (s_e + s_p),
        "logpool": sigmoid(0.5 * (logit(s_e) + logit(s_p))),
    }


def calib_tau(real_scores: np.ndarray, target_fpr: float) -> float:
    """Conservative tau at target FPR — smallest tau s.t. FPR <= target."""
    s = np.sort(real_scores)
    n = len(s)
    # Index of (1 - target_fpr) quantile, conservative (FPR <= target)
    idx = min(n - 1, max(0, int(np.ceil(n * (1.0 - target_fpr))) - 1))
    return float(s[idx])


def dist_stats(x: np.ndarray) -> dict[str, float]:
    if len(x) == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "p25": float("nan"), "p50": float("nan"), "p75": float("nan")}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p25": float(np.percentile(x, 25)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
    }


def job2_ensemble() -> pd.DataFrame:
    """Per-frame ensembles. Calibrate tau on teams_real_all_dev, evaluate
    recall on teams_real_all_lockbox (FPR) and on each fake suite (TPR)."""
    suite_data: dict[str, pd.DataFrame] = {}
    for s in [*REAL_SUITES, *FAKE_SUITES]:
        suite_data[s] = join_e2b_p8a(s)
        if not suite_data[s].empty:
            n = len(suite_data[s])
            n_real = int((suite_data[s]["label"] == 0).sum())
            n_fake = int((suite_data[s]["label"] == 1).sum())
            print(f"  {s}: joined {n} frames (real={n_real}, fake={n_fake})")

    if suite_data["teams_real_all_dev"].empty:
        print("FATAL: cannot calibrate without teams_real_all_dev frames", file=sys.stderr)
        return pd.DataFrame()

    cal = suite_data["teams_real_all_dev"]
    cal_real = cal[cal["label"] == 0].copy()
    cal_ens = build_ensembles(cal_real)

    rows = []
    for rule, _ in cal_ens.items():
        # Distribution stats per suite under this rule
        for s in [*REAL_SUITES, *FAKE_SUITES]:
            df = suite_data[s]
            if df.empty:
                continue
            ens = build_ensembles(df)[rule]
            # Real subset (for FPR-relevant suites) and fake subset
            real_mask = (df["label"] == 0).to_numpy()
            fake_mask = (df["label"] == 1).to_numpy()

            for subset_name, mask in [("real", real_mask), ("fake", fake_mask)]:
                if mask.sum() == 0:
                    continue
                stats = dist_stats(ens[mask])
                rows.append({
                    "rule": rule, "suite": s, "subset": subset_name,
                    "n": stats["n"], "mean": stats["mean"], "std": stats["std"],
                    "p25": stats["p25"], "p50": stats["p50"], "p75": stats["p75"],
                    "metric_kind": "dist",
                })

        # Calibrate at target FPR levels using teams_real_all_dev real subset
        for target_fpr in [0.05, 0.10, 0.20]:
            tau = calib_tau(cal_ens[rule], target_fpr)
            cal_ach = float((cal_ens[rule] >= tau).mean())
            # FPR on lockbox real
            lock_df = suite_data["teams_real_all_lockbox"]
            lock_real = lock_df[lock_df["label"] == 0]
            if not lock_real.empty:
                lock_ens = build_ensembles(lock_real)[rule]
                lock_fpr = float((lock_ens >= tau).mean())
            else:
                lock_fpr = float("nan")

            row_base = {
                "rule": rule, "subset": "operating_point",
                "target_fpr_dev": target_fpr, "tau": tau,
                "achieved_fpr_dev": cal_ach,
                "lockbox_fpr_real": lock_fpr,
                "metric_kind": "operating_point",
            }
            # Recall per fake suite at this tau
            for fs in FAKE_SUITES:
                df = suite_data[fs]
                if df.empty:
                    continue
                ens = build_ensembles(df)[rule]
                fmask = (df["label"] == 1).to_numpy()
                if fmask.sum() == 0:
                    rec = float("nan")
                else:
                    rec = float((ens[fmask] >= tau).mean())
                rows.append({**row_base, "suite": fs, "recall_fake": rec})
    return pd.DataFrame(rows)


def main():
    print("=== Job 1: score-distribution profile across E2B steps ===")
    j1 = job1_distribution_profile()
    j1_path = OUT / "score_distribution_profile.csv"
    j1.to_csv(j1_path, index=False)
    print(f"wrote {j1_path}  ({len(j1)} rows)")

    print("\n=== Job 2: P8A + E2B_step3200 ensemble (per-frame) ===")
    j2 = job2_ensemble()
    j2_path = OUT / "ensemble_results.csv"
    j2.to_csv(j2_path, index=False)
    print(f"wrote {j2_path}  ({len(j2)} rows)")

    # Helpful console summaries
    print("\n--- Job 1 quick view: per-suite mean_prob trajectory across steps (UNKNOWN groups only) ---")
    pivot = j1[j1["group_key"].isin(["unknown_real", "unknown_fake", "deeplive_enhanced_fake"])].pivot_table(
        index=["suite", "group_key"], columns="step", values="mean_prob", aggfunc="first"
    )
    print(pivot.round(4).to_string())

    print("\n--- Job 1 quick view: spread (p90 - mean) trajectory ---")
    pivot2 = j1[j1["group_key"].isin(["unknown_real", "unknown_fake", "deeplive_enhanced_fake"])].pivot_table(
        index=["suite", "group_key"], columns="step", values="spread_p90_minus_mean", aggfunc="first"
    )
    print(pivot2.round(4).to_string())

    print("\n--- Job 2 quick view: operating_point rows ---")
    op = j2[j2["metric_kind"] == "operating_point"].copy()
    if not op.empty:
        op_view = op.pivot_table(
            index=["rule", "target_fpr_dev"], columns="suite",
            values="recall_fake", aggfunc="first"
        )
        print(op_view.round(4).to_string())
        print("\n  lockbox FPR per (rule, target_fpr_dev):")
        lk = op[["rule", "target_fpr_dev", "tau", "achieved_fpr_dev",
                 "lockbox_fpr_real"]].drop_duplicates(["rule", "target_fpr_dev"])
        print(lk.round(4).to_string(index=False))

    print("\n--- Job 2 quick view: real-suite mean prob_fake per ensemble rule ---")
    dist = j2[(j2["metric_kind"] == "dist") & (j2["subset"] == "real")]
    if not dist.empty:
        dview = dist.pivot_table(index="rule", columns="suite", values="mean", aggfunc="first")
        print(dview.round(4).to_string())


if __name__ == "__main__":
    main()
