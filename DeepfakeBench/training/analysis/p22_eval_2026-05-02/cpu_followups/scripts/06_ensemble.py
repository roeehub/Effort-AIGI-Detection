"""Job F — cross-packet ensemble study.

Test whether a P8A + P18T + P22 step1k/step8k ensemble beats individuals at
joint dev+lockbox FPR=2%. Three ensemble rules:

1. Average:    score = mean(scores)
2. Max:        score = max(scores) — conservative on real (any model that says fake → fake)
3. Min:        score = min(scores) — conservative on fake (must convince all models)
4. Voting:     score = mean(score >= τ_per_model)

For each rule, find joint-compliant τ and report fake recall.
"""
from __future__ import annotations

import sys; sys.path.insert(0, "analysis/p22_eval_2026-05-02/cpu_followups/scripts")

import numpy as np
import pandas as pd

from _common import OUT, FAKE_SUITES, REAL_SUITES, load_per_frame_scores

CKPTS = ["P8A_REFERENCE_STEP5000", "P18T_GRL_TREATMENT_STEP4000",
         "P22_AUG_STEP1000", "P22_AUG_STEP8000"]


def load_aligned_scores(suite, ckpts):
    """Load per-frame scores for each ckpt × suite, align by frame_path,
    return DataFrame with columns frame_path + score_<ckpt>."""
    parts = []
    for ckpt in ckpts:
        df = load_per_frame_scores(ckpt, suite)
        if df is None:
            return None
        df = df[["frame_path", "frame_prob"]].rename(columns={"frame_prob": f"score_{ckpt}"})
        parts.append(df)
    merged = parts[0]
    for p in parts[1:]:
        merged = merged.merge(p, on="frame_path", how="inner")
    return merged


def fpr_recall_at_tau(real_arr, fake_arrs, tau):
    fpr = float((real_arr >= tau).mean()) if len(real_arr) else float("nan")
    recalls = {s: float((arr >= tau).mean()) if len(arr) else float("nan")
               for s, arr in fake_arrs.items()}
    return fpr, recalls


def find_joint_compliant(real_dev, real_lock, fake_arrs, floor=0.02, n_grid=999):
    """Find smallest τ s.t. dev_FPR ≤ floor AND lockbox_FPR ≤ floor."""
    grid = np.linspace(0.001, 0.999, n_grid)
    rows = []
    for tau in grid:
        dev_fpr = float((real_dev >= tau).mean())
        lock_fpr = float((real_lock >= tau).mean())
        row = {"tau": tau, "dev_fpr": dev_fpr, "lockbox_fpr": lock_fpr}
        for s, arr in fake_arrs.items():
            row[f"recall_{s}"] = float((arr >= tau).mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    valid = df[(df.dev_fpr <= floor + 1e-9) & (df.lockbox_fpr <= floor + 1e-9)]
    if len(valid) == 0:
        return None
    return valid.loc[valid.tau.idxmin()]


def run_ensembles():
    rows = []

    # Single-model baselines
    for ckpt in CKPTS:
        rd = load_per_frame_scores(ckpt, "teams_real_all_dev")
        rl = load_per_frame_scores(ckpt, "teams_real_all_lockbox")
        if rd is None or rl is None: continue
        rd_arr = rd["frame_prob"].to_numpy()
        rl_arr = rl["frame_prob"].to_numpy()
        fakes = {}
        for s in FAKE_SUITES:
            f = load_per_frame_scores(ckpt, s)
            if f is not None:
                fakes[s] = f["frame_prob"].to_numpy()
        chosen = find_joint_compliant(rd_arr, rl_arr, fakes, floor=0.02)
        if chosen is None:
            rows.append({"label": ckpt, "ensemble": "single", "joint_compliant": False})
            continue
        row = {"label": ckpt, "ensemble": "single", "joint_compliant": True,
               "tau": chosen.tau, "dev_fpr": chosen.dev_fpr, "lockbox_fpr": chosen.lockbox_fpr}
        for s in FAKE_SUITES:
            col = f"recall_{s}"
            if col in chosen.index:
                row[col] = chosen[col]
        rows.append(row)

    # Ensemble combos: pairs and triple
    combos = [
        ("P8A+P22step1k", ["P8A_REFERENCE_STEP5000", "P22_AUG_STEP1000"]),
        ("P8A+P22step8k", ["P8A_REFERENCE_STEP5000", "P22_AUG_STEP8000"]),
        ("P8A+P18T", ["P8A_REFERENCE_STEP5000", "P18T_GRL_TREATMENT_STEP4000"]),
        ("P8A+P18T+P22step1k", ["P8A_REFERENCE_STEP5000", "P18T_GRL_TREATMENT_STEP4000",
                                  "P22_AUG_STEP1000"]),
        ("P8A+P22step1k+P22step8k", ["P8A_REFERENCE_STEP5000", "P22_AUG_STEP1000",
                                      "P22_AUG_STEP8000"]),
        ("ALL_FOUR", CKPTS),
    ]

    for combo_name, combo_ckpts in combos:
        # Need aligned scores per suite
        real_dev = load_aligned_scores("teams_real_all_dev", combo_ckpts)
        real_lock = load_aligned_scores("teams_real_all_lockbox", combo_ckpts)
        if real_dev is None or real_lock is None: continue

        cols = [f"score_{c}" for c in combo_ckpts]

        for rule_name, fn in [
            ("mean", lambda df: df[cols].mean(axis=1).to_numpy()),
            ("max", lambda df: df[cols].max(axis=1).to_numpy()),
            ("min", lambda df: df[cols].min(axis=1).to_numpy()),
        ]:
            rd_arr = fn(real_dev)
            rl_arr = fn(real_lock)
            fakes = {}
            for s in FAKE_SUITES:
                f = load_aligned_scores(s, combo_ckpts)
                if f is not None:
                    fakes[s] = fn(f)
            chosen = find_joint_compliant(rd_arr, rl_arr, fakes, floor=0.02)
            label = f"{combo_name}::{rule_name}"
            if chosen is None:
                rows.append({"label": label, "ensemble": rule_name, "joint_compliant": False})
                continue
            row = {"label": label, "ensemble": rule_name, "joint_compliant": True,
                   "tau": chosen.tau, "dev_fpr": chosen.dev_fpr, "lockbox_fpr": chosen.lockbox_fpr}
            for s in FAKE_SUITES:
                col = f"recall_{s}"
                if col in chosen.index:
                    row[col] = chosen[col]
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = run_ensembles()
    df.to_csv(OUT / "06_ensemble.csv", index=False)

    # Filter to joint-compliant + key columns
    print("=" * 110)
    print("Cross-packet ensembles at joint dev+lockbox FPR ≤ 2%")
    print("=" * 110)
    cols = ["label", "ensemble", "tau", "dev_fpr", "lockbox_fpr",
            "recall_teams_fake_all_dev", "recall_visomaster_enhanced_macro_dev",
            "recall_deeplive_enhanced_dev", "recall_teams_fake_all_lockbox"]
    cols = [c for c in cols if c in df.columns]
    show = df[df.joint_compliant == True][cols] if "joint_compliant" in df.columns else df[cols]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))


if __name__ == "__main__":
    main()
