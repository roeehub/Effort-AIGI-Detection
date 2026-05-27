"""Compute P8A + E2B_TOP_N_STEP3200 ensemble (max-rule, mean-rule, OR-rule).

For each suite, calibrate to the same real_fpr targets used in the contract
(FPR=2/5/10%) on dev real, then read off fake recall on each fake suite.
"""
from __future__ import annotations

import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

DATA_DIR = Path("/tmp/e2b_ensemble_analysis")
OUT_DIR = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/e2b_ensemble_2026-05-04")
OUT_DIR.mkdir(exist_ok=True)

REAL_SUITES = ["teams_real_all_dev", "teams_real_poor_quality_dev", "teams_real_lighting_extreme_dev"]
FAKE_SUITES = ["visomaster_enhanced_macro_dev", "deeplive_enhanced_dev", "teams_fake_all_dev"]

CKPT_A = "p8a_reference_step5000"
CKPT_B = "e2b_top_n_step3200"


def load_frames(suite: str, ckpt: str) -> dict[str, float]:
    """Return {frame_path: score}."""
    f = DATA_DIR / f"{suite}_{ckpt}_frames_report.csv"
    if not f.exists():
        return {}
    out = {}
    with f.open() as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            out[row["frame_path"]] = float(row["frame_prob"])
    return out


def join_scores(suite: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (labels, p8a_scores, e2b_scores) joined on frame_path."""
    sa = load_frames(suite, CKPT_A)
    sb = load_frames(suite, CKPT_B)
    common = sorted(set(sa.keys()) & set(sb.keys()))
    if not common:
        return np.array([]), np.array([]), np.array([])
    # We need labels too — re-read one of the files to get them
    f = DATA_DIR / f"{suite}_{CKPT_A}_frames_report.csv"
    label_map = {}
    with f.open() as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            label_map[row["frame_path"]] = int(row["label"])
    labels = np.array([label_map[p] for p in common])
    p8a = np.array([sa[p] for p in common])
    e2b = np.array([sb[p] for p in common])
    return labels, p8a, e2b


def fpr_at_threshold(scores_real: np.ndarray, threshold: float) -> float:
    return float((scores_real >= threshold).mean())


def recall_at_threshold(scores_fake: np.ndarray, threshold: float) -> float:
    return float((scores_fake >= threshold).mean())


def calibrate_threshold(scores_real: np.ndarray, target_fpr: float) -> float:
    """Find smallest τ such that FPR ≤ target_fpr on real."""
    sorted_scores = np.sort(scores_real)
    # Need (1 - target_fpr) quantile
    idx = max(0, int(np.ceil(len(sorted_scores) * (1 - target_fpr))) - 1)
    if idx >= len(sorted_scores):
        return 1.01
    return float(sorted_scores[idx])


def main():
    # 1. Pool real frames across all 3 dev real suites for joint FPR calibration
    real_p8a, real_e2b = [], []
    for suite in REAL_SUITES:
        labels, p8a, e2b = join_scores(suite)
        if len(labels) == 0:
            print(f"WARN no frames for {suite}")
            continue
        real_mask = labels == 0
        real_p8a.extend(p8a[real_mask].tolist())
        real_e2b.extend(e2b[real_mask].tolist())
        print(f"{suite}: {real_mask.sum()} real frames pooled")

    real_p8a = np.array(real_p8a)
    real_e2b = np.array(real_e2b)
    print(f"\nTotal real frames pooled: {len(real_p8a)}")

    # Define ensemble rules
    def max_rule(p, e): return np.maximum(p, e)
    def mean_rule(p, e): return (p + e) / 2.0
    # rank-fusion (per-suite)
    def make_rank_rule(p_ref, e_ref):
        def rank_rule(p, e):
            from scipy.stats import rankdata
            r_p = rankdata(np.concatenate([p_ref, p]))[len(p_ref):] / (len(p_ref) + len(p))
            r_e = rankdata(np.concatenate([e_ref, e]))[len(e_ref):] / (len(e_ref) + len(e))
            return (r_p + r_e) / 2.0
        return rank_rule

    rules = {
        "P8A_alone": lambda p, e: p,
        "E2B_3200_alone": lambda p, e: e,
        "max(P8A, E2B)": max_rule,
        "mean(P8A, E2B)": mean_rule,
    }

    target_fprs = [0.02, 0.05, 0.10]

    print(f"\n{'rule':25s} {'tgt_fpr':8s} {'τ':10s} {'real_FPR':10s} {'viso':10s} {'deeplive':10s} {'teams_fake':12s}")
    print("=" * 100)

    for rule_name, rule in rules.items():
        # Compute ensemble real scores using the rule
        ens_real = rule(real_p8a, real_e2b)
        for target_fpr in target_fprs:
            tau = calibrate_threshold(ens_real, target_fpr)
            achieved_fpr = fpr_at_threshold(ens_real, tau)
            recalls = {}
            for fake_suite in FAKE_SUITES:
                labels, p8a, e2b = join_scores(fake_suite)
                if len(labels) == 0:
                    recalls[fake_suite] = float("nan")
                    continue
                fake_mask = labels == 1
                ens_fake = rule(p8a[fake_mask], e2b[fake_mask])
                recalls[fake_suite] = recall_at_threshold(ens_fake, tau)
            print(f"{rule_name:25s} {target_fpr:.3f}    {tau:.4f}     {achieved_fpr:.4f}     "
                  f"{recalls['visomaster_enhanced_macro_dev']:.4f}     "
                  f"{recalls['deeplive_enhanced_dev']:.4f}     "
                  f"{recalls['teams_fake_all_dev']:.4f}")
        print()

    # Lockbox readout for the most promising rule (max)
    print("\n=== Lockbox readout (using ensemble τ calibrated on dev real) ===")
    for fpr_target in target_fprs:
        ens_real = np.maximum(real_p8a, real_e2b)
        tau = calibrate_threshold(ens_real, fpr_target)

        # Lockbox real
        labels_lr, p8a_lr, e2b_lr = join_scores("teams_real_all_lockbox")
        ens_lr = np.maximum(p8a_lr, e2b_lr)
        lockbox_real_fpr = float((ens_lr[labels_lr == 0] >= tau).mean()) if (labels_lr == 0).sum() > 0 else float("nan")
        # Lockbox fake
        labels_lf, p8a_lf, e2b_lf = join_scores("teams_fake_all_lockbox")
        ens_lf = np.maximum(p8a_lf, e2b_lf)
        lockbox_fake_recall = float((ens_lf[labels_lf == 1] >= tau).mean()) if (labels_lf == 1).sum() > 0 else float("nan")
        print(f"  max-rule @ dev_FPR={fpr_target:.2f}: lockbox_real_FPR={lockbox_real_fpr:.4f}, lockbox_fake_recall={lockbox_fake_recall:.4f}")


if __name__ == "__main__":
    main()
