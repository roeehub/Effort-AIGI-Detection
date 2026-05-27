"""3-way ensemble: P8A + E2B_TOP_N_STEP3200 + E3_TOP_N_STEP6600.

Tests max-rule, mean-rule, percentile-rank-avg, and per-ckpt OR.
Also tests pairwise: P8A+E3_6600 vs P8A+E2B_3200.

Both E2b and E3 trim4 scorecards used the same suite manifest, so frame_path
should match across runs for the same video frames.
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

E2B_DIR = Path("/tmp/e2b_ensemble_analysis")  # has P8A + E2B_3200
E3_DIR = Path("/tmp/e3_ensemble_analysis")    # has P8A + E3_4800/6600/7200

REAL_SUITES = ["teams_real_all_dev", "teams_real_poor_quality_dev", "teams_real_lighting_extreme_dev"]
FAKE_SUITES = ["visomaster_enhanced_macro_dev", "deeplive_enhanced_dev", "teams_fake_all_dev"]


def load(d: Path, suite: str, ckpt: str) -> dict[str, tuple[int, float]]:
    f = d / f"{suite}_{ckpt}_frames_report.csv"
    if not f.exists():
        return {}
    out = {}
    with f.open() as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            out[row["frame_path"]] = (int(row["label"]), float(row["frame_prob"]))
    return out


def join_3way(suite: str):
    """Returns (labels, p8a, e2b_3200, e3_6600) on common frame paths."""
    p8a_e2b = load(E2B_DIR, suite, "p8a_reference_step5000")
    e2b = load(E2B_DIR, suite, "e2b_top_n_step3200")
    p8a_e3 = load(E3_DIR, suite, "p8a_reference_step5000")
    e3 = load(E3_DIR, suite, "e3_top_n_step6600")

    common = sorted(set(p8a_e2b.keys()) & set(e2b.keys()) & set(p8a_e3.keys()) & set(e3.keys()))
    if not common:
        return np.array([]), np.array([]), np.array([]), np.array([])

    labels = np.array([p8a_e2b[p][0] for p in common])
    p8a = np.array([p8a_e2b[p][1] for p in common])
    e2b_s = np.array([e2b[p][1] for p in common])
    e3_s = np.array([e3[p][1] for p in common])
    return labels, p8a, e2b_s, e3_s


def calib_tau(real_scores, target_fpr):
    sorted_s = np.sort(real_scores)
    idx = max(0, int(np.ceil(len(sorted_s) * (1 - target_fpr))) - 1)
    return float(sorted_s[idx]) if idx < len(sorted_s) else 1.01


def main():
    # Pool real for joint calibration
    pools = {"p8a": [], "e2b": [], "e3": []}
    for s in REAL_SUITES:
        labels, p, e, ee = join_3way(s)
        if len(labels) == 0:
            print(f"WARN no joined frames for {s}")
            continue
        pools["p8a"].extend(p[labels == 0].tolist())
        pools["e2b"].extend(e[labels == 0].tolist())
        pools["e3"].extend(ee[labels == 0].tolist())
    p8a_real = np.array(pools["p8a"])
    e2b_real = np.array(pools["e2b"])
    e3_real = np.array(pools["e3"])
    print(f"Pooled real frames: {len(p8a_real)}")

    # Strategies
    rules = {
        "P8A_alone": lambda p, e2, e3: p,
        "E2B_3200_alone": lambda p, e2, e3: e2,
        "E3_6600_alone": lambda p, e2, e3: e3,
        "max(P8A, E2B)": lambda p, e2, e3: np.maximum(p, e2),
        "max(P8A, E3)": lambda p, e2, e3: np.maximum(p, e3),
        "max(E2B, E3)": lambda p, e2, e3: np.maximum(e2, e3),
        "max(P8A, E2B, E3)": lambda p, e2, e3: np.maximum.reduce([p, e2, e3]),
        "mean(all 3)": lambda p, e2, e3: (p + e2 + e3) / 3.0,
    }

    print(f"\n{'rule':25s} {'tgt_fpr':8s} {'real_FPR':10s} {'viso':10s} {'deeplive':10s} {'teams_fake':12s}")
    print("=" * 100)

    target_fprs = [0.05, 0.10]
    for rule_name, rule in rules.items():
        ens_real = rule(p8a_real, e2b_real, e3_real)
        for target_fpr in target_fprs:
            tau = calib_tau(ens_real, target_fpr)
            achieved = (ens_real >= tau).mean()
            recalls = {}
            for fs in FAKE_SUITES:
                labels, p, e2, e3 = join_3way(fs)
                fake_mask = labels == 1
                ens_fake = rule(p[fake_mask], e2[fake_mask], e3[fake_mask])
                recalls[fs] = float((ens_fake >= tau).mean())
            print(f"{rule_name:25s} {target_fpr:.3f}    {achieved:.4f}     "
                  f"{recalls['visomaster_enhanced_macro_dev']:.4f}     "
                  f"{recalls['deeplive_enhanced_dev']:.4f}     "
                  f"{recalls['teams_fake_all_dev']:.4f}")
        print()

    # Per-ckpt-calibrated OR (each at FPR/3 for joint ~target)
    print("=== Per-ckpt OR (each at FPR/3 for joint ~target) ===")
    print(f"{'tgt_joint_FPR':15s} {'ach_FPR':10s} {'viso':10s} {'deeplive':10s} {'teams_fake':12s}")
    for joint_target in [0.05, 0.10]:
        per_target = joint_target / 3.0
        tau_a = calib_tau(p8a_real, per_target)
        tau_b = calib_tau(e2b_real, per_target)
        tau_c = calib_tau(e3_real, per_target)
        joint_real = ((p8a_real >= tau_a) | (e2b_real >= tau_b) | (e3_real >= tau_c)).mean()
        recalls = {}
        for fs in FAKE_SUITES:
            labels, p, e2, e3 = join_3way(fs)
            fake_mask = labels == 1
            joint_fake = ((p[fake_mask] >= tau_a) | (e2[fake_mask] >= tau_b) | (e3[fake_mask] >= tau_c)).mean()
            recalls[fs] = float(joint_fake)
        print(f"{joint_target:.4f}          {joint_real:.4f}     {recalls['visomaster_enhanced_macro_dev']:.4f}     "
              f"{recalls['deeplive_enhanced_dev']:.4f}     {recalls['teams_fake_all_dev']:.4f}")


if __name__ == "__main__":
    main()
