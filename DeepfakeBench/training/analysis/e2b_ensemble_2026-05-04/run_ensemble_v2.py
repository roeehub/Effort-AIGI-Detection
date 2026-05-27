"""V2: try z-score normalized combining, calibrated OR-rule, per-ckpt-then-combine."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

DATA_DIR = Path("/tmp/e2b_ensemble_analysis")

REAL_SUITES = ["teams_real_all_dev", "teams_real_poor_quality_dev", "teams_real_lighting_extreme_dev"]
FAKE_SUITES = ["visomaster_enhanced_macro_dev", "deeplive_enhanced_dev", "teams_fake_all_dev"]

CKPT_A = "p8a_reference_step5000"
CKPT_B = "e2b_top_n_step3200"


def load(suite: str, ckpt: str) -> dict[str, tuple[int, float]]:
    f = DATA_DIR / f"{suite}_{ckpt}_frames_report.csv"
    if not f.exists():
        return {}
    out = {}
    with f.open() as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            out[row["frame_path"]] = (int(row["label"]), float(row["frame_prob"]))
    return out


def join(suite: str):
    sa = load(suite, CKPT_A)
    sb = load(suite, CKPT_B)
    common = sorted(set(sa.keys()) & set(sb.keys()))
    if not common:
        return np.array([]), np.array([]), np.array([])
    labels = np.array([sa[p][0] for p in common])
    p8a = np.array([sa[p][1] for p in common])
    e2b = np.array([sb[p][1] for p in common])
    return labels, p8a, e2b


def calib_tau(real_scores, target_fpr):
    sorted_s = np.sort(real_scores)
    idx = max(0, int(np.ceil(len(sorted_s) * (1 - target_fpr))) - 1)
    return float(sorted_s[idx]) if idx < len(sorted_s) else 1.01


def main():
    # Pool real for joint calibration
    real_p8a, real_e2b = [], []
    for s in REAL_SUITES:
        labels, p, e = join(s)
        real_p8a.extend(p[labels == 0].tolist())
        real_e2b.extend(e[labels == 0].tolist())
    real_p8a = np.array(real_p8a)
    real_e2b = np.array(real_e2b)
    print(f"Pooled real frames: {len(real_p8a)}")

    # Strategy 1: per-ckpt calibrated, OR rule (each ckpt → fire/no-fire, OR for combined detect)
    print("\n=== Strategy 1: Per-ckpt calibrated OR (each at half-FPR for joint ~target) ===")
    print(f"{'tgt_joint_FPR':15s} {'achieved_FPR':12s} {'viso':10s} {'deeplive':10s} {'teams_fake':12s}")
    for joint_target in [0.02, 0.05, 0.10]:
        # Calibrate each to ~target/2 (under independence — actually conservative)
        per_target = joint_target / 2.0
        tau_a = calib_tau(real_p8a, per_target)
        tau_b = calib_tau(real_e2b, per_target)
        # Achieved joint FPR on pooled real
        joint_real = ((real_p8a >= tau_a) | (real_e2b >= tau_b)).mean()
        recalls = {}
        for fs in FAKE_SUITES:
            labels, p, e = join(fs)
            fake_mask = labels == 1
            joint_fake = ((p[fake_mask] >= tau_a) | (e[fake_mask] >= tau_b)).mean()
            recalls[fs] = float(joint_fake)
        print(f"{joint_target:.4f}          {joint_real:.4f}        {recalls['visomaster_enhanced_macro_dev']:.4f}     "
              f"{recalls['deeplive_enhanced_dev']:.4f}     {recalls['teams_fake_all_dev']:.4f}")

    # Strategy 2: z-score normalized score → average → calibrate
    print("\n=== Strategy 2: z-score normalize each, then average, then calibrate ===")
    mu_p, sd_p = real_p8a.mean(), real_p8a.std()
    mu_e, sd_e = real_e2b.mean(), real_e2b.std()
    real_z_p = (real_p8a - mu_p) / sd_p
    real_z_e = (real_e2b - mu_e) / sd_e
    real_z_avg = (real_z_p + real_z_e) / 2.0
    print(f"{'tgt_FPR':10s} {'τ':10s} {'real_FPR':10s} {'viso':10s} {'deeplive':10s} {'teams_fake':12s}")
    for tgt in [0.02, 0.05, 0.10]:
        tau_z = calib_tau(real_z_avg, tgt)
        achieved = (real_z_avg >= tau_z).mean()
        recalls = {}
        for fs in FAKE_SUITES:
            labels, p, e = join(fs)
            fake_mask = labels == 1
            zp = (p[fake_mask] - mu_p) / sd_p
            ze = (e[fake_mask] - mu_e) / sd_e
            za = (zp + ze) / 2.0
            recalls[fs] = float((za >= tau_z).mean())
        print(f"{tgt:.4f}    {tau_z:.4f}     {achieved:.4f}     {recalls['visomaster_enhanced_macro_dev']:.4f}     "
              f"{recalls['deeplive_enhanced_dev']:.4f}     {recalls['teams_fake_all_dev']:.4f}")

    # Strategy 3: per-suite-optimized OR (each ckpt at FPR allocation that maximizes recall on the suite)
    # — too many degrees of freedom; skip for now

    # Strategy 4: rank-based on real, then combine
    print("\n=== Strategy 4: percentile rank within real, average, then calibrate on rank ===")
    # Convert each fake score to "what percentile would it land in real distribution"
    sorted_p = np.sort(real_p8a)
    sorted_e = np.sort(real_e2b)
    # Real percentiles are uniform on [0,1) by construction
    real_pct_p = np.searchsorted(sorted_p, real_p8a, side="right") / len(sorted_p)
    real_pct_e = np.searchsorted(sorted_e, real_e2b, side="right") / len(sorted_e)
    real_avg_pct = (real_pct_p + real_pct_e) / 2.0
    print(f"{'tgt_FPR':10s} {'τ_pct':10s} {'real_FPR':10s} {'viso':10s} {'deeplive':10s} {'teams_fake':12s}")
    for tgt in [0.02, 0.05, 0.10]:
        tau_pct = calib_tau(real_avg_pct, tgt)
        achieved = (real_avg_pct >= tau_pct).mean()
        recalls = {}
        for fs in FAKE_SUITES:
            labels, p, e = join(fs)
            fake_mask = labels == 1
            pct_p = np.searchsorted(sorted_p, p[fake_mask], side="right") / len(sorted_p)
            pct_e = np.searchsorted(sorted_e, e[fake_mask], side="right") / len(sorted_e)
            avg_pct = (pct_p + pct_e) / 2.0
            recalls[fs] = float((avg_pct >= tau_pct).mean())
        print(f"{tgt:.4f}    {tau_pct:.4f}     {achieved:.4f}     {recalls['visomaster_enhanced_macro_dev']:.4f}     "
              f"{recalls['deeplive_enhanced_dev']:.4f}     {recalls['teams_fake_all_dev']:.4f}")


if __name__ == "__main__":
    main()
