"""Score-calibration probe for T4_step10500 vs P8A.

Question 1 (dispositive): Does T4_step10500 preserve fake-vs-real ordering on
lockbox? If lockbox AUC ≥ P8A AUC → features separate, only threshold is broken,
calibration can recover. If AUC drops → features fail, calibration won't help.

Question 2 (conditional): Does dev-fit isotonic regression on T4 scores
transfer to lockbox? Specifically: at lockbox real_fpr ≤ 0.018 (P8A's value),
what fake recall does calibrated T4 achieve?

Inputs: video-level reports in _t4_scorecard_local/.
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/cpu_diagnostics_2026-05-10/_t4_scorecard_local")

CKPTS = {
    "P8A": "p8a_reference_step5000",
    "T4_step10500": "t4_lambda1_top_n_step10500",
}

DEV_FAKE_SUITES = ["teams_fake_all_dev", "deeplive_enhanced_dev", "visomaster_enhanced_macro_dev"]
DEV_REAL_SUITES = ["teams_real_all_dev"]
LOCKBOX_REAL = "teams_real_all_lockbox"
LOCKBOX_FAKE = "teams_fake_all_lockbox"


def load(suite: str, ckpt_slug: str):
    p = ROOT / f"{suite}_{ckpt_slug}_videos_report.csv"
    out = []
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            out.append({
                "video_id": row["video_id"],
                "method": row["method"],
                "label": int(row["label"]),
                "score": float(row["avg_video_prob"]),
            })
    return out


def auc(rows):
    """Mann-Whitney U / N_pos / N_neg implementation."""
    pos = sorted([r["score"] for r in rows if r["label"] == 1])
    neg = sorted([r["score"] for r in rows if r["label"] == 0])
    if not pos or not neg:
        return None
    # Brute force is fine for n ≤ a few thousand
    wins = 0
    ties = 0
    for n in neg:
        for p in pos:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def fpr_recall_at_tau(rows, tau, is_fake_suite):
    if is_fake_suite:
        positives = [r for r in rows if r["label"] == 1]
        if not positives:
            return None, 0.0
        tp = sum(1 for r in positives if r["score"] >= tau)
        return None, tp / len(positives)
    else:
        negatives = [r for r in rows if r["label"] == 0]
        if not negatives:
            return 0.0, None
        fp = sum(1 for r in negatives if r["score"] >= tau)
        return fp / len(negatives), None


def find_tau_at_target_fpr(real_rows, target_fpr):
    """Lowest τ such that real FPR ≤ target_fpr."""
    negatives = sorted([r["score"] for r in real_rows if r["label"] == 0])
    n = len(negatives)
    if n == 0:
        return None
    # Largest k such that k/n ≤ target_fpr → tau just above negatives[n-k-1]
    k = int(target_fpr * n)
    if k == 0:
        return negatives[-1] + 1e-9  # no FPs allowed
    return negatives[n - k]


def isotonic_fit(scores, labels):
    """Fit isotonic regression mapping raw_score → P(fake).
    Using the PAV-like simple method (no sklearn dependency).
    Returns a sorted list of (raw_threshold, calibrated_prob).
    """
    pairs = sorted(zip(scores, labels))
    sorted_scores = [s for s, _ in pairs]
    sorted_labels = [l for _, l in pairs]

    # Cumulative blocks
    blocks = [(s, float(l), 1) for s, l in zip(sorted_scores, sorted_labels)]
    while True:
        merged = False
        new_blocks = []
        i = 0
        while i < len(blocks):
            if i + 1 < len(blocks):
                s1, m1, w1 = blocks[i]
                s2, m2, w2 = blocks[i + 1]
                if m1 >= m2:
                    # merge (decreasing — must equalize)
                    new_blocks.append((max(s1, s2), (m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2))
                    i += 2
                    merged = True
                    continue
            new_blocks.append(blocks[i])
            i += 1
        blocks = new_blocks
        if not merged:
            break

    # blocks is now sorted by raw_threshold, with monotonically non-decreasing mean labels
    return [(s, m) for s, m, _ in blocks]


def isotonic_apply(model, score):
    if not model:
        return 0.5
    # Find the rightmost block with threshold ≤ score
    lo, hi = 0, len(model)
    while lo < hi:
        mid = (lo + hi) // 2
        if model[mid][0] <= score:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return model[0][1]
    return model[lo - 1][1]


def main():
    for ckpt_name in CKPTS:
        slug = CKPTS[ckpt_name]
        print("=" * 80)
        print(f"CKPT: {ckpt_name}")
        print("=" * 80)

        # 1. AUCs per suite
        print("\nPer-suite AUC and score-distribution stats:")
        for fake_suite in DEV_FAKE_SUITES + [LOCKBOX_FAKE]:
            real_suite = DEV_REAL_SUITES[0] if fake_suite.endswith("_dev") else LOCKBOX_REAL
            fake_rows = load(fake_suite, slug)
            real_rows = load(real_suite, slug)
            combined = fake_rows + real_rows
            a = auc(combined)
            n_fake = sum(1 for r in fake_rows if r["label"] == 1)
            n_real = sum(1 for r in real_rows if r["label"] == 0)
            print(f"  {fake_suite}  vs  {real_suite}: AUC={a:.4f}  n_fake={n_fake} n_real={n_real}")

        # 2. The dispositive question — lockbox AUC
        fake_rows_lb = load(LOCKBOX_FAKE, slug)
        real_rows_lb = load(LOCKBOX_REAL, slug)

    print()
    print("=" * 80)
    print("KEY: LOCKBOX AUC COMPARISON")
    print("=" * 80)
    for ckpt_name, slug in CKPTS.items():
        fake_rows_lb = load(LOCKBOX_FAKE, slug)
        real_rows_lb = load(LOCKBOX_REAL, slug)
        a = auc(fake_rows_lb + real_rows_lb)
        print(f"  {ckpt_name}: lockbox AUC = {a:.4f}")

    # 3. Calibration: fit isotonic on dev (all dev fakes + dev reals), apply to lockbox
    print()
    print("=" * 80)
    print("CALIBRATION PROBE: isotonic fit on DEV, applied to LOCKBOX")
    print("=" * 80)
    for ckpt_name, slug in CKPTS.items():
        # Pool dev fakes + dev reals
        dev_fakes = []
        for s in DEV_FAKE_SUITES:
            dev_fakes.extend(load(s, slug))
        dev_reals = load(DEV_REAL_SUITES[0], slug)
        train_rows = dev_fakes + dev_reals
        scores = [r["score"] for r in train_rows]
        labels = [r["label"] for r in train_rows]
        print(f"\n[{ckpt_name}] training calibration on n={len(train_rows)} dev videos "
              f"({sum(labels)} fake, {len(labels) - sum(labels)} real)")
        model = isotonic_fit(scores, labels)
        print(f"  isotonic model: {len(model)} blocks")

        # Apply to lockbox
        fake_rows_lb = load(LOCKBOX_FAKE, slug)
        real_rows_lb = load(LOCKBOX_REAL, slug)
        for r in fake_rows_lb + real_rows_lb:
            r["calibrated"] = isotonic_apply(model, r["score"])

        # AUC after calibration (should be same — isotonic is monotonic, preserves AUC)
        # Compute fake_recall at target real_fpr = 0.018 (P8A's value)
        target_fpr = 0.018
        target_fpr_p8a = target_fpr if ckpt_name == "P8A" else 0.018
        for tau_label, target in [("0.018 (P8A baseline)", 0.018), ("0.050", 0.05), ("0.020", 0.02)]:
            # On calibrated scores
            negs_cal = sorted([r["calibrated"] for r in real_rows_lb if r["label"] == 0])
            pos_cal = [r["calibrated"] for r in fake_rows_lb if r["label"] == 1]
            n_neg = len(negs_cal)
            k = int(target * n_neg)
            if k == 0:
                tau_cal = negs_cal[-1] + 1e-9
            else:
                tau_cal = negs_cal[n_neg - k]
            fake_recall_cal = sum(1 for p in pos_cal if p >= tau_cal) / len(pos_cal) if pos_cal else 0.0

            # Raw score for comparison
            negs_raw = sorted([r["score"] for r in real_rows_lb if r["label"] == 0])
            pos_raw = [r["score"] for r in fake_rows_lb if r["label"] == 1]
            k_raw = int(target * len(negs_raw))
            if k_raw == 0:
                tau_raw = negs_raw[-1] + 1e-9
            else:
                tau_raw = negs_raw[len(negs_raw) - k_raw]
            fake_recall_raw = sum(1 for p in pos_raw if p >= tau_raw) / len(pos_raw) if pos_raw else 0.0

            print(f"  at lockbox_real_fpr ≤ {tau_label}: raw_fake_recall={fake_recall_raw:.4f}  "
                  f"calibrated_fake_recall={fake_recall_cal:.4f}  (τ_raw={tau_raw:.4f}, τ_cal={tau_cal:.4f})")


if __name__ == "__main__":
    main()
