"""Sweep video-level τ for T4_L1_step10500 vs P8A on lockbox + dor_dev suites.

Question: Can T4_step10500 beat P8A on lockbox_fake_recall at any τ
that also satisfies the dev gates (real_fpr ≤ 0.07, stress ≤ 0.10)?
"""
from __future__ import annotations
import csv
from pathlib import Path
import json

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/cpu_diagnostics_2026-05-10/_t4_scorecard_local")

CKPTS = {
    "P8A": "p8a_reference_step5000",
    "T4_step10500": "t4_lambda1_top_n_step10500",
    "T4_lambda2_step1500": "t4_lambda2_periodic_step1500",
}

SUITES = ["teams_real_all_lockbox", "teams_fake_all_lockbox", "teams_real_dor_dev"]


def load_videos(suite: str, ckpt_slug: str):
    p = ROOT / f"{suite}_{ckpt_slug}_videos_report.csv"
    rows = []
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "video_id": row["video_id"],
                "method": row["method"],
                "label": int(row["label"]),
                "avg_video_prob": float(row["avg_video_prob"]),
            })
    return rows


def metric_at_tau(rows, tau, is_fake: bool):
    """For fake suite: recall = TP/N_fake at score >= tau.
    For real suite: FPR = FP/N_real at score >= tau."""
    if not rows:
        return 0.0
    if is_fake:
        positives = [r for r in rows if r["label"] == 1]
        if not positives:
            return 0.0
        tp = sum(1 for r in positives if r["avg_video_prob"] >= tau)
        return tp / len(positives)
    else:
        negatives = [r for r in rows if r["label"] == 0]
        if not negatives:
            return 0.0
        fp = sum(1 for r in negatives if r["avg_video_prob"] >= tau)
        return fp / len(negatives)


def main():
    data = {}
    for ckpt_name, slug in CKPTS.items():
        data[ckpt_name] = {}
        for suite in SUITES:
            rows = load_videos(suite, slug)
            data[ckpt_name][suite] = rows

    # 1. Sanity check at selected τ matches scorecard
    selected_taus = {
        "P8A": 0.915605,
        "T4_step10500": 0.843967,
        "T4_lambda2_step1500": 0.784609,
    }
    print("=" * 80)
    print("SANITY CHECK at selected τ (should match scorecard)")
    print("=" * 80)
    print(f"{'ckpt':<25}{'τ':<10}{'real_lockbox_fpr':<22}{'fake_lockbox_recall':<24}{'dor_dev_fpr':<14}")
    for ckpt_name, tau in selected_taus.items():
        real_fpr = metric_at_tau(data[ckpt_name]["teams_real_all_lockbox"], tau, is_fake=False)
        fake_rec = metric_at_tau(data[ckpt_name]["teams_fake_all_lockbox"], tau, is_fake=True)
        dor_fpr = metric_at_tau(data[ckpt_name]["teams_real_dor_dev"], tau, is_fake=False)
        print(f"{ckpt_name:<25}{tau:<10.4f}{real_fpr:<22.4f}{fake_rec:<24.4f}{dor_fpr:<14.4f}")

    # 2. τ-sweep: vary τ for T4_step10500 to find a τ that beats P8A on lockbox_fake_recall
    print()
    print("=" * 80)
    print("τ-SWEEP T4_step10500 — does any τ beat P8A_lockbox_fake_recall=0.3874")
    print("at acceptable lockbox_real_fpr (target ≤ 0.05 ideally)?")
    print("=" * 80)
    print(f"{'τ':<10}{'real_lockbox_fpr':<22}{'fake_lockbox_recall':<24}{'dor_dev_fpr':<14}{'note':<30}")
    P8A_TARGETS = {"real_lockbox_fpr": 0.0184, "fake_lockbox_recall": 0.3874, "dor_dev_fpr": 0.080}
    candidates = []
    for tau_x100 in range(50, 100, 1):  # 0.50 to 0.99
        tau = tau_x100 / 100
        real_fpr = metric_at_tau(data["T4_step10500"]["teams_real_all_lockbox"], tau, is_fake=False)
        fake_rec = metric_at_tau(data["T4_step10500"]["teams_fake_all_lockbox"], tau, is_fake=True)
        dor_fpr = metric_at_tau(data["T4_step10500"]["teams_real_dor_dev"], tau, is_fake=False)
        note = ""
        if fake_rec > P8A_TARGETS["fake_lockbox_recall"] and real_fpr <= P8A_TARGETS["real_lockbox_fpr"]:
            note = "BEATS P8A on both"
            candidates.append((tau, real_fpr, fake_rec, dor_fpr))
        elif fake_rec > P8A_TARGETS["fake_lockbox_recall"]:
            note = f"better recall, real_fpr {real_fpr:.3f}"
        if tau_x100 % 5 == 0 or note:
            print(f"{tau:<10.2f}{real_fpr:<22.4f}{fake_rec:<24.4f}{dor_fpr:<14.4f}{note:<30}")

    if not candidates:
        print()
        print("⚠ NO τ for T4_step10500 simultaneously beats P8A on lockbox_fake_recall AND maintains real_fpr ≤ P8A.")
    else:
        print()
        print(f"✓ Found {len(candidates)} τ candidates for T4_step10500 that beat P8A on both axes.")

    # 3. T4_λ2_step1500 (the lockbox-recall outlier)
    print()
    print("=" * 80)
    print("τ-SWEEP T4_λ2_step1500 — the 77% lockbox-recall outlier")
    print("=" * 80)
    print(f"{'τ':<10}{'real_lockbox_fpr':<22}{'fake_lockbox_recall':<24}{'dor_dev_fpr':<14}")
    for tau_x100 in [50, 60, 70, 75, 78, 80, 85, 90, 95, 99]:
        tau = tau_x100 / 100
        real_fpr = metric_at_tau(data["T4_lambda2_step1500"]["teams_real_all_lockbox"], tau, is_fake=False)
        fake_rec = metric_at_tau(data["T4_lambda2_step1500"]["teams_fake_all_lockbox"], tau, is_fake=True)
        dor_fpr = metric_at_tau(data["T4_lambda2_step1500"]["teams_real_dor_dev"], tau, is_fake=False)
        print(f"{tau:<10.2f}{real_fpr:<22.4f}{fake_rec:<24.4f}{dor_fpr:<14.4f}")

    # 4. Score distribution stats
    print()
    print("=" * 80)
    print("SCORE DISTRIBUTION STATS — lockbox cells")
    print("=" * 80)
    for suite in ["teams_real_all_lockbox", "teams_fake_all_lockbox", "teams_real_dor_dev"]:
        print(f"\n[{suite}]")
        print(f"{'ckpt':<25}{'n':<6}{'p10':<8}{'p25':<8}{'p50':<8}{'p75':<8}{'p90':<8}{'max':<8}")
        for ckpt_name in CKPTS:
            rows = data[ckpt_name][suite]
            # focus on the relevant label
            if suite.startswith("teams_real"):
                rows = [r for r in rows if r["label"] == 0]
            else:
                rows = [r for r in rows if r["label"] == 1]
            scores = sorted(r["avg_video_prob"] for r in rows)
            if not scores:
                continue
            def pct(p):
                idx = max(0, min(len(scores) - 1, int(round(p / 100 * (len(scores) - 1)))))
                return scores[idx]
            print(f"{ckpt_name:<25}{len(scores):<6}{pct(10):<8.4f}{pct(25):<8.4f}{pct(50):<8.4f}{pct(75):<8.4f}{pct(90):<8.4f}{max(scores):<8.4f}")


if __name__ == "__main__":
    main()
