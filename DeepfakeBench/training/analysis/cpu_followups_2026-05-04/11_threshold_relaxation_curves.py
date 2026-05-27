"""How does viso recall scale with relaxed FPR floor?
Tells us whether the "invisible 66%" of viso fakes are fundamentally low-scoring
or just below the FPR=10% threshold."""
import csv
from pathlib import Path
import numpy as np

ANALYSIS_DIR = Path(__file__).parent
RAW_DIR = ANALYSIS_DIR / "raw_reports"
OUT_DIR = ANALYSIS_DIR / "outputs"

CKPTS = {
    "P8A": "p8a_reference_step5000",
    "E2B_3200": "e2b_top_n_step3200",
    "E3_6600": "e3_top_n_step6600",
}

FAKE_SUITES = ["visomaster_enhanced_macro_dev", "deeplive_enhanced_dev", "teams_fake_all_dev"]


def load_scores(suite, ckpt_token):
    f = RAW_DIR / f"{suite}_{ckpt_token}_frames_report.csv"
    if not f.exists():
        return np.array([]), np.array([])
    s, l = [], []
    with f.open() as fh:
        for row in csv.DictReader(fh):
            s.append(float(row["frame_prob"]))
            l.append(int(row["label"]))
    return np.array(s), np.array(l)


def main():
    rows = []
    fpr_targets = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    for ckpt_name, ckpt_token in CKPTS.items():
        real_s, real_l = load_scores("teams_real_all_dev", ckpt_token)
        real_only = real_s[real_l == 0]
        for fs in FAKE_SUITES:
            fake_s, fake_l = load_scores(fs, ckpt_token)
            fake_only = fake_s[fake_l == 1]
            for fpr in fpr_targets:
                # Calibrate
                sorted_real = np.sort(real_only)
                idx = max(0, int(np.ceil(len(sorted_real) * (1 - fpr))) - 1)
                tau = float(sorted_real[idx])
                achieved = float((real_only >= tau).mean())
                recall = float((fake_only >= tau).mean())
                rows.append({
                    "ckpt": ckpt_name,
                    "fake_suite": fs,
                    "fpr_target": fpr,
                    "fpr_actual": achieved,
                    "tau": tau,
                    "recall": recall,
                })
    out = OUT_DIR / "11_threshold_relaxation_curves.csv"
    with out.open("w", newline="") as fh:
        wrt = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wrt.writeheader()
        wrt.writerows(rows)
    print(f"  → {out}")
    # Pretty print viso row
    print(f"\nVisomaster recall vs threshold relaxation (P8A vs scratch ckpts):")
    print(f"{'fpr_target':10s} {'P8A':10s} {'E2B':10s} {'E3':10s}")
    by_ckpt_fpr = {}
    for r in rows:
        if r["fake_suite"] == "visomaster_enhanced_macro_dev":
            by_ckpt_fpr[(r["ckpt"], r["fpr_target"])] = r["recall"]
    for fpr in fpr_targets:
        p = by_ckpt_fpr.get(("P8A", fpr), 0)
        e2 = by_ckpt_fpr.get(("E2B_3200", fpr), 0)
        e3 = by_ckpt_fpr.get(("E3_6600", fpr), 0)
        print(f"{fpr:.3f}      {p:.3f}     {e2:.3f}     {e3:.3f}")


if __name__ == "__main__":
    main()
