"""Fix: per-suite AUCs failed because each suite has single-label.
Compute proper binary AUC by joining each fake suite vs the dev real suite.
"""
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

REAL_SUITE_FOR_JOIN = "teams_real_all_dev"
FAKE_SUITES = [
    "visomaster_enhanced_macro_dev",
    "deeplive_enhanced_dev",
    "teams_fake_all_dev",
    "teams_fake_all_lockbox",
]


def load_scores(suite, ckpt_token):
    f = RAW_DIR / f"{suite}_{ckpt_token}_frames_report.csv"
    if not f.exists():
        return np.array([]), np.array([])
    scores, labels = [], []
    with f.open() as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            scores.append(float(row["frame_prob"]))
            labels.append(int(row["label"]))
    return np.array(scores), np.array(labels)


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n_pos, n_neg = len(pos), len(neg)
    all_scores = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_scores)) + 1
    pos_rank_sum = float(ranks[:n_pos].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    rows = []
    # Use teams_real_all_dev as the negative class (apples-to-apples)
    for ckpt_name, ckpt_token in CKPTS.items():
        real_scores, real_labels = load_scores(REAL_SUITE_FOR_JOIN, ckpt_token)
        real_only = real_scores[real_labels == 0]
        for fs in FAKE_SUITES:
            fake_scores, fake_labels = load_scores(fs, ckpt_token)
            fake_only = fake_scores[fake_labels == 1]
            if len(fake_only) == 0:
                continue
            a = auc(fake_only, real_only)
            # also compute mean separability gap
            gap = float(fake_only.mean() - real_only.mean())
            rows.append({
                "ckpt": ckpt_name,
                "fake_suite": fs,
                "real_suite": REAL_SUITE_FOR_JOIN,
                "n_real": len(real_only),
                "n_fake": len(fake_only),
                "auc": a,
                "real_mean_score": float(real_only.mean()),
                "fake_mean_score": float(fake_only.mean()),
                "mean_separation": gap,
            })
    out = OUT_DIR / "02b_cross_suite_auc.csv"
    with out.open("w", newline="") as fh:
        wrt = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wrt.writeheader()
        wrt.writerows(rows)
    print(f"→ {out} ({len(rows)} rows)")
    # print summary
    print(f"\n{'ckpt':10s} {'fake_suite':35s} {'AUC':8s} {'gap':10s}")
    for r in rows:
        print(f"{r['ckpt']:10s} {r['fake_suite']:35s} {r['auc']:.4f}   {r['mean_separation']:+.4f}")


if __name__ == "__main__":
    main()
