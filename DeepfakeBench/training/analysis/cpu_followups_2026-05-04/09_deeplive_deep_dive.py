"""Deep-dive on the deeplive ceiling break.

Per (deeplive method) × ckpt: recall at FPR=2/5/10%, identifying which deeplive
sub-method drives the 43%→88% lift. Useful to know whether the lift is broad or
concentrated on a single deeplive variant.

Also: which exact frames does P8A miss but E2B catches? Are they the same frames
across ckpts?
"""
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

ANALYSIS_DIR = Path(__file__).parent
RAW_DIR = ANALYSIS_DIR / "raw_reports"
OUT_DIR = ANALYSIS_DIR / "outputs"

CKPTS = {
    "P8A": "p8a_reference_step5000",
    "E2B_3200": "e2b_top_n_step3200",
    "E3_6600": "e3_top_n_step6600",
}


def load(suite, ckpt_token):
    f = RAW_DIR / f"{suite}_{ckpt_token}_frames_report.csv"
    if not f.exists():
        return []
    with f.open() as fh:
        return [
            {"method": row["method"], "label": int(row["label"]),
             "video_id": row["video_id"], "frame_path": row["frame_path"],
             "score": float(row["frame_prob"])}
            for row in csv.DictReader(fh)
        ]


def calib(real_scores, target_fpr):
    s = np.sort(real_scores)
    idx = max(0, int(np.ceil(len(s) * (1 - target_fpr))) - 1)
    return float(s[idx]) if idx < len(s) else 1.01


def main():
    print("=== Deeplive deep-dive ===")
    # Per ckpt, calibrate τ on teams_real_all_dev
    taus = {}
    for ckpt_name, ckpt_token in CKPTS.items():
        real = load("teams_real_all_dev", ckpt_token)
        rs = np.array([f["score"] for f in real if f["label"] == 0])
        taus[ckpt_name] = {fpr: calib(rs, fpr) for fpr in [0.02, 0.05, 0.10]}

    # Per deeplive method
    rows = []
    by_method_ckpt = defaultdict(dict)
    deeplive_frames = {}
    for ckpt_name, ckpt_token in CKPTS.items():
        deeplive_frames[ckpt_name] = load("deeplive_enhanced_dev", ckpt_token)
        for f in deeplive_frames[ckpt_name]:
            if f["label"] != 1:
                continue
            method = f["method"]
            by_method_ckpt[method].setdefault(ckpt_name, []).append(f["score"])

    # Output: per (method, ckpt, fpr_target) recall table
    for method, by_ckpt in sorted(by_method_ckpt.items()):
        for ckpt_name, scores in by_ckpt.items():
            arr = np.array(scores)
            for fpr_target in [0.02, 0.05, 0.10]:
                tau = taus[ckpt_name][fpr_target]
                recall = float((arr >= tau).mean())
                rows.append({
                    "method": method,
                    "ckpt": ckpt_name,
                    "fpr_target": fpr_target,
                    "tau": tau,
                    "n_frames": len(arr),
                    "recall": recall,
                })
    out = OUT_DIR / "09_deeplive_per_method_recall.csv"
    with out.open("w", newline="") as fh:
        wrt = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wrt.writeheader()
        wrt.writerows(rows)
    print(f"  → {out} ({len(rows)} rows)")

    # Cross-ckpt frame coverage at FPR=10% on deeplive
    # For each deeplive frame: which ckpts catch it?
    frame_caught = defaultdict(dict)  # (method, frame_path) → {ckpt: 0/1}
    for ckpt_name in CKPTS:
        tau = taus[ckpt_name][0.10]
        for f in deeplive_frames[ckpt_name]:
            if f["label"] != 1:
                continue
            frame_caught[(f["method"], f["frame_path"])][ckpt_name] = int(f["score"] >= tau)

    # Coverage by ckpt-set
    coverage = defaultdict(int)
    method_counts = defaultdict(int)
    for (method, _), caught_dict in frame_caught.items():
        method_counts[method] += 1
        if not all(c in caught_dict for c in CKPTS):
            continue
        caught = tuple(sorted(c for c, v in caught_dict.items() if v))
        coverage[(method, caught)] += 1

    # Format as: per method, what fraction of frames is caught by each unique subset?
    cov_rows = []
    for method, total in method_counts.items():
        for k in coverage:
            if k[0] == method:
                caught_set = "+".join(k[1]) if k[1] else "NONE"
                cov_rows.append({
                    "method": method,
                    "caught_by": caught_set,
                    "n_frames": coverage[k],
                    "fraction_of_method": coverage[k] / total,
                })
    cov_rows.sort(key=lambda r: (r["method"], -r["n_frames"]))
    out2 = OUT_DIR / "09b_deeplive_frame_coverage_by_subset.csv"
    with out2.open("w", newline="") as fh:
        wrt = csv.DictWriter(fh, fieldnames=list(cov_rows[0].keys()))
        wrt.writeheader()
        wrt.writerows(cov_rows)
    print(f"  → {out2} ({len(cov_rows)} rows)")

    # Print summary
    print(f"\nPer deeplive method, recall @ FPR=10%:")
    print(f"{'method':40s} {'P8A':8s} {'E2B':8s} {'E3':8s} {'lift_E2B':10s}")
    summary_by_method = defaultdict(dict)
    for r in rows:
        if r["fpr_target"] == 0.10:
            summary_by_method[r["method"]][r["ckpt"]] = r["recall"]
    for m, d in sorted(summary_by_method.items()):
        p, e2, e3 = d.get("P8A", 0), d.get("E2B_3200", 0), d.get("E3_6600", 0)
        lift = e2 - p
        print(f"{m:40s} {p:.3f}    {e2:.3f}    {e3:.3f}    {lift:+.3f}")


if __name__ == "__main__":
    main()
