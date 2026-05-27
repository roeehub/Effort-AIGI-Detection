"""For each fake suite, compute frame coverage by ckpt-subset.
Tells us: which frames are caught by which combinations of ckpts at FPR=10%.

Key question: where P8A wins on viso, does it catch frames the scratch ckpts ALSO catch
(plus more), or DIFFERENT frames?
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

FAKE_SUITES = ["visomaster_enhanced_macro_dev", "deeplive_enhanced_dev",
               "teams_fake_all_dev", "teams_fake_all_lockbox"]


def load(suite, ckpt_token):
    f = RAW_DIR / f"{suite}_{ckpt_token}_frames_report.csv"
    if not f.exists():
        return []
    with f.open() as fh:
        return list(csv.DictReader(fh))


def calib(real_scores, target_fpr):
    s = np.sort(real_scores)
    idx = max(0, int(np.ceil(len(s) * (1 - target_fpr))) - 1)
    return float(s[idx]) if idx < len(s) else 1.01


def main():
    # Calibrate τ for each ckpt on teams_real_all_dev
    taus = {}
    for ckpt_name, ckpt_token in CKPTS.items():
        real = load("teams_real_all_dev", ckpt_token)
        rs = np.array([float(f["frame_prob"]) for f in real if int(f["label"]) == 0])
        taus[ckpt_name] = calib(rs, 0.10)

    rows = []
    for suite in FAKE_SUITES:
        # For each fake frame: which ckpts catch it at FPR=10%?
        frame_caught = defaultdict(dict)  # frame_path → {ckpt: 0/1}
        for ckpt_name, ckpt_token in CKPTS.items():
            tau = taus[ckpt_name]
            for f in load(suite, ckpt_token):
                if int(f["label"]) != 1:
                    continue
                frame_caught[f["frame_path"]][ckpt_name] = int(float(f["frame_prob"]) >= tau)

        coverage = defaultdict(int)
        total = 0
        for frame_path, caught_dict in frame_caught.items():
            if not all(c in caught_dict for c in CKPTS):
                continue
            total += 1
            caught = tuple(sorted(c for c, v in caught_dict.items() if v))
            coverage[caught] += 1

        for subset, n in sorted(coverage.items(), key=lambda x: -x[1]):
            label = "+".join(subset) if subset else "NONE"
            rows.append({
                "suite": suite,
                "caught_by": label,
                "n_frames": n,
                "fraction_of_total_fakes": n / max(total, 1),
            })
        rows.append({"suite": suite, "caught_by": "TOTAL_FRAMES", "n_frames": total, "fraction_of_total_fakes": 1.0})

    out = OUT_DIR / "10_frame_coverage_all_suites.csv"
    with out.open("w", newline="") as fh:
        wrt = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wrt.writeheader()
        wrt.writerows(rows)
    print(f"  → {out}")

    # Pretty print
    print(f"\n{'suite':35s} {'caught_by':30s} {'n':6s} {'frac':6s}")
    for r in rows:
        print(f"{r['suite']:35s} {r['caught_by']:30s} {r['n_frames']:6d} {r['fraction_of_total_fakes']:.3f}")


if __name__ == "__main__":
    main()
