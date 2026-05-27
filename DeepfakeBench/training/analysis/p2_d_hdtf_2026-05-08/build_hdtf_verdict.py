"""Reconstruct HDTF Phase C verdict from per-frame reports.

The cloud `score_teams_promotion_contract.py` aggregator failed at the end of
the Vertex job (known bug — name-mismatch between Phase A `teams_*` suite
names and Phase C `proper_*` suite names; same as P1 PE 2026-05-07). All 48
per-frame reports (3 ckpts × 16 HDTF suites) are intact in `raw_reports/`.

This script applies each ckpt's Phase A contract-selected τ (from
`analysis/p2_eval_2026-05-08/P2_PHASE_A_VERDICT_FACTS_2026-05-08.md`) to
the HDTF per-frame reports and computes per-suite real_fpr / fake_recall
at frame level and (for cross-comparison with the τ=0.5 sidecar) at video
level via mean-of-frame aggregation.

Run: `python build_hdtf_verdict.py`
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "raw_reports"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# Phase A contract-selected τ from
# analysis/p2_eval_2026-05-08/P2_PHASE_A_VERDICT_FACTS_2026-05-08.md §1.
TAU_PHASE_A = {
    "p8a_reference_step5000": 0.9156,
    "e2b_top_n_step3200": 0.7108,
    "p2_d_fourier_periodic_step3000": 0.4600,
}

CKPT_DISPLAY = {
    "p8a_reference_step5000": "P8A_REFERENCE_STEP5000",
    "e2b_top_n_step3200": "E2B_TOP_N_STEP3200",
    "p2_d_fourier_periodic_step3000": "P2_D_FOURIER_PERIODIC_STEP3000",
}

REAL_SUITES = {
    "proper_real_teams_dev",
    "proper_real_teams_lockbox",
    "proper_real_clean_dev",
    "proper_real_clean_lockbox",
}

FAKE_SUITES = {
    "proper_fake_teams_all_dev",
    "proper_fake_teams_all_lockbox",
    "proper_fake_clean_all_dev",
    "proper_fake_clean_all_lockbox",
    "proper_visomaster_teams_dev",
    "proper_visomaster_teams_lockbox",
    "proper_visomaster_clean_dev",
    "proper_visomaster_clean_lockbox",
    "proper_visomaster_enhanced_teams_dev",
    "proper_visomaster_enhanced_teams_lockbox",
    "proper_visomaster_enhanced_clean_dev",
    "proper_visomaster_enhanced_clean_lockbox",
}

ALL_SUITES = REAL_SUITES | FAKE_SUITES
assert len(ALL_SUITES) == 16, f"Expected 16 suites, got {len(ALL_SUITES)}"


def parse_filename(fname: str) -> tuple[str, str]:
    """`proper_real_teams_dev_p8a_reference_step5000_frames_report.csv` →
    (`proper_real_teams_dev`, `p8a_reference_step5000`)."""
    stem = fname[: -len("_frames_report.csv")]
    for ckpt_key in TAU_PHASE_A:
        suffix = "_" + ckpt_key
        if stem.endswith(suffix):
            return stem[: -len(suffix)], ckpt_key
    raise ValueError(f"Unknown ckpt suffix in: {fname}")


def per_suite_stats(rows: list[dict], tau: float) -> dict:
    """Compute frame-level real_fpr (label==0) or fake_recall (label==1) at τ.

    Also computes video-level via mean-of-frame aggregation per video_id (for
    parity comparison with the cloud diagnostic_scorecard τ=0.5 sidecar
    which is video-level)."""
    n_frames = 0
    n_pos_frames = 0  # frames with prob > τ
    label_set = set()
    per_video: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        label = int(r["label"])
        label_set.add(label)
        prob = float(r["frame_prob"])
        n_frames += 1
        if prob > tau:
            n_pos_frames += 1
        per_video[r["video_id"]].append(prob)
    n_videos = len(per_video)
    n_pos_videos = sum(1 for probs in per_video.values()
                       if (sum(probs) / len(probs)) > tau)
    label = label_set.pop() if len(label_set) == 1 else None
    return {
        "n_frames": n_frames,
        "n_videos": n_videos,
        "label": label,
        "frame_pos_rate": n_pos_frames / max(n_frames, 1),
        "video_pos_rate": n_pos_videos / max(n_videos, 1),
    }


def main():
    rows_out: list[dict] = []
    files = sorted(RAW.glob("*_frames_report.csv"))
    assert len(files) == 48, f"Expected 48 reports, got {len(files)}"

    for path in files:
        suite, ckpt_key = parse_filename(path.name)
        assert suite in ALL_SUITES, f"Unexpected suite: {suite}"
        tau = TAU_PHASE_A[ckpt_key]
        with path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        stats = per_suite_stats(rows, tau)
        is_real = suite in REAL_SUITES
        # Sanity: label must match suite type
        if stats["label"] is not None:
            assert stats["label"] == (0 if is_real else 1), (
                f"Label mismatch: suite={suite} label={stats['label']}"
            )
        metric_name = "real_fpr" if is_real else "fake_recall"
        rows_out.append({
            "ckpt": CKPT_DISPLAY[ckpt_key],
            "ckpt_key": ckpt_key,
            "suite": suite,
            "is_real": is_real,
            "tau": tau,
            "n_frames": stats["n_frames"],
            "n_videos": stats["n_videos"],
            "metric": metric_name,
            "frame_level": stats["frame_pos_rate"],
            "video_level": stats["video_pos_rate"],
        })

    # Write per-cell table
    out_csv = OUT / "hdtf_per_cell_at_phase_a_tau.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ckpt", "ckpt_key", "suite", "is_real", "tau",
            "n_frames", "n_videos", "metric",
            "frame_level", "video_level",
        ])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Wrote {out_csv} ({len(rows_out)} rows)")

    # Aggregations
    summary = {ckpt_key: {} for ckpt_key in TAU_PHASE_A}
    for r in rows_out:
        ck = r["ckpt_key"]
        summary[ck][r["suite"]] = {
            "metric": r["metric"],
            "frame_level": r["frame_level"],
            "video_level": r["video_level"],
            "n_frames": r["n_frames"],
            "n_videos": r["n_videos"],
        }
        # Also store τ
        summary[ck]["_tau_phase_a"] = TAU_PHASE_A[ck]
        summary[ck]["_ckpt"] = CKPT_DISPLAY[ck]

    # Macro aggregates per ckpt
    for ck, suites in summary.items():
        real_fprs = [v["frame_level"] for k, v in suites.items()
                     if k in REAL_SUITES]
        fake_recalls = [v["frame_level"] for k, v in suites.items()
                        if k in FAKE_SUITES]
        suites["_macro_real_fpr_frame"] = sum(real_fprs) / len(real_fprs)
        suites["_macro_fake_recall_frame"] = sum(fake_recalls) / len(fake_recalls)
        # Worst real / fake
        suites["_worst_real_fpr_frame"] = max(real_fprs)
        suites["_worst_fake_recall_frame"] = min(fake_recalls)
        real_fprs_v = [v["video_level"] for k, v in suites.items()
                       if k in REAL_SUITES]
        fake_recalls_v = [v["video_level"] for k, v in suites.items()
                          if k in FAKE_SUITES]
        suites["_macro_real_fpr_video"] = sum(real_fprs_v) / len(real_fprs_v)
        suites["_macro_fake_recall_video"] = sum(fake_recalls_v) / len(fake_recalls_v)

    out_json = OUT / "hdtf_summary.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_json}")

    # Print headline table for the FACTS doc
    print()
    print(f"{'suite':<48} {'P8A τ=0.916':>13} {'E2B τ=0.711':>13} {'P2D τ=0.460':>13}")
    print("-" * 92)
    for suite in sorted(REAL_SUITES) + sorted(FAKE_SUITES):
        line = f"{suite:<48}"
        for ck in ["p8a_reference_step5000", "e2b_top_n_step3200",
                   "p2_d_fourier_periodic_step3000"]:
            cell = summary[ck][suite]
            line += f" {cell['frame_level']:>13.4f}"
        print(line)
    print()
    print("Macro aggregates (frame-level):")
    for ck in ["p8a_reference_step5000", "e2b_top_n_step3200",
               "p2_d_fourier_periodic_step3000"]:
        s = summary[ck]
        print(f"  {CKPT_DISPLAY[ck]:<32}"
              f" macro_real_fpr={s['_macro_real_fpr_frame']:.4f}"
              f" macro_fake_recall={s['_macro_fake_recall_frame']:.4f}"
              f" worst_real_fpr={s['_worst_real_fpr_frame']:.4f}"
              f" worst_fake_recall={s['_worst_fake_recall_frame']:.4f}")


if __name__ == "__main__":
    main()
