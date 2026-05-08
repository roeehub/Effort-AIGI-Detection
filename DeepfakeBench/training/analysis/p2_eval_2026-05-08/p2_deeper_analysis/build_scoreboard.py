#!/usr/bin/env python3
"""Build unified per-frame scoreboard from the 203 *_frames_report.csv files.

Produces `outputs/scoreboard.parquet` with one row per (frame_path, suite),
with one column per ckpt (frame_prob) plus label + family_key. This is the
foundation for J1-J5.

Source: gs://training-job-outputs/test_results/teams_promotion_contract/
        p2-scratch-scorecard-2026-05-08/reports/*_frames_report.csv

Frame report schema (per row):
    method, label, video_id, frame_path, frame_prob, group_key, family_key

Output schema:
    suite (str), frame_path (str), label (int), method (str), family_key (str),
    {ckpt_alias} (float, prob_fake) for each of 7 ckpts.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO,
                    datefmt="%H:%M:%S")
log = logging.getLogger("scoreboard")

THIS = Path(__file__).resolve().parent
REPORTS = THIS / "frame_reports"
OUT = THIS / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

CKPT_KEYS = [
    "p8a_reference_step5000",
    "e2b_top_n_step3200",
    "p2_c_pairrank_top_n_step7000",
    "p2_c_pairrank_periodic_step3000",
    "p2_d_fourier_periodic_step3000",
    "p2_d_fourier_periodic_step8000",
    "p2_d_fourier_top_n_step19000",
]

# Suite list from the contract manifest.
SUITES = [
    "teams_real_all_dev", "teams_real_poor_quality_dev",
    "teams_real_lighting_extreme_dev", "teams_fake_all_dev",
    "visomaster_enhanced_macro_dev", "deeplive_enhanced_dev",
    "teams_real_all_lockbox", "teams_fake_all_lockbox", "teams_real_dor_dev",
    "teams_real_poor_quality_lockbox", "teams_real_lighting_extreme_lockbox",
    "teams_capture_cam_test_dev", "teams_capture_test_cam_dev",
    "teams_capture_noyn_sharker_dev", "teams_capture_pc_generator_dev",
    "teams_capture_dor_shkedi_dev", "teams_capture_cam_test_s35_dev",
    "teams_capture_noyn_sharker_s23_dev", "teams_capture_cam_test_s32_dev",
    "teams_flat_xiang_xiang2_feng_dev", "teams_capture_pc_generator_s3_dev",
    "teams_capture_test_cam_s53_dev", "teams_capture_cam_test_s46_dev",
    "teams_capture_test_cam_s76_dev", "teams_capture_dor_shkedi_s16_dev",
    "teams_capture_test_cam_s73_dev", "teams_capture_cam_test_s38_dev",
    "teams_capture_pc_generator_s9_dev", "teams_capture_pc_generator_s4_dev",
]


def parse_filename(p: Path) -> tuple[str, str] | None:
    """Return (suite_name, ckpt_alias) from a frame_report filename, or None."""
    name = p.name.replace("_frames_report.csv", "")
    # Try each suite (longest match first to handle nested name overlaps).
    for suite in sorted(SUITES, key=len, reverse=True):
        prefix = f"{suite}_"
        if name.startswith(prefix):
            ckpt = name[len(prefix):]
            return suite, ckpt
    return None


def main():
    files = sorted(REPORTS.glob("*_frames_report.csv"))
    log.info("found %d frame reports", len(files))

    parts: list[pd.DataFrame] = []
    skipped = 0
    for f in files:
        parsed = parse_filename(f)
        if parsed is None:
            log.warning("skip unrecognized filename: %s", f.name)
            skipped += 1
            continue
        suite, ckpt = parsed
        df = pd.read_csv(f)
        df = df[["frame_path", "label", "method", "family_key", "frame_prob"]].copy()
        df["suite"] = suite
        df["ckpt"] = ckpt
        parts.append(df)
    log.info("loaded %d reports, %d skipped", len(parts), skipped)

    long = pd.concat(parts, axis=0, ignore_index=True)
    log.info("long-form rows: %d (suite × ckpt × frame)", len(long))

    # Pivot so each row is one (frame_path, suite) with per-ckpt columns.
    # Some frames may appear in multiple suites (per-identity slices share frames
    # with macro suites); we keep them all — same frame in two suites is two
    # rows.
    wide = long.pivot_table(
        index=["suite", "frame_path", "label", "method", "family_key"],
        columns="ckpt",
        values="frame_prob",
        aggfunc="first",
    ).reset_index()

    # Coerce CKPT columns to float; flatten names.
    wide.columns.name = None
    for ckpt in CKPT_KEYS:
        if ckpt not in wide.columns:
            log.warning("ckpt %s missing from pivot", ckpt)
            wide[ckpt] = float("nan")

    log.info("wide-form rows: %d", len(wide))
    log.info("per-ckpt non-null counts:\n%s",
             wide[CKPT_KEYS].notna().sum().to_string())

    out_path = OUT / "scoreboard.parquet"
    wide.to_parquet(out_path, index=False)
    log.info("wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1024 / 1024)

    # Brief per-suite summary.
    print()
    print("=== Per-suite n_frames ===")
    summary = long.groupby(["suite", "ckpt"])["frame_path"].count().unstack(fill_value=0)
    print(summary.to_string())


if __name__ == "__main__":
    main()
