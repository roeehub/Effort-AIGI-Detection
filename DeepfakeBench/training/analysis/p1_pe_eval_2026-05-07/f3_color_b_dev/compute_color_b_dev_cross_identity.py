"""
Cross-identity color_b_dev Pearson r — extension of the Roy_D regression
axis-attribution to non-target base_identities on `teams_real_all_dev`.

Open question: Roy_D shows P8A r(score, color_b_dev) = -0.714 and the P1
destruction axis (P1 - P8A) is +0.714 on color_b_dev. Is this an
identity-specific quirk on Roy_D, or a generalized learned feature?

Method:
  1. Reuse the F3 per_frame_color_b_dev.csv (no frame re-decoding needed —
     all 4564 teams_real_all_dev frames already have color_b_dev computed).
  2. For each non-Roy_D base_identity in the target set, join per-frame
     scores from Phase A reports (P8A_REFERENCE_STEP5000 + P1_BUNDLE_PERIODIC_STEP500).
  3. Compute three Pearson r values per identity:
       a. r(P8A_score, color_b_dev) — baseline P8A signal magnitude
       b. r(P1_BUNDLE_step500_score, color_b_dev) — destruction generalization
       c. r((P1 - P8A) Δ_score, color_b_dev) — destruction axis itself
  4. Apply mechanical pass/fail vs |r| > 0.5 threshold.

Output:
  COLOR_B_DEV_CROSS_IDENTITY_FACTS_2026-05-07.md
  cross_identity_color_b_dev_r.csv
"""
from __future__ import annotations

import gc
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")
log = logging.getLogger("color_b_dev_cross_id")

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
THIS_DIR = ROOT / "analysis/p1_pe_eval_2026-05-07/f3_color_b_dev"
PHASE_A = ROOT / "analysis/p1_pe_eval_2026-05-07/raw_reports/phase_a"
PER_FRAME_CSV = THIS_DIR / "per_frame_color_b_dev.csv"

SUITE = "teams_real_all_dev"
P8A_TAG = "p8a_reference_step5000"
P1_TAG = "p1_bundle_periodic_step500"

# Target identities (from FOLLOWUPS_FACTS_2026-05-07.md §6 table — 6 identities
# with n>=100 in teams_real_all_dev, excluding Roy_D, chronic-target Q, and
# small-n identities orel/dor_shkedi/ilan).
TARGET_IDENTITIES = [
    "Test_Cam",
    "PC_Generator",
    "Md_noyn_Sharker",
    "bla_bla_chow",
    "Xiang_Xiang2_Feng",
    "dor",
]

# regex from phase_d/run_chronic_filter.py — collapses session/frame/crop/seq
# tokens to person id.
_IDENT_STRIP_BASE = re.compile(
    r"(__seq\d+|__seg_[\d.]+|__s\d+|_s\d+|_frame_\d+|_seq\d+|__real|__fake|_real|_fake|_crop_\d+|__[0-9a-f]{6,})",
    flags=re.IGNORECASE,
)


def extract_base_identity(vid: str) -> str:
    if not isinstance(vid, str):
        return "UNK"
    s = vid
    prev = None
    while prev != s:
        prev = s
        s = _IDENT_STRIP_BASE.sub("", s)
    return s.strip("_") or "UNK"


def pearson_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return float("nan"), n
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), n
    return float(np.corrcoef(x, y)[0, 1]), n


def main() -> None:
    # 1. Load per-frame color_b_dev (already cached — no re-decoding).
    log.info("loading per_frame_color_b_dev.csv")
    pf_df = pd.read_csv(PER_FRAME_CSV)
    pf_df = pf_df[pf_df["suite"] == SUITE].copy()
    log.info("teams_real_all_dev rows: %d, nonNaN color_b_dev: %d",
             len(pf_df), pf_df["color_b_dev"].notna().sum())

    # 2. Load P8A and P1_BUNDLE_step500 per-frame reports.
    p8a_path = PHASE_A / f"{SUITE}_{P8A_TAG}_frames_report.csv"
    p1_path = PHASE_A / f"{SUITE}_{P1_TAG}_frames_report.csv"
    log.info("loading P8A: %s", p8a_path)
    p8a_df = pd.read_csv(p8a_path)[["video_id", "frame_path", "frame_prob"]]
    p8a_df = p8a_df.rename(columns={"frame_prob": "p8a_score"})
    log.info("loading P1_BUNDLE_step500: %s", p1_path)
    p1_df = pd.read_csv(p1_path)[["video_id", "frame_path", "frame_prob"]]
    p1_df = p1_df.rename(columns={"frame_prob": "p1_bundle_step500_score"})

    # 3. Merge: scores + color_b_dev per frame_path.
    merged = p8a_df.merge(p1_df[["frame_path", "p1_bundle_step500_score"]],
                          on="frame_path", how="inner")
    merged = merged.merge(pf_df[["frame_path", "color_b_dev"]],
                          on="frame_path", how="inner")
    merged["delta_score"] = merged["p1_bundle_step500_score"] - merged["p8a_score"]
    merged["base_identity"] = merged["video_id"].map(extract_base_identity)
    log.info("merged rows: %d (expected ~4564)", len(merged))

    # 4. Per-identity Pearson r.
    rows: list[dict] = []
    for ident in TARGET_IDENTITIES + ["Roy_D"]:  # include Roy_D as in-context anchor
        sub = merged[merged["base_identity"] == ident].copy()
        n_total = len(sub)
        if n_total == 0:
            log.warning("identity %s: no rows in merged", ident)
            continue
        sub = sub.dropna(subset=["color_b_dev", "p8a_score",
                                 "p1_bundle_step500_score", "delta_score"])
        n_eff = len(sub)
        r_p8a, n_p8a = pearson_safe(sub["p8a_score"].values,
                                    sub["color_b_dev"].values)
        r_p1, n_p1 = pearson_safe(sub["p1_bundle_step500_score"].values,
                                  sub["color_b_dev"].values)
        r_delta, n_delta = pearson_safe(sub["delta_score"].values,
                                        sub["color_b_dev"].values)
        rows.append({
            "base_identity": ident,
            "n_frames": n_total,
            "n_effective": n_eff,
            "r_p8a_score_color_b_dev": r_p8a,
            "r_p1_bundle_step500_score_color_b_dev": r_p1,
            "r_delta_p1_minus_p8a_color_b_dev": r_delta,
            "abs_r_p8a": abs(r_p8a) if not np.isnan(r_p8a) else float("nan"),
            "abs_r_p1": abs(r_p1) if not np.isnan(r_p1) else float("nan"),
            "abs_r_delta": abs(r_delta) if not np.isnan(r_delta) else float("nan"),
        })
        log.info("identity=%s n=%d r(P8A)=%+.4f r(P1)=%+.4f r(Δ)=%+.4f",
                 ident, n_eff, r_p8a, r_p1, r_delta)
        gc.collect()

    out_df = pd.DataFrame(rows)
    out_csv = THIS_DIR / "cross_identity_color_b_dev_r.csv"
    out_df.to_csv(out_csv, index=False)
    log.info("wrote %s", out_csv)

    # 5. Markdown report.
    md = []
    md.append("# Cross-identity color_b_dev — Pearson r per base_identity")
    md.append("")
    md.append("**Status**: factual-only. No interpretation. Numbers and direct observations.")
    md.append("")
    md.append("**Question being answered**: P8A's r(score, color_b_dev) on Roy_D = -0.7140; "
              "the P1_BUNDLE_step500 destruction axis Δ-r = +0.7140. Is the P8A "
              "color_b_dev=>real signal Roy_D-specific, or a generalized learned feature?")
    md.append("")
    md.append("**Method**: Reused the F3 `per_frame_color_b_dev.csv` (4564 teams_real_all_dev "
              "frames, color_b_dev = `np.std(BGR_channel_0)` on 0-255 uint8 decode). Joined "
              "per-frame scores from `raw_reports/phase_a/teams_real_all_dev_p8a_reference_step5000_frames_report.csv` "
              "and `..._p1_bundle_periodic_step500_frames_report.csv`. Grouped by `base_identity` "
              "(via the `extract_base_identity` regex from `phase_d/run_chronic_filter.py`). "
              "Per identity, computed three Pearson r values via `numpy.corrcoef` on "
              "(P8A_score, color_b_dev), (P1_BUNDLE_step500_score, color_b_dev), and "
              "((P1-P8A) Δ_score, color_b_dev).")
    md.append("")
    md.append("**Sample size**: 6 non-Roy_D base_identities (all n ≥ 100 in `teams_real_all_dev`; "
              "non-chronic-target). Roy_D included as the anchor row to verify reproduction.")
    md.append("")
    md.append("**Compute**: CPU only, num_workers=0, gc.collect() after each identity. No DataLoader.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Per-identity Pearson r")
    md.append("")
    md.append("| base_identity | n | r(P8A, color_b_dev) | r(P1_BUNDLE_step500, color_b_dev) | r(Δ_score, color_b_dev) |")
    md.append("| --- | ---: | ---: | ---: | ---: |")
    for ident in TARGET_IDENTITIES + ["Roy_D"]:
        sub = out_df[out_df["base_identity"] == ident]
        if len(sub) == 0:
            continue
        row = sub.iloc[0]
        md.append(
            f"| {ident} | {row['n_effective']} | {row['r_p8a_score_color_b_dev']:+.4f} | "
            f"{row['r_p1_bundle_step500_score_color_b_dev']:+.4f} | "
            f"{row['r_delta_p1_minus_p8a_color_b_dev']:+.4f} |"
        )
    md.append("")
    md.append("## Mechanical pass/fail against |r| > 0.5")
    md.append("")
    md.append("Threshold: |r(P8A, color_b_dev)| > 0.5 indicates a strong P8A learned signal "
              "on the identity. If the magnitude is consistent across identities, the signal "
              "is generalized; if only Roy_D meets threshold, it is identity-specific.")
    md.append("")
    md.append("| base_identity | |r(P8A)| | meets |r|>0.5 | |r(P1)| | meets |r|>0.5 | |r(Δ)| | meets |r|>0.5 |")
    md.append("| --- | ---: | :---: | ---: | :---: | ---: | :---: |")
    for ident in TARGET_IDENTITIES + ["Roy_D"]:
        sub = out_df[out_df["base_identity"] == ident]
        if len(sub) == 0:
            continue
        row = sub.iloc[0]
        ar_p8a = row["abs_r_p8a"]
        ar_p1 = row["abs_r_p1"]
        ar_d = row["abs_r_delta"]
        md.append(
            f"| {ident} | {ar_p8a:.4f} | {'YES' if ar_p8a > 0.5 else 'no'} | "
            f"{ar_p1:.4f} | {'YES' if ar_p1 > 0.5 else 'no'} | "
            f"{ar_d:.4f} | {'YES' if ar_d > 0.5 else 'no'} |"
        )
    md.append("")
    md.append("## Direct observations")
    md.append("")

    # Counts of identities meeting threshold (excluding Roy_D anchor)
    nonroy = out_df[out_df["base_identity"] != "Roy_D"]
    n_meet_p8a = int((nonroy["abs_r_p8a"] > 0.5).sum())
    n_meet_p1 = int((nonroy["abs_r_p1"] > 0.5).sum())
    n_meet_delta = int((nonroy["abs_r_delta"] > 0.5).sum())
    n_total = len(nonroy)

    roy = out_df[out_df["base_identity"] == "Roy_D"].iloc[0]
    md.append(
        f"1. Roy_D anchor row reproduces the prior |r(P8A)| = {roy['abs_r_p8a']:.3f} "
        f"({'matches' if abs(roy['abs_r_p8a'] - 0.714) < 0.01 else 'differs from'} the 0.714 "
        f"reading in `roy_d_with_axes.csv`)."
    )
    md.append("")
    md.append(
        f"2. Of {n_total} non-Roy_D identities tested, {n_meet_p8a} meet |r(P8A)| > 0.5; "
        f"{n_meet_p1} meet |r(P1_BUNDLE_step500)| > 0.5; "
        f"{n_meet_delta} meet |r(Δ)| > 0.5."
    )
    md.append("")

    # Aggregate stats on non-Roy identities
    p8a_mean_abs = nonroy["abs_r_p8a"].mean()
    p8a_max_abs = nonroy["abs_r_p8a"].max()
    p1_mean_abs = nonroy["abs_r_p1"].mean()
    delta_mean_abs = nonroy["abs_r_delta"].mean()
    md.append(
        f"3. Non-Roy_D aggregate: mean |r(P8A)| = {p8a_mean_abs:.3f}, "
        f"max |r(P8A)| = {p8a_max_abs:.3f}; "
        f"mean |r(P1_BUNDLE_step500)| = {p1_mean_abs:.3f}; "
        f"mean |r(Δ)| = {delta_mean_abs:.3f}."
    )
    md.append("")

    # Sign analysis: how many P8A r have the same sign as Roy_D's negative
    p8a_neg = nonroy[nonroy["r_p8a_score_color_b_dev"] < 0]
    p8a_pos = nonroy[nonroy["r_p8a_score_color_b_dev"] >= 0]
    md.append(
        f"4. Sign of r(P8A, color_b_dev) on non-Roy_D identities: {len(p8a_neg)} negative "
        f"(same sign as Roy_D), {len(p8a_pos)} non-negative."
    )
    md.append("")

    # Compare Roy_D vs non-Roy maxima
    md.append(
        f"5. Roy_D |r(P8A)| = {roy['abs_r_p8a']:.3f}. Highest non-Roy_D |r(P8A)| = "
        f"{p8a_max_abs:.3f} on identity = "
        f"{nonroy.loc[nonroy['abs_r_p8a'].idxmax(), 'base_identity']}. "
        f"Ratio: {roy['abs_r_p8a'] / max(p8a_max_abs, 1e-9):.2f}×."
    )
    md.append("")

    # Δ axis structure: do destruction-axis r values mirror P8A r magnitudes?
    md.append("6. Per-identity correspondence between |r(P8A)| and |r(Δ)|:")
    md.append("")
    md.append("| base_identity | |r(P8A)| | |r(Δ)| | ratio (Δ/P8A) |")
    md.append("| --- | ---: | ---: | ---: |")
    for ident in TARGET_IDENTITIES + ["Roy_D"]:
        sub = out_df[out_df["base_identity"] == ident]
        if len(sub) == 0:
            continue
        row = sub.iloc[0]
        ar_p8a = row["abs_r_p8a"]
        ar_d = row["abs_r_delta"]
        ratio = ar_d / max(ar_p8a, 1e-9) if ar_p8a > 1e-9 else float("nan")
        md.append(f"| {ident} | {ar_p8a:.4f} | {ar_d:.4f} | {ratio:.2f} |")
    md.append("")
    md.append("## Subsampling")
    md.append("")
    md.append("None. All 4564 `teams_real_all_dev` Phase A frames were used; "
              "per-identity n is the full count from the merged report.")
    md.append("")
    md.append("## Companion artifacts")
    md.append("")
    md.append("- `cross_identity_color_b_dev_r.csv` — per-identity Pearson r table "
              "(7 rows × 9 cols).")
    md.append("- Source data: `f3_color_b_dev/per_frame_color_b_dev.csv` (color_b_dev), "
              "`raw_reports/phase_a/teams_real_all_dev_{p8a_reference_step5000,p1_bundle_periodic_step500}_frames_report.csv` (scores).")
    md.append("")
    md.append("## Cross-references")
    md.append("")
    md.append("- `roy_d_regression/ROY_D_REGRESSION_FACTS_2026-05-07.md` — original Roy_D r=-0.714 finding.")
    md.append("- `FOLLOWUPS_FACTS_2026-05-07.md` §1 — Task A roy_d axis attribution.")
    md.append("- `FOLLOWUPS_FACTS_2026-05-07.md` §6 — top 13 base_identities × Δ FPR table.")

    md_path = THIS_DIR / "COLOR_B_DEV_CROSS_IDENTITY_FACTS_2026-05-07.md"
    md_path.write_text("\n".join(md))
    log.info("wrote %s", md_path)
    log.info("done")


if __name__ == "__main__":
    main()
