"""Join the per-ckpt may6/may5 scores into a single wide table + compute mode-band FPRs.

Reads:
  - outputs/scores_<KEY>.csv for each ckpt (produced by score_may6_may5.py)

Writes:
  - outputs/joint_scores.csv         — long table: ckpt × population × frame × prob_fake
  - outputs/wide_scores.csv          — wide table: one row per frame, one col per ckpt
  - outputs/fpr_by_mode.csv          — per (ckpt, cohort, threshold) FPR + dist stats
  - outputs/fpr_by_mode.txt          — human-readable summary

Usage:
    python analysis/xinhe_may6_t5c_revisit_2026-05-23/scripts/analyze.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = THIS_DIR / "outputs"
CACHED_DIR = THIS_DIR.parent / "xinhe_cross_camera_audit_2026-05-06" / "outputs"

CKPTS = [
    "P8A_REFERENCE_STEP5000",
    "E2B_TOP_N_STEP3200",
    "T5C_PERIODIC_STEP3500",
    "SLOT_A_V2_CLS_STEP3500",
    "SLOT_A_V2_FACE_POOL_STEP3500",
]

# τ thresholds: original E2B claim used 0.5; modes A/B/C from
# project_deployment_three_modes_slot_a_v2_2026-05-21.md
TAUS = {
    "tau_0p50_e2b_2026_05_06": 0.50,
    "modeA_tau_0p535": 0.535,
    "modeB_tau_0p780": 0.78,
    "modeC_tau_0p870": 0.87,
}


def load_per_ckpt() -> pd.DataFrame:
    """Long table: rows = (ckpt, population, frame_basename, frame_path, prob_fake)."""
    frames = []
    for ck in CKPTS:
        path = OUT_DIR / f"scores_{ck}.csv"
        if not path.exists():
            print(f"[warn] missing scores file: {path}")
            continue
        df = pd.read_csv(path)
        df["ckpt"] = ck
        frames.append(df)
    if not frames:
        raise RuntimeError(f"no per-ckpt CSVs found under {OUT_DIR}")
    return pd.concat(frames, ignore_index=True)


def load_cached() -> pd.DataFrame:
    """Load the 2026-05-06 cached P8A + E2B scores from the prior audit."""
    path = CACHED_DIR / "scores_all_5_ckpts_may6_may5.csv"
    return pd.read_csv(path)


def to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one row per frame, one column per ckpt."""
    wide = long_df.pivot_table(
        index=["population", "frame_basename"],
        columns="ckpt", values="prob_fake",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def compute_fpr_table(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ck in long_df.ckpt.unique():
        for pop in ["may6_falseflag", "may5_correct"]:
            sub = long_df[(long_df.ckpt == ck) & (long_df.population == pop)]
            n = len(sub)
            if n == 0:
                continue
            scores = sub.prob_fake.values
            row = {
                "ckpt": ck,
                "cohort": pop,
                "n": n,
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
                "p95": float(np.percentile(scores, 95)),
                "max": float(scores.max()),
                "n_gt_0p9": int((scores > 0.9).sum()),
            }
            for label, tau in TAUS.items():
                row[f"fpr_{label}"] = float((scores >= tau).mean())
                row[f"n_flag_{label}"] = int((scores >= tau).sum())
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


def reproduction_check(fresh_long: pd.DataFrame, cached: pd.DataFrame) -> str:
    """Compare fresh P8A/E2B scores to the 2026-05-06 cached numbers; emit a report."""
    out_lines = []
    for ck_fresh, ck_cached in [
        ("P8A_REFERENCE_STEP5000", "P8A"),
        ("E2B_TOP_N_STEP3200", "E2B"),
    ]:
        fr = fresh_long[fresh_long.ckpt == ck_fresh].copy()
        if len(fr) == 0:
            out_lines.append(f"[{ck_fresh}] not in fresh scores; skip reproduction check")
            continue
        # Cached CSV has frame_path (full) + per-ckpt column
        # Join on basename
        fr["basename"] = fr.frame_basename.astype(str)
        cached_copy = cached.copy()
        cached_copy["basename"] = cached_copy.frame_path.map(lambda p: p.rsplit("/", 1)[-1])
        cached_copy = cached_copy[["basename", "population", ck_cached]].rename(
            columns={ck_cached: "cached_prob_fake"}
        )
        merged = fr.merge(cached_copy, on=["basename", "population"], how="left")
        merged["abs_diff"] = (merged.prob_fake - merged.cached_prob_fake).abs()
        max_diff = merged.abs_diff.max()
        mean_diff = merged.abs_diff.mean()
        for pop in ["may6_falseflag", "may5_correct"]:
            mp = merged[merged.population == pop]
            fresh_fpr = (mp.prob_fake >= 0.5).mean() * 100
            cached_fpr = (mp.cached_prob_fake >= 0.5).mean() * 100
            fresh_mean = mp.prob_fake.mean()
            cached_mean = mp.cached_prob_fake.mean()
            out_lines.append(
                f"[{ck_fresh}|{pop}] fresh fpr@0.5={fresh_fpr:5.1f}% mean={fresh_mean:.3f}  "
                f"cached fpr@0.5={cached_fpr:5.1f}% mean={cached_mean:.3f}  "
                f"|Δfpr|={abs(fresh_fpr - cached_fpr):.1f}pp  |Δmean|={abs(fresh_mean - cached_mean):.3f}"
            )
        out_lines.append(
            f"[{ck_fresh}] per-frame |Δprob|: max={max_diff:.4f}  mean={mean_diff:.4f}"
        )
    return "\n".join(out_lines)


def format_human_summary(fpr_df: pd.DataFrame, repro_text: str) -> str:
    L = []
    L.append("=" * 86)
    L.append("Xinhe may6/may5 cohort — T5C revisit (2026-05-23)")
    L.append("=" * 86)
    L.append("")
    L.append("Reproduction check (fresh vs 2026-05-06 cached P8A/E2B):")
    L.append(repro_text)
    L.append("")
    L.append("Per (ckpt × cohort) FPR table:")
    L.append("")
    L.append(
        f"{'ckpt':<32}{'cohort':<18}{'n':>4}{'mean':>8}{'p95':>8}{'>0.9':>6}"
        f"{'FPR@0.5':>9}{'modeA':>8}{'modeB':>8}{'modeC':>8}"
    )
    L.append("-" * 110)
    # Order by ckpt then by cohort (may5 first)
    order = {"may5_correct": 0, "may6_falseflag": 1}
    fpr_df = fpr_df.assign(_cohort_order=fpr_df.cohort.map(order))
    fpr_df = fpr_df.sort_values(by=["ckpt", "_cohort_order"]).drop(columns="_cohort_order")
    for _, r in fpr_df.iterrows():
        L.append(
            f"{r['ckpt']:<32}{r['cohort']:<18}{r['n']:>4d}"
            f"{r['mean']:>8.3f}{r['p95']:>8.3f}{r['n_gt_0p9']:>6d}"
            f"{r['fpr_tau_0p50_e2b_2026_05_06']*100:>8.1f}%"
            f"{r['fpr_modeA_tau_0p535']*100:>7.1f}%"
            f"{r['fpr_modeB_tau_0p780']*100:>7.1f}%"
            f"{r['fpr_modeC_tau_0p870']*100:>7.1f}%"
        )
    L.append("")
    L.append("Legend: modeA=τ0.535 (lockbox FPR=10% target), modeB=τ0.78 (FPR=2%, contract-compliant),")
    L.append("        modeC=τ0.87 (FPR=1%, conservative); FPR@0.5 = original 2026-05-06 E2B-claim threshold")
    return "\n".join(L)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_df = load_per_ckpt()
    long_df.to_csv(OUT_DIR / "joint_scores.csv", index=False)
    wide_df = to_wide(long_df)
    wide_df.to_csv(OUT_DIR / "wide_scores.csv", index=False)
    fpr_df = compute_fpr_table(long_df)
    fpr_df.to_csv(OUT_DIR / "fpr_by_mode.csv", index=False)

    cached = load_cached()
    repro_text = reproduction_check(long_df, cached)
    human = format_human_summary(fpr_df, repro_text)
    (OUT_DIR / "fpr_by_mode.txt").write_text(human + "\n")
    print(human)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
