"""
Cross-validation of Options 1/2/3 on training-time evaluation suites.

These suites are INDEPENDENT of the 71-identity pool that was used to derive Option 3.
If Option 3 still leads here, the rule is structural rather than fit to test pool.

Suites (T5C step3500 scored frames at /scorecard_reports/):
- teams_real_all_dev   (4564 reals)
- teams_real_all_lockbox (1418 reals)
- teams_fake_all_dev   (3039 fakes)
- teams_fake_all_lockbox (425 fakes)
- visomaster_enhanced_macro_dev (550 fakes)
- deeplive_enhanced_dev (545 fakes)
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
SCORECARD = Path(
    "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/"
    "DeepfakeBench/training/analysis/cpu_diagnostics_2026-05-12_stage_a/_scorecard_reports"
)
OUT_DIR = ROOT / "outputs"

SUITES = {
    "teams_real_all_dev":          (0, "teams_real_all_dev_t5c_periodic_step3500_frames_report.csv"),
    "teams_real_all_lockbox":      (0, "teams_real_all_lockbox_t5c_periodic_step3500_frames_report.csv"),
    "teams_fake_all_dev":          (1, "teams_fake_all_dev_t5c_periodic_step3500_frames_report.csv"),
    "teams_fake_all_lockbox":      (1, "teams_fake_all_lockbox_t5c_periodic_step3500_frames_report.csv"),
    "visomaster_enhanced_dev":     (1, "visomaster_enhanced_macro_dev_t5c_periodic_step3500_frames_report.csv"),
    "deeplive_enhanced_dev":       (1, "deeplive_enhanced_dev_t5c_periodic_step3500_frames_report.csv"),
}


def parse_base_identity(video_id: str) -> str:
    """video_id like `Cam_Test__s32__seg_308.0__real` -> base_identity `Cam_Test__s32`."""
    m = re.match(r"^(.*?)__seg_", video_id)
    if m:
        return m.group(1)
    return video_id


def load_all() -> pd.DataFrame:
    rows = []
    for suite, (lbl, fname) in SUITES.items():
        path = SCORECARD / fname
        df = pd.read_csv(path)
        df["suite"] = suite
        df["label"] = lbl
        df["base_identity"] = df["video_id"].astype(str).map(parse_base_identity)
        rows.append(df[["suite", "label", "base_identity", "video_id", "frame_prob"]])
    out = pd.concat(rows, ignore_index=True)
    print(f"Total frames: {len(out)}")
    for s, g in out.groupby("suite"):
        print(f"  {s}: {len(g)} frames, {g['base_identity'].nunique()} unique base_identities, "
              f"{g['video_id'].nunique()} unique video_ids, label={g['label'].iloc[0]}")
    return out


def aggregate_per_unit(df: pd.DataFrame, group_col="video_id") -> pd.DataFrame:
    """Aggregate frame probs per group (video_id by default, or base_identity)."""
    rows = []
    for (suite, gid), g in df.groupby(["suite", group_col]):
        scores = g["frame_prob"].values
        if len(scores) == 0:
            continue
        rec = {
            "suite": suite,
            group_col: gid,
            "label": int(g["label"].iloc[0]),
            "n_frames": int(len(scores)),
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "p90": float(np.percentile(scores, 90)),
            "max": float(scores.max()),
        }
        for t in [0.3, 0.4, 0.49, 0.5, 0.6, 0.7, 0.8, 0.9]:
            rec[f"frac_gt_{t}"] = float((scores > t).mean())
            rec[f"count_gt_{t}"] = int((scores > t).sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def apply_rules(per_unit: pd.DataFrame) -> pd.DataFrame:
    """Apply Options 1, 2, 3 plus selected N1 variants. Return per-unit verdicts."""
    df = per_unit.copy()
    df["opt1"] = (df["frac_gt_0.49"] > 0.5).astype(int)
    df["opt2"] = (df["frac_gt_0.6"]  > 0.4).astype(int)
    df["opt3"] = ((df["frac_gt_0.6"] > 0.4) & (df["count_gt_0.9"] >= 1)).astype(int)
    # Theoretical variants worth checking
    df["opt3b_stricter_bulk"] = ((df["frac_gt_0.7"] > 0.3) & (df["count_gt_0.9"] >= 1)).astype(int)
    df["opt3c_count_only"]    = ((df["frac_gt_0.49"] > 0.5) & (df["count_gt_0.9"] >= 1)).astype(int)
    return df


def score_suite(df: pd.DataFrame, rule_col: str) -> dict:
    """Per-suite scoring: identity-correct rate, FPR, recall."""
    rows = []
    for suite, g in df.groupby("suite"):
        truth = g["label"].values
        verdict = g[rule_col].values
        correct = (verdict == truth)
        if truth.sum() > 0:
            recall = float((verdict & truth).sum() / truth.sum())
        else:
            recall = float("nan")
        if (1 - truth).sum() > 0:
            fpr = float((verdict & (1 - truth)).sum() / (1 - truth).sum())
        else:
            fpr = float("nan")
        rows.append({
            "rule": rule_col,
            "suite": suite,
            "n": len(g),
            "label": int(truth[0]) if len(truth) else -1,
            "verdict_pos": int(verdict.sum()),
            "correct": int(correct.sum()),
            "rate": float(correct.mean()),
            "recall_if_fake_suite": recall,
            "fpr_if_real_suite": fpr,
        })
    return rows


def bootstrap_paired_delta(df, vA, vB, n_iter=5000, seed=9501):
    """Paired bootstrap: how often is rule_B better than rule_A across resamples?"""
    rng = np.random.default_rng(seed)
    truth = df["label"].values
    correct_A = (vA == truth).astype(float)
    correct_B = (vB == truth).astype(float)
    n = len(df)
    deltas = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        deltas[i] = correct_B[idx].mean() - correct_A[idx].mean()
    return {
        "delta_mean": float(deltas.mean()),
        "delta_p2.5": float(np.percentile(deltas, 2.5)),
        "delta_p97.5": float(np.percentile(deltas, 97.5)),
        "p_gt_0": float((deltas > 0).mean()),
    }


def main():
    raw = load_all()

    # Per video_id (production-relevant: each video = one identity to verdict)
    per_video = aggregate_per_unit(raw, group_col="video_id")
    per_video.to_csv(OUT_DIR / "validation_per_video.csv", index=False)
    print(f"\n=== Per-video aggregations: {len(per_video)} videos ===")
    print(per_video.groupby("suite").size())

    # Per base_identity (chunkier — multiple videos per identity collapse)
    per_id = aggregate_per_unit(raw, group_col="base_identity")
    per_id.to_csv(OUT_DIR / "validation_per_identity.csv", index=False)
    print(f"\n=== Per-base_identity aggregations: {len(per_id)} identities ===")
    print(per_id.groupby("suite").size())

    # Apply rules
    per_video_v = apply_rules(per_video)
    per_id_v = apply_rules(per_id)

    # Per-suite scoring (per_video granularity)
    print("\n=== PER-VIDEO RULE SCORECARD ===")
    all_rule_rows = []
    for col in ["opt1", "opt2", "opt3", "opt3b_stricter_bulk", "opt3c_count_only"]:
        rows = score_suite(per_video_v, col)
        all_rule_rows.extend(rows)
    df_video_score = pd.DataFrame(all_rule_rows)
    # Pretty table by rule + suite
    for rule in ["opt1", "opt2", "opt3", "opt3b_stricter_bulk", "opt3c_count_only"]:
        sub = df_video_score[df_video_score["rule"] == rule]
        print(f"\n--- {rule} ---")
        print(sub[["suite", "n", "label", "verdict_pos", "rate", "recall_if_fake_suite", "fpr_if_real_suite"]].to_string(index=False))

    df_video_score.to_csv(OUT_DIR / "validation_per_video_scorecard.csv", index=False)

    # Per-identity scorecard
    print("\n=== PER-BASE_IDENTITY RULE SCORECARD ===")
    all_rule_rows_id = []
    for col in ["opt1", "opt2", "opt3", "opt3b_stricter_bulk", "opt3c_count_only"]:
        rows = score_suite(per_id_v, col)
        all_rule_rows_id.extend(rows)
    df_id_score = pd.DataFrame(all_rule_rows_id)
    for rule in ["opt1", "opt2", "opt3", "opt3b_stricter_bulk", "opt3c_count_only"]:
        sub = df_id_score[df_id_score["rule"] == rule]
        print(f"\n--- {rule} ---")
        print(sub[["suite", "n", "label", "verdict_pos", "rate", "recall_if_fake_suite", "fpr_if_real_suite"]].to_string(index=False))

    df_id_score.to_csv(OUT_DIR / "validation_per_identity_scorecard.csv", index=False)

    # Bootstrap: Option 3 vs Option 2 vs Option 1 on the combined validation pool
    print("\n=== BOOTSTRAP COMPARISONS ON COMBINED VALIDATION POOL ===")
    print(f"\nPer-video pool ({len(per_video_v)} videos)")
    for v in ["opt2", "opt3", "opt3b_stricter_bulk", "opt3c_count_only"]:
        bs = bootstrap_paired_delta(per_video_v, per_video_v["opt1"].values, per_video_v[v].values)
        print(f"  {v} vs opt1: Δ={bs['delta_mean']:+.4f} [{bs['delta_p2.5']:+.4f}, {bs['delta_p97.5']:+.4f}] "
              f"P(Δ>0)={bs['p_gt_0']:.3f}")
    for v in ["opt3", "opt3b_stricter_bulk", "opt3c_count_only"]:
        bs = bootstrap_paired_delta(per_video_v, per_video_v["opt2"].values, per_video_v[v].values)
        print(f"  {v} vs opt2: Δ={bs['delta_mean']:+.4f} [{bs['delta_p2.5']:+.4f}, {bs['delta_p97.5']:+.4f}] "
              f"P(Δ>0)={bs['p_gt_0']:.3f}")

    print(f"\nPer-identity pool ({len(per_id_v)} identities)")
    for v in ["opt2", "opt3", "opt3b_stricter_bulk", "opt3c_count_only"]:
        bs = bootstrap_paired_delta(per_id_v, per_id_v["opt1"].values, per_id_v[v].values)
        print(f"  {v} vs opt1: Δ={bs['delta_mean']:+.4f} [{bs['delta_p2.5']:+.4f}, {bs['delta_p97.5']:+.4f}] "
              f"P(Δ>0)={bs['p_gt_0']:.3f}")
    for v in ["opt3", "opt3b_stricter_bulk", "opt3c_count_only"]:
        bs = bootstrap_paired_delta(per_id_v, per_id_v["opt2"].values, per_id_v[v].values)
        print(f"  {v} vs opt2: Δ={bs['delta_mean']:+.4f} [{bs['delta_p2.5']:+.4f}, {bs['delta_p97.5']:+.4f}] "
              f"P(Δ>0)={bs['p_gt_0']:.3f}")

    # FP identities under each rule (which reals get falsely flagged?)
    print("\n=== REAL VIDEOS FALSELY FLAGGED (per_video pool) ===")
    reals = per_video_v[per_video_v["label"] == 0]
    for rule in ["opt1", "opt2", "opt3"]:
        flagged = reals[reals[rule] == 1]
        print(f"\n--- {rule}: {len(flagged)} of {len(reals)} reals flagged ({100*len(flagged)/len(reals):.2f}%) ---")
        # Top 10 worst
        if len(flagged):
            top = flagged.nlargest(10, "mean")[["suite", "video_id", "n_frames", "mean", "frac_gt_0.6", "frac_gt_0.7", "frac_gt_0.9", "count_gt_0.9"]]
            print(top.to_string(index=False))

    # Per-identity FP audit
    print("\n=== BASE IDENTITIES FALSELY FLAGGED (per_id pool) ===")
    reals_id = per_id_v[per_id_v["label"] == 0]
    for rule in ["opt1", "opt2", "opt3"]:
        flagged = reals_id[reals_id[rule] == 1]
        print(f"\n--- {rule}: {len(flagged)} of {len(reals_id)} real-identities flagged "
              f"({100*len(flagged)/len(reals_id):.2f}%) ---")
        if len(flagged):
            top = flagged.nlargest(10, "mean")[["suite", "base_identity", "n_frames", "mean", "frac_gt_0.6", "frac_gt_0.7", "frac_gt_0.9", "count_gt_0.9"]]
            print(top.to_string(index=False))


if __name__ == "__main__":
    main()
