"""
Validation v2: focused on the right granularity.

Training-eval pool is mostly 1-2 frames per video_id. Rules like Option 3
need >= ~20 frames to aggregate meaningfully. So:

  Path A (production-realistic aggregation):
    - Group all frames per (suite, base_identity) using best-effort regex
    - Keep only identities with >= MIN_FRAMES_PER_IDENTITY frames
    - Apply Options 1/2/3 and report identity-correct
    - This validates the RULE on independent data

  Path B (frame-level scorecard for context):
    - All frames at τ=0.49, 0.6, 0.7 single thresholds
    - Per-suite FPR / recall
    - Tests T5C's raw discriminatory power, not the rule
"""
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
    "teams_real_all_dev":     (0, "teams_real_all_dev_t5c_periodic_step3500_frames_report.csv"),
    "teams_real_all_lockbox": (0, "teams_real_all_lockbox_t5c_periodic_step3500_frames_report.csv"),
    "teams_fake_all_dev":     (1, "teams_fake_all_dev_t5c_periodic_step3500_frames_report.csv"),
    "teams_fake_all_lockbox": (1, "teams_fake_all_lockbox_t5c_periodic_step3500_frames_report.csv"),
    "visomaster_enhanced":    (1, "visomaster_enhanced_macro_dev_t5c_periodic_step3500_frames_report.csv"),
    "deeplive_enhanced":      (1, "deeplive_enhanced_dev_t5c_periodic_step3500_frames_report.csv"),
}

MIN_FRAMES_PER_IDENTITY = 20  # multi-frame aggregation rules need this many


def parse_base_identity(video_id: str) -> str:
    """Handle both `Name__s##__seg_X__suffix` and `Name__seqXXXX__suffix` formats."""
    s = str(video_id)
    m = re.match(r"^(.*?)__seg_", s)
    if m:
        return m.group(1)
    m = re.match(r"^(.*?)__seq", s)
    if m:
        return m.group(1)
    return s


def load_all() -> pd.DataFrame:
    rows = []
    for suite, (lbl, fname) in SUITES.items():
        path = SCORECARD / fname
        df = pd.read_csv(path)
        df["suite"] = suite
        df["label"] = lbl
        df["base_identity"] = df["video_id"].astype(str).map(parse_base_identity)
        rows.append(df[["suite", "label", "base_identity", "video_id", "frame_prob"]])
    return pd.concat(rows, ignore_index=True)


def aggregate_identity(df: pd.DataFrame, min_frames: int) -> pd.DataFrame:
    rows = []
    for (suite, bid), g in df.groupby(["suite", "base_identity"]):
        scores = g["frame_prob"].values
        if len(scores) < min_frames:
            continue
        rec = {
            "suite": suite,
            "base_identity": bid,
            "label": int(g["label"].iloc[0]),
            "n_frames": int(len(scores)),
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "p90": float(np.percentile(scores, 90)),
        }
        for t in [0.49, 0.5, 0.6, 0.7, 0.8, 0.9]:
            rec[f"frac_gt_{t}"] = float((scores > t).mean())
            rec[f"count_gt_{t}"] = int((scores > t).sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def apply_rules(per_id: pd.DataFrame) -> pd.DataFrame:
    df = per_id.copy()
    df["opt1"] = (df["frac_gt_0.49"] > 0.5).astype(int)
    df["opt2"] = (df["frac_gt_0.6"]  > 0.4).astype(int)
    df["opt3"] = ((df["frac_gt_0.6"] > 0.4) & (df["count_gt_0.9"] >= 1)).astype(int)
    return df


def score_per_suite(per_id_with_rules: pd.DataFrame, rule_col: str):
    rows = []
    for suite, g in per_id_with_rules.groupby("suite"):
        truth = g["label"].values
        verdict = g[rule_col].values
        if truth.sum() > 0:
            recall = float((verdict & truth).sum() / truth.sum())
        else:
            recall = None
        if (1 - truth).sum() > 0:
            fpr = float((verdict & (1 - truth)).sum() / (1 - truth).sum())
        else:
            fpr = None
        rows.append({
            "rule": rule_col,
            "suite": suite,
            "n_identities": len(g),
            "label": int(truth[0]),
            "correct": int((verdict == truth).sum()),
            "rate": float((verdict == truth).mean()),
            "recall_fake": recall,
            "fpr_real": fpr,
        })
    return rows


def frame_level_scorecard(df: pd.DataFrame, taus=(0.49, 0.5, 0.6, 0.7, 0.8, 0.9)):
    rows = []
    for suite, g in df.groupby("suite"):
        truth = g["label"].values
        scores = g["frame_prob"].values
        for tau in taus:
            verdict = (scores > tau).astype(int)
            if truth.sum() > 0:
                recall = float((verdict & truth).sum() / truth.sum())
            else:
                recall = None
            if (1 - truth).sum() > 0:
                fpr = float((verdict & (1 - truth)).sum() / (1 - truth).sum())
            else:
                fpr = None
            rows.append({
                "suite": suite,
                "n_frames": len(g),
                "label": int(truth[0]),
                "tau": tau,
                "recall_fake": recall,
                "fpr_real": fpr,
            })
    return pd.DataFrame(rows)


def main():
    raw = load_all()
    print(f"Loaded {len(raw)} frames")
    print(raw.groupby("suite").size())

    # ===== PATH A: per-identity aggregation =====
    per_id = aggregate_identity(raw, min_frames=MIN_FRAMES_PER_IDENTITY)
    print(f"\n=== PATH A: per-identity with >={MIN_FRAMES_PER_IDENTITY} frames ===")
    print(f"Total qualifying identities: {len(per_id)}")
    print(per_id.groupby(["suite", "label"]).size().to_string())

    per_id = apply_rules(per_id)
    per_id.to_csv(OUT_DIR / "validation_v2_per_identity.csv", index=False)

    print("\n=== Per-suite rule outcomes ===")
    all_rows = []
    for rule in ["opt1", "opt2", "opt3"]:
        rows = score_per_suite(per_id, rule)
        all_rows.extend(rows)
    df_sc = pd.DataFrame(all_rows)
    df_sc.to_csv(OUT_DIR / "validation_v2_per_suite.csv", index=False)
    for rule in ["opt1", "opt2", "opt3"]:
        print(f"\n--- {rule} ---")
        sub = df_sc[df_sc["rule"] == rule][["suite", "n_identities", "label", "correct", "rate", "recall_fake", "fpr_real"]]
        print(sub.to_string(index=False))

    # Combined identity-correct across all suites
    print("\n=== Combined identity-correct rate (all qualifying identities) ===")
    truth = per_id["label"].values
    for rule in ["opt1", "opt2", "opt3"]:
        verdict = per_id[rule].values
        correct = (verdict == truth).sum()
        total = len(per_id)
        fp = ((verdict == 1) & (truth == 0)).sum()
        fn = ((verdict == 0) & (truth == 1)).sum()
        print(f"  {rule}: {correct}/{total} = {100*correct/total:.2f}%  (FP={fp}, FN={fn})")

    # Identify FPs under each rule
    print("\n=== REAL identities falsely flagged under each rule ===")
    reals = per_id[per_id["label"] == 0]
    print(f"Total real identities: {len(reals)}")
    for rule in ["opt1", "opt2", "opt3"]:
        flagged = reals[reals[rule] == 1]
        print(f"\n--- {rule}: {len(flagged)} FPs ({100*len(flagged)/len(reals):.2f}%) ---")
        if len(flagged):
            cols = ["suite", "base_identity", "n_frames", "mean", "frac_gt_0.6", "frac_gt_0.7", "frac_gt_0.9", "count_gt_0.9"]
            print(flagged[cols].sort_values("mean", ascending=False).head(15).to_string(index=False))

    # Fakes that escape each rule
    print("\n=== FAKE identities that ESCAPE each rule ===")
    fakes = per_id[per_id["label"] == 1]
    print(f"Total fake identities: {len(fakes)}")
    for rule in ["opt1", "opt2", "opt3"]:
        escaped = fakes[fakes[rule] == 0]
        print(f"\n--- {rule}: {len(escaped)} FNs ({100*len(escaped)/len(fakes):.2f}%) ---")
        if len(escaped):
            cols = ["suite", "base_identity", "n_frames", "mean", "frac_gt_0.6", "frac_gt_0.7", "frac_gt_0.9", "count_gt_0.9"]
            print(escaped[cols].sort_values("mean", ascending=False).head(15).to_string(index=False))

    # ===== PATH B: frame-level scorecard =====
    print("\n\n=== PATH B: frame-level scorecard at fixed thresholds ===")
    df_frame = frame_level_scorecard(raw)
    df_frame.to_csv(OUT_DIR / "validation_v2_frame_level.csv", index=False)

    # Print real suites (we want low FPR)
    print("\n--- Real suites (lower FPR is better) ---")
    print(df_frame[df_frame["label"] == 0].pivot(index="suite", columns="tau", values="fpr_real").round(4))
    print("\n--- Fake suites (higher recall is better) ---")
    print(df_frame[df_frame["label"] == 1].pivot(index="suite", columns="tau", values="recall_fake").round(4))


if __name__ == "__main__":
    main()
