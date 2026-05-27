"""
Double-tail rule exploration: bulk + count>0.9 + count>0.8.

User intuition: "≥1 above 0.9 AND ≥2 above 0.8 per 32 frames" — require the tail
to be 'wider' than a single outlier extreme frame.

Translating to proportional rule for pools with varying n:
  1/32 = 3.13%   (frac > 0.9 ≥ 0.0313)
  2/32 = 6.25%   (frac > 0.8 ≥ 0.0625)

We test:
  - The exact proposed rule
  - Variants of (K1 over 0.9, K2 over 0.8)
  - Both as absolute counts and as fractions

On BOTH pools: 71-identity (deriving pool) + 47-identity independent training-eval pool.
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
SUITES = {
    "teams_real_all_dev":     (0, "teams_real_all_dev_t5c_periodic_step3500_frames_report.csv"),
    "teams_real_all_lockbox": (0, "teams_real_all_lockbox_t5c_periodic_step3500_frames_report.csv"),
    "teams_fake_all_dev":     (1, "teams_fake_all_dev_t5c_periodic_step3500_frames_report.csv"),
    "teams_fake_all_lockbox": (1, "teams_fake_all_lockbox_t5c_periodic_step3500_frames_report.csv"),
    "visomaster_enhanced":    (1, "visomaster_enhanced_macro_dev_t5c_periodic_step3500_frames_report.csv"),
    "deeplive_enhanced":      (1, "deeplive_enhanced_dev_t5c_periodic_step3500_frames_report.csv"),
}
P71_PATH = ROOT / "outputs" / "extreme_rule_per_identity.csv"
OUT_DIR = ROOT / "outputs"
MIN_FRAMES = 20


def parse_base_identity(video_id):
    s = str(video_id)
    m = re.match(r"^(.*?)__seg_", s)
    if m:
        return m.group(1)
    m = re.match(r"^(.*?)__seq", s)
    if m:
        return m.group(1)
    return s


def load_training_eval_per_identity() -> pd.DataFrame:
    rows = []
    for suite, (lbl, fname) in SUITES.items():
        df = pd.read_csv(SCORECARD / fname)
        df["suite"] = suite
        df["label"] = lbl
        df["base_identity"] = df["video_id"].astype(str).map(parse_base_identity)
        rows.append(df[["suite", "label", "base_identity", "frame_prob"]])
    raw = pd.concat(rows, ignore_index=True)

    out = []
    for (suite, bid), g in raw.groupby(["suite", "base_identity"]):
        scores = g["frame_prob"].values
        if len(scores) < MIN_FRAMES:
            continue
        rec = {"suite": suite, "base_identity": bid, "label": int(g["label"].iloc[0]),
               "n_frames": int(len(scores))}
        for t in [0.49, 0.5, 0.6, 0.7, 0.8, 0.9]:
            rec[f"frac_gt_{t}"] = float((scores > t).mean())
            rec[f"count_gt_{t}"] = int((scores > t).sum())
        out.append(rec)
    return pd.DataFrame(out)


def load_71_pool() -> pd.DataFrame:
    df = pd.read_csv(P71_PATH)
    keep = ["pool", "suite", "base_identity", "label", "n_frames",
            "frac_gt_0.49", "count_gt_0.49",
            "frac_gt_0.5", "count_gt_0.5",
            "frac_gt_0.6", "count_gt_0.6",
            "frac_gt_0.7", "count_gt_0.7",
            "frac_gt_0.8", "count_gt_0.8",
            "frac_gt_0.9", "count_gt_0.9"]
    return df[keep].copy()


def apply_rule(df, predicate, name):
    v = predicate(df).astype(bool)
    truth = (df["label"] == 1)
    correct = (v == truth).sum()
    return {
        "rule": name,
        "n": len(df),
        "correct": int(correct),
        "rate": float(correct / len(df)),
        "fp": int((v & ~truth).sum()),
        "fn": int((~v & truth).sum()),
    }, v.values


def define_rules():
    rules = []
    # Reference rules
    rules.append(("opt1_baseline",
                  lambda d: d["frac_gt_0.49"] > 0.5))
    rules.append(("opt2_stricter",
                  lambda d: d["frac_gt_0.6"] > 0.4))
    rules.append(("opt3_combined",
                  lambda d: (d["frac_gt_0.6"] > 0.4) & (d["count_gt_0.9"] >= 1)))

    # Proportional double-tail (matched to 32-frame window)
    # User's exact proposal: 1/32 above 0.9 AND 2/32 above 0.8
    rules.append(("DT_prop_1of32_2of32",
                  lambda d: (d["frac_gt_0.6"] > 0.4)
                           & (d["frac_gt_0.9"] >= 1/32)
                           & (d["frac_gt_0.8"] >= 2/32)))
    # Variants
    rules.append(("DT_prop_1of32_3of32",
                  lambda d: (d["frac_gt_0.6"] > 0.4)
                           & (d["frac_gt_0.9"] >= 1/32)
                           & (d["frac_gt_0.8"] >= 3/32)))
    rules.append(("DT_prop_1of32_4of32",
                  lambda d: (d["frac_gt_0.6"] > 0.4)
                           & (d["frac_gt_0.9"] >= 1/32)
                           & (d["frac_gt_0.8"] >= 4/32)))
    rules.append(("DT_prop_2of32_4of32",
                  lambda d: (d["frac_gt_0.6"] > 0.4)
                           & (d["frac_gt_0.9"] >= 2/32)
                           & (d["frac_gt_0.8"] >= 4/32)))
    rules.append(("DT_prop_2of32_5of32",
                  lambda d: (d["frac_gt_0.6"] > 0.4)
                           & (d["frac_gt_0.9"] >= 2/32)
                           & (d["frac_gt_0.8"] >= 5/32)))

    # Absolute (less interpretable across varying n but tests user's literal proposal)
    rules.append(("DT_abs_1and2",
                  lambda d: (d["frac_gt_0.6"] > 0.4)
                           & (d["count_gt_0.9"] >= 1)
                           & (d["count_gt_0.8"] >= 2)))
    rules.append(("DT_abs_1and3",
                  lambda d: (d["frac_gt_0.6"] > 0.4)
                           & (d["count_gt_0.9"] >= 1)
                           & (d["count_gt_0.8"] >= 3)))
    rules.append(("DT_abs_2and4",
                  lambda d: (d["frac_gt_0.6"] > 0.4)
                           & (d["count_gt_0.9"] >= 2)
                           & (d["count_gt_0.8"] >= 4)))

    # Just-tail variants (drop count>0.9 ≥ 1, replace with frac>0.9 fraction)
    rules.append(("ALT_frac_only_1.5pct_9_5pct_8",
                  lambda d: (d["frac_gt_0.6"] > 0.4)
                           & (d["frac_gt_0.9"] >= 0.015)
                           & (d["frac_gt_0.8"] >= 0.05)))
    rules.append(("ALT_frac_only_3pct_9_6pct_8",
                  lambda d: (d["frac_gt_0.6"] > 0.4)
                           & (d["frac_gt_0.9"] >= 0.03)
                           & (d["frac_gt_0.8"] >= 0.06)))
    rules.append(("ALT_frac_only_5pct_9_10pct_8",
                  lambda d: (d["frac_gt_0.6"] > 0.4)
                           & (d["frac_gt_0.9"] >= 0.05)
                           & (d["frac_gt_0.8"] >= 0.10)))
    return rules


def list_errors(df, verdict, kind="fp"):
    truth = (df["label"] == 1).values
    if kind == "fp":
        mask = verdict & ~truth
    else:
        mask = ~verdict & truth
    cols = ["base_identity", "n_frames",
            "frac_gt_0.6", "frac_gt_0.7", "frac_gt_0.8", "frac_gt_0.9",
            "count_gt_0.8", "count_gt_0.9"]
    return df.loc[mask, cols].copy()


def bootstrap_paired(df, vA, vB, n_iter=5000, seed=7777):
    rng = np.random.default_rng(seed)
    truth = (df["label"] == 1).values
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


def score_pool(df, label):
    rules = define_rules()
    results = []
    verdicts = {}
    for name, pred in rules:
        rec, v = apply_rule(df, pred, name)
        results.append(rec)
        verdicts[name] = v
    df_results = pd.DataFrame(results)
    print(f"\n=== {label} (n={len(df)}) ===")
    print(df_results.sort_values(["rate", "fp"], ascending=[False, True]).to_string(index=False))

    # Detailed view of top double-tail vs opt3
    print(f"\n--- Errors on opt3 ---")
    fp = list_errors(df, verdicts["opt3_combined"], "fp")
    fn = list_errors(df, verdicts["opt3_combined"], "fn")
    print(f"FPs ({len(fp)}):"); print(fp.to_string(index=False))
    print(f"FNs ({len(fn)}):"); print(fn.to_string(index=False))

    print(f"\n--- Errors on DT_prop_1of32_2of32 (user's exact proposal) ---")
    fp = list_errors(df, verdicts["DT_prop_1of32_2of32"], "fp")
    fn = list_errors(df, verdicts["DT_prop_1of32_2of32"], "fn")
    print(f"FPs ({len(fp)}):"); print(fp.to_string(index=False))
    print(f"FNs ({len(fn)}):"); print(fn.to_string(index=False))

    print(f"\n--- Bootstrap vs opt3 ---")
    v_opt3 = verdicts["opt3_combined"]
    for name in ["DT_prop_1of32_2of32", "DT_prop_1of32_3of32", "DT_prop_1of32_4of32",
                 "DT_prop_2of32_4of32", "DT_prop_2of32_5of32",
                 "DT_abs_1and2", "DT_abs_1and3", "DT_abs_2and4",
                 "ALT_frac_only_1.5pct_9_5pct_8", "ALT_frac_only_3pct_9_6pct_8",
                 "ALT_frac_only_5pct_9_10pct_8"]:
        v = verdicts[name]
        bs = bootstrap_paired(df, v_opt3, v)
        print(f"  {name:38s}: Δ={bs['delta_mean']:+.4f} [{bs['delta_p2.5']:+.4f},{bs['delta_p97.5']:+.4f}] P(Δ>0)={bs['p_gt_0']:.3f}")

    return df_results, verdicts


def main():
    # Pool 1: 71-identity derivation pool
    df71 = load_71_pool()
    df71_results, _ = score_pool(df71, "71-identity DERIVATION pool")

    # Pool 2: 47-identity independent training-eval pool
    df_te = load_training_eval_per_identity()
    df_te_results, _ = score_pool(df_te, "47-identity INDEPENDENT training-eval pool")

    # Save
    df71_results.to_csv(OUT_DIR / "double_tail_71pool_summary.csv", index=False)
    df_te_results.to_csv(OUT_DIR / "double_tail_trainevalpool_summary.csv", index=False)
    print("\nSaved: double_tail_71pool_summary.csv, double_tail_trainevalpool_summary.csv")


if __name__ == "__main__":
    main()
