"""
ULTRATHINK checkpoint search 2026-05-14.

Goal: Find a checkpoint where rules (opt3, conditional mean, single-tail-0.8) actually
create SEPARATION between the production threat (visomaster_enhanced_teams) and the
chronic FPs (PC_Generator__s22/s45, Q__s6, bla_bla_chow__s1).

On T5C step3500, the conditional-mean rule's threshold 0.75 fails because:
  - viso_enhanced_teams mean_above_0.5 = 0.713  (production threat, BELOW threshold → MISSED)
  - PC_Generator__s22       mean_above_0.5 = 0.727  (chronic FP, BELOW threshold → rescued)
  - The gap is +0.014 in the WRONG direction (FP > threat).

We need a checkpoint where viso_teams scores HIGHER conditional mean than PC_Generator.

Strategy:
1. Map every available checkpoint to its 6-suite reports (find them across all dirs).
2. For each checkpoint, compute per-identity features for the critical identities.
3. Compute separation = viso_teams.mean_gt_0.5 - PC_Generator__s22.mean_gt_0.5.
   Positive = useful separation. Negative = same problem as T5C.
4. Also test: max(viso_teams) - max(PC_Gen) (extreme-frame separation).
5. For checkpoints with promising separation, run scorecards on the full validation set.
"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis")
OUT_DIR = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/risky_remediation_2026-05-14/outputs")

SUITES = {
    "teams_real_all_dev":     0,
    "teams_real_all_lockbox": 0,
    "teams_fake_all_dev":     1,
    "teams_fake_all_lockbox": 1,
    "visomaster_enhanced_macro_dev": 1,
    "deeplive_enhanced_dev":  1,
}

# Critical identities we want to track
CRITICAL_IDS = [
    # Production threats (must be flagged as fake)
    "visomaster_enhanced_teams",   # ← THE production threat
    "visomaster_enhanced_raw",     # ← viso without Teams pipeline
    "dor_shkedi__s16",             # weak fake
    # Chronic FPs (should NOT be flagged)
    "PC_Generator__s22",
    "PC_Generator__s45",
    "Q__s6",
    "bla_bla_chow__s1",
    "Roy_D",                       # unrescuable but track
    # Reference well-caught fakes
    "deeplive_dor",
    "Cam_Test__s35",
]


def parse_base_identity(video_id):
    s = str(video_id)
    m = re.match(r"^(.*?)__seg_", s)
    if m:
        return m.group(1)
    m = re.match(r"^(.*?)__seq", s)
    if m:
        return m.group(1)
    return s


def build_checkpoint_map():
    """For each (checkpoint, suite) find the report file path."""
    all_reports = list(glob.glob(str(ROOT / "**" / "*_frames_report.csv"), recursive=True))
    ckpt_map = {}  # ckpt_name -> {suite: path}
    suite_re = "(" + "|".join(SUITES.keys()) + ")"
    pat = re.compile(rf"^{suite_re}_(.+)_frames_report\.csv$")
    for path in all_reports:
        name = Path(path).name
        m = pat.match(name)
        if not m:
            continue
        suite, ckpt = m.group(1), m.group(2)
        if ckpt not in ckpt_map:
            ckpt_map[ckpt] = {}
        # Prefer reports in cpu_diagnostics_2026-05-12_stage_a (most consistent labels)
        if suite in ckpt_map[ckpt]:
            old = ckpt_map[ckpt][suite]
            if "cpu_diagnostics_2026-05-12_stage_a" in path and "cpu_diagnostics_2026-05-12_stage_a" not in old:
                ckpt_map[ckpt][suite] = path
        else:
            ckpt_map[ckpt][suite] = path
    # Keep only checkpoints with all 6 suites
    return {c: s for c, s in ckpt_map.items() if len(s) == 6}


def load_checkpoint_frames(ckpt_paths):
    """Concatenate frames across all 6 suites for one checkpoint."""
    rows = []
    for suite, lbl in SUITES.items():
        df = pd.read_csv(ckpt_paths[suite])
        df["suite"] = suite
        df["label"] = lbl
        df["base_identity"] = df["video_id"].astype(str).map(parse_base_identity)
        rows.append(df[["suite", "label", "base_identity", "frame_prob"]])
    return pd.concat(rows, ignore_index=True)


def aggregate_per_identity(raw, min_frames=20):
    out = []
    for (suite, bid), g in raw.groupby(["suite", "base_identity"]):
        scores = g["frame_prob"].values
        if len(scores) < min_frames:
            continue
        rec = {
            "suite": suite, "base_identity": bid,
            "label": int(g["label"].iloc[0]),
            "n_frames": int(len(scores)),
            "mean": float(scores.mean()),
            "max": float(scores.max()),
        }
        for t in [0.49, 0.5, 0.6, 0.7, 0.8, 0.9]:
            rec[f"frac_gt_{t}"] = float((scores > t).mean())
            rec[f"count_gt_{t}"] = int((scores > t).sum())
        for t in [0.5, 0.6, 0.7]:
            sub = scores[scores > t]
            if len(sub) > 0:
                rec[f"mean_gt_{t}"] = float(sub.mean())
                rec[f"max_gt_{t}"]  = float(sub.max())
            else:
                rec[f"mean_gt_{t}"] = np.nan
                rec[f"max_gt_{t}"]  = np.nan
        out.append(rec)
    return pd.DataFrame(out)


def apply_rules(per_id):
    df = per_id.copy()
    df["opt1"] = (df["frac_gt_0.49"] > 0.5).astype(int)
    df["opt2"] = (df["frac_gt_0.6"]  > 0.4).astype(int)
    df["opt3"] = ((df["frac_gt_0.6"] > 0.4) & (df["count_gt_0.9"] >= 1)).astype(int)
    df["cm_75"] = ((df["frac_gt_0.6"] > 0.4) & (df["count_gt_0.9"] >= 1) & (df["mean_gt_0.5"] > 0.75)).astype(int)
    df["cm_72"] = ((df["frac_gt_0.6"] > 0.4) & (df["count_gt_0.9"] >= 1) & (df["mean_gt_0.5"] > 0.72)).astype(int)
    df["cm_only_75_no_extreme"] = ((df["frac_gt_0.6"] > 0.4) & (df["mean_gt_0.5"] > 0.75)).astype(int)
    return df


def score_all(per_id_with_rules):
    truth = per_id_with_rules["label"].values
    n = len(per_id_with_rules)
    out = {}
    for rule in ["opt1", "opt2", "opt3", "cm_75", "cm_72", "cm_only_75_no_extreme"]:
        v = per_id_with_rules[rule].values
        out[f"{rule}_rate"] = float((v == truth).mean())
        out[f"{rule}_fp"]   = int(((v == 1) & (truth == 0)).sum())
        out[f"{rule}_fn"]   = int(((v == 0) & (truth == 1)).sum())
    out["n_identities"] = n
    return out


def main():
    ckpt_map = build_checkpoint_map()
    print(f"Found {len(ckpt_map)} checkpoints with all 6 suite reports")
    for ckpt in sorted(ckpt_map.keys()):
        print(f"  {ckpt}")

    # Per-checkpoint analysis
    summary_rows = []
    critical_rows = []
    rule_score_rows = []

    for ckpt in sorted(ckpt_map.keys()):
        print(f"\n--- {ckpt} ---")
        try:
            raw = load_checkpoint_frames(ckpt_map[ckpt])
            per_id = aggregate_per_identity(raw, min_frames=20)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        # Critical identities
        crit = per_id[per_id["base_identity"].isin(CRITICAL_IDS)].copy()
        crit["ckpt"] = ckpt
        critical_rows.append(crit)

        # Separation metric: viso_enhanced_teams vs PC_Generator__s22
        viso = crit[crit["base_identity"] == "visomaster_enhanced_teams"]
        pc22 = crit[crit["base_identity"] == "PC_Generator__s22"]
        pc45 = crit[crit["base_identity"] == "PC_Generator__s45"]

        sep_mean = np.nan
        sep_mean_pc45 = np.nan
        viso_mean = np.nan
        pc22_mean = np.nan
        pc45_mean = np.nan
        if len(viso) > 0 and len(pc22) > 0:
            viso_mean = float(viso.iloc[0]["mean_gt_0.5"])
            pc22_mean = float(pc22.iloc[0]["mean_gt_0.5"])
            sep_mean = viso_mean - pc22_mean
        if len(viso) > 0 and len(pc45) > 0:
            pc45_mean = float(pc45.iloc[0]["mean_gt_0.5"])
            sep_mean_pc45 = float(viso.iloc[0]["mean_gt_0.5"]) - pc45_mean

        # Rule outcomes
        per_id_v = apply_rules(per_id)
        sc = score_all(per_id_v)
        row = {"ckpt": ckpt, "n_id": sc["n_identities"],
               "viso_mean_gt_0.5": viso_mean, "pc22_mean_gt_0.5": pc22_mean, "pc45_mean_gt_0.5": pc45_mean,
               "sep_viso_minus_pc22": sep_mean, "sep_viso_minus_pc45": sep_mean_pc45,
               **sc}
        summary_rows.append(row)

        # Print critical-id summary
        crit_focus = crit[crit["base_identity"].isin(
            ["visomaster_enhanced_teams", "visomaster_enhanced_raw",
             "PC_Generator__s22", "PC_Generator__s45", "Q__s6", "bla_bla_chow__s1", "Roy_D"]
        )][["base_identity", "label", "n_frames", "mean",
            "frac_gt_0.6", "frac_gt_0.7", "frac_gt_0.9", "count_gt_0.9",
            "mean_gt_0.5", "mean_gt_0.6"]]
        print(crit_focus.to_string(index=False))
        print(f"  Separation (viso_teams - PC_22) on mean_gt_0.5: {sep_mean:+.4f}" if not np.isnan(sep_mean) else "")
        print(f"  opt3 rate: {sc['opt3_rate']:.4f} (FP={sc['opt3_fp']}, FN={sc['opt3_fn']})")
        print(f"  CM75 rate: {sc['cm_75_rate']:.4f} (FP={sc['cm_75_fp']}, FN={sc['cm_75_fn']})")

    # Save
    df_summary = pd.DataFrame(summary_rows).sort_values("sep_viso_minus_pc22", ascending=False)
    df_summary.to_csv(OUT_DIR / "checkpoint_search_summary.csv", index=False)
    df_critical = pd.concat(critical_rows, ignore_index=True) if critical_rows else pd.DataFrame()
    df_critical.to_csv(OUT_DIR / "checkpoint_search_critical_ids.csv", index=False)

    # ======= REPORTS =======
    print("\n\n=== TOP 15 CHECKPOINTS BY SEPARATION (viso_teams.mean_gt_0.5 - PC_Generator__s22.mean_gt_0.5) ===")
    print("Positive = viso_teams scores HIGHER than PC_Gen → rule-friendly")
    cols = ["ckpt", "viso_mean_gt_0.5", "pc22_mean_gt_0.5", "sep_viso_minus_pc22",
            "opt3_rate", "opt3_fp", "opt3_fn", "cm_75_rate", "cm_75_fp", "cm_75_fn"]
    print(df_summary[cols].head(15).to_string(index=False))

    print("\n\n=== ALL CHECKPOINTS RANKED BY opt3_rate ===")
    print(df_summary.sort_values("opt3_rate", ascending=False)[
        ["ckpt", "n_id", "opt3_rate", "opt3_fp", "opt3_fn", "cm_75_rate", "cm_75_fp", "cm_75_fn",
         "sep_viso_minus_pc22"]
    ].to_string(index=False))

    print("\n\n=== ALL CHECKPOINTS RANKED BY cm_75_rate ===")
    print(df_summary.sort_values("cm_75_rate", ascending=False)[
        ["ckpt", "n_id", "opt3_rate", "opt3_fp", "opt3_fn", "cm_75_rate", "cm_75_fp", "cm_75_fn",
         "sep_viso_minus_pc22"]
    ].to_string(index=False))

    # Best by composite: (opt3 OR cm_75) rate × (low fp)
    print("\n\n=== CHECKPOINTS WHERE cm_75 BEATS opt3 (FP-reduction lever works) ===")
    best_cm = df_summary[df_summary["cm_75_rate"] > df_summary["opt3_rate"]].sort_values("cm_75_rate", ascending=False)
    print(best_cm[["ckpt", "n_id", "opt3_rate", "opt3_fp", "cm_75_rate", "cm_75_fp",
                   "viso_mean_gt_0.5", "pc22_mean_gt_0.5", "sep_viso_minus_pc22"]].to_string(index=False))


if __name__ == "__main__":
    main()
