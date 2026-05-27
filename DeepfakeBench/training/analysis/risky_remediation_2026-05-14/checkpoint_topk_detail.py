"""
Drill down into the top candidate checkpoints from the search.

Goals:
- Show full per-identity profile (incl. critical IDs) for top 5 ckpts
- For each, list the actual FPs and FNs under opt3/cm75
- Test ensemble: T5C step3500 AND deeplive_corr (both must flag) for double-confirmation
- Compare to T5C step3500 baseline
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

TOP_CKPTS = [
    "t5c_periodic_step3500",          # baseline
    "deeplive_corr_top_n_step4800",   # 0 FP under opt3 winner
    "viso_corr_top_n_step600",        # 0 FP under CM75 winner
    "e2b_top_n_step3200",             # strongest separation
    "deeplive_corr_top_n_step1800",   # also top
    "pa_top_n_step3800",              # check
    "p1_pairrank_top_n_step6000",     # 0 FN under opt3
    "p1_pairrank_top_n_step6750",     # 0 FN under opt3
]


def parse_base_identity(video_id):
    s = str(video_id)
    m = re.match(r"^(.*?)__seg_", s)
    if m: return m.group(1)
    m = re.match(r"^(.*?)__seq", s)
    if m: return m.group(1)
    return s


def build_checkpoint_map():
    all_reports = list(glob.glob(str(ROOT / "**" / "*_frames_report.csv"), recursive=True))
    ckpt_map = {}
    suite_re = "(" + "|".join(SUITES.keys()) + ")"
    pat = re.compile(rf"^{suite_re}_(.+)_frames_report\.csv$")
    for path in all_reports:
        name = Path(path).name
        m = pat.match(name)
        if not m: continue
        suite, ckpt = m.group(1), m.group(2)
        if ckpt not in ckpt_map:
            ckpt_map[ckpt] = {}
        if suite in ckpt_map[ckpt]:
            old = ckpt_map[ckpt][suite]
            if "cpu_diagnostics_2026-05-12_stage_a" in path and "cpu_diagnostics_2026-05-12_stage_a" not in old:
                ckpt_map[ckpt][suite] = path
        else:
            ckpt_map[ckpt][suite] = path
    return {c: s for c, s in ckpt_map.items() if len(s) == 6}


def load_ckpt(ckpt_paths):
    rows = []
    for suite, lbl in SUITES.items():
        df = pd.read_csv(ckpt_paths[suite])
        df["suite"] = suite
        df["label"] = lbl
        df["base_identity"] = df["video_id"].astype(str).map(parse_base_identity)
        rows.append(df[["suite", "label", "base_identity", "frame_prob"]])
    return pd.concat(rows, ignore_index=True)


def agg_per_identity(raw, min_frames=20):
    out = []
    for (suite, bid), g in raw.groupby(["suite", "base_identity"]):
        scores = g["frame_prob"].values
        if len(scores) < min_frames: continue
        rec = {"suite": suite, "base_identity": bid, "label": int(g["label"].iloc[0]),
               "n": len(scores), "mean": float(scores.mean()), "max": float(scores.max())}
        for t in [0.49, 0.5, 0.6, 0.7, 0.8, 0.9]:
            rec[f"f_{t}"] = float((scores > t).mean())
            rec[f"c_{t}"] = int((scores > t).sum())
        for t in [0.5, 0.6]:
            sub = scores[scores > t]
            rec[f"cmean_{t}"] = float(sub.mean()) if len(sub) > 0 else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def rules(df):
    df = df.copy()
    df["opt1"] = (df["f_0.49"] > 0.5).astype(int)
    df["opt2"] = (df["f_0.6"] > 0.4).astype(int)
    df["opt3"] = ((df["f_0.6"] > 0.4) & (df["c_0.9"] >= 1)).astype(int)
    df["cm75"] = ((df["f_0.6"] > 0.4) & (df["c_0.9"] >= 1) & (df["cmean_0.5"] > 0.75)).astype(int)
    df["cm72"] = ((df["f_0.6"] > 0.4) & (df["c_0.9"] >= 1) & (df["cmean_0.5"] > 0.72)).astype(int)
    return df


def score(df, rule):
    truth = df["label"].values
    v = df[rule].values
    return {
        "rule": rule, "rate": float((v == truth).mean()),
        "fp": int(((v == 1) & (truth == 0)).sum()),
        "fn": int(((v == 0) & (truth == 1)).sum()),
    }


def main():
    ckpt_map = build_checkpoint_map()

    # Per-identity tables for each top ckpt
    print("="*80)
    print("PER-IDENTITY DEEP-DIVE for top candidates")
    print("="*80)
    all_perid = {}
    for ckpt in TOP_CKPTS:
        if ckpt not in ckpt_map:
            print(f"\n[skip] {ckpt} not found")
            continue
        raw = load_ckpt(ckpt_map[ckpt])
        per_id = agg_per_identity(raw)
        per_id_r = rules(per_id)
        all_perid[ckpt] = per_id_r

        # Show critical identities
        print(f"\n=== {ckpt} ===")
        crit = per_id_r[per_id_r["base_identity"].isin([
            "visomaster_enhanced_teams", "visomaster_enhanced_raw",
            "dor_shkedi__s16", "PC_Generator__s22", "PC_Generator__s45",
            "Q__s6", "bla_bla_chow__s1", "Roy_D",
            "deeplive_dor", "Cam_Test__s35",
        ])]
        # de-dup by base_identity (some appear in 2 suites)
        crit = crit.drop_duplicates("base_identity", keep="first")
        cols = ["base_identity", "label", "n", "mean", "f_0.6", "f_0.7", "f_0.9", "c_0.9", "cmean_0.5", "cmean_0.6",
                "opt3", "cm75"]
        print(crit[cols].sort_values(["label", "cmean_0.5"], ascending=[True, False]).to_string(index=False))

        for r in ["opt1", "opt2", "opt3", "cm75", "cm72"]:
            s = score(per_id_r, r)
            print(f"  {r}: rate={s['rate']:.4f}  FP={s['fp']}  FN={s['fn']}")

        # List errors
        truth = per_id_r["label"].values
        for r in ["opt3", "cm75"]:
            v = per_id_r[r].values
            fps = per_id_r[(v == 1) & (truth == 0)][["suite", "base_identity", "mean", "cmean_0.5", "c_0.9"]]
            fns = per_id_r[(v == 0) & (truth == 1)][["suite", "base_identity", "mean", "cmean_0.5", "c_0.9", "f_0.6"]]
            print(f"\n  {r} FPs ({len(fps)}):")
            print(fps.to_string(index=False) if len(fps) else "    none")
            print(f"  {r} FNs ({len(fns)}):")
            print(fns.to_string(index=False) if len(fns) else "    none")

    # Ensemble experiments
    print("\n" + "="*80)
    print("ENSEMBLES: T5C step3500 AND <other-ckpt>")
    print("="*80)
    base_ckpt = "t5c_periodic_step3500"
    if base_ckpt not in all_perid:
        return
    base = all_perid[base_ckpt].set_index(["suite", "base_identity"])

    for partner in TOP_CKPTS:
        if partner == base_ckpt or partner not in all_perid:
            continue
        ptn = all_perid[partner].set_index(["suite", "base_identity"])
        # Inner join — identities present in both
        common = base.index.intersection(ptn.index)
        if len(common) == 0:
            continue
        base_c = base.loc[common]
        ptn_c = ptn.loc[common]
        truth = base_c["label"].values
        # AND rule: both must flag
        for rule in ["opt3", "cm75"]:
            v_and = (base_c[rule].values == 1) & (ptn_c[rule].values == 1)
            v_or  = (base_c[rule].values == 1) | (ptn_c[rule].values == 1)
            for name, v in [("AND", v_and), ("OR", v_or)]:
                rate = float((v == truth).mean())
                fp = int(((v == 1) & (truth == 0)).sum())
                fn = int(((v == 0) & (truth == 1)).sum())
                print(f"  T5C+{partner} ({rule}, {name}): n={len(common)} rate={rate:.4f}  FP={fp}  FN={fn}")


if __name__ == "__main__":
    main()
