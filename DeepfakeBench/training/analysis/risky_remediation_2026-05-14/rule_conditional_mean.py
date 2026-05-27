"""
Phase 1-5: Single-tail-0.8 and conditional-mean rules.

Goal: find a rule that beats Option 3 on both 71-pool AND independent 47-pool.

Approach:
  Phase 2: count>0.8 ≥ K sweeps, frac>0.8 ≥ p sweeps
  Phase 3: mean_given_gt_T > θ rules + combinations with extreme-tail check
  Phase 4: descriptive separation of conditional-mean distribution
           between weak fakes, chronic FPs, bla-bla-chow type FPs
  Phase 5: bootstrap CIs for top candidates on BOTH pools (the cross-check)
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
SUITES_TRAINEVAL = {
    "teams_real_all_dev":     (0, "teams_real_all_dev_t5c_periodic_step3500_frames_report.csv"),
    "teams_real_all_lockbox": (0, "teams_real_all_lockbox_t5c_periodic_step3500_frames_report.csv"),
    "teams_fake_all_dev":     (1, "teams_fake_all_dev_t5c_periodic_step3500_frames_report.csv"),
    "teams_fake_all_lockbox": (1, "teams_fake_all_lockbox_t5c_periodic_step3500_frames_report.csv"),
    "visomaster_enhanced":    (1, "visomaster_enhanced_macro_dev_t5c_periodic_step3500_frames_report.csv"),
    "deeplive_enhanced":      (1, "deeplive_enhanced_dev_t5c_periodic_step3500_frames_report.csv"),
}

P71_RAW = ROOT / "outputs" / "g2_pass_pool.csv"
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


def enrich_features(scores: np.ndarray) -> dict:
    out = {}
    out["n_frames"] = int(len(scores))
    out["mean"] = float(scores.mean())
    out["median"] = float(np.median(scores))
    out["max"] = float(scores.max())
    for t in [0.49, 0.5, 0.6, 0.7, 0.8, 0.9]:
        out[f"frac_gt_{t}"] = float((scores > t).mean())
        out[f"count_gt_{t}"] = int((scores > t).sum())
    # Conditional means above threshold
    for t in [0.5, 0.6, 0.7, 0.8]:
        sub = scores[scores > t]
        if len(sub) > 0:
            out[f"mean_gt_{t}"] = float(sub.mean())
            out[f"std_gt_{t}"]  = float(sub.std()) if len(sub) > 1 else 0.0
            out[f"max_gt_{t}"]  = float(sub.max())
            out[f"p75_gt_{t}"]  = float(np.percentile(sub, 75))
        else:
            out[f"mean_gt_{t}"] = np.nan
            out[f"std_gt_{t}"]  = np.nan
            out[f"max_gt_{t}"]  = np.nan
            out[f"p75_gt_{t}"]  = np.nan
    return out


def load_71_pool() -> pd.DataFrame:
    """Re-derive per-identity from g2_pass_pool (has T5C_orig scores) so we get cond features."""
    df = pd.read_csv(P71_RAW)
    # Re-derive pool & label & identity
    rows = []
    for (suite, label), g in df.groupby(["suite", "label"]):
        for bid, gg in g.groupby("base_identity"):
            scores = gg["T5C_orig"].values
            if len(scores) < MIN_FRAMES:
                continue
            rec = {"pool": f"71_{suite}", "suite": suite, "base_identity": bid, "label": int(label)}
            rec.update(enrich_features(scores))
            rows.append(rec)
    return pd.DataFrame(rows)


def load_traineval_pool() -> pd.DataFrame:
    rows = []
    for suite, (lbl, fname) in SUITES_TRAINEVAL.items():
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
        rec = {"pool": f"te_{suite}", "suite": suite, "base_identity": bid, "label": int(g["label"].iloc[0])}
        rec.update(enrich_features(scores))
        out.append(rec)
    return pd.DataFrame(out)


def apply_rule(df, predicate, name):
    v = predicate(df).fillna(False).astype(bool)
    truth = (df["label"] == 1)
    return {
        "rule": name,
        "n": len(df),
        "correct": int((v == truth).sum()),
        "rate": float((v == truth).mean()),
        "fp": int((v & ~truth).sum()),
        "fn": int((~v & truth).sum()),
    }, v.values


def define_rules():
    rules = []
    # Baselines
    rules.append(("opt1_baseline",
                  lambda d: d["frac_gt_0.49"] > 0.5))
    rules.append(("opt2_stricter",
                  lambda d: d["frac_gt_0.6"] > 0.4))
    rules.append(("opt3_combined",
                  lambda d: (d["frac_gt_0.6"] > 0.4) & (d["count_gt_0.9"] >= 1)))

    # Phase 2: single-tail-0.8 (count) variants
    for K in [1, 2, 3, 5, 7, 10]:
        rules.append((f"T8_count_K{K}",
                      lambda d, K=K: (d["frac_gt_0.6"] > 0.4) & (d["count_gt_0.8"] >= K)))
    # Phase 2: frac-based 0.8
    for p in [0.05, 0.10, 0.15, 0.20, 0.25]:
        rules.append((f"T8_frac_p{int(p*100):02d}",
                      lambda d, p=p: (d["frac_gt_0.6"] > 0.4) & (d["frac_gt_0.8"] >= p)))

    # Phase 3: conditional-mean rules (no extreme requirement)
    for t in [0.5, 0.6, 0.7]:
        for theta in [0.70, 0.75, 0.80, 0.85]:
            rules.append((f"CM_t{t:.1f}_th{theta:.2f}_no_extreme",
                          lambda d, t=t, theta=theta:
                            (d["frac_gt_0.6"] > 0.4) & (d[f"mean_gt_{t}"] > theta)))
    # Phase 3: with extreme requirement
    for t in [0.5, 0.6, 0.7]:
        for theta in [0.70, 0.75, 0.80, 0.85]:
            rules.append((f"CM_t{t:.1f}_th{theta:.2f}_with_extreme",
                          lambda d, t=t, theta=theta:
                            (d["frac_gt_0.6"] > 0.4) & (d["count_gt_0.9"] >= 1) & (d[f"mean_gt_{t}"] > theta)))

    # Phase 3b: max-given-gt-T rules (heavy upper tail check)
    for t in [0.6, 0.7]:
        for theta in [0.85, 0.90, 0.95]:
            rules.append((f"MX_t{t:.1f}_th{theta:.2f}",
                          lambda d, t=t, theta=theta:
                            (d["frac_gt_0.6"] > 0.4) & (d[f"max_gt_{t}"] > theta)))
    return rules


def bootstrap_paired(df, vA, vB, n_iter=3000, seed=11):
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
        "p_gt_0": float((deltas > 0).mean()),
        "p25": float(np.percentile(deltas, 2.5)),
        "p975": float(np.percentile(deltas, 97.5)),
    }


def score_pool(df, label):
    rules = define_rules()
    results = []
    verdicts = {}
    for name, pred in rules:
        rec, v = apply_rule(df, pred, name)
        results.append(rec)
        verdicts[name] = v
    df_r = pd.DataFrame(results)
    df_r = df_r.sort_values(["rate", "fp"], ascending=[False, True]).reset_index(drop=True)
    print(f"\n=== {label} (n={len(df)}) — top 20 rules ===")
    print(df_r.head(20).to_string(index=False))
    return df_r, verdicts


def main():
    print("Loading 71-pool from g2_pass_pool.csv ...")
    df71 = load_71_pool()
    print(f"71-pool identities (≥{MIN_FRAMES} frames): {len(df71)}")
    print(df71.groupby(["suite", "label"]).size().to_string())

    print("\nLoading independent training-eval pool ...")
    df_te = load_traineval_pool()
    print(f"Training-eval identities (≥{MIN_FRAMES} frames): {len(df_te)}")
    print(df_te.groupby(["suite", "label"]).size().to_string())

    # Save enriched feature tables
    df71.to_csv(OUT_DIR / "cm_features_71pool.csv", index=False)
    df_te.to_csv(OUT_DIR / "cm_features_trainevalpool.csv", index=False)

    # ====== Phase 4: Descriptive separation =======
    print("\n=== PHASE 4: conditional-mean by label, by pool ===")
    for poolname, df in [("71-pool", df71), ("train-eval", df_te)]:
        print(f"\n--- {poolname} mean_gt_0.6 (conditional mean given above 0.6) ---")
        # Per-label aggregates
        for label_val, grp in df.groupby("label"):
            vals = grp["mean_gt_0.6"].dropna().values
            print(f"  label={label_val} (n={len(vals)}): "
                  f"mean={vals.mean():.3f} median={np.median(vals):.3f} "
                  f"q25={np.percentile(vals,25):.3f} q75={np.percentile(vals,75):.3f}")

    # Look at specific identities to understand the conditional mean structure
    print("\n=== Specific identities: conditional features ===")
    print(f"\n--- 71-pool, focus on FPs (real, opt3-positive) and weak fakes ---")
    focus_71 = df71[df71["base_identity"].isin([
        "Roy_D", "bla_bla_chow__s1", "dor_shkedi", "dor_morning",
        "dor_fake_simswap", "dor_fake_inswapper_128res_gpen512",
        "dor_fake_inswapper_128res_gpen1024", "dor_fake_ghostface_v2", "dor_fake_ghostface_v3",
    ])]
    cols = ["pool", "base_identity", "label", "n_frames", "mean",
            "frac_gt_0.6", "frac_gt_0.8", "frac_gt_0.9", "count_gt_0.9",
            "mean_gt_0.5", "mean_gt_0.6", "mean_gt_0.7",
            "max_gt_0.6"]
    print(focus_71[cols].sort_values(["label", "mean_gt_0.6"]).to_string(index=False))

    print(f"\n--- Train-eval pool, focus on chronic FPs and weak fakes ---")
    focus_te = df_te[df_te["base_identity"].isin([
        "Roy_D", "PC_Generator__s22", "PC_Generator__s45", "Q__s6",
        "bla_bla_chow__s1",
    ])]
    print(focus_te[cols].to_string(index=False))

    # Also show ALL fake identities so we can see the conditional mean for weak fakes in train-eval
    print(f"\n--- Train-eval pool: ALL fake identities sorted by mean_gt_0.6 ---")
    print(df_te[df_te["label"] == 1].sort_values("mean_gt_0.6")[cols].to_string(index=False))

    # ====== Phase 2+3: rule sweeps ======
    df71_r, v71 = score_pool(df71, "71-POOL")
    df_te_r, v_te = score_pool(df_te, "TRAIN-EVAL POOL")

    # ====== Phase 5: cross-validation of TOP candidates ======
    # A top candidate must appear in top 10 on BOTH pools and beat opt3 on at least one
    print("\n=== PHASE 5: cross-pool consistency check ===")
    top71 = set(df71_r.head(20)["rule"])
    topte = set(df_te_r.head(20)["rule"])
    cross = top71 & topte
    # Rank on each pool
    rank71 = {r: i for i, r in enumerate(df71_r["rule"])}
    rankte = {r: i for i, r in enumerate(df_te_r["rule"])}
    rate71 = dict(zip(df71_r["rule"], df71_r["rate"]))
    ratete = dict(zip(df_te_r["rule"], df_te_r["rate"]))
    opt3_rate_71 = rate71["opt3_combined"]
    opt3_rate_te = ratete["opt3_combined"]
    print(f"opt3 rate 71-pool: {opt3_rate_71:.4f}, train-eval: {opt3_rate_te:.4f}")

    candidates = []
    for rule_name in cross:
        r71 = rate71[rule_name]
        rte = ratete[rule_name]
        beats71 = r71 > opt3_rate_71
        beats_te = rte > opt3_rate_te
        ties71 = abs(r71 - opt3_rate_71) < 1e-9
        ties_te = abs(rte - opt3_rate_te) < 1e-9
        if (beats71 or ties71) and (beats_te or ties_te):
            if beats71 or beats_te:  # strictly better on at least one
                candidates.append({"rule": rule_name, "rate_71": r71, "rate_te": rte,
                                   "rank_71": rank71[rule_name], "rank_te": rankte[rule_name]})
    if candidates:
        candidates_df = pd.DataFrame(candidates).sort_values(["rate_te", "rate_71"], ascending=[False, False])
    else:
        candidates_df = pd.DataFrame()
    print(f"\nRules that match/beat opt3 on BOTH pools and strictly beat on at least one:")
    print(candidates_df.to_string(index=False) if len(candidates_df) else "  (none)")

    # Also show all rules' performance on BOTH pools (cross-pool table)
    print("\n=== ALL RULES ACROSS BOTH POOLS (rate_71 vs rate_te, FP_71 vs FP_te) ===")
    df71_lookup = df71_r.set_index("rule")
    df_te_lookup = df_te_r.set_index("rule")
    cross_rows = []
    for rule_name in set(df71_lookup.index) & set(df_te_lookup.index):
        r71 = df71_lookup.loc[rule_name]
        rte = df_te_lookup.loc[rule_name]
        cross_rows.append({
            "rule": rule_name,
            "rate_71": r71["rate"], "fp_71": r71["fp"], "fn_71": r71["fn"],
            "rate_te": rte["rate"], "fp_te": rte["fp"], "fn_te": rte["fn"],
        })
    cross_df = pd.DataFrame(cross_rows)
    cross_df["min_rate"] = cross_df[["rate_71", "rate_te"]].min(axis=1)
    cross_df["fp_total"] = cross_df["fp_71"] + cross_df["fp_te"]
    cross_df = cross_df.sort_values(["min_rate", "fp_total"], ascending=[False, True])
    print(cross_df.head(30).to_string(index=False))
    cross_df.to_csv(OUT_DIR / "cm_cross_pool_summary.csv", index=False)

    if len(candidates_df) > 0:
        print("\n=== BOOTSTRAP for top candidates vs opt3 (on each pool) ===")
        for rule_name in candidates_df["rule"].head(8):
            bs71 = bootstrap_paired(df71, v71["opt3_combined"], v71[rule_name])
            bste = bootstrap_paired(df_te, v_te["opt3_combined"], v_te[rule_name])
            print(f"{rule_name}:")
            print(f"  71-pool:    Δ={bs71['delta_mean']:+.4f} [{bs71['p25']:+.4f},{bs71['p975']:+.4f}] P(Δ>0)={bs71['p_gt_0']:.3f}")
            print(f"  train-eval: Δ={bste['delta_mean']:+.4f} [{bste['p25']:+.4f},{bste['p975']:+.4f}] P(Δ>0)={bste['p_gt_0']:.3f}")

    # Save
    df71_r.to_csv(OUT_DIR / "cm_rules_71pool_summary.csv", index=False)
    df_te_r.to_csv(OUT_DIR / "cm_rules_trainevalpool_summary.csv", index=False)


if __name__ == "__main__":
    main()
