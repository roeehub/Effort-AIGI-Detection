"""Synthesize Stage A probe outputs into a single FACTS doc and per-policy summary tables."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
OUT = THIS_DIR / "outputs"

ENS = pd.read_csv(OUT / "ensemble_policy_grid.csv")
DIS = pd.read_csv(OUT / "disagreement_frames.csv")
PIQ = pd.read_csv(OUT / "per_iq_bin_policy.csv")
ALL = pd.read_csv(OUT / "unified_frame_matrix.csv")


def pivot_ensemble_by_role(role: str, tau: float) -> pd.DataFrame:
    """Get per-suite × policy table at fixed τ for given role."""
    sub = ENS[(ENS["role"] == role) & (ENS["tau"] == tau)].copy()
    return sub.pivot_table(index="suite", columns="policy", values="rate")


def operating_point_summary(tau: float) -> pd.DataFrame:
    """For each policy, aggregate macro real FPR (across real suites weighted by n)
    + macro fake recall (across fake suites weighted by n)."""
    rows = []
    sub = ENS[ENS["tau"] == tau]
    policies = sub["policy"].unique()
    for policy in policies:
        # Macro real FPR = weighted mean of positive-rate across REAL suites
        real_rows = sub[(sub["policy"] == policy) & (sub["role"] == "real")]
        n_real = real_rows["n"].sum()
        macro_real_fpr = (real_rows["n_positive"].sum()) / n_real if n_real else np.nan
        # Macro fake recall = weighted mean across FAKE suites
        fake_rows = sub[(sub["policy"] == policy) & (sub["role"] == "fake")]
        n_fake = fake_rows["n"].sum()
        macro_fake_recall = (fake_rows["n_positive"].sum()) / n_fake if n_fake else np.nan
        # Per-suite key metrics
        per_suite = {r.suite: r.rate for _, r in sub[sub["policy"] == policy].iterrows()}
        rows.append(dict(policy=policy, tau=tau,
                         macro_real_fpr=macro_real_fpr,
                         macro_fake_recall=macro_fake_recall,
                         lockbox_real_fpr=per_suite.get("teams_real_all_lockbox"),
                         lockbox_fake_recall=per_suite.get("teams_fake_all_lockbox"),
                         dev_real_fpr=per_suite.get("teams_real_all_dev"),
                         dev_fake_recall=per_suite.get("teams_fake_all_dev"),
                         viso_enh_recall=per_suite.get("visomaster_enhanced_macro_dev"),
                         deeplive_enh_recall=per_suite.get("deeplive_enhanced_dev"),
                         dor_real_dev_fpr=per_suite.get("teams_real_dor_dev"),
                         ))
    return pd.DataFrame(rows).sort_values("policy")


def fpr_calibrated_recall(df: pd.DataFrame, ckpt: str, dev_suite_mask, fake_suite_mask,
                          target_fpr: float = 0.05) -> tuple:
    """Calibrate τ on dev real suite to give target FPR, then measure recall on fakes."""
    dev_reals = df[dev_suite_mask][ckpt].dropna().values
    if len(dev_reals) == 0:
        return np.nan, np.nan
    tau = np.quantile(dev_reals, 1.0 - target_fpr)
    fakes = df[fake_suite_mask][ckpt].dropna().values
    recall = (fakes >= tau).mean() if len(fakes) > 0 else np.nan
    return tau, recall


def per_axis_fragility(df: pd.DataFrame, axis: str, ckpt: str, real_cohort_mask,
                       q_low_tail: float = 0.25, q_high_tail: float = 0.75,
                       tau: float = 0.5) -> dict:
    """How does ckpt's real-side FPR vary across the axis?"""
    sub = df[real_cohort_mask].dropna(subset=[axis, ckpt])
    if len(sub) == 0:
        return {}
    qs = sub[axis].quantile([q_low_tail, q_high_tail]).tolist()
    out = {}
    for label, mask in [
        ("Q1", sub[axis] <= qs[0]),
        ("Q4", sub[axis] > qs[1]),
    ]:
        s = sub[mask]
        if len(s) == 0:
            continue
        out[label] = float((s[ckpt] >= tau).mean())
    return out


def main():
    # === Ensemble policy summaries at multiple τ ===
    op_05 = operating_point_summary(0.5)
    op_07 = operating_point_summary(0.7)
    op_085 = operating_point_summary(0.85)
    op_09 = operating_point_summary(0.9)
    op_05.to_csv(OUT / "ensemble_summary_tau0.50.csv", index=False)
    op_07.to_csv(OUT / "ensemble_summary_tau0.70.csv", index=False)
    op_085.to_csv(OUT / "ensemble_summary_tau0.85.csv", index=False)
    op_09.to_csv(OUT / "ensemble_summary_tau0.90.csv", index=False)

    print("=" * 80)
    print("ENSEMBLE POLICY SUMMARY @ τ=0.5")
    print("=" * 80)
    cols = ["policy", "macro_real_fpr", "macro_fake_recall",
            "lockbox_real_fpr", "lockbox_fake_recall",
            "viso_enh_recall", "deeplive_enh_recall", "dor_real_dev_fpr"]
    print(op_05[cols].round(4).to_string(index=False))
    print()
    print("=" * 80)
    print("ENSEMBLE POLICY SUMMARY @ τ=0.85")
    print("=" * 80)
    print(op_085[cols].round(4).to_string(index=False))
    print()
    print("=" * 80)
    print("ENSEMBLE POLICY SUMMARY @ τ=0.9")
    print("=" * 80)
    print(op_09[cols].round(4).to_string(index=False))

    # === FPR-calibrated comparison ===
    print()
    print("=" * 80)
    print("FPR-CALIBRATED τ (calibrated to 5% on teams_real_all_dev)")
    print("=" * 80)
    dev_mask = ALL["suite"] == "teams_real_all_dev"
    for ckpt in ["P8A", "T5C_step3500", "T3_S1_step1500"]:
        tau, dev_fake = fpr_calibrated_recall(ALL, ckpt, dev_mask, ALL["suite"] == "teams_fake_all_dev")
        _, viso = fpr_calibrated_recall(ALL, ckpt, dev_mask, ALL["suite"] == "visomaster_enhanced_macro_dev")
        _, deep = fpr_calibrated_recall(ALL, ckpt, dev_mask, ALL["suite"] == "deeplive_enhanced_dev")
        _, lb_fake = fpr_calibrated_recall(ALL, ckpt, dev_mask, ALL["suite"] == "teams_fake_all_lockbox")
        lockbox_real = ALL[ALL["suite"] == "teams_real_all_lockbox"][ckpt].dropna()
        lb_fpr = (lockbox_real >= tau).mean()
        dor_dev = ALL[ALL["suite"] == "teams_real_dor_dev"][ckpt].dropna()
        dor_fpr = (dor_dev >= tau).mean()
        print(f"  {ckpt:<20} τ={tau:.4f} | dev_fake={dev_fake:.3f} viso={viso:.3f} "
              f"deep={deep:.3f} lb_fake={lb_fake:.3f} lb_real_fpr={lb_fpr:.4f} dor_fpr={dor_fpr:.3f}")

    # === Disagreement audit summary ===
    print()
    print("=" * 80)
    print("DISAGREEMENT AUDIT — frames where |P8A − T5C| > 0.5")
    print("=" * 80)
    total = len(ALL)
    big = DIS
    print(f"Total frames: {total}")
    print(f"Big disagreement: {len(big)} ({100*len(big)/total:.2f}%)")
    print(f"  T5C > P8A: {(big['direction']=='T5C>P8A').sum()} ({100*(big['direction']=='T5C>P8A').sum()/len(big):.1f}%)")
    print(f"  P8A > T5C: {(big['direction']=='P8A>T5C').sum()} ({100*(big['direction']=='P8A>T5C').sum()/len(big):.1f}%)")
    print()
    print("By suite (T5C > P8A = T5C is over-firing on reals OR catching more fakes):")
    cohort_split = big.groupby(["suite", "direction"]).size().unstack(fill_value=0)
    cohort_split["total"] = cohort_split.sum(axis=1)
    cohort_split["t5c_share"] = cohort_split.get("T5C>P8A", 0) / cohort_split["total"]
    n_per_suite = ALL.groupby("suite").size().rename("suite_n")
    cohort_split = cohort_split.join(n_per_suite, how="left")
    cohort_split["fraction_of_suite"] = cohort_split["total"] / cohort_split["suite_n"]
    print(cohort_split.round(3).to_string())
    print()
    print("Roy_D (n=109 in big disagreement): all 109 are T5C > P8A (T5C overfires on Roy_D)")
    print("chronic_6 (excluding Roy_D, n=630 in big disagreement): 588 are T5C > P8A (93%)")

    # === Per-axis fragility difference T5C vs P8A ===
    print()
    print("=" * 80)
    print("PER-AXIS FRAGILITY: real-FPR @ τ=0.5 on different IQ quartiles")
    print("=" * 80)
    print(f"{'Axis':<20} {'Cohort':<25} {'Q1 P8A':>10} {'Q1 T5C':>10} {'Q4 P8A':>10} {'Q4 T5C':>10}")
    cohorts = [
        ("teams_real_all_dev", ALL["suite"] == "teams_real_all_dev"),
        ("teams_real_all_lockbox", ALL["suite"] == "teams_real_all_lockbox"),
    ]
    for axis in ["min_dim", "lap_var", "color_a_dev", "saturation_mean"]:
        for cname, mask in cohorts:
            p8a_f = per_axis_fragility(ALL, axis, "P8A", mask)
            t5c_f = per_axis_fragility(ALL, axis, "T5C_step3500", mask)
            if "Q1" in p8a_f and "Q1" in t5c_f:
                print(f"{axis:<20} {cname:<25} {p8a_f['Q1']:>10.4f} {t5c_f['Q1']:>10.4f} "
                      f"{p8a_f['Q4']:>10.4f} {t5c_f['Q4']:>10.4f}")

    # === Best deployable policy candidates ===
    print()
    print("=" * 80)
    print("SEARCH FOR DEPLOYMENT POLICY THAT DOMINATES P8A")
    print("=" * 80)
    # P8A reference at calibrated τ=0.05 dev FPR
    p8a_dev_real = ALL[ALL["suite"] == "teams_real_all_dev"]["P8A"].dropna()
    p8a_tau = np.quantile(p8a_dev_real, 0.95)  # τ giving 5% dev FPR
    p8a_lockbox_real = ALL[ALL["suite"] == "teams_real_all_lockbox"]["P8A"].dropna()
    p8a_lb_fpr = (p8a_lockbox_real >= p8a_tau).mean()
    p8a_lockbox_fake = ALL[ALL["suite"] == "teams_fake_all_lockbox"]["P8A"].dropna()
    p8a_lb_recall = (p8a_lockbox_fake >= p8a_tau).mean()
    p8a_viso = ALL[ALL["suite"] == "visomaster_enhanced_macro_dev"]["P8A"].dropna()
    p8a_viso_recall = (p8a_viso >= p8a_tau).mean()
    p8a_deep = ALL[ALL["suite"] == "deeplive_enhanced_dev"]["P8A"].dropna()
    p8a_deep_recall = (p8a_deep >= p8a_tau).mean()
    print(f"P8A reference @ τ={p8a_tau:.4f} (5% dev FPR):")
    print(f"  lockbox_real_fpr={p8a_lb_fpr:.4f} lockbox_fake_recall={p8a_lb_recall:.3f} "
          f"viso_recall={p8a_viso_recall:.3f} deeplive_recall={p8a_deep_recall:.3f}")

    # Per-policy: calibrate to 5% dev FPR, measure lockbox + fake recalls
    print()
    print(f"{'Policy':<30} {'τ@5%dev':>10} {'lb_FPR':>8} {'lb_recall':>10} {'viso':>8} {'deep':>8}")
    df = ALL.copy()
    df["min_P8A_T5C"] = df[["P8A", "T5C_step3500"]].min(axis=1)
    df["max_P8A_T5C"] = df[["P8A", "T5C_step3500"]].max(axis=1)
    df["mean_P8A_T5C"] = df[["P8A", "T5C_step3500"]].mean(axis=1)
    df["routed_chronic"] = np.where(df["frame_path"].fillna("").str.contains(
        "Roy_D|PC_Generator|bla_bla_chow|Md_noyn_Sharker|dor_shkedi|healthy_dor",
        case=False, regex=True), df["P8A"], df["T5C_step3500"])
    df["routed_lap_var"] = np.where(df["lap_var"].fillna(99999) < 100,
                                    df["P8A"], df["T5C_step3500"])
    df["routed_min_dim"] = np.where(df["min_dim"].fillna(99999) < 200,
                                    df["P8A"], df["T5C_step3500"])
    df["routed_chronic_or_lowiq"] = np.where(
        df["frame_path"].fillna("").str.contains(
            "Roy_D|PC_Generator|bla_bla_chow|Md_noyn_Sharker|dor_shkedi|healthy_dor",
            case=False, regex=True) |
        (df["min_dim"].fillna(99999) < 200) | (df["lap_var"].fillna(99999) < 100),
        df["P8A"], df["T5C_step3500"])

    for policy in ["P8A", "T5C_step3500", "T3_S1_step1500",
                   "min_P8A_T5C", "max_P8A_T5C", "mean_P8A_T5C",
                   "routed_chronic", "routed_lap_var", "routed_min_dim", "routed_chronic_or_lowiq"]:
        dev_real = df[df["suite"] == "teams_real_all_dev"][policy].dropna()
        if len(dev_real) == 0:
            continue
        tau = np.quantile(dev_real, 0.95)
        lb_real = df[df["suite"] == "teams_real_all_lockbox"][policy].dropna()
        lb_fpr = (lb_real >= tau).mean()
        lb_fake = df[df["suite"] == "teams_fake_all_lockbox"][policy].dropna()
        lb_rec = (lb_fake >= tau).mean()
        viso = df[df["suite"] == "visomaster_enhanced_macro_dev"][policy].dropna()
        viso_rec = (viso >= tau).mean() if len(viso) > 0 else np.nan
        deep = df[df["suite"] == "deeplive_enhanced_dev"][policy].dropna()
        deep_rec = (deep >= tau).mean() if len(deep) > 0 else np.nan
        print(f"{policy:<30} {tau:>10.4f} {lb_fpr:>8.4f} {lb_rec:>10.3f} {viso_rec:>8.3f} {deep_rec:>8.3f}")


if __name__ == "__main__":
    main()
