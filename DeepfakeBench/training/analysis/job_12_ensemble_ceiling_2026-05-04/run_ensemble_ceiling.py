#!/usr/bin/env python3
"""
Job 12 — Ensemble ceiling analysis across (P8A, E2B_3200, E3_6600).

Pure arithmetic on cached per-frame scores. n_jobs=1.
Treat the three ckpts symmetrically (no anchor on P8A).
"""
import os
import re
import numpy as np
import pandas as pd

ROOT = "/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis"
RAW = os.path.join(ROOT, "cpu_followups_2026-05-04", "raw_reports")
AUC_FILE = os.path.join(ROOT, "cpu_followups_2026-05-04", "outputs", "02b_cross_suite_auc.csv")
OUT = os.path.join(ROOT, "job_12_ensemble_ceiling_2026-05-04")
os.makedirs(OUT, exist_ok=True)

CKPTS = ["P8A", "E2B", "E3"]
CKPT_FILE_TAGS = {
    "P8A": "p8a_reference_step5000",
    "E2B": "e2b_top_n_step3200",
    "E3":  "e3_top_n_step6600",
}

REAL_SUITES = ["teams_real_all_dev", "teams_real_all_lockbox"]
FAKE_SUITES = [
    "visomaster_enhanced_macro_dev",
    "deeplive_enhanced_dev",
    "teams_fake_all_dev",
    "teams_fake_all_lockbox",
]
ALL_SUITES = REAL_SUITES + FAKE_SUITES

CALIB_REAL_SUITE = "teams_real_all_dev"
FPR_TARGETS = [0.02, 0.05, 0.10, 0.20]

# AUC weights map (suite-aware)
AUC_DF = pd.read_csv(AUC_FILE)


def get_auc(ckpt: str, fake_suite_with_dev: str) -> float:
    """Look up AUC for ckpt × fake_suite (the AUC file uses *_dev style)."""
    df = AUC_DF[(AUC_DF["ckpt"] == ckpt) & (AUC_DF["fake_suite"] == fake_suite_with_dev)]
    if len(df) == 0:
        return np.nan
    return float(df["auc"].iloc[0])


def load_suite(suite: str) -> pd.DataFrame:
    """Inner-join per-frame scores across the 3 ckpts."""
    dfs = {}
    for ckpt in CKPTS:
        path = os.path.join(RAW, f"{suite}_{CKPT_FILE_TAGS[ckpt]}_frames_report.csv")
        df = pd.read_csv(path, usecols=["frame_path", "frame_prob", "label", "video_id", "group_key"])
        df = df.rename(columns={"frame_prob": f"prob_{ckpt}"})
        dfs[ckpt] = df

    base = dfs["P8A"][["frame_path", "label", "video_id", "group_key", "prob_P8A"]].copy()
    for ckpt in ["E2B", "E3"]:
        base = base.merge(
            dfs[ckpt][["frame_path", f"prob_{ckpt}"]],
            on="frame_path",
            how="inner",
        )
    return base


# ---- Strategies ---------------------------------------------------

def apply_strategy(df: pd.DataFrame, strategy: str, fake_suite_for_weights: str | None = None) -> np.ndarray:
    p = df["prob_P8A"].values.astype(float)
    e2 = df["prob_E2B"].values.astype(float)
    e3 = df["prob_E3"].values.astype(float)
    stack = np.stack([p, e2, e3], axis=0)  # (3, N)

    if strategy == "P8A_ONLY":
        return p
    if strategy == "E2B_ONLY":
        return e2
    if strategy == "E3_ONLY":
        return e3
    if strategy == "MAX":
        return stack.max(axis=0)
    if strategy == "MIN":
        return stack.min(axis=0)
    if strategy == "MEAN":
        return stack.mean(axis=0)
    if strategy == "MEDIAN":
        return np.median(stack, axis=0)
    if strategy == "PRODUCT":
        return stack.prod(axis=0)
    if strategy == "GEOMEAN":
        # safe geometric mean: clip floor
        s = np.clip(stack, 1e-12, 1.0)
        return np.exp(np.log(s).mean(axis=0))
    if strategy == "WEIGHTED_BY_AUC":
        # Use AUC for the fake suite as weights. For real suites we still need *some* weight
        # — use an equal weight if not provided (only matters for calibration table).
        if fake_suite_for_weights is None:
            w = np.array([1.0, 1.0, 1.0])
        else:
            w = np.array([
                get_auc("P8A", fake_suite_for_weights),
                get_auc("E2B_3200", fake_suite_for_weights),
                get_auc("E3_6600", fake_suite_for_weights),
            ])
            if np.any(np.isnan(w)):
                w = np.array([1.0, 1.0, 1.0])
        w = w / w.sum()
        return (stack * w[:, None]).sum(axis=0)
    if strategy == "ORACLE_FAKE_MAX_REAL_MIN":
        labels = df["label"].values
        out = np.where(labels == 1, stack.max(axis=0), stack.min(axis=0))
        return out
    raise ValueError(strategy)


SINGLE_STRATEGIES = ["P8A_ONLY", "E2B_ONLY", "E3_ONLY"]
ENSEMBLE_STRATEGIES = [
    "MAX", "MIN", "MEAN", "MEDIAN", "PRODUCT", "GEOMEAN",
    "WEIGHTED_BY_AUC", "ORACLE_FAKE_MAX_REAL_MIN",
]
ALL_STRATEGIES = SINGLE_STRATEGIES + ENSEMBLE_STRATEGIES


# ---- τ calibration ------------------------------------------------

def calibrate_tau(real_scores: np.ndarray, fpr_target: float) -> float:
    """τ = (1 - fpr_target) quantile of real scores. (recall = #fakes >= τ.)"""
    return float(np.quantile(real_scores, 1.0 - fpr_target))


def recall_at_tau(scores: np.ndarray, tau: float) -> tuple[float, int, int]:
    n = len(scores)
    if n == 0:
        return 0.0, 0, 0
    caught = int((scores >= tau).sum())
    return caught / n, caught, n


# ---- Step 1+2 — load all ------------------------------------------

print("[Job 12] Loading suites...")
SUITE_DATA: dict[str, pd.DataFrame] = {}
for suite in ALL_SUITES:
    df = load_suite(suite)
    SUITE_DATA[suite] = df
    print(f"  {suite}: n={len(df)}")

# ---- Step 3 — recall per strategy × suite × FPR -------------------
print("[Job 12] Computing per-strategy recall + FPR...")

# fake_suite_with_dev mapping (the 02b_cross_suite_auc.csv uses *_dev only)
FAKE_SUITE_WEIGHT_KEY = {
    "visomaster_enhanced_macro_dev": "visomaster_enhanced_macro_dev",
    "deeplive_enhanced_dev": "deeplive_enhanced_dev",
    "teams_fake_all_dev": "teams_fake_all_dev",
    "teams_fake_all_lockbox": "teams_fake_all_lockbox",
    # for real suites, no per-fake weight; calibrator uses an "average" surrogate
    "teams_real_all_dev": "teams_fake_all_dev",
    "teams_real_all_lockbox": "teams_fake_all_dev",
}

calib_df = SUITE_DATA[CALIB_REAL_SUITE]
records: list[dict] = []

for strategy in ALL_STRATEGIES:
    for fpr_target in FPR_TARGETS:
        # Calibrate τ on teams_real_all_dev with this strategy.
        # For WEIGHTED_BY_AUC and ORACLE we still have to pick a weight reference for calibration —
        # we fix the weight reference per-(target-suite) below in the recall step. To keep one τ
        # per strategy×fpr, we calibrate using equal-weight (None) surrogate; this is documented.
        calib_scores = apply_strategy(calib_df, strategy, fake_suite_for_weights=None)
        tau = calibrate_tau(calib_scores, fpr_target)
        actual_calib_fpr, _, _ = recall_at_tau(calib_scores, tau)

        for suite in ALL_SUITES:
            df = SUITE_DATA[suite]
            weight_key = FAKE_SUITE_WEIGHT_KEY[suite]
            scores = apply_strategy(df, strategy, fake_suite_for_weights=weight_key)
            # For weighted strategy, calibrate τ using the SAME weights so calibration is honest.
            if strategy in ("WEIGHTED_BY_AUC",):
                # Re-calibrate τ using the same weight as evaluation suite.
                calib_scores_w = apply_strategy(calib_df, strategy, fake_suite_for_weights=weight_key)
                tau_use = calibrate_tau(calib_scores_w, fpr_target)
            else:
                tau_use = tau
            recall, caught, n = recall_at_tau(scores, tau_use)
            records.append({
                "strategy": strategy,
                "fpr_target": fpr_target,
                "calib_real_suite": CALIB_REAL_SUITE,
                "calib_actual_fpr": actual_calib_fpr,
                "tau": tau_use,
                "eval_suite": suite,
                "is_real_suite": suite in REAL_SUITES,
                "recall_or_fpr": recall,
                "caught": caught,
                "n": n,
            })

ensemble_recall_df = pd.DataFrame(records)
ensemble_recall_df.to_csv(os.path.join(OUT, "ensemble_recall_per_strategy.csv"), index=False)
print(f"  wrote ensemble_recall_per_strategy.csv ({len(ensemble_recall_df)} rows)")

# ---- ensemble-vs-single lift -------------------------------------

best_single = (
    ensemble_recall_df[ensemble_recall_df["strategy"].isin(SINGLE_STRATEGIES)]
    .groupby(["fpr_target", "eval_suite"])
    .agg(best_single_recall=("recall_or_fpr", "max"),
         best_single_strategy=("recall_or_fpr", lambda s: s.idxmax()))
    .reset_index()
)
# Reformulate best_single_strategy properly
def _bs_strat(group: pd.DataFrame) -> str:
    g = group.sort_values("recall_or_fpr", ascending=False).iloc[0]
    return g["strategy"]
best_single_strat = (
    ensemble_recall_df[ensemble_recall_df["strategy"].isin(SINGLE_STRATEGIES)]
    .groupby(["fpr_target", "eval_suite"]).apply(_bs_strat).rename("best_single_strategy").reset_index()
)
best_single = best_single.drop(columns=["best_single_strategy"]).merge(
    best_single_strat, on=["fpr_target", "eval_suite"], how="left"
)

lift_records = []
for (fpr, suite), sub in ensemble_recall_df.groupby(["fpr_target", "eval_suite"]):
    bs = best_single[(best_single["fpr_target"] == fpr) & (best_single["eval_suite"] == suite)].iloc[0]
    for strategy in ENSEMBLE_STRATEGIES:
        row = sub[sub["strategy"] == strategy].iloc[0]
        lift_records.append({
            "strategy": strategy,
            "fpr_target": fpr,
            "eval_suite": suite,
            "is_real_suite": suite in REAL_SUITES,
            "best_single_strategy": bs["best_single_strategy"],
            "best_single_recall": bs["best_single_recall"],
            "ensemble_recall": row["recall_or_fpr"],
            "lift_abs": row["recall_or_fpr"] - bs["best_single_recall"],
            "lift_rel": (row["recall_or_fpr"] / bs["best_single_recall"]) if bs["best_single_recall"] > 0 else float("nan"),
            "n": row["n"],
        })

lift_df = pd.DataFrame(lift_records)
lift_df.to_csv(os.path.join(OUT, "ensemble_vs_single_lift.csv"), index=False)
print(f"  wrote ensemble_vs_single_lift.csv ({len(lift_df)} rows)")


# ---- Step 4 — MAX-strategy Venn at FPR=10% -----------------------
print("[Job 12] Step 4 Venn (MAX @ FPR=10%)...")

calib_scores_per_ckpt = {
    "P8A": SUITE_DATA[CALIB_REAL_SUITE]["prob_P8A"].values,
    "E2B": SUITE_DATA[CALIB_REAL_SUITE]["prob_E2B"].values,
    "E3":  SUITE_DATA[CALIB_REAL_SUITE]["prob_E3"].values,
}
TAU10 = {ckpt: calibrate_tau(calib_scores_per_ckpt[ckpt], 0.10) for ckpt in CKPTS}

venn_records = []
for suite in FAKE_SUITES + REAL_SUITES:
    df = SUITE_DATA[suite]
    p_caught = (df["prob_P8A"].values >= TAU10["P8A"]).astype(int)
    e2_caught = (df["prob_E2B"].values >= TAU10["E2B"]).astype(int)
    e3_caught = (df["prob_E3"].values >= TAU10["E3"]).astype(int)
    n_caught = p_caught + e2_caught + e3_caught
    bucket_counts: dict[str, int] = {
        "caught_by_0": int(((p_caught == 0) & (e2_caught == 0) & (e3_caught == 0)).sum()),
        "caught_P8A_only": int(((p_caught == 1) & (e2_caught == 0) & (e3_caught == 0)).sum()),
        "caught_E2B_only": int(((p_caught == 0) & (e2_caught == 1) & (e3_caught == 0)).sum()),
        "caught_E3_only": int(((p_caught == 0) & (e2_caught == 0) & (e3_caught == 1)).sum()),
        "caught_P8A_E2B": int(((p_caught == 1) & (e2_caught == 1) & (e3_caught == 0)).sum()),
        "caught_P8A_E3": int(((p_caught == 1) & (e2_caught == 0) & (e3_caught == 1)).sum()),
        "caught_E2B_E3": int(((p_caught == 0) & (e2_caught == 1) & (e3_caught == 1)).sum()),
        "caught_all_3": int(((p_caught == 1) & (e2_caught == 1) & (e3_caught == 1)).sum()),
    }
    n = len(df)
    for bucket, cnt in bucket_counts.items():
        venn_records.append({
            "suite": suite,
            "is_real_suite": suite in REAL_SUITES,
            "caught_by": bucket,
            "count": cnt,
            "n_total": n,
            "fraction": cnt / n if n > 0 else 0.0,
        })

venn_df = pd.DataFrame(venn_records)
venn_df.to_csv(os.path.join(OUT, "max_ensemble_venn.csv"), index=False)
print(f"  wrote max_ensemble_venn.csv")


# ---- Step 5 — viso subtype stratification ------------------------
print("[Job 12] Step 5 — viso subtype stratification...")

VISO = SUITE_DATA["visomaster_enhanced_macro_dev"].copy()
# Subtype from frame_path: 'visomaster_enhanced_raw__' vs 'visomaster_enhanced_teams__'
def _subtype(fp: str) -> str:
    fn = os.path.basename(fp)
    if "visomaster_enhanced_raw__" in fn:
        return "raw"
    if "visomaster_enhanced_teams__" in fn:
        return "teams"
    return "other"

VISO["subtype"] = VISO["frame_path"].apply(_subtype)
print("  viso subtype counts:", VISO["subtype"].value_counts().to_dict())

subtype_records = []
for fpr_target in FPR_TARGETS:
    for strategy in ALL_STRATEGIES:
        # τ from teams_real_all_dev (same as Step 3)
        calib_scores = apply_strategy(SUITE_DATA[CALIB_REAL_SUITE], strategy,
                                      fake_suite_for_weights="visomaster_enhanced_macro_dev")
        tau = calibrate_tau(calib_scores, fpr_target)
        scores_v = apply_strategy(VISO, strategy, fake_suite_for_weights="visomaster_enhanced_macro_dev")
        for sub in ["raw", "teams"]:
            mask = (VISO["subtype"] == sub).values
            if mask.sum() == 0:
                continue
            recall, caught, n = recall_at_tau(scores_v[mask], tau)
            subtype_records.append({
                "subtype": sub,
                "strategy": strategy,
                "fpr_target": fpr_target,
                "tau": tau,
                "recall": recall,
                "caught": caught,
                "n": n,
            })

subtype_df = pd.DataFrame(subtype_records)
subtype_df.to_csv(os.path.join(OUT, "subtype_ensemble_recall.csv"), index=False)
print(f"  wrote subtype_ensemble_recall.csv ({len(subtype_df)} rows)")

print("[Job 12] Done.")
