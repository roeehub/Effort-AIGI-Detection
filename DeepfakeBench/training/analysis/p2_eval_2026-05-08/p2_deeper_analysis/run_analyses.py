#!/usr/bin/env python3
"""J1-J5 analyses on the unified scoreboard.

J1 — F4-filtered τ recalibration. Drop chronic-6 from the scoreboard, re-run
     the lex policy τ search per ckpt. F4-lite per memory
     project_job14_substrate_clean_2026-05-04 ("chronic-6 axis alone explains
     96-108% of FPR drop").
J2 — FPR-matched τ counterfactual. For each ckpt, find τ* such that
     dev_worst_real_stress_fpr matches P8A's selected-τ stress_fpr (0.069).
     At τ*, report dev_fake_macro_recall + lockbox readouts.
J3 — Per-frame disagreement decomposition. Compute Δ = D_step3000 − P8A score
     per frame across dev real + fake suites. Surface high-Δ frames in each
     direction and decompose by suite + family_key. Cross-reference with IQ
     atlas's per-frame data where overlap exists.
J4 — Per-identity FPR breakdown. For each teams_capture_<identity>_dev suite,
     compute per-ckpt FPR at the contract-selected τ. Tabulate.
J5 — Per-method recall × IQ signature. For each fake suite, compute per-ckpt
     recall. Cross with method-IQ-signature from atlas.

Outputs:
  outputs/j1_f4_recalibration.csv        per-ckpt F4-lite scorecard
  outputs/j1_f4_promotion_winner.json    F4-lite lex-policy winner
  outputs/j2_fpr_matched_tau.csv         per-ckpt at FPR-matched τ
  outputs/j3_disagreement_top.csv        top-50 |Δ| frames each direction
  outputs/j3_disagreement_summary.csv    Δ stats per (suite, family_key)
  outputs/j4_per_identity_fpr.csv        identity × ckpt FPR table
  outputs/j5_per_method_recall.csv       method × ckpt recall + IQ p50
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO,
                    datefmt="%H:%M:%S")
log = logging.getLogger("p2_analyses")

THIS = Path(__file__).resolve().parent
OUT = THIS / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

ATLAS_PARQUET = (THIS.parents[2] / "iq_data_atlas_2026-05-08" /
                 "outputs" / "per_frame.parquet")

CKPT_KEYS = [
    "p8a_reference_step5000",
    "e2b_top_n_step3200",
    "p2_c_pairrank_top_n_step7000",
    "p2_c_pairrank_periodic_step3000",
    "p2_d_fourier_periodic_step3000",
    "p2_d_fourier_periodic_step8000",
    "p2_d_fourier_top_n_step19000",
]

# Selected τ per checkpoint_summary.csv (from the contract scorer).
SELECTED_TAU = {
    "p8a_reference_step5000": 0.915605,
    "e2b_top_n_step3200": 0.71077,
    "p2_c_pairrank_top_n_step7000": 0.997462,
    "p2_c_pairrank_periodic_step3000": 0.780938,
    "p2_d_fourier_periodic_step3000": 0.460036,
    "p2_d_fourier_periodic_step8000": 0.966358,
    "p2_d_fourier_top_n_step19000": 0.988899,
}

CHRONIC_PATTERNS = [
    "PC_Generator", "Roy_D", "bla_bla_chow",
    # Q has many subjects; the chronic one is Q__s6 specifically per canary.
    "Q_",
]

# Suite groupings from the contract.
DEV_REAL_PRIMARY = "teams_real_all_dev"
DEV_REAL_STRESS = ["teams_real_poor_quality_dev", "teams_real_lighting_extreme_dev"]
DEV_FAKE_SUITES = ["teams_fake_all_dev", "visomaster_enhanced_macro_dev",
                   "deeplive_enhanced_dev"]
LOCKBOX_REAL = "teams_real_all_lockbox"
LOCKBOX_FAKE = "teams_fake_all_lockbox"
TARGET_REAL_FPR = 0.07
TARGET_STRESS_FPR = 0.10
TARGET_RECALL_MIN = 0.30

CAPTURE_IDENTITY_SUITES = [
    "teams_capture_cam_test_dev", "teams_capture_test_cam_dev",
    "teams_capture_noyn_sharker_dev", "teams_capture_pc_generator_dev",
    "teams_capture_dor_shkedi_dev", "teams_capture_cam_test_s35_dev",
    "teams_capture_noyn_sharker_s23_dev", "teams_capture_cam_test_s32_dev",
    "teams_flat_xiang_xiang2_feng_dev", "teams_capture_pc_generator_s3_dev",
    "teams_capture_test_cam_s53_dev", "teams_capture_cam_test_s46_dev",
    "teams_capture_test_cam_s76_dev", "teams_capture_dor_shkedi_s16_dev",
    "teams_capture_test_cam_s73_dev", "teams_capture_cam_test_s38_dev",
    "teams_capture_pc_generator_s9_dev", "teams_capture_pc_generator_s4_dev",
]


def _is_chronic(frame_path: str) -> bool:
    bn = frame_path.split("/")[-1]
    return any(p in bn for p in CHRONIC_PATTERNS)


# ----------------------------------------------------------------------
# J1 — F4-filtered τ recalibration
# ----------------------------------------------------------------------

def fpr(scores: np.ndarray, threshold: float) -> float:
    if len(scores) == 0:
        return float("nan")
    return float((scores >= threshold).sum() / len(scores))


def recall(scores: np.ndarray, threshold: float) -> float:
    if len(scores) == 0:
        return float("nan")
    return float((scores >= threshold).sum() / len(scores))


def lex_select_tau(real_primary: np.ndarray,
                   real_stress_per_suite: dict[str, np.ndarray],
                   fake_per_suite: dict[str, np.ndarray],
                   tau_grid: np.ndarray,
                   target_real_fpr: float = TARGET_REAL_FPR,
                   target_stress_fpr: float = TARGET_STRESS_FPR,
                   target_recall_min: float = TARGET_RECALL_MIN) -> dict:
    """Return the τ that minimizes primary FPR → stress FPR → maximizes
    fake macro recall, subject to recall floor."""
    best = None
    rows = []
    for tau in tau_grid:
        primary_fpr = fpr(real_primary, tau)
        stress = max(fpr(real_stress_per_suite[s], tau)
                     for s in real_stress_per_suite)
        macro = float(np.mean([recall(fake_per_suite[s], tau)
                               for s in fake_per_suite]))
        meets_floor = macro >= target_recall_min
        rows.append({
            "tau": float(tau),
            "primary_fpr": primary_fpr,
            "stress_fpr": stress,
            "fake_macro_recall": macro,
            "meets_recall_floor": bool(meets_floor),
        })
    df = pd.DataFrame(rows)
    eligible = df[df["meets_recall_floor"]]
    pool = eligible if len(eligible) > 0 else df
    pool = pool.sort_values(by=["primary_fpr", "stress_fpr",
                                "fake_macro_recall"],
                            ascending=[True, True, False])
    best = pool.iloc[0].to_dict()
    return {"selected": best, "grid_size": len(df)}


def run_j1_f4(score_df: pd.DataFrame) -> None:
    log.info("J1 — F4-filtered τ recalibration starting")

    # F4-lite filter: drop chronic-6 frames from real cohorts.
    score_df = score_df.copy()
    score_df["chronic"] = score_df["frame_path"].apply(_is_chronic)

    # Apply F4-lite to REAL cohorts only (chronic identities can also appear
    # as fakes in capture_pc_generator_*; we don't filter those).
    f4 = score_df[
        ~((score_df["label"] == 0) & (score_df["chronic"]))
    ].copy()
    log.info("F4-lite: %d → %d frames (%d real chronic dropped)",
             len(score_df), len(f4),
             ((score_df["label"] == 0) & (score_df["chronic"])).sum())

    rows = []
    for ckpt in CKPT_KEYS:
        # Pull score arrays per suite × label.
        real_primary = f4[(f4["suite"] == DEV_REAL_PRIMARY) &
                          (f4["label"] == 0)][ckpt].dropna().values
        real_stress_per_suite = {
            s: f4[(f4["suite"] == s) & (f4["label"] == 0)][ckpt].dropna().values
            for s in DEV_REAL_STRESS
        }
        fake_per_suite = {
            s: f4[(f4["suite"] == s) & (f4["label"] == 1)][ckpt].dropna().values
            for s in DEV_FAKE_SUITES
        }
        lockbox_real = f4[(f4["suite"] == LOCKBOX_REAL) &
                          (f4["label"] == 0)][ckpt].dropna().values
        lockbox_fake = f4[(f4["suite"] == LOCKBOX_FAKE) &
                          (f4["label"] == 1)][ckpt].dropna().values

        # Build a τ-grid from the union of real-primary scores (the same
        # candidate-threshold logic as score_teams_promotion_contract.py).
        candidates = np.unique(np.concatenate(
            [real_primary] + [v for v in real_stress_per_suite.values()]
            + [v for v in fake_per_suite.values()]))
        # Use a quantile-based subgrid to keep it tractable.
        if len(candidates) > 5000:
            candidates = np.unique(np.quantile(
                candidates, np.linspace(0, 1, 5000)))

        result = lex_select_tau(real_primary, real_stress_per_suite,
                                fake_per_suite, candidates)
        sel = result["selected"]
        tau = sel["tau"]
        # Lockbox readouts at the F4 τ (computed on the F4 substrate too).
        lock_real_fpr = fpr(lockbox_real, tau)
        lock_fake_recall = recall(lockbox_fake, tau)

        rows.append({
            "ckpt": ckpt,
            "f4_selected_tau": tau,
            "f4_primary_fpr": sel["primary_fpr"],
            "f4_stress_fpr": sel["stress_fpr"],
            "f4_fake_macro_recall": sel["fake_macro_recall"],
            "f4_meets_floor": sel["meets_recall_floor"],
            "f4_lockbox_real_fpr": lock_real_fpr,
            "f4_lockbox_fake_recall": lock_fake_recall,
        })

    f4_df = pd.DataFrame(rows)
    # Lex-rank.
    f4_df = f4_df.sort_values(by=["f4_primary_fpr", "f4_stress_fpr",
                                  "f4_fake_macro_recall"],
                              ascending=[True, True, False]).reset_index(drop=True)
    f4_df["rank"] = f4_df.index + 1
    out = OUT / "j1_f4_recalibration.csv"
    f4_df.to_csv(out, index=False)
    log.info("wrote %s", out)
    print()
    print("=== J1 — F4-lite τ recalibration ===")
    print(f4_df.round(4).to_string(index=False))

    winner = f4_df.iloc[0].to_dict()
    (OUT / "j1_f4_promotion_winner.json").write_text(
        json.dumps(winner, indent=2, default=str))


# ----------------------------------------------------------------------
# J2 — FPR-matched τ counterfactual
# ----------------------------------------------------------------------

def run_j2_fpr_matched_tau(score_df: pd.DataFrame) -> None:
    """For each ckpt: find τ such that worst stress FPR ≤ P8A's stress_fpr
    (0.069). At that τ, report fake_macro_recall + lockbox readouts."""
    log.info("J2 — FPR-matched τ counterfactual starting")

    P8A_STRESS_FPR_BAR = 0.069  # the bar to match
    rows = []
    for ckpt in CKPT_KEYS:
        real_stress_per_suite = {
            s: score_df[(score_df["suite"] == s) & (score_df["label"] == 0)][ckpt].dropna().values
            for s in DEV_REAL_STRESS
        }
        fake_per_suite = {
            s: score_df[(score_df["suite"] == s) & (score_df["label"] == 1)][ckpt].dropna().values
            for s in DEV_FAKE_SUITES
        }
        lockbox_real = score_df[(score_df["suite"] == LOCKBOX_REAL) &
                                (score_df["label"] == 0)][ckpt].dropna().values
        lockbox_fake = score_df[(score_df["suite"] == LOCKBOX_FAKE) &
                                (score_df["label"] == 1)][ckpt].dropna().values

        # Sweep τ ∈ [0, 1] and find the smallest τ that meets the stress bar.
        tau_grid = np.linspace(0, 1, 1001)
        # stress_fpr is monotonically decreasing in τ; find the smallest τ
        # where worst stress ≤ bar.
        ok_taus = []
        for t in tau_grid:
            worst = max(fpr(real_stress_per_suite[s], t)
                        for s in real_stress_per_suite)
            if worst <= P8A_STRESS_FPR_BAR:
                ok_taus.append(t)
                break  # smallest is the most permissive that meets the bar
        if not ok_taus:
            tau_matched = 1.0
        else:
            tau_matched = ok_taus[0]
        macro = float(np.mean([recall(fake_per_suite[s], tau_matched)
                               for s in fake_per_suite]))
        lock_real_fpr = fpr(lockbox_real, tau_matched)
        lock_fake_recall = recall(lockbox_fake, tau_matched)

        rows.append({
            "ckpt": ckpt,
            "tau_matched_to_P8A_stress": tau_matched,
            "stress_fpr_at_tau": max(fpr(real_stress_per_suite[s], tau_matched)
                                     for s in real_stress_per_suite),
            "fake_macro_recall_at_tau": macro,
            "lockbox_real_fpr_at_tau": lock_real_fpr,
            "lockbox_fake_recall_at_tau": lock_fake_recall,
            "selected_tau_in_contract": SELECTED_TAU[ckpt],
        })

    j2_df = pd.DataFrame(rows)
    out = OUT / "j2_fpr_matched_tau.csv"
    j2_df.to_csv(out, index=False)
    log.info("wrote %s", out)
    print()
    print("=== J2 — FPR-matched τ counterfactual (bar: stress_fpr ≤ 0.069 = P8A) ===")
    print(j2_df.round(4).to_string(index=False))


# ----------------------------------------------------------------------
# J3 — Per-frame disagreement decomposition
# ----------------------------------------------------------------------

def run_j3_disagreement(score_df: pd.DataFrame) -> None:
    log.info("J3 — Per-frame disagreement decomposition starting")

    score_df = score_df.copy()
    # Δ = D_step3000 score − P8A score per frame.
    score_df["delta_D_minus_P8A"] = (
        score_df["p2_d_fourier_periodic_step3000"]
        - score_df["p8a_reference_step5000"]
    )
    score_df["delta_D_minus_E2B"] = (
        score_df["p2_d_fourier_periodic_step3000"]
        - score_df["e2b_top_n_step3200"]
    )

    # Try to attach IQ-atlas data.
    iq_attached = False
    if ATLAS_PARQUET.exists():
        atlas = pd.read_parquet(ATLAS_PARQUET)
        # Atlas frame_path is a basename or full GCS URL? Check.
        atlas_cols = [c for c in atlas.columns]
        log.info("atlas columns: %s", atlas_cols[:8])
        # If frame_path matches GCS URL, merge directly; otherwise on basename.
        if "frame_path" in atlas.columns:
            iq_keep = ["frame_path", "min_dim", "lap_var",
                       "luma_mean", "color_b_dev", "edge_mag", "skin_frac"]
            iq_keep = [c for c in iq_keep if c in atlas.columns]
            score_df = score_df.merge(atlas[iq_keep], on="frame_path", how="left")
            iq_attached = score_df["min_dim"].notna().sum() > 0
            log.info("IQ atlas merge: %d / %d frames attached",
                     score_df["min_dim"].notna().sum(), len(score_df))

    # Top-K |Δ| frames in each direction (D much higher / D much lower).
    top_d_higher = score_df.assign(abs_delta=score_df["delta_D_minus_P8A"].abs()) \
        .nlargest(50, "delta_D_minus_P8A") \
        .copy()
    top_d_lower = score_df.assign(abs_delta=score_df["delta_D_minus_P8A"].abs()) \
        .nsmallest(50, "delta_D_minus_P8A") \
        .copy()
    keep_cols = ["suite", "frame_path", "label", "method", "family_key",
                 "p8a_reference_score" if False else "p8a_reference_step5000",
                 "p2_d_fourier_periodic_step3000",
                 "delta_D_minus_P8A", "delta_D_minus_E2B"]
    if iq_attached:
        keep_cols += ["min_dim", "lap_var", "luma_mean", "color_b_dev"]
    top_combined = pd.concat([
        top_d_higher[keep_cols].assign(direction="D_HIGHER"),
        top_d_lower[keep_cols].assign(direction="D_LOWER"),
    ], axis=0, ignore_index=True)
    out_top = OUT / "j3_disagreement_top.csv"
    top_combined.to_csv(out_top, index=False)
    log.info("wrote %s", out_top)

    # Per-suite × per-label summary.
    summary = score_df.groupby(["suite", "label"]).agg(
        n=("delta_D_minus_P8A", "count"),
        mean_delta_D_minus_P8A=("delta_D_minus_P8A", "mean"),
        median_delta_D_minus_P8A=("delta_D_minus_P8A", "median"),
        std_delta_D_minus_P8A=("delta_D_minus_P8A", "std"),
        frac_D_higher=("delta_D_minus_P8A", lambda s: float((s > 0).mean())),
        frac_D_higher_by_0p2=("delta_D_minus_P8A", lambda s: float((s > 0.2).mean())),
        frac_D_lower_by_0p2=("delta_D_minus_P8A", lambda s: float((s < -0.2).mean())),
    ).reset_index()
    out_sum = OUT / "j3_disagreement_summary.csv"
    summary.to_csv(out_sum, index=False)
    log.info("wrote %s", out_sum)
    print()
    print("=== J3 — disagreement summary (D_step3000 vs P8A) ===")
    print(summary.round(4).to_string(index=False))


# ----------------------------------------------------------------------
# J4 — Per-identity FPR breakdown
# ----------------------------------------------------------------------

def run_j4_per_identity_fpr(score_df: pd.DataFrame) -> None:
    """Per-identity FPR at the contract-selected τ for each ckpt."""
    log.info("J4 — Per-identity FPR breakdown starting")

    # Note: capture_<identity>_dev suites are FAKE methods (face-swap of an
    # identity), not real frames. The "FPR" question on these suites is
    # actually "fake_recall" — does the model catch face-swaps of this
    # identity? We compute recall here (label=1) for the per-identity capture
    # suites. For real-side per-identity, the only suite we have is
    # teams_real_dor_dev (which IS in the contract).
    rows = []
    for suite in CAPTURE_IDENTITY_SUITES + ["teams_real_dor_dev"]:
        for ckpt in CKPT_KEYS:
            tau = SELECTED_TAU[ckpt]
            sub = score_df[score_df["suite"] == suite]
            scores = sub[ckpt].dropna().values
            label = sub["label"].iloc[0] if len(sub) > 0 else None
            if label is None or len(scores) == 0:
                continue
            metric_name = "recall" if label == 1 else "fpr"
            value = float((scores >= tau).sum() / len(scores))
            rows.append({
                "suite": suite,
                "ckpt": ckpt,
                "label": int(label),
                "metric": metric_name,
                "n_videos": len(scores),
                "value_at_contract_tau": value,
                "tau": tau,
            })

    j4_df = pd.DataFrame(rows)
    # Pivot: rows=suite, columns=ckpt, values=value (separately for real / fake).
    real = j4_df[j4_df["label"] == 0]
    fake = j4_df[j4_df["label"] == 1]
    if len(real) > 0:
        real_pivot = real.pivot(index="suite", columns="ckpt",
                                values="value_at_contract_tau")
        real_pivot.to_csv(OUT / "j4_per_identity_real_fpr.csv")
        print()
        print("=== J4 — Per-identity REAL FPR (only teams_real_dor_dev in contract) ===")
        print(real_pivot.round(4).to_string())
    if len(fake) > 0:
        fake_pivot = fake.pivot(index="suite", columns="ckpt",
                                values="value_at_contract_tau")
        fake_pivot.to_csv(OUT / "j4_per_identity_fake_recall.csv")
        print()
        print("=== J4 — Per-identity FAKE recall (capture_<identity>_dev suites) ===")
        print(fake_pivot.round(4).to_string())
    log.info("wrote J4 csvs")


# ----------------------------------------------------------------------
# J5 — Per-method recall × IQ signature
# ----------------------------------------------------------------------

def run_j5_per_method(score_df: pd.DataFrame) -> None:
    log.info("J5 — Per-method recall × IQ signature starting")

    # For each fake-bearing suite, compute per-ckpt recall at the
    # contract-selected τ.
    fake_suites = DEV_FAKE_SUITES + [LOCKBOX_FAKE] + CAPTURE_IDENTITY_SUITES
    rows = []
    for suite in fake_suites:
        sub = score_df[score_df["suite"] == suite]
        if len(sub) == 0 or sub["label"].iloc[0] != 1:
            continue
        for ckpt in CKPT_KEYS:
            tau = SELECTED_TAU[ckpt]
            scores = sub[ckpt].dropna().values
            if len(scores) == 0:
                continue
            rec = float((scores >= tau).sum() / len(scores))
            rows.append({
                "suite": suite,
                "ckpt": ckpt,
                "n_videos": len(scores),
                "recall_at_contract_tau": rec,
            })

    j5_df = pd.DataFrame(rows)
    pivot = j5_df.pivot(index="suite", columns="ckpt",
                        values="recall_at_contract_tau")

    # Attach IQ signature per suite from the atlas.
    iq_by_pool = None
    if ATLAS_PARQUET.exists():
        atlas = pd.read_parquet(ATLAS_PARQUET)
        if "pool" in atlas.columns and "lap_var" in atlas.columns:
            iq_by_pool = atlas.groupby("pool")["lap_var"].median()
            pivot["lap_var_p50"] = [iq_by_pool.get(s, float("nan"))
                                    for s in pivot.index]

    pivot.to_csv(OUT / "j5_per_method_recall.csv")
    log.info("wrote J5 csv")
    print()
    print("=== J5 — Per-method recall at contract τ × suite lap_var p50 ===")
    print(pivot.round(4).to_string())


# ----------------------------------------------------------------------

def main():
    sb = pd.read_parquet(THIS / "outputs" / "scoreboard.parquet")
    log.info("loaded scoreboard: %d rows × %d cols", *sb.shape)
    run_j1_f4(sb)
    run_j2_fpr_matched_tau(sb)
    run_j3_disagreement(sb)
    run_j4_per_identity_fpr(sb)
    run_j5_per_method(sb)
    log.info("ALL JOBS COMPLETE")


if __name__ == "__main__":
    main()
