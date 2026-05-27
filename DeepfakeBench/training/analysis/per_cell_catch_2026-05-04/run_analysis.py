#!/usr/bin/env python3
"""Per-cell catch-pattern analysis (Job 2 of 3, 2026-05-04).

Tests whether the cross-ckpt orthogonality observed at suite-level (e.g. P8A vs E2B
Pearson=0.30 on viso fakes) is concentrated in specific (source_method, enhancement,
teams_passthrough) cells or diffuse.

Inputs:
  analysis/suite_composition_audit_2026-05-04/frames_tagged.csv    (Job 1)
  analysis/fp_tail_characterization_2026-05-04/fp_cohort_per_ckpt.csv (Job 3, optional)

Ckpts analysed: p8a_reference_step5000, e2b_top_n_step3200, e3_top_n_step6600
(canonical 3-ckpt set per project memory).

No GPU, no inference, no parallel sklearn (project memory: avoid n_jobs=-1 on Mac).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]  # analysis/
FRAMES_CSV = ROOT / "suite_composition_audit_2026-05-04" / "frames_tagged.csv"
FP_COHORT_CSV = ROOT / "fp_tail_characterization_2026-05-04" / "fp_cohort_per_ckpt.csv"
OUT_DIR = ROOT / "per_cell_catch_2026-05-04"
OUT_DIR.mkdir(exist_ok=True)

CKPTS = {
    "P8A": "p8a_reference_step5000",
    "E2B": "e2b_top_n_step3200",
    "E3": "e3_top_n_step6600",
}
FPR_TARGETS = [0.02, 0.05, 0.10, 0.20, 0.30]
MIN_CELL_N = 30


# ---------- helpers ----------------------------------------------------------

def calibrate_tau(real_probs: np.ndarray, fpr_target: float) -> float:
    """tau s.t. fraction(real_probs >= tau) ~= fpr_target. Use upper quantile."""
    if len(real_probs) == 0:
        return float("nan")
    return float(np.quantile(real_probs, 1.0 - fpr_target))


def caught_set(df_cell_ckpt: pd.DataFrame, tau: float) -> set:
    return set(df_cell_ckpt.loc[df_cell_ckpt["frame_prob"] >= tau, "frame_path"].tolist())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / max(1, len(a | b))


# ---------- main -------------------------------------------------------------

def main():
    df = pd.read_csv(FRAMES_CSV)
    # Restrict to our 3 canonical ckpts.
    df = df[df["ckpt"].isin(CKPTS.values())].copy()

    # ----- Cell definition for fakes
    # Cell key = (source_method, enhancement, teams_passthrough). Drop dup
    # (frame_path, ckpt) since deeplive/viso/Xiang fakes appear in both
    # `<method>_enhanced_dev` and `teams_fake_all_dev` with identical scores.
    fakes_all = df[df["label"] == 1].copy()
    # Distinguish lockbox vs dev for the teams_capture cell because lockbox
    # is only 2 sessions and gets a separate row + warning.
    fakes_all["pool"] = np.where(
        fakes_all["suite"].str.contains("lockbox"), "lockbox", "dev",
    )
    # Dedup on (frame_path, ckpt, pool); when same frame appears in dev under
    # multiple suites (e.g. deeplive in deeplive_enhanced_dev + teams_fake_all_dev)
    # they share scores so dedup by frame_path/ckpt/pool is safe.
    fakes = fakes_all.drop_duplicates(subset=["frame_path", "ckpt", "pool"]).copy()

    fakes["cell"] = (
        fakes["source_method"].astype(str)
        + "|" + fakes["enhancement"].astype(str)
        + "|" + fakes["teams_passthrough"].astype(str)
        + "|" + fakes["pool"].astype(str)
    )

    # ----- Real-side: calibrate tau on teams_real_all_dev per ckpt per fpr_target
    reals_dev = df[(df["suite"] == "teams_real_all_dev") & (df["label"] == 0)].copy()
    tau_table: dict[tuple[str, float], float] = {}
    for short, ckpt_name in CKPTS.items():
        rs = reals_dev.loc[reals_dev["ckpt"] == ckpt_name, "frame_prob"].to_numpy()
        for fpr in FPR_TARGETS:
            tau_table[(short, fpr)] = calibrate_tau(rs, fpr)

    # ----- Cell inventory (size per ckpt is identical because we dedup by frame)
    inv_rows = []
    for cell, sub in fakes.groupby("cell"):
        n = sub.loc[sub["ckpt"] == CKPTS["P8A"]].shape[0]
        if n < MIN_CELL_N:
            continue
        sm, en, tp, pool = cell.split("|")
        inv_rows.append({
            "cell": cell, "source_method": sm, "enhancement": en,
            "teams_passthrough": tp, "pool": pool, "n_frames": n,
            "n_identities": sub.loc[sub["ckpt"] == CKPTS["P8A"], "identity"].nunique(),
        })
    inv = pd.DataFrame(inv_rows).sort_values(["pool", "n_frames"], ascending=[True, False])
    inv.to_csv(OUT_DIR / "cell_inventory.csv", index=False)

    # ----- Per-cell recall + score distribution per (cell, ckpt, fpr_target)
    rec_rows = []
    for cell in inv["cell"]:
        sub_cell = fakes[fakes["cell"] == cell]
        for short, ckpt_name in CKPTS.items():
            sub = sub_cell[sub_cell["ckpt"] == ckpt_name]
            probs = sub["frame_prob"].to_numpy()
            if len(probs) == 0:
                continue
            base = {
                "cell": cell, "ckpt": short, "n_frames": len(probs),
                "mean_prob_fake": float(probs.mean()),
                "p10": float(np.quantile(probs, 0.10)),
                "p50": float(np.quantile(probs, 0.50)),
                "p90": float(np.quantile(probs, 0.90)),
            }
            for fpr in FPR_TARGETS:
                tau = tau_table[(short, fpr)]
                recall = float((probs >= tau).mean())
                rec_rows.append({**base, "fpr_target": fpr, "tau": tau,
                                  "recall": recall})
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(OUT_DIR / "per_cell_recall.csv", index=False)

    # ----- Per-cell pairwise correlation + jaccard at FPR
    pair_rows = []
    pairs = [("P8A", "E2B"), ("P8A", "E3"), ("E2B", "E3")]
    for cell in inv["cell"]:
        sub_cell = fakes[fakes["cell"] == cell]
        # Pivot probs by frame_path x ckpt
        wide = sub_cell.pivot_table(
            index="frame_path", columns="ckpt", values="frame_prob",
        )
        # Rename to short
        wide = wide.rename(columns={v: k for k, v in CKPTS.items()})
        wide = wide.dropna()
        if len(wide) < MIN_CELL_N:
            continue
        for a, b in pairs:
            if a not in wide.columns or b not in wide.columns:
                continue
            x = wide[a].to_numpy()
            y = wide[b].to_numpy()
            try:
                pr = float(pearsonr(x, y)[0]) if x.std() > 0 and y.std() > 0 else float("nan")
            except Exception:
                pr = float("nan")
            try:
                sr = float(spearmanr(x, y).correlation) if x.std() > 0 and y.std() > 0 else float("nan")
            except Exception:
                sr = float("nan")
            for fpr in FPR_TARGETS:
                tau_a = tau_table[(a, fpr)]
                tau_b = tau_table[(b, fpr)]
                ca = set(wide.index[wide[a] >= tau_a])
                cb = set(wide.index[wide[b] >= tau_b])
                pair_rows.append({
                    "cell": cell, "ckpt_a": a, "ckpt_b": b,
                    "fpr_target": fpr, "pearson": pr, "spearman": sr,
                    "n_frames": len(wide), "n_caught_a": len(ca),
                    "n_caught_b": len(cb),
                    "jaccard_caught": jaccard(ca, cb),
                })
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(OUT_DIR / "per_cell_correlation.csv", index=False)

    # ----- Per-cell coverage (Venn buckets) at each FPR target
    cov_rows = []
    for cell in inv["cell"]:
        sub_cell = fakes[fakes["cell"] == cell]
        wide = sub_cell.pivot_table(
            index="frame_path", columns="ckpt", values="frame_prob",
        ).rename(columns={v: k for k, v in CKPTS.items()}).dropna()
        if len(wide) < MIN_CELL_N:
            continue
        n_total = len(wide)
        for fpr in FPR_TARGETS:
            taus = {k: tau_table[(k, fpr)] for k in CKPTS}
            caught = {k: set(wide.index[wide[k] >= taus[k]]) for k in CKPTS}
            buckets = {
                "NONE": set(wide.index) - caught["P8A"] - caught["E2B"] - caught["E3"],
                "P8A_only": caught["P8A"] - caught["E2B"] - caught["E3"],
                "E2B_only": caught["E2B"] - caught["P8A"] - caught["E3"],
                "E3_only": caught["E3"] - caught["P8A"] - caught["E2B"],
                "P8A+E2B": (caught["P8A"] & caught["E2B"]) - caught["E3"],
                "P8A+E3": (caught["P8A"] & caught["E3"]) - caught["E2B"],
                "E2B+E3": (caught["E2B"] & caught["E3"]) - caught["P8A"],
                "P8A+E2B+E3": caught["P8A"] & caught["E2B"] & caught["E3"],
            }
            for label, frames in buckets.items():
                cov_rows.append({
                    "cell": cell, "fpr_target": fpr, "caught_by": label,
                    "n_frames": len(frames), "fraction": len(frames) / n_total,
                    "n_total": n_total,
                })
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(OUT_DIR / "per_cell_coverage.csv", index=False)

    # ----- Real-side cell-conditioned (auxiliary): use Job 3 fp_cohort
    aux = {}
    if FP_COHORT_CSV.exists():
        fpc = pd.read_csv(FP_COHORT_CSV)
        # The top FP-prone identities (Job 3 named: bla_bla_chow, bla_bla_chow__s2,
        # PC_Generator__s22, PC_Generator__s45, roy_d, Q__s6).
        offenders = ["bla_bla_chow", "bla_bla_chow__s2", "PC_Generator__s22",
                     "PC_Generator__s45", "roy_d", "Q__s6"]
        if "identity_key" in fpc.columns and "clip_capture_mode" in fpc.columns:
            sub = fpc[fpc["identity_key"].isin(offenders)]
            if len(sub):
                aux["offender_capture_mode_breakdown"] = (
                    sub.groupby(["identity_key", "clip_capture_mode"]).size()
                       .unstack(fill_value=0).to_dict(orient="index")
                )
        # Save for transparency
        with open(OUT_DIR / "offender_aux.json", "w") as f:
            json.dump(aux, f, indent=2, default=str)

    # ----- Summary stats for FINDINGS
    summary = {
        "n_cells_kept": int(len(inv)),
        "min_cell_n": MIN_CELL_N,
        "tau_table": {f"{k[0]}@fpr{k[1]:.2f}": v for k, v in tau_table.items()},
        "ckpts": CKPTS,
        "fpr_targets": FPR_TARGETS,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Wrote:")
    for p in [OUT_DIR / "cell_inventory.csv",
              OUT_DIR / "per_cell_recall.csv",
              OUT_DIR / "per_cell_correlation.csv",
              OUT_DIR / "per_cell_coverage.csv",
              OUT_DIR / "summary.json"]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
