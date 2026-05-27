"""Bootstrap 95% CIs on per-human FPR / fake-recall.

Validates whether borderline cells (P8A Xinhe fake recall 0.487 at mode B,
SlotAv2_FACE dor recall 0.499 at per-ckpt τ=0.681, T5C dor FPR 0.052 at mode B)
are sample noise or signal.

Outputs:
  outputs/per_cell_ci.csv     — long table: ckpt × tau_kind × human × role × {n, point, ci_lo, ci_hi}
  outputs/borderline_check.csv — focused borderline-cell readout
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
SRC = ROOT / "analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/per_frame_full.csv"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Per-ckpt best-τ values from analysis/per_ckpt_tau_recal_2026-05-23/outputs/per_ckpt_summary.json
# (user_bar gate, midpoint of widest passing range, or argmax recall if no pass)
PER_CKPT_TAU = {
    "P8A":           {"mode_A": 0.535, "mode_B": 0.78, "mode_C": 0.87, "per_ckpt": 0.590},
    "E2B":           {"mode_A": 0.535, "mode_B": 0.78, "mode_C": 0.87, "per_ckpt": 0.718},
    "T5C":           {"mode_A": 0.535, "mode_B": 0.78, "mode_C": 0.87, "per_ckpt": 0.790},
    "SlotAv2_CLS":   {"mode_A": 0.535, "mode_B": 0.78, "mode_C": 0.87, "per_ckpt": 0.562},
    "SlotAv2_FACE":  {"mode_A": 0.535, "mode_B": 0.78, "mode_C": 0.87, "per_ckpt": 0.681},
}

CKPTS = list(PER_CKPT_TAU.keys())
SCORE_COL = {c: f"prob_{c}" for c in CKPTS}
REAL_HUMANS = ["Noyn", "Roee_Windows", "Xiang", "Xinhe", "dor"]
FAKE_HUMANS = ["Xiang", "Xinhe", "dor"]
FAKE_ROLE = {h: f"fake_target_{h}" for h in FAKE_HUMANS}

N_BOOT = 2000
RNG_SEED = 42


def bootstrap_proportion(values_above_tau: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Returns (point, ci_lo_95, ci_hi_95) for the proportion of values above τ.

    values_above_tau is a 0/1 array.
    """
    n = len(values_above_tau)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(values_above_tau.mean())
    # Bootstrap: resample indices with replacement
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values_above_tau[idx].mean(axis=1)
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    return point, ci_lo, ci_hi


def main() -> None:
    df = pd.read_csv(SRC)
    df = df[df["deploy_relevant"] == True].copy()
    print(f"loaded {len(df)} deploy-relevant frames")

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    borderline_rows = []

    for ckpt in CKPTS:
        score_col = SCORE_COL[ckpt]
        taus = PER_CKPT_TAU[ckpt]
        for tau_kind, tau in taus.items():
            # Per-human real FPR
            for h in REAL_HUMANS:
                sub = df[(df["human"] == h) & (df["role"] == "real")][score_col].to_numpy()
                above = (sub >= tau).astype(int)
                point, lo, hi = bootstrap_proportion(above, N_BOOT, rng)
                rows.append({
                    "ckpt": ckpt, "tau_kind": tau_kind, "tau": tau,
                    "human": h, "metric": "real_FPR",
                    "n": len(sub), "point": point, "ci_lo": lo, "ci_hi": hi,
                    "ci_width": hi - lo,
                })

            # Per-human fake recall
            for h in FAKE_HUMANS:
                sub = df[(df["human"] == h) & (df["role"] == FAKE_ROLE[h])][score_col].to_numpy()
                above = (sub >= tau).astype(int)
                point, lo, hi = bootstrap_proportion(above, N_BOOT, rng)
                rows.append({
                    "ckpt": ckpt, "tau_kind": tau_kind, "tau": tau,
                    "human": h, "metric": "fake_recall",
                    "n": len(sub), "point": point, "ci_lo": lo, "ci_hi": hi,
                    "ci_width": hi - lo,
                })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT / "per_cell_ci.csv", index=False)

    # Borderline cells (point estimate within ±2pp of 5% or 50% floors)
    borderline = out_df[
        ((out_df["metric"] == "real_FPR") & (out_df["point"].between(0.03, 0.07))) |
        ((out_df["metric"] == "fake_recall") & (out_df["point"].between(0.45, 0.55)))
    ].copy()
    borderline["floor"] = borderline["metric"].map({"real_FPR": 0.05, "fake_recall": 0.50})
    borderline["dist_from_floor"] = borderline["point"] - borderline["floor"]
    # Does the CI cross the floor?
    borderline["ci_crosses_floor"] = (
        (borderline["ci_lo"] <= borderline["floor"]) & (borderline["ci_hi"] >= borderline["floor"])
    )
    # Is the gate-relevant side passing? For real_FPR, floor is ceiling — passing means below.
    # For fake_recall, floor is floor — passing means above.
    borderline["point_passes_gate"] = (
        ((borderline["metric"] == "real_FPR") & (borderline["point"] <= 0.05)) |
        ((borderline["metric"] == "fake_recall") & (borderline["point"] >= 0.50))
    )
    borderline["ci_definitively_passes"] = (
        ((borderline["metric"] == "real_FPR") & (borderline["ci_hi"] < 0.05)) |
        ((borderline["metric"] == "fake_recall") & (borderline["ci_lo"] > 0.50))
    )
    borderline["ci_definitively_fails"] = (
        ((borderline["metric"] == "real_FPR") & (borderline["ci_lo"] > 0.05)) |
        ((borderline["metric"] == "fake_recall") & (borderline["ci_hi"] < 0.50))
    )
    borderline.to_csv(OUT / "borderline_check.csv", index=False)

    print("\n=== Borderline cells (point estimate within ±2pp of floor) ===")
    for _, r in borderline.iterrows():
        verdict = ("PASS+CI" if r["ci_definitively_passes"]
                   else ("FAIL+CI" if r["ci_definitively_fails"]
                         else "AMBIGUOUS"))
        print(f"  {r['ckpt']:13s} {r['tau_kind']:10s} τ={r['tau']:.3f} {r['human']:14s} "
              f"{r['metric']:11s} n={r['n']:4d}  point={r['point']:.3f}  "
              f"CI=[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  width={r['ci_width']:.3f}  {verdict}")

    # Also print summary at per-ckpt τs for the key cells
    print("\n=== Headline cells (per-ckpt-calibrated τ) ===")
    key_cells = out_df[(out_df["tau_kind"] == "per_ckpt") & out_df["human"].isin(["dor", "Xinhe"])]
    for _, r in key_cells.iterrows():
        print(f"  {r['ckpt']:13s} τ={r['tau']:.3f} {r['human']:6s} {r['metric']:11s} "
              f"n={r['n']:4d}  point={r['point']:.3f}  CI=[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")


if __name__ == "__main__":
    main()
