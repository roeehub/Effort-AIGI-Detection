"""Job C — F2 reals-only sanity check.

Compute R²(score | {laplacian_var, luma_mean, skin_frac}) on REALS ONLY for
each checkpoint. If P22's R² rose on reals-only, the F2 increase is a real
shortcut shift. If R² stayed flat or dropped on reals-only while pooled R²
rose, F2's pooled rise is a between-class variance artifact.
"""
from __future__ import annotations

import sys; sys.path.insert(0, "analysis/p22_eval_2026-05-02/cpu_followups/scripts")

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from _common import OUT, ATTRS_CSV, load_per_frame_scores

CKPTS = ["P8A_REFERENCE_STEP5000", "P18T_GRL_TREATMENT_STEP4000",
         "P22_AUG_STEP1000", "P22_AUG_STEP4000", "P22_AUG_STEP8000"]

ATTR_COLS = ["laplacian_var", "luma_mean", "skin_frac"]
REAL_SUITES_FOR_R2 = ["teams_real_all_dev"]  # most populated real suite in attrs (n=198)
FAKE_SUITES_FOR_R2 = ["teams_fake_all_dev", "deeplive_enhanced_dev"]


def r2_on_subset(scores_df, attrs):
    """Fit LinearRegression(attrs -> score), return R² and n."""
    merged = scores_df.merge(attrs[["frame_path"] + ATTR_COLS],
                              on="frame_path", how="inner")
    merged = merged.dropna(subset=ATTR_COLS + ["frame_prob"])
    if len(merged) < 30:
        return float("nan"), len(merged), float("nan")
    X = StandardScaler().fit_transform(merged[ATTR_COLS].to_numpy())
    s = merged["frame_prob"].to_numpy()
    lr = LinearRegression().fit(X, s)
    r2 = lr.score(X, s)
    var_score = float(np.var(s))
    return float(r2), int(len(merged)), var_score


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    attrs = pd.read_csv(ATTRS_CSV)

    rows = []
    for ckpt in CKPTS:
        # Reals only
        for suite in REAL_SUITES_FOR_R2:
            df = load_per_frame_scores(ckpt, suite)
            if df is None: continue
            r2, n, var = r2_on_subset(df, attrs)
            rows.append({"ckpt": ckpt, "scope": f"reals_only:{suite}",
                         "n": n, "R2": r2, "var_score": var})

        # Fakes only
        fake_parts = []
        for suite in FAKE_SUITES_FOR_R2:
            df = load_per_frame_scores(ckpt, suite)
            if df is not None: fake_parts.append(df)
        if fake_parts:
            fakes = pd.concat(fake_parts, ignore_index=True)
            r2, n, var = r2_on_subset(fakes, attrs)
            rows.append({"ckpt": ckpt, "scope": "fakes_pooled",
                         "n": n, "R2": r2, "var_score": var})

        # Pool: real + fakes (this is what the original F2 used)
        all_parts = [load_per_frame_scores(ckpt, s) for s in
                     REAL_SUITES_FOR_R2 + FAKE_SUITES_FOR_R2]
        all_parts = [d for d in all_parts if d is not None]
        if all_parts:
            pool = pd.concat(all_parts, ignore_index=True)
            r2, n, var = r2_on_subset(pool, attrs)
            rows.append({"ckpt": ckpt, "scope": "pool_real+fake",
                         "n": n, "R2": r2, "var_score": var})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "03_f2_reals_only.csv", index=False)

    print("=" * 80)
    print("R²(score | attrs) — reals-only vs fakes-only vs pooled")
    print("=" * 80)
    pivot = df.pivot(index="ckpt", columns="scope", values="R2")
    pivot = pivot.reindex(CKPTS)
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "—"))

    print("\nVar(score) per (ckpt, scope) — captures distribution width on each subset:")
    pivot_var = df.pivot(index="ckpt", columns="scope", values="var_score")
    pivot_var = pivot_var.reindex(CKPTS)
    print(pivot_var.to_string(float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "—"))

    print("\nN per (ckpt, scope):")
    pivot_n = df.pivot(index="ckpt", columns="scope", values="n").reindex(CKPTS)
    print(pivot_n.to_string(float_format=lambda x: f"{int(x)}" if pd.notna(x) else "—"))

    # Interpretation
    p8a = df[df.ckpt == "P8A_REFERENCE_STEP5000"]
    p22_8k = df[df.ckpt == "P22_AUG_STEP8000"]
    print("\nINTERPRETATION:")
    for scope in ["reals_only:teams_real_all_dev", "fakes_pooled", "pool_real+fake"]:
        a = p8a[p8a.scope == scope].R2.iloc[0] if (p8a.scope == scope).any() else None
        b = p22_8k[p22_8k.scope == scope].R2.iloc[0] if (p22_8k.scope == scope).any() else None
        if a is not None and b is not None:
            print(f"  {scope:36s}  P8A R²={a:.3f}  →  P22 step8k R²={b:.3f}  Δ={b-a:+.3f}")


if __name__ == "__main__":
    main()
