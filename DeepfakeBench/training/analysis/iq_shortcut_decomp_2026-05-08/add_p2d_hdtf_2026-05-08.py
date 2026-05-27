"""Add P2D HDTF cells to the IQ R² decomposition (Stage 1 follow-up).

The original `decompose.py` had P2D's HDTF cells absent because Phase C
had not run for P2-D as of the 2026-05-08 morning authorship of
`IQ_DECOMP_FACTS_2026-05-08.md`. The 2026-05-08 evening Vertex run
`6632089555598049280` (`p2-d-step3000-hdtf-2026-05-08`) measured P2D on
the same HDTF substrate; reports are cached locally at
`analysis/p2_d_hdtf_2026-05-08/raw_reports/`.

This script unifies all 3 ckpts on the 2026-05-08 evening HDTF run for
the 4 HDTF pool groups (HDTF_CLEAN_DEV, HDTF_CLEAN_LOCKBOX,
HDTF_TEAMS_DEV, HDTF_TEAMS_LOCKBOX) and recomputes R² + residual AUC.

Outputs to `outputs/iq_decomp_hdtf_unified_2026-05-08.csv`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

THIS_DIR = Path(__file__).resolve().parent
ATLAS_PARQUET = THIS_DIR.parent / "iq_data_atlas_2026-05-08" / "outputs" / "per_frame.parquet"
LOCAL_REPORTS = THIS_DIR.parent / "p2_d_hdtf_2026-05-08" / "raw_reports"
OUTPUTS = THIS_DIR / "outputs"

CKPTS = {
    "P8A_REFERENCE_STEP5000": "p8a_reference_step5000",
    "E2B_TOP_N_STEP3200": "e2b_top_n_step3200",
    "P2_D_FOURIER_PERIODIC_STEP3000": "p2_d_fourier_periodic_step3000",
}

# (atlas_pool, label) → suite name in 2026-05-08 HDTF reports.
PHASE_C_SUITES = {
    "hdtf_real_clean_dev": (0, "proper_real_clean_dev"),
    "hdtf_real_clean_lockbox": (0, "proper_real_clean_lockbox"),
    "hdtf_real_teams_dev": (0, "proper_real_teams_dev"),
    "hdtf_real_teams_lockbox": (0, "proper_real_teams_lockbox"),
    "hdtf_fake_clean_dev": (1, "proper_fake_clean_all_dev"),
    "hdtf_fake_clean_lockbox": (1, "proper_fake_clean_all_lockbox"),
    "hdtf_fake_teams_dev": (1, "proper_fake_teams_all_dev"),
    "hdtf_fake_teams_lockbox": (1, "proper_fake_teams_all_lockbox"),
}

POOL_GROUPS = {
    "HDTF_CLEAN_DEV": ["hdtf_real_clean_dev", "hdtf_fake_clean_dev"],
    "HDTF_CLEAN_LOCKBOX": ["hdtf_real_clean_lockbox", "hdtf_fake_clean_lockbox"],
    "HDTF_TEAMS_DEV": ["hdtf_real_teams_dev", "hdtf_fake_teams_dev"],
    "HDTF_TEAMS_LOCKBOX": ["hdtf_real_teams_lockbox", "hdtf_fake_teams_lockbox"],
}

IQ_FEATURES = [
    "lap_var", "min_dim", "luma_mean", "color_b_dev", "edge_mag", "skin_frac",
]
IQ_FEATURES_EXPANDED = IQ_FEATURES + [
    "color_a_dev", "luma_std", "contrast_l", "saturation_mean",
]


def load_local_scores(ckpt_key: str, suite_basename: str) -> pd.DataFrame:
    suite_prefix = CKPTS[ckpt_key]
    path = LOCAL_REPORTS / f"{suite_basename}_{suite_prefix}_frames_report.csv"
    return pd.read_csv(path)


def build_panel() -> pd.DataFrame:
    print(f"[load atlas] {ATLAS_PARQUET}")
    atlas = pd.read_parquet(ATLAS_PARQUET)
    atlas = atlas[["frame_path", "pool"] + IQ_FEATURES_EXPANDED].copy()
    atlas = atlas.rename(columns={"pool": "atlas_pool"})

    rows = []
    for ckpt_key in CKPTS:
        for atlas_pool, (label, suite_basename) in PHASE_C_SUITES.items():
            scores = load_local_scores(ckpt_key, suite_basename)
            scores = scores[["frame_path", "frame_prob"]].copy()
            scores["ckpt"] = ckpt_key
            scores["atlas_pool"] = atlas_pool
            scores["label"] = label
            rows.append(scores)

    scored = pd.concat(rows, ignore_index=True)
    print(f"[scored] total rows pre-join: {len(scored):,}")
    joined = scored.merge(atlas, on=["frame_path", "atlas_pool"], how="inner")
    print(f"[joined] rows after inner join: {len(joined):,}")
    return joined


def fit_one_cell(df: pd.DataFrame, feats: list[str]) -> dict:
    X = df[feats].to_numpy(dtype=np.float64)
    y = df["frame_prob"].to_numpy(dtype=np.float64)
    labels = df["label"].to_numpy()
    reg = LinearRegression(n_jobs=1)
    reg.fit(X, y)
    y_hat = reg.predict(X)
    resid = y - y_hat
    r2 = reg.score(X, y)
    raw_auc = roc_auc_score(labels, y) if labels.min() == 0 and labels.max() == 1 else float("nan")
    resid_auc = roc_auc_score(labels, resid) if labels.min() == 0 and labels.max() == 1 else float("nan")
    return {
        "n_total": len(df),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "r2": float(r2),
        "raw_auc": float(raw_auc),
        "residual_auc": float(resid_auc),
        "score_p50_real": float(np.median(y[labels == 0])) if (labels == 0).any() else float("nan"),
        "score_p50_fake": float(np.median(y[labels == 1])) if (labels == 1).any() else float("nan"),
        "coef": {f: float(c) for f, c in zip(feats, reg.coef_)},
    }


def main():
    panel = build_panel()
    rows = []
    for ckpt in CKPTS:
        for pg_name, atlas_pools in POOL_GROUPS.items():
            sub = panel[(panel["ckpt"] == ckpt) & (panel["atlas_pool"].isin(atlas_pools))].copy()
            if len(sub) == 0 or sub["label"].nunique() < 2:
                continue
            for feat_set_name, feats in [
                ("primary_6", IQ_FEATURES),
                ("expanded_10", IQ_FEATURES_EXPANDED),
            ]:
                clean = sub.dropna(subset=feats + ["frame_prob"])
                if clean["label"].nunique() < 2 or len(clean) < 30:
                    continue
                row = {
                    "ckpt": ckpt,
                    "pool_group": pg_name,
                    "feature_set": feat_set_name,
                }
                row.update(fit_one_cell(clean, feats))
                row["coef"] = json.dumps(row["coef"])
                rows.append(row)

    out = OUTPUTS / "iq_decomp_hdtf_unified_2026-05-08.csv"
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out, index=False)
    print(f"[save] {out} rows={len(df_out)}")

    # Print headline table for the FACTS doc
    print()
    print("R² (primary_6) by ckpt × pool_group:")
    print(f"{'pool_group':<24} {'P8A':>10} {'E2B':>10} {'P2D':>10}")
    print("-" * 56)
    for pg_name in POOL_GROUPS:
        line = f"{pg_name:<24}"
        for ck in CKPTS:
            mask = (df_out["ckpt"] == ck) & (df_out["pool_group"] == pg_name) & (df_out["feature_set"] == "primary_6")
            if mask.any():
                r2 = df_out.loc[mask, "r2"].iloc[0]
                line += f" {r2:>10.4f}"
            else:
                line += f" {'n/a':>10}"
        print(line)

    print()
    print("Residual AUC (primary_6) by ckpt × pool_group:")
    print(f"{'pool_group':<24} {'P8A':>10} {'E2B':>10} {'P2D':>10}")
    print("-" * 56)
    for pg_name in POOL_GROUPS:
        line = f"{pg_name:<24}"
        for ck in CKPTS:
            mask = (df_out["ckpt"] == ck) & (df_out["pool_group"] == pg_name) & (df_out["feature_set"] == "primary_6")
            if mask.any():
                ra = df_out.loc[mask, "residual_auc"].iloc[0]
                line += f" {ra:>10.4f}"
            else:
                line += f" {'n/a':>10}"
        print(line)


if __name__ == "__main__":
    main()
