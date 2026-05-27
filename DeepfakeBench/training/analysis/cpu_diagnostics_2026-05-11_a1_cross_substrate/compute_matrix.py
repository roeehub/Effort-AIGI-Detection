"""Build the cross-substrate AUC matrix for T4 + P8A + E2B.

Sources:
- dev/lockbox (already-scored teams suites):
  * `analysis/cpu_diagnostics_2026-05-10/_t4_scorecard_local/`
    `<suite>_<ckpt>_videos_report.csv` (for AUC on video-level scores)
- HDTF (subsampled 50 videos × 8 cells = 3200 frames):
  * `t4_hdtf_per_frame.csv`
  * `hdtf_subsample_p8a_e2b_match.csv`
- may5/may6 (no fakes; report mean + FPR@0.5):
  * `t4_may56_per_frame.csv` (T4)
  * `analysis/xinhe_cross_camera_audit_2026-05-06/outputs/scores_{P8A,E2B}.csv`

Outputs:
- `cross_substrate_auc.csv`
- `per_substrate_percentiles.csv`
- prints the headline tables for the FACTS doc
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]  # training/
SCORECARD_LOCAL = REPO_ROOT / "analysis" / "cpu_diagnostics_2026-05-10" / "_t4_scorecard_local"
XINHE_OUT = REPO_ROOT / "analysis" / "xinhe_cross_camera_audit_2026-05-06" / "outputs"

# === SECTION A: dev/lockbox suites (from local scorecard reports) =================
# Map (substrate, ckpt) → video-level CSV path. For "real" we use the corresponding
# *_videos_report.csv and for "fake" similarly; AUC is computed per (real_pool, fake_pool).
DEV_LOCKBOX_PAIRS = [
    # (substrate_key, real_suite, fake_suite)
    ("dev_teams_all", "teams_real_all_dev", "teams_fake_all_dev"),
    ("dev_teams_viso_enh", "teams_real_all_dev", "visomaster_enhanced_macro_dev"),
    ("dev_teams_deeplive_enh", "teams_real_all_dev", "deeplive_enhanced_dev"),
    ("lockbox_teams_all", "teams_real_all_lockbox", "teams_fake_all_lockbox"),
]

CKPTS_DEV_LOCKBOX = {
    "P8A_REFERENCE_STEP5000": "p8a_reference_step5000",
    "T4_LAMBDA1_TOP_N_STEP10500": "t4_lambda1_top_n_step10500",
    "T4_LAMBDA2_PERIODIC_STEP1500": "t4_lambda2_periodic_step1500",
}


def load_video_scores(suite_key: str, ckpt_dir_name: str) -> pd.DataFrame | None:
    """Return DataFrame of (video_id, avg_video_prob) or None if missing."""
    path = SCORECARD_LOCAL / f"{suite_key}_{ckpt_dir_name}_videos_report.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def auc_pair(real_df: pd.DataFrame, fake_df: pd.DataFrame, score_col: str = "avg_video_prob") -> dict:
    if real_df is None or fake_df is None or len(real_df) == 0 or len(fake_df) == 0:
        return {"auc": None, "n_real": 0, "n_fake": 0}
    y = np.concatenate([np.zeros(len(real_df)), np.ones(len(fake_df))])
    s = np.concatenate([real_df[score_col].values, fake_df[score_col].values])
    return {"auc": float(roc_auc_score(y, s)), "n_real": int(len(real_df)), "n_fake": int(len(fake_df))}


def compute_dev_lockbox():
    rows = []
    for substrate_key, real_suite, fake_suite in DEV_LOCKBOX_PAIRS:
        for ckpt_key, ckpt_dir in CKPTS_DEV_LOCKBOX.items():
            real_df = load_video_scores(real_suite, ckpt_dir)
            fake_df = load_video_scores(fake_suite, ckpt_dir)
            res = auc_pair(real_df, fake_df)
            rows.append({
                "substrate": substrate_key,
                "ckpt": ckpt_key,
                "auc": res["auc"],
                "n_real": res["n_real"],
                "n_fake": res["n_fake"],
                "real_p50": float(real_df["avg_video_prob"].quantile(0.50)) if real_df is not None and len(real_df) else None,
                "fake_p50": float(fake_df["avg_video_prob"].quantile(0.50)) if fake_df is not None and len(fake_df) else None,
            })
    return rows


# === SECTION B: HDTF (subsampled) =================================================
HDTF_PAIRS = [
    # substrate_key, real_cell, fake_cell
    ("hdtf_clean_dev", "hdtf_real_clean_dev", "hdtf_fake_clean_dev"),
    ("hdtf_teams_dev", "hdtf_real_teams_dev", "hdtf_fake_teams_dev"),
    ("hdtf_teams_lockbox", "hdtf_real_teams_lockbox", "hdtf_fake_teams_lockbox"),
    # visomaster_enhanced_teams uses the teams reals pool (same substrate).
    ("hdtf_viso_enh_teams_dev", "hdtf_real_teams_dev", "hdtf_viso_enh_teams_dev"),
    ("hdtf_viso_enh_teams_lockbox", "hdtf_real_teams_lockbox", "hdtf_viso_enh_teams_lockbox"),
]


def video_avg(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return df.groupby("video_id")[score_col].mean().reset_index().rename(columns={score_col: "avg_video_prob"})


def compute_hdtf():
    t4 = pd.read_csv(THIS_DIR / "t4_hdtf_per_frame.csv")
    ref = pd.read_csv(THIS_DIR / "hdtf_subsample_p8a_e2b_match.csv")

    rows = []
    for substrate_key, real_cell, fake_cell in HDTF_PAIRS:
        # T4
        t4_real = t4[t4["cell_key"] == real_cell]
        t4_fake = t4[t4["cell_key"] == fake_cell]
        t4_real_v = video_avg(t4_real, "t4_frame_prob")
        t4_fake_v = video_avg(t4_fake, "t4_frame_prob")
        res_t4 = auc_pair(t4_real_v, t4_fake_v, "avg_video_prob")

        # P8A
        ref_real = ref[ref["cell_key"] == real_cell]
        ref_fake = ref[ref["cell_key"] == fake_cell]
        p8a_real_v = video_avg(ref_real, "p8a_frame_prob")
        p8a_fake_v = video_avg(ref_fake, "p8a_frame_prob")
        res_p8a = auc_pair(p8a_real_v, p8a_fake_v, "avg_video_prob")

        # E2B
        e2b_real_v = video_avg(ref_real, "e2b_frame_prob")
        e2b_fake_v = video_avg(ref_fake, "e2b_frame_prob")
        res_e2b = auc_pair(e2b_real_v, e2b_fake_v, "avg_video_prob")

        for ckpt_key, res, real_v, fake_v in [
            ("T4_LAMBDA1_TOP_N_STEP10500", res_t4, t4_real_v, t4_fake_v),
            ("P8A_REFERENCE_STEP5000", res_p8a, p8a_real_v, p8a_fake_v),
            ("E2B_TOP_N_STEP3200", res_e2b, e2b_real_v, e2b_fake_v),
        ]:
            rows.append({
                "substrate": substrate_key,
                "ckpt": ckpt_key,
                "auc": res["auc"],
                "n_real": res["n_real"],
                "n_fake": res["n_fake"],
                "real_p50": float(real_v["avg_video_prob"].quantile(0.50)) if len(real_v) else None,
                "fake_p50": float(fake_v["avg_video_prob"].quantile(0.50)) if len(fake_v) else None,
            })
    return rows


# === SECTION C: may5 / may6 (no fakes) ============================================
def compute_may56():
    """Report mean(score), p50, FPR@τ=0.5 per (substrate, ckpt). All frames are reals;
    'FPR' here means fraction of frames scored ≥ 0.5 by the ckpt."""
    t4 = pd.read_csv(THIS_DIR / "t4_may56_per_frame.csv")
    p8a = pd.read_csv(XINHE_OUT / "scores_P8A.csv")
    e2b = pd.read_csv(XINHE_OUT / "scores_E2B.csv")

    rows = []
    for substrate_key in ["may5_correct", "may6_falseflag"]:
        for ckpt_key, df in [
            ("T4_LAMBDA1_TOP_N_STEP10500", t4),
            ("P8A_REFERENCE_STEP5000", p8a),
            ("E2B_TOP_N_STEP3200", e2b),
        ]:
            sub = df[df["population"] == substrate_key]
            if len(sub) == 0:
                continue
            scores = sub["prob_fake"].values
            rows.append({
                "substrate": substrate_key,
                "ckpt": ckpt_key,
                "auc": None,  # no fakes
                "n_real": int(len(scores)),
                "n_fake": 0,
                "real_p50": float(np.median(scores)),
                "fake_p50": None,
                "real_mean": float(np.mean(scores)),
                "fpr_tau_0.5": float((scores >= 0.5).mean()),
                "real_max": float(np.max(scores)),
            })
    return rows


# === SECTION D: percentiles =======================================================
PERCENTILES = [0.10, 0.25, 0.50, 0.75, 0.90]


def per_substrate_percentiles():
    """Per-substrate × per-ckpt score percentiles (real-only and fake-only when present)."""
    rows = []

    # dev/lockbox suites: use the videos_report.csv files
    DEV_LOCKBOX_SUITES = {
        "teams_real_all_dev": ("dev", "real"),
        "teams_real_all_lockbox": ("lockbox", "real"),
        "teams_fake_all_dev": ("dev_teams_all", "fake"),
        "teams_fake_all_lockbox": ("lockbox_teams_all", "fake"),
        "visomaster_enhanced_macro_dev": ("dev_viso_enh", "fake"),
        "deeplive_enhanced_dev": ("dev_deeplive_enh", "fake"),
    }
    for suite_key, (substrate, role) in DEV_LOCKBOX_SUITES.items():
        for ckpt_key, ckpt_dir in CKPTS_DEV_LOCKBOX.items():
            df = load_video_scores(suite_key, ckpt_dir)
            if df is None or len(df) == 0:
                continue
            scores = df["avg_video_prob"].values
            row = {"substrate": substrate, "role": role, "suite": suite_key, "ckpt": ckpt_key, "n": int(len(scores))}
            for q in PERCENTILES:
                row[f"p{int(q*100)}"] = float(np.quantile(scores, q))
            rows.append(row)

    # HDTF
    t4 = pd.read_csv(THIS_DIR / "t4_hdtf_per_frame.csv")
    ref = pd.read_csv(THIS_DIR / "hdtf_subsample_p8a_e2b_match.csv")
    for substrate_key, real_cell, fake_cell in HDTF_PAIRS:
        for cell, role in [(real_cell, "real"), (fake_cell, "fake")]:
            for ckpt_key, df, score_col in [
                ("T4_LAMBDA1_TOP_N_STEP10500", t4, "t4_frame_prob"),
                ("P8A_REFERENCE_STEP5000", ref, "p8a_frame_prob"),
                ("E2B_TOP_N_STEP3200", ref, "e2b_frame_prob"),
            ]:
                sub = df[df["cell_key"] == cell]
                if len(sub) == 0:
                    continue
                v = video_avg(sub, score_col)["avg_video_prob"].values
                row = {"substrate": substrate_key, "role": role, "suite": cell, "ckpt": ckpt_key, "n": int(len(v))}
                for q in PERCENTILES:
                    row[f"p{int(q*100)}"] = float(np.quantile(v, q))
                rows.append(row)

    # may5/may6
    t4_may = pd.read_csv(THIS_DIR / "t4_may56_per_frame.csv")
    p8a_may = pd.read_csv(XINHE_OUT / "scores_P8A.csv")
    e2b_may = pd.read_csv(XINHE_OUT / "scores_E2B.csv")
    for substrate_key in ["may5_correct", "may6_falseflag"]:
        for ckpt_key, df in [
            ("T4_LAMBDA1_TOP_N_STEP10500", t4_may),
            ("P8A_REFERENCE_STEP5000", p8a_may),
            ("E2B_TOP_N_STEP3200", e2b_may),
        ]:
            sub = df[df["population"] == substrate_key]
            if len(sub) == 0:
                continue
            scores = sub["prob_fake"].values
            row = {"substrate": substrate_key, "role": "real", "suite": substrate_key, "ckpt": ckpt_key, "n": int(len(scores))}
            for q in PERCENTILES:
                row[f"p{int(q*100)}"] = float(np.quantile(scores, q))
            rows.append(row)

    return rows


def main():
    print("=" * 70)
    print("Cross-substrate AUC matrix builder")
    print("=" * 70)

    auc_rows = []
    auc_rows.extend(compute_dev_lockbox())
    auc_rows.extend(compute_hdtf())
    auc_rows.extend(compute_may56())
    df_auc = pd.DataFrame(auc_rows)
    df_auc.to_csv(THIS_DIR / "cross_substrate_auc.csv", index=False)
    print(f"\n[save] cross_substrate_auc.csv n={len(df_auc)}")

    pct_rows = per_substrate_percentiles()
    df_pct = pd.DataFrame(pct_rows)
    df_pct.to_csv(THIS_DIR / "per_substrate_percentiles.csv", index=False)
    print(f"[save] per_substrate_percentiles.csv n={len(df_pct)}")

    # === Print headline tables for the FACTS doc ===
    print("\n" + "=" * 80)
    print("AUC matrix (video-level): rows=substrate, cols=ckpt")
    print("=" * 80)
    pivot = df_auc.pivot_table(index="substrate", columns="ckpt", values="auc", aggfunc="first")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "n/a"))

    print("\nΔ vs P8A (T4_L1 minus P8A) per substrate:")
    if "T4_LAMBDA1_TOP_N_STEP10500" in pivot.columns and "P8A_REFERENCE_STEP5000" in pivot.columns:
        delta = pivot["T4_LAMBDA1_TOP_N_STEP10500"] - pivot["P8A_REFERENCE_STEP5000"]
        for sub, d in delta.items():
            if pd.notna(d):
                print(f"  {sub:<35} Δ={d:+.4f}")

    # may56 special table
    print("\n" + "=" * 80)
    print("may5/may6 (no fakes): mean score + FPR@τ=0.5 per ckpt")
    print("=" * 80)
    may_rows = [r for r in auc_rows if r["substrate"].startswith("may")]
    df_may = pd.DataFrame(may_rows)
    for col in ["real_mean", "fpr_tau_0.5", "real_max"]:
        if col in df_may.columns:
            t = df_may.pivot_table(index="substrate", columns="ckpt", values=col, aggfunc="first")
            print(f"\n--- {col} ---")
            print(t.to_string(float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "n/a"))

    # Sample sizes
    print("\n" + "=" * 80)
    print("Sample sizes per (substrate, ckpt): n_real / n_fake")
    print("=" * 80)
    for r in auc_rows:
        if r["ckpt"] == "T4_LAMBDA1_TOP_N_STEP10500":
            print(f"  {r['substrate']:<35} n_real={r['n_real']:>6} n_fake={r['n_fake']:>6}")


if __name__ == "__main__":
    main()
