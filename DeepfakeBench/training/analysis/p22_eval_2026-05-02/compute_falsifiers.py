"""P22 post-scorecard CPU analysis — 3 falsifiers vs P8A baseline.

Inputs:
  --grid          arena/promotion_contracts/<run>/promotion_contract/threshold_grid.csv
  --frames-dir    arena/promotion_contracts/<run>/reports/  (per-frame report CSVs)
  --attrs         analysis/score_distribution_2026-05-02/outputs/cross_suite_attributes.csv
  --out           analysis/p22_eval_2026-05-02/

Falsifiers (per CPU decision packet, 2026-05-02 PM):
  F1: |Pearson r(score, laplacian_var)| on dev viso drops by >= 0.15 vs P8A
  F2 (proxy): R²(score | {laplacian, luma, skin}) drops by >= 0.05 vs P8A
              [original "standalone LR AUC" is model-independent; we use the
              model-dependent residualization R² as the analogue]
  F3: dev viso recall at FPR=2% rises by >= 2.5pp vs P8A's ~1.1%
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ATTR_COLS = ["laplacian_var", "luma_mean", "skin_frac"]
P8A_BASELINE = {
    "viso_pearson_r": None,  # filled below
    "R2_score_attrs": 0.107,
    "AUC_orig": 0.843,
    "AUC_residual": 0.702,
    "viso_recall_at_2pct_fpr": 0.011,
}


def find_per_frame_csv(frames_dir: Path, suite: str, ckpt_key: str) -> Path | None:
    """Per-frame CSVs are named like <suite>_<ckpt_lower>_frames_report.csv.
    The arena writer lowercases checkpoint keys with `__` between words."""
    if not frames_dir.exists():
        return None
    pat = re.compile(rf"^{re.escape(suite)}_.*frames.*report\.csv$", re.IGNORECASE)
    candidates = [p for p in frames_dir.rglob("*.csv") if pat.match(p.name)]
    ck_lower = ckpt_key.lower()
    matched = [p for p in candidates if ck_lower in p.name.lower()]
    if matched:
        return matched[0]
    # Fallback: contains both suite + ckpt
    for p in candidates:
        if suite.lower() in p.name.lower() and ck_lower in p.name.lower():
            return p
    return None


def compute_pearson_with_laplacian(scores: pd.DataFrame, attrs: pd.DataFrame) -> Tuple[float, int]:
    merged = scores.merge(attrs[["frame_path", "laplacian_var"]], on="frame_path", how="inner")
    merged = merged.dropna(subset=["frame_prob", "laplacian_var"])
    if len(merged) < 30:
        return float("nan"), len(merged)
    r, _ = pearsonr(merged["frame_prob"], merged["laplacian_var"])
    return float(r), int(len(merged))


def compute_score_residualization(scores_by_suite: dict[str, pd.DataFrame],
                                  attrs: pd.DataFrame) -> dict:
    """Pool all suites for the residualization, mirroring 04_shortcut_strength.py."""
    parts: List[pd.DataFrame] = []
    for suite, df in scores_by_suite.items():
        merge = df.merge(attrs[["frame_path"] + ATTR_COLS + ["suite"]],
                         on="frame_path", how="inner", suffixes=("", "_attr"))
        if "suite_attr" in merge.columns:
            merge = merge.drop(columns=["suite_attr"])
        merge["suite"] = suite
        # Real if in *real* suite, fake if in *fake*/*deeplive*/visomaster suite
        if "real" in suite:
            merge["bin_label"] = 0
        else:
            merge["bin_label"] = 1
        parts.append(merge[["frame_path", "frame_prob", "bin_label"] + ATTR_COLS])
    if not parts:
        return {"n": 0}
    pool = pd.concat(parts, ignore_index=True).dropna(subset=ATTR_COLS + ["frame_prob"])
    if len(pool) < 50:
        return {"n": int(len(pool))}
    X = StandardScaler().fit_transform(pool[ATTR_COLS].to_numpy())
    s = pool["frame_prob"].to_numpy()
    y = pool["bin_label"].astype(int).to_numpy()
    lr = LinearRegression().fit(X, s)
    pred = lr.predict(X)
    resid = s - pred
    return {
        "n": int(len(pool)),
        "R2_score_explained_by_attrs": float(lr.score(X, s)),
        "AUC_orig_score": float(roc_auc_score(y, s)),
        "AUC_pred_from_attrs": float(roc_auc_score(y, pred)),
        "AUC_score_residual": float(roc_auc_score(y, resid)),
    }


def load_threshold_grid(grid_csv: Path) -> pd.DataFrame:
    return pd.read_csv(grid_csv)


def viso_recall_at_fpr_floor(grid: pd.DataFrame, ckpt_key: str, floor: float = 0.02) -> dict:
    sub = grid[grid.checkpoint_key == ckpt_key].copy()
    if len(sub) == 0:
        return {"recall_at_fpr": float("nan"), "tau": float("nan"), "actual_fpr": float("nan")}
    valid = sub[sub["dev_primary_real_fpr"] <= floor + 1e-12]
    if len(valid) == 0:
        chosen = sub.loc[sub["dev_primary_real_fpr"].idxmin()]
    else:
        chosen = valid.loc[valid["threshold"].idxmin()]
    recall = chosen.get("visomaster_enhanced_macro_dev__fake_recall", float("nan"))
    return {
        "recall_at_fpr": float(recall) if pd.notna(recall) else float("nan"),
        "tau": float(chosen["threshold"]),
        "actual_fpr": float(chosen["dev_primary_real_fpr"]),
    }


def _load_p8a_baseline(p8a_grid_csv: Path | None, attrs: pd.DataFrame) -> dict:
    """Re-derive P8A baseline numbers if a fresh grid path is provided; else use saved."""
    base = dict(P8A_BASELINE)
    # Pearson on viso always re-derived from cross_suite_attributes (has score_P8A column)
    sub = attrs.dropna(subset=["score_P8A", "laplacian_var"])
    sub = sub[sub.suite == "visomaster_enhanced_macro_dev"] if "visomaster_enhanced_macro_dev" in attrs.suite.unique() else sub
    if len(sub) >= 30:
        r, _ = pearsonr(sub["score_P8A"], sub["laplacian_var"])
        base["viso_pearson_r"] = float(r)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="threshold_grid.csv from new scorecard")
    ap.add_argument("--frames-dir", required=True, help="reports/ dir with per-frame CSVs")
    ap.add_argument("--attrs", default="analysis/score_distribution_2026-05-02/outputs/cross_suite_attributes.csv")
    ap.add_argument("--out", default="analysis/p22_eval_2026-05-02")
    ap.add_argument("--checkpoints", nargs="+", default=["P8A_REFERENCE_STEP5000",
                                                         "P22_AUG_STEP1000",
                                                         "P22_AUG_STEP4000",
                                                         "P22_AUG_STEP8000"])
    args = ap.parse_args()

    grid = load_threshold_grid(Path(args.grid))
    attrs = pd.read_csv(args.attrs)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.frames_dir)

    fake_suites = ["teams_fake_all_dev", "visomaster_enhanced_macro_dev", "deeplive_enhanced_dev"]
    real_suites = ["teams_real_all_dev"]
    all_suites = real_suites + fake_suites

    p8a_baseline = _load_p8a_baseline(Path(args.grid), attrs)

    rows = []
    residual_summary = []
    for ckpt in args.checkpoints:
        # F1: Pearson on viso
        viso_csv = find_per_frame_csv(frames_dir, "visomaster_enhanced_macro_dev", ckpt)
        if viso_csv is None:
            print(f"[WARN] viso per-frame CSV not found for {ckpt}, skipping F1")
            r_viso, n_viso = float("nan"), 0
        else:
            scores = pd.read_csv(viso_csv)
            r_viso, n_viso = compute_pearson_with_laplacian(scores, attrs)

        # F2 proxy: residualization R²
        scores_by_suite = {}
        for s in all_suites:
            csv = find_per_frame_csv(frames_dir, s, ckpt)
            if csv is None: continue
            scores_by_suite[s] = pd.read_csv(csv)
        resid = compute_score_residualization(scores_by_suite, attrs)
        residual_summary.append({"checkpoint": ckpt, **resid})

        # F3: recall at 2% FPR
        f3 = viso_recall_at_fpr_floor(grid, ckpt, floor=0.02)
        f3_5 = viso_recall_at_fpr_floor(grid, ckpt, floor=0.05)
        f3_10 = viso_recall_at_fpr_floor(grid, ckpt, floor=0.10)

        rows.append({
            "checkpoint": ckpt,
            "F1_pearson_r_score_laplacian_viso": r_viso,
            "F1_n_frames": n_viso,
            "F2_R2_score_explained_by_attrs": resid.get("R2_score_explained_by_attrs", float("nan")),
            "F2_AUC_orig": resid.get("AUC_orig_score", float("nan")),
            "F2_AUC_residual": resid.get("AUC_score_residual", float("nan")),
            "F3_viso_recall_at_2pct_fpr": f3["recall_at_fpr"],
            "F3_viso_recall_at_5pct_fpr": f3_5["recall_at_fpr"],
            "F3_viso_recall_at_10pct_fpr": f3_10["recall_at_fpr"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "falsifiers_per_checkpoint.csv", index=False)
    pd.DataFrame(residual_summary).to_csv(out_dir / "score_residual_per_checkpoint.csv", index=False)

    # Verdict: compare each P22 ckpt to P8A baseline
    print("=" * 80)
    print(f"P8A baseline: viso Pearson r = {p8a_baseline.get('viso_pearson_r', 'N/A')}")
    print(f"P8A baseline: R²(score|attrs) = {p8a_baseline['R2_score_attrs']:.3f}")
    print(f"P8A baseline: viso recall @ FPR=2% = {p8a_baseline['viso_recall_at_2pct_fpr']:.3f}")
    print("=" * 80)

    p8a_r = p8a_baseline.get("viso_pearson_r")
    verdicts = []
    for r in rows:
        if "P8A" in r["checkpoint"]: continue
        f1_pass = (p8a_r is not None
                   and pd.notna(r["F1_pearson_r_score_laplacian_viso"])
                   and abs(p8a_r) - abs(r["F1_pearson_r_score_laplacian_viso"]) >= 0.15)
        f2_pass = (pd.notna(r["F2_R2_score_explained_by_attrs"])
                   and p8a_baseline["R2_score_attrs"] - r["F2_R2_score_explained_by_attrs"] >= 0.05)
        f3_pass = (pd.notna(r["F3_viso_recall_at_2pct_fpr"])
                   and r["F3_viso_recall_at_2pct_fpr"] - p8a_baseline["viso_recall_at_2pct_fpr"] >= 0.025)
        score = sum([f1_pass, f2_pass, f3_pass])
        outcome = ("SUCCESS" if score >= 2 else "AMBIGUOUS" if score == 1 else "DID NOT BITE")
        verdicts.append({"checkpoint": r["checkpoint"], "F1": f1_pass, "F2": f2_pass, "F3": f3_pass,
                          "score": score, "verdict": outcome})
        print(f"\n{r['checkpoint']}:")
        print(f"  F1 |Δ Pearson r| ≥ 0.15: {f1_pass}  (P22={r['F1_pearson_r_score_laplacian_viso']:.3f}, P8A={p8a_r})")
        print(f"  F2 R² drops ≥ 0.05    : {f2_pass}  (P22={r['F2_R2_score_explained_by_attrs']:.3f}, P8A={p8a_baseline['R2_score_attrs']:.3f})")
        print(f"  F3 viso@2% rises ≥ 2.5pp: {f3_pass}  (P22={r['F3_viso_recall_at_2pct_fpr']:.3f}, P8A={p8a_baseline['viso_recall_at_2pct_fpr']:.3f})")
        print(f"  → {outcome} ({score}/3)")

    pd.DataFrame(verdicts).to_csv(out_dir / "verdicts.csv", index=False)
    with open(out_dir / "p8a_baseline.json", "w") as f:
        json.dump(p8a_baseline, f, indent=2)
    print(f"\nWrote: {out_dir}/falsifiers_per_checkpoint.csv")
    print(f"Wrote: {out_dir}/verdicts.csv")


if __name__ == "__main__":
    main()
