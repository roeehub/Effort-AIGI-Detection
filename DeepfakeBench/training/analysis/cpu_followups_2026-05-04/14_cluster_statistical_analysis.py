"""Statistical analysis of why viso fakes cluster the way they do.

Joins:
  - viso_per_frame_with_all_ckpts.csv (550 frames, scores per ckpt, caught_by_subset)
  - crop_attributes.csv (550 frames, image-level features: laplacian_var, luma_mean, skin_frac, ...)

Outputs:
  outputs/14a_cluster_feature_stats.csv     — per-cluster mean/std/median for each feature
  outputs/14b_uncaught_vs_caught_ttest.csv  — Welch's t-test: which features differ between
                                                 uncaught vs caught? Effect sizes (Cohen's d).
  outputs/14c_logistic_feature_importance.csv — coefficients from L1-regularized logistic
                                                  predicting caught/uncaught
  outputs/14d_pairwise_cluster_comparison.csv — pairwise t-tests between caught_by clusters
  outputs/14e_method_subtype_breakdown.csv  — viso has 2 subtypes: enhanced_raw vs enhanced_teams.
                                                 Does that explain anything?
"""
import csv
from pathlib import Path
import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).parent
OUT_DIR = ANALYSIS_DIR / "outputs"

# Load score data + caught_by labels
scores_df = pd.read_csv(OUT_DIR / "viso_per_frame_with_all_ckpts.csv")
print(f"Score data: {len(scores_df)} rows")

# Load crop attributes
crops_df = pd.read_csv("analysis/score_distribution_2026-05-02/outputs/crop_attributes.csv")
print(f"Crop attributes: {len(crops_df)} rows")

# Join via filename — extract filename from gs:// URI
scores_df["filename"] = scores_df["frame_path"].str.rsplit("/", n=1).str[-1]
joined = scores_df.merge(crops_df, on="filename", how="inner")
print(f"Joined: {len(joined)} rows")
if len(joined) < len(scores_df):
    missing = set(scores_df["filename"]) - set(crops_df["filename"])
    print(f"  WARNING: {len(missing)} score rows missing from crop_attributes")

FEATURE_COLS = ["h", "w", "luma_mean", "luma_std", "luma_p10", "luma_p90",
                "laplacian_var", "sobel_edge_mean", "saturation_mean", "skin_frac"]

# ============================================================
# 14a: per-cluster summary statistics
# ============================================================
print("\n=== 14a: per-cluster feature statistics ===")
groups = joined.groupby("caught_by_subset")
rows_a = []
for cluster_name, g in groups:
    for col in FEATURE_COLS:
        rows_a.append({
            "caught_by_subset": cluster_name,
            "n": len(g),
            "feature": col,
            "mean": float(g[col].mean()),
            "std": float(g[col].std()),
            "p10": float(g[col].quantile(0.10)),
            "p50": float(g[col].quantile(0.50)),
            "p90": float(g[col].quantile(0.90)),
        })
out_a = OUT_DIR / "14a_cluster_feature_stats.csv"
pd.DataFrame(rows_a).to_csv(out_a, index=False)
print(f"  → {out_a}")

# ============================================================
# 14b: uncaught vs caught — Welch's t-test + Cohen's d
# ============================================================
print("\n=== 14b: uncaught vs caught — t-test + Cohen's d ===")
from scipy import stats
joined["is_caught"] = joined["caught_by_subset"] != "uncaught"
caught = joined[joined["is_caught"]]
uncaught = joined[~joined["is_caught"]]
print(f"  caught: {len(caught)}, uncaught: {len(uncaught)}")

rows_b = []
for col in FEATURE_COLS:
    a = caught[col].values
    b = uncaught[col].values
    t, p = stats.ttest_ind(a, b, equal_var=False)
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d = float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else float("nan")
    rows_b.append({
        "feature": col,
        "caught_mean": float(a.mean()),
        "uncaught_mean": float(b.mean()),
        "delta_mean": float(a.mean() - b.mean()),
        "cohens_d": d,
        "abs_cohens_d": abs(d),
        "p_value": float(p),
        "significant_at_p_01": bool(p < 0.01),
    })
rows_b.sort(key=lambda r: -r["abs_cohens_d"])
out_b = OUT_DIR / "14b_uncaught_vs_caught_ttest.csv"
pd.DataFrame(rows_b).to_csv(out_b, index=False)
print(f"  → {out_b}")
print(f"\n  Top features distinguishing caught vs uncaught:")
print(f"  {'feature':20s} {'caught_mean':12s} {'uncaught_mean':14s} {'cohens_d':10s} {'p_value':10s}")
for r in rows_b[:6]:
    sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
    print(f"  {r['feature']:20s} {r['caught_mean']:12.3f} {r['uncaught_mean']:14.3f} {r['cohens_d']:+.3f}    {r['p_value']:.2e} {sig}")

# ============================================================
# 14c: L1 logistic regression — which features predict caught/uncaught?
# ============================================================
print("\n=== 14c: L1 logistic regression predicting caught/uncaught ===")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(joined[FEATURE_COLS].values)
y = joined["is_caught"].astype(int).values

# Per memory `feedback_sklearn_njobs.md`: n_jobs=1
clf = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=1000, random_state=42)
clf.fit(X, y)

# 5-fold cross-val accuracy for context
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=5, n_jobs=1)
print(f"  5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
print(f"  Baseline (always 'uncaught'): {(y == 0).mean():.3f}")

rows_c = []
for col, coef in sorted(zip(FEATURE_COLS, clf.coef_[0]), key=lambda x: -abs(x[1])):
    rows_c.append({
        "feature": col,
        "standardized_coefficient": float(coef),
        "abs_coefficient": abs(float(coef)),
        "interpretation": "↑ feature = ↑ caught" if coef > 0 else "↑ feature = ↓ caught" if coef < 0 else "(zeroed by L1)",
    })
out_c = OUT_DIR / "14c_logistic_feature_importance.csv"
pd.DataFrame(rows_c).to_csv(out_c, index=False)
print(f"  → {out_c}")
print(f"\n  Top features (L1 coefficients, standardized):")
for r in rows_c[:6]:
    print(f"  {r['feature']:20s} {r['standardized_coefficient']:+.3f}   {r['interpretation']}")

# ============================================================
# 14d: pairwise cluster comparison (most-distinguishing feature per pair)
# ============================================================
print("\n=== 14d: pairwise cluster t-tests ===")
clusters = sorted(joined["caught_by_subset"].unique())
rows_d = []
for i, c1 in enumerate(clusters):
    for c2 in clusters[i + 1:]:
        g1 = joined[joined["caught_by_subset"] == c1]
        g2 = joined[joined["caught_by_subset"] == c2]
        if len(g1) < 5 or len(g2) < 5:
            continue
        for col in FEATURE_COLS:
            a, b = g1[col].values, g2[col].values
            t, p = stats.ttest_ind(a, b, equal_var=False)
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) or 1e-9
            d = (a.mean() - b.mean()) / pooled
            rows_d.append({
                "cluster_a": c1,
                "cluster_b": c2,
                "n_a": len(g1),
                "n_b": len(g2),
                "feature": col,
                "mean_a": float(a.mean()),
                "mean_b": float(b.mean()),
                "cohens_d": float(d),
                "p_value": float(p),
            })
out_d = OUT_DIR / "14d_pairwise_cluster_comparison.csv"
pd.DataFrame(rows_d).to_csv(out_d, index=False)
print(f"  → {out_d}")

# Pretty-print: most-distinguishing feature per pair
print("\n  Most-distinguishing feature per cluster pair (|Cohen's d|):")
df_d = pd.DataFrame(rows_d)
df_d["abs_d"] = df_d["cohens_d"].abs()
top_per_pair = df_d.sort_values("abs_d", ascending=False).groupby(["cluster_a", "cluster_b"]).first().reset_index()
for _, r in top_per_pair.iterrows():
    print(f"  {r['cluster_a']:25s} vs {r['cluster_b']:25s}  → {r['feature']:18s} d={r['cohens_d']:+.2f} (p={r['p_value']:.1e})")

# ============================================================
# 14e: subtype breakdown (visomaster_enhanced_raw vs visomaster_enhanced_teams)
# ============================================================
print("\n=== 14e: viso subtype breakdown ===")
joined["subtype"] = joined["filename"].str.extract(r"(visomaster_enhanced_\w+?)__")[0]
sub_counts = joined.groupby(["subtype", "caught_by_subset"]).size().unstack(fill_value=0)
sub_counts["total"] = sub_counts.sum(axis=1)
out_e = OUT_DIR / "14e_method_subtype_breakdown.csv"
sub_counts.to_csv(out_e)
print(f"  → {out_e}")
print(sub_counts)

print("\nAll done.")
