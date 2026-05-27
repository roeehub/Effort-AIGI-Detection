"""
Shortcut strength quantification.

Two complementary measurements:

(a) "How strong is the shortcut as a standalone classifier?"
    Train sklearn LR using ONLY {laplacian_var, luma_mean, skin_frac} as
    features on (real, fake) labels per-suite. Report 5-fold CV AUC.
    If AUC ~ 0.5 → shortcut is weak (noise). If AUC ~ 0.85+ → the model
    could plausibly classify by attributes alone — strong shortcut.

(b) "How much of P8A's score is shortcut-attributable?"
    Regress score on attributes (linear). Take residual. Compute residual-
    based recall vs FPR curve. Compare to original. If the residual
    score still discriminates → P8A has signal beyond the shortcut.
    If it collapses → P8A is mostly a sharpness detector.

Critical for confidence: tells us whether *fixing* the shortcut would leave
the model with anything.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ATTR_PATH = Path("analysis/score_distribution_2026-05-02/outputs/cross_suite_attributes.csv")
OUT = Path("analysis/cpu_decision_2026-05-02_pm_late/outputs")
FIG = Path("analysis/cpu_decision_2026-05-02_pm_late/figures")

df = pd.read_csv(ATTR_PATH)
print(f"loaded {len(df)} frames with attributes")
print("suites:", df.suite.unique())
print("tags:", df.tag.unique() if "tag" in df.columns else "n/a")
print()

# Real / fake mapping per suite
suite_label = {
    "teams_real_all_dev": 0,
    "teams_real_dor_dev": 0,
    "teams_real_all_lockbox": 0,
    "teams_fake_all_dev": 1,
    "teams_fake_all_lockbox": 1,
    "deeplive_enhanced_dev": 1,
}
df["bin_label"] = df.suite.map(suite_label)
df_l = df.dropna(subset=["bin_label"]).copy()
print(f"{len(df_l)} frames have a binary label (real=0/fake=1)")

attr_cols = ["laplacian_var", "luma_mean", "skin_frac"]
df_l = df_l.dropna(subset=attr_cols)
print(f"{len(df_l)} frames after dropna on attrs")
print(f"  reals (label=0): {(df_l.bin_label==0).sum()}")
print(f"  fakes (label=1): {(df_l.bin_label==1).sum()}")
print()

# (a) Shortcut as standalone classifier
print("=== (a) Standalone-shortcut classifier ===")
X = df_l[attr_cols].to_numpy()
y = df_l.bin_label.astype(int).to_numpy()
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

clf = LogisticRegression(max_iter=2000)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_s, y, cv=cv, scoring="roc_auc")
print(f"5-fold AUC using ONLY {attr_cols}: mean={scores.mean():.4f}  std={scores.std():.4f}")

# Marginals
for col in attr_cols:
    Xs = StandardScaler().fit_transform(df_l[[col]].to_numpy())
    s = cross_val_score(clf, Xs, y, cv=cv, scoring="roc_auc")
    print(f"  with ONLY {col:20s}: mean AUC={s.mean():.4f}")

# Per-suite real-vs-fake separability using only attrs
print("\n=== Per-suite separability (real-vs-fake real_anchor=teams_real_all_dev) ===")
real = df_l[df_l.suite == "teams_real_all_dev"]
out_per_suite = []
for fake_suite in ["teams_fake_all_dev", "teams_fake_all_lockbox", "deeplive_enhanced_dev"]:
    fake = df_l[df_l.suite == fake_suite]
    pair = pd.concat([real, fake], ignore_index=True)
    if len(pair) < 30:
        continue
    Xp = StandardScaler().fit_transform(pair[attr_cols].to_numpy())
    yp = pair.bin_label.astype(int).to_numpy()
    cvf = StratifiedKFold(n_splits=min(5, min(np.bincount(yp))), shuffle=True, random_state=42)
    s = cross_val_score(clf, Xp, yp, cv=cvf, scoring="roc_auc")
    out_per_suite.append({"fake_suite": fake_suite, "n_real": len(real), "n_fake": len(fake), "auc_attrs_only": s.mean()})
    print(f"  {fake_suite:30s} n_real={len(real)} n_fake={len(fake)} AUC(attrs only) = {s.mean():.4f}")
pd.DataFrame(out_per_suite).to_csv(OUT / "shortcut_per_suite_auc.csv", index=False)

# (b) How much of P8A's score is shortcut-attributable
print("\n=== (b) Score residual after partialing out attrs ===")
results = []
for model_col, label_col in [("score_P8A", "label_P8A"), ("score_P18T", "label_P18T"), ("score_P18C", "label_P18C")]:
    sub = df_l.dropna(subset=[model_col]).copy()
    if len(sub) == 0:
        continue
    X_sub = StandardScaler().fit_transform(sub[attr_cols].to_numpy())
    s = sub[model_col].to_numpy()
    # Predict score from attributes
    lr = LinearRegression().fit(X_sub, s)
    pred = lr.predict(X_sub)
    resid = s - pred
    sub["score_pred_from_attrs"] = pred
    sub["score_residual"] = resid
    # AUC of (predicted-from-attrs) score
    auc_attrs = roc_auc_score(sub.bin_label, pred)
    auc_orig = roc_auc_score(sub.bin_label, s)
    auc_resid = roc_auc_score(sub.bin_label, resid)
    r2 = lr.score(X_sub, s)
    results.append({
        "model": model_col.replace("score_", ""),
        "n": len(sub),
        "R2_score_explained_by_attrs": r2,
        "AUC_orig_score": auc_orig,
        "AUC_pred_from_attrs": auc_attrs,
        "AUC_score_residual": auc_resid,
    })
    print(f"  {model_col}: n={len(sub)}  R²(score|attrs)={r2:.4f}  AUC orig={auc_orig:.4f}  AUC pred={auc_attrs:.4f}  AUC residual={auc_resid:.4f}")

res_df = pd.DataFrame(results)
res_df.to_csv(OUT / "shortcut_score_residual.csv", index=False)


# Plot: standalone shortcut AUC distribution per suite
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.bar([r["fake_suite"].replace("_dev", "").replace("_lockbox", "_LB") for r in out_per_suite],
       [r["auc_attrs_only"] for r in out_per_suite], color="tab:purple")
ax.axhline(0.5, color="grey", linestyle="--")
ax.axhline(0.7, color="orange", linestyle="--", alpha=0.6)
ax.axhline(0.85, color="red", linestyle="--", alpha=0.6)
ax.set_ylabel("AUC")
ax.set_title("Sharpness/luma/skin — standalone\nfake classifier AUC (per suite)")
ax.set_ylim(0.4, 1.0)
plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
labels = [r["model"] for r in results]
ax.bar(labels, [r["AUC_orig_score"] for r in results], width=0.25, label="original score", color="tab:blue")
ax.bar(np.arange(len(labels)) + 0.25, [r["AUC_pred_from_attrs"] for r in results], width=0.25, label="predicted from attrs", color="tab:purple")
ax.bar(np.arange(len(labels)) + 0.50, [r["AUC_score_residual"] for r in results], width=0.25, label="residual", color="tab:green")
ax.set_xticks(np.arange(len(labels)) + 0.25)
ax.set_xticklabels(labels)
ax.axhline(0.5, color="grey", linestyle="--")
ax.set_ylabel("AUC")
ax.set_title("Score decomposition — attrs vs residual\n(if residual AUC < orig, shortcut explains some signal)")
ax.legend(fontsize=9, loc="lower left")
ax.grid(alpha=0.3)
ax.set_ylim(0.4, 1.0)

ax = axes[2]
ax.bar(labels, [r["R2_score_explained_by_attrs"] for r in results], color="tab:red")
ax.set_ylabel("R²")
ax.set_title("Fraction of score variance explained by\nlaplacian+luma+skin (3 attrs only)")
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.0)

plt.suptitle("Shortcut strength quantification", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG / "shortcut_strength.png", dpi=140, bbox_inches="tight")
print(f"\nwrote {FIG / 'shortcut_strength.png'}")
