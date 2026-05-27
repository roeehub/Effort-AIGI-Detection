"""Probe 2 — Frozen-CLIP linear separability of real vs fake within the over-covered
chronic identities.

Hypothesis (from joint-marginal audit §4): the over-covered chronic identities
(bla_bla_chow, PC_Generator__s22, Xiang_Xiang2_Feng — those sitting in IQ-dense
training regions but still chronic-FP at inference) are failing for a non-IQ-gap
reason. Two readings:

  (a) AUC ≥ 0.99 in frozen CLIP: the information to separate real/fake is there;
      the FT recipe destroys it. → representation-side lever (anchor_aware,
      output-preservation loss, multi-layer-GRL).
  (b) AUC < 0.85 in frozen CLIP: the encoder doesn't see these particular fakes
      cleanly. → backbone/objective gap. (Would refute D2's chronic-6 AUC=1.000
      generalization at the per-sub-identity level.)

We also include the under-covered chronic identities (Chikara_Takahashi__s22,
dor_shkedi) as a comparison group.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_TAB = "analysis/cpu_probes_2026-05-19/tables"
OUT_FIG = "analysis/cpu_probes_2026-05-19/figs"
os.makedirs(OUT_TAB, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

print("Loading full_tags ...")
df = pd.read_parquet("analysis/lockbox_tagging/full_tags_2026-04-27.parquet")

# Identity groups from the audit
OVER_COVERED = ["bla_bla_chow", "PC_Generator__s22", "Xiang_Xiang2_Feng",
                "PC_Generator__s8", "dor_shkedi__s16", "PC_Generator__s3",
                "Cam_Test__s33", "PC_Generator__s4"]
UNDER_COVERED = ["Chikara_Takahashi__s22", "dor_shkedi",
                 "PC_Generator__s15", "PC_Generator__s14"]
# Also reference identities (non-chronic, used as a sanity check)
REFERENCE = ["Test_Cam__s41", "Md_noyn_Sharker__s15", "deeplive_dor"]


def probe_identity(idkey, df, feature_col="clip_embed", min_fakes=5, min_reals=5):
    """Return AUC under 5-fold CV of frozen feature linear probe for real vs fake,
    *within this identity_key*. Returns None if too few samples in either class.

    Falls back to leave-one-out if a fold gets fewer than 2 per class.
    """
    sub = df[df["identity_key"] == idkey].copy()
    n_fake = (sub["label"] == "fake").sum()
    n_real = (sub["label"] == "real").sum()
    if n_fake < min_fakes or n_real < min_reals:
        return {
            "identity_key": idkey, "n": len(sub),
            "n_real": int(n_real), "n_fake": int(n_fake),
            "auc_mean": None, "auc_std": None, "skipped": "too_few_per_class"
        }
    X = np.stack(sub[feature_col].values).astype(np.float32)
    y = (sub["label"] == "fake").astype(int).values
    # 5-fold (or LOO if too small)
    n_splits = min(5, n_fake, n_real)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    aucs = []
    for tr, te in skf.split(X, y):
        if len(np.unique(y[te])) < 2:
            continue
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, y[tr])
        prob = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y[te], prob))
    return {
        "identity_key": idkey, "n": len(sub),
        "n_real": int(n_real), "n_fake": int(n_fake),
        "auc_mean": float(np.mean(aucs)) if aucs else None,
        "auc_std": float(np.std(aucs)) if aucs else None,
        "n_folds": len(aucs),
        "skipped": None,
    }


groups = [("over-covered", OVER_COVERED), ("under-covered", UNDER_COVERED),
          ("reference",    REFERENCE)]

rows = []
for group_name, ids in groups:
    print(f"\n=== {group_name} ===")
    for idkey in ids:
        r = probe_identity(idkey, df, feature_col="clip_embed")
        r["group"] = group_name
        rows.append(r)
        if r["auc_mean"] is None:
            print(f"  {idkey:<28} n={r['n']:>4}  real={r['n_real']:>3}  fake={r['n_fake']:>3}  skipped={r['skipped']}")
        else:
            print(f"  {idkey:<28} n={r['n']:>4}  real={r['n_real']:>3}  fake={r['n_fake']:>3}  AUC={r['auc_mean']:.4f} ± {r['auc_std']:.4f}  (k={r['n_folds']})")

# Also run on ArcFace embeddings as comparison (does identity-embedding separate within-identity real vs fake?)
print("\n=== Same probe on ArcFace embedding (comparison; expect lower because ArcFace is identity-focused) ===")
af_rows = []
for group_name, ids in groups:
    for idkey in ids:
        r = probe_identity(idkey, df, feature_col="arcface_embed")
        r["group"] = group_name
        af_rows.append(r)
        if r["auc_mean"] is not None:
            print(f"  {idkey:<28} ArcFace-probe AUC={r['auc_mean']:.4f}")

# Save tables
clip_df = pd.DataFrame(rows)
af_df = pd.DataFrame(af_rows).rename(columns={"auc_mean": "arcface_auc_mean", "auc_std": "arcface_auc_std"})
merged = clip_df.merge(af_df[["identity_key", "arcface_auc_mean", "arcface_auc_std"]], on="identity_key", how="left")
merged.to_csv(f"{OUT_TAB}/probe2_per_identity_auc.csv", index=False)

# Aggregate per group
agg_rows = []
for group_name in ["over-covered", "under-covered", "reference"]:
    sub = clip_df[(clip_df["group"] == group_name) & clip_df["auc_mean"].notna()]
    if len(sub):
        agg_rows.append({
            "group": group_name,
            "n_identities": len(sub),
            "median_clip_auc": float(np.median(sub["auc_mean"])),
            "mean_clip_auc": float(np.mean(sub["auc_mean"])),
            "min_clip_auc": float(np.min(sub["auc_mean"])),
            "max_clip_auc": float(np.max(sub["auc_mean"])),
            "frac_above_0_99": float((sub["auc_mean"] >= 0.99).mean()),
            "frac_above_0_95": float((sub["auc_mean"] >= 0.95).mean()),
            "frac_below_0_85": float((sub["auc_mean"] < 0.85).mean()),
        })
agg = pd.DataFrame(agg_rows)
agg.to_csv(f"{OUT_TAB}/probe2_group_summary.csv", index=False)
print("\n=== Group summary ===")
print(agg.to_string(index=False))

# Figure
fig, ax = plt.subplots(figsize=(11, 6))
colors = {"over-covered": "C3", "under-covered": "C0", "reference": "gray"}
y_pos = 0
yticks, ylabels = [], []
for g in ["over-covered", "under-covered", "reference"]:
    sub = clip_df[clip_df["group"] == g].sort_values("auc_mean", na_position="last")
    for _, row in sub.iterrows():
        if row["auc_mean"] is None:
            continue
        ax.barh(y_pos, row["auc_mean"], xerr=row["auc_std"], color=colors[g], alpha=0.8)
        ax.text(row["auc_mean"] + 0.005, y_pos, f"{row['auc_mean']:.3f}", va="center", fontsize=8)
        yticks.append(y_pos)
        ylabels.append(f"{row['identity_key']} (n={row['n']})")
        y_pos += 1
    y_pos += 0.5  # space between groups
ax.axvline(0.99, color="green", linestyle="--", alpha=0.5, label="0.99 (info preserved → representation lever)")
ax.axvline(0.85, color="red", linestyle="--", alpha=0.5, label="0.85 (info gone → backbone gap)")
ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
ax.set_xlabel("Frozen-CLIP linear-probe AUC (real vs fake, within identity)")
ax.set_xlim(0.4, 1.05)
ax.set_title("Within-identity real/fake separability in frozen CLIP (probe2)")
ax.legend(loc="lower right")
# Group color legend
from matplotlib.patches import Patch
group_legend = [Patch(facecolor=c, label=g) for g, c in colors.items()]
leg2 = ax.legend(handles=group_legend, loc="upper left", title="group")
ax.add_artist(leg2)
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/probe2_within_identity_auc.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# Save summary JSON
summary = {
    "groups": {
        g: {
            "identities": [r for r in rows if r["group"] == g],
            "agg": next((a for a in agg_rows if a["group"] == g), None)
        }
        for g in ["over-covered", "under-covered", "reference"]
    }
}
with open(f"{OUT_TAB}/probe2_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nDONE.")
