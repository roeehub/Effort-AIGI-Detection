"""Probe 3 — Roy_D IQ characterization.

The joint-marginal audit couldn't place Roy_D on the under/over-covered partition
because he's absent from full_tags. He IS present in the atlas data (63 frames
across 4 pools, including 14 in train_teams_real_pool). This probe:

  (1) Identifies all Roy_D frames in atlas data
  (2) Computes summary stats on the 7 IQ axes for Roy_D
  (3) Places him on the under/over-covered map via density ratio (top-3 PCs)
  (4) Compares his pocket to dor_shkedi (the prototypical under-covered identity)
      and Xiang_Xiang2_Feng (over-covered)
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_TAB = "analysis/cpu_probes_2026-05-19/tables"
OUT_FIG = "analysis/cpu_probes_2026-05-19/figs"
os.makedirs(OUT_TAB, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

AXES = ["lap_var","luma_mean","color_a_dev","color_b_dev","saturation_mean","min_dim","skin_frac"]

print("Loading unified atlas dataframe ...")
df = pd.read_parquet("analysis/joint_marginal_audit_2026-05-19/artifacts/unified_tags.parquet")

# Identify Roy_D frames by frame_path
mask_royd = df["frame_path"].astype(str).str.contains("Roy_D|roy_d|RoyD", case=False, na=False)
royd = df[mask_royd].copy()
print(f"  Roy_D atlas frames: {len(royd)}; per pool: {royd['pool'].value_counts().to_dict()}")
print(f"  Roy_D per bucket: {royd['bucket'].value_counts().to_dict()}")

train_real = df[df["bucket"] == "train_real"]
lockbox_real = df[df["bucket"] == "lockbox_real"]
dev_real = df[df["bucket"] == "dev_real"]

# Per-axis summary
def summary_axes(name, sub):
    out = {"group": name, "n": len(sub)}
    for a in AXES:
        out[f"{a}_median"] = float(sub[a].median())
        out[f"{a}_q05"] = float(sub[a].quantile(0.05))
        out[f"{a}_q95"] = float(sub[a].quantile(0.95))
    return out

groups_for_summary = {
    "Roy_D (all)": royd,
    "Roy_D in train_teams_real_pool": royd[royd["pool"] == "train_teams_real_pool"],
    "Roy_D in dev pools": royd[royd["bucket"].isin(["dev_real", "dev_fake"])],
    "train_real (all)": train_real,
    "lockbox_real (all)": lockbox_real,
    "dev_real (all)": dev_real,
}
summary_rows = [summary_axes(name, g) for name, g in groups_for_summary.items() if len(g) > 0]
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{OUT_TAB}/probe3_royd_axis_summary.csv", index=False)

print("\n=== Roy_D vs train_real per-axis medians ===")
print(f"{'axis':<20} {'royd_med':>10} {'train_med':>11} {'train_q05':>11} {'train_q95':>11} {'royd_in_tail':>13}")
for a in AXES:
    rm = royd[a].median()
    tm = train_real[a].median()
    tq05 = train_real[a].quantile(0.05)
    tq95 = train_real[a].quantile(0.95)
    royd_tail_frac = ((royd[a] < tq05) | (royd[a] > tq95)).mean() * 100
    print(f"  {a:<18} {rm:>10.2f} {tm:>11.2f} {tq05:>11.2f} {tq95:>11.2f} {royd_tail_frac:>12.1f}%")

# Place Roy_D in 3-PC space (same projection used in audit)
all_real = df[df["bucket"].isin(["train_real","dev_real","lockbox_real"])][AXES].values.astype(float)
scaler = StandardScaler().fit(all_real)
all_real_scaled = scaler.transform(all_real)
pca = PCA(n_components=5, random_state=0).fit(all_real_scaled)

def project(sub):
    X = sub[AXES].values.astype(float)
    X = scaler.transform(X)
    return pca.transform(X)

P_train = project(train_real)
P_lock = project(lockbox_real)
P_royd = project(royd)
K = 3
kde_train = KernelDensity(bandwidth="scott", kernel="gaussian").fit(P_train[:, :K])
kde_lock = KernelDensity(bandwidth="scott", kernel="gaussian").fit(P_lock[:, :K])

# Density ratio at each Roy_D frame
d_train = np.exp(kde_train.score_samples(P_royd[:, :K]))
d_lock = np.exp(kde_lock.score_samples(P_royd[:, :K]))
ratio = d_train / np.maximum(d_lock, 1e-30)

print(f"\n=== Roy_D density ratio (train / lockbox-KDE) ===")
print(f"  median ratio: {np.median(ratio):.4f}")
print(f"  q25:          {np.quantile(ratio, 0.25):.4f}")
print(f"  q75:          {np.quantile(ratio, 0.75):.4f}")
print(f"  frac < 0.10:  {(ratio < 0.10).mean():.3f}")
print(f"  frac < 0.25:  {(ratio < 0.25).mean():.3f}")
print(f"  frac < 0.50:  {(ratio < 0.50).mean():.3f}")

# Comparison: dor_shkedi, Xiang_Xiang2_Feng, bla_bla_chow density ratios
# Use the chronic identity table from the audit
audit_table = pd.read_csv("analysis/joint_marginal_audit_2026-05-19/tables/per_chronic_identity_location.csv")
print("\n=== Comparison with audit's chronic identity table (median ratio sorted) ===")
print(audit_table[["identity_key","n","bucket","median_ratio","frac_below_0_10","frac_below_0_25"]].to_string(index=False))

# Visualize Roy_D position on PC1-PC2 scatter
fig, ax = plt.subplots(figsize=(11, 8))
ax.scatter(P_train[:, 0], P_train[:, 1], s=4, alpha=0.25, label=f"train_real (n={len(P_train)})", color="C0")
ax.scatter(P_lock[:, 0], P_lock[:, 1], s=4, alpha=0.40, label=f"lockbox_real (n={len(P_lock)})", color="C3")
ax.scatter(P_royd[:, 0], P_royd[:, 1], s=70, alpha=0.85, marker="*", color="black",
           edgecolor="white", linewidth=0.5, label=f"Roy_D (n={len(P_royd)})")
# Overlay chronic dor_shkedi for comparison
dor_mask = df["identity_key"].astype(str).str.contains("dor_shkedi", case=False, na=False)
P_dor = project(df[dor_mask])
ax.scatter(P_dor[:, 0], P_dor[:, 1], s=40, alpha=0.75, marker="s", color="C2",
           edgecolor="black", linewidth=0.3, label=f"dor_shkedi (n={len(P_dor)})")
# Overlay Xiang
xiang_mask = df["identity_key"].astype(str).str.contains("Xiang_Xiang2_Feng", case=False, na=False)
P_xiang = project(df[xiang_mask])
ax.scatter(P_xiang[:, 0], P_xiang[:, 1], s=40, alpha=0.65, marker="^", color="C1",
           edgecolor="black", linewidth=0.3, label=f"Xiang_Xiang2_Feng (n={len(P_xiang)})")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%}) — color/sat/face-fraction")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%}) — resolution/brightness")
ax.set_title("Roy_D position in 7-axis IQ PC space vs chronic comparison identities")
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/probe3_royd_pca_position.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# Per-axis distribution comparison plot
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes_flat = axes.flatten()
for i, axis_name in enumerate(AXES):
    ax = axes_flat[i]
    for name, sub, c, alpha in [
        ("train_real",   train_real,   "C0", 0.30),
        ("lockbox_real", lockbox_real, "C3", 0.30),
        ("Roy_D",        royd,         "black", 0.55),
    ]:
        if len(sub) == 0: continue
        v = sub[axis_name].values
        lo, hi = np.percentile(np.concatenate([train_real[axis_name].values,
                                                lockbox_real[axis_name].values]),
                                [1, 99])
        bins = np.linspace(lo, hi, 40)
        ax.hist(v, bins=bins, density=True, alpha=alpha, label=f"{name} (n={len(v)})", color=c)
    ax.set_title(axis_name)
    ax.legend(fontsize=8)
for j in range(len(AXES), len(axes_flat)):
    axes_flat[j].axis("off")
fig.suptitle("Roy_D per-axis IQ distribution vs train/lockbox", fontsize=14)
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/probe3_royd_per_axis.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# Summary JSON
summary = {
    "royd_total_atlas_frames": int(len(royd)),
    "royd_per_pool": royd["pool"].value_counts().to_dict(),
    "royd_per_bucket": royd["bucket"].value_counts().to_dict(),
    "royd_median_density_ratio_train_over_lockbox": float(np.median(ratio)),
    "royd_frac_ratio_below_0_10": float((ratio < 0.10).mean()),
    "royd_frac_ratio_below_0_25": float((ratio < 0.25).mean()),
    "royd_per_axis_tail_frac_outside_train_q05_q95": {
        a: float(((royd[a] < train_real[a].quantile(0.05)) |
                  (royd[a] > train_real[a].quantile(0.95))).mean())
        for a in AXES
    },
    "verdict": (
        "under-covered" if np.median(ratio) < 0.5 else
        "over-covered" if np.median(ratio) > 2.0 else
        "mixed / borderline"
    ),
}
with open(f"{OUT_TAB}/probe3_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== PROBE 3 SUMMARY ===")
print(json.dumps(summary, indent=2))
print("\nDONE.")
