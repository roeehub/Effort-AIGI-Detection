"""Apply G2(200) gate to ALL frames (incl. unknown gate_status) by computing
W,H from the local image file. Then run per-pool analysis on the surviving
production-eligible frames.
"""
from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

OUT = Path(__file__).resolve().parent / "outputs"
df = pd.read_csv(OUT / "all_cohorts_scored.csv")
print(f"loaded {len(df)} frames")

# For each frame, get the actual image dims. If gate_status is already drop_*, skip work.
def derive_g2(row):
    if row["gate_status"] in ("pass", "drop_lowres", "drop_both", "drop_no_face"):
        # Already labeled — trust it
        return row["gate_status"] == "pass"
    # Unknown — read image
    p = row["local"]
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    return min(h, w) >= 200


t0 = time.time()
g2_results = []
for i, row in df.iterrows():
    g2_results.append(derive_g2(row))
    if (i+1) % 2000 == 0:
        print(f"  {i+1}/{len(df)} ({time.time()-t0:.1f}s)")
df["g2_pass"] = g2_results
print(f"\nG2 pass rate: {df['g2_pass'].sum()} / {len(df)} ({df['g2_pass'].mean()*100:.1f}%)")
print(f"per-suite G2-pass:")
print(df.groupby(["suite", "label", "g2_pass"]).size().to_string())

# Restrict to G2-passing
df_g2 = df[df["g2_pass"]].reset_index(drop=True)
print(f"\nG2-pass pool size: {len(df_g2)}")

# Save augmented manifest
df_g2.to_csv(OUT / "g2_pass_pool.csv", index=False)

# Per-pool analysis on G2-pass-only
POOL_DEFS = {
    "teams_dev": ["teams_real_all_dev", "teams_fake_all_dev"],
    "teams_lockbox": ["teams_real_all_lockbox", "teams_fake_all_lockbox"],
    "live_prod": ["live_reals_teams_prod", "live_fakes_teams_prod"],
    "dor_cross": ["dor_evening", "dor_morning", "dor_fake_local", "visomaster_v2_dor"],
}


def recall_at_fpr(scores, labels, target=0.05):
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = fpr <= target
    if ok.sum() == 0: return float("nan"), float("nan")
    idx = np.where(ok)[0][-1]
    return float(tpr[idx]), float(fpr[idx])


print()
print("=" * 100)
print("G2-FILTERED per-pool: AUC and recall@5%FPR")
print("=" * 100)
all_rows = []
for pool_name, suites in POOL_DEFS.items():
    sub = df_g2[df_g2["suite"].isin(suites)]
    if sub["label"].nunique() < 2:
        continue
    n_r = (sub["label"] == 0).sum()
    n_f = (sub["label"] == 1).sum()
    print(f"\n### {pool_name}  (G2-pass: {n_r} real / {n_f} fake)")
    for ckpt in ["T5C", "P8A"]:
        s_o = sub[f"{ckpt}_orig"].values
        s_b = sub[f"{ckpt}_blend_050"].values
        labels = sub["label"].values
        auc_o = roc_auc_score(labels, s_o)
        auc_b = roc_auc_score(labels, s_b)
        rec_o, fpr_o = recall_at_fpr(s_o, labels)
        rec_b, fpr_b = recall_at_fpr(s_b, labels)
        # Wilcoxon on per-frame Δ for the (correct-direction) test
        d = s_b - s_o
        d_real = d[labels == 0]
        d_fake = d[labels == 1]
        try:
            p_real = stats.wilcoxon(d_real, alternative="less").pvalue
            p_fake = stats.wilcoxon(d_fake, alternative="greater").pvalue
        except Exception:
            p_real = p_fake = float("nan")
        row = {
            "pool": pool_name, "ckpt": ckpt,
            "n_real": n_r, "n_fake": n_f,
            "AUC_orig": auc_o, "AUC_blend": auc_b, "ΔAUC": auc_b - auc_o,
            "rec@5_orig": rec_o, "rec@5_blend": rec_b, "Δrecall": rec_b - rec_o,
            "Δreal_median": float(np.median(d_real)),
            "Δfake_median": float(np.median(d_fake)),
            "wilcoxon_p_real": p_real,
            "wilcoxon_p_fake": p_fake,
        }
        all_rows.append(row)
        print(f"  {ckpt}: AUC {auc_o:.4f}→{auc_b:.4f} (Δ={auc_b-auc_o:+.4f})   "
              f"rec@5 {rec_o:.4f}→{rec_b:.4f} (Δ={rec_b-rec_o:+.4f})")
        print(f"        Δreal_med={np.median(d_real):+.4f} (p={p_real:.3e})   "
              f"Δfake_med={np.median(d_fake):+.4f} (p={p_fake:.3e})")

rd = pd.DataFrame(all_rows)
rd.to_csv(OUT / "g2_filtered_per_pool.csv", index=False)
print()
print("=" * 100)
print("CROSS-POOL SIGN TEST — does blend produce ΔAUC > 0 across pools?")
print("=" * 100)
for ckpt in ["T5C", "P8A"]:
    sub = rd[rd["ckpt"] == ckpt]
    n_pos = (sub["ΔAUC"] > 0).sum()
    n_neg = (sub["ΔAUC"] < 0).sum()
    pval = stats.binomtest(int(n_pos), int(n_pos + n_neg), p=0.5, alternative="greater").pvalue if n_pos + n_neg > 0 else float("nan")
    print(f"  {ckpt}: positive cohorts {n_pos} / negative {n_neg} / sign-test p={pval:.4f}")

print("\nDone.")
