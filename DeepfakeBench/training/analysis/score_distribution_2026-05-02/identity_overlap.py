"""Cross-reference identity strings between dev and lockbox real suites.

Dev has identities: dor, dor_shkedi, PC_Generator, Roy_D, Q, etc.
Lockbox has identities: dor_shkedi, real_dor, bla_bla_chow, Chikara_Takahashi, PC_Generator.

Question: which identity strings appear in BOTH dev and lockbox? Do the same
people appear under different naming conventions? Cross-reference all identity
labels across all real suites.

Output:
  outputs/identity_overlap_table.csv
  figures/identity_overlap_heatmap.png
"""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"

DEPLOYED_TAU = {"P8A": 0.990946, "P18T": 0.99359, "P18C": 0.994625}

REAL_SUITES = ["teams_real_all_dev", "teams_real_all_lockbox", "teams_real_dor_dev",
               "teams_real_lighting_extreme_dev", "teams_real_poor_quality_dev"]


def parse_identity(p: str) -> str:
    """Take the first __-separated chunk as the identity name.

    Lockbox frames have additional session/timestamp metadata between the
    identity and the frame number (e.g. Cam_Test__s32_308.0_frame_NNNN__hash.png).
    Dev frames usually have just identity__frame_NNNN.png. Splitting on the
    first '__' gives a consistent identity prefix.
    """
    name = p.rsplit("/", 1)[-1]
    if "__" in name:
        return name.split("__", 1)[0]
    return name.rsplit(".", 1)[0]


def main():
    df = pd.read_parquet(OUT / "combined_frames.parquet")
    real = df[df["suite"].isin(REAL_SUITES) & (df["model"] == "P8A")].copy()
    real["identity"] = real["frame_path"].apply(parse_identity)

    # Per-suite identity counts (frame-level)
    print("=== Identity strings per real suite ===")
    overlap = real.groupby(["suite", "identity"]).size().unstack(fill_value=0)
    print(overlap.to_string())
    overlap.to_csv(OUT / "identity_overlap_table.csv")

    # Identity in BOTH dev and lockbox?
    dev_ids = set(real[real["suite"] == "teams_real_all_dev"]["identity"].unique())
    lock_ids = set(real[real["suite"] == "teams_real_all_lockbox"]["identity"].unique())
    print(f"\nDev identities: {sorted(dev_ids)}")
    print(f"Lockbox identities: {sorted(lock_ids)}")
    print(f"\nIn BOTH dev and lockbox: {sorted(dev_ids & lock_ids)}")
    print(f"Dev only: {sorted(dev_ids - lock_ids)}")
    print(f"Lockbox only: {sorted(lock_ids - dev_ids)}")

    # Per-identity FPR across BOTH dev and lockbox
    print("\n=== Per-identity FPR per suite (deployed τ) ===")
    summary = []
    for ident in sorted(set(dev_ids | lock_ids)):
        row = {"identity": ident}
        for suite in REAL_SUITES:
            for model in ["P8A", "P18T", "P18C"]:
                sub = df[(df["suite"] == suite) & (df["model"] == model)]
                sub = sub[sub["frame_path"].apply(parse_identity) == ident]
                tau = DEPLOYED_TAU[model]
                if not sub.empty:
                    row[f"{suite[:30]}_{model}_FPR"] = float((sub["frame_prob"] >= tau).mean())
                    row[f"{suite[:30]}_n"] = int(len(sub))
        summary.append(row)
    sum_df = pd.DataFrame(summary)
    sum_df.to_csv(OUT / "identity_per_suite_fpr.csv", index=False)
    print("\nKey columns (FPR @ deployed τ for P8A):")
    fpr_cols = [c for c in sum_df.columns if c.endswith("_P8A_FPR")]
    n_cols = [c for c in sum_df.columns if c.endswith("_n")]
    show_cols = ["identity"] + fpr_cols
    print(sum_df[show_cols].to_string(index=False, float_format="%.3f"))

    # Heatmap: identity × suite, colored by P8A FPR
    fpr_data = []
    for ident in sorted(set(dev_ids | lock_ids)):
        row = []
        for suite in REAL_SUITES:
            sub = df[(df["suite"] == suite) & (df["model"] == "P8A")]
            sub = sub[sub["frame_path"].apply(parse_identity) == ident]
            tau = DEPLOYED_TAU["P8A"]
            row.append(float((sub["frame_prob"] >= tau).mean()) if not sub.empty else np.nan)
        fpr_data.append(row)
    fpr_arr = np.array(fpr_data)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(fpr_arr, cmap="Reds", vmin=0, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(REAL_SUITES)))
    ax.set_xticklabels(REAL_SUITES, rotation=30, ha="right")
    ax.set_yticks(range(len(set(dev_ids | lock_ids))))
    ax.set_yticklabels(sorted(set(dev_ids | lock_ids)))
    for i in range(fpr_arr.shape[0]):
        for j in range(fpr_arr.shape[1]):
            v = fpr_arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if v < 0.3 else "white")
            else:
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=7, color="grey")
    plt.colorbar(im, ax=ax, label="P8A FPR @ deployed τ")
    ax.set_title("P8A FPR per identity per real-suite")
    fig.tight_layout()
    fig.savefig(FIG / "identity_overlap_heatmap.png", dpi=120)
    plt.close(fig)
    print(f"\n[done] outputs in {OUT}")


if __name__ == "__main__":
    main()
