"""Phase 1 — Quality-gate sweep using cached P8A scores.

No re-inference. Uses prob_fake_p8a from p8a_lockbox_join_2026-04-27.csv to
sweep gate criteria and report FPR / recall / coverage at each threshold.

Gates swept:
  G1  face_area_ratio >= threshold         (crop tightness)
  G2  clip_capture_mode in allowed set     (capture mode)
  G3  face_pixel_area >= threshold         (minimum face size)
  G4  NOT is_no_face AND NOT is_low_quality (basic quality floor)
  G5  Combined: G1 + G2 + G3 + G4

Outputs (analysis/quality_gate_2026-04-30/outputs/):
  phase1_gate_G1_sweep.csv
  phase1_gate_combined_sweep.csv
  phase1_fpr_coverage_plot.png
  phase1_capture_mode_table.csv
  phase1_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
REPO = Path("/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training")
CSV = REPO / "analysis/crop_shortcut_2026-04-27/p8a_lockbox_join_2026-04-27.csv"
OUT = REPO / "analysis/quality_gate_2026-04-30/outputs"
OUT.mkdir(parents=True, exist_ok=True)

# P8A lockbox-calibrated τ for ~5% FPR (documented in STATE_2026-04-30.md)
TAU = 0.9741
SCORE_COL = "prob_fake_p8a"

# Production-like capture modes (exclude webcam + screen captures)
PROD_MODES = {"normal_photo", "phone_screen"}


# ---------------------------------------------------------------------------
def fpr_recall(df: pd.DataFrame, tau: float) -> tuple[float, float, int, int]:
    reals = df[df["label"] == "real"][SCORE_COL].dropna()
    fakes = df[df["label"] == "fake"][SCORE_COL].dropna()
    fpr = (reals >= tau).mean() if len(reals) else float("nan")
    recall = (fakes >= tau).mean() if len(fakes) else float("nan")
    return float(fpr), float(recall), int(len(reals)), int(len(fakes))


def coverage(n_passing: int, n_total: int) -> float:
    return n_passing / n_total if n_total else float("nan")


def roc_auc(df: pd.DataFrame) -> float:
    from sklearn.metrics import roc_auc_score
    y = (df["label"] == "fake").astype(int)
    s = df[SCORE_COL]
    mask = s.notna() & y.notna()
    if mask.sum() < 2 or y[mask].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y[mask], s[mask]))


# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading {CSV}")
    raw = pd.read_csv(CSV, low_memory=False)
    print(f"  {len(raw)} rows, splits: {dict(raw['split'].value_counts())}")

    # Work on lockbox split for FPR headline; dev split for recall headline
    lockbox = raw[raw["split"] == "lockbox"].copy()
    dev = raw[raw["split"] == "dev"].copy()
    full = raw.copy()

    n_total_lb = len(lockbox)
    n_total_dev = len(dev)
    n_total_all = len(full)

    print(f"  Lockbox: {n_total_lb} frames  Dev: {n_total_dev} frames")
    print(f"  Calibrated τ = {TAU}  (score col = {SCORE_COL})")
    print()

    # -----------------------------------------------------------------------
    # Baseline (no gate)
    # -----------------------------------------------------------------------
    base_fpr_lb, base_rec_lb, nr_lb, nf_lb = fpr_recall(lockbox, TAU)
    base_fpr_dev, base_rec_dev, nr_dev, nf_dev = fpr_recall(dev, TAU)
    base_auc_lb = roc_auc(lockbox)
    print(f"BASELINE  lockbox: FPR={base_fpr_lb:.3f} recall={base_rec_lb:.3f} "
          f"(n_real={nr_lb} n_fake={nf_lb})  AUC={base_auc_lb:.4f}")
    print(f"BASELINE  dev:     FPR={base_fpr_dev:.3f} recall={base_rec_dev:.3f} "
          f"(n_real={nr_dev} n_fake={nf_dev})")
    print()

    # -----------------------------------------------------------------------
    # G1: face_area_ratio sweep
    # -----------------------------------------------------------------------
    print("G1 — face_area_ratio sweep ...")
    thresholds = np.linspace(0.05, 0.75, 60)
    g1_rows = []
    for thr in thresholds:
        for split_name, df_split, n_tot in [
            ("lockbox", lockbox, n_total_lb),
            ("dev",     dev,     n_total_dev),
        ]:
            mask = df_split["face_area_ratio"].notna() & (df_split["face_area_ratio"] >= thr)
            sub = df_split[mask]
            cov = coverage(len(sub), n_tot)
            fpr, rec, nr, nf = fpr_recall(sub, TAU)
            g1_rows.append({
                "split": split_name, "gate": "face_area_ratio",
                "threshold": round(float(thr), 4),
                "coverage": round(cov, 4),
                "n_passing": len(sub), "n_real": nr, "n_fake": nf,
                "fpr": round(fpr, 4), "recall": round(rec, 4),
            })

    g1_df = pd.DataFrame(g1_rows)
    g1_df.to_csv(OUT / "phase1_gate_G1_sweep.csv", index=False)
    print(f"  Saved {OUT / 'phase1_gate_G1_sweep.csv'}")

    # -----------------------------------------------------------------------
    # G2: capture mode breakdown (no threshold sweep — categorical)
    # -----------------------------------------------------------------------
    print("G2 — capture mode breakdown ...")
    mode_rows = []
    for split_name, df_split, n_tot in [("lockbox", lockbox, n_total_lb), ("dev", dev, n_total_dev)]:
        # Baseline (all modes)
        fpr, rec, nr, nf = fpr_recall(df_split, TAU)
        mode_rows.append({"split": split_name, "mode_filter": "all",
                           "n_real": nr, "n_fake": nf,
                           "coverage": round(coverage(nr + nf, n_tot), 4),
                           "fpr": round(fpr, 4), "recall": round(rec, 4)})
        # Per-mode
        for mode in sorted(df_split["clip_capture_mode"].dropna().unique()):
            sub = df_split[df_split["clip_capture_mode"] == mode]
            fpr, rec, nr, nf = fpr_recall(sub, TAU)
            mode_rows.append({"split": split_name, "mode_filter": mode,
                               "n_real": nr, "n_fake": nf,
                               "coverage": round(coverage(len(sub), n_tot), 4),
                               "fpr": round(fpr, 4), "recall": round(rec, 4)})
        # Production modes only
        sub = df_split[df_split["clip_capture_mode"].isin(PROD_MODES)]
        fpr, rec, nr, nf = fpr_recall(sub, TAU)
        mode_rows.append({"split": split_name, "mode_filter": "prod_modes_only",
                           "n_real": nr, "n_fake": nf,
                           "coverage": round(coverage(len(sub), n_tot), 4),
                           "fpr": round(fpr, 4), "recall": round(rec, 4)})
        # Exclude webcam
        sub = df_split[df_split["clip_capture_mode"] != "webcam"]
        fpr, rec, nr, nf = fpr_recall(sub, TAU)
        mode_rows.append({"split": split_name, "mode_filter": "exclude_webcam",
                           "n_real": nr, "n_fake": nf,
                           "coverage": round(coverage(len(sub), n_tot), 4),
                           "fpr": round(fpr, 4), "recall": round(rec, 4)})

    mode_df = pd.DataFrame(mode_rows)
    mode_df.to_csv(OUT / "phase1_capture_mode_table.csv", index=False)
    print(f"  Saved {OUT / 'phase1_capture_mode_table.csv'}")
    print()
    print("  Lockbox capture-mode breakdown:")
    print(mode_df[mode_df["split"] == "lockbox"][
        ["mode_filter", "n_real", "n_fake", "coverage", "fpr", "recall"]
    ].to_string(index=False))
    print()

    # -----------------------------------------------------------------------
    # G3: face_pixel_area sweep
    # -----------------------------------------------------------------------
    print("G3 — face_pixel_area sweep (lockbox only) ...")
    px_thresholds = [1000, 5000, 10000, 20000, 40000, 60000, 80000]
    g3_rows = []
    for thr in px_thresholds:
        mask = lockbox["face_pixel_area"].notna() & (lockbox["face_pixel_area"] >= thr)
        sub = lockbox[mask]
        fpr, rec, nr, nf = fpr_recall(sub, TAU)
        g3_rows.append({
            "gate": "face_pixel_area", "min_px": thr,
            "coverage": round(coverage(len(sub), n_total_lb), 4),
            "n_real": nr, "n_fake": nf,
            "fpr": round(fpr, 4), "recall": round(rec, 4),
        })
    g3_df = pd.DataFrame(g3_rows)
    print(g3_df[["min_px", "coverage", "n_real", "fpr", "recall"]].to_string(index=False))
    print()

    # -----------------------------------------------------------------------
    # G4: quality floor (is_no_face=False, is_low_quality=False)
    # -----------------------------------------------------------------------
    print("G4 — basic quality floor ...")
    qf_rows = []
    for split_name, df_split, n_tot in [("lockbox", lockbox, n_total_lb), ("dev", dev, n_total_dev)]:
        for label, mask in [
            ("drop_no_face", ~df_split["is_no_face"].fillna(False)),
            ("drop_low_quality", ~df_split["is_low_quality"].fillna(False)),
            ("drop_both", ~df_split["is_no_face"].fillna(False) & ~df_split["is_low_quality"].fillna(False)),
        ]:
            sub = df_split[mask]
            fpr, rec, nr, nf = fpr_recall(sub, TAU)
            qf_rows.append({"split": split_name, "quality_filter": label,
                             "coverage": round(coverage(len(sub), n_tot), 4),
                             "n_real": nr, "n_fake": nf,
                             "fpr": round(fpr, 4), "recall": round(rec, 4)})
    qf_df = pd.DataFrame(qf_rows)
    print(qf_df[qf_df["split"] == "lockbox"][
        ["quality_filter", "coverage", "n_real", "fpr", "recall"]
    ].to_string(index=False))
    print()

    # -----------------------------------------------------------------------
    # G5: Combined gate sweep (face_area_ratio + prod mode + quality floor)
    # -----------------------------------------------------------------------
    print("G5 — combined gate sweep (face_area_ratio + exclude_webcam + quality_floor) ...")
    combined_rows = []
    base_mask_lb = (
        (lockbox["clip_capture_mode"] != "webcam")
        & (~lockbox["is_no_face"].fillna(False))
        & (~lockbox["is_low_quality"].fillna(False))
    )
    base_mask_dev = (
        (dev["clip_capture_mode"] != "webcam")
        & (~dev["is_no_face"].fillna(False))
        & (~dev["is_low_quality"].fillna(False))
    )

    for thr in thresholds:
        for split_name, df_split, n_tot, base_mask in [
            ("lockbox", lockbox, n_total_lb, base_mask_lb),
            ("dev",     dev,     n_total_dev, base_mask_dev),
        ]:
            mask = base_mask & df_split["face_area_ratio"].notna() & (df_split["face_area_ratio"] >= thr)
            sub = df_split[mask]
            cov = coverage(len(sub), n_tot)
            fpr, rec, nr, nf = fpr_recall(sub, TAU)
            combined_rows.append({
                "split": split_name, "gate": "combined",
                "far_threshold": round(float(thr), 4),
                "coverage": round(cov, 4),
                "n_passing": len(sub), "n_real": nr, "n_fake": nf,
                "fpr": round(fpr, 4), "recall": round(rec, 4),
            })

    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv(OUT / "phase1_gate_combined_sweep.csv", index=False)
    print(f"  Saved {OUT / 'phase1_gate_combined_sweep.csv'}")

    # -----------------------------------------------------------------------
    # Print headline table: FPR / recall / coverage for key thresholds
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("HEADLINE: lockbox FPR / fake_recall / coverage at key thresholds")
    print("=" * 70)
    key_thrs = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    print(f"{'Gate':<40} {'far_thr':>7} {'coverage':>9} {'FPR':>7} {'recall':>7}")
    print("-" * 72)

    fpr, rec, _, _ = fpr_recall(lockbox, TAU)
    print(f"{'Baseline (no gate)':<40} {'—':>7} {'100%':>9} {fpr:.3f}  {rec:.3f}")

    sub = lockbox[lockbox["clip_capture_mode"] != "webcam"]
    fpr, rec, nr, nf = fpr_recall(sub, TAU)
    print(f"{'Exclude webcam':<40} {'—':>7} {coverage(len(sub),n_total_lb):>9.1%} {fpr:.3f}  {rec:.3f}")

    for thr in key_thrs:
        # G1 only
        mask = lockbox["face_area_ratio"].notna() & (lockbox["face_area_ratio"] >= thr)
        sub = lockbox[mask]
        fpr, rec, nr, nf = fpr_recall(sub, TAU)
        print(f"{'G1 far>=' + str(thr):<40} {thr:>7.2f} {coverage(len(sub),n_total_lb):>9.1%} {fpr:.3f}  {rec:.3f}")

    print()
    print("G5 combined (excl_webcam + quality + far_thr):")
    print(f"{'Gate':<40} {'far_thr':>7} {'coverage':>9} {'FPR':>7} {'recall':>7}")
    print("-" * 72)
    for thr in key_thrs:
        mask = base_mask_lb & lockbox["face_area_ratio"].notna() & (lockbox["face_area_ratio"] >= thr)
        sub = lockbox[mask]
        fpr, rec, nr, nf = fpr_recall(sub, TAU)
        print(f"{'G5 far>=' + str(thr):<40} {thr:>7.2f} {coverage(len(sub),n_total_lb):>9.1%} {fpr:.3f}  {rec:.3f}")

    # -----------------------------------------------------------------------
    # Plot: FPR vs coverage for G1 and G5 (lockbox)
    # -----------------------------------------------------------------------
    print()
    print("Generating plot ...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Quality-gate sweep — P8A on lockbox  (τ=0.9741)", fontsize=13)

    lb_g1 = g1_df[g1_df["split"] == "lockbox"].sort_values("threshold")
    lb_g5 = combined_df[combined_df["split"] == "lockbox"].sort_values("far_threshold")
    lb_g1_dev = g1_df[g1_df["split"] == "dev"].sort_values("threshold")
    lb_g5_dev = combined_df[combined_df["split"] == "dev"].sort_values("far_threshold")

    # Panel 1: FPR vs coverage
    ax = axes[0]
    ax.plot(lb_g1["coverage"], lb_g1["fpr"], "b-o", ms=3, label="G1 (far only)")
    ax.plot(lb_g5["coverage"], lb_g5["fpr"], "r-s", ms=3, label="G5 (combined)")
    ax.axhline(base_fpr_lb, color="gray", ls="--", lw=1, label=f"baseline FPR={base_fpr_lb:.3f}")
    ax.axhline(0.01, color="green", ls=":", lw=1, label="1% FPR target")
    ax.axhline(0.05, color="orange", ls=":", lw=1, label="5% FPR target")
    ax.set_xlabel("Coverage (fraction of lockbox frames passing gate)")
    ax.set_ylabel("FPR @ τ=0.9741")
    ax.set_title("FPR vs Coverage (lockbox reals)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.005, max(base_fpr_lb + 0.02, 0.15))
    ax.grid(True, alpha=0.3)

    # Panel 2: Recall vs coverage (dev fakes)
    ax = axes[1]
    ax.plot(lb_g1_dev["coverage"], lb_g1_dev["recall"], "b-o", ms=3, label="G1 (far only)")
    ax.plot(lb_g5_dev["coverage"], lb_g5_dev["recall"], "r-s", ms=3, label="G5 (combined)")
    ax.axhline(base_rec_dev, color="gray", ls="--", lw=1, label=f"baseline recall={base_rec_dev:.3f}")
    ax.set_xlabel("Coverage (fraction of dev frames passing gate)")
    ax.set_ylabel("Fake recall @ τ=0.9741")
    ax.set_title("Recall vs Coverage (dev fakes)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel 3: FPR-recall Pareto curve as gate tightens (G5, lockbox FPR vs dev recall)
    ax = axes[2]
    # Merge on far_threshold
    merged = lb_g5.merge(
        lb_g5_dev.rename(columns={"coverage": "cov_dev", "fpr": "fpr_dev", "recall": "recall_dev",
                                   "n_real": "nr_dev", "n_fake": "nf_dev", "n_passing": "n_pass_dev"}),
        on="far_threshold",
    )
    sc = ax.scatter(merged["fpr"], merged["recall_dev"], c=merged["far_threshold"],
                    cmap="viridis_r", s=20)
    plt.colorbar(sc, ax=ax, label="face_area_ratio threshold")
    ax.axvline(0.05, color="orange", ls=":", lw=1, label="5% FPR")
    ax.axvline(0.01, color="green", ls=":", lw=1, label="1% FPR")
    ax.set_xlabel("Lockbox FPR @ τ=0.9741")
    ax.set_ylabel("Dev fake recall @ τ=0.9741")
    ax.set_title("FPR–Recall Pareto (G5 combined gate)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = OUT / "phase1_fpr_coverage_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {plot_path}")

    # -----------------------------------------------------------------------
    # Summary JSON
    # -----------------------------------------------------------------------
    summary = {
        "tau": TAU,
        "score_col": SCORE_COL,
        "baseline_lockbox": {"fpr": round(base_fpr_lb, 4), "recall": round(base_rec_lb, 4),
                              "n_real": nr_lb, "n_fake": nf_lb, "auc": round(base_auc_lb, 4)},
        "baseline_dev": {"fpr": round(base_fpr_dev, 4), "recall": round(base_rec_dev, 4),
                         "n_real": nr_dev, "n_fake": nf_dev},
        "g2_capture_mode": mode_rows,
        "g3_pixel_area": g3_rows,
        "g4_quality_floor": qf_rows,
        "g5_highlights": [],
    }

    for thr in [0.25, 0.35, 0.45, 0.55]:
        mask = base_mask_lb & lockbox["face_area_ratio"].notna() & (lockbox["face_area_ratio"] >= thr)
        sub = lockbox[mask]
        fpr, rec, nr, nf = fpr_recall(sub, TAU)
        summary["g5_highlights"].append({
            "far_threshold": thr,
            "coverage": round(coverage(len(sub), n_total_lb), 4),
            "fpr": round(fpr, 4), "recall": round(rec, 4),
            "n_real": nr, "n_fake": nf,
        })

    with open(OUT / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {OUT / 'phase1_summary.json'}")
    print()
    print("Phase 1 complete.")


if __name__ == "__main__":
    main()
