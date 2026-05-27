"""Visualizations for the viewer:
  figures/score_histogram_<suite>.png      — overlaid real/fake distributions for all 3 ckpts
  figures/roc_curve_<fake_suite>.png       — ROC curves for all 3 ckpts (vs teams_real_all_dev)
  figures/recall_vs_fpr_<fake_suite>.png   — calibration view (per-ckpt)
  figures/disagreement_scatter_<suite>.png — P8A vs E2B_3200 score scatter, colored by label

Also produces small CSVs in viewer_artifacts/ for the dashboard.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANALYSIS_DIR = Path(__file__).parent
RAW_DIR = ANALYSIS_DIR / "raw_reports"
FIG_DIR = ANALYSIS_DIR / "figures"
VIEWER_DIR = ANALYSIS_DIR / "viewer_artifacts"
FIG_DIR.mkdir(exist_ok=True)
VIEWER_DIR.mkdir(exist_ok=True)

CKPTS = {
    "P8A": "p8a_reference_step5000",
    "E2B_3200": "e2b_top_n_step3200",
    "E3_6600": "e3_top_n_step6600",
}
COLORS = {"P8A": "tab:blue", "E2B_3200": "tab:orange", "E3_6600": "tab:green"}

REAL_NEG = "teams_real_all_dev"
FAKE_SUITES = ["visomaster_enhanced_macro_dev", "deeplive_enhanced_dev", "teams_fake_all_dev", "teams_fake_all_lockbox"]
ALL_SUITES = [REAL_NEG, "teams_real_poor_quality_dev", "teams_real_lighting_extreme_dev",
              "teams_real_all_lockbox", "teams_real_dor_dev"] + FAKE_SUITES


def load_scores(suite, ckpt_token):
    f = RAW_DIR / f"{suite}_{ckpt_token}_frames_report.csv"
    if not f.exists():
        return np.array([]), np.array([])
    s, l = [], []
    with f.open() as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            s.append(float(row["frame_prob"]))
            l.append(int(row["label"]))
    return np.array(s), np.array(l)


# ============================================================
# 1. Score-distribution histograms (per-suite, overlay 3 ckpts)
# ============================================================
def plot_histograms():
    print("Histograms…")
    for suite in ALL_SUITES:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
        for i, (ckpt_name, ckpt_token) in enumerate(CKPTS.items()):
            scores, labels = load_scores(suite, ckpt_token)
            ax = axes[i]
            if len(scores) == 0:
                ax.set_title(f"{ckpt_name} on {suite}\n(no data)")
                continue
            real_s = scores[labels == 0]
            fake_s = scores[labels == 1]
            bins = np.linspace(0, 1, 51)
            if len(real_s) > 0:
                ax.hist(real_s, bins=bins, alpha=0.6, color="tab:green", label=f"real (n={len(real_s)})", density=True)
            if len(fake_s) > 0:
                ax.hist(fake_s, bins=bins, alpha=0.6, color="tab:red", label=f"fake (n={len(fake_s)})", density=True)
            ax.set_title(f"{ckpt_name}")
            ax.set_xlabel("score")
            ax.set_xlim(0, 1)
            ax.legend(fontsize=8)
        fig.suptitle(f"Score distribution: {suite}")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"score_histogram_{suite}.png", dpi=80)
        plt.close(fig)
    print(f"  {len(ALL_SUITES)} histogram PNGs in {FIG_DIR}")


# ============================================================
# 2. ROC curves per fake suite (vs teams_real_all_dev as negatives)
# ============================================================
def plot_roc_curves():
    print("ROC curves…")
    for fs in FAKE_SUITES:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="chance")
        for ckpt_name, ckpt_token in CKPTS.items():
            real_scores, real_labels = load_scores(REAL_NEG, ckpt_token)
            fake_scores, fake_labels = load_scores(fs, ckpt_token)
            real_only = real_scores[real_labels == 0]
            fake_only = fake_scores[fake_labels == 1]
            # ROC: sweep threshold
            all_thresh = np.unique(np.concatenate([real_only, fake_only]))
            all_thresh = np.sort(all_thresh)[::-1]  # high → low
            tprs, fprs = [], []
            for t in all_thresh:
                tprs.append((fake_only >= t).mean())
                fprs.append((real_only >= t).mean())
            tprs = np.array(tprs)
            fprs = np.array(fprs)
            # AUC
            sort_idx = np.argsort(fprs)
            auc = np.trapz(tprs[sort_idx], fprs[sort_idx])
            ax.plot(fprs, tprs, color=COLORS[ckpt_name], label=f"{ckpt_name} (AUC={auc:.3f})", linewidth=2)
        ax.set_xlabel("FPR (on teams_real_all_dev)")
        ax.set_ylabel("TPR (on fake suite)")
        ax.set_title(f"ROC: {fs}")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        # mark FPR=2/5/10
        for fpr_target in [0.02, 0.05, 0.10]:
            ax.axvline(fpr_target, color="gray", linestyle=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"roc_curve_{fs}.png", dpi=80)
        plt.close(fig)
    print(f"  {len(FAKE_SUITES)} ROC PNGs in {FIG_DIR}")


# ============================================================
# 3. Recall vs FPR curves (zoomed to operational region)
# ============================================================
def plot_recall_vs_fpr():
    print("Recall-vs-FPR (operational zoom)…")
    for fs in FAKE_SUITES:
        fig, ax = plt.subplots(figsize=(7, 5))
        for ckpt_name, ckpt_token in CKPTS.items():
            real_scores, real_labels = load_scores(REAL_NEG, ckpt_token)
            fake_scores, fake_labels = load_scores(fs, ckpt_token)
            real_only = real_scores[real_labels == 0]
            fake_only = fake_scores[fake_labels == 1]
            # Sweep FPR target 0..20% in fine steps
            fpr_targets = np.linspace(0, 0.20, 41)
            recalls = []
            for ft in fpr_targets:
                # Find tau at this FPR target
                if ft == 0:
                    tau = real_only.max() + 1e-6
                else:
                    sorted_real = np.sort(real_only)
                    idx = max(0, int(np.ceil(len(sorted_real) * (1 - ft))) - 1)
                    tau = sorted_real[idx]
                recalls.append((fake_only >= tau).mean())
            ax.plot(fpr_targets * 100, np.array(recalls) * 100,
                    color=COLORS[ckpt_name], label=ckpt_name, linewidth=2)
        ax.set_xlabel("FPR target (% on teams_real_all_dev)")
        ax.set_ylabel("Recall (% on fake suite)")
        ax.set_title(f"Operational recall vs FPR: {fs}")
        ax.legend()
        ax.grid(alpha=0.3)
        for fpr_target in [2, 5, 10]:
            ax.axvline(fpr_target, color="gray", linestyle=":", alpha=0.4)
            ax.text(fpr_target + 0.2, 5, f"FPR={fpr_target}%", fontsize=8, color="gray")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 100)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"recall_vs_fpr_{fs}.png", dpi=80)
        plt.close(fig)
    print(f"  {len(FAKE_SUITES)} operational PNGs in {FIG_DIR}")


# ============================================================
# 4. Score scatter: P8A vs E2B_3200 (per fake suite, colored by label)
# ============================================================
def plot_score_scatter():
    print("Disagreement scatter plots…")
    pairs = [("P8A", "E2B_3200"), ("P8A", "E3_6600"), ("E2B_3200", "E3_6600")]
    for fs in FAKE_SUITES:
        for c1, c2 in pairs:
            f1 = RAW_DIR / f"{fs}_{CKPTS[c1]}_frames_report.csv"
            f2 = RAW_DIR / f"{fs}_{CKPTS[c2]}_frames_report.csv"
            if not (f1.exists() and f2.exists()):
                continue
            a = {}
            with f1.open() as fh:
                for row in csv.DictReader(fh):
                    a[row["frame_path"]] = (int(row["label"]), float(row["frame_prob"]))
            b = {}
            with f2.open() as fh:
                for row in csv.DictReader(fh):
                    b[row["frame_path"]] = (int(row["label"]), float(row["frame_prob"]))
            common = sorted(set(a.keys()) & set(b.keys()))
            if not common:
                continue
            x = np.array([a[p][1] for p in common])
            y = np.array([b[p][1] for p in common])
            labels = np.array([a[p][0] for p in common])
            fig, ax = plt.subplots(figsize=(6, 6))
            for label_val, color, name in [(0, "tab:green", "real"), (1, "tab:red", "fake")]:
                mask = labels == label_val
                if mask.sum() > 0:
                    ax.scatter(x[mask], y[mask], c=color, alpha=0.4, s=8, label=f"{name} (n={mask.sum()})")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
            r = float(np.corrcoef(x, y)[0, 1])
            ax.set_xlabel(f"{c1} score")
            ax.set_ylabel(f"{c2} score")
            ax.set_title(f"{fs}\nPearson r = {r:.3f}")
            ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"scatter_{fs}_{c1}_vs_{c2}.png", dpi=80)
            plt.close(fig)
    print(f"  Generated scatter plots in {FIG_DIR}")


def main():
    plot_histograms()
    plot_roc_curves()
    plot_recall_vs_fpr()
    plot_score_scatter()
    print("\nAll visualizations done.")


if __name__ == "__main__":
    main()
