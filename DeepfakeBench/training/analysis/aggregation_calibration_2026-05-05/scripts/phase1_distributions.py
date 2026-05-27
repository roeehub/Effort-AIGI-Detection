"""Phase 1 — Frame-level distribution characterization.

Per (suite, label):
  - histogram (numpy bins=50)
  - tail percentiles p25 p50 p75 p90 p95 p99
  - muddy-zone fraction in [0.2, 0.8] and [0.1, 0.9]

Per real-vs-fake suite-pair on same family (e.g., teams_*_dev):
  - KL divergence
  - Wasserstein-1 (EMD)
  - bimodality coefficient on the union
  - frame-level ROC-AUC of the discriminator

Outputs:
  data/<ckpt>/distributions.csv  — long format per-suite stats
  data/<ckpt>/separation_pairs.csv  — pair-level (real_suite, fake_suite) separation metrics
  findings/<ckpt>/phase1_distributions.md
  figures/<ckpt>/distributions.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from _common import ensure_dirs, get_ckpt_cfg, load_suite_csv  # noqa: E402


# real-vs-fake pairs we want to compute separation on.
PAIR_SPECS = [
    ("teams_real_all_dev", "teams_fake_all_dev"),
    ("teams_real_all_lockbox", "teams_fake_all_lockbox"),
    ("teams_real_all_dev", "deeplive_enhanced_dev"),
]


def compute_dist_stats(scores: np.ndarray) -> dict:
    if len(scores) == 0:
        return {"n_frames": 0}
    s = scores.astype(np.float64)
    return {
        "n_frames": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p01": float(np.percentile(s, 1)),
        "p05": float(np.percentile(s, 5)),
        "p10": float(np.percentile(s, 10)),
        "p25": float(np.percentile(s, 25)),
        "p50": float(np.percentile(s, 50)),
        "p75": float(np.percentile(s, 75)),
        "p90": float(np.percentile(s, 90)),
        "p95": float(np.percentile(s, 95)),
        "p99": float(np.percentile(s, 99)),
        "muddy_2_8_frac": float(((s >= 0.2) & (s <= 0.8)).mean()),
        "muddy_1_9_frac": float(((s >= 0.1) & (s <= 0.9)).mean()),
        "frac_above_0.5": float((s >= 0.5).mean()),
        "frac_above_0.7": float((s >= 0.7).mean()),
        "frac_above_0.9": float((s >= 0.9).mean()),
        "frac_above_0.95": float((s >= 0.95).mean()),
        "frac_above_0.98": float((s >= 0.98).mean()),
    }


def kl_divergence(p_scores: np.ndarray, q_scores: np.ndarray, bins: int = 50) -> float:
    """KL(P || Q) on histograms over [0,1]."""
    edges = np.linspace(0, 1, bins + 1)
    p, _ = np.histogram(p_scores, bins=edges, density=False)
    q, _ = np.histogram(q_scores, bins=edges, density=False)
    p = p.astype(np.float64) + 1e-9
    q = q.astype(np.float64) + 1e-9
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    """1D Wasserstein-1 = ∫|F_a - F_b| via sorted-sample formula."""
    a_s = np.sort(a)
    b_s = np.sort(b)
    # Equalize lengths via interpolation on quantiles.
    n = max(len(a_s), len(b_s))
    qa = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(a_s)), a_s)
    qb = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(b_s)), b_s)
    return float(np.mean(np.abs(qa - qb)))


def bimodality_coefficient(s: np.ndarray) -> float:
    """SAS bimodality coefficient: (skew^2 + 1) / (kurt + 3*(n-1)^2/((n-2)(n-3)))."""
    n = len(s)
    if n < 4:
        return float("nan")
    m = s.mean()
    var = s.var()
    if var <= 0:
        return float("nan")
    sk = float(((s - m) ** 3).mean() / (var ** 1.5))
    kt = float(((s - m) ** 4).mean() / (var ** 2)) - 3.0
    correction = 3.0 * (n - 1) ** 2 / max(1, ((n - 2) * (n - 3)))
    return (sk * sk + 1.0) / (kt + correction)


def roc_auc(real_scores: np.ndarray, fake_scores: np.ndarray) -> float:
    """ROC-AUC where positive class = fake."""
    if len(real_scores) == 0 or len(fake_scores) == 0:
        return float("nan")
    # Compute via Mann-Whitney U for stability w/o sklearn.
    y = np.concatenate([np.zeros(len(real_scores)), np.ones(len(fake_scores))])
    s = np.concatenate([real_scores, fake_scores])
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average ranks for ties
    df = pd.DataFrame({"s": s, "y": y})
    df["r"] = ranks
    avg_r = df.groupby("s")["r"].transform("mean")
    pos_r = avg_r[df["y"] == 1].sum()
    n_pos = (df["y"] == 1).sum()
    n_neg = (df["y"] == 0).sum()
    auc = (pos_r - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def plot_distributions(per_suite: dict, out_path: Path, ckpt_full: str) -> None:
    n = len(per_suite)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for i, (suite_name, info) in enumerate(per_suite.items()):
        ax = axes[i]
        s = info["scores"]
        color = "C3" if info["label_class"] == "fake" else "C0"
        ax.hist(s, bins=50, color=color, alpha=0.7)
        ax.axvspan(0.2, 0.8, color="gray", alpha=0.15, label="muddy [0.2,0.8]")
        ax.set_title(f"{suite_name}\n(n={len(s)}, {info['label_class']})", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("frame_prob", fontsize=7)
        ax.tick_params(labelsize=6)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Phase 1 — frame_prob distributions — {ckpt_full}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(ckpt_name: str) -> None:
    cfg = get_ckpt_cfg(ckpt_name)
    dirs = ensure_dirs(ckpt_name)
    full_name = cfg["full_name"]
    t0 = time.time()

    # Load all suites.
    per_suite = {}
    rows = []
    for suite_name in cfg["suites"].keys():
        try:
            df = load_suite_csv(cfg, suite_name)
        except FileNotFoundError as e:
            print(f"  [WARN] suite {suite_name}: {e}")
            continue
        s = df["frame_prob"].to_numpy(dtype=np.float64)
        per_suite[suite_name] = {"scores": s, "label_class": df["label_class"].iloc[0]}
        stats = compute_dist_stats(s)
        rows.append({"suite": suite_name, "label_class": df["label_class"].iloc[0], **stats})

    pd.DataFrame(rows).to_csv(dirs["data"] / "distributions.csv", index=False)

    # Separation metrics on canonical real-vs-fake pairs.
    pair_rows = []
    for real_suite, fake_suite in PAIR_SPECS:
        if real_suite not in per_suite or fake_suite not in per_suite:
            continue
        r = per_suite[real_suite]["scores"]
        f = per_suite[fake_suite]["scores"]
        pair_rows.append(
            {
                "real_suite": real_suite,
                "fake_suite": fake_suite,
                "n_real": int(len(r)),
                "n_fake": int(len(f)),
                "kl_real_to_fake": kl_divergence(r, f),
                "kl_fake_to_real": kl_divergence(f, r),
                "wasserstein1": wasserstein1(r, f),
                "bimodality_union": bimodality_coefficient(np.concatenate([r, f])),
                "roc_auc_fake_pos": roc_auc(r, f),
            }
        )
    pd.DataFrame(pair_rows).to_csv(dirs["data"] / "separation_pairs.csv", index=False)

    # Plot.
    plot_distributions(per_suite, dirs["figures"] / "distributions.png", full_name)

    # Markdown summary.
    md = [f"# Phase 1 — frame-level distributions — {full_name}", ""]
    md.append("## Per-suite stats\n")
    md.append("| suite | label | n | mean | p50 | p95 | p99 | muddy[0.2,0.8] | muddy[0.1,0.9] | frac>=0.98 |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['suite']} | {r['label_class']} | {r['n_frames']} | {r['mean']:.3f} | "
            f"{r['p50']:.3f} | {r['p95']:.3f} | {r['p99']:.3f} | "
            f"{r['muddy_2_8_frac']:.3f} | {r['muddy_1_9_frac']:.3f} | {r['frac_above_0.98']:.3f} |"
        )
    md.append("\n## Real-vs-fake separation\n")
    md.append("| real_suite | fake_suite | n_real | n_fake | KL(r→f) | W1 | bimod_union | AUC(fake+) |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for p in pair_rows:
        md.append(
            f"| {p['real_suite']} | {p['fake_suite']} | {p['n_real']} | {p['n_fake']} | "
            f"{p['kl_real_to_fake']:.3f} | {p['wasserstein1']:.3f} | "
            f"{p['bimodality_union']:.3f} | {p['roc_auc_fake_pos']:.3f} |"
        )
    md.append("")
    md.append(f"_Wall time: {time.time() - t0:.1f}s_")
    (dirs["findings"] / "phase1_distributions.md").write_text("\n".join(md))
    print(f"[phase1] {full_name} done in {time.time() - t0:.1f}s -> {dirs['findings']/'phase1_distributions.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    main(a.ckpt)
