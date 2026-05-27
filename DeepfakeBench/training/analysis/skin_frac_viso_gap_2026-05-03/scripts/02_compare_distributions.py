"""Step 2: Compare per-group distributions of skin_frac, laplacian_var, luma_mean.

Inputs : outputs/per_frame_attrs.csv
Outputs:
  outputs/group_stats.csv
  outputs/ks_tests.json
  outputs/distributions.png
  FINDINGS.md (verdict + table + implications)

Verdict logic on skin_frac:
  CONFIRMED gap:
    KS p<0.01 (eval_viso_fake vs train_viso_fake)
    AND |mean(eval_viso_fake) - mean(train_viso_fake)| > 0.10
  NO GAP:
    KS p>0.05 OR |delta| < 0.05
  PARTIAL otherwise.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

DIAG_ROOT = Path(__file__).resolve().parent.parent
OUT = DIAG_ROOT / "outputs"
LOG = DIAG_ROOT / "run.log"

ATTRS = ["skin_frac", "laplacian_var", "luma_mean"]
GROUPS = ["eval_viso_fake", "train_viso_fake", "eval_real", "train_viso_real"]
PAIRS = [
    ("eval_viso_fake", "train_viso_fake"),
    ("eval_viso_fake", "eval_real"),
    ("train_viso_fake", "train_viso_real"),
]


def write_log(msg: str) -> None:
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def group_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in GROUPS:
        sub = df[df["group"] == g]
        if sub.empty:
            continue
        row = {"group": g, "n": len(sub)}
        for a in ATTRS:
            v = sub[a].to_numpy(dtype=float)
            row[f"{a}_mean"] = float(np.mean(v))
            row[f"{a}_median"] = float(np.median(v))
            row[f"{a}_std"] = float(np.std(v))
            row[f"{a}_p10"] = float(np.percentile(v, 10))
            row[f"{a}_p90"] = float(np.percentile(v, 90))
        rows.append(row)
    return pd.DataFrame(rows)


def ks_tests(df: pd.DataFrame) -> dict:
    out = {}
    for a in ATTRS:
        out[a] = {}
        for ga, gb in PAIRS:
            va = df.loc[df["group"] == ga, a].to_numpy(dtype=float)
            vb = df.loc[df["group"] == gb, a].to_numpy(dtype=float)
            if len(va) < 5 or len(vb) < 5:
                out[a][f"{ga}_vs_{gb}"] = {"statistic": None, "p_value": None,
                                           "n_a": int(len(va)), "n_b": int(len(vb))}
                continue
            stat, p = ks_2samp(va, vb, alternative="two-sided", mode="auto")
            out[a][f"{ga}_vs_{gb}"] = {
                "statistic": float(stat),
                "p_value": float(p),
                "n_a": int(len(va)),
                "n_b": int(len(vb)),
                "mean_a": float(np.mean(va)),
                "mean_b": float(np.mean(vb)),
                "delta_mean_a_minus_b": float(np.mean(va) - np.mean(vb)),
            }
    return out


def plot_distributions(df: pd.DataFrame, out_path: Path) -> None:
    colors = {
        "eval_viso_fake":  "tab:red",
        "train_viso_fake": "tab:blue",
        "eval_real":       "tab:green",
        "train_viso_real": "tab:gray",
    }
    show_groups = ["eval_viso_fake", "train_viso_fake", "eval_real", "train_viso_real"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 11))
    bin_specs = {
        "skin_frac":    (np.linspace(0, 1, 41), "skin_frac (BT.601 YCrCb fraction)"),
        "laplacian_var":(np.linspace(0, np.percentile(df["laplacian_var"], 99), 41), "laplacian_var (sharpness)"),
        "luma_mean":    (np.linspace(0, 255, 41), "luma_mean (brightness 0-255)"),
    }
    for ax, attr in zip(axes, ATTRS):
        bins, label = bin_specs[attr]
        for g in show_groups:
            sub = df[df["group"] == g]
            if sub.empty:
                continue
            ax.hist(
                sub[attr].to_numpy(dtype=float),
                bins=bins, density=True, histtype="step",
                linewidth=2.0, label=f"{g} (n={len(sub)})", color=colors[g],
            )
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.set_title(f"Distribution: {attr}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def verdict(ks: dict, stats: pd.DataFrame) -> tuple[str, str]:
    # Pull skin_frac comparison eval_viso_fake vs train_viso_fake
    rec = ks["skin_frac"]["eval_viso_fake_vs_train_viso_fake"]
    p = rec["p_value"]
    delta = rec["delta_mean_a_minus_b"]
    if p is None:
        return "INSUFFICIENT_DATA", f"skin_frac KS could not run (n_a={rec['n_a']}, n_b={rec['n_b']})"
    if (p < 0.01) and (abs(delta) > 0.10):
        v = "CONFIRMED"
    elif (p > 0.05) or (abs(delta) < 0.05):
        v = "NO_GAP"
    else:
        v = "PARTIAL"
    detail = f"KS p={p:.3g}, |delta|={abs(delta):.3f} (eval_viso_fake mean={rec['mean_a']:.3f} vs train_viso_fake mean={rec['mean_b']:.3f})"
    return v, detail


def write_findings(verdict_str: str, detail: str, ks: dict, stats: pd.DataFrame) -> None:
    lines = []
    lines.append(f"# skin_frac viso eval-vs-train gap — VERDICT: {verdict_str}")
    lines.append("")
    lines.append(f"_{detail}_")
    lines.append("")
    lines.append(f"Run: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Group distribution table (skin_frac, laplacian_var, luma_mean)")
    lines.append("")
    lines.append("| group | n | skin_frac mean±std | skin_frac p10/p90 | laplacian_var mean±std | luma_mean mean±std |")
    lines.append("|---|---:|---|---|---|---|")
    for _, r in stats.iterrows():
        lines.append(
            f"| {r['group']} | {int(r['n'])} | "
            f"{r['skin_frac_mean']:.3f}±{r['skin_frac_std']:.3f} | "
            f"{r['skin_frac_p10']:.3f}/{r['skin_frac_p90']:.3f} | "
            f"{r['laplacian_var_mean']:.1f}±{r['laplacian_var_std']:.1f} | "
            f"{r['luma_mean_mean']:.1f}±{r['luma_mean_std']:.1f} |"
        )
    lines.append("")
    lines.append("## KS-test summary (two-sample, two-sided)")
    lines.append("")
    lines.append("| attr | pair | statistic | p_value | mean_a | mean_b | delta(a-b) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for attr, pairings in ks.items():
        for pair_name, rec in pairings.items():
            if rec["statistic"] is None:
                continue
            lines.append(
                f"| {attr} | {pair_name} | {rec['statistic']:.3f} | {rec['p_value']:.3g} | "
                f"{rec['mean_a']:.3f} | {rec['mean_b']:.3f} | {rec['delta_mean_a_minus_b']:+.3f} |"
            )
    lines.append("")
    lines.append("## Implications for next packet")
    lines.append("")
    if verdict_str == "CONFIRMED":
        lines.append(
            "- Skin-aware augmentation (e.g. random skin-tone shifts in YCrCb, randomised "
            "background re-paste, or skin/non-skin masked colour jitter) is a credible "
            "single-lever try, since the eval and train distributions on the third axis "
            "of the known image-quality shortcut diverge meaningfully on the same suite "
            "(visomaster_enhanced) where the recall ceiling has held."
        )
        lines.append(
            "- Recommended packet: a P-* sister to P22 that adds *only* a skin-aware "
            "aug to the existing curriculum; keep all other knobs fixed; promote via the "
            "promotion contract; compare to the P22 step1k baseline."
        )
    elif verdict_str == "NO_GAP":
        lines.append(
            "- The skin_frac axis does NOT show a measurable train-vs-eval mismatch on "
            "the visomaster_enhanced suite. A skin-aware aug is unlikely to be the cheap "
            "lever that breaks the 27% viso recall ceiling. Next packets should look "
            "elsewhere (architecture, identity-axis diversity in unused buckets, or a "
            "structural change as flagged in the viso-ceiling memory)."
        )
    else:
        lines.append(
            "- Ambiguous gap on skin_frac. KS is significant but the magnitude is borderline, "
            "so a skin-aware aug might help marginally but is not the obviously dominant "
            "lever. Recommend keeping it on the bench while pursuing the structural "
            "options first."
        )
    (DIAG_ROOT / "FINDINGS.md").write_text("\n".join(lines))


def main():
    write_log(f"\n=== 02_compare_distributions.py @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    df = pd.read_csv(OUT / "per_frame_attrs.csv")
    write_log(f"[load] n={len(df)} rows; groups={df['group'].value_counts().to_dict()}")

    stats = group_stats(df)
    stats.to_csv(OUT / "group_stats.csv", index=False)
    write_log(f"[stats] wrote {OUT / 'group_stats.csv'}")

    ks = ks_tests(df)
    (OUT / "ks_tests.json").write_text(json.dumps(ks, indent=2))
    write_log(f"[ks] wrote {OUT / 'ks_tests.json'}")

    plot_distributions(df, OUT / "distributions.png")
    write_log(f"[plot] wrote {OUT / 'distributions.png'}")

    v, detail = verdict(ks, stats)
    write_findings(v, detail, ks, stats)
    write_log(f"[verdict] {v} :: {detail}")
    print(f"\nVERDICT: {v}\n  {detail}", flush=True)


if __name__ == "__main__":
    main()
