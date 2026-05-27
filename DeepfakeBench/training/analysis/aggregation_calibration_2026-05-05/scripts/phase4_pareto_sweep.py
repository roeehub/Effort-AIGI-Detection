"""Phase 4 — Window-aggregation Pareto sweep.

For every (policy, suite) compute video-level (= stream-level) recall on
fake suites and FPR on real suites.

Multiprocessing is used at the (policy, suite) granularity. We pre-build
the streams-per-suite ONCE in the parent process, then ship into workers
via a global module-level variable to avoid re-parsing per worker.

Output:
  data/<ckpt>/pareto_curve.csv  — rows = (suite, policy_id, params_json,
                                          n_streams, n_flagged, rate, label_class)
  data/<ckpt>/optimal_policy.json — chosen policy + alternatives
  findings/<ckpt>/phase4_pareto.md
  figures/<ckpt>/pareto_curve.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from _common import ensure_dirs, get_ckpt_cfg, load_policy_grid, load_suite_csv  # noqa: E402
from aggregator_sim import (  # noqa: E402
    build_streams_for_suite,
    expand_policy_grid,
    simulate_streams,
)

# Tuning: number of workers. Mac 10 physical cores, leave a couple for OS.
N_WORKERS = 8


# ----------------------------------------------------------------------
# Worker globals. Pool fork model copies these once per worker.
# ----------------------------------------------------------------------
_STREAMS_PER_SUITE: Dict[str, List[np.ndarray]] = {}
_LABEL_PER_SUITE: Dict[str, str] = {}
_POLICIES: List[dict] = []


def _worker_init(streams_per_suite, label_per_suite, policies):
    global _STREAMS_PER_SUITE, _LABEL_PER_SUITE, _POLICIES
    _STREAMS_PER_SUITE = streams_per_suite
    _LABEL_PER_SUITE = label_per_suite
    _POLICIES = policies


def _eval_one(args):
    suite, policy_idx = args
    policy = _POLICIES[policy_idx]
    streams = _STREAMS_PER_SUITE[suite]
    n_total, n_flagged = simulate_streams(streams, policy)
    return {
        "suite": suite,
        "label_class": _LABEL_PER_SUITE[suite],
        "policy_id": policy["policy_id"],
        "window_size": policy["window_size"],
        "strategy": policy["strategy"],
        "params_json": json.dumps(policy["strategy_params"], sort_keys=True),
        "override_rule": policy["override_rule"],
        "n_streams": int(n_total),
        "n_flagged_fake": int(n_flagged),
        "rate": float(n_flagged / n_total) if n_total else float("nan"),
    }


def _build_streams_per_suite(ckpt_cfg) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, str]]:
    streams = {}
    labels = {}
    for suite_name in ckpt_cfg["suites"].keys():
        try:
            df = load_suite_csv(ckpt_cfg, suite_name)
        except FileNotFoundError as e:
            print(f"  [WARN] suite {suite_name}: {e}")
            continue
        streams[suite_name] = build_streams_for_suite(df, suite_name)
        labels[suite_name] = df["label_class"].iloc[0]
    return streams, labels


def _select_optimal_policy(df: pd.DataFrame) -> dict:
    """Pareto-optimal policy at multiple FPR caps.

    Primary criterion: max(video_FPR across real suites) <= cap.
    Sub-criterion: maximize avg(video_recall across fake suites).

    Returns the chosen policy at FPR<=5% (preferred) or 10% (fallback). Also
    surfaces:
      - chosen at strict cap
      - top alternative under each of FPR<=5,10,20 — so the operator can pick
        a higher-recall policy if the FPR cap can be relaxed.
      - a "spike-resistant" subset (excluding run_length and *_consec overrides)
        for production policies that need to ignore short bursts.
    """
    pivot = df.pivot_table(
        index=["policy_id", "window_size", "strategy", "params_json", "override_rule"],
        columns=["suite", "label_class"],
        values="rate",
    )
    flat = pivot.copy()
    flat.columns = [f"{s}|{lc}" for s, lc in flat.columns]
    flat = flat.reset_index()

    fake_cols = [c for c in flat.columns if c.endswith("|fake")]
    real_cols = [c for c in flat.columns if c.endswith("|real")]
    if not fake_cols or not real_cols:
        return {"error": "missing fake or real suites in pareto grid"}

    flat["fake_macro_recall"] = flat[fake_cols].mean(axis=1)
    flat["real_max_fpr"] = flat[real_cols].max(axis=1)

    def to_record(r):
        per_suite = {}
        for c in fake_cols + real_cols:
            per_suite[c] = float(r[c]) if pd.notna(r[c]) else None
        return {
            "policy_id": int(r["policy_id"]),
            "window_size": int(r["window_size"]),
            "strategy": r["strategy"],
            "strategy_params": json.loads(r["params_json"]),
            "override_rule": r["override_rule"],
            "fake_macro_recall": float(r["fake_macro_recall"]),
            "real_max_fpr": float(r["real_max_fpr"]),
            "per_suite_rate": per_suite,
        }

    def best_under(cap, restrict_to_spike_resistant=False):
        sub = flat[flat["real_max_fpr"] <= cap].copy()
        if restrict_to_spike_resistant:
            # Exclude run_length and overrides that are spike-vulnerable
            spike_safe_overrides = ["none", "one_frame_above_0.98", "two_frames_above_0.98_in_window"]
            sub = sub[
                (sub["strategy"] != "run_length")
                & (sub["override_rule"].isin(spike_safe_overrides))
            ]
        if sub.empty:
            return None, None
        sub = sub.sort_values(by=["fake_macro_recall", "real_max_fpr"], ascending=[False, True])
        return to_record(sub.iloc[0]), [to_record(r) for _, r in sub.head(3).iloc[1:].iterrows()]

    # Pick primary chosen
    chosen, alts = best_under(0.05)
    fpr_cap_used = 0.05
    if chosen is None:
        chosen, alts = best_under(0.10)
        fpr_cap_used = 0.10

    # Build the alternative-at-each-cap report
    alt_caps = {}
    for cap in [0.05, 0.10, 0.15, 0.20]:
        b, _ = best_under(cap)
        if b:
            alt_caps[f"FPR<={cap:.2f}_best"] = b

    # Spike-resistant variants
    spike_alts = {}
    for cap in [0.05, 0.10, 0.15, 0.20]:
        b, _ = best_under(cap, restrict_to_spike_resistant=True)
        if b:
            spike_alts[f"FPR<={cap:.2f}_best_spike_resistant"] = b

    return {
        "fpr_cap_used": fpr_cap_used,
        "chosen": chosen,
        "alternatives": alts or [],
        "best_at_each_fpr_cap": alt_caps,
        "spike_resistant_at_each_fpr_cap": spike_alts,
    }


def _plot_pareto(df: pd.DataFrame, out_path: Path, ckpt_full: str) -> None:
    """Plot fake-recall vs real-FPR scatter, color by strategy."""
    pivot = df.pivot_table(
        index=["policy_id", "strategy"],
        columns=["suite", "label_class"],
        values="rate",
    )
    flat = pivot.copy()
    flat.columns = [f"{s}|{lc}" for s, lc in flat.columns]
    flat = flat.reset_index()
    fake_cols = [c for c in flat.columns if c.endswith("|fake")]
    real_cols = [c for c in flat.columns if c.endswith("|real")]
    if not fake_cols or not real_cols:
        return
    flat["fake_macro_recall"] = flat[fake_cols].mean(axis=1)
    flat["real_max_fpr"] = flat[real_cols].max(axis=1)
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = {
        "majority_vote": "C0",
        "high_vote_floor": "C1",
        "trimmed_mean_above_thresh": "C2",
        "median_above_thresh": "C3",
        "rolling_mean_then_thresh": "C4",
        "run_length": "C5",
    }
    for strat, sub in flat.groupby("strategy"):
        ax.scatter(
            sub["real_max_fpr"],
            sub["fake_macro_recall"],
            s=12,
            alpha=0.55,
            c=palette.get(strat, "gray"),
            label=strat,
        )
    ax.axvline(0.05, color="red", linestyle="--", lw=0.8, label="FPR=5%")
    ax.axvline(0.10, color="orange", linestyle="--", lw=0.8, label="FPR=10%")
    ax.set_xlabel("max(video_FPR across real suites)")
    ax.set_ylabel("mean(video_recall across fake suites)")
    ax.set_title(f"Phase 4 — Pareto curve — {ckpt_full}")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(ckpt_name: str) -> None:
    cfg = get_ckpt_cfg(ckpt_name)
    dirs = ensure_dirs(ckpt_name)
    full_name = cfg["full_name"]
    t0 = time.time()

    grid = load_policy_grid()
    policies = expand_policy_grid(grid)
    print(f"[phase4] {full_name}: {len(policies)} policies")

    streams, labels = _build_streams_per_suite(cfg)
    suites = list(streams.keys())
    print(f"[phase4] {full_name}: streams_per_suite =", {s: len(streams[s]) for s in suites})

    work = [(suite, idx) for suite in suites for idx in range(len(policies))]
    print(f"[phase4] {full_name}: {len(work)} work units, {N_WORKERS} workers")

    with Pool(
        N_WORKERS,
        initializer=_worker_init,
        initargs=(streams, labels, policies),
    ) as pool:
        results = pool.map(_eval_one, work, chunksize=200)

    df = pd.DataFrame(results)
    df.to_csv(dirs["data"] / "pareto_curve.csv", index=False)

    # Choose optimal policy.
    sel = _select_optimal_policy(df)
    (dirs["data"] / "optimal_policy.json").write_text(json.dumps(sel, indent=2))

    # Plot.
    _plot_pareto(df, dirs["figures"] / "pareto_curve.png", full_name)

    # Markdown.
    md = [f"# Phase 4 — Pareto sweep — {full_name}", ""]
    if sel.get("chosen") is None:
        md.append("**No policy met FPR cap; reporting top-3 by (fake_recall − real_fpr).**\n")
    else:
        ch = sel["chosen"]
        md.append(f"**FPR cap used (primary)**: {sel['fpr_cap_used']}")
        md.append(
            f"\n**Chosen policy** (#{ch['policy_id']}): `{ch['strategy']}` "
            f"W={ch['window_size']} params={ch['strategy_params']} override=`{ch['override_rule']}`"
        )
        md.append(f"- fake_macro_recall: **{ch['fake_macro_recall']:.3f}**")
        md.append(f"- real_max_fpr: **{ch['real_max_fpr']:.3f}**")
        md.append("\n**Per-suite rates**:")
        for k, v in ch["per_suite_rate"].items():
            if v is not None:
                md.append(f"- {k}: {v:.3f}")

        md.append("\n**Top-3 at primary cap**:\n")
        md.append("| rank | strategy | W | params | override | fake_macro_R | real_max_FPR |")
        md.append("|---:|---|---:|---|---|---:|---:|")
        all_top = [ch] + (sel.get("alternatives") or [])
        for i, alt in enumerate(all_top, 1):
            md.append(
                f"| {i} | {alt['strategy']} | {alt['window_size']} | "
                f"{alt['strategy_params']} | {alt['override_rule']} | "
                f"{alt['fake_macro_recall']:.3f} | {alt['real_max_fpr']:.3f} |"
            )

        # Best at each cap
        md.append("\n## Best policy at each FPR cap (any strategy / override)\n")
        md.append("| FPR cap | strategy | W | params | override | fake_macro_R | real_max_FPR | per-suite (deeplive / fake_dev / fake_lockbox) |")
        md.append("|---|---|---:|---|---|---:|---:|---|")
        for cap_label, alt in (sel.get("best_at_each_fpr_cap") or {}).items():
            ps = alt["per_suite_rate"]
            md.append(
                f"| {cap_label} | {alt['strategy']} | {alt['window_size']} | "
                f"{alt['strategy_params']} | {alt['override_rule']} | "
                f"{alt['fake_macro_recall']:.3f} | {alt['real_max_fpr']:.3f} | "
                f"{ps.get('deeplive_enhanced_dev|fake', float('nan')):.3f} / "
                f"{ps.get('teams_fake_all_dev|fake', float('nan')):.3f} / "
                f"{ps.get('teams_fake_all_lockbox|fake', float('nan')):.3f} |"
            )

        md.append("\n## Best SPIKE-RESISTANT policy at each FPR cap\n")
        md.append("(excluding `run_length` strategy and `*_consec_*` overrides — these are vulnerable to short bursts of high-prob frames)\n")
        md.append("| FPR cap | strategy | W | params | override | fake_macro_R | real_max_FPR | per-suite (deeplive / fake_dev / fake_lockbox) |")
        md.append("|---|---|---:|---|---|---:|---:|---|")
        for cap_label, alt in (sel.get("spike_resistant_at_each_fpr_cap") or {}).items():
            ps = alt["per_suite_rate"]
            md.append(
                f"| {cap_label} | {alt['strategy']} | {alt['window_size']} | "
                f"{alt['strategy_params']} | {alt['override_rule']} | "
                f"{alt['fake_macro_recall']:.3f} | {alt['real_max_fpr']:.3f} | "
                f"{ps.get('deeplive_enhanced_dev|fake', float('nan')):.3f} / "
                f"{ps.get('teams_fake_all_dev|fake', float('nan')):.3f} / "
                f"{ps.get('teams_fake_all_lockbox|fake', float('nan')):.3f} |"
            )

    md.append(f"\n_Wall time: {time.time() - t0:.1f}s, {len(work)} work units, {N_WORKERS} workers_")
    (dirs["findings"] / "phase4_pareto.md").write_text("\n".join(md))
    print(f"[phase4] {full_name} done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    a = ap.parse_args()
    N_WORKERS = a.workers
    main(a.ckpt)
