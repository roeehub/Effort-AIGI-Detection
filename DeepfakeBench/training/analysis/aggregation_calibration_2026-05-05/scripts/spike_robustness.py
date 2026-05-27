"""Optional spike-robustness probe — used to populate the FINDINGS.md headline.

Compare vanilla majority_vote(τ=0.5, vote=0.5, W=32, override=none) against the
Phase 4 chosen policy under simulated spikes. We synthesize spikes on the
multi-frame REAL streams (length >= 32) by injecting K consecutive frames
clamped to 0.99 (false-spike) or 0.01 (false-real) and measure how often
the policy's verdict on the stream FLIPS.

This is a structural sanity check; it doesn't change the chosen policy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from _common import ensure_dirs, get_ckpt_cfg, load_suite_csv  # noqa: E402
from aggregator_sim import (  # noqa: E402
    build_streams_for_suite,
    simulate_video,
)


def vanilla_policy():
    return {
        "window_size": 32,
        "strategy": "majority_vote",
        "strategy_params": {"threshold": 0.5, "vote_majority": 0.5},
        "override_rule": "none",
    }


def inject_spike(scores: np.ndarray, k: int, value: float) -> np.ndarray:
    if len(scores) < k:
        return scores.copy()
    out = scores.copy()
    start = max(0, (len(out) - k) // 2)
    out[start : start + k] = value
    return out


def main(ckpt_name: str) -> None:
    cfg = get_ckpt_cfg(ckpt_name)
    dirs = ensure_dirs(ckpt_name)
    full_name = cfg["full_name"]

    optimal = json.loads((dirs["data"] / "optimal_policy.json").read_text())
    chosen = optimal.get("chosen")
    if chosen is None:
        print("[spike] no chosen policy")
        return
    chosen_policy = {
        "window_size": chosen["window_size"],
        "strategy": chosen["strategy"],
        "strategy_params": chosen["strategy_params"],
        "override_rule": chosen["override_rule"],
    }
    vanilla = vanilla_policy()

    chosen_no_override = dict(chosen_policy)
    chosen_no_override["override_rule"] = "none"

    rows = []
    for suite_name, suite_meta in cfg["suites"].items():
        if suite_meta.get("label_class") != "real":
            continue
        try:
            df = load_suite_csv(cfg, suite_name)
        except FileNotFoundError:
            continue
        streams = build_streams_for_suite(df, suite_name)
        long_streams = [s for s in streams if len(s) >= 32]
        if len(long_streams) < 1:
            continue
        for k_spike in [4, 8, 12]:
            for label, policy in [
                ("vanilla", vanilla),
                ("chosen", chosen_policy),
                ("chosen_no_override", chosen_no_override),
            ]:
                base_flagged = sum(1 for s in long_streams if simulate_video(s, policy))
                spiked_flagged = sum(
                    1
                    for s in long_streams
                    if simulate_video(inject_spike(s, k_spike, 0.99), policy)
                )
                rows.append(
                    {
                        "suite": suite_name,
                        "policy_label": label,
                        "k_spike": k_spike,
                        "n_streams": len(long_streams),
                        "baseline_FPR": base_flagged / len(long_streams),
                        "spiked_FPR": spiked_flagged / len(long_streams),
                        "delta_FPR": (spiked_flagged - base_flagged) / len(long_streams),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(dirs["data"] / "spike_robustness.csv", index=False)
    if df.empty:
        print("[spike] no eligible long streams")
        return

    md = [f"# Spike robustness — {full_name}", ""]
    md.append("Inject K consecutive frames at 0.99 into REAL multi-frame streams (>=32 frames). "
              "Measure the FPR delta. Policy is robust if delta is small (< 5pp).\n")
    md.append("| suite | policy | K_spike | n | baseline_FPR | spiked_FPR | Δ |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for _, r in df.iterrows():
        md.append(
            f"| {r['suite']} | {r['policy_label']} | {r['k_spike']} | "
            f"{int(r['n_streams'])} | {r['baseline_FPR']:.3f} | "
            f"{r['spiked_FPR']:.3f} | {r['delta_FPR']:+.3f} |"
        )
    (dirs["findings"] / "spike_robustness.md").write_text("\n".join(md))
    print("[spike] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    main(a.ckpt)
