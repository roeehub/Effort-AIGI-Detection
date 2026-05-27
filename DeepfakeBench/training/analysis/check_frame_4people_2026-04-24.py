#!/usr/bin/env python3
"""Score 4 identity folders from session_20260424_110139 via check-frame.

Parses stdout JSON, builds summary table (mean/std prob, fraction>0.9, fraction<0.1),
persists to analysis/check_frame_4people_2026-04-24.json.
"""
from __future__ import annotations

import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path("/tmp/dor_session_20260424")
OUT = Path(__file__).resolve().parent / "check_frame_4people_2026-04-24.json"
IDENTS = ["Xiang_Xiang2_Feng", "Xinhe_XH68_Wang", "dor_shkedi", "tester_tester"]


def score(ident: str) -> dict:
    p = subprocess.run(
        ["check-frame", str(ROOT / ident), "-n", "32"],
        capture_output=True,
        text=True,
        check=True,
    )
    start = p.stdout.find("{")
    if start < 0:
        raise RuntimeError(f"no JSON for {ident}")
    payload, _ = json.JSONDecoder().raw_decode(p.stdout[start:])
    results = payload.get("results", [])
    probs = [float(r["confidence"]) for r in results]
    return {
        "identity": ident,
        "n_frames": len(probs),
        "overall_label": payload.get("overall_label"),
        "overall_confidence": payload.get("overall_confidence"),
        "mean_prob_fake": statistics.mean(probs) if probs else None,
        "std_prob_fake": statistics.stdev(probs) if len(probs) > 1 else 0.0,
        "min_prob_fake": min(probs) if probs else None,
        "max_prob_fake": max(probs) if probs else None,
        "frac_gt_0_9": sum(1 for p in probs if p > 0.9) / len(probs) if probs else None,
        "frac_lt_0_1": sum(1 for p in probs if p < 0.1) / len(probs) if probs else None,
        "per_frame": [
            {"file": Path(r["file"]).name, "pred": r["pred_label"], "prob_fake": float(r["confidence"])}
            for r in results
        ],
    }


def main():
    summary = {"identities": [score(i) for i in IDENTS]}
    OUT.write_text(json.dumps(summary, indent=2))

    print(f"{'identity':<22} {'n':>3} {'mean':>7} {'std':>6} {'>0.9':>5} {'<0.1':>5} overall")
    print("-" * 72)
    for e in summary["identities"]:
        print(
            f"{e['identity']:<22} {e['n_frames']:>3} "
            f"{e['mean_prob_fake']:>7.3f} {e['std_prob_fake']:>6.3f} "
            f"{e['frac_gt_0_9']:>5.1%} {e['frac_lt_0_1']:>5.1%} "
            f"{e['overall_label']} ({e['overall_confidence']:.3f})"
        )
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
