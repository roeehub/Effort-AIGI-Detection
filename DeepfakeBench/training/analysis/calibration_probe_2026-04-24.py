"""Per-camera calibration probe (WS-P1).

Tests the hypothesis: the false-flag phenomenon is a per-camera score-distribution
SHIFT (a calibration problem) rather than a learned artifact the model cannot unlearn.

Procedure:
  1. Split each of the 6 pools (each 30 frames of REAL subjects) 50/50 (15 train / 15 test).
  2. For each target FPR level (5, 10, 15, 25%):
     a. Fit global tau on pooled train scores so P(score > tau | train) = target.
     b. Fit per-pool tau_p on each pool's train scores at same target.
  3. Evaluate on test halves:
     a. Per-pool test FPR under global tau.
     b. Per-pool test FPR under per-pool tau_p.
  4. Gap = max(FPR) - min(FPR) across pools.
     Gap closure = (gap_global - gap_perpool) / gap_global.

Decision gate (plan WS-P1):
  - Gap closure >= 60%  -> calibration is right lever; WS-P4 likely wrong direction.
  - Gap closure <= 30%  -> training-time augmentation is the right lever.
  - 30-60%              -> both in parallel.

Note: all 180 frames are REAL subjects. "FPR" here means fraction of real frames
the detector would falsely flag as fake above the chosen threshold. We have no
positive class in this session, so we cannot measure TPR impact -- that comes
from the retro-score path on the lockbox suites.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

DATA_PATH = Path("/tmp/dor_roee_combined_2026-04-24/combined_frame_tags.json")
OUT_DIR = Path(__file__).resolve().parent
OUT_JSON = OUT_DIR / "calibration_probe_2026-04-24.summary.json"

SEED = 42
TARGET_FPRS = [0.05, 0.10, 0.15, 0.25]

POOL_ORDER = [
    "dor-real-laptop-correct-no-virtual-bg-whiteish",
    "dor-real-laptop-correct-no-virtual-bg-yellowish",
    "roee-real-windows-laptop-correct",
    "dor-real-webcam-false-flag",
    "dor-real-webcam-false-flag-no-virtual-bg",
    "roee-mac-laptop-false-flag-virtual-bg",
]

POOL_IS_FAILING = {
    "dor-real-laptop-correct-no-virtual-bg-whiteish": False,
    "dor-real-laptop-correct-no-virtual-bg-yellowish": False,
    "roee-real-windows-laptop-correct": False,
    "dor-real-webcam-false-flag": True,
    "dor-real-webcam-false-flag-no-virtual-bg": True,
    "roee-mac-laptop-false-flag-virtual-bg": True,
}


def load_pool_scores() -> Dict[str, np.ndarray]:
    with DATA_PATH.open() as f:
        data = json.load(f)
    pools: Dict[str, np.ndarray] = {}
    for tag in data["tags"]:
        scores = np.array(
            [it["score"] for it in tag["items"] if it.get("score") is not None],
            dtype=np.float64,
        )
        pools[tag["name"]] = scores
    return pools


def split_pool(scores: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(scores))
    half = len(scores) // 2
    return scores[idx[:half]], scores[idx[half:]]


def fit_threshold_at_fpr(train_scores: np.ndarray, target_fpr: float) -> float:
    """Return tau such that (1 - ecdf(tau)) == target_fpr on train_scores."""
    # tau is at the (1 - target_fpr) quantile. Use higher-quantile tie-break.
    q = 1.0 - target_fpr
    return float(np.quantile(train_scores, q, method="higher"))


def fpr_at(scores: np.ndarray, tau: float) -> float:
    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores > tau))


def run_probe() -> dict:
    pools = load_pool_scores()
    # Validate
    missing = [p for p in POOL_ORDER if p not in pools]
    if missing:
        raise RuntimeError(f"Missing pools: {missing}")

    per_pool_split: Dict[str, Dict[str, np.ndarray]] = {}
    # Deterministic split: per-pool seed derived from SEED + pool index
    for i, name in enumerate(POOL_ORDER):
        tr, te = split_pool(pools[name], SEED + i)
        per_pool_split[name] = {"train": tr, "test": te}

    all_train = np.concatenate([per_pool_split[n]["train"] for n in POOL_ORDER])

    results_by_target: List[dict] = []
    for target_fpr in TARGET_FPRS:
        tau_global = fit_threshold_at_fpr(all_train, target_fpr)

        per_pool_tau: Dict[str, float] = {}
        for name in POOL_ORDER:
            per_pool_tau[name] = fit_threshold_at_fpr(
                per_pool_split[name]["train"], target_fpr
            )

        per_pool_report: List[dict] = []
        global_test_fprs = []
        perpool_test_fprs = []
        for name in POOL_ORDER:
            test = per_pool_split[name]["test"]
            fpr_g = fpr_at(test, tau_global)
            fpr_p = fpr_at(test, per_pool_tau[name])
            global_test_fprs.append(fpr_g)
            perpool_test_fprs.append(fpr_p)
            per_pool_report.append(
                {
                    "pool": name,
                    "is_failing": POOL_IS_FAILING[name],
                    "train_mean": float(np.mean(per_pool_split[name]["train"])),
                    "train_std": float(np.std(per_pool_split[name]["train"])),
                    "test_mean": float(np.mean(test)),
                    "tau_pool": per_pool_tau[name],
                    "test_fpr_global_tau": fpr_g,
                    "test_fpr_perpool_tau": fpr_p,
                }
            )

        gap_global = float(max(global_test_fprs) - min(global_test_fprs))
        gap_perpool = float(max(perpool_test_fprs) - min(perpool_test_fprs))
        gap_closure = (gap_global - gap_perpool) / gap_global if gap_global > 0 else 0.0

        results_by_target.append(
            {
                "target_fpr": target_fpr,
                "tau_global": tau_global,
                "per_pool": per_pool_report,
                "test_fpr_under_global_tau": {
                    "per_pool": dict(zip(POOL_ORDER, global_test_fprs)),
                    "max": float(max(global_test_fprs)),
                    "min": float(min(global_test_fprs)),
                    "gap": gap_global,
                    "mean": float(np.mean(global_test_fprs)),
                },
                "test_fpr_under_perpool_tau": {
                    "per_pool": dict(zip(POOL_ORDER, perpool_test_fprs)),
                    "max": float(max(perpool_test_fprs)),
                    "min": float(min(perpool_test_fprs)),
                    "gap": gap_perpool,
                    "mean": float(np.mean(perpool_test_fprs)),
                },
                "gap_closure": gap_closure,
            }
        )

    # Supplementary diagnostic: pool-mean structure independent of any tau.
    # If scores are pure mean-shift, subtracting per-pool means should collapse
    # the cross-pool distribution onto one common distribution.
    raw_pool_means = np.array(
        [np.mean(pools[n]) for n in POOL_ORDER]
    )
    # Residual after subtracting per-pool mean is mean-zero by construction,
    # so cross-pool spread collapses by definition. Instead report pool-mean
    # spread and pool-std structure for interpretation.
    pool_stats = [
        {
            "pool": n,
            "is_failing": POOL_IS_FAILING[n],
            "mean": float(np.mean(pools[n])),
            "std": float(np.std(pools[n])),
            "n": int(len(pools[n])),
        }
        for n in POOL_ORDER
    ]

    # Mean gap closure across target FPR levels -- the headline number.
    avg_gap_closure = float(
        np.mean([r["gap_closure"] for r in results_by_target])
    )

    if avg_gap_closure >= 0.60:
        verdict = "calibration_is_right_lever"
    elif avg_gap_closure <= 0.30:
        verdict = "training_augmentation_is_right_lever"
    else:
        verdict = "mixed_both_levers_needed"

    return {
        "config": {
            "data_path": str(DATA_PATH),
            "seed": SEED,
            "target_fprs": TARGET_FPRS,
            "pool_order": POOL_ORDER,
        },
        "pool_stats": pool_stats,
        "results_by_target_fpr": results_by_target,
        "summary": {
            "avg_gap_closure_across_target_fprs": avg_gap_closure,
            "gap_closure_per_target_fpr": {
                r["target_fpr"]: r["gap_closure"] for r in results_by_target
            },
            "verdict": verdict,
            "verdict_thresholds": {
                "calibration_is_right_lever": ">= 0.60",
                "training_augmentation_is_right_lever": "<= 0.30",
            },
        },
    }


def print_report(report: dict) -> None:
    print("=" * 78)
    print("Per-camera calibration probe (WS-P1)")
    print("=" * 78)
    print()
    print(f"Data: {report['config']['data_path']}")
    print(f"Seed: {report['config']['seed']}  Pools: {len(report['config']['pool_order'])}")
    print()
    print("Pool raw score stats (all 30 frames, REAL subjects only):")
    print(f"  {'pool':<55} {'n':>3} {'mean':>7} {'std':>7} {'fail?':>6}")
    for s in report["pool_stats"]:
        tag = "yes" if s["is_failing"] else "no"
        print(f"  {s['pool']:<55} {s['n']:>3d} {s['mean']:>7.4f} {s['std']:>7.4f} {tag:>6}")
    print()
    print("FPR gap reduction (test halves, per target FPR level):")
    print(f"  {'target':>7} {'gap_global':>12} {'gap_perpool':>12} {'gap_closure':>12}")
    for r in report["results_by_target_fpr"]:
        print(
            f"  {r['target_fpr']:>7.2f} "
            f"{r['test_fpr_under_global_tau']['gap']:>12.3f} "
            f"{r['test_fpr_under_perpool_tau']['gap']:>12.3f} "
            f"{r['gap_closure']:>12.3f}"
        )
    print()
    print("Per-pool FPR under global tau (target=0.10):")
    for r in report["results_by_target_fpr"]:
        if abs(r["target_fpr"] - 0.10) < 1e-6:
            for pool, fpr in r["test_fpr_under_global_tau"]["per_pool"].items():
                tag = "FAIL" if POOL_IS_FAILING[pool] else "clean"
                print(f"  [{tag}] {pool:<55} test FPR = {fpr:.3f}")
    print()
    s = report["summary"]
    print(
        f"Avg gap closure across target FPRs: "
        f"{s['avg_gap_closure_across_target_fprs']:.3f}"
    )
    print(f"VERDICT: {s['verdict']}")
    print()
    print(
        "Decision gates:\n"
        "  >= 0.60  calibration is the right lever (descope WS-P4 training augs)\n"
        "  <= 0.30  training-time augmentation is the right lever\n"
        "  0.30-0.60 both in parallel"
    )


def main() -> None:
    report = run_probe()
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print_report(report)
    print()
    print(f"Wrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
