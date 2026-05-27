"""End-to-end orchestrator for the production-aggregation calibration framework.

Usage:
    python run_pipeline.py --ckpt e2b
    python run_pipeline.py --ckpt e2b --phases 1,2,3,4,5
    python run_pipeline.py --ckpt e2b --phases 4 --workers 10

Adding a new ckpt: add an entry to configs/ckpts.yaml.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

PHASE_MODULES = {
    "1": "phase1_distributions",
    "2": "phase2_within_video",
    "3": "phase3_temporal",
    "4": "phase4_pareto_sweep",
    "5": "phase5_stratified_eval",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="ckpt key from configs/ckpts.yaml")
    ap.add_argument("--phases", default="1,2,3,4,5",
                    help="comma-separated phase indices (default 1,2,3,4,5)")
    ap.add_argument("--workers", type=int, default=8, help="phase4 worker count")
    args = ap.parse_args()

    requested = [p.strip() for p in args.phases.split(",") if p.strip()]
    for p in requested:
        if p not in PHASE_MODULES:
            raise SystemExit(f"unknown phase: {p}")

    t_total = time.time()
    timings = []
    for p in requested:
        modname = PHASE_MODULES[p]
        mod = importlib.import_module(modname)
        t0 = time.time()
        if modname == "phase4_pareto_sweep":
            mod.N_WORKERS = args.workers
        mod.main(args.ckpt)
        dt = time.time() - t0
        timings.append((modname, dt))
        print(f"  Phase {p} ({modname}): {dt:.1f}s")

    total = time.time() - t_total
    print(f"\nPipeline complete for ckpt={args.ckpt} in {total:.1f}s")
    print("Per-phase timings:")
    for modname, dt in timings:
        print(f"  {modname}: {dt:.1f}s")


if __name__ == "__main__":
    main()
