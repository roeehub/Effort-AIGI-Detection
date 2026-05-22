#!/usr/bin/env python
"""Check that the T5C smoke logged a non-zero substrate_pair_asymmetric_loss.

Exit code 0 + stdout 'OK ...' = smoke passes the loss-fires criterion.
Exit code 1 + stdout 'NO_METRIC|NO_VALUES|FIRES_ZERO|ERROR ...' = blocked.

Usage:
    python check_t5c_smoke_loss.py --run-id <wandb_run_id>
"""
import argparse
import os
import sys

import wandb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--entity", default="dtect-vision")
    ap.add_argument("--project", default="effort-r13-phase2")
    ap.add_argument("--api-key", default=os.environ.get("WANDB_API_KEY",
        "bb5a8ea4a27ebe45917587df8c46674d26e43966"))
    args = ap.parse_args()

    os.environ["WANDB_API_KEY"] = args.api_key
    api = wandb.Api()
    try:
        run = api.run(f"{args.entity}/{args.project}/{args.run_id}")
        hist = run.history(keys=["train/loss/substrate_pair_asymmetric"], samples=500, pandas=True)
        if hist.empty or "train/loss/substrate_pair_asymmetric" not in hist.columns:
            print("NO_METRIC")
            return 1
        col = hist["train/loss/substrate_pair_asymmetric"].dropna()
        if len(col) == 0:
            print("NO_VALUES")
            return 1
        n_nonzero = int((col > 0).sum())
        if n_nonzero >= 5:
            print(f"OK n={len(col)} nonzero={n_nonzero} max={col.max():.4f} mean={col.mean():.4f}")
            return 0
        print(f"FIRES_ZERO n={len(col)} nonzero={n_nonzero} max={col.max():.4f}")
        return 1
    except Exception as e:
        print(f"ERROR {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
