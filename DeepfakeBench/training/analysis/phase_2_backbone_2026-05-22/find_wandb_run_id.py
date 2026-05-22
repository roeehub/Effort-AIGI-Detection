#!/usr/bin/env python
"""Find W&B run ID for a given experiment-config name.

Usage:
    python find_wandb_run_id.py --name R13_GROUPDRO_SUBSTRATE_SLOTAV2_2026-05-26
    python find_wandb_run_id.py --name R13_PAIR_LOSS_ASYM_T5C_2026-05-26

Prints the run ID (8-char alphanumeric) to stdout on success, or "" if
no matching run found (e.g. job hasn't logged config.name yet).

Caveats:
- Queries the dtect-vision/effort-r13-phase2 W&B project
- Filters by config.name (the yaml's `name:` field which the training
  script writes via wandb.init(config=cfg))
- If multiple matches (e.g. retries), returns the most recent by
  created_at timestamp
"""
import argparse
import os
import sys

import wandb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="config.name value (yaml's name: field)")
    ap.add_argument("--entity", default="dtect-vision")
    ap.add_argument("--project", default="effort-r13-phase2")
    ap.add_argument("--api-key", default=os.environ.get("WANDB_API_KEY",
        "bb5a8ea4a27ebe45917587df8c46674d26e43966"))
    args = ap.parse_args()

    os.environ["WANDB_API_KEY"] = args.api_key
    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}"
    runs = api.runs(
        project_path,
        filters={"config.name": args.name},
        order="-created_at",
    )
    runs_list = list(runs)
    if not runs_list:
        print("", end="")
        return 1
    print(runs_list[0].id, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
