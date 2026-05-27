#!/usr/bin/env python3
"""Build a checkpoint_map yaml for an E-packet (E1/E2/E3) from its W&B run dir.

Lists periodic checkpoints in gs://training-job-outputs/phase2r13_experiments/<run_id>/,
filters to a target subset (default: every 500 steps + trainer's best ckpts),
and emits a yaml with stable CKPT_KEY → gs:// path entries for the scorecard.

Always includes P8A_REFERENCE_STEP5000 as the side-by-side baseline.

Usage:
  python arena/build_e_packet_checkpoint_map.py --packet E1 --run-id x1q3csh3 \
    --out arena/checkpoint_maps/teams_target_domain.e1_2026-05-03.yaml
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

P8A_BASELINE = (
    "P8A_REFERENCE_STEP5000",
    "gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth",
)

# Default step subsets per packet — minimize scorecard cost while sampling
# the training trajectory (early / mid / late).
DEFAULT_STEP_SUBSETS = {
    "E1": [200, 500, 1000, 1500, 2000],          # 2000-step cap
    "E2": [500, 1500, 3000, 5000, 7000, 8000],   # 8000-step scratch
    "E3": [500, 1500, 3000, 5000, 7000, 8000],   # 8000-step scratch
}

PERIODIC_RE = re.compile(r"periodic_effort_\d+_step(\d+)_auc[\d.]+_eer[\d.]+\.pth")
BEST_RE = re.compile(r"(value_composite|top_n|first_best|ood_composite)_effort_\d+_(?:step\d+_)?(?:ep\d+_)?auc[\d.]+_eer[\d.]+\.pth")


def list_ckpts(run_id: str) -> list[str]:
    """List all .pth files for a run across both known GCS prefixes.

    Layout history:
      - Pre-2026-05-03 runs: gs://training-job-outputs/phase2r13_experiments/<run_id>/
      - From 2026-05-03 build (1.3.249+): gs://training-job-outputs/best_checkpoints/<run_id>/
        (best/top_n/first_best go here; periodic_*.pth may also live here)
    Returns the union of both prefixes' .pth files.
    """
    candidates = [
        f"gs://training-job-outputs/phase2r13_experiments/{run_id}/",
        f"gs://training-job-outputs/best_checkpoints/{run_id}/",
    ]
    all_paths: list[str] = []
    for uri in candidates:
        try:
            out = subprocess.check_output(["gsutil", "ls", uri], text=True, timeout=30, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            continue  # Prefix doesn't exist for this run; that's OK.
        all_paths.extend(line.strip() for line in out.splitlines() if line.strip().endswith(".pth"))
    return all_paths


def pick_periodic(ckpts: list[str], wanted_steps: list[int]) -> dict[int, str]:
    """Map step → gs:// path for periodic_*.pth matching wanted_steps."""
    by_step: dict[int, str] = {}
    for path in ckpts:
        name = path.rsplit("/", 1)[-1]
        m = PERIODIC_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        if step in wanted_steps:
            by_step[step] = path
    return by_step


def pick_trainer_best(ckpts: list[str]) -> list[str]:
    """Return paths of trainer's curated best ckpts (value_composite, top_n, first_best).

    Skips periodic_*.pth (those are for periodic_saves)."""
    return [p for p in ckpts if BEST_RE.match(p.rsplit("/", 1)[-1])]


def build_map(packet: str, run_id: str, wanted_steps: list[int] | None) -> dict[str, str]:
    if wanted_steps is None:
        wanted_steps = DEFAULT_STEP_SUBSETS.get(packet, [500, 1000, 2000])

    ckpts = list_ckpts(run_id)
    if not ckpts:
        raise SystemExit(f"No checkpoints found at gs://training-job-outputs/phase2r13_experiments/{run_id}/")

    out: dict[str, str] = {}
    out[P8A_BASELINE[0]] = P8A_BASELINE[1]

    by_step = pick_periodic(ckpts, wanted_steps)
    for step in sorted(by_step):
        key = f"{packet}_STEP{step}"
        out[key] = by_step[step]

    # Also include the trainer's curated best ckpts (value_composite, top_n).
    best_ckpts = pick_trainer_best(ckpts)
    for path in best_ckpts:
        name = path.rsplit("/", 1)[-1]
        # Extract a unique key suffix from the filename
        m = re.match(r"(value_composite|top_n|first_best|ood_composite)_effort_\d+_(?:step(\d+)|ep(\d+))_auc([\d.]+)_eer([\d.]+)\.pth", name)
        if not m:
            continue
        kind = m.group(1).upper()
        step_or_ep = m.group(2) or m.group(3)
        suffix = "STEP" if m.group(2) else "EP"
        key = f"{packet}_{kind}_{suffix}{step_or_ep}"
        out[key] = path

    return out


def emit_yaml(packet: str, run_id: str, ckpt_map: dict[str, str], out_path: Path) -> None:
    today = "2026-05-03"
    lines = [
        f"# Promotion-contract validation map for {packet} (run {run_id})",
        f"#",
        f"# Auto-generated 2026-05-03 by arena/build_e_packet_checkpoint_map.py.",
        f"# Includes P8A_REFERENCE_STEP5000 baseline for direct comparison.",
        f"# ",
        f"# Packet context (per yaml R13_{packet}_*.yaml):",
        f"#   - E1: B16 FT-from-P8A + heavier eval-targeted aug curriculum",
        f"#   - E2: B16 SCRATCH from CLIP + same eval-targeted aug",
        f"#   - E3: L14 SCRATCH from CLIP + same eval-targeted aug",
        f"# Falsifier targets per yaml: viso recall at joint dev+lockbox FPR=10% > 35%",
        f"#",
    ]
    for key, path in ckpt_map.items():
        lines.append(f'{key}:    "{path}"')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path} with {len(ckpt_map)} entries:")
    for key in ckpt_map:
        print(f"  {key}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--packet", required=True, choices=["E1", "E2", "E3"])
    p.add_argument("--run-id", required=True, help="W&B run ID (folder name in GCS bucket)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--steps", default=None, help="Comma-separated step list (overrides default subset)")
    args = p.parse_args()

    wanted = None
    if args.steps:
        wanted = [int(s.strip()) for s in args.steps.split(",")]

    ckpt_map = build_map(args.packet, args.run_id, wanted)
    emit_yaml(args.packet, args.run_id, ckpt_map, args.out)


if __name__ == "__main__":
    main()
