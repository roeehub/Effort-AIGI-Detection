#!/usr/bin/env python3
"""
Prepare a short fixed-seed training-speedup validation config.

The generated config keeps the source experiment's seed and learning schedule
intact, but makes the run short and comparable by:
  - stopping after a fixed number of train steps
  - suppressing validation and OOD monitoring overhead
  - forcing frequent W&B progress logging
  - reducing checkpoint churn
"""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any, Dict, Sequence

import yaml


DEFAULT_MAX_TRAIN_STEPS = 100
DEFAULT_LOG_PROGRESS_STEPS = 1


def _sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(label).strip().lower())
    return cleaned.strip("-") or "validation"


def _append_unique(items: Sequence[Any], value: str) -> list[str]:
    out: list[str] = []
    for item in items:
        text = str(item)
        if text not in out:
            out.append(text)
    if value not in out:
        out.append(value)
    return out


def apply_short_validation_overrides(
    config: Dict[str, Any],
    *,
    label: str,
    max_train_steps: int = DEFAULT_MAX_TRAIN_STEPS,
    log_progress_steps: int = DEFAULT_LOG_PROGRESS_STEPS,
) -> Dict[str, Any]:
    """
    Return a derived config for a short fixed-seed validation run.

    The source config is not mutated.
    """
    if max_train_steps < 1:
        raise ValueError("max_train_steps must be >= 1")
    if log_progress_steps < 1:
        raise ValueError("log_progress_steps must be >= 1")

    cfg = copy.deepcopy(config)
    label_slug = _sanitize_label(label)
    holdoff_steps = int(max_train_steps) + 1000

    base_name = str(cfg.get("name") or "unnamed_experiment")
    cfg["name"] = f"{base_name}__speedval_{label_slug}"

    desc = str(cfg.get("description") or "").strip()
    suffix = f"[training speedup validation: {label_slug}; max_train_steps={max_train_steps}]"
    cfg["description"] = f"{desc} {suffix}".strip()

    cfg["max_train_steps"] = int(max_train_steps)
    cfg["evaluate_every_steps"] = holdoff_steps
    cfg["ood_monitoring_enabled"] = False
    cfg["ood_monitoring_start_step"] = holdoff_steps
    cfg["ood_monitoring_every_steps"] = holdoff_steps
    cfg["early_stopping_enabled"] = False

    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        wandb_cfg = {}
    tags = wandb_cfg.get("tags") or []
    tags = _append_unique(tags, "training-speedup-validation")
    tags = _append_unique(tags, "fixed-seed")
    tags = _append_unique(tags, label_slug)
    wandb_cfg["tags"] = tags
    wandb_cfg["log_progress_steps"] = int(log_progress_steps)
    cfg["wandb"] = wandb_cfg

    checkpointing = cfg.get("checkpointing")
    if isinstance(checkpointing, dict):
        checkpointing = copy.deepcopy(checkpointing)
        current_save_every = checkpointing.get("save_every_steps")
        try:
            current_save_every_int = int(current_save_every)
        except (TypeError, ValueError):
            current_save_every_int = 0
        checkpointing["save_every_steps"] = max(current_save_every_int, holdoff_steps)
        checkpointing["keep_last_n"] = 1
        cfg["checkpointing"] = checkpointing

    combined_cfg = cfg.get("combined_paired")
    if isinstance(combined_cfg, dict):
        combined_cfg = copy.deepcopy(combined_cfg)
        ood_cfg = combined_cfg.get("ood_monitoring")
        if isinstance(ood_cfg, dict):
            ood_cfg = copy.deepcopy(ood_cfg)
            ood_cfg["enabled"] = False
            combined_cfg["ood_monitoring"] = ood_cfg
        cfg["combined_paired"] = combined_cfg

    top_level_ood_cfg = cfg.get("ood_monitoring")
    if isinstance(top_level_ood_cfg, dict):
        top_level_ood_cfg = copy.deepcopy(top_level_ood_cfg)
        top_level_ood_cfg["enabled"] = False
        cfg["ood_monitoring"] = top_level_ood_cfg

    return cfg


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a top-level mapping")
    return data


def _write_yaml(path: Path, config: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a short fixed-seed config for training-speedup validation.",
    )
    parser.add_argument("--input-config", required=True, help="Source experiment YAML path.")
    parser.add_argument("--output-config", required=True, help="Output YAML path.")
    parser.add_argument("--label", required=True, help="Short label such as 'main' or 'speedup'.")
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=DEFAULT_MAX_TRAIN_STEPS,
        help=f"Maximum train steps for the validation run (default: {DEFAULT_MAX_TRAIN_STEPS}).",
    )
    parser.add_argument(
        "--log-progress-steps",
        type=int,
        default=DEFAULT_LOG_PROGRESS_STEPS,
        help=(
            "W&B progress logging cadence to bake into the derived config "
            f"(default: {DEFAULT_LOG_PROGRESS_STEPS})."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output config if it already exists.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input_config)
    output_path = Path(args.output_config)

    if not input_path.exists():
        raise SystemExit(f"Input config does not exist: {input_path}")
    if output_path.exists() and not args.force:
        raise SystemExit(f"Output config already exists (use --force to overwrite): {output_path}")

    source_config = _load_yaml(input_path)
    derived_config = apply_short_validation_overrides(
        source_config,
        label=args.label,
        max_train_steps=args.max_train_steps,
        log_progress_steps=args.log_progress_steps,
    )
    _write_yaml(output_path, derived_config)

    print(f"Wrote validation config: {output_path}")
    print(f"  source: {input_path}")
    print(f"  label: {args.label}")
    print(f"  max_train_steps: {args.max_train_steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
