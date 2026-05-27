"""Shared utilities for the aggregation calibration framework."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml


# scripts/.. = aggregation_calibration_2026-05-05; .../analysis/...; .../training/
# layout: training / analysis / aggregation_calibration_2026-05-05 / scripts / _common.py
# parents[0]=scripts, [1]=aggregation_..., [2]=analysis, [3]=training
FW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]  # = .../DeepfakeBench/training


def load_ckpts_yaml() -> dict:
    p = FW_ROOT / "configs" / "ckpts.yaml"
    return yaml.safe_load(p.read_text())


def load_policy_grid() -> dict:
    p = FW_ROOT / "configs" / "policy_grid.yaml"
    return yaml.safe_load(p.read_text())


def get_ckpt_cfg(ckpt_name: str) -> dict:
    cfg = load_ckpts_yaml()
    if ckpt_name not in cfg["ckpts"]:
        raise KeyError(f"ckpt {ckpt_name!r} not in configs/ckpts.yaml")
    return cfg["ckpts"][ckpt_name]


def out_dirs(ckpt_name: str) -> Dict[str, Path]:
    full = load_ckpts_yaml()["ckpts"][ckpt_name]["full_name"]
    return {
        "data": FW_ROOT / "data" / full,
        "findings": FW_ROOT / "findings" / full,
        "figures": FW_ROOT / "figures" / full,
        "full_name": Path(full),
    }


def ensure_dirs(ckpt_name: str) -> Dict[str, Path]:
    d = out_dirs(ckpt_name)
    for k, p in d.items():
        if k != "full_name":
            p.mkdir(parents=True, exist_ok=True)
    return d


def resolve_csv(rel_or_abs: str) -> str:
    """If the path in YAML is repo-relative, resolve from REPO_ROOT."""
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return str(REPO_ROOT / rel_or_abs)


def load_suite_csv(ckpt_cfg: dict, suite_name: str) -> pd.DataFrame:
    suite = ckpt_cfg["suites"][suite_name]
    csv_path = resolve_csv(suite["csv"])
    df = pd.read_csv(csv_path)
    df["suite"] = suite_name
    df["label_class"] = suite["label_class"]
    return df
