"""Shared helpers for the P22 CPU follow-up analyses (2026-05-02 evening)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Per-frame report locations
P22_FRAMES_DIR = Path("/tmp/p22_frames_local")
P18_FRAMES_DIR = Path("analysis/score_distribution_2026-05-02/raw_reports")
ATTRS_CSV = Path("analysis/score_distribution_2026-05-02/outputs/cross_suite_attributes.csv")

# Promotion contract grids
P22_GRID = Path("analysis/p22_eval_2026-05-02/scorecard_data/promotion_contract/threshold_grid.csv")
P18_GRID = Path("analysis/p18_probe_2026-05-01/d_results/promotion_contract/threshold_grid.csv")

OUT = Path("analysis/p22_eval_2026-05-02/cpu_followups/outputs")
FIG = Path("analysis/p22_eval_2026-05-02/cpu_followups/figures")

# Suite -> bin label (real=0, fake=1)
SUITE_LABEL = {
    "teams_real_all_dev": 0,
    "teams_real_all_lockbox": 0,
    "teams_real_dor_dev": 0,
    "teams_real_lighting_extreme_dev": 0,
    "teams_real_poor_quality_dev": 0,
    "teams_fake_all_dev": 1,
    "teams_fake_all_lockbox": 1,
    "visomaster_enhanced_macro_dev": 1,
    "deeplive_enhanced_dev": 1,
}

REAL_SUITES = [s for s, l in SUITE_LABEL.items() if l == 0]
FAKE_SUITES = [s for s, l in SUITE_LABEL.items() if l == 1]
ALL_SUITES = REAL_SUITES + FAKE_SUITES


# Checkpoint-key suffix used in filenames (lower-case, no leading/trailing underscores)
CKPT_FILE_SUFFIX: Dict[str, str] = {
    "P8A_REFERENCE_STEP5000": "p8a_reference_step5000",
    "P22_AUG_STEP1000": "p22_aug_step1000",
    "P22_AUG_STEP4000": "p22_aug_step4000",
    "P22_AUG_STEP8000": "p22_aug_step8000",
    "P18T_GRL_TREATMENT_STEP4000": "p18t_grl_treatment_step4000",
    "P18C_NO_GRL_CONTROL_STEP4000": "p18c_no_grl_control_step4000",
}

# Map ckpt -> directory with its per-frame CSVs
CKPT_FRAMES_DIR: Dict[str, Path] = {
    "P8A_REFERENCE_STEP5000": P22_FRAMES_DIR,  # Use the freshest P8A run
    "P22_AUG_STEP1000": P22_FRAMES_DIR,
    "P22_AUG_STEP4000": P22_FRAMES_DIR,
    "P22_AUG_STEP8000": P22_FRAMES_DIR,
    "P18T_GRL_TREATMENT_STEP4000": P18_FRAMES_DIR,
    "P18C_NO_GRL_CONTROL_STEP4000": P18_FRAMES_DIR,
}


def load_per_frame_scores(ckpt: str, suite: str) -> Optional[pd.DataFrame]:
    """Load the per-frame report CSV for (ckpt, suite). Returns None if missing."""
    suffix = CKPT_FILE_SUFFIX[ckpt]
    base = CKPT_FRAMES_DIR[ckpt]
    fname = f"{suite}_{suffix}_frames_report.csv"
    path = base / fname
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["suite"] = suite
    df["checkpoint"] = ckpt
    return df


def load_pool(ckpt: str, suites=None) -> pd.DataFrame:
    """Load and concat per-frame scores across all suites for a checkpoint."""
    if suites is None:
        suites = ALL_SUITES
    parts = []
    for s in suites:
        df = load_per_frame_scores(ckpt, s)
        if df is None:
            continue
        df["bin_label"] = SUITE_LABEL[s]
        parts.append(df)
    if not parts:
        raise RuntimeError(f"No per-frame data loaded for {ckpt}")
    return pd.concat(parts, ignore_index=True)


def fpr_at_tau(real_scores, tau):
    if len(real_scores) == 0:
        return float("nan")
    return float((real_scores >= tau).mean())


def recall_at_tau(fake_scores, tau):
    if len(fake_scores) == 0:
        return float("nan")
    return float((fake_scores >= tau).mean())
