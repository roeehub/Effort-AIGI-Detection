"""Shared helpers for the 2026-05-19 pre-plan CPU diagnostics."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "analysis/cpu_diagnostics_2026-05-19_pre_plan"
GCS_CACHE = HERE / "gcs_cache"
OUT = HERE / "outputs"
FIGS = HERE / "figs"
LOGS = HERE / "logs"

CKPTS = [
    ("P8A_REFERENCE_STEP5000", "p8a_reference_step5000", 0.915605),
    ("T5C_PERIODIC_STEP3500", "t5c_periodic_step3500", 0.830874),
    ("SLOT_A_RESCHAIN_STEP1500", "slot_a_reschain_step1500", 0.898922),
    ("SLOT_B_6AXIS_GRL_STEP3500", "slot_b_6axis_grl_step3500", 0.816038),
    ("SLOT_A_RESCHAIN_STEP3500", "slot_a_reschain_step3500", 0.859501),
]

CHRONIC_PATTERNS = [
    "PC_Generator",
    "Roy_D",
    "bla_bla_chow",
    "dor_shkedi",
    "xiang",
    "Chikara_Takahashi",
    "Cam_Test",
    "dor",  # short root
]


def extract_identity(video_id: str) -> str:
    """Robust identity extraction from lockbox video_id.

    Lockbox video_ids encode identity at multiple granularities:
      - 'Chikara_Takahashi__s22__seg_10.0__real'  → 'Chikara_Takahashi__s22'  (__s session tag)
      - 'dor_shkedi__seq4490__real'                → 'dor_shkedi'              (__seq is per-clip sequence, NOT session)
      - 'real_dor__seq3982__real'                  → 'real_dor'                (same)
      - 'PC_Generator__s14_video__seg_3__real'     → 'PC_Generator__s14'

    The joint-marginal table uses identity keys of form `<root>__s<N>` (session)
    or just `<root>`. So __seq → drop; __s<N> → keep.
    """
    if not isinstance(video_id, str):
        return "unknown"
    # First strip __seg_* (segment-within-clip)
    head = video_id.split("__seg_")[0]
    # Now drop trailing __seq<N> (per-clip sequence ID — not session)
    head = re.sub(r"__seq\d+.*$", "", head)
    # Drop trailing __real / __fake suffixes
    head = re.sub(r"__(real|fake)$", "", head)
    return head


def identity_root(identity_key: str) -> str:
    """Strip __sNN suffix to get root identity."""
    return re.sub(r"__s\d+$", "", identity_key)


def is_chronic(identity_key: str) -> bool:
    root = identity_root(identity_key)
    return any(pat.lower() in root.lower() or pat.lower() in identity_key.lower() for pat in CHRONIC_PATTERNS)


def load_lockbox_frames(suite: str = "teams_real_all_lockbox") -> pd.DataFrame:
    """Merge per-ckpt frames_reports into a single wide DataFrame keyed by frame_path."""
    pieces = []
    for key, slug, tau in CKPTS:
        path = GCS_CACHE / f"{suite}_{slug}_frames_report.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df.rename(columns={"frame_prob": f"prob_{key}"})
        if not pieces:
            pieces.append(df[["frame_path", "video_id", "label", "method", "group_key", "family_key", f"prob_{key}"]])
        else:
            pieces.append(df[["frame_path", f"prob_{key}"]])
    if not pieces:
        return pd.DataFrame()
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on="frame_path", how="outer")
    out["identity_key"] = out["video_id"].apply(extract_identity)
    out["identity_root"] = out["identity_key"].apply(identity_root)
    out["is_chronic"] = out["identity_key"].apply(is_chronic)
    return out


def load_bimodal_partition() -> pd.DataFrame:
    """Per-identity median train/lockbox density ratio."""
    p = ROOT / "analysis/joint_marginal_audit_2026-05-19/tables/per_chronic_identity_location.csv"
    df = pd.read_csv(p)

    def coverage(r):
        if r < 0.5:
            return "under_covered"
        if r > 100:
            return "over_covered"
        return "moderate"

    df["coverage_class"] = df["median_ratio"].apply(coverage)
    return df


def load_unified_tags() -> pd.DataFrame:
    p = ROOT / "analysis/joint_marginal_audit_2026-05-19/artifacts/unified_tags.parquet"
    return pd.read_parquet(p)


def load_lockbox_full_tags() -> pd.DataFrame:
    p = ROOT / "analysis/lockbox_tagging/full_tags_2026-04-27.parquet"
    return pd.read_parquet(p)
