"""Resolve `gs://local/...` paths to local filesystem paths.

The `bucket=local` rows in grouped_manifest_v2.csv have frame_paths like
`gs://local/<cohort_dir>/<filename>` which are NOT real GCS — they're a
marker that the frame lives in one of the analysis/raw subtrees.

This module provides a single function `resolve_local(uri)` that searches
the candidate dirs and returns the absolute local path if found, else None.

Mapping (discovered by inspection):
- gs://local/faces_dor/evening/<file> -> analysis/dor_evening_2026-05-05/raw/all/dor_evening/<file>
- gs://local/faces_dor/morning/<file> -> analysis/dor_morning_2026-05-05/raw/all/dor_morning/<file>
- gs://local/dor_deep_live_cam/<cohort>/<file> -> analysis/dor_fake_local_2026-05-05/raw/all/<cohort>/<file>
                                                or analysis/new_data_batch_2026-05-05/raw/all/<cohort>/<file>
- gs://local/extra/<cohort>/<file> -> analysis/extra_2026-05-05/raw/all/<cohort>/<file>
- (visomaster fake): gs://local/dor_deep_live_cam/dor_fake_<inswap|instyle|ghost|sim>* -> visomaster_v2_2026-05-05/raw/all/<cohort>/<file>

We cache the directory listings for fast repeated lookup.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]

# Roots to search for local files (ordered: first match wins)
LOCAL_ROOTS = [
    REPO_ROOT / "analysis/dor_evening_2026-05-05/raw/all",
    REPO_ROOT / "analysis/dor_morning_2026-05-05/raw/all",
    REPO_ROOT / "analysis/dor_fake_local_2026-05-05/raw/all",
    REPO_ROOT / "analysis/extra_2026-05-05/raw/all",
    REPO_ROOT / "analysis/new_data_batch_2026-05-05/raw/all",
    REPO_ROOT / "analysis/visomaster_v2_2026-05-05/raw/all",
    REPO_ROOT / "analysis/live_reals_2026-05-05/raw/all",
    REPO_ROOT / "analysis/live_fakes_teams_prod_2026-05-05/raw/all",
    REPO_ROOT / "analysis/dor_drift_mechanism_2026-05-06/raw",
]


@lru_cache(maxsize=None)
def _list_dir(d: Path) -> set:
    if not d.exists():
        return set()
    return set(os.listdir(d))


def resolve_local(uri: str) -> Optional[str]:
    """Map gs://local/<cohort>/<file> or gs://local/<sub>/<cohort>/<file> to a local absolute path.

    Also handles `gs://live-fakes-teams-prod/real/<cohort>/<file>` for the
    Roee_Windows real cohorts whose manifest paths were stale (the actual GCS
    layout uses `session_*/...` but the manifest used the cohort-name shortcut).

    The cohort name (penultimate path component) is the discriminator.
    """
    if not uri.startswith("gs://"):
        return None
    # Handle live-fakes-teams-prod fall-through (manifest paths use cohort-name not session-name)
    if uri.startswith("gs://live-fakes-teams-prod/real/"):
        rel = uri[len("gs://live-fakes-teams-prod/real/"):]  # cohort/file
    elif uri.startswith("gs://local/"):
        rel = uri[len("gs://local/"):]  # e.g., "faces_dor/evening/frame_x.png"
    else:
        return None
    parts = rel.split("/")
    if len(parts) < 2:
        return None
    fname = parts[-1]
    # cohort = last directory component before filename
    cohort = parts[-2]

    # Specific cohort renaming
    cohort_renames = {
        "evening": "dor_evening",
        "morning": "dor_morning",
    }
    cohort_to_try = [cohort, cohort_renames.get(cohort, cohort)]

    for root in LOCAL_ROOTS:
        for cname in cohort_to_try:
            d = root / cname
            files = _list_dir(d)
            if fname in files:
                return str((d / fname).resolve())
    # As fallback: try just /raw/<cohort>/file (some have raw, no raw/all)
    extra_roots = [
        REPO_ROOT / "analysis/dor_fake_local_2026-05-05/raw",
        REPO_ROOT / "analysis/extra_2026-05-05/raw",
        REPO_ROOT / "analysis/visomaster_v2_2026-05-05/raw",
        REPO_ROOT / "analysis/dor_evening_2026-05-05/raw",
        REPO_ROOT / "analysis/dor_morning_2026-05-05/raw",
        REPO_ROOT / "analysis/new_data_batch_2026-05-05/raw",
    ]
    for root in extra_roots:
        for cname in cohort_to_try:
            d = root / cname
            files = _list_dir(d)
            if fname in files:
                return str((d / fname).resolve())
    return None


if __name__ == "__main__":
    # Quick smoke test
    import sys
    test_uris = [
        "gs://local/faces_dor/evening/frame_000001_seq388.png",
        "gs://local/faces_dor/morning/frame_000001_seq44.png",
        "gs://local/dor_deep_live_cam/dor_fake_elone_enhanced/frame_001173_seq1453.png",
        "gs://local/extra/extra_xiang_real/frame_000043_seq323.png",
        "gs://local/extra/extra_xiang_fake/frame_002207_seq3781.png",
        "gs://local/extra/extra_xinghe_real/frame_000001_seq370.png",
        "gs://local/extra/extra_xinghe_fake/frame_000001_seq1790.png",
    ]
    for u in test_uris:
        r = resolve_local(u)
        status = "OK" if (r and Path(r).exists()) else "MISSING"
        print(f"{status:8s} {u} -> {r}")
