"""
Identity-based splitting — standalone port of the split logic from
combined_paired.py.  No torch dependency.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class SampleInfo:
    """Lightweight sample metadata (no image data)."""
    sample_id: str
    source: str          # df40 | deeplive | visomaster | visomaster_enhanced |
                         # visomaster_teams_enhanced | deeplive_teams | external
    method: str          # e.g. "simswap", "visomaster_CSCS", "deeplive_edge_cases"
    identity: str        # prefixed: df40_001, realpool_xyz, external_vcd_abc
    label: int           # 0=real, 1=fake (paired samples appear twice — once for each label)
    bucket: str          # GCS bucket name
    frame_count: int = 0
    strategy: str = ""
    swap_model: str = ""
    enhancer: str = ""
    tier: str = ""
    original_video_name: str = ""
    has_pair: bool = False         # has real/fake pair in same sample dir
    has_teams_counterpart: bool = False
    teams_bucket: str = ""
    sampling_family_key: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # Assigned after splitting
    split: str = ""      # train | val_in_dist | test


def extract_identity_from_video_name(name: str) -> str:
    """Strip cropped_ prefix and .mp4 suffix → identity string."""
    identity = name
    if identity.startswith("cropped_"):
        identity = identity[8:]
    if identity.endswith(".mp4"):
        identity = identity[:-4]
    return identity


def build_identity(source: str, raw_identity: str) -> str:
    """Apply the same identity-prefix scheme as combined_paired.py."""
    if source == "df40":
        return f"df40_{raw_identity}"
    if source == "external":
        return f"external_vcd_{raw_identity}"
    # deeplive, visomaster, visomaster_enhanced, visomaster_teams_enhanced,
    # visomaster_res_variant, deeplive_teams — all share realpool_
    return f"realpool_{raw_identity}"


def presplit_external_reals(
    samples: List[SampleInfo],
    identity_train_fraction: float = 0.40,
    identity_split_seed: int = 737,
) -> List[SampleInfo]:
    """
    Pre-split external reals: only keep training-fraction identities.
    Matches the logic in combined_paired._discover_external_training_reals().
    """
    ext = [s for s in samples if s.source == "external"]
    non_ext = [s for s in samples if s.source != "external"]
    if not ext:
        return samples

    by_identity: Dict[str, List[SampleInfo]] = defaultdict(list)
    for s in ext:
        by_identity[s.identity].append(s)

    all_ids = sorted(by_identity.keys())
    rng = random.Random(identity_split_seed)
    rng.shuffle(all_ids)
    n_train = max(1, int(len(all_ids) * identity_train_fraction))
    train_ids = set(all_ids[:n_train])

    kept = [s for s in ext if s.identity in train_ids]
    return non_ext + kept


def split_samples_by_identity(
    samples: List[SampleInfo],
    train_split: float = 0.85,
    val_split: float = 0.10,
    seed: int = 737,
) -> List[SampleInfo]:
    """
    Replicate the exact identity-stratified split from combined_paired.py.
    Mutates each sample's .split field in-place and returns the input list.
    """
    rng = random.Random(seed)

    by_identity: Dict[str, List[SampleInfo]] = defaultdict(list)
    for s in samples:
        by_identity[s.identity].append(s)

    identities = list(by_identity.keys())
    rng.shuffle(identities)

    n = len(identities)
    n_train = int(n * train_split)
    n_val = int(n * val_split)

    train_identities = set(identities[:n_train])
    val_identities = set(identities[n_train : n_train + n_val])
    # test_identities = everything else

    for s in samples:
        if s.identity in train_identities:
            s.split = "train"
        elif s.identity in val_identities:
            s.split = "val_in_dist"
        else:
            s.split = "test"

    return samples
