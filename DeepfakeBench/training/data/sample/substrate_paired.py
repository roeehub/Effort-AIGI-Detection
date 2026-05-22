"""Substrate-paired metadata stamper for BACKBONE-SlotAv2 / BACKBONE-T5C.

This module stamps each per-frame yield-row (or per-video collate-time row)
with two new fields used by:

  - BACKBONE-SlotAv2 (GroupDRO substrate-balanced) — reads `group_id`
    derived from (substrate_transport × source) to balance the loss
    contribution across clean/teams substrates.
  - BACKBONE-T5C (asymmetric pair-loss) — reads `substrate_pair_id`
    + `substrate_transport` to peel matched clean/teams pairs from the
    batch and apply a one-sided hinge.

The two new fields per stamped frame are:

  - `substrate_pair_id: int`   — identity-level unique integer; shared
                                  across the same identity's clean and
                                  teams sides. -1 when identity is
                                  outside the inventory.
  - `substrate_transport: int` — 0 = clean, 1 = teams, -1 = unknown.

Companion plumbing (per `data/sources/visomaster.py:748,1085,1105,1176,
1539-1554`):

  - The `visomaster_teams_enhanced` lane already yields paired clean+teams
    real-side frames for the 54 `teams_v2_companion` resolver rows. Those
    frames carry `companion_domain='teams_v2'` and `source` of either
    `'deeplive_teams'` (real-side, post-teams) or `'visomaster_enhanced'`
    (fake-side, post-enhance) per `combined_paired.py:3154-3162`. We treat
    the real-side row as `substrate_transport=1` and pair it with the
    same identity's clean-bucket real frames (yielded by other lanes).
  - The `hdtf_visomaster_teams` and `quickclips_visomaster_teams` sources
    enumerated in `analysis/substrate_pair_geometry_2026-05-22/
    inventory_manifest.csv` (1,826 of the 1,880 rows) are NOT currently
    wired as separate training data lanes — the inventory's frames sit in
    GCS buckets that no existing iterator reads. They are recorded here
    so the stamper can deterministically assign pair_ids if/when a follow-
    up lane is added. Today they contribute 0 frames to the batch.

Usage:
  ```python
  from data.sample.substrate_paired import (
      SubstratePairStamper, get_active_stamper,
  )

  # At trainer init:
  stamper = SubstratePairStamper.from_config(
      cfg={'enabled': True, 'inventory_path': '.../inventory_manifest.csv'},
  )

  # In collate_fn for each per-video row:
  pair_id, transport = stamper.lookup(identity=<str>, source=<str>)
  ```

The stamper is intentionally NOT registered as a "data source". It is a
collate-time annotator; the data still flows through the existing paired
iterators. This avoids deep modifications to `CombinedPairedIterableDataset`.

Config (yaml block under `combined_paired.substrate_pair_sampling`):

  ```yaml
  combined_paired:
    substrate_pair_sampling:
      enabled: true
      inventory_path: "analysis/substrate_pair_geometry_2026-05-22/inventory_manifest.csv"
      sources:
        - hdtf_visomaster_teams
        - quickclips_visomaster_teams
        - visomaster_teams_enhanced
        - deeplive_teams
      pair_fraction: 0.25  # informational; honored by sampler only when a
                            # dedicated lane is wired. Default lane today
                            # yields ~0% paired-real fraction.
  ```
"""
from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

# Default repo-relative inventory path. Resolved against the training cwd.
DEFAULT_INVENTORY_PATH = (
    "analysis/substrate_pair_geometry_2026-05-22/inventory_manifest.csv"
)

# Substrate transport code constants.
TRANSPORT_UNKNOWN = -1
TRANSPORT_CLEAN = 0
TRANSPORT_TEAMS = 1


def _normalize_identity(identity: str) -> str:
    """Strip the loader's `realpool_` / `visomaster_` prefix when present.

    The inventory CSV stores raw identities like 'RD_Radio14'. The
    combined_paired loaders prepend 'realpool_' or 'visomaster_' to avoid
    cross-source collisions. This normalizer reverses both prefixes so
    the stamper can match by raw identity.
    """
    if not isinstance(identity, str) or not identity:
        return ""
    if identity.startswith("realpool_"):
        return identity[len("realpool_"):]
    if identity.startswith("visomaster_"):
        return identity[len("visomaster_"):]
    return identity


def _is_teams_source(source: str) -> bool:
    """Per-row source label → 'is this the teams-side of a substrate pair?'.

    Mirrors the heuristic in `data/sources/combined_paired.py::_transport_from_source`
    but returns a binary clean-vs-teams call (vs the multi-class transport bucket
    the GroupDRO key uses). The `visomaster_teams_enhanced` source yields BOTH
    sides (real-side row is post-teams; clean-side is yielded by other iterators).
    For yield rows whose collate row has source='visomaster_teams_enhanced', the
    `companion_domain='teams_v2'` carries the teams-side. We treat the row as
    teams if either the source matches a *_teams label OR if `companion_domain`
    indicates the post-teams branch.
    """
    s = (source or "").lower()
    if not s:
        return False
    if s.endswith("_teams") or "teams" in s:
        return True
    return False


@dataclass
class SubstratePairStamper:
    """Identity → (substrate_pair_id, substrate_transport) lookup.

    Attributes
    ----------
    enabled:
        When False, `lookup` returns (-1, -1) for every input. Safe no-op.
    sources_allowlist:
        Optional set of source labels (e.g., 'hdtf_visomaster_teams') to
        restrict pair stamping. None means "stamp whenever the identity
        is in the inventory".
    pair_fraction:
        Informational target fraction of batch that should be substrate-
        paired reals. Today the default training lane only emits a few %
        substrate-paired frames; recorded for downstream observability.
    inventory_path:
        Absolute or repo-relative path to the inventory CSV.
    """

    enabled: bool = False
    inventory_path: str = DEFAULT_INVENTORY_PATH
    sources_allowlist: Optional[frozenset] = None
    pair_fraction: float = 0.25

    # Built at __post_init__:
    _identity_to_pair_id: Dict[str, int] = field(default_factory=dict, repr=False)
    _pair_id_to_identity: Dict[int, str] = field(default_factory=dict, repr=False)
    _pair_id_to_source: Dict[int, str] = field(default_factory=dict, repr=False)
    _n_pairs: int = 0

    def __post_init__(self) -> None:
        if not self.enabled:
            logger.info("SubstratePairStamper DISABLED (enabled=False)")
            return
        path = self.inventory_path
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if not os.path.exists(path):
            logger.warning(
                "SubstratePairStamper inventory not found at %s — disabling.", path,
            )
            self.enabled = False
            return
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        if not rows:
            logger.warning(
                "SubstratePairStamper inventory %s has no rows — disabling.", path,
            )
            self.enabled = False
            return
        # Allocate one pair_id per (identity_id, base_capture_id) row.
        # We use the row index (1-based) as the pair_id for traceability.
        for idx, row in enumerate(rows, start=1):
            identity = str(row.get("identity_id", "")).strip()
            source = str(row.get("source", "")).strip()
            if not identity:
                continue
            # When the allowlist is set, only register identities from
            # those source labels.
            if self.sources_allowlist and source not in self.sources_allowlist:
                continue
            # Multiple base captures can share an identity (e.g., RD_Radio14 → 2
            # captures). We use the latest seen pair_id as the canonical id
            # for that identity. The matched-pair loss only requires that
            # clean+teams from the SAME identity-frame share a pair_id; we
            # stamp at video-grain (sample_id) downstream, so duplicate
            # identities are fine.
            self._identity_to_pair_id[identity] = idx
            self._pair_id_to_identity[idx] = identity
            self._pair_id_to_source[idx] = source
            self._n_pairs += 1
        logger.info(
            "SubstratePairStamper ENABLED: %d inventory rows, %d unique identities, "
            "pair_fraction=%.2f, allowlist=%s, path=%s",
            len(rows),
            len(self._identity_to_pair_id),
            self.pair_fraction,
            (sorted(self.sources_allowlist) if self.sources_allowlist else "any"),
            path,
        )

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> "SubstratePairStamper":
        cfg = cfg or {}
        sources = cfg.get("sources")
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            inventory_path=str(cfg.get("inventory_path", DEFAULT_INVENTORY_PATH)),
            sources_allowlist=frozenset(sources) if sources else None,
            pair_fraction=float(cfg.get("pair_fraction", 0.25)),
        )

    def lookup(
        self,
        identity: str,
        source: str,
        companion_domain: Optional[str] = None,
        label: int = 0,
    ) -> Tuple[int, int]:
        """Return (substrate_pair_id, substrate_transport) for a frame.

        Returns (-1, -1) when the identity is not in the inventory OR when
        the stamper is disabled OR when the row carries label != 0 (the
        pair-loss only operates on real-real pairs).
        """
        if not self.enabled:
            return (TRANSPORT_UNKNOWN, TRANSPORT_UNKNOWN)
        if int(label) != 0:
            # Pair-loss is real-vs-real only; fake-side rows are out of scope.
            return (TRANSPORT_UNKNOWN, TRANSPORT_UNKNOWN)
        norm = _normalize_identity(identity)
        pair_id = self._identity_to_pair_id.get(norm)
        if pair_id is None:
            # Try the raw identity in case the loader is already emitting unprefixed.
            pair_id = self._identity_to_pair_id.get(identity)
        if pair_id is None:
            return (TRANSPORT_UNKNOWN, TRANSPORT_UNKNOWN)
        # Companion domain dominates the call when present (teams_v2 → teams).
        cd = (companion_domain or "").lower()
        if cd == "teams_v2":
            transport = TRANSPORT_TEAMS
        elif _is_teams_source(source):
            transport = TRANSPORT_TEAMS
        else:
            transport = TRANSPORT_CLEAN
        return (pair_id, transport)

    def n_pairs(self) -> int:
        return self._n_pairs

    def stats(self) -> Dict[str, int]:
        from collections import Counter
        per_source = Counter(self._pair_id_to_source.values())
        return {
            "enabled": int(self.enabled),
            "n_inventory_rows": self._n_pairs,
            "n_unique_identities": len(self._identity_to_pair_id),
            **{f"per_source/{k}": v for k, v in per_source.items()},
        }


# Module-level singleton — set by trainer init, read by collate.
_ACTIVE_STAMPER: Optional[SubstratePairStamper] = None


def set_active_stamper(stamper: Optional[SubstratePairStamper]) -> None:
    """Install a process-global stamper. Called once at trainer init."""
    global _ACTIVE_STAMPER
    _ACTIVE_STAMPER = stamper
    if stamper is not None and stamper.enabled:
        logger.info(
            "SubstratePairStamper registered as process-global stamper "
            "(n_pairs=%d).", stamper.n_pairs(),
        )


def get_active_stamper() -> Optional[SubstratePairStamper]:
    """Return the process-global stamper (or None)."""
    return _ACTIVE_STAMPER


def stamp_row(
    row: dict,
    *,
    stamper: Optional[SubstratePairStamper] = None,
) -> dict:
    """Convenience: mutate `row` in place with substrate_pair_id + transport.

    Reads `row['identity']`, `row['source']`, optional `row['companion_domain']`
    and `row['label']`. Writes:
      - row['substrate_pair_id']: int (or -1)
      - row['substrate_transport']: int (0/1/-1)
    """
    stamper = stamper if stamper is not None else _ACTIVE_STAMPER
    if stamper is None or not stamper.enabled:
        row.setdefault("substrate_pair_id", TRANSPORT_UNKNOWN)
        row.setdefault("substrate_transport", TRANSPORT_UNKNOWN)
        return row
    pair_id, transport = stamper.lookup(
        identity=row.get("identity", "") or "",
        source=row.get("source", "") or "",
        companion_domain=row.get("companion_domain"),
        label=int(row.get("label", 0) or 0),
    )
    row["substrate_pair_id"] = pair_id
    row["substrate_transport"] = transport
    return row


__all__ = [
    "SubstratePairStamper",
    "TRANSPORT_UNKNOWN",
    "TRANSPORT_CLEAN",
    "TRANSPORT_TEAMS",
    "DEFAULT_INVENTORY_PATH",
    "get_active_stamper",
    "set_active_stamper",
    "stamp_row",
]
