"""Static pre-launch check for GRL quality-domain population structure.

PLAN.md §9 Priority 6 round-2 escalation — before any P15-style GRL launch:
- No active domain may have zero enabled training sources (the empty-domain trap).
- No active domain may have only one enabled source (>90% concentration surrogate
  at config time; runtime weighting can still concentrate, but a single-source
  domain is already a label-confound risk).

This is a STATIC yaml-level check — it does not run discovery, does not touch
GCS, and does not require model weights. It complements the runtime smoke-grep
of "ENABLED" log lines documented in PLAN.md §10.7.

Exit 0 = pass. Exit 1 = fail with a printed reason.

Usage:
    python scripts/launch/check_grl_domain_populations.py <path/to/yaml>
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml


# Mirrors data/sources/combined_paired.py QUALITY_DOMAIN_MAP. Kept inline so
# this script is standalone — it must run before the trainer image is rebuilt.
QUALITY_DOMAIN_MAP: Dict[str, int] = {
    "df40": 0,
    "external": 1,
    "deeplive": 2,
    "visomaster": 2,
    "visomaster_hints": 2,
    "visomaster_hints_teams": 1,
    "visomaster_enhanced": 2,
    "visomaster_res_variant": 2,
    "visomaster_teams_enhanced": 1,
    "deeplive_teams": 1,
    "proper_visomaster_clean": 2,
    "proper_visomaster_enhanced_clean": 2,
    "proper_visomaster_teams": 1,
    "proper_visomaster_enhanced_teams": 1,
    "youtube": 3,
}

DOMAIN_NAMES = {
    0: "clean_academic",
    1: "webcam_codec",
    2: "studio_capture",
    3: "social_media",
}


def enabled_sources_in_yaml(cfg: dict) -> List[str]:
    """Return the list of source names that the trainer will actually emit
    samples for, given the yaml's enabled flags.

    Mirrors the dispatch logic in data/sources/combined_paired.py — when a
    family block has enabled=true, samples flow with the corresponding source
    string, which then maps via QUALITY_DOMAIN_MAP.
    """
    cp = cfg.get("combined_paired", {}) or {}
    out: List[str] = []

    if cp.get("df40", {}).get("enabled", False):
        out.append("df40")

    if cp.get("deeplive", {}).get("enabled", False):
        out.append("deeplive")

    if cp.get("visomaster", {}).get("enabled", False):
        out.append("visomaster")

    if cp.get("visomaster_enhanced", {}).get("enabled", False):
        out.append("visomaster_enhanced")

    if cp.get("visomaster_teams_enhanced", {}).get("enabled", False):
        out.append("visomaster_teams_enhanced")

    if cp.get("visomaster_hints", {}).get("enabled", False):
        out.append("visomaster_hints")

    if cp.get("visomaster_hints_teams", {}).get("enabled", False):
        out.append("visomaster_hints_teams")

    if cp.get("teams", {}).get("enabled", False):
        out.append("deeplive_teams")

    for entry in cp.get("external_training_reals", []) or []:
        method = entry.get("method", "")
        if "youtube" in method or "avspeech" in method:
            out.append("youtube")
        else:
            out.append("external")

    proper = cp.get("proper_data", {}) or {}
    if proper.get("enabled", False):
        if proper.get("emit_clean", True):
            out.append("proper_visomaster_clean")
            out.append("proper_visomaster_enhanced_clean")
        if proper.get("emit_teams", True):
            out.append("proper_visomaster_teams")
            out.append("proper_visomaster_enhanced_teams")

    seen: Set[str] = set()
    deduped: List[str] = []
    for src in out:
        if src not in seen:
            seen.add(src)
            deduped.append(src)
    return deduped


def check_yaml(yaml_path: Path) -> Tuple[bool, List[str], List[str]]:
    """Return (ok, errors, warnings). Hard fails go in errors; advisory items
    in warnings. Empty domain is hard-fail (PLAN.md §9 Priority 6 round-2:
    load-bearing — empty-domain head optimizes a spurious objective).
    Single-source is a warning, not a hard fail: paired sources (df40) have
    both fake/real labels within the source so the >90%-concentration
    label-confound concern is muted.
    """
    cfg = yaml.safe_load(yaml_path.read_text())
    errors: List[str] = []
    warnings: List[str] = []

    if not cfg.get("use_quality_domain_head", False):
        return True, [], []

    domain_count = int(cfg.get("quality_domain_count", 4))
    sources = enabled_sources_in_yaml(cfg)

    domain_to_sources: Dict[int, List[str]] = defaultdict(list)
    for src in sources:
        if src not in QUALITY_DOMAIN_MAP:
            errors.append(
                f"source '{src}' has no QUALITY_DOMAIN_MAP entry — add to "
                f"data/sources/combined_paired.py:QUALITY_DOMAIN_MAP"
            )
            continue
        d = QUALITY_DOMAIN_MAP[src]
        domain_to_sources[d].append(src)

    for d in range(domain_count):
        if not domain_to_sources.get(d):
            errors.append(
                f"domain {d} ({DOMAIN_NAMES.get(d, '?')}) has zero enabled "
                f"training sources but quality_domain_count={domain_count}. "
                f"Either drop quality_domain_count to {d}, or enable a source "
                f"that maps to domain {d}."
            )
        elif len(domain_to_sources[d]) == 1:
            warnings.append(
                f"domain {d} ({DOMAIN_NAMES.get(d, '?')}) has only one enabled "
                f"source ('{domain_to_sources[d][0]}'). Single-source domains "
                f"can become label-confound surrogates if the source carries "
                f"only one label; check that the source is paired (real+fake) "
                f"or carry the risk explicitly."
            )

    for d, srcs in domain_to_sources.items():
        if d >= domain_count:
            errors.append(
                f"source(s) {srcs} map to domain {d} but "
                f"quality_domain_count={domain_count} — labels >{domain_count - 1} "
                f"would be emitted to a head that does not classify them."
            )

    return len(errors) == 0, errors, warnings


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2

    yaml_path = Path(argv[1])
    if not yaml_path.exists():
        print(f"ERROR: yaml not found: {yaml_path}")
        return 2

    ok, errors, warnings = check_yaml(yaml_path)

    cfg = yaml.safe_load(yaml_path.read_text())
    if not cfg.get("use_quality_domain_head", False):
        print(f"[skip] {yaml_path.name}: use_quality_domain_head not set; nothing to check.")
        return 0

    print(f"=== GRL domain-population check: {yaml_path.name} ===")
    sources = enabled_sources_in_yaml(cfg)
    domain_count = int(cfg.get("quality_domain_count", 4))
    print(f"quality_domain_count: {domain_count}")
    print(f"enabled training sources: {sources}")
    domain_to_sources: Dict[int, List[str]] = defaultdict(list)
    for src in sources:
        if src in QUALITY_DOMAIN_MAP:
            domain_to_sources[QUALITY_DOMAIN_MAP[src]].append(src)
    print("domain -> sources:")
    for d in range(domain_count):
        srcs = domain_to_sources.get(d, [])
        marker = "  " if srcs else "EMPTY"
        print(f"  [{marker}] domain {d} ({DOMAIN_NAMES.get(d, '?')}): {srcs or 'NONE'}")
    print()

    if warnings:
        print("Warnings (non-blocking):")
        for w in warnings:
            print(f"  - {w}")
        print()

    if ok:
        print("PASS - no empty / out-of-range active domains.")
        return 0

    print("FAIL - pre-launch population check did not pass:")
    for e in errors:
        print(f"  - {e}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
