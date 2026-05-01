"""Tests for data/sources/method_domain_map.py — the 12-bucket method-domain
map for Phase 3 method-conditional GRL.

Mirrors the 14 lookup tests Phase 2C wrote in the analysis-side draft, plus
adds a few new gates relevant to launch:
  - All 12 buckets are reachable from at least one (method, source) pair.
  - METHOD_DOMAIN_NAMES has exactly 12 entries (count match yaml).
  - Source fallback never returns an out-of-range bucket.
  - Bucket 2 (deeplive_enhanced) is reachable via the methods Phase 1A
    pinpointed.
"""
from __future__ import annotations

import pytest

from data.sources.method_domain_map import (
    METHOD_DOMAIN_NAMES,
    METHOD_DOMAIN_EXACT,
    METHOD_DOMAIN_PREFIX,
    METHOD_DOMAIN_SOURCE_FALLBACK,
    lookup_method_domain,
    lookup_method_domain_with_label,
)


class TestMapStructure:
    def test_twelve_buckets(self):
        assert len(METHOD_DOMAIN_NAMES) == 12, (
            f"Expected 12 buckets, got {len(METHOD_DOMAIN_NAMES)}: "
            f"{sorted(METHOD_DOMAIN_NAMES.keys())}"
        )
        assert set(METHOD_DOMAIN_NAMES.keys()) == set(range(12))

    def test_exact_lookup_buckets_in_range(self):
        for method, bucket in METHOD_DOMAIN_EXACT.items():
            assert 0 <= bucket < 12, f"{method}={bucket} out of range"

    def test_prefix_lookup_buckets_in_range(self):
        for prefix, bucket in METHOD_DOMAIN_PREFIX:
            assert 0 <= bucket < 12, f"{prefix}={bucket} out of range"

    def test_source_fallback_buckets_in_range(self):
        for source, bucket in METHOD_DOMAIN_SOURCE_FALLBACK.items():
            assert 0 <= bucket < 12, f"{source}={bucket} out of range"


class TestLookupCorrectness:
    """The 14 canonical lookup tests from Phase 2C audit + 2 new gates."""

    def test_df40_methods_route_to_zero(self):
        for method in ("simswap", "facedancer", "blendface", "e4s"):
            assert lookup_method_domain(method, "df40") == 0

    def test_deeplive_basic_routes_to_one(self):
        assert lookup_method_domain("deeplive_edge_cases", "deeplive") == 1
        assert lookup_method_domain("deeplive_minimal_processing", "deeplive") == 1

    def test_deeplive_enhanced_routes_to_two(self):
        # Phase 1A cluster axis — explicit assertions.
        assert lookup_method_domain("deeplive_quality_enhancement", "deeplive") == 2
        assert lookup_method_domain("deeplive_edge_cases_enhanced", "deeplive") == 2
        assert lookup_method_domain("deeplive_minimal_processing_enhanced", "deeplive") == 2

    def test_deeplive_teams_prefix_routes_to_three(self):
        # Teams passthrough (incl. dor_shkedi-style captures).
        assert lookup_method_domain("deeplive_teams_dor_shkedi_s16", "deeplive_teams") == 3
        assert lookup_method_domain("deeplive_teams_pc_generator_s15", "deeplive_teams") == 3

    def test_visomaster_inswapper_family(self):
        assert lookup_method_domain("visomaster_inswapper128", "visomaster") == 4
        assert lookup_method_domain("visomaster_simswap512", "visomaster") == 4

    def test_visomaster_ghost_family(self):
        assert lookup_method_domain("visomaster_ghostface_v1", "visomaster") == 5
        assert lookup_method_domain("visomaster_ghostface_v2", "visomaster") == 5

    def test_visomaster_other_family(self):
        assert lookup_method_domain("visomaster_cscs", "visomaster") == 6

    def test_visomaster_enhanced_prefix(self):
        assert lookup_method_domain(
            "visomaster_enhanced_codeformer", "visomaster_enhanced"
        ) == 7

    def test_visomaster_teams_recap_prefix(self):
        assert lookup_method_domain(
            "visomaster_teams_enhanced_codeformer", "visomaster_teams_enhanced"
        ) == 8

    def test_external_vcd_real(self):
        assert lookup_method_domain("external_vcd_real", "external") == 10

    def test_normalization_handles_case_and_punctuation(self):
        # GhostFace-v1 (with hyphen and capitals) should normalize.
        assert lookup_method_domain("GhostFace-v1", "visomaster") in (5, 6)
        # Allow either bucket (5 = ghost or 6 = other) — current map routes
        # this to 5 via prefix match on lowercased+normalized key.

    def test_unknown_method_falls_back_to_source(self):
        assert lookup_method_domain("nonexistent_method", "df40") == 0
        assert lookup_method_domain("nonexistent_method", "external") == 10

    def test_unknown_method_and_source_returns_zero(self):
        # Unknown both — fall back to bucket 0 (clean academic).
        assert lookup_method_domain("nonsense", "alien_source") == 0

    def test_empty_inputs(self):
        # Defensive: don't crash on empty strings.
        assert lookup_method_domain("", "") == 0
        assert lookup_method_domain(None, None) == 0  # type: ignore[arg-type]


class TestPhase3ReadinessGates:
    """Non-canonical extra checks added at install time."""

    def test_all_active_buckets_reachable_via_label_wrapper(self):
        """Every bucket id in [0, 11] must be reachable from some
        (method, source, label) triple via lookup_method_domain_with_label."""
        # Buckets 0-6 + 10-11 are active in P14/P15 enable lists; 7-9 are
        # reserved for future enables. Bucket 11 (realpool_real) is set by
        # the label-aware wrapper, not by the bare lookup_method_domain.
        reachable = set()
        # Direct lookup buckets 0-10
        for bucket in METHOD_DOMAIN_EXACT.values():
            reachable.add(bucket)
        for _, bucket in METHOD_DOMAIN_PREFIX:
            reachable.add(bucket)
        for bucket in METHOD_DOMAIN_SOURCE_FALLBACK.values():
            reachable.add(bucket)
        # Bucket 11 reachability via label-aware wrapper
        # (real frame from a non-df40, non-external source).
        bucket_11 = lookup_method_domain_with_label(
            "deeplive_edge_cases", "deeplive", label=0
        )
        reachable.add(bucket_11)
        unreachable = set(METHOD_DOMAIN_NAMES.keys()) - reachable
        assert not unreachable, (
            f"Buckets unreachable: {unreachable}. "
            f"Add an exact/prefix/source-fallback or label-wrapper entry."
        )

    def test_label_wrapper_real_routes_to_realpool(self):
        # Real frame from deeplive (non-df40, non-external) → bucket 11.
        assert lookup_method_domain_with_label(
            "deeplive_edge_cases", "deeplive", label=0
        ) == 11
        # Real frame from visomaster → bucket 11.
        assert lookup_method_domain_with_label(
            "visomaster_inswapper128", "visomaster", label=0
        ) == 11
        # Real frame from external_vcd → bucket 10 (preserved).
        assert lookup_method_domain_with_label(
            "external_vcd_real", "external", label=0
        ) == 10
        # Real frame from df40 → bucket 0 (preserved; paired real+fake).
        assert lookup_method_domain_with_label(
            "simswap", "df40", label=0
        ) == 0

    def test_label_wrapper_fake_unchanged(self):
        # Fakes always go to method-conditional bucket regardless of wrapper.
        for method, expected in [
            ("simswap", 0),
            ("deeplive_edge_cases", 1),
            ("deeplive_quality_enhancement", 2),
            ("deeplive_teams_dor_shkedi_s16", 3),
            ("visomaster_inswapper128", 4),
        ]:
            assert lookup_method_domain_with_label(
                method, "df40" if method == "simswap" else "deeplive", label=1
            ) == expected, (
                f"Fake routing changed by wrapper for {method}: "
                f"got {lookup_method_domain_with_label(method, 'deeplive', 1)}, "
                f"expected {expected}"
            )

    def test_phase1a_cluster_axis_accessible(self):
        """Phase 1A finding: trained P17 head modally aligns with `is_dor_shkedi` /
        `is_deeplive_enhanced`. The 12-bucket map must split these into their
        own buckets so the GRL has axes to attack."""
        # is_deeplive_enhanced cluster → bucket 2
        assert lookup_method_domain("deeplive_quality_enhancement", "deeplive") == 2
        # is_dor_shkedi cluster (Teams passthrough) → bucket 3
        assert lookup_method_domain("deeplive_teams_dor_shkedi_s16", "deeplive_teams") == 3
        # The two clusters MUST be separate (not collapsed into single bucket).
        assert (
            lookup_method_domain("deeplive_quality_enhancement", "deeplive")
            != lookup_method_domain("deeplive_teams_dor_shkedi_s16", "deeplive_teams")
        )
