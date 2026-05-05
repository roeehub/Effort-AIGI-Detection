"""Tests for utils/grouping.py family-key inference, specifically guarding
against the `quality_enhancement` misrouting bug fixed 2026-05-05.

Background: commit 38558ee5 (2026-03-19) incorrectly grouped the
`quality_enhancement` deeplive strategy with GFPGAN-applied strategies. Bucket
manifest evidence (no `enhancement: GFPGAN_sample` field) and visual inspection
both confirmed the strategy is a BASE strategy, not enhanced. The fix removed
`quality_enhancement` from `DEFAULT_ENHANCED_STRATEGIES` and from the
`infer_family_key` mapping that routed `deeplive_quality_enhancement_fake`
group_key into the `deeplive_enhanced_fake` family.

These tests guard against re-introduction.

See: docs/packet_retrospectives/threads/quality_enhancement_strategy_misrouting.md
"""
from __future__ import annotations

from utils.grouping import (
    DEFAULT_ENHANCED_STRATEGIES,
    infer_family_key,
    infer_group_key,
)


class TestQualityEnhancementRouting:
    def test_quality_enhancement_not_in_default_enhanced(self):
        """`quality_enhancement` must NOT appear in the default enhanced set."""
        assert "quality_enhancement" not in DEFAULT_ENHANCED_STRATEGIES, (
            "quality_enhancement was re-added to DEFAULT_ENHANCED_STRATEGIES. "
            "Per bucket manifest evidence (no GFPGAN tag) it is a base strategy. "
            "See quality_enhancement_strategy_misrouting.md before re-adding."
        )

    def test_quality_enhancement_routes_to_non_enhanced_family(self):
        """`quality_enhancement` deeplive fakes must route to deeplive_non_enhanced_fake.

        This is the load-bearing test: even if a yaml passes
        `quality_enhancement` in `enhanced_strategy_names`, the family
        routing must NOT collapse it into the GFPGAN-applied family.
        """
        family = infer_family_key(
            label=1,
            method="quality_enhancement_0042",
            source="deeplive",
        )
        assert family == "deeplive_non_enhanced_fake", (
            f"quality_enhancement fake routed to {family!r} but should route "
            "to 'deeplive_non_enhanced_fake' (it's a base strategy without GFPGAN)."
        )

    def test_quality_enhancement_routing_invariant_under_yaml_override(self):
        """Even when a yaml passes `quality_enhancement` in enhanced_strategy_names
        (as historical R13 yamls do), the family routing must still be correct.
        """
        family = infer_family_key(
            label=1,
            method="quality_enhancement_0042",
            source="deeplive",
            enhanced_strategy_names=(
                "quality_enhancement",  # historical override — ignored by routing
                "edge_cases_enhanced",
                "minimal_processing_enhanced",
            ),
        )
        assert family == "deeplive_non_enhanced_fake"

    def test_edge_cases_enhanced_still_routes_to_enhanced(self):
        """Sanity check: actual GFPGAN-applied strategies must still route correctly."""
        family = infer_family_key(
            label=1,
            method="edge_cases_enhanced_0001",
            source="deeplive",
        )
        assert family == "deeplive_enhanced_fake"

    def test_minimal_processing_enhanced_still_routes_to_enhanced(self):
        """Sanity check: actual GFPGAN-applied strategies must still route correctly."""
        family = infer_family_key(
            label=1,
            method="minimal_processing_enhanced_0000",
            source="deeplive",
        )
        assert family == "deeplive_enhanced_fake"

    def test_edge_cases_base_routes_to_non_enhanced(self):
        """Sanity check: base strategies (no GFPGAN) must route to non_enhanced."""
        family = infer_family_key(
            label=1,
            method="edge_cases_0001",
            source="deeplive",
        )
        assert family == "deeplive_non_enhanced_fake"

    def test_minimal_processing_base_routes_to_non_enhanced(self):
        """Sanity check: base strategies (no GFPGAN) must route to non_enhanced."""
        family = infer_family_key(
            label=1,
            method="minimal_processing_0000",
            source="deeplive",
        )
        assert family == "deeplive_non_enhanced_fake"

    def test_quality_enhancement_real_routing_unchanged(self):
        """Real-side routing for the canonical `quality_enhancement` strategy
        is its own group (`deeplive_quality_enhancement_real`); this was NOT
        affected by the bug and should remain stable as a regression guard.
        """
        group = infer_group_key(
            label=0,
            method="quality_enhancement",
            source="deeplive",
        )
        assert group == "deeplive_quality_enhancement_real"
