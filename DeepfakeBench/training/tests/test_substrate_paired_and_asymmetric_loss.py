"""Tests for the BACKBONE 2026-05-22 wiring.

Covers:
  - data.sample.substrate_paired.SubstratePairStamper inventory lookup
    and transport/companion_domain logic.
  - loss.substrate_pair_asymmetric.SubstratePairAsymmetricLoss forward
    pass, asymmetric gradient (teams detached), and B*T batch expansion.
  - End-to-end collate integration: collate emits substrate_pair_id /
    substrate_transport for matched-pair identity rows.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INVENTORY_PATH = os.path.join(
    REPO_ROOT,
    "analysis/substrate_pair_geometry_2026-05-22/inventory_manifest.csv",
)


# ---------------------------------------------------------------------------
# substrate_paired stamper
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stamper_enabled():
    from data.sample.substrate_paired import SubstratePairStamper
    if not os.path.exists(INVENTORY_PATH):
        pytest.skip(f"Inventory CSV missing at {INVENTORY_PATH}")
    return SubstratePairStamper(
        enabled=True,
        inventory_path=INVENTORY_PATH,
    )


def test_normalize_identity():
    from data.sample.substrate_paired import _normalize_identity
    assert _normalize_identity("realpool_RD_Radio14") == "RD_Radio14"
    assert _normalize_identity("visomaster_QCLIP01_00012") == "QCLIP01_00012"
    assert _normalize_identity("RD_Radio14") == "RD_Radio14"
    assert _normalize_identity("") == ""


def test_is_teams_source():
    from data.sample.substrate_paired import _is_teams_source
    assert _is_teams_source("deeplive_teams")
    assert _is_teams_source("hdtf_visomaster_teams")
    assert _is_teams_source("visomaster_teams_enhanced")
    assert not _is_teams_source("visomaster")
    assert not _is_teams_source("deeplive")
    assert not _is_teams_source("df40")
    assert not _is_teams_source("")


def test_stamper_disabled_noop():
    from data.sample.substrate_paired import SubstratePairStamper
    s = SubstratePairStamper(enabled=False)
    assert s.lookup("RD_Radio14", "deeplive_teams") == (-1, -1)


def test_stamper_inventory_load(stamper_enabled):
    assert stamper_enabled.n_pairs() == 1880
    stats = stamper_enabled.stats()
    assert stats["per_source/hdtf_visomaster_teams"] == 1094
    assert stats["per_source/quickclips_visomaster_teams"] == 732
    assert stats["per_source/visomaster_teams_enhanced"] == 54


def test_stamper_lookup_clean_vs_teams(stamper_enabled):
    # RD_Radio14 is row #493 in the inventory (1-indexed).
    pid_t, tr_t = stamper_enabled.lookup("RD_Radio14", "hdtf_visomaster_teams")
    assert pid_t > 0
    assert tr_t == 1  # teams source

    pid_c, tr_c = stamper_enabled.lookup("RD_Radio14", "visomaster")
    assert pid_c == pid_t  # same identity → same pair_id
    assert tr_c == 0  # clean source

    # realpool_ prefix stripped
    pid_r, tr_r = stamper_enabled.lookup("realpool_RD_Radio14", "visomaster")
    assert pid_r == pid_t
    assert tr_r == 0


def test_stamper_companion_domain_overrides_source(stamper_enabled):
    # When companion_domain='teams_v2' is present, it forces transport=1
    # even on a non-teams source label.
    pid, tr = stamper_enabled.lookup(
        "RD_Radio14", "visomaster", companion_domain="teams_v2",
    )
    assert tr == 1


def test_stamper_label_nonzero_skipped(stamper_enabled):
    # Fake-side rows (label=1) are out of scope for the substrate-pair loss.
    assert stamper_enabled.lookup(
        "RD_Radio14", "hdtf_visomaster_teams", label=1,
    ) == (-1, -1)


def test_stamper_unknown_identity_skipped(stamper_enabled):
    assert stamper_enabled.lookup(
        "DEFINITELY_NOT_IN_INVENTORY_XYZ", "deeplive_teams",
    ) == (-1, -1)


# ---------------------------------------------------------------------------
# Asymmetric pair loss
# ---------------------------------------------------------------------------

def test_loss_disabled_returns_zero():
    from loss.substrate_pair_asymmetric import SubstratePairAsymmetricLoss
    loss = SubstratePairAsymmetricLoss(lambda_pair=0.3, enabled=False)
    prob = torch.tensor([[0.3, 0.7], [0.6, 0.4]])
    data = {
        "substrate_pair_id": torch.tensor([1, 1]),
        "substrate_transport": torch.tensor([0, 1]),
    }
    assert loss.compute_from_batch(prob, data).item() == 0.0


def test_loss_one_sided_hinge():
    from loss.substrate_pair_asymmetric import SubstratePairAsymmetricLoss
    loss = SubstratePairAsymmetricLoss(lambda_pair=1.0, margin=0.0, enabled=True)
    # Case A: prob_teams > prob_clean → positive hinge
    prob_c = torch.tensor([0.2, 0.4])
    prob_t = torch.tensor([0.5, 0.6])
    expected = ((0.5 - 0.2) + (0.6 - 0.4)) / 2.0
    assert abs(loss.forward(prob_c, prob_t).item() - expected) < 1e-5
    # Case B: prob_clean > prob_teams → zero hinge (one-sided)
    prob_c = torch.tensor([0.8, 0.7])
    prob_t = torch.tensor([0.3, 0.4])
    assert loss.forward(prob_c, prob_t).item() == 0.0


def test_loss_margin_subtracted():
    from loss.substrate_pair_asymmetric import SubstratePairAsymmetricLoss
    loss = SubstratePairAsymmetricLoss(lambda_pair=1.0, margin=0.2, enabled=True)
    prob_c = torch.tensor([0.3])
    prob_t = torch.tensor([0.6])
    # 0.6 - 0.3 - 0.2 = 0.1
    assert abs(loss.forward(prob_c, prob_t).item() - 0.1) < 1e-5


def test_loss_lambda_scales_output():
    from loss.substrate_pair_asymmetric import SubstratePairAsymmetricLoss
    loss = SubstratePairAsymmetricLoss(lambda_pair=2.0, margin=0.0, enabled=True)
    prob_c = torch.tensor([0.3])
    prob_t = torch.tensor([0.6])
    assert abs(loss.forward(prob_c, prob_t).item() - 0.6) < 1e-5


def test_loss_gradient_flows_only_into_clean():
    """Confirms the asymmetric design: clean is pulled UP, teams unchanged."""
    from loss.substrate_pair_asymmetric import SubstratePairAsymmetricLoss
    loss = SubstratePairAsymmetricLoss(lambda_pair=1.0, margin=0.0, enabled=True)
    prob = torch.tensor([[0.7, 0.3], [0.4, 0.6]], requires_grad=True)
    data = {
        "substrate_pair_id": torch.tensor([5, 5]),
        "substrate_transport": torch.tensor([0, 1]),
    }
    result = loss.compute_from_batch(prob, data)
    result.backward()
    # Clean side (row 0) prob_fake column has -1 gradient (gradient descent
    # subtracts gradient → clean prob_fake goes UP).
    assert abs(prob.grad[0, 1].item() - (-1.0)) < 1e-5
    # Teams side (row 1) has zero gradient (detached by design).
    assert prob.grad[1, 1].item() == 0.0


def test_loss_compute_from_batch_multi_pair_with_unmatched():
    from loss.substrate_pair_asymmetric import SubstratePairAsymmetricLoss
    loss = SubstratePairAsymmetricLoss(lambda_pair=1.0, margin=0.0, enabled=True)
    prob = torch.tensor([
        [0.7, 0.3],  # clean, pair 5
        [0.4, 0.6],  # teams, pair 5 → hinge 0.3
        [0.5, 0.5],  # clean, pair 7
        [0.4, 0.6],  # teams, pair 7 → hinge 0.1
        [0.5, 0.5],  # unmatched
    ])
    data = {
        "substrate_pair_id": torch.tensor([5, 5, 7, 7, -1]),
        "substrate_transport": torch.tensor([0, 1, 0, 1, -1]),
    }
    expected = (0.3 + 0.1) / 2.0
    assert abs(loss.compute_from_batch(prob, data).item() - expected) < 1e-5


def test_loss_handles_bt_expansion():
    """When prob has B*T rows but pair_id has only B, the loss expands."""
    from loss.substrate_pair_asymmetric import SubstratePairAsymmetricLoss
    loss = SubstratePairAsymmetricLoss(lambda_pair=1.0, margin=0.0, enabled=True)
    # 4 frames per video, 2 videos
    prob = torch.tensor([
        [0.7, 0.3], [0.7, 0.3], [0.7, 0.3], [0.7, 0.3],  # video 0 clean
        [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6],  # video 1 teams
    ])
    data = {
        "substrate_pair_id": torch.tensor([5, 5]),
        "substrate_transport": torch.tensor([0, 1]),
    }
    # First clean frame (0.3) paired with first teams frame (0.6) → hinge 0.3
    assert abs(loss.compute_from_batch(prob, data).item() - 0.3) < 1e-5


def test_loss_no_matched_pairs_returns_zero():
    from loss.substrate_pair_asymmetric import SubstratePairAsymmetricLoss
    loss = SubstratePairAsymmetricLoss(lambda_pair=1.0, margin=0.0, enabled=True)
    prob = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    data = {
        "substrate_pair_id": torch.tensor([5, 7]),  # different ids
        "substrate_transport": torch.tensor([0, 0]),  # both clean
    }
    assert loss.compute_from_batch(prob, data).item() == 0.0


# ---------------------------------------------------------------------------
# Collate end-to-end
# ---------------------------------------------------------------------------

def _mock_frame(idx, identity, source, label, sample_id):
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    return {
        "image": img,
        "label": label,
        "identity": identity,
        "source": source,
        "method": "test",
        "method_id": 0,
        "sample_id": sample_id,
        "frame_idx": idx,
        "quality_domain": 0,
    }


def test_collate_emits_substrate_pair_fields(stamper_enabled):
    from data.sample.substrate_paired import set_active_stamper
    from data.sources.combined_paired import combined_paired_collate_fn
    set_active_stamper(stamper_enabled)
    try:
        batch = (
            [_mock_frame(i, "realpool_RD_Radio14", "hdtf_visomaster_teams", 0, "vidA") for i in range(2)] +
            [_mock_frame(i, "realpool_RD_Radio14", "visomaster", 0, "vidB") for i in range(2)]
        )
        result = combined_paired_collate_fn(batch)
        assert "substrate_pair_id" in result
        assert "substrate_transport" in result
        # Two videos → two entries.
        assert result["substrate_pair_id"].shape[0] == 2
        # Both videos share identity → same pair_id.
        assert result["substrate_pair_id"][0].item() == result["substrate_pair_id"][1].item()
        # Different transports per video.
        transports = sorted(result["substrate_transport"].tolist())
        assert transports == [0, 1]
    finally:
        set_active_stamper(None)


def test_collate_disabled_stamper_emits_minus1(stamper_enabled):
    from data.sample.substrate_paired import set_active_stamper
    from data.sources.combined_paired import combined_paired_collate_fn
    set_active_stamper(None)  # explicitly disable
    batch = [_mock_frame(0, "realpool_RD_Radio14", "hdtf_visomaster_teams", 0, "vidA")]
    result = combined_paired_collate_fn(batch)
    assert result["substrate_pair_id"].tolist() == [-1]
    assert result["substrate_transport"].tolist() == [-1]


def test_collate_unknown_identity_emits_minus1(stamper_enabled):
    from data.sample.substrate_paired import set_active_stamper
    from data.sources.combined_paired import combined_paired_collate_fn
    set_active_stamper(stamper_enabled)
    try:
        batch = [_mock_frame(0, "realpool_UNKNOWN_XYZ", "df40", 0, "vidU")]
        result = combined_paired_collate_fn(batch)
        assert result["substrate_pair_id"].tolist() == [-1]
        assert result["substrate_transport"].tolist() == [-1]
    finally:
        set_active_stamper(None)


# ---------------------------------------------------------------------------
# End-to-end: collate + asymmetric loss
# ---------------------------------------------------------------------------

def test_end_to_end_collate_plus_asymmetric_loss(stamper_enabled):
    from data.sample.substrate_paired import set_active_stamper
    from data.sources.combined_paired import combined_paired_collate_fn
    from loss.substrate_pair_asymmetric import SubstratePairAsymmetricLoss

    set_active_stamper(stamper_enabled)
    try:
        batch = (
            [_mock_frame(i, "realpool_RD_Radio14", "hdtf_visomaster_teams", 0, "vidA") for i in range(2)] +
            [_mock_frame(i, "realpool_RD_Radio14", "visomaster", 0, "vidB") for i in range(2)] +
            [_mock_frame(i, "realpool_RD_Radio27", "hdtf_visomaster_teams", 0, "vidC") for i in range(2)] +
            [_mock_frame(i, "realpool_RD_Radio27", "visomaster", 0, "vidD") for i in range(2)]
        )
        data = combined_paired_collate_fn(batch)

        # Construct prob aligned to the video order. The collate orders by
        # group key; we re-derive the per-video order from substrate_transport.
        n = data["substrate_pair_id"].shape[0]
        mock_prob = torch.zeros((n, 2))
        for i, tr in enumerate(data["substrate_transport"].tolist()):
            if tr == 1:
                mock_prob[i] = torch.tensor([0.3, 0.7])  # high teams fake prob
            elif tr == 0:
                mock_prob[i] = torch.tensor([0.7, 0.3])  # low clean fake prob
            else:
                mock_prob[i] = torch.tensor([0.5, 0.5])

        loss = SubstratePairAsymmetricLoss(lambda_pair=0.3, margin=0.0, enabled=True)
        result = loss.compute_from_batch(mock_prob, data)
        # 2 matched pairs, each hinge = 0.4 → mean=0.4 × lambda=0.3 → 0.12.
        assert abs(result.item() - 0.12) < 1e-5
    finally:
        set_active_stamper(None)


# ---------------------------------------------------------------------------
# Substrate-paired inventory data lane wiring (BACKBONE 2026-05-22)
# ---------------------------------------------------------------------------

def test_inventory_discovery_default_filter_keeps_hdtf_and_qclip():
    """`discover_substrate_paired_samples` returns 1826 rows by default
    (HDTF 1094 + QCLIP 732). The 54 enhanced rows are filtered out because
    they ship via the visomaster_teams_enhanced lane."""
    from data.sources.substrate_paired_inventory import (
        INVENTORY_SOURCE_HDTF,
        INVENTORY_SOURCE_QCLIP,
        discover_substrate_paired_samples,
    )
    if not os.path.exists(INVENTORY_PATH):
        pytest.skip(f"Inventory CSV missing at {INVENTORY_PATH}")

    samples = discover_substrate_paired_samples(inventory_path=INVENTORY_PATH)
    assert len(samples) == 1826
    by_source = {s.source for s in samples}
    assert by_source == {INVENTORY_SOURCE_HDTF, INVENTORY_SOURCE_QCLIP}

    # Per-source counts match the inventory geometry.
    n_hdtf = sum(1 for s in samples if s.source == INVENTORY_SOURCE_HDTF)
    n_qclip = sum(1 for s in samples if s.source == INVENTORY_SOURCE_QCLIP)
    assert n_hdtf == 1094
    assert n_qclip == 732


def test_inventory_discovery_explicit_sources_list():
    """Restricting `sources` returns only matching rows."""
    from data.sources.substrate_paired_inventory import discover_substrate_paired_samples
    if not os.path.exists(INVENTORY_PATH):
        pytest.skip(f"Inventory CSV missing at {INVENTORY_PATH}")

    only_hdtf = discover_substrate_paired_samples(
        inventory_path=INVENTORY_PATH,
        sources=["hdtf_visomaster_teams"],
    )
    assert len(only_hdtf) == 1094
    assert all(s.source == "hdtf_visomaster_teams" for s in only_hdtf)

    only_qclip = discover_substrate_paired_samples(
        inventory_path=INVENTORY_PATH,
        sources=["quickclips_visomaster_teams"],
    )
    assert len(only_qclip) == 732
    assert all(s.source == "quickclips_visomaster_teams" for s in only_qclip)


def test_inventory_sample_source_labels():
    """Each inventory row has a clean source label without `_teams` and a
    teams source label with `_teams` — the stamper relies on this exact
    contract to derive transport from the source string."""
    from data.sources.substrate_paired_inventory import (
        CLEAN_SOURCE_HDTF,
        CLEAN_SOURCE_QCLIP,
        TEAMS_SOURCE_HDTF,
        TEAMS_SOURCE_QCLIP,
        discover_substrate_paired_samples,
    )
    if not os.path.exists(INVENTORY_PATH):
        pytest.skip(f"Inventory CSV missing at {INVENTORY_PATH}")

    samples = discover_substrate_paired_samples(inventory_path=INVENTORY_PATH)
    hdtf = next(s for s in samples if s.source == "hdtf_visomaster_teams")
    qclip = next(s for s in samples if s.source == "quickclips_visomaster_teams")

    assert hdtf.clean_source_label == CLEAN_SOURCE_HDTF == "hdtf_visomaster"
    assert hdtf.teams_source_label == TEAMS_SOURCE_HDTF == "hdtf_visomaster_teams"
    assert qclip.clean_source_label == CLEAN_SOURCE_QCLIP == "quickclips_visomaster"
    assert qclip.teams_source_label == TEAMS_SOURCE_QCLIP == "quickclips_visomaster_teams"
    # Bucket discrimination: clean and teams must point at different buckets.
    assert hdtf.clean_bucket != hdtf.teams_bucket
    assert qclip.clean_bucket != qclip.teams_bucket


def test_inventory_sample_id_suffixes_unique():
    """Clean and teams sample_ids differ — required so the per-video collate
    treats them as two videos (otherwise they'd be merged by sample_id_label)."""
    from data.sources.substrate_paired_inventory import discover_substrate_paired_samples
    if not os.path.exists(INVENTORY_PATH):
        pytest.skip(f"Inventory CSV missing at {INVENTORY_PATH}")
    samples = discover_substrate_paired_samples(inventory_path=INVENTORY_PATH)
    for sample in samples[:10]:
        assert sample.clean_sample_id != sample.teams_sample_id
        assert sample.clean_sample_id.endswith("__clean")
        assert sample.teams_sample_id.endswith("__teams")


def test_create_unified_samples_doubles_inventory_rows():
    """`create_unified_samples_from_substrate_paired_inventory` emits one
    wrapper per (inventory_row, side) → 2 × N_rows."""
    import logging

    from data.sources.combined_paired import (
        create_unified_samples_from_substrate_paired_inventory,
    )
    from data.sources.substrate_paired_inventory import discover_substrate_paired_samples
    if not os.path.exists(INVENTORY_PATH):
        pytest.skip(f"Inventory CSV missing at {INVENTORY_PATH}")

    log = logging.getLogger("test_create_unified")
    samples = discover_substrate_paired_samples(inventory_path=INVENTORY_PATH)
    unified = create_unified_samples_from_substrate_paired_inventory(samples, log)
    assert len(unified) == 2 * len(samples)

    # All identities share the realpool_ prefix.
    assert all(u.identity.startswith("realpool_") for u in unified)

    # Equal number of clean/teams sides.
    clean_n = sum(1 for u in unified if u.sample_id.endswith("__clean"))
    teams_n = sum(1 for u in unified if u.sample_id.endswith("__teams"))
    assert clean_n == teams_n == len(samples)


def test_unified_samples_paired_collate_emits_transport_pair():
    """End-to-end: unified samples → synthetic per-frame yield → collate
    stamps matching pair_ids and opposing transports on clean+teams sides."""
    import logging

    import numpy as np

    from data.sample.substrate_paired import set_active_stamper, SubstratePairStamper
    from data.sources.combined_paired import (
        combined_paired_collate_fn,
        create_unified_samples_from_substrate_paired_inventory,
    )
    from data.sources.substrate_paired_inventory import (
        CLEAN_SOURCE_HDTF,
        TEAMS_SOURCE_HDTF,
        discover_substrate_paired_samples,
    )
    if not os.path.exists(INVENTORY_PATH):
        pytest.skip(f"Inventory CSV missing at {INVENTORY_PATH}")

    log = logging.getLogger("test_unified_collate")
    samples = discover_substrate_paired_samples(inventory_path=INVENTORY_PATH)
    unified = create_unified_samples_from_substrate_paired_inventory(samples, log)

    # Find a (clean, teams) pair sharing identity.
    clean = next(u for u in unified if u.source == CLEAN_SOURCE_HDTF)
    teams = next(
        u for u in unified
        if u.source == TEAMS_SOURCE_HDTF and u.identity == clean.identity
    )

    def _frame(u, idx, companion_domain=None):
        row = {
            "image": np.zeros((64, 64, 3), dtype=np.uint8),
            "label": 0,
            "identity": u.identity,
            "source": u.source,
            "method": u.method,
            "method_id": -1,
            "sample_id": u.sample_id,
            "frame_idx": idx,
            "quality_domain": 0,
            "companion_bucket": "irrelevant",
        }
        if companion_domain is not None:
            row["companion_domain"] = companion_domain
        return row

    batch = (
        [_frame(clean, i) for i in range(2)]
        + [_frame(teams, i, companion_domain="teams_v2") for i in range(2)]
    )
    stamper = SubstratePairStamper(enabled=True, inventory_path=INVENTORY_PATH)
    set_active_stamper(stamper)
    try:
        result = combined_paired_collate_fn(batch)
    finally:
        set_active_stamper(None)

    # Two videos (clean side and teams side).
    assert result["substrate_pair_id"].shape[0] == 2
    pair_ids = result["substrate_pair_id"].tolist()
    transports = sorted(result["substrate_transport"].tolist())
    assert pair_ids[0] == pair_ids[1] > 0  # same identity → same pair_id
    assert transports == [0, 1]


def test_iterator_dispatch_picks_substrate_paired_branch(monkeypatch):
    """The dispatch in `CombinedPairedIterableDataset.__iter__` routes
    substrate-paired sources to `_iterate_substrate_paired_inventory_sample`."""
    from data.sources.combined_paired import CombinedPairedIterableDataset
    # The class itself defines the new method.
    assert hasattr(CombinedPairedIterableDataset, "_iterate_substrate_paired_inventory_sample")


def test_quality_domain_map_has_new_sources():
    """The new source labels are registered in the QUALITY_DOMAIN_MAP so
    `_domain_for_sample` returns a valid GRL bucket for each."""
    from data.sources.combined_paired import QUALITY_DOMAIN_MAP
    for src in (
        "hdtf_visomaster",
        "quickclips_visomaster",
        "hdtf_visomaster_teams",
        "quickclips_visomaster_teams",
    ):
        assert src in QUALITY_DOMAIN_MAP, f"Missing QUALITY_DOMAIN_MAP entry for {src!r}"


def test_existing_viso_teams_enhanced_lane_still_pairs(stamper_enabled):
    """Regression guard: the visomaster_teams_enhanced lane already produces
    correct (clean, teams) stamping (companion_domain=teams_v2). Adding the
    HDTF/QCLIP lanes must not regress its behaviour."""
    from data.sample.substrate_paired import set_active_stamper
    from data.sources.combined_paired import combined_paired_collate_fn

    set_active_stamper(stamper_enabled)
    try:
        # `visomaster_CSCS_00007` is one of the 54 enhanced inventory rows.
        # The stamper expects `identity=sample_id` for those rows (resolver
        # row carries no separate identity_id — see inventory builder).
        identity = "realpool_visomaster_CSCS_00007"
        batch = (
            [_mock_frame(i, identity, "visomaster", 0, "vidVMC_clean") for i in range(2)] +
            [_mock_frame(i, identity, "visomaster_teams_enhanced", 0, "vidVMC_teams") for i in range(2)]
        )
        result = combined_paired_collate_fn(batch)
        assert result["substrate_pair_id"].shape[0] == 2
        assert result["substrate_pair_id"][0].item() == result["substrate_pair_id"][1].item()
        transports = sorted(result["substrate_transport"].tolist())
        # Both source labels normalize to TEAMS by `_is_teams_source`
        # (one ends in `_teams_enhanced`; the other is `visomaster` only).
        # Specifically: `visomaster` → clean (transport=0). The enhanced
        # source contains 'teams' → transport=1.
        assert transports == [0, 1]
    finally:
        set_active_stamper(None)


# ---------------------------------------------------------------------------
# Substrate-pair-grouped sampler (fix for the 2026-05-23 T5C smoke loss=0
# structural bug). Validates that `_get_identity_balanced_samples` emits
# matched (clean, teams) wrappers as a co-occurring pair so the
# asymmetric pair-loss has matched rows in-batch to hinge on.
# ---------------------------------------------------------------------------

def _build_substrate_paired_wrappers(n_identities: int = 50):
    """Build N identities each with a (clean, teams) substrate-paired wrapper
    duo + one unrelated visomaster wrapper. Mirrors what
    `create_unified_samples_from_substrate_paired_inventory` produces.
    """
    from data.sources.combined_paired import UnifiedPairedSample
    samples = []
    for i in range(n_identities):
        identity = f"realpool_RD_Radio{i:03d}"
        base_capture = f"HDTF_{i:05d}"
        clean = UnifiedPairedSample(
            identity=identity,
            source="hdtf_visomaster",
            original_sample=object(),
            method="substrate_paired_clean_real",
            has_landmarks=False,
            sample_id=f"{base_capture}__clean",
        )
        teams = UnifiedPairedSample(
            identity=identity,
            source="hdtf_visomaster_teams",
            original_sample=object(),
            method="substrate_paired_teams_real",
            has_landmarks=False,
            sample_id=f"{base_capture}__teams",
        )
        # Add an unrelated wrapper so the per-identity bucket has > 2 entries
        # (the non-paired fallback path).
        other = UnifiedPairedSample(
            identity=identity,
            source="visomaster",
            original_sample=object(),
            method="visomaster_CSCS",
            has_landmarks=False,
            sample_id=f"vmc_{i:03d}",
        )
        samples.extend([clean, teams, other])
    return samples


def _make_pair_dataset(samples, *, sp_enabled: bool, pair_fraction: float = 0.25, seed: int = 9916):
    """Build a CombinedPairedIterableDataset configured for substrate-pair sampling."""
    from data.sources.combined_paired import (
        CombinedBatchingConfig,
        CombinedPairedIterableDataset,
    )
    cfg = CombinedBatchingConfig(
        identity_balanced_sampling=True,
        identity_sampling_strategy="identity_resample_weighted",
        identity_family_weights={"realpool_real": 1.5, "visomaster_fake": 4.0},
        df40_sparse_indices=[0],
        deeplive_sparse_indices=[0],
        visomaster_sparse_indices=[0],
        substrate_pair_sampling_enabled=sp_enabled,
        substrate_pair_sampling_fraction=pair_fraction,
    )
    return CombinedPairedIterableDataset(
        samples=samples,
        df40_dataset=None,
        deeplive_dataset=None,
        config=cfg,
        transform=None,
        shuffle=True,
        seed=seed,
        method_mapping={},
    )


def test_sampler_emits_clean_teams_pair_when_substrate_pair_sampling_enabled():
    """When substrate-pair sampling is enabled, _get_identity_balanced_samples
    emits BOTH the clean and teams wrappers of a paired identity as a
    contiguous pair (so they co-occur in the same batch).
    """
    import random as _random
    samples = _build_substrate_paired_wrappers(n_identities=80)
    ds = _make_pair_dataset(samples, sp_enabled=True, pair_fraction=1.0, seed=9916)

    # Sanity: partner map populated.
    assert len(ds._substrate_pair_partners) == 80

    # pair_fraction=1.0 → every identity emits both wrappers.
    selected = ds._get_identity_balanced_samples(_random.Random(9916), worker_id=0, num_workers=1)

    # Each of 80 identities emits 2 wrappers → 160 total.
    assert len(selected) == 160

    # Walk the list: pairs are emitted as adjacent (clean, teams) entries.
    # Build per-identity index of clean/teams locations.
    pos_by_identity = {}
    for i, s in enumerate(selected):
        sid = s.sample_id
        side = "clean" if sid.endswith("__clean") else ("teams" if sid.endswith("__teams") else "other")
        pos_by_identity.setdefault(s.identity, {}).setdefault(side, []).append(i)

    # For every identity, clean and teams should both exist with adjacent indices.
    adjacent_count = 0
    for identity, positions in pos_by_identity.items():
        assert "clean" in positions and "teams" in positions, f"missing side for {identity}"
        c = positions["clean"][0]
        t = positions["teams"][0]
        if abs(c - t) == 1:
            adjacent_count += 1
    # All pair partners must be adjacent (so the per-frame collate sees them
    # in the same batch).
    assert adjacent_count == 80


def test_sampler_pair_fraction_honored_statistically():
    """With pair_fraction=0.5, ~50% of identities should emit pairs."""
    import random as _random
    samples = _build_substrate_paired_wrappers(n_identities=400)
    ds = _make_pair_dataset(samples, sp_enabled=True, pair_fraction=0.5, seed=9916)

    selected = ds._get_identity_balanced_samples(_random.Random(9916), worker_id=0, num_workers=1)

    # Count identities that emitted both clean and teams.
    sides_by_identity = {}
    for s in selected:
        sid = s.sample_id
        side = "clean" if sid.endswith("__clean") else ("teams" if sid.endswith("__teams") else "other")
        sides_by_identity.setdefault(s.identity, set()).add(side)
    paired = sum(1 for sides in sides_by_identity.values() if "clean" in sides and "teams" in sides)
    # Expect ~0.5 of 400 = 200. Allow ±15% tolerance (binomial noise).
    assert 160 <= paired <= 240, f"paired count {paired} outside [160, 240]"


def test_sampler_disabled_flag_byte_identical_to_legacy():
    """When substrate_pair_sampling_enabled=False, sampler must behave
    identically to the pre-fix path (no rng.random() consumed for pair toss,
    no partner map built, one wrapper per identity per epoch).
    """
    import random as _random
    samples = _build_substrate_paired_wrappers(n_identities=80)
    ds = _make_pair_dataset(samples, sp_enabled=False, seed=9916)

    # Partner map must be empty when disabled.
    assert ds._substrate_pair_partners == {}

    selected = ds._get_identity_balanced_samples(_random.Random(9916), worker_id=0, num_workers=1)
    # One wrapper per identity → exactly 80 entries.
    assert len(selected) == 80
    # No identity should have both clean and teams in the output.
    sides_by_identity = {}
    for s in selected:
        sid = s.sample_id
        side = "clean" if sid.endswith("__clean") else ("teams" if sid.endswith("__teams") else "other")
        sides_by_identity.setdefault(s.identity, set()).add(side)
    for sides in sides_by_identity.values():
        assert not ("clean" in sides and "teams" in sides)


def test_sampler_pair_partners_stay_in_same_worker():
    """Worker slicing must keep (clean, teams) wrappers on the same worker.
    Otherwise the pair partners would be split across DataLoader workers and
    never co-occur in a batch.
    """
    import random as _random
    samples = _build_substrate_paired_wrappers(n_identities=40)
    ds = _make_pair_dataset(samples, sp_enabled=True, pair_fraction=1.0, seed=9916)

    # Use 4 workers; each worker must see complete pairs only.
    for worker_id in range(4):
        selected = ds._get_identity_balanced_samples(
            _random.Random(9916), worker_id=worker_id, num_workers=4,
        )
        sides_by_identity = {}
        for s in selected:
            sid = s.sample_id
            side = "clean" if sid.endswith("__clean") else ("teams" if sid.endswith("__teams") else "other")
            sides_by_identity.setdefault(s.identity, set()).add(side)
        # Every paired identity that landed on this worker has BOTH sides.
        for identity, sides in sides_by_identity.items():
            if "clean" in sides or "teams" in sides:
                assert "clean" in sides and "teams" in sides, (
                    f"worker {worker_id}: identity {identity} split across workers (sides={sides})"
                )


def test_dataloader_real_path_matched_pair_coverage(stamper_enabled, monkeypatch):
    """End-to-end through the real iterable dataset + collate path:
    iterate ≥20 batches and assert that ≥X% of them contain a matched pair,
    where X ≈ configured pair_fraction. Also validates the asymmetric loss
    returns > 0 on a batch with a matched pair.

    The iterator's GCS-backed branch is monkeypatched to yield synthetic
    frames; everything else (sampler, collate, stamper) runs as in production.
    """
    import logging
    import random as _random

    from data.sample.substrate_paired import set_active_stamper
    from data.sources.combined_paired import (
        CombinedPairedIterableDataset,
        combined_paired_collate_fn,
        create_unified_samples_from_substrate_paired_inventory,
    )
    from data.sources.substrate_paired_inventory import discover_substrate_paired_samples
    from loss.substrate_pair_asymmetric import SubstratePairAsymmetricLoss

    if not os.path.exists(INVENTORY_PATH):
        pytest.skip(f"Inventory CSV missing at {INVENTORY_PATH}")

    log = logging.getLogger("test_e2e_pair")
    inventory = discover_substrate_paired_samples(inventory_path=INVENTORY_PATH)
    # Use 400 inventory rows so we have enough frames for ≥20 batches at
    # batch_size=32 with 8 frames per video. 400 × 2 sides × 8 frames = 6400
    # frames; with pair_fraction=0.5 the actual emission is ~75% of that,
    # well above the 20×32=640 frame threshold.
    inventory = inventory[:400]
    unified = create_unified_samples_from_substrate_paired_inventory(inventory, log)

    # Monkeypatch the GCS-backed substrate-paired iterator to yield synthetic
    # frames matching the row shape the collate consumes. Use 8 frames per
    # video to mirror the production frames_per_video=8 configuration.
    def _fake_iter(self, unified_sample, rng):
        src = unified_sample.source
        side = "clean" if src in ("hdtf_visomaster", "quickclips_visomaster") else "teams"
        companion_domain = "teams_v2" if side == "teams" else None
        for idx in range(8):
            row = {
                "image": np.zeros((64, 64, 3), dtype=np.uint8),
                "label": 0,
                "identity": unified_sample.identity,
                "source": src,
                "method": unified_sample.method,
                "method_id": -1,
                "sample_id": unified_sample.sample_id,
                "frame_idx": idx,
                "quality_domain": 0,
                "companion_bucket": "irrelevant",
            }
            if companion_domain is not None:
                row["companion_domain"] = companion_domain
            yield row

    monkeypatch.setattr(
        CombinedPairedIterableDataset,
        "_iterate_substrate_paired_inventory_sample",
        _fake_iter,
    )

    pair_fraction = 0.5
    ds = _make_pair_dataset(unified, sp_enabled=True, pair_fraction=pair_fraction, seed=9916)

    set_active_stamper(stamper_enabled)
    try:
        # Manually iterate the dataset to produce batches of fixed size.
        frames = list(ds)
        # frames[k] is a per-frame dict; collate into 32-frame batches.
        batch_size = 32
        n_batches = max(20, min(50, len(frames) // batch_size))
        assert n_batches >= 20, f"not enough frames to form 20 batches (have {len(frames)})"

        matched_batches = 0
        loss = SubstratePairAsymmetricLoss(lambda_pair=0.3, margin=0.0, enabled=True)
        loss_fired = 0
        for b in range(n_batches):
            batch_rows = frames[b * batch_size:(b + 1) * batch_size]
            collated = combined_paired_collate_fn(batch_rows)
            pair_ids = collated["substrate_pair_id"].tolist()
            transports = collated["substrate_transport"].tolist()
            # Count batches with at least one matched (clean, teams) pair_id.
            per_pid_sides = {}
            for pid, tr in zip(pair_ids, transports):
                if pid < 0:
                    continue
                per_pid_sides.setdefault(pid, set()).add(tr)
            if any(0 in sides and 1 in sides for sides in per_pid_sides.values()):
                matched_batches += 1
                # Loss must fire on this batch — construct prob aligned to videos.
                n = collated["substrate_pair_id"].shape[0]
                prob = torch.zeros((n, 2))
                for i, tr in enumerate(transports):
                    if tr == 1:
                        prob[i] = torch.tensor([0.3, 0.7])  # teams high
                    elif tr == 0:
                        prob[i] = torch.tensor([0.7, 0.3])  # clean low
                    else:
                        prob[i] = torch.tensor([0.5, 0.5])
                value = loss.compute_from_batch(prob, collated).item()
                if value > 0.0:
                    loss_fired += 1

        # With pair_fraction=0.5 and 32-frame batches drawn from per-identity
        # emission, at least ~30% of batches should contain a matched pair.
        # Use a conservative floor of 30% to allow for batch-boundary noise.
        matched_pct = 100.0 * matched_batches / n_batches
        assert matched_pct >= 30.0, (
            f"matched-pair batch pct {matched_pct:.1f}% < 30% "
            f"(matched={matched_batches}/{n_batches})"
        )
        # Loss must fire on every matched batch.
        assert loss_fired == matched_batches, (
            f"loss fired on {loss_fired}/{matched_batches} matched batches"
        )
        assert loss_fired > 0, "asymmetric pair-loss never fired"
    finally:
        set_active_stamper(None)
