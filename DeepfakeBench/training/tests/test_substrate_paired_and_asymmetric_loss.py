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
