"""
PE_PAIR_RANK_DRO wiring tests (added 2026-05-07).

Covers three changes landed for the P1 packet:
1. `combined_paired_collate_fn` emits per-video `pair_id` and `group_id`.
2. `EffortDetector._compute_pair_rank_loss` — softplus margin loss in
   logit space on real-vs-fake same-source paired videos.
3. `GroupDROMixin` accepts `data_params.group_id_mapping` (string keys via
   the R-D / F-B asymmetric scheme) alongside the legacy `method_mapping`.

Tests follow the import-from-source pattern used by
`tests/test_unpaired_reals_and_grl.py` so they run without a full package install.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Bootstrap — same shape as test_unpaired_reals_and_grl.py
# ---------------------------------------------------------------------------

_TRAINING_ROOT = Path(__file__).resolve().parents[1]

if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))


def _load_module(rel_path: str, module_name: str):
    module_path = _TRAINING_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_combined_paired():
    """Load combined_paired.py with proper package context for relative imports."""
    if "utils" not in sys.modules:
        utils_pkg = types.ModuleType("utils")
        utils_pkg.__path__ = [str(_TRAINING_ROOT / "utils")]
        sys.modules["utils"] = utils_pkg
    if "utils.grouping" not in sys.modules:
        grouping_mod = _load_module("utils/grouping.py", "utils.grouping")
        sys.modules["utils.grouping"] = grouping_mod

    if "data" not in sys.modules:
        data_pkg = types.ModuleType("data")
        data_pkg.__path__ = [str(_TRAINING_ROOT / "data")]
        sys.modules["data"] = data_pkg

    if "data.sources" not in sys.modules:
        init_path = _TRAINING_ROOT / "data" / "sources" / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            "data.sources", init_path,
            submodule_search_locations=[str(_TRAINING_ROOT / "data" / "sources")],
        )
        src_mod = importlib.util.module_from_spec(spec)
        sys.modules["data.sources"] = src_mod
        spec.loader.exec_module(src_mod)

    sys.modules.pop("data.sources.combined_paired", None)
    module_path = _TRAINING_ROOT / "data" / "sources" / "combined_paired.py"
    spec = importlib.util.spec_from_file_location(
        "data.sources.combined_paired", module_path,
        submodule_search_locations=[],
    )
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "data.sources"
    sys.modules["data.sources.combined_paired"] = module
    spec.loader.exec_module(module)
    return module


def _load_group_dro_mixin():
    """Load trainer/mixins/group_dro.py as a free module (no package deps)."""
    return _load_module("trainer/mixins/group_dro.py", "group_dro_mod")


# ---------------------------------------------------------------------------
# Part 1 — Collate emits pair_id and group_id
# ---------------------------------------------------------------------------


class TestCollatePairAndGroupIds:
    def _get_collate(self):
        return _load_combined_paired().combined_paired_collate_fn

    def _make_batch(
        self,
        sample_ids: list,
        labels: list,
        group_ids: list | None = None,
        frames_per_video: int = 2,
    ):
        """Build a flat per-frame batch list. len(sample_ids)==len(labels)==len(group_ids)."""
        if group_ids is None:
            group_ids = [None] * len(sample_ids)
        batch = []
        for sid, lbl, gid in zip(sample_ids, labels, group_ids):
            for f_idx in range(frames_per_video):
                row = {
                    "image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                    "label": lbl,
                    "identity": f"id_{sid}",
                    "source": "test_source",
                    "method": "test_method",
                    "method_id": 0,
                    "sample_id": sid,
                    "frame_idx": f_idx,
                    "quality_domain": 0,
                }
                if gid is not None:
                    row["group_id"] = gid
                batch.append(row)
        return batch

    def test_pair_id_emitted(self):
        collate = self._get_collate()
        batch = self._make_batch(sample_ids=["s_a", "s_a"], labels=[0, 1])
        out = collate(batch)
        assert "pair_id" in out
        assert isinstance(out["pair_id"], list)
        # Two videos: one real (sample s_a, label 0), one fake (sample s_a, label 1).
        # Both share pair_id == "s_a".
        assert len(out["pair_id"]) == 2
        assert out["pair_id"] == ["s_a", "s_a"]

    def test_pair_id_distinct_for_distinct_samples(self):
        collate = self._get_collate()
        batch = self._make_batch(
            sample_ids=["s_a", "s_a", "s_b", "s_b"],
            labels=[0, 1, 0, 1],
        )
        out = collate(batch)
        assert sorted(out["pair_id"]) == sorted(["s_a", "s_a", "s_b", "s_b"])

    def test_pair_id_empty_for_empty_sample_id(self):
        """Unpaired reals (e.g. external_vcd_real) default sample_id = ""."""
        collate = self._get_collate()
        batch = self._make_batch(sample_ids=[""], labels=[0])
        out = collate(batch)
        assert out["pair_id"] == [""]

    def test_group_id_passes_through(self):
        collate = self._get_collate()
        batch = self._make_batch(
            sample_ids=["s_a", "s_a"],
            labels=[0, 1],
            group_ids=["real|src|teams|hi-q|regular", "fake|deeplive|gpen|teams|hi-q"],
        )
        out = collate(batch)
        assert "group_id" in out
        assert sorted(out["group_id"]) == sorted([
            "real|src|teams|hi-q|regular",
            "fake|deeplive|gpen|teams|hi-q",
        ])

    def test_group_id_derived_when_loader_omits_field(self):
        """When the loader doesn't pre-stamp group_id, the collate derives it
        from the existing method/source/identity fields. This was changed
        2026-05-07 — previously the collate just passed None through; now it
        invokes _derive_group_id_for_yield_row so PE_PAIR_RANK_DRO works on
        any loader without per-yield-site changes."""
        collate = self._get_collate()
        batch = self._make_batch(sample_ids=["s_a"], labels=[0])  # no group_id key
        out = collate(batch)
        assert len(out["group_id"]) == 1
        gid = out["group_id"][0]
        # Real-side row → starts with "real|"
        assert gid is not None and gid.startswith("real|"), gid
        # Source 'test_source' is in row, transport falls through to raw_capture
        assert "raw_capture" in gid

    def test_group_id_explicit_overrides_derivation(self):
        """When the loader DOES pre-stamp group_id, the collate uses that
        verbatim — derivation is the fallback, not a forced rewrite."""
        collate = self._get_collate()
        batch = self._make_batch(
            sample_ids=["s_a"],
            labels=[0],
            group_ids=["EXPLICIT_GID_FROM_LOADER"],
        )
        out = collate(batch)
        assert out["group_id"] == ["EXPLICIT_GID_FROM_LOADER"]

    def test_empty_batch_returns_empty_pair_and_group_ids(self):
        collate = self._get_collate()
        out = collate([])
        assert out["pair_id"] == []
        assert out["group_id"] == []


# ---------------------------------------------------------------------------
# Part 2 — EffortDetector._compute_pair_rank_loss
# ---------------------------------------------------------------------------


def _load_pair_rank_helper():
    """Pull the unbound `_compute_pair_rank_loss` method off EffortDetector
    without instantiating the (open_clip-dependent) detector."""
    detector_mod = _load_module("detectors/effort_detector.py", "effort_for_pairtest")
    return detector_mod.EffortDetector._compute_pair_rank_loss


class TestPairRankLoss:
    """Test the pair-rank loss directly via the unbound method."""

    def setup_method(self):
        self._fn = _load_pair_rank_helper()

    def _call(self, scores, labels, pair_ids, margin=0.5):
        # Bind a dummy `self` (only used inside the method? No — method is a
        # regular method but doesn't reference self.* anywhere). We can pass
        # any sentinel.
        return self._fn(None, scores=scores, labels=labels, pair_ids=pair_ids, margin=margin)

    def test_margin_satisfied_low_loss(self):
        # fake score >> real score in same pair → softplus(margin - large_gap) ≈ 0
        scores = torch.tensor([0.05, 0.95])  # real, fake
        labels = torch.tensor([0, 1], dtype=torch.long)
        pair_ids = ["pair1", "pair1"]
        loss = self._call(scores, labels, pair_ids, margin=0.5)
        assert loss.item() < 0.05  # softplus is monotone; gap is ~6 logit units

    def test_margin_violated_positive_loss(self):
        # fake score < real score → big softplus
        scores = torch.tensor([0.95, 0.05])  # real, fake
        labels = torch.tensor([0, 1], dtype=torch.long)
        pair_ids = ["pair1", "pair1"]
        loss = self._call(scores, labels, pair_ids, margin=0.5)
        # softplus(0.5 - (-6)) ≈ 6.5
        assert loss.item() > 5.0

    def test_no_eligible_pairs_returns_zero_with_grad(self):
        # All same label → no real+fake pair
        scores = torch.tensor([0.3, 0.4], requires_grad=True)
        labels = torch.tensor([0, 0], dtype=torch.long)
        pair_ids = ["p1", "p2"]
        loss = self._call(scores, labels, pair_ids)
        assert float(loss) == 0.0
        assert loss.requires_grad

    def test_empty_pair_id_skipped(self):
        # Only the empty-pair_id real and fake share an "id"; with the skip
        # rule on empty strings, no pair groups remain.
        scores = torch.tensor([0.2, 0.8])
        labels = torch.tensor([0, 1], dtype=torch.long)
        pair_ids = ["", ""]
        loss = self._call(scores, labels, pair_ids)
        assert float(loss) == 0.0

    def test_only_paired_pairs_contribute(self):
        """A batch with one paired and one unpaired pair_id should only fire
        on the paired one."""
        scores = torch.tensor([0.05, 0.95, 0.95, 0.05])  # 2x paired, 2x both-real-different-pid
        labels = torch.tensor([0, 1, 0, 0], dtype=torch.long)
        pair_ids = ["p1", "p1", "p2", "p3"]
        loss = self._call(scores, labels, pair_ids)
        # Only p1 contributes; gap is large positive → low softplus
        assert loss.item() < 0.05

    def test_shape_mismatch_safe_noop(self):
        scores = torch.tensor([0.5, 0.5], requires_grad=True)
        labels = torch.tensor([0, 1], dtype=torch.long)
        pair_ids = ["p1"]  # length mismatch
        loss = self._call(scores, labels, pair_ids)
        assert float(loss) == 0.0
        assert loss.requires_grad

    def test_gradient_flows_through_scores(self):
        scores = torch.tensor([0.6, 0.4], requires_grad=True)
        labels = torch.tensor([0, 1], dtype=torch.long)
        pair_ids = ["p1", "p1"]
        loss = self._call(scores, labels, pair_ids, margin=0.5)
        loss.backward()
        assert scores.grad is not None
        # Loss penalizes fake_score < real_score: real grad should be positive (push real lower).
        # Numerically: ∂loss/∂real_logit > 0 for real, < 0 for fake.
        # In probability space we check the same direction holds via chain rule.
        assert scores.grad[0].item() > 0  # real wants to decrease score
        assert scores.grad[1].item() < 0  # fake wants to increase score

    def test_multiple_pair_ids_averaged(self):
        # Two pair groups, both with margin violations of different sizes.
        # Loss should be the mean of the two softplus terms.
        scores = torch.tensor([0.9, 0.1, 0.7, 0.3])
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        pair_ids = ["p1", "p1", "p2", "p2"]
        loss = self._call(scores, labels, pair_ids, margin=0.5)
        # Both pairs have fake_score < real_score; loss must be positive,
        # roughly bounded above by softplus(0.5 - delta_p1) ≈ 4.9 (worst).
        assert 0.5 < loss.item() < 5.0


# ---------------------------------------------------------------------------
# Part 3 — GroupDRO mixin: group_id_mapping path + method_mapping fallback
# ---------------------------------------------------------------------------


class _FakeHostBase:
    """Minimal host satisfying GroupDROMixin's `self.config / self.model / self.logger` contract."""

    def __init__(self, config: dict):
        self.config = config
        self.model = nn.Linear(2, 2)
        self.logger = logging.getLogger("test_group_dro")
        self.logger.setLevel(logging.WARNING)


def _make_host(mixin_cls, config):
    cls = type("_FakeHost", (_FakeHostBase, mixin_cls), {})
    host = cls(config)
    host.init_group_dro()
    return host


class TestGroupDROGroupIdPath:
    def setup_method(self):
        self._mixin = _load_group_dro_mixin().GroupDROMixin

    def _config_with_group_id(self, mapping=None):
        if mapping is None:
            mapping = {
                "real|src1|teams|hi-q|regular": 0,
                "fake|deeplive|gpen|teams|hi-q": 1,
                "real|src1|teams|hi-q|chronic": 2,
            }
        return {
            "group_dro_params": {
                "beta": 3.0,
                "ema_alpha": 0.1,
                "warmup_steps": 0,  # disable warmup so weights activate immediately
            },
            "data_params": {"group_id_mapping": mapping},
        }

    def test_init_with_group_id_mapping(self):
        host = _make_host(self._mixin, self._config_with_group_id())
        assert host.num_groups == 3
        assert host.group_id_mapping is not None
        assert host.method_mapping is None
        assert host.group_losses_ema.shape == (3,)

    def test_init_with_method_mapping_fallback(self):
        method_mapping = {"FaceSwap": 0, "InSwapper": 1, "real": 2, "DeepLive": 3}
        config = {
            "group_dro_params": {"warmup_steps": 0},
            "data_params": {"method_mapping": method_mapping},
        }
        host = _make_host(self._mixin, config)
        assert host.num_groups == 4
        assert host.method_mapping is not None
        assert host.group_id_mapping is None

    def test_init_raises_when_neither_mapping_present(self):
        config = {"group_dro_params": {}, "data_params": {}}
        with pytest.raises(ValueError, match="group_id_mapping.*method_mapping"):
            _make_host(self._mixin, config)

    def test_init_raises_when_group_id_mapping_wrong_type(self):
        config = {
            "group_dro_params": {},
            "data_params": {"group_id_mapping": ["not", "a", "dict"]},
        }
        with pytest.raises(ValueError, match="must be a dict"):
            _make_host(self._mixin, config)

    def test_loss_with_group_id_strings(self):
        host = _make_host(self._mixin, self._config_with_group_id())
        per_sample_loss = torch.tensor([0.4, 1.2, 0.6])
        data_dict = {
            "group_id": [
                "real|src1|teams|hi-q|regular",
                "fake|deeplive|gpen|teams|hi-q",
                "real|src1|teams|hi-q|chronic",
            ],
        }
        out = host.calculate_group_dro_loss(data_dict, per_sample_loss)
        assert "overall" in out
        assert torch.is_tensor(out["overall"])
        assert out["overall"].dim() == 0  # scalar
        assert out["group_weights"].shape == (3,)

    def test_unknown_group_id_buckets_to_zero_with_warning(self, caplog):
        host = _make_host(self._mixin, self._config_with_group_id())
        per_sample_loss = torch.tensor([0.5, 0.5])
        data_dict = {"group_id": ["definitely_not_in_mapping", None]}
        with caplog.at_level(logging.WARNING, logger="test_group_dro"):
            host.calculate_group_dro_loss(data_dict, per_sample_loss)
        assert any("missing or unmapped group_id" in r.message for r in caplog.records)

    def test_missing_group_id_in_batch_raises(self):
        host = _make_host(self._mixin, self._config_with_group_id())
        per_sample_loss = torch.tensor([0.5])
        data_dict = {"label": torch.tensor([0])}  # no group_id key
        with pytest.raises(KeyError, match="group_id"):
            host.calculate_group_dro_loss(data_dict, per_sample_loss)

    def test_legacy_method_id_path_still_works(self):
        method_mapping = {"FaceSwap": 0, "InSwapper": 1}
        config = {
            "group_dro_params": {"warmup_steps": 0},
            "data_params": {"method_mapping": method_mapping},
        }
        host = _make_host(self._mixin, config)
        per_sample_loss = torch.tensor([0.5, 0.7])
        data_dict = {"method_id": torch.tensor([0, 1], dtype=torch.long)}
        out = host.calculate_group_dro_loss(data_dict, per_sample_loss)
        assert torch.is_tensor(out["overall"])
        assert out["group_weights"].shape == (2,)

    def test_warmup_returns_uniform_weights_then_activates(self):
        config = self._config_with_group_id()
        config["group_dro_params"]["warmup_steps"] = 2
        host = _make_host(self._mixin, config)
        per_sample_loss = torch.tensor([0.4, 1.2, 0.6])
        data_dict = {
            "group_id": [
                "real|src1|teams|hi-q|regular",
                "fake|deeplive|gpen|teams|hi-q",
                "real|src1|teams|hi-q|chronic",
            ],
        }
        # First two calls are warmup → uniform weights
        out1 = host.calculate_group_dro_loss(data_dict, per_sample_loss)
        out2 = host.calculate_group_dro_loss(data_dict, per_sample_loss)
        assert torch.allclose(out1["group_weights"], torch.ones(3))
        assert torch.allclose(out2["group_weights"], torch.ones(3))
        assert float(out1["group_dro_in_warmup"]) == 1.0
        # Third call is post-warmup → reweighting kicks in (group with higher EMA loss → higher weight)
        out3 = host.calculate_group_dro_loss(data_dict, per_sample_loss)
        assert float(out3["group_dro_in_warmup"]) == 0.0
        # Group 1 (the fake with loss 1.2) should be upweighted relative to others
        assert out3["group_weights"][1].item() >= out3["group_weights"][0].item()

    def test_per_frame_loss_expansion(self):
        """When per_sample_loss has B*T elements but group_id has B, expand."""
        host = _make_host(self._mixin, self._config_with_group_id())
        # B=2 videos, T=4 frames each → per_sample_loss has 8 elements
        per_sample_loss = torch.tensor([0.4, 0.4, 0.4, 0.4, 1.2, 1.2, 1.2, 1.2])
        data_dict = {
            "group_id": [
                "real|src1|teams|hi-q|regular",
                "fake|deeplive|gpen|teams|hi-q",
            ],
        }
        out = host.calculate_group_dro_loss(data_dict, per_sample_loss)
        # No crash; weighted loss is a scalar
        assert out["overall"].dim() == 0


# ---------------------------------------------------------------------------
# Part 4 — make_group_id snippet sanity (script in analysis/ is the source-of-truth)
# ---------------------------------------------------------------------------


class TestMakeGroupIdSnippet:
    """Smoke-test the snippet from analysis/group_id_design_audit_2026-05-06."""

    def _load_snippet(self):
        snippet_path = (
            _TRAINING_ROOT
            / "analysis"
            / "group_id_design_audit_2026-05-06"
            / "outputs"
            / "group_id_python_snippet.py"
        )
        return _load_module(str(snippet_path.relative_to(_TRAINING_ROOT)), "group_id_snippet")

    def test_make_group_id_real_chronic(self):
        mod = self._load_snippet()
        row = {
            "label": 0,
            "source": "teams_real_dev",
            "transport": "teams_capture",
            "quality": "lo-q",
            "base_identity": "PC_Generator__s22",  # chronic
            "method_family": "real_or_unknown",
            "enhancer_family": "none",
        }
        gid = mod.make_group_id(row)
        assert gid.startswith("real|")
        assert gid.endswith("|chronic"), gid

    def test_make_group_id_real_regular(self):
        mod = self._load_snippet()
        row = {
            "label": 0,
            "source": "teams_real_dev",
            "transport": "teams_capture",
            "quality": "hi-q",
            "base_identity": "regular_user",
        }
        gid = mod.make_group_id(row)
        assert gid.endswith("|regular")

    def test_make_group_id_fake(self):
        mod = self._load_snippet()
        row = {
            "label": 1,
            "method_family": "deeplive",
            "enhancer_family": "gpen",
            "transport": "teams_capture",
            "quality": "hi-q",
            "source": "deeplive_v2",
            "base_identity": "any",
        }
        gid = mod.make_group_id(row)
        assert gid.startswith("fake|")
        assert "deeplive" in gid
        assert "gpen" in gid

    def test_quality_band_unknown(self):
        mod = self._load_snippet()
        assert mod.quality_band(None) == "unknown"
        assert mod.quality_band("hi-q") == "hi-q"
        assert mod.quality_band("LO-Q") == "lo-q"
        assert mod.quality_band("nonsense") == "unknown"

    def test_is_chronic_handles_substring(self):
        mod = self._load_snippet()
        assert mod.is_chronic("PC_Generator__s22")
        assert mod.is_chronic("PC_Generator__s22__extra_suffix")
        assert not mod.is_chronic("regular_user")
        assert not mod.is_chronic(None)

    def test_build_group_id_mapping_dedupes(self):
        mod = self._load_snippet()
        rows = [
            {"label": 0, "source": "s1", "transport": "raw_capture", "quality": "hi-q", "base_identity": "u1"},
            {"label": 0, "source": "s1", "transport": "raw_capture", "quality": "hi-q", "base_identity": "u2"},
            {"label": 1, "method_family": "df40", "enhancer_family": "none", "transport": "raw_capture", "quality": "hi-q"},
        ]
        mapping = mod.build_group_id_mapping(rows)
        assert isinstance(mapping, dict)
        # u1 and u2 are both regular → same group → 2 unique groups total
        assert len(mapping) == 2
        # All values are contiguous ints 0..N-1
        assert sorted(mapping.values()) == list(range(len(mapping)))


# ---------------------------------------------------------------------------
# Part 5 — combined_paired.py embedded derive helpers (parity with snippet)
# ---------------------------------------------------------------------------


class TestCombinedPairedDeriveHelpers:
    """The loader embeds is_chronic / make_group_id locally rather than
    importing the snippet (the snippet's directory has hyphens — not a valid
    package path). These tests pin the embedded copy against expected
    behaviour."""

    def setup_method(self):
        self._mod = _load_combined_paired()

    def test_chronic_identity_match(self):
        f = self._mod._is_chronic_identity
        assert f("PC_Generator__s22")
        assert f("PC_Generator__s22__variant")  # substring match
        assert f("bla_bla_chow")
        assert f("BLA_BLA_CHOW")  # case-insensitive
        assert not f("regular_user")
        assert not f("")
        assert not f(None)

    def test_method_family_real_label(self):
        """All real-side rows collapse into 'real_or_unknown'."""
        f = self._mod._method_family_from
        assert f("anything", 0, "any_source") == "real_or_unknown"

    def test_method_family_visomaster_source_anchored(self):
        """Visomaster source dominates over per-method keyword."""
        f = self._mod._method_family_from
        assert f("ghostface_v2", 1, "visomaster") == "visomaster"
        assert f("Inswapper128", 1, "visomaster_enhanced") == "visomaster"

    def test_method_family_deeplive_source_anchored(self):
        f = self._mod._method_family_from
        assert f("edge_cases_enhanced", 1, "deeplive") == "deeplive"
        assert f("minimal_processing", 1, "deeplive_teams") == "deeplive"

    def test_method_family_keyword_match(self):
        f = self._mod._method_family_from
        assert f("simswap", 1, "df40") == "simswap"
        assert f("inswap_512_unofficial", 1, "df40") == "inswapper"
        assert f("ghostfaceV1", 1, "df40") == "ghostface"

    def test_enhancer_family(self):
        f = self._mod._enhancer_family_from
        assert f("inswap_gpen", "df40") == "gpen"
        assert f("any", "visomaster_enhanced") == "enhanced_unknown"
        assert f("simswap", "df40") == "none"
        assert f("any_codeformer_thing", "any") == "codeformer"

    def test_transport_from_source(self):
        f = self._mod._transport_from_source
        assert f("teams", "any") == "teams_capture"
        assert f("deeplive_teams", "any") == "teams_capture"
        assert f("visomaster_teams_enhanced", "any") == "teams_capture"
        assert f("visomaster", "any") == "visomaster"
        assert f("visomaster_enhanced", "any") == "visomaster"
        assert f("df40", "any") == "raw_capture"
        assert f("deeplive", "any") == "raw_capture"
        assert f("external", "any") == "external"

    def test_derive_group_id_for_yield_row_real_chronic(self):
        f = self._mod._derive_group_id_for_yield_row
        row = {
            "label": 0,
            "source": "teams",
            "method": "real",
            "identity": "PC_Generator__s22",  # chronic
        }
        gid = f(row)
        assert gid is not None
        assert gid.startswith("real|")
        assert gid.endswith("|chronic")
        assert "teams_capture" in gid

    def test_derive_group_id_for_yield_row_fake_visomaster(self):
        f = self._mod._derive_group_id_for_yield_row
        row = {
            "label": 1,
            "source": "visomaster_enhanced",
            "method": "Inswapper128",
            "identity": "u1",
        }
        gid = f(row)
        assert gid is not None
        assert gid.startswith("fake|visomaster|enhanced_unknown|visomaster|")

    def test_derive_group_id_returns_none_on_missing_label(self):
        f = self._mod._derive_group_id_for_yield_row
        assert f({"method": "x", "source": "y"}) is None

    def test_build_group_id_mapping_for_samples_paired(self):
        """A paired sample contributes BOTH label-0 and label-1 group_ids."""
        f = self._mod.build_group_id_mapping_for_samples

        # Duck-type a paired sample
        class _Sample:
            method = "Inswapper128"
            source = "visomaster"
            identity = "u_paired"
            is_unpaired_real = False

        mapping = f([_Sample()])
        assert len(mapping) == 2  # one real + one fake
        assert any(g.startswith("real|") for g in mapping)
        assert any(g.startswith("fake|") for g in mapping)
        # Values are contiguous ints
        assert sorted(mapping.values()) == [0, 1]

    def test_build_group_id_mapping_for_samples_unpaired_real(self):
        """An unpaired real sample contributes ONLY the label-0 group_id."""
        f = self._mod.build_group_id_mapping_for_samples

        class _UnpairedReal:
            method = "external_vcd_real"
            source = "external"
            identity = "ext_id"
            is_unpaired_real = True

        mapping = f([_UnpairedReal()])
        assert len(mapping) == 1
        assert next(iter(mapping)).startswith("real|")
