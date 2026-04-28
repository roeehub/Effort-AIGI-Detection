"""Smoke test for trainer._to_plain_dict — covers the wandb.Config sub-object case.

Why this test exists: the earlier `isinstance(raw, dict)` guard silently
dropped wandb.Config nested sub-objects (because wandb.Config is not a dict
subclass). At runtime this caused `anchor_aware`, `face_scale_jitter` and
`periodic_saves` to load with empty configs even when the yaml was correct,
and the trainer fell back to disabled-by-default. This test exercises a
fake wandb.Config-shaped object so the regression cannot recur silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.trainer import _to_plain_dict


class _FakeWandbConfig:
    """Mimics wandb.sdk.wandb_config.Config: dict-like but not a dict subclass."""

    def __init__(self, data):
        self._data = dict(data)

    def keys(self):
        return self._data.keys()

    def __getitem__(self, k):
        return self._data[k]

    def __contains__(self, k):
        return k in self._data

    def __iter__(self):
        return iter(self._data)


def test_none():
    assert _to_plain_dict(None) == {}


def test_empty_dict():
    assert _to_plain_dict({}) == {}


def test_plain_dict_passthrough():
    src = {"enabled": True, "weight": 5.0, "target_mean_prob": 0.10}
    out = _to_plain_dict(src)
    assert out == src
    assert isinstance(out, dict)
    # ensure shallow copy (mutation isolation)
    out["enabled"] = False
    assert src["enabled"] is True


def test_fake_wandb_config_unwraps():
    """The bug: wandb.Config returned False from isinstance(dict) -> empty fallback."""
    fake = _FakeWandbConfig({"enabled": True, "weight": 5.0, "samples_per_step": 16})
    out = _to_plain_dict(fake)
    assert isinstance(out, dict)
    assert out == {"enabled": True, "weight": 5.0, "samples_per_step": 16}
    assert out.get("enabled") is True


def test_step_list_in_fake_config():
    """periodic_saves nested under wandb.Config retains its step_list."""
    fake = _FakeWandbConfig({"enabled": True, "step_list": [2000, 4000, 6000, 9000, 12000, 15000, 18000]})
    out = _to_plain_dict(fake)
    assert out["enabled"] is True
    assert out["step_list"] == [2000, 4000, 6000, 9000, 12000, 15000, 18000]


def test_object_without_keys_returns_empty():
    class NotDictLike:
        pass

    assert _to_plain_dict(NotDictLike()) == {}


def test_isinstance_dict_old_pattern_would_have_dropped_fake():
    """Documents the original bug: the old guard would silently return {}."""
    fake = _FakeWandbConfig({"enabled": True})
    # The buggy pattern that was in trainer.py before:
    old_result = (fake or {}) if isinstance(fake, dict) else {}
    assert old_result == {}, "fake config should NOT pass isinstance(dict) — that was the bug"
    # The new helper recovers the data:
    assert _to_plain_dict(fake)["enabled"] is True


if __name__ == "__main__":
    import traceback

    fns = [
        test_none, test_empty_dict, test_plain_dict_passthrough,
        test_fake_wandb_config_unwraps, test_step_list_in_fake_config,
        test_object_without_keys_returns_empty,
        test_isinstance_dict_old_pattern_would_have_dropped_fake,
    ]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed} passed, {failed} failed")
    raise SystemExit(failed)
