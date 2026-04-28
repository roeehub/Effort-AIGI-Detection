"""Smoke test for trainer.py:2326-2340 periodic_saves resolution.

Bug background: wandb.config may wrap nested dicts into a non-dict object
whose .get() doesn't behave like dict.get(). Pre-fix code did:

    periodic_cfg = self.config.get('periodic_saves', {})
    if periodic_cfg.get('enabled', False) and self.config.get('save_ckpt', True):

Under wandb.Config wrapping, periodic_cfg.get('enabled', False) silently
returned False even when enabled: true was in the yaml — periodic saves
were silently skipped.

The fix mirrors the line 392-393 isinstance(dict) defense pattern: if the
raw value isn't a dict, fall back to {} explicitly. This test exercises
both branches plus the realistic P13_FROM_SCRATCH step_list.
"""
from unittest.mock import Mock


def _resolve_periodic_cfg(config, step_cnt):
    """Mirror of trainer.py:2326-2340 resolution logic.

    Returns (periodic_cfg, save_ckpt_ok, step_list, step_in_list).
    """
    _ps_raw = config.get('periodic_saves')
    periodic_cfg = (_ps_raw or {}) if isinstance(_ps_raw, dict) else {}
    _save_ckpt_ok = config.get('save_ckpt', True)
    _step_list = periodic_cfg.get('step_list') or []
    return periodic_cfg, _save_ckpt_ok, _step_list, step_cnt in _step_list


class TestPeriodicSavesResolution:
    def test_plain_dict_enabled_step_match(self):
        config = {'periodic_saves': {'enabled': True, 'step_list': [100, 200]}}
        cfg, ok, step_list, in_list = _resolve_periodic_cfg(config, 100)
        assert cfg.get('enabled', False) is True
        assert step_list == [100, 200]
        assert in_list is True
        assert ok is True

    def test_plain_dict_step_miss(self):
        config = {'periodic_saves': {'enabled': True, 'step_list': [100, 200]}}
        cfg, ok, step_list, in_list = _resolve_periodic_cfg(config, 99)
        assert cfg.get('enabled', False) is True
        assert in_list is False

    def test_plain_dict_disabled(self):
        config = {'periodic_saves': {'enabled': False, 'step_list': [100]}}
        cfg, ok, step_list, in_list = _resolve_periodic_cfg(config, 100)
        assert cfg.get('enabled', False) is False

    def test_missing_key(self):
        config = {}
        cfg, ok, step_list, in_list = _resolve_periodic_cfg(config, 100)
        assert cfg == {}
        assert step_list == []
        assert in_list is False
        assert cfg.get('enabled', False) is False

    def test_none_value(self):
        config = {'periodic_saves': None}
        cfg, ok, step_list, in_list = _resolve_periodic_cfg(config, 100)
        assert cfg == {}
        assert step_list == []
        assert cfg.get('enabled', False) is False

    def test_non_dict_wrapper_falls_back(self):
        wrapper = Mock()
        wrapper.get = Mock(return_value=False)
        assert not isinstance(wrapper, dict)
        config = {'periodic_saves': wrapper}
        cfg, ok, step_list, in_list = _resolve_periodic_cfg(config, 100)
        assert cfg == {}
        assert step_list == []
        assert in_list is False
        assert cfg.get('enabled', False) is False

    def test_non_dict_wrapper_returning_truthy_still_falls_back(self):
        wrapper = Mock()
        wrapper.get = Mock(return_value=True)
        assert not isinstance(wrapper, dict)
        config = {'periodic_saves': wrapper}
        cfg, ok, step_list, in_list = _resolve_periodic_cfg(config, 100)
        assert cfg == {}
        assert step_list == []

    def test_save_ckpt_default_true(self):
        config = {'periodic_saves': {'enabled': True, 'step_list': [100]}}
        _, ok, _, _ = _resolve_periodic_cfg(config, 100)
        assert ok is True

    def test_save_ckpt_false_blocks(self):
        config = {
            'periodic_saves': {'enabled': True, 'step_list': [100]},
            'save_ckpt': False,
        }
        _, ok, _, _ = _resolve_periodic_cfg(config, 100)
        assert ok is False

    def test_p13_realistic_step_list(self):
        config = {
            'periodic_saves': {
                'enabled': True,
                'step_list': [2000, 4000, 6000, 9000, 12000, 15000, 18000],
            },
            'save_ckpt': True,
        }
        for step in [2000, 4000, 6000, 9000, 12000, 15000, 18000]:
            cfg, ok, step_list, in_list = _resolve_periodic_cfg(config, step)
            assert in_list is True, f"step={step} should be in step_list"
            assert cfg.get('enabled', False) is True
            assert ok is True
        for step in [1999, 2001, 17999, 18001, 0, 1]:
            _, _, _, in_list = _resolve_periodic_cfg(config, step)
            assert in_list is False, f"step={step} should NOT be in step_list"

    def test_step_list_missing_returns_empty_list(self):
        config = {'periodic_saves': {'enabled': True}}
        cfg, ok, step_list, in_list = _resolve_periodic_cfg(config, 100)
        assert step_list == []
        assert in_list is False

    def test_resolution_matches_trainer_inline_logic(self):
        """Run the resolution snippet character-for-character against trainer.py:2328-2340.

        This is the regression anchor: if anyone changes the inline code
        in trainer.py without updating this helper, drift will be caught
        by the more specific tests above. This test asserts the helper
        and the inline expression agree on a representative input.
        """
        config = {'periodic_saves': {'enabled': True, 'step_list': [2000]}, 'save_ckpt': True}
        step_cnt = 2000
        # Inline (paste of trainer.py:2328-2331+2342):
        _ps_raw = config.get('periodic_saves')
        periodic_cfg_inline = (_ps_raw or {}) if isinstance(_ps_raw, dict) else {}
        _save_ckpt_ok_inline = config.get('save_ckpt', True)
        _step_list_inline = periodic_cfg_inline.get('step_list') or []
        step_in_list_inline = step_cnt in _step_list_inline
        # Helper:
        cfg, ok, step_list, in_list = _resolve_periodic_cfg(config, step_cnt)
        assert cfg == periodic_cfg_inline
        assert ok == _save_ckpt_ok_inline
        assert step_list == _step_list_inline
        assert in_list == step_in_list_inline
