"""Quick smoke test for preset_overrides feature."""
from data.augmentations.pipelines import (
    QualityTargetedFamilyRouter,
    create_quality_targeted_family_router,
    _QUALITY_TARGETED_PRESETS,
)

# 1. No overrides — should use preset as-is
r1 = create_quality_targeted_family_router(strength='light')
assert r1._preset['context_variation_enabled'] == False

# 2. Override context_variation_enabled
r2 = create_quality_targeted_family_router(
    strength='light',
    preset_overrides={'context_variation_enabled': True, 'context_variation_brightness': 0.25},
)
assert r2._preset['context_variation_enabled'] == True
assert r2._preset['context_variation_brightness'] == 0.25
assert r2._preset['jpeg_lower'] == 62  # non-overridden key stays

# 3. Override should win even when preset already has the key set
r3 = create_quality_targeted_family_router(
    strength='vcd_targeted',
    preset_overrides={'context_variation_enabled': False},
)
assert r3._preset['context_variation_enabled'] == False

# 4. None overrides — backward compat
r4 = create_quality_targeted_family_router(strength='vcd_targeted', preset_overrides=None)
assert r4._preset['context_variation_enabled'] == True

# 5. Original preset dict is NOT mutated
assert _QUALITY_TARGETED_PRESETS['light']['context_variation_enabled'] == False

print('All assertions passed!')
