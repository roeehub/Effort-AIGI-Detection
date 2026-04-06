#!/usr/bin/env python3
"""
Quick verification script for refactored modules.

Tests that all refactored modules can be imported and basic functionality works.
Run from the training/ directory:
    python test_refactored_modules.py
"""

import sys
from pathlib import Path

# Ensure we're in the right directory
TRAINING_DIR = Path(__file__).parent
sys.path.insert(0, str(TRAINING_DIR))

print("=" * 60)
print("REFACTORED MODULES VERIFICATION")
print("=" * 60)

errors = []
successes = []


def test_import(module_name: str, description: str):
    """Test that a module can be imported."""
    try:
        __import__(module_name)
        successes.append(f"✅ {description}: {module_name}")
        return True
    except Exception as e:
        errors.append(f"❌ {description}: {module_name}\n   Error: {e}")
        return False


# ==============================================================================
# Test 1: Config System
# ==============================================================================
print("\n--- Testing Config System ---")

test_import("config_system", "Config system package")
test_import("config_system.schema", "Config schema dataclasses")
test_import("config_system.loader", "Config loader")
test_import("config_system.validation", "Config validation")

# Test loading defaults.yaml
try:
    from config_system.loader import load_yaml, DEFAULT_CONFIG_PATH
    if DEFAULT_CONFIG_PATH.exists():
        defaults = load_yaml(DEFAULT_CONFIG_PATH)
        assert 'model' in defaults, "defaults.yaml missing 'model' section"
        assert 'optimizer' in defaults, "defaults.yaml missing 'optimizer' section"
        assert 'training' in defaults, "defaults.yaml missing 'training' section"
        successes.append(f"✅ defaults.yaml: Loaded with {len(defaults)} top-level keys")
    else:
        errors.append(f"❌ defaults.yaml: File not found at {DEFAULT_CONFIG_PATH}")
except Exception as e:
    errors.append(f"❌ defaults.yaml loading: {e}")

# Test config loading function
try:
    from config_system import load_config, TrainingConfig
    # Load with just defaults (no other files needed)
    config = load_config()
    assert isinstance(config, TrainingConfig), "load_config() didn't return TrainingConfig"
    assert config.model.model_name == "effort", f"Unexpected model name: {config.model.model_name}"
    successes.append("✅ load_config(): Returns valid TrainingConfig from defaults")
except Exception as e:
    errors.append(f"❌ load_config(): {e}")

# ==============================================================================
# Test 2: Utils Module
# ==============================================================================
print("\n--- Testing Utils Module ---")

test_import("utils", "Utils package")
test_import("utils.setup", "Setup utilities")
test_import("utils.gcs", "GCS utilities")

# Test specific functions exist
try:
    from utils import init_seed, choose_optimizer, choose_scheduler, choose_metric
    successes.append("✅ utils: Core functions importable (init_seed, choose_optimizer, etc.)")
except Exception as e:
    errors.append(f"❌ utils core functions: {e}")

# ==============================================================================
# Test 3: Data Splitting Module
# ==============================================================================
print("\n--- Testing Data Splitting Module ---")

test_import("data", "Data package")
test_import("data.splitting", "Splitting subpackage")
test_import("data.splitting.base", "Splitter base class")
test_import("data.splitting.splitters", "Splitter implementations")

# Test splitter factory
try:
    from data.splitting import get_splitter, Splitter, SplitResult
    assert callable(get_splitter), "get_splitter is not callable"
    successes.append("✅ data.splitting: Factory function available")
except Exception as e:
    errors.append(f"❌ data.splitting factory: {e}")

# ==============================================================================
# Test 4: Data Batching Module
# ==============================================================================
print("\n--- Testing Data Batching Module ---")

test_import("data.batching", "Batching subpackage")
test_import("data.batching.base", "Batching base class")
test_import("data.batching.factory", "Batching factory")

# Test batching factory
try:
    from data.batching import get_batching_strategy
    assert callable(get_batching_strategy), "get_batching_strategy is not callable"
    successes.append("✅ data.batching: Factory function available")
except Exception as e:
    errors.append(f"❌ data.batching factory: {e}")

# ==============================================================================
# Test 5: Data Augmentations Module
# ==============================================================================
print("\n--- Testing Data Augmentations Module ---")

test_import("data.augmentations", "Augmentations subpackage")
test_import("data.augmentations.transforms", "Custom transforms")
test_import("data.augmentations.pipelines", "Augmentation pipelines")
test_import("data.augmentations.registry", "Augmentation registry")

# Test registry
try:
    from data.augmentations import get_augmentation_pipeline, AUGMENTATION_REGISTRY
    assert callable(get_augmentation_pipeline), "get_augmentation_pipeline is not callable"
    successes.append("✅ data.augmentations: Registry and factory available")
except Exception as e:
    errors.append(f"❌ data.augmentations registry: {e}")

# ==============================================================================
# Test 6: Bridge Function in prepare_splits
# ==============================================================================
print("\n--- Testing prepare_splits Bridge ---")

try:
    from prepare_splits import prepare_video_splits_v2
    import inspect
    sig = inspect.signature(prepare_video_splits_v2)
    params = list(sig.parameters.keys())
    if 'use_refactored' in params:
        successes.append("✅ prepare_splits: Bridge function has use_refactored parameter")
    else:
        errors.append("❌ prepare_splits: Missing use_refactored parameter")
except Exception as e:
    errors.append(f"❌ prepare_splits import: {e}")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "=" * 60)
print("VERIFICATION RESULTS")
print("=" * 60)

print(f"\n✅ PASSED: {len(successes)}")
for s in successes:
    print(f"   {s}")

if errors:
    print(f"\n❌ FAILED: {len(errors)}")
    for e in errors:
        print(f"   {e}")
    print("\n⚠️  Some tests failed. Review the errors above.")
    sys.exit(1)
else:
    print("\n🎉 All verification tests passed!")
    sys.exit(0)
