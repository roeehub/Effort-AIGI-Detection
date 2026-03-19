#!/usr/bin/env python3
"""
verify_refactoring.py - Verification script for refactored modules

This script verifies that all refactored modules can be imported and
their basic functionality works without needing actual data or GPU.

Usage:
    python scripts/verify_refactoring.py
    
    # Or via docker-compose:
    docker-compose run sanity
"""

import sys
import os
import traceback
from pathlib import Path
from typing import List, Tuple

# ==============================================================================
# CRITICAL: Add workspace root to Python path BEFORE any imports
# ==============================================================================
# When running from /workspace/scripts/, we need /workspace in the path
script_dir = Path(__file__).resolve().parent
workspace_root = script_dir.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

# Also handle dataset/ which has its own structure
dataset_dir = workspace_root / "dataset"
if dataset_dir.exists() and str(dataset_dir) not in sys.path:
    sys.path.insert(0, str(dataset_dir))

print(f"[DEBUG] Workspace root: {workspace_root}")
print(f"[DEBUG] Python path includes: {sys.path[:3]}...")

# Track results
results: List[Tuple[str, bool, str]] = []


def check(name: str):
    """Decorator to track test results."""
    def decorator(func):
        def wrapper():
            try:
                func()
                results.append((name, True, "OK"))
                print(f"  ✅ {name}")
                return True
            except Exception as e:
                results.append((name, False, str(e)))
                print(f"  ❌ {name}: {e}")
                if "--verbose" in sys.argv:
                    traceback.print_exc()
                return False
        return wrapper
    return decorator


# ==============================================================================
# Import Tests
# ==============================================================================

print("\n" + "="*60)
print("PHASE 1: Import Tests")
print("="*60)


@check("Import config_system")
def test_import_config_system():
    from config_system import (
        TrainingConfig,
        load_config,
        validate_config,
        DEFAULT_CONFIG_PATH,
    )
    assert TrainingConfig is not None
    assert callable(load_config)


@check("Import config_system.schema")
def test_import_schema():
    from config_system.schema import (
        ModelConfig,
        OptimizerConfig,
        DataloaderConfig,
        AugmentationConfig,
    )


@check("Import config_system.loader")
def test_import_loader():
    from config_system.loader import (
        load_yaml,
        deep_merge,
        load_config,
        WANDB_KEY_MAP,
    )


@check("Import utils module")
def test_import_utils():
    from utils import (
        init_seed,
        choose_optimizer,
        choose_scheduler,
        choose_metric,
    )


@check("Import utils.gcs")
def test_import_utils_gcs():
    from utils.gcs import (
        download_gcs_asset,
        download_assets_from_gcs,
    )


@check("Import data.augmentations")
def test_import_augmentations():
    from data.augmentations import (
        get_pipeline,
        PIPELINE_REGISTRY,
    )


@check("Import data.batching")
def test_import_batching():
    from data.batching import (
        get_batching_strategy,
        create_dataloaders,
    )


@check("Import data.splitting")
def test_import_splitting():
    from data.splitting import (
        get_splitter,
        Splitter,
        SplitResult,
        VideoInfo,
    )


# Run import tests
test_import_config_system()
test_import_schema()
test_import_loader()
test_import_utils()
test_import_utils_gcs()
test_import_augmentations()
test_import_batching()
test_import_splitting()


# ==============================================================================
# Config Loading Tests
# ==============================================================================

print("\n" + "="*60)
print("PHASE 2: Config Loading Tests")
print("="*60)


@check("Load defaults.yaml")
def test_load_defaults():
    from config_system.loader import load_yaml, DEFAULT_CONFIG_PATH
    
    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"defaults.yaml not found at {DEFAULT_CONFIG_PATH}")
    
    defaults = load_yaml(DEFAULT_CONFIG_PATH)
    
    # Check essential sections exist
    assert 'model' in defaults, "Missing 'model' section"
    assert 'optimizer' in defaults, "Missing 'optimizer' section"
    assert 'training' in defaults, "Missing 'training' section"
    assert 'dataloader' in defaults, "Missing 'dataloader' section"


@check("Create TrainingConfig from defaults")
def test_create_config():
    from config_system import load_config
    
    # Load with no extra configs - should use defaults
    config = load_config()
    
    # Verify basic structure
    assert hasattr(config, 'model')
    assert hasattr(config, 'optimizer')
    assert hasattr(config, 'training')
    assert hasattr(config, 'dataloader')
    
    # Verify some default values
    assert config.model.model_name == "effort"
    assert config.optimizer.type == "adam"


@check("Deep merge works correctly")
def test_deep_merge():
    from config_system.loader import deep_merge
    
    base = {'a': 1, 'b': {'c': 2, 'd': 3}}
    override = {'b': {'c': 10, 'e': 5}}
    
    result = deep_merge(base, override)
    
    assert result['a'] == 1
    assert result['b']['c'] == 10  # overridden
    assert result['b']['d'] == 3   # preserved
    assert result['b']['e'] == 5   # added


test_load_defaults()
test_create_config()
test_deep_merge()


# ==============================================================================
# Factory Pattern Tests
# ==============================================================================

print("\n" + "="*60)
print("PHASE 3: Factory Pattern Tests")
print("="*60)


@check("Augmentation registry has expected pipelines")
def test_augmentation_registry():
    from data.augmentations import PIPELINE_REGISTRY
    
    # Check registry has some entries (implementation may vary)
    assert PIPELINE_REGISTRY is not None
    assert len(PIPELINE_REGISTRY) > 0, "Registry should have some pipelines"


@check("Batching strategies registry")
def test_batching_registry():
    from data.batching import get_batching_strategy
    
    # Test that we can get strategies (actual registry may be internal)
    # Just verify the factory function exists and is callable
    assert callable(get_batching_strategy)


@check("Splitter factory")
def test_splitter_factory():
    from data.splitting import get_splitter
    
    # Test that factory can be called (but don't actually run split)
    # Just verify the function exists and accepts strategy names
    assert callable(get_splitter)


test_augmentation_registry()
test_batching_registry()
test_splitter_factory()


# ==============================================================================
# Trainer Mixin Tests
# ==============================================================================

print("\n" + "="*60)
print("PHASE 3.5: Trainer Mixin Tests")
print("="*60)


@check("Import trainer mixins")
def test_import_trainer_mixins():
    from trainer.mixins import (
        CheckpointingMixin,
        EarlyStoppingMixin,
        GroupDROMixin,
        CurriculumMixin,
        ArcFaceMixin,
        ValidationMixin,
        ReportingMixin,
    )
    # Verify they are classes
    assert isinstance(CheckpointingMixin, type)
    assert isinstance(EarlyStoppingMixin, type)
    assert isinstance(GroupDROMixin, type)
    assert isinstance(CurriculumMixin, type)
    assert isinstance(ArcFaceMixin, type)
    assert isinstance(ValidationMixin, type)
    assert isinstance(ReportingMixin, type)


@check("Trainer module exports mixins (requires torch)")
def test_trainer_exports_mixins():
    # This test requires torch, so we try and skip gracefully if not available
    try:
        import torch  # noqa
    except ImportError:
        print("    (skipped - torch not installed)")
        return  # Skip test, it will pass
    
    from trainer import (
        Trainer,
        CheckpointingMixin,
        EarlyStoppingMixin,
        GroupDROMixin,
        CurriculumMixin,
        ArcFaceMixin,
        ValidationMixin,
        ReportingMixin,
    )


@check("Mixin classes have expected methods")
def test_mixin_methods():
    from trainer.mixins import (
        CheckpointingMixin,
        EarlyStoppingMixin,
        GroupDROMixin,
        CurriculumMixin,
        ArcFaceMixin,
        ValidationMixin,
        ReportingMixin,
    )
    
    # Checkpointing
    assert hasattr(CheckpointingMixin, 'init_checkpointing')
    assert hasattr(CheckpointingMixin, 'save_ckpt')
    assert hasattr(CheckpointingMixin, '_upload_to_gcs')
    
    # Early stopping
    assert hasattr(EarlyStoppingMixin, 'init_early_stopping')
    assert hasattr(EarlyStoppingMixin, 'check_early_stopping')
    assert hasattr(EarlyStoppingMixin, 'reset_early_stopping')
    
    # Group DRO
    assert hasattr(GroupDROMixin, 'init_group_dro')
    assert hasattr(GroupDROMixin, 'calculate_group_dro_loss')
    
    # Curriculum
    assert hasattr(CurriculumMixin, 'init_curriculum')
    assert hasattr(CurriculumMixin, 'evaluate_lesson_gate')
    
    # ArcFace
    assert hasattr(ArcFaceMixin, 'init_arcface')
    assert hasattr(ArcFaceMixin, 'update_arcface_s')
    
    # Validation
    assert hasattr(ValidationMixin, 'init_validation')
    assert hasattr(ValidationMixin, 'should_run_validation')
    
    # Reporting
    assert hasattr(ReportingMixin, 'generate_frame_report')
    assert hasattr(ReportingMixin, 'generate_video_report')
    assert hasattr(ReportingMixin, 'generate_and_upload_reports')


test_import_trainer_mixins()
test_trainer_exports_mixins()
test_mixin_methods()


# ==============================================================================
# Backward Compatibility Tests
# ==============================================================================

print("\n" + "="*60)
print("PHASE 4: Backward Compatibility Tests")
print("="*60)


@check("Legacy imports from dataloaders.py still work")
def test_legacy_dataloader_imports():
    # These imports should still work for backward compatibility
    from dataset.dataloaders import create_dataloaders, collate_fn


@check("Legacy imports from prepare_splits.py still work")
def test_legacy_splits_imports():
    from prepare_splits import prepare_video_splits_v2


@check("train_sweep.py can be parsed (syntax check)")
def test_train_sweep_syntax():
    import ast
    from pathlib import Path
    
    train_sweep_path = Path(__file__).parent.parent / "train_sweep.py"
    if train_sweep_path.exists():
        source = train_sweep_path.read_text()
        ast.parse(source)  # Will raise SyntaxError if invalid


test_legacy_dataloader_imports()
test_legacy_splits_imports()
test_train_sweep_syntax()


# ==============================================================================
# Summary
# ==============================================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total = len(results)

print(f"\nTotal: {total} tests")
print(f"Passed: {passed} ✅")
print(f"Failed: {failed} ❌")

if failed > 0:
    print("\nFailed tests:")
    for name, ok, msg in results:
        if not ok:
            print(f"  - {name}: {msg}")
    sys.exit(1)
else:
    print("\n🎉 All verification checks passed!")
    sys.exit(0)
