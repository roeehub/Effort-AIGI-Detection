# Testing Plan

> **Purpose:** Define how to verify the refactoring doesn't break anything  
> **Goal:** Confidence that each change works before moving to the next

---

## Testing Philosophy

1. **Test incrementally** - After each small change, not just at the end
2. **Test at multiple levels** - Unit → Integration → End-to-end
3. **Preserve behavior** - Refactoring should not change outputs
4. **Fast feedback** - Quick tests for development, thorough tests before merge

---

## Test Categories

### Level 1: Unit Tests

Test individual functions/classes in isolation.

**Location:** `training/tests/unit/`

**What to test:**
- Config parsing and validation
- Augmentation pipelines produce valid outputs
- Data transforms work correctly
- Utility functions behave as expected

**Example:**
```python
# tests/unit/test_augmentations.py
import pytest
import numpy as np
from training.data.augmentations import get_pipeline

class TestAugmentationRegistry:
    def test_v3_pipeline_exists(self):
        pipeline = get_pipeline(version=3)
        assert pipeline is not None
    
    def test_v3_produces_valid_output(self):
        pipeline = get_pipeline(version=3)
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = pipeline(image=img)['image']
        
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255
    
    def test_unknown_version_raises(self):
        with pytest.raises(ValueError, match="Unknown augmentation"):
            get_pipeline(version=999)
    
    def test_surgical_requires_properties(self):
        with pytest.raises(ValueError, match="requires frame_properties"):
            get_pipeline(version='surgical')  # No properties provided

# tests/unit/test_config.py
class TestConfigValidation:
    def test_valid_config_passes(self):
        config = load_config('tests/fixtures/valid_config.yaml')
        assert config.model.name == 'effort'
    
    def test_missing_required_field_fails(self):
        with pytest.raises(ConfigurationError, match="data.gcs_bucket is required"):
            load_config('tests/fixtures/missing_bucket_config.yaml')
    
    def test_invalid_strategy_fails(self):
        with pytest.raises(ConfigurationError, match="Unknown batching strategy"):
            load_config('tests/fixtures/invalid_strategy_config.yaml')
```

**Run:** `pytest tests/unit/ -v`

---

### Level 2: Integration Tests

Test components working together.

**Location:** `training/tests/integration/`

**What to test:**
- Config → Data pipeline → Loader creation
- Data loading → Augmentation → Batching
- Model forward/backward pass with real batches

**Example:**
```python
# tests/integration/test_data_pipeline.py
import pytest
from training.config import load_config
from training.data.pipeline import create_pipeline

class TestDataPipeline:
    @pytest.fixture
    def config(self):
        return load_config('tests/fixtures/integration_config.yaml')
    
    def test_pipeline_creates_loaders(self, config):
        pipeline = create_pipeline(config)
        
        assert pipeline.train_loader is not None
        assert len(pipeline.train_loader) > 0
    
    def test_batch_has_correct_shape(self, config):
        pipeline = create_pipeline(config)
        batch = next(iter(pipeline.train_loader))
        
        expected_shape = (config.data.batching.batch_size, 3, 224, 224)
        assert batch['image'].shape == expected_shape
    
    def test_batch_has_balanced_labels(self, config):
        pipeline = create_pipeline(config)
        batch = next(iter(pipeline.train_loader))
        
        labels = batch['label'].numpy()
        real_count = (labels == 0).sum()
        fake_count = (labels == 1).sum()
        
        # Should be roughly balanced
        assert abs(real_count - fake_count) < config.data.batching.batch_size * 0.3

# tests/integration/test_training_step.py
class TestTrainingStep:
    @pytest.fixture
    def setup(self):
        config = load_config('tests/fixtures/integration_config.yaml')
        model = create_model(config)
        optimizer = create_optimizer(model, config)
        return config, model, optimizer
    
    def test_forward_pass(self, setup):
        config, model, optimizer = setup
        pipeline = create_pipeline(config)
        batch = next(iter(pipeline.train_loader))
        
        output = model(batch['image'])
        assert output.shape == (config.data.batching.batch_size, 2)
    
    def test_backward_pass(self, setup):
        config, model, optimizer = setup
        pipeline = create_pipeline(config)
        batch = next(iter(pipeline.train_loader))
        
        output = model(batch['image'])
        loss = F.cross_entropy(output, batch['label'])
        
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
```

**Run:** `pytest tests/integration/ -v`

---

### Level 3: End-to-End Tests

Test the full training pipeline.

**Location:** `training/tests/e2e/`

**What to test:**
- Full training run (few epochs, small data)
- Checkpoint save/load cycle
- W&B logging (mock)
- Validation metrics computed

**Example:**
```python
# tests/e2e/test_full_training.py
import pytest
import tempfile
from pathlib import Path
from training.train import run_training

class TestFullTraining:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)
    
    def test_training_runs_without_error(self, temp_dir):
        """Smoke test: training should complete without crashing."""
        config_overrides = {
            'training.epochs': 2,
            'data.splitting.data_subset_percentage': 0.01,
            'checkpointing.output_dir': str(temp_dir),
            'logging.wandb_mode': 'disabled',
        }
        
        result = run_training(
            config_path='tests/fixtures/e2e_config.yaml',
            overrides=config_overrides
        )
        
        assert result.completed
        assert result.final_epoch == 2
    
    def test_checkpoint_saved(self, temp_dir):
        """Verify checkpoint is saved during training."""
        config_overrides = {
            'training.epochs': 1,
            'data.splitting.data_subset_percentage': 0.01,
            'checkpointing.output_dir': str(temp_dir),
        }
        
        run_training(
            config_path='tests/fixtures/e2e_config.yaml',
            overrides=config_overrides
        )
        
        checkpoints = list(temp_dir.glob('*.pth'))
        assert len(checkpoints) > 0
    
    def test_metrics_improve(self, temp_dir):
        """Verify model learns something (metrics improve)."""
        config_overrides = {
            'training.epochs': 5,
            'data.splitting.data_subset_percentage': 0.05,
            'checkpointing.output_dir': str(temp_dir),
        }
        
        result = run_training(
            config_path='tests/fixtures/e2e_config.yaml',
            overrides=config_overrides
        )
        
        # AUC should be better than random (0.5)
        assert result.best_val_auc > 0.55
```

**Run:** `pytest tests/e2e/ -v --slow` (mark as slow, run separately)

---

### Level 4: Regression Tests

Compare outputs between old and new code.

**Location:** `training/tests/regression/`

**What to test:**
- Same config produces same batches (with fixed seed)
- Same training produces similar loss curves
- Model outputs match for same inputs

**Example:**
```python
# tests/regression/test_batch_consistency.py
import pytest
import torch
from training.data.pipeline import create_pipeline
from training.config import load_config

class TestBatchConsistency:
    """Verify refactored code produces same batches as original."""
    
    @pytest.fixture
    def baseline_batch(self):
        """Pre-computed batch from original code (saved to file)."""
        return torch.load('tests/fixtures/baseline_batch.pt')
    
    def test_batch_matches_baseline(self, baseline_batch):
        # Use same config and seed
        config = load_config('tests/fixtures/regression_config.yaml')
        torch.manual_seed(42)
        
        pipeline = create_pipeline(config)
        new_batch = next(iter(pipeline.train_loader))
        
        # Images should be identical (same seed, same transforms)
        torch.testing.assert_close(
            new_batch['image'], 
            baseline_batch['image'],
            atol=1e-5, rtol=1e-5
        )
        
        # Labels should match
        assert torch.equal(new_batch['label'], baseline_batch['label'])
```

---

## Test Fixtures

### Configuration Fixtures

```yaml
# tests/fixtures/valid_config.yaml
model:
  name: effort
  backbone: openai/clip-vit-large-patch14

data:
  gcs_bucket: df40-frames-recropped-rfa85
  splitting:
    seed: 42
    data_subset_percentage: 0.01  # Tiny subset for tests
  batching:
    strategy: video_level
    batch_size: 4
    frames_per_video: 4

training:
  epochs: 1
  learning_rate: 0.0001
```

### Data Fixtures

For fast testing without GCS access:

```python
# tests/conftest.py
import pytest
import numpy as np
from PIL import Image
from pathlib import Path

@pytest.fixture(scope="session")
def mock_frames(tmp_path_factory):
    """Create mock frame images for testing."""
    frames_dir = tmp_path_factory.mktemp("frames")
    
    for label in ['real', 'fake']:
        for method in ['method1', 'method2']:
            method_dir = frames_dir / label / method / 'video_001'
            method_dir.mkdir(parents=True)
            
            for i in range(10):
                img = Image.fromarray(
                    np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                )
                img.save(method_dir / f'frame_{i:03d}.png')
    
    return frames_dir

@pytest.fixture
def mock_gcs(monkeypatch, mock_frames):
    """Mock GCS to use local files."""
    def mock_open(path, mode='rb'):
        local_path = str(mock_frames / path.replace('gs://bucket/', ''))
        return open(local_path, mode)
    
    monkeypatch.setattr('fsspec.open', mock_open)
```

---

## Checkpoint Tests (Per-Phase)

After each refactoring phase, run these standardized tests:

### Phase 1 Checkpoint (Augmentation Extraction)

```bash
# 1. Unit tests for new augmentation module
pytest tests/unit/test_augmentations.py -v

# 2. Integration: verify augmentation produces same results
pytest tests/regression/test_augmentation_consistency.py -v

# 3. Quick training run
python train.py \
    --config tests/fixtures/checkpoint_config.yaml \
    --data.splitting.data_subset_percentage 0.01 \
    --training.epochs 2 \
    --dry-run
```

### Phase 2 Checkpoint (Config System)

```bash
# 1. Unit tests for config loading
pytest tests/unit/test_config.py -v

# 2. Integration: verify config produces same runtime values
pytest tests/regression/test_config_consistency.py -v

# 3. Full training with new config system
python train.py \
    --config tests/fixtures/checkpoint_config.yaml \
    --data.splitting.data_subset_percentage 0.01 \
    --training.epochs 2
```

### Phase 3 Checkpoint (Data Pipeline)

```bash
# 1. Unit tests for splitters and batchers
pytest tests/unit/test_splitting.py tests/unit/test_batching.py -v

# 2. Integration: verify batches match baseline
pytest tests/regression/test_batch_consistency.py -v

# 3. Full training with new data pipeline
python train.py \
    --config tests/fixtures/checkpoint_config.yaml \
    --data.splitting.data_subset_percentage 0.05 \
    --training.epochs 3
```

### Phase 4 Checkpoint (Trainer)

```bash
# 1. Unit tests for trainer mixins
pytest tests/unit/test_trainer_mixins.py -v

# 2. Integration: verify training step produces same loss
pytest tests/regression/test_training_consistency.py -v

# 3. Full training with checkpointing
python train.py \
    --config tests/fixtures/checkpoint_config.yaml \
    --data.splitting.data_subset_percentage 0.05 \
    --training.epochs 5
```

### Final Checkpoint

```bash
# Full end-to-end test suite
pytest tests/ -v --slow

# Full training run matching production config
python train.py \
    --config config/production.yaml \
    --data.splitting.data_subset_percentage 0.2 \
    --training.epochs 10
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements-test.txt
      - run: pytest tests/unit/ -v
  
  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements-test.txt
      - run: pytest tests/integration/ -v
  
  # E2E tests only on main branch (expensive)
  e2e-tests:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    needs: integration-tests
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements-test.txt
      - run: pytest tests/e2e/ -v --slow
```

---

## Test Data Management

### Baseline Generation

```python
# scripts/generate_test_baselines.py
"""
Run this script to generate baseline data for regression tests.
Should be run BEFORE refactoring to capture current behavior.
"""
import torch
from training.data.pipeline import create_pipeline
from training.config import load_config

def generate_batch_baseline():
    config = load_config('tests/fixtures/regression_config.yaml')
    torch.manual_seed(42)
    
    pipeline = create_pipeline(config)
    batch = next(iter(pipeline.train_loader))
    
    torch.save(batch, 'tests/fixtures/baseline_batch.pt')
    print(f"Saved baseline batch with shape {batch['image'].shape}")

if __name__ == '__main__':
    generate_batch_baseline()
```

### Test Requirements

```
# requirements-test.txt
pytest>=7.0
pytest-cov
pytest-timeout
torch  # for tensor comparisons
Pillow  # for image fixtures
```

---

## Quick Test Commands

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_augmentations.py -v

# Run with coverage
pytest tests/ --cov=training --cov-report=html

# Run only fast tests (exclude slow marker)
pytest tests/ -v -m "not slow"

# Run with debug output
pytest tests/ -v -s --log-cli-level=DEBUG

# Run specific test by name
pytest tests/ -v -k "test_v3_pipeline"
```
