# Logging & Error Handling Conventions

> **Purpose:** Establish consistent logging and error handling across the training codebase  
> **Goal:** Full visibility into what's happening during training, with actionable error messages

---

## Core Principles

### 1. **Structured Logging**
Every log message should answer: **What happened? Where? Why does it matter?**

### 2. **Log Levels Used Consistently**
- `DEBUG` - Detailed info for debugging (batch contents, tensor shapes)
- `INFO` - Normal operation milestones (epoch start, checkpoint saved)
- `WARNING` - Something unexpected but recoverable (missing optional config)
- `ERROR` - Something failed but training can continue (single batch failed)
- `CRITICAL` - Training must stop (model cannot be loaded)

### 3. **Fail Fast with Context**
When something goes wrong, fail immediately with a clear message explaining:
- What we were trying to do
- What went wrong
- How to fix it

### 4. **Progress Visibility**
Long operations should show progress (tqdm, periodic logs)

---

## Logger Setup

### Standard Logger Configuration

```python
# utils/logging.py
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True
) -> logging.Logger:
    """
    Create a logger with consistent formatting.
    
    Args:
        name: Logger name (usually module name)
        log_dir: Directory for log files
        level: Minimum log level
        console: Whether to log to console
        file: Whether to log to file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Format: timestamp | level | module | message
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(
            log_dir / f'training_{timestamp}.log'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Module-level logger creation pattern
def get_logger(name: str) -> logging.Logger:
    """Get or create a logger for a module."""
    return logging.getLogger(f'training.{name}')
```

### Per-Module Usage

```python
# Example: data/batching/property_balanced.py
from utils.logging import get_logger

log = get_logger('batching.property_balanced')

class PropertyBalancedStrategy:
    def create_train_loader(self, train_data, config):
        log.info(f"Creating property-balanced loader with {len(train_data):,} frames")
        log.debug(f"Config: batch_size={config.batch_size}, workers={config.num_workers}")
        
        # ... implementation ...
        
        log.info(f"Loader created: {len(loader)} batches per epoch")
        return loader
```

---

## Logging Standards by Component

### Configuration Loading

```python
# config/loader.py
log = get_logger('config')

def load_config(yaml_path: str, overrides: dict = None) -> TrainingConfig:
    log.info(f"Loading configuration from: {yaml_path}")
    
    if not Path(yaml_path).exists():
        log.error(f"Config file not found: {yaml_path}")
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    log.debug(f"Loaded {len(raw)} top-level config keys")
    
    if overrides:
        log.info(f"Applying {len(overrides)} config overrides")
        for key, value in overrides.items():
            log.debug(f"  Override: {key} = {value}")
    
    config = _parse_config(raw, overrides)
    log.info(f"Configuration loaded successfully")
    log.info(f"  Model: {config.model.name}")
    log.info(f"  Epochs: {config.training.epochs}")
    log.info(f"  Batch size: {config.data.batching.batch_size}")
    
    return config
```

### Data Pipeline

```python
# data/pipeline.py
log = get_logger('data.pipeline')

def create_pipeline(config: DataConfig) -> DataPipeline:
    log.info("=" * 60)
    log.info("INITIALIZING DATA PIPELINE")
    log.info("=" * 60)
    
    # Splitting
    log.info(f"Splitting strategy: {config.splitting.strategy}")
    splitter = get_splitter(config.splitting.strategy)
    train_data, val_in_dist, val_holdout = splitter.split(config)
    
    log.info("Split results:")
    log.info(f"  Training:       {len(train_data):>8,} samples")
    log.info(f"  Val (in-dist):  {len(val_in_dist):>8,} videos")
    log.info(f"  Val (holdout):  {len(val_holdout):>8,} videos")
    
    # Method distribution
    if hasattr(train_data[0], 'method'):
        methods = Counter(d.method for d in train_data)
        log.info("Training method distribution:")
        for method, count in sorted(methods.items(), key=lambda x: -x[1]):
            log.info(f"  {method:.<30} {count:>8,} ({100*count/len(train_data):.1f}%)")
    
    # Batching
    log.info(f"Batching strategy: {config.batching.strategy}")
    batcher = get_batcher(config.batching.strategy)
    train_loader = batcher.create_train_loader(train_data, config)
    
    log.info(f"Train loader: {len(train_loader)} batches")
    log.info("=" * 60)
    
    return DataPipeline(train_loader, val_in_dist_loader, val_holdout_loader)
```

### Training Loop

```python
# trainer/base.py
log = get_logger('trainer')

class Trainer:
    def train_epoch(self, train_loader, epoch):
        log.info(f"{'='*60}")
        log.info(f"EPOCH {epoch + 1}/{self.config.training.epochs}")
        log.info(f"{'='*60}")
        
        self.model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            try:
                loss = self._train_step(batch)
                epoch_loss += loss
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss:.4f}'})
                
                # Periodic detailed logging
                if batch_idx % self.config.logging.log_interval == 0:
                    log.debug(f"Batch {batch_idx}: loss={loss:.4f}, "
                             f"lr={self.optimizer.param_groups[0]['lr']:.2e}")
                    
            except Exception as e:
                log.error(f"Error in batch {batch_idx}: {e}")
                log.error(f"Batch info: {self._describe_batch(batch)}")
                if self.config.training.fail_on_batch_error:
                    raise
                continue
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_loader)
        
        log.info(f"Epoch {epoch+1} complete:")
        log.info(f"  Average loss: {avg_loss:.4f}")
        log.info(f"  Time: {epoch_time:.1f}s ({epoch_time/len(train_loader)*1000:.1f}ms/batch)")
        log.info(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        return avg_loss
    
    def _describe_batch(self, batch: dict) -> str:
        """Generate a debug description of a batch."""
        desc = []
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                desc.append(f"{key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, list):
                desc.append(f"{key}: list[{len(value)}]")
            else:
                desc.append(f"{key}: {type(value).__name__}")
        return ", ".join(desc)
```

### Checkpoint Operations

```python
# trainer/mixins/checkpointing.py
log = get_logger('trainer.checkpoint')

class CheckpointingMixin:
    def save_checkpoint(self, epoch: int, metrics: dict) -> str:
        log.info(f"Saving checkpoint for epoch {epoch}")
        log.info(f"  Metrics: {metrics}")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'metrics': metrics,
        }
        
        local_path = self._get_checkpoint_path(epoch, metrics)
        torch.save(checkpoint, local_path)
        log.info(f"  Saved locally: {local_path}")
        
        if self.config.checkpointing.upload_to_gcs:
            gcs_path = self._upload_to_gcs(local_path)
            log.info(f"  Uploaded to GCS: {gcs_path}")
            return gcs_path
        
        return local_path
    
    def load_checkpoint(self, path: str, validate: bool = True):
        log.info(f"Loading checkpoint from: {path}")
        
        if path.startswith('gs://'):
            log.info("  Downloading from GCS...")
            path = self._download_from_gcs(path)
        
        if not Path(path).exists():
            log.error(f"Checkpoint file not found: {path}")
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        log.info(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        log.info(f"  Checkpoint metrics: {checkpoint.get('metrics', {})}")
        
        if validate:
            self._validate_checkpoint(checkpoint)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        log.info("  Model state loaded")
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            log.info("  Optimizer state loaded")
        
        log.info("Checkpoint loaded successfully")
```

### Validation

```python
# trainer/validation.py
log = get_logger('trainer.validation')

def run_validation(model, val_loader, config, prefix="val"):
    log.info(f"Running {prefix} validation on {len(val_loader)} batches")
    
    model.eval()
    all_preds, all_labels = [], []
    val_start = time.time()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"{prefix} validation"):
            preds = model(batch)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
    
    # Compute metrics
    metrics = compute_metrics(all_preds, all_labels)
    val_time = time.time() - val_start
    
    log.info(f"{prefix.upper()} Validation Results:")
    log.info(f"  AUC:      {metrics['auc']:.4f}")
    log.info(f"  EER:      {metrics['eer']:.4f}")
    log.info(f"  Accuracy: {metrics['acc']:.4f}")
    log.info(f"  Time:     {val_time:.1f}s")
    
    # Per-method breakdown if available
    if 'method' in metrics:
        log.info("  Per-method AUC:")
        for method, method_auc in sorted(metrics['method'].items()):
            log.info(f"    {method:.<30} {method_auc:.4f}")
    
    return metrics
```

---

## Error Handling Patterns

### Configuration Errors

```python
class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass

def validate_config(config: TrainingConfig) -> None:
    """Validate configuration with helpful error messages."""
    errors = []
    
    # Check required fields
    if not config.data.gcs_bucket:
        errors.append("data.gcs_bucket is required")
    
    # Check logical consistency
    if config.data.batching.strategy == 'property_balancing':
        if not config.data.property_balancing.enabled:
            errors.append(
                "property_balancing batching strategy requires "
                "data.property_balancing.enabled=True"
            )
        if not config.data.property_balancing.parquet_path:
            errors.append(
                "property_balancing requires data.property_balancing.parquet_path"
            )
    
    # Check numerical ranges
    if config.training.learning_rate <= 0:
        errors.append(f"learning_rate must be positive, got {config.training.learning_rate}")
    
    if config.data.batching.batch_size < 1:
        errors.append(f"batch_size must be >= 1, got {config.data.batching.batch_size}")
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        log.error(error_msg)
        raise ConfigurationError(error_msg)
    
    log.info("Configuration validation passed")
```

### Data Loading Errors

```python
class DataLoadingError(Exception):
    """Raised when data cannot be loaded."""
    pass

def load_frame(path: str, config: dict) -> torch.Tensor:
    """Load a single frame with comprehensive error handling."""
    try:
        with fsspec.open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
    except FileNotFoundError:
        log.error(f"Frame not found: {path}")
        raise DataLoadingError(f"Frame file not found: {path}")
    except Exception as e:
        log.error(f"Failed to open frame {path}: {e}")
        raise DataLoadingError(f"Cannot open frame {path}: {e}")
    
    try:
        img = img.resize((config['resolution'], config['resolution']), Image.BICUBIC)
    except Exception as e:
        log.error(f"Failed to resize frame {path}: {e}")
        raise DataLoadingError(f"Cannot resize frame {path}: {e}")
    
    return img

def load_frame_safe(path: str, config: dict) -> Optional[torch.Tensor]:
    """Load a frame, returning None on failure (for batch processing)."""
    try:
        return load_frame(path, config)
    except DataLoadingError as e:
        log.warning(f"Skipping frame: {e}")
        return None
```

### GCS Errors

```python
class GCSError(Exception):
    """Raised when GCS operations fail."""
    pass

def download_from_gcs(gcs_path: str, local_path: str, retries: int = 3) -> str:
    """Download from GCS with retries and clear error messages."""
    if not gcs_path.startswith('gs://'):
        raise GCSError(f"Invalid GCS path (must start with gs://): {gcs_path}")
    
    for attempt in range(retries):
        try:
            log.debug(f"GCS download attempt {attempt + 1}/{retries}: {gcs_path}")
            
            storage_client = storage.Client()
            bucket_name = gcs_path.split('/')[2]
            blob_path = '/'.join(gcs_path.split('/')[3:])
            
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            if not blob.exists():
                raise GCSError(f"GCS object not found: {gcs_path}")
            
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(local_path)
            
            log.info(f"Downloaded {gcs_path} -> {local_path}")
            return local_path
            
        except exceptions.Forbidden as e:
            log.error(f"GCS permission denied for {gcs_path}")
            log.error("Ensure service account has 'Storage Object Viewer' role")
            raise GCSError(f"Permission denied for {gcs_path}: {e}")
            
        except Exception as e:
            log.warning(f"GCS download attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                raise GCSError(f"Failed to download {gcs_path} after {retries} attempts: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
```

---

## Progress Indicators

### Long Operations

```python
# For operations with known length
def process_frames(frames: list, config: dict) -> list:
    results = []
    for frame in tqdm(frames, desc="Processing frames"):
        result = process_single_frame(frame, config)
        results.append(result)
    return results

# For operations with unknown length
def stream_from_gcs(bucket: str, prefix: str):
    log.info(f"Streaming from gs://{bucket}/{prefix}")
    count = 0
    for blob in bucket.list_blobs(prefix=prefix):
        yield blob
        count += 1
        if count % 1000 == 0:
            log.info(f"  Processed {count:,} blobs...")
    log.info(f"  Complete: {count:,} blobs total")
```

### Batch Processing with Stats

```python
def create_train_loader_with_stats(train_data, config):
    """Create loader while collecting and logging statistics."""
    log.info("Creating training data loader...")
    
    stats = {
        'total_samples': len(train_data),
        'by_label': Counter(),
        'by_method': Counter(),
        'skipped': 0,
    }
    
    valid_data = []
    for item in tqdm(train_data, desc="Validating data"):
        if is_valid(item):
            valid_data.append(item)
            stats['by_label'][item['label']] += 1
            stats['by_method'][item['method']] += 1
        else:
            stats['skipped'] += 1
    
    log.info("Data statistics:")
    log.info(f"  Total samples:   {stats['total_samples']:,}")
    log.info(f"  Valid samples:   {len(valid_data):,}")
    log.info(f"  Skipped:         {stats['skipped']:,}")
    log.info(f"  Label distribution:")
    for label, count in stats['by_label'].items():
        log.info(f"    {label}: {count:,} ({100*count/len(valid_data):.1f}%)")
    
    return create_loader(valid_data, config), stats
```

---

## W&B Integration

```python
# Structured logging to W&B
def log_to_wandb(metrics: dict, step: int, prefix: str = ""):
    """Log metrics to W&B with consistent naming."""
    wandb_metrics = {}
    for key, value in metrics.items():
        full_key = f"{prefix}/{key}" if prefix else key
        wandb_metrics[full_key] = value
    
    wandb.log(wandb_metrics, step=step)
    log.debug(f"Logged to W&B: {list(wandb_metrics.keys())}")

# Example usage
log_to_wandb({
    'loss': train_loss,
    'lr': current_lr,
}, step=global_step, prefix='train')

log_to_wandb({
    'auc': val_metrics['auc'],
    'eer': val_metrics['eer'],
}, step=global_step, prefix='val')
```

---

## Debug Mode

```python
# Enable verbose debugging
def set_debug_mode(enabled: bool = True):
    """Enable debug mode for all training loggers."""
    level = logging.DEBUG if enabled else logging.INFO
    
    for name in ['training', 'training.config', 'training.data', 'training.trainer']:
        logger = logging.getLogger(name)
        logger.setLevel(level)
    
    if enabled:
        log.info("Debug mode ENABLED - verbose logging active")
    else:
        log.info("Debug mode DISABLED - standard logging")

# Usage in config
if config.debug:
    set_debug_mode(True)
```

---

## Summary: Logging Checklist

For each new module/function:

- [ ] Import logger: `log = get_logger('module.submodule')`
- [ ] Log entry point with key parameters: `log.info(f"Starting X with {params}")`
- [ ] Log completion with results: `log.info(f"X complete: {results}")`
- [ ] Log errors with context: `log.error(f"Failed to X: {error}, input was {input}")`
- [ ] Use appropriate levels (DEBUG for details, INFO for milestones)
- [ ] Add progress indicators for long operations
- [ ] Include batch/tensor shapes in debug logs
