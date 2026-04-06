# Training Flow

This document describes the end-to-end training pipeline from launch to model output.

## 1. Launch Sequence

### 1.1 Local Development
```
./dev.sh build-prod -y          # Build Docker image via Cloud Build
./launch_experiment.sh <args>    # Submit Vertex AI training job
```

### 1.2 Production (GCP Vertex AI)
```
launch_experiment.sh
    ↓
launch_experiment_jobs.sh       # Creates Vertex AI custom job
    ↓
entrypoint.sh                   # Container entrypoint
    ↓
train_sweep.py                  # Main training script
```

## 2. Initialization Phase

### 2.1 Configuration Loading
```python
# train_sweep.py / train_deeplive.py
config = load_config(param_config_path)  # Load YAML
config = apply_defaults(config)           # Merge with defaults
wandb.init(config=config)                 # Initialize W&B
```

### 2.2 Model Construction
```python
# detectors/effort_detector.py
class EffortDetector(nn.Module):
    def __init__(self, config):
        self.backbone = self.build_backbone(config)  # CLIP vision encoder
        self.head = ArcMarginProduct(...) or nn.Linear(...)  # Classification head
        self.loss_func = CrossEntropyLoss() or FocalLoss()
```

### 2.3 Backbone Loading
The backbone loading path depends on the `source` field:

```
backbone.source == 'openai'
    → _build_huggingface_backbone()
    → CLIPModel.from_pretrained('openai/clip-vit-*')
    → apply_svd_residual_to_self_attn()

backbone.source == 'laion' or 'openclip'
    → _build_openclip_backbone()
    → open_clip.create_model_and_transforms()
    → apply_svd_residual_to_openclip_attn()
```

### 2.4 SVD Residual Application
For each attention layer's `out_proj`:
1. Perform SVD: `U, S, Vh = svd(weight)`
2. Keep top-r singular values as frozen `weight_main`
3. Initialize trainable `U_residual`, `S_residual`, `V_residual`
4. Effective weight = `weight_main + U_residual @ diag(S_residual) @ V_residual`

## 3. Data Loading Phase

### 3.1 Dataset Discovery (DeepLive)
```python
# dataset/deeplive_dataset.py
dataset = DeepLiveDataset(bucket_name="live-deepfake-methods-real-and-fake-frames")
samples = dataset.discover_samples()  # Scan GCS for manifest.json files
```

### 3.2 Train/Val/Test Split
```python
# Typically 80/10/10 split controlled by:
deeplive:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  seed: 737  # For reproducibility
```

### 3.3 Dataloader Strategy
The `dataloader_strategy` config determines batching behavior:

| Strategy | Description |
|----------|-------------|
| `deeplive` | Paired real/fake frames from DeepLive dataset |
| `frame_level` | Sample individual frames across all videos |
| `video_level` | Sample frames within selected videos |
| `property_balancing` | Balance by frame properties (sharpness buckets) |
| `per_method` | Balance across manipulation methods |

## 4. Training Loop

### 4.1 High-Level Flow
```python
# trainer/trainer.py
for epoch in range(nEpochs):
    for batch in train_loader:
        # 1. Forward pass
        predictions = model(batch)
        
        # 2. Loss computation
        losses = model.get_losses(batch, predictions)
        
        # 3. Backward pass
        scaler.scale(losses['overall']).backward()
        
        # 4. Optimizer step
        scaler.step(optimizer)
        scheduler.step()
        
        # 5. Logging
        wandb.log(metrics)
        
        # 6. Periodic evaluation
        if step % evaluate_every_steps == 0:
            run_validation()
```

### 4.2 Forward Pass Details
```python
# detectors/effort_detector.py :: forward()
def forward(self, data_dict, inference=False):
    image = data_dict['image']  # [B, C, H, W] or [B, T, C, H, W]
    
    # Handle video tensors
    if is_video:
        image = image.view(B * T, C, H, W)
        label = label.repeat_interleave(T)
    
    # Extract features through CLIP backbone
    features = self.backbone(image)['pooler_output']  # [B, hidden_size]
    
    # Classification head
    if self.use_arcface_head:
        logits = self.head(features, label)  # Margin-penalized logits
    else:
        logits = self.head(features)  # Standard linear
    
    return {'cls': logits, 'prob': softmax(logits), 'feat': features}
```

### 4.3 Loss Computation
```python
# Standard loss
loss = CrossEntropyLoss(predictions, labels)

# Plus regularization (SVD residual)
for module in SVDResidualLinear modules:
    loss += lambda_reg * module.compute_orthogonal_loss()
    loss += lambda_reg * module.compute_keepsv_loss()
```

### 4.4 ArcFace Head (if enabled)
```python
# Margin-based loss for better discrimination
cosine = F.linear(F.normalize(features), F.normalize(weight))
theta = acos(cosine)
margin_theta = theta + m  # Add angular margin to target class
logits = s * cos(margin_theta)  # Scale factor s
```

### 4.5 Learning Rate Schedule
```python
# cosine_with_warmup (typical)
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)  # Linear warmup
else:
    lr = base_lr * cosine_decay(step)     # Cosine decay
```

## 5. Validation Phase

### 5.1 Evaluation Triggers
- Step-based: `evaluate_every_steps` (e.g., every 500 steps)
- Epoch-based: `evaluation_frequency` (e.g., 2x per epoch)
- Max steps: Final evaluation before stopping

### 5.2 Metrics Computed
```python
# Per validation set (in-dist, holdout, OOD)
metrics = {
    'auc': sklearn.metrics.roc_auc_score(),
    'eer': equal_error_rate(),
    'accuracy': (pred == label).mean(),
    'ap': average_precision_score(),
}
```

### 5.3 Checkpointing
- Top-N checkpoints kept based on validation AUC
- Uploaded to GCS: `gs://training-job-outputs/{run_id}/`

## 6. Early Stopping & Lesson Gates

### 6.1 Standard Early Stopping
```yaml
early_stopping_enabled: true
early_stopping_patience: 15    # Stop after N evals without improvement
early_stopping_min_delta: 0.001
```

### 6.2 Lesson Gate (Curriculum Learning)
```yaml
lesson_gate:
  enabled: true
  checks:
    - metric: auc
      dataset: val_holdout
      comparison: ge
      threshold: 0.85
  plateau:
    enabled: true
    patience: 2
    min_delta: 0.001
```

## 7. Key Files Reference

| File | Responsibility |
|------|---------------|
| `train_sweep.py` | Entry point for training jobs |
| `trainer/trainer.py` | Training loop, validation, checkpointing |
| `detectors/effort_detector.py` | Model architecture, SVD residual |
| `dataset/deeplive_dataset.py` | DeepLive dataset loading |
| `dataset/dataloaders.py` | Batching strategies, augmentation |
| `data/augmentations/` | Augmentation pipelines |
| `data/batching/` | Dataloader factories |

---

*See also: [02_HYPERPARAMETERS.md](02_HYPERPARAMETERS.md) for parameter details*
