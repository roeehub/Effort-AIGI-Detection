# Hyperparameters Reference

This document catalogs all configurable training parameters, their defaults, and effects.

## 1. Optimizer & Learning Rate

### Core Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | `1e-4` | Base learning rate |
| `weight_decay` | float | `0.05` | L2 regularization strength |
| `optimizer_eps` | float | `1e-8` | Adam epsilon for numerical stability |

### Learning Rate Schedule
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr_scheduler` | str | `cosine_with_warmup` | Scheduler type |
| `total_training_steps` | int | `7000` | Total steps for scheduler calculation |
| `lr_scheduler_warmup_steps` | int | `700` | Linear warmup steps (typically 10%) |

**Scheduler Options:**
- `cosine_with_warmup`: Linear warmup → cosine decay (most common)
- `linear`: Linear decay
- `constant`: No decay after warmup

### Gradient Control
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gradient_clip_val` | float | `1.0` | Max gradient norm (0 = no clipping) |

**When to use gradient clipping:**
- If you see `NaN` losses
- If `train/grad_norm` spikes to very high values (>100)
- Typical value: `1.0` to `10.0`

**Note (Jan 4, 2026):** All DeepLive experiment configs now include `gradient_clip_val: 1.0` 
to address gradient spikes observed in B16 training.

## 2. SVD Residual (EFFORT Method)

### Core SVD Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rank` | int | `1023` | Number of top singular values to keep frozen |
| `lambda_reg` | float | `1.0` | Regularization weight for orthogonality/norm losses |

**How `rank` works:**
- For a weight matrix W ∈ ℝ^{m×n}, SVD gives W = UΣV^T
- We keep top `rank` singular values as frozen `weight_main`
- Remaining singular values/vectors form the trainable residual
- **Rule of thumb**: `rank = hidden_size - 1` (train 1 singular value)

**Effect of `lambda_reg`:**
- Controls strength of orthogonality and norm-preserving losses
- Higher values → more constrained residual, slower divergence from pretrained
- Lower values → more freedom to adapt, risk of overfitting

### Backbone-Specific Ranks
| Backbone | Hidden Size | Recommended Rank |
|----------|-------------|------------------|
| ViT-L-14 (OpenAI) | 1024 | 1023 |
| ViT-B-16 (OpenAI) | 768 | 767 |
| ViT-B-32 (OpenAI) | 768 | 767 |
| ViT-B-16 (LAION) | 512* | 511 |

*Note: LAION ViT-B-16 projects 768 internal dim to 512 output dim.

## 3. ArcFace Head

### Enable/Disable
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_arcface_head` | bool | `True` | Use ArcFace margin-based head |

### ArcFace Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arcface_s` | float | `30.0` | Scale factor (temperature) |
| `arcface_m` | float | `0.28` | Angular margin (radians) |

### Scale Annealing
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `s_start` | float | `10.0` | Initial scale factor |
| `s_end` | float | `30.0` | Final scale factor |
| `anneal_steps` | int | `2000` | Steps to anneal s from start to end |

**Why anneal `s`?**
- Low `s` at start → softer probabilities, easier gradients
- High `s` at end → sharper predictions, better discrimination
- Helps avoid mode collapse in early training

## 4. Loss Function

### Loss Selection
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_focal_loss` | bool | `False` | Use Focal Loss instead of CE |
| `focal_loss_gamma` | float | `2.0` | Focal loss focusing parameter |
| `focal_loss_alpha` | float | `None` | Class weighting (None = no weighting) |

**Note:** When `use_arcface_head=True`, CrossEntropyLoss is always used (the margin provides similar effect to Focal Loss).

## 5. Training Schedule

### Duration Control
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nEpochs` | int | `50` | Maximum epochs |
| `max_train_steps` | int | `None` | Maximum steps (overrides epochs) |
| `evaluate_every_steps` | int | `500` | Validation frequency (steps) |

### Early Stopping
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `early_stopping_enabled` | bool | `True` | Enable early stopping |
| `early_stopping_patience` | int | `15` | Evals without improvement before stop |
| `early_stopping_min_delta` | float | `0.001` | Minimum improvement threshold |

## 6. Dataloader Configuration

### Strategy Selection
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataloader_strategy` | str | `deeplive` | Batching strategy |

**Strategies:**
- `deeplive`: Paired real/fake from DeepLive GCS bucket
- `frame_level`: Random frame sampling
- `video_level`: Sample frames from selected videos
- `property_balancing`: Balance by frame properties
- `per_method`: Balance across manipulation methods

### Batch Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frames_per_batch` | int | `32` | GPU batch size (frames) |
| `frames_per_video` | int | `8` | Frames to sample per video |
| `videos_per_batch` | int | `64` | Videos per batch (video_level) |
| `num_workers` | int | `4` | DataLoader workers |
| `prefetch_factor` | int | `2` | Batches to prefetch per worker |

### DeepLive-Specific
```yaml
deeplive:
  gcs_bucket: "live-deepfake-methods-real-and-fake-frames"
  sampling_mode: "sparse"  # sparse | full | pairs
  anchor_indices: [0, 2, 4, 6, 8, 10, 12, 14]  # 8 frames from 16
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  seed: 737
```

## 7. Augmentation

### Version Selection
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `augmentation_version` | int | `7` | Augmentation pipeline version |

### Version Overview
| Version | Description | Use Case |
|---------|-------------|----------|
| 3 | Legacy revised pipeline | Backward compatibility |
| 4 | Generalist pipeline | Balanced augmentation |
| 5 | Medium augmentation | General purpose |
| 6 | Social media simulator | Domain-specific |
| 7 | Landmark occlusion | Face-part robustness |

### Landmark Occlusion (V7)
```yaml
augmentation:
  version: "landmark_occlusion"
  occlusion_type: "mixed"  # solid | blur | pixelate | mixed
  regions:
    - "left_eye"
    - "right_eye"
    - "nose"
    - "mouth"
  num_regions: [1, 2]  # Occlude 1-2 regions
  occlusion_prob: 0.2  # 20% probability
  landmark_format: "mediapipe"
```

## 8. Backbone Configuration

### Structure
```yaml
backbone:
  name: "vit_b_16_openai"         # Identifier
  variant: "ViT-B-16"              # Model variant
  source: "openai"                 # openai | laion
  hidden_size: 768                 # Output feature dimension
  resolution: 224                  # Input resolution
  
  # For OpenAI (HuggingFace)
  huggingface_id: "openai/clip-vit-base-patch16"
  
  # For LAION (OpenCLIP)
  openclip_model: "ViT-B-16"
  openclip_pretrained: "datacomp_xl_s13b_b90k"
```

## 9. Checkpointing

```yaml
checkpointing:
  gcs_prefix: "gs://training-job-outputs/experiments/"
  save_every_steps: 1000   # Checkpoint frequency
  keep_last_n: 3           # Top-N checkpoints to keep
```

## 10. W&B Configuration

```yaml
wandb:
  project: "deeplive-experiments"
  tags:
    - "deeplive"
    - "vit-b-16"
    - "openai"
```

---

## Parameter Tuning Guidelines

### If model is not learning (stuck at 50% accuracy):
1. ✅ First check: SVD residual properly applied (run `test_svd_residual.py`)
2. Check `train/grad_norm` is non-zero
3. Check `train/logits/diff_std` is non-zero
4. If all zeros: backbone may be frozen incorrectly

### If model collapses (predicts all one class):
1. Lower `arcface_s` or use annealing
2. Reduce `learning_rate`
3. Check class balance in data

### If overfitting:
1. Increase `weight_decay`
2. Increase `lambda_reg`
3. Enable more augmentation
4. Use early stopping with patience

### If underfitting:
1. Increase `learning_rate`
2. Decrease `rank` (more trainable parameters)
3. Train longer (more steps/epochs)
4. Reduce augmentation strength

---

*See also: [03_BACKBONES.md](03_BACKBONES.md) for backbone details*
