# Copilot Instructions for Effort-AIGI-Detection

## Project Overview
Deepfake/AIGI detection system implementing the **Effort** method (ICML 2025 Oral): SVD-based orthogonal subspace decomposition for generalizable AI-generated image detection. Trains on CLIP backbones, deploys to GCP Vertex AI.

## Architecture Overview

### Core Training Pipeline (`DeepfakeBench/training/`)
```
train_sweep.py → Trainer → EffortDetector
     ↓               ↓           ↓
Data Sources    Mixins      SVDResidualLinear
(factory)      (modular)    (core method)
```

**Key Flow:**
1. `train_sweep.py` is the entry point - loads config, initializes W&B, creates data pipeline
2. `data/sources/` factory creates data loaders based on `data_source` config (`manifest`, `deeplive`, `df40_paired`)
3. `Trainer` (composed of mixins) manages training loop, validation, checkpointing
4. `EffortDetector` wraps CLIP backbone + `SVDResidualLinear` layers for fine-tuning

### Data Sources (Factory Pattern)
Register new sources with `@register_data_source('name')` in `data/sources/`:
- `manifest` - Load from JSON manifest files
- `deeplive` - GCS bucket with paired real/fake frames + landmarks
- `df40_paired` - DF40 dataset (NO landmarks available)

### Trainer Mixins (`trainer/mixins/`)
Modular capabilities: `CheckpointingMixin`, `EarlyStoppingMixin`, `GroupDROMixin`, `CurriculumMixin`, `ArcFaceMixin`, `ValidationMixin`, `ReportingMixin`

## Key Conventions

### Config Hierarchy
1. `config/defaults.yaml` - base defaults
2. `config/detector/effort.yaml` - model-specific
3. `experiments/*.yaml` - experiment-specific (e.g., `df40_paired_vit_B16_laion.yaml`)
4. W&B sweep config - hyperparameter overrides

### Backbone Configuration
Use `config/backbone_registry.yaml` to find supported backbones:
```yaml
backbone:
  source: "openai"  # or "laion"
  variant: "ViT-L-14"  # or "ViT-B-16", "ViT-B-16-DataComp-XL"
  hidden_size: 1024
```
LAION models use `open_clip` library; OpenAI models use HuggingFace `transformers`.

### Augmentation Pipelines
Use `data/augmentations/` registry:
```python
from data.augmentations import get_pipeline
pipeline = get_pipeline(version='landmark_occlusion', occlusion_type='mixed')
```
**Important:** DF40 has NO landmarks - use `version='base_only'` for that dataset.

### Batching Strategies (`data/batching/`)
- `per_method` - Round-robin across generation methods
- `property_balanced` - Hierarchical sampling by method categories
- `video_level` / `frame_level` - Random sampling variants

## Developer Commands

### Local Development
```bash
cd DeepfakeBench/training
./dev.sh shell-prod    # Enter production container
./dev.sh train-dry-run # Verify config without model instantiation
pytest tests/          # Run unit tests (70+ tests)
```

### Launch Training on Vertex AI
```bash
./launch_experiment.sh <WANDB_PROJECT> <REGION> <PARAM_CONFIG>
# Example:
./launch_experiment.sh df40-experiments asia-southeast1 experiments/df40_paired_vit_B16_laion.yaml
```

### Docker
```bash
# Build production image (VERSION file auto-incremented)
./dev.sh build-prod

# Image naming: us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:$(cat VERSION)
```

## Code Patterns

### Adding a New Data Source
```python
# In data/sources/my_source.py
from . import register_data_source, DataPipelineResult

@register_data_source('my_source')
def create_my_pipeline(config, data_config, logger, **kwargs) -> DataPipelineResult:
    # Return DataPipelineResult with train_loader, val_in_dist_loader, etc.
```

### SVD Residual Linear (Core Method)
The `SVDResidualLinear` class in `detectors/effort_detector.py` implements Effort:
- Freezes top-r singular components (`weight_main`)
- Trains only residual components (`U_residual`, `S_residual`, `V_residual`)
- Computes orthogonal loss to maintain SVD structure

### Config Helpers Pattern
W&B overrides are applied via `utils/config_helpers.py`:
```python
from utils import apply_all_wandb_overrides, load_base_configs
config, data_config = load_base_configs(detector_path, dataloader_config_path)
apply_all_wandb_overrides(config, data_config, wandb.config, logger)
```

## Testing
```bash
pytest tests/test_config_helpers.py -v      # Config system tests
pytest tests/test_trainer_mixins.py -v      # Trainer mixin tests
pytest tests/test_deeplive_pipeline.py -v   # Data pipeline tests
```

## Important Files
| File | Purpose |
|------|---------|
| `train_sweep.py` | Main training entry point |
| `trainer/trainer.py` | Trainer class with mixin composition |
| `detectors/effort_detector.py` | Effort model + SVDResidualLinear |
| `config/defaults.yaml` | All default config values |
| `config/backbone_registry.yaml` | Supported backbone definitions |
| `experiments/*.yaml` | Experiment configurations |
| `docs/REFACTORING_STATUS.md` | Codebase architecture documentation |
| `mcp/web-server/server.py` | MCP web-tools server (fetch, search) |

## Web Access (MCP web-tools)

All agents and sub-agents have access to MCP web tools for internet access.

### Available tools
| Tool | Purpose |
|------|---------|
| `web-tools: fetch_url` | Fetch raw HTML from any URL (truncated to 20k chars) |
| `web-tools: fetch_text` | Fetch clean readable text from a URL (HTML stripped) |
| `web-tools: fetch_arxiv` | Get structured title/authors/abstract from arXiv papers |
| `web-tools: search_web` | Search DuckDuckGo and get ranked results with snippets |
| `web-tools: fetch_github_raw` | Fetch raw file content from public GitHub repos |

### When to use web access
- **Research**: finding papers, methods, or benchmarks — use `search_web` then `fetch_text` or `fetch_arxiv`
- **Documentation**: reading library docs, API references — use `fetch_text`
- **Code inspection**: reading files from external repos — use `fetch_github_raw`
- **General**: any time external knowledge is needed that is not in the workspace

### Guidelines
- ALWAYS prefer web tools over guessing when external facts are needed
- Use `fetch_arxiv` for arXiv papers (structured output) instead of `fetch_text`
- Truncate/summarize web results before passing to sub-agents to save context
- Chain `search_web` → `fetch_text` for research workflows
