# Docker Build Caching - Fast Builds Guide

## ✅ Solution Implemented: Pre-built Base Image

We've implemented **Option B** from the original troubleshooting - a pre-built base image approach that **guarantees fast builds (~30 seconds)**.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Dockerfile.base (built ONCE, ~10 min)                          │
│  - CUDA 12.1 + cuDNN 8                                          │
│  - Python 3.10                                                   │
│  - All pip dependencies (PyTorch, requirements.txt)             │
│  → us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector-base:latest │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Dockerfile / Dockerfile.dev (built every time, ~30 sec)        │
│  - Just copies your Python code                                  │
│  → effort-detector-training:latest / effort-detector-test:latest│
└─────────────────────────────────────────────────────────────────┘
```

### Quick Start

**Step 1: Build the base image (one-time, or when requirements.txt changes)**
```bash
cd DeepfakeBench/training
gcloud builds submit --config cloudbuild_base_image.yaml --project train-cvit2
```

**Step 2: Run your tests/training (FAST - ~30 seconds)**
```bash
# Tests
gcloud builds submit --config cloudbuild_test.yaml --project train-cvit2

# Training
gcloud builds submit --config cloudbuild_train_deeplive.yaml \
  --substitutions=_EXPERIMENT=deeplive_vit_B16 \
  --project train-cvit2
```

### When to Rebuild the Base Image

Only rebuild when these change:
- `requirements.txt` - new Python dependencies
- `Dockerfile.base` - system packages or PyTorch version

**You do NOT need to rebuild when:**
- Changing any `.py` files
- Updating experiment configs
- Modifying shell scripts

### Files

| File | Purpose | Build Time |
|------|---------|------------|
| `Dockerfile.base` | All dependencies (CUDA, Python, pip) | ~10 min (first time) |
| `Dockerfile` | Production training container | ~30 sec |
| `Dockerfile.dev` | Development/test container | ~30 sec |
| `cloudbuild_base_image.yaml` | Builds base image | Run manually |
| `cloudbuild_train_deeplive.yaml` | Training jobs | Fast! |
| `cloudbuild_test.yaml` | E2E tests | Fast! |

---

## Legacy: Original Troubleshooting Notes

<details>
<summary>Click to expand original troubleshooting (for reference)</summary>

### Problem
Kaniko cache is not working as expected - builds still take 10+ minutes even after the first build should have populated the cache.

### Root Causes Found

1. **Different Dockerfiles** - `cloudbuild_test.yaml` used `Dockerfile.dev`, `cloudbuild_train_deeplive.yaml` used `Dockerfile`. Each had separate caches.

2. **`COPY . .` invalidates cache** - Any code change invalidated everything after the COPY layer, forcing pip install to re-run.

3. **Cache retrieval overhead** - Even with cache hits, Kaniko still takes time to check and retrieve layers.

### Solution: Pre-built Base Image

Instead of relying on Kaniko's layer caching, we:
1. Pre-build a base image with ALL dependencies
2. Main Dockerfiles just `FROM base` and copy code
3. Result: **30 second builds** instead of 10+ minutes

</details>
