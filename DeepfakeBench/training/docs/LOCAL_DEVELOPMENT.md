# Local Development Guide

This guide explains how to set up and use the local development environment for testing and debugging the training pipeline.

## Quick Start

```bash
# 1. Build the development Docker image (fast, CPU-only)
./dev.sh build

# 2. Run verification to check all modules work
./dev.sh verify

# 3. Open interactive shell for exploration
./dev.sh shell
```

## Production Container Testing (Same as GCP)

To test with the exact same container that runs on GCP:

```bash
# 1. Build production image locally (same as Cloud Build)
./dev.sh build-prod

# 2. Open shell in production container
./dev.sh shell-prod

# 3. Run with entrypoint (simulates GCP execution)
./dev.sh run-prod --mode train

# 4. Run minimal training verification
./dev.sh train-dry-run
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Development Flow                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Local Machine                  Docker Container            │
│   ┌─────────────┐               ┌─────────────────┐         │
│   │ Source Code │──── mount ───→│  /workspace     │         │
│   │ (./*)       │               │                 │         │
│   └─────────────┘               │  Python 3.10    │         │
│                                 │  PyTorch (CPU)  │         │
│   VS Code ←─── debugpy:5678 ───→│  debugpy        │         │
│                                 │                 │         │
│   localhost:5678                └─────────────────┘         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Available Commands

| Command | Description |
|---------|-------------|
| `./dev.sh build` | Build the development Docker image (CPU-only, fast) |
| `./dev.sh build-prod` | Build production Docker image (same as Cloud Build) |
| `./dev.sh shell` | Open interactive bash shell in dev container |
| `./dev.sh shell-prod` | Open interactive shell in production container |
| `./dev.sh run-prod` | Run production container with entrypoint (simulates GCP) |
| `./dev.sh verify` | Run refactoring verification script |
| `./dev.sh test` | Run pytest test suite |
| `./dev.sh debug` | Start container with debugpy (attach VS Code) |
| `./dev.sh lint` | Run code linting (black, isort) |
| `./dev.sh clean` | Clean up Docker resources |
| `./dev.sh train-dry-run` | Run minimal training loop for verification |

## Debugging with VS Code

### Method 1: Remote Attach (Recommended for Docker)

1. Start the debug container:
   ```bash
   ./dev.sh debug
   ```

2. In VS Code, open the Run and Debug panel (Cmd+Shift+D)

3. Select "Python: Attach to Docker (debugpy)"

4. Click the green play button to attach

5. Set breakpoints in your code - they will be hit!

### Method 2: Local Debugging (Without Docker)

If you have the Python environment set up locally:

1. Open VS Code in the training directory

2. Select "Python: Verify Refactoring (Local)" from the debug configurations

3. Set breakpoints and press F5

### Method 3: Debug a Specific Script

```bash
# Run any script with debugger waiting for VS Code
docker-compose run --rm -p 5678:5678 dev \
  python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
  YOUR_SCRIPT.py --your-args
```

Then attach VS Code as in Method 1.

## Test Configuration

For local debugging, use the minimal test configuration:

```bash
# In the container:
python train_sweep.py --param-config config/test_debug.yaml
```

The `config/test_debug.yaml` provides:
- Minimal data (0.1% subset)
- Only 10 training steps
- No GPU required
- No W&B uploads
- Simple dataloader strategy

## Running Tests

```bash
# Run all tests
./dev.sh test

# Run specific test file
./dev.sh test tests/test_config_system.py

# Run with verbose output
./dev.sh test -v

# Run with coverage
docker-compose run --rm dev python -m pytest --cov=. tests/
```

## Directory Structure

```
training/
├── dev.sh                  # Development helper script
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile.dev          # Development Dockerfile (CPU-only)
├── Dockerfile              # Production Dockerfile (GPU/CUDA)
├── pytest.ini             # Pytest configuration
├── config/
│   ├── defaults.yaml      # Default configuration values
│   └── test_debug.yaml    # Minimal config for debugging
├── scripts/
│   └── verify_refactoring.py  # Verification script
├── tests/
│   ├── __init__.py
│   ├── test_config_system.py
│   └── test_data_modules.py
└── .vscode/
    └── launch.json        # VS Code debug configurations
```

## Troubleshooting

### Docker build fails

```bash
# Clean up and rebuild
./dev.sh clean
./dev.sh build
```

### Can't attach debugger

1. Make sure port 5678 is not in use:
   ```bash
   lsof -i :5678
   ```

2. Check container is running:
   ```bash
   docker ps
   ```

3. Check debugpy is listening:
   ```bash
   docker-compose logs dev-debug
   ```

### Import errors

If you see import errors, make sure:
1. You're in the `/workspace` directory inside the container
2. The code is properly mounted (check with `ls`)
3. Dependencies are installed (check with `pip list`)

### W&B errors

For local development, W&B is set to offline mode. If you need online mode:
```bash
export WANDB_MODE=online
export WANDB_API_KEY=your_key
```

## Comparison: Dev vs Production

| Aspect | Development (Dockerfile.dev) | Production (Dockerfile) |
|--------|------------------------------|-------------------------|
| Base Image | python:3.10-slim | nvidia/cuda:12.1 |
| PyTorch | CPU-only (~2GB) | CUDA 12.1 (~8GB) |
| Code | Mounted (live changes) | Mounted locally / Copied in GCP |
| Tools | pytest, debugpy, black | Minimal |
| Build Time | ~2 min | ~10 min |
| Use Case | Testing, debugging | GCP training jobs |

**Note:** When running production container locally with `./dev.sh shell-prod` or `./dev.sh run-prod`, 
the code is mounted for live editing and GPU is disabled. This lets you test the exact same 
dependencies and environment as GCP, just without GPU acceleration.

## Local → GCP Workflow

The recommended workflow for developing and deploying training code:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Local → GCP Workflow                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   1. DEVELOP (fast iteration)                                               │
│      ./dev.sh build                                                          │
│      ./dev.sh shell          # Make code changes, test imports               │
│      ./dev.sh verify         # Run verification tests                        │
│      ./dev.sh test           # Run unit tests                                │
│                                                                              │
│   2. VALIDATE (production parity)                                            │
│      ./dev.sh build-prod     # Build same image as Cloud Build               │
│      ./dev.sh shell-prod     # Test in production environment                │
│      ./dev.sh train-dry-run  # Verify training loop works                    │
│                                                                              │
│   3. DEPLOY (GCP)                                                            │
│      gcloud builds submit --config cloudbuild.yaml                           │
│      ./launch_experiment_jobs.sh ...                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Next Steps

After local testing passes:

1. Build and push production image:
   ```bash
   gcloud builds submit --config cloudbuild.yaml --substitutions=_VERSION_TAG="v1.2.3"
   ```

2. Launch training job:
   ```bash
   ./launch_experiment_jobs.sh --mode train \
     --project "${PROJECT}" \
     --image-uri "${IMAGE_URI}" \
     --main-script train_sweep.py
   ```
