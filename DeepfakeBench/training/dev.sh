#!/usr/bin/env bash
# ==============================================================================
# dev.sh - Local Development Helper Script
# ==============================================================================
#
# Usage:
#   ./dev.sh build         # Build dev Docker image (CPU-only, fast)
#   ./dev.sh build-prod    # Build production Docker image (same as Cloud Build)
#   ./dev.sh shell         # Open interactive shell in dev container
#   ./dev.sh shell-prod    # Open interactive shell in production container
#   ./dev.sh verify        # Run refactoring verification
#   ./dev.sh test          # Run pytest tests
#   ./dev.sh debug         # Start container with debugger (attach VS Code)
#   ./dev.sh lint          # Run code linting
#   ./dev.sh clean         # Clean up containers and volumes
#   ./dev.sh train-dry-run # Run minimal training loop (1 batch) for verification
#
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ==============================================================================
# Configuration - matches Cloud Build settings
# ==============================================================================
PROD_IMAGE_NAME="effort-detector"
PROD_IMAGE_TAG="local"
VERSION_TAG=$(cat VERSION 2>/dev/null || echo "0.0.0")

# ==============================================================================
# Commands
# ==============================================================================

cmd_build() {
    info "Building development Docker image..."
    docker-compose build dev
    info "Build complete!"
}

cmd_build_prod() {
    info "Building PRODUCTION Docker image..."
    
    # Check for -y flag (auto-confirm)
    AUTO_CONFIRM=false
    if [[ "${1:-}" == "-y" ]] || [[ "${1:-}" == "--yes" ]]; then
        AUTO_CONFIRM=true
    fi
    
    # Check architecture - CUDA images only work on x86_64 Linux
    ARCH=$(uname -m)
    OS=$(uname -s)
    
    if [[ "$OS" == "Darwin" ]] || [[ "$ARCH" == "arm64" ]] || [[ "$ARCH" == "aarch64" ]]; then
        warn "⚠️  Production Dockerfile requires x86_64 Linux (for CUDA support)"
        warn "   Your system: $OS / $ARCH"
        echo ""
        info "Will use Google Cloud Build instead (same as CI/CD pipeline)"
        echo ""
        
        # Auto-increment version
        CURRENT_VERSION=$(cat VERSION)
        MAJOR=$(echo $CURRENT_VERSION | cut -d. -f1)
        MINOR=$(echo $CURRENT_VERSION | cut -d. -f2)
        PATCH=$(echo $CURRENT_VERSION | cut -d. -f3)
        NEW_PATCH=$((PATCH + 1))
        NEW_VERSION="${MAJOR}.${MINOR}.${NEW_PATCH}"
        
        info "Current version: $CURRENT_VERSION"
        info "New version:     $NEW_VERSION"
        echo ""
        
        if [[ "$AUTO_CONFIRM" == true ]]; then
            REPLY="y"
        else
            read -p "Build production image via Cloud Build with version $NEW_VERSION? [Y/n] " -n 1 -r
            echo
        fi
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            # Update VERSION file
            echo "$NEW_VERSION" > VERSION
            info "Updated VERSION file to $NEW_VERSION"
            
            # Run Cloud Build
            info "Submitting build to Google Cloud Build..."
            info "This will take a few minutes..."
            echo ""
            
            if gcloud builds submit --config cloudbuild.yaml --substitutions=_VERSION_TAG="$NEW_VERSION"; then
                info "✅ Build successful!"
                info "Image pushed to: us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:$NEW_VERSION"
                echo ""
                
                # Offer to pull the image (auto-skip if -y)
                if [[ "$AUTO_CONFIRM" == true ]]; then
                    info "Skipping local pull (-y flag)"
                else
                    read -p "Pull the new image locally? [Y/n] " -n 1 -r
                    echo
                    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                        info "Pulling image..."
                        docker pull "us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:$NEW_VERSION"
                        docker tag "us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:$NEW_VERSION" "${PROD_IMAGE_NAME}:${PROD_IMAGE_TAG}"
                        info "Tagged as ${PROD_IMAGE_NAME}:${PROD_IMAGE_TAG}"
                        info "Run './dev.sh shell-prod' to use it"
                    fi
                fi
            else
                error "❌ Cloud Build failed!"
                # Revert VERSION file
                echo "$CURRENT_VERSION" > VERSION
                warn "Reverted VERSION file to $CURRENT_VERSION"
                exit 1
            fi
        else
            info "Build cancelled."
        fi
        return 0
    fi
    
    # On x86_64 Linux, we can build locally
    info "Version: $VERSION_TAG"
    docker build \
        -t "${PROD_IMAGE_NAME}:${VERSION_TAG}" \
        -t "${PROD_IMAGE_NAME}:latest" \
        -t "${PROD_IMAGE_NAME}:${PROD_IMAGE_TAG}" \
        -f Dockerfile \
        .
    
    info "Production image built successfully!"
    info "Tagged as:"
    info "  - ${PROD_IMAGE_NAME}:${VERSION_TAG}"
    info "  - ${PROD_IMAGE_NAME}:latest"
    info "  - ${PROD_IMAGE_NAME}:${PROD_IMAGE_TAG}"
    echo ""
    info "To run: ./dev.sh shell-prod"
}

cmd_shell() {
    info "Starting interactive shell in DEV container..."
    docker-compose run --rm dev bash
}

cmd_shell_prod() {
    info "Starting interactive shell in PRODUCTION container..."
    info "This is the same container that runs on GCP (but in CPU mode locally)"
    
    # Check if production image exists (local or pulled from GCR)
    if docker image inspect "${PROD_IMAGE_NAME}:${PROD_IMAGE_TAG}" &>/dev/null; then
        IMAGE_TO_USE="${PROD_IMAGE_NAME}:${PROD_IMAGE_TAG}"
    elif docker image inspect "us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:latest" &>/dev/null; then
        IMAGE_TO_USE="us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:latest"
        info "Using pulled GCR image"
    else
        error "Production image not found."
        error "Options:"
        error "  1. Run './dev.sh build-prod' to pull from GCR"
        error "  2. Use './dev.sh shell' for dev container instead"
        exit 1
    fi
    
    # Run production container with:
    # - Code mounted for live editing
    # - CPU mode (no GPU)
    # - W&B offline mode
    # - Same environment as GCP but without GPU
    docker run --rm -it \
        -v "$(pwd):/workspace" \
        -e PYTHONUNBUFFERED=1 \
        -e WANDB_MODE=offline \
        -e CUDA_VISIBLE_DEVICES="" \
        -e DEBUG_MODE=1 \
        -w /workspace \
        --entrypoint bash \
        "$IMAGE_TO_USE"
}

cmd_run_prod() {
    # Run the production container with entrypoint (like GCP would)
    info "Running PRODUCTION container with entrypoint (simulating GCP)..."
    
    if ! docker image inspect "${PROD_IMAGE_NAME}:${PROD_IMAGE_TAG}" &>/dev/null; then
        error "Production image not found. Run './dev.sh build-prod' first."
        exit 1
    fi
    
    # Pass through all arguments to entrypoint
    docker run --rm -it \
        -v "$(pwd):/workspace" \
        -e PYTHONUNBUFFERED=1 \
        -e WANDB_MODE=offline \
        -e CUDA_VISIBLE_DEVICES="" \
        -e DEBUG_MODE=1 \
        -w /workspace \
        "${PROD_IMAGE_NAME}:${PROD_IMAGE_TAG}" "$@"
}

cmd_verify() {
    info "Running refactoring verification..."
    docker-compose run --rm dev python scripts/verify_refactoring.py "$@"
}

cmd_test() {
    info "Running pytest tests..."
    docker-compose run --rm test "$@"
}

cmd_debug() {
    info "Starting debug container (attach VS Code to localhost:5678)..."
    echo ""
    echo "To debug a specific script, modify docker-compose.yml or run:"
    echo "  docker-compose run --rm -p 5678:5678 dev python -m debugpy --listen 0.0.0.0:5678 --wait-for-client YOUR_SCRIPT.py"
    echo ""
    docker-compose up dev-debug
}

cmd_lint() {
    info "Running linters..."
    docker-compose run --rm dev bash -c "
        echo '=== Running black (check mode) ===' && \
        black --check --diff . || true && \
        echo '' && \
        echo '=== Running isort (check mode) ===' && \
        isort --check-only --diff . || true
    "
}

cmd_clean() {
    info "Cleaning up Docker resources..."
    docker-compose down -v --remove-orphans
    docker image rm effort-detector-dev:latest 2>/dev/null || true
    info "Cleanup complete!"
}

cmd_quick_test() {
    # Quick test without Docker - runs directly on host
    # Useful when you have the Python environment set up locally
    info "Running quick verification (no Docker)..."
    python scripts/verify_refactoring.py "$@"
}

cmd_train_dry_run() {
    # Run a minimal training loop to verify everything works
    info "Running training dry-run (minimal batch test)..."
    
    # Use dev container for quick iteration, or prod for full simulation
    CONTAINER="${TRAIN_CONTAINER:-dev}"
    
    if [[ "$CONTAINER" == "prod" ]]; then
        if ! docker image inspect "${PROD_IMAGE_NAME}:${PROD_IMAGE_TAG}" &>/dev/null; then
            error "Production image not found. Run './dev.sh build-prod' first."
            exit 1
        fi
        
        docker run --rm -it \
            -v "$(pwd):/workspace" \
            -e PYTHONUNBUFFERED=1 \
            -e WANDB_MODE=offline \
            -e CUDA_VISIBLE_DEVICES="" \
            -e DEBUG_MODE=1 \
            -w /workspace \
            --entrypoint python \
            "${PROD_IMAGE_NAME}:${PROD_IMAGE_TAG}" \
            train_simple.py --config config/test_debug.yaml --dry-run "$@"
    else
        docker-compose run --rm dev \
            python train_simple.py --config config/test_debug.yaml --dry-run "$@"
    fi
}

cmd_help() {
    cat << EOF
Development Helper Script

Usage: ./dev.sh <command> [options]

Development Commands (CPU-only, fast iteration):
  build         Build the development Docker image (CPU-only)
  shell         Open an interactive shell in dev container
  verify        Run refactoring verification script
  test          Run pytest tests
  debug         Start container with debugpy (attach VS Code to localhost:5678)
  lint          Run code linting (black, isort)
  clean         Clean up Docker containers and volumes
  quick-test    Run verification directly (no Docker, needs local Python env)

Production Commands (same as Cloud Build / GCP):
  build-prod    Build production Docker image (same as Cloud Build)
  shell-prod    Open interactive shell in production container
  run-prod      Run production container with entrypoint (simulates GCP)
  train-dry-run Run minimal training loop for verification

  help          Show this help message

Examples:
  # Development workflow
  ./dev.sh build
  ./dev.sh shell
  ./dev.sh verify --verbose
  ./dev.sh test -v tests/test_config.py

  # Production simulation (same as GCP)
  ./dev.sh build-prod                    # Build same image as Cloud Build
  ./dev.sh shell-prod                    # Interactive shell in prod container
  ./dev.sh run-prod --mode train         # Run with entrypoint like GCP
  ./dev.sh train-dry-run                 # Quick training verification

  # Train dry-run with production container
  TRAIN_CONTAINER=prod ./dev.sh train-dry-run

For VS Code debugging:
  1. Run: ./dev.sh debug
  2. In VS Code, use "Python: Remote Attach" to localhost:5678

Production Image Info:
  The production image is built from Dockerfile (with CUDA) but runs in
  CPU mode locally. This lets you test the exact same container that
  will run on GCP, just without GPU acceleration.
EOF
}

# ==============================================================================
# Main
# ==============================================================================

case "${1:-help}" in
    build)         shift; cmd_build "$@" ;;
    build-prod)    shift; cmd_build_prod "$@" ;;
    shell)         shift; cmd_shell "$@" ;;
    shell-prod)    shift; cmd_shell_prod "$@" ;;
    run-prod)      shift; cmd_run_prod "$@" ;;
    verify)        shift; cmd_verify "$@" ;;
    test)          shift; cmd_test "$@" ;;
    debug)         shift; cmd_debug "$@" ;;
    lint)          shift; cmd_lint "$@" ;;
    clean)         shift; cmd_clean "$@" ;;
    quick-test)    shift; cmd_quick_test "$@" ;;
    train-dry-run) shift; cmd_train_dry_run "$@" ;;
    help|--help|-h) cmd_help ;;
    *)
        error "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac
