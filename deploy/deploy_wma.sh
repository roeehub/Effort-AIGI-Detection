#!/bin/bash

# ==============================================================================
#  WMA Backend Deployment Script
# ==============================================================================
# This script automates the deployment of the wma-backend systemd service.
# It performs checks, copies files, and restarts the service.
#
# Run this from your repository root:
# ./deploy.sh
# ==============================================================================

# --- Configuration ---
# Stop the script if any command fails
set -e

# Project paths
REPO_ROOT="/home/roee/repos/Effort-AIGI-Detection-Fork"
PROJECT_PATH="${REPO_ROOT}/DeepfakeBench/training/wma"
VENV_PYTHON="${REPO_ROOT}/venv/bin/python"

# Service configuration
SERVICE_NAME="wma-backend.service"
SRC_SERVICE_FILE="${PROJECT_PATH}/${SERVICE_NAME}"
DST_SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}"

# --- Helper Functions for Colored Output ---
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${YELLOW}[INFO] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# --- 1. Sanity Checks ---
info "Starting deployment sanity checks..."

if [ ! -d "$PROJECT_PATH" ]; then
    error "Project directory not found at: ${PROJECT_PATH}"
    exit 1
fi

if [ ! -f "$VENV_PYTHON" ]; then
    error "Python virtual environment not found at: ${VENV_PYTHON}"
    exit 1
fi

if [ ! -f "$SRC_SERVICE_FILE" ]; then
    error "Source service file not found at: ${SRC_SERVICE_FILE}"
    exit 1
fi
success "All sanity checks passed."

# --- 2. Stop and Copy Service File ---
info "Stopping existing service (if running)..."
sudo systemctl stop "$SERVICE_NAME" || true # || true ignores error if not running

info "Copying service file to systemd..."
sudo cp "$SRC_SERVICE_FILE" "$DST_SERVICE_FILE"
success "Service file copied to ${DST_SERVICE_FILE}"

# --- 3. Reload and Restart Service ---
info "Reloading systemd daemon..."
sudo systemctl daemon-reload

info "Restarting the ${SERVICE_NAME} service..."
sudo systemctl restart "$SERVICE_NAME"

# --- 4. Final Status Check ---
info "Waiting 3 seconds for service to initialize..."
sleep 3

info "Checking final status of the service..."

# Use `is-active` for a clean check in scripts
if systemctl is-active --quiet "$SERVICE_NAME"; then
    success "Service is ACTIVE and RUNNING."
    info "Showing last few log lines:"
    sudo journalctl -u "$SERVICE_NAME" -n 5 --no-pager
else
    error "Service FAILED to start."
    info "Showing detailed status and error logs:"
    sudo systemctl status "$SERVICE_NAME" --no-pager
    echo "----------------- LATEST LOGS -----------------"
    sudo journalctl -u "$SERVICE_NAME" -n 20 --no-pager
    exit 1
fi