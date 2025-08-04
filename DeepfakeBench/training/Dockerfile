# ────────────────────────────────────────────────────────────────
# Effort-AIGI-Detection | Training container for Vertex AI Custom Jobs
# Base: CUDA 12.1 runtime + cuDNN 8 on Ubuntu 22.04 (includes NVIDIA drivers)
# Python: 3.10
# ────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# --- Environment ---
# Set non-interactive frontend for package installations
ENV DEBIAN_FRONTEND=noninteractive \
    # Standard for Python logging in containers
    PYTHONUNBUFFERED=1 \
    # Disable the pip progress bar for cleaner build logs
    PIP_PROGRESS_BAR=off \
    # Ensures wandb waits for network before failing (useful on vertex init)
    WANDB__SERVICE_WAIT=300

# --- System packages, Python 3.10, and Build Tools ---
# Install Python, pip, and essential build tools in a single layer.
# - build-essential, cmake: For compiling packages like dlib.
# - libgl1: A common runtime dependency for opencv-python.
# - git, curl, ca-certificates: General utilities.
# Finally, clean up the apt cache to reduce image size.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        python3-pip \
        build-essential \
        cmake \
        libgl1-mesa-glx \
        libglib2.0-0 \
        git \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# --- Create a symlink for `python` command ---
# Makes commands like `python -m pip` work without needing `python3.10`
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set the working directory for subsequent commands
WORKDIR /workspace

# --- Install Python Dependencies ---
# Use --no-cache-dir to keep the image lean by not storing the pip cache.

# 1. Upgrade pip to the latest version.
RUN python -m pip install --no-cache-dir --upgrade pip

# 2. Install PyTorch and Torchvision first.
# This uses versions confirmed to be available for your base image and CUDA version.
# Separating this step improves layer caching.
RUN python -m pip install --no-cache-dir \
    torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Copy and install the rest of your application's dependencies.
# Make sure this requirements.txt has had torch, torchvision, and nvidia-* packages removed.
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# --- Copy Application Code ---
# Copy your code last, so that code edits don’t invalidate the dependency layers,
# leading to faster rebuilds during development.
COPY . /workspace/

# --- Container Entrypoint ---
# This setup remains from your original file, allowing flexible command execution.
ENTRYPOINT ["/bin/bash", "/workspace/entrypoint.sh"]
CMD ["train"]