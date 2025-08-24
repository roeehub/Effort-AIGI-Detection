# ────────────────────────────────────────────────────────────────
# Effort-AIGI-Detection | Training container for Vertex AI Custom Jobs
# Base: CUDA 12.1 runtime + cuDNN 8 on Ubuntu 22.04 (includes NVIDIA drivers)
# Python: 3.10
# ────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# --- Environment ---
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_PROGRESS_BAR=off \
    WANDB__SERVICE_WAIT=300

# --- System packages, Python 3.10, and Build Tools ---
# This layer is stable and will be cached.
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
        ca-certificates \
        # Add required libraries for Pillow-SIMD compilation
        libjpeg-dev \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# --- Create a symlink for `python` command ---
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set the working directory for subsequent commands
WORKDIR /workspace

# --- Install Python Dependencies ---
# STEP 1: Copy ONLY the requirements file. This is the key to caching.
# The following RUN command will only be re-executed if this file changes.
COPY requirements.txt .

# STEP 2: Install all dependencies in a single, cached layer.
# We combine these commands to reduce the number of layers.
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
        torch==2.1.1 torchvision==0.16.1 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    python -m pip install --no-cache-dir -r requirements.txt


# Reinstall pillow-simd to leverage SIMD optimizations, separately just to save the existing cache above
RUN python -m pip install --no-cache-dir --force-reinstall pillow-simd

# --- Copy Application Code ---
# STEP 3: Now that slow dependencies are installed, copy the application code.
# Changes to your scripts (.py, .sh) will only invalidate the cache from this point on.
COPY . .


# --- Container Entrypoint ---
# This final step is very fast to build.
RUN chmod +x /workspace/entrypoint.sh
ENTRYPOINT ["/bin/bash", "/workspace/entrypoint.sh"]
