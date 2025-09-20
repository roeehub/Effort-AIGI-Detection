#!/usr/bin/env python3
"""
Backend startup script for orchestrator integration.

Simple wrapper around the main server.py for easy startup.
"""

import sys
import os
from pathlib import Path

backend_dir = Path(__file__).parent           # .../training/wma
repo_root = backend_dir.parent                # .../training
sys.path.insert(0, str(repo_root))            # finds utils/, detectors/, app3.py
sys.path.insert(0, str(backend_dir))          # finds wma/*

DEFAULTS = {
    # Your preferred default checkpoint (overridable via env)
    "CHECKPOINT_GCS_PATH": "gs://training-job-outputs/best_checkpoints/gqsvxems/top_n_effort_20250917_ep1_auc0.9700_eer0.0805.pth",
    "CUSTOM_MODEL_USE_ARCFACE": "true",

    # Inference defaults for WMA backend
    "WMA_INFER_THRESHOLD": "0.46",
    "WMA_INFER_BATCH": "16",
    "WMA_BAND_MARGIN": "0.15",
}
for k, v in DEFAULTS.items():
    os.environ.setdefault(k, v)

# Import and run the server
try:
    from server import serve
    import asyncio

    print("[BackendStartup] Starting WMA Backend gRPC Server")
    print(f"[BackendStartup] Backend directory: {backend_dir}")
    print(f"[BackendStartup] Python version: {sys.version}")

    # Run the server
    asyncio.run(serve())

except KeyboardInterrupt:
    print("\n[BackendStartup] Received interrupt, shutting down...")
    sys.exit(0)
except Exception as e:
    print(f"[BackendStartup] Error starting backend: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
