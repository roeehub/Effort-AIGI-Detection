#!/usr/bin/env python3
"""
Backend startup script for orchestrator integration.

Simple wrapper around the main server.py for easy startup.
"""

# --- bootstrap paths + shim local `utils` package (no repo edits needed) ---
import sys, os, types, importlib.util
from pathlib import Path

backend_dir = Path(__file__).parent  # .../training/wma
repo_root = backend_dir.parent  # .../training
sys.path.insert(0, str(repo_root))  # ensure app3, detectors, etc.
sys.path.insert(0, str(backend_dir))  # ensure wma/*

# Force-load our local training/utils as the top-level "utils" package
local_utils_dir = repo_root / "utils"
if local_utils_dir.is_dir():
    # Create a synthetic "utils" package pointing to training/utils
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = [str(local_utils_dir)]
    sys.modules.setdefault("utils", utils_pkg)

    # Preload the modules you need (add more if needed)
    for modname in ("registry", "metrics"):
        mod_path = local_utils_dir / f"{modname}.py"
        if mod_path.exists():
            spec = importlib.util.spec_from_file_location(f"utils.{modname}", str(mod_path))
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader, f"Cannot load {mod_path}"
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            sys.modules[f"utils.{modname}"] = mod
            setattr(utils_pkg, modname, mod)
# --- end bootstrap ---

DEFAULTS = {
    # Video API configuration
    "VIDEO_API_HOST": "34.16.217.28",
    "VIDEO_API_PORT": "8999",
    "VIDEO_API_TIMEOUT": "30",
    "VIDEO_YOLO_CONF_THRESHOLD": "0.80",

    # Inference defaults for WMA backend
    # "WMA_INFER_THRESHOLD": "0.46",
    "WMA_INFER_THRESHOLD": "0.75",
    "WMA_INFER_BATCH": "16",
    # "WMA_BAND_MARGIN": "0.03",
    "WMA_BAND_MARGIN": "0.02",
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
