#!/usr/bin/env python3
"""
Backend startup script for orchestrator integration.

Simple wrapper around the main server.py for easy startup.
"""

import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# print the sys path
print(f"[BackendStartup] Python sys.path: {sys.path}")
exit(0)



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