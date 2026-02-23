"""
Thin entry point for Railway deployment.
=========================================
Creates the FastAPI app IMMEDIATELY so uvicorn can bind to the port
and pass healthchecks, then defers all heavy imports to a startup event.

Procfile: web: python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
"""

import os
import sys
import time

# Force UTF-8 stdout/stderr (prevents Railway emoji crashes)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)
        sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)
    except Exception:
        pass

_start = time.time()
print(f"[BOOT] app.py loading - PID {os.getpid()}, Python {sys.version_info.major}.{sys.version_info.minor}", flush=True)

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ----- App created in <0.5s so uvicorn can bind and serve healthchecks ------
app = FastAPI(title="AnalysisGrid", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

_ready = False

@app.get("/api/health")
async def healthcheck():
    """Instant healthcheck - Railway hits this to confirm the container is alive."""
    return {"status": "ok", "ready": _ready}


@app.on_event("startup")
async def load_full_server():
    """Import the full unified_server module and mount everything."""
    global _ready
    print(f"[BOOT] Binding took {time.time()-_start:.1f}s - now loading unified_server...", flush=True)

    try:
        import unified_server as us

        # Steal all routes, exception handlers, and middleware from the real app
        # so that this thin app serves exactly the same API.
        app.routes.clear()
        for route in us.app.routes:
            app.routes.append(route)

        # Re-register the lightweight healthcheck (it was cleared above)
        @app.get("/api/health")
        async def healthcheck_ready():
            return {"status": "ok", "ready": True}

        # Copy exception handlers
        for exc_class, handler in us.app.exception_handlers.items():
            app.exception_handlers[exc_class] = handler

        # Fire the original startup events (Discord, auto-scanner, etc.)
        for handler in us.app.router.on_startup:
            if handler.__name__ != "load_full_server":
                try:
                    await handler()
                except Exception as e:
                    print(f"[BOOT] Startup handler error: {e}", flush=True)

        _ready = True
        elapsed = time.time() - _start
        print(f"[BOOT] Full server ready in {elapsed:.1f}s", flush=True)

    except Exception as e:
        import traceback
        print(f"[BOOT] FATAL: Failed to load unified_server: {e}", flush=True)
        traceback.print_exc()
        # Keep the health endpoint alive so Railway doesn't restart-loop
        _ready = False
