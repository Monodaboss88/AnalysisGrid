"""
Thin entry point for Railway deployment.
Serves /api/health instantly, defers unified_server import to startup.
"""
import os
import sys
import time
import asyncio

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)
        sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)
    except Exception:
        pass

_start = time.time()
print(f"[BOOT] app.py PID={os.getpid()} PORT={os.environ.get('PORT','?')}", flush=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AnalysisGrid", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"], expose_headers=["*"])

_ready = False

@app.get("/api/health")
async def health():
    return {"status": "ok", "ready": _ready, "uptime": round(time.time() - _start, 1)}

@app.get("/api/status")
async def status():
    return {"status": "running" if _ready else "loading", "ready": _ready}

@app.get("/")
async def root():
    return {"message": "AnalysisGrid API running", "ready": _ready}

print(f"[BOOT] app ready in {time.time()-_start:.2f}s", flush=True)


@app.on_event("startup")
async def load_full_server():
    global _ready
    print(f"[BOOT] Loading unified_server...", flush=True)

    try:
        import unified_server as us
        print(f"[BOOT] unified_server imported in {time.time()-_start:.1f}s", flush=True)

        # Copy all routes from the real app
        app.routes.clear()
        for route in us.app.routes:
            app.routes.append(route)

        # Re-add healthcheck (cleared above)
        @app.get("/api/health")
        async def health_ready():
            return {"status": "ok", "ready": True, "uptime": round(time.time() - _start, 1)}

        # Copy exception handlers
        for exc_class, handler in us.app.exception_handlers.items():
            app.exception_handlers[exc_class] = handler

        # Fire unified_server startup handlers
        for handler in us.app.router.on_startup:
            if handler.__name__ != "load_full_server":
                try:
                    await asyncio.wait_for(handler(), timeout=30)
                except asyncio.TimeoutError:
                    print(f"[BOOT] Handler {handler.__name__} timed out", flush=True)
                except Exception as e:
                    print(f"[BOOT] Handler {handler.__name__} error: {e}", flush=True)

        _ready = True
        print(f"[BOOT] READY in {time.time()-_start:.1f}s", flush=True)

    except Exception as e:
        import traceback
        print(f"[BOOT] FATAL: {e}", flush=True)
        traceback.print_exc()
        _ready = False
