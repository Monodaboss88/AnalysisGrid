"""
Railway entry point — instant healthcheck, background import.
The startup event only kicks off a background thread. 
Uvicorn completes startup in <1s so healthcheck always passes.
"""
import os
import sys
import time
import threading

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
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AnalysisGrid", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"], expose_headers=["*"])

_ready = False
_error = None

@app.get("/api/health")
async def health():
    return {"status": "ok", "ready": _ready, "uptime": round(time.time() - _start, 1)}

@app.get("/api/status")
async def status():
    if _error:
        return {"status": "error", "error": _error}
    return {"status": "running" if _ready else "loading", "ready": _ready}

@app.get("/")
async def root():
    return {"message": "AnalysisGrid API", "ready": _ready}

print(f"[BOOT] app ready in {time.time()-_start:.2f}s", flush=True)


def _load_in_background():
    """Import unified_server in a background thread, then inject routes."""
    global _ready, _error
    try:
        print("[BOOT-BG] Importing unified_server...", flush=True)
        import unified_server as us
        print(f"[BOOT-BG] Import done in {time.time()-_start:.1f}s", flush=True)

        # Inject routes from unified_server into this app
        app.routes.clear()
        for route in us.app.routes:
            app.routes.append(route)

        # Re-add healthcheck
        @app.get("/api/health")
        async def health_ready():
            return {"status": "ok", "ready": True, "uptime": round(time.time() - _start, 1)}

        # Copy exception handlers
        for exc_class, handler in us.app.exception_handlers.items():
            app.exception_handlers[exc_class] = handler

        _ready = True
        print(f"[BOOT-BG] FULLY READY in {time.time()-_start:.1f}s", flush=True)

    except Exception as e:
        import traceback
        _error = str(e)
        print(f"[BOOT-BG] FATAL: {e}", flush=True)
        traceback.print_exc()


@app.on_event("startup")
async def on_startup():
    """Just kick off background thread — don't block uvicorn startup."""
    t = threading.Thread(target=_load_in_background, daemon=True)
    t.start()
    print("[BOOT] Background loader started, uvicorn proceeding", flush=True)
