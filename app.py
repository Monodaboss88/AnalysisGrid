"""
Railway entry point — instant healthcheck, background server load.
"""
import os
import sys
import time
import threading
import asyncio

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)
        sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)
    except Exception:
        pass

_start = time.time()
print(f"[BOOT] app.py PID={os.getpid()} PORT={os.environ.get('PORT','?')}", flush=True)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Wrapper app: thin layer that proxies to unified_server once loaded ──
app = FastAPI(title="AnalysisGrid", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"], expose_headers=["*"])

_ready = False
_error = None
_real_app = None


@app.get("/api/health")
async def health():
    return {"status": "ok", "ready": _ready, "uptime": round(time.time() - _start, 1)}


@app.get("/api/status")
async def status_loading():
    """Return status-shaped response while unified_server loads."""
    if _error:
        return {
            "status": "error", "error": _error, "ready": False,
            "finnhub_connected": False, "polygon_connected": False,
            "alpaca_connected": False, "chatgpt_enabled": False,
            "data_source": f"Error: {_error}", "streaming": None,
            "watchlists": 0, "total_symbols": 0,
            "active_alerts": 0, "pending_trades": 0,
            "cache": {}, "timestamp": None,
        }
    return {
        "status": "loading", "ready": False,
        "finnhub_connected": False, "polygon_connected": False,
        "alpaca_connected": False, "chatgpt_enabled": False,
        "data_source": "Loading...", "streaming": None,
        "watchlists": 0, "total_symbols": 0,
        "active_alerts": 0, "pending_trades": 0,
        "cache": {}, "timestamp": None,
    }


@app.get("/")
async def root():
    return {"message": "AnalysisGrid API", "ready": _ready}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def catch_all(request: Request, path: str):
    """While loading, return a helpful message for any unmatched route."""
    if _error:
        return JSONResponse(status_code=503, content={"detail": f"Server startup error: {_error}"})
    return JSONResponse(status_code=503, content={"detail": "Server is still loading, please retry in a few seconds."})


print(f"[BOOT] Stub app ready in {time.time()-_start:.2f}s", flush=True)


def _load_in_background():
    global _ready, _error, _real_app
    try:
        print("[BOOT-BG] Importing unified_server...", flush=True)
        import unified_server as us
        print(f"[BOOT-BG] Import done in {time.time()-_start:.1f}s", flush=True)

        # Clear all stub routes and replace with unified_server's full routes
        app.routes.clear()
        for route in us.app.routes:
            app.routes.append(route)

        # Re-add healthcheck on top (since routes were cleared)
        @app.get("/api/health")
        async def health_ready():
            return {"status": "ok", "ready": True, "uptime": round(time.time() - _start, 1)}

        # Copy exception handlers
        for exc_class, handler in us.app.exception_handlers.items():
            app.exception_handlers[exc_class] = handler

        _real_app = us.app
        _ready = True
        print(f"[BOOT-BG] FULLY READY in {time.time()-_start:.1f}s", flush=True)

    except Exception as e:
        import traceback
        _error = str(e)
        print(f"[BOOT-BG] FATAL: {e}", flush=True)
        traceback.print_exc()


@app.on_event("startup")
async def on_startup():
    # ── Expand default executor IMMEDIATELY — this is the REAL startup ──
    # unified_server's on_startup never fires because its app isn't run by uvicorn
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=20, thread_name_prefix="async-io"))
    print("[BOOT] Default executor expanded to 20 threads", flush=True)

    t = threading.Thread(target=_load_in_background, daemon=True)
    t.start()
    print("[BOOT] Background loader started, uvicorn proceeding", flush=True)
