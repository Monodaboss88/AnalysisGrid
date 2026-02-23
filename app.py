"""
AnalysisGrid — Self-contained server.
No unified_server import. All endpoints defined here.
Modules loaded lazily one by one.
"""
import os
import sys
import time

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)
        sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)
    except Exception:
        pass

_start = time.time()
print(f"[BOOT] app.py PID={os.getpid()} PORT={os.environ.get('PORT','?')}", flush=True)

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone

app = FastAPI(title="AnalysisGrid", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"], expose_headers=["*"])

print(f"[BOOT] app ready in {time.time()-_start:.2f}s", flush=True)

# ── Healthcheck & Status ────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "uptime": round(time.time() - _start, 1)}

@app.get("/api/status")
async def status():
    has_polygon = bool(os.environ.get("POLYGON_API_KEY"))
    has_finnhub = bool(os.environ.get("FINNHUB_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return {
        "status": "running",
        "deploy_version": "v7-lean",
        "polygon_connected": has_polygon,
        "finnhub_connected": has_finnhub,
        "chatgpt_enabled": has_anthropic,
        "data_source": "Polygon.io (REAL-TIME)" if has_polygon else "None",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/")
async def root():
    return HTMLResponse("<h1>AnalysisGrid API</h1><p>Server running.</p>")

# ── Polygon Key ─────────────────────────────────────────────────────────────

@app.get("/api/polygon-key")
async def get_polygon_key():
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        raise HTTPException(status_code=400, detail="POLYGON_API_KEY not set")
    return {"key": key}

# ── Quote (lazy import polygon_data) ────────────────────────────────────────

_polygon_data = None

def _get_polygon():
    global _polygon_data
    if _polygon_data is None:
        try:
            import polygon_data as pd_mod
            _polygon_data = pd_mod
        except Exception as e:
            print(f"[WARN] polygon_data import failed: {e}", flush=True)
            raise HTTPException(status_code=500, detail=f"polygon_data not available: {e}")
    return _polygon_data

@app.get("/api/quote/{symbol}")
async def get_quote(symbol: str):
    symbol = symbol.upper().strip()
    pd_mod = _get_polygon()
    try:
        quote = pd_mod.get_price_quote(symbol)
        if not quote:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        return quote
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Key Management ──────────────────────────────────────────────────────────

@app.post("/api/set-polygon-key")
async def set_polygon_key(api_key: str):
    os.environ["POLYGON_API_KEY"] = api_key
    return {"status": "ok", "message": "Polygon key set"}

@app.post("/api/set-key")
async def set_finnhub_key(api_key: str):
    os.environ["FINNHUB_API_KEY"] = api_key
    return {"status": "ok", "message": "Finnhub key set"}

@app.post("/api/set-openai-key")
async def set_openai_key(api_key: str):
    os.environ["ANTHROPIC_API_KEY"] = api_key
    return {"status": "ok", "message": "Claude AI key set"}

# ── Subscription stub ───────────────────────────────────────────────────────

@app.get("/api/payments/subscription")
async def subscription():
    return {"status": "active", "tier": "premium", "message": "All features enabled"}

# ── AI Kill Switch stub ─────────────────────────────────────────────────────

@app.get("/api/config/ai-kill-switch")
async def get_kill_switch():
    return {"enabled": False}

# ── Monitor stub ────────────────────────────────────────────────────────────

@app.get("/api/monitor/status")
async def monitor_status():
    return {"running": False, "message": "Monitor not loaded in lean build"}

# ── Watchlists (lazy) ───────────────────────────────────────────────────────

_watchlist_mgr = None

def _get_watchlists():
    global _watchlist_mgr
    if _watchlist_mgr is None:
        try:
            from watchlist_manager import WatchlistManager
            _watchlist_mgr = WatchlistManager()
        except Exception as e:
            print(f"[WARN] watchlist_manager failed: {e}", flush=True)
            return None
    return _watchlist_mgr

@app.get("/api/watchlist")
@app.get("/api/watchlists")
async def get_watchlists():
    mgr = _get_watchlists()
    if not mgr:
        return []
    watchlists = mgr.get_all_watchlists()
    return [{
        "name": wl.name,
        "description": getattr(wl, "description", ""),
        "symbols": [s.symbol for s in wl.symbols],
        "count": len(wl.symbols),
    } for wl in watchlists]

@app.get("/api/watchlists/{name}")
async def get_watchlist(name: str):
    mgr = _get_watchlists()
    if not mgr:
        raise HTTPException(status_code=500, detail="Watchlist manager not available")
    wl = mgr.get_watchlist(name)
    if not wl:
        raise HTTPException(status_code=404, detail=f"Watchlist '{name}' not found")
    return {
        "name": wl.name,
        "symbols": [{"symbol": s.symbol, "name": getattr(s, "name", s.symbol)} for s in wl.symbols],
        "count": len(wl.symbols),
    }

# ── Alerts stub ─────────────────────────────────────────────────────────────

@app.get("/api/alerts")
async def get_alerts():
    return []

@app.get("/api/trades")
async def get_trades():
    return []

print(f"[BOOT] All endpoints registered in {time.time()-_start:.2f}s", flush=True)
