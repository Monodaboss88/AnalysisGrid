"""
Unified Server — Clean Base Build
===================================
Minimal server with Polygon API data layer.
All heavy modules loaded lazily or via optional routers.

Run: uvicorn unified_server:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import time
import math
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import asdict
from collections import OrderedDict

# Force UTF-8 stdout/stderr (fixes Railway emoji crashes)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)
        sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)
    except Exception:
        pass

print(f"[BOOT] unified_server.py loading — Python {sys.version_info.major}.{sys.version_info.minor}, PID {os.getpid()}", flush=True)

# Data libraries
try:
    import pandas as pd
    from polygon_data import get_bars, get_price_quote
    print("[BOOT] polygon_data loaded", flush=True)
except Exception as e:
    pd = None
    get_bars = None
    get_price_quote = None
    print(f"[BOOT] polygon_data FAILED: {e}", flush=True)

# FastAPI
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
import uvicorn

# Our core modules — wrapped so a missing dep doesn't crash the whole server
try:
    from chart_input_analyzer import ChartInputSystem, ChartInput
    print("[BOOT] chart_input_analyzer loaded", flush=True)
except Exception as e:
    ChartInputSystem = None
    ChartInput = None
    print(f"[BOOT] chart_input_analyzer FAILED: {e}", flush=True)

try:
    from finnhub_scanner_v2 import FinnhubScanner, TechnicalCalculator
    print("[BOOT] finnhub_scanner_v2 loaded", flush=True)
except Exception as e:
    FinnhubScanner = None
    TechnicalCalculator = None
    print(f"[BOOT] finnhub_scanner_v2 FAILED: {e}", flush=True)

try:
    from watchlist_manager import WatchlistManager
    print("[BOOT] watchlist_manager loaded", flush=True)
except Exception as e:
    WatchlistManager = None
    print(f"[BOOT] watchlist_manager FAILED: {e}", flush=True)

print("[BOOT] Core imports done", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONAL MODULES — each in try/except so missing deps never crash the server
# ═══════════════════════════════════════════════════════════════════════════════

# Anthropic (Claude) for AI commentary
try:
    import anthropic
    anthropic_available = True
except ImportError:
    anthropic_available = False
    anthropic = None

# Discord Bot
try:
    from discord_endpoints import discord_router, setup_discord
    discord_available = True
except ImportError as e:
    discord_available = False
    setup_discord = None
    print(f"[BOOT] Discord not loaded: {e}")

# AI Advisor
try:
    from ai_advisor_endpoints import ai_router
    ai_advisor_available = True
except ImportError:
    ai_advisor_available = False

# Trade Journal
try:
    from trade_journal_endpoints import journal_router
    trade_journal_available = True
except ImportError:
    trade_journal_available = False

# Signal Probability
try:
    from signal_endpoints import signal_router
    signal_available = True
except ImportError:
    signal_available = False

# Range Watcher
try:
    from rangewatcher.range_watcher_endpoints import range_router, set_scanner as set_range_scanner, set_openai_client as set_range_openai
    range_watcher_available = True
except ImportError:
    range_watcher_available = False
    set_range_scanner = None
    set_range_openai = None

# Auth Middleware
try:
    from auth_middleware import (
        init_firebase, get_current_user, require_auth, require_admin,
        require_manager, OrgContext, log_audit, SUBSCRIPTION_TIERS
    )
    auth_available = True
except ImportError:
    auth_available = False

# Firestore Storage
try:
    from firestore_store import get_firestore, UserAlert, UserTrade
    firestore_available = True
except ImportError:
    firestore_available = False

# Authorize.net Payments
try:
    from authorize_payments import payment_router
    payments_available = True
except ImportError:
    payments_available = False

# Workflow
try:
    from workflow_endpoints import workflow_router
    workflow_available = True
except ImportError:
    workflow_available = False

# Entry Scanner
try:
    from emtryscan.entry_scanner_endpoints import entry_router, set_finnhub_scanner as set_entry_scanner, set_finnhub_scanner_getter
    entry_scanner_available = True
except ImportError:
    entry_scanner_available = False
    set_entry_scanner = None
    set_finnhub_scanner_getter = None

# Trade Rule Engine
try:
    from rule_engine_endpoints import rule_router, set_scanner_for_rules
    rule_engine_available = True
except ImportError:
    rule_engine_available = False
    set_scanner_for_rules = None

# Compression Scanner
try:
    from compscan.fastapi_endpoints import compression_router
    compression_scanner_available = True
except ImportError:
    compression_scanner_available = False

# Capitulation & Euphoria Detector
try:
    from capitulation_detector_v2 import (
        CapitulationDetectorV2 as CapitulationDetector, CapitulationLevel, scan_for_capitulation,
        EuphoriaLevel, scan_for_euphoria
    )
    capitulation_available = True
except ImportError:
    capitulation_available = False

# Structure Reversal Detector
try:
    from structure_reversal_detector import StructureReversalDetector, StructureContext, ReversalAlert
    from rangewatcher.range_watcher import RangeWatcher
    structure_reversal_detector = StructureReversalDetector(min_confidence=40.0)
    range_watcher_analyzer = RangeWatcher()
    structure_reversal_available = True
except ImportError:
    structure_reversal_available = False
    structure_reversal_detector = None
    range_watcher_analyzer = None

# Absorption Detector
try:
    from absorption_detector import AbsorptionDetector, AbsorptionResult
    absorption_detector = AbsorptionDetector()
    absorption_available = True
except ImportError:
    absorption_available = False
    absorption_detector = None

# Squeeze Detector
try:
    from squeeze_detector_v2 import SqueezeDetectorV2 as SqueezeDetector, SqueezeMetrics, scan_for_squeezes_v2 as scan_for_squeezes
    squeeze_available = True
except ImportError:
    squeeze_available = False

# Run Sustainability
try:
    from sustainability_endpoints import sustainability_router
    sustainability_available = True
except ImportError:
    sustainability_available = False

# Alpha Scanner
try:
    from alpha_endpoints import alpha_router
    alpha_scanner_available = True
except ImportError:
    alpha_scanner_available = False

# Convergence Scanner
try:
    from convergence_endpoint import convergence_router
    convergence_available = True
except ImportError:
    convergence_available = False

# Admin / Enterprise Endpoints
try:
    from admin_endpoints import admin_router
    admin_available = True
except ImportError:
    admin_available = False

# Auto-Scanner
try:
    from auto_scanner import setup_auto_scanner, get_auto_scanner
    auto_scanner_available = True
except ImportError:
    auto_scanner_available = False
    setup_auto_scanner = None

# WebSocket Streaming
try:
    from polygon_websocket import StreamingManager, MinuteBar
    streaming_available = True
    streaming_manager = StreamingManager.get_instance()
except ImportError:
    streaming_available = False
    streaming_manager = None

# Extension Predictor
try:
    from extension_predictor_v2 import ExtensionPredictor, CandleData
    extension_predictor = ExtensionPredictor(candle_minutes=120)
    extension_available = True
except ImportError:
    extension_available = False
    extension_predictor = None

# Dual Setup Generator
try:
    from dual_setup_generator_v2 import DualSetupGenerator, generate_dual_setup
    dual_setup_available = True
except ImportError:
    dual_setup_available = False
    generate_dual_setup = None

# MTF Auction Scanner V2
try:
    from mtf_auction_scanner_v2 import MTFAuctionScanner
    mtf_scanner = MTFAuctionScanner()
    mtf_scanner_available = True
except ImportError:
    mtf_scanner_available = False
    mtf_scanner = None

# Overnight Model V2
try:
    from overnight_model_v2 import OvernightModelV2, OvernightPrediction, scan_overnight as _scan_overnight_single
    overnight_model = OvernightModelV2()
    overnight_available = True
except ImportError:
    overnight_available = False
    overnight_model = None
    _scan_overnight_single = None

print("[BOOT] Optional modules loaded", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_dict(obj):
    """Convert a dataclass (or dict) to JSON-safe dict, handling numpy types."""
    import numpy as np
    if obj is None:
        return None
    d = asdict(obj) if hasattr(obj, '__dataclass_fields__') else dict(obj)
    def _convert(v):
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, dict):
            return {k: _convert(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_convert(item) for item in v]
        return v
    return {k: _convert(v) for k, v in d.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ChartInputRequest(BaseModel):
    symbol: str
    price: float
    vah: float
    poc: float
    val: float
    vwap: float
    rsi: float
    timeframe: str = "1HR"


class AlertRequest(BaseModel):
    symbol: str
    level: float
    direction: str
    action: str
    note: str = ""


class TradeRequest(BaseModel):
    symbol: str
    timeframe: str
    direction: str
    entry: float
    stop: float
    target: float
    target2: Optional[float] = None
    signal: str = ""
    confidence: float = 0
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# TTL CACHE
# ═══════════════════════════════════════════════════════════════════════════════
class TTLCache:
    def __init__(self, ttl_seconds=60, max_size=500):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._cache = OrderedDict()

    def get(self, key):
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry['ts'] < self.ttl:
                self._cache.move_to_end(key)
                return entry['data']
            else:
                del self._cache[key]
        return None

    def set(self, key, data):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = {'data': data, 'ts': time.time()}
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()

    @property
    def size(self):
        return len(self._cache)

candle_cache = TTLCache(ttl_seconds=60, max_size=300)
analysis_cache = TTLCache(ttl_seconds=45, max_size=300)
squeeze_cache = TTLCache(ttl_seconds=120, max_size=200)
absorption_cache = TTLCache(ttl_seconds=90, max_size=200)


# ═══════════════════════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="AnalysisGrid", version="2.0.0")

# ── Custom JSON encoder that handles numpy types ──
import json as _json

class _SafeEncoder(_json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)

def _safe_json_response(data, status_code=200):
    """JSONResponse that handles numpy types gracefully."""
    from fastapi.responses import Response
    content = _json.dumps(data, cls=_SafeEncoder)
    return Response(content=content, status_code=status_code,
                    media_type="application/json")

# Global exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail},
                        headers={"Access-Control-Allow-Origin": "*"})

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    print(f"[ERROR] {exc}\n{traceback.format_exc()[:500]}")
    return JSONResponse(status_code=500, content={"detail": str(exc)},
                        headers={"Access-Control-Allow-Origin": "*"})

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# RATE LIMITING MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════════════

import time as _time
from collections import defaultdict as _defaultdict

class _RateLimitStore:
    """In-memory sliding-window rate limiter per IP."""
    def __init__(self):
        self.requests: dict[str, list[float]] = _defaultdict(list)

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        now = _time.time()
        cutoff = now - window_seconds
        # Prune old entries
        self.requests[key] = [t for t in self.requests[key] if t > cutoff]
        if len(self.requests[key]) >= max_requests:
            return False
        self.requests[key].append(now)
        return True

_rate_store = _RateLimitStore()

# Default limits: 120 requests per minute per IP
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Per-IP rate limiting. Skips health/static endpoints."""
    path = request.url.path
    # Skip rate limiting for static files and health checks
    if path.startswith(("/icons", "/manifest", "/config.js", "/notifications.js")) or path == "/api/health":
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    if not _rate_store.is_allowed(client_ip, RATE_LIMIT_PER_MINUTE, 60):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again shortly."},
            headers={"Retry-After": "60", "Access-Control-Allow-Origin": "*"}
        )
    response = await call_next(request)
    return response


# ── Lightweight healthcheck ──────────────────────────────────────────────────
@app.get("/api/health")
async def healthcheck():
    return {"status": "ok", "version": "v8-thread-fix"}


# ── Register optional routers ───────────────────────────────────────────────
if ai_advisor_available:
    app.include_router(ai_router, prefix="/api/ai")
if trade_journal_available:
    app.include_router(journal_router)
if signal_available:
    app.include_router(signal_router)
if range_watcher_available:
    app.include_router(range_router, prefix="/api/range")
if payments_available:
    app.include_router(payment_router)
if workflow_available:
    app.include_router(workflow_router)
if entry_scanner_available:
    app.include_router(entry_router)
if rule_engine_available:
    app.include_router(rule_router, prefix="/api/rules")
if compression_scanner_available:
    app.include_router(compression_router)
if sustainability_available:
    app.include_router(sustainability_router)
if discord_available:
    app.include_router(discord_router, prefix="/discord")
if alpha_scanner_available:
    app.include_router(alpha_router, prefix="")
if convergence_available:
    app.include_router(convergence_router, prefix="/api/convergence")
if admin_available:
    app.include_router(admin_router)

# Trading Card router
try:
    from card_endpoints import card_router
    app.include_router(card_router)
except Exception:
    pass

# Combo Scanner router
try:
    from combo_endpoints import combo_router
    app.include_router(combo_router)
    print("[BOOT] Combo Scanner router loaded", flush=True)
except Exception as e:
    print(f"[BOOT] Combo Scanner not available: {e}", flush=True)

print("[BOOT] Routers registered", flush=True)


# ── Startup ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    print("[BOOT] on_startup() fired", flush=True)

    # ── Expand default executor — prevents scanner queue starvation ──
    from concurrent.futures import ThreadPoolExecutor as _TPE
    loop = asyncio.get_running_loop()
    loop.set_default_executor(_TPE(max_workers=20, thread_name_prefix="async-io"))
    print("[BOOT] Default executor set to 20 threads", flush=True)

    # DISCORD PAUSED — change `False` to `discord_available` to re-enable
    if False and discord_available and setup_discord:
        try:
            await asyncio.wait_for(setup_discord(app), timeout=30)
        except asyncio.TimeoutError:
            print("[BOOT] Discord timed out after 30s — continuing")
        except Exception as e:
            print(f"[BOOT] Discord error: {e}")

    # AUTO-SCANNER PAUSED — set auto_start=True to re-enable Discord scan messages
    if False and auto_scanner_available and setup_auto_scanner:
        try:
            from discord_bot import get_discord
            discord_client = get_discord()
            async def _start_auto_scanner():
                await asyncio.sleep(5)
                setup_auto_scanner(watchlist_mgr=watchlist_mgr, discord_client=discord_client, auto_start=True)
                print("[BOOT] Auto-Scanner started")
            asyncio.create_task(_start_auto_scanner())
        except Exception as e:
            print(f"[BOOT] Auto-Scanner error: {e}")

    print("[BOOT] Startup complete", flush=True)


# ── Initialize Firebase Auth (non-blocking) ─────────────────────────────────
if auth_available:
    try:
        init_firebase()
    except Exception as e:
        print(f"[BOOT] Firebase init error: {e}")

print("[BOOT] Initializing components...", flush=True)

# Core components — safe init
chart_system = ChartInputSystem(data_dir="./scanner_data") if ChartInputSystem else None
watchlist_mgr = WatchlistManager() if WatchlistManager else None

# Finnhub scanner (lazy init)
finnhub_scanner: Optional[FinnhubScanner] = None

# Anthropic client
anthropic_client = None
if anthropic_available and os.environ.get("ANTHROPIC_API_KEY"):
    try:
        anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        print("[BOOT] Claude AI enabled")
        if set_range_openai:
            set_range_openai(anthropic_client)
    except Exception as e:
        print(f"[BOOT] Anthropic init error: {e}")


def get_finnhub_scanner() -> FinnhubScanner:
    global finnhub_scanner
    if finnhub_scanner is None:
        api_key = os.environ.get("FINNHUB_API_KEY")
        polygon_key = os.environ.get("POLYGON_API_KEY")
        if not api_key and not polygon_key:
            raise HTTPException(status_code=400, detail="No API keys set.")
        if not api_key:
            api_key = "dummy_key_polygon_only"
        finnhub_scanner = FinnhubScanner(api_key)
        if polygon_key and hasattr(finnhub_scanner, 'set_polygon_key'):
            finnhub_scanner.set_polygon_key(polygon_key)
        if set_range_scanner:
            set_range_scanner(finnhub_scanner)
        if rule_engine_available and set_scanner_for_rules:
            set_scanner_for_rules(finnhub_scanner)
        if entry_scanner_available and set_entry_scanner:
            set_entry_scanner(finnhub_scanner)
    return finnhub_scanner

if entry_scanner_available and set_finnhub_scanner_getter:
    set_finnhub_scanner_getter(get_finnhub_scanner)

# Options Flow Stream (real-time options flow detection)
try:
    from options_flow_stream import get_flow_stream
    flow_stream = get_flow_stream()
    flow_stream_available = True
    print("[BOOT] Options Flow Stream loaded")
except ImportError as e:
    flow_stream = None
    flow_stream_available = False
    print(f"[BOOT] Options Flow Stream not loaded: {e}")

print(f"[BOOT] Ready — PORT={os.environ.get('PORT', '8000')}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS — Status, Quotes, Polygon Key
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/status")
async def get_status():
    has_polygon = bool(os.environ.get("POLYGON_API_KEY"))
    has_finnhub = bool(os.environ.get("FINNHUB_API_KEY"))
    has_alpaca = bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"))
    watchlists = watchlist_mgr.get_all_watchlists()

    streaming_status = None
    if streaming_available and streaming_manager and streaming_manager.streamer:
        streaming_status = streaming_manager.get_status()

    if streaming_status and streaming_status.get('connected'):
        data_source = "Polygon.io WebSocket (LIVE STREAMING)"
    elif has_polygon:
        data_source = "Polygon.io (REAL-TIME)"
    elif has_alpaca:
        data_source = "Alpaca (REAL-TIME)"
    elif has_finnhub:
        data_source = "Finnhub (15-min delayed)"
    else:
        data_source = "None"

    return {
        "status": "running",
        "deploy_version": "v6-clean-base",
        "finnhub_connected": has_finnhub,
        "alpaca_connected": has_alpaca,
        "polygon_connected": has_polygon,
        "chatgpt_enabled": anthropic_client is not None,
        "data_source": data_source,
        "streaming": streaming_status,
        "watchlists": len(watchlists),
        "total_symbols": sum(len(lst.symbols) for lst in watchlists),
        "active_alerts": len(chart_system.get_alerts()),
        "pending_trades": len(chart_system.get_pending_trades()),
        "cache": {
            "analysis": analysis_cache.size,
            "squeeze": squeeze_cache.size,
            "absorption": absorption_cache.size,
            "candles": candle_cache.size,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/polygon-key")
async def get_polygon_key():
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        raise HTTPException(status_code=400, detail="POLYGON_API_KEY not set")
    return {"key": key}


@app.get("/api/quote/{symbol}")
async def get_quote(symbol: str):
    symbol = symbol.upper().strip()
    try:
        quote = get_price_quote(symbol)
        if not quote:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        return quote
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# WATCHLISTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/watchlist")
@app.get("/api/watchlists")
async def get_watchlists():
    watchlists = watchlist_mgr.get_all_watchlists()
    return [{
        "name": wl.name,
        "description": getattr(wl, "description", ""),
        "symbols": [s.symbol for s in wl.symbols],
        "count": len(wl.symbols),
    } for wl in watchlists]


@app.get("/api/watchlists/{name}")
async def get_watchlist(name: str):
    wl = watchlist_mgr.get_watchlist(name)
    if not wl:
        raise HTTPException(status_code=404, detail=f"Watchlist '{name}' not found")
    return {
        "name": wl.name,
        "description": getattr(wl, "description", ""),
        "symbols": [{"symbol": s.symbol, "name": getattr(s, "name", s.symbol)} for s in wl.symbols],
        "count": len(wl.symbols),
    }


@app.post("/api/watchlists")
async def create_watchlist(request: Request):
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Watchlist name required")
    try:
        wl = watchlist_mgr.create_watchlist(name, body.get("description", ""))
        return {"status": "ok", "name": wl.name, "message": f"Watchlist '{name}' created"}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/api/watchlists/{name}/symbols")
async def add_symbol_to_watchlist(name: str, request: Request):
    body = await request.json()
    symbol = body.get("symbol", "").strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol required")
    try:
        sym = watchlist_mgr.add_symbol(name, symbol, name=body.get("name", ""))
        return {"status": "ok", "symbol": sym.symbol, "watchlist": name}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/api/watchlists/{name}/symbols/{symbol}")
async def remove_symbol_from_watchlist(name: str, symbol: str):
    removed = watchlist_mgr.remove_symbol(name, symbol.upper())
    if not removed:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found in '{name}'")
    return {"status": "ok", "removed": symbol.upper(), "watchlist": name}


@app.delete("/api/watchlists/{name}")
async def delete_watchlist(name: str):
    try:
        deleted = watchlist_mgr.delete_watchlist(name)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Watchlist '{name}' not found")
        return {"status": "ok", "deleted": name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE ANALYSIS (Polygon-powered)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/analyze/live/{symbol}")
async def analyze_live(
    symbol: str,
    timeframe: str = Query("1HR", description="30MIN, 1HR, 2HR, 4HR, DAILY"),
    with_ai: bool = Query(True, description="Include AI commentary"),
    use_rules: bool = Query(False, description="Use rule-based analysis instead of AI"),
    entry_signal: str = Query(None, description="Entry signal e.g. 'failed_breakout:short'"),
    vp_period: str = Query("swing", description="VP lookback: 'day','swing','position','investment'")
):
    """Analyze symbol with live Polygon data"""
    try:
        symbol = symbol.upper().strip()
        cache_key = f"{symbol}:{timeframe.upper()}:{vp_period}"
        cached = analysis_cache.get(cache_key)
        if cached and not with_ai:
            cached['_cached'] = True
            return cached

        scanner = None
        use_yfinance = False
        try:
            scanner = get_finnhub_scanner()
        except Exception:
            use_yfinance = True

        # Resolution mapping
        resolution_map = {
            "5MIN": "5", "15MIN": "15", "30MIN": "30",
            "1HR": "60", "2HR": "60", "4HR": "60", "DAILY": "D"
        }
        resolution = resolution_map.get(timeframe.upper(), "60")

        # VP_BARS based on vp_period
        swing_bars = {"5MIN": 200, "15MIN": 80, "30MIN": 50, "1HR": 35, "2HR": 20, "4HR": 12, "DAILY": 20}
        period_multipliers = {"day": 0.2, "swing": 1.0, "position": 4.0, "investment": 12.0}
        base_bars = swing_bars.get(timeframe.upper(), 35)
        multiplier = period_multipliers.get(vp_period.lower(), 1.0)
        VP_BARS = int(base_bars * multiplier)
        if vp_period.lower() == "investment" and timeframe.upper() == "DAILY":
            VP_BARS = 200

        # Days of data to fetch
        days_base = {"5MIN": 3, "15MIN": 5, "30MIN": 7, "1HR": 10, "2HR": 20, "4HR": 40, "DAILY": 60}
        days_multiplier = {"day": 1, "swing": 1, "position": 3, "investment": 5}
        days_back = days_base.get(timeframe.upper(), 10) * days_multiplier.get(vp_period.lower(), 1)
        days_back = min(days_back, 365)

        # Get candle data
        df = None
        if scanner:
            df = scanner._get_candles(symbol, resolution, days_back)

        if df is None or len(df) < 10:
            use_yfinance = True
            if get_bars:
                yf_interval_map = {"5MIN": "5m", "15MIN": "15m", "30MIN": "30m", "1HR": "1h", "2HR": "1h", "4HR": "1h", "DAILY": "1d"}
                yf_period_map = {"5MIN": "5d", "15MIN": "1mo", "30MIN": "1mo", "1HR": "1mo", "2HR": "3mo", "4HR": "3mo", "DAILY": "1y"}
                yf_interval = yf_interval_map.get(timeframe.upper(), "1h")
                yf_period = yf_period_map.get(timeframe.upper(), "1mo")
                if vp_period.lower() in ("position", "investment"):
                    if timeframe.upper() == "DAILY":
                        yf_period = "1y"
                    elif timeframe.upper() in ("1HR", "2HR", "4HR"):
                        yf_period = "3mo"
                df = get_bars(symbol, period=yf_period, interval=yf_interval)
                if df is not None and not df.empty:
                    df.columns = [c.lower() for c in df.columns]
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('US/Eastern')
                else:
                    df = None
                print(f"Polygon fallback for {symbol}: {len(df) if df is not None else 0} bars ({yf_interval})")

        # Resample for 2HR/4HR
        resample_map = {"2HR": "2h", "4HR": "4h"}
        if df is not None and timeframe.upper() in resample_map:
            if scanner and not use_yfinance:
                df = scanner._resample_to_timeframe(df, timeframe)
            else:
                df = df.resample(resample_map[timeframe.upper()]).agg({
                    'open': 'first', 'high': 'max', 'low': 'min',
                    'close': 'last', 'volume': 'sum'
                }).dropna()

        # Trim to VP_BARS
        if df is not None and len(df) > VP_BARS:
            df = df.tail(VP_BARS)

        print(f"VP Calculation using {len(df) if df is not None else 0} bars for {symbol} {timeframe}")

        # Calculate VP levels
        _atr = 0
        if df is not None and len(df) >= 10:
            calc = scanner.calc if scanner else (TechnicalCalculator() if TechnicalCalculator else None)
            if calc:
                poc, vah, val = calc.calculate_volume_profile(df)
                vwap = calc.calculate_vwap(df)
                rsi = calc.calculate_rsi(df)
                _atr = calc.calculate_atr(df)
            else:
                poc, vah, val, vwap, rsi = 0, 0, 0, 0, 50
        else:
            poc, vah, val, vwap, rsi = 0, 0, 0, 0, 50

        # Get real-time quote
        quote = None
        day_open = day_high = day_low = prev_close = 0
        current_price = 0
        quote_source = 'none'
        if scanner:
            quote = scanner.get_quote(symbol)
        if quote and quote.get('current'):
            current_price = float(quote['current'])
            quote_source = quote.get('source', 'unknown')
            day_open = float(quote.get('open', 0) or 0)
            day_high = float(quote.get('high', 0) or 0)
            day_low = float(quote.get('low', 0) or 0)
            prev_close = float(quote.get('prev_close', 0) or 0)
        elif df is not None and len(df) > 0:
            current_price = float(df['close'].iloc[-1])
            quote_source = 'yfinance' if use_yfinance else 'candle_fallback'
            day_open = float(df['open'].iloc[-1]) if 'open' in df.columns else 0
            day_high = float(df['high'].iloc[-1]) if 'high' in df.columns else 0
            day_low = float(df['low'].iloc[-1]) if 'low' in df.columns else 0
            prev_close = float(df['close'].iloc[-2]) if len(df) > 1 else 0

        # Run analysis
        result = None
        if scanner:
            result = scanner.analyze(symbol, timeframe)
        if result is None and df is not None and len(df) >= 10 and chart_system:
            calc = scanner.calc if scanner else (TechnicalCalculator() if TechnicalCalculator else None)
            if calc:
                _rvol = calc.calculate_relative_volume(df)
                _vol_trend = calc.calculate_volume_trend(df)
                _vol_div = calc.detect_volume_divergence(df)
                _has_rejection = False
                if current_price and current_price < val and val > 0:
                    _has_rejection = calc.is_rejection_candle(df, "bullish")
                elif current_price and current_price > vah and vah > 0:
                    _has_rejection = calc.is_rejection_candle(df, "bearish")
                result = chart_system.analyze(
                    symbol=symbol, price=current_price,
                    vah=vah, poc=poc, val=val, vwap=vwap, rsi=rsi,
                    timeframe=timeframe, rvol=_rvol,
                    volume_trend=_vol_trend, volume_divergence=_vol_div,
                    atr=_atr, has_rejection=_has_rejection
                )

        if not result:
            raise HTTPException(status_code=404, detail=f"Could not analyze {symbol}")

        # Get order flow
        order_flow = None
        try:
            if scanner:
                of_result = scanner.get_order_flow_analysis(symbol, timeframe, vp_period)
                if of_result and hasattr(of_result, 'to_dict'):
                    order_flow = of_result.to_dict()
                elif of_result and isinstance(of_result, dict):
                    order_flow = of_result
        except Exception as e:
            print(f"Order flow error: {e}")

        response = {
            "symbol": symbol,
            "timeframe": str(result.timeframe),
            "signal": str(result.signal),
            "signal_emoji": str(result.signal_emoji),
            "bull_score": float(result.bull_score),
            "bear_score": float(result.bear_score),
            "confidence": float(result.confidence),
            "high_prob": float(result.high_prob),
            "low_prob": float(result.low_prob),
            "position": str(result.position),
            "vwap_zone": str(result.vwap_zone),
            "rsi_zone": str(result.rsi_zone),
            "notes": [str(n) for n in result.notes] if result.notes else [],
            "timestamp": datetime.now().isoformat(),
            "current_price": float(current_price) if current_price else 0,
            "day_open": float(day_open),
            "day_high": float(day_high),
            "day_low": float(day_low),
            "prev_close": float(prev_close),
            "quote_source": quote_source,
            "vah": float(vah) if vah else 0,
            "poc": float(poc) if poc else 0,
            "val": float(val) if val else 0,
            "vwap": float(vwap) if vwap else 0,
            "rsi": float(rsi) if rsi else 50,
            "rvol": float(getattr(result, 'rvol', 1.0)),
            "volume_trend": str(getattr(result, 'volume_trend', 'neutral')),
            "volume_divergence": bool(getattr(result, 'volume_divergence', False)),
            "signal_type": str(getattr(result, 'signal_type', 'none')),
            "signal_strength": str(getattr(result, 'signal_strength', 'moderate')),
            "atr": float(getattr(result, 'atr', 0)),
            "extension_atr": float(getattr(result, 'extension_atr', 0)),
            "has_rejection": bool(getattr(result, 'has_rejection', False)),
            "order_flow": order_flow
        }

        # Fibonacci retracement levels
        try:
            fib_period_days = {"day": 5, "swing": 20, "position": 60, "investment": 120}
            fib_days = fib_period_days.get(vp_period.lower(), 20)
            df_fib = None
            if scanner:
                df_fib = scanner._get_candles(symbol, "D", fib_days)
            if df_fib is None or len(df_fib) < 5:
                if get_bars:
                    fib_period_yf = "6mo" if fib_days <= 120 else "1y"
                    df_fib = get_bars(symbol, period=fib_period_yf, interval="1d")
                    if df_fib is not None and not df_fib.empty:
                        df_fib.columns = [c.lower() for c in df_fib.columns]
                    else:
                        df_fib = None
            if df_fib is not None and len(df_fib) >= 5:
                df_fib = df_fib.tail(fib_days)
                cp = float(df_fib['close'].iloc[-1])
                valid_lows = df_fib['low'][df_fib['low'] > cp * 0.5]
                valid_highs = df_fib['high'][df_fib['high'] < cp * 2.0]
                swing_high = float(valid_highs.max()) if len(valid_highs) > 0 else float(df_fib['high'].iloc[-5:].max())
                swing_low = float(valid_lows.min()) if len(valid_lows) > 0 else float(df_fib['low'].iloc[-5:].min())
                fib_range = swing_high - swing_low

                if fib_range > swing_high * 0.20:
                    swing_high = float(df_fib['high'].quantile(0.95))
                    swing_low = float(df_fib['low'].quantile(0.05))
                    fib_range = swing_high - swing_low

                high_idx = df_fib['high'].idxmax()
                low_idx = df_fib['low'].idxmin()
                high_pos = list(df_fib.index).index(high_idx) if high_idx in df_fib.index else len(df_fib) - 1
                low_pos = list(df_fib.index).index(low_idx) if low_idx in df_fib.index else 0
                uptrend = high_pos > low_pos

                if fib_range < swing_high * 0.01:
                    fib_range = swing_high * 0.05

                bull_fib_236 = swing_low + (fib_range * 0.236)
                bull_fib_382 = swing_low + (fib_range * 0.382)
                bull_fib_500 = swing_low + (fib_range * 0.500)
                bull_fib_618 = swing_low + (fib_range * 0.618)
                bull_fib_786 = swing_low + (fib_range * 0.786)
                bear_fib_236 = swing_high - (fib_range * 0.236)
                bear_fib_382 = swing_high - (fib_range * 0.382)
                bear_fib_500 = swing_high - (fib_range * 0.500)
                bear_fib_618 = swing_high - (fib_range * 0.618)
                bear_fib_786 = swing_high - (fib_range * 0.786)

                response["fib_levels"] = {
                    "swing_high": swing_high, "swing_low": swing_low,
                    "lookback_days": fib_days,
                    "trend": "UPTREND" if uptrend else "DOWNTREND",
                    "bull_fib_236": bull_fib_236, "bull_fib_382": bull_fib_382,
                    "bull_fib_500": bull_fib_500, "bull_fib_618": bull_fib_618,
                    "bull_fib_786": bull_fib_786,
                    "bear_fib_236": bear_fib_236, "bear_fib_382": bear_fib_382,
                    "bear_fib_500": bear_fib_500, "bear_fib_618": bear_fib_618,
                    "bear_fib_786": bear_fib_786,
                    "fib_236": bear_fib_236, "fib_382": bear_fib_382,
                    "fib_500": bear_fib_500, "fib_618": bear_fib_618,
                    "fib_786": bear_fib_786
                }

                # Fib position
                if uptrend:
                    if cp >= swing_high: response["fib_position"] = "Above swing high (extended)"
                    elif cp >= bull_fib_786: response["fib_position"] = "Above Bull 78.6% (strong uptrend)"
                    elif cp >= bull_fib_618: response["fib_position"] = "Above Bull 61.8% (healthy uptrend)"
                    elif cp >= bull_fib_500: response["fib_position"] = "Bull 50%-61.8% GOLDEN ZONE (best long entry)"
                    elif cp >= bull_fib_382: response["fib_position"] = "Bull 38.2%-50% (pullback entry zone)"
                    elif cp >= bull_fib_236: response["fib_position"] = "Bull 23.6%-38.2% (shallow pullback)"
                    else: response["fib_position"] = "Below Bull 23.6% (trend may be broken)"
                else:
                    if cp <= swing_low: response["fib_position"] = "Below swing low (extended)"
                    elif cp <= bear_fib_786: response["fib_position"] = "Below Bear 78.6% (strong downtrend)"
                    elif cp <= bear_fib_618: response["fib_position"] = "Below Bear 61.8% (healthy downtrend)"
                    elif cp <= bear_fib_500: response["fib_position"] = "Bear 50%-61.8% GOLDEN ZONE (best short entry)"
                    elif cp <= bear_fib_382: response["fib_position"] = "Bear 38.2%-50% (bounce entry zone)"
                    elif cp <= bear_fib_236: response["fib_position"] = "Bear 23.6%-38.2% (shallow bounce)"
                    else: response["fib_position"] = "Above Bear 23.6% (trend may be reversing)"

                # Fib score adjustment
                fib_pos = response.get("fib_position", "")
                fib_bull_adj = fib_bear_adj = 0
                signal_str = str(response.get("signal", "")).upper()
                is_long_signal = "LONG" in signal_str
                is_short_signal = "SHORT" in signal_str
                is_counter_trend = (not uptrend and is_long_signal) or (uptrend and is_short_signal)

                adj_map = {
                    "GOLDEN ZONE": 18 if is_counter_trend else 15,
                    "38.2%-50%": 10,
                    "23.6%-38.2%": 5,
                    "78.6%": 3 if is_counter_trend else 5,
                }
                for label, adj in adj_map.items():
                    if label in fib_pos and "GOLDEN" not in fib_pos.replace(label, ""):
                        if is_counter_trend:
                            if is_long_signal: fib_bull_adj = adj
                            else: fib_bear_adj = adj
                        elif uptrend: fib_bull_adj = adj
                        else: fib_bear_adj = adj
                        break
                if "61.8%" in fib_pos and "GOLDEN" not in fib_pos:
                    adj = 8
                    if is_counter_trend:
                        if is_long_signal: fib_bull_adj = adj
                        else: fib_bear_adj = adj
                    elif uptrend: fib_bull_adj = adj
                    else: fib_bear_adj = adj

                if fib_bull_adj or fib_bear_adj:
                    response["bull_score"] = max(0, min(100, response["bull_score"] + fib_bull_adj))
                    response["bear_score"] = max(0, min(100, response["bear_score"] + fib_bear_adj))
                    response["fib_score_adj"] = {"bull": fib_bull_adj, "bear": fib_bear_adj}
                    new_bull, new_bear = response["bull_score"], response["bear_score"]
                    gap = abs(new_bull - new_bear)
                    winning = max(new_bull, new_bear)
                    response["confidence"] = min(95, 40 + (winning * 0.5) + (gap * 0.1))
                    total = new_bull + new_bear
                    if total > 0:
                        response["high_prob"] = round(max(new_bull, new_bear) / total * 100, 1)
                        response["low_prob"] = round(min(new_bull, new_bear) / total * 100, 1)
                    response["notes"].append(f"Fib adj: bull {fib_bull_adj:+d}, bear {fib_bear_adj:+d} ({fib_pos})")

                # VP + Fib confluence
                active_fibs = {
                    "23.6%": bull_fib_236 if uptrend else bear_fib_236,
                    "38.2%": bull_fib_382 if uptrend else bear_fib_382,
                    "50%": bull_fib_500 if uptrend else bear_fib_500,
                    "61.8%": bull_fib_618 if uptrend else bear_fib_618,
                    "78.6%": bull_fib_786 if uptrend else bear_fib_786,
                }
                confluences = []
                for fn, fv in active_fibs.items():
                    if vah > 0 and abs(vah - fv) / vah < 0.015:
                        confluences.append(f"VAH ~ Fib {fn} at ${vah:.2f}")
                    if poc > 0 and abs(poc - fv) / poc < 0.015:
                        confluences.append(f"POC ~ Fib {fn} at ${poc:.2f}")
                    if val > 0 and abs(val - fv) / val < 0.015:
                        confluences.append(f"VAL ~ Fib {fn} at ${val:.2f}")
                if confluences:
                    response["fib_confluence"] = confluences
                    response["notes"].append(f"Fib Confluence: {'; '.join(confluences)}")

                # Trade scenarios
                if current_price > 0 and vah > 0 and val > 0 and poc > 0:
                    vp_range = vah - val if vah > val else 1
                    scen_atr = _atr if _atr > 0 else vp_range * 0.3
                    long_entry_low, long_entry_high = val, poc
                    long_mid = (long_entry_low + long_entry_high) / 2
                    long_stop = long_mid - (scen_atr * 0.5)
                    long_target1 = max(vah, long_mid + scen_atr)
                    long_target2 = max(bull_fib_786 if bull_fib_786 > long_target1 else swing_high, long_mid + scen_atr * 2)
                    long_risk = long_mid - long_stop
                    long_reward = long_target1 - long_mid
                    long_rr = long_reward / long_risk if long_risk > 0 else 0

                    short_entry_low, short_entry_high = poc, vah
                    short_mid = (short_entry_low + short_entry_high) / 2
                    short_stop = short_mid + (scen_atr * 0.5)
                    short_target1 = min(val, short_mid - scen_atr)
                    short_target2 = min(bear_fib_618 if bear_fib_618 < short_target1 else swing_low, short_mid - scen_atr * 2)
                    short_risk = short_stop - short_mid
                    short_reward = short_mid - short_target1
                    short_rr = short_reward / short_risk if short_risk > 0 else 0

                    agg_long_stop = current_price - (scen_atr * 0.5)
                    agg_short_stop = current_price + (scen_atr * 0.5)
                    long_agg_valid = current_price > long_stop
                    short_agg_valid = current_price < short_stop
                    agg_long_rr = (long_target1 - current_price) / (current_price - agg_long_stop) if current_price > agg_long_stop else 0
                    agg_short_rr = (current_price - short_target1) / (agg_short_stop - current_price) if agg_short_stop > current_price else 0
                    agg_risk_pct = (scen_atr * 0.5 / current_price) * 100 if current_price > 0 else 0

                    response["trade_scenarios"] = {
                        "long": {
                            "entry_zone": [f"{long_entry_low:.2f}", f"{long_entry_high:.2f}"],
                            "stop_loss": long_stop, "target": long_target1, "target2": long_target2,
                            "r_r_ratio": f"{long_rr:.1f}:1",
                            "aggressive_entry": current_price if long_agg_valid else None,
                            "aggressive_stop": agg_long_stop, "aggressive_valid": long_agg_valid,
                            "aggressive_rr": f"{agg_long_rr:.1f}:1" if long_agg_valid else None,
                            "aggressive_risk_pct": round(agg_risk_pct, 1) if long_agg_valid else None
                        },
                        "short": {
                            "entry_zone": [f"{short_entry_low:.2f}", f"{short_entry_high:.2f}"],
                            "stop_loss": short_stop, "target": short_target1, "target2": short_target2,
                            "r_r_ratio": f"{short_rr:.1f}:1",
                            "aggressive_entry": current_price if short_agg_valid else None,
                            "aggressive_stop": agg_short_stop, "aggressive_valid": short_agg_valid,
                            "aggressive_rr": f"{agg_short_rr:.1f}:1" if short_agg_valid else None,
                            "aggressive_risk_pct": round(agg_risk_pct, 1) if short_agg_valid else None
                        },
                        "decision_point": {
                            "bull_trigger": vah + (vp_range * 0.02),
                            "bear_trigger": val - (vp_range * 0.02),
                            "current_price": current_price
                        }
                    }
        except Exception as e:
            print(f"Fib/Trade scenarios error: {e}")

        if entry_signal:
            response["entry_signal"] = entry_signal

        # AI commentary
        if with_ai and anthropic_client:
            try:
                response["ai_commentary"] = _get_ai_commentary(response, symbol)
                response["analysis_source"] = "claude"
            except Exception as e:
                print(f"AI commentary error: {e}")
                response["analysis_source"] = "none"

        # Cache non-AI
        if not with_ai:
            analysis_cache.set(cache_key, response)

        return _safe_dict(response)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"analyze_live error for {symbol}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


def _get_ai_commentary(analysis: dict, symbol: str) -> str:
    """Generate AI commentary using Claude"""
    if not anthropic_client:
        return ""
    try:
        prompt = f"""Analyze this stock data for {symbol} and give a concise 2-3 sentence trading outlook:
Signal: {analysis.get('signal')} | Bull: {analysis.get('bull_score')}/100 | Bear: {analysis.get('bear_score')}/100
Price: ${analysis.get('current_price', 0):.2f} | VAH: ${analysis.get('vah', 0):.2f} | POC: ${analysis.get('poc', 0):.2f} | VAL: ${analysis.get('val', 0):.2f}
RSI: {analysis.get('rsi', 50):.1f} | VWAP Zone: {analysis.get('vwap_zone')} | Position: {analysis.get('position')}
Fib Position: {analysis.get('fib_position', 'N/A')}
Notes: {'; '.join(analysis.get('notes', [])[:5])}
Be specific about price levels and actionable."""
        msg = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text
    except Exception as e:
        print(f"Claude AI error: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# MTF (Multi-Timeframe) ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/analyze/live/mtf/{symbol}")
async def analyze_live_mtf(symbol: str):
    """Multi-timeframe analysis with live data"""
    try:
        scanner = get_finnhub_scanner()
        result = scanner.analyze_mtf(symbol.upper())

        if not result:
            raise HTTPException(status_code=404, detail=f"Could not analyze {symbol}")

        # Get real-time price
        current_price = None
        try:
            quote = scanner.get_quote(symbol.upper())
            if quote and quote.get('current'):
                current_price = float(quote['current'])
        except Exception:
            pass

        # Convert to JSON
        tf_results = {}
        for tf, r in result.timeframe_results.items():
            tf_results[tf] = {
                "signal": r.signal,
                "signal_emoji": r.signal_emoji,
                "bull_score": r.bull_score,
                "bear_score": r.bear_score,
                "confidence": r.confidence,
                "position": r.position,
                "rsi_zone": r.rsi_zone
            }

        return {
            "symbol": result.symbol,
            "current_price": current_price,
            "timestamp": str(result.timestamp) if result.timestamp else None,
            "dominant_signal": result.dominant_signal,
            "signal_emoji": result.signal_emoji,
            "confluence_pct": float(result.confluence_pct) if result.confluence_pct else 0,
            "weighted_bull": float(result.weighted_bull) if result.weighted_bull else 0,
            "weighted_bear": float(result.weighted_bear) if result.weighted_bear else 0,
            "high_prob": float(result.high_prob) if result.high_prob else 0,
            "low_prob": float(result.low_prob) if result.low_prob else 0,
            "timeframes": tf_results,
            "notes": list(result.notes) if result.notes else []
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"MTF analysis error for {symbol}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"MTF analysis error: {str(e)}")


@app.post("/api/analyze/live/mtf/{symbol}/ai")
async def analyze_mtf_with_ai(
    symbol: str,
    trade_tf: str = Query("swing", description="Trade timeframe: intraday, swing, position, longterm, investment"),
    entry_signal: str = Query(None, description="Entry signal e.g. 'failed_breakout:short'")
):
    """Generate AI trade plan using full MTF context"""
    if not anthropic_client:
        raise HTTPException(status_code=400, detail="AI API key not set")

    scanner = get_finnhub_scanner()
    result = scanner.analyze_mtf(symbol.upper())
    if not result:
        raise HTTPException(status_code=404, detail=f"Could not analyze {symbol}")

    # Timeframe config — includes card TF names (scalp, daytrade) + legacy names
    tf_config = {
        "scalp":      {"days": 1, "label": "SCALP (15 min – 2 hrs)",  "stop_mult": 0.15, "target_mult": 0.3,  "hold": "15 min – 2 hours", "candle_res": "5",  "candle_bars": 50},
        "intraday":   {"days": 1, "label": "SAME DAY (Intraday)",     "stop_mult": 0.3,  "target_mult": 0.5,  "hold": "1-4 hours",        "candle_res": "15", "candle_bars": 40},
        "daytrade":   {"days": 2, "label": "1-2 DAY TRADE",           "stop_mult": 0.4,  "target_mult": 0.7,  "hold": "1-2 days",         "candle_res": "30", "candle_bars": 30},
        "swing":      {"days": 5, "label": "3-5 DAY SWING",           "stop_mult": 0.5,  "target_mult": 1.0,  "hold": "3-5 days",         "candle_res": "60", "candle_bars": 20},
        "position":   {"days": 21, "label": "2-4 WEEK POSITION",      "stop_mult": 1.0,  "target_mult": 2.0,  "hold": "2-4 weeks",        "candle_res": "D",  "candle_bars": 30},
        "longterm":   {"days": 60, "label": "1-3 MONTH SETUP",        "stop_mult": 2.0,  "target_mult": 4.0,  "hold": "1-3 months",       "candle_res": "D",  "candle_bars": 60},
        "investment": {"days": 180, "label": "6+ MONTH INVESTMENT",    "stop_mult": 5.0,  "target_mult": 10.0, "hold": "6+ months",        "candle_res": "D",  "candle_bars": 120}
    }
    config = tf_config.get(trade_tf, tf_config["swing"])

    # Calculate VP levels from candle data — resolution varies by TF
    candle_res = config.get("candle_res", "60")
    candle_bars = config.get("candle_bars", 20)
    df = scanner._get_candles(symbol.upper(), candle_res, candle_bars)
    poc, vah, val, vwap, rsi, rvol, volume_trend, current_price = 0, 0, 0, 0, 50, 1.0, "neutral", 0
    if df is not None and len(df) >= 5:
        poc, vah, val = scanner.calc.calculate_volume_profile(df)
        vwap = scanner.calc.calculate_vwap(df)
        rsi = scanner.calc.calculate_rsi(df)
        rvol = scanner.calc.calculate_relative_volume(df)
        volume_trend = scanner.calc.calculate_volume_trend(df)
        current_price = float(df['close'].iloc[-1])
        quote = scanner.get_quote(symbol.upper())
        if quote and quote.get('current'):
            current_price = float(quote['current'])

    # Fib levels — lookback scales with TF
    fib_text = ""
    fib_days = min(config["days"] * 3, 60) if config["days"] <= 5 else 15
    fib_days = max(fib_days, 5)  # at least 5 days
    try:
        df_daily = scanner._get_candles(symbol.upper(), "D", fib_days)
        if df_daily is not None and len(df_daily) >= 5:
            swing_high = float(df_daily['high'].max())
            swing_low = float(df_daily['low'].min())
            fib_range = swing_high - swing_low
            fib_236 = swing_high - (fib_range * 0.236)
            fib_382 = swing_high - (fib_range * 0.382)
            fib_500 = swing_high - (fib_range * 0.500)
            fib_618 = swing_high - (fib_range * 0.618)
            fib_786 = swing_high - (fib_range * 0.786)
            fib_text = (
                f"Fib ({fib_days}d): High ${swing_high:.2f} Low ${swing_low:.2f} | "
                f"23.6% ${fib_236:.2f} | 38.2% ${fib_382:.2f} | 50% ${fib_500:.2f} | "
                f"61.8% ${fib_618:.2f} | 78.6% ${fib_786:.2f}"
            )
    except Exception:
        pass

    # ATR for scaling
    atr_daily = 0
    try:
        if df is not None and len(df) >= 14:
            _high = df['high']
            _low = df['low']
            _prev = df['close'].shift(1)
            import pandas as _pd
            _tr = _pd.concat([_high - _low, (_high - _prev).abs(), (_low - _prev).abs()], axis=1).max(axis=1)
            atr_daily = float(_tr.rolling(14).mean().iloc[-1])
    except Exception:
        pass
    if atr_daily <= 0:
        atr_daily = (vah - val) * 0.3 if vah > val else current_price * 0.015 if current_price > 0 else 1
    stop_distance = atr_daily * config["stop_mult"]
    target_distance = atr_daily * config["target_mult"]

    # Leading direction from scores
    bull_total = result.weighted_bull or 0
    bear_total = result.weighted_bear or 0
    score_diff = bull_total - bear_total
    if entry_signal:
        parts = entry_signal.split(':')
        leading_direction = parts[1].upper() if len(parts) > 1 else ("LONG" if score_diff >= 0 else "SHORT")
        leading_reason = f"VP Entry Signal: {parts[0].replace('_', ' ').title()}" if parts else "Entry signal"
    elif score_diff > 10:
        leading_direction = "LONG"
        leading_reason = f"Bull/Bear Score: {bull_total:.0f} vs {bear_total:.0f}"
    elif score_diff < -10:
        leading_direction = "SHORT"
        leading_reason = f"Bull/Bear Score: {bull_total:.0f} vs {bear_total:.0f}"
    elif current_price > poc and poc > 0:
        leading_direction = "SHORT"
        leading_reason = f"Price above POC (${poc:.2f})"
    elif poc > 0:
        leading_direction = "LONG"
        leading_reason = f"Price below POC (${poc:.2f})"
    else:
        leading_direction = "LONG" if "LONG" in str(result.dominant_signal) else "SHORT"
        leading_reason = f"MTF Dominant: {result.dominant_signal}"

    # Build TF summary
    tf_summary = []
    for tf, r in result.timeframe_results.items():
        tf_summary.append(f"{tf}: {r.signal} (Bull:{r.bull_score}, Bear:{r.bear_score})")

    prompt = f"""ANALYZE MTF: {symbol.upper()} @ ${current_price:.2f} | {config["label"]}
VP RESOLUTION: {candle_res} candles ({candle_bars} bars) — levels reflect THIS timeframe

LEADING DIRECTION: {leading_direction} ({leading_reason})
MTF CONFLUENCE: {result.confluence_pct}% | Dominant: {result.dominant_signal}
HIGH PROB: {result.high_prob:.0f}% | LOW PROB: {result.low_prob:.0f}%
Bull: {result.weighted_bull:.0f} | Bear: {result.weighted_bear:.0f}

VOLUME: RVOL {rvol:.1f}x | Trend: {volume_trend}
TIMEFRAMES: {' | '.join(tf_summary)}

LEVELS: VAH ${vah:.2f} | POC ${poc:.2f} | VAL ${val:.2f} | VWAP ${vwap:.2f} | RSI {rsi:.0f}
{fib_text}

ATR: ${atr_daily:.2f} | Stop: ${stop_distance:.2f} ({config['stop_mult']}x ATR) | Target: ${target_distance:.2f} ({config['target_mult']}x ATR)
Hold: {config['hold']}
IMPORTANT: Size entries, stops, and targets for a {config['label']} trade — NOT a swing/position trade.

NOTES: {'; '.join(result.notes[:3]) if result.notes else 'None'}

Give BOTH long and short setups with entry zones, stops, targets, R:R math.
Lead with {leading_direction}. End with a VERDICT picking the preferred direction."""

    system_prompt = f"""You are an expert MTF trading analyst planning a {config['label']} trade.
Output FULL SETUPS for BOTH directions using this EXACT format (including emojis):

🟢 LONG SETUP
GRADE: [A+ to F]
CONVICTION: [1-10]/10
PROBABILITY: [X]% [LABEL]
ENTRY ZONE: $[low] – $[high]
STOP: $[price]
T1: $[price]
T2: $[price]
R:R = [X:X]
EV: $[X] per $100 risked → [POSITIVE/NEGATIVE]
TRIGGER: [what confirms entry]
INVALID IF: [what kills the trade]
WHY LONG: [1-2 sentence reason]
SIZE: [X]R
HOLD: [duration]

🔴 SHORT SETUP
GRADE: [A+ to F]
CONVICTION: [1-10]/10
PROBABILITY: [X]% [LABEL]
ENTRY ZONE: $[low] – $[high]
STOP: $[price]
T1: $[price]
T2: $[price]
R:R = [X:X]
EV: $[X] per $100 risked → [POSITIVE/NEGATIVE]
TRIGGER: [what confirms entry]
INVALID IF: [what kills the trade]
WHY SHORT: [1-2 sentence reason]
SIZE: [X]R
HOLD: [duration]

⚖️ VERDICT
PREFERRED: [LONG or SHORT]
KEY LEVEL: $[price]
[1-2 sentence summary]

Use VP levels and Fib levels for entries/stops/targets.
CRITICAL: All entries, stops, and targets MUST be sized for a {config['hold']} hold period.
For scalps/intraday: tighter entries, closer stops, smaller targets.
For position/longterm: wider entries, wider stops, larger targets.
Follow the format EXACTLY — no extra sections, no missing fields."""

    try:
        msg = await asyncio.wait_for(
            asyncio.to_thread(
                anthropic_client.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=1200,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            ),
            timeout=45  # 45 second max for AI call
        )
        return {
            "symbol": symbol.upper(),
            "ai_commentary": msg.content[0].text,
            "high_prob": result.high_prob,
            "low_prob": result.low_prob,
            "confluence": result.confluence_pct,
            "dominant_signal": result.dominant_signal,
            "trade_timeframe": config["label"],
            "leading_direction": leading_direction,
            "leading_reason": leading_reason,
            "bull_score": result.weighted_bull,
            "bear_score": result.weighted_bear,
            "rvol": rvol,
            "volume_trend": volume_trend,
            "vah": vah, "poc": poc, "val": val, "vwap": vwap, "rsi": rsi,
            "current_price": current_price
        }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="AI analysis timed out (45s). Try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")


@app.get("/api/scan/live")
async def scan_live(watchlist: str = "Tech Giants", limit: int = 20):
    try:
        scanner = get_finnhub_scanner()
        wl = watchlist_mgr.get_watchlist(watchlist)
        if not wl:
            raise HTTPException(status_code=404, detail=f"Watchlist '{watchlist}' not found")
        symbols = [s.symbol for s in wl.symbols][:limit]
        results = []
        for sym in symbols:
            try:
                analysis = scanner.analyze(sym)
                if analysis:
                    results.append(_safe_dict(analysis))
            except Exception:
                continue
        return {"watchlist": watchlist, "count": len(results), "results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# BUFFETT BLOOD SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

_buffett_cache: dict = {}        # key -> (timestamp, result_dict)
_BUFFETT_TTL = 300               # 5 min
_buffett_hits = 0                # track cache effectiveness

@app.get("/api/buffett-scan")
async def buffett_scan(tickers: str = "", preset: str = ""):
    """Buffett Blood Scanner — scan tickers for value + crisis metrics"""
    global _buffett_hits
    try:
        from buffett_scanner import async_scan_tickers, PRESETS
        import time as _t

        if preset and preset in PRESETS:
            symbols = PRESETS[preset]
        elif tickers:
            symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        else:
            raise HTTPException(status_code=400, detail="Provide tickers or preset param")

        if len(symbols) > 30:
            raise HTTPException(status_code=400, detail="Max 30 tickers per scan")

        # Server-level cache — keyed on sorted ticker list
        cache_key = ",".join(sorted(symbols))
        now = _t.time()
        entry = _buffett_cache.get(cache_key)
        if entry and (now - entry[0]) < _BUFFETT_TTL:
            _buffett_hits += 1
            data = entry[1]
            data["meta"]["cached"] = True
            data["meta"]["cache_age"] = round(now - entry[0], 1)
            data["meta"]["total_hits"] = _buffett_hits
            return data

        t0 = _t.time()
        data = await async_scan_tickers(symbols)
        elapsed = _t.time() - t0

        # Store in server-level cache
        _buffett_cache[cache_key] = (now, data)

        data["meta"]["cached"] = False
        data["meta"]["scan_time"] = round(elapsed, 2)
        data["meta"]["total_hits"] = _buffett_hits
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONS FLOW SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/options-flow")
async def options_flow_scan(tickers: str = "", preset: str = ""):
    """Options Flow Scanner — scan tickers for unusual options activity via Polygon"""
    try:
        from options_flow_scanner import async_scan_tickers as async_options_scan, PRESETS as OPT_PRESETS

        if preset and preset in OPT_PRESETS:
            symbols = OPT_PRESETS[preset]
        elif tickers:
            symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        else:
            raise HTTPException(status_code=400, detail="Provide tickers or preset param")

        if len(symbols) > 12:
            raise HTTPException(status_code=400, detail="Max 12 tickers per scan")

        data = await async_options_scan(symbols)
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Options Flow Stream — SSE Real-Time Flow ──

@app.post("/api/options-flow/stream/start")
async def start_options_flow_stream(tickers: str = "", preset: str = ""):
    """Start real-time options flow streaming for given tickers."""
    if not flow_stream_available or not flow_stream:
        raise HTTPException(status_code=400, detail="Options flow stream module not available")
    from options_flow_scanner import PRESETS as OPT_PRESETS
    if preset and preset in OPT_PRESETS:
        symbols = OPT_PRESETS[preset]
    elif tickers:
        symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    else:
        raise HTTPException(status_code=400, detail="Provide tickers or preset param")
    if len(symbols) > 8:
        raise HTTPException(status_code=400, detail="Max 8 tickers for live stream (API rate limits)")
    flow_stream.set_tickers(symbols)
    if not flow_stream.is_running:
        flow_stream.start_background()
    return {"status": "ok", "message": f"Flow stream started for {len(symbols)} tickers", "tickers": symbols, "stream": flow_stream.get_status()}


@app.post("/api/options-flow/stream/stop")
async def stop_options_flow_stream():
    """Stop the options flow stream."""
    if not flow_stream_available or not flow_stream:
        raise HTTPException(status_code=400, detail="Flow stream not available")
    flow_stream.stop()
    return {"status": "ok", "message": "Flow stream stopped"}


@app.get("/api/options-flow/stream/status")
async def options_flow_stream_status():
    """Get current flow stream status."""
    if not flow_stream_available or not flow_stream:
        return {"available": False, "running": False}
    status = flow_stream.get_status()
    status["available"] = True
    return status


@app.get("/api/options-flow/stream/events")
async def options_flow_sse(request: Request):
    """Server-Sent Events — streams live options flow events to the browser."""
    if not flow_stream_available or not flow_stream:
        raise HTTPException(status_code=400, detail="Flow stream not available")
    q = flow_stream.subscribe()
    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'connected', 'status': flow_stream.get_status()})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=25.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            flow_stream.unsubscribe(q)
    from starlette.responses import StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ═══════════════════════════════════════════════════════════════════════════════
# WAR ROOM
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/war-room")
async def war_room_scan(tickers: str = "", preset: str = ""):
    """War Room — Pre-market extension DNA analysis via Polygon intraday bars"""
    try:
        from war_room import async_run_war_room, PRESETS as WR_PRESETS

        if preset and preset in WR_PRESETS:
            symbols = WR_PRESETS[preset]
        elif tickers:
            symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        else:
            raise HTTPException(status_code=400, detail="Provide tickers or preset param")

        if len(symbols) > 15:
            raise HTTPException(status_code=400, detail="Max 15 tickers per scan")

        data = await async_run_war_room(symbols)
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level singleton — cache persists between requests (5-min TTL)
from regime_scanner import RegimeScanner as _RegimeScanner
_regime_scanner = _RegimeScanner()

_regime_cache: dict = {}         # key -> (timestamp, response_dict)
_REGIME_TTL = 300                # 5 min

@app.get("/api/regime-scan")
async def regime_scan(tickers: str = "", days: int = 30):
    """Regime Scanner — cross-gate strategy analysis for one or more symbols"""
    try:
        import time as _t
        if not tickers:
            raise HTTPException(status_code=400, detail="Provide tickers param (comma-separated)")
        symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if len(symbols) > 15:
            raise HTTPException(status_code=400, detail="Max 15 tickers per scan")
        days = min(max(days, 5), 90)

        # Server-level cache — keyed on sorted tickers + days
        cache_key = f"{','.join(sorted(symbols))}:{days}"
        now = _t.time()
        entry = _regime_cache.get(cache_key)
        if entry and (now - entry[0]) < _REGIME_TTL:
            data = entry[1]
            data["_cache"] = {"hit": True, "age": round(now - entry[0], 1)}
            return _safe_json_response(data)

        t0 = _t.time()
        scanner = _regime_scanner

        if len(symbols) == 1:
            result = await asyncio.to_thread(scanner.scan, symbols[0], days)
            resp = result.to_dict()
        else:
            results = await asyncio.to_thread(scanner.scan_watchlist, symbols, days)
            comparison = scanner.compare_watchlist(results)
            result_dicts = {}
            for sym, r in results.items():
                try:
                    result_dicts[sym] = r.to_dict()
                except Exception as e2:
                    logger.error("to_dict failed for %s: %s", sym, e2)
                    result_dicts[sym] = {"symbol": sym, "error": str(e2), "days_analyzed": 0}
            resp = {"comparison": comparison, "results": result_dicts}

        elapsed = _t.time() - t0
        resp["_cache"] = {"hit": False, "scan_time": round(elapsed, 2)}
        _regime_cache[cache_key] = (now, resp)
        return _safe_json_response(resp)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error("Regime scan error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/regime-levels/{symbol}")
async def regime_levels(symbol: str):
    """Get ATR-scaled strategy levels for a symbol"""
    try:
        data = await asyncio.to_thread(_regime_scanner.get_strategy_levels, symbol.upper())
        if "error" in data:
            raise HTTPException(status_code=404, detail=data["error"])
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# KEY MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/set-key")
async def set_finnhub_key(api_key: str):
    global finnhub_scanner
    os.environ["FINNHUB_API_KEY"] = api_key
    try:
        finnhub_scanner = FinnhubScanner(api_key)
        return {"status": "ok", "message": "Finnhub key set"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/set-polygon-key")
async def set_polygon_key(api_key: str):
    os.environ["POLYGON_API_KEY"] = api_key
    return {"status": "ok", "message": "Polygon key set"}


@app.post("/api/set-openai-key")
async def set_openai_key(api_key: str):
    """Set Anthropic API key (legacy endpoint name kept for compatibility)"""
    global anthropic_client
    os.environ["ANTHROPIC_API_KEY"] = api_key
    if not anthropic_available:
        raise HTTPException(status_code=400, detail="anthropic package not installed")
    try:
        anthropic_client = anthropic.Anthropic(api_key=api_key)
        if set_range_openai:
            set_range_openai(anthropic_client)
        return {"status": "ok", "message": "Claude AI enabled"}
    except Exception as e:
        anthropic_client = None
        raise HTTPException(status_code=400, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# ALERTS & TRADES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/alerts")
async def get_alerts():
    return chart_system.get_alerts()


@app.post("/api/alerts")
async def create_alert(alert: AlertRequest):
    chart_system.add_alert(alert.symbol, alert.level, alert.direction, alert.action, alert.note)
    return {"status": "ok"}


@app.delete("/api/alerts")
async def delete_alert(symbol: str, level: float):
    chart_system.remove_alert(symbol, level)
    return {"status": "ok"}


@app.get("/api/trades")
async def get_trades():
    return chart_system.get_pending_trades()


@app.get("/api/trades/analytics")
async def get_trade_analytics(user_id: str = None, days: int = 90):
    """Comprehensive trading analytics"""
    try:
        from journal_analytics import compute_journal_analytics

        trades = []
        storage = "local"

        if user_id and firestore_available:
            fs = get_firestore()
            if fs.is_available():
                trades = fs.get_trades(user_id)
                storage = "firestore"

        if not trades and chart_system:
            local_trades = chart_system.tracker.trades if hasattr(chart_system, 'tracker') else []
            trades = [asdict(t) for t in local_trades]
            storage = "local"

        analytics = compute_journal_analytics(trades, days=days)
        analytics["storage"] = storage
        return analytics

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trades/analytics/report")
async def get_trade_analytics_report(user_id: str = None, days: int = 90):
    """Get a text-format trading performance report"""
    try:
        from journal_analytics import compute_journal_analytics, generate_analytics_report

        trades = []
        if user_id and firestore_available:
            fs = get_firestore()
            if fs.is_available():
                trades = fs.get_trades(user_id)
        if not trades and chart_system:
            local_trades = chart_system.tracker.trades if hasattr(chart_system, 'tracker') else []
            trades = [asdict(t) for t in local_trades]

        analytics = compute_journal_analytics(trades, days=days)
        report = generate_analytics_report(analytics)
        return {"report": report, "analytics": analytics}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMING (WebSocket passthrough)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/streaming/start")
async def start_streaming(tickers: str = ""):
    if not streaming_available or not streaming_manager:
        raise HTTPException(status_code=400, detail="Streaming not available")
    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not symbols:
        raise HTTPException(status_code=400, detail="No tickers provided")
    streaming_manager.add_symbols(symbols)
    if not streaming_manager.streamer or not streaming_manager.streamer.connected:
        streaming_manager.start()
    return {"status": "ok", "symbols": symbols}


@app.post("/api/streaming/stop")
async def stop_streaming():
    if streaming_available and streaming_manager:
        streaming_manager.stop()
    return {"status": "ok"}


@app.get("/api/streaming/status")
async def streaming_status():
    if not streaming_available or not streaming_manager:
        return {"available": False}
    return streaming_manager.get_status()


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — AI Kill Switch (org-scoped)
# ═══════════════════════════════════════════════════════════════════════════════

# Global fallback (for non-org users)
AI_KILL_SWITCH: bool = False
# Per-org kill state: { org_id: bool }
_org_ai_killed: dict = {}


def is_ai_killed_for_org(org_id: str = "personal") -> bool:
    """Check if AI is killed for a specific org.  Checks:
    1. Per-org admin disable (ai_enabled=False in Firestore config)
    2. Per-org runtime kill switch
    3. Global kill switch fallback"""
    if org_id in _org_ai_killed:
        return _org_ai_killed[org_id]
    return AI_KILL_SWITCH


@app.get("/api/config/ai-kill-switch")
async def get_ai_kill_switch(request: Request):
    """Get AI status. If user is authed, returns org-scoped status."""
    org_id = "personal"
    try:
        from auth_middleware import get_current_user
        user = await get_current_user(request)
        if user:
            org_id = user.org_id
            # Also check org admin config
            try:
                from admin_endpoints import _get_org_config
                config = _get_org_config(org_id)
                if config.get("ai_enabled") is False:
                    return {"killed": True, "org_id": org_id, "reason": "disabled_by_admin"}
            except Exception:
                pass
    except Exception:
        pass
    killed = is_ai_killed_for_org(org_id)
    return {"killed": killed, "enabled": not killed, "org_id": org_id}


@app.post("/api/config/ai-kill-switch")
async def set_ai_kill_switch(request: Request):
    """Toggle AI kill switch. Org-scoped if user is authed."""
    global AI_KILL_SWITCH
    body = await request.json()
    killed = body.get("killed", False)

    org_id = "personal"
    try:
        from auth_middleware import get_current_user
        user = await get_current_user(request)
        if user:
            org_id = user.org_id
    except Exception:
        pass

    if org_id and org_id != "personal":
        _org_ai_killed[org_id] = killed
    else:
        AI_KILL_SWITCH = killed

    return {"killed": killed, "org_id": org_id}


# ═══════════════════════════════════════════════════════════════════════════════
# API STATUS PAGE
# ═══════════════════════════════════════════════════════════════════════════════

_server_start_time = _time.time()

@app.get("/api/status")
async def api_status():
    """Platform status page — data source health, uptime, rate limits."""
    uptime_seconds = int(_time.time() - _server_start_time)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Check data source availability
    sources = {}
    try:
        import polygon_data
        sources["polygon"] = {"status": "connected", "type": "primary"}
    except Exception:
        sources["polygon"] = {"status": "unavailable", "type": "primary"}

    try:
        import yfinance
        sources["yfinance"] = {"status": "connected", "type": "fallback"}
    except Exception:
        sources["yfinance"] = {"status": "unavailable", "type": "fallback"}

    return {
        "status": "operational",
        "uptime": f"{hours}h {minutes}m {seconds}s",
        "uptime_seconds": uptime_seconds,
        "rate_limit": f"{RATE_LIMIT_PER_MINUTE}/min per IP",
        "data_sources": sources,
        "ai_global_killed": AI_KILL_SWITCH,
        "version": "2.0.0",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PAYMENTS / SUBSCRIPTION (stub if not available)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/payments/subscription")
async def get_subscription_status(request: Request):
    """Return subscription status — works even if auth/payments modules aren't loaded."""
    return {
        "status": "active",
        "tier": "premium",
        "message": "All features enabled",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MONITOR STATUS (stub)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/monitor/status")
async def monitor_status():
    return {"running": False, "message": "Trade monitor not loaded in base build"}


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURE REVERSAL DETECTOR ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/structure/reversals/{symbol}")
async def structure_reversals(symbol: str, min_confidence: float = 40.0):
    """
    Structure-based reversal detection using macro structure analysis.
    Detects: STRUCTURE_BREAK, MOMENTUM_EXHAUSTION, RANGE_EXTREME_REVERSAL,
    COMPRESSION_BREAKOUT, STRUCTURE_DIVERGENCE
    """
    if not structure_reversal_available:
        raise HTTPException(status_code=400, detail="Structure Reversal Detector not available")

    try:
        symbol = symbol.upper()
        df_daily = get_bars(symbol, period="3mo", interval="1d")
        df_weekly = get_bars(symbol, period="1y", interval="1wk")

        if df_daily.empty or len(df_daily) < 30:
            raise HTTPException(status_code=404, detail=f"Insufficient daily data for {symbol}")
        if df_weekly.empty or len(df_weekly) < 8:
            raise HTTPException(status_code=404, detail=f"Insufficient weekly data for {symbol}")

        df_daily.columns = [c.lower() for c in df_daily.columns]
        df_weekly.columns = [c.lower() for c in df_weekly.columns]

        # Run RangeWatcher analysis
        range_result = range_watcher_analyzer.analyze(df_daily, symbol=symbol)

        # Get weekly structure
        from chart_input_analyzer import RangeContext
        from finnhub_scanner_v2 import TechnicalCalculator

        range_context = TechnicalCalculator.calculate_range_structure(
            df_weekly=df_weekly,
            df_daily=df_daily,
            current_price=df_daily['close'].iloc[-1]
        )

        period_3d = range_result.periods.get(3)
        period_6d = range_result.periods.get(6)
        period_30d = range_result.periods.get(30)

        structure_ctx = StructureContext(
            weekly_trend=range_context.trend,
            weekly_hh=range_context.hh_count,
            weekly_hl=range_context.hl_count,
            weekly_lh=range_context.lh_count,
            weekly_ll=range_context.ll_count,
            weekly_close_position=range_context.weekly_close_position,
            period_3d_hh=period_3d.higher_highs if period_3d else False,
            period_3d_hl=period_3d.higher_lows if period_3d else False,
            period_3d_lh=period_3d.lower_highs if period_3d else False,
            period_3d_ll=period_3d.lower_lows if period_3d else False,
            period_6d_hh=period_6d.higher_highs if period_6d else False,
            period_6d_hl=period_6d.higher_lows if period_6d else False,
            period_6d_lh=period_6d.lower_highs if period_6d else False,
            period_6d_ll=period_6d.lower_lows if period_6d else False,
            period_30d_hh=period_30d.higher_highs if period_30d else False,
            period_30d_hl=period_30d.higher_lows if period_30d else False,
            period_30d_lh=period_30d.lower_highs if period_30d else False,
            period_30d_ll=period_30d.lower_lows if period_30d else False,
            current_price=range_result.current_price,
            position_in_3d_range=period_3d.position_in_range if period_3d else 0.5,
            position_in_30d_range=period_30d.position_in_range if period_30d else 0.5,
            compression_ratio=range_context.compression_ratio,
            nearest_resistance=period_30d.nearest_resistance if period_30d else range_result.current_price * 1.05,
            nearest_support=period_30d.nearest_support if period_30d else range_result.current_price * 0.95,
        )

        # Volume Profile confluence (optional)
        vp_data = None
        try:
            from volume_profile import calculate_volume_profile
            vp = calculate_volume_profile(df_daily.tail(20))
            vp_data = {'val': vp.get('val'), 'vah': vp.get('vah'), 'poc': vp.get('poc')}
        except Exception:
            pass

        structure_reversal_detector.min_confidence = min_confidence
        alerts = structure_reversal_detector.analyze(
            df=df_daily, structure_context=structure_ctx,
            symbol=symbol, vp_data=vp_data
        )

        alerts_dict = []
        for alert in alerts:
            alerts_dict.append({
                "alert_type": alert.alert_type.value,
                "severity": alert.severity.value,
                "confidence": round(alert.confidence, 1),
                "current_price": round(alert.current_price, 2),
                "trigger_level": round(alert.trigger_level, 2) if alert.trigger_level else None,
                "target_level": round(alert.target_level, 2) if alert.target_level else None,
                "stop_level": round(alert.stop_level, 2) if alert.stop_level else None,
                "description": alert.description,
                "timeframe": alert.timeframe,
                "signals": alert.signals,
                "structure_score": round(alert.structure_score, 1),
                "volume_score": round(alert.volume_score, 1),
                "vp_confluence": round(alert.vp_confluence, 1),
                "momentum_score": round(alert.momentum_score, 1),
                "range_position": round(alert.range_position, 1),
                "divergence_score": round(alert.divergence_score, 1),
            })

        return {
            "symbol": symbol,
            "alert_count": len(alerts),
            "alerts": alerts_dict,
            "structure_context": {
                "weekly_trend": structure_ctx.weekly_trend,
                "weekly_hh": structure_ctx.weekly_hh,
                "weekly_hl": structure_ctx.weekly_hl,
                "weekly_lh": structure_ctx.weekly_lh,
                "weekly_ll": structure_ctx.weekly_ll,
                "compression_ratio": round(structure_ctx.compression_ratio, 3),
                "position_in_30d_range": round(structure_ctx.position_in_30d_range * 100, 1),
            },
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Structure reversal analysis failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# STATIC FILE ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

_no_cache = {"Cache-Control": "no-cache, no-store, must-revalidate"}

@app.get("/")
async def serve_root():
    for f in ("public/index.html", "index.html"):
        if os.path.exists(f):
            return FileResponse(f, headers=_no_cache)
    return HTMLResponse("<h1>AnalysisGrid API</h1><p>Server is running.</p>")

@app.get("/v2")
async def serve_v2():
    for f in ("public/stock-options-scanner.html", "stock-options-scanner.html"):
        if os.path.exists(f):
            return FileResponse(f, headers=_no_cache)
    return FileResponse("stock-options-scanner.html", headers=_no_cache)

@app.get("/desk")
async def serve_desk():
    for f in ("public/trade-desk.html", "trade-desk.html"):
        if os.path.exists(f):
            return FileResponse(f, headers=_no_cache)
    return FileResponse("trade-desk.html", headers=_no_cache)

@app.get("/login.html")
async def serve_login():
    return FileResponse("login.html", headers=_no_cache)

# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCH BUILDER API
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/research/build")
async def research_build(request: Request):
    """Build a Picks & Shovels research report from config — parallel fetch."""
    import asyncio as _aio
    from pathlib import Path as _Path

    body = await request.json()
    cfg = body.get("config", {})
    mode = body.get("mode", "full")

    if not cfg.get("title"):
        raise HTTPException(status_code=400, detail="Config must have a title")

    # Get all investable tickers
    tickers = []
    for item in cfg.get("layer2", []) + cfg.get("layer3", []):
        t = item.get("ticker", "").strip().upper()
        if t and t not in tickers:
            tickers.append(t)

    fundamentals, performance, profiles = {}, {}, {}
    all_logs = []

    if mode == "full" and tickers:
        # Parallel fetch — all tickers at once
        from picks_shovels_builder import fetch_all_parallel
        loop = _aio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: fetch_all_parallel(tickers, cfg, max_workers=5))
        fundamentals = result["fundamentals"]
        profiles = result["profiles"]
        performance = result["performance"]
        all_logs = result["logs"]
    elif mode == "html-only":
        # Try loading cached data
        safe_name = cfg.get("title", "research").lower().replace(" ", "_").replace("&", "and")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c == '_')
        cache_path = os.path.join("reports", f"{safe_name}_data.json")
        if os.path.exists(cache_path):
            import json as _json
            with open(cache_path) as f:
                cached = _json.load(f)
            fundamentals = cached.get("fundamentals", {})
            performance = cached.get("performance", {})
            profiles = cached.get("profiles", {})
            all_logs.append("⚡ Using cached data (no API calls)")

    # Generate HTML
    try:
        from picks_shovels_builder import generate_html
        html = generate_html(cfg, fundamentals, performance, profiles)
    except Exception as e:
        html = f"<html><body><h1>Error generating report: {e}</h1></body></html>"

    # Save
    safe_name = cfg.get("title", "research").lower().replace(" ", "_").replace("&", "and")
    safe_name = "".join(c for c in safe_name if c.isalnum() or c == '_')
    out_path = os.path.join("reports", f"{safe_name}.html")
    _Path("reports").mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Cache data for quick rebuilds
    cache_path = out_path.replace(".html", "_data.json")
    import json as _json
    with open(cache_path, "w") as f:
        _json.dump({"fundamentals": fundamentals, "performance": performance, "profiles": profiles}, f, indent=2, default=str)

    return {
        "status": "ok",
        "report_url": f"/reports/{safe_name}.html",
        "fundamentals_log": all_logs,
        "performance_log": [],
        "profile_log": [],
        "tickers_processed": len(tickers),
    }


@app.get("/api/research/reports")
async def research_list_reports():
    """List all generated research reports."""
    from pathlib import Path as _Path
    reports_dir = _Path("reports")
    if not reports_dir.exists():
        return {"reports": []}

    reports = []
    for f in sorted(reports_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True):
        reports.append({
            "name": f.stem.replace("_", " ").title(),
            "url": f"/reports/{f.name}",
            "modified": f.stat().st_mtime,
        })
    return {"reports": reports}


# Try to serve static files from public/ (MUST be AFTER all API routes)
if os.path.isdir("public"):
    os.makedirs("reports", exist_ok=True)
    app.mount("/reports", StaticFiles(directory="reports", html=True), name="reports")
    app.mount("/", StaticFiles(directory="public", html=True), name="static")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  AnalysisGrid — Clean Base Build")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()

