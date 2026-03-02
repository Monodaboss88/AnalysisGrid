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
import logging as _logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import asdict
from collections import OrderedDict

_zombie_logger = _logging.getLogger("zombie_guard")

# ── Global scan lock — only 1 heavy scan at a time to prevent OOM on Railway ──
_scan_semaphore = asyncio.Semaphore(1)


async def safe_timeout(coro, *, timeout: float, label: str = "task"):
    """Run a coroutine with a timeout that does NOT create zombie threads.
    
    The critical difference from asyncio.wait_for(asyncio.to_thread(...)):
    - wait_for CANCELS the underlying future on timeout, but the thread keeps
      running forever, permanently consuming an executor slot (zombie thread).
    - This function uses asyncio.shield() so the thread finishes naturally,
      freeing its slot. We just stop waiting for the result.
    
    After enough wait_for timeouts, ALL 20 executor slots fill with zombies
    and the entire server freezes — even the health endpoint.
    """
    shielded = asyncio.shield(coro)
    try:
        return await asyncio.wait_for(shielded, timeout=timeout)
    except asyncio.TimeoutError:
        _zombie_logger.warning("[%s] timed out after %.0fs — thread will finish in background", label, timeout)
        # Thread keeps running in background and will free its executor slot
        # when the underlying HTTP call times out (requests timeout=15-30s)
        raise

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

# Structure Reversal Detector — MOVED to scanner_router.py (lazy-loaded)
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
    # Force no-cache on all HTML/JS pages so browsers always get latest version
    if request.url.path.endswith((".html", ".js")) or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# ── Lightweight healthcheck ──────────────────────────────────────────────────
@app.get("/api/health")
async def healthcheck():
    import gc as _gc
    # Include executor stats for debugging thread exhaustion
    loop = asyncio.get_running_loop()
    executor = getattr(loop, '_default_executor', None)
    ex_info = {}
    if executor:
        ex_info = {
            "executor_threads": len(getattr(executor, '_threads', [])),
            "executor_max": getattr(executor, '_max_workers', '?'),
            "executor_pending": getattr(executor, '_work_queue', None) and executor._work_queue.qsize() or 0,
        }
    # Memory usage — helps diagnose Railway OOM kills
    try:
        import resource
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB -> MB
    except Exception:
        try:
            import psutil
            mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            mem_mb = None
    mem_info = {"memory_mb": round(mem_mb, 1)} if mem_mb else {}
    # Check if any heavy scan is running (from scanner_router)
    scan_busy = False
    try:
        from scanner_router import _scan_semaphore as _sr_sem
        scan_busy = _sr_sem.locked()
    except Exception:
        scan_busy = _scan_semaphore.locked()
    return {"status": "ok", "version": "v18-scan-lock", "scan_busy": scan_busy, **ex_info, **mem_info}


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

    # Keep-alive removed — self-ping causes crash loops on Railway.
    # Frontend pages now have resilient 15s periodic re-checks instead.


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
_ai_response_cache: dict = {}  # key -> (timestamp, response_dict) — 5 min TTL
if anthropic_available and os.environ.get("ANTHROPIC_API_KEY"):
    try:
        anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=25.0)
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

# Options Flow Stream — MOVED to scanner_router.py (lazy-loaded)
flow_stream = None
flow_stream_available = False

print(f"[BOOT] Ready — PORT={os.environ.get('PORT', '8000')}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS — Status, Quotes, Polygon Key
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/status")
async def get_status():
    has_polygon = bool(os.environ.get("POLYGON_API_KEY"))
    watchlists = watchlist_mgr.get_all_watchlists()

    streaming_status = None
    if streaming_available and streaming_manager and streaming_manager.streamer:
        streaming_status = streaming_manager.get_status()

    if streaming_status and streaming_status.get('connected'):
        data_source = "Polygon.io WebSocket (LIVE)"
    else:
        data_source = "Polygon.io"

    return {
        "status": "running",
        "deploy_version": "v6-clean-base",
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


# ═══════════════════════════════════════════════════════════════════════════════
# POLYGON PROXY — server-side calls so frontend never needs the API key
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/polygon/bars/{ticker}")
async def polygon_bars_proxy(ticker: str, from_date: str = "", to_date: str = "",
                             timespan: str = "day", multiplier: int = 1,
                             adjusted: str = "true", sort: str = "asc", limit: int = 5000):
    """Proxy for Polygon /v2/aggs/ticker/{ticker}/range/{m}/{ts}/{from}/{to}"""
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        raise HTTPException(status_code=500, detail="POLYGON_API_KEY not configured")
    if not from_date or not to_date:
        raise HTTPException(status_code=400, detail="from_date and to_date required")
    ticker = ticker.upper().strip()
    url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}"
           f"/{from_date}/{to_date}?adjusted={adjusted}&sort={sort}&limit={limit}&apiKey={key}")
    try:
        import requests as _req
        def _fetch():
            r = _req.get(url, timeout=15)
            r.raise_for_status()
            return r.json()
        data = await asyncio.to_thread(_fetch)
        return data
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Polygon API error: {e}")


@app.get("/api/polygon/news/{ticker}")
async def polygon_news_proxy(ticker: str, from_date: str = "", to_date: str = "",
                             limit: int = 3, sort: str = "published_utc"):
    """Proxy for Polygon /v2/reference/news"""
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        raise HTTPException(status_code=500, detail="POLYGON_API_KEY not configured")
    ticker = ticker.upper().strip()
    params = f"ticker={ticker}&limit={limit}&sort={sort}&apiKey={key}"
    if from_date:
        params += f"&published_utc.gte={from_date}"
    if to_date:
        params += f"&published_utc.lte={to_date}"
    url = f"https://api.polygon.io/v2/reference/news?{params}"
    try:
        import requests as _req
        def _fetch():
            r = _req.get(url, timeout=15)
            r.raise_for_status()
            return r.json()
        data = await asyncio.to_thread(_fetch)
        return data
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Polygon API error: {e}")


@app.get("/api/quote/{symbol}")
async def get_quote(symbol: str):
    symbol = symbol.upper().strip()
    try:
        quote = await safe_timeout(asyncio.to_thread(get_price_quote, symbol), timeout=15, label="quote")
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
# LIVE ANALYSIS — MOVED to analysis_router.py (chart + MTF + AI commentary)
# ═══════════════════════════════════════════════════════════════════════════════


# ── Scanner endpoints (buffett, warroom, regime, options-flow, scan/live, structure) ──
# MOVED to scanner_router.py (lazy-loaded, no module-level singletons)

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


# ── Structure Reversal ── MOVED to scanner_router.py


# Pages moved to pages_router.py — wired below

_no_cache = {"Cache-Control": "no-cache, no-store, must-revalidate"}


# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCH BUILDER API  (kept inline — lightweight, rarely used)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/research/build")
async def research_build(request: Request):
    """Build a Picks & Shovels research report from config — parallel fetch."""
    from pathlib import Path as _Path

    body = await request.json()
    cfg = body.get("config", {})
    mode = body.get("mode", "full")

    if not cfg.get("title"):
        raise HTTPException(status_code=400, detail="Config must have a title")

    tickers = []
    for item in cfg.get("layer2", []) + cfg.get("layer3", []):
        t = item.get("ticker", "").strip().upper()
        if t and t not in tickers:
            tickers.append(t)

    fundamentals, performance, profiles = {}, {}, {}
    all_logs = []

    if mode == "full" and tickers:
        from picks_shovels_builder import fetch_all_parallel
        result = await safe_timeout(
            asyncio.to_thread(fetch_all_parallel, tickers, cfg, 5),
            timeout=60, label="research-build"
        )
        fundamentals = result["fundamentals"]
        profiles = result["profiles"]
        performance = result["performance"]
        all_logs = result["logs"]
    elif mode == "html-only":
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
            all_logs.append("Using cached data (no API calls)")

    try:
        from picks_shovels_builder import generate_html
        html = generate_html(cfg, fundamentals, performance, profiles)
    except Exception as e:
        html = f"<html><body><h1>Error generating report: {e}</h1></body></html>"

    safe_name = cfg.get("title", "research").lower().replace(" ", "_").replace("&", "and")
    safe_name = "".join(c for c in safe_name if c.isalnum() or c == '_')
    out_path = os.path.join("reports", f"{safe_name}.html")
    _Path("reports").mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

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


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTED ROUTERS — scanner, analysis, pages (MUST be before static mount)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from scanner_router import router as scanner_router
    app.include_router(scanner_router)
    print("[BOOT] scanner_router loaded (lazy scanners)")
except Exception as e:
    print(f"[BOOT] scanner_router FAILED: {e}")

try:
    from analysis_router import router as analysis_router
    app.include_router(analysis_router)
    print("[BOOT] analysis_router loaded (chart + MTF + AI)")
except Exception as e:
    print(f"[BOOT] analysis_router FAILED: {e}")

try:
    from pages_router import router as pages_router
    app.include_router(pages_router)
    print("[BOOT] pages_router loaded (HTML page serving)")
except Exception as e:
    print(f"[BOOT] pages_router FAILED: {e}")


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
    print("  AnalysisGrid — Modular Build")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()