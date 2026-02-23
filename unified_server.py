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
    from auth_middleware import init_firebase, get_current_user, require_auth, SUBSCRIPTION_TIERS
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

# ── Lightweight healthcheck ──────────────────────────────────────────────────
@app.get("/api/health")
async def healthcheck():
    return {"status": "ok"}


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

# Trading Card router
try:
    from card_endpoints import card_router
    app.include_router(card_router)
except Exception:
    pass

print("[BOOT] Routers registered", flush=True)


# ── Startup ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    print("[BOOT] on_startup() fired", flush=True)
    if discord_available and setup_discord:
        try:
            await asyncio.wait_for(setup_discord(app), timeout=30)
        except asyncio.TimeoutError:
            print("[BOOT] Discord timed out after 30s — continuing")
        except Exception as e:
            print(f"[BOOT] Discord error: {e}")

    if auto_scanner_available and setup_auto_scanner:
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

# Polygon Options API
try:
    from polygon_options import polygon_options_router
    app.include_router(polygon_options_router)
    print("[BOOT] Polygon Options API enabled")
except Exception as e:
    print(f"[BOOT] Polygon Options not loaded: {e}")

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


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE ANALYSIS (Polygon-powered)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/analyze/live/{symbol}")
async def analyze_live(symbol: str, period: str = "6mo"):
    symbol = symbol.upper().strip()
    cache_key = f"analysis:{symbol}:{period}"
    cached = analysis_cache.get(cache_key)
    if cached:
        return cached

    try:
        scanner = get_finnhub_scanner()
        result = scanner.analyze_symbol(symbol)
        if result:
            response = _safe_dict(result)
            analysis_cache.set(cache_key, response)
            return response
        else:
            raise HTTPException(status_code=404, detail=f"No analysis for {symbol}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
                analysis = scanner.analyze_symbol(sym)
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
# CONFIG — AI Kill Switch
# ═══════════════════════════════════════════════════════════════════════════════

AI_KILL_SWITCH: bool = False

@app.get("/api/config/ai-kill-switch")
async def get_ai_kill_switch():
    return {"enabled": AI_KILL_SWITCH}

@app.post("/api/config/ai-kill-switch")
async def set_ai_kill_switch(enabled: bool = False):
    global AI_KILL_SWITCH
    AI_KILL_SWITCH = enabled
    return {"enabled": AI_KILL_SWITCH}


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

# Try to serve static files from public/
if os.path.isdir("public"):
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

