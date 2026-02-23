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

