"""
Unified MTF Auction Scanner Server
===================================
One server that connects everything:
- Finnhub real-time data
- MTF Scanner analysis
- Watchlist manager (163 symbols)
- Alert triggers
- Trade tracker

Run: python unified_server.py
Open: http://localhost:8000

Author: Rob's Trading Systems
Version: 2.0.0
"""

import os
import json
import time
import math
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import asdict
from collections import OrderedDict

# Data libraries
import pandas as pd
from polygon_data import get_bars, get_price_quote

# FastAPI
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
import uvicorn

# Our modules
from chart_input_analyzer import ChartInputSystem, ChartInput
from finnhub_scanner_v2 import FinnhubScanner, TechnicalCalculator
from watchlist_manager import WatchlistManager

# Anthropic (Claude) for AI commentary
try:
    import anthropic
    anthropic_available = True
except ImportError:
    anthropic_available = False
    anthropic = None

# AI Advisor (hedge fund level intelligence)
try:
    from ai_advisor_endpoints import ai_router
    ai_advisor_available = True
except ImportError as e:
    ai_advisor_available = False
    print(f"⚠️ AI Advisor not loaded: {e}")

# Market Pulse Analyzer
try:
    from market_pulse_endpoints import pulse_router
    market_pulse_available = True
except ImportError as e:
    market_pulse_available = False
    print(f"⚠️ Market Pulse not loaded: {e}")

# Trade Journal
try:
    from trade_journal_endpoints import journal_router
    trade_journal_available = True
except ImportError as e:
    trade_journal_available = False
    print(f"⚠️ Trade Journal not loaded: {e}")

# Signal Probability (polygon_signal_tool)
try:
    from signal_endpoints import signal_router
    signal_available = True
except ImportError as e:
    signal_available = False
    print(f"⚠️ Signal Probability not loaded: {e}")

# Options Analysis (Greeks + IV)
try:
    from options_greeks_calculator import OptionsStrategyAnalyzer, OptionLeg
    from iv_analysis import IVAnalyzer
    from earnings_calendar import EarningsCalendar
    options_analysis_available = True
    earnings_cal = EarningsCalendar()
except ImportError as e:
    options_analysis_available = False
    earnings_cal = None
    print(f"⚠️ Options Analysis not loaded: {e}")

# Range Watcher (multi-period HH/HL/LH/LL structure analysis)
try:
    from rangewatcher.range_watcher_endpoints import range_router, set_scanner as set_range_scanner, set_openai_client as set_range_openai
    range_watcher_available = True
except ImportError as e:
    range_watcher_available = False
    set_range_scanner = None
    set_range_openai = None
    print(f"⚠️ Range Watcher not loaded: {e}")

# Authentication Middleware
try:
    from auth_middleware import init_firebase, get_current_user, require_auth, SUBSCRIPTION_TIERS
    auth_available = True
except ImportError as e:
    auth_available = False
    print(f"⚠️ Auth middleware not loaded: {e}")

# Firestore Storage (per-user data)
try:
    from firestore_store import get_firestore, UserAlert, UserTrade
    firestore_available = True
except ImportError as e:
    firestore_available = False
    print(f"⚠️ Firestore store not loaded: {e}")

# Authorize.net Payments
try:
    from authorize_payments import payment_router
    payments_available = True
except ImportError as e:
    payments_available = False
    print(f"⚠️ Authorize.net payments not loaded: {e}")

# Workflow & Discipline System
try:
    from workflow_endpoints import workflow_router
    workflow_available = True
except ImportError as e:
    workflow_available = False
    print(f"⚠️ Workflow endpoints not loaded: {e}")

# Entry Scanner (Volume Profile based entry detection)
try:
    from emtryscan.entry_scanner_endpoints import entry_router, set_finnhub_scanner as set_entry_scanner, set_finnhub_scanner_getter
    entry_scanner_available = True
except ImportError as e:
    entry_scanner_available = False
    set_entry_scanner = None
    set_finnhub_scanner_getter = None
    print(f"⚠️ Entry Scanner not loaded: {e}")

# Trade Rule Engine (deterministic rules + learning)
try:
    from rule_engine_endpoints import rule_router, set_scanner_for_rules
    rule_engine_available = True
except ImportError as e:
    rule_engine_available = False
    set_scanner_for_rules = None
    print(f"⚠️ Trade Rule Engine not loaded: {e}")

# Compression Reversal Scanner (options setups)
try:
    from compscan.fastapi_endpoints import compression_router
    compression_scanner_available = True
except ImportError as e:
    compression_scanner_available = False
    print(f"⚠️ Compression Scanner not loaded: {e}")

# Capitulation Detector (catching bottoms) and Euphoria Detector (catching tops)
try:
    from capitulation_detector_v2 import (
        CapitulationDetectorV2 as CapitulationDetector, CapitulationLevel, scan_for_capitulation,
        EuphoriaLevel, scan_for_euphoria
    )
    capitulation_available = True
    print("✅ Capitulation & Euphoria Detector V2 enabled")
except ImportError as e:
    capitulation_available = False
    print(f"⚠️ Capitulation Detector not loaded: {e}")

# Structure Reversal Detector (macro structure-based reversals)
try:
    from structure_reversal_detector import StructureReversalDetector, StructureContext, ReversalAlert
    from rangewatcher.range_watcher import RangeWatcher
    structure_reversal_detector = StructureReversalDetector(min_confidence=40.0)
    range_watcher_analyzer = RangeWatcher()
    structure_reversal_available = True
    print("✅ Structure Reversal Detector enabled")
except ImportError as e:
    structure_reversal_available = False
    structure_reversal_detector = None
    range_watcher_analyzer = None
    print(f"⚠️ Structure Reversal Detector not loaded: {e}")

# Absorption Detector (passive limit order walls)
try:
    from absorption_detector import AbsorptionDetector, AbsorptionResult
    absorption_detector = AbsorptionDetector()
    absorption_available = True
    print("✅ Absorption Detector enabled")
except ImportError as e:
    absorption_available = False
    absorption_detector = None
    print(f"⚠️ Absorption Detector not loaded: {e}")

# Squeeze Detector (volatility compression)
try:
    from squeeze_detector_v2 import SqueezeDetectorV2 as SqueezeDetector, SqueezeMetrics, scan_for_squeezes_v2 as scan_for_squeezes
    squeeze_available = True
    print("✅ Squeeze Detector V2 enabled")
except ImportError as e:
    squeeze_available = False
    print(f"⚠️ Squeeze Detector not loaded: {e}")

# Run Sustainability Analyzer (100%+ run evaluation)
try:
    from sustainability_endpoints import sustainability_router
    sustainability_available = True
except ImportError as e:
    sustainability_available = False
    print(f"⚠️ Run Sustainability Analyzer not loaded: {e}")

# Discord Bot + Hybrid Task Queue
try:
    from discord_endpoints import discord_router, setup_discord
    discord_available = True
except ImportError as e:
    discord_available = False
    setup_discord = None
    print(f"⚠️ Discord integration not loaded: {e}")

# Telegram Bot + Hybrid Task Queue
try:
    from telegram_endpoints import telegram_router, setup_telegram
    telegram_available = True
except ImportError as e:
    telegram_available = False
    setup_telegram = None
    print(f"⚠️ Telegram integration not loaded: {e}")

# Alpha Scanner (7-step automated bullish finder)
try:
    from alpha_endpoints import alpha_router
    alpha_scanner_available = True
except ImportError as e:
    alpha_scanner_available = False
    print(f"⚠️ Alpha Scanner not loaded: {e}")

# Auto-Scanner (30-min background stock scanner)
try:
    from auto_scanner import setup_auto_scanner, get_auto_scanner
    auto_scanner_available = True
except ImportError as e:
    auto_scanner_available = False
    setup_auto_scanner = None
    get_auto_scanner = None
    print(f"⚠️ Auto-Scanner not loaded: {e}")

# WebSocket Streaming (real-time minute bars)
try:
    from polygon_websocket import StreamingManager, MinuteBar
    streaming_available = True
    streaming_manager = StreamingManager.get_instance()
except ImportError as e:
    streaming_available = False
    streaming_manager = None
    print(f"⚠️ WebSocket streaming not loaded: {e}")

# Options Flow Stream (real-time options flow detection)
try:
    from options_flow_stream import get_flow_stream, OptionsFlowStream
    flow_stream = get_flow_stream()
    flow_stream_available = True
    print("✅ Options Flow Stream module loaded")
except ImportError as e:
    flow_stream = None
    flow_stream_available = False
    print(f"⚠️ Options Flow Stream not loaded: {e}")

# Trade Monitor (auto-close on target/stop hit)
try:
    from trade_monitor import get_trade_monitor, TradeMonitor
    trade_monitor = get_trade_monitor(interval=30)
    trade_monitor_available = True
    print("✅ Trade Monitor module loaded")
except ImportError as e:
    trade_monitor = None
    trade_monitor_available = False
    print(f"⚠️ Trade Monitor not loaded: {e}")

# Notification Service (Firebase Cloud Messaging)
try:
    from notification_service import get_notification_service
    notification_service = get_notification_service()
    notification_available = True
    print("✅ Notification Service loaded")
except ImportError as e:
    notification_service = None
    notification_available = False
    print(f"⚠️ Notification Service not loaded: {e}")

# Extension Duration Predictor (THE EDGE)
try:
    from extension_predictor_v2 import ExtensionPredictor, CandleData
    extension_predictor = ExtensionPredictor(candle_minutes=120)
    extension_available = True
    print("✅ Extension Duration Predictor V2 enabled")
except ImportError as e:
    extension_available = False
    extension_predictor = None
    print(f"⚠️ Extension Predictor not loaded: {e}")

# Dual Setup Generator (Zero-cost AI alternative)
try:
    from dual_setup_generator_v2 import DualSetupGenerator, generate_dual_setup
    dual_setup_available = True
    print("✅ Dual Setup Generator V2 enabled (zero API cost)")
except ImportError as e:
    dual_setup_available = False
    generate_dual_setup = None
    print(f"⚠️ Dual Setup Generator not loaded: {e}")

# MTF Auction Scanner V2 (multi-timeframe non-bias scoring)
try:
    from mtf_auction_scanner_v2 import MTFAuctionScanner
    mtf_scanner = MTFAuctionScanner()
    mtf_scanner_available = True
    print("✅ MTF Auction Scanner V2 enabled")
except ImportError as e:
    mtf_scanner_available = False
    mtf_scanner = None
    print(f"⚠️ MTF Auction Scanner not loaded: {e}")

# Overnight / Gap Prediction Model V2
try:
    from overnight_model_v2 import (
        OvernightModelV2, OvernightPrediction,
        scan_overnight as _scan_overnight_single
    )
    overnight_model = OvernightModelV2()
    overnight_available = True
    print("✅ Overnight Model V2 enabled")
except ImportError as e:
    overnight_available = False
    overnight_model = None
    _scan_overnight_single = None
    print(f"⚠️ Overnight Model not loaded: {e}")


# =============================================================================
# HELPERS
# =============================================================================

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


def _watchlist_symbols(lst) -> List[str]:
    """Return a JSON-friendly list of symbols from a Watchlist."""
    try:
        # WatchlistManager stores Watchlist.symbols as WatchlistSymbol objects
        return [s.symbol for s in getattr(lst, "symbols", [])]
    except Exception:
        # Fallback: best effort
        return [str(s) for s in getattr(lst, "symbols", [])]


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ChartInputRequest(BaseModel):
    """Single timeframe chart input"""
    symbol: str
    price: float
    vah: float
    poc: float
    val: float
    vwap: float
    rsi: float
    timeframe: str = "1HR"


class MTFInputRequest(BaseModel):
    """Multi-timeframe chart input"""
    symbol: str
    timeframes: Dict[str, Dict[str, float]]
    # Example: {"30MIN": {"price": 619, "vah": 666, ...}, "1HR": {...}}


class AlertRequest(BaseModel):
    """Alert creation request"""
    symbol: str
    level: float
    direction: str  # "above" or "below"
    action: str     # "LONG", "SHORT", "EXIT", "ALERT"
    note: str = ""


class AIAlertsRequest(BaseModel):
    """AI Trade Plan alerts - creates multiple alerts at once"""
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry: float
    stop: float
    target1: float
    target2: float = 0
    trade_timeframe: str = "SWING"


class TradeRequest(BaseModel):
    """Trade logging request"""
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


class TradeUpdateRequest(BaseModel):
    """Trade update request"""
    symbol: str
    status: str  # "ACTIVE", "WIN", "LOSS", "SCRATCH"
    exit_price: Optional[float] = None


# =============================================================================
# TTL CACHE FOR SCAN SPEED
# =============================================================================
class TTLCache:
    """Simple TTL cache - stores results for a configurable number of seconds"""
    def __init__(self, ttl_seconds=60, max_size=500):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._cache = OrderedDict()
    
    def get(self, key):
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry['ts'] < self.ttl:
                # Move to end (LRU)
                self._cache.move_to_end(key)
                return entry['data']
            else:
                del self._cache[key]
        return None
    
    def set(self, key, data):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = {'data': data, 'ts': time.time()}
        # Evict oldest if over max_size
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
    
    def clear(self):
        self._cache.clear()
    
    @property
    def size(self):
        return len(self._cache)

# Cache instances: 60s TTL for live data, 300s for slower-changing data
candle_cache = TTLCache(ttl_seconds=60, max_size=300)    # Raw candle data
analysis_cache = TTLCache(ttl_seconds=45, max_size=300)  # Full analysis results  
squeeze_cache = TTLCache(ttl_seconds=120, max_size=200)  # Squeeze scans (less volatile)
absorption_cache = TTLCache(ttl_seconds=90, max_size=200) # Absorption zones

print("✅ TTL Cache initialized (candles=60s, analysis=45s, squeeze=120s, absorption=90s)")

# =============================================================================
# SERVER
# =============================================================================

app = FastAPI(
    title="MTF Auction Scanner",
    description="Real-time multi-timeframe auction analysis",
    version="2.0.0"
)

# Global exception handler for debugging
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTPException with CORS headers"""
    print(f"⚠️ HTTP Error {exc.status_code}: {exc.detail}")
    print(f"Request: {request.url}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    error_trace = traceback.format_exc()
    print(f"❌ UNHANDLED ERROR: {exc}")
    print(f"Request: {request.url}")
    print(f"Traceback:\n{error_trace}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": error_trace[:500]},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

# CORS - Allow all origins for API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Register AI Advisor router
if ai_advisor_available:
    app.include_router(ai_router, prefix="/api/ai")
    print("✅ AI Advisor (hedge fund level) enabled")

# Register Market Pulse router
if market_pulse_available:
    app.include_router(pulse_router)
    print("✅ Market Pulse Analyzer enabled")

# Register Trade Journal router
if trade_journal_available:
    app.include_router(journal_router)
    print("✅ Trade Journal enabled")

# Register Signal Probability router
if signal_available:
    app.include_router(signal_router)
    print("✅ Signal Probability (historical odds) enabled")

# Register Range Watcher router
if range_watcher_available:
    app.include_router(range_router, prefix="/api/range")
    print("✅ Range Watcher enabled")

# Register Payments router
if payments_available:
    app.include_router(payment_router)
    print("✅ Authorize.net Payments enabled")

# Register Workflow router
if workflow_available:
    app.include_router(workflow_router)
    print("✅ Workflow & Discipline System enabled")

# Register Entry Scanner router
if entry_scanner_available:
    app.include_router(entry_router)
    print("✅ Entry Scanner (VP Entries) enabled")

# Register Trade Rule Engine router
if rule_engine_available:
    app.include_router(rule_router, prefix="/api/rules")
    print("✅ Trade Rule Engine (deterministic rules + learning) enabled")

# Register Compression Reversal Scanner router
if compression_scanner_available:
    app.include_router(compression_router)
    print("✅ Compression Reversal Scanner (options setups) enabled")

# Register Run Sustainability Analyzer router
if sustainability_available:
    app.include_router(sustainability_router)
    print("✅ Run Sustainability Analyzer enabled")

# Register Discord Bot + Hybrid Task Queue router
if discord_available:
    app.include_router(discord_router, prefix="/discord")
    print("✅ Discord Bot + Hybrid Task Queue enabled")

# Register Alpha Scanner router
if alpha_scanner_available:
    app.include_router(alpha_router, prefix="")
    print("✅ Alpha Scanner (7-step bullish finder) enabled")

# Register Telegram Bot + Hybrid Task Queue router
if telegram_available:
    app.include_router(telegram_router, prefix="/telegram")
    print("✅ Telegram Bot + Hybrid Task Queue enabled")

# Register Trading Card router
try:
    from card_endpoints import card_router
    app.include_router(card_router)
    card_available = True
    print("✅ Trading Card system enabled")
except Exception as e:
    card_available = False
    print(f"⚠️ Trading Card system not available: {e}")

# Messaging startup event — initialize webhooks + background workers
@app.on_event("startup")
async def on_startup():
    if discord_available and setup_discord:
        try:
            await setup_discord(app)
        except Exception as e:
            print(f"⚠️ Discord startup error: {e}")
    if telegram_available and setup_telegram:
        try:
            await setup_telegram(app)
        except Exception as e:
            print(f"⚠️ Telegram startup error: {e}")

    # Start auto-scanner after Discord is ready
    if auto_scanner_available and setup_auto_scanner:
        try:
            from discord_bot import get_discord
            discord_client = get_discord()
            # watchlist_mgr is initialized below, so we defer start
            import asyncio
            async def _start_auto_scanner():
                await asyncio.sleep(5)  # Let watchlist_mgr initialize
                setup_auto_scanner(
                    watchlist_mgr=watchlist_mgr,
                    discord_client=discord_client,
                    auto_start=True
                )
                print("✅ Auto-Scanner started (30-min interval)")
            asyncio.create_task(_start_auto_scanner())
        except Exception as e:
            print(f"⚠️ Auto-Scanner startup error: {e}")

    # Start Trade Monitor (auto-close engine)
    if trade_monitor_available and trade_monitor:
        try:
            async def _start_trade_monitor():
                await asyncio.sleep(8)  # Let Firestore + Polygon initialize
                trade_monitor.start_background()
                print("✅ Trade Monitor started (30s interval, market hours only)")

                # Hook notification service into trade close events
                if notification_available and notification_service:
                    trade_monitor.on_close(notification_service.notify_trade_close)
                    print("✅ Trade close → push notification hook connected")

            asyncio.create_task(_start_trade_monitor())
        except Exception as e:
            print(f"⚠️ Trade Monitor startup error: {e}")

# Initialize Firebase Auth
if auth_available:
    init_firebase()

# Initialize components
chart_system = ChartInputSystem(data_dir="./scanner_data")
watchlist_mgr = WatchlistManager()

# Finnhub scanner (initialized on first use with API key)
finnhub_scanner: Optional[FinnhubScanner] = None

# ═══════════════════════════════════════════════════════════════════════════════
# AI KILL SWITCH — Global toggle to disable all LLM API calls
# When True: all AI endpoints fall back to deterministic rule-based analysis
# Controlled via /api/config/ai-kill-switch endpoint or Trade Desk UI
# ═══════════════════════════════════════════════════════════════════════════════
AI_KILL_SWITCH: bool = False
AI_KILL_SWITCH_REASON: str = ""
AI_KILL_SWITCH_TOGGLED_AT: Optional[str] = None

def is_ai_enabled() -> bool:
    """Check if AI calls are allowed (kill switch is OFF)"""
    return not AI_KILL_SWITCH


def _save_ai_suggestion_bg(symbol: str, suggestion_type: str, content: str, metadata: dict = None):
    """Fire-and-forget save of AI suggestion to Firestore (non-blocking)."""
    if not content:
        return
    try:
        from firestore_store import get_firestore
        fs = get_firestore()
        if fs.is_available():
            doc_id = fs.save_ai_suggestion(symbol, suggestion_type, content, metadata)
            if doc_id:
                print(f"💾 AI suggestion saved: {doc_id}")
    except Exception as e:
        print(f"⚠️ AI suggestion save failed: {e}")


# Anthropic client for AI commentary
anthropic_client = None
if anthropic_available and os.environ.get("ANTHROPIC_API_KEY"):
    try:
        anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        print("✅ Claude AI commentary enabled (from env)")
        # Share with Range Watcher
        if set_range_openai:
            set_range_openai(anthropic_client)
    except Exception as e:
        print(f"⚠️ Anthropic init failed: {e}")


def get_finnhub_scanner() -> FinnhubScanner:
    """Get or create Finnhub scanner"""
    global finnhub_scanner
    if finnhub_scanner is None:
        api_key = os.environ.get("FINNHUB_API_KEY")
        polygon_key = os.environ.get("POLYGON_API_KEY")
        
        if not api_key and not polygon_key:
            raise HTTPException(
                status_code=400,
                detail="No API keys set. Set POLYGON_API_KEY or FINNHUB_API_KEY environment variable."
            )
        
        # Use a dummy Finnhub key if only Polygon is available
        if not api_key:
            api_key = "dummy_key_polygon_only"
            
        try:
            finnhub_scanner = FinnhubScanner(api_key)
            
            # Set Polygon key if available
            if polygon_key and hasattr(finnhub_scanner, 'set_polygon_key'):
                finnhub_scanner.set_polygon_key(polygon_key)
                print(f"✅ Polygon.io enabled (from env)")
            
            # Update Range Watcher with the scanner
            if set_range_scanner:
                set_range_scanner(finnhub_scanner)
            
            # Update Rule Engine with the scanner (for weekly structure)
            if rule_engine_available and set_scanner_for_rules:
                set_scanner_for_rules(finnhub_scanner)
            
            # Update Entry Scanner with the scanner
            if entry_scanner_available and set_entry_scanner:
                set_entry_scanner(finnhub_scanner)
        except ValueError as e:
            # Keep the server running even if live-data deps are missing.
            raise HTTPException(status_code=400, detail=str(e))
    return finnhub_scanner


# Set up lazy getter for Entry Scanner now that get_finnhub_scanner is defined
if entry_scanner_available and set_finnhub_scanner_getter:
    set_finnhub_scanner_getter(get_finnhub_scanner)


# =============================================================================
# API ENDPOINTS
# =============================================================================

# -----------------------------------------------------------------------------
# SYSTEM
# -----------------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve main HTML interface"""
    return FileResponse("unified_ui.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/v2")
async def root_v2():
    """Serve V2 HTML interface"""
    return FileResponse("unified_ui_v2.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/index")
async def serve_index():
    """Serve public/index.html interface"""
    return FileResponse("public/index.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/simple.html")
async def serve_simple():
    """Serve Simple Scanner interface"""
    return FileResponse("public/simple.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/growth")
async def serve_growth():
    """Serve Capital Growth Engine"""
    return FileResponse("public/growth.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/desk")
async def serve_desk():
    """Serve Trade Desk - Command Center Hub"""
    return FileResponse("public/desk.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/research")
async def serve_research():
    """Serve Research Builder - Picks & Shovels"""
    return FileResponse("public/research.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/sustainability")
async def serve_sustainability():
    """Serve Run Sustainability Analyzer"""
    return FileResponse("public/sustainability.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/charts")
async def serve_charts():
    """Serve TradingView Multi-Panel Charts"""
    return FileResponse("public/charts.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/login.html")
async def serve_login():
    """Serve login page"""
    return FileResponse("public/login.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/upgrade.html")
async def serve_upgrade():
    """Serve upgrade page"""
    return FileResponse("public/upgrade.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/catalyst")
async def serve_catalyst():
    """Serve Stock Catalyst Scanner"""
    return FileResponse("stock-catalyst-scanner_1.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/buffett")
async def serve_buffett():
    """Serve Buffett Blood Scanner"""
    return FileResponse("stock-buffett-scanner.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/journal")
async def serve_journal_analytics():
    """Serve Journal Analytics Dashboard"""
    return FileResponse("public/journal.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/options-flow")
async def serve_options_flow():
    """Serve Options Flow Scanner"""
    return FileResponse("stock-options-scanner.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/war-room")
@app.get("/warroom")
async def serve_war_room():
    """Serve War Room Pre-Market Scanner"""
    import os as _os
    for f in ("public/warroom.html", "stock-war-room.html"):
        if _os.path.exists(f):
            return FileResponse(f, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return FileResponse("stock-war-room.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/api/status")
async def get_status():
    """Get system status"""
    has_finnhub = bool(os.environ.get("FINNHUB_API_KEY"))
    has_alpaca = bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"))
    has_polygon = bool(os.environ.get("POLYGON_API_KEY"))
    has_openai = anthropic_client is not None
    watchlists = watchlist_mgr.get_all_watchlists()
    
    # Get streaming status
    streaming_status = None
    if streaming_available and streaming_manager and streaming_manager.streamer:
        streaming_status = streaming_manager.get_status()
    
    # Determine data source
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
        "deploy_version": "v5-stop-fix",
        "finnhub_connected": has_finnhub,
        "alpaca_connected": has_alpaca,
        "polygon_connected": has_polygon,
        "chatgpt_enabled": has_openai,
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
            "candles": candle_cache.size
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/debug/firestore-alerts")
async def debug_firestore_alerts(symbol: str = None):
    """Debug endpoint - list all users and their alerts in Firestore"""
    try:
        from firestore_store import get_firestore
        try:
            import firebase_admin
            fb_installed = True
        except ImportError:
            fb_installed = False
        has_env = bool(os.environ.get('FIREBASE_SERVICE_ACCOUNT'))
        
        fs = get_firestore()
        db = fs.db if fs else None
        if not db:
            return {
                "error": "Firestore not initialized",
                "firebase_admin_installed": fb_installed,
                "FIREBASE_SERVICE_ACCOUNT_set": has_env,
                "FIREBASE_SERVICE_ACCOUNT_length": len(os.environ.get('FIREBASE_SERVICE_ACCOUNT', '')),
                "fs_object": str(fs),
                "fs_db": str(db),
                "firestore_available_global": firestore_available,
                "deploy_version": "debug-v5"
            }
        
        users_ref = db.collection('users')
        user_docs = list(users_ref.list_documents())
        
        result = {
            "total_users_found": len(user_docs),
            "users": [],
            "deploy_version": "debug-v4",
            "timestamp": datetime.now().isoformat()
        }
        
        for user_doc in user_docs[:20]:  # Limit to 20 users
            uid = user_doc.id
            user_info = {"uid": uid, "alerts": []}
            
            # Get alerts subcollection
            alerts_ref = db.collection('users').document(uid).collection('alerts')
            alert_docs = list(alerts_ref.stream())
            
            for alert_doc in alert_docs:
                alert_data = alert_doc.to_dict()
                alert_data["_doc_id"] = alert_doc.id
                if symbol:
                    sym = alert_data.get("symbol", "").upper()
                    if sym == symbol.upper():
                        user_info["alerts"].append(alert_data)
                else:
                    user_info["alerts"].append(alert_data)
            
            user_info["alert_count"] = len(user_info["alerts"])
            if user_info["alert_count"] > 0 or not symbol:
                result["users"].append(user_info)
        
        result["total_alerts"] = sum(u["alert_count"] for u in result["users"])
        return result
        
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/api/debug/firestore-rest")
async def debug_firestore_rest(symbol: str = None):
    """Debug endpoint - test Firestore REST API client"""
    try:
        from firestore_rest import search_all_alerts, get_status, is_available, get_all_user_ids, _sign_in, get_bot_uid
        import firestore_rest
        
        status = get_status()
        if not is_available():
            return {"error": "REST client not configured", "status": status}
        
        # Explicitly try sign-in
        token = _sign_in()
        sign_in_ok = token is not None
        bot_uid = get_bot_uid()
        
        # Try collection group query directly with error tracking
        query_error = None
        try:
            alerts = search_all_alerts(symbol)
        except Exception as qe:
            alerts = []
            query_error = str(qe)
        
        # Also try direct httpx call for debugging
        direct_result = None
        if token:
            try:
                import httpx
                query_body = {"structuredQuery": {"from": [{"collectionId": "alerts", "allDescendants": True}], "limit": 10}}
                if symbol:
                    query_body["structuredQuery"]["where"] = {
                        "fieldFilter": {
                            "field": {"fieldPath": "symbol"},
                            "op": "EQUAL",
                            "value": {"stringValue": symbol.upper()}
                        }
                    }
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Referer": "https://analysis-grid.web.app"
                }
                url = f"https://firestore.googleapis.com/v1/projects/analysis-grid/databases/(default)/documents:runQuery"
                resp = httpx.post(url, json=query_body, headers=headers, timeout=15)
                direct_result = {"status_code": resp.status_code, "body_preview": resp.text[:500]}
            except Exception as de:
                direct_result = {"error": str(de)}
        
        return {
            "sign_in_ok": sign_in_ok,
            "bot_uid": bot_uid,
            "token_preview": token[:20] + "..." if token else None,
            "status": get_status(),
            "query_error": query_error,
            "direct_query_result": direct_result,
            "total_alerts": len(alerts),
            "alerts": alerts[:20],
            "deploy_version": "rest-v4"
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/api/debug/quote/{symbol}")
async def debug_quote(symbol: str):
    """Debug endpoint - shows raw quote data from all sources"""
    scanner = get_finnhub_scanner()
    results = {
        "symbol": symbol.upper(),
        "timestamp": datetime.now().isoformat(),
        "polygon_client_exists": scanner.polygon_client is not None,
        "alpaca_client_exists": scanner.alpaca_client is not None,
        "polygon_snapshot": None,
        "polygon_last_trade": None,
        "polygon_prev_close": None,
        "alpaca_quote": None,
        "finnhub_quote": None,
        "final_quote": None
    }
    
    if scanner.polygon_client:
        # Try Polygon snapshot (BEST - real-time price)
        try:
            snapshot = scanner.polygon_client.get_snapshot_ticker("stocks", symbol.upper())
            if snapshot:
                snap_data = {"raw_type": str(type(snapshot))}
                
                if hasattr(snapshot, 'last_trade') and snapshot.last_trade:
                    snap_data["last_trade_price"] = float(snapshot.last_trade.price) if hasattr(snapshot.last_trade, 'price') and snapshot.last_trade.price else None
                
                if hasattr(snapshot, 'day') and snapshot.day:
                    snap_data["day_close"] = float(snapshot.day.close) if hasattr(snapshot.day, 'close') and snapshot.day.close else None
                    snap_data["day_open"] = float(snapshot.day.open) if hasattr(snapshot.day, 'open') and snapshot.day.open else None
                    snap_data["day_high"] = float(snapshot.day.high) if hasattr(snapshot.day, 'high') and snapshot.day.high else None
                    snap_data["day_low"] = float(snapshot.day.low) if hasattr(snapshot.day, 'low') and snapshot.day.low else None
                
                if hasattr(snapshot, 'prev_day') and snapshot.prev_day:
                    snap_data["prev_day_close"] = float(snapshot.prev_day.close) if hasattr(snapshot.prev_day, 'close') and snapshot.prev_day.close else None
                
                if hasattr(snapshot, 'last_quote') and snapshot.last_quote:
                    snap_data["last_quote_bid"] = float(snapshot.last_quote.bid) if hasattr(snapshot.last_quote, 'bid') and snapshot.last_quote.bid else None
                    snap_data["last_quote_ask"] = float(snapshot.last_quote.ask) if hasattr(snapshot.last_quote, 'ask') and snapshot.last_quote.ask else None
                
                results["polygon_snapshot"] = snap_data
        except Exception as e:
            results["polygon_snapshot"] = {"error": str(e)}
        
        # Try Polygon last_trade
        try:
            last_trade = scanner.polygon_client.get_last_trade(symbol.upper())
            if last_trade:
                results["polygon_last_trade"] = {
                    "price": float(last_trade.price) if hasattr(last_trade, 'price') and last_trade.price else None,
                    "size": last_trade.size if hasattr(last_trade, 'size') else None,
                    "timestamp": str(last_trade.participant_timestamp) if hasattr(last_trade, 'participant_timestamp') else None,
                    "raw": str(last_trade)[:500]
                }
        except Exception as e:
            results["polygon_last_trade"] = {"error": str(e)}
    
        # Try Polygon prev_close
        try:
            prev_close = scanner.polygon_client.get_previous_close(symbol.upper())
            if prev_close and prev_close.results and len(prev_close.results) > 0:
                r = prev_close.results[0]
                results["polygon_prev_close"] = {
                    "close": float(r.close),
                    "open": float(r.open) if hasattr(r, 'open') else None,
                    "high": float(r.high) if hasattr(r, 'high') else None,
                    "low": float(r.low) if hasattr(r, 'low') else None,
                    "volume": int(r.volume) if hasattr(r, 'volume') else None
                }
        except Exception as e:
            results["polygon_prev_close"] = {"error": str(e)}
    
    # Try Alpaca
    if scanner.alpaca_client:
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol.upper())
            quote_data = scanner.alpaca_client.get_stock_latest_quote(request)
            if symbol.upper() in quote_data:
                q = quote_data[symbol.upper()]
                results["alpaca_quote"] = {
                    "bid_price": float(q.bid_price) if q.bid_price else None,
                    "ask_price": float(q.ask_price) if q.ask_price else None,
                    "mid_price": (float(q.bid_price) + float(q.ask_price)) / 2 if q.bid_price and q.ask_price else None,
                    "timestamp": str(q.timestamp)
                }
        except Exception as e:
            results["alpaca_quote"] = {"error": str(e)}
    
    # Try Finnhub
    try:
        fh_quote = scanner.client.quote(symbol.upper())
        results["finnhub_quote"] = {
            "current": fh_quote.get('c'),
            "open": fh_quote.get('o'),
            "high": fh_quote.get('h'),
            "low": fh_quote.get('l'),
            "prev_close": fh_quote.get('pc'),
            "timestamp": fh_quote.get('t')
        }
    except Exception as e:
        results["finnhub_quote"] = {"error": str(e)}
    
    # Final quote using get_quote method
    try:
        final = scanner.get_quote(symbol.upper())
        results["final_quote"] = final
    except Exception as e:
        results["final_quote"] = {"error": str(e)}
    
    return results


@app.get("/api/debug/vp/{symbol}")
async def debug_vp(symbol: str, timeframe: str = "1HR"):
    """Debug endpoint - shows VP levels and data info per timeframe"""
    scanner = get_finnhub_scanner()
    
    # Map timeframe to resolution
    resolution_map = {
        "5MIN": "5", "15MIN": "15", "30MIN": "30",
        "1HR": "60", "2HR": "60", "4HR": "60", "DAILY": "D"
    }
    resolution = resolution_map.get(timeframe.upper(), "60")
    
    # VP_BARS: Number of bars to use for Volume Profile (like Webull's visible range)
    VP_BARS = 30
    
    # Fetch enough days to get VP_BARS candles
    days_map = {
        "5MIN": 1, "15MIN": 2, "30MIN": 5,
        "1HR": 7, "2HR": 15, "4HR": 30, "DAILY": 60
    }
    days_back = days_map.get(timeframe.upper(), 7)
    
    # Fetch candles
    df = scanner._get_candles(symbol.upper(), resolution, days_back)
    
    results = {
        "symbol": symbol.upper(),
        "timeframe": timeframe.upper(),
        "resolution": resolution,
        "vp_bars": VP_BARS,
        "raw_candles": len(df) if df is not None else 0,
        "first_candle": str(df.index[0]) if df is not None and len(df) > 0 else None,
        "last_candle": str(df.index[-1]) if df is not None and len(df) > 0 else None,
    }
    
    # Resample if needed
    if df is not None and timeframe.upper() in ["2HR", "4HR"]:
        df = scanner._resample_to_timeframe(df, timeframe)
        results["resampled_candles"] = len(df)
    
    # Trim to last VP_BARS for consistent VP calculation
    if df is not None and len(df) > VP_BARS:
        df = df.tail(VP_BARS)
    results["bars_used"] = len(df) if df is not None else 0
    
    # Calculate VP
    if df is not None and len(df) >= 10:
        poc, vah, val = scanner.calc.calculate_volume_profile(df)
        vwap = scanner.calc.calculate_vwap(df)
        results["poc"] = round(poc, 2)
        results["vah"] = round(vah, 2)
        results["val"] = round(val, 2)
        results["vwap"] = round(vwap, 2)
        results["price_range"] = f"{round(df['low'].min(), 2)} - {round(df['high'].max(), 2)}"
    else:
        results["error"] = "Insufficient data"
    
    return results


@app.post("/api/set-key")
async def set_api_key(key: str):
    """Set Finnhub API key"""
    global finnhub_scanner
    os.environ["FINNHUB_API_KEY"] = key
    try:
        finnhub_scanner = FinnhubScanner(key)
    except ValueError as e:
        finnhub_scanner = None
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "message": "API key set successfully"}


@app.post("/api/set-alpaca-keys")
async def set_alpaca_keys(api_key: str, secret_key: str):
    """Set Alpaca API keys for real-time data"""
    global finnhub_scanner
    os.environ["ALPACA_API_KEY"] = api_key
    os.environ["ALPACA_SECRET_KEY"] = secret_key
    
    # Reinitialize scanner to pick up Alpaca
    finnhub_key = os.environ.get("FINNHUB_API_KEY", "")
    if not finnhub_key:
        # Set a dummy key since Finnhub is required for quotes
        finnhub_key = "dummy_for_alpaca"
        os.environ["FINNHUB_API_KEY"] = finnhub_key
    
    try:
        finnhub_scanner = FinnhubScanner(finnhub_key)
        if finnhub_scanner.alpaca_client:
            return {"status": "ok", "message": "Alpaca real-time data enabled!"}
        else:
            return {"status": "warning", "message": "Alpaca keys set but client failed to initialize"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/polygon-key")
async def get_polygon_key():
    """Return the Polygon API key for client-side scanner pages"""
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        raise HTTPException(status_code=404, detail="Polygon API key not configured")
    return {"key": key}


@app.get("/api/buffett-scan")
async def buffett_scan(tickers: str = "", preset: str = ""):
    """Buffett Blood Scanner — scan tickers for value + crisis metrics"""
    try:
        from buffett_scanner import async_scan_tickers, PRESETS

        if preset and preset in PRESETS:
            symbols = PRESETS[preset]
        elif tickers:
            symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        else:
            raise HTTPException(status_code=400, detail="Provide tickers or preset param")

        if len(symbols) > 30:
            raise HTTPException(status_code=400, detail="Max 30 tickers per scan")

        data = await async_scan_tickers(symbols)
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


# =============================================================================
# OPTIONS FLOW STREAM — SSE Real-Time Flow
# =============================================================================

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

    return {
        "status": "ok",
        "message": f"Flow stream started for {len(symbols)} tickers",
        "tickers": symbols,
        "stream": flow_stream.get_status(),
    }


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


@app.get("/api/options-flow/stream/buffer")
async def options_flow_stream_buffer(limit: int = 100):
    """Get the recent flow events buffer (for initial load before SSE connects)."""
    if not flow_stream_available or not flow_stream:
        return {"events": [], "status": "unavailable"}
    return {
        "events": flow_stream.get_buffer(limit),
        "status": flow_stream.get_status(),
    }


@app.get("/api/options-flow/stream/events")
async def options_flow_sse(request: Request):
    """Server-Sent Events endpoint — streams live options flow events to the browser."""
    if not flow_stream_available or not flow_stream:
        raise HTTPException(status_code=400, detail="Flow stream not available")

    q = flow_stream.subscribe()

    async def event_generator():
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'connected', 'status': flow_stream.get_status()})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=25.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat to keep connection alive
                    yield f"data: {json.dumps({'type': 'heartbeat', 'ts': datetime.now(timezone.utc).isoformat()})}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            flow_stream.unsubscribe(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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


@app.post("/api/set-polygon-key")
async def set_polygon_key(api_key: str):
    """Set Polygon.io API key for market data"""
    global finnhub_scanner
    os.environ["POLYGON_API_KEY"] = api_key
    
    # Reinitialize scanner to pick up Polygon
    finnhub_key = os.environ.get("FINNHUB_API_KEY", "")
    if not finnhub_key:
        finnhub_key = "dummy_for_polygon"
        os.environ["FINNHUB_API_KEY"] = finnhub_key
    
    try:
        finnhub_scanner = FinnhubScanner(finnhub_key)
        # Update Range Watcher with the scanner
        if set_range_scanner:
            set_range_scanner(finnhub_scanner)
        # Update Rule Engine with the scanner (for weekly structure)
        if rule_engine_available and set_scanner_for_rules:
            set_scanner_for_rules(finnhub_scanner)
        # Update Entry Scanner with the scanner
        if entry_scanner_available and set_entry_scanner:
            set_entry_scanner(finnhub_scanner)
        if finnhub_scanner.polygon_client:
            return {"status": "ok", "message": "Polygon.io data enabled!"}
        else:
            return {"status": "warning", "message": "Polygon key set but client failed to initialize"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# WEBSOCKET STREAMING ENDPOINTS
# =============================================================================

@app.post("/api/streaming/start")
async def start_streaming():
    """Start WebSocket streaming for real-time minute bars"""
    if not streaming_available or not streaming_manager:
        raise HTTPException(status_code=400, detail="WebSocket streaming not available. Install: pip install websockets")
    
    polygon_key = os.environ.get("POLYGON_API_KEY")
    if not polygon_key:
        raise HTTPException(status_code=400, detail="Polygon API key required for WebSocket streaming")
    
    try:
        # Initialize if needed
        if not streaming_manager.streamer:
            streaming_manager.initialize(polygon_key)
        
        # Start streaming in background
        streaming_manager.start()
        
        return {
            "status": "ok",
            "message": "WebSocket streaming started! 📡",
            "streaming": streaming_manager.get_status()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start streaming: {str(e)}")


@app.post("/api/streaming/stop")
async def stop_streaming():
    """Stop WebSocket streaming"""
    if not streaming_available or not streaming_manager:
        raise HTTPException(status_code=400, detail="Streaming not available")
    
    streaming_manager.stop()
    return {"status": "ok", "message": "Streaming stopped"}


@app.get("/api/streaming/status")
async def get_streaming_status():
    """Get current streaming status"""
    if not streaming_available or not streaming_manager:
        return {
            "available": False,
            "connected": False,
            "message": "WebSocket streaming not loaded"
        }
    
    status = streaming_manager.get_status()
    status["available"] = True
    return status


@app.post("/api/streaming/subscribe")
async def subscribe_symbols(symbols: List[str]):
    """Subscribe to symbols for live streaming"""
    if not streaming_available or not streaming_manager:
        raise HTTPException(status_code=400, detail="Streaming not available")
    
    if not streaming_manager.streamer:
        raise HTTPException(status_code=400, detail="Streaming not started. Call /api/streaming/start first")
    
    streaming_manager.subscribe(symbols)
    return {
        "status": "ok",
        "message": f"Subscribed to {len(symbols)} symbols",
        "symbols": symbols
    }


@app.post("/api/streaming/unsubscribe")
async def unsubscribe_symbols(symbols: List[str]):
    """Unsubscribe from symbols"""
    if not streaming_available or not streaming_manager:
        raise HTTPException(status_code=400, detail="Streaming not available")
    
    streaming_manager.unsubscribe(symbols)
    return {"status": "ok", "message": f"Unsubscribed from {len(symbols)} symbols"}


@app.get("/api/streaming/latest/{symbol}")
async def get_latest_bar(symbol: str):
    """Get the latest streamed bar for a symbol"""
    if not streaming_available or not streaming_manager:
        raise HTTPException(status_code=400, detail="Streaming not available")
    
    bar = streaming_manager.get_latest(symbol)
    if not bar:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}. Is it subscribed?")
    
    return bar


# =============================================================================
# EXTENSION DURATION PREDICTOR ENDPOINTS (THE EDGE)
# =============================================================================

@app.get("/api/extension/status")
async def get_extension_status():
    """Get Extension Predictor status"""
    return {
        "available": extension_available,
        "tracked_symbols": len(extension_predictor._streaks) if extension_predictor else 0,
        "description": "Tracks HOW LONG price has been extended from key levels"
    }


@app.get("/api/extension/analyze/{symbol}")
async def analyze_extension(symbol: str, timeframe: str = "2HR"):
    """
    Analyze extension duration for a symbol (V2).
    
    Self-contained: V2 calculates VP levels, weekly context, squeeze,
    and everything else internally from the DataFrame.
    
    Returns snap-back probability based on time extended.
    """
    if not extension_available or not extension_predictor:
        raise HTTPException(status_code=400, detail="Extension Predictor not available")
    
    try:
        # Try FinnhubScanner first, fall back to yfinance
        df = None
        try:
            scanner = get_finnhub_scanner()
            df = scanner._get_candles(symbol.upper(), "60", days_back=10)
        except Exception:
            pass
        
        if df is None or len(df) < 20:
            df = get_bars(symbol.upper(), period="1mo", interval="1h")
            if not df.empty:
                df.columns = [c.lower() for c in df.columns]
        
        if df is None or len(df) < 20:
            raise HTTPException(status_code=404, detail=f"Insufficient data for {symbol}")
        
        # Resample to requested timeframe
        resample_map = {"2HR": "2h", "4HR": "4h"}
        if timeframe.upper() in resample_map:
            df = df.resample(resample_map[timeframe.upper()]).agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum'
            }).dropna()
        
        if len(df) < 5:
            raise HTTPException(status_code=404, detail=f"Insufficient {timeframe} data for {symbol}")
        
        # V2: Full self-contained analysis
        analysis = extension_predictor.analyze(df, symbol.upper())
        
        if analysis is None:
            return {"symbol": symbol.upper(), "error": "Analysis returned no results"}
        
        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe.upper(),
            "score": analysis.extension_score,
            "quality_grade": analysis.quality_grade,
            "trigger_level": analysis.trigger_level,
            "zone": analysis.zone,
            "snap_back_probability": round(analysis.snap_back_probability, 1),
            "risk_reward": round(analysis.risk_reward, 2),
            "trade_direction": analysis.trade_direction,
            "setup_type": analysis.setup_type,
            "entry_trigger": analysis.entry_trigger,
            "current_price": round(analysis.current_price, 2),
            "snap_back_target": round(analysis.snap_back_target, 2),
            "stop_loss": round(analysis.stop_loss, 2),
            
            "levels": {
                "vwap": round(analysis.volume_profile.vwap, 2) if analysis.volume_profile else 0,
                "poc": round(analysis.volume_profile.poc, 2) if analysis.volume_profile else 0,
                "vah": round(analysis.volume_profile.vah, 2) if analysis.volume_profile else 0,
                "val": round(analysis.volume_profile.val, 2) if analysis.volume_profile else 0,
            },
            
            "extension": {
                "active_streaks": analysis.active_streaks,
                "hottest_setup": analysis.hottest_streak,
                "distance_from_vwap_atr": round(analysis.distance_from_vwap_atr, 2),
            },
            
            "v2_context": {
                "rsi": round(analysis.rsi, 1),
                "rsi_extreme": analysis.rsi_extreme,
                "weekly": _safe_dict(analysis.weekly) if analysis.weekly else None,
                "squeeze": _safe_dict(analysis.squeeze) if analysis.squeeze else None,
                "trend": _safe_dict(analysis.trend) if analysis.trend else None,
                "options": _safe_dict(analysis.options) if analysis.options else None,
            },
            
            "factors": analysis.factors,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extension analysis failed: {str(e)}")


@app.get("/api/extension/alerts")
async def get_extension_alerts():
    """Get all active extension alerts across analyzed symbols"""
    if not extension_available or not extension_predictor:
        raise HTTPException(status_code=400, detail="Extension Predictor not available")
    
    alerts = []
    for symbol, streaks in extension_predictor._streaks.items():
        for key, streak in streaks.items():
            if streak.count >= 2:  # At least ALERT level
                prob = extension_predictor.BASE_PROBABILITIES.get(
                    min(streak.count, 10), 0.92
                ) * 100
                alerts.append({
                    "symbol": symbol,
                    "level": streak.level_name,
                    "direction": streak.direction,
                    "candles": streak.count,
                    "hours": streak.hours,
                    "trigger": streak.trigger.name,
                    "snap_back_prob": round(prob, 1),
                })
    
    alerts.sort(key=lambda a: a['candles'], reverse=True)
    
    return {
        "count": len(alerts),
        "alerts": alerts,
        "tracked_symbols": list(extension_predictor._streaks.keys()),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/extension/scan")
async def scan_extensions(symbols: List[str] = None):
    """
    Scan multiple symbols for extension setups (V2).
    
    Returns symbols with ALERT+ extension triggers, sorted by score.
    """
    if not extension_available or not extension_predictor:
        raise HTTPException(status_code=400, detail="Extension Predictor not available")
    
    # Get symbols from watchlist if not provided
    if not symbols:
        watchlists = watchlist_mgr.get_all_watchlists()
        symbols = []
        for wl in watchlists:
            symbols.extend(_watchlist_symbols(wl))
        symbols = list(set(symbols))[:50]  # Limit to 50
    
    # Try FinnhubScanner first, allow yfinance fallback per-symbol
    scanner = None
    try:
        scanner = get_finnhub_scanner()
    except Exception:
        pass
    
    results = []
    
    for symbol in symbols:
        try:
            df = None
            if scanner:
                df = scanner._get_candles(symbol.upper(), "60", days_back=10)
            if df is None or len(df) < 20:
                df = get_bars(symbol.upper(), period="1mo", interval="1h")
                if not df.empty:
                    df.columns = [c.lower() for c in df.columns]
            if df is None or len(df) < 20:
                continue
            
            # Resample to 2HR
            df_2h = df.resample('2h').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum'
            }).dropna()
            
            if len(df_2h) < 5:
                continue
            
            # V2: self-contained analysis
            analysis = extension_predictor.analyze(df_2h, symbol.upper())
            
            if analysis and analysis.trigger_level in ["ALERT", "HIGH_PROB", "EXTREME"]:
                results.append({
                    "symbol": symbol.upper(),
                    "score": analysis.extension_score,
                    "trigger_level": analysis.trigger_level,
                    "quality_grade": analysis.quality_grade,
                    "setup": analysis.hottest_streak if analysis.hottest_streak else {},
                    "snap_back_prob": round(analysis.snap_back_probability, 1),
                    "trade_direction": analysis.trade_direction,
                    "setup_type": analysis.setup_type,
                    "price": round(analysis.current_price, 2),
                })
                
        except Exception as e:
            print(f"Extension scan error for {symbol}: {e}")
            continue
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        "count": len(results),
        "results": results,
        "scanned": len(symbols),
        "timestamp": datetime.now().isoformat()
    }


class BatchScanRequest(BaseModel):
    symbols: List[str]
    timeframe: str = "1HR"
    scan_type: str = "all"


@app.post("/api/scan/batch")
async def batch_scan(request: BatchScanRequest):
    """
    Batch scan multiple symbols at once - OPTIMIZED
    
    scan_type: 'bullish', 'bearish', 'compression', 'structure', 'all'
    """
    try:
        symbols = request.symbols
        timeframe = request.timeframe
        scan_type = request.scan_type
        scanner = get_finnhub_scanner()
        
        results = {
            "bullish": [],
            "bearish": [],
            "compression": [],
            "structure": []
        }
        
        for symbol in symbols[:50]:  # Limit to 50
            try:
                # Get data once per symbol
                df = scanner._get_candles(symbol.upper(), "60", days_back=7)
                if df is None or len(df) < 15:
                    continue
                
                # Analyze
                result = scanner.analyze(symbol.upper(), timeframe)
                if not result:
                    continue
                
                # Bullish scan — only confirmed LONG_SETUP or strong lean (score>=60, gap>=20)
                if scan_type in ['bullish', 'all']:
                    is_bullish = result.signal and 'LONG' in str(result.signal)
                    bull = result.bull_score or 0
                    bear = result.bear_score or 0
                    score_gap = bull - bear
                    is_strong_lean = score_gap >= 20 and bull >= 60
                    
                    if is_bullish or is_strong_lean:
                        results["bullish"].append({
                            "symbol": symbol.upper(),
                            "signal": result.signal or "NEUTRAL",
                            "confidence": result.confidence or 0,
                            "bull_score": bull,
                            "bear_score": bear,
                            "position": result.position or "-"
                        })
                
                # Bearish scan — only confirmed SHORT_SETUP or strong lean (score>=60, gap>=20)
                if scan_type in ['bearish', 'all']:
                    is_bearish = result.signal and 'SHORT' in str(result.signal)
                    bull = result.bull_score or 0
                    bear = result.bear_score or 0
                    score_gap = bear - bull
                    is_strong_lean = score_gap >= 20 and bear >= 60
                    
                    if is_bearish or is_strong_lean:
                        results["bearish"].append({
                            "symbol": symbol.upper(),
                            "signal": result.signal or "NEUTRAL",
                            "confidence": result.confidence or 0,
                            "bull_score": bull,
                            "bear_score": bear,
                            "position": result.position or "-"
                        })
                
                # Compression scan
                if scan_type in ['compression', 'all']:
                    # Simple compression detection via price action
                    recent = df.tail(10)
                    if len(recent) >= 5:
                        high_range = float(recent['high'].max() - recent['low'].min())
                        avg_range = float((recent['high'] - recent['low']).mean())
                        compression_ratio = avg_range / high_range if high_range > 0 else 0
                        
                        # Bollinger squeeze detection
                        bb_width = float((recent['high'].rolling(5).mean() - recent['low'].rolling(5).mean()).iloc[-1])
                        avg_bb = float((df['high'].rolling(20).mean() - df['low'].rolling(20).mean()).mean())
                        squeeze = bb_width < avg_bb * 0.5 if avg_bb > 0 else False
                        
                        if compression_ratio > 0.7 or squeeze:
                            results["compression"].append({
                                "symbol": symbol.upper(),
                                "compression": bool(compression_ratio > 0.7),
                                "squeeze": bool(squeeze),
                                "compression_score": round(float(compression_ratio * 100), 1)
                            })
                
                # Structure scan (HH/HL or LH/LL patterns)
                if scan_type in ['structure', 'all']:
                    if len(df) >= 20:
                        highs = df['high'].rolling(5).max()
                        lows = df['low'].rolling(5).min()
                        
                        # Check last 4 swing points
                        recent_highs = highs.tail(10).dropna()
                        recent_lows = lows.tail(10).dropna()
                        
                        if len(recent_highs) >= 4 and len(recent_lows) >= 4:
                            # Bullish structure: HH + HL
                            hh = recent_highs.iloc[-1] > recent_highs.iloc[-4]
                            hl = recent_lows.iloc[-1] > recent_lows.iloc[-4]
                            
                            # Bearish structure: LH + LL
                            lh = recent_highs.iloc[-1] < recent_highs.iloc[-4]
                            ll = recent_lows.iloc[-1] < recent_lows.iloc[-4]
                            
                            if hh and hl:
                                results["structure"].append({
                                    "symbol": symbol.upper(),
                                    "structure": "bullish (HH + HL)",
                                    "trend": "uptrend"
                                })
                            elif lh and ll:
                                results["structure"].append({
                                    "symbol": symbol.upper(),
                                    "structure": "bearish (LH + LL)",
                                    "trend": "downtrend"
                                })
                                
            except Exception as e:
                print(f"Batch scan error for {symbol}: {e}")
                continue
        
        # Sort results
        results["bullish"].sort(key=lambda x: x.get('confidence', 0), reverse=True)
        results["bearish"].sort(key=lambda x: x.get('confidence', 0), reverse=True)
        results["compression"].sort(key=lambda x: x.get('compression_score', 0), reverse=True)
        
        return {
            "bullish": {"count": len(results["bullish"]), "results": results["bullish"]},
            "bearish": {"count": len(results["bearish"]), "results": results["bearish"]},
            "compression": {"count": len(results["compression"]), "results": results["compression"]},
            "structure": {"count": len(results["structure"]), "results": results["structure"]},
            "scanned": len(symbols),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Batch scan error: {e}")
        return {
            "bullish": {"count": 0, "results": []},
            "bearish": {"count": 0, "results": []},
            "compression": {"count": 0, "results": []},
            "structure": {"count": 0, "results": []},
            "scanned": 0,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.post("/api/set-openai-key")
async def set_openai_key(api_key: str):
    """Set Anthropic API key for Claude AI commentary (legacy endpoint name kept for compatibility)"""
    global anthropic_client
    os.environ["ANTHROPIC_API_KEY"] = api_key
    
    if not anthropic_available:
        raise HTTPException(status_code=400, detail="Anthropic package not installed. Run: pip install anthropic")
    
    try:
        anthropic_client = anthropic.Anthropic(api_key=api_key)
        # Test the connection
        anthropic_client.models.list()
        
        # Share Anthropic client with Range Watcher
        if set_range_openai:
            set_range_openai(anthropic_client)
        
        return {"status": "ok", "message": "Claude AI commentary enabled!"}
    except Exception as e:
        anthropic_client = None
        raise HTTPException(status_code=400, detail=f"Anthropic error: {str(e)}")


# =============================================================================
# SIGNAL-SPECIFIC AI PROMPTS - Each entry type has its own playbook
# =============================================================================

SIGNAL_PLAYBOOKS = {
    # =========================================================================
    # ENTRY SCANNER SIGNALS (from emtryscan)
    # =========================================================================
    "failed_breakout": {
        "name": "Failed Breakout (Bull Trap)",
        "direction": "SHORT",
        "setup": "Price broke ABOVE VAH but failed to hold and reversed back inside value area",
        "entry_rule": "Enter SHORT after price closes back below VAH",
        "stop_rule": "Stop above the failed breakout high (the wick that poked above VAH)",
        "target_1": "POC (fair value) - this is where trapped longs will cover",
        "target_2": "VAL (support) - full reversion to opposite side of value",
        "base_prob": "70% - failed breakouts have high success rate because longs are trapped",
        "key_confirm": "Volume should spike on the rejection candle, RSI divergence is bonus"
    },
    "failed_breakdown": {
        "name": "Failed Breakdown (Bear Trap)",
        "direction": "LONG",
        "setup": "Price broke BELOW VAL but failed to hold and reversed back inside value area",
        "entry_rule": "Enter LONG after price closes back above VAL",
        "stop_rule": "Stop below the failed breakdown low (the wick that poked below VAL)",
        "target_1": "POC (fair value) - this is where trapped shorts will cover",
        "target_2": "VAH (resistance) - full reversion to opposite side of value",
        "base_prob": "70% - failed breakdowns have high success rate because shorts are trapped",
        "key_confirm": "Volume should spike on the rejection candle, RSI divergence is bonus"
    },
    "vah_rejection": {
        "name": "VAH Rejection",
        "direction": "SHORT",
        "setup": "Price tested VAH (resistance) from below and got rejected",
        "entry_rule": "Enter SHORT on rejection candle close or break of rejection candle low",
        "stop_rule": "Stop above VAH plus a small buffer (0.2-0.3%)",
        "target_1": "POC - mean reversion target",
        "target_2": "VAL - full range target if momentum continues",
        "base_prob": "65% on first test, 50% on 2nd+ test",
        "key_confirm": "Look for upper wick rejection, volume spike, RSI overbought"
    },
    "val_touch_rejection": {
        "name": "VAL Touch Rejection",
        "direction": "LONG",
        "setup": "Price tested VAL (support) from above and bounced",
        "entry_rule": "Enter LONG on bounce candle close or break of bounce candle high",
        "stop_rule": "Stop below VAL plus a small buffer (0.2-0.3%)",
        "target_1": "POC - mean reversion target",
        "target_2": "VAH - full range target if momentum continues",
        "base_prob": "65% on first test, 50% on 2nd+ test",
        "key_confirm": "Look for lower wick rejection, volume spike, RSI oversold"
    },
    "poc_magnet": {
        "name": "POC Magnet",
        "direction": "NEUTRAL",
        "setup": "Price is gravitating toward POC (fair value) - magnetic pull effect",
        "entry_rule": "Fade moves away from POC, expect price to return to fair value",
        "stop_rule": "Stop beyond VAH (if short) or VAL (if long)",
        "target_1": "POC - the magnet level",
        "target_2": "Slight overshoot through POC possible",
        "base_prob": "70% - price returns to POC within session 70% of the time",
        "key_confirm": "Works best in range-bound, low-momentum environments"
    },
    "breakout_entry": {
        "name": "Breakout Entry",
        "direction": "LONG",
        "setup": "Clean break above VAH with volume confirmation - trend continuation",
        "entry_rule": "Enter LONG on breakout close above VAH OR on pullback to VAH (now support)",
        "stop_rule": "Stop below VAH (breakout level becomes support)",
        "target_1": "VAH + 1x value area range",
        "target_2": "VAH + 2x value area range or next resistance",
        "base_prob": "55% - breakouts have lower base rate, need volume confirmation",
        "key_confirm": "MUST have volume >1.5x average, otherwise likely false breakout"
    },
    "breakdown_entry": {
        "name": "Breakdown Entry",
        "direction": "SHORT",
        "setup": "Clean break below VAL with volume confirmation - trend continuation",
        "entry_rule": "Enter SHORT on breakdown close below VAL OR on pullback to VAL (now resistance)",
        "stop_rule": "Stop above VAL (breakdown level becomes resistance)",
        "target_1": "VAL - 1x value area range",
        "target_2": "VAL - 2x value area range or next support",
        "base_prob": "55% - breakdowns have lower base rate, need volume confirmation",
        "key_confirm": "MUST have volume >1.5x average, otherwise likely false breakdown"
    },
    
    # =========================================================================
    # SIGNAL TYPE CLASSIFICATION (from concept/signal_generator.py)
    # =========================================================================
    "long_mean_reversion": {
        "name": "Long Mean Reversion",
        "direction": "LONG",
        "setup": "Price extended BELOW VAL (support) by 1.5+ ATR - stretched rubber band effect",
        "entry_rule": "Enter LONG on rejection candle (hammer) or after price closes back above VAL",
        "stop_rule": "Stop 1 ATR below the low of the extension (or below recent swing low)",
        "target_1": "POC (fair value) - mean reversion target, ~70% of trades reach this",
        "target_2": "VAH (resistance) - full range target if momentum continues",
        "base_prob": "65-75% - extended price tends to revert to mean",
        "key_confirm": "Rejection candle (hammer), volume spike on reversal, RSI oversold (<30)"
    },
    "short_mean_reversion": {
        "name": "Short Mean Reversion",
        "direction": "SHORT",
        "setup": "Price extended ABOVE VAH (resistance) by 1.5+ ATR - stretched rubber band effect",
        "entry_rule": "Enter SHORT on rejection candle (shooting star) or after price closes back below VAH",
        "stop_rule": "Stop 1 ATR above the high of the extension (or above recent swing high)",
        "target_1": "POC (fair value) - mean reversion target, ~70% of trades reach this",
        "target_2": "VAL (support) - full range target if momentum continues",
        "base_prob": "65-75% - extended price tends to revert to mean",
        "key_confirm": "Rejection candle (shooting star), volume spike on reversal, RSI overbought (>70)"
    },
    "long_trend": {
        "name": "Long Trend Continuation",
        "direction": "LONG",
        "setup": "Price holding ABOVE VAH - value area has shifted higher, trend is up",
        "entry_rule": "Enter LONG on pullback to VAH (now support) or on break of consolidation high",
        "stop_rule": "Stop below VAH (if VAH breaks, trend thesis is invalid)",
        "target_1": "VAH + 1x value area range (measured move)",
        "target_2": "VAH + 2x value area range or next resistance level",
        "base_prob": "55-60% - trend continuation has lower base rate than mean reversion",
        "key_confirm": "VWAP rising, volume on breakout >1.5x average, higher lows forming"
    },
    "short_trend": {
        "name": "Short Trend Continuation",
        "direction": "SHORT",
        "setup": "Price holding BELOW VAL - value area has shifted lower, trend is down",
        "entry_rule": "Enter SHORT on pullback to VAL (now resistance) or on break of consolidation low",
        "stop_rule": "Stop above VAL (if VAL breaks, trend thesis is invalid)",
        "target_1": "VAL - 1x value area range (measured move)",
        "target_2": "VAL - 2x value area range or next support level",
        "base_prob": "55-60% - trend continuation has lower base rate than mean reversion",
        "key_confirm": "VWAP falling, volume on breakdown >1.5x average, lower highs forming"
    },
    "long_setup": {
        "name": "Long Setup (Generic)",
        "direction": "LONG",
        "setup": "Bullish bias detected - price structure favors upside",
        "entry_rule": "Enter LONG on pullback to support or break of resistance",
        "stop_rule": "Stop below recent swing low or key support level",
        "target_1": "Next resistance level or POC",
        "target_2": "VAH or extended target based on ATR",
        "base_prob": "55-65% - depends on specific setup quality",
        "key_confirm": "Confirm with volume, RSI not overbought, VWAP supportive"
    },
    "short_setup": {
        "name": "Short Setup (Generic)",
        "direction": "SHORT",
        "setup": "Bearish bias detected - price structure favors downside",
        "entry_rule": "Enter SHORT on rally to resistance or break of support",
        "stop_rule": "Stop above recent swing high or key resistance level",
        "target_1": "Next support level or POC",
        "target_2": "VAL or extended target based on ATR",
        "base_prob": "55-65% - depends on specific setup quality",
        "key_confirm": "Confirm with volume, RSI not oversold, VWAP resistive"
    }
}


def get_signal_specific_prompt(signal_type: str, analysis_data: dict, symbol: str) -> str:
    """Generate a signal-specific AI prompt based on the entry type"""
    
    playbook = SIGNAL_PLAYBOOKS.get(signal_type.lower())
    if not playbook:
        return None  # Use default prompt
    
    current_price = analysis_data.get("current_price", 0)
    vah = analysis_data.get("vah", 0)
    val = analysis_data.get("val", 0)
    poc = analysis_data.get("poc", 0)
    vwap = analysis_data.get("vwap", 0)
    rvol = analysis_data.get("rvol", 1.0)
    
    return f"""SPECIFIC SETUP: {playbook['name']}

🎯 THIS IS A {playbook['direction']} SETUP - Give {playbook['direction']} trade advice ONLY.

WHAT HAPPENED: {playbook['setup']}

PLAYBOOK FOR THIS SETUP:
- ENTRY RULE: {playbook['entry_rule']}
- STOP RULE: {playbook['stop_rule']}
- TARGET 1: {playbook['target_1']}
- TARGET 2: {playbook['target_2']}
- BASE PROBABILITY: {playbook['base_prob']}
- KEY CONFIRMATION: {playbook['key_confirm']}

CURRENT DATA:
- Symbol: {symbol} @ ${current_price:.2f}
- VAH: ${vah:.2f} | POC: ${poc:.2f} | VAL: ${val:.2f}
- VWAP: ${vwap:.2f}
- Volume: {rvol:.1f}x average

Calculate the specific entry, stop, and targets based on the levels above.
Follow your standard output format but USE THIS PLAYBOOK's rules for the analysis."""


def get_rule_based_commentary(analysis_data: dict, symbol: str) -> str:
    """
    Generate trading commentary using deterministic rules (zero API cost).
    Same output format as ChatGPT but with no external API calls.
    """
    if not dual_setup_available or generate_dual_setup is None:
        return ""
    
    try:
        # Build data dict for the generator
        data = {
            'symbol': symbol,
            'current_price': analysis_data.get('current_price', 0),
            'vah': analysis_data.get('vah', 0),
            'poc': analysis_data.get('poc', 0),
            'val': analysis_data.get('val', 0),
            'vwap': analysis_data.get('vwap', 0),
            'bull_score': analysis_data.get('bull_score', 0),
            'bear_score': analysis_data.get('bear_score', 0),
            'rsi': analysis_data.get('rsi', 50),
            'rvol': analysis_data.get('rvol', 1.0),
            'atr': analysis_data.get('atr', 0),
            'order_flow': analysis_data.get('order_flow', {})
        }
        
        # Generate dual setup text
        commentary = generate_dual_setup(data)
        return commentary
        
    except Exception as e:
        print(f"❌ Rule-based commentary error: {e}")
        import traceback
        traceback.print_exc()
        return ""


def get_ai_commentary(analysis_data: dict, symbol: str, entry_signal: str = None) -> str:
    """Generate AI trading commentary using Claude"""
    if AI_KILL_SWITCH or anthropic_client is None:
        return get_rule_based_commentary(analysis_data, symbol)
    
    try:
        # Check if we have a specific entry signal with its own playbook
        signal_type = None
        forced_direction = None
        if entry_signal:
            parts = entry_signal.split(':')
            signal_type = parts[0] if len(parts) > 0 else None
            forced_direction = parts[1].upper() if len(parts) > 1 else None
        
        # If no entry_signal, use signal_type from analysis (MR vs Trend classification)
        if not signal_type:
            signal_type = analysis_data.get("signal_type", None)
        
        # Build context for GPT
        signal = analysis_data.get("signal", "UNKNOWN")
        confidence = analysis_data.get("confidence", 0)
        position = analysis_data.get("position", "")
        vwap_zone = analysis_data.get("vwap_zone", "")
        rsi_zone = analysis_data.get("rsi_zone", "")
        notes = analysis_data.get("notes", [])
        bull_score = analysis_data.get("bull_score", 0)
        bear_score = analysis_data.get("bear_score", 0)
        
        # Get price levels from analysis
        current_price = analysis_data.get("current_price", 0)
        vah = analysis_data.get("vah", 0)
        val = analysis_data.get("val", 0)
        poc = analysis_data.get("poc", 0)
        vwap = analysis_data.get("vwap", 0)
        high_prob = analysis_data.get("high_prob", 0)
        low_prob = analysis_data.get("low_prob", 0)
        
        # Get Extension Predictor data (THE EDGE)
        extension_text = ""
        extension_data = analysis_data.get("extension", {})
        hottest = extension_data.get("hottest_setup") if extension_data else None
        if hottest and hottest.get("trigger") not in ["NONE", "WATCHING", None]:
            extension_text = f"""

⏱️ EXTENSION DURATION PREDICTOR (THE EDGE):
- Trigger Level: {hottest.get('trigger', 'NONE')}
- Extended From: {hottest.get('level', 'N/A').upper() if hottest.get('level') else 'N/A'}
- Duration: {hottest.get('candles', 0)} candles ({hottest.get('hours', 0)}h)
- Snap-Back Probability: {hottest.get('snap_back_prob', 0)}%
- Direction: {hottest.get('direction', 'N/A')}
⚠️ EXTENDED: Snap-back {hottest.get('snap_back_prob', 0)}% likely - wait for pullback!
"""
        
        # Get Fibonacci levels and confluence
        fib_text = ""
        fib_levels = analysis_data.get("fib_levels", {})
        if fib_levels:
            fib_position = analysis_data.get("fib_position", "N/A")
            fib_confluence = analysis_data.get("fib_confluence", [])
            fib_text = f"""

📐 FIBONACCI RETRACEMENT (15-Day Swing):
Swing High: ${fib_levels.get('swing_high', 0):.2f} | Swing Low: ${fib_levels.get('swing_low', 0):.2f}
Fib 23.6%: ${fib_levels.get('fib_236', 0):.2f} | Fib 38.2%: ${fib_levels.get('fib_382', 0):.2f} | Fib 50%: ${fib_levels.get('fib_500', 0):.2f}
Fib 61.8%: ${fib_levels.get('fib_618', 0):.2f} | Fib 78.6%: ${fib_levels.get('fib_786', 0):.2f}
📍 Price Position: {fib_position}
{"🎯 VP+FIB CONFLUENCE: " + "; ".join(fib_confluence) if fib_confluence else ""}
"""
        
        # Get trade scenarios
        trade_scenarios_text = ""
        trade_scenarios = analysis_data.get("trade_scenarios", {})
        if trade_scenarios:
            long = trade_scenarios.get("long", {})
            short = trade_scenarios.get("short", {})
            decision = trade_scenarios.get("decision_point", {})
            
            # Calculate aggressive R:R (only when valid)
            long_agg_valid = long.get("aggressive_valid", False)
            long_agg_entry = long.get("aggressive_entry") or 0
            long_agg_stop = long.get("aggressive_stop", 0)
            long_target = long.get("target", 0)
            long_agg_risk = long_agg_entry - long_agg_stop if long_agg_entry and long_agg_stop else 0
            long_agg_reward = long_target - long_agg_entry if long_target and long_agg_entry else 0
            long_agg_rr = f"{long_agg_reward / long_agg_risk:.1f}:1" if long_agg_risk > 0 else "N/A"
            long_agg_risk_pct = ((long_agg_entry - long_agg_stop) / long_agg_entry * 100) if long_agg_entry > 0 else 0
            
            short_agg_valid = short.get("aggressive_valid", False)
            short_agg_entry = short.get("aggressive_entry") or 0
            short_agg_stop = short.get("aggressive_stop", 0) 
            short_target = short.get("target", 0)
            short_agg_risk = short_agg_stop - short_agg_entry if short_agg_stop and short_agg_entry else 0
            short_agg_reward = short_agg_entry - short_target if short_agg_entry and short_target else 0
            short_agg_rr = f"{short_agg_reward / short_agg_risk:.1f}:1" if short_agg_risk > 0 else "N/A"
            short_agg_risk_pct = ((short_agg_stop - short_agg_entry) / short_agg_entry * 100) if short_agg_entry > 0 else 0
            
            long_agg_text = f"Entry ${long_agg_entry:.2f} NOW | Stop ${long_agg_stop:.2f} ({long_agg_risk_pct:.1f}% risk) | R:R {long_agg_rr}" if long_agg_valid else "INVALID — price below stop (wait for entry zone)"
            short_agg_text = f"Entry ${short_agg_entry:.2f} NOW | Stop ${short_agg_stop:.2f} ({short_agg_risk_pct:.1f}% risk) | R:R {short_agg_rr}" if short_agg_valid else "INVALID — price above stop (wait for entry zone)"
            
            trade_scenarios_text = f"""

📊 PRE-CALCULATED TRADE SCENARIOS:
📍 DECISION POINT: Bull Above ${decision.get('bull_trigger', 0):.2f} | Bear Below ${decision.get('bear_trigger', 0):.2f}

🟢 LONG:
   Conservative: Entry ${long.get('entry_zone', ['0','0'])[0]} - ${long.get('entry_zone', ['0','0'])[1]} | Stop ${long.get('stop_loss', 0):.2f} | Target ${long.get('target', 0):.2f} | R:R {long.get('r_r_ratio', 'N/A')}
   ⚡ AGGRESSIVE: {long_agg_text}

🔴 SHORT:
   Conservative: Entry ${short.get('entry_zone', ['0','0'])[0]} - ${short.get('entry_zone', ['0','0'])[1]} | Stop ${short.get('stop_loss', 0):.2f} | Target ${short.get('target', 0):.2f} | R:R {short.get('r_r_ratio', 'N/A')}
   ⚡ AGGRESSIVE: {short_agg_text}
"""
        
        # Determine direction - priority: forced > signal_type playbook > score-based
        score_gap = abs(bull_score - bear_score)
        is_clear_direction = score_gap >= 15  # Clear winner if 15+ point gap
        
        if forced_direction:
            primary_direction = f"{forced_direction} (from entry scanner)"
            direction_note = f"Entry scanner detected {signal_type.replace('_', ' ').upper()} - this is a {forced_direction} setup."
            is_clear_direction = True  # Entry scanner = clear direction
        elif signal_type and signal_type.lower() in SIGNAL_PLAYBOOKS:
            playbook = SIGNAL_PLAYBOOKS[signal_type.lower()]
            primary_direction = f"{playbook['direction']} ({playbook['name']})"
            direction_note = f"This is a {playbook['name']} setup. {playbook['setup']}"
            is_clear_direction = True  # Signal playbook = clear direction
        elif bear_score > bull_score:
            primary_direction = "SHORT (bearish)"
            direction_note = "Bear score is higher - this is a SHORT/SELL setup. Give SHORT trade advice."
        elif bull_score > bear_score:
            primary_direction = "LONG (bullish)"
            direction_note = "Bull score is higher - this is a LONG/BUY setup. Give LONG trade advice."
        else:
            primary_direction = "NEUTRAL"
            direction_note = "Scores are equal - no clear direction. Likely NO TRADE."
            is_clear_direction = False
        
        # Use signal-specific prompt if available
        specific_prompt = get_signal_specific_prompt(signal_type, analysis_data, symbol) if signal_type else None
        
        if specific_prompt:
            prompt = specific_prompt + extension_text
        else:
            # Default prompt for regular analysis
            prompt = f"""ANALYZE: {symbol} @ ${current_price:.2f}

⚠️ DIRECTION: {primary_direction}
{direction_note}

SIGNAL: {signal} | Confidence: {confidence}% | Bull: {bull_score} | Bear: {bear_score}
POSITION: {position} | VWAP Zone: {vwap_zone} | RSI: {rsi_zone}
SCENARIO BIAS: High {high_prob:.0f}% vs Low {low_prob:.0f}% → {'STRONG BULL' if high_prob - low_prob >= 25 else 'LEAN BULL' if high_prob - low_prob >= 10 else 'NEUTRAL' if abs(high_prob - low_prob) < 10 else 'LEAN BEAR' if low_prob - high_prob >= 10 else 'STRONG BEAR'} (structural probability from MTF volume profile confluence)

LEVELS: VAH ${vah:.2f} | POC ${poc:.2f} | VAL ${val:.2f} | VWAP ${vwap:.2f}

{"⚠️ DIRECTIONAL SETUP: Focus ONLY on the " + primary_direction.split()[0].upper() + " direction. Output a single detailed setup for this direction." if is_clear_direction else "⚠️ CONFLICTING SIGNALS: Bull/Bear scores within 15 points - output BOTH 🟢 LONG and 🔴 SHORT setups so trader can see both sides."}
{extension_text}
{fib_text}
{trade_scenarios_text}
Use the Fib levels and pre-calculated scenarios for your entries/stops/targets.

⚠️ MANDATORY: You MUST output BOTH a 🟢 LONG SETUP and a 🔴 SHORT SETUP. If one direction is weak, grade it C or F but STILL output it. The trader needs both sides."""

        # Comprehensive system prompt with all rules
        system_prompt = """You are an expert quantitative trading analyst. Follow this EXACT decision process:

═══════════════════════════════════════════
DECISION TREE (Follow in order - STOP at first NO)
═══════════════════════════════════════════
1. CONFLICTING SIGNALS? (Bull/Bear scores within 10) → NO TRADE
2. EXTENDED > 75% snap-back? → WAIT (don't enter now, give pullback target)
3. PROBABILITY > 55%? No → NO TRADE
4. R:R > 2:1? No → NO TRADE
5. EV POSITIVE? No → NO TRADE
6. ALL PASS? → TRADE with position size based on confidence

═══════════════════════════════════════════
PROBABILITY RULES (Memorize these)
═══════════════════════════════════════════
VOLUME PROFILE BASE RATES:
- First test of VAH (from below): 65% rejection
- First test of VAL (from above): 65% bounce
- After 2+ tests: drops to 50%
- Virgin/untested levels: 75% reaction
- POC: 70% price returns within session

LEVEL AGE DECAY:
- Today's level: full probability
- Yesterday's: -10%
- 3+ days old: -20%

MTF ALIGNMENT BONUS:
- All timeframes aligned: +15%
- 2 of 3 aligned: +0%
- Conflicting: -15% (likely NO TRADE)

OTHER ADJUSTMENTS:
- Above VWAP in uptrend: +5%
- Below VWAP in downtrend: +5%
- Extended from mean (>2 ATR): -10%
- Rejection candle forming: +10%

═══════════════════════════════════════════
POSITION SIZING RULES
═══════════════════════════════════════════
- HIGH confidence (prob 70%+): 1.0R (1% account risk)
- MEDIUM confidence (55-70%): 0.75R
- Extended (snap-back >70%): 0.5R max
- Multiple concerns: 0.5R or pass

═══════════════════════════════════════════
FIBONACCI RULES
═══════════════════════════════════════════
- Fib 38.2%-50%: Healthy pullback zone for continuation
- Fib 50%-61.8%: GOLDEN ZONE - highest probability reversal
- VP + Fib confluence (<1.5% apart): HIGH CONVICTION levels
- Use Fib levels for entries, stops, and targets

═══════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════
⚠️ CRITICAL: 
- If primary_direction is CLEAR (LONG or SHORT), output ONLY that direction's setup
- If CONFLICTING signals (scores within 15 points), output BOTH setups
- Never output a setup that contradicts the technical scores

🟢 LONG SETUP
⭐ GRADE: [A+ / A / B / C / F] | 🎯 CONVICTION: X/10
📈 PROBABILITY: X-Y% [High/Med/Low]
📍 ENTRY: $XX.XX - $XX.XX (conservative, wait for pullback)
⚡ AGGRESSIVE: $XX.XX NOW (XX% risk, higher R:R but riskier)
🛑 STOP: $XX.XX (below Fib/VP level)
💰 T1: $XX.XX | 🚀 T2: $XX.XX
📐 R:R: X.X:1 | 💹 EV: $X.XX/100
✅ TRIGGER: [What confirms this]
❌ INVALID: [What kills this]
💡 WHY: [1 sentence with VP/Fib reference]

🔴 SHORT SETUP
⭐ GRADE: [A+ / A / B / C / F] | 🎯 CONVICTION: X/10
📈 PROBABILITY: X-Y% [High/Med/Low]
📍 ENTRY: $XX.XX - $XX.XX (conservative, wait for rally)
⚡ AGGRESSIVE: $XX.XX NOW (XX% risk, higher R:R but riskier)
🛑 STOP: $XX.XX (above Fib/VP level)
💰 T1: $XX.XX | 🚀 T2: $XX.XX
📐 R:R: X.X:1 | 💹 EV: $X.XX/100
✅ TRIGGER: [What confirms this]
❌ INVALID: [What kills this]
💡 WHY: [1 sentence with VP/Fib reference]

⚖️ VERDICT: [LONG or SHORT] preferred because [reason]
⚠️ KEY LEVEL: $XX.XX - Above = Long, Below = Short

📊 BOOKMAP ORDER FLOW CHECKLIST (confirm before entry):
🔍 LONG: Look for absorption at $XX.XX (buyers absorbing sellers), delta flip positive, iceberg bids
🔍 SHORT: Look for Follow the directional instruction in the prompt. If it says "Focus ONLY on LONG/SHORT", output a single setup. If it says "CONFLICTING", output both

🚨 FINAL REMINDER: Output BOTH 🟢 LONG SETUP and 🔴 SHORT SETUP sections. Do NOT skip the SHORT section."""

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1400,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        ai_text = response.content[0].text.strip()
        
        # Persist AI suggestion to Firestore
        _save_ai_suggestion_bg(symbol, "commentary", ai_text, {
            "model": "claude-sonnet-4-20250514",
            "signal": analysis_data.get("signal"),
            "confidence": analysis_data.get("confidence"),
            "price": analysis_data.get("current_price")
        })
        
        return ai_text
    
    except Exception as e:
        print(f"⚠️ Claude AI error: {e}")
        return ""


# =============================================================================
# AI KILL SWITCH ENDPOINTS
# =============================================================================

@app.get("/api/config/ai-kill-switch")
async def get_kill_switch():
    """Get current AI kill switch status"""
    return {
        "killed": AI_KILL_SWITCH,
        "reason": AI_KILL_SWITCH_REASON,
        "toggled_at": AI_KILL_SWITCH_TOGGLED_AT,
        "fallback": "deterministic_rules",
        "ai_providers": {
            "claude": bool(anthropic_client),
            "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY"))
        }
    }


@app.post("/api/config/ai-kill-switch")
async def set_kill_switch(request: Request):
    """
    Toggle AI kill switch ON/OFF.
    
    Body: { "killed": true/false, "reason": "optional reason" }
    
    When killed=true:
    - All LLM API calls (OpenAI, Claude) are skipped
    - Trade plans use deterministic rule engine only
    - AI commentary falls back to rule-based dual setup generator
    - Hybrid router falls back to keyword classification
    - Card builder skips MTF AI call
    - Zero API cost mode
    """
    global AI_KILL_SWITCH, AI_KILL_SWITCH_REASON, AI_KILL_SWITCH_TOGGLED_AT
    
    body = await request.json()
    killed = body.get("killed", not AI_KILL_SWITCH)  # Toggle if not specified
    reason = body.get("reason", "")
    
    AI_KILL_SWITCH = bool(killed)
    AI_KILL_SWITCH_REASON = reason if killed else ""
    AI_KILL_SWITCH_TOGGLED_AT = datetime.now().isoformat()
    
    status = "🔴 AI KILLED" if AI_KILL_SWITCH else "🟢 AI ENABLED"
    print(f"{status} — {reason or 'manual toggle'} at {AI_KILL_SWITCH_TOGGLED_AT}")
    
    return {
        "killed": AI_KILL_SWITCH,
        "reason": AI_KILL_SWITCH_REASON,
        "toggled_at": AI_KILL_SWITCH_TOGGLED_AT,
        "message": f"AI {'disabled — using deterministic rules' if AI_KILL_SWITCH else 'enabled — LLM calls active'}"
    }


@app.get("/api/ai-suggestions")
async def get_ai_suggestions(
    symbol: str = Query(None, description="Filter by ticker symbol"),
    suggestion_type: str = Query(None, description="Filter by type: mtf_plan, commentary, quick_commentary, full_analysis, rule_explanation, rule_based_fallback"),
    limit: int = Query(30, description="Max results")
):
    """
    Retrieve saved AI suggestions from Firestore.
    
    All AI outputs (GPT trade plans, Claude analysis, rule-based fallbacks)
    are persisted here for learning and audit trail.
    """
    try:
        from firestore_store import get_firestore
        fs = get_firestore()
        if not fs.is_available():
            return {"suggestions": [], "message": "Firestore not available"}
        
        suggestions = fs.get_ai_suggestions(symbol, suggestion_type, limit)
        return {
            "count": len(suggestions),
            "suggestions": suggestions
        }
    except Exception as e:
        return {"suggestions": [], "error": str(e)}


@app.get("/api/ai-suggestions/{doc_id}")
async def get_ai_suggestion_detail(doc_id: str):
    """Get a specific AI suggestion by document ID"""
    try:
        from firestore_store import get_firestore
        fs = get_firestore()
        if not fs.is_available():
            raise HTTPException(status_code=503, detail="Firestore not available")
        
        suggestion = fs.get_ai_suggestion_content(doc_id)
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        return suggestion
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _rule_based_mtf_plan(symbol: str, trade_tf: str = "swing", entry_signal: str = None) -> dict:
    """
    Deterministic MTF trade plan fallback when AI is killed.
    Uses dual setup generator + rule engine instead of GPT.
    """
    scanner = get_finnhub_scanner()
    result = scanner.analyze_mtf(symbol.upper())
    if not result:
        raise HTTPException(status_code=404, detail=f"Could not analyze {symbol}")
    
    # Get VP levels
    df = scanner._get_candles(symbol.upper(), "60", 20)
    if df is not None and len(df) >= 5:
        poc, vah, val = scanner.calc.calculate_volume_profile(df)
        vwap = scanner.calc.calculate_vwap(df)
        rsi = scanner.calc.calculate_rsi(df)
        rvol = scanner.calc.calculate_relative_volume(df)
        volume_trend = scanner.calc.calculate_volume_trend(df)
        current_price = float(df['close'].iloc[-1])
        quote = scanner.get_quote(symbol.upper())
        if quote and quote.get('current'):
            current_price = quote['current']
    else:
        poc, vah, val, vwap, rsi = 0, 0, 0, 0, 50
        rvol, volume_trend = 1.0, "neutral"
        current_price = 0
    
    # Determine direction from scores
    bull_total = result.weighted_bull or 0
    bear_total = result.weighted_bear or 0
    score_diff = bull_total - bear_total
    
    if entry_signal:
        parts = entry_signal.split(':')
        direction = parts[1].upper() if len(parts) > 1 else ("LONG" if score_diff >= 0 else "SHORT")
        leading_reason = f"VP Entry Signal: {parts[0].replace('_', ' ').title()}"
    elif score_diff > 10:
        direction = "LONG"
        leading_reason = f"Bull/Bear Score: {bull_total:.0f} vs {bear_total:.0f}"
    elif score_diff < -10:
        direction = "SHORT"
        leading_reason = f"Bull/Bear Score: {bull_total:.0f} vs {bear_total:.0f}"
    elif current_price > poc:
        direction = "SHORT"
        leading_reason = "Price above POC — mean reversion"
    else:
        direction = "LONG"
        leading_reason = "Price below POC — mean reversion"
    
    # Build rule-based commentary
    commentary_data = {
        'current_price': current_price, 'vah': vah, 'poc': poc, 'val': val,
        'vwap': vwap, 'bull_score': bull_total, 'bear_score': bear_total,
        'rsi': rsi, 'rvol': rvol, 'atr': (vah - val) * 0.3 if vah > val else current_price * 0.015,
        'order_flow': {}
    }
    rule_commentary = get_rule_based_commentary(commentary_data, symbol.upper())
    
    # Timeframe configs
    tf_config = {
        "intraday": {"label": "SAME DAY (Intraday)", "hold": "1-4 hours"},
        "swing": {"label": "3-5 DAY SWING", "hold": "3-5 days"},
        "position": {"label": "2-4 WEEK POSITION", "hold": "2-4 weeks"},
        "longterm": {"label": "1-3 MONTH SETUP", "hold": "1-3 months"},
        "investment": {"label": "6+ MONTH INVESTMENT", "hold": "6+ months"}
    }
    config = tf_config.get(trade_tf, tf_config["swing"])
    
    # Build deterministic output
    ai_text = f"""⚙️ RULE-BASED ANALYSIS (AI Kill Switch Active)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 {symbol.upper()} @ ${current_price:.2f} | {config['label']}
🎯 DIRECTION: {direction} ({leading_reason})
📈 MTF Confluence: {result.confluence_pct}% | Dominant: {result.dominant_signal}
Bull: {bull_total:.0f} | Bear: {bear_total:.0f} | RSI: {rsi:.0f} | RVOL: {rvol:.1f}x

📍 LEVELS: VAH ${vah:.2f} | POC ${poc:.2f} | VAL ${val:.2f} | VWAP ${vwap:.2f}

{rule_commentary if rule_commentary else "No dual setup available."}

⚠️ AI commentary disabled — using deterministic trade decision rules.
Toggle AI back on from the Trade Desk kill switch when ready."""

    # Persist rule-based fallback to Firestore
    _save_ai_suggestion_bg(symbol.upper(), "rule_based_fallback", ai_text, {
        "model": "kill_switch_rules",
        "trade_timeframe": config["label"],
        "leading_direction": direction,
        "price": current_price
    })

    return {
        "symbol": symbol.upper(),
        "ai_commentary": ai_text,
        "high_prob": result.high_prob,
        "low_prob": result.low_prob,
        "confluence": result.confluence_pct,
        "dominant_signal": result.dominant_signal,
        "trade_timeframe": config["label"],
        "leading_direction": direction,
        "leading_reason": leading_reason,
        "extension_override": False,
        "extension_snap_prob": None,
        "bull_score": result.weighted_bull,
        "bear_score": result.weighted_bear,
        "rvol": rvol,
        "volume_trend": volume_trend,
        "vah": vah,
        "poc": poc,
        "val": val,
        "vwap": vwap,
        "rsi": rsi,
        "current_price": current_price,
        "analysis_source": "rule_based_kill_switch"
    }


# -----------------------------------------------------------------------------
# MANUAL CHART INPUT
# -----------------------------------------------------------------------------

@app.post("/api/analyze/manual")
async def analyze_manual(request: ChartInputRequest, with_ai: bool = Query(False, description="Include ChatGPT commentary")):
    """Analyze from manually entered chart values"""
    result = chart_system.analyze(
        symbol=request.symbol,
        price=request.price,
        vah=request.vah,
        poc=request.poc,
        val=request.val,
        vwap=request.vwap,
        rsi=request.rsi,
        timeframe=request.timeframe
    )
    
    response = {
        "symbol": request.symbol,
        "timeframe": result.timeframe,
        "signal": result.signal,
        "signal_emoji": result.signal_emoji,
        "bull_score": result.bull_score,
        "bear_score": result.bear_score,
        "confidence": result.confidence,
        "high_prob": result.high_prob,
        "low_prob": result.low_prob,
        "position": result.position,
        "vwap_zone": result.vwap_zone,
        "rsi_zone": result.rsi_zone,
        "notes": result.notes
    }
    
    # Add AI commentary if requested and available
    if with_ai and anthropic_client:
        response["ai_commentary"] = get_ai_commentary(response, request.symbol)
    
    return response


@app.post("/api/analyze/manual/mtf")
async def analyze_manual_mtf(request: MTFInputRequest):
    """Analyze multiple timeframes from manual input"""
    result = chart_system.analyze_mtf(
        symbol=request.symbol,
        timeframes=request.timeframes
    )
    
    # Convert to JSON-serializable format
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
        "timestamp": result.timestamp,
        "dominant_signal": result.dominant_signal,
        "signal_emoji": result.signal_emoji,
        "confluence_pct": result.confluence_pct,
        "weighted_bull": result.weighted_bull,
        "weighted_bear": result.weighted_bear,
        "high_prob": result.high_prob,
        "low_prob": result.low_prob,
        "timeframes": tf_results,
        "notes": result.notes
    }


# -----------------------------------------------------------------------------
# LIVE DATA (FINNHUB)
# -----------------------------------------------------------------------------

@app.get("/api/quote/{symbol}")
async def get_quote(symbol: str):
    """Get real-time quote"""
    scanner = get_finnhub_scanner()
    quote = scanner.get_quote(symbol.upper())
    
    if not quote:
        raise HTTPException(status_code=404, detail=f"No quote for {symbol}")
    
    return quote


# =============================================================================
# POLYGON OPTIONS HELPERS
# =============================================================================

print(f"🔧 Deploy timestamp: {datetime.now().isoformat()}")

from polygon_options import (
    fetch_options_snapshot_filtered, parse_contract, group_by_expiration,
    async_fetch_snapshot_filtered
)
print("✅ Polygon Options API enabled")


async def get_polygon_options(symbol: str):
    """Fetch options chain via Polygon snapshot API (replaces Tradier)."""
    raw = await async_fetch_snapshot_filtered(symbol, dte_min=0, dte_max=60, strike_range_pct=0.20)
    contracts = raw.get("contracts", [])

    if not contracts:
        return {"symbol": symbol.upper(), "error": "No options available", "expirations": []}

    # Parse and group
    parsed = [parse_contract(c) for c in contracts]
    grouped = group_by_expiration(parsed)

    expirations = sorted(grouped.keys())
    results = []

    for exp in expirations[:3]:
        clist = grouped[exp]
        calls = [c for c in clist if c["contractType"] == "call"]
        puts  = [c for c in clist if c["contractType"] == "put"]

        calls_by_vol = sorted(calls, key=lambda x: x.get("dayVolume") or 0, reverse=True)[:5]
        puts_by_vol  = sorted(puts,  key=lambda x: x.get("dayVolume") or 0, reverse=True)[:5]

        total_call_vol = sum(c.get("dayVolume") or 0 for c in calls)
        total_put_vol  = sum(p.get("dayVolume") or 0 for p in puts)
        pc_ratio = round(total_put_vol / total_call_vol, 2) if total_call_vol > 0 else 0

        max_call_oi = max(calls, key=lambda x: x.get("openInterest") or 0) if calls else {}
        max_put_oi  = max(puts,  key=lambda x: x.get("openInterest") or 0) if puts  else {}

        call_ivs = [c["iv"] for c in calls if c.get("iv")]
        put_ivs  = [p["iv"] for p in puts  if p.get("iv")]
        avg_call_iv = round(sum(call_ivs) / len(call_ivs) * 100, 1) if call_ivs else 0
        avg_put_iv  = round(sum(put_ivs)  / len(put_ivs)  * 100, 1) if put_ivs  else 0

        top_calls = [{
            "strike": c.get("strike", 0),
            "lastPrice": c.get("lastPrice") or c.get("midpoint") or 0,
            "bid": c.get("bid") or 0,
            "ask": c.get("ask") or 0,
            "volume": c.get("dayVolume") or 0,
            "openInterest": c.get("openInterest") or 0,
            "impliedVolatility": c.get("iv") or 0,
        } for c in calls_by_vol]

        top_puts = [{
            "strike": p.get("strike", 0),
            "lastPrice": p.get("lastPrice") or p.get("midpoint") or 0,
            "bid": p.get("bid") or 0,
            "ask": p.get("ask") or 0,
            "volume": p.get("dayVolume") or 0,
            "openInterest": p.get("openInterest") or 0,
            "impliedVolatility": p.get("iv") or 0,
        } for p in puts_by_vol]

        results.append({
            "expiration": exp,
            "total_call_volume": total_call_vol,
            "total_put_volume": total_put_vol,
            "pc_ratio": pc_ratio,
            "max_call_oi_strike": max_call_oi.get("strike", 0),
            "max_put_oi_strike": max_put_oi.get("strike", 0),
            "avg_call_iv": avg_call_iv,
            "avg_put_iv": avg_put_iv,
            "top_calls": top_calls,
            "top_puts": top_puts,
        })

    return {
        "symbol": symbol.upper(),
        "expirations": expirations,
        "data": results,
        "source": "polygon",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/options-debug/{symbol}")
async def debug_options(symbol: str):
    """Debug endpoint — test Polygon options API directly"""
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        return {"error": "No POLYGON_API_KEY in env", "key_present": False}
    try:
        raw = await async_fetch_snapshot_filtered(symbol, dte_min=0, dte_max=30)
        return {
            "contract_count": raw.get("contractCount", 0),
            "underlying_price": raw.get("underlyingPrice"),
            "pages": raw.get("pages"),
            "key_length": len(key),
            "sample": [parse_contract(c) for c in raw.get("contracts", [])[:3]],
        }
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}


@app.get("/api/options/{symbol}")
async def get_options(symbol: str):
    """Get options chain data via Polygon (real-time)"""
    try:
        return await get_polygon_options(symbol)
    except Exception as e:
        return {"symbol": symbol.upper(), "error": str(e), "expirations": []}


# =============================================================================
# CAPITULATION DETECTOR ENDPOINT
# =============================================================================

@app.get("/api/capitulation/{symbol}")
async def get_capitulation(symbol: str):
    """
    Analyze a symbol for capitulation conditions.
    
    Capitulation = when sellers are exhausted and a bottom is forming.
    Look for: price down 20%+, volume exhaustion, RSI extreme, reversal candles.
    
    Returns tradeable=True for CLIMAX or EXHAUSTION levels.
    """
    if not capitulation_available:
        return {"error": "Capitulation detector not available", "symbol": symbol}
    
    try:
        metrics = scan_for_capitulation(symbol.upper())
        
        if metrics is None:
            return {"error": f"Could not fetch data for {symbol}", "symbol": symbol}
        
        return {
            "symbol": symbol.upper(),
            "score": int(metrics.capitulation_score),
            "level": metrics.capitulation_level.value,
            "tradeable": bool(metrics.capitulation_level.tradeable),
            
            # Price decline
            "decline_pct": float(metrics.decline_from_high_pct),
            "days_since_high": int(metrics.days_since_high),
            
            # Volume
            "rvol": float(metrics.current_rvol),
            "climax_detected": bool(metrics.climax_volume_detected),
            "volume_exhaustion": bool(metrics.volume_exhaustion),
            
            # RSI
            "rsi": float(metrics.rsi),
            "rsi_oversold": bool(metrics.rsi_oversold),
            "rsi_extreme": bool(metrics.rsi_extreme),
            "rsi_divergence": bool(metrics.rsi_divergence),
            
            # Candles
            "reversal_candle": bool(metrics.reversal_candle),
            "long_lower_wick": bool(metrics.long_lower_wick),
            
            # NEW: Additional factors
            "consecutive_down_days": int(metrics.consecutive_down_days),
            "session_context": str(metrics.session_context),
            
            # Trade levels
            "price": float(metrics.entry_zone[0]) if metrics.entry_zone else 0,
            "entry_zone": [float(x) for x in metrics.entry_zone] if metrics.entry_zone else [0, 0],
            "stop_loss": float(metrics.stop_loss),
            "target_1": float(metrics.target_1),
            "target_2": float(metrics.target_2),
            
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


# =============================================================================
# OPTIONS ANALYSIS ENDPOINT (Greeks + IV + Strategy)
# =============================================================================

@app.get("/api/options/analyze/{symbol}")
async def analyze_options_strategy(symbol: str, strategy: str = "hedged_long"):
    """
    Comprehensive options analysis with Greeks, IV metrics, and risk assessment.
    
    Args:
        symbol: Stock ticker
        strategy: Strategy type (hedged_long, call_debit, put_debit, etc.)
    
    Returns:
        - Greeks (Delta, Gamma, Theta, Vega, Rho)
        - IV Metrics (IV Rank, IV Percentile, HV vs IV)
        - Volatility regime (LOW/NORMAL/ELEVATED/HIGH/EXTREME)
        - Position sizing multiplier based on IV
        - Exit rules based on strategy + IV regime
        - Warnings (earnings, IV crush, theta burn)
        - Expected move (1SD and 2SD)
    """
    if not options_analysis_available:
        raise HTTPException(status_code=503, detail="Options analysis modules not available")
    
    try:
        symbol = symbol.upper()
        
        # 1. Get current price and historical data
        hist = get_bars(symbol, period="3mo", interval="1d")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        current_price = float(hist['Close'].iloc[-1])
        
        # 2. Get options chain via Polygon
        raw = await async_fetch_snapshot_filtered(symbol, dte_min=0, dte_max=60, strike_range_pct=0.20)
        all_contracts = raw.get("contracts", [])
        if not all_contracts:
            raise HTTPException(status_code=404, detail=f"No options available for {symbol}")

        parsed = [parse_contract(c) for c in all_contracts]
        grouped = group_by_expiration(parsed)
        expirations = sorted(grouped.keys())

        if not expirations:
            raise HTTPException(status_code=404, detail=f"No options expirations for {symbol}")

        # Find next 2-3 weeks expiration (14-30 days ideal)
        target_expiration = None
        target_dte = None
        today = datetime.now()

        for exp in expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d")
            dte = (exp_date - today).days
            if 14 <= dte <= 30:
                target_expiration = exp
                target_dte = dte
                break

        if not target_expiration:
            target_expiration = expirations[0]
            exp_date = datetime.strptime(target_expiration, "%Y-%m-%d")
            target_dte = (exp_date - today).days

        # Build DataFrames from Polygon parsed contracts (yfinance-compatible columns)
        exp_contracts = grouped[target_expiration]
        call_rows = [c for c in exp_contracts if c["contractType"] == "call"]
        put_rows  = [c for c in exp_contracts if c["contractType"] == "put"]

        if not call_rows or not put_rows:
            raise HTTPException(status_code=404, detail="Options chain is empty")

        calls = pd.DataFrame([{
            "strike": c["strike"],
            "lastPrice": c.get("lastPrice") or c.get("midpoint") or 0,
            "bid": c.get("bid") or 0,
            "ask": c.get("ask") or 0,
            "volume": c.get("dayVolume") or 0,
            "openInterest": c.get("openInterest") or 0,
            "impliedVolatility": c.get("iv") or 0,
        } for c in call_rows])

        puts = pd.DataFrame([{
            "strike": p["strike"],
            "lastPrice": p.get("lastPrice") or p.get("midpoint") or 0,
            "bid": p.get("bid") or 0,
            "ask": p.get("ask") or 0,
            "volume": p.get("dayVolume") or 0,
            "openInterest": p.get("openInterest") or 0,
            "impliedVolatility": p.get("iv") or 0,
        } for p in put_rows])
        
        # 3. Calculate IV metrics
        iv_analyzer = IVAnalyzer()
        
        # Get average IV from ATM options
        atm_calls = calls[abs(calls['strike'] - current_price) < current_price * 0.05]
        atm_puts = puts[abs(puts['strike'] - current_price) < current_price * 0.05]
        
        avg_call_iv = atm_calls['impliedVolatility'].mean() if not atm_calls.empty else 0
        avg_put_iv = atm_puts['impliedVolatility'].mean() if not atm_puts.empty else 0
        current_iv = (avg_call_iv + avg_put_iv) / 2
        
        # Build IV metrics with historical data
        iv_metrics = iv_analyzer.calculate_iv_metrics(
            symbol=symbol,
            current_iv=current_iv,
            historical_df=hist
        )
        
        # 4. Check for earnings
        earnings_info = None
        if earnings_cal:
            try:
                earnings_info = earnings_cal.get_earnings_info(symbol)
            except:
                pass
        
        # 5. Strategy-specific analysis
        strategy_data = None
        
        if strategy == "hedged_long":
            # Hedged long: ATM call + OTM put
            
            # Find ATM call
            call_strike = calls.iloc[(calls['strike'] - current_price).abs().argsort()[0]]['strike']
            call_price = calls[calls['strike'] == call_strike]['lastPrice'].values[0]
            call_iv = calls[calls['strike'] == call_strike]['impliedVolatility'].values[0]
            
            # Find OTM put (5-10% below for elevated IV, 10-15% for extreme)
            protection_pct = 0.10 if iv_metrics.iv_rank < 70 else 0.12
            target_put_strike = current_price * (1 - protection_pct)
            put_strike = puts.iloc[(puts['strike'] - target_put_strike).abs().argsort()[0]]['strike']
            put_price = puts[puts['strike'] == put_strike]['lastPrice'].values[0]
            put_iv = puts[puts['strike'] == put_strike]['impliedVolatility'].values[0]
            
            # Adjust put DTE (shorter for high IV to reduce cost)
            put_dte = target_dte if iv_metrics.iv_rank < 70 else max(7, target_dte // 2)
            
            # Create legs
            call_leg = OptionLeg(
                option_type='call',
                strike=float(call_strike),
                expiration_days=target_dte,
                premium=float(call_price),
                quantity=1,
                implied_vol=float(call_iv)
            )
            
            put_leg = OptionLeg(
                option_type='put',
                strike=float(put_strike),
                expiration_days=put_dte,
                premium=float(put_price),
                quantity=1,
                implied_vol=float(put_iv)
            )
            
            # Analyze strategy
            analyzer = OptionsStrategyAnalyzer()
            analysis = analyzer.analyze_hedged_long(
                spot_price=current_price,
                call_strike=float(call_strike),
                call_dte=target_dte,
                call_premium=float(call_price),
                call_iv=float(call_iv),
                put_strike=float(put_strike),
                put_dte=put_dte,
                put_premium=float(put_price),
                put_iv=float(put_iv)
            )
            
            # Calculate expected move (use call IV as reference)
            expected_move_1sd = current_price * call_iv * math.sqrt(target_dte / 365)
            expected_move_2sd = expected_move_1sd * 2
            
            strategy_data = {
                "call_strike": float(call_strike),
                "call_premium": float(call_price),
                "call_dte": target_dte,
                "put_strike": float(put_strike),
                "put_premium": float(put_price),
                "put_dte": put_dte,
                "total_debit": float(analysis.net_debit),
                "max_risk": float(analysis.max_loss),
                "max_profit": "Unlimited" if analysis.max_profit == float('inf') else float(analysis.max_profit),
                "breakeven": float(analysis.breakeven_points[0]) if analysis.breakeven_points else 0,
                "portfolio_greeks": {
                    "delta": round(analysis.total_delta, 3),
                    "gamma": round(analysis.total_gamma, 4),
                    "theta": round(analysis.total_theta, 2),
                    "vega": round(analysis.total_vega, 2),
                },
                "warnings": analysis.warnings,
                "expected_move_1sd": round(expected_move_1sd, 2),
                "expected_move_2sd": round(expected_move_2sd, 2)
            }
        
        # 6. Position sizing multiplier based on IV regime
        position_size_multiplier = 1.0
        if iv_metrics.iv_rank > 70:
            position_size_multiplier = 0.5  # Reduce size in high IV
        elif iv_metrics.iv_rank > 80:
            position_size_multiplier = 0.33  # Significantly reduce in extreme IV
        elif iv_metrics.iv_rank < 30:
            position_size_multiplier = 1.5  # Can increase in low IV (cheap options)
        
        # 7. Exit rules based on strategy + IV regime
        exit_rules = []
        
        if strategy == "hedged_long":
            # Time-based exits
            if target_dte <= 10:
                exit_rules.append("Close 2 days before expiration to avoid gamma risk")
            else:
                exit_rules.append(f"Close call at {target_dte - 3} DTE if not profitable")
            
            # Profit targets
            if iv_metrics.iv_regime in ["HIGH", "EXTREME"]:
                exit_rules.append("Take 50% off at 50% profit (IV crush risk)")
                exit_rules.append("Full exit at 100% profit or -30% loss")
            else:
                exit_rules.append("Take 50% off at 100% profit")
                exit_rules.append("Full exit at 200% profit or -50% loss")
            
            # Greeks-based
            if strategy_data and strategy_data['portfolio_greeks']['theta'] < -50:
                exit_rules.append(f"High theta burn (${abs(strategy_data['portfolio_greeks']['theta']):.0f}/day) - consider exiting if no movement in 3 days")
            
            # IV regime specific
            if iv_metrics.iv_regime == "EXTREME":
                exit_rules.append("Exit immediately after earnings announcement (IV crush expected)")
            
            # Earnings proximity
            if earnings_info and earnings_info.days_until <= 7:
                exit_rules.append(f"EARNINGS IN {earnings_info.days_until} DAYS - Exit before or accept 30-50% IV crush")
        
        # 8. Compile response
        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "iv_metrics": {
                "current_iv": round(iv_metrics.current_iv * 100, 1),
                "iv_rank": round(iv_metrics.iv_rank, 1),
                "iv_percentile": round(iv_metrics.iv_percentile, 1),
                "hv_20d": round(iv_metrics.hv_20 * 100, 1),
                "hv_vs_iv": round(iv_metrics.hv_vs_iv, 3),
                "regime": iv_metrics.iv_regime,
                "strategy_bias": iv_metrics.strategy_bias,
                "z_score": round(iv_metrics.z_score, 2)
            },
            "earnings": {
                "date": earnings_info.date if earnings_info else None,
                "days_until": earnings_info.days_until if earnings_info else None,
                "timing": earnings_info.timing if earnings_info else None,
                "warning": f"⚠️ EARNINGS IN {earnings_info.days_until} DAYS" if earnings_info and earnings_info.days_until <= 7 else None
            },
            "strategy": {
                "type": strategy,
                "expiration": target_expiration,
                "dte": target_dte,
                **strategy_data
            } if strategy_data else {"type": strategy, "error": "Strategy analysis not available"},
            "position_sizing": {
                "multiplier": position_size_multiplier,
                "reasoning": f"IV Rank {iv_metrics.iv_rank:.0f}% - {'Reduce size' if position_size_multiplier < 1 else 'Normal size' if position_size_multiplier == 1 else 'Can increase size'}"
            },
            "exit_rules": exit_rules,
            "iv_warnings": iv_metrics.warnings,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }


# =============================================================================
# EUPHORIA DETECTOR ENDPOINT (SHORT setups - catching tops)
# =============================================================================

@app.get("/api/euphoria/{symbol}")
async def get_euphoria(symbol: str):
    """
    Analyze a symbol for euphoria/blow-off top conditions (SHORT setups).
    
    Euphoria = when buyers are exhausted and a top is forming.
    Look for: price up 20%+, buying climax, RSI extreme overbought, bearish reversal candles.
    
    Returns tradeable=True for CLIMAX or EXHAUSTION levels.
    """
    if not capitulation_available:
        return {"error": "Euphoria detector not available", "symbol": symbol}
    
    try:
        metrics = scan_for_euphoria(symbol.upper())
        
        if metrics is None:
            return {"error": f"Could not fetch data for {symbol}", "symbol": symbol}
        
        return {
            "symbol": symbol.upper(),
            "direction": "SHORT",
            "score": int(metrics.euphoria_score),
            "level": metrics.euphoria_level.value,
            "tradeable": bool(metrics.euphoria_level.tradeable),
            
            # Price advance
            "advance_pct": float(metrics.advance_from_low_pct),
            "days_since_low": int(metrics.days_since_low),
            
            # Volume
            "rvol": float(metrics.current_rvol),
            "climax_detected": bool(metrics.climax_volume_detected),
            "volume_exhaustion": bool(metrics.volume_exhaustion),
            
            # RSI
            "rsi": float(metrics.rsi),
            "rsi_overbought": bool(metrics.rsi_overbought),
            "rsi_extreme": bool(metrics.rsi_extreme),
            "rsi_divergence": bool(metrics.rsi_divergence),
            
            # Candles
            "reversal_candle": bool(metrics.reversal_candle),
            "long_upper_wick": bool(metrics.long_upper_wick),
            
            # Additional factors
            "consecutive_up_days": int(metrics.consecutive_up_days),
            "at_resistance_level": bool(metrics.at_resistance_level),
            "session_context": str(metrics.session_context),
            
            # Trade levels (SHORT)
            "price": float(metrics.entry_zone[0]) if metrics.entry_zone else 0,
            "entry_zone": [float(x) for x in metrics.entry_zone] if metrics.entry_zone else [0, 0],
            "stop_loss": float(metrics.stop_loss),  # Above current price
            "target_1": float(metrics.target_1),
            "target_2": float(metrics.target_2),
            
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


# =============================================================================
# SQUEEZE DETECTOR ENDPOINT
# =============================================================================

@app.get("/api/squeeze/{symbol}")
async def get_squeeze(symbol: str):
    """
    Analyze a symbol for volatility squeeze conditions.
    
    Enhanced squeeze detection with:
    - TTM Squeeze (BB inside Keltner) - The gold standard
    - ATR Compression
    - ADX < 20 (low directional movement)
    - RSI neutral zone
    - Squeeze duration
    - Direction bias prediction
    
    Tiers:
    - FORMING (50-69): Squeeze is developing
    - ACTIVE (70-84): Squeeze is tight
    - EXTREME (85+): High priority watchlist
    """
    if not squeeze_available:
        return {"error": "Squeeze detector not available", "symbol": symbol}
    
    try:
        # --- CACHE CHECK ---
        sq_cache_key = f"squeeze:{symbol.upper()}"
        cached_sq = squeeze_cache.get(sq_cache_key)
        if cached_sq:
            cached_sq['_cached'] = True
            return cached_sq
        
        detector = SqueezeDetector()
        metrics = detector.analyze(symbol.upper())
        
        if metrics is None:
            return {"error": f"Could not fetch data for {symbol}", "symbol": symbol}
        
        result = {
            "symbol": symbol.upper(),
            "score": int(metrics.score),
            "tier": str(metrics.tier),
            "quality_grade": str(metrics.quality_grade),
            "factors": list(metrics.factors),
            
            # Individual scores
            "ttm_squeeze": bool(metrics.ttm_squeeze),
            "ttm_score": int(metrics.ttm_score),
            "atr_compression": float(metrics.atr_compression),
            "atr_score": int(metrics.atr_score),
            "adx": float(metrics.adx),
            "adx_score": int(metrics.adx_score),
            "rsi": float(metrics.rsi),
            "rsi_score": int(metrics.rsi_score),
            "rsi_zone": str(metrics.rsi_zone),
            "range_vs_atr": float(metrics.range_vs_atr),
            "range_score": int(metrics.range_score),
            "rvol": float(metrics.rvol),
            "rvol_score": int(metrics.rvol_score),
            "squeeze_duration": int(metrics.squeeze_duration),
            "duration_score": int(metrics.duration_score),
            
            # Direction bias
            "direction_bias": str(metrics.direction_bias),
            "bias_score": int(metrics.bias_score),
            "bias_reasons": list(metrics.bias_reasons),
            
            # V2: Setup
            "setup_type": str(metrics.setup_type),
            "entry_trigger": str(metrics.entry_trigger),
            
            # V2: Volume Profile context
            "vp_score": int(metrics.vp_score),
            "volume_profile": _safe_dict(metrics.volume_profile) if metrics.volume_profile else None,
            
            # V2: Weekly alignment
            "weekly_score": int(metrics.weekly_score),
            
            # V2: Squeeze release
            "release": _safe_dict(metrics.release) if metrics.release else None,
            
            # V2: IV context
            "iv_percentile": round(float(metrics.iv_percentile), 1),
            "iv_score": int(metrics.iv_score),
            
            # Price levels
            "current_price": float(metrics.current_price),
            "upper_band": float(metrics.upper_band),
            "lower_band": float(metrics.lower_band),
            "atr": float(metrics.atr),
            "avg_daily_range": float(metrics.avg_daily_range),
            
            "timestamp": datetime.now().isoformat()
        }
        
        # --- CACHE STORE ---
        squeeze_cache.set(sq_cache_key, result)
        return result
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


# =============================================================================
# SQUEEZE BATCH SCAN ENDPOINT
# =============================================================================

@app.post("/api/squeeze/scan")
async def scan_squeezes(symbols: List[str] = None, min_tier: str = "FORMING"):
    """
    Batch scan symbols for squeeze setups.
    
    Returns list sorted by score, filtered by minimum tier.
    Tiers: FORMING, ACTIVE, PRIME, TEXTBOOK
    """
    if not squeeze_available:
        return {"error": "Squeeze detector not available"}
    
    # Get symbols from watchlist if not provided
    if not symbols:
        watchlists = watchlist_mgr.get_all_watchlists()
        symbols = []
        for wl in watchlists:
            symbols.extend(_watchlist_symbols(wl))
        symbols = list(set(symbols))[:50]
    
    try:
        results_list = scan_for_squeezes(symbols, min_tier=min_tier)
        
        results = []
        for m in results_list:
            results.append({
                "symbol": m.symbol,
                "score": m.score,
                "tier": m.tier,
                "quality_grade": m.quality_grade,
                "direction_bias": m.direction_bias,
                "setup_type": m.setup_type,
                "entry_trigger": m.entry_trigger,
                "ttm_squeeze": m.ttm_squeeze,
                "squeeze_duration": m.squeeze_duration,
                "current_price": round(m.current_price, 2),
                "factors": m.factors[:4],
            })
        
        return {
            "count": len(results),
            "results": results,
            "scanned": len(symbols),
            "min_tier": min_tier,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# AUTO-SCANNER ENDPOINTS (30-min background scanner)
# =============================================================================

@app.get("/api/autoscan/status")
async def autoscan_status():
    """Get auto-scanner status"""
    if not auto_scanner_available or not get_auto_scanner:
        return {"error": "Auto-scanner not available"}
    scanner = get_auto_scanner()
    if not scanner:
        return {"running": False, "message": "Auto-scanner not initialized"}
    return scanner.status

@app.post("/api/autoscan/trigger")
async def autoscan_trigger():
    """Manually trigger a scan cycle immediately"""
    if not auto_scanner_available or not get_auto_scanner:
        return {"error": "Auto-scanner not available"}
    scanner = get_auto_scanner()
    if not scanner:
        return {"error": "Auto-scanner not initialized"}
    result = await scanner.run_now()
    return {
        "message": "Scan complete",
        "squeezes": len(result.squeeze_setups),
        "setups": len(result.dual_setups),
        "capitulations": len(result.capitulation_signals),
        "euphoria": len(result.euphoria_signals),
        "errors": result.errors,
        "timestamp": result.timestamp.isoformat()
    }

@app.post("/api/autoscan/start")
async def autoscan_start():
    """Start the auto-scanner"""
    if not auto_scanner_available or not get_auto_scanner:
        return {"error": "Auto-scanner not available"}
    scanner = get_auto_scanner()
    if not scanner:
        return {"error": "Auto-scanner not initialized"}
    scanner.start()
    return {"message": "Auto-scanner started", "status": scanner.status}

@app.post("/api/autoscan/stop")
async def autoscan_stop():
    """Stop the auto-scanner"""
    if not auto_scanner_available or not get_auto_scanner:
        return {"error": "Auto-scanner not available"}
    scanner = get_auto_scanner()
    if not scanner:
        return {"error": "Auto-scanner not initialized"}
    scanner.stop()
    return {"message": "Auto-scanner stopped", "status": scanner.status}


# =============================================================================
# MTF AUCTION SCANNER V2 ENDPOINTS
# =============================================================================

@app.get("/api/mtf/scan/{symbol}")
async def mtf_scan(symbol: str):
    """
    Multi-Timeframe Auction scan for a symbol.
    
    Scans 30min, 1hr, 2hr, 4hr timeframes independently.
    Agreement = higher confidence. Disagreement = YELLOW (wait).
    
    V2: Includes volume profile, VWAP, flow, and RSI per timeframe.
    """
    if not mtf_scanner_available or not mtf_scanner:
        raise HTTPException(status_code=400, detail="MTF Auction Scanner not available")
    
    try:
        # Try FinnhubScanner first, fall back to yfinance
        df = None
        try:
            scanner = get_finnhub_scanner()
            df = scanner._get_candles(symbol.upper(), "5", days_back=10)
            if df is None or len(df) < 100:
                df = scanner._get_candles(symbol.upper(), "60", days_back=15)
        except Exception:
            pass
        
        # Polygon fallback
        if df is None or len(df) < 20:
            df = get_bars(symbol.upper(), period="1mo", interval="1h")
            if df.empty:
                raise HTTPException(status_code=404, detail=f"No data for {symbol}")
            df.columns = [c.lower() for c in df.columns]
        
        if len(df) < 20:
            raise HTTPException(status_code=404, detail=f"Insufficient data for {symbol}")
        
        result = mtf_scanner.scan(df, symbol=symbol.upper())
        
        # Build per-timeframe detail
        tf_details = {}
        for tf, analysis in result.timeframe_analyses.items():
            tf_details[tf.label] = {
                "signal": analysis.signal.value,
                "signal_emoji": analysis.signal.emoji,
                "signal_action": analysis.signal.action,
                "bull_score": round(analysis.bull_score, 1),
                "bear_score": round(analysis.bear_score, 1),
                "confidence": round(analysis.confidence, 1),
                "position": analysis.position_in_value,
                "rsi": round(analysis.rsi.value, 1),
                "rsi_zone": analysis.rsi.zone,
                "rsi_divergence": analysis.rsi.divergence,
                "flow_imbalance": round(analysis.flow.flow_imbalance, 3),
                "flow_state": analysis.flow.flow_state,
                "buy_pct": round(analysis.flow.buy_volume_pct, 1),
                "poc": round(analysis.volume_profile.poc, 2),
                "vah": round(analysis.volume_profile.vah, 2),
                "val": round(analysis.volume_profile.val, 2),
                "vwap": round(analysis.vwap.vwap, 2) if analysis.vwap else None,
                "vwap_zone": analysis.vwap.zone if analysis.vwap else None,
                "notes": analysis.notes[:3] if analysis.notes else [],
            }
        
        return {
            "symbol": symbol.upper(),
            "dominant_signal": result.dominant_signal.value,
            "dominant_emoji": result.dominant_signal.emoji,
            "dominant_action": result.dominant_signal.action,
            "confluence_score": round(result.confluence_score, 1),
            "actionable": result.actionable,
            "high_scenario_prob": round(result.high_scenario_prob * 100, 1),
            "low_scenario_prob": round(result.low_scenario_prob * 100, 1),
            "neutral_prob": round(result.neutral_prob * 100, 1),
            "summary": result.summary,
            "timeframes": tf_details,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MTF scan failed: {str(e)}")


@app.post("/api/mtf/scan/batch")
async def mtf_scan_batch(symbols: List[str] = None):
    """
    Batch MTF Auction scan across multiple symbols.
    Returns actionable setups sorted by confluence score.
    """
    if not mtf_scanner_available or not mtf_scanner:
        raise HTTPException(status_code=400, detail="MTF Auction Scanner not available")
    
    # Get symbols from watchlist if not provided
    if not symbols:
        watchlists = watchlist_mgr.get_all_watchlists()
        symbols = []
        for wl in watchlists:
            symbols.extend(_watchlist_symbols(wl))
        symbols = list(set(symbols))[:30]  # Limit — MTF scans are heavier
    
    # Try to get scanner, allow yfinance fallback per-symbol
    scanner = None
    try:
        scanner = get_finnhub_scanner()
    except Exception:
        pass
    
    results = []
    
    for sym in symbols:
        try:
            df = None
            if scanner:
                df = scanner._get_candles(sym.upper(), "60", days_back=10)
            if df is None or len(df) < 20:
                df = get_bars(sym.upper(), period="1mo", interval="1h")
                if not df.empty:
                    df.columns = [c.lower() for c in df.columns]
            if df is None or len(df) < 20:
                continue
            
            result = mtf_scanner.scan(df, symbol=sym.upper())
            
            if result.actionable:
                results.append({
                    "symbol": sym.upper(),
                    "dominant_signal": result.dominant_signal.value,
                    "dominant_emoji": result.dominant_signal.emoji,
                    "confluence_score": round(result.confluence_score, 1),
                    "high_prob": round(result.high_scenario_prob * 100, 1),
                    "low_prob": round(result.low_scenario_prob * 100, 1),
                    "summary": result.summary[:120],
                })
        except Exception as e:
            print(f"MTF batch scan error for {sym}: {e}")
            continue
    
    results.sort(key=lambda x: x['confluence_score'], reverse=True)
    
    return {
        "count": len(results),
        "results": results,
        "scanned": len(symbols),
        "timestamp": datetime.now().isoformat()
    }


# =============================================================================
# STRUCTURE REVERSAL DETECTOR ENDPOINT
# =============================================================================

@app.get("/api/structure/reversals/{symbol}")
async def structure_reversals(symbol: str, min_confidence: float = 40.0):
    """
    Structure-based reversal detection using macro structure analysis.
    
    Detects 5 types of reversals:
    1. STRUCTURE_BREAK - HH/LL counter to established trend
    2. MOMENTUM_EXHAUSTION - Failing to make new highs/lows
    3. RANGE_EXTREME_REVERSAL - At 90/10% of range with structure weakening
    4. COMPRESSION_BREAKOUT - Tight range at key support/resistance
    5. STRUCTURE_DIVERGENCE - Multi-timeframe structure conflicts
    
    Returns:
        List of ReversalAlert objects with confidence scores and actionable levels
    """
    if not structure_reversal_available:
        raise HTTPException(status_code=400, detail="Structure Reversal Detector not available")
    
    try:
        # Get data (1D for daily structure, 1wk for weekly macro)
        symbol = symbol.upper()
        df_daily = get_bars(symbol, period="3mo", interval="1d")
        df_weekly = get_bars(symbol, period="1y", interval="1wk")
        
        if df_daily.empty or len(df_daily) < 30:
            raise HTTPException(status_code=404, detail=f"Insufficient daily data for {symbol}")
        if df_weekly.empty or len(df_weekly) < 8:
            raise HTTPException(status_code=404, detail=f"Insufficient weekly data for {symbol}")
        
        # Normalize columns
        df_daily.columns = [c.lower() for c in df_daily.columns]
        df_weekly.columns = [c.lower() for c in df_weekly.columns]
        
        # Run RangeWatcher analysis
        range_result = range_watcher_analyzer.analyze(df_daily, symbol=symbol)
        
        # Get weekly structure from TechnicalCalculator
        from chart_input_analyzer import RangeContext
        from finnhub_scanner_v2 import TechnicalCalculator
        
        range_context = TechnicalCalculator.calculate_range_structure(
            df_weekly=df_weekly,
            df_daily=df_daily,
            current_price=df_daily['close'].iloc[-1]
        )
        
        # Build StructureContext from range analysis
        period_3d = range_result.periods.get(3)
        period_6d = range_result.periods.get(6)
        period_30d = range_result.periods.get(30)
        
        structure_ctx = StructureContext(
            # Weekly structure
            weekly_trend=range_context.trend,
            weekly_hh=range_context.hh_count,
            weekly_hl=range_context.hl_count,
            weekly_lh=range_context.lh_count,
            weekly_ll=range_context.ll_count,
            weekly_close_position=range_context.weekly_close_position,
            
            # Multi-period structure
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
            
            # Range metrics
            current_price=range_result.current_price,
            position_in_3d_range=period_3d.position_in_range if period_3d else 0.5,
            position_in_30d_range=period_30d.position_in_range if period_30d else 0.5,
            compression_ratio=range_context.compression_ratio,
            
            # Support/Resistance
            nearest_resistance=period_30d.nearest_resistance if period_30d else range_result.current_price * 1.05,
            nearest_support=period_30d.nearest_support if period_30d else range_result.current_price * 0.95,
        )
        
        # Get Volume Profile data for confluence (if available)
        vp_data = None
        try:
            from volume_profile import calculate_volume_profile
            vp = calculate_volume_profile(df_daily.tail(20))
            vp_data = {
                'val': vp.get('val'),
                'vah': vp.get('vah'),
                'poc': vp.get('poc')
            }
        except Exception:
            pass
        
        # Run reversal detection
        structure_reversal_detector.min_confidence = min_confidence
        alerts = structure_reversal_detector.analyze(
            df=df_daily,
            structure_context=structure_ctx,
            symbol=symbol,
            vp_data=vp_data
        )
        
        # Convert alerts to dict
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


# =============================================================================
# ABSORPTION DETECTOR ENDPOINT
# =============================================================================

@app.get("/api/absorption/{symbol}")
async def absorption_analysis(symbol: str, interval: str = "15m"):
    """
    Detects passive limit order walls absorbing aggressive flow.
    
    Identifies WHERE and WHY price moves are dying:
    - CEILING: Passive seller absorbing buyers (resistance wall)
    - FLOOR: Passive buyer absorbing sellers (support wall)
    - PINNING: Absorption on both sides (range-bound)
    
    Uses 15-min candles by default for optimal detection.
    
    Returns:
        AbsorptionResult with detected zones, scores, and trade implications
    """
    if not absorption_available:
        raise HTTPException(status_code=400, detail="Absorption Detector not available")
    
    try:
        symbol = symbol.upper()
        
        # --- CACHE CHECK ---
        abs_cache_key = f"abs:{symbol}:{interval}"
        cached_abs = absorption_cache.get(abs_cache_key)
        if cached_abs:
            cached_abs['_cached'] = True
            return cached_abs
        
        # Get intraday candles
        period_map = {"5m": "1d", "15m": "5d", "30m": "5d"}
        period = period_map.get(interval, "5d")
        
        df = get_bars(symbol, period=period, interval=interval)
        
        if df.empty or len(df) < 20:
            raise HTTPException(status_code=404, detail=f"Insufficient {interval} data for {symbol}")
        
        # Normalize columns
        df.columns = [c.lower() for c in df.columns]
        df.index.name = 'timestamp'
        
        # Get volume profile levels for context
        current_price = df['close'].iloc[-1]
        
        # Simple VP calculation (or get from existing endpoint if available)
        try:
            # Try to get existing VP data
            vp_result = await get_vp_analysis(symbol)
            vah = vp_result['vah']
            val = vp_result['val']
            poc = vp_result['poc']
            vwap = vp_result.get('vwap')
        except:
            # Fallback: estimate from price range
            price_range = df['high'].max() - df['low'].min()
            vah = current_price + price_range * 0.3
            val = current_price - price_range * 0.3
            poc = current_price
            vwap = df['close'].mean()
        
        # Run absorption analysis
        result = absorption_detector.analyze(
            df=df,
            symbol=symbol,
            vah=vah,
            poc=poc,
            val=val,
            vwap=vwap,
            fib_levels=None,  # Could add fib integration later
            exhaustion_price=None  # Could integrate with 3HER
        )
        
        # Convert to dict (with explicit type conversions for JSON serialization)
        zones_dict = []
        for zone in result.zones[:5]:  # Top 5 zones
            zones_dict.append({
                "center_price": round(float(zone.center_price), 2),
                "upper_bound": round(float(zone.upper_bound), 2),
                "lower_bound": round(float(zone.lower_bound), 2),
                "zone_width_pct": round(float(zone.zone_width_pct) * 100, 3),
                "absorption_type": zone.absorption_type.value,
                "strength": zone.strength.value,
                "status": zone.status.value,
                "total_touches": int(zone.total_touches),
                "total_volume": int(zone.total_volume),
                "rvol_ratio": round(float(zone.rvol_ratio), 2),
                "delta_imbalance": round(float(zone.delta_imbalance), 2),
                "time_spent_bars": int(zone.time_spent_bars),
                "score": int(zone.score),
                "near_vah": bool(zone.near_vah),
                "near_val": bool(zone.near_val),
                "near_poc": bool(zone.near_poc),
                "near_vwap": bool(zone.near_vwap),
                "notes": zone.notes,
                "emoji": zone.absorption_type.emoji,
                "trade_implication": zone.absorption_type.trade_implication
            })
        
        primary_zone_dict = None
        if result.primary_zone:
            pz = result.primary_zone
            primary_zone_dict = {
                "center_price": round(float(pz.center_price), 2),
                "absorption_type": pz.absorption_type.value,
                "strength": pz.strength.value,
                "status": pz.status.value,
                "total_touches": int(pz.total_touches),
                "rvol_ratio": round(float(pz.rvol_ratio), 2),
                "score": int(pz.score),
                "emoji": pz.absorption_type.emoji,
                "trade_implication": pz.absorption_type.trade_implication
            }
        
        abs_response = {
            "symbol": symbol,
            "scan_time": result.scan_time.isoformat(),
            "current_price": round(float(result.current_price), 2),
            "zones": zones_dict,
            "primary_zone": primary_zone_dict,
            "absorption_active": bool(result.absorption_active),
            "dominant_type": result.dominant_type.value,
            "scores": {
                "touch_frequency": int(result.touch_frequency_score),
                "volume_concentration": int(result.volume_concentration_score),
                "displacement_failure": int(result.displacement_failure_score),
                "delta_absorption": int(result.delta_absorption_score),
                "total": int(result.total_score),
                "label": result.score_label
            },
            "supports_exhaustion": bool(result.supports_exhaustion),
            "exhaustion_confluence_level": result.exhaustion_confluence_level,
            "notes": result.notes,
            "timestamp": datetime.now().isoformat()
        }
        
        # --- CACHE STORE ---
        absorption_cache.set(abs_cache_key, abs_response)
        return abs_response
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Absorption analysis failed: {str(e)}")


# =============================================================================
# OVERNIGHT GAP MODEL ENDPOINT
# =============================================================================


# =============================================================================
# OVERNIGHT / GAP PREDICTION V2 ENDPOINTS
# =============================================================================

@app.get("/api/overnight/{symbol}")
async def get_overnight_prediction(symbol: str):
    """
    Overnight/gap prediction for a symbol.
    
    Uses C.O.R.E. methodology: gap analysis, overnight session direction,
    prior day context, weekly alignment, RSI, squeeze, and IV context.
    
    Prediction score: 0-100 (50 = neutral, >55 = bullish, <45 = bearish).
    """
    if not overnight_available or not overnight_model:
        raise HTTPException(status_code=400, detail="Overnight Model not available")
    
    try:
        from polygon_data import get_bars
        df = get_bars(symbol.upper(), period="3mo", interval="1d")
        
        if df.empty or len(df) < 20:
            raise HTTPException(status_code=404, detail=f"Insufficient data for {symbol}")
        
        pred = overnight_model.analyze(df, symbol.upper())
        
        if pred is None:
            return {"error": f"Analysis returned no results for {symbol}", "symbol": symbol}
        
        response = {
            "symbol": symbol.upper(),
            "bias": pred.bias,
            "bias_emoji": pred.bias_emoji,
            "confidence": round(pred.confidence, 1),
            "prediction_score": pred.prediction_score,
            "quality_grade": pred.quality_grade,
            "trade_direction": pred.trade_direction,
            "setup_type": pred.setup_type,
            "entry_trigger": pred.entry_trigger,
            
            "scenarios": {
                "bull": pred.bull_scenario,
                "bear": pred.bear_scenario,
            },
            
            "key_levels": {k: round(v, 2) for k, v in pred.key_levels.items()} if pred.key_levels else {},
            
            "factors": pred.factors,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add component detail (safe serialization)
        components = {}
        components["rsi"] = round(pred.rsi, 1)
        components["rsi_zone"] = pred.rsi_zone
        
        if pred.gap:
            components["gap"] = {
                "gap_pct": round(pred.gap.gap_pct, 2),
                "gap_type": pred.gap.gap_type.value,
                "gap_atr_ratio": round(pred.gap.gap_atr_ratio, 2),
                "gap_fill_probability": pred.gap.gap_fill_probability.value,
                "gap_fill_level": round(pred.gap.gap_fill_level, 2),
            }
        
        if pred.overnight:
            components["overnight"] = {
                "direction": pred.overnight.overnight_direction,
                "range": round(pred.overnight.overnight_range, 2),
                "delta": round(pred.overnight.overnight_delta, 2),
                "high": round(pred.overnight.overnight_high, 2),
                "low": round(pred.overnight.overnight_low, 2),
                "poc": round(pred.overnight.overnight_poc, 2),
            }
        
        if pred.prior_day:
            components["prior_day"] = {
                "day_type": pred.prior_day.day_type.value,
                "close_vs_range": round(pred.prior_day.close_vs_range, 2),
                "poc": round(pred.prior_day.prior_poc, 2),
                "vah": round(pred.prior_day.prior_vah, 2),
                "val": round(pred.prior_day.prior_val, 2),
                "atr": round(pred.prior_day.atr, 2),
            }
        
        if pred.weekly:
            components["weekly"] = _safe_dict(pred.weekly)
        if pred.squeeze:
            components["squeeze"] = _safe_dict(pred.squeeze)
        if pred.options:
            components["options"] = _safe_dict(pred.options)
        
        response["components"] = components
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Overnight analysis failed: {str(e)}")


@app.post("/api/overnight/scan")
async def scan_overnight_batch(symbols: List[str] = None, min_confidence: float = 30.0):
    """
    Batch overnight/gap prediction across symbols.
    
    Returns bullish, bearish, and neutral lists sorted by conviction.
    """
    if not overnight_available or not overnight_model:
        raise HTTPException(status_code=400, detail="Overnight Model not available")
    
    # Get symbols from watchlist if not provided
    if not symbols:
        watchlists = watchlist_mgr.get_all_watchlists()
        symbols = []
        for wl in watchlists:
            symbols.extend(_watchlist_symbols(wl))
        symbols = list(set(symbols))[:40]
    
    from polygon_data import get_bars
    
    bullish = []
    bearish = []
    neutral = []
    
    for sym in symbols:
        try:
            df = get_bars(sym.upper(), period="3mo", interval="1d")
            if df.empty or len(df) < 20:
                continue
            
            pred = overnight_model.analyze(df, sym.upper())
            if pred is None or pred.confidence < min_confidence:
                continue
            
            entry = {
                "symbol": sym.upper(),
                "bias": pred.bias,
                "bias_emoji": pred.bias_emoji,
                "prediction_score": pred.prediction_score,
                "confidence": round(pred.confidence, 1),
                "quality_grade": pred.quality_grade,
                "trade_direction": pred.trade_direction,
                "setup_type": pred.setup_type,
                "factors": pred.factors[:3],
            }
            
            if pred.prediction_score > 55:
                bullish.append(entry)
            elif pred.prediction_score < 45:
                bearish.append(entry)
            else:
                neutral.append(entry)
                
        except Exception as e:
            print(f"Overnight scan error for {sym}: {e}")
            continue
    
    bullish.sort(key=lambda x: x['prediction_score'], reverse=True)
    bearish.sort(key=lambda x: x['prediction_score'])
    
    return {
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
        "total_scanned": len(symbols),
        "total_signals": len(bullish) + len(bearish),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/test-analyze/{symbol}")
async def test_analyze(symbol: str):
    """Debug endpoint for analyze"""
    try:
        scanner = get_finnhub_scanner()
        
        # Step 1: Get candles
        df = scanner._get_candles(symbol.upper(), "60", 7)
        step = "got_candles"
        
        # Step 2: Calculate levels
        if df is not None and len(df) >= 15:
            today = datetime.now().date()
            df_today = df[df.index.date == today] if hasattr(df.index, 'date') else df.tail(8)
            if len(df_today) >= 3:
                poc, vah, val = scanner.calc.calculate_volume_profile(df_today)
                vwap = scanner.calc.calculate_vwap(df_today)
            else:
                poc, vah, val = scanner.calc.calculate_volume_profile(df.tail(8))
                vwap = scanner.calc.calculate_vwap(df.tail(8))
            rsi = scanner.calc.calculate_rsi(df)
        else:
            poc, vah, val, vwap, rsi = 0, 0, 0, 0, 50
        step = "calculated_levels"
        
        # Step 3: Get quote
        quote = scanner.get_quote(symbol.upper())
        if quote and quote.get('current'):
            current_price = float(quote['current'])
            quote_source = quote.get('source', 'unknown')
        elif df is not None and len(df) > 0:
            current_price = float(df['close'].iloc[-1])
            quote_source = 'candle_fallback'
        else:
            current_price = 0
            quote_source = 'none'
        step = "got_quote"
        
        # Step 4: Analyze
        result = scanner.analyze(symbol.upper(), "1HR")
        step = "analyzed"
        
        if not result:
            return {"step": step, "error": "No result"}
        
        # Step 5: Build response exactly like analyze_live
        step = "building_response"
        response = {
            "symbol": symbol.upper(),
            "timeframe": result.timeframe,
            "signal": result.signal,
            "signal_emoji": result.signal_emoji,
            "bull_score": result.bull_score,
            "bear_score": result.bear_score,
            "confidence": result.confidence,
            "high_prob": result.high_prob,
            "low_prob": result.low_prob,
            "position": result.position,
            "vwap_zone": result.vwap_zone,
            "rsi_zone": result.rsi_zone,
            "notes": result.notes,
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
            "quote_source": quote_source,
            "vah": float(vah) if vah else 0,
            "poc": float(poc) if poc else 0,
            "val": float(val) if val else 0,
            "vwap": float(vwap) if vwap else 0,
            "rsi": float(rsi) if rsi else 50,
            "rvol": float(getattr(result, 'rvol', 1.0)),
            "volume_trend": getattr(result, 'volume_trend', 'neutral'),
            "volume_divergence": bool(getattr(result, 'volume_divergence', False)),
            "signal_type": getattr(result, 'signal_type', 'none'),
            "signal_strength": getattr(result, 'signal_strength', 'moderate'),
            "atr": float(getattr(result, 'atr', 0)),
            "extension_atr": float(getattr(result, 'extension_atr', 0)),
            "has_rejection": bool(getattr(result, 'has_rejection', False))
        }
        step = "response_built"
        
        # --- CACHE STORE ---
        analysis_cache.set(cache_key, response)
        
        return response
        
    except Exception as e:
        import traceback
        return {"error": str(e), "step": step if 'step' in dir() else "unknown", "traceback": traceback.format_exc()[:1000]}


@app.get("/api/analyze/live/{symbol}")
async def analyze_live(
    symbol: str,
    timeframe: str = Query("1HR", description="30MIN, 1HR, 2HR, 4HR, DAILY"),
    with_ai: bool = Query(True, description="Include ChatGPT commentary"),
    use_rules: bool = Query(False, description="Use rule-based analysis (zero API cost) instead of ChatGPT"),
    entry_signal: str = Query(None, description="Entry signal from scanner, e.g. 'failed_breakout:short'"),
    vp_period: str = Query("swing", description="VP lookback: 'day' (1d), 'swing' (5d), 'position' (20d), 'investment' (60d+)")
):
    """Analyze symbol with live Finnhub data (yfinance fallback if no API keys)"""
    try:
        # --- CACHE CHECK ---
        cache_key = f"{symbol.upper()}:{timeframe.upper()}:{vp_period}"
        cached = analysis_cache.get(cache_key)
        if cached and not with_ai:
            # Return cached result for non-AI requests (scanner calls)
            cached['_cached'] = True
            return cached
        
        # Try FinnhubScanner first, fall back to yfinance
        scanner = None
        use_yfinance = False
        try:
            scanner = get_finnhub_scanner()
        except Exception:
            use_yfinance = True
        
        # Map timeframe to resolution for candle fetching
        resolution_map = {
            "5MIN": "5", "15MIN": "15", "30MIN": "30",
            "1HR": "60", "2HR": "60", "4HR": "60", "DAILY": "D"
        }
        resolution = resolution_map.get(timeframe.upper(), "60")
        
        # VP_BARS: Number of bars to use for Volume Profile
        # Configurable by vp_period parameter:
        # - 'day': 1 trading day (intraday traders)
        # - 'swing': 3-5 trading days (swing traders) - DEFAULT
        # - 'position': 15-20 trading days (position traders)
        # - 'investment': 60+ trading days (investors, 3-9 months)
        
        # Base bars per timeframe for 'swing' (5 days)
        swing_bars = {
            "5MIN": 200, "15MIN": 80, "30MIN": 50,
            "1HR": 35, "2HR": 20, "4HR": 12, "DAILY": 20
        }
        
        # Multipliers for different periods
        period_multipliers = {
            "day": 0.2,        # ~1 day
            "swing": 1.0,      # ~5 days (default)
            "position": 4.0,   # ~20 days (1 month)
            "investment": 12.0 # ~60 days (3 months) - can go up to 9 months on daily
        }
        
        base_bars = swing_bars.get(timeframe.upper(), 35)
        multiplier = period_multipliers.get(vp_period.lower(), 1.0)
        VP_BARS = int(base_bars * multiplier)
        
        # For investment on daily, allow up to 200 bars (9+ months)
        if vp_period.lower() == "investment" and timeframe.upper() == "DAILY":
            VP_BARS = 200  # ~9 months of daily bars
        
        # Fetch enough days based on period - need more data for longer periods
        days_base = {"5MIN": 3, "15MIN": 5, "30MIN": 7, "1HR": 10, "2HR": 20, "4HR": 40, "DAILY": 60}
        days_multiplier = {"day": 1, "swing": 1, "position": 3, "investment": 5}
        days_back = days_base.get(timeframe.upper(), 10) * days_multiplier.get(vp_period.lower(), 1)
        
        # Cap at reasonable limits
        days_back = min(days_back, 365)  # Max 1 year of data
        
        # Get candle data — scanner or yfinance
        df = None
        if scanner:
            df = scanner._get_candles(symbol.upper(), resolution, days_back)
        
        if df is None or len(df) < 10:
            # Polygon fallback for candle data
            use_yfinance = True
            yf_interval_map = {
                "5MIN": "5m", "15MIN": "15m", "30MIN": "30m",
                "1HR": "1h", "2HR": "1h", "4HR": "1h", "DAILY": "1d"
            }
            yf_period_map = {
                "5MIN": "5d", "15MIN": "1mo", "30MIN": "1mo",
                "1HR": "1mo", "2HR": "3mo", "4HR": "3mo", "DAILY": "1y"
            }
            yf_interval = yf_interval_map.get(timeframe.upper(), "1h")
            yf_period = yf_period_map.get(timeframe.upper(), "1mo")
            # Extend period for longer VP lookbacks
            if vp_period.lower() in ("position", "investment"):
                if timeframe.upper() == "DAILY":
                    yf_period = "1y"
                elif timeframe.upper() in ("1HR", "2HR", "4HR"):
                    yf_period = "3mo"
            
            df = get_bars(symbol.upper(), period=yf_period, interval=yf_interval)
            if not df.empty:
                df.columns = [c.lower() for c in df.columns]
                # Ensure datetime index with timezone
                if df.index.tz is None:
                    df.index = df.index.tz_localize('US/Eastern')
            else:
                df = None
            print(f"📊 Polygon fallback for {symbol}: {len(df) if df is not None else 0} bars ({yf_interval})")
        
        # Resample if needed for 2HR/4HR
        resample_map = {"2HR": "2h", "4HR": "4h"}
        if df is not None and timeframe.upper() in resample_map:
            if scanner and not use_yfinance:
                df = scanner._resample_to_timeframe(df, timeframe)
            else:
                df = df.resample(resample_map[timeframe.upper()]).agg({
                    'open': 'first', 'high': 'max', 'low': 'min',
                    'close': 'last', 'volume': 'sum'
                }).dropna()
        
        # Trim to last VP_BARS for consistent VP calculation (matches Webull visible range)
        if df is not None and len(df) > VP_BARS:
            df = df.tail(VP_BARS)
        
        # DEBUG: Log how many bars we're using for VP
        print(f"✅ VP Calculation using {len(df) if df is not None else 0} bars for {symbol} {timeframe}")
        
        if df is not None and len(df) >= 10:
            calc = scanner.calc if scanner else TechnicalCalculator()
            poc, vah, val = calc.calculate_volume_profile(df)
            vwap = calc.calculate_vwap(df)
            rsi = calc.calculate_rsi(df)
        else:
            poc, vah, val, vwap, rsi = 0, 0, 0, 0, 50
        
        # Get REAL-TIME quote (Polygon paid = real-time, yfinance = delayed)
        quote = None
        if scanner:
            quote = scanner.get_quote(symbol.upper())
        if quote and quote.get('current'):
            current_price = float(quote['current'])
            quote_source = quote.get('source', 'unknown')
            # OHLC data
            day_open = float(quote.get('open')) if quote.get('open') else 0
            day_high = float(quote.get('high')) if quote.get('high') else 0
            day_low = float(quote.get('low')) if quote.get('low') else 0
            prev_close = float(quote.get('prev_close')) if quote.get('prev_close') else 0
        elif df is not None and len(df) > 0:
            current_price = float(df['close'].iloc[-1])
            quote_source = 'yfinance' if use_yfinance else 'candle_fallback'
            # Use last candle data for OHLC
            day_open = float(df['open'].iloc[-1]) if 'open' in df.columns else 0
            day_high = float(df['high'].iloc[-1]) if 'high' in df.columns else 0
            day_low = float(df['low'].iloc[-1]) if 'low' in df.columns else 0
            prev_close = float(df['close'].iloc[-2]) if len(df) > 1 else 0
        else:
            current_price = 0
            quote_source = 'none'
            day_open = day_high = day_low = prev_close = 0
        
        # Run analysis — scanner path or yfinance fallback path
        result = None
        if scanner:
            result = scanner.analyze(symbol.upper(), timeframe)
        
        if result is None and df is not None and len(df) >= 10:
            # yfinance fallback: compute indicators and call chart_system.analyze()
            calc = TechnicalCalculator()
            _rvol = calc.calculate_relative_volume(df)
            _vol_trend = calc.calculate_volume_trend(df)
            _vol_div = calc.detect_volume_divergence(df)
            _atr = calc.calculate_atr(df)
            _has_rejection = False
            if current_price and current_price < val and val > 0:
                _has_rejection = calc.is_rejection_candle(df, "bullish")
            elif current_price and current_price > vah and vah > 0:
                _has_rejection = calc.is_rejection_candle(df, "bearish")
            result = chart_system.analyze(
                symbol=symbol.upper(), price=current_price,
                vah=vah, poc=poc, val=val, vwap=vwap, rsi=rsi,
                timeframe=timeframe, rvol=_rvol,
                volume_trend=_vol_trend, volume_divergence=_vol_div,
                atr=_atr, has_rejection=_has_rejection
            )
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Could not analyze {symbol}")
        
        # Get order flow analysis (matches user's timeframe and lookback)
        order_flow = None
        try:
            if scanner:
                of_result = scanner.get_order_flow_analysis(symbol.upper(), timeframe, vp_period)
                if of_result and hasattr(of_result, 'to_dict'):
                    order_flow = of_result.to_dict()
                elif of_result and isinstance(of_result, dict):
                    order_flow = of_result
            if order_flow:
                print(f"📊 Order flow ({timeframe}/{vp_period}): {order_flow.get('flow_bias')} ({order_flow.get('buy_pressure')}% buy)")
        except Exception as e:
            print(f"⚠️ Order flow error: {e}")
        
        response = {
            "symbol": symbol.upper(),
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
            "day_open": float(day_open) if day_open else 0,
            "day_high": float(day_high) if day_high else 0,
            "day_low": float(day_low) if day_low else 0,
            "prev_close": float(prev_close) if prev_close else 0,
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
        
        # Add Extension Duration data (THE EDGE)
        if extension_available and extension_predictor and df is not None:
            try:
                df_2h = df.resample('2h').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                if len(df_2h) >= 5:
                    extension_predictor.analyze_from_dataframe(
                        symbol=symbol.upper(),
                        df=df_2h,
                        vwap=vwap,
                        poc=poc,
                        vah=vah,
                        val=val
                    )
                    
                    streaks = extension_predictor.get_active_streaks(symbol.upper())
                    hottest = extension_predictor.get_hottest_setup(symbol.upper())
                    
                    response["extension"] = {
                        "active_streaks": streaks,
                        "hottest_setup": hottest
                    }
                    
                    if hottest and hottest.get('candles', 0) >= 3:
                        extension_bonus = 10 + (hottest.get('candles', 0) - 2) * 5
                        response["extension_bonus"] = extension_bonus
                        response["notes"].append(f"🔥 Extension: {hottest.get('trigger', '')} - {hottest.get('candles', 0)} candles ({hottest.get('snap_back_prob', 0)}% snap-back)")
            except Exception as e:
                print(f"Extension analysis error: {e}")
        
        # Add Fibonacci retracement levels - match vp_period and detect real swings
        try:
            # Map vp_period to lookback days for Fib calculation
            fib_period_days = {
                "day": 5,        # 1 week for day traders
                "swing": 20,     # ~1 month for swing traders
                "position": 60,  # ~3 months for position traders
                "investment": 120 # ~6 months for investors
            }
            fib_days = fib_period_days.get(vp_period.lower(), 20)
            
            df_fib = None
            if scanner:
                df_fib = scanner._get_candles(symbol.upper(), "D", fib_days)
            if df_fib is None or len(df_fib) < 5:
                # Polygon fallback for fib data
                fib_period_yf = "6mo" if fib_days <= 120 else "1y"
                df_fib = get_bars(symbol.upper(), period=fib_period_yf, interval="1d")
                if not df_fib.empty:
                    df_fib.columns = [c.lower() for c in df_fib.columns]
                else:
                    df_fib = None
            if df_fib is not None and len(df_fib) >= 5:
                # Use simple max/min of lookback period (clear and reliable)
                # Only use the last fib_days worth of data
                df_fib = df_fib.tail(fib_days)
                
                # Get current price for sanity checking
                current_price = float(df_fib['close'].iloc[-1])
                
                # Filter out bad data points (more than 50% away from current price)
                valid_lows = df_fib['low'][df_fib['low'] > current_price * 0.5]
                valid_highs = df_fib['high'][df_fib['high'] < current_price * 2.0]
                
                swing_high = float(valid_highs.max()) if len(valid_highs) > 0 else float(df_fib['high'].iloc[-5:].max())
                swing_low = float(valid_lows.min()) if len(valid_lows) > 0 else float(df_fib['low'].iloc[-5:].min())
                fib_range = swing_high - swing_low
                
                # Additional sanity check: if range still too large, use percentile
                if fib_range > swing_high * 0.20:
                    print(f"⚠️ Fib range still large for {symbol}: {fib_range:.2f} ({fib_range/swing_high*100:.1f}%)")
                    # Use 95th percentile high and 5th percentile low to exclude outliers
                    swing_high = float(df_fib['high'].quantile(0.95))
                    swing_low = float(df_fib['low'].quantile(0.05))
                    fib_range = swing_high - swing_low
                    print(f"  Using percentile: High={swing_high:.2f}, Low={swing_low:.2f}, Range={fib_range:.2f}")
                
                # Find indices of high and low for trend detection
                high_idx = df_fib['high'].idxmax()
                low_idx = df_fib['low'].idxmin()
                
                # Convert to position in dataframe
                high_pos = list(df_fib.index).index(high_idx) if high_idx in df_fib.index else len(df_fib) - 1
                low_pos = list(df_fib.index).index(low_idx) if low_idx in df_fib.index else 0
                
                # Validate range is meaningful (at least 1%)
                if fib_range < swing_high * 0.01:
                    print(f"⚠️ Fib range too small for {symbol}: {fib_range:.2f}")
                    fib_range = swing_high * 0.05  # Use 5% as minimum
                
                # Determine trend direction (which extreme came more recently)
                # More recent high = uptrend, more recent low = downtrend
                uptrend = high_pos > low_pos
                
                # BULLISH Fibs (retracement from swing low up to swing high)
                # These are pullback levels in an uptrend
                bull_fib_236 = swing_low + (fib_range * 0.236)
                bull_fib_382 = swing_low + (fib_range * 0.382)
                bull_fib_500 = swing_low + (fib_range * 0.500)
                bull_fib_618 = swing_low + (fib_range * 0.618)
                bull_fib_786 = swing_low + (fib_range * 0.786)
                
                # BEARISH Fibs (retracement from swing high down to swing low)
                # These are pullback levels in a downtrend
                bear_fib_236 = swing_high - (fib_range * 0.236)
                bear_fib_382 = swing_high - (fib_range * 0.382)
                bear_fib_500 = swing_high - (fib_range * 0.500)
                bear_fib_618 = swing_high - (fib_range * 0.618)
                bear_fib_786 = swing_high - (fib_range * 0.786)
                
                response["fib_levels"] = {
                    "swing_high": swing_high,
                    "swing_low": swing_low,
                    "lookback_days": fib_days,
                    "trend": "UPTREND" if uptrend else "DOWNTREND",
                    # Bullish Fibs (support levels for pullbacks in uptrend)
                    "bull_fib_236": bull_fib_236,
                    "bull_fib_382": bull_fib_382,
                    "bull_fib_500": bull_fib_500,
                    "bull_fib_618": bull_fib_618,
                    "bull_fib_786": bull_fib_786,
                    # Bearish Fibs (resistance levels for bounces in downtrend)
                    "bear_fib_236": bear_fib_236,
                    "bear_fib_382": bear_fib_382,
                    "bear_fib_500": bear_fib_500,
                    "bear_fib_618": bear_fib_618,
                    "bear_fib_786": bear_fib_786,
                    # Legacy format for backwards compatibility
                    "fib_236": bear_fib_236,
                    "fib_382": bear_fib_382,
                    "fib_500": bear_fib_500,
                    "fib_618": bear_fib_618,
                    "fib_786": bear_fib_786
                }
                
                # Determine price position relative to Fib levels (use trend-appropriate fibs)
                if uptrend:
                    # In uptrend, look at bullish fib levels (support)
                    if current_price >= swing_high:
                        response["fib_position"] = "Above swing high (extended, watch for reversal)"
                    elif current_price >= bull_fib_786:
                        response["fib_position"] = "Above Bull 78.6% (strong uptrend)"
                    elif current_price >= bull_fib_618:
                        response["fib_position"] = "Above Bull 61.8% (healthy uptrend)"
                    elif current_price >= bull_fib_500:
                        response["fib_position"] = "Bull 50%-61.8% GOLDEN ZONE (best long entry)"
                    elif current_price >= bull_fib_382:
                        response["fib_position"] = "Bull 38.2%-50% (pullback entry zone)"
                    elif current_price >= bull_fib_236:
                        response["fib_position"] = "Bull 23.6%-38.2% (shallow pullback)"
                    else:
                        response["fib_position"] = "Below Bull 23.6% (trend may be broken)"
                else:
                    # In downtrend, look at bearish fib levels (resistance)
                    if current_price <= swing_low:
                        response["fib_position"] = "Below swing low (extended, watch for bounce)"
                    elif current_price <= bear_fib_786:
                        response["fib_position"] = "Below Bear 78.6% (strong downtrend)"
                    elif current_price <= bear_fib_618:
                        response["fib_position"] = "Below Bear 61.8% (healthy downtrend)"
                    elif current_price <= bear_fib_500:
                        response["fib_position"] = "Bear 50%-61.8% GOLDEN ZONE (best short entry)"
                    elif current_price <= bear_fib_382:
                        response["fib_position"] = "Bear 38.2%-50% (bounce entry zone)"
                    elif current_price <= bear_fib_236:
                        response["fib_position"] = "Bear 23.6%-38.2% (shallow bounce)"
                    else:
                        response["fib_position"] = "Above Bear 23.6% (trend may be reversing)"
                
                # ----- Fib Position Score Adjustment -----
                fib_pos = response.get("fib_position", "")
                fib_bull_adj = 0
                fib_bear_adj = 0
                
                # Detect if signal is counter-trend (mean reversion)
                # e.g. downtrend but signal is "LONG" = mean reversion long
                signal_str = str(response.get("signal", "")).upper()
                is_long_signal = "LONG" in signal_str
                is_short_signal = "SHORT" in signal_str
                is_counter_trend = (not uptrend and is_long_signal) or (uptrend and is_short_signal)
                
                if is_counter_trend:
                    # COUNTER-TREND (mean reversion) — fib zones show 
                    # where price has pulled back TO. Golden Zone = optimal 
                    # bounce/fade zone. Apply bonus to the SETUP direction.
                    if "GOLDEN ZONE" in fib_pos:
                        # Best mean-reversion entry — golden pocket
                        if is_long_signal:
                            fib_bull_adj = +18
                        else:
                            fib_bear_adj = +18
                    elif "38.2%-50%" in fib_pos:
                        # Decent pullback zone for counter-trend
                        if is_long_signal:
                            fib_bull_adj = +10
                        else:
                            fib_bear_adj = +10
                    elif "23.6%-38.2%" in fib_pos:
                        # Shallow — less ideal for bounce/fade
                        if is_long_signal:
                            fib_bull_adj = +5
                        else:
                            fib_bear_adj = +5
                    elif "61.8%" in fib_pos and "GOLDEN" not in fib_pos:
                        # Past golden zone, deeper retrace — still ok but riskier
                        if is_long_signal:
                            fib_bull_adj = +8
                        else:
                            fib_bear_adj = +8
                    elif "78.6%" in fib_pos:
                        # Very deep — extended, higher risk of trend continuation
                        if is_long_signal:
                            fib_bull_adj = +3
                        else:
                            fib_bear_adj = +3
                    elif "swing high" in fib_pos or "swing low" in fib_pos:
                        # Beyond swing — catching a falling knife / fading a rocket
                        if is_long_signal:
                            fib_bull_adj = -3
                        else:
                            fib_bear_adj = -3
                    elif "trend may be broken" in fib_pos or "trend may be reversing" in fib_pos:
                        # Trend reversing = good for counter-trend!
                        if is_long_signal:
                            fib_bull_adj = +8
                        else:
                            fib_bear_adj = +8
                else:
                    # WITH-TREND — boost the trend direction as before
                    if "GOLDEN ZONE" in fib_pos:
                        if uptrend:
                            fib_bull_adj = +15
                        else:
                            fib_bear_adj = +15
                    elif "38.2%-50%" in fib_pos:
                        if uptrend:
                            fib_bull_adj = +10
                        else:
                            fib_bear_adj = +10
                    elif "23.6%-38.2%" in fib_pos:
                        if uptrend:
                            fib_bull_adj = +5
                        else:
                            fib_bear_adj = +5
                    elif "61.8%" in fib_pos and "GOLDEN" not in fib_pos:
                        if uptrend:
                            fib_bull_adj = +8
                        else:
                            fib_bear_adj = +8
                    elif "78.6%" in fib_pos:
                        if uptrend:
                            fib_bull_adj = +5
                        else:
                            fib_bear_adj = +5
                    elif "swing high" in fib_pos or "swing low" in fib_pos:
                        if uptrend:
                            fib_bull_adj = -5
                            fib_bear_adj = +3
                        else:
                            fib_bear_adj = -5
                            fib_bull_adj = +3
                    elif "trend may be broken" in fib_pos or "trend may be reversing" in fib_pos:
                        if uptrend:
                            fib_bull_adj = -10
                            fib_bear_adj = +5
                        else:
                            fib_bear_adj = -10
                            fib_bull_adj = +5
                
                if fib_bull_adj != 0 or fib_bear_adj != 0:
                    old_bull = response["bull_score"]
                    old_bear = response["bear_score"]
                    response["bull_score"] = max(0, min(100, response["bull_score"] + fib_bull_adj))
                    response["bear_score"] = max(0, min(100, response["bear_score"] + fib_bear_adj))
                    response["fib_score_adj"] = {"bull": fib_bull_adj, "bear": fib_bear_adj}
                    
                    # Recalculate confidence and probabilities
                    new_bull = response["bull_score"]
                    new_bear = response["bear_score"]
                    gap = abs(new_bull - new_bear)
                    winning = max(new_bull, new_bear)
                    response["confidence"] = min(95, 40 + (winning * 0.5) + (gap * 0.1))
                    total = new_bull + new_bear
                    if total > 0:
                        response["high_prob"] = round(max(new_bull, new_bear) / total * 100, 1)
                        response["low_prob"] = round(min(new_bull, new_bear) / total * 100, 1)
                    
                    response["notes"].append(f"📐 Fib adjustment: bull {fib_bull_adj:+d}, bear {fib_bear_adj:+d} (from {fib_pos})")
                    print(f"📐 Fib score adj: bull {old_bull:.0f}→{new_bull:.0f} ({fib_bull_adj:+d}), bear {old_bear:.0f}→{new_bear:.0f} ({fib_bear_adj:+d})")
                
                # VP + Fib confluence detection (use active trend fibs)
                confluences = []
                active_fibs = {
                    "23.6%": bull_fib_236 if uptrend else bear_fib_236,
                    "38.2%": bull_fib_382 if uptrend else bear_fib_382,
                    "50%": bull_fib_500 if uptrend else bear_fib_500,
                    "61.8%": bull_fib_618 if uptrend else bear_fib_618,
                    "78.6%": bull_fib_786 if uptrend else bear_fib_786
                }
                
                for fib_name, fib_val in active_fibs.items():
                    if vah > 0 and abs(vah - fib_val) / vah < 0.015:
                        confluences.append(f"VAH ≈ Fib {fib_name} at ${vah:.2f}")
                    if poc > 0 and abs(poc - fib_val) / poc < 0.015:
                        confluences.append(f"POC ≈ Fib {fib_name} at ${poc:.2f}")
                    if val > 0 and abs(val - fib_val) / val < 0.015:
                        confluences.append(f"VAL ≈ Fib {fib_name} at ${val:.2f}")
                
                if confluences:
                    response["fib_confluence"] = confluences
                    response["notes"].append(f"📐 Fib Confluence: {'; '.join(confluences)}")
                
                # Calculate trade scenarios (non-bias) — ATR-aware targets
                if current_price > 0 and vah > 0 and val > 0 and poc > 0:
                    vp_range = vah - val if vah > val else 1
                    # Use ATR for realistic stop/target distances
                    scen_atr = _atr if _atr > 0 else vp_range * 0.3
                    
                    # LONG scenario — target must be ABOVE entry
                    long_entry_low = val
                    long_entry_high = poc
                    long_mid = (long_entry_low + long_entry_high) / 2
                    long_stop = long_mid - (scen_atr * 0.5)
                    # Target: whichever is higher — VAH or 1 ATR above entry
                    long_target1 = max(vah, long_mid + scen_atr)
                    long_target2 = max(bull_fib_786 if bull_fib_786 > long_target1 else swing_high, long_mid + scen_atr * 2)
                    long_risk = long_mid - long_stop
                    long_reward = long_target1 - long_mid
                    long_rr = long_reward / long_risk if long_risk > 0 else 0
                    
                    # SHORT scenario — target must be BELOW entry
                    short_entry_low = poc
                    short_entry_high = vah
                    short_mid = (short_entry_low + short_entry_high) / 2
                    short_stop = short_mid + (scen_atr * 0.5)
                    # Target: whichever is lower — VAL or 1 ATR below entry
                    short_target1 = min(val, short_mid - scen_atr)
                    short_target2 = min(bear_fib_618 if bear_fib_618 < short_target1 else swing_low, short_mid - scen_atr * 2)
                    short_risk = short_stop - short_mid
                    short_reward = short_mid - short_target1
                    short_rr = short_reward / short_risk if short_risk > 0 else 0
                    
                    # Aggressive entries - validate position relative to stops
                    agg_long_stop = current_price - (scen_atr * 0.5)
                    agg_short_stop = current_price + (scen_atr * 0.5)
                    
                    # Only valid if price won't be immediately stopped out
                    long_agg_valid = current_price > long_stop
                    short_agg_valid = current_price < short_stop
                    
                    # Aggressive R:R
                    agg_long_risk = current_price - agg_long_stop
                    agg_long_reward = long_target1 - current_price
                    agg_long_rr = agg_long_reward / agg_long_risk if agg_long_risk > 0 else 0
                    agg_long_risk_pct = (scen_atr * 0.5 / current_price) * 100
                    
                    agg_short_risk = agg_short_stop - current_price
                    agg_short_reward = current_price - short_target1
                    agg_short_rr = agg_short_reward / agg_short_risk if agg_short_risk > 0 else 0
                    agg_short_risk_pct = (scen_atr * 0.5 / current_price) * 100
                    
                    response["trade_scenarios"] = {
                        "long": {
                            "entry_zone": [f"{long_entry_low:.2f}", f"{long_entry_high:.2f}"],
                            "stop_loss": long_stop,
                            "target": long_target1,
                            "target2": long_target2,
                            "r_r_ratio": f"{long_rr:.1f}:1",
                            "aggressive_entry": current_price if long_agg_valid else None,
                            "aggressive_stop": agg_long_stop,
                            "aggressive_valid": long_agg_valid,
                            "aggressive_rr": f"{agg_long_rr:.1f}:1" if long_agg_valid else None,
                            "aggressive_risk_pct": round(agg_long_risk_pct, 1) if long_agg_valid else None
                        },
                        "short": {
                            "entry_zone": [f"{short_entry_low:.2f}", f"{short_entry_high:.2f}"],
                            "stop_loss": short_stop,
                            "target": short_target1,
                            "target2": short_target2,
                            "r_r_ratio": f"{short_rr:.1f}:1",
                            "aggressive_entry": current_price if short_agg_valid else None,
                            "aggressive_stop": agg_short_stop,
                            "aggressive_valid": short_agg_valid,
                            "aggressive_rr": f"{agg_short_rr:.1f}:1" if short_agg_valid else None,
                            "aggressive_risk_pct": round(agg_short_risk_pct, 1) if short_agg_valid else None
                        },
                        "decision_point": {
                            "bull_trigger": vah + (vp_range * 0.02),
                            "bear_trigger": val - (vp_range * 0.02),
                            "current_price": current_price
                        }
                    }
        except Exception as e:
            print(f"Fib/Trade scenarios error: {e}")
        
        # Add entry_signal to response for diagram lookup
        if entry_signal:
            response["entry_signal"] = entry_signal
        
        # Add AI commentary if requested and available
        if with_ai:
            if use_rules:
                # Zero-cost rule-based analysis
                response["ai_commentary"] = get_rule_based_commentary(response, symbol.upper())
                response["analysis_source"] = "rule_based"
            elif anthropic_client:
                # Claude AI analysis (costs API credits)
                response["ai_commentary"] = get_ai_commentary(response, symbol.upper(), entry_signal)
                response["analysis_source"] = "claude"
            else:
                # Fallback to rule-based if no Anthropic client
                response["ai_commentary"] = get_rule_based_commentary(response, symbol.upper())
                response["analysis_source"] = "rule_based_fallback"
        
        # Sanitize all numpy types before returning (yfinance data has numpy types)
        return _safe_dict(response)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ analyze_live error for {symbol}: {e}")
        print(f"Traceback:\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.get("/api/analyze/live/mtf/{symbol}")
async def analyze_live_mtf(symbol: str):
    """Multi-timeframe analysis with live data"""
    try:
        scanner = get_finnhub_scanner()
        result = scanner.analyze_mtf(symbol.upper())
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Could not analyze {symbol}")
        
        # Get real-time price using get_quote (Polygon > Alpaca > Finnhub)
        current_price = None
        try:
            quote = scanner.get_quote(symbol.upper())
            if quote and quote.get('current'):
                current_price = float(quote['current'])
        except:
            pass
        
        # Convert to JSON-serializable format
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
            "current_price": current_price,  # Real-time from Polygon
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
        error_trace = traceback.format_exc()
        print(f"❌ MTF analysis error for {symbol}: {e}")
        print(f"Traceback:\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"MTF analysis error: {str(e)}")


@app.post("/api/analyze/live/mtf/{symbol}/ai")
async def analyze_mtf_with_ai(
    symbol: str, 
    trade_tf: str = Query("swing", description="Trade timeframe: intraday, swing, position, longterm, investment"),
    entry_signal: str = Query(None, description="Entry signal from scanner: e.g. 'failed_breakout:short' or 'val_touch_rejection:long'")
):
    """Generate AI trade plan using full MTF context with specific trade timeframe"""
    # Sanitize Query objects when called directly (not via HTTP)
    _trade_tf = trade_tf if isinstance(trade_tf, str) else "swing"
    _entry_signal = entry_signal if isinstance(entry_signal, str) else None
    
    if AI_KILL_SWITCH:
        # Kill switch ON — return deterministic rule-based plan
        return await _rule_based_mtf_plan(symbol, _trade_tf, _entry_signal)
    if not anthropic_client:
        raise HTTPException(status_code=400, detail="Anthropic API key not set")
    
    scanner = get_finnhub_scanner()
    
    # Check if symbol is on watchlist
    is_on_watchlist = watchlist_mgr.is_in_watchlist(symbol.upper())
    watchlist_info = watchlist_mgr.get_symbol_info(symbol.upper())
    symbol_lists = watchlist_mgr.get_symbol_lists(symbol.upper()) if is_on_watchlist else []
    
    # Get fresh MTF data
    result = scanner.analyze_mtf(symbol.upper())
    if not result:
        raise HTTPException(status_code=404, detail=f"Could not analyze {symbol}")
    
    # Get Extension Predictor data (THE EDGE)
    extension_text = ""
    try:
        if extension_available and extension_predictor:
            df_ext = scanner._get_candles(symbol.upper(), "60", days_back=10)
            if df_ext is not None and len(df_ext) >= 20:
                df_2h = df_ext.resample('2h').agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                }).dropna()
                
                if len(df_2h) >= 5:
                    poc_ext, vah_ext, val_ext = scanner.calc.calculate_volume_profile(df_ext)
                    vwap_ext = scanner.calc.calculate_vwap(df_ext)
                    
                    extension_predictor.analyze_from_dataframe(
                        symbol=symbol.upper(), df=df_2h,
                        vwap=vwap_ext, poc=poc_ext, vah=vah_ext, val=val_ext
                    )
                    
                    hottest = extension_predictor.get_hottest_setup(symbol.upper())
                    if hottest and hottest.get('candles', 0) >= 2:
                        extension_text = f"""
═══════════════════════════════════════════
⏱️ EXTENSION DURATION PREDICTOR (THE EDGE)
═══════════════════════════════════════════
Trigger Level: {hottest.get('trigger', 'NONE')}
Extended From: {hottest.get('level', 'N/A').upper() if hottest.get('level') else 'N/A'}
Duration: {hottest.get('candles', 0)} candles ({hottest.get('hours', 0)}h)
Snap-Back Probability: {hottest.get('snap_back_prob', 0)}%
Direction: {hottest.get('direction', 'N/A')}

⚠️ CRITICAL: If snap-back probability > 70%, price is OVEREXTENDED - wait for pullback before entry!
"""
    except Exception as e:
        print(f"Extension predictor error in MTF AI: {e}")
    
    # Trade timeframe settings — stop_mult/target_mult are ATR multipliers
    tf_config = {
        "intraday": {"days": 1, "label": "SAME DAY (Intraday)", "stop_mult": 0.3, "target_mult": 0.5, "hold": "1-4 hours"},
        "swing": {"days": 5, "label": "3-5 DAY SWING", "stop_mult": 0.5, "target_mult": 1.0, "hold": "3-5 days"},
        "position": {"days": 21, "label": "2-4 WEEK POSITION", "stop_mult": 1.0, "target_mult": 2.0, "hold": "2-4 weeks"},
        "longterm": {"days": 60, "label": "1-3 MONTH SETUP", "stop_mult": 2.0, "target_mult": 4.0, "hold": "1-3 months"},
        "investment": {"days": 180, "label": "6+ MONTH INVESTMENT", "stop_mult": 5.0, "target_mult": 10.0, "hold": "6+ months"}
    }
    config = tf_config.get(trade_tf, tf_config["swing"])
    
    # Calculate levels from candle data (AnalysisResult doesn't store vah/poc/val/vwap)
    df = scanner._get_candles(symbol.upper(), "60", 20)
    if df is not None and len(df) >= 5:
        poc, vah, val = scanner.calc.calculate_volume_profile(df)
        vwap = scanner.calc.calculate_vwap(df)
        rsi = scanner.calc.calculate_rsi(df)
        rvol = scanner.calc.calculate_relative_volume(df)
        volume_trend = scanner.calc.calculate_volume_trend(df)
        current_price = float(df['close'].iloc[-1])
        
        # Try to get real-time quote for more accurate price
        quote = scanner.get_quote(symbol.upper())
        if quote and quote.get('current'):
            current_price = quote['current']
    else:
        poc, vah, val, vwap, rsi = 0, 0, 0, 0, 50
        rvol, volume_trend = 1.0, "neutral"
        current_price = 0
    
    # Calculate Fibonacci retracement levels from 15-day swing high/low
    fib_text = ""
    fib_236, fib_382, fib_500, fib_618, fib_786 = 0, 0, 0, 0, 0
    swing_high, swing_low = 0, 0
    try:
        df_15d = scanner._get_candles(symbol.upper(), "D", 15)
        if df_15d is not None and len(df_15d) >= 5:
            swing_high = float(df_15d['high'].max())
            swing_low = float(df_15d['low'].min())
            fib_range = swing_high - swing_low
            
            # Fib retracement levels (from high)
            fib_236 = swing_high - (fib_range * 0.236)
            fib_382 = swing_high - (fib_range * 0.382)
            fib_500 = swing_high - (fib_range * 0.500)
            fib_618 = swing_high - (fib_range * 0.618)
            fib_786 = swing_high - (fib_range * 0.786)
            
            # Determine price position relative to Fib levels
            fib_position = ""
            if current_price >= fib_236:
                fib_position = "Above Fib 23.6% (strong/extended)"
            elif current_price >= fib_382:
                fib_position = "Between Fib 23.6%-38.2% (healthy pullback zone)"
            elif current_price >= fib_500:
                fib_position = "Between Fib 38.2%-50% (potential reversal zone)"
            elif current_price >= fib_618:
                fib_position = "Between Fib 50%-61.8% (GOLDEN ZONE - high probability reversal)"
            else:
                fib_position = "Below Fib 61.8% (deep retracement/trend change)"
            
            # Check VP + Fib confluence
            confluences = []
            if abs(vah - fib_382) / vah < 0.015:
                confluences.append(f"VAH ≈ Fib 38.2% at ${vah:.2f} (STRONG RESISTANCE)")
            if abs(poc - fib_500) / poc < 0.015:
                confluences.append(f"POC ≈ Fib 50% at ${poc:.2f} (KEY LEVEL)")
            if abs(val - fib_618) / val < 0.015:
                confluences.append(f"VAL ≈ Fib 61.8% at ${val:.2f} (GOLDEN SUPPORT)")
            if abs(poc - fib_236) / poc < 0.015:
                confluences.append(f"POC ≈ Fib 23.6% at ${poc:.2f} (CONFLUENCE)")
            
            fib_text = f"""
═══════════════════════════════════════════
📐 FIBONACCI RETRACEMENT (15-Day Swing)
═══════════════════════════════════════════
Swing High: ${swing_high:.2f} | Swing Low: ${swing_low:.2f}
Fib 23.6%: ${fib_236:.2f} | Fib 38.2%: ${fib_382:.2f} | Fib 50%: ${fib_500:.2f}
Fib 61.8%: ${fib_618:.2f} | Fib 78.6%: ${fib_786:.2f}

📍 PRICE POSITION: {fib_position}
{"🎯 VP+FIB CONFLUENCE: " + "; ".join(confluences) if confluences else ""}

⚠️ Use Fib levels for targets and stops. Golden zone (50-61.8%) = high prob reversal.
"""
    except Exception as e:
        print(f"Fib calculation error: {e}")
    
    # Calculate ATR from daily data for timeframe-scaled targets
    atr_daily = 0
    try:
        if df_15d is not None and len(df_15d) >= 5:
            _high = df_15d['high']
            _low = df_15d['low']
            _prev_close = df_15d['close'].shift(1)
            _tr = pd.concat([_high - _low, (_high - _prev_close).abs(), (_low - _prev_close).abs()], axis=1).max(axis=1)
            atr_daily = float(_tr.rolling(min(14, len(_tr))).mean().iloc[-1])
    except:
        pass
    if atr_daily <= 0:
        atr_daily = (vah - val) * 0.3 if vah > val else current_price * 0.015
    
    # ATR-scaled distances for this timeframe
    stop_distance = atr_daily * config["stop_mult"]
    target_distance = atr_daily * config["target_mult"]
    
    # Calculate trade scenarios (non-bias) for AI reference — SCALED BY TIMEFRAME
    trade_scenarios_text = ""
    if current_price > 0 and vah > 0 and val > 0 and poc > 0:
        vp_range = vah - val if vah > val else 1
        
        # LONG scenario calculations — targets scale with timeframe
        long_conservative_entry_low = val
        long_conservative_entry_high = poc
        long_conservative_stop = val - stop_distance
        
        # Target: use VP levels for short TFs, ATR-scaled for longer TFs
        if config["target_mult"] <= 0.5:          # intraday — target = VAH
            long_conservative_target = vah
        elif config["target_mult"] <= 1.0:        # swing — target = VAH or 1 ATR
            long_conservative_target = max(vah, current_price + target_distance)
        elif config["target_mult"] <= 2.0:        # position — swing high or 2 ATR
            long_conservative_target = max(swing_high if swing_high > 0 else vah, current_price + target_distance)
        else:                                      # longterm/investment — ATR-scaled
            long_conservative_target = current_price + target_distance
        
        long_cons_mid = (long_conservative_entry_low + long_conservative_entry_high) / 2
        long_cons_risk = long_cons_mid - long_conservative_stop
        long_cons_reward = long_conservative_target - long_cons_mid
        long_cons_rr = f"{long_cons_reward / long_cons_risk:.1f}:1" if long_cons_risk > 0 else "N/A"
        
        # Long aggressive (current price entry, ATR-scaled stop)
        long_agg_entry = current_price
        long_agg_stop = current_price - stop_distance
        long_agg_target = long_conservative_target
        long_agg_risk = long_agg_entry - long_agg_stop
        long_agg_reward = long_agg_target - long_agg_entry
        long_agg_rr = f"{long_agg_reward / long_agg_risk:.1f}:1" if long_agg_risk > 0 else "N/A"
        long_agg_risk_pct = (stop_distance / long_agg_entry) * 100
        
        # SHORT scenario calculations — targets scale with timeframe
        short_conservative_entry_low = poc
        short_conservative_entry_high = vah
        short_conservative_stop = vah + stop_distance
        
        if config["target_mult"] <= 0.5:          # intraday
            short_conservative_target = val
        elif config["target_mult"] <= 1.0:        # swing
            short_conservative_target = min(val, current_price - target_distance)
        elif config["target_mult"] <= 2.0:        # position
            short_conservative_target = min(swing_low if swing_low > 0 else val, current_price - target_distance)
        else:                                      # longterm/investment
            short_conservative_target = current_price - target_distance
        
        short_cons_mid = (short_conservative_entry_low + short_conservative_entry_high) / 2
        short_cons_risk = short_conservative_stop - short_cons_mid
        short_cons_reward = short_cons_mid - short_conservative_target
        short_cons_rr = f"{short_cons_reward / short_cons_risk:.1f}:1" if short_cons_risk > 0 else "N/A"
        
        # Short aggressive (current price entry, ATR-scaled stop)
        short_agg_entry = current_price
        short_agg_stop = current_price + stop_distance
        short_agg_target = short_conservative_target
        short_agg_risk = short_agg_stop - short_agg_entry
        short_agg_reward = short_agg_entry - short_agg_target
        short_agg_rr = f"{short_agg_reward / short_agg_risk:.1f}:1" if short_agg_risk > 0 else "N/A"
        short_agg_risk_pct = (stop_distance / short_agg_entry) * 100
        
        # Decision point
        bull_trigger = vah + (vp_range * 0.02)
        bear_trigger = val - (vp_range * 0.02)
        
        trade_scenarios_text = f"""
═══════════════════════════════════════════
📊 PRE-CALCULATED TRADE SCENARIOS — {config['label']}
═══════════════════════════════════════════
⏱️ HOLD: {config['hold']} | ATR(14d): ${atr_daily:.2f}
📏 Stop Distance: {config['stop_mult']}x ATR = ${stop_distance:.2f} | Target Distance: {config['target_mult']}x ATR = ${target_distance:.2f}
📍 DECISION POINT: Bull Above ${bull_trigger:.2f} | Bear Below ${bear_trigger:.2f}

🟢 LONG SETUP:
   Conservative (pullback to VAL/POC):
      Entry Zone: ${long_conservative_entry_low:.2f} - ${long_conservative_entry_high:.2f}
      Stop: ${long_conservative_stop:.2f} | Target: ${long_conservative_target:.2f} | R:R {long_cons_rr}
   ⚡ Aggressive (enter now):
      Entry: ${long_agg_entry:.2f} | Stop: ${long_agg_stop:.2f} ({long_agg_risk_pct:.1f}% risk)
      Target: ${long_agg_target:.2f} | R:R {long_agg_rr}

🔴 SHORT SETUP:
   Conservative (rally to POC/VAH):
      Entry Zone: ${short_conservative_entry_low:.2f} - ${short_conservative_entry_high:.2f}
      Stop: ${short_conservative_stop:.2f} | Target: ${short_conservative_target:.2f} | R:R {short_cons_rr}
   ⚡ Aggressive (enter now):
      Entry: ${short_agg_entry:.2f} | Stop: ${short_agg_stop:.2f} ({short_agg_risk_pct:.1f}% risk)
      Target: ${short_agg_target:.2f} | R:R {short_agg_rr}
"""
    
    # Determine leading direction
    # Priority: 1) Extension Predictor (if strong), 2) Entry scanner signal, 3) Bull/Bear differential, 4) Price position
    leading_direction = None
    leading_reason = ""
    extension_override = False
    
    # Get extension predictor info
    extension_direction = None
    extension_snap_prob = 0
    try:
        if extension_available and extension_predictor:
            hottest = extension_predictor.get_hottest_setup(symbol.upper())
            if hottest and hottest.get('candles', 0) >= 2:
                extension_direction = hottest.get('direction', '').upper()
                extension_snap_prob = hottest.get('snap_back_prob', 0)
    except:
        pass
    
    # Priority 1: Extension Predictor with high snap-back (>70%)
    # This OVERRIDES other signals - if extended with high snap-back, respect it
    if extension_direction and extension_snap_prob >= 70:
        if extension_direction == 'SHORT':
            leading_direction = "SHORT"
            leading_reason = f"Extension Predictor: {extension_snap_prob}% snap-back probability (SHORT SETUP)"
            extension_override = True
        elif extension_direction == 'LONG':
            leading_direction = "LONG"
            leading_reason = f"Extension Predictor: {extension_snap_prob}% snap-back probability (LONG SETUP)"
            extension_override = True
    
    # Priority 2: Entry scanner signal (if no extension override)
    if not leading_direction and entry_signal:
        # Entry scanner provided signal (e.g. "failed_breakout:short")
        parts = entry_signal.split(':')
        signal_type = parts[0] if len(parts) > 0 else ""
        direction = parts[1] if len(parts) > 1 else ""
        leading_direction = direction.upper() if direction else None
        leading_reason = f"VP Entry Signal: {signal_type.replace('_', ' ').title()}"
    
    # Priority 3: Strong Bull/Bear score differential (>10 points)
    if not leading_direction:
        bull_total = result.weighted_bull or 0
        bear_total = result.weighted_bear or 0
        score_diff = bull_total - bear_total
        
        if score_diff > 10:
            leading_direction = "LONG"
            leading_reason = f"Bull/Bear Score: {bull_total:.0f} vs {bear_total:.0f} (Bulls +{score_diff:.0f})"
        elif score_diff < -10:
            leading_direction = "SHORT"
            leading_reason = f"Bull/Bear Score: {bull_total:.0f} vs {bear_total:.0f} (Bears +{abs(score_diff):.0f})"
    
    # Priority 4: Price position relative to VP levels
    if not leading_direction and current_price > 0 and vah > 0 and val > 0:
        # Calculate from price position relative to VP levels
        vp_range = vah - val if vah > val else 1
        touch_buffer = vp_range * 0.02  # 2% of range
        
        if current_price >= vah - touch_buffer:
            # Price at VAH - look for rejection SHORT first
            leading_direction = "SHORT"
            leading_reason = "Price at VAH resistance zone"
        elif current_price <= val + touch_buffer:
            # Price at VAL - look for bounce LONG first
            leading_direction = "LONG"
            leading_reason = "Price at VAL support zone"
        elif current_price > poc:
            # Above POC, bias short toward mean
            leading_direction = "SHORT"
            leading_reason = f"Price above POC (${poc:.2f}), mean reversion bias"
        else:
            # Below POC, bias long toward mean
            leading_direction = "LONG" 
            leading_reason = f"Price below POC (${poc:.2f}), mean reversion bias"
    
    if not leading_direction:
        # Fallback to dominant MTF signal
        leading_direction = "LONG" if result.dominant_signal in ["LONG 🟢", "LONG ✅"] else "SHORT"
        leading_reason = f"MTF Dominant: {result.dominant_signal}"
    
    # Build timeframe summary
    tf_summary = []
    for tf, r in result.timeframe_results.items():
        tf_summary.append(f"{tf}: {r.signal} (Bull:{r.bull_score}, Bear:{r.bear_score})")
    
    # Extension Predictor override warning for AI
    extension_warning = ""
    if extension_override:
        extension_warning = f"""
⚠️ EXTENSION PREDICTOR OVERRIDE ACTIVE ⚠️
Direction MUST be {leading_direction} - snap-back probability is {extension_snap_prob}%
DO NOT recommend the opposite direction as leading trade.
"""
    
    # Bull/Bear differential warning
    bb_warning = ""
    bull_total = result.weighted_bull or 0
    bear_total = result.weighted_bear or 0
    if bear_total > bull_total + 10:
        bb_warning = f"\n🐻 BEARS DOMINATE: {bear_total:.0f} vs {bull_total:.0f} - strongly favor SHORT scenarios"
    elif bull_total > bear_total + 10:
        bb_warning = f"\n🐂 BULLS DOMINATE: {bull_total:.0f} vs {bear_total:.0f} - strongly favor LONG scenarios"
    
    # Volume context
    rvol_emoji = "🔥" if rvol >= 2.0 else "📈" if rvol >= 1.5 else "📉" if rvol < 0.7 else ""
    volume_warning = ""
    if rvol >= 2.0:
        volume_warning = f"\n🔥 HIGH VOLUME ALERT: RVOL {rvol}x - strong conviction, moves may accelerate"
    elif rvol < 0.7:
        volume_warning = f"\n⚠️ LOW VOLUME WARNING: RVOL {rvol}x - weak conviction, be cautious of false moves"
    
    # Lean user prompt - just the data
    prompt = f"""ANALYZE MTF: {symbol.upper()} @ ${current_price:.2f} | {config["label"]}

🎯 LEADING DIRECTION: {leading_direction} ({leading_reason})
{extension_warning}{bb_warning}{volume_warning}
MTF CONFLUENCE: {result.confluence_pct}% | Dominant: {result.dominant_signal}
HIGH PROB: {result.high_prob:.0f}% | LOW PROB: {result.low_prob:.0f}% → {'STRONG BULL' if result.high_prob - result.low_prob >= 25 else 'LEAN BULL' if result.high_prob - result.low_prob >= 10 else 'NEUTRAL' if abs(result.high_prob - result.low_prob) < 10 else 'LEAN BEAR' if result.low_prob - result.high_prob >= 10 else 'STRONG BEAR'} bias
Bull: {result.weighted_bull:.0f} | Bear: {result.weighted_bear:.0f}

VOLUME: {rvol_emoji} RVOL {rvol}x | Trend: {volume_trend}
TIMEFRAMES: {' | '.join(tf_summary)}

LEVELS: VAH ${vah:.2f} | POC ${poc:.2f} | VAL ${val:.2f} | VWAP ${vwap:.2f} | RSI {rsi:.0f}

NOTES: {'; '.join(result.notes[:3]) if result.notes else 'None'}
{extension_text}
{fib_text}
{trade_scenarios_text}
CRITICAL: Use the Fibonacci levels and pre-calculated scenarios above for your entries/stops/targets. Apply decision tree. Lead with {leading_direction} scenario first, then show flip scenario. Output using the exact format from instructions."""

    # Comprehensive system prompt with dual-direction output
    mtf_system_prompt = f"""You are an expert MTF trading analyst planning a {config['label']} trade.

CRITICAL: Output FULL SETUPS for BOTH directions (LONG and SHORT). This is a NON-BIAS approach.
Give each direction equal treatment with complete entry zones, stops, targets, R:R math, and reasoning.
At the end, provide a VERDICT with your preferred direction based on the data.

═══════════════════════════════════════════
DECISION TREE (Apply to EACH direction separately)
═══════════════════════════════════════════
For each direction, evaluate:
1. MTF CONFLUENCE < 60%? → Lower grade
2. HIGH vs LOW PROB within 15%? (conflicting) → Lower conviction  
3. EXTENDED > 75% snap-back? → Note caution, reduce size
4. PROBABILITY < 55%? → Grade C or lower
5. R:R < 2:1? → Note poor risk/reward
6. EV NEGATIVE? → Grade F

EXTENSION PREDICTOR RULE:
If Extension Predictor shows 70%+ snap-back probability:
- Note this in the PREFERRED verdict
- Reduce conviction for the opposite direction

BULL/BEAR SCORE RULE:
If Bear >> Bull (10+ difference), prefer SHORT scenarios
If Bull >> Bear (10+ difference), prefer LONG scenarios

═══════════════════════════════════════════
VOLUME RULES (RVOL = Relative Volume)
═══════════════════════════════════════════
RVOL >= 2.0x: HIGH conviction - breakouts/breakdowns more likely to hold
RVOL 1.5-2.0x: ABOVE average - good confirmation
RVOL 0.7-1.5x: NORMAL - standard setups
RVOL < 0.7x: LOW volume - be cautious, false moves more likely

Volume Trend:
- "increasing" on breakout = CONFIRMATION
- "decreasing" on breakout = SUSPECT (may fail)
- "increasing" into resistance = potential rejection

═══════════════════════════════════════════
PROBABILITY RULES
═══════════════════════════════════════════
MTF ALIGNMENT (Most Important):
- All TFs aligned: +15% to base
- 2 of 3 aligned: +0%
- Conflicting: -15% (likely NO TRADE)

VOLUME PROFILE LEVELS:
- First VAH/VAL test: 65%
- After 2+ tests: 50%
- Virgin levels: 75%
- POC magnet: 70%

LEVEL AGE: Today=full, Yesterday=-10%, 3+days=-20%

═══════════════════════════════════════════
FIBONACCI RULES (CRITICAL)
═══════════════════════════════════════════
Use Fib levels for entries, stops, and targets:

REVERSAL ZONES (High Probability):
- Fib 38.2%-50%: Healthy pullback zone, good for continuation entries
- Fib 50%-61.8%: GOLDEN ZONE - highest probability reversal area
- Fib 61.8% (Golden Ratio): Key support/resistance level

VP + FIB CONFLUENCE:
When VP levels (VAH/POC/VAL) align with Fib levels (<1.5%):
- This creates HIGH CONVICTION levels
- Use these for entries and stops
- Institutional orders often cluster here

TARGETS:
- LONG target: Use Fib 23.6% or swing high (MUST be ABOVE entry)
- SHORT target: Use Fib 61.8% or swing low (MUST be BELOW entry)
- T2: Use next Fib level beyond T1

STOPS:
- LONG stop: Below Fib 78.6% or swing low
- SHORT stop: Above Fib 23.6% or swing high

═══════════════════════════════════════════
⏱️ TIMEFRAME TARGET SCALING ({config['label']}) — CRITICAL
═══════════════════════════════════════════
ATR (14-day): ${atr_daily:.2f}
Expected stop distance: {config['stop_mult']}x ATR = ${stop_distance:.2f}
Expected target distance: {config['target_mult']}x ATR = ${target_distance:.2f}
Expected hold time: {config['hold']}

MINIMUM targets for this timeframe:
- T1: AT LEAST ${target_distance:.2f} from entry (={config['target_mult']}x ATR)
- T2: AT LEAST ${target_distance * 1.5:.2f} from entry
- Stop: AT LEAST ${stop_distance:.2f} from entry (={config['stop_mult']}x ATR)

DO NOT suggest targets closer than the above minimums.
A {config['label']} trade should NOT have intraday-sized targets.

═══════════════════════════════════════════
POSITION SIZING ({config['label']})
═══════════════════════════════════════════
- HIGH confidence (70%+): 1.0R
- MEDIUM confidence (55-70%): 0.75R
- If extended: max 0.5R
- LOW confidence: 0.5R or PASS
- LOW VOLUME (RVOL < 0.7): Reduce size by 50%

═══════════════════════════════════════════
OUTPUT FORMAT - DUAL DIRECTION (Both Full Setups)
═══════════════════════════════════════════

🟢 SCENARIO 1: LONG SETUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐ GRADE: [A+ / A / B / C / F]
🎯 CONVICTION: X/10
📈 PROBABILITY: X-Y% [High/Med/Low]

📍 ENTRY ZONE: $XX.XX - $XX.XX (near Fib X% / VP level)
📍 ENTRY (midpoint): $XX.XX
🛑 STOP: $XX.XX (below Fib X% / VP level)
💰 T1: $XX.XX (at Fib X% / VAH) | 🚀 T2: $XX.XX

📐 R:R MATH: 
   Risk = $X.XX | T1 Reward = $X.XX → R:R = X.X:1
💹 EV: $X.XX per $100 risked → [POSITIVE/NEGATIVE]

📊 SIZE: X.XXR | ⏱️ HOLD: {config['hold']}
✅ TRIGGER: [What confirms this setup - e.g. "Break above VAH with volume"]
❌ INVALID IF: [What kills this setup - e.g. "Breaks below VAL"]

💡 WHY LONG: [1-2 sentences on bull case with VP/Fib reference]

🔴 SCENARIO 2: SHORT SETUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐ GRADE: [A+ / A / B / C / F]
🎯 CONVICTION: X/10
📈 PROBABILITY: X-Y% [High/Med/Low]

📍 ENTRY ZONE: $XX.XX - $XX.XX (near Fib X% / VP level)
📍 ENTRY (midpoint): $XX.XX
🛑 STOP: $XX.XX (above Fib X% / VP level)
💰 T1: $XX.XX (at Fib X% / VAL) | 🚀 T2: $XX.XX

📐 R:R MATH: 
   Risk = $X.XX | T1 Reward = $X.XX → R:R = X.X:1
💹 EV: $X.XX per $100 risked → [POSITIVE/NEGATIVE]

📊 SIZE: X.XXR | ⏱️ HOLD: {config['hold']}
✅ TRIGGER: [What confirms this setup - e.g. "Rejection at VAH with volume"]
❌ INVALID IF: [What kills this setup - e.g. "Breaks above Fib 23.6%"]

💡 WHY SHORT: [1-2 sentences on bear case with VP/Fib reference]

⚖️ VERDICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏆 PREFERRED: [LONG or SHORT] because [1 sentence reason]
⚠️ KEY LEVEL: $XX.XX - Above = Long bias, Below = Short bias
"""

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            system=mtf_system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        ai_text = response.content[0].text.strip()
        
        # Persist AI suggestion to Firestore
        _save_ai_suggestion_bg(symbol.upper(), "mtf_plan", ai_text, {
            "model": "claude-sonnet-4-20250514",
            "trade_timeframe": config["label"],
            "leading_direction": leading_direction,
            "leading_reason": leading_reason,
            "confluence": result.confluence_pct,
            "bull_score": float(result.weighted_bull or 0),
            "bear_score": float(result.weighted_bear or 0),
            "price": current_price,
            "vah": vah, "poc": poc, "val": val
        })
        
        return {
            "symbol": symbol.upper(),
            "ai_commentary": ai_text,
            "high_prob": result.high_prob,
            "low_prob": result.low_prob,
            "confluence": result.confluence_pct,
            "dominant_signal": result.dominant_signal,
            "trade_timeframe": config["label"],
            "leading_direction": leading_direction,
            "leading_reason": leading_reason,
            "extension_override": extension_override,
            "extension_snap_prob": extension_snap_prob if extension_override else None,
            "bull_score": result.weighted_bull,
            "bear_score": result.weighted_bear,
            "rvol": rvol,
            "volume_trend": volume_trend,
            "vah": vah,
            "poc": poc,
            "val": val,
            "vwap": vwap,
            "rsi": rsi,
            "current_price": current_price
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")


@app.get("/api/scan/live")
async def scan_live(
    symbols: str = Query(..., description="Comma-separated symbols"),
    timeframe: str = Query("1HR", description="Timeframe for analysis")
):
    """Scan multiple symbols with live data"""
    scanner = get_finnhub_scanner()
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    results = []
    for symbol in symbol_list[:20]:  # Limit to 20 for rate limiting
        result = scanner.analyze(symbol, timeframe)
        if result:
            results.append({
                "symbol": symbol,
                "signal": result.signal,
                "signal_emoji": result.signal_emoji,
                "bull_score": result.bull_score,
                "bear_score": result.bear_score,
                "confidence": result.confidence,
                "position": result.position,
                "signal_type": str(getattr(result, 'signal_type', 'none'))
            })
        time.sleep(0.5)  # Rate limiting
    
    # Sort by actionability
    signal_order = {"LONG_SETUP": 0, "SHORT_SETUP": 1, "YELLOW": 2}
    results.sort(key=lambda x: (signal_order.get(x["signal"], 3), -x["confidence"]))
    
    return {
        "count": len(results),
        "timeframe": timeframe,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


# -----------------------------------------------------------------------------
# WATCHLISTS
# -----------------------------------------------------------------------------

@app.get("/api/watchlist")
@app.get("/api/watchlists")
async def get_watchlists():
    """Get all watchlists"""
    lists = watchlist_mgr.get_all_watchlists()
    return {
        "count": len(lists),
        "watchlists": [
            {
                "name": lst.name,
                "description": lst.description,
                "symbol_count": len(lst.symbols),
                "enabled_count": sum(1 for s in lst.symbols if getattr(s, "enabled", True)),
                "symbols": _watchlist_symbols(lst)
            }
            for lst in lists
        ]
    }


@app.get("/api/watchlists/{name}")
async def get_watchlist(name: str):
    """Get specific watchlist"""
    lst = watchlist_mgr.get_watchlist(name)
    if not lst:
        raise HTTPException(status_code=404, detail=f"Watchlist '{name}' not found")
    
    return {
        "name": lst.name,
        "description": lst.description,
        "symbol_count": len(lst.symbols),
        "enabled_count": sum(1 for s in lst.symbols if getattr(s, "enabled", True)),
        "symbols": _watchlist_symbols(lst)
    }


@app.get("/api/watchlists/{name}/scan")
async def scan_watchlist(
    name: str,
    timeframe: str = Query("1HR", description="Timeframe for analysis"),
    limit: int = Query(50, ge=1, le=200, description="Max symbols to scan")
):
    """Scan entire watchlist - optimized with parallel processing"""
    import asyncio
    import concurrent.futures
    
    lst = watchlist_mgr.get_watchlist(name)
    if not lst:
        raise HTTPException(status_code=404, detail=f"Watchlist '{name}' not found")
    
    scanner = get_finnhub_scanner()

    # Scan enabled symbols only
    symbols = watchlist_mgr.get_enabled_symbols(watchlist_name=name)
    symbols = symbols[:limit]
    
    def analyze_symbol(symbol):
        """Analyze a single symbol (runs in thread pool)"""
        try:
            result = scanner.analyze(symbol, timeframe)
            if result:
                return {
                    "symbol": symbol,
                    "signal": result.signal,
                    "signal_emoji": result.signal_emoji,
                    "bull_score": result.bull_score,
                    "bear_score": result.bear_score,
                    "confidence": result.confidence,
                    "position": result.position,
                    "rsi_zone": result.rsi_zone
                }
        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
        return None
    
    # Run analysis in parallel using thread pool
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(analyze_symbol, s): s for s in symbols}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    
    # Sort
    signal_order = {"LONG_SETUP": 0, "SHORT_SETUP": 1, "YELLOW": 2}
    results.sort(key=lambda x: (signal_order.get(x["signal"], 3), -x["confidence"]))
    
    return {
        "watchlist": name,
        "timeframe": timeframe,
        "count": len(results),
        "requested": len(symbols),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


# -----------------------------------------------------------------------------
# WORKFLOW (Trading workflow state management)
# -----------------------------------------------------------------------------

@app.get("/api/workflow/status")
async def get_workflow_status():
    """Get current workflow/mental state status"""
    # This is a simple stub - in production would track daily trades and mental state
    return {
        "trades": 0,
        "max_trades": 4,
        "total_r": 0.0,
        "consecutive_losses": 0,
        "mental_state": "GREEN",
        "mental_state_description": "Ready to trade - all systems go"
    }


@app.get("/api/workflow/pre-trade-check")
async def run_pre_trade_check(signal: str = "LONG_SETUP", confidence: int = 50):
    """Run pre-trade gate checks"""
    results = [
        {"gate": "Mental State", "passed": True, "message": "Mental state is GREEN - clear to trade"},
        {"gate": "Daily Limit", "passed": True, "message": "0/4 trades taken today"},
        {"gate": "Consecutive Losses", "passed": True, "message": "0 consecutive losses"},
        {"gate": "Signal Quality", "passed": confidence >= 60, "message": f"Signal confidence: {confidence}%"},
        {"gate": "Market Conditions", "passed": True, "message": "Market conditions acceptable"}
    ]
    
    all_passed = all(r["passed"] for r in results)
    
    return {
        "can_trade": all_passed,
        "results": results,
        "signal": signal,
        "confidence": confidence
    }


# -----------------------------------------------------------------------------
# ALERTS — Firestore-first (always persists)
# -----------------------------------------------------------------------------

@app.get("/api/alerts")
async def get_alerts(symbol: str = None, user_id: str = None):
    """Get active alerts - always uses Firestore when available"""
    uid = user_id or DEFAULT_TRADE_USER
    if firestore_available:
        fs = get_firestore()
        if fs.is_available():
            alerts = fs.get_alerts(uid, symbol)
            return {
                "count": len(alerts),
                "alerts": alerts,
                "storage": "firestore"
            }
    
    # Fallback to local storage
    alerts = chart_system.get_alerts(symbol)
    return {
        "count": len(alerts),
        "alerts": [asdict(a) for a in alerts],
        "storage": "local"
    }


@app.post("/api/alerts")
async def create_alert(request: AlertRequest, user_id: str = None):
    """Create new alert - always uses Firestore when available"""
    uid = user_id or DEFAULT_TRADE_USER
    if firestore_available:
        fs = get_firestore()
        if fs.is_available():
            alert = UserAlert(
                symbol=request.symbol,
                level=request.level,
                direction=request.direction,
                action=request.action,
                note=request.note or ""
            )
            result = fs.add_alert(uid, alert)
            if result:
                return {"status": "created", "alert": result, "storage": "firestore"}
    
    # Fallback to local storage
    alert = chart_system.add_alert(
        symbol=request.symbol,
        level=request.level,
        direction=request.direction,
        action=request.action,
        note=request.note
    )
    return {"status": "created", "alert": asdict(alert), "storage": "local"}


@app.delete("/api/alerts")
async def delete_alert(symbol: str, level: float, user_id: str = None):
    """Delete an alert"""
    uid = user_id or DEFAULT_TRADE_USER
    if firestore_available:
        fs = get_firestore()
        if fs.is_available():
            success = fs.delete_alert(uid, symbol, level)
            if success:
                return {"status": "deleted", "storage": "firestore"}
            raise HTTPException(status_code=404, detail="Alert not found")
    
    # Fallback to local
    success = chart_system.alerts.remove_alert(symbol, level)
    if success:
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Alert not found")


@app.post("/api/alerts/check")
async def check_alerts(symbol: str, price: float):
    """Check if any alerts triggered"""
    triggered = chart_system.alerts.check_alerts(symbol, price)
    return {
        "triggered": len(triggered),
        "alerts": [asdict(a) for a in triggered]
    }


@app.post("/api/alerts/create")
async def create_ai_alerts(request: AIAlertsRequest, user_id: str = None):
    """Create multiple alerts from AI trade plan (entry, stop, targets)"""
    created = []
    symbol = request.symbol.upper()
    is_long = request.direction.upper() == "LONG"
    uid = user_id or DEFAULT_TRADE_USER
    
    # Helper function to add alert
    def add_alert_helper(level, direction, action, note):
        if firestore_available:
            fs = get_firestore()
            if fs.is_available():
                alert = UserAlert(
                    symbol=symbol,
                    level=level,
                    direction=direction,
                    action=action,
                    note=note
                )
                result = fs.add_alert(uid, alert)
                return result.get('id') if result else None
        
        # Fallback to local
        local_alert = chart_system.add_alert(
            symbol=symbol,
            level=level,
            direction=direction,
            action=action,
            note=note
        )
        return id(local_alert)
    
    # Entry alert - price approaching entry zone
    if request.entry > 0:
        alert_id = add_alert_helper(
            request.entry,
            "below" if is_long else "above",
            "LONG" if is_long else "SHORT",
            f"📍 {request.trade_timeframe} Entry Zone"
        )
        created.append({"type": "entry", "level": request.entry, "id": alert_id})
    
    # Stop loss alert
    if request.stop > 0:
        alert_id = add_alert_helper(
            request.stop,
            "below" if is_long else "above",
            "EXIT",
            f"🛑 {request.trade_timeframe} Stop Loss"
        )
        created.append({"type": "stop", "level": request.stop, "id": alert_id})
    
    # Target 1 alert
    if request.target1 > 0:
        alert_id = add_alert_helper(
            request.target1,
            "above" if is_long else "below",
            "ALERT",
            f"🎯 {request.trade_timeframe} Target 1"
        )
        created.append({"type": "target1", "level": request.target1, "id": alert_id})
    
    # Target 2 alert
    if request.target2 > 0:
        alert_id = add_alert_helper(
            request.target2,
            "above" if is_long else "below",
            "ALERT",
            f"🎯 {request.trade_timeframe} Target 2"
        )
        created.append({"type": "target2", "level": request.target2, "id": alert_id})
    
    return {
        "status": "created",
        "symbol": symbol,
        "direction": request.direction,
        "alerts_created": len(created),
        "alerts": created,
        "storage": "firestore" if firestore_available else "local"
    }


# -----------------------------------------------------------------------------
# TRADE TRACKER — Firestore-first (always persists)
# -----------------------------------------------------------------------------

DEFAULT_TRADE_USER = "system"  # fallback user_id when none provided

@app.get("/api/trades")
async def get_trades(symbol: str = None, status: str = None, user_id: str = None):
    """Get trades - always uses Firestore when available"""
    uid = user_id or DEFAULT_TRADE_USER
    if firestore_available:
        fs = get_firestore()
        if fs.is_available():
            trades = fs.get_trades(uid, symbol, status)
            return {
                "count": len(trades),
                "trades": trades,
                "storage": "firestore",
                "user_id": uid
            }
    
    # Fallback to local only if Firestore unavailable
    if status == "pending":
        trades = chart_system.get_pending_trades(symbol)
    else:
        trades = chart_system.tracker.trades
        if symbol:
            trades = [t for t in trades if t.symbol == symbol.upper()]
    
    return {
        "count": len(trades),
        "trades": [asdict(t) for t in trades],
        "storage": "local"
    }


@app.post("/api/trades")
async def log_trade(request: TradeRequest, user_id: str = None):
    """Log a new trade setup - always uses Firestore when available"""
    uid = user_id or DEFAULT_TRADE_USER
    if firestore_available:
        fs = get_firestore()
        if fs.is_available():
            trade = UserTrade(
                symbol=request.symbol,
                timeframe=request.timeframe,
                direction=request.direction,
                entry=request.entry,
                stop=request.stop,
                target=request.target,
                target2=request.target2 or 0,
                signal=request.signal or "YELLOW",
                confidence=request.confidence or 50,
                notes=request.notes or ""
            )
            result = fs.add_trade(uid, trade)
            if result:
                return {"status": "logged", "trade": result, "storage": "firestore", "user_id": uid}
    
    # Fallback to local only if Firestore unavailable
    trade = chart_system.log_trade(
        symbol=request.symbol,
        timeframe=request.timeframe,
        direction=request.direction,
        entry=request.entry,
        stop=request.stop,
        target=request.target,
        target2=request.target2,
        signal=request.signal,
        confidence=request.confidence,
        notes=request.notes
    )
    return {"status": "logged", "trade": asdict(trade), "storage": "local"}


@app.put("/api/trades")
async def update_trade(request: TradeUpdateRequest, user_id: str = None):
    """Update trade status - always uses Firestore when available"""
    uid = user_id or DEFAULT_TRADE_USER
    if firestore_available:
        fs = get_firestore()
        if fs.is_available():
            # For Firestore, we need trade_id instead of symbol
            if hasattr(request, 'trade_id') and request.trade_id:
                result = fs.close_trade(uid, request.trade_id, request.exit_price or 0, request.status)
                if result:
                    return {"status": "updated", "trade": result, "storage": "firestore"}
    
    # Fallback to local only if Firestore unavailable
    trade = chart_system.update_trade(
        request.symbol,
        request.status,
        request.exit_price
    )
    if trade:
        return {"status": "updated", "trade": asdict(trade), "storage": "local"}
    raise HTTPException(status_code=404, detail="Trade not found")


@app.get("/api/trades/stats")
async def get_trade_stats(user_id: str = None):
    """Get trading statistics - always uses Firestore when available"""
    uid = user_id or DEFAULT_TRADE_USER
    if firestore_available:
        fs = get_firestore()
        if fs.is_available():
            return fs.get_trade_stats(uid)
    
    return chart_system.get_trade_stats()


# =============================================================================
# JOURNAL ANALYTICS — Rich Performance Analytics
# =============================================================================

# =============================================================================
# TRADE MONITOR — Auto-Close Engine API
# =============================================================================

@app.get("/api/monitor/status")
async def get_monitor_status():
    """Get trade monitor status and stats"""
    if not trade_monitor_available or not trade_monitor:
        return {"running": False, "error": "Trade monitor not available"}
    return trade_monitor.get_status()


@app.get("/api/monitor/trades")
async def get_monitored_trades():
    """Get all trades currently being watched"""
    if not trade_monitor_available or not trade_monitor:
        return []
    return trade_monitor.get_monitored_trades()


@app.get("/api/monitor/events")
async def get_monitor_events(limit: int = 50):
    """Get recent auto-close events"""
    if not trade_monitor_available or not trade_monitor:
        return []
    return trade_monitor.get_events(limit=limit)


@app.post("/api/monitor/start")
async def start_monitor():
    """Manually start the trade monitor"""
    if not trade_monitor_available or not trade_monitor:
        raise HTTPException(status_code=503, detail="Trade monitor not available")
    trade_monitor.start_background()
    return {"status": "started", "interval": trade_monitor.interval}


@app.post("/api/monitor/stop")
async def stop_monitor():
    """Stop the trade monitor"""
    if not trade_monitor_available or not trade_monitor:
        raise HTTPException(status_code=503, detail="Trade monitor not available")
    trade_monitor.stop()
    return {"status": "stopped"}


@app.post("/api/monitor/config")
async def configure_monitor(
    interval: int = None,
    trailing_stop: bool = None,
    trailing_stop_pct: float = None
):
    """Update monitor configuration"""
    if not trade_monitor_available or not trade_monitor:
        raise HTTPException(status_code=503, detail="Trade monitor not available")
    if interval is not None:
        trade_monitor.interval = max(10, interval)  # min 10s
    if trailing_stop is not None:
        trade_monitor.trailing_stop_enabled = trailing_stop
    if trailing_stop_pct is not None:
        trade_monitor.trailing_stop_pct = max(0.005, min(0.10, trailing_stop_pct))
    return trade_monitor.get_status()


@app.post("/api/monitor/test")
async def test_monitor_cycle():
    """Force one monitoring cycle immediately (works outside market hours for testing)"""
    if not trade_monitor_available or not trade_monitor:
        raise HTTPException(status_code=503, detail="Trade monitor not available")
    # Run cycle directly instead of waiting for loop
    try:
        await trade_monitor._run_cycle()
    except Exception as e:
        return {"status": "error", "error": str(e)}
    return {
        "status": "cycle_complete",
        "monitor": trade_monitor.get_status(),
        "monitored_trades": trade_monitor.get_monitored_trades(),
        "recent_events": trade_monitor.get_events(limit=10)
    }


@app.get("/api/trades/analytics")
async def get_trade_analytics(user_id: str = None, days: int = 90):
    """
    Comprehensive trading analytics: win rate by symbol/direction/timeframe/signal,
    equity curve, monthly/weekly breakdown, streaks, drawdown, profit factor, 
    holding times, day-of-week patterns, best/worst trades.
    """
    try:
        from journal_analytics import compute_journal_analytics

        trades = []
        storage = "local"

        # Pull from Firestore if user_id provided
        if user_id and firestore_available:
            fs = get_firestore()
            if fs.is_available():
                trades = fs.get_trades(user_id)
                storage = "firestore"

        # Fallback to local chart system
        if not trades:
            local_trades = chart_system.tracker.trades
            trades = [asdict(t) for t in local_trades]
            storage = "local"

        analytics = compute_journal_analytics(trades, days=days)
        analytics["storage"] = storage
        return analytics

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trades/analytics/report")
async def get_trade_analytics_report(user_id: str = None, days: int = 90):
    """Get a text-format trading performance report."""
    try:
        from journal_analytics import compute_journal_analytics, generate_analytics_report

        trades = []
        if user_id and firestore_available:
            fs = get_firestore()
            if fs.is_available():
                trades = fs.get_trades(user_id)
        if not trades:
            trades = [asdict(t) for t in chart_system.tracker.trades]

        analytics = compute_journal_analytics(trades, days=days)
        report = generate_analytics_report(analytics)
        return {"report": report, "analytics": analytics}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# BACKTEST ENGINE API
# =============================================================================

@app.post("/api/backtest/strategy")
async def run_strategy_backtest(request: Request):
    """
    Run a strategy backtest over historical data.
    Body: {symbols: ["AAPL","NVDA"], days_back: 90, signal_filter: "GREEN", min_confidence: 50, timeframe: "swing"}
    """
    try:
        from backtest_engine import BacktestEngine
        body = await request.json()

        symbols = body.get("symbols", [])
        if not symbols:
            raise HTTPException(status_code=400, detail="symbols required")

        engine = BacktestEngine()
        result = engine.run_strategy(
            symbols=symbols,
            days_back=body.get("days_back", 90),
            signal_filter=body.get("signal_filter"),
            min_confidence=body.get("min_confidence", 0),
            timeframe=body.get("timeframe", "swing"),
            scan_interval_days=body.get("scan_interval", 5),
            max_hold_bars=body.get("max_hold_bars", 60),
        )
        return result.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest/replay")
async def run_replay_backtest(request: Request, user_id: str = None):
    """
    Replay journal trades against real price data to validate outcomes.
    Body: {bar_interval: "1d", max_hold_bars: 60} or pass trades in body.
    """
    try:
        from backtest_engine import BacktestEngine
        body = await request.json() if await request.body() else {}

        trades = body.get("trades", [])

        # If no trades provided, pull from Firestore or local
        if not trades:
            if user_id and firestore_available:
                fs = get_firestore()
                if fs.is_available():
                    trades = fs.get_trades(user_id)
            if not trades:
                local = chart_system.tracker.trades
                trades = [asdict(t) for t in local]

        if not trades:
            return {"error": "No trades found", "trades": [], "summary": {}}

        engine = BacktestEngine()
        result = engine.replay_trades(
            trades=trades,
            bar_interval=body.get("bar_interval", "1d"),
            max_hold_bars=body.get("max_hold_bars", 60),
        )
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/quick")
async def quick_backtest_endpoint(
    symbols: str = "AAPL,NVDA,TSLA,META,MSFT",
    days: int = 90,
    signal: str = None,
    confidence: int = 50,
):
    """
    Quick backtest via GET. Symbols comma-separated.
    Example: /api/backtest/quick?symbols=AAPL,NVDA&days=90&signal=GREEN
    """
    try:
        from backtest_engine import quick_backtest
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        return quick_backtest(
            symbols=sym_list,
            days_back=days,
            signal_filter=signal if signal else None,
            min_confidence=confidence,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest")
async def serve_backtest_page():
    """Serve the backtest dashboard"""
    return FileResponse("public/backtest.html")


@app.post("/api/backtest/custom")
async def run_custom_backtest(request: Request):
    """
    Run a custom-rule backtest over historical data.
    Body: {
        symbols: ["AAPL","NVDA"],
        days_back: 90,
        direction: "LONG",
        rr_ratio: 2.0,
        stop_atr_mult: 1.5,
        max_hold_bars: 60,
        rules: [
            {"type": "move_off_open", "min_pct": 0.75, "max_pct": 1.25},
            {"type": "above_ma", "period": 20}
        ]
    }
    Rule types: move_off_open, rsi_range, above_ma, below_ma, rvol_min,
                gap_up, gap_down, range_pct
    """
    try:
        from backtest_engine import BacktestEngine
        body = await request.json()

        symbols = body.get("symbols", [])
        if not symbols:
            raise HTTPException(status_code=400, detail="symbols required")

        rules = body.get("rules", [])
        if not rules:
            raise HTTPException(status_code=400, detail="rules required (at least one)")

        engine = BacktestEngine()
        result = engine.run_custom(
            symbols=symbols,
            days_back=body.get("days_back", 90),
            rules=rules,
            direction=body.get("direction", "LONG"),
            rr_ratio=body.get("rr_ratio", 2.0),
            stop_atr_mult=body.get("stop_atr_mult", 1.5),
            max_hold_bars=body.get("max_hold_bars", 60),
        )
        return result.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest/stats")
async def run_stats_scan(request: Request):
    """
    Pure probability / frequency scanner — no stops, no targets, no cooldown.
    Counts every day that matches the rules and returns per-day breakdown +
    aggregate stats (frequency%, green/red close%, next-day follow-through).

    Body: {
        symbols: ["IWM"],
        days_back: 30,
        direction: "LONG",
        rules: [{"type": "high_off_open", "min_pct": 0.20, "max_pct": 1.25}]
    }
    """
    try:
        from backtest_engine import BacktestEngine
        body = await request.json()

        symbols = body.get("symbols", [])
        if not symbols:
            raise HTTPException(status_code=400, detail="symbols required")

        rules = body.get("rules", [])
        if not rules:
            raise HTTPException(status_code=400, detail="rules required (at least one)")

        engine = BacktestEngine()
        result = engine.run_stats(
            symbols=symbols,
            days_back=body.get("days_back", 90),
            rules=rules,
            direction=body.get("direction", "LONG"),
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest/stats/insight")
async def stats_ai_insight(request: Request):
    """
    Generate AI commentary on stats scan results.
    Body: { summary: {...}, days: [...], meta: {...} }
    """
    try:
        import anthropic as anth
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(status_code=503, detail="AI not configured (no ANTHROPIC_API_KEY)")

        body = await request.json()
        summary = body.get("summary", {})
        days = body.get("days", [])[:20]  # limit context size
        meta = body.get("meta", {})

        symbols = ", ".join(meta.get("symbols", []))
        rules_desc = ", ".join(r.get("type", "").replace("_", " ") for r in meta.get("rules", []))
        direction = meta.get("direction", "LONG")

        # Build day table for AI
        day_lines = []
        for d in days:
            nd = f", next day: {d.get('next_day_pct', 'N/A')}%" if "next_day_pct" in d else ""
            day_lines.append(
                f"  {d.get('date')}: open ${d.get('open')}, high ${d.get('high')}, low ${d.get('low')}, "
                f"close ${d.get('close')}, high_off_open +{d.get('high_off_open_pct')}%, "
                f"low_off_open {d.get('low_off_open_pct')}%, close_vs_open {d.get('close_vs_open_pct')}%, "
                f"RSI {d.get('rsi')}, RVOL {d.get('rvol')}x{nd}"
            )
        day_table = "\n".join(day_lines)

        prompt = f"""Analyze this intraday stats scan for a trader:

SYMBOL(S): {symbols}
DIRECTION: {direction}
FILTER RULES: {rules_desc}
LOOKBACK: {meta.get('days_back', '?')} days

SUMMARY:
- Qualifying days: {summary.get('qualifying_days')} / {summary.get('total_days_scanned')} ({summary.get('frequency_pct')}%)
- Closed green: {summary.get('closed_green')} ({summary.get('green_pct')}%)
- Closed red: {summary.get('closed_red')}
- Avg high off open: +{summary.get('avg_high_off_open')}%
- Avg low off open: {summary.get('avg_low_off_open')}%
- Avg close vs open: {summary.get('avg_close_vs_open')}%
- Avg day range: {summary.get('avg_range_pct')}%
- Next-day green: {summary.get('next_day_green_pct')}%
- Avg next-day move: {summary.get('avg_next_day_pct')}%

QUALIFYING DAYS:
{day_table}

Provide a concise trading insight (3-5 bullet points) covering:
1. The probability edge (how often this pattern occurs and which direction favors)
2. Best days vs worst days — any clustering or pattern
3. Follow-through analysis — does the pattern predict next-day behavior?
4. Actionable takeaway — what a trader should do with this data
5. Risk warning — when this pattern fails or reverses

Keep it practical, no fluff. Use specific numbers from the data."""

        client = anth.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        insight = response.content[0].text.strip()

        return {"insight": insight, "model": "claude-sonnet-4-20250514"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# NOTIFICATIONS — Firebase Cloud Messaging
# =============================================================================

@app.post("/api/notifications/register")
async def register_fcm_token(request: Request):
    """Register an FCM token for push notifications"""
    if not notification_available:
        raise HTTPException(status_code=503, detail="Notification service not available")
    body = await request.json()
    token = body.get("token", "")
    device_type = body.get("device_type", "web")
    user_id = body.get("user_id", "anonymous")
    if not token:
        raise HTTPException(status_code=400, detail="token required")
    ok = notification_service.register_token(user_id, token, device_type)
    return {"registered": ok, "device_type": device_type}


@app.post("/api/notifications/unregister")
async def unregister_fcm_token(request: Request):
    """Remove an FCM token"""
    if not notification_available:
        raise HTTPException(status_code=503, detail="Notification service not available")
    body = await request.json()
    token = body.get("token", "")
    user_id = body.get("user_id", "anonymous")
    ok = notification_service.unregister_token(user_id, token)
    return {"removed": ok}


@app.get("/api/notifications/settings")
async def get_notification_settings(user_id: str = "anonymous"):
    """Get user notification preferences"""
    if not notification_available:
        raise HTTPException(status_code=503, detail="Notification service not available")
    return notification_service.get_prefs(user_id)


@app.post("/api/notifications/settings")
async def update_notification_settings(request: Request):
    """Update user notification preferences"""
    if not notification_available:
        raise HTTPException(status_code=503, detail="Notification service not available")
    body = await request.json()
    user_id = body.pop("user_id", "anonymous")
    prefs = notification_service.update_prefs(user_id, body)
    return {"updated": True, "prefs": prefs}


@app.post("/api/notifications/test")
async def send_test_notification(request: Request):
    """Send a test push notification to a user"""
    if not notification_available:
        raise HTTPException(status_code=503, detail="Notification service not available")
    body = await request.json()
    user_id = body.get("user_id", "anonymous")
    result = notification_service.send_to_user(
        user_id=user_id,
        title="🔔 Analysis Grid Test",
        body="Push notifications are working!",
        data={"type": "test"},
        category="test"
    )
    return result


@app.get("/api/notifications/status")
async def notification_status():
    """Get notification service status"""
    import os
    if not notification_available:
        return {"active": False, "reason": "service not loaded"}
    status = notification_service.get_status()
    status["env_debug"] = {
        "FIREBASE_SERVICE_ACCOUNT": "set" if os.getenv("FIREBASE_SERVICE_ACCOUNT") else "empty",
        "GOOGLE_CREDENTIALS_JSON": "set" if os.getenv("GOOGLE_CREDENTIALS_JSON") else "empty",
        "GOOGLE_APPLICATION_CREDENTIALS": "set" if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") else "empty",
    }
    return status


# =============================================================================
# RESEARCH BUILDER API
# =============================================================================

@app.post("/api/research/build")
async def research_build(request: Request):
    """Build a Picks & Shovels research report from config"""
    import time as _time
    import requests
    from datetime import datetime as _dt, timedelta as _td
    from pathlib import Path as _Path

    body = await request.json()
    cfg = body.get("config", {})
    mode = body.get("mode", "full")

    if not cfg.get("title"):
        raise HTTPException(status_code=400, detail="Config must have a title")

    POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "")
    FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY", "")

    # Get all investable tickers (layer2 + layer3)
    tickers = []
    for item in cfg.get("layer2", []) + cfg.get("layer3", []):
        t = item.get("ticker", "").strip().upper()
        if t and t not in tickers:
            tickers.append(t)

    fundamentals = {}
    performance = {}
    fund_log = []
    perf_log = []

    if mode == "full":
        # ── Pull Finnhub fundamentals ──
        if FINNHUB_KEY:
            for t in tickers:
                try:
                    data = {}
                    r = requests.get("https://finnhub.io/api/v1/stock/metric",
                                     params={"symbol": t, "metric": "all", "token": FINNHUB_KEY}, timeout=10)
                    if r.ok:
                        m = r.json().get("metric", {})
                        data["gross_margin"] = m.get("grossMarginTTM") or m.get("grossMarginAnnual")
                        data["debt_equity"] = m.get("totalDebt/totalEquityQuarterly") or m.get("totalDebt/totalEquityAnnual")
                        data["rev_growth"] = m.get("revenueGrowthTTMYoy") or m.get("revenueGrowth3Y")
                    _time.sleep(0.12)

                    r2 = requests.get("https://finnhub.io/api/v1/stock/recommendation",
                                      params={"symbol": t, "token": FINNHUB_KEY}, timeout=10)
                    if r2.ok and r2.json():
                        rec = r2.json()[0]
                        data["analyst_buy"] = rec.get("buy", 0) + rec.get("strongBuy", 0)
                        data["analyst_sell"] = rec.get("sell", 0) + rec.get("strongSell", 0)
                    _time.sleep(0.12)

                    fundamentals[t] = data
                    fund_log.append(f"✓ {t}: margin={data.get('gross_margin','—')}%")
                except Exception as e:
                    fund_log.append(f"✗ {t}: {str(e)}")
                    fundamentals[t] = {}

        # ── Pull Polygon performance ──
        if POLYGON_KEY:
            today = _dt.now()
            for t in tickers:
                try:
                    perf = {}
                    r = requests.get(f"https://api.polygon.io/v2/aggs/ticker/{t}/prev",
                                     params={"apiKey": POLYGON_KEY}, timeout=10)
                    if r.ok and r.json().get("results"):
                        current = r.json()["results"][0]["c"]
                        perf["price"] = current
                        _time.sleep(0.12)

                        for label, days in [("1Y", 365), ("2Y", 730)]:
                            past = (today - _td(days=days)).strftime("%Y-%m-%d")
                            r2 = requests.get(f"https://api.polygon.io/v2/aggs/ticker/{t}/range/1/day/{past}/{past}",
                                              params={"apiKey": POLYGON_KEY}, timeout=10)
                            if r2.ok and r2.json().get("results"):
                                old = r2.json()["results"][0]["c"]
                                perf[label] = round(((current - old) / old) * 100, 1)
                            else:
                                for off in [1, 2, 3, -1, -2]:
                                    alt = (today - _td(days=days + off)).strftime("%Y-%m-%d")
                                    r3 = requests.get(f"https://api.polygon.io/v2/aggs/ticker/{t}/range/1/day/{alt}/{alt}",
                                                      params={"apiKey": POLYGON_KEY}, timeout=10)
                                    if r3.ok and r3.json().get("results"):
                                        old = r3.json()["results"][0]["c"]
                                        perf[label] = round(((current - old) / old) * 100, 1)
                                        break
                                    _time.sleep(0.1)
                            _time.sleep(0.12)

                    performance[t] = perf
                    perf_log.append(f"✓ {t}: ${perf.get('price', '—')}")
                except Exception as e:
                    perf_log.append(f"✗ {t}: {str(e)}")
                    performance[t] = {}

    # ── Fetch company profiles for trajectory analysis ──
    profiles = {}
    profile_log = []
    try:
        from picks_shovels_builder import fetch_company_profiles, _profiles_from_config
        if FINNHUB_KEY:
            profiles = fetch_company_profiles(tickers, cfg)
            profile_log.append(f"Fetched profiles for {len(profiles)} tickers")
        else:
            profiles = _profiles_from_config(tickers, cfg)
            profile_log.append("Using config-embedded founding years (no Finnhub key)")
    except Exception as e:
        profile_log.append(f"Profile fetch error: {e}")

    # ── Generate HTML using picks_shovels_builder ──
    try:
        from picks_shovels_builder import generate_html
        html = generate_html(cfg, fundamentals, performance, profiles)
    except Exception as e:
        # Fallback: minimal HTML
        html = f"<html><body><h1>Error generating report: {e}</h1></body></html>"

    # Save to reports/
    safe_name = cfg.get("title", "research").lower().replace(" ", "_").replace("&", "and")
    safe_name = "".join(c for c in safe_name if c.isalnum() or c == '_')
    out_path = os.path.join("reports", f"{safe_name}.html")
    _Path("reports").mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Cache data
    cache_path = out_path.replace(".html", "_data.json")
    import json as _json
    with open(cache_path, "w") as f:
        _json.dump({"fundamentals": fundamentals, "performance": performance, "profiles": profiles}, f, indent=2, default=str)

    return {
        "status": "ok",
        "report_url": f"/reports/{safe_name}.html",
        "fundamentals_log": fund_log,
        "performance_log": perf_log,
        "profile_log": profile_log,
        "tickers_processed": len(tickers)
    }


@app.get("/api/research/reports")
async def research_list_reports():
    """List all generated research reports"""
    from pathlib import Path as _Path
    reports_dir = _Path("reports")
    if not reports_dir.exists():
        return {"reports": []}

    reports = []
    for f in sorted(reports_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = f.stat()
        size_kb = stat.st_size / 1024
        size_str = f"{size_kb:.0f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
        from datetime import datetime as _dt
        reports.append({
            "name": f.stem.replace("_", " ").title(),
            "filename": f.name,
            "url": f"/reports/{f.name}",
            "date": _dt.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            "size": size_str
        })

    return {"reports": reports}


# Serve reports directory as static files
import pathlib
pathlib.Path("reports").mkdir(exist_ok=True)
app.mount("/reports", StaticFiles(directory="reports"), name="reports")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the server"""
    print("=" * 60)
    print("MTF AUCTION SCANNER - UNIFIED SERVER")
    print("=" * 60)
    
    # Check for API key
    has_key = bool(os.environ.get("FINNHUB_API_KEY"))
    
    print(f"\n📊 System Status:")
    print(f"   Finnhub API Key: {'✅ Set' if has_key else '⚠️ Not set'}")
    watchlists = watchlist_mgr.get_all_watchlists()
    print(f"   Watchlists: {len(watchlists)}")
    print(f"   Total Symbols: {sum(len(lst.symbols) for lst in watchlists)}")
    
    if not has_key:
        print("\n⚠️  To enable live scanning:")
        print("   macOS/Linux: export FINNHUB_API_KEY=your_key_here")
        print('   Windows PowerShell: $env:FINNHUB_API_KEY = "your_key_here"')
        print("   Windows cmd.exe: set FINNHUB_API_KEY=your_key_here")
        print("   Or use the web interface to set it")
    
    print(f"\n🌐 Starting server...")
    print(f"   Open: http://localhost:8000")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

