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
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import asdict

# FastAPI
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
import uvicorn

# Our modules
from chart_input_analyzer import ChartInputSystem, ChartInput
from finnhub_scanner import FinnhubScanner, TechnicalCalculator
from watchlist_manager import WatchlistManager

# OpenAI for AI commentary
try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False
    OpenAI = None

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

# WebSocket Streaming (real-time minute bars)
try:
    from polygon_websocket import StreamingManager, MinuteBar
    streaming_available = True
    streaming_manager = StreamingManager.get_instance()
except ImportError as e:
    streaming_available = False
    streaming_manager = None
    print(f"⚠️ WebSocket streaming not loaded: {e}")

# Extension Duration Predictor (THE EDGE)
try:
    from extension_predictor import ExtensionPredictor, CandleData, ExtensionAlert
    extension_predictor = ExtensionPredictor(candle_minutes=120)
    extension_available = True
    print("✅ Extension Duration Predictor enabled")
except ImportError as e:
    extension_available = False
    extension_predictor = None
    print(f"⚠️ Extension Predictor not loaded: {e}")


# =============================================================================
# HELPERS
# =============================================================================

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
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
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

# Initialize Firebase Auth
if auth_available:
    init_firebase()

# Initialize components
chart_system = ChartInputSystem(data_dir="./scanner_data")
watchlist_mgr = WatchlistManager()

# Finnhub scanner (initialized on first use with API key)
finnhub_scanner: Optional[FinnhubScanner] = None

# OpenAI client for AI commentary
openai_client = None
if openai_available and os.environ.get("OPENAI_API_KEY"):
    try:
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        print("✅ ChatGPT commentary enabled (from env)")
        # Share with Range Watcher
        if set_range_openai:
            set_range_openai(openai_client)
    except Exception as e:
        print(f"⚠️ OpenAI init failed: {e}")


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
    return FileResponse("unified_ui.html")


@app.get("/api/status")
async def get_status():
    """Get system status"""
    has_finnhub = bool(os.environ.get("FINNHUB_API_KEY"))
    has_alpaca = bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"))
    has_polygon = bool(os.environ.get("POLYGON_API_KEY"))
    has_openai = openai_client is not None
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
        "timestamp": datetime.now().isoformat()
    }


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
    Analyze extension duration for a symbol
    
    Returns snap-back probability based on time extended
    """
    if not extension_available or not extension_predictor:
        raise HTTPException(status_code=400, detail="Extension Predictor not available")
    
    try:
        scanner = get_finnhub_scanner()
        
        # Get 2-hour candles (resample from 1HR)
        df = scanner._get_candles(symbol.upper(), "60", days_back=10)
        if df is None or len(df) < 20:
            raise HTTPException(status_code=404, detail=f"Insufficient data for {symbol}")
        
        # Resample to 2HR
        df_2h = df.resample('2h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        if len(df_2h) < 5:
            raise HTTPException(status_code=404, detail=f"Insufficient 2H data for {symbol}")
        
        # Calculate VP levels
        poc, vah, val = scanner.calc.calculate_volume_profile(df)
        vwap = scanner.calc.calculate_vwap(df)
        
        # Analyze extension
        from extension_predictor import CandleData
        alerts = extension_predictor.analyze_from_dataframe(
            symbol=symbol.upper(),
            df=df_2h,
            vwap=vwap,
            poc=poc,
            vah=vah,
            val=val
        )
        
        # Get active streaks
        streaks = extension_predictor.get_active_streaks(symbol.upper())
        hottest = extension_predictor.get_hottest_setup(symbol.upper())
        
        return {
            "symbol": symbol.upper(),
            "timeframe": "2HR",
            "levels": {
                "vwap": round(vwap, 2),
                "poc": round(poc, 2),
                "vah": round(vah, 2),
                "val": round(val, 2)
            },
            "extension": {
                "active_streaks": streaks,
                "hottest_setup": hottest,
                "alerts": [a.to_dict() for a in alerts] if alerts else []
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extension analysis failed: {str(e)}")


@app.get("/api/extension/alerts")
async def get_extension_alerts():
    """Get all active extension alerts across watched symbols"""
    if not extension_available or not extension_predictor:
        raise HTTPException(status_code=400, detail="Extension Predictor not available")
    
    alerts = extension_predictor.get_all_alerts()
    
    return {
        "count": len(alerts),
        "alerts": alerts,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/extension/scan")
async def scan_extensions(symbols: List[str] = None):
    """
    Scan multiple symbols for extension setups
    
    Returns symbols with HIGH_PROB or EXTREME extension alerts
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
    
    scanner = get_finnhub_scanner()
    results = []
    
    for symbol in symbols:
        try:
            # Quick analysis
            df = scanner._get_candles(symbol.upper(), "60", days_back=10)
            if df is None or len(df) < 20:
                continue
            
            # Resample to 2HR
            df_2h = df.resample('2h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            if len(df_2h) < 5:
                continue
            
            # Calculate levels
            poc, vah, val = scanner.calc.calculate_volume_profile(df)
            vwap = scanner.calc.calculate_vwap(df)
            
            # Analyze
            alerts = extension_predictor.analyze_from_dataframe(
                symbol=symbol.upper(),
                df=df_2h,
                vwap=vwap,
                poc=poc,
                vah=vah,
                val=val
            )
            
            hottest = extension_predictor.get_hottest_setup(symbol.upper())
            
            if hottest and hottest.get('candles', 0) >= 2:
                results.append({
                    "symbol": symbol.upper(),
                    "setup": hottest,
                    "price": round(float(df_2h['close'].iloc[-1]), 2)
                })
                
        except Exception as e:
            print(f"Extension scan error for {symbol}: {e}")
            continue
    
    # Sort by candle count
    results.sort(key=lambda x: x['setup'].get('candles', 0), reverse=True)
    
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
                
                # Bullish scan
                if scan_type in ['bullish', 'all']:
                    is_bullish = result.signal and 'LONG' in str(result.signal)
                    score_gap = (result.bull_score or 0) - (result.bear_score or 0)
                    is_bullish_lean = score_gap >= 15 and (result.bull_score or 0) >= 45
                    
                    if is_bullish or is_bullish_lean:
                        results["bullish"].append({
                            "symbol": symbol.upper(),
                            "signal": result.signal or "NEUTRAL",
                            "confidence": result.confidence or 0,
                            "bull_score": result.bull_score or 0,
                            "bear_score": result.bear_score or 0,
                            "position": result.position or "-"
                        })
                
                # Bearish scan
                if scan_type in ['bearish', 'all']:
                    is_bearish = result.signal and 'SHORT' in str(result.signal)
                    score_gap = (result.bear_score or 0) - (result.bull_score or 0)
                    is_bearish_lean = score_gap >= 15 and (result.bear_score or 0) >= 45
                    
                    if is_bearish or is_bearish_lean:
                        results["bearish"].append({
                            "symbol": symbol.upper(),
                            "signal": result.signal or "NEUTRAL",
                            "confidence": result.confidence or 0,
                            "bull_score": result.bull_score or 0,
                            "bear_score": result.bear_score or 0,
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
    """Set OpenAI API key for ChatGPT commentary"""
    global openai_client
    os.environ["OPENAI_API_KEY"] = api_key
    
    if not openai_available:
        raise HTTPException(status_code=400, detail="OpenAI package not installed. Run: pip install openai")
    
    try:
        openai_client = OpenAI(api_key=api_key)
        # Test the connection
        openai_client.models.list()
        
        # Share OpenAI client with Range Watcher
        if set_range_openai:
            set_range_openai(openai_client)
        
        return {"status": "ok", "message": "ChatGPT commentary enabled!"}
    except Exception as e:
        openai_client = None
        raise HTTPException(status_code=400, detail=f"OpenAI error: {str(e)}")


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


def get_ai_commentary(analysis_data: dict, symbol: str, entry_signal: str = None) -> str:
    """Generate AI trading commentary using ChatGPT"""
    if openai_client is None:
        return ""
    
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
        
        # Determine direction - priority: forced > signal_type playbook > score-based
        if forced_direction:
            primary_direction = f"{forced_direction} (from entry scanner)"
            direction_note = f"Entry scanner detected {signal_type.replace('_', ' ').upper()} - this is a {forced_direction} setup."
        elif signal_type and signal_type.lower() in SIGNAL_PLAYBOOKS:
            playbook = SIGNAL_PLAYBOOKS[signal_type.lower()]
            primary_direction = f"{playbook['direction']} ({playbook['name']})"
            direction_note = f"This is a {playbook['name']} setup. {playbook['setup']}"
        elif bear_score > bull_score:
            primary_direction = "SHORT (bearish)"
            direction_note = "Bear score is higher - this is a SHORT/SELL setup. Give SHORT trade advice."
        elif bull_score > bear_score:
            primary_direction = "LONG (bullish)"
            direction_note = "Bull score is higher - this is a LONG/BUY setup. Give LONG trade advice."
        else:
            primary_direction = "NEUTRAL"
            direction_note = "Scores are equal - no clear direction. Likely NO TRADE."
        
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

LEVELS: VAH ${vah:.2f} | POC ${poc:.2f} | VAL ${val:.2f} | VWAP ${vwap:.2f}

NOTES: {'; '.join(notes[:3]) if notes else 'None'}
{extension_text}
Provide your analysis using the decision tree and output format from your instructions. IMPORTANT: Match the direction indicated above!"""

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
OUTPUT FORMAT (Use exactly this structure)
═══════════════════════════════════════════

📊 TRADE BIAS: [LONG / SHORT / NO TRADE / WAIT]
⭐ SETUP GRADE: [A+ / A / B / C / F]
🎯 CONVICTION: X/10

📈 PROBABILITY:
- Base Rate: X% (state the pattern)
- Adjustments: [list +/- factors]
- Final: X-Y% (±3% if high conf, ±5% medium, ±8% low)
- Confidence: [High/Medium/Low]

📍 ENTRY: $XX.XX - $XX.XX (use MIDPOINT for calculations)
🛑 STOP: $XX.XX (X% risk from entry midpoint)
💰 T1: $XX.XX | 🚀 T2: $XX.XX

📐 R:R CALCULATION (show your math):
- Risk = |Entry - Stop|
- T1 Reward = |T1 - Entry|
- T2 Reward = |T2 - Entry|
- T1 R:R = T1 Reward ÷ Risk
- T2 R:R = T2 Reward ÷ Risk
Format: T1=X.X:1 | T2=X.X:1

💹 EV CALCULATION (show your math):
- EV = (Win% × Reward) - (Loss% × Risk)
- For $100 risk: EV = (Win% × T1_RR × $100) - (Loss% × $100)
- Breakeven win rate = 1 ÷ (1 + T1_RR)
- If your prob > breakeven → POSITIVE, else NEGATIVE

📊 SIZE: X.XX R (reason for sizing)

❌ INVALID IF: [specific price level or action]

💡 REASONING: [2-3 sentences max]

🔄 IF NO TRADE, WOULD RECONSIDER IF:
- [condition 1]
- [condition 2]"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.2
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"⚠️ ChatGPT error: {e}")
        return ""


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
    if with_ai and openai_client:
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
        
        return response
        
    except Exception as e:
        import traceback
        return {"error": str(e), "step": step if 'step' in dir() else "unknown", "traceback": traceback.format_exc()[:1000]}


@app.get("/api/analyze/live/{symbol}")
async def analyze_live(
    symbol: str,
    timeframe: str = Query("1HR", description="30MIN, 1HR, 2HR, 4HR, DAILY"),
    with_ai: bool = Query(True, description="Include ChatGPT commentary"),
    entry_signal: str = Query(None, description="Entry signal from scanner, e.g. 'failed_breakout:short'")
):
    """Analyze symbol with live Finnhub data"""
    try:
        scanner = get_finnhub_scanner()
        
        # Get candle data - use 7 days to have enough for RSI calculation
        df = scanner._get_candles(symbol.upper(), "60", 7)
        if df is not None and len(df) >= 15:
            # Filter to today's session only for VP/VWAP (like Webull)
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
        
        # Get REAL-TIME quote (Polygon paid = real-time)
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
        
        result = scanner.analyze(symbol.upper(), timeframe)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Could not analyze {symbol}")
        
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
        
        # Add entry_signal to response for diagram lookup
        if entry_signal:
            response["entry_signal"] = entry_signal
        
        # Add AI commentary if requested and available
        if with_ai and openai_client:
            response["ai_commentary"] = get_ai_commentary(response, symbol.upper(), entry_signal)
        
        return response
        
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
    trade_tf: str = Query("swing", description="Trade timeframe: intraday, swing, position, longterm"),
    entry_signal: str = Query(None, description="Entry signal from scanner: e.g. 'failed_breakout:short' or 'val_touch_rejection:long'")
):
    """Generate AI trade plan using full MTF context with specific trade timeframe"""
    if not openai_client:
        raise HTTPException(status_code=400, detail="OpenAI API key not set")
    
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
    
    # Trade timeframe settings
    tf_config = {
        "intraday": {"days": 1, "label": "SAME DAY (Intraday)", "stop_mult": 0.3, "target_mult": 0.5},
        "swing": {"days": 5, "label": "3-5 DAY SWING", "stop_mult": 0.5, "target_mult": 1.0},
        "position": {"days": 14, "label": "2 WEEK POSITION", "stop_mult": 1.0, "target_mult": 2.0},
        "longterm": {"days": 30, "label": "30+ DAY SETUP", "stop_mult": 2.0, "target_mult": 4.0}
    }
    config = tf_config.get(trade_tf, tf_config["swing"])
    
    # Use the SAME levels that the UI shows (from the 1HR timeframe analysis)
    # This ensures AI trade plan matches the levels displayed to the user
    key_levels = result.key_levels or {}
    
    # Prefer 1HR levels as that's what the UI typically shows
    vah = key_levels.get("1HR_VAH") or key_levels.get("2HR_VAH") or key_levels.get("30MIN_VAH", 0)
    poc = key_levels.get("1HR_POC") or key_levels.get("2HR_POC") or key_levels.get("30MIN_POC", 0)
    val = key_levels.get("1HR_VAL") or key_levels.get("2HR_VAL") or key_levels.get("30MIN_VAL", 0)
    vwap = key_levels.get("1HR_VWAP") or key_levels.get("2HR_VWAP") or key_levels.get("30MIN_VWAP", 0)
    current_price = key_levels.get("CURRENT", 0)
    
    # Get RSI and volume from fresh data
    df = scanner._get_candles(symbol.upper(), "60", 5)
    if df is not None and len(df) >= 5:
        rsi = scanner.calc.calculate_rsi(df)
        rvol = scanner.calc.calculate_relative_volume(df)
        volume_trend = scanner.calc.calculate_volume_trend(df)
        if current_price == 0:
            current_price = float(df['close'].iloc[-1])
    else:
        rsi = 50
        rvol, volume_trend = 1.0, "neutral"
    
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
HIGH PROB: {result.high_prob:.0f}% | LOW PROB: {result.low_prob:.0f}%
Bull: {result.weighted_bull:.0f} | Bear: {result.weighted_bear:.0f}

VOLUME: {rvol_emoji} RVOL {rvol}x | Trend: {volume_trend}
TIMEFRAMES: {' | '.join(tf_summary)}

LEVELS: VAH ${vah:.2f} | POC ${poc:.2f} | VAL ${val:.2f} | VWAP ${vwap:.2f} | RSI {rsi:.0f}

NOTES: {'; '.join(result.notes[:3]) if result.notes else 'None'}
{extension_text}
Apply decision tree. Lead with {leading_direction} scenario first, then show flip scenario. Output using the exact format from instructions."""

    # Comprehensive system prompt with dual-direction output
    mtf_system_prompt = f"""You are an expert MTF trading analyst planning a {config['label']} trade.

CRITICAL: Output BOTH directions - the LEADING trade first, then the FLIP scenario.
CRITICAL: The LEADING DIRECTION provided is MANDATORY - it was calculated from Extension Predictor and Bull/Bear scores. DO NOT contradict it.

═══════════════════════════════════════════
DECISION TREE (Follow in order - STOP at first failure)
═══════════════════════════════════════════
1. MTF CONFLUENCE < 60%? → NO TRADE
2. HIGH vs LOW PROB within 15%? (conflicting) → NO TRADE  
3. EXTENDED > 75% snap-back? → WAIT (give pullback entry OR trade in snap-back direction only)
4. PROBABILITY < 55%? → NO TRADE
5. R:R < 2:1? → NO TRADE
6. EV NEGATIVE? → NO TRADE
7. ALL PASS → TRADE with sized position

EXTENSION PREDICTOR RULE:
If Extension Predictor shows 70%+ snap-back probability, you MUST respect its direction.
- SHORT SETUP with high snap-back = Lead with SHORT, not LONG
- LONG SETUP with high snap-back = Lead with LONG, not SHORT

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
POSITION SIZING ({config['label']})
═══════════════════════════════════════════
- HIGH confidence (70%+): 1.0R
- MEDIUM confidence (55-70%): 0.75R
- If extended: max 0.5R
- LOW confidence: 0.5R or PASS
- LOW VOLUME (RVOL < 0.7): Reduce size by 50%

═══════════════════════════════════════════
OUTPUT FORMAT - DUAL DIRECTION (Required)
═══════════════════════════════════════════

▶️ PHASE 1 ({leading_direction}) - LEADING TRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 BIAS: {leading_direction}
⭐ GRADE: [A+ / A / B / C / F]
🎯 CONVICTION: X/10

📈 PROBABILITY: X-Y% [High/Med/Low]

📍 ENTRY ZONE: $XX.XX - $XX.XX
📍 ENTRY (midpoint): $XX.XX ← use this for R:R calc
🛑 STOP: $XX.XX
💰 T1: $XX.XX | 🚀 T2: $XX.XX

📐 R:R MATH: 
   Risk = |$Entry - $Stop| = $X.XX
   T1 Reward = |$T1 - $Entry| = $X.XX → T1 R:R = X.X:1
   T2 Reward = |$T2 - $Entry| = $X.XX → T2 R:R = X.X:1
💹 EV: (Win% × Reward) - (Loss% × Risk) = $X.XX per $100 risked → [POSITIVE/NEGATIVE]

📊 SIZE: X.XXR
⏱️ HOLD: X hours/days

💡 WHY: [1-2 sentences on why this is the leading scenario]

🔄 PHASE 2 ({"SHORT" if leading_direction == "LONG" else "LONG"}) - FLIP SCENARIO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ INVALIDATES IF: [specific price/action that kills Phase 1]
🔀 FLIP TO {"SHORT" if leading_direction == "LONG" else "LONG"} IF: [condition, e.g. "VAH reclaimed with volume"]

📍 FLIP ENTRY: $XX.XX
🛑 FLIP STOP: $XX.XX  
💰 FLIP TARGET: $XX.XX

💡 FLIP LOGIC: [1-2 sentences - what would cause the flip and why it becomes valid]

⚠️ CRITICAL: If Phase 1 stops out, don't automatically take Phase 2. Re-evaluate the setup.
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": mtf_system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.2
        )
        
        return {
            "symbol": symbol.upper(),
            "ai_commentary": response.choices[0].message.content.strip(),
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
                "position": result.position
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
# ALERTS (with Firestore support for authenticated users)
# -----------------------------------------------------------------------------

@app.get("/api/alerts")
async def get_alerts(symbol: str = None, user_id: str = None):
    """Get active alerts - uses Firestore for authenticated users"""
    # Use Firestore if user_id provided and Firestore available
    if user_id and firestore_available:
        fs = get_firestore()
        if fs.is_available():
            alerts = fs.get_alerts(user_id, symbol)
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
    """Create new alert - uses Firestore for authenticated users"""
    # Use Firestore if user_id provided
    if user_id and firestore_available:
        fs = get_firestore()
        if fs.is_available():
            alert = UserAlert(
                symbol=request.symbol,
                level=request.level,
                direction=request.direction,
                action=request.action,
                note=request.note or ""
            )
            result = fs.add_alert(user_id, alert)
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
    # Use Firestore if user_id provided
    if user_id and firestore_available:
        fs = get_firestore()
        if fs.is_available():
            success = fs.delete_alert(user_id, symbol, level)
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
    
    # Helper function to add alert
    def add_alert_helper(level, direction, action, note):
        if user_id and firestore_available:
            fs = get_firestore()
            if fs.is_available():
                alert = UserAlert(
                    symbol=symbol,
                    level=level,
                    direction=direction,
                    action=action,
                    note=note
                )
                result = fs.add_alert(user_id, alert)
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
        "storage": "firestore" if (user_id and firestore_available) else "local"
    }


# -----------------------------------------------------------------------------
# TRADE TRACKER (with Firestore support for authenticated users)
# -----------------------------------------------------------------------------

@app.get("/api/trades")
async def get_trades(symbol: str = None, status: str = None, user_id: str = None):
    """Get trades - uses Firestore for authenticated users"""
    # Use Firestore if user_id provided
    if user_id and firestore_available:
        fs = get_firestore()
        if fs.is_available():
            trades = fs.get_trades(user_id, symbol, status)
            return {
                "count": len(trades),
                "trades": trades,
                "storage": "firestore"
            }
    
    # Fallback to local
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
    """Log a new trade setup - uses Firestore for authenticated users"""
    # Use Firestore if user_id provided
    if user_id and firestore_available:
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
            result = fs.add_trade(user_id, trade)
            if result:
                return {"status": "logged", "trade": result, "storage": "firestore"}
    
    # Fallback to local
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
    """Update trade status - uses Firestore for authenticated users"""
    # Use Firestore if user_id provided
    if user_id and firestore_available:
        fs = get_firestore()
        if fs.is_available():
            # For Firestore, we need trade_id instead of symbol
            if hasattr(request, 'trade_id') and request.trade_id:
                result = fs.close_trade(user_id, request.trade_id, request.exit_price or 0, request.status)
                if result:
                    return {"status": "updated", "trade": result, "storage": "firestore"}
    
    # Fallback to local
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
    """Get trading statistics - uses Firestore for authenticated users"""
    if user_id and firestore_available:
        fs = get_firestore()
        if fs.is_available():
            return fs.get_trade_stats(user_id)
    
    return chart_system.get_trade_stats()


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
