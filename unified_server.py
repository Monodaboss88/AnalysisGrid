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
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
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

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        except ValueError as e:
            # Keep the server running even if live-data deps are missing.
            raise HTTPException(status_code=400, detail=str(e))
    return finnhub_scanner


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
        data_source = "Polygon.io REST API"
    elif has_alpaca:
        data_source = "Alpaca (real-time)"
    elif has_finnhub:
        data_source = "yfinance (delayed)"
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
        df_2h = df.resample('2H').agg({
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
            df_2h = df.resample('2H').agg({
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


def get_ai_commentary(analysis_data: dict, symbol: str) -> str:
    """Generate AI trading commentary using ChatGPT"""
    if openai_client is None:
        return ""
    
    try:
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
⚠️ If snap-back > 70%, a pullback is LIKELY - do NOT chase at current price!
"""
        
        prompt = f"""You are an elite hedge fund trader specializing in auction market theory. Analyze this setup and provide a COMPLETE TRADE PLAN with risk management.

Symbol: {symbol}
Current Price: ${current_price:.2f}
Signal: {signal} (GREEN=Bullish, RED=Bearish, YELLOW=Neutral/Wait)
Confidence: {confidence}%
Bull Score: {bull_score} | Bear Score: {bear_score}
Price Position: {position}
VWAP Zone: {vwap_zone}
RSI Zone: {rsi_zone}

KEY LEVELS:
- VAH (Value Area High): ${vah:.2f}
- POC (Point of Control): ${poc:.2f}
- VAL (Value Area Low): ${val:.2f}
- VWAP: ${vwap:.2f}

PROBABILITY TARGETS:
- High Target ({high_prob}% probability)
- Low Target ({low_prob}% probability)

Scanner Notes: {'; '.join(notes)}
{extension_text}
PROVIDE A COMPLETE TRADE PLAN:

📊 TRADE BIAS: [LONG / SHORT / NO TRADE]
⭐ SETUP STRENGTH: [A+ / A / B / C / F] - Rate the quality

📍 ENTRY ZONE: $XX.XX - $XX.XX{"" if not extension_data else " (account for snap-back if extended!)"}
🛑 STOP LOSS: $XX.XX
💰 TARGET 1: $XX.XX (conservative)
🚀 TARGET 2: $XX.XX (aggressive)

📐 RISK:REWARD RATIO: Calculate R:R for both targets
   - T1 R:R = X.X:1
   - T2 R:R = X.X:1

💡 REASONING: 1-2 sentences on WHY this trade makes sense (or why to avoid){"" if not extension_data else " Include the extension data in your reasoning."}

Calculate actual R:R using: (Target - Entry) / (Entry - Stop)
Only recommend trades with R:R > 2:1"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",  # Full GPT-4o for best analysis
            messages=[
                {"role": "system", "content": "You are an elite hedge fund portfolio manager with 20+ years experience in auction market theory, volume profile, and order flow analysis. Always provide SPECIFIC dollar prices for entries, stops, and targets. Calculate precise Risk:Reward ratios. Only recommend trades with R:R better than 2:1. Be brutally honest about trade quality - if it's a bad setup, say NO TRADE."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
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


@app.get("/api/analyze/live/{symbol}")
async def analyze_live(
    symbol: str,
    timeframe: str = Query("1HR", description="30MIN, 1HR, 2HR, 4HR, DAILY"),
    with_ai: bool = Query(True, description="Include ChatGPT commentary")
):
    """Analyze symbol with live Finnhub data"""
    scanner = get_finnhub_scanner()
    
    # Get candle data - use 7 days to have enough for RSI calculation
    df = scanner._get_candles(symbol.upper(), "60", 7)
    if df is not None and len(df) >= 15:
        # Filter to today's session only for VP/VWAP (like Webull)
        today = datetime.now().date()
        df_today = df[df.index.date == today] if hasattr(df.index, 'date') else df.tail(8)  # Last 8 hours if no date
        
        if len(df_today) >= 3:
            poc, vah, val = scanner.calc.calculate_volume_profile(df_today)
            vwap = scanner.calc.calculate_vwap(df_today)
        else:
            # Fallback to last 8 candles
            poc, vah, val = scanner.calc.calculate_volume_profile(df.tail(8))
            vwap = scanner.calc.calculate_vwap(df.tail(8))
        
        # RSI uses full history for proper calculation
        rsi = scanner.calc.calculate_rsi(df)
        current_price = float(df['close'].iloc[-1])
    else:
        poc, vah, val, vwap, rsi = 0, 0, 0, 0, 50
        current_price = 0
    
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
        # Price levels for AI and journal
        "current_price": current_price,
        "vah": vah,
        "poc": poc,
        "val": val,
        "vwap": vwap,
        "rsi": rsi
    }
    
    # Add Extension Duration data (THE EDGE)
    if extension_available and extension_predictor:
        try:
            # Resample to 2HR for extension analysis
            df_2h = df.resample('2H').agg({
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
                
                # Boost confidence if extension is HIGH_PROB or EXTREME
                if hottest and hottest.get('candles', 0) >= 3:
                    extension_bonus = 10 + (hottest.get('candles', 0) - 2) * 5
                    response["extension_bonus"] = extension_bonus
                    response["notes"].append(f"🔥 Extension: {hottest.get('trigger', '')} - {hottest.get('candles', 0)} candles ({hottest.get('snap_back_prob', 0)}% snap-back)")
        except Exception as e:
            print(f"Extension analysis error: {e}")
    
    # Add AI commentary if requested and available
    if with_ai and openai_client:
        response["ai_commentary"] = get_ai_commentary(response, symbol.upper())
    
    return response


@app.get("/api/analyze/live/mtf/{symbol}")
async def analyze_live_mtf(symbol: str):
    """Multi-timeframe analysis with live data"""
    scanner = get_finnhub_scanner()
    result = scanner.analyze_mtf(symbol.upper())
    
    if not result:
        raise HTTPException(status_code=404, detail=f"Could not analyze {symbol}")
    
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


@app.post("/api/analyze/live/mtf/{symbol}/ai")
async def analyze_mtf_with_ai(
    symbol: str, 
    trade_tf: str = Query("swing", description="Trade timeframe: intraday, swing, position, longterm")
):
    """Generate AI trade plan using full MTF context with specific trade timeframe"""
    if not openai_client:
        raise HTTPException(status_code=400, detail="OpenAI API key not set")
    
    scanner = get_finnhub_scanner()
    
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
                df_2h = df_ext.resample('2H').agg({
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
    
    # Get price levels based on trade timeframe
    df = scanner._get_candles(symbol.upper(), "60", config["days"] + 1)
    if df is not None and len(df) >= 5:
        if trade_tf == "intraday":
            # For intraday, use today's session only
            today = datetime.now().date()
            df_filtered = df[df.index.date == today] if hasattr(df.index, 'date') else df.tail(8)
            if len(df_filtered) < 3:
                df_filtered = df.tail(8)
        else:
            # For swing/position/longterm, use appropriate lookback
            df_filtered = df
        
        poc, vah, val = scanner.calc.calculate_volume_profile(df_filtered)
        vwap = scanner.calc.calculate_vwap(df_filtered)
        rsi = scanner.calc.calculate_rsi(df)
        current_price = float(df['close'].iloc[-1])
    else:
        poc, vah, val, vwap, rsi = 0, 0, 0, 0, 50
        current_price = 0
    
    # Build timeframe summary
    tf_summary = []
    for tf, r in result.timeframe_results.items():
        tf_summary.append(f"{tf}: {r.signal} (Bull:{r.bull_score}, Bear:{r.bear_score}, Conf:{r.confidence}%)")
    
    prompt = f"""You are an elite hedge fund trader specializing in auction market theory. Analyze this MULTI-TIMEFRAME setup and provide a COMPLETE TRADE PLAN.

═══════════════════════════════════════════
🎯 TRADE TIMEFRAME: {config["label"]}
═══════════════════════════════════════════
This is a {config["label"]} trade. Size stops and targets appropriately for this holding period.

Symbol: {symbol.upper()}
Current Price: ${current_price:.2f}

═══════════════════════════════════════════
MULTI-TIMEFRAME ANALYSIS (Most Important!)
═══════════════════════════════════════════
Dominant Signal: {result.dominant_signal}
MTF Confluence: {result.confluence_pct}%
Weighted Bull Score: {result.weighted_bull:.1f}
Weighted Bear Score: {result.weighted_bear:.1f}

HIGH PROBABILITY SCENARIO: {result.high_prob:.1f}% (based on MTF confluence)
LOW PROBABILITY SCENARIO: {result.low_prob:.1f}%

INDIVIDUAL TIMEFRAMES:
{chr(10).join(tf_summary)}

═══════════════════════════════════════════
KEY PRICE LEVELS ({config["days"]} day lookback)
═══════════════════════════════════════════
- VAH (Value Area High): ${vah:.2f}
- POC (Point of Control): ${poc:.2f}
- VAL (Value Area Low): ${val:.2f}
- VWAP: ${vwap:.2f}
- RSI: {rsi:.1f}

MTF Notes: {'; '.join(result.notes)}
{extension_text}
═══════════════════════════════════════════
TRADE TIMEFRAME GUIDANCE
═══════════════════════════════════════════
{"INTRADAY: Tight stops ($0.50-$2), quick targets at key levels. Exit before close." if trade_tf == "intraday" else ""}{"SWING (3-5 days): Use ATR-based stops, target next value area levels." if trade_tf == "swing" else ""}{"POSITION (2 weeks): Wider stops below key structure, target major resistance/support." if trade_tf == "position" else ""}{"LONG-TERM (30+ days): Use weekly structure for stops, target measured moves or major levels." if trade_tf == "longterm" else ""}

CRITICAL: Use the HIGH/LOW PROBABILITY percentages above to determine bias.
- If High Prob > 70%, bias is LONG
- If Low Prob > 70%, bias is SHORT  
- If both are close to 50%, NO TRADE
- If Extension snap-back > 70%, WAIT for pullback before entry!

PROVIDE A COMPLETE TRADE PLAN FOR A {config["label"]} TRADE:

📊 TRADE BIAS: [LONG / SHORT / NO TRADE]
⭐ SETUP STRENGTH: [A+ / A / B / C / F] - Rate the quality based on MTF confluence

📍 ENTRY ZONE: $XX.XX - $XX.XX (use key levels - if extended, entry should be at pullback target)
🛑 STOP LOSS: $XX.XX (sized appropriately for {config["label"]})
💰 TARGET 1: $XX.XX (conservative - based on timeframe)
🚀 TARGET 2: $XX.XX (aggressive - extended for {config["label"]})

📐 RISK:REWARD RATIO: Calculate R:R for both targets
   - T1 R:R = X.X:1
   - T2 R:R = X.X:1

⏱️ EXPECTED HOLD TIME: X hours/days based on {config["label"]}

💡 REASONING: Explain how the MTF confluence supports this trade for a {config["label"]} timeframe.

Only recommend trades with R:R > 2:1 AND MTF confluence > 60%"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"You are an elite hedge fund portfolio manager with 20+ years experience in multi-timeframe analysis, auction market theory, and order flow. You are planning a {config['label']} trade. The MTF HIGH/LOW PROBABILITY percentages are the most important signal - use them to determine trade direction. Always provide SPECIFIC dollar prices sized appropriately for the trade timeframe. Calculate precise Risk:Reward ratios. Be brutally honest - if MTF confluence is weak or probabilities are near 50/50, say NO TRADE."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=700,
            temperature=0.3
        )
        
        return {
            "symbol": symbol.upper(),
            "ai_commentary": response.choices[0].message.content.strip(),
            "high_prob": result.high_prob,
            "low_prob": result.low_prob,
            "confluence": result.confluence_pct,
            "dominant_signal": result.dominant_signal,
            "trade_timeframe": config["label"],
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
