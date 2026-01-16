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
        print("✅ ChatGPT commentary enabled")
    except Exception as e:
        print(f"⚠️ OpenAI init failed: {e}")


def get_finnhub_scanner() -> FinnhubScanner:
    """Get or create Finnhub scanner"""
    global finnhub_scanner
    if finnhub_scanner is None:
        api_key = os.environ.get("FINNHUB_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="FINNHUB_API_KEY not set. Set environment variable or use /api/set-key endpoint."
            )
        try:
            finnhub_scanner = FinnhubScanner(api_key)
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
    
    # Determine data source
    if has_polygon:
        data_source = "Polygon.io"
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
        if finnhub_scanner.polygon_client:
            return {"status": "ok", "message": "Polygon.io data enabled!"}
        else:
            return {"status": "warning", "message": "Polygon key set but client failed to initialize"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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

PROVIDE A COMPLETE TRADE PLAN:

📊 TRADE BIAS: [LONG / SHORT / NO TRADE]
⭐ SETUP STRENGTH: [A+ / A / B / C / F] - Rate the quality

📍 ENTRY ZONE: $XX.XX - $XX.XX
🛑 STOP LOSS: $XX.XX
💰 TARGET 1: $XX.XX (conservative)
🚀 TARGET 2: $XX.XX (aggressive)

📐 RISK:REWARD RATIO: Calculate R:R for both targets
   - T1 R:R = X.X:1
   - T2 R:R = X.X:1

💡 REASONING: 1-2 sentences on WHY this trade makes sense (or why to avoid)

Calculate actual R:R using: (Target - Entry) / (Entry - Stop)
Only recommend trades with R:R > 2:1"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional hedge fund risk manager. Always calculate precise Risk:Reward ratios. Only recommend trades with R:R better than 2:1. Format all prices as $XXX.XX. Be brutally honest about trade quality."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.5
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
    with_ai: bool = Query(False, description="Include ChatGPT commentary")
):
    """Analyze symbol with live Finnhub data"""
    scanner = get_finnhub_scanner()
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
        "timestamp": datetime.now().isoformat()
    }
    
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
    signal_order = {"LONG_SETUP": 0, "SHORT_SETUP": 1, "YELLOW": 2, "NEUTRAL": 3}
    results.sort(key=lambda x: (signal_order.get(x["signal"], 4), -x["confidence"]))
    
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
    limit: int = Query(20, ge=1, le=200, description="Max symbols to scan (rate-limited)")
):
    """Scan entire watchlist"""
    lst = watchlist_mgr.get_watchlist(name)
    if not lst:
        raise HTTPException(status_code=404, detail=f"Watchlist '{name}' not found")
    
    scanner = get_finnhub_scanner()

    # Scan enabled symbols only
    symbols = watchlist_mgr.get_enabled_symbols(watchlist_name=name)
    symbols = symbols[:limit]
    
    results = []
    for i, symbol in enumerate(symbols):
        print(f"Scanning {symbol} ({i+1}/{len(symbols)})...")
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
                "rsi_zone": result.rsi_zone
            })
        time.sleep(0.5)
    
    # Sort
    signal_order = {"LONG_SETUP": 0, "SHORT_SETUP": 1, "YELLOW": 2, "NEUTRAL": 3}
    results.sort(key=lambda x: (signal_order.get(x["signal"], 4), -x["confidence"]))
    
    return {
        "watchlist": name,
        "timeframe": timeframe,
        "count": len(results),
        "requested": len(symbols),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


# -----------------------------------------------------------------------------
# ALERTS
# -----------------------------------------------------------------------------

@app.get("/api/alerts")
async def get_alerts(symbol: str = None):
    """Get active alerts"""
    alerts = chart_system.get_alerts(symbol)
    return {
        "count": len(alerts),
        "alerts": [asdict(a) for a in alerts]
    }


@app.post("/api/alerts")
async def create_alert(request: AlertRequest):
    """Create new alert"""
    alert = chart_system.add_alert(
        symbol=request.symbol,
        level=request.level,
        direction=request.direction,
        action=request.action,
        note=request.note
    )
    return {"status": "created", "alert": asdict(alert)}


@app.delete("/api/alerts")
async def delete_alert(symbol: str, level: float):
    """Delete an alert"""
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


# -----------------------------------------------------------------------------
# TRADE TRACKER
# -----------------------------------------------------------------------------

@app.get("/api/trades")
async def get_trades(symbol: str = None, status: str = None):
    """Get trades"""
    if status == "pending":
        trades = chart_system.get_pending_trades(symbol)
    else:
        trades = chart_system.tracker.trades
        if symbol:
            trades = [t for t in trades if t.symbol == symbol.upper()]
    
    return {
        "count": len(trades),
        "trades": [asdict(t) for t in trades]
    }


@app.post("/api/trades")
async def log_trade(request: TradeRequest):
    """Log a new trade setup"""
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
    return {"status": "logged", "trade": asdict(trade)}


@app.put("/api/trades")
async def update_trade(request: TradeUpdateRequest):
    """Update trade status"""
    trade = chart_system.update_trade(
        request.symbol,
        request.status,
        request.exit_price
    )
    if trade:
        return {"status": "updated", "trade": asdict(trade)}
    raise HTTPException(status_code=404, detail="Trade not found")


@app.get("/api/trades/stats")
async def get_trade_stats():
    """Get trading statistics"""
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
