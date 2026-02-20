"""
MTF Auction Scanner - API Server v2
===================================
FastAPI server with full watchlist management, search, and scanning.

Run with: uvicorn server_v2:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

# Import our modules
from watchlist_manager import WatchlistManager, WatchlistSymbol, quick_scan_list
from integrated_scanner import IntegratedScanner, IntegratedAnalysis
from mtf_auction_scanner import MTFAuctionScanner, SignalState
from overnight_model import OvernightPredictionEngine

# Try polygon_data
try:
    from polygon_data import get_bars
    POLYGON_DATA_AVAILABLE = True
except ImportError:
    POLYGON_DATA_AVAILABLE = False

import pandas as pd
import numpy as np

# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="MTF Auction Scanner API v2",
    description="Complete trading scanner with watchlist management",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class SymbolAdd(BaseModel):
    symbol: str
    name: str = ""
    category: str = "Custom"
    sector: str = ""
    tags: List[str] = []

class WatchlistCreate(BaseModel):
    name: str
    description: str = ""

class ScanRequest(BaseModel):
    symbols: List[str]

class QuickScanRequest(BaseModel):
    list_type: str  # "indices", "sectors", "mega", "dow", "nasdaq", "sp500"

# =============================================================================
# GLOBAL STATE
# =============================================================================

class AppState:
    def __init__(self):
        self.watchlist_manager = WatchlistManager()
        self.scanner = IntegratedScanner()
        self.mtf_scanner = MTFAuctionScanner()
        self.data_cache: Dict[str, tuple] = {}
        self.scan_results: Dict[str, Any] = {}
        self.last_scan_time: Optional[datetime] = None

state = AppState()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def fetch_data(symbol: str, days: int = 15) -> Optional[pd.DataFrame]:
    """Fetch market data with caching"""
    symbol = symbol.upper()
    
    # Check cache (5 min)
    if symbol in state.data_cache:
        data, ts = state.data_cache[symbol]
        if (datetime.now() - ts).seconds < 300:
            return data
    
    if POLYGON_DATA_AVAILABLE:
        try:
            df = get_bars(symbol, period=f"{days}d", interval="5m")
            if not df.empty:
                df.columns = df.columns.str.lower()
                df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
                state.data_cache[symbol] = (df, datetime.now())
                return df
        except:
            pass
    
    # Demo data fallback
    return generate_demo_data(symbol, days)


def generate_demo_data(symbol: str, days: int = 10) -> pd.DataFrame:
    """Generate demo data"""
    np.random.seed(hash(symbol) % 2**32)
    periods = days * 24 * 12
    base = 50 + (hash(symbol) % 500)
    
    data = []
    ts = datetime.now() - pd.Timedelta(days=days)
    price = base
    
    for i in range(periods):
        noise = np.random.randn() * 0.002
        open_p = price
        close_p = price * (1 + noise)
        high_p = max(open_p, close_p) * 1.001
        low_p = min(open_p, close_p) * 0.999
        vol = int(np.random.exponential(100000))
        
        data.append({
            'timestamp': ts,
            'open': open_p, 'high': high_p, 'low': low_p,
            'close': close_p, 'volume': vol
        })
        price = close_p
        ts += pd.Timedelta(minutes=5)
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


def format_scan_result(symbol: str, analysis: IntegratedAnalysis) -> Dict[str, Any]:
    """Format analysis to JSON-serializable dict"""
    mtf = analysis.mtf_scan
    overnight = analysis.overnight
    sym_info = state.watchlist_manager.all_symbols.get(symbol.upper())
    
    # Get primary timeframe
    tf_data = {}
    for tf, a in mtf.timeframe_analyses.items():
        tf_data[tf.label] = {
            "signal": a.signal.value,
            "signal_emoji": a.signal.emoji,
            "bull_score": round(a.bull_score, 1),
            "bear_score": round(a.bear_score, 1),
            "confidence": round(a.confidence, 1),
            "position": a.position_in_value,
            "rsi": {
                "value": round(a.rsi.value, 1),
                "zone": a.rsi.zone,
            },
            "flow": {
                "imbalance": round(a.flow.flow_imbalance, 3),
                "state": a.flow.flow_state,
            },
            "volume_profile": {
                "poc": round(a.volume_profile.poc, 2),
                "vah": round(a.volume_profile.vah, 2),
                "val": round(a.volume_profile.val, 2),
            }
        }
        if a.vwap:
            tf_data[tf.label]["vwap"] = {
                "value": round(a.vwap.vwap, 2),
                "zone": a.vwap.zone,
                "deviation_pct": round(a.vwap.deviation_pct, 2)
            }
    
    result = {
        "symbol": symbol.upper(),
        "name": sym_info.name if sym_info else "",
        "category": sym_info.category if sym_info else "",
        "sector": sym_info.sector if sym_info else "",
        "scan_time": analysis.analysis_time.isoformat(),
        
        "signal": mtf.dominant_signal.value,
        "signal_emoji": mtf.dominant_signal.emoji,
        "combined_bias": analysis.combined_bias,
        "confidence": round(analysis.combined_confidence, 1),
        "actionable": analysis.trade_plan is not None,
        "confluence": round(mtf.confluence_score, 1),
        
        "probabilities": {
            "high": round(analysis.high_scenario_prob * 100, 1),
            "low": round(analysis.low_scenario_prob * 100, 1),
            "chop": round(analysis.chop_scenario_prob * 100, 1)
        },
        
        "timeframes": tf_data,
        "key_levels": {k: round(v, 2) for k, v in analysis.key_levels.items()},
        "trade_plan": analysis.trade_plan,
        "notes": analysis.notes,
        
        "in_sp500": sym_info.in_sp500 if sym_info else False,
        "in_nasdaq100": sym_info.in_nasdaq100 if sym_info else False,
        "in_dow30": sym_info.in_dow30 if sym_info else False
    }
    
    # Overnight data
    if overnight:
        result["overnight"] = {
            "bias": overnight.bias.value,
            "bias_emoji": overnight.bias.emoji,
            "gap_type": overnight.gap.gap_type.value,
            "gap_pct": round(overnight.gap.gap_pct, 2),
            "gap_fill_prob": overnight.gap.gap_fill_probability.value,
            "overnight_high": round(overnight.overnight.overnight_high, 2),
            "overnight_low": round(overnight.overnight.overnight_low, 2),
            "prior_close": round(overnight.prior_day.prior_close, 2)
        }
    
    return result


def symbol_to_dict(sym: WatchlistSymbol) -> Dict:
    """Convert WatchlistSymbol to dict"""
    return {
        "symbol": sym.symbol,
        "name": sym.name,
        "category": sym.category,
        "sector": sym.sector,
        "market_cap": sym.market_cap,
        "enabled": sym.enabled,
        "is_favorite": sym.is_favorite,
        "in_sp500": sym.in_sp500,
        "in_nasdaq100": sym.in_nasdaq100,
        "in_dow30": sym.in_dow30,
        "tags": sym.tags
    }


# =============================================================================
# ROUTES - FRONTEND
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML frontend"""
    try:
        with open("index_v2.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        try:
            with open("index.html", "r") as f:
                return f.read()
        except:
            return HTMLResponse("<h1>Frontend not found</h1>")


# =============================================================================
# ROUTES - SEARCH
# =============================================================================

@app.get("/api/search")
async def search_symbols(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Search for symbols by ticker or name
    
    Examples:
    - /api/search?q=AAPL
    - /api/search?q=apple
    - /api/search?q=tech&limit=50
    """
    results = state.watchlist_manager.search(q, limit=limit)
    return {
        "query": q,
        "count": len(results),
        "results": [symbol_to_dict(s) for s in results]
    }


@app.get("/api/search/index/{index}")
async def search_by_index(index: str):
    """
    Get all symbols in an index
    
    Supported: SPY, QQQ, DIA (or SPX, NDX, DJIA)
    """
    symbols = state.watchlist_manager.get_by_index(index)
    return {
        "index": index.upper(),
        "count": len(symbols),
        "symbols": [symbol_to_dict(s) for s in symbols]
    }


@app.get("/api/search/sector/{sector}")
async def search_by_sector(sector: str):
    """Get all symbols in a sector"""
    symbols = state.watchlist_manager.get_by_sector(sector)
    return {
        "sector": sector,
        "count": len(symbols),
        "symbols": [symbol_to_dict(s) for s in symbols]
    }


@app.get("/api/sectors")
async def get_all_sectors():
    """Get list of all available sectors"""
    return {"sectors": state.watchlist_manager.get_all_sectors()}


@app.get("/api/categories")
async def get_all_categories():
    """Get list of all available categories"""
    return {"categories": state.watchlist_manager.get_all_categories()}


# =============================================================================
# ROUTES - WATCHLISTS
# =============================================================================

@app.get("/api/watchlists")
async def get_all_watchlists():
    """Get all watchlist names and metadata"""
    watchlists = []
    for name in state.watchlist_manager.get_watchlist_names():
        wl = state.watchlist_manager.get_watchlist(name)
        watchlists.append({
            "name": wl.name,
            "description": wl.description,
            "symbol_count": wl.symbol_count,
            "enabled_count": wl.enabled_count,
            "is_builtin": wl.is_builtin
        })
    return {"watchlists": watchlists}


@app.get("/api/watchlist/{name}")
async def get_watchlist(name: str):
    """Get a specific watchlist with all symbols"""
    wl = state.watchlist_manager.get_watchlist(name)
    if not wl:
        raise HTTPException(status_code=404, detail=f"Watchlist '{name}' not found")
    
    return {
        "name": wl.name,
        "description": wl.description,
        "is_builtin": wl.is_builtin,
        "symbol_count": wl.symbol_count,
        "symbols": [symbol_to_dict(s) for s in wl.symbols]
    }


@app.post("/api/watchlist")
async def create_watchlist(request: WatchlistCreate):
    """Create a new custom watchlist"""
    try:
        wl = state.watchlist_manager.create_watchlist(request.name, request.description)
        return {"message": f"Watchlist '{wl.name}' created", "name": wl.name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/watchlist/{name}")
async def delete_watchlist(name: str):
    """Delete a custom watchlist"""
    try:
        if state.watchlist_manager.delete_watchlist(name):
            return {"message": f"Watchlist '{name}' deleted"}
        raise HTTPException(status_code=404, detail="Watchlist not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/watchlist/{name}/symbol")
async def add_symbol_to_watchlist(name: str, request: SymbolAdd):
    """Add a symbol to a watchlist"""
    try:
        sym = state.watchlist_manager.add_symbol(
            name, 
            request.symbol,
            name=request.name,
            sector=request.sector,
            tags=request.tags
        )
        return {"message": f"Added {sym.symbol}", "symbol": symbol_to_dict(sym)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/watchlist/{name}/symbol/{symbol}")
async def remove_symbol_from_watchlist(name: str, symbol: str):
    """Remove a symbol from a watchlist"""
    if state.watchlist_manager.remove_symbol(name, symbol):
        return {"message": f"Removed {symbol.upper()} from {name}"}
    raise HTTPException(status_code=404, detail="Symbol not found in watchlist")


@app.post("/api/watchlist/{name}/symbol/{symbol}/toggle")
async def toggle_symbol_enabled(name: str, symbol: str):
    """Toggle a symbol's enabled state"""
    if state.watchlist_manager.toggle_symbol(name, symbol):
        return {"message": f"Toggled {symbol.upper()}"}
    raise HTTPException(status_code=404, detail="Symbol not found")


@app.post("/api/symbol/{symbol}/favorite")
async def toggle_favorite(symbol: str):
    """Toggle favorite status for a symbol"""
    if state.watchlist_manager.toggle_favorite(symbol):
        sym = state.watchlist_manager.all_symbols.get(symbol.upper())
        return {
            "symbol": symbol.upper(),
            "is_favorite": sym.is_favorite if sym else False
        }
    raise HTTPException(status_code=404, detail="Symbol not found")


# =============================================================================
# ROUTES - SCANNING
# =============================================================================

@app.get("/api/scan/{symbol}")
async def scan_single_symbol(symbol: str):
    """Scan a single symbol"""
    df = fetch_data(symbol)
    if df is None or len(df) < 100:
        raise HTTPException(status_code=400, detail=f"Could not fetch data for {symbol}")
    
    analysis = state.scanner.analyze(df, symbol=symbol.upper())
    result = format_scan_result(symbol, analysis)
    state.scan_results[symbol.upper()] = result
    
    return result


@app.post("/api/scan")
async def scan_symbols(request: ScanRequest):
    """Scan a list of symbols"""
    results = []
    for symbol in request.symbols:
        df = fetch_data(symbol)
        if df is not None and len(df) >= 100:
            analysis = state.scanner.analyze(df, symbol=symbol.upper())
            result = format_scan_result(symbol, analysis)
            results.append(result)
            state.scan_results[symbol.upper()] = result
    
    state.last_scan_time = datetime.now()
    
    # Sort by actionability
    results.sort(key=lambda x: (x['actionable'], x['confidence']), reverse=True)
    
    return {
        "results": results,
        "count": len(results),
        "scan_time": state.last_scan_time.isoformat()
    }


@app.get("/api/scan/watchlist/{name}")
async def scan_watchlist(name: str, enabled_only: bool = True):
    """Scan all symbols in a watchlist"""
    wl = state.watchlist_manager.get_watchlist(name)
    if not wl:
        raise HTTPException(status_code=404, detail=f"Watchlist '{name}' not found")
    
    if enabled_only:
        symbols = [s.symbol for s in wl.symbols if s.enabled]
    else:
        symbols = [s.symbol for s in wl.symbols]
    
    results = []
    for symbol in symbols:
        df = fetch_data(symbol)
        if df is not None and len(df) >= 100:
            analysis = state.scanner.analyze(df, symbol=symbol)
            result = format_scan_result(symbol, analysis)
            results.append(result)
            state.scan_results[symbol] = result
    
    state.last_scan_time = datetime.now()
    results.sort(key=lambda x: (x['actionable'], x['confidence']), reverse=True)
    
    return {
        "watchlist": name,
        "results": results,
        "count": len(results),
        "actionable_count": sum(1 for r in results if r['actionable']),
        "scan_time": state.last_scan_time.isoformat()
    }


@app.get("/api/scan/quick/{list_type}")
async def scan_quick(list_type: str):
    """
    Quick scan using predefined list
    
    list_type options: indices, sectors, mega, dow, nasdaq, sp500, all
    """
    symbols = quick_scan_list(state.watchlist_manager, list_type)
    
    if not symbols:
        raise HTTPException(status_code=400, detail=f"Unknown list type: {list_type}")
    
    results = []
    for symbol in symbols:
        df = fetch_data(symbol)
        if df is not None and len(df) >= 100:
            analysis = state.scanner.analyze(df, symbol=symbol)
            result = format_scan_result(symbol, analysis)
            results.append(result)
            state.scan_results[symbol] = result
    
    state.last_scan_time = datetime.now()
    results.sort(key=lambda x: (x['actionable'], x['confidence']), reverse=True)
    
    return {
        "list_type": list_type,
        "results": results,
        "count": len(results),
        "actionable_count": sum(1 for r in results if r['actionable']),
        "scan_time": state.last_scan_time.isoformat()
    }


@app.get("/api/scan/index/{index}")
async def scan_index_components(index: str):
    """Scan all components of an index (SPY, QQQ, DIA)"""
    symbols = state.watchlist_manager.get_by_index(index)
    
    if not symbols:
        raise HTTPException(status_code=400, detail=f"Unknown index: {index}")
    
    results = []
    for sym in symbols:
        df = fetch_data(sym.symbol)
        if df is not None and len(df) >= 100:
            analysis = state.scanner.analyze(df, symbol=sym.symbol)
            result = format_scan_result(sym.symbol, analysis)
            results.append(result)
            state.scan_results[sym.symbol] = result
    
    state.last_scan_time = datetime.now()
    results.sort(key=lambda x: (x['actionable'], x['confidence']), reverse=True)
    
    return {
        "index": index.upper(),
        "results": results,
        "count": len(results),
        "actionable_count": sum(1 for r in results if r['actionable']),
        "scan_time": state.last_scan_time.isoformat()
    }


@app.get("/api/scan/sector/{sector}")
async def scan_sector(sector: str):
    """Scan all symbols in a sector"""
    symbols = state.watchlist_manager.get_by_sector(sector)
    
    if not symbols:
        raise HTTPException(status_code=400, detail=f"No symbols found for sector: {sector}")
    
    results = []
    for sym in symbols:
        df = fetch_data(sym.symbol)
        if df is not None and len(df) >= 100:
            analysis = state.scanner.analyze(df, symbol=sym.symbol)
            result = format_scan_result(sym.symbol, analysis)
            results.append(result)
            state.scan_results[sym.symbol] = result
    
    state.last_scan_time = datetime.now()
    results.sort(key=lambda x: (x['actionable'], x['confidence']), reverse=True)
    
    return {
        "sector": sector,
        "results": results,
        "count": len(results),
        "actionable_count": sum(1 for r in results if r['actionable']),
        "scan_time": state.last_scan_time.isoformat()
    }


# =============================================================================
# ROUTES - RESULTS & STATS
# =============================================================================

@app.get("/api/results")
async def get_cached_results():
    """Get cached scan results"""
    return {
        "results": list(state.scan_results.values()),
        "count": len(state.scan_results),
        "last_scan": state.last_scan_time.isoformat() if state.last_scan_time else None
    }


@app.get("/api/results/filter")
async def filter_results(
    signal: Optional[str] = None,
    min_confidence: Optional[float] = None,
    actionable_only: bool = False,
    sector: Optional[str] = None,
    in_index: Optional[str] = None
):
    """
    Filter cached scan results
    
    Parameters:
    - signal: LONG_SETUP, SHORT_SETUP, YELLOW, NEUTRAL
    - min_confidence: Minimum confidence threshold
    - actionable_only: Only return actionable setups
    - sector: Filter by sector
    - in_index: Filter by index (SPY, QQQ, DIA)
    """
    results = list(state.scan_results.values())
    filtered = []
    
    for r in results:
        if signal and r['signal'] != signal:
            continue
        if min_confidence and r['confidence'] < min_confidence:
            continue
        if actionable_only and not r['actionable']:
            continue
        if sector and r.get('sector') != sector:
            continue
        if in_index:
            idx = in_index.upper()
            if idx in ["SPY", "SPX"] and not r.get('in_sp500'):
                continue
            if idx in ["QQQ", "NDX"] and not r.get('in_nasdaq100'):
                continue
            if idx in ["DIA", "DOW"] and not r.get('in_dow30'):
                continue
        
        filtered.append(r)
    
    return {
        "results": filtered,
        "count": len(filtered),
        "filters_applied": {
            "signal": signal,
            "min_confidence": min_confidence,
            "actionable_only": actionable_only,
            "sector": sector,
            "in_index": in_index
        }
    }


@app.get("/api/stats")
async def get_stats():
    """Get overall statistics"""
    results = list(state.scan_results.values())
    wl_count = len(state.watchlist_manager.get_watchlist_names())
    total_symbols = len(state.watchlist_manager.all_symbols)
    
    return {
        "total_watchlists": wl_count,
        "total_symbols_available": total_symbols,
        "symbols_scanned": len(results),
        "actionable": sum(1 for r in results if r.get('actionable')),
        "long_setups": sum(1 for r in results if r.get('signal') == 'LONG_SETUP'),
        "short_setups": sum(1 for r in results if r.get('signal') == 'SHORT_SETUP'),
        "yellow": sum(1 for r in results if r.get('signal') == 'YELLOW'),
        "neutral": sum(1 for r in results if r.get('signal') in ['NEUTRAL', 'NO_DATA']),
        "last_scan": state.last_scan_time.isoformat() if state.last_scan_time else None,
        "yfinance_available": YFINANCE_AVAILABLE
    }


# =============================================================================
# ROUTES - HEALTH
# =============================================================================

@app.get("/api/health")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "yfinance_available": YFINANCE_AVAILABLE,
        "watchlists_loaded": len(state.watchlist_manager.get_watchlist_names()),
        "symbols_available": len(state.watchlist_manager.all_symbols),
        "timestamp": datetime.now().isoformat()
    }


# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("=" * 60)
    print("MTF Auction Scanner API v2")
    print("=" * 60)
    print(f"Watchlists loaded: {len(state.watchlist_manager.get_watchlist_names())}")
    print(f"Symbols available: {len(state.watchlist_manager.all_symbols)}")
    print(f"yfinance: {'Available' if YFINANCE_AVAILABLE else 'Not available (using demo data)'}")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
