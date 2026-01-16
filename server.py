"""
MTF Auction Scanner - API Backend
=================================
FastAPI server providing REST endpoints for the scanner operations.

Run with: uvicorn server:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import os
import asyncio

# Import our scanner modules
from mtf_auction_scanner import MTFAuctionScanner, SignalState, Timeframe
from scanner_config import SwingTradeConfig, ScoringConfig

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

import pandas as pd
import numpy as np

# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="MTF Auction Scanner API",
    description="Multi-Timeframe Non-Bias Setup Detection System",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATA MODELS
# =============================================================================

class WatchlistItem(BaseModel):
    id: Optional[str] = None
    symbol: str
    name: str = ""
    category: str = "Custom"
    enabled: bool = True
    created_at: Optional[str] = None

class WatchlistCreate(BaseModel):
    symbol: str
    name: str = ""
    category: str = "Custom"

class WatchlistUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    enabled: Optional[bool] = None

class ScanRequest(BaseModel):
    symbols: List[str]
    timeframes: Optional[List[str]] = None

class ConfigUpdate(BaseModel):
    strong_threshold: Optional[float] = None
    moderate_threshold: Optional[float] = None
    min_score_gap: Optional[float] = None
    min_confluence: Optional[float] = None
    value_area_pct: Optional[float] = None
    rsi_period: Optional[int] = None
    rsi_overbought: Optional[float] = None
    rsi_oversold: Optional[float] = None

# =============================================================================
# IN-MEMORY STORAGE (replace with database for production)
# =============================================================================

class DataStore:
    def __init__(self):
        self.watchlist: Dict[str, WatchlistItem] = {}
        self.scan_results: Dict[str, Any] = {}
        self.config = SwingTradeConfig()
        self.last_scan_time: Optional[datetime] = None
        self._init_default_watchlist()
    
    def _init_default_watchlist(self):
        defaults = [
            ("SPY", "S&P 500 ETF", "Index"),
            ("QQQ", "Nasdaq 100 ETF", "Index"),
            ("IWM", "Russell 2000 ETF", "Index"),
            ("DIA", "Dow Jones ETF", "Index"),
            ("XLF", "Financials", "Sector"),
            ("XLK", "Technology", "Sector"),
            ("XLE", "Energy", "Sector"),
            ("AAPL", "Apple", "Mega Cap"),
            ("MSFT", "Microsoft", "Mega Cap"),
            ("NVDA", "Nvidia", "Mega Cap"),
            ("GOOGL", "Google", "Mega Cap"),
            ("AMZN", "Amazon", "Mega Cap"),
            ("TSLA", "Tesla", "Mega Cap"),
            ("META", "Meta", "Mega Cap"),
        ]
        
        for symbol, name, category in defaults:
            item_id = symbol.lower()
            self.watchlist[item_id] = WatchlistItem(
                id=item_id,
                symbol=symbol,
                name=name,
                category=category,
                enabled=True,
                created_at=datetime.now().isoformat()
            )

store = DataStore()
scanner = MTFAuctionScanner()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def fetch_market_data(symbol: str, days: int = 15, interval: str = "5m") -> Optional[pd.DataFrame]:
    """Fetch market data for a symbol"""
    if not YFINANCE_AVAILABLE:
        return generate_demo_data(symbol)
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{days}d", interval=interval)
        
        if df.empty:
            return generate_demo_data(symbol)
        
        df.columns = df.columns.str.lower()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in required_cols if c in df.columns]]
        df = df.dropna()
        
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return generate_demo_data(symbol)

def generate_demo_data(symbol: str, days: int = 10) -> pd.DataFrame:
    """Generate demo data when real data unavailable"""
    np.random.seed(hash(symbol) % 2**32)
    
    periods = days * 24 * 12
    price = 100 + np.random.rand() * 400
    
    data = []
    timestamp = datetime.now() - pd.Timedelta(days=days)
    
    # Add some directional bias based on symbol hash
    bias = (hash(symbol) % 3 - 1) * 0.0001
    
    for i in range(periods):
        progress = i / periods
        trend = bias + 0.0001 * np.sin(i / 100)
        noise = np.random.randn() * 0.002
        
        hour = (i * 5 // 60) % 24
        volume_mult = 1.5 if 9 <= hour <= 16 else 0.5
        
        returns = trend + noise
        
        open_price = price
        close_price = price * (1 + returns)
        high_price = max(open_price, close_price) * (1 + abs(np.random.randn()) * 0.001)
        low_price = min(open_price, close_price) * (1 - abs(np.random.randn()) * 0.001)
        volume = int(np.random.exponential(100000) * volume_mult)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        price = close_price
        timestamp += pd.Timedelta(minutes=5)
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

def format_scan_result(symbol: str, result) -> Dict[str, Any]:
    """Format scan result for JSON response"""
    timeframe_data = {}
    
    for tf, analysis in result.timeframe_analyses.items():
        tf_data = {
            "signal": analysis.signal.value,
            "signal_emoji": analysis.signal.emoji,
            "bull_score": round(analysis.bull_score, 1),
            "bear_score": round(analysis.bear_score, 1),
            "confidence": round(analysis.confidence, 1),
            "position": analysis.position_in_value,
            "rsi": {
                "value": round(analysis.rsi.value, 1),
                "zone": analysis.rsi.zone,
                "slope": round(analysis.rsi.slope, 2),
                "divergence": analysis.rsi.divergence
            },
            "flow": {
                "imbalance": round(analysis.flow.flow_imbalance, 3),
                "state": analysis.flow.flow_state,
                "delta_momentum": round(analysis.flow.delta_momentum, 2)
            },
            "volume_profile": {
                "poc": round(analysis.volume_profile.poc, 2),
                "vah": round(analysis.volume_profile.vah, 2),
                "val": round(analysis.volume_profile.val, 2),
                "width": round(analysis.volume_profile.value_width, 2)
            },
            "notes": analysis.notes
        }
        
        # Add VWAP if available
        if analysis.vwap:
            tf_data["vwap"] = {
                "value": round(analysis.vwap.vwap, 2),
                "upper_1sd": round(analysis.vwap.upper_band_1, 2),
                "lower_1sd": round(analysis.vwap.lower_band_1, 2),
                "upper_2sd": round(analysis.vwap.upper_band_2, 2),
                "lower_2sd": round(analysis.vwap.lower_band_2, 2),
                "deviation_pct": round(analysis.vwap.deviation_pct, 2),
                "zone": analysis.vwap.zone
            }
        
        timeframe_data[tf.label] = tf_data
    
    return {
        "symbol": symbol,
        "scan_time": result.scan_time.isoformat() if result.scan_time else datetime.now().isoformat(),
        "signal": result.dominant_signal.value,
        "signal_emoji": result.dominant_signal.emoji,
        "confluence": round(result.confluence_score, 1),
        "actionable": result.actionable,
        "probabilities": {
            "high": round(result.high_scenario_prob * 100, 1),
            "low": round(result.low_scenario_prob * 100, 1),
            "neutral": round(result.neutral_prob * 100, 1)
        },
        "timeframes": timeframe_data,
        "summary": result.summary
    }

# =============================================================================
# API ENDPOINTS
# =============================================================================

# --- Serve Frontend ---
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML frontend"""
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Frontend not found. Please ensure index.html exists.</h1>")

# --- Watchlist CRUD ---
@app.get("/api/watchlist")
async def get_watchlist():
    """Get all watchlist items"""
    items = list(store.watchlist.values())
    return {
        "items": [item.dict() for item in items],
        "total": len(items)
    }

@app.post("/api/watchlist")
async def create_watchlist_item(item: WatchlistCreate):
    """Add a new symbol to watchlist"""
    item_id = item.symbol.upper().replace(" ", "_")
    
    if item_id.lower() in store.watchlist:
        raise HTTPException(status_code=400, detail="Symbol already exists in watchlist")
    
    new_item = WatchlistItem(
        id=item_id.lower(),
        symbol=item.symbol.upper(),
        name=item.name or item.symbol.upper(),
        category=item.category,
        enabled=True,
        created_at=datetime.now().isoformat()
    )
    
    store.watchlist[item_id.lower()] = new_item
    return new_item.dict()

@app.get("/api/watchlist/{item_id}")
async def get_watchlist_item(item_id: str):
    """Get a specific watchlist item"""
    if item_id.lower() not in store.watchlist:
        raise HTTPException(status_code=404, detail="Item not found")
    return store.watchlist[item_id.lower()].dict()

@app.put("/api/watchlist/{item_id}")
async def update_watchlist_item(item_id: str, update: WatchlistUpdate):
    """Update a watchlist item"""
    if item_id.lower() not in store.watchlist:
        raise HTTPException(status_code=404, detail="Item not found")
    
    item = store.watchlist[item_id.lower()]
    
    if update.name is not None:
        item.name = update.name
    if update.category is not None:
        item.category = update.category
    if update.enabled is not None:
        item.enabled = update.enabled
    
    return item.dict()

@app.delete("/api/watchlist/{item_id}")
async def delete_watchlist_item(item_id: str):
    """Remove a symbol from watchlist"""
    if item_id.lower() not in store.watchlist:
        raise HTTPException(status_code=404, detail="Item not found")
    
    deleted = store.watchlist.pop(item_id.lower())
    return {"deleted": deleted.dict()}

# --- Scanning ---
@app.post("/api/scan")
async def run_scan(request: ScanRequest):
    """Run scan on specified symbols"""
    results = []
    
    for symbol in request.symbols:
        df = fetch_market_data(symbol)
        
        if df is not None and len(df) > 100:
            result = scanner.scan(df, symbol=symbol)
            formatted = format_scan_result(symbol, result)
            results.append(formatted)
            store.scan_results[symbol] = formatted
    
    store.last_scan_time = datetime.now()
    
    return {
        "results": results,
        "scan_time": store.last_scan_time.isoformat(),
        "count": len(results)
    }

@app.get("/api/scan/all")
async def scan_all_watchlist():
    """Scan all enabled watchlist symbols"""
    symbols = [
        item.symbol for item in store.watchlist.values() 
        if item.enabled
    ]
    
    results = []
    for symbol in symbols:
        df = fetch_market_data(symbol)
        
        if df is not None and len(df) > 100:
            result = scanner.scan(df, symbol=symbol)
            formatted = format_scan_result(symbol, result)
            results.append(formatted)
            store.scan_results[symbol] = formatted
    
    store.last_scan_time = datetime.now()
    
    # Sort by actionability and signal strength
    results.sort(key=lambda x: (
        x['actionable'],
        x['signal'] in ['LONG_SETUP', 'SHORT_SETUP'],
        x['confluence']
    ), reverse=True)
    
    return {
        "results": results,
        "scan_time": store.last_scan_time.isoformat(),
        "count": len(results),
        "actionable_count": sum(1 for r in results if r['actionable'])
    }

@app.get("/api/scan/{symbol}")
async def scan_single(symbol: str):
    """Scan a single symbol"""
    df = fetch_market_data(symbol.upper())
    
    if df is None or len(df) < 100:
        raise HTTPException(status_code=400, detail=f"Could not fetch data for {symbol}")
    
    result = scanner.scan(df, symbol=symbol.upper())
    formatted = format_scan_result(symbol.upper(), result)
    store.scan_results[symbol.upper()] = formatted
    
    return formatted

@app.get("/api/results")
async def get_cached_results():
    """Get cached scan results"""
    return {
        "results": list(store.scan_results.values()),
        "last_scan": store.last_scan_time.isoformat() if store.last_scan_time else None
    }

# --- Configuration ---
@app.get("/api/config")
async def get_config():
    """Get current scanner configuration"""
    return {
        "scoring": {
            "strong_threshold": store.config.scoring.strong_threshold,
            "moderate_threshold": store.config.scoring.moderate_threshold,
            "min_score_gap": store.config.scoring.min_score_gap,
            "min_confluence_actionable": store.config.scoring.min_confluence_actionable
        },
        "volume_profile": {
            "value_area_pct": store.config.volume_profile.value_area_pct,
            "num_bins": store.config.volume_profile.num_bins
        },
        "rsi": {
            "period": store.config.rsi.period,
            "overbought": store.config.rsi.overbought,
            "oversold": store.config.rsi.oversold
        },
        "flow": {
            "momentum_period": store.config.flow.momentum_period,
            "strong_imbalance": store.config.flow.strong_imbalance
        },
        "trade_management": {
            "stop_atr_mult": store.config.default_stop_atr_mult,
            "target1_atr_mult": store.config.target1_atr_mult,
            "target2_atr_mult": store.config.target2_atr_mult,
            "max_hold_days": store.config.max_hold_days
        }
    }

@app.put("/api/config")
async def update_config(update: ConfigUpdate):
    """Update scanner configuration"""
    if update.strong_threshold is not None:
        store.config.scoring.strong_threshold = update.strong_threshold
    if update.moderate_threshold is not None:
        store.config.scoring.moderate_threshold = update.moderate_threshold
    if update.min_score_gap is not None:
        store.config.scoring.min_score_gap = update.min_score_gap
    if update.min_confluence is not None:
        store.config.scoring.min_confluence_actionable = update.min_confluence
    if update.value_area_pct is not None:
        store.config.volume_profile.value_area_pct = update.value_area_pct
    if update.rsi_period is not None:
        store.config.rsi.period = update.rsi_period
    if update.rsi_overbought is not None:
        store.config.rsi.overbought = update.rsi_overbought
    if update.rsi_oversold is not None:
        store.config.rsi.oversold = update.rsi_oversold
    
    return await get_config()

@app.post("/api/config/preset/{preset_name}")
async def apply_preset(preset_name: str):
    """Apply a configuration preset"""
    presets = {
        "conservative": SwingTradeConfig.conservative_swing,
        "balanced": SwingTradeConfig.balanced_swing,
        "active": SwingTradeConfig.active_swing
    }
    
    if preset_name.lower() not in presets:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {preset_name}")
    
    store.config = presets[preset_name.lower()]()
    return await get_config()

# --- Stats ---
@app.get("/api/stats")
async def get_stats():
    """Get overall statistics"""
    results = list(store.scan_results.values())
    
    if not results:
        return {
            "total_symbols": len(store.watchlist),
            "scanned": 0,
            "actionable": 0,
            "long_setups": 0,
            "short_setups": 0,
            "yellow": 0,
            "neutral": 0,
            "last_scan": None
        }
    
    return {
        "total_symbols": len(store.watchlist),
        "scanned": len(results),
        "actionable": sum(1 for r in results if r['actionable']),
        "long_setups": sum(1 for r in results if r['signal'] == 'LONG_SETUP'),
        "short_setups": sum(1 for r in results if r['signal'] == 'SHORT_SETUP'),
        "yellow": sum(1 for r in results if r['signal'] == 'YELLOW'),
        "neutral": sum(1 for r in results if r['signal'] in ['NEUTRAL', 'NO_DATA']),
        "last_scan": store.last_scan_time.isoformat() if store.last_scan_time else None
    }

# --- Health Check ---
@app.get("/api/health")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "yfinance_available": YFINANCE_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
