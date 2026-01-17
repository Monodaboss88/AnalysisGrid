"""
Range Watcher API Endpoints
===========================
Add to unified_server.py:

    from rangewatcher.range_watcher_endpoints import range_router
    app.include_router(range_router, prefix="/api/range")
"""

from datetime import datetime
from typing import List, Optional, Dict
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rangewatcher.range_watcher import RangeWatcher, RangeWatcherResult, generate_demo_data

# Scanner will be set from unified_server
_scanner = None

def set_scanner(scanner):
    """Set the scanner instance from unified_server"""
    global _scanner
    _scanner = scanner

def get_scanner():
    """Get the scanner - returns None if not set"""
    return _scanner


def fetch_data_polygon(symbol: str, days: int = 60):
    """Fetch data using Polygon via FinnhubScanner"""
    scanner = get_scanner()
    if scanner is not None:
        # Get daily candles
        df = scanner._get_candles(symbol, "D", days)
        return df
    return None


def fetch_data(symbol: str, days: int = 60):
    """Fetch OHLCV data - try Polygon first, then yfinance"""
    # Try Polygon first
    df = fetch_data_polygon(symbol, days)
    if df is not None and len(df) >= 30:
        return df
    
    # Fallback to yfinance
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{days}d")
        if len(df) > 0:
            df.columns = [c.lower() for c in df.columns]
            return df
    except:
        pass
    
    return None


# =============================================================================
# MODELS
# =============================================================================

class RangeAnalysisResponse(BaseModel):
    """Response model for range analysis"""
    symbol: str
    current_price: float
    trend_structure: str
    trend_emoji: str
    trend_bias: str
    trend_strength: float
    range_state: str
    
    # Period summaries
    periods: Dict[int, dict]
    
    # Key levels
    resistance_levels: List[dict]
    support_levels: List[dict]
    breakout_watch: Optional[float]
    breakdown_watch: Optional[float]
    
    # Notes
    notes: List[str]
    
    timestamp: str


class MultiSymbolRequest(BaseModel):
    """Request for scanning multiple symbols"""
    symbols: List[str]


# =============================================================================
# ROUTER
# =============================================================================

range_router = APIRouter(tags=["Range Watcher"])

# Global watcher instance
_watcher: Optional[RangeWatcher] = None


def get_watcher() -> RangeWatcher:
    global _watcher
    if _watcher is None:
        _watcher = RangeWatcher()
    return _watcher


# =============================================================================
# ENDPOINTS
# =============================================================================

@range_router.get("/analyze/{symbol}")
async def analyze_symbol(
    symbol: str,
    days: int = Query(60, ge=30, le=180, description="Days of data to analyze")
):
    """
    Analyze range structure for a single symbol.
    
    Returns:
    - Trend structure (HH/HL/LH/LL patterns)
    - Range compression/expansion state
    - Key support/resistance levels
    - Breakout/breakdown watch levels
    """
    watcher = get_watcher()
    
    # Fetch data
    df = fetch_data(symbol.upper(), days=days)
    
    if df is None or len(df) < 30:
        raise HTTPException(
            status_code=404,
            detail=f"Could not fetch sufficient data for {symbol}. Need at least 30 days."
        )
    
    # Run analysis
    result = watcher.analyze(df, symbol=symbol.upper())
    
    # Convert to response format
    return _format_response(result)


@range_router.get("/analyze/{symbol}/period/{period_days}")
async def analyze_single_period(
    symbol: str,
    period_days: int
):
    """
    Get detailed analysis for a specific period (3, 6, 9, 12, 15, or 30 days).
    """
    if period_days not in [3, 6, 9, 12, 15, 30]:
        raise HTTPException(
            status_code=400,
            detail="Period must be one of: 3, 6, 9, 12, 15, 30"
        )
    
    watcher = get_watcher()
    df = fetch_data(symbol.upper(), days=max(60, period_days * 2))
    
    if df is None or len(df) < period_days:
        raise HTTPException(status_code=404, detail=f"Insufficient data for {symbol}")
    
    result = watcher.analyze(df, symbol=symbol.upper())
    period = result.periods.get(period_days)
    
    if period is None:
        raise HTTPException(status_code=404, detail=f"No analysis for {period_days}D period")
    
    return {
        "symbol": symbol.upper(),
        "period_days": period_days,
        "high": period.high,
        "low": period.low,
        "range_size": period.range_size,
        "range_pct": period.range_pct,
        "current_price": period.current_price,
        "position_in_range": period.position_in_range,
        "higher_highs": period.higher_highs,
        "higher_lows": period.higher_lows,
        "lower_highs": period.lower_highs,
        "lower_lows": period.lower_lows,
        "nearest_resistance": period.nearest_resistance,
        "nearest_support": period.nearest_support,
        "swing_highs_count": len(period.swing_highs),
        "swing_lows_count": len(period.swing_lows)
    }


@range_router.post("/scan")
async def scan_multiple(request: MultiSymbolRequest):
    """
    Scan multiple symbols and return summary.
    
    Good for watchlist screening.
    """
    watcher = get_watcher()
    results = []
    
    for symbol in request.symbols[:20]:  # Limit to 20
        try:
            df = fetch_data(symbol.upper(), days=60)
            
            if df is not None and len(df) >= 30:
                result = watcher.analyze(df, symbol=symbol.upper())
                
                results.append({
                    "symbol": symbol.upper(),
                    "trend": result.trend_structure.value,
                    "trend_emoji": result.trend_structure.emoji,
                    "bias": result.trend_structure.bias,
                    "strength": result.trend_strength,
                    "range_state": result.range_state.value,
                    "price": result.current_price,
                    "breakout_watch": result.breakout_watch,
                    "breakdown_watch": result.breakdown_watch
                })
        except Exception as e:
            results.append({
                "symbol": symbol.upper(),
                "error": str(e)
            })
    
    # Sort by trend strength (strongest trends first)
    results.sort(key=lambda x: abs(x.get("strength", 0)), reverse=True)
    
    return {
        "count": len(results),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


@range_router.get("/levels/{symbol}")
async def get_key_levels(symbol: str):
    """
    Get just the key support/resistance levels for a symbol.
    
    Useful for quick level reference.
    """
    watcher = get_watcher()
    df = fetch_data(symbol.upper(), days=60)
    
    if df is None or len(df) < 30:
        raise HTTPException(status_code=404, detail=f"Insufficient data for {symbol}")
    
    result = watcher.analyze(df, symbol=symbol.upper())
    
    return {
        "symbol": symbol.upper(),
        "current_price": result.current_price,
        "resistance": [
            {"price": p, "description": d, "distance_pct": ((p - result.current_price) / result.current_price) * 100}
            for p, d in result.major_resistance_levels
        ],
        "support": [
            {"price": p, "description": d, "distance_pct": ((result.current_price - p) / result.current_price) * 100}
            for p, d in result.major_support_levels
        ],
        "breakout_watch": result.breakout_watch,
        "breakdown_watch": result.breakdown_watch
    }


@range_router.get("/compression-scan")
async def scan_for_compression(
    symbols: str = Query(..., description="Comma-separated symbols")
):
    """
    Scan symbols for range compression (potential breakout setups).
    
    Returns symbols sorted by compression level.
    """
    watcher = get_watcher()
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    compressed = []
    
    for symbol in symbol_list[:30]:
        try:
            df = fetch_data(symbol, days=60)
            
            if df is not None and len(df) >= 30:
                result = watcher.analyze(df, symbol=symbol)
                
                # Calculate compression ratio
                p6 = result.periods.get(6)
                p30 = result.periods.get(30)
                
                if p6 and p30 and p30.range_pct > 0:
                    compression = p6.range_pct / p30.range_pct
                    
                    compressed.append({
                        "symbol": symbol,
                        "compression_ratio": compression,
                        "range_state": result.range_state.value,
                        "6d_range_pct": p6.range_pct,
                        "30d_range_pct": p30.range_pct,
                        "trend": result.trend_structure.value,
                        "bias": result.trend_structure.bias,
                        "breakout_watch": result.breakout_watch,
                        "breakdown_watch": result.breakdown_watch
                    })
        except:
            continue
    
    # Sort by compression (most compressed first)
    compressed.sort(key=lambda x: x["compression_ratio"])
    
    return {
        "count": len(compressed),
        "most_compressed": compressed[:10],
        "timestamp": datetime.now().isoformat()
    }


@range_router.get("/demo")
async def demo_analysis():
    """
    Run analysis on demo data for testing.
    """
    watcher = get_watcher()
    df = generate_demo_data(days=60)
    
    result = watcher.analyze(df, symbol="DEMO")
    
    return _format_response(result)


# =============================================================================
# HELPERS
# =============================================================================

def _to_native(val):
    """Convert numpy types to native Python types"""
    import numpy as np
    if isinstance(val, (np.integer, np.floating)):
        return float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    return val


def _format_response(result: RangeWatcherResult) -> dict:
    """Format RangeWatcherResult for API response"""
    
    periods_dict = {}
    for period, analysis in result.periods.items():
        periods_dict[period] = {
            "high": _to_native(analysis.high),
            "low": _to_native(analysis.low),
            "range_pct": _to_native(analysis.range_pct),
            "position_in_range": _to_native(analysis.position_in_range),
            "higher_highs": _to_native(analysis.higher_highs),
            "higher_lows": _to_native(analysis.higher_lows),
            "lower_highs": _to_native(analysis.lower_highs),
            "lower_lows": _to_native(analysis.lower_lows),
            "structure": _get_structure_string(analysis)
        }
    
    return {
        "symbol": result.symbol,
        "current_price": _to_native(result.current_price),
        "trend_structure": result.trend_structure.value,
        "trend_emoji": result.trend_structure.emoji,
        "trend_bias": result.trend_structure.bias,
        "trend_strength": _to_native(result.trend_strength),
        "range_state": result.range_state.value,
        "periods": periods_dict,
        "resistance_levels": [
            {"price": _to_native(p), "description": d}
            for p, d in result.major_resistance_levels
        ],
        "support_levels": [
            {"price": _to_native(p), "description": d}
            for p, d in result.major_support_levels
        ],
        "breakout_watch": _to_native(result.breakout_watch) if result.breakout_watch else None,
        "breakdown_watch": _to_native(result.breakdown_watch) if result.breakdown_watch else None,
        "notes": result.notes,
        "timestamp": result.analysis_time.isoformat()
    }


def _get_structure_string(analysis) -> str:
    """Get structure string for a period"""
    if analysis.higher_highs and analysis.higher_lows:
        return "HH+HL"
    elif analysis.lower_highs and analysis.lower_lows:
        return "LH+LL"
    elif analysis.higher_highs:
        return "HH"
    elif analysis.higher_lows:
        return "HL"
    elif analysis.lower_highs:
        return "LH"
    elif analysis.lower_lows:
        return "LL"
    return "NEUTRAL"
