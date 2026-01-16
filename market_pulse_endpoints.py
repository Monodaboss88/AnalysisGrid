"""
Market Pulse API Endpoints
===========================
REST API for market pulse analysis
"""

from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional
from datetime import datetime

from market_pulse import MarketPulseAnalyzer

# Create router
pulse_router = APIRouter(prefix="/api/pulse", tags=["Market Pulse"])

# Initialize analyzer
analyzer = MarketPulseAnalyzer()


@pulse_router.get("/status")
async def pulse_status():
    """Get market pulse system status"""
    return {
        "status": "running",
        "database": analyzer.db_path,
        "scanner_ready": analyzer.scanner is not None,
        "timestamp": datetime.now().isoformat()
    }


@pulse_router.post("/scan")
async def scan_market(
    symbols: Optional[List[str]] = None,
    timeframe: str = Query("4hr", description="Timeframe: 1hr, 4hr, 1d")
):
    """Scan multiple symbols and generate market pulse"""
    
    # Default watchlist if none provided
    if not symbols:
        symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
            "JPM", "BAC", "WFC", "GS",
            "XOM", "CVX", "COP",
            "UNH", "JNJ", "PFE", "LLY",
            "HD", "NKE", "SBUX", "MCD"
        ]
    
    pulse = analyzer.scan_watchlist(symbols, timeframe)
    
    return {
        "timestamp": pulse.timestamp,
        "market_bias": pulse.market_bias,
        "total_scanned": pulse.total_scanned,
        "bullish": {
            "count": pulse.bullish_count,
            "percent": round(pulse.bullish_pct, 1)
        },
        "bearish": {
            "count": pulse.bearish_count,
            "percent": round(pulse.bearish_pct, 1)
        },
        "neutral": {
            "count": pulse.neutral_count
        },
        "avg_confidence": round(pulse.avg_confidence, 1),
        "avg_rsi": round(pulse.avg_rsi, 1),
        "sectors": pulse.sector_breakdown,
        "top_bullish": pulse.top_bullish,
        "top_bearish": pulse.top_bearish
    }


@pulse_router.get("/stock/{symbol}")
async def get_stock_pulse(
    symbol: str,
    timeframe: str = Query("4hr", description="Timeframe: 1hr, 4hr, 1d")
):
    """Get pulse data for a single stock"""
    pulse = analyzer.scan_stock(symbol.upper(), timeframe)
    
    if not pulse:
        raise HTTPException(status_code=404, detail=f"Could not scan {symbol}")
    
    return {
        "symbol": pulse.symbol,
        "signal": pulse.signal,
        "confidence": pulse.confidence,
        "bull_score": pulse.bull_score,
        "bear_score": pulse.bear_score,
        "rsi": round(pulse.rsi, 1),
        "vwap_deviation": round(pulse.vwap_deviation, 2),
        "position": pulse.position,
        "current_price": pulse.current_price,
        "levels": {
            "vah": pulse.vah,
            "poc": pulse.poc,
            "val": pulse.val
        },
        "sector": pulse.sector,
        "timestamp": pulse.timestamp
    }


@pulse_router.get("/history")
async def get_pulse_history(
    days: int = Query(7, description="Number of days of history")
):
    """Get historical market pulse data"""
    history = analyzer.get_historical_pulses(days)
    
    return {
        "days": days,
        "count": len(history),
        "pulses": history
    }


@pulse_router.get("/stock/{symbol}/history")
async def get_stock_history(
    symbol: str,
    days: int = Query(30, description="Number of days of history")
):
    """Get historical data for a specific stock"""
    history = analyzer.get_stock_history(symbol.upper(), days)
    
    return {
        "symbol": symbol.upper(),
        "days": days,
        "count": len(history),
        "history": history
    }


@pulse_router.get("/sectors")
async def get_sector_performance(
    days: int = Query(1, description="Number of days to analyze")
):
    """Get sector performance breakdown"""
    sectors = analyzer.get_sector_performance(days)
    
    # Sort by bullish percentage
    sorted_sectors = sorted(
        sectors.items(),
        key=lambda x: x[1].get("bullish_pct", 0),
        reverse=True
    )
    
    return {
        "days": days,
        "sectors": dict(sorted_sectors),
        "strongest": sorted_sectors[0][0] if sorted_sectors else None,
        "weakest": sorted_sectors[-1][0] if sorted_sectors else None
    }


@pulse_router.get("/export")
async def export_data(
    table: str = Query("stock_pulses", description="Table to export: stock_pulses or market_pulses"),
    format: str = Query("json", description="Format: json or csv")
):
    """Export pulse data for external analysis"""
    if table not in ["stock_pulses", "market_pulses"]:
        raise HTTPException(status_code=400, detail="Invalid table name")
    
    data = analyzer.export_data(table, format)
    
    return {
        "table": table,
        "format": format,
        "data": data if format == "json" else None,
        "csv": data if format == "csv" else None
    }


@pulse_router.get("/breadth")
async def get_market_breadth():
    """Get current market breadth from recent scans"""
    # Get last hour of data
    history = analyzer.get_historical_pulses(days=1)
    
    if not history:
        return {
            "status": "no_data",
            "message": "Run a market scan first with POST /api/pulse/scan"
        }
    
    latest = history[0]
    
    # Calculate breadth indicators
    bullish_pct = latest.get("bullish_pct", 0)
    bearish_pct = latest.get("bearish_pct", 0)
    
    # Breadth thrust indicator
    breadth_thrust = bullish_pct - bearish_pct
    
    # Market regime
    if breadth_thrust > 40:
        regime = "STRONG_BULL"
    elif breadth_thrust > 20:
        regime = "BULL"
    elif breadth_thrust > -20:
        regime = "NEUTRAL"
    elif breadth_thrust > -40:
        regime = "BEAR"
    else:
        regime = "STRONG_BEAR"
    
    return {
        "timestamp": latest.get("timestamp"),
        "bullish_pct": round(bullish_pct, 1),
        "bearish_pct": round(bearish_pct, 1),
        "breadth_thrust": round(breadth_thrust, 1),
        "regime": regime,
        "avg_rsi": round(latest.get("avg_rsi", 50), 1),
        "interpretation": _interpret_breadth(breadth_thrust, latest.get("avg_rsi", 50))
    }


def _interpret_breadth(thrust: float, avg_rsi: float) -> str:
    """Generate human-readable interpretation"""
    if thrust > 40 and avg_rsi > 60:
        return "Strong bullish momentum across the market. Consider trend-following longs."
    elif thrust > 20:
        return "Bullish bias. Look for dips to buy in strong sectors."
    elif thrust < -40 and avg_rsi < 40:
        return "Strong bearish momentum. Caution on longs, consider hedging."
    elif thrust < -20:
        return "Bearish bias. Be selective on longs, watch for shorts in weak sectors."
    else:
        return "Mixed market. Focus on individual setups rather than broad direction."
