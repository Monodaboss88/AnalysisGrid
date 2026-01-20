"""
Entry Scanner API Endpoints
===========================
Exposes Volume Profile entry detection via REST API.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vp_entry_detector import (
    VolumeProfileEntryDetector,
    VolumeProfileLevels,
    PriceBar,
    ProfileType,
    EntrySignal,
    Direction
)
from vp_scanner_integration import VolumeProfileScanner, convert_finnhub_candles

entry_router = APIRouter(prefix="/api/entry-scan", tags=["Entry Scanner"])

# Shared scanner instance
_scanner = None
_finnhub_scanner = None


def set_finnhub_scanner(scanner):
    """Set the Finnhub scanner reference from unified_server"""
    global _finnhub_scanner
    _finnhub_scanner = scanner


def get_scanner() -> VolumeProfileScanner:
    global _scanner
    if _scanner is None:
        _scanner = VolumeProfileScanner()
    return _scanner


def _convert_df_to_bars(df) -> List[PriceBar]:
    """Convert pandas DataFrame to PriceBar list"""
    bars = []
    for idx, row in df.iterrows():
        bars.append(PriceBar(
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row['volume']),
            timestamp=str(idx)
        ))
    return bars


@entry_router.get("/scan/{symbol}")
async def scan_symbol(
    symbol: str,
    resolution: str = Query("60", description="Candle resolution: 5, 15, 30, 60"),
    days: int = Query(5, ge=1, le=30, description="Days of data to analyze")
):
    """
    Scan a single symbol for volume profile entry setups.
    
    Returns detected entries: VAL Touch, POC Reclaim, Failed Breakdown,
    VAH Touch, Breakout Retest, Volume Break, etc.
    """
    if not _finnhub_scanner:
        raise HTTPException(status_code=500, detail="Scanner not initialized")
    
    try:
        # Get candle data
        df = _finnhub_scanner._get_candles(symbol.upper(), resolution, days)
        
        if df is None or len(df) < 10:
            raise HTTPException(status_code=404, detail=f"Insufficient data for {symbol}")
        
        # Convert to PriceBar list
        bars = _convert_df_to_bars(df)
        
        # Calculate average volume
        avg_volume = sum(b.volume for b in bars) / len(bars) if bars else 0
        
        # Run scan
        scanner = get_scanner()
        result = scanner.scan_symbol(symbol.upper(), bars, avg_volume=avg_volume)
        
        # Format response
        signals = []
        for sig in result.signals:
            signals.append({
                "entry_type": sig.entry_type.value,
                "direction": sig.direction.value,
                "entry_price": round(sig.entry_price, 2),
                "stop_price": round(sig.stop_price, 2),
                "target_1": round(sig.target_1, 2),
                "target_2": round(sig.target_2, 2) if sig.target_2 else None,
                "rr_ratio": round(sig.rr_ratio, 2),
                "confidence": round(sig.confidence, 0),
                "notes": sig.notes
            })
        
        return {
            "symbol": symbol.upper(),
            "signals": signals,
            "signal_count": len(signals),
            "levels": {
                "vah": round(result.levels.vah, 2) if result.levels else None,
                "poc": round(result.levels.poc, 2) if result.levels else None,
                "val": round(result.levels.val, 2) if result.levels else None,
                "profile_type": result.levels.profile_type.value if result.levels else None
            },
            "current_price": round(result.current_price, 2),
            "timestamp": result.timestamp
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Entry scan error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@entry_router.post("/batch")
async def batch_scan(
    symbols: List[str],
    resolution: str = Query("60", description="Candle resolution"),
    min_confidence: int = Query(50, ge=0, le=100, description="Minimum confidence filter")
):
    """
    Batch scan multiple symbols for entry setups.
    
    Returns only symbols with detected signals above confidence threshold.
    """
    if not _finnhub_scanner:
        raise HTTPException(status_code=500, detail="Scanner not initialized")
    
    results = []
    scanner = get_scanner()
    
    for symbol in symbols[:30]:  # Limit to 30 symbols
        try:
            df = _finnhub_scanner._get_candles(symbol.upper(), resolution, 5)
            
            if df is None or len(df) < 10:
                continue
            
            bars = _convert_df_to_bars(df)
            avg_volume = sum(b.volume for b in bars) / len(bars) if bars else 0
            
            result = scanner.scan_symbol(symbol.upper(), bars, avg_volume=avg_volume)
            
            # Filter by confidence
            high_conf_signals = [s for s in result.signals if s.confidence >= min_confidence]
            
            if high_conf_signals:
                results.append({
                    "symbol": symbol.upper(),
                    "signals": [{
                        "entry_type": s.entry_type.value,
                        "direction": s.direction.value,
                        "entry_price": round(s.entry_price, 2),
                        "stop_price": round(s.stop_price, 2),
                        "target_1": round(s.target_1, 2),
                        "rr_ratio": round(s.rr_ratio, 2),
                        "confidence": round(s.confidence, 0),
                        "notes": s.notes
                    } for s in high_conf_signals],
                    "levels": {
                        "vah": round(result.levels.vah, 2) if result.levels else None,
                        "poc": round(result.levels.poc, 2) if result.levels else None,
                        "val": round(result.levels.val, 2) if result.levels else None
                    },
                    "current_price": round(result.current_price, 2)
                })
                
        except Exception as e:
            print(f"Batch scan error for {symbol}: {e}")
            continue
    
    # Sort by highest confidence signal
    results.sort(key=lambda x: max(s['confidence'] for s in x['signals']), reverse=True)
    
    return {
        "count": len(results),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


@entry_router.get("/entry-types")
async def get_entry_types():
    """Get list of all detectable entry types with descriptions"""
    return {
        "normal_profile_longs": [
            {"type": "val_touch_rejection", "name": "VAL Touch Rejection", "description": "Price touches VAL with bullish rejection candle (long lower wick)"},
            {"type": "poc_reclaim", "name": "POC Reclaim", "description": "Price dips below POC then closes back above with buffer"},
            {"type": "failed_breakdown", "name": "Failed Breakdown", "description": "Bear trap - broke below VAL but reclaimed (last 4 bars)"}
        ],
        "normal_profile_shorts": [
            {"type": "vah_touch_rejection", "name": "VAH Touch Rejection", "description": "Price touches VAH with bearish rejection candle (long upper wick)"},
            {"type": "poc_rejection", "name": "POC Rejection", "description": "Price rallies to POC from below and gets rejected"},
            {"type": "failed_breakout", "name": "Failed Breakout", "description": "Bull trap - broke above VAH but reversed back inside"}
        ],
        "inverted_profile": [
            {"type": "breakout_retest_long", "name": "Breakout Retest Long", "description": "Broke above VAH, retesting as support"},
            {"type": "breakout_retest_short", "name": "Breakout Retest Short", "description": "Broke below VAL, retesting as resistance"},
            {"type": "volume_break_long", "name": "Volume Break Long", "description": "Break above VAH with 1.5x+ volume surge"},
            {"type": "volume_break_short", "name": "Volume Break Short", "description": "Break below VAL with 1.5x+ volume surge"}
        ]
    }
