"""
Entry Scanner API Endpoints
===========================
Exposes Volume Profile entry detection via REST API.
Includes MTF trend filter for higher probability setups.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime

from .vp_entry_detector import (
    VolumeProfileEntryDetector,
    VolumeProfileLevels,
    PriceBar,
    ProfileType,
    EntrySignal,
    Direction
)
from .vp_scanner_integration import VolumeProfileScanner, convert_finnhub_candles

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


def _get_htf_bias(symbol: str) -> dict:
    """
    Get higher timeframe bias for the symbol.
    Returns dict with 'bias' ('bullish', 'bearish', 'neutral') and scores.
    """
    if not _finnhub_scanner:
        return {"bias": "neutral", "bull_score": 50, "bear_score": 50, "confidence": 0}
    
    try:
        # Analyze on 4HR timeframe for swing context
        result = _finnhub_scanner.analyze(symbol.upper(), "4HR")
        
        if not result:
            return {"bias": "neutral", "bull_score": 50, "bear_score": 50, "confidence": 0}
        
        bull = result.bull_score or 50
        bear = result.bear_score or 50
        score_diff = bull - bear
        
        # Determine bias
        if score_diff >= 15:
            bias = "bullish"
        elif score_diff <= -15:
            bias = "bearish"
        else:
            bias = "neutral"
        
        return {
            "bias": bias,
            "bull_score": bull,
            "bear_score": bear,
            "signal": result.signal,
            "confidence": abs(score_diff)
        }
        
    except Exception as e:
        print(f"HTF bias error for {symbol}: {e}")
        return {"bias": "neutral", "bull_score": 50, "bear_score": 50, "confidence": 0}


def _filter_by_htf(signals: List[EntrySignal], htf_bias: dict, require_alignment: bool = True) -> List[tuple]:
    """
    Filter signals by higher timeframe alignment.
    Returns list of (signal, alignment_bonus) tuples.
    """
    filtered = []
    bias = htf_bias.get("bias", "neutral")
    htf_confidence = htf_bias.get("confidence", 0)
    
    for sig in signals:
        alignment_bonus = 0
        aligned = False
        contrary = False
        
        if sig.direction == Direction.LONG:
            if bias == "bullish":
                aligned = True
                alignment_bonus = min(20, htf_confidence // 2)  # Up to +20 confidence
            elif bias == "bearish":
                contrary = True
                alignment_bonus = -15  # Penalty for fighting HTF trend
        else:  # SHORT
            if bias == "bearish":
                aligned = True
                alignment_bonus = min(20, htf_confidence // 2)
            elif bias == "bullish":
                contrary = True
                alignment_bonus = -15
        
        # If require_alignment is True, skip contrary signals
        if require_alignment and contrary:
            continue
        
        filtered.append((sig, alignment_bonus, aligned))
    
    return filtered


@entry_router.get("/scan/{symbol}")
async def scan_symbol(
    symbol: str,
    resolution: str = Query("60", description="Candle resolution: 5, 15, 30, 60"),
    days: int = Query(5, ge=1, le=30, description="Days of data to analyze"),
    mtf_filter: bool = Query(True, description="Filter signals by 4H trend alignment"),
    require_alignment: bool = Query(False, description="Only show signals aligned with HTF trend")
):
    """
    Scan a single symbol for volume profile entry setups.
    
    Returns detected entries: VAL Touch, POC Reclaim, Failed Breakdown,
    VAH Touch, Breakout Retest, Volume Break, etc.
    
    MTF Filter: When enabled, adds/subtracts confidence based on 4H trend alignment.
    Require Alignment: When True, only returns signals that match HTF direction.
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
        
        # Get HTF bias if MTF filter enabled
        htf_bias = None
        if mtf_filter:
            htf_bias = _get_htf_bias(symbol.upper())
        
        # Format response with MTF filtering
        signals = []
        
        if mtf_filter and htf_bias:
            filtered = _filter_by_htf(result.signals, htf_bias, require_alignment)
            for sig, alignment_bonus, aligned in filtered:
                adjusted_confidence = min(100, max(0, sig.confidence + alignment_bonus))
                signals.append({
                    "entry_type": sig.entry_type.value,
                    "direction": sig.direction.value,
                    "entry_price": round(sig.entry_price, 2),
                    "stop_price": round(sig.stop_price, 2),
                    "target_1": round(sig.target_1, 2),
                    "target_2": round(sig.target_2, 2) if sig.target_2 else None,
                    "rr_ratio": round(sig.rr_ratio, 2),
                    "confidence": round(adjusted_confidence, 0),
                    "base_confidence": round(sig.confidence, 0),
                    "htf_aligned": aligned,
                    "htf_bonus": alignment_bonus,
                    "notes": sig.notes
                })
        else:
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
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
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
            "htf_context": htf_bias,
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
    min_confidence: int = Query(50, ge=0, le=100, description="Minimum confidence filter"),
    mtf_filter: bool = Query(True, description="Filter by 4H trend alignment"),
    aligned_only: bool = Query(True, description="Only return HTF-aligned signals")
):
    """
    Batch scan multiple symbols for entry setups.
    
    Returns only symbols with detected signals above confidence threshold.
    When aligned_only=True, only returns signals matching 4H trend direction.
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
            
            if not result.signals:
                continue
            
            # Get HTF bias
            htf_bias = None
            if mtf_filter:
                htf_bias = _get_htf_bias(symbol.upper())
            
            # Filter and adjust signals
            processed_signals = []
            
            if mtf_filter and htf_bias:
                filtered = _filter_by_htf(result.signals, htf_bias, aligned_only)
                for sig, alignment_bonus, aligned in filtered:
                    adjusted_conf = min(100, max(0, sig.confidence + alignment_bonus))
                    if adjusted_conf >= min_confidence:
                        processed_signals.append({
                            "entry_type": sig.entry_type.value,
                            "direction": sig.direction.value,
                            "entry_price": round(sig.entry_price, 2),
                            "stop_price": round(sig.stop_price, 2),
                            "target_1": round(sig.target_1, 2),
                            "rr_ratio": round(sig.rr_ratio, 2),
                            "confidence": round(adjusted_conf, 0),
                            "htf_aligned": aligned,
                            "notes": sig.notes
                        })
            else:
                for sig in result.signals:
                    if sig.confidence >= min_confidence:
                        processed_signals.append({
                            "entry_type": sig.entry_type.value,
                            "direction": sig.direction.value,
                            "entry_price": round(sig.entry_price, 2),
                            "stop_price": round(sig.stop_price, 2),
                            "target_1": round(sig.target_1, 2),
                            "rr_ratio": round(sig.rr_ratio, 2),
                            "confidence": round(sig.confidence, 0),
                            "notes": sig.notes
                        })
            
            if processed_signals:
                results.append({
                    "symbol": symbol.upper(),
                    "signals": processed_signals,
                    "htf_bias": htf_bias.get("bias") if htf_bias else None,
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
        "filters": {
            "mtf_filter": mtf_filter,
            "aligned_only": aligned_only,
            "min_confidence": min_confidence
        },
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
