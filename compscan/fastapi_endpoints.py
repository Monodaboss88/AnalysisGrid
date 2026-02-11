"""
Compression Reversal FastAPI Endpoints
======================================
FastAPI endpoints for the compression reversal scanner.

Author: Rob's Trading Systems
Version: 1.0.0
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compscan.compression_reversal import (
    CompressionReversalScanner,
    CompressionReversalSetup,
    CompressionReversalSetupShort,
    SetupQuality,
    format_setup_alert,
    quick_scan
)

# Create Router
compression_router = APIRouter(prefix="/api/compression", tags=["Compression Reversal"])

# Global scanner instance
_scanner: Optional[CompressionReversalScanner] = None


def get_scanner() -> CompressionReversalScanner:
    """Get or create scanner instance"""
    global _scanner
    if _scanner is None:
        _scanner = CompressionReversalScanner()
    return _scanner


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ScanRequest(BaseModel):
    """Single symbol scan request"""
    symbol: str
    days: int = 30
    interval: str = "1h"


class WatchlistScanRequest(BaseModel):
    """Watchlist scan request"""
    symbols: List[str]
    min_quality: str = "B"  # "A+", "A", "B", "C"
    days: int = 30
    interval: str = "1h"


class ConfigureRequest(BaseModel):
    """Scanner configuration"""
    val_proximity_pct: float = 0.5
    rsi_target: float = 37
    rsi_tolerance: float = 5
    stop_loss_pct: float = 12.5
    target_move_pct: float = 1.5
    delta: float = 0.65
    min_dte: int = 21


# =============================================================================
# DATA FETCHING HELPER
# =============================================================================

async def fetch_candles(symbol: str, days: int = 30, interval: str = "1h"):
    """Fetch candle data for symbol"""
    import pandas as pd
    
    # Try Finnhub first
    try:
        from finnhub_scanner_v2 import get_finnhub_scanner
        scanner = get_finnhub_scanner()
        df = scanner._get_candles(symbol.upper(), "60" if interval == "1h" else "D", days)
        if df is not None and len(df) >= 50:
            return df
    except Exception as e:
        print(f"Finnhub fetch error for {symbol}: {e}")
    
    # Fallback to yfinance
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol.upper())
        df = ticker.history(period=f"{days}d", interval=interval)
        df.columns = df.columns.str.lower()
        if len(df) >= 50:
            return df
    except Exception as e:
        print(f"yfinance fetch error for {symbol}: {e}")
    
    return None


# =============================================================================
# ENDPOINTS
# =============================================================================

@compression_router.post("/scan")
async def scan_symbol(request: ScanRequest):
    """
    Scan a single symbol for compression reversal setup
    
    Returns complete setup analysis including:
    - Profile shape (normal = football)
    - Compression level
    - RSI zone
    - Reversal candle
    - Setup score and quality grade
    - Entry/stop/target levels
    - Options parameters (delta, DTE, stops)
    """
    try:
        symbol = request.symbol.upper()
        
        # Fetch data
        df = await fetch_candles(symbol, request.days, request.interval)
        
        if df is None or len(df) < 50:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for {symbol}"
            )
        
        # Run scan
        scanner = get_scanner()
        setup = scanner.scan(df, symbol=symbol)
        
        return {
            "success": True,
            "setup": setup.to_dict(),
            "alert": format_setup_alert(setup)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@compression_router.post("/scan-watchlist")
async def scan_watchlist(request: WatchlistScanRequest):
    """
    Scan multiple symbols for compression reversal setups
    
    Returns list of tradeable setups sorted by score
    """
    try:
        if not request.symbols:
            raise HTTPException(status_code=400, detail="Symbols list required")
        
        # Parse min quality
        quality_map = {
            'A+': SetupQuality.A_PLUS,
            'A': SetupQuality.A,
            'B': SetupQuality.B,
            'C': SetupQuality.C
        }
        min_quality = quality_map.get(request.min_quality, SetupQuality.B)
        
        scanner = get_scanner()
        setups = []
        errors = []
        
        for symbol in request.symbols:
            try:
                symbol = symbol.upper()
                df = await fetch_candles(symbol, request.days, request.interval)
                
                if df is None or len(df) < 50:
                    errors.append(f"{symbol}: Insufficient data")
                    continue
                
                setup = scanner.scan(df, symbol=symbol)
                
                # Filter by quality
                if setup.setup_quality.tradeable:
                    quality_order = [SetupQuality.NO_SETUP, SetupQuality.C, SetupQuality.B, SetupQuality.A, SetupQuality.A_PLUS]
                    if quality_order.index(setup.setup_quality) >= quality_order.index(min_quality):
                        setups.append(setup.to_dict())
            
            except Exception as e:
                errors.append(f"{symbol}: {str(e)}")
        
        # Sort by score
        setups.sort(key=lambda x: x['setup_score'], reverse=True)
        
        return {
            "success": True,
            "setups": setups,
            "scanned": len(request.symbols),
            "found": len(setups),
            "errors": errors if errors else None
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@compression_router.get("/criteria")
async def get_criteria():
    """
    Get the setup criteria and parameters
    
    Returns current scanner configuration and quality level definitions
    """
    scanner = get_scanner()
    
    return {
        "success": True,
        "criteria": {
            "profile_shape": "NORMAL (football) - indicates fair value, good for mean reversion",
            "compression": "EXTREME, HIGH, or MODERATE - price is squeezed, move imminent",
            "val_proximity_pct": scanner.val_proximity_pct,
            "rsi_target": scanner.rsi_target,
            "rsi_tolerance": scanner.rsi_tolerance,
            "reversal_candle": "Hammer, Bullish Engulfing, or Doji - confirms buyers"
        },
        "options_params": {
            "direction": "CALL",
            "delta": scanner.delta,
            "min_dte": scanner.min_dte,
            "stop_loss_pct": scanner.stop_loss_pct,
            "target_move_pct": scanner.target_move_pct,
            "target_rsi": 72
        },
        "quality_levels": {
            "A+": "90+ score - textbook setup, all criteria met perfectly",
            "A": "80-89 score - strong setup, high probability",
            "B": "70-79 score - good setup, tradeable",
            "C": "60-69 score - marginal, needs confirmation",
            "NO": "< 60 score - criteria not met, no trade"
        }
    }


@compression_router.post("/configure")
async def configure_scanner(config: ConfigureRequest):
    """
    Update scanner parameters
    
    Allows customizing:
    - VAL proximity threshold
    - RSI target and tolerance
    - Options parameters (delta, DTE, stops, targets)
    """
    global _scanner
    
    try:
        # Create new scanner with updated params
        _scanner = CompressionReversalScanner(
            val_proximity_pct=config.val_proximity_pct,
            rsi_target=config.rsi_target,
            rsi_tolerance=config.rsi_tolerance,
            stop_loss_pct=config.stop_loss_pct,
            target_move_pct=config.target_move_pct,
            delta=config.delta,
            min_dte=config.min_dte
        )
        
        return {
            "success": True,
            "config": {
                "val_proximity_pct": _scanner.val_proximity_pct,
                "rsi_target": _scanner.rsi_target,
                "rsi_tolerance": _scanner.rsi_tolerance,
                "stop_loss_pct": _scanner.stop_loss_pct,
                "target_move_pct": _scanner.target_move_pct,
                "delta": _scanner.delta,
                "min_dte": _scanner.min_dte
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@compression_router.get("/quick/{symbol}")
async def quick_scan_symbol(symbol: str):
    """
    Quick scan a symbol - returns simplified result
    
    Faster than full scan, good for screening
    """
    try:
        symbol = symbol.upper()
        df = await fetch_candles(symbol, days=30, interval="1h")
        
        if df is None or len(df) < 50:
            return {
                "symbol": symbol,
                "tradeable": False,
                "error": "Insufficient data"
            }
        
        scanner = get_scanner()
        setup = scanner.scan(df, symbol=symbol)
        
        return {
            "symbol": symbol,
            "tradeable": setup.setup_quality.tradeable,
            "quality": setup.setup_quality.value,
            "score": setup.setup_score,
            "compression": setup.compression.compression_level.value,
            "rsi": round(setup.rsi.current_rsi, 1),
            "at_val": setup.at_val,
            "reversal_candle": setup.reversal_candle.detected,
            "price": setup.current_price,
            "val": setup.profile.val,
            "poc": setup.profile.poc
        }
    
    except Exception as e:
        return {
            "symbol": symbol,
            "tradeable": False,
            "error": str(e)
        }


# =============================================================================
# SHORT SCAN ENDPOINTS (VAH approach - PUT setups)
# =============================================================================

@compression_router.post("/scan-short")
async def scan_symbol_short(request: ScanRequest):
    """
    Scan a single symbol for compression reversal SHORT setup (VAH approach)
    
    Returns complete SHORT setup analysis including:
    - Profile shape (normal = football)
    - Compression level
    - RSI overbought zone (60-65, target 63)
    - Bearish reversal candle
    - Setup score and quality grade
    - Entry/stop/target levels (SHORT)
    - Options parameters (PUT, delta, DTE, stops)
    """
    try:
        symbol = request.symbol.upper()
        
        df = await fetch_candles(symbol, request.days, request.interval)
        
        if df is None or len(df) < 50:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for {symbol}"
            )
        
        scanner = get_scanner()
        setup = scanner.scan_short(df, symbol=symbol)
        
        return {
            "success": True,
            "direction": "SHORT",
            "setup": setup.to_dict(),
            "notes": setup.notes
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@compression_router.post("/scan-watchlist-short")
async def scan_watchlist_short(request: WatchlistScanRequest):
    """
    Scan multiple symbols for compression reversal SHORT setups (VAH approach)
    
    Returns list of tradeable SHORT setups sorted by score
    """
    try:
        if not request.symbols:
            raise HTTPException(status_code=400, detail="Symbols list required")
        
        quality_map = {
            'A+': SetupQuality.A_PLUS,
            'A': SetupQuality.A,
            'B': SetupQuality.B,
            'C': SetupQuality.C
        }
        min_quality = quality_map.get(request.min_quality, SetupQuality.B)
        
        scanner = get_scanner()
        setups = []
        errors = []
        
        for symbol in request.symbols:
            try:
                symbol = symbol.upper()
                df = await fetch_candles(symbol, request.days, request.interval)
                
                if df is None or len(df) < 50:
                    errors.append(f"{symbol}: Insufficient data")
                    continue
                
                setup = scanner.scan_short(df, symbol=symbol)
                
                if setup.setup_quality.tradeable:
                    quality_order = [SetupQuality.NO_SETUP, SetupQuality.C, SetupQuality.B, SetupQuality.A, SetupQuality.A_PLUS]
                    if quality_order.index(setup.setup_quality) >= quality_order.index(min_quality):
                        setups.append(setup.to_dict())
            
            except Exception as e:
                errors.append(f"{symbol}: {str(e)}")
        
        setups.sort(key=lambda x: x['setup_score'], reverse=True)
        
        return {
            "success": True,
            "direction": "SHORT",
            "setups": setups,
            "scanned": len(request.symbols),
            "found": len(setups),
            "errors": errors if errors else None
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@compression_router.get("/quick-short/{symbol}")
async def quick_scan_symbol_short(symbol: str):
    """
    Quick SHORT scan a symbol - returns simplified result
    
    Faster than full scan, good for screening for puts
    """
    try:
        symbol = symbol.upper()
        df = await fetch_candles(symbol, days=30, interval="1h")
        
        if df is None or len(df) < 50:
            return {
                "symbol": symbol,
                "direction": "SHORT",
                "tradeable": False,
                "error": "Insufficient data"
            }
        
        scanner = get_scanner()
        setup = scanner.scan_short(df, symbol=symbol)
        
        return {
            "symbol": symbol,
            "direction": "SHORT",
            "tradeable": setup.setup_quality.tradeable,
            "quality": setup.setup_quality.value,
            "score": setup.setup_score,
            "compression": setup.compression.compression_level.value,
            "rsi": round(setup.rsi.current_rsi, 1),
            "at_vah": setup.at_vah,
            "reversal_candle": setup.reversal_candle.detected,
            "price": setup.current_price,
            "vah": setup.profile.vah,
            "poc": setup.profile.poc
        }
    
    except Exception as e:
        return {
            "symbol": symbol,
            "direction": "SHORT",
            "tradeable": False,
            "error": str(e)
        }
