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

# Try to import RangeContext for weekly structure
try:
    from chart_input_analyzer import RangeContext
except ImportError:
    RangeContext = None

# Scanner will be set from unified_server
_scanner = None

def set_scanner(scanner):
    """Set the scanner instance from unified_server"""
    global _scanner
    _scanner = scanner

def get_scanner():
    """Get the scanner - returns None if not set"""
    return _scanner


def fetch_weekly_structure(symbol: str) -> dict:
    """Fetch weekly HH/HL/LH/LL structure using scanner's calculate_range_structure"""
    import os
    print(f"📊 fetch_weekly_structure called for {symbol}")
    
    scanner = get_scanner()
    print(f"📊 Scanner from get_scanner(): {scanner is not None}")
    
    # Fallback: create scanner if not set (happens on fresh deploy)
    if scanner is None:
        try:
            from finnhub_scanner import FinnhubScanner
            fallback_key = os.environ.get("FINNHUB_API_KEY", "dummy_for_polygon")
            scanner = FinnhubScanner(fallback_key)
            set_scanner(scanner)  # Cache it for next time
            print(f"📊 Created fallback scanner for weekly structure")
        except Exception as e:
            print(f"❌ Could not create fallback scanner: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    if scanner is None:
        print(f"❌ Scanner still None after fallback attempt")
        return None
    
    try:
        # Get weekly candles (at least 8 weeks for structure)
        print(f"📊 Fetching weekly candles for {symbol}...")
        weekly_df = scanner._get_candles(symbol, "W", 52)  # 1 year of weekly data
        print(f"📊 Weekly candles: {len(weekly_df) if weekly_df is not None else 'None'}")
        
        if weekly_df is None or len(weekly_df) < 4:
            print(f"❌ Weekly structure: insufficient weekly data for {symbol}")
            return None
        
        # Get daily candles for proximity analysis
        daily_df = scanner._get_candles(symbol, "D", 60)
        print(f"📊 Daily candles: {len(daily_df) if daily_df is not None else 'None'}")
        
        if daily_df is None or len(daily_df) < 5:
            print(f"❌ Insufficient daily data for {symbol}")
            return None
        
        # Get current price
        current_price = daily_df['close'].iloc[-1] if len(daily_df) > 0 else None
        
        # Use scanner's TechnicalCalculator.calculate_range_structure (static method)
        if hasattr(scanner, 'calc') and hasattr(scanner.calc, 'calculate_range_structure'):
            print(f"📊 Calling calculate_range_structure...")
            range_ctx = scanner.calc.calculate_range_structure(weekly_df, daily_df, current_price)
            print(f"📊 Got RangeContext: trend={range_ctx.trend}, signal={range_ctx.weekly_close_signal}")
            
            # Convert RangeContext to dict - ensure all values are native Python types (not numpy)
            result = {
                "trend": str(range_ctx.trend),
                "range_state": str(range_ctx.range_state),
                "compression_ratio": float(range_ctx.compression_ratio),
                "hh_count": int(range_ctx.hh_count),
                "hl_count": int(range_ctx.hl_count),
                "lh_count": int(range_ctx.lh_count),
                "ll_count": int(range_ctx.ll_count),
                "total_periods": int(range_ctx.total_periods),
                "near_support": bool(range_ctx.near_support),  # Convert numpy.bool_ to Python bool
                "near_resistance": bool(range_ctx.near_resistance),  # Convert numpy.bool_ to Python bool
                "breakout_watch": float(range_ctx.breakout_watch) if range_ctx.breakout_watch else None,
                "breakdown_watch": float(range_ctx.breakdown_watch) if range_ctx.breakdown_watch else None,
                "weekly_structure_string": _weekly_structure_string(range_ctx),
                # Weekly close analysis
                "weekly_close_position": float(range_ctx.weekly_close_position),
                "weekly_close_signal": str(range_ctx.weekly_close_signal),
                "last_week_structure": str(range_ctx.last_week_structure)
            }
            print(f"✅ Weekly structure complete for {symbol}")
            return result
        else:
            print(f"❌ Scanner missing calc or calculate_range_structure method")
    except Exception as e:
        print(f"❌ Weekly structure error for {symbol}: {e}")
        import traceback
        traceback.print_exc()
    
    return None


def _weekly_structure_string(ctx) -> str:
    """Generate descriptive string for weekly structure"""
    parts = []
    if ctx.hh_count > 0:
        parts.append(f"{ctx.hh_count}HH")
    if ctx.hl_count > 0:
        parts.append(f"{ctx.hl_count}HL")
    if ctx.lh_count > 0:
        parts.append(f"{ctx.lh_count}LH")
    if ctx.ll_count > 0:
        parts.append(f"{ctx.ll_count}LL")
    
    if not parts:
        return "NEUTRAL"
    
    return " ".join(parts)


def fetch_data_polygon(symbol: str, days: int = 60):
    """Fetch data using Polygon via FinnhubScanner"""
    scanner = get_scanner()
    if scanner is not None:
        # Get daily candles
        df = scanner._get_candles(symbol, "D", days)
        return df
    return None


def fetch_realtime_price(symbol: str) -> float:
    """Fetch the latest real-time price for a symbol"""
    scanner = get_scanner()
    
    # Method 1: Use scanner's get_quote (already handles Polygon -> Alpaca -> fallback)
    if scanner is not None and hasattr(scanner, 'get_quote'):
        try:
            quote = scanner.get_quote(symbol.upper())
            if quote and quote.get('current'):
                return float(quote['current'])
        except Exception as e:
            print(f"get_quote failed for {symbol}: {e}")
    
    # Method 2: Try streaming cache (if WebSocket is running)
    if scanner is not None and hasattr(scanner, 'streaming_cache'):
        cached = scanner.streaming_cache.get(symbol.upper())
        if cached and 'price' in cached:
            return cached['price']
    
    # Method 3: Fallback to yfinance (near real-time during market hours)
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        
        # Try fast_info.lastPrice (most reliable for real-time)
        try:
            fi = ticker.fast_info
            if hasattr(fi, 'lastPrice') and fi.lastPrice:
                return float(fi.lastPrice)
        except:
            pass
        
        # Try info dict
        try:
            info = ticker.info
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            if price:
                return float(price)
        except:
            pass
            
    except:
        pass
    
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
    - Weekly macro structure (HH/HL/LH/LL from weekly candles)
    """
    watcher = get_watcher()
    
    # Fetch data
    df = fetch_data(symbol.upper(), days=days)
    
    if df is None or len(df) < 30:
        raise HTTPException(
            status_code=404,
            detail=f"Could not fetch sufficient data for {symbol}. Need at least 30 days."
        )
    
    # Get real-time price to override the potentially delayed close price
    realtime_price = fetch_realtime_price(symbol.upper())
    if realtime_price:
        # Update the last row's close with real-time price for accurate current_price
        df.iloc[-1, df.columns.get_loc('close')] = realtime_price
    
    # Run analysis
    result = watcher.analyze(df, symbol=symbol.upper())
    
    # Get weekly macro structure
    weekly_structure = fetch_weekly_structure(symbol.upper())
    
    # Convert to response format
    return _format_response(result, weekly_structure)


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
    
    # Get real-time price
    realtime_price = fetch_realtime_price(symbol.upper())
    if realtime_price:
        df.iloc[-1, df.columns.get_loc('close')] = realtime_price
    
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
                # Get real-time price
                realtime_price = fetch_realtime_price(symbol.upper())
                if realtime_price:
                    df.iloc[-1, df.columns.get_loc('close')] = realtime_price
                
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
    
    # Get real-time price
    realtime_price = fetch_realtime_price(symbol.upper())
    if realtime_price:
        df.iloc[-1, df.columns.get_loc('close')] = realtime_price
    
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


# OpenAI client reference - will be set from unified_server
_openai_client = None

def set_openai_client(client):
    """Set OpenAI client from unified_server"""
    global _openai_client
    _openai_client = client

def get_openai_client():
    """Get OpenAI client"""
    return _openai_client


@range_router.get("/analyze/{symbol}/ai")
async def analyze_with_ai(
    symbol: str,
    days: int = Query(60, ge=30, le=180, description="Days of data to analyze")
):
    """
    Analyze range structure with AI-powered technical context.
    
    Returns range analysis + AI observations about:
    - Support/Resistance significance
    - Range compression/expansion context
    - Structure patterns and implications
    - Key levels to watch
    - Volume Profile levels (VAH/POC/VAL/VWAP)
    - MTF confluence signals
    - Extension predictor insights
    
    NO trade advice - just technical observations.
    """
    try:
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
        
        # Get weekly macro structure
        weekly_structure = fetch_weekly_structure(symbol.upper())
        
        response = _format_response(result, weekly_structure)
        
        # Get additional context from scanner for more comprehensive AI analysis
        extra_context = {}
        scanner = get_scanner()
        if scanner:
            try:
                # Get single-timeframe analysis result (same as Quick Analyze)
                # This gives us accurate VP levels and bull/bear scores
                single_tf_result = scanner.analyze(symbol.upper(), "1HR")
                if single_tf_result:
                    extra_context['signal'] = single_tf_result.signal
                    extra_context['bull_score'] = single_tf_result.bull_score
                    extra_context['bear_score'] = single_tf_result.bear_score
                    extra_context['confidence'] = single_tf_result.confidence
                    extra_context['position'] = single_tf_result.position
                    extra_context['vwap_zone'] = single_tf_result.vwap_zone
                    extra_context['rsi_zone'] = single_tf_result.rsi_zone
                
                # Get Volume Profile levels (use same 30 bar window as single-TF for consistency)
                df_hourly = scanner._get_candles(symbol.upper(), "60", 7)  # ~30 bars like single-TF
                if df_hourly is not None and len(df_hourly) >= 5:
                    # Trim to last 30 bars for VP consistency
                    if len(df_hourly) > 30:
                        df_hourly = df_hourly.tail(30)
                    
                    poc, vah, val = scanner.calc.calculate_volume_profile(df_hourly)
                    vwap = scanner.calc.calculate_vwap(df_hourly)
                    rsi = scanner.calc.calculate_rsi(df_hourly)
                    rvol = scanner.calc.calculate_relative_volume(df_hourly)
                    volume_trend = scanner.calc.calculate_volume_trend(df_hourly)
                    
                    extra_context['vah'] = vah
                    extra_context['poc'] = poc
                    extra_context['val'] = val
                    extra_context['vwap'] = vwap
                    extra_context['rsi'] = rsi
                    extra_context['rvol'] = rvol
                    extra_context['volume_trend'] = volume_trend
                    
                    # Determine price position relative to VP levels
                    current_price = response.get('current_price', 0)
                    if current_price > vah:
                        extra_context['vp_position'] = 'ABOVE_VAH (Bullish - Extended)'
                    elif current_price > poc:
                        extra_context['vp_position'] = 'ABOVE_POC (Bullish Bias)'
                    elif current_price > val:
                        extra_context['vp_position'] = 'BELOW_POC (Bearish Bias)'
                    else:
                        extra_context['vp_position'] = 'BELOW_VAL (Bearish - Extended)'
                
                # Get MTF signals
                mtf_result = scanner.analyze_mtf(symbol.upper())
                if mtf_result:
                    extra_context['mtf_dominant'] = mtf_result.dominant_signal
                    extra_context['mtf_confluence'] = mtf_result.confluence_pct
                    extra_context['high_prob'] = mtf_result.high_prob
                    extra_context['low_prob'] = mtf_result.low_prob
                    extra_context['weighted_bull'] = mtf_result.weighted_bull
                    extra_context['weighted_bear'] = mtf_result.weighted_bear
                    
                    # Individual timeframe signals
                    tf_signals = {}
                    for tf, tf_data in mtf_result.timeframe_results.items():
                        tf_signals[tf] = tf_data.signal
                    extra_context['tf_signals'] = tf_signals
                
                # Calculate Fibonacci retracement levels from 15D range
                periods = response.get('periods', {})
                p15 = periods.get('15', {})
                swing_high = p15.get('high', 0)
                swing_low = p15.get('low', 0)
                
                if swing_high and swing_low and swing_high > swing_low:
                    fib_range = swing_high - swing_low
                    fib_236 = swing_low + (fib_range * 0.236)
                    fib_382 = swing_low + (fib_range * 0.382)
                    fib_500 = swing_low + (fib_range * 0.500)
                    fib_618 = swing_low + (fib_range * 0.618)
                    fib_786 = swing_low + (fib_range * 0.786)
                    
                    extra_context['fib_swing_high'] = swing_high
                    extra_context['fib_swing_low'] = swing_low
                    extra_context['fib_236'] = fib_236
                    extra_context['fib_382'] = fib_382
                    extra_context['fib_500'] = fib_500
                    extra_context['fib_618'] = fib_618
                    extra_context['fib_786'] = fib_786
                    
                    # Check for confluence between VP levels and Fib levels
                    vah = extra_context.get('vah', 0)
                    poc = extra_context.get('poc', 0)
                    val = extra_context.get('val', 0)
                    current_price = response.get('current_price', 0)
                    
                    fib_confluences = []
                    for vp_level, vp_name in [(vah, 'VAH'), (poc, 'POC'), (val, 'VAL')]:
                        if vp_level:
                            for fib_level, fib_name in [(fib_236, '23.6%'), (fib_382, '38.2%'), (fib_500, '50%'), (fib_618, '61.8%'), (fib_786, '78.6%')]:
                                diff_pct = abs(vp_level - fib_level) / vp_level * 100
                                if diff_pct < 1.5:
                                    fib_confluences.append(f"STRONG: {vp_name} (${vp_level:.2f}) = Fib {fib_name} (${fib_level:.2f})")
                                elif diff_pct < 3.0:
                                    fib_confluences.append(f"CLOSE: {vp_name} (${vp_level:.2f}) ~ Fib {fib_name} (${fib_level:.2f})")
                    
                    extra_context['fib_confluences'] = fib_confluences
                    extra_context['has_fib_confluence'] = len([c for c in fib_confluences if 'STRONG' in c]) > 0
                    
                    # Determine price position relative to Fib
                    if current_price > fib_786:
                        extra_context['fib_position'] = 'Above 78.6% (strong recovery)'
                    elif current_price > fib_618:
                        extra_context['fib_position'] = 'Between 61.8% and 78.6%'
                    elif current_price > fib_500:
                        extra_context['fib_position'] = 'Between 50% and 61.8% (KEY ZONE)'
                    elif current_price > fib_382:
                        extra_context['fib_position'] = 'Between 38.2% and 50%'
                    elif current_price > fib_236:
                        extra_context['fib_position'] = 'Between 23.6% and 38.2%'
                    else:
                        extra_context['fib_position'] = 'Below 23.6% (deep retracement)'
                    
            except Exception as e:
                print(f"⚠️ Extra context fetch error: {e}")
                import traceback
                traceback.print_exc()
        
        # Generate AI context with all available data
        ai_context = await _generate_range_ai_context(response, extra_context)
        response["ai_context"] = ai_context
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Range AI analysis error for {symbol}: {e}")
        print(f"Traceback:\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"Range analysis error: {str(e)}")


async def _generate_range_ai_context(range_data: dict, extra_context: dict = None) -> str:
    """Generate AI-powered technical context for range analysis"""
    client = get_openai_client()
    
    if client is None:
        return "AI analysis unavailable - OpenAI not configured"
    
    extra_context = extra_context or {}
    
    try:
        symbol = range_data.get("symbol", "Unknown")
        current_price = range_data.get("current_price", 0)
        trend = range_data.get("trend_structure", "")
        trend_bias = range_data.get("trend_bias", "")
        trend_strength = range_data.get("trend_strength", 0)
        range_state = range_data.get("range_state", "")
        
        # Build period summary
        periods = range_data.get("periods", {})
        period_lines = []
        for period, p in periods.items():
            structure = p.get("structure", "-")
            pos = p.get("position_in_range", 0) * 100
            period_lines.append(f"  {period}D: Range {p.get('range_pct', 0):.1f}% | Position {pos:.0f}% | Structure: {structure}")
        period_summary = "\n".join(period_lines)
        
        # Resistance levels
        resistance = range_data.get("resistance_levels", [])
        resistance_lines = [f"  ${r.get('price', 0):.2f} ({r.get('description', '')})" for r in resistance[:5]]
        
        # Support levels  
        support = range_data.get("support_levels", [])
        support_lines = [f"  ${s.get('price', 0):.2f} ({s.get('description', '')})" for s in support[:5]]
        
        breakout = range_data.get("breakout_watch")
        breakdown = range_data.get("breakdown_watch")
        breakout_str = f"${breakout:.2f}" if breakout else "N/A"
        breakdown_str = f"${breakdown:.2f}" if breakdown else "N/A"
        
        # Build weekly macro structure summary
        weekly = range_data.get("weekly_structure")
        weekly_section = ""
        if weekly:
            weekly_trend = weekly.get("trend", "NEUTRAL")
            weekly_state = weekly.get("range_state", "N/A")
            hh = weekly.get("hh_count", 0)
            hl = weekly.get("hl_count", 0)
            lh = weekly.get("lh_count", 0)
            ll = weekly.get("ll_count", 0)
            close_pos = weekly.get("weekly_close_position", 0) * 100
            close_signal = weekly.get("weekly_close_signal", "")
            last_week = weekly.get("last_week_structure", "")
            weekly_section = f"""
WEEKLY MACRO STRUCTURE (from actual weekly candles):
  Weekly Trend: {weekly_trend}
  Weekly Range State: {weekly_state}
  HH/HL/LH/LL Counts: {hh} Higher Highs, {hl} Higher Lows, {lh} Lower Highs, {ll} Lower Lows
  Weekly Close Position: {close_pos:.0f}% (where week closed in its range)
  Weekly Close Signal: {close_signal}
  Last Week Structure: {last_week}
"""
        
        # Build Volume Profile section
        vp_section = ""
        if extra_context.get('vah'):
            vah = extra_context.get('vah', 0)
            poc = extra_context.get('poc', 0)
            val = extra_context.get('val', 0)
            vwap = extra_context.get('vwap', 0)
            rsi = extra_context.get('rsi', 50)
            rvol = extra_context.get('rvol', 1.0)
            volume_trend = extra_context.get('volume_trend', 'neutral')
            vp_position = extra_context.get('vp_position', 'N/A')
            
            # Include signal and scores if available
            signal = extra_context.get('signal', 'N/A')
            bull_score = extra_context.get('bull_score', 0)
            bear_score = extra_context.get('bear_score', 0)
            confidence = extra_context.get('confidence', 0)
            position = extra_context.get('position', 'N/A')
            vwap_zone = extra_context.get('vwap_zone', 'N/A')
            
            vp_section = f"""
VOLUME PROFILE & SIGNAL (1HR timeframe):
  VAH (Value Area High): ${vah:.2f}
  POC (Point of Control): ${poc:.2f}
  VAL (Value Area Low): ${val:.2f}
  VWAP: ${vwap:.2f}
  Current Price Position: {vp_position}
  Position Class: {position}
  VWAP Zone: {vwap_zone}
  
  **CURRENT SIGNAL: {signal}**
  Bull Score: {bull_score:.0f}
  Bear Score: {bear_score:.0f}
  Confidence: {confidence:.0f}%
  
  RSI: {rsi:.1f}
  Relative Volume: {rvol:.2f}x average
  Volume Trend: {volume_trend}
"""
        
        # Build MTF section
        mtf_section = ""
        if extra_context.get('mtf_dominant'):
            mtf_dominant = extra_context.get('mtf_dominant', 'WAIT')
            mtf_confluence = extra_context.get('mtf_confluence', 0)
            high_prob = extra_context.get('high_prob', 0)
            low_prob = extra_context.get('low_prob', 0)
            weighted_bull = extra_context.get('weighted_bull', 0)
            weighted_bear = extra_context.get('weighted_bear', 0)
            tf_signals = extra_context.get('tf_signals', {})
            
            tf_signal_lines = [f"  {tf}: {sig}" for tf, sig in tf_signals.items()]
            
            mtf_section = f"""
MULTI-TIMEFRAME ANALYSIS:
  Dominant Signal: {mtf_dominant}
  MTF Confluence: {mtf_confluence:.0f}%
  High Scenario Probability: {high_prob:.0f}%
  Low Scenario Probability: {low_prob:.0f}%
  Weighted Bull Score: {weighted_bull:.1f}
  Weighted Bear Score: {weighted_bear:.1f}
  Individual Timeframe Signals:
{chr(10).join(tf_signal_lines) if tf_signal_lines else '  N/A'}
"""
        
        # Build Fibonacci section
        fib_section = ""
        if extra_context.get('fib_236'):
            fib_swing_high = extra_context.get('fib_swing_high', 0)
            fib_swing_low = extra_context.get('fib_swing_low', 0)
            fib_236 = extra_context.get('fib_236', 0)
            fib_382 = extra_context.get('fib_382', 0)
            fib_500 = extra_context.get('fib_500', 0)
            fib_618 = extra_context.get('fib_618', 0)
            fib_786 = extra_context.get('fib_786', 0)
            fib_position = extra_context.get('fib_position', 'N/A')
            fib_confluences = extra_context.get('fib_confluences', [])
            has_confluence = extra_context.get('has_fib_confluence', False)
            
            confluence_lines = chr(10).join([f"  {c}" for c in fib_confluences]) if fib_confluences else "  None found"
            
            fib_section = f"""
FIBONACCI RETRACEMENT (15D Swing: ${fib_swing_low:.2f} to ${fib_swing_high:.2f}):
  23.6%: ${fib_236:.2f}
  38.2%: ${fib_382:.2f}
  50.0%: ${fib_500:.2f}
  61.8%: ${fib_618:.2f}
  78.6%: ${fib_786:.2f}
  Current Price Position: {fib_position}
  
  **VP + FIB CONFLUENCE:**
{confluence_lines}
  Has Strong Confluence: {'YES - HIGH CONVICTION LEVELS' if has_confluence else 'No'}
"""
        
        prompt = f"""You are a technical analyst providing OBSERVATIONAL context about price structure. 
DO NOT provide trade advice, entry/exit points, or recommendations. 
Only describe what you observe in the data and what it typically indicates.

Symbol: {symbol}
Current Price: ${current_price:.2f}

═══════════════════════════════════════
TREND & RANGE ANALYSIS
═══════════════════════════════════════
TREND STRUCTURE: {trend} ({trend_bias})
Trend Strength: {trend_strength:.1f}%
Range State: {range_state}

PERIOD ANALYSIS (Daily):
{period_summary}
{weekly_section}{vp_section}{mtf_section}{fib_section}
═══════════════════════════════════════
KEY LEVELS
═══════════════════════════════════════
RESISTANCE LEVELS:
{chr(10).join(resistance_lines) if resistance_lines else "  None identified"}

SUPPORT LEVELS:
{chr(10).join(support_lines) if support_lines else "  None identified"}

WATCH LEVELS:
  Breakout Watch: {breakout_str}
  Breakdown Watch: {breakdown_str}

═══════════════════════════════════════

Provide comprehensive technical observations (7-10 bullet points) covering:

**Signal & Potential Shift:**
• The current 1HR signal and bull/bear scores - what do they indicate?
• **CRITICAL: Look for POTENTIAL SHIFT patterns** - if longer-term trend is bearish but short-term structure is neutral/bullish AND price is above POC, this is a potential reversal setup
• Is there compression + neutral short-term structure + price above POC? This could signal accumulation/bottoming

**Structure & Trend:**
• What the multi-period daily structure indicates about trend health
• Note if short-term periods (3D, 6D) show different structure than longer periods (15D, 30D) - this divergence matters!
• Weekly macro context - what the HH/HL/LH/LL structure reveals about the bigger picture
• Weekly close signal and what it suggests about momentum

**Volume Profile & Price Position:**
• Where price is relative to VAH/POC/VAL and what that typically indicates
• If price is above POC during a downtrend, this is often a potential shift signal
• VWAP relationship and intraday bias implications
• RSI reading and what it suggests (overbought/oversold/neutral)
• Volume characteristics (RVOL, trend) and their significance

**Multi-Timeframe:**
• MTF signal confluence and alignment (or divergence) between timeframes
• What the probability skew (high vs low scenario) suggests

**Support/Resistance:**
• Key S/R zones and their confluence
• Breakout/breakdown levels to watch
• Any notable divergences between timeframes

**Fibonacci Confluence (IMPORTANT):**
• If VP levels (VAH/POC/VAL) align with Fib retracement levels within 1-2%, this is HIGH CONVICTION confluence
• Two completely different mathematical approaches (volume distribution vs golden ratio) finding the same price = institutional order clustering
• When strong confluence exists, INCREASE your conviction in those levels as support/resistance
• Note the current price position relative to Fib levels

Keep it factual and observational - no trade recommendations."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a technical analyst who provides objective, comprehensive observations about price structure, volume profile, support/resistance, and multi-timeframe analysis. IMPORTANTLY: Look for potential shift/reversal patterns when short-term signals diverge from longer-term trends (e.g., bullish signal + price above POC during a longer-term downtrend = potential bottom). CRITICAL: When Fibonacci retracement levels align with Volume Profile levels (VAH/POC/VAL), this is HIGH CONVICTION confluence - two independent mathematical methods finding the same price indicates institutional order clustering. Give extra weight to these confluent levels. Include all relevant context. Never provide trade advice or recommendations - only factual technical observations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"⚠️ Range AI error: {e}")
        return f"AI analysis error: {str(e)}"


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
    if val is None:
        return None
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    # Handle generic numpy types
    if hasattr(val, 'item'):
        return val.item()
    return val

def _format_response(result: RangeWatcherResult, weekly_structure: dict = None) -> dict:
    """Format RangeWatcherResult for API response"""
    
    periods_dict = {}
    for period, analysis in result.periods.items():
        periods_dict[period] = {
            "high": _to_native(analysis.high),
            "low": _to_native(analysis.low),
            "range_pct": _to_native(analysis.range_pct),
            "period_change_pct": _to_native(analysis.period_change_pct),
            "position_in_range": _to_native(analysis.position_in_range),
            "higher_highs": _to_native(analysis.higher_highs),
            "higher_lows": _to_native(analysis.higher_lows),
            "lower_highs": _to_native(analysis.lower_highs),
            "lower_lows": _to_native(analysis.lower_lows),
            "structure": _get_structure_string(analysis)
        }
    
    response = {
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
    
    # Add weekly structure if available
    if weekly_structure:
        response["weekly_structure"] = weekly_structure
    
    return response


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
